-- Copyright 2024 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- runs-with:
-- [["-ffuture", "1"], ["-ffuture", "0"], ["-findex-launch", "0"], ["-fflow", "0"]]

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

terra e(x : int) : int
  return 3
end

terra e_bad(x : c.legion_runtime_t) : int
  return 3
end

task f(r : region(ispace(int1d), int)) : int
where reads(r) do
  return 5
end

task f2(r : region(ispace(int1d), int), s : region(ispace(int1d), int)) : int
where reads(r, s) do
  return 5
end

task g(r : region(ispace(int1d), int)) : int
where reads(r), writes(r) do
  return 5
end

task g2(r : region(ispace(int1d), int), s : region(ispace(int1d), int)) : int
where reads(r, s), writes(r) do
  return 5
end

task h(r : region(ispace(int1d), int)) : int
where reduces +(r) do
  return 5
end

task h2(r : region(ispace(int1d), int), s : region(ispace(int1d), int)) : int
where reduces +(r, s) do
  return 5
end

task h2b(r : region(ispace(int1d), int), s : region(ispace(int1d), int)) : int
where reduces +(r), reduces *(s) do
  return 5
end

task return_2468() return 2468 end

task return_partition(r : region(ispace(int1d), int),
                      p : partition(disjoint, r, ispace(int1d)))
  return p
end

task check(r : region(ispace(int1d), int), v : int)
where reads(r)
do
  for e in r do
    regentlib.assert(@e == v, "test failed")
  end
end

-- FIXME: Dataflow analysis currently can't handle aliased regions
-- with no common ancestor.

if not regentlib.config["flow"] then
  task with_partitions(cs : ispace(int1d),
                       r0 : region(ispace(int1d), int),
                       p0_disjoint : partition(disjoint, r0, cs),
                       r1 : region(ispace(int1d), int),
                       p1_disjoint : partition(disjoint, r1, cs))
  where reads writes(r0, r1) do
    -- not optimized: projectable argument is (statically) interfering
    for i in cs do
      g2(p0_disjoint[i], p1_disjoint[i])
    end

    -- not optimized: projectable argument is (statically) interfering
    for i in cs do
      h2b(p0_disjoint[i], p1_disjoint[i])
    end

    -- optimized: projectable argument is non-interfering
    __demand(__index_launch)
    for i in cs do
      h2(p0_disjoint[i], p1_disjoint[i])
    end

    -- optimized: projectable argument is non-interfering
    __demand(__index_launch)
    for i in cs do
      g(p0_disjoint[i])
    end
  end
end -- not flow

task main()
  var n = 5
  var r = region(ispace(int1d, n), int)
  fill(r, 1)

  var cs = ispace(int1d, n)
  var p_disjoint = partition(equal, r, cs)
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(equal, r0, cs)
  var p1_disjoint = partition(equal, r1, cs)

  var s = region(cs, rect1d)
  s[0] = rect1d { 0, 1 }
  s[1] = rect1d { 0, 2 }
  s[2] = rect1d { 1, 3 }
  s[3] = rect1d { 2, 4 }
  s[4] = rect1d { 3, 4 }
  var ps = partition(equal, s, cs)
  var p_aliased = image(r, ps, s)

  -- not optimized: body is not a single statement
  for i in cs do
    f(p_disjoint[i])
    f(p_disjoint[i])
  end

  -- not optimized: body is not a bare function call
  for i in cs do
    var x = f(p_disjoint[i])
  end

  -- not optimized: function is not a task
  for i in cs do
    e(i)
  end

  -- not optimized: argument 1 is not side-effect free
  for i in cs do
    f(p_disjoint[e_bad(__runtime())])
  end

  -- not optimized: can't analyze loop-variant argument
  for i in cs do
    f(p_disjoint[(int(i) + 1) % n])
  end

  -- not optimized: loop-invariant argument is interfering
  do
    var j = 3
    for i in cs do
      g(p_disjoint[(j + 1) % n])
    end
  end

  -- not optimized: loop-variant argument is interfering
  for i in cs do
    g(p_aliased[i])
  end

  -- not optimized: reduction is interfering
  for i in cs do
    r[0] += f(p_disjoint[i])
  end

  -- not optimized: reduction is interfering
  var x0 = &r[0]
  for i in cs do
    @x0 += f(p_disjoint[i])
  end

  -- optimized: loop-invariant argument is read-only
  do
    var j = 3
    __demand(__index_launch)
    for i in cs do
      f(p_disjoint[(j + 1) % n])
    end
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    f(p_disjoint[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    f(p_aliased[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    h(p_aliased[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    g2(p0_disjoint[i], p1_disjoint[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    g(p_disjoint[i])
  end

  -- optimized: reduction is non-interfering
  var y = 0
  __demand(__index_launch)
  for i in cs do
    y += f(p_disjoint[i])
  end
  regentlib.assert(y == 25, "test failed")

  -- optimized: loop-variant argument is non-interfering
  __demand(__index_launch)
  for i in cs do
    fill((p_disjoint[i]), 54321)
  end
  __demand(__index_launch)
  for i in cs do
    check(p_disjoint[i], 54321)
  end

  var v = return_2468()
  var p_disjoint_copy = return_partition(r, p_disjoint)

  __demand(__index_launch)
  for i in cs do
    fill((p_disjoint_copy[i]), 12345)
  end
  __demand(__index_launch)
  for i in cs do
    check(p_disjoint[i], 12345)
  end

  __demand(__index_launch)
  for i in cs do
    fill((p0_disjoint[i]), v)
  end
  __demand(__index_launch)
  for i in cs do
    check(p0_disjoint[i], 2468)
  end
  __demand(__index_launch)
  for i in cs do
    check(p1_disjoint[i], 12345)
  end

  rescape
    if not regentlib.config["flow"] then
      remit rquote
        with_partitions(cs, r0, p0_disjoint, r1, p1_disjoint)
      end
    end
  end
end
regentlib.start(main)
