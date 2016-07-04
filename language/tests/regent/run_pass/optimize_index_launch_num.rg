-- Copyright 2016 Stanford University
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

task f(r : region(int)) : int
where reads(r) do
  return 5
end

task f2(r : region(int), s : region(int)) : int
where reads(r, s) do
  return 5
end

task g(r : region(int)) : int
where reads(r), writes(r) do
  return 5
end

task g2(r : region(int), s : region(int)) : int
where reads(r, s), writes(r) do
  return 5
end

task h(r : region(int)) : int
where reduces +(r) do
  return 5
end

task h2(r : region(int), s : region(int)) : int
where reduces +(r, s) do
  return 5
end

task h2b(r : region(int), s : region(int)) : int
where reduces +(r), reduces *(s) do
  return 5
end

-- FIXME: Dataflow analysis currently can't handle aliased regions
-- with no common ancestor.

-- task with_partitions(r0 : region(int), p0_disjoint : partition(disjoint, r0),
--                      r1 : region(int), p1_disjoint : partition(disjoint, r1),
--                      n : int)
-- where reads(r0, r1), writes(r0, r1) do

--   -- not optimized: loop-variant argument is (statically) interfering
--   for i = 0, n do
--     g2(p0_disjoint[i], p1_disjoint[i])
--   end

--   -- not optimized: loop-variant argument is (statically) interfering
--   for i = 0, n do
--     h2b(p0_disjoint[i], p1_disjoint[i])
--   end

--   -- optimized: loop-variant argument is non-interfering
--   __demand(__parallel)
--   for i = 0, n do
--     h2(p0_disjoint[i], p1_disjoint[i])
--   end

--   -- optimized: loop-variant argument is non-interfering
--   __demand(__parallel)
--   for i = 0, n do
--     g(p0_disjoint[i])
--   end
-- end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var x0 = new(ptr(int, r))
  var x1 = new(ptr(int, r))
  var x2 = new(ptr(int, r))
  var x3 = new(ptr(int, r))
  var x4 = new(ptr(int, r))
  fill(r, 1)

  var p_disjoint = partition(equal, r, ispace(int1d, n))
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(equal, r0, ispace(int1d, n))
  var p1_disjoint = partition(equal, r1, ispace(int1d, n))

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(x0))
  c.legion_coloring_add_point(rc, 0, __raw(x1))
  c.legion_coloring_add_point(rc, 1, __raw(x0))
  c.legion_coloring_add_point(rc, 1, __raw(x1))
  c.legion_coloring_add_point(rc, 1, __raw(x2))
  c.legion_coloring_add_point(rc, 2, __raw(x1))
  c.legion_coloring_add_point(rc, 2, __raw(x2))
  c.legion_coloring_add_point(rc, 2, __raw(x3))
  c.legion_coloring_add_point(rc, 3, __raw(x2))
  c.legion_coloring_add_point(rc, 3, __raw(x3))
  c.legion_coloring_add_point(rc, 3, __raw(x4))
  c.legion_coloring_add_point(rc, 4, __raw(x3))
  c.legion_coloring_add_point(rc, 4, __raw(x4))
  var p_aliased = partition(aliased, r, rc)
  c.legion_coloring_destroy(rc)

  -- not optimized: can't analyze stride
  for i = 0, n, 2 do
    f(p_disjoint[i])
  end

  -- not optimized: can't analyze stride
  do
    var s = 1
    for i = 0, n, s do
      f(p_disjoint[i])
    end
  end

  -- not optimized: body is not a single statement
  for i = 0, n do
    f(p_disjoint[i])
    f(p_disjoint[i])
  end

  -- not optimized: body is not a bare function call
  for i = 0, n do
    var x = f(p_disjoint[i])
  end

  -- not optimized: function is not a task
  for i = 0, n do
    e(i)
  end

  -- not optimized: argument 1 is not side-effect free
  for i = 0, n do
    f(p_disjoint[e_bad(__runtime())])
  end

  -- not optimized: can't analyze loop-variant argument
  for i = 0, n do
    f(p_disjoint[(i + 1) % n])
  end

  -- not optimized: loop-invariant argument is interfering
  do
    var j = 3
    for i = 0, n do
      g(p_disjoint[(j + 1) % n])
    end
  end

  -- not optimized: loop-variant argument is interfering
  for i = 0, n do
    g(p_aliased[i])
  end

  -- not optimized: reductino is interfering
  for i = 0, n do
    @x0 += f(p_disjoint[i])
  end

  -- optimized: loop-invariant argument is read-only
  do
    var j = 3
    __demand(__parallel)
    for i = 0, n do
      f(p_disjoint[(j + 1) % n])
    end
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__parallel)
  for i = 0, n do
    f(p_disjoint[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__parallel)
  for i = 0, n do
    f(p_aliased[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__parallel)
  for i = 0, n do
    h(p_aliased[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__parallel)
  for i = 0, n do
    g2(p0_disjoint[i], p1_disjoint[i])
  end

  -- optimized: loop-variant argument is non-interfering
  __demand(__parallel)
  for i = 0, n do
    g(p_disjoint[i])
  end

  -- optimized: reduction is non-interfering
  var y = 0
  __demand(__parallel)
  for i = 0, n do
    y += f(p_disjoint[i])
  end
  regentlib.assert(y == 25, "test failed")

  -- with_partitions(r0, p0_disjoint, r1, p1_disjoint, n)
end
regentlib.start(main)
