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

task main()
  var n = 5
  var cs = ispace(ptr, n)
  var r = region(ispace(ptr, n), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)
  var x4 = dynamic_cast(ptr(int, r), 4)
  fill(r, 1)

  var p_disjoint = partition(equal, r, cs)
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(equal, r0, cs)
  var p1_disjoint = partition(equal, r1, cs)

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

  -- not optimized: reductino is interfering
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

  -- with_partitions(r0, p0_disjoint, r1, p1_disjoint, n)
end
regentlib.start(main)
