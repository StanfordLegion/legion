-- Copyright 2021 Stanford University
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

-- fails-with:
-- optimize_index_launch_num10.rg:78: loop optimization failed: argument 2 interferes with argument 1
--     g2(p0_disjoint[i], p1_disjoint[i])
--      ^

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

task with_partitions(r0 : region(int), p0_disjoint : partition(disjoint, r0),
                     r1 : region(int), p1_disjoint : partition(disjoint, r1),
                     n : int)
where reads(r0, r1), writes(r0, r1) do

  -- not optimized: loop-variant argument is (statically) interfering
  __demand(__index_launch)
  for i = 0, n do
    g2(p0_disjoint[i], p1_disjoint[i])
  end
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var rc = c.legion_coloring_create()
  for i = 0, n do
    c.legion_coloring_ensure_color(rc, i)
  end
  var p_disjoint = partition(disjoint, r, rc)
  var p_aliased = partition(aliased, r, rc)
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(disjoint, r0, rc)
  var p1_disjoint = partition(disjoint, r1, rc)
  c.legion_coloring_destroy(rc)

  with_partitions(r0, p0_disjoint, r1, p1_disjoint, n)
end
regentlib.start(main)

