-- Copyright 2023 Stanford University
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
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * accesses to cross-product inner partitions

local c = regentlib.c

task mul(r : region(int), p : partition(aliased, r), n : int, y : int)
where reads(r), writes(r) do
  for i = 0, n do
    var s = p[i]
    for x in s do
      @x *= y
    end
  end
end

task main()
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)

  var cp = c.legion_coloring_create()
  c.legion_coloring_add_point(cp, 0, __raw(x0))
  c.legion_coloring_add_point(cp, 1, __raw(x1))
  c.legion_coloring_add_point(cp, 2, __raw(x2))
  c.legion_coloring_add_point(cp, 3, __raw(x3))
  var p = partition(disjoint, r, cp)
  c.legion_coloring_destroy(cp)

  var cq = c.legion_coloring_create()
  c.legion_coloring_add_point(cq, 0, __raw(x0))
  c.legion_coloring_add_point(cq, 0, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x0))
  c.legion_coloring_add_point(cq, 1, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x1))
  c.legion_coloring_add_point(cq, 2, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x3))
  c.legion_coloring_add_point(cq, 3, __raw(x2))
  c.legion_coloring_add_point(cq, 3, __raw(x3))
  var q = partition(aliased, r, cq)
  c.legion_coloring_destroy(cq)

  var m = cross_product(p, q)

  for x in r do
    @x = 1
  end

  for x in r do
    c.printf("x %d %d\n", int(x), @x)
  end

  var tinit, tfinal = 0, 1

  __demand(__spmd)
  for t = tinit, tfinal do
    for i = 0, 4 do
      mul(p[i], m[i], 4, 2)
    end
  end

  for x in r do
    c.printf("x %d %d\n", int(x), @x)
  end

  regentlib.assert(@x0 == 4, "test failed")
  regentlib.assert(@x1 == 8, "test failed")
  regentlib.assert(@x2 == 8, "test failed")
  regentlib.assert(@x3 == 4, "test failed")
end
regentlib.start(main)
