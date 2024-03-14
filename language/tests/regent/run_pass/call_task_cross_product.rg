-- Copyright 2024 Stanford University, NVIDIA Corporation
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

local c = regentlib.c

task f(r : region(int),
       p0 : partition(disjoint, r), p1 : partition(disjoint, r),
       p2 : partition(disjoint, r),
       cp : cross_product(p0, p1, p2))
where reads(r) do

  var r000 = cp[0][0][0]
  var r001 = cp[0][0][1]
  var r010 = cp[0][1][0]
  var r011 = cp[0][1][1]
  var r100 = cp[1][0][0]
  var r101 = cp[1][0][1]
  var r110 = cp[1][1][0]
  var r111 = cp[1][1][1]

  var s = 0
  for x in r000 do s += @x*1 end
  for x in r001 do s += @x*10 end
  for x in r010 do s += @x*100 end
  for x in r011 do s += @x*1000 end
  for x in r100 do s += @x*10000 end
  for x in r101 do s += @x*100000 end
  for x in r110 do s += @x*1000000 end
  for x in r111 do s += @x*10000000 end
  return s
end

task main()
  var r = region(ispace(ptr, 8), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)
  var x4 = dynamic_cast(ptr(int, r), 4)
  var x5 = dynamic_cast(ptr(int, r), 5)
  var x6 = dynamic_cast(ptr(int, r), 6)
  var x7 = dynamic_cast(ptr(int, r), 7)

  var colors0 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors0, 0, __raw(x0))
  c.legion_coloring_add_point(colors0, 0, __raw(x1))
  c.legion_coloring_add_point(colors0, 0, __raw(x2))
  c.legion_coloring_add_point(colors0, 0, __raw(x3))
  c.legion_coloring_add_point(colors0, 1, __raw(x4))
  c.legion_coloring_add_point(colors0, 1, __raw(x5))
  c.legion_coloring_add_point(colors0, 1, __raw(x6))
  c.legion_coloring_add_point(colors0, 1, __raw(x7))
  var part0 = partition(disjoint, r, colors0)
  c.legion_coloring_destroy(colors0)

  var colors1 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors1, 0, __raw(x0))
  c.legion_coloring_add_point(colors1, 0, __raw(x1))
  c.legion_coloring_add_point(colors1, 1, __raw(x2))
  c.legion_coloring_add_point(colors1, 1, __raw(x3))
  c.legion_coloring_add_point(colors1, 0, __raw(x4))
  c.legion_coloring_add_point(colors1, 0, __raw(x5))
  c.legion_coloring_add_point(colors1, 1, __raw(x6))
  c.legion_coloring_add_point(colors1, 1, __raw(x7))
  var part1 = partition(disjoint, r, colors1)
  c.legion_coloring_destroy(colors1)

  var colors2 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors2, 0, __raw(x0))
  c.legion_coloring_add_point(colors2, 1, __raw(x1))
  c.legion_coloring_add_point(colors2, 0, __raw(x2))
  c.legion_coloring_add_point(colors2, 1, __raw(x3))
  c.legion_coloring_add_point(colors2, 0, __raw(x4))
  c.legion_coloring_add_point(colors2, 1, __raw(x5))
  c.legion_coloring_add_point(colors2, 0, __raw(x6))
  c.legion_coloring_add_point(colors2, 1, __raw(x7))
  var part2 = partition(disjoint, r, colors2)
  c.legion_coloring_destroy(colors2)

  var prod = cross_product(part0, part1, part2)
  for x in r do @x = 1 end

  regentlib.assert(f(r, part0, part1, part2, prod) == 11111111, "test failed")
end
regentlib.start(main)
