-- Copyright 2023 Stanford University, NVIDIA Corporation
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

task assert_disjoint(r : region(int), s : region(int))
where reads(r, s), r * s do
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

  var r0 = part0[0]
  var r1 = part0[1]

  -- Check static constraints
  assert_disjoint(r0, r1)

  var r000 = prod[0][0][0]
  var r001 = prod[0][0][1]
  var r010 = prod[0][1][0]
  var r011 = prod[0][1][1]
  var r100 = prod[1][0][0]
  var r101 = prod[1][0][1]
  var r110 = prod[1][1][0]
  var r111 = prod[1][1][1]

  -- Check static constraints
  assert_disjoint(r000, r001)
  assert_disjoint(r000, r010)
  assert_disjoint(r000, r011)
  assert_disjoint(r000, r100)
  assert_disjoint(r000, r101)
  assert_disjoint(r000, r110)
  assert_disjoint(r000, r111)

  -- Check that regions were actually computed correctly
  for x in r do @x = 0 end

  for x in r000 do @x += 1 end
  for x in r001 do @x += 1 end
  for x in r010 do @x += 1 end
  for x in r011 do @x += 1 end
  for x in r100 do @x += 1 end
  for x in r101 do @x += 1 end
  for x in r110 do @x += 1 end
  for x in r111 do @x += 1 end

  var s = 0
  for x in r do
    regentlib.assert(@x == 1, "test failed")
    s += @x
  end
  regentlib.assert(s == 8, "test failed")
end
regentlib.start(main)
