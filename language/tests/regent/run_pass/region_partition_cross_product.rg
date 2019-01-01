-- Copyright 2019 Stanford University, NVIDIA Corporation
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
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)

  var colors0 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors0, 0, __raw(x0))
  c.legion_coloring_add_point(colors0, 0, __raw(x1))
  c.legion_coloring_add_point(colors0, 1, __raw(x2))
  c.legion_coloring_add_point(colors0, 1, __raw(x3))
  var part0 = partition(disjoint, r, colors0)
  c.legion_coloring_destroy(colors0)

  var colors1 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors1, 0, __raw(x0))
  c.legion_coloring_add_point(colors1, 1, __raw(x1))
  c.legion_coloring_add_point(colors1, 0, __raw(x2))
  c.legion_coloring_add_point(colors1, 1, __raw(x3))
  var part1 = partition(disjoint, r, colors1)
  c.legion_coloring_destroy(colors1)

  var prod = cross_product(part0, part1)

  var r0 = part0[0]
  var r1 = part0[1]

  -- Check static constraints
  assert_disjoint(r0, r1)

  var r00 = prod[0][0]
  var r01 = prod[0][1]
  var r10 = prod[1][0]
  var r11 = prod[1][1]

  -- Check static constraints
  assert_disjoint(r00, r01)
  assert_disjoint(r00, r10)
  assert_disjoint(r00, r11)
  assert_disjoint(r01, r10)
  assert_disjoint(r01, r11)
  assert_disjoint(r10, r11)

  -- Check that regions were actually computed correctly
  for x in r do @x = 0 end

  for x in r00 do @x += 1 end
  for x in r01 do @x += 1 end
  for x in r10 do @x += 1 end
  for x in r11 do @x += 1 end

  var s = 0
  for x in r do
    regentlib.assert(@x == 1, "test failed")
    s += @x
  end
  regentlib.assert(s == 4, "test failed")
end
regentlib.start(main)
