-- Copyright 2016 Stanford University, NVIDIA Corporation
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

terra create_colorings()
  return [&c.legion_coloring_t](c.malloc(sizeof(c.legion_coloring_t) * 2))
end

task main()
  var r = region(ispace(ptr, 5), int)
  var x0 = new(ptr(int, r))
  var x1 = new(ptr(int, r))
  var x2 = new(ptr(int, r))
  var x3 = new(ptr(int, r))

  var colors0 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors0, 0, __raw(x0))
  c.legion_coloring_add_point(colors0, 0, __raw(x1))
  c.legion_coloring_add_point(colors0, 1, __raw(x2))
  c.legion_coloring_add_point(colors0, 1, __raw(x3))
  var part0 = partition(disjoint, r, colors0)
  c.legion_coloring_destroy(colors0)

  var colors1 = create_colorings()
  colors1[0] = c.legion_coloring_create()
  colors1[1] = c.legion_coloring_create()
  c.legion_coloring_add_point(colors1[0], 0, __raw(x0))
  c.legion_coloring_add_point(colors1[0], 1, __raw(x1))
  c.legion_coloring_add_point(colors1[1], 0, __raw(x2))
  c.legion_coloring_add_point(colors1[1], 1, __raw(x3))

  var prod = cross_product_array(part0, disjoint, colors1)
  c.legion_coloring_destroy(colors1[0])
  c.legion_coloring_destroy(colors1[1])
  c.free(colors1)

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
