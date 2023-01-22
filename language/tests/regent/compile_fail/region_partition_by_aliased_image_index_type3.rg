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

-- fails-with:
-- region_partition_by_aliased_image_index_type3.rg:74: invalid call missing constraint $q0 * $q1
--   assert_disjoint(q0, q1)
--                  ^


import "regent"

local c = regentlib.c

task assert_disjoint(x : region(ispace(int3d), int), y : region(ispace(int3d), int))
where x * y do end

task f()
  var r = region(ispace(int3d, {4, 1, 1}), int)
  var x0 = dynamic_cast(int3d(int, r), { 0, 0, 0 })
  var x1 = dynamic_cast(int3d(int, r), { 1, 0, 0 })
  var x2 = dynamic_cast(int3d(int, r), { 2, 0, 0 })
  var x3 = dynamic_cast(int3d(int, r), { 3, 0, 0 })
  var s = region(ispace(ptr, 4), int3d)
  var y0 = dynamic_cast(ptr(int3d, s), 0)
  var y1 = dynamic_cast(ptr(int3d, s), 1)
  var y2 = dynamic_cast(ptr(int3d, s), 2)
  var y3 = dynamic_cast(ptr(int3d, s), 3)

  @y0 = x0
  @y1 = x1
  @y2 = x1
  @y3 = x2

  var sc = c.legion_coloring_create()
  c.legion_coloring_add_point(sc, 0, __raw(y0))
  c.legion_coloring_add_point(sc, 1, __raw(y1))
  c.legion_coloring_add_point(sc, 2, __raw(y2))
  var p = partition(disjoint, s, sc)
  c.legion_coloring_destroy(sc)

  var q = image(aliased, r, p, s)

  for x in r do
    @x = 1
  end

  for i = 0, 3 do
    var ri = q[i]
    for x in ri do
      @x *= i + 2
    end
  end

  var t = 0
  for x in r do
    t += @x
  end

  var q0 = q[0]
  var q1 = q[1]
  var q2 = q[2]

  assert_disjoint(q0, q1)
  assert_disjoint(q0, q2)
  assert_disjoint(q1, q2)

  return t
end

task main()
  regentlib.assert(f() == 16, "test failed")
end
regentlib.start(main)
