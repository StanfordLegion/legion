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

local c = regentlib.c

task main()
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(x))
  c.legion_coloring_ensure_color(rc, 1)

  var p = partition(disjoint, r, rc)
  var r0 = p[0]
  var r1 = p[1]

  var p0 = partition(disjoint, r0, rc)
  var r00 = p0[0]
  var r01 = p0[1]
  c.legion_coloring_destroy(rc)

  var rc1 = c.legion_coloring_create()
  c.legion_coloring_ensure_color(rc1, 0)
  c.legion_coloring_ensure_color(rc1, 1)
  var p1 = partition(disjoint, r1, rc1)
  var r10 = p1[0]
  var r11 = p1[1]
  c.legion_coloring_destroy(rc1)

  var x00 = dynamic_cast(ptr(int, r00), x)
  var x0 = static_cast(ptr(int, r0), x00)

  var x00_01 = static_cast(ptr(int, r00, r01), x00)
  var x01_00 = static_cast(ptr(int, r01, r00), x00_01)

  var x00_01_10_11 = static_cast(ptr(int, r00, r01, r10, r11), x00)
  var x01_10_11_00 = static_cast(ptr(int, r01, r10, r11, r00), x00_01_10_11)

  var x_ = static_cast(ptr(int, r), x01_10_11_00)

  @x_ = 123

  regentlib.assert(@x0 == 123, "test failed")
  regentlib.assert(@x00 == 123, "test failed")
  regentlib.assert(@x00_01 == 123, "test failed")
  regentlib.assert(@x01_00 == 123, "test failed")
  regentlib.assert(@x00_01_10_11 == 123, "test failed")
  regentlib.assert(@x01_10_11_00 == 123, "test failed")
end
regentlib.start(main)
