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

  var x0 = dynamic_cast(ptr(int, r0), x)
  regentlib.assert(not isnull(x0), "test failed")

  @x0 = 123

  var x1 = dynamic_cast(ptr(int, r1), x)
  regentlib.assert(isnull(x1), "test failed")

  var x01 = dynamic_cast(ptr(int, r0, r1), x)
  regentlib.assert(not isnull(x01), "test failed")

  regentlib.assert(@x01 == 123, "test failed")

  c.legion_coloring_destroy(rc)
end
regentlib.start(main)
