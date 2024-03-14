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

terra to_domain_point(x : c.coord_t)
  return c.legion_domain_point_from_point_1d(
    c.legion_point_1d_t { x = arrayof(c.coord_t, x) })
end

task f() : int
  var r = region(ispace(ptr, 1), int)
  var x = dynamic_cast(ptr(int, r), 0)

  var colors = ispace(int1d, 1)

  var rc = c.legion_point_coloring_create()
  c.legion_point_coloring_add_point(rc, to_domain_point(0), __raw(x))
  var p = partition(disjoint, r, rc, colors)
  c.legion_point_coloring_destroy(rc)
  var r0 = p[0]

  var x0 = dynamic_cast(ptr(int, r0), x)
  regentlib.assert(not isnull(x0), "test failed")
  @x = 5
  return @x0
end

task main()
  regentlib.assert(f() == 5, "test failed")
end
regentlib.start(main)
