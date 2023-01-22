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
  var is = ispace(int2d, {4, 8})
  var r = region(is, int)
  var p = partition(equal, r, ispace(int2d, {2, 2}))
  var r00 = p[{0, 0}]
  var r11 = p[{1, 1}]
  var is_bounds = is.bounds
  var r_bounds = r.bounds
  var r00_bounds = r00.bounds
  var r11_bounds = r11.bounds

  var r_size = r_bounds:size()

  regentlib.assert(is_bounds.lo.x == 0, "test failed")
  regentlib.assert(is_bounds.lo.y == 0, "test failed")
  regentlib.assert(is_bounds.hi.x == 3, "test failed")
  regentlib.assert(is_bounds.hi.y == 7, "test failed")
  regentlib.assert(r_bounds.lo.x == 0, "test failed")
  regentlib.assert(r_bounds.lo.y == 0, "test failed")
  regentlib.assert(r_bounds.hi.x == 3, "test failed")
  regentlib.assert(r_bounds.hi.y == 7, "test failed")
  regentlib.assert(r00_bounds.lo.x == 0, "test failed")
  regentlib.assert(r00_bounds.lo.y == 0, "test failed")
  regentlib.assert(r00_bounds.hi.x == 1, "test failed")
  regentlib.assert(r00_bounds.hi.y == 3, "test failed")
  regentlib.assert(r11_bounds.lo.x == 2, "test failed")
  regentlib.assert(r11_bounds.lo.y == 4, "test failed")
  regentlib.assert(r11_bounds.hi.x == 3, "test failed")
  regentlib.assert(r11_bounds.hi.y == 7, "test failed")
  regentlib.assert(r_size.x == 4, "test failed")
  regentlib.assert(r_size.y == 8, "test failed")

  var coloring = c.legion_domain_coloring_create()
  c.legion_domain_coloring_color_domain(
    coloring, 0, rect2d { {0, 0}, {3, 3} })
  c.legion_domain_coloring_color_domain(
    coloring, 1, rect2d { {0, 4}, {3, 7} })
  var p2 = partition(disjoint, r, coloring)
  var lp = list_duplicate_partition(p2, list_range(0, 2))
  c.legion_domain_coloring_destroy(coloring)
  regentlib.assert(lp[0].bounds:size() == lp[1].bounds:size(),
                   "test failed")
end
regentlib.start(main)
