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

task calculate_halo_size(bounds : rect2d) : rect2d
  return rect2d { bounds.lo - {2, 2}, bounds.hi + {2, 2} }
end

task toplevel()
  var r = region(ispace(int2d, {10, 10}), int)
  var p_equal = partition(equal, r, ispace(int2d, {2, 2}))
  var coloring = c.legion_domain_coloring_create()
  c.legion_domain_coloring_color_domain(coloring, 0, rect2d { {2, 2}, {7, 7} })
  var p_interior = partition(disjoint, r, coloring)
  c.legion_domain_coloring_destroy(coloring)
  var cp = cross_product(p_interior, p_equal)
  var p_image = image(r, cp[0], calculate_halo_size)

  regentlib.assert(p_image[{0, 0}].bounds.lo.x == 0, "test failed")
  regentlib.assert(p_image[{1, 0}].bounds.lo.x == 3, "test failed")
  regentlib.assert(p_image[{0, 0}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p_image[{1, 0}].bounds.hi.x == 9, "test failed")
  regentlib.assert(p_image[{0, 1}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p_image[{1, 1}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p_image[{0, 1}].bounds.hi.y == 9, "test failed")
  regentlib.assert(p_image[{1, 1}].bounds.hi.y == 9, "test failed")
end
regentlib.start(toplevel)
