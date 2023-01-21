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

fspace assert_subregion {
  x : region(int),
  y : region(int),
} where x <= y end

fspace assert_disjoint {
  x : region(int),
  y : region(int),
} where x * y end

task main()
  var t = region(ispace(ptr, 5), int)
  var colors = c.legion_coloring_create()
  c.legion_coloring_ensure_color(colors, 0)
  c.legion_coloring_ensure_color(colors, 1)

  var tp = partition(disjoint, t, colors)
  var t0 = tp[0]
  var t1 = tp[1]

  var t0p = partition(disjoint, t0, colors)
  var t00 = t0p[0]
  var t01 = t0p[1]

  var t1p = partition(disjoint, t1, colors)
  var t10 = t1p[0]
  var t11 = t1p[1]

  assert_subregion { x = t, y = t }
  assert_subregion { x = t0, y = t }
  assert_subregion { x = t00, y = t }
  assert_subregion { x = t01, y = t }
  assert_subregion { x = t1, y = t }
  assert_subregion { x = t10, y = t }
  assert_subregion { x = t11, y = t }

  assert_disjoint { x = t0, y = t1 }
  assert_disjoint { x = t0, y = t10 }
  assert_disjoint { x = t0, y = t11 }
  assert_disjoint { x = t00, y = t1 }
  assert_disjoint { x = t00, y = t10 }
  assert_disjoint { x = t00, y = t11 }
  assert_disjoint { x = t01, y = t1 }
  assert_disjoint { x = t01, y = t10 }
  assert_disjoint { x = t01, y = t11 }

  c.legion_coloring_destroy(colors)
end
regentlib.start(main)
