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

task f() : int
  var r = region(ispace(int2d, {3, 2}), int)
  var colors = ispace(int3d, {2, 2, 2})

  var rc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(rc, int3d{0, 0, 0}, rect2d{{0, 0}, {1, 1}})
  var p = partition(disjoint, r, rc, colors)
  c.legion_domain_point_coloring_destroy(rc)
  var r0 = p[{0, 0, 0}]

  fill(r, 1)
  fill(r0, 10)

  var t = 0
  for i in r do
    t += r[i]
  end
  return t
end

task main()
  regentlib.assert(f() == 42, "test failed")
end
regentlib.start(main)
