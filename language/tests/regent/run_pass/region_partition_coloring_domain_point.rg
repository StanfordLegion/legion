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
  var r = region(ispace(int1d, 5), int)
  var colors = ispace(int1d, 1)

  var rc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(rc, int1d(0), rect1d{0, 3})
  var p = partition(disjoint, r, rc, colors)
  c.legion_domain_point_coloring_destroy(rc)
  var r0 = p[0]

  fill(r, 1)
  fill(r0, 10)

  var t = 0
  for i in r do
    t += r[i]
  end
  return t
end

task main()
  regentlib.assert(f() == 41, "test failed")
end
regentlib.start(main)
