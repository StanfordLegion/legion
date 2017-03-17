-- Copyright 2017 Stanford University
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

terra to_domain(lo : c.coord_t, hi : c.coord_t)
  return c.legion_domain_from_rect_1d(
    c.legion_rect_1d_t {
      lo = c.legion_point_1d_t { x = arrayof(c.coord_t, lo) },
      hi = c.legion_point_1d_t { x = arrayof(c.coord_t, hi) },
    })
end

terra to_domain_point(x : c.coord_t)
  return c.legion_domain_point_from_point_1d(
    c.legion_point_1d_t { x = arrayof(c.coord_t, x) })
end

task f() : int
  var r = region(ispace(int1d, 5), int)
  var colors = ispace(int1d, 1)

  var rc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(rc, to_domain_point(0), to_domain(0, 3))
  var p = partition(disjoint, r, rc, colors)
  c.legion_domain_point_coloring_destroy(rc)
  var r0 = p[0]

  -- FIXME: Fills are buggy in this branch so use loops to set these values
  -- fill(r, 1)
  -- fill(r0, 10)
  for i in r do r[i] = 1 end
  for i in r0 do r[i] = 10 end

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
