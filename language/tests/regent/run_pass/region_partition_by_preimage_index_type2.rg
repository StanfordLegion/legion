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

task f()
  var r = region(ispace(int3d, { 5, 1, 1 }), int)
  var s = region(ispace(ptr, 4), int3d)
  var y0 = dynamic_cast(ptr(int3d, s), 0)
  var y1 = dynamic_cast(ptr(int3d, s), 1)
  var y2 = dynamic_cast(ptr(int3d, s), 2)
  var y3 = dynamic_cast(ptr(int3d, s), 3)
  @y0 = [int3d] { 1, 0, 0 }
  @y1 = [int3d] { 0, 0, 0 }
  @y2 = [int3d] { 1, 0, 0 }
  @y3 = [int3d] { 2, 0, 0 }

  var rc = c.legion_domain_coloring_create()
  c.legion_domain_coloring_color_domain(rc, 0, rect3d { { 0, 0, 0 }, { 0, 0, 0 } })
  c.legion_domain_coloring_color_domain(rc, 1, rect3d { { 1, 0, 0 }, { 1, 0, 0 } })
  c.legion_domain_coloring_color_domain(rc, 2, rect3d { { 2, 0, 0 }, { 2, 0, 0 } })
  var p = partition(disjoint, r, rc)
  c.legion_domain_coloring_destroy(rc)

  var q = preimage(s, p, s)

  for x in r do
    @x = 1
  end

  for i = 0, 3 do
    var si = q[i]
    -- FIXME: This should be vectorized
    for y in si do
      r[@y] *= __raw(y).value + 2
    end
  end

  var t = 0
  for i = 0, 3 do
    t += r[{ i, 0, 0 }]
  end

  return t
end

task main()
  regentlib.assert(f() == 16, "test failed")
end
regentlib.start(main)
