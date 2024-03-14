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
  var r = region(ispace(int1d, 5), int)
  -- pointers in s will be initialized to the first point in r
  var s = region(ispace(int2d, { 5, 1 }), int1d(int, r))
  s[{ 0, 0 }] = dynamic_cast(int1d(int, r), 0)
  s[{ 1, 0 }] = dynamic_cast(int1d(int, r), 1)
  s[{ 2, 0 }] = dynamic_cast(int1d(int, r), 2)
  s[{ 3, 0 }] = dynamic_cast(int1d(int, r), 3)
  s[{ 4, 0 }] = dynamic_cast(int1d(int, r), 4)

  var rc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(rc, [int3d] { 0, 0, 0 }, [rect1d] { 0, 0 })
  c.legion_domain_point_coloring_color_domain(rc, [int3d] { 1, 0, 0 }, [rect1d] { 1, 1 })
  c.legion_domain_point_coloring_color_domain(rc, [int3d] { 2, 0, 0 }, [rect1d] { 2, 2 })
  var cs = ispace(int3d, { 3, 1, 1 })
  var p = partition(disjoint, r, rc, cs)
  c.legion_domain_point_coloring_destroy(rc)

  var q = preimage(s, p, s)

  for x in r do
    @x = 1
  end

  for color in cs do
    var ri = p[color]
    var si = q[color]
    for y in si do
      if @y <= ri.bounds then
        @@y *= color.x + 2
      end
    end
  end

  var t = 0
  for x in r do
    t += @x
  end

  return t
end

task main()
  regentlib.assert(f() == 11, "test failed")
end
regentlib.start(main)
