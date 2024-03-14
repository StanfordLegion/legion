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
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)
  var s = region(ispace(int2d, { 4, 1 }), ptr)
  var y0 = dynamic_cast(int2d(ptr, s), { 0, 0 })
  var y1 = dynamic_cast(int2d(ptr, s), { 1, 0 })
  var y2 = dynamic_cast(int2d(ptr, s), { 2, 0 })
  var y3 = dynamic_cast(int2d(ptr, s), { 3, 0 })

  @y0 = x1
  @y1 = x0
  @y2 = x1
  @y3 = x2

  var sc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(sc, [int1d](0), rect2d { y0, y0 })
  c.legion_domain_point_coloring_color_domain(sc, [int1d](1), rect2d { y1, y1 })
  c.legion_domain_point_coloring_color_domain(sc, [int1d](2), rect2d { y2, y2 })
  var p = partition(disjoint, s, sc, ispace(int1d, 3))
  c.legion_domain_point_coloring_destroy(sc)

  var q = image(r, p, s)

  for x in r do
    @x = 1
  end

  for i = 0, 3 do
    var ri = q[i]
    for x in ri do
      @x *= i + 2
    end
  end

  var t = 0
  for x in r do
    t += @x
  end

  return t
end

task main()
  regentlib.assert(f() == 13, "test failed")
end
regentlib.start(main)
