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
  -- pointers in s will be initialized to point to x0
  var s = region(ispace(int2d, { 5, 1 }), ptr)
  s[{ 0, 0 }] = x0
  s[{ 1, 0 }] = x1
  s[{ 2, 0 }] = x2
  s[{ 3, 0 }] = x3
  s[{ 4, 0 }] = x3

  var rc = c.legion_point_coloring_create()
  c.legion_point_coloring_add_point(rc, [int2d] { 0, 0 }, __raw(x0))
  c.legion_point_coloring_add_point(rc, [int2d] { 1, 0 }, __raw(x1))
  c.legion_point_coloring_add_point(rc, [int2d] { 2, 0 }, __raw(x2))
  var cs = ispace(int2d, { 3, 1 })
  var p = partition(disjoint, r, rc, cs)
  c.legion_point_coloring_destroy(rc)

  var q = preimage(s, p, s)

  for x in r do
    @x = 1
  end

  for color in cs do
    var ri = p[color]
    var si = q[color]
    for y in si do
      if not isnull(dynamic_cast(ptr(int, ri), @y)) then
        r[@y] *= color.x + 2
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
  regentlib.assert(f() == 10, "test failed")
end
regentlib.start(main)
