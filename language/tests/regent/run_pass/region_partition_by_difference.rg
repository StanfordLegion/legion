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

task f()
  var r = region(ispace(ptr, 5), int)
  var x0 = new(ptr(int, r))
  var x1 = new(ptr(int, r))
  var x2 = new(ptr(int, r))
  var x3 = new(ptr(int, r))
  var s = region(ispace(ptr, 5), ptr(int, r))
  var y0 = new(ptr(ptr(int, r), s))
  var y1 = new(ptr(ptr(int, r), s))
  var y2 = new(ptr(ptr(int, r), s))
  var y3 = new(ptr(ptr(int, r), s))

  @y0 = x1
  @y1 = x0
  @y2 = x1
  @y3 = x2

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(x0))
  c.legion_coloring_add_point(rc, 0, __raw(x1))
  c.legion_coloring_add_point(rc, 1, __raw(x2))
  c.legion_coloring_add_point(rc, 2, __raw(x3))
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)

  var q = preimage(s, p, s)
  var p1 = p - image(r, q, s)

  for x in r do
    @x = 1
  end

  for i = 0, 3 do
    var ri = p1[i]
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
  regentlib.assert(f() == 7, "test failed")
end
regentlib.start(main)
