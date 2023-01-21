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

-- This tests a bug in the high-level runtime, resulting in missed
-- copies from reduction instances.

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task inc_ba(r : region(t)) where reads(r.b), reduces +(r.a) do
  for x in r do
    x.a += x.b
  end
end

task avg_ac(r : region(t), q : region(t))
where reads(r.a, q.a), reads writes(r.c) do
  var s, t = 0, 0
  for y in q do
    c.printf("y.a %d\n", y.a)
    s += y.a
    t += 1
  end
  var a = s / t
  for x in r do
    x.c += x.a - a
  end
end

task rot_cb(r : region(t), y : int, z : int)
where reads(r.c), reads writes(r.b) do
  for x in r do
    if x.c >= 0 then
      x.b = (x.c + y) % z
    else
      x.b = -((-x.c + y) % z)
    end
  end
end

task main()
  var r = region(ispace(ptr, 4), t)
  var x0 = dynamic_cast(ptr(t, r), 0)
  var x1 = dynamic_cast(ptr(t, r), 1)
  var x2 = dynamic_cast(ptr(t, r), 2)
  var x3 = dynamic_cast(ptr(t, r), 3)

  var cp = c.legion_coloring_create()
  c.legion_coloring_add_point(cp, 0, __raw(x0))
  c.legion_coloring_add_point(cp, 1, __raw(x1))
  c.legion_coloring_add_point(cp, 2, __raw(x2))
  c.legion_coloring_add_point(cp, 3, __raw(x3))
  var p = partition(disjoint, r, cp)
  c.legion_coloring_destroy(cp)

  var cq = c.legion_coloring_create()
  c.legion_coloring_add_point(cq, 0, __raw(x0))
  c.legion_coloring_add_point(cq, 0, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x0))
  c.legion_coloring_add_point(cq, 1, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x1))
  c.legion_coloring_add_point(cq, 2, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x3))
  c.legion_coloring_add_point(cq, 3, __raw(x2))
  c.legion_coloring_add_point(cq, 3, __raw(x3))
  var q = partition(aliased, r, cq)
  c.legion_coloring_destroy(cq)

  var i = 0
  for x in r do
    x.a = 10000 + 10 * i
    x.b = 0
    x.c = 0
    i += 1
  end

  for x in r do
    c.printf("x.{a, b, c} %d %d %d\n", x.a, x.b, x.c)
  end

  var pieces = 4

  for i = 0, 2 do
    for i = 0, pieces do
      inc_ba(p[i])
    end
    -- Communication happens here.
    for i = 0, pieces do
      avg_ac(p[i], q[i])
    end
    for i = 0, pieces do
      rot_cb(p[i], 300, 137)
    end
  end

  for x in r do
    c.printf("x.{a, b, c} %d %d %d\n", x.a, x.b, x.c)
  end

  regentlib.assert(x0.c == -38, "test failed")
  regentlib.assert(x1.c ==  19, "test failed")
  regentlib.assert(x2.c ==  -1, "test failed")
  regentlib.assert(x3.c ==  13, "test failed")
end
regentlib.start(main)
