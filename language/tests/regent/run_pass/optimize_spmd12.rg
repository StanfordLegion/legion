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

-- runs-with:
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * multiple fields which require copies

local c = regentlib.c

struct t {
  a1 : int,
  a2 : int,
  b1 : int,
  b2 : int,
  c1 : int,
  c2 : int,
}

task inc_ba(r : region(t))
where reads(r.{b1, b2}), reads writes(r.{a1, a2}) do
  for x in r do
    x.a1 += x.b1
    x.a2 += x.b2
  end
end

task avg_ac(r : region(t), q : region(t))
where reads(r.{a1, a2}, q.{a1, a2}), reads writes(r.{c1, c2}) do
  var s1, s2, t = 0, 0, 0
  for y in q do
    s1 += y.a1
    s2 += y.a2
    t += 1
  end
  var a1 = s1 / t
  var a2 = s2 / t
  for x in r do
    x.c1 += x.a1 - a1
    x.c2 += x.a2 - a2
  end
end

task rot_cb(r : region(t), y : int, z : int)
where reads(r.{c1, c2}), reads writes(r.{b1, b2}) do
  for x in r do
    if x.c1 >= 0 then
      x.b1 = (x.c1 + y) % z
    else
      x.b1 = -((-x.c1 + y) % z)
    end
    if x.c2 >= 0 then
      x.b2 = (x.c2 + y) % z
    else
      x.b2 = -((-x.c2 + y) % z)
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
    x.a1 = 10000 + 10 * i
    x.a2 = 100 * i
    x.b1 = 0
    x.b2 = 0
    x.c1 = 0
    x.c2 = 0
    i += 1
  end

  for x in r do
    c.printf("x %d %d %d %d %d %d\n", x.a1, x.a2, x.b1, x.b2, x.c1, x.c2)
  end

  var pieces = 4

  for i = 0, pieces do
    inc_ba(p[i])
  end

  __demand(__spmd)
  for t = 0, 3 do
    for i = 0, pieces do
      avg_ac(p[i], q[i])
    end
    for i = 0, pieces do
      rot_cb(p[i], 300, 137)
    end
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
    for i = 0, pieces do
      inc_ba(p[i])
    end
    -- Communication happens here.
  end

  for x in r do
    c.printf("x %d %d %d %d %d %d\n", x.a1, x.a2, x.b1, x.b2, x.c1, x.c2)
  end

  regentlib.assert(x0.a1 ==  9550, "test failed")
  regentlib.assert(x1.a1 == 10598, "test failed")
  regentlib.assert(x2.a1 ==  9409, "test failed")
  regentlib.assert(x3.a1 == 10660, "test failed")

  regentlib.assert(x0.a2 == -534, "test failed")
  regentlib.assert(x1.a2 ==  706, "test failed")
  regentlib.assert(x2.a2 == -176, "test failed")
  regentlib.assert(x3.a2 ==  940, "test failed")
end
regentlib.start(main)
