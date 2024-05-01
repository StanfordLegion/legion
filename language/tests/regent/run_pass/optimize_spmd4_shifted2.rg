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

-- This tests the SPMD optimization of the compiler with:
--   * disjoint and aliased regions
--   * multiple fields
--   * multiple read-write tasks
--   * loop-carried region dependencies
-- where the partition colors (structured) don't start from (0, 0).

local c = regentlib.c
local format = require("std/format")

struct t {
  a : int,
  b : int,
  c : int,
}

task inc_ba(r : region(ispace(int2d), t))
where reads(r.b), reads writes(r.a) do
  for x in r do
    x.a += x.b
  end
end

task avg_ac(r : region(ispace(int2d), t), q : region(ispace(int2d), t))
where reads(r.a, q.a), reads writes(r.c) do
  var s, t = 0, 0
  for y in q do
    s += y.a
    t += 1
  end
  var a = s / t
  for x in r do
    x.c += x.a - a
  end
end

task rot_cb(r : region(ispace(int2d), t), y : int, z : int)
where reads(r.c), reads writes(r.b) do
  for x in r do
    if x.c >= 0 then
      x.b = (x.c + y) % z
    else
      x.b = -((-x.c + y) % z)
    end
  end
end

__demand(__replicable)
task main()
  var r = region(ispace(int2d, { x = 4, y = 4 }), t)
  var is_part = ispace(int2d, { x = 2, y = 2 }, { x = -1, y = -3 })

  var cp = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(cp, int2d{-1, -3}, rect2d{{0, 0}, {1, 1}})
  c.legion_domain_point_coloring_color_domain(cp, int2d{-1, -2}, rect2d{{0, 2}, {1, 3}})
  c.legion_domain_point_coloring_color_domain(cp, int2d{0, -3}, rect2d{{2, 0}, {3, 1}})
  c.legion_domain_point_coloring_color_domain(cp, int2d{0, -2}, rect2d{{2, 2}, {3, 3}})
  var p = partition(disjoint, r, cp, is_part)
  c.legion_domain_point_coloring_destroy(cp)

  var cq = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(cq, int2d{-1, -3}, rect2d{{0, 0}, {2, 3}})
  c.legion_domain_point_coloring_color_domain(cq, int2d{-1, -2}, rect2d{{1, 0}, {3, 2}})
  c.legion_domain_point_coloring_color_domain(cq, int2d{0, -3}, rect2d{{0, 3}, {3, 3}})
  c.legion_domain_point_coloring_color_domain(cq, int2d{0, -2}, rect2d{{3, 0}, {3, 3}})
  var q = partition(aliased, r, cq, is_part)
  c.legion_domain_point_coloring_destroy(cq)

  for x in r do
    x.a = 10000 + 10 * (x.x * r.bounds:size().y + x.y)
    x.b = 0
    x.c = 0
  end

  for x in r do
    format.println("x {} {} {}", x.a, x.b, x.c)
  end

  for i in is_part do
    inc_ba(p[i])
  end

  for t = 0, 3 do
    for i in is_part do
      avg_ac(p[i], q[i])
    end
    for i in is_part do
      rot_cb(p[i], 300, 137)
    end
    for i in is_part do
      inc_ba(p[i])
    end
    -- Communication happens here.
  end

  for x in r do
    format.println("x {} {} {}", x.a, x.b, x.c)
  end

  regentlib.assert(r[{ x=0, y=0 }].a ==  9812, "test failed")
  regentlib.assert(r[{ x=0, y=3 }].a ==  9820, "test failed")
  regentlib.assert(r[{ x=1, y=1 }].a == 10070, "test failed")
  regentlib.assert(r[{ x=2, y=0 }].a ==  9942, "test failed")
  regentlib.assert(r[{ x=3, y=2 }].a == 10028, "test failed")
  regentlib.assert(r[{ x=3, y=3 }].a == 10314, "test failed")
end
regentlib.start(main)
