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

-- runs-with:
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with:
--   * a partition which is opened before the loop starts
--   * multiple fields, **only some of which are written in the main loop**

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task inc_ba(r : region(t))
where reads(r.b), reads writes(r.a) do
  for x in r do
    x.a += x.b
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
    x.a = 10000
    x.b = 10 * i
    x.c = 0
    i += 1
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  var pieces = 4

  for i = 0, pieces do
    inc_ba(p[i])
  end

  __demand(__spmd)
  for t = 0, 10 do
    for i = 0, pieces do
      inc_ba(p[i])
    end
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  regentlib.assert(x0.a == 10000, "test failed")
  regentlib.assert(x1.a == 10110, "test failed")
  regentlib.assert(x2.a == 10220, "test failed")
  regentlib.assert(x3.a == 10330, "test failed")
end
regentlib.start(main)
