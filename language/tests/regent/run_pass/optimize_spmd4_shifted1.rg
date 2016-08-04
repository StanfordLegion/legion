-- Copyright 2016 Stanford University
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
--   * disjoint and aliased regions
--   * multiple fields
--   * multiple read-write tasks
--   * loop-carried region dependencies
-- where the partition colors don't start from 0.

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

task avg_ac(r : region(t), q : region(t))
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
  var r = region(ispace(ptr, 5), t)
  var x0 = new(ptr(t, r))
  var x1 = new(ptr(t, r))
  var x2 = new(ptr(t, r))
  var x3 = new(ptr(t, r))

  var cp = c.legion_coloring_create()
  c.legion_coloring_add_point(cp, 10, __raw(x0))
  c.legion_coloring_add_point(cp, 11, __raw(x1))
  c.legion_coloring_add_point(cp, 12, __raw(x2))
  c.legion_coloring_add_point(cp, 13, __raw(x3))
  var p = partition(disjoint, r, cp)
  c.legion_coloring_destroy(cp)

  var cq = c.legion_coloring_create()
  c.legion_coloring_add_point(cq, 10, __raw(x0))
  c.legion_coloring_add_point(cq, 10, __raw(x1))
  c.legion_coloring_add_point(cq, 11, __raw(x0))
  c.legion_coloring_add_point(cq, 11, __raw(x1))
  c.legion_coloring_add_point(cq, 11, __raw(x2))
  c.legion_coloring_add_point(cq, 12, __raw(x1))
  c.legion_coloring_add_point(cq, 12, __raw(x2))
  c.legion_coloring_add_point(cq, 12, __raw(x3))
  c.legion_coloring_add_point(cq, 13, __raw(x2))
  c.legion_coloring_add_point(cq, 13, __raw(x3))
  var q = partition(aliased, r, cq)
  c.legion_coloring_destroy(cq)

  var i = 0
  -- Note: This test hits a bug in the vectorizer if vectorization is allowed.
  __forbid(__vectorize)
  for x in r do
    x.a = 10000 + 10 * i
    x.b = 0
    x.c = 0
    i += 1
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  var lower = 10
  var upper = 14

  for i = lower, upper do
    inc_ba(p[i])
  end

  __demand(__spmd)
  for t = 0, 3 do
    for i = lower, upper do
      avg_ac(p[i], q[i])
    end
    for i = lower, upper do
      rot_cb(p[i], 300, 137)
    end
    for i = lower, upper do
      inc_ba(p[i])
    end
    -- Communication happens here.
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  regentlib.assert(x0.a ==  9890, "test failed")
  regentlib.assert(x1.a == 10206, "test failed")
  regentlib.assert(x2.a ==  9945, "test failed")
  regentlib.assert(x3.a == 10180, "test failed")
end
regentlib.start(main)
