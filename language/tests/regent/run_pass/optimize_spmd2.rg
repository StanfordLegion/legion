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
--   * disjoint regions
--   * multiple fields
--   * multiple read-write tasks
--   * variable loop bounds
--   * constant time bounds
--   * read-only references to scalars inside the loop

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task inc_ab(r : region(t), y : int)
where reads writes(r.{a, b}) do
  for x in r do
    x.a += y
    x.b += y
  end
end

task inc_bc(r : region(t), y : int)
where reads writes(r.{b, c}) do
  for x in r do
    x.b += y
    x.c += y
  end
end

task inc_ca(r : region(t), y : int)
where reads writes(r.{c, a}) do
  for x in r do
    x.c += y
    x.a += y
  end
end

task check(r : region(t))
where reads(r) do
  for x in r do
    regentlib.c.printf("%d %d %d\n", x.a, x.b, x.c)
    regentlib.assert(x.a == 30903, "test failed")
    regentlib.assert(x.b == 50063, "test failed")
    regentlib.assert(x.c == 70960, "test failed")
  end
end

__demand(__replicable)
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

  for x in r do
    x.a = 30000
    x.b = 50000
    x.c = 70000
  end

  var start = 0
  var stop = 4
  var inc_by_ab = 1
  var inc_by_bc = 20
  var inc_by_ca = 300

  __demand(__spmd)
  for t = 0, 3 do
    for i = start, stop do
      inc_ab(p[i], inc_by_ab)
    end
    for i = start, stop do
      inc_bc(p[i], inc_by_bc)
    end
    for i = start, stop do
      inc_ca(p[i], inc_by_ca)
    end
  end

  check(r)
end
regentlib.start(main)
