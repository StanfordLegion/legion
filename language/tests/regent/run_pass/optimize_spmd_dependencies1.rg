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
--   * loop-carried dependencies for multiple aliased partitions

fspace fs {
  a : int,
  b : int,
}

task taskA(r : region(ispace(int1d), fs), g : region(ispace(int1d), fs))
where reads(r.a, g.a), reads writes(r.b) do
  for i in r do
    r[i].b = (r[i].b + r[i].a * g[max(i-1, g.bounds.lo)].a) % 3617
  end
end

task taskB(r : region(ispace(int1d), fs))
where reads writes(r.{a, b}) do
  for i in r do
    r[i].a += 1
    r[i].b += 20
  end
end

task taskC(r : region(ispace(int1d), fs), g : region(ispace(int1d), fs))
where reads(r.a, g.a), reads writes(r.b) do
  for i in r do
    r[i].b = (r[i].b + r[i].a * g[min(i+1, g.bounds.hi)].a) % 5119
  end
end

task taskD(r : region(ispace(int1d), fs))
where reads writes(r.{a, b}) do
  for i in r do
    r[i].a += 300
    r[i].b += 4000
  end
end

local c = regentlib.c

task toplevel()
  var r = region(ispace(int1d, 4), fs)

  var cs = ispace(int1d, 2)
  var rp = partition(equal, r, cs)

  var coloring1 = c.legion_domain_point_coloring_create()
  var coloring2 = c.legion_domain_point_coloring_create()

  c.legion_domain_point_coloring_color_domain(coloring1, int1d(0), rect1d { 0, 2 })
  c.legion_domain_point_coloring_color_domain(coloring1, int1d(1), rect1d { 2, 3 })
  c.legion_domain_point_coloring_color_domain(coloring2, int1d(0), rect1d { 0, 1 })
  c.legion_domain_point_coloring_color_domain(coloring2, int1d(1), rect1d { 1, 3 })

  var gp1 = partition(aliased, r, coloring1, cs)
  var gp2 = partition(aliased, r, coloring2, cs)

  c.legion_domain_point_coloring_destroy(coloring1)
  c.legion_domain_point_coloring_destroy(coloring2)

  for i in r do
    r[i].a = i*(i+1)
    r[i].b = 1000 - i
  end

  __demand(__spmd)
  for t = 0, 3 do
    for c in cs do taskA(rp[c], gp2[c]) end

    for c in cs do taskB(rp[c]) end

    for c in cs do taskC(rp[c], gp1[c]) end

    for c in cs do taskD(rp[c]) end
  end

  for i in r do
    c.printf("%d: %d %d\n", i, r[i].a, r[i].b)
  end

  regentlib.assert(r[0].b == 7741, "test failed")
  regentlib.assert(r[1].b == 6282, "test failed")
  regentlib.assert(r[2].b == 5005, "test failed")
  regentlib.assert(r[3].b == 6011, "test failed")
end

regentlib.start(toplevel)
