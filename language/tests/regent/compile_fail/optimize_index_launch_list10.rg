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
-- [["-findex-launch-dynamic", "0"]]

-- fails-with:
-- optimize_index_launch_list10.rg:88: loop optimization failed: argument 2 interferes with argument 1
--     g2(p0_disjoint[i], p1_disjoint[i])
--       ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

struct t {
  f: int1d,
  n: int,
}

terra e(x : int) : int
  return 3
end

terra e_bad(x : c.legion_runtime_t) : int
  return 3
end

task f(r : region(ispace(int1d), t)) : int
where reads(r) do
  return 5
end

task f2(r : region(ispace(int1d), t), s : region(ispace(int1d), t)) : int
where reads(r, s) do
  return 5
end

task g(r : region(ispace(int1d), t)) : int
where reads(r), writes(r) do
  return 5
end

task g2(r : region(ispace(int1d), t), s : region(ispace(int1d), t)) : int
where reads(r, s), writes(r) do
  return 5
end

task h(r : region(ispace(int1d), t)) : int
where reduces +(r.n) do
  return 5
end

task h2(r : region(ispace(int1d), t), s : region(ispace(int1d), t)) : int
where reduces +(r.n, s.n) do
  return 5
end

task h2b(r : region(ispace(int1d), t), s : region(ispace(int1d), t)) : int
where reduces +(r.n), reduces *(s.n) do
  return 5
end

task with_partitions(cs : ispace(int1d),
                     r0 : region(ispace(int1d), t),
                     p0_disjoint : partition(disjoint, r0, cs),
                     r1 : region(ispace(int1d), t),
                     p1_disjoint : partition(disjoint, r1, cs))
where reads(r0, r1), writes(r0, r1) do

  -- not optimized: loop-variant argument is (statically) interfering
  __demand(__index_launch)
  for i in cs do
    g2(p0_disjoint[i], p1_disjoint[i])
  end
end

task main()
  var n = 5
  var cs = ispace(int1d, n)
  var r = region(cs, t)
  for i in cs do
    r[i].f = i/2
    r[i].n = i/2
  end
  var p_disjoint = partition(equal, r, cs)
  var p_aliased = image(r, p_disjoint, r.f)
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(equal, r0, cs)
  var p1_disjoint = partition(equal, r1, cs)

  with_partitions(cs, r0, p0_disjoint, r1, p1_disjoint)
end
regentlib.start(main)
