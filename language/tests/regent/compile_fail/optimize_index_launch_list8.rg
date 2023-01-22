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

-- fails-with:
-- optimize_index_launch_list8.rg:72: loop optimization failed: argument 1 interferes with itself
--     g(p_aliased[i])
--      ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

struct t {
  f: int1d,
}

terra e(x : int) : int
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
where reads(r, s), writes(r, s) do
  return 5
end

task main()
  var n = 5
  var cs = ispace(int1d, n)
  var r = region(cs, t)
  for i in cs do
    r[i].f = i/2
  end
  var p_disjoint = partition(equal, r, cs)
  var p_aliased = image(r, p_disjoint, r.f)
  var r0 = p_disjoint[0]
  var r1 = p_disjoint[1]
  var p0_disjoint = partition(equal, r0, cs)
  var p1_disjoint = partition(equal, r1, cs)

  -- not optimized: loop-variant argument is interfering
  __demand(__index_launch)
  for i in cs do
    g(p_aliased[i])
  end
end
regentlib.start(main)
