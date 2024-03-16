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

-- This tests the compiler's ability to reuse the same projection
-- functor multiple times.

local c = regentlib.c

terra e(x : int) : int
  return 3
end

task f1(r : region(ispace(int1d), int), i : int1d)
where reads(r) do
  for x in r do
    regentlib.assert(x == (i + 1) % 5, "test failed")
  end
end

task g1(r : region(ispace(int1d), int), i : int1d)
where reads writes(r) do
  for x in r do
    regentlib.assert(x == (i + 1) % 5, "test failed")
  end
end

task h1(r : region(ispace(int1d), int), i : int1d) : int
where reduces+(r) do
  for x in r do
    regentlib.assert(x == (i + 1) % 5, "test failed")
  end
  return 5
end

task main()
  var n = 5
  var r = region(ispace(int1d, n), int)
  fill(r, 1)

  var cs = ispace(int1d, n)
  var p_disjoint = partition(equal, r, cs)

  -- case 1. no capture
  __demand(__index_launch)
  for i in cs do
    f1(p_disjoint[(i + 1) % 5], i)
  end
  __demand(__index_launch)
  for i in cs do
    g1(p_disjoint[(i + 1) % 5], i)
  end
  var x = 0
  __demand(__index_launch)
  for i in cs do
    x += h1(p_disjoint[(i + 1) % 5], i)
  end
  regentlib.assert(x == 25, "test failed")

  -- case 2. capture simple variable
  __demand(__index_launch)
  for i in cs do
    f1(p_disjoint[(i + 1) % n], i)
  end
  __demand(__index_launch)
  for i in cs do
    g1(p_disjoint[(i + 1) % n], i)
  end
  x = 0
  __demand(__index_launch)
  for i in cs do
    x += h1(p_disjoint[(i + 1) % n], i)
  end
  regentlib.assert(x == 25, "test failed")
end
regentlib.start(main)
assert(regentlib.count_projection_functors() == 2)
