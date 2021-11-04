-- Copyright 2021 Stanford University
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
-- optimize_index_launch_num_nested.rg:49: loop optimization failed: argument 1 interferes with itself
--       g(s[compute_index(i, j)][j])
--        ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

task g(r : region(int)) : int
where reads(r), writes(r) do
  return 5
end

terra compute_index(i : int, j : int)
  return i
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var p = partition(equal, r, ispace(int1d, 2))
  var q = partition(equal, r, ispace(int1d, 2))
  var s = cross_product(p, q)

  for i = 0, 2 do
    __demand(__index_launch)
    for j = 0, 2 do
      g(s[compute_index(i, j)][j])
    end
  end
end
regentlib.start(main)
