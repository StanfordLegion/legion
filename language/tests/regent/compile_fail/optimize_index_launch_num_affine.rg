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
-- optimize_index_launch_num_affine.rg:42: loop optimization failed: argument 1 interferes with itself
--     g(p[i * 2 - i - i])
--      ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

local c = regentlib.c

task g(r : region(int)) : int
where reads(r), writes(r) do
  return 5
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var p = partition(equal, r, ispace(int1d, 5))

  __demand(__index_launch)
  for i = 0, 2 do
    g(p[i * 2 - i - i])
  end
end
regentlib.start(main)
