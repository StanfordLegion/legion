-- Copyright 2018 Stanford University
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
-- optimize_index_launch_num_affine2.rg:37: loop optimization failed: argument 2 interferes with argument 1
--     g2(p[i], p[i+1])
--       ^

import "regent"

-- This tests the various loop optimizations supported by the
-- compiler.

task g2(r : region(int), s : region(int)) : int
where reads writes(r, s) do
  return 5
end

task main()
  var n = 5
  var r = region(ispace(ptr, n), int)
  var p = partition(equal, r, ispace(int1d, 5))

  __demand(__parallel)
  for i = 0, 2 do
    g2(p[i], p[i+1])
  end
end
regentlib.start(main)
