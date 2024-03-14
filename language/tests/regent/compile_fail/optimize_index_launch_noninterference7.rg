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
-- optimize_index_launch_noninterference7.rg:45: loop optimization failed: argument 1 interferes with itself
--     f_mixed(p[(i+1)%10])
--            ^

import "regent"

-- Tests for dynamic index launches.

fspace fs {
  x : int,
  y : int,
  z : int,
}

task f_mixed(r : region(fs))
where reads(r.x), reads writes(r.y) do
end

task main()
  var r = region(ispace(ptr, 10), fs)
  var p = partition(equal, r, ispace(int1d, 10))
  fill(r.{x, y, z}, 0)

  -- This cannot be proven statically (even though at runtime it's ok)
  __demand(__index_launch)
  for i = 0, 10 do
    f_mixed(p[(i+1)%10])
  end
end
regentlib.start(main)
