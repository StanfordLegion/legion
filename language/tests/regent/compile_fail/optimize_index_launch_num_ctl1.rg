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
-- optimize_index_launch_num_ctl1.rg:40: constant time launch failed: argument 4 is not provably projectable or invariant
--     f(r, 0, j, j+i+2)
--      ^

import "regent"

task f(r : region(ispace(int1d), int), x : int, y : int, z : int)
where reads(r) do
end

task g(r : region(ispace(int1d), int), x : int, y : int, z : int)
where reads writes(r) do
end

task main()
  var r = region(ispace(int1d, 10), int)
  var p = partition(equal, r, ispace(int1d, 4))
  fill(r, 0)

  var j = 2
  var k = 3

  __demand(__constant_time_launch)
  for i = 0, 4 do
    f(r, 0, j, j+i+2)
  end
end
regentlib.start(main)
