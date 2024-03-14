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

-- fails-with:
-- optimize_index_launch_list_ctl2.rg:41: constant time launch failed: argument 3 is not provably projectable or invariant
--     f(r, 0, i, j+k+2)
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
  var is = ispace(int1d, 4)
  var p = partition(equal, r, is)
  fill(r, 0)

  var j = 2
  var k = 3

  __demand(__constant_time_launch)
  for i in is do
    f(r, 0, i, j+k+2)
  end
end
regentlib.start(main)
