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

-- A series of progressively more complicated tests to test Legion Spy capability.

import "regent"

task hello(r : region(int))
where reads writes(r) do
end

task world(r : region(int))
where reads(r) do
end

task main()
  var r = region(ispace(ptr, 5), int)
  var p = partition(equal, r, ispace(int1d, 2))
  fill(r, 0)
  __demand(__index_launch)
  for i = 0, 2 do
    hello(p[i])
  end
  for i = 0, 2 do
    world(p[i])
  end
end
regentlib.start(main)
