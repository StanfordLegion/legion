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

-- This tests the inner annotation.

task inc(r : region(int), y : int)
where reads writes(r) do
  for x in r do
    @x += y
  end
end

-- Can also do this to an empty task. While the task is eligible for
-- both inner and leaf, the user's annotation wins.
__demand(__inner)
task f()
end

__demand(__inner)
task main()
  var r = region(ispace(ptr, 4), int)
  fill(r, 0)
  inc(r, 100)
  f()
end
regentlib.start(main)
