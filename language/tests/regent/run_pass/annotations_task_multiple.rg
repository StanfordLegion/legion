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

import "regent"

-- This tests multiple annotations to a task

task inc(r : region(int), y : int)
where reads writes(r) do
  for x in r do
    @x += y
  end
end

__demand(__inner) __forbid(__optimize)
task main()
  var r = region(ispace(ptr, 4), int)
  fill(r, 0)
  inc(r, 100)
end
regentlib.start(main)
