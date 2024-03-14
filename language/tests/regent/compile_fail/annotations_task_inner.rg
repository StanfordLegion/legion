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
-- annotations_task_inner.rg:27: task is not a valid inner task
--   r[0] = 5
--    ^

import "regent"

-- This tests the inner annotation.

__demand(__inner)
task f()
  var r = region(ispace(int1d, 4), int)
  r[0] = 5
end
f:compile()
