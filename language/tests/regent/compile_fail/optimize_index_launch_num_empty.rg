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
-- optimize_index_launch_num_empty.rg:26: loop optimization failed: body is empty
--   for i = 0, 4 do
--     ^

import "regent"

__forbid(__leaf)
__demand(__inner)
task main()
  __demand(__index_launch)
  for i = 0, 4 do
  end
end
regentlib.start(main)
