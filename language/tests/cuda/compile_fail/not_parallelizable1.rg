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
-- not_parallelizable1.rg:29: CUDA code generation failed: found a region access outside parallelizable loops
--   s[0] = 0
--    ^

import "regent"

__demand(__cuda)
task f(r : region(ispace(int1d), int),
       s : region(ispace(int1d), int))
where reads writes(r, s) do
  for e1 in r do
    r[e1] = 0
  end
  s[0] = 0
end
