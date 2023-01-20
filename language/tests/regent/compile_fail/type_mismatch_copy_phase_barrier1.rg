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
-- type_mismatch_copy_phase_barrier1.rg:24: copy must await list of same or less depth than destination
--   copy(x, y, awaits(z))
--      ^

import "regent"

task f(x : region(int), y : region(int), z : regentlib.list(phase_barrier))
where reads writes(x, y) do
  copy(x, y, awaits(z))
end
f:compile()
