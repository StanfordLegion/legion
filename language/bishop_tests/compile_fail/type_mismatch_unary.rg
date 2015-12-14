-- Copyright 2015 Stanford University, NVIDIA Corporation
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
-- ...tanford/projects/legion/language/src/bishop/type_check.t:115: unary op - requires the rhs to be of integer type
-- type_mismatch_unary.rg:26: 
--   target : processors[-$proc];
--                          ^

import "bishop"

mapper

task[target=$proc] {
  target : processors[-$proc];
}

end
