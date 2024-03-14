-- Copyright 2024 Stanford University, NVIDIA Corporation
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
-- type_mismatch_ternary2.rg:25: ternary op expects the same type on both expressions, but got types 'processor_list_type' and 'processor_type'
-- target : processors.size > 0 ? processors : processors[1];
--                                         ^

import "bishop"

mapper

task[target=$proc] {
  target : processors.size > 0 ? processors : processors[1];
}

end
