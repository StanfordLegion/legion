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
-- annotations_for_list_cuda_without_task.rg:25: option __demand(__cuda) is not permitted
--   for i in is do end
--     ^

import "regent"

task f()
  var is = ispace(int1d, 5)
  __demand(__cuda) -- invalid because task is not __demand(__cuda)
  for i in is do end
end
f:compile()
