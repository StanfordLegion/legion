-- Copyright 2016 Stanford University
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
-- type_mismatch_list_ispace_mapping4.rg:28: no applicable overloaded function mapping for arguments int3d, rect3d
--   var l = list_ispace(is, mapping)
--                                 ^

import "regent"

terra mapping(p : int2d, s : rect2d)
  return 0
end

task f()
  var is = ispace(int3d, { x = 5, y = 5, z = 5 })
  var l = list_ispace(is, mapping)
end
