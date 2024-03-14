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
-- type_mismatch_static_cast3.rg:26: incompatible pointers for static_cast: ptr(int32, $r) and ptr(float, $s)
--   var y = static_cast(ptr(int, r), x)
--                     ^

import "regent"

task f()
  var r = region(ispace(ptr, 5), int)
  var s = region(ispace(ptr, 5), float)
  var x = dynamic_cast(ptr(float, s), 0)
  var y = static_cast(ptr(int, r), x)
end
f:compile()
