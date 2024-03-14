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
-- type_mismatch_assignment3.rg:29: type mismatch in assignment: expected ptr(int32, $s) but got ptr(int32, $t)
--   xs = xt
--    ^

import "regent"

task f()
  var s = region(ispace(ptr, 5), int)
  var t = region(ispace(ptr, 5), int)

  var xs = dynamic_cast(ptr(int, s), 0)
  var xt = dynamic_cast(ptr(int, t), 0)

  xs = xt
end
f:compile()
