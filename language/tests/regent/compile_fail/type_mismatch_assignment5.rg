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
-- type_mismatch_assignment5.rg:23: type mismatch in assignment: expected ptr(int32, $s, $t) but got ptr(int32, $t, $s)
--   xst = xts
--     ^

import "regent"

task f(s : region(int), t : region(int), xst : ptr(int, s, t), xts : ptr(int, t, s))
  xst = xts
end
f:compile()
