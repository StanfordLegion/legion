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
-- type_mismatch_call_partition3.rg:31: type mismatch in argument 3: expected partition(disjoint, $r) but got partition(disjoint, $r, $cs)
--   f(is, r, p)
--   ^

import "regent"

local c = regentlib.c

task f(is : ispace(int2d), x : region(is, int), p : partition(disjoint, x)) end

task g()
  var is = ispace(int2d, { x = 2, y = 2 })
  var r = region(is, int)
  var cs = ispace(int2d, { x = 2, y = 1 })
  var p = partition(equal, r, cs)
  f(is, r, p)
end
g:compile()
