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
-- type_mismatch_partition4.rg:28: type mismatch in argument 3: expected legion_point_coloring_t but got legion_domain_point_coloring_t
--   var p = partition(disjoint, r, s, t)
--                   ^

import "regent"

local c = regentlib.c

task f() : int
  var r = region(ispace(ptr, 5), int)
  var s = c.legion_domain_point_coloring_create()
  var t = ispace(int2d, { x = 1, y = 2 })
  var p = partition(disjoint, r, s, t)
end
f:compile()
