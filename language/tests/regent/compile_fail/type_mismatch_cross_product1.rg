-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- type_mismatch_cross_product1.rg:26: cross product expected at least 2 arguments, got 1
--   var cp = cross_product(p1)
--                        ^

import "regent"

task f() : int
  var r = region(ispace(ptr, 5), int)
  var c : regentlib.c.legion_coloring_t
  var p1 = partition(disjoint, r, c)
  var cp = cross_product(p1)
end
f:compile()
