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
-- type_mismatch_list_invert3.rg:26: type mismatch: expected list(list(phase_barrier)) but got int32
--   var x = list_invert(d, c, 1)
--                     ^

import "regent"

task f(r : region(int), p : partition(aliased, r), i : regentlib.list(int))
  var d = list_duplicate_partition(p, i)
  var c = list_cross_product(d, d)
  var b = list_phase_barriers(c)
  var x = list_invert(d, c, 1)
end
