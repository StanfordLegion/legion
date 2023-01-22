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
-- type_mismatch_partition_by_intersection_mirror2.rg:26: type mismatch: expected partition of int1d but got partition of int2d
--   var q = r & p
--               ^

import "regent"

task f()
  var r = region(ispace(int1d, 5), int)
  var s = region(ispace(int2d, {5, 5}), int)
  var p = partition(equal, s, ispace(int2d, {5, 5}))
  var q = r & p
end
f:compile()
