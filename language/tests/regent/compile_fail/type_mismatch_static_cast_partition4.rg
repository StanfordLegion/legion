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
-- type_mismatch_static_cast_partition4.rg:27: the region $r is not a subregion of $s
--   var y = static_cast(partition(disjoint, s, cs), p)
--                     ^

import "regent"

task f()
  var r = region(ispace(int1d, 4), int)
  var s = region(ispace(int1d, 4), int)
  var cs = ispace(int1d, 2)
  var p = partition(equal, r, cs)
  var y = static_cast(partition(disjoint, s, cs), p)
end
f:compile()
