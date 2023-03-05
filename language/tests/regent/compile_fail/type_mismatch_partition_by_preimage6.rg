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
-- type_mismatch_partition_by_preimage6.rg:35: type mismatch in argument 3: expected field of 64-bit index type (for now) but got i2d(int32, $r)
--   var q = preimage(s, p, s)
--                  ^

import "regent"

local struct i2 { x : int, y : int }
terra i2.metamethods.__add(a : i2, b : i2) : i2
  return i2 { x = a.x + b.x, y = a.y + b.y }
end
terra i2.metamethods.__sub(a : i2, b : i2) : i2
  return i2 { x = a.x - b.x, y = a.y - b.y }
end
local i2d = index_type(i2, "i2d")

task f()
  var r = region(ispace(i2d, { 5, 5 }), int)
  var s = region(ispace(i2d, { 5, 5 }), i2d(int, r))
  var p = partition(equal, r, ispace(i2d, { 3, 3 }))
  var q = preimage(s, p, s)
end
f:compile()
