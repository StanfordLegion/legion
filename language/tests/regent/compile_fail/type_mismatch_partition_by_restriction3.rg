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
-- type_mismatch_partition_by_restriction3.rg:27: type mismatch in argument 4: expected ispace but got int32
--   var p = restrict(r, t, e, colors)
--                  ^

import "regent"

task f()
  var r = region(ispace(int2d, {3, 1}), int)
  var t : transform(2, 1)
  var e : rect2d
  var colors : int
  var p = restrict(r, t, e, colors)
end
