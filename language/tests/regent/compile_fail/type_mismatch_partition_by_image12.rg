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
-- type_mismatch_partition_by_image12.rg:27: type mismatch in argument 3: expected field with dim 2 but got int1d
--   var i = image(r, p, s) -- type error: int1d vs int2d
--               ^

import "regent"

task main()
  var r = region(ispace(int2d, {4, 4}), double)
  var s = region(ispace(int3d, {2, 2, 2}), int1d)
  fill(s, 0)
  var p = partition(equal, s, ispace(int1d, 4))
  var i = image(r, p, s) -- type error: int1d vs int2d
end
regentlib.start(main)
