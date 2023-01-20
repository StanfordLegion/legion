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

import "regent"

task assert_disjoint(x : region(ispace(int3d), int),
                     y : region(ispace(int3d), int))
where x * y do end

local zero = terralib.constant(`int3d { regentlib.__int3d { 0, 0, 0 } })
local one = terralib.constant(`int3d { regentlib.__int3d { 0, 0, 1 } })

task main()
  var r = region(ispace(int3d, { 2, 2, 2 }), int)
  var p = partition(equal, r, ispace(int3d, { 1, 1, 2 }))
  var r0 = p[zero]
  var r1 = p[one]
  assert_disjoint(r0, r1)
end
regentlib.start(main)
