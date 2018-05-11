-- Copyright 2018 Stanford University
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
-- constraint_affine_2d.rg:34: invalid cast missing constraint $x * $y
--     assert_disjoint { x = p[i], y = p[i + { 0, 0 }] }
--                     ^

import "regent"

fspace assert_disjoint {
  x : region(ispace(int2d), int),
  y : region(ispace(int2d), int),
} where x * y end

task main()
  var t = region(ispace(int2d, { 5, 5 }), int)
  var p = partition(equal, t, ispace(int2d, { 4, 2 }))

  var s = ispace(int2d, { 2, 1 })

  for i in s do
    assert_disjoint { x = p[i], y = p[i + { 0, 0 }] }
  end
end
regentlib.start(main)
