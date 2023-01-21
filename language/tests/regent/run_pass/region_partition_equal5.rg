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

task test_1d()
  var r = region(ispace(int1d, 10, 2), int)
  var p = partition(equal, r, ispace(int1d, 2, 2))

  -- FIXME: Inlining these expressions in the following assertions
  --        breaks the compiler
  var bounds0 = p[2].bounds
  var bounds1 = p[3].bounds

  regentlib.assert([int](bounds0.lo) == 2, "test failed")
  regentlib.assert([int](bounds0.hi) == 6, "test failed")

  regentlib.assert([int](bounds1.lo) == 7, "test failed")
  regentlib.assert([int](bounds1.hi) == 11, "test failed")
end

task test_2d()
  var r = region(ispace(int2d, {10, 10}, {2, 3}), int)
  var p = partition(equal, r, ispace(int2d, {2, 2}, {3, 3}))

  regentlib.assert(p[{3, 3}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{3, 3}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{3, 3}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p[{3, 3}].bounds.hi.y == 7, "test failed")

  regentlib.assert(p[{4, 3}].bounds.lo.x == 7, "test failed")
  regentlib.assert(p[{4, 3}].bounds.hi.x == 11, "test failed")
  regentlib.assert(p[{4, 3}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p[{4, 3}].bounds.hi.y == 7, "test failed")

  regentlib.assert(p[{3, 4}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{3, 4}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{3, 4}].bounds.lo.y == 8, "test failed")
  regentlib.assert(p[{3, 4}].bounds.hi.y == 12, "test failed")

  regentlib.assert(p[{4, 4}].bounds.lo.x == 7, "test failed")
  regentlib.assert(p[{4, 4}].bounds.hi.x == 11, "test failed")
  regentlib.assert(p[{4, 4}].bounds.lo.y == 8, "test failed")
  regentlib.assert(p[{4, 4}].bounds.hi.y == 12, "test failed")
end

task test_3d()
  var r = region(ispace(int3d, {10, 10, 10}, {2, 3, 4}), int)
  var p = partition(equal, r, ispace(int3d, {2, 2, 2}, {4, 4, 4}))

  regentlib.assert(p[{4, 4, 4}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{4, 4, 4}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{4, 4, 4}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p[{4, 4, 4}].bounds.hi.y == 7, "test failed")
  regentlib.assert(p[{4, 4, 4}].bounds.lo.z == 4, "test failed")
  regentlib.assert(p[{4, 4, 4}].bounds.hi.z == 8, "test failed")

  regentlib.assert(p[{4, 5, 4}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{4, 5, 4}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{4, 5, 4}].bounds.lo.y == 8, "test failed")
  regentlib.assert(p[{4, 5, 4}].bounds.hi.y == 12, "test failed")
  regentlib.assert(p[{4, 5, 4}].bounds.lo.z == 4, "test failed")
  regentlib.assert(p[{4, 5, 4}].bounds.hi.z == 8, "test failed")

  regentlib.assert(p[{4, 4, 5}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{4, 4, 5}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{4, 4, 5}].bounds.lo.y == 3, "test failed")
  regentlib.assert(p[{4, 4, 5}].bounds.hi.y == 7, "test failed")
  regentlib.assert(p[{4, 4, 5}].bounds.lo.z == 9, "test failed")
  regentlib.assert(p[{4, 4, 5}].bounds.hi.z == 13, "test failed")

  regentlib.assert(p[{5, 5, 4}].bounds.lo.x == 7, "test failed")
  regentlib.assert(p[{5, 5, 4}].bounds.hi.x == 11, "test failed")
  regentlib.assert(p[{5, 5, 4}].bounds.lo.y == 8, "test failed")
  regentlib.assert(p[{5, 5, 4}].bounds.hi.y == 12, "test failed")
  regentlib.assert(p[{5, 5, 4}].bounds.lo.z == 4, "test failed")
  regentlib.assert(p[{5, 5, 4}].bounds.hi.z == 8, "test failed")

end

task toplevel()
  test_1d()
  test_2d()
  test_3d()
end

regentlib.start(toplevel)
