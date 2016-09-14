-- Copyright 2016 Stanford University
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

task toplevel()
  var r = region(ispace(int2d, {10, 10}, {2, 2}), int)
  var p = partition(equal, r, ispace(int2d, {2, 2}))

  regentlib.assert(p[{0, 0}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{0, 0}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{0, 0}].bounds.lo.y == 2, "test failed")
  regentlib.assert(p[{0, 0}].bounds.hi.y == 6, "test failed")

  regentlib.assert(p[{1, 0}].bounds.lo.x == 7, "test failed")
  regentlib.assert(p[{1, 0}].bounds.hi.x == 11, "test failed")
  regentlib.assert(p[{1, 0}].bounds.lo.y == 2, "test failed")
  regentlib.assert(p[{1, 0}].bounds.hi.y == 6, "test failed")

  regentlib.assert(p[{0, 1}].bounds.lo.x == 2, "test failed")
  regentlib.assert(p[{0, 1}].bounds.hi.x == 6, "test failed")
  regentlib.assert(p[{0, 1}].bounds.lo.y == 7, "test failed")
  regentlib.assert(p[{0, 1}].bounds.hi.y == 11, "test failed")

  regentlib.assert(p[{1, 1}].bounds.lo.x == 7, "test failed")
  regentlib.assert(p[{1, 1}].bounds.hi.x == 11, "test failed")
  regentlib.assert(p[{1, 1}].bounds.lo.y == 7, "test failed")
  regentlib.assert(p[{1, 1}].bounds.hi.y == 11, "test failed")
end

regentlib.start(toplevel)
