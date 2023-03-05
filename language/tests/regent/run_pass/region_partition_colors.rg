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

task main()
  var c0 = ispace(ptr, 5)
  var r0 = region(c0, int)
  var p0 = partition(equal, r0, c0)
  regentlib.assert(p0.colors.volume == 5, "test failed")

  var c1 = ispace(int1d, 12)
  var r1 = region(c1, int1d)
  var p1 = partition(equal, r1, c1)
  regentlib.assert(p1.colors.volume == 12, "test failed")

  var c2 = ispace(int2d, { 4, 4 })
  var r2 = region(c2, int2d)
  var p2 = partition(equal, r2, c2)
  regentlib.assert(p2.colors.volume == 16, "test failed")

  var c3 = ispace(int3d, { 4, 5, 6 })
  var r3 = region(c3, int3d)
  var p3 = partition(equal, r3, c3)
  regentlib.assert(p3.colors.volume == 120, "test failed")
end
regentlib.start(main)
