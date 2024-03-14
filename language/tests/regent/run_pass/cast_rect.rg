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

import "regent"

task main()
  var x1 = rect1d { 1, 5 }
  var y1 : regentlib.c.legion_rect_1d_t = x1
  var z1 : rect1d = y1
  regentlib.assert(z1.lo == int1d(1), "test failed")
  regentlib.assert(z1.hi == int1d(5), "test failed")

  var x2 = rect2d { { 2, 2 }, { 4, 4 } }
  var y2 : regentlib.c.legion_rect_2d_t = x2
  var z2 : rect2d = y2
  regentlib.assert(z2.lo == int2d { 2, 2 }, "test failed")
  regentlib.assert(z2.hi == int2d { 4, 4 }, "test failed")

  var x3 = rect3d { { 1, 2, 3 }, { 4, 5, 6 } }
  var y3 : regentlib.c.legion_rect_3d_t = x3
  var z3 : rect3d = y3
  regentlib.assert(z3.lo == int3d { 1, 2, 3 }, "test failed")
  regentlib.assert(z3.hi == int3d { 4, 5, 6 }, "test failed")
end
regentlib.start(main)
