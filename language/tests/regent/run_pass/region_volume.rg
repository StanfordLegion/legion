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

local c = regentlib.c

task main()
  var is = ispace(int2d, {4, 8})
  var r = region(is, int)
  var p = partition(equal, r, ispace(int2d, {2, 2}))
  var r00 = p[{0, 0}]
  var r11 = p[{1, 1}]
  var r_volume = r.volume
  var r00_volume = r00.volume
  var r11_volume = r11.volume

  regentlib.assert(r_volume == 32, "test failed")
  regentlib.assert(r00_volume == 8, "test failed")
  regentlib.assert(r11_volume == 8, "test failed")
end
regentlib.start(main)
