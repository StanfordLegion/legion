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

local c = regentlib.c

task main()
  var t = region(ispace(ptr, 5), int)
  var tp = partition(equal, t, ispace(int1d, 2))
  var t0 = tp[0]
  var t1 = tp[1]

  var x1 = dynamic_cast(ptr(int, t1), 4)
  var x01 = static_cast(ptr(int, t0, t1), x1)
  var x01_0 = static_cast(ptr(int, t0), x01)
  var x01_1 = static_cast(ptr(int, t1), x01)
  regentlib.assert(isnull(x01_0), "test failed")
  regentlib.assert(not isnull(x01_1), "test failed")

  var x10 = static_cast(ptr(int, t1, t0), x1)
  var x10_0 = static_cast(ptr(int, t0), x10)
  var x10_1 = static_cast(ptr(int, t1), x10)
  regentlib.assert(isnull(x10_0), "test failed")
  regentlib.assert(not isnull(x10_1), "test failed")
end
regentlib.start(main)
