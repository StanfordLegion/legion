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

local format = require("std/format")

task f(x : int)
  return x + 1
end

task main()
  var y : int = 123
  var z = regentlib.c.legion_future_from_untyped_pointer(__runtime(), &y, [terralib.sizeof(int)])
  var w = __future(int, z)
  regentlib.assert(f(w) == 124, "test failed")

  var a = __future(int64, regentlib.c.legion_issue_timing_op_microseconds(__runtime(), __context()))
  format.println("current time is {}", a/1.0e6)
end
regentlib.start(main)
