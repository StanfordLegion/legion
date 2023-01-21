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

-- runs-with:
-- [["-ll:py", "1", "-ll:pyimport", "python_interop"]]

import "regent"

local format = require("std/format")

fspace fs1 {
  x : int32,
}

fspace fs2 {
  x : double,
}

extern task hello() : int32
hello:set_task_id(2)

extern task inc1(r : region(ispace(int1d), fs1), x : int32)
where reads writes(r) end
inc1:set_task_id(3)

-- Tasks in Pygion can be polymorphic (at least up to an extent), which means
-- we can actually redeclare this task multiple times with different types,
-- and it will work.
extern task inc2(r : region(ispace(int2d), fs2), x : int32)
where reads writes(r) end
inc2:set_task_id(3)

task main()
  var x = hello()
  format.println("result from Python: {}", x)
  regentlib.assert(x == 123, "test failed")

  var r = region(ispace(int1d, 10), fs1)
  fill(r.x, 1)
  inc1(r, 20)
  format.println("result in Regent: {}", r[0].x)
  regentlib.assert(r[0].x == 21, "test failed")

  var s = region(ispace(int2d, {2, 2}), fs2)
  fill(s.x, 1.5)
  inc2(s, 20)
  format.println("result in Regent: {}", s[int2d {0, 0}].x)
  regentlib.assert(s[{0, 0}].x == 21.5, "test failed")
end
regentlib.start(main)
