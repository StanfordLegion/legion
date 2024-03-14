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

local vec4 = vector(double, 4)

-- For this test, it is important that the tasks actually pass vectors
-- by value. In certain cases, malloc does not return blocks of memory
-- aligned to appropriate SSE boundaries, so this test ensures that
-- the compiler emits correct code for those cases.

task f(x : vec4, y : vec4) : vec4
  var z = x + y
  var w = 1000 * x
  return (-z) + w
end

task g() : double
  var x = vector(1.0, 20.0, 300.0, 4000.0)
  var y = vector(0.5, 6.0, 70.0, 800.0)
  var z = f(x, y)
  return z[0] + z[1] + z[2] + z[3]
end

task main()
  regentlib.assert(g() == 4315802.5, "test failed")
end
regentlib.start(main)
