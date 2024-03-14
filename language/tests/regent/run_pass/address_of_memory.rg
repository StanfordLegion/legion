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
  var y : int64 = 456
  regentlib.assert(not isnull(&y), "test failed")
  regentlib.assert(y == @&y, "test failed")

  var x = [&int64](regentlib.c.malloc([terralib.sizeof(int64) * 2]))
  regentlib.assert(not isnull(&x), "test failed")
  regentlib.assert(not isnull(x), "test failed")
  x[0] = 123
  x[1] = 789
  regentlib.assert(x == &x[0], "test failed")
  regentlib.assert(&x[0] ~= &x[1], "test failed")
  regentlib.assert(x[0] == @&x[0], "test failed")
  regentlib.assert(x[0] ~= @&x[1], "test failed")
end
regentlib.start(main)
