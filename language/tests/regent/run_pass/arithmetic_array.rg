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

task f(x : int[3], y : int[3]) : int[3]
  var z : int[3]
  var w : int[3]
  z = x + y
  w = x * y
  z = z / array(2, 2, 2)
  return -(z + (-w))
end


task g(x : int[2], y : int[2]) : int[2]
  return x % y
end

task main()
  regentlib.assert(f(array(3, 3, 3), array(4, 4, 4))[1] == 9, "test failed")
  regentlib.assert(g(array(2, 2), array(4, 4))[0] == 2, "test failed")
  regentlib.assert(g(array(128, 128), array(7, 7))[1] == 2, "test failed")
end
regentlib.start(main)
