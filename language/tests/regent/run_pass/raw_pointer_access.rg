-- Copyright 2018 Stanford University
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
  var x : &int = [&int](regentlib.c.malloc([terralib.sizeof(int)]))
  regentlib.assert(x ~= [&int](0), "malloc failed")
  x[0] = 123
  regentlib.c.printf("x: %d\n", x[0])
  regentlib.c.free(x)
end
regentlib.start(main)
