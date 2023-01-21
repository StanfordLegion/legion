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
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  regentlib.assert(not isnull(x), "test failed")
  r[0] = 123
  r[1] = 456
  regentlib.assert(x == &r[0], "test failed")
  regentlib.assert(x ~= &r[1], "test failed")
  regentlib.assert(@&r[0] == 123, "test failed")
  regentlib.assert(@&r[1] == 456, "test failed")
end
regentlib.start(main)
