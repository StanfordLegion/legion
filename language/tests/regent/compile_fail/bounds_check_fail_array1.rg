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

-- fails-with:
-- bounds_check_fail_array1.rg:23: pointer int1d(int32, $r) is out-of-bounds

import "regent"

task main()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  r[int1d(5)] = 123 -- Out of bounds
end
regentlib.start(main)
