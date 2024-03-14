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

-- runs-with:
-- [["-ll:ocpu", "1", "-ll:othr", "4", "-fopenmp-strict", "1"]]

-- fails-with:
-- bounds_check_fail1.rg:35: pointer int1d(int32, $r) is out-of-bounds

import "regent"

-- This tests the compiler bounds checks for regions.

task should_fail(r : region(ispace(int1d), int))
where
  reads(r)
do
  var sum = 0
  __demand(__openmp)
  for i in r do
    if int(i) < 50 then
      sum += r[i]
    else
      sum += r[i + 1]
    end
  end
end

task main()
  var r = region(ispace(int1d, 100), int)
  fill(r, 1)
  should_fail(r)
end

regentlib.start(main)
