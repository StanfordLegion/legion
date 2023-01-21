-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- [["-fopenmp", "1", "-ll:ocpu", "1", "-ll:othr", "4", "-ll:cpu", "0", "-ll:okindhack" ]]

import "regent"

local fabs = regentlib.fabs(double)

task red(r : region(ispace(int2d), int),
         s : region(ispace(int1d), int))
where reads(r), reduces+(s)
do
  __demand(__openmp)
  for e in r do
    s[int1d(e.x)] += @e
  end
end

__demand(__leaf)
task check(r : region(ispace(int1d), int))
where reads(r)
do
  for e in r do
    regentlib.assert(@e == 4, "test failed")
  end
end

task main()
   var is_2d = ispace(int2d, {1000, 4})
   var is_1d = ispace(int1d, 1000)
   var r = region(is_2d, int)
   var s = region(is_1d, int)

   fill(r, 1)
   fill(s, 0)

   red(r, s)
   check(s)
end

regentlib.start(main)

