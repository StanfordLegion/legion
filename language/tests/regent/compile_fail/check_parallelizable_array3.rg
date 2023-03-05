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
-- check_parallelizable_array3.rg:29: vectorization failed: found a loop-carried dependence
--     s += a[ [int](e) ]
--           ^

import "regent"

task f(r : region(ispace(int1d), int))
where reads writes(r) do
  var a : int[10]
  var s = 0
  __demand(__vectorize)
  for e in r do
    a[([int](e) + 1) % 10] += 1
    s += a[ [int](e) ]
  end
end
