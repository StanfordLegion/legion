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

task f(r : region(ispace(int3d), int))
where
  reads(r)
do
  var sum = 0
  var bounds = r.bounds
  __demand(__openmp)
  for e in bounds do
    sum += r[e]
  end
  return sum
end

task toplevel()
  var r = region(ispace(int3d, {4, 4, 4}), int)
  fill(r, 2)
  var sum = f(r)
  regentlib.assert(sum == 128, "test failed")
end

regentlib.start(toplevel)
