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
  __demand(__openmp)
  for e1 in r do
    for e2 in r do
      sum += @e2
    end
  end
  return sum
end

task toplevel()
  var r = region(ispace(int3d, {4, 4, 4}), int)
  fill(r, 2)
  var s = region(ispace(int1d, 2), rect3d)
  s[0] = rect3d { int3d { 0, 0, 0 }, int3d { 1, 1, 1 } }
  s[1] = rect3d { int3d { 2, 2, 2 }, int3d { 3, 3, 3 } }
  var s_p = partition(equal, s, ispace(int1d, 1))
  var r_p = image(r, s_p, s)
  var sum = f(r_p[0])
  regentlib.assert(sum == 512, "test failed")
end

regentlib.start(toplevel)
