-- Copyright 2024 Stanford University, NVIDIA Corporation
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
-- [["-ll:gpu", "1" ]]

import "regent"

__demand(__cuda)
task f(r : region(ispace(int3d), int))
where
  reads(r)
do
  var sum = 0
  for e in r do
    sum += @e
  end
  return sum
end

task toplevel()
  var r = region(ispace(int3d, {8, 8, 8}), int)
  fill(r, 2)
  var s = region(ispace(int1d, 4), rect3d)
  s[0] = rect3d { int3d { 0, 0, 0 }, int3d { 1, 1, 1 } }
  s[1] = rect3d { int3d { 2, 2, 2 }, int3d { 3, 3, 3 } }
  s[2] = rect3d { int3d { 4, 4, 4 }, int3d { 5, 5, 5 } }
  s[3] = rect3d { int3d { 6, 6, 6 }, int3d { 7, 7, 7 } }
  var s_p = partition(equal, s, ispace(int1d, 2))
  var r_p = image(r, s_p, s)
  var sum = 0
  for c in r_p.colors do
    sum += f(r_p[c])
  end
  regentlib.assert(sum == 64, "test failed")
end

regentlib.start(toplevel)
