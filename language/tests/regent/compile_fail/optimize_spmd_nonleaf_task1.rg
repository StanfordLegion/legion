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
-- [["-ll:cpu", "4", "-fflow-spmd", "1"]]

-- fails-with:
-- optimize_spmd_nonleaf_task1.rg:44: unable to apply SPMD transformation: a task outside of a leaf loop requests privileges on a region
--     f(p[0]) -- Bad
--      ^

-- Test: SPMD currently doesn't support raw tasks with privileges

import "regent"

task f(r : region(int))
where reads writes(r) do
end

task g(r : region(int), x : int, y : int)
end

task main()
  var r = region(ispace(ptr, 5), int)
  var p = partition(equal, r, ispace(int1d, 3))
  var t = 0
  __demand(__spmd)
  while t < 2 do
    for i = 0, 2 do
      f(p[i]) -- Ok
    end
    g(r, t, 1234) -- Ok
    f(p[0]) -- Bad
    t += 1
  end
end
