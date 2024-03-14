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
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This tests a compiler bug when using two identical region arguments
-- to the same task.

task taskA(s : region(ispace(int3d), int),
           r : region(ispace(int3d), int))
where
  reads writes(r), s <= r
do end

task toplevel()
  var r = region(ispace(int3d, {4, 4, 4}), int)

  var cs = ispace(int3d, {2, 2, 1})
  var rp = partition(equal, r, cs)

  __demand(__spmd)
  do
    for c in cs do
      taskA(rp[c], rp[c])
    end
  end
end
regentlib.start(toplevel)
