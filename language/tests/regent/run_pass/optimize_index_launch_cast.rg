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

-- This tests a bug in Regent where casts aren't stripped properly.

__demand(__inline)
task is_rank1d(color : int3d)
  var proc_grid_size = int3d{2, 2, 2}
  return color.x + proc_grid_size.x*(color.y + proc_grid_size.z*color.z)
end

task f(r : region(ispace(int1d), double))
where writes(r) do
end

task main()
  var is_rank = ispace(int3d, {2, 2, 2})
  var r = region(ispace(int1d, 10), double)
  var p = partition(equal, r, ispace(int1d, 8))

  __demand(__index_launch)
  for color in is_rank do
    f(p[is_rank1d(color)])
  end
end
regentlib.start(main)
