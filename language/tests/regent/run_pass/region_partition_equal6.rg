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

task main()
  -- Case 1. unstructured partitioned by 3D
  var r = region(ispace(ptr, 10), int)
  var p = partition(equal, r, ispace(int3d, {5, 5, 5}))

  -- Case 2. 1D partitioned by 3D
  var s = region(ispace(int1d, 10), int)
  var q = partition(equal, s, ispace(int3d, {5, 5, 5}))

  -- Case 3. 2D partitioned by 3D
  var t = region(ispace(int2d, {4, 4}), int)
  var u = partition(equal, t, ispace(int3d, {5, 5, 5}))
end
regentlib.start(main)
