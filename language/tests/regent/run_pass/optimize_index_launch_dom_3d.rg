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

task sweep(r_points : region(ispace(int3d), int),
           r_x_faces : region(ispace(int2d), int),
           r_y_faces : region(ispace(int2d), int),
           r_z_faces : region(ispace(int2d), int))
where reads writes(r_points, r_x_faces, r_y_faces, r_z_faces) do
end

task f(d : int, nx : int, ny : int, nz : int)
  -- this is degenerate and doesn't actually do anything interesting,
  -- but is sufficient for the static check
  return ispace(int3d, {1, 1, 1}, {d, d, d})
end

task main()
  var r_points = region(ispace(int3d, {10, 10, 10}), int)
  var r_x_faces = region(ispace(int2d, {10, 10}), int)
  var r_y_faces = region(ispace(int2d, {10, 10}), int)
  var r_z_faces = region(ispace(int2d, {10, 10}), int)

  var p_points = partition(equal, r_points, ispace(int3d, {4, 4, 4}))
  var p_x_faces = partition(equal, r_x_faces, ispace(int2d, {4, 4}))
  var p_y_faces = partition(equal, r_y_faces, ispace(int2d, {4, 4}))
  var p_z_faces = partition(equal, r_z_faces, ispace(int2d, {4, 4}))

  for d = 0, 4 do
     __demand(__index_launch)
    for c in f(d, 4, 4, 4) do -- f maps diagonals to subsets of NX x NY
      sweep(p_points[c], p_x_faces[{c.y, c.z}], p_y_faces[{c.x, c.z}], p_z_faces[{c.x, c.y}])
    end
  end
end
regentlib.start(main)
