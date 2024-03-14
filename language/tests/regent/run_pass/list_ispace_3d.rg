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

-- Test: `list_ispace` operator on 3d structured ispace.

import "regent"

local c = regentlib.c

task main()
  var is = ispace(int3d, { x = 5, y = 2, z = 3 })
  var r = region(is, int)

  var l = list_ispace(is)

  -- ForList loop through list.
  do
    var i = 0
    for p in l do
      regentlib.assert(p.x == i % 5, "x index doesn't match in ForList loop.")
      regentlib.assert(p.y == (i / 5) % 2, "y index doesn't match in ForList loop.")
      regentlib.assert(p.z == i / 5 / 2, "z index doesn't match in ForList loop.")
      i += 1
    end
    regentlib.assert(i == 30, "list length is incorrect.")
  end

  -- ForNum loop through list.
  do
    for i = 0, 30 do
      var p = l[i]
      regentlib.assert(p.x == i % 5, "x index doesn't match in ForNum loop.")
      regentlib.assert(p.y == (i / 5) % 2, "y index doesn't match in ForNum loop.")
      regentlib.assert(p.z == i / 5 / 2, "z index doesn't match in ForNum loop.")
    end
  end


  -- Verify that elements of `l` can be used to index into region.
  r[l[5]] = 42
  regentlib.assert(r[l[5]] == 42, "assignment into region failed.")
end
regentlib.start(main)
