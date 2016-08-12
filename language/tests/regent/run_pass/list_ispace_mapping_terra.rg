-- Copyright 2016 Stanford University
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

-- Test: `list_ispace` operator on 3d structured ispace, with a Terra function
-- as mapping.

import "regent"

local c = regentlib.c

terra mapping(p : int3d, r : rect3d)
  var size = r:size()
  return p.__ptr.y + p.__ptr.x * size.__ptr.y + p.__ptr.z * size.__ptr.y * size.__ptr.x
end

task main()
  var is = ispace(int3d, { x = 2, y = 3, z = 4 })
  var l = list_ispace(is, mapping)

  var i = 0
  for p in l do
    regentlib.assert(p.x == (i / 3) % 2, "x index doesn't match")
    regentlib.assert(p.y == i % 3, "y index doesn't match.")
    regentlib.assert(p.z == i / 6, "z index doesn't match.")
    i += 1
  end
  regentlib.assert(i == 24, "list length is incorrect.")
end
regentlib.start(main)
