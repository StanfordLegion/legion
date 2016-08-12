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

-- Test: `list_ispace` operator on 2d structured ispace, with a Task mapping.

import "regent"

local c = regentlib.c

task row_major(p : int2d, r : rect2d)
  var size = r:size()
  return p.x + p.y * size.x
end

task main()
  var is = ispace(int2d, { x = 4, y = 4 })
  var l = list_ispace(is, row_major)

  var i = 0
  for p in l do
    regentlib.assert(p.x == i % 4, "x index doesn't match.")
    regentlib.assert(p.y == i / 4, "y index doesn't match.")
    i += 1
  end
  regentlib.assert(i == 16, "list length is incorrect.")
end
regentlib.start(main)
