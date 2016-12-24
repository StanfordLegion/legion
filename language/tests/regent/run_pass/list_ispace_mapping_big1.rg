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

-- Test: `list_ispace` operator on 2d structured ispace, where the mapping
-- function returns large, signed 64-bit values.

import "regent"

local c = regentlib.c

terra mapping(p : int2d, r : rect2d) : int64
  var size = r:size()
  var offset : int64 = p.__ptr.y + p.__ptr.x * size.__ptr.y
  if offset % 2 == 0 then
    return -(offset << 35)
  else
    return offset << 35
  end
end

task main()
  var is = ispace(int2d, { x = 2, y = 3 })
  var l = list_ispace(is, mapping)

  regentlib.assert(l[0] == int2d { x = 1, y = 1 }, "element 0 doesn't match")
  regentlib.assert(l[1] == int2d { x = 0, y = 2 }, "element 1 doesn't match")
  regentlib.assert(l[2] == int2d { x = 0, y = 0 }, "element 2 doesn't match")
  regentlib.assert(l[3] == int2d { x = 0, y = 1 }, "element 3 doesn't match")
  regentlib.assert(l[4] == int2d { x = 1, y = 0 }, "element 4 doesn't match")
  regentlib.assert(l[5] == int2d { x = 1, y = 2 }, "element 5 doesn't match")
end
regentlib.start(main)
