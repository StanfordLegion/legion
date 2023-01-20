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

-- Test: `list_ispace` operator on unstructured ispace.

import "regent"

local c = regentlib.c

task main()
  var is = ispace(ptr, 4)
  var r = region(is, int)

  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)

  var l = list_ispace(is)

  -- ForList loop through list.
  do
    fill(r, 0)

    var i = 0
    for p in l do
      r[p] = 42    -- I don't assume order of iteration here.
      i += 1
    end
    regentlib.assert(i == 4, "list length is incorrect.")
    regentlib.assert(@x0 == 42, "assignment failed in ForList loop: @x0.")
    regentlib.assert(@x1 == 42, "assignment failed in ForList loop: @x1.")
    regentlib.assert(@x2 == 42, "assignment failed in ForList loop: @x2.")
    regentlib.assert(@x3 == 42, "assignment failed in ForList loop: @x3.")
  end

  -- ForNum loop through list.
  do
    fill(r, 0)
    for i = 0, 4 do
      var p = l[i]
      r[p] = 7
    end
    regentlib.assert(@x0 == 7, "assignment failed in ForNum loop: @x0.")
    regentlib.assert(@x1 == 7, "assignment failed in ForNum loop: @x1.")
    regentlib.assert(@x2 == 7, "assignment failed in ForNum loop: @x2.")
    regentlib.assert(@x3 == 7, "assignment failed in ForNum loop: @x3.")
  end
end
regentlib.start(main)
