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

-- fails-with:
-- bounds_check_fail4.rg:47: pointer ptr(int32, $r0, $r1) is out-of-bounds

import "regent"

-- This tests the compiler bounds checks for Terra regions.

local c = regentlib.c

function raw_to_ptr(t)
  return terra(x : int, y : int)
    return [t]{ __ptr = [ptr]{ __ptr = c.legion_ptr_t { value = x } }, __index = y }
  end
end

task main()
  var r = region(ispace(ptr, 3), int)
  var x = dynamic_cast(ptr(int, r), 0)

  var colors0 = c.legion_coloring_create()
  c.legion_coloring_add_point(colors0, 0, __raw(x))
  c.legion_coloring_ensure_color(colors0, 1)
  var part0 = partition(disjoint, r, colors0)
  c.legion_coloring_destroy(colors0)

  var r0 = part0[0]
  var r1 = part0[1]

  -- Tag the pointer x with the wrong region index.
  var i = __raw(x).value
  var x01 = [raw_to_ptr(ptr(int, r0, r1))](i, 2)

  @x01 = 5
end
regentlib.start(main)
