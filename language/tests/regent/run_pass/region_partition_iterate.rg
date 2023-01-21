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

local c = regentlib.c

task f() : int
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(x))
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)
  var r0 = p[0]

  -- Check that pointer shows up in child.
  @x = 100
  var s = 0
  for y in r0 do
    s += @y
  end

  -- Check that pointer shows up in parent.
  @x = 20
  for y in r do
    s += @y
  end

  -- And of course pointer itself should be accessible.
  @x = 3
  return s + @x
end

task main()
  regentlib.assert(f() == 123, "test failed")
end
regentlib.start(main)
