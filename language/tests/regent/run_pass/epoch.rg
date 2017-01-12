-- Copyright 2017 Stanford University
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

-- runs-with:
-- [["-ll:cpu", "4"]]

import "regent"

local c = regentlib.c

task g(s : region(int), z : int)
where reads(s), writes(s) do
  for y in s do
    @y += @y * z
  end
end

task k() : int
  var r = region(ispace(ptr, 5), int)
  var x = new(ptr(int, r))
  var s = region(ispace(ptr, 5), int)
  var y0 = new(ptr(int, s))
  var y1 = new(ptr(int, s))
  var y2 = new(ptr(int, s))

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(y0))
  c.legion_coloring_add_point(rc, 1, __raw(y1))
  c.legion_coloring_add_point(rc, 2, __raw(y2))
  var p = partition(disjoint, s, rc)
  c.legion_coloring_destroy(rc)

  @x = 1
  @y0 = 200
  @y1 = 30000
  @y2 = 4000000

  must_epoch
    __demand(__parallel)
    for i = 0, 3 do
      g(p[i], 20)
    end
    g(r, 10)
  end
  return @x + @y0 + @y1 + @y2
end

task main()
  regentlib.assert(k() == 84634211, "test failed")
end
regentlib.start(main)
