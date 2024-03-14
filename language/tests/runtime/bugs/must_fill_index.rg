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

-- runs-with:
-- [["-ll:cpu", "4"]]

-- This test examines a currently-problematic interaction between fills,
--  index launches (or more likely, projection functions), and must epoch
--  launches.  The failure mode is that an initial fill gets re-applied after
--  the index launch, overwriting the modifications made by the index launch.
-- Removal of any of the three key components (listed below as KEY [123]) will
--  yield the expected output.

import "regent"

local c = regentlib.c

task g(s : region(int), z : int)
where reads(s), writes(s) do
  for y in s do
    @y += @y * z
  end
end

task k() : int
  var s = region(ispace(ptr, 3), int)
  var y0 = dynamic_cast(ptr(int, s), 0)
  var y1 = dynamic_cast(ptr(int, s), 1)
  var y2 = dynamic_cast(ptr(int, s), 2)

  var rc = c.legion_coloring_create()
  c.legion_coloring_add_point(rc, 0, __raw(y0))
  c.legion_coloring_add_point(rc, 1, __raw(y1))
  c.legion_coloring_add_point(rc, 2, __raw(y2))
  var p = partition(disjoint, s, rc)
  c.legion_coloring_destroy(rc)

-- KEY 1: replacing this fill of the parent with fills of the subregions
--  eliminates the error
  fill(s, 11)
  --fill((p[0]), 11)
  --fill((p[1]), 11)
  --fill((p[2]), 11)

-- KEY 2: error only occurs if the index launch happens in a must epoch
  must_epoch
  --do

-- KEY 3: error only occurs with an index launch - individual task launches for
--  each subregion works fine
    __demand(__index_launch)
    -- __forbid(__index_launch)
    for i = 0, 3 do
      g(p[i], 20)
    end
  end
  for y in s do
    if(@y ~= 231) then
      c.printf("MISMATCH: [%d] = %d (!= 231)\n", __raw(y), @y)
    end
  end
  return @y0 + @y1 + @y2
end

task main()
  regentlib.assert(k() == 3*231, "test failed")
end
regentlib.start(main)
