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

-- runs-with:
-- []

-- FIXME:
-- [["-ll:cpu", "4"]]

-- This test examines a problem with index launches that have simultaneous
--  access to the same regions.  Fills that occur before the index launch
--  seem to get repeated between the steps of the index launch, overwriting
--  data written by some of the index tasks.  Worse, the dependencies occur
--  even with a must epoch, and would cause deadlock if there were any
--  synchronization events in this example.

import "regent"

local c = regentlib.c

task h(s0 : region(int), s1 : region(int), s2 : region(int), z : int, i : int)
where reads writes simultaneous(s0), reads writes simultaneous(s1), reads writes simultaneous(s2) do
  for y in s0 do
    -- these tests make sure each index task's writes are disjoint
    if((__raw(y).value % 3) == i) then @y += @y * z end
  end
  for y in s1 do
    if((__raw(y).value % 3) == i) then @y += @y * z end
  end
  for y in s2 do
    if((__raw(y).value % 3) == i) then @y += @y * z end
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

-- this error occurs even when the fills are done on the subregions
  --fill(s, 11)
  fill((p[0]), 11)
  fill((p[1]), 11)
  fill((p[2]), 11)

-- error occurs whether or not index launch is in a must epoch, although only
--  the must epoch case would be vulnerable to a deadlock as a result
  must_epoch
  --do

-- KEY 1: error only occurs with an index launch - individual task launches
--  work fine
    __demand(__index_launch)
    -- __forbid(__index_launch)
    for i = 0, 3 do
      h(p[0], p[1], p[2], 20, i)
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
