-- Copyright 2018 Stanford University
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

local VALUE = 1

struct GridData {
  vel : double;
}

task dumpTile(r : region(ispace(int1d),GridData),
                    s : region(ispace(int1d),GridData))
where reads(r.vel), reads writes(s.vel), r * s do
  copy(r.vel, s.vel)
end

__demand(__inline)
task dump(colors : ispace(int1d),
          r : region(ispace(int1d),GridData),
          s : region(ispace(int1d),GridData),
          p_r : partition(disjoint, r, colors),
          p_s : partition(disjoint, s, colors))
where reads(r.vel), reads writes(s.vel), r * s do
  for c in colors do
    dumpTile(p_r[c], p_s[c])
  end
end

task checkTile(s : region(ispace(int1d),GridData))
where reads(s.vel) do
  for x in s do
    regentlib.assert(x.vel == VALUE, "test failed")
  end
end

__demand(__inline)
task check(colors : ispace(int1d),
           s : region(ispace(int1d),GridData),
           p_s : partition(disjoint, s, colors))
where reads(s.vel) do
  for c in colors do
    checkTile(p_s[c])
  end
end

task main()
  var is = ispace(int1d, 10)
  var grid = region(is, GridData)
  var grid_copy = region(is, GridData)
  var tiles = ispace(int1d, 2)
  var p_grid = partition(equal, grid, tiles)
  var p_grid_copy = partition(equal, grid_copy, tiles)
  fill(grid.vel, VALUE)
  dump(tiles, grid, grid_copy, p_grid, p_grid_copy)
  check(tiles, grid_copy, p_grid_copy)
end

regentlib.start(main)
