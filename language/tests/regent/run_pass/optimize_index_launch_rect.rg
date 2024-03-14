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
-- [["-fflow", "0"]]

-- FIXME: This test is broken by RDIR.

import "regent"

-- This tests a fix for bug #558.

local C = regentlib.c

local Np = 10

local Nx = 2
local Ny = 2
local Nz = 2

struct Fields {
   y : double,
}

task doStuff(r : region(ispace(int3d),Fields))
where reads writes(r)
do
end

task doOtherStuff(r : region(ispace(int3d),Fields),
                  s : region(ispace(int2d),Fields))
where reads writes(r), reads(s)
do
end

task main()
  var r = region(ispace(int3d,{Np,Np,Np}),Fields)
  var s = region(ispace(int2d,{Np,Np}),Fields)

  var rect = rect3d({ {0,0,0}, {Nx-1,Ny-1,Nz-1} })
  var cr = ispace(int3d, rect:size(), rect.lo)
  var pr = partition(equal, r, cr)

  var cs = ispace(int2d,{Nx,Ny})
  var ps = partition(equal, s, cs)

  fill(r.y, 0.0)
  fill(s.y, 0.0)

  __demand(__index_launch)
  for c in rect do
    doStuff(pr[c])
  end

  __demand(__index_launch)
  for c in rect do
    doOtherStuff(pr[c], ps[int2d{c.x,c.y}]) -- Note: This cast was causing the issue.
  end
end
regentlib.start(main)
