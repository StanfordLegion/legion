-- Copyright 2019 Stanford University
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

-- fails-with:
-- optimize_index_launch_noninterference2.rg:83: loop optimization failed: argument 1 interferes with itself
--      func3(pr[int3d{c.x + c.y, 0, c.z}])
--           ^

import "regent"

local C = regentlib.c

local Np = 10

local Nx = 4
local Ny = 4
local Nz = 4

struct Fields {
   y : double,
}

task func1(t : region(ispace(int1d),Fields))
where
   reads writes(t)
do
end

task func2(s : region(ispace(int2d),Fields))
where
   reads writes(s)
do
end

task func3(r : region(ispace(int3d),Fields))
where
   reads writes(r)
do
end

task func5(r : region(ispace(int3d),Fields),
           s : region(ispace(int2d),Fields))
where
   reads writes(r, s)
do
end

task main()
   var r = region(ispace(int3d,{Np,Np,Np}),Fields)
   var s = region(ispace(int2d,{Np,Np}),Fields)
   var t = region(ispace(int1d,{Np}),Fields)

   var cr = ispace(int3d,{Nx,Ny,Nz})
   var pr = partition(equal, r, cr)

   var cs = ispace(int2d,{Nx,Ny})
   var ps = partition(equal, s, cs)

   var ct = ispace(int1d,{Nx})
   var pt = partition(equal, t, ct)

   fill(r.y, 0.0)
   fill(s.y, 0.0)


   -- projecting down dimensions
   __demand(__parallel)
   for c in cr do
      func3(pr[int3d{c.x + c.y, 0, c.z}])
   end
end
regentlib.start(main)
