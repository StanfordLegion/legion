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

import "regent"

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


  -- simple multi-dimensional index ctors
  __demand(__index_launch)
  for c in cr do
    func3(pr[int3d{c.x, c.y, c.z}])
  end

  -- simple multi-dimensional index ctors
  __demand(__index_launch)
  for i in cs do
    func5(pr[int3d{i.x, i.y, 0}], ps[int2d{0,0} + i])
  end

  __demand(__index_launch)
  for i in cs do
   func2(ps[int2d{0,0} + i])
  end

  __demand(__index_launch)
  for a in ct do
    func1(pt[a + 0])
  end

  __demand(__index_launch)
  for a in ct do
    func1(pt[0 + a])
  end

  __demand(__index_launch)
  for B in ct do
    var temp = 0
    func1(pt[B + 0])
  end

  __demand(__index_launch)
  for a in ct do
    func1(pt[2 + 1 * a - 2])
  end

  __demand(__index_launch)
  for i in cr do
    var temp = int3d{0,0,0}
    func3(pr[{0,0,0} + i + {0,0,0}])
  end

  -- nested loop free variable
  for i = 0, 2 do
    __demand(__index_launch)
    for c in ct do
      func2(ps[int2d{c, i}])
    end
  end

  var zero = 0
  __demand(__index_launch)
  for c in ct do
    func1(pt[zero + c])
  end

  for i = 0, 4 do
    __demand(__index_launch)
    for c in ct do
      var temp = c - i
      func1(pt[temp + i])
    end
  end

  -- constant * index
  __demand(__index_launch)
  for i = 0,2 do
    var temp = 1
    func1(pt[2 * i + temp])
  end



  -- not optimized: projecting down dimension
  for c in cr do
    func3(pr[int3d{(c.x + c.y) % Nx, 0, c.z}])
  end

  -- not optimized: cancelling index field
  for c in cs do
    var temp = int2d{0, c.y}
    func2(ps[c - temp])
  end

  -- not optimized: cancelling index field (mult by 0)
  for c in cs do
    var temp = 0
    func2(ps[int2d{c.x, temp * c.y}])
  end

  -- not optimized: cancelling coefficients
  for c in ct do
    func1(pt[zero * c])
  end

  -- not optimized: cancelling coefficients
  for i = 0,4 do
    var temp = 2 * i
    func1(pt[ temp - 2 * i + 1])
  end
end
regentlib.start(main)
