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
-- [["-ll:cpu", "4", "-fbounds-checks", "1"]]

import "regent"

local c = regentlib.c

fspace fs
{
  f : double,
  g : double,
  h : double,
}

__demand(__parallel)
task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  for e in r do e.f = 0.3 * (e.x + 1) + 0.7 * (e.y + 1) end
  for e in r do e.g = 0 end
  for e in r do e.h = 0 end
end

__demand(__parallel)
task stencil1(interior : region(ispace(int2d), fs),
                    r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g), interior <= r
do
  for e in interior do
    var center = e
    var idx1 = e + {-2,  0}
    var idx2 = e + { 0, -1}
    do
      var idx3 = e + { 1,  0}
      var idx4 = e + { 0,  2}
      var v1 = r[idx1].f + r[idx2].f
      var v2 = r[idx3].f
      r[center].g = 0.5 * (r[center].f + v1 + v2 + r[idx4].f)
    end
  end
end

__demand(__parallel)
task stencil2(interior : region(ispace(int2d), fs),
                    r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g), interior <= r
do
  for e in interior do
    var center = e
    var idx1 = e + {-1,  0}
    var idx4 = e + { 0,  1}
    do
      var idx2 = e + { 0, -1}
      var idx3 = e + { 1,  0}
      var v1 = r[center].f + r[idx1].f + r[idx2].f
      var v2 = r[idx3].f + r[idx4].f
      r[center].g += 0.3 * (v1 + v2)
    end
  end
end

task stencil_serial(interior : region(ispace(int2d), fs),
                           r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h), interior <= r
do
  for e in interior do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {-2, 0}].f + r[e + {0, -1}].f +
                    r[e + { 1, 0}].f + r[e + {0,  2}].f)
    r[e].h += 0.3 * (r[e].f +
                     r[e + {-1, 0}].f + r[e + {0, -1}].f +
                     r[e + { 1, 0}].f + r[e + {0,  1}].f)
  end
end

__demand(__parallel)
task increment(r : region(ispace(int2d), fs), c : double)
where reads writes(r.f)
do
  for e in r do e.f += e.f + c end
end

local cmath = terralib.includec("math.h")

task check(r : region(ispace(int2d), fs))
where reads(r.{g, h})
do
  for e in r do
    regentlib.assert(cmath.fabs(e.h - e.g) < 0.000001, "test failed")
  end
end

task test(size : int)
  c.srand48(12345)
  var is = ispace(int2d, {size, size})
  var primary_region = region(is, fs)
  fill(primary_region.{f, g, h}, 0.0)
  var np = 2
  var bounds = primary_region.bounds
  var coloring = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](0),
                                              rect2d { bounds.lo + {2, 1},
                                                       bounds.hi - {1, 2} })
  var interior_partition =
    partition(disjoint, primary_region, coloring, ispace(int1d, 1))
  c.legion_domain_point_coloring_destroy(coloring)
  var interior_region = interior_partition[0]

  var steps = 4
  while steps > 0 do
    for idx = 0, 1 do
      stencil1(interior_region, primary_region)
      stencil2(interior_region, primary_region)
      stencil_serial(interior_region, primary_region)
      increment(primary_region, 1)
    end
    increment(primary_region, 2)
    check(primary_region)
    steps -= 1
  end
end

task toplevel()
  test(100)
end

regentlib.start(toplevel)
