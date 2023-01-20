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
-- [
--  ["-ll:cpu", "4", "-fbounds-checks", "1"],
--  ["-ll:cpu", "4"]
-- ]

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
task stencil(s : region(ispace(int2d), fs), r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g)
do
  for e in s do
    r[e].g = 0.5 * (r[e].f +
                    r[e + {-2, 0}].f +
                    r[e - {1, 1}].f +
                    r[e - {-2, -2}].f +
                    r[e + {1, 0}].f)
  end
end

task stencil_serial(s : region(ispace(int2d), fs), r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in s do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {-2, 0}].f +
                    r[e - {1, 1}].f +
                    r[e - {-2, -2}].f +
                    r[e + {1, 0}].f)
  end
end

local cmath = terralib.includec("math.h")

task test(size : int, p : int)
  c.srand48(12345)
  var is = ispace(int2d, {size, size})
  var primary_region = region(is, fs)
  var primary_partition = partition(equal, primary_region, ispace(int2d, {p, p}))
  var bounds = primary_region.bounds
  var coloring = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](0),
                                              rect2d { bounds.lo + {2, 2},
                                                       bounds.hi - {2, 2} })
  var interior_partition = partition(disjoint, primary_region, coloring, ispace(int1d, 1))
  c.legion_domain_point_coloring_destroy(coloring)
  var interior_region = interior_partition[0]
  __parallelize_with primary_partition
  do
    init(primary_region)
    stencil(interior_region, primary_region)
    stencil_serial(interior_region, primary_region)
  end
  for e in primary_region do
    regentlib.assert(cmath.fabs(e.h - e.g) < 0.000001, "test failed")
  end
end

task toplevel()
  test(100, 2)
  test(300, 3)
end

regentlib.start(toplevel)
