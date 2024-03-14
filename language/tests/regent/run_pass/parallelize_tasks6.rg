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
-- [
--  ["-ll:cpu", "4"]
-- ]

-- FIXME: Breaks runtime
--        Put back this test case after fixing runtime:
--        ["-ll:cpu", "4", "-fbounds-checks", "1", "-fdebug", "1",
--         "-fparallelize-dop", "8"],

import "regent"

local c = regentlib.c

fspace fs
{
  f : double,
  g : double,
  h : double,
}

__demand(__parallel)
task init(r : region(ispace(int3d), fs))
where reads writes(r)
do
  __demand(__openmp)
  for e in r do e.f = 0.3 * (e.x + 1) + 0.3 * (e.y + 1) + 0.4 * (e.z + 1) end
  __demand(__openmp)
  for e in r do e.g = 0 end
  __demand(__openmp)
  for e in r do e.h = 0 end
end

__demand(__parallel)
task stencil1(interior : region(ispace(int3d), fs),
                     r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.g), interior <= r
do
  __demand(__openmp)
  for e in interior do
    r[e].g = 0.5 * (r[e].f +
                    r[e + {0, -2, 0}].f + r[e + {2, 0, -1}].f +
                    r[e + {0,  1, 0}].f + r[e + {2, 0,  2}].f)
  end
end

__demand(__parallel)
task stencil2(interior : region(ispace(int3d), fs),
                     r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.g), interior <= r
do
  __demand(__openmp)
  for e in interior do
    r[e].g = 0.5 * (r[e].f +
                    r[e + {1, -1, 0}].f + r[e + {0, 0, -1}].f +
                    r[e + {1,  1, 0}].f + r[e + {0, 0,  1}].f)
  end
end

__demand(__parallel)
task stencil3(r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.g)
do
  __demand(__openmp)
  for e in r do
    r[e].g = 0.5 * (r[e].f + r[(e + {1, 2, 0}) % r.bounds].f)
  end
end

task stencil1_serial(interior : region(ispace(int3d), fs),
                            r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.h), interior <= r
do
  for e in interior do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {0, -2, 0}].f + r[e + {2, 0, -1}].f +
                    r[e + {0,  1, 0}].f + r[e + {2, 0,  2}].f)
  end
end

task stencil2_serial(interior : region(ispace(int3d), fs),
                            r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.h), interior <= r
do
  for e in interior do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {1, -1, 0}].f + r[e + {0, 0, -1}].f +
                    r[e + {1,  1, 0}].f + r[e + {0, 0,  1}].f)
  end
end

task stencil3_serial(r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in r do
    r[e].h = 0.5 * (r[e].f + r[(e + {1, 2, 0}) % r.bounds].f)
  end
end

local cmath = terralib.includec("math.h")

task check(r : region(ispace(int3d), fs))
where reads(r.{g, h})
do
  for e in r do
    var abs = cmath.fabs(e.h - e.g)
    regentlib.assert(abs < 0.000001, "test failed")
  end
end

task test(size : int)
  c.srand48(12345)
  var is = ispace(int3d, {size, size, size})
  var primary_region = region(is, fs)
  var bounds = primary_region.bounds
  var coloring = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](0),
                                              rect3d { bounds.lo + {0, 2, 1},
                                                       bounds.hi - {2, 1, 2} })
  var interior_partition = partition(disjoint, primary_region, coloring, ispace(int1d, 1))
  c.legion_domain_point_coloring_destroy(coloring)
  var interior_region = interior_partition[0]

  init(primary_region)

  do
    stencil1(interior_region, primary_region)
    stencil1_serial(interior_region, primary_region)
    check(primary_region)
  end

  while true do
    for idx = 0, 2 do
      stencil3(primary_region)
      stencil3_serial(primary_region)
      check(primary_region)
    end
    break
  end

  do
    stencil2(interior_region, primary_region)
    stencil2_serial(interior_region, primary_region)
    check(primary_region)
  end
end

task toplevel()
  test(30)
end

regentlib.start(toplevel)
