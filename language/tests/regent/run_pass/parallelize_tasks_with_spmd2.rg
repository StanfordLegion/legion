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
--  ["-ll:cpu", "3", "-fbounds-checks", "1",
--   "-fparallelize-dop", "9"]
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
task stencil1(       r : region(ispace(int2d), fs),
              interior : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g)
do
  for e in interior do
    r[e].g = 0.5 * (r[e].f +
                    r[e + {-2, 0}].f + r[e + {0, -1}].f +
                    r[e + { 1, 0}].f + r[e + {0,  2}].f)
  end
end

__demand(__parallel)
task stencil2(       r : region(ispace(int2d), fs),
              interior : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g)
do
  for e in interior do
    r[e].g = 0.5 * (r[e].f +
                    r[e + {-1, 0}].f + r[e + {0, -1}].f +
                    r[e + { 1, 0}].f + r[e + {0,  1}].f)
  end
end

__demand(__parallel)
task stencil3(r : region(ispace(int2d), fs))
where reads(r.f), reads(r.g)
do
  var sum : double = 3
  for e in r do
    sum += 0.5 * (r[e].g + r[e].f + r[(e + {2, 0}) % r.bounds].f)
  end
  return sum
end

task stencil1_serial(       r : region(ispace(int2d), fs),
                     interior : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in interior do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {-2, 0}].f + r[e + {0, -1}].f +
                    r[e + { 1, 0}].f + r[e + {0,  2}].f)
  end
end

task stencil2_serial(       r : region(ispace(int2d), fs),
                     interior : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in interior do
    r[e].h = 0.5 * (r[e].f +
                    r[e + {-1, 0}].f + r[e + {0, -1}].f +
                    r[e + { 1, 0}].f + r[e + {0,  1}].f)
  end
end

task stencil3_serial(r : region(ispace(int2d), fs))
where reads(r.f), reads(r.h)
do
  var sum : double = 3
  for e in r do
    sum += 0.5 * (r[e].h + r[e].f + r[(e + {2, 0}) % r.bounds].f)
  end
  return sum
end

__demand(__parallel)
task copy_scalar_reduction(r : region(ispace(int2d), double), v : double)
where reads writes(r)
do
  for e in r do @e = v end
end

task copy_out_result(r : region(ispace(int2d), double))
where reads(r)
do
  for e in r do return @e end
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
  init(primary_region)

  var bounds = primary_region.bounds
  var coloring = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(coloring, [int1d](0),
                                              rect2d { bounds.lo + {2, 1},
                                                       bounds.hi - {1, 2} })
  var interior_partition = partition(disjoint, primary_region, coloring, ispace(int1d, 1))
  c.legion_domain_point_coloring_destroy(coloring)
  var interior_region = interior_partition[0]
  var tmp_region = region(is, double)

  var sum : double = 0
  var cnt = 0
  var steps = 10
  while cnt < steps do
    stencil1(primary_region, interior_region)
    sum += stencil3(primary_region)
    sum += stencil3(primary_region)
    stencil2(primary_region, interior_region)
    copy_scalar_reduction(tmp_region, sum)
    cnt += 1
  end
  sum = copy_out_result(tmp_region)

  var sum_serial : double = 0
  cnt = 0
  while cnt < steps do
    stencil1_serial(primary_region, interior_region)
    sum_serial += stencil3_serial(primary_region)
    sum_serial += stencil3_serial(primary_region)
    stencil2_serial(primary_region, interior_region)
    cnt += 1
  end

  regentlib.assert(cmath.fabs(sum - sum_serial) < 0.000001, "test failed")
  check(primary_region)
end

task toplevel()
  test(400)
end

regentlib.start(toplevel)
