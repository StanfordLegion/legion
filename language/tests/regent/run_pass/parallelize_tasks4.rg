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
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "9"],
--  ["-ll:cpu", "4", "-fparallelize-dop", "10"]
-- ]

import "regent"

local c = regentlib.c

struct vec2
{
  x : double,
  y : double,
}

fspace fs
{
  f : vec2,
  g : double,
  a : double,
  b : double,
}

__demand(__parallel)
task stencil(r : region(ispace(int2d), fs))
where reads(r.{f, g}), reads writes(r.a)
do
  __demand(__openmp)
  for e in r do
    e.a = 0.5 * (e.g +
                 r[(e + {-2,  0}) % r.bounds].f.x +
                 r[(e - { 1,  1}) % r.bounds].f.x +
                 r[(e - {-2, -2}) % r.bounds].f.y +
                 r[(e + { 1,  0}) % r.bounds].f.y +
                 r[(e + {-2,  0}) % r.bounds].g +
                 r[(e - { 1,  1}) % r.bounds].g +
                 r[(e - {-2, -2}) % r.bounds].g +
                 r[(e + { 1,  0}) % r.bounds].g)
  end
end

__demand(__parallel)
task init(r : region(ispace(int2d), fs))
where reads writes(r)
do
  __demand(__openmp)
  for e in r do
    e.f.x = 0.3 * (e.x + 1) + 0.7 * (e.y + 1)
    e.f.y = 0.4 * (e.x + 1) + 0.6 * (e.y + 1)
    e.g = 0.5 * (e.x + 1) + 0.5 * (e.y + 1)
    e.a = 0
    e.b = 0
  end
end

task stencil_serial(r : region(ispace(int2d), fs))
where reads(r.{f, g}), reads writes(r.b)
do
  for e in r do
    e.b = 0.5 * (e.g +
                 r[(e + {-2,  0}) % r.bounds].f.x +
                 r[(e - { 1,  1}) % r.bounds].f.x +
                 r[(e - {-2, -2}) % r.bounds].f.y +
                 r[(e + { 1,  0}) % r.bounds].f.y +
                 r[(e + {-2,  0}) % r.bounds].g +
                 r[(e - { 1,  1}) % r.bounds].g +
                 r[(e - {-2, -2}) % r.bounds].g +
                 r[(e + { 1,  0}) % r.bounds].g)
  end
end

local cmath = terralib.includec("math.h")

task test(size : int)
  c.srand48(12345)
  var is = ispace(int2d, {size, size})
  var primary_region = region(is, fs)
  init(primary_region)
  stencil(primary_region)
  stencil_serial(primary_region)
  for e in primary_region do
    regentlib.assert(cmath.fabs(e.a - e.b) < 0.000001, "test failed")
  end
end

task toplevel()
  test(10)
  test(1000)
end

regentlib.start(toplevel)
