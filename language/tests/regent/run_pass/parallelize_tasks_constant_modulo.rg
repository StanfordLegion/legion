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
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "5"],
--  ["-ll:cpu", "4", "-fparallelize-dop", "10"]
-- ]

import "regent"

local c = regentlib.c
local SIZE = 1000

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
task stencil(r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g)
do
  for e in r do
    e.g = 0.5 * (e.f +
                 r[(e + {-2,  0}) % rect2d {int2d {0, 0}, int2d {x = [SIZE - 1], y = [SIZE - 1]}}].f +
                 r[(e - { 1,  1}) % rect2d {int2d {0, 0}, int2d {x = [SIZE - 1], y = [SIZE - 1]}}].f +
                 r[(e - {-2, -2}) % rect2d {int2d {0, 0}, int2d {x = [SIZE - 1], y = [SIZE - 1]}}].f +
                 r[(e + { 1,  0}) % rect2d {int2d {0, 0}, int2d {x = [SIZE - 1], y = [SIZE - 1]}}].f)
  end
end

task stencil_serial(r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in r do
    e.h = 0.5 * (e.f +
                 r[(e + {-2,  0}) % rect2d {{0, 0}, {[SIZE - 1], [SIZE - 1]}}].f +
                 r[(e - { 1,  1}) % rect2d {{0, 0}, {[SIZE - 1], [SIZE - 1]}}].f +
                 r[(e - {-2, -2}) % rect2d {{0, 0}, {[SIZE - 1], [SIZE - 1]}}].f +
                 r[(e + { 1,  0}) % rect2d {{0, 0}, {[SIZE - 1], [SIZE - 1]}}].f)
  end
end

local cmath = terralib.includec("math.h")

task test()
  c.srand48(12345)
  var is = ispace(int2d, {SIZE, SIZE})
  var primary_region = region(is, fs)
  init(primary_region)
  stencil(primary_region)
  stencil_serial(primary_region)
  for e in primary_region do
    regentlib.assert(cmath.fabs(e.h - e.g) < 0.000001, "test failed")
  end
end

task toplevel()
  test()
end

regentlib.start(toplevel)
