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
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "2,2"],
--  ["-ll:cpu", "4", "-fparallelize-dop", "2,5"],
--  ["-ll:cpu", "4", "-fparallelize-dop", "5,5"]
-- ]

import "regent"

local c = regentlib.c

fspace fs1
{
  f : double,
}

fspace fs2(r : region(ispace(int2d), fs1))
{
  p : int2d(fs1, r),
  g : double,
  h : double,
}

task init(r1 : region(ispace(int2d), fs1), r2 : region(fs2(r1)), size : int)
where reads writes(r1, r2)
do
  for e in r1 do e.f = 0.3 * (e.x + 1) + 0.7 * (e.y + 1) end
  for e in r2 do
    e.p = unsafe_cast(int2d(fs1, r1), int2d { __raw(e).value / size,
                                              __raw(e).value % size })
    e.g = 0
    e.h = 0
  end
end

__demand(__parallel)
task stencil(r1 : region(ispace(int2d), fs1), r2 : region(fs2(r1)))
where reads(r1.f, r2.p), reads writes(r2.g)
do
  for e in r2 do
    e.g += 0.5 * r1[e.p].f * e.p.f
  end
end

task stencil_serial(r1 : region(ispace(int2d), fs1), r2 : region(fs2(r1)))
where reads(r1.f, r2.p), reads writes(r2.h)
do
  for e in r2 do
    e.h += 0.5 * e.p.f * e.p.f
  end
end

local cmath = terralib.includec("math.h")

task check(r1 : region(ispace(int2d), fs1), r2 : region(fs2(r1)))
where reads(r2.{g, h})
do
  for e in r2 do
    regentlib.assert(cmath.fabs(e.h - e.g) < 0.000001, "test failed")
  end
end

task test(size : int)
  c.srand48(12345)
  var r1 = region(ispace(int2d, {size, size}), fs1)
  var r2 = region(ispace(ptr, size * size), fs2(wild))
  init(r1, r2, size)
  for i = 0, 3 do
    stencil(r1, r2)
    stencil_serial(r1, r2)
    check(r1, r2)
  end
end

task toplevel()
  test(100)
end

regentlib.start(toplevel)
