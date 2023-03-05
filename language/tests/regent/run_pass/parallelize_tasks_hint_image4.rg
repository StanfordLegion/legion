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

fspace fs1
{
  f : double,
  g : double,
}

fspace fs2(r : region(fs1))
{
  p : ptr(fs1, r),
}

task init(r1 : region(fs1), r2 : region(fs2(r1)), size : int)
where reads writes(r1, r2)
do
  for e in r1 do
    e.f = 0.3 * (ptr(e) + 1)
    e.g = 0.3 * (ptr(e) + 1)
  end
  for e in r2 do
    e.p = unsafe_cast(ptr(fs1, r1), ptr(e))
  end
end

__demand(__parallel)
task stencil(r1 : region(fs1), r2 : region(fs2(r1)))
where reads(r2.p), reads writes(r1.f)
do
  for e in r2 do
    r1[e.p].f += 0.5
  end
end

task stencil_serial(r1 : region(fs1), r2 : region(fs2(r1)))
where reads(r2.p), reads writes(r1.g)
do
  for e in r2 do
    r1[e.p].g += 0.5
  end
end

local cmath = terralib.includec("math.h")

task check(r1 : region(fs1))
where reads(r1.{f, g})
do
  for e in r1 do
    regentlib.assert(cmath.fabs(e.f - e.g) < 0.000001, "test failed")
  end
end

task test(size : int, p : int)
  c.srand48(12345)
  var r1 = region(ispace(ptr, size), fs1)
  var r2 = region(ispace(ptr, size), fs2(wild))

  init(r1, r2, size)
  var p_r2 = partition(equal, r2, ispace(int1d, p))
  var p_r1 = partition(equal, r1, ispace(int1d, p))

  -- stencil cannot be parallelized without the hints
  __parallelize_with p_r2, p_r1, image(r1, p_r2, r2.p) <= p_r1
  do
    for i = 0, 3 do
      stencil(r1, r2)
      stencil_serial(r1, r2)
      check(r1)
    end
  end
end

task toplevel()
  test(1000, 10)
end

regentlib.start(toplevel)
