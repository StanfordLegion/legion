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
--   ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "9"],
--   ["-ll:cpu", "4", "-fparallelize-dop", "4"]
-- ]

import "regent"

local c = regentlib.c

__demand(__parallel)
task init(r : region(ispace(int1d), double))
where reads writes(r)
do
  for e in r do @e = 0.7 * (e + 1) end
end

__demand(__parallel)
task stencil(r : region(ispace(int1d), double))
where reads(r)
do
  var sum : double = 0.03
  __demand(__openmp)
  for e in r do
    sum += 0.5 * (@e +
                  r[(e - 1) % r.bounds] +
                  r[(e + 1) % r.bounds])
  end
  return sum
end

task stencil_serial(r : region(ispace(int1d), double))
where reads(r)
do
  var sum : double = 0.03
  for e in r do
    sum += 0.5 * (@e +
                  r[(e - 1) % r.bounds] +
                  r[(e + 1) % r.bounds])
  end
  return sum
end

local cmath = terralib.includec("math.h")

terra wait_for(x : double) return 1 end

task test(size : int)
  c.srand48(12345)
  var is = ispace(int1d, size)
  var primary_region = region(is, double)
  init(primary_region)
  var result1 = stencil(primary_region)
  do
    result1 = stencil(primary_region)
    for idx = 0, 1 do
      result1 -= stencil(primary_region)
      result1 += stencil(primary_region)
    end
  end
  wait_for(result1)
  var result2 : double = 0
  result2 = stencil_serial(primary_region)
  wait_for(result2)
  regentlib.assert(cmath.fabs(result1 - result2) < 1e-3, "test failed")
  return 1
end

task toplevel()
  wait_for(test(50000))
end

regentlib.start(toplevel)
