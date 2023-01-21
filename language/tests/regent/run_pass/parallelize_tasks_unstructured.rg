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
-- [["-ll:cpu", "4"],
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "2,2"],
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fparallelize-dop", "4,2,2"]]

import "regent"

local c = regentlib.c

fspace fs
{
  f : double,
  g : double,
}

__demand(__parallel)
task init(r : region(fs))
where reads writes(r)
do
  __demand(__openmp)
  for e in r do e.f = 0 end
  __demand(__openmp)
  for e in r do e.g = 0 end
end

__demand(__parallel)
task increment(r : region(fs), c : double)
where reads writes(r.f)
do
  __demand(__openmp)
  for e in r do e.f += e.f + c end
end

__demand(__parallel)
task reduce(r : region(fs))
where reads writes(r.f)
do
  var sum = 0
  __demand(__openmp)
  for e in r do sum += e.f end
  return sum
end

task increment_serial(r : region(fs), c : double)
where reads writes(r.g)
do
  for e in r do e.g += e.g + c end
end

task reduce_serial(r : region(fs))
where reads writes(r.f)
do
  var sum = 0
  for e in r do sum += e.f end
  return sum
end

local cmath = terralib.includec("math.h")

task check(r : region(fs))
where reads(r.{f, g})
do
  for e in r do
    regentlib.assert(cmath.fabs(e.f - e.g) < 0.000001, "test failed")
  end
end

task test(size : int)
  c.srand48(12345)
  var is = ispace(ptr, size)
  var primary_region = region(is, fs)

  init(primary_region)
  for i = 0, 5 do
    increment(primary_region, i + 1.5)
    increment_serial(primary_region, i + 1.5)
    check(primary_region)
  end

  var sum1 = reduce(primary_region)
  var sum2 = reduce_serial(primary_region)
  regentlib.assert(cmath.fabs(sum1 - sum2) < 1e-3, "test failed")
end

task toplevel()
  test(1000)
end

regentlib.start(toplevel)
