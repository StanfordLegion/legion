-- Copyright 2018 Stanford University
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
--  ["-ll:cpu", "4", "-fbounds-checks", "1", "-fdebug", "1"],
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
task stencil(r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.g)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r do
    e.g = 0.5 * (e.f +
                 r[(e + {-2, 0}) % r.bounds].f +
                 r[(e - {1, 1}) % r.bounds].f +
                 r[(e - {-2, -2}) % r.bounds].f +
                 r[(e + {1, 0}) % r.bounds].f)
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("parallel version: %lu us\n", ts_end - ts_start)
end

task stencil_serial(r : region(ispace(int2d), fs))
where reads(r.f), reads writes(r.h)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r do
    e.h = 0.5 * (e.f +
                 r[(e + {-2, 0}) % r.bounds].f +
                 r[(e - {1, 1}) % r.bounds].f +
                 r[(e - {-2, -2}) % r.bounds].f +
                 r[(e + {1, 0}) % r.bounds].f)
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("serial version: %lu us\n", ts_end - ts_start)
end

local cmath = terralib.includec("math.h")

task test(size : int, p : int)
  c.srand48(12345)
  var is = ispace(int2d, {size, size})
  var cs = ispace(int2d, {p, p})
  var primary_region = region(is, fs)
  var primary_partition = partition(equal, primary_region, cs)
  __parallelize_with primary_partition, cs
  do
    init(primary_region)
    stencil(primary_region)
    stencil_serial(primary_region)
  end
  __parallelize_with primary_partition, cs
  do
    init(primary_region)
    stencil(primary_region)
    stencil_serial(primary_region)
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
