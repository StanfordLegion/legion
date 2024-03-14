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
-- [["-ll:gpu", "1" ]]

import "regent"

fspace fs
{
  v : double,
  v_cpu : double,
  v_cpu_f : float,
  v_gpu : double,
  v_gpu_f : float,
}

local fabs = regentlib.fabs(double)
local rsqrt = regentlib.sqrt(double)
local rsqrtf = regentlib.sqrt(float)

task init(r : region(ispace(int1d), fs))
where
  reads writes(r.v)
do
  for e in r do
    e.v = [double]([int](e))
  end
end

task f(r : region(ispace(int1d), fs))
where
  reads(r.v), reads writes(r.{v_cpu, v_cpu_f})
do
  for e in r do
    e.v_cpu = rsqrt(e.v + 10)
    e.v_cpu_f = rsqrtf(e.v + 10)
  end
end

__demand(__cuda)
task g(r : region(ispace(int1d), fs))
where
  reads(r.v), reads writes(r.{v_gpu, v_gpu_f})
do
  for e in r do
    e.v_gpu = rsqrt(e.v + 10)
    e.v_gpu_f = rsqrtf(e.v + 10)
  end
end

task toplevel()
  var n = 100
  var r = region(ispace(int1d, n), fs)
  init(r)
  f(r)
  g(r)

  for e in r do
    regentlib.assert(fabs(e.v_cpu - e.v_gpu) < 1.0e-6, "test failed")
    regentlib.assert(fabs(e.v_cpu_f - e.v_gpu_f) < 1.0e-6, "test failed")
  end
end

regentlib.start(toplevel)
