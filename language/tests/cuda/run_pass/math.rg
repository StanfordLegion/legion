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
-- [["-ll:gpu", "1" ]]

import "regent"

fspace fs
{
  v : double,
  v_cpu : double,
  v_cpu_vec : double,
  v_gpu : double,
}

local ceil = regentlib.ceil(double)
local log2 = regentlib.log2(double)
local log10 = regentlib.log10(double)
local log = regentlib.log(double)
local cos = regentlib.cos(double)
local sin = regentlib.sin(double)
local exp2 = regentlib.exp2(double)
local fabs = regentlib.fabs(double)
local trunc = regentlib.trunc(double)
local floor = regentlib.floor(double)
local sqrt = regentlib.sqrt(double)
local exp = regentlib.exp(double)

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
  reads(r.v),
  reads writes(r.{v_cpu, v_cpu_vec})
do
  __forbid(__vectorize)
  for e in r do
    e.v_cpu = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(e.v + 10))))))))))))
  end
  __demand(__vectorize)
  for e in r do
    e.v_cpu_vec = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(e.v + 10))))))))))))
  end
end

__demand(__cuda)
task g(r : region(ispace(int1d), fs))
where
  reads(r.v),
  reads writes(r.v_gpu)
do
  for e in r do
    e.v_gpu = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(e.v + 10))))))))))))
  end
end

task toplevel()
  var n = 100
  var r = region(ispace(int1d, n), fs)
  init(r)
  f(r)
  g(r)

  for e in r do
    regentlib.assert(fabs(e.v_cpu - e.v_cpu_vec) < 1.0e-6, "test failed")
    regentlib.assert(fabs(e.v_cpu - e.v_gpu) < 1.0e-6, "test failed")
  end
end

regentlib.start(toplevel)
