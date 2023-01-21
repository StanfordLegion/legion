-- Copyright 2023 Stanford University, NVIDIA Corporation
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

local nan = regentlib.nan(double)
local isnan = regentlib.isnan(double)
local finite = regentlib.finite(double)

__demand(__cuda)
task red_nan(r : region(ispace(int1d), double))
where
  reads(r)
do
  var sum = 0.0
  for e in r do
    sum += @e * nan("1")
  end
  return sum
end

__demand(__cuda)
task red_finite(r : region(ispace(int1d), double))
where
  reads(r)
do
  var sum = 0.0
  for e in r do
    sum += @e * 10
  end
  return sum
end

task toplevel()
  var r = region(ispace(int1d, 1000000), double)
  fill(r, 0.1)
  for i = 0, 10 do
    var must_be_nan = red_nan(r)
    var must_be_finite = red_finite(r)
    regentlib.assert([bool](isnan(must_be_nan)), "test failed")
    regentlib.assert([bool](finite(must_be_finite)), "test failed")
  end
end

regentlib.start(toplevel)
