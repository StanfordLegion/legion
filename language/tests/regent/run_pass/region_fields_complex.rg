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

import "regent"

local c = regentlib.c

-- This tests that complex values can be used as fields in regions.

task f()
  var r = region(ispace(int1d, 5), complex)

  var sum : complex = complex { 0.0, 0.0 }
  for i = 0, 5 do
    r[i].real = double(i + 1)
    r[i].imag = double(i + 2)
  end
  for e in r do
    sum += @e
  end

  regentlib.assert(sum.real == 15.0, "test failed")
  regentlib.assert(sum.imag == 20.0, "test failed")
end
regentlib.start(f)
