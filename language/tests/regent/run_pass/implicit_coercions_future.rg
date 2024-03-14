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

local fabs = regentlib.fabs(double)

task f() : double
  return 1.25
end

task g() : int
  return 200
end

task h()
  var x : double = 1 -- Promotion, cast (from int) in initialization.
  var y = x
  x = f()            -- No promotion, no cast in assignment.
  y += x
  x = 10             -- Promotion, cast (from int) in assignment.
  y += x
  x = g()            -- No promotion, cast (from future of int) in assignment.
  y += x
  x = 3000.5          -- Promotion, no cast (from double) in assignment.
  y += x
  return y
end

task main()
  regentlib.assert(fabs(h() - 3212.75) < 0.00001, "test failed")
end
regentlib.start(main)
