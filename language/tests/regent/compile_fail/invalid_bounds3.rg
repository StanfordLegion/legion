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

-- fails-with:
-- invalid_bounds3.rg:28: type mismatch: write expected an lvalue but got rect1d
--   r.bounds = new_bounds
--    ^

import "regent"

local c = regentlib.c

task main()
  var is = ispace(int1d, 10)
  var r = region(is, int)
  var new_bounds = rect1d { lo = 0, hi = 10 }
  r.bounds = new_bounds
end
