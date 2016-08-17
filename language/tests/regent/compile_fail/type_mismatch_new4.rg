-- Copyright 2016 Stanford University
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
-- type_mismatch_new4.rg:26: type mismatch in argument 1: expected int32, got double
--   var p = new(ptr(double, r), 5)
--             ^

import "regent"

local c = regentlib.c

task main()
  var r = region(ispace(ptr, 5), int)
  var p = new(ptr(double, r), 5)
end
