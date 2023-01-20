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

-- fails-with:
-- type_mismatch_for8.rg:25: iterator for loop expected symbol of type int2d(int32, $r), got ptr(int32, $r)
--   for [x] in r do end
--     ^

import "regent"

local r = regentlib.newsymbol(region(ispace(int2d), int), "r")
local x = regentlib.newsymbol(ptr(int, r), "x")
task f([r])
  for [x] in r do end
end
f:compile()
