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
-- invalid_inline_task5.rg:42: ambiguous field access in __fields: every field path in an inline task must be fully specified.
--   var a = __fields(t.{A})
--                       ^

import "regent"

local c = regentlib.c

struct Vec
{
  x : double,
  y : double,
}

fspace Fields
{
  A : Vec,
  B : Vec,
  C : Vec
}

__demand(__inline)
task test(t : region(ispace(int1d), Fields))
where
  reads writes(t)
do
  var a = __fields(t.{A})
end
