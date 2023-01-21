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
-- type_mismatch_call_polymorphic17.rg:36: type mismatch in argument 1: expected region(int32) but got region({a : double})
--   f(r.{[name]=[field]})
--    ^

import "regent"

struct vec2
{
  x : double;
  y : double;
}

local name = "a"
local field = "x"

task f(x : region(int))
where reads writes(x) do end

task g()
  var r = region(ispace(ptr, 5), vec2)
  f(r.{[name]=[field]})
end
