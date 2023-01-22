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
-- type_mismatch_call_polymorphic25.rg:51: type mismatch: expected double for field b but got int32
--   f(r.{[names]=[field_paths]})
--    ^

import "regent"

struct vec2
{
  x : double;
  y : int;
}

struct fs
{
  z : vec2;
  w : vec2;
}

struct iface
{
  a : double;
  b : double;
}

local names = terralib.newlist({"b", "a"})
local field_paths = terralib.newlist({
  regentlib.field_path("w", "y"),
  regentlib.field_path("z", "x"),
})

task f(x : region(iface))
where reads writes(x) do end

task g()
  var r = region(ispace(ptr, 5), fs)
  f(r.{[names]=[field_paths]})
end
