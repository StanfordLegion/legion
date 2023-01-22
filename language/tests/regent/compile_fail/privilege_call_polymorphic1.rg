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
--privilege_call_polymorphic1.rg:50: invalid privileges in argument 1: reads($r.{b=f.v.y,a=f.v.x}.b)
--  f(r.{b=f.v.y, a=f.v.x})
--   ^

import "regent"

struct vec2
{
  x : double;
  y : double;
}

struct st
{
  v : vec2;
  i : int;
  l : long;
}

fspace fs
{
  f : st;
}

struct iface
{
  a : double;
  b : double;
}

task f(s : region(iface))
where reads writes(s.a), writes(s.b) do end

task g(r : region(fs))
  f(r.{b=f.v.y, a=f.v.x})
end
