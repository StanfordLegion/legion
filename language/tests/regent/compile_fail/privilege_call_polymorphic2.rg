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
-- privilege_call_polymorphic2.rg:57: invalid privileges in argument 1: writes($r.{c.a=f.v.x,c.b=f.v.y,d=f.i}.c.a)
--   f(r.{c=f.{a=v.x, b=v.y}, d=f.i})
--    ^

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

struct iface1
{
  a : double;
  b : double;
}

struct iface2
{
  c : iface1;
  d : int;
}

task f(x : region(iface2))
where writes(x.c) do end

task g(r : region(fs))
where reads(r) do
  f(r.{c=f.{a=v.x, b=v.y}, d=f.i})
end
