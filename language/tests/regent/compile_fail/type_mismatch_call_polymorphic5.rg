-- Copyright 2019 Stanford University
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
-- type_mismatch_call_polymorphic5.rg:55: incompatible types: vec2 has 2 fields but iface1 expects 1 fields
--   f(r.{f.v, f.i})
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
}

struct iface2
{
  c : iface1;
  d : int;
}

task f(x : region(iface2)) end

task g()
  var r = region(ispace(ptr, 5), fs)
  f(r.{f.v, f.i})
end
