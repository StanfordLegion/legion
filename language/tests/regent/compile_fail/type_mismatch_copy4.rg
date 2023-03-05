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
-- type_mismatch_copy4.rg:45: mismatch in number of fields between 3 and 4
--   copy(x.k.{e.{a, b}, f.a}, y.{g, l.{c, d, e.a}})
--      ^

import "regent"

struct p1 {
  a : int,
  b : int,
}

struct p2 {
  c : int,
  d : int,
  e : p1,
  f : p1,
}

struct p3 {
  g : int,
  h : int,
  i : p1,
  j : p1,
  k : p2,
  l : p2,
}

task f(x : region(p3), y : region(p3))
where reads writes(x, y) do
  copy(x.k.{e.{a, b}, f.a}, y.{g, l.{c, d, e.a}})
end
f:compile()
