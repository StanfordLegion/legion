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
-- privilege_call_polymorphic3.rg:55: invalid privileges in argument 1: writes($p[int1d($c)].{b=v._x,a=v._y}.b)
--     init_double(p[c].{b=v._x, a=v._y})
--                ^

import "regent"

struct vec2
{
  _x : double;
  _y : double;
}

fspace fs
{
  v : vec2;
  i : int;
}

struct iface
{
  a : double;
  b : double;
}

task init_double(x : region(ispace(int1d), iface))
where reads writes(x)
do
  for e in x do
    e.a = 12345.0
    e.b = 54321.0
  end
end

task test(r : region(ispace(int1d), fs),
          p : partition(disjoint, r, ispace(int1d)))
where
  reads(r)
do
  for c in p.colors do
    init_double(p[c].{b=v._x, a=v._y})
  end
end
