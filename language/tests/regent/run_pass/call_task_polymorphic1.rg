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

-- runs-with:
-- [ ["-fflow", "0"] ]

-- This test showcases various forms of region projections used for
-- launching field polymorphic tasks.

import "regent"

struct vec2
{
  _x : double;
  _y : double;
}

fspace fs
{
  i : int;
  v : vec2;
}

struct iface1
{
  c : int
}

struct iface2
{
  a : double;
  b : int;
}

task t1(x : region(vec2))
where reads writes(x)
do
  for e in x do
    e._x = 12345.0
    e._y = 54321.0
  end
end

task t2(x : region(int))
where reads writes(x)
do
  for e in x do
    @e = 32123
  end
end

task t3(x : region(iface1))
where reads writes(x)
do
  for e in x do
    e.c += 12321
  end
end

task t4(x : region(iface2))
where reads writes(x)
do
  var cnt = 1
  for e in x do
    e.a += double(cnt)
    e.b += cnt * 2
    cnt += 1
  end
end

task sum(r : region(fs), p : ptr(fs, r)) : double
where reads(r) do
  return p.v._x + p.v._y + p.i
end

local names1 = terralib.newlist({"b", "a"})
local field_paths1 = terralib.newlist({"i", regentlib.field_path("v", "_y")})

task main()
  var r = region(ispace(ptr, 5), fs)
  var x = dynamic_cast(ptr(fs, r), 2)

  t1(r.{v})
  t2(r.{i})
  t3(r.{c=i})
  t4(r.{a=v._x, b=i})
  t4(r.{[names1]=[field_paths1]})
  regentlib.assert(sum(r, x) == 111128.0, "test failed")
end

regentlib.start(main)
