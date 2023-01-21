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

-- runs-with:
-- [ ["-fflow", "0"] ]

-- This test showcases various forms of region projections used for
-- launching field polymorphic tasks.

import "regent"

local format = require("std/format")

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

task t1(x : region(ispace(int1d), vec2))
where reads writes(x)
do
  for e in x do
    e._x = 12345.0
    e._y = 54321.0
  end
end

task t2(x : region(ispace(int1d), int))
where reads writes(x)
do
  for e in x do
    @e = 32123
  end
end

task t3(x : region(ispace(int1d), iface1))
where reads writes(x)
do
  for e in x do
    e.c += 12321
  end
end

task t4(x : region(ispace(int1d), iface2), c : int1d)
where reads writes(x)
do
  var cnt = [int](c)
  for e in x do
    e.a += double(cnt)
    e.b += cnt * 2
    cnt += 1
  end
end

task sum(r : region(ispace(int1d), fs), p : int1d(fs, r)) : double
where reads(r) do
  return p.v._x + p.v._y + p.i
end

local names1 = terralib.newlist({"b", "a"})
local field_paths1 = terralib.newlist({"i", regentlib.field_path("v", "_y")})

task main()
  var r = region(ispace(int1d, 5), fs)
  var cs = ispace(int1d, 5, 1)
  var p = partition(equal, r, cs)

  var ds = ispace(int1d, 3)
  var s = region(ispace(int1d, 3), rect1d)
  s[0] = rect1d { 0, 2 }
  s[1] = rect1d { 1, 3 }
  s[2] = rect1d { 2, 4 }
  var q = partition(equal, s, ds)
  var u = image(r, q, s)

  var v = cross_product(u, p)

  fill(r.{i, v.{_x, _y}}, 0)

  for d in ds do
    __demand(__index_launch)
    for c in cs do
      t1(v[d][c].{v})
    end
    __demand(__index_launch)
    for c in cs do
      t2(v[d][c].{i})
    end
    __demand(__index_launch)
    for c in cs do
      t3(v[d][c].{c=i})
    end
    __demand(__index_launch)
    for c in cs do
      t4(v[d][c].{a=v._x, b=i}, c)
    end
    __demand(__index_launch)
    for c in cs do
      t4(v[d][c].{[names1]=[field_paths1]}, c)
    end

    var w = u[d]
    var x = dynamic_cast(int1d(fs, w), w.bounds.lo)
    format.println("sum == {}", sum(w, x))
    regentlib.assert(sum(u[d], x) == 111116.0 + 6.0 * int(d), "test failed")
  end
end

regentlib.start(main)
