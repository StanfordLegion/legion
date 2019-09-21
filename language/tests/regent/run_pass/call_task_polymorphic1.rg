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

-- runs-with:
-- [ ["-fflow", "0"] ]

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

struct iface
{
  a : double;
  b : int;
}

task f(x : region(vec2))
where reads writes(x)
do
  for e in x do
    e._x = 12345.0
    e._y = 54321.0
  end
end

task g(x : region(int))
where reads writes(x)
do
  for e in x do
    @e = 32123
  end
end

task h(x : region(iface))
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

task main()
  var r = region(ispace(ptr, 5), fs)
  var x = dynamic_cast(ptr(fs, r), 2)

  f(r.{v})
  g(r.{i})
  h(r.{v._x, i})
  regentlib.assert(sum(r, x) == 98798.0, "test failed")
end

regentlib.start(main)
