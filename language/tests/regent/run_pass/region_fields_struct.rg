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

task f(r : region(p1), x : ptr(p1, r)) : int
where reads(r.{a, b}) do
  return x.a + x.b
end

task tf() : int
  var r = region(ispace(ptr, 1), p1)
  var x = dynamic_cast(ptr(p1, r), 0)
  x.a = 1
  x.b = 20
  return f(r, x)
end

task g(r : region(p2), x : ptr(p2, r))
where
  reads(r.{}), writes(r.e.a, r.f)
do
  x.e.a = 1
  x.f.a = 20
  x.f.b = 300
end

task tg()
  var r = region(ispace(ptr, 1), p2)
  var x = dynamic_cast(ptr(p2, r), 0)
  g(r, x)
  return x.e.a + x.f.a + x.f.b
end

task z(r : region(p3))
where
  reads(r.{i.a, j.b, k.{c, e.b}, l.{e.{a, b}, d, c, f.{}}})
do
end

task main()
  regentlib.assert(tf() == 21, "test failed")
  regentlib.assert(tg() == 321, "test failed")
end
regentlib.start(main)
