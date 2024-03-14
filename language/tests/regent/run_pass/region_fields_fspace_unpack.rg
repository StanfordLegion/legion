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

import "regent"

fspace innermost {
  _0 : int,
  _1 : int,
  _2 : int,
}

fspace inner {
  a : innermost,
  b : innermost,
  c : innermost,
}

fspace middle(d : region(inner)) {
  e : region(inner),
  f : ptr(inner, e),
} where e <= d end

fspace outer(g : region(inner), h : region(middle(g))) {
  i : region(middle(g)),
  j : ptr(middle(g), i),
} where i <= h end

task unpack_from_value(l : region(inner), m : region(middle(l)),
                       n : outer(l, m))
where reads writes(l, m) do
  var o = n
  o.j.f.a = { _0 = 1, _1 = 2, _2 = 3 }
  o.j.f.b._0 = 72
  o.j.f.b._1 = 72
  o.j.f.b._2 = 72
  o.j.f.c._0 = 53
  return o
end

task unpack_from_pointer(p : region(inner), q : region(middle(p)),
                         r : region(outer(p, q)), s : ptr(outer(p, q), r))
where reads writes(p, q, r) do
  var x = s.j.f.a._0
  s.j.f.a._0 = 101
  s.j.f.b._0 = x + 200
  s.j.f.c._0 = 306
  s.j.f.c._1 = 306
  s.j.f.c._2 = 306
end

task main()
  var t = region(ispace(ptr, 5), inner)
  var u = region(ispace(ptr, 5), middle(t))
  var v = region(ispace(ptr, 5), outer(t, u))
  var w = dynamic_cast(ptr(inner, t), 0)
  var x = dynamic_cast(ptr(middle(t), u), 0)
  var y = dynamic_cast(ptr(outer(t, u), v), 0)

  w.a._0 = 0
  w.a._1 = 0
  w.a._2 = 0
  w.b._0 = 0
  w.b._1 = 0
  w.b._2 = 0
  w.c._0 = 0
  w.c._1 = 0
  w.c._2 = 0
  @x = [middle(t)] { e = t, f = w }
  @y = [outer(t, u)] { i = u, j = x }

  unpack_from_value(t, u, @y)
  regentlib.assert(w.a._0 == 1 and w.a._1 == 2 and w.a._2 == 3, "test failed")
  regentlib.assert(w.b._0 == 72 and w.b._1 == 72 and w.b._2 == 72, "test failed")
  regentlib.assert(w.c._0 == 53, "test failed")

  unpack_from_pointer(t, u, v, y)
  regentlib.assert(w.a._0 == 101, "test failed")
  regentlib.assert(w.b._0 == 201, "test failed")
  regentlib.assert(w.c._0 == 306 and w.c._1 == 306 and w.c._2 == 306, "test failed")
end
regentlib.start(main)
