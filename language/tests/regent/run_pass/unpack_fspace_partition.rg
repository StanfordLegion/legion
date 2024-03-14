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

fspace p1(a : region(int), e : ispace(int1d)) {
  b : partition(disjoint, a, e),
  c : partition(disjoint, a, e),
  d : cross_product(b, c),
}

task n1(r : region(int), s : ispace(int1d), t : p1(r, s))
where reads writes(r) do
  var { b, c, d } = t
end

task m1()
  var u = ispace(ptr, 5)
  var v = region(u, int)
  var i = ispace(int1d, 3)
  var w = partition(equal, v, i)
  var x = partition(equal, v, i)
  var y = cross_product(w, x)
  var z = [p1(v, i)]{ b = w, c = x, d = y }
end

fspace p2(a : region(ispace(int2d), int)) {
  b : partition(disjoint, a, ispace(int2d)),
  c : partition(disjoint, a, ispace(int2d)),
}

task n2(r : region(ispace(int2d), int), s : p2(r))
where reads writes(r) do
  var { b, c } = s
end

task m2()
  var u = ispace(int2d, { x = 4, y = 4 })
  var v = region(u, int)
  var w = partition(equal, v, ispace(int2d, { x = 2, y = 1 }))
  var x = partition(equal, v, ispace(int2d, { x = 1, y = 2 }))
  var z = [p2(v)]{ b = w, c = x }
end

task main()
  m1()
  m2()
end
regentlib.start(main)
