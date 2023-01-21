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

local c = regentlib.c

fspace k (r : region(int)) {
  s : region(int),
  x : ptr(int, s),
} where s <= r end

task f(m : region(int), n : ptr(int, m)) : int
where reads(m) do
  return @n
end

task g(a : region(int), b : k(a)) : int
where reads(a) do
  var { c = s, d = x } = b
  return f(c, d)
end

task h() : int
  var t = region(ispace(ptr, 5), int)
  var a = dynamic_cast(ptr(int, t), 0)

  var tc = c.legion_coloring_create()
  c.legion_coloring_add_point(tc, 0, __raw(a))
  var u = partition(disjoint, t, tc)
  c.legion_coloring_destroy(tc)
  var v = u[0]
  var y = dynamic_cast(ptr(int, v), a)
  @y = 7
  var z = [k(t)]{ s = v, x = y }
  return g(t, z)
end

task main()
  regentlib.assert(h() == 7, "test failed")
end
regentlib.start(main)
