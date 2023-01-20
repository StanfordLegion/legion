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

fspace k (r : region(int)) {
  s : region(int),
} where s <= r end

task f(m : region(int)) : int
where reads(m) do
  var o = 0
  for n in m do
    o += @n
  end
  return o
end

task g(a : region(int), b : k(a), c : ptr(int, a)) : int
where reads(a), writes(a) do
  var d = b.s
  var e = dynamic_cast(ptr(int, d), c)
  @e = 30
  return f(d)
end

task h() : int
  var t = region(ispace(ptr, 5), int)
  var y = dynamic_cast(ptr(int, t), 0)
  @y = 7
  var a = dynamic_cast(ptr(int, t), 1)
  @a = 1000
  var u = partition(equal, t, ispace(int1d, 1))
  var v = u[0]
  var z = [k(t)]{ s = v }
  return g(t, z, a)
end

task main()
  regentlib.assert(h() == 37, "test failed")
end
regentlib.start(main)
