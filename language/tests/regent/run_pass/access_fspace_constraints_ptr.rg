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

task g(a : region(int), b : k(a)) : int
where reads(a) do
  return @(b.x)
end

task h() : int
  var t = region(ispace(ptr, 5), int)
  var y = dynamic_cast(ptr(int, t), 0)
  @y = 7
  var u = partition(equal, t, ispace(int1d, 1))
  var v = u[0]
  var w = dynamic_cast(ptr(int, v), y)
  var z = [k(t)]{ s = v, x = w }
  return g(t, z)
end

task main()
  regentlib.assert(h() == 7, "test failed")
end
regentlib.start(main)
