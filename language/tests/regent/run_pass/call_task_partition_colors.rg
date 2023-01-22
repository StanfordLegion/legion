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

task f(s : region(int), t : partition(disjoint, s, ispace(int1d)), n : int) : int
where reads(s) do
  var w = 0
  for i = 0, n do
    var si = t[i]
    for y in si do
      w += @y
    end
  end
  return w
end

task g() : int
  var r = region(ispace(ptr, 3), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)

  var n = 3
  var cs = ispace(int1d, n)
  var p = partition(equal, r, cs)

  var r0 = p[0]
  var x0_ = dynamic_cast(ptr(int, r0), x0)
  regentlib.assert(not isnull(x0_), "test failed1")
  @x0_ = 100

  var r1 = p[1]
  var x1_ = dynamic_cast(ptr(int, r1), x1)
  regentlib.assert(not isnull(x1_), "test failed2")
  @x1_ = 20

  var r2 = p[2]
  var x2_ = dynamic_cast(ptr(int, r2), x2)
  regentlib.assert(not isnull(x2_), "test failed3")
  @x2_ = 3

  return f(r, p, n)
end

task main()
  regentlib.assert(g() == 123, "test failed4")
end
regentlib.start(main)
