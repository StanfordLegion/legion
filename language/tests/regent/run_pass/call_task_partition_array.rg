-- Copyright 2016 Stanford University
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

task f(is : ispace(int2d), cs : ispace(int2d),
       s : region(is, int), t : partition(disjoint, s, cs)) : int
where reads(s) do
  var w = 0
  for c in cs do
    var sc = t[c]
    for y in sc do
      w += @y
    end
  end
  return w
end

task g() : int
  var is = ispace(int2d, { x = 4, y = 4 })
  var r = region(is, int)

  var cs = ispace(int2d, { x = 2, y = 2 })
  var p = partition(equal, r, cs)

  for i in is do
    r[i] = i.x + 10*i.y
  end

  return f(is, cs, r, p)
end

task main()
  regentlib.assert(g() == 264, "test failed")
end
regentlib.start(main)
