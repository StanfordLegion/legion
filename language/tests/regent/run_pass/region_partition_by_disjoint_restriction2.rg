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

struct t {
  value : int,
}

task f()
  var r = region(ispace(int2d, {3, 1}), t)
  var t : transform(2, 1)
  t[{0, 0}] = 1
  t[{1, 0}] = 0
  var e : rect2d = [rect2d] { lo = [int2d] { 0, 0 }, hi = [int2d] { 0, 0 } }
  var colors = ispace(int1d, 3)
  var p = restrict(disjoint, r, t, e, colors)

  for i in colors do
    var ri = p[i]
    for x in ri do
      x.value = (1 + int(i)) * (1 + int(i))
    end
  end

  var s = 0
  for i in colors do
    var ri = p[i]
    for x in ri do
      s += x.value
    end
  end

  return s
end

task main()
  regentlib.assert(f() == 14, "test failed")
end
regentlib.start(main)
