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

task f0(is : ispace(ptr), i : ptr)
end

task f1(is : ispace(int1d), i : int1d)
end

task f2(is : ispace(int2d), i : int2d)
end

task main()
  var is0 = ispace(ptr, 5)
  var is1 = ispace(int1d, 7, 2)
  var is2 = ispace(int2d, { x = 2, y = 2 }, { x = 3, y = 4 })

  var i0 : ptr = 3
  f0(is0, i0)
  f0(is0, 4)

  var i1 : int1d = 14
  f1(is1, i1)
  f1(is1, 3)

  var s1 = 0
  for j1 in is1 do
    s1 += j1
  end
  regentlib.assert(s1 == 35, "test failed")

  var i2 : int2d = { x = 4, y = 5 }
  f2(is2, i2)
  f2(is2, int2d { x = 3, y = 5 })

  var s2 = int2d { x = 0, y = 0 }
  for j2 in is2 do
    s2 += j2
  end
  regentlib.assert(s2.x == 14 and s2.y == 18, "test failed")

end
regentlib.start(main)
