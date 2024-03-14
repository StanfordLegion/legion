-- Copyright 2024 Stanford University, NVIDIA Corporation
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

fspace a {
  x1 : int32,
  x2 : int32,
  y1 : int64,
  y2 : int64,
  z1 : float,
  z2 : float,
  w1 : double,
  w2 : double,
}

task f(r : region(a))
where reduces+(r) do
  for e in r do
    e.x1 += 1
    e.x2 += 2
    e.y1 += 3
    e.y2 += 4
    e.z1 += 5
    e.z2 += 6
    e.w1 += 7
    e.w2 += 8
  end
end

task main()
  var r = region(ispace(ptr, 1), a)
  fill(r.{x1, x2, y1, y2, z1, z2, w1, w2}, 0)
  f(r)
  for e in r do
    regentlib.assert(e.x1 == 1, "test failed")
    regentlib.assert(e.x2 == 2, "test failed")
    regentlib.assert(e.y1 == 3, "test failed")
    regentlib.assert(e.y2 == 4, "test failed")
    regentlib.assert(e.z1 == 5, "test failed")
    regentlib.assert(e.z2 == 6, "test failed")
    regentlib.assert(e.w1 == 7, "test failed")
    regentlib.assert(e.w2 == 8, "test failed")
  end
end
regentlib.start(main)
