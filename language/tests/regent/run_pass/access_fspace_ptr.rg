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

fspace point {
  x : int,
  y : int,
  z : int,
}

task f(r : region(point), p : ptr(point, r)) : int
where reads(r) do
  return p.x + p.y + p.z
end

task g() : point
  return point { x = 1, y = 20, z = 300 }
end

task h() : int
  var r = region(ispace(ptr, 5), point)
  var p = dynamic_cast(ptr(point, r), 0)
  @p = g()
  return f(r, p)
end

task main()
  regentlib.assert(h() == 321, "test failed")
end
regentlib.start(main)
