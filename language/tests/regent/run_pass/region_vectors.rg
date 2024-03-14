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

fspace t {
  x : vector(int, 2),
  y : vector(int, 2),
}

task f() : int
  var r = region(ispace(ptr, 5), t)

  var p0 = dynamic_cast(ptr(t, r), 0)
  var p1 = dynamic_cast(ptr(t, r), 1)
  var p2 = dynamic_cast(ptr(t, r), 2)
  var p3 = dynamic_cast(ptr(t, r), 3)
  var p4 = dynamic_cast(ptr(t, r), 4)

  p0.x = vector(5, 4)
  p1.x = vector(40, 39)
  p2.x = vector(300, 299)
  p3.x = vector(2000, 1999)
  p4.x = vector(10000, 9999)

  p0.y = vector(4, 5)
  p1.y = vector(39, 40)
  p2.y = vector(299, 300)
  p3.y = vector(1999, 2000)
  p4.y = vector(9999, 10000)

  for p in r do
    p.y = p.y * (p.x + 100)
  end

  var s : int = 0
  for p in r do
    s = s + p.y[1] - p.y[0]
  end

  return s
end

task main()
  regentlib.assert(f() == 500, "test failed")
end
regentlib.start(main)
