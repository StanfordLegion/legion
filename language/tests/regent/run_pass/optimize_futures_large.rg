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

-- This test exists to ensure that task arguments and return values
-- (and particularly futures) play nicely with large values.

struct s {
  x : int64,
  y : int64,
  z : int32,
  a : double,
  b : double,
  c : bool,
}

task f(v : s) : s
  v.x *= 2
  v.y *= 3
  v.z *= 5
  v.a += 1.5
  v.b += 2.25
  v.c = not v.c
  return v
end

task main()
  var v = s { x = 1, y = 1, z = 1, a = 1, b = 1, c = true }
  v = f(v)
  v = f(v)
  v = f(v)

  regentlib.assert(v.x == 8, "test failed")
  regentlib.assert(v.y == 27, "test failed")
  regentlib.assert(v.z == 125, "test failed")
  regentlib.assert(v.a == 5.5, "test failed")
  regentlib.assert(v.b == 7.75, "test failed")
  regentlib.assert(v.c == false, "test failed")
end
regentlib.start(main)
