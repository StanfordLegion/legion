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

struct foo
{
  x : int,
  y : int,
  z : int,
}

task f() : double
  var a : foo, b : foo, c : foo
  a.{x, y, z} = 3
  b.x, b.y, a.z = 6, 3, 2
  c.{x, y, z,} = a.{x, y, z,} * b.{y, x, z,} -- Note: Test trailing commas.
  c.{x, y, z} = c.{x, y, z} * 2 + a.x
  c.{x, y}, c.z += 100 - c.{z, y}, 100 - c.x
  return c.x + c.y + c.z
end

task g() : double
  var a : foo, b : foo, c : foo
  a.x, a.y, a.z = 3, 3, 3
  b.x, b.y, a.z = 6, 3, 2
  c.x, c.y, c.z = a.x * b.y, a.y * b.x, a.z * b.z
  c.x, c.y, c.z = c.x * 2 + a.x, c.y * 2 + a.x, c.z * 2 + a.x
  c.x, c.y, c.z += 100 - c.z, 100 - c.y, 100 - c.x
  return c.x + c.y + c.z
end

task main()
  regentlib.assert(f() == g(), "test failed")
end
regentlib.start(main)
