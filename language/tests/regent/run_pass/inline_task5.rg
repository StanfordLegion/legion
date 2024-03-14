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

struct vec2 { x : int, y : int }
vec2.metamethods.__add =
  terra(a : vec2, b : vec2)
    return vec2 { a.x + b.x, a.y + b.y }
  end

__demand(__inline)
task t1(x : vec2)
  x.x += 1
  regentlib.assert(x.x == 322, "test failed")
end

__demand(__inline)
task t2(a : vec2)
  a.y = 1
end

__demand(__inline)
task t3(b : vec2)
  b = vec2 { 4, 5 }
end

__demand(__inline)
task t4(c : vec2)
  c += vec2 { 4, 5 }
end

__demand(__inline)
task t5(d : vec2)
  var e = d.x
end

task test1()
  var k = { 321, 123 }
  t1(k)
  t2(k)
  t3(k)
  t4(k)
  t5(k)
  var s = vec2(k)
  regentlib.assert(s.x == 321, "test failed")
  regentlib.assert(s.y == 123, "test failed")
end

__demand(__inline)
task ta(a : int)
  var b = &a
  @b = 2
end

__demand(__inline)
task tb(a : int)
  var c : &int
  c = &a
  @c = 2
end

task test2()
  var x = 1
  ta(x)
  tb(x)
  regentlib.assert(x == 1, "test failed")
end

__demand(__inline)
task tc(a : int[3])
  for i = 0, 3 do a[i] += i end
end

struct fs
{
  z : int[3];
}

__demand(__inline)
task td(a : fs)
  for i = 0, 3 do a.z[i] += i end
end

task test3()
  var x : int[3] = array(1, 2, 3)
  var y : fs = [fs] { z = x }
  tc(x)
  td(y)
  regentlib.assert(x[0] == 1, "test failed")
  regentlib.assert(x[1] == 2, "test failed")
  regentlib.assert(x[2] == 3, "test failed")
  regentlib.assert(y.z[0] == 1, "test failed")
  regentlib.assert(y.z[1] == 2, "test failed")
  regentlib.assert(y.z[2] == 3, "test failed")
end

task main()
  test1()
  test2()
  test3()
end

regentlib.start(main)
