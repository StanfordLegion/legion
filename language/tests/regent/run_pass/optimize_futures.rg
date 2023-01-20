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

task f() : int
  return 5
end

struct t {
  a0 : int64,
  a1 : int64,
  a2 : int64,
  a3 : int64,
  a4 : int64,
  a5 : int64,
  a6 : int64,
  a7 : int64,
  a8 : int64,
}

task g() : t
  var x : t
  x.a0 = 1
  x.a1 = 20
  x.a2 = 300
  x.a3 = 4000
  x.a4 = 50000
  x.a5 = 600000
  x.a6 = 7000000
  x.a7 = 80000000
  x.a8 = 900000000
  return x
end

task h()
  return 0.0
end

task main()
  var x = 10
  var y = 20
  y = x
  regentlib.assert(y == 10, "test failed")

  var z = f()
  y = z
  regentlib.assert(y == 5, "test failed")

  y += 30
  regentlib.assert(y == 35, "test failed")

  f() -- Do cleanups for raw function calls. DO NOT block.

  y; y -- Don't do cleanups for raw variable references.

  repeat
    var w = f()
  until w > 0 -- Future must be valid at condition.

  var a = g()
  var b : t
  b = a
  regentlib.assert(b.a4 == 50000, "test failed")

  var d = 1e99
  var e = h()
  e = d
  regentlib.assert(e == d, "test failed")
end
regentlib.start(main)
