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

task f(x : int[4])
  return x[0] + x[1] + x[2] + x[3]
end

task g(x : regentlib.string[4])
  return regentlib.c.strlen(x[0]) +
    regentlib.c.strlen(x[1]) +
    regentlib.c.strlen(x[2]) +
    regentlib.c.strlen(x[3])
end

task h(x : int[2][2])
  x[0][0] += 1
  x[0][1] += 2
  x[1][0] += 3
  x[1][1] += 4
  return x
end

task i(x : regentlib.string[2])
  var z : int[2]
  z[0] = regentlib.c.strlen(x[0])
  z[1] = regentlib.c.strlen(x[1])

  var y : regentlib.string[2][2]
  y[0][0] = [rawstring](regentlib.c.calloc(z[0] + z[0] + 1, 1))
  y[0][1] = [rawstring](regentlib.c.calloc(z[0] + z[1] + 1, 1))
  y[1][0] = [rawstring](regentlib.c.calloc(z[1] + z[0] + 1, 1))
  y[1][1] = [rawstring](regentlib.c.calloc(z[1] + z[1] + 1, 1))

  regentlib.c.strcat(y[0][0], x[0])
  regentlib.c.strcat(y[0][0], x[0])

  regentlib.c.strcat(y[0][1], x[0])
  regentlib.c.strcat(y[0][1], x[1])

  regentlib.c.strcat(y[1][0], x[1])
  regentlib.c.strcat(y[1][0], x[0])

  regentlib.c.strcat(y[1][1], x[1])
  regentlib.c.strcat(y[1][1], x[1])

  return y
end

task main()
  var a : int[4]
  a[0] = 1
  a[1] = 20
  a[2] = 300
  a[3] = 4000
  regentlib.assert(f(a) == 4321, "test failed")

  var b : regentlib.string[4]
  b[0] = "1"
  b[1] = "12"
  b[2] = "123"
  b[3] = "1234"
  regentlib.assert(g(b) == 10, "test failed")

  var c : int[2][2]
  c[0][0] = 10
  c[0][1] = 20
  c[1][0] = 30
  c[1][1] = 40
  var d = h(c)
  regentlib.assert(d[0][0] == 11, "test failed")
  regentlib.assert(d[0][1] == 22, "test failed")
  regentlib.assert(d[1][0] == 33, "test failed")
  regentlib.assert(d[1][1] == 44, "test failed")

  var e : regentlib.string[2]
  e[0] = "1"
  e[1] = "2"
  var j = i(e)
  regentlib.assert(regentlib.c.strcmp(j[0][0], "11") == 0, "test failed")
  regentlib.assert(regentlib.c.strcmp(j[0][1], "12") == 0, "test failed")
  regentlib.assert(regentlib.c.strcmp(j[1][0], "21") == 0, "test failed")
  regentlib.assert(regentlib.c.strcmp(j[1][1], "22") == 0, "test failed")
end
regentlib.start(main)
