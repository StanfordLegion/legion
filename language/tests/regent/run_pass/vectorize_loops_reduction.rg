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

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task main()
  var r = region(ispace(ptr, 5), t)
  var x0 = dynamic_cast(ptr(t, r), 0)
  var x1 = dynamic_cast(ptr(t, r), 1)
  var x2 = dynamic_cast(ptr(t, r), 2)
  var x3 = dynamic_cast(ptr(t, r), 3)

  var i = 0
  for x in r do
    x.a = 10000 + 10 * i
    x.b = 0
    x.c = 0
    i += 1
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  regentlib.assert(x0.a == 10000, "test failed")
  regentlib.assert(x1.a == 10010, "test failed")
  regentlib.assert(x2.a == 10020, "test failed")
  regentlib.assert(x3.a == 10030, "test failed")
end
regentlib.start(main)
