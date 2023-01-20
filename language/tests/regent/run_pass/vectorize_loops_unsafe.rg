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

-- FIXME: Turn off this test until vectorizer supports scattered reads again.
-- runs-with:
-- []
--

import "regent"

local c = regentlib.c

task foo(r : region(ispace(ptr), int))
where reads writes(r) do
  __demand(__vectorize)
  for p = 0, 10 do
    var e = unsafe_cast(ptr(int, r), p)
    @e = 10
  end

  for p = 0, 10 do
    var e = unsafe_cast(ptr(int, r), p)
    regentlib.assert(@e == 10, "test failed")
  end
end

task bar(r1 : region(ispace(ptr), int),
         r2 : region(ispace(ptr), int),
         s : region(ispace(ptr), ptr(int, r1, r2)))
where writes(r1, r2), reads writes(s) do
  __demand(__vectorize)
  for p1_raw = 0, 10 do
    var p1 = unsafe_cast(ptr(ptr(int, r1, r2), s), p1_raw)
    var p2 = unsafe_cast(ptr(int, r1), @p1)
    @p2 = 20
  end
end

task toplevel()
  var x = region(ispace(ptr, 10), int)
  var y = region(ispace(ptr, 10), int)
  var z = region(ispace(ptr, 10), ptr(int, x, y))
  for p in z do
    @p = static_cast(ptr(int, x, y), dynamic_cast(ptr(int, x), __raw(p).value))
  end
  foo(x)
  bar(x, y, z)
  for p in x do
    regentlib.assert(@p == 20, "test failed")
  end
end

regentlib.start(toplevel)
