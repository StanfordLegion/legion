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

-- runs-with:
-- [["-fmapping", "0"]]

-- FIXME: This is hitting a bug in inline optimization.

import "regent"

task f(r : region(int), i : int)
where reads writes(r) do
  for x in r do
    @x += i*int(x)
  end
end

task check(r : region(int), x0 : ptr(int, r), x1 : ptr(int, r), x2 : ptr(int, r), x3 : ptr(int, r), x4 : ptr(int, r))
where reads(r) do
  regentlib.assert(@x0 == 0, "test failed")
  regentlib.assert(@x1 == 2, "test failed")
  regentlib.assert(@x2 == 6, "test failed")
  regentlib.assert(@x3 == 12, "test failed")
  regentlib.assert(@x4 == 20, "test failed")
end

task main()
  var r = region(ispace(ptr, 5), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)
  var x4 = dynamic_cast(ptr(int, r), 4)
  var p = partition(equal, r, ispace(int1d, 5))

  fill(r, 0)

  __demand(__index_launch)
  for i = 0, 5 do
    var j = i + 1
    f(p[i], j)
  end

  for x in r do
    regentlib.c.printf("pointer %d value %d\n", int(x), @x)
  end

  check(r, x0, x1, x2, x3, x4)

  -- regentlib.assert(@x0 == 0, "test failed")
  -- regentlib.assert(@x1 == 2, "test failed")
  -- regentlib.assert(@x2 == 6, "test failed")
  -- regentlib.assert(@x3 == 12, "test failed")
  -- regentlib.assert(@x4 == 20, "test failed")
end
regentlib.start(main)
