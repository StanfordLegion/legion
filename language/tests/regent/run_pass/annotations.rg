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

-- runs-with:
-- [["-ll:cpu", "4"]]

import "regent"

-- This tests the parser with multiple options on a single statement.

task inc(r : region(int), y : int)
where reads writes(r) do
  for x in r do
    @x += y
  end
end

task f(t : int)
  return t < 1
end

task main()
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)
  var p = partition(equal, r, ispace(int1d, 3, 0))

  @x0 = 1
  @x1 = 2
  @x2 = 3
  @x3 = 5

  var t = 0

  var c = f(t)
  __demand(__predicate, __trace)
  while c do
    for i = 0, 3 do
      inc(p[i], 100)
    end
    t += 1
    c = f(t)
  end

  regentlib.assert(@x0 == 101, "test failed")
  regentlib.assert(@x1 == 102, "test failed")
  regentlib.assert(@x2 == 103, "test failed")
  regentlib.assert(@x3 == 105, "test failed")
end
regentlib.start(main)
