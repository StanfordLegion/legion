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

task main()
  var r = region(ispace(ptr, 5), int)
  var x0 = new(ptr(int, r))
  var x1 = new(ptr(int, r))
  var x2 = new(ptr(int, r))
  var x3 = new(ptr(int, r))
  var p = partition(equal, r, ispace(int1d, 3, 0))

  @x0 = 1
  @x1 = 2
  @x2 = 3
  @x3 = 5

  var t = 0

  __demand(__spmd, __trace)
  do
    for i = 0, 3 do
      inc(p[i], 100)
    end
  end

  regentlib.assert(@x0 == 101, "test failed")
  regentlib.assert(@x1 == 102, "test failed")
  regentlib.assert(@x2 == 103, "test failed")
  regentlib.assert(@x3 == 105, "test failed")
end
regentlib.start(main)
