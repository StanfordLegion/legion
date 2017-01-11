-- Copyright 2017 Stanford University
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

  @x0 = 1
  @x1 = 2
  @x2 = 3
  @x3 = 5

  var t = 0

  __demand(__trace)
  while t < 2 do
    inc(r, 10)
    t += 1
  end

  __demand(__trace)
  for i = 0, 3 do
    inc(r, 100)
  end

  __demand(__trace)
  do
    inc(r, 1000)
  end

  regentlib.assert(@x0 == 1321, "test failed")
  regentlib.assert(@x1 == 1322, "test failed")
  regentlib.assert(@x2 == 1323, "test failed")
  regentlib.assert(@x3 == 1325, "test failed")
end
regentlib.start(main)
