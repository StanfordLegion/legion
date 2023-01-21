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

-- This tests various aspects of the inline mapping optimization.

task f(r : region(int), x : ptr(int, r))
where reads(r), writes(r) do
  @x += 10
  return @x
end

task g(r : region(int), x : ptr(int, r))
where reads(r), writes(r) do
  f(r, x)
  f(r, x)
  f(r, x)
  @x += 4000
end

task main()
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  @x = 3
  f(r, x)
  f(r, x)
  regentlib.assert(@x == 23, "test failed")

  if @x < 30 then
    f(r, x)
    f(r, x)
  elseif f(r, x) < 20 then
    @x += 3
  end
  regentlib.assert(@x == 43, "test failed")

  while true do
    f(r, x)
    @x += 100
    break
  end
  regentlib.assert(@x == 153, "test failed")

  for i = 0, @x, 100 do
    f(r, x)
  end
  regentlib.assert(@x == 173, "test failed")

  repeat
    f(r, x)
    @x += 100
  until @x > 200
  regentlib.assert(@x == 283, "test failed")

  @x = 7
  g(r, x)
  regentlib.assert(@x == 4037, "test failed")
end
regentlib.start(main)
