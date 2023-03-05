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

-- This checks if the inline mapping optimization handles blocks correctly.

task f(r : region(int), x : ptr(int, r))
where reads(r), writes(r) do
  @x += 10
end

task g(r : region(int), x : ptr(int, r))
where reads(r), writes(r) do
end

task main()
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  @x = 3
  f(r, x)

  if true then
    var v = @x
    regentlib.assert(v == 13, "test failed")
  end

  g(r, x)

  if false then
  else
    var v = @x
    regentlib.assert(v == 13, "test failed")
  end

  g(r, x)

  if true then
    var v = @x
    regentlib.assert(v == 13, "test failed")
  else
    var v = @x
    regentlib.assert(v == 13, "test failed")
  end

  @x += 10
  f(r, x)
  while true do
    regentlib.assert(@x == 33, "test failed")
    break
  end

  @x += 10
  f(r, x)
  repeat
    regentlib.assert(@x == 53, "test failed")
  until true

  @x += 10
  f(r, x)
  for i = 0, 1 do
    regentlib.assert(@x == 73, "test failed")
  end

  @x += 10
  f(r, x)
  for i in r.ispace do
    regentlib.assert(@x == 93, "test failed")
    break
  end

  @x += 10
  f(r, x)
  do
    regentlib.assert(@x == 113, "test failed")
  end
end
regentlib.start(main)
