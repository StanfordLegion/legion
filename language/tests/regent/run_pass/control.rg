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

task f() : int
  var x = 0

  for y : int = 0, 4 do
    x = x + 10000000
  end

  for y = 0, 4, 2 do
    x = x + 1000000
  end

  if true then
    x = x + 100000
  end

  if false then
    x = x + 10000
  else
    x = x + 20000
  end

  if false then
    x = x + 1000
  elseif true then
    x = x + 2000
  end

  if false then
    x = x + 100
  elseif false then
    x = x + 200
  else
    x = x + 300
  end

  while true do
    x = x + 10
    break
  end

  repeat
    var y = x + 1
    x = y
  until y > 0

  return x
end

task main()
  regentlib.assert(f() == 42122311, "test failed")
end
regentlib.start(main)
