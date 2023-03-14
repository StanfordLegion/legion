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
-- [
--   ["-fpredicate", "1"],
--   ["-fpredicate", "0"]
-- ]

import "regent"

local c = regentlib.c

local format = require("std/format")

task condition1()
  return true
end

task condition2()
  return false
end

task body1(i : int)
  return i * 2
end

task main()
  var x = 0

  -- x += y gets desugared to x = x + y
  __demand(__predicate)
  if condition1() then
    x += 1
  end
  regentlib.assert(x == 1, "test failed")

  __demand(__predicate)
  if condition2() then
    x += 20
  end
  regentlib.assert(x == 1, "test failed")

  __demand(__predicate)
  if condition1() then
    var y = 0   -- (Not a future, inside predication scope)
    y += 20     -- Ok, not a future
    x += y      -- Ok, gets desugared
  end
  regentlib.assert(x == 21, "test failed")

  __demand(__predicate)
  if condition2() then
    var y = 5   -- (Not a future, inside predication scope)
    y += 30     -- Ok, not a future
    x += y      -- Ok, gets desugared
  end
  regentlib.assert(x == 21, "test failed")

  __demand(__predicate)
  if condition1() then
    __demand(__index_launch)
    for i = 0, 3 do
      x += body1(i*10)
    end
  end
  regentlib.assert(x == 81, "test failed")

  __demand(__predicate)
  if condition2() then
    __demand(__index_launch)
    for i = 0, 10 do
      x += body1(i*100)
    end
  end
  regentlib.assert(x == 81, "test failed")
end
regentlib.start(main)
