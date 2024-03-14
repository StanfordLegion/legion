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
-- [
--   ["-fpredicate", "1"],
--   ["-fpredicate", "0"]
-- ]

import "regent"

local c = regentlib.c

task condition1()
  return true
end

task condition2()
  return false
end

task body1(count : region(ispace(int1d), int))
where reads writes(count) do
  for x in count do
    @x += 1
  end
end

task body2(x : int)
  return x + 1
end

task main()
  var count = region(ispace(int1d, 2), int)
  var pcount = partition(equal, count, ispace(int1d, 2))
  fill(count, 0)

  regentlib.assert(count[0] == 0 and count[1] == 0, "test failed")

  __demand(__predicate)
  if condition1() then
    body1(count)
  end

  regentlib.assert(count[0] == 1 and count[1] == 1, "test failed")

  -- Basic control flow, variables, assignment, simple expressions are all ok.
  var z = 123
  var t = true
  __demand(__predicate)
  if condition1() then
    do
      var x = 1
      do
        var y = body2(x)
      end
      z = body2(x) + 10
      if t then
        z = z + 200
      end
      while not t do
        z = z + 3000
      end
    end

    for i = 0, 2 do
      __demand(__index_launch)
      for j = 0, 2 do
        body1(pcount[j])
      end
    end
  end
  regentlib.assert(z == 212, "test failed")

  regentlib.assert(count[0] == 3 and count[1] == 3, "test failed")

  -- Make sure assignment doesn't take effect if the condition is false.
  var w = 123
  var u = 456
  var v = 789 -- this variable is NOT assigned to the result of a task
  __demand(__predicate)
  if condition2() then
    w = body2(10) + 10
    u = body2(100)
    v = 1000
    body1(count)
    __demand(__index_launch)
    for j = 0, 2 do
      body1(pcount[j])
    end
  end
  regentlib.assert(w == 123, "test failed")
  regentlib.assert(u == 456, "test failed")
  regentlib.assert(v == 789, "test failed")

  -- Nested conditions are ok.
  __demand(__predicate)
  if condition1() then
    __demand(__predicate)
    if condition2() then
      v = 2000
    end
  end

  regentlib.assert(v == 789, "test failed")

  regentlib.assert(count[0] == 3 and count[1] == 3, "test failed")
end
regentlib.start(main)
