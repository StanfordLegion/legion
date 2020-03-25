-- Copyright 2020 Stanford University
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

local c = regentlib.c

task condition1()
  return true
end

task condition2()
  return false
end

task body1()
  return 1 + 1
end

task body2(x : int)
  return x + 1
end

task main()
  __demand(__predicate)
  if condition1() then
    body1()
  end

  -- Do blocks, variables, assignment, simple expressions are all ok.
  var z = 123
  __demand(__predicate)
  if condition1() then
    do
      var x = 1
      do
        var y = body2(x)
      end
      z = body2(x) + 10
      z = z + 200
    end
  end
  regentlib.assert(z == 212, "test failed")

  -- Make sure assignment doesn't take effect if the condition is false.
  var w = 123
  var u = 456
  var v = 789 -- this variable is NOT assigned to the result of a task
  __demand(__predicate)
  if condition2() then
    w = body2(10) + 10
    u = body2(100)
    v = 1000
  end
  regentlib.assert(w == 123, "test failed")
  regentlib.assert(u == 456, "test failed")
  regentlib.assert(v == 789, "test failed")
end
regentlib.start(main)
