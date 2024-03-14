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

-- This tests that tasks can accept parameters via either task
-- arguments or futures.

task f(x : int) : int
  return 5 * x
end

task g(x : int, y : int, z : int)
  return x + y + z
end

task main()
  regentlib.assert(g(1, 20, 300) == 321, "test failed")
  regentlib.assert(g(1, 20, f(100)) == 521, "test failed")
  regentlib.assert(g(1, f(10), 300) == 351, "test failed")
  regentlib.assert(g(1, f(10), f(100)) == 551, "test failed")
  regentlib.assert(g(f(1), 20, 300) == 325, "test failed")
  regentlib.assert(g(f(1), 20, f(100)) == 525, "test failed")
  regentlib.assert(g(f(1), f(10), 300) == 355, "test failed")
  regentlib.assert(g(f(1), f(10), f(100)) == 555, "test failed")

  do
    var x = 0
    var y = f(2)
    for i = 1, 11 do
      -- This should not concretize because every point is the same.
      x += f(y)
    end
    regentlib.assert(x == 500, "test failed")
  end

  var x = -f(5)
  regentlib.assert(x == -25, "test failed")

  var y = f(10) + f(100)
  regentlib.assert(y == 550, "test failed")
end
regentlib.start(main)
