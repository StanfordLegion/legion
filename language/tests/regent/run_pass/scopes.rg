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

x = 1
y = 2

task f(y : int)
  regentlib.assert(x == 1 and y == 20, "test failed")
  var x = 10
  regentlib.assert(x == 10, "test failed")
  do
    x = 100
    regentlib.assert(x == 100, "test failed")

    var y = 200
    regentlib.assert(y == 200, "test failed")

    if y == 200 then
      var y = 2000
      regentlib.assert(y == 2000, "test failed")
    else
      regentlib.assert(false, "test failed")
    end
    regentlib.assert(y == 200, "test failed")

    do
      var z = false
      while y == 200 do
        var y = 2000
        regentlib.assert(y == 2000, "test failed")
        z = true
        break
      end
      regentlib.assert(z, "test failed")
    end
    regentlib.assert(y == 200, "test failed")

    for y = y*10, y*10 + 1 do
      regentlib.assert(y == 2000, "test failed")
      var y = 20000
      regentlib.assert(y == 20000, "test failed")
    end
    regentlib.assert(y == 200, "test failed")

    repeat
      var y = 2000
      regentlib.assert(y == 2000, "test failed")
    until y == 2000
    regentlib.assert(y == 200, "test failed")

  end
  regentlib.assert(x == 100 and y == 20, "test failed")
end

task main()
  f(20)
end
regentlib.start(main)
