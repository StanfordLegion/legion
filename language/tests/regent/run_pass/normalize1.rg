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

struct st
{
  a : int,
  b : int,
}

task main()
  var x = 1
  var y = 2
  var z : st = { 3, 4 }
  do
    var y, x = x + 1, y + 2
    regentlib.assert(x == 4, "test failed")
    regentlib.assert(y == 2, "test failed")
  end
  do
    var x, y = 2, x + 3
    regentlib.assert(x == 2, "test failed")
    regentlib.assert(y == 4, "test failed")
  end
  do
    var x, y = x + 1, 3
    regentlib.assert(x == 2, "test failed")
    regentlib.assert(y == 3, "test failed")
  end
  do
    var x, y = x + y, 3
    regentlib.assert(x == 3, "test failed")
    regentlib.assert(y == 3, "test failed")
  end
  do
    var z, x, y = 3, z.a + z.b, x + y
    regentlib.assert(x == 7, "test failed")
    regentlib.assert(y == 3, "test failed")
    regentlib.assert(z == 3, "test failed")
  end
end

regentlib.start(main)
