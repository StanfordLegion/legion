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

local f = terralib.overloadedfunction(
  "f", {
    terra(x : int) : int
      return x + 1
    end,
    terra(x : double) : int
      return x - 1
    end
  })

task g(x : bool) : int
  var y : int = 4
  var z : double = 7.3
  if x then
    return f(y)
  else
    return f(z)
  end
end

task main()
  regentlib.assert(g(true) == 5, "test failed")
  regentlib.assert(g(false) == 6, "test failed")
end
regentlib.start(main)
