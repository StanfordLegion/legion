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

-- fails-with:
-- specialize_lua_function.rg:30: unable to specialize lua function (use terralib.cast to explicitly cast it to a terra function type)
--   g(z)
--   ^

import "regent"

x = false

function g(y)
  regentlib.assert(y == 5, "test failed")
  x = true
end

task f(z : int)
  g(z)
end

task main()
  f(5)
end

regentlib.assert(not x, "test failed")
regentlib.start(main)
regentlib.assert(x, "test failed")
