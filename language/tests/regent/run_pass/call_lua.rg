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

x = 4

function g(y)
  regentlib.assert(y == 30, "test failed")
  return 1200 + x + y
end
local tg = terralib.cast({int} -> int, g)

task f(z : int)
  return tg(z)
end

task main()
  regentlib.assert(f(30) == 1234, "test failed")
end
regentlib.start(main)
