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

local function make_task()
  local b = regentlib.newsymbol(int, "b")
  local c = regentlib.newsymbol(double, "c")
  local d = regentlib.newsymbol(bool, "d")
  local rest = terralib.newlist({ c, d })
  local task result(a : int, [b], [rest]) : int
    if d then
      return 314
    else
      return a + b*c
    end
  end
  return result
end
local f = make_task()

task main()
  regentlib.assert(f(23, 200, 0.5, false) == 123, "test failed")
end
regentlib.start(main)
