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

-- Parameters to a task should be able to use the same name string (as
-- long as the symbols themselves are unique).

local function make_task()
  local b = regentlib.newsymbol(int, "same_name")
  local c = regentlib.newsymbol(double, "same_name")
  local d = regentlib.newsymbol(bool, "same_name")
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
