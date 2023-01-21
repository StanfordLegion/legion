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

local function make_sum_task()
  local x = regentlib.newsymbol(int, "x")
  local y = regentlib.newsymbol(int, "y")
  local z = regentlib.newsymbol(int, "z")
  local args = terralib.newlist({ x, y, z })
  local task sum([args])
    return [x] + [y] + [z]
  end
  return sum
end
local sum = make_sum_task()

local function make_main_task()
  local x = regentlib.newsymbol(int, "x")
  local args = terralib.newlist({300, rexpr [x] end, rexpr [x] + 1 end})
  local task main()
    var [x] = 10
    regentlib.c.printf("sum %d\n", sum(args))
    regentlib.assert(sum(args) == 321, "test failed")
  end
  return main
end
local main = make_main_task()
regentlib.start(main)
