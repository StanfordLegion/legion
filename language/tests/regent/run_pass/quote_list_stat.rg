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

function make_constant_task(value)
  local x = regentlib.newsymbol(int, "x")
  local stats = terralib.newlist()
  stats:insert(rquote var [x] = [value] end)
  stats:insert(rquote return [x] end)
  local task t()
    [stats]
  end
  return t
end

local f = make_constant_task(7)

do
  local stats = terralib.newlist({
    rexpr regentlib.c.printf("result %d\n", f()) end,
    rexpr regentlib.assert(f() == 7, "test failed") end
  })
  task main()
    -- Hack: Regent perceives this as an expression, so the contents
    -- also need to be expressions.
    stats
  end
end
regentlib.start(main)
