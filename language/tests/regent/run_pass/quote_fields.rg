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

local fields = terralib.newlist({"a", "b", "c"})

local elt = terralib.types.newstruct("elt")
elt.entries = fields:map(function(field) return { field, int } end)

function make_inc(fields)
  if type(fields) == "string" then fields = terralib.newlist({fields}) end
  local task inc(r : region(elt), v : int)
  where reads writes(r.[fields]) do
    for x in r do
      [fields:map(function(field) return rquote x.[field] += v end end)]
    end
  end
  return inc
end

local inc_a = make_inc("a")
local inc_bc = make_inc(terralib.newlist({"b", "c"}))

task main()
  var r = region(ispace(ptr, 3), elt)

  fill(r.{a, b, c}, 1000)
  inc_a(r, 500)
  inc_bc(r, 2)

  for x in r do
    regentlib.assert(x.a + x.b + x.c == 3504, "test failed")
  end
end
regentlib.start(main)
