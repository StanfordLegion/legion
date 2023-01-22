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

struct s { a : int, b : int }

local r = regentlib.newsymbol(region(s), "r")
local reads_r_a = regentlib.privilege(regentlib.reads, r, "a")
local writes_r_a = regentlib.privilege(regentlib.writes, r, "a")
local reads_r_b = regentlib.privilege(regentlib.reads, r, "a")
local reads_writes_r_a = terralib.newlist({reads_r_a, writes_r_a, reads_r_b})
local atomic_r_a = regentlib.coherence(regentlib.atomic, r, "a")
local atomic_r_b = regentlib.coherence(regentlib.atomic, r, "a")
local atomic_r_a_b = terralib.newlist({atomic_r_a, atomic_r_b})

task f([r])
where [reads_writes_r_a], [atomic_r_a_b] do
  for i in r do
    i.a += 1
  end
end

task main()
  var r = region(ispace(ptr, 3), s)
  fill(r.a, 10)
  f(r)

  var t = 0
  for i in r do
    t += i.a
  end
  regentlib.assert(t == 33, "test failed")
end
regentlib.start(main)
