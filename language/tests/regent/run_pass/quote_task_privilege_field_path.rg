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

struct u { b : int, c : int }
struct s { a : u }

local r = regentlib.newsymbol(region(s), "r")
local reads_r_a_b =
  regentlib.privilege(regentlib.reads, r, regentlib.field_path("a", "b"))
task f([r])
where [reads_r_a_b] do
  var t = 0
  for x in r do
    t += x.a.b
  end
  return t
end

task main()
  var r = region(ispace(ptr, 3), s)
  fill(r.a.b, 10)
  regentlib.assert(f(r) == 30, "test failed")
end
regentlib.start(main)
