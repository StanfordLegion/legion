-- Copyright 2015 Stanford University
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

local c = regentlib.c

-- task f(x : regentlib.list(region))
-- end

task main()
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  for i = 0, 7 do
    c.legion_coloring_ensure_color(rc, i)
  end
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)

  var x = list_range(3, 7)
  var y = list_duplicate_partition(p, x)
  -- f(x)
end
regentlib.start(main)
