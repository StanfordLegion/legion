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

-- This tests that the compiler properly roots regions which are
-- nested more than one level deep.

local c = regentlib.c

task g(r : region(int)) : int
where reads(r), writes(r) do
  return 5
end

task main()
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  c.legion_coloring_ensure_color(rc, 0)
  var p_disjoint = partition(disjoint, r, rc)
  var r0 = p_disjoint[0]
  var p0_disjoint = partition(disjoint, r0, rc)
  c.legion_coloring_destroy(rc)

  var r00 = p0_disjoint[0]
  g(r00)
end
regentlib.start(main)
