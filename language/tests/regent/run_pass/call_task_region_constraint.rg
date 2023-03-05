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

local c = regentlib.c

fspace assert_disjoint {
  x : region(int),
  y : region(int),
} where x * y end

task f(r : region(int), s : region(int))
where
  reads(r, s),
  writes(r, s),
  r * s
do
  assert_disjoint { x = r, y = s }
end

task main()
  var a = region(ispace(ptr, 5), int)

  var ac = c.legion_coloring_create()
  c.legion_coloring_ensure_color(ac, 0)
  c.legion_coloring_ensure_color(ac, 1)
  var p = partition(disjoint, a, ac)
  c.legion_coloring_destroy(ac)

  var b = p[0]
  var d = p[1]

  f(b, d)
end
regentlib.start(main)
