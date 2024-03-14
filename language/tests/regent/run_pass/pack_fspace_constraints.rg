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

local c = regentlib.c

fspace s(a : region(int)) {
  b : region(int),
} where b <= a end

task f()
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  c.legion_coloring_ensure_color(rc, 0)
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)
  var t = p[0]
  var x = [s(r)] { b = t }
end

task g()
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  c.legion_coloring_ensure_color(rc, 0)
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)
  var t = p[0]
  var tc = c.legion_coloring_create()
  c.legion_coloring_ensure_color(tc, 0)
  var q = partition(disjoint, t, tc)
  c.legion_coloring_destroy(tc)
  var u = q[0]
  var x = [s(r)] { b = u }
end

task main()
  f()
  g()
end
regentlib.start(main)
