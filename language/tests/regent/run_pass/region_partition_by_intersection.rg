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

task f()
  var r = region(ispace(ptr, 4), int)
  var x0 = dynamic_cast(ptr(int, r), 0)
  var x1 = dynamic_cast(ptr(int, r), 1)
  var x2 = dynamic_cast(ptr(int, r), 2)
  var x3 = dynamic_cast(ptr(int, r), 3)

  var pc = c.legion_coloring_create()
  c.legion_coloring_add_point(pc, 0, __raw(x0))
  c.legion_coloring_add_point(pc, 0, __raw(x1))
  c.legion_coloring_add_point(pc, 1, __raw(x2))
  c.legion_coloring_add_point(pc, 2, __raw(x3))
  var p = partition(disjoint, r, pc)
  c.legion_coloring_destroy(pc)

  var qc = c.legion_coloring_create()
  c.legion_coloring_add_point(qc, 1, __raw(x0))
  c.legion_coloring_add_point(qc, 0, __raw(x1))
  c.legion_coloring_add_point(qc, 2, __raw(x2))
  c.legion_coloring_add_point(qc, 2, __raw(x3))
  var q = partition(disjoint, r, qc)
  c.legion_coloring_destroy(qc)

  var pq = p & q

  for x in r do
    @x = 1
  end

  for i = 0, 3 do
    var ri = pq[i]
    for x in ri do
      @x *= i + 2
    end
  end

  var t = 0
  for x in r do
    t += @x
  end

  return t
end

task main()
  regentlib.assert(f() == 8, "test failed")
end
regentlib.start(main)
