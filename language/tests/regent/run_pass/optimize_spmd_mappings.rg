-- Copyright 2016 Stanford University
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

-- runs-with:
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-fflow-spmd-mapping", "1"],
--   ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-fflow-spmd-mapping", "2"],
--   ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-fflow-spmd-mapping", "3"],
-- ]

import "regent"

-- This tests the SPMD optimization of the compiler with partition-to-shard
-- mappings turned on.  Currently there's no easy way to tell if the mappings
-- make correct assignments; this test merely makes sure that the code runs and
-- produces the correct result.

local c = regentlib.c

terra to_rect(lo : int2d, hi : int2d) : c.legion_rect_2d_t
  return c.legion_rect_2d_t {
    lo = lo:to_point(),
    hi = hi:to_point(),
  }
end

task inc(r : region(ispace(int2d), int), delta : int)
where reads writes(r) do
  for p in r do
    r[p] += delta
  end
end

task main()
  var r = region(ispace(int2d, { x = 8, y = 8 }), int)
  var is_part = ispace(int2d, { x = 4, y = 4 })

  var p = partition(equal, r, is_part)

  var cq = c.legion_domain_point_coloring_create()
  for p in is_part do
    c.legion_domain_point_coloring_color_domain(
      cq, (int2d { x=p.x, y=p.y }):to_domain_point(),
      c.legion_domain_from_rect_2d(
        to_rect(int2d { x=p.y*2, y=p.x*2 }, int2d { x=p.y*2+1, y=p.x*2+1 })))
  end
  var q = partition(disjoint, r, cq, is_part)
  c.legion_domain_point_coloring_destroy(cq)

  fill(r, 0)

  __demand(__spmd)
  for t = 0, 3 do
    for i in is_part do
      inc(p[i], 3)
    end
    for i in is_part do
      inc(q[i], 5)
    end
  end

  for x in r do
    regentlib.assert(@x == 24, "value incorrect")
  end
end
regentlib.start(main)
