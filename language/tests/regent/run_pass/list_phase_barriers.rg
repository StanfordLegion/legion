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

-- runs-with:
-- [["-ll:cpu", "4"]]

import "regent"

-- A test of various language features need for expressing the
-- automatic SPMD optimization.

local c = regentlib.c

struct elt {
  a : int,
  b : int,
  c : int,
  d : int,
}

task shard(empty_in : regentlib.list(phase_barrier))
  c.printf("running shard\n")

  empty_in = advance(empty_in)
end

task main()
  var lo, hi = 0, 4

  var r_private = region(ispace(ptr, hi-lo), elt)
  var x0 = dynamic_cast(ptr(elt, r_private), 0)
  var x1 = dynamic_cast(ptr(elt, r_private), 1)
  var x2 = dynamic_cast(ptr(elt, r_private), 2)
  var x3 = dynamic_cast(ptr(elt, r_private), 3)

  var cp = c.legion_coloring_create()
  for i = lo, hi do
    c.legion_coloring_ensure_color(cp, i)
  end
  c.legion_coloring_add_point(cp, 0, __raw(x0))
  c.legion_coloring_add_point(cp, 1, __raw(x1))
  c.legion_coloring_add_point(cp, 2, __raw(x2))
  c.legion_coloring_add_point(cp, 3, __raw(x3))
  var p_private = partition(disjoint, r_private, cp)
  c.legion_coloring_destroy(cp)

  var r_ghost = region(ispace(ptr, hi-lo), elt)
  var y0 = dynamic_cast(ptr(elt, r_ghost), 0)
  var y1 = dynamic_cast(ptr(elt, r_ghost), 1)
  var y2 = dynamic_cast(ptr(elt, r_ghost), 2)
  var y3 = dynamic_cast(ptr(elt, r_ghost), 3)

  var cg = c.legion_coloring_create()
  for i = lo, hi do
    c.legion_coloring_ensure_color(cg, i)
  end
  c.legion_coloring_add_point(cg, 0, __raw(y0))
  c.legion_coloring_add_point(cg, 0, __raw(y1))
  c.legion_coloring_add_point(cg, 1, __raw(y0))
  c.legion_coloring_add_point(cg, 1, __raw(y1))
  c.legion_coloring_add_point(cg, 1, __raw(y2))
  c.legion_coloring_add_point(cg, 2, __raw(y1))
  c.legion_coloring_add_point(cg, 2, __raw(y2))
  c.legion_coloring_add_point(cg, 2, __raw(y3))
  c.legion_coloring_add_point(cg, 3, __raw(y2))
  c.legion_coloring_add_point(cg, 3, __raw(y3))
  var p_ghost = partition(aliased, r_ghost, cg)
  c.legion_coloring_destroy(cg)

  var rs_private = list_duplicate_partition(p_private, list_range(lo, hi))
  var rs_ghost = list_duplicate_partition(p_ghost, list_range(lo, hi))
  var rs_ghost_product = list_cross_product(rs_ghost, rs_ghost)
  var rs_ghost_empty_in = list_phase_barriers(rs_ghost_product)
  must_epoch
    for i = lo, hi do
      var rs_p = rs_private[i]
      var rs_g = rs_ghost[i]
      var rs_g_e_i = rs_ghost_empty_in[i]
      shard(rs_g_e_i)
    end
  end
end
regentlib.start(main)
