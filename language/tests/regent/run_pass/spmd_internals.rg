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
-- []

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

task phase1(r_private : region(elt), r_ghost : region(elt))
where reads writes(r_private), reads(r_ghost) do
end

task phase2(r_private : region(elt), r_ghost : region(elt))
where reads writes(r_private.{a, b}), reduces +(r_ghost.{a, b}) do
end

task phase3(r_private : region(elt), r_ghost : region(elt))
where reads writes(r_private), reads(r_ghost) do
end

task shard(is : regentlib.list(int),
           rs_private : regentlib.list(region(elt)),
           rs_ghost : regentlib.list(region(elt)),
           rs_ghost_product : regentlib.list(
             regentlib.list(region(elt), nil, 1), nil, 1),
           empty_in : regentlib.list(regentlib.list(phase_barrier)),
           empty_out : regentlib.list(regentlib.list(phase_barrier)),
           full_in : regentlib.list(regentlib.list(phase_barrier)),
           full_out : regentlib.list(regentlib.list(phase_barrier)))
where
  reads writes(rs_private, rs_ghost, rs_ghost_product),
  simultaneous(rs_ghost, rs_ghost_product),
  no_access_flag(rs_ghost_product)-- ,
  -- rs_private * rs_ghost,
  -- rs_private * rs_ghost_product,
  -- rs_ghost * rs_ghost_product
do
  c.printf("running shard\n")
  var f = allocate_scratch_fields(rs_ghost.{a, b})
  for i in is do
    phase1(rs_private[i], rs_ghost[i])
  end

  -- Zero the reduction fields:
  fill((with_scratch_fields(rs_ghost.{a, b}, f)).{a, b}, 0)
  for i in is do
    phase2(rs_private[i], with_scratch_fields((rs_ghost[i]).{a, b}, f))
  end
  copy((with_scratch_fields(rs_ghost.{a, b}, f)).{a, b}, rs_ghost.{a, b}, +)
  copy((with_scratch_fields(rs_ghost.{a, b}, f)).{a, b}, rs_ghost_product.{a, b}, +,
       awaits(empty_out), arrives(full_out))

  -- awaits(advance(full_in)), arrives(empty_in)
  for i in is do
    phase3(rs_private[i], rs_ghost[i])
  end

  empty_in = advance(empty_in)
  empty_out = advance(empty_out)
  full_in = advance(full_in)
  full_out = advance(full_out)
end

-- x : regentlib.list(regentlib.list(region(...))) = list_cross_product(y, z)
-- x[i][j] is the subregion of z[j] that intersects with y[i]
-- Note: This means there is NO x[i] such that x[i][j] <= x[i]
-- (because x[i][j] <= z[j] instead of y[i]).

task main()
  var lo, hi, stride = 0, 10, 3

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
  copy(r_private, rs_private)
  copy(r_ghost, rs_ghost)
  var rs_ghost_product = list_cross_product(rs_ghost, rs_ghost)
  var rs_ghost_empty_in = list_phase_barriers(rs_ghost_product)
  var rs_ghost_empty_out = list_invert(
    rs_ghost, rs_ghost_product, rs_ghost_empty_in)
  var rs_ghost_full_in = list_phase_barriers(rs_ghost_product)
  var rs_ghost_full_out = list_invert(
    rs_ghost, rs_ghost_product, rs_ghost_full_in)
  must_epoch
    for i = lo, hi, stride do
      var ilo, ihi = i, regentlib.fmin(i+stride, hi)
      c.printf("launching shard ilo..ihi %d..%d\n",
               ilo, ihi)
      var is = list_range(ilo, ihi)
      var iis = list_range(0, ihi-ilo)
      var rs_p = rs_private[is]
      var rs_g = rs_ghost[is]
      var rs_g_p = rs_ghost_product[is]
      var rs_g_e_i = rs_ghost_empty_in[is]
      var rs_g_e_o = rs_ghost_empty_out[is]
      var rs_g_f_i = rs_ghost_full_in[is]
      var rs_g_f_o = rs_ghost_full_out[is]
      shard(iis, rs_p, rs_g, rs_g_p, rs_g_e_i, rs_g_e_o, rs_g_f_i, rs_g_f_o)
    end
  end
end
regentlib.start(main)
