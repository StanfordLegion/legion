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

-- A test of various language features need for expressing the
-- automatic SPMD optimization.

local c = regentlib.c

task phase1(r_private : region(int), r_ghost : region(int))
where reads writes(r_private), reads(r_ghost) do
end

task phase2(r_private : region(int), r_ghost : region(int))
where reads writes(r_private), reduces +(r_ghost) do
end

task phase3(r_private : region(int), r_ghost : region(int))
where reads writes(r_private), reads(r_ghost) do
end

if false then
task shard(is : regentlib.list(int),
           rs_private : regentlib.list(region(int)),
           rs_ghost : regentlib.list(region(int)),
           rs_ghost_product : regentlib.list(regentlib.list(region(int))))
where
  reads writes(rs_private, rs_ghost, rs_ghost_product),
  simultaneous(rs_ghost, rs_ghost_product),
  rs_private * rs_ghost,
  rs_private * rs_ghost_product,
  rs_ghost * rs_ghost_product
do
  for i in is do
    phase1(rs_private[i], rs_ghost[i])
  end

  -- FIXME: Somehow we need to handle the use of reduction fields (or
  -- regions, but I believe fields are easier). This is marked using
  -- with_reduction_fields below.

  -- Zero the reduction fields:
  for i in is do
    fill(with_reduction_fields(rs_ghost[i]), 0) -- awaits(...)
  end
  for i in is do
    phase2(rs_private[i], with_reduction_fields(rs_ghost[i]))
  end
  copy((with_reduction_fields(rs_ghost)), rs_ghost, +) -- arrives(...)
  copy((with_reduction_fields(rs_ghost)), rs_ghost_product, +) -- arrives(...)
  -- Explicitly:
  -- for i in is do
  --   for rs_ghost_product_i_j in rs_ghost_product[i] do
  --     copy(rs_ghost[i], rs_ghost_product_i_j)
  --   end
  -- end

  -- awaits(...)
  for i in is do
    phase3(rs_private[i], rs_ghost[i])
  end
end
end

-- x : regentlib.list(regentlib.list(region(...))) = list_cross_product(y, z)
-- x[i][j] is the subregion of z[j] that intersects with y[i]
-- Note: This means there is NO x[i] such that x[i][j] <= x[i]
-- (because x[i][j] <= z[j] instead of y[i]).

task main()
  var lo, hi, stride = 0, 10, 3

  var r_private = region(ispace(ptr, hi-lo), int)
  var r_ghost = region(ispace(ptr, hi-lo), int)

  var rc = c.legion_coloring_create()
  for i = lo, hi do
    c.legion_coloring_ensure_color(rc, i)
  end
  var p_private = partition(disjoint, r_private, rc)
  var p_ghost = partition(aliased, r_ghost, rc)
  c.legion_coloring_destroy(rc)

  var rs_private = list_duplicate_partition(p_private, list_range(lo, hi))
  var rs_ghost = list_duplicate_partition(p_ghost, list_range(lo, hi))
  -- var rs_ghost_product = list_cross_product(rs_ghost, rs_ghost)
  -- must_epoch
    for i = lo, hi, stride do
      var ilo, ihi = i, regentlib.fmin(i+stride, hi)
      c.printf("launching shard ilo..ihi %d..%d\n",
               ilo, ihi)
      -- var is = list_range(ilo, ihi)
      -- var rs_p = rs_private[is]
      -- var rs_g = rs_ghost[is]
      -- var rs_g_p = rs_ghost_product[is]
      -- shard(is, rs_p, rs_g, rs_g_p)
    end
  -- end
end
regentlib.start(main)
