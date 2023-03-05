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

local format = require("std/format")
regentlib.c.printf.replicable = true

local get_partition_color = regentlib.macro(function(lp)
  return rexpr
    regentlib.c.legion_index_partition_get_color(
      __runtime(), __raw(lp).index_partition)
  end
end)

local import_cross_product = regentlib.macro(function(...)
  local args = terralib.newlist({...})
  local partitions = args:sub(1, #args - 1)
  local colors = args[#args]
  return rexpr
    __import_cross_product(
        partitions, colors,
        regentlib.c.legion_terra_index_cross_product_import(
          __raw([ partitions[1] ]).index_partition, colors[1]))
  end
end)

task init_part(lr_int   : region(ispace(int3d), double),
               color    : int3d,
               subranks : int)
where
  writes(lr_int)
do
  var bounds = lr_int.bounds

  var blocking_factor = ((bounds.hi.z - bounds.lo.z + 1) + subranks - 1) / subranks

  var e = rect3d{ bounds.lo, int3d{bounds.hi.x, bounds.hi.y, bounds.lo.z+blocking_factor-1} }

  var t : transform(3, 1)
  t[{0, 0}] = 0
  t[{1, 0}] = 0
  t[{2, 0}] = blocking_factor

  var is_subrank = ispace(int1d, subranks)
  var lp_subrank = restrict(disjoint, complete, lr_int, t, e, is_subrank)

  format.println("COLOR: {} Bounds: {} Blocking Factor: {}",
           color, bounds, blocking_factor)

  for subrank_color in is_subrank do
    var subrank_bounds = lp_subrank[subrank_color].bounds
    format.println("\tCOLOR: {} SUBRANK COLOR: {} Bounds: {} Vol: {}",
                   color, subrank_color, subrank_bounds,
                   lp_subrank[subrank_color].volume)
  end

  return get_partition_color(lp_subrank)
end

__demand(__cuda)
task test(lr_int : region(ispace(int3d), double),
          color : int4d)
  var bounds = lr_int.bounds
  format.println("COLOR: {} Bounds: {}", color, bounds)
end

__demand(__replicable)
task main()
  var local_grid_size = int3d{64, 64, 64}
  var proc_grid_size = int3d{1, 1, 2}
  var global_grid_size = local_grid_size * proc_grid_size
  format.println("GLOBAL GRID SIZE: {}", global_grid_size)
  var is_grid = ispace(int3d, global_grid_size)

  var blocking_factor = global_grid_size / proc_grid_size
  format.println("RANK BLOCKING FACTOR: {}", blocking_factor)
  var ranks = proc_grid_size.x * proc_grid_size.y * proc_grid_size.z
  var subranks = 4
  format.println("RANKS: {} SUBRANKS: {}", ranks, subranks)

  var is_bounds = is_grid.bounds
  var is_rank = ispace(int3d, int3d{
                           (((is_bounds.hi.x - is_bounds.lo.x) + blocking_factor.x) / blocking_factor.x),
                           (((is_bounds.hi.y - is_bounds.lo.y) + blocking_factor.y) / blocking_factor.y),
                           (((is_bounds.hi.z - is_bounds.lo.z) + blocking_factor.z) / blocking_factor.z)},
                         int3d{0, 0, 0})

  var is_subrank = ispace(int1d, subranks)

  var lr_int = region(is_grid, double)
  fill(lr_int, 0)

  var t : transform(3, 3)
  for i = 0, 3 do
    for j = 0, 3 do
      t[{i, j}] = 0
    end
  end

  t[{0, 0}] = blocking_factor.x
  t[{1, 1}] = blocking_factor.y
  t[{2, 2}] = blocking_factor.z

  var e = rect3d{ int3d{0, 0, 0}, blocking_factor - int3d{1, 1, 1}}

  var lp_int_rank = restrict(disjoint, complete, lr_int, t, e, is_rank)

  var rank_color = get_partition_color(lp_int_rank)
  var subrank_color : regentlib.c.legion_color_t = -1

  for color in is_rank do
    subrank_color = init_part(lp_int_rank[color], color, subranks)
  end

  var fake_subrank = partition(equal, lr_int, is_subrank)

  var colors2 : regentlib.c.legion_color_t[2]
  colors2[0] = rank_color
  colors2[1] = subrank_color
  var cp2 = import_cross_product(lp_int_rank, fake_subrank, colors2)

  var rank_subrank = ispace(
    int4d,
    { is_rank.bounds.hi.x - is_rank.bounds.lo.x + 1,
      is_rank.bounds.hi.y - is_rank.bounds.lo.y + 1,
      is_rank.bounds.hi.z - is_rank.bounds.lo.z + 1,
      is_subrank.bounds.hi - is_subrank.bounds.lo + 1 },
    { is_rank.bounds.lo.x, is_rank.bounds.lo.y, is_rank.bounds.lo.z, is_subrank.bounds.lo })

  __demand(__index_launch)
  for i in rank_subrank do
    test(cp2[int3d{i.x, i.y, i.z}][i.w], i)
  end
end

regentlib.start(main)
