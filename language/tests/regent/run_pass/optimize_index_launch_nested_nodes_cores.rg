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
-- [["-fflow", "0", "-foverride-demand-index-launch", "1"]]

-- Test index launches with cross-products in a semi-realistic scenario with
-- nodes, cores and ghost cells.

import "regent"

local format = require("std/format")

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

fspace fs {
  a : int,
  b : int,
}

task flat_launch(data : region(ispace(int1d), fs))
where reads writes(data) do
end

task nested_launch(data : region(ispace(int1d), fs),
                   data_by_core : partition(disjoint, data, ispace(int1d)))
where reads writes(data) do
  __demand(__index_launch)
  for core in data_by_core.colors do
    flat_launch(data_by_core[core])
  end
end

task flat_stencil(data : region(ispace(int1d), fs),
                  right : region(ispace(int1d), fs),
                  left : region(ispace(int1d), fs))
where reads writes(data.a), reads(data.b, right.b, left.b) do
end

task main()
  var data = region(ispace(int1d, 24), fs)

  var nodes = ispace(int1d, 4)
  var cores = ispace(int1d, 3)
  var right_left = ispace(int1d, 2)

  -- Set up the partitioning. Note we don't need to keep the nested
  -- partitions, just their colors.
  var data_by_node = partition(equal, data, nodes)
  var node_color = get_partition_color(data_by_node)
  var core_color : regentlib.c.legion_color_t = -1
  var right_left_color : regentlib.c.legion_color_t = -1
  for node in nodes do
    var data_by_core = partition(equal, data_by_node[node], cores)
    var color = get_partition_color(data_by_core)
    if core_color == regentlib.c.legion_color_t(-1) then
      core_color = color
    end
    regentlib.assert(
      color == core_color,
      "unable to assign same color to all second-level partitions")

    for core in cores do
      var data_by_right_left = partition(equal, data_by_core[core], right_left)

      var color = get_partition_color(data_by_right_left)
      if right_left_color == regentlib.c.legion_color_t(-1) then
        right_left_color = color
      end
      regentlib.assert(
        color == right_left_color,
        "unable to assign same color to all third-level partitions")
    end
  end

  -- Set up cross products. Note that we need a few fake partitions here. The
  -- exact values don't matter as long as the parent region, disjointness, and
  -- color space are correct.
  var fake_data_by_core = partition(equal, data, cores)
  var fake_data_by_right_left = partition(equal, data, right_left)

  var colors2 : regentlib.c.legion_color_t[2]
  colors2[0] = node_color
  colors2[1] = core_color
  var cp2 = import_cross_product(data_by_node, fake_data_by_core, colors2)

  var colors3 : regentlib.c.legion_color_t[3]
  colors3[0] = node_color
  colors3[1] = core_color
  colors3[2] = right_left_color
  var cp3 = import_cross_product(data_by_node, fake_data_by_core, fake_data_by_right_left, colors3)

  -- Sanity check we got this right so far.
  for node in nodes do
    format.println("node[{}] = {}", node, data_by_node[node].bounds)
    for core in cores do
      format.println("  core[{}] = {}", core, cp2[node][core].bounds)
      for rl in right_left do
        format.println("    right_left[{}] = {}", rl, cp3[node][core][rl].bounds)
      end
    end
  end

  -- Launch patterns:

  fill(data.{a, b}, 0)

  -- Nested launch. Note this doesn't work with tracing (no way to trace the
  -- second level), so not the most useful pattern. But still, it works.
  __demand(__index_launch)
  for node in nodes do
    nested_launch(data_by_node[node], cp2[node])
  end

  var nodes_cores = ispace(
    int2d,
    { nodes.bounds.hi - nodes.bounds.lo + 1, cores.bounds.hi - cores.bounds.lo + 1 },
    { nodes.bounds.lo, cores.bounds.lo })

  -- Flat launch with a 2D index space. The dimensions of the launch
  -- correspond to the first and second levels of the cross-product,
  -- respectively.

  -- Note: this does **NOT** work with the static/dynamic interference checks
  -- we have right now, so we need to -foverride-demand-index launch on this
  -- one.
  __demand(__index_launch)
  for i in nodes_cores do
    flat_launch(cp2[i.x][i.y])
  end

  -- Similar, but now with a ghost update (like we'd use for a stencil).

  -- Note: I haven't bothered to set up proper boundary conditions here so I
  -- just use max/min to make sure we don't go out of bounds. In a real
  -- implementation you'd use empty boundary regions around the edges (or similar).
  __demand(__index_launch)
  for i in nodes_cores do
    flat_stencil(cp2[i.x][i.y], cp3[i.x][max(i.y-1,0)][0], cp3[i.x][min(i.y+1,2)][1])
  end
end
regentlib.start(main)
