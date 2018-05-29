-- Copyright 2018 Stanford University, NVIDIA Corporation
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

-- Mapping Optimizer
--
-- Attempts to place map/unmap calls to avoid thrashing inline mappings.

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local context = {}

function context:__index (field)
  local value = context [field]
  if value ~= nil then
    return value
  end
  error ("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex (field, value)
  error ("context has no field '" .. field .. "' (in assignment)", 2)
end

function context:new_task_scope(constraints, region_universe)
  assert(constraints and region_universe)
  local cx = {
    constraints = constraints,
    region_universe = region_universe,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

-- Possible polarities of region usage.
local inline = "inline"
local remote = "remote"

local function uses(cx, region_type, polarity)
  -- In order for this to be sound, we need to unmap *all* regions
  -- that could potentially alias with this one, not just just the
  -- region itself. Not all of these regions will necessarily be
  -- mapped, but codegen will be able to do the right thing by
  -- ignoring anything that isn't a root in the region forest.

  assert(std.type_supports_privileges(region_type))
  local usage = data.newmap()
  usage[region_type] = polarity

  for other_region_type, _ in cx.region_universe:items() do
    if std.is_region(other_region_type) then -- Skip lists of regions
      local constraint = std.constraint(
        region_type,
        other_region_type,
        std.disjointness)
      if std.type_maybe_eq(region_type:fspace(), other_region_type:fspace()) and
        not std.check_constraint(cx, constraint)
      then
        usage[other_region_type] = polarity
      end
    end
  end
  return usage
end

local function usage_meet_polarity(a, b)
  if not a then
    return b
  elseif not b then
    return a
  elseif a == b then
    return a
  else
    -- This is safe because the Legion runtime knows how to map and
    -- unmap appropriately for each API call... it's just inefficient
    -- to do so. But if we reached this point then we are expecting to
    -- use the region both inline and remotely, so this is the best we
    -- can do (without rearranging the code).
    return inline
  end
end

local function usage_meet(...)
  local usage = data.newmap()
  for _, a in pairs({...}) do
    if a then
      for region_type, polarity in a:items() do
        usage[region_type] = usage_meet_polarity(usage[region_type], polarity)
      end
    end
  end
  return usage
end

local function usage_diff_polarity(a, b)
  if not a or not b or a == b then
    return nil
  else
    return b
  end
end

local function usage_diff(a, b)
  if not a or not b then
    return nil
  end
  local usage = data.newmap()
  for region_type, a_polarity in a:items() do
    local b_polarity = b[region_type]
    local diff = usage_diff_polarity(a_polarity, b_polarity)
    if diff then
      usage[region_type] = diff
    end
  end
  return usage
end

local function usage_apply_polarity(a, b)
  if not a then
    return b
  elseif not b then
    return a
  else
    return b
  end
end

local function usage_apply(...)
  local usage = data.newmap()
  for _, a in pairs({...}) do
    if a then
      for region_type, polarity in a:items() do
        usage[region_type] = usage_apply_polarity(usage[region_type], polarity)
      end
    end
  end
  return usage
end

local function analyze_usage_node(cx)
  return function(node)
    if node:is(ast.typed.expr.Call) and std.is_task(node.fn.value) then
      local usage
      for _, arg in ipairs(node.args) do
        local arg_type = std.as_read(arg.expr_type)
        if std.is_region(arg_type) then
          usage = usage_meet(usage, uses(cx, arg_type, remote))
        end
      end
      return usage
    elseif node:is(ast.typed.expr.RawPhysical) then
      local region_type = std.as_read(node.region.expr_type)
      return uses(cx, region_type, inline)
    elseif node:is(ast.typed.expr.Copy) then
      local src_type = std.as_read(node.src.expr_type)
      local dst_type = std.as_read(node.dst.expr_type)
      return usage_meet(
        uses(cx, src_type, remote),
        uses(cx, dst_type, remote))
    elseif node:is(ast.typed.expr.Fill) then
      local dst_type = std.as_read(node.dst.expr_type)
      return uses(cx, dst_type, remote)
    elseif node:is(ast.typed.expr.Acquire) then
      local region_type = std.as_read(node.region.expr_type)
      return uses(cx, region_type, remote)
    elseif node:is(ast.typed.expr.Release) then
      local region_type = std.as_read(node.region.expr_type)
      return uses(cx, region_type, remote)
    elseif node:is(ast.typed.expr.AttachHDF5) then
      local region_type = std.as_read(node.region.expr_type)
      return uses(cx, region_type, remote)
    elseif node:is(ast.typed.expr.DetachHDF5) then
      local region_type = std.as_read(node.region.expr_type)
      return uses(cx, region_type, remote)
    elseif node:is(ast.typed.expr.Region) then
      return uses(cx, node.expr_type, inline)
    elseif node:is(ast.typed.expr.PartitionByField) then
      return uses(cx, node.region.expr_type, remote)
    elseif node:is(ast.typed.expr.Image) then
      return uses(cx, node.region.expr_type, remote)
    elseif node:is(ast.typed.expr.Preimage) then
      return uses(cx, node.region.expr_type, remote)
    elseif node:is(ast.typed.expr.IndexAccess) then
      local base_type = std.as_read(node.value.expr_type)
      if std.is_region(base_type) then
        return uses(cx, base_type, inline)
      end
    elseif node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.Deref)
    then
      local ptr_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(ptr_type) and ptr_type:is_ptr() then
        return data.reduce(
          usage_meet,
          ptr_type:bounds():map(
            function(region) return uses(cx, region, inline) end))
      end
    end
  end
end

local function analyze_usage(cx, node)
  assert(node)
  return ast.mapreduce_node_postorder(
    analyze_usage_node(cx),
    usage_meet,
    node, nil)
end

local optimize_mapping = {}

local function annotate(node, in_usage, out_usage)
  return { node, in_usage, out_usage }
end

local function annotated_in_usage(annotated_node)
  return annotated_node[2]
end

local function annotated_out_usage(annotated_node)
  return annotated_node[3]
end

local function map_regions(diff)
  local result = terralib.newlist()
  if diff then
    local region_types_by_polarity = {}
    for region_type, polarity in diff:items() do
      if not region_types_by_polarity[polarity] then
        region_types_by_polarity[polarity] = terralib.newlist()
      end
      region_types_by_polarity[polarity]:insert(region_type)
    end
    for polarity, region_types in pairs(region_types_by_polarity) do
      if polarity == inline then
        result:insert(
          ast.typed.stat.MapRegions {
            region_types = region_types,
            annotations = ast.default_annotations(),
            span = ast.trivial_span(),
          })
      elseif polarity == remote then
        result:insert(
          ast.typed.stat.UnmapRegions {
            region_types = region_types,
            annotations = ast.default_annotations(),
            span = ast.trivial_span(),
          })
      else
        assert(false)
      end
    end
  end
  return result
end

local function fixup_block(annotated_block, in_usage, out_usage)
  local node, node_in_usage, node_out_usage = unpack(annotated_block)
  local stats = terralib.newlist()
  stats:insertall(map_regions(usage_diff(in_usage, node_in_usage)))
  stats:insertall(node.stats)
  stats:insertall(map_regions(usage_diff(node_out_usage, out_usage)))
  return node { stats = stats }
end

local function fixup_elseif(annotated_node, in_usage, out_usage)
  local node, node_in_usage, node_out_usage = unpack(annotated_node)
  local annotated_block = annotate(node.block, node_in_usage, node_out_usage)
  local block = fixup_block(annotated_block, in_usage, out_usage)
  return node { block = block }
end

function optimize_mapping.block(cx, node)
  local stats = node.stats:map(
    function(stat) return optimize_mapping.stat(cx, stat) end)

  local result_stats = terralib.newlist()
  local out_usage
  for _, stat_annotated in ipairs(stats) do
    local stat, stat_in_usage, stat_out_usage = unpack(stat_annotated)
    result_stats:insertall(map_regions(usage_diff(out_usage, stat_in_usage)))
    result_stats:insert(stat)
    out_usage = usage_apply(out_usage, stat_out_usage)
  end

  -- Now apply a reverse pass to compute in_usage.
  local in_usage = out_usage
  for i = #stats, 1, -1 do
    local stat_annotated = stats[i]
    local stat, stat_in_usage, stat_out_usage = unpack(stat_annotated)
    in_usage = usage_apply(in_usage, stat_in_usage)
  end

  return annotate(
    node { stats = result_stats },
    in_usage, out_usage)
end

function optimize_mapping.stat_if(cx, node)
  local then_cond_usage = analyze_usage(cx, node.cond)
  local elseif_cond_usage = node.elseif_blocks:map(
    function(block) return analyze_usage(cx, block.cond) end)

  local then_annotated = optimize_mapping.block(cx, node.then_block)
  local elseif_annotated = node.elseif_blocks:map(
    function(block) return optimize_mapping.stat_elseif(cx, block) end)
  local else_annotated = optimize_mapping.block(cx, node.else_block)

  local initial_usage = data.reduce(
    usage_meet,
    elseif_annotated:map(annotated_in_usage),
    usage_meet(usage_meet(annotated_in_usage(then_annotated),
                          annotated_in_usage(else_annotated)),
               data.reduce(usage_meet, elseif_cond_usage, then_cond_usage)))
  local final_usage = data.reduce(
    usage_meet,
    elseif_annotated:map(annotated_out_usage),
    usage_meet(annotated_out_usage(then_annotated),
               annotated_out_usage(else_annotated)))

  local then_block = fixup_block(then_annotated, initial_usage, final_usage)
  local elseif_blocks = elseif_annotated:map(
    function(block) return fixup_elseif(block, initial_usage, final_usage) end)
  local else_block = fixup_block(else_annotated, initial_usage, final_usage)

  return annotate(
    node {
      then_block = then_block,
      elseif_blocks = elseif_blocks,
      else_block = else_block,
    },
    initial_usage, final_usage)
end

function optimize_mapping.stat_elseif(cx, node)
  local block, in_usage, out_usage = unpack(optimize_mapping.block(cx, node.block))
  return annotate(
    node { block = block },
    in_usage, out_usage)
end

function optimize_mapping.stat_while(cx, node)
  local cond_usage = analyze_usage(cx, node.cond)
  local annotated_block = optimize_mapping.block(cx, node.block)
  local loop_usage = usage_meet(cond_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_mapping.stat_for_num(cx, node)
  local values_usage = analyze_usage(cx, node.values)
  local annotated_block = optimize_mapping.block(cx, node.block)
  local loop_usage = usage_meet(values_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_mapping.stat_for_list(cx, node)
  local value_usage = analyze_usage(cx, node.value)
  local annotated_block = optimize_mapping.block(cx, node.block)
  local loop_usage = usage_meet(value_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_mapping.stat_repeat(cx, node)
  local annotated_block = optimize_mapping.block(cx, node.block)
  local until_cond_usage = analyze_usage(cx, node.until_cond)
  local loop_usage = usage_meet(until_cond_usage,
                                annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_mapping.stat_must_epoch(cx, node)
  local block, block_in_usage, block_out_usage = unpack(
    optimize_mapping.block(cx, node.block))
  return annotate(
    node { block = block },
    block_in_usage, block_out_usage)
end

function optimize_mapping.stat_block(cx, node)
  local block, block_in_usage, block_out_usage = unpack(
    optimize_mapping.block(cx, node.block))
  return annotate(
    node { block = block },
    block_in_usage, block_out_usage)
end

function optimize_mapping.stat_index_launch_num(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_index_launch_list(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_var(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_var_unpack(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_return(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_break(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_assignment(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_reduce(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_expr(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat_raw_delete(cx, node)
  local usage = analyze_usage(cx, node)
  return annotate(node, usage, usage)
end

function optimize_mapping.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return optimize_mapping.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return optimize_mapping.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return optimize_mapping.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return optimize_mapping.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return optimize_mapping.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return optimize_mapping.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return optimize_mapping.stat_block(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return optimize_mapping.stat_index_launch_num(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return optimize_mapping.stat_index_launch_list(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return optimize_mapping.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return optimize_mapping.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return optimize_mapping.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return optimize_mapping.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    return optimize_mapping.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return optimize_mapping.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return optimize_mapping.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    return optimize_mapping.stat_raw_delete(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function task_initial_usage(cx, privileges)
  local usage = nil
  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      local region = privilege.region
      assert(std.type_supports_privileges(region:gettype()))
      usage = usage_meet(usage, uses(cx, region:gettype(), inline))
    end
  end
  return usage
end

function optimize_mapping.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope(
    node.prototype:get_constraints(),
    node.prototype:get_region_universe())
  local initial_usage = task_initial_usage(cx, node.privileges)
  local annotated_body = optimize_mapping.block(cx, node.body)
  local body = fixup_block(annotated_body, initial_usage, nil)

  return node { body = body }
end

function optimize_mapping.top(cx, node)
  if node:is(ast.typed.top.Task) and
     not node.config_options.inner
  then
    return optimize_mapping.top_task(cx, node)

  else
    return node
  end
end

function optimize_mapping.entry(node)
  local cx = context.new_global_scope({})
  return optimize_mapping.top(cx, node)
end

optimize_mapping.pass_name = "optimize_mapping"

return optimize_mapping
