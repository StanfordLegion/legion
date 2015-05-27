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

-- Legion Inline Optimizer
--
-- Attempts to place map/unmap calls to avoid thrashing inlines.

local ast = require("legion/ast")
local std = require("legion/std")

local context = {}
context.__index = context

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

local region_usage = {}
region_usage.__index = region_usage

function region_usage:__tostring()
  local result = "region_usage(\n"
  for region_type, polarity in pairs(self) do
    result = result .. "  " .. tostring(region_type) .. " = " .. tostring(polarity) .. ",\n"
  end
  result = result .. ")"
  return result
end

function uses(cx, region_type, polarity)
  -- In order for this to be sound, we need to unmap *all* regions
  -- that could potentially alias with this one, not just just the
  -- region itself. Not all of these regions will necessarily be
  -- mapped, but codegen will be able to do the right thing by
  -- ignoring anything that isn't a root in the region forest.

  local usage = { [region_type] = polarity }
  for other_region_type, _ in pairs(cx.region_universe) do
    local constraint = {
      lhs = region_type,
      rhs = other_region_type,
      op = "*"
    }
    if std.type_maybe_eq(region_type.element_type, other_region_type.element_type) and
      not std.check_constraint(cx, constraint)
    then
      usage[other_region_type] = polarity
    end
  end
  return setmetatable(usage, region_usage)
end

function usage_meet_polarity(a, b)
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

function usage_meet(...)
  local usage = {}
  for _, a in pairs({...}) do
    if a then
      for region_type, polarity in pairs(a) do
        usage[region_type] = usage_meet_polarity(usage[region_type], polarity)
      end
    end
  end
  return setmetatable(usage, region_usage)
end

function usage_diff_polarity(a, b)
  if not a or not b or a == b then
    return nil
  else
    return b
  end
end

function usage_diff(a, b)
  if not a or not b then
    return nil
  end
  local usage = {}
  for region_type, a_polarity in pairs(a) do
    local b_polarity = b[region_type]
    local diff = usage_diff_polarity(a_polarity, b_polarity)
    if diff then
      usage[region_type] = diff
    end
  end
  return setmetatable(usage, region_usage)
end

function usage_apply_polarity(a, b)
  if not a then
    return b
  elseif not b then
    return a
  else
    return b
  end
end

function usage_apply(...)
  local usage = {}
  for _, a in pairs({...}) do
    if a then
      for region_type, polarity in pairs(a) do
        usage[region_type] = usage_apply_polarity(usage[region_type], polarity)
      end
    end
  end
  return setmetatable(usage, region_usage)
end

local analyze_usage = {}

function analyze_usage.expr_field_access(cx, node)
  return analyze_usage.expr(cx, node.value)
end

function analyze_usage.expr_index_access(cx, node)
  return usage_meet(
    analyze_usage.expr(cx, node.value),
    analyze_usage.expr(cx, node.index))
end

function analyze_usage.expr_method_call(cx, node)
  local usage = analyze_usage.expr(cx, node.value)
  for _, arg in ipairs(node.args) do
    usage = usage_meet(usage, analyze_usage.expr(cx, arg))
  end
  return usage
end

function analyze_usage.expr_call(cx, node)
  local is_task = std.is_task(node.fn.value)
  local usage = analyze_usage.expr(cx, node.fn)
  for _, arg in ipairs(node.args) do
    local arg_type = std.as_read(arg.expr_type)
    usage = usage_meet(usage, analyze_usage.expr(cx, arg))
    if is_task and std.is_region(arg_type) then
      usage = usage_meet(usage, uses(cx, arg_type, remote))
    end
  end
  return usage
end

function analyze_usage.expr_cast(cx, node)
  return usage_meet(analyze_usage.expr(cx, node.fn),
                    analyze_usage.expr(cx, node.arg))
end

function analyze_usage.expr_ctor(cx, node)
  local usage = nil
  for _, field in ipairs(node.fields) do
    usage = usage_meet(usage, analyze_usage.expr(cx, field.value))
  end
  return usage
end

function analyze_usage.expr_raw_physical(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  return usage_meet(analyze_usage.expr(cx, node.region),
                    uses(cx, region_type, inline))
end

function analyze_usage.expr_raw_fields(cx, node)
  return analyze_usage.expr(cx, node.region)
end

function analyze_usage.expr_isnull(cx, node)
  return analyze_usage.expr(cx, node.pointer)
end

function analyze_usage.expr_dynamic_cast(cx, node)
  return analyze_usage.expr(cx, node.value)
end

function analyze_usage.expr_static_cast(cx, node)
  return analyze_usage.expr(cx, node.value)
end

function analyze_usage.expr_ispace(cx, node)
  return usage_meet(
    analyze_usage.expr(cx, node.lower_bound),
    node.upper_bound and analyze_usage.expr(cx, node.upper_bound))
end

function analyze_usage.expr_region(cx, node)
  return analyze_usage.expr(cx, node.size)
end

function analyze_usage.expr_partition(cx, node)
  return analyze_usage.expr(cx, node.coloring)
end

function analyze_usage.expr_cross_product(cx, node)
  return usage_meet(analyze_usage.expr(cx, node.lhs),
                    analyze_usage.expr(cx, node.rhs))
end

function analyze_usage.expr_unary(cx, node)
  return analyze_usage.expr(cx, node.rhs)
end

function analyze_usage.expr_binary(cx, node)
  return usage_meet(analyze_usage.expr(cx, node.lhs),
                    analyze_usage.expr(cx, node.rhs))
end

function analyze_usage.expr_deref(cx, node)
  local ptr_type = std.as_read(node.value.expr_type)
  return std.reduce(
    usage_meet,
    ptr_type:points_to_regions():map(
      function(region) return uses(cx, region, inline) end),
    analyze_usage.expr(cx, node.value))
end

function analyze_usage.expr_future(cx, node)
  return analyze_usage.expr(cx, node.value)
end

function analyze_usage.expr_future_get_result(cx, node)
  return analyze_usage.expr(cx, node.value)
end

function analyze_usage.expr(cx, node)
  if node:is(ast.typed.ExprID) then
    return nil

  elseif node:is(ast.typed.ExprConstant) then
    return nil

  elseif node:is(ast.typed.ExprFunction) then
    return nil

  elseif node:is(ast.typed.ExprFieldAccess) then
    return analyze_usage.expr_field_access(cx, node)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return analyze_usage.expr_index_access(cx, node)

  elseif node:is(ast.typed.ExprMethodCall) then
    return analyze_usage.expr_method_call(cx, node)

  elseif node:is(ast.typed.ExprCall) then
    return analyze_usage.expr_call(cx, node)

  elseif node:is(ast.typed.ExprCast) then
    return analyze_usage.expr_cast(cx, node)

  elseif node:is(ast.typed.ExprCtor) then
    return analyze_usage.expr_ctor(cx, node)

  elseif node:is(ast.typed.ExprRawContext) then
    return nil

  elseif node:is(ast.typed.ExprRawFields) then
    return analyze_usage.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return analyze_usage.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return nil

  elseif node:is(ast.typed.ExprIsnull) then
    return analyze_usage.expr_isnull(cx, node)

  elseif node:is(ast.typed.ExprNew) then
    return nil

  elseif node:is(ast.typed.ExprNull) then
    return nil

  elseif node:is(ast.typed.ExprDynamicCast) then
    return analyze_usage.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.ExprStaticCast) then
    return analyze_usage.expr_static_cast(cx, node)

  elseif node:is(ast.typed.ExprIspace) then
    return analyze_usage.expr_ispace(cx, node)

  elseif node:is(ast.typed.ExprRegion) then
    return analyze_usage.expr_region(cx, node)

  elseif node:is(ast.typed.ExprPartition) then
    return analyze_usage.expr_partition(cx, node)

  elseif node:is(ast.typed.ExprCrossProduct) then
    return analyze_usage.expr_cross_product(cx, node)

  elseif node:is(ast.typed.ExprUnary) then
    return analyze_usage.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return analyze_usage.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return analyze_usage.expr_deref(cx, node)

  elseif node:is(ast.typed.ExprFuture) then
    return analyze_usage.expr_future(cx, node)

  elseif node:is(ast.typed.ExprFutureGetResult) then
    return analyze_usage.expr_future_get_result(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end


local optimize_inlines = {}

function annotate(node, in_usage, out_usage)
  return { node, in_usage, out_usage }
end

function annotated_in_usage(annotated_node)
  return annotated_node[2]
end

function annotated_out_usage(annotated_node)
  return annotated_node[3]
end

function map_regions(diff)
  local result = terralib.newlist()
  if diff then
    local region_types_by_polarity = {}
    for region_type, polarity in pairs(diff) do
      if not region_types_by_polarity[polarity] then
        region_types_by_polarity[polarity] = terralib.newlist()
      end
      region_types_by_polarity[polarity]:insert(region_type)
    end
    for polarity, region_types in pairs(region_types_by_polarity) do
      if polarity == inline then
        result:insert(
          ast.typed.StatMapRegions {
            region_types = region_types,
          })
      elseif polarity == remote then
        result:insert(
          ast.typed.StatUnmapRegions {
            region_types = region_types
          })
      else
        assert(false)
      end
    end
  end
  return result
end

function fixup_block(annotated_block, in_usage, out_usage)
  local node, node_in_usage, node_out_usage = unpack(annotated_block)
  local stats = terralib.newlist()
  stats:insertall(map_regions(usage_diff(in_usage, node_in_usage)))
  stats:insertall(node.stats)
  stats:insertall(map_regions(usage_diff(node_out_usage, out_usage)))
  return node { stats = stats }
end

function fixup_elseif(annotated_node, in_usage, out_usage)
  local node, node_in_usage, node_out_usage = unpack(annotated_node)
  local annotated_block = annotate(node.block, node_in_usage, node_out_usage)
  local block = fixup_block(annotated_block, in_usage, out_usage)
  return node { block = block }
end

function optimize_inlines.block(cx, node)
  local stats = node.stats:map(
    function(stat) return optimize_inlines.stat(cx, stat) end)

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

function optimize_inlines.stat_if(cx, node)
  local then_cond_usage = analyze_usage.expr(cx, node.cond)
  local elseif_cond_usage = node.elseif_blocks:map(
    function(block) return analyze_usage.expr(cx, block.cond) end)

  local then_annotated = optimize_inlines.block(cx, node.then_block)
  local elseif_annotated = node.elseif_blocks:map(
    function(block) return optimize_inlines.stat_elseif(cx, block) end)
  local else_annotated = optimize_inlines.block(cx, node.else_block)

  local initial_usage = std.reduce(usage_meet, elseif_cond_usage, then_cond_usage)
  local final_usage = std.reduce(
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

function optimize_inlines.stat_elseif(cx, node)
  local block, in_usage, out_usage = unpack(optimize_inlines.block(cx, node.block))
  return annotate(
    node { block = block },
    in_usage, out_usage)
end

function optimize_inlines.stat_while(cx, node)
  local cond_usage = analyze_usage.expr(cx, node.cond)
  local annotated_block = optimize_inlines.block(cx, node.block)
  local loop_usage = usage_meet(cond_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_inlines.stat_for_num(cx, node)
  local values_usage = std.reduce(
    usage_meet,
    node.values:map(function(value) return analyze_usage.expr(cx, value) end))
  local annotated_block = optimize_inlines.block(cx, node.block)
  local loop_usage = usage_meet(values_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_inlines.stat_for_list(cx, node)
  local value_usage = analyze_usage.expr(cx, node.value)
  local annotated_block = optimize_inlines.block(cx, node.block)
  local loop_usage = usage_meet(value_usage, annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_inlines.stat_repeat(cx, node)
  local annotated_block = optimize_inlines.block(cx, node.block)
  local until_cond_usage = analyze_usage.expr(cx, node.until_cond)
  local loop_usage = usage_meet(until_cond_usage,
                                annotated_in_usage(annotated_block))
  local block = fixup_block(annotated_block, loop_usage, loop_usage)
  return annotate(
    node { block = block },
    loop_usage, loop_usage)
end

function optimize_inlines.stat_block(cx, node)
  local block, block_in_usage, block_out_usage = unpack(
    optimize_inlines.block(cx, node.block))
  return annotate(
    node { block = block },
    block_in_usage, block_out_usage)
end

function optimize_inlines.stat_index_launch(cx, node)
  local domain_usage = std.reduce(
    usage_meet,
    node.domain:map(function(value) return analyze_usage.expr(cx, value) end))
  local reduce_lhs_usage = (node.reduce_lhs and
                              analyze_usage.expr(cx, node.reduce_lhs))
  local call_usage = analyze_usage.expr(cx, node.call)
  local usage = usage_meet(domain_usage, reduce_lhs_usage, call_usage)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_var(cx, node)
  local usage = nil
  for _, value in ipairs(node.values) do
    usage = usage_meet(usage, analyze_usage.expr(cx, value))
  end
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_var_unpack(cx, node)
  local usage = analyze_usage.expr(cx, node.value)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_return(cx, node)
  local usage = node.value and analyze_usage.expr(cx, node.value)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_break(cx, node)
  return annotate(node, nil, nil)
end

function optimize_inlines.stat_assignment(cx, node)
  local usage = std.reduce(
    usage_meet,
    node.lhs:map(function(lh) return analyze_usage.expr(cx, lh) end))
  usage = std.reduce(
    usage_meet,
    node.rhs:map(function(rh) return analyze_usage.expr(cx, rh) end),
    usage)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_reduce(cx, node)
  local usage = std.reduce(
    usage_meet,
    node.lhs:map(function(lh) return analyze_usage.expr(cx, lh) end))
  usage = std.reduce(
    usage_meet,
    node.rhs:map(function(rh) return analyze_usage.expr(cx, rh) end),
    usage)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat_expr(cx, node)
  local usage = analyze_usage.expr(cx, node.expr)
  return annotate(node, usage, usage)
end

function optimize_inlines.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return optimize_inlines.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return optimize_inlines.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return optimize_inlines.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return optimize_inlines.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return optimize_inlines.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return optimize_inlines.stat_block(cx, node)

  elseif node:is(ast.typed.StatIndexLaunch) then
    return optimize_inlines.stat_index_launch(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return optimize_inlines.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return optimize_inlines.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return optimize_inlines.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return optimize_inlines.stat_break(cx, node)

  elseif node:is(ast.typed.StatAssignment) then
    return optimize_inlines.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return optimize_inlines.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return optimize_inlines.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function task_initial_usage(cx, privileges)
  local usage = nil
  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      local region = privilege.region
      assert(std.is_region(region.type))
      usage = usage_meet(usage, uses(cx, region.type, inline))
    end
  end
  return usage
end

function optimize_inlines.stat_task(cx, node)
  local cx = cx:new_task_scope(
    node.prototype:get_constraints(),
    node.prototype:get_region_universe())
  local initial_usage = task_initial_usage(cx, node.privileges)
  local annotated_body = optimize_inlines.block(cx, node.body)
  local body = fixup_block(annotated_body, initial_usage, nil)

  return node { body = body }
end

function optimize_inlines.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return optimize_inlines.stat_task(cx, node)

  else
    return node
  end
end

function optimize_inlines.entry(node)
  local cx = context.new_global_scope({})
  return optimize_inlines.stat_top(cx, node)
end

return optimize_inlines
