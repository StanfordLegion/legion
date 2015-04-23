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

-- Legion Divergence Optimizer
--
-- This pass analyzes the code for divergence resulting from
-- multi-region pointer accesses. Regions accessed in multi-pointer
-- derefs are marked to facilitate dynamic branch elision in code
-- generation.

local ast = require("legion/ast")
local std = require("legion/std")
local union_find = require("legion/union_find")

local context = {}
context.__index = context

function context:new_task_scope()
  local cx = {
    region_div = union_find.new(),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:mark_region_divergence(...)
  local rs = {...}
  if #rs > 0 then
    local r1 = rs[1]
    for _, r in ipairs(rs) do
      self.region_div:union_keys(r1, r)
    end
  end
end

local analyze_region_divergence = {}

function analyze_region_divergence.expr_field_access(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  if std.is_ptr(value_type) and #value_type:points_to_regions() > 1 then
    cx:mark_region_divergence(unpack(value_type:points_to_regions()))
  end
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr_index_access(cx, node)
  analyze_region_divergence.expr(cx, node.value)
  analyze_region_divergence.expr(cx, node.index)
end

function analyze_region_divergence.expr_method_call(cx, node)
  analyze_region_divergence.expr(cx, node.value)
  node.args:map(function(arg) analyze_region_divergence.expr(cx, arg) end)
end

function analyze_region_divergence.expr_call(cx, node)
  analyze_region_divergence.expr(cx, node.fn)
  node.args:map(function(arg) analyze_region_divergence.expr(cx, arg) end)
end

function analyze_region_divergence.expr_cast(cx, node)
  analyze_region_divergence.expr(cx, node.fn)
  analyze_region_divergence.expr(cx, node.arg)
end

function analyze_region_divergence.expr_ctor(cx, node)
  node.fields:map(function(field) analyze_region_divergence.expr(cx, field.value) end)
end

function analyze_region_divergence.expr_raw_physical(cx, node)
  analyze_region_divergence.expr(cx, node.region)
end

function analyze_region_divergence.expr_raw_fields(cx, node)
  analyze_region_divergence.expr(cx, node.region)
end

function analyze_region_divergence.expr_isnull(cx, node)
  analyze_region_divergence.expr(cx, node.pointer)
end

function analyze_region_divergence.expr_dynamic_cast(cx, node)
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr_static_cast(cx, node)
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr_unary(cx, node)
  analyze_region_divergence.expr(cx, node.rhs)
end

function analyze_region_divergence.expr_binary(cx, node)
  analyze_region_divergence.expr(cx, node.lhs)
  analyze_region_divergence.expr(cx, node.rhs)
end

function analyze_region_divergence.expr_deref(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  if std.is_ptr(value_type) and #value_type:points_to_regions() > 1 then
    cx:mark_region_divergence(unpack(value_type:points_to_regions()))
  end
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr_future(cx, node)
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr_future_get_result(cx, node)
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.expr(cx, node)
  if node:is(ast.typed.ExprID) then
    return

  elseif node:is(ast.typed.ExprConstant) then
    return

  elseif node:is(ast.typed.ExprFunction) then
    return

  elseif node:is(ast.typed.ExprFieldAccess) then
    return analyze_region_divergence.expr_field_access(cx, node)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return analyze_region_divergence.expr_index_access(cx, node)

  elseif node:is(ast.typed.ExprMethodCall) then
    return analyze_region_divergence.expr_method_call(cx, node)

  elseif node:is(ast.typed.ExprCall) then
    return analyze_region_divergence.expr_call(cx, node)

  elseif node:is(ast.typed.ExprCast) then
    return analyze_region_divergence.expr_cast(cx, node)

  elseif node:is(ast.typed.ExprCtor) then
    return analyze_region_divergence.expr_ctor(cx, node)

  elseif node:is(ast.typed.ExprRawContext) then
    return

  elseif node:is(ast.typed.ExprRawFields) then
    return analyze_region_divergence.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return analyze_region_divergence.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return

  elseif node:is(ast.typed.ExprIsnull) then
    return analyze_region_divergence.expr_isnull(cx, node)

  elseif node:is(ast.typed.ExprNew) then
    return

  elseif node:is(ast.typed.ExprNull) then
    return

  elseif node:is(ast.typed.ExprDynamicCast) then
    return analyze_region_divergence.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.ExprStaticCast) then
    return analyze_region_divergence.expr_static_cast(cx, node)

  elseif node:is(ast.typed.ExprRegion) then
    return

  elseif node:is(ast.typed.ExprPartition) then
    return

  elseif node:is(ast.typed.ExprCrossProduct) then
    return

  elseif node:is(ast.typed.ExprUnary) then
    return analyze_region_divergence.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return analyze_region_divergence.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return analyze_region_divergence.expr_deref(cx, node)

  elseif node:is(ast.typed.ExprFuture) then
    return analyze_region_divergence.expr_future(cx, node)

  elseif node:is(ast.typed.ExprFutureGetResult) then
    return analyze_region_divergence.expr_future_get_result(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_region_divergence.block(cx, node)
  node.stats:map(function(stat) analyze_region_divergence.stat(cx, stat) end)
end

function analyze_region_divergence.stat_if(cx, node)
  analyze_region_divergence.expr(cx, node.cond)
  analyze_region_divergence.block(cx, node.then_block)
  node.elseif_blocks:map(
    function(block) analyze_region_divergence.stat_elseif(cx, block) end)
  analyze_region_divergence.block(cx, node.else_block)
end

function analyze_region_divergence.stat_elseif(cx, node)
  analyze_region_divergence.expr(cx, node.cond)
  analyze_region_divergence.block(cx, node.block)
end

function analyze_region_divergence.stat_while(cx, node)
  analyze_region_divergence.expr(cx, node.cond)
  analyze_region_divergence.block(cx, node.block)
end

function analyze_region_divergence.stat_for_num(cx, node)
  node.values:map(function(value) analyze_region_divergence.expr(cx, value) end)
  analyze_region_divergence.block(cx, node.block)
end

function analyze_region_divergence.stat_for_list(cx, node)
  analyze_region_divergence.expr(cx, node.value)
  analyze_region_divergence.block(cx, node.block)
end

function analyze_region_divergence.stat_repeat(cx, node)
  analyze_region_divergence.block(cx, node.block)
  analyze_region_divergence.expr(cx, node.until_cond)
end

function analyze_region_divergence.stat_block(cx, node)
  analyze_region_divergence.block(cx, node.block)
end

function analyze_region_divergence.stat_index_launch(cx, node)
  analyze_region_divergence.expr(cx, node.call)
  if node.reduce_lhs then
    analyze_region_divergence.expr(cx, node.reduce_lhs)
  end
end

function analyze_region_divergence.stat_var(cx, node)
  node.values:map(function(value) analyze_region_divergence.expr(cx, value) end)
end

function analyze_region_divergence.stat_var_unpack(cx, node)
  analyze_region_divergence.expr(cx, node.value)
end

function analyze_region_divergence.stat_return(cx, node)
  if node.value then
    analyze_region_divergence.expr(cx, node.value)
  end
end

function analyze_region_divergence.stat_assignment(cx, node)
  node.lhs:map(function(lh) analyze_region_divergence.expr(cx, lh) end)
  node.rhs:map(function(rh) analyze_region_divergence.expr(cx, rh) end)
end

function analyze_region_divergence.stat_reduce(cx, node)
  node.lhs:map(function(lh) analyze_region_divergence.expr(cx, lh) end)
  node.rhs:map(function(rh) analyze_region_divergence.expr(cx, rh) end)
end

function analyze_region_divergence.stat_expr(cx, node)
  analyze_region_divergence.expr(cx, node.expr)
end

function analyze_region_divergence.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    analyze_region_divergence.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    analyze_region_divergence.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    analyze_region_divergence.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    analyze_region_divergence.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    analyze_region_divergence.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    analyze_region_divergence.stat_block(cx, node)

  elseif node:is(ast.typed.StatIndexLaunch) then
    analyze_region_divergence.stat_index_launch(cx, node)

  elseif node:is(ast.typed.StatVar) then
    analyze_region_divergence.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    analyze_region_divergence.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    analyze_region_divergence.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return

  elseif node:is(ast.typed.StatAssignment) then
    analyze_region_divergence.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    analyze_region_divergence.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    analyze_region_divergence.stat_expr(cx, node)

  elseif node:is(ast.typed.StatMapRegions) then
    return

  elseif node:is(ast.typed.StatUnmapRegions) then
    return

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local function invert_forest(forest)
  local result = {}
  local ks = forest:keys()
  for _, k in ipairs(ks) do
    local root = forest:find_key(k)
    if not rawget(result, root) then
      result[root] = terralib.newlist()
    end
    result[root]:insert(k)
  end
  return result
end

local optimize_divergence = {}

function optimize_divergence.stat_task(cx, node)
  local cx = cx:new_task_scope()
  analyze_region_divergence.block(cx, node.body)
  local divergence = invert_forest(cx.region_div)

  return ast.typed.StatTask {
    name = node.name,
    params = node.params,
    return_type = node.return_type,
    privileges = node.privileges,
    constraints = node.constraints,
    body = node.body,
    config_options = node.config_options,
    region_divergence = divergence,
    prototype = node.prototype,
    span = node.span,
  }
end

function optimize_divergence.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return optimize_divergence.stat_task(cx, node)

  else
    return node
  end
end

function optimize_divergence.entry(node)
  local cx = context.new_global_scope()
  return optimize_divergence.stat_top(cx, node)
end

return optimize_divergence
