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

-- Legion Task Config Option Optimizer
--
-- Determines which of the following config options are applicable to
-- a task:
--
--   * Leaf: Task issues no sub-operations
--   * Inner: Task does not access any regions
--   * Idempotent: Task has no external side-effects
--
-- (Currently the optimization returns false for idempotent.)

local ast = require("legion/ast")
local log = require("legion/log")
local std = require("legion/std")

local context = {}
context.__index = context

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local analyze_leaf = {}

function analyze_leaf.expr_field_access(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr_index_access(cx, node)
  return
    analyze_leaf.expr(cx, node.value) and
    analyze_leaf.expr(cx, node.index)
end

function analyze_leaf.expr_method_call(cx, node)
  return
    analyze_leaf.expr(cx, node.value) and
    std.all(node.args:map(function(arg) return analyze_leaf.expr(cx, arg) end))
end

function analyze_leaf.expr_call(cx, node)
  if std.is_task(node.fn.value) then
    return false
  end

  return
    analyze_leaf.expr(cx, node.fn) and
    std.all(node.args:map(function(arg) return analyze_leaf.expr(cx, arg) end))
end

function analyze_leaf.expr_cast(cx, node)
  return
    analyze_leaf.expr(cx, node.fn) and
    analyze_leaf.expr(cx, node.arg)
end

function analyze_leaf.expr_ctor(cx, node)
  return std.all(
    node.fields:map(function(field) return analyze_leaf.expr(cx, field.value) end))
end

function analyze_leaf.expr_raw_physical(cx, node)
  return analyze_leaf.expr(cx, node.region)
end

function analyze_leaf.expr_raw_fields(cx, node)
  return analyze_leaf.expr(cx, node.region)
end

function analyze_leaf.expr_isnull(cx, node)
  return analyze_leaf.expr(cx, node.pointer)
end

function analyze_leaf.expr_dynamic_cast(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr_static_cast(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr_unary(cx, node)
  return analyze_leaf.expr(cx, node.rhs)
end

function analyze_leaf.expr_binary(cx, node)
  return
    analyze_leaf.expr(cx, node.lhs) and
    analyze_leaf.expr(cx, node.rhs)
end

function analyze_leaf.expr_deref(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr_future(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr_future_get_result(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.expr(cx, node)
  if node:is(ast.typed.ExprID) then
    return true

  elseif node:is(ast.typed.ExprConstant) then
    return true

  elseif node:is(ast.typed.ExprFunction) then
    return true

  elseif node:is(ast.typed.ExprFieldAccess) then
    return analyze_leaf.expr_field_access(cx, node)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return analyze_leaf.expr_index_access(cx, node)

  elseif node:is(ast.typed.ExprMethodCall) then
    return analyze_leaf.expr_method_call(cx, node)

  elseif node:is(ast.typed.ExprCall) then
    return analyze_leaf.expr_call(cx, node)

  elseif node:is(ast.typed.ExprCast) then
    return analyze_leaf.expr_cast(cx, node)

  elseif node:is(ast.typed.ExprCtor) then
    return analyze_leaf.expr_ctor(cx, node)

  elseif node:is(ast.typed.ExprRawContext) then
    return false

  elseif node:is(ast.typed.ExprRawFields) then
    return analyze_leaf.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return analyze_leaf.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return true

  elseif node:is(ast.typed.ExprIsnull) then
    return analyze_leaf.expr_isnull(cx, node)

  elseif node:is(ast.typed.ExprNew) then
    return false

  elseif node:is(ast.typed.ExprNull) then
    return true

  elseif node:is(ast.typed.ExprDynamicCast) then
    return analyze_leaf.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.ExprStaticCast) then
    return analyze_leaf.expr_static_cast(cx, node)

  elseif node:is(ast.typed.ExprRegion) then
    return false

  elseif node:is(ast.typed.ExprPartition) then
    return false

  elseif node:is(ast.typed.ExprCrossProduct) then
    return false

  elseif node:is(ast.typed.ExprUnary) then
    return analyze_leaf.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return analyze_leaf.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return analyze_leaf.expr_deref(cx, node)

  elseif node:is(ast.typed.ExprFuture) then
    return analyze_leaf.expr_future(cx, node)

  elseif node:is(ast.typed.ExprFutureGetResult) then
    return analyze_leaf.expr_future_get_result(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_leaf.block(cx, node)
  return std.all(
    node.stats:map(function(stat) return analyze_leaf.stat(cx, stat) end))
end

function analyze_leaf.stat_if(cx, node)
  return
    analyze_leaf.expr(cx, node.cond) and
    analyze_leaf.block(cx, node.then_block) and
    std.all(
      node.elseif_blocks:map(
        function(block) return analyze_leaf.stat_elseif(cx, block) end)) and
    analyze_leaf.block(cx, node.else_block)
end

function analyze_leaf.stat_elseif(cx, node)
  return
    analyze_leaf.expr(cx, node.cond) and
    analyze_leaf.block(cx, node.block)
end

function analyze_leaf.stat_while(cx, node)
  return
    analyze_leaf.expr(cx, node.cond) and
    analyze_leaf.block(cx, node.block)
end

function analyze_leaf.stat_for_num(cx, node)
  return
    std.all(
      node.values:map(function(value) return analyze_leaf.expr(cx, value) end)) and
    analyze_leaf.block(cx, node.block)
end

function analyze_leaf.stat_for_list(cx, node)
  return
    analyze_leaf.expr(cx, node.value) and
    analyze_leaf.block(cx, node.block)
end

function analyze_leaf.stat_repeat(cx, node)
  return
    analyze_leaf.block(cx, node.block) and
    analyze_leaf.expr(cx, node.until_cond)
end

function analyze_leaf.stat_block(cx, node)
  return analyze_leaf.block(cx, node.block)
end

function analyze_leaf.stat_index_launch(cx, node)
  return false
end

function analyze_leaf.stat_var(cx, node)
  return std.all(
    node.values:map(function(value) return analyze_leaf.expr(cx, value) end))
end

function analyze_leaf.stat_var_unpack(cx, node)
  return analyze_leaf.expr(cx, node.value)
end

function analyze_leaf.stat_return(cx, node)
  if node.value then
    return analyze_leaf.expr(cx, node.value)
  else
    return true
  end
end

function analyze_leaf.stat_break(cx, node)
  return true
end

function analyze_leaf.stat_assignment(cx, node)
  return
    std.all(
      node.lhs:map(function(lh) return analyze_leaf.expr(cx, lh) end)) and
    std.all(
      node.rhs:map(function(rh) return analyze_leaf.expr(cx, rh) end))
end

function analyze_leaf.stat_reduce(cx, node)
  return
    std.all(
      node.lhs:map(function(lh) return analyze_leaf.expr(cx, lh) end)) and
    std.all(
      node.rhs:map(function(rh) return analyze_leaf.expr(cx, rh) end))
end

function analyze_leaf.stat_expr(cx, node)
  return analyze_leaf.expr(cx, node.expr)
end

function analyze_leaf.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return analyze_leaf.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return analyze_leaf.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return analyze_leaf.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return analyze_leaf.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return analyze_leaf.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return analyze_leaf.stat_block(cx, node)

  elseif node:is(ast.typed.StatIndexLaunch) then
    return analyze_leaf.stat_index_launch(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return analyze_leaf.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return analyze_leaf.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return analyze_leaf.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return analyze_leaf.stat_break(cx, node)

  elseif node:is(ast.typed.StatAssignment) then
    return analyze_leaf.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return analyze_leaf.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return analyze_leaf.stat_expr(cx, node)

  elseif node:is(ast.typed.StatMapRegions) then
    return false

  elseif node:is(ast.typed.StatUnmapRegions) then
    return true

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local analyze_inner = {}

function analyze_inner.expr_field_access(cx, node)
  return
    not std.is_ptr(std.as_read(node.value.expr_type)) and
    analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr_index_access(cx, node)
  return
    analyze_inner.expr(cx, node.value) and
    analyze_inner.expr(cx, node.index)
end

function analyze_inner.expr_method_call(cx, node)
  return
    analyze_inner.expr(cx, node.value) and
    std.all(
      node.args:map(function(arg) return analyze_inner.expr(cx, arg) end))
end

function analyze_inner.expr_call(cx, node)
  return
    analyze_leaf.expr(cx, node.fn) and
    std.all(
      node.args:map(function(arg) return analyze_inner.expr(cx, arg) end))
end

function analyze_inner.expr_cast(cx, node)
  return
    analyze_inner.expr(cx, node.fn) and
    analyze_inner.expr(cx, node.arg)
end

function analyze_inner.expr_ctor(cx, node)
  return std.all(
    node.fields:map(function(field) return analyze_inner.expr(cx, field.value) end))
end

function analyze_inner.expr_raw_fields(cx, node)
  return analyze_inner.expr(cx, node.region)
end

function analyze_inner.expr_isnull(cx, node)
  return analyze_inner.expr(cx, node.pointer)
end

function analyze_inner.expr_dynamic_cast(cx, node)
  return analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr_static_cast(cx, node)
  return analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr_region(cx, node)
  return analyze_inner.expr(cx, node.size)
end

function analyze_inner.expr_partition(cx, node)
  return analyze_inner.expr(cx, node.coloring)
end

function analyze_inner.expr_cross_product(cx, node)
  return
    analyze_inner.expr(cx, node.lhs) and
    analyze_inner.expr(cx, node.rhs)
end

function analyze_inner.expr_unary(cx, node)
  return analyze_inner.expr(cx, node.rhs)
end

function analyze_inner.expr_binary(cx, node)
  return
    analyze_inner.expr(cx, node.lhs) and
    analyze_inner.expr(cx, node.rhs)
end

function analyze_inner.expr_deref(cx, node)
  return
    not std.is_ptr(std.as_read(node.value.expr_type)) and
    analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr_future(cx, node)
  return analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr_future_get_result(cx, node)
  return analyze_inner.expr(cx, node.value)
end

function analyze_inner.expr(cx, node)
  if node:is(ast.typed.ExprID) then
    return true

  elseif node:is(ast.typed.ExprConstant) then
    return true

  elseif node:is(ast.typed.ExprFunction) then
    return true

  elseif node:is(ast.typed.ExprFieldAccess) then
    return analyze_inner.expr_field_access(cx, node)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return analyze_inner.expr_index_access(cx, node)

  elseif node:is(ast.typed.ExprMethodCall) then
    return analyze_inner.expr_method_call(cx, node)

  elseif node:is(ast.typed.ExprCall) then
    return analyze_inner.expr_call(cx, node)

  elseif node:is(ast.typed.ExprCast) then
    return analyze_inner.expr_cast(cx, node)

  elseif node:is(ast.typed.ExprCtor) then
    return analyze_inner.expr_ctor(cx, node)

  elseif node:is(ast.typed.ExprRawContext) then
    return true

  elseif node:is(ast.typed.ExprRawFields) then
    return analyze_inner.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return false

  elseif node:is(ast.typed.ExprRawRuntime) then
    return true

  elseif node:is(ast.typed.ExprIsnull) then
    return analyze_inner.expr_isnull(cx, node)

  elseif node:is(ast.typed.ExprNew) then
    return true

  elseif node:is(ast.typed.ExprNull) then
    return true

  elseif node:is(ast.typed.ExprDynamicCast) then
    return analyze_inner.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.ExprStaticCast) then
    return analyze_inner.expr_static_cast(cx, node)

  elseif node:is(ast.typed.ExprRegion) then
    return analyze_inner.expr_region(cx, node)

  elseif node:is(ast.typed.ExprPartition) then
    return analyze_inner.expr_partition(cx, node)

  elseif node:is(ast.typed.ExprCrossProduct) then
    return analyze_inner.expr_cross_product(cx, node)

  elseif node:is(ast.typed.ExprUnary) then
    return analyze_inner.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return analyze_inner.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return analyze_inner.expr_deref(cx, node)

  elseif node:is(ast.typed.ExprFuture) then
    return analyze_inner.expr_future(cx, node)

  elseif node:is(ast.typed.ExprFutureGetResult) then
    return analyze_inner.expr_future_get_result(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_inner.block(cx, node)
  return std.all(
    node.stats:map(function(stat) return analyze_inner.stat(cx, stat) end))
end

function analyze_inner.stat_if(cx, node)
  return
    analyze_inner.expr(cx, node.cond) and
    analyze_inner.block(cx, node.then_block) and
    std.all(
      node.elseif_blocks:map(
        function(block) return analyze_inner.stat_elseif(cx, block) end)) and
    analyze_inner.block(cx, node.else_block)
end

function analyze_inner.stat_elseif(cx, node)
  return
    analyze_inner.expr(cx, node.cond) and
    analyze_inner.block(cx, node.block)
end

function analyze_inner.stat_while(cx, node)
  return
    analyze_inner.expr(cx, node.cond) and
    analyze_inner.block(cx, node.block)
end

function analyze_inner.stat_for_num(cx, node)
  return
    std.all(
      node.values:map(function(value) return analyze_inner.expr(cx, value) end)) and
    analyze_inner.block(cx, node.block)
end

function analyze_inner.stat_for_list(cx, node)
  return
    analyze_inner.expr(cx, node.value) and
    analyze_inner.block(cx, node.block)
end

function analyze_inner.stat_repeat(cx, node)
  return
    analyze_inner.block(cx, node.block) and
    analyze_inner.expr(cx, node.until_cond)
end

function analyze_inner.stat_block(cx, node)
  return analyze_inner.block(cx, node.block)
end

function analyze_inner.stat_index_launch(cx, node)
  return
    std.all(node.domain:map(function(value) return analyze_inner.expr(cx, value) end)) and
    analyze_inner.expr(cx, node.call) and
    (node.reduce_lhs and analyze_inner.expr(cx, node.reduce_lhs))
end

function analyze_inner.stat_var(cx, node)
  return std.all(
    node.values:map(function(value) return analyze_inner.expr(cx, value) end))
end

function analyze_inner.stat_var_unpack(cx, node)
  return analyze_inner.expr(cx, node.value)
end

function analyze_inner.stat_return(cx, node) 
  if node.value then
    return analyze_inner.expr(cx, node.value)
  else
    return true
  end
end

function analyze_inner.stat_break(cx, node)
  return true
end

function analyze_inner.stat_assignment(cx, node)
  return
    std.all(
      node.lhs:map(function(lh) return analyze_inner.expr(cx, lh) end)) and
    std.all(
      node.rhs:map(function(rh) return analyze_inner.expr(cx, rh) end))
end

function analyze_inner.stat_reduce(cx, node)
  return
    std.all(
      node.lhs:map(function(lh) return analyze_inner.expr(cx, lh) end)) and
    std.all(
      node.rhs:map(function(rh) return analyze_inner.expr(cx, rh) end))
end

function analyze_inner.stat_expr(cx, node)
  return analyze_inner.expr(cx, node.expr)
end

function analyze_inner.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return analyze_inner.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return analyze_inner.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return analyze_inner.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return analyze_inner.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return analyze_inner.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return analyze_inner.stat_block(cx, node)

  elseif node:is(ast.typed.StatIndexLaunch) then
    return analyze_inner.stat_index_launch(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return analyze_inner.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return analyze_inner.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return analyze_inner.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return analyze_inner.stat_break(cx, node)

  elseif node:is(ast.typed.StatAssignment) then
    return analyze_inner.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return analyze_inner.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return analyze_inner.stat_expr(cx, node)

  elseif node:is(ast.typed.StatMapRegions) then
    return false

  elseif node:is(ast.typed.StatUnmapRegions) then
    return true

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local optimize_config_options = {}

function optimize_config_options.stat_task(cx, node)
  local leaf = analyze_leaf.block(cx, node.body)
  local inner = not leaf and analyze_inner.block(cx, node.body)

  return node {
    config_options = ast.typed.StatTaskConfigOptions {
      leaf = leaf,
      inner = inner,
      idempotent = false,
    },
  }
end

function optimize_config_options.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return optimize_config_options.stat_task(cx, node)

  else
    return node
  end
end

function optimize_config_options.entry(node)
  local cx = context.new_global_scope()
  return optimize_config_options.stat_top(cx, node)
end

return optimize_config_options
