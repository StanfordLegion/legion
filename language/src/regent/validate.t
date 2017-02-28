-- Copyright 2017 Stanford University, NVIDIA Corporation
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

-- Regent AST Validator

local ast = require("regent/ast")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local context = {}
context.__index = context

function context.new_global_scope(env)
  local cx = {
    env = terralib.newlist({symbol_table.new_global_scope(env)}),
  }
  setmetatable(cx, context)
  return cx
end

function context:push_local_scope()
  self.env:insert(self.env[#self.env]:new_local_scope())
end

function context:pop_local_scope()
  assert(self.env:remove())
end

function context:intern_variable(node, symbol)
  assert(ast.is_node(node))
  if not std.is_symbol(symbol) then
    report.error(node, "expected a symbol, got " .. tostring(symbol))
  end
  self.env[#self.env]:insert(node, symbol, symbol)
end

function context:intern_variables(node, symbols)
  assert(ast.is_node(node) and terralib.islist(symbols))
  symbols:map(function(symbol) self:intern_variable(node, symbol) end)
end

function context:check_variable(node, symbol, expected_type)
  assert(ast.is_node(node))

  if not std.is_symbol(symbol) then
    report.error(node, "expected symbol, got " .. tostring(symbol))
  end

  self.env[#self.env]:lookup(node, symbol)

  if not terralib.types.istype(symbol:hastype()) then
    report.error(node, "expected typed symbol, got untyped symbol " .. tostring(symbol))
  end

  if not std.type_eq(symbol:gettype(), std.as_read(symbol:gettype())) then
    report.error(node, "expected non-reference symbol type, got " .. tostring(symbol:gettype()))
  end

  if not std.type_eq(symbol:gettype(), std.as_read(expected_type)) then
    report.error(node, "expected " .. tostring(std.as_read(expected_type)) .. ", got " .. tostring(symbol:gettype()))
  end
end

local function validate_vars_node(cx)
  return function(node, continuation)
    if node:is(ast.typed.expr.ID) then
      cx:check_variable(node, node.value, node.expr_type)

    elseif node:is(ast.typed.expr.FieldAccess) then
      -- Field accesses used to autoref pointers. The type checker now
      -- desugars into a deref and a separate field access.
      local value_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(value_type) and
        value_type:is_ptr() and
        not std.get_field(value_type.index_type.base_type, node.field_name)
      then
        report.error(node, "expected desugared autoref field access, got " .. tostring(value_type))
      end
      continuation(node, true)

    elseif node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Call) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.New) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.CrossProductArray) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AttachHDF5) or
      node:is(ast.typed.expr.DetachHDF5) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.Future) or
      node:is(ast.typed.expr.FutureGetResult)
    then
      continuation(node, true)

    elseif node:is(ast.typed.stat.If) then
      continuation(node.cond)

      cx:push_local_scope()
      continuation(node.then_block)
      cx:pop_local_scope()

      continuation(node.elseif_blocks)

      cx:push_local_scope()
      continuation(node.else_block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.Elseif) or
      node:is(ast.typed.stat.While)
    then
      continuation(node.cond)

      cx:push_local_scope()
      continuation(node.block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.ForNum) then
      continuation(node.values)

      cx:push_local_scope()
      cx:intern_variable(node, node.symbol)
      continuation(node.block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.ForList) then
      continuation(node.value)

      cx:push_local_scope()
      cx:intern_variable(node, node.symbol)
      continuation(node.block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.ForListVectorized) then
      continuation(node.value)

      cx:push_local_scope()
      cx:intern_variable(node, node.symbol)
      continuation(node.block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.Repeat) then
      cx:push_local_scope()
      continuation(node.block)
      continuation(node.until_cond)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.MustEpoch) or
      node:is(ast.typed.stat.Block)
    then
      cx:push_local_scope()
      continuation(node.block)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.IndexLaunchNum) then
      continuation(node.values)

      cx:push_local_scope()
      cx:intern_variable(node, node.symbol)
      continuation(node.preamble)
      continuation(node.reduce_lhs)
      continuation(node.call)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.IndexLaunchList) then
      continuation(node.value)

      cx:push_local_scope()
      cx:intern_variable(node, node.symbol)
      continuation(node.preamble)
      continuation(node.reduce_lhs)
      continuation(node.call)
      cx:pop_local_scope()

    elseif node:is(ast.typed.stat.Var) then
      continuation(node.values)
      cx:intern_variables(node, node.symbols)
      for i, symbol in ipairs(node.symbols) do
        local var_type = node.types[i]
        cx:check_variable(node, symbol, var_type)
      end

    elseif node:is(ast.typed.stat.VarUnpack) then
      continuation(node.value)
      cx:intern_variables(node, node.symbols)

    elseif node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete) or
      node:is(ast.typed.stat.BeginTrace) or
      node:is(ast.typed.stat.EndTrace) or
      node:is(ast.typed.stat.MapRegions) or
      node:is(ast.typed.stat.UnmapRegions) or
      node:is(ast.typed.Block) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      continuation(node, true)

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end
  end
end

local function validate_variables(cx, node)
  ast.traverse_node_continuation(validate_vars_node(cx), node)
end

local validate = {}

function validate.top_task(cx, node)
  node.params:map(function(param) cx:intern_variable(param, param.symbol) end)

  validate_variables(cx, node.body)
end

function validate.top(cx, node)
  if node:is(ast.typed.top.Task) then
    validate.top_task(cx, node)

  elseif node:is(ast.typed.top.Fspace) or
    node:is(ast.specialized.top.QuoteExpr) or
    node:is(ast.specialized.top.QuoteStat)
  then
    return

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function validate.entry(node)
  local cx = context.new_global_scope({})
  return validate.top(cx, node)
end

return validate
