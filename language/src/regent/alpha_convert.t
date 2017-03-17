-- Copyright 2017 Stanford University
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

-- Alpha Conversion for Specialized AST

-- Alpha conversion fixes two issues in Regent specialization:
--
--  1. Duplicate symbols in the AST. A single symbol can be spliced
--     into the AST in multiple places with conflicting types.
--
--  2. Mutation to symbols after specialization. (Symbols are, after
--     all, part of the user-facing API.) We need to copy them to
--     avoid any potential bad behavior.

local ast = require("regent/ast")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local context = {}
context.__index = context

function context.new_global_scope(env, mapping)
  local cx = {
    env = terralib.newlist({env}),
    mapping = terralib.newlist({mapping}),
  }
  setmetatable(cx, context)
  return cx
end

function context:push_local_scope()
  self.env:insert(self.env[#self.env]:new_local_scope())

  local copy_mapping = {}
  for k, v in pairs(self.mapping[#self.mapping]) do
    copy_mapping[k] = v
  end
  self.mapping:insert(copy_mapping)
end

function context:pop_local_scope()
  assert(self.env:remove())
  assert(self.mapping:remove())
end

function context:intern_variable(node, symbol)
  assert(ast.is_node(node))
  if not std.is_symbol(symbol) then
    report.error(node, "expected a symbol, got " .. tostring(symbol))
  end
  self.env[#self.env]:insert(node, symbol, symbol)
end

function context:replace_variable(node, symbol)
  assert(ast.is_node(node))
  if not std.is_symbol(symbol) then
    report.error(node, "expected a symbol, got " .. tostring(symbol))
  end

  local new_symbol = std.newsymbol(symbol:hasname())
  self.mapping[#self.mapping][symbol] = new_symbol
  if symbol:hastype() then
    new_symbol:settype(std.type_sub(symbol:gettype(), self.mapping[#self.mapping]))
  end

  self.env[#self.env]:insert(node, symbol, new_symbol)

  return new_symbol
end

function context:update_symbol(node, symbol)
  return self.env[#self.env]:safe_lookup(symbol) or symbol
end

function context:update_type(value_type)
  return std.type_sub(value_type, self.mapping[#self.mapping])
end

local function alpha_convert_node(cx)
  return function(node, continuation)
    if node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      return continuation(node, true)

    elseif node:is(ast.specialized.region.Bare) then
      return node { value = cx:update_symbol(node, node.value) }

    elseif node:is(ast.specialized.region.Root) then
      return node { value = cx:update_symbol(node, node.value) }

    elseif node:is(ast.specialized.region.Field) then
      return continuation(node, true)

    elseif node:is(ast.specialized.expr.ID) then
      return node { value = cx:update_symbol(node, node.value) }

    elseif node:is(ast.specialized.expr.Function) then
      if terralib.types.istype(node.value) then
        return continuation(node, true) {
          value = cx:update_type(node.value),
        }
      else
        return continuation(node, true)
      end

    elseif node:is(ast.specialized.expr.New) or
      node:is(ast.specialized.expr.Null)
    then
      return continuation(node, true) {
        pointer_type = cx:update_type(node.pointer_type),
      }

    elseif node:is(ast.specialized.expr.DynamicCast) or
      node:is(ast.specialized.expr.StaticCast) or
      node:is(ast.specialized.expr.UnsafeCast)
    then
      return continuation(node, true) {
        expr_type = cx:update_type(node.expr_type),
      }

    elseif node:is(ast.specialized.expr.Ispace) then
      return continuation(node, true) {
        index_type = cx:update_type(node.index_type),
      }

    elseif node:is(ast.specialized.expr.Region) then
      return continuation(node, true) {
        fspace_type = cx:update_type(node.fspace_type),
      }

    elseif node:is(ast.specialized.expr.Constant) or
      node:is(ast.specialized.expr.FieldAccess) or
      node:is(ast.specialized.expr.IndexAccess) or
      node:is(ast.specialized.expr.MethodCall) or
      node:is(ast.specialized.expr.Call) or
      node:is(ast.specialized.expr.Cast) or
      node:is(ast.specialized.expr.Ctor) or
      node:is(ast.specialized.expr.CtorListField) or
      node:is(ast.specialized.expr.CtorRecField) or
      node:is(ast.specialized.expr.RawContext) or
      node:is(ast.specialized.expr.RawFields) or
      node:is(ast.specialized.expr.RawPhysical) or
      node:is(ast.specialized.expr.RawRuntime) or
      node:is(ast.specialized.expr.RawValue) or
      node:is(ast.specialized.expr.Isnull) or
      node:is(ast.specialized.expr.Partition) or
      node:is(ast.specialized.expr.PartitionEqual) or
      node:is(ast.specialized.expr.PartitionByField) or
      node:is(ast.specialized.expr.Image) or
      node:is(ast.specialized.expr.Preimage) or
      node:is(ast.specialized.expr.CrossProduct) or
      node:is(ast.specialized.expr.CrossProductArray) or
      node:is(ast.specialized.expr.ListSlicePartition) or
      node:is(ast.specialized.expr.ListDuplicatePartition) or
      node:is(ast.specialized.expr.ListCrossProduct) or
      node:is(ast.specialized.expr.ListCrossProductComplete) or
      node:is(ast.specialized.expr.ListPhaseBarriers) or
      node:is(ast.specialized.expr.ListInvert) or
      node:is(ast.specialized.expr.ListRange) or
      node:is(ast.specialized.expr.ListIspace) or
      node:is(ast.specialized.expr.PhaseBarrier) or
      node:is(ast.specialized.expr.DynamicCollective) or
      node:is(ast.specialized.expr.DynamicCollectiveGetResult) or
      node:is(ast.specialized.expr.Advance) or
      node:is(ast.specialized.expr.Adjust) or
      node:is(ast.specialized.expr.Arrive) or
      node:is(ast.specialized.expr.Await) or
      node:is(ast.specialized.expr.Copy) or
      node:is(ast.specialized.expr.Fill) or
      node:is(ast.specialized.expr.Acquire) or
      node:is(ast.specialized.expr.Release) or
      node:is(ast.specialized.expr.AttachHDF5) or
      node:is(ast.specialized.expr.DetachHDF5) or
      node:is(ast.specialized.expr.AllocateScratchFields) or
      node:is(ast.specialized.expr.WithScratchFields) or
      node:is(ast.specialized.expr.RegionRoot) or
      node:is(ast.specialized.expr.Condition) or
      node:is(ast.specialized.expr.Unary) or
      node:is(ast.specialized.expr.Binary) or
      node:is(ast.specialized.expr.Deref)
    then
      return continuation(node, true)

    elseif node:is(ast.specialized.expr.LuaTable) then
      report.error(node, "unable to specialize value of type table")

    elseif node:is(ast.specialized.stat.If) then
      local cond = continuation(node.cond)

      cx:push_local_scope()
      local then_block = continuation(node.then_block)
      cx:pop_local_scope()

      local elseif_blocks = continuation(node.elseif_blocks)

      cx:push_local_scope()
      local else_block = continuation(node.else_block)
      cx:pop_local_scope()

      return node {
        cond = cond,
        then_block = then_block,
        elseif_blocks = elseif_blocks,
        else_block = else_block,
      }

    elseif node:is(ast.specialized.stat.Elseif) or
      node:is(ast.specialized.stat.While)
    then
      local cond = continuation(node.cond)

      cx:push_local_scope()
      local block = continuation(node.block)
      cx:pop_local_scope()

      return node {
        cond = cond,
        block = block,
      }

    elseif node:is(ast.specialized.stat.ForNum) then
      local values = continuation(node.values)

      cx:push_local_scope()
      local symbol = cx:replace_variable(node, node.symbol)
      local block = continuation(node.block)
      cx:pop_local_scope()

      return node {
        symbol = symbol,
        values = values,
        block = block,
      }

    elseif node:is(ast.specialized.stat.ForList) then
      local value = continuation(node.value)

      cx:push_local_scope()
      local symbol = cx:replace_variable(node, node.symbol)
      local block = continuation(node.block)
      cx:pop_local_scope()

      return node {
        symbol = symbol,
        value = value,
        block = block,
      }

    elseif node:is(ast.specialized.stat.Repeat) then
      cx:push_local_scope()
      local block = continuation(node.block)
      local until_cond = continuation(node.until_cond)
      cx:pop_local_scope()

      return node {
        block = block,
        until_cond = until_cond,
      }

    elseif node:is(ast.specialized.stat.MustEpoch) or
      node:is(ast.specialized.stat.Block) or
      node:is(ast.specialized.stat.ParallelizeWith)
    then
      cx:push_local_scope()
      local block = continuation(node.block)
      cx:pop_local_scope()

      return node {
        block = block,
      }

    elseif node:is(ast.specialized.stat.Var) then
      local symbols = terralib.newlist()
      for i, symbol in ipairs(node.symbols) do
        if node.values[i] and node.values[i]:is(ast.specialized.expr.Region) then
          symbols[i] = cx:replace_variable(node, symbol)
        end
      end

      local values = continuation(node.values)

      for i, symbol in ipairs(node.symbols) do
        if not symbols[i] then
          symbols[i] = cx:replace_variable(node, symbol)
        end
      end

      return node {
        symbols = symbols,
        values = values,
      }

    elseif node:is(ast.specialized.stat.VarUnpack) then
      local value = continuation(node.value)
      local symbols = node.symbols:map(
        function(symbol) return cx:replace_variable(node, symbol) end)

      return node {
        symbols = symbols,
        value = value,
      }

    elseif node:is(ast.specialized.stat.Return) or
      node:is(ast.specialized.stat.Break) or
      node:is(ast.specialized.stat.Assignment) or
      node:is(ast.specialized.stat.Reduce) or
      node:is(ast.specialized.stat.Expr) or
      node:is(ast.specialized.stat.RawDelete) or
      node:is(ast.specialized.Block) or
      node:is(ast.location) or
      node:is(ast.annotation)
    then
      return continuation(node, true)

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end
  end
end

local function alpha_convert_body(cx, node)
  return ast.map_node_continuation(alpha_convert_node(cx), node)
end

local alpha_convert = {}

function alpha_convert.top_task(cx, node)
  node.params:map(function(param) cx:intern_variable(param, param.symbol) end)

  return node { body = alpha_convert_body(cx, node.body) }
end

function alpha_convert.top_quote_expr(cx, node)
  return node { expr = alpha_convert_body(cx, node.expr) }
end

function alpha_convert.top_quote_stat(cx, node)
  return node { block = alpha_convert_body(cx, node.block) }
end

function alpha_convert.top(cx, node)
  if node:is(ast.specialized.top.Task) then
    return alpha_convert.top_task(cx, node)

  elseif node:is(ast.specialized.top.Fspace) then
    return node

  elseif node:is(ast.specialized.top.QuoteExpr) then
    return alpha_convert.top_quote_expr(cx, node)

  elseif node:is(ast.specialized.top.QuoteStat) then
    return alpha_convert.top_quote_stat(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function alpha_convert.entry(node, env, mapping)
  if env == nil then env = symbol_table.new_global_scope({}) end
  if mapping == nil then mapping = {} end
  local cx = context.new_global_scope(env, mapping)
  return alpha_convert.top(cx, node)
end

return alpha_convert
