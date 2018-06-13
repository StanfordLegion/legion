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

-- Regent AST Validator

local ast = require("regent/ast")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local context = {}

function context:__index(field)
  local value = context[field]
  if value ~= nil then
    return value
  end
  error("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex(field, value)
  error("context has no field '" .. field .. "' (in assignment)", 2)
end

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

local function unreachable(node)
  assert(false, "unreachable")
end

local continue = function(cx, node, continuation)
  continuation(node, true)
end

local function validate_block(cx, node, continuation)
  cx:push_local_scope()
  continuation(node.block)
  cx:pop_local_scope()
end

local validate_loop = terralib.memoize(
  function(symbol_field, value_field)
    return function (cx, node, continuation)
      if value_field then
        continuation(node[value_field])
      end

      cx:push_local_scope()
      if symbol_field then
        cx:intern_variable(node, node[symbol_field])
      end
      continuation(node.block)
      cx:pop_local_scope()
    end
  end)

local node_vars_are_valid = {
  -- Expressions:
  [ast.typed.expr.ID] = function(cx, node, continuation)
    cx:check_variable(node, node.value, node.expr_type)
  end,

  [ast.typed.expr.FieldAccess] = function(cx, node, continuation)
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
  end,

  [ast.typed.expr.Constant]                   = continue,
  [ast.typed.expr.Function]                   = continue,
  [ast.typed.expr.IndexAccess]                = continue,
  [ast.typed.expr.MethodCall]                 = continue,
  [ast.typed.expr.Call]                       = continue,
  [ast.typed.expr.Cast]                       = continue,
  [ast.typed.expr.Ctor]                       = continue,
  [ast.typed.expr.CtorListField]              = continue,
  [ast.typed.expr.CtorRecField]               = continue,
  [ast.typed.expr.RawContext]                 = continue,
  [ast.typed.expr.RawFields]                  = continue,
  [ast.typed.expr.RawPhysical]                = continue,
  [ast.typed.expr.RawRuntime]                 = continue,
  [ast.typed.expr.RawValue]                   = continue,
  [ast.typed.expr.Isnull]                     = continue,
  [ast.typed.expr.Null]                       = continue,
  [ast.typed.expr.DynamicCast]                = continue,
  [ast.typed.expr.StaticCast]                 = continue,
  [ast.typed.expr.UnsafeCast]                 = continue,
  [ast.typed.expr.Ispace]                     = continue,
  [ast.typed.expr.Region]                     = continue,
  [ast.typed.expr.Partition]                  = continue,
  [ast.typed.expr.PartitionEqual]             = continue,
  [ast.typed.expr.PartitionByField]           = continue,
  [ast.typed.expr.Image]                      = continue,
  [ast.typed.expr.ImageByTask]                = continue,
  [ast.typed.expr.Preimage]                   = continue,
  [ast.typed.expr.CrossProduct]               = continue,
  [ast.typed.expr.CrossProductArray]          = continue,
  [ast.typed.expr.ListSlicePartition]         = continue,
  [ast.typed.expr.ListDuplicatePartition]     = continue,
  [ast.typed.expr.ListSliceCrossProduct]      = continue,
  [ast.typed.expr.ListCrossProduct]           = continue,
  [ast.typed.expr.ListCrossProductComplete]   = continue,
  [ast.typed.expr.ListPhaseBarriers]          = continue,
  [ast.typed.expr.ListInvert]                 = continue,
  [ast.typed.expr.ListRange]                  = continue,
  [ast.typed.expr.ListIspace]                 = continue,
  [ast.typed.expr.ListFromElement]            = continue,
  [ast.typed.expr.PhaseBarrier]               = continue,
  [ast.typed.expr.DynamicCollective]          = continue,
  [ast.typed.expr.DynamicCollectiveGetResult] = continue,
  [ast.typed.expr.Advance]                    = continue,
  [ast.typed.expr.Adjust]                     = continue,
  [ast.typed.expr.Arrive]                     = continue,
  [ast.typed.expr.Await]                      = continue,
  [ast.typed.expr.Copy]                       = continue,
  [ast.typed.expr.Fill]                       = continue,
  [ast.typed.expr.Acquire]                    = continue,
  [ast.typed.expr.Release]                    = continue,
  [ast.typed.expr.AttachHDF5]                 = continue,
  [ast.typed.expr.DetachHDF5]                 = continue,
  [ast.typed.expr.AllocateScratchFields]      = continue,
  [ast.typed.expr.WithScratchFields]          = continue,
  [ast.typed.expr.RegionRoot]                 = continue,
  [ast.typed.expr.Condition]                  = continue,
  [ast.typed.expr.Unary]                      = continue,
  [ast.typed.expr.Binary]                     = continue,
  [ast.typed.expr.Deref]                      = continue,
  [ast.typed.expr.Future]                     = continue,
  [ast.typed.expr.FutureGetResult]            = continue,
  [ast.typed.expr.ParallelizerConstraint]     = continue,

  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.If] = function(cx, node, continuation)
    continuation(node.cond)

    cx:push_local_scope()
    continuation(node.then_block)
    cx:pop_local_scope()

    continuation(node.elseif_blocks)

    cx:push_local_scope()
    continuation(node.else_block)
    cx:pop_local_scope()
  end,

  [ast.typed.stat.Elseif] = validate_loop(nil, "cond"),
  [ast.typed.stat.While] = validate_loop(nil, "cond"),

  [ast.typed.stat.ForNum] = validate_loop("symbol", "values"),

  [ast.typed.stat.ForNumVectorized] = validate_loop("symbol", "values"),

  [ast.typed.stat.ForList] = validate_loop("symbol", "value"),

  [ast.typed.stat.ForListVectorized] = validate_loop("symbol", "value"),

  [ast.typed.stat.Repeat] = function(cx, node, continuation)
    cx:push_local_scope()
    continuation(node.block)
    continuation(node.until_cond)
    cx:pop_local_scope()
  end,

  [ast.typed.stat.MustEpoch] = validate_block,
  [ast.typed.stat.Block] = validate_block,

  [ast.typed.stat.IndexLaunchNum] = function(cx, node, continuation)
    continuation(node.values)

    cx:push_local_scope()
    cx:intern_variable(node, node.symbol)
    continuation(node.preamble)
    continuation(node.reduce_lhs)
    continuation(node.call)
    cx:pop_local_scope()
  end,

  [ast.typed.stat.IndexLaunchList] = function(cx, node, continuation)
    continuation(node.value)

    cx:push_local_scope()
    cx:intern_variable(node, node.symbol)
    continuation(node.preamble)
    continuation(node.reduce_lhs)
    continuation(node.call)
    cx:pop_local_scope()
  end,

  [ast.typed.stat.Var] = function(cx, node, continuation)
    continuation(node.value)
    cx:intern_variable(node, node.symbol)
    cx:check_variable(node, node.symbol, node.type)
  end,

  [ast.typed.stat.VarUnpack] = function(cx, node, continuation)
    continuation(node.value)
    cx:intern_variables(node, node.symbols)
  end,

  [ast.typed.stat.Return]          = continue,
  [ast.typed.stat.Break]           = continue,
  [ast.typed.stat.Assignment]      = continue,
  [ast.typed.stat.Reduce]          = continue,
  [ast.typed.stat.Expr]            = continue,
  [ast.typed.stat.RawDelete]       = continue,
  [ast.typed.stat.Fence]           = continue,
  [ast.typed.stat.ParallelizeWith] = continue,
  [ast.typed.stat.BeginTrace]      = continue,
  [ast.typed.stat.EndTrace]        = continue,
  [ast.typed.stat.MapRegions]      = continue,
  [ast.typed.stat.UnmapRegions]    = continue,

  [ast.typed.stat.Internal]        = unreachable,

  -- Miscellaneous:
  [ast.typed.Block]       = continue,
  [ast.location]          = continue,
  [ast.annotation]        = continue,
  [ast.condition_kind]    = continue,
  [ast.disjointness_kind] = continue,
  [ast.fence_kind]        = continue,
}

local validate_vars_node = ast.make_single_dispatch(
  node_vars_are_valid,
  {ast.typed.expr, ast.typed.stat})

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
