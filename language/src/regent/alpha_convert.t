-- Copyright 2018 Stanford University
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

local function pass_through(cx, node, continuation)
  return continuation(node, true)
end

local function update_symbol(field_name)
  return function(cx, node, continuation)
    return continuation(node, true) {
      [field_name] = cx:update_symbol(node, node[field_name])
    }
  end
end

local update_value = update_symbol("value")

local function update_type(field_name)
  return function(cx, node, continuation)
    return continuation(node, true) {
      [field_name] = cx:update_type(node[field_name])
    }
  end
end

local update_pointer_type = update_type("pointer_type")
local update_expr_type = update_type("expr_type")
local update_index_type = update_type("index_type")
local update_fspace_type = update_type("fspace_type")

local function update_block(cx, node, continuation)
  cx:push_local_scope()
  local block = continuation(node.block)
  cx:pop_local_scope()

  return node {
    block = block,
  }
end

local function update_block_with_value(field_name)
  return function(cx, node, continuation)
    local value = continuation(node[field_name])

    cx:push_local_scope()
    local block = continuation(node.block)
    cx:pop_local_scope()

    return node {
      [field_name] = value,
      block = block,
    }
  end
end

local update_block_with_cond = update_block_with_value("cond")
local update_block_with_until_cond = update_block_with_value("until_cond")

local function update_block_with_values(value_field)
  return function(cx, node, continuation)
    local value = continuation(node[value_field])

    cx:push_local_scope()
    local block = continuation(node.block)
    cx:pop_local_scope()

    return node {
      [value_field] = value,
      block = block,
    }
  end
end

local update_block_with_hints = update_block_with_value("hints")

local function update_block_with_symbol_value(symbol_field, value_field)
  return function(cx, node, continuation)
    local value = continuation(node[value_field])

    cx:push_local_scope()
    local symbol = cx:replace_variable(node, node[symbol_field])
    local block = continuation(node.block)
    cx:pop_local_scope()

    return node {
      [symbol_field] = symbol,
      [value_field] = value,
      block = block,
    }
  end
end

local update_block_with_loop_value = update_block_with_symbol_value("symbol", "value")
local update_block_with_loop_values = update_block_with_symbol_value("symbol", "values")

local node_alpha_conversion = {
  [ast.condition_kind]           = pass_through,
  [ast.disjointness_kind]        = pass_through,
  [ast.fence_kind]               = pass_through,

  [ast.specialized.region.Bare]  = update_value,
  [ast.specialized.region.Root]  = update_value,

  [ast.specialized.region.Field] = pass_through,

  [ast.specialized.expr.ID]      = update_value,

  [ast.specialized.expr.Function] = function(cx, node, continuation)
    if terralib.types.istype(node.value) then
      return continuation(node, true) {
        value = cx:update_type(node.value),
      }
    else
      return continuation(node, true)
    end
  end,

  [ast.specialized.expr.New]         = update_pointer_type,
  [ast.specialized.expr.Null]        = update_pointer_type,
  [ast.specialized.expr.DynamicCast] = update_expr_type,
  [ast.specialized.expr.StaticCast]  = update_expr_type,
  [ast.specialized.expr.UnsafeCast]  = update_expr_type,
  [ast.specialized.expr.Ispace]      = update_index_type,
  [ast.specialized.expr.Region]      = update_fspace_type,

  [ast.specialized.expr.Constant]                   = pass_through,
  [ast.specialized.expr.FieldAccess]                = pass_through,
  [ast.specialized.expr.IndexAccess]                = pass_through,
  [ast.specialized.expr.MethodCall]                 = pass_through,
  [ast.specialized.expr.Call]                       = pass_through,
  [ast.specialized.expr.Cast]                       = pass_through,
  [ast.specialized.expr.Ctor]                       = pass_through,
  [ast.specialized.expr.CtorListField]              = pass_through,
  [ast.specialized.expr.CtorRecField]               = pass_through,
  [ast.specialized.expr.RawContext]                 = pass_through,
  [ast.specialized.expr.RawFields]                  = pass_through,
  [ast.specialized.expr.RawPhysical]                = pass_through,
  [ast.specialized.expr.RawRuntime]                 = pass_through,
  [ast.specialized.expr.RawValue]                   = pass_through,
  [ast.specialized.expr.Isnull]                     = pass_through,
  [ast.specialized.expr.Partition]                  = pass_through,
  [ast.specialized.expr.PartitionEqual]             = pass_through,
  [ast.specialized.expr.PartitionByField]           = pass_through,
  [ast.specialized.expr.Image]                      = pass_through,
  [ast.specialized.expr.Preimage]                   = pass_through,
  [ast.specialized.expr.CrossProduct]               = pass_through,
  [ast.specialized.expr.CrossProductArray]          = pass_through,
  [ast.specialized.expr.ListSlicePartition]         = pass_through,
  [ast.specialized.expr.ListDuplicatePartition]     = pass_through,
  [ast.specialized.expr.ListCrossProduct]           = pass_through,
  [ast.specialized.expr.ListCrossProductComplete]   = pass_through,
  [ast.specialized.expr.ListPhaseBarriers]          = pass_through,
  [ast.specialized.expr.ListInvert]                 = pass_through,
  [ast.specialized.expr.ListRange]                  = pass_through,
  [ast.specialized.expr.ListIspace]                 = pass_through,
  [ast.specialized.expr.ListFromElement]            = pass_through,
  [ast.specialized.expr.PhaseBarrier]               = pass_through,
  [ast.specialized.expr.DynamicCollective]          = pass_through,
  [ast.specialized.expr.DynamicCollectiveGetResult] = pass_through,
  [ast.specialized.expr.Advance]                    = pass_through,
  [ast.specialized.expr.Adjust]                     = pass_through,
  [ast.specialized.expr.Arrive]                     = pass_through,
  [ast.specialized.expr.Await]                      = pass_through,
  [ast.specialized.expr.Copy]                       = pass_through,
  [ast.specialized.expr.Fill]                       = pass_through,
  [ast.specialized.expr.Acquire]                    = pass_through,
  [ast.specialized.expr.Release]                    = pass_through,
  [ast.specialized.expr.AttachHDF5]                 = pass_through,
  [ast.specialized.expr.DetachHDF5]                 = pass_through,
  [ast.specialized.expr.AllocateScratchFields]      = pass_through,
  [ast.specialized.expr.WithScratchFields]          = pass_through,
  [ast.specialized.expr.RegionRoot]                 = pass_through,
  [ast.specialized.expr.Condition]                  = pass_through,
  [ast.specialized.expr.Unary]                      = pass_through,
  [ast.specialized.expr.Binary]                     = pass_through,
  [ast.specialized.expr.Deref]                      = pass_through,

  [ast.specialized.expr.LuaTable] = function(cx, node, continuation)
    report.error(node, "unable to specialize value of type table")
  end,

  [ast.specialized.stat.If] = function(cx, node, continuation)
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
  end,

  [ast.specialized.stat.Elseif]          = update_block_with_cond,
  [ast.specialized.stat.While]           = update_block_with_cond,
  [ast.specialized.stat.ForNum]          = update_block_with_loop_values,
  [ast.specialized.stat.ForList]         = update_block_with_loop_value,
  [ast.specialized.stat.Repeat]          = update_block_with_until_cond,
  [ast.specialized.stat.MustEpoch]       = update_block,
  [ast.specialized.stat.Block]           = update_block,
  [ast.specialized.stat.ParallelizeWith] = update_block_with_hints,

  [ast.specialized.stat.Var] = function(cx, node, continuation)
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
  end,

  [ast.specialized.stat.VarUnpack] = function(cx, node, continuation)
    local value = continuation(node.value)
    local symbols = node.symbols:map(
      function(symbol) return cx:replace_variable(node, symbol) end)

    return node {
      symbols = symbols,
      value = value,
    }
  end,

  [ast.specialized.stat.Return] = pass_through,
  [ast.specialized.stat.Break] = pass_through,
  [ast.specialized.stat.Assignment] = pass_through,
  [ast.specialized.stat.Reduce] = pass_through,
  [ast.specialized.stat.Expr] = pass_through,
  [ast.specialized.stat.RawDelete] = pass_through,
  [ast.specialized.stat.Fence] = pass_through,
  [ast.specialized.Block] = pass_through,
  [ast.location] = pass_through,
  [ast.annotation] = pass_through,
}

local alpha_convert_node = ast.make_single_dispatch(
  node_alpha_conversion,
  {ast.specialized.expr, ast.specialized.stat})

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
