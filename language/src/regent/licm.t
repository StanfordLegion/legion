-- Copyright 2022 Stanford University
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

-- Loop Invariant Code Motion

-- This pass hoists read-only region accesses in a loop to the outside.

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

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

function context.new_loop_scope(loop_var)
  return setmetatable({
    invariant_vars = data.map_from_table({[loop_var] = true}),
    region_updates = data.new_recursive_map(2),
    nested = false,
  }, context)
end

function context:new_local_scope()
  return setmetatable({
    invariant_vars = self.invariant_vars,
    region_updates = self.region_updates,
    nested = true,
  }, context)
end

function context:mark_invariant(symbol)
  if not self.nested then
    self.invariant_vars[symbol] = true
  end
end

function context:mark_variant(symbol)
  self.invariant_vars[symbol] = false
end

function context:is_invariant(symbol)
  return not self.nested and self.invariant_vars[symbol] == true
end

function context:record_update(region, field_path)
  self.region_updates[region][field_path] = true
end

function context:is_read_only(region, field_path)
  return not self.nested and self.region_updates[region][field_path] ~= true
end

function context:__tostring()
  return "Invariant vars: " .. tostring(self.invariant_vars) .. ", " ..
         "Updated regions: " .. tostring(self.region_updates)
end

local check_invariant = {}

local function check_invariant_pass_through_expr(cx, expr)
  return true
end

local check_invariant_expr_table = {
  [ast.typed.expr.ID]     = function(cx, node)
    return cx:is_invariant(node.value) or std.is_region(std.as_read(node.expr_type))
  end,
  [ast.typed.expr.FieldAccess]  = function(cx, node)
    if std.is_ref(node.expr_type) then
      return data.all(unpack(node.expr_type:bounds():map(function(bound)
        return cx:is_read_only(bound, node.expr_type.field_path)
      end)))
    end
  end,
  [ast.typed.expr]        = check_invariant_pass_through_expr,
  [ast.condition_kind]    = check_invariant_pass_through_expr,
  [ast.disjointness_kind] = check_invariant_pass_through_expr,
  [ast.fence_kind]        = check_invariant_pass_through_expr,
  [ast.location]          = check_invariant_pass_through_expr,
  [ast.annotation]        = check_invariant_pass_through_expr,
}

local check_invariant_expr = ast.make_single_dispatch(
  check_invariant_expr_table,
  {})

function check_invariant.expr(cx, node)
  return ast.mapreduce_node_postorder(
      check_invariant_expr(cx),
      function(a, b) return a and b end,
      node, true)
end

local function check_invariant_pass_through_stat(cx, node)
  return
end

local function check_invariant_stat_if(cx, node)
  local cx = cx:new_local_scope()
  check_invariant.block(cx, node.then_block)
  check_invariant.block(cx, node.else_block)
end

local function check_invariant_stat_nested_block(cx, node)
  local cx = cx:new_local_scope()
  check_invariant.block(cx, node.block)
end

local function check_invariant_stat_block(cx, node)
  check_invariant.block(cx, node.block)
end

local function check_invariant_stat_var(cx, node)
  if node.value then
    local invariant = check_invariant.expr(cx, node.value)
    if invariant then cx:mark_invariant(node.symbol) end
  end
end

local function check_invariant_stat_var_unpack(cx, node)
  local invariant = check_invariant.expr(cx, node.value)
  if invariant then
    node.symbols:map(function(symbol) cx:mark_invariant(symbol) end)
  end
end

local function check_invariant_lhs(cx, expr, first)
  if expr:is(ast.typed.expr.ID) then
    if not std.is_region(std.as_read(expr.expr_type)) then
      cx:mark_variant(expr.value)
    end
  elseif expr:is(ast.typed.expr.FieldAccess) then
    if std.is_ref(expr.expr_type) then
      first = false
      expr.expr_type:bounds():map(function(type)
        cx:record_update(type, expr.expr_type.field_path)
      end)
    end
    check_invariant_lhs(cx, expr.value, first)
  elseif expr:is(ast.typed.expr.IndexAccess) then
    if std.is_ref(expr.expr_type) then
      first = false
      expr.expr_type:bounds():map(function(type)
        cx:record_update(type, expr.expr_type.field_path)
      end)
    end
    check_invariant_lhs(cx, expr.value, first)
  end
end

local function check_invariant_stat_assignment_or_reduce(cx, node)
  check_invariant_lhs(cx, node.lhs, true)
end

local check_invariant_stat_table = {
  [ast.typed.stat.If]              = check_invariant_stat_if,
  [ast.typed.stat.While]           = check_invariant_stat_nested_block,
  [ast.typed.stat.ForNum]          = check_invariant_stat_nested_block,
  [ast.typed.stat.ForList]         = check_invariant_stat_nested_block,
  [ast.typed.stat.Repeat]          = check_invariant_stat_nested_block,
  [ast.typed.stat.MustEpoch]       = check_invariant_stat_nested_block,
  [ast.typed.stat.Block]           = check_invariant_stat_block,
  [ast.typed.stat.Var]             = check_invariant_stat_var,
  [ast.typed.stat.VarUnpack]       = check_invariant_stat_var_unpack,
  [ast.typed.stat.Assignment]      = check_invariant_stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]          = check_invariant_stat_assignment_or_reduce,

  [ast.typed.stat.Elseif]          = function() assert(false) end,
  [ast.typed.stat]                 = check_invariant_pass_through_stat,
}

local check_invariant_stat = ast.make_single_dispatch(
  check_invariant_stat_table,
  {})

function check_invariant.stat(cx, node)
  check_invariant_stat(cx)(node)
end

function check_invariant.block(cx, node)
  node.stats:map(function(stat) check_invariant.stat(cx, stat) end)
end

-- TODO: We only hoist statements of the form "var x = r[y]" where
--       both x and y are invariant in the loop
local function hoist_region_accesses(cx, inner_loop)
  local hoisted = terralib.newlist()
  local stats = terralib.newlist()
  inner_loop.block.stats:map(function(stat)
    if stat:is(ast.typed.stat.Var) and stat.value then
      -- Must check if the value is still invariant due to loop-carried dependence
      local invariant = check_invariant.expr(cx, stat.value)
      if invariant then
        hoisted:insert(stat)
      else
        stats:insert(stat)
        cx:mark_variant(stat.symbol)
      end
    else
      stats:insert(stat)
    end
  end)
  if #hoisted > 0 then
    hoisted:insert(inner_loop { block = inner_loop.block { stats = stats } })
    return ast.typed.stat.Block {
      block = ast.typed.Block {
        stats = hoisted,
        span = inner_loop.span,
      },
      annotations = ast.default_annotations(),
      span = inner_loop.span,
    }
  else
    return inner_loop
  end
end

local licm = {}

function licm.entry(loop_var, block)
  if #block.stats == 1 and block.stats[1]:is(ast.typed.stat.ForNum) then
    local inner_loop = block.stats[1]
    local cx = context.new_loop_scope(loop_var)
    check_invariant.block(cx, inner_loop.block)
    inner_loop = hoist_region_accesses(cx, inner_loop)
    return block { stats = terralib.newlist({ inner_loop }) }
  else
    return block
  end
end

return licm
