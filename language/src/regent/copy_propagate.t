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

-- Regent Copy Propagation Pass

-- For every declaration 'var x = y' where both x and y are variables,
-- this pass replaces all occurrences of x with y and remove the original
-- statement.


local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local function unreachable(cx, node) assert(false) end

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

function context.new_global_scope()
  local cx = {
    mapping = terralib.newlist({data.newmap()}),
    kill = data.newmap(),
  }
  setmetatable(cx, context)
  return cx
end

function context:push_local_scope()
  local copy_mapping = self.mapping[#self.mapping]:copy()
  self.mapping:insert(copy_mapping)
end

function context:pop_local_scope()
  assert(self.mapping:remove())
end

function context:update_kill(symbol)
  self.kill[symbol] = true
end

function context:update_mapping(from, to)
  self.mapping[#self.mapping][from] = to
end

function context:replace_symbol(from)
  local mapping = self.mapping[#self.mapping]
  local to = from
  while mapping[to] and
        not self.kill[to] and
        not self.kill[mapping[to]]
  do
    to = mapping[to]
  end
  return to
end

local strip_expr
do
  local strip_expr_table = {
    [ast.typed.expr.FieldAccess]  = function(expr) return strip_expr(expr.value) end,
    [ast.typed.expr.Deref]        = function(expr) return strip_expr(expr.value) end,
    [ast.typed.expr.IndexAccess]  = function(expr) return strip_expr(expr.value) end,
    [ast.typed.expr.ID]           = function(expr) return expr.value end,
  }

  strip_expr = ast.make_single_dispatch(strip_expr_table, {}, function(expr) return false end)()
end

-- We run a flow-insensitive analysis to collect all potential kills of definitions

local collect_kills = {}

function collect_kills.stat_if(cx, stat)
  collect_kills.block(cx, stat.then_block)
  collect_kills.block(cx, stat.else_block)
end

function collect_kills.stat_block(cx, stat)
  collect_kills.block(cx, stat.block)
end

function collect_kills.stat_var(cx, stat)
  local value = stat.value
  if not value then
    return
  elseif value:is(ast.typed.expr.Cast) and value.fn.value:ispointer() then
    local symbol = strip_expr(value.arg)
    if symbol then cx:update_kill(symbol) end
  elseif value:is(ast.typed.expr.AddressOf) then
    local symbol = strip_expr(value.value)
    if symbol then cx:update_kill(symbol) end
  end
end

function collect_kills.stat_assignment_or_reduce(cx, stat)
  cx:update_kill(strip_expr(stat.lhs))
end

function collect_kills.pass_through_stat(cx, stat) end

local collect_kills_stat_table = {
  [ast.typed.stat.If]              = collect_kills.stat_if,
  [ast.typed.stat.While]           = collect_kills.stat_block,
  [ast.typed.stat.ForNum]          = collect_kills.stat_block,
  [ast.typed.stat.ForList]         = collect_kills.stat_block,
  [ast.typed.stat.Repeat]          = collect_kills.stat_block,
  [ast.typed.stat.Block]           = collect_kills.stat_block,

  [ast.typed.stat.Var]             = collect_kills.stat_var,
  [ast.typed.stat.Assignment]      = collect_kills.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]          = collect_kills.stat_assignment_or_reduce,

  [ast.typed.stat.Expr]            = collect_kills.pass_through_stat,
  [ast.typed.stat.Return]          = collect_kills.pass_through_stat,
  [ast.typed.stat.VarUnpack]       = collect_kills.pass_through_stat,
  [ast.typed.stat.ParallelPrefix]  = collect_kills.pass_through_stat,
  [ast.typed.stat.RawDelete]       = collect_kills.pass_through_stat,
  [ast.typed.stat.Break]           = collect_kills.pass_through_stat,
  [ast.typed.stat.Fence]           = collect_kills.pass_through_stat,

  [ast.typed.stat.Elseif]          = unreachable,
  [ast.typed.stat.Internal]        = unreachable,

  [ast.typed.stat.MustEpoch]         = unreachable,
  [ast.typed.stat.ParallelizeWith]   = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.IndexLaunchNum]    = unreachable,
  [ast.typed.stat.IndexLaunchList]   = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local collect_kills_stat = ast.make_single_dispatch(
  collect_kills_stat_table,
  {ast.typed.stat})

function collect_kills.stat(cx, stat)
  collect_kills_stat(cx)(stat)
end

function collect_kills.block(cx, node)
  node.stats:map(function(stat) collect_kills.stat(cx, stat) end)
end

local copy_propagate = {}

function copy_propagate.pass_through_expr(cx, expr) return expr end

function copy_propagate.expr_regent_cast(cx, expr)
  return expr {
    value = copy_propagate.expr(cx, expr.value),
  }
end

function copy_propagate.expr_field_access_or_deref_or_address_of(cx, expr)
  return expr {
    value = copy_propagate.expr(cx, expr.value),
  }
end

function copy_propagate.expr_index_access(cx, expr)
  return expr {
    index = copy_propagate.expr(cx, expr.index),
    value = copy_propagate.expr(cx, expr.value),
  }
end

function copy_propagate.expr_ctor(cx, expr)
  return expr {
    fields = expr.fields:map(function(field)
      return field {
        value = copy_propagate.expr(cx, field.value),
      }
    end),
  }
end

function copy_propagate.expr_unary(cx, expr)
  return expr {
    rhs = copy_propagate.expr(cx, expr.rhs),
  }
end

function copy_propagate.expr_binary(cx, expr)
  return expr {
    lhs = copy_propagate.expr(cx, expr.lhs),
    rhs = copy_propagate.expr(cx, expr.rhs),
  }
end

function copy_propagate.expr_cast(cx, expr)
  return expr { arg = copy_propagate.expr(cx, expr.arg) }
end

function copy_propagate.expr_call(cx, expr)
  return expr {
    args = expr.args:map(function(arg)
      return copy_propagate.expr(cx, arg)
    end),
  }
end

function copy_propagate.expr_method_call(cx, expr)
  return expr {
    value = copy_propagate.expr(cx, expr.value),
    args = expr.args:map(function(arg)
      return copy_propagate.expr(cx, arg)
    end),
  }
end

function copy_propagate.expr_id(cx, expr)
  local value = cx:replace_symbol(expr.value)
  if value ~= expr.value then
    return expr { value = value }
  else
    return expr
  end
end

function copy_propagate.expr_is_null(cx, expr)
  return expr {
    pointer = copy_propagate.expr(cx, expr.pointer),
  }
end

local copy_propagate_expr_table = {
  [ast.typed.expr.DynamicCast]                = copy_propagate.expr_regent_cast,
  [ast.typed.expr.StaticCast]                 = copy_propagate.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]                 = copy_propagate.expr_regent_cast,
  [ast.typed.expr.FieldAccess]                = copy_propagate.expr_field_access_or_deref_or_address_of,
  [ast.typed.expr.Deref]                      = copy_propagate.expr_field_access_or_deref_or_address_of,
  [ast.typed.expr.AddressOf]                  = copy_propagate.expr_field_access_or_deref_or_address_of,
  [ast.typed.expr.IndexAccess]                = copy_propagate.expr_index_access,
  [ast.typed.expr.Ctor]                       = copy_propagate.expr_ctor,
  [ast.typed.expr.Unary]                      = copy_propagate.expr_unary,
  [ast.typed.expr.Binary]                     = copy_propagate.expr_binary,
  [ast.typed.expr.Cast]                       = copy_propagate.expr_cast,


  [ast.typed.expr.Call]                       = copy_propagate.expr_call,
  [ast.typed.expr.MethodCall]                 = copy_propagate.expr_method_call,
  [ast.typed.expr.ID]                         = copy_propagate.expr_id,
  [ast.typed.expr.Isnull]                     = copy_propagate.expr_is_null,

  [ast.typed.expr.RawFields]                  = copy_propagate.pass_through_expr,
  [ast.typed.expr.RawFuture]                  = copy_propagate.pass_through_expr,
  [ast.typed.expr.RawPhysical]                = copy_propagate.pass_through_expr,
  [ast.typed.expr.RawRuntime]                 = copy_propagate.pass_through_expr,
  [ast.typed.expr.RawTask]                    = copy_propagate.pass_through_expr,
  [ast.typed.expr.RawValue]                   = copy_propagate.pass_through_expr,
  [ast.typed.expr.ListInvert]                 = copy_propagate.pass_through_expr,
  [ast.typed.expr.ListRange]                  = copy_propagate.pass_through_expr,
  [ast.typed.expr.ListIspace]                 = copy_propagate.pass_through_expr,
  [ast.typed.expr.ListFromElement]            = copy_propagate.pass_through_expr,
  [ast.typed.expr.RegionRoot]                 = copy_propagate.pass_through_expr,

  [ast.typed.expr.Function]                   = copy_propagate.pass_through_expr,
  [ast.typed.expr.Constant]                   = copy_propagate.pass_through_expr,
  [ast.typed.expr.Global]                     = copy_propagate.pass_through_expr,
  [ast.typed.expr.Null]                       = copy_propagate.pass_through_expr,
  [ast.typed.expr.Projection]                 = copy_propagate.pass_through_expr,

  [ast.typed.expr.CtorListField]              = unreachable,
  [ast.typed.expr.CtorRecField]               = unreachable,
  [ast.typed.expr.Internal]                   = unreachable,
  [ast.typed.expr.RawContext]                 = unreachable,
  [ast.typed.expr.Ispace]                     = unreachable,
  [ast.typed.expr.Region]                     = unreachable,
  [ast.typed.expr.Partition]                  = unreachable,
  [ast.typed.expr.PartitionEqual]             = unreachable,
  [ast.typed.expr.PartitionByField]           = unreachable,
  [ast.typed.expr.PartitionByRestriction]     = unreachable,
  [ast.typed.expr.Image]                      = unreachable,
  [ast.typed.expr.ImageByTask]                = unreachable,
  [ast.typed.expr.Preimage]                   = unreachable,
  [ast.typed.expr.CrossProduct]               = unreachable,
  [ast.typed.expr.CrossProductArray]          = unreachable,
  [ast.typed.expr.ListSlicePartition]         = unreachable,
  [ast.typed.expr.ListDuplicatePartition]     = unreachable,
  [ast.typed.expr.ListSliceCrossProduct]      = unreachable,
  [ast.typed.expr.ListCrossProduct]           = unreachable,
  [ast.typed.expr.ListCrossProductComplete]   = unreachable,
  [ast.typed.expr.ListPhaseBarriers]          = unreachable,
  [ast.typed.expr.PhaseBarrier]               = unreachable,
  [ast.typed.expr.DynamicCollective]          = unreachable,
  [ast.typed.expr.DynamicCollectiveGetResult] = unreachable,
  [ast.typed.expr.Advance]                    = unreachable,
  [ast.typed.expr.Adjust]                     = unreachable,
  [ast.typed.expr.Arrive]                     = unreachable,
  [ast.typed.expr.Await]                      = unreachable,
  [ast.typed.expr.Copy]                       = unreachable,
  [ast.typed.expr.Fill]                       = unreachable,
  [ast.typed.expr.Acquire]                    = unreachable,
  [ast.typed.expr.Release]                    = unreachable,
  [ast.typed.expr.AttachHDF5]                 = unreachable,
  [ast.typed.expr.DetachHDF5]                 = unreachable,
  [ast.typed.expr.AllocateScratchFields]      = unreachable,
  [ast.typed.expr.WithScratchFields]          = unreachable,
  [ast.typed.expr.Condition]                  = unreachable,
  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.ParallelizerConstraint]     = unreachable,
  [ast.typed.expr.ImportIspace]               = unreachable,
  [ast.typed.expr.ImportRegion]               = unreachable,
  [ast.typed.expr.ImportPartition]            = unreachable,
  [ast.typed.expr.ImportCrossProduct]         = unreachable,
}

local copy_propagate_expr = ast.make_single_dispatch(
  copy_propagate_expr_table,
  {ast.typed.expr})

function copy_propagate.expr(cx, expr)
  return copy_propagate_expr(cx)(expr)
end

function copy_propagate.stat_if(cx, stat)
  return stat {
    cond = copy_propagate.expr(cx, stat.cond),
    then_block = copy_propagate.block(cx, stat.then_block),
    else_block = copy_propagate.block(cx, stat.else_block),
  }
end

function copy_propagate.stat_while(cx, stat)
  return stat {
    cond = copy_propagate.expr(cx, stat.cond),
    block = copy_propagate.block(cx, stat.block),
  }
end

function copy_propagate.stat_for_num(cx, stat)
  return stat {
    values = stat.values:map(function(value)
      return copy_propagate.expr(cx, value)
    end),
    block = copy_propagate.block(cx, stat.block),
  }
end

function copy_propagate.stat_for_list(cx, stat)
  return stat {
    value = copy_propagate.expr(cx, stat.value),
    block = copy_propagate.block(cx, stat.block),
  }
end

function copy_propagate.stat_repeat(cx, stat)
  return stat {
    until_cond = copy_propagate.expr(cx, stat.until_cond),
    block = copy_propagate.block(cx, stat.block),
  }
end

function copy_propagate.stat_block(cx, stat)
  return stat {
    block = copy_propagate.block(cx, stat.block),
  }
end

local function is_singleton_type(type)
  return std.is_ispace(type) or std.is_region(type) or
         std.is_list_of_regions(type) or
         std.is_partition(type) or std.is_cross_product(type)
end

function copy_propagate.stat_var(cx, stat)
  local value = stat.value
  if not value then return stat end
  if value:is(ast.typed.expr.ID) and
     std.is_rawref(value.expr_type) and
     not is_singleton_type(std.as_read(value.expr_type)) and
     std.type_eq(stat.type, std.as_read(value.expr_type))
  then
    cx:update_mapping(stat.symbol, value.value)
  end
  return stat {
    value = copy_propagate.expr(cx, value),
  }
end

function copy_propagate.stat_var_unpack(cx, stat)
  return stat {
    value = copy_propagate.expr(cx, stat.value),
  }
end

function copy_propagate.stat_assignment_or_reduce(cx, stat)
  return stat {
    lhs = copy_propagate.expr(cx, stat.lhs),
    rhs = copy_propagate.expr(cx, stat.rhs),
  }
end

function copy_propagate.stat_return(cx, stat)
  return stat {
    value = stat.value and copy_propagate.expr(cx, stat.value),
  }
end

function copy_propagate.stat_expr(cx, stat)
  return stat {
    expr = copy_propagate.expr(cx, stat.expr),
  }
end

function copy_propagate.stat_parallel_prefix(cx, stat)
  return stat {
    lhs = copy_propagate.expr(cx, stat.lhs),
    rhs = copy_propagate.expr(cx, stat.rhs),
    dir = copy_propagate.expr(cx, stat.dir),
  }
end

function copy_propagate.stat_raw_delete(cx, stat)
  return stat {
    value = copy_propagate.expr(cx, stat.value),
  }
end

function copy_propagate.pass_through_stat(cx, stat) return stat end

local copy_propagate_stat_table = {
  [ast.typed.stat.If]              = copy_propagate.stat_if,
  [ast.typed.stat.While]           = copy_propagate.stat_while,
  [ast.typed.stat.ForNum]          = copy_propagate.stat_for_num,
  [ast.typed.stat.ForList]         = copy_propagate.stat_for_list,
  [ast.typed.stat.Repeat]          = copy_propagate.stat_repeat,
  [ast.typed.stat.Block]           = copy_propagate.stat_block,

  [ast.typed.stat.Var]             = copy_propagate.stat_var,
  [ast.typed.stat.VarUnpack]       = copy_propagate.stat_var_unpack,
  [ast.typed.stat.Assignment]      = copy_propagate.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]          = copy_propagate.stat_assignment_or_reduce,
  [ast.typed.stat.Return]          = copy_propagate.stat_return,
  [ast.typed.stat.Expr]            = copy_propagate.stat_expr,

  [ast.typed.stat.ParallelPrefix]  = copy_propagate.stat_parallel_prefix,
  [ast.typed.stat.RawDelete]       = copy_propagate.stat_raw_delete,

  [ast.typed.stat.Break]           = copy_propagate.pass_through_stat,
  [ast.typed.stat.Fence]           = copy_propagate.pass_through_stat,

  [ast.typed.stat.Elseif]          = unreachable,
  [ast.typed.stat.Internal]        = unreachable,

  [ast.typed.stat.MustEpoch]         = unreachable,
  [ast.typed.stat.ParallelizeWith]   = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.IndexLaunchNum]    = unreachable,
  [ast.typed.stat.IndexLaunchList]   = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local copy_propagate_stat = ast.make_single_dispatch(
  copy_propagate_stat_table,
  {ast.typed.stat})

function copy_propagate.stat(cx, stat)
  return copy_propagate_stat(cx)(stat)
end

function copy_propagate.block(cx, node)
  cx:push_local_scope()
  local stats = node.stats:map(function(stat)
    return copy_propagate.stat(cx, stat)
  end)
  cx:pop_local_scope()
  return node { stats = stats }
end

function copy_propagate.top_task(node)
  local cx = context.new_global_scope()
  if node.body then
    collect_kills.block(cx, node.body)
    return node { body = copy_propagate.block(cx, node.body) }
  else
    return node
  end
end

function copy_propagate.entry(node)
  if node:is(ast.typed.top.Task) and node.config_options.leaf then
    return copy_propagate.top_task(node)
  else
    return node
  end
end

copy_propagate.pass_name = "copy_propagate"

return copy_propagate
