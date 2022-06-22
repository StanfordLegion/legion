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

-- Regent Normalizer for Typed ASTs

-- Some expressions cannot be normalized until type checking, because the type
-- checker materializes implicit type castings and pointer deferences. The
-- remaining expressions are normalized in this normalizer.
--
-- Note that this normalizer is supposed to be used only by the parallelizability
-- checker and thus does not have the usual entry function.

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local normalize_access = {}

local function unreachable(cx, node) assert(false) end

function normalize_access.pass_through_expr(stats, expr) return expr end

local normalize_expr_factory = data.weak_memoize(function(field, read)
  assert(field ~= nil)
  assert(read ~= nil)
  return function(stats, expr)
    return expr { [field] = normalize_access.expr(stats, expr[field], read) }
  end
end)

local normalized_predicates = {
  [ast.typed.expr.ID]       = function(node) return true end,
  [ast.typed.expr.Constant] = function(node) return true end,
  [ast.typed.expr.Global]   = function(node) return true end,
  [ast.typed.expr.Function] = function(node) return true end,
  [ast.typed.expr.FieldAccess] =
    function(node)
      return node.field_name == "bounds" and
             node.value:is(ast.typed.expr.FieldAccess) and
             node.value.field_name == "ispace"
    end,
  [ast.typed.expr.Unary]    =
    function(node)
      return normalize_access.normalized(node.rhs)
    end,
  [ast.typed.expr.Binary]   =
    function(node)
      return normalize_access.normalized(node.lhs) and normalize_access.normalized(node.rhs)
    end,
  [ast.typed.expr.Cast]     =
    function(node) return normalize_access.normalized(node.arg) end,
  [ast.typed.expr.Ctor]     =
    function(node)
      return data.all(node.fields:map(function(field)
        return normalize_access.normalized(field.value)
      end))
    end,
}

normalize_access.normalized = data.weak_memoize(function(expr)
  local predicate = normalized_predicates[expr.node_type]
  return predicate and predicate(expr) or false
end)

normalize_access.expr_regent_cast = normalize_expr_factory("value", true)

function normalize_access.expr_field_access(stats, expr)
  local read = not (expr:is(ast.typed.expr.FieldAccess) or
                    expr:is(ast.typed.expr.IndexAccess) or
                    expr:is(ast.typed.expr.Deref))
  local value = normalize_access.expr(stats, expr.value, read)
  return expr { value = value }
end

normalize_access.expr_deref = normalize_expr_factory("value", true)

normalize_access.expr_address_of = normalize_expr_factory("value", false)

function normalize_access.expr_index_access(stats, expr)
  local index = normalize_access.expr(stats, expr.index, true)
  local value = normalize_access.expr(stats, expr.value, false)
  return expr {
    index = index,
    value = value,
  }
end

function normalize_access.expr_call(stats, expr)
  return expr {
    args = expr.args:map(function(value)
      return normalize_access.expr(stats, value, true)
    end),
  }
end

function normalize_access.expr_ctor(stats, expr)
  return expr {
    fields = expr.fields:map(function(field)
      return field {
        value = normalize_access.expr(stats, field.value, true),
      }
    end),
  }
end

normalize_access.expr_unary = normalize_expr_factory("rhs", true)

function normalize_access.expr_binary(stats, expr)
  local lhs = normalize_access.expr(stats, expr.lhs, true)
  local rhs = normalize_access.expr(stats, expr.rhs, true)
  return expr {
    lhs = lhs,
    rhs = rhs,
  }
end

function normalize_access.expr_cast(stats, expr)
  local read =
    not (expr.fn.value:ispointer() and std.as_read(expr.arg.expr_type):isarray())
  return expr {
    arg = normalize_access.expr(stats, expr.arg, read),
  }
end

local normalize_access_expr_table = {
  [ast.typed.expr.DynamicCast]                = normalize_access.expr_regent_cast,
  [ast.typed.expr.StaticCast]                 = normalize_access.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]                 = normalize_access.expr_regent_cast,
  [ast.typed.expr.FieldAccess]                = normalize_access.expr_field_access,
  [ast.typed.expr.Deref]                      = normalize_access.expr_deref,
  [ast.typed.expr.AddressOf]                  = normalize_access.expr_address_of,
  [ast.typed.expr.IndexAccess]                = normalize_access.expr_index_access,
  [ast.typed.expr.Ctor]                       = normalize_access.expr_ctor,
  [ast.typed.expr.Unary]                      = normalize_access.expr_unary,
  [ast.typed.expr.Binary]                     = normalize_access.expr_binary,
  [ast.typed.expr.Cast]                       = normalize_access.expr_cast,


  [ast.typed.expr.Call]                       = normalize_access.expr_call,
  [ast.typed.expr.MethodCall]                 = normalize_access.expr_call,
  [ast.typed.expr.RawFields]                  = normalize_access.pass_through_expr,
  [ast.typed.expr.RawFuture]                  = normalize_access.pass_through_expr,
  [ast.typed.expr.RawPhysical]                = normalize_access.pass_through_expr,
  [ast.typed.expr.RawRuntime]                 = normalize_access.pass_through_expr,
  [ast.typed.expr.RawTask]                    = normalize_access.pass_through_expr,
  [ast.typed.expr.RawValue]                   = normalize_access.pass_through_expr,
  [ast.typed.expr.ListInvert]                 = normalize_access.pass_through_expr,
  [ast.typed.expr.ListRange]                  = normalize_access.pass_through_expr,
  [ast.typed.expr.ListIspace]                 = normalize_access.pass_through_expr,
  [ast.typed.expr.ListFromElement]            = normalize_access.pass_through_expr,
  [ast.typed.expr.RegionRoot]                 = normalize_access.pass_through_expr,
  [ast.typed.expr.Projection]                 = normalize_access.pass_through_expr,
  [ast.typed.expr.FutureGetResult]            = normalize_access.pass_through_expr,

  -- Normal expressions
  [ast.typed.expr.ID]                         = normalize_access.pass_through_expr,
  [ast.typed.expr.Function]                   = normalize_access.pass_through_expr,
  [ast.typed.expr.Constant]                   = normalize_access.pass_through_expr,
  [ast.typed.expr.Global]                     = normalize_access.pass_through_expr,
  [ast.typed.expr.Null]                       = normalize_access.pass_through_expr,
  [ast.typed.expr.Isnull]                     = normalize_access.pass_through_expr,

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
  [ast.typed.expr.ParallelizerConstraint]     = unreachable,
  [ast.typed.expr.ImportIspace]               = unreachable,
  [ast.typed.expr.ImportRegion]               = unreachable,
  [ast.typed.expr.ImportPartition]            = unreachable,
  [ast.typed.expr.ImportCrossProduct]         = unreachable,
}

local normalize_access_expr = ast.make_single_dispatch(
  normalize_access_expr_table,
  {ast.typed.expr})

function normalize_access.expr(stats, expr, read)
  local expr = normalize_access_expr(stats)(expr, read)
  if read and not normalize_access.normalized(expr) then
    local temp_var = std.newsymbol(std.as_read(expr.expr_type))
    stats:insert(ast.typed.stat.Var {
      symbol = temp_var,
      type = std.as_read(temp_var:gettype()),
      value = expr,
      span = expr.span,
      annotations = ast.default_annotations(),
    })
    return ast.typed.expr.ID {
      value = temp_var,
      expr_type = temp_var:gettype(),
      span = expr.span,
      annotations = ast.default_annotations(),
    }
  else
    return expr
  end
end

function normalize_access.stat_if(stats, stat)
  local then_block = normalize_access.block(stat.then_block)
  local else_block = normalize_access.block(stat.else_block)
  stats:insert(stat {
    then_block = then_block,
    else_block = else_block,
  })
end

function normalize_access.stat_block(stats, stat)
  stats:insert(stat { block = normalize_access.block(stat.block) })
end

function normalize_access.stat_var(stats, stat)
  local value = stat.value and normalize_access.expr(stats, stat.value, false) or false
  stats:insert(stat { value = value })
end

function normalize_access.stat_var_unpack(stats, stat)
  local value = normalize_access.expr(stats, stat.value, true)
  stats:insert(stat { value = value })
end

function normalize_access.stat_assignment_or_reduce(stats, stat)
  local lhs = normalize_access.expr(stats, stat.lhs, false)
  local rhs = normalize_access.expr(stats, stat.rhs, true)
  stats:insert(stat {
    lhs = lhs,
    rhs = rhs,
  })
end

function normalize_access.stat_return(stats, stat)
  if stat.value and not stat.value:is(ast.typed.expr.ID) then
    local value = stat.value
    local temp_var = std.newsymbol(std.as_read(value.expr_type))
    stats:insert(ast.typed.stat.Var {
      symbol = temp_var,
      type = std.as_read(temp_var:gettype()),
      value = value,
      span = stat.span,
      annotations = ast.default_annotations(),
    })
    stats:insert(stat {
      value = ast.typed.expr.ID {
        value = temp_var,
        expr_type = temp_var:gettype(),
        span = value.span,
        annotations = ast.default_annotations(),
      }
    })
  else
    stats:insert(stat)
  end
end

function normalize_access.stat_expr(stats, stat)
  local expr = normalize_access.expr(stats, stat.expr, false)
  stats:insert(stat { expr = expr })
end

function normalize_access.pass_through_stat(stats, stat) stats:insert(stat) end

local normalize_access_stat_table = {
  [ast.typed.stat.If]              = normalize_access.stat_if,
  [ast.typed.stat.While]           = normalize_access.stat_block,
  [ast.typed.stat.ForNum]          = normalize_access.stat_block,
  [ast.typed.stat.ForList]         = normalize_access.stat_block,
  [ast.typed.stat.Repeat]          = normalize_access.stat_block,
  [ast.typed.stat.Block]           = normalize_access.stat_block,

  [ast.typed.stat.Var]             = normalize_access.stat_var,
  [ast.typed.stat.VarUnpack]       = normalize_access.stat_var_unpack,
  [ast.typed.stat.Assignment]      = normalize_access.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]          = normalize_access.stat_assignment_or_reduce,
  [ast.typed.stat.Return]          = normalize_access.stat_return,
  [ast.typed.stat.Expr]            = normalize_access.stat_expr,

  [ast.typed.stat.Break]           = normalize_access.pass_through_stat,
  [ast.typed.stat.ParallelPrefix]  = normalize_access.pass_through_stat,
  [ast.typed.stat.RawDelete]       = normalize_access.pass_through_stat,
  [ast.typed.stat.Fence]           = normalize_access.pass_through_stat,

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

local normalize_access_stat = ast.make_single_dispatch(
  normalize_access_stat_table,
  {ast.typed.stat})

function normalize_access.stat(stats, stat)
  normalize_access_stat(stats)(stat)
end

function normalize_access.block(node)
  local stats = terralib.newlist()
  node.stats:map(function(stat) normalize_access.stat(stats, stat) end)
  return node { stats = stats }
end

return normalize_access
