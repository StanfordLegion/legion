-- Copyright 2019 Stanford University, NVIDIA Corporation
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

-- Regent Normalization Pass

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local normalize = {}

-- Multi-field accesses are allowed only in unary, binary, dereference,
-- and casting expressions.

local function join_num_accessed_fields(a, b)
  if a == 1 then
    return b
  elseif b == 1 then
    return a
  elseif a ~= b then
    return false
  else
    return a
  end
end

local function get_num_accessed_fields(node)
  if not ast.is_node(node) then
    return 1

  elseif node:is(ast.specialized.expr.ID) then
    return 1

  elseif node:is(ast.specialized.expr.Constant) then
    return 1

  elseif node:is(ast.specialized.expr.FieldAccess) then
    if terralib.islist(node.field_name) then
      return get_num_accessed_fields(node.value) * #node.field_name
    else
      return get_num_accessed_fields(node.value)
    end

  elseif node:is(ast.specialized.expr.IndexAccess) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    if get_num_accessed_fields(node.index) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.MethodCall) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    for _, arg in pairs(node.args) do
      if get_num_accessed_fields(arg) > 1 then return false end
    end
    return 1

  elseif node:is(ast.specialized.expr.Call) then
    if get_num_accessed_fields(node.fn) > 1 then return false end
    for _, arg in pairs(node.args) do
      if get_num_accessed_fields(arg) > 1 then return false end
    end
    return 1

  elseif node:is(ast.specialized.expr.Ctor) then
    node.fields:map(function(field)
      if field:is(ast.specialized.expr.CtorListField) then
        if get_num_accessed_fields(field.value) > 1 then return false end
      elseif field:is(ast.specialized.expr.CtorListField) then
        if get_num_accessed_fields(field.num_expr) > 1 then return false end
        if get_num_accessed_fields(field.value) > 1 then return false end
      end
    end)
    return 1

  elseif node:is(ast.specialized.expr.RawContext) then
    return 1

  elseif node:is(ast.specialized.expr.RawFields) then
    return 1

  elseif node:is(ast.specialized.expr.RawPhysical) then
    return 1

  elseif node:is(ast.specialized.expr.RawRuntime) then
    return 1

  elseif node:is(ast.specialized.expr.RawValue) then
    return 1

  elseif node:is(ast.specialized.expr.Isnull) then
    if get_num_accessed_fields(node.pointer) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.New) then
    if get_num_accessed_fields(node.extent) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.Null) then
    return 1

  elseif node:is(ast.specialized.expr.DynamicCast) then
    return get_num_accessed_fields(node.value)

  elseif node:is(ast.specialized.expr.StaticCast) then
    return get_num_accessed_fields(node.value)

  elseif node:is(ast.specialized.expr.UnsafeCast) then
    return get_num_accessed_fields(node.value)

  elseif node:is(ast.specialized.expr.Ispace) then
    if get_num_accessed_fields(node.extent) > 1 then return false end
    if get_num_accessed_fields(node.start) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.Region) then
    if get_num_accessed_fields(node.ispace) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.Partition) then
    if get_num_accessed_fields(node.coloring) > 1 then return false end
    return 1

  elseif node:is(ast.specialized.expr.PartitionEqual) then
    return 1

  elseif node:is(ast.specialized.expr.PartitionByField) then
    return 1

  elseif node:is(ast.specialized.expr.Image) then
    return 1

  elseif node:is(ast.specialized.expr.Preimage) then
    return 1

  elseif node:is(ast.specialized.expr.CrossProduct) then
    return 1

  elseif node:is(ast.specialized.expr.CrossProductArray) then
    return 1

  elseif node:is(ast.specialized.expr.ListSlicePartition) then
    return 1

  elseif node:is(ast.specialized.expr.ListDuplicatePartition) then
    return 1

  elseif node:is(ast.specialized.expr.ListCrossProduct) then
    return 1

  elseif node:is(ast.specialized.expr.ListCrossProductComplete) then
    return 1

  elseif node:is(ast.specialized.expr.ListPhaseBarriers) then
    return 1

  elseif node:is(ast.specialized.expr.ListInvert) then
    return 1

  elseif node:is(ast.specialized.expr.ListRange) then
    return 1

  elseif node:is(ast.specialized.expr.PhaseBarrier) then
    return 1

  elseif node:is(ast.specialized.expr.DynamicCollective) then
    return 1

  elseif node:is(ast.specialized.expr.Advance) then
    return 1

  elseif node:is(ast.specialized.expr.Arrive) then
    return 1

  elseif node:is(ast.specialized.expr.Await) then
    return 1

  elseif node:is(ast.specialized.expr.DynamicCollectiveGetResult) then
    return 1

  elseif node:is(ast.specialized.expr.Copy) then
    return 1

  elseif node:is(ast.specialized.expr.Fill) then
    return 1

  elseif node:is(ast.specialized.expr.Acquire) then
    return 1

  elseif node:is(ast.specialized.expr.Release) then
    return 1

  elseif node:is(ast.specialized.expr.Unary) then
    return get_num_accessed_fields(node.rhs)

  elseif node:is(ast.specialized.expr.Binary) then
    return join_num_accessed_fields(get_num_accessed_fields(node.lhs),
                                    get_num_accessed_fields(node.rhs))

  elseif node:is(ast.specialized.expr.Deref) then
    return get_num_accessed_fields(node.value)

  elseif node:is(ast.specialized.expr.Cast) then
    return get_num_accessed_fields(node.args[1])

  else
    return 1

  end
end

local function has_all_valid_field_accesses(node)
  local valid = true
  data.zip(node.lhs, node.rhs):map(function(pair)
    if valid then
      local lh, rh = unpack(pair)
      local num_accessed_fields_lh = get_num_accessed_fields(lh)
      local num_accessed_fields_rh = get_num_accessed_fields(rh)
      local num_accessed_fields =
        join_num_accessed_fields(num_accessed_fields_lh,
                                 num_accessed_fields_rh)
      if num_accessed_fields == false then
        valid = false
      -- Special case when there is only one assignee for multiple
      -- values on the RHS
      elseif num_accessed_fields_lh == 1 and
             num_accessed_fields_rh > 1 then
        valid = false
      end
    end
  end)

  return valid
end

local function get_nth_field_access(node, idx)
  if node:is(ast.specialized.expr.FieldAccess) then
    local num_accessed_fields_value = get_num_accessed_fields(node.value)
    local num_accessed_fields = #node.field_name

    local idx1 = math.floor((idx - 1) / num_accessed_fields) + 1
    local idx2 = (idx - 1) % num_accessed_fields + 1

    return node {
      value = get_nth_field_access(node.value, idx1),
      field_name = node.field_name[idx2],
    }

  elseif node:is(ast.specialized.expr.Unary) then
    return node { rhs = get_nth_field_access(node.rhs, idx) }

  elseif node:is(ast.specialized.expr.Binary) then
    return node {
      lhs = get_nth_field_access(node.lhs, idx),
      rhs = get_nth_field_access(node.rhs, idx),
    }

  elseif node:is(ast.specialized.expr.Deref) then
    return node {
      value = get_nth_field_access(node.value, idx),
    }

  elseif node:is(ast.specialized.expr.DynamicCast) or
         node:is(ast.specialized.expr.StaticCast) or
         node:is(ast.specialized.expr.UnsafeCast) then
    return node {
      value = get_nth_field_access(node.value, idx),
    }

  elseif node:is(ast.specialized.expr.Cast) then
    return node {
      args = node.args:map(function(arg)
        return get_nth_field_access(arg, idx)
      end)
    }
  else
    return node
  end
end

local function flatten_multifield_accesses(node)
  if not has_all_valid_field_accesses(node) then
    report.error(node, "invalid use of multi-field access")
  end

  local flattened_lhs = terralib.newlist()
  local flattened_rhs = terralib.newlist()

  data.zip(node.lhs, node.rhs):map(function(pair)
    local lh, rh = unpack(pair)
    local num_accessed_fields =
      join_num_accessed_fields(get_num_accessed_fields(lh),
                               get_num_accessed_fields(rh))
    assert(num_accessed_fields ~= false, "unreachable")
    for idx = 1, num_accessed_fields do
      flattened_lhs:insert(get_nth_field_access(lh, idx))
      flattened_rhs:insert(get_nth_field_access(rh, idx))
    end
  end)

  if #node.lhs == flattened_lhs and #node.rhs == flattened_rhs then
    return node
  else
    return node {
      lhs = flattened_lhs,
      rhs = flattened_rhs,
    }
  end
end

-- Normalization for Expressions

local function pass_through_expr(stats, expr) return expr end

local function unreachable(stats, node) assert(false) end

local normalize_expr_factory = terralib.memoize(function(field, is_list, read)
  assert(field ~= nil)
  assert(is_list ~= nil)
  assert(read ~= nil)
  if is_list then
    return function(stats, expr)
      return expr {
        [field] = expr[field]:map(function(value) return normalize.expr(stats, value, read) end),
      }
    end

  else
    return function(stats, expr)
      return expr { [field] = normalize.expr(stats, expr[field], read) }
    end
  end
end)

local predicates = {
  [ast.specialized.expr.ID]       = function(node) return true end,
  [ast.specialized.expr.Constant] = function(node) return true end,
  [ast.specialized.expr.Function] = function(node) return true end,
  [ast.specialized.expr.FieldAccess] =
    function(node)
      return normalize.normalized(node.value)
    end,
  [ast.specialized.expr.Deref]       =
    function(node)
      return normalize.normalized(node.value)
    end,
  [ast.specialized.expr.IndexAccess] =
    function(node)
      return
        normalize.normalized(node.value) and
        normalize.normalized(node.index)
      end,
  [ast.specialized.expr.Unary]    =
    function(node)
      return normalize.normalized(node.rhs)
    end,
  [ast.specialized.expr.Binary]   =
    function(node)
      return normalize.normalized(node.lhs) and normalize.normalized(node.rhs)
    end,
  [ast.specialized.expr.Cast]     =
    function(node)
      return normalize.normalized(node.args[1])
    end,
  [ast.specialized.expr.Ctor]     =
    function(node)
      return data.all(node.fields:map(function(field)
        return normalize.normalized(field.value)
      end))
    end,
}

normalize.normalized = terralib.memoize(function(expr)
  local predicate = predicates[expr.node_type]
  return predicate and predicate(expr) or false
end)

local expr_regent_cast = normalize_expr_factory("value", false, true)

local function expr_ispace(stats, expr)
  local extent = normalize.expr(stats, expr.extent, true)
  local start = expr.start and normalize.expr(stats, expr.start, true) or false
  return expr {
    extent = extent,
    start = start,
  }
end

local expr_region = normalize_expr_factory("ispace", false, true)

local expr_field_access = normalize_expr_factory("value", false, true)

local expr_deref = normalize_expr_factory("value", false, true)

local function expr_index_access(stats, expr)
  local value = normalize.expr(stats, expr.value, true)
  local index = normalize.expr(stats, expr.index, true)
  return expr {
    index = index,
    value = value,
  }
end

local function is_projection(node)
  return node:is(ast.specialized.expr.IndexAccess) and
         (normalize.normalized(node.value) or is_projection(node.value)) and
         normalize.normalized(node.index)
end

local function expr_call(stats, expr)
  local args = expr.args
  -- TODO: We handle task launches specially here to make the index launch optimizer
  --       (and potentially other optimization passes as well) happy
  if std.is_task(expr.fn.value) then
    args = args:map(function(arg)
      return normalize.expr(stats, arg, not is_projection(arg))
    end)
  else
    args = args:map(function(arg) return normalize.expr(stats, arg, true) end)
  end
  return expr { args = args }
end

local expr_method_call = normalize_expr_factory("args", true, true)

local expr_ctor = normalize_expr_factory("fields", true, false)

local expr_ctor_field = normalize_expr_factory("value", false, true)

local expr_is_null = normalize_expr_factory("pointer", false, true)

local expr_unary = normalize_expr_factory("rhs", false, true)

local function expr_binary(stats, expr)
  local lhs = normalize.expr(stats, expr.lhs, true)
  local rhs = normalize.expr(stats, expr.rhs, true)
  return expr {
    lhs = lhs,
    rhs = rhs,
  }
end

local expr_cast = normalize_expr_factory("args", true, true)

local expr_import_ispace = normalize_expr_factory("value", false, true)

local function expr_import_region(stats, expr)
  local ispace    = normalize.expr(stats, expr.ispace   , true)
  local value     = normalize.expr(stats, expr.value    , true)
  local field_ids = normalize.expr(stats, expr.field_ids, true)
  return expr {
    ispace = ispace,
    value = value,
    field_ids = field_ids,
  }
end

local function expr_import_partition(stats, expr)
  local region = normalize.expr(stats, expr.region, true)
  local colors = normalize.expr(stats, expr.colors, true)
  local value  = normalize.expr(stats, expr.value,  true)
  return expr {
    region = region,
    colors = colors,
    value = value,
  }
end

local normalize_expr_table = {
  [ast.specialized.expr.DynamicCast]                = expr_regent_cast,
  [ast.specialized.expr.StaticCast]                 = expr_regent_cast,
  [ast.specialized.expr.UnsafeCast]                 = expr_regent_cast,
  [ast.specialized.expr.Ispace]                     = expr_ispace,
  [ast.specialized.expr.Region]                     = expr_region,
  [ast.specialized.expr.FieldAccess]                = expr_field_access,
  [ast.specialized.expr.Deref]                      = expr_deref,
  [ast.specialized.expr.IndexAccess]                = expr_index_access,
  [ast.specialized.expr.MethodCall]                 = expr_method_call,
  [ast.specialized.expr.Call]                       = expr_call,
  [ast.specialized.expr.Ctor]                       = expr_ctor,
  [ast.specialized.expr.CtorListField]              = expr_ctor_field,
  [ast.specialized.expr.CtorRecField]               = expr_ctor_field,
  [ast.specialized.expr.Isnull]                     = expr_is_null,
  [ast.specialized.expr.Unary]                      = expr_unary,
  [ast.specialized.expr.Binary]                     = expr_binary,
  [ast.specialized.expr.Cast]                       = expr_cast,

  -- Normal expressions
  [ast.specialized.expr.ID]                         = pass_through_expr,
  [ast.specialized.expr.Function]                   = pass_through_expr,
  [ast.specialized.expr.Constant]                   = pass_through_expr,

  -- Expressions that do not need to be normalized
  [ast.specialized.expr.New]                        = pass_through_expr,
  [ast.specialized.expr.Null]                       = pass_through_expr,
  [ast.specialized.expr.RawContext]                 = pass_through_expr,
  [ast.specialized.expr.RawFields]                  = pass_through_expr,
  [ast.specialized.expr.RawPhysical]                = pass_through_expr,
  [ast.specialized.expr.RawRuntime]                 = pass_through_expr,
  [ast.specialized.expr.RawValue]                   = pass_through_expr,
  [ast.specialized.expr.Partition]                  = pass_through_expr,
  [ast.specialized.expr.PartitionEqual]             = pass_through_expr,
  [ast.specialized.expr.PartitionByField]           = pass_through_expr,
  [ast.specialized.expr.PartitionByRestriction]     = pass_through_expr,
  [ast.specialized.expr.Image]                      = pass_through_expr,
  [ast.specialized.expr.Preimage]                   = pass_through_expr,
  [ast.specialized.expr.CrossProduct]               = pass_through_expr,
  [ast.specialized.expr.CrossProductArray]          = pass_through_expr,
  [ast.specialized.expr.ListSlicePartition]         = pass_through_expr,
  [ast.specialized.expr.ListDuplicatePartition]     = pass_through_expr,
  [ast.specialized.expr.ListCrossProduct]           = pass_through_expr,
  [ast.specialized.expr.ListCrossProductComplete]   = pass_through_expr,
  [ast.specialized.expr.ListPhaseBarriers]          = pass_through_expr,
  [ast.specialized.expr.ListInvert]                 = pass_through_expr,
  [ast.specialized.expr.ListRange]                  = pass_through_expr,
  [ast.specialized.expr.ListIspace]                 = pass_through_expr,
  [ast.specialized.expr.ListFromElement]            = pass_through_expr,
  [ast.specialized.expr.PhaseBarrier]               = pass_through_expr,
  [ast.specialized.expr.DynamicCollective]          = pass_through_expr,
  [ast.specialized.expr.DynamicCollectiveGetResult] = pass_through_expr,
  [ast.specialized.expr.Advance]                    = pass_through_expr,
  [ast.specialized.expr.Adjust]                     = pass_through_expr,
  [ast.specialized.expr.Arrive]                     = pass_through_expr,
  [ast.specialized.expr.Await]                      = pass_through_expr,
  [ast.specialized.expr.Copy]                       = pass_through_expr,
  [ast.specialized.expr.Fill]                       = pass_through_expr,
  [ast.specialized.expr.Acquire]                    = pass_through_expr,
  [ast.specialized.expr.Release]                    = pass_through_expr,
  [ast.specialized.expr.AttachHDF5]                 = pass_through_expr,
  [ast.specialized.expr.DetachHDF5]                 = pass_through_expr,
  [ast.specialized.expr.AllocateScratchFields]      = pass_through_expr,
  [ast.specialized.expr.WithScratchFields]          = pass_through_expr,
  [ast.specialized.expr.RegionRoot]                 = pass_through_expr,
  [ast.specialized.expr.Condition]                  = pass_through_expr,
  [ast.specialized.expr.ImportIspace]               = expr_import_ispace,
  [ast.specialized.expr.ImportRegion]               = expr_import_region,
  [ast.specialized.expr.ImportPartition]            = expr_import_partition,

  [ast.specialized.expr.LuaTable]                   = pass_through_expr,
}

local normalize_expr = ast.make_single_dispatch(
  normalize_expr_table,
  {ast.specialized.expr})

function normalize.expr(stats, expr, read)
  local expr = normalize_expr(stats)(expr, read)
  if read and not normalize.normalized(expr) then
    local temp_var = std.newsymbol()
    stats:insert(ast.specialized.stat.Var {
      symbols = temp_var,
      values = expr,
      span = expr.span,
      annotations = ast.default_annotations(),
    })
    return ast.specialized.expr.ID {
      value = temp_var,
      span = expr.span,
      annotations = ast.default_annotations(),
    }
  else
    return expr
  end
end

-- Normalization for Statements

local function stat_if(stats, stat)
  local cond = normalize.expr(stats, stat.cond, true)
  local then_block = normalize.block(stat.then_block)
  local else_block = normalize.block(stat.else_block)

  for idx = #stat.elseif_blocks, 1, -1 do
    local elseif_stats = terralib.newlist()

    local elseif_block = stat.elseif_blocks[idx]
    local elseif_cond = normalize.expr(elseif_stats, elseif_block.cond, true)
    elseif_stats:insert(ast.specialized.stat.If {
      cond = elseif_cond,
      then_block = normalize.block(elseif_block.block),
      -- TODO: We will set this to false eventually
      elseif_blocks = terralib.newlist(),
      else_block = else_block,
      span = elseif_block.span,
      annotations = stat.annotations,
    })
    else_block = else_block { stats = elseif_stats }
  end

  stats:insert(stat {
    cond = cond,
    then_block = then_block,
    -- TODO: We will set this to false eventually
    elseif_blocks = terralib.newlist(),
    else_block = else_block,
  })
end

local function stat_while(stats, stat)
  local cond_stats = terralib.newlist()
  local cond = normalize.expr(cond_stats, stat.cond, true)
  if not cond:is(ast.specialized.expr.ID) then
    local cond_var = std.newsymbol()
    cond_stats:insert(ast.specialized.stat.Var {
      symbols = cond_var,
      values = cond,
      span = cond.span,
      annotations = ast.default_annotations(),
    })
    cond = ast.specialized.expr.ID {
      value = cond_var,
      span = cond.span,
      annotations = ast.default_annotations(),
    }
  end
  local block = normalize.block(stat.block)
  local block_stats = block.stats

  cond_stats:map(function(stat)
    if stat:is(ast.specialized.stat.Var) and stat.symbols == cond.value then
      block_stats:insert(ast.specialized.stat.Assignment {
        lhs = ast.specialized.expr.ID {
          value = cond.value,
          span = cond.span,
          annotations = ast.default_annotations(),
        },
        rhs = stat.values,
        span = cond.span,
        annotations = ast.default_annotations(),
      })
    else
      block_stats:insert(stat)
    end
  end)

  stats:insertall(cond_stats)
  stats:insert(stat {
    cond = cond,
    block = block { stats = block_stats },
  })
end

local function stat_for_num(stats, stat)
  local values = stat.values:map(function(value)
    return normalize.expr(stats, value, true)
  end)
  stats:insert(stat {
    values = values,
    block = normalize.block(stat.block),
  })
end

local function stat_for_list(stats, stat)
  local value = normalize.expr(stats, stat.value, true)
  stats:insert(stat {
    value = value,
    block = normalize.block(stat.block),
  })
end

local function stat_repeat(stats, stat)
  local block = normalize.block(stat.block)
  local block_stats = block.stats
  local until_cond = normalize.expr(block_stats, stat.until_cond, true)
  stats:insert(stat {
    until_cond = until_cond,
    block = block { stats = block_stats },
  })
end

local function stat_block(stats, stat)
  stats:insert(stat { block = normalize.block(stat.block) })
end

local function has_value(values, idx)
  return #values >= idx and values[idx] or false
end

local function stat_var(stats, stat)
  local values = stat.values:map(function(value)
    return normalize.expr(stats, value, false)
  end)
  if #stat.symbols == 1 then
    stats:insert(stat {
      symbols = stat.symbols[1],
      values = has_value(values, 1),
    })

  else
    local temp_vars = terralib.newlist()
    for idx = 1, #stat.symbols do
      if has_value(values, idx) then
        local temp_var = std.newsymbol()
        temp_vars:insert(temp_var)
        stats:insert(stat {
          symbols = temp_var,
          values = values[idx]
        })
      end
    end

    for idx = 1, #stat.symbols do
      if has_value(values, idx) then
        stats:insert(stat {
          symbols = stat.symbols[idx],
          values = ast.specialized.expr.ID {
              value = temp_vars[idx],
              span = stat.span,
              annotations = stat.annotations,
            },
        })

      else
        stats:insert(stat {
          symbols = stat.symbols[idx],
          values = false,
        })
      end
    end
  end
end

local function stat_var_unpack(stats, stat)
  stats:insert(stat {
    value = normalize.expr(stats, stat.value, true)
  })
end

local function stat_return(stats, stat)
  local value = stat.value and normalize.expr(stats, stat.value, true) or false
  stats:insert(stat { value = value })
end

local function stat_assignment_or_reduce(stats, stat)
  stat = flatten_multifield_accesses(stat)

  if #stat.lhs == 1 then
    assert(#stat.rhs == 1)
    local lhs = normalize.expr(stats, stat.lhs[1], false)
    local rhs = normalize.expr(stats, stat.rhs[1], not normalize.normalized(lhs))
    stats:insert(stat {
      lhs = lhs,
      rhs = rhs,
    })

  else
    local temp_vars = stat.lhs:map(function(lh) return std.newsymbol() end)
    local lhs = stat.lhs:map(function(lh) return normalize.expr(stats, lh, false) end)
    local rhs = stat.rhs:map(function(rh) return normalize.expr(stats, rh, false) end)

    data.zip(temp_vars, rhs):map(function(pair)
      local symbol, rh = unpack(pair)
      stats:insert(ast.specialized.stat.Var {
        symbols = symbol,
        values = rh,
        annotations = rh.annotations,
        span = rh.span,
      })
    end)
    data.zip(lhs, temp_vars):map(function(pair)
      local lh, symbol = unpack(pair)
      stats:insert(stat {
        lhs = lh,
        rhs = ast.specialized.expr.ID {
          value = symbol,
          span = lh.span,
          annotations = lh.annotations,
        },
      })
    end)
  end
end

local function stat_expr(stats, stat)
  local expr = normalize.expr(stats, stat.expr, false)
  stats:insert(stat { expr = expr })
end

local function stat_raw_delete(stats, stat)
  local value = normalize.expr(stats, stat.value, false)
  stats:insert(stat { value = value })
end

local function stat_parallel_prefix(stats, stat)
  local dir = normalize.expr(stats, stat.dir, true)
  stats:insert(stat { dir = dir })
end

local function pass_through_stat(stats, stat) stats:insert(stat) end

local normalize_stat_table = {
  [ast.specialized.stat.If]              = stat_if,
  [ast.specialized.stat.While]           = stat_while,
  [ast.specialized.stat.ForNum]          = stat_for_num,
  [ast.specialized.stat.ForList]         = stat_for_list,
  [ast.specialized.stat.Repeat]          = stat_repeat,
  [ast.specialized.stat.MustEpoch]       = stat_block,
  [ast.specialized.stat.Block]           = stat_block,
  [ast.specialized.stat.ParallelizeWith] = stat_block,

  [ast.specialized.stat.Var]             = stat_var,
  [ast.specialized.stat.VarUnpack]       = stat_var_unpack,
  [ast.specialized.stat.Return]          = stat_return,
  [ast.specialized.stat.Break]           = pass_through_stat,
  [ast.specialized.stat.Assignment]      = stat_assignment_or_reduce,
  [ast.specialized.stat.Reduce]          = stat_assignment_or_reduce,
  [ast.specialized.stat.Expr]            = stat_expr,
  [ast.specialized.stat.ParallelPrefix]  = stat_parallel_prefix,
  [ast.specialized.stat.RawDelete]       = stat_raw_delete,
  [ast.specialized.stat.Fence]           = pass_through_stat,

  [ast.specialized.stat.Elseif]          = unreachable,
}

local normalize_stat = ast.make_single_dispatch(
  normalize_stat_table,
  {ast.specialized.stat})

function normalize.stat(stats, stat)
  normalize_stat(stats)(stat)
end

function normalize.block(node)
  local stats = terralib.newlist()
  node.stats:map(function(stat)
    normalize.stat(stats, stat)
  end)
  return node { stats = stats }
end

function normalize.top_task(node)
  return node { body = node.body and normalize.block(node.body) }
end

function normalize.entry(node)
  if node:is(ast.specialized.top.Task) then
    return normalize.top_task(node)
  else
    return node
  end
end

return normalize
