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

local function has_value(values, idx)
  return #values >= idx and values[idx] or false
end

function normalize.stat_var(node)
  if #node.symbols == 1 then
    local value = has_value(node.values, 1)
    return node {
      symbols = node.symbols[1],
      values = value,
    }
  else
    local temp_vars = terralib.newlist()
    local flattened = terralib.newlist()
    for idx = 1, #node.symbols do
      if has_value(node.values, idx) then
        local temp_var = std.newsymbol()
        temp_vars:insert(temp_var)
        flattened:insert(node {
          symbols = temp_var,
          values = node.values[idx]
        })
      end
    end

    for idx = 1, #node.symbols do
      if has_value(node.values, idx) then
        flattened:insert(node {
          symbols = node.symbols[idx],
          values = ast.specialized.expr.ID {
              value = temp_vars[idx],
              span = node.span,
              annotations = node.annotations,
            },
        })
      else
        flattened:insert(node {
          symbols = node.symbols[idx],
          values = false,
        })
      end
    end
    return flattened
  end
end

function normalize.stat_assignment_or_reduce(node)
  if #node.lhs == 1 then
    assert(#node.rhs == 1)
    return node {
      lhs = node.lhs[1],
      rhs = node.rhs[1],
    }
  else
    local temp_vars = node.lhs:map(function(lh) return std.newsymbol() end)
    local flattened = data.zip(temp_vars, node.rhs):map(function(pair)
      local symbol, rh = unpack(pair)
      return ast.specialized.stat.Var {
        symbols = symbol,
        values = rh,
        annotations = rh.annotations,
        span = rh.span,
      }
    end)

    data.zip(node.lhs, temp_vars):map(function(pair)
      local lh, symbol = unpack(pair)
      flattened:insert(node {
        lhs = lh,
        rhs = ast.specialized.expr.ID {
          value = symbol,
          span = lh.span,
          annotations = lh.annotations,
        },
      })
    end)

    return flattened
  end
end

function normalize.stat()
  return function(node, continuation)
    if node:is(ast.specialized.stat.Assignment) or
       node:is(ast.specialized.stat.Reduce) then
      local flattened = flatten_multifield_accesses(node)
      return normalize.stat_assignment_or_reduce(flattened)
    elseif node:is(ast.specialized.stat.Var) then
      return normalize.stat_var(node)
    else
      return continuation(node, true)
    end
  end
end

function normalize.top_task(node)
  return node {
    body = node.body and node.body {
      stats = ast.flatmap_node_continuation(
        normalize.stat(),
        node.body.stats)
      }
  }
end

function normalize.entry(node)
  if node:is(ast.specialized.top.Task) then
    return normalize.top_task(node)
  else
    return node
  end
end

return normalize
