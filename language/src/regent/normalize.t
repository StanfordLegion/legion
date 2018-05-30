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

local function has_reference(node, symbols)
  if node == nil then return false end
  local found = false
  ast.traverse_node_postorder(function(node)
    if not found and node:is(ast.specialized.expr.ID) then
      found = symbols[node.value:hasname()] ~= nil
    end
  end, node)
  return found
end

function normalize.stat_var(node)
  if #node.symbols == 1 then return node
  else
    local symbols = {}
    local needs_temporary = terralib.newlist()
    local temporaries = {}
    for idx = 1, #node.symbols do
      if has_reference(node.values[idx], symbols) then
        needs_temporary:insert(idx)
        temporaries[idx] = std.newsymbol()
      end
      symbols[node.symbols[idx]:hasname()] = true
    end

    local flattened = terralib.newlist()
    for idx = 1, #needs_temporary do
      flattened:insert(node {
        symbols = terralib.newlist {temporaries[needs_temporary[idx]]},
        values = terralib.newlist {node.values[needs_temporary[idx]]},
      })
    end

    for idx = 1, #node.symbols do
      if temporaries[idx] then
        flattened:insert(node {
          symbols = terralib.newlist {node.symbols[idx]},
          values = terralib.newlist {
            ast.specialized.expr.ID {
              value = temporaries[idx],
              span = node.span,
              annotations = node.annotations,
            }
          },
        })
      else
        flattened:insert(node {
          symbols = terralib.newlist {node.symbols[idx]},
          values = terralib.newlist {node.values[idx]},
        })
      end
    end
    return flattened
  end
end

local function get_index_path(expr)
  if expr:is(ast.typed.expr.Constant) and
     expr.expr_type.type == "integer" then
    return data.newtuple(tostring(expr.value))
  elseif expr:is(ast.typed.expr.Cast) then
    return get_index_path(expr.arg)
  elseif expr:is(ast.typed.expr.Ctor) then
    local fields = expr.fields:map(get_index_path)
    local path = fields[1]
    for idx = 2, #fields do
      path = path .. fields[idx]
    end
    return path
  elseif expr:is(ast.typed.expr.CtorListField) or
         expr:is(ast.typed.expr.CtorRecField) then
    return get_index_path(expr.value)
  else
    return data.newtuple("*")
  end
end

local function get_access_paths(expr)
  if expr:is(ast.typed.expr.FieldAccess) then
    local paths, next_exprs = get_access_paths(expr.value)
    return paths:map(function(path)
      return path .. data.newtuple(expr.field_name)
    end), next_exprs

  elseif expr:is(ast.typed.expr.IndexAccess) then
    local paths, next_exprs = get_access_paths(expr.value)
    local index_path = get_index_path(expr.index)
    next_exprs:insert(expr.index)
    return paths:map(function(path)
      return path .. index_path
    end), next_exprs

  elseif expr:is(ast.typed.expr.Deref) then
    return expr.expr_type.bounds_symbols:map(function(bound)
      return data.newtuple(bound)
    end), terralib.newlist {expr.value}

  elseif expr:is(ast.typed.expr.ID) then
    return terralib.newlist {data.newtuple(expr.value)},
      terralib.newlist()

  else
    assert(false)
  end
end

local pretty = require("regent/pretty")

local function symbols_aliased(cx, sym1, sym2)
  local function is_value_type(ty)
    return ty:isprimitive() or std.is_fspace_instance(ty) or ty:isstruct()
  end

  local function pointers_aliased(ty1, ty2)
    assert(ty1:ispointer() or ty2:ispointer())
    -- Regent doesn't have an address-of operator, so the address of
    -- a value-typed variable can neither be taken nor aliased. However,
    -- pointers to arrays can be passed first to a Terra function and then
    -- back to the Regent code, so the alasing between a pointer and an array
    -- is still possible.
    if ty1:ispointer() then
      if ty2:ispointer() or ty2:isarray() then
        return true
      else
        return false
      end
    end
    assert(ty2:ispointer())
    return pointers_aliased(ty2, ty1)
  end

  if sym1 == sym2 then return true
  else
    local ty1 = std.as_read(sym1:gettype())
    local ty2 = std.as_read(sym2:gettype())
    if std.is_region(ty1) and std.is_region(ty2) then
      return std.type_maybe_eq(ty1:fspace(), ty2:fspace()) and
             not std.check_constraint(cx,
               std.constraint(ty1, ty2, std.disjointness))
    elseif ty1:isarray() and ty2:isarray() then
      return false
    elseif ty1:ispointer() or ty2:ispointer() then
      return pointers_aliased(ty1, ty2)
    elseif is_value_type(ty1) or is_value_type(ty2) then
      return false
    else
      assert(false, "should be unreachable")
    end
  end
  return false
end

local function refs_aliased(cx, path1, path2)
  local len = math.min(#path1, #path2)
  local sym1 = path1[1]
  local sym2 = path2[1]

  if symbols_aliased(cx, sym1, sym2) then
    for idx = 2, len do
      local e1 = path1[idx]
      local e2 = path2[idx]
      if e1 ~= e2 and not (e1 == "*" or e2 == "*") then
        return false
      end
    end
    return true
  end
  return false
end

local function can_alias(node, cx, updates)
  if ast.is_node(node) and node:is(ast.typed.expr) then
    if std.is_ref(node.expr_type) or std.is_rawref(node.expr_type) then
      local paths, next_exprs = get_access_paths(node)
      for i = 1, #paths do
        for j = 1, #updates do
          if refs_aliased(cx, updates[j], paths[i]) then
            return true
          end
        end
      end

      for i = 1, #next_exprs do
        if can_alias(next_exprs[i], cx, updates) then
          return true
        end
      end

    else
      for k, child in pairs(node) do
        if k ~= "node_type" and k ~= "node_id" then
          if can_alias(child, cx, updates) then
            return true
          end
        end
      end
    end

  elseif terralib.islist(node) then
    for _, child in ipairs(node) do
      if can_alias(child, cx, updates) then
        return true
      end
    end
  end

  return false
end

local function update_alias(expr, updates)
  local paths = get_access_paths(expr)
  updates:insertall(paths)
end

function normalize.stat_assignment_or_reduce(cx, node)
  if #node.lhs == 1 then
    assert(#node.rhs == 1)
    return node {
      lhs = node.lhs[1],
      rhs = node.rhs[1],
    }
  else
    local updates = terralib.newlist()
    local needs_temporary = terralib.newlist()
    local temporaries = {}
    for idx = 1, #node.lhs do
      local lh = node.lhs[idx]
      local rh = node.rhs[idx]
      if can_alias(rh, cx, updates) then
        needs_temporary:insert(idx)
        temporaries[idx] = std.newsymbol(std.as_read(rh.expr_type))
      end
      update_alias(lh, updates)
    end

    local flattened = terralib.newlist()
    for idx = 1, #needs_temporary do
      local symbol = temporaries[needs_temporary[idx]]
      flattened:insert(ast.typed.stat.Var {
        symbol = symbol,
        type = symbol:gettype(),
        value = node.rhs[needs_temporary[idx]],
        annotations = node.annotations,
        span = node.span,
      })
    end

    for idx = 1, #node.lhs do
      if temporaries[idx] then
        flattened:insert(node {
          lhs = node.lhs[idx],
          rhs = ast.typed.expr.ID {
            value = temporaries[idx],
            expr_type = std.rawref(&temporaries[idx]:gettype()),
            span = node.span,
            annotations = node.annotations,
          },
        })
      else
        flattened:insert(node {
          lhs = node.lhs[idx],
          rhs = node.rhs[idx],
        })
      end
    end

    return flattened
  end
end

--
-- De-sugar statement "var ip = image(r, p, f)" into the following statements:
--
-- var coloring : legion_domain_point_coloring_t
-- coloring = legion_domain_point_coloring_create()
-- for color in p.colors do
--   legion_domain_point_coloring_color_domain(
--     coloring, color, f(p[color].bounds))
-- end
-- var ip = partition(aliased, r, coloring)
-- legion_domain_point_coloring_destroy(coloring)
--
local capi = std.c

local function desugar_image_by_task(cx, node)
  local parent = node.value.parent.value
  local parent_type = parent:gettype()
  local partition = node.value.partition
  local partition_type = std.as_read(partition.expr_type)
  local image_partition_type = node.type

  local stats = terralib.newlist()

  local coloring_symbol =
    regentlib.newsymbol(capi.legion_domain_point_coloring_t)
  local coloring_expr = ast_util.mk_expr_id(coloring_symbol)
  stats:insert(
    ast_util.mk_stat_var(
      coloring_symbol, nil,
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_create)))

  local colors_symbol = regentlib.newsymbol(partition_type:colors())
  local color_symbol =
    regentlib.newsymbol(partition_type:colors().index_type(colors_symbol))
  local colors_expr = ast_util.mk_expr_colors_access(partition)
  local subregion_type = partition_type:subregion_dynamic()
  std.add_constraint(cx, subregion_type, partition_type, std.subregion, false)

  local subregion_expr =
    ast_util.mk_expr_index_access(partition,
                                  ast_util.mk_expr_id(color_symbol),
                                  subregion_type)
  local rect_expr =
    ast_util.mk_expr_call(node.value.task.value,
                          ast_util.mk_expr_bounds_access(subregion_expr))
  local loop_body =
    ast_util.mk_stat_expr(
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_color_domain,
                            terralib.newlist { coloring_expr,
                                               ast_util.mk_expr_id(color_symbol),
                                               rect_expr }))

  stats:insert(ast_util.mk_stat_var(colors_symbol, nil, colors_expr))
  stats:insert(
    ast_util.mk_stat_for_list(color_symbol,
                              ast_util.mk_expr_id(colors_symbol),
                              ast_util.mk_block(loop_body)))

  stats:insert(
    ast_util.mk_stat_var(node.symbol, image_partition_type,
                         ast_util.mk_expr_partition(image_partition_type,
                                                    ast_util.mk_expr_id(colors_symbol),
                                                    coloring_expr)))
  std.add_constraint(cx, image_partition_type, parent_type, std.subregion, false)

  stats:insert(
    ast_util.mk_stat_expr(
      ast_util.mk_expr_call(capi.legion_domain_point_coloring_destroy,
                            coloring_expr)))

  return stats
end

function normalize.stat(cx)
  return function(node, continuation)
    if node:is(ast.specialized.stat.Assignment) or
       node:is(ast.specialized.stat.Reduce) then
      node = flatten_multifield_accesses(node)
      return node
    elseif node:is(ast.typed.stat.Assignment) or
           node:is(ast.typed.stat.Reduce) then
      return normalize.stat_assignment_or_reduce(cx, node)
    elseif not std.config["parallelize"] and
           node:is(ast.typed.stat.ParallelizeWith) then
      node = ast.typed.stat.Block {
        block = node.block,
        span = node.span,
        annotations = node.annotations,
      }
      return continuation(node, true)
    elseif node:is(ast.specialized.stat.Var) then
      return normalize.stat_var(node)
    elseif node:is(ast.typed.stat.Var) and node.value and
           node.value:is(ast.typed.expr.ImageByTask) then
      return desugar_image_by_task(cx, node)
    else
      return continuation(node, true)
    end
  end
end

function normalize.top_task(node)
  local cx = {}
  if node:is(ast.typed) then
    cx.constraints = node.prototype:get_constraints()
  end
  return node {
    body = node.body and node.body {
      stats = ast.flatmap_node_continuation(
        normalize.stat(cx),
        node.body.stats)
      }
  }
end

function normalize.entry(node)
  if node:is(ast.specialized.top.Task) or node:is(ast.typed.top.Task) then
    return normalize.top_task(node)
  else
    return node
  end
end

return normalize
