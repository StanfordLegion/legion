-- Copyright 2022 Stanford University, NVIDIA Corporation
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

-- Legion Type Checker

local affine_helper = require("regent/affine_helper")
local ast = require("regent/ast")
local data = require("common/data")
local pretty = require("regent/pretty")
local report = require("regent/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local type_check = {}

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

function context:new_local_scope(must_epoch, breakable_loop)
  assert(not (self.must_epoch and must_epoch))
  must_epoch = self.must_epoch or must_epoch or false
  breakable_loop = self.breakable_loop or breakable_loop or false
  local cx = {
    type_env = self.type_env:new_local_scope(),
    privileges = self.privileges,
    constraints = self.constraints,
    region_universe = self.region_universe,
    expected_return_type = self.expected_return_type,
    is_cuda = self.is_cuda,
    fixup_nodes = self.fixup_nodes,
    must_epoch = must_epoch,
    breakable_loop = breakable_loop,
  }
  setmetatable(cx, context)
  return cx
end

function context:new_task_scope(expected_return_type, is_cuda)
  local cx = {
    type_env = self.type_env:new_local_scope(),
    privileges = data.newmap(),
    constraints = data.new_recursive_map(2),
    region_universe = data.newmap(),
    expected_return_type = {expected_return_type},
    is_cuda = is_cuda,
    fixup_nodes = terralib.newlist(),
    must_epoch = false,
    breakable_loop = false,
  }
  setmetatable(cx, context)
  return cx
end

function context.new_global_scope(type_env)
  local cx = {
    type_env = symbol_table.new_global_scope(type_env),
  }
  setmetatable(cx, context)
  return cx
end

function context:intern_region(region_type)
  assert(self.region_universe)
  self.region_universe[region_type] = true
end

function context:get_return_type()
  if self.expected_return_type then
    return self.expected_return_type[1]
  end
end

function context:set_return_type(t)
  assert(self.expected_return_type)
  self.expected_return_type[1] = t
end

function type_check.region_field(cx, node, region, prefix_path, value_type)
  assert(std.is_symbol(region))
  local field_name = node.field_name
  if not data.is_tuple(field_name) then
    field_name = data.newtuple(node.field_name)
  end
  local field_path = prefix_path .. field_name
  local field_type = value_type
  for _, f in ipairs(field_name) do
    field_type = std.get_field(field_type, f)
    if not field_type then
      break
    end
  end
  if not field_type then
    report.error(node, "no field '" .. field_name:mkstring(".") ..
                "' in region " .. (data.newtuple(region) .. prefix_path):mkstring("."))
  end

  return type_check.region_fields(
    cx, node.fields, region, field_path, field_type)
end

function type_check.region_fields(cx, node, region, prefix_path, value_type)
  assert(std.is_symbol(region))
  if not node then
    return terralib.newlist({prefix_path})
  end
  local result = terralib.newlist()
  for _, field in ipairs(node) do
    result:insertall(
      type_check.region_field(cx, field, region, prefix_path, value_type))
  end
  return result
end

function type_check.region_bare(cx, node)
  local region = node.symbol
  if not std.is_symbol(region) then
    report.error(node, "privilege target is not a region")
  end
  local region_type = region:gettype()
  if not (std.type_supports_privileges(region_type)) then
    report.error(node, "type mismatch: expected a region but got " .. tostring(region_type))
  end
  return region
end

local region_field = {}

function region_field:__tostring()
  local region = self.region
  local fields = self.fields
  if #fields == 1 then
    return terralib.newlist({tostring(region), unpack(fields[1])}):concat(".")
  else
    local result = tostring(region) .. ".{" .. fields:map(function(field) return field:concat(".") end):concat(", ") .. "}"
  end
end

function type_check.region_root(cx, node)
  local region = type_check.region_bare(cx, node)
  local region_type = region:gettype()
  local value_type = region_type:fspace()
  return setmetatable({
    region = region,
    fields = type_check.region_fields(
      cx, node.fields, region, data.newtuple(), value_type),
  }, region_field)
end

function type_check.expr_region_root(cx, node)
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)
  if not std.type_supports_privileges(region_type) then
    report.error(node, "type mismatch: expected a region but got " .. tostring(region_type))
  end

  local region_symbol
  if region:is(ast.typed.expr.ID) then
    region_symbol = region.value
  else
    region_symbol = std.newsymbol(std.as_read(region.expr_type))
  end

  local value_type = region_type:fspace()
  return ast.typed.expr.RegionRoot {
    region = region,
    fields = type_check.region_fields(
      cx, node.fields, region_symbol, data.newtuple(), value_type),
    expr_type = region_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.regions(cx, node)
  return node:map(
    function(region) return type_check.region_root(cx, region) end)
end

function type_check.condition_variable(cx, node)
  local symbol = node.symbol
  local var_type = symbol:gettype()
  while std.is_list(var_type) do
    var_type = var_type.element_type
  end
  if not std.is_phase_barrier(var_type)  then
    report.error(node, "type mismatch: expected " .. tostring(std.phase_barrier) .. " but got " .. tostring(var_type))
  end
  return symbol
end

function type_check.condition_variables(cx, node)
  return node:map(
    function(region) return type_check.condition_variable(cx, region) end)
end

function type_check.privilege_kind(cx, node)
  if node:is(ast.privilege_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.privilege_kinds(cx, node)
  return node:map(
    function(privilege) return type_check.privilege_kind(cx, privilege) end)
end

function type_check.privilege(cx, node)
  local privileges = type_check.privilege_kinds(cx, node.privileges)
  local region_fields = type_check.regions(cx, node.regions)
  return data.flatmap(
    function(privilege) return std.privileges(privilege, region_fields) end,
    privileges)
end

function type_check.privileges(cx, node)
  local result = terralib.newlist()
  for _, privilege in ipairs(node) do
    result:insert(type_check.privilege(cx, privilege))
  end
  return result
end

function type_check.coherence_kind(cx, node)
  if node:is(ast.coherence_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.coherence_kinds(cx, node)
  return node:map(
    function(coherence) return type_check.coherence_kind(cx, coherence) end)
end

local function check_coherence_conflict_field(node, region, field,
                                              coherence, other_field, result)
  local region_type = region:gettype()
  if field:starts_with(other_field) or other_field:starts_with(field) then
    local other_coherence = result[region_type][other_field]
    assert(other_coherence)
    if other_coherence ~= coherence then
      report.error(
        node, "conflicting coherence modes: " .. tostring(other_coherence) .. "(" ..
          (data.newtuple(region) .. other_field):mkstring(".") .. ")" ..
          " and " .. tostring(coherence) .. "(" ..
          (data.newtuple(region) .. field):mkstring(".") .. ")")
    end
  end
end

local function check_coherence_conflict(node, region, field, coherence, result)
  local region_type = region:gettype()
  for _, other_field in result[region_type]:keys() do
    check_coherence_conflict_field(
      node, region, field, coherence, other_field, result)
  end
end

local function coherence_string(coherence, region_field)
  return tostring(coherence) .. "(" .. tostring(region_field) .. ")"
end

function type_check.coherence(cx, node, param_type_set, result)
  local coherence_modes = type_check.coherence_kinds(cx, node.coherence_modes)
  local region_fields = type_check.regions(cx, node.regions)

  for _, coherence in ipairs(coherence_modes) do
    for _, region_field in ipairs(region_fields) do
      local region = region_field.region
      local region_type = region:gettype()
      assert(std.type_supports_privileges(region_type))
      if not param_type_set[region_type] then
        report.error(node, "requested " .. coherence_string(coherence, region_field) ..
          " but " .. tostring(region) .. " is not a parameter")
      end

      local fields = region_field.fields
      for _, field in ipairs(fields) do
        check_coherence_conflict(node, region, field, coherence, result)
        result[region_type][field] = coherence
      end
    end
  end
end

function type_check.coherence_modes(cx, node, param_type_set)
  local result = data.new_recursive_map(1)
  for _, coherence in ipairs(node) do
    type_check.coherence(cx, coherence, param_type_set, result)
  end
  return result
end

function type_check.flag_kind(cx, node)
  if node:is(ast.flag_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.flag_kinds(cx, node)
  return node:map(function(flag) return type_check.flag_kind(cx, flag) end)
end

function type_check.flag(cx, node, result)
  local flags = type_check.flag_kinds(cx, node.flags)
  local region_fields = type_check.regions(cx, node.regions)

  for _, flag in ipairs(flags) do
    for _, region_field in ipairs(region_fields) do
      local region = region_field.region
      local region_type = region:gettype()
      assert(std.type_supports_privileges(region_type))

      local fields = region_field.fields
      for _, field in ipairs(fields) do
        result[region_type][field][flag] = true
      end
    end
  end
end

function type_check.flags(cx, node)
  local result = data.new_recursive_map(2)
  for _, flag in ipairs(node) do
    type_check.flag(cx, flag, result)
  end
  return result
end

function type_check.condition_kind(cx, node)
  if node:is(ast.condition_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.condition_kinds(cx, node)
  return node:map(
    function(condition) return type_check.condition_kind(cx, condition) end)
end

function type_check.condition(cx, node, params, result)
  local conditions = type_check.condition_kinds(cx, node.conditions)
  local variables = type_check.condition_variables(cx, node.variables)

  for _, symbol in ipairs(variables) do
    for _, condition in ipairs(conditions) do
      local i = params[symbol]
      assert(i)
      result[condition][i] = symbol
    end
  end
end

function type_check.expr_condition(cx, node)
  local conditions = type_check.condition_kinds(cx, node.conditions)
  local values = node.values:map(
    function(value) return type_check.expr(cx, value) end)
  local value_types = values:map(
    function(value) return std.check_read(cx, value) end)
  for _, value_type in ipairs(value_types) do
    if not (std.is_phase_barrier(value_type) or
            std.is_list_of_phase_barriers(value_type)) then
      report.error(node, "type mismatch: expected " ..
                  tostring(std.phase_barrier) .. " but got " ..
                  tostring(value_type))
    end
  end

  return values:map(
    function(value)
      return ast.typed.expr.Condition {
        conditions = conditions,
        value = value,
        expr_type = std.as_read(value.expr_type),
        annotations = node.annotations,
        span = node.span,
      }
    end)
end

function type_check.conditions(cx, node, params)
  local param_index_by_symbol = {}
  for i, param in ipairs(params) do
    param_index_by_symbol[param.symbol] = i
  end

  local result = data.newmap()
  result[std.arrives] = data.newmap()
  result[std.awaits] = data.newmap()

  node:map(
    function(condition)
      return type_check.condition(cx, condition, param_index_by_symbol, result)
    end)
  return result
end

function type_check.constraint_kind(cx, node)
  if node:is(ast.constraint_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.constraint(cx, node)
  local lhs = type_check.region_bare(cx, node.lhs)
  local op = type_check.constraint_kind(cx, node.op)
  local rhs = type_check.region_bare(cx, node.rhs)
  return std.constraint(lhs, rhs, op)
end

function type_check.constraints(cx, node)
  return node:map(
    function(constraint) return type_check.constraint(cx, constraint) end)
end

function type_check.expr_id(cx, node)
  local expr_type = cx.type_env:lookup(node, node.value)

  return ast.typed.expr.ID {
    value = node.value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_constant(cx, node)
  return ast.typed.expr.Constant {
    value = node.value,
    expr_type = node.expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_global(cx, node)
  return ast.typed.expr.Global {
    value = node.value,
    expr_type = node.expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

local untyped = std.untyped
local untyped_fn = terralib.types.functype(terralib.newlist({untyped}), terralib.types.unit, true)
local function cast_fn(to_type)
  return terralib.types.functype(terralib.newlist({untyped}), to_type, false)
end

local function insert_implicit_cast(node, from_type, to_type)
  assert(std.validate_implicit_cast(from_type, to_type))
  if not std.type_eq(from_type, to_type) then
    -- It is safe to make this an explicit cast, because every valid
    -- implicit cast should also be a valid explicit cast.
    -- (The inverse is not true.)
    return ast.typed.expr.Cast {
      fn = ast.typed.expr.Function {
        value = to_type,
        span = node.span,
        annotations = node.annotations,
        expr_type = cast_fn(to_type),
      },
      arg = node,
      span = node.span,
      annotations = node.annotations,
      expr_type = to_type,
    }
  else
    return node
  end
end

function type_check.expr_function(cx, node)
  -- Functions are type checked at the call site.
  return ast.typed.expr.Function {
    value = node.value,
    expr_type = untyped,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_field_access(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = value.expr_type -- Keep references, do NOT std.check_read

  -- Resolve automatic dereferences by desugaring into a separate
  -- deref and field access.
  if std.is_bounded_type(std.as_read(value_type)) and
    std.as_read(value_type):is_ptr() and
    -- Note: Bounded types with fields take precedence over dereferences.
    not std.get_field(std.as_read(value_type).index_type.base_type, node.field_name)
  then
    return type_check.expr(
      cx,
      node {
        value = ast.specialized.expr.Deref {
          value = node.value,
          span = node.value.span,
          annotations = node.value.annotations,
        }
      })
  elseif std.as_read(value_type):ispointer() then
    return type_check.expr(
      cx,
      node {
        value = ast.specialized.expr.Deref {
          value = node.value,
          span = node.value.span,
          annotations = node.value.annotations,
        }
      })
  end

  local unpack_type = value_type
  local constraints

  -- Resolve index and bounded types and automatic unpacks of fspaces.
  do
    local result_type, result_constraints
    if std.is_index_type(std.as_read(unpack_type)) then
      result_type, result_constraints = std.as_read(unpack_type).base_type
    elseif std.is_bounded_type(std.as_read(unpack_type)) and
      std.get_field(std.as_read(unpack_type).index_type.base_type, node.field_name)
    then
      result_type, result_constraints = std.as_read(unpack_type).index_type.base_type
    elseif std.is_fspace_instance(std.as_read(unpack_type)) then
      result_type, result_constraints = std.unpack_fields(std.as_read(unpack_type))
    end

    -- Since we may have stripped off a reference from the incoming
    -- type, restore it before continuing.
    if not std.type_eq(std.as_read(unpack_type), result_type) then
      constraints = result_constraints
      if (std.is_index_type(std.as_read(unpack_type)) or
          std.is_bounded_type(std.as_read(unpack_type)))
      then
        unpack_type = std.rawref(&result_type)
      elseif std.is_ref(unpack_type) then
        unpack_type = std.ref(unpack_type.pointer_type.index_type(result_type, unpack(unpack_type.bounds_symbols)),
                              unpack(unpack_type.field_path))
      elseif std.is_rawref(unpack_type) then
        unpack_type = std.rawref(&result_type)
      else
        unpack_type = result_type
      end
    end
  end

  if constraints then
    std.add_constraints(cx, constraints)
  end

  local field_type
  if std.is_region(std.as_read(unpack_type)) and node.field_name == "ispace" then
    field_type = std.as_read(unpack_type):ispace()
  elseif std.is_ispace(std.as_read(unpack_type)) and node.field_name == "bounds" then
    local index_type = std.as_read(unpack_type).index_type
    if index_type:is_opaque() then
      report.error(node, "no field '" .. node.field_name .. "' in type " ..
                  tostring(std.as_read(unpack_type)))
    end
    field_type = std.rect_type(index_type)
  elseif std.is_ispace(std.as_read(unpack_type)) and node.field_name == "volume" then
    -- Volume can be retrieved on any ispace.
    field_type = int64
  elseif std.is_region(std.as_read(unpack_type)) and (node.field_name == "bounds" or node.field_name == "volume") then
    -- Index space fields can also be retrieved through a region.
    return type_check.expr(
      cx,
      node {
        value = ast.specialized.expr.FieldAccess {
          value = node.value,
          field_name = "ispace",
          span = node.value.span,
          annotations = node.value.annotations,
        }
      })
  elseif std.is_partition(std.as_read(unpack_type)) and node.field_name == "colors" then
    field_type = std.as_read(unpack_type):colors()
  elseif std.is_cross_product(std.as_read(unpack_type)) and node.field_name == "colors" then
    field_type = std.as_read(unpack_type):partition():colors()
  elseif std.type_is_opaque_to_field_accesses(std.as_read(unpack_type)) then
    local hint = ""
    if std.is_region(std.as_read(unpack_type)) and
       std.get_field(std.as_read(unpack_type):fspace(), node.field_name)
    then
      hint = ". If you wanted to project the region, please wrap the field name with braces (i.e., " ..
        string.gsub((pretty.entry_expr(value) .. ".{" .. node.field_name .. "}"), "[$]", "") .. ")."
    end
    report.error(node, "no field '" .. node.field_name .. "' in type " ..
                tostring(std.as_read(value_type)) .. hint)
  else
    field_type = std.get_field(unpack_type, node.field_name)
    if not field_type then
      report.error(node, "no field '" .. node.field_name .. "' in type " ..
                  tostring(std.as_read(value_type)))
    end
  end

  return ast.typed.expr.FieldAccess {
    value = value,
    field_name = node.field_name,
    expr_type = field_type,
    annotations = node.annotations,
    span = node.span,
  }
end

local function add_analyzable_disjointness_constraints(cx, partition, subregion)
  local index = subregion:get_index_expr()
  local other_subregions = partition:subregions_constant()
  for _, other_subregion in other_subregions:items() do
    local other_index = other_subregion:get_index_expr()
    if affine_helper.analyze_index_noninterference(index, other_index) then
      std.add_constraint(cx, subregion, other_subregion, std.disjointness, true)
    end
  end
end

function type_check.expr_index_access(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)
  local index = type_check.expr(cx, node.index)
  local index_type = std.check_read(cx, index)

  -- Some kinds of operations require information about the index used
  -- (e.g. partition access at a constant). Save that index now to
  -- avoid getting entangled in any implicit casts.
  local analyzable = affine_helper.is_analyzable_index_expression(index)

  if std.is_partition(value_type) then
    local color_type = value_type:colors().index_type
    if not std.validate_implicit_cast(index_type, color_type) then
      report.error(node, "type mismatch: expected " .. tostring(color_type) .. " but got " .. tostring(index_type))
    end
    index = insert_implicit_cast(index, index_type, color_type)

    local partition = value_type:partition()
    local parent = value_type:parent_region()

    local subregion
    if analyzable then
      subregion = value_type:subregion_constant(index)

      if value_type:is_disjoint() then
        add_analyzable_disjointness_constraints(cx, value_type, subregion)
      end
    else
      subregion = value_type:subregion_dynamic()
    end

    std.add_constraint(cx, partition, parent, std.subregion, false)
    std.add_constraint(cx, subregion, partition, std.subregion, false)

    return ast.typed.expr.IndexAccess {
      value = value,
      index = index,
      expr_type = subregion,
      annotations = node.annotations,
      span = node.span,
    }
  elseif std.is_cross_product(value_type) then
    local color_type = value_type:partition():colors().index_type
    if not std.validate_implicit_cast(index_type, color_type) then
      report.error(node, "type mismatch: expected " .. tostring(color_type) .. " but got " .. tostring(index_type))
    end
    index = insert_implicit_cast(index, index_type, color_type)

    local partition = value_type:partition()
    local parent = value_type:parent_region()
    local subregion, subpartition
    if analyzable then
      subpartition = value_type:subpartition_constant(index)
      subregion = subpartition:parent_region()

      if value_type:is_disjoint() then
        add_analyzable_disjointness_constraints(cx, value_type, subregion)
      end
    else
      subpartition = value_type:subpartition_dynamic()
      subregion = subpartition:parent_region()
    end

    std.add_constraint(cx, partition, parent, std.subregion, false)
    std.add_constraint(cx, subregion, partition, std.subregion, false)
    std.add_constraint(cx, subpartition:partition(), subregion, std.subregion, false)

    return ast.typed.expr.IndexAccess {
      value = value,
      index = index,
      expr_type = subpartition,
      annotations = node.annotations,
      span = node.span,
    }
  elseif std.is_region(value_type) then
    -- FIXME: Need to check if this is a bounded type (with the right
    -- bound) and, if not, insert a dynamic cast.

    -- Elliott: Careful! A bounded type generally indicates that the
    -- value in question is within bounds. This is not necessarily
    -- true for array accesses. If we want to be able to do analyses
    -- with this information later, we should be careful not to cast
    -- to a bounded type here.
    local region_index_type = value_type:ispace().index_type
    if not std.validate_implicit_cast(index_type, region_index_type) then
      report.error(node, "type mismatch: expected " .. tostring(region_index_type) .. " but got " .. tostring(index_type))
    end

    local region_symbol
    if value:is(ast.typed.expr.ID) then
      region_symbol = value.value
    else
      region_symbol = std.newsymbol(value_type)
    end
    local result_type = std.ref(region_index_type(value_type:fspace(), region_symbol))

    return ast.typed.expr.IndexAccess {
      value = value,
      index = index,
      expr_type = result_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif std.is_list(value_type) then
    local slice = std.type_eq(index_type, std.list(int))
    if not slice then
      if not std.validate_implicit_cast(index_type, int) then
        report.error(node, "type mismatch: expected " .. tostring(int) .. " or " ..
                    tostring(std.list(int)) .. " but got " ..
                    tostring(index_type))
      end
      index = insert_implicit_cast(index, index_type, int)
    end

    if not value_type:is_list_of_regions() then
      local expr_type = value_type:leaf_element_type()
      local start = 2
      if slice then
        start = 1
      end
      for i = start, value_type:list_depth() do
        expr_type = std.list(expr_type)
      end
      return ast.typed.expr.IndexAccess {
        value = value,
        index = index,
        expr_type = expr_type,
        annotations = node.annotations,
        span = node.span,
      }
    else
      local expr_type
      if slice then
        expr_type = value_type:slice()
      else
        expr_type = value_type:slice(1)
      end
      std.add_constraint(cx, expr_type, value_type, std.subregion, false)

      return ast.typed.expr.IndexAccess {
        value = value,
        index = index,
        expr_type = expr_type,
        annotations = node.annotations,
        span = node.span,
      }
    end
  elseif std.is_transform_type(value_type) then
    local expected = std.int2d
    if not std.validate_implicit_cast(index_type, expected) then
      report.error(node, "type mismatch: expected " .. tostring(expected) .. " but got " .. tostring(index_type))
    end
    index = insert_implicit_cast(index, index_type, expected)
    return ast.typed.expr.IndexAccess {
      value = value,
      index = index,
      expr_type = std.rawref(&int64),
      annotations = node.annotations,
      span = node.span,
    }
  else
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local test
    if std.is_regent_array(value_type) then
      test = function()
        local terra query(a : value_type, i : index_type)
          return a.impl[i]
        end
        return query:gettype().returntype
      end
    else
      test = function()
        local terra query(a : value_type, i : index_type)
          return a[i]
        end
        return query:gettype().returntype
      end
    end
    local valid, result_type = pcall(test)

    if not valid then
      report.error(node, "invalid index access for " .. tostring(value_type) .. " and " .. tostring(index_type))
    end

    -- Hack: Fix up the type to be a reference if the original was.
    if std.is_ref(value.expr_type) then
      local value_type = value.expr_type
      local index = affine_helper.is_constant_expr(index) and
                    affine_helper.convert_constant_expr(index) or false
      result_type = std.ref(value_type.pointer_type.index_type(value_type.refers_to_type, unpack(value_type.bounds_symbols)),
                            unpack(value_type.field_path .. data.newtuple(index)))
    elseif std.is_rawref(value.expr_type) then
      result_type = std.rawref(&result_type)
    end

    return ast.typed.expr.IndexAccess {
      value = value,
      index = index,
      expr_type = result_type,
      annotations = node.annotations,
      span = node.span,
    }
  end
end

local function get_function_definitions(fn)
  if terralib.isfunction(fn) or
     terralib.isoverloadedfunction(fn) then
    return terralib.newlist({rawget(fn, "definition")}) or
           rawget(fn, "definitions")
  else
    return terralib.newlist()
  end
end

function type_check.expr_method_call(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  local args = node.args:map(
    function(arg) return type_check.expr(cx, arg) end)
  local arg_types = args:map(
    function(arg) return std.check_read(cx, arg) end)

  local arg_symbols = arg_types:map(
    function(arg_type) return terralib.newsymbol(arg_type) end)

  local function test()
    local terra query(self : value_type, [arg_symbols])
      return [self]:[node.method_name]([arg_symbols])
    end
    return query:gettype().returntype
  end
  local valid, expr_type = pcall(test)

  if not valid then
    report.error(node, "invalid method call for " .. tostring(value_type) .. ":" ..
                node.method_name .. "(" .. data.newtuple(unpack(arg_types)):mkstring(", ") .. ")")
  end

  local defs = get_function_definitions(value_type.methods[node.method_name])
  if #defs == 1 then
    local args_with_casts = terralib.newlist()
    assert(not defs[1].type.isvararg)
    local param_types = defs[1].type.parameters
    for idx, arg_type in ipairs(arg_types) do
      args_with_casts:insert(
        insert_implicit_cast(args[idx], arg_type, param_types[idx + 1]))
    end
    args = args_with_casts
  end

  return ast.typed.expr.MethodCall {
    value = value,
    method_name = node.method_name,
    args = args,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_call(cx, node)
  local fn = type_check.expr(cx, node.fn)
  local args = node.args:map(
    function(arg) return type_check.expr(cx, arg) end)
  local arg_types = args:map(
    function(arg) return std.check_read(cx, arg) end)
  local conditions = terralib.newlist()
  for _, condition in ipairs(node.conditions) do
    conditions:insertall(type_check.expr_condition(cx, condition))
  end

  -- For macros, run macro expansion and type check the result.
  if std.is_macro(fn.value) then
    local quotes = data.mapi(
      function(i, arg)
        return std.newrquote(
          ast.specialized.top.QuoteExpr {
            expr = node.args[i],
            expr_type = arg_types[i],
            annotations = node.annotations,
            span = node.span,
          })
      end,
      args)
    local result = fn.value.fn(unpack(quotes))
    if not (std.is_rquote(result) and ast.is_node(result:getast()) and
              result:getast():is(ast.specialized.top.QuoteExpr))
    then
      report.error(node, "macro was expected to return an rexpr, but got " .. tostring(result))
    end
    return type_check.expr(cx, result:getast().expr)
  end

  -- Determine the type of the function being called.
  local fn_type
  local def_type
  if fn.expr_type == untyped then
    if terralib.isfunction(fn.value) or
      terralib.isoverloadedfunction(fn.value) or
      terralib.ismacro(fn.value) or
      type(fn.value) == "cdata"
    then
      -- Ask the Terra compiler to determine which overloaded function
      -- to call (or for macros, determine the type of the resulting
      -- expression).
      local arg_symbols = arg_types:map(
        function(arg_type) return terralib.newsymbol(arg_type) end)

      local function test()
        local terra query([arg_symbols])
          return [fn.value]([arg_symbols])
        end
        return query:gettype()
      end
        local valid, result_type = pcall(test)

      if valid then
        local defs = get_function_definitions(fn.value)
        if #defs == 1 then
          def_type = defs[1].type
        end
        fn_type = result_type
      else
        local fn_name = fn.value.name or tostring(fn.value)
        fn_name = string.gsub(fn_name, "^std[.]", "regentlib.")
        report.error(node, "no applicable overloaded function " .. tostring(fn_name) ..
                  " for arguments " .. data.newtuple(unpack(arg_types)):mkstring(", "))
      end
    elseif std.is_task(fn.value) then
      fn_type = fn.value:get_type()
    elseif std.is_math_fn(fn.value) then
      fn_type = fn.value:get_definition().type
    elseif type(fn.value) == "function" then
      fn_type = untyped_fn
    else
      error("unreachable")
    end
  else
    fn_type = fn.expr_type
  end
  def_type = def_type or fn_type
  assert(terralib.types.istype(fn_type) and
           (fn_type:isfunction() or fn_type:ispointertofunction()))
  -- Store the determined type back into the AST node for the function.
  fn.expr_type = def_type

  local param_symbols
  if std.is_task(fn.value) then
    param_symbols = fn.value:get_param_symbols()
  else
    param_symbols = std.fn_param_symbols(def_type)
  end
  local arg_symbols = terralib.newlist()
  for i, arg in ipairs(args) do
    local arg_type = arg_types[i]
    if arg:is(ast.typed.expr.ID) then
      arg_symbols:insert(arg.value)
    else
      arg_symbols:insert(std.newsymbol(arg_type))
    end
  end
  local expr_type, need_cast = std.validate_args(
    node, param_symbols, arg_symbols, def_type.isvararg, def_type.returntype, {}, false, true)

  if std.is_task(fn.value) then
    if fn.value.is_local then
      if not fn.value:has_primary_variant() then
        report.error(node, "cannot call a local task that does not have a variant defined")
      end
      local variant_ast = fn.value:get_primary_variant():get_ast()
      local variant_is_cuda = variant_ast.annotations.cuda:is(ast.annotation.Demand)
      if cx.is_cuda and not variant_is_cuda then
        report.error(node, "calling a local task without a CUDA variant from a CUDA variant")
      end
    end

    if cx.must_epoch then
      -- Inside a must epoch tasks are not allowed to return.
      expr_type = terralib.types.unit
    end

    local mapping = {}
    for i, arg_symbol in ipairs(arg_symbols) do
      local param_symbol = param_symbols[i]
      local param_type = fn_type.parameters[i]
      mapping[param_symbol] = arg_symbol
      mapping[param_type] = arg_symbol
    end

    local privileges = fn.value:get_privileges()
    for _, privilege_list in ipairs(privileges) do
      for _, privilege in ipairs(privilege_list) do
        local privilege_type = privilege.privilege
        local region = privilege.region
        local field_path = privilege.field_path
        assert(std.type_supports_privileges(region:gettype()))
        local arg_region = mapping[region:gettype()]

        local fspace = region:gettype():fspace()
        -- If the field space of the parameter's region type is not a field space instance,
        -- we individually check privileges for fields because they can be renamed via
        -- field polymorphism.
        if not std.is_fspace_instance(fspace) then
          local field_path_mapping = data.dict(data.zip(
                  std.flatten_struct_fields(fspace),
                  std.flatten_struct_fields(arg_region:gettype():fspace())))

          std.get_absolute_field_paths(fspace, field_path):map(function(field_path)
            local arg_field_path = field_path_mapping[field_path]
            if not std.check_privilege(cx, privilege_type, arg_region:gettype(), arg_field_path) then
              for i, arg in ipairs(arg_symbols) do
                if std.type_eq(arg:gettype(), arg_region:gettype()) then
                  report.error(
                    node, "invalid privileges in argument " .. tostring(i) ..
                      ": " .. tostring(privilege_type) .. "(" ..  pretty.entry_expr(args[i]) ..
                      (((#arg_field_path > 0) and ("." .. arg_field_path:mkstring("."))) or "") ..  ")")
                end
              end
              assert(false)
            end
          end)

        -- Otherwise, we perform nominal type checking
        else
          if not std.check_privilege(cx, privilege_type, arg_region:gettype(), field_path) then
            for i, arg in ipairs(arg_symbols) do
              if std.type_eq(arg:gettype(), arg_region:gettype()) then
                report.error(
                  node, "invalid privileges in argument " .. tostring(i) ..
                    ": " .. tostring(privilege_type) .. "(" ..
                    (data.newtuple(arg_region) .. field_path):mkstring(".") ..
                    ")")
              end
            end
            assert(false)
          end
        end
      end
    end

    local constraints = fn.value:get_param_constraints()
    local satisfied, constraint = std.check_constraints(cx, constraints, mapping)
    if not satisfied then
      report.error(node, "invalid call missing constraint " .. tostring(constraint.lhs) ..
                  " " .. tostring(constraint.op) .. " " .. tostring(constraint.rhs))
    end
  end

  local param_types = terralib.newlist()
  param_types:insertall(def_type.parameters)
  if def_type.isvararg then
    for idx = #def_type.parameters + 1, #arg_types do
      param_types:insert(arg_types[idx])
      need_cast:insert(false)
    end
    -- Hack: set this back to the concrete type inferred by query above.
    --       since RDIR doesn't understand functions with varargs
    fn.expr_type = fn_type
  end
  args =
    data.zip(args, arg_types, param_types, need_cast):map(function(tuple)
      local arg, arg_type, param_type, need_cast = unpack(tuple)
      if not need_cast then
        return arg
      else
        return insert_implicit_cast(arg, arg_type, param_type)
      end
    end)

  local result = ast.typed.expr.Call {
    fn = fn,
    args = args,
    conditions = conditions,
    predicate = false,
    predicate_else_value = false,
    replicable = false,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
  if expr_type == untyped then
    cx.fixup_nodes:insert(result)
  end
  return result
end

function type_check.expr_cast(cx, node)
  local fn = type_check.expr(cx, node.fn)

  local to_type = fn.value
  assert(terralib.types.istype(to_type))
  fn.expr_type = cast_fn(to_type)

  if #node.args ~= 1 then
    report.error(node, "expected 1 arguments but got " .. tostring(#node.args))
  end
  local arg = type_check.expr(cx, node.args[1])
  local from_type = std.check_read(cx, arg)

  if std.is_fspace_instance(to_type) then
    if not (from_type:isstruct() or std.is_fspace_instance(from_type)) then
      report.error(node, "type mismatch: expected struct or fspace but got " .. tostring(from_type))
    end

    local to_params = to_type:getparams()
    local to_args = to_type.args
    local to_constraints = to_type:getconstraints()

    local to_fields = std.struct_entries_symbols(to_type)

    local from_symbols = {}
    if arg:is(ast.typed.expr.Ctor) and arg.named then
      for _, field in ipairs(arg.fields) do
        if field.value:is(ast.typed.expr.ID) and
          std.is_symbol(field.value.value) and
          field.value.value:hasname() and
          field.value.value:hastype()
        then
          from_symbols[field.value.value:gettype()] = field.value.value
        end
      end
    end
    local from_fields = std.struct_entries_symbols(from_type, from_symbols)

    local mapping = {}
    for i, param in ipairs(to_params) do
      local arg = to_args[i]
      mapping[param] = arg
      mapping[param:gettype()] = arg:gettype()
    end

    std.validate_args(node, to_fields, from_fields, false, terralib.types.unit, mapping, false)
    local satisfied, constraint = std.check_constraints(cx, to_constraints, mapping)
    if not satisfied then
      report.error(node, "invalid cast missing constraint " .. tostring(constraint.lhs) ..
                  " " .. tostring(constraint.op) .. " " .. tostring(constraint.rhs))
    end
  else
    if not std.validate_explicit_cast(from_type, to_type) then
      report.error(node, "invalid cast from " .. tostring(from_type) .. " to " .. tostring(to_type))
    end
  end

  if std.type_eq(from_type, to_type) then
    return arg
  else
    return ast.typed.expr.Cast {
      fn = fn,
      arg = arg,
      expr_type = to_type,
      annotations = node.annotations,
      span = node.span,
    }
  end
end

function type_check.expr_ctor_list_field(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  return ast.typed.expr.CtorListField {
    value = value,
    expr_type = value_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_ctor_rec_field(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  return ast.typed.expr.CtorRecField {
    name = node.name,
    value = value,
    expr_type = value_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_ctor_field(cx, node)
  if node:is(ast.specialized.expr.CtorListField) then
    return type_check.expr_ctor_list_field(cx, node)
  elseif node:is(ast.specialized.expr.CtorRecField) then
    return type_check.expr_ctor_rec_field(cx, node)
  else
    assert(false)
  end
end

function type_check.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return type_check.expr_ctor_field(cx, field) end)

  local expr_type
  if node.named then
    expr_type = std.ctor_named(
      fields:map(
        function(field) return { field.name, field.expr_type } end))
  else
    expr_type = std.ctor_tuple(fields:map(
      function(field) return field.expr_type end))
  end

  return ast.typed.expr.Ctor {
    fields = fields,
    named = node.named,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_context(cx, node)
  return ast.typed.expr.RawContext {
    expr_type = std.c.legion_context_t,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_fields(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region.region)

  local absolute_field_paths =
    std.get_absolute_field_paths(region_type:fspace(), region.fields)
  local privilege_fields = terralib.newlist()
  for _, field_path in ipairs(absolute_field_paths) do
    if not std.check_any_privilege(cx, region_type, field_path) then
      report.error(node, "invalid privilege: task has no privilege on " ..
          string.gsub((pretty.entry_expr(region.region) .. "." ..
              field_path:mkstring(".")), "[$]", ""))
    end
    privilege_fields:insert(field_path)
  end
  local fields_type = std.c.legion_field_id_t[#privilege_fields]

  return ast.typed.expr.RawFields {
    region = region,
    fields = privilege_fields,
    expr_type = fields_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_future(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)

  if value_type ~= std.c.legion_future_t then
    report.error(node, "type mismatch in argument 2: expected legion_future_t but got " .. tostring(value_type))
  end

  return ast.typed.expr.RawFuture {
    value = value,
    expr_type = node.value_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_physical(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region.region)

  local absolute_field_paths =
    std.get_absolute_field_paths(region_type:fspace(), region.fields)
  local privilege_fields = terralib.newlist()
  for _, field_path in ipairs(absolute_field_paths) do
    if not std.check_any_privilege(cx, region_type, field_path) then
      report.error(node, "invalid privilege: task has no privilege on " ..
          string.gsub((pretty.entry_expr(region.region) .. "." ..
              field_path:mkstring(".")), "[$]", ""))
    end
    privilege_fields:insert(field_path)
  end
  local physical_type = std.c.legion_physical_region_t[#privilege_fields]

  return ast.typed.expr.RawPhysical {
    region = region,
    fields = privilege_fields,
    expr_type = physical_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_runtime(cx, node)
  return ast.typed.expr.RawRuntime {
    expr_type = std.c.legion_runtime_t,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_task(cx, node)
  return ast.typed.expr.RawTask {
    expr_type = std.c.legion_task_t,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_raw_value(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  local expr_type
  if std.is_ispace(value_type) then
    expr_type = std.c.legion_index_space_t
  elseif std.is_region(value_type) then
    expr_type = std.c.legion_logical_region_t
  elseif std.is_partition(value_type) then
    expr_type = std.c.legion_logical_partition_t
  elseif std.is_cross_product(value_type) then
    expr_type = std.c.legion_terra_index_cross_product_t
  elseif std.is_bounded_type(value_type) then
    expr_type = value_type.index_type.impl_type
  else
    report.error(node, "raw expected an ispace, region, partition, or cross product, got " .. tostring(value_type))
  end

  return ast.typed.expr.RawValue {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_isnull(cx, node)
  local pointer = type_check.expr(cx, node.pointer)
  local pointer_type = std.check_read(cx, pointer)
  if not (std.is_bounded_type(pointer_type) or pointer_type:ispointer() or pointer_type == niltype) then
    report.error(node, "isnull requires bounded type, got " .. tostring(pointer_type))
  end
  return ast.typed.expr.Isnull {
    pointer = pointer,
    expr_type = bool,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_new(cx, node)
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)
  local extent = node.extent and type_check.expr(cx, node.extent)
  local extent_type = extent and std.check_read(cx, extent)

  -- Pointer and region types checked in specialize.

  if std.is_region(region_type) and
     not std.type_eq(region_type:fspace(), node.pointer_type.points_to_type) then
    report.error(node, "type mismatch in argument 1: expected " .. tostring(region_type:fspace()) ..
                       ", got " .. tostring(node.pointer_type.points_to_type))
  end

  local index_type = node.pointer_type.index_type
  if extent and not std.validate_implicit_cast(extent_type, index_type) then
    report.error(node, "type mismatch in argument 2: expected " .. tostring(index_type) .. ", got " .. tostring(extent_type))
  end
  if extent then
    extent = insert_implicit_cast(extent, extent_type, index_type)
  end

  report.error(node, "operator new has been removed, instead all regions are allocated by default")
  assert(false, "unreachable")
end

function type_check.expr_null(cx, node)
  local pointer_type = node.pointer_type
  if not (std.is_bounded_type(pointer_type) or pointer_type:ispointer() or pointer_type == niltype) then
    report.error(node, "null requires bounded type, got " .. tostring(pointer_type))
  end
  return ast.typed.expr.Null {
    pointer_type = pointer_type,
    expr_type = pointer_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_dynamic_cast(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if std.is_bounded_type(node.expr_type) then
    if not std.validate_implicit_cast(value_type, node.expr_type.index_type) then
      report.error(node, "type mismatch in dynamic_cast: expected " .. tostring(node.expr_type.index_type) .. ", got " .. tostring(value_type))
    end

    if not std.type_eq(node.expr_type.points_to_type, value_type.points_to_type) then
      report.error(node, "type mismatch in dynamic_cast: expected a pointer to " .. tostring(node.expr_type.points_to_type) .. ", got " .. tostring(value_type.points_to_type))
    end

    std.check_bounds(node, node.expr_type)

  elseif std.is_partition(node.expr_type) then
    if not std.is_partition(value_type) then
      report.error(node, "type mismatch in dynamic_cast: expected a partition, got " .. tostring(value_type))
    end

    if not (std.type_eq(node.expr_type:colors(), value_type:colors()) and
            node.expr_type.parent_region_symbol == value_type.parent_region_symbol) then
      report.error(node, "incompatible partitions for dynamic_cast: " .. tostring(node.expr_type) .. " and " .. tostring(value_type))
    end
  else
    report.error(node, "dynamic_cast requires ptr type or partition type as argument 1, got " .. tostring(node.expr_type))
  end

  return ast.typed.expr.DynamicCast {
    value = value,
    expr_type = node.expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_static_cast(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  local expr_type = node.expr_type

  if std.is_partition(expr_type) then
    if not std.is_partition(value_type) then
      report.error(node, "static_cast requires partition type as argument 2, got " .. tostring(value_type))
    end
    if expr_type.disjointness ~= value_type.disjointness then
      report.error(node, tostring(value_type.disjointness) .. " partitions cannot be casted to " ..
          tostring(expr_type.disjointness) .. " partitions")
    end
    if not std.type_eq(expr_type:colors(), value_type:colors()) then
      report.error(node, "type mismatch in argument 2: expected " .. tostring(expr_type:colors()) ..
            " for color space but got " .. tostring(value_type:colors()))
    end
    local value_region_symbol = value_type.parent_region_symbol
    local expr_region_symbol = expr_type.parent_region_symbol
    local constraint = std.constraint(value_region_symbol, expr_region_symbol, std.subregion)
    if not std.check_constraint(cx, constraint) then
      report.error(node,
          "the region " .. tostring(value_region_symbol) .. " is not a subregion of " .. tostring(expr_region_symbol))
    end
    local parent_region_map = {}
    return ast.typed.expr.StaticCast {
      value = value,
      parent_region_map = parent_region_map,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif not std.is_bounded_type(expr_type) then
    report.error(node, "static_cast requires partition or ptr type as argument 1, got " .. tostring(expr_type))
  end
  if not std.is_bounded_type(value_type) then
    report.error(node, "static_cast requires ptr as argument 2, got " .. tostring(value_type))
  end
  if not std.type_eq(expr_type.points_to_type, value_type.points_to_type) then
    report.error(node, "incompatible pointers for static_cast: " .. tostring(expr_type) .. " and " .. tostring(value_type))
  end

  local parent_region_map = {}
  for i, value_region_symbol in ipairs(value_type.bounds_symbols) do
    for j, expr_region_symbol in ipairs(expr_type.bounds_symbols) do
      local constraint = std.constraint(
        value_region_symbol,
        expr_region_symbol,
        std.subregion)
      if std.check_constraint(cx, constraint) then
        parent_region_map[i] = j
        break
      end
    end
  end

  return ast.typed.expr.StaticCast {
    value = value,
    parent_region_map = parent_region_map,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_unsafe_cast(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  local expr_type = node.expr_type

  if not std.is_bounded_type(expr_type) then
    report.error(node, "unsafe_cast requires ptr type as argument 1, got " .. tostring(expr_type))
  end
  if #std.check_bounds(node, expr_type) ~= 1 then
    report.error(node, "unsafe_cast requires single ptr type as argument 1, got " .. tostring(expr_type))
  end
  if not std.validate_implicit_cast(value_type, node.expr_type.index_type) then
    report.error(node, "unsafe_cast requires ptr as argument 2, got " .. tostring(value_type))
  end
  if std.is_bounded_type(value_type) and not std.type_eq(node.expr_type.points_to_type, value_type.points_to_type) then
    report.error(node, "incompatible pointers for unsafe_cast: " .. tostring(expr_type) .. " and " .. tostring(value_type))
  end

  return ast.typed.expr.UnsafeCast {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_ispace(cx, node)
  local index_type = node.index_type
  local extent = type_check.expr(cx, node.extent)
  local extent_type = std.check_read(cx, extent)
  local start = node.start and type_check.expr(cx, node.start)
  local start_type = node.start and std.check_read(cx, start)

  if not std.is_index_type(index_type) then
    report.error(node, "type mismatch in argument 1: expected an index type but got " .. tostring(index_type))
  end
  if not std.validate_implicit_cast(extent_type, index_type) then
    report.error(node, "type mismatch in argument 2: expected " ..
                tostring(index_type) .. " but got " .. tostring(extent_type))
  end
  if start_type and not std.validate_implicit_cast(start_type, index_type) then
    report.error(node, "type mismatch in argument 3: expected " ..
                tostring(index_type) .. " but got " .. tostring(start_type))
  end
  extent = insert_implicit_cast(extent, extent_type, index_type)
  if node.start then
    start = insert_implicit_cast(start, start_type, index_type)
  end

  local expr_type = std.ispace(index_type)

  return ast.typed.expr.Ispace {
    index_type = index_type,
    extent = extent,
    start = start,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_region(cx, node)
  local ispace = type_check.expr(cx, node.ispace)
  local ispace_type = std.check_read(cx, ispace)
  if not std.is_ispace(ispace_type) then
    report.error(node, "type mismatch in argument 1: expected an ispace but got " .. tostring(ispace_type))
  end

  local ispace_symbol
  if ispace:is(ast.typed.expr.ID) then
    ispace_symbol = ispace.value
  else
    ispace_symbol = std.newsymbol()
  end
  local region = std.region(ispace_symbol, node.fspace_type)

  -- Hack: Stuff the ispace type back into the ispace symbol so it is
  -- accessible to the region type.
  if not ispace_symbol:hastype() then
    ispace_symbol:settype(ispace_type)
  end
  assert(std.type_eq(ispace_symbol:gettype(), ispace_type))

  std.add_privilege(cx, std.reads, region, data.newtuple())
  std.add_privilege(cx, std.writes, region, data.newtuple())
  -- Freshly created regions are, by definition, disjoint from all
  -- other regions.
  for other_region, _ in cx.region_universe:items() do
    assert(not std.type_eq(region, other_region))
    -- But still, don't bother litering the constraint space with
    -- trivial constraints.
    if std.type_maybe_eq(region:fspace(), other_region:fspace()) then
      std.add_constraint(cx, region, other_region, std.disjointness, true)
    end
  end
  cx:intern_region(region)

  return ast.typed.expr.Region {
    ispace = ispace,
    fspace_type = node.fspace_type,
    expr_type = region,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_partition(cx, node)
  local disjointness = node.disjointness
  local completeness = node.completeness or std.incomplete
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)
  local coloring = type_check.expr(cx, node.coloring)
  local coloring_type = std.check_read(cx, coloring)
  local colors = node.colors and type_check.expr(cx, node.colors)
  local colors_type = colors and std.check_read(cx, colors)

  -- Note: This test can't fail because disjointness is tested in the parser.
  assert(disjointness:is(ast.disjointness_kind))
  -- Note: Same, except defaults to incomplete when unspecified.
  assert(completeness:is(ast.completeness_kind))

  if not std.is_region(region_type) then
    report.error(node, "type mismatch in argument 2: expected region but got " ..
                tostring(region_type))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if colors and not std.is_ispace(colors_type) then
    report.error(node, "type mismatch in argument 4: expected ispace but got " ..
                tostring(colors_type))
  end

  if region_type:is_opaque() then
    if colors then
      if coloring_type ~= std.c.legion_point_coloring_t then
        report.error(node,
                  "type mismatch in argument 3: expected legion_point_coloring_t but got " ..
                    tostring(coloring_type))
      end
    else
      if coloring_type ~= std.c.legion_coloring_t then
        report.error(node,
                  "type mismatch in argument 3: expected legion_coloring_t but got " ..
                    tostring(coloring_type))
      end
    end
  else
    if colors then
      if coloring_type ~= std.c.legion_domain_point_coloring_t and
        coloring_type ~= std.c.legion_multi_domain_point_coloring_t
      then
        report.error(node,
                  "type mismatch in argument 3: expected legion_domain_point_coloring_t or legion_multi_domain_point_coloring_t but got " ..
                    tostring(coloring_type))
      end
    else
      if coloring_type ~= std.c.legion_domain_coloring_t then
        report.error(node,
                  "type mismatch in argument 3: expected legion_domain_coloring_t but got " ..
                    tostring(coloring_type))
      end
    end
  end

  if coloring_type == std.c.legion_coloring_t then
    report.warn(node, "WARNING: using old style partition API with legion_coloring_t, please consider upgrading to legion_point_coloring_t")
  end

  if coloring_type == std.c.legion_domain_coloring_t then
    report.warn(node, "WARNING: using old style partition API with legion_domain_coloring_t, please consider upgrading to legion_domain_point_coloring_t")
  end

  local region_symbol
  if region:is(ast.typed.expr.ID) then
    region_symbol = region.value
  else
    region_symbol = std.newsymbol()
  end
  local colors_symbol
  if colors and colors:is(ast.typed.expr.ID) then
    colors_symbol = colors.value
  elseif colors then
    colors_symbol = std.newsymbol(colors_type)
  end
  local expr_type = std.partition(disjointness, completeness, region_symbol, colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(region_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == region_type)

  return ast.typed.expr.Partition {
    disjointness = disjointness,
    completeness = node.completeness,
    region = region,
    coloring = coloring,
    colors = colors,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_partition_equal(cx, node)
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)
  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.check_read(cx, colors)

  if not std.is_region(region_type) then
    report.error(node, "type mismatch in argument 1: expected region but got " ..
                tostring(region_type))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_ispace(colors_type) then
    report.error(node, "type mismatch in argument 2: expected ispace but got " ..
                tostring(colors_type))
  end

  local region_symbol
  if region:is(ast.typed.expr.ID) then
    region_symbol = region.value
  else
    region_symbol = std.newsymbol(region_type)
  end
  local colors_symbol
  if colors:is(ast.typed.expr.ID) then
    colors_symbol = colors.value
  else
    colors_symbol = std.newsymbol(colors_type)
  end
  local expr_type = std.partition(std.disjoint, region_symbol, colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(region_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == region_type)

  return ast.typed.expr.PartitionEqual {
    region = region,
    colors = colors,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_partition_by_field(cx, node)
  local completeness = node.completeness or std.incomplete

  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)

  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.check_read(cx, colors)

  if #region.fields ~= 1 then
    report.error(node, "type mismatch in argument 1: expected 1 field but got " ..
                tostring(#region.fields))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_ispace(colors_type) then
    report.error(node, "type mismatch in argument 2: expected ispace but got " ..
                tostring(colors_type))
  end

  -- Field type should be the same as the base type of the colors space is
  local field_type = std.get_field_path(region_type:fspace(), region.fields[1])
  if not std.type_eq(field_type, colors_type.index_type) then
    report.error(node, "type mismatch in argument 1: expected field of type " .. tostring(colors_type.index_type) ..
                " but got " .. tostring(field_type))
  end

  local region_symbol
  if region.region:is(ast.typed.expr.ID) then
    region_symbol = region.region.value
  else
    region_symbol = std.newsymbol()
  end
  local colors_symbol
  if colors:is(ast.typed.expr.ID) then
    colors_symbol = colors.value
  else
    colors_symbol = std.newsymbol(colors_type)
  end
  local expr_type = std.partition(std.disjoint, completeness, region_symbol, colors_symbol)

  if not std.check_privilege(cx, std.reads, region_type, region.fields[1]) then
    report.error(
      node, "invalid privileges in argument 1: " .. tostring(std.reads) .. "(" ..
        (data.newtuple(expr_type.parent_region_symbol) .. region.fields[1]):mkstring(".") ..
        ")")
  end

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(region_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == region_type)

  return ast.typed.expr.PartitionByField {
    completeness = node.completeness,
    region = region,
    colors = colors,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_partition_by_restriction(cx, node)
  local disjointness = node.disjointness or std.aliased
  local completeness = node.completeness or std.incomplete

  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)

  local transform = type_check.expr(cx, node.transform)
  local transform_type = std.check_read(cx, transform)

  local extent = type_check.expr(cx, node.extent)
  local extent_type = std.check_read(cx, extent)

  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.check_read(cx, colors)

  if not std.is_region(region_type) then
    report.error(node, "type mismatch in argument 1: expected region type but got " ..
                 tostring(region_type))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_transform_type(transform_type) then
    report.error(node, "type mismatch in argument 2: expected transform type but got " ..
                 tostring(transform_type))
  end

  if not std.is_rect_type(extent_type) then
    report.error(node, "type mismatch in argument 3: expected rect type but got " ..
                 tostring(extent_type))
  end

  if not std.is_ispace(colors_type) then
    report.error(node, "type mismatch in argument 4: expected ispace but got " ..
                 tostring(colors_type))
  end

  local M = region_type:ispace().dim
  local N = colors_type.dim

  if transform_type.M ~= M then
    report.error(node, "type mismatch: expected transform(" .. tostring(M) .. ",*) type but got " ..
                 tostring(transform_type))
  end

  if transform_type.N ~= N then
    report.error(node, "type mismatch: expected transform(*," .. tostring(N) .. ") type but got " ..
                 tostring(transform_type))
  end

  if extent_type.dim ~= M then
    report.error(node, "type mismatch: expected rect" .. tostring(M) .. "d type but got " ..
                 tostring(extent_type))
  end

  local region_symbol
  if region:is(ast.typed.expr.ID) then
    region_symbol = region.value
  else
    region_symbol = std.newsymbol()
  end

  local colors_symbol
  if colors:is(ast.typed.expr.ID) then
    colors_symbol = colors.value
  else
    colors_symbol = std.newsymbol(colors_type)
  end

  local expr_type = std.partition(disjointness, completeness, region_symbol, colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(region_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == region_type)

  return ast.typed.expr.PartitionByRestriction {
    disjointness = node.disjointness,
    completeness = node.completeness,
    region = region,
    transform = transform,
    extent = extent,
    colors = colors,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_image(cx, node)
  local disjointness = node.disjointness or std.aliased
  local completeness = node.completeness or std.incomplete
  local parent = type_check.expr(cx, node.parent)
  local parent_type = std.check_read(cx, parent)
  local partition = type_check.expr(cx, node.partition)
  local partition_type = std.check_read(cx, partition)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)

  if not std.is_region(parent_type) then
    report.error(node, "type mismatch in argument 1: expected region but got " ..
                tostring(parent_type))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_partition(partition_type) then
    report.error(node, "type mismatch in argument 2: expected partition but got " ..
                tostring(partition_type))
  end

  if #region.fields ~= 1 then
    report.error(node, "type mismatch in argument 3: expected 1 field but got " ..
                tostring(#region.fields))
  end

  local field_type = std.get_field_path(region_type:fspace(), region.fields[1])

  local index_type = field_type
  if std.is_bounded_type(index_type) then
    index_type = index_type.index_type
  end
  if not (std.is_index_type(index_type) or std.is_rect_type(index_type)) then
    report.error(node, "type mismatch in argument 3: expected field of index or rect type but got " .. tostring(field_type))
  end

  -- TODO: indexspaces should be parametrized by index types.
  --       currently they only support 64-bit points, which is why we do this check here.
  local function is_base_type_64bit(ty)
    if std.type_eq(ty, opaque) or std.type_eq(ty, int64) then
      return true
    elseif ty:isstruct() then
      for _, entry in ipairs(ty:getentries()) do
        local entry_type = entry[2] or entry.type
        if not is_base_type_64bit(entry_type) then return false end
      end
      return true
    else return false end
  end

  if not is_base_type_64bit(index_type.base_type) then
    report.error(node, "type mismatch in argument 3: expected field of 64-bit index type (for now) but got " .. tostring(field_type))
  end

  if index_type.dim ~= parent_type:ispace().index_type.dim and not (index_type.dim <= 1 and parent_type:ispace().index_type.dim <= 1) then
    report.error(node, "type mismatch in argument 3: expected field with dim " .. tostring(parent_type:ispace().index_type.dim) .. " but got " .. tostring(field_type))
  end

  local region_symbol
  if region.region:is(ast.typed.expr.ID) then
    region_symbol = region.region.value
  else
    region_symbol = std.newsymbol(region_type)
  end

  if not std.check_privilege(cx, std.reads, region_type, region.fields[1]) then
    report.error(
      node, "invalid privileges in argument 3: " .. tostring(std.reads) .. "(" ..
        (data.newtuple(region_symbol) .. region.fields[1]):mkstring(".") ..
        ")")
  end

  local parent_symbol
  if parent:is(ast.typed.expr.ID) then
    parent_symbol = parent.value
  else
    parent_symbol = std.newsymbol()
  end
  local expr_type = std.partition(disjointness, completeness, parent_symbol, partition_type.colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(parent_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == parent_type)

  -- Check that partition's parent region is a subregion of region.
  do
    local constraint = std.constraint(
      partition_type.parent_region_symbol,
      region_symbol,
      std.subregion)
    if not std.check_constraint(cx, constraint) then
      report.error(node, "invalid image missing constraint " ..
                  tostring(constraint.lhs) .. " " .. tostring(constraint.op) ..
                  " " .. tostring(constraint.rhs))
    end
  end

  if std.is_bounded_type(field_type) and field_type:is_ptr() then
    -- Check that parent is a subregion of the field bounds.
    for _, bound_symbol in ipairs(field_type.bounds_symbols) do
      local constraint = std.constraint(
        parent_symbol,
        bound_symbol,
        std.subregion)
      if not std.check_constraint(cx, constraint) then
        report.error(node, "invalid image missing constraint " ..
                    tostring(constraint.lhs) .. " " .. tostring(constraint.op) ..
                    " " .. tostring(constraint.rhs))
      end
    end
  end

  return ast.typed.expr.Image {
    disjointness = node.disjointness,
    completeness = node.completeness,
    parent = parent,
    partition = partition,
    region = region,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_image_by_task(cx, node)
  local disjointness = node.disjointness or std.aliased
  local completeness = node.completeness or std.incomplete
  local parent = type_check.expr(cx, node.parent)
  local parent_type = std.check_read(cx, parent)
  local index_type = parent_type:ispace().index_type
  local partition = type_check.expr(cx, node.partition)
  local partition_type = std.check_read(cx, partition)
  local task = type_check.expr_function(cx, node.region.region)
  local task_type
  if std.is_task(task.value) then
    task_type = task.value:get_type()
  else
    task_type = task.value:gettype()
  end

  if parent_type:is_opaque() then
    report.error(node, "type mismatch in argument 1: expected region with structured indexspace " ..
                 "but got one with unstructured indexspace")
  end

  if index_type ~= partition_type:parent_region():ispace().index_type then
    report.error(node, "index type mismatch between " .. tostring(index_type) ..
                 " (argument 1) and " .. tostring(partition_type:parent_region():ispace().index_type) ..
                 " (argument 2)")
  end

  if not std.is_region(parent_type) then
    report.error(node, "type mismatch in argument 1: expected region but got " ..
                tostring(parent_type))
  end

  if parent_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_partition(partition_type) then
    report.error(node, "type mismatch in argument 2: expected partition but got " ..
                tostring(partition_type))
  end

  if not std.is_task(task.value) then
    report.error(node, "type mismatch in argument 3: expected task but got a non-task value")
  end

  local rect_type = std.rect_type(index_type)
  local expected = terralib.types.functype(terralib.newlist({rect_type}), rect_type, false)
  if task_type ~= expected then
    report.error(node, "type mismatch in argument 3: expected " .. tostring(expected) ..
                 " but got " .. tostring(task_type))
  end

  local parent_symbol
  if parent:is(ast.typed.expr.ID) then
    parent_symbol = parent.value
  else
    parent_symbol = std.newsymbol()
  end
  local expr_type = std.partition(disjointness, completeness, parent_symbol, partition_type.colors_symbol)

  return ast.typed.expr.ImageByTask {
    disjointness = node.disjointness,
    completeness = node.completeness,
    parent = parent,
    partition = partition,
    task = task,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_preimage(cx, node)
  local disjointness = node.disjointness
  local completeness = node.completeness or std.incomplete
  local parent = type_check.expr(cx, node.parent)
  local parent_type = std.check_read(cx, parent)
  local partition = type_check.expr(cx, node.partition)
  local partition_type = std.check_read(cx, partition)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)

  if not std.is_region(parent_type) then
    report.error(node, "type mismatch in argument 1: expected region but got " ..
                tostring(parent_type))
  end

  if region_type:is_projected() then
    report.error(node, "a projected region cannot be partitioned")
  end

  if not std.is_partition(partition_type) then
    report.error(node, "type mismatch in argument 2: expected partition but got " ..
                tostring(partition_type))
  end

  if #region.fields ~= 1 then
    report.error(node, "type mismatch in argument 3: expected 1 field but got " ..
                tostring(#region.fields))
  end

  local field_type = std.get_field_path(region_type:fspace(), region.fields[1])
  if not ((std.is_bounded_type(field_type) and std.is_index_type(field_type.index_type)) or
           std.is_index_type(field_type) or std.is_rect_type(field_type)) then
    report.error(node, "type mismatch in argument 3: expected field of index or rect type but got " .. tostring(field_type))
  else
    -- TODO: indexspaces should be parametrized by index types.
    --       currently they only support 64-bit points, which is why we do this check here.
    local function is_base_type_64bit(ty)
      if std.type_eq(ty, opaque) or std.type_eq(ty, int64) then
        return true
      elseif ty:isstruct() then
        for _, entry in ipairs(ty:getentries()) do
          local entry_type = entry[2] or entry.type
          if not is_base_type_64bit(entry_type) then return false end
        end
        return true
      else return false end
    end

    local index_type = field_type
    if std.is_bounded_type(index_type) then
      index_type = index_type.index_type
    end
    if not is_base_type_64bit(index_type.base_type) then
      report.error(node, "type mismatch in argument 3: expected field of 64-bit index type (for now) but got " .. tostring(field_type))
    end
  end

  local region_symbol
  if region.region:is(ast.typed.expr.ID) then
    region_symbol = region.region.value
  else
    region_symbol = std.newsymbol(region_type)
  end

  if not std.check_privilege(cx, std.reads, region_type, region.fields[1]) then
    report.error(
      node, "invalid privileges in argument 3: " .. tostring(std.reads) .. "(" ..
        (data.newtuple(region_symbol) .. region.fields[1]):mkstring(".") ..
        ")")
  end

  local parent_symbol
  if parent:is(ast.typed.expr.ID) then
    parent_symbol = parent.value
  else
    parent_symbol = std.newsymbol()
  end
  if std.is_rect_type(field_type) then
    disjointness = disjointness or std.aliased
  else
    disjointness = disjointness or partition_type.disjointness
  end
  local expr_type = std.partition(disjointness, completeness, parent_symbol, partition_type.colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(parent_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == parent_type)

  -- Check that parent is a subregion of region.
  do
    local constraint = std.constraint(
      parent_symbol,
      region_symbol,
      std.subregion)
    if not std.check_constraint(cx, constraint) then
      report.error(node, "invalid image missing constraint " ..
                  tostring(constraint.lhs) .. " " .. tostring(constraint.op) ..
                  " " .. tostring(constraint.rhs))
    end
  end

  if std.is_bounded_type(field_type) and field_type:is_ptr() then
    -- Check that partitions's parent is a subregion of the field bounds.
    for _, bound_symbol in ipairs(field_type.bounds_symbols) do
      local constraint = std.constraint(
        partition_type.parent_region_symbol,
        bound_symbol,
        std.subregion)
      if not std.check_constraint(cx, constraint) then
        report.error(node, "invalid image missing constraint " ..
                    tostring(constraint.lhs) .. " " .. tostring(constraint.op) ..
                    " " .. tostring(constraint.rhs))
      end
    end
  end

  return ast.typed.expr.Preimage {
    disjointness = node.disjointness,
    completeness = node.completeness,
    partition = partition,
    region = region,
    parent = parent,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_cross_product(cx, node)
  local args = node.args:map(function(arg) return type_check.expr(cx, arg) end)
  local arg_types = args:map(function(arg) return std.check_read(cx, arg) end)

  if #arg_types < 2 then
    report.error(node, "cross product expected at least 2 arguments, got " ..
                tostring(#arg_types))
  end

  for i, arg_type in ipairs(arg_types) do
    if not std.is_partition(arg_type) then
      report.error(node, "type mismatch in argument " .. tostring(i) ..
                  ": expected partition but got " .. tostring(arg_type))
    end
  end

  local arg_symbols = args:map(
    function(arg)
      if arg:is(ast.typed.expr.ID) then
        return arg.value
      else
        return std.newsymbol(std.as_read(arg.expr_type))
      end
  end)
  local expr_type = std.cross_product(unpack(arg_symbols))

  return ast.typed.expr.CrossProduct {
    args = args,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_cross_product_array(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.as_read(lhs.expr_type)
  local lhs_symbol = std.newsymbol(lhs_type)
  local disjointness = node.disjointness
  local rhs_type = std.partition(disjointness, lhs_type.parent_region_symbol)
  local rhs_symbol = std.newsymbol(rhs_type)
  local expr_type = std.cross_product(lhs_symbol, rhs_symbol)
  local colorings = type_check.expr(cx, node.colorings)

  return ast.typed.expr.CrossProductArray {
    lhs = lhs,
    disjointness = disjointness,
    colorings = colorings,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_slice_partition(cx, node)
  local partition = type_check.expr(cx, node.partition)
  local partition_type = std.check_read(cx, partition)
  local indices = type_check.expr(cx, node.indices)
  local indices_type = std.check_read(cx, indices)
  if not std.is_partition(partition_type) then
    report.error(node, "type mismatch: expected a partition but got " .. tostring(partition_type))
  end
  if not std.validate_implicit_cast(indices_type, std.list(int)) then
    report.error(node, "type mismatch: expected " .. tostring(std.list(int)) .. " but got " .. tostring(indices_type))
  end
  indices = insert_implicit_cast(indices, indices_type, std.list(int))
  local expr_type = std.list(
    std.region(
      std.ispace(partition_type:parent_region():ispace().index_type),
      partition_type:parent_region():fspace()),
    partition_type, 1)
  -- FIXME: The privileges for these region aren't necessarily exactly
  -- one level up.

  std.copy_privileges(cx, partition_type:parent_region(), expr_type)
  -- FIXME: Copy constraints.
  cx:intern_region(expr_type)

  return ast.typed.expr.ListSlicePartition {
    partition = partition,
    indices = indices,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_duplicate_partition(cx, node)
  local partition = type_check.expr(cx, node.partition)
  local partition_type = std.check_read(cx, partition)
  local indices = type_check.expr(cx, node.indices)
  local indices_type = std.check_read(cx, indices)
  if not std.is_partition(partition_type) then
    report.error(node, "type mismatch: expected a partition but got " .. tostring(partition_type))
  end
  if not std.validate_implicit_cast(indices_type, std.list(int)) then
    report.error(node, "type mismatch: expected " .. tostring(std.list(int)) .. " but got " .. tostring(indices_type))
  end
  indices = insert_implicit_cast(indices, indices_type, std.list(int))
  local expr_type = std.list(
    std.region(
      std.newsymbol(std.ispace(partition_type:parent_region():ispace().index_type)),
      partition_type:parent_region():fspace()),
    partition_type)

  std.add_privilege(cx, std.reads, expr_type, data.newtuple())
  std.add_privilege(cx, std.writes, expr_type, data.newtuple())
  -- Freshly created regions are, by definition, disjoint from all
  -- other regions.
  for other_region, _ in cx.region_universe:items() do
    assert(not std.type_eq(expr_type, other_region))
    -- But still, don't bother litering the constraint space with
    -- trivial constraints.
    if std.type_maybe_eq(expr_type:fspace(), other_region:fspace()) then
      std.add_constraint(cx, expr_type, other_region, std.disjointness, true)
    end
  end
  cx:intern_region(expr_type)

  return ast.typed.expr.ListDuplicatePartition {
    partition = partition,
    indices = indices,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_cross_product(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_read(cx, lhs)
  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)
  if not std.is_list_of_regions(lhs_type) and lhs_type:depth() == 1 then
    report.error(node, "type mismatch: expected a list of regions but got " .. tostring(lhs_type))
  end
  if not std.is_list_of_regions(rhs_type) and lhs_type:depth() == 1 then
    report.error(node, "type mismatch: expected a list of regions but got " .. tostring(rhs_type))
  end
  local expr_type
  if node.shallow then
    expr_type = std.list(
      std.list(rhs_type:subregion_dynamic(), nil, nil, nil, true),
      nil, nil, nil, true)
  else
    expr_type = std.list(std.list(rhs_type:subregion_dynamic(), nil, 1), nil, 1)
  end

  std.add_constraint(cx, expr_type, rhs_type, std.subregion, false)

  return ast.typed.expr.ListCrossProduct {
    lhs = lhs,
    rhs = rhs,
    shallow = node.shallow,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_cross_product_complete(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_read(cx, lhs)
  local product = type_check.expr(cx, node.product)
  local product_type = std.check_read(cx, product)
  if not std.is_list_of_regions(lhs_type) and lhs_type:depth() == 1 then
    report.error(node, "type mismatch: expected a list of regions but got " .. tostring(lhs_type))
  end
  if not std.is_list_of_regions(product_type) and product_type:depth() == 2 then
    report.error(node, "type mismatch: expected a list of lists of regions but got " .. tostring(product_type))
  end

  local expr_type = std.list(
    std.list(product_type:subregion_dynamic(), nil, 1),
    nil, 1)

  std.add_constraint(cx, expr_type, product_type, std.subregion, false)

  return ast.typed.expr.ListCrossProductComplete {
    lhs = lhs,
    product = product,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_phase_barriers(cx, node)
  local product = type_check.expr(cx, node.product)
  local product_type = std.check_read(cx, product)
  if not std.is_list_of_regions(product_type) or product_type:list_depth() ~= 2
  then
    report.error(node, "type mismatch: expected a list cross-product but got " ..
                tostring(product_type))
  end
  local expr_type = std.list(std.list(std.phase_barrier))

  return ast.typed.expr.ListPhaseBarriers {
    product = product,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_invert(cx, node)
  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)
  local product = type_check.expr(cx, node.product)
  local product_type = std.check_read(cx, product)
  local barriers = type_check.expr(cx, node.barriers)
  local barriers_type = std.check_read(cx, barriers)
  if not std.is_list_of_regions(rhs_type) or rhs_type:list_depth() ~= 1 then
    report.error(node, "type mismatch: expected a list of regions but got " ..
                tostring(product_type))
  end
  if not std.is_list_of_regions(product_type) or product_type:list_depth() ~= 2
  then
    report.error(node, "type mismatch: expected a list cross-product but got " ..
                tostring(product_type))
  end
  if not std.type_eq(barriers_type, std.list(std.list(std.phase_barrier))) then
    report.error(node, "type mismatch: expected " ..
                tostring(std.list(std.list(std.phase_barrier))) ..
                " but got " .. tostring(barriers_type))
  end
  local expr_type = barriers_type

  return ast.typed.expr.ListInvert {
    rhs = rhs,
    product = product,
    barriers = barriers,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_range(cx, node)
  local start = type_check.expr(cx, node.start)
  local start_type = std.check_read(cx, start)
  local stop = type_check.expr(cx, node.stop)
  local stop_type = std.check_read(cx, stop)
  if not std.validate_implicit_cast(start_type, int) then
    report.error(node, "type mismatch: expected " .. tostring(int) .. " but got " .. tostring(start_type))
  end
  if not std.validate_implicit_cast(stop_type, int) then
    report.error(node, "type mismatch: expected " .. tostring(int) .. " but got " .. tostring(stop_type))
  end
  start = insert_implicit_cast(start, start_type, int)
  stop = insert_implicit_cast(stop, stop_type, int)
  local expr_type = std.list(int)

  return ast.typed.expr.ListRange {
    start = start,
    stop = stop,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_ispace(cx, node)
  local ispace = type_check.expr(cx, node.ispace)
  local ispace_type = std.check_read(cx, ispace)
  if not std.is_ispace(ispace_type) then
    report.error(node, "type mismatch in argument 1: expected an ispace but got " .. tostring(ispace_type))
  end
  local expr_type = std.list(ispace_type.index_type)

  return ast.typed.expr.ListIspace {
    ispace = ispace,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_list_from_element(cx, node)
  local list = type_check.expr(cx, node.list)
  local list_type = std.check_read(cx, list)
  if not std.is_list(list_type) then
    report.error(node, "type mismatch in argument 1: expected a list but got " .. tostring(list_type))
  end
  local value = type_check.expr(cx, node.value)
  local expr_type = std.as_read(value.expr_type)
  for i = 1, list_type:list_depth() do
    expr_type = std.list(expr_type)
  end

  return ast.typed.expr.ListFromElement {
    list = list,
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_phase_barrier(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  if not std.validate_implicit_cast(value_type, int) then
    report.error(node, "type mismatch: expected " .. tostring(int) .. " but got " .. tostring(value_type))
  end
  value = insert_implicit_cast(value, value_type, int)

  return ast.typed.expr.PhaseBarrier {
    value = value,
    expr_type = std.phase_barrier,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_dynamic_collective(cx, node)
  local arrivals = type_check.expr(cx, node.arrivals)
  local arrivals_type = std.check_read(cx, arrivals)
  if not std.validate_implicit_cast(arrivals_type, int) then
    report.error(node, "type mismatch in argument 3: expected " .. tostring(int) .. " but got " .. tostring(arrivals_type))
  end
  arrivals = insert_implicit_cast(arrivals, arrivals_type, int)

  local expr_type = std.dynamic_collective(node.value_type)

  return ast.typed.expr.DynamicCollective {
    value_type = node.value_type,
    op = node.op,
    arrivals = arrivals,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_dynamic_collective_get_result(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  if not std.is_dynamic_collective(value_type) then
    report.error(node, "type mismatch: expected a dynamic collective but got " .. tostring(value_type))
  end
  local expr_type = value_type.result_type

  return ast.typed.expr.DynamicCollectiveGetResult {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_advance(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  if not (std.validate_implicit_cast(value_type, std.phase_barrier) or
            std.is_dynamic_collective(value_type) or
            std.is_list_of_phase_barriers(value_type))
  then
    report.error(node, "type mismatch: expected a phase barrier or dynamic collective but got " .. tostring(value_type))
  end
  local expr_type = value_type

  return ast.typed.expr.Advance {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_adjust(cx, node)
  local barrier = type_check.expr(cx, node.barrier)
  local barrier_type = std.check_read(cx, barrier)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  if not (std.validate_implicit_cast(barrier_type, std.phase_barrier) or
            std.is_list_of_phase_barriers(barrier_type) or
          std.is_dynamic_collective(barrier_type)) then
    report.error(node, "type mismatch in argument 1: expected a phase barrier but got " .. tostring(barrier_type))
  end
  if not std.validate_implicit_cast(value_type, int) then
    report.error(node, "type mismatch in argument 2: expected " ..
                tostring(int) .. " but got " .. tostring(value_type))
  end
  local expr_type = barrier_type

  return ast.typed.expr.Adjust {
    barrier = barrier,
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_arrive(cx, node)
  local barrier = type_check.expr(cx, node.barrier)
  local barrier_type = std.check_read(cx, barrier)
  local value = node.value and type_check.expr(cx, node.value)
  local value_type = node.value and std.check_read(cx, value)
  if not (std.is_phase_barrier(barrier_type) or
          std.is_list_of_phase_barriers(barrier_type) or
          std.is_dynamic_collective(barrier_type)) then
    report.error(node, "type mismatch in argument 1: expected a phase barrier but got " .. tostring(barrier_type))
  end
  if std.is_phase_barrier(barrier_type) and value_type then
    report.error(node, "type mismatch in arrive: expected 1 argument but got 2")
  end
  if std.is_dynamic_collective(barrier_type) and not std.validate_implicit_cast(value_type, barrier_type.result_type) then
    report.error(node, "type mismatch in argument 2: expected " ..
                tostring(barrier_type.result_type) .. " but got " ..
                tostring(value_type))
  end
  local expr_type = barrier_type

  return ast.typed.expr.Arrive {
    barrier = barrier,
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_await(cx, node)
  local barrier = type_check.expr(cx, node.barrier)
  local barrier_type = std.check_read(cx, barrier)
  if not (std.is_phase_barrier(barrier_type) or std.is_list_of_phase_barriers(barrier_type)) then
    report.error(node, "type mismatch in argument 1: expected a phase barrier but got " .. tostring(barrier_type))
  end
  local expr_type = terralib.types.unit

  return ast.typed.expr.Await {
    barrier = barrier,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_copy(cx, node)
  local src = type_check.expr_region_root(cx, node.src)
  local src_type = std.check_read(cx, src)
  local dst = type_check.expr_region_root(cx, node.dst)
  local dst_type = std.check_read(cx, dst)
  local conditions = terralib.newlist()
  for _, condition in ipairs(node.conditions) do
    conditions:insertall(type_check.expr_condition(cx, condition))
  end
  local expr_type = terralib.types.unit

  if src_type:list_depth() > dst_type:list_depth() then
    report.error(node, "must copy from less-nested to more-nested list")
  end

  for _, condition in ipairs(conditions) do
    local value = condition.value
    local value_type = std.check_read(cx, value)
    for _, condition_kind in ipairs(condition.conditions) do
      if condition_kind:is(ast.condition_kind.Awaits) and
        value_type:list_depth() > dst_type:list_depth()
      then
        report.error(node, "copy must await list of same or less depth than destination")
      elseif condition_kind:is(ast.condition_kind.Arrives) and
        value_type:list_depth() > dst_type:list_depth()
      then
        report.error(node, "copy must arrive list of same or less depth than destination")
      end
    end
  end

  if #src.fields ~= #dst.fields then
    report.error(node, "mismatch in number of fields between " .. tostring(#src.fields) ..
                " and " .. tostring(#dst.fields))
  end

  for i, src_field in ipairs(src.fields) do
    local dst_field = dst.fields[i]
    local src_type = std.get_field_path(src_type:fspace(), src_field)
    local dst_type = std.get_field_path(dst_type:fspace(), dst_field)
    if not std.type_eq(src_type, dst_type) then
      report.error(node, "type mismatch between " .. tostring(src_type) ..
                  " and " .. tostring(dst_type))
    end
  end

  for _, field_path in ipairs(src.fields) do
    if not std.check_privilege(cx, std.reads, src_type, field_path) then
      local src_symbol
      if node.src.region:is(ast.specialized.expr.ID) then
        src_symbol = node.src.region.value
      else
        src_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in copy: " .. tostring(std.reads) .. "(" ..
          (data.newtuple(src_symbol) .. field_path):mkstring(".") .. ")")
    end
  end
  for _, field_path in ipairs(dst.fields) do
    if node.op then
      if not std.check_privilege(cx, std.reduces(node.op), dst_type, field_path)
      then
        local dst_symbol
        if node.dst.region:is(ast.specialized.expr.ID) then
          dst_symbol = node.dst.region.value
        else
          dst_symbol = std.newsymbol()
        end
        report.error(
          node,
          "invalid privileges in copy: " .. tostring(std.reduces(node.op)) ..
            "(" .. (data.newtuple(dst_symbol) .. field_path):mkstring(".") ..
            ")")
      end
    else
      if not std.check_privilege(cx, std.reads, dst_type, field_path) then
        local dst_symbol
        if node.dst.region:is(ast.specialized.expr.ID) then
          dst_symbol = node.dst.region.value
        else
          dst_symbol = std.newsymbol()
        end
        report.error(
          node, "invalid privileges in copy: " .. tostring(std.reads) ..
            "(" .. (data.newtuple(dst_symbol) .. field_path):mkstring(".") ..
            ")")
      end
      if not std.check_privilege(cx, std.writes, dst_type, field_path) then
        local dst_symbol
        if node.dst.region:is(ast.specialized.expr.ID) then
          dst_symbol = node.dst.region.value
        else
          dst_symbol = std.newsymbol()
        end
        report.error(
          node, "invalid privileges in copy: " .. tostring(std.writes) ..
            "(" .. (data.newtuple(dst_symbol) .. field_path):mkstring(".") ..
            ")")
      end
    end
  end

  return ast.typed.expr.Copy {
    src = src,
    dst = dst,
    op = node.op,
    conditions = conditions,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_fill(cx, node)
  local dst = type_check.expr_region_root(cx, node.dst)
  local dst_type = std.check_read(cx, dst)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)
  local conditions = terralib.newlist()
  for _, condition in ipairs(node.conditions) do
    conditions:insertall(type_check.expr_condition(cx, condition))
  end
  local expr_type = terralib.types.unit

  for i, dst_field in ipairs(dst.fields) do
    local dst_type = std.get_field_path(dst_type:fspace(), dst_field)
    if not std.validate_implicit_cast(value_type, dst_type) then
      report.error(node, "type mismatch between " .. tostring(value_type) ..
                  " and " .. tostring(dst_type))
    end
  end

  for _, field_path in ipairs(dst.fields) do
    if not std.check_privilege(cx, std.writes, dst_type, field_path) then
      local dst_symbol
      if node.dst.region:is(ast.specialized.expr.ID) then
        dst_symbol = node.dst.region.value
      else
        dst_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in fill: " .. tostring(std.writes) ..
          "(" .. (data.newtuple(dst_symbol) .. field_path):mkstring(".") .. ")")
    end
  end

  for _, field_path in ipairs(dst.fields) do
    local sliced, field_type = std.check_field_sliced(dst_type:fspace(), field_path)
    if not sliced then
      report.error(
        node, "partial fill with type " .. tostring(field_type) .. " is not allowed")
    end
  end

  return ast.typed.expr.Fill {
    dst = dst,
    value = value,
    conditions = conditions,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_acquire(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local conditions = terralib.newlist()
  for _, condition in ipairs(node.conditions) do
    conditions:insertall(type_check.expr_condition(cx, condition))
  end
  local expr_type = terralib.types.unit

  for _, field_path in ipairs(region.fields) do
    if not std.check_privilege(cx, std.reads, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in acquire: " .. tostring(std.reads) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
    if not std.check_privilege(cx, std.writes, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in acquire: " .. tostring(std.writes) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
  end

  return ast.typed.expr.Acquire {
    region = region,
    conditions = conditions,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_release(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local conditions = terralib.newlist()
  for _, condition in ipairs(node.conditions) do
    conditions:insertall(type_check.expr_condition(cx, condition))
  end
  local expr_type = terralib.types.unit

  for _, field_path in ipairs(region.fields) do
    if not std.check_privilege(cx, std.reads, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in release: " .. tostring(std.reads) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
    if not std.check_privilege(cx, std.writes, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in release: " .. tostring(std.writes) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
  end

  return ast.typed.expr.Release {
    region = region,
    conditions = conditions,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_attach_hdf5(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local filename = type_check.expr(cx, node.filename)
  local filename_type = std.check_read(cx, filename)
  local mode = type_check.expr(cx, node.mode)
  local mode_type = std.check_read(cx, mode)
  local field_map = node.field_map and type_check.expr(cx, node.field_map)
  local field_map_type = field_map and std.check_read(cx, field_map)
  local expr_type = terralib.types.unit

  if not std.is_region(region_type) then
    -- Explicitly disallow lists of regions.
    report.error(node, "type mismatch in argument 1: expected a region but got " .. tostring(region_type))
  end

  if not std.validate_implicit_cast(filename_type, rawstring) then
    report.error(node, "type mismatch in argument 2: expected " .. tostring(rawstring) .. " but got " .. tostring(filename_type))
  end
  filename = insert_implicit_cast(filename, filename_type, rawstring)

  if not std.validate_implicit_cast(mode_type, std.c.legion_file_mode_t) then
    report.error(node, "type mismatch in argument 3: expected " .. tostring(std.c.legion_file_mode_t) .. " but got " .. tostring(mode_type))
  end
  mode = insert_implicit_cast(mode, mode_type, std.c.legion_file_mode_t)

  if field_map and not std.validate_implicit_cast(field_map_type, &rawstring) then
    report.error(node, "type mismatch in argument 2: expected " .. tostring(&rawstring) .. " but got " .. tostring(field_map_type))
  end
  field_map = field_map and insert_implicit_cast(field_map, field_map_type, &rawstring)

  for _, field_path in ipairs(region.fields) do
    if not std.check_privilege(cx, std.reads, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in attach: " .. tostring(std.reads) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
    if not std.check_privilege(cx, std.writes, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in detach: " .. tostring(std.writes) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
  end

  return ast.typed.expr.AttachHDF5 {
    region = region,
    filename = filename,
    mode = mode,
    field_map = field_map,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_detach_hdf5(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local expr_type = terralib.types.unit

  if not std.is_region(region_type) then
    -- Explicitly disallow lists of regions.
    report.error(node, "type mismatch in argument 1: expected a region but got " .. tostring(region_type))
  end

  for _, field_path in ipairs(region.fields) do
    if not std.check_privilege(cx, std.reads, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in detach: " .. tostring(std.reads) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
    if not std.check_privilege(cx, std.writes, region_type, field_path) then
      local region_symbol
      if node.region.region:is(ast.specialized.expr.ID) then
        region_symbol = node.region.region.value
      else
        region_symbol = std.newsymbol()
      end
      report.error(
        node, "invalid privileges in detach: " .. tostring(std.writes) ..
          "(" .. (data.newtuple(region_symbol) .. field_path):mkstring(".") .. ")")
    end
  end

  return ast.typed.expr.DetachHDF5 {
    region = region,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_allocate_scratch_fields(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local expr_type = std.c.legion_field_id_t[#region.fields]

  return ast.typed.expr.AllocateScratchFields {
    region = region,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_with_scratch_fields(cx, node)
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)
  local field_ids = type_check.expr(cx, node.field_ids)
  local field_ids_type = std.check_read(cx, field_ids)

  local expr_type = std.region(
    std.newsymbol(region_type:ispace()),
    region_type:fspace())
  if std.is_list_of_regions(region_type) then
    for i = 1, region_type:list_depth() do
      expr_type = std.list(
        expr_type, region_type:partition(), region_type.privilege_depth)
    end
  end

  std.copy_privileges(cx, region_type, expr_type)

  return ast.typed.expr.WithScratchFields {
    region = region,
    field_ids = field_ids,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

local function unary_op_type(op)
  return function(cx, node, rhs_type)
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local function test()
      local terra query(rhs : rhs_type)
        return [ std.quote_unary_op(op, rhs) ]
      end
      return query:gettype().returntype
    end
    local valid, result_type = pcall(test)

    if not valid then
      if not rhs_type:isarray() then
        report.error(node, "invalid argument to unary operator " .. tostring(rhs_type))
      end

      local function test()
        local terra query(rhs : rhs_type.type)
          return [ std.quote_unary_op(op, rhs) ]
        end
        return query:gettype().returntype
      end

      local valid, result_type = pcall(test)
      if not valid then
        report.error(node, "invalid argument to unary operator " .. tostring(rhs_type))
      end
      return result_type[rhs_type.N]
    end

    return result_type
  end
end

local unary_ops = {
  ["-"] = unary_op_type("-"),
  ["not"] = unary_op_type("not"),
}

function type_check.expr_unary(cx, node)
  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)

  local expr_type = unary_ops[node.op](cx, node, rhs_type)

  return ast.typed.expr.Unary {
    op = node.op,
    rhs = rhs,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

local function binary_op_type(op)
  return function(cx, node, lhs_type, rhs_type)
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local function test()
      local terra query(lhs : lhs_type, rhs : rhs_type)
        return [ std.quote_binary_op(op, lhs, rhs) ]
      end
      return query:gettype().returntype
    end
    local valid, result_type = pcall(test)

    if not valid then
      if not (lhs_type:isarray() and std.type_eq(lhs_type, rhs_type)) then
        report.error(node, "type mismatch between " .. tostring(lhs_type) ..
                    " and " .. tostring(rhs_type))
      end

      local function test()
        local terra query(lhs : lhs_type.type, rhs : rhs_type.type)
          return [ std.quote_binary_op(op, lhs, rhs) ]
        end
        return query:gettype().returntype
      end

      local valid, result_type = pcall(test)
      if not valid then
        report.error(node, "type mismatch between " .. tostring(lhs_type) ..
                    " and " .. tostring(rhs_type))
      end
      return result_type[lhs_type.N]
    end

    return result_type
  end
end

local function binary_equality(op)
  local check = binary_op_type(op)
  return function(cx, node, lhs_type, rhs_type)
    if std.is_bounded_type(lhs_type) and std.is_bounded_type(rhs_type) then
      if not std.type_eq(lhs_type, rhs_type) then
        report.error(node, "type mismatch between " .. tostring(lhs_type) ..
                    " and " .. tostring(rhs_type))
      end
      return bool
    else
      return check(cx, node, lhs_type, rhs_type)
    end
  end
end

local binary_ops = {
  ["*"] = binary_op_type("*"),
  ["/"] = binary_op_type("/"),
  ["%"] = binary_op_type("%"),
  ["+"] = binary_op_type("+"),
  ["-"] = binary_op_type("-"),
  ["<"] = binary_op_type("<"),
  [">"] = binary_op_type(">"),
  ["^"] = binary_op_type("^"),
  ["<<"] = binary_op_type("<<"),
  [">>"] = binary_op_type(">>"),
  ["<="] = binary_op_type("<="),
  [">="] = binary_op_type(">="),
  ["=="] = binary_equality("=="),
  ["~="] = binary_equality("~="),
  ["and"] = binary_op_type("and"),
  ["or"] = binary_op_type("or"),
  ["max"] = binary_op_type("max"),
  ["min"] = binary_op_type("min"),
}

function type_check.expr_binary(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_read(cx, lhs)
  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)

  local expr_type
  if std.is_partition(lhs_type) then
    if not std.is_partition(rhs_type) then
      report.error(node.rhs, "type mismatch: expected a partition but got " .. tostring(rhs_type))
    end
    if not std.type_eq(lhs_type:fspace(), rhs_type:fspace()) then
      report.error(node, "type mismatch: expected partition of " .. tostring(lhs_type:fspace()) .. " but got partition of " .. tostring(rhs_type:fspace()))
    end
    if not (node.op == "-" or node.op == "&" or node.op == "|") then
      report.error(node.rhs, "operator " .. tostring(node.op) ..
                  " not supported on partitions")
    end

    local disjointness
    if node.op == "-" then
      disjointness = lhs_type.disjointness
    elseif node.op == "&" then
      if lhs_type:is_disjoint() or rhs_type:is_disjoint() then
        disjointness = std.disjoint
      else
        disjointness = std.aliased
      end
    elseif node.op == "|" then
      disjointness = std.aliased
    end

    expr_type = std.partition(
      disjointness, lhs_type.parent_region_symbol, lhs_type.colors_symbol)
  elseif std.is_region(lhs_type) then
    if lhs_type:is_projected() then
      report.error(node, "a projected region cannot be partitioned")
    end

    if not std.is_partition(rhs_type) then
      report.error(node.rhs, "type mismatch: expected a partition but got " .. tostring(rhs_type))
    end
    local lhs_index_type = lhs_type:ispace().index_type
    local rhs_index_type = rhs_type:parent_region():ispace().index_type
    if not std.type_eq(lhs_index_type, rhs_index_type) then
      report.error(node.rhs, "type mismatch: expected partition of " .. tostring(lhs_index_type) ..
          " but got partition of " .. tostring(rhs_index_type))
    end
    if node.op ~= "&" then
      report.error(node.rhs, "operator " .. tostring(node.op) ..  " not supported on partitions")
    end

    local region_symbol
    if lhs:is(ast.typed.expr.ID) then
      region_symbol = lhs.value
    else
      region_symbol = std.newsymbol(lhs_type)
    end
    expr_type = std.partition(
      rhs_type.disjointness, region_symbol, rhs_type.colors_symbol)
  elseif std.is_index_type(lhs_type) and (std.is_region(rhs_type) or std.is_ispace(rhs_type)) then
    if node.op ~= "<=" then
      report.error(node.rhs, "operator " .. tostring(node.op) ..
                  " not supported on " .. tostring(lhs_type) .. " and " ..
                  tostring(rhs_type))
    end
    expr_type = bool
  elseif std.is_ispace(lhs_type) and std.is_ispace(rhs_type) then
    if not (node.op == "&" or node.op == "|")  then
      report.error(node.rhs, "operator " .. tostring(node.op) ..
                  " not supported on " .. tostring(lhs_type) .. " and " ..
                  tostring(rhs_type))
    end
    expr_type = std.ispace(lhs_type.index_type)
  else
    if node.op == "&" or node.op == "|" then
      report.error(node.rhs, "operator " .. tostring(node.op) ..
                  " not supported on " .. tostring(lhs_type) .. " and " ..
                  tostring(rhs_type))
    end
    expr_type = binary_ops[node.op](cx, node, lhs_type, rhs_type)
  end

  return ast.typed.expr.Binary {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_deref(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if not (value_type:ispointer() or
          (std.is_bounded_type(value_type) and
           std.is_region(std.check_bounds(node, value_type)[1])))
  then
    report.error(node, "dereference of non-pointer type " .. tostring(value_type))
  end

  local expr_type
  if value_type:ispointer() then
    expr_type = std.rawref(value_type)
  elseif std.is_bounded_type(value_type) then
    expr_type = std.ref(value_type)
  else
    assert(false)
  end

  return ast.typed.expr.Deref {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_address_of(cx, node)
  local value = type_check.expr(cx, node.value)
  local ref_type = value.expr_type

  if not (std.is_ref(ref_type) or std.is_rawref(ref_type))
  then
    report.error(node, "attempting to take address of a non-l-value " .. tostring(ref_type))
  end

  if std.is_ref(ref_type) and #ref_type.field_path > 0 then
    report.error(node, "attempting to take address of a field of an element in a region")
  end

  local expr_type = ref_type.pointer_type

  return ast.typed.expr.AddressOf {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_import_ispace(cx, node)
  if not std.is_index_type(node.index_type) then
    report.error(node, "type mismatch in argument 1: expected index type but got " ..
      tostring(node.index_type))
  end
  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)
  if value_type ~= std.c.legion_index_space_t then
    report.error(node.value,
      "type mismatch in argument 2: expected an index space handle but got " ..
      tostring(value_type))
  end
  local expr_type = std.ispace(node.index_type)
  return ast.typed.expr.ImportIspace {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_import_region(cx, node)
  local ispace = type_check.expr(cx, node.ispace)
  local ispace_type = std.as_read(ispace.expr_type)

  if not std.is_ispace(ispace_type) then
    report.error(node.ispace, "type mismatch in argument 1: expected index space but got " ..
      tostring(ispace_type))
  end

  if not terralib.types.istype(node.fspace_type) then
    report.error(node, "type mismatch in argument 2: expected field space but got " ..
      tostring(node.fspace_type))
  end

  local ispace_symbol
  if ispace:is(ast.typed.expr.ID) then
    ispace_symbol = ispace.value
  else
    ispace_symbol = std.newsymbol()
  end
  local region = std.region(ispace_symbol, node.fspace_type)

  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)
  if value_type ~= std.c.legion_logical_region_t then
    report.error(node.value,
      "type mismatch in argument 3: expected a logical region handle but got " ..
      tostring(value_type))
  end

  local field_paths, _ = std.flatten_struct_fields(region:fspace())
  local expected = std.c.legion_field_id_t[#field_paths]

  local field_ids = type_check.expr(cx, node.field_ids)
  local field_ids_type = std.as_read(field_ids.expr_type)
  if field_ids_type ~= expected then
    report.error(node.field_ids,
      "type mismatch in argument 4: expected " .. tostring(expected) .. " but got " ..
      tostring(field_ids_type))
  end

  -- Hack: Stuff the ispace type back into the ispace symbol so it is
  -- accessible to the region type.
  if not ispace_symbol:hastype() then
    ispace_symbol:settype(ispace_type)
  end
  assert(std.type_eq(ispace_symbol:gettype(), ispace_type))

  std.add_privilege(cx, std.reads, region, data.newtuple())
  std.add_privilege(cx, std.writes, region, data.newtuple())
  -- Freshly imported regions are considered as disjoint from all
  -- other regions.
  for other_region, _ in cx.region_universe:items() do
    assert(not std.type_eq(region, other_region))
    if std.type_maybe_eq(region:fspace(), other_region:fspace()) then
      std.add_constraint(cx, region, other_region, std.disjointness, true)
    end
  end
  cx:intern_region(region)

  return ast.typed.expr.ImportRegion {
    ispace = ispace,
    value = value,
    field_ids = field_ids,
    expr_type = region,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_import_partition(cx, node)
  local region = type_check.expr(cx, node.region)
  local region_type = std.as_read(region.expr_type)
  if not std.is_region(region_type) then
    report.error(node.region, "type mismatch in argument 2: expected region but got " ..
      tostring(region_type))
  end

  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.as_read(colors.expr_type)
  if not std.is_ispace(colors_type) then
    report.error(node.colors, "type mismatch in argument 3: expected ispace but got " ..
      tostring(colors_type))
  end

  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)
  if value_type ~= std.c.legion_logical_partition_t then
    report.error(node.value,
      "type mismatch in argument 4: expected a logical partition handle but got " ..
      tostring(value_type))
  end

  local region_symbol
  if region:is(ast.typed.expr.ID) then
    region_symbol = region.value
  else
    region_symbol = std.newsymbol()
  end
  local colors_symbol
  if colors and colors:is(ast.typed.expr.ID) then
    colors_symbol = colors.value
  elseif colors then
    colors_symbol = std.newsymbol(colors_type)
  end
  local partition = std.partition(node.disjointness, region_symbol, colors_symbol)

  return ast.typed.expr.ImportPartition {
    region = region,
    colors = colors,
    value = value,
    expr_type = partition,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_import_cross_product(cx, node)
  local partitions = node.partitions:map(function(p) return type_check.expr(cx, p) end)
  local partition_type
  for idx, p in ipairs(partitions) do
    partition_type = std.as_read(p.expr_type)
    if not std.is_partition(partition_type) then
      report.error(p,
        "type mismatch in argument " .. tostring(idx) ..
          ": expected a partition but got " ..  tostring(partition_type))
    end
  end

  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.as_read(colors.expr_type)
  if not std.validate_implicit_cast(colors_type, std.c.legion_color_t[#partitions]) then
    report.error(colors,
      "type mismatch in argument " .. tostring(#partitions + 1) ..
        ": expected legion_color_t[" .. #partitions ..  "] but got " ..  tostring(colors_type))
  end

  local value = type_check.expr(cx, node.value)
  local value_type = std.as_read(value.expr_type)
  if not std.validate_implicit_cast(value_type, std.c.legion_terra_index_cross_product_t) then
    report.error(value,
      "type mismatch in argument " .. tostring(#node.partitions + 1) ..
        ": expected a logical cross product handle but got " ..
      tostring(value_type))
  end

  return ast.typed.expr.ImportCrossProduct {
    partitions = partitions,
    colors = colors,
    value = value,
    expr_type = std.cross_product(unpack(partitions:map(function(p) return p.value end))),
    annotations = node.annotations,
    span = node.span,
  }
end

local entry_tree = {}

function entry_tree.new()
  local t = {
    __tree = terralib.newlist(),
    __indices = data.newmap(),
    __fields = terralib.newlist(),
  }
  return setmetatable(t, entry_tree)
end

function entry_tree.is_entry_tree(x)
  return getmetatable(x) == entry_tree
end

function entry_tree:__index(field)
  return self.__tree[self.__indices[field]] or entry_tree[field]
end

function entry_tree:__newindex(field, value)
  self:put(field, value)
end

function entry_tree:has(field)
  return self.__indices:has(field)
end

function entry_tree:get(field)
  if self:has(field) then
    return self.__tree[self.__indices[field]]
  else
    return nil
  end
end

function entry_tree:put(field, subtree)
  assert(field ~= nil)
  assert(not self.__indices:has(field))
  assert(#self.__tree == #self.__fields)
  self.__tree:insert(subtree)
  self.__fields:insert(field)
  self.__indices[field] = #self.__tree
end

function entry_tree:replace(field, subtree)
  assert(field ~= nil)
  assert(self.__indices:has(field))
  assert(#self.__tree == #self.__fields)
  self.__tree[self.__indices[field]] = subtree
end

function entry_tree:next_item(k)
  if k == nil then
    if #self.__fields == 0 then
      return
    else
      return self.__fields[1], self.__tree[1]
    end
  end
  local i = self.__indices[k]
  if i == nil then
    return
  else
    return self.__fields[i + 1], self.__tree[i + 1]
  end
end

function entry_tree:items()
  if #self.__tree == 0 then
    return function() return nil, nil end
  else
    return entry_tree.next_item, self, nil
  end
end

function entry_tree:map_list(fn)
  local result = terralib.newlist()
  for k, v in self:items() do
    result:insert(fn(k, v))
  end
  return result
end

function entry_tree:__tostring()
  return "{" .. self:map_list(
    function(k, v)
      return tostring(k) .. "=" .. tostring(v)
    end):concat(",") .. "}"
end

function entry_tree.unify(tree1, tree2)
  if not entry_tree.is_entry_tree(tree1) or
     not entry_tree.is_entry_tree(tree2)
  then
    return nil
  end

  local result = entry_tree.new()
  for field, subtree in tree1:items() do
    result[field] = subtree
  end
  for field, subtree in tree2:items() do
    if result:has(field) then
      local u = entry_tree.unify(result[field], subtree)
      if u == nil then return nil, field end
      result[field] = u
    else
      result[field] = subtree
    end
  end
  return result
end

local function gather_field_types(node, type, fields)
  if not fields then
    return type, nil
  end

  if #fields == 1 then
    local field = fields[1]
    local subtree, suffix =
      gather_field_types(node, std.get_field(type, field.field_name), field.fields)

    if field.rename then
      local entry_tree = entry_tree.new()
      entry_tree[field.rename] = subtree
      return entry_tree, field.rename

    else
      return subtree, suffix or field.field_name
    end

  else -- #fields > 1
    assert(type:isstruct())

    local entry_tree = entry_tree.new()
    for idx, field in ipairs(fields) do
      local subtree, suffix =
        gather_field_types(node, std.get_field(type, field.field_name), field.fields)

      local key = nil
      if field.rename then
        key = field.rename
      else
        key = suffix or field.field_name
      end

      if entry_tree:has(key) then
        local result, colliding_field =
          entry_tree.unify(entry_tree[key], subtree)
        if result == nil then
          report.error(node, "field name " .. tostring(colliding_field or key) ..
              " collides in projection")
        end
        entry_tree:replace(key, result)
      else
        entry_tree[key] = subtree
      end
    end

    return entry_tree, nil
  end
end

local function convert_to_struct(entry_tree)
  local entries = terralib.newlist()
  local field_paths = terralib.newlist()

  for field, subtree in entry_tree:items() do
    local type = nil
    if entry_tree.is_entry_tree(subtree) then
      local field_type, suffixes = convert_to_struct(subtree)
      type = field_type
      field_paths:insertall(suffixes:map(
        function(suffix)
          return data.newtuple(field) .. suffix
        end))
    else
      type = subtree
      field_paths:insert(data.newtuple(field))
    end
    entries:insert({ field, type })
  end

  local result = terralib.types.newstruct("{" ..
    entries:map(function(entry)
      return entry[1] .. " : " .. tostring(entry[2])
    end):concat(", ") .."}")
  result.entries:insertall(entries)

  return result, field_paths
end

local function project_type(node, type, fields, field_paths)
  local entry_tree = gather_field_types(node, type, fields)

  if terralib.types.istype(entry_tree) then
    return entry_tree, terralib.newlist({ { field_paths[1], data.newtuple() } })
  else
    local subtype, subtype_field_paths = convert_to_struct(entry_tree)
    return subtype, data.zip(field_paths, subtype_field_paths)
  end
end

function type_check.project_field(cx, node, region, prefix_path, value_type)
  if node.rename then
    if type(node.rename) ~= "string" then
      report.error(node, "type mismatch: expected string for renaming but found " ..
          type(node.rename))
    end
  end

  if type(node.field_name) ~= "string" then
    report.error(node, "type mismatch: expected string for field name but found " ..
        type(node.field_name))
  end

  if value_type.__no_field_slicing then
    report.error(node, "type mismatch: projection onto " .. node.field_name ..
        " requires a field space type that permits field slicing but " ..
        tostring(value_type) .. " disallowed field slicing")
  end

  local field_path = prefix_path .. data.newtuple(node.field_name)
  local field_type = std.get_field(value_type, node.field_name)
  if not field_type then
    local region = pretty.entry_expr(region)
    report.error(node, "no field '" .. node.field_name ..
                "' in region " .. (data.newtuple(region) .. prefix_path):mkstring("."))
  end

  return type_check.project_fields(
    cx, node.fields, region, field_path, field_type)
end

function type_check.project_fields(cx, node, region, prefix_path, value_type)
  if not node then
    return terralib.newlist({prefix_path})
  end
  local result = terralib.newlist()
  for _, field in ipairs(node) do
    result:insertall(
      type_check.project_field(cx, field, region, prefix_path, value_type))
  end
  return result
end

function type_check.expr_projection(cx, node)
  if #node.fields == 0 then
    report.error(node, "projection needs at least one field path")
  end

  local region = type_check.expr(cx, node.region)
  local region_type = std.as_read(region.expr_type)

  if not std.is_region(region_type) then
    report.error(node.region, "type mismatch: expected region but got " .. tostring(region_type))
  end

  if region_type:is_projected() then
    report.error(node.region, "nested projection is not allowed")
  end

  local fs_type = region_type:fspace()

  if not (fs_type:isstruct() or std.is_fspace_instance(fs_type)) then
    report.error(node, "type mismatch: expected struct or fspace but got " .. tostring(fs_type))
  end

  local fields =
    type_check.project_fields(cx, node.fields, region, data.newtuple(), fs_type)
  local fs_subtype, field_mapping = project_type(node, fs_type, node.fields, fields)
  local region_subtype = std.region(region_type:ispace(), fs_subtype)
  region_subtype:set_projection_source(region_type)

  std.add_constraint(cx, region_subtype, region_type, std.subregion, false)

  local parent_region_type = region_type
  while parent_region_type ~= nil do
    std.copy_privileges(cx, parent_region_type, region_subtype, field_mapping)
    local child_region_type = parent_region_type
    parent_region_type = nil
    for region_type, _ in cx.region_universe:items() do
      if not std.type_eq(child_region_type, region_type) and
         std.check_constraint(cx,
          std.constraint(child_region_type, region_type, std.subregion))
      then
        parent_region_type = region_type
        -- We assume that there is only a single parent for any given region
        break
      end
    end
  end
  cx:intern_region(region_subtype)

  return ast.typed.expr.Projection {
    region = region,
    field_mapping = field_mapping,
    expr_type = region_subtype,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_parallelizer_constraint(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.as_read(lhs.expr_type)
  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.as_read(rhs.expr_type)
  if not std.is_partition(lhs_type) then
    report.error(lhs,
      "type mismatch in __parallelize_with: expected a partition or constraint on partitions but got " ..
      tostring(lhs_type))
  end
  if not std.is_partition(rhs_type) then
    report.error(rhs,
      "type mismatch in __parallelize_with: expected a partition or constraint on partitions but got " ..
      tostring(rhs_type))
  end
  return ast.typed.expr.ParallelizerConstraint {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
    annotations = node.annotations,
    span = node.span,
    expr_type = bool, -- type doesn't really matter
  }
end

local function unreachable(node)
  assert(false, "unreachable")
end

local type_check_expr_node = {
  [ast.specialized.expr.ID]                         = type_check.expr_id,
  [ast.specialized.expr.Constant]                   = type_check.expr_constant,
  [ast.specialized.expr.Global]                     = type_check.expr_global,
  [ast.specialized.expr.Function]                   = type_check.expr_function,
  [ast.specialized.expr.FieldAccess]                = type_check.expr_field_access,
  [ast.specialized.expr.IndexAccess]                = type_check.expr_index_access,
  [ast.specialized.expr.MethodCall]                 = type_check.expr_method_call,
  [ast.specialized.expr.Call]                       = type_check.expr_call,
  [ast.specialized.expr.Cast]                       = type_check.expr_cast,
  [ast.specialized.expr.Ctor]                       = type_check.expr_ctor,
  [ast.specialized.expr.RawContext]                 = type_check.expr_raw_context,
  [ast.specialized.expr.RawFields]                  = type_check.expr_raw_fields,
  [ast.specialized.expr.RawFuture]                  = type_check.expr_raw_future,
  [ast.specialized.expr.RawPhysical]                = type_check.expr_raw_physical,
  [ast.specialized.expr.RawRuntime]                 = type_check.expr_raw_runtime,
  [ast.specialized.expr.RawTask]                    = type_check.expr_raw_task,
  [ast.specialized.expr.RawValue]                   = type_check.expr_raw_value,
  [ast.specialized.expr.Isnull]                     = type_check.expr_isnull,
  [ast.specialized.expr.New]                        = type_check.expr_new,
  [ast.specialized.expr.Null]                       = type_check.expr_null,
  [ast.specialized.expr.DynamicCast]                = type_check.expr_dynamic_cast,
  [ast.specialized.expr.StaticCast]                 = type_check.expr_static_cast,
  [ast.specialized.expr.UnsafeCast]                 = type_check.expr_unsafe_cast,
  [ast.specialized.expr.Ispace]                     = type_check.expr_ispace,
  [ast.specialized.expr.Region]                     = type_check.expr_region,
  [ast.specialized.expr.Partition]                  = type_check.expr_partition,
  [ast.specialized.expr.PartitionEqual]             = type_check.expr_partition_equal,
  [ast.specialized.expr.PartitionByField]           = type_check.expr_partition_by_field,
  [ast.specialized.expr.PartitionByRestriction]     = type_check.expr_partition_by_restriction,

  [ast.specialized.expr.Image] = function(cx, node)
    if not node.region.fields and
       node.region.region:is(ast.specialized.expr.Function) then
      return type_check.expr_image_by_task(cx, node)
    else
      return type_check.expr_image(cx, node)
    end
  end,

  [ast.specialized.expr.Preimage]                   = type_check.expr_preimage,
  [ast.specialized.expr.CrossProduct]               = type_check.expr_cross_product,
  [ast.specialized.expr.CrossProductArray]          = type_check.expr_cross_product_array,
  [ast.specialized.expr.ListSlicePartition]         = type_check.expr_list_slice_partition,
  [ast.specialized.expr.ListDuplicatePartition]     = type_check.expr_list_duplicate_partition,
  [ast.specialized.expr.ListCrossProduct]           = type_check.expr_list_cross_product,
  [ast.specialized.expr.ListCrossProductComplete]   = type_check.expr_list_cross_product_complete,
  [ast.specialized.expr.ListPhaseBarriers]          = type_check.expr_list_phase_barriers,
  [ast.specialized.expr.ListInvert]                 = type_check.expr_list_invert,
  [ast.specialized.expr.ListRange]                  = type_check.expr_list_range,
  [ast.specialized.expr.ListIspace]                 = type_check.expr_list_ispace,
  [ast.specialized.expr.ListFromElement]            = type_check.expr_list_from_element,
  [ast.specialized.expr.PhaseBarrier]               = type_check.expr_phase_barrier,
  [ast.specialized.expr.DynamicCollective]          = type_check.expr_dynamic_collective,
  [ast.specialized.expr.DynamicCollectiveGetResult] = type_check.expr_dynamic_collective_get_result,
  [ast.specialized.expr.Advance]                    = type_check.expr_advance,
  [ast.specialized.expr.Adjust]                     = type_check.expr_adjust,
  [ast.specialized.expr.Arrive]                     = type_check.expr_arrive,
  [ast.specialized.expr.Await]                      = type_check.expr_await,
  [ast.specialized.expr.Copy]                       = type_check.expr_copy,
  [ast.specialized.expr.Fill]                       = type_check.expr_fill,
  [ast.specialized.expr.Acquire]                    = type_check.expr_acquire,
  [ast.specialized.expr.Release]                    = type_check.expr_release,
  [ast.specialized.expr.AttachHDF5]                 = type_check.expr_attach_hdf5,
  [ast.specialized.expr.DetachHDF5]                 = type_check.expr_detach_hdf5,
  [ast.specialized.expr.AllocateScratchFields]      = type_check.expr_allocate_scratch_fields,
  [ast.specialized.expr.WithScratchFields]          = type_check.expr_with_scratch_fields,
  [ast.specialized.expr.Unary]                      = type_check.expr_unary,
  [ast.specialized.expr.Binary]                     = type_check.expr_binary,
  [ast.specialized.expr.Deref]                      = type_check.expr_deref,
  [ast.specialized.expr.AddressOf]                  = type_check.expr_address_of,
  [ast.specialized.expr.ImportIspace]               = type_check.expr_import_ispace,
  [ast.specialized.expr.ImportRegion]               = type_check.expr_import_region,
  [ast.specialized.expr.ImportPartition]            = type_check.expr_import_partition,
  [ast.specialized.expr.ImportCrossProduct]         = type_check.expr_import_cross_product,
  [ast.specialized.expr.Projection]                 = type_check.expr_projection,

  [ast.specialized.expr.LuaTable] = function(cx, node)
    report.error(node, "unable to specialize value of type table")
  end,

  [ast.specialized.expr.CtorListField] = unreachable,
  [ast.specialized.expr.CtorRecField]  = unreachable,
  [ast.specialized.expr.RegionRoot]    = unreachable,
  [ast.specialized.expr.Condition]     = unreachable,
}

local type_check_expr = ast.make_single_dispatch(
  type_check_expr_node,
  {ast.specialized.expr})

function type_check.expr(cx, node)
  return type_check_expr(cx)(node)
end

function type_check.block(cx, node)
  return ast.typed.Block {
    stats = node.stats:map(
      function(stat) return type_check.stat(cx, stat) end),
    span = node.span,
  }
end

function type_check.stat_if(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond)
  if not std.validate_implicit_cast(cond_type, bool) then
    report.error(node.cond, "type mismatch: expected " .. tostring(bool) .. " but got " .. tostring(cond_type))
  end
  cond = insert_implicit_cast(cond, cond_type, bool)

  local then_cx = cx:new_local_scope()
  local else_cx = cx:new_local_scope()
  return ast.typed.stat.If {
    cond = cond,
    then_block = type_check.block(then_cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return type_check.stat_elseif(cx, block) end),
    else_block = type_check.block(else_cx, node.else_block),
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_elseif(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond)
  if not std.validate_implicit_cast(cond_type, bool) then
    report.error(node.cond, "type mismatch: expected " .. tostring(bool) .. " but got " .. tostring(cond_type))
  end
  cond = insert_implicit_cast(cond, cond_type, bool)

  local body_cx = cx:new_local_scope()
  return ast.typed.stat.Elseif {
    cond = cond,
    block = type_check.block(body_cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_while(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond)
  if not std.validate_implicit_cast(cond_type, bool) then
    report.error(node.cond, "type mismatch: expected " .. tostring(bool) .. " but got " .. tostring(cond_type))
  end
  cond = insert_implicit_cast(cond, cond_type, bool)

  local body_cx = cx:new_local_scope(nil, true)
  return ast.typed.stat.While {
    cond = cond,
    block = type_check.block(body_cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return type_check.expr(cx, value) end)
  local value_types = values:map(
    function(value) return std.check_read(cx, value) end)

  for _, value_type in ipairs(value_types) do
    if not value_type:isintegral() then
      report.error(node, "numeric for loop expected integral type, got " .. tostring(value_type))
    end
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type = node.symbol:hastype() or value_types[1]
  if value_types[3] then
    var_type = binary_op_type("+")(cx, node, var_type, value_types[3])
  end
  if not var_type:isintegral() then
    report.error(node, "numeric for loop expected integral type, got " .. tostring(var_type))
  end
  if not node.symbol:hastype() then
    node.symbol:settype(var_type)
  end
  assert(std.type_eq(var_type, node.symbol:gettype()))
  cx.type_env:insert(node, node.symbol, var_type)

  -- Enter scope for body.
  local cx = cx:new_local_scope(nil, true)
  return ast.typed.stat.ForNum {
    symbol = node.symbol,
    values = values,
    block = type_check.block(cx, node.block),
    metadata = false,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_for_list(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if not (std.is_ispace(value_type) or std.is_region(value_type) or
          std.is_rect_type(value_type) or
            (std.is_list(value_type) and not value_type:is_list_of_regions()))
  then
    report.error(node, "iterator for loop expected ispace, region, rect or list, got " ..
                tostring(value_type))
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()

  -- Hack: Try to recover the original symbol for this bound if possible
  local bound
  if value:is(ast.typed.expr.ID) then
    bound = value.value
  else
    bound = std.newsymbol(value_type)
  end

  local expected_var_type
  if std.is_ispace(value_type) then
    local index_type = value_type.index_type
    expected_var_type = index_type(bound)
  elseif std.is_region(value_type) then
    local index_type = value_type:ispace().index_type
    expected_var_type = index_type(value_type:fspace(), bound)
  elseif std.is_rect_type(value_type) then
    expected_var_type = value_type.index_type
  elseif std.is_list(value_type) then
    expected_var_type = value_type.element_type
  else
    assert(false)
  end

  local var_type = node.symbol:hastype()
  if not var_type then
    var_type = expected_var_type
  end

  if not std.type_eq(expected_var_type, var_type) then
    report.error(node, "iterator for loop expected symbol of type " ..
                tostring(expected_var_type) .. ", got " .. tostring(var_type))
  end

  -- Hack: Stuff the type back into the symbol so it's available
  -- to ptr types if necessary.
  if not node.symbol:hastype() then
    node.symbol:settype(var_type)
  end
  assert(std.type_eq(var_type, node.symbol:gettype()))
  cx.type_env:insert(node, node.symbol, var_type)

  -- Enter scope for body.
  local cx = cx:new_local_scope(nil, true)
  return ast.typed.stat.ForList {
    symbol = node.symbol,
    value = value,
    block = type_check.block(cx, node.block),
    metadata = false,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_repeat(cx, node)
  local block_cx = cx:new_local_scope(nil, true)
  local block = type_check.block(cx, node.block)

  local until_cond = type_check.expr(block_cx, node.until_cond)
  local until_cond_type = std.check_read(block_cx, until_cond)
  if not std.validate_implicit_cast(until_cond_type, bool) then
    report.error(node.until_cond, "type mismatch: expected " .. tostring(bool) .. " but got " .. tostring(until_cond_type))
  end
  until_cond = insert_implicit_cast(until_cond, until_cond_type, bool)

  return ast.typed.stat.Repeat {
    block = block,
    until_cond = until_cond,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_must_epoch(cx, node)
  if cx.must_epoch then
    report.error(node, "nested must epochs are not supported")
  end

  local cx = cx:new_local_scope(true)
  return ast.typed.stat.MustEpoch {
    block = type_check.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return ast.typed.stat.Block {
    block = type_check.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_var(cx, node)
  local value = node.values and type_check.expr(cx, node.values) or false
  local value_type = value and std.check_read(cx, value) or nil

  local symbol = node.symbols
  local var_type = symbol:hastype()

  if var_type then
    if value and not std.validate_implicit_cast(value_type, var_type) then
      report.error(node, "type mismatch in var: expected " .. tostring(var_type) .. " but got " .. tostring(value_type))
    end
    if not value and std.type_supports_constraints(var_type) then
      report.error(node, "variable of type " .. tostring(var_type) .. " must be initialized")
    end
    if std.is_bounded_type(var_type) then
      std.check_bounds(node, var_type)
    end
  else
    if not value then
      report.error(node, "type must be specified for uninitialized variables")
    end
    var_type = value_type

    -- Hack: Stuff the type back into the symbol so it's available
    -- to ptr types if necessary.
    symbol:settype(var_type)
  end
  cx.type_env:insert(node, symbol, std.rawref(&var_type))

  value = value and insert_implicit_cast(value, value_type, symbol:gettype()) or false

  return ast.typed.stat.Var {
    symbol = symbol,
    type = var_type,
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_var_unpack(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if not (value_type:isstruct() or std.is_fspace_instance(value_type)) then
    report.error(node, "destructuring var expected struct or fspace, got " .. tostring(value_type))
  end

  local unpack_type, constraints = value_type
  local mapping = {}
  if std.is_fspace_instance(value_type) then
    for i, symbol in ipairs(node.symbols) do
      local field = node.fields[i]
      if not mapping[field] then
        mapping[field] = symbol
      end
    end
    unpack_type, constraints = std.unpack_fields(value_type, mapping)
  elseif value_type:isstruct() then
    -- Ok
  else
    assert(false)
  end
  local entries = unpack_type:getentries()

  local index = {}
  for i, entry in ipairs(entries) do
    index[entry[1] or entry.field] = entry[2] or entry.type
  end

  local field_types = terralib.newlist()
  for i, symbol in ipairs(node.symbols) do
    local field = node.fields[i]
    if mapping[field] then
      field = mapping[field]:getname()
    end
    local field_type = index[field]
    if not field_type then
      report.error(node, "no field '" .. tostring(field) .. "' in type " .. tostring(value_type))
    end
    if not symbol:hastype(field_type) then
      symbol:settype(field_type)
    end
    assert(symbol:gettype() == field_type)
    cx.type_env:insert(node, symbol, std.rawref(&field_type))
    field_types:insert(field_type)
  end

  if constraints then
    std.add_constraints(cx, constraints)
  end

  return ast.typed.stat.VarUnpack {
    symbols = node.symbols,
    fields = node.fields,
    field_types = field_types,
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_return(cx, node)
  local value = node.value and type_check.expr(cx, node.value)
  local value_type
  if value then
    value_type = std.check_read(cx, value)
  else
    value_type = terralib.types.unit
  end

  local expected_type = cx:get_return_type()
  assert(expected_type)
  if std.type_eq(expected_type, std.untyped) then
    cx:set_return_type(value_type)
  else
    local result_type = std.type_meet(value_type, expected_type)
    if not result_type then
      report.error(node, "type mismatch in return: expected " .. tostring(expected_type) .. " but got " .. tostring(value_type))
    end
    cx:set_return_type(result_type)
  end

  return ast.typed.stat.Return {
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_break(cx, node)
  if not cx.breakable_loop then
    report.error(node, "break must be inside a loop")
  end
  return ast.typed.stat.Break {
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_assignment(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_write(cx, lhs)

  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)

  if not std.validate_implicit_cast(rhs_type, lhs_type) then
    report.error(node, "type mismatch in assignment: expected " .. tostring(lhs_type) .. " but got " .. tostring(rhs_type))
  end

  rhs = insert_implicit_cast(rhs, rhs_type, lhs_type)

  return ast.typed.stat.Assignment {
    lhs = lhs,
    rhs = rhs,
    metadata = false,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_reduce(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_reduce(cx, node.op, lhs)

  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs)

  local expr_type = binary_ops[node.op](cx, node, lhs_type, rhs_type)
  if not std.validate_explicit_cast(expr_type, lhs_type) then
    report.error(node, "type mismatch between " .. tostring(expr_type) .. " and " .. tostring(lhs_type))
  end

  return ast.typed.stat.Reduce {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
    metadata = false,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_expr(cx, node)
  local value = type_check.expr(cx, node.expr)
  local value_type = std.check_read(cx, value)

  return ast.typed.stat.Expr {
    expr = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_raw_delete(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if not (std.is_region(value_type) or std.is_partition(value_type)) then
    report.error(node, "type mismatch in delete: expected a region or partition but got " .. tostring(value_type))
  end

  return ast.typed.stat.RawDelete {
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_fence(cx, node)
  return ast.typed.stat.Fence {
    kind = node.kind,
    blocking = node.blocking,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_parallelize_with(cx, node)
  local hints = node.hints:map(function(expr)
    if expr:is(ast.specialized.expr.ID) then
      local value = type_check.expr_id(cx, expr)
      local value_type = std.check_read(cx, value)
      if not (std.is_partition(value_type) or std.is_ispace(value_type)) then
        report.error(node,
          "type mismatch in __parallelize_with: expected a partition, index space, or constraint on partitions but got " ..
          tostring(value_type))
      end
      return value
    elseif expr:is(ast.specialized.expr.Binary) then
      return type_check.expr_parallelizer_constraint(cx, expr)
    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end
  end)

  return ast.typed.stat.ParallelizeWith {
    hints = hints,
    block = type_check.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

local supported_parallel_prefix_ops = {
  ["+"] = true,
  ["*"] = true,
  ["min"] = true,
  ["max"] = true,
}

function type_check.stat_parallel_prefix(cx, node)
  local lhs = type_check.expr_region_root(cx, node.lhs)
  local rhs = type_check.expr_region_root(cx, node.rhs)
  local op = node.op
  local dir = type_check.expr(cx, node.dir)

  local args = terralib.newlist({lhs, rhs})
  local prev_field_type = nil
  local prev_ispace_type = nil
  for i = 1, #args do
    local arg = args[i]
    assert(std.is_region(arg.expr_type))
    local prefix = "type mismatch in argument " .. tostring(i) .. ": "
    if #arg.fields > 1 then
      report.error(arg, prefix .. "expected one or no field path, but got " .. tostring(#arg.fields))
    end

    local field_type = arg.expr_type:fspace()
    if #arg.fields > 0 then
      field_type = std.get_field_path(field_type, arg.fields[1])
    end
    if not field_type:isprimitive() then
      report.error(arg, prefix .. "expected a primitive type, but got " .. tostring(field_type))
    end
    if prev_field_type ~= nil and not std.type_eq(prev_field_type, field_type) then
      report.error(arg,
          prefix .. "expected " .. tostring(prev_field_type) .. ", but got " .. tostring(field_type))
    end
    prev_field_type = field_type

    local ispace_type = arg.expr_type:ispace()
    -- TODO: Need to extend the parallel prefix operator implementation to multi-dimensional regions
    if ispace_type.dim ~= 1 then
      report.error(arg,
          prefix .. "expected a region of " .. tostring(ispace(int1d)) ..
          ", but got a region of " .. tostring(ispace_type))
    end
    prev_ispace_type = ispace_type
  end

  if not supported_parallel_prefix_ops[node.op] then
    report.error(node,
        "type mismatch in argument 3: operator " .. node.op .. " is not a parallel prefix operator")
  end

  if not std.as_read(dir.expr_type):isintegral() then
    report.error(node.dir,
        "type mismatch in argument 4: expected an integer type, but got " .. tostring(dir.expr_type))
  end
  local lhs_field_path = data.newtuple()
  if #lhs.fields > 0 then lhs_field_path = lhs.fields[1] end
  if not std.check_privilege(cx, std.writes, lhs.expr_type, lhs_field_path) then
    report.error(lhs,
        "invalid privilege in argument 1: " .. tostring(std.writes) .. "(" ..
        (data.newtuple(lhs.region.value) .. lhs_field_path):mkstring(".") .. ")")
  end
  local rhs_field_path = data.newtuple()
  if #rhs.fields > 0 then rhs_field_path = rhs.fields[1] end
  if not std.check_privilege(cx, std.reads, rhs.expr_type, rhs_field_path) then
    report.error(rhs,
        "invalid privilege in argument 2: " .. tostring(std.reads) .. "(" ..
        (data.newtuple(rhs.region.value) .. rhs_field_path):mkstring(".") .. ")")
  end

  return ast.typed.stat.ParallelPrefix {
    lhs = lhs,
    rhs = rhs,
    op = op,
    dir = dir,
    annotations = node.annotations,
    span = node.span,
  }
end

local type_check_stat_node = {
  [ast.specialized.stat.If]              = type_check.stat_if,
  [ast.specialized.stat.While]           = type_check.stat_while,
  [ast.specialized.stat.ForNum]          = type_check.stat_for_num,
  [ast.specialized.stat.ForList]         = type_check.stat_for_list,
  [ast.specialized.stat.Repeat]          = type_check.stat_repeat,
  [ast.specialized.stat.MustEpoch]       = type_check.stat_must_epoch,
  [ast.specialized.stat.Block]           = type_check.stat_block,
  [ast.specialized.stat.Var]             = type_check.stat_var,
  [ast.specialized.stat.VarUnpack]       = type_check.stat_var_unpack,
  [ast.specialized.stat.Return]          = type_check.stat_return,
  [ast.specialized.stat.Break]           = type_check.stat_break,
  [ast.specialized.stat.Assignment]      = type_check.stat_assignment,
  [ast.specialized.stat.Reduce]          = type_check.stat_reduce,
  [ast.specialized.stat.Expr]            = type_check.stat_expr,
  [ast.specialized.stat.RawDelete]       = type_check.stat_raw_delete,
  [ast.specialized.stat.Fence]           = type_check.stat_fence,
  [ast.specialized.stat.ParallelizeWith] = type_check.stat_parallelize_with,
  [ast.specialized.stat.ParallelPrefix]  = type_check.stat_parallel_prefix,

  [ast.specialized.stat.Elseif] = unreachable,
}

local type_check_stat = ast.make_single_dispatch(
  type_check_stat_node,
  {ast.specialized.stat})

function type_check.stat(cx, node)
  return type_check_stat(cx)(node)
end

local opaque_types = {
  [std.c.legion_domain_point_iterator_t]       = true,
  [std.c.legion_coloring_t]                    = true,
  [std.c.legion_domain_coloring_t]             = true,
  [std.c.legion_point_coloring_t]              = true,
  [std.c.legion_domain_point_coloring_t]       = true,
  [std.c.legion_multi_domain_point_coloring_t] = true,
  [std.c.legion_index_space_allocator_t]       = true,
  [std.c.legion_field_allocator_t]             = true,
  [std.c.legion_argument_map_t]                = true,
  [std.c.legion_predicate_t]                   = true,
  [std.c.legion_future_t]                      = true,
  [std.c.legion_future_map_t]                  = true,
  [std.c.legion_task_launcher_t]               = true,
  [std.c.legion_index_launcher_t]              = true,
  [std.c.legion_inline_launcher_t]             = true,
  [std.c.legion_copy_launcher_t]               = true,
  [std.c.legion_index_copy_launcher_t]         = true,
  [std.c.legion_acquire_launcher_t]            = true,
  [std.c.legion_release_launcher_t]            = true,
  [std.c.legion_attach_launcher_t]             = true,
  [std.c.legion_must_epoch_launcher_t]         = true,
  [std.c.legion_physical_region_t]             = true,
  [std.c.legion_task_t]                        = true,
  [std.c.legion_inline_t]                      = true,
  [std.c.legion_mappable_t]                    = true,
  [std.c.legion_region_requirement_t]          = true,
  [std.c.legion_machine_t]                     = true,
  [std.c.legion_mapper_t]                      = true,
  [std.c.legion_default_mapper_t]              = true,
  [std.c.legion_processor_query_t]             = true,
  [std.c.legion_memory_query_t]                = true,
  [std.c.legion_machine_query_interface_t]     = true,
  [std.c.legion_execution_constraint_set_t]    = true,
  [std.c.legion_layout_constraint_set_t]       = true,
  [std.c.legion_task_layout_constraint_set_t]  = true,
  [std.c.legion_physical_instance_t]           = true,
  [std.c.legion_field_map_t]                   = true,
}

do
  for d = 1, std.max_dim do
    local dim = tostring(d)
    opaque_types[ std.c["legion_rect_in_domain_iterator_" .. dim .. "d_t"] ] = true
    opaque_types[ std.c["legion_deferred_buffer_char_" .. dim .. "d_t"] ]    = true
    opaque_types[ std.c["legion_accessor_array_" .. dim .. "d_t"] ]          = true
  end
end

local function is_param_type_dangerous(param_type)
  return opaque_types[param_type] or param_type:ispointer()
end

local function is_param_type_inadmissible(param_type)
  return param_type == std.c.legion_runtime_t or
         param_type == std.c.legion_context_t
end

function type_check.top_task_param(cx, node, task, mapping, is_defined)
  local param_type = node.symbol:gettype()
  cx.type_env:insert(node, node.symbol, std.rawref(&param_type))

  -- Check for parameters with duplicate types.
  if std.type_supports_constraints(param_type) then
    if mapping[param_type] then
      report.error(node, "parameters " .. tostring(node.symbol) .. " and " ..
                  tostring(mapping[param_type]) ..
                  " have the same type, but are required to be distinct")
    end
    mapping[param_type] = node.symbol
  end

  -- Check for parameters with inadmissible types.
  if is_param_type_inadmissible(param_type) then
    report.error(node, "parameter " .. tostring(node.symbol) .. " has inadmissible type " ..
                tostring(param_type))
  end

  -- Warn for parameters with dangerous types used unless the task is an inline task.
  if is_param_type_dangerous(param_type) and
     not (std.config["inline"] and task.annotations.inline:is(ast.annotation.Demand))
  then
    report.warn(node, "WARNING: parameter " .. tostring(node.symbol) .. " has raw pointer type " ..
                tostring(param_type) .. ". please consider replacing it with a non-pointer type.")
  end

  -- Check for use of futures in a defined task.
  if node.future and is_defined then
    report.error(node, "futures may be used as parameters only when a task is defined externally")
  end

  return ast.typed.top.TaskParam {
    symbol = node.symbol,
    param_type = param_type,
    future = node.future,
    annotations = node.annotations,
    span = node.span,
  }
end

local function privilege_string(privilege)
  return tostring(privilege.privilege) .. "(" ..
    terralib.newlist({tostring(privilege.region), unpack(privilege.field_path)}):concat(".") .. ")"
end

function type_check.top_task(cx, node)
  local return_type = node.return_type
  local is_cuda = node.annotations.cuda:is(ast.annotation.Demand)
  local cx = cx:new_task_scope(return_type, is_cuda)

  local is_defined = node.prototype:has_primary_variant()

  local mapping = {}
  local params = node.params:map(
    function(param)
      return type_check.top_task_param(cx, param, node, mapping, is_defined)
    end)
  local prototype = node.prototype
  prototype:set_param_symbols(
    params:map(function(param) return param.symbol end))

  local param_types = params:map(function(param) return param.param_type end)
  local task_type = terralib.types.functype(param_types, return_type, false)
  prototype:set_type(task_type)

  local param_type_set = data.set(param_types)
  local privileges = type_check.privileges(cx, node.privileges)
  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      local privilege_type = privilege.privilege
      local region = privilege.region
      local region_type = region:gettype()
      local field_path = privilege.field_path
      assert(std.type_supports_privileges(region_type))
      if not param_type_set[region_type] then
        report.error(node, "requested " .. privilege_string(privilege) ..
          " but " .. tostring(region) .. " is not a parameter")
      end
      if std.is_reduce(privilege_type) then
        local _, field_types =
          std.flatten_struct_fields(std.get_field_path(region_type:fspace(), field_path))
        field_types:map(function(field_type)
          if field_type:isprimitive() or
             (std.is_complex_type(field_type) and
              field_type:support_reduction(privilege_type.op)) or
             (field_type:isarray() and field_type.type:isprimitive())
          then
            std.update_reduction_op(privilege_type.op, field_type)
          else
            report.error(node, "invalid field type for " .. privilege_string(privilege) ..
              ": " .. tostring(field_type))
          end
        end)
      end
      std.add_privilege(cx, privilege_type, region_type, field_path)
      cx:intern_region(region_type)
    end
  end
  prototype:set_privileges(privileges)

  local coherence_modes = type_check.coherence_modes(cx, node.coherence_modes, param_type_set)
  prototype:set_coherence_modes(coherence_modes)

  local flags = type_check.flags(cx, node.flags)
  prototype:set_flags(flags)

  local conditions = type_check.conditions(cx, node.conditions, params)
  prototype:set_conditions(conditions)

  local constraints = type_check.constraints(cx, node.constraints)
  std.add_constraints(cx, constraints)
  prototype:set_param_constraints(constraints)

  local body = node.body and type_check.block(cx, node.body)

  return_type = cx:get_return_type()
  if std.type_eq(return_type, std.untyped) then
    return_type = terralib.types.unit
  end
  task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:set_type(task_type, true)

  for _, fixup_node in ipairs(cx.fixup_nodes) do
    if fixup_node:is(ast.typed.expr.Call) then
      local fn_type = fixup_node.fn.value:get_type()
      assert(fn_type.returntype ~= untyped)
      fixup_node.expr_type = fn_type.returntype
    else
      assert(false)
    end
  end

  prototype:set_constraints(cx.constraints)
  prototype:set_region_universe(cx.region_universe)

  return ast.typed.top.Task {
    name = node.name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    flags = flags,
    conditions = conditions,
    constraints = constraints,
    body = body,
    config_options = ast.TaskConfigOptions {
      leaf = false,
      inner = false,
      idempotent = false,
      replicable = false,
    },
    region_usage = false,
    region_divergence = false,
    metadata = false,
    prototype = prototype,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.top_fspace(cx, node)
  node.fspace.constraints = type_check.constraints(cx, node.constraints)
  return ast.typed.top.Fspace {
    name = node.name,
    fspace = node.fspace,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.top_quote_expr(cx, node)
  -- Type check lazily, when the expression is interpolated.
  return node
end

function type_check.top_quote_stat(cx, node)
  -- Type check lazily, when the statement is interpolated.
  return node
end

function type_check.top(cx, node)
  if node:is(ast.specialized.top.Task) then
    return type_check.top_task(cx, node)

  elseif node:is(ast.specialized.top.Fspace) then
    return type_check.top_fspace(cx, node)

  elseif node:is(ast.specialized.top.QuoteExpr) then
    return type_check.top_quote_expr(cx, node)

  elseif node:is(ast.specialized.top.QuoteStat) then
    return type_check.top_quote_stat(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.entry(node)
  local cx = context.new_global_scope({})
  return type_check.top(cx, node)
end

return type_check
