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

-- Legion Type Checker

local affine_helper = require("regent/affine_helper")
local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
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
    fixup_nodes = self.fixup_nodes,
    must_epoch = must_epoch,
    breakable_loop = breakable_loop,
    external = self.external,
  }
  setmetatable(cx, context)
  return cx
end

function context:new_task_scope(expected_return_type)
  local cx = {
    type_env = self.type_env:new_local_scope(),
    privileges = data.newmap(),
    constraints = data.new_recursive_map(2),
    region_universe = data.newmap(),
    expected_return_type = {expected_return_type},
    fixup_nodes = terralib.newlist(),
    must_epoch = false,
    breakable_loop = false,
    external = false,
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

function context:get_external()
  return self.external
end

function context:set_external(external)
  self.external = external
end

function type_check.region_field(cx, node, region, prefix_path, value_type)
  assert(std.is_symbol(region))
  local field_path = prefix_path .. data.newtuple(node.field_name)
  local field_type = std.get_field(value_type, node.field_name)
  if not field_type then
    report.error(node, "no field '" .. node.field_name ..
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
  local region_type = region:gettype()
  if not (std.type_supports_privileges(region_type)) then
    report.error(node, "type mismatch: expected a region but got " .. tostring(region_type))
  end
  return region
end

function type_check.region_root(cx, node)
  local region = type_check.region_bare(cx, node)
  local region_type = region:gettype()
  local value_type = region_type:fspace()
  return {
    region = region,
    fields = type_check.region_fields(
      cx, node.fields, region, data.newtuple(), value_type),
  }
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

function type_check.coherence(cx, node, result)
  local coherence_modes = type_check.coherence_kinds(cx, node.coherence_modes)
  local region_fields = type_check.regions(cx, node.regions)

  for _, coherence in ipairs(coherence_modes) do
    for _, region_field in ipairs(region_fields) do
      local region = region_field.region
      local region_type = region:gettype()
      assert(std.type_supports_privileges(region_type))

      local fields = region_field.fields
      for _, field in ipairs(fields) do
        check_coherence_conflict(node, region, field, coherence, result)
        result[region_type][field] = coherence
      end
    end
  end
end

function type_check.coherence_modes(cx, node)
  local result = data.new_recursive_map(1)
  for _, coherence in ipairs(node) do
    type_check.coherence(cx, coherence, result)
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

  local result = {}
  result[std.arrives] = {}
  result[std.awaits] = {}

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
      if std.is_ref(unpack_type) then
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
  elseif std.is_region(std.as_read(unpack_type)) and node.field_name == "bounds" then
    local index_type = std.as_read(unpack_type):ispace().index_type
    if index_type:is_opaque() then
      report.error(node, "no field '" .. node.field_name .. "' in type " ..
                  tostring(std.as_read(unpack_type)))
    end
    field_type = std.rect_type(index_type)
  elseif std.is_partition(std.as_read(unpack_type)) and node.field_name == "colors" then
    field_type = std.as_read(unpack_type):colors()
    if field_type:is_opaque() then
      report.error(node, "no field '" .. node.field_name .. "' in type " ..
                  tostring(std.as_read(unpack_type)))
    end
  elseif std.type_is_opaque_to_field_accesses(std.as_read(unpack_type)) then
    report.error(node, "no field '" .. node.field_name .. "' in type " ..
                tostring(std.as_read(value_type)))
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
  local value_type = std.check_read(cx, value)
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
      region_symbol = terralib.newsymbol(value_type)
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
  else
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local function test()
      local terra query(a : value_type, i : index_type)
        return a[i]
      end
      return query:gettype().returntype
    end
    local valid, result_type = pcall(test)

    if not valid then
      report.error(node, "invalid index access for " .. tostring(value_type) .. " and " .. tostring(index_type))
    end

    -- Hack: Fix up the type to be a reference if the original was.
    if std.is_ref(value.expr_type) then
      result_type = std.rawref(&result_type)
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
    for idx, arg_type in pairs(arg_types) do
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
    node, param_symbols, arg_symbols, def_type.isvararg, def_type.returntype, {}, false)

  if std.is_task(fn.value) then
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

  return ast.typed.expr.Cast {
    fn = fn,
    arg = arg,
    expr_type = to_type,
    annotations = node.annotations,
    span = node.span,
  }
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

local function extract_field_path(node)
  if node:is(ast.specialized.expr.ID) then
    return node, data.newtuple()
  elseif node:is(ast.specialized.expr.FieldAccess) then
    local base, path = extract_field_path(node.value)
    return base, path .. data.newtuple(node.field_name)
  else
    report.error(node, "unexpected type of expression in raw operator")
  end
end

function type_check.expr_raw_fields(cx, node)
  local base, region_field_path = extract_field_path(node.region)
  local region = type_check.expr(cx, base)
  local region_type = std.check_read(cx, region)

  local field_paths, _ = std.flatten_struct_fields(region_type:fspace())
  local privilege_fields = terralib.newlist()
  for _, field_path in ipairs(field_paths) do
    if std.check_any_privilege(cx, region_type, field_path) and
       field_path:starts_with(region_field_path)
    then
      privilege_fields:insert(field_path)
    end
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

function type_check.expr_raw_physical(cx, node)
  local base, region_field_path = extract_field_path(node.region)
  local region = type_check.expr(cx, base)
  local region_type = std.check_read(cx, region)

  local field_paths, _ = std.flatten_struct_fields(region_type:fspace())
  local privilege_fields = terralib.newlist()
  for _, field_path in ipairs(field_paths) do
    if std.check_any_privilege(cx, region_type, field_path) and
       field_path:starts_with(region_field_path)
    then
      privilege_fields:insert(field_path)
    end
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
  if not std.is_bounded_type(pointer_type) then
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
  if not std.is_bounded_type(pointer_type) then
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

  if not (std.is_bounded_type(node.expr_type) or std.is_partition(node.expr_type)) then
    report.error(node, "dynamic_cast requires ptr type or partition type as argument 1, got " .. tostring(node.expr_type))
  end
  if not (std.validate_implicit_cast(value_type, node.expr_type.index_type) or
          std.is_partition(value_type)) then
    report.error(node, "dynamic_cast requires ptr or partition as argument 2, got " .. tostring(value_type))
  end
  if std.is_bounded_type(value_type) and not std.type_eq(node.expr_type.points_to_type, value_type.points_to_type) then
    report.error(node, "incompatible pointers for dynamic_cast: " .. tostring(node.expr_type) .. " and " .. tostring(value_type))
  end
  if std.is_partition(value_type) and
     not (std.type_eq(node.expr_type:colors(), value_type:colors()) and
          node.expr_type.parent_region_symbol == value_type.parent_region_symbol) then
    report.error(node, "incompatible partitions for dynamic_cast: " .. tostring(node.expr_type) .. " and " .. tostring(value_type))
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

  if not std.is_bounded_type(expr_type) then
    report.error(node, "static_cast requires ptr type as argument 1, got " .. tostring(expr_type))
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
  if #expr_type:bounds() ~= 1 then
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
  if ispace:is(ast.specialized.expr.ID) then
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
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region)
  local coloring = type_check.expr(cx, node.coloring)
  local coloring_type = std.check_read(cx, coloring)
  local colors = node.colors and type_check.expr(cx, node.colors)
  local colors_type = colors and std.check_read(cx, colors)

  -- Note: This test can't fail because disjointness is tested in the parser.
  assert(disjointness:is(ast.disjointness_kind))

  if not std.is_region(region_type) then
    report.error(node, "type mismatch in argument 2: expected region but got " ..
                tostring(region_type))
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
  local expr_type = std.partition(disjointness, region_symbol, colors_symbol)

  -- Hack: Stuff the region type back into the partition's region
  -- argument, if necessary.
  if not expr_type.parent_region_symbol:hastype() then
    expr_type.parent_region_symbol:settype(region_type)
  end
  assert(expr_type.parent_region_symbol:gettype() == region_type)

  return ast.typed.expr.Partition {
    disjointness = disjointness,
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

  if not std.is_ispace(colors_type) then
    report.error(node, "type mismatch in argument 2: expected ispace but got " ..
                tostring(colors_type))
  end
  if region_type:ispace().dim > 1 and data.max(colors_type.dim, 1) ~= data.max(region_type:ispace().dim, 1) then
    report.error(node, "type mismatch in argument 2: expected ispace with " ..
                tostring(region_type:ispace().dim) .. " dimensions but got " ..
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
  local region = type_check.expr_region_root(cx, node.region)
  local region_type = std.check_read(cx, region)

  local colors = type_check.expr(cx, node.colors)
  local colors_type = std.check_read(cx, colors)

  if #region.fields ~= 1 then
    report.error(node, "type mismatch in argument 1: expected 1 field but got " ..
                tostring(#region.fields))
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
  local expr_type = std.partition(std.disjoint, region_symbol, colors_symbol)

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
    region = region,
    colors = colors,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_image(cx, node)
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
        for _, entry in pairs(ty:getentries()) do
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
  local expr_type = std.partition(std.aliased, parent_symbol, partition_type.colors_symbol)

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
    parent = parent,
    partition = partition,
    region = region,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_image_by_task(cx, node)
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
  local expr_type = std.partition(std.aliased, parent_symbol, partition_type.colors_symbol)

  return ast.typed.expr.ImageByTask {
    parent = parent,
    partition = partition,
    task = task,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.expr_preimage(cx, node)
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
        for _, entry in pairs(ty:getentries()) do
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
  local disjointness = partition_type.disjointness
  if std.is_rect_type(field_type) then
    disjointness = std.aliased
  end
  local expr_type = std.partition(disjointness, parent_symbol, partition_type.colors_symbol)

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
    if not std.check_privilege(cx, std.reads, dst_type, field_path) then
      local dst_symbol
      if node.dst.region:is(ast.specialized.expr.ID) then
        dst_symbol = node.dst.region.value
      else
        dst_symbol = sdt.newsymbol()
      end
      report.error(
        node, "invalid privileges in fill: " .. tostring(std.reads) ..
          "(" .. (data.newtuple(dst_symbol) .. field_path):mkstring(".") .. ")")
    end
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
      report.error(node, "invalid argument to unary operator " .. tostring(rhs_type))
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
      report.error(node, "type mismatch between " .. tostring(lhs_type) ..
                  " and " .. tostring(rhs_type))
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

  if not (value_type:ispointer() or std.is_bounded_type(value_type)) then
    report.error(node, "dereference of non-pointer type " .. tostring(value_type))
  end

  if cx:get_external() then
    report.error(node, "dereference in an external task")
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

function type_check.expr(cx, node)
  if node:is(ast.specialized.expr.ID) then
    return type_check.expr_id(cx, node)

  elseif node:is(ast.specialized.expr.Constant) then
    return type_check.expr_constant(cx, node)

  elseif node:is(ast.specialized.expr.Function) then
    return type_check.expr_function(cx, node)

  elseif node:is(ast.specialized.expr.FieldAccess) then
    return type_check.expr_field_access(cx, node)

  elseif node:is(ast.specialized.expr.IndexAccess) then
    return type_check.expr_index_access(cx, node)

  elseif node:is(ast.specialized.expr.MethodCall) then
    return type_check.expr_method_call(cx, node)

  elseif node:is(ast.specialized.expr.Call) then
    return type_check.expr_call(cx, node)

  elseif node:is(ast.specialized.expr.Cast) then
    return type_check.expr_cast(cx, node)

  elseif node:is(ast.specialized.expr.Ctor) then
    return type_check.expr_ctor(cx, node)

  elseif node:is(ast.specialized.expr.RawContext) then
    return type_check.expr_raw_context(cx, node)

  elseif node:is(ast.specialized.expr.RawFields) then
    return type_check.expr_raw_fields(cx, node)

  elseif node:is(ast.specialized.expr.RawPhysical) then
    return type_check.expr_raw_physical(cx, node)

  elseif node:is(ast.specialized.expr.RawRuntime) then
    return type_check.expr_raw_runtime(cx, node)

  elseif node:is(ast.specialized.expr.RawValue) then
    return type_check.expr_raw_value(cx, node)

  elseif node:is(ast.specialized.expr.Isnull) then
    return type_check.expr_isnull(cx, node)

  elseif node:is(ast.specialized.expr.New) then
    return type_check.expr_new(cx, node)

  elseif node:is(ast.specialized.expr.Null) then
    return type_check.expr_null(cx, node)

  elseif node:is(ast.specialized.expr.DynamicCast) then
    return type_check.expr_dynamic_cast(cx, node)

  elseif node:is(ast.specialized.expr.StaticCast) then
    return type_check.expr_static_cast(cx, node)

  elseif node:is(ast.specialized.expr.UnsafeCast) then
    return type_check.expr_unsafe_cast(cx, node)

  elseif node:is(ast.specialized.expr.Ispace) then
    return type_check.expr_ispace(cx, node)

  elseif node:is(ast.specialized.expr.Region) then
    return type_check.expr_region(cx, node)

  elseif node:is(ast.specialized.expr.Partition) then
    return type_check.expr_partition(cx, node)

  elseif node:is(ast.specialized.expr.PartitionEqual) then
    return type_check.expr_partition_equal(cx, node)

  elseif node:is(ast.specialized.expr.PartitionByField) then
    return type_check.expr_partition_by_field(cx, node)

  elseif node:is(ast.specialized.expr.Image) then
    if not node.region.fields and
       node.region.region:is(ast.specialized.expr.Function) then
      return type_check.expr_image_by_task(cx, node)
    else
      return type_check.expr_image(cx, node)
    end

  elseif node:is(ast.specialized.expr.Preimage) then
    return type_check.expr_preimage(cx, node)

  elseif node:is(ast.specialized.expr.CrossProduct) then
    return type_check.expr_cross_product(cx, node)

  elseif node:is(ast.specialized.expr.CrossProductArray) then
    return type_check.expr_cross_product_array(cx, node)

  elseif node:is(ast.specialized.expr.ListSlicePartition) then
    return type_check.expr_list_slice_partition(cx, node)

  elseif node:is(ast.specialized.expr.ListDuplicatePartition) then
    return type_check.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.specialized.expr.ListCrossProduct) then
    return type_check.expr_list_cross_product(cx, node)

  elseif node:is(ast.specialized.expr.ListCrossProductComplete) then
    return type_check.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.specialized.expr.ListPhaseBarriers) then
    return type_check.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.specialized.expr.ListInvert) then
    return type_check.expr_list_invert(cx, node)

  elseif node:is(ast.specialized.expr.ListRange) then
    return type_check.expr_list_range(cx, node)

  elseif node:is(ast.specialized.expr.ListIspace) then
    return type_check.expr_list_ispace(cx, node)

  elseif node:is(ast.specialized.expr.ListFromElement) then
    return type_check.expr_list_from_element(cx, node)

  elseif node:is(ast.specialized.expr.PhaseBarrier) then
    return type_check.expr_phase_barrier(cx, node)

  elseif node:is(ast.specialized.expr.DynamicCollective) then
    return type_check.expr_dynamic_collective(cx, node)

  elseif node:is(ast.specialized.expr.DynamicCollectiveGetResult) then
    return type_check.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.specialized.expr.Advance) then
    return type_check.expr_advance(cx, node)

  elseif node:is(ast.specialized.expr.Adjust) then
    return type_check.expr_adjust(cx, node)

  elseif node:is(ast.specialized.expr.Arrive) then
    return type_check.expr_arrive(cx, node)

  elseif node:is(ast.specialized.expr.Await) then
    return type_check.expr_await(cx, node)

  elseif node:is(ast.specialized.expr.Copy) then
    return type_check.expr_copy(cx, node)

  elseif node:is(ast.specialized.expr.Fill) then
    return type_check.expr_fill(cx, node)

  elseif node:is(ast.specialized.expr.Acquire) then
    return type_check.expr_acquire(cx, node)

  elseif node:is(ast.specialized.expr.Release) then
    return type_check.expr_release(cx, node)

  elseif node:is(ast.specialized.expr.AttachHDF5) then
    return type_check.expr_attach_hdf5(cx, node)

  elseif node:is(ast.specialized.expr.DetachHDF5) then
    return type_check.expr_detach_hdf5(cx, node)

  elseif node:is(ast.specialized.expr.AllocateScratchFields) then
    return type_check.expr_allocate_scratch_fields(cx, node)

  elseif node:is(ast.specialized.expr.WithScratchFields) then
    return type_check.expr_with_scratch_fields(cx, node)

  elseif node:is(ast.specialized.expr.Unary) then
    return type_check.expr_unary(cx, node)

  elseif node:is(ast.specialized.expr.Binary) then
    return type_check.expr_binary(cx, node)

  elseif node:is(ast.specialized.expr.Deref) then
    return type_check.expr_deref(cx, node)

  elseif node:is(ast.specialized.expr.LuaTable) then
    report.error(node, "unable to specialize value of type table")

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
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
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_for_list(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value)

  if not (std.is_ispace(value_type) or std.is_region(value_type) or
            (std.is_list(value_type) and not value_type:is_list_of_regions()))
  then
    report.error(node, "iterator for loop expected ispace, region or list, got " ..
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
  local values = node.values:map(
    function(value) return type_check.expr(cx, value) end)
  local value_types = values:map(
    function(value) return std.check_read(cx, value) end)

  local types = terralib.newlist()
  for i, symbol in ipairs(node.symbols) do
    local var_type = symbol:hastype()

    local value = values[i]
    local value_type = value_types[i]
    if var_type then
      if value and not std.validate_implicit_cast(value_type, var_type) then
        report.error(node, "type mismatch in var: expected " .. tostring(var_type) .. " but got " .. tostring(value_type))
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
    types:insert(var_type)
  end

  values = data.zip(node.symbols, values, value_types):map(function(tuple)
    local sym, value, value_type = unpack(tuple)
    return insert_implicit_cast(value, value_type, sym:gettype())
  end)

  local value = false
  if #values > 0 then value = values[1] end

  return ast.typed.stat.Var {
    symbol = node.symbols[1],
    type = types[1],
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
  local lhs = node.lhs:map(
    function(value) return type_check.expr(cx, value) end)
  local lhs_types = lhs:map(
    function(lh) return std.check_write(cx, lh) end)

  local rhs = node.rhs:map(
    function(value) return type_check.expr(cx, value) end)
  local rhs_types = rhs:map(
    function(rh) return std.check_read(cx, rh) end)

  for i, lhs_type in ipairs(lhs_types) do
    local rhs_type = rhs_types[i]

    if not std.validate_implicit_cast(rhs_type, lhs_type) then
      report.error(node, "type mismatch in assignment: expected " .. tostring(lhs_type) .. " but got " .. tostring(rhs_type))
    end
  end

  rhs = data.zip(lhs_types, rhs, rhs_types):map(function(tuple)
    local lh_type, rh, rh_type = unpack(tuple)
    return insert_implicit_cast(rh, rh_type, lh_type)
  end)

  return ast.typed.stat.Assignment {
    lhs = lhs,
    rhs = rhs,
    annotations = node.annotations,
    span = node.span,
  }
end

function type_check.stat_reduce(cx, node)
  local lhs = node.lhs:map(
    function(value) return type_check.expr(cx, value) end)
  local lhs_types = lhs:map(
    function(lh) return std.check_reduce(cx, node.op, lh) end)

  local rhs = node.rhs:map(
    function(value) return type_check.expr(cx, value) end)
  local rhs_types = rhs:map(
    function(rh) return std.check_read(cx, rh) end)

  data.zip(lhs_types, rhs_types):map(
    function(types)
      local lhs_type, rhs_type = unpack(types)
      local expr_type = binary_ops[node.op](cx, node, lhs_type, rhs_type)
      if not std.validate_explicit_cast(expr_type, lhs_type) then
        report.error(node, "type mismatch between " .. tostring(expr_type) .. " and " .. tostring(lhs_type))
      end
    end)


  return ast.typed.stat.Reduce {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
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

function type_check.stat(cx, node)
  if node:is(ast.specialized.stat.If) then
    return type_check.stat_if(cx, node)

  elseif node:is(ast.specialized.stat.While) then
    return type_check.stat_while(cx, node)

  elseif node:is(ast.specialized.stat.ForNum) then
    return type_check.stat_for_num(cx, node)

  elseif node:is(ast.specialized.stat.ForList) then
    return type_check.stat_for_list(cx, node)

  elseif node:is(ast.specialized.stat.Repeat) then
    return type_check.stat_repeat(cx, node)

  elseif node:is(ast.specialized.stat.MustEpoch) then
    return type_check.stat_must_epoch(cx, node)

  elseif node:is(ast.specialized.stat.Block) then
    return type_check.stat_block(cx, node)

  elseif node:is(ast.specialized.stat.Var) then
    return type_check.stat_var(cx, node)

  elseif node:is(ast.specialized.stat.VarUnpack) then
    return type_check.stat_var_unpack(cx, node)

  elseif node:is(ast.specialized.stat.Return) then
    return type_check.stat_return(cx, node)

  elseif node:is(ast.specialized.stat.Break) then
    return type_check.stat_break(cx, node)

  elseif node:is(ast.specialized.stat.Assignment) then
    return type_check.stat_assignment(cx, node)

  elseif node:is(ast.specialized.stat.Reduce) then
    return type_check.stat_reduce(cx, node)

  elseif node:is(ast.specialized.stat.Expr) then
    return type_check.stat_expr(cx, node)

  elseif node:is(ast.specialized.stat.RawDelete) then
    return type_check.stat_raw_delete(cx, node)

  elseif node:is(ast.specialized.stat.Fence) then
    return type_check.stat_fence(cx, node)

  elseif node:is(ast.specialized.stat.ParallelizeWith) then
    return type_check.stat_parallelize_with(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.top_task_param(cx, node, mapping, is_defined)
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

function type_check.top_task(cx, node)
  local return_type = node.return_type
  local cx = cx:new_task_scope(return_type)

  local is_defined = node.prototype:has_primary_variant()
  if is_defined then
    cx:set_external(node.prototype:get_primary_variant():is_external())
  end

  local mapping = {}
  local params = node.params:map(
    function(param)
      return type_check.top_task_param(cx, param, mapping, is_defined)
    end)
  local prototype = node.prototype
  prototype:set_param_symbols(
    params:map(function(param) return param.symbol end))

  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:set_type(task_type)

  local privileges = type_check.privileges(cx, node.privileges)
  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      local privilege_type = privilege.privilege
      local region = privilege.region
      local field_path = privilege.field_path
      assert(std.type_supports_privileges(region:gettype()))
      std.add_privilege(cx, privilege_type, region:gettype(), field_path)
      cx:intern_region(region:gettype())
    end
  end
  prototype:set_privileges(privileges)

  local coherence_modes = type_check.coherence_modes(cx, node.coherence_modes)
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
    },
    region_divergence = false,
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
    local new_node = type_check.top_task(cx, node)
    if new_node.prototype:has_primary_variant() then
      new_node.prototype:get_primary_variant():set_ast(new_node)
    end
    return new_node

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
