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

-- Regent Standard Library

local affine_helper = require("regent/affine_helper")
local ast = require("regent/ast")
local base = require("regent/std_base")
local cudahelper = require("regent/cudahelper")
local data = require("common/data")
local ffi = require("ffi")
local header_helper = require("regent/header_helper")
local log = require("common/log")
local pretty = require("regent/pretty")
local profile = require("regent/profile")
local report = require("common/report")

local std = {}

std.config, std.args = base.config, base.args

local c = base.c
std.c = c

std.file_read_only = c.LEGION_FILE_READ_ONLY
std.file_read_write = c.LEGION_FILE_READ_WRITE
std.file_create = c.LEGION_FILE_CREATE
std.check_cuda_available = cudahelper.check_cuda_available

-- #####################################
-- ## Utilities
-- #################

std.assert_error = base.assert_error
std.assert = base.assert
std.domain_from_bounds_1d = base.domain_from_bounds_1d
std.domain_from_bounds_2d = base.domain_from_bounds_2d
std.domain_from_bounds_3d = base.domain_from_bounds_3d

-- #####################################
-- ## Kinds
-- #################

-- Privileges

std.reads = ast.privilege_kind.Reads {}
std.writes = ast.privilege_kind.Writes {}
function std.reduces(op)
  local ops = {
    ["+"] = true, ["-"] = true, ["*"] = true, ["/"] = true,
    ["max"] = true, ["min"] = true,
  }
  assert(ops[op])
  return ast.privilege_kind.Reduces { op = op }
end

function std.is_reduce(privilege)
  return privilege:is(ast.privilege_kind.Reduces)
end

-- Coherence Modes

std.exclusive = ast.coherence_kind.Exclusive {}
std.atomic = ast.coherence_kind.Atomic {}
std.simultaneous = ast.coherence_kind.Simultaneous {}
std.relaxed = ast.coherence_kind.Relaxed {}

-- Flags

std.no_access_flag = ast.flag_kind.NoAccessFlag {}

-- Conditions

std.arrives = ast.condition_kind.Arrives {}
std.awaits = ast.condition_kind.Awaits {}

-- Constraints

std.subregion = ast.constraint_kind.Subregion {}
std.disjointness = ast.constraint_kind.Disjointness {}

-- #####################################
-- ## Privileges
-- #################

function std.privilege(privilege, region, field_path)
  assert(privilege:is(ast.privilege_kind), "privilege expected argument 1 to be a privilege kind")
  assert(std.is_symbol(region), "privilege expected argument 2 to be a symbol")

  if field_path == nil then
    field_path = data.newtuple()
  elseif type(field_path) == "string" then
    field_path = data.newtuple(field_path)
  end
  assert(data.is_tuple(field_path), "privilege expected argument 3 to be a field")

  return ast.privilege.Privilege {
    region = region,
    field_path = field_path,
    privilege = privilege,
  }
end

function std.privileges(privilege, regions_fields)
  local privileges = terralib.newlist()
  for _, region_fields in ipairs(regions_fields) do
    local region, fields
    if std.is_symbol(region_fields) then
      region = region_fields
      fields = terralib.newlist({data.newtuple()})
    else
      region = region_fields.region
      fields = region_fields.fields
    end
    assert(std.is_symbol(region) and terralib.islist(fields))
    for _, field in ipairs(fields) do
      privileges:insert(std.privilege(privilege, region, field))
    end
  end
  return privileges
end

-- #####################################
-- ## Constraints
-- #################

function std.constraint(lhs, rhs, op)
  assert(op:is(ast.constraint_kind))
  return ast.constraint.Constraint {
    lhs = lhs,
    rhs = rhs,
    op = op,
  }
end

-- #####################################
-- ## Privilege and Constraint Helpers
-- #################

function std.add_privilege(cx, privilege, region, field_path)
  assert(privilege:is(ast.privilege_kind))
  assert(std.type_supports_privileges(region))
  assert(data.is_tuple(field_path))
  if not cx.privileges[privilege] then
    cx.privileges[privilege] = data.newmap()
  end
  if not cx.privileges[privilege][region] then
    cx.privileges[privilege][region] = data.newmap()
  end
  cx.privileges[privilege][region][field_path] = true
end

function std.copy_privileges(cx, from_region, to_region)
  assert(std.type_supports_privileges(from_region))
  assert(std.type_supports_privileges(to_region))
  local privileges_to_copy = terralib.newlist()
  for privilege, privilege_regions in cx.privileges:items() do
    local privilege_fields = privilege_regions[from_region]
    if privilege_fields then
      for _, field_path in privilege_fields:keys() do
        privileges_to_copy:insert({privilege, to_region, field_path})
      end
    end
  end
  for _, privilege in ipairs(privileges_to_copy) do
    std.add_privilege(cx, unpack(privilege))
  end
end

function std.add_constraint(cx, lhs, rhs, op, symmetric)
  if std.is_cross_product(lhs) then lhs = lhs:partition() end
  if std.is_cross_product(rhs) then rhs = rhs:partition() end
  assert(std.type_supports_constraints(lhs))
  assert(std.type_supports_constraints(rhs))
  cx.constraints[op][lhs][rhs] = true
  if symmetric then
    std.add_constraint(cx, rhs, lhs, op, false)
  end
end

function std.add_constraints(cx, constraints)
  for _, constraint in ipairs(constraints) do
    local lhs, rhs, op = constraint.lhs, constraint.rhs, constraint.op
    local symmetric = op == std.disjointness
    std.add_constraint(cx, lhs:gettype(), rhs:gettype(), op, symmetric)
  end
end

function std.search_constraint_predicate(cx, region, visited, predicate)
  if predicate(cx, region) then
    return region
  end

  if visited[region] then
    return nil
  end
  visited[region] = true

  if cx.constraints:has(std.subregion) and cx.constraints[std.subregion]:has(region) then
    for subregion, _ in cx.constraints[std.subregion][region]:items() do
      local result = std.search_constraint_predicate(
        cx, subregion, visited, predicate)
      if result then return result end
    end
  end
  return nil
end

function std.search_privilege(cx, privilege, region, field_path, visited)
  assert(privilege:is(ast.privilege_kind))
  assert(std.type_supports_privileges(region))
  assert(data.is_tuple(field_path))
  return std.search_constraint_predicate(
    cx, region, visited,
    function(cx, region)
      return cx.privileges[privilege] and
        cx.privileges[privilege][region] and
        cx.privileges[privilege][region][field_path]
    end)
end

function std.check_privilege(cx, privilege, region, field_path)
  assert(privilege:is(ast.privilege_kind))
  assert(std.type_supports_privileges(region))
  assert(data.is_tuple(field_path))
  for i = #field_path, 0, -1 do
    if std.search_privilege(cx, privilege, region, field_path:slice(1, i), {}) then
      return true
    end
    if std.is_reduce(privilege) then
      if std.search_privilege(cx, std.reads, region, field_path:slice(1, i), {}) and
        std.search_privilege(cx, std.writes, region, field_path:slice(1, i), {})
      then
        return true
      end
    end
  end
  return false
end

function std.search_any_privilege(cx, region, field_path, visited)
  assert(std.is_region(region) and data.is_tuple(field_path))
  return std.search_constraint_predicate(
    cx, region, visited,
    function(cx, region)
      for _, regions in cx.privileges:items() do
        if regions[region] and regions[region][field_path] then
          return true
        end
      end
      return false
    end)
end

function std.check_any_privilege(cx, region, field_path)
  assert(std.is_region(region) and data.is_tuple(field_path))
  for i = #field_path, 0, -1 do
    if std.search_any_privilege(cx, region, field_path:slice(1, i), {}) then
      return true
    end
  end
  return false
end

local function analyze_uses_variables_node(variables)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.ID) then
      return variables[node.value]

    elseif node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Call) or
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Deref)
    then
      return false

    elseif node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary)
    then
      return false

    -- Miscellaneous:
    elseif node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind)
    then
      return false

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_uses_variables(node, variables)
  return ast.mapreduce_node_postorder(
    analyze_uses_variables_node(variables),
    data.any,
    node, false)
end

local function uses_variables(region, variables)
  if region:has_index_expr() then
    return analyze_uses_variables(region:get_index_expr(), variables)
  end
  return false
end

function std.search_constraint(cx, region, constraint, exclude_variables,
                               visited, reflexive, symmetric)
  return std.search_constraint_predicate(
    cx, region, visited,
    function(cx, region)
      if reflexive and region == constraint.rhs then
        return true
      end

      if cx.constraints:has(constraint.op) and
        cx.constraints[constraint.op]:has(region) and
        cx.constraints[constraint.op][region][constraint.rhs] and
        not (exclude_variables and
               uses_variables(region, exclude_variables) and
               uses_variables(constraint.rhs, exclude_variables))
      then
        return true
      end

      if symmetric then
        local constraint = {
          lhs = constraint.rhs,
          rhs = region,
          op = constraint.op,
        }
        if std.search_constraint(cx, constraint.lhs, constraint,
                                 exclude_variables, {}, reflexive, false)
        then
          return true
        end
      end

      return false
    end)
end

function std.check_constraint(cx, constraint, exclude_variables)
  local lhs = constraint.lhs
  if lhs == std.wild then
    return true
  elseif std.is_symbol(lhs) then
    lhs = lhs:gettype()
  end
  if std.is_cross_product(lhs) then lhs = lhs:partition() end
  assert(std.type_supports_constraints(lhs))

  local rhs = constraint.rhs
  if rhs == std.wild then
    return true
  elseif std.is_symbol(rhs) then
    rhs = rhs:gettype()
  end
  if std.is_cross_product(rhs) then rhs = rhs:partition() end
  assert(std.type_supports_constraints(rhs))

  local constraint = {
    lhs = lhs,
    rhs = rhs,
    op = constraint.op,
  }
  return std.search_constraint(
    cx, constraint.lhs, constraint, exclude_variables, {},
    constraint.op == std.subregion --[[ reflexive ]],
    constraint.op == std.disjointness --[[ symmetric ]])
end

function std.check_constraints(cx, constraints, mapping)
  if not mapping then
    mapping = {}
  end

  for _, constraint in ipairs(constraints) do
    local constraint = {
      lhs = mapping[constraint.lhs] or constraint.lhs,
      rhs = mapping[constraint.rhs] or constraint.rhs,
      op = constraint.op,
    }
    if not std.check_constraint(cx, constraint) then
      return false, constraint
    end
  end
  return true
end

-- #####################################
-- ## Physical Privilege Helpers
-- #################

std.reduction_op_init = base.reduction_op_init
std.reduction_op_ids = base.reduction_op_ids
std.is_reduction_op = base.is_reduction_op
std.get_reduction_op = base.get_reduction_op
std.meet_privilege = base.meet_privilege
std.meet_coherence = base.meet_coherence
std.meet_flag = base.meet_flag
std.find_task_privileges = base.find_task_privileges
std.group_task_privileges_by_field_path = base.group_task_privileges_by_field_path
std.privilege_mode = base.privilege_mode
std.coherence_mode = base.coherence_mode
std.flag_mode = base.flag_mode

-- #####################################
-- ## Type Helpers
-- #################

for k, v in pairs(base.types) do
  std[k] = v
end

std.untyped = terralib.types.newstruct("untyped")

function std.type_sub(t, mapping)
  if mapping[t] then
    return mapping[t]
  elseif std.is_bounded_type(t) then
    if t.points_to_type then
      return t.index_type(
        std.type_sub(t.points_to_type, mapping),
        unpack(t.bounds_symbols:map(
                 function(bound) return std.type_sub(bound, mapping) end)))
    else
      return t.index_type(
        unpack(t.bounds_symbols:map(
                 function(bound) return std.type_sub(bound, mapping) end)))
    end
  elseif std.is_fspace_instance(t) then
    return t.fspace(unpack(t.args:map(
      function(arg) return std.type_sub(arg, mapping) end)))
  elseif std.is_rawref(t) then
    return std.rawref(std.type_sub(t.pointer_type, mapping))
  elseif std.is_ref(t) then
    return std.ref(std.type_sub(t.pointer_type, mapping), unpack(t.field_path))
  elseif terralib.types.istype(t) and t:ispointer() then
    return &std.type_sub(t.type, mapping)
  elseif std.is_partition(t) then
    local parent_region_symbol = mapping[t.parent_region_symbol] or t.parent_region_symbol
    local colors_symbol = mapping[t.colors_symbol] or t.colors_symbol
    if parent_region_symbol == t.parent_region_symbol and
       colors_symbol == t.colors_symbol then
       return t
    else
      return std.partition(t.disjointness, parent_region_symbol, colors_symbol)
    end
  elseif terralib.types.istype(t) and t:isarray() then
    return std.type_sub(t.type, mapping)[t.N]
  else
    return t
  end
end

function std.type_eq(a, b, mapping)
  -- Determine if a == b with substitutions mapping a -> b

  if not mapping then
    mapping = {}
  end

  if a == b then
    return true
  elseif mapping[a] == b or mapping[b] == a then
    return true
  elseif std.is_symbol(a) and std.is_symbol(b) then
    if a == std.wild or b == std.wild then
      return true
    end
    return std.type_eq(a:gettype(), b:gettype(), mapping)
  elseif a == std.wild_type and std.is_region(b) then
    return true
  elseif b == std.wild_type and std.is_region(a) then
    return true
  elseif std.is_bounded_type(a) and std.is_bounded_type(b) then
    if not std.type_eq(a.index_type, b.index_type, mapping) then
      return false
    end
    if not std.type_eq(a.points_to_type, b.points_to_type, mapping) then
      return false
    end
    -- FIXME: pass in bounds properly for type_eq
    local a_bounds, a_error_message = a:bounds()
    if a_bounds == nil then report.error({ span = ast.trivial_span() }, a_error_message) end
    local b_bounds, b_error_message = b:bounds()
    if b_bounds == nil then report.error({ span = ast.trivial_span() }, b_error_message) end
    if #a_bounds ~= #b_bounds then
      return false
    end
    for i, a_region in ipairs(a_bounds) do
      local b_region = b_bounds[i]
      if not std.type_eq(a_region, b_region, mapping) then
        return false
      end
    end
    return true
  elseif std.is_fspace_instance(a) and std.is_fspace_instance(b) and
    a.fspace == b.fspace
  then
    for i, a_arg in ipairs(a.args) do
      local b_arg = b.args[i]
      if not std.type_eq(a_arg, b_arg, mapping) then
        return false
      end
    end
    return true
  elseif std.is_list(a) and std.is_list(b) then
    return std.type_eq(a.element_type, b.element_type, mapping)
  else
    return false
  end
end

function std.type_maybe_eq(a, b, mapping)
  -- Returns false ONLY if a and b are provably DIFFERENT types. So
  --
  --     type_maybe_eq(ptr(int, a), ptr(int, b))
  --
  -- might return true (even if a and b are NOT type_eq) because if
  -- the regions a and b alias then it is possible for a value to
  -- inhabit both types.

  if std.type_eq(a, b, mapping) then
    return true
  elseif std.is_bounded_type(a) and std.is_bounded_type(b) then
    return std.type_maybe_eq(a.points_to_type, b.points_to_type, mapping)
  elseif std.is_fspace_instance(a) and std.is_fspace_instance(b) and
    a.fspace == b.fspace
  then
    return true
  elseif (std.is_ctor(a) and std.validate_implicit_cast(a, b)) or
    (std.is_ctor(b) and std.validate_implicit_cast(b, a))
  then
    return true
  elseif std.is_list(a) and std.is_list(b) then
    return std.type_maybe_eq(a.element_type, b.element_type, mapping)
  else
    return false
  end
end

local function add_region_symbol(symbols, region)
  if not symbols[region:gettype()] then
    symbols[region:gettype()] = region
  end
end

local function add_type(symbols, type)
  if std.is_bounded_type(type) then
    for _, bound in ipairs(type.bounds_symbols) do
      add_region_symbol(symbols, bound)
    end
  elseif std.is_fspace_instance(type) then
    for _, arg in ipairs(type.args) do
      add_region_symbol(symbols, arg)
    end
  elseif std.is_list(type) then
    add_type(symbols, type.element_type)
  elseif std.is_region(type) then
    -- FIXME: Would prefer to not get errors at all here.
    pcall(function() add_type(symbols, type.fspace_type) end)
  end
end

function std.struct_entries_symbols(fs, symbols)
  if not symbols then
    symbols = {}
  end
  fs:getentries():map(function(entry)
      add_type(symbols, entry[2] or entry.type)
  end)
  if std.is_fspace_instance(fs) then
    fs:getconstraints():map(function(constraint)
      add_region_symbol(symbols, constraint.lhs)
      add_region_symbol(symbols, constraint.rhs)
    end)
  end

  local entries_symbols = terralib.newlist()
  for _, entry in ipairs(fs:getentries()) do
    local field_name = entry[1] or entry.field
    local field_type = entry[2] or entry.type
    if terralib.islabel(field_name) then
      entries_symbols:insert(field_name)
    elseif symbols[field_type] then
      entries_symbols:insert(symbols[field_type])
    else
      local new_symbol = std.newsymbol(field_type, field_name)
      entries_symbols:insert(new_symbol)
    end
  end

  return entries_symbols
end

function std.fn_param_symbols(fn_type)
  local params = fn_type.parameters
  local symbols = {}
  params:map(function(param) add_type(symbols, param) end)
  add_type(symbols, fn_type.returntype)

  local param_symbols = terralib.newlist()
  for _, param in ipairs(params) do
    if symbols[param] then
      param_symbols:insert(symbols[param])
    else
      param_symbols:insert(std.newsymbol(param))
    end
  end

  return param_symbols
end

local function type_compatible(a, b)
  return (std.is_ispace(a) and std.is_ispace(b)) or
    (std.is_region(a) and std.is_region(b)) or
    (std.is_partition(a) and std.is_partition(b)) or
    (std.is_cross_product(a) and std.is_cross_product(b)) or
    (std.is_list_of_regions(a) and std.is_list_of_regions(b))
end

local function type_isomorphic(param_type, arg_type, check, mapping)
  if std.is_ispace(param_type) and std.is_ispace(arg_type) then
    return std.type_eq(param_type.index_type, arg_type.index_type, mapping)
  elseif std.is_region(param_type) and std.is_region(arg_type) then
    return std.type_eq(param_type:ispace(), arg_type:ispace(), mapping) and
      std.type_eq(param_type.fspace_type, arg_type.fspace_type, mapping)
  elseif std.is_partition(param_type) and std.is_partition(arg_type) then
    return param_type:is_disjoint() == arg_type:is_disjoint() and
      check(param_type:parent_region(), arg_type:parent_region(), mapping) and
      check(param_type:colors(), arg_type:colors(), mapping)
  elseif
    std.is_cross_product(param_type) and std.is_cross_product(arg_type)
  then
    return (#param_type:partitions() == #arg_type:partitions()) and
      data.all(
        unpack(data.zip(param_type:partitions(), arg_type:partitions()):map(
          function(pair)
            local param_partition, arg_partition = unpack(pair)
            return check(param_partition, arg_partition, mapping)
      end)))
  elseif std.is_list_of_regions(param_type) and std.is_list_of_regions(arg_type)
  then
    return std.type_eq(
      param_type.element_type:fspace(), arg_type.element_type:fspace())
  else
    return false
  end
end

local function unify_param_type_args(param, param_type, arg_type, mapping)
  if std.is_region(param_type) and
    type_compatible(param_type:ispace(), arg_type:ispace()) and
    not (mapping[param] or mapping[param_type] or mapping[param_type:ispace()]) and
    type_isomorphic(param_type:ispace(), arg_type:ispace(), mapping)
  then
    mapping[param_type:ispace()] = arg_type:ispace()
  elseif std.is_partition(param_type) and
    type_compatible(param_type:colors(), arg_type:colors()) and
    not (mapping[param] or mapping[param_type] or mapping[param_type:colors()]) and
    type_isomorphic(param_type:colors(), arg_type:colors(), mapping)
  then
    mapping[param_type:colors()] = arg_type:colors()
  end
end

local function reconstruct_param_as_arg_symbol(param_type, mapping)
  local param_as_arg_symbol = mapping[param_type]
  for k, v in pairs(mapping) do
    if std.is_symbol(v) and (v:gettype() == param_type or v:gettype() == mapping[param_type]) then
      param_as_arg_symbol = v
    end
  end
  return param_as_arg_symbol
end

local function reconstruct_param_as_arg_type(param_type, mapping, optional)
  if std.is_ispace(param_type) then
    local index_type = std.type_sub(param_type.index_type, mapping)
    return std.ispace(index_type)
  elseif std.is_region(param_type) then
    local param_ispace_as_arg_type =
      reconstruct_param_as_arg_symbol(param_type:ispace(), mapping) or
      param_type:ispace()
    local fspace_type = std.type_sub(param_type.fspace_type, mapping)
    return std.region(param_ispace_as_arg_type, fspace_type)
  elseif std.is_partition(param_type) then
    local param_parent_region_as_arg_type =
      reconstruct_param_as_arg_symbol(param_type:parent_region(), mapping)
    local param_colors_as_arg_type =
      reconstruct_param_as_arg_symbol(param_type:colors(), mapping) or
      reconstruct_param_as_arg_type(param_type:colors(), mapping)
    return std.partition(
      param_type.disjointness, param_parent_region_as_arg_type,
      param_colors_as_arg_type)
  elseif std.is_cross_product(param_type) then
    local param_partitions = param_type:partitions()
    local param_partitions_as_arg_type = param_partitions:map(
      function(param_partition)
        return reconstruct_param_as_arg_symbol(param_partition, mapping)
    end)
    return std.cross_product(unpack(param_partitions_as_arg_type))
  elseif std.is_list_of_regions(param_type) then
    local fspace_type = std.type_sub(param_type.element_type.fspace_type, mapping)
    return std.list(std.region(fspace_type))
  else
    assert(optional)
  end
end

local function reconstruct_return_as_arg_type(return_type, mapping)
  if mapping[return_type] then
    return std.type_sub(return_type, mapping)
  end

  local result = reconstruct_param_as_arg_type(return_type, mapping, true)
  if result then return result end

  return std.type_sub(return_type, mapping)
end

function std.validate_args(node, params, args, isvararg, return_type, mapping, strict)
  if (#args < #params) or (#args > #params and not isvararg) then
    report.error(node, "expected " .. tostring(#params) .. " arguments but got " .. tostring(#args))
  end

  -- FIXME: All of these calls are being done with the order backwards
  -- for validate_implicit_cast, but everything breaks if I swap the
  -- order. For the moment, the fix is to make validate_implicit_cast
  -- symmetric as much as possible.
  local check
  if strict then
    check = std.type_eq
  else
    check = std.validate_implicit_cast
  end

  if not mapping then
    mapping = {}
  end

  local need_cast = terralib.newlist()
  for i, param in ipairs(params) do
    local arg = args[i]
    local param_type = param:gettype()
    local arg_type = arg:gettype()

    -- Sanity check that we're not getting references here.
    assert(not (std.is_ref(arg_type) or std.is_rawref(arg_type)))

    if param_type == std.untyped or
      arg_type == std.untyped or
      param_type == arg_type or
      mapping[param_type] == arg_type
    then
      -- Ok
      need_cast[i] = false
    elseif type_compatible(param_type, arg_type) then
      -- Regions (and other unique types) require a special pass here

      -- Check for previous mappings. This can happen if two
      -- parameters are aliased to the same region.
      if (mapping[param] or mapping[param_type]) and
        not (mapping[param] == arg or mapping[param_type] == arg_type)
      then
        local param_as_arg_type = mapping[param_type]
        for k, v in pairs(mapping) do
          if std.is_symbol(v) and v:gettype() == mapping[param_type] then
            param_as_arg_type = v
          end
        end
        report.error(node, "type mismatch in argument " .. tostring(i) ..
                    ": expected " .. tostring(param_as_arg_type) ..
                    " but got " .. tostring(arg))
      end

      -- Allow type arguments to unify (if any).
      unify_param_type_args(param, param_type, arg_type, mapping)

      mapping[param] = arg
      mapping[param_type] = arg_type
      if not type_isomorphic(param_type, arg_type, check, mapping) then
        local param_as_arg_type = reconstruct_param_as_arg_type(param_type, mapping)
        report.error(node, "type mismatch in argument " .. tostring(i) ..
                    ": expected " .. tostring(param_as_arg_type) ..
                    " but got " .. tostring(arg_type))
      end
      need_cast[i] = false
    elseif not check(arg_type, param_type, mapping) then
      local param_as_arg_type = std.type_sub(param_type, mapping)
      report.error(node, "type mismatch in argument " .. tostring(i) ..
                  ": expected " .. tostring(param_as_arg_type) ..
                  " but got " .. tostring(arg_type))
    else
      need_cast[i] = not std.type_eq(arg_type, param_type, mapping)
    end
  end
  return reconstruct_return_as_arg_type(return_type, mapping), need_cast
end

local function unpack_type(old_type, mapping)
  if std.is_ispace(old_type) then
    local index_type = std.type_sub(old_type.index_type, mapping)
    return std.ispace(index_type), true
  elseif std.is_region(old_type) then
    local ispace_type
    if not mapping[old_type:ispace()] then
      ispace_type = unpack_type(old_type:ispace(), mapping)
    else
      ispace_type = std.type_sub(old_type:ispace(), mapping)
    end
    local fspace_type = std.type_sub(old_type.fspace_type, mapping)
    return std.region(std.newsymbol(ispace_type, old_type.ispace_symbol:hasname()), fspace_type), true
  elseif std.is_partition(old_type) then
    local parent_region_type = std.type_sub(old_type:parent_region(), mapping)
    local colors_type = std.type_sub(old_type:colors(), mapping)
    local parent_region_symbol = old_type.parent_region_symbol
    local colors_symbol = old_type.colors_symbol
    assert(not mapping[parent_region_symbol] or
           mapping[parent_region_symbol]:gettype() == parent_region_type)
    assert(not mapping[colors_symbol] or
           mapping[colors_symbol]:gettype() == colors_type)
    return std.partition(
      old_type.disjointness,
      mapping[parent_region_symbol] or
      std.newsymbol(parent_region_type, parent_region_symbol:hasname()),
      mapping[colors_symbol] or
      std.newsymbol(colors_type, colors_symbol:hasname())), true
  elseif std.is_cross_product(old_type) then
    local partitions = data.zip(old_type:partitions(), old_type.partition_symbols):map(
      function(pair)
        local old_partition_type, old_partition_symbol = unpack(pair)
        return std.newsymbol(
          std.type_sub(old_partition_type, mapping),
          old_partition_symbol:getname())
    end)
    return std.cross_product(unpack(partitions)), true
  elseif std.is_list_of_regions(old_type) then
    return std.list(unpack_type(old_type.element_type, mapping)), true
  else
    return std.type_sub(old_type, mapping), false
  end
end

function std.validate_fields(fields, constraints, params, args)
  local mapping = {}
  for i, param in ipairs(params) do
    local arg = args[i]
    mapping[param] = arg
    mapping[param:gettype()] = arg:gettype()
  end

  local new_fields = terralib.newlist()
  for _, old_field in ipairs(fields) do
    local old_symbol, old_type = old_field.field, old_field.type
    local new_symbol = std.newsymbol(old_symbol:getname())
    mapping[old_symbol] = new_symbol
    local new_type = unpack_type(old_type, mapping)
    mapping[old_type] = new_type
    new_symbol:settype(new_type)
    new_fields:insert({
        field = new_symbol:getname(),
        type = new_type,
    })
  end

  local new_constraints = terralib.newlist()
  for _, constraint in ipairs(constraints) do
    local lhs = mapping[constraint.lhs] or constraint.lhs
    local rhs = mapping[constraint.rhs] or constraint.rhs
    local op = constraint.op
    assert(lhs and rhs and op)
    new_constraints:insert({
        lhs = lhs,
        rhs = rhs,
        op = op,
    })
  end

  return new_fields, new_constraints
end

-- Terra differentiates between implicit and explicit
-- casting. Therefore, if you explicitly cast here then e.g. bool ->
-- int appears valid, but if you implicitly cast, this is invalid. For
-- now, use implicit casts. Unfortunately, for compatibility with
-- Terra, we need both behaviors.

function std.validate_implicit_cast(from_type, to_type, mapping)
  if std.type_eq(from_type, to_type, mapping) then
    return true
  end

  -- Ask the Terra compiler to kindly tell us the cast is valid.
  local function test()
    local terra query(x : from_type) : to_type
      return x
    end
    return query:gettype().returntype
  end
  local valid = pcall(test)

  return valid
end

function std.validate_explicit_cast(from_type, to_type, mapping)
  if std.type_eq(from_type, to_type, mapping) then
    return true
  end

  -- Ask the Terra compiler to kindly tell us the cast is valid.
  local function test()
    local terra query(x : from_type) : to_type
      return [to_type](x)
    end
    return query:gettype().returntype
  end
  local valid = pcall(test)

  return valid
end

function std.unpack_fields(fs, symbols)
  assert(std.is_fspace_instance(fs))

  fs:complete() -- Need fields
  local old_symbols = std.struct_entries_symbols(fs)

  -- give an identity mapping for field space arguments
  -- to avoid having two different symbols of the same name
  -- (which can later seriously confuse type substitution)
  local mapping = {}
  for i, arg in ipairs(fs.args) do
    mapping[arg] = arg
    mapping[arg:gettype()] = arg:gettype()
  end

  local new_fields = terralib.newlist()
  local new_constraints = terralib.newlist()
  local needs_unpack = false
  for i, old_field in ipairs(fs:getentries()) do
    local old_symbol, old_type = old_symbols[i], old_field.type
    local new_symbol
    local field_name = old_field[1] or old_field.field
    if symbols and symbols[field_name] then
      new_symbol = symbols[field_name]
    else
      new_symbol = std.newsymbol(old_symbol:getname())
    end

    mapping[old_symbol] = new_symbol
    local new_type, is_unpack = unpack_type(old_type, mapping)
    mapping[old_type] = new_type
    needs_unpack = needs_unpack or is_unpack

    if std.is_fspace_instance(new_type) then
      local sub_type, sub_constraints = std.unpack_fields(new_type)
      new_type = sub_type
      new_constraints:insertall(sub_constraints)
    end

    new_symbol:settype(new_type)
    new_fields:insert({
        field = new_symbol:getname(),
        type = new_type,
    })
  end

  if not needs_unpack and not symbols then
    return fs, terralib.newlist()
  end

  local constraints = fs:getconstraints()
  for _, constraint in ipairs(constraints) do
    local lhs = mapping[constraint.lhs] or constraint.lhs
    local rhs = mapping[constraint.rhs] or constraint.rhs
    local op = constraint.op
    new_constraints:insert({
        lhs = lhs,
        rhs = rhs,
        op = op,
    })
  end

  local result_type = std.ctor_named(new_fields)
  result_type.is_unpack_result = true

  return result_type, new_constraints
end

function std.as_read(t)
  assert(terralib.types.istype(t))
  if std.is_ref(t) then
    local field_type = t.refers_to_type
    for _, field in ipairs(t.field_path) do
      field_type = std.get_field(field_type, field)
      if not field_type then
        return nil
      end
    end
    assert(not std.is_ref(field_type))
    return field_type
  elseif std.is_rawref(t) then
    return t.refers_to_type
  else
    return t
  end
end

function std.check_read(cx, node)
  local t = node.expr_type
  assert(terralib.types.istype(t))
  if std.is_ref(t) then
    local region_types, error_message = t:bounds()
    if region_types == nil then report.error(node, error_message) end
    local field_path = t.field_path
    for i, region_type in ipairs(region_types) do
      if not std.check_privilege(cx, std.reads, region_type, field_path) then
        local regions = t.bounds_symbols
        local ref_as_ptr = t.pointer_type.index_type(t.refers_to_type, unpack(regions))
        report.error(node, "invalid privilege reads(" ..
                  (data.newtuple(regions[i]) .. field_path):mkstring(".") ..
                  ") for dereference of " .. tostring(ref_as_ptr))
      end
    end
  end
  return std.as_read(t)
end

function std.check_write(cx, node)
  local t = node.expr_type
  assert(terralib.types.istype(t))
  if std.is_ref(t) then
    local region_types, error_message = t:bounds()
    if region_types == nil then report.error(node, error_message) end
    local field_path = t.field_path
    for i, region_type in ipairs(region_types) do
      if not std.check_privilege(cx, std.writes, region_type, field_path) then
        local regions = t.bounds_symbols
        local ref_as_ptr = t.pointer_type.index_type(t.refers_to_type, unpack(regions))
        report.error(node, "invalid privilege writes(" ..
                  (data.newtuple(regions[i]) .. field_path):mkstring(".") ..
                  ") for dereference of " .. tostring(ref_as_ptr))
      end
    end
    return std.as_read(t)
  elseif std.is_rawref(t) then
    return std.as_read(t)
  else
    report.error(node, "type mismatch: write expected an lvalue but got " .. tostring(t))
  end
end

function std.check_reduce(cx, op, node)
  local t = node.expr_type
  assert(terralib.types.istype(t))
  if std.is_ref(t) then
    local region_types, error_message = t:bounds()
    if region_types == nil then report.error(node, error_message) end
    local field_path = t.field_path
    for i, region_type in ipairs(region_types) do
      if not std.check_privilege(cx, std.reduces(op), region_type, field_path) then
        local regions = t.bounds_symbols
        local ref_as_ptr = t.pointer_type.index_type(t.refers_to_type, unpack(regions))
        report.error(node, "invalid privilege " .. tostring(std.reduces(op)) .. "(" ..
                  (data.newtuple(regions[i]) .. field_path):mkstring(".") ..
                  ") for dereference of " .. tostring(ref_as_ptr))
      end
    end
    return std.as_read(t)
  elseif std.is_rawref(t) then
    return std.as_read(t)
  else
    report.error(node, "type mismatch: reduce expected an lvalue but got " .. tostring(t))
  end
end

function std.get_field(t, f)
  assert(terralib.types.istype(t))
  if std.is_bounded_type(t) then
    if not t:is_ptr() then
      return nil
    end
    local field_type = std.ref(t, f)
    if not std.as_read(field_type) then
      return nil
    end
    return field_type
  elseif std.is_ref(t) then
    local field_path = t.field_path .. data.newtuple(f)
    local field_type = std.ref(t, unpack(field_path))
    if not std.as_read(field_type) then
      return nil
    end
    return field_type
  elseif std.is_rawref(t) then
    local field_type = std.get_field(std.as_read(t), f)
    if std.is_ref(field_type) then
      return field_type
    elseif field_type then
      return std.rawref(&field_type)
    else
      return nil
    end
  else
    -- Ask the Terra compiler to kindly tell us the type of the requested field.
    local function test()
      local terra query(x : t)
        return x.[f]
      end
      return query:gettype().returntype
    end
    local exists, field_type = pcall(test)
    if exists then
      return field_type
    else
      return nil
    end
  end
end

function std.get_field_path(value_type, field_path)
  local field_type = value_type
  for _, field_name in ipairs(field_path) do
    field_type = std.get_field(field_type, field_name)
    assert(field_type, tostring(value_type) .. " has no field " .. tostring(field_path))
  end
  return field_type
end

local function type_requires_force_cast(a, b)
  return (std.is_ispace(a) and std.is_ispace(b)) or
    (std.is_region(a) and std.is_region(b)) or
    (std.is_partition(a) and std.is_partition(b)) or
    (std.is_cross_product(a) and std.is_cross_product(b)) or
    (std.is_list_of_regions(a) and std.is_list_of_regions(b)) or
    (std.is_bounded_type(a) and std.is_bounded_type(b)) or
    (std.is_fspace_instance(a) and std.is_fspace_instance(b))
end

function std.implicit_cast(from, to, expr)
  assert(not (std.is_ref(from) or std.is_rawref(from)))
  if type_requires_force_cast(from, to) then
    return to:force_cast(from, to, expr)
  else
    return `([to]([expr]))
  end
end

function std.explicit_cast(from, to, expr)
  assert(not (std.is_ref(from) or std.is_rawref(from)))
  if type_requires_force_cast(from, to) then
    return to:force_cast(from, to, expr)
  else
    return `([to]([expr]))
  end
end

function std.fn_params_with_privileges_by_index(fn_type)
  local params = fn_type.parameters
  return data.filteri(std.type_supports_privileges, params)
end

function std.fn_param_regions_by_index(fn_type)
  local params = fn_type.parameters
  return data.filteri(std.is_region, params)
end

function std.fn_param_lists_of_regions_by_index(fn_type)
  local params = fn_type.parameters
  return data.filteri(function(t) return std.is_list_of_regions(t) end, params)
end

-- #####################################
-- ## Serialization Helpers
-- #################

local function compute_serialized_size_inner(value_type, value)
  if std.is_list(value_type) then
    local result = terralib.newsymbol(c.size_t, "result")
    local element_type = value_type.element_type
    local element = terralib.newsymbol(&element_type)

    local size_actions, size_value = compute_serialized_size_inner(
      element_type, `(@element))
    local actions = quote
      var [result] = 0
      for i = 0, [value].__size do
        var [element] = ([&element_type]([value].__data)) + i
        [size_actions]
        [result] = [result] + terralib.sizeof(element_type) + [size_value]
      end
    end
    return actions, result
  elseif std.is_string(value_type) then
    return quote end, `(c.strlen([rawstring](value)) + 1)
  else
    return quote end, 0
  end
end

local compute_serialized_size_helper = terralib.memoize(function(value_type)
  local value = terralib.newsymbol(value_type, "value")
  local actions, result = compute_serialized_size_inner(value_type, value)
  local terra compute_serialized_size([value]) : c.size_t
    [actions];
    return [result]
  end
  compute_serialized_size:setinlined(false)
  return compute_serialized_size
end)

function std.compute_serialized_size(value_type, value)
  local helper = compute_serialized_size_helper(value_type)
  local result = terralib.newsymbol(c.size_t, "result")
  local actions = quote
    var [result] = helper([value])
  end
  return actions, result
end

local function serialize_inner(value_type, value, fixed_ptr, data_ptr)
  -- Force unaligned access because malloc does not provide
  -- blocks aligned for all purposes (e.g. SSE vectors).
  local value_type_alignment = 1 -- data.min(terralib.sizeof(value_type), 8)
  local actions = quote
    terralib.attrstore(
      [&value_type](fixed_ptr), value,
      { align = [value_type_alignment] })
  end

  if std.is_list(value_type) then
    local element_type = value_type.element_type
    local element = terralib.newsymbol(element_type)
    local element_ptr = terralib.newsymbol(&element_type)

    local ser_actions = std.serialize(
      element_type, element, element_ptr, data_ptr)
    actions = quote
      [actions]
      for i = 0, [value].__size do
        var [element] = ([&element_type]([value].__data))[i]
        var [element_ptr] = [&element_type](@[data_ptr])
        @[data_ptr] = @[data_ptr] + terralib.sizeof(element_type)
        [ser_actions]
      end
    end
  elseif std.is_string(value_type) then
    actions = quote
      [actions]
      c.strcpy([rawstring](@[data_ptr]), [rawstring]([value]))
      @[data_ptr] = @[data_ptr] + c.strlen([rawstring]([value])) + 1
    end
  end

  return actions
end

local serialize_helper = terralib.memoize(function(value_type)
  local value = terralib.newsymbol(value_type, "value")
  local fixed_ptr = terralib.newsymbol(&opaque, "fixed_ptr")
  local data_ptr = terralib.newsymbol(&&uint8, "data_ptr")
  local actions = serialize_inner(value_type, value, fixed_ptr, data_ptr)
  local terra serialize([value], [fixed_ptr], [data_ptr])
    [actions]
  end
  serialize:setinlined(false)
  return serialize
end)

function std.serialize(value_type, value, fixed_ptr, data_ptr)
  local helper = serialize_helper(value_type)
  local actions = quote
    helper([value], [fixed_ptr], [data_ptr])
  end
  return actions
end

local function deserialize_inner(value_type, fixed_ptr, data_ptr)
  -- Force unaligned access because malloc does not provide
  -- blocks aligned for all purposes (e.g. SSE vectors).
  local value_type_alignment = 1 -- data.min(terralib.sizeof(value_type), 8)
  local result = terralib.newsymbol(value_type, "result")
  local actions = quote
    var [result] = terralib.attrload(
      [&value_type]([fixed_ptr]),
      { align = [value_type_alignment] })
  end

  if std.is_list(value_type) then
    local element_type = value_type.element_type
    local element_ptr = terralib.newsymbol(&element_type)

    local deser_actions, deser_value = deserialize_inner(
      element_type, element_ptr, data_ptr)
    actions = quote
      [actions]
      [result].__data = c.malloc(
        terralib.sizeof(element_type) * [result].__size)
      std.assert([result].__data ~= nil, "malloc failed in deserialize")
      for i = 0, [result].__size do
        var [element_ptr] = [&element_type](@[data_ptr])
        @[data_ptr] = @[data_ptr] + terralib.sizeof(element_type)
        [deser_actions]
        ([&element_type]([result].__data))[i] = [deser_value]
      end
    end
  elseif std.is_string(value_type) then
    actions = quote
      [actions]
      [result] = c.strdup([rawstring](@[data_ptr]))
      @[data_ptr] = @[data_ptr] + c.strlen([rawstring]([result])) + 1
    end
  end

  return actions, result
end

local deserialize_helper = terralib.memoize(function(value_type)
  local fixed_ptr = terralib.newsymbol(&opaque, "fixed_ptr")
  local data_ptr = terralib.newsymbol(&&uint8, "data_ptr")
  local actions, result = deserialize_inner(value_type, fixed_ptr, data_ptr)
  -- Force unaligned access because malloc does not provide
  -- blocks aligned for all purposes (e.g. AVX vectors).
  local value_type_alignment = 1 -- data.min(terralib.sizeof(value_type),8)
  local terra deserialize([fixed_ptr], [data_ptr])
    [actions];
    -- FIXME: Terra on PowerPC has buggy support returning structs, so
    -- work around it by mallocing the result and returning a pointer.
    var result_data = [&value_type](c.malloc([terralib.sizeof(value_type)]))
    std.assert(result_data ~= nil, "malloc failed in deserialize")
    terralib.attrstore(result_data, [result], { align = [value_type_alignment] })
    return result_data
  end
  deserialize:setinlined(false)
  return deserialize
end)

function std.deserialize(value_type, fixed_ptr, data_ptr)
  local helper = deserialize_helper(value_type)
  local result = terralib.newsymbol(value_type, "result")
  -- Force unaligned access because malloc does not provide
  -- blocks aligned for all purposes (e.g. AVX vectors).
  local value_type_alignment = 1 -- data.min(terralib.sizeof(value_type),8)
  local actions = quote
    var result_data = helper([fixed_ptr], [data_ptr])
    var [result] = terralib.attrload(result_data, { align = [value_type_alignment] })
    c.free(result_data)
  end
  return actions, result
end

-- This is a type representing a buffer containing a serialized value.
-- The value owns the buffer.
struct std.serialized_value {
  value: &opaque,
  size: uint64,
}

function std.type_size_bucket_type(value_type)
  if value_type == terralib.types.unit then
    return terralib.types.unit
  else
    return std.serialized_value
  end
end


-- #####################################
-- ## Symbols
-- #################

std.newsymbol = base.newsymbol
std.is_symbol = base.is_symbol

-- #####################################
-- ## Quotes
-- #################

local rquote = {}
function rquote:__index(field)
  local value = rquote[field]
  if value ~= nil then return value end
  error("rquote has no field '" .. field .. "' (in lookup)", 2)
end

function rquote:__newindex(field, value)
  error("rquote has no field '" .. field .. "' (in assignment)", 2)
end

function std.newrquote(ast)
  assert(ast ~= nil)

  return setmetatable({
    ast = ast,
  }, rquote)
end

function std.is_rquote(x)
  return getmetatable(x) == rquote
end

function rquote:getast()
  return self.ast
end

function rquote:__tostring()
  return self.ast:tostring(true)
end

-- #####################################
-- ## Codegen Helpers
-- #################

std.type_meet = base.type_meet
std.fmax = base.fmax
std.fmin = base.fmin
std.quote_unary_op = base.quote_unary_op
std.quote_binary_op = base.quote_binary_op

-- #####################################
-- ## Types
-- #################

local arithmetic_combinators = {
  ["__add"] = function(a, b) return `([a] + [b]) end,
  ["__sub"] = function(a, b) return `([a] - [b]) end,
  ["__mul"] = function(a, b) return `([a] * [b]) end,
  ["__div"] = function(a, b) return `([a] / [b]) end,
  ["__mod"] = function(a, b) return `([a] % [b]) end,
}

local function generate_arithmetic_metamethod_body(ty, method, e1, e2)
  local combinator = arithmetic_combinators[method]
  if ty:isprimitive() then
    return combinator(e1, e2)
  elseif ty:isstruct() then
    if ty.metamethods[method] then
      return combinator(e1, e2)
    end
    local entries = ty:getentries():map(function(entry)
      return generate_arithmetic_metamethod_body(entry.type, method,
                                                 `(e1.[entry.field]),
                                                 `(e2.[entry.field]))
    end)
    return `([ty] { [entries] })
  elseif ty:isarray() then
    local entries = terralib.newlist()
    for idx = 0, ty.N - 1 do
      entries:insert(
        generate_arithmetic_metamethod_body(ty.type, method,
                                            `(e1[ [idx] ]),
                                            `(e2[ [idx] ])))
    end
    return `(arrayof([ty.type], [entries]))
  end
  assert(false)
end

function std.generate_arithmetic_metamethod(ty, method)
  local a = terralib.newsymbol(ty, "a")
  local b = terralib.newsymbol(ty, "b")
  local body = generate_arithmetic_metamethod_body(ty, method, a, b)
  return terra([a], [b]) : ty return [body] end
end

function std.generate_arithmetic_metamethods(ty)
  local methods = {}
  for method, _ in pairs(arithmetic_combinators) do
    methods[method] = std.generate_arithmetic_metamethod(ty, method)
  end
  return methods
end

function std.generate_arithmetic_metamethods_for_bounded_type(ty)
  local methods = {}
  for method, _ in pairs(arithmetic_combinators) do
    methods[method] = terralib.overloadedfunction(
      method,
      {
        terra(a : ty, b : ty) : ty.index_type
          return [generate_arithmetic_metamethod_body(ty.index_type, method, `(a.__ptr), `(b.__ptr))]
        end,
        terra(a : ty, b : ty.index_type) : ty.index_type
          return [generate_arithmetic_metamethod_body(ty.index_type, method, `(a.__ptr), b)]
        end,
        terra(a : ty.index_type, b : ty) : ty.index_type
          return [generate_arithmetic_metamethod_body(ty.index_type, method, a, `(b.__ptr))]
        end
      }
    )
  end
  return methods
end

local and_combinator = function(a, b) return `(([a]) and ([b])) end
local or_combinator = function(a, b) return `(([a]) or ([b])) end
local conditional_combinators = {
  ["__eq"] = { elem_comb = function(a, b) return `([a] == [b]) end,
               res_comb = and_combinator, },
  ["__ne"] = { elem_comb = function(a, b) return `([a] ~= [b]) end,
               res_comb = or_combinator, },
  ["__le"] = { elem_comb = function(a, b) return `([a] <= [b]) end,
               res_comb = and_combinator, },
  ["__lt"] = { elem_comb = function(a, b) return `([a] < [b]) end,
               res_comb = and_combinator, },
  ["__ge"] = { elem_comb = function(a, b) return `([a] >= [b]) end,
               res_comb = and_combinator, },
  ["__gt"] = { elem_comb = function(a, b) return `([a] > [b]) end,
               res_comb = and_combinator, },
}

local function generate_conditional_metamethod_body(ty, method, e1, e2)
  local combinators = conditional_combinators[method]
  if ty:isprimitive() then
    return combinators.elem_comb(e1, e2)
  elseif ty:isstruct() then
    local res
    local entries = ty:getentries():map(function(entry)
      local entry =
        generate_conditional_metamethod_body(entry.type, method,
                                            `(e1.[entry.field]),
                                            `(e2.[entry.field]))
      if not res then res = entry
      else res = combinators.res_comb(res, entry) end
    end)
    return res
  elseif ty:isarray() then
    local entries = terralib.newlist()
    local res
    for idx = 0, ty.N - 1 do
      local entry =
        generate_conditional_metamethod_body(ty.type, method,
                                            `(e1[ [idx] ]),
                                            `(e2[ [idx] ]))
      if not res then res = entry
      else res = combinators.res_comb(res, entry) end
    end
    return res
  end
  assert(false)
end

function std.generate_conditional_metamethod(ty, method)
  return macro(function(a, b)
    if a:gettype() == b:gettype() then
      return generate_conditional_metamethod_body(ty, method, a, b)
    elseif method == "__le" or method == "__lt" then
      local combinators = conditional_combinators[method]
      local lhs = generate_conditional_metamethod_body(ty, method, `([b].lo), a)
      local rhs = generate_conditional_metamethod_body(ty, method, a, `([b].hi))
      return combinators.res_comb(lhs, rhs)
    end
  end)
end

function std.generate_conditional_metamethods(ty)
  local methods = {}
  for method, _ in pairs(conditional_combinators) do
    methods[method] = std.generate_conditional_metamethod(ty, method)
  end
  return methods
end

function std.generate_conditional_metamethod_for_bounded_type(ty, method)
  return macro(function(a, b)
    if not std.is_rect_type(b:gettype()) then
      if std.is_bounded_type(a:gettype()) then a = `(a.__ptr) end
      if std.is_bounded_type(b:gettype()) then b = `(b.__ptr) end
      return generate_conditional_metamethod_body(ty.index_type, method, a, b)
    elseif method == "__le" or method == "__lt" then
      assert(std.is_bounded_type(a:gettype()))
      a = `(a.__ptr)
      local combinators = conditional_combinators[method]
      local lhs = generate_conditional_metamethod_body(ty.index_type, method, `([b].lo), a)
      local rhs = generate_conditional_metamethod_body(ty.index_type, method, a, `([b].hi))
      return combinators.res_comb(lhs, rhs)
    end
  end)
end

function std.generate_conditional_metamethods_for_bounded_type(ty)
  local methods = {}
  for method, _ in pairs(conditional_combinators) do
    methods[method] = std.generate_conditional_metamethod_for_bounded_type(ty, method)
  end
  return methods
end

-- WARNING: Bounded types are NOT unique. If two regions are aliased
-- then it is possible for two different pointer types to be equal:
--
-- var r = region(ispace(ptr, n), t)
-- var s = r
-- var x = new(ptr(t, r))
-- var y = new(ptr(t, s))
--
-- The types of x and y are distinct objects, but are still type_eq.
local bounded_type = terralib.memoize(function(index_type, ...)
  assert(std.is_index_type(index_type))
  local bounds = data.newtuple(...)
  local points_to_type = false
  if #bounds > 0 then
    if terralib.types.istype(bounds[1]) then
      points_to_type = bounds[1]
      bounds:remove(1)
    end
  end
  if #bounds <= 0 then
    error(tostring(index_type) .. " expected at least one ispace or region, got none")
  end
  for i, bound in ipairs(bounds) do
    if not std.is_symbol(bound) then
      local offset = 0
      if points_to_type then
        offset = offset + 1
      end
      error(tostring(index_type) .. " expected a symbol as argument " ..
              tostring(i+offset) .. ", got " .. tostring(bound))
    end
  end

  local st = terralib.types.newstruct(tostring(index_type))
  st.entries = terralib.newlist({
      { "__ptr", index_type },
  })
  if #bounds > 1 then
    -- Find the smallest bitmask that will fit.
    -- TODO: Would be nice to compress smaller than one byte.
   local bitmask_type
    if terralib.llvmversion >= 38 then
      if #bounds <= bit.lshift(1, 8) then
        bitmask_type = uint8
      elseif #bounds <= bit.lshift(1, 16) then
        bitmask_type = uint16
      -- XXX: What we really want here is bit.lshift(1ULL, 32),
      --      which is supported only in LuaJIT 2.1 or higher
      elseif #bounds <= bit.lshift(1, 30) then
        bitmask_type = uint32
      else
        assert(false) -- really?
      end
    else
      assert(#bounds <= bit.lshift(1, 30))
      bitmask_type = uint32
    end
    st.entries:insert({ "__index", bitmask_type })
  end

  st.is_bounded_type = true
  st.index_type = index_type
  st.points_to_type = points_to_type
  st.bounds_symbols = bounds
  st.dim = index_type.dim
  st.fields = index_type.fields

  function st:is_ptr()
    return self.points_to_type ~= false
  end

  function st:bounds()
    local bounds = data.newtuple()
    local is_ispace = false
    local is_region = false
    for i, bound_symbol in ipairs(self.bounds_symbols) do
      local bound = bound_symbol:gettype()
      if terralib.types.istype(bound) then
        bound = std.as_read(bound)
      end
      if not (terralib.types.istype(bound) and
              (bound == std.wild_type or std.is_ispace(bound) or std.is_region(bound)))
      then
        --report.error(nil, tostring(self.index_type) ..
        --            " expected an ispace or region as argument " ..
        --            tostring(i+1) .. ", got " .. tostring(bound))
        return nil, tostring(self.index_type) ..
                    " expected an ispace or region as argument " ..
                    tostring(i+1) .. ", got " .. tostring(bound)
      end
      if std.is_region(bound) and
        not (std.type_eq(bound.fspace_type, self.points_to_type) or
             (self.points_to_type:isvector() and
              std.type_eq(bound.fspace_type, self.points_to_type.type)) or
             std.is_unpack_result(self.points_to_type))
      then
        --report.error(nil, tostring(self.index_type) .. " expected region(" ..
        --            tostring(self.points_to_type) .. ") as argument " ..
        --            tostring(i+1) .. ", got " .. tostring(bound))
        return nil, tostring(self.index_type) .. " expected region(" ..
                    tostring(self.points_to_type) .. ") as argument " ..
                    tostring(i+1) .. ", got " .. tostring(bound)
      end
      if std.is_ispace(bound) then is_ispace = true end
      if std.is_region(bound) then is_region = true end
      bounds:insert(bound)
    end
    if is_ispace and is_region then
      --report.error(nil, tostring(self.index_type) .. " bounds may not mix ispaces and regions")
      return nil, tostring(self.index_type) .. " bounds may not mix ispaces and regions"
    end
    return bounds
  end

  st.metamethods.__eq = macro(function(a, b)
      assert(std.is_bounded_type(a:gettype()) and std.is_bounded_type(b:gettype()))
      assert(a.index_type == b.index_type)
      return `(a.__ptr.value == b.__ptr.value)
  end)

  st.metamethods.__ne = macro(function(a, b)
      assert(std.is_bounded_type(a:gettype()) and std.is_bounded_type(b:gettype()))
      assert(a.index_type == b.index_type)
      return `(a.__ptr.value ~= b.__ptr.value)
  end)

  function st.metamethods.__cast(from, to, expr)
    if std.is_bounded_type(from) then
      if std.validate_implicit_cast(from.index_type, to) then
        return `([to]([expr].__ptr))
      end
    end
    assert(false)
  end

  -- Important: This has to downgrade the type, because arithmetic
  -- isn't guarranteed to stay within bounds.
  for method_name, method in pairs(std.generate_arithmetic_metamethods_for_bounded_type(st)) do
    st.metamethods[method_name] = method
  end
  for method_name, method in pairs(std.generate_conditional_metamethods_for_bounded_type(st)) do
    st.metamethods[method_name] = method
  end

  terra st:to_point()
    return ([index_type](@self)):to_point()
  end

  terra st:to_domain_point()
    return ([index_type](@self)):to_domain_point()
  end

  function st:force_cast(from, to, expr)
    assert(std.is_bounded_type(from) and std.is_bounded_type(to) and
             (#(from:bounds()) > 1) == (#(to:bounds()) > 1))
    if #(to:bounds()) == 1 then
      return `([to]{ __ptr = [expr].__ptr })
    else
      return quote var x = [expr] in [to]{ __ptr = x.__ptr, __index = x.__index} end
    end
  end

  if false then -- std.config["debug"] then
    function st.metamethods:__typename()
      local bounds = self.bounds_symbols

      if self.points_to_type then
        return tostring(self.index_type) .. "(" .. tostring(self.points_to_type) .. ", " .. tostring(bounds:mkstring(", ")) .. " : " .. tostring(self:bounds():mkstring(", ")) .. ")"
      else
        return tostring(self.index_type) .. "(" .. tostring(bounds:mkstring(", ")) .. " : " .. tostring(self:bounds():mkstring(", ")) .. ")"
      end
    end
  else
    function st.metamethods:__typename()
      local bounds = self.bounds_symbols

      if self.points_to_type then
        return tostring(self.index_type) .. "(" .. tostring(self.points_to_type) .. ", " .. tostring(bounds:mkstring(", ")) .. ")"
      else
        return tostring(self.index_type) .. "(" .. tostring(bounds:mkstring(", ")) .. ")"
      end
    end
  end

  return st
end)

local function validate_index_base_type(base_type)
  assert(terralib.types.istype(base_type),
         "Index type expected a type, got " .. tostring(base_type))
  if std.type_eq(base_type, opaque) then
    return c.legion_ptr_t, 0, terralib.newlist({"value"}), terralib.sizeof(c.legion_ptr_t) * 8
  elseif std.type_eq(base_type, int32) or std.type_eq(base_type, int64) then
    return base_type, 1, false, terralib.sizeof(base_type) * 8
  elseif base_type:isstruct() then
    local entries = base_type:getentries()
    assert(#entries >= 1 and #entries <= 3,
           "Multi-dimensional index type expected 1 to 3 fields, got " ..
             tostring(#entries))
    local num_bits = nil
    for _, entry in ipairs(entries) do
      local field_type = entry[2] or entry.type
      assert(std.type_eq(field_type, int32) or std.type_eq(field_type, int64),
             "Multi-dimensional index type expected fields to be " .. tostring(int32) .. " or " ..
             tostring(int64) ..  ", got " .. tostring(field_type))
      if num_bits == nil then
        num_bits = terralib.sizeof(field_type) * 8
      else
        assert(num_bits == terralib.sizeof(field_type) * 8,
               "Multi-dimensional index type expected fields to have the same precision")
      end
    end
    return base_type, #entries, entries:map(function(entry) return entry[1] or entry.field end), num_bits
  else
    assert(false, "Index type expected " .. tostring(opaque) .. ", " ..
             tostring(int32) .. ", " .. tostring(int64) .. " or a struct, got " .. tostring(base_type))
  end
end

-- Hack: Terra uses getmetatable() in terralib.types.istype(), so
-- setting a custom metatable on a type requires some trickery. The
-- approach used here is to define __metatable() to return the
-- expected type metatable so that the object is recongized as a type.

local index_type = {}
do
  local st = terralib.types.newstruct()
  for k, v in pairs(getmetatable(st)) do
    index_type[k] = v
  end
  index_type.__call = bounded_type
  index_type.__metatable = getmetatable(st)
end

std.rect_type = terralib.memoize(function(index_type)
  local st = terralib.types.newstruct("rect" .. tostring(index_type.dim) .. "d")
  assert(not index_type:is_opaque())
  st.entries = terralib.newlist({
      { "lo", index_type },
      { "hi", index_type },
  })

  st.is_rect_type = true
  st.index_type = index_type
  st.dim = index_type.dim

  st.metamethods.__eq = macro(function(a, b)
    return `([a].lo == [b].lo and [a].hi == [b].hi)
  end)

  st.metamethods.__ne = macro(function(a, b)
    return `([a].lo ~= [b].lo and [a].hi ~= [b].hi)
  end)

  function st.metamethods.__cast(from, to, expr)
    if std.is_rect_type(from) then
      if std.type_eq(to, c["legion_rect_" .. tostring(st.dim) .. "d_t"]) then
        local ty = to.entries[1].type
        return `([to] { lo = [ty]([expr].lo),
                        hi = [ty]([expr].hi) })
      elseif std.type_eq(to, c.legion_domain_t) then
        return `([expr]:to_domain())
      end
    end
    assert(false)
  end

  terra st:to_domain()
    return [c["legion_domain_from_rect_" .. tostring(st.dim) .. "d"]](@self)
  end

  terra st:size()
    return self.hi - self.lo + [st.index_type:const(1)]
  end

  if index_type.fields then
    terra st:volume()
      var size = self:size()
      return [data.reduce(
        function(result, field) return `([result] * ([size].__ptr.[field])) end,
        index_type.fields, `(1))]
    end
  else
    terra st:volume()
      return int(self:size().__ptr)
    end
  end

  return st
end)
function std.index_type(base_type, displayname)
  local impl_type, dim, fields, num_bits = validate_index_base_type(base_type)

  local st = terralib.types.newstruct(displayname)
  st.entries = terralib.newlist({
      { "__ptr", impl_type },
  })

  st.is_index_type = true
  st.base_type = base_type
  st.impl_type = impl_type
  st.dim = dim
  st.fields = fields

  function st:is_opaque()
    return std.type_eq(self.base_type, opaque)
  end

  function st.metamethods.__cast(from, to, expr)
    if std.is_index_type(to) then
      if std.type_eq(from, c.legion_domain_point_t) then
        return `([to.from_domain_point]([expr]))
      elseif to:is_opaque() and std.validate_implicit_cast(from, int) then
        return `([to]{ __ptr = c.legion_ptr_t { value = [expr] } })
      elseif not to:is_opaque() and std.validate_implicit_cast(from, to.base_type) then
        return `([to]{ __ptr = [expr] })
      elseif to:is_opaque() and std.type_eq(from, c.legion_ptr_t) then
        return `([to]{ __ptr = expr })
      end
    elseif std.is_index_type(from) then
      if std.type_eq(to, c.legion_domain_point_t) then
        return `([expr]:to_domain_point())
      elseif from:is_opaque() then
        if std.validate_implicit_cast(int, to) then
          return `([to]([expr].__ptr.value))
        end
      else
        assert(not from:is_opaque())
        if std.type_eq(to, c["legion_point_" .. tostring(st.dim) .. "d_t"]) then
          return `([expr]:to_point())
        elseif std.validate_implicit_cast(from.base_type, to) then
          return `([to]([expr].__ptr))
        end
      end
    end
    assert(false)
  end

  function st:const(v)
    assert(self.dim >= 1)
    local fields = self.fields
    local pt = c["legion_point_" .. tostring(self.dim) .. "d_t"]

    if fields then
      return `(self { __ptr = [self.impl_type] { [fields:map(function(_) return v end)] } })
    else
      return `(self({ __ptr = [self.impl_type](v) }))
    end
  end

  function st:zero()
    return st:const(0)
  end

  function st:nil_index()
    return st:const(-(2^(num_bits - 1) - 1))
  end

  local function make_point(expr)
    local dim = data.max(st.dim, 1)
    local fields = st.fields
    local pt = c["legion_point_" .. tostring(dim) .. "d_t"]

    if fields then
      return quote
        var v = [expr].__ptr
      in
        pt { x = arrayof(c.coord_t, [fields:map(function(field) return `(v.[field]) end)]) }
      end
    else
      return quote var v = [expr].__ptr in pt { x = arrayof(c.coord_t, v) } end
    end
  end

  terra st:to_point()
    return [make_point(self)]
  end

  local function make_domain_point(expr)
    local index = terralib.newsymbol(st.impl_type)

    local values
    if st.fields then
      values = st.fields:map(function(field) return `(index.[field]) end)
    else
      values = terralib.newlist({index})
    end
    for _ = #values + 1, 3 do
      values:insert(0)
    end

    return quote
      var [index] = [expr].__ptr
    in
      c.legion_domain_point_t {
        dim = [data.max(st.dim, 1)],
        point_data = arrayof(c.coord_t, [values]),
      }
    end
  end

  terra st:to_domain_point()
    return [make_domain_point(self)]
  end

  -- Generate `from_domain_point` function.
  local function make_from_domain_point(pt_expr)
    local fields = st.fields

    local dimensionality_match_cond
    if st.dim <= 1 then  -- We regard 0-dim and 1-dim points as interchangeable.
      dimensionality_match_cond = `([pt_expr].dim <= 1)
    else
      dimensionality_match_cond = `([pt_expr].dim == [st.dim])
    end
    local error_message =
      "from_domain_point (" .. tostring(st) .. "): dimensionality mismatch"
    local nil_case = quote end
    if st.dim >= 1 then
      nil_case = quote
        if [pt_expr].dim == -1 then return [st:nil_index()] end
      end
    end
    if fields then
      return quote
        [nil_case]
        std.assert([dimensionality_match_cond], [error_message])
        return st { __ptr = [st.impl_type] {
          [data.mapi(function(i) return `([pt_expr].point_data[ [i-1] ]) end, st.fields)] } }
      end
    else
      return quote
        [nil_case]
        std.assert([dimensionality_match_cond], [error_message])
        return st { __ptr = [st.impl_type]([pt_expr].point_data[0]) }
      end
    end
  end
  terra st.from_domain_point(pt : c.legion_domain_point_t)
    [make_from_domain_point(pt)]
  end

  for method_name, method in pairs(std.generate_arithmetic_metamethods(st)) do
    st.metamethods[method_name] = method
  end
  for method_name, method in pairs(std.generate_conditional_metamethods(st)) do
    st.metamethods[method_name] = method
  end
  if not st:is_opaque() then
    st.metamethods.__mod = terralib.overloadedfunction(
      "__mod", {
        st.metamethods.__mod,
        terra(a : st, b : std.rect_type(st)) : st
          var sz = b:size()
          return (a + sz) % sz
        end
      })
  end

  -- Makes a Terra function that performs an operation element-wise on two
  -- values of this index type.
  local function make_element_wise_op(op)
    local fields = st.fields
    if fields then
      return terra(i1 : st, i2 : st)
        var p1, p2 = i1.__ptr, i2.__ptr
        return [st] { __ptr = [st.impl_type]
          { [fields:map(function(field) return `([op](p1.[field], p2.[field])) end)] } }
      end
    else
      return terra(i1 : st, i2 : st)
        return [st] { __ptr = [st.impl_type]([op](i1.__ptr, i2.__ptr)) }
      end
    end
  end
  -- Element-wise min and max.
  st.elem_min = make_element_wise_op(std.fmin)
  st.elem_max = make_element_wise_op(std.fmax)

  return setmetatable(st, index_type)
end

struct std.__int2d { x : int64, y : int64 }
struct std.__int3d { x : int64, y : int64, z : int64 }
std.ptr = std.index_type(opaque, "ptr")
std.int1d = std.index_type(int64, "int1d")
std.int2d = std.index_type(std.__int2d, "int2d")
std.int3d = std.index_type(std.__int3d, "int3d")

std.rect1d = std.rect_type(std.int1d)
std.rect2d = std.rect_type(std.int2d)
std.rect3d = std.rect_type(std.int3d)

do
  local next_ispace_id = 1
  function std.ispace(index_type)
    assert(terralib.types.istype(index_type) and std.is_index_type(index_type),
           "Ispace type requires index type")

    local st = terralib.types.newstruct("ispace")
    st.entries = terralib.newlist({
        { "impl", c.legion_index_space_t },
    })

    st.is_ispace = true
    st.index_type = index_type
    st.dim = index_type.dim

    function st:is_opaque()
      return self.index_type:is_opaque()
    end

    function st:force_cast(from, to, expr)
      assert(std.is_ispace(from) and std.is_ispace(to))
      return `([to] { impl = [expr].impl })
    end

    local id = next_ispace_id
    next_ispace_id = next_ispace_id + 1

    local hash_value = "__ispace_#" .. tostring(id)
    function st:hash()
      return hash_value
    end

    if std.config["debug"] then
      function st.metamethods.__typename(st)
        return "ispace#" .. tostring(id) .. "(" .. tostring(st.index_type) .. ")"
      end
    else
      function st.metamethods.__typename(st)
        return "ispace(" .. tostring(st.index_type) .. ")"
      end
    end

    return st
  end
end

do
  local next_region_id = 1
  function std.region(ispace_symbol, fspace_type)
    if fspace_type == nil then
      fspace_type = ispace_symbol
      ispace_symbol = std.newsymbol(std.ispace(std.ptr))
    end
    if terralib.types.istype(ispace_symbol) then
      ispace_symbol = std.newsymbol(ispace_symbol)
    end

    if not std.is_symbol(ispace_symbol) then
      error("region expected ispace as argument 1, got " .. tostring(ispace_symbol), 2)
    end
    if not terralib.types.istype(fspace_type) then
      error("region expected fspace as argument 2, got " .. tostring(fspace_type), 2)
    end
    if std.is_list_of_regions(fspace_type) then
      error("region expected fspace to not be a list, got " .. tostring(fspace_type), 2)
    end

    local st = terralib.types.newstruct("region")
    st.entries = terralib.newlist({
        { "impl", c.legion_logical_region_t },
    })

    st.is_region = true
    st.ispace_symbol = ispace_symbol
    st.fspace_type = fspace_type
    st.index_expr = false

    function st:ispace()
      local ispace = self.ispace_symbol:gettype()
      assert(terralib.types.istype(ispace) and
               std.is_ispace(ispace),
             "Parition type requires ispace")
      return ispace
    end

    function st:is_opaque()
      return self:ispace():is_opaque()
    end

    function st:fspace()
      return st.fspace_type
    end

    function st:has_index_expr()
      return st.index_expr
    end

    function st:get_index_expr()
      assert(st.index_expr)
      return st.index_expr
    end

    function st:set_index_expr(expr)
      assert(not st.index_expr)
      assert(ast.is_node(expr))
      st.index_expr = expr
    end

    -- For API compatibility with std.list:
    function st:list_depth()
      return 0
    end

    function st:force_cast(from, to, expr)
      assert(std.is_region(from) and std.is_region(to))
      return `([to] { impl = [expr].impl })
    end

    local id = next_region_id
    next_region_id = next_region_id + 1

    local hash_value = "__region_#" .. tostring(id)
    function st:hash()
      return hash_value
    end

    if std.config["debug"] then
      function st.metamethods.__typename(st)
        if st:is_opaque() then
          return "region#" .. tostring(id) .. "(" .. tostring(st.fspace_type) .. ")"
        else
          return "region#" .. tostring(id) .. "(" .. tostring((st.ispace_symbol:hasname() and st.ispace_symbol) or st:ispace()) .. ", " .. tostring(st.fspace_type) .. ")"
        end
      end
    else
      function st.metamethods.__typename(st)
        if st:is_opaque() then
          return "region(" .. tostring(st.fspace_type) .. ")"
        else
          return "region(" .. tostring((st.ispace_symbol:hasname() and st.ispace_symbol) or st:ispace()) .. ", " .. tostring(st.fspace_type) .. ")"
        end
      end
    end

    return st
  end
end

std.wild_type = terralib.types.newstruct("wild_type")
std.wild = std.newsymbol(std.wild_type, "wild")

std.disjoint = ast.disjointness_kind.Disjoint {}
std.aliased = ast.disjointness_kind.Aliased {}

do
  local next_partition_id = 1
  function std.partition(disjointness, region_symbol, colors_symbol)
    if colors_symbol == nil then
      colors_symbol = std.newsymbol(std.ispace(std.ptr))
    end
    if terralib.types.istype(colors_symbol) then
      colors_symbol = std.newsymbol(colors_symbol)
    end

    assert(disjointness:is(ast.disjointness_kind),
           "Partition type requires disjointness to be one of disjoint or aliased")
    assert(std.is_symbol(region_symbol),
           "Partition type requires region to be a symbol")
    if region_symbol:hastype() then
      assert(terralib.types.istype(region_symbol:gettype()) and
               std.is_region(region_symbol:gettype()),
             "Parition type requires region")
    end
    assert(std.is_symbol(colors_symbol),
           "Partition type requires colors to be a symbol")
    if colors_symbol:hastype() then
      assert(terralib.types.istype(colors_symbol:gettype()) and
               std.is_ispace(colors_symbol:gettype()),
             "Parition type requires colors")
    end

    local st = terralib.types.newstruct("partition")
    st.entries = terralib.newlist({
        { "impl", c.legion_logical_partition_t },
    })

    st.is_partition = true
    st.disjointness = disjointness
    st.parent_region_symbol = region_symbol
    st.colors_symbol = colors_symbol
    st.index_expr = false
    st.subregions = data.newmap()

    function st:is_disjoint()
      return self.disjointness:is(ast.disjointness_kind.Disjoint)
    end

    function st:partition()
      return self
    end

    function st:parent_region()
      local region = self.parent_region_symbol:gettype()
      assert(terralib.types.istype(region) and
               std.is_region(region),
             "Parition type requires region")
      return region
    end

    function st:colors()
      local colors = self.colors_symbol:gettype()
      assert(terralib.types.istype(colors) and
               std.is_ispace(colors),
             "Parition type requires colors")
      return colors
    end

    function st:fspace()
      return self:parent_region():fspace()
    end

    function st:has_index_expr()
      return st.index_expr
    end

    function st:get_index_expr()
      assert(st.index_expr)
      return st.index_expr
    end

    function st:set_index_expr(expr)
      assert(not st.index_expr)
      assert(ast.is_node(expr))
      st.index_expr = expr
    end

    function st:subregions_constant()
      return self.subregions
    end

    function st:subregion_constant(index_expr)
      local index = affine_helper.get_subregion_index(index_expr)
      if not self.subregions[index] then
        self.subregions[index] = self:subregion_dynamic()
        self.subregions[index]:set_index_expr(index_expr)
      end
      return self.subregions[index]
    end

    function st:subregion_dynamic()
      local parent = self:parent_region()
      return std.region(
        std.newsymbol(std.ispace(parent:ispace().index_type)),
        parent.fspace_type)
    end

    function st:force_cast(from, to, expr)
      assert(std.is_partition(from) and std.is_partition(to))
      return `([to] { impl = [expr].impl })
    end

    local id = next_partition_id
    next_partition_id = next_partition_id + 1

    local hash_value = "__partition_#" .. tostring(id)
    function st:hash()
      return hash_value
    end

    if std.config["debug"] then
      function st.metamethods.__typename(st)
        if st:colors():is_opaque() then
          return "partition#" .. tostring(id) .. "(" .. tostring(st.disjointness) .. ", " .. tostring(st.parent_region_symbol) .. ")"
        else
          return "partition#" .. tostring(id) .. "(" .. tostring(st.disjointness) .. ", " .. tostring(st.parent_region_symbol) .. ", " .. tostring((st.colors_symbol:hasname() and st.colors_symbol) or st:colors()) .. ")"
        end
      end
    else
      function st.metamethods.__typename(st)
        if st:colors():is_opaque() then
          return "partition(" .. tostring(st.disjointness) .. ", " .. tostring(st.parent_region_symbol) .. ")"
        else
          return "partition(" .. tostring(st.disjointness) .. ", " .. tostring(st.parent_region_symbol) .. ", " .. tostring((st.colors_symbol:hasname() and st.colors_symbol) or st:colors()) .. ")"
        end
      end
    end

    return st
  end
end

do
local next_cross_product_id = 1
function std.cross_product(...)
  local partition_symbols = terralib.newlist({...})
  assert(#partition_symbols >= 2, "Cross product type requires at least 2 arguments")
  for i, partition_symbol in ipairs(partition_symbols) do
    assert(std.is_symbol(partition_symbol),
           "Cross product type requires argument " .. tostring(i) .. " to be a symbol")
    if terralib.types.istype(partition_symbol:gettype()) then
      assert(std.is_partition(partition_symbol:gettype()),
             "Cross prodcut type requires argument " .. tostring(i) .. " to be a partition")
    end
  end

  local st = terralib.types.newstruct("cross_product")
  st.entries = terralib.newlist({
      { "impl", c.legion_logical_partition_t },
      { "product", c.legion_terra_index_cross_product_t },
      { "colors", c.legion_color_t[#partition_symbols] },
  })

  st.is_cross_product = true
  st.partition_symbols = data.newtuple(unpack(partition_symbols))
  st.index_expr = false
  st.subpartitions = data.newmap()

  function st:partitions()
    return self.partition_symbols:map(
      function(partition_symbol)
        local partition = partition_symbol:gettype()
        assert(terralib.types.istype(partition) and
                 std.is_partition(partition),
               "Cross product type requires partition")
        return partition
    end)
  end

  function st:partition(i)
    return self:partitions()[i or 1]
  end

  function st:fspace()
    return self:partition():fspace()
  end

  function st:is_disjoint()
    return self:partition():is_disjoint()
  end

  function st:parent_region()
    return self:partition():parent_region()
  end

    function st:has_index_expr()
      return st.index_expr
    end

    function st:get_index_expr()
      assert(st.index_expr)
      return st.index_expr
    end

    function st:set_index_expr(expr)
      assert(not st.index_expr)
      assert(ast.is_node(expr))
      st.index_expr = expr
    end

  function st:subregion_constant(i)
    local region_type = self:partition():subregion_constant(i)
    return region_type
  end

  function st:subregions_constant()
    return self:partition():subregions_constant()
  end

  function st:subregion_dynamic()
    local region_type = self:partition():subregion_dynamic()
    return region_type
  end

  function st:subpartition_constant(index_expr)
    local region_type = self:subregion_constant(index_expr)
    local index = affine_helper.get_subregion_index(index_expr)
    if not self.subpartitions[index] then
      local partition = st:subpartition_dynamic(region_type)
      self.subpartitions[index] = partition
      self.subpartitions[index]:set_index_expr(index_expr)
    end
    return self.subpartitions[index]
  end

  function st:subpartition_dynamic(region_type)
    region_type = region_type or self:subregion_dynamic()
    assert(std.is_region(region_type))
    local region_symbol = std.newsymbol(region_type)
    local partition = std.partition(self:partition(2).disjointness, region_symbol,
                                    self:partition(2).colors_symbol)
    if #partition_symbols > 2 then
      local partition_symbol = std.newsymbol(partition)
      local subpartition_symbols = terralib.newlist({partition_symbol})
      for i = 3, #partition_symbols do
        subpartition_symbols:insert(partition_symbols[i])
      end
      return std.cross_product(unpack(subpartition_symbols))
    else
      return partition
    end
  end

  function st:force_cast(from, to, expr)
    assert(std.is_cross_product(from) and std.is_cross_product(to))
    -- FIXME: Potential for double (triple) evaluation here.
    return `([to] { impl = [expr].impl, product = [expr].product, colors = [expr].colors })
  end

  local id = next_cross_product_id
  next_cross_product_id = next_cross_product_id + 1

  local hash_value = "__cross_product_#" .. tostring(id)
  function st:hash()
    return hash_value
  end

  function st.metamethods.__typename(st)
    return "cross_product(" .. st.partition_symbols:mkstring(", ") .. ")"
  end

  return st
end
end

std.vptr = terralib.memoize(function(width, points_to_type, ...)
  local bounds = data.newtuple(...)

  local vec = vector(int64, width)
  local struct legion_vptr_t {
    value : vec
  }
  local st = terralib.types.newstruct("vptr")
  st.entries = terralib.newlist({
      { "__ptr", legion_vptr_t },
  })

  local bitmask_type
  if #bounds > 1 then
    -- Find the smallest bitmask that will fit.
    -- TODO: Would be nice to compress smaller than one byte.
    if #bounds < bit.lshift(1, 8) - 1 and terralib.llvmversion >= 38 then
      bitmask_type = vector(uint8, width)
    elseif #bounds < bit.lshift(1, 16) - 1 then
      bitmask_type = vector(uint16, width)
    elseif #bounds < bit.lshift(1, 32) - 1 then
      bitmask_type = vector(uint32, width)
    else
      assert(false) -- really?
    end
    st.entries:insert({ "__index", bitmask_type })
  end

  st.is_vpointer = true
  st.points_to_type = points_to_type
  st.bounds_symbols = bounds
  st.N = width
  st.type = ptr(points_to_type, ...)
  st.impl_type = legion_vptr_t

  function st:bounds()
    local bounds = terralib.newlist()
    for i, region_symbol in ipairs(self.bounds_symbols) do
      local region = region_symbol:gettype()
      if not (terralib.types.istype(region) and std.is_region(region)) then
        --report.error(nil, "vptr expected a region as argument " .. tostring(i+1) ..
        --            ", got " .. tostring(region.type))
        return nil, "vptr expected a region as argument " .. tostring(i+1) ..
                    ", got " .. tostring(region.type)
      end
      if not std.type_eq(region.fspace_type, points_to_type) then
        --report.error(nil, "vptr expected region(" .. tostring(points_to_type) ..
        --            ") as argument " .. tostring(i+1) ..
        --            ", got " .. tostring(region))
        return nil, "vptr expected region(" .. tostring(points_to_type) ..
                    ") as argument " .. tostring(i+1) ..
                    ", got " .. tostring(region)
      end
      bounds:insert(region)
    end
    return bounds
  end

  function st.metamethods.__typename(st)
    local bounds = st.bounds_symbols

    return "vptr(" .. st.N .. ", " ..
           tostring(st.points_to_type) .. ", " ..
           tostring(bounds:mkstring(", ")) .. ")"
  end

  function st:isvector()
    return true
  end

  return st
end)

std.sov = terralib.memoize(function(struct_type, width)
  -- Sanity check that referee type is not a ref.
  assert(not std.is_ref(struct_type))
  assert(not std.is_rawref(struct_type))

  local st = terralib.types.newstruct("sov")
  st.entries = terralib.newlist()
  for _, entry in pairs(struct_type:getentries()) do
    local entry_field = entry[1] or entry.field
    local entry_type = entry[2] or entry.type
    if entry_type:isprimitive() then
      st.entries:insert{entry_field, vector(entry_type, width)}
    elseif entry_type:isarray() then
      st.entries:insert{entry_field, vector(entry_type.type, width)[entry_type.N]}
    else
      st.entries:insert{entry_field, std.sov(entry_type, width)}
    end
  end
  st.is_struct_of_vectors = true
  st.type = struct_type
  st.N = width

  function st.metamethods.__typename(st)
    return "sov(" .. tostring(st.type) .. ", " .. tostring(st.N) .. ")"
  end

  function st:isvector()
    return true
  end

  return st
end)

-- The ref type is a reference to a ptr type. Note that ref is
-- different from ptr in that it is not intended to be used by code;
-- it exists mainly to facilitate field-sensitive privilege checks in
-- the type system.
std.ref = terralib.memoize(function(pointer_type, ...)
  if not terralib.types.istype(pointer_type) then
    error("ref expected a type as argument 1, got " .. tostring(pointer_type))
  end
  if not (std.is_bounded_type(pointer_type) or std.is_ref(pointer_type)) then
    error("ref expected a bounded type or ref as argument 1, got " .. tostring(pointer_type))
  end
  if std.is_ref(pointer_type) then
    pointer_type = pointer_type.pointer_type
  end

  local st = terralib.types.newstruct("ref")

  st.is_ref = true
  st.pointer_type = pointer_type
  st.refers_to_type = pointer_type.points_to_type
  st.bounds_symbols = pointer_type.bounds_symbols
  st.field_path = data.newtuple(...)

  function st:bounds()
    return self.pointer_type:bounds()
  end

  if std.config["debug"] then
    function st.metamethods.__typename(st)
      local bounds = st.bounds_symbols

      return "ref(" .. tostring(st.refers_to_type) .. ", " .. tostring(bounds:mkstring(", ")) .. " : " .. tostring(st:bounds():mkstring(", ")) .. ", " .. tostring(st.field_path) .. ")"
    end
  else
    function st.metamethods.__typename(st)
      local bounds = st.bounds_symbols

      return "ref(" .. tostring(st.refers_to_type) .. ", " .. tostring(bounds:mkstring(", ")) .. ")"
    end
  end

  return st
end)

std.rawref = terralib.memoize(function(pointer_type)
  if not terralib.types.istype(pointer_type) then
    error("rawref expected a type as argument 1, got " .. tostring(pointer_type))
  end
  if not pointer_type:ispointer() then
    error("rawref expected a pointer type as argument 1, got " .. tostring(pointer_type))
  end
  -- Sanity check that referee type is not a ref.
  assert(not std.is_ref(pointer_type.type))

  local st = terralib.types.newstruct("rawref")

  st.is_rawref = true
  st.pointer_type = pointer_type
  st.refers_to_type = pointer_type.type

  function st.metamethods.__typename(st)
    return "rawref(" .. tostring(st.refers_to_type) .. ")"
  end

  return st
end)

std.future = terralib.memoize(function(result_type)
  if not terralib.types.istype(result_type) then
    error("future expected a type as argument 1, got " .. tostring(result_type))
  end
  assert(not std.is_rawref(result_type))

  local st = terralib.types.newstruct("future")
  st.entries = terralib.newlist({
      { "__result", c.legion_future_t },
  })

  st.is_future = true
  st.result_type = result_type

  function st.metamethods.__typename(st)
    return "future(" .. tostring(st.result_type) .. ")"
  end

  return st
end)

do
local next_list_id = 1
std.list = terralib.memoize(function(element_type, partition_type, privilege_depth, region_root, shallow, barrier_depth)
  if not terralib.types.istype(element_type) then
    error("list expected a type as argument 1, got " .. tostring(element_type))
  end

  if partition_type and not std.is_partition(partition_type) then
    error("list expected a partition type as argument 2, got " .. tostring(partition_type))
  end

  if privilege_depth and type(privilege_depth) ~= "number" then
    error("list expected a number as argument 3, got " .. tostring(privilege_depth))
  end

  if region_root and not std.is_region(region_root) then
    error("list expected a region type as argument 4, got " .. tostring(region_root))
  end

  if shallow and not type(shallow) == "boolean" then
    error("list expected a boolean as argument 5, got " .. tostring(shallow))
  end

  if barrier_depth and type(barrier_depth) ~= "number" then
    error("list expected a number as argument 3, got " .. tostring(barrier_depth))
  end

  if region_root and privilege_depth and privilege_depth ~= 0 then
    error("list privilege depth and region root are mutually exclusive")
  end

  local st = terralib.types.newstruct("list")
  st.entries = terralib.newlist({
      { "__size", uint64 }, -- in elements
      { "__data", &opaque },
  })

  st.is_list = true
  st.element_type = element_type
  st.partition_type = partition_type or false
  st.privilege_depth = privilege_depth or 0
  st.region_root = region_root or false
  st.shallow = shallow or false
  st.barrier_depth = barrier_depth or false

  function st:is_list_of_regions()
    return std.is_region(self.element_type) or
      std.is_list_of_regions(self.element_type)
  end

  function st:is_list_of_partitions()
    return std.is_partition(self.element_type) or
      std.is_list_of_partitions(self.element_type)
  end

  function st:is_list_of_phase_barriers()
    return std.is_phase_barrier(self.element_type) or
      std.is_list_of_phase_barriers(self.element_type)
  end

  function st:partition()
    return self.partition_type
  end

  function st:list_depth()
    if std.is_list(self.element_type) then
      return 1 + self.element_type:list_depth()
    else
      return 1
    end
  end

  function st:leaf_element_type()
    if std.is_list(self.element_type) then
      return self.element_type:leaf_element_type()
    end
    return self.element_type
  end

  function st:base_type()
    if std.is_list(self.element_type) then
      return self.element_type:base_type()
    end
    return self.element_type
  end

  function st:ispace()
    assert(std.is_list_of_regions(self))
    return self:base_type():ispace()
  end

  function st:fspace()
    assert(std.is_list_of_regions(self) or std.is_list_of_partitions(self))
    return self:base_type():fspace()
  end

  function st:subregion_dynamic()
    assert(std.is_list_of_regions(self))
    local ispace = std.newsymbol(
      std.ispace(self:ispace().index_type),
      self:base_type().ispace_symbol:hasname())
    return std.region(ispace, self:fspace())
  end

  function st:subpartition_dynamic()
    assert(std.is_list_of_partitions(self))
    return std.partition(
      self:base_type().disjointness, self:base_type().parent_region_symbol)
  end

  function st:slice(strip_levels)
    if strip_levels == nil then strip_levels = 0 end
    if std.is_list_of_regions(self) then
      local slice_type = self:subregion_dynamic()
      for i = 1 + strip_levels, self:list_depth() do
        slice_type = std.list(
          slice_type, self:partition(), self.privilege_depth, self.region_root, self.shallow)
      end
      return slice_type
    elseif std.is_list_of_partitions(self) then
      local slice_type = self:subpartition_dynamic()
      for i = 1 + strip_levels, self:list_depth() do
        slice_type = std.list(
          slice_type, self:partition(), self.privilege_depth, self.region_root, self.shallow)
      end
      return slice_type
    else
      assert(false)
    end
  end

  -- FIXME: Make the compiler manage cleanups, including lists.

  function st:data(value)
    return `([&self.element_type]([value].__data))
  end

  if not std.is_list(element_type) then
    terra st:num_leaves() : uint64
      return self.__size
    end
  else
    terra st:num_leaves() : uint64
      var sum : uint64 = 0
      for i = 0, self.__size do
        sum = sum + [st:data(self)][i]:num_leaves()
      end
      return sum
    end
  end

  local id = next_list_id
  next_list_id = next_list_id + 1

  local hash_value = "__list_#" .. tostring(id)
  function st:hash()
    return hash_value
  end

  function st:force_cast(from, to, expr)
    assert(std.is_list_of_regions(from) and std.is_list_of_regions(to))
    -- FIXME: This would result in memory corruption if we ever freed
    -- the original data.
    if to:partition() then
      assert(from:partition())
      return `([to] {
          __size = [expr].__size,
          __data = [expr].__data,
        })
    else
      return `([to] { __size = [expr].__size, __data = [expr].__data })
    end
  end

  if std.config["debug"] then
    function st.metamethods.__typename(st)
      return "list(" .. tostring(st.element_type) .. ", " .. tostring(st.partition_type) .. ", " ..
        tostring(st.privilege_depth) .. ", " .. tostring(st.region_root) .. ", " .. tostring(st.shallow) .. ")"
    end
  else
    function st.metamethods.__typename(st)
      return "list(" .. tostring(st.element_type) .. ")"
    end
  end

  return st
end)
end

do
  local st = terralib.types.newstruct("phase_barrier")
  std.phase_barrier = st
  st.entries = terralib.newlist({
      { "impl", c.legion_phase_barrier_t },
  })

  st.is_phase_barrier = true

  -- For API compatibility with std.list:
  function st:list_depth()
    return 0
  end
end

std.dynamic_collective = terralib.memoize(function(result_type)
  if not terralib.types.istype(result_type) then
    error("dynamic_collective expected a type as argument 1, got " .. tostring(result_type))
  end
  assert(not std.is_rawref(result_type))

  local st = terralib.types.newstruct("dynamic_collective")
  st.entries = terralib.newlist({
      { "impl", c.legion_dynamic_collective_t },
  })

  st.is_dynamic_collective = true
  st.result_type = result_type

  function st.metamethods.__typename(st)
    return "dynamic_collective(" .. tostring(st.result_type) .. ")"
  end

  return st
end)

do
  local function field_name(field)
    local field_name = field["field"] or field[1]
    if terralib.issymbol(field_name) then
      return field_name.displayname
    else
      return field_name
    end
  end

  local function field_type(field)
    return field["type"] or field[2]
  end

  function std.ctor_named(fields)
    local st = terralib.types.newstruct()
    st.entries = fields
    st.is_ctor = true
    st.metamethods.__cast = function(from, to, expr)
      if to:isstruct() then
        local from_fields = {}
        for _, from_field in ipairs(from:getentries()) do
          from_fields[field_name(from_field)] = field_type(from_field)
        end
        local mapping = terralib.newlist()
        for _, to_field in ipairs(to:getentries()) do
          local to_field_name = field_name(to_field)
          local to_field_type = field_type(to_field)
          local from_field_type = from_fields[to_field_name]
          if not from_field_type then
            error("type mismatch: ctor cast missing field " .. tostring(to_field_name))
          end
          mapping:insert({from_field_type, to_field_type, to_field_name})
        end

        local v = terralib.newsymbol(from)
        local fields = mapping:map(
          function(field_mapping)
            local from_field_type, to_field_type, to_field_name = unpack(
              field_mapping)
            return std.implicit_cast(
              from_field_type, to_field_type, `([v].[to_field_name]))
          end)

        return quote var [v] = [expr] in [to]({ [fields] }) end
      else
        error("ctor must cast to a struct")
      end
    end
    return st
  end

  function std.ctor_tuple(fields)
    local st = terralib.types.newstruct()
    st.entries = terralib.newlist()
    for i, field in ipairs(fields) do
      st.entries:insert({"_" .. tostring(i), field})
    end
    st.is_ctor = true
    st.metamethods.__cast = function(from, to, expr)
      if to:isstruct() then
        local from_fields = {}
        for i, from_field in ipairs(from:getentries()) do
          from_fields[i] = field_type(from_field)
        end
        local mapping = terralib.newlist()
        for i, to_field in ipairs(to:getentries()) do
          local to_field_type = field_type(to_field)
          local from_field_type = from_fields[i]
          if not from_field_type then
            error("type mismatch: ctor cast has insufficient fields")
          end
          mapping:insert({from_field_type, to_field_type, i})
        end

        local v = terralib.newsymbol(from)
        local fields = mapping:map(
          function(field_mapping)
            local from_field_type, to_field_type, i = unpack(
              field_mapping)
            return std.implicit_cast(
              from_field_type, to_field_type, `([v].["_" .. tostring(i)]))
          end)

        return quote var [v] = [expr] in [to]({ [fields] }) end
      else
        error("ctor must cast to a struct")
      end
    end
    return st
  end
end

do
  -- Wrapper for a null-terminated string.
  local st = terralib.types.newstruct("string")
  st.entries = terralib.newlist({
      { "impl", rawstring },
  })
  st.is_string = true

  function st.metamethods.__cast(from, to, expr)
    if std.is_string(to) then
      if std.type_eq(from, rawstring) then
        return `([to]{ impl = [expr] })
      end
    elseif std.type_eq(to, rawstring) then
      if std.is_string(from) then
        return `([expr].impl)
      end
    end
    assert(false)
  end
  std.string = st
end

-- #####################################
-- ## Tasks
-- #################

std.initial_regent_task_id = base.initial_regent_task_id

std.new_variant = base.new_variant
std.is_variant = base.is_variant

std.new_task = base.new_task
std.is_task = base.is_task

function base.task:printpretty()
  print(pretty.entry(self:get_primary_variant():get_ast()))
end

-- Calling convention:
std.convention = {}
std.convention.regent = ast.convention.Regent {}

function std.convention.manual(params)
  if not params then
    params = ast.convention_kind.Padded {}
  end
  return ast.convention.Manual { params = params }
end

function std.convention.is_manual(convention)
  return convention:is(ast.convention.Manual)
end

function base.task:set_calling_convention(convention)
  assert(not self.calling_convention)
  assert(convention:is(ast.convention))

  if std.convention.is_manual(convention) then
    if #self:get_variants() > 0 then
      error("manual calling convention can only be used with empty task")
    end
    -- Ok
  elseif not convention == std.convention.regent then
    if #self:get_variants() == 0 then
      error("regent calling convention cannot be used with empty task")
    end
    -- Ok
  else
    error("unrecognized calling convention " .. tostring(convention:type()))
  end

  self.calling_convention = convention
end

-- #####################################
-- ## Fspaces
-- #################

local fspace = {}
fspace.__index = fspace

fspace.__call = terralib.memoize(function(fs, ...)
  -- Do NOT attempt to access fs.params or fs.fields; they are not ready yet.

  local args = data.newtuple(...)
  -- Complain early if args are not symbols, but don't check types
  -- yet, since they may not be there at this point.
  for i, arg in ipairs(args) do
    if not std.is_symbol(arg) then
      error("expected a symbol as argument " .. tostring(i) .. ", got " .. tostring(arg))
    end
  end

  local st = terralib.types.newstruct(fs.name)
  st.is_fspace_instance = true
  st.fspace = fs
  st.args = args

  function st:getparams()
    return rawget(self, "params") or self.fspace.params
  end

  function st:getconstraints()
    st:getentries() -- Computes constraints as well.
    local constraints = rawget(self, "__constraints")
    assert(constraints)
    return constraints
  end

  function st.metamethods.__getentries(st)
    local params = st:getparams()
    local fields = rawget(st, "fields") or fs.fields
    local constraints = rawget(st, "constraints") or fs.constraints
    assert(params and fields, "Attempted to complete fspace too early.")

    std.validate_args(fs.node, params, args, false, terralib.types.unit, {}, true)

    local entries, st_constraints = std.validate_fields(fields, constraints, params, args)
    st.__constraints = st_constraints
    return entries
  end

  function st:force_cast(from, to, expr)
    if from:ispointer() then
      from = from.type
    end
    assert(std.is_fspace_instance(from) and std.is_fspace_instance(to) and
             from.fspace == to.fspace)

    local v = terralib.newsymbol((`expr):gettype())
    local fields = terralib.newlist()
    for i, to_field in ipairs(to:getentries()) do
      local from_field = from:getentries()[i]

      fields:insert(
        std.implicit_cast(from_field.type, to_field.type, `(v.[to_field.field])))
    end

    return quote var [v] = [expr] in [to]({ [fields] }) end
  end

  function st.metamethods.__typename(st)
    return st.fspace.name .. "(" .. st.args:mkstring(", ") .. ")"
  end

  return st
end)

function std.newfspace(node, name, has_params)
  local fs = setmetatable({node = node, name = name}, fspace)
  if not has_params then
    fs = fs()
  end
  return fs
end

-- #####################################
-- ## Main
-- #################

local projection_functors = terralib.newlist()

do
  local next_id = 1
  function std.register_projection_functor(depth, region_functor, partition_functor)
    local id = next_id
    next_id = next_id + 1

    projection_functors:insert(terralib.newlist({id, depth, region_functor, partition_functor}))

    return id
  end
end

local variants = terralib.newlist()

function std.register_variant(variant)
  assert(std.is_variant(variant))
  variants:insert(variant)
end

local max_dim = 3 -- Maximum dimension of an index space supported in Regent

local function make_ordering_constraint(layout, dim)
  -- This code would need to be taught about higher dimensions if the
  -- maximum dimension were ever increased.
  assert(max_dim == 3)
  assert(dim >= 1 and dim <= max_dim)

  local result = terralib.newlist()

  -- SOA, Fortran array order
  local dims = terralib.newsymbol(c.legion_dimension_kind_t[dim+1], "dims")
  result:insert(quote var [dims] end)
  result:insert(quote dims[0] = c.DIM_X end)
  if dim >= 2 then result:insert(quote dims[1] = c.DIM_Y end) end
  if dim >= 3 then result:insert(quote dims[2] = c.DIM_Z end) end
  result:insert(quote dims[ [dim] ] = c.DIM_F end)
  result:insert(quote c.legion_layout_constraint_set_add_ordering_constraint([layout], [dims], [dim+1], true) end)

  return result
end

local function make_normal_layout(dim)
  local layout_id = terralib.newsymbol(c.legion_layout_constraint_id_t, "layout_id")
  return layout_id, quote
    var layout = c.legion_layout_constraint_set_create()

    -- SOA, Fortran array order
    [make_ordering_constraint(layout, dim)]

    -- Normal instance
    c.legion_layout_constraint_set_add_specialized_constraint(
      layout, c.NORMAL_SPECIALIZE, 0)

    var [layout_id] = c.legion_layout_constraint_set_preregister(layout, "SOA")
    c.legion_layout_constraint_set_destroy(layout)
  end
end

local function make_virtual_layout()
  local layout_id = terralib.newsymbol(c.legion_layout_constraint_id_t, "layout_id")
  return layout_id, quote
    var layout = c.legion_layout_constraint_set_create()

    -- Virtual instance
    c.legion_layout_constraint_set_add_specialized_constraint(
      layout, c.VIRTUAL_SPECIALIZE, 0)

    var [layout_id] = c.legion_layout_constraint_set_preregister(layout, "virtual")
    c.legion_layout_constraint_set_destroy(layout)
  end
end

local function make_unconstrained_layout()
  local layout_id = terralib.newsymbol(c.legion_layout_constraint_id_t, "layout_id")
  return layout_id, quote
    var layout = c.legion_layout_constraint_set_create()

    c.legion_layout_constraint_set_add_specialized_constraint(
      layout, c.NO_SPECIALIZE, 0)

    var [layout_id] = c.legion_layout_constraint_set_preregister(layout, "unconstrained")
    c.legion_layout_constraint_set_destroy(layout)
  end
end

local function make_reduction_layout(dim, op_id)
  local layout_id = terralib.newsymbol(c.legion_layout_constraint_id_t, "layout_id")
  return layout_id, quote
    var layout = c.legion_layout_constraint_set_create()

    -- SOA, Fortran array order
    [make_ordering_constraint(layout, dim)]

    -- Reduction fold instance
    c.legion_layout_constraint_set_add_specialized_constraint(
      layout, c.REDUCTION_FOLD_SPECIALIZE, [op_id])

    var [layout_id] = c.legion_layout_constraint_set_preregister(layout, ["SOA(" .. tostring(op_id) .. ")"])
    c.legion_layout_constraint_set_destroy(layout)
  end
end

function std.setup(main_task, extra_setup_thunk, task_wrappers, registration_name)
  assert(not main_task or std.is_task(main_task))

  if not registration_name then
    registration_name = "main"
  end

  local reduction_registrations = terralib.newlist()
  for _, op in ipairs(base.reduction_ops) do
    for _, op_type in ipairs(base.reduction_types) do
      local register = c["register_reduction_" .. op.name .. "_" .. tostring(op_type)]
      local op_id = std.reduction_op_ids[op.op][op_type]
      reduction_registrations:insert(
        quote
          [register](op_id)
        end)
    end
  end

  local layout_registrations = terralib.newlist()
  local layout_normal = data.newmap()
  do
    for dim = 1, max_dim do
      local layout_id, layout_actions = make_normal_layout(dim)
      layout_registrations:insert(layout_actions)
      layout_normal[dim] = layout_id
    end
  end

  local layout_virtual
  do
    local layout_id, layout_actions = make_virtual_layout()
    layout_registrations:insert(layout_actions)
    layout_virtual = layout_id
  end

  local layout_unconstrained
  do
    local layout_id, layout_actions = make_unconstrained_layout()
    layout_registrations:insert(layout_actions)
    layout_unconstrained = layout_id
  end

  local layout_reduction = data.new_recursive_map(2)
  for dim = 1, max_dim do
    for _, op in ipairs(base.reduction_ops) do
      for _, op_type in ipairs(base.reduction_types) do
        local op_id = std.reduction_op_ids[op.op][op_type]
        local layout_id, layout_actions = make_reduction_layout(dim, op_id)
        layout_registrations:insert(layout_actions)
        layout_reduction[dim][op.op][op_type] = layout_id
      end
    end
  end

  local projection_functor_registrations = projection_functors:map(
    function(args)
      local id, depth, region_functor, partition_functor = unpack(args)
      -- Hack: Work around Terra not wanting to escape nil.
      if region_functor and partition_functor then
        return quote
          c.legion_runtime_preregister_projection_functor(
            id, depth, region_functor, partition_functor)
        end
      elseif region_functor then
        return quote
          c.legion_runtime_preregister_projection_functor(
            id, depth, region_functor, nil)
        end
      elseif partition_functor then
        return quote
          c.legion_runtime_preregister_projection_functor(
            id, depth, nil, partition_functor)
        end
      else
        assert(false)
      end
    end)

  local task_registrations = variants:map(
    function(variant)
      local task = variant.task

      local options = variant:get_config_options()

      local proc_types = {c.LOC_PROC, c.IO_PROC}

      -- Check if this is an OpenMP task.
      local openmp = false
      ast.traverse_node_postorder(
        function(node)
          if node:is(ast.typed.stat) and
            node.annotations.openmp:is(ast.annotation.Demand)
          then
            openmp = true
          end
        end,
        variant:has_ast() and variant:get_ast())
      if openmp then
        if std.config["openmp-strict"] then
          proc_types = {c.OMP_PROC}
        else
          proc_types = {c.OMP_PROC, c.LOC_PROC}
        end
      end

      -- Check if this is a GPU task.
      if variant:is_cuda() then proc_types = {c.TOC_PROC} end
      if std.config["cuda"] and task:is_shard_task() then
        proc_types[#proc_types + 1] = c.TOC_PROC
      end

      local layout_constraints = terralib.newsymbol(
        c.legion_task_layout_constraint_set_t, "layout_constraints")
      local layout_constraint_actions = terralib.newlist()
      if std.config["layout-constraints"] then
        local fn_type = task:get_type()
        local param_types = fn_type.parameters
        local region_i = 0
        for _, param_i in ipairs(std.fn_param_regions_by_index(fn_type)) do
          local param_type = param_types[param_i]
          local dim = data.max(param_type:ispace().dim, 1)
          local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
            std.find_task_privileges(param_type, task)
          for i, privilege in ipairs(privileges) do
            local field_types = privilege_field_types[i]

            local layout = layout_normal[dim]
            if std.is_reduction_op(privilege) then
              local op = std.get_reduction_op(privilege)
              assert(#field_types == 1)
              local field_type = field_types[1]
              layout = layout_reduction[dim][op][field_type]
            end
            if options.inner or variant:is_external() then
              -- No layout constraints for inner tasks
              layout = layout_unconstrained
            end
            layout_constraint_actions:insert(
              quote
                c.legion_task_layout_constraint_set_add_layout_constraint(
                  [layout_constraints], [region_i], [layout])
              end)
            region_i = region_i + 1
          end
        end
      end

      local registration_actions = terralib.newlist()
      for _, proc_type in ipairs(proc_types) do
        registration_actions:insert(quote
          var execution_constraints = c.legion_execution_constraint_set_create()
          c.legion_execution_constraint_set_add_processor_constraint(execution_constraints, proc_type)
          var [layout_constraints] = c.legion_task_layout_constraint_set_create()
          [layout_constraint_actions]
          var options = c.legion_task_config_options_t {
            leaf = [ options.leaf and std.config["legion-leaf"] ],
            inner = [ options.inner and std.config["legion-inner"] ],
            idempotent = options.idempotent,
          }

          c.legion_runtime_preregister_task_variant_fnptr(
            [task:get_task_id()],
            [task:get_name():concat(".")],
            execution_constraints, layout_constraints, options,
            [task_wrappers[variant:wrapper_name()]], nil, 0)
          c.legion_execution_constraint_set_destroy(execution_constraints)
          c.legion_task_layout_constraint_set_destroy(layout_constraints)
        end)
      end

      return quote
        [registration_actions]
      end
    end)
  local cuda_setup = quote end
  if std.config["cuda"] and cudahelper.check_cuda_available() then
    cudahelper.link_driver_library()
    local all_kernels = {}
    variants:map(function(variant)
      if variant:is_cuda() then
        local kernels = variant:get_cuda_kernels()
        if kernels ~= nil then
          for k, v in pairs(kernels) do
            all_kernels[k] = v
          end
        end
      end
    end)
    cuda_setup = cudahelper.jit_compile_kernels_and_register(all_kernels)
  end

  local extra_setup = quote end
  if extra_setup_thunk then
    extra_setup = quote
      [extra_setup_thunk]()
    end
  end

  local argc = terralib.newsymbol(int, "argc")
  local argv = terralib.newsymbol(&rawstring, "argv")

  local main_setup = quote end
  if main_task then
    main_setup = quote
      c.legion_runtime_set_top_level_task_id([main_task:get_task_id()])
      return c.legion_runtime_start([argc], [argv], false)
    end
  end

  local terra main([argc], [argv])
    [reduction_registrations];
    [layout_registrations];
    [projection_functor_registrations];
    [task_registrations];
    [cuda_setup];
    [extra_setup];
    [main_setup]
  end
  main:setname(registration_name)

  local names = {[registration_name] = main}
  return main, names
end

-- Generate all task wrappers in this process, the compiler will pick them up
-- automatically.
local function make_task_wrappers()
  local task_wrappers = {}
  for _,variant in ipairs(variants) do
    task_wrappers[variant:wrapper_name()] = variant:make_wrapper()
  end
  return task_wrappers
end

local struct Pipe {
  read_end : int,
  write_end : int,
}

local terra make_pipe()
  var fd : int[2]
  var res = c.pipe(fd)
  if res ~= 0 then
    c.perror('pipe creation failed')
    c.exit(c.EXIT_FAILURE)
  end
  return Pipe{ read_end = fd[0], write_end = fd[1] }
end

terra Pipe:close_read_end()
  c.close(self.read_end)
end

terra Pipe:close_write_end()
  c.close(self.write_end)
end

terra Pipe:write_int(x : int)
  var bytes_written = c.write(self.write_end, &x, sizeof(int))
  if bytes_written ~= sizeof(int) then
    c.perror('pipe: int write error')
    c.exit(c.EXIT_FAILURE)
  end
end

terra Pipe:read_int() : int
  var x : int
  var bytes_read = c.read(self.read_end, &x, sizeof(int))
  if bytes_read ~= sizeof(int) then
    c.perror('pipe: int read error')
    c.exit(c.EXIT_FAILURE)
  end
  return x
end

-- String can be up to 255 characters long.
terra Pipe:write_string(str : &int8)
  var len = c.strlen(str)
  if len >= 256 then
    var stderr = c.fdopen(2, "w")
    c.fprintf(stderr, 'pipe: string too long for writing')
    c.fflush(stderr)
    c.exit(c.EXIT_FAILURE)
  end
  var bytes_written = c.write(self.write_end, str, len + 1)
  if bytes_written ~= len + 1 then
    c.perror('pipe: string write error')
    c.exit(c.EXIT_FAILURE)
  end
end

-- Returned pointer must be manually free'd.
terra Pipe:read_string() : &int8
  var buf : int8[256]
  var bytes_read = c.read(self.read_end, &(buf[0]), 256)
  if bytes_read <= 0 then
    c.perror('pipe: string read error')
    c.exit(c.EXIT_FAILURE)
  end
  var len = c.strlen(buf)
  if len >= 256 then
    c.perror('pipe: string read error')
    c.exit(c.EXIT_FAILURE)
  end
  var str = [&int8](c.malloc(len + 1))
  c.strncpy(str, buf, len + 1)
  return str
end

local function compile_tasks_in_parallel()
  -- Force codegen; the main process will need to codegen later anyway, so we
  -- might as well do it now and not duplicate the work on the children.
  for _,variant in ipairs(variants) do
    variant.task:complete()
  end

  -- Don't spawn extra processes if jobs == 1.
  local num_slaves = math.max(tonumber(std.config["jobs"]) or 1, 1)
  if num_slaves == 1 then
    return terralib.newlist(), make_task_wrappers()
  end

  -- Spawn slave processes & distribute work to them on demand.
  -- TODO: Terra functions used by more than one task may get compiled
  -- multiple times, and included in multiple object files by different
  -- children. This will cause bloat in the final executable.
  local pclog = log.make_logger('paral_compile')
  local objfiles = terralib.newlist()
  local slave_pids = terralib.newlist()
  local slave_pipes = terralib.newlist()
  local slave2master = make_pipe()
  for slave_id = 1,num_slaves do
    pclog:info('master: spawning slave ' .. slave_id)
    local master2slave = make_pipe()
    slave_pipes:insert(master2slave)
    local pid = c.fork()
    assert(pid >= 0, 'fork failed')
    if pid == 0 then
      slave2master:close_read_end()
      master2slave:close_write_end()
      while true do
        pclog:info('slave ' .. slave_id .. ': signaling master to send work')
        slave2master:write_int(slave_id)
        local variant_id = master2slave:read_int()
        assert(0 <= variant_id and variant_id <= #variants,
               'slave ' .. slave_id .. ': variant id read error')
        if variant_id == 0 then
          pclog:info('slave ' .. slave_id .. ': stopping')
          break
        end
        local raw_filename = master2slave:read_string()
        local filename = ffi.string(raw_filename)
        c.free(raw_filename)
        local variant = variants[variant_id]
        pclog:info('slave ' .. slave_id .. ': compiling ' ..
                   tostring(variant) .. ' to file ' .. filename)
        local exports = {}
        exports[variant:wrapper_name()] = variant:make_wrapper()
        profile('compile', variant, function()
          terralib.saveobj(filename, 'object', exports)
        end)()
      end
      slave2master:close_write_end()
      master2slave:close_read_end()
      os.exit(c.EXIT_SUCCESS)
    else
      pclog:info('master: slave ' .. slave_id .. ' spawned as pid ' .. pid)
      slave_pids:insert(pid)
      master2slave:close_read_end()
    end
  end
  slave2master:close_write_end()
  local next_variant = 1
  local num_stopped = 0
  while num_stopped < num_slaves do
    pclog:info('master: waiting for next available slave')
    local slave_id = slave2master:read_int()
    assert(1 <= slave_id and slave_id <= num_slaves,
           'master: slave id read error')
    local master2slave = slave_pipes[slave_id]
    if next_variant <= #variants then
      local objfile = os.tmpname()
      objfiles:insert(objfile)
      pclog:info('master: assigning ' .. tostring(variants[next_variant]) ..
                 ' to slave ' .. slave_id .. ', to be compiled to ' .. objfile)
      master2slave:write_int(next_variant)
      master2slave:write_string(objfile)
      next_variant = next_variant + 1
    else
      pclog:info('master: sending stop command to slave ' .. slave_id)
      master2slave:write_int(0)
      master2slave:close_write_end()
      num_stopped = num_stopped + 1
    end
  end
  for slave_id,pid in ipairs(slave_pids) do
    pclog:info('master: waiting for slave ' .. slave_id .. ' to finish')
    c.waitpid(pid, nil, 0)
  end
  slave2master:close_read_end()

  -- Declare all task wrappers using a (fake) header file, so the compiler will
  -- expect them to be linked-in later.
  local header = [[
#include "legion.h"
#include "legion_terra.h"
#include "legion_terra_partitions.h"
]] ..
  table.concat(
    variants:map(function(variant) return variant:wrapper_sig() .. '\n' end)
  )
  local task_wrappers = terralib.includecstring(header)

  return objfiles,task_wrappers
end

function std.start(main_task, extra_setup_thunk)
  if std.config["pretty"] then os.exit() end

  assert(std.is_task(main_task))
  local objfiles,task_wrappers = compile_tasks_in_parallel()

  -- If task wrappers were compiled on separate processes, link them all into a
  -- dynamic library and load that.
  if #objfiles > 0 then
    local dylib = os.tmpname()
    local cmd = os.getenv('CXX') or 'c++'
    if os.execute('test "$(uname)" = Darwin') == 0 then
      cmd = cmd .. ' -dynamiclib -single_module -undefined dynamic_lookup -fPIC'
    else
      cmd = cmd .. ' -shared -fPIC'
    end
    cmd = cmd .. ' -o ' .. dylib
    for _,f in ipairs(objfiles) do
      cmd = cmd .. ' ' .. f
    end
    assert(os.execute(cmd) == 0)
    terralib.linklibrary(dylib)
  end

  local main = std.setup(main_task, extra_setup_thunk, task_wrappers)

  local args = std.args
  local argc = #args
  local argv = terralib.newsymbol((&int8)[argc + 1], "argv")
  local argv_setup = terralib.newlist({quote var [argv] end})
  for i, arg in ipairs(args) do
    argv_setup:insert(quote
      [argv][ [i - 1] ] = [arg]
    end)
  end
  argv_setup:insert(quote [argv][ [argc] ] = [&int8](0) end)

  local terra wrapper()
    [argv_setup];
    return main([argc], [argv])
  end

  profile('compile', nil, function() wrapper:compile() end)()

  profile.print_summary()

  wrapper()
end

local function infer_filetype(filename)
  if filename:match("%.o$") then
    return "object"
  elseif filename:match("%.bc$") then
    return "bitcode"
  elseif filename:match("%.ll$") then
    return "llvmir"
  elseif filename:match("%.so$") or filename:match("%.dylib$") or filename:match("%.dll$") then
    return "sharedlibrary"
  elseif filename:match("%.s") then
    return "asm"
  else
    return "executable"
  end
end

function std.saveobj(main_task, filename, filetype, extra_setup_thunk, link_flags)
  assert(std.is_task(main_task))
  filetype = filetype or infer_filetype(filename)
  assert(not link_flags or filetype == 'sharedlibrary' or filetype == 'executable',
         'Link flags are ignored unless saving to shared library or executable')

  local objfiles,task_wrappers = compile_tasks_in_parallel()
  if #objfiles > 0 then
    assert(filetype == "object" or filetype == "executable",
           'Parallel compilation only supported for object or executable output')
  end

  local main, names = std.setup(main_task, extra_setup_thunk, task_wrappers)

  local flags = terralib.newlist()
  flags:insertall(objfiles)
  local use_cmake = os.getenv("USE_CMAKE") == "1"
  local lib_dir = os.getenv("LG_RT_DIR") .. "/../bindings/regent"
  if use_cmake then
    lib_dir = os.getenv("CMAKE_BUILD_DIR") .. "/lib"
  end
  if os.getenv('CRAYPE_VERSION') then
    flags:insert("-Wl,-Bdynamic")
  end
  if link_flags then flags:insertall(link_flags) end
  if os.getenv('CRAYPE_VERSION') then
    for flag in os.getenv('CRAY_UGNI_POST_LINK_OPTS'):gmatch("%S+") do
      flags:insert(flag)
    end
    flags:insert("-lugni")
    for flag in os.getenv('CRAY_UDREG_POST_LINK_OPTS'):gmatch("%S+") do
      flags:insert(flag)
    end
    flags:insert("-ludreg")
  end
  flags:insertall({"-L" .. lib_dir, "-lregent"})
  if use_cmake then
    flags:insertall({"-llegion", "-lrealm"})
  end

  profile('compile', nil, function()
    if #objfiles > 0 and filetype == 'object' then
      -- Terra will not read the link flags in this case, to collect the code
      -- that was compiled on different processes, so we have to combine all
      -- the object files manually.
      local mainobj = os.tmpname()
      terralib.saveobj(mainobj, 'object', names)
      local cmd = os.getenv('CXX') or 'c++'
      cmd = cmd .. ' -Wl,-r'
      cmd = cmd .. ' ' .. mainobj
      for _,f in ipairs(objfiles) do
        cmd = cmd .. ' ' .. f
      end
      cmd = cmd .. ' -o ' .. filename
      cmd = cmd .. ' -nostdlib'
      assert(os.execute(cmd) == 0)
    else
      terralib.saveobj(filename, filetype, names, flags)
    end
  end)()
  profile.print_summary()
end

local function generate_task_interfaces()
  local tasks = {}
  for _, variant in ipairs(variants) do
    tasks[variant.task] = true
  end

  local task_c_iface = terralib.newlist()
  local task_cxx_iface = terralib.newlist()
  local task_impl = {}
  for task, _ in pairs(tasks) do
    task_c_iface:insert(header_helper.generate_task_c_interface(task))
    task_cxx_iface:insert(header_helper.generate_task_cxx_interface(task))
    local definitions = header_helper.generate_task_implementation(task)
    for _, definition in ipairs(definitions) do
      task_impl[definition[1]] = definition[2]
    end
  end

  return task_c_iface:concat("\n\n"), task_cxx_iface:concat("\n\n"), task_impl
end

local function generate_header(header_filename, registration_name, task_c_iface, task_cxx_iface)
  local header_basename = header_helper.normalize_name(header_filename)
  return string.format(
[[
#ifndef __%s__
#define __%s__

#include "stdint.h"

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"

#ifdef __cplusplus
#include "legion/legion_c_util.h"
#endif

// C API bindings

#ifdef __cplusplus
extern "C" {
#endif

void %s(void);

%s

#ifdef __cplusplus
}
#endif

// C++ API bindings

#ifdef __cplusplus

%s

#endif

#endif // __%s__
]],
  header_basename,
  header_basename,
  registration_name,
  task_c_iface,
  task_cxx_iface,
  header_basename)
end

local function write_header(header_filename)
  local registration_name = header_helper.normalize_name(header_filename) .. "_register"

  local task_c_iface, task_cxx_iface, task_impl = generate_task_interfaces()

  local header = io.open(header_filename, "w")
  assert(header)
  header:write(generate_header(header_filename, registration_name, task_c_iface, task_cxx_iface))
  header:close()

  return registration_name, task_impl
end

function std.save_tasks(header_filename, filename, filetype,
                       extra_setup_thunk, link_flags)
  assert(header_filename and filename)
  local task_wrappers = make_task_wrappers()
  local registration_name, task_impl = write_header(header_filename)
  local _, names = std.setup(nil, extra_setup_thunk, task_wrappers, registration_name)
  local use_cmake = os.getenv("USE_CMAKE") == "1"
  local lib_dir = os.getenv("LG_RT_DIR") .. "/../bindings/regent"
  if use_cmake then
    lib_dir = os.getenv("CMAKE_BUILD_DIR") .. "/lib"
  end

  -- Export task interface implementations
  for k, v in pairs(task_impl) do
    names[k] = v
  end

  local flags = terralib.newlist()
  if link_flags then flags:insertall(link_flags) end
  flags:insertall({"-L" .. lib_dir, "-lregent"})
  if use_cmake then
    flags:insertall({"-llegion", "-lrealm"})
  end
  profile('compile', nil, function()
    if filetype ~= nil then
      terralib.saveobj(filename, filetype, names, flags)
    else
      terralib.saveobj(filename, names, flags)
    end
  end)()
  profile.print_summary()
end

-- #####################################
-- ## Vector Operators
-- #################
do
  local to_math_op_name = {}
  local function math_op_factory(fname)
    return terralib.memoize(function(arg_type)
      local intrinsic_name = "llvm." .. fname .. "."
      local elmt_type = arg_type
      if arg_type:isvector() then
        intrinsic_name = intrinsic_name .. "v" .. arg_type.N
        elmt_type = elmt_type.type
      end
      assert(elmt_type == float or elmt_type == double)
      intrinsic_name = intrinsic_name .. "f" .. (sizeof(elmt_type) * 8)
      local op = terralib.intrinsic(intrinsic_name, arg_type -> arg_type)
      to_math_op_name[op] = fname
      return op
    end)
  end

  local supported_math_ops = { -- math operators supported by ISA
    "ceil",
    "cos",
    "exp",
    "exp2",
    "fabs",
    "floor",
    "log",
    "log2",
    "log10",
    "sin",
    "sqrt",
    "trunc"
  }

  local cmath_ops = { -- math operators backed by standard library
    "acos",
    "asin",
    "atan",
    "cbrt",
    "tan",
    "pow",
    "fmod",
  }

  for _, fname in pairs(supported_math_ops) do
    std[fname] = math_op_factory(fname)
    if std.config["cuda"] and cudahelper.check_cuda_available() then
      cudahelper.register_builtin(fname, std[fname](double))
    end
  end

  local cmath = terralib.includec("math.h")
  for _, fname in pairs(cmath_ops) do
    std[fname] = function(arg_type) return cmath[fname] end
    if std.config["cuda"] and cudahelper.check_cuda_available() then
      cudahelper.register_builtin(fname, std[fname](double))
    end
  end

  function std.is_math_op(op)
    return to_math_op_name[op] ~= nil
  end

  function std.convert_math_op(op, arg_type)
    return std[to_math_op_name[op]](arg_type)
  end

  function std.get_math_op_name(op, arg_type)
    return '[regentlib.' .. to_math_op_name[op] .. '(' .. tostring(arg_type) .. ')]'
  end
end

do
  local intrinsic_names = {}
  if os.execute("bash -c \"[ `uname` == 'Linux' ]\"") == 0 and
    os.execute("grep altivec /proc/cpuinfo > /dev/null") == 0
  then
    intrinsic_names[vector(float,  4)] = "llvm.ppc.altivec.v%sfp"
    intrinsic_names[vector(double, 2)] = "llvm.ppc.vsx.xv%sdp"
  else
    intrinsic_names[vector(float,  4)] = "llvm.x86.sse.%s.ps"
    intrinsic_names[vector(double, 2)] = "llvm.x86.sse2.%s.pd"
    intrinsic_names[vector(float,  8)] = "llvm.x86.avx.%s.ps.256"
    intrinsic_names[vector(double, 4)] = "llvm.x86.avx.%s.pd.256"
  end

  local function math_binary_op_factory(fname)
    return terralib.memoize(function(arg_type)
      assert(arg_type:isvector())
      assert((arg_type.type == float and 4 <= arg_type.N and arg_type.N <= 8) or
             (arg_type.type == double and 2 <= arg_type.N and arg_type.N <= 4))

      local intrinsic_name = string.format(intrinsic_names[arg_type], fname)
      return terralib.intrinsic(intrinsic_name,
                                {arg_type, arg_type} -> arg_type)
    end)
  end

  local supported_math_binary_ops = { "min", "max", }
  for _, fname in pairs(supported_math_binary_ops) do
    std["v" .. fname] = math_binary_op_factory(fname)
  end

  function std.is_minmax_supported(arg_type)
    assert(not (std.is_ref(arg_type) or std.is_rawref(arg_type)))
    if not arg_type:isvector() then return false end
    if not ((arg_type.type == float and
             4 <= arg_type.N and arg_type.N <= 8) or
            (arg_type.type == double and
             2 <= arg_type.N and arg_type.N <= 4)) then
      return false
    end
    return true
  end
end

return std
