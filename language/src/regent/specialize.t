-- Copyright 2016 Stanford University, NVIDIA Corporation
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

-- Legion Specialization Pass

local ast = require("regent/ast")
local data = require("regent/data")
local log = require("regent/log")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local specialize = {}

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    env = self.env:new_local_scope(),
  }
  setmetatable(cx, context)
  return cx
end

function context:new_global_scope(env)
  local cx = {
    env = symbol_table.new_global_scope(env),
  }
  setmetatable(cx, context)
  return cx
end

local function guess_type_for_literal(value)
  if type(value) == "number" then
    if terralib.isintegral(value) then
      return int
    else
      return double
    end
  elseif type(value) == "boolean" then
    return bool
  elseif type(value) == "string" then
    return rawstring
  elseif type(value) == "cdata" then
    return (`value):gettype()
  else
    assert(false)
  end
end

local function convert_lua_value(cx, node, value)
  if type(value) == "number" or type(value) == "boolean" or type(value) == "string" then
    local expr_type = guess_type_for_literal(value)
    return ast.specialized.expr.Constant {
      value = value,
      expr_type = expr_type,
      options = node.options,
      span = node.span,
    }
  elseif terralib.isfunction(value) or
    terralib.isoverloadedfunction(value) or
    terralib.ismacro(value) or
    terralib.types.istype(value) or std.is_task(value)
  then
    return ast.specialized.expr.Function {
      value = value,
      options = node.options,
      span = node.span,
    }
  elseif type(value) == "function" then
    log.error(node, "unable to specialize lua function (use terralib.cast to explicitly cast it to a terra function type)")
  elseif type(value) == "cdata" then
    local expr_type = guess_type_for_literal(value)
    if expr_type:isfunction() or expr_type:ispointertofunction() then
      return ast.specialized.expr.Function {
        value = value,
        options = node.options,
        span = node.span,
      }
    else
      return ast.specialized.expr.Constant {
        value = value,
        expr_type = expr_type,
        options = node.options,
        span = node.span,
      }
    end
  elseif terralib.isconstant(value) then
    local expr_type = value:gettype()
    return ast.specialized.expr.Constant {
      value = value,
      expr_type = expr_type,
      options = node.options,
      span = node.span,
    }
  elseif std.is_symbol(value) then
    return ast.specialized.expr.ID {
      value = value,
      options = node.options,
      span = node.span,
    }
  elseif std.is_rquote(value) then
    value = value:getast()
    if value:is(ast.typed.top.QuoteExpr) then
      assert(value.expr:is(ast.specialized.expr))
      return value.expr
    elseif value:is(ast.typed.top.QuoteStat) then
      log.error(node, "unable to specialize quoted statement as an expression")
    else
      log.error(node, "unexpected node type " .. tostring(value:type()))
    end
  elseif terralib.issymbol(value) then
    log.error(node, "unable to specialize terra symbol " .. tostring(value))
  elseif terralib.isquote(value) then
    log.error(node, "unable to specialize terra quote " .. tostring(value))
  elseif type(value) == "table" then
    return ast.specialized.expr.LuaTable {
      value = value,
      options = node.options,
      span = node.span,
    }
  else
    log.error(node, "unable to specialize value of type " .. tostring(type(value)))
  end
end

-- for the moment, multi-field accesses should be used only in
-- unary and binary expressions

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
  if type(node) == "function" then return 1 end

  if node:is(ast.unspecialized.expr.ID) then
    return 1

  elseif node:is(ast.unspecialized.expr.Escape) then
    if get_num_accessed_fields(node.expr) > 1 then return false
    else return 1 end

  elseif node:is(ast.unspecialized.expr.Constant) then
    return 1

  elseif node:is(ast.unspecialized.expr.FieldAccess) then
    return get_num_accessed_fields(node.value) * #node.field_names

  elseif node:is(ast.unspecialized.expr.IndexAccess) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    if get_num_accessed_fields(node.index) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.MethodCall) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    for _, arg in pairs(node.args) do
      if get_num_accessed_fields(arg) > 1 then return false end
    end
    return 1

  elseif node:is(ast.unspecialized.expr.Call) then
    if get_num_accessed_fields(node.fn) > 1 then return false end
    for _, arg in pairs(node.args) do
      if get_num_accessed_fields(arg) > 1 then return false end
    end
    return 1

  elseif node:is(ast.unspecialized.expr.Ctor) then
    node.fields:map(function(field)
      if field:is(ast.unspecialized.expr.CtorListField) then
        if get_num_accessed_fields(field.value) > 1 then return false end
      elseif field:is(ast.unspecialized.expr.CtorListField) then
        if get_num_accessed_fields(field.num_expr) > 1 then return false end
        if get_num_accessed_fields(field.value) > 1 then return false end
      end
    end)
    return 1

  elseif node:is(ast.unspecialized.expr.RawContext) then
    return 1

  elseif node:is(ast.unspecialized.expr.RawFields) then
    return 1

  elseif node:is(ast.unspecialized.expr.RawPhysical) then
    return 1

  elseif node:is(ast.unspecialized.expr.RawRuntime) then
    return 1

  elseif node:is(ast.unspecialized.expr.RawValue) then
    return 1

  elseif node:is(ast.unspecialized.expr.Isnull) then
    if get_num_accessed_fields(node.pointer) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.New) then
    if get_num_accessed_fields(node.pointer_type_expr) > 1 then return false end
    if get_num_accessed_fields(node.extent) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Null) then
    if get_num_accessed_fields(node.pointer_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.DynamicCast) then
    if get_num_accessed_fields(node.type_expr) > 1 then return false end
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.StaticCast) then
    if get_num_accessed_fields(node.type_expr) > 1 then return false end
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.UnsafeCast) then
    if get_num_accessed_fields(node.type_expr) > 1 then return false end
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Ispace) then
    if get_num_accessed_fields(node.fspace_type_expr) > 1 then return false end
    if get_num_accessed_fields(node.size) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Region) then
    if get_num_accessed_fields(node.ispace) > 1 then return false end
    if get_num_accessed_fields(node.fspace_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Partition) then
    if get_num_accessed_fields(node.region) > 1 then return false end
    if get_num_accessed_fields(node.coloring) > 1 then return false end
    if get_num_accessed_fields(node.colors) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.PartitionEqual) then
    if get_num_accessed_fields(node.region) > 1 then return false end
    if get_num_accessed_fields(node.colors) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.PartitionByField) then
    if get_num_accessed_fields(node.region) > 1 then return false end
    if get_num_accessed_fields(node.colors) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Image) then
    if get_num_accessed_fields(node.parent) > 1 then return false end
    if get_num_accessed_fields(node.partition) > 1 then return false end
    if get_num_accessed_fields(node.region) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.Preimage) then
    if get_num_accessed_fields(node.parent) > 1 then return false end
    if get_num_accessed_fields(node.partition) > 1 then return false end
    if get_num_accessed_fields(node.region) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.expr.CrossProduct) then
    return 1

  elseif node:is(ast.unspecialized.expr.CrossProductArray) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListSlicePartition) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListDuplicatePartition) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListCrossProduct) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListCrossProductComplete) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListPhaseBarriers) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListInvert) then
    return 1

  elseif node:is(ast.unspecialized.expr.ListRange) then
    return 1

  elseif node:is(ast.unspecialized.expr.PhaseBarrier) then
    return 1

  elseif node:is(ast.unspecialized.expr.DynamicCollective) then
    return 1

  elseif node:is(ast.unspecialized.expr.Advance) then
    return 1

  elseif node:is(ast.unspecialized.expr.Arrive) then
    return 1

  elseif node:is(ast.unspecialized.expr.Await) then
    return 1

  elseif node:is(ast.unspecialized.expr.DynamicCollectiveGetResult) then
    return 1

  elseif node:is(ast.unspecialized.expr.Copy) then
    return 1

  elseif node:is(ast.unspecialized.expr.Fill) then
    return 1

  elseif node:is(ast.unspecialized.expr.Unary) then
    return get_num_accessed_fields(node.rhs)

  elseif node:is(ast.unspecialized.expr.Binary) then
    return join_num_accessed_fields(get_num_accessed_fields(node.lhs),
                                    get_num_accessed_fields(node.rhs))

  elseif node:is(ast.unspecialized.expr.Deref) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

local function get_nth_field_access(node, idx)
  if node:is(ast.unspecialized.expr.FieldAccess) then
    local num_accessed_fields_value = get_num_accessed_fields(node.value)
    local num_accessed_fields = #node.field_names

    local idx1 = math.floor((idx - 1) / num_accessed_fields) + 1
    local idx2 = (idx - 1) % num_accessed_fields + 1

    local field_names = terralib.newlist()
    field_names:insert(node.field_names[idx2])
    return node {
      value = get_nth_field_access(node.value, idx1),
      field_names = field_names,
    }

  elseif node:is(ast.unspecialized.expr.Unary) then
    return node { rhs = get_nth_field_access(node.rhs, idx) }

  elseif node:is(ast.unspecialized.expr.Binary) then
    return node {
      lhs = get_nth_field_access(node.lhs, idx),
      rhs = get_nth_field_access(node.rhs, idx),
    }

  else
    return node
  end
end

local function has_all_valid_field_accesses(node)
  if node:is(ast.unspecialized.stat.Assignment) or
     node:is(ast.unspecialized.stat.Reduce) then

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
        -- special case when there is only one assignee for multiple
        -- values on the RHS
        elseif num_accessed_fields_lh == 1 and
               num_accessed_fields_rh > 1 then
          valid = false
        end
      end
    end)

    return valid
  else
    assert(false, "unreachable")
  end
end

function specialize.field_names(cx, node)
  if type(node.names_expr) == "string" then
    return terralib.newlist({node.names_expr})
  else
    local value = node.names_expr(cx.env:env())
    if type(value) == "string" then
      return terralib.newlist({value})
    elseif terralib.islist(value) and
      data.all(value:map(function(v) return type(v) == "string" end))
    then
      return value
    else
      log.error(node, "unable to specialize value of type " .. tostring(type(value)))
    end
  end
end

function specialize.region_field(cx, node)
  local field_names = specialize.field_names(cx, node.field_name)
  return field_names:map(
    function(field_name)
      return ast.specialized.region.Field {
        field_name = field_name,
        fields = specialize.region_fields(cx, node.fields),
        span = node.span,
      }
    end)
end

function specialize.region_fields(cx, node)
  return node and data.flatmap(
    function(field) return specialize.region_field(cx, field) end,
    node)
end

function specialize.region_root(cx, node)
  local region = cx.env:lookup(node, node.region_name)
  return ast.specialized.region.Root {
    symbol = region,
    fields = specialize.region_fields(cx, node.fields),
    span = node.span,
  }
end

function specialize.expr_region_root(cx, node)
  return ast.specialized.expr.RegionRoot {
    region = specialize.expr(cx, node.region),
    fields = specialize.region_fields(cx, node.fields),
    options = node.options,
    span = node.span,
  }
end

function specialize.region_bare(cx, node)
  local region = cx.env:lookup(node, node.region_name)
  return ast.specialized.region.Bare {
    symbol = region,
    span = node.span,
  }
end

function specialize.regions(cx, node)
  return node:map(
    function(region) return specialize.region_root(cx, region) end)
end

function specialize.condition_variable(cx, node)
  local symbol = cx.env:lookup(node, node.name)
  return ast.specialized.ConditionVariable {
    symbol = symbol,
    span = node.span,
  }
end

function specialize.condition_variables(cx, node)
  return node:map(
    function(variable) return specialize.condition_variable(cx, variable) end)
end

function specialize.privilege_kind(cx, node)
  if node:is(ast.unspecialized.privilege_kind.Reads) then
    return ast.specialized.privilege_kind.Reads(node)
  elseif node:is(ast.unspecialized.privilege_kind.Writes) then
    return ast.specialized.privilege_kind.Writes(node)
  elseif node:is(ast.unspecialized.privilege_kind.Reduces) then
    return ast.specialized.privilege_kind.Reduces(node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.privilege_kinds(cx, node)
  return node:map(
    function(privilege) return specialize.privilege_kind(cx, privilege) end)
end

function specialize.privilege(cx, node)
  return ast.specialized.Privilege {
    privileges = specialize.privilege_kinds(cx, node.privileges),
    regions = specialize.regions(cx, node.regions),
    span = node.span,
  }
end

function specialize.privileges(cx, node)
  return node:map(
    function(privilege) return specialize.privilege(cx, privilege) end)
end

function specialize.coherence_kind(cx, node)
  if node:is(ast.unspecialized.coherence_kind.Exclusive) then
    return ast.specialized.coherence_kind.Exclusive(node)
  elseif node:is(ast.unspecialized.coherence_kind.Atomic) then
    return ast.specialized.coherence_kind.Atomic(node)
  elseif node:is(ast.unspecialized.coherence_kind.Simultaneous) then
    return ast.specialized.coherence_kind.Simultaneous(node)
  elseif node:is(ast.unspecialized.coherence_kind.Relaxed) then
    return ast.specialized.coherence_kind.Relaxed(node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.coherence_kinds(cx, node)
  return node:map(
    function(coherence) return specialize.coherence_kind(cx, coherence) end)
end

function specialize.coherence(cx, node)
  return ast.specialized.Coherence {
    coherence_modes = specialize.coherence_kinds(cx, node.coherence_modes),
    regions = specialize.regions(cx, node.regions),
    span = node.span,
  }
end

function specialize.coherence_modes(cx, node)
  return node:map(
    function(coherence) return specialize.coherence(cx, coherence) end)
end

function specialize.flag_kind(cx, node)
  if node:is(ast.unspecialized.flag_kind.NoAccessFlag) then
    return ast.specialized.flag_kind.NoAccessFlag(node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.flag_kinds(cx, node)
  return node:map(function(flag) return specialize.flag_kind(cx, flag) end)
end

function specialize.flag(cx, node)
  return ast.specialized.Flag {
    flags = specialize.flag_kinds(cx, node.flags),
    regions = specialize.regions(cx, node.regions),
    span = node.span,
  }
end

function specialize.flags(cx, node)
  return node:map(function(flag) return specialize.flag(cx, flag) end)
end

function specialize.condition_kind(cx, node)
  if node:is(ast.unspecialized.condition_kind.Arrives) then
    return ast.specialized.condition_kind.Arrives(node)
  elseif node:is(ast.unspecialized.condition_kind.Awaits) then
    return ast.specialized.condition_kind.Awaits(node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.condition_kinds(cx, node)
  return node:map(
    function(condition) return specialize.condition_kind(cx, condition) end)
end

function specialize.condition(cx, node)
  return ast.specialized.Condition {
    conditions = specialize.condition_kinds(cx, node.conditions),
    variables = specialize.condition_variables(cx, node.variables),
    span = node.span,
  }
end

function specialize.expr_condition(cx, node)
  return ast.specialized.expr.Condition {
    conditions = specialize.condition_kinds(cx, node.conditions),
    values = node.values:map(
      function(value) return specialize.expr(cx, value) end),
    options = node.options,
    span = node.span,
  }
end

function specialize.conditions(cx, node)
  return node:map(
    function(condition) return specialize.condition(cx, condition) end)
end

function specialize.expr_conditions(cx, node)
  return node:map(
    function(condition) return specialize.expr_condition(cx, condition) end)
end

function specialize.constraint_kind(cx, node)
  if node:is(ast.unspecialized.constraint_kind.Subregion) then
    return ast.specialized.constraint_kind.Subregion(node)
  elseif node:is(ast.unspecialized.constraint_kind.Disjointness) then
    return ast.specialized.constraint_kind.Disjointness(node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.constraint(cx, node)
  return ast.specialized.Constraint {
    lhs = specialize.region_bare(cx, node.lhs),
    op = specialize.constraint_kind(cx, node.op),
    rhs = specialize.region_bare(cx, node.rhs),
    span = node.span,
  }
end

function specialize.constraints(cx, node)
  return node:map(
    function(constraint) return specialize.constraint(cx, constraint) end)
end

function specialize.disjointness_kind(cx, node)
  if node:is(ast.unspecialized.disjointness_kind.Aliased) then
    return std.aliased
  elseif node:is(ast.unspecialized.disjointness_kind.Disjoint) then
    return std.disjoint
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.expr_id(cx, node)
  local value = cx.env:lookup(node, node.name)
  return convert_lua_value(cx, node, value)
end

function specialize.expr_escape(cx, node)
  local value = node.expr(cx.env:env())
  return convert_lua_value(cx, node, value)
end

function specialize.expr_constant(cx, node)
  return ast.specialized.expr.Constant {
    value = node.value,
    expr_type = node.expr_type,
    options = node.options,
    span = node.span,
  }
end

-- assumes multi-field accesses have already been flattened by the caller
function specialize.expr_field_access(cx, node)
  if #node.field_names ~= 1 then
    log.error(node, "illegal use of multi-field access")
  end
  local value = specialize.expr(cx, node.value)

  local field_names = specialize.field_names(cx, node.field_names[1])
  if #field_names ~= 1 then
    log.error(node, "FIXME: handle specialization of multiple fields")
  end
  local field_name = field_names[1]

  if value:is(ast.specialized.expr.LuaTable) then
    return convert_lua_value(cx, node, value.value[field_name])
  else
    return ast.specialized.expr.FieldAccess {
      value = value,
      field_name = field_name,
      options = node.options,
      span = node.span,
    }
  end
end

function specialize.expr_index_access(cx, node)
  return ast.specialized.expr.IndexAccess {
    value = specialize.expr(cx, node.value),
    index = specialize.expr(cx, node.index),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_method_call(cx, node)
  return ast.specialized.expr.MethodCall {
    value = specialize.expr(cx, node.value),
    method_name = node.method_name,
    args = node.args:map(
      function(arg) return specialize.expr(cx, arg) end),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_call(cx, node)
  local fn = specialize.expr(cx, node.fn)
  if terralib.isfunction(fn.value) or
    terralib.isoverloadedfunction(fn.value) or
    terralib.ismacro(fn.value) or
    std.is_task(fn.value) or
    type(fn.value) == "cdata"
  then
    if not std.is_task(fn.value) and #node.conditions > 0 then
      log.error(node.conditions[1],
        "terra function call cannot have conditions")
    end
    return ast.specialized.expr.Call {
      fn = fn,
      args = node.args:map(
        function(arg) return specialize.expr(cx, arg) end),
      conditions = specialize.expr_conditions(cx, node.conditions),
      options = node.options,
      span = node.span,
    }
  elseif terralib.types.istype(fn.value) then
    return ast.specialized.expr.Cast {
      fn = fn,
      args = node.args:map(
        function(arg) return specialize.expr(cx, arg) end),
      options = node.options,
      span = node.span,
    }
  else
    assert(false, "unreachable")
  end
end

function specialize.expr_ctor_list_field(cx, node)
  return ast.specialized.expr.CtorListField {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_ctor_rec_field(cx, node)
  local name = node.name_expr(cx.env:env())
  if terralib.issymbol(name) then
    name = name.displayname
  elseif not type(name) == "string" then
    assert("expected a string or symbol but found " .. tostring(type(name)))
  end

  return ast.specialized.expr.CtorRecField {
    name = name,
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_ctor_field(cx, node)
  if node:is(ast.unspecialized.expr.CtorListField) then
    return specialize.expr_ctor_list_field(cx, node)
  elseif node:is(ast.unspecialized.expr.CtorRecField) then
    return specialize.expr_ctor_rec_field(cx, node)
  else
  end
end

function specialize.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return specialize.expr_ctor_field(cx, field) end)

  -- Validate that fields are either all named or all unnamed.
  local all_named = false
  local all_unnamed = false
  for _, field in ipairs(fields) do
    if field:is(ast.specialized.expr.CtorRecField) then
      assert(not all_unnamed,
             "some entries in constructor are named while others are not")
      all_named = true
    elseif field:is(ast.specialized.expr.CtorListField) then
      assert(not all_named,
             "some entries in constructor are named while others are not")
      all_unnamed = true
    else
      assert(false)
    end
  end

  return ast.specialized.expr.Ctor {
    fields = fields,
    named = all_named,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_raw_context(cx, node)
  return ast.specialized.expr.RawContext {
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_raw_fields(cx, node)
  return ast.specialized.expr.RawFields {
    region = specialize.expr(cx, node.region),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_raw_physical(cx, node)
  return ast.specialized.expr.RawPhysical {
    region = specialize.expr(cx, node.region),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_raw_runtime(cx, node)
  return ast.specialized.expr.RawRuntime {
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_raw_value(cx, node)
  return ast.specialized.expr.RawValue {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_isnull(cx, node)
  local pointer = specialize.expr(cx, node.pointer)
  return ast.specialized.expr.Isnull {
    pointer = pointer,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_new(cx, node)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  if not std.is_bounded_type(pointer_type) then
    log.error(node, "new requires bounded type, got " .. tostring(pointer_type))
  end
  local bounds = pointer_type.bounds_symbols
  if #bounds ~= 1 then
    log.error(node, "new requires bounded type with exactly one region, got " .. tostring(pointer_type))
  end
  local region = ast.specialized.expr.ID {
    value = bounds[1],
    options = node.options,
    span = node.span,
  }
  return ast.specialized.expr.New {
    pointer_type = pointer_type,
    extent = node.extent and specialize.expr(cx, node.extent),
    region = region,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_null(cx, node)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  return ast.specialized.expr.Null {
    pointer_type = pointer_type,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_dynamic_cast(cx, node)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.DynamicCast {
    value = value,
    expr_type = expr_type,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_static_cast(cx, node)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.StaticCast {
    value = value,
    expr_type = expr_type,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_unsafe_cast(cx, node)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.UnsafeCast {
    value = value,
    expr_type = expr_type,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_ispace(cx, node)
  local index_type = node.index_type_expr(cx.env:env())
  return ast.specialized.expr.Ispace {
    index_type = index_type,
    extent = specialize.expr(cx, node.extent),
    start = node.start and specialize.expr(cx, node.start),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_region(cx, node)
  local ispace = specialize.expr(cx, node.ispace)
  local fspace_type = node.fspace_type_expr(cx.env:env())
  return ast.specialized.expr.Region {
    ispace = ispace,
    fspace_type = fspace_type,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_partition(cx, node)
  return ast.specialized.expr.Partition {
    disjointness = specialize.disjointness_kind(cx, node.disjointness),
    region = specialize.expr(cx, node.region),
    coloring = specialize.expr(cx, node.coloring),
    colors = node.colors and specialize.expr(cx, node.colors),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_partition_equal(cx, node)
  return ast.specialized.expr.PartitionEqual {
    region = specialize.expr(cx, node.region),
    colors = specialize.expr(cx, node.colors),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_partition_by_field(cx, node)
  return ast.specialized.expr.PartitionByField {
    region = specialize.expr_region_root(cx, node.region),
    colors = specialize.expr(cx, node.colors),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_image(cx, node)
  return ast.specialized.expr.Image {
    parent = specialize.expr(cx, node.parent),
    partition = specialize.expr(cx, node.partition),
    region = specialize.expr_region_root(cx, node.region),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_preimage(cx, node)
  return ast.specialized.expr.Preimage {
    parent = specialize.expr(cx, node.parent),
    partition = specialize.expr(cx, node.partition),
    region = specialize.expr_region_root(cx, node.region),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_cross_product(cx, node)
  return ast.specialized.expr.CrossProduct {
    args = node.args:map(
      function(arg) return specialize.expr(cx, arg) end),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_cross_product_array(cx, node)
  return ast.specialized.expr.CrossProductArray {
    lhs = specialize.expr(cx, node.lhs),
    disjointness = specialize.disjointness_kind(cx, node.disjointness),
    colorings = specialize.expr(cx, node.colorings),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_slice_partition(cx, node)
  return ast.specialized.expr.ListSlicePartition {
    partition = specialize.expr(cx, node.partition),
    indices = specialize.expr(cx, node.indices),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_duplicate_partition(cx, node)
  return ast.specialized.expr.ListDuplicatePartition {
    partition = specialize.expr(cx, node.partition),
    indices = specialize.expr(cx, node.indices),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_cross_product(cx, node)
  return ast.specialized.expr.ListCrossProduct {
    lhs = specialize.expr(cx, node.lhs),
    rhs = specialize.expr(cx, node.rhs),
    shallow = node.shallow,
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_cross_product_complete(cx, node)
  return ast.specialized.expr.ListCrossProductComplete {
    lhs = specialize.expr(cx, node.lhs),
    product = specialize.expr(cx, node.product),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_phase_barriers(cx, node)
  return ast.specialized.expr.ListPhaseBarriers {
    product = specialize.expr(cx, node.product),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_invert(cx, node)
  return ast.specialized.expr.ListInvert {
    rhs = specialize.expr(cx, node.rhs),
    product = specialize.expr(cx, node.product),
    barriers = specialize.expr(cx, node.barriers),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_list_range(cx, node)
  return ast.specialized.expr.ListRange {
    start = specialize.expr(cx, node.start),
    stop = specialize.expr(cx, node.stop),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_phase_barrier(cx, node)
  return ast.specialized.expr.PhaseBarrier {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_dynamic_collective(cx, node)
  local value_type = node.value_type_expr(cx.env:env())
  return ast.specialized.expr.DynamicCollective {
    value_type = value_type,
    op = node.op,
    arrivals = specialize.expr(cx, node.arrivals),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_dynamic_collective_get_result(cx, node)
  return ast.specialized.expr.DynamicCollectiveGetResult {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_advance(cx, node)
  return ast.specialized.expr.Advance {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_arrive(cx, node)
  return ast.specialized.expr.Arrive {
    barrier = specialize.expr(cx, node.barrier),
    value = node.value and specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_await(cx, node)
  return ast.specialized.expr.Await {
    barrier = specialize.expr(cx, node.barrier),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_copy(cx, node)
  return ast.specialized.expr.Copy {
    src = specialize.expr_region_root(cx, node.src),
    dst = specialize.expr_region_root(cx, node.dst),
    op = node.op,
    conditions = specialize.expr_conditions(cx, node.conditions),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_fill(cx, node)
  return ast.specialized.expr.Fill {
    dst = specialize.expr_region_root(cx, node.dst),
    value = specialize.expr(cx, node.value),
    conditions = specialize.expr_conditions(cx, node.conditions),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_allocate_scratch_fields(cx, node)
  return ast.specialized.expr.AllocateScratchFields {
    region = specialize.expr_region_root(cx, node.region),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_with_scratch_fields(cx, node)
  return ast.specialized.expr.WithScratchFields {
    region = specialize.expr_region_root(cx, node.region),
    field_ids = specialize.expr(cx, node.field_ids),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_unary(cx, node)
  return ast.specialized.expr.Unary {
    op = node.op,
    rhs = specialize.expr(cx, node.rhs),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_binary(cx, node)
  return ast.specialized.expr.Binary {
    op = node.op,
    lhs = specialize.expr(cx, node.lhs),
    rhs = specialize.expr(cx, node.rhs),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr_deref(cx, node)
  return ast.specialized.expr.Deref {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.expr(cx, node)
  if node:is(ast.unspecialized.expr.ID) then
    return specialize.expr_id(cx, node)

  elseif node:is(ast.unspecialized.expr.Escape) then
    return specialize.expr_escape(cx, node)

  elseif node:is(ast.unspecialized.expr.Constant) then
    return specialize.expr_constant(cx, node)

  elseif node:is(ast.unspecialized.expr.FieldAccess) then
    return specialize.expr_field_access(cx, node)

  elseif node:is(ast.unspecialized.expr.IndexAccess) then
    return specialize.expr_index_access(cx, node)

  elseif node:is(ast.unspecialized.expr.MethodCall) then
    return specialize.expr_method_call(cx, node)

  elseif node:is(ast.unspecialized.expr.Call) then
    return specialize.expr_call(cx, node)

  elseif node:is(ast.unspecialized.expr.Ctor) then
    return specialize.expr_ctor(cx, node)

  elseif node:is(ast.unspecialized.expr.RawContext) then
    return specialize.expr_raw_context(cx, node)

  elseif node:is(ast.unspecialized.expr.RawFields) then
    return specialize.expr_raw_fields(cx, node)

  elseif node:is(ast.unspecialized.expr.RawPhysical) then
    return specialize.expr_raw_physical(cx, node)

  elseif node:is(ast.unspecialized.expr.RawRuntime) then
    return specialize.expr_raw_runtime(cx, node)

  elseif node:is(ast.unspecialized.expr.RawValue) then
    return specialize.expr_raw_value(cx, node)

  elseif node:is(ast.unspecialized.expr.Isnull) then
    return specialize.expr_isnull(cx, node)

  elseif node:is(ast.unspecialized.expr.New) then
    return specialize.expr_new(cx, node)

  elseif node:is(ast.unspecialized.expr.Null) then
    return specialize.expr_null(cx, node)

  elseif node:is(ast.unspecialized.expr.DynamicCast) then
    return specialize.expr_dynamic_cast(cx, node)

  elseif node:is(ast.unspecialized.expr.StaticCast) then
    return specialize.expr_static_cast(cx, node)

  elseif node:is(ast.unspecialized.expr.UnsafeCast) then
    return specialize.expr_unsafe_cast(cx, node)

  elseif node:is(ast.unspecialized.expr.Ispace) then
    return specialize.expr_ispace(cx, node)

  elseif node:is(ast.unspecialized.expr.Region) then
    return specialize.expr_region(cx, node)

  elseif node:is(ast.unspecialized.expr.Partition) then
    return specialize.expr_partition(cx, node)

  elseif node:is(ast.unspecialized.expr.PartitionEqual) then
    return specialize.expr_partition_equal(cx, node)

  elseif node:is(ast.unspecialized.expr.PartitionByField) then
    return specialize.expr_partition_by_field(cx, node)

  elseif node:is(ast.unspecialized.expr.Image) then
    return specialize.expr_image(cx, node)

  elseif node:is(ast.unspecialized.expr.Preimage) then
    return specialize.expr_preimage(cx, node)

  elseif node:is(ast.unspecialized.expr.CrossProduct) then
    return specialize.expr_cross_product(cx, node)

  elseif node:is(ast.unspecialized.expr.CrossProductArray) then
    return specialize.expr_cross_product_array(cx, node)

  elseif node:is(ast.unspecialized.expr.ListSlicePartition) then
    return specialize.expr_list_slice_partition(cx, node)

  elseif node:is(ast.unspecialized.expr.ListDuplicatePartition) then
    return specialize.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.unspecialized.expr.ListCrossProduct) then
    return specialize.expr_list_cross_product(cx, node)

  elseif node:is(ast.unspecialized.expr.ListCrossProductComplete) then
    return specialize.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.unspecialized.expr.ListPhaseBarriers) then
    return specialize.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.unspecialized.expr.ListInvert) then
    return specialize.expr_list_invert(cx, node)

  elseif node:is(ast.unspecialized.expr.ListRange) then
    return specialize.expr_list_range(cx, node)

  elseif node:is(ast.unspecialized.expr.PhaseBarrier) then
    return specialize.expr_phase_barrier(cx, node)

  elseif node:is(ast.unspecialized.expr.DynamicCollective) then
    return specialize.expr_dynamic_collective(cx, node)

  elseif node:is(ast.unspecialized.expr.Advance) then
    return specialize.expr_advance(cx, node)

  elseif node:is(ast.unspecialized.expr.Arrive) then
    return specialize.expr_arrive(cx, node)

  elseif node:is(ast.unspecialized.expr.Await) then
    return specialize.expr_await(cx, node)

  elseif node:is(ast.unspecialized.expr.DynamicCollectiveGetResult) then
    return specialize.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.unspecialized.expr.Copy) then
    return specialize.expr_copy(cx, node)

  elseif node:is(ast.unspecialized.expr.Fill) then
    return specialize.expr_fill(cx, node)

  elseif node:is(ast.unspecialized.expr.AllocateScratchFields) then
    return specialize.expr_allocate_scratch_fields(cx, node)

  elseif node:is(ast.unspecialized.expr.WithScratchFields) then
    return specialize.expr_with_scratch_fields(cx, node)

  elseif node:is(ast.unspecialized.expr.Unary) then
    return specialize.expr_unary(cx, node)

  elseif node:is(ast.unspecialized.expr.Binary) then
    return specialize.expr_binary(cx, node)

  elseif node:is(ast.unspecialized.expr.Deref) then
    return specialize.expr_deref(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function specialize.block(cx, node)
  local stats = terralib.newlist()
  for _, stat in ipairs(node.stats) do
    local value = specialize.stat(cx, stat)
    if terralib.islist(value) then
      stats:insertall(value)
    else
      stats:insert(value)
    end
  end

  return ast.specialized.Block {
    stats = stats,
    span = node.span,
  }
end

function specialize.stat_if(cx, node)
  local then_cx = cx:new_local_scope()
  local else_cx = cx:new_local_scope()
  return ast.specialized.stat.If {
    cond = specialize.expr(cx, node.cond),
    then_block = specialize.block(then_cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return specialize.stat_elseif(cx, block) end),
    else_block = specialize.block(else_cx, node.else_block),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_elseif(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.stat.Elseif {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_while(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.stat.While {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return specialize.expr(cx, value) end)

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type = node.type_expr(cx.env:env())
  local symbol = std.newsymbol(var_type, node.name)
  cx.env:insert(node, node.name, symbol)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.stat.ForNum {
    symbol = symbol,
    values = values,
    block = block,
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_for_list(cx, node)
  local value = specialize.expr(cx, node.value)

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type
  if node.type_expr then
    var_type = node.type_expr(cx.env:env())
  end
  local symbol = std.newsymbol(var_type, node.name)
  cx.env:insert(node, node.name, symbol)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.stat.ForList {
    symbol = symbol,
    value = value,
    block = block,
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_repeat(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.Repeat {
    block = specialize.block(cx, node.block),
    until_cond = specialize.expr(cx, node.until_cond),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_must_epoch(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.MustEpoch {
    block = specialize.block(cx, node.block),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.Block {
    block = specialize.block(cx, node.block),
    options = node.options,
    span = node.span,
  }
end

local function make_symbol(cx, node, var_name, var_type)
  if type(var_name) == "string" then
    return std.newsymbol(var_type or nil, var_name)
  end

  var_name = var_name(cx.env:env())
  if std.is_symbol(var_name) then
    return var_name
  end

  log.error(node, "unable to specialize value of type " .. tostring(type(var_name)))
end

function specialize.stat_var(cx, node)
  -- Hack: To handle recursive regions, need to put a proxy into place
  -- before looking at either types or values.
  local symbols = terralib.newlist()
  for i, var_name in ipairs(node.var_names) do
    if node.values[i] and node.values[i]:is(ast.unspecialized.expr.Region) then
      local symbol = make_symbol(cx, node, var_name)
      cx.env:insert(node, var_name, symbol)
      symbols[i] = symbol
    end
  end

  local types = node.type_exprs:map(
    function(type_expr) return type_expr and type_expr(cx.env:env()) end)
  local values = node.values:map(
      function(value) return specialize.expr(cx, value) end)

  -- Then we patch up any region values so they have the type we
  -- claimed they originally had (closing the cycle).
  for i, var_name in ipairs(node.var_names) do
    local var_type = types[i]
    local symbol = symbols[i]
    if not symbol then
      symbol = make_symbol(cx, node, var_name, var_type)
      cx.env:insert(node, var_name, symbol)
      symbols[i] = symbol
    end
  end

  return ast.specialized.stat.Var {
    symbols = symbols,
    values = values,
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_var_unpack(cx, node)
  local symbols = terralib.newlist()
  for _, var_name in ipairs(node.var_names) do
    local symbol = std.newsymbol(var_name)
    cx.env:insert(node, var_name, symbol)
    symbols:insert(symbol)
  end

  local value = specialize.expr(cx, node.value)

  return ast.specialized.stat.VarUnpack {
    symbols = symbols,
    fields = node.fields,
    value = value,
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_return(cx, node)
  return ast.specialized.stat.Return {
    value = node.value and specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_break(cx, node)
  return ast.specialized.stat.Break {
    options = node.options,
    span = node.span,
  }
end

function specialize.stat_assignment_or_stat_reduce(cx, node)
  if not has_all_valid_field_accesses(node) then
    log.error(node, "invalid use of multi-field access")
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
      flattened_lhs:insert(specialize.expr(cx, get_nth_field_access(lh, idx)))
      flattened_rhs:insert(specialize.expr(cx, get_nth_field_access(rh, idx)))
    end
  end)

  if node:is(ast.unspecialized.stat.Assignment) then
    return ast.specialized.stat.Assignment {
      lhs = flattened_lhs,
      rhs = flattened_rhs,
      options = node.options,
      span = node.span,
    }

  elseif node:is(ast.unspecialized.stat.Reduce) then
    return ast.specialized.stat.Reduce {
      lhs = flattened_lhs,
      rhs = flattened_rhs,
      op = node.op,
      options = node.options,
      span = node.span,
    }
  else
    assert(false)
  end
end

function specialize.stat_expr(cx, node)
  return ast.specialized.stat.Expr {
    expr = specialize.expr(cx, node.expr),
    options = node.options,
    span = node.span,
  }
end

local function get_quote_contents(expr)
  assert(std.is_rquote(expr))

  local value = expr:getast()
  if value:is(ast.typed.top.QuoteExpr) then
    assert(value.expr:is(ast.specialized.expr))
    return terralib.newlist({
      ast.specialized.stat.Expr {
        expr = value.expr,
        options = node.options,
        span = node.span,
      },
    })
  elseif value:is(ast.typed.top.QuoteStat) then
    assert(value.block:is(ast.specialized.Block))
    return value.block.stats
  else
    assert(false)
  end
end

function specialize.stat_escape(cx, node)
  local expr = node.expr(cx.env:env())
  if std.is_rquote(expr) then
    return get_quote_contents(expr)
  elseif terralib.islist(expr) then
    if not data.all(expr:map(function(v) return std.is_rquote(v) end)) then
      log.error(node, "unable to specialize value of type " .. tostring(type(expr)))
    end
    return data.flatmap(get_quote_contents, expr)
  else
    log.error(node, "unable to specialize value of type " .. tostring(type(expr)))
  end
end

function specialize.stat_raw_delete(cx, node)
  return ast.specialized.stat.RawDelete {
    value = specialize.expr(cx, node.value),
    options = node.options,
    span = node.span,
  }
end

function specialize.stat(cx, node)
  if node:is(ast.unspecialized.stat.If) then
    return specialize.stat_if(cx, node)

  elseif node:is(ast.unspecialized.stat.While) then
    return specialize.stat_while(cx, node)

  elseif node:is(ast.unspecialized.stat.ForNum) then
    return specialize.stat_for_num(cx, node)

  elseif node:is(ast.unspecialized.stat.ForList) then
    return specialize.stat_for_list(cx, node)

  elseif node:is(ast.unspecialized.stat.Repeat) then
    return specialize.stat_repeat(cx, node)

  elseif node:is(ast.unspecialized.stat.MustEpoch) then
    return specialize.stat_must_epoch(cx, node)

  elseif node:is(ast.unspecialized.stat.Block) then
    return specialize.stat_block(cx, node)

  elseif node:is(ast.unspecialized.stat.Var) then
    return specialize.stat_var(cx, node)

  elseif node:is(ast.unspecialized.stat.VarUnpack) then
    return specialize.stat_var_unpack(cx, node)

  elseif node:is(ast.unspecialized.stat.Return) then
    return specialize.stat_return(cx, node)

  elseif node:is(ast.unspecialized.stat.Break) then
    return specialize.stat_break(cx, node)

  elseif node:is(ast.unspecialized.stat.Assignment) then
    return specialize.stat_assignment_or_stat_reduce(cx, node)

  elseif node:is(ast.unspecialized.stat.Reduce) then
    return specialize.stat_assignment_or_stat_reduce(cx, node)

  elseif node:is(ast.unspecialized.stat.Expr) then
    return specialize.stat_expr(cx, node)

  elseif node:is(ast.unspecialized.stat.Escape) then
    return specialize.stat_escape(cx, node)

  elseif node:is(ast.unspecialized.stat.RawDelete) then
    return specialize.stat_raw_delete(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.top_task_param(cx, node)
  -- Hack: Params which are regions can be recursive on the name of
  -- the region so introduce the symbol before type checking to allow
  -- for this recursion.
  local symbol = std.newsymbol(node.param_name)
  cx.env:insert(node, node.param_name, symbol)
  local param_type = node.type_expr(cx.env:env())
  if not param_type then
    log.error(node, "param type is undefined or nil")
  end
  symbol:settype(param_type)

  return ast.specialized.top.TaskParam {
    symbol = symbol,
    options = node.options,
    span = node.span,
  }
end

function specialize.top_task_params(cx, node)
  return node:map(
    function(param) return specialize.top_task_param(cx, param) end)
end

function specialize.top_task(cx, node)
  local cx = cx:new_local_scope()
  local proto = std.newtask(node.name)
  proto:setinline(node.options.inline)
  if #node.name == 1 then
    cx.env:insert(node, node.name[1], proto)
  end
  cx = cx:new_local_scope()

  local params = specialize.top_task_params(cx, node.params)
  local return_type = node.return_type_expr(cx.env:env())
  local privileges = specialize.privileges(cx, node.privileges)
  local coherence_modes = specialize.coherence_modes(cx, node.coherence_modes)
  local flags = specialize.flags(cx, node.flags)
  local conditions = specialize.conditions(cx, node.conditions)
  local constraints = specialize.constraints(cx, node.constraints)
  local body = specialize.block(cx, node.body)

  return ast.specialized.top.Task {
    name = node.name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    flags = flags,
    conditions = conditions,
    constraints = constraints,
    body = body,
    prototype = proto,
    options = node.options,
    span = node.span,
  }
end

function specialize.top_fspace_param(cx, node, mapping)
  -- Insert symbol into environment first to allow circular types.
  local symbol = std.newsymbol(node.param_name)
  cx.env:insert(node, node.param_name, symbol)

  local param_type = node.type_expr(cx.env:env())
  symbol:settype(param_type)

  -- Check for fields with duplicate types.
  if std.type_supports_constraints(param_type) then
    if mapping[param_type] then
      log.error(node, "parameters " .. tostring(symbol) .. " and " ..
                  tostring(mapping[param_type]) ..
                  " have the same type, but are required to be distinct")
    end
    mapping[param_type] = symbol
  end

  return symbol
end

function specialize.top_fspace_field(cx, node, mapping)
  -- Insert symbol into environment first to allow circular types.
  local symbol = std.newsymbol(node.field_name)
  cx.env:insert(node, node.field_name, symbol)

  local field_type = node.type_expr(cx.env:env())
  if not field_type then
    log.error(node, "field type is undefined or nil")
  end
  symbol:settype(field_type)

  -- Check for fields with duplicate types.
  if std.type_supports_constraints(field_type) then
    if mapping[field_type] then
      log.error(node, "fields " .. tostring(symbol) .. " and " ..
                  tostring(mapping[field_type]) ..
                  " have the same type, but are required to be distinct")
    end
    mapping[field_type] = symbol
  end

  return  {
    field = symbol,
    type = field_type,
  }
end

function specialize.top_fspace(cx, node)
  local cx = cx:new_local_scope()
  local fs = std.newfspace(node, node.name, #node.params > 0)
  cx.env:insert(node, node.name, fs)

  local mapping = {}
  fs.params = node.params:map(
      function(param) return specialize.top_fspace_param(cx, param, mapping) end)
  fs.fields = node.fields:map(
      function(field) return specialize.top_fspace_field(cx, field, mapping) end)
  local constraints = specialize.constraints(cx, node.constraints)

  return ast.specialized.top.Fspace {
    name = node.name,
    fspace = fs,
    constraints = constraints,
    options = node.options,
    span = node.span,
  }
end

function specialize.top_quote_expr(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.top.QuoteExpr {
    expr = specialize.expr(cx, node.expr),
    options = node.options,
    span = node.span,
  }
end

function specialize.top_quote_stat(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.top.QuoteStat {
    block = specialize.block(cx, node.block),
    options = node.options,
    span = node.span,
  }
end

function specialize.top(cx, node)
  if node:is(ast.unspecialized.top.Task) then
    return specialize.top_task(cx, node)

  elseif node:is(ast.unspecialized.top.Fspace) then
    return specialize.top_fspace(cx, node)

  elseif node:is(ast.unspecialized.top.QuoteExpr) then
    return specialize.top_quote_expr(cx, node)

  elseif node:is(ast.unspecialized.top.QuoteStat) then
    return specialize.top_quote_stat(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.entry(env, node)
  local cx = context:new_global_scope(env)
  return specialize.top(cx, node)
end

return specialize
