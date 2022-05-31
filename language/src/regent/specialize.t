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

-- Regent Specialization Pass

local alpha_convert = require("regent/alpha_convert")
local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local specialize = {}

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

function context:new_local_scope(is_quote)
  local copy_mapping = {}
  for k, v in pairs(self.mapping) do
    copy_mapping[k] = v
  end

  local cx = {
    env = self.env:new_local_scope(),
    mapping = copy_mapping,
    is_quote = rawget(self, "is_quote") or is_quote or false,
  }
  setmetatable(cx, context)
  return cx
end

function context:new_global_scope(env)
  local cx = {
    env = symbol_table.new_global_scope(env),
    mapping = {},
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

local function convert_lua_value(cx, node, value, allow_lists)
  if type(value) == "number" or type(value) == "boolean" or type(value) == "string" then
    local expr_type = guess_type_for_literal(value)
    return ast.specialized.expr.Constant {
      value = value,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif terralib.isfunction(value) or
    terralib.isoverloadedfunction(value) or
    terralib.ismacro(value) or
    terralib.types.istype(value) or
    std.is_task(value) or std.is_math_fn(value) or
    std.is_macro(value)
  then
    return ast.specialized.expr.Function {
      value = value,
      annotations = node.annotations,
      span = node.span,
    }
  elseif type(value) == "function" then
    report.error(node, "unable to specialize lua function (use terralib.cast to explicitly cast it to a terra function type)")
  elseif type(value) == "cdata" then
    local expr_type = guess_type_for_literal(value)
    if expr_type:isfunction() or expr_type:ispointertofunction() then
      return ast.specialized.expr.Function {
        value = value,
        annotations = node.annotations,
        span = node.span,
      }
    else
      return ast.specialized.expr.Constant {
        value = value,
        expr_type = expr_type,
        annotations = node.annotations,
        span = node.span,
      }
    end
  elseif terralib.isconstant(value) then
    local expr_type = value:gettype()
    return ast.specialized.expr.Constant {
      value = value,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif terralib.isglobalvar(value) then
    local expr_type = value:gettype()
    return ast.specialized.expr.Global {
      value = value,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif std.is_symbol(value) then
    value = cx.env:safe_lookup(value) or value
    return ast.specialized.expr.ID {
      value = value,
      annotations = node.annotations,
      span = node.span,
    }
  elseif std.is_rquote(value) then
    value = value:getast()
    if value:is(ast.specialized.top.QuoteExpr) then
      assert(value.expr:is(ast.specialized.expr))
      if not cx.is_quote then
        value = alpha_convert.entry(value, cx.env, cx.mapping)
      end
      return value.expr
    elseif value:is(ast.specialized.top.QuoteStat) then
      report.error(node, "unable to specialize quoted statement as an expression")
    else
      report.error(node, "unexpected node type " .. tostring(value:type()))
    end
  elseif terralib.issymbol(value) then
    report.error(node, "unable to specialize terra symbol " .. tostring(value))
  elseif terralib.isquote(value) then
    report.error(node, "unable to specialize terra quote " .. tostring(value))
  elseif terralib.islist(value) then
    if not allow_lists then
      report.error(node, "unable to specialize terra list in this position")
    end
    return value:map(
      function(arg) return convert_lua_value(cx, node, arg, false) end)
  elseif type(value) == "table" then
    return ast.specialized.expr.LuaTable {
      value = value,
      annotations = node.annotations,
      span = node.span,
    }
  else
    report.error(node, "unable to specialize value of type " .. tostring(type(value)))
  end
end

function specialize.field_names(cx, node)
  if type(node.names_expr) == "string" then
    return terralib.newlist({node.names_expr})
  else
    local value = node.names_expr(cx.env:env())
    if type(value) == "string" or data.is_tuple(value) then
      return terralib.newlist({value})
    elseif terralib.islist(value) then
      value:map(function(v)
        if not (type(v) == "string" or data.is_tuple(v)) then
          report.error(node, "unable to specialize value of type " .. type(v))
        end
      end)
      return value
    else
      report.error(node, "unable to specialize value of type " .. tostring(type(value)))
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
    annotations = node.annotations,
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

function specialize.constraint_kind(cx, node)
  if node:is(ast.constraint_kind) then
    return node
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

function specialize.privilege_kind(cx, node)
  if node:is(ast.privilege_kind) then
    return node
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
  if node:is(ast.coherence_kind) then
    return node
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
  if node:is(ast.flag_kind) then
    return node
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
  if node:is(ast.condition_kind) then
    return node
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
    annotations = node.annotations,
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

function specialize.disjointness_kind(cx, node)
  if node:is(ast.disjointness_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.completeness_kind(cx, node)
  if node:is(ast.completeness_kind) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.effect_expr(cx, node)
  local span = node.span

  local function make_field(field_path, i)
    if i > #field_path then
      return false
    end
    local fields = make_field(field_path, i + 1)
    return ast.specialized.region.Field {
      field_name = field_path[i],
      fields = (fields and terralib.newlist({ fields })) or false,
      span = span,
    }
  end

  local function make_fields(field_path)
    if #field_path == 0 then
      return false
    end
    return terralib.newlist({make_field(field_path, 1)})
  end

  local function make_privilege(value)
    return ast.specialized.Privilege {
      privileges = terralib.newlist({value.privilege}),
      regions = terralib.newlist({
        ast.specialized.region.Root {
          symbol = value.region,
          fields = make_fields(value.field_path),
          span = span,
        }
      }),
      span = span,
    }
  end

  local function make_coherence(value)
    return ast.specialized.Coherence {
      coherence_modes = terralib.newlist({value.coherence_mode}),
      regions = terralib.newlist({
        ast.specialized.region.Root {
          symbol = value.region,
          fields = make_fields(value.field_path),
          span = span,
        }
      }),
      span = span,
    }
  end

  local value = node.expr(cx.env:env())
  if terralib.islist(value) then
    return value:map(
      function(v)
        if v:is(ast.privilege.Privilege) then
          return make_privilege(v)
        elseif v:is(ast.coherence.Coherence) then
          return make_coherence(v)
        else
          assert(false, "unexpected value type " .. tostring(value:type()))
        end
      end)
  elseif value:is(ast.privilege.Privilege) then
    return make_privilege(value)
  elseif value:is(ast.coherence.Coherence) then
    return make_coherence(value)
  else
    assert(false, "unexpected value type " .. tostring(value:type()))
  end
end

function specialize.effect(cx, node)
  if node:is(ast.unspecialized.Constraint) then
    return specialize.constraint(cx, node)
  elseif node:is(ast.unspecialized.Privilege) then
    return specialize.privilege(cx, node)
  elseif node:is(ast.unspecialized.Coherence) then
    return specialize.coherence(cx, node)
  elseif node:is(ast.unspecialized.Flag) then
    return specialize.flag(cx, node)
  elseif node:is(ast.unspecialized.Condition) then
    return specialize.condition(cx, node)
  elseif node:is(ast.unspecialized.Effect) then
    return specialize.effect_expr(cx, node)
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.effects(cx, node)
  local constraints = terralib.newlist()
  local privileges = terralib.newlist()
  local coherence_modes = terralib.newlist()
  local flags = terralib.newlist()
  local conditions = terralib.newlist()

  for _, effect_expr in ipairs(node) do
    local effects = specialize.effect(cx, effect_expr)
    if not terralib.islist(effects) then
      effects = terralib.newlist({effects})
    end

    for _, effect in ipairs(effects) do
      if effect:is(ast.specialized.Constraint) then
        constraints:insert(effect)
      elseif effect:is(ast.specialized.Privilege) then
        privileges:insert(effect)
      elseif effect:is(ast.specialized.Coherence) then
        coherence_modes:insert(effect)
      elseif effect:is(ast.specialized.Flag) then
        flags:insert(effect)
      elseif effect:is(ast.specialized.Condition) then
        conditions:insert(effect)
      else
        assert(false, "unexpected node type " .. tostring(node:type()))
      end
    end
  end

  return privileges, coherence_modes, flags, conditions, constraints
end

function specialize.expr_id(cx, node, allow_lists)
  local value = cx.env:lookup(node, node.name)
  return convert_lua_value(cx, node, value, allow_lists)
end

function specialize.expr_escape(cx, node, allow_lists)
  local value = node.expr(cx.env:env())
  return convert_lua_value(cx, node, value, allow_lists)
end

function specialize.expr_constant(cx, node, allow_lists)
  return ast.specialized.expr.Constant {
    value = node.value,
    expr_type = node.expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

-- assumes multi-field accesses have already been flattened by the caller
function specialize.expr_field_access(cx, node, allow_lists)
  local value = specialize.expr(cx, node.value)

  assert(#node.field_names == 1)
  local field_names = specialize.field_names(cx, node.field_names[1])

  if type(field_names[1]) == "string" or data.is_tuple(field_names[1]) then
    if #field_names > 1 then
      report.error(expr, "multi-field access is not allowed")
    end
    field_names = field_names[1]
  end

  if value:is(ast.specialized.expr.LuaTable) then
    if type(field_names) ~= "string" then
      report.error(node, "unable to specialize multi-field access")
    end
    return convert_lua_value(cx, node, value.value[field_names])
  elseif data.is_tuple(field_names) then
    for _, field_name in ipairs(field_names) do
      value = ast.specialized.expr.FieldAccess {
        value = value,
        field_name = field_name,
        annotations = node.annotations,
        span = node.span,
      }
    end
    return value
  else
    return ast.specialized.expr.FieldAccess {
      value = value,
      field_name = field_names,
      annotations = node.annotations,
      span = node.span,
    }
  end
end

function specialize.expr_index_access(cx, node, allow_lists)
  return ast.specialized.expr.IndexAccess {
    value = specialize.expr(cx, node.value),
    index = specialize.expr(cx, node.index),
    annotations = node.annotations,
    span = node.span,
  }
end

local function specialize_expr_list(cx, node)
  assert(terralib.islist(node))
  local result = terralib.newlist()
  for _, arg in ipairs(node) do
    local value = specialize.expr(cx, arg, true)
    if terralib.islist(value) then
      result:insertall(value)
    else
      result:insert(value)
    end
  end
  return result
end

function specialize.expr_method_call(cx, node, allow_lists)
  return ast.specialized.expr.MethodCall {
    value = specialize.expr(cx, node.value),
    method_name = node.method_name,
    args = specialize_expr_list(cx, node.args),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_call(cx, node, allow_lists)
  local fn = specialize.expr(cx, node.fn)

  if not (fn:is(ast.specialized.expr.Function) or
          fn:is(ast.specialized.expr.ID))
  then
    report.error(fn, "unable to specialize complex expression in function call position")
  end

  if terralib.isfunction(fn.value) or
    terralib.isoverloadedfunction(fn.value) or
    terralib.ismacro(fn.value) or
    std.is_task(fn.value) or
    std.is_math_fn(fn.value) or
    std.is_macro(fn.value) or
    type(fn.value) == "cdata"
  then
    if not std.is_task(fn.value) and #node.conditions > 0 then
      report.error(node.conditions[1],
        "terra function call cannot have conditions")
    end
    return ast.specialized.expr.Call {
      fn = fn,
      args = specialize_expr_list(cx, node.args),
      conditions = specialize.expr_conditions(cx, node.conditions),
      annotations = node.annotations,
      span = node.span,
    }
  elseif terralib.types.istype(fn.value) then
    return ast.specialized.expr.Cast {
      fn = fn,
      args = specialize_expr_list(cx, node.args),
      annotations = node.annotations,
      span = node.span,
    }
  else
    report.error(fn, "unable to specialize non-function in function call position")
  end
end

function specialize.expr_ctor_list_field(cx, node, allow_lists)
  local value = specialize.expr(cx, node.value, true)

  local results = terralib.newlist()
  if terralib.islist(value) then
    results:insertall(value)
  else
    results:insert(value)
  end

  return results:map(
    function(result)
      return ast.specialized.expr.CtorListField {
        value = result,
        annotations = node.annotations,
        span = node.span,
      }
    end)
end

function specialize.expr_ctor_rec_field(cx, node, allow_lists)
  local name = node.name_expr(cx.env:env())
  if type(name) ~= "string" then
    report.error(node, "expected a string but found " .. tostring(type(name)))
  end

  return terralib.newlist({
    ast.specialized.expr.CtorRecField {
      name = name,
      value = specialize.expr(cx, node.value),
      annotations = node.annotations,
      span = node.span,
    }
  })
end

function specialize.expr_ctor_field(cx, node, allow_lists)
  if node:is(ast.unspecialized.expr.CtorListField) then
    return specialize.expr_ctor_list_field(cx, node)
  elseif node:is(ast.unspecialized.expr.CtorRecField) then
    return specialize.expr_ctor_rec_field(cx, node)
  else
    assert(false)
  end
end

function specialize.expr_ctor(cx, node, allow_lists)
  local fields = data.flatmap(
    function(field) return specialize.expr_ctor_field(cx, field) end,
    node.fields)

  -- Validate that fields are either all named or all unnamed.
  local all_named = false
  local all_unnamed = false
  for _, field in ipairs(fields) do
    if field:is(ast.specialized.expr.CtorRecField) then
      if all_unnamed then
        report.error(node, "some entries in constructor are named while others are not")
      end
      all_named = true
    elseif field:is(ast.specialized.expr.CtorListField) then
      if all_named then
        report.error(node, "some entries in constructor are named while others are not")
      end
      all_unnamed = true
    else
      assert(false)
    end
  end

  return ast.specialized.expr.Ctor {
    fields = fields,
    named = all_named,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_context(cx, node, allow_lists)
  return ast.specialized.expr.RawContext {
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_fields(cx, node, allow_lists)
  return ast.specialized.expr.RawFields {
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_future(cx, node, allow_lists)
  return ast.specialized.expr.RawFuture {
    value_type = node.value_type_expr(cx.env:env()),
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_physical(cx, node, allow_lists)
  return ast.specialized.expr.RawPhysical {
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_runtime(cx, node, allow_lists)
  return ast.specialized.expr.RawRuntime {
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_task(cx, node, allow_lists)
  return ast.specialized.expr.RawTask {
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_raw_value(cx, node, allow_lists)
  return ast.specialized.expr.RawValue {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_isnull(cx, node, allow_lists)
  local pointer = specialize.expr(cx, node.pointer)
  return ast.specialized.expr.Isnull {
    pointer = pointer,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_new(cx, node, allow_lists)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  if not std.is_bounded_type(pointer_type) then
    report.error(node, "new requires bounded type, got " .. tostring(pointer_type))
  end
  local bounds = pointer_type.bounds_symbols
  if #bounds ~= 1 then
    report.error(node, "new requires bounded type with exactly one region, got " .. tostring(pointer_type))
  end
  local region = ast.specialized.expr.ID {
    value = bounds[1],
    annotations = node.annotations,
    span = node.span,
  }
  return ast.specialized.expr.New {
    pointer_type = pointer_type,
    extent = node.extent and specialize.expr(cx, node.extent),
    region = region,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_null(cx, node, allow_lists)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  return ast.specialized.expr.Null {
    pointer_type = pointer_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_dynamic_cast(cx, node, allow_lists)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.DynamicCast {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_static_cast(cx, node, allow_lists)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.StaticCast {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_unsafe_cast(cx, node, allow_lists)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.expr.UnsafeCast {
    value = value,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_ispace(cx, node, allow_lists)
  local index_type = node.index_type_expr(cx.env:env())
  return ast.specialized.expr.Ispace {
    index_type = index_type,
    extent = specialize.expr(cx, node.extent),
    start = node.start and specialize.expr(cx, node.start),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_region(cx, node, allow_lists)
  local ispace = specialize.expr(cx, node.ispace)
  local fspace_type = node.fspace_type_expr(cx.env:env())
  if not fspace_type then
    report.error(node, "fspace type is undefined or nil")
  end
  return ast.specialized.expr.Region {
    ispace = ispace,
    fspace_type = fspace_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_partition(cx, node, allow_lists)
  return ast.specialized.expr.Partition {
    disjointness = specialize.disjointness_kind(cx, node.disjointness),
    completeness = node.completeness and specialize.completeness_kind(cx, node.completeness),
    region = specialize.expr(cx, node.region),
    coloring = specialize.expr(cx, node.coloring),
    colors = node.colors and specialize.expr(cx, node.colors),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_partition_equal(cx, node, allow_lists)
  return ast.specialized.expr.PartitionEqual {
    region = specialize.expr(cx, node.region),
    colors = specialize.expr(cx, node.colors),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_partition_by_field(cx, node, allow_lists)
  return ast.specialized.expr.PartitionByField {
    completeness = node.completeness and specialize.completeness_kind(cx, node.completeness),
    region = specialize.expr_region_root(cx, node.region),
    colors = specialize.expr(cx, node.colors),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_partition_by_restriction(cx, node, allow_lists)
  return ast.specialized.expr.PartitionByRestriction {
    disjointness = node.disjointness and specialize.disjointness_kind(cx, node.disjointness),
    completeness = node.completeness and specialize.completeness_kind(cx, node.completeness),
    region = specialize.expr(cx, node.region),
    transform = specialize.expr(cx, node.transform),
    extent = specialize.expr(cx, node.extent),
    colors = specialize.expr(cx, node.colors),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_image(cx, node, allow_lists)
  return ast.specialized.expr.Image {
    disjointness = node.disjointness and specialize.disjointness_kind(cx, node.disjointness),
    completeness = node.completeness and specialize.completeness_kind(cx, node.completeness),
    parent = specialize.expr(cx, node.parent),
    partition = specialize.expr(cx, node.partition),
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_preimage(cx, node, allow_lists)
  return ast.specialized.expr.Preimage {
    disjointness = node.disjointness and specialize.disjointness_kind(cx, node.disjointness),
    completeness = node.completeness and specialize.completeness_kind(cx, node.completeness),
    parent = specialize.expr(cx, node.parent),
    partition = specialize.expr(cx, node.partition),
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_cross_product(cx, node, allow_lists)
  return ast.specialized.expr.CrossProduct {
    args = node.args:map(
      function(arg) return specialize.expr(cx, arg) end),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_cross_product_array(cx, node, allow_lists)
  return ast.specialized.expr.CrossProductArray {
    lhs = specialize.expr(cx, node.lhs),
    disjointness = specialize.disjointness_kind(cx, node.disjointness),
    colorings = specialize.expr(cx, node.colorings),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_slice_partition(cx, node, allow_lists)
  return ast.specialized.expr.ListSlicePartition {
    partition = specialize.expr(cx, node.partition),
    indices = specialize.expr(cx, node.indices),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_duplicate_partition(cx, node, allow_lists)
  return ast.specialized.expr.ListDuplicatePartition {
    partition = specialize.expr(cx, node.partition),
    indices = specialize.expr(cx, node.indices),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_cross_product(cx, node, allow_lists)
  return ast.specialized.expr.ListCrossProduct {
    lhs = specialize.expr(cx, node.lhs),
    rhs = specialize.expr(cx, node.rhs),
    shallow = node.shallow,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_cross_product_complete(cx, node, allow_lists)
  return ast.specialized.expr.ListCrossProductComplete {
    lhs = specialize.expr(cx, node.lhs),
    product = specialize.expr(cx, node.product),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_phase_barriers(cx, node, allow_lists)
  return ast.specialized.expr.ListPhaseBarriers {
    product = specialize.expr(cx, node.product),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_invert(cx, node, allow_lists)
  return ast.specialized.expr.ListInvert {
    rhs = specialize.expr(cx, node.rhs),
    product = specialize.expr(cx, node.product),
    barriers = specialize.expr(cx, node.barriers),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_range(cx, node, allow_lists)
  return ast.specialized.expr.ListRange {
    start = specialize.expr(cx, node.start),
    stop = specialize.expr(cx, node.stop),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_ispace(cx, node, allow_lists)
  return ast.specialized.expr.ListIspace {
    ispace = specialize.expr(cx, node.ispace),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_list_from_element(cx, node, allow_lists)
  return ast.specialized.expr.ListFromElement {
    list = specialize.expr(cx, node.list),
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_phase_barrier(cx, node, allow_lists)
  return ast.specialized.expr.PhaseBarrier {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_dynamic_collective(cx, node, allow_lists)
  local value_type = node.value_type_expr(cx.env:env())
  return ast.specialized.expr.DynamicCollective {
    value_type = value_type,
    op = node.op,
    arrivals = specialize.expr(cx, node.arrivals),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_dynamic_collective_get_result(cx, node, allow_lists)
  return ast.specialized.expr.DynamicCollectiveGetResult {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_advance(cx, node, allow_lists)
  return ast.specialized.expr.Advance {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_adjust(cx, node, allow_lists)
  return ast.specialized.expr.Adjust {
    barrier = specialize.expr(cx, node.barrier),
    value = node.value and specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_arrive(cx, node, allow_lists)
  return ast.specialized.expr.Arrive {
    barrier = specialize.expr(cx, node.barrier),
    value = node.value and specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_await(cx, node, allow_lists)
  return ast.specialized.expr.Await {
    barrier = specialize.expr(cx, node.barrier),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_copy(cx, node, allow_lists)
  return ast.specialized.expr.Copy {
    src = specialize.expr_region_root(cx, node.src),
    dst = specialize.expr_region_root(cx, node.dst),
    op = node.op,
    conditions = specialize.expr_conditions(cx, node.conditions),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_fill(cx, node, allow_lists)
  return ast.specialized.expr.Fill {
    dst = specialize.expr_region_root(cx, node.dst),
    value = specialize.expr(cx, node.value),
    conditions = specialize.expr_conditions(cx, node.conditions),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_acquire(cx, node, allow_lists)
  return ast.specialized.expr.Acquire {
    region = specialize.expr_region_root(cx, node.region),
    conditions = specialize.expr_conditions(cx, node.conditions),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_release(cx, node, allow_lists)
  return ast.specialized.expr.Release {
    region = specialize.expr_region_root(cx, node.region),
    conditions = specialize.expr_conditions(cx, node.conditions),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_attach_hdf5(cx, node, allow_lists)
  return ast.specialized.expr.AttachHDF5 {
    region = specialize.expr_region_root(cx, node.region),
    filename = specialize.expr(cx, node.filename),
    mode = specialize.expr(cx, node.mode),
    field_map = node.field_map and specialize.expr(cx, node.field_map),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_detach_hdf5(cx, node, allow_lists)
  return ast.specialized.expr.DetachHDF5 {
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_allocate_scratch_fields(cx, node, allow_lists)
  return ast.specialized.expr.AllocateScratchFields {
    region = specialize.expr_region_root(cx, node.region),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_with_scratch_fields(cx, node, allow_lists)
  return ast.specialized.expr.WithScratchFields {
    region = specialize.expr_region_root(cx, node.region),
    field_ids = specialize.expr(cx, node.field_ids),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_unary(cx, node, allow_lists)
  return ast.specialized.expr.Unary {
    op = node.op,
    rhs = specialize.expr(cx, node.rhs),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_binary(cx, node, allow_lists)
  return ast.specialized.expr.Binary {
    op = node.op,
    lhs = specialize.expr(cx, node.lhs),
    rhs = specialize.expr(cx, node.rhs),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_deref(cx, node, allow_lists)
  return ast.specialized.expr.Deref {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_address_of(cx, node, allow_lists)
  return ast.specialized.expr.AddressOf {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_import_ispace(cx, node)
  return ast.specialized.expr.ImportIspace {
    index_type = node.index_type_expr(cx.env:env()),
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_import_region(cx, node)
  local fspace_type = node.fspace_type_expr(cx.env:env())
  return ast.specialized.expr.ImportRegion {
    ispace = specialize.expr(cx, node.ispace),
    fspace_type = fspace_type,
    value = specialize.expr(cx, node.value),
    field_ids = specialize.expr(cx, node.field_ids),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_import_partition(cx, node)
  return ast.specialized.expr.ImportPartition {
    disjointness = node.disjointness,
    region = specialize.expr(cx, node.region),
    colors = specialize.expr(cx, node.colors),
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.expr_import_cross_product(cx, node)
  return ast.specialized.expr.ImportCrossProduct {
    partitions = specialize_expr_list(cx, node.partitions),
    colors = specialize.expr(cx, node.colors),
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.projection_field(cx, node)
  local renames = node.rename and specialize.field_names(cx, node.rename)
  local field_names = specialize.field_names(cx, node.field_name)
  if renames and #renames ~= #field_names then
    report.error(node, "mismatch in specialization: expected " .. tostring(#renames) ..
        " fields to rename but got " .. tostring(#field_names) .. " fields")
  end
  local fields = specialize.projection_fields(cx, node.fields)
  if renames then
    return data.zip(renames, field_names):map(
      function(pair)
        local rename, field_path = unpack(pair)
        if not data.is_tuple(field_path) then
          field_path = data.newtuple(field_path)
        end
        local result = fields
        for i = #field_path, 1, -1 do
          result = terralib.newlist({
            ast.specialized.projection.Field {
              rename = (i == 1 and rename) or false,
              field_name = field_path[i],
              fields = result,
              span = node.span,
            }})
        end
        return result[1]
      end)
  else
    return field_names:map(
      function(field_path)
        if not data.is_tuple(field_path) then
          field_path = data.newtuple(field_path)
        end
        local result = fields
        for i = #field_path, 1, -1 do
          result = terralib.newlist({
            ast.specialized.projection.Field {
              rename = false,
              field_name = field_path[i],
              fields = result,
              span = node.span,
            }})
        end
        return result[1]
      end)
  end
end

function specialize.projection_fields(cx, node)
  return node and data.flatmap(
    function(field) return specialize.projection_field(cx, field) end,
    node)
end

function specialize.expr_projection(cx, node)
  local region = specialize.expr(cx, node.region)
  local fields = specialize.projection_fields(cx, node.fields)
  return ast.specialized.expr.Projection {
    region = region,
    fields = fields,
    span = node.span,
    annotations = node.annotations,
  }
end

function specialize.expr(cx, node, allow_lists)
  if node:is(ast.unspecialized.expr.ID) then
    return specialize.expr_id(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Escape) then
    return specialize.expr_escape(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Constant) then
    return specialize.expr_constant(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.FieldAccess) then
    return specialize.expr_field_access(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.IndexAccess) then
    return specialize.expr_index_access(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.MethodCall) then
    return specialize.expr_method_call(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Call) then
    return specialize.expr_call(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Ctor) then
    return specialize.expr_ctor(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawContext) then
    return specialize.expr_raw_context(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawFields) then
    return specialize.expr_raw_fields(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawFuture) then
    return specialize.expr_raw_future(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawPhysical) then
    return specialize.expr_raw_physical(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawRuntime) then
    return specialize.expr_raw_runtime(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawTask) then
    return specialize.expr_raw_task(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.RawValue) then
    return specialize.expr_raw_value(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Isnull) then
    return specialize.expr_isnull(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.New) then
    return specialize.expr_new(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Null) then
    return specialize.expr_null(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.DynamicCast) then
    return specialize.expr_dynamic_cast(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.StaticCast) then
    return specialize.expr_static_cast(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.UnsafeCast) then
    return specialize.expr_unsafe_cast(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Ispace) then
    return specialize.expr_ispace(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Region) then
    return specialize.expr_region(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Partition) then
    return specialize.expr_partition(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.PartitionEqual) then
    return specialize.expr_partition_equal(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.PartitionByField) then
    return specialize.expr_partition_by_field(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.PartitionByRestriction) then
    return specialize.expr_partition_by_restriction(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Image) then
    return specialize.expr_image(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Preimage) then
    return specialize.expr_preimage(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.CrossProduct) then
    return specialize.expr_cross_product(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.CrossProductArray) then
    return specialize.expr_cross_product_array(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListSlicePartition) then
    return specialize.expr_list_slice_partition(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListDuplicatePartition) then
    return specialize.expr_list_duplicate_partition(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListCrossProduct) then
    return specialize.expr_list_cross_product(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListCrossProductComplete) then
    return specialize.expr_list_cross_product_complete(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListPhaseBarriers) then
    return specialize.expr_list_phase_barriers(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListInvert) then
    return specialize.expr_list_invert(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListRange) then
    return specialize.expr_list_range(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ListIspace) then
    return specialize.expr_list_ispace(cx, node)

  elseif node:is(ast.unspecialized.expr.ListFromElement) then
    return specialize.expr_list_from_element(cx, node)

  elseif node:is(ast.unspecialized.expr.PhaseBarrier) then
    return specialize.expr_phase_barrier(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.DynamicCollective) then
    return specialize.expr_dynamic_collective(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Advance) then
    return specialize.expr_advance(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Adjust) then
    return specialize.expr_adjust(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Arrive) then
    return specialize.expr_arrive(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Await) then
    return specialize.expr_await(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.DynamicCollectiveGetResult) then
    return specialize.expr_dynamic_collective_get_result(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Copy) then
    return specialize.expr_copy(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Fill) then
    return specialize.expr_fill(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Acquire) then
    return specialize.expr_acquire(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Release) then
    return specialize.expr_release(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.AttachHDF5) then
    return specialize.expr_attach_hdf5(cx, node)

  elseif node:is(ast.unspecialized.expr.DetachHDF5) then
    return specialize.expr_detach_hdf5(cx, node)

  elseif node:is(ast.unspecialized.expr.AllocateScratchFields) then
    return specialize.expr_allocate_scratch_fields(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.WithScratchFields) then
    return specialize.expr_with_scratch_fields(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Unary) then
    return specialize.expr_unary(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Binary) then
    return specialize.expr_binary(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.Deref) then
    return specialize.expr_deref(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.AddressOf) then
    return specialize.expr_address_of(cx, node, allow_lists)

  elseif node:is(ast.unspecialized.expr.ImportIspace) then
    return specialize.expr_import_ispace(cx, node)

  elseif node:is(ast.unspecialized.expr.ImportRegion) then
    return specialize.expr_import_region(cx, node)

  elseif node:is(ast.unspecialized.expr.ImportPartition) then
    return specialize.expr_import_partition(cx, node)

  elseif node:is(ast.unspecialized.expr.ImportCrossProduct) then
    return specialize.expr_import_cross_product(cx, node)

  elseif node:is(ast.unspecialized.expr.Projection) then
    return specialize.expr_projection(cx, node)

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
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_elseif(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.stat.Elseif {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_while(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.stat.While {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

local function make_symbol(cx, node, var_name, var_type)
  if type(var_name) == "string" then
    return var_name, std.newsymbol(var_type or nil, var_name)
  end

  var_name = var_name(cx.env:env())
  if std.is_symbol(var_name) then
    if cx.is_quote then
      return var_name, var_name
    else
      return var_name, std.newsymbol(var_name:hastype(), var_name:hasname())
    end
  end

  report.error(node, "mismatch in specialization: expected a symbol but got value of type " .. tostring(type(var_name)))
end

local function convert_index_launch_annotations(node)
  if node.annotations.parallel:is(ast.annotation.Demand) then
    if not (node.annotations.index_launch:is(ast.annotation.Allow) or
              node.annotations.index_launch:is(ast.annotation.Demand))
    then
      report.error(node, "conflicting annotations for __parallel and __index_launch")
    end
    report.error(node, "__demand(__parallel) is no longer supported on loops, please use __demand(__index_launch)")
  elseif node.annotations.parallel:is(ast.annotation.Forbid) then
    if not (node.annotations.index_launch:is(ast.annotation.Allow) or
              node.annotations.index_launch:is(ast.annotation.Forbid))
    then
      report.error(node, "conflicting annotations for __parallel and __index_launch")
    end
    report.error(node, "__forbid(__parallel) is no longer supported on loops, please use __forbid(__index_launch)")
  else
    return node.annotations
  end
end

function specialize.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return specialize.expr(cx, value) end)

  local var_type
  if node.type_expr then
    var_type = node.type_expr(cx.env:env())
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_name, symbol = make_symbol(cx, node, node.name)
  if std.is_symbol(var_name) then
    cx.mapping[var_name] = symbol
  else
    cx.env:insert(node, symbol, symbol)
  end
  cx.env:insert(node, var_name, symbol)

  if var_type then
    symbol:settype(var_type)
  end

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.stat.ForNum {
    symbol = symbol,
    values = values,
    block = block,
    annotations = convert_index_launch_annotations(node),
    span = node.span,
  }
end

function specialize.stat_for_list(cx, node)
  local value = specialize.expr(cx, node.value)

  local var_type
  if node.type_expr then
    var_type = node.type_expr(cx.env:env())
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_name, symbol = make_symbol(cx, node, node.name)
  if std.is_symbol(var_name) then
    cx.mapping[var_name] = symbol
  else
    cx.env:insert(node, symbol, symbol)
  end
  cx.env:insert(node, var_name, symbol)

  if var_type then
    symbol:settype(var_type)
  end

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.stat.ForList {
    symbol = symbol,
    value = value,
    block = block,
    annotations = convert_index_launch_annotations(node),
    span = node.span,
  }
end

function specialize.stat_repeat(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.Repeat {
    block = specialize.block(cx, node.block),
    until_cond = specialize.expr(cx, node.until_cond),
    annotations = convert_index_launch_annotations(node),
    span = node.span,
  }
end

function specialize.stat_must_epoch(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.MustEpoch {
    block = specialize.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.stat.Block {
    block = specialize.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_var(cx, node)
  -- Hack: To handle recursive regions, need to put a proxy into place
  -- before looking at either types or values.
  local symbols = terralib.newlist()
  for i, var_name in ipairs(node.var_names) do
    if node.values[i] and node.values[i]:is(ast.unspecialized.expr.Region) then
      local var_name, symbol = make_symbol(cx, node, var_name)
      if std.is_symbol(var_name) then
        cx.mapping[var_name] = symbol
      else
        cx.env:insert(node, symbol, symbol)
      end
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
      var_name, symbol = make_symbol(cx, node, var_name, var_type)
      if std.is_symbol(var_name) then
        cx.mapping[var_name] = symbol
      else
        cx.env:insert(node, symbol, symbol)
      end
      cx.env:insert(node, var_name, symbol)
      symbols[i] = symbol
    end
  end

  return ast.specialized.stat.Var {
    symbols = symbols,
    values = values,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_var_unpack(cx, node)
  local symbols = terralib.newlist()
  for _, var_name in ipairs(node.var_names) do
    local symbol = std.newsymbol(var_name)
    cx.env:insert(node, var_name, symbol)
    cx.env:insert(node, symbol, symbol)
    symbols:insert(symbol)
  end

  local value = specialize.expr(cx, node.value)

  return ast.specialized.stat.VarUnpack {
    symbols = symbols,
    fields = node.fields,
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_return(cx, node)
  return ast.specialized.stat.Return {
    value = node.value and specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_break(cx, node)
  return ast.specialized.stat.Break {
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_assignment(cx, node)
  return ast.specialized.stat.Assignment {
    lhs = node.lhs:map(
      function(lh) return specialize.expr(cx, lh) end),
    rhs = node.rhs:map(
      function(rh) return specialize.expr(cx, rh) end),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_reduce(cx, node)
  return ast.specialized.stat.Reduce {
    lhs = node.lhs:map(
      function(lh) return specialize.expr(cx, lh) end),
    rhs = node.rhs:map(
      function(rh) return specialize.expr(cx, rh) end),
    op = node.op,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_expr(cx, node)
  local value = specialize.expr(cx, node.expr, true)
  if not terralib.islist(value) then
    value = terralib.newlist({value})
  end
  return value:map(
    function(expr)
      return ast.specialized.stat.Expr {
        expr = expr,
        annotations = node.annotations,
        span = node.span,
      }
    end)
end

local function get_quote_contents(cx, expr)
  assert(std.is_rquote(expr))

  local value = expr:getast()
  if value:is(ast.specialized.top.QuoteExpr) then
    assert(value.expr:is(ast.specialized.expr))
    if not cx.is_quote then
      value = alpha_convert.entry(value, cx.env, cx.mapping)
    end
    return terralib.newlist({
      ast.specialized.stat.Expr {
        expr = value.expr,
        annotations = value.annotations,
        span = value.span,
      },
    })
  elseif value:is(ast.specialized.top.QuoteStat) then
    assert(value.block:is(ast.specialized.Block))
    if not cx.is_quote then
      value = alpha_convert.entry(value, cx.env, cx.mapping)
    end
    return value.block.stats
  else
    assert(false)
  end
end

function specialize.stat_escape(cx, node)
  local expr = node.expr(cx.env:env())
  if std.is_rquote(expr) then
    return get_quote_contents(cx, expr)
  elseif terralib.islist(expr) then
    if not data.all(expr:map(function(v) return std.is_rquote(v) end)) then
      report.error(node, "unable to specialize value of type " .. tostring(type(expr)))
    end
    return data.flatmap(function(x) return get_quote_contents(cx, x) end, expr)
  else
    report.error(node, "unable to specialize value of type " .. tostring(type(expr)))
  end
end

-- Hack: use a separate global symbol table to track stracked
-- rescapes/remits. This is basically implementing a stack, but I'm
-- lazy and don't want to write a dedicated stack just to do this.
--
-- Why not use the normal symbol table? Because rescape shouldn't
-- actually introduce a new scope.
local rescape_remit_key = std.newsymbol("rescape_remit_key")
local rescape_remit_env = symbol_table.new_global_scope({})
function specialize.stat_rescape(cx, node)
  local result = terralib.newlist()

  -- This is ugly, but we have to push/pop the rescape scope because
  -- we can only pass one environment through the Lua function that
  -- evaluates the escape contents.
  local old_env = rescape_remit_env
  rescape_remit_env = rescape_remit_env:new_local_scope()
  rescape_remit_env:insert(node, rescape_remit_key, result)

  -- Now run the escape code. It will populate the result by looking
  -- up the escape key.
  node.stats(cx.env:env())

  local result = data.flatmap(function(x) return get_quote_contents(cx, x) end, result)

  rescape_remit_env = old_env
  return result
end

function specialize.stat_raw_delete(cx, node)
  return ast.specialized.stat.RawDelete {
    value = specialize.expr(cx, node.value),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_fence(cx, node)
  return ast.specialized.stat.Fence {
    kind = node.kind,
    blocking = node.blocking,
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_parallelize_with(cx, node)
  local hints = data.flatmap(function(expr)
    return specialize.expr(cx, expr, true) end, node.hints)
  hints = hints:map(function(hint)
    if not (hint:is(ast.specialized.expr.ID) or hint:is(ast.specialized.expr.Binary)) then
      report.error(hint, "parallelizer hint should be a partition or constraint on two partitions")
    elseif hint:is(ast.specialized.expr.Binary) then
      if not (hint.op == "<=" or hint.op == ">=") then
        report.error(hint, "operator '" .. hint.op .."' is not supported in parallelizer hints")
      end
      if hint.op == ">=" then
        hint = hint {
          lhs = hint.rhs,
          rhs = hint.lhs,
          op = "<="
        }
      end
      if not ((hint.lhs:is(ast.specialized.expr.Image) and hint.rhs:is(ast.specialized.expr.ID))) then
        report.error(node, "unsupported constraint for parallelizer hints")
      end
    end
    return hint
  end)

  local cx = cx:new_local_scope()
  return ast.specialized.stat.ParallelizeWith {
    hints = hints,
    block = specialize.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.stat_parallel_prefix(cx, node)
  local lhs = specialize.expr_region_root(cx, node.lhs)
  local rhs = specialize.expr_region_root(cx, node.rhs)
  local op = node.op
  local dir = specialize.expr(cx, node.dir)
  return ast.specialized.stat.ParallelPrefix {
    lhs = lhs,
    rhs = rhs,
    op = op,
    dir = dir,
    annotations = node.annotations,
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
    return specialize.stat_assignment(cx, node)

  elseif node:is(ast.unspecialized.stat.Reduce) then
    return specialize.stat_reduce(cx, node)

  elseif node:is(ast.unspecialized.stat.Expr) then
    return specialize.stat_expr(cx, node)

  elseif node:is(ast.unspecialized.stat.Escape) then
    return specialize.stat_escape(cx, node)

  elseif node:is(ast.unspecialized.stat.Rescape) then
    return specialize.stat_rescape(cx, node)

  elseif node:is(ast.unspecialized.stat.RawDelete) then
    return specialize.stat_raw_delete(cx, node)

  elseif node:is(ast.unspecialized.stat.Fence) then
    return specialize.stat_fence(cx, node)

  elseif node:is(ast.unspecialized.stat.ParallelizeWith) then
    return specialize.stat_parallelize_with(cx, node)

  elseif node:is(ast.unspecialized.stat.ParallelPrefix) then
    return specialize.stat_parallel_prefix(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local function make_symbols(cx, node, var_name)
  if type(var_name) == "string" then
    return terralib.newlist({{var_name, std.newsymbol(var_name)}})
  end

  var_name = var_name(cx.env:env())
  if std.is_symbol(var_name) then
    return terralib.newlist({{var_name, var_name}})
  elseif terralib.islist(var_name) then
    if not data.all(unpack(var_name:map(std.is_symbol))) then
      report.error(node, "param list contains non-symbol")
    end
    return var_name:map(function(v) return {v, v} end)
  end

  report.error(node, "mismatch in specialization: expected a symbol or list of symbols but got value of type " .. tostring(type(var_name)))
end

function specialize.top_task_param(cx, node, params)
  local result = terralib.newlist()
  for _, param in ipairs(params) do
    local param_name, symbol = unpack(param)

    local param_type
    if std.is_symbol(param_name) then
      if not param_name:hastype() then
        report.error(node, "param symbol must be typed")
      end
      param_type = param_name:gettype()
    else
      param_type = node.type_expr(cx.env:env())
    end

    if not param_type then
      report.error(node, "param type is undefined or nil")
    end
    if not terralib.types.istype(param_type) then
      report.error(node, "param type is not a type")
    end

    -- If the type is a future, strip that out and record it separately.
    local future = false
    if std.is_future(param_type) then
      future = true
      param_type = param_type.result_type
    end

    if not symbol:hastype() then
      symbol:settype(param_type)
    end
    assert(std.type_eq(symbol:gettype(), param_type))

    result:insert(
      ast.specialized.top.TaskParam {
        symbol = symbol,
        future = future,
        annotations = node.annotations,
        span = node.span,
      })
  end
  return result
end

function specialize.top_task_params(cx, node)
  -- Hack: Params which are regions can be recursive on the name of
  -- the region so introduce the symbol before completing specialization to allow
  -- for this recursion.
  local symbols = node:map(
    function(node)
      local params = make_symbols(cx, node, node.param_name)
      for _, param in ipairs(params) do
        local param_name, symbol = unpack(param)
        cx.env:insert(node, param_name, symbol)
        if not std.is_symbol(param_name) then cx.env:insert(node, symbol, symbol) end
      end

      return terralib.newlist({node, params})
    end)

  return data.flatmap(
    function(group)
      local param, symbol = unpack(group)
      return specialize.top_task_param(cx, param, symbol)
    end,
    symbols)
end

function specialize.top_task(cx, node)
  local cx = cx:new_local_scope()

  local task = std.new_task(node.name, node.span, not node.body)

  if node.body then
    local variant = task:make_variant("primary")
    task:set_primary_variant(variant)
    variant:set_is_inline(node.annotations.inline:is(ast.annotation.Demand))
  end

  task:set_is_local(node.annotations.local_launch:is(ast.annotation.Demand))

  if #node.name == 1 then
    cx.env:insert(node, node.name[1], task)
  end
  cx = cx:new_local_scope()

  local params = specialize.top_task_params(cx, node.params)
  local return_type = node.return_type_expr(cx.env:env())
  local privileges, coherence_modes, flags, conditions, constraints =
    specialize.effects(cx, node.effect_exprs)
  local body = node.body and specialize.block(cx, node.body)

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
    prototype = task,
    annotations = node.annotations,
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
      report.error(node, "parameters " .. tostring(symbol) .. " and " ..
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
    report.error(node, "field type is undefined or nil")
  end
  symbol:settype(field_type)

  -- Check for fields with duplicate types.
  if std.type_supports_constraints(field_type) then
    if mapping[field_type] then
      report.error(node, "fields " .. tostring(symbol) .. " and " ..
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
    annotations = node.annotations,
    span = node.span,
  }
end

function specialize.top_remit(cx, node)
  local result = rescape_remit_env:safe_lookup(rescape_remit_key)
  if not result then
    report.error(node, "remit must be used inside of rescape")
  end

  local expr = node.expr(cx.env:env())
  if std.is_rquote(expr) then
    result:insert(expr)
  elseif terralib.islist(expr) then
    if not data.all(expr:map(function(v) return std.is_rquote(v) end)) then
      report.error(node, "unable to specialize value of type " .. tostring(type(expr)))
    end
    result:insertall(expr)
  else
    report.error(node, "unable to specialize value of type " .. tostring(type(expr)))
  end

  return false -- nothing to do at this point, statement has been executed
end

function specialize.top_quote_expr(cx, node)
  local cx = cx:new_local_scope(true)
  return std.newrquote(ast.specialized.top.QuoteExpr {
    expr = specialize.expr(cx, node.expr),
    expr_type = false,
    annotations = node.annotations,
    span = node.span,
  })
end

function specialize.top_quote_stat(cx, node)
  local cx = cx:new_local_scope(true)
  return std.newrquote(ast.specialized.top.QuoteStat {
    block = specialize.block(cx, node.block),
    annotations = node.annotations,
    span = node.span,
  })
end

function specialize.top(cx, node)
  if node:is(ast.unspecialized.top.Task) then
    return specialize.top_task(cx, node)

  elseif node:is(ast.unspecialized.top.Fspace) then
    return specialize.top_fspace(cx, node)

  elseif node:is(ast.unspecialized.top.Remit) then
    return specialize.top_remit(cx, node)

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
