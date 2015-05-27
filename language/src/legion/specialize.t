-- Copyright 2015 Stanford University
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

local ast = require("legion/ast")
local log = require("legion/log")
local std = require("legion/std")
local symbol_table = require("legion/symbol_table")

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
  end
end

function convert_lua_value(cx, node, value)
  if type(value) == "number" or type(value) == "boolean" then
    local expr_type = guess_type_for_literal(value)
    return ast.specialized.ExprConstant {
      value = value,
      expr_type = expr_type,
      span = node.span,
    }
  elseif type(value) == "function" or terralib.isfunction(value) or
    terralib.isfunctiondefinition(value) or terralib.ismacro(value) or
    terralib.types.istype(value) or std.is_task(value)
  then
    return ast.specialized.ExprFunction {
      value = value,
      span = node.span,
    }
  elseif terralib.isconstant(value) then
    if value.type then
      return ast.specialized.ExprConstant {
        value = value.object,
        expr_type = value.type,
        span = node.span,
      }
    else
      local expr_type = guess_type_for_literal(value.object)
      return ast.specialized.ExprConstant {
        value = value.object,
        expr_type = expr_type,
        span = node.span,
      }
    end
  elseif terralib.issymbol(value) then
    return ast.specialized.ExprID {
      value = value,
      span = node.span,
    }
  elseif type(value) == "table" then
    return ast.specialized.ExprLuaTable {
      value = value,
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

  if node:is(ast.unspecialized.ExprID) then
    return 1

  elseif node:is(ast.unspecialized.ExprEscape) then
    if get_num_accessed_fields(node.expr) > 1 then return false
    else return 1 end

  elseif node:is(ast.unspecialized.ExprConstant) then
    return 1

  elseif node:is(ast.unspecialized.ExprFieldAccess) then
    return get_num_accessed_fields(node.value) * #node.field_names

  elseif node:is(ast.unspecialized.ExprIndexAccess) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    if get_num_accessed_fields(node.index) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprMethodCall) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    node.args:map(function(arg)
      if get_num_accessed_fields(arg) > 1 then return false end
    end)
    return 1

  elseif node:is(ast.unspecialized.ExprCall) then
    if get_num_accessed_fields(node.fn) > 1 then return false end
    node.args:map(function(arg)
      if get_num_accessed_fields(arg) > 1 then return false end
    end)
    return 1

  elseif node:is(ast.unspecialized.ExprCtor) then
    node.fields:map(function(field)
      if field:is(ast.unspecialized.ExprCtorListField) then
        if get_num_accessed_fields(field.value) > 1 then return false end
      elseif field:is(ast.unspecialized.ExprCtorListField) then
        if get_num_accessed_fields(field.num_expr) > 1 then return false end
        if get_num_accessed_fields(field.value) > 1 then return false end
      end
    end)
    return 1

  elseif node:is(ast.unspecialized.ExprRawContext) then
    return 1

  elseif node:is(ast.unspecialized.ExprRawFields) then
    return 1

  elseif node:is(ast.unspecialized.ExprRawPhysical) then
    return 1

  elseif node:is(ast.unspecialized.ExprRawRuntime) then
    return 1

  elseif node:is(ast.unspecialized.ExprIsnull) then
    if get_num_accessed_fields(node.pointer) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprNew) then
    if get_num_accessed_fields(node.pointer_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprNull) then
    if get_num_accessed_fields(node.pointer_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprDynamicCast) then
    if get_num_accessed_fields(node.type_expr) > 1 then return false end
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprStaticCast) then
    if get_num_accessed_fields(node.type_expr) > 1 then return false end
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprIspace) then
    if get_num_accessed_fields(node.element_type_expr) > 1 then return false end
    if get_num_accessed_fields(node.size) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprRegion) then
    if get_num_accessed_fields(node.element_type_expr) > 1 then return false end
    if get_num_accessed_fields(node.size) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprPartition) then
    if get_num_accessed_fields(node.disjointness_expr) > 1 then return false end
    if get_num_accessed_fields(node.region_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprCrossProduct) then
    if get_num_accessed_fields(node.lhs_type_expr) > 1 then return false end
    if get_num_accessed_fields(node.rhs_type_expr) > 1 then return false end
    return 1

  elseif node:is(ast.unspecialized.ExprUnary) then
    return get_num_accessed_fields(node.rhs)

  elseif node:is(ast.unspecialized.ExprBinary) then
    return join_num_accessed_fields(get_num_accessed_fields(node.lhs),
                                    get_num_accessed_fields(node.rhs))

  elseif node:is(ast.unspecialized.ExprDeref) then
    if get_num_accessed_fields(node.value) > 1 then return false end
    return 1

  else
    assert(false, "unreachable")
  end
end

local function get_nth_field_access(node, idx)
  if node:is(ast.unspecialized.ExprFieldAccess) then
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

  elseif node:is(ast.unspecialized.ExprUnary) then
    return node { rhs = get_nth_field_access(node.rhs, idx) }

  elseif node:is(ast.unspecialized.ExprBinary) then
    return node {
      lhs = get_nth_field_access(node.lhs, idx),
      rhs = get_nth_field_access(node.rhs, idx),
    }

  else
    return node
  end
end

local function has_all_valid_field_accesses(node)
  if node:is(ast.unspecialized.StatAssignment) or
     node:is(ast.unspecialized.StatReduce) then

    local valid = true
    std.zip(node.lhs, node.rhs):map(function(pair)
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

function specialize.expr_id(cx, node)
  local value = cx.env:lookup(node, node.name)
  return convert_lua_value(cx, node, value)
end

function specialize.expr_escape(cx, node)
  local value = node.expr(cx.env:env())
  return convert_lua_value(cx, node, value)
end

function specialize.expr_constant(cx, node)
  return ast.specialized.ExprConstant {
    value = node.value,
    expr_type = node.expr_type,
    span = node.span,
  }
end

-- assumes multi-field accesses have already been flattened by the caller
function specialize.expr_field_access(cx, node)
  if #node.field_names ~= 1 then
    log.error(node, "illegal use of multi-field access")
  end
  local value = specialize.expr(cx, node.value)
  if value:is(ast.specialized.ExprLuaTable) then
    return convert_lua_value(cx, node, value.value[node.field_names[1]])
  else
    return ast.specialized.ExprFieldAccess {
      value = value,
      field_name = node.field_names[1],
      span = node.span,
    }
  end
end

function specialize.expr_index_access(cx, node)
  return ast.specialized.ExprIndexAccess {
    value = specialize.expr(cx, node.value),
    index = specialize.expr(cx, node.index),
    span = node.span,
  }
end

function specialize.expr_method_call(cx, node)
  return ast.specialized.ExprMethodCall {
    value = specialize.expr(cx, node.value),
    method_name = node.method_name,
    args = node.args:map(
      function(arg) return specialize.expr(cx, arg) end),
    span = node.span,
  }
end

function specialize.expr_call(cx, node)
  local fn = specialize.expr(cx, node.fn)
  if terralib.isfunction(fn.value) or
    terralib.isfunctiondefinition(fn.value) or
    terralib.ismacro(fn.value) or
    std.is_task(fn.value) or
    type(fn.value) == "function"
  then
    return ast.specialized.ExprCall {
      fn = fn,
      args = node.args:map(
        function(arg) return specialize.expr(cx, arg) end),
      span = node.span,
    }
  elseif terralib.types.istype(fn.value) then
    return ast.specialized.ExprCast {
      fn = fn,
      args = node.args:map(
        function(arg) return specialize.expr(cx, arg) end),
      span = node.span,
    }
  else
    assert(false, "unreachable")
  end
end

function specialize.expr_ctor_list_field(cx, node)
  return ast.specialized.ExprCtorListField {
    value = specialize.expr(cx, node.value),
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

  return ast.specialized.ExprCtorRecField {
    name = name,
    value = specialize.expr(cx, node.value),
    span = node.span,
  }
end

function specialize.expr_ctor_field(cx, node)
  if node:is(ast.unspecialized.ExprCtorListField) then
    return specialize.expr_ctor_list_field(cx, node)
  elseif node:is(ast.unspecialized.ExprCtorRecField) then
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
    if field:is(ast.specialized.ExprCtorRecField) then
      assert(not all_unnamed,
             "some entries in constructor are named while others are not")
      all_named = true
    elseif field:is(ast.specialized.ExprCtorListField) then
      assert(not all_named,
             "some entries in constructor are named while others are not")
      all_unnamed = true
    else
      assert(false)
    end
  end

  return ast.specialized.ExprCtor {
    fields = fields,
    named = all_named,
    span = node.span,
  }
end

function specialize.expr_raw_context(cx, node)
  return ast.specialized.ExprRawContext {
    span = node.span,
  }
end

function specialize.expr_raw_fields(cx, node)
  return ast.specialized.ExprRawFields {
    region = specialize.expr(cx, node.region),
    span = node.span,
  }
end

function specialize.expr_raw_physical(cx, node)
  return ast.specialized.ExprRawPhysical {
    region = specialize.expr(cx, node.region),
    span = node.span,
  }
end

function specialize.expr_raw_runtime(cx, node)
  return ast.specialized.ExprRawRuntime {
    span = node.span,
  }
end

function specialize.expr_isnull(cx, node)
  local pointer = specialize.expr(cx, node.pointer)
  return ast.specialized.ExprIsnull {
    pointer = pointer,
    span = node.span,
  }
end

function specialize.expr_new(cx, node)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  assert(std.is_ptr(pointer_type))
  local regions = pointer_type.points_to_region_symbols
  if #regions ~= 1 then
   log.error(node, "new requires pointer type with exactly one region, got " .. tostring(pointer_type))
  end
  local region = ast.specialized.ExprID {
    value = regions[1],
    span = node.span,
  }
  return ast.specialized.ExprNew {
    pointer_type = pointer_type,
    region = region,
    span = node.span,
  }
end

function specialize.expr_null(cx, node)
  local pointer_type = node.pointer_type_expr(cx.env:env())
  return ast.specialized.ExprNull {
    pointer_type = pointer_type,
    span = node.span,
  }
end

function specialize.expr_dynamic_cast(cx, node)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.ExprDynamicCast {
    value = value,
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_static_cast(cx, node)
  local expr_type = node.type_expr(cx.env:env())
  local value = specialize.expr(cx, node.value)
  return ast.specialized.ExprStaticCast {
    value = value,
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_ispace(cx, node)
  local index_type = node.index_type_expr(cx.env:env())
  local expr_type = std.ispace(index_type)
  return ast.specialized.ExprIspace {
    index_type = index_type,
    lower_bound = specialize.expr(cx, node.lower_bound),
    upper_bound = node.upper_bound and specialize.expr(cx, node.upper_bound),
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_region(cx, node)
  local element_type = node.element_type_expr(cx.env:env())
  local expr_type = std.region(element_type)
  return ast.specialized.ExprRegion {
    element_type = element_type,
    size = specialize.expr(cx, node.size),
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_partition(cx, node)
  local disjointness = node.disjointness_expr(cx.env:env())
  local region_type = node.region_type_expr(cx.env:env())
  -- Hack: Need to do this type checking early because otherwise we
  -- can't construct a type here.
  if disjointness ~= std.disjoint and disjointness ~= std.aliased then
    log.error(node, "type mismatch in argument 1: expected disjoint or aliased but got " ..
                tostring(disjointness))
  end
  local expr_type = std.partition(disjointness, region_type)
  local region = ast.specialized.ExprID {
    value = expr_type.parent_region_symbol,
    span = node.span,
  }
  return ast.specialized.ExprPartition {
    disjointness = disjointness,
    region = region,
    coloring = specialize.expr(cx, node.coloring),
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_cross_product(cx, node)
  local lhs_type = node.lhs_type_expr(cx.env:env())
  local rhs_type = node.rhs_type_expr(cx.env:env())
  local expr_type = std.cross_product(lhs_type, rhs_type)
  local lhs = ast.specialized.ExprID {
    value = expr_type.lhs_partition_symbol,
    span = node.span,
  }
  local rhs = ast.specialized.ExprID {
    value = expr_type.rhs_partition_symbol,
    span = node.span,
  }
  return ast.specialized.ExprCrossProduct {
    lhs = lhs,
    rhs = rhs,
    expr_type = expr_type,
    span = node.span,
  }
end

function specialize.expr_unary(cx, node)
  return ast.specialized.ExprUnary {
    op = node.op,
    rhs = specialize.expr(cx, node.rhs),
    span = node.span,
  }
end

function specialize.expr_binary(cx, node)
  return ast.specialized.ExprBinary {
    op = node.op,
    lhs = specialize.expr(cx, node.lhs),
    rhs = specialize.expr(cx, node.rhs),
    span = node.span,
  }
end

function specialize.expr_deref(cx, node)
  return ast.specialized.ExprDeref {
    value = specialize.expr(cx, node.value),
    span = node.span,
  }
end

function specialize.expr(cx, node)
  if node:is(ast.unspecialized.ExprID) then
    return specialize.expr_id(cx, node)

  elseif node:is(ast.unspecialized.ExprEscape) then
    return specialize.expr_escape(cx, node)

  elseif node:is(ast.unspecialized.ExprConstant) then
    return specialize.expr_constant(cx, node)

  elseif node:is(ast.unspecialized.ExprFieldAccess) then
    return specialize.expr_field_access(cx, node)

  elseif node:is(ast.unspecialized.ExprIndexAccess) then
    return specialize.expr_index_access(cx, node)

  elseif node:is(ast.unspecialized.ExprMethodCall) then
    return specialize.expr_method_call(cx, node)

  elseif node:is(ast.unspecialized.ExprCall) then
    return specialize.expr_call(cx, node)

  elseif node:is(ast.unspecialized.ExprCtor) then
    return specialize.expr_ctor(cx, node)

  elseif node:is(ast.unspecialized.ExprRawContext) then
    return specialize.expr_raw_context(cx, node)

  elseif node:is(ast.unspecialized.ExprRawFields) then
    return specialize.expr_raw_fields(cx, node)

  elseif node:is(ast.unspecialized.ExprRawPhysical) then
    return specialize.expr_raw_physical(cx, node)

  elseif node:is(ast.unspecialized.ExprRawRuntime) then
    return specialize.expr_raw_runtime(cx, node)

  elseif node:is(ast.unspecialized.ExprIsnull) then
    return specialize.expr_isnull(cx, node)

  elseif node:is(ast.unspecialized.ExprNew) then
    return specialize.expr_new(cx, node)

  elseif node:is(ast.unspecialized.ExprNull) then
    return specialize.expr_null(cx, node)

  elseif node:is(ast.unspecialized.ExprDynamicCast) then
    return specialize.expr_dynamic_cast(cx, node)

  elseif node:is(ast.unspecialized.ExprStaticCast) then
    return specialize.expr_static_cast(cx, node)

  elseif node:is(ast.unspecialized.ExprIspace) then
    return specialize.expr_ispace(cx, node)

  elseif node:is(ast.unspecialized.ExprRegion) then
    return specialize.expr_region(cx, node)

  elseif node:is(ast.unspecialized.ExprPartition) then
    return specialize.expr_partition(cx, node)

  elseif node:is(ast.unspecialized.ExprCrossProduct) then
    return specialize.expr_cross_product(cx, node)

  elseif node:is(ast.unspecialized.ExprUnary) then
    return specialize.expr_unary(cx, node)

  elseif node:is(ast.unspecialized.ExprBinary) then
    return specialize.expr_binary(cx, node)

  elseif node:is(ast.unspecialized.ExprDeref) then
    return specialize.expr_deref(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function specialize.block(cx, node)
  return ast.specialized.Block {
    stats = node.stats:map(
      function(stat) return specialize.stat(cx, stat) end),
    span = node.span,
  }
end

function specialize.stat_if(cx, node)
  local then_cx = cx:new_local_scope()
  local else_cx = cx:new_local_scope()
  return ast.specialized.StatIf {
    cond = specialize.expr(cx, node.cond),
    then_block = specialize.block(then_cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return specialize.stat_elseif(cx, block) end),
    else_block = specialize.block(else_cx, node.else_block),
    span = node.span,
  }
end

function specialize.stat_elseif(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.StatElseif {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    span = node.span,
  }
end

function specialize.stat_while(cx, node)
  local body_cx = cx:new_local_scope()
  return ast.specialized.StatWhile {
    cond = specialize.expr(cx, node.cond),
    block = specialize.block(body_cx, node.block),
    span = node.span,
  }
end

function specialize.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return specialize.expr(cx, value) end)

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type = node.type_expr(cx.env:env())
  local symbol = terralib.newsymbol(var_type, node.name)
  cx.env:insert(node, node.name, symbol)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.StatForNum {
    symbol = symbol,
    values = values,
    block = block,
    parallel = node.parallel,
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
  local symbol = terralib.newsymbol(var_type, node.name)
  cx.env:insert(node, node.name, symbol)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  local block = specialize.block(cx, node.block)

  return ast.specialized.StatForList {
    symbol = symbol,
    value = value,
    block = block,
    vectorize = node.vectorize,
    span = node.span,
  }
end

function specialize.stat_repeat(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.StatRepeat {
    block = specialize.block(cx, node.block),
    until_cond = specialize.expr(cx, node.until_cond),
    span = node.span,
  }
end

function specialize.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return ast.specialized.StatBlock {
    block = specialize.block(cx, node.block),
    span = node.span,
  }
end

function specialize.stat_var(cx, node)
  -- Hack: To handle recursive regions, need to put a proxy into place
  -- before looking at either types or values.
  local symbols = terralib.newlist()
  for i, var_name in ipairs(node.var_names) do
    if node.values[i] and node.values[i]:is(ast.unspecialized.ExprRegion) then
      local symbol = terralib.newsymbol(var_name)
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
      symbol = terralib.newsymbol(var_type, var_name)
      cx.env:insert(node, var_name, symbol)
      symbols[i] = symbol
    end
  end

  return ast.specialized.StatVar {
    symbols = symbols,
    values = values,
    span = node.span,
  }
end

function specialize.stat_var_unpack(cx, node)
  local symbols = terralib.newlist()
  for _, var_name in ipairs(node.var_names) do
    local symbol = terralib.newsymbol(var_name)
    cx.env:insert(node, var_name, symbol)
    symbols:insert(symbol)
  end

  local value = specialize.expr(cx, node.value)

  return ast.specialized.StatVarUnpack {
    symbols = symbols,
    fields = node.fields,
    value = value,
    span = node.span,
  }
end

function specialize.stat_return(cx, node)
  return ast.specialized.StatReturn {
    value = node.value and specialize.expr(cx, node.value),
    span = node.span,
  }
end

function specialize.stat_break(cx, node)
  return ast.specialized.StatBreak {
    span = node.span,
  }
end

function specialize.stat_assignment_or_stat_reduce(cx, node)
  if not has_all_valid_field_accesses(node) then
    log.error(node, "invalid use of multi-field access")
  end

  local flattened_lhs = terralib.newlist()
  local flattened_rhs = terralib.newlist()

  std.zip(node.lhs, node.rhs):map(function(pair)
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

  if node:is(ast.unspecialized.StatAssignment) then
    return ast.specialized.StatAssignment {
      lhs = flattened_lhs,
      rhs = flattened_rhs,
      span = node.span,
    }

  else -- if node:is(ast.unspecialized.StatReduce)
    return ast.specialized.StatReduce {
      lhs = flattened_lhs,
      rhs = flattened_rhs,
      op = node.op,
      span = node.span,
    }
  end
end

function specialize.stat_expr(cx, node)
  return ast.specialized.StatExpr {
    expr = specialize.expr(cx, node.expr),
    span = node.span,
  }
end

function specialize.stat(cx, node)
  if node:is(ast.unspecialized.StatIf) then
    return specialize.stat_if(cx, node)

  elseif node:is(ast.unspecialized.StatWhile) then
    return specialize.stat_while(cx, node)

  elseif node:is(ast.unspecialized.StatForNum) then
    return specialize.stat_for_num(cx, node)

  elseif node:is(ast.unspecialized.StatForList) then
    return specialize.stat_for_list(cx, node)

  elseif node:is(ast.unspecialized.StatRepeat) then
    return specialize.stat_repeat(cx, node)

  elseif node:is(ast.unspecialized.StatBlock) then
    return specialize.stat_block(cx, node)

  elseif node:is(ast.unspecialized.StatVar) then
    return specialize.stat_var(cx, node)

  elseif node:is(ast.unspecialized.StatVarUnpack) then
    return specialize.stat_var_unpack(cx, node)

  elseif node:is(ast.unspecialized.StatReturn) then
    return specialize.stat_return(cx, node)

  elseif node:is(ast.unspecialized.StatBreak) then
    return specialize.stat_break(cx, node)

  elseif node:is(ast.unspecialized.StatAssignment) then
    return specialize.stat_assignment_or_stat_reduce(cx, node)

  elseif node:is(ast.unspecialized.StatReduce) then
    return specialize.stat_assignment_or_stat_reduce(cx, node)

  elseif node:is(ast.unspecialized.StatExpr) then
    return specialize.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.privilege_region_field(cx, node)
  local prefix = std.newtuple(node.field_name)
  local fields = specialize.privilege_region_fields(cx, node.fields)
  return fields:map(
    function(field) return prefix .. field end)
end

function specialize.privilege_region_fields(cx, node)
  if not node then
    return terralib.newlist({std.newtuple()})
  end
  local fields = node:map(
    function(field) return specialize.privilege_region_field(cx, field) end)
  local result = terralib.newlist()
  for _, f in ipairs(fields) do
    result:insertall(f)
  end
  return result
end

function specialize.privilege_region(cx, node)
  local region = cx.env:lookup(node, node.region_name)
  local fields = specialize.privilege_region_fields(cx, node.fields)

  return {
    region = region,
    fields = fields,
  }
end

function specialize.privilege(cx, node)
  local privilege
  if node.privilege == "reads" then
    privilege = std.reads
  elseif node.privilege == "writes" then
    privilege = std.writes
  elseif node.privilege == "reduces" then
    privilege = std.reduces(node.op)
  else
    assert(false)
  end

  local region_fields = node.regions:map(
    function(region) return specialize.privilege_region(cx, region) end)
  return std.privilege(privilege, region_fields)
end

function specialize.constraint(cx, node)
  local lhs = cx.env:lookup(node, node.lhs)
  local rhs = cx.env:lookup(node, node.rhs)

  return std.constraint(lhs, rhs, node.op)
end

function specialize.stat_task_param(cx, node)
  -- Hack: Params which are regions can be recursive on the name of
  -- the region so introduce the symbol before type checking to allow
  -- for this recursion.
  local symbol = terralib.newsymbol(node.param_name)
  cx.env:insert(node, node.param_name, symbol)
  local param_type = node.type_expr(cx.env:env())
  symbol.type = param_type

  return ast.specialized.StatTaskParam {
    symbol = symbol,
    span = node.span,
  }
end

function specialize.stat_task(cx, node)
  local cx = cx:new_local_scope()
  local proto = std.newtask(node.name)
  cx.env:insert(node, node.name, proto)
  cx = cx:new_local_scope()

  local params = node.params:map(
    function(param) return specialize.stat_task_param(cx, param) end)
  local return_type = node.return_type_expr(cx.env:env())
  local privileges = node.privileges:map(
    function(privilege) return specialize.privilege(cx, privilege) end)
  local constraints = node.constraints:map(
    function(constraint) return specialize.constraint(cx, constraint) end)
  local body = specialize.block(cx, node.body)

  return ast.specialized.StatTask {
    name = node.name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    constraints = constraints,
    body = body,
    prototype = proto,
    inline = node.inline,
    cuda = node.cuda,
    span = node.span,
  }
end

function specialize.stat_fspace_param(cx, node)
  -- Insert symbol into environment first to allow circular types.
  local symbol = terralib.newsymbol(node.param_name)
  cx.env:insert(node, node.param_name, symbol)

  local param_type = node.type_expr(cx.env:env())
  symbol.type = param_type

  return symbol
end

function specialize.stat_fspace_field(cx, node)
  -- Insert symbol into environment first to allow circular types.
  local symbol = terralib.newsymbol(node.field_name)
  cx.env:insert(node, node.field_name, symbol)

  local field_type = node.type_expr(cx.env:env())
  symbol.type = field_type

  return  {
    field = symbol,
    type = field_type,
  }
end

function specialize.stat_fspace(cx, node)
  local cx = cx:new_local_scope()
  local fs = std.newfspace(node, node.name, #node.params > 0)
  cx.env:insert(node, node.name, fs)

  fs.params = node.params:map(
      function(param) return specialize.stat_fspace_param(cx, param) end)
  fs.fields = node.fields:map(
      function(field) return specialize.stat_fspace_field(cx, field) end)
  fs.constraints = node.constraints:map(
      function(constraint) return specialize.constraint(cx, constraint) end)

  return ast.specialized.StatFspace {
    name = node.name,
    fspace = fs,
    span = node.span,
  }
end

function specialize.stat_top(cx, node)
  if node:is(ast.unspecialized.StatTask) then
    return specialize.stat_task(cx, node)

  elseif node:is(ast.unspecialized.StatFspace) then
    return specialize.stat_fspace(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function specialize.entry(env, node)
  local cx = context:new_global_scope(env)
  return specialize.stat_top(cx, node)
end

return specialize
