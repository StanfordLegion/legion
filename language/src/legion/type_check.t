-- Copyright 2014 Stanford University
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

local ast = terralib.require("legion/ast")
local log = terralib.require("legion/log")
local std = terralib.require("legion/std")
local symbol_table = terralib.require("legion/symbol_table")

local type_check = {}

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    type_env = self.type_env:new_local_scope(),
    privileges = self.privileges,
    constraints = self.constraints,
    expected_return_type = self.expected_return_type,
  }
  setmetatable(cx, context)
  return cx
end

function context:new_task_scope(expected_return_type)
  local cx = {
    type_env = self.type_env:new_local_scope(),
    privileges = {},
    constraints = {},
    expected_return_type = { [ 1 ] = expected_return_type },
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

function context:get_return_type()
  if self.expected_return_type then
    return self.expected_return_type[1]
  end
end

function context:set_return_type(t)
  assert(self.expected_return_type)
  self.expected_return_type[1] = t
end

function type_check.expr_id(cx, node)
  local expr_type = cx.type_env:lookup(node.value)

  return ast.typed.ExprID {
    value = node.value,
    expr_type = expr_type,
  }
end

function type_check.expr_constant(cx, node)
  return ast.typed.ExprConstant {
    value = node.value,
    expr_type = node.expr_type,
  }
end

local untyped = std.untyped
local untyped_fn = terralib.types.functype({}, terralib.types.unit, true)
local function cast_fn(to_type)
  return terralib.types.functype({untyped}, to_type, false)
end

function type_check.expr_function(cx, node)
  -- Functions are type checked at the call site.
  return ast.typed.ExprFunction {
    value = node.value,
    expr_type = untyped,
  }
end

function type_check.expr_field_access(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = value.expr_type -- Keep references, do NOT std.check_read

  -- If the value is an fspace instance, unpack before allowing access.
  local unpack_type, constraints = value_type
  if std.is_fspace_instance(value_type) or
    (std.is_ptr(value_type) and std.is_fspace_instance(value_type.points_to_type)) or
    (std.is_ref(value_type) and std.is_fspace_instance(value_type.refers_to_type))
  then
    local fspace = value_type
    if std.is_ptr(value_type) then
      fspace = value_type.points_to_type
    elseif std.is_ref(value_type) then
      fspace = value_type.refers_to_type
    end
    unpack_type, constraints = std.unpack_fields(fspace)
  end

  if constraints then
    std.add_constraints(cx, constraints)
  end

  local field_type = std.get_field(unpack_type, node.field_name)

  if not field_type then
    log.error("no field '" .. node.field_name .. "' in type " .. tostring(value_type))
  end

  return ast.typed.ExprFieldAccess {
    value = value,
    field_name = node.field_name,
    expr_type = field_type,
  }
end

function type_check.expr_index_access(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)
  local index = type_check.expr(cx, node.index)
  local index_type = std.check_read(cx, index.expr_type)

  if std.is_partition(value_type) then
    if index:is(ast.typed.ExprConstant) then
      local parent = value_type:parent_region()
      local subregion = value_type:subregion_constant(index.value)
      std.add_constraint(cx, subregion, parent, "<=", false)

      return ast.typed.ExprIndexAccess {
        value = value,
        index = index,
        expr_type = subregion,
      }
    else
      error("unimplemented index access over non-constant indices")
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
      log.error("invalid index access for " .. tostring(value_type) .. " and " .. tostring(index_type))
    end

    return ast.typed.ExprIndexAccess {
      value = value,
      index = index,
      expr_type = result_type,
    }
  end
end

function type_check.expr_method_call(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)
  local args = node.args:map(
    function(arg) return type_check.expr(cx, arg) end)
  local arg_types = args:map(
    function(arg) return std.check_read(cx, arg.expr_type) end)

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
    log.error("invalid method call for " .. tostring(value_type) .. ":" ..
                node.method_name .. "(" .. arg_types:mkstring(", ") .. ")")
  end

  return ast.typed.ExprMethodCall {
    value = value,
    method_name = node.method_name,
    args = args,
    expr_type = expr_type,
  }
end


function type_check.expr_call(cx, node)
  local fn = type_check.expr(cx, node.fn)
  local args = node.args:map(
    function(arg) return type_check.expr(cx, arg) end)
  local arg_types = args:map(
    function(arg) return std.check_read(cx, arg.expr_type) end)

  -- Determine the type of the function being called.
  local fn_type
  if fn.expr_type == untyped then
    if terralib.isfunction(fn.value) or
      terralib.isfunctiondefinition(fn.value) or
      terralib.ismacro(fn.value)
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
        fn_type = result_type
      else
        log.error("no applicable overloaded function for arguments " .. arg_types:mkstring(", "))
      end
    elseif std.is_task(fn.value) then
      fn_type = fn.value:gettype()
    elseif type(fn.value) == "function" then
      fn_type = untyped_fn
    else
      error("unreachable")
    end
  else
    fn_type = fn.expr_type
  end
  assert(terralib.types.istype(fn_type) and
           (fn_type:isfunction() or fn_type:ispointertofunction()))
  -- Store the determined type back into the AST node for the function.
  fn.expr_type = fn_type

  local param_symbols = std.fn_param_symbols(fn_type)
  local arg_symbols = terralib.newlist()
  for i, arg in ipairs(args) do
    local arg_type = arg_types[i]
    if arg:is(ast.typed.ExprID) then
      arg_symbols:insert(arg.value)
    else
      arg_symbols:insert(terralib.newsymbol(arg_type))
    end
  end
  local expr_type = std.validate_args(
    param_symbols, arg_symbols, fn_type.isvararg, fn_type.returntype, {}, true)

  local privileges = std.is_task(fn.value) and fn.value:getprivileges()
  if privileges then
    local mapping = {}
    for i, arg_symbol in ipairs(arg_symbols) do
      local param_type = fn_type.parameters[i]
      mapping[param_type] = arg_symbol
    end

    for _, privilege_list in ipairs(fn.value.privileges) do
      for _, privilege in ipairs(privilege_list) do
        local privilege_type = privilege.privilege
        local region = privilege.region
        local field_path = privilege.field_path
        assert(std.is_region(region.type))
        local arg_region = mapping[region.type]
        if not std.check_privilege(cx, privilege_type, arg_region.type, field_path) then
          for i, arg in ipairs(arg_symbols) do
            if std.type_eq(arg.type, arg_region.type) then
              log.error("invalid privileges in argument " .. tostring(i) ..
                          ": " .. tostring(privilege_type) .. "(" ..
                          (std.newtuple(arg_region) .. field_path):hash() .. ")")
            end
          end
          assert(false)
        end
      end
    end
  end

  return ast.typed.ExprCall {
    fn = fn,
    args = args,
    expr_type = expr_type,
  }
end

function type_check.expr_cast(cx, node)
  local fn = type_check.expr(cx, node.fn)

  local to_type = fn.value
  assert(terralib.types.istype(to_type))
  fn.expr_type = cast_fn(to_type)

  if #node.args ~= 1 then
    log.error("expected 1 arguments but got " .. tostring(#node.args))
  end
  local arg = type_check.expr(cx, node.args[1])
  local from_type = std.check_read(cx, arg.expr_type)

  if std.is_fspace_instance(to_type) then
    if not (from_type:isstruct() or std.is_fspace_instance(from_type)) then
      log.error("type mismatch: expected struct or fspace but got " .. tostring(from_type))
    end

    local to_params = to_type:getparams()
    local to_args = to_type.args
    local to_constraints = to_type:getconstraints()

    local to_fields = std.struct_entries_symbols(to_type)

    local from_symbols = {}
    if arg:is(ast.typed.ExprCtor) and arg.named then
      for _, field in ipairs(arg.fields) do
        if field.value:is(ast.typed.ExprID) and
          terralib.issymbol(field.value.value) and
          terralib.types.istype(field.value.value.type)
        then
          from_symbols[field.value.value.type] = field.value.value
        end
      end
    end
    local from_fields = std.struct_entries_symbols(from_type, from_symbols)

    local mapping = {}
    for i, param in ipairs(to_params) do
      local arg = to_args[i]
      mapping[param] = arg
    end

    std.validate_args(to_fields, from_fields, false, terralib.types.unit, mapping, false)
    local satisfied, constraint = std.check_constraints(cx, to_constraints, mapping)
    if not satisfied then
      log.error("invalid cast missing constraint " .. tostring(constraint.lhs) ..
                  " " .. tostring(constraint.op) .. " " .. tostring(constraint.rhs))
    end
  else
    if not std.validate_explicit_cast(from_type, to_type) then
      log.error("invalid cast from " .. tostring(from_type) .. " to " .. tostring(to_type))
    end
  end

  return ast.typed.ExprCast {
    fn = fn,
    arg = arg,
    expr_type = to_type,
  }
end

function type_check.expr_ctor_list_field(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)
  return ast.typed.ExprCtorListField {
    value = value,
  }
end

function type_check.expr_ctor_rec_field(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)
  return ast.typed.ExprCtorRecField {
    name = node.name,
    value = value,
  }
end

function type_check.expr_ctor_field(cx, node)
  if node:is(ast.specialized.ExprCtorListField) then
    return type_check.expr_ctor_list_field(cx, node)
  elseif node:is(ast.specialized.ExprCtorRecField) then
    return type_check.expr_ctor_rec_field(cx, node)
  else
  end
end

function type_check.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return type_check.expr_ctor_field(cx, field) end)

  local expr_type
  if node.named then
    expr_type = std.ctor(
      fields:map(
        function(field) return { field.name, field.value.expr_type } end))
  else
    expr_type = terralib.types.tuple(unpack(fields:map(
      function(field) return field.value.expr_type end)))
  end

  return ast.typed.ExprCtor {
    fields = fields,
    named = node.named,
    expr_type = expr_type,
  }
end

function type_check.expr_isnull(cx, node)
  local pointer = type_check.expr(cx, node.pointer)
  local pointer_type = std.check_read(cx, pointer.expr_type)
  return ast.typed.ExprIsnull {
    pointer = pointer,
    expr_type = bool,
  }
end

function type_check.expr_new(cx, node)
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region.expr_type)
  return ast.typed.ExprNew {
    pointer_type = node.pointer_type,
    region = region,
    expr_type = node.pointer_type,
  }
end

function type_check.expr_null(cx, node)
  return ast.typed.ExprNull {
    pointer_type = node.pointer_type,
    expr_type = node.pointer_type,
  }
end

function type_check.expr_region(cx, node)
  local size = type_check.expr(cx, node.size)
  local size_type = std.check_read(cx, size.expr_type)

  local region = node.expr_type
  std.add_privilege(cx, "reads", region, std.newtuple())
  std.add_privilege(cx, "writes", region, std.newtuple())

  return ast.typed.ExprRegion {
    element_type = node.element_type,
    size = size,
    expr_type = region,
  }
end

function type_check.expr_partition(cx, node)
  local disjointness = node.disjointness
  local region = type_check.expr(cx, node.region)
  local region_type = std.check_read(cx, region.expr_type)

  local coloring = type_check.expr(cx, node.coloring)
  local coloring_type = std.check_read(cx, coloring.expr_type)

  if not (disjointness == std.disjoint or disjointness == std.aliased) then
    log.error("type mismatch in argument 1: expected disjoint or aliased but got " ..
                tostring(disjointness))
  end

  if not std.is_region(region_type) then
    log.error("type mismatch in argument 2: expected region but got " ..
                tostring(region_type))
  end

  if coloring_type ~= std.c.legion_coloring_t then
    log.error(
      "type mismatch in argument 3: expected legion_coloring_t but got " ..
        tostring(coloring_type))
  end

  return ast.typed.ExprPartition {
    disjointness = disjointness,
    region = region,
    coloring = coloring,
    expr_type = node.expr_type,
  }
end

local function unary_op_type(op)
  return function(cx, rhs_type)
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local function test()
      local terra query(rhs : rhs_type)
        return [ std.quote_unary_op(op, rhs) ]
      end
      return query:gettype().returntype
    end
    local valid, result_type = pcall(test)

    if not valid then
      log.error("invalid argument to unary operator " .. tostring(rhs_type))
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
  local rhs_type = std.check_read(cx, rhs.expr_type)

  local expr_type = unary_ops[node.op](cx, rhs_type)

  return ast.typed.ExprUnary {
    op = node.op,
    rhs = rhs,
    expr_type = expr_type,
  }
end

local function binary_op_type(op)
  return function(cx, lhs_type, rhs_type)
    -- Ask the Terra compiler to kindly tell us what type this operator returns.
    local function test()
      local terra query(lhs : lhs_type, rhs : rhs_type)
        return [ std.quote_binary_op(op, lhs, rhs) ]
      end
      return query:gettype().returntype
    end
    local valid, result_type = pcall(test)

    if not valid then
      log.error("type mismatch between " .. tostring(lhs_type) ..
                  " and " .. tostring(rhs_type))
    end

    return result_type
  end
end

local function binary_equality(op)
  local check = binary_op_type(op)
  return function(cx, lhs_type, rhs_type)
    if std.is_ptr(lhs_type) and std.is_ptr(rhs_type) then
      if not std.type_eq(lhs_type, rhs_type) then
        log.error("type mismatch between " .. tostring(lhs_type) ..
                    " and " .. tostring(rhs_type))
      end
      return bool
    else
      return check(cx, lhs_type, rhs_type)
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
}

function type_check.expr_binary(cx, node)
  local lhs = type_check.expr(cx, node.lhs)
  local lhs_type = std.check_read(cx, lhs.expr_type)

  local rhs = type_check.expr(cx, node.rhs)
  local rhs_type = std.check_read(cx, rhs.expr_type)

  local expr_type = binary_ops[node.op](cx, lhs_type, rhs_type)

  return ast.typed.ExprBinary {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
    expr_type = expr_type,
  }
end

function type_check.expr_deref(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)

  if not std.is_ptr(value_type) then
    log.error("dereference of non-pointer type " .. tostring(value_type))
  end

  local expr_type = std.ref(value_type.points_to_type, value_type.points_to_region_symbol)

  return ast.typed.ExprDeref {
    value = value,
    expr_type = expr_type,
  }
end

function type_check.expr(cx, node)
  if node:is(ast.specialized.ExprID) then
    return type_check.expr_id(cx, node)

  elseif node:is(ast.specialized.ExprConstant) then
    return type_check.expr_constant(cx, node)

  elseif node:is(ast.specialized.ExprFunction) then
    return type_check.expr_function(cx, node)

  elseif node:is(ast.specialized.ExprFieldAccess) then
    return type_check.expr_field_access(cx, node)

  elseif node:is(ast.specialized.ExprIndexAccess) then
    return type_check.expr_index_access(cx, node)

  elseif node:is(ast.specialized.ExprMethodCall) then
    return type_check.expr_method_call(cx, node)

  elseif node:is(ast.specialized.ExprCall) then
    return type_check.expr_call(cx, node)

  elseif node:is(ast.specialized.ExprCast) then
    return type_check.expr_cast(cx, node)

  elseif node:is(ast.specialized.ExprCtor) then
    return type_check.expr_ctor(cx, node)

  elseif node:is(ast.specialized.ExprIsnull) then
    return type_check.expr_isnull(cx, node)

  elseif node:is(ast.specialized.ExprNew) then
    return type_check.expr_new(cx, node)

  elseif node:is(ast.specialized.ExprNull) then
    return type_check.expr_null(cx, node)

  elseif node:is(ast.specialized.ExprRegion) then
    return type_check.expr_region(cx, node)

  elseif node:is(ast.specialized.ExprPartition) then
    return type_check.expr_partition(cx, node)

  elseif node:is(ast.specialized.ExprUnary) then
    return type_check.expr_unary(cx, node)

  elseif node:is(ast.specialized.ExprBinary) then
    return type_check.expr_binary(cx, node)

  elseif node:is(ast.specialized.ExprDeref) then
    return type_check.expr_deref(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function type_check.block(cx, node)
  return ast.typed.Block {
    stats = node.stats:map(
      function(stat) return type_check.stat(cx, stat) end),
  }
end

function type_check.stat_if(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond.expr_type)

  local then_cx = cx:new_local_scope()
  local else_cx = cx:new_local_scope()
  return ast.typed.StatIf {
    cond = cond,
    then_block = type_check.block(then_cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return type_check.stat_elseif(cx, block) end),
    else_block = type_check.block(else_cx, node.else_block),
  }
end

function type_check.stat_elseif(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond.expr_type)

  local body_cx = cx:new_local_scope()
  return ast.typed.StatElseif {
    cond = cond,
    block = type_check.block(body_cx, node.block),
  }
end

function type_check.stat_while(cx, node)
  local cond = type_check.expr(cx, node.cond)
  local cond_type = std.check_read(cx, cond.expr_type)

  local body_cx = cx:new_local_scope()
  return ast.typed.StatWhile {
    cond = cond,
    block = type_check.block(body_cx, node.block),
  }
end

function type_check.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return type_check.expr(cx, value) end)
  local value_types = values:map(
    function(value) return std.check_read(cx, value.expr_type) end)

  for _, value_type in ipairs(value_types) do
    if not value_type:isintegral() then
      log.error("numeric for loop expected integral type, got " .. tostring(value_type))
    end
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type = node.symbol.type or value_types[1]
  if value_types[3] then
    var_type = binary_op_type("+")(cx, var_type, value_types[3])
  end
  if not var_type:isintegral() then
    log.error("numeric for loop expected integral type, got " .. tostring(var_type))
  end
  node.symbol.type = var_type
  cx.type_env:insert(node.symbol, var_type)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  return ast.typed.StatForNum {
    symbol = node.symbol,
    values = values,
    block = type_check.block(cx, node.block),
  }
end

function type_check.stat_for_list(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)

  if not std.is_region(value_type) then
    log.error("iterator for loop expected region, got " .. tostring(value_type))
  end

  -- Enter scope for header.
  local cx = cx:new_local_scope()
  local var_type = node.symbol.type
  if not var_type then
    local region = value_type
    var_type = std.ptr(region.element_type, terralib.newsymbol(region))
  end
  if not std.is_ptr(var_type) then
    log.error("iterator for loop expected pointer type, got " .. tostring(var_type))
  end

  -- Hack: Stuff the type back into the symbol so it's available
  -- to ptr types if necessary.
  node.symbol.type = var_type
  cx.type_env:insert(node.symbol, var_type)

  -- Enter scope for body.
  local cx = cx:new_local_scope()
  return ast.typed.StatForList {
    symbol = node.symbol,
    value = value,
    block = type_check.block(cx, node.block),
  }
end

function type_check.stat_repeat(cx, node)
  local until_cond = type_check.expr(cx, node.until_cond)
  local until_cond_type = std.check_read(cx, until_cond.expr_type)

  local cx = cx:new_local_scope()
  return ast.typed.StatRepeat {
    block = type_check.block(cx, node.block),
    until_cond = until_cond,
  }
end

function type_check.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return ast.typed.StatBlock {
    block = type_check.block(cx, node.block)
  }
end

function type_check.stat_var(cx, node)
  for i, symbol in ipairs(node.symbols) do
    local var_type = symbol.type
    local value = node.values[i]
    if value and value:is(ast.specialized.ExprRegion) then
      cx.type_env:insert(symbol, std.as_read(value.expr_type))
    end
  end

  local values = node.values:map(
    function(value) return type_check.expr(cx, value) end)
  local value_types = values:map(
    function(value) return std.check_read(cx, value.expr_type) end)

  local types = terralib.newlist()
  for i, symbol in ipairs(node.symbols) do
    local var_type = symbol.type

    local value = values[i]
    local value_type = value_types[i]
    if var_type then
      if value and not std.validate_implicit_cast(value_type, var_type, {}) then
        log.error("type mismatch in var: expected " .. tostring(var_type) .. " but got " .. tostring(value_type))
      end
    else
      if not value then
        log.error("type must be specified for uninitialized variables")
      end
      var_type = value_type

      -- Hack: Stuff the type back into the symbol so it's available
      -- to ptr types if necessary.
      symbol.type = var_type
    end
    if not (node.values[i] and node.values[i]:is(ast.specialized.ExprRegion)) then
      cx.type_env:insert(symbol, var_type)
    end
    types:insert(var_type)
  end

  return ast.typed.StatVar {
    symbols = node.symbols,
    types = types,
    values = values,
  }
end

function type_check.stat_var_unpack(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)

  if not (value_type:isstruct() or std.is_fspace_instance(value_type)) then
    log.error("destructuring var expected struct or fspace, got " .. tostring(value_type))
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
      field = mapping[field].displayname
    end
    local field_type = index[field]
    if not field_type then
      log.error("no field '" .. tostring(field) .. "' in type " .. tostring(value_type))
    end
    symbol.type = field_type
    cx.type_env:insert(symbol, field_type)
    field_types:insert(field_type)
  end

  if constraints then
    std.add_constraints(cx, constraints)
  end

  return ast.typed.StatVarUnpack {
    symbols = node.symbols,
    fields = node.fields,
    field_types = field_types,
    value = value,
  }
end

function type_check.stat_return(cx, node)
  local value = type_check.expr(cx, node.value)
  local value_type = std.check_read(cx, value.expr_type)

  local expected_type = cx:get_return_type()
  assert(expected_type)
  if std.type_eq(expected_type, std.untyped) then
    cx:set_return_type(value_type)
  else
    local result_type = std.type_meet(value_type, expected_type)
    if not result_type then
      log.error("type mismatch in return: expected " .. tostring(expected_type) .. " but got " .. tostring(value_type))
    end
    cx:set_return_type(result_type)
  end

  return ast.typed.StatReturn {
    value = value,
  }
end

function type_check.stat_break(cx, node)
  return ast.typed.StatBreak {}
end

function type_check.stat_assignment(cx, node)
  local lhs = node.lhs:map(
    function(value) return type_check.expr(cx, value) end)
  local lhs_types = lhs:map(
    function(lh) return std.check_write(cx, lh.expr_type) end)

  local rhs = node.rhs:map(
    function(value) return type_check.expr(cx, value) end)
  local rhs_types = rhs:map(
    function(rh) return std.check_read(cx, rh.expr_type) end)

  for i, lhs_type in ipairs(lhs_types) do
    local rhs_type = rhs_types[i]

    if not std.validate_implicit_cast(rhs_type, lhs_type, {}) then
      log.error("type mismatch in assignment: expected " .. tostring(lhs_type) .. " but got " .. tostring(rhs_type))
    end
  end

  return ast.typed.StatAssignment {
    lhs = lhs,
    rhs = rhs,
  }
end

function type_check.stat_reduce(cx, node)
  local lhs = node.lhs:map(
    function(value) return type_check.expr(cx, value) end)
  local lhs_types = lhs:map(
    function(lh) return std.check_reduce(cx, node.op, lh.expr_type) end)

  local rhs = node.rhs:map(
    function(value) return type_check.expr(cx, value) end)
  local rhs_types = rhs:map(
    function(rh) return std.check_read(cx, rh.expr_type) end)

  return ast.typed.StatReduce {
    op = node.op,
    lhs = lhs,
    rhs = rhs,
  }
end

function type_check.stat_expr(cx, node)
  local value = type_check.expr(cx, node.expr)
  local value_type = std.check_read(cx, value.expr_type)

  return ast.typed.StatExpr {
    expr = value,
  }
end

function type_check.stat(cx, node)
  if node:is(ast.specialized.StatIf) then
    return type_check.stat_if(cx, node)

  elseif node:is(ast.specialized.StatWhile) then
    return type_check.stat_while(cx, node)

  elseif node:is(ast.specialized.StatForNum) then
    return type_check.stat_for_num(cx, node)

  elseif node:is(ast.specialized.StatForList) then
    return type_check.stat_for_list(cx, node)

  elseif node:is(ast.specialized.StatRepeat) then
    return type_check.stat_repeat(cx, node)

  elseif node:is(ast.specialized.StatBlock) then
    return type_check.stat_block(cx, node)

  elseif node:is(ast.specialized.StatVar) then
    return type_check.stat_var(cx, node)

  elseif node:is(ast.specialized.StatVarUnpack) then
    return type_check.stat_var_unpack(cx, node)

  elseif node:is(ast.specialized.StatReturn) then
    return type_check.stat_return(cx, node)

  elseif node:is(ast.specialized.StatBreak) then
    return type_check.stat_break(cx, node)

  elseif node:is(ast.specialized.StatAssignment) then
    return type_check.stat_assignment(cx, node)

  elseif node:is(ast.specialized.StatReduce) then
    return type_check.stat_reduce(cx, node)

  elseif node:is(ast.specialized.StatExpr) then
    return type_check.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.stat_task_param(cx, node)
  local param_type = node.symbol.type
  cx.type_env:insert(node.symbol, param_type)

  return ast.typed.StatTaskParam {
    symbol = node.symbol,
    param_type = param_type,
  }
end

function type_check.stat_task(cx, node)
  local return_type = node.return_type
  local cx = cx:new_task_scope(return_type)

  local params = node.params:map(
    function(param) return type_check.stat_task_param(cx, param) end)
  local privileges = node.privileges
  local prototype = node.prototype

  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:settype(task_type)

  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      local privilege_type = privilege.privilege
      local region = privilege.region
      local field_path = privilege.field_path
      assert(std.is_region(region.type))
      std.add_privilege(cx, privilege_type, region.type, field_path)
    end
  end
  prototype:setprivileges(privileges)

  local body = type_check.block(cx, node.body)

  return_type = cx:get_return_type()
  if std.type_eq(return_type, std.untyped) then
    return_type = terralib.types.unit
  end
  task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:settype(task_type)

  prototype:set_constraints(cx.constraints)

  return ast.typed.StatTask {
    name = node.name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    body = body,
    prototype = prototype,
  }
end

function type_check.stat_fspace(cx, node)
  return ast.typed.StatFspace {
    name = node.name,
    fspace = node.fspace,
  }
end

function type_check.stat_top(cx, node)
  if node:is(ast.specialized.StatTask) then
    return type_check.stat_task(cx, node)

  elseif node:is(ast.specialized.StatFspace) then
    return type_check.stat_fspace(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function type_check.entry(node)
  local cx = context.new_global_scope({})
  return type_check.stat_top(cx, node)
end

return type_check
