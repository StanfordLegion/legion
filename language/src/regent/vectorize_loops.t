-- Copyright 2016 Stanford University
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

-- Legion Loop Vectorizer
--
-- Attempts to vectorize the body of loops
--

local ast = require("regent/ast")
local data = require("regent/data")
local log = require("regent/log")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local min = math.min

local bounds_checks = std.config["bounds-checks"]

-- vectorizer

local SIMD_REG_SIZE
if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  if os.execute("sysctl -a | grep machdep.cpu.features | grep AVX > /dev/null") == 0 then
    SIMD_REG_SIZE = 32
  elseif os.execute("sysctl -a | grep machdep.cpu.features | grep SSE > /dev/null") == 0 then
    SIMD_REG_SIZE = 16
  else
    error("Unable to determine CPU architecture")
  end
else
  if os.execute("grep avx /proc/cpuinfo > /dev/null") == 0 then
    SIMD_REG_SIZE = 32
  elseif os.execute("grep sse /proc/cpuinfo > /dev/null") == 0 then
    SIMD_REG_SIZE = 16
  else
    error("Unable to determine CPU architecture")
  end
end

local V = {}
V.__index = V
function V:__tostring() return "vector" end
setmetatable(V, V)
local S = {}
S.__index = S
function S:__tostring() return "scalar" end
setmetatable(S, S)

local function join(fact1, fact2)
  if fact1 == fact2 then return fact1 else return V end
end

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    var_type = self.var_type:new_local_scope(),
    subst = self.subst:new_local_scope(),
    expr_type = self.expr_type,
    demanded = self.demanded,
  }
  return setmetatable(cx, context)
end

function context:new_global_scope()
  local cx = {
    var_type = symbol_table:new_global_scope(),
    subst = symbol_table:new_global_scope(),
    expr_type = {},
    demanded = false,
  }
  return setmetatable(cx, context)
end

function context:assign(symbol, fact)
  self.var_type:insert(nil, symbol, fact)
end

function context:join(symbol, fact)
  local var_type = self.var_type
  local old_fact = var_type:safe_lookup(symbol)
  assert(old_fact)
  local new_fact = join(old_fact, fact)
  var_type:insert(nil, symbol, new_fact)
end

function context:lookup_expr_type(node)
  return self.expr_type[node]
end

function context:assign_expr_type(node, fact)
  self.expr_type[node] = fact
end

function context:join_expr_type(node, fact)
  self.expr_type[node] = join(fact, self.expr_type[node])
end

function context:report_error_when_demanded(node, error_msg)
  if self.demanded then log.error(node, error_msg) end
end

function context:add_substitution(from, to)
  self.subst:insert(nil, from, to)
end

function context:find_replacement(from)
  local to = self.subst:safe_lookup(from)
  assert(to)
  return to
end

local flip_types = {}

function flip_types.block(cx, simd_width, symbol, node)
  local stats = terralib.newlist()
  local function flip_type_each(stat)
    stats:insert(flip_types.stat(cx, simd_width, symbol, stat))
  end
  node.stats:map(flip_type_each)
  return node { stats = stats }
end

function flip_types.stat(cx, simd_width, symbol, node)
  if node:is(ast.typed.stat.Block) then
    return node { block = flip_types.block(cx, simd_width, symbol, node.block) }

  elseif node:is(ast.typed.stat.Var) then
    local symbols = terralib.newlist()
    local types = terralib.newlist()
    local values = terralib.newlist()

    for i = 1, #node.symbols do
      types:insert(flip_types.type(simd_width, node.types[i]))
      symbols:insert(std.newsymbol(types[i], node.symbols[i]:getname() .. "_vectorized"))
      cx:add_substitution(node.symbols[i], symbols[i])
      if i <= #node.values then
        values:insert(flip_types.expr(cx, simd_width, symbol, node.values[i]))
      end
    end

    return node { symbols = symbols, types = types, values = values }

  elseif node:is(ast.typed.stat.Assignment) or
         node:is(ast.typed.stat.Reduce) then
    local lhs = terralib.newlist()
    local rhs = terralib.newlist()

    node.lhs:map(function(exp)
      lhs:insert(flip_types.expr(cx, simd_width, symbol, exp))
    end)
    node.rhs:map(function(exp)
      rhs:insert(flip_types.expr(cx, simd_width, symbol, exp))
    end)

    if node:is(ast.typed.stat.Assignment) then
      return node { lhs = lhs, rhs = rhs }
    else -- node:is(ast.typed.stat.Reduce)
      return node { lhs = lhs, rhs = rhs }
    end

  elseif node:is(ast.typed.stat.ForNum) then
    return node { block = flip_types.block(cx, simd_width, symbol, node.block) }

  elseif node:is(ast.typed.stat.If) then
    return node {
      then_block = flip_types.block(cx, simd_width, symbol, node.then_block),
      elseif_blocks = node.elseif_blocks:map(function(elseif_block)
        return flip_types.stat(cx, simd_width, symbol, elseif_block)
      end),
      else_block = flip_types.block(cx, simd_width, symbol, node.else_block),
    }

  elseif node:is(ast.typed.stat.Elseif) then
    return node { block = flip_types.block(cx, simd_width, symbol, node.block) }

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end

  return res
end

function flip_types.expr(cx, simd_width, symbol, node)
  local new_node = node:fields()
  if node:is(ast.typed.expr.FieldAccess) then
    new_node.value = flip_types.expr(cx, simd_width, symbol, node.value)

  elseif node:is(ast.typed.expr.IndexAccess) then
    new_node.value = flip_types.expr(cx, simd_width, symbol, node.value)

  elseif node:is(ast.typed.expr.Binary) then
    new_node.lhs = flip_types.expr(cx, simd_width, symbol, new_node.lhs)
    new_node.rhs = flip_types.expr(cx, simd_width, symbol, new_node.rhs)

    local expr_type
    if std.as_read(new_node.lhs.expr_type):isvector() then
      expr_type = new_node.lhs.expr_type
    else
      expr_type = new_node.rhs.expr_type
    end

    if (node.op == "min" or node.op == "max") and
      std.is_minmax_supported(std.as_read(expr_type)) then
      local args = terralib.newlist()
      args:insert(new_node.lhs)
      args:insert(new_node.rhs)
      local rval_type = std.as_read(expr_type)
      local fn = std["v" .. node.op](rval_type)
      local fn_type = ({rval_type, rval_type} -> rval_type).type
      local fn_node = ast.typed.expr.Function {
        expr_type = fn_type,
        value = fn,
        options = node.options,
        span = node.span,
      }
      return ast.typed.expr.Call {
        fn = fn_node,
        args = args,
        conditions = terralib.newlist(),
        expr_type = rval_type,
        options = node.options,
        span = node.span,
      }
    end

  elseif node:is(ast.typed.expr.Unary) then
    new_node.rhs = flip_types.expr(cx, simd_width, symbol, new_node.rhs)

  elseif node:is(ast.typed.expr.Ctor) then
    new_node.fields = node.fields:map(
      function(field)
        return flip_types.expr(cx, simd_width, symbol, field)
      end)

  elseif node:is(ast.typed.expr.CtorRecField) then
    new_node.value = flip_types.expr(cx, simd_width, symbol, node.value)

  elseif node:is(ast.typed.expr.Call) then
    new_node.args = node.args:map(
      function(arg)
        return flip_types.expr(cx, simd_width, symbol, arg)
      end)
    new_node.fn = flip_types.expr_function(simd_width, node.fn)

  elseif node:is(ast.typed.expr.Cast) then
    if cx:lookup_expr_type(node) == V then
      new_node.arg = flip_types.expr(cx, simd_width, symbol, node.arg)
      new_node.fn = ast.typed.expr.Function {
        expr_type = flip_types.type(simd_width, node.fn.expr_type),
        value = flip_types.type(simd_width, node.fn.value),
        options = node.options,
        span = node.span,
      }
    end

  elseif node:is(ast.typed.expr.UnsafeCast) then
    if cx:lookup_expr_type(node) == V then
      new_node.value = flip_types.expr(cx, simd_width, symbol, node.value)
    end

  elseif node:is(ast.typed.expr.Deref) then

  elseif node:is(ast.typed.expr.ID) then
    if cx:lookup_expr_type(node) == V and not (node.value == symbol) then
      local sym = cx:find_replacement(node.value)
      new_node.value = sym
    end

  elseif node:is(ast.typed.expr.Constant) then

  else
    assert(false, "unexpected node type " .. tostring(node:type()))

  end
  if cx:lookup_expr_type(node) == V and
    not (node:is(ast.typed.expr.ID) and node.value == symbol)
  then
    new_node.expr_type = flip_types.type(simd_width, new_node.expr_type)
  end
  return node:type()(new_node)
end

function flip_types.expr_function(simd_width, node)
  local elmt_type = node.expr_type.returntype
  local vector_type = vector(elmt_type, simd_width)

  local value = std.convert_math_op(node.value, vector_type)
  local expr_type = flip_types.type(simd_width, node.expr_type)

  return node {
    expr_type = expr_type,
    value = value,
  }
end

function flip_types.type(simd_width, ty)
  if std.is_ref(ty) then
    local vector_type = flip_types.type(simd_width, std.as_read(ty))
    return std.ref(ty.pointer_type.index_type(vector_type, unpack(ty.bounds_symbols)))
  elseif std.is_rawref(ty) then
    local vector_type = flip_types.type(simd_width, std.as_read(ty))
    return std.rawref(&vector_type)
  elseif ty:isprimitive() then
    return vector(ty, simd_width)
  elseif ty:isarray() then
    return (vector(ty.type, simd_width))[ty.N]
  elseif std.is_bounded_type(ty) then
    return std.vptr(simd_width,
                    ty.points_to_type,
                    unpack(ty.bounds_symbols))
  elseif ty:isstruct() then
    return std.sov(ty, simd_width)
  elseif ty:isfunction() then
    local params = ty.parameters:map(
      function(ty)
        flip_types.type(simd_width, ty)
      end)
    local returntype = flip_types.type(simd_width, ty.returntype)
    return (params -> returntype).type
  else
    assert(false, "unexpected type " .. tostring(ty))
  end
end

local min_simd_width = {}

function min_simd_width.block(cx, reg_size, node)
  local simd_width = reg_size
  node.stats:map(function(stat)
    local simd_width_ = min_simd_width.stat(cx, reg_size, stat)
    simd_width = min(simd_width, simd_width_)
  end)
  return simd_width
end

function min_simd_width.stat(cx, reg_size, node)
  if node:is(ast.typed.stat.Block) then
    return min_simd_width.block(cx, reg_size, node.block)

  elseif node:is(ast.typed.stat.Var) then
    local simd_width = reg_size
    node.types:map(function(type)
      simd_width = min(simd_width, min_simd_width.type(reg_size, type))
    end)
    node.values:map(function(value)
      simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, value))
    end)
    return simd_width

  elseif node:is(ast.typed.stat.Assignment) or
         node:is(ast.typed.stat.Reduce) then
    local simd_width = reg_size
    node.lhs:map(function(lh)
      simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, lh))
    end)
    node.rhs:map(function(rh)
      simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, rh))
    end)
    return simd_width

  elseif node:is(ast.typed.stat.ForNum) then
    return min_simd_width.block(cx, reg_size, node.block)

  elseif node:is(ast.typed.stat.If) then
    local simd_width = reg_size
    min(simd_width, min_simd_width.block(cx, reg_size, node.then_block))
    node.elseif_blocks:map(function(elseif_block)
      min(simd_width, min_simd_width.stat(cx, reg_size, elseif_block))
    end)
    min(simd_width, min_simd_width.block(cx, reg_size, node.else_block))
    return simd_width

  elseif node:is(ast.typed.stat.Elseif) then
    return min_simd_width.block(cx, reg_size, node.block)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function min_simd_width.expr(cx, reg_size, node)
  local simd_width = reg_size
  if cx:lookup_expr_type(node) == V then
    simd_width = min_simd_width.type(reg_size, std.as_read(node.expr_type))
  end

  if node:is(ast.typed.expr.ID) then

  elseif node:is(ast.typed.expr.FieldAccess) then
    if not node.value:is(ast.typed.expr.ID) then
      simd_width = min(simd_width,
                       min_simd_width.expr(cx, reg_size, node.value))
    end

  elseif node:is(ast.typed.expr.IndexAccess) then
    simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, node.value))

  elseif node:is(ast.typed.expr.Unary) then
    simd_width = min_simd_width.expr(cx, reg_size, node.rhs)

  elseif node:is(ast.typed.expr.Binary) then
    simd_width = min(min_simd_width.expr(cx, reg_size, node.lhs),
                     min_simd_width.expr(cx, reg_size, node.rhs))

  elseif node:is(ast.typed.expr.Ctor) then
    for _, field in pairs(node.fields) do
      simd_width = min(simd_width, min_simd_width.expr(cx, reg_size,field))
    end

  elseif node:is(ast.typed.expr.CtorRecField) then
    simd_width = min_simd_width.expr(cx, reg_size, node.value)

  elseif node:is(ast.typed.expr.Constant) then

  elseif node:is(ast.typed.expr.Call) then
    for _, arg in pairs(node.args) do
      simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, arg))
    end

  elseif node:is(ast.typed.expr.Cast) then
    simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, node.arg))

  elseif node:is(ast.typed.expr.UnsafeCast) then
    simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, node.value))

  elseif node:is(ast.typed.expr.Deref) then
    simd_width = min(simd_width, min_simd_width.expr(cx, reg_size, node.value))

  else
    assert(false, "unexpected node type " .. tostring(node:type()))

  end

  return simd_width
end

function min_simd_width.type(reg_size, ty)
  assert(not (std.is_ref(ty) or std.is_rawref(ty)))
  if std.is_bounded_type(ty) then
    return reg_size / sizeof(uint32)
  elseif ty:isarray() then
    return reg_size / sizeof(ty.type)
  elseif ty:isstruct() then
    local simd_width = reg_size
    for _, entry in pairs(ty.entries) do
      local entry_type = entry[2] or entry.type
      simd_width =
        min(simd_width,
            min_simd_width.type(reg_size, std.as_read(entry_type)))
    end
    return simd_width
  else
    return reg_size / sizeof(ty)
  end
end

local vectorize = {}
function vectorize.stat_for_list(cx, node)
  local simd_width = min_simd_width.block(cx, SIMD_REG_SIZE, node.block)
  assert(simd_width >= 1)
  local body = flip_types.block(cx, simd_width, node.symbol, node.block)
  return ast.typed.stat.ForListVectorized {
    symbol = node.symbol,
    value = node.value,
    block = body,
    orig_block = node.block,
    vector_width = simd_width,
    options = node.options,
    span = node.span,
  }
end

function collect_bounds(node)
  local bounds = terralib.newlist()
  if node:is(ast.typed.expr.FieldAccess) then
    local value_type = std.as_read(node.value.expr_type)
    if std.is_bounded_type(value_type) then
      value_type:bounds():map(function(bound)
        bounds:insert(data.newtuple(bound, node.field_name))
      end)
    end
    bounds:insertall(collect_bounds(node.value))

  elseif node:is(ast.typed.expr.IndexAccess) then
    bounds:insertall(collect_bounds(node.value))
    bounds:insertall(collect_bounds(node.index))

  elseif node:is(ast.typed.expr.Unary) then
    bounds:insertall(collect_bounds(node.rhs))

  elseif node:is(ast.typed.expr.Binary) then
    bounds:insertall(collect_bounds(node.lhs))
    bounds:insertall(collect_bounds(node.rhs))

  elseif node:is(ast.typed.expr.Ctor) then
    for _, field in pairs(node.fields) do
      bounds:insertall(collect_bounds(field))
    end

  elseif node:is(ast.typed.expr.CtorRecField) then
    bounds:insertall(collect_bounds(node.value))

  elseif node:is(ast.typed.expr.Call) then
    for _, arg in pairs(node.args) do
      bounds:insertall(collect_bounds(arg))
    end

  elseif node:is(ast.typed.expr.Cast) then
    bounds:insertall(collect_bounds(node.arg))
  end

  return bounds
end

-- vectorizability check returns truen when the statement is vectorizable
local check_vectorizability = {}
local error_prefix = "vectorization failed: loop body has "

function check_vectorizability.block(cx, node)
  cx = cx:new_local_scope()
  for i, stat in ipairs(node.stats) do
    local vectorizable = check_vectorizability.stat(cx, stat)
    if not vectorizable then return false end
  end
  return true
end

function check_vectorizability.stat(cx, node)
  if node:is(ast.typed.stat.Block) then
    return check_vectorizability.block(cx, node.block)

  elseif node:is(ast.typed.stat.Var) then
    for i, symbol in pairs(node.symbols) do
      if #node.values > 0 then
        local value = node.values[i]
        if not check_vectorizability.expr(cx, value) then return false end
      end

      local ty = node.types[i]
      local type_vectorizable =
        (ty:isarray() and ty.type:isprimitive()) or
        check_vectorizability.type(ty)
      if not type_vectorizable then
        cx:report_error_when_demanded(node,
          error_prefix .. "a variable declaration of an inadmissible type")
        return type_vectorizable
      end
      cx:assign(symbol, V)
    end
    return true

  elseif node:is(ast.typed.stat.Assignment) or
         node:is(ast.typed.stat.Reduce) then
    local bounds_lhs = {}
    local bounds_rhs = {}
    for i, rh in pairs(node.rhs) do
      local lh = node.lhs[i]

      if not check_vectorizability.expr(cx, lh) or
         not check_vectorizability.expr(cx, rh) then return false end

      if cx:lookup_expr_type(lh) == S and cx:lookup_expr_type(rh) == V then
        cx:report_error_when_demanded(node, error_prefix ..
          "an assignment of a non-scalar expression to a scalar expression")
        return false
      end

      -- TODO: we could accept statements with no loop carrying dependence
      if cx:lookup_expr_type(lh) == S then
        cx:report_error_when_demanded(node, error_prefix ..
          "an assignment to a scalar expression")
        return false
      end

      -- TODO: for the moment we reject an assignment such as
      -- 'r[i] = i' where 'i' is of an index type
      if std.is_bounded_type(rh.expr_type) and
         rh.expr_type.dim >= 1 then
        cx:report_error_when_demanded(node, error_prefix ..
          "a corner case statement not supported for the moment")
        return false
      end

      if not (check_vectorizability.type(std.as_read(lh.expr_type)) and
              check_vectorizability.type(std.as_read(rh.expr_type))) then
        cx:report_error_when_demanded(node, error_prefix ..
          "an assignment between expressions that have inadmissible types")
        return false
      end

      -- bookkeeping for alias analysis
      collect_bounds(lh):map(function(pair)
        local ty, field = unpack(pair)
        if not bounds_lhs[ty] then bounds_lhs[ty] = {} end
        bounds_lhs[ty][field] = true
      end)
      collect_bounds(rh):map(function(pair)
        local ty, field = unpack(pair)
        if not bounds_rhs[ty] then bounds_rhs[ty] = {} end
        bounds_rhs[ty][field] = true
      end)
    end

    -- reject an aliasing between the read set and write set
    for ty, fields in pairs(bounds_lhs) do
      if bounds_rhs[ty] then
        for field, _ in pairs(fields) do
          if bounds_rhs[ty][field] then
            cx:report_error_when_demanded(node, error_prefix ..
              "aliasing references of path " ..  tostring(ty) .. "." .. field)
            return false
          end
        end
      end
    end
    return true

  elseif node:is(ast.typed.stat.ForNum) then
    for _, value in pairs(node.values) do
      if not check_vectorizability.expr(cx, value) then return false end
      if cx:lookup_expr_type(value) ~= S then
        cx:report_error_when_demanded(node,
          error_prefix ..  "a non-scalar loop condition")
        return false
      end
    end
    cx = cx:new_local_scope()
    cx:assign(node.symbol, S)
    return check_vectorizability.block(cx, node.block)

  elseif node:is(ast.typed.stat.If) then
    if not check_vectorizability.expr(cx, node.cond) then return false end
    if cx:lookup_expr_type(node.cond) ~= S then
      cx:report_error_when_demanded(node,
        error_prefix ..  "a non-scalar if-condition")
      return false
    end

    if not check_vectorizability.block(cx, node.then_block) then
      return false
    end

    for _, elseif_block in ipairs(node.elseif_blocks) do
      if not check_vectorizability.stat(cx, elseif_block) then return false end
    end

    return check_vectorizability.block(cx, node.else_block)

  elseif node:is(ast.typed.stat.Elseif) then
    if not check_vectorizability.expr(cx, node.cond) then return false end
    if cx:lookup_expr_type(node.cond) ~= S then
      cx:report_error_when_demanded(node,
        error_prefix ..  "a non-scalar if-condition")
      return false
    end

    return check_vectorizability.block(cx, node.block)

  else
    if node:is(ast.typed.stat.While) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.ForList) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.Repeat) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.VarUnpack) then
      cx:report_error_when_demanded(node, error_prefix .. "an unpack statement")

    elseif node:is(ast.typed.stat.Return) then
      cx:report_error_when_demanded(node, error_prefix .. "a return statement")

    elseif node:is(ast.typed.stat.Break) then
      cx:report_error_when_demanded(node, error_prefix .. "a break statement")

    elseif node:is(ast.typed.stat.Expr) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an expression as a statement")

    elseif node:is(ast.typed.stat.BeginTrace) then
      cx:report_error_when_demanded(node, error_prefix .. "a trace statement")

    elseif node:is(ast.typed.stat.EndTrace) then
      cx:report_error_when_demanded(node, error_prefix .. "a trace statement")

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end

    return false
  end
end

function check_vectorizability.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    -- treats variables from the outer scope as scalars
    local fact = cx.var_type:safe_lookup(node.value) or S
    cx:assign_expr_type(node, fact)
    return true

  elseif node:is(ast.typed.expr.FieldAccess) then
    if not check_vectorizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.IndexAccess) then
    if not node.value:is(ast.typed.expr.ID) then
      cx:report_error_when_demanded(node, "vectorization failed: " ..
        "array access should be in a form of 'a[exp]'")
      return false
    end

    if not check_vectorizability.expr(cx, node.value) or
       not check_vectorizability.expr(cx, node.index) then
      return false
    end

    if not std.is_bounded_type(node.index.expr_type) and
       cx:lookup_expr_type(node.index) ~= S then
      cx:report_error_when_demanded(node,
        error_prefix .. "an array access with a non-scalar index")
      return false
    end

    cx:assign_expr_type(node,
      join(cx:lookup_expr_type(node.value),
           cx:lookup_expr_type(node.index)))
    return true

  elseif node:is(ast.typed.expr.Unary) then
    if not check_vectorizability.expr(cx, node.rhs) then return true end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.rhs))
    return true

  elseif node:is(ast.typed.expr.Binary) then
    if not check_vectorizability.binary_op(node.op,
                                           std.as_read(node.expr_type)) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an unsupported binary operator")
      return false
    end

    if not check_vectorizability.expr(cx, node.lhs) or
       not check_vectorizability.expr(cx, node.rhs) then
      return false
    end

    cx:assign_expr_type(node,
      join(cx:lookup_expr_type(node.lhs),
           cx:lookup_expr_type(node.rhs)))
    return true

  elseif node:is(ast.typed.expr.Ctor) then
    cx:assign_expr_type(node, S)
    for _, field in pairs(node.fields) do
      if not check_vectorizability.expr(cx, field) then return false end
      cx:join_expr_type(node, cx:lookup_expr_type(field))
    end
    return true

  elseif node:is(ast.typed.expr.CtorRecField) then
    if not check_vectorizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.Constant) then
    cx:assign_expr_type(node, S)
    return true

  elseif node:is(ast.typed.expr.Call) then
    return check_vectorizability.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    if not check_vectorizability.expr(cx, node.arg) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.arg))
    return true

  elseif node:is(ast.typed.expr.UnsafeCast) then
    if not check_vectorizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.Deref) then
    if not check_vectorizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  else
    if node:is(ast.typed.expr.MethodCall) then
      cx:report_error_when_demanded(node, error_prefix .. "a method call")

    elseif node:is(ast.typed.expr.CtorListField) then
      cx:report_error_when_demanded(node, error_prefix .. "a list constructor")

    elseif node:is(ast.typed.expr.RawContext) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawFields) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawPhysical) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawRuntime) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.Isnull) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an isnull expression")

    elseif node:is(ast.typed.expr.New) then
      cx:report_error_when_demanded(node, error_prefix .. "a new expression")

    elseif node:is(ast.typed.expr.Null) then
      cx:report_error_when_demanded(node, error_prefix .. "a null expression")

    elseif node:is(ast.typed.expr.DynamicCast) then
      cx:report_error_when_demanded(node, error_prefix .. "a dynamic cast")

    elseif node:is(ast.typed.expr.StaticCast) then
      cx:report_error_when_demanded(node, error_prefix .. "a static cast")

    elseif node:is(ast.typed.expr.Region) then
      cx:report_error_when_demanded(node, error_prefix .. "a region expression")

    elseif node:is(ast.typed.expr.Partition) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a patition expression")

    elseif node:is(ast.typed.expr.CrossProduct) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a cross product operation")

    elseif node:is(ast.typed.expr.Function) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a function reference")

    elseif node:is(ast.typed.expr.RawValue) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a raw operator")

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end

    return false
  end
end

local predefined_functions = {}

function check_vectorizability.expr_call(cx, node)
  assert(node:is(ast.typed.expr.Call))

  if std.is_math_op(node.fn.value) then
    local fact = S
    for _, arg in pairs(node.args) do
      if not check_vectorizability.expr(cx, arg) then return false end
      fact = join(fact, cx:lookup_expr_type(arg))
    end
    cx:assign_expr_type(node, fact)
    return true

  else
    cx:report_error_when_demanded(node,
      error_prefix .. "an unsupported function call")
    return false
  end
end

function check_vectorizability.binary_op(op, arg_type)
  if (op == "max" or op == "min") and
     not (arg_type == float or arg_type == double) then
    return false
  end
  return arg_type:isprimitive()
end

-- check if the type is admissible to vectorizer
function check_vectorizability.type(ty)
  if ty:isprimitive() or std.is_bounded_type(ty) then
    return true
  elseif ty:isstruct() then
    for _, entry in pairs(ty.entries) do
      local entry_type = entry[2] or entry.type
      if not check_vectorizability.type(entry_type) then
        return false
      end
    end
    return true
  else
    return false
  end
end

-- visitor for each statement type
local vectorize_loops = {}

function vectorize_loops.block(node)
  return node {
    stats = node.stats:map(
      function(stat) return vectorize_loops.stat(stat) end)
  }
end

function vectorize_loops.stat_if(node)
  return node {
    then_block = vectorize_loops.block(node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return vectorize_loops.stat_elseif(block) end),
    else_block = vectorize_loops.block(node.else_block),
  }
end

function vectorize_loops.stat_elseif(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat_while(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat_for_num(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat_for_list(node)
  if node.options.vectorize:is(ast.options.Forbid) then return node end
  local cx = context:new_global_scope()
  cx:assign(node.symbol, V)
  cx.demanded = node.options.vectorize:is(ast.options.Demand)

  local vectorizable = check_vectorizability.block(cx, node.block)
  if vectorizable and not bounds_checks then
    return vectorize.stat_for_list(cx, node)
  else
    return node { block = node.block }
  end
end

function vectorize_loops.stat_repeat(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat_must_epoch(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat_block(node)
  return node { block = vectorize_loops.block(node.block) }
end

function vectorize_loops.stat(node)
  if node:is(ast.typed.stat.If) then
    return vectorize_loops.stat_if(node)

  elseif node:is(ast.typed.stat.While) then
    return vectorize_loops.stat_while(node)

  elseif node:is(ast.typed.stat.ForNum) then
    return vectorize_loops.stat_for_num(node)

  elseif node:is(ast.typed.stat.ForList) then
    if std.is_bounded_type(node.symbol:gettype()) and node.symbol:gettype().dim <= 1 then
      return vectorize_loops.stat_for_list(node)
    else
      return node { block = vectorize_loops.block(node.block) }
    end

  elseif node:is(ast.typed.stat.Repeat) then
    return vectorize_loops.stat_repeat(node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return vectorize_loops.stat_must_epoch(node)

  elseif node:is(ast.typed.stat.Block) then
    return vectorize_loops.stat_block(node)

  elseif node:is(ast.typed.stat.IndexLaunch) then
    return node

  elseif node:is(ast.typed.stat.Var) then
    return node

  elseif node:is(ast.typed.stat.VarUnpack) then
    return node

  elseif node:is(ast.typed.stat.Return) then
    return node

  elseif node:is(ast.typed.stat.Break) then
    return node

  elseif node:is(ast.typed.stat.Assignment) then
    return node

  elseif node:is(ast.typed.stat.Reduce) then
    return node

  elseif node:is(ast.typed.stat.Expr) then
    return node

  elseif node:is(ast.typed.stat.BeginTrace) then
    return node

  elseif node:is(ast.typed.stat.EndTrace) then
    return node

  elseif node:is(ast.typed.stat.MapRegions) then
    return node

  elseif node:is(ast.typed.stat.UnmapRegions) then
    return node

  elseif node:is(ast.typed.stat.RawDelete) then
    return node

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function vectorize_loops.top_task(node)
  local body = vectorize_loops.block(node.body)

  return node { body = body }
end

function vectorize_loops.top(node)
  if node:is(ast.typed.top.Task) then
    return vectorize_loops.top_task(node)
  else
    return node
  end
end

function vectorize_loops.entry(node)
  return vectorize_loops.top(node)
end

return vectorize_loops
