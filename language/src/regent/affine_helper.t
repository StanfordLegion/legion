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

-- Helper for Affine Analysis

local base = require("regent/std_base")
local ast = require("regent/ast")
local data = require("common/data")

local affine = {}

-- #####################################
-- ## Constant Evaluation
-- #################

function affine.is_constant_expr(node)
  if node:is(ast.typed.expr.Constant) then
    return true
  end

  if node:is(ast.typed.expr.Ctor) then
    for _, field in ipairs(node.fields) do
      if not affine.is_constant_expr(field.value) then
        return false
      end
    end
    return true
  end

  if node:is(ast.typed.expr.Cast) then
    return affine.is_constant_expr(node.arg)
  end

  if node:is(ast.typed.expr.Unary) then
    return affine.is_constant_expr(node.rhs)
  end

  if node:is(ast.typed.expr.Binary) then
    return node.lhs:is(ast.typed.expr.Constant) and
           node.rhs:is(ast.typed.expr.Constant)
  end

  return false
end

local convert_constant_expr

local function convert_terra_constant(value)
  if terralib.isconstant(value) then
    -- Terra, pretty please give me the value inside this constant
    local value = (terra() return value end)()
    if base.types.is_index_type(value.type) then
      return data.newvector(
        unpack(
          value.type.fields:map(
            function(field_name) return value.__ptr[field_name] end)))
    else
      return value
    end
  else
    return value
  end
end

local function convert_ctor_to_constant(node)
  assert(node:is(ast.typed.expr.Ctor))
  return data.newvector(
    unpack(
      node.fields:map(
        function(field)
          return convert_constant_expr(field.value)
  end)))
end

function convert_constant_expr(node)
  if node:is(ast.typed.expr.Constant) then
    return convert_terra_constant(node.value)
  elseif node:is(ast.typed.expr.Ctor) then
    return convert_ctor_to_constant(node)
  elseif node:is(ast.typed.expr.Cast) then
    return convert_constant_expr(node.arg)
  elseif node:is(ast.typed.expr.Unary) and node.op == "-" then
    return -convert_constant_expr(node.rhs)
  elseif node:is(ast.typed.expr.Binary) then
    local op = node.op
    if op == "*" then
      return convert_constant_expr(node.lhs) * convert_constant_expr(node.rhs)
    elseif op == "/" then
      return convert_constant_expr(node.lhs) / convert_constant_expr(node.rhs)
    elseif op == "%" then
      return convert_constant_expr(node.lhs) % convert_constant_expr(node.rhs)
    elseif op == "+" then
      return convert_constant_expr(node.lhs) + convert_constant_expr(node.rhs)
    elseif op == "-" then
      return convert_constant_expr(node.lhs) - convert_constant_expr(node.rhs)
    elseif op == "<" then
      return convert_constant_expr(node.lhs) < convert_constant_expr(node.rhs)
    elseif op == ">" then
      return convert_constant_expr(node.lhs) > convert_constant_expr(node.rhs)
    elseif op == "<=" then
      return convert_constant_expr(node.lhs) <= convert_constant_expr(node.rhs)
    elseif op == ">=" then
      return convert_constant_expr(node.lhs) >= convert_constant_expr(node.rhs)
    elseif op == "==" then
      return convert_constant_expr(node.lhs) == convert_constant_expr(node.rhs)
    elseif op == "~=" then
      return convert_constant_expr(node.lhs) ~= convert_constant_expr(node.rhs)
    elseif op == "and" then
      return convert_constant_expr(node.lhs) and convert_constant_expr(node.rhs)
    elseif op == "or" then
      return convert_constant_expr(node.lhs) or convert_constant_expr(node.rhs)
    elseif op == "max" then
      local lhs, rhs = convert_constant_expr(node.lhs), convert_constant_expr(node.rhs)
      if lhs < rhs then
        return rhs
      end
      return lhs
    elseif op == "min" then
      local lhs, rhs = convert_constant_expr(node.lhs), convert_constant_expr(node.rhs)
      if lhs < rhs then
        return lhs
      end
      return rhs
    else
      assert(false)
    end
  else
    assert(false)
  end
end

-- #####################################
-- ## Affine Analysis
-- #################

local function is_analyzable_index(node)
  return affine.is_constant_expr(node) or
    (node:is(ast.typed.expr.ID) and not base.types.is_rawref(node.expr_type))
end

function affine.is_analyzable_index_expression(node)
  return is_analyzable_index(node) or
    (node:is(ast.typed.expr.Binary) and
       is_analyzable_index(node.lhs) and
       affine.is_constant_expr(node.rhs) and
       (node.op == "+" or node.op == "-"))
end

local function get_affine_coefficients(node)
  -- Return `x`, `b` satisfying the equation `expr = x + b`.

  if affine.is_constant_expr(node) then
    return nil, convert_constant_expr(node)
  end

  if node:is(ast.typed.expr.ID) and not base.types.is_rawref(node.expr_type) then
    return node.value, 0
  end

  if node:is(ast.typed.expr.Binary) and
      node.lhs:is(ast.typed.expr.ID) and
      (node.op == "+" or node.op == "-")
  then
    local rhs_value = convert_constant_expr(node.rhs)
    return node.lhs.value, rhs_value * (node.op == "-" and -1 or 1)
  end
end

local function strip_casts(node)
  if node:is(ast.typed.expr.Cast) then
    return node.arg
  end
  return node
end

affine.convert_constant_expr = convert_constant_expr

function affine.analyze_index_noninterference_self(loop_index, arg, field_name)
  local arg_index = strip_casts(arg)

  if arg_index:is(ast.typed.expr.ID) then
    return arg_index == loop_index

  elseif arg_index:is(ast.typed.expr.Binary) then
    if arg_index.op == "+" or arg_index.op == "-" then
      local left = affine.analyze_index_noninterference_self(loop_index, arg_index.lhs, field_name)
      local right = affine.analyze_index_noninterference_self(loop_index, arg_index.rhs, field_name)
      return left ~= right

    elseif arg_index.op == "*" then
      if affine.is_constant_expr(arg_index.lhs) then
        local coeff = affine.convert_constant_expr(arg_index.lhs)
        return (coeff >= 1 or coeff <= -1) and
          affine.analyze_index_noninterference_self(loop_index, arg_index.rhs, field_name)
      elseif affine.is_constant_expr(arg_index.rhs) then
        local coeff = affine.convert_constant_expr(arg_index.rhs)
        return -1 <= coeff and coeff <= 1 and
          affine.analyze_index_noninterference_self(loop_index, arg_index.lhs, field_name)
      end
    elseif arg_index.op == "/" then
      if affine.is_constant_expr(arg_index.rhs) then
        local coeff = affine.convert_constant_expr(arg_index.rhs)
        return -1 <= coeff and coeff <= 1 and
          affine.analyze_index_noninterference_self(loop_index, arg_index.lhs, field_name)
      end
    -- TODO: add mod operator check
    else
      return false
    end

  elseif arg_index:is(ast.typed.expr.Ctor) then
    local loop_index_type = loop_index:gettype()
    if loop_index_type.fields then

      for _, loop_index_field in ipairs(loop_index_type.fields) do
        local field_present = false
        for _, ctor_field in ipairs(arg_index.fields) do
          field_present = field_present or
            affine.analyze_index_noninterference_self(loop_index, ctor_field.value, loop_index_field)
        end
        if not field_present then
          return false
        end
      end

      return true
    else
      assert(false) -- unreachable?
    end

  elseif arg_index:is(ast.typed.expr.FieldAccess) then
    local id = arg_index.value
    return loop_index == id.value and field_name == arg_index.field_name
  end

  return false
end

function affine.analyze_index_noninterference(index, other_index)
  -- Can we prove that these two indexes will always be
  -- non-interfering?

  local index = strip_casts(index)
  local other_index = strip_casts(other_index)

  -- Attempt a simple affine analysis.
  local x1, b1 = get_affine_coefficients(index)
  local x2, b2 = get_affine_coefficients(other_index)
  if b1 and b2 and x1 == x2 and not data.vector.eq(b1, b2) then
    return true
  end

  return false
end

-- This is used in methods such as subregion_constant where the index
-- of a subregion has to be munged to make it safe to go in a map.
function affine.get_subregion_index(node)
  assert(ast.is_node(node))

  local node = strip_casts(node)

  -- FIXME: This breaks something in RDIR right now and causes
  -- subregions to not be deduplicated correctly. For the moment, just
  -- limit the set of expressions we consider to strict constants. The
  -- only side effect of this should be that we will create more
  -- copies of the region and partition types, but the affine analysis
  -- should still consider them equivalent.

  -- if affine.is_constant_expr(node) then
  --   return convert_constant_expr(node)
  -- end

  if node:is(ast.typed.expr.Constant) then
    return convert_constant_expr(node)
  end


  if node:is(ast.typed.expr.ID) and not base.types.is_rawref(node.expr_type) then
    return node.value
  end

  return node
end

return affine
