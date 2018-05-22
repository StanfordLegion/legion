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

local evaluate_cache = data.newmap()
local function evaluate(node)
  assert(node.lhs:is(ast.typed.expr.Constant) and node.rhs:is(ast.typed.expr.Constant))
  local key = data.newtuple(node.lhs.value, node.op, node.rhs.value)
  local value = evaluate_cache[key]
  if value == nil then
    value = (terra()
      return [base.quote_binary_op(node.op, node.lhs.value, node.rhs.value)]
    end)()
    evaluate_cache[key] = value
  end
  assert(value ~= nil)
  return value
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
    return evaluate(node)
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

  else
    assert(false)
  end
end

local function strip_casts(node)
  if node:is(ast.typed.expr.Cast) then
    return node.arg
  end
  return node
end

function affine.analyze_index_noninterference(index, other_index)
  -- Can we prove that these two indexes will always be
  -- non-interfering?

  local index = strip_casts(index)
  local other_index = strip_casts(other_index)

  -- Attempt a simple affine analysis.
  local x1, b1 = get_affine_coefficients(index)
  local x2, b2 = get_affine_coefficients(other_index)
  if x1 == x1 and not data.vector.eq(b1, b2) then
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
