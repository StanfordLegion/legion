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

-- Helper for making dummy tasks

local ast = require("regent/ast")
local codegen = require("regent/codegen")
local data = require("common/data")
local std = require("regent/std")

local task_helper = {}

local function make_task(name, symbols, expr)
  local name = data.newtuple(name)
  local task = std.new_task(name)
  local variant = task:make_variant("primary")
  task:set_primary_variant(variant)
  local expr_type = std.as_read(expr.expr_type)
  local node = ast.typed.top.Task {
    name = name,
    params = symbols:map(
      function(symbol)
        return ast.typed.top.TaskParam {
          symbol = symbol,
          param_type = symbol:gettype(),
          future = false,
          annotations = ast.default_annotations(),
          span = ast.trivial_span(),
        }
      end
    ),
    return_type = expr_type,
    privileges = terralib.newlist(),
    coherence_modes = data.newmap(),
    flags = data.newmap(),
    conditions = data.newmap(),
    constraints = terralib.newlist(),
    body = ast.typed.Block {
      stats = terralib.newlist({
          ast.typed.stat.Return {
            value = expr,
            annotations = ast.default_annotations(),
            span = expr.span,
          },
      }),
      span = expr.span,
    },
    config_options = ast.TaskConfigOptions {
      leaf = true,
      inner = false,
      idempotent = true,
      replicable = false,
    },
    region_usage = false,
    region_divergence = false,
    metadata = false,
    prototype = task,
    annotations = ast.default_annotations(),
    span = expr.span,
  }
  task:set_type(
    terralib.types.functype(symbols:map(function(symbol) return symbol:gettype() end), expr_type, false))
  task:set_privileges(node.privileges)
  task:set_conditions(data.newmap())
  task:set_param_constraints(node.constraints)
  task:set_constraints(data.newmap())
  task:set_region_universe(data.newmap())
  return codegen.entry(node)
end

local make_identity_task_helper = data.weak_memoize(
  function(value_type)
    local name = "__identity"
    local value = std.newsymbol(value_type, "value")
    local expr = ast.typed.expr.ID {
      value = value,
      expr_type = value_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    return make_task(name, terralib.newlist({value}), expr)
  end)

function task_helper.make_identity_task(value_type)
  assert(terralib.types.istype(value_type))
  -- Strip futures here to maximize the value of memoization.
  if std.is_future(value_type) then
    value_type = value_type.result_type
  end
  return make_identity_task_helper(value_type)
end

local make_cast_task_helper = data.weak_memoize(
  function (from_type, to_type)
    local name = "__cast_" .. tostring(from_type) .. "_" .. tostring(to_type)
    local from_symbol = std.newsymbol(from_type, "from")
    local expr = ast.typed.expr.Cast {
      fn = ast.typed.expr.Function {
        value = to_type,
        expr_type = terralib.types.functype(
          terralib.newlist({std.untyped}), to_type, false),
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      arg = ast.typed.expr.ID {
        value = from_symbol,
        expr_type = from_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      expr_type = to_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    return make_task(name, terralib.newlist({from_symbol}), expr)
  end)

function task_helper.make_cast_task(from_type, to_type)
  assert(terralib.types.istype(from_type) and
           terralib.types.istype(to_type))
  -- Strip futures here to maximize the value of memoization.
  if std.is_future(from_type) then
    from_type = from_type.result_type
  end
  if std.is_future(to_type) then
    to_type = to_type.result_type
  end
  return make_cast_task_helper(from_type, to_type)
end

local make_unary_task_helper = data.weak_memoize(
  function (op, rhs_type, expr_type)
    local name = "__unary_" .. tostring(rhs_type) .. "_" .. tostring(op)
    local rhs_symbol = std.newsymbol(rhs_type, "rhs")
    local expr = ast.typed.expr.Unary {
      op = op,
      rhs = ast.typed.expr.ID {
        value = rhs_symbol,
        expr_type = rhs_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      expr_type = expr_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    return make_task(name, terralib.newlist({rhs_symbol}), expr)
  end)

function task_helper.make_unary_task(op, rhs_type, expr_type)
  assert(terralib.types.istype(rhs_type) and
           terralib.types.istype(expr_type))
  -- Strip futures here to maximize the value of memoization.
  if std.is_future(rhs_type) then
    rhs_type = rhs_type.result_type
  end
  if std.is_future(expr_type) then
    expr_type = expr_type.result_type
  end
  return make_unary_task_helper(op, rhs_type, expr_type)
end

local make_binary_task_helper = data.weak_memoize(
  function (op, lhs_type, rhs_type, expr_type)
    local name = "__binary_" .. tostring(lhs_type) .. "_" ..
      tostring(rhs_type) .. "_" .. tostring(op)
    local lhs_symbol = std.newsymbol(lhs_type, "lhs")
    local rhs_symbol = std.newsymbol(rhs_type, "rhs")
    local expr = ast.typed.expr.Binary {
      op = op,
      lhs = ast.typed.expr.ID {
        value = lhs_symbol,
        expr_type = lhs_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      rhs = ast.typed.expr.ID {
        value = rhs_symbol,
        expr_type = rhs_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      expr_type = expr_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    return make_task(name, terralib.newlist({lhs_symbol, rhs_symbol}), expr)
  end)

function task_helper.make_binary_task(op, lhs_type, rhs_type, expr_type)
  assert(terralib.types.istype(lhs_type) and
           terralib.types.istype(rhs_type) and
           terralib.types.istype(expr_type))
  -- Strip futures here to maximize the value of memoization.
  if std.is_future(lhs_type) then
    lhs_type = lhs_type.result_type
  end
  if std.is_future(rhs_type) then
    rhs_type = rhs_type.result_type
  end
  if std.is_future(expr_type) then
    expr_type = expr_type.result_type
  end
  return make_binary_task_helper(op, lhs_type, rhs_type, expr_type)
end

return task_helper
