-- Copyright 2022 Stanford University
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

-- Skip Empty Tasks
--
-- This pass will modify the program so that the following task calls
-- are skipped:
--   * Task is the top node of an expression statement
--   * Task has at least one region argument
--   * All region arguments are empty (volume == 0)
--   * Task has not been optimized as an index launch

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local context = {}

function context:__index (field)
  local value = context [field]
  if value ~= nil then
    return value
  end
  error ("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex (field, value)
  error ("context has no field '" .. field .. "' (in assignment)", 2)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local skip_empty_tasks = {}

local function skip_empty_tasks_sat_expr(cx, node)
  local call = node.expr

  if not call:is(ast.typed.expr.Call) then
    return node -- Not a call
  end

  if not std.is_task(call.fn.value) then
    return node -- Not a call to a task
  end

  if #data.filter(
    function(arg) return std.is_region(std.as_read(arg.expr_type)) end,
    call.args) == 0
  then
    return node -- Task call has no region arguments
  end

  local conditions = terralib.newlist()
  local result_stats = terralib.newlist()
  local result_args = terralib.newlist()
  for _, arg in ipairs(call.args) do
    local arg_type = std.as_read(arg.expr_type)

    if not std.is_region(arg_type) then
      result_args:insert(arg)
    else
      local arg_symbol = std.newsymbol(arg_type)
      local arg_var = ast.typed.stat.Var {
        symbol = arg_symbol,
        type = arg_type,
        value = arg,
        annotations = ast.default_annotations(),
        span = arg.span,
      }
      local arg_ref = ast.typed.expr.ID {
        value = arg_symbol,
        expr_type = std.rawref(&arg_type),
        annotations = ast.default_annotations(),
        span = arg.span,
      }
      local condition = ast.typed.expr.Binary {
        op = "~=",
        lhs = ast.typed.expr.FieldAccess {
          value = ast.typed.expr.FieldAccess {
            value = arg_ref,
            field_name = "ispace",
            expr_type = arg_type:ispace(),
            annotations = ast.default_annotations(),
            span = arg.span,
          },
          field_name = "volume",
          expr_type = int64,
          annotations = ast.default_annotations(),
          span = arg.span,
        },
        rhs = ast.typed.expr.Constant {
          value = 0,
          expr_type = int64,
          annotations = ast.default_annotations(),
          span = arg.span,
        },
        expr_type = bool,
        annotations = ast.default_annotations(),
        span = arg.span,
      }

      result_stats:insert(arg_var)
      result_args:insert(arg_ref)
      conditions:insert(condition)
    end
  end

  local skip_condition = data.reduce(
    function(expr, condition)
      return ast.typed.expr.Binary {
        op = "or",
        lhs = expr,
        rhs = condition,
        expr_type = bool,
        annotations = ast.default_annotations(),
        span = call.span,
      }
    end,
    conditions)

  local maybe_call = ast.typed.stat.If {
    cond = skip_condition,
    then_block = ast.typed.Block {
      stats = terralib.newlist({ node { expr = call { args = result_args } } }),
      span = call.span,
    },
    elseif_blocks = terralib.newlist(),
    else_block = ast.typed.Block {
      stats = terralib.newlist(),
      span = call.span,
    },
    annotations = ast.default_annotations(),
    span = call.span,
  }

  result_stats:insert(maybe_call)

  return ast.typed.stat.Block {
    block = ast.typed.Block {
      stats = result_stats,
      span = call.span,
    },
    annotations = ast.default_annotations(),
    span = call.span,
  }
end

local function skip_empty_tasks_stat_block(cx, node)
  return node { block = skip_empty_tasks.block(cx, node.block) }
end

local function skip_empty_tasks_if(cx, node)
  local then_block = skip_empty_tasks.block(cx, node.then_block)
  local elseif_blocks = node.elseif_blocks:map(function(elseif_block)
    return skip_empty_tasks.stat(cx, elseif_block)
  end)
  local else_block = skip_empty_tasks.block(cx, node.else_block)
  return node {
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
  }
end

local function skip_empty_tasks_elseif(cx, node)
  local block = skip_empty_tasks.block(cx, node.block)
  return node { block = block }
end

local function do_nothing(cx, node) return node end

local skip_empty_tasks_stat_table = {
  [ast.typed.stat.While]     = skip_empty_tasks_stat_block,
  [ast.typed.stat.ForNum]    = skip_empty_tasks_stat_block,
  [ast.typed.stat.ForList]   = skip_empty_tasks_stat_block,
  [ast.typed.stat.Repeat]    = skip_empty_tasks_stat_block,
  [ast.typed.stat.Block]     = skip_empty_tasks_stat_block,
  [ast.typed.stat.MustEpoch] = skip_empty_tasks_stat_block,
  [ast.typed.stat.If]        = skip_empty_tasks_if,
  [ast.typed.stat.Elseif]    = skip_empty_tasks_elseif,
  [ast.typed.stat.Expr]      = skip_empty_tasks_sat_expr,
}

local skip_empty_tasks_stat = ast.make_single_dispatch(
  skip_empty_tasks_stat_table,
  {},
  do_nothing)

function skip_empty_tasks.stat(cx, node)
  return skip_empty_tasks_stat(cx)(node)
end

function skip_empty_tasks.block(cx, node)
  return node {
    stats = node.stats:map(function(stat) return skip_empty_tasks.stat(cx, stat) end),
  }
end

function skip_empty_tasks.top_task(cx, node)
  if not node.body then return node end

  local body = skip_empty_tasks.block(cx, node.body)

  return node { body = body }
end

function skip_empty_tasks.top(cx, node)
  if node:is(ast.typed.top.Task) and
     not node.config_options.leaf
  then
    return skip_empty_tasks.top_task(cx, node)

  else
    return node
  end
end

function skip_empty_tasks.entry(node)
  local cx = context.new_global_scope({})
  return skip_empty_tasks.top(cx, node)
end

skip_empty_tasks.pass_name = "skip_empty_tasks"

return skip_empty_tasks
