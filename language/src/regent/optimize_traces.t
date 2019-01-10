-- Copyright 2019 Stanford University
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

-- Legion Trace Optimizer
--
-- Inserts begin/end trace calls to control runtime tracing.

local ast = require("regent/ast")
local std = require("regent/std")

local c = std.c

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

function context:new_task_scope()
  local cx = {
    next_trace_id = 1,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local optimize_traces = {}

local function apply_tracing_while(cx, node)
  if not node.annotations.trace:is(ast.annotation.Demand) then
    return node
  end

  if node.cond:is(ast.typed.expr.FutureGetResult) and
     node.cond.value:is(ast.typed.expr.Call) then

    local trace_id = ast.typed.expr.Constant {
      value = cx.next_trace_id,
      expr_type = c.legion_trace_id_t,
      annotations = ast.default_annotations(),
      span = node.span,
    }
    cx.next_trace_id = cx.next_trace_id + 1

    local call = node.cond.value
    local future_type = call.expr_type
    assert(std.is_future(future_type))
    local future_var = std.newsymbol(future_type, "__while_cond")

    local inner_stats = terralib.newlist()

    inner_stats:insert(
      ast.typed.stat.BeginTrace {
        trace_id = trace_id,
        annotations = ast.default_annotations(),
        span = node.span,
    })
    inner_stats:insert(
      ast.typed.stat.Block {
        block = optimize_traces.block(cx, node.block),
        annotations = ast.default_annotations(),
        span = node.span,
    })
    inner_stats:insert(
      ast.typed.stat.Assignment {
        lhs = ast.typed.expr.ID {
          value = future_var,
          expr_type = std.rawref(&future_type),
          annotations = ast.default_annotations(),
          span = node.span,
        },
        rhs = call,
        annotations = ast.default_annotations(),
        span = node.span,
      }
    )
    inner_stats:insert(
      ast.typed.stat.EndTrace {
        trace_id = trace_id,
        annotations = ast.default_annotations(),
        span = node.span,
    })

    local outer_stats = terralib.newlist()

    outer_stats:insert(
      ast.typed.stat.Var {
        symbol = future_var,
        type = future_type,
        value = call,
        annotations = ast.default_annotations(),
        span = node.span,
    })
    outer_stats:insert(
      node {
        cond = node.cond {
          value = ast.typed.expr.ID {
            value = future_var,
            expr_type = future_type,
            annotations = ast.default_annotations(),
            span = node.span,
          }
        },
        block = node.block {
          stats = inner_stats,
        }
    })

    return ast.typed.stat.Block {
      block = ast.typed.Block {
        stats = outer_stats,
        span = node.span,
      },
      annotations = ast.default_annotations(),
      span = node.span,
    }

  else
    local trace_id = ast.typed.expr.Constant {
      value = cx.next_trace_id,
      expr_type = c.legion_trace_id_t,
      annotations = ast.default_annotations(),
      span = node.span,
    }
    cx.next_trace_id = cx.next_trace_id + 1

    local stats = terralib.newlist()
    stats:insert(
      ast.typed.stat.BeginTrace {
        trace_id = trace_id,
        annotations = ast.default_annotations(),
        span = node.span,
    })
    stats:insert(
      ast.typed.stat.Block {
        block = optimize_traces.block(cx, node.block),
        annotations = ast.default_annotations(),
        span = node.span,
    })
    stats:insert(
      ast.typed.stat.EndTrace {
        trace_id = trace_id,
        annotations = ast.default_annotations(),
        span = node.span,
    })

    return node { block = node.block { stats = stats } }
  end
end

local function apply_tracing_block(cx, node)
  if not node.annotations.trace:is(ast.annotation.Demand) then
    return node
  end

  local trace_id = ast.typed.expr.Constant {
    value = cx.next_trace_id,
    expr_type = c.legion_trace_id_t,
    annotations = ast.default_annotations(),
    span = node.span,
  }
  cx.next_trace_id = cx.next_trace_id + 1

  local stats = terralib.newlist()
  stats:insert(
    ast.typed.stat.BeginTrace {
      trace_id = trace_id,
      annotations = ast.default_annotations(),
      span = node.span,
  })
  stats:insert(
    ast.typed.stat.Block {
      block = optimize_traces.block(cx, node.block),
      annotations = ast.default_annotations(),
      span = node.span,
  })
  stats:insert(
    ast.typed.stat.EndTrace {
      trace_id = trace_id,
      annotations = ast.default_annotations(),
      span = node.span,
  })

  return node { block = node.block { stats = stats } }
end

local function apply_tracing_if(cx, node)
  local then_block = optimize_traces.block(cx, node.then_block)
  local elseif_blocks = node.elseif_blocks:map(function(elseif_block)
    return optimize_traces.stat(cx, elseif_block)
  end)
  local else_block = optimize_traces.block(cx, node.else_block)
  return node {
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
  }
end

local function apply_tracing_elseif(cx, node)
  local block = optimize_traces.block(cx, node.block)
  return node { block = block }
end

local function do_nothing(cx, node) return node end

local node_tracing = {
  [ast.typed.stat.While]     = apply_tracing_while,
  [ast.typed.stat.ForNum]    = apply_tracing_block,
  [ast.typed.stat.ForList]   = apply_tracing_block,
  [ast.typed.stat.Repeat]    = apply_tracing_block,
  [ast.typed.stat.Block]     = apply_tracing_block,
  [ast.typed.stat.MustEpoch] = apply_tracing_block,
  [ast.typed.stat.If]        = apply_tracing_if,
  [ast.typed.stat.Elseif]    = apply_tracing_elseif,
  [ast.typed.stat]           = do_nothing,
}

local apply_tracing_node = ast.make_single_dispatch(
  node_tracing,
  {})

function optimize_traces.stat(cx, node)
  return apply_tracing_node(cx)(node)
end

function optimize_traces.block(cx, block)
  return block {
    stats = block.stats:map(function(stat) return optimize_traces.stat(cx, stat) end),
  }
end

function optimize_traces.top_task(cx, node)
  local cx = cx:new_task_scope()
  local body = node.body and optimize_traces.block(cx, node.body) or false

  return node { body = body }
end

function optimize_traces.top(cx, node)
  if node:is(ast.typed.top.Task) and
     not node.config_options.leaf
  then
    return optimize_traces.top_task(cx, node)

  else
    return node
  end
end

function optimize_traces.entry(node)
  local cx = context.new_global_scope({})
  return optimize_traces.top(cx, node)
end

optimize_traces.pass_name = "optimize_traces"

return optimize_traces
