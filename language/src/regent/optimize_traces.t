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

-- Legion Trace Optimizer
--
-- Inserts begin/end trace calls to control runtime tracing.

local ast = require("regent/ast")
local std = require("regent/std")

local c = std.c

local context = {}
context.__index = context

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

local function apply_tracing_node(cx)
  return function(node)
    if node:is(ast.typed.stat.While) or
      node:is(ast.typed.stat.ForNum) or
      node:is(ast.typed.stat.ForList) or
      node:is(ast.typed.stat.Repeat) or
      node:is(ast.typed.stat.Block)
    then
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
      stats:insertall(node.block.stats)
      stats:insert(
        ast.typed.stat.EndTrace {
          trace_id = trace_id,
          annotations = ast.default_annotations(),
          span = node.span,
      })

      return node { block = node.block { stats = stats } }

    else
      return node
    end
  end
end

local function apply_tracing(cx, node)
  return ast.map_node_postorder(apply_tracing_node(cx), node)
end

local optimize_traces = {}

function optimize_traces.top_task(cx, node)
  local cx = cx:new_task_scope()
  local body = apply_tracing(cx, node.body)

  return node { body = body }
end

function optimize_traces.top(cx, node)
  if node:is(ast.typed.top.Task) then
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
