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

-- Legion Loop Optimizer
--
-- Attempts to determine which loops can be transformed into index
-- space task launches.

local ast = require("legion/ast")
local log = require("legion/log")
local std = require("legion/std")

local optimize_loops = {}

local context = {}
context.__index = context

function context:new_loop_scope(loop_variable)
  local cx = {
    loop_variable = loop_variable,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local analyze_is_loop_invariant = {}

function analyze_is_loop_invariant.expr(cx, node)
  return false
end

function optimize_loops.block(cx, node)
  return ast.typed.Block {
    stats = node.stats:map(
      function(stat) return optimize_loops.stat(cx, stat) end),
  }
end

function optimize_loops.stat_if(cx, node)
  return ast.typed.StatIf {
    cond = node.cond,
    then_block = optimize_loops.block(cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return optimize_loops.stat_elseif(cx, block) end),
    else_block = optimize_loops.block(cx, node.else_block),
  }
end

function optimize_loops.stat_elseif(cx, node)
  return ast.typed.StatElseif {
    cond = node.cond,
    block = optimize_loops.block(cx, node.block),
  }
end

function optimize_loops.stat_while(cx, node)
  return ast.typed.StatWhile {
    cond = node.cond,
    block = optimize_loops.block(cx, node.block),
  }
end

function optimize_loops.stat_for_num(cx, node)
  if node.values[3] and not (
    node.values[3]:is(ast.typed.ExprConstant) and
    node.values[3].value == 1)
  then
    log.warn("loop optimization failed: stride not equal to 1")
    return node
  end

  if #node.block.stats ~= 1 then
    log.warn("loop optimization failed: body has multiple statements")
    return node
  end

  local body = node.block.stats[1]
  if not body:is(ast.typed.StatExpr) or
    not body.expr:is(ast.typed.ExprCall)
  then
    log.warn("loop optimization failed: body is not a function call")
    return node
  end

  if not std.is_task(body.expr.fn.value) then
    log.warn("loop optimization failed: function is not a task")
    return node
  end

  local loop_cx = cx.new_loop_scope(node.symbol)
  local args = body.expr.args
  local args_invariant = terralib.newlist()
  for i, arg in ipairs(args) do
    local invariant = analyze_is_loop_invariant.expr(cx, arg)
    args_invariant[i] = invariant
    if not invariant then
      if not arg:is(ast.typed.ExprIndexAccess) or
        not std.is_partition(std.as_read(arg.value.expr_type)) or
        not arg.index:is(ast.typed.ExprID) or
        not arg.index.value == node.symbol
      then
        log.warn("loop optimization failed: argument " .. tostring(i) .. " failed analysis")
        return node
      end

      local partition_type = arg.value.expr_type
      if not partition_type.disjoint then
        -- FIXME: This is actually ok if the parameter is read-only.
        log.warn("loop optimization failed: argument " .. tostring(i) .. " not disjoint")
        return node
      end
    end
  end

  log.warn("loop optimization succeeded")
  return ast.typed.StatIndexLaunch {
    symbol = node.symbol,
    domain = node.values,
    expr = body.expr,
    args_invariant = args_invariant,
  }
end

function optimize_loops.stat_for_list(cx, node)
  return ast.typed.StatForList {
    symbol = node.symbol,
    value = node.value,
    block = optimize_loops.block(cx, node.block),
  }
end

function optimize_loops.stat_repeat(cx, node)
  return ast.typed.StatRepeat {
    block = optimize_loops.block(cx, node.block),
    until_cond = node.until_cond,
  }
end

function optimize_loops.stat_block(cx, node)
  return ast.typed.StatBlock {
    block = optimize_loops.block(cx, node.block)
  }
end

function optimize_loops.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return optimize_loops.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return optimize_loops.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return optimize_loops.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return optimize_loops.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return optimize_loops.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return optimize_loops.stat_block(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return node

  elseif node:is(ast.typed.StatVarUnpack) then
    return node

  elseif node:is(ast.typed.StatReturn) then
    return node

  elseif node:is(ast.typed.StatBreak) then
    return node

  elseif node:is(ast.typed.StatAssignment) then
    return node

  elseif node:is(ast.typed.StatReduce) then
    return node

  elseif node:is(ast.typed.StatExpr) then
    return node

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function optimize_loops.stat_task(cx, node)
  local body = optimize_loops.block(cx, node.body)

  return ast.typed.StatTask {
    name = node.name,
    params = node.params,
    return_type = node.return_type,
    privileges = node.privileges,
    constraints = node.constraints,
    body = body,
    prototype = node.prototype,
  }
end

function optimize_loops.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return optimize_loops.stat_task(cx, node)

  else
    return node
  end
end

function optimize_loops.entry(node)
  local cx = context.new_global_scope({})
  return optimize_loops.stat_top(cx, node)
end

return optimize_loops
