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

-- Regent Auto-parallelizer

local ast = require("regent/ast")
local std = require("regent/std")

local infer_constraints     = require("regent/parallelizer/infer_constraints")
local solve_constraints     = require("regent/parallelizer/solve_constraints")
local rewrite_task_launches = require("regent/parallelizer/rewrite_task_launches")

local parallelize_tasks = {}

function parallelize_tasks.stat_parallelize_with(cx, stat)
  local solution = solve_constraints.solve(cx, stat)
  return rewrite_task_launches.rewrite(solution, cx.constraints, stat)
end

function parallelize_tasks.stat_block(cx, stat)
  local block = parallelize_tasks.block(cx, stat.block)
  return stat { block = block }
end

function parallelize_tasks.stat_if(cx, stat)
  local then_block = parallelize_tasks.block(cx, stat.then_block)
  local else_block = parallelize_tasks.block(cx, stat.else_block)
  return stat {
    then_block = then_block,
    else_block = else_block,
  }
end

function parallelize_tasks.pass_through_stat(cx, stat)
  return stat
end

local parallelize_tasks_stat_table = {
  [ast.typed.stat.ParallelizeWith] = parallelize_tasks.stat_parallelize_with,

  [ast.typed.stat.ForList]    = parallelize_tasks.stat_block,
  [ast.typed.stat.ForNum]     = parallelize_tasks.stat_block,
  [ast.typed.stat.While]      = parallelize_tasks.stat_block,
  [ast.typed.stat.Repeat]     = parallelize_tasks.stat_block,
  [ast.typed.stat.Block]      = parallelize_tasks.stat_block,
  [ast.typed.stat.MustEpoch]  = parallelize_tasks.stat_block,

  [ast.typed.stat.If]         = parallelize_tasks.stat_if,

  [ast.typed.stat]            = parallelize_tasks.pass_through_stat,
}

local parallelize_tasks_stat = ast.make_single_dispatch(
  parallelize_tasks_stat_table,
  {ast.typed.stat})

function parallelize_tasks.stat(cx, node)
  return parallelize_tasks_stat(cx)(node)
end

function parallelize_tasks.block(cx, block)
  local stats = block.stats:map(function(stat)
    return parallelize_tasks.stat(cx, stat)
  end)
  return block { stats = stats }
end

function parallelize_tasks.top_task(node)
  local cx = { constraints = node.prototype:get_constraints() }
  local body = parallelize_tasks.block(cx, node.body)
  return node { body = body }
end

function parallelize_tasks.entry(node)
  if node:is(ast.typed.top.Task) then
    if node.annotations.parallel:is(ast.annotation.Demand) then
      assert(node.config_options.leaf)
      assert(node.metadata)
      infer_constraints.top_task(node)
      return node
    elseif not node.config_options.leaf then
      return parallelize_tasks.top_task(node)
    else
      return node
    end
  else
    return node
  end
end

parallelize_tasks.pass_name = "parallelize_tasks"

return parallelize_tasks
