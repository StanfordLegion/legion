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

-- Very simple dead code elimination

local ast = require("regent/ast")

local eliminate_dead_code = {}

local function eliminate_dead_if_node(node, continuation)
  if node.cond:is(ast.typed.expr.Constant) and
     not node.cond.value and #node.elseif_blocks == 0
  then
    return node.else_block.stats

  else
    return continuation(node, true)
  end
end

local function apply_dce_block_stat(node, continuation)
  return node { block = continuation(node.block) }
end

local function apply_dce_block(node, continuation)
  return node { stats = continuation(node.stats) }
end

local function do_nothing(node, continuation) return node end

local dce_table = {
  [ast.typed.stat.If]              = eliminate_dead_if_node,
  [ast.typed.stat.Elseif]          = apply_dce_block_stat,
  [ast.typed.stat.While]           = apply_dce_block_stat,
  [ast.typed.stat.ForNum]          = apply_dce_block_stat,
  [ast.typed.stat.ForList]         = apply_dce_block_stat,
  [ast.typed.stat.Repeat]          = apply_dce_block_stat,
  [ast.typed.stat.MustEpoch]       = apply_dce_block_stat,
  [ast.typed.stat.Block]           = apply_dce_block_stat,
  [ast.typed.stat.ParallelizeWith] = apply_dce_block_stat,

  [ast.typed.Block]             = apply_dce_block,
}

local apply_dce_node = ast.make_single_dispatch(
  dce_table,
  {},
  do_nothing)

function eliminate_dead_code.top(node)
  if node:is(ast.typed.top.Task) then
    return node { body = ast.flatmap_node_continuation(apply_dce_node(), node.body) }
  else
    return node
  end
end

function eliminate_dead_code.entry(node)
  return eliminate_dead_code.top(node)
end

eliminate_dead_code.pass_name = "eliminate_dead_code"

return eliminate_dead_code
