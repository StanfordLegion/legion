-- Copyright 2018 Stanford University
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

function eliminate_dead_code.stat(node)
  if node:is(ast.typed.stat.If) and
     node.cond:is(ast.typed.expr.Constant) and
     not node.cond.value and #node.elseif_blocks == 0
  then
    return node.else_block.stats
  else
    return node
  end
end

function eliminate_dead_code.top(node)
  if node:is(ast.typed.top.Task) then
    return node {
      body = ast.flatmap_node_postorder(eliminate_dead_code.stat, node.body),
    }
  else
    return node
  end
end

function eliminate_dead_code.entry(node)
  return eliminate_dead_code.top(node)
end

eliminate_dead_code.pass_name = "eliminate_dead_code"

return eliminate_dead_code
