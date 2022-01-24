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

-- Bishop Regex Library

local ast = require("bishop/ast")

local regex = ast.make_factory("ast")

regex:inner("symbol", { "value" })
regex.symbol:leaf("TaskId", { "task_name" })
regex.symbol:leaf("Class", { "class_name" })
regex.symbol:leaf("Constraint", { "constraint" })

regex:inner("expr")
regex.expr:leaf("Concat", { "values" })
regex.expr:leaf("Disj", { "values" })
regex.expr:leaf("Kleene", { "value" })

function regex.pretty(node)
  if node:is(regex.symbol.TaskId) then
    return node.task_name
  elseif node:is(regex.symbol.Class) then
    return node.class_name
  elseif node:is(regex.symbol.Constraint) then
    return node.constraint:unparse()
  elseif node:is(regex.expr.Concat) or node:is(regex.expr.Disj) then
    local delim = " "
    if node:is(regex.expr.Disj) then delim = " | " end
    local str = regex.pretty(node.values[1])
    for i = 2, #node.values do
      str = str .. delim .. regex.pretty(node.values[i])
    end
    if node:is(regex.expr.Disj) then
      str = "(" .. str .. ")"
    end
    return str
  elseif node:is(regex.expr.Kleene) then
    if node.value:is(regex.expr.Disj) or
       node.value:is(regex.symbol.TaskId) or
       node.value:is(regex.symbol.Class) or
       node.value:is(regex.symbol.Constraint) then
      return regex.pretty(node.value) .. "*"
    else
      return "(" .. regex.pretty(node.value) .. ")*"
    end
  else
    assert(false, "unreachable")
  end
end

return regex
