-- Copyright 2024 Stanford University, NVIDIA Corporation
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

-- Future optimization is required to main some invariants, check them here

local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local function all_with_provenance(acc, value)
  if not value[1] then
    return value
  end
  return acc
end

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

function context.new_global_scope(node)
  local cx = {
    valid = { true, nil },
  }
  return setmetatable(cx, context)
end

function context:update(field, value)
  self[field] = all_with_provenance(self[field], value)
end

function context:analysis_done()
  return not self.valid[1]
end

local function analyze_valid_node(node)
  if not (node:is(ast.typed.expr) or node:is(ast.typed.stat)) then
    return {true, node}
  end

  if node:is(ast.typed.stat.IndexLaunchNum) or
    node:is(ast.typed.stat.IndexLaunchList) or
    node:is(ast.typed.stat.Var) or
    node:is(ast.typed.stat.Assignment) or
    node:is(ast.typed.stat.Expr)
  then
    -- Ok for variables and expression statements to capture futures.
    return {true, node}
  end

  local function check(v)
    return ast.is_node(v) and
      v:is(ast.typed.expr) and
      not v:is(ast.typed.expr.ID) and
      std.is_future(std.as_read(v.expr_type))
  end

  for k, v in pairs(node) do
    if check(v) then
      return {false, node}
    elseif terralib.islist(v) then
      for _, vv in ipairs(v) do
        if check(vv) then
          return {false, node}
        end
      end
    end
  end
  return {true, node}
end

local function analyze_all_node(cx)
  return function(node, continuation)
    if cx.valid[1] then cx:update("valid", analyze_valid_node(node)) end
    if not cx:analysis_done() then continuation(node, true) end
  end
end

local function analyze_all(cx, node)
  return ast.traverse_node_continuation(
    analyze_all_node(cx),
    node)
end

local validate_futures = {}

function validate_futures.top_task(cx, node)
  if not node.body then return node end

  analyze_all(cx, node.body)

  local valid, valid_node = unpack(cx.valid)

  if not valid then
    report.error(valid_node, "Task contains nested unnormalized expressions on futures. This is an internal compiler error, please report it to the developers.")
  end
end

function validate_futures.top(cx, node)
  if node:is(ast.typed.top.Task) then
    validate_futures.top_task(cx, node)
  end
end

function validate_futures.entry(node)
  local cx = context.new_global_scope(node)
  validate_futures.top(cx, node)
end

validate_futures.pass_name = "validate_futures"

return validate_futures
