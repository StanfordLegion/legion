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

-- Legion Divergence Optimizer
--
-- This pass analyzes the code for divergence resulting from
-- multi-region pointer accesses. Regions accessed in multi-pointer
-- derefs are marked to facilitate dynamic branch elision in code
-- generation.

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")
local union_find = require("regent/union_find")

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
    region_div = union_find.new(),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:mark_region_divergence(...)
  local rs = {...}
  if #rs > 0 then
    local r1 = rs[1]
    for _, r in ipairs(rs) do
      self.region_div:union_keys(r1, r)
    end
  end
end

local function analyze_region_divergence_node(cx)
  return function(node)
    if node:is(ast.typed.expr.Deref) then
      local value_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(value_type) and #value_type:bounds() > 1 then
        cx:mark_region_divergence(unpack(value_type:bounds()))
      end
    end
  end
end

local function analyze_region_divergence(cx, node)
  return ast.traverse_node_postorder(analyze_region_divergence_node(cx), node)
end

local function invert_forest(forest)
  local result = data.newmap()
  local ks = forest:keys()
  for _, k in ipairs(ks) do
    local root = forest:find_key(k)
    if not result[root] then
      result[root] = terralib.newlist()
    end
    result[root]:insert(k)
  end
  return result
end

local optimize_divergence = {}

function optimize_divergence.top_task(cx, node)
  local cx = cx:new_task_scope()
  analyze_region_divergence(cx, node.body)
  local divergence = invert_forest(cx.region_div)

  return node { region_divergence = divergence }
end

function optimize_divergence.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return optimize_divergence.top_task(cx, node)

  else
    return node
  end
end

function optimize_divergence.entry(node)
  local cx = context.new_global_scope()
  return optimize_divergence.top(cx, node)
end

optimize_divergence.pass_name = "optimize_divergence"

return optimize_divergence
