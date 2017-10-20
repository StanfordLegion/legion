-- Copyright 2017 Stanford University
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

-- Simple Loop Invariant Code Motion for with_scratch_fields
--
-- Hoists any 'with_scratch_fields' statements out from for loops

local ast = require("regent/ast")
local data = require("common/data")
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

function context.new_global_scope()
  local cx = {
    scratch_fields = terralib.newlist(),
  }
  setmetatable(cx, context)
  return cx
end

function context:push_local_scope()
  self.scratch_fields:insert(terralib.newlist())
end

function context:pop_local_scope()
  assert(self.scratch_fields:remove())
end

function context:insert_with_scratch_fields(node)
  if #self.scratch_fields == 0 then return end
  self.scratch_fields[#self.scratch_fields]:insert(node)
end

function context:has_any_with_scratch_fields()
  return #self.scratch_fields[#self.scratch_fields] > 0
end

function context:get_with_scratch_fields()
  if #self.scratch_fields == 0 then return nil end
  return self.scratch_fields[#self.scratch_fields]
end

local function hoist_stats_from_block(block, stats)
  local set = {}
  for i = 1, #stats do set[stats[i]] = true end
  return block {
    stats = data.filter(function(stat) return not set[stat] end, block.stats)
  }
end

local function hoist_with_scratch_fields(cx)
  return function (node, continuation)
    if node:is(ast.typed.stat.Var) then
      if #node.values == 1 and
         node.values[1]:is(ast.typed.expr.WithScratchFields) then
        cx:insert_with_scratch_fields(node)
      end
      return node

    elseif node:is(ast.typed.stat.While) then
      cx:push_local_scope()
      local block = continuation(node.block)
      local hoisted_stats = terralib.newlist()
      if cx:has_any_with_scratch_fields() then
        hoisted_stats:insertall(cx:get_with_scratch_fields())
        block = hoist_stats_from_block(block, cx:get_with_scratch_fields())
      end
      cx:pop_local_scope()

      node = node {
        cond = node.cond,
        block = block,
      }

      if #hoisted_stats > 0 then
        hoisted_stats:insert(node)
        return hoisted_stats
      else
        return node
      end

    elseif node:is(ast.typed.stat.ForNum) then
      cx:push_local_scope()
      local block = continuation(node.block)
      local hoisted_stats = terralib.newlist()
      if cx:has_any_with_scratch_fields() then
        hoisted_stats:insertall(cx:get_with_scratch_fields())
        block = hoist_stats_from_block(block, cx:get_with_scratch_fields())
      end
      cx:pop_local_scope()

      node = node {
        symbol = node.symbol,
        values = node.values,
        block = block,
      }

      if #hoisted_stats > 0 then
        hoisted_stats:insert(node)
        return hoisted_stats
      else
        return node
      end

    elseif node:is(ast.typed.stat.ForList) then
      cx:push_local_scope()
      local block = continuation(node.block)
      if cx:has_any_with_scratch_fields() then
        hoisted_stats:insertall(cx:get_with_scratch_fields())
        block = hoist_stats_from_block(block, cx:get_with_scratch_fields())
      end
      cx:pop_local_scope()

      node = node {
        symbol = node.symbol,
        value = node.value,
        block = block,
      }

      if #hoisted_stats > 0 then
        hoisted_stats:insert(node)
        return hoisted_stats
      else
        return node
      end

    elseif node:is(ast.typed.stat.Repeat) then
      cx:push_local_scope()
      local block = continuation(node.block)
      if cx:has_any_with_scratch_fields() then
        hoisted_stats:insertall(cx:get_with_scratch_fields())
        block = hoist_stats_from_block(block, cx:get_with_scratch_fields())
      end
      cx:pop_local_scope()

      node = node {
        block = block,
        until_cond = node.until_cond,
      }

      if #hoisted_stats > 0 then
        hoisted_stats:insert(node)
        return hoisted_stats
      else
        return node
      end

    else
      return continuation(node, true)
    end
  end
end

local licm_scratch_fields = {}

function licm_scratch_fields.top_task(cx, node)
  local cx = context:new_global_scope()
  local body = ast.flatmap_node_continuation(
      hoist_with_scratch_fields(cx),
      node.body)
  return node { body = body }
end

function licm_scratch_fields.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return licm_scratch_fields.top_task(cx, node)

  else
    return node
  end
end

function licm_scratch_fields.entry(node)
  local cx = context.new_global_scope({})
  return licm_scratch_fields.top(cx, node)
end

licm_scratch_fields.pass_name = "licm_scratch_fields"

return licm_scratch_fields
