-- Copyright 2016 Stanford University, NVIDIA Corporation
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

-- Regent Task Config Option Optimizer
--
-- Determines which of the following config options are applicable to
-- a task:
--
--   * Leaf: Task issues no sub-operations
--   * Inner: Task does not access any regions
--   * Idempotent: Task has no external side-effects
--
-- (Currently the optimization returns false for idempotent.)

local ast = require("regent/ast")
local data = require("regent/data")
local log = require("regent/log")
local std = require("regent/std")

local context = {}
context.__index = context

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local function analyze_leaf_node(cx)
  return function(node)
    if node:is(ast.typed.expr.Call) then
      return not std.is_task(node.fn.value)
    elseif node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.New) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.stat.IndexLaunchNum) or
      node:is(ast.typed.stat.IndexLaunchList)
    then
      return false
    end
    return true
  end
end

local function analyze_leaf(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_leaf_node(cx),
    data.all,
    node, true)
end

local function analyze_inner_node(cx)
  return function(node)
    if node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.IndexAccess)
    then
      return not std.is_ref(node.expr_type)
    elseif node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.stat.MapRegions)
    then
      return false
    end
    return true
  end
end

local function analyze_inner(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_inner_node(cx),
    data.all,
    node, true)
end

local optimize_config_options = {}

function optimize_config_options.top_task(cx, node)
  local leaf = analyze_leaf(cx, node.body)
  local inner = not leaf and analyze_inner(cx, node.body)

  return node {
    config_options = ast.TaskConfigOptions {
      leaf = leaf,
      inner = inner,
      idempotent = false,
    },
  }
end

function optimize_config_options.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return optimize_config_options.top_task(cx, node)

  else
    return node
  end
end

function optimize_config_options.entry(node)
  local cx = context.new_global_scope()
  return optimize_config_options.top(cx, node)
end

optimize_config_options.pass_name = "optimize_config_options"

return optimize_config_options
