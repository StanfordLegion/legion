-- Copyright 2018 Stanford University, NVIDIA Corporation
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
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

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
  local cx = {}
  return setmetatable(cx, context)
end

local function analyze_leaf_node(cx)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.Call) then
      return not std.is_task(node.fn.value)

    elseif node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.CrossProductArray) or
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
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AttachHDF5) or
      node:is(ast.typed.expr.DetachHDF5) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Future) or
      node:is(ast.typed.expr.FutureGetResult)
    then
      return false

    elseif node:is(ast.typed.expr.ID) or
      node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
      node:is(ast.typed.expr.ListFromElement) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Deref)
    then
      return true

    -- Statements:
    elseif node:is(ast.typed.stat.MustEpoch) or
      node:is(ast.typed.stat.IndexLaunchNum) or
      node:is(ast.typed.stat.IndexLaunchList)
    then
      return false

    elseif node:is(ast.typed.stat.If) or
      node:is(ast.typed.stat.Elseif) or
      node:is(ast.typed.stat.While) or
      node:is(ast.typed.stat.ForNum) or
      node:is(ast.typed.stat.ForList) or
      node:is(ast.typed.stat.Repeat) or
      node:is(ast.typed.stat.Block) or
      node:is(ast.typed.stat.Var) or
      node:is(ast.typed.stat.VarUnpack) or
      node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.typed.Block) or
      node:is(ast.IndexLaunchArgsProvably) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
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
    -- Expressions:
    if node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.IndexAccess)
    then
      return not std.is_ref(node.expr_type)

    elseif node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await)
    then
      return false

    elseif node:is(ast.typed.expr.ID) or
      node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Call) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.CrossProductArray) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
      node:is(ast.typed.expr.ListFromElement) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AttachHDF5) or
      node:is(ast.typed.expr.DetachHDF5) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Future) or
      node:is(ast.typed.expr.FutureGetResult)
    then
      return true

    -- Statements:
    elseif node:is(ast.typed.stat.If) or
      node:is(ast.typed.stat.Elseif) or
      node:is(ast.typed.stat.While) or
      node:is(ast.typed.stat.ForNum) or
      node:is(ast.typed.stat.ForList) or
      node:is(ast.typed.stat.Repeat) or
      node:is(ast.typed.stat.MustEpoch) or
      node:is(ast.typed.stat.Block) or
      node:is(ast.typed.stat.IndexLaunchNum) or
      node:is(ast.typed.stat.IndexLaunchList) or
      node:is(ast.typed.stat.Var) or
      node:is(ast.typed.stat.VarUnpack) or
      node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.typed.Block) or
      node:is(ast.IndexLaunchArgsProvably) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_inner(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_inner_node(cx),
    data.all,
    node, true)
end

local function analyze_idempotent_node(cx)
  return function(node)
    -- For now idempotent tasks are ones that
    -- do no external call of any kind and also
    -- do not perform any kind of file I/O
    if node:is(ast.typed.expr.Call) then
      return std.is_task(node.fn.value) or node.replicable

    elseif node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.AttachHDF5) or
      node:is(ast.typed.expr.DetachHDF5)
    then
      return false

    elseif node:is(ast.typed.expr.ID) or
      node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.CrossProductArray) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
      node:is(ast.typed.expr.ListFromElement) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Future) or
      node:is(ast.typed.expr.FutureGetResult)
    then
      return true

    -- Statements:
    elseif node:is(ast.typed.stat.If) or
      node:is(ast.typed.stat.Elseif) or
      node:is(ast.typed.stat.While) or
      node:is(ast.typed.stat.ForNum) or
      node:is(ast.typed.stat.ForList) or
      node:is(ast.typed.stat.Repeat) or
      node:is(ast.typed.stat.MustEpoch) or
      node:is(ast.typed.stat.Block) or
      node:is(ast.typed.stat.IndexLaunchNum) or
      node:is(ast.typed.stat.IndexLaunchList) or
      node:is(ast.typed.stat.Var) or
      node:is(ast.typed.stat.VarUnpack) or
      node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.typed.Block) or
      node:is(ast.IndexLaunchArgsProvably) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_idempotent(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_idempotent_node(cx),
    data.all,
    node, true)
end

local function analyze_replicable_node(cx)
  return function(node)
    -- We don't support any kind of external
    -- call for replicable which could call
    -- a random number generator, so no 
    -- non-task calls and no method calls
    -- If the replicable field is set then we
    -- know that the node was inserted by the
    -- compiler and therefore must be a legion
    -- runtime call
    if node:is(ast.typed.expr.Call) then
      return std.is_task(node.fn.value) or node.replicable
    
    elseif node:is(ast.typed.expr.MethodCall) then
      return false

    elseif node:is(ast.typed.expr.ID) or
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Ispace) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.CrossProductArray) or
      node:is(ast.typed.expr.ListSlicePartition) or
      node:is(ast.typed.expr.ListDuplicatePartition) or
      node:is(ast.typed.expr.ListSliceCrossProduct) or
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
      node:is(ast.typed.expr.ListFromElement) or
      node:is(ast.typed.expr.PhaseBarrier) or
      node:is(ast.typed.expr.DynamicCollective) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AttachHDF5) or
      node:is(ast.typed.expr.DetachHDF5) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Future) or
      node:is(ast.typed.expr.FutureGetResult)
    then
      return true

    -- Statements:
    elseif node:is(ast.typed.stat.If) or
      node:is(ast.typed.stat.Elseif) or
      node:is(ast.typed.stat.While) or
      node:is(ast.typed.stat.ForNum) or
      node:is(ast.typed.stat.ForList) or
      node:is(ast.typed.stat.Repeat) or
      node:is(ast.typed.stat.MustEpoch) or
      node:is(ast.typed.stat.Block) or
      node:is(ast.typed.stat.IndexLaunchNum) or
      node:is(ast.typed.stat.IndexLaunchList) or
      node:is(ast.typed.stat.Var) or
      node:is(ast.typed.stat.VarUnpack) or
      node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.typed.Block) or
      node:is(ast.IndexLaunchArgsProvably) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_replicable(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_replicable_node(cx),
    data.all,
    node, true)
end

local optimize_config_options = {}

function optimize_config_options.top_task(cx, node)
  if not node.body then return node end

  -- Do the analysis first and then mask it out if the configuration
  -- is disabled. This is to ensure that the analysis always works.
  local leaf = analyze_leaf(cx, node.body) and std.config["leaf"]
  local inner = analyze_inner(cx, node.body) and std.config["inner"] and not leaf
  local idempotent = analyze_idempotent(cx, node.body) and std.config["idempotent"]
  local replicable = analyze_replicable(cx, node.body) and
    idempotent and -- Replicable tasks must also be idempotent
    std.config["replicable"]

  if std.config["leaf"] and not leaf and
    node.annotations.leaf:is(ast.annotation.Demand)
  then
    report.error(node, "task is not a valid leaf task")
  end

  if std.config["inner"] and not inner and
    node.annotations.inner:is(ast.annotation.Demand)
  then
    report.error(node, "task is not a valid inner task")
  end

  if std.config["idempotent"] and not idempotent and
    node.annotations.idempotent:is(ast.annotation.Demand)
  then
    report.error(node, "task is not a valid idempotent task")
  end

  if std.config["replicable"] and not replicable and
    node.annotations.replicable:is(ast.annotation.Demand)
  then
    report.error(node, "task is not a valid replicable task")
  end

  return node {
    config_options = ast.TaskConfigOptions {
      leaf = leaf,
      inner = inner,
      idempotent = idempotent,
      replicable = replicable,
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
