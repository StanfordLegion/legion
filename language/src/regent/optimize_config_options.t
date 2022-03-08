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

-- Regent Task Config Option Optimizer
--
-- Determines which of the following config options are applicable to
-- a task:
--
--   * Leaf: Task issues no sub-operations
--   * Inner: Task does not access any regions
--   * Idempotent: Task has no external side-effects
--   * Replicable: Task's control flow can be replicated
--

local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local function always_true(node)
  return {true, node}
end

local function always_false(node)
  return {false, node}
end

local function unreachable(node)
  assert(false, "unreachable")
end

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
    leaf       = { not node.annotations.leaf:is(ast.annotation.Forbid), nil },
    inner      = { not node.annotations.inner:is(ast.annotation.Forbid), nil },
    idempotent = { not node.annotations.idempotent:is(ast.annotation.Forbid), nil },
    replicable = { not node.annotations.replicable:is(ast.annotation.Forbid), nil },
  }
  return setmetatable(cx, context)
end

function context:update(field, value)
  self[field] = all_with_provenance(self[field], value)
end

function context:analysis_done()
  return not (self.leaf[1] or self.inner[1] or self.idempotent[1] or self.replicable[1])
end

local node_is_leaf = {
  -- Expressions:
  [ast.typed.expr.Call] = function(node)
    return {not std.is_task(node.fn.value) or
              (node.fn.value.is_local and node.fn.value:get_primary_variant():get_config_options().leaf),
            node}
  end,

  [ast.typed.expr.RawContext]                 = always_false,
  [ast.typed.expr.Ispace]                     = always_false,
  [ast.typed.expr.Region]                     = always_false,
  [ast.typed.expr.Partition]                  = always_false,
  [ast.typed.expr.PartitionEqual]             = always_false,
  [ast.typed.expr.PartitionByField]           = always_false,
  [ast.typed.expr.PartitionByRestriction]     = always_false,
  [ast.typed.expr.Image]                      = always_false,
  [ast.typed.expr.ImageByTask]                = always_false,
  [ast.typed.expr.Preimage]                   = always_false,
  [ast.typed.expr.CrossProduct]               = always_false,
  [ast.typed.expr.CrossProductArray]          = always_false,
  [ast.typed.expr.ListSlicePartition]         = always_false,
  [ast.typed.expr.ListDuplicatePartition]     = always_false,
  [ast.typed.expr.ListSliceCrossProduct]      = always_false,
  [ast.typed.expr.ListCrossProduct]           = always_false,
  [ast.typed.expr.ListCrossProductComplete]   = always_false,
  [ast.typed.expr.ListPhaseBarriers]          = always_false,
  [ast.typed.expr.PhaseBarrier]               = always_false,
  [ast.typed.expr.DynamicCollective]          = always_false,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_false,
  [ast.typed.expr.Advance]                    = always_false,
  [ast.typed.expr.Adjust]                     = always_false,
  [ast.typed.expr.Arrive]                     = always_false,
  [ast.typed.expr.Await]                      = always_false,
  [ast.typed.expr.Copy]                       = always_false,
  [ast.typed.expr.Fill]                       = always_false,
  [ast.typed.expr.Acquire]                    = always_false,
  [ast.typed.expr.Release]                    = always_false,
  [ast.typed.expr.AttachHDF5]                 = always_false,
  [ast.typed.expr.DetachHDF5]                 = always_false,
  [ast.typed.expr.AllocateScratchFields]      = always_false,
  [ast.typed.expr.WithScratchFields]          = always_false,
  [ast.typed.expr.Condition]                  = always_false,
  [ast.typed.expr.Future]                     = always_false,
  [ast.typed.expr.FutureGetResult]            = always_false,
  [ast.typed.expr.ParallelizerConstraint]     = always_false,
  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]         = always_false,

  [ast.typed.expr.ID]              = always_true,
  [ast.typed.expr.Constant]        = always_true,
  [ast.typed.expr.Global]          = always_true,
  [ast.typed.expr.Function]        = always_true,
  [ast.typed.expr.FieldAccess]     = always_true,
  [ast.typed.expr.IndexAccess]     = always_true,
  [ast.typed.expr.MethodCall]      = always_true,
  [ast.typed.expr.Cast]            = always_true,
  [ast.typed.expr.Ctor]            = always_true,
  [ast.typed.expr.CtorListField]   = always_true,
  [ast.typed.expr.CtorRecField]    = always_true,
  [ast.typed.expr.RawFields]       = always_true,
  [ast.typed.expr.RawFuture]       = always_true,
  [ast.typed.expr.RawPhysical]     = always_true,
  [ast.typed.expr.RawRuntime]      = always_true,
  [ast.typed.expr.RawTask]         = always_true,
  [ast.typed.expr.RawValue]        = always_true,
  [ast.typed.expr.Isnull]          = always_true,
  [ast.typed.expr.Null]            = always_true,
  [ast.typed.expr.DynamicCast]     = always_true,
  [ast.typed.expr.StaticCast]      = always_true,
  [ast.typed.expr.UnsafeCast]      = always_true,
  [ast.typed.expr.ListInvert]      = always_true,
  [ast.typed.expr.ListRange]       = always_true,
  [ast.typed.expr.ListIspace]      = always_true,
  [ast.typed.expr.ListFromElement] = always_true,
  [ast.typed.expr.RegionRoot]      = always_true,
  [ast.typed.expr.Unary]           = always_true,
  [ast.typed.expr.Binary]          = always_true,
  [ast.typed.expr.Deref]           = always_true,
  [ast.typed.expr.AddressOf]       = always_true,
  [ast.typed.expr.Projection]      = always_true,

  [ast.typed.expr.Internal]        = unreachable,

  -- Statements:
  [ast.typed.stat.MustEpoch]       = always_false,
  [ast.typed.stat.IndexLaunchNum]  = always_false,
  [ast.typed.stat.IndexLaunchList] = always_false,
  [ast.typed.stat.RawDelete]       = always_false,
  [ast.typed.stat.Fence]           = always_false,
  [ast.typed.stat.ParallelizeWith] = always_false,

  [ast.typed.stat.If]             = always_true,
  [ast.typed.stat.Elseif]         = always_true,
  [ast.typed.stat.While]          = always_true,
  [ast.typed.stat.ForNum]         = always_true,
  [ast.typed.stat.ForList]        = always_true,
  [ast.typed.stat.Repeat]         = always_true,
  [ast.typed.stat.Block]          = always_true,
  [ast.typed.stat.Var]            = always_true,
  [ast.typed.stat.VarUnpack]      = always_true,
  [ast.typed.stat.Return]         = always_true,
  [ast.typed.stat.Break]          = always_true,
  [ast.typed.stat.Assignment]     = always_true,
  [ast.typed.stat.Reduce]         = always_true,
  [ast.typed.stat.Expr]           = always_true,
  [ast.typed.stat.ParallelPrefix] = always_true,

  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local analyze_leaf_node = ast.make_single_dispatch(
  node_is_leaf,
  {ast.typed.expr, ast.typed.stat},
  always_true)

local function analyze_leaf(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_leaf_node(),
    all_with_provenance,
    node, {true})
end

local node_is_inner = {
  -- Expressions:
  [ast.typed.expr.Deref] = function(node)
    return {not std.is_ref(node.expr_type), node}
  end,
  [ast.typed.expr.IndexAccess] = function(node)
    return {not std.is_ref(node.expr_type), node}
  end,

  [ast.typed.expr.Call] = function(node)
    return {not std.is_task(node.fn.value) or
              not node.fn.value.is_local or
              node.fn.value:get_primary_variant():get_config_options().inner,
            node}
  end,

  [ast.typed.expr.RawPhysical] = always_false,
  [ast.typed.expr.Adjust]      = always_false,
  [ast.typed.expr.Arrive]      = always_false,
  [ast.typed.expr.Await]       = always_false,

  [ast.typed.expr.ID]                         = always_true,
  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.MethodCall]                 = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawContext]                 = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawRuntime]                 = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Region]                     = always_true,
  [ast.typed.expr.Partition]                  = always_true,
  [ast.typed.expr.PartitionEqual]             = always_true,
  [ast.typed.expr.PartitionByField]           = always_true,
  [ast.typed.expr.PartitionByRestriction]     = always_true,
  [ast.typed.expr.Image]                      = always_true,
  [ast.typed.expr.ImageByTask]                = always_true,
  [ast.typed.expr.Preimage]                   = always_true,
  [ast.typed.expr.CrossProduct]               = always_true,
  [ast.typed.expr.CrossProductArray]          = always_true,
  [ast.typed.expr.ListSlicePartition]         = always_true,
  [ast.typed.expr.ListDuplicatePartition]     = always_true,
  [ast.typed.expr.ListSliceCrossProduct]      = always_true,
  [ast.typed.expr.ListCrossProduct]           = always_true,
  [ast.typed.expr.ListCrossProductComplete]   = always_true,
  [ast.typed.expr.ListPhaseBarriers]          = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.PhaseBarrier]               = always_true,
  [ast.typed.expr.DynamicCollective]          = always_true,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_true,
  [ast.typed.expr.Advance]                    = always_true,
  [ast.typed.expr.Copy]                       = always_true,
  [ast.typed.expr.Fill]                       = always_true,
  [ast.typed.expr.Acquire]                    = always_true,
  [ast.typed.expr.Release]                    = always_true,
  [ast.typed.expr.AttachHDF5]                 = always_true,
  [ast.typed.expr.DetachHDF5]                 = always_true,
  [ast.typed.expr.AllocateScratchFields]      = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Condition]                  = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.Future]                     = always_true,
  [ast.typed.expr.FutureGetResult]            = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,
  [ast.typed.expr.ImportIspace]               = always_true,
  [ast.typed.expr.ImportRegion]               = always_true,
  [ast.typed.expr.ImportPartition]            = always_true,
  [ast.typed.expr.ImportCrossProduct]         = always_true,
  [ast.typed.expr.Projection]                 = always_true,

  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.If]              = always_true,
  [ast.typed.stat.Elseif]          = always_true,
  [ast.typed.stat.While]           = always_true,
  [ast.typed.stat.ForNum]          = always_true,
  [ast.typed.stat.ForList]         = always_true,
  [ast.typed.stat.Repeat]          = always_true,
  [ast.typed.stat.MustEpoch]       = always_true,
  [ast.typed.stat.Block]           = always_true,
  [ast.typed.stat.IndexLaunchNum]  = always_true,
  [ast.typed.stat.IndexLaunchList] = always_true,
  [ast.typed.stat.Var]             = always_true,
  [ast.typed.stat.VarUnpack]       = always_true,
  [ast.typed.stat.Return]          = always_true,
  [ast.typed.stat.Break]           = always_true,
  [ast.typed.stat.Assignment]      = always_true,
  [ast.typed.stat.Reduce]          = always_true,
  [ast.typed.stat.Expr]            = always_true,
  [ast.typed.stat.RawDelete]       = always_true,
  [ast.typed.stat.Fence]           = always_true,
  [ast.typed.stat.ParallelizeWith] = always_true,

  [ast.typed.stat.ParallelPrefix]  = always_false,

  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local analyze_inner_node = ast.make_single_dispatch(
  node_is_inner,
  {ast.typed.expr, ast.typed.stat},
  always_true)

local function analyze_inner(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_inner_node(),
    all_with_provenance,
    node, {true})
end

local node_is_idempotent = {
  -- Expressions:

  -- For now idempotent tasks are ones that
  -- do no external call of any kind and also
  -- do not perform any kind of file I/O
  [ast.typed.expr.Call] = function(node)
    return {(std.is_task(node.fn.value) and
               (not node.fn.value.is_local or node.fn.value:get_primary_variant():get_config_options().idempotent)) or
              std.is_math_fn(node.fn.value) or
              node.replicable,
            node}
  end,

  [ast.typed.expr.MethodCall] = always_false,
  [ast.typed.expr.Adjust]     = always_false,
  [ast.typed.expr.Arrive]     = always_false,
  [ast.typed.expr.Await]      = always_false,
  [ast.typed.expr.AttachHDF5] = always_false,
  [ast.typed.expr.DetachHDF5] = always_false,

  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]         = always_false,

  [ast.typed.expr.ID]                         = always_true,
  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Deref]                      = always_true,
  [ast.typed.expr.IndexAccess]                = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawContext]                 = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawPhysical]                = always_true,
  [ast.typed.expr.RawRuntime]                 = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Region]                     = always_true,
  [ast.typed.expr.Partition]                  = always_true,
  [ast.typed.expr.PartitionEqual]             = always_true,
  [ast.typed.expr.PartitionByField]           = always_true,
  [ast.typed.expr.PartitionByRestriction]     = always_true,
  [ast.typed.expr.Image]                      = always_true,
  [ast.typed.expr.ImageByTask]                = always_true,
  [ast.typed.expr.Preimage]                   = always_true,
  [ast.typed.expr.CrossProduct]               = always_true,
  [ast.typed.expr.CrossProductArray]          = always_true,
  [ast.typed.expr.ListSlicePartition]         = always_true,
  [ast.typed.expr.ListDuplicatePartition]     = always_true,
  [ast.typed.expr.ListSliceCrossProduct]      = always_true,
  [ast.typed.expr.ListCrossProduct]           = always_true,
  [ast.typed.expr.ListCrossProductComplete]   = always_true,
  [ast.typed.expr.ListPhaseBarriers]          = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.PhaseBarrier]               = always_true,
  [ast.typed.expr.DynamicCollective]          = always_true,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_true,
  [ast.typed.expr.Advance]                    = always_true,
  [ast.typed.expr.Copy]                       = always_true,
  [ast.typed.expr.Fill]                       = always_true,
  [ast.typed.expr.Acquire]                    = always_true,
  [ast.typed.expr.Release]                    = always_true,
  [ast.typed.expr.AllocateScratchFields]      = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Condition]                  = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.Future]                     = always_true,
  [ast.typed.expr.FutureGetResult]            = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,
  [ast.typed.expr.Projection]                 = always_true,

  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.If]              = always_true,
  [ast.typed.stat.Elseif]          = always_true,
  [ast.typed.stat.While]           = always_true,
  [ast.typed.stat.ForNum]          = always_true,
  [ast.typed.stat.ForList]         = always_true,
  [ast.typed.stat.Repeat]          = always_true,
  [ast.typed.stat.MustEpoch]       = always_true,
  [ast.typed.stat.Block]           = always_true,
  [ast.typed.stat.IndexLaunchNum]  = always_true,
  [ast.typed.stat.IndexLaunchList] = always_true,
  [ast.typed.stat.Var]             = always_true,
  [ast.typed.stat.VarUnpack]       = always_true,
  [ast.typed.stat.Return]          = always_true,
  [ast.typed.stat.Break]           = always_true,
  [ast.typed.stat.Assignment]      = always_true,
  [ast.typed.stat.Reduce]          = always_true,
  [ast.typed.stat.Expr]            = always_true,
  [ast.typed.stat.RawDelete]       = always_true,
  [ast.typed.stat.Fence]           = always_true,
  [ast.typed.stat.ParallelizeWith] = always_true,
  [ast.typed.stat.ParallelPrefix]  = always_true,

  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local analyze_idempotent_node = ast.make_single_dispatch(
  node_is_idempotent,
  {ast.typed.expr, ast.typed.stat},
  always_true)

local function analyze_idempotent(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_idempotent_node(),
    all_with_provenance,
    node, {true})
end

local node_is_replicable = {
  -- Expressions:

  -- A function call is ok if it is:
  --
  --   * A task call
  --   * Marked as "replicable" by the compiler (such calls are
  --     inserted by compiler passes and are known to be safe)
  --   * Contained in the whitelist of known ok C functions (so no
  --     "rand", etc.)
  --   * The user promises the function is replicable by marking the
  --     field "replicable"
  --
  -- General C and Terra calls are not supported because there is no
  -- way to know in general if they do something non-deterministic.
  [ast.typed.expr.Call] = function(node)
    local is_user_replicable = type(node.fn.value) == "table" and rawget(node.fn.value, "replicable")
    return {(std.is_task(node.fn.value) and
              (not node.fn.value.is_local or node.fn.value:get_primary_variant():get_config_options().replicable)) or
              std.is_math_fn(node.fn.value) or
              is_user_replicable or
              node.replicable or
              std.replicable_whitelist[node.fn.value] or false,
            node}
  end,

  [ast.typed.expr.MethodCall] = always_false,
  [ast.typed.expr.Adjust]     = always_false,
  [ast.typed.expr.Arrive]     = always_false,
  [ast.typed.expr.Await]      = always_false,
  [ast.typed.expr.Advance]    = always_false,

  [ast.typed.expr.ID]                         = always_true,
  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Deref]                      = always_true,
  [ast.typed.expr.IndexAccess]                = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawContext]                 = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawPhysical]                = always_true,
  [ast.typed.expr.RawRuntime]                 = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Ispace]                     = always_true,
  [ast.typed.expr.Region]                     = always_true,
  [ast.typed.expr.Partition]                  = always_true,
  [ast.typed.expr.PartitionEqual]             = always_true,
  [ast.typed.expr.PartitionByField]           = always_true,
  [ast.typed.expr.PartitionByRestriction]     = always_true,
  [ast.typed.expr.Image]                      = always_true,
  [ast.typed.expr.ImageByTask]                = always_true,
  [ast.typed.expr.Preimage]                   = always_true,
  [ast.typed.expr.CrossProduct]               = always_true,
  [ast.typed.expr.CrossProductArray]          = always_true,
  [ast.typed.expr.ListSlicePartition]         = always_true,
  [ast.typed.expr.ListDuplicatePartition]     = always_true,
  [ast.typed.expr.ListSliceCrossProduct]      = always_true,
  [ast.typed.expr.ListCrossProduct]           = always_true,
  [ast.typed.expr.ListCrossProductComplete]   = always_true,
  [ast.typed.expr.ListPhaseBarriers]          = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.PhaseBarrier]               = always_true,
  [ast.typed.expr.DynamicCollective]          = always_true,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_true,
  [ast.typed.expr.Copy]                       = always_true,
  [ast.typed.expr.Fill]                       = always_true,
  [ast.typed.expr.Acquire]                    = always_true,
  [ast.typed.expr.Release]                    = always_true,
  [ast.typed.expr.AttachHDF5]                 = always_true,
  [ast.typed.expr.DetachHDF5]                 = always_true,
  [ast.typed.expr.AllocateScratchFields]      = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Condition]                  = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.Future]                     = always_true,
  [ast.typed.expr.FutureGetResult]            = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,
  [ast.typed.expr.ImportIspace]               = always_true,
  [ast.typed.expr.ImportRegion]               = always_true,
  [ast.typed.expr.ImportPartition]            = always_true,
  [ast.typed.expr.ImportCrossProduct]         = always_true,
  [ast.typed.expr.Projection]                 = always_true,

  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.If] = always_true,
  [ast.typed.stat.Elseif] = always_true,
  [ast.typed.stat.While] = always_true,
  [ast.typed.stat.ForNum] = always_true,
  [ast.typed.stat.ForList] = always_true,
  [ast.typed.stat.Repeat] = always_true,
  [ast.typed.stat.MustEpoch] = always_true,
  [ast.typed.stat.Block] = always_true,
  [ast.typed.stat.IndexLaunchNum] = always_true,
  [ast.typed.stat.IndexLaunchList] = always_true,
  [ast.typed.stat.Var] = always_true,
  [ast.typed.stat.VarUnpack] = always_true,
  [ast.typed.stat.Return] = always_true,
  [ast.typed.stat.Break] = always_true,
  [ast.typed.stat.Assignment] = always_true,
  [ast.typed.stat.Reduce] = always_true,
  [ast.typed.stat.Expr] = always_true,
  [ast.typed.stat.RawDelete] = always_true,
  [ast.typed.stat.Fence] = always_true,
  [ast.typed.stat.ParallelizeWith] = always_true,
  [ast.typed.stat.ParallelPrefix] = always_true,

  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
}

local analyze_replicable_node = ast.make_single_dispatch(
  node_is_replicable,
  {ast.typed.expr, ast.typed.stat},
  always_true)

local function analyze_replicable(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_replicable_node(),
    all_with_provenance,
    node, {true})
end

local function analyze_all_node(cx)
  local analyze_leaf_node = analyze_leaf_node()
  local analyze_inner_node = analyze_inner_node()
  local analyze_idempotent_node = analyze_idempotent_node()
  local analyze_replicable_node = analyze_replicable_node()
  return function(node, continuation)
    if cx.leaf[1] then cx:update("leaf", analyze_leaf_node(node)) end
    if cx.inner[1] then cx:update("inner", analyze_inner_node(node)) end
    if cx.idempotent[1] then cx:update("idempotent", analyze_idempotent_node(node)) end
    if cx.replicable[1] then cx:update("replicable", analyze_replicable_node(node)) end
    if not cx:analysis_done() then continuation(node, true) end
  end
end

local function analyze_all(cx, node)
  return ast.traverse_node_continuation(
    analyze_all_node(cx),
    node)
end

local optimize_config_options = {}

function optimize_config_options.top_task(cx, node)
  if not node.body then return node end

  -- Do the analysis first and then mask it out if the configuration
  -- is disabled. This is to ensure that the analysis always works.
  analyze_all(cx, node.body)

  local leaf, leaf_node = unpack(cx.leaf)
  local inner, inner_node = unpack(cx.inner)
  local idempotent, idempotent_node = unpack(cx.idempotent)
  local replicable, replicable_node = unpack(cx.replicable)

  leaf = leaf and not node.annotations.leaf:is(ast.annotation.Forbid) and std.config["leaf"] and not node.annotations.inner:is(ast.annotation.Demand)
  inner = inner and not node.annotations.inner:is(ast.annotation.Forbid) and std.config["inner"] and not leaf
  idempotent = idempotent and not node.annotations.idempotent:is(ast.annotation.Forbid) and std.config["idempotent"]
  replicable = replicable and not node.annotations.replicable:is(ast.annotation.Forbid) and std.config["replicable"]

  if std.config["leaf"] and not leaf and
    node.annotations.leaf:is(ast.annotation.Demand)
  then
    report.error(leaf_node, "task is not a valid leaf task")
  end

  if std.config["inner"] and not inner and
    node.annotations.inner:is(ast.annotation.Demand)
  then
    report.error(inner_node or node, "task is not a valid inner task")
  end

  if std.config["idempotent"] and not idempotent and
    node.annotations.idempotent:is(ast.annotation.Demand)
  then
    report.error(idempotent_node, "task is not a valid idempotent task")
  end

  if std.config["replicable"] and not replicable and
    node.annotations.replicable:is(ast.annotation.Demand)
  then
    report.error(replicable_node, "task is not a valid replicable task")
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
  local cx = context.new_global_scope(node)
  return optimize_config_options.top(cx, node)
end

optimize_config_options.pass_name = "optimize_config_options"

return optimize_config_options
