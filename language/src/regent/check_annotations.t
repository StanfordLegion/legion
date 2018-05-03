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

-- Legion Option Checker

-- This pass implements a simple syntactic option checker. The purpose
-- of this checker is not to determine which options can be applied
-- (this can only be done in the individual optimization passes, in
-- general), but to reject options that are obviously syntactically
-- invalid, such as:
--
-- __demand(__inline)
-- var x = 0
--
-- By doing this here, we can avoid a lot of boilerplate code in the
-- individual optimization passes that would otherwise need to check
-- these properties.

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

local check_annotations = {}

local function render_option(option, value)
  local value_name = string.lower(
    string.gsub(tostring(value:type()), "[A-Za-z0-9_]+[.]", ""))
  return "__" .. value_name .. "(__" .. tostring(option) .. ")"
end

local function check(node, allowed_set)
  -- Sanity check the allowed set.
  for option, _ in pairs(allowed_set) do
    assert(node.annotations[option]:is(ast.annotation))
  end

  -- Check that only options in the allowed_set are enabled.
  for option, value in pairs(node.annotations) do
    if ast.is_node(value) and not value:is(ast.annotation.Allow) and
      not allowed_set[option]
    then
      report.error(node, "option " .. render_option(option, value) ..
                  " is not permitted")
    end
  end
end

local function allow(allowed_list)
  local allowed_set = data.set(allowed_list)
  return function(node)
    check(node, allowed_set)
  end
end
local deny_all = allow({})

local function pass(node)
end

local function unreachable(node)
  assert(false, "unreachable")
end

local permitted_for_num_annotations = terralib.newlist({"parallel", "spmd", "trace"})
if std.config["vectorize-unsafe"] then
  permitted_for_num_annotations:insert("vectorize")
end

local node_allow_annotations = {
  -- Expressions:
  [ast.typed.expr.Call] = allow({"parallel", "inline"}),

  [ast.typed.expr.ID]                         = deny_all,
  [ast.typed.expr.Constant]                   = deny_all,
  [ast.typed.expr.Function]                   = deny_all,
  [ast.typed.expr.FieldAccess]                = deny_all,
  [ast.typed.expr.IndexAccess]                = deny_all,
  [ast.typed.expr.MethodCall]                 = deny_all,
  [ast.typed.expr.Cast]                       = deny_all,
  [ast.typed.expr.Ctor]                       = deny_all,
  [ast.typed.expr.CtorListField]              = deny_all,
  [ast.typed.expr.CtorRecField]               = deny_all,
  [ast.typed.expr.RawContext]                 = deny_all,
  [ast.typed.expr.RawFields]                  = deny_all,
  [ast.typed.expr.RawPhysical]                = deny_all,
  [ast.typed.expr.RawRuntime]                 = deny_all,
  [ast.typed.expr.RawValue]                   = deny_all,
  [ast.typed.expr.Isnull]                     = deny_all,
  [ast.typed.expr.Null]                       = deny_all,
  [ast.typed.expr.DynamicCast]                = deny_all,
  [ast.typed.expr.StaticCast]                 = deny_all,
  [ast.typed.expr.UnsafeCast]                 = deny_all,
  [ast.typed.expr.Ispace]                     = deny_all,
  [ast.typed.expr.Region]                     = deny_all,
  [ast.typed.expr.Partition]                  = deny_all,
  [ast.typed.expr.PartitionEqual]             = deny_all,
  [ast.typed.expr.PartitionByField]           = deny_all,
  [ast.typed.expr.Image]                      = deny_all,
  [ast.typed.expr.ImageByTask]                = deny_all,
  [ast.typed.expr.Preimage]                   = deny_all,
  [ast.typed.expr.CrossProduct]               = deny_all,
  [ast.typed.expr.CrossProductArray]          = deny_all,
  [ast.typed.expr.ListSlicePartition]         = deny_all,
  [ast.typed.expr.ListDuplicatePartition]     = deny_all,
  [ast.typed.expr.ListSliceCrossProduct]      = deny_all,
  [ast.typed.expr.ListCrossProduct]           = deny_all,
  [ast.typed.expr.ListCrossProductComplete]   = deny_all,
  [ast.typed.expr.ListPhaseBarriers]          = deny_all,
  [ast.typed.expr.ListInvert]                 = deny_all,
  [ast.typed.expr.ListRange]                  = deny_all,
  [ast.typed.expr.ListIspace]                 = deny_all,
  [ast.typed.expr.ListFromElement]            = deny_all,
  [ast.typed.expr.PhaseBarrier]               = deny_all,
  [ast.typed.expr.DynamicCollective]          = deny_all,
  [ast.typed.expr.DynamicCollectiveGetResult] = deny_all,
  [ast.typed.expr.Advance]                    = deny_all,
  [ast.typed.expr.Adjust]                     = deny_all,
  [ast.typed.expr.Arrive]                     = deny_all,
  [ast.typed.expr.Await]                      = deny_all,
  [ast.typed.expr.Copy]                       = deny_all,
  [ast.typed.expr.Fill]                       = deny_all,
  [ast.typed.expr.Acquire]                    = deny_all,
  [ast.typed.expr.Release]                    = deny_all,
  [ast.typed.expr.AttachHDF5]                 = deny_all,
  [ast.typed.expr.DetachHDF5]                 = deny_all,
  [ast.typed.expr.AllocateScratchFields]      = deny_all,
  [ast.typed.expr.WithScratchFields]          = deny_all,
  [ast.typed.expr.RegionRoot]                 = deny_all,
  [ast.typed.expr.Condition]                  = deny_all,
  [ast.typed.expr.Unary]                      = deny_all,
  [ast.typed.expr.Binary]                     = deny_all,
  [ast.typed.expr.Deref]                      = deny_all,
  [ast.typed.expr.ParallelizerConstraint]     = deny_all,

  [ast.typed.expr.Internal]                   = unreachable,
  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,

  -- Statements:
  [ast.typed.stat.If]        = deny_all,
  [ast.typed.stat.Elseif]    = deny_all,
  [ast.typed.stat.While]     = allow({"spmd", "trace"}),
  [ast.typed.stat.ForNum]    = allow(permitted_for_num_annotations),
  [ast.typed.stat.ForList]   = allow({"openmp", "parallel", "spmd", "trace", "vectorize"}),
  [ast.typed.stat.Repeat]    = allow({"spmd", "trace"}),
  [ast.typed.stat.MustEpoch] = deny_all,
  [ast.typed.stat.Block]     = allow({"spmd", "trace"}),

  [ast.typed.stat.Var]             = deny_all,
  [ast.typed.stat.VarUnpack]       = deny_all,
  [ast.typed.stat.Return]          = deny_all,
  [ast.typed.stat.Break]           = deny_all,
  [ast.typed.stat.Assignment]      = deny_all,
  [ast.typed.stat.Reduce]          = deny_all,
  [ast.typed.stat.Expr]            = deny_all,
  [ast.typed.stat.RawDelete]       = deny_all,
  [ast.typed.stat.Fence]           = deny_all,
  [ast.typed.stat.ParallelizeWith] = deny_all,

  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.IndexLaunchNum]    = unreachable,
  [ast.typed.stat.IndexLaunchList]   = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,

  -- Tasks:
  [ast.typed.top.TaskParam] = deny_all,

  [ast.typed.top.Task] = allow({
    "cuda",
    "external",
    "inline",
    "inner",
    "leaf",
    "optimize",
    "parallel",
  }),

  [ast.typed.top.Fspace] = deny_all,

  -- Specialized ASTs:
  [ast.specialized.region] = pass,
  [ast.specialized.expr] = pass,
  [ast.specialized.stat] = pass,
  [ast.specialized.Block] = pass,
  [ast.specialized.top.QuoteExpr] = deny_all,
  [ast.specialized.top.QuoteStat] = deny_all,

  -- Miscellaneous:
  [ast.typed.Block] = pass,
  [ast.location] = pass,
  [ast.annotation] = pass,
  [ast.constraint_kind] = pass,
  [ast.privilege_kind] = pass,
  [ast.condition_kind] = pass,
  [ast.disjointness_kind] = pass,
  [ast.fence_kind] = pass,
  [ast.constraint] = pass,
  [ast.privilege] = pass,
  [ast.TaskConfigOptions] = pass,
}

local check_annotations_node = ast.make_single_dispatch(
  node_allow_annotations,
  {ast.typed})

function check_annotations.top(cx, node)
  ast.traverse_node_postorder(check_annotations_node(), node)
end

function check_annotations.entry(node)
  local cx = context.new_global_scope()
  check_annotations.top(cx, node)
  return node
end

return check_annotations
