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

-- Regent Predicated Execution Optimizer
--
-- Attempts to determine which branches can be transformed into
-- predicated execution.

local affine_helper = require("regent/affine_helper")
local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")
local task_helper = require("regent/task_helper")

local optimize_predicate = {}

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

function context:new_local_scope(cond)
  local cx = {
    cond = cond
  }
  return setmetatable(cx, context)
end

function context:new_task_scope()
  local cx = {
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local function always_true(cx, node)
  return {true, node}
end

local function always_false(cx, node)
  return {false, node}
end

local function unreachable(cx, node)
  assert(false, "unreachable")
end

local function all_with_provenance(acc, value)
  if not value[1] then
    return value
  end
  return acc
end

-- IMPORTANT: This analysis is DIFFERENT than the analysis by the same
-- name for index launches.
local node_is_side_effect_free = {
  -- Expressions:
  [ast.typed.expr.IndexAccess] = function(cx, node)
    return {not std.is_ref(node.expr_type), node}
  end,

  [ast.typed.expr.Call] = function(cx, node)
    -- IMPORTANT: Tasks are ok (we can predicate them), but non-tasks
    -- (or local tasks) are not (we have no way to mitigate or analyze
    -- potential side-effects).
    return {std.is_task(node.fn.value) and not node.fn.value.is_local, node}
  end,

  [ast.typed.expr.MethodCall]                 = always_false,
  [ast.typed.expr.RawContext]                 = always_false,
  [ast.typed.expr.RawPhysical]                = always_false,
  [ast.typed.expr.RawRuntime]                 = always_false,
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
  [ast.typed.expr.Condition]                  = always_false,
  [ast.typed.expr.Deref]                      = always_false,
  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]            = always_false,

  [ast.typed.expr.ID]                         = always_true,
  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_true,
  [ast.typed.expr.Advance]                    = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.Future]                     = always_true,
  [ast.typed.expr.FutureGetResult]            = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,
  [ast.typed.expr.Projection]                 = always_true,

  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.IndexLaunchList] = function(cx, node)
    return {not node.reduce_op, node}
  end,
  [ast.typed.stat.IndexLaunchNum] = function(cx, node)
    return {not node.reduce_op, node}
  end,

  -- Currently we can only support unpredicated conditionals inside of
  -- a predicated statement.
  [ast.typed.stat.If] = function(cx, node)
    return {not node.cond:is(ast.typed.expr.FutureGetResult), node}
  end,
  [ast.typed.stat.While] = function(cx, node)
    return {not node.cond:is(ast.typed.expr.FutureGetResult), node}
  end,

  [ast.typed.stat.Elseif]                     = always_false,
  [ast.typed.stat.MustEpoch]                  = always_false,
  [ast.typed.stat.Return]                     = always_false,
  [ast.typed.stat.Break]                      = always_false,
  [ast.typed.stat.Reduce]                     = always_false,
  [ast.typed.stat.RawDelete]                  = always_false,
  [ast.typed.stat.Fence]                      = always_false,
  [ast.typed.stat.ParallelizeWith]            = always_false,
  [ast.typed.stat.ParallelPrefix]             = always_false,
  [ast.typed.stat.BeginTrace]                 = always_false,
  [ast.typed.stat.EndTrace]                   = always_false,

  [ast.typed.stat.ForNum]                     = always_true,
  [ast.typed.stat.ForList]                    = always_true,
  [ast.typed.stat.Repeat]                     = always_true,
  [ast.typed.stat.Block]                      = always_true,
  [ast.typed.stat.Var]                        = always_true,
  [ast.typed.stat.VarUnpack]                  = always_true,
  [ast.typed.stat.Assignment]                 = always_true,
  [ast.typed.stat.Expr]                       = always_true,

  -- TODO: unimplemented

  [ast.typed.stat.Internal]                   = unreachable,
  [ast.typed.stat.ForNumVectorized]           = unreachable,
  [ast.typed.stat.ForListVectorized]          = unreachable,
  [ast.typed.stat.MapRegions]                 = unreachable,
  [ast.typed.stat.UnmapRegions]               = unreachable,
}

local analyze_is_side_effect_free_node = ast.make_single_dispatch(
  node_is_side_effect_free,
  {ast.typed.expr, ast.typed.stat})

local function analyze_is_side_effect_free(cx, node)
  return ast.mapreduce_expr_stat_postorder(
    analyze_is_side_effect_free_node(cx),
    all_with_provenance,
    node, {true})
end

local function predicate_call(cx, node)
  return node {
    predicate = cx.cond,
  }
end

local function predicate_index_launch(cx, node)
  return node {
    call = node.call {
      predicate = cx.cond,
    },
  }
end

local function predicate_assignment(cx, node)
  local lhs_type = std.as_read(node.lhs.expr_type) -- FIXME: this is a write??
  local rhs_type = std.as_read(node.rhs.expr_type)

  -- This is an assignment to a variable defined in the predication
  -- scope, don't need to predicate it.
  if not std.is_future(lhs_type) then
    return node
  end

  assert(std.is_future(rhs_type))

  -- If there's an existing call, reuse it.
  if node.rhs:is(ast.typed.expr.Call) then
    return node {
      rhs = node.rhs {
        predicate_else_value = node.lhs,
      }
    }
  end

  -- Otherwise create a dummy task and run the result through it.
  local value_type = rhs_type.result_type
  local identity_task = task_helper.make_identity_task(value_type)
  return node {
    rhs = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = identity_task,
        expr_type = identity_task:get_type(),
        annotations = ast.default_annotations(),
        span = node.span,
      },
      args = terralib.newlist({
        node.rhs,
      }),
      conditions = terralib.newlist(),
      predicate = cx.cond,
      predicate_else_value = node.lhs,
      replicable = false,
      expr_type = rhs_type,
      annotations = ast.default_annotations(),
      span = node.span,
    },
  }
end

local function do_nothing(cx, node) return node end

local predicate_node_table = {
  [ast.typed.expr.Call]            = predicate_call,
  [ast.typed.expr]                 = do_nothing,

  [ast.typed.stat.IndexLaunchNum]  = predicate_index_launch,
  [ast.typed.stat.IndexLaunchList] = predicate_index_launch,
  [ast.typed.stat.Assignment]      = predicate_assignment,
  [ast.typed.stat]                 = do_nothing,
}

local predicate_node = ast.make_single_dispatch(
  predicate_node_table, {})

local function predicate_block(cx, node)
  return ast.map_expr_stat_postorder(predicate_node(cx), node)
end

function optimize_predicate.stat_if(cx, node)
  local report_fail = report.info
  if node.annotations.predicate:is(ast.annotation.Demand) then
    report_fail = report.error
  end

  if node.annotations.predicate:is(ast.annotation.Forbid) then
    return node
  end

  if not node.cond:is(ast.typed.expr.FutureGetResult) then
    report_fail(node, "cannot predicate if statement: condition is not a future")
    return node
  end

  assert(#node.elseif_blocks == 0) -- should be handled by normalizer

  if #node.else_block.stats > 0 then
    report_fail(node, "cannot predicate if statement: contains else block")
    return node
  end

  local result, result_node = unpack(analyze_is_side_effect_free(cx, node.then_block))
  if not result then
    report_fail(result_node, "cannot predicate if statement: body is not side-effect free")
    return node
  end

  -- After optimize_futures cond is always wrapped in FutureGetResult
  assert(node.cond:is(ast.typed.expr.FutureGetResult))
  local cond = node.cond.value

  assert(cond:is(ast.typed.expr.ID)) -- should be handled by normalizer

  local then_cx = cx:new_local_scope(cond)

  return ast.typed.stat.Block {
    block = predicate_block(then_cx, node.then_block),
    annotations = ast.default_annotations(),
    span = node.span,
  }
end

function optimize_predicate.stat_while(cx, node)
  local report_fail = report.info
  if node.annotations.predicate:is(ast.annotation.Demand) then
    report_fail = report.error
  end

  if node.annotations.predicate:is(ast.annotation.Forbid) then
    return node
  end

  if not node.cond:is(ast.typed.expr.FutureGetResult) then
    report_fail(node, "cannot predicate while loop: condition is not a future")
    return node
  end

  local result, result_node = unpack(analyze_is_side_effect_free(cx, node.block))
  if not result then
    report_fail(result_node, "cannot predicate while loop: body is not side-effect free")
    return node
  end

  -- After optimize_futures cond is always wrapped in FutureGetResult
  assert(node.cond:is(ast.typed.expr.FutureGetResult))
  local cond = node.cond.value
  local cond_type = cond.value:gettype()

  assert(cond:is(ast.typed.expr.ID)) -- should be handled by normalizer

  local body_cx = cx:new_local_scope(cond)
  local body = predicate_block(body_cx, node.block)

  local conds = terralib.newlist()
  conds:insert(cond.value)
  conds:insertall(
    data.range(0, std.config["predicate-unroll"]):map(
      function(i)
        return std.newsymbol(cond_type, "__predicate_cond_unroll_" .. tostring(i))
      end))

  local setup = terralib.newlist()
  for i = 2, #conds do
    setup:insert(
      ast.typed.stat.Var {
        symbol = conds[i],
        type = cond_type,
        value = cond,
        annotations = ast.default_annotations(),
        span = node.span,
      }
    )
  end

  local update = terralib.newlist()
  for i = #conds, 2, -1 do
    update:insert(
      ast.typed.stat.Assignment {
        lhs = ast.typed.expr.ID {
          value = conds[i],
          expr_type = std.rawref(&cond_type),
          annotations = ast.default_annotations(),
          span = node.span,
        },
        rhs = ast.typed.expr.ID {
          value = conds[i-1],
          expr_type = std.rawref(&cond_type),
          annotations = ast.default_annotations(),
          span = node.span,
        },
        metadata = false,
        annotations = ast.default_annotations(),
        span = node.span,
      })
  end

  local new_body_block = terralib.newlist()
  new_body_block:insertall(body.stats)
  new_body_block:insertall(update)

  local new_block = terralib.newlist()
  new_block:insertall(setup)
  new_block:insert(
    ast.typed.stat.While {
      cond = ast.typed.expr.FutureGetResult {
        value = ast.typed.expr.ID {
          value = conds[#conds],
          expr_type = std.rawref(&cond_type),
          annotations = ast.default_annotations(),
          span = node.span,
        },
        expr_type = cond_type.result_type,
        annotations = ast.default_annotations(),
        span = node.span,
      },
      block = ast.typed.Block {
        stats = new_body_block,
        span = node.span,
      },
      annotations = ast.default_annotations(),
      span = node.span,
    })

  return ast.typed.stat.Block {
    block = ast.typed.Block {
      stats = new_block,
      span = node.span,
    },
    annotations = ast.default_annotations(),
    span = node.span,
  }
end

local optimize_predicate_stat_table = {
  [ast.typed.stat.If]     = optimize_predicate.stat_if,
  [ast.typed.stat.While]  = optimize_predicate.stat_while,
  [ast.typed.stat]        = do_nothing
}

local optimize_predicate_stat = ast.make_single_dispatch(
  optimize_predicate_stat_table, {})

function optimize_predicate.stat(cx, node)
  return optimize_predicate_stat(cx)(node)
end

function optimize_predicate.block(cx, node)
  return node {
    stats = node.stats:map(function(stat)
      return optimize_predicate.stat(cx, stat)
    end)
  }
end

function optimize_predicate.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope()
  local body = optimize_predicate.block(cx, node.body)

  return node { body = body }
end

function optimize_predicate.top(cx, node)
  if node:is(ast.typed.top.Task) and
     not node.config_options.leaf
  then
    return optimize_predicate.top_task(cx, node)

  else
    return node
  end
end

function optimize_predicate.entry(node)
  local cx = context.new_global_scope({})
  return optimize_predicate.top(cx, node)
end

optimize_predicate.pass_name = "optimize_predicate"

return optimize_predicate
