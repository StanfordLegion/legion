-- Copyright 2017 Stanford University, NVIDIA Corporation
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
context.__index = context

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

local function check(cx, node, allowed_set)
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

local function check_annotations_node(cx)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.Call) then
      check(cx, node, data.set({"parallel", "inline"}))

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
      node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.New) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
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
      node:is(ast.typed.expr.ListCrossProduct) or
      node:is(ast.typed.expr.ListCrossProductComplete) or
      node:is(ast.typed.expr.ListPhaseBarriers) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.ListIspace) or
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
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary) or
      node:is(ast.typed.expr.Deref) or
      node:is(ast.typed.expr.ParallelizerConstraint)
    then
      check(cx, node, data.set({}))

    -- Statements:
    elseif node:is(ast.typed.stat.If) or
      node:is(ast.typed.stat.Elseif)
    then
      check(cx, node, data.set({}))

    elseif node:is(ast.typed.stat.While) then
      check(cx, node, data.set({"spmd", "trace"}))

    elseif node:is(ast.typed.stat.ForNum) then
      local annotations = {"parallel", "spmd", "trace"}
      if std.config["vectorize-unsafe"] then
        annotations[#annotations + 1] = "vectorize"
      end
      check(cx, node, data.set(annotations))

    elseif node:is(ast.typed.stat.ForList) then
      check(cx, node, data.set({"parallel", "spmd", "trace", "vectorize"}))

    elseif node:is(ast.typed.stat.Repeat) then
      check(cx, node, data.set({"spmd", "trace"}))

    elseif node:is(ast.typed.stat.MustEpoch) then
      check(cx, node, data.set({}))

    elseif node:is(ast.typed.stat.Block) then
      check(cx, node, data.set({"spmd", "trace"}))

    elseif node:is(ast.typed.stat.Var) or
      node:is(ast.typed.stat.VarUnpack) or
      node:is(ast.typed.stat.Return) or
      node:is(ast.typed.stat.Break) or
      node:is(ast.typed.stat.Assignment) or
      node:is(ast.typed.stat.Reduce) or
      node:is(ast.typed.stat.Expr) or
      node:is(ast.typed.stat.RawDelete) or
      node:is(ast.typed.stat.ParallelizeWith)
    then
      check(cx, node, data.set({}))

    -- Tasks:
    elseif node:is(ast.typed.top.TaskParam) then
      check(cx, node, data.set({}))

    elseif node:is(ast.typed.top.Task) then
      check(cx, node, data.set({"cuda", "external", "inline", "parallel"}))

    -- Miscellaneous:
    elseif node:is(ast.typed.Block) or
      node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.constraint_kind) or
      node:is(ast.privilege_kind) or
      node:is(ast.condition_kind) or
      node:is(ast.disjointness_kind) or
      node:is(ast.constraint) or
      node:is(ast.privilege) or
      node:is(ast.TaskConfigOptions)
    then
      -- Pass

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function check_annotations_top(cx, node)
  ast.traverse_node_postorder(check_annotations_node(cx), node)
end

function check_annotations.top_task(cx, node)
  check_annotations_top(cx, node)
end

function check_annotations.top_fspace(cx, node)
  check(cx, node, data.set({}))
end

function check_annotations.top_quote_expr(cx, node)
  check(cx, node, data.set({}))
end

function check_annotations.top_quote_stat(cx, node)
  check(cx, node, data.set({}))
end

function check_annotations.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return check_annotations.top_task(cx, node)

  elseif node:is(ast.typed.top.Fspace) then
    return check_annotations.top_fspace(cx, node)

  elseif node:is(ast.specialized.top.QuoteExpr) then
    return check_annotations.top_quote_expr(cx, node)

  elseif node:is(ast.specialized.top.QuoteStat) then
    return check_annotations.top_quote_stat(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function check_annotations.entry(node)
  local cx = context.new_global_scope()
  check_annotations.top(cx, node)
  return node
end

return check_annotations
