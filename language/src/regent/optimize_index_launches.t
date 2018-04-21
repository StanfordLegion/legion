-- Copyright 2018 Stanford University
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

-- Regent Index Launch Optimizer
--
-- Attempts to determine which loops can be transformed into index
-- space task launches.

local affine_helper = require("regent/affine_helper")
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

function context:new_local_scope()
  local cx = {
    constraints = self.constraints,
    loop_index = false,
    loop_variables = {},
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(constraints)
  local cx = {
    constraints = constraints,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:set_loop_index(loop_index)
  assert(not self.loop_index)
  self.loop_index = loop_index
end

function context:add_loop_variable(loop_variable)
  assert(self.loop_variables)
  self.loop_variables[loop_variable] = true
end

function context:is_loop_variable(variable)
  assert(self.loop_variables)
  return self.loop_variables[variable]
end

function context:is_loop_index(variable)
  assert(self.loop_index)
  return self.loop_index == variable
end

local function check_privilege_noninterference(cx, task, arg,
                                         other_arg, mapping)
  local region_type = std.as_read(arg.expr_type)
  local other_region_type = std.as_read(other_arg.expr_type)
  local param_region_type = mapping[arg]
  local other_param_region_type = mapping[other_arg]
  assert(param_region_type and other_param_region_type)

  local privileges_by_field_path, coherence_modes_by_field_path =
    std.group_task_privileges_by_field_path(
      std.find_task_privileges(param_region_type, task))
  local other_privileges_by_field_path, other_coherence_modes_by_field_path =
    std.group_task_privileges_by_field_path(
      std.find_task_privileges(other_param_region_type, task))

  for field_path, privilege in pairs(privileges_by_field_path) do
    local other_privilege = other_privileges_by_field_path[field_path]

    if not (
        not privilege or privilege == "none" or
        not other_privilege or other_privilege == "none" or
        (privilege == "reads" and other_privilege == "reads") or
        (std.is_reduction_op(privilege) and privilege == other_privilege) or
        (coherence_modes_by_field_path[field_path] == "simultaneous" and
         other_coherence_modes_by_field_path[field_path] == "simultaneous"))
    then
      return false
    end
  end
  return true
end

local function strip_casts(node)
  if node:is(ast.typed.expr.Cast) then
    return node.arg
  end
  return node
end

local function check_index_noninterference_self(cx, arg)
  local index = strip_casts(arg.index)

  -- Easy case: index is just the loop variable.
  if (index:is(ast.typed.expr.ID) and cx:is_loop_index(index.value)) then
    return true
  end

  -- Another easy case: index is loop variable plus or minus a constant.
  if (index:is(ast.typed.expr.Binary) and
      index.lhs:is(ast.typed.expr.ID) and cx:is_loop_index(index.lhs.value) and
      affine_helper.is_constant_expr(index.rhs) and
      (index.op == "+" or index.op == "-"))
  then
    return true
  end

  -- FIXME: Do a proper affine analysis of the index expression.

  -- Otherwise return false.
  return false
end

local function analyze_noninterference_previous(
    cx, task, arg, regions_previously_used, mapping)
  local region_type = std.as_read(arg.expr_type)
  for i, other_arg in pairs(regions_previously_used) do
    local other_region_type = std.as_read(other_arg.expr_type)
    local constraint = std.constraint(
      region_type,
      other_region_type,
      std.disjointness)
    local exclude_variables = { [cx.loop_index] = true }

    if not (
        not std.type_maybe_eq(region_type.fspace_type, other_region_type.fspace_type) or
        std.check_constraint(cx, constraint, exclude_variables) or
        check_privilege_noninterference(cx, task, arg, other_arg, mapping))
        -- Index non-interference is handled at the type checker level
        -- and is captured in the constraints.
    then
      return false, i
    end
  end
  return true
end

local function analyze_noninterference_self(
    cx, task, arg, partition_type, mapping)
  local region_type = std.as_read(arg.expr_type)
  if partition_type and partition_type:is_disjoint() and
    check_index_noninterference_self(cx, arg)
  then
    return true
  end

  local param_region_type = mapping[arg]
  assert(param_region_type)
  local privileges, privilege_field_paths, privilege_field_types,
        privilege_coherence_modes, privilege_flags = std.find_task_privileges(
    param_region_type, task)
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local coherence = privilege_coherence_modes[i]
    if not (
        privilege == "none" or
        privilege == "reads" or
        std.is_reduction_op(privilege) or
        coherence == "simultaneous" or
        #field_paths == 0)
    then
      return false
    end
  end
  return true
end

local function analyze_is_side_effect_free_node(cx)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.IndexAccess) then
      return not std.is_ref(node.expr_type)
    elseif node:is(ast.typed.expr.Call) then
      return not std.is_task(node.fn.value)
    elseif node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
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
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Deref)
    then
      return false

    elseif node:is(ast.typed.expr.ID) or
      node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.RawFields) or
      node:is(ast.typed.expr.RawValue) or
      node:is(ast.typed.expr.Isnull) or
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.DynamicCollectiveGetResult) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary)
    then
      return true

    -- Statements:
    elseif node:is(ast.typed.stat.Var) then
      return true

    -- Miscellaneous:
    elseif node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_is_side_effect_free(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_is_side_effect_free_node(cx),
    data.all,
    node, true)
end

local function analyze_is_loop_invariant_node(cx)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.ID) then
      return not cx:is_loop_variable(node.value)
    elseif node:is(ast.typed.expr.IndexAccess) then
      return not std.is_ref(node.expr_type)
    elseif node:is(ast.typed.expr.Call) or
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
      node:is(ast.typed.expr.Adjust) or
      node:is(ast.typed.expr.Arrive) or
      node:is(ast.typed.expr.Await) or
      node:is(ast.typed.expr.Copy) or
      node:is(ast.typed.expr.Fill) or
      node:is(ast.typed.expr.Acquire) or
      node:is(ast.typed.expr.Release) or
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Deref)
    then
      return false

    elseif node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.FieldAccess) or
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
      node:is(ast.typed.expr.Null) or
      node:is(ast.typed.expr.DynamicCast) or
      node:is(ast.typed.expr.StaticCast) or
      node:is(ast.typed.expr.UnsafeCast) or
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
      node:is(ast.typed.expr.Advance) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_is_loop_invariant(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_is_loop_invariant_node(cx),
    data.all,
    node, true)
end

local function analyze_is_simple_index_expression_node(cx)
  return function(node)
    -- Expressions:
    if node:is(ast.typed.expr.ID) then
      -- Right now we can't capture a closure on any variable other
      -- than the loop variable, because that's the only variable that
      -- gets supplied through the projection functor API.
      return cx:is_loop_index(node.value)
    elseif node:is(ast.typed.expr.FieldAccess) or
      node:is(ast.typed.expr.IndexAccess) or
      node:is(ast.typed.expr.MethodCall) or
      node:is(ast.typed.expr.Call) or
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
      node:is(ast.typed.expr.ListInvert) or
      node:is(ast.typed.expr.ListRange) or
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
      node:is(ast.typed.expr.AllocateScratchFields) or
      node:is(ast.typed.expr.WithScratchFields) or
      node:is(ast.typed.expr.RegionRoot) or
      node:is(ast.typed.expr.Condition) or
      node:is(ast.typed.expr.Deref)
    then
      return false

    elseif node:is(ast.typed.expr.Constant) or
      node:is(ast.typed.expr.Function) or
      node:is(ast.typed.expr.Cast) or
      node:is(ast.typed.expr.Ctor) or
      node:is(ast.typed.expr.CtorListField) or
      node:is(ast.typed.expr.CtorRecField) or
      node:is(ast.typed.expr.Unary) or
      node:is(ast.typed.expr.Binary)
    then
      return true

    -- Miscellaneous:
    elseif node:is(ast.location) or
      node:is(ast.annotation) or
      node:is(ast.condition_kind)
    then
      return true

    else
      assert(false, "unexpected node type " .. tostring(node.node_type))
    end
  end
end

local function analyze_is_simple_index_expression(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_is_simple_index_expression_node(cx),
    data.all,
    node, true)
end

local function analyze_is_projectable(cx, arg)
  -- 1. We can project any index access `p[...]`
  if not arg:is(ast.typed.expr.IndexAccess) then
    return false
  end

  -- 2. As long as `p` is a partition or cross product (otherwise this
  -- is irrelevant, since we wouldn't be producing a region requirement).
  if not (std.is_partition(std.as_read(arg.value.expr_type)) or
       std.is_cross_product(std.as_read(arg.value.expr_type)))
  then
    return false
  end

  -- 3. And as long as `p` is loop-invariant (we have to index from
  -- the same partition every time).
  if not analyze_is_loop_invariant(cx, arg.value) then
    return false
  end

  -- 4. And as long as the index itself is a simple expression of the
  -- loop index.
  return analyze_is_simple_index_expression(cx, arg.index)
end

local optimize_index_launch = {}

local function ignore(...) end

local function optimize_loop_body(cx, node, report_pass, report_fail)
  if #node.block.stats == 0 then
    report_fail(node, "loop optimization failed: body is empty")
    return
  end

  local loop_cx = cx:new_local_scope()
  loop_cx:set_loop_index(node.symbol)
  loop_cx:add_loop_variable(node.symbol)

  local preamble = terralib.newlist()
  local call_stat
  for i = 1, #node.block.stats - 1 do
    local stat = node.block.stats[i]
    if not stat:is(ast.typed.stat.Var) then
      report_fail(stat, "loop optimization failed: preamble statement is not a variable")
      return
    end
    if not analyze_is_side_effect_free(loop_cx, stat) then
      if not (i == #node.block.stats - 1 and
              stat.value and
              stat.value:is(ast.typed.expr.Call)) then
        report_fail(stat, "loop optimization failed: preamble statement is not side-effect free")
        return
      else
        call_stat = stat
      end
    end

    if call_stat == nil then
      if stat.value and not analyze_is_loop_invariant(loop_cx, stat.value) then
        loop_cx:add_loop_variable(stat.symbol)
      end
      preamble:insert(stat)
    end
  end

  local body = node.block.stats[#node.block.stats]
  local call
  local reduce_lhs, reduce_op = false, false
  if call_stat ~= nil then
    if body:is(ast.typed.stat.Reduce) and
      body.rhs:is(ast.typed.expr.ID) and
      call_stat.symbol == body.rhs.value
    then
      call = call_stat.value
      reduce_lhs = body.lhs
      reduce_op = body.op
    else
      report_fail(call_stat, "loop optimization failed: preamble statement is not side-effect free")
      return
    end
  else
    if body:is(ast.typed.stat.Expr) and
      body.expr:is(ast.typed.expr.Call)
    then
      call = body.expr
    elseif body:is(ast.typed.stat.Reduce) and
      body.rhs:is(ast.typed.expr.Call)
    then
      call = body.rhs
      reduce_lhs = body.lhs
      reduce_op = body.op
    else
      report_fail(body, "loop optimization failed: body is not a function call")
      return
    end
  end

  local task = call.fn.value
  if not std.is_task(task) then
    report_fail(call, "loop optimization failed: function is not a task")
    return
  end

  if #call.conditions > 0 then
    report_fail(call, "FIXME: handle analysis of ad-hoc conditions")
    return
  end

  if reduce_lhs then
    local reduce_as_type = std.as_read(call.expr_type)
    if not std.reduction_op_ids[reduce_op][reduce_as_type] then
      report_fail(body, "loop optimization failed: reduction over " .. tostring(reduce_op) .. " " .. tostring(reduce_as_type) .. " not supported")
    end

    if not analyze_is_side_effect_free(loop_cx, reduce_lhs) then
      report_fail(body, "loop optimization failed: reduction is not side-effect free")
    end
  end

  -- Perform static dependence analysis on the loop body to make sure
  -- that the loop is safe to run in parallel. This analysis proceeds
  -- by induction on arguments.
  --
  -- For each argument:
  --
  --  0. Determine whether the argument is provably side-effect
  --     free. (Technically, what this actually does is prove that any
  --     side-effects are non-interfering with the task call itself
  --     and therefore can be split out from the index launch.)
  --
  --  1. Determine whether the argument is provably loop-invariant or
  --     provably "loop-variant" (i.e., a simple function of the loop
  --     index). This is important because index space task launches
  --     are restricted in the forms of region (and index space)
  --     requirements they can accept. All regions (and index spaces)
  --     must be one of:
  --
  --      a. Provably loop-invariant.
  --      b. Provably projectable (i.e. can be expressed as a
  --         projection functor from the partition and index), with
  --         simple indexing which can be analyzed by the optimizer.
  --      c. (Not yet implemented.) Neither, but (i) the region has no
  --         privileges, and (ii) it is provably a subregion of one of
  --         the other region arguments to the task (which must
  --         satisfy either (a) or (b)).
  --
  --  2. For region-typed arguments, compare for non-interference
  --     against previous arguments. Regions must be provably
  --     non-interfering with all previous arguments to pass this
  --     analysis.
  --
  --  3. For region-typed arguments, compare for non-interference
  --     against itself. This means that any arguments that are not
  --     provably loop-variant must be read-only (or reductions,
  --     atomic, simultaneous, etc.). Loop-variant arguments can
  --     additionally prove non-interference by demonstrating that
  --     they come from a disjoint partition, as long as indexing into
  --     said partition is provably disjoint.

  local param_types = task:get_type().parameters
  local args = call.args
  local args_provably = ast.IndexLaunchArgsProvably {
    invariant = terralib.newlist(),
    projectable = terralib.newlist(),
  }
  local regions_previously_used = terralib.newlist()
  local mapping = {}
  for i, arg in ipairs(args) do
    if not analyze_is_side_effect_free(loop_cx, arg) then
      report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not side-effect free")
      return
    end

    local arg_invariant = analyze_is_loop_invariant(loop_cx, arg)

    local arg_projectable = false
    local partition_type

    local arg_type = std.as_read(arg.expr_type)
    -- XXX: This will break again if arg isn't unique for each argument,
    --      which can happen when de-duplicating AST nodes.
    assert(mapping[arg] == nil)
    mapping[arg] = param_types[i]
    -- Tests for conformance to index launch requirements.
    if std.is_ispace(arg_type) or std.is_region(arg_type) then
      if analyze_is_projectable(loop_cx, arg) then
        partition_type = std.as_read(arg.value.expr_type)
        arg_projectable = true
      end

      if not (arg_projectable or arg_invariant) then
        report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not provably projectable or invariant")
        return
      end
    end

    if std.is_phase_barrier(arg_type) then
      -- Phase barriers must be invariant, or must not be used as an arrival/wait.
      if not arg_invariant then
        for _, variables in pairs(task:get_conditions()) do
          if variables[i] then
            report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not provably invariant")
            return
          end
        end
      end
    end

    if std.is_list(arg_type) and arg_type:is_list_of_regions() then
      -- FIXME: Deoptimize lists of regions for the moment. Lists
      -- would have to be (at a minimum) invariant though other
      -- restrictions may apply.
      report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is a list of regions")
      return
    end

    -- Tests for non-interference.
    if std.is_region(arg_type) then
      do
        local passed, failure_i = analyze_noninterference_previous(
          loop_cx, task, arg, regions_previously_used, mapping)
        if not passed then
          report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " interferes with argument " .. tostring(failure_i))
          return
        end
      end

      do
        local passed = analyze_noninterference_self(
          loop_cx, task, arg, partition_type, mapping)
        if not passed then
          report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " interferes with itself")
          return
        end
      end
    end

    args_provably.invariant[i] = arg_invariant
    args_provably.projectable[i] = arg_projectable

    regions_previously_used[i] = nil
    if std.is_region(arg_type) then
      regions_previously_used[i] = arg
    end
  end

  report_pass("loop optimization succeeded")
  return {
    preamble = preamble,
    call = call,
    reduce_lhs = reduce_lhs,
    reduce_op = reduce_op,
    args_provably = args_provably,
  }
end

function optimize_index_launch.stat_for_num(cx, node)
  local report_pass = ignore
  local report_fail = report.info
  if node.annotations.parallel:is(ast.annotation.Demand) then
    report_pass = ignore
    report_fail = report.error
  end

  if node.annotations.parallel:is(ast.annotation.Forbid) then
    return node
  end

  if node.values[3] and not (
    node.values[3]:is(ast.typed.expr.Constant) and
    node.values[3].value == 1)
  then
    report_fail(node, "loop optimization failed: stride not equal to 1")
    return node
  end

  local body = optimize_loop_body(cx, node, report_pass, report_fail)
  if not body then
    return node
  end

  return ast.typed.stat.IndexLaunchNum {
    symbol = node.symbol,
    values = node.values,
    preamble = body.preamble,
    call = body.call,
    reduce_lhs = body.reduce_lhs,
    reduce_op = body.reduce_op,
    args_provably = body.args_provably,
    annotations = node.annotations,
    span = node.span,
  }
end

function optimize_index_launch.stat_for_list(cx, node)
  local report_pass = ignore
  local report_fail = report.info
  if node.annotations.parallel:is(ast.annotation.Demand) then
    report_pass = ignore
    report_fail = report.error
  end

  if node.annotations.parallel:is(ast.annotation.Forbid) then
    return node
  end

  local value_type = std.as_read(node.value.expr_type)
  if not (std.is_ispace(value_type) or std.is_region(value_type)) then
    report_fail(node, "loop optimization failed: domain is not a ispace or region")
    return node
  end

  local body = optimize_loop_body(cx, node, report_pass, report_fail)
  if not body then
    return node
  end

  return ast.typed.stat.IndexLaunchList {
    symbol = node.symbol,
    value = node.value,
    preamble = body.preamble,
    call = body.call,
    reduce_lhs = body.reduce_lhs,
    reduce_op = body.reduce_op,
    args_provably = body.args_provably,
    annotations = node.annotations,
    span = node.span,
  }
end

local optimize_index_launches = {}

local function optimize_index_launches_node(cx)
  return function(node)
    if node:is(ast.typed.stat.ForNum) then
      return optimize_index_launch.stat_for_num(cx, node)
    elseif node:is(ast.typed.stat.ForList) then
      return optimize_index_launch.stat_for_list(cx, node)
    end
    return node
  end
end

function optimize_index_launches.block(cx, node)
  return ast.map_node_postorder(optimize_index_launches_node(cx), node)
end

function optimize_index_launches.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope(node.prototype:get_constraints())
  local body = optimize_index_launches.block(cx, node.body)

  return node { body = body }
end

function optimize_index_launches.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return optimize_index_launches.top_task(cx, node)

  else
    return node
  end
end

function optimize_index_launches.entry(node)
  local cx = context.new_global_scope({})
  return optimize_index_launches.top(cx, node)
end

optimize_index_launches.pass_name = "optimize_index_launches"

return optimize_index_launches
