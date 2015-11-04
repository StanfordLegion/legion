-- Copyright 2015 Stanford University
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

-- Regent Loop Optimizer
--
-- Attempts to determine which loops can be transformed into index
-- space task launches.

local ast = require("regent/ast")
local data = require("regent/data")
local log = require("regent/log")
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_loop_scope(loop_variable)
  local cx = {
    constraints = self.constraints,
    loop_variable = loop_variable,
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

function check_privilege_noninterference(cx, task, region_type,
                                         other_region_type, mapping)
  local param_region_type = mapping[region_type]
  local other_param_region_type = mapping[other_region_type]
  assert(param_region_type and other_param_region_type)

  local privileges_by_field_path =
    std.group_task_privileges_by_field_path(
      std.find_task_privileges(
        param_region_type, task:getprivileges(), task:get_coherence_modes()))
  local other_privileges_by_field_path =
    std.group_task_privileges_by_field_path(
      std.find_task_privileges(
        other_param_region_type,
        task:getprivileges(), task:get_coherence_modes()))

  for field_path, privilege in pairs(privileges_by_field_path) do
    local other_privilege = other_privileges_by_field_path[field_path]

    if not (
        not privilege or privilege == "none" or
        not other_privilege or other_privilege == "none" or
        (privilege == "reads" and other_privilege == "reads") or
        (std.is_reduction_op(privilege) and privilege == other_privilege))
    then
      return false
    end
  end
  return true
end

function analyze_noninterference_previous(cx, task, region_type,
                                          regions_previously_used, mapping)
  for i, other_region_type in pairs(regions_previously_used) do
    local constraint = {
      lhs = region_type,
      rhs = other_region_type,
      op = "*"
    }

    if std.type_maybe_eq(region_type.fspace_type, other_region_type.fspace_type) and
      not std.check_constraint(cx, constraint) and
      not check_privilege_noninterference(cx, task, region_type, other_region_type, mapping)
    then
      return false, i
    end
  end
  return true
end

function analyze_noninterference_self(cx, task, region_type,
                                      partition_type, mapping)
  if partition_type and partition_type:is_disjoint() then
    return true
  end

  local param_region_type = mapping[region_type]
  assert(param_region_type)
  local privileges, privilege_field_paths = std.find_task_privileges(
    param_region_type, task:getprivileges(), task:get_coherence_modes())
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    if not (
        privilege == "none" or
        privilege == "reads" or
        std.is_reduction_op(privilege) or
        #field_paths == 0)
    then
      return false
    end
  end
  return true
end

local function analyze_is_side_effect_free_node(cx)
  return function(node)
    if node:is(ast.typed.expr.FieldAccess) then
      local ptr_type = std.as_read(node.value.expr_type)
      return not (std.is_bounded_type(ptr_type) or std.is_ref(ptr_type))
    elseif node:is(ast.typed.expr.Call) then
      return not std.is_task(node.fn.value)
    elseif node:is(ast.typed.expr.RawContext) or
      node:is(ast.typed.expr.RawPhysical) or
      node:is(ast.typed.expr.RawRuntime) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.Deref)
    then
      return false
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
    if node:is(ast.typed.expr.ID) then
      assert(cx.loop_variable)
      return node.value ~= cx.loop_variable
    elseif node:is(ast.typed.expr.FieldAccess) then
      local ptr_type = std.as_read(node.value.expr_type)
      return not (std.is_bounded_type(ptr_type) or std.is_ref(ptr_type))
    elseif node:is(ast.typed.expr.Call) or
      node:is(ast.typed.expr.Region) or
      node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.CrossProduct) or
      node:is(ast.typed.expr.Deref)
    then
      return false
    end
  end
end

local function analyze_is_loop_invariant(cx, node)
  return ast.mapreduce_node_postorder(
    analyze_is_loop_invariant_node(cx),
    data.all,
    node, true)
end

local optimize_index_launch_loops = {}

function ignore(...) end

function optimize_index_launch_loops.stat_for_num(cx, node)
  local log_pass = ignore
  local log_fail = ignore
  if node.options.parallel:is(ast.options.Demand) then
    log_pass = ignore -- log.warn
    log_fail = log.error
  end

  if node.values[3] and not (
    node.values[3]:is(ast.typed.expr.Constant) and
    node.values[3].value == 1)
  then
    log_fail(node, "loop optimization failed: stride not equal to 1")
    return node
  end

  if #node.block.stats ~= 1 then
    log_fail(node, "loop optimization failed: body has multiple statements")
    return node
  end

  local body = node.block.stats[1]
  local call
  local reduce_lhs, reduce_op = false, false
  if body:is(ast.typed.stat.Expr) and
    body.expr:is(ast.typed.expr.Call)
  then
    call = body.expr
  elseif body:is(ast.typed.stat.Reduce) and
    #body.lhs == 1 and
    #body.rhs == 1 and
    body.rhs[1]:is(ast.typed.expr.Call)
  then
    call = body.rhs[1]
    reduce_lhs = body.lhs[1]
    reduce_op = body.op
  else
    log_fail(body, "loop optimization failed: body is not a function call")
    return node
  end

  local task = call.fn.value
  if not std.is_task(task) then
    log_fail(call, "loop optimization failed: function is not a task")
    return node
  end

  if reduce_lhs then
    local reduce_as_type = std.as_read(call.expr_type)
    if not std.reduction_op_ids[reduce_op][reduce_as_type] then
      log_fail(body, "loop optimization failed: reduction over " .. tostring(reduce_op) .. " " .. tostring(reduce_as_type) .. " not supported")
    end

    if not analyze_is_side_effect_free(cx, reduce_lhs) then
      log_fail(body, "loop optimization failed: reduction is not side-effect free")
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
  --      b. Provably loop-variant, with simple indexing which can be
  --         analyzed by the optimizer.
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

  local loop_cx = cx:new_loop_scope(node.symbol)
  local param_types = task:gettype().parameters
  local args = call.args
  local args_provably = ast.IndexLaunchArgsProvably {
    invariant = terralib.newlist(),
    variant = terralib.newlist(),
  }
  local regions_previously_used = terralib.newlist()
  local mapping = {}
  for i, arg in ipairs(args) do
    if not analyze_is_side_effect_free(cx, arg) then
      log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not side-effect free")
      return node
    end

    local arg_invariant = analyze_is_loop_invariant(loop_cx, arg)
    local arg_variant = false
    local partition_type

    local arg_type = std.as_read(arg.expr_type)
    mapping[arg_type] = param_types[i]
    -- Tests for conformance to index launch requirements.
    if std.is_ispace(arg_type) or std.is_region(arg_type) then
      if arg:is(ast.typed.expr.IndexAccess) and
        (std.is_partition(std.as_read(arg.value.expr_type)) or
           std.is_cross_product(std.as_read(arg.value.expr_type))) and
        arg.index:is(ast.typed.expr.ID) and
        arg.index.value == node.symbol
      then
        partition_type = std.as_read(arg.value.expr_type)
        arg_variant = true
      end

      if not (arg_variant or arg_invariant) then
        log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not provably variant or invariant")
        return node
      end
    end

    if std.is_phase_barrier(arg_type) then
      -- Phase barriers must be invariant, or must not be used as an arrival/wait.
      if not arg_invariant then
        for _, variables in pairs(task:get_conditions()) do
          if variables[i] then
            log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not provably invariant")
            return node
          end
        end
      end
    end

    if std.is_list(arg_type) and arg_type:is_list_of_regions() then
      -- FIXME: Deoptimize lists of regions for the moment. Lists
      -- would have to be (at a minimum) invariant though other
      -- restrictions may apply.
      log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is a list of regions")
      return node
    end

    -- Tests for non-interference.
    if std.is_region(arg_type) then
      do
        local passed, failure_i = analyze_noninterference_previous(
          cx, task, arg_type, regions_previously_used, mapping)
        if not passed then
          log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " interferes with argument " .. tostring(failure_i))
          return node
        end
      end

      do
        local passed = analyze_noninterference_self(
          cx, task, arg_type, partition_type, mapping)
        if not passed then
          log_fail(call, "loop optimization failed: argument " .. tostring(i) .. " interferes with itself")
          return node
        end
      end
    end

    args_provably.invariant[i] = arg_invariant
    args_provably.variant[i] = arg_variant

    regions_previously_used[i] = nil
    if std.is_region(arg_type) then
      regions_previously_used[i] = arg_type
    end
  end

  log_pass("loop optimization succeeded")
  return ast.typed.stat.IndexLaunch {
    symbol = node.symbol,
    domain = node.values,
    call = call,
    reduce_lhs = reduce_lhs,
    reduce_op = reduce_op,
    args_provably = args_provably,
    options = node.options,
    span = node.span,
  }
end

local optimize_loops = {}

local function optimize_loops_node(cx)
  return function(node)
    if node:is(ast.typed.stat.ForNum) then
      return optimize_index_launch_loops.stat_for_num(cx, node)
    end
    return node
  end
end

function optimize_loops.block(cx, node)
  return ast.map_node_postorder(optimize_loops_node(cx), node)
end

function optimize_loops.stat_task(cx, node)
  local cx = cx:new_task_scope(node.prototype:get_constraints())
  local body = optimize_loops.block(cx, node.body)

  return node { body = body }
end

function optimize_loops.stat_top(cx, node)
  if node:is(ast.typed.stat.Task) then
    return optimize_loops.stat_task(cx, node)

  else
    return node
  end
end

function optimize_loops.entry(node)
  local cx = context.new_global_scope({})
  return optimize_loops.stat_top(cx, node)
end

return optimize_loops
