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

-- Legion Future Optimizer
--
-- This pass changes all non-void task calls to return
-- futures. Wherever possible, this pass attempts to work with futures
-- directly rather than blocking in order to obtain a concrete value.

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")
local task_helper = require("regent/task_helper")

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

function context:new_stat_scope()
  local cx = {
    task_is_leaf = self.task_is_leaf,
    local_symbols = self.local_symbols,
    var_flows = self.var_flows,
    var_futures = self.var_futures,
    var_symbols = self.var_symbols,
    conds = self.conds,
    spills = terralib.newlist(),
  }
  return setmetatable(cx, context)
end

function context:new_local_scope(cond)
  local conds = terralib.newlist()
  conds:insertall(self.conds)
  conds:insert(cond)
  local cx = {
    task_is_leaf = self.task_is_leaf,
    local_symbols = self.local_symbols:copy(),
    var_flows = self.var_flows,
    var_futures = self.var_futures,
    var_symbols = self.var_symbols,
    conds = conds,
    spills = terralib.newlist(),
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(task_is_leaf)
  assert(task_is_leaf ~= nil)
  local cx = {
    task_is_leaf = task_is_leaf,
    local_symbols = data.newmap(),
    var_flows = data.newmap(),
    var_futures = data.newmap(),
    var_symbols = data.newmap(),
    conds = terralib.newlist(),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:get_flow(v)
  if not self.var_flows[v] then
    self.var_flows[v] = data.newmap()
  end
  return self.var_flows[v]
end

function context:is_var_future(v)
  return self.var_futures[v]
end

function context:symbol(v)
  return self.var_symbols[v]
end

function context:add_spill(stat)
  self.spills:insert(stat)
  return self
end

function context:get_spills()
  return self.spills
end

local analyze_var_flow = {}

function flow_empty()
  return data.newmap()
end

function flow_future()
  local result = data.newmap()
  result[true] = true -- Represents an unconditional future r-value
  return result
end

function flow_var(v)
  local result = data.newmap()
  result[v] = true -- Represents a variable
  return result
end

function flow_future_into(cx, lhs) -- Unconditionally flow future into l-value
  for _, v in lhs:keys() do
    local var_flow = cx:get_flow(v)
    var_flow[true] = true
  end
end

function flow_value_into_var(cx, symbol, value) -- Flow r-value into variable
  local var_flow = cx:get_flow(symbol)
  for _, v in value:keys() do
    var_flow[v] = true
  end
end

function flow_value_into(cx, lhs, rhs) -- Flow r-value into l-value
  for _, lhv in lhs:keys() do
    local lhv_flow = cx:get_flow(lhv)
    for _, rhv in rhs:keys() do
      lhv_flow[rhv] = true
      end
  end
end

function meet_flow(...)
  local flow = data.newmap()
  for _, a in ipairs({...}) do
    for _, v in a:keys() do
      flow[v] = true
    end
  end
  return flow
end

function analyze_var_flow.expr_id(cx, node)
  return flow_var(node.value)
end

function analyze_var_flow.expr_call(cx, node)
  if std.is_task(node.fn.value) and
    not node.fn.value.is_local and
    node.expr_type ~= terralib.types.unit
  then
    return flow_future()
  end
  return flow_empty()
end

function analyze_var_flow.expr_dynamic_collective_get_result(cx, node)
  return flow_future()
end

function analyze_var_flow.expr_unary(cx, node)
  return analyze_var_flow.expr(cx, node.rhs)
end

function analyze_var_flow.expr_binary(cx, node)
  return meet_flow(
    analyze_var_flow.expr(cx, node.lhs),
    analyze_var_flow.expr(cx, node.rhs))
end

function analyze_var_flow.expr_raw_future(cx, noe)
  if cx.task_is_leaf then
    return flow_empty()
  else
    return flow_future()
  end
end

function analyze_var_flow.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    return analyze_var_flow.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) or
    node:is(ast.typed.expr.Global) or
    node:is(ast.typed.expr.Function) or
    node:is(ast.typed.expr.FieldAccess) or
    node:is(ast.typed.expr.IndexAccess) or
    node:is(ast.typed.expr.MethodCall)
  then
    return flow_empty()

  elseif node:is(ast.typed.expr.Call) then
    return analyze_var_flow.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) or
    node:is(ast.typed.expr.Ctor) or
    node:is(ast.typed.expr.RawContext) or
    node:is(ast.typed.expr.RawFields) or
    node:is(ast.typed.expr.RawPhysical) or
    node:is(ast.typed.expr.RawRuntime) or
    node:is(ast.typed.expr.RawTask) or
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
    node:is(ast.typed.expr.PartitionByRestriction) or
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
    node:is(ast.typed.expr.DynamicCollective)
  then
    return flow_empty()

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return analyze_var_flow.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) or
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
    node:is(ast.typed.expr.ImportIspace) or
    node:is(ast.typed.expr.ImportRegion) or
    node:is(ast.typed.expr.ImportPartition) or
    node:is(ast.typed.expr.ImportCrossProduct) or
    node:is(ast.typed.expr.Projection)
  then
    return flow_empty()

  elseif node:is(ast.typed.expr.Unary) then
    return analyze_var_flow.expr_unary(cx, node)

  elseif node:is(ast.typed.expr.Binary) then
    return analyze_var_flow.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.Deref) then
    return flow_empty()

  elseif node:is(ast.typed.expr.AddressOf) then
    return flow_empty()

  elseif node:is(ast.typed.expr.RawFuture) then
    return analyze_var_flow.expr_raw_future(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_var_flow.block(cx, node)
  node.stats:map(
    function(stat) return analyze_var_flow.stat(cx, stat) end)
end

function analyze_var_flow.stat_if(cx, node)
  local cond = analyze_var_flow.expr(cx, node.cond)
  local cx = cx:new_local_scope(cond)
  analyze_var_flow.block(cx, node.then_block)
  node.elseif_blocks:map(
    function(block) return analyze_var_flow.stat_elseif(cx, block) end)
  analyze_var_flow.block(cx, node.else_block)
end

function analyze_var_flow.stat_elseif(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_while(cx, node)
  local cond = analyze_var_flow.expr(cx, node.cond)
  local cx = cx:new_local_scope(cond)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_for_num(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_for_list(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_repeat(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_must_epoch(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_block(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_index_launch_num(cx, node)
  node.preamble:map(function(stat) analyze_var_flow.stat(cx, stat) end)
  local reduce_lhs = node.reduce_lhs and
    analyze_var_flow.expr(cx, node.reduce_lhs) or
    flow_empty()
  flow_future_into(cx, reduce_lhs)
end

function analyze_var_flow.stat_index_launch_list(cx, node)
  node.preamble:map(function(stat) analyze_var_flow.stat(cx, stat) end)
  local reduce_lhs = node.reduce_lhs and
    analyze_var_flow.expr(cx, node.reduce_lhs) or
    flow_empty()
  flow_future_into(cx, reduce_lhs)
end

function analyze_var_flow.stat_var(cx, node)
  local value = node.value and analyze_var_flow.expr(cx, node.value) or flow_empty()
  cx.local_symbols[node.symbol] = #cx.conds
  flow_value_into_var(cx, node.symbol, value)
end

function analyze_var_flow.stat_assignment(cx, node)
  local lhs = analyze_var_flow.expr(cx, node.lhs)
  local rhs = analyze_var_flow.expr(cx, node.rhs)
  flow_value_into(cx, lhs, rhs)

  -- Hack: SCR breaks if certain values are futures, for now just make
  -- sure we don't do this for certain types.
  local lhs_type = std.as_read(node.lhs.expr_type)
  if not (std.is_list(lhs_type) or std.is_phase_barrier(lhs_type) or std.is_dynamic_collective(lhs_type)) then
    -- Make sure any dominating conditions flow into this assignment.
    local lhs_symbol = node.lhs:is(ast.typed.expr.ID) and node.lhs.value
    for i, cond in ipairs(cx.conds) do
      if not cx.local_symbols[lhs_symbol] or i > cx.local_symbols[lhs_symbol] then
        flow_value_into(cx, lhs, cond)
      end
    end
  end
end

function analyze_var_flow.stat_reduce(cx, node)
  local lhs = analyze_var_flow.expr(cx, node.lhs)
  local rhs = analyze_var_flow.expr(cx, node.rhs)
  flow_value_into(cx, lhs, rhs)

  -- Hack: SCR breaks if certain values are futures, for now just make
  -- sure we don't do this for certain types.
  local lhs_type = std.as_read(node.lhs.expr_type)
  if not (std.is_list(lhs_type) or std.is_phase_barrier(lhs_type) or std.is_dynamic_collective(lhs_type)) then
    -- Make sure any dominating conditions flow into this assignment.
    local lhs_symbol = node.lhs:is(ast.typed.expr.ID) and node.lhs.value
    for i, cond in ipairs(cx.conds) do
      if not cx.local_symbols[lhs_symbol] or i > cx.local_symbols[lhs_symbol] then
        flow_value_into(cx, lhs, cond)
      end
    end
  end
end

function analyze_var_flow.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return analyze_var_flow.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return analyze_var_flow.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return analyze_var_flow.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return analyze_var_flow.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return analyze_var_flow.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return analyze_var_flow.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return analyze_var_flow.stat_block(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return analyze_var_flow.stat_index_launch_num(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return analyze_var_flow.stat_index_launch_list(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return analyze_var_flow.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return

  elseif node:is(ast.typed.stat.Return) then
    return

  elseif node:is(ast.typed.stat.Break) then
    return

  elseif node:is(ast.typed.stat.Assignment) then
    return analyze_var_flow.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return analyze_var_flow.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return

  elseif node:is(ast.typed.stat.RawDelete) then
    return

  elseif node:is(ast.typed.stat.Fence) then
    return

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    return

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local function compute_var_futures(cx)
  local inflow = data.newmap()
  for v1, flow in cx.var_flows:items() do
    for _, v2 in flow:keys() do
      if not inflow[v2] then
        inflow[v2] = data.newmap()
      end
      inflow[v2][v1] = true
    end
  end

  local var_futures = cx.var_futures
  var_futures[true] = true
  repeat
    local changed = false
    for _, v1 in var_futures:keys() do
      if inflow[v1] then
        for _, v2 in inflow[v1]:keys() do
          if not var_futures[v2] then
            var_futures[v2] = true
            changed = true
          end
        end
      end
    end
  until not changed
end

local function compute_var_symbols(cx)
  for v, is_future in cx.var_futures:items() do
    if std.is_symbol(v) and is_future then
      assert(terralib.types.istype(v:hastype()) and not std.is_future(v:gettype()))
      cx.var_symbols[v] = std.newsymbol(std.future(v:gettype()), v:hasname())
    end
  end
end

local optimize_futures = {}

-- Normalize all sub-expressions that could be lifted to tasks.
-- This will help us track futures from those lifted tasks accurately.
local function normalize_compound_expr(cx, expr)
  if expr:is(ast.typed.expr.Cast) or
     expr:is(ast.typed.expr.Unary) or
     expr:is(ast.typed.expr.Binary) or
     expr:is(ast.typed.expr.Call) or
     expr:is(ast.typed.expr.Future) or
     expr:is(ast.typed.expr.DynamicCollectiveGetResult)
  then
    local temp_var = std.newsymbol(expr.expr_type, "__normalized_in_future_opt")
    cx:add_spill(ast.typed.stat.Var {
      symbol = temp_var,
      type = expr.expr_type,
      value = expr,
      span = expr.span,
      annotations = ast.default_annotations(),
    })
    return ast.typed.expr.ID {
      value = temp_var,
      expr_type = expr.expr_type,
      span = expr.span,
      annotations = ast.default_annotations(),
    }
  else
    return expr
  end
end

local function normalize(cx, node)
  if node:is(ast.typed.expr.Binary) then
    local lhs = normalize_compound_expr(cx, normalize(cx, node.lhs))
    local rhs = normalize_compound_expr(cx, normalize(cx, node.rhs))
    return node {
      lhs = lhs,
      rhs = rhs,
    }
  elseif node:is(ast.typed.expr.Unary) then
    local rhs = normalize_compound_expr(cx, normalize(cx, node.rhs))
    return node {
      rhs = rhs,
    }
  elseif node:is(ast.typed.expr.Cast) then
    local arg = normalize_compound_expr(cx, normalize(cx, node.arg))
    return node {
      arg = arg,
    }
  else
    return node
  end
end

local function concretize(cx, node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    if not node:is(ast.typed.expr.ID) then
      node = normalize_compound_expr(cx, normalize(cx, node))
    end
    return ast.typed.expr.FutureGetResult {
      value = node,
      expr_type = expr_type.result_type,
      annotations = node.annotations,
      span = node.span,
    }
  end
  return node
end

local function promote(cx, node, expected_type)
  assert(std.is_future(expected_type))

  local expr_type = std.as_read(node.expr_type)
  if not std.is_future(expr_type) then
    return normalize_compound_expr(cx,
      ast.typed.expr.Future {
        value = node,
        expr_type = expected_type,
        annotations = node.annotations,
        span = node.span,
      })
  elseif not std.type_eq(expr_type, expected_type) then
    -- FIXME: This requires a cast. For now, just concretize and re-promote.
    return promote(cx, concretize(cx, node), expected_type)
  end
  return normalize(cx, node)
end

function optimize_futures.expr_region_root(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_condition(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_id(cx, node)
  if cx:is_var_future(node.value) then
    local expr_type
    if std.is_rawref(node.expr_type) then
      expr_type = std.rawref(&std.future(std.as_read(node.expr_type)))
    else
      expr_type = std.future(node.expr_type)
    end
    return node {
      value = cx:symbol(node.value),
      expr_type = expr_type,
    }
  end
  return node
end

function optimize_futures.expr_field_access(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_index_access(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local index = concretize(cx, optimize_futures.expr(cx, node.index))
  return node {
    value = value,
    index = index,
  }
end

function optimize_futures.expr_method_call(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local args = node.args:map(
    function(arg) return concretize(cx, optimize_futures.expr(cx, arg)) end)
  return node {
    value = value,
    args = args,
  }
end

function optimize_futures.expr_call(cx, node)
  local fn = concretize(cx, optimize_futures.expr(cx, node.fn))
  local args = node.args:map(
    function(arg) return optimize_futures.expr(cx, arg) end)
  if not std.is_task(node.fn.value) then
    args = args:map(
      function(arg) return concretize(cx, arg) end)
  else
    args = args:map(function(arg)
      if std.is_future(arg.expr_type) then
          return normalize_compound_expr(cx, arg)
      else
        return arg
      end
    end)
  end
  local expr_type = node.expr_type
  if std.is_task(node.fn.value) and
    not node.fn.value.is_local and
    expr_type ~= terralib.types.unit
  then
    expr_type = std.future(expr_type)
  end

  return node {
    fn = fn,
    args = args,
    expr_type = expr_type,
  }
end

local function lift_cast_to_futures(node)
  local arg_type = std.as_read(node.arg.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local task = task_helper.make_cast_task(arg_type, expr_type)

  return ast.typed.expr.Call {
    fn = ast.typed.expr.Function {
      value = task,
      expr_type = task:get_type(),
      annotations = ast.default_annotations(),
      span = node.span,
    },
    args = terralib.newlist({
      node.arg,
    }),
    conditions = terralib.newlist(),
    predicate = false,
    predicate_else_value = false,
    replicable = false,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function optimize_futures.expr_cast(cx, node)
  local fn = concretize(cx, optimize_futures.expr(cx, node.fn))
  local arg = optimize_futures.expr(cx, node.arg)
  local arg_type = std.as_read(arg.expr_type)

  local expr_type = node.expr_type
  if std.is_future(arg_type) then
    expr_type = std.future(expr_type)
  end

  node = node {
    fn = fn,
    arg = arg,
    expr_type = expr_type,
  }

  if std.is_future(arg_type) then
    return lift_cast_to_futures(node)
  end
  return node
end

function optimize_futures.expr_ctor_list_field(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_ctor_rec_field(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_ctor_field(cx, node)
  if node:is(ast.typed.expr.CtorListField) then
    return optimize_futures.expr_ctor_list_field(cx, node)
  elseif node:is(ast.typed.expr.CtorRecField) then
    return optimize_futures.expr_ctor_rec_field(cx, node)
  else
    assert(false)
  end
end

function optimize_futures.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return optimize_futures.expr_ctor_field(cx, field) end)
  return node { fields = fields }
end

function optimize_futures.expr_raw_fields(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node { region = region }
end

function optimize_futures.expr_raw_future(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local result = node {
    value = node.value,
    expr_type = std.future(node.expr_type),
  }
  if cx.task_is_leaf then
    -- If we're in a leaf task, immediate concretize the result so
    -- that it doesn't propagate, because escaping future values may
    -- cause unexpected task launches (e.g., on arithmetic).
    result = concretize(cx, result)
  end
  return result
end

function optimize_futures.expr_raw_physical(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node { region = region }
end

function optimize_futures.expr_raw_value(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_isnull(cx, node)
  local pointer = concretize(cx, optimize_futures.expr(cx, node.pointer))
  return node { pointer = pointer }
end

function optimize_futures.expr_null(cx, node)
  return node
end

function optimize_futures.expr_dynamic_cast(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_static_cast(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_unsafe_cast(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_ispace(cx, node)
  local extent = concretize(cx, optimize_futures.expr(cx, node.extent))
  local start = node.start and
    concretize(cx, optimize_futures.expr(cx, node.start))
  return node {
    extent = extent,
    start = start,
  }
end

function optimize_futures.expr_region(cx, node)
  local ispace = concretize(cx, optimize_futures.expr(cx, node.ispace))
  return node { ispace = ispace }
end

function optimize_futures.expr_partition(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  local coloring = concretize(cx, optimize_futures.expr(cx, node.coloring))
  return node {
    region = region,
    coloring = coloring,
  }
end

function optimize_futures.expr_partition_equal(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  local colors = concretize(cx, optimize_futures.expr(cx, node.colors))
  return node {
    region = region,
    colors = colors,
  }
end

function optimize_futures.expr_partition_by_field(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  local colors = concretize(cx, optimize_futures.expr(cx, node.colors))
  return node {
    region = region,
    colors = colors,
  }
end

function optimize_futures.expr_partition_by_restriction(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  local transform = concretize(cx, optimize_futures.expr(cx, node.transform))
  local extent = concretize(cx, optimize_futures.expr(cx, node.extent))
  local colors = concretize(cx, optimize_futures.expr(cx, node.colors))
  return node {
    region = region,
    transform = transform,
    extent = extent,
    colors = colors,
  }
end

function optimize_futures.expr_image(cx, node)
  local parent = concretize(cx, optimize_futures.expr(cx, node.parent))
  local partition = concretize(cx, optimize_futures.expr(cx, node.partition))
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node {
    parent = parent,
    partition = partition,
    region = region,
  }
end

function optimize_futures.expr_preimage(cx, node)
  local parent = concretize(cx, optimize_futures.expr(cx, node.parent))
  local partition = concretize(cx, optimize_futures.expr(cx, node.partition))
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node {
    parent = parent,
    partition = partition,
    region = region,
  }
end

function optimize_futures.expr_cross_product(cx, node)
  local args = node.args:map(
    function(arg) return concretize(cx, optimize_futures.expr(cx, arg)) end)
  return node {
    args = args,
  }
end

function optimize_futures.expr_cross_product_array(cx, node)
  return node {
    lhs = concretize(cx, optimize_futures.expr(cx, node.lhs)),
    disjointness = node.disjointness,
    colorings = concretize(cx, optimize_futures.expr(cx, node.colorings)),
  }
end

function optimize_futures.expr_list_slice_partition(cx, node)
  local partition = concretize(cx, optimize_futures.expr(cx, node.partition))
  local indices = concretize(cx, optimize_futures.expr(cx, node.indices))
  return node {
    partition = partition,
    indices = indices,
  }
end

function optimize_futures.expr_list_duplicate_partition(cx, node)
  local partition = concretize(cx, optimize_futures.expr(cx, node.partition))
  local indices = concretize(cx, optimize_futures.expr(cx, node.indices))
  return node {
    partition = partition,
    indices = indices,
  }
end

function optimize_futures.expr_list_slice_cross_product(cx, node)
  local product = concretize(cx, optimize_futures.expr(cx, node.product))
  local indices = concretize(cx, optimize_futures.expr(cx, node.indices))
  return node {
    product = product,
    indices = indices,
  }
end

function optimize_futures.expr_list_cross_product(cx, node)
  local lhs = concretize(cx, optimize_futures.expr(cx, node.lhs))
  local rhs = concretize(cx, optimize_futures.expr(cx, node.rhs))
  return node {
    lhs = lhs,
    rhs = rhs,
    shallow = node.shallow,
  }
end

function optimize_futures.expr_list_cross_product_complete(cx, node)
  local lhs = concretize(cx, optimize_futures.expr(cx, node.lhs))
  local product = concretize(cx, optimize_futures.expr(cx, node.product))
  return node {
    lhs = lhs,
    product = product,
  }
end

function optimize_futures.expr_list_phase_barriers(cx, node)
  local product = concretize(cx, optimize_futures.expr(cx, node.product))
  return node {
    product = product,
  }
end

function optimize_futures.expr_list_invert(cx, node)
  local rhs = concretize(cx, optimize_futures.expr(cx, node.rhs))
  local product = concretize(cx, optimize_futures.expr(cx, node.product))
  local barriers = concretize(cx, optimize_futures.expr(cx, node.barriers))
  return node {
    rhs = rhs,
    product = product,
    barriers = barriers,
  }
end

function optimize_futures.expr_list_range(cx, node)
  local start = concretize(cx, optimize_futures.expr(cx, node.start))
  local stop = concretize(cx, optimize_futures.expr(cx, node.stop))
  return node {
    start = start,
    stop = stop,
  }
end

function optimize_futures.expr_list_ispace(cx, node)
  local ispace = concretize(cx, optimize_futures.expr(cx, node.ispace))
  return node {
    ispace = ispace,
  }
end

function optimize_futures.expr_list_from_element(cx, node)
  local list = concretize(cx, optimize_futures.expr(cx, node.list))
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    list = list,
    value = value,
  }
end

function optimize_futures.expr_phase_barrier(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_dynamic_collective(cx, node)
  local arrivals = concretize(cx, optimize_futures.expr(cx, node.arrivals))
  return node {
    arrivals = arrivals,
  }
end

function optimize_futures.expr_dynamic_collective_get_result(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    value = value,
    expr_type = std.future(node.expr_type),
  }
end

function optimize_futures.expr_advance(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_adjust(cx, node)
  local barrier = concretize(cx, optimize_futures.expr(cx, node.barrier))
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    barrier = barrier,
    value = value,
  }
end

function optimize_futures.expr_arrive(cx, node)
  local barrier = concretize(cx, optimize_futures.expr(cx, node.barrier))
  local value = node.value and optimize_futures.expr(cx, node.value)
  return node {
    barrier = barrier,
    value = value,
  }
end

function optimize_futures.expr_await(cx, node)
  local barrier = concretize(cx, optimize_futures.expr(cx, node.barrier))
  return node {
    barrier = barrier,
  }
end

function optimize_futures.expr_copy(cx, node)
  local src = concretize(cx, optimize_futures.expr_region_root(cx, node.src))
  local dst = concretize(cx, optimize_futures.expr_region_root(cx, node.dst))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(cx, optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    src = src,
    dst = dst,
    conditions = conditions,
  }
end

function optimize_futures.expr_fill(cx, node)
  local dst = concretize(cx, optimize_futures.expr_region_root(cx, node.dst))
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(cx, optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    dst = dst,
    value = value,
    conditions = conditions,
  }
end

function optimize_futures.expr_acquire(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(cx, optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    region = region,
    conditions = conditions,
  }
end

function optimize_futures.expr_release(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(cx, optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    region = region,
    conditions = conditions,
  }
end

function optimize_futures.expr_attach_hdf5(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  local filename = concretize(cx, optimize_futures.expr(cx, node.filename))
  local mode = concretize(cx, optimize_futures.expr(cx, node.mode))
  return node {
    region = region,
    filename = filename,
    mode = mode,
  }
end

function optimize_futures.expr_detach_hdf5(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_allocate_scratch_fields(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_with_scratch_fields(cx, node)
  local region = concretize(cx, optimize_futures.expr_region_root(cx, node.region))
  local field_ids = concretize(cx, optimize_futures.expr(cx, node.field_ids))
  return node {
    region = region,
    field_ids = field_ids,
  }
end

local function lift_unary_op_to_futures(node)
  local rhs_type = std.as_read(node.rhs.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local task = task_helper.make_unary_task(node.op, rhs_type, expr_type)

  return ast.typed.expr.Call {
    fn = ast.typed.expr.Function {
      value = task,
      expr_type = task:get_type(),
      annotations = ast.default_annotations(),
      span = node.span,
    },
    args = terralib.newlist({
      node.rhs,
    }),
    conditions = terralib.newlist(),
    predicate = false,
    predicate_else_value = false,
    replicable = false,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function optimize_futures.expr_unary(cx, node)
  local rhs = optimize_futures.expr(cx, node.rhs)
  local rhs_type = std.as_read(rhs.expr_type)

  local expr_type = node.expr_type
  if std.is_future(rhs_type) then
    expr_type = std.future(expr_type)
  end

  node = node {
    rhs = rhs,
    expr_type = expr_type,
  }

  if std.is_future(rhs_type) then
    return lift_unary_op_to_futures(node)
  end
  return node
end

local function lift_binary_op_to_futures(node)
  local lhs_type = std.as_read(node.lhs.expr_type)
  local rhs_type = std.as_read(node.rhs.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local task = task_helper.make_binary_task(node.op, lhs_type, rhs_type, expr_type)

  return ast.typed.expr.Call {
    fn = ast.typed.expr.Function {
      value = task,
      expr_type = task:get_type(),
      annotations = ast.default_annotations(),
      span = node.span,
    },
    args = terralib.newlist({
      node.lhs,
      node.rhs,
    }),
    conditions = terralib.newlist(),
    predicate = false,
    predicate_else_value = false,
    replicable = false,
    expr_type = expr_type,
    annotations = node.annotations,
    span = node.span,
  }
end

function optimize_futures.expr_binary(cx, node)
  local lhs = optimize_futures.expr(cx, node.lhs)
  local lhs_type = std.as_read(lhs.expr_type)
  local rhs = optimize_futures.expr(cx, node.rhs)
  local rhs_type = std.as_read(rhs.expr_type)

  local expr_type = node.expr_type
  if std.is_future(lhs_type) or std.is_future(rhs_type) then
    expr_type = std.future(expr_type)
  end

  node = node {
    lhs = lhs,
    rhs = rhs,
    expr_type = expr_type,
  }

  if std.is_future(lhs_type) or std.is_future(rhs_type) then
    return lift_binary_op_to_futures(node)
  end
  return node
end

function optimize_futures.expr_deref(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_address_of(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_import_ispace(cx, node)
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_import_region(cx, node)
  local ispace    = concretize(cx, optimize_futures.expr(cx, node.ispace))
  local value     = concretize(cx, optimize_futures.expr(cx, node.value))
  local field_ids = concretize(cx, optimize_futures.expr(cx, node.field_ids))
  return node {
    ispace = ispace,
    value = value,
    field_ids = field_ids,
  }
end

function optimize_futures.expr_import_partition(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  local colors = concretize(cx, optimize_futures.expr(cx, node.colors))
  local value  = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    region = region,
    colors = colors,
    value = value,
  }
end

function optimize_futures.expr_import_cross_product(cx, node)
  local partitions = node.partitions:map(function(p) return concretize(cx, optimize_futures.expr(cx, p)) end)
  local colors  = concretize(cx, optimize_futures.expr(cx, node.colors))
  local value  = concretize(cx, optimize_futures.expr(cx, node.value))
  return node {
    partitions = partitions,
    colors = colors,
    value = value,
  }
end

function optimize_futures.expr_projection(cx, node)
  local region = concretize(cx, optimize_futures.expr(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    return optimize_futures.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
    return node

  elseif node:is(ast.typed.expr.Global) then
    return node

  elseif node:is(ast.typed.expr.Function) then
    return node

  elseif node:is(ast.typed.expr.FieldAccess) then
    return optimize_futures.expr_field_access(cx, node)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return optimize_futures.expr_index_access(cx, node)

  elseif node:is(ast.typed.expr.MethodCall) then
    return optimize_futures.expr_method_call(cx, node)

  elseif node:is(ast.typed.expr.Call) then
    return optimize_futures.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    return optimize_futures.expr_cast(cx, node)

  elseif node:is(ast.typed.expr.Ctor) then
    return optimize_futures.expr_ctor(cx, node)

  elseif node:is(ast.typed.expr.RawContext) then
    return node

  elseif node:is(ast.typed.expr.RawFields) then
    return optimize_futures.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.expr.RawFuture) then
    return optimize_futures.expr_raw_future(cx, node)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return optimize_futures.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return node

  elseif node:is(ast.typed.expr.RawTask) then
    return node

  elseif node:is(ast.typed.expr.RawValue) then
    return optimize_futures.expr_raw_value(cx, node)

  elseif node:is(ast.typed.expr.Isnull) then
    return optimize_futures.expr_isnull(cx, node)

  elseif node:is(ast.typed.expr.Null) then
    return optimize_futures.expr_null(cx, node)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return optimize_futures.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.expr.StaticCast) then
    return optimize_futures.expr_static_cast(cx, node)

  elseif node:is(ast.typed.expr.UnsafeCast) then
    return optimize_futures.expr_unsafe_cast(cx, node)

  elseif node:is(ast.typed.expr.Ispace) then
    return optimize_futures.expr_ispace(cx, node)

  elseif node:is(ast.typed.expr.Region) then
    return optimize_futures.expr_region(cx, node)

  elseif node:is(ast.typed.expr.Partition) then
    return optimize_futures.expr_partition(cx, node)

  elseif node:is(ast.typed.expr.PartitionEqual) then
    return optimize_futures.expr_partition_equal(cx, node)

  elseif node:is(ast.typed.expr.PartitionByField) then
    return optimize_futures.expr_partition_by_field(cx, node)

  elseif node:is(ast.typed.expr.PartitionByRestriction) then
    return optimize_futures.expr_partition_by_restriction(cx, node)

  elseif node:is(ast.typed.expr.Image) then
    return optimize_futures.expr_image(cx, node)

  elseif node:is(ast.typed.expr.Preimage) then
    return optimize_futures.expr_preimage(cx, node)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return optimize_futures.expr_cross_product(cx, node)

  elseif node:is(ast.typed.expr.CrossProductArray) then
    return optimize_futures.expr_cross_product_array(cx, node)

  elseif node:is(ast.typed.expr.ListSlicePartition) then
    return optimize_futures.expr_list_slice_partition(cx, node)

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return optimize_futures.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.typed.expr.ListSliceCrossProduct) then
    return optimize_futures.expr_list_slice_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProduct) then
    return optimize_futures.expr_list_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProductComplete) then
    return optimize_futures.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.typed.expr.ListPhaseBarriers) then
    return optimize_futures.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.typed.expr.ListInvert) then
    return optimize_futures.expr_list_invert(cx, node)

  elseif node:is(ast.typed.expr.ListRange) then
    return optimize_futures.expr_list_range(cx, node)

  elseif node:is(ast.typed.expr.ListIspace) then
    return optimize_futures.expr_list_ispace(cx, node)

  elseif node:is(ast.typed.expr.ListFromElement) then
    return optimize_futures.expr_list_from_element(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return optimize_futures.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollective) then
    return optimize_futures.expr_dynamic_collective(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return optimize_futures.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return optimize_futures.expr_advance(cx, node)

  elseif node:is(ast.typed.expr.Adjust) then
    return optimize_futures.expr_adjust(cx, node)

  elseif node:is(ast.typed.expr.Arrive) then
    return optimize_futures.expr_arrive(cx, node)

  elseif node:is(ast.typed.expr.Await) then
    return optimize_futures.expr_await(cx, node)

  elseif node:is(ast.typed.expr.Copy) then
    return optimize_futures.expr_copy(cx, node)

  elseif node:is(ast.typed.expr.Fill) then
    return optimize_futures.expr_fill(cx, node)

  elseif node:is(ast.typed.expr.Acquire) then
    return optimize_futures.expr_acquire(cx, node)

  elseif node:is(ast.typed.expr.Release) then
    return optimize_futures.expr_release(cx, node)

  elseif node:is(ast.typed.expr.AttachHDF5) then
    return optimize_futures.expr_attach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.DetachHDF5) then
    return optimize_futures.expr_detach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.AllocateScratchFields) then
    return optimize_futures.expr_allocate_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.WithScratchFields) then
    return optimize_futures.expr_with_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.Unary) then
    return optimize_futures.expr_unary(cx, node)

  elseif node:is(ast.typed.expr.Binary) then
    return optimize_futures.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.Deref) then
    return optimize_futures.expr_deref(cx, node)

  elseif node:is(ast.typed.expr.AddressOf) then
    return optimize_futures.expr_address_of(cx, node)

  elseif node:is(ast.typed.expr.ImportIspace) then
    return optimize_futures.expr_import_ispace(cx, node)

  elseif node:is(ast.typed.expr.ImportRegion) then
    return optimize_futures.expr_import_region(cx, node)

  elseif node:is(ast.typed.expr.ImportPartition) then
    return optimize_futures.expr_import_partition(cx, node)

  elseif node:is(ast.typed.expr.ImportCrossProduct) then
    return optimize_futures.expr_import_cross_product(cx, node)

  elseif node:is(ast.typed.expr.Projection) then
    return optimize_futures.expr_projection(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function optimize_futures.block(cx, node)
  return node {
    stats = data.flatmap(
      function(stat) return optimize_futures.stat(cx, stat) end,
      node.stats),
  }
end

function optimize_futures.stat_if(cx, node)
  local cx = cx:new_stat_scope()
  local cond = concretize(cx, optimize_futures.expr(cx, node.cond))
  local then_block = optimize_futures.block(cx, node.then_block)
  local else_block = node.else_block
  for idx = #node.elseif_blocks, 1, -1 do
    local elseif_block = node.elseif_blocks[idx]
    else_block = ast.typed.Block {
      stats = terralib.newlist({
        ast.typed.stat.If {
          cond = elseif_block.cond,
          then_block = elseif_block.block,
          elseif_blocks = terralib.newlist(),
          else_block = else_block,
          span = elseif_block.span,
          annotations = elseif_block.annotations,
        }
      }),
      span = elseif_block.span,
    }
  end
  local else_block = optimize_futures.block(cx, else_block)

  return cx:add_spill(
    node {
      cond = cond,
      then_block = then_block,
      elseif_blocks = terralib.newlist(),
      else_block = else_block,
    }
  ):get_spills()
end

function optimize_futures.stat_elseif(cx, node)
  -- Should be unreachable
  assert(false)
end

function optimize_futures.stat_while(cx, node)
  -- This is guaranteed by the normalizer
  assert(node.cond:is(ast.typed.expr.ID))
  return terralib.newlist({
    node {
      cond = concretize(cx, optimize_futures.expr(cx, node.cond)),
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_for_num(cx, node)
  local cx = cx:new_stat_scope()
  local values = node.values:map(function(value)
      return concretize(cx, optimize_futures.expr(cx, value))
    end)
  local block = optimize_futures.block(cx, node.block)
  return cx:add_spill(
    node {
      values = values,
      block = block,
    }
  ):get_spills()
end

function optimize_futures.stat_for_list(cx, node)
  local cx = cx:new_stat_scope()
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local block = optimize_futures.block(cx, node.block)
  return cx:add_spill(
    node {
      value = value,
      block = block,
    }
  ):get_spills()
end

function optimize_futures.stat_repeat(cx, node)
  -- Should be unreachable
  assert(false)
end

function optimize_futures.stat_must_epoch(cx, node)
  return terralib.newlist({
    node {
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_block(cx, node)
  return terralib.newlist({
    node {
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_index_launch_num(cx, node)
  local cx = cx:new_stat_scope()
  local values = node.values:map(
    function(value) return concretize(cx, optimize_futures.expr(cx, value)) end)
  local preamble = terralib.newlist()
  node.preamble:map(function(stat)
    preamble:insertall(optimize_futures.stat(cx, stat))
  end)
  local call = optimize_futures.expr(cx, node.call)
  local reduce_lhs = node.reduce_lhs and
    optimize_futures.expr(cx, node.reduce_lhs)
  local reduce_task = false

  if call:is(ast.typed.expr.Call) then
    local args = terralib.newlist()
    for i, arg in ipairs(call.args) do
      if std.is_future(std.as_read(arg.expr_type)) and
         not node.args_provably.invariant[i]
      then
        arg = concretize(cx, arg)
      elseif std.is_future(std.as_read(arg.expr_type)) then
        arg = normalize_compound_expr(cx, arg)
      end
      args:insert(arg)
    end
    call.args = args

    if reduce_lhs then
      local call_type = std.as_read(call.expr_type)
      local reduce_type = std.as_read(reduce_lhs.expr_type)
      if std.is_future(call_type) and not std.is_future(reduce_type) then
        call.expr_type = call_type.result_type
      end
      reduce_task = task_helper.make_binary_task(node.reduce_op, reduce_type, call_type, reduce_type)
    end

  elseif call:is(ast.typed.expr.Fill) then
    local region = call.dst.region
    local value = call.value
    if std.is_future(std.as_read(value.expr_type)) then
      value = concretize(cx, value)
    end
    if std.is_future(std.as_read(region.expr_type)) then
      region = concretize(cx, region)
    end
    call = call {
      dst = call.dst { region = region },
      value = value,
    }

  else
    assert(false)
  end

  return cx:add_spill(
    node {
      values = values,
      preamble = preamble,
      call = call,
      reduce_lhs = reduce_lhs,
      reduce_task = reduce_task,
    }
  ):get_spills()
end

function optimize_futures.stat_index_launch_list(cx, node)
  local cx = cx:new_stat_scope()
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  local preamble = terralib.newlist()
  node.preamble:map(function(stat)
    preamble:insertall(optimize_futures.stat(cx, stat))
  end)
  local call = optimize_futures.expr(cx, node.call)
  local reduce_lhs = node.reduce_lhs and
    optimize_futures.expr(cx, node.reduce_lhs)
  local reduce_task = false

  if call:is(ast.typed.expr.Call) then
    local args = terralib.newlist()
    for i, arg in ipairs(call.args) do
      if std.is_future(std.as_read(arg.expr_type)) and
         not node.args_provably.invariant[i]
      then
        arg = concretize(cx, arg)
      elseif std.is_future(std.as_read(arg.expr_type)) then
        arg = normalize_compound_expr(cx, arg)
      end
      args:insert(arg)
    end
    call.args = args

    if reduce_lhs then
      local call_type = std.as_read(call.expr_type)
      local reduce_type = std.as_read(reduce_lhs.expr_type)
      if std.is_future(call_type) and not std.is_future(reduce_type) then
        call.expr_type = call_type.result_type
      end
      reduce_task = task_helper.make_binary_task(node.reduce_op, reduce_type, call_type, reduce_type)
    end

  elseif call:is(ast.typed.expr.Fill) then
    local region = call.dst.region
    local value = call.value
    if std.is_future(std.as_read(value.expr_type)) then
      value = concretize(cx, value)
    end
    if std.is_future(std.as_read(region.expr_type)) then
      region = concretize(cx, region)
    end
    call = call {
      dst = call.dst { region = region },
      value = value,
    }

  else
    assert(false)
  end

  return cx:add_spill(
    node {
      value = value,
      preamble = preamble,
      call = call,
      reduce_lhs = reduce_lhs,
      reduce_task = reduce_task,
    }
  ):get_spills()
end

function optimize_futures.stat_var(cx, node)
  local cx = cx:new_stat_scope()
  local stats = terralib.newlist()

  local value = node.value
  local value_type = node.type

  local new_symbol = node.symbol
  if cx:is_var_future(node.symbol) then
    new_symbol = cx:symbol(node.symbol)
  end

  local new_type = value_type
  if cx:is_var_future(node.symbol) then
    new_type = std.future(value_type)
  end

  local new_value = value
  if value then
    if cx:is_var_future(node.symbol) then
      new_value = promote(cx, optimize_futures.expr(cx, value), new_type)
    else
      new_value = concretize(cx, optimize_futures.expr(cx, value))
    end
  else
    if cx:is_var_future(node.symbol) then
      -- This is an uninitialized future. Create an empty value and
      -- use it to initialize the future, otherwise future will hold
      -- an uninitialized pointer.
      local empty_symbol = std.newsymbol(value_type)
      local empty_var = node {
        symbol = empty_symbol,
        type = value_type,
        value = false,
      }
      local empty_ref = ast.typed.expr.ID {
        value = empty_symbol,
        expr_type = std.rawref(&value_type),
        annotations = ast.default_annotations(),
        span = node.span,
      }

      cx:add_spill(empty_var)
      new_value = promote(cx, empty_ref, new_type)
    end
  end

  return cx:add_spill(node {
    symbol = new_symbol,
    type = new_type,
    value = new_value,
  }):get_spills()
end

function optimize_futures.stat_var_unpack(cx, node)
  local cx = cx:new_stat_scope()
  local value = concretize(cx, optimize_futures.expr(cx, node.value))
  return cx:add_spill(
    node {
      value = value,
    }
  ):get_spills()
end

function optimize_futures.stat_return(cx, node)
  local cx = cx:new_stat_scope()
  local value = node.value and concretize(cx, optimize_futures.expr(cx, node.value))
  return cx:add_spill(node { value = value }):get_spills()
end

function optimize_futures.stat_break(cx, node)
  return terralib.newlist({node})
end

local function unwrap_access(node)
  if node:is(ast.typed.expr.FieldAccess) or node:is(ast.typed.expr.IndexAccess) then
    return unwrap_access(node.value)
  end
  return node
end

local function rewrap_access(node, replacement)
  if node:is(ast.typed.expr.FieldAccess) or node:is(ast.typed.expr.IndexAccess) then
    return node { value = rewrap_access(node.value, replacement) }
  end
  return replacement
end

local function is_future_modify_assignment(node)
  return (node:is(ast.typed.expr.FieldAccess) or node:is(ast.typed.expr.IndexAccess)) and
    unwrap_access(node):is(ast.typed.expr.FutureGetResult)
end

local function handle_future_modify_assignment(cx, lhs, rhs)
  local lhs_value = unwrap_access(lhs)
  local lhs_type = std.as_read(lhs_value.expr_type)
  local symbol = std.newsymbol(lhs_type)

  local symbol_value = ast.typed.expr.ID {
    value = symbol,
    expr_type = std.rawref(&lhs_type),
    annotations = ast.default_annotations(),
    span = lhs.span,
  }

  return cx:add_spill(
    ast.typed.stat.Var {
      symbol = symbol,
      type = lhs_type,
      value = lhs_value,
      annotations = ast.default_annotations(),
      span = lhs.span,
    }):add_spill(
    ast.typed.stat.Assignment {
      lhs = rewrap_access(lhs, symbol_value),
      rhs = rhs,
      metadata = false,
      annotations = ast.default_annotations(),
      span = lhs.span,
    }):add_spill(
    ast.typed.stat.Assignment {
      lhs = lhs_value.value,
      rhs = ast.typed.expr.Future {
        value = symbol_value,
        expr_type = std.future(lhs_type),
        annotations = ast.default_annotations(),
        span = lhs.span,
      },
      metadata = false,
      annotations = ast.default_annotations(),
      span = lhs.span,
    }):get_spills()
end

function optimize_futures.stat_assignment(cx, node)
  local cx = cx:new_stat_scope()
  local lhs = optimize_futures.expr(cx, node.lhs)
  local rhs = optimize_futures.expr(cx, node.rhs)

  local normalized_rhs
  local lhs_type = std.as_read(lhs.expr_type)
  if std.is_future(lhs_type) then
    normalized_rhs = promote(cx, rhs, lhs_type)
  else
    normalized_rhs = concretize(cx, rhs)
  end

  -- Hack: Can't write directly to the field of a future; must write
  -- an entire struct. Generate an updated struct and assign it.
  if is_future_modify_assignment(lhs) then
    return handle_future_modify_assignment(cx, lhs, normalized_rhs)
  end

  return cx:add_spill(
    node {
      lhs = lhs,
      rhs = normalized_rhs,
    }
  ):get_spills()
end

function optimize_futures.stat_reduce(cx, node)
  local cx = cx:new_stat_scope()
  local lhs = optimize_futures.expr(cx, node.lhs)
  local rhs = optimize_futures.expr(cx, node.rhs)

  local normalized_rhs
  local lhs_type = std.as_read(lhs.expr_type)
  if std.is_future(lhs_type) then
    normalized_rhs = promote(cx, rhs, lhs_type)

    return cx:add_spill(
      ast.typed.stat.Assignment {
        lhs = lhs,
        rhs = lift_binary_op_to_futures(
          ast.typed.expr.Binary {
            op = node.op,
            lhs = lhs,
            rhs = normalized_rhs,
            expr_type = lhs_type,
            annotations = ast.default_annotations(),
            span = node.span,
          }
        ),
        metadata = false,
        annotations = ast.default_annotations(),
        span = node.span,
      }
    ):get_spills()
  else
    normalized_rhs = concretize(cx, rhs)
  end

  return cx:add_spill(
    node {
      lhs = lhs,
      rhs = normalized_rhs,
    }
  ):get_spills()
end

function optimize_futures.stat_expr(cx, node)
  local cx = cx:new_stat_scope()
  local expr = optimize_futures.expr(cx, node.expr)
  return cx:add_spill(
    node {
      expr = expr,
    }
  ):get_spills()
end

function optimize_futures.stat_raw_delete(cx, node)
  local cx = cx:new_stat_scope()
  local value = optimize_futures.expr(cx, node.value)
  return cx:add_spill(
    node {
      value = value,
    }
  ):get_spills()
end

function optimize_futures.stat_fence(cx, node)
  return terralib.newlist({node})
end

function optimize_futures.stat_parallel_prefix(cx, node)
  local cx = cx:new_stat_scope()
  local dir = optimize_futures.expr(cx, node.dir)
  return cx:add_spill(
    node {
      dir = dir,
    }
  ):get_spills()
end

function optimize_futures.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return optimize_futures.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return optimize_futures.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return optimize_futures.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return optimize_futures.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return optimize_futures.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return optimize_futures.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return optimize_futures.stat_block(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return optimize_futures.stat_index_launch_num(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return optimize_futures.stat_index_launch_list(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return optimize_futures.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return optimize_futures.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return optimize_futures.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return optimize_futures.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    return optimize_futures.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return optimize_futures.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return optimize_futures.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    return optimize_futures.stat_raw_delete(cx, node)

  elseif node:is(ast.typed.stat.Fence) then
    return optimize_futures.stat_fence(cx, node)

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    return optimize_futures.stat_parallel_prefix(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function optimize_futures.top_task_param(cx, param)
  if cx:is_var_future(param.symbol) then
    local new_type = std.future(param.param_type)

    local new_symbol = cx:symbol(param.symbol)

    local cx = cx:new_stat_scope()
    local value = promote(cx,
      ast.typed.expr.ID {
        value = param.symbol,
        expr_type = std.rawref(&param.param_type),
        annotations = param.annotations,
        span = param.span,
      },
      new_type)
    return cx:add_spill(ast.typed.stat.Var {
      symbol = new_symbol,
      type = new_type,
      value = value,
      annotations = param.annotations,
      span = param.span,
    }):get_spills()
  end
end

function optimize_futures.top_task_params(cx, node)
  local actions = terralib.newlist()
  for _, param in ipairs(node.params) do
    local param_actions =
      optimize_futures.top_task_param(cx, param)
    if param_actions then actions:insertall(param_actions) end
  end
  return actions
end

function optimize_futures.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope(node.config_options.leaf)
  analyze_var_flow.block(cx, node.body)
  compute_var_futures(cx)
  compute_var_symbols(cx)
  local actions = optimize_futures.top_task_params(cx, node)
  local body = optimize_futures.block(cx, node.body)

  if #actions > 0 then
    actions:insertall(body.stats)
    body = body { stats = actions }
  end

  return node {
    body = body
  }
end

function optimize_futures.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return optimize_futures.top_task(cx, node)

  else
    return node
  end
end

function optimize_futures.entry(node)
  local cx = context.new_global_scope()
  return optimize_futures.top(cx, node)
end

optimize_futures.pass_name = "optimize_futures"

return optimize_futures
