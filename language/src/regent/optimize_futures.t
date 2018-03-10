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

-- Legion Future Optimizer
--
-- This pass changes all non-void task calls to return
-- futures. Wherever possible, this pass attempts to work with futures
-- directly rather than blocking in order to obtain a concrete value.

local ast = require("regent/ast")
local data = require("common/data")
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

function context:new_task_scope()
  local cx = {
    var_flows = {},
    var_futures = {},
    var_symbols = {},
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:get_flow(v)
  if not rawget(self.var_flows, v) then
    self.var_flows[v] = {}
  end
  return self.var_flows[v]
end

function context:is_var_future(v)
  return self.var_futures[v]
end

function context:symbol(v)
  return self.var_symbols[v]
end

local analyze_var_flow = {}

function flow_future()
  return {[true] = true} -- Represents an unconditional future r-value
end

function flow_var(v)
  return {[v] = true} -- Represents a variable
end

function flow_future_into(cx, lhs) -- Unconditionally flow future into l-value
  if lhs then
    for v, _ in pairs(lhs) do
      local var_flow = cx:get_flow(v)
      var_flow[true] = true
    end
  end
end

function flow_value_into_var(cx, symbol, value) -- Flow r-value into variable
  local var_flow = cx:get_flow(symbol)
  if value then
    for v, _ in pairs(value) do
      var_flow[v] = true
    end
  end
end

function flow_value_into(cx, lhs, rhs) -- Flow r-value into l-value
  if lhs and rhs then
    for lhv, _ in pairs(lhs) do
      local lhv_flow = cx:get_flow(lhv)
      for rhv, _ in pairs(rhs) do
        lhv_flow[rhv] = true
      end
    end
  end
end

function meet_flow(...)
  local flow = {}
  for _, a in ipairs({...}) do
    if a then
      for v, _ in pairs(a) do
        flow[v] = true
      end
    end
  end
  return flow
end

function analyze_var_flow.expr_id(cx, node)
  return flow_var(node.value)
end

function analyze_var_flow.expr_call(cx, node)
  if std.is_task(node.fn.value) and
    node.expr_type ~= terralib.types.unit
  then
    return flow_future()
  end
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

function analyze_var_flow.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    return analyze_var_flow.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) or
    node:is(ast.typed.expr.Function) or
    node:is(ast.typed.expr.FieldAccess) or
    node:is(ast.typed.expr.IndexAccess) or
    node:is(ast.typed.expr.MethodCall)
  then
    return nil

  elseif node:is(ast.typed.expr.Call) then
    return analyze_var_flow.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) or
    node:is(ast.typed.expr.Ctor) or
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
    return nil

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
    node:is(ast.typed.expr.WithScratchFields)
  then
    return nil

  elseif node:is(ast.typed.expr.Unary) then
    return analyze_var_flow.expr_unary(cx, node)

  elseif node:is(ast.typed.expr.Binary) then
    return analyze_var_flow.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.Deref) then
    return nil

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_var_flow.block(cx, node)
  node.stats:map(
    function(stat) return analyze_var_flow.stat(cx, stat) end)
end

function analyze_var_flow.stat_if(cx, node)
  analyze_var_flow.block(cx, node.then_block)
  node.elseif_blocks:map(
    function(block) return analyze_var_flow.stat_elseif(cx, block) end)
  analyze_var_flow.block(cx, node.else_block)
end

function analyze_var_flow.stat_elseif(cx, node)
  analyze_var_flow.block(cx, node.block)
end

function analyze_var_flow.stat_while(cx, node)
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
  local reduce_lhs = node.reduce_lhs and
    analyze_var_flow.expr(cx, node.reduce_lhs)
  flow_future_into(cx, reduce_lhs)
end

function analyze_var_flow.stat_index_launch_list(cx, node)
  local reduce_lhs = node.reduce_lhs and
    analyze_var_flow.expr(cx, node.reduce_lhs)
  flow_future_into(cx, reduce_lhs)
end

function analyze_var_flow.stat_var(cx, node)
  local value = node.value
  if value then
    value = analyze_var_flow.expr(cx, value)
  end
  flow_value_into_var(cx, node.symbol, value)
end

function analyze_var_flow.stat_assignment(cx, node)
  local lhs = analyze_var_flow.expr(cx, node.lhs)
  local rhs = analyze_var_flow.expr(cx, node.rhs)
  flow_value_into(cx, lhs, rhs)
end

function analyze_var_flow.stat_reduce(cx, node)
  local lhs = analyze_var_flow.expr(cx, node.lhs)
  local rhs = analyze_var_flow.expr(cx, node.rhs)
  flow_value_into(cx, lhs, rhs)
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

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local function compute_var_futures(cx)
  local inflow = {}
  for v1, flow in pairs(cx.var_flows) do
    for v2, _ in pairs(flow) do
      if not rawget(inflow, v2) then
        inflow[v2] = {}
      end
      inflow[v2][v1] = true
    end
  end

  local var_futures = cx.var_futures
  var_futures[true] = true
  repeat
    local changed = false
    for v1, _ in pairs(var_futures) do
      if rawget(inflow, v1) then
        for v2, _ in pairs(inflow[v1]) do
          if not rawget(var_futures, v2) then
            var_futures[v2] = true
            changed = true
          end
        end
      end
    end
  until not changed
end

local function compute_var_symbols(cx)
  for v, is_future in pairs(cx.var_futures) do
    if std.is_symbol(v) and is_future then
      assert(terralib.types.istype(v:hastype()) and not std.is_future(v:gettype()))
      cx.var_symbols[v] = std.newsymbol(std.future(v:gettype()), v:hasname())
    end
  end
end

local optimize_futures = {}

local function concretize(node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    return ast.typed.expr.FutureGetResult {
      value = node,
      expr_type = expr_type.result_type,
      annotations = node.annotations,
      span = node.span,
    }
  end
  return node
end

local function promote(node, expected_type)
  assert(std.is_future(expected_type))

  local expr_type = std.as_read(node.expr_type)
  if not std.is_future(expr_type) then
    return ast.typed.expr.Future {
      value = node,
      expr_type = expected_type,
      annotations = node.annotations,
      span = node.span,
    }
  elseif not std.type_eq(expr_type, expected_type) then
    -- FIXME: This requires a cast. For now, just concretize and re-promote.
    return promote(concretize(node), expected_type)
  end
  return node
end

function optimize_futures.expr_region_root(cx, node)
  local region = concretize(optimize_futures.expr(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_condition(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
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
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_index_access(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  local index = concretize(optimize_futures.expr(cx, node.index))
  return node {
    value = value,
    index = index,
  }
end

function optimize_futures.expr_method_call(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  local args = node.args:map(
    function(arg) return concretize(optimize_futures.expr(cx, arg)) end)
  return node {
    value = value,
    args = args,
  }
end

function optimize_futures.expr_call(cx, node)
  local fn = concretize(optimize_futures.expr(cx, node.fn))
  local args = node.args:map(
    function(arg) return optimize_futures.expr(cx, arg) end)
  if not std.is_task(node.fn.value) then
    args = args:map(
      function(arg) return concretize(arg) end)
  end
  local expr_type = node.expr_type
  if std.is_task(node.fn.value) and expr_type ~= terralib.types.unit then
    expr_type = std.future(expr_type)
  end

  return node {
    fn = fn,
    args = args,
    expr_type = expr_type,
  }
end

function optimize_futures.expr_cast(cx, node)
  local fn = concretize(optimize_futures.expr(cx, node.fn))
  local arg = concretize(optimize_futures.expr(cx, node.arg))
  return node {
    fn = fn,
    arg = arg,
  }
end

function optimize_futures.expr_ctor_list_field(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_ctor_rec_field(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
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
  local region = concretize(optimize_futures.expr(cx, node.region))
  return node { region = region }
end

function optimize_futures.expr_raw_physical(cx, node)
  local region = concretize(optimize_futures.expr(cx, node.region))
  return node { region = region }
end

function optimize_futures.expr_raw_value(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_isnull(cx, node)
  local pointer = concretize(optimize_futures.expr(cx, node.pointer))
  return node { pointer = pointer }
end

function optimize_futures.expr_null(cx, node)
  return node
end

function optimize_futures.expr_dynamic_cast(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_static_cast(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_unsafe_cast(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr_ispace(cx, node)
  local extent = concretize(optimize_futures.expr(cx, node.extent))
  local start = node.start and
    concretize(optimize_futures.expr(cx, node.start))
  return node {
    extent = extent,
    start = start,
  }
end

function optimize_futures.expr_region(cx, node)
  local ispace = concretize(optimize_futures.expr(cx, node.ispace))
  return node { ispace = ispace }
end

function optimize_futures.expr_partition(cx, node)
  local region = concretize(optimize_futures.expr(cx, node.region))
  local coloring = concretize(optimize_futures.expr(cx, node.coloring))
  return node {
    region = region,
    coloring = coloring,
  }
end

function optimize_futures.expr_partition_equal(cx, node)
  local region = concretize(optimize_futures.expr(cx, node.region))
  local colors = concretize(optimize_futures.expr(cx, node.colors))
  return node {
    region = region,
    colors = colors,
  }
end

function optimize_futures.expr_partition_by_field(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  local colors = concretize(optimize_futures.expr(cx, node.colors))
  return node {
    region = region,
    colors = colors,
  }
end

function optimize_futures.expr_image(cx, node)
  local parent = concretize(optimize_futures.expr(cx, node.parent))
  local partition = concretize(optimize_futures.expr(cx, node.partition))
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  return node {
    parent = parent,
    partition = partition,
    region = region,
  }
end

function optimize_futures.expr_preimage(cx, node)
  local parent = concretize(optimize_futures.expr(cx, node.parent))
  local partition = concretize(optimize_futures.expr(cx, node.partition))
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  return node {
    parent = parent,
    partition = partition,
    region = region,
  }
end

function optimize_futures.expr_cross_product(cx, node)
  local args = node.args:map(
    function(arg) return concretize(optimize_futures.expr(cx, arg)) end)
  return node {
    args = args,
  }
end

function optimize_futures.expr_cross_product_array(cx, node)
  return node {
    lhs = concretize(optimize_futures.expr(cx, node.lhs)),
    disjointness = node.disjointness,
    colorings = concretize(optimize_futures.expr(cx, node.colorings)),
  }
end

function optimize_futures.expr_list_slice_partition(cx, node)
  local partition = concretize(optimize_futures.expr(cx, node.partition))
  local indices = concretize(optimize_futures.expr(cx, node.indices))
  return node {
    partition = partition,
    indices = indices,
  }
end

function optimize_futures.expr_list_duplicate_partition(cx, node)
  local partition = concretize(optimize_futures.expr(cx, node.partition))
  local indices = concretize(optimize_futures.expr(cx, node.indices))
  return node {
    partition = partition,
    indices = indices,
  }
end

function optimize_futures.expr_list_slice_cross_product(cx, node)
  local product = concretize(optimize_futures.expr(cx, node.product))
  local indices = concretize(optimize_futures.expr(cx, node.indices))
  return node {
    product = product,
    indices = indices,
  }
end

function optimize_futures.expr_list_cross_product(cx, node)
  local lhs = concretize(optimize_futures.expr(cx, node.lhs))
  local rhs = concretize(optimize_futures.expr(cx, node.rhs))
  return node {
    lhs = lhs,
    rhs = rhs,
    shallow = node.shallow,
  }
end

function optimize_futures.expr_list_cross_product_complete(cx, node)
  local lhs = concretize(optimize_futures.expr(cx, node.lhs))
  local product = concretize(optimize_futures.expr(cx, node.product))
  return node {
    lhs = lhs,
    product = product,
  }
end

function optimize_futures.expr_list_phase_barriers(cx, node)
  local product = concretize(optimize_futures.expr(cx, node.product))
  return node {
    product = product,
  }
end

function optimize_futures.expr_list_invert(cx, node)
  local rhs = concretize(optimize_futures.expr(cx, node.rhs))
  local product = concretize(optimize_futures.expr(cx, node.product))
  local barriers = concretize(optimize_futures.expr(cx, node.barriers))
  return node {
    rhs = rhs,
    product = product,
    barriers = barriers,
  }
end

function optimize_futures.expr_list_range(cx, node)
  local start = concretize(optimize_futures.expr(cx, node.start))
  local stop = concretize(optimize_futures.expr(cx, node.stop))
  return node {
    start = start,
    stop = stop,
  }
end

function optimize_futures.expr_list_ispace(cx, node)
  local ispace = concretize(optimize_futures.expr(cx, node.ispace))
  return node {
    ispace = ispace,
  }
end

function optimize_futures.expr_list_from_element(cx, node)
  local list = concretize(optimize_futures.expr(cx, node.list))
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    list = list,
    value = value,
  }
end

function optimize_futures.expr_phase_barrier(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_dynamic_collective(cx, node)
  local arrivals = concretize(optimize_futures.expr(cx, node.arrivals))
  return node {
    arrivals = arrivals,
  }
end

function optimize_futures.expr_dynamic_collective_get_result(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    value = value,
    expr_type = std.future(node.expr_type),
  }
end

function optimize_futures.expr_advance(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_adjust(cx, node)
  local barrier = concretize(optimize_futures.expr(cx, node.barrier))
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    barrier = barrier,
    value = value,
  }
end

function optimize_futures.expr_arrive(cx, node)
  local barrier = concretize(optimize_futures.expr(cx, node.barrier))
  local value = node.value and optimize_futures.expr(cx, node.value)
  return node {
    barrier = barrier,
    value = value,
  }
end

function optimize_futures.expr_await(cx, node)
  local barrier = concretize(optimize_futures.expr(cx, node.barrier))
  return node {
    barrier = barrier,
  }
end

function optimize_futures.expr_copy(cx, node)
  local src = concretize(optimize_futures.expr_region_root(cx, node.src))
  local dst = concretize(optimize_futures.expr_region_root(cx, node.dst))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    src = src,
    dst = dst,
    conditions = conditions,
  }
end

function optimize_futures.expr_fill(cx, node)
  local dst = concretize(optimize_futures.expr_region_root(cx, node.dst))
  local value = concretize(optimize_futures.expr(cx, node.value))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    dst = dst,
    value = value,
    conditions = conditions,
  }
end

function optimize_futures.expr_acquire(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    region = region,
    conditions = conditions,
  }
end

function optimize_futures.expr_release(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  local conditions = node.conditions:map(
    function(condition)
      return concretize(optimize_futures.expr_condition(cx, condition))
    end)
  return node {
    region = region,
    conditions = conditions,
  }
end

function optimize_futures.expr_attach_hdf5(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  local filename = concretize(optimize_futures.expr(cx, node.filename))
  local mode = concretize(optimize_futures.expr(cx, node.mode))
  return node {
    region = region,
    filename = filename,
    mode = mode,
  }
end

function optimize_futures.expr_detach_hdf5(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_allocate_scratch_fields(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  return node {
    region = region,
  }
end

function optimize_futures.expr_with_scratch_fields(cx, node)
  local region = concretize(optimize_futures.expr_region_root(cx, node.region))
  local field_ids = concretize(optimize_futures.expr(cx, node.field_ids))
  return node {
    region = region,
    field_ids = field_ids,
  }
end

function optimize_futures.expr_unary(cx, node)
  local rhs = optimize_futures.expr(cx, node.rhs)
  local rhs_type = std.as_read(rhs.expr_type)

  local expr_type = node.expr_type
  if std.is_future(rhs_type) then
    expr_type = std.future(expr_type)
  end

  return node {
    rhs = rhs,
    expr_type = expr_type,
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

  return node {
    lhs = lhs,
    rhs = rhs,
    expr_type = expr_type,
  }
end

function optimize_futures.expr_deref(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    return optimize_futures.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
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

  elseif node:is(ast.typed.expr.RawPhysical) then
    return optimize_futures.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
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
  return terralib.newlist({
    node {
      cond = concretize(optimize_futures.expr(cx, node.cond)),
      then_block = optimize_futures.block(cx, node.then_block),
      elseif_blocks = node.elseif_blocks:map(
        function(block) return optimize_futures.stat_elseif(cx, block) end),
      else_block = optimize_futures.block(cx, node.else_block),
    }
  })
end

function optimize_futures.stat_elseif(cx, node)
  return node {
    cond = concretize(optimize_futures.expr(cx, node.cond)),
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_while(cx, node)
  return terralib.newlist({
    node {
      cond = concretize(optimize_futures.expr(cx, node.cond)),
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_for_num(cx, node)
  return terralib.newlist({
    node {
      values = node.values:map(
        function(value) return concretize(optimize_futures.expr(cx, value)) end),
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_for_list(cx, node)
  return terralib.newlist({
    node {
      value = concretize(optimize_futures.expr(cx, node.value)),
      block = optimize_futures.block(cx, node.block),
    }
  })
end

function optimize_futures.stat_repeat(cx, node)
  return terralib.newlist({
    node {
      block = optimize_futures.block(cx, node.block),
      until_cond = concretize(optimize_futures.expr(cx, node.until_cond)),
    }
  })
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
  local values = node.values:map(
    function(value) return concretize(optimize_futures.expr(cx, value)) end)
  local call = optimize_futures.expr(cx, node.call)
  local reduce_lhs = node.reduce_lhs and
    optimize_futures.expr(cx, node.reduce_lhs)

  local args = terralib.newlist()
  for i, arg in ipairs(call.args) do
    if std.is_future(std.as_read(arg.expr_type)) and
      not node.args_provably.invariant[i]
    then
      arg = concretize(arg)
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
  end

  return terralib.newlist({
    node {
      values = values,
      call = call,
      reduce_lhs = reduce_lhs,
    }
  })
end

function optimize_futures.stat_index_launch_list(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  local call = optimize_futures.expr(cx, node.call)
  local reduce_lhs = node.reduce_lhs and
    optimize_futures.expr(cx, node.reduce_lhs)

  local args = terralib.newlist()
  for i, arg in ipairs(call.args) do
    if std.is_future(std.as_read(arg.expr_type)) and
      not node.args_provably.invariant[i]
    then
      arg = concretize(arg)
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
  end

  return terralib.newlist({
    node {
      value = value,
      call = call,
      reduce_lhs = reduce_lhs,
    }
  })
end

function optimize_futures.stat_var(cx, node)
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
      new_value = promote(optimize_futures.expr(cx, value), new_type)
    else
      new_value = concretize(optimize_futures.expr(cx, value))
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

      new_value = promote(empty_ref, new_type)
      stats:insert(empty_var)
    end
  end

  stats:insert(node {
    symbol = new_symbol,
    type = new_type,
    value = new_value,
  })
  return stats
end

function optimize_futures.stat_var_unpack(cx, node)
  return terralib.newlist({
    node {
      value = concretize(optimize_futures.expr(cx, node.value)),
    }
  })
end

function optimize_futures.stat_return(cx, node)
  local value = node.value and concretize(optimize_futures.expr(cx, node.value))
  return terralib.newlist({node { value = value } })
end

function optimize_futures.stat_break(cx, node)
  return terralib.newlist({node})
end

local function unwrap_field_access(node)
  if node:is(ast.typed.expr.FieldAccess) then
    return unwrap_field_access(node.value)
  end
  return node
end

local function rewrap_field_access(node, replacement)
  if node:is(ast.typed.expr.FieldAccess) then
    return node { value = rewrap_field_access(node.value, replacement) }
  end
  return replacement
end

local function is_future_field_assignment(node)
  return node:is(ast.typed.expr.FieldAccess) and
    unwrap_field_access(node):is(ast.typed.expr.FutureGetResult)
end

local function handle_future_field_assignment(cx, lhs, rhs)
  local lhs_value = unwrap_field_access(lhs)
  local lhs_type = std.as_read(lhs_value.expr_type)
  local symbol = std.newsymbol(lhs_type)

  local symbol_value = ast.typed.expr.ID {
    value = symbol,
    expr_type = std.rawref(&lhs_type),
    annotations = ast.default_annotations(),
    span = lhs.span,
  }

  return terralib.newlist({
    ast.typed.stat.Var {
      symbol = symbol,
      type = lhs_type,
      value = lhs_value,
      annotations = ast.default_annotations(),
      span = lhs.span,
    },
    ast.typed.stat.Assignment {
      lhs = rewrap_field_access(lhs, symbol_value),
      rhs = rhs,
      annotations = ast.default_annotations(),
      span = lhs.span,
    },
    ast.typed.stat.Assignment {
      lhs = lhs_value.value,
      rhs = ast.typed.expr.Future {
        value = symbol_value,
        expr_type = std.future(lhs_type),
        annotations = ast.default_annotations(),
        span = lhs.span,
      },
      annotations = ast.default_annotations(),
      span = lhs.span,
    },
  })
end

function optimize_futures.stat_assignment(cx, node)
  local lhs = optimize_futures.expr(cx, node.lhs)
  local rhs = optimize_futures.expr(cx, node.rhs)

  local normalized_rhs
  local lhs_type = std.as_read(lhs.expr_type)
  if std.is_future(lhs_type) then
    normalized_rhs = promote(rhs, lhs_type)
  else
    normalized_rhs = concretize(rhs)
  end

  -- Hack: Can't write directly to the field of a future; must write
  -- an entire struct. Generate an updated struct and assign it.
  if is_future_field_assignment(lhs) then
    return handle_future_field_assignment(cx, lhs, normalized_rhs)
  end

  return terralib.newlist({
    node {
      lhs = lhs,
      rhs = normalized_rhs,
    }
  })
end

function optimize_futures.stat_reduce(cx, node)
  local lhs = optimize_futures.expr(cx, node.lhs)
  local rhs = optimize_futures.expr(cx, node.rhs)

  local normalized_rhs
  local lhs_type = std.as_read(lhs.expr_type)
  if std.is_future(lhs_type) then
    normalized_rhs = promote(rhs, lhs_type)
  else
    normalized_rhs = concretize(rhs)
  end

  return terralib.newlist({
    node {
      lhs = lhs,
      rhs = normalized_rhs,
    }
  })
end

function optimize_futures.stat_expr(cx, node)
  return terralib.newlist({
    node {
      expr = optimize_futures.expr(cx, node.expr),
    }
  })
end

function optimize_futures.stat_raw_delete(cx, node)
  return terralib.newlist({
    node {
      value = optimize_futures.expr(cx, node.value),
    }
  })
end

function optimize_futures.stat_fence(cx, node)
  return terralib.newlist({node})
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

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function optimize_futures.top_task_param(cx, param)
  if cx:is_var_future(param.symbol) then
    local new_type = std.future(param.param_type)

    local new_symbol = cx:symbol(param.symbol)

    local new_var = ast.typed.stat.Var {
      symbol = new_symbol,
      type = new_type,
      value = promote(
        ast.typed.expr.ID {
          value = param.symbol,
          expr_type = std.rawref(&param.param_type),
          annotations = param.annotations,
          span = param.span,
        },
        new_type),
      annotations = param.annotations,
      span = param.span,
    }

    return new_var
  end
end

function optimize_futures.top_task_params(cx, node)
  local actions = terralib.newlist()
  for _, param in ipairs(node.params) do
    local action = optimize_futures.top_task_param(cx, param)
    if action then actions:insert(action) end
  end
  return actions
end

function optimize_futures.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope()
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
