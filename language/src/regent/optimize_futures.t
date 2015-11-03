-- Copyright 2015 Stanford University, NVIDIA Corporation
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
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_task_scope()
  local cx = {
    var_flows = {},
    var_futures = {},
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

function context:is_future(v)
  return self.var_futures[v]
end

local analyze_var_flow = {}

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
  return {[node.value] = true}
end

function analyze_var_flow.expr_call(cx, node)
  if std.is_task(node.fn.value) and
    node.expr_type ~= terralib.types.unit
  then
    return {[true] = true}
  end
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

  elseif node:is(ast.typed.expr.Constant) then
    return nil

  elseif node:is(ast.typed.expr.Function) then
    return nil

  elseif node:is(ast.typed.expr.FieldAccess) then
    return nil

  elseif node:is(ast.typed.expr.IndexAccess) then
    return nil

  elseif node:is(ast.typed.expr.MethodCall) then
    return nil

  elseif node:is(ast.typed.expr.Call) then
    return analyze_var_flow.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    return nil

  elseif node:is(ast.typed.expr.Ctor) then
    return nil

  elseif node:is(ast.typed.expr.RawContext) then
    return nil

  elseif node:is(ast.typed.expr.RawFields) then
    return nil

  elseif node:is(ast.typed.expr.RawPhysical) then
    return nil

  elseif node:is(ast.typed.expr.RawRuntime) then
    return nil

  elseif node:is(ast.typed.expr.RawValue) then
    return nil

  elseif node:is(ast.typed.expr.Isnull) then
    return nil

  elseif node:is(ast.typed.expr.New) then
    return nil

  elseif node:is(ast.typed.expr.Null) then
    return nil

  elseif node:is(ast.typed.expr.DynamicCast) then
    return nil

  elseif node:is(ast.typed.expr.StaticCast) then
    return nil

  elseif node:is(ast.typed.expr.Ispace) then
    return nil

  elseif node:is(ast.typed.expr.Region) then
    return nil

  elseif node:is(ast.typed.expr.Partition) then
    return nil

  elseif node:is(ast.typed.expr.CrossProduct) then
    return nil

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return nil

  elseif node:is(ast.typed.expr.ListRange) then
    return nil

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return nil

  elseif node:is(ast.typed.expr.Advance) then
    return nil

  elseif node:is(ast.typed.expr.Copy) then
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

function analyze_var_flow.stat_index_launch(cx, node)
  local reduce_lhs = node.reduce_lhs and
    analyze_var_flow.expr(cx, node.reduce_lhs)

  if reduce_lhs then
    for v, _ in pairs(reduce_lhs) do
      local var_flow = cx:get_flow(v)
      var_flow[true] = true
    end
  end
end

function analyze_var_flow.stat_var(cx, node)
  local values = node.values:map(
    function(value) return analyze_var_flow.expr(cx, value) end)

  for i, symbol in ipairs(node.symbols) do
    local var_flow = cx:get_flow(symbol)
    local value = values[i]
    if value then
      for v, _ in pairs(value) do
        var_flow[v] = true
      end
    end
  end
end

function analyze_var_flow.stat_assignment(cx, node)
  local lhs = node.lhs:map(
    function(lh) return analyze_var_flow.expr(cx, lh) end)
  local rhs = node.rhs:map(
    function(rh) return analyze_var_flow.expr(cx, rh) end)

  for i, lh in ipairs(lhs) do
    local rh = rhs[i]
    if lh and rh then
      for lhv, _ in pairs(lh) do
        local lhv_flow = cx:get_flow(lhv)
        for rhv, _ in pairs(rh) do
          lhv_flow[rhv] = true
        end
      end
    end
  end
end

function analyze_var_flow.stat_reduce(cx, node)
  local lhs = node.lhs:map(
    function(lh) return analyze_var_flow.expr(cx, lh) end)
  local rhs = node.rhs:map(
    function(rh) return analyze_var_flow.expr(cx, rh) end)

  for i, lh in ipairs(lhs) do
    local rh = rhs[i]
    if lh and rh then
      for lhv, _ in pairs(lh) do
        local lhv_flow = cx:get_flow(lhv)
        for rhv, _ in pairs(rh) do
          lhv_flow[rhv] = true
        end
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

  elseif node:is(ast.typed.stat.IndexLaunch) then
    return analyze_var_flow.stat_index_launch(cx, node)

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

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function compute_var_futures(cx)
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

local optimize_futures = {}

function concretize(node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    return ast.typed.expr.FutureGetResult {
      value = node,
      expr_type = expr_type.result_type,
      options = node.options,
      span = node.span,
    }
  end
  return node
end

function promote(node)
  local expr_type = std.as_read(node.expr_type)
  if not std.is_future(expr_type) then
    return ast.typed.expr.Future {
      value = node,
      expr_type = std.future(expr_type),
      options = node.options,
      span = node.span,
    }
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
  local values = node.values:map(
    function(value)
      return concretize(optimize_futures.expr(cx, node.region))
    end)
  return node {
    values = values,
  }
end

function optimize_futures.expr_id(cx, node)
  if cx:is_future(node.value) then
    if std.is_rawref(node.expr_type) then
      return node {
        expr_type = std.rawref(&std.future(std.as_read(node.expr_type))),
      }
    else
      return node {
        expr_type = std.future(node.expr_type),
      }
    end
  else
    return node
  end
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

function optimize_futures.expr_new(cx, node)
  local region = concretize(optimize_futures.expr(cx, node.region))
  return node { region = region }
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

function optimize_futures.expr_cross_product(cx, node)
  local args = node.args:map(
    function(arg) return concretize(optimize_futures.expr(cx, arg)) end)
  return node {
    args = args,
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

function optimize_futures.expr_list_range(cx, node)
  local start = concretize(optimize_futures.expr(cx, node.start))
  local stop = concretize(optimize_futures.expr(cx, node.stop))
  return node {
    start = start,
    stop = stop,
  }
end

function optimize_futures.expr_phase_barrier(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    value = value,
  }
end

function optimize_futures.expr_advance(cx, node)
  local value = concretize(optimize_futures.expr(cx, node.value))
  return node {
    value = value,
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

  elseif node:is(ast.typed.expr.New) then
    return optimize_futures.expr_new(cx, node)

  elseif node:is(ast.typed.expr.Null) then
    return optimize_futures.expr_null(cx, node)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return optimize_futures.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.expr.StaticCast) then
    return optimize_futures.expr_static_cast(cx, node)

  elseif node:is(ast.typed.expr.Ispace) then
    return optimize_futures.expr_ispace(cx, node)

  elseif node:is(ast.typed.expr.Region) then
    return optimize_futures.expr_region(cx, node)

  elseif node:is(ast.typed.expr.Partition) then
    return optimize_futures.expr_partition(cx, node)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return optimize_futures.expr_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return optimize_futures.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.typed.expr.ListRange) then
    return optimize_futures.expr_list_range(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return optimize_futures.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return optimize_futures.expr_advance(cx, node)

  elseif node:is(ast.typed.expr.Copy) then
    return optimize_futures.expr_copy(cx, node)

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
    stats = node.stats:map(
      function(stat) return optimize_futures.stat(cx, stat) end),
  }
end

function optimize_futures.stat_if(cx, node)
  return node {
    cond = concretize(optimize_futures.expr(cx, node.cond)),
    then_block = optimize_futures.block(cx, node.then_block),
    elseif_blocks = node.elseif_blocks:map(
      function(block) return optimize_futures.stat_elseif(cx, block) end),
    else_block = optimize_futures.block(cx, node.else_block),
  }
end

function optimize_futures.stat_elseif(cx, node)
  return node {
    cond = concretize(optimize_futures.expr(cx, node.cond)),
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_while(cx, node)
  return node {
    cond = concretize(optimize_futures.expr(cx, node.cond)),
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_for_num(cx, node)
  return node {
    values = node.values:map(
      function(value) return concretize(optimize_futures.expr(cx, value)) end),
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_for_list(cx, node)
  return node {
    value = concretize(optimize_futures.expr(cx, node.value)),
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_repeat(cx, node)
  return node {
    block = optimize_futures.block(cx, node.block),
    until_cond = concretize(optimize_futures.expr(cx, node.until_cond)),
  }
end

function optimize_futures.stat_must_epoch(cx, node)
  return node {
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_block(cx, node)
  return node {
    block = optimize_futures.block(cx, node.block),
  }
end

function optimize_futures.stat_index_launch(cx, node)
  local domain = node.domain:map(
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

  return node {
    domain = domain,
    call = call,
    reduce_lhs = reduce_lhs,
  }
end

function optimize_futures.stat_var(cx, node)
  local types = terralib.newlist()
  local values = terralib.newlist()
  for i, symbol in ipairs(node.symbols) do
    local value_type = node.types[i]
    if cx:is_future(symbol) then
      types:insert(std.future(value_type))

      -- FIXME: Would be better to generate fresh symbols.
      symbol.type = std.future(value_type)
    else
      types:insert(value_type)
    end

    local value = node.values[i]
    if value then
      if cx:is_future(symbol) then
        values:insert(promote(optimize_futures.expr(cx, value)))
      else
        values:insert(concretize(optimize_futures.expr(cx, value)))
      end
    end
  end

  return node {
    types = types,
    values = values,
  }
end

function optimize_futures.stat_var_unpack(cx, node)
  return node {
    value = concretize(optimize_futures.expr(cx, node.value)),
  }
end

function optimize_futures.stat_return(cx, node)
  local value = node.value and concretize(optimize_futures.expr(cx, node.value))
  return node { value = value }
end

function optimize_futures.stat_break(cx, node)
  return node
end

function optimize_futures.stat_assignment(cx, node)
  local lhs = node.lhs:map(function(lh) return optimize_futures.expr(cx, lh) end)
  local rhs = node.rhs:map(function(rh) return optimize_futures.expr(cx, rh) end)

  local normalized_rhs = terralib.newlist()
  for i, lh in ipairs(lhs) do
    local rh = rhs[i]

    if std.is_future(std.as_read(lh.expr_type)) then
      normalized_rhs:insert(promote(rh))
    else
      normalized_rhs:insert(concretize(rh))
    end
  end

  return node {
    lhs = lhs,
    rhs = normalized_rhs,
  }
end

function optimize_futures.stat_reduce(cx, node)
  return node {
    lhs = node.lhs:map(function(lh) return optimize_futures.expr(cx, lh) end),
    rhs = node.rhs:map(function(rh) return optimize_futures.expr(cx, rh) end),
  }
end

function optimize_futures.stat_expr(cx, node)
  return node {
    expr = optimize_futures.expr(cx, node.expr),
  }
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

  elseif node:is(ast.typed.stat.IndexLaunch) then
    return optimize_futures.stat_index_launch(cx, node)

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

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function optimize_futures.stat_task(cx, node)
  local cx = cx:new_task_scope()
  analyze_var_flow.block(cx, node.body)
  compute_var_futures(cx)
  local body = optimize_futures.block(cx, node.body)

  return node { body = body }
end

function optimize_futures.stat_top(cx, node)
  if node:is(ast.typed.stat.Task) then
    return optimize_futures.stat_task(cx, node)

  else
    return node
  end
end

function optimize_futures.entry(node)
  local cx = context.new_global_scope()
  return optimize_futures.stat_top(cx, node)
end

return optimize_futures
