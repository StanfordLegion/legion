-- Copyright 2019 Stanford University
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

-- Regent Parallelizability Checker

--
-- A for loop (whether it is a for-list loop or a for-num loop) is parallelizable when
-- its body has no loop carried dependences and no inadmissible statements.
--
-- A loop body is free of loop carried dependences if every access in each iteration only reads from
-- or reduces to a location, or is made on a location private to the iteration, including variables
-- new to the scope or region elements indexed directly or indirectly by the loop variable. When the
-- loop is nested, each access in the loop is analyzed with respect to that loop and all the outer
-- loops as well. For example, the inner loop in the following program is parallelizable as the
-- the region element r[e2] is private to each iteration, whereas the outer loop is not
-- parallelizable because r[e2] can be an arbitrary element with respect to this loop:
--
--   for e1 in is1 do
--     for e2 in is2 do
--       r[e2] = 1
--     end
--   end
--
-- The following is the list of inadmissible statements:
--   > parallel prefix
--   > return
--   > break
--   > raw delete
--   > any top-level expression as a statement, including function calls
-- If a loop nests another loop with an inadmissible statement, that loop also becomes
-- non-parallelizable.
--
-- Vectorization and OpenMP code generation require the for loop to be parallelizable.
-- CUDA and partition driven auto-parallelizer, which are applied to the whole task, have
-- the following additional requirements:
--
-- * CUDA
--   - No parallelizable for-list loop can have other parallelizable for-list loops.
--     In case that happens, the outer loop is tagged as non-parallelizable.
--   - No region access can appear outside parallelizable for loops.
--
-- * Partition Driven Auto-Parallelizer
--   - Inadmissible statements are not allowed in anywhere in the task, except the return
--     statement that returns a scalar reduction variable is allowed. This return statement
--     must appear at the end of the task.
--   - No scalar variables or arrays can be created outside parallelizable for loops and
--     accessed inside, except up to one scalar variable for reductions is allowed.
--

local ast = require("regent/ast")
local data = require("common/data")
local normalize_access = require("regent/normalize_access")
local pretty = require("regent/pretty")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

-- Utilities

local prefixes = {
  ["vectorize"] = "vectorization",
  ["openmp"]    = "OpenMP code generation",
  ["cuda"]      = "CUDA code generation",
  ["parallel"]  = "partition driven auto-parallelization",
}

local function unreachable(cx, node) assert(false) end
local function pass_through(cx, node) return node end

local function make_singleton(name)
  local tbl = {}
  tbl.__index = tbl
  function tbl.__tostring() return name end
  return setmetatable(tbl, tbl)
end

local PRIVATE = make_singleton("private")
local SHARED_REDUCE = make_singleton("shared_reduce")
local SHARED_READ = make_singleton("shared_read")
local MIXED = make_singleton("mixed")

local function join(v1, v2)
  if v1 == nil then return v2 end
  if v2 == nil then return v1 end
  if v1 ~= v2 or v1 == MIXED or v2 == MIXED then
    return MIXED
  else
    return v1
  end
end

-- Loop context object that tracks parallelizability of a single loop

local loop_context = {}

function loop_context:__index (field)
  local value = loop_context [field]
  if value ~= nil then
    return value
  end
  error ("loop_context has no field '" .. field .. "' (in lookup)", 2)
end

function loop_context:__newindex (field, value)
  error ("loop_context has no field '" .. field .. "' (in assignment)", 2)
end

function loop_context.new_scope(loop, loop_var)
  local cx = {
    loop = loop,
    loop_var = loop_var,
    needs_iterator = loop_var and loop:is(ast.typed.stat.ForList),
    -- Tracks variables that have the same value as the loop variable
    centers = data.newmap(),
    -- Tracks variables that are private to each iteration
    -- * private[v] == PRIVATE ==> variable v is defined in the loop
    -- * private[v] == SHARED_REDUCE ==> variable v is defined outside the loop and
    --                                   used for scalar reduction
    -- * private[v] == SHARED_READ ==> variable v is defined outside the loop and
    --                                 read in the loop
    -- * private[v] == MIXED ==> variable v is defined outside the loop and
    --                           both read and used for scalar reduction
    private = data.newmap(),
    -- Remebers scalar variables used for reductions
    -- * reductions[v] == true ==> private[v] == nil
    reductions = data.newmap(),
    -- * privileges[region][field] = { privilege, all private }
    privileges = data.newmap(),
    admissible = { true, nil },
    parallel = { true, nil },
  }
  if loop_var then
    cx.centers[loop_var] = true
    cx.private[loop_var] = PRIVATE
  end
  return setmetatable(cx, loop_context)
end

function loop_context:is_parallel()
  return self.parallel[1]
end

function loop_context:is_admissible()
  return self.admissible[1]
end

function loop_context:is_parallelizable()
  return self:is_parallel() and self:is_admissible()
end

function loop_context:mark_serial(evidence)
  if self:is_parallel() then
    self.parallel = { false, evidence }
  end
end

function loop_context:mark_inadmissible(evidence)
  if self:is_admissible() then
    self.admissible = { false, evidence }
  end
end

function loop_context:get_serial_node()
  assert(not self:is_parallel())
  return self.parallel[2]
end

function loop_context:get_inadmissible_node()
  assert(not self:is_admissible())
  return self.admissible[2]
end

function loop_context:get_metadata()
  assert(self.loop_var)
  local parallelizable = self:is_parallelizable()
  local reductions = parallelizable and terralib.newlist()
  if parallelizable then
    for k, _ in self.reductions:items() do
      reductions:insert(k)
    end
  end
  return ast.metadata.Loop {
    parallelizable = self:is_parallelizable(),
    reductions = reductions,
  }
end

function loop_context:mark_reduction_variable(variable)
  self.reductions[variable] = true
end

function loop_context:unmark_reduction_variable(variable)
  self.reductions[variable] = false
end

function loop_context:is_reduction_variable(variable)
  return self.reductions[variable]
end

function loop_context:is_private(symbol)
  return self.private[symbol]
end

function loop_context:is_center(symbol)
  return self.centers[symbol] or false
end

function loop_context:update_private(symbol, value)
  self.private[symbol] = value
end

function loop_context:update_center(symbol, value)
  self.centers[symbol] = value
end

function loop_context:update_privilege(node, region_type, field_path, new_priv, private)
  local privileges = self.privileges[region_type]
  if not privileges then
    privileges = data.newmap()
    self.privileges[region_type] = privileges
  end
  local privilege, all_private, any_private =
    unpack(privileges[field_path] or { nil, true, false })
  privilege = std.meet_privilege(privilege, new_priv)
  all_private = all_private and private
  any_private = any_private or private
  privileges[field_path] = { privilege, all_private, any_private }
  if (privilege == "reads_writes" or privilege == std.writes) and
     not all_private
  then
    self:mark_serial(node)
  end
end

function loop_context:update_privileges(node, ref_type, new_priv, private)
  local bounds = ref_type:bounds()
  local field_path = ref_type.field_path
  bounds:map(function(bound)
    self:update_privilege(node, bound, field_path, new_priv, private)
  end)
end

-- Context that maintains parallelizability of the whole task

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

function context.new_task_scope(task)
  local cx = {
    task = task,
    demand_cuda = std.config["cuda"] and
                  task.annotations.cuda:is(ast.annotation.Demand),
    contexts = terralib.newlist({loop_context.new_scope(task, false)}),
  }
  return setmetatable(cx, context)
end

function context:push_loop_context(loop, loop_symbol)
  self.contexts:insert(loop_context.new_scope(loop, loop_symbol))
end

function context:pop_loop_context(node)
  local cx = self:current_context()
  self.contexts:remove()

  -- Propagate all region accesses in the scope to the outer ones
  for region, privileges in cx.privileges:items() do
    for field, pair in privileges:items() do
      local privilege, all_private, any_private = unpack(pair)
      if any_private then
        self:forall_context(function(cx)
          cx:update_privilege(node, region, field, privilege, false)
          return false
        end)
      end
    end
  end

  -- When the task demands CUDA code generation, we only keep the innermost
  -- parallelizable for list loop.
  if self.demand_cuda and cx:is_parallelizable() and cx.needs_iterator then
    self:forall_context(function(cx)
      cx:mark_serial(node)
      return false
    end)
  end
end

function context:current_context()
  assert(self.contexts[#self.contexts] ~= nil)
  return self.contexts[#self.contexts]
end

function context:mark_inadmissible(node)
  self:current_context():mark_inadmissible(node)
end

function context:mark_serial(node)
  self:current_context():mark_serial(node)
end

function context:get_metadata()
  return self:current_context():get_metadata()
end

function context:forall_context(f)
  for idx = #self.contexts, 1, -1 do
    local stop = f(self.contexts[idx])
    if stop then return end
  end
end

-- #####################################
-- # Parallelizability checker
-- #################

local analyze_access = {}

function analyze_access.expr_id(cx, node, privilege)
  local symbol = node.value
  local private = cx:is_private(symbol)
  local center = cx:is_center(symbol)

  -- If the variable is local to the loop, any access is safe
  if private == PRIVATE then
    if privilege ~= nil and std.is_reduce(privilege) then
      cx:mark_reduction_variable(symbol)
    end
    return true, center
  else
    if privilege == nil then
      return false, center
    end

    -- Any write access to a variable defined outside is not parallelizable
    if privilege == std.writes then
      private = join(private, MIXED)
    elseif privilege == std.reads then
      private = join(private, SHARED_READ)
    elseif std.is_reduce(privilege) then
      private = join(private, SHARED_REDUCE)
    end

    cx:update_private(symbol, private)
    if private == MIXED then
      cx:mark_serial(node)
      cx:unmark_reduction_variable(symbol)
    elseif private == SHARED_REDUCE then
      cx:mark_reduction_variable(symbol)
    end

    return false, center
  end
end

function analyze_access.expr_field_access(cx, node, privilege)
  local expr_type = node.expr_type
  if std.is_ref(expr_type) then
    local deref = node
    while deref:is(ast.typed.expr.FieldAccess) do
      deref = deref.value
    end
    local private, center = analyze_access.expr(cx, deref, nil)
    cx:update_privileges(node, expr_type, privilege, private)
    return private, center
  else
    return analyze_access.expr(cx, node.value, privilege)
  end
end

function analyze_access.expr_deref(cx, node, privilege)
  local expr_type = node.expr_type
  local private, center = analyze_access.expr(cx, node.value)
  if std.is_ref(expr_type) then
    cx:update_privileges(node, expr_type, privilege, center)
  end
  return center, false
end

function analyze_access.expr_index_access(cx, node, privilege)
  local expr_type = node.expr_type
  if std.is_ref(expr_type) then
    analyze_access.expr(cx, node.value, nil)
    local private, center =
      analyze_access.expr(cx, node.index, std.reads)
    cx:update_privileges(node, expr_type, privilege, center)
    return center, false
  else
    -- We don't do accurate analysis for array accesses,
    -- because the primary parallelization target is a loop
    -- iterating over region elements.
    local value_private, value_center =
      analyze_access.expr(cx, node.value, privilege)
    local index_private, index_center =
      analyze_access.expr(cx, node.index, std.reads)
    return value_private or index_center, false
  end
end

function analyze_access.expr_constant(cx, node, privilege)
  return true, false
end

function analyze_access.expr_binary(cx, node, privilege)
  analyze_access.expr(cx, node.lhs, privilege)
  analyze_access.expr(cx, node.rhs, privilege)
  return false, false
end

function analyze_access.expr_unary(cx, node, privilege)
  analyze_access.expr(cx, node.rhs, privilege)
  return false, false
end

function analyze_access.expr_cast(cx, node, privilege)
  return analyze_access.expr(cx, node.arg, privilege)
end

function analyze_access.expr_regent_cast(cx, node, privilege)
  return analyze_access.expr(cx, node.value, privilege)
end

local whitelist = {
  [array]                                   = true,
  [std.assert]                              = true,
  [std.c.legion_get_current_time_in_micros] = true,
  [std.c.legion_get_current_time_in_nanos]  = true,
  [std.c.srand48_r]                         = true,
  [std.c.drand48_r]                         = true,
}

local function is_admissible_function(fn)
  return std.is_math_fn(fn) or whitelist[fn]
end

function analyze_access.expr_call(cx, node, privilege)
  if not is_admissible_function(node.fn.value) then
    cx:mark_inadmissible(node)
  end
  node.args:map(function(arg)
    analyze_access.expr(cx, arg, std.reads)
  end)
  return false, false
end

function analyze_access.expr_isnull(cx, node, privilege)
  analyze_access.expr(cx, node.pointer, std.reads)
  return false, false
end

function analyze_access.expr_ctor(cx, node, privilege)
  node.fields:map(function(field)
    analyze_access.expr(cx, field.value, std.reads)
  end)
  return false, false
end

function analyze_access.expr_not_analyzable(cx, node, privilege)
  cx:mark_inadmissible(node)
  return false, false
end

local analyze_access_expr_table = {
  [ast.typed.expr.ID]           = analyze_access.expr_id,
  [ast.typed.expr.FieldAccess]  = analyze_access.expr_field_access,
  [ast.typed.expr.Deref]        = analyze_access.expr_deref,
  [ast.typed.expr.IndexAccess]  = analyze_access.expr_index_access,
  [ast.typed.expr.Constant]     = analyze_access.expr_constant,
  [ast.typed.expr.Binary]       = analyze_access.expr_binary,
  [ast.typed.expr.Unary]        = analyze_access.expr_unary,
  [ast.typed.expr.Cast]         = analyze_access.expr_cast,
  [ast.typed.expr.DynamicCast]  = analyze_access.expr_regent_cast,
  [ast.typed.expr.StaticCast]   = analyze_access.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]   = analyze_access.expr_regent_cast,
  [ast.typed.expr.Call]         = analyze_access.expr_call,
  [ast.typed.expr.Null]         = analyze_access.expr_constant,
  [ast.typed.expr.Isnull]       = analyze_access.expr_isnull,
  [ast.typed.expr.Ctor]         = analyze_access.expr_ctor,
  [ast.typed.expr]              = analyze_access.expr_not_analyzable,
}

local analyze_access_expr = ast.make_single_dispatch(
  analyze_access_expr_table,
  {ast.typed.expr})

function analyze_access.expr(cx, node, privilege)
  return analyze_access_expr(cx)(node, privilege)
end

local function check_demand(cx, node, annotations, type)
  if std.config[type] and
     node.annotations[type]:is(ast.annotation.Demand) and
     not (node.metadata and node.metadata.parallelizable)
  then
    if not cx:is_admissible() then
      report.error(cx:get_inadmissible_node(), prefixes[type] ..
          " failed: found an inadmissible statement")
    else
      report.error(cx:get_serial_node(), prefixes[type] ..
          " failed: found a loop-carried dependence")
    end
  end
end

local function check_demands(cx, node)
  check_demand(cx, node, node.annotations, "vectorize")
  check_demand(cx, node, node.annotations, "openmp")
end

function analyze_access.stat_for_list(cx, node)
  local symbol_type = node.symbol:gettype()
  local candidate = std.is_index_type(symbol_type) or
                    std.is_bounded_type(symbol_type)
  if not candidate then cx:mark_inadmissible(node) end

  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.value, std.reads)
  end)

  cx:push_loop_context(node, candidate and node.symbol)
  local block = analyze_access.block(cx, node.block)
  local node = node {
    block = block,
    metadata = candidate and cx:get_metadata(),
  }
  check_demands(cx:current_context(), node)
  cx:pop_loop_context(node)
  return node
end

function analyze_access.stat_for_num(cx, node)
  local has_stride = #node.values > 2

  cx:forall_context(function(cx)
    return data.all(node.values:map(function(value)
      return analyze_access.expr(cx, value, std.reads)
    end))
  end)

  cx:push_loop_context(node, not has_stride and node.symbol)
  local block = analyze_access.block(cx, node.block)
  local node = node {
    block = block,
    metadata = not has_stride and cx:get_metadata(),
  }
  check_demands(cx:current_context(), node)
  cx:pop_loop_context(node)
  return node
end

function analyze_access.stat_if(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.cond, std.reads)
  end)
  local then_block = analyze_access.block(cx, node.then_block)
  local else_block = analyze_access.block(cx, node.else_block)
  return node {
    then_block = then_block,
    else_block = else_block,
  }
end

function analyze_access.stat_while(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.cond, std.reads)
  end)
  return node { block = analyze_access.block(cx, node.block) }
end

function analyze_access.stat_repeat(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.until_cond, std.reads)
  end)
  return node { block = analyze_access.block(cx, node.block) }
end

function analyze_access.stat_block(cx, node)
  return node { block = analyze_access.block(cx, node.block) }
end

function analyze_access.stat_var(cx, node)
  local center = false
  if node.value then
    local first = true
    cx:forall_context(function(cx)
      local value_private, value_center =
        analyze_access.expr(cx, node.value, std.reads)
      if first then
        center = value_center
        first = false
      end
      return value_private
    end)
  end
  local cx = cx:current_context()
  local symbol = node.symbol
  cx:update_center(symbol, center)
  cx:update_private(symbol, PRIVATE)
  return node
end

function analyze_access.stat_var_unpack(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.value, std.reads)
  end)

  local cx = cx:current_context()
  node.symbols:map(function(symbol)
    cx:update_center(symbol, false)
    cx:update_private(symbol, PRIVATE)
  end)

  return node
end

function analyze_access.stat_assignment(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.rhs, std.reads)
  end)

  local first = true
  local atomic = nil
  cx:forall_context(function(cx)
    local private, center = analyze_access.expr(cx, node.lhs, std.writes)
    if first then
      atomic = not private
      first = false
    end
    return private
  end)

  assert(atomic ~= nil)
  return node {
    metadata = ast.metadata.Stat {
      atomic = atomic,
      scalar = false,
    }
  }
end

function analyze_access.stat_reduce(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.rhs, std.reads)
  end)

  local first = true
  local atomic = nil
  local scalar = false
  cx:forall_context(function(cx)
    local private, center =
      analyze_access.expr(cx, node.lhs, std.reduces(node.op))
    if first then
      if node.lhs:is(ast.typed.expr.ID) then
        scalar = cx:is_reduction_variable(node.lhs.value) or false
      end
      atomic = not private
      first = false
    end
    return private
  end)

  assert(atomic ~= nil)
  return node {
    metadata = ast.metadata.Stat {
      atomic = atomic and not scalar,
      scalar = scalar,
    }
  }
end

function analyze_access.stat_expr(cx, node)
  if node.expr:is(ast.typed.expr.Call) then
    cx:forall_context(function(cx)
      return analyze_access.expr_call(cx, node.expr, nil)
    end)
  end
  return node
end

function analyze_access.stat_inadmissible(cx, node)
  cx:mark_inadmissible(node)
  return node
end

local analyze_access_stat_table = {
  [ast.typed.stat.ForList]         = analyze_access.stat_for_list,
  [ast.typed.stat.ForNum]          = analyze_access.stat_for_num,

  [ast.typed.stat.If]              = analyze_access.stat_if,
  [ast.typed.stat.While]           = analyze_access.stat_while,
  [ast.typed.stat.Repeat]          = analyze_access.stat_repeat,
  [ast.typed.stat.Block]           = analyze_access.stat_block,

  [ast.typed.stat.Var]             = analyze_access.stat_var,
  [ast.typed.stat.VarUnpack]       = analyze_access.stat_var_unpack,
  [ast.typed.stat.Assignment]      = analyze_access.stat_assignment,
  [ast.typed.stat.Reduce]          = analyze_access.stat_reduce,

  [ast.typed.stat.Expr]            = analyze_access.stat_expr,

  [ast.typed.stat.Return]          = analyze_access.stat_inadmissible,

  [ast.typed.stat.ParallelPrefix]  = analyze_access.stat_inadmissible,
  [ast.typed.stat.Break]           = analyze_access.stat_inadmissible,
  [ast.typed.stat.RawDelete]       = analyze_access.stat_inadmissible,

  [ast.typed.stat.Elseif]            = unreachable,
  [ast.typed.stat.Internal]          = unreachable,
  [ast.typed.stat.MustEpoch]         = unreachable,
  [ast.typed.stat.ParallelizeWith]   = unreachable,
  [ast.typed.stat.ForNumVectorized]  = unreachable,
  [ast.typed.stat.ForListVectorized] = unreachable,
  [ast.typed.stat.IndexLaunchNum]    = unreachable,
  [ast.typed.stat.IndexLaunchList]   = unreachable,
  [ast.typed.stat.BeginTrace]        = unreachable,
  [ast.typed.stat.EndTrace]          = unreachable,
  [ast.typed.stat.MapRegions]        = unreachable,
  [ast.typed.stat.UnmapRegions]      = unreachable,
  [ast.typed.stat.Fence]             = unreachable,
}

local analyze_access_stat = ast.make_single_dispatch(
  analyze_access_stat_table,
  {ast.typed.stat})

function analyze_access.stat(cx, node)
  return analyze_access_stat(cx)(node)
end

function analyze_access.block(cx, node)
  local stats = node.stats:map(function(stat)
    return analyze_access.stat(cx, stat)
  end)
  return node { stats = stats }
end

local check_context = {}

function check_context:__index (field)
  local value = check_context [field]
  if value ~= nil then
    return value
  end
  error ("check_context has no field '" .. field .. "' (in lookup)", 2)
end

function check_context:__newindex (field, value)
  error ("check_context has no field '" .. field .. "' (in assignment)", 2)
end

function check_context.new_task_scope(prefix)
  local cx = {
    prefix = prefix,
    contained = false,
  }
  return setmetatable(cx, check_context)
end

function check_context:new_local_scope(loop)
  local cx = {
    prefix = self.prefix,
    contained =
      (loop.metadata and loop.metadata.parallelizable) or
      self.contained,
  }
  return setmetatable(cx, check_context)
end

function check_context:is_contained()
  return self.contained
end

function check_context:report(node)
  if not self:is_contained() then
    report.error(node, self.prefix ..
      " failed: found a region access outside parallelizable loops")
  end
end

local check_region_access_contained = {}

function check_region_access_contained.pass_through(cx, node)
end

function check_region_access_contained.expr_field_access(cx, node)
  if std.is_ref(node.expr_type) then
    cx:report(node)
  else
    check_region_access_contained.expr(cx, node.value)
  end
end

function check_region_access_contained.expr_deref(cx, node)
  if std.is_ref(node.expr_type) then cx:report(node) end
end

function check_region_access_contained.expr_index_access(cx, node)
  if std.is_ref(node.expr_type) then
    cx:report(node)
  else
    check_region_access_contained.expr(cx, node.value)
  end
end

local check_region_access_contained_expr_table = {
  [ast.typed.expr.FieldAccess]  = check_region_access_contained.expr_field_access,
  [ast.typed.expr.Deref]        = check_region_access_contained.expr_deref,
  [ast.typed.expr.IndexAccess]  = check_region_access_contained.expr_index_access,
  [ast.typed.expr]              = check_region_access_contained.pass_through,
}

local check_region_access_contained_expr = ast.make_single_dispatch(
  check_region_access_contained_expr_table,
  {ast.typed.expr})

function check_region_access_contained.expr(cx, node)
  check_region_access_contained_expr(cx)(node)
end

function check_region_access_contained.stat_for_loop(cx, node)
  local cx = cx:new_local_scope(node)
  check_region_access_contained.block(cx, node.block)
end

function check_region_access_contained.stat_if(cx, node)
  check_region_access_contained.block(cx, node.then_block)
  check_region_access_contained.block(cx, node.else_block)
end

function check_region_access_contained.stat_block(cx, node)
  check_region_access_contained.block(cx, node.block)
end

function check_region_access_contained.stat_var(cx, node)
  if node.value then
    check_region_access_contained.expr(cx, node.value)
  end
end

function check_region_access_contained.stat_assignment_or_reduce(cx, node)
  check_region_access_contained.expr(cx, node.lhs)
  check_region_access_contained.expr(cx, node.rhs)
end

local check_region_access_contained_stat_table = {
  [ast.typed.stat.ForList]    = check_region_access_contained.stat_for_loop,
  [ast.typed.stat.ForNum]     = check_region_access_contained.stat_for_loop,

  [ast.typed.stat.If]         = check_region_access_contained.stat_if,
  [ast.typed.stat.While]      = check_region_access_contained.stat_block,
  [ast.typed.stat.Repeat]     = check_region_access_contained.stat_block,
  [ast.typed.stat.Block]      = check_region_access_contained.stat_block,

  [ast.typed.stat.Var]        = check_region_access_contained.stat_var,
  [ast.typed.stat.Assignment] = check_region_access_contained.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = check_region_access_contained.stat_assignment_or_reduce,

  [ast.typed.stat]            = check_region_access_contained.pass_through,
}

local check_region_access_contained_stat = ast.make_single_dispatch(
  check_region_access_contained_stat_table,
  {ast.typed.stat})

function check_region_access_contained.stat(cx, node)
  check_region_access_contained_stat(cx)(node)
end

function check_region_access_contained.block(cx, node)
  node.stats:map(function(stat)
    check_region_access_contained.stat(cx, stat)
  end)
end

function check_region_access_contained.top_task(node)
  local demand_cuda = std.config["cuda"] and
    node.annotations.cuda:is(ast.annotation.Demand)
  local demand_parallel = std.config["parallelize"] and
    node.annotations.parallel:is(ast.annotation.Demand)
  if not node.body or not (demand_cuda or demand_parallel) then
    return
  end
  local prefix = (demand_cuda and prefixes["cuda"]) or
                 (demand_parallel and prefixes["parallel"]) or
                 assert(false)
  local cx = check_context.new_task_scope(prefix)
  check_region_access_contained.block(cx, node.body)
end

local function check_demand_parallel(cx, node)
  if std.config["parallelize"] and
     node.annotations.parallel:is(ast.annotation.Demand) and
     not cx:is_admissible()
  then
    local inadmissible_node = cx:get_inadmissible_node()
    if not inadmissible_node:is(ast.typed.stat.Return) then
      report.error(inadmissible_node, prefixes["parallel"] ..
          " failed: found an inadmissible statement")
    end
    local stats = node.body.stats
    if inadmissible_node ~= stats[#stats] or
       not inadmissible_node.value or
       not cx:is_reduction_variable(inadmissible_node.value.value)
    then
      report.error(inadmissible_node, prefixes["parallel"] ..
          " failed: found an inadmissible statement")
    end
  end
end

local check_parallelizable = {}

function check_parallelizable.top_task(node)
  local body = node.body
  if body then
    local cx = context.new_task_scope(node)
    body = normalize_access.block(node.body)
    body = analyze_access.block(cx, body)
    node = node { body = body }

    check_region_access_contained.top_task(node)
    check_demand_parallel(cx:current_context(), node)
  end

  return node
end

function check_parallelizable.top(node)
  if node:is(ast.typed.top.Task) and node.config_options.leaf then
    return check_parallelizable.top_task(node)

  else
    return node
  end
end

function check_parallelizable.entry(node)
  return check_parallelizable.top(node)
end

check_parallelizable.pass_name = "check_parallelizable"

return check_parallelizable
