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
-- * Constraint-Based Auto-Parallelizer
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

function loop_context.new_scope(task, loop, loop_var)
  local needs_iterator = loop_var and loop:is(ast.typed.stat.ForList) and
                         not std.is_rect_type(std.as_read(loop.value.expr_type))
  local cx = {
    loop = loop,
    loop_var = loop_var,
    demand_vectorize = loop.annotations.vectorize:is(ast.annotation.Demand),
    demand_openmp = loop.annotations.openmp:is(ast.annotation.Demand),
    demand_cuda = task.annotations.cuda:is(ast.annotation.Demand),
    demand_parallel = task.annotations.parallel:is(ast.annotation.Demand),
    needs_iterator = needs_iterator,
    -- Tracks variables that have the same value as the loop variable
    centers = data.newmap(),
    -- Tracks variables that are private to each iteration
    -- * scalars[symbol] = { privilege, private }
    scalars = data.newmap(),
    -- * privileges[region][field] = { privilege, all private, any private }
    privileges = data.newmap(),
    -- * array_accesses[array] = { privilege, all private }
    array_accesses = data.newmap(),
    -- Remebers variables used for scalar reductions
    reductions = data.newmap(),
    admissible = { true, nil },
    parallel = { true, nil },
    innermost = true,
    outermost = false,
  }
  cx = setmetatable(cx, loop_context)
  if loop_var then
    cx:update_center(loop_var, true)
    cx:update_scalar(loop_var, std.reads, true)
  end
  return cx
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

function loop_context:cache_reduction_variables()
  -- Cache all scalar reduction variables
  for symbol, pair in self.scalars:items() do
    local privilege, private = unpack(pair)
    if not private and std.is_reduce(privilege) then
      self.reductions[symbol] = true
    end
  end
end

function loop_context:filter_shared_variables(reductions)
  local result = data.newmap()
  for symbol, _ in reductions:items() do
    if not self:is_local_variable(symbol) then
      result[symbol] = true
    end
  end
  return result
end

function loop_context:union_reduction_variables(reductions)
  for symbol, _ in reductions:items() do
    self.reductions[symbol] = true
  end
end

function loop_context:is_reduction_variable(symbol)
  return self.reductions[symbol] or false
end

function loop_context:get_metadata()
  assert(self.loop_var)
  local parallelizable = self:is_parallelizable()
  if std.config["override-demand-cuda"] and self.demand_cuda and
     not parallelizable and self:is_admissible() and self.innermost
  then
    report.warn(self:get_serial_node(),
        "WARNING: ignoring a potential loop-carried dependence in a task " ..
        "that demands CUDA. Please check that this loop is indeed parallelizable.")
    parallelizable = true
  end
  local reductions = parallelizable and terralib.newlist()
  if parallelizable then
    for symbol, _ in self.reductions:items() do
      reductions:insert(symbol)
    end
  end
  return ast.metadata.Loop {
    parallelizable = parallelizable,
    reductions = reductions,
  }
end

function loop_context:is_center(symbol)
  return self.centers[symbol] or false
end

function loop_context:get_scalar(symbol)
  local pair = self.scalars[symbol]
  if pair == nil then
    pair = { nil, false }
    self.scalars[symbol] = pair
  end
  return pair
end

function loop_context:update_scalar(symbol, privilege, private)
  self.scalars[symbol] = { privilege, private }
end

function loop_context:is_local_variable(symbol)
  return self:get_scalar(symbol)[2]
end

function loop_context:add_local_variable(symbol)
  self.scalars[symbol] = { nil, true }
end

function loop_context:update_center(symbol, value)
  self.centers[symbol] = value
end

function loop_context:check_privilege(node, privilege, private)
  if (privilege == "reads_writes" or privilege == std.writes) and not private then
    self:mark_serial(node)
  end
end

function loop_context:update_privilege(node, region_type, field_path, new_priv, private)
  local field_path = field_path or data.newtuple()
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
  self:check_privilege(node, privilege, all_private)
end

function loop_context:update_privileges(node, regions, field_path, new_priv, private)
  regions:map(function(region_type)
    self:update_privilege(node, region_type, field_path, new_priv, private)
  end)
end

function loop_context:update_array_access(node, symbol, new_priv, private)
  local privilege, all_private =
    unpack(self.array_accesses[symbol] or { nil, true })
  privilege = std.meet_privilege(privilege, new_priv)
  all_private = all_private and private
  self.array_accesses[symbol] = { privilege, all_private }
  self:check_privilege(node, privilege, all_private)
end

function loop_context:set_innermost(value)
  self.innermost = value
end

function loop_context:set_outermost(value)
  self.outermost = value
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
    demand_cuda = task.annotations.cuda:is(ast.annotation.Demand),
    demand_parallel = task.annotations.parallel:is(ast.annotation.Demand),
    contexts = terralib.newlist({loop_context.new_scope(task, task, false)}),
  }
  cx.contexts[1]:set_outermost(true)
  return setmetatable(cx, context)
end

function context:push_loop_context(loop, loop_symbol)
  self.contexts:insert(loop_context.new_scope(self.task, loop, loop_symbol))
end

function context:pop_loop_context(node)
  local cx = self:current_context()
  self.contexts:remove()
  cx:cache_reduction_variables()

  -- Propagate all region accesses in the scope to the outer ones
  for region, privileges in cx.privileges:items() do
    for field, tuple in privileges:items() do
      local privilege, all_private, any_private = unpack(tuple)
      if any_private then
        self:forall_context(function(cx)
          cx:update_privilege(node, region, field, privilege, false)
          return false
        end)
      end
    end
  end

  -- Propagate all array accesses in the scope to the outer ones
  for symbol, pair in cx.array_accesses:items() do
    local privilege, all_private = unpack(pair)
    if not cx:is_local_variable(symbol) then
      self:forall_context(function(cx)
        local private = cx:is_local_variable(symbol)
        cx:update_array_access(node, symbol, privilege, private)
        return private
      end)
    end
  end

  -- Propagate all reduction variables accesses in the scope to the outer ones
  local to_propagate = cx:filter_shared_variables(cx.reductions)
  self:forall_context(function(cx_outer)
    if not cx_outer.outermost then
      to_propagate = cx_outer:filter_shared_variables(to_propagate)
    end
    cx_outer:union_reduction_variables(to_propagate)
    return to_propagate:is_empty()
  end)

  if cx.demand_cuda and cx.needs_iterator then
    -- When the task demands CUDA code generation, we only keep the innermost
    -- parallelizable for list loop.
    self:forall_context(function(cx_outer)
      cx_outer:set_innermost(false)
      if cx_outer.loop_var then cx_outer:mark_inadmissible(node) end
      return false
    end)
  end

  if (cx.demand_cuda or cx.demand_openmp) and cx:is_parallelizable() then
    -- When the loop that demands OpenMP or is in a CUDA task updates
    -- arrays defined outside, we mark it inadmissible.
    for symbol, pair in cx.array_accesses:items() do
      if not cx:is_local_variable(symbol) then
        local privilege, private = unpack(pair)
        if privilege and privilege ~= std.reads then
          cx:mark_inadmissible(node)
          break
        end
      end
    end
  end

  return cx
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

function analyze_access.expr_id(cx, node, new_privilege, field_path)
  local symbol = node.value
  local privilege, private = unpack(cx:get_scalar(symbol))
  local center = cx:is_center(symbol)
  privilege = std.meet_privilege(privilege, new_privilege)
  cx:update_scalar(symbol, privilege, private)
  cx:check_privilege(node, privilege, private)
  return private, center, false
end

function analyze_access.expr_field_access(cx, node, privilege, field_path)
  if std.is_ref(node.expr_type) then
    local field_path = field_path or node.expr_type.field_path
    return analyze_access.expr(cx, node.value, privilege, field_path)
  else
    local private, center =
      analyze_access.expr(cx, node.value, privilege, field_path)
    return private, false
  end
end

function analyze_access.expr_deref(cx, node, privilege, field_path)
  local expr_type = node.expr_type
  local private, center = analyze_access.expr(cx, node.value, std.reads)
  local region_access = false
  if std.is_ref(expr_type) then
    -- We disallow any multi-region pointer in a parallelizable loop
    -- when the task demands partition driven auto-parallelization
    if cx.demand_parallel and #expr_type:bounds() > 1 then
      cx:mark_inadmissible(node)
    else
      cx:update_privileges(node, expr_type:bounds(), field_path, privilege, center)
    end
    region_access = true
  elseif std.as_read(node.value.expr_type):ispointer() then
    -- We disallow any raw pointer dereferences in a parallelizable loop
    -- as we do not know about their aliasing.
    cx:mark_inadmissible(node)
  end
  return center, false, region_access
end

function analyze_access.expr_index_access(cx, node, privilege, field_path)
  local expr_type = node.expr_type
  if std.is_ref(expr_type) and #expr_type.field_path == 0 then
    analyze_access.expr(cx, node.value, nil)
    local private, center = analyze_access.expr(cx, node.index, std.reads)
    cx:update_privileges(node, expr_type:bounds(), field_path, privilege, center)
    return center, false, true
  else
    local value_type = std.as_read(node.value.expr_type)
    local value_symbol =
      node.value:is(ast.typed.expr.ID) and node.value.value or false
    if value_type:isarray() and value_symbol then
      local value_private, value_center, region_access =
        analyze_access.expr(cx, node.value)
      local index_private, index_center =
        analyze_access.expr(cx, node.index, std.reads)
      local private = value_private or index_center
      cx:update_array_access(node, value_symbol, privilege, private)
      return private, false, region_access
    elseif value_type:ispointer() then
      -- We disallow any raw pointer dereferences in a parallelizable loop
      -- as we do not know about their aliasing.
      cx:mark_inadmissible(node)
      return false, false, false
    else
      analyze_access.expr(cx, node.index, std.reads)
      return analyze_access.expr(cx, node.value, privilege)
    end
    assert(false)
  end
end

function analyze_access.expr_constant(cx, node, privilege, field_path)
  return true, false, false
end

function analyze_access.expr_binary(cx, node, privilege, field_path)
  analyze_access.expr(cx, node.lhs, privilege)
  analyze_access.expr(cx, node.rhs, privilege)
  return false, false, false
end

function analyze_access.expr_unary(cx, node, privilege, field_path)
  analyze_access.expr(cx, node.rhs, privilege)
  return false, false, false
end

function analyze_access.expr_cast(cx, node, privilege, field_path)
  return analyze_access.expr(cx, node.arg, privilege)
end

function analyze_access.expr_regent_cast(cx, node, privilege, field_path)
  return analyze_access.expr(cx, node.value, privilege)
end

local whitelist = {
  [array]                                   = true,
  [arrayof]                                 = true,
  [vector]                                  = true,
  [vectorof]                                = true,
  [std.assert]                              = true,
  [std.assert_error]                        = true,
  [std.c.printf]                            = true,
}

local function is_admissible_function(cx, fn)
  if cx.demand_cuda and cx.loop_var then
    return std.is_math_fn(fn) or fn == array or fn == arrayof
  else
    return std.is_math_fn(fn) or whitelist[fn]
  end
end

function analyze_access.expr_call(cx, node, privilege, field_path)
  if not is_admissible_function(cx, node.fn.value) then
    cx:mark_inadmissible(node)
  end
  node.args:map(function(arg)
    analyze_access.expr(cx, arg, std.reads)
  end)
  return false, false, false
end

function analyze_access.expr_isnull(cx, node, privilege, field_path)
  analyze_access.expr(cx, node.pointer, std.reads)
  return false, false, false
end

function analyze_access.expr_ctor(cx, node, privilege, field_path)
  node.fields:map(function(field)
    analyze_access.expr(cx, field.value, std.reads)
  end)
  return false, false, false
end

function analyze_access.expr_global(cx, node, privilege, field_path)
  return false, false, false
end

function analyze_access.expr_not_analyzable(cx, node, privilege, field_path)
  cx:mark_inadmissible(node)
  return false, false, false
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
  [ast.typed.expr.Global]       = analyze_access.expr_global,
  [ast.typed.expr]              = analyze_access.expr_not_analyzable,
}

local analyze_access_expr = ast.make_single_dispatch(
  analyze_access_expr_table,
  {ast.typed.expr})

function analyze_access.expr(cx, node, privilege, field_path)
  return analyze_access_expr(cx)(node, privilege, field_path)
end

local function check_demands(cx, node)
  if (cx.demand_vectorize or cx.demand_openmp) and
     not (node.metadata and node.metadata.parallelizable)
  then
    local prefix = (cx.demand_vectorize and prefixes["vectorize"]) or
                   (cx.demand_openmp and prefixes["openmp"])
    if not cx:is_admissible() then
      report.error(cx:get_inadmissible_node(), prefix ..
          " failed: found an inadmissible statement")
    else
      if cx.demand_vectorize or not std.config["override-demand-openmp"] then
        report.error(cx:get_serial_node(), prefix ..
            " failed: found a loop-carried dependence")
      else
        report.warn(cx:get_serial_node(),
            "WARNING: ignoring a potential loop-carried dependence in a loop " ..
            "that demands OpenMP. Please check that this loop is indeed parallelizable.")
      end
    end
  end
end

function analyze_access.stat_for_list(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.value, std.reads)
  end)

  local symbol_type = node.symbol:gettype()
  local candidate = std.is_index_type(symbol_type) or
                    std.is_bounded_type(symbol_type)
  cx:push_loop_context(node, candidate and node.symbol)
  if not candidate then cx:mark_inadmissible(node) end
  local block = analyze_access.block(cx, node.block)
  local cx = cx:pop_loop_context(node)
  local node = node {
    block = block,
    metadata = candidate and cx:get_metadata(),
  }
  check_demands(cx, node)
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
  local cx = cx:pop_loop_context(node)
  local node = node {
    block = block,
    metadata = not has_stride and cx:get_metadata(),
  }
  check_demands(cx, node)
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
  local symbol = node.symbol
  if node.value then
    cx:forall_context(function(cx)
      local value_private, value_center =
        analyze_access.expr(cx, node.value, std.reads)
      cx:update_center(symbol, value_center)
      return false
    end)
  end
  local cx = cx:current_context()
  cx:add_local_variable(symbol)
  return node
end

function analyze_access.stat_var_unpack(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.value, std.reads)
  end)

  local cx = cx:current_context()
  node.symbols:map(function(symbol)
    cx:update_center(symbol, false)
    cx:add_local_variable(symbol)
  end)

  return node
end

function analyze_access.stat_assignment(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.rhs, std.reads)
  end)

  cx:forall_context(function(cx)
    local private, center = analyze_access.expr(cx, node.lhs, std.writes)
    return private
  end)

  return node {
    metadata = ast.metadata.Stat {
      centers = false,
      scalar = false,
    }
  }
end

-- TODO: This function should change once we add support for custom reduction operators
local function reducible_type(type)
  return type:isprimitive() or (type:isarray() and type.type:isprimitive())
end

function analyze_access.stat_reduce(cx, node)
  cx:forall_context(function(cx)
    return analyze_access.expr(cx, node.rhs, std.reads)
  end)

  local lhs_type = std.as_read(node.lhs.expr_type)
  local lhs_scalar = node.lhs:is(ast.typed.expr.ID)
  local rhs_type = std.as_read(node.rhs.expr_type)
  local first = true
  local scalar = false
  local privilege =
    (reducible_type(lhs_type) and reducible_type(rhs_type) and std.reduces(node.op)) or
    "reads_writes"
  local centers = (std.is_reduce(privilege) and data.newmap()) or false
  cx:forall_context(function(cx)
    local private, center, region_access =
      analyze_access.expr(cx, node.lhs, privilege)
    if (cx.demand_openmp or cx.demand_cuda) and
       not (private or lhs_scalar or region_access)
    then
      cx:mark_inadmissible(node)
    end
    if first then
      if lhs_scalar then
        scalar = not cx:is_local_variable(node.lhs.value)
      end
      first = false
    end
    if private and region_access and std.is_reduce(privilege) then
      centers[cx.loop_var] = true
    end
    return private
  end)

  return node {
    metadata = ast.metadata.Stat {
      centers = centers,
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
      (loop:is(ast.typed.stat.ForList) and
       loop.metadata and loop.metadata.parallelizable) or
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
  local demand_cuda = node.annotations.cuda:is(ast.annotation.Demand)
  local demand_parallel = node.annotations.parallel:is(ast.annotation.Demand)
  if not node.body or not (demand_cuda or demand_parallel) then
    return
  end
  local prefix = (demand_cuda and prefixes["cuda"]) or
                 (demand_parallel and prefixes["parallel"]) or
                 assert(false)
  local cx = check_context.new_task_scope(prefix)
  check_region_access_contained.block(cx, node.body)
end

-- Here we check additional constraints when the task demands
-- partition driven auto-parallelization.
local function check_demand_parallel(cx, node)
  if std.config["parallelize"] and
     node.annotations.parallel:is(ast.annotation.Demand)
  then
    if not cx:is_admissible() then
      local inadmissible_node = cx:get_inadmissible_node()
      if not inadmissible_node:is(ast.typed.stat.Return) then
        report.error(inadmissible_node, prefixes["parallel"] ..
            " failed: found an inadmissible statement")
      end
      local stats = node.body.stats
      assert(not inadmissible_node.value or
             inadmissible_node.value:is(ast.typed.expr.ID))
      local return_variable =
        (inadmissible_node.value and inadmissible_node.value.value) or
        false
      if inadmissible_node ~= stats[#stats] or not return_variable or
         not cx:is_reduction_variable(return_variable)
      then
        report.error(inadmissible_node, prefixes["parallel"] ..
            " failed: found an inadmissible statement")
      end
      for symbol, _ in cx.reductions:items() do
        if symbol ~= return_variable then
          report.error(cx.loop, prefixes["parallel"] ..
              " failed: task can have only up to one scalar reduction variable")
        end
      end
    else
      if not cx.reductions:is_empty() then
        report.error(cx.loop, prefixes["parallel"] ..
            " failed: task must return the scalar reduction result")
      end
    end
    local reduction = false
    local op = false
    for symbol, _ in cx.reductions:items() do
      local privilege, _ = unpack(cx:get_scalar(symbol))
      if not std.is_reduce(privilege) then
        report.error(cx.loop, prefixes["parallel"] ..
            " failed: task must not read reduction variable")
      end
      reduction = symbol
      op = privilege.op
    end
    node = node {
      metadata = ast.metadata.Task {
        reduction = reduction,
        op = op,
      }
    }
  end
  return node
end

local check_parallelizable = {}

function check_parallelizable.top_task(node)
  local body = node.body
  if body then
    local cx = context.new_task_scope(node)
    body = normalize_access.block(node.body)
    body = analyze_access.block(cx, body)
    node = node { body = body }

    node = check_demand_parallel(cx:current_context(), node)
    check_region_access_contained.top_task(node)
  end

  return node
end

function check_parallelizable.top(node)
  if node:is(ast.typed.top.Task) then
    if not node.config_options.leaf then
      if node.annotations.cuda:is(ast.annotation.Demand) then
        report.error(node,
          "option __demand(__cuda) is not permitted for non-leaf task")
      elseif node.annotations.parallel:is(ast.annotation.Demand) then
        report.error(node,
          "option __demand(__parallel) is not permitted for non-leaf task")
      end
      return node
    else
      return check_parallelizable.top_task(node)
    end

  else
    return node
  end
end

function check_parallelizable.entry(node)
  return check_parallelizable.top(node)
end

check_parallelizable.pass_name = "check_parallelizable"

return check_parallelizable
