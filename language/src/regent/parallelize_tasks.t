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

-- Regent Auto-parallelizer

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local data = require("common/data")
local std = require("regent/std")
local report = require("common/report")
local symbol_table = require("regent/symbol_table")
local passes = require("regent/passes")
local pretty = require("regent/pretty")

local c = std.c

-- #####################################
-- ## Some context objects
-- #################

local parallel_task_context = {}

parallel_task_context.__index = parallel_task_context

local range_complex = std.newsymbol("__top__")

local function find_or_create(map, key, init)
  local init = init or data.newmap
  local value = map[key]
  if value == nil then
    value = init()
    map[key] = value
  end
  return value
end

local function unify(a, b, map)
  local seta = map[a]
  local setb = map[b]
  if seta == nil and setb == nil then
    local u = data.newmap()
    u[a] = true
    u[b] = true
    map[a] = u
    map[b] = u
  elseif seta == nil then
    local u = map[b]
    u[a] = true
    map[a] = u
  elseif setb == nil then
    local u = map[a]
    u[b] = true
    map[b] = u
  else
    for k, _ in setb:items() do
      seta[k] = true
      map[k] = seta
    end
  end
end

local function map_tostring(map)
  local str = nil
  for k, _ in map:items() do
    if str == nil then
      str = tostring(k)
    else
      str = str .. "," .. tostring(k)
    end
  end
  return str or ""
end

function parallel_task_context.new()
  local cx = {
    ranges                   = data.newmap(),
    all_ranges               = data.newmap(),

    field_accesses           = data.newmap(),
    field_accesses_summary   = data.newmap(),

    image_constraints        = data.newmap(),
    analytic_constraints     = data.newmap(),
    completeness_constraints = data.newmap(),
    disjointness_constraints = data.newmap(),
    equality_constraints     = data.newmap(),
    union_constraints        = data.newmap(),
    -- Complexity of the system of constraints is a 4-tuple of
    --  # of image/analytic constraints,
    --  # of disjointness/completeness constraints,
    --  # of equality constraints, and
    --  # of distinct field accesses
    complexity               = false,
  }

  return setmetatable(cx, parallel_task_context)
end

local function cmp_complexity(c1, c2)
  for idx = 1, 4 do
    if c1[idx] < c2[idx] then return true
    elseif c1[idx] > c2[idx] then return false end
  end
  return false
end

function parallel_task_context:print_all_constraints()
  print("================")
  print("* complexity:")
  print("    " .. self.complexity:mkstring("(", ", ", ")"))
  print("* all ranges:")
  for range, _ in self.all_ranges:items() do
    if range ~= range_complex then
      print("    " .. tostring(range))
    end
  end
  print("* image constraints:")
  for range, all_constraints in self.image_constraints:items() do
    for region, constraints in all_constraints:items() do
      for field_path, range_ in constraints:items() do
        print("    " .. tostring(region) .. "[" .. tostring(range) .. "]." ..
          field_path:mkstring("", ".", "") .. " <= " .. tostring(range_))
      end
    end
  end
  print("* analytic constraints:")
  for range, constraints in self.analytic_constraints:items() do
    for offset, range_ in constraints:items() do
      if std.is_symbol(offset[#offset]) then
        local symbol = offset[#offset]
        local o = offset:slice(1, #offset - 1)
        local divider =
          (std.is_region(symbol:gettype()) and tostring(symbol) .. ".ispace.bounds") or
          (std.is_ispace(symbol:gettype()) and tostring(symbol) .. ".bounds") or
          assert(false)
        print("    (" ..tostring(range) .. " + {" ..
          o:mkstring("",",","") .. "}) % " ..
          divider .. " <= " .. tostring(range_))
      else
        print("    " ..tostring(range) .. " + {" ..
          offset:mkstring("",",","") .. "} <= " .. tostring(range_))
      end
    end
  end
  print("* completeness constraints:")
  for range, _ in self.completeness_constraints:items() do
    print("    " .. tostring(range) .. " : complete")
  end
  print("* disjointness constraints:")
  for range, _ in self.disjointness_constraints:items() do
    print("    " .. tostring(range) .. " : disjoint")
  end
  print("* equality constraints:")
  do
    local duplicate = {}
    for range, equivalence_set in self.equality_constraints:items() do
      if duplicate[equivalence_set] == nil then
        print("    {" .. map_tostring(equivalence_set) .. "}")
        duplicate[equivalence_set] = true
      end
    end
  end
  print("* union constraints:")
  do
    local duplicate = {}
    for range, concurrent_set in self.union_constraints:items() do
      if duplicate[concurrent_set] == nil then
        print("    {" .. map_tostring(concurrent_set) .. "}")
        duplicate[concurrent_set] = true
      end
    end
  end
  print("* access summary:")
  for region, accesses_summary in self.field_accesses_summary:items() do
    for field_path, summary in accesses_summary:items() do
      local ranges, privilege = unpack(summary)
      local range = tostring(ranges[1])
      for idx = 2, #ranges do
        range = range .. "," .. tostring(ranges[idx])
      end
      print("    " .. tostring(region) .. "[{ " .. range .. " }]." ..
        field_path:mkstring("", ".", "") .. " @ " .. tostring(privilege))
    end
  end
  print("================")
end

function parallel_task_context:add_field_access(region_symbol, range, field_path, privilege)
  local all_field_accesses = find_or_create(self.field_accesses, region_symbol)
  local field_accesses = find_or_create(all_field_accesses, field_path)
  field_accesses[range] = std.meet_privilege(field_accesses[range], privilege)
end

function parallel_task_context:summarize_accesses()
  local complexity =  data.newtuple(0, 0, 0, 0)
  for region_symbol, all_field_accesses in self.field_accesses:items() do
    local accesses_summary = find_or_create(self.field_accesses_summary, region_symbol)
    for field_path, accesses in all_field_accesses:items() do
      local ranges = terralib.newlist()
      local joined_privilege = nil
      for range, privilege in accesses:items() do
        ranges:insert(range)
        joined_privilege = std.meet_privilege(joined_privilege, privilege)
      end
      accesses_summary[field_path] = { ranges, joined_privilege }

      complexity[4] = complexity[4] + 1

      if (joined_privilege == std.writes or joined_privilege == "reads_writes") then
        self:add_disjointness_constraint(ranges[1])
      end

      if #ranges > 1 then
        local constrain = nil
        if (joined_privilege == std.writes or joined_privilege == "reads_writes") then
          constrain = self.add_equality_constraint
        else
          constrain = self.add_union_constraint
        end
        for idx = 1, #ranges - 1 do
          constrain(self, ranges[idx], ranges[idx + 1])
        end
      end
    end
  end

  for range, all_constraints in self.image_constraints:items() do
    for region, constraints in all_constraints:items() do
      for field_path, range_ in constraints:items() do
        complexity[1] = complexity[1] + 1
      end
    end
  end
  for range, constraints in self.analytic_constraints:items() do
    for offset, range_ in constraints:items() do
      complexity[1] = complexity[1] + 1
    end
  end
  for range, _ in self.completeness_constraints:items() do
    complexity[2] = complexity[2] + 1
  end
  for range, _ in self.disjointness_constraints:items() do
    complexity[2] = complexity[2] + 1
  end
  local duplicate = {}
  for range, equivalence_set in self.equality_constraints:items() do
    if duplicate[equivalence_set] == nil then
      complexity[3] = complexity[3] + 1
      duplicate[equivalence_set] = true
    end
  end
  self.complexity = complexity
end

function parallel_task_context:find_or_create_image_constraint(src_range, region, field_path)
  local all_image_constraints = find_or_create(self.image_constraints, src_range)
  local image_constraints = find_or_create(all_image_constraints, region)
  return find_or_create(image_constraints, field_path, std.newsymbol)
end

function parallel_task_context:find_or_create_analytic_constraint(src_range, offset)
  local constraints = find_or_create(self.analytic_constraints, src_range)
  return find_or_create(constraints, offset, std.newsymbol)
end

function parallel_task_context:set_range(symbol, range)
  self.ranges[symbol] = range
  self.all_ranges[range] = true
end

function parallel_task_context:get_range(symbol)
  return self.ranges[symbol]
end

function parallel_task_context:add_completeness_constraint(range)
  self.completeness_constraints[range] = true
end

function parallel_task_context:add_disjointness_constraint(range)
  self.disjointness_constraints[range] = true
end

function parallel_task_context:add_equality_constraint(range1, range2)
  unify(range1, range2, self.equality_constraints)
end

function parallel_task_context:add_union_constraint(range1, range2)
  unify(range1, range2, self.union_constraints)
end

-- #####################################
-- ## Partitioning constraint inference
-- #################

-- The infer_constraints registers image partitioning constraints to the context
-- whenever the index is derived from a region access or an analyzable expression.

local infer_constraints = {}

local function unreachable(cx, node) node:printpretty(true) assert(false) end

function infer_constraints.expr_id(cx, expr, privilege, field_path)
  return cx:get_range(expr.value) or range_complex
end

function infer_constraints.expr_field_access(cx, expr, privilege, field_path)
  local field_path = field_path
  if std.is_ref(expr.expr_type) then
    field_path = field_path or expr.expr_type.field_path
  end
  return infer_constraints.expr(cx, expr.value, privilege, field_path)
end

function infer_constraints.expr_deref(cx, expr, privilege, field_path)
  assert(std.is_ref(expr.expr_type) and #expr.expr_type:bounds() == 1)
  local region_symbol = expr.expr_type.bounds_symbols[1]
  local region_type = expr.expr_type:bounds()[1]
  local expr = ast.typed.expr.IndexAccess {
    value = ast.typed.expr.ID {
      value = region_symbol,
      expr_type = std.rawref(&region_type),
      span = expr.span,
      annotations = expr.annotations,
    },
    index = expr.value,
    expr_type = expr.expr_type,
    span = expr.span,
    annotations = expr.annotations,
  }
  return infer_constraints.expr(cx, expr, privilege, field_path)
end

function infer_constraints.expr_index_access(cx, expr, privilege, field_path)
  local field_path = field_path or data.newtuple()
  if not std.is_ref(expr.expr_type) then return range_complex end
  assert(expr.value:is(ast.typed.expr.ID) and
         std.is_region(std.as_read(expr.value.expr_type)))
  local index_range = infer_constraints.expr(cx, expr.index)
  local region = expr.value.value
  local fspace = std.as_read(expr.expr_type)
  local field_type = std.get_field_path(fspace, field_path)
  local field_paths = std.flatten_struct_fields(field_type)
  field_paths:map(function(suffix)
    local field_path = field_path .. suffix
    cx:add_field_access(region, index_range, field_path, privilege)
  end)

  local field_type = std.get_field_path(std.as_read(expr.expr_type), field_path)
  if std.is_index_type(field_type) or std.is_bounded_type(field_type) then
    return cx:find_or_create_image_constraint(index_range, region, field_path)
  else
    return range_complex
  end
end

local function extract_offset(cx, expr)
  if expr:is(ast.typed.expr.Binary) then
    local lhs_range, lhs_offset = extract_offset(cx, expr.lhs)
    local rhs_range, rhs_offset = extract_offset(cx, expr.rhs)
    if (expr.op == "%" or expr.op == "+") and rhs_range == nil then
      return lhs_range, lhs_offset == nil and rhs_offset or lhs_offset .. rhs_offset
    end
  elseif expr:is(ast.typed.expr.FieldAccess) then
    if std.is_rect_type(expr.expr_type) and expr.field_name == "bounds" and
       std.is_ispace(std.as_read(expr.value.expr_type))
    then
      local base = expr.value
      if base:is(ast.typed.expr.ID) then
        return nil, data.newtuple(base.value)
      elseif base:is(ast.typed.expr.FieldAccess) and
             base.field_name == "ispace" and
             base.value:is(ast.typed.expr.ID)
      then
        assert(std.is_region(std.as_read(base.value.expr_type)))
        return nil, data.newtuple(base.value.value)
      end
    end
  elseif expr:is(ast.typed.expr.ID) then
    return cx:get_range(expr.value), nil
  elseif expr:is(ast.typed.expr.Constant) then
    return nil, data.newtuple(expr.value)
  elseif expr:is(ast.typed.expr.Ctor) then
    local all_constant = true
    local values = expr.fields:map(function(field)
      if field.value:is(ast.typed.expr.Constant) then
        return field.value.value
      elseif field.value:is(ast.typed.expr.Unary) and
             field.value.op == "-" and
             field.value.rhs:is(ast.typed.expr.Constant)
      then
        assert(field.value.expr_type:isintegral())
        return -field.value.rhs.value
      else
        all_constant = false
        return nil
      end
    end)
    if all_constant then return nil, data.newtuple(unpack(values)) end
  end
  return range_complex, nil
end

function infer_constraints.expr_binary(cx, expr, privilege, field_path)
  if not std.is_index_type(expr.expr_type) then return range_complex end
  local src_range, offset = extract_offset(cx, expr)
  if offset ~= nil then
    return cx:find_or_create_analytic_constraint(src_range, offset)
  else
    return range_complex
  end
end

function infer_constraints.expr_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.arg, privilege) or range_complex
end

function infer_constraints.expr_regent_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.value, privilege) or range_complex
end

function infer_constraints.expr_complex(cx, expr, privilege, field_path)
  return range_complex
end

local infer_constraints_expr_table = {
  [ast.typed.expr.ID]           = infer_constraints.expr_id,
  [ast.typed.expr.FieldAccess]  = infer_constraints.expr_field_access,
  [ast.typed.expr.Deref]        = infer_constraints.expr_deref,
  [ast.typed.expr.IndexAccess]  = infer_constraints.expr_index_access,
  [ast.typed.expr.Binary]       = infer_constraints.expr_binary,
  [ast.typed.expr.Cast]         = infer_constraints.expr_cast,
  [ast.typed.expr.DynamicCast]  = infer_constraints.expr_regent_cast,
  [ast.typed.expr.StaticCast]   = infer_constraints.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]   = infer_constraints.expr_regent_cast,

  [ast.typed.expr.Constant]     = infer_constraints.expr_complex,
  [ast.typed.expr.Call]         = infer_constraints.expr_complex,
  [ast.typed.expr.Null]         = infer_constraints.expr_complex,
  [ast.typed.expr.Isnull]       = infer_constraints.expr_complex,
  [ast.typed.expr.Ctor]         = infer_constraints.expr_complex,
  [ast.typed.expr.Unary]        = infer_constraints.expr_complex,
  [ast.typed.expr]              = unreachable,
}

local infer_constraints_expr = ast.make_single_dispatch(
  infer_constraints_expr_table,
  {ast.typed.expr})

function infer_constraints.expr(cx, node, privilege, field_path)
  return infer_constraints_expr(cx)(node, privilege, field_path)
end

function infer_constraints.stat_for_list(cx, stat)
  if stat.metadata and stat.metadata.parallelizable then
    cx:set_range(stat.symbol, stat.value.value)
    cx:add_completeness_constraint(stat.value.value)
    if stat.metadata.reductions and #stat.metadata.reductions > 0 then
      cx:add_disjointness_constraint(stat.value.value)
    end
  end
  infer_constraints.block(cx, stat.block)
end

function infer_constraints.stat_block(cx, stat)
  infer_constraints.block(cx, stat.block)
end

function infer_constraints.stat_if(cx, stat)
  infer_constraints.block(cx, stat.then_block)
  infer_constraints.block(cx, stat.else_block)
end

local function infer_constraints_stat_var(cx, stat)
  if not stat.value then return end

  local range = infer_constraints.expr(cx, stat.value, std.reads)
  cx:set_range(stat.symbol, range)
end

function infer_constraints_stat_assignment_or_reduce(cx, stat)
  assert(not std.is_ref(stat.rhs.expr_type))

  if std.is_ref(stat.lhs.expr_type) then
    local privilege = (stat:is(ast.typed.stat.Assignment) and std.writes) or
                      std.reduces(stat.op)
    infer_constraints.expr(cx, stat.lhs, privilege)
  end
end

function infer_constraints.pass_through_stat(cx, stat) end

local infer_constraints_stat_table = {
  [ast.typed.stat.ForList]    = infer_constraints.stat_for_list,

  [ast.typed.stat.ForNum]     = infer_constraints.stat_block,
  [ast.typed.stat.While]      = infer_constraints.stat_block,
  [ast.typed.stat.Repeat]     = infer_constraints.stat_block,
  [ast.typed.stat.Block]      = infer_constraints.stat_block,

  [ast.typed.stat.If]         = infer_constraints.stat_if,

  [ast.typed.stat.Var]        = infer_constraints_stat_var,
  [ast.typed.stat.Assignment] = infer_constraints_stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = infer_constraints_stat_assignment_or_reduce,

  [ast.typed.stat.Return]     = infer_constraints.pass_through_stat,
  [ast.typed.stat.Expr]       = infer_constraints.pass_through_stat,

  [ast.typed.stat]            = unreachable,
}

local infer_constraints_stat = ast.make_single_dispatch(
  infer_constraints_stat_table,
  {ast.typed.stat})

function infer_constraints.stat(cx, node)
  infer_constraints_stat(cx)(node)
end

function infer_constraints.block(cx, body)
  body.stats:map(function(stat) infer_constraints.stat(cx, stat) end)
end

function infer_constraints.top_task(node)
  local proto = node.prototype
  assert(proto)

  -- Analyze loops in the task
  local cx = parallel_task_context.new()
  infer_constraints.block(cx, node.body)
  cx:summarize_accesses()
  proto:set_partitioning_constraints(cx)
  print(node.name)
  cx:print_all_constraints()

  --local task_name = node.name .. data.newtuple("parallelized")
  --local task = std.new_task(task_name)
  --local variant = task:make_variant("primary")

  --local params = terralib.newlist()
  --local privileges = terralib.newlist()
  --local region_universe = node.prototype:get_region_universe():copy()
  --local coherence_modes = node.prototype:get_coherence_modes():copy()

  --params:insertall(node.params)
  --params:map(function(param)
  --  if std.is_region(param.param_type) then
  --    local accesses = cx.field_accesses[param.symbol]
  --    if accesses then
  --      local primary_privileges = terralib.newlist()
  --      for field, privilege in accesses:items() do
  --        if privilege == "reads_writes" then
  --          primary_privileges:insert(
  --            std.privilege(std.reads, param.symbol, field))
  --          primary_privileges:insert(
  --            std.privilege(std.writes, param.symbol, field))
  --        else
  --          primary_privileges:insert(
  --            std.privilege(privilege, param.symbol, field))
  --        end
  --      end
  --      privileges:insert(primary_privileges)
  --    end
  --  end
  --end)

  --cx.image_regions:map(function(region_symbol)
  --  local accesses = cx.field_accesses[region_symbol]
  --  if accesses then
  --    params:insert(ast_util.mk_task_param(region_symbol))
  --    local image_privileges = terralib.newlist()
  --    for field, privilege in accesses:items() do
  --      image_privileges:insert(
  --        std.privilege(privilege, region_symbol, field))
  --    end
  --    privileges:insert(image_privileges)
  --    region_universe[region_symbol:gettype()] = true
  --  end
  --end)

  --task:set_type(terralib.types.functype(
  --    params:map(function(param) return param.param_type end),
  --    node.return_type,
  --    false))
  --task:set_param_symbols(params:map(function(param) return param.symbol end))

  --task:set_primary_variant(variant)
  --task:set_privileges(privileges)
  --task:set_coherence_modes(coherence_modes)
  --task:set_flags(node.flags)
  --task:set_conditions(node.conditions)
  --task:set_param_constraints(node.prototype:get_param_constraints())
  --task:set_constraints(node.prototype:get_constraints())
  --task:set_region_universe(region_universe)

  --local task_ast = ast.typed.top.Task {
  --  name = task_name,
  --  params = params,
  --  return_type = node.return_type,
  --  privileges = privileges,
  --  coherence_modes = coherence_modes,
  --  flags = node.flags,
  --  conditions = node.conditions,
  --  constraints = node.constraints,
  --  body = body,
  --  config_options = node.config_options,
  --  region_divergence = false,
  --  metadata = false,
  --  prototype = task,
  --  annotations = node.annotations {
  --    parallel = ast.annotation.Forbid { value = false },
  --  },
  --  span = node.span,
  --}

  --local task_ast_optimized = passes.optimize(task_ast)
  --task = passes.codegen(task_ast_optimized, true)

  --return task, cx
end

local parallelize_tasks = {}

function parallelize_tasks.entry(node)
  if node:is(ast.typed.top.Task) then
    if node.annotations.parallel:is(ast.annotation.Demand) then
      assert(node.config_options.leaf)
      assert(node.metadata)
      infer_constraints.top_task(node)
      return node
    --elseif not node.config_options.leaf then
    --  return parallelize_task_calls.top_task(global_context, node)
    else
      return node
    end
  else
    return node
  end
end

parallelize_tasks.pass_name = "parallelize_tasks"

return parallelize_tasks
