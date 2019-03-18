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

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local hash_set                 = require("regent/parallelizer/hash_set")
local ranges                   = require("regent/parallelizer/ranges")
local partition_info           = require("regent/parallelizer/partition_info")
local partitioning_constraints = require("regent/parallelizer/partitioning_constraints")
local task_generator           = require("regent/parallelizer/task_generator")

local function find_or_create(map, key, init)
  local init = init or data.newmap
  local value = map[key]
  if value == nil then
    value = init()
    map[key] = value
  end
  return value
end

local parallel_task_context = {}

parallel_task_context.__index = parallel_task_context

function parallel_task_context.new(task)
  local cx = {
    task                     = task,
    env                      = data.newmap(),
    sources                  = data.newmap(),
    sources_by_regions       = data.newmap(),

    field_accesses           = data.newmap(),
    field_accesses_summary   = data.newmap(),

    ranges_to_indices        = data.newmap(),

    constraints              = partitioning_constraints.new(),

    loop_ranges              = data.newmap(),

    params                   = hash_set.from_list(task.prototype:get_param_symbols()),

    -- Complexity of the system of constraints is a pair of
    --  # of subset/image/analytic constraints,
    --  # of distinct field accesses
    complexity               = false,
  }

  return setmetatable(cx, parallel_task_context)
end

function parallel_task_context:print_all_constraints()
  print("================")
  print("* complexity:")
  print("    " .. self.complexity:mkstring("(", ", ", ")"))
  print("* sources:")
  for range, _ in self.sources:items() do
    print("    " .. tostring(range))
  end
  print("* sources by regions:")
  for region, source in self.sources_by_regions:items() do
    print("    " .. tostring(region) .. " => " .. tostring(source))
  end
  self.constraints:print_constraints()
  print("* accesses:")
  for region_symbol, accesses_summary in self.field_accesses_summary:items() do
    for field_path, summary in accesses_summary:items() do
      local ranges_set, privilege = unpack(summary)
      print("    " .. tostring(region_symbol) .. "[" .. tostring(ranges_set) .. "]." ..
        field_path:mkstring("", ".", "") .. " @ " .. tostring(privilege))
    end
  end
  print("================")
end

local require_disjoint_partition = {
  [std.writes]     = true,
  ["reads_writes"] = true,
}

function parallel_task_context:add_field_access(region_symbol, range, field_path, privilege)
  local partition = self.constraints:get_partition(range)
  if not (partition == nil or partition.region == region_symbol) then
    local new_range =
      self.constraints:find_or_create_subset_constraint(range, region_symbol)
    range = new_range
    partition = self.constraints:get_partition(range)
  end

  local all_field_accesses = find_or_create(self.field_accesses, region_symbol)
  local field_accesses = find_or_create(all_field_accesses, field_path)
  local join = std.meet_privilege(field_accesses[range], privilege)
  field_accesses[range] = join
  if partition == nil then
    local disjoint = require_disjoint_partition[join]
    self:set_partition(range, partition_info.new(region_symbol, disjoint, false))
  else
    partition:meet_disjointness(require_disjoint_partition[join])
  end
end

function parallel_task_context:summarize_accesses()
  local range_mapping = data.newmap()
  local needs_unification = false
  local num_accesses = 0
  for region_symbol, all_field_accesses in self.field_accesses:items() do
    for field_path, accesses in all_field_accesses:items() do
      local ranges = terralib.newlist()
      local join = nil
      for range, privilege in accesses:items() do
        ranges:insert(range)
        join = std.meet_privilege(join, privilege)
      end
      local disjoint = require_disjoint_partition[join]
      if disjoint and #ranges > 1 then
        local equivalence_class = ranges[1]
        needs_unification = true
        ranges:map(function(range) range_mapping[range] = equivalence_class end)
      end
      num_accesses = num_accesses + 1
    end
  end

  if needs_unification then
    self.constraints = self.constraints:clone(range_mapping)
    self.loop_ranges = self.loop_ranges:map(function(_, range)
      return range_mapping[range] or range
    end)
  end
  for region_symbol, all_field_accesses in self.field_accesses:items() do
    local accesses_summary = find_or_create(self.field_accesses_summary, region_symbol)
    for field_path, accesses in all_field_accesses:items() do
      local ranges_set = hash_set.new()
      local join = nil
      for range, privilege in accesses:items() do
        local new_range = range_mapping[range] or range
        ranges_set:insert(new_range)
        join = std.meet_privilege(join, privilege)
      end
      -- Force canonicalization
      ranges_set:hash()
      accesses_summary[field_path] = { ranges_set, join }
    end
  end
  self.constraints:remove_unnecessary_constraints()

  local satisfiable, error_message = self.constraints:propagate_disjointness()
  if not satisfiable then
    report.error(self.task,
      prefix .. " failed: found an unsatisfiable constraint during unification: " ..
      error_message)
  end

  self.complexity = data.newtuple(self.constraints:get_complexity(), num_accesses)
end

function parallel_task_context:find_or_create_image_constraint(src_range, region_symbol, field_path)
  return self.constraints:find_or_create_image_constraint(src_range, region_symbol, field_path)
end

function parallel_task_context:find_or_create_analytic_constraint(src_range, offset)
  return self.constraints:find_or_create_analytic_constraint(src_range, offset)
end

function parallel_task_context:set_range(symbol, range)
  self.env[symbol] = range
end

function parallel_task_context:get_range(symbol)
  return self.env[symbol]
end

function parallel_task_context:set_partition(range, partition)
  if range == ranges.range_complex then return end
  self.constraints:set_partition(range, partition)
end

function parallel_task_context:update_partition(range, partition)
  if range == ranges.range_complex then return end
  local my_partition = self.constraints:get_partition(range)
  if my_partition == nil then
    self.constraints:set_partition(range, partition)
  else
    my_partition:meet_disjointness(partition.disjoint)
    my_partition:meet_completeness(partition.complete)
  end
end

function parallel_task_context:find_or_create_source_range(region)
  local range = find_or_create(self.sources_by_regions, region, ranges.new)
  self.sources[range] = true
  return range
end

function parallel_task_context:add_loop_range(loop_var, range)
  assert(self.loop_ranges[loop_var] == nil)
  self.loop_ranges[loop_var] = range
end

function parallel_task_context:is_param(symbol)
  return self.params:has(symbol)
end

function parallel_task_context:record_index_of_range(range, index)
  if self.ranges_to_indices[range] == nil then
    self.ranges_to_indices[range] = terralib.newlist()
  end
  self.ranges_to_indices[range]:insert(index)
end

function parallel_task_context:find_indices_of_range(range)
  return self.ranges_to_indices[range]
end

local function unreachable(cx, node) assert(false) end

local infer_constraints = {}

function infer_constraints.expr_id(cx, expr, privilege, field_path)
  return cx:get_range(expr.value) or ranges.range_complex
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
  if not std.is_ref(expr.expr_type) then return ranges.range_complex end
  assert(expr.value:is(ast.typed.expr.ID) and
         std.is_region(std.as_read(expr.value.expr_type)))
  local index_range = infer_constraints.expr(cx, expr.index)
  local region_symbol = expr.value.value
  local fspace = std.as_read(expr.expr_type)
  local field_type = std.get_field_path(fspace, field_path)
  local field_paths = std.flatten_struct_fields(field_type)
  field_paths:map(function(suffix)
    local field_path = field_path .. suffix
    cx:add_field_access(region_symbol, index_range, field_path, privilege)
  end)
  cx:record_index_of_range(index_range, expr.index)

  local field_type = std.get_field_path(std.as_read(expr.expr_type), field_path)
  if std.is_index_type(field_type) or std.is_bounded_type(field_type) then
    return cx:find_or_create_image_constraint(index_range, region_symbol, field_path)
  else
    return ranges.range_complex
  end
end

local function extract_offset(cx, expr, positive)
  if expr:is(ast.typed.expr.Binary) then
    if expr.op == "%" then
      local lhs_range, lhs_offset = extract_offset(cx, expr.lhs, positive)
      if expr.rhs:is(ast.typed.expr.ID) and cx:is_param(expr.rhs.value) and
         std.is_rect_type(std.as_read(expr.rhs.expr_type))
      then
        return lhs_range, lhs_offset .. data.newtuple(expr.rhs.value)
      end
    elseif expr.op == "+" or expr.op == "-" then
      local lhs_range, lhs_offset = extract_offset(cx, expr.lhs, positive)
      if expr.op == "-" then positive = not positive end
      local rhs_range, rhs_offset = extract_offset(cx, expr.rhs, positive)
      if lhs_range ~= ranges.range_complex and rhs_offset ~= nil then
        return lhs_range, rhs_offset
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
        if positive then
          return field.value.value
        else
          return -field.value.value
        end
      elseif field.value:is(ast.typed.expr.Unary) and
             field.value.op == "-" and
             field.value.rhs:is(ast.typed.expr.Constant)
      then
        assert(field.value.expr_type:isintegral())
        if positive then
          return -field.value.rhs.value
        else
          return field.value.rhs.value
        end
      else
        all_constant = false
        return nil
      end
    end)
    if all_constant then return nil, data.newtuple(unpack(values)) end
  end
  return ranges.range_complex, nil
end

function infer_constraints.expr_binary(cx, expr, privilege, field_path)
  if not std.is_index_type(expr.expr_type) then
    return ranges.range_complex
  end
  local src_range, offset = extract_offset(cx, expr, true)
  if src_range ~= ranges.range_complex and offset ~= nil then
    return cx:find_or_create_analytic_constraint(src_range, offset)
  else
    return ranges.range_complex
  end
end

function infer_constraints.expr_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.arg, privilege) or ranges.range_complex
end

function infer_constraints.expr_regent_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.value, privilege) or ranges.range_complex
end

function infer_constraints.expr_complex(cx, expr, privilege, field_path)
  return ranges.range_complex
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
    local region_symbol = stat.value.value
    local range = cx:find_or_create_source_range(region_symbol)
    cx:set_range(stat.symbol, range)
    local disjoint = stat.metadata.reductions and #stat.metadata.reductions > 0
    cx:update_partition(range, partition_info.new(region_symbol, disjoint, true))
    cx:add_loop_range(stat.symbol, range)
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

function infer_constraints.stat_var(cx, stat)
  if not stat.value then return end

  local range = infer_constraints.expr(cx, stat.value, std.reads)
  cx:set_range(stat.symbol, range)
end

function infer_constraints.stat_assignment_or_reduce(cx, stat)
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

  [ast.typed.stat.Var]        = infer_constraints.stat_var,
  [ast.typed.stat.Assignment] = infer_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = infer_constraints.stat_assignment_or_reduce,

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

function infer_constraints.block(cx, block)
  block.stats:map(function(stat) infer_constraints.stat(cx, stat) end)
end

function infer_constraints.top_task(node)
  local proto = node.prototype
  assert(proto)

  -- Analyze loops in the task
  local cx = parallel_task_context.new(node)
  infer_constraints.block(cx, node.body)
  cx:summarize_accesses()
  proto:set_partitioning_constraints(cx)
  local generator = task_generator.new(node)
  proto:set_parallel_task_generator(generator)
end

return infer_constraints
