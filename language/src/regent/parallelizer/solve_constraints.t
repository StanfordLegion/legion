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
local data = require("common/data")
local std = require("regent/std")

local hash_set                 = require("regent/parallelizer/hash_set")
local partitioning_constraints = require("regent/parallelizer/partitioning_constraints")

local c = std.c

local function find_or_create(map, key, init)
  local init = init or data.newmap
  local value = map[key]
  if value == nil then
    value = init()
    map[key] = value
  end
  return value
end

local function cmp_complexity(c1, c2)
  if c1[1] > c2[1] then return true
  elseif c1[1] < c2[1] then return false
  else
    if c1[2] > c2[2] then return true
    elseif c1[2] < c2[2] then return false end
  end
  return false
end

local function cmp_tasks(t1, t2)
  return cmp_complexity(
    t1:get_partitioning_constraints().complexity,
    t2:get_partitioning_constraints().complexity)
end

-- #####################################
-- ## Partitioning constraint solver
-- #################

-- The collect_constraints pass collects all partitioning constraints

local collector_context = {}

collector_context.__index = collector_context

function collector_context.new()
  local cx = {
    all_tasks = terralib.newlist(),
    task_mappings = {},
  }

  return setmetatable(cx, collector_context)
end

function collector_context:add_task(task, mapping)
  self.all_tasks:insert(task)
  self.task_mappings[task] = mapping
end

function collector_context:get_mapping(task)
  local mapping = self.task_mappings[task]
  assert(mapping ~= nil)
  return mapping
end

local collect_constraints = {}

function collect_constraints.stat_parallelize_with(cx, stat)
  report.error(stat, prefix .. " failed: parallelize_with blocks cannot be nested")
end

function collect_constraints.stat_block(cx, stat)
  collect_constraints.block(cx, stat.block)
end

function collect_constraints.stat_if(cx, stat)
  collect_constraints.block(cx, stat.then_block)
  collect_constraints.block(cx, stat.else_block)
end

local function add_task(cx, call)
  assert(call:is(ast.typed.expr.Call))
  local task = call.fn.value
  if not std.is_task(task) or
     not task:has_partitioning_constraints()
  then
    return
  end
  local params = task:get_param_symbols()
  local region_params = data.filter(
    function(param) return std.is_region(param:gettype()) end,
    params)
  local region_args = data.filter(
    function(arg) return std.is_region(std.as_read(arg.expr_type)) end,
    call.args)
  local region_arg_symbols = region_args:map(function(arg)
    assert(arg:is(ast.typed.expr.ID))
    return arg.value
  end)
  assert(#region_params == #region_arg_symbols)
  local mapping = data.newmap()
  data.zip(region_params, region_arg_symbols):map(function(pair)
    local param, arg = unpack(pair)
    mapping[param] = arg
  end)
  cx:add_task(task, mapping)
end

function collect_constraints.stat_var(cx, stat)
  if not (stat.value and stat.value:is(ast.typed.expr.Call)) then
    return
  end
  add_task(cx, stat.value)
end

function collect_constraints.stat_assignment_or_reduce(cx, stat)
  if not stat.rhs:is(ast.typed.expr.Call) then return end
  add_task(cx, stat.rhs)
end

function collect_constraints.stat_expr(cx, stat)
  if not stat.expr:is(ast.typed.expr.Call) then return end
  add_task(cx, stat.expr)
end

function collect_constraints.pass_through_stat(cx, stat) end

local collect_constraints_stat_table = {
  [ast.typed.stat.ParallelizeWith] = collect_constraints.stat_parallelize_with,

  [ast.typed.stat.Var]        = collect_constraints.stat_var,
  [ast.typed.stat.Assignment] = collect_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = collect_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Expr]       = collect_constraints.stat_expr,

  [ast.typed.stat.ForList]    = collect_constraints.stat_block,
  [ast.typed.stat.ForNum]     = collect_constraints.stat_block,
  [ast.typed.stat.While]      = collect_constraints.stat_block,
  [ast.typed.stat.Repeat]     = collect_constraints.stat_block,
  [ast.typed.stat.Block]      = collect_constraints.stat_block,
  [ast.typed.stat.MustEpoch]  = collect_constraints.stat_block,

  [ast.typed.stat.If]         = collect_constraints.stat_if,

  [ast.typed.stat]            = collect_constraints.pass_through_stat,
}

local collect_constraints_stat = ast.make_single_dispatch(
  collect_constraints_stat_table,
  {ast.typed.stat})

function collect_constraints.stat(cx, node)
  collect_constraints_stat(cx)(node)
end

function collect_constraints.block(cx, block)
  block.stats:map(function(stat) collect_constraints.stat(cx, stat) end)
end

local solver_context = {}

solver_context.__index = solver_context

function solver_context.new()
  local cx                   = {
    sources                  = hash_set.new(),
    sources_by_regions       = data.newmap(),
    constraints              = partitioning_constraints.new(),
    field_accesses           = data.newmap(),
    mappings_by_access_paths = data.newmap(),
    loop_ranges              = hash_set.new(),
    loop_range_partitions    = data.newmap(),
  }
  return setmetatable(cx, solver_context)
end

local function find_unifiable_ranges(constraints1, constraints2, source1, source2)
  -- TODO: Need to check if the resulting constraints are unsatisfiable
  local mapping = data.newmap()
  local worklist = terralib.newlist({data.newtuple(source1, source2)})
  local idx = 1
  while idx <= #worklist do
    local range1, range2 = unpack(worklist[idx])
    mapping[range1] = range2
    idx = idx + 1
    local all_constraints1 = constraints1.constraints[range1]
    local all_constraints2 = constraints2.constraints[range2]
    if all_constraints1 ~= nil and all_constraints2 ~= nil then
      for key, dst_range1 in all_constraints1:items() do
        local dst_range2 = all_constraints2[key]
        if dst_range2 ~= nil then
          worklist:insert(data.newtuple(dst_range1, dst_range2))
        end
      end
    end
  end
  return true, mapping
end

function solver_context:unify(name, new_constraints, region_mapping)
  local range_mapping = data.newmap()
  if new_constraints.constraints:is_empty() then
    return range_mapping
  elseif self.constraints:is_empty() then
    for source, _ in new_constraints.sources:items() do
      self.sources:insert(source)
    end
    for region, source in new_constraints.sources_by_regions:items() do
      local my_region = region_mapping[region]
      self.sources_by_regions[my_region] = source
    end
    self.constraints = new_constraints.constraints:clone(region_mapping)
    for region, accesses_summary in new_constraints.field_accesses_summary:items() do
      local my_region = region_mapping[region]
      local all_accesses = find_or_create(self.field_accesses, my_region)
      for field_path, summary in accesses_summary:items() do
        local accesses = find_or_create(all_accesses, field_path)
        local ranges_set, privilege = unpack(summary)
        if privilege == "reads_writes" then privilege = std.writes end
        local my_ranges_set = find_or_create(accesses, privilege, hash_set.new)
        my_ranges_set:insert_all(ranges_set)
      end
    end
    for range, _ in new_constraints.constraints.ranges:items() do
      range_mapping[range] = range
    end
    for _, range in new_constraints.loop_ranges:items() do
      self.loop_ranges:insert(range_mapping[range])
    end
    return range_mapping
  end

  local to_unify = new_constraints.constraints:clone(region_mapping)
  for region, source in new_constraints.sources_by_regions:items() do
    local my_region = region_mapping[region]
    local my_source = self.sources_by_regions[my_region]
    if my_source ~= nil then
      local unifiable, mapping =
        find_unifiable_ranges(to_unify, self.constraints, source, my_source)
      if unifiable then
        for src, tgt in mapping:items() do
          range_mapping[src] = tgt
        end
      end
    else
      local partition = to_unify:get_partition(source)
      for my_range, my_partition in self.constraints.ranges:items() do
        if partition == my_partition then
          range_mapping[source] = my_range
          break
        end
      end
    end
  end
  for range, partition in to_unify.ranges:items() do
    if range_mapping[range] == nil then
      range_mapping[range] = range
      if new_constraints.sources[range] then
        self.sources:insert(range)
        assert(self.sources_by_regions[partition.region] == nil)
        self.sources_by_regions[partition.region] = range
      end
    end
  end

  self.constraints:join(to_unify, range_mapping)

  for region, accesses_summary in new_constraints.field_accesses_summary:items() do
    local my_region = region_mapping[region]
    local all_accesses = find_or_create(self.field_accesses, my_region)
    for field_path, summary in accesses_summary:items() do
      local accesses = find_or_create(all_accesses, field_path)
      local ranges_set, privilege = unpack(summary)
      if privilege == "reads_writes" then privilege = std.writes end
      local my_ranges_set = find_or_create(accesses, privilege, hash_set.new)
      my_ranges_set:insert_all(ranges_set:map_table(range_mapping))
    end
  end

  for _, range in new_constraints.loop_ranges:items() do
    self.loop_ranges:insert(range_mapping[range])
  end

  return range_mapping
end

local function create_equal_partition(variable, region_symbol, color_space_symbol)
  local partition_type = std.partition(std.disjoint, region_symbol, color_space_symbol)
  variable:settype(partition_type)
  return ast.typed.stat.Var {
    symbol = variable,
    type = partition_type,
    value = ast.typed.expr.PartitionEqual {
      region = ast.typed.expr.ID {
        value = region_symbol,
        expr_type = std.rawref(&region_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      colors = ast.typed.expr.ID {
        value = color_space_symbol,
        expr_type = std.rawref(&color_space_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = partition_type,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function create_preimage_partition(variable, region_symbol, src_partition, key)
  local src_partition_type = src_partition:gettype()
  local partition_type = std.partition(src_partition_type.disjointness, region_symbol,
      src_partition_type.colors_symbol)
  variable:settype(partition_type)
  local mapping_region_symbol, field_path = unpack(key)
  return ast.typed.stat.Var {
    symbol = variable,
    type = partition_type,
    value = ast.typed.expr.Preimage {
      expr_type = partition_type,
      parent = ast.typed.expr.ID {
        value = region_symbol,
        expr_type = std.rawref(&region_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      partition = ast.typed.expr.ID {
        value = src_partition,
        expr_type = std.rawref(&src_partition:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      region = ast.typed.expr.RegionRoot {
        expr_type = mapping_region_symbol:gettype(),
        region = ast.typed.expr.ID {
          value = mapping_region_symbol,
          expr_type = std.rawref(&mapping_region_symbol:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        fields = terralib.newlist {field_path},
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function create_image_partition(variable, region_symbol, src_partition, key)
  local partition_type = std.partition(std.aliased, region_symbol,
      src_partition:gettype().colors_symbol)
  variable:settype(partition_type)
  local mapping_region_symbol, field_path = unpack(key)
  return ast.typed.stat.Var {
    symbol = variable,
    type = partition_type,
    value = ast.typed.expr.Image {
      expr_type = partition_type,
      parent = ast.typed.expr.ID {
        value = region_symbol,
        expr_type = std.rawref(&region_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      partition = ast.typed.expr.ID {
        value = src_partition,
        expr_type = std.rawref(&src_partition:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      region = ast.typed.expr.RegionRoot {
        expr_type = mapping_region_symbol:gettype(),
        region = ast.typed.expr.ID {
          value = mapping_region_symbol,
          expr_type = std.rawref(&mapping_region_symbol:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        fields = terralib.newlist {field_path},
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local op_name = {
  ["|"] = "or",
  ["-"] = "minus",
}

local function create_partition_by_binary_op(lhs, rhs, op)
  if rhs == nil then return lhs, nil end
  local lhs_type = lhs:gettype()
  local partition_type = std.partition(std.aliased,
      lhs_type.parent_region_symbol, lhs_type.colors_symbol)
  local variable = std.newsymbol(partition_type,
      lhs:getname() .. "_" .. op_name[op] .. "_" .. rhs:getname())
  return variable, ast.typed.stat.Var {
    symbol = variable,
    type = partition_type,
    value = ast.typed.expr.Binary {
      op = op,
      expr_type = partition_type,
      lhs = ast.typed.expr.ID {
        value = lhs,
        expr_type = std.rawref(&lhs_type),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      rhs = ast.typed.expr.ID {
        value = rhs,
        expr_type = std.rawref(&rhs:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function create_union_partition(lhs, rhs)
  return create_partition_by_binary_op(lhs, rhs, "|")
end

local function create_difference_partition(lhs, rhs)
  return create_partition_by_binary_op(lhs, rhs, "-")
end

local function create_union_partitions(partitions, cached_unions)
  local stats = terralib.newlist()
  if #partitions == 1 then return partitions[1], stats end

  -- Find the existing partitions that cover the current set of ranges
  local remaining = hash_set.from_list(partitions)
  local subsets = terralib.newlist()
  for ranges, partition in cached_unions:items() do
    if ranges <= remaining then
      subsets:insert({ranges, partition})
    end
  end
  table.sort(subsets, function(p1, p2) return p1[1]:size() > p2[1]:size() end)

  local to_join = terralib.newlist()
  for idx = 1, #subsets do
    local subset, partition = unpack(subsets[idx])
    if subset <= remaining then
      remaining:remove_all(subset)
      to_join:insert({subset, partition})
    end
    if remaining:size() <= 1 then break end
  end
  to_join:insertall(remaining:to_list():map(function(range)
    local singleton = hash_set.new()
    singleton:insert(range)
    return {singleton, range}
  end))

  while #to_join > 1 do
    local results = terralib.newlist()
    for idx = 1, #to_join, 2 do
      local lhs_ranges, lhs = unpack(to_join[idx])
      local rhs_ranges, rhs = unpack(to_join[idx + 1] or {hash_set.new(), nil})
      local result, stat = create_union_partition(lhs, rhs)
      local all_ranges = lhs_ranges + rhs_ranges
      if stat ~= nil then
        stats:insert(stat)
        cached_unions[all_ranges] = result
      end
      results:insert({all_ranges, result})
    end
    to_join = results
  end

  return to_join[1][2], stats
end

function solver_context:synthesize_partitions(color_space_symbol)
  local stats = terralib.newlist()

  local created = data.newmap()

  -- First, we find paths to disjoint child ranges. For each of the paths,
  -- we create a series of preimage partitions in the reverse order.
  local paths = self.constraints:find_paths_to_disjoint_children(self.sources)
  paths:map(function(path)
    local parent = nil
    for idx = 1, #path do
      local key, range = unpack(path[idx])
      local partition = self.constraints:get_partition(range)
      assert(created[range] == nil)
      created[range] = true
      if key == nil then
        assert(parent == nil)
        stats:insert(create_equal_partition(range, partition.region, color_space_symbol))
      else
        assert(parent ~= nil)
        local type, info = unpack(key)
        stats:insert(
          create_preimage_partition(range, partition.region, parent, info))
      end
      parent = range
    end
  end)

  -- Once we create all disjoint partitions, we derive all dependent partitions
  local worklist = paths:map(function(path)
    local key, range = unpack(path[#path])
    return range
  end)
  local idx = 1
  while idx <= #worklist do
    local src_range = worklist[idx]
    idx = idx + 1
    local image_constraints = self.constraints.constraints[src_range]
    if image_constraints ~= nil then
      for key, dst_range in image_constraints:items() do
        -- TODO: We need to create partitions other than image partitions
        local type, info = unpack(key)
        if created[dst_range] == nil then
          local dst_partition = self.constraints:get_partition(dst_range)
          stats:insert(
            create_image_partition(dst_range, dst_partition.region, src_range, info))
          created[dst_range] = true
        end
        worklist:insert(dst_range)
      end
    end
  end

  -- Finally, we create union partitions for accesses with multiple ranges
  local union_partitions = data.newmap()
  local diff_partitions = data.newmap()
  local mappings_by_range_sets = data.newmap()

  for region_symbol, accesses_summary in self.field_accesses:items() do
    for field_path, summary in accesses_summary:items() do
      local all_ranges = hash_set.new()
      local has_reduce = false

      for privilege, range_set in summary:items() do
        all_ranges:insert_all(range_set)
        has_reduce = has_reduce or std.is_reduce(privilege)
      end

      local mapping = nil

      if all_ranges:size() == 1 then
        local range = all_ranges:to_list()[1]
        mapping = data.newmap()
        mapping[range] = terralib.newlist({range})

        local partition_type = range:gettype()
        if partition_type:is_disjoint() and self.loop_ranges:has(range) then
          self.loop_range_partitions[range] = mapping[range]
        end

      else
        local primary_range = nil
        local secondary_ranges = terralib.newlist()
        all_ranges:foreach(function(range)
          local partition_type = range:gettype()
          if partition_type:is_disjoint() then
            assert(primary_range == nil)
            primary_range = range
          else
            secondary_ranges:insert(range)
          end
        end)
        assert(primary_range ~= nil)

        mapping = mappings_by_range_sets[all_ranges]
        if mapping == nil then
          mapping = data.newmap()
          -- TODO: Need to create private-vs-shared partitions for reductions
          local primary_partitions = terralib.newlist({primary_range})
          mapping[primary_range] = primary_partitions
          if self.loop_ranges:has(primary_range) then
            self.loop_range_partitions[primary_range] = primary_partitions
          end

          local union_range, union_partition_stats =
            create_union_partitions(secondary_ranges, union_partitions)
          stats:insertall(union_partition_stats)
          local ghost_range =
            diff_partitions[data.newtuple(union_range, primary_range)]
          if ghost_range == nil then
            local diff_range, diff_partition_stat =
              create_difference_partition(union_range, primary_range)
            stats:insert(diff_partition_stat)
            ghost_range = diff_range
            diff_partitions[data.newtuple(union_range, primary_range)] = diff_range
          end

          local ghost_ranges = terralib.newlist({primary_range, ghost_range})
          secondary_ranges:map(function(range) mapping[range] = ghost_ranges end)
          mappings_by_range_sets[all_ranges] = mapping
        end
      end

      assert(mapping ~= nil)
      local key = data.newtuple(region_symbol, field_path)
      self.mappings_by_access_paths[key] = mapping
    end
  end

  return stats
end

function solver_context:print_all_constraints()
  print("################")
  print("* sources:")
  for source, _ in self.sources:items() do
    print("    " .. tostring(source))
  end
  print("* sources by regions:")
  for region, source in self.sources_by_regions:items() do
    print("    " .. tostring(region) .. " -> " .. tostring(source))
  end
  self.constraints:print_constraints()
  print("* accesses:")
  for region_symbol, accesses in self.field_accesses:items() do
    for field_path, summary in accesses:items() do
      print("    " .. tostring(region_symbol) .. "." .. field_path:mkstring("", ".", ""))
      for privilege, ranges_set in summary:items() do
        print("        " .. tostring(region_symbol) .. "[" ..
          tostring(ranges_set) .. "]." ..  field_path:mkstring("", ".", "") ..
          " @ " .. tostring(privilege))
      end
    end
  end
  print("################")
end

local solve_constraints = {}

function solve_constraints.solve(stat)
  assert(stat:is(ast.typed.stat.ParallelizeWith))
  local collector_cx = collector_context.new()
  collect_constraints.block(collector_cx, stat.block)

  local all_tasks = collector_cx.all_tasks:map(function(task) return task end)
  if #all_tasks == 0 then
    local solution = {
      all_tasks = all_tasks,
      all_mappings = false,
      mappings_by_access_paths = false,
      loop_range_partitions = false,
      color_space_symbol = false,
      partition_stats = false,
    }
    return solution
  end

  table.sort(all_tasks, cmp_tasks)

  -- TODO: 1) We need to enforce that either a color space or at least one
  --          partition must be provided
  --       2) We need to handle hints other than color space
  local color_space_symbol = nil
  for idx = 1, #stat.hints do
    local hint = stat.hints[idx]
    if hint:is(ast.typed.expr.ID) and
       std.is_ispace(std.as_read(hint.expr_type))
    then
      color_space_symbol = hint.value
    end
  end
  assert(color_space_symbol ~= nil)

  local solver_cx = solver_context.new()
  local mappings = all_tasks:map(function(task)
    local constraints = task:get_partitioning_constraints()
    local region_mapping = collector_cx:get_mapping(task)
    local range_mapping = solver_cx:unify(task.name, constraints, region_mapping)
    return {range_mapping, region_mapping}
  end)

  solver_cx.constraints:print_constraints()

  local partition_stats =
    solver_cx:synthesize_partitions(color_space_symbol)

  local all_mappings = {}
  for idx = 1, #all_tasks do
    all_mappings[all_tasks[idx]] = mappings[idx]
  end

  return {
    all_tasks = all_tasks,
    all_mappings = all_mappings,
    mappings_by_access_paths = solver_cx.mappings_by_access_paths,
    loop_range_partitions = solver_cx.loop_range_partitions,
    color_space_symbol = color_space_symbol,
    partition_stats = partition_stats,
  }
end

return solve_constraints
