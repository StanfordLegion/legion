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
local pretty = require("regent/pretty")
local std = require("regent/std")

local hash_set                 = require("regent/parallelizer/hash_set")
local partition_info           = require("regent/parallelizer/partition_info")
local partitioning_constraints = require("regent/parallelizer/partitioning_constraints")
local ranges                   = require("regent/parallelizer/ranges")

local c = std.c

local new_range = ranges.new

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
  local function is_mappable_type(t)
    return std.is_region(t) or std.is_rect_type(t)
  end
  local mappable_params = data.filter(
    function(param) return is_mappable_type(param:gettype()) end,
    params)
  local mappable_args = data.filter(
    function(arg) return is_mappable_type(std.as_read(arg.expr_type)) end,
    call.args)
  local mappable_arg_symbols = mappable_args:map(function(arg)
    assert(arg:is(ast.typed.expr.ID))
    return arg.value
  end)
  assert(#mappable_params == #mappable_arg_symbols)
  local mapping = data.newmap()
  data.zip(mappable_params, mappable_arg_symbols):map(function(pair)
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

function solver_context.new(task_constraints)
  local cx                   = {
    sources                  = hash_set.new(),
    sources_by_regions       = data.newmap(),
    constraints              = partitioning_constraints.new(),
    field_accesses           = data.newmap(),
    mappings_by_access_paths = data.newmap(),
    loop_ranges              = hash_set.new(),
    loop_range_partitions    = data.newmap(),
    incl_check_caches        = data.newmap(),
    task_constraints         = task_constraints,
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
        local privileges = terralib.newlist()
        if privilege == "reads_writes" then
          privileges:insert(std.reads)
          privileges:insert(std.writes)
        else
          privileges:insert(privilege)
        end
        privileges:map(function(privilege)
          local my_ranges_set = find_or_create(accesses, privilege, hash_set.new)
          my_ranges_set:insert_all(ranges_set)
        end)
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
      local privileges = terralib.newlist()
      if privilege == "reads_writes" then
        privileges:insert(std.reads)
        privileges:insert(std.writes)
      else
        privileges:insert(privilege)
      end
      local mapped_ranges_set = ranges_set:map_table(range_mapping)
      privileges:map(function(privilege)
        local my_ranges_set = find_or_create(accesses, privilege, hash_set.new)
        my_ranges_set:insert_all(mapped_ranges_set)
      end)
    end
  end

  for _, range in new_constraints.loop_ranges:items() do
    self.loop_ranges:insert(range_mapping[range])
  end

  return range_mapping
end

local function create_equal_partition(variable, region_symbol, color_space_symbol,
    existing_disjoint_partitions)
  -- TODO: Here we should also check if the existing partition is complete
  --       once completeness of partitions is tracked
  if existing_disjoint_partitions[region_symbol] ~= nil then
    local partition = existing_disjoint_partitions[region_symbol]
    variable:settype(partition:gettype())
    return ast.typed.stat.Var {
      symbol = variable,
      type = variable:gettype(),
      value = ast.typed.expr.ID {
        value = partition,
        expr_type = std.rawref(&partition:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    }
  else
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
end

local function create_preimage_partition(variable, region_symbol, src_partition, info)
  assert(not info:is_affine())
  local src_partition_type = src_partition:gettype()
  local partition_type = std.partition(src_partition_type.disjointness, region_symbol,
      src_partition_type.colors_symbol)
  variable:settype(partition_type)
  local mapping_region_symbol, field_path = unpack(info.info)
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

local function create_image_partition(variable, region_symbol, src_partition, info)
  if info:is_image() then
    local partition_type = std.partition(std.aliased, region_symbol,
        src_partition:gettype().colors_symbol)
    variable:settype(partition_type)
    local mapping_region_symbol, field_path = unpack(info.info)
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
  else
    assert(info:is_affine())

    local src_partition_type = src_partition:gettype()
    local colors_symbol = src_partition_type.colors_symbol
    -- TODO: affine partitions are actually disjoint except for modulo case
    local partition_type = std.partition(std.aliased, region_symbol, colors_symbol)
    variable:settype(partition_type)
    local stats = terralib.newlist()

    local arg_symbols = terralib.newlist({
      std.newsymbol(c.legion_runtime_t, "runtime"),
      std.newsymbol(c.legion_context_t, "context"),
      std.newsymbol(c.legion_logical_region_t, "handle"),
      std.newsymbol(c.legion_logical_partition_t, "projection"),
      std.newsymbol(c.regent_affine_descriptor_t, "descriptor"),
    })

    stats:insert(ast.typed.stat.Var {
      symbol = arg_symbols[1],
      type = arg_symbols[1]:gettype(),
      value = ast.typed.expr.RawRuntime {
        expr_type = c.legion_runtime_t,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = arg_symbols[2],
      type = arg_symbols[2]:gettype(),
      value = ast.typed.expr.RawContext {
        expr_type = c.legion_context_t,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = arg_symbols[3],
      type = arg_symbols[3]:gettype(),
      value = ast.typed.expr.RawValue {
        value = ast.typed.expr.ID {
          value = region_symbol,
          expr_type = std.rawref(&region_symbol:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = c.legion_logical_region_t,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = arg_symbols[4],
      type = arg_symbols[4]:gettype(),
      value = ast.typed.expr.RawValue {
        value = ast.typed.expr.ID {
          value = src_partition,
          expr_type = std.rawref(&src_partition:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = c.legion_logical_partition_t,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = arg_symbols[5],
      type = arg_symbols[5]:gettype(),
      value = false,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    local offset = nil
    local modulo = false
    if std.is_symbol(info.info[#info.info]) then
      offset = terralib.newlist({unpack(info.info:slice(1, #info.info - 1))})
      modulo = info.info[#info.info]
    else
      offset = terralib.newlist({unpack(info.info)})
    end
    local index_type = std["int" .. #offset .. "d"]
    local rect_type = std["rect" .. #offset .. "d"]
    local desc = ast.typed.expr.ID {
      value = arg_symbols[5],
      expr_type = std.rawref(&arg_symbols[5]:gettype()),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    }
    stats:insert(ast.typed.stat.Assignment {
      lhs = ast.typed.expr.FieldAccess {
        value = desc,
        field_name = "offset",
        expr_type = std.rawref(&c.legion_domain_point_t),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      rhs = ast.typed.expr.Cast {
        fn = ast.typed.expr.Function {
          value = index_type,
          expr_type = index_type,
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        arg = ast.typed.expr.Ctor {
          named = false,
          fields = offset:map(function(o)
            return ast.typed.expr.CtorListField {
              value = ast.typed.expr.Constant {
                value = o,
                expr_type = int,
                span = ast.trivial_span(),
                annotations = ast.default_annotations(),
              },
              expr_type = int,
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            }
          end),
          expr_type = std.ctor_tuple(offset:map(function(_) return int end)),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = index_type,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      metadata = false,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    if modulo then
      stats:insert(ast.typed.stat.Assignment {
        lhs = ast.typed.expr.FieldAccess {
          value = desc,
          field_name = "modulo",
          expr_type = std.rawref(&c.legion_domain_t),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        rhs = ast.typed.expr.ID {
          value = modulo,
          expr_type = rect_type,
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        metadata = false,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      })
    end
    stats:insert(ast.typed.stat.Assignment {
      lhs = ast.typed.expr.FieldAccess {
        value = desc,
        field_name = "is_modulo",
        expr_type = std.rawref(&bool),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      rhs = ast.typed.expr.Constant {
        value = modulo and true,
        expr_type = bool,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      metadata = false,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })

    local args = arg_symbols:map(function(arg)
      return ast.typed.expr.ID {
        value = arg,
        expr_type = std.rawref(&arg:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      }
    end)
    args:insert(ast.typed.expr.Constant {
      value = partition_type:is_disjoint() and c.DISJOINT_KIND or c.COMPUTE_KIND,
      expr_type = c.legion_partition_kind_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    args:insert(ast.typed.expr.Constant {
      value = -1,
      expr_type = int,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    local raw_result_lp = std.newsymbol(c.legion_logical_partition_t, "raw_result_lp")
    stats:insert(ast.typed.stat.Var {
      symbol = raw_result_lp,
      type = raw_result_lp:gettype(),
      value = ast.typed.expr.Call {
        fn = ast.typed.expr.Function {
          value = c.legion_logical_partition_create_by_affine_image,
          expr_type = c.legion_logical_partition_create_by_affine_image:gettype(),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = c.legion_logical_partition_t,
        args = args,
        conditions = terralib.newlist(),
        replicable = true,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })

    stats:insert(ast.typed.stat.Var {
      symbol = variable,
      type = variable:gettype(),
      value = ast.typed.expr.ImportPartition {
        region = ast.typed.expr.ID {
          value = region_symbol,
          expr_type = std.rawref(&region_symbol:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        colors = ast.typed.expr.ID {
          value = colors_symbol,
          expr_type = std.rawref(&colors_symbol:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        value = ast.typed.expr.ID {
          value = raw_result_lp,
          expr_type = std.rawref(&raw_result_lp:gettype()),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = partition_type,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })

    return stats
  end
end

local op_name = {
  ["|"] = "|",
  ["-"] = "-",
  ["&"] = "&",
}

local function create_partition_by_binary_op(disjointness, lhs, rhs, op, result_symbol)
  if rhs == nil then return lhs, nil end
  local lhs_type = lhs:gettype()
  local partition_type = std.partition(disjointness,
      lhs_type.parent_region_symbol, lhs_type.colors_symbol)
  local variable = result_symbol
  if variable == nil then
    if std.config["parallelize-debug"] then
      variable = std.newsymbol(partition_type,
          "(".. lhs:getname() .. op_name[op] .. rhs:getname() .. ")")
    else
      variable = new_range()
      variable:settype(partition_type)
    end
  end
  return variable, ast.typed.stat.Var {
    symbol = variable,
    type = variable:gettype(),
    value = ast.typed.expr.Binary {
      op = op,
      expr_type = variable:gettype(),
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

local function create_union_partition(lhs, rhs, result_symbol)
  return create_partition_by_binary_op(std.aliased, lhs, rhs, "|", result_symbol)
end

local function create_disjoint_union_partition(lhs, rhs, result_symbol)
  return create_partition_by_binary_op(std.disjoint, lhs, rhs, "|", result_symbol)
end
local function create_difference_partition(lhs, rhs, result_symbol)
  return create_partition_by_binary_op(lhs:gettype().disjointness, lhs, rhs, "-", result_symbol)
end

local function create_intersection_partition(lhs, rhs, result_symbol)
  local disjointness = std.aliased
  if lhs:gettype():is_disjoint() or rhs:gettype():is_disjoint() then
    disjointness = std.disjoint
  end
  return create_partition_by_binary_op(disjointness, lhs, rhs, "&", result_symbol)
end

local terra _create_complement_partition(runtime  : c.legion_runtime_t,
                                         context  : c.legion_context_t,
                                         lp       : c.legion_logical_partition_t)
  var ip = lp.index_partition
  var parent_is =
    c.legion_index_partition_get_parent_index_space(runtime, ip)

  var cs = c.legion_index_partition_get_color_space(runtime, ip)

  var complement : c.legion_index_space_t
  if false then
    var dom = c.legion_index_space_get_domain(runtime, cs)
    var vol = c.legion_domain_get_volume(dom)
    var spaces =
      [&c.legion_index_space_t](c.malloc([sizeof(c.legion_index_space_t)] * vol))

    var it = c.legion_domain_point_iterator_create(dom)
    var idx = 0
    while c.legion_domain_point_iterator_has_next(it) do
      var color = c.legion_domain_point_iterator_next(it)
      spaces[idx] =
        c.legion_index_partition_get_index_subspace_domain_point(runtime, ip, color)
      idx = idx + 1
    end
    c.legion_domain_point_iterator_destroy(it)

    var union_is = c.legion_index_space_union(runtime, context, spaces, vol)
    complement = c.legion_index_space_subtraction(runtime, context, parent_is, union_is)
  else
    var union_pt : c.legion_point_1d_t, complement_pt : c.legion_point_1d_t
    union_pt.x[0], complement_pt.x[0]  = 0, 1
    var color_rect = [c.legion_rect_1d_t] { lo = union_pt, hi = complement_pt }

    var union_color = c.legion_domain_point_from_point_1d(union_pt)
    var complement_color = c.legion_domain_point_from_point_1d(complement_pt)
    var color_domain = c.legion_domain_from_rect_1d(color_rect)
    var color_space = c.legion_index_space_create_domain(runtime, context, color_domain)

    var pending_ip =
      c.legion_index_partition_create_pending_partition(runtime, context, parent_is,
          color_space, c.DISJOINT_KIND, -1)

    var union_is : c.legion_index_space_t[1]
    union_is[0] = c.legion_index_partition_create_index_space_union_partition(
        runtime, context, pending_ip, union_color, ip)
    complement = c.legion_index_partition_create_index_space_difference(
        runtime, context, pending_ip, complement_color, parent_is, union_is, 1)
  end

  var complement_ip = c.legion_index_partition_create_equal(runtime, context,
      complement, cs, 1, -1)

  return c.legion_logical_partition_create_by_tree(
      runtime, context, complement_ip, lp.field_space, lp.tree_id)
end

local function create_complement_partition(range)
  local stats = terralib.newlist()
  local arg_symbols = terralib.newlist({
    std.newsymbol(c.legion_runtime_t, "runtime"),
    std.newsymbol(c.legion_context_t, "context"),
    std.newsymbol(c.legion_logical_partition_t, "raw_" .. range:getname()),
  })
  local args = arg_symbols:map(function(arg)
    return ast.typed.expr.ID {
      value = arg,
      expr_type = std.rawref(&arg:gettype()),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    }
  end)
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[1],
    type = arg_symbols[1]:gettype(),
    value = ast.typed.expr.RawRuntime {
      expr_type = c.legion_runtime_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[2],
    type = arg_symbols[2]:gettype(),
    value = ast.typed.expr.RawContext {
      expr_type = c.legion_context_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[3],
    type = arg_symbols[3]:gettype(),
    value = ast.typed.expr.RawValue {
      value = ast.typed.expr.ID {
        value = range,
        expr_type = std.rawref(&range:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = c.legion_logical_partition_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })

  local parent_region_symbol = range:gettype().parent_region_symbol
  local color_space_symbol = range:gettype().colors_symbol
  local complement = new_range()
  complement:settype(
    std.partition(std.disjoint, parent_region_symbol, color_space_symbol))
  local raw_complement = std.newsymbol(c.legion_logical_partition_t,
                                        "raw_" .. complement:getname())

  stats:insert(ast.typed.stat.Var {
    symbol = raw_complement,
    type = raw_complement:gettype(),
    value = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = _create_complement_partition,
        expr_type = _create_complement_partition:gettype(),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = c.legion_logical_partition_t,
      args = args,
      conditions = terralib.newlist(),
      replicable = true,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })

  stats:insert(ast.typed.stat.Var {
    symbol = complement,
    type = complement:gettype(),
    value = ast.typed.expr.ImportPartition {
      region = ast.typed.expr.ID {
        value = parent_region_symbol,
        expr_type = std.rawref(&parent_region_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      colors = ast.typed.expr.ID {
        value = color_space_symbol,
        expr_type = std.rawref(&color_space_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      value = ast.typed.expr.ID {
        value = raw_complement,
        expr_type = std.rawref(&raw_complement:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = complement:gettype(),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  return complement, stats
end

local terra _create_pvs_partition(runtime : c.legion_runtime_t,
                                  context : c.legion_context_t,
                                  private : c.legion_logical_partition_t,
                                  shared : c.legion_logical_partition_t,
                                  check : bool)
  var private_ip = private.index_partition
  var shared_ip = shared.index_partition
  var parent_is =
    c.legion_index_partition_get_parent_index_space(runtime, private_ip)

  var private_pt : c.legion_point_1d_t, shared_pt : c.legion_point_1d_t
  private_pt.x[0], shared_pt.x[0]  = 0, 1
  var color_rect = [c.legion_rect_1d_t] { lo = private_pt, hi = shared_pt }

  var private_color = c.legion_domain_point_from_point_1d(private_pt)
  var shared_color = c.legion_domain_point_from_point_1d(shared_pt)
  var color_domain = c.legion_domain_from_rect_1d(color_rect)
  var color_space = c.legion_index_space_create_domain(runtime, context, color_domain)

  var pvs_partition : c.legion_index_partition_t
  if check then
    pvs_partition =
      c.legion_index_partition_create_pending_partition(runtime, context, parent_is,
        color_space, c.COMPUTE_KIND, -1)
  else
    pvs_partition =
      c.legion_index_partition_create_pending_partition(runtime, context, parent_is,
          color_space, c.DISJOINT_COMPLETE_KIND, -1)
  end

  var union_is : c.legion_index_space_t[1]
  union_is[0] = c.legion_index_partition_create_index_space_union_partition(
      runtime, context, pvs_partition, private_color, private_ip)
  c.legion_index_partition_create_index_space_difference(
      runtime, context, pvs_partition, shared_color, parent_is, union_is, 1)

  if check then
    std.assert(c.legion_index_partition_is_disjoint(runtime, pvs_partition),
               "private-vs-shared partition turns out to be aliased")
    std.assert(c.legion_index_partition_is_complete(runtime, pvs_partition),
               "private-vs-shared partition turns out to be incomplete")
  end

  return c.legion_logical_partition_create_by_tree(
      runtime, context, pvs_partition, private.field_space, private.tree_id)
end

local function create_pvs_partition(private, shared)
  local stats = terralib.newlist()
  local arg_symbols = terralib.newlist({
    std.newsymbol(c.legion_runtime_t, "runtime"),
    std.newsymbol(c.legion_context_t, "context"),
    std.newsymbol(c.legion_logical_partition_t, "private"),
    std.newsymbol(c.legion_logical_partition_t, "shared"),
  })
  local args = arg_symbols:map(function(arg)
    return ast.typed.expr.ID {
      value = arg,
      expr_type = std.rawref(&arg:gettype()),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    }
  end)
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[1],
    type = arg_symbols[1]:gettype(),
    value = ast.typed.expr.RawRuntime {
      expr_type = c.legion_runtime_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[2],
    type = arg_symbols[2]:gettype(),
    value = ast.typed.expr.RawContext {
      expr_type = c.legion_context_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[3],
    type = arg_symbols[3]:gettype(),
    value = ast.typed.expr.RawValue {
      value = ast.typed.expr.ID {
        value = private,
        expr_type = std.rawref(&private:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = c.legion_logical_partition_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = arg_symbols[4],
    type = arg_symbols[4]:gettype(),
    value = ast.typed.expr.RawValue {
      value = ast.typed.expr.ID {
        value = shared,
        expr_type = std.rawref(&shared:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = c.legion_logical_partition_t,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })

  local parent_region_symbol = private:gettype().parent_region_symbol
  local pvs_color_space_symbol = std.newsymbol(std.ispace(std.int1d), "pvs_colors")
  local pvs_partition_type =
    std.partition(std.disjoint, parent_region_symbol, pvs_color_space_symbol)
  local pvs_partition = nil
  if std.config["parallelize-debug"] then
    pvs_partition = std.newsymbol(pvs_partition_type,
        private:getname() .. "_vs_" .. shared:getname())
  else
    pvs_partition = new_range()
    pvs_partition:settype(pvs_partition_type)
  end
  local raw_pvs_partition =
    std.newsymbol(c.legion_logical_partition_t, "raw_" .. pvs_partition:getname())

  args:insert(ast.typed.expr.Constant {
    value = std.config["parallelize-debug"],
    expr_type = bool,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = raw_pvs_partition,
    type = raw_pvs_partition:gettype(),
    value = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = _create_pvs_partition,
        expr_type = _create_pvs_partition:gettype(),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = c.legion_logical_partition_t,
      args = args,
      conditions = terralib.newlist(),
      replicable = true,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })

  stats:insert(ast.typed.stat.Var {
    symbol = pvs_color_space_symbol,
    type = pvs_color_space_symbol:gettype(),
    value = ast.typed.expr.Ispace {
      index_type = std.int1d,
      expr_type = pvs_color_space_symbol:gettype(),
      extent = ast.typed.expr.Ctor {
        named = false,
        fields = terralib.newlist({
          ast.typed.expr.CtorListField {
            value = ast.typed.expr.Constant {
              value = 0,
              expr_type = int,
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            },
            expr_type = int,
            span = ast.trivial_span(),
            annotations = ast.default_annotations(),
          },
          ast.typed.expr.CtorListField {
            value = ast.typed.expr.Constant {
              value = 1,
              expr_type = int,
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            },
            expr_type = int,
            span = ast.trivial_span(),
            annotations = ast.default_annotations(),
          },
        }),
        expr_type = std.ctor_tuple(terralib.newlist({int, int})),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      start = false,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })

  stats:insert(ast.typed.stat.Var {
    symbol = pvs_partition,
    type = pvs_partition:gettype(),
    value = ast.typed.expr.ImportPartition {
      region = ast.typed.expr.ID {
        value = parent_region_symbol,
        expr_type = std.rawref(&parent_region_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      colors = ast.typed.expr.ID {
        value = pvs_color_space_symbol,
        expr_type = std.rawref(&pvs_color_space_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      value = ast.typed.expr.ID {
        value = raw_pvs_partition,
        expr_type = std.rawref(&raw_pvs_partition:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = pvs_partition_type,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  return pvs_partition, stats
end

local function retrieve_subregion(cx, pvs, color)
  local index = ast.typed.expr.Constant {
    value = color,
    expr_type = int,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
  local pvs_type = pvs:gettype()
  local parent_type = pvs_type:parent_region()
  local subregion_type = pvs_type:subregion_constant(index)
  local subregion_symbol =
    std.newsymbol(subregion_type, pvs:getname() .. "_" ..tostring(color))
  std.add_constraint(cx.task_constraints, pvs_type, parent_type, std.subregion, false)
  std.add_constraint(cx.task_constraints, subregion_type, pvs_type, std.subregion, false)
  return subregion_symbol, ast.typed.stat.Var {
    symbol = subregion_symbol,
    type = subregion_type,
    value = ast.typed.expr.IndexAccess {
      index = index,
      value = ast.typed.expr.ID {
        value = pvs,
        expr_type = std.rawref(&pvs:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = subregion_type,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function retrieve_private_shared_subregions(cx, pvs)
  local stats = terralib.newlist()
  local private_subregion, stat = retrieve_subregion(cx, pvs, 0)
  stats:insert(stat)
  local shared_subregion, stat = retrieve_subregion(cx, pvs, 1)
  stats:insert(stat)
  std.add_constraint(cx.task_constraints, private_subregion:gettype(),
                     shared_subregion:gettype(), std.disjointness, true)
  return private_subregion, shared_subregion, stats
end

local function create_intersection_partition_region(lhs, rhs, result_symbol)
  local rhs_type = rhs:gettype()
  local result_type =
    std.partition(rhs_type.disjointness, lhs, rhs_type.colors_symbol)
  if result_symbol ~= nil then
    result_symbol:settype(result_type)
  else
    if std.config["parallelize-debug"] then
      result_symbol = std.newsymbol(result_type,
          "(" .. lhs:getname() .. "&" .. rhs:getname() .. ")")
    else
      result_symbol = new_range()
      result_symbol:settype(result_type)
    end
  end

  return result_symbol, ast.typed.stat.Var {
    symbol = result_symbol,
    type = result_type,
    value = ast.typed.expr.Binary {
      op = "&",
      expr_type = result_type,
      lhs = ast.typed.expr.ID {
        value = lhs,
        expr_type = std.rawref(&lhs:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      rhs = ast.typed.expr.ID {
        value = rhs,
        expr_type = std.rawref(&rhs_type),
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


local function create_union_partitions(partitions, cached_unions)
  local stats = terralib.newlist()
  if #partitions == 1 then return partitions[1], stats end

  -- Find the existing partitions that cover the current set of ranges
  local remaining = terralib.islist(partitions) and hash_set.from_list(partitions) or
                    partitions
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

local function create_isomorphic_region(name, orig_region, elem_type)
  local stats = terralib.newlist()
  local ispace_symbol = std.newsymbol(orig_region:gettype():ispace())
  local region_type = std.region(ispace_symbol, elem_type)
  local region_symbol = std.newsymbol(region_type, name)

  stats:insert(ast.typed.stat.Var {
    symbol = ispace_symbol,
    type = ispace_symbol:gettype(),
    value = ast.typed.expr.FieldAccess {
      value = ast.typed.expr.ID {
        value = orig_region,
        expr_type = std.rawref(&orig_region:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      field_name = "ispace",
      expr_type = orig_region:gettype():ispace(),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  stats:insert(ast.typed.stat.Var {
    symbol = region_symbol,
    type = region_symbol:gettype(),
    value = ast.typed.expr.Region {
      ispace = ast.typed.expr.ID {
        value = ispace_symbol,
        expr_type = std.rawref(&ispace_symbol:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      fspace_type = elem_type,
      expr_type = region_symbol:gettype(),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  })
  return region_symbol, stats
end

local function create_index_fill(cx, partition, color_space_symbol, elem_type, value)
  local partition_type = partition:gettype()
  local parent_type = partition_type:parent_region()
  local stats = terralib.newlist()
  local color_symbol = std.newsymbol(color_space_symbol:gettype().index_type, "color")
  local color = ast.typed.expr.ID {
    value = color_symbol,
    expr_type = std.rawref(&color_symbol:gettype()),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
  local subregion_type = partition_type:subregion_constant(color)
  std.add_constraint(cx, partition_type, parent_type, std.subregion, false)
  std.add_constraint(cx, subregion_type, partition_type, std.subregion, false)
  return ast.typed.stat.ForList {
    symbol = color_symbol,
    value = ast.typed.expr.ID {
      value = color_space_symbol,
      expr_type = std.rawref(&color_space_symbol:gettype()),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    block = ast.typed.Block {
      stats = terralib.newlist({
        ast.typed.stat.Expr {
          expr = ast.typed.expr.Fill {
            dst = ast.typed.expr.RegionRoot {
              region = ast.typed.expr.IndexAccess {
                value = ast.typed.expr.ID {
                  value = partition,
                  expr_type = std.rawref(&partition:gettype()),
                  span = ast.trivial_span(),
                  annotations = ast.default_annotations(),
                },
                index = color,
                expr_type = subregion_type,
                span = ast.trivial_span(),
                annotations = ast.default_annotations(),
              },
              fields = terralib.newlist({data.newtuple()}),
              expr_type = subregion_type,
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            },
            value = (value and ast.typed.expr.Constant {
              value = value,
              expr_type = elem_type,
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            }) or color,
            expr_type = terralib.types.unit,
            conditions = terralib.newlist(),
            span = ast.trivial_span(),
            annotations = ast.default_annotations(),
          },
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        }
      }),
      span = ast.trivial_span(),
    },
    metadata = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function create_assignment(lhs, rhs)
  lhs:settype(rhs:gettype())
  return ast.typed.stat.Var {
    symbol = lhs,
    type = rhs:gettype(),
    value = ast.typed.expr.ID {
      value = rhs,
      expr_type = std.rawref(&rhs:gettype()),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function dynamic_cast_to_disjoint_partition(range)
  local disjoint_range = new_range()
  local orig_partition_type = range:gettype()
  disjoint_range:settype(
    std.partition(std.disjoint,
      orig_partition_type.parent_region_symbol,
      orig_partition_type.colors_symbol))
  return disjoint_range, ast.typed.stat.Var {
    symbol = disjoint_range,
    type = disjoint_range:gettype(),
    value = ast.typed.expr.DynamicCast {
      value = ast.typed.expr.ID {
        value = range,
        expr_type = std.rawref(&range:gettype()),
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      expr_type = disjoint_range:gettype(),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function combine_mapping(m1, m2)
  local m = data.newmap()
  for from, to in m2:items() do
    m[from] = m1[to] or to
  end
  return m
end

function solver_context:synthesize_partitions(existing_disjoint_partitions, color_space_symbol)
  local stats = terralib.newlist()

  local ts_start, ts_end
  if std.config["parallelize-time-deppart"] then
    ts_start = std.newsymbol(uint64, "__ts_start")
    stats:insert(ast.typed.stat.Fence {
      kind = ast.fence_kind.Execution {},
      blocking = true,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = ts_start,
      type = uint64,
      value = ast.typed.expr.Call {
        fn = ast.typed.expr.Function {
          value = c.legion_get_current_time_in_micros,
          expr_type = c.legion_get_current_time_in_micros:gettype(),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = uint64,
        args = terralib.newlist(),
        conditions = terralib.newlist(),
        replicable = true,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
  end

  local created = hash_set.new()
  for region, partition in existing_disjoint_partitions:items() do
    created:insert(partition)
  end
  local disjoint_partitions = data.newmap()
  local preimage_partitions = data.newmap()
  local image_partitions = data.newmap()
  local union_partitions = data.newmap()
  local diff_partitions = data.newmap()
  local mappings_by_range_sets = data.newmap()
  local ghost_to_primary = data.newmap()

  -- Sort source ranges for deterministic compilation
  self.sources:canonicalize()

  -- First try to find constraints subsumed by others. Specifically,
  -- when we have an image constraint C = R[P1] <= P2 and we need a
  -- disjoint partition for P1, then any image constraint R[P3] <= P4,
  -- for which we also know P3 <= P1, is discharged by C. Using
  -- subsumptions, we can further unify partitions.

  -- TODO: We look at only immediate children of subset constraints
  --       and do not traverse down distant children.
  local subsumed_constraints = terralib.newlist()
  local unifiable_ranges = data.newmap()
  self.sources:foreach(function(source)
    if self.constraints.constraints[source] ~= nil then
      -- Find all image or affine constraints of the source.
      local source_constraints = data.newmap()
      for info, dst_range in self.constraints.constraints[source]:items() do
        if not info:is_subset() then
          source_constraints[info] = dst_range
        end
      end

      -- Then, find all ranges that are subsumed by the source
      local source_region = self.constraints:get_partition(source).region
      local subsumed_ranges = terralib.newlist()
      for src_range, constraints in self.constraints.constraints:items() do
        for info, dst_range in constraints:items() do
          if info:is_subset() and
             source_region == self.constraints:get_partition(dst_range).region
          then
            subsumed_ranges:insert({src_range, dst_range})
          end
        end
      end

      -- Finally, find all subsumed constraints
      for _, pair in ipairs(subsumed_ranges) do
        local subsumed_range, subsumed_parent = unpack(pair)
        for info, dst_range in self.constraints.constraints[subsumed_range]:items() do
          local source_dst_range = source_constraints[info]
          if source_dst_range ~= nil then
            subsumed_constraints:insert({subsumed_range, info})
            unifiable_ranges[dst_range] = source_dst_range
            unifiable_ranges[subsumed_parent] = source
          end
        end
      end
    end
  end)

  if #subsumed_constraints > 0 then
    for _, pair in ipairs(subsumed_constraints) do
      local range, info = unpack(pair)
      self.constraints:remove_constraint(range, info)
    end
    self.constraints = self.constraints:clone(unifiable_ranges)

    local function mapping(range) return unifiable_ranges[range] or range end
    local field_accesses = data.newmap()
    for region_symbol, accesses_summary in self.field_accesses:items() do
      local new_access_summary = data.newmap()
      for field_path, summary in accesses_summary:items() do
        local new_summary = data.newmap()
        for privilege, ranges_set in summary:items() do
          local unified = ranges_set:map(mapping)
          unified:canonicalize()
          new_summary[privilege] = unified
        end
        new_access_summary[field_path] = new_summary
      end
      field_accesses[region_symbol] = new_access_summary
    end
    self.field_accesses = field_accesses
  end

  -- Synthesize partitions on non-trivial paths to disjoint children
  -- in reverse topological order, as they cannot be partitioned otherwise.
  -- We create equal partitions for the children and derive partitions
  -- for their ancestors.
  local paths = data.filter(function(path)
    return #path > 1 and
           data.any(unpack(path:map(function(range)
             return not created:has(range)
           end)))
  end,
  self.constraints:find_paths_to_disjoint_children(self.sources))

  paths:map(function(path)
    local parent = nil
    for idx = 1, #path do
      local info, range = unpack(path[idx])
      local partition = self.constraints:get_partition(range)
      assert(disjoint_partitions[range] == nil)
      created:insert(range)
      find_or_create(disjoint_partitions, range, hash_set.new):insert(range)
      if info == nil then
        if not created:has(range) then
          assert(parent == nil)
          stats:insert(create_equal_partition(range, partition.region, color_space_symbol,
              existing_disjoint_partitions))
        end
      else
        if info:is_image() then
          if not created:has(range) then
            assert(parent ~= nil)
            stats:insert(
              create_preimage_partition(range, partition.region, parent, info))
          end
        elseif info:is_subset() then
          local range, partition_stat =
            create_intersection_partition_region(partition.region, parent, range)
          stats:insert(partition_stat)
        else
          assert(false)
        end
      end
      parent = range
    end
  end)

  -- Find image ranges that are used for reductions but can be reindexed to
  -- a disjoint range. The reindexing may require sources of the original
  -- image ranges to be aliased when they need to be a union of multiple
  -- preimage ranges, and this cannot be done if those sources are being
  -- used for write accesses, which can be parallelized only with
  -- disjoint ranges or relaxed coherence that is yet to be implemented.
  -- Until relaxed coherence becomes available and then we can do a
  -- cost-based reasoning on multiple partitioning strategies, we do this
  -- reindexing only when source ranges are aliased.

  -- TODO: The previous step that synthesizes partitions for disjoint children
  --       can be merged with this step.
  local reduction_ranges = hash_set.new()
  for region_symbol, accesses_summary in self.field_accesses:items() do
    for field_path, summary in accesses_summary:items() do
      for privilege, range_set in summary:items() do
        if std.is_reduce(privilege) then
          range_set:foreach(function(range)
            if not created:has(range) then reduction_ranges:insert(range) end
          end)
        end
      end
    end
  end
  reduction_ranges:canonicalize()

  local reindexed = data.newmap()
  local replaced_constraints = terralib.newlist()
  for src_range, constraints in self.constraints.constraints:items() do
    local src_partition = self.constraints:get_partition(src_range)
    for info, dst_range in constraints:items() do
      local dst_partition = self.constraints:get_partition(dst_range)
      if reduction_ranges:has(dst_range) and info:is_image() and
         not src_partition.disjoint and
         -- TODO: Here we limit ourselves to indirect accesses with only one region
         --       (e.g., R[S[e]] += ...), even though reindexing is applicable to
         --       those with more than one region being involed.
         --       (e.g., R[S[T[U[e]]]] += ...)
         self.loop_ranges:has(src_range)
      then
        local new_dst_range = self.sources_by_regions[dst_partition.region]
        if new_dst_range == nil then
          local new_dst_range = new_range()
          -- We are going to use an equal partition, hence disjoint and complete.
          self.constraints:set_partition(new_dst_range,
              partition_info.new(dst_partition.region, true, true))
          self.sources_by_regions[dst_partition.region] = new_dst_range
        else
          local new_dst_partition = self.constraints:get_partition(new_dst_range)
          new_dst_partition:meet_disjointness(true)
          new_dst_partition:meet_completeness(true)
        end

        reindexed[dst_range] = new_dst_range
        replaced_constraints:insert({src_range, info, dst_range})
      end
    end
  end
  if #replaced_constraints > 0 then
    for _, tuple in ipairs(replaced_constraints) do
      local src_range, info, dst_range = unpack(tuple)
      local src_partition = self.constraints:get_partition(src_range)
      local new_dst_range = reindexed[dst_range]
      local dst_partition = self.constraints:get_partition(new_dst_range)
      local region_symbol, field_path = unpack(info.info)
      local preimage_range = self.constraints:find_or_create_preimage_constraint(
          new_dst_range, region_symbol, field_path)
      -- TODO: The Realm implementation of preimage operator doesn't preserve
      --       completeness, though there is an obvious way to make the result complete.
      --       We simply assume that preiamge partitions are complete,
      --       which implies that the region is a surjective function.
      self.constraints:set_partition(preimage_range,
        partition_info.new(src_partition.region, true, true))
      self.constraints:add_subset_constraint(preimage_range, dst_partition.region,
        src_range)
      self.sources:remove(src_range)
      if self.sources_by_regions[src_partition.region] == src_range then
        self.sources_by_regions[src_partition.region] = nil
      end
      assert(ghost_to_primary[dst_range] == nil or
             ghost_to_primary[dst_range] == new_dst_range)
      ghost_to_primary[dst_range] = new_dst_range
    end

    -- TODO: Need to refactor substitution for field accesses
    local function mapping(range) return reindexed[range] or range end
    local field_accesses = data.newmap()
    for region_symbol, accesses_summary in self.field_accesses:items() do
      local new_access_summary = data.newmap()
      for field_path, summary in accesses_summary:items() do
        local new_summary = data.newmap()
        for privilege, ranges_set in summary:items() do
          if std.is_reduce(privilege) then
            local unified = ranges_set:map(mapping)
            local unified_list = unified:to_list()
            if #unified_list == 1 and
               self.constraints:get_partition(unified_list[1]).disjoint
            then
              privilege = "reads_writes"
            else
              unified:canonicalize()
            end
            if privilege == "reads_writes" then
              find_or_create(new_summary, std.reads, hash_set.new):insert_all(unified)
              find_or_create(new_summary, std.writes, hash_set.new):insert_all(unified)
            else
              find_or_create(new_summary, privilege, hash_set.new):insert_all(unified)
            end
          else
            find_or_create(new_summary, privilege, hash_set.new):insert_all(ranges_set)
          end
        end
        new_access_summary[field_path] = new_summary
      end
      field_accesses[region_symbol] = new_access_summary
    end
    self.field_accesses = field_accesses
  end

  -- Synthesize a partition for each of the constraints using the following heuristics:
  --  * For each constraint R[P1] <= P2 where P2 is a partition of S,
  --    we emit an image partition `var P2 = image(S, P1, R)`. If P1 does not exist,
  --    we create an equal partition of R for P2.
  --  * For constraints P1 <= Q, P2 <= Q, ..., Pn <= Q, we emit a union partition
  --    `var Q = P1 | P2 | ... | Pn`. The synthesis fails (or can produce a code that is
  --    conditionally parallelizable) when Q needs to be disjoint.
  --  * For each constraint P1 <= P2 where P1 is a partition of R and P2 is of S,
  --    we emit `var P1 = R & P2`. If P2 does not exist, we create an equal partition of
  --    S for P2.
  --  * For each constraint (P1 + offset) % mod <= P2, we use a variant of image partition
  --    `var P2 = image(S, P1, \x.(x + offset) % mod). This is currently backed by
  --    a shim as Realm doesn't have support for this.

  -- Synthesize all preimage partitions
  local worklist = self.sources:to_list()
  local idx = 1
  local visited = hash_set.new()
  while idx <= #worklist do
    local src_range = worklist[idx]
    visited:insert(src_range)
    idx = idx + 1
    local constraints = self.constraints.constraints[src_range]
    if constraints ~= nil then
      for info, dst_range in constraints:items() do
        if info:is_preimage() then
          if not created:has(src_range) then
            local src_partition = self.constraints:get_partition(src_range)
            stats:insert(
              create_equal_partition(src_range, src_partition.region, color_space_symbol,
                existing_disjoint_partitions))
            created:insert(src_range)
          end
          if not created:has(dst_range) then
            local dst_partition = self.constraints:get_partition(dst_range)
            stats:insert(
              create_preimage_partition(dst_range, dst_partition.region, src_range, info))
            created:insert(dst_range)

            local key = data.newtuple(dst_partition.region, src_range, info.info)
            if preimage_partitions[key] == nil then preimage_partitions[key] = dst_range end
          end
        end
        if not visited:has(dst_range) then
          worklist:insert(dst_range)
        end
      end
    end
  end

  -- Build subset and supset relations from subset constraints.
  local subset_relation = data.newmap()
  local supset_relation = data.newmap()
  for src_range, constraints in self.constraints.constraints:items() do
    for info, dst_range in constraints:items() do
      if info:is_subset() then
        find_or_create(subset_relation, src_range, hash_set.new):insert(dst_range)
        find_or_create(supset_relation, dst_range, hash_set.new):insert(src_range)
      end
    end
  end

  -- Find subset constraints P <= Q where P is fixed and there is no other P' such that
  -- P' <= Q. We can simply put `var Q = P` for such P and Q.
  do
    local sources = hash_set.new()
    for src_range, _ in subset_relation:items() do
      if supset_relation[src_range] == nil and created:has(src_range) then
        sources:insert(src_range)
      end
    end
    sources:canonicalize()
    local worklist = sources:to_list()
    local idx = 1
    while idx <= #worklist do
      local src_range = worklist[idx]
      local src_partition = self.constraints:get_partition(src_range)
      idx = idx + 1
      if created:has(src_range) then
        local supsets = subset_relation[src_range]
        if supsets ~= nil then
          supsets:filter(function(range)
            return supset_relation[range] ~= nil and not created:has(range)
          end):foreach(function(dst_range)
            local subsets_list = supset_relation[dst_range]:to_list()
            if #subsets_list == 1 then
              local dst_partition = self.constraints:get_partition(dst_range)
              assert(not created:has(dst_range))
              if src_partition.region == dst_partition.region and
                 src_partition.complete
              then
                stats:insert(create_assignment(dst_range, src_range))
                dst_partition:meet_disjointness(src_partition.disjoint)
                dst_partition:meet_completeness(src_partition.complete)
                created:insert(dst_range)
              end
            end
            worklist:insert(dst_range)
          end)
        end
      end
    end

    local sources = hash_set.new()
    for src_range, _ in supset_relation:items() do
      if subset_relation[src_range] == nil then
        sources:insert(src_range)
      end
    end
    sources:canonicalize()
    local worklist = sources:to_list()
    local idx = 1
    while idx <= #worklist do
      local src_range = worklist[idx]
      idx = idx + 1
      local subsets = supset_relation[src_range]
      if subsets ~= nil then
        if not created:has(src_range) then
          local src_partition = self.constraints:get_partition(src_range)
          local created_subsets = subsets:filter(function(range)
            return created:has(range) end)
          if created_subsets:size() > 0 then
            local created_subsets_list = created_subsets:to_list()
            assert(data.any(unpack(created_subsets_list:map(function(range)
              local partition = self.constraints:get_partition(range)
              return partition.region == src_partition.region and partition.complete
            end))))
            local src_partition = self.constraints:get_partition(src_range)
            local lifted_subsets = created_subsets_list:map(function(range)
              local partition = self.constraints:get_partition(range)
              if partition.region == src_partition.region then
                return range
              else
                local new_range, partition_stat =
                  create_intersection_partition_region(src_partition.region, range)
                stats:insert(partition_stat)
                return new_range
              end
            end)
            local union_range, partition_stats =
              create_union_partitions(lifted_subsets, union_partitions)
            stats:insertall(partition_stats)
            stats:insert(create_assignment(src_range, union_range))
          else
            stats:insert(
              create_equal_partition(src_range, src_partition.region, color_space_symbol,
                existing_disjoint_partitions))
          end
          created:insert(src_range)
          find_or_create(disjoint_partitions, src_range, hash_set.new):insert(src_range)
        end
        subsets:foreach(function(dst_range)
          if not created:has(dst_range) then
            local dst_partition = self.constraints:get_partition(dst_range)
            local dst_range, partition_stat =
              create_intersection_partition_region(dst_partition.region, src_range, dst_range)
            stats:insert(partition_stat)
            created:insert(dst_range)
          end
          worklist:insert(dst_range)
        end)
      end
    end
  end

  -- Now we synthesize image partitions
  local worklist = self.sources:to_list()
  local idx = 1
  local visited = hash_set.new()
  while idx <= #worklist do
    local src_range = worklist[idx]
    visited:insert(src_range)
    idx = idx + 1
    local constraints = self.constraints.constraints[src_range]
    if constraints ~= nil then
      for info, dst_range in constraints:items() do
        if not created:has(src_range) then
          local src_partition = self.constraints:get_partition(src_range)
          stats:insert(
            create_equal_partition(src_range, src_partition.region, color_space_symbol,
              existing_disjoint_partitions))
          created:insert(src_range)
          find_or_create(disjoint_partitions, src_range, hash_set.new):insert(src_range)
        end

        if not created:has(dst_range) then
          assert(not info:is_subset())
          local dst_partition = self.constraints:get_partition(dst_range)
          local partition_stats =
            create_image_partition(dst_range, dst_partition.region, src_range, info)
          if terralib.islist(partition_stats) then
            stats:insertall(partition_stats)
          else
            stats:insert(partition_stats)
          end
          created:insert(dst_range)
          image_partitions[dst_range] = {info, src_range}
        end
        if not visited:has(dst_range) then
          worklist:insert(dst_range)
        end
      end
    end
  end

  -- Here, we find reductions on regions that are accessed with more than one range.
  -- For these regions, we will synthesize private-vs-shared partitions to minimize
  -- the size of reduction instances.
  --
  -- We can compute the biggest private part P' in the image f(P) as follows:
  --
  --   P' = f(P) - f(f^-1(f(P)) - P)
  --
  -- The partition P' is proven to be disjoint, and optimal in the sense that
  -- there is no bigger disjoint partition that is subregion-wise contained
  -- in f(P). Furthermore, P' and f(P) - P' are disjoint, which allows us
  -- to create a disjoint private-vs-shared partition. When there is no constraint
  -- that requires a complete partition for the region of P', we can use
  -- this private-vs-shared partition (R is the region of P').
  --
  --     R
  --     | *
  --     +--------------+
  --     |              |
  --   union(P')   union(f(P)-P')
  --
  -- When there is such a constraint, we need to compute a complement of f(P) because
  -- the function f may not be surjective. In this case, the region tree looks like
  -- this:
  --
  --     R
  --     | *
  --     +---------------------------------+
  --     |                                 |
  --   union(P') u (R-union(f(P)))    union(f(P)-P')
  --
  -- In any case, we can't use f(P)-P' directly for any write accesses, because it is
  -- not a disjoint partition. Therefore, we derive a disjoint shared partition and
  -- a ghost partition as follows:
  --
  --   fP_shared = equal(union(f(P) - P'))
  --   fP_ghost  = f(P) - P' - fP_shared
  --
  -- Finally, if the user provides a disjoint, complete partition P1 that subsumes f(P),
  -- we can use it to derive a private partition and a shared partition using this
  -- formulation:
  --
  --   P1_shared  = P1 & f(f^-1(P1) - P)
  --   P1_private = P1 - P1_shared
  --
  -- All key properties (P1_shared's disjointness, P1_private's disjointness, and
  -- disjointness between P1_shared and P1_private) are trivially and P1_private enjoys
  -- the optimality.

  -- TODO: Ideally, we do private-vs-shared partitioning only for fields with reductions,
  --       but we create a global private-vs-shared for the entire region as
  --       sets of fields that are concurrently accesed are not yet tracked.

  local ranges_need_pvs = data.newmap()
  if std.config["parallelize-use-pvs"] then
    for region_symbol, accesses_summary in self.field_accesses:items() do
      local all_region_ranges = hash_set.new()
      for field_path, summary in accesses_summary:items() do
        local all_field_ranges = hash_set.new()
        local has_reduce = false
        for privilege, range_set in summary:items() do
          all_field_ranges:insert_all(range_set)
          has_reduce = has_reduce or std.is_reduce(privilege)
        end
        local all_field_ranges_list = all_field_ranges:to_list()
        -- If the reduction is made with a single range constrained by some constraint,
        -- we still need a private-vs-shared partitioning.
        if has_reduce and
           (#all_field_ranges_list > 1 or
            image_partitions[all_field_ranges_list[1]] ~= nil)
        then
          all_region_ranges:insert_all(all_field_ranges)
        end
      end
      if not all_region_ranges:is_empty() then
        ranges_need_pvs[region_symbol] = all_region_ranges
      end
    end
    --for region_symbol, accesses_summary in self.field_accesses:items() do
    --  for field_path, summary in accesses_summary:items() do
    --    local has_any = false
    --    for privilege, range_set in summary:items() do
    --      range_set:foreach(function(range)
    --        has_any = has_any or
    --          (ranges_need_pvs[region_symbol] and
    --           ranges_need_pvs[region_symbol]:has(range))
    --      end)
    --    end
    --    if has_any then
    --      for privilege, range_set in summary:items() do
    --        ranges_need_pvs[region_symbol]:insert_all(range_set)
    --      end
    --    end
    --  end
    --end
    --for region_symbol, ranges in ranges_need_pvs:items() do
    --  print(region_symbol, tostring(ranges))
    --end
  end

  for region_symbol, ranges in ranges_need_pvs:items() do
    local disjoint_range = nil
    local image_ranges = hash_set.new()
    ranges:foreach(function(range)
      local partition = self.constraints:get_partition(range)
      if partition.disjoint then
        -- TODO: Handle when multiple disjoint ranges exist
        assert(disjoint_range == nil)
        disjoint_range = range
      else
        image_ranges:insert(range)
      end
    end)
    -- TODO: Handle when no disjoint range exists
    assert(disjoint_range ~= nil)

    if not std.config["parallelize-tight-pvs"] then
      if not created:has(disjoint_range) then
        local partition = self.constraints:get_partition(disjoint_range)
        stats:insert(
          create_equal_partition(disjoint_range, partition.region, color_space_symbol,
            existing_disjoint_partitions))
        created:insert(disjoint_range)
      end
    end

    -- Now we gather all shared parts
    local shared_ranges = image_ranges:map(function(image_range)
      local prev_preimage_range = nil
      if std.config["parallelize-tight-pvs"] then
        prev_preimage_range = image_range
      else
        prev_preimage_range = disjoint_range
      end
      local preimage_range = nil
      local dst_range = image_range
      local image_infos = terralib.newlist()
      while image_partitions[dst_range] ~= nil do
        local info, src_range = unpack(image_partitions[dst_range])
        local src_partition = self.constraints:get_partition(src_range)
        preimage_range = new_range()
        stats:insert(
          create_preimage_partition(preimage_range, src_partition.region, prev_preimage_range, info))
        image_infos:insert({info, dst_range})
        dst_range = src_range
        prev_preimage_range = preimage_range
      end
      assert(#image_infos > 0 and preimage_range ~= nil)

      local diff_range, diff_partition_stat =
        create_difference_partition(preimage_range, dst_range)
      stats:insert(diff_partition_stat)
      diff_partitions[data.newtuple(preimage_range, dst_range)] = diff_range

      local prev_shared_range = diff_range
      local shared_range = nil
      for idx = #image_infos, 1, -1 do
        local info, dst_range = unpack(image_infos[idx])
        local dst_partition = self.constraints:get_partition(dst_range)
        shared_range = new_range()
        local partition_stats =
          create_image_partition(shared_range, dst_partition.region, prev_shared_range, info)
        if terralib.islist(partition_stats) then
          stats:insertall(partition_stats)
        else
          stats:insert(partition_stats)
        end
        prev_shared_range = shared_range
      end

      assert(std.type_eq(image_range:gettype():parent_region(),
                         shared_range:gettype():parent_region()))
      return shared_range
    end)
    local all_image_range, union_partition_stats =
      create_union_partitions(image_ranges, union_partitions)
    stats:insertall(union_partition_stats)
    local all_shared_range, union_partition_stats =
      create_union_partitions(shared_ranges, union_partitions)
    stats:insertall(union_partition_stats)

    local private_range = nil
    local shared_range = nil
    local ghost_range = nil
    if std.config["parallelize-tight-pvs"] then
      local private_image_range, diff_partition_stat =
        create_difference_partition(all_image_range, all_shared_range)
      local complement_range, partition_stats = create_complement_partition(all_image_range)
      stats:insertall(partition_stats)

      local range, partition_stat =
        create_disjoint_union_partition(private_image_range, complement_range)
      stats:insert(partition_stat)
      private_range = range

      local range, partition_stats = create_complement_partition(private_range)
      stats:insertall(partition_stats)
      shared_range = range

      local range, diff_partition_stat =
        create_difference_partition(all_shared_range, shared_range)
      ghost_range = range

    else
      local range, diff_partition_stat =
        create_difference_partition(disjoint_range, all_shared_range)
      private_range = range
      stats:insert(diff_partition_stat)
      diff_partitions[data.newtuple(disjoint_range, all_shared_range)] = private_range

      local range, diff_partition_stat =
        create_difference_partition(disjoint_range, private_range)
      shared_range = range
      stats:insert(diff_partition_stat)
      diff_partitions[data.newtuple(disjoint_range, private_range)] = shared_range

      local range, diff_partition_stat =
        create_difference_partition(all_image_range, disjoint_range)
      ghost_range = range
      stats:insert(diff_partition_stat)
    end

    local pvs_partition, partition_stats = create_pvs_partition(private_range, shared_range)
    stats:insertall(partition_stats)

    local private_subregion, shared_subregion, subregion_stats =
      retrieve_private_shared_subregions(self, pvs_partition)
    stats:insertall(subregion_stats)
    local private_range, partition_stat =
      create_intersection_partition_region(private_subregion, private_range)
    stats:insert(partition_stat)
    local shared_range, partition_stat =
      create_intersection_partition_region(shared_subregion, shared_range)
    stats:insert(partition_stat)
    local ghost_range, partition_stat =
      create_intersection_partition_region(shared_subregion, ghost_range)
    stats:insert(partition_stat)

    local disjoint_ranges =
      find_or_create(disjoint_partitions, disjoint_range, hash_set.new)
    disjoint_ranges:insert(private_range)
    disjoint_ranges:insert(shared_range)

    diff_partitions[data.newtuple(all_image_range, disjoint_range)] = ghost_range
  end

  self.sources:foreach(function(source)
    if not created:has(source) then
      local partition = self.constraints:get_partition(source)
      stats:insert(
        create_equal_partition(source, partition.region, color_space_symbol,
          existing_disjoint_partitions))
      created:insert(source)
    end
  end)

  -- Finally, we create union partitions for accesses with multiple ranges
  for region_symbol, accesses_summary in self.field_accesses:items() do
    for field_path, summary in accesses_summary:items() do
      local all_ranges = hash_set.new()
      local has_reduce = false

      -- TODO: We can remove any range R from all_ranges if it is subsumed by
      --       another range R' (i.e., R <=^+ R')
      for privilege, range_set in summary:items() do
        all_ranges:insert_all(range_set)
        has_reduce = has_reduce or std.is_reduce(privilege)
      end

      local mapping = nil

      if all_ranges:size() == 1 then
        local range = all_ranges:to_list()[1]
        mapping = data.newmap()

        assert(created:has(range))
        if disjoint_partitions[range] ~= nil then
          mapping[range] = disjoint_partitions[range]:to_list()
        else
          local singleton = hash_set.new()
          singleton:insert(range)
          disjoint_partitions[range] = singleton
          mapping[range] = terralib.newlist({range})
        end

        local partition_type = range:gettype()
        if partition_type:is_disjoint() and self.loop_ranges:has(range) then
          self.loop_range_partitions[range] = mapping[range]
        end

      else
        all_ranges:canonicalize()

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

        mapping = mappings_by_range_sets[all_ranges]
        local function all_affine_images(ranges)
          return data.all(unpack(secondary_ranges:map(function(range)
            local pair = image_partitions[range]
            if pair then
              local info, _ = unpack(pair)
              return info:is_affine()
            end
            return false
          end)))
        end
        if mapping == nil then
          mapping = data.newmap()
          if primary_range == nil then
            assert(not has_reduce)
            local all_affine = all_affine_images(secondary_ranges)
            if all_affine then
              all_ranges:foreach(function(range) mapping[range] = range end)
            else
              local union_range, union_partition_stats =
                create_union_partitions(secondary_ranges, union_partitions)
              stats:insertall(union_partition_stats)
              local union_range = terralib.newlist({union_range})
              all_ranges:foreach(function(range) mapping[range] = union_range end)
            end
          else
            local primary_partitions = nil
            if disjoint_partitions[primary_range] ~= nil then
              primary_partitions = disjoint_partitions[primary_range]:to_list()
            else
              primary_partitions = terralib.newlist({primary_range})
            end

            mapping[primary_range] = primary_partitions
            if self.loop_ranges:has(primary_range) then
              self.loop_range_partitions[primary_range] = primary_partitions
            end

            local all_affine = all_affine_images(secondary_ranges)
            if all_affine then
              secondary_ranges:map(function(range)
                local ghost_range = diff_partitions[data.newtuple(range, primary_range)]
                if ghost_range == nil then
                  local diff_range, diff_partition_stat =
                    create_difference_partition(range, primary_range)
                  stats:insert(diff_partition_stat)
                  diff_partitions[data.newtuple(range, primary_range)] = diff_range
                  ghost_range = diff_range
                end
                local ghost_ranges = terralib.newlist()
                ghost_ranges:insertall(primary_partitions)
                ghost_ranges:insert(ghost_range)
                mapping[range] = ghost_ranges
              end)
            else
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

              local ghost_ranges = terralib.newlist()
              ghost_ranges:insertall(primary_partitions)
              ghost_ranges:insert(ghost_range)
              secondary_ranges:map(function(range) mapping[range] = ghost_ranges end)
            end

            secondary_ranges:map(function(range)
              assert(ghost_to_primary[range] == nil or
                     ghost_to_primary[range] == primary_range)
              ghost_to_primary[range] = primary_range
            end)
          end
          mappings_by_range_sets[all_ranges] = mapping
        end
      end

      assert(mapping ~= nil)
      local key = data.newtuple(region_symbol, field_path)
      self.mappings_by_access_paths[key] = mapping
    end
  end

  self.loop_ranges:foreach(function(range)
    if self.loop_range_partitions[range] == nil then
      local partition = self.constraints:get_partition(range)
      assert(range:hastype() and partition.complete)
      self.loop_range_partitions[range] = terralib.newlist({range})
    end
  end)

  if std.config["parallelize-cache-incl-check"] then
    for ghost_range, primary_range in ghost_to_primary:items() do
      local primary_partitions = disjoint_partitions[primary_range]:to_list() or
                                 terralib.newlist({primary_range})
      local num_cases = #primary_partitions + 1

      local loop_range = ghost_range
      local paths = terralib.newlist()
      while image_partitions[loop_range] do
        local info, src_range = unpack(image_partitions[loop_range])
        paths:insert({info, src_range})
        loop_range = src_range
      end
      -- TODO: Handle cases when one ghost range corresponds to a non-source loop range
      assert(self.loop_ranges:has(loop_range))

      -- partitions in loop_range_partitions are just a decomposition of the original loop range
      assert(loop_range:hastype())
      local loop_range_type = loop_range:gettype()
      local cache_value_type = uint8
      if not loop_range_type:is_disjoint() then
        -- TODO: We support only the cases when all reductions are reindexed
        assert(num_cases == 2 and #primary_partitions == 1)
        cache_value_type = loop_range_type:colors().index_type
      end
      local orig_region = loop_range_type.parent_region_symbol
      local cache_region, cache_stats =
        create_isomorphic_region("cache_" .. tostring(ghost_range), orig_region, cache_value_type)
      stats:insertall(cache_stats)
      self.task_constraints.region_universe[cache_region:gettype()] = true
      std.add_privilege(self.task_constraints, std.reads, cache_region:gettype(), data.newtuple())
      std.add_privilege(self.task_constraints, std.writes, cache_region:gettype(), data.newtuple())

      -- We mark all the entries as potentially pointing to ghost elements
      local cache_partition, partition_stat =
        create_intersection_partition_region(cache_region, loop_range)
      stats:insert(partition_stat)
      if loop_range_type:is_disjoint() then
        local fill_loop = create_index_fill(self.task_constraints, cache_partition,
            color_space_symbol, cache_value_type, num_cases)
        stats:insert(fill_loop)
      end

      local cases = data.newmap()
      for case_id, primary_partition in ipairs(primary_partitions) do
        local preimage_partition = primary_partition
        for i = 1, #paths do
          local info, range = unpack(paths[i])
          local region = self.constraints:get_partition(range).region
          local next_preimage_partition =
            preimage_partitions[data.newtuple(region, preimage_partition, info.info)]
          if next_preimage_partition == nil then
            next_preimage_partition = new_range()
            stats:insert(
              create_preimage_partition(next_preimage_partition, region, preimage_partition, info))
          end
          preimage_partition = next_preimage_partition
        end
        local cache_subpartition, partition_stat =
          create_intersection_partition(cache_partition, preimage_partition)
        stats:insert(partition_stat)
        local fill_loop = create_index_fill(self.task_constraints, cache_subpartition,
            color_space_symbol, cache_value_type, loop_range_type:is_disjoint() and case_id)
        stats:insert(fill_loop)
        cases[primary_partition] = case_id
      end

      -- We can use the same cache for any partition of a subset of the loop range
      local loop_range_partitions = terralib.newlist()
      loop_range_partitions:insertall(self.loop_range_partitions[loop_range])
      -- TODO: We need to traverse down the chain of subset constraints
      for src_range, constraints in self.constraints.constraints:items() do
        for info, dst_range in constraints:items() do
          if info:is_subset() and dst_range == loop_range and
             self.loop_ranges:has(src_range)
          then
            loop_range_partitions:insertall(self.loop_range_partitions[src_range])
          end
        end
      end
      assert(#loop_range_partitions > 0)
      for _, loop_range_partition in ipairs(loop_range_partitions) do
        local all_caches = find_or_create(self.incl_check_caches, ghost_range)
        assert(all_caches[loop_range_partition] == nil)
        all_caches[loop_range_partition] =
          { cache_partition, cases, num_cases, loop_range_type:is_disjoint() }
      end
    end
  end

  if std.config["parallelize-time-deppart"] then
    ts_end = std.newsymbol(uint64, "__ts_end")
    stats:insert(ast.typed.stat.Fence {
      kind = ast.fence_kind.Execution {},
      blocking = true,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Var {
      symbol = ts_end,
      type = uint64,
      value = ast.typed.expr.Call {
        fn = ast.typed.expr.Function {
          value = c.legion_get_current_time_in_micros,
          expr_type = c.legion_get_current_time_in_micros:gettype(),
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = uint64,
        args = terralib.newlist(),
        conditions = terralib.newlist(),
        replicable = true,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
    stats:insert(ast.typed.stat.Expr {
      expr = ast.typed.expr.Call {
        fn = ast.typed.expr.Function {
          value = c.printf,
          expr_type = ({rawstring,uint64}->int32).type,
          span = ast.trivial_span(),
          annotations = ast.default_annotations(),
        },
        expr_type = int32,
        args = terralib.newlist({
          ast.typed.expr.Constant {
            value = "partitioning took %llu us\n",
            expr_type = rawstring,
            span = ast.trivial_span(),
            annotations = ast.default_annotations(),
          },
          ast.typed.expr.Binary {
            op = "-",
            lhs = ast.typed.expr.ID {
              value = ts_end,
              expr_type = std.rawref(&ts_end:gettype()),
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            },
            rhs = ast.typed.expr.ID {
              value = ts_start,
              expr_type = std.rawref(&ts_start:gettype()),
              span = ast.trivial_span(),
              annotations = ast.default_annotations(),
            },
            expr_type = uint64,
            span = ast.trivial_span(),
            annotations = ast.default_annotations(),
          }
        }),
        conditions = terralib.newlist(),
        replicable = true,
        span = ast.trivial_span(),
        annotations = ast.default_annotations(),
      },
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    })
  end

  return stats, unifiable_ranges, reindexed
end

function solver_context:print_all_constraints()
  print("################")
  print("* sources:")
  self.sources:foreach(function(source)
    print("    " .. tostring(source))
  end)
  print("* sources by regions:")
  for region, source in self.sources_by_regions:items() do
    print("    " .. tostring(region) .. " -> " .. tostring(source))
  end
  print("* loop ranges:")
  self.loop_ranges:foreach(function(range)
    print("    " .. tostring(range))
  end)
  self.constraints:print_constraints()
  print("* accesses:")
  for region_symbol, accesses in self.field_accesses:items() do
    for field_path, summary in accesses:items() do
      print("    " .. (data.newtuple(region_symbol) .. field_path):mkstring("."))
      for privilege, ranges_set in summary:items() do
        print("        " .. (data.newtuple(region_symbol) .. field_path):mkstring(".") ..
            "[" .. tostring(ranges_set) .. "] @ " .. tostring(privilege))
      end
    end
  end
  print("################")
end

local solve_constraints = {}

function solve_constraints.solve(cx, stat)
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
  local existing_disjoint_partitions = data.newmap()
  local user_constraints = partitioning_constraints.new()
  for idx = 1, #stat.hints do
    local hint = stat.hints[idx]
    if hint:is(ast.typed.expr.ID) then
      local expr_type = std.as_read(hint.expr_type)
      if std.is_ispace(expr_type) then
        color_space_symbol = hint.value
      elseif std.is_partition(expr_type) and expr_type:is_disjoint() then
        existing_disjoint_partitions[expr_type.parent_region_symbol] = hint.value
      end
    elseif hint:is(ast.typed.expr.ParallelizerConstraint) then
      if hint.op == "<=" and hint.lhs:is(ast.typed.expr.Image) and
         hint.rhs:is(ast.typed.expr.ID)
      then
        local src_range = hint.lhs.partition.value
        local mapping_region = hint.lhs.region.region.value
        local field_path = hint.lhs.region.fields[1]
        local dst_range = hint.rhs.value
        user_constraints:add_image_constraint(src_range, mapping_region, field_path, dst_range)

        local function add_partition_info(range)
          local partition = partition_info.new(
              range:gettype().parent_region_symbol,
              range:gettype():is_disjoint(),
              range:gettype():is_disjoint())
          if user_constraints:get_partition(range) == nil then
            user_constraints:set_partition(range, partition)
          end
          if partition.disjoint then
            existing_disjoint_partitions[partition.region] = range
          end
        end
        add_partition_info(src_range)
        add_partition_info(dst_range)
      end
    end
  end
  assert(color_space_symbol ~= nil)

  local solver_cx = solver_context.new(cx)
  local mappings = all_tasks:map(function(task)
    local constraints = task:get_partitioning_constraints()
    local region_mapping = collector_cx:get_mapping(task)
    local range_mapping = solver_cx:unify(task.name, constraints, region_mapping)
    return {range_mapping, region_mapping}
  end)

  local user_mapping = data.newmap()
  if not user_constraints:is_empty() then
    local user_sources = hash_set.new()
    for range, _ in user_constraints.ranges:items() do
      user_sources:insert(range)
    end
    for src_range, constraints in user_constraints.constraints:items() do
      for info, dst_range in constraints:items() do
        if src_range ~= dst_range then
          user_sources:remove(dst_range)
        end
      end
    end
    user_sources:foreach(function(source)
      local my_source =
        solver_cx.sources_by_regions[source:gettype().parent_region_symbol]
      local unifiable, mapping =
        find_unifiable_ranges(solver_cx.constraints, user_constraints, my_source, source)
      if unifiable then
        local unified = user_constraints:join(solver_cx.constraints, mapping)
        for from, to in unified:items() do
          mapping[from] = to
        end
        solver_cx.constraints = user_constraints

        local new_sources = solver_cx.sources:map(function(range)
          return mapping[range] or range
        end)
        solver_cx.sources = new_sources

        local new_sources_by_regions = data.newmap()
        for region, source in solver_cx.sources_by_regions:items() do
          new_sources_by_regions[region] = mapping[source] or source
        end
        solver_cx.sources_by_regions = new_sources_by_regions

        local new_loop_ranges = solver_cx.loop_ranges:map(function(range)
          return mapping[range] or range
        end)
        solver_cx.loop_ranges = new_loop_ranges

        local function mapping_fn(range) return mapping[range] or range end
        local field_accesses = data.newmap()
        for region_symbol, accesses_summary in solver_cx.field_accesses:items() do
          local new_access_summary = data.newmap()
          for field_path, summary in accesses_summary:items() do
            local new_summary = data.newmap()
            for privilege, ranges_set in summary:items() do
              local unified = ranges_set:map(mapping_fn)
              unified:canonicalize()
              new_summary[privilege] = unified
            end
            new_access_summary[field_path] = new_summary
          end
          field_accesses[region_symbol] = new_access_summary
        end
        solver_cx.field_accesses = field_accesses

        for from, to in mapping:items() do
          user_mapping[from] = to
        end
      end
    end)
  end

  local partition_stats, unified_ranges, reindexed_ranges =
    solver_cx:synthesize_partitions(existing_disjoint_partitions, color_space_symbol)

  if not (unified_ranges:is_empty() and user_mapping:is_empty()) then
    mappings = mappings:map(function(pair)
      local range_mapping, region_mapping = unpack(pair)
      return {
        combine_mapping(unified_ranges, combine_mapping(user_mapping, range_mapping)),
        region_mapping
      }
    end)
  end

  local all_mappings = {}
  for idx = 1, #all_tasks do
    all_mappings[all_tasks[idx]] = mappings[idx]
  end

  local transitively_closed = data.newmap()
  for src_range, constraints in user_constraints.constraints:items() do
    for info, dst_range in constraints:items() do
      if src_range == dst_range then
        assert(info:is_image())
        local _, field_path = unpack(info.info)
        find_or_create(transitively_closed, src_range,
            hash_set.new):insert(field_path)
      end
    end
  end

  return {
    all_tasks                = all_tasks,
    all_mappings             = all_mappings,
    mappings_by_access_paths = solver_cx.mappings_by_access_paths,
    loop_range_partitions    = solver_cx.loop_range_partitions,
    incl_check_caches        = solver_cx.incl_check_caches,
    color_space_symbol       = color_space_symbol,
    partition_stats          = partition_stats,
    reindexed_ranges         = reindexed_ranges,
    transitively_closed      = transitively_closed,
  }
end

return solve_constraints
