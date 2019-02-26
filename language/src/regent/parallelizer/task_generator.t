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
local ast_util = require("regent/ast_util")
local data = require("common/data")
local passes = require("regent/passes")
local std = require("regent/std")

local hash_set = require("regent/parallelizer/hash_set")

local function unreachable(cx, node) assert(false) end

local function find_or_create(map, key, init)
  local init = init or data.newmap
  local value = map[key]
  if value == nil then
    value = init()
    map[key] = value
  end
  return value
end

local rewriter_context = {}

rewriter_context.__index = rewriter_context

function rewriter_context.new(accesses_to_region_params,
                              loop_var_to_regions,
                              param_mapping,
                              demand_cuda)
  local cx = {
    accesses_to_region_params = accesses_to_region_params,
    loop_var_to_regions       = loop_var_to_regions,
    symbol_mapping            = param_mapping:copy(),
    demand_cuda               = demand_cuda,
  }
  return setmetatable(cx, rewriter_context)
end

function rewriter_context:update_mapping(src_symbol, tgt_symbol)
  self.symbol_mapping[src_symbol] = tgt_symbol
end

function rewriter_context:get_mapping(src_symbol)
  return self.symbol_mapping[src_symbol]
end

function rewriter_context:rewrite_symbol(symbol)
  return self.symbol_mapping[symbol] or symbol
end

function rewriter_context:rewrite_type(type)
  return std.type_sub(type, self.symbol_mapping)
end

local rewrite_accesses = {}

function rewrite_accesses.pass_through_expr(cx, expr) return expr end

function rewrite_accesses.expr_id(cx, expr)
  return expr {
    value = cx:rewrite_symbol(expr.value),
    expr_type = cx:rewrite_type(expr.expr_type),
  }
end

function rewrite_accesses.expr_regent_cast(cx, expr)
  return expr {
    value = rewrite_accesses.expr(cx, expr.value),
    expr_type = cx:rewrite_type(expr.expr_type),
  }
end

function rewrite_accesses.expr_ctor(cx, expr)
  return expr {
    fields = expr.fields:map(function(field)
      return field {
        value = rewrite_accesses.expr(cx, field.value),
      }
    end),
  }
end

function rewrite_accesses.expr_unary(cx, expr)
  return expr {
    rhs = rewrite_accesses.expr(cx, expr.rhs),
  }
end

function rewrite_accesses.expr_binary(cx, expr)
  return expr {
    lhs = rewrite_accesses.expr(cx, expr.lhs),
    rhs = rewrite_accesses.expr(cx, expr.rhs),
  }
end

function rewrite_accesses.expr_cast(cx, expr)
  return expr {
    arg = rewrite_accesses.expr(cx, expr.arg),
  }
end

function rewrite_accesses.expr_call(cx, expr)
  return expr {
    args = expr.args:map(function(arg)
      return rewrite_accesses.expr(cx, arg)
    end),
  }
end

function rewrite_accesses.expr_method_call(cx, expr)
  return expr {
    value = rewrite_accesses.expr(cx, expr.value),
    args = expr.args:map(function(arg)
      return rewrite_accesses.expr(cx, arg)
    end),
  }
end

function rewrite_accesses.expr_field_access(cx, expr)
  return expr {
    value = rewrite_accesses.expr(cx, expr.value),
  }
end

function rewrite_accesses.expr_index_access(cx, expr)
  return expr {
    index = rewrite_accesses.expr(cx, expr.index),
    value = rewrite_accesses.expr(cx, expr.value),
  }
end

local rewrite_accesses_expr_table = {
  [ast.typed.expr.ID]                         = rewrite_accesses.expr_id,
  [ast.typed.expr.DynamicCast]                = rewrite_accesses.expr_regent_cast,
  [ast.typed.expr.StaticCast]                 = rewrite_accesses.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]                 = rewrite_accesses.expr_regent_cast,
  [ast.typed.expr.Ctor]                       = rewrite_accesses.expr_ctor,
  [ast.typed.expr.Unary]                      = rewrite_accesses.expr_unary,
  [ast.typed.expr.Binary]                     = rewrite_accesses.expr_binary,
  [ast.typed.expr.Cast]                       = rewrite_accesses.expr_cast,

  [ast.typed.expr.Call]                       = rewrite_accesses.expr_call,
  [ast.typed.expr.MethodCall]                 = rewrite_accesses.expr_method_call,

  [ast.typed.expr.RawFields]                  = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.RawPhysical]                = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.RawRuntime]                 = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.RawValue]                   = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.ListInvert]                 = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.ListRange]                  = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.ListIspace]                 = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.ListFromElement]            = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.RegionRoot]                 = rewrite_accesses.pass_through_expr,

  [ast.typed.expr.Function]                   = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.Constant]                   = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.Null]                       = rewrite_accesses.pass_through_expr,
  [ast.typed.expr.Isnull]                     = rewrite_accesses.pass_through_expr,

  [ast.typed.expr.FieldAccess]                = rewrite_accesses.expr_field_access,
  [ast.typed.expr.IndexAccess]                = rewrite_accesses.expr_index_access,
  [ast.typed.expr.Deref]                      = unreachable,

  [ast.typed.expr.CtorListField]              = unreachable,
  [ast.typed.expr.CtorRecField]               = unreachable,
  [ast.typed.expr.Internal]                   = unreachable,
  [ast.typed.expr.RawContext]                 = unreachable,
  [ast.typed.expr.Ispace]                     = unreachable,
  [ast.typed.expr.Region]                     = unreachable,
  [ast.typed.expr.Partition]                  = unreachable,
  [ast.typed.expr.PartitionEqual]             = unreachable,
  [ast.typed.expr.PartitionByField]           = unreachable,
  [ast.typed.expr.PartitionByRestriction]     = unreachable,
  [ast.typed.expr.Image]                      = unreachable,
  [ast.typed.expr.ImageByTask]                = unreachable,
  [ast.typed.expr.Preimage]                   = unreachable,
  [ast.typed.expr.CrossProduct]               = unreachable,
  [ast.typed.expr.CrossProductArray]          = unreachable,
  [ast.typed.expr.ListSlicePartition]         = unreachable,
  [ast.typed.expr.ListDuplicatePartition]     = unreachable,
  [ast.typed.expr.ListSliceCrossProduct]      = unreachable,
  [ast.typed.expr.ListCrossProduct]           = unreachable,
  [ast.typed.expr.ListCrossProductComplete]   = unreachable,
  [ast.typed.expr.ListPhaseBarriers]          = unreachable,
  [ast.typed.expr.PhaseBarrier]               = unreachable,
  [ast.typed.expr.DynamicCollective]          = unreachable,
  [ast.typed.expr.DynamicCollectiveGetResult] = unreachable,
  [ast.typed.expr.Advance]                    = unreachable,
  [ast.typed.expr.Adjust]                     = unreachable,
  [ast.typed.expr.Arrive]                     = unreachable,
  [ast.typed.expr.Await]                      = unreachable,
  [ast.typed.expr.Copy]                       = unreachable,
  [ast.typed.expr.Fill]                       = unreachable,
  [ast.typed.expr.Acquire]                    = unreachable,
  [ast.typed.expr.Release]                    = unreachable,
  [ast.typed.expr.AttachHDF5]                 = unreachable,
  [ast.typed.expr.DetachHDF5]                 = unreachable,
  [ast.typed.expr.AllocateScratchFields]      = unreachable,
  [ast.typed.expr.WithScratchFields]          = unreachable,
  [ast.typed.expr.Condition]                  = unreachable,
  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.ParallelizerConstraint]     = unreachable,
  [ast.typed.expr.ImportIspace]               = unreachable,
  [ast.typed.expr.ImportRegion]               = unreachable,
  [ast.typed.expr.ImportPartition]            = unreachable,
}

local rewrite_accesses_expr = ast.make_single_dispatch(
  rewrite_accesses_expr_table,
  {ast.typed.expr})

function rewrite_accesses.expr(cx, expr)
  return rewrite_accesses_expr(cx)(expr)
end

function rewrite_accesses.stat_for_list(cx, stat)
  local symbol = stat.symbol
  local symbol_type = symbol:gettype()
  local value = stat.value
  local value_type = std.as_read(stat.value.expr_type)
  assert(not std.is_ispace(value_type))

  if std.is_region(value_type) then
    assert(value:is(ast.typed.expr.ID))
    assert(std.is_bounded_type(symbol_type))
    local region_symbol = value.value
    local region_params = cx.loop_var_to_regions[symbol]:to_list()
    -- TODO: Need to split the loop into multiple loops if the original region
    --       is mapped to multiple regions
    assert(#region_params == 1)
    local region_param = region_params[1]

    cx:update_mapping(region_symbol, region_param)
    value = ast.typed.expr.ID {
      value = region_param,
      expr_type = std.rawref(&region_param:gettype()),
      span = value.span,
      annotations = value.annotations,
    }
    local new_symbol = std.newsymbol(cx:rewrite_type(symbol_type), symbol:getname())
    cx:update_mapping(symbol, new_symbol)
    symbol = new_symbol
  end

  return stat {
    symbol = symbol,
    value = value,
    block = rewrite_accesses.block(cx, stat.block),
    metadata = false,
  }
end

function rewrite_accesses.stat_if(cx, stat)
  local cond = rewrite_accesses.expr(cx, stat.cond)
  local then_block = rewrite_accesses.block(cx, stat.then_block)
  local else_block = rewrite_accesses.block(cx, stat.else_block)
  return stat {
    cond = cond,
    then_block = then_block,
    else_block = else_block,
  }
end

function rewrite_accesses.stat_while(cx, stat)
  local cond = rewrite_accesses.expr(cx, stat.cond)
  local block = rewrite_accesses.block(cx, stat.block)
  return stat {
    cond = cond,
    block = block,
  }
end

function rewrite_accesses.stat_for_num(cx, stat)
  local values = stat.values:map(function(value)
    return rewrite_accesses.expr(cx, value)
  end)
  local block = rewrite_accesses.block(cx, stat.block)
  return stat {
    values = values,
    block = block,
  }
end

function rewrite_accesses.stat_repeat(cx, stat)
  local until_cond = rewrite_accesses.expr(cx, stat.until_cond)
  local block = rewrite_accesses.block(cx, stat.block)
  return stat {
    until_cond = until_cond,
    block = block,
  }
end

function rewrite_accesses.stat_block(cx, stat)
  return stat { block = rewrite_accesses.block(cx, stat.block) }
end

local function find_index(expr)
  if expr:is(ast.typed.expr.FieldAccess) then
    return find_index(expr.value)
  elseif expr:is(ast.typed.expr.Deref) then
    return expr.value
  elseif expr:is(ast.typed.expr.IndexAccess) then
    if std.is_ref(expr.expr_type) then
      return expr.index
    else
      return find_index(expr.value)
    end
  else
    assert(false)
  end
end

local function rewrite_region_access(cx, local_mapping, expr)
  local expr_type =
    cx:rewrite_type(std.type_sub(expr.expr_type, local_mapping))
  if expr:is(ast.typed.expr.FieldAccess) then
    return expr {
      value = rewrite_region_access(cx, local_mapping, expr.value),
      expr_type = expr_type,
    }

  elseif expr:is(ast.typed.expr.Deref) then
    assert(std.is_ref(expr.expr_type))
    assert(#expr.expr_type.bounds_symbols == 1)
    local region_symbol = local_mapping[expr.expr_type.bounds_symbols[1]]
    assert(region_symbol ~= nil)
    local index = rewrite_accesses.expr(cx, expr.value)
    return ast.typed.expr.IndexAccess {
      value = ast.typed.expr.ID {
        value = region_symbol,
        expr_type = std.rawref(&region_symbol:gettype()),
        span = expr.span,
        annotations = ast.default_annotations(),
      },
      index = index,
      expr_type = expr_type,
      span = expr.span,
      annotations = ast.default_annotations(),
    }

  elseif expr:is(ast.typed.expr.IndexAccess) then
    local index = rewrite_accesses.expr(cx, expr.index)
    local value = expr.value
    if std.is_ref(expr_type) then
      assert(value:is(ast.typed.expr.ID))
      assert(local_mapping[value.value] ~= nil)
      local region_symbol = local_mapping[value.value]
      value = value {
        value = region_symbol,
        expr_type = std.rawref(&region_symbol:gettype()),
      }
    else
      value = rewrite_region_access(cx, local_mapping, value)
    end
    return expr {
      value = value,
      index = index,
      expr_type = expr_type,
    }

  else
    assert(false)
  end
end

local has_region_access_checks = {
  [ast.typed.expr.IndexAccess] = function(expr, fn)
    return fn(expr.value, fn)
  end,
  [ast.typed.expr.Deref] = function(expr, fn)
    -- Must be unreachable
    assert(false)
    return fn(expr.value, fn)
  end,
  [ast.typed.expr.FieldAccess] = function(expr, fn)
    return fn(expr.value, fn)
  end,
}
local function has_region_access(expr)
  if std.is_ref(expr.expr_type) then
    return expr.expr_type
  else
    local check = has_region_access_checks[expr.node_type] or false
    return check and check(expr, has_region_access)
  end
end

-- TODO: Optimize cases when regions are co-located in one instance
local function split_region_access(cx, lhs, rhs, ref_type, reads, template)
  assert(#ref_type.bounds_symbols == 1)
  local region_symbol = ref_type.bounds_symbols[1]
  local value_type = std.as_read(ref_type)
  local field_paths = std.flatten_struct_fields(std.as_read(ref_type))
  local keys = field_paths:map(function(field_path)
    return data.newtuple(region_symbol, ref_type.field_path .. field_path)
  end)
  local region_params_set = hash_set.new()
  keys:map(function(key)
    region_params_set:insert_all(cx.accesses_to_region_params[key])
  end)
  local region_params = region_params_set:to_list()

  local index = nil
  if reads then
    index = find_index(rhs)
  else
    index = find_index(lhs)
  end
  index = rewrite_accesses.expr(cx, index)

  local cases = region_params:map(function(region_param)
    local case_lhs = lhs
    local case_rhs = rhs
    local local_mapping = { [region_symbol] = region_param }
    if reads then
      case_rhs = rewrite_region_access(cx, local_mapping, case_rhs)
    else
      case_lhs = rewrite_region_access(cx, local_mapping, case_lhs)
    end
    return template {
      lhs = case_lhs,
      rhs = case_rhs,
    }
  end)

  if #cases == 1 then
    return cases[1]
  else
    local else_block = false
    if not cx.demand_cuda then
      local guard = ast_util.mk_expr_call(
        std.assert,
        terralib.newlist({
          ast_util.mk_expr_constant(false, bool),
          ast_util.mk_expr_constant("unreachable", rawstring)}),
        true)
      else_block = ast.typed.Block {
        stats = terralib.newlist({
          ast.typed.stat.Expr {
            expr = guard,
            span = template.span,
            annotations = ast.default_annotations(),
          }
        }),
        span = template.span,
      }
    else
      else_block = ast.typed.Block {
        stats = terralib.newlist(),
        span = template.span,
      }
    end
    local stat = nil
    for idx = #region_params, 1, -1 do
      local region_param = region_params[idx]
      local case = cases[idx]
      local cond = ast.typed.expr.Binary {
        op = "<=",
        lhs = index,
        rhs = ast.typed.expr.FieldAccess {
          field_name = "ispace",
          value = ast.typed.expr.ID {
            value = region_param,
            expr_type = std.rawref(&region_param:gettype()),
            span = template.span,
            annotations = ast.default_annotations(),
          },
          expr_type = region_param:gettype():ispace(),
          span = template.span,
          annotations = ast.default_annotations(),
        },
        expr_type = bool,
        span = template.span,
        annotations = ast.default_annotations(),
      }
      stat = ast.typed.stat.If {
        cond = cond,
        then_block = ast.typed.Block {
          stats = terralib.newlist({case}),
          span = template.span,
        },
        elseif_blocks = terralib.newlist(),
        else_block = else_block,
        span = template.span,
        annotations = ast.default_annotations(),
      }
      else_block = ast.typed.Block {
        stats = terralib.newlist({stat}),
        span = template.span,
      }
    end
    return stat
  end
end

function rewrite_accesses.stat_var(cx, stat)
  local symbol = stat.symbol
  local type = cx:rewrite_type(stat.type)
  if type ~= stat.type then
    symbol = std.newsymbol(type, symbol:getname())
    cx:update_mapping(stat.symbol, symbol)
  end
  local value = stat.value
  if value then
    local ref_type = has_region_access(value)
    if ref_type and
       data.all(ref_type.bounds_symbols:map(function(bound_symbol)
             return std.is_region(std.as_read(bound_symbol:gettype()))
           end))
    then
      local stats = terralib.newlist()
      stats:insert(stat {
        symbol = symbol,
        type = type,
        value = false,
      })
      local lhs = ast.typed.expr.ID {
        value = symbol,
        expr_type = std.rawref(&type),
        span = stat.span,
        annotations = ast.default_annotations(),
      }
      local template = ast.typed.stat.Assignment {
        lhs = lhs,
        rhs = value,
        span = stat.span,
        annotations = ast.default_annotations(),
        metadata = false,
      }
      stats:insert(split_region_access(cx, lhs, value, ref_type, true, template))
      return stats
    end
    value = rewrite_accesses.expr(cx, value)
  end
  return stat {
    symbol = symbol,
    type = type,
    value = value,
  }
end

function rewrite_accesses.stat_assignment_or_reduce(cx, stat)
  local lhs = stat.lhs
  local rhs = rewrite_accesses.expr(cx, stat.rhs)
  local ref_type = has_region_access(lhs)
  if ref_type and
     data.all(ref_type.bounds_symbols:map(function(bound_symbol)
           return std.is_region(std.as_read(bound_symbol:gettype()))
         end))
  then
    return split_region_access(cx, lhs, rhs, ref_type, false, stat)
  end
  lhs = rewrite_accesses.expr(cx, lhs)
  return stat {
    lhs = lhs,
    rhs = rhs,
  }
end

function rewrite_accesses.pass_through_stat(cx, stat) return stat end

local rewrite_accesses_stat_table = {
  [ast.typed.stat.ForList]         = rewrite_accesses.stat_for_list,

  [ast.typed.stat.If]              = rewrite_accesses.stat_if,
  [ast.typed.stat.While]           = rewrite_accesses.stat_while,
  [ast.typed.stat.ForNum]          = rewrite_accesses.stat_for_num,
  [ast.typed.stat.Repeat]          = rewrite_accesses.stat_repeat,
  [ast.typed.stat.Block]           = rewrite_accesses.stat_block,

  [ast.typed.stat.Var]             = rewrite_accesses.stat_var,
  [ast.typed.stat.Assignment]      = rewrite_accesses.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]          = rewrite_accesses.stat_assignment_or_reduce,

  [ast.typed.stat.VarUnpack]       = rewrite_accesses.pass_through_stat,
  [ast.typed.stat.Return]          = rewrite_accesses.pass_through_stat,
  [ast.typed.stat.Expr]            = rewrite_accesses.pass_through_stat,

  [ast.typed.stat.Break]           = rewrite_accesses.pass_through_stat,
  [ast.typed.stat.ParallelPrefix]  = rewrite_accesses.pass_through_stat,
  [ast.typed.stat.RawDelete]       = rewrite_accesses.pass_through_stat,
  [ast.typed.stat.Fence]           = rewrite_accesses.pass_through_stat,

  [ast.typed.stat.Elseif]          = unreachable,
  [ast.typed.stat.Internal]        = unreachable,

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
}

local rewrite_accesses_stat = ast.make_single_dispatch(
  rewrite_accesses_stat_table,
  {ast.typed.stat})

function rewrite_accesses.stat(cx, stat)
  return rewrite_accesses_stat(cx)(stat)
end

function rewrite_accesses.block(cx, node)
  local stats = terralib.newlist()
  node.stats:map(function(stat)
    local result = rewrite_accesses.stat(cx, stat)
    if terralib.islist(result) then
      stats:insertall(result)
    else
      stats:insert(result)
    end
  end)
  return node { stats = stats }
end

local task_generator = {}

function task_generator.new(node)
  local cx = node.prototype:get_partitioning_constraints()
  return function(pair_of_mappings,
                  mappings_by_access_paths,
                  loop_range_partitions)
    local my_ranges_to_caller_ranges, my_regions_to_caller_regions =
      unpack(pair_of_mappings)
    local partitions_to_region_params = data.newmap()
    local privileges_by_region_params = data.newmap()
    local my_accesses_to_region_params = data.newmap()
    local loop_var_to_regions = data.newmap()
    local region_params_to_partitions = data.newmap()

    for my_region_symbol, accesses_summary in cx.field_accesses_summary:items() do
      for field_path, summary in accesses_summary:items() do
        local my_ranges_set, my_privilege = unpack(summary)
        local all_region_params = hash_set.new()
        my_ranges_set:foreach(function(my_range)
          local caller_region = my_regions_to_caller_regions[my_region_symbol]
          local key = data.newtuple(caller_region, field_path)
          local range = my_ranges_to_caller_ranges[my_range]
          local partitions = mappings_by_access_paths[key][range]

          local region_params = partitions:map(function(partition)
            return find_or_create(partitions_to_region_params, partition,
              function()
                local region = my_region_symbol:gettype()
                local ispace = std.newsymbol(std.ispace(region:ispace().index_type))
                local new_region = std.region(ispace, region:fspace())
                local new_region_name = my_region_symbol:getname()
                if not partition:gettype():is_disjoint() then
                  new_region_name = new_region_name .. "_g"
                end
                return std.newsymbol(new_region, new_region_name)
              end)
          end)

          data.zip(region_params, partitions):map(function(pair)
            local region_param, partition = unpack(pair)
            assert(region_params_to_partitions[region_param] == nil or
                   region_params_to_partitions[region_param] == partition)
            region_params_to_partitions[region_param] = partition
          end)

          data.zip(region_params, partitions):map(function(pair)
            local region_param, partition = unpack(pair)
            local all_privileges = find_or_create(privileges_by_region_params, region_param)
            local field_privileges = find_or_create(all_privileges, field_path, hash_set.new)
            local privilege = my_privilege
            -- TODO: Need to create private-vs-shared partitions to actually do this
            --if std.is_reduce(privilege) and partition:gettype():is_disjoint() then
            --  privilege = "reads_writes"
            --end
            if privilege == "reads_writes" then
              field_privileges:insert(std.reads)
              field_privileges:insert(std.writes)
            else
              field_privileges:insert(privilege)
            end
          end)

          region_params:map(function(region_param)
            all_region_params:insert(region_param)
          end)
        end)

        local key = data.newtuple(my_region_symbol, field_path)
        my_accesses_to_region_params[key] = all_region_params
      end
    end

    for loop_var, my_range in cx.loop_ranges:items() do
      local range = my_ranges_to_caller_ranges[my_range]
      local partitions = loop_range_partitions[range]
      local region_params = partitions:map(function(partition)
        local region_param = partitions_to_region_params[partition]
        assert(region_param ~= nil)
        return region_param
      end)
      assert(loop_var_to_regions[loop_var] == nil)
      loop_var_to_regions[loop_var] = hash_set.from_list(region_params)
    end

    local serial_task_ast = node
    local parallel_task_name = serial_task_ast.name .. data.newtuple("parallel")
    local parallel_task = std.new_task(parallel_task_name)

    local params = terralib.newlist()
    local privileges = terralib.newlist()
    local region_universe = data.newmap()
    -- TODO: Inherit coherence modes from the serial task
    local coherence_modes = data.new_recursive_map(1)
    for region_param, all_privileges in privileges_by_region_params:items() do
      local region = region_param:gettype()
      params:insert(ast.typed.top.TaskParam {
        symbol = region_param,
        param_type = region_param:gettype(),
        future = false,
        span = serial_task_ast.span,
        annotations = ast.default_annotations(),
      })
      region_universe[region] = true
      local region_privileges = terralib.newlist()
      for field, field_privileges in all_privileges:items() do
        field_privileges:foreach(function(privilege)
          region_privileges:insert(std.privilege(privilege, region_param, field))
        end)
      end
      privileges:insert(region_privileges)
    end

    local param_mapping = data.newmap()
    serial_task_ast.params:map(function(param)
      if not std.is_region(param.param_type) then
        local new_param_symbol =
          std.newsymbol(param.symbol:gettype(), param.symbol:getname())
        param_mapping[param.symbol] = new_param_symbol
        params:insert(ast.typed.top.TaskParam {
          symbol = new_param_symbol,
          param_type = param.param_type,
          future = param.future,
          span = param.span,
          annotations = param.annotations,
        })
      end
    end)

    parallel_task:set_type(terralib.types.functype(
      params:map(function(param) return param.param_type end),
      serial_task_ast.return_type, false))
    parallel_task:set_param_symbols(params:map(function(param) return param.symbol end))

    parallel_task:set_primary_variant(parallel_task:make_variant("primary"))
    parallel_task:set_privileges(privileges)
    parallel_task:set_region_universe(region_universe)
    parallel_task:set_coherence_modes(coherence_modes)

    -- TODO: These should be inherited from the original task
    local flags = data.new_recursive_map(2)
    local param_constraints = terralib.newlist()
    local constraints = data.new_recursive_map(2)
    parallel_task:set_flags(flags)
    parallel_task:set_conditions({})
    parallel_task:set_param_constraints(param_constraints)
    parallel_task:set_constraints(constraints)
    local rewriter_cx =
      rewriter_context.new(my_accesses_to_region_params,
                           loop_var_to_regions,
                           param_mapping,
                           serial_task_ast.annotations.cuda:is(ast.annotation.Demand))
    local parallel_task_ast = ast.typed.top.Task {
      name = parallel_task_name,
      params = params,
      return_type = serial_task_ast.return_type,
      privileges = privileges,
      coherence_modes = coherence_modes,
      body = rewrite_accesses.block(rewriter_cx, serial_task_ast.body),

      -- TODO: These should be inherited from the original task
      flags = flags,
      conditions = {},
      constraints = param_constraints,

      config_options = serial_task_ast.config_options,
      region_divergence = false,
      metadata = false,
      prototype = parallel_task,
      annotations = serial_task_ast.annotations {
        parallel = ast.annotation.Forbid { value = false },
      },
      span = serial_task_ast.span,
    }
    passes.codegen(passes.optimize(parallel_task_ast), true)
    return parallel_task, region_params_to_partitions, serial_task_ast.metadata
  end
end

return task_generator
