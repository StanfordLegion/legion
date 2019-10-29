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

local function clone_list(l) return l:map(function(x) return x end) end

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
                              node_ids_to_ranges,
                              reindexed_accesses,
                              region_params_to_partitions,
                              loop_var_to_regions,
                              param_mapping,
                              incl_check_caches,
                              transitively_closed,
                              demand_cuda,
                              task_color_symbol,
                              options)
  local cx = {
    accesses_to_region_params   = accesses_to_region_params,
    node_ids_to_ranges          = node_ids_to_ranges,
    reindexed_accesses          = reindexed_accesses,
    region_params_to_partitions = region_params_to_partitions,
    loop_var_to_regions         = loop_var_to_regions,
    symbol_mapping              = param_mapping:copy(),
    incl_check_caches           = incl_check_caches,
    transitively_closed         = transitively_closed,
    demand_cuda                 = demand_cuda,
    task_color_symbol           = task_color_symbol,
    cache_keys                  = hash_set.new(),
    cache_users                 = data.newmap(),
    loop_range                  = false,
    loop_var                    = false,
    disjoint_loop_range         = true,
    colocation                  = options.colocation or false,
    demoted_centers             = data.newmap(),
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

function rewriter_context:add_cache_key(variable)
  self.cache_keys:insert(variable)
end

function rewriter_context:is_cache_key(variable)
  return self.cache_keys:has(variable)
end

function rewriter_context:add_cache_user(variable, key)
  self.cache_users[variable] = key
end

function rewriter_context:find_cache_key_for_user(variable)
  return self.cache_users[variable]
          -- TODO: Here we need to return the last cache key used in the
          --       sequence of pointer chasings instead of the loop variable
         or self.loop_var
end

function rewriter_context:is_transitively_closed(region_symbol, field_path)
  return self.transitively_closed[region_symbol] and
         self.transitively_closed[region_symbol]:has(field_path)
end

function rewriter_context:set_loop_context(loop_range, loop_var)
  self.loop_range = loop_range
  self.loop_var = loop_var
  self.disjoint_loop_range =
    self.region_params_to_partitions[loop_range]:gettype():is_disjoint()
end

function rewriter_context:get_loop_var()
  return self.loop_var
end

function rewriter_context:get_incl_cache(index)
  -- TODO: We use AST nodes of index expressions as keys of a map,
  --       which are brittle as any AST transformation can break this.
  --       Until we find a better way, we keep this scheme.
  return (self.incl_check_caches[index] and
          self.incl_check_caches[index][self.loop_range]) or false
end

function rewriter_context:add_demoted_center(symbol, region_symbol)
  self.demoted_centers[symbol] = region_symbol
end

function rewriter_context:is_demoted_center(symbol)
  return self.demoted_centers[symbol] or false
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
  [ast.typed.expr.RawTask]                    = rewrite_accesses.pass_through_expr,
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
  [ast.typed.expr.Global]                     = unreachable,
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
  [ast.typed.expr.AddressOf]                  = unreachable,
  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.ParallelizerConstraint]     = unreachable,
  [ast.typed.expr.ImportIspace]               = unreachable,
  [ast.typed.expr.ImportRegion]               = unreachable,
  [ast.typed.expr.ImportPartition]            = unreachable,
  [ast.typed.expr.Projection]                 = unreachable,
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
    return region_params:map(function(region_param)
      cx:update_mapping(region_symbol, region_param)
      value = ast.typed.expr.ID {
        value = region_param,
        expr_type = std.rawref(&region_param:gettype()),
        span = value.span,
        annotations = value.annotations,
      }
      local new_symbol = std.newsymbol(cx:rewrite_type(symbol_type), symbol:getname())
      cx:update_mapping(symbol, new_symbol)
      cx:set_loop_context(region_param, new_symbol)
      return stat {
        symbol = new_symbol,
        value = value,
        block = rewrite_accesses.block(cx, stat.block),
        metadata = false,
      }
    end)
  else
    return stat {
      symbol = symbol,
      value = value,
      block = rewrite_accesses.block(cx, stat.block),
      metadata = false,
    }
  end
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
    return expr.value, expr.node_id
  elseif expr:is(ast.typed.expr.IndexAccess) then
    if std.is_ref(expr.expr_type) then
      return expr.index, expr.node_id
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

local function get_source_location(node)
  assert(node.span.source and node.span.start.line)
  return tostring(node.span.source) .. ":" .. tostring(node.span.start.line)
end

local function split_region_access(cx, lhs, rhs, ref_type, reads, template)
  assert(#ref_type.bounds_symbols == 1)
  local region_symbol = ref_type.bounds_symbols[1]

  local index = nil
  local node_id = nil
  if reads then
    index, node_id = find_index(rhs)
  else
    index, node_id = find_index(lhs)
  end
  assert(node_id ~= nil)
  local cache = cx:get_incl_cache(index)

  local index_type = index.expr_type
  if std.is_rawref(index_type) then
    index_type = std.as_read(index_type)
  end
  local centered = false
  local region_param = false
  local centered =
    (std.is_bounded_type(index_type) and
     #index_type.bounds_symbols == 1 and
     index_type.bounds_symbols[1] == region_symbol) or
     cx:is_demoted_center(index.value) == region_symbol
  if centered then
    region_param = cx.symbol_mapping[region_symbol]
  end
  if not centered then
    centered =
      cx:is_demoted_center(index.value) == region_symbol
  end
  index = rewrite_accesses.expr(cx, index)

  if centered then
    assert(region_symbol)
    local index_type = index.expr_type
    if std.is_rawref(index_type) then
      index_type = std.as_read(index_type)
    end
    local local_mapping = { [region_symbol] = region_param }
    if reads then
      rhs = rewrite_region_access(cx, local_mapping, rhs)
    else
      lhs = rewrite_region_access(cx, local_mapping, lhs)
    end
    return template {
      lhs = lhs,
      rhs = rhs,
    }
  end

  local value_type = std.as_read(ref_type)
  local field_paths = std.flatten_struct_fields(std.as_read(ref_type))
  local keys = field_paths:map(function(field_path)
    return data.newtuple(region_symbol, ref_type.field_path .. field_path)
  end)
  local region_params_set = hash_set.new()
  local reindexed = false
  local range = cx.node_ids_to_ranges[node_id]
  keys:map(function(key)
    assert(cx.accesses_to_region_params[key][range] ~= nil)
    region_params_set:insert_all(cx.accesses_to_region_params[key][range])
    reindexed = reindexed or cx.reindexed_accesses:has(key)
  end)
  local region_params = region_params_set:to_list()
  local primary_idx = 1
  for idx, region_param in ipairs(region_params) do
    local partition = cx.region_params_to_partitions[region_param]
    if partition:gettype():is_disjoint() then
      primary_idx = idx
    end
  end

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

  if cx.colocation and template:is(ast.typed.stat.Assignment) then
    return cases[primary_idx]
  elseif #cases == 1 then
    local case = cases[1]
    if not reindexed then
      return case
    else
      if std.config["parallelize-cache-incl-check"] then
        local cache_key = cx:find_cache_key_for_user(index.value)
        local cache_region, _, _, is_disjoint = unpack(cache)
        local rhs = ast.typed.expr.Constant {
          value = 1,
          expr_type = uint8,
          span = template.span,
          annotations = ast.default_annotations(),
        }
        if not is_disjoint then
          rhs = ast.typed.expr.ID {
            value = cx.task_color_symbol,
            expr_type = std.rawref(&cx.task_color_symbol:gettype()),
            span = template.span,
            annotations = ast.default_annotations(),
          }
        end
        return ast.typed.stat.If {
          cond = ast.typed.expr.Binary {
            op = "==",
            lhs = ast.typed.expr.IndexAccess {
              value = ast.typed.expr.ID {
                value = cache_region,
                expr_type = std.rawref(&cache_region:gettype()),
                span = template.span,
                annotations = ast.default_annotations(),
              },
              index = ast.typed.expr.ID {
                value = cache_key,
                expr_type = std.rawref(&cache_key:gettype()),
                span = template.span,
                annotations = ast.default_annotations(),
              },
              expr_type = std.ref(cache_region:gettype():ispace().index_type(
                    cache_region:gettype():fspace(), cache_region)),
              span = template.span,
              annotations = ast.default_annotations(),
            },
            rhs = rhs,
            expr_type = bool,
            span = template.span,
            annotations = ast.default_annotations(),
          },
          then_block = ast.typed.Block {
            stats = terralib.newlist({case}),
            span = template.span,
          },
          elseif_blocks = terralib.newlist(),
          else_block = ast.typed.Block {
            stats = terralib.newlist(),
            span = template.span,
          },
          span = template.span,
          annotations = ast.default_annotations(),
        }

      else
        local region_param = region_params[1]
        return ast.typed.stat.If {
          cond = ast.typed.expr.Binary {
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
          },
          then_block = ast.typed.Block {
            stats = terralib.newlist({case}),
            span = template.span,
          },
          elseif_blocks = terralib.newlist(),
          else_block = ast.typed.Block {
            stats = terralib.newlist(),
            span = template.span,
          },
          span = template.span,
          annotations = ast.default_annotations(),
        }
      end
    end
  else
    local conds = nil
    if std.config["parallelize-cache-incl-check"] then
      local cache_key = cx:find_cache_key_for_user(index.value)
      assert(cache)
      local cache_region, case_ids, num_cases, is_disjoint = unpack(cache)
      if not is_disjoint then
        -- Make sure the true case is reached last
        local rearranged_indices = data.filteri(function(region_param)
          return case_ids[region_param] ~= nil end, region_params)
        rearranged_indices:insertall(data.filteri(function(region_param)
          return case_ids[region_param] == nil end, region_params))
        region_params = rearranged_indices:map(function(idx) return region_params[idx] end)
        cases = rearranged_indices:map(function(idx) return cases[idx] end)
      end
      conds = region_params:map(function(region_param)
        local case_id = case_ids[region_param]
        if case_id == nil then
          if not is_disjoint then
            -- TODO: This case better be checked in the parallelizer itself,
            --       but we will depend on the bounds checks for the moment.
            return ast.typed.expr.Constant {
              value = true,
              expr_type = bool,
              span = template.span,
              annotations = ast.default_annotations(),
            }
          end
          case_id = num_cases
        end
        local rhs = ast.typed.expr.Constant {
          value = case_id,
          expr_type = uint8,
          span = template.span,
          annotations = ast.default_annotations(),
        }
        if not is_disjoint then
          rhs = ast.typed.expr.ID {
            value = cx.task_color_symbol,
            expr_type = std.rawref(&cx.task_color_symbol:gettype()),
            span = template.span,
            annotations = ast.default_annotations(),
          }
        end
        return ast.typed.expr.Binary {
          op = "==",
          lhs = ast.typed.expr.IndexAccess {
            value = ast.typed.expr.ID {
              value = cache_region,
              expr_type = std.rawref(&cache_region:gettype()),
              span = template.span,
              annotations = ast.default_annotations(),
            },
            index = ast.typed.expr.ID {
              value = cache_key,
              expr_type = std.rawref(&cache_key:gettype()),
              span = template.span,
              annotations = ast.default_annotations(),
            },
            expr_type = std.ref(cache_region:gettype():ispace().index_type(
                  cache_region:gettype():fspace(), cache_region)),
            span = template.span,
            annotations = ast.default_annotations(),
          },
          rhs = rhs,
          expr_type = bool,
          span = template.span,
          annotations = ast.default_annotations(),
        }
      end)
    else
      conds = region_params:map(function(region_param)
        return ast.typed.expr.Binary {
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
      end)
    end

    local else_block = false
    if not cx.demand_cuda then
      local guard = ast_util.mk_expr_call(
        std.assert,
        terralib.newlist({
          ast_util.mk_expr_constant(false, bool),
          ast_util.mk_expr_constant(
              "found an unreachable indirect access at " .. get_source_location(template) ..
              ". this must be a bug.",
              rawstring)}),
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
      local case = cases[idx]
      local cond = conds[idx]
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

      if stats[#stats]:is(ast.typed.stat.Assignment) then
        local stat = stats[#stats]
        local ref_type = stat.rhs.expr_type
        local value = stat.rhs.value
        while not std.is_ref(ref_type) do
          ref_type = value.expr_type
          value = value.value
        end
        if #ref_type.bounds_symbols == 1 and
           cx:is_transitively_closed(ref_type.bounds_symbols[1], ref_type.field_path)
        then
          cx:add_cache_key(symbol)
        end
        local index = find_index(stat.rhs)
        if cx:is_cache_key(index.value) then
          cx:add_cache_user(symbol, index.value)
        end
      end
      return stats
    elseif value:is(ast.typed.expr.Cast) and
           std.is_bounded_type(std.as_read(value.arg.expr_type)) and
           std.is_index_type(value.fn.value)
    then
      local bounded_type = std.as_read(value.arg.expr_type)
      if #bounded_type.bounds_symbols == 1 then
        cx:add_demoted_center(symbol, bounded_type.bounds_symbols[1])
      end
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

function rewrite_accesses.stat_var_unpack(cx, stat)
  return stat { value = rewrite_accesses.expr(cx, stat.value) }
end

function rewrite_accesses.stat_expr(cx, stat)
  return stat { expr = rewrite_accesses.expr(cx, stat.expr) }
end

function rewrite_accesses.stat_raw_delete(cx, stat)
  return stat { value = rewrite_accesses.expr(cx, stat.value) }
end

function rewrite_accesses.stat_return(cx, stat)
  return stat { value = stat.value and rewrite_accesses.expr(cx, stat.value) }
end

function rewrite_accesses.stat_parallel_prefix(cx, stat)
  return stat { dir = rewrite_accesses.expr(cx, stat.dir) }
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

  [ast.typed.stat.VarUnpack]       = rewrite_accesses.stat_var_unpack,
  [ast.typed.stat.Expr]            = rewrite_accesses.stat_expr,
  [ast.typed.stat.Return]          = rewrite_accesses.stat_return,
  [ast.typed.stat.RawDelete]       = rewrite_accesses.stat_raw_delete,
  [ast.typed.stat.ParallelPrefix]  = rewrite_accesses.stat_parallel_prefix,

  [ast.typed.stat.Break]           = rewrite_accesses.pass_through_stat,
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
  return function(pair_of_mappings, caller_cx)
    local mappings_by_access_paths = caller_cx.mappings_by_access_paths
    local loop_range_partitions = caller_cx.loop_range_partitions
    local incl_check_caches = caller_cx.incl_check_caches
    local reindexed_ranges = caller_cx.reindexed_ranges
    local transitively_closed = caller_cx.transitively_closed
    local my_ranges_to_caller_ranges, my_regions_to_caller_regions =
      unpack(pair_of_mappings)

    local partitions_to_region_params = data.newmap()
    local privileges_by_region_params = data.newmap()
    local my_accesses_to_region_params = data.newmap()
    local reindexed_accesses = hash_set.new()
    local loop_var_to_regions = data.newmap()
    local region_params_to_partitions = data.newmap()
    local param_index = 1
    local colocation_constraints_map = data.newmap()

    for my_region_symbol, accesses_summary in cx.field_accesses_summary:items() do
      for field_path, summary in accesses_summary:items() do
        local my_ranges_set, my_privilege = unpack(summary)
        local all_region_params = hash_set.new()
        local has_reduction = false
        my_ranges_set:foreach(function(my_range)
          local caller_region = my_regions_to_caller_regions[my_region_symbol]
          local key = data.newtuple(caller_region, field_path)
          local range = my_ranges_to_caller_ranges[my_range]
          if std.is_reduce(my_privilege) then
            if reindexed_ranges[range] ~= nil then
              range = reindexed_ranges[range]
              reindexed_accesses:insert(
                data.newtuple(my_region_symbol, field_path))
            end
          end
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
                new_region_name = new_region_name .. tostring(param_index)
                param_index = param_index + 1
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
            if std.config["parallelize-use-pvs"] and std.is_reduce(privilege) and
               partition:gettype():is_disjoint() and
               data.all(unpack(partitions:map(function(other_partition)
                 if partition == other_partition then return true end
                 local partition_type = partition:gettype()
                 local other_partition_type = other_partition:gettype()
                 local constraint =
                   std.constraint(partition_type:parent_region(),
                                  other_partition_type:parent_region(),
                                  std.disjointness)
                 return std.check_constraint(caller_cx, constraint) ~= nil
               end)))
            then
              privilege = "reads_writes"
            end
            if privilege == "reads_writes" then
              field_privileges:insert(std.reads)
              field_privileges:insert(std.writes)
            else
              field_privileges:insert(privilege)
              has_reduction = has_reduction or std.is_reduce(privilege)
            end
          end)

          region_params:map(function(region_param)
            all_region_params:insert(region_param)
          end)

          local key = data.newtuple(my_region_symbol, field_path)
          find_or_create(my_accesses_to_region_params, key)[my_range] =
            hash_set.from_list(region_params)
        end)

        -- Force canonicalization
        all_region_params:hash()

        -- Collect accesses that are mapped to multiple regions in order to
        -- generate colocation constraints later
        if all_region_params:size() > 1 then
          local key = data.newtuple(my_region_symbol, field_path)
          if has_reduction then
            local reduction_params = data.filter(function(param)
                local privileges = privileges_by_region_params[param][field_path]:to_list()
                return #privileges == 1 and std.is_reduce(privileges[1])
              end, all_region_params:to_list())
            if #reduction_params > 1 then
              colocation_constraints_map[key] = reduction_params
            end
          else
            colocation_constraints_map[key] = all_region_params:to_list()
          end
        end
      end
    end

    -- Add region parameters for ranges that are not found in any region accesses
    -- but are used in loops
    for loop_var, my_range in cx.loop_ranges:items() do
      local range = my_ranges_to_caller_ranges[my_range]
      local partitions = loop_range_partitions[range]
      local region_params = partitions:map(function(partition)
        local region_param =
          find_or_create(partitions_to_region_params, partition,
            function()
              local my_region_symbol = cx.constraints:get_partition(my_range).region
              local region = my_region_symbol:gettype()
              local ispace = std.newsymbol(std.ispace(region:ispace().index_type))
              local new_region = std.region(ispace, region:fspace())
              local new_region_name = my_region_symbol:getname() .. tostring(param_index)
              param_index = param_index + 1
              local region_param = std.newsymbol(new_region, new_region_name)
              assert(region_params_to_partitions[region_param] == nil)
              region_params_to_partitions[region_param] = partition
              return region_param
            end)
        return region_param
      end)
      assert(loop_var_to_regions[loop_var] == nil)
      loop_var_to_regions[loop_var] = hash_set.from_list(region_params)
    end

    local cache_params = terralib.newlist()
    local my_incl_check_caches = {}

    -- Add region parameters for inclusion check caches
    if std.config["parallelize-cache-incl-check"] then
      local cache_params_map = data.newmap()
      for my_range, _ in cx.constraints.ranges:items() do
        local range = my_ranges_to_caller_ranges[my_range]
        if incl_check_caches[range] ~= nil then
          local my_incl_check_cache = data.newmap()
          for loop_range_partition, tuple in incl_check_caches[range]:items() do
            local cache_partition, case_ids, num_cases, is_disjoint = unpack(tuple)
            local my_cache_param = find_or_create(cache_params_map, range,
              function()
                local region_symbol = cache_partition:gettype().parent_region_symbol
                local region = region_symbol:gettype()
                local ispace = std.newsymbol(std.ispace(region:ispace().index_type))
                local new_region = std.region(ispace, region:fspace())
                local new_region_name =
                  region_symbol:getname() .. "_" .. tostring(param_index)
                param_index = param_index + 1
                local region_param = std.newsymbol(new_region, new_region_name)
                assert(region_params_to_partitions[region_param] == nil)
                region_params_to_partitions[region_param] = cache_partition
                cache_params:insert(region_param)
                return region_param
              end)
            local loop_range_region_param =
              partitions_to_region_params[loop_range_partition]
            if loop_range_region_param ~= nil then
              local my_case_ids = data.newmap()
              case_ids:map(function(partition, case_id)
                my_case_ids[partitions_to_region_params[partition]] = case_id
              end)
              my_incl_check_cache[loop_range_region_param] =
                { my_cache_param, my_case_ids, num_cases, is_disjoint }
            end
          end
          local indices = cx:find_indices_of_range(my_range)
          for _, index in ipairs(indices) do
            my_incl_check_caches[index] = my_incl_check_cache
          end
        end
      end
    end

    local my_transitively_closed = data.newmap()
    for partition, field_paths in transitively_closed:items() do
      local my_region_param = partitions_to_region_params[partition]
      if my_region_param ~= nil then
        find_or_create(my_transitively_closed, my_region_param,
            hash_set.new):insert_all(field_paths)
      end
    end

    local serial_task_ast = node
    local parallel_task_name = serial_task_ast.name .. data.newtuple("parallel")

    -- TODO: We want to create task variants from different ASTs, which are not
    --       supported by the compiler at the moment. So, we use a hack that
    --       creates a complete task for each unique AST and later forces
    --       the tasks to have the same task id.
    local parallel_task_variants = data.newmap()
    parallel_task_variants["colocation"] = std.new_task(parallel_task_name)
    parallel_task_variants["primary"] = std.new_task(parallel_task_name)
    for _, parallel_task in parallel_task_variants:items() do
      parallel_task:set_task_id_unsafe(parallel_task_variants["primary"]:get_task_id())
    end
    local parallel_task_variants_list = terralib.newlist({"colocation", "primary"})

    -- We prepare the metadata that is shared by all task variants
    local params = terralib.newlist()
    local privileges = terralib.newlist()
    local region_universe = data.newmap()

    -- TODO: Inherit coherence modes from the serial task
    local coherence_modes = data.new_recursive_map(1)
    local region_params_with_accesses = hash_set.new()
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
          local field_type = std.get_field_path(region_param:gettype():fspace(), field)
          if std.is_reduce(privilege) and field_type:isarray() then
            std.update_reduction_op(privilege.op, field_type)
          end
        end)
      end
      privileges:insert(region_privileges)
      region_params_with_accesses:insert(region_param)
    end
    for loop_var, region_params in loop_var_to_regions:items() do
      region_params:foreach(function(region_param)
        if not region_params_with_accesses:has(region_param) then
          local region = region_param:gettype()
          params:insert(ast.typed.top.TaskParam {
            symbol = region_param,
            param_type = region_param:gettype(),
            future = false,
            span = serial_task_ast.span,
            annotations = ast.default_annotations(),
          })
          region_universe[region] = true
        end
      end)
    end

    cache_params:map(function(region_param)
      local region = region_param:gettype()
      params:insert(ast.typed.top.TaskParam {
        symbol = region_param,
        param_type = region_param:gettype(),
        future = false,
        span = serial_task_ast.span,
        annotations = ast.default_annotations(),
      })
      privileges:insert(terralib.newlist({
        std.privilege(std.reads, region_param, data.newtuple())}))
      region_universe[region] = true
    end)

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

    for i, variant_type in ipairs(parallel_task_variants_list) do
      local parallel_task = parallel_task_variants[variant_type]

      -- Make a copy from the shared metadata
      local params = clone_list(params)
      local privileges = clone_list(privileges)
      local region_universe = region_universe:copy()
      local coherence_modes = coherence_modes:copy()
      local options = { [variant_type] = true }

      local task_color_symbol = std.newsymbol(
          caller_cx.color_space_symbol:gettype().index_type,
          "task_color")
      params:insert(ast.typed.top.TaskParam {
        symbol = task_color_symbol,
        param_type = task_color_symbol:gettype(),
        future = false,
        span = serial_task_ast.span,
        annotations = ast.default_annotations(),
      })

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
                             cx.node_ids_to_ranges,
                             reindexed_accesses,
                             region_params_to_partitions,
                             loop_var_to_regions,
                             param_mapping,
                             my_incl_check_caches,
                             my_transitively_closed,
                             serial_task_ast.annotations.cuda:is(ast.annotation.Demand),
                             task_color_symbol,
                             options)
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
          cuda = ((std.config["parallelize-cache-incl-check"] or
                   variant_type == "colocation") and
                  serial_task_ast.annotations.cuda) or
                 ast.annotation.Forbid { value = false }
        },
        span = serial_task_ast.span,
      }
      passes.codegen(passes.optimize(parallel_task_ast), true)

      if variant_type == "colocation" then
        local colocation_constraints = terralib.newlist()
        for key, region_params in colocation_constraints_map:items() do
          assert(#region_params > 1)
          local _, field_path = unpack(key)
          colocation_constraints:insert(
            std.layout.colocation_constraint(
              region_params:map(function(region_param)
                return std.layout.field_constraint(region_param:getname(),
                  terralib.newlist({field_path}))
              end)))
        end

        parallel_task:get_variants():map(function(variant)
          colocation_constraints:map(function(colocation_constraint)
            variant:set_name("colocation")
            variant:add_execution_constraint(colocation_constraint)
          end)
        end)
      end
    end

    return parallel_task_variants, region_params_to_partitions, serial_task_ast.metadata
  end
end

return task_generator
