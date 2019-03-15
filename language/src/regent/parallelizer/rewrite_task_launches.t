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

local rewriter_context = {}

rewriter_context.__index = rewriter_context

function rewriter_context.new(mappings,
                              mappings_by_access_paths,
                              loop_range_partitions,
                              color_space_symbol,
                              constraints)
  local cx = {
    mappings = mappings,
    mappings_by_access_paths = mappings_by_access_paths,
    loop_range_partitions = loop_range_partitions,
    color_space_symbol = color_space_symbol,
    constraints = constraints,
  }

  return setmetatable(cx, rewriter_context)
end

local rewrite_task_launches = {}

function rewrite_task_launches.stat_block(cx, stat)
  local block = rewrite_task_launches.block(cx, stat.block)
  return stat { block = block }
end

function rewrite_task_launches.stat_if(cx, stat)
  local then_block = rewrite_task_launches.block(cx, stat.then_block)
  local else_block = rewrite_task_launches.block(cx, stat.else_block)
  return stat {
    then_block = then_block,
    else_block = else_block,
  }
end

local function create_index_launch(cx, task, call, stat)
  local generator = task:get_parallel_task_generator()
  local pair_of_mappings = cx.mappings[task]
  assert(pair_of_mappings ~= nil)
  local parallel_task_variants, params_to_partitions, metadata = generator(pair_of_mappings, cx)
  local parallel_task = parallel_task_variants["primary"]

  -- Create an index space launch
  local loop_var_type = cx.color_space_symbol:gettype().index_type(cx.color_space_symbol)
  local loop_var_symbol = std.newsymbol(loop_var_type, "color")
  local loop_var = ast.typed.expr.ID {
    value = loop_var_symbol,
    expr_type = std.rawref(&loop_var_symbol:gettype()),
    span = call.span,
    annotations = ast.default_annotations(),
  }
  local color_space = ast.typed.expr.ID {
    value = cx.color_space_symbol,
    expr_type = std.rawref(&cx.color_space_symbol:gettype()),
    span = call.span,
    annotations = ast.default_annotations(),
  }
  local region_param_symbols = data.filter(function(param_symbol)
      return std.is_region(param_symbol:gettype())
    end, parallel_task:get_param_symbols())
  local args = region_param_symbols:map(function(param_symbol)
    local partition_symbol = params_to_partitions[param_symbol]
    assert(partition_symbol ~= nil)
    local partition_type = partition_symbol:gettype()
    local parent_type = partition_type:parent_region()
    local subregion_type = partition_type:subregion_constant(loop_var)
    std.add_constraint(cx, partition_type, parent_type, std.subregion, false)
    std.add_constraint(cx, subregion_type, partition_type, std.subregion, false)
    local partition = ast.typed.expr.ID {
      value = partition_symbol,
      expr_type = std.rawref(&partition_symbol:gettype()),
      span = call.span,
      annotations = ast.default_annotations(),
    }
    return ast.typed.expr.IndexAccess {
      value = partition,
      index = loop_var,
      expr_type = subregion_type,
      span = call.span,
      annotations = ast.default_annotations(),
    }
  end)
  args:insertall(data.filter(function(arg)
    return not std.is_region(std.as_read(arg.expr_type)) end,
  call.args))

  if stat:is(ast.typed.stat.Var) and metadata.reduction then
    local value_type = metadata.reduction:gettype()
    assert(std.type_eq(value_type, stat.symbol:gettype()))
    local init = std.reduction_op_init[metadata.op][value_type]
    local stats = terralib.newlist()
    local lhs = ast.typed.expr.ID {
      value = stat.symbol,
      expr_type = std.rawref(&value_type),
      span = stat.span,
      annotations = ast.default_annotations(),
    }
    stats:insert(stat {
      value = ast.typed.expr.Constant {
        value = init,
        expr_type = value_type,
        span = call.span,
        annotations = ast.default_annotations(),
      },
    })
    stats:insert(ast.typed.stat.ForList {
      symbol = loop_var_symbol,
      value = color_space,
      block = ast.typed.Block {
        stats = terralib.newlist({
          ast.typed.stat.Reduce {
            op = metadata.op,
            lhs = lhs,
            rhs = call {
              fn = call.fn {
                expr_type = parallel_task:get_type(),
                value = parallel_task,
              },
              args = args,
            },
            metadata = false,
            span = stat.span,
            annotations = ast.default_annotations(),
          }
        }),
        span = call.span,
      },
      metadata = false,
      span = call.span,
      annotations = ast.default_annotations(),
    })
    return stats
  else
    return ast.typed.stat.ForList {
      symbol = loop_var_symbol,
      value = color_space,
      block = ast.typed.Block {
        stats = terralib.newlist({
          stat {
            expr = call {
              fn = call.fn {
                expr_type = parallel_task:get_type(),
                value = parallel_task,
              },
              args = args,
            }
          }
        }),
        span = call.span,
      },
      metadata = false,
      span = call.span,
      annotations = ast.default_annotations(),
    }
  end
end

function rewrite_task_launches.stat_var(cx, stat)
  if not (stat.value and stat.value:is(ast.typed.expr.Call)) then
    return stat
  end
  local call = stat.value
  local task = call.fn.value
  if not (std.is_task(task) and task:has_partitioning_constraints()) then
    return stat
  end
  return create_index_launch(cx, task, call, stat)
end

function rewrite_task_launches.stat_assignment_or_reduce(cx, stat)
  if not stat.rhs:is(ast.typed.expr.Call) then return stat end
  local stats = terralib.newlist()
  local tmp_var = std.newsymbol(stat.rhs.expr_type)
  local call_stats = rewrite_task_launches.stat(cx,
    ast.typed.stat.Var {
      symbol = tmp_var,
      type = tmp_var:gettype(),
      value = stat.rhs,
      span = stat.rhs.span,
      annotations = ast.default_annotations(),
    })
  if terralib.islist(call_stats) then
    stats:insertall(call_stats)
  else
    stats:insert(call_stats)
  end
  stats:insert(stat {
    rhs = ast.typed.expr.ID {
      value = tmp_var,
      expr_type = std.rawref(&tmp_var:gettype()),
      span = stat.rhs.span,
      annotations = ast.default_annotations(),
    },
  })
  return stats
end

function rewrite_task_launches.stat_expr(cx, stat)
  if not stat.expr:is(ast.typed.expr.Call) then return stat end

  local call = stat.expr
  local task = call.fn.value
  if not (std.is_task(task) and task:has_partitioning_constraints()) then
    return stat
  end
  return create_index_launch(cx, task, call, stat)
end

function rewrite_task_launches.pass_through_stat(cx, stat)
  return stat
end

local rewrite_task_launches_stat_table = {
  [ast.typed.stat.ParallelizeWith] = rewrite_task_launches.stat_parallelize_with,

  [ast.typed.stat.Var]        = rewrite_task_launches.stat_var,
  [ast.typed.stat.Expr]       = rewrite_task_launches.stat_expr,
  [ast.typed.stat.Assignment] = rewrite_task_launches.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = rewrite_task_launches.stat_assignment_or_reduce,

  [ast.typed.stat.ForList]    = rewrite_task_launches.stat_block,
  [ast.typed.stat.ForNum]     = rewrite_task_launches.stat_block,
  [ast.typed.stat.While]      = rewrite_task_launches.stat_block,
  [ast.typed.stat.Repeat]     = rewrite_task_launches.stat_block,
  [ast.typed.stat.Block]      = rewrite_task_launches.stat_block,
  [ast.typed.stat.MustEpoch]  = rewrite_task_launches.stat_block,

  [ast.typed.stat.If]         = rewrite_task_launches.stat_if,

  [ast.typed.stat]            = rewrite_task_launches.pass_through_stat,
}

local rewrite_task_launches_stat = ast.make_single_dispatch(
  rewrite_task_launches_stat_table,
  {ast.typed.stat})

function rewrite_task_launches.stat(cx, node)
  return rewrite_task_launches_stat(cx)(node)
end

function rewrite_task_launches.block(cx, block)
  local stats = terralib.newlist()
  block.stats:map(function(stat)
    local result = rewrite_task_launches.stat(cx, stat)
    if terralib.islist(result) then
      stats:insertall(result)
    else
      stats:insert(result)
    end
  end)
  return block { stats = stats }
end

function rewrite_task_launches.rewrite(solution, caller_constraints, stat)
  if #solution.all_tasks == 0 then
    return ast.typed.stat.Block {
      block = stat.block,
      span = stat.span,
      annotations = stat.annotations,
    }
  end

  local cx = rewriter_context.new(
      solution.all_mappings,
      solution.mappings_by_access_paths,
      solution.loop_range_partitions,
      solution.color_space_symbol,
      caller_constraints)
  local block = rewrite_task_launches.block(cx, stat.block)

  -- TODO: Need a dataflow analysis to find the right place to put partitioning calls
  local stats = terralib.newlist()
  stats:insertall(solution.partition_stats)
  stats:insert(ast.typed.stat.Block {
    block = block,
    span = stat.block.span,
    annotations = stat.annotations,
  })

  return ast.typed.stat.Block {
    block = ast.typed.Block {
      stats = stats,
      span = stat.span,
    },
    span = stat.span,
    annotations = stat.annotations,
  }
end

return rewrite_task_launches
