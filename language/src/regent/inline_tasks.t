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

-- Regent Task Inliner

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")
local report = require("common/report")
local symbol_table = require("regent/symbol_table")

local inline_tasks = {}

local function check_valid_inline_task(node, task)
  if not task then
    report.error(node, "cannot inline a task that has no definition")
  end

  local body = task.body
  assert(body)

  local function count_return(node)
    return (node:is(ast.specialized.stat.Return) and 1) or 0
  end
  local function add(x, y) return x + y end
  local num_returns = ast.mapreduce_node_postorder(count_return, add, body, 0)
  if num_returns > 1 then
    report.error(task, "inline tasks cannot have multiple return statements")
  end
  if num_returns == 1 and not body.stats[#body.stats]:is(ast.specialized.stat.Return) then
    report.error(task, "the return statement in an inline task should be the last statement")
  end

  local function is_recursive_call(node)
    return node:is(ast.specialized.expr.Call) and node.fn.value == task.prototype
  end
  local function lor(x, y) return x or y end
  local has_self_recursion = ast.mapreduce_node_postorder(is_recursive_call, lor, body, false)
  if has_self_recursion then
    report.error(task, "inline tasks cannot be recursive")
  end
end

local function is_singleton_type(type)
  return std.is_ispace(type) or std.is_region(type) or
         std.is_list_of_regions(type) or
         std.is_partition(type) or std.is_cross_product(type)
end

local substitute = {}

local function pass_through(cx, node) return node end

local function unreachable(cx, node) return assert(false) end

local function substitute_expr_id(cx, node)
  local new_node = cx.expr_mapping[node.value] or
                   (cx.symbol_mapping[node.value] ~= nil and
                    node { value = cx.symbol_mapping[node.value] }) or
                   node
  return new_node
end

local function substitute_expr_regent_cast(cx, node)
  return node { expr_type = std.type_sub(node.expr_type, cx.symbol_mapping) }
end

local function substitute_expr_cast(cx, node)
  return node { fn = node.fn { value = std.type_sub(node.fn.value, cx.symbol_mapping) } }
end

local function substitute_expr_null(cx, node)
  return node { pointer_type = std.type_sub(node.pointer_type, cx.symbol_mapping) }
end

local function substitute_expr_region(cx, node)
  return node { fspace_type = std.type_sub(node.fspace_type, cx.symbol_mapping) }
end

local substitute_expr_table = {
  [ast.specialized.expr.ID]          = substitute_expr_id,
  [ast.specialized.expr.DynamicCast] = substitute_expr_regent_cast,
  [ast.specialized.expr.StaticCast]  = substitute_expr_regent_cast,
  [ast.specialized.expr.UnsafeCast]  = substitute_expr_regent_cast,
  [ast.specialized.expr.Cast]        = substitute_expr_cast,
  [ast.specialized.expr.Null]        = substitute_expr_null,
  [ast.specialized.expr.Region]      = substitute_expr_region,
  [ast.specialized.expr]             = pass_through,
  [ast.specialized.region]           = pass_through,
  [ast.condition_kind]               = pass_through,
  [ast.disjointness_kind]            = pass_through,
  [ast.fence_kind]                   = pass_through,
  [ast.location]                     = pass_through,
  [ast.annotation]                   = pass_through,
}

local substitute_expr = ast.make_single_dispatch(
  substitute_expr_table,
  {})

function substitute.expr(cx, node)
  return ast.map_node_postorder(substitute_expr(cx), node)
end

local function substitute_stat_if(cx, node)
  local cond = substitute.expr(cx, node.cond)
  local then_block = substitute.block(cx, node.then_block)
  local else_block = substitute.block(cx, node.else_block)
  return node {
    cond = cond,
    then_block = then_block,
    else_block = else_block,
  }
end

local function substitute_stat_while(cx, node)
  local cond = substitute.expr(cx, node.cond)
  local block = substitute.block(cx, node.block)
  return node {
    cond = cond,
    block = block,
  }
end

local function substitute_stat_for_num(cx, node)
  local values = node.values:map(function(value)
    return substitute.expr(cx, value)
  end)
  local block = substitute.block(cx, node.block)
  return node {
    values = values,
    block = block,
  }
end

local function substitute_stat_for_list(cx, node)
  local symbol = node.symbol
  local new_symbol = std.newsymbol(nil, symbol:hasname())
  cx.symbol_mapping[symbol] = new_symbol
  local value = substitute.expr(cx, node.value)
  local block = substitute.block(cx, node.block)
  return node {
    symbol = new_symbol,
    value = value,
    block = block,
  }
end

local function substitute_stat_repeat(cx, node)
  local until_cond = substitute.expr(cx, node.until_cond)
  local block = substitute.block(cx, node.block)
  return node {
    until_cond = until_cond,
    block = block,
  }
end

local function substitute_stat_block(cx, node)
  return node { block = substitute.block(cx, node.block) }
end

local function substitute_stat_var(cx, node)
  local symbol = node.symbols
  -- We should ignore the type of the existing symbol as we want to type check it again.
  local symbol_type = symbol:hastype()
  if is_singleton_type(symbol_type) then
    symbol_type = nil
  else
    symbol_type = std.type_sub(symbol_type, cx.symbol_mapping)
  end
  local new_symbol = std.newsymbol(symbol_type, symbol:hasname())
  cx.symbol_mapping[symbol] = new_symbol
  local value = node.values and substitute.expr(cx, node.values) or false
  return node {
    symbols = new_symbol,
    values = value,
  }
end

local function substitute_stat_var_unpack(cx, node)
  local symbols = node.symbols:map(function(symbol)
    -- We should ignore the type of the existing symbol as we want to type check it again.
    local symbol_type = symbol:hastype()
    if is_singleton_type(symbol_type) then
      symbol_type = nil
    else
      symbol_type = std.type_sub(symbol_type, cx.symbol_mapping)
    end
    local new_symbol = std.newsymbol(symbol_type, symbol:hasname())
    cx.symbol_mapping[symbol] = new_symbol
    return new_symbol
  end)
  local value = substitute.expr(cx, node.value)
  return node {
    symbols = symbols,
    value = value,
  }
end

local function substitute_stat_return(cx, node)
  local value = node.value and substitute.expr(cx, node.value) or false
  return node { value = value }
end

local function substitute_stat_assignment_or_reduce(cx, node)
  local lhs = substitute.expr(cx, node.lhs)
  local rhs = substitute.expr(cx, node.rhs)
  return node {
    lhs = lhs,
    rhs = rhs,
  }
end

local function substitute_stat_expr(cx, node)
  return node { expr = substitute.expr(cx, node.expr) }
end

local function substitute_stat_raw_delete(cx, node)
  return node { value = substitute.expr(cx, node.value) }
end

local function substitute_stat_parallel_prefix(cx, node)
  return node { dir = substitute.expr(cx, node.dir) }
end

local substitute_stat_table = {
  [ast.specialized.stat.If]              = substitute_stat_if,
  [ast.specialized.stat.Elseif]          = unreachable,
  [ast.specialized.stat.While]           = substitute_stat_while,
  [ast.specialized.stat.ForNum]          = substitute_stat_for_num,
  [ast.specialized.stat.ForList]         = substitute_stat_for_list,
  [ast.specialized.stat.Repeat]          = substitute_stat_repeat,
  [ast.specialized.stat.MustEpoch]       = substitute_stat_block,
  [ast.specialized.stat.Block]           = substitute_stat_block,
  [ast.specialized.stat.Var]             = substitute_stat_var,
  [ast.specialized.stat.VarUnpack]       = substitute_stat_var_unpack,
  [ast.specialized.stat.Return]          = substitute_stat_return,
  [ast.specialized.stat.Assignment]      = substitute_stat_assignment_or_reduce,
  [ast.specialized.stat.Reduce]          = substitute_stat_assignment_or_reduce,
  [ast.specialized.stat.Expr]            = substitute_stat_expr,
  [ast.specialized.stat.RawDelete]       = substitute_stat_raw_delete,
  -- TODO: Symbols in the constraints should be handled here
  [ast.specialized.stat.ParallelizeWith] = substitute_stat_block,
  [ast.specialized.stat.ParallelPrefix]  = substitute_stat_parallel_prefix,
  [ast.specialized.stat]                 = pass_through,
}

local substitute_stat = ast.make_single_dispatch(
  substitute_stat_table,
  {})

function substitute.stat(cx, node)
  return substitute_stat(cx)(node)
end

function substitute.block(cx, node)
  return node {
    stats = node.stats:map(function(stat) return substitute.stat(cx, stat) end),
  }
end

-- To be able to correctly type check the task call after inlining,
-- we preserve the original expression in an if statement that never executes.
local function preserve_task_call(node)
  return ast.specialized.stat.If {
    cond = ast.specialized.expr.Constant {
      value = false,
      expr_type = bool,
      annotations = ast.default_annotations(),
      span = node.span,
    },
    then_block = ast.specialized.Block {
      stats = terralib.newlist { node },
      span = node.span,
    },
    elseif_blocks = terralib.newlist(),
    else_block = ast.specialized.Block {
      stats = terralib.newlist(),
      span = node.span,
    },
    annotations = ast.default_annotations(),
    span = node.span,
  }
end

function inline_tasks.expr_call(call)
  -- Task T is inlined at call site C only if T demands inlining and
  -- C does not forbid inlining.
  local function demands_inlining(node)
    return node.annotations.inline:is(ast.annotation.Demand)
  end
  local function forbids_inlining(node)
    return node.annotations.inline:is(ast.annotation.Forbid)
  end

  if not call:is(ast.specialized.expr.Call) or forbids_inlining(call) then
    if call:is(ast.specialized.expr.Call) and std.is_task(call.fn.value) and
       call.fn.value.is_inline
    then
      call.fn.value:optimize()
    end
    return call, false
  end

  local task = call.fn.value
  if not std.is_task(task) then return call, false end

  local task_ast = task:has_primary_variant() and
                   task:get_primary_variant():has_untyped_ast()
  if not task_ast then
    if demands_inlining(call) then
      report.error(call, "cannot inline a task that does not demand inlining")
    end
    return call, false
  end
  assert(demands_inlining(task_ast))

  -- If the task does not demand inlining, check its validity before inlining it.
  if not demands_inlining(task_ast) then check_valid_inline_task(call, task_ast) end

  -- Once we reach this point, the task and the task call are valid for inlining.
  local actions = terralib.newlist()

  -- First, inline any task calls in the argument list.
  local args = call.args:map(function(arg)
    local new_arg, inlined, arg_actions = inline_tasks.expr(arg)
    assert(new_arg ~= nil)
    if not inlined then return arg
    else
      actions:insertall(arg_actions)
      return new_arg
    end
  end)

  -- Second, make assignments to temporary variables.
  local params = task_ast.params:map(function(param) return param.symbol end)
  local param_types = params:map(function(param) return param:gettype() end)
  local symbol_mapping = {}
  local expr_mapping = {}
  data.zip(params, param_types, args):map(function(tuple)
    local param, param_type, arg = unpack(tuple)
    if is_singleton_type(param_type) and arg:is(ast.specialized.expr.ID) then
      expr_mapping[param] = arg
      symbol_mapping[param] = arg.value
    else
      local symbol_type = nil
      if param_type:isprimitive() or param_type:ispointer() then
        symbol_type = param_type
      end
      local new_symbol = std.newsymbol(symbol_type, param:hasname())
      symbol_mapping[param] = new_symbol
      actions:insert(ast.specialized.stat.Var {
        symbols = new_symbol,
        values = arg,
        annotations = ast.default_annotations(),
        span = call.span,
      })
    end
  end)

  local cx = {
    symbol_mapping = symbol_mapping,
    expr_mapping = expr_mapping,
  }

  local stats = substitute.block(cx, task_ast.body).stats

  -- Finally, convert any return statement to an assignment to a temporary variable
  local return_var_expr = nil
  if #stats > 0 and stats[#stats]:is(ast.specialized.stat.Return) then
    local task_name = task_ast.name:mkstring("", "_", "")
    local return_var = std.newsymbol(nil, "__" .. task_name .."_ret")
    return_var_expr = ast.specialized.expr.ID {
      value = return_var,
      annotations = ast.default_annotations(),
      span = call.span,
    }
    local return_stat = stats[#stats]
    stats[#stats] = ast.specialized.stat.Var {
      symbols = return_var,
      values = return_stat.value,
      annotations = ast.default_annotations(),
      span = return_stat.span,
    }
  end

  actions:insertall(stats)
  return return_var_expr, true, actions
end

function inline_tasks.expr(node, no_return)
  local inlined_any = false
  local actions = terralib.newlist()
  local new_node = ast.map_node_postorder(function(node)
    local new_node, inlined, node_actions = inline_tasks.expr_call(node)
    if not inlined then return node
    else
      inlined_any = true
      actions:insertall(node_actions)
      return new_node
    end
  end, node)
  if not inlined_any then
    return node, false
  else
    if no_return and new_node ~= nil then
      actions:insert(ast.specialized.stat.Expr {
        expr = new_node,
        annotations = ast.default_annotations(),
        span = node.span,
      })
      return nil, true, actions
    else
      return new_node, true, actions
    end
  end
end

function inline_tasks.stat_if(node)
  local inlined_any = false
  local actions = terralib.newlist()
  local new_cond, inlined, cond_actions = inline_tasks.expr(node.cond)
  if inlined then
    assert(new_cond ~= nil)
    inlined_any = true
    actions:insertall(cond_actions)
  end
  local new_elseif_blocks = node.elseif_blocks:map(function(elseif_block)
    local new_cond, inlined, cond_actions =
      inline_tasks.expr(elseif_block.cond)
    if inlined then
      inlined_any = true
      actions:insertall(cond_actions)
      return elseif_block { cond = new_cond }
    else
      return elseif_block
    end
  end)
  if inlined_any then
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node {
      cond = new_cond,
      elseif_blocks = new_elseif_blocks,
    })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_while(node)
  local value, inlined, actions = inline_tasks.expr(node.cond)
  if inlined then
    assert(value ~= nil)
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node { cond = value })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_fornum(node)
  local inlined_any = false
  local actions = terralib.newlist()
  local values = node.values:map(function(value)
    local new_value, inlined, value_actions = inline_tasks.expr(value)
    if inlined then
      assert(new_value ~= nil)
      inlined_any = true
      actions:insertall(value_actions)
      return new_value
    else
      return value
    end
  end)
  if inlined_any then
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node { values = values })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_repeat(node)
  local new_cond, inlined, actions = inline_tasks.expr(node.until_cond)
  if inlined then
    assert(new_cond ~= nil)
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node { until_cond = new_cond })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_var(node)
  if not node.values then return node end
  local value, inlined, actions = inline_tasks.expr(node.values)
  assert(not inlined or value ~= nil)
  if inlined then
    local stats = terralib.newlist { preserve_task_call(node {
        symbols = std.newsymbol(nil, node.symbols:hasname()),
      })
    }
    stats:insertall(actions)
    stats:insert(node { values = value })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_forlist_or_varunpack(node)
  local value, inlined, actions = inline_tasks.expr(node.value)
  assert(not inlined or value ~= nil)
  if inlined then
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node { value = value })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_return(node)
  if not node.value then return node end
  local value, inlined, actions = inline_tasks.expr(node.value)
  assert(not inlined or value ~= nil)
  if inlined then
    local stats = terralib.newlist { preserve_task_call(ast.specialized.stat.Expr {
        expr = node.value,
        annotations = ast.default_annotations(),
        span = node.span,
      })
    }
    stats:insertall(actions)
    stats:insert(node { value = value })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_assignment_or_reduce(node)
  local inlined_any = false
  local actions = terralib.newlist()
  local new_lhs, inlined, lhs_actions = inline_tasks.expr(node.lhs)
  if inlined then
    assert(new_lhs ~= nil)
    inlined_any = true
    actions:insertall(lhs_actions)
  end
  local new_rhs, inlined, rhs_actions = inline_tasks.expr(node.rhs)
  if inlined then
    assert(new_rhs ~= nil)
    inlined_any = true
    actions:insertall(rhs_actions)
  end
  if inlined_any then
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node {
      lhs = new_lhs,
      rhs = new_rhs,
    })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_expr(node)
  local value, inlined, actions = inline_tasks.expr(node.expr, true)
  if inlined then
    assert(value == nil)
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    return stats
  else
    return node
  end
end

local dispatch_table = {
  [ast.specialized.stat.If] = inline_tasks.stat_if,
  [ast.specialized.stat.While] = inline_tasks.stat_while,
  [ast.specialized.stat.ForNum] = inline_tasks.stat_fornum,
  [ast.specialized.stat.Repeat] = inline_tasks.stat_repeat,
  [ast.specialized.stat.Var] = inline_tasks.stat_var,
  [ast.specialized.stat.ForList] = inline_tasks.stat_forlist_or_varunpack,
  [ast.specialized.stat.VarUnpack] = inline_tasks.stat_forlist_or_varunpack,
  [ast.specialized.stat.Return] = inline_tasks.stat_return,
  [ast.specialized.stat.Assignment] = inline_tasks.stat_assignment_or_reduce,
  [ast.specialized.stat.Reduce] = inline_tasks.stat_assignment_or_reduce,
  [ast.specialized.stat.Expr] = inline_tasks.stat_expr,
}

function inline_tasks.stat(node)
  if dispatch_table[node.node_type] ~= nil then
    return dispatch_table[node.node_type](node)
  else
    return node
  end
end

function inline_tasks.top_task(node)
  return ast.flatmap_node_postorder(inline_tasks.stat, node)
end

function inline_tasks.top(node)
  if node:is(ast.specialized.top.Task) then
    if node.annotations.inline:is(ast.annotation.Demand) then
      check_valid_inline_task(node, node)
      node.prototype:set_is_inline(true)
    end
    local new_node = inline_tasks.top_task(node)
    if new_node.prototype:has_primary_variant() and
       node.annotations.inline:is(ast.annotation.Demand)
    then
      new_node.prototype:get_primary_variant():set_untyped_ast(new_node)
    end
    return new_node
  elseif node:is(ast.specialized.top.Fspace) or
         node:is(ast.specialized.top.QuoteExpr) or
         node:is(ast.specialized.top.QuoteStat) then
    return node
  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function inline_tasks.entry(node)
  return inline_tasks.top(node)
end

inline_tasks.pass_name = "inline_tasks"

return inline_tasks
