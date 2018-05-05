-- Copyright 2018 Stanford University
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
    local new_arg, inlined, arg_actions = inline_tasks.expr(arg, true)
    assert(new_arg ~= nil)
    if not inlined then return arg
    else
      actions:insertall(arg_actions)
      return new_arg
    end
  end)

  local function is_singleton_type(type)
    return std.is_ispace(type) or std.is_region(type) or
           std.is_list_of_regions(type) or
           std.is_partition(type) or std.is_cross_product(type)
  end
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
        symbols = terralib.newlist { new_symbol },
        values = terralib.newlist { arg },
        annotations = ast.default_annotations(),
        span = call.span,
      })
    end
  end)

  -- Do alpha conversion to avoid type collision.
  local stats = ast.map_node_postorder(function(node)
    if node:is(ast.specialized.stat.Var) then
      assert(#node.symbols == 1)
      local symbol = node.symbols[1]
      -- We should ignore the type of the existing symbol as we want to type check it again.
      local symbol_type = symbol:hastype()
      if #node.values > 0 or is_singleton_type(symbol_type) then symbol_type = nil
      else symbol_type = std.type_sub(symbol_type, symbol_mapping) end
      local new_symbol = std.newsymbol(symbol_type, symbol:hasname())
      symbol_mapping[symbol] = new_symbol
      return node { symbols = terralib.newlist { new_symbol } }
    elseif node:is(ast.specialized.stat.ForList) then
      local symbol = node.symbol
      local symbol_type = symbol:hastype()
      local new_symbol = std.newsymbol(nil, symbol:hasname())
      symbol_mapping[symbol] = new_symbol
      return node { symbol = new_symbol }
    else
      return node
    end
  end, task_ast.body.stats)

  -- Third, replace all occurrences of parameters in the task body with the new temporary variables.
  stats = ast.map_node_postorder(function(node)
    if node:is(ast.specialized.expr.ID) then
      local new_node = expr_mapping[node.value] or
                       (symbol_mapping[node.value] ~= nil and
                        node { value = symbol_mapping[node.value] }) or
                       node
      return new_node
    elseif node:is(ast.specialized.expr.DynamicCast) or
           node:is(ast.specialized.expr.StaticCast) or
           node:is(ast.specialized.expr.UnsafeCast)
    then
      return node { expr_type = std.type_sub(node.expr_type, symbol_mapping) }
    elseif node:is(ast.specialized.expr.Null) then
      return node { pointer_type = std.type_sub(node.pointer_type, symbol_mapping) }
    elseif node:is(ast.specialized.expr.Region) then
      return node { fspace_type = std.type_sub(node.fspace_type, symbol_mapping) }
    else
      return node
    end
  end, stats)

  -- Finally, convert any return statement to an assignment
  local return_type = task:get_type().returntype
  assert(return_type ~= nil)
  return_type = std.type_sub(return_type, symbol_mapping)
  local task_name = task_ast.name:mkstring("", "_", "")
  local return_var = std.newsymbol(return_type, "__" .. task_name .."_ret")

  local return_var_expr = ast.specialized.expr.ID {
    value = return_var,
    annotations = ast.default_annotations(),
    span = call.span,
  }
  local return_var_decl = ast.specialized.stat.Var {
    symbols = terralib.newlist { return_var },
    values = terralib.newlist(),
    annotations = ast.default_annotations(),
    span = call.span,
  }
  if #stats > 0 and stats[#stats]:is(ast.specialized.stat.Return) then
    local return_stat = stats[#stats]
    stats[#stats] = ast.specialized.stat.Assignment {
      lhs = terralib.newlist { return_var_expr },
      rhs = terralib.newlist { return_stat.value },
      annotations = ast.default_annotations(),
      span = return_stat.span,
    }
  end

  actions:insertall(stats)
  local new_block = ast.specialized.stat.Block {
    block = ast.specialized.Block {
      stats = actions,
      span = call.span,
    },
    annotations = ast.default_annotations(),
    span = call.span,
  }
  actions = terralib.newlist()
  actions:insert(return_var_decl)
  actions:insert(new_block)
  return return_var_expr, true, actions
end

function inline_tasks.expr(node)
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
    return new_node, true, actions
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
  if #node.values == 0 then return node end
  assert(#node.symbols == 1 and #node.values == 1)
  local value, inlined, actions = inline_tasks.expr(node.values[1])
  assert(not inlined or value ~= nil)
  if inlined then
    local stats = terralib.newlist { preserve_task_call(node {
        symbols = terralib.newlist {
          std.newsymbol(nil, node.symbols[1]:hasname())
        },
      })
    }
    stats:insertall(actions)
    stats:insert(node { values = terralib.newlist { value } })
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
  local new_rhs = node.rhs:map(function(rh)
    local new_rh, inlined, rh_actions = inline_tasks.expr(rh)
    if inlined then
      assert(new_rh ~= nil)
      inlined_any = true
      actions:insertall(rh_actions)
      return new_rh
    else
      return rh
    end
  end)
  if inlined_any then
    local stats = terralib.newlist { preserve_task_call(node) }
    stats:insertall(actions)
    stats:insert(node { rhs = new_rhs })
    return stats
  else
    return node
  end
end

function inline_tasks.stat_expr(node)
  local value, inlined, actions = inline_tasks.expr(node.expr)
  if inlined then
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
