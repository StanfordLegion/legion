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
  -- Reject any access to a field id or a physical region of a region argument
  -- if it uses a field path that is not locally fully qualified. This is to
  -- prevent the access from changing its meaning in a caller context.
  local check_field_paths_qualified
  local function check_field_path_qualified(name, field, type)
    check_field_paths_qualified(name, field, field.fields, std.get_field(type, field.field_name))
  end
  function check_field_paths_qualified(name, node, fields, type)
    if not fields then
      if not (type:isprimitive() or type:ispointer() or type.__no_field_slicing or std.is_bounded_type(type)) then
        report.error(node, "ambiguous field access in " .. name ..
            ": every field path in an inline task must be fully specified.")
      end
    else
      fields:map(function(field) check_field_path_qualified(name, field, type) end)
    end
  end
  local function check_region_qualified(node)
    if node:is(ast.specialized.expr.RawFields) or node:is(ast.specialized.expr.RawPhysical) then
      local region_node = node.region
      local region_symbol = region_node.region.value
      -- If the region symbol does not have a type, that means the region is created
      -- within this inline task, so the access will has the same semantics in any context.
      if not region_symbol:hastype() then return false end
      local name = (node:is(ast.specialized.expr.RawFields) and "__fields") or
                   (node:is(ast.specialized.expr.RawPhysical) and "__physcal")
      check_field_paths_qualified(name, region_node, region_node.fields,
                                  region_symbol:hastype():fspace())
    end
  end
  ast.traverse_node_postorder(check_region_qualified, body)
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
}

local substitute_expr = ast.make_single_dispatch(
  substitute_expr_table,
  {},
  pass_through)

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
  local symbol = node.symbol
  local new_symbol = std.newsymbol(nil, symbol:hasname())
  cx.symbol_mapping[symbol] = new_symbol
  local values = node.values:map(function(value)
    return substitute.expr(cx, value)
  end)
  local block = substitute.block(cx, node.block)
  return node {
    symbol = new_symbol,
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
}

local substitute_stat = ast.make_single_dispatch(
  substitute_stat_table,
  {},
  pass_through)

function substitute.stat(cx, node)
  return substitute_stat(cx)(node)
end

function substitute.block(cx, node)
  return node {
    stats = node.stats:map(function(stat) return substitute.stat(cx, stat) end),
  }
end

local find_lvalues = {}

local function find_lvalues_expr_address_of(cx, node)
  if node.value:is(ast.specialized.expr.ID) then
    cx.lvalues[node.value.value] = true
  end
end

local find_lvalues_expr_table = {
  [ast.specialized.expr.AddressOf] = find_lvalues_expr_address_of,
}

local find_lvalues_expr = ast.make_single_dispatch(
  find_lvalues_expr_table,
  {},
  pass_through)

function find_lvalues.expr(cx, node)
  return ast.map_node_postorder(find_lvalues_expr(cx), node)
end

local function find_lvalues_stat_if(cx, node)
  find_lvalues.block(cx, node.then_block)
  find_lvalues.block(cx, node.else_block)
end

local function find_lvalues_stat_block(cx, node)
  find_lvalues.block(cx, node.block)
end

local function find_lvalues_stat_var(cx, node)
  if node.values then find_lvalues.expr(cx, node.values) end
end

local function find_lvalues_lhs(cx, expr)
  if expr:is(ast.specialized.expr.ID) then
    cx.lvalues[expr.value] = true
  elseif expr:is(ast.specialized.expr.FieldAccess) then
    find_lvalues_lhs(cx, expr.value)
  elseif expr:is(ast.specialized.expr.IndexAccess) then
    find_lvalues_lhs(cx, expr.value)
  end
end

local function find_lvalues_stat_assignment_or_reduce(cx, node)
  find_lvalues_lhs(cx, node.lhs)
  find_lvalues.expr(cx, node.rhs)
end

local find_lvalues_stat_table = {
  [ast.specialized.stat.If]              = find_lvalues_stat_if,
  [ast.specialized.stat.Elseif]          = unreachable,
  [ast.specialized.stat.While]           = find_lvalues_stat_block,
  [ast.specialized.stat.ForNum]          = find_lvalues_stat_block,
  [ast.specialized.stat.ForList]         = find_lvalues_stat_block,
  [ast.specialized.stat.Repeat]          = find_lvalues_stat_block,
  [ast.specialized.stat.MustEpoch]       = find_lvalues_stat_block,
  [ast.specialized.stat.Block]           = find_lvalues_stat_block,
  [ast.specialized.stat.Var]             = find_lvalues_stat_var,
  [ast.specialized.stat.Assignment]      = find_lvalues_stat_assignment_or_reduce,
  [ast.specialized.stat.Reduce]          = find_lvalues_stat_assignment_or_reduce,
}

local find_lvalues_stat = ast.make_single_dispatch(
  find_lvalues_stat_table,
  {},
  pass_through)

function find_lvalues.stat(cx, node)
  find_lvalues_stat(cx)(node)
end

function find_lvalues.block(cx, node)
  node.stats:map(function(stat) find_lvalues.stat(cx, stat) end)
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
      stats = terralib.newlist {
        ast.specialized.stat.Expr {
          expr = node,
          annotations = ast.default_annotations(),
          span = node.span,
        }
      },
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

function inline_tasks.expr_call(stats, call)
  if not call:is(ast.specialized.expr.Call) then return call end

  -- Task T is inlined at call site C only if T demands inlining and
  -- C does not forbid inlining.

  local task = call.fn.value
  if not std.is_task(task) then return call end

  if call.annotations.inline:is(ast.annotation.Forbid) then
    if std.is_task(task) and task.is_inline then task:optimize() end
    return call
  end

  local task_ast = task:has_primary_variant() and
                   task:get_primary_variant():has_untyped_ast()
  if not task_ast then
    if call.annotations.inline:is(ast.annotation.Demand) then
      report.error(call, "cannot inline a task that does not demand inlining")
    end
    return call
  end

  local args = call.args

  -- Preserve the original call expression for type checking
  stats:insert(preserve_task_call(call))

  -- Find parameters used as l-values, as they cannot be directly replaced with arguments
  local params = task_ast.params:map(function(param) return param.symbol end)
  local lvalues = data.newmap()
  -- Initially, we assume no parameter is used as an l-value
  params:map(function(param) lvalues[param] = false end)
  find_lvalues.block({ lvalues = lvalues }, task_ast.body)

  -- Make assignments to temporary variables.
  local param_types = params:map(function(param) return param:gettype() end)
  local symbol_mapping = {}
  local expr_mapping = {}
  data.zip(params, param_types, args):map(function(tuple)
    local param, param_type, arg = unpack(tuple)
    if arg:is(ast.specialized.expr.ID) and
       ((not lvalues[param] and not param_type:ispointer()) or
        is_singleton_type(param_type))
    then
      symbol_mapping[param] = arg.value
      if not (is_singleton_type(param_type) or
              std.is_fspace_instance(param_type))
      then
        arg = ast.specialized.expr.Cast {
          fn = ast.specialized.expr.Function {
            value = std.type_sub(param_type, symbol_mapping),
            span = arg.span,
            annotations = arg.annotations,
          },
          args = terralib.newlist({arg}),
          span = arg.span,
          annotations = arg.annotations,
        }
      end
      expr_mapping[param] = arg
    else
      local symbol_type = nil
      if not (is_singleton_type(param_type) or
              std.is_fspace_instance(param_type))
      then
        symbol_type = param_type
      end
      local new_symbol = std.newsymbol(symbol_type, param:hasname())
      symbol_mapping[param] = new_symbol
      stats:insert(ast.specialized.stat.Var {
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

  local body_stats = substitute.block(cx, task_ast.body).stats

  -- Convert any return statement to an assignment to a temporary variable
  local return_var_expr = nil
  if #body_stats > 0 and body_stats[#body_stats]:is(ast.specialized.stat.Return) then
    local task_name = task_ast.name:mkstring("", "_", "")
    local return_var = std.newsymbol(nil, "__" .. task_name .."_ret")
    return_var_expr = ast.specialized.expr.ID {
      value = return_var,
      annotations = ast.default_annotations(),
      span = call.span,
    }
    local return_stat = body_stats[#body_stats]
    body_stats[#body_stats] = ast.specialized.stat.Var {
      symbols = return_var,
      values = return_stat.value,
      annotations = ast.default_annotations(),
      span = return_stat.span,
    }
  end
  stats:insertall(body_stats)

  return return_var_expr
end

function inline_tasks.expr(stats, node, expects_value)
  local expr = inline_tasks.expr_call(stats, node)
  if expects_value and expr == nil then
    report.error(node, "cannot inline a task with no return value in a place where a value is expected")
  end
  return expr
end

function inline_tasks.stat_if(stats, node)
  local then_block = inline_tasks.block(node.then_block)
  local else_block = inline_tasks.block(node.else_block)
  stats:insert(node {
    then_block = then_block,
    else_block = else_block,
  })
end

function inline_tasks.stat_var(stats, node)
  if not node.values then
    stats:insert(node)
  else
    local value = inline_tasks.expr(stats, node.values, true)
    stats:insert(node { values = value })
  end
end

function inline_tasks.stat_expr(stats, node)
  local expr = inline_tasks.expr(stats, node.expr, false)
  if expr ~= nil then
    stats:insert(node { expr = expr })
  end
end

function inline_tasks.stat_block(stats, node)
  stats:insert(node { block = inline_tasks.block(node.block) })
end

function inline_tasks.stat_assignment_or_reduce(stats, node)
  -- Only the RHS can have a task launch
  local rhs = inline_tasks.expr(stats, node.rhs, true)
  stats:insert(node { rhs = rhs })
end

local function pass_through_stat(stats, node) stats:insert(node) end

local inline_tasks_stat_table = {
  [ast.specialized.stat.If]              = inline_tasks.stat_if,
  [ast.specialized.stat.Var]             = inline_tasks.stat_var,
  [ast.specialized.stat.Expr]            = inline_tasks.stat_expr,

  [ast.specialized.stat.While]           = inline_tasks.stat_block,
  [ast.specialized.stat.ForNum]          = inline_tasks.stat_block,
  [ast.specialized.stat.ForList]         = inline_tasks.stat_block,
  [ast.specialized.stat.Repeat]          = inline_tasks.stat_block,
  [ast.specialized.stat.MustEpoch]       = inline_tasks.stat_block,
  [ast.specialized.stat.Block]           = inline_tasks.stat_block,
  [ast.specialized.stat.ParallelizeWith] = inline_tasks.stat_block,

  [ast.specialized.stat.Assignment]      = inline_tasks.stat_assignment_or_reduce,
  [ast.specialized.stat.Reduce]          = inline_tasks.stat_assignment_or_reduce,

  [ast.specialized.stat.Return]          = pass_through_stat,
  [ast.specialized.stat.VarUnpack]       = pass_through_stat,
}

local inline_tasks_stat = ast.make_single_dispatch(
  inline_tasks_stat_table,
  {},
  pass_through_stat)

function inline_tasks.stat(stats, node)
  inline_tasks_stat(stats)(node)
end

function inline_tasks.block(node)
  local stats = terralib.newlist()
  node.stats:map(function(stat) inline_tasks.stat(stats, stat) end)
  return node { stats = stats }
end

function inline_tasks.top_task(node)
  local body = node.body and inline_tasks.block(node.body) or false
  return node { body = body }
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
