-- Copyright 2016 Stanford University
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
local data = require("regent/data")
local std = require("regent/std")
local log = require("regent/log")
local symbol_table = require("regent/symbol_table")

local inline_tasks = {}

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    env = self.env:new_local_scope(),
  }
  setmetatable(cx, context)
  return cx
end

function context:new_global_scope(env)
  local cx = {
    env = symbol_table.new_global_scope(env),
  }
  setmetatable(cx, context)
  return cx
end

local global_env = context:new_global_scope({})

local function expr_id(sym, node)
  return ast.typed.expr.ID {
    value = sym,
    expr_type = std.rawref(&sym:gettype()),
    options = node.options,
    span = node.span,
  }
end

local function make_block(stats, options, span)
  return ast.typed.stat.Block {
    block = ast.typed.Block {
      stats = stats,
      span = span,
    },
    options = options,
    span = span,
  }
end

local function stat_var(lhs, rhs, node)
  local symbols = terralib.newlist()
  local types = terralib.newlist()
  local values = terralib.newlist()

  symbols:insert(lhs)
  types:insert(lhs:gettype())
  if rhs then values:insert(rhs) end
  return ast.typed.stat.Var {
    symbols = symbols,
    types = types,
    values = values,
    options = node.options,
    span = node.span,
  }
end

local function stat_asgn(lh, rh, node)
  local lhs = terralib.newlist()
  local rhs = terralib.newlist()

  lhs:insert(lh)
  rhs:insert(rh)
  return ast.typed.stat.Assignment {
    lhs = lhs,
    rhs = rhs,
    options = node.options,
    span = node.span,
  }
end

local function count_returns(node)
  local num_returns = 0

  if node:is(ast.typed.stat.Return) then num_returns = 1 end

  for _, field in pairs(node) do
    if ast.is_node(field) and field:is(ast.typed.stat) then
      num_returns = num_returns + count_returns(field)

    elseif ast.is_node(field) and field:is(ast.typed.Block) then
      field.stats:map(function(node)
        num_returns = num_returns + count_returns(node)
      end)

    elseif terralib.islist(field) then
      field:map(function(field)
        if ast.is_node(field) and field:is(ast.typed.stat) then
          num_returns = num_returns + count_returns(field)
        end
      end)
    end
  end

  return num_returns
end

local function find_self_recursion(prototype, node)
  if node:is(ast.typed.expr.Call) and node.fn.value == prototype then
    return true
  end

  for _, field in pairs(node) do
    if ast.is_node(field) and
      (field:is(ast.typed.stat) or field:is(ast.typed.expr))
    then
      if find_self_recursion(prototype, field) then return true end

    elseif ast.is_node(field) and field:is(ast.typed.Block) then
      local recursion_found = false
      field.stats:map(function(field)
        recursion_found = recursion_found or find_self_recursion(prototype, field)
      end)
      if recursion_found then return true end

    elseif terralib.islist(field) then
      local recursion_found = false
      field:map(function(field)
        if ast.is_node(field) and
          (field:is(ast.typed.stat) or field:is(ast.typed.expr))
        then
          recursion_found = recursion_found or find_self_recursion(prototype, field)
        end
      end)
      if recursion_found then return true end
    end
  end

  return false
end

local function check_valid_inline_task(task)
  local body = task.body
  local num_returns = count_returns(body)
  if num_returns > 1 then
    log.error(task, "inline tasks cannot have multiple return statements")
  end
  if num_returns == 1 and not body.stats[#body.stats]:is(ast.typed.stat.Return) then
    log.error(task, "the return statement in an inline task should be the last statement")
  end

  if find_self_recursion(task.prototype, body) then
    log.error(task, "inline tasks cannot be recursive")
  end
end

local function check_rf(node)
  if node:is(ast.typed.expr.ID) then return true
  elseif node:is(ast.typed.expr.Constant) then return true
  elseif node:is(ast.typed.expr.FieldAccess) then
    return check_rf(node.value)
  elseif node:is(ast.typed.expr.IndexAccess) then
    return check_rf(node.value) and check_rf(node.index)
  else
    return false
  end
end

local function get_root_subregion(node)
  if node:is(ast.typed.expr.ID) then
    local ty = std.as_read(node.expr_type)
    if std.is_region(ty) then return node.value
    elseif std.is_partition(ty) then return ty.parent_region_symbol
    else return nil end
  elseif node:is(ast.typed.expr.IndexAccess) then
    return get_root_subregion(node.value)
  end
  return nil
end

function inline_tasks.expr(cx, node)
  if node:is(ast.typed.expr.Call) then
    local stats = terralib.newlist()
    if type(node.fn.value) ~= "table" or not std.is_task(node.fn.value) then
      return stats, node
    end

    local task = node.fn.value
    local task_ast = task:hasast()
    if node.options.inline:is(ast.options.Demand) then
      check_valid_inline_task(task_ast)
    elseif not task_ast or
      not task_ast.options.inline:is(ast.options.Demand) or
      node.options.inline:is(ast.options.Forbid)
    then
      return stats, node
    end

    local args = node.args:map(function(arg)
      local new_stats, new_node = inline_tasks.expr(cx, arg)
      stats:insertall(new_stats)
      return new_node
    end)
    local params = task_ast.params:map(function(param) return param.symbol end)
    local param_types = params:map(function(param) return param:gettype() end)
    local task_body = task_ast.body
    local return_var = std.newsymbol(task_ast.return_type)
    local return_var_expr = expr_id(return_var, node)

    stats:insert(stat_var(return_var, nil, node))
    local new_block
    do
      local stats = terralib.newlist()
      local expr_mapping = {}
      local type_mapping = {}
      local new_local_params = terralib.newlist()
      local new_local_param_types = terralib.newlist()
      local new_args = terralib.newlist()
      data.zip(params, param_types, args):map(function(tuple)
        local param, param_type, arg = unpack(tuple)
        if check_rf(arg) then
          expr_mapping[param] = arg
          if std.is_region(param_type) then
            type_mapping[param_type] = std.as_read(arg.expr_type)
            type_mapping[param] = get_root_subregion(arg)
            type_mapping[param_type.ispace_symbol] =
              type_mapping[param_type].ispace_symbol
            type_mapping[param_type:ispace()] =
              type_mapping[param_type]:ispace()
          end
        else
          local new_var = std.newsymbol(std.as_read(arg.expr_type))
          expr_mapping[param] = new_var
          new_local_params:insert(new_var)
          new_local_param_types:insert(new_var:gettype())
          new_args:insert(arg)
          if std.is_region(param_type) then
            type_mapping[param] = new_var
            type_mapping[param_type] = new_var:gettype()
            type_mapping[param_type.ispace_symbol] =
              type_mapping[param_type].ispace_symbol
            type_mapping[param_type:ispace()] =
              type_mapping[param_type]:ispace()
          end
        end
      end)

      if #new_local_params > 0 then
        stats:insert(ast.typed.stat.Var {
          symbols = new_local_params,
          types = new_local_param_types,
          values = new_args,
          options = node.options,
          span = node.span
        })
      end
      local function subst(node)
        if rawget(node, "expr_type") then
          if node:is(ast.typed.expr.ID) and expr_mapping[node.value] then
            local tgt = expr_mapping[node.value]
            if rawget(tgt, "expr_type") then node = tgt
            else node = node { value = tgt } end
          elseif node:is(ast.typed.expr.Ispace) then
            local ispace = std.as_read(node.expr_type)
            local new_ispace = std.ispace(ispace.index_type)
            type_mapping[ispace] = new_ispace
          elseif node:is(ast.typed.expr.Region) then
            local region = std.as_read(node.expr_type)
            local ispace =
              std.type_sub(std.as_read(region:ispace()), type_mapping)
            local fspace =
              std.type_sub(std.as_read(region:fspace()), type_mapping)
            local new_region = std.region(ispace, fspace)
            type_mapping[region] = new_region
          end
          return node { expr_type = std.type_sub(node.expr_type, type_mapping) }
        elseif node:is(ast.typed.stat.Var) then
          local new_symbols = terralib.newlist()
          local new_types = terralib.newlist()

          for i = 1, #node.symbols do
            local new_ty = std.type_sub(node.types[i], type_mapping)
            if new_ty ~= node.types[i] then
              local sym = node.symbols[i]
              local new_sym = std.newsymbol(new_ty)
              new_symbols:insert(new_sym)
              new_types:insert(new_ty)
              if std.is_region(new_ty) then
                expr_mapping[sym] = new_sym
                type_mapping[sym] = new_sym
              end
            else
              new_symbols:insert(node.symbols[i])
              new_types:insert(node.types[i])
            end
          end
          return node {
            symbols = new_symbols,
            types = new_types,
          }
        else
          return node
        end
      end
      stats:insertall(task_body.stats)
      if stats[#stats]:is(ast.typed.stat.Return) then
        local num_stats = #stats
        local return_stat = stats[num_stats]
        stats[num_stats] = stat_asgn(return_var_expr, return_stat.value, return_stat)
      end
      stats = ast.map_node_postorder(subst, stats)
      new_block = make_block(stats, node.options, node.span)
    end
    stats:insert(new_block)

    return stats, return_var_expr

  else
    local stats = terralib.newlist()
    local fields = {}

    for k, field in pairs(node) do
      if k ~= "node_type" then
        if ast.is_node(field) and field:is(ast.typed.expr) then
          local new_stats, new_node = inline_tasks.expr(cx, field)
          stats:insertall(new_stats)
          fields[k] = new_node

        else
          fields[k] = field
        end
      end
    end

    return stats, node(fields)
  end
end

function inline_tasks.block(cx, node)
  local stats = terralib.newlist()
  node.stats:map(function(node)
    local new_stats, new_node = inline_tasks.stat(cx, node)
    stats:insertall(new_stats)
    stats:insert(new_node)
  end)
  return node { stats = stats }
end

function inline_tasks.stat(cx, node)
  local stats = terralib.newlist()
  local fields = {}

  for k, field in pairs(node) do
    if k ~= "node_type" then
      if ast.is_node(field) and field:is(ast.typed.stat) then
        local new_stats, new_node = inline_tasks.stat(cx, field)
        stats:insertall(new_stats)
        fields[k] = field

      elseif ast.is_node(field) and field:is(ast.typed.expr) then
        local new_stats, new_node = inline_tasks.expr(cx, field)
        stats:insertall(new_stats)
        fields[k] = new_node

      elseif ast.is_node(field) and field:is(ast.typed.Block) then
        local new_node = inline_tasks.block(cx, field)
        fields[k] = new_node

      elseif ast.is_node(field) and field:is(ast.location.Span) then
        fields[k] = field

      elseif terralib.islist(field) then
        fields[k] = field:map(function(field)
          if type(field) ~= "table" then
            return field
          else
            if ast.is_node(field) and field:is(ast.typed.stat) then
              local new_stats, new_node = inline_tasks.stat(cx, field)
              stats:insertall(new_stats)
              return new_node

            elseif ast.is_node(field) and field:is(ast.typed.expr) then
              local new_stats, new_node = inline_tasks.expr(cx, field)
              stats:insertall(new_stats)
              return new_node

            else
              return field
            end
          end
        end)

      else
        fields[k] = field
      end
    end
  end

  node = node(fields)
  return stats, node
end

function inline_tasks.top_task(cx, node)
  return node {
    body = inline_tasks.block(cx, node.body)
  }
end

function inline_tasks.top(cx, node)
  if node:is(ast.typed.top.Task) then
    if node.options.inline:is(ast.options.Demand) then
      check_valid_inline_task(node)
    end
    local new_node = inline_tasks.top_task(cx, node)
    new_node.prototype:setast(new_node)
    return new_node

  elseif node:is(ast.typed.top.Fspace) then
    return node

  elseif node:is(ast.typed.top.QuoteExpr) then
    return node

  elseif node:is(ast.typed.top.QuoteStat) then
    return node

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end

end

function inline_tasks.entry(node)
  local cx = context:new_global_scope({})
  return inline_tasks.top(cx, node)
end

return inline_tasks
