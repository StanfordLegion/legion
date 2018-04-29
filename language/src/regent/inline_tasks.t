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

local context = {}

function context:__index (field)
  local value = context [field]
  if value ~= nil then
    return value
  end
  error ("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex (field, value)
  error ("context has no field '" .. field .. "' (in assignment)", 2)
end

function context:new_local_scope()
  local cx = {
    env = self.env:new_local_scope(),
    constraints = self.constraints,
  }
  setmetatable(cx, context)
  return cx
end

function context:new_global_scope(env, constraints)
  local cx = {
    env = symbol_table.new_global_scope(env),
    constraints = constraints,
  }
  setmetatable(cx, context)
  return cx
end

local function expr_id(sym, node)
  return ast.typed.expr.ID {
    value = sym,
    expr_type = std.rawref(&sym:gettype()),
    annotations = node.annotations,
    span = node.span,
  }
end

local function make_block(stats, annotations, span)
  return ast.typed.stat.Block {
    block = ast.typed.Block {
      stats = stats,
      span = span,
    },
    annotations = annotations,
    span = span,
  }
end

local function stat_var(lhs, rhs, node)
  local value = false
  if rhs then value = rhs end
  return ast.typed.stat.Var {
    symbol = lhs,
    type = lhs:gettype(),
    value = value,
    annotations = node.annotations,
    span = node.span,
  }
end

local function stat_asgn(lhs, rhs, node)
  return ast.typed.stat.Assignment {
    lhs = lhs,
    rhs = rhs,
    annotations = node.annotations,
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

local function check_valid_inline_task(node, task)
  if not task then
    report.error(node, "cannot inline a task that has no definition")
  end

  local body = task.body
  assert(body)

  local num_returns = count_returns(body)
  if num_returns > 1 then
    report.error(task, "inline tasks cannot have multiple return statements")
  end
  if num_returns == 1 and not body.stats[#body.stats]:is(ast.typed.stat.Return) then
    report.error(task, "the return statement in an inline task should be the last statement")
  end

  if find_self_recursion(task.prototype, body) then
    report.error(task, "inline tasks cannot be recursive")
  end
end

local function check_rf_type(ty)
  if std.is_ispace(ty) or std.is_region(ty) or
     std.is_partition(ty) or std.is_cross_product(ty) then
     return true
  elseif std.is_fspace_instance(ty) then
    for _, field in ipairs(ty:getentries()) do
      if not check_rf_type(field.type) then
        return false
      end
      return true
    end
  else
    return false
  end
end

local function check_rf(node)
  if node:is(ast.typed.expr.ID) then
    return check_rf_type(std.as_read(node.expr_type))
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
    local task_ast = task:has_primary_variant() and task:get_primary_variant():has_ast()
    if node.annotations.inline:is(ast.annotation.Demand) then
      check_valid_inline_task(node, task_ast)
    elseif not task_ast or
      not task_ast.annotations.inline:is(ast.annotation.Demand) or
      node.annotations.inline:is(ast.annotation.Forbid)
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
    local return_var
    local return_var_expr

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
          elseif std.is_partition(param_type) then
            type_mapping[param_type] = std.as_read(arg.expr_type)
            type_mapping[param_type:colors()] =
              type_mapping[param_type]:colors()
            type_mapping[param_type.colors_symbol] =
              type_mapping[param_type].colors_symbol
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
        for i = 1, #new_local_params do
          stats:insert(ast.typed.stat.Var {
            symbol = new_local_params[i],
            type = new_local_param_types[i],
            value = new_args[i],
            annotations = node.annotations,
            span = node.span
          })
        end
      end
      local function subst_pre(node)
        if node:is(ast.typed.stat.ForList) then
          local old_type = node.symbol:gettype()
          local new_type = std.type_sub(old_type, type_mapping)

          local new_symbol = std.newsymbol(new_type, node.symbol:hasname())
          expr_mapping[node.symbol] = new_symbol
          return node { symbol = new_symbol }
        elseif node:is(ast.typed.expr.IndexAccess) then
          local expr_type = std.as_read(node.expr_type)
          if std.is_region(expr_type) then
            local partition_type = std.as_read(node.value.expr_type)
            assert(std.is_partition(partition_type))
            local new_partition_type = type_mapping[partition_type]
            if new_partition_type == nil then
              new_partition_type = std.type_sub(partition_type, type_mapping)
              type_mapping[partition_type] = new_partition_type
            end
            local new_subregion_type = type_mapping[expr_type]
            if new_subregion_type == nil then
              new_subregion_type = new_partition_type:subregion_dynamic()
              type_mapping[expr_type] = new_subregion_type
            end
            local parent_region_type = new_partition_type:parent_region()
            std.add_constraint(cx, new_partition_type, parent_region_type, std.subregion, false)
            std.add_constraint(cx, new_subregion_type, new_partition_type, std.subregion, false)
          end
          return node
        else
          return node
        end
      end

      local function subst_post(node)
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
          return node {
            expr_type = type_mapping[node.expr_type] or
                        std.type_sub(node.expr_type, type_mapping)
          }
        elseif node:is(ast.typed.stat.Var) then
          local new_symbol = nil
          local new_type = type_mapping[node.type] or std.type_sub(node.type, type_mapping)
          if not std.type_eq(new_type, node.type) then
            new_symbol = std.newsymbol(new_type)
            expr_mapping[node.symbol] = new_symbol
            if std.is_region(new_type) then
              type_mapping[node.symbol] = new_symbol
            end
          else
            new_symbol = node.symbol
            new_type = node.type
          end
          return node {
            symbol = new_symbol,
            type = new_type,
          }
        else
          return node
        end
      end
      return_var = std.newsymbol(
          type_mapping[task_ast.return_type] or
          std.type_sub(task_ast.return_type, type_mapping))
      return_var_expr = expr_id(return_var, node)
      stats:insertall(task_body.stats)
      if stats[#stats]:is(ast.typed.stat.Return) then
        local num_stats = #stats
        local return_stat = stats[num_stats]
        stats[num_stats] = stat_asgn(return_var_expr, return_stat.value, return_stat)
      end
      stats = ast.map_node_prepostorder(subst_pre, subst_post, stats)
      new_block = make_block(stats, node.annotations, node.span)
    end
    stats:insert(stat_var(return_var, nil, node))
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

        elseif terralib.islist(field) then
          fields[k] = field:map(function(field)
            if ast.is_node(field) and field:is(ast.typed.expr) then
              local new_stats, new_node = inline_tasks.expr(cx, field)
              stats:insertall(new_stats)
              return new_node
            else
              return field
            end
          end)
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
    body = node.body and inline_tasks.block(cx, node.body)
  }
end

function inline_tasks.top(node)
  if node:is(ast.typed.top.Task) then
    local cx = context:new_global_scope({}, node.prototype:get_constraints())
    if node.annotations.inline:is(ast.annotation.Demand) then
      check_valid_inline_task(node, node)
    end
    local new_node = inline_tasks.top_task(cx, node)
    if new_node.prototype:has_primary_variant() then
      new_node.prototype:get_primary_variant():set_ast(new_node)
    end
    return new_node

  elseif node:is(ast.typed.top.Fspace) then
    return node

  elseif node:is(ast.specialized.top.QuoteExpr) then
    return node

  elseif node:is(ast.specialized.top.QuoteStat) then
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
