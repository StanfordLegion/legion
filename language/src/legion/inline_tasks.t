-- Copyright 2015 Stanford University
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

local ast = require("legion/ast")
local std = require("legion/std")
local log = require("legion/log")
local symbol_table = require("legion/symbol_table")

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
  return ast.typed.ExprID {
    value = sym,
    expr_type = std.rawref(&sym.type),
    span = node.span,
  }
end

local function make_block(stats, span)
  return ast.typed.StatBlock {
    block = ast.typed.Block {
      stats = stats,
      span = span,
    },
    span = span,
  }
end

local function stat_var(lhs, rhs, node)
  local symbols = terralib.newlist()
  local types = terralib.newlist()
  local values = terralib.newlist()

  symbols:insert(lhs)
  types:insert(lhs.type)
  if rhs then values:insert(rhs) end
  return ast.typed.StatVar {
    symbols = symbols,
    types = types,
    values = values,
    span = node.span,
  }
end

local function stat_asgn(lh, rh, node)
  local lhs = terralib.newlist()
  local rhs = terralib.newlist()

  lhs:insert(lh)
  rhs:insert(rh)
  return ast.typed.StatAssignment {
    lhs = lhs,
    rhs = rhs,
    span = node.span,
  }
end

local function count_returns(node)
  local num_returns = 0
  local ctor = rawget(node, "node_type")

  if node:is(ast.typed.StatReturn) then num_returns = 1 end

  for _, k in ipairs(ctor.expected_fields) do
    local field = node[k]
    if type(field) == "table" then
      local node_type = tostring(rawget(field, "node_type"))
      if node_type and string.find(node_type, ".Stat") then
        num_returns = num_returns + count_returns(field)

      elseif node_type and string.find(node_type, ".Block") then
        field.stats:map(function(node)
          num_returns = num_returns + count_returns(node)
        end)

      elseif terralib.islist(field) then
        field:map(function(field)
          if type(field) == "table" then
            local node_type = tostring(rawget(field, "node_type"))
            if node_type and string.find(node_type, ".Stat") then
              num_returns = num_returns + count_returns(field)
            end
          end
        end)
      end
    end
  end

  return num_returns
end

local function find_self_recursion(prototype, node)
  local ctor = rawget(node, "node_type")

  if node:is(ast.typed.ExprCall) and node.fn.value == prototype then
    return true
  end

  for _, k in ipairs(ctor.expected_fields) do
    local field = node[k]
    if type(field) == "table" then
      local node_type = tostring(rawget(field, "node_type"))
      if node_type and
        (string.find(node_type, ".Stat") or string.find(node_type, ".Expr")) then
        if find_self_recursion(prototype, field) then return true end

      elseif node_type and string.find(node_type, ".Block") then
        local recursion_found = false
        field.stats:map(function(field)
          recursion_found = recursion_found or find_self_recursion(prototype, field)
        end)
        if recursion_found then return true end

      elseif terralib.islist(field) then
        local recursion_found = false
        field:map(function(field)
          if type(field) == "table" then
            local node_type = tostring(rawget(field, "node_type"))
            if node_type and
              (string.find(node_type, ".Stat") or string.find(node_type, ".Expr"))  then
              recursion_found = recursion_found or find_self_recursion(prototype, field)
            end
          end
        end)
        if recursion_found then return true end
      end
    end
  end

  return false
end

local function check_valid_inline_task(task)
  --task.params:map(function(param)
  --  local ty = param.param_type
  --  if std.is_ispace(ty) or std.is_region(ty) or
  --    std.is_partition(ty) or std.is_cross_product(ty) or
  --    std.is_bounded_type(ty) or std.is_ref(ty) or std.is_rawref(ty) or
  --    std.is_future(ty) or std.is_unpack_result(ty) then
  --    log.error(param, "inline tasks cannot have a parameter of type " .. tostring(ty))
  --  end
  --end)

  local body = task.body
  local num_returns = count_returns(body)
  if num_returns > 1 then
    log.error(task, "inline tasks cannot have multiple return statements")
  end
  if num_returns == 1 and not body.stats[#body.stats]:is(ast.typed.StatReturn) then
    log.error(task, "the return statement in an inline task should be the last statement")
  end

  if find_self_recursion(task.prototype, body) then
    log.error(task, "inline tasks cannot be recursive")
  end
end

local function check_rf(node)
  if node:is(ast.typed.ExprID) then return true
  elseif node:is(ast.typed.ExprConstant) then return true
  elseif node:is(ast.typed.ExprFieldAccess) then
    return check_rf(node.value)
  elseif node:is(ast.typed.ExprIndexAccess) then
    return check_rf(node.value) and check_rf(node.index)
  else
    return false
  end
end

local function get_root_subregion(node)
  if node:is(ast.typed.ExprID) then
    local ty = std.as_read(node.expr_type)
    if std.is_region(ty) then return node.value
    elseif std.is_partition(ty) then return ty.parent_region_symbol
    else return nil end
  elseif node:is(ast.typed.ExprIndexAccess) then
    return get_root_subregion(node.value)
  end
  return nil
end

function inline_tasks.expr(cx, node)
  if node:is(ast.typed.ExprCall) then
    local stats = terralib.newlist()
    if type(node.fn.value) ~= "table" or not std.is_task(node.fn.value) then
      return stats, node
    end

    local task = node.fn.value
    local task_ast = task:getast()
    if node.inline == "demand" then
      check_valid_inline_task(task_ast)
    elseif not task_ast or not task_ast.inline or node.inline == "forbid" then
      return stats, node
    end

    local args = node.args:map(function(arg)
      local new_stats, new_node = inline_tasks.expr(cx, arg)
      stats:insertall(new_stats)
      return new_node
    end)
    local params = task_ast.params:map(function(param) return param.symbol end)
    local param_types = params:map(function(param) return param.type end)
    local task_body = task_ast.body
    local return_var = terralib.newsymbol(task_ast.return_type)
    local return_var_expr = expr_id(return_var, node)

    if task_ast.return_type ~= terralib.types.unit then
      stats:insert(stat_var(return_var, nil, node))
    end
    local new_block
    do
      local stats = terralib.newlist()
      local expr_mapping = {}
      local type_mapping = {}
      local new_local_params = terralib.newlist()
      local new_local_param_types = terralib.newlist()
      local new_args = terralib.newlist()
      std.zip(params, param_types, args):map(function(tuple)
        local param, param_type, arg = unpack(tuple)
        if check_rf(arg) then
          expr_mapping[param] = arg
          if std.is_region(param_type) then
            type_mapping[param_type] = std.as_read(arg.expr_type)
            type_mapping[param] = get_root_subregion(arg)
          end
        else
          local new_var = terralib.newsymbol(std.as_read(arg.expr_type))
          expr_mapping[param] = new_var
          new_local_params:insert(new_var)
          new_local_param_types:insert(new_var.type)
          new_args:insert(arg)
          if std.is_region(param_type) then
            type_mapping[param] = new_var
            type_mapping[param_type] = new_var.type
          end
        end
      end)

      if #new_local_params > 0 then
        stats:insert(ast.typed.StatVar {
          symbols = new_local_params,
          types = new_local_param_types,
          values = new_args,
          span = node.span
        })
      end
      local function subst(node)
        if rawget(node, "expr_type") then
          if node:is(ast.typed.ExprID) and expr_mapping[node.value] then
            local tgt = expr_mapping[node.value]
            if rawget(tgt, "expr_type") then node = tgt
            else node = node { value = tgt } end
          end
          return node { expr_type = std.type_sub(node.expr_type, type_mapping) }
        else
          return node
        end
      end
      stats:insertall(task_body.stats)
      if stats[#stats]:is(ast.typed.StatReturn) then
        local num_stats = #stats
        local return_stat = stats[num_stats]
        stats[num_stats] = stat_asgn(return_var_expr, return_stat.value, return_stat)
      end
      stats = ast.map_node_postorder(subst, stats)
      new_block = make_block(stats, node.span)
    end
    stats:insert(new_block)

    return stats, return_var_expr

  else
    local stats = terralib.newlist()
    local fields = {}
    local ctor = rawget(node, "node_type")

    for _, k in ipairs(ctor.expected_fields) do
      local field = node[k]
      if type(field) ~= "table" then
        fields[k] = field
      else
        local node_type = tostring(rawget(field, "node_type"))

        if node_type and string.find(node_type, ".Expr") then
          local new_stats, new_node = inline_tasks.expr(cx, field)
          stats:insertall(new_stats)
          fields[k] = new_node

        else
          fields[k] = field
        end
      end
    end

    return stats, ctor(fields)
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
  local ctor = rawget(node, "node_type")

  for _, k in ipairs(ctor.expected_fields) do
    local field = node[k]
    if type(field) ~= "table" then
      fields[k] = field
    else
      local node_type = tostring(rawget(field, "node_type"))
      if node_type and string.find(node_type, ".Stat") then
        local new_stats, new_node = inline_tasks.stat(cx, field)
        stats:insertall(new_stats)
        fields[k] = field

      elseif node_type and string.find(node_type, ".Expr") then
        local new_stats, new_node = inline_tasks.expr(cx, field)
        stats:insertall(new_stats)
        fields[k] = new_node

      elseif node_type and string.find(node_type, ".Block") then
        local new_node = inline_tasks.block(cx, field)
        fields[k] = new_node

      elseif node_type and string.find(node_type, ".Span") then
        fields[k] = field

      elseif terralib.islist(field) then
        fields[k] = field:map(function(field)
          if type(field) ~= "table" then
            return field
          else
            local node_type = tostring(rawget(field, "node_type"))
            if node_type and string.find(node_type, ".Stat") then
              local new_stats, new_node = inline_tasks.stat(cx, field)
              stats:insertall(new_stats)
              return new_node

            elseif node_type and string.find(node_type, ".Expr") then
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

  local node = ctor(fields)
  return stats, node
end

function inline_tasks.stat_task(cx, node)
  return node {
    body = inline_tasks.block(cx, node.body)
  }
end

function inline_tasks.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    if node.inline then check_valid_inline_task(node) end
    local new_node = inline_tasks.stat_task(cx, node)
    new_node.prototype:setast(new_node)
    return new_node

  elseif node:is(ast.typed.StatFspace) then
    return node

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end

end

function inline_tasks.entry(node)
  local cx = context:new_global_scope({})
  return inline_tasks.stat_top(cx, node)
end

return inline_tasks
