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

-- Utility functions for creating AST nodes

local ast = require("regent/ast")
local pretty = require("regent/pretty")
local std = require("regent/std")

local ast_util = {}

function ast_util.mk_stat_break()
  return ast.typed.stat.Break {
    span = ast.trivial_span(),
    annotations = ast.default_annotations()
  }
end

function ast_util.mk_expr_id(sym, ty)
  ty = ty or sym:gettype()
  return ast.typed.expr.ID {
    value = sym,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_id_rawref(sym, ty)
  ty = ty or std.rawref(&sym:gettype())
  return ast.typed.expr.ID {
    value = sym,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_index_access(value, index, ty)
  return ast.typed.expr.IndexAccess {
    value = value,
    index = index,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_field_access(value, field, ty)
  return ast.typed.expr.FieldAccess {
    value = value,
    field_name = field,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_bounds_access(value)
  if std.is_symbol(value) then
    value = ast_util.mk_expr_id(value, std.rawref(&value:gettype()))
  end
  local expr_type = std.as_read(value.expr_type)
  local index_type
  if std.is_region(expr_type) then
    return ast_util.mk_expr_bounds_access(
      ast_util.mk_expr_field_access(value, "ispace", expr_type:ispace()))
  elseif std.is_ispace(expr_type) then
    index_type = expr_type.index_type
  else
    assert(false, "unreachable")
  end
  return ast_util.mk_expr_field_access(value, "bounds", std.rect_type(index_type))
end

function ast_util.mk_expr_colors_access(value)
  if std.is_symbol(value) then
    value = ast_util.mk_expr_id(value, std.rawref(&value:gettype()))
  end
  local color_space_type = std.as_read(value.expr_type):colors()
  return ast_util.mk_expr_field_access(value, "colors", color_space_type)
end

function ast_util.mk_expr_binary(op, lhs, rhs)
  local lhs_type = std.as_read(lhs.expr_type)
  local rhs_type = std.as_read(rhs.expr_type)
  local function test()
    local terra query(lhs : lhs_type, rhs : rhs_type)
      return [ std.quote_binary_op(op, lhs, rhs) ]
    end
    return query:gettype().returntype
  end
  local valid, result_type = pcall(test)
  assert(valid)
  return ast.typed.expr.Binary {
    op = op,
    lhs = lhs,
    rhs = rhs,
    expr_type = result_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_method_call(value, expr_type, method_name, args)
  return ast.typed.expr.MethodCall {
    value = value,
    expr_type = expr_type,
    method_name = method_name,
    args = args,
    span = ast.trivial_span(),
    annotations = ast.default_annotations()
  }
end
function ast_util.mk_expr_call(fn, args, replicable)
  args = args or terralib.newlist()
  if not terralib.islist(args) then
    args = terralib.newlist {args}
  end
  local fn_type
  local expr_type
  if not std.is_task(fn) then
    local arg_symbols = args:map(function(arg)
      return terralib.newsymbol(arg.expr_type)
    end)
    local function test()
      local terra query([arg_symbols])
        return [fn]([arg_symbols])
      end
      return query:gettype()
    end
    local valid, query_type = pcall(test)
    assert(valid)
    fn_type = query_type
    expr_type = fn_type.returntype or terralib.types.unit
  else
    fn_type = fn:get_type()
    expr_type = fn_type.returntype or terralib.types.unit
  end

  return ast.typed.expr.Call {
    fn = ast.typed.expr.Function {
      value = fn,
      expr_type = fn_type,
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    args = args,
    expr_type = expr_type,
    conditions = terralib.newlist(),
    predicate = false,
    predicate_else_value = false,
    replicable = replicable or false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_constant(value, ty)
  return ast.typed.expr.Constant {
    value = value,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_ctor_list_field(expr)
  return ast.typed.expr.CtorListField {
    value = expr,
    expr_type = expr.expr_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_ctor_list_field_constant(c, ty)
  return ast_util.mk_expr_ctor_list_field(ast_util.mk_expr_constant(c, ty))
end

function ast_util.mk_expr_ctor(ls)
  local fields = ls:map(ast_util.mk_expr_ctor_list_field)
  local expr_type = std.ctor_tuple(fields:map(
    function(field) return field.expr_type end))
  return ast.typed.expr.Ctor {
    fields = fields,
    named = false,
    expr_type = expr_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_cast(ty, expr)
  return ast.typed.expr.Cast {
    fn = ast.typed.expr.Function {
      value = ty,
      expr_type =
        terralib.types.functype(terralib.newlist({std.untyped}), ty, false),
      span = ast.trivial_span(),
      annotations = ast.default_annotations(),
    },
    arg = expr,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_zeros(index_type)
  if index_type.dim == 1 then
    return ast_util.mk_expr_constant(0, int)
  else
    local fields = terralib.newlist()
    for idx = 1, index_type.dim do
      fields:insert(ast_util.mk_expr_constant(0, int))
    end
    return ast_util.mk_expr_cast(index_type, ast_util.mk_expr_ctor(fields))
  end
end

function ast_util.mk_expr_partition(partition_type, colors, coloring)
  return ast.typed.expr.Partition {
    disjointness = partition_type.disjointness,
    completeness = partition_type.completeness,
    region = ast_util.mk_expr_id(partition_type.parent_region_symbol),
    coloring = coloring,
    colors = colors,
    expr_type = partition_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_ispace(index_type, extent)
  return ast.typed.expr.Ispace {
    extent = extent,
    start = false,
    index_type = index_type,
    expr_type = std.ispace(index_type),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_expr_partition_equal(partition_type, colors)
  return ast.typed.expr.PartitionEqual {
    region = ast_util.mk_expr_id(partition_type.parent_region_symbol),
    colors = colors,
    expr_type = partition_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_var(sym, ty, value)
  ty = ty or sym:gettype()
  value = value or false
  return ast.typed.stat.Var {
    symbol = sym,
    type = ty,
    value = value,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_empty_block()
  return ast.typed.Block {
    stats = terralib.newlist(),
    span = ast.trivial_span(),
  }
end

function ast_util.mk_stat_expr(expr)
  return ast.typed.stat.Expr {
    expr = expr,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_block(block)
  return ast.typed.stat.Block {
    block = block,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_block(stats)
  if terralib.islist(stats) then
    return ast.typed.Block {
      stats = stats,
      span = ast.trivial_span(),
    }
  else
    return ast.typed.Block {
      stats = terralib.newlist {stats},
      span = ast.trivial_span(),
    }
  end
end

function ast_util.mk_stat_if(cond, stat)
  return ast.typed.stat.If {
    cond = cond,
    then_block = ast_util.mk_block(stat),
    elseif_blocks = terralib.newlist(),
    else_block = ast_util.mk_empty_block(),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_if_else(cond, then_stats, else_stats)
  return ast.typed.stat.If {
    cond = cond,
    then_block = ast_util.mk_block(then_stats),
    elseif_blocks = terralib.newlist(),
    else_block = ast_util.mk_block(else_stats),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_elseif(cond, stat)
  return ast.typed.stat.Elseif {
    cond = cond,
    block = ast_util.mk_block(stat),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_assignment(lhs, rhs)
  return ast.typed.stat.Assignment {
    lhs = lhs,
    rhs = rhs,
    metadata = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_reduce(op, lhs, rhs)
  return ast.typed.stat.Reduce {
    op = op,
    lhs = lhs,
    rhs = rhs,
    metadata = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_for_list(symbol, value, block)
  return ast.typed.stat.ForList {
    symbol = symbol,
    value = value,
    block = block,
    metadata = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.mk_stat_for_num(symbol, values, block)
  return ast.typed.stat.ForNum {
    block = block,
    values = values,
    symbol = symbol,
    metadata = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations()
  }
end

function ast_util.mk_task_param(symbol)
  return ast.typed.top.TaskParam {
    symbol = symbol,
    param_type = symbol:gettype(),
    future = false,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

function ast_util.render(expr)
  if expr == nil then return "nil" end
  if expr:is(ast.typed.expr) then
    return pretty.entry_expr(expr)
  elseif expr:is(ast.typed.stat) then
    return pretty.entry_stat(expr)
  end
end

function ast_util.get_base_indexed_node(node, previous_index)
  if node:is(ast.typed.expr.IndexAccess) then
    return ast_util.get_base_indexed_node(node.value, node.index)
  end
  return node, previous_index
end

function ast_util.replace_base_indexed_node(node, replacement)
  if node:is(ast.typed.expr.IndexAccess) then
    return node { value = ast_util.replace_base_indexed_node(node.value, replacement) }
  end
  assert(std.type_eq(std.as_read(node.expr_type), std.as_read(replacement.expr_type)))
  return replacement
end

return ast_util
