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

-- Regent Auto-parallelizer

local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")
local report = require("common/report")
local pretty = require("regent/pretty")
local symbol_table = require("regent/symbol_table")
local passes = require("regent/passes")

local c = std.c

-- Initialize all sub-components
local Lambda
local Stencil

local parallel_task_context = {}
local parallel_param = {}
local caller_context = {}
local global_context = {}

local check_parallelizable = {}
local normalize_accesses = {}
local reduction_analysis = {}
local stencil_analysis = {}
local parallelize_task_calls = {}
local parallelize_tasks = {}

-- #####################################
-- ## Utilities for point and rect manipulation
-- #################

local print_rect = {
  [std.rect1d] = terra(r : std.rect1d)
    c.printf("%d -- %d\n", r.lo.__ptr, r.hi.__ptr)
  end,
  [std.rect2d] = terra(r : std.rect2d)
    c.printf("(%d, %d) -- (%d, %d)\n", r.lo.__ptr.x, r.lo.__ptr.y, r.hi.__ptr.x, r.hi.__ptr.y)
  end,
  [std.rect3d] = terra(r : std.rect3d)
    c.printf("(%d, %d, %d) -- (%d, %d, %d)\n",
             r.lo.__ptr.x, r.lo.__ptr.y, r.lo.__ptr.z,
             r.hi.__ptr.x, r.hi.__ptr.y, r.hi.__ptr.z)
  end
}

local print_point = {
  [std.int1d] = terra(p : std.int1d)
    c.printf("%d\n", p.__ptr)
  end,
  [std.int2d] = terra(p : std.int2d)
    c.printf("(%d, %d)\n", p.__ptr.x, p.__ptr.y)
  end,
  [std.int3d] = terra(p : std.int3d)
    c.printf("(%d, %d, %d)\n", p.__ptr.x, p.__ptr.y, p.__ptr.z)
  end
}

-- TODO: needs to automatically generate these functions
local function get_ghost_rect_body(res, sz, r, s, f)
  local acc = function(expr) return `([expr].[f]) end
  if f == nil then acc = function(expr) return expr end end
  return quote
    if [acc(`([s].lo.__ptr))] == [acc(`([r].lo.__ptr))] then
      [acc(`([res].lo.__ptr))] = [acc(`([r].lo.__ptr))]
      [acc(`([res].hi.__ptr))] = [acc(`([r].hi.__ptr))]
    else
      -- wrapped around left, periodic boundary
      if [acc(`([s].lo.__ptr))] > [acc(`([s].hi.__ptr))] then
        -- shift left
        if [acc(`([r].lo.__ptr))] <= [acc(`([s].hi.__ptr))] then
          [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
          [acc(`([res].hi.__ptr))] = ([acc(`([r].lo.__ptr))] - 1 + [acc(`([sz].__ptr))]) % [acc(`([sz].__ptr))]
        -- shift right
        elseif [acc(`([s].lo.__ptr))] <= [acc(`([r].hi.__ptr))] then
          [acc(`([res].lo.__ptr))] = ([acc(`([r].hi.__ptr))] + 1) % [acc(`([sz].__ptr))]
          [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
        else
          std.assert(false,
            "ambiguous ghost region, primary region should be bigger for this stencil")
        end
      else -- [acc(`([s].lo.__ptr))] < [acc(`([r].hi.__ptr))]
        -- shift left
        if [acc(`([s].lo.__ptr))] < [acc(`([r].lo.__ptr))] then
          [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
          [acc(`([res].hi.__ptr))] = [acc(`([r].lo.__ptr))] - 1
        -- shift right
        else -- [acc(`([s].lo.__ptr)) > [acc(`([r].lo.__ptr))
          [acc(`([res].lo.__ptr))] = [acc(`([r].hi.__ptr))] + 1
          [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
        end
      end
    end
  end
end

local function bounds_checks(res)
  local checks = quote end
  if std.config["debug"] then
    checks = quote
      std.assert(
        [res].lo <= [res].hi,
        "invalid size for a ghost region. the serial code has an out-of-bounds access")
    end
  end
  return checks
end

local get_ghost_rect = {
  [std.rect1d] = terra(root : std.rect1d, r : std.rect1d, s : std.rect1d) : std.rect1d
    var sz = root:size()
    var diff_rect : std.rect1d
    [get_ghost_rect_body(diff_rect, sz, r, s)]
    [bounds_checks(diff_rect)]
    return diff_rect
  end,
  [std.rect2d] = terra(root : std.rect2d, r : std.rect2d, s : std.rect2d) : std.rect2d
    var sz = root:size()
    var diff_rect : std.rect2d
    [get_ghost_rect_body(diff_rect, sz, r, s, "x")]
    [get_ghost_rect_body(diff_rect, sz, r, s, "y")]
    [bounds_checks(diff_rect)]
    return diff_rect
  end,
  [std.rect3d] = terra(root : std.rect3d, r : std.rect3d, s : std.rect3d) : std.rect3d
    var sz = root:size()
    var diff_rect : std.rect3d
    [get_ghost_rect_body(diff_rect, sz, r, s, "x")]
    [get_ghost_rect_body(diff_rect, sz, r, s, "y")]
    [get_ghost_rect_body(diff_rect, sz, r, s, "z")]
    [bounds_checks(diff_rect)]
    return diff_rect
  end
}

local min_points
local max_points

do
  local function min(a, b) return `terralib.select(a < b, a, b) end
  local function max(a, b) return `terralib.select(a > b, a, b) end
  local function gen(f)
    return {
      [std.int1d] = terra(p1 : std.int1d, p2 : std.int1d) : std.int1d
        var p : std.int1d
        p.__ptr = [f(`(p1.__ptr), `(p2.__ptr))]
      end,
      [std.int2d] = terra(p1 : std.int2d, p2 : std.int2d) : std.int2d
        var p : std.int2d
        p.__ptr.x = [f(`(p1.__ptr.x), `(p2.__ptr.x))]
        p.__ptr.y = [f(`(p1.__ptr.y), `(p2.__ptr.y))]
        return p
      end,
      [std.int3d] = terra(p1 : std.int3d, p2 : std.int3d) : std.int3d
        var p : std.int3d
        p.__ptr.x = [f(`(p1.__ptr.x), `(p2.__ptr.x))]
        p.__ptr.y = [f(`(p1.__ptr.y), `(p2.__ptr.y))]
        p.__ptr.z = [f(`(p1.__ptr.z), `(p2.__ptr.z))]
        return p
      end,
    }
  end
  min_points = gen(min)
  max_points = gen(max)
end

local function get_intersection(rect_type)
  return terra(r1 : rect_type, r2 : rect_type) : rect_type
    var r : rect_type
    r.lo = [ max_points[rect_type.index_type] ](r1.lo, r2.lo)
    r.hi = [ min_points[rect_type.index_type] ](r1.hi, r2.hi)
    return r
  end
end

local function render(expr)
  if expr == nil then return "nil" end
  if expr:is(ast.typed.expr) then
    return pretty.render.entry(nil, pretty.expr(nil, expr))
  elseif expr:is(ast.typed.stat) then
    return pretty.render.entry(nil, pretty.stat(nil, expr))
  end
end

local function shallow_copy(tbl)
  local new_tbl = {}
  for k, v in pairs(tbl) do
    new_tbl[k] = v
  end
  return new_tbl
end

local function factorize(number, ways)
  local factors = terralib.newlist()
  for k = ways, 2, -1 do
    local limit = math.ceil(math.pow(number, 1 / k))
    local factor
    for idx = 1, limit do
      if number % idx == 0 then
        factor = idx
      end
    end
    number = number / factor
    factors:insert(factor)
  end
  factors:insert(number)
  factors:sort(function(a, b) return a > b end)
  return factors
end

-- #####################################
-- ## Utilities for AST node manipulation
-- #################

local tmp_var_id = 0

local function get_new_tmp_var(ty)
  local sym = std.newsymbol(ty, "__t".. tostring(tmp_var_id))
  tmp_var_id = tmp_var_id + 1
  return sym
end

local function mk_expr_id(sym, ty)
  ty = ty or sym:gettype()
  return ast.typed.expr.ID {
    value = sym,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_index_access(value, index, ty)
  return ast.typed.expr.IndexAccess {
    value = value,
    index = index,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_field_access(value, field, ty)
  return ast.typed.expr.FieldAccess {
    value = value,
    field_name = field,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_bounds_access(value)
  if std.is_symbol(value) then
    value = mk_expr_id(value, std.rawref(&value:gettype()))
  end
  local expr_type = std.as_read(value.expr_type)
  local index_type
  if std.is_region(expr_type) then
    index_type = expr_type:ispace().index_type
  elseif std.is_ispace(expr_type) then
    index_type = expr_type.index_type
  else
    assert(false, "unreachable")
  end
  return mk_expr_field_access(value, "bounds", std.rect_type(index_type))
end

local function mk_expr_colors_access(value)
  if std.is_symbol(value) then
    value = mk_expr_id(value, std.rawref(&value:gettype()))
  end
  local color_space_type = std.as_read(value.expr_type):colors()
  return mk_expr_field_access(value, "colors", color_space_type)
end

local function mk_expr_binary(op, lhs, rhs)
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

local function mk_expr_call(fn, args)
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
    fn_type = fn:gettype()
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
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_constant(value, ty)
  return ast.typed.expr.Constant {
    value = value,
    expr_type = ty,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_ctor_list_field(expr)
  return ast.typed.expr.CtorListField {
    value = expr,
    expr_type = expr.expr_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_ctor_list_field_constant(c, ty)
  return mk_expr_ctor_list_field(mk_expr_constant(c, ty))
end

local function mk_expr_ctor(ls)
  local fields = ls:map(mk_expr_ctor_list_field)
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

local function mk_expr_cast(ty, expr)
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

local function mk_expr_zeros(index_type)
  if index_type.dim == 1 then
    return mk_expr_constant(0, int)
  else
    local fields = terralib.newlist()
    for idx = 1, index_type.dim do
      fields:insert(mk_expr_constant(0, int))
    end
    return mk_expr_cast(index_type, mk_expr_ctor(fields))
  end
end

local function mk_expr_partition(partition_type, colors, coloring)
  return ast.typed.expr.Partition {
    disjointness = partition_type.disjointness,
    region = mk_expr_id(partition_type.parent_region_symbol),
    coloring = coloring,
    colors = colors,
    expr_type = partition_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_ispace(index_type, extent)
  assert(not index_type:is_opaque())
  return ast.typed.expr.Ispace {
    extent = extent,
    start = false,
    index_type = index_type,
    expr_type = std.ispace(index_type),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_expr_partition_equal(partition_type, colors)
  return ast.typed.expr.PartitionEqual {
    region = mk_expr_id(partition_type.parent_region_symbol),
    colors = colors,
    expr_type = partition_type,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_var(sym, ty, value)
  ty = ty or sym:gettype()
  return ast.typed.stat.Var {
    symbols = terralib.newlist {sym},
    types = terralib.newlist {ty},
    values = terralib.newlist {value},
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_empty_block()
  return ast.typed.Block {
    stats = terralib.newlist(),
    span = ast.trivial_span(),
  }
end

local function mk_stat_expr(expr)
  return ast.typed.stat.Expr {
    expr = expr,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_block(block)
  return ast.typed.stat.Block {
    block = block,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_block(stats)
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

local function mk_stat_if(cond, stat)
  return ast.typed.stat.If {
    cond = cond,
    then_block = mk_block(stat),
    elseif_blocks = terralib.newlist(),
    else_block = mk_empty_block(),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_elseif(cond, stat)
  return ast.typed.stat.Elseif {
    cond = cond,
    block = mk_block(stat),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_assignment(lhs, rhs)
  return ast.typed.stat.Assignment {
    lhs = terralib.newlist {lhs},
    rhs = terralib.newlist {rhs},
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_reduce(op, lhs, rhs)
  return ast.typed.stat.Reduce {
    op = op,
    lhs = terralib.newlist {lhs},
    rhs = terralib.newlist {rhs},
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_stat_for_list(symbol, value, block)
  return ast.typed.stat.ForList {
    symbol = symbol,
    value = value,
    block = block,
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function mk_task_param(symbol)
  return ast.typed.top.TaskParam {
    symbol = symbol,
    param_type = symbol:gettype(),
    span = ast.trivial_span(),
    annotations = ast.default_annotations(),
  }
end

local function copy_region_type(old_type)
  return std.region(std.ispace(old_type:ispace().index_type),
                    old_type:fspace())
end

local function copy_region_symbol(old, name)
  local region_type = copy_region_type(old:gettype())
  return std.newsymbol(region_type, name)
end

local function extract_expr(node, pred, fn)
  ast.traverse_node_postorder(function(node)
    if pred(node) then fn(node) end
  end, node)
end

local function extract_index_access_expr(node)
  local index_access
  extract_expr(node,
    function(node) return node:is(ast.typed.expr.IndexAccess) end,
    function(node) index_access = node end)
  return index_access
end

local function extract_index_expr(node)
  local index_access = extract_index_access_expr(node)
  return index_access.index
end

local function extract_ctor(node)
  local ctor_expr
  extract_expr(node,
    function(node) return node:is(ast.typed.expr.Ctor) end,
    function(node) ctor_expr = node end)
  return ctor_expr
end

local function extract_symbol(pred, node)
  local sym
  extract_expr(node,
    function(node) return node:is(ast.typed.expr.ID) and pred(node) end,
    function(node) sym = node.value end)
  return sym
end

local function always(node) return true end

local function extract_symbols(pred, node)
  local symbol_set = {}
  extract_expr(node,
    function(node) return node:is(ast.typed.expr.ID) and pred(node) end,
    function(node) symbol_set[node.value] = true end)
  local symbols = terralib.newlist()
  for sym, _ in pairs(symbol_set) do symbols:insert(sym) end
  return symbols
end

local function rewrite_expr(node, pred, fn)
  return ast.map_node_continuation(function(node, continuation)
    if pred(node) then return fn(node)
    else return continuation(node, true) end
  end, node)
end

local function rewrite_expr_id(node, from_sym, to_expr)
  return rewrite_expr(node,
    function(node)
      return node:is(ast.typed.expr.ID) and
             node.value == from_sym end,
    function(node) return to_expr end)
end

local function rewrite_symbol_pred(node, pred, sym)
  return rewrite_expr(node,
    function(node) return node:is(ast.typed.expr.ID) and pred(node) end,
    function(node) return node {
      value = sym,
      expr_type = sym:gettype(),
    }
    end)
end

local function rewrite_symbol(node, from, to)
  return rewrite_symbol_pred(node,
    function(node) return node.value == from end,
    to)
end

-- #####################################
-- ## Stencil and Lambda
-- #################

do
  local lambda = {}

  lambda.__index = lambda

  function lambda.__call(self, arg)
    local expr = self:expr()
    if Lambda.is_lambda(expr) then
      local binder = expr:binder()
      expr = Lambda {
        expr = expr:expr(),
        binder = self:binder(),
      }
      return Lambda {
        expr = expr(arg),
        binder = binder,
      }
    else
      if std.is_symbol(arg) then
        return rewrite_symbol(expr, self:binder(), arg)
      else
        return rewrite_expr_id(expr, self:binder(), arg)
      end
    end
  end

  function lambda:expr()
    return self.__expr
  end

  function lambda:binder()
    return self.__binder
  end

  function lambda:all_binders(l)
    l = l or terralib.newlist()
    l:insert(self.__binder)
    if Lambda.is_lambda(self:expr()) then
      return lambda:all_binders(l)
    else
      return l
    end
  end

  function lambda:field_path()
    if Lambda.is_lambda(self:expr()) then
      return self:expr():field_path()
    else
      if self:expr():is(ast.typed.expr.FieldAccess) then
        return self:expr().expr_type.field_path
      else
        return data.newtuple()
      end
    end
  end

  function lambda:fmap(fn)
    local expr = self:expr()
    if Lambda.is_lambda(expr) then
      return Lambda {
        expr = expr:fmap(fn),
        binder = self:binder(),
      }
    else
      return Lambda {
        expr = fn(expr),
        binder = self:binder(),
      }
    end
  end

  function lambda:__tostring()
    local binder_str = tostring(self:binder())
    local expr_str
    if Lambda.is_lambda(self:expr()) then
      expr_str = tostring(self:expr())
    else
      expr_str = render(self:expr())
    end
    return "\\" .. binder_str .. "." .. expr_str
  end

  local lambda_factory = {}

  lambda_factory.__index = lambda_factory

  function lambda_factory.__call(self, args)
    assert(args.expr)
    assert((args.binder ~= nil or args.binders ~= nil) and
           not (args.binder ~= nil and args.binders ~= nil))
    local binders = args.binders or { args.binder }
    local expr = args.expr
    for idx = #binders, 1, -1 do
      expr = setmetatable({
        __expr = expr,
        __binder = binders[idx],
      }, lambda)
    end
    return expr
  end

  function lambda_factory.is_lambda(e)
    return getmetatable(e) == lambda
  end

  Lambda = setmetatable({}, lambda_factory)
end

do
  local stencil = {}
  stencil.__index = stencil

  function stencil:subst(mapping)
    return Stencil {
      region = self:region(mapping),
      index = self:index(mapping),
      range = self:range(mapping),
      fields = shallow_copy(self.__fields),
    }
  end

  function stencil:region(mapping)
    local region = self.__region
    if mapping then
      region = mapping[region] or region
    end
    return region
  end

  function stencil:index(mapping)
    local index = self.__index_expr
    if mapping then
      local symbols = extract_symbols(always, index)
      index = Lambda {
        binders = symbols,
        expr = index,
      }
      local unbound = terralib.newlist()
      for idx = 1, #symbols do
        if mapping[symbols[idx]] then
          index = index(mapping[symbols[idx]])
        else
          index = index(symbols[idx])
          unbound:insert(symbols[idx])
        end
      end
      if #unbound > 0 then
        index = Lambda {
          binders = unbound,
          expr = index,
        }
      end
    end
    return index
  end

  function stencil:range(mapping)
    local range = self.__range
    if mapping then
      range = mapping[range] or range
    end
    return range
  end

  function stencil:fields()
    local fields = terralib.newlist()
    for _, field in pairs(self.__fields) do
      fields:insert(field)
    end
    return fields
  end

  function stencil:has_field(field)
    return self.__fields[field:hash()] ~= nil
  end

  function stencil:add_field(field)
    if not self:has_field(field) then
      self.__fields[field:hash()] = field
    end
  end

  function stencil:add_fields(fields)
    fields:map(function(field) self:add_field(field) end)
  end

  function stencil:replace_index(index)
    return Stencil {
      region = self:region(),
      index = index,
      range = self:range(),
      fields = shallow_copy(self.__fields),
    }
  end

  function stencil:to_expr()
    assert(Stencil.is_singleton(self))
    local region = self:region()
    local expr = self:index()
    expr = mk_expr_index_access(
      mk_expr_id(region, std.rawref(&region:gettype())),
      expr, std.ref(region, region:fspace()))
  end

  function stencil:__tostring()
    local fields = self:fields()
    local field_str = fields[1]:mkstring("", ".")
    for idx = 2, #fields do
      field_str = field_str .. ", " .. fields[idx]:mkstring("", ".")
    end
    if #fields > 1 then field_str = "{" .. field_str .. "}" end
    local index_str = tostring(self:index())
    if not Lambda.is_lambda(self:index()) then
      index_str = render(self:index())
    end
    return tostring(self:region()) .. "[" ..  index_str .. "]." ..
           field_str .. " (range: " ..  tostring(self:range()) .. ")"
  end

  local stencil_factory = {}
  stencil_factory.__index = stencil_factory

  function stencil_factory.__call(self, args)
    assert(args.region)
    assert(args.index)
    assert(args.range)
    assert(args.fields)
    return setmetatable({
      __region = args.region,
      __index_expr = args.index,
      __range = args.range,
      __fields = args.fields,
    }, stencil)
  end

  function stencil_factory.is_stencil(s)
    return type(s) == "table" and getmetatable(s) == stencil
  end

  function stencil_factory.is_singleton(s)
    local cnt = 0
    for field, _ in pairs(s.__fields) do
      cnt = cnt + 1
      if cnt > 1 then return false end
    end
    return true
  end

  Stencil = setmetatable({}, stencil_factory)
end

-- #####################################
-- ## Some context objects
-- #################

parallel_task_context.__index = parallel_task_context

function parallel_task_context.new_task_scope(params)
  local region_params = {}
  local region_param_map = {}
  local region_param_indices = terralib.newlist()
  local index_type
  for idx = 1, #params do
    if std.is_region(params[idx].param_type) then
      local symbol = params[idx].symbol
      local region_type = symbol:gettype()
      local name = symbol:getname()
      local bounds_symbol = std.newsymbol(
        std.rect_type(region_type:ispace().index_type),
        "__" .. name .. "_bounds")
      region_params[idx] = {
        region = symbol,
        bounds = bounds_symbol,
      }
      region_param_map[symbol] = region_params[idx]
      region_param_indices:insert(idx)
      assert(index_type == nil or
             index_type == symbol:gettype():ispace().index_type)
      index_type = symbol:gettype():ispace().index_type
    end
  end

  local cx = {}
  cx.region_params = region_params
  cx.region_param_map = region_param_map
  cx.region_param_indices = region_param_indices
  cx.task_point_symbol = get_new_tmp_var(index_type)
  cx.field_accesses = {}
  cx.field_access_stats = {}
  cx.stencils = terralib.newlist()
  cx.ghost_symbols = terralib.newlist()
  return setmetatable(cx, parallel_task_context)
end

function parallel_task_context:is_region_param(idx)
  return self.region_params[idx] ~= nil
end

function parallel_task_context:get_task_point_symbol()
  return self.task_point_symbol
end

function parallel_task_context:find_metadata_parameters(region_symbol)
  return self.region_param_map[region_symbol]
end

function parallel_task_context:insert_metadata_parameters(params)
  for idx = 1, #self.region_param_indices do
    local region_param =
      self.region_params[self.region_param_indices[idx]]
    params:insert(mk_task_param(region_param.bounds))
  end
  params:insert(mk_task_param(self:get_task_point_symbol()))
end

function parallel_task_context:make_param_arg_mapping(caller_cx, args)
  local mapping = {}
  for idx = 1, #args do
    if self:is_region_param(idx) then
      assert(args[idx]:is(ast.typed.expr.ID))
      local orig_param = self.region_params[idx]
      mapping[orig_param.region] = args[idx].value
      -- TODO: Handle other metadata symbols
      mapping[orig_param.bounds] =
        caller_cx:find_bounds_symbol(args[idx].value)
      assert(mapping[orig_param.bounds])
    end
  end
  return mapping
end

function parallel_task_context:access_requires_stencil_analysis(access)
  return self.field_accesses[access] ~= nil
end

function parallel_task_context:record_stat_requires_case_split(stat)
  self.field_access_stats[stat] = true
end

function parallel_task_context:stat_requires_case_split(stat)
  return self.field_access_stats[stat]
end

function parallel_task_context:add_access(access, stencil)
  self.field_accesses[access] = {
    stencil = stencil,
    ghost_indices = terralib.newlist(),
    exploded_stencils = terralib.newlist(),
  }
end

parallel_param.__index = parallel_param

function parallel_param.new(dop)
  local params = {
    __dop = dop,
  }
  return setmetatable(params, parallel_param)
end

function parallel_param:hash()
  return "dop" .. tostring(self.__dop)
end

function parallel_param:dop()
  return self.__dop
end

caller_context.__index = caller_context

function caller_context.new(constraints)
  local param = parallel_param.new(std.config["parallelize-dop"])
  local cx = {
    __param_stack = terralib.newlist { param },
    -- No need to track scopes for declarations
    -- since symbols are already disambiguated
    __region_decls = {},
    -- Map from region declaration to sets of call expressions
    -- grouped by parallelization parameters
    __call_exprs_by_region_decl = {},
    -- call expr -> parallelization parameter
    __parallel_params = {},
    -- region symbol x param -> partition symbol
    __primary_partitions = {},
    -- call expr x stencil -> list of partition symbols
    __ghost_partitions = {},
    -- param -> symbol
    __color_spaces = data.newmap(),
    -- keep parent-child relationship in region tree
    __parent_region = {},
    -- the constraint graph to update for later stages
    constraints = constraints,
    -- symbols for caching region metadata
    __region_metadata_symbols = {},
  }
  return setmetatable(cx, caller_context)
end

function caller_context:push_scope()
  -- TODO: this should come from bishop
  local param = parallel_param.new(std.config["parallelize-dop"])
  self.__param_stack:insert(param)
end

function caller_context:pop_scope()
  self.__param_stack[#self.__param_stack] = nil
end

function caller_context:add_region_decl(symbol, stat)
  self.__region_decls[symbol] = stat
end

function caller_context:add_bounds_symbol(region_symbol, bounds_symbol)
  if self.__region_metadata_symbols[region_symbol] == nil then
    self.__region_metadata_symbols[region_symbol] = {}
  end
  self.__region_metadata_symbols[region_symbol] = {
    bounds = bounds_symbol
  }
end

function caller_context:find_bounds_symbol(region_symbol)
  return self.__region_metadata_symbols[region_symbol] and
         self.__region_metadata_symbols[region_symbol].bounds
end

function caller_context:add_call(expr)
  local param = self.__param_stack[#self.__param_stack]
  for idx = 1, #expr.args do
    if std.is_region(std.as_read(expr.args[idx].expr_type)) then
      -- TODO: Arguments can be some_partition[some_index].
      --       Normalization required.
      assert(expr.args[idx]:is(ast.typed.expr.ID))
      local region_symbol = expr.args[idx].value
      local decl = self.__region_decls[region_symbol]
      if self.__call_exprs_by_region_decl[decl] == nil then
        self.__call_exprs_by_region_decl[decl] = data.newmap()
      end
      if self.__call_exprs_by_region_decl[decl][param] == nil then
        self.__call_exprs_by_region_decl[decl][param] = {}
      end
      self.__call_exprs_by_region_decl[decl][param][expr] = true
    end
  end
  if self.__parallel_params[expr] == nil then
    self.__parallel_params[expr] = param
  end
end

function caller_context:get_call_exprs(decl)
  return self.__call_exprs_by_region_decl[decl]
end

function caller_context:add_primary_partition(region, param, partition)
  if self.__primary_partitions[region] == nil then
    self.__primary_partitions[region] = data.newmap()
  end
  self.__primary_partitions[region][param] = partition
end

function caller_context:add_ghost_partition(call, stencil, partition)
  if self.__ghost_partitions[call] == nil then
    self.__ghost_partitions[call] = {}
  end
  assert(self.__ghost_partitions[call][stencil] == nil)
  self.__ghost_partitions[call][stencil] = partition
end

function caller_context:add_parent_region(region, parent_region)
  self.__parent_region[region] = parent_region
end

function caller_context:add_color_space(param, color)
  assert(self.__color_spaces[param] == nil)
  self.__color_spaces[param] = color
end

function caller_context:find_primary_partition(param, region)
  assert(self.__primary_partitions[region][param])
  return self.__primary_partitions[region][param]
end

function caller_context:find_primary_partition_by_call(call, region)
  local param = self.__parallel_params[call]
  assert(self.__primary_partitions[region][param])
  return self.__primary_partitions[region][param]
end

function caller_context:find_ghost_partition(call, stencil)
  assert(self.__ghost_partitions[call][stencil])
  return self.__ghost_partitions[call][stencil]
end

function caller_context:find_parent_region(region)
  return self.__parent_region[region]
end

function caller_context:find_color_space(param)
  return self.__color_spaces[param]
end

function caller_context:find_color_space_by_call(call)
  local param = self.__parallel_params[call]
  assert(self.__color_spaces[param])
  return self.__color_spaces[param]
end

function caller_context:update_constraint(expr)
  assert(expr:is(ast.typed.expr.IndexAccess))
  local value_type = expr.value.expr_type
  local partition = value_type:partition()
  local parent = value_type:parent_region()
  local subregion = expr.expr_type
  std.add_constraint(self, partition, parent, std.subregion, false)
  std.add_constraint(self, subregion, partition, std.subregion, false)
end

-- #####################################
-- ## Task parallelizability checker
-- #################

function check_parallelizable.top_task(node)
  -- TODO: raise an error if a task has unsupported syntax

  -- conditions of parallelizable tasks
  -- 1. no task call; i.e. needs to be leaf
  -- 2. no aliasing between read and write sets (a subset of 6)
  -- 3. no region or partition creation or deletion
  -- 4. no region allocation
  -- 5. no break or return in the middle of control flow
  -- 6. loops should be vectorizable, but scalar reductions are allowed
  -- 7. only math function calls are allowed (through std.*)
  -- 8. uncentered accesses should have indices of form either
  --    e +/- c or (e +/- c) % r.bounds where r is a region
end

-- #####################################
-- ## Code generation utilities for partition creation and indexspace launch
-- #################

local function create_equal_partition(caller_cx, region_symbol, pparam)
  local region_type = region_symbol:gettype()
  local index_type = region_type:ispace().index_type
  local dim = index_type.dim
  local factors = factorize(pparam:dop(), dim)
  local extent_expr
  if dim > 1 then
    extent_expr = mk_expr_ctor(factors:map(function(f)
      return mk_expr_constant(f, int)
    end))
  else
    extent_expr = mk_expr_constant(factors[1], int)
  end

  local color_space_expr = mk_expr_ispace(index_type, extent_expr)
  local partition_type =
    std.partition(disjoint, region_symbol, color_space_expr.expr_type)
  local partition_symbol = get_new_tmp_var(partition_type)

  local stats = terralib.newlist { mk_stat_var(
    partition_symbol,
    nil,
    mk_expr_partition_equal(partition_type, color_space_expr))
  }

  caller_cx:add_primary_partition(region_symbol, pparam, partition_symbol)

  local color_space_symbol = caller_cx:find_color_space(pparam)
  if color_space_symbol == nil then
    color_space_symbol = get_new_tmp_var(partition_type:colors())
    local colors_expr = mk_expr_colors_access(partition_symbol)
    stats:insert(mk_stat_var(color_space_symbol, nil, colors_expr))
    caller_cx:add_color_space(pparam, color_space_symbol)
  elseif std.config["debug"] then
    local bounds = mk_expr_bounds_access(color_space_symbol)
    local my_bounds = mk_expr_bounds_access(
      mk_expr_colors_access(partition_symbol))
    stats:insert(mk_stat_expr(mk_expr_call(
        std.assert, terralib.newlist {
          mk_expr_binary("==", bounds, my_bounds),
          mk_expr_constant("color space bounds mismatch", rawstring)
        })))
  end

  return partition_symbol, stats
end

-- makes a loop as follows to create a coloring object:
--
--   for c in primary_partition.colors do
--     legion_domain_point_coloring_color_domain(coloring,
--       c, get_ghost_rect(primary_region.bounds,
--                         primary_partition[c].bounds,
--                         stencil(primary_partition[c].bounds)))
--   var ghost_partition = partition(disjoint, primary_region, coloring)
--
local function create_image_partition(caller_cx, pr, pp, stencil, pparam)
  local color_space_symbol = caller_cx:find_color_space(pparam)
  assert(color_space_symbol)
  local color_space_expr = mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type)
  local color_expr = mk_expr_id(color_symbol)

  local pr_type = std.as_read(pr:gettype())
  local pr_index_type = pr_type:ispace().index_type
  local pr_rect_type = std.rect_type(pr_index_type)

  -- TODO: disjointness check
  local gp_type = std.partition(std.aliased, pr, color_space_symbol:gettype())
  local gp_symbol = get_new_tmp_var(gp_type)
  local stats = terralib.newlist()

  local coloring_symbol = get_new_tmp_var(c.legion_domain_point_coloring_t)
  local coloring_expr = mk_expr_id(coloring_symbol)
  stats:insert(mk_stat_var(coloring_symbol, nil,
                           mk_expr_call(c.legion_domain_point_coloring_create)))

  local loop_body = terralib.newlist()
  local pr_expr = mk_expr_id(pr)
  local pp_expr = mk_expr_id(pp)
  local sr_type = pp:gettype():subregion_dynamic()
  local sr_expr = mk_expr_index_access(pp_expr, color_expr, sr_type)
  local pr_bounds_expr = mk_expr_bounds_access(pr_expr)
  local sr_bounds_expr = mk_expr_bounds_access(sr_expr)
  local sr_lo_expr = mk_expr_field_access(sr_bounds_expr, "lo", pr_index_type)
  local sr_hi_expr = mk_expr_field_access(sr_bounds_expr, "hi", pr_index_type)
  local shift_lo_expr = stencil(sr_lo_expr)
  local shift_hi_expr = stencil(sr_hi_expr)
  local tmp_var = get_new_tmp_var(pr_rect_type)
  loop_body:insert(mk_stat_var(tmp_var, nil,
    mk_expr_ctor(terralib.newlist {shift_lo_expr, shift_hi_expr})))
  local ghost_rect_expr =
    mk_expr_call(get_ghost_rect[pr_rect_type],
                 terralib.newlist { pr_bounds_expr,
                                    sr_bounds_expr,
                                    mk_expr_id(tmp_var) })
  --loop_body:insert(mk_stat_block(mk_block(terralib.newlist {
  --  mk_stat_expr(mk_expr_call(print_rect[pr_rect_type],
  --                            pr_bounds_expr)),
  --  mk_stat_expr(mk_expr_call(print_rect[pr_rect_type],
  --                            sr_bounds_expr)),
  --  mk_stat_expr(mk_expr_call(print_rect[pr_rect_type],
  --                            mk_expr_id(tmp_var))),
  --  mk_stat_expr(mk_expr_call(print_rect[pr_rect_type],
  --                            ghost_rect_expr)),
  --  mk_stat_expr(mk_expr_call(c.printf,
  --                            mk_expr_constant("\n", rawstring)))
  --  })))
  loop_body:insert(mk_stat_expr(
    mk_expr_call(c.legion_domain_point_coloring_color_domain,
                 terralib.newlist { coloring_expr,
                                    color_expr,
                                    ghost_rect_expr })))

  stats:insert(
    mk_stat_for_list(color_symbol, color_space_expr, mk_block(loop_body)))
  stats:insert(
    mk_stat_var(gp_symbol, nil,
                mk_expr_partition(gp_type, color_space_expr, coloring_expr)))

  stats:insert(mk_stat_expr(mk_expr_call(c.legion_domain_point_coloring_destroy,
                                         coloring_expr)))

  if std.config["debug"] then
    local bounds = mk_expr_bounds_access(color_space_symbol)
    local my_bounds = mk_expr_bounds_access(
      mk_expr_colors_access(gp_symbol))
    stats:insert(mk_stat_expr(mk_expr_call(
        std.assert, terralib.newlist {
          mk_expr_binary("==", bounds, my_bounds),
          mk_expr_constant("color space bounds mismatch", rawstring)
        })))
  end

  caller_cx:update_constraint(sr_expr)

  return gp_symbol, stats
end

-- makes a loop as follows to create a coloring object:
--
--   for c in primary_partition.colors do
--     legion_domain_point_coloring_color_domain(coloring,
--       c, get_intersection(subregion.bounds,
--                           primary_partition[c].bounds)
--   var subset_partition = partition(*, subregion, coloring)
--
local function create_subset_partition(caller_cx, sr, pp, pparam)
  local color_space_symbol = caller_cx:find_color_space(pparam)
  assert(color_space_symbol)
  local color_space_expr = mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type)
  local color_expr = mk_expr_id(color_symbol)

  local sr_type = std.as_read(sr:gettype())
  local sr_index_type = sr_type:ispace().index_type
  local sr_rect_type = std.rect_type(sr_index_type)

  local sp_type =
    std.partition(pp:gettype().disjointness, sr, color_space_symbol:gettype())
  local sp_symbol = get_new_tmp_var(sp_type)
  local stats = terralib.newlist()

  local coloring_symbol = get_new_tmp_var(c.legion_domain_point_coloring_t)
  local coloring_expr = mk_expr_id(coloring_symbol)
  stats:insert(mk_stat_var(coloring_symbol, nil,
                           mk_expr_call(c.legion_domain_point_coloring_create)))

  local loop_body = terralib.newlist()
  local sr_expr = mk_expr_id(sr)
  local pp_expr = mk_expr_id(pp)
  local srpp_type = pp:gettype():subregion_dynamic()
  local srpp_expr = mk_expr_index_access(pp_expr, color_expr, srpp_type)
  local sr_bounds_expr = mk_expr_bounds_access(sr_expr)
  local srpp_bounds_expr = mk_expr_bounds_access(srpp_expr)
  local intersect_expr =
    mk_expr_call(get_intersection(sr_rect_type),
                 terralib.newlist { sr_bounds_expr,
                                    srpp_bounds_expr })
  --loop_body:insert(mk_stat_expr(mk_expr_call(print_rect[sr_rect_type],
  --                                           sr_bounds_expr)))
  --loop_body:insert(mk_stat_expr(mk_expr_call(print_rect[sr_rect_type],
  --                                           srpp_bounds_expr)))
  --loop_body:insert(mk_stat_expr(mk_expr_call(print_rect[sr_rect_type],
  --                                           intersect_expr)))
  --loop_body:insert(mk_stat_expr(mk_expr_call(c.printf,
  --                                           mk_expr_constant("\n", rawstring))))
  loop_body:insert(mk_stat_expr(
    mk_expr_call(c.legion_domain_point_coloring_color_domain,
                 terralib.newlist { coloring_expr,
                                    color_expr,
                                    intersect_expr })))

  stats:insert(
    mk_stat_for_list(color_symbol, color_space_expr, mk_block(loop_body)))
  stats:insert(
    mk_stat_var(sp_symbol, nil,
                mk_expr_partition(sp_type, color_space_expr, coloring_expr)))

  stats:insert(mk_stat_expr(mk_expr_call(c.legion_domain_point_coloring_destroy,
                                         coloring_expr)))

  if std.config["debug"] then
    local bounds = mk_expr_bounds_access(color_space_symbol)
    local my_bounds = mk_expr_bounds_access(
      mk_expr_colors_access(sp_symbol))
    stats:insert(mk_stat_expr(mk_expr_call(
        std.assert, terralib.newlist {
          mk_expr_binary("==", bounds, my_bounds),
          mk_expr_constant("color space bounds mismatch", rawstring)
        })))
  end

  caller_cx:add_primary_partition(sr, pparam, sp_symbol)
  caller_cx:update_constraint(srpp_expr)

  return sp_symbol, stats
end

local function create_indexspace_launch(parallelizable, caller_cx, expr, lhs)
  local info = parallelizable(expr)
  local task = info.task
  local task_cx = info.cx
  local color_space_symbol = caller_cx:find_color_space_by_call(expr)
  assert(color_space_symbol)
  local color_space_expr = mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type)
  local color_expr = mk_expr_id(color_symbol)

  local args = terralib.newlist()
  local stats = terralib.newlist()
  -- Find all primary partitions for region-typed arguments and
  -- replace them with sub-region accesses.
  for idx = 1, #expr.args do
    if task_cx:is_region_param(idx) then
      assert(expr.args[idx]:is(ast.typed.expr.ID))
      local region_symbol = expr.args[idx].value
      local partition_symbol =
        caller_cx:find_primary_partition_by_call(expr, region_symbol)
      local partition_expr = mk_expr_id(partition_symbol)
      assert(partition_symbol)
      local subregion_type = partition_symbol:gettype():subregion_dynamic()
      args:insert(
        mk_expr_index_access(partition_expr, color_expr, subregion_type))
      caller_cx:update_constraint(args[#args])
    else
      args:insert(expr.args[idx])
    end
  end
  -- Now push subregions of ghost partitions
  for idx = 1, #task_cx.stencils do
    local gp = caller_cx:find_ghost_partition(expr, task_cx.stencils[idx])
    local pr_type = gp:gettype().parent_region_symbol:gettype()
    local sr_type = gp:gettype():subregion_dynamic()
    args:insert(mk_expr_index_access(mk_expr_id(gp), color_expr, sr_type))
    caller_cx:update_constraint(args[#args])
  end
  -- Append the original region-typed arguments at the end
  for idx = 1, #expr.args do
    if task_cx:is_region_param(idx) then
      local bounds_symbol =
        caller_cx:find_bounds_symbol(expr.args[idx].value)
      args:insert(mk_expr_id(bounds_symbol))
    end
  end
  args:insert(color_expr)

  local expr = mk_expr_call(task, args)
  local call_stat
  if lhs then
    call_stat = mk_stat_reduce(task_cx.reduction_info.op, lhs, expr)
  else
    call_stat = mk_stat_expr(expr)
  end
  local index_launch =
    mk_stat_for_list(color_symbol, color_space_expr, mk_block(call_stat))
  if not std.config["flow-spmd"] then
    index_launch = index_launch {
      annotations = index_launch.annotations {
        parallel = ast.annotation.Demand
      }
    }
  end
  stats:insert(index_launch)
  return stats
end

-- #####################################
-- ## Task call transformer
-- #################

local function normalize_calls(parallelizable, call_stats)
  local function normalize(node, field)
    local stat_vars = terralib.newlist()
    local values = node[field]:map(function(value)
      if parallelizable(value) then
        local tmp_var = get_new_tmp_var(std.as_read(value.expr_type))
        stat_vars:insert(mk_stat_var(tmp_var, nil, value))
        return mk_expr_id(tmp_var)
      else
        return value
      end
    end)
    return stat_vars, node { [field] = values }
  end

  return function(node, continuation)
    if node:is(ast.typed.stat.Var) and
           data.any(unpack(node.values:map(parallelizable))) then
      if #node.values == 1 then
        call_stats[node] = true
        return node
      else
        local normalized, new_node = normalize(node, "values")
        normalized:map(function(stat) call_stats[stat] = true end)
        normalized:insert(new_node)
        -- This should be flattened later in the outer scope
        return normalized
      end

    elseif (node:is(ast.typed.stat.Assignment) or
            node:is(ast.typed.stat.Reduce)) and
           data.any(unpack(node.rhs:map(parallelizable))) then
      if node:is(ast.typed.stat.Reduce) and
         parallelizable(node.rhs[1]).cx.reduction_info.op == node.op and
         #node.rhs == 1 then
        call_stats[node] = true
        return node
      end
      local normalized, new_node = normalize(node, "rhs")
      normalized:map(function(stat) call_stats[stat] = true end)
      normalized:insert(new_node)
      -- This should be flattened later in the outer scope
      return normalized

    elseif node:is(ast.typed.stat.Expr) and parallelizable(node.expr) then
      call_stats[node] = true
      return node
    else
      return continuation(node, true)
    end
  end
end

local function collect_calls(cx, parallelizable)
  return function(node, continuation)
    if parallelizable(node) then
      cx:add_call(node)
    else
      if node:is(ast.typed.stat.If) or
         node:is(ast.typed.stat.Elseif) or
         node:is(ast.typed.stat.While) or
         node:is(ast.typed.stat.ForNum) or
         node:is(ast.typed.stat.ForList) or
         node:is(ast.typed.stat.Repeat) or
         node:is(ast.typed.stat.Block) then
        cx:push_scope()
        continuation(node, true)
        cx:pop_scope()
      elseif node:is(ast.typed.stat.Var) then
        for idx = 1, #node.symbols do
          local symbol_type = node.symbols[idx]:gettype()
          if std.is_region(symbol_type) then
            cx:add_region_decl(node.symbols[idx], node)
            -- Reserve symbols for metadata
            local bounds_symbol = get_new_tmp_var(
              std.rect_type(symbol_type:ispace().index_type))
            cx:add_bounds_symbol(node.symbols[idx], bounds_symbol)

            if node.values[idx]:is(ast.typed.expr.IndexAccess) then
              local partition_type =
                std.as_read(node.values[idx].value.expr_type)
              cx:add_parent_region(node.symbols[idx],
                partition_type.parent_region_symbol)
            end
          end
        end
        continuation(node, true)
      else
        continuation(node, true)
      end
    end
  end
end

local function insert_partition_creation(parallelizable, caller_cx, call_stats)
  return function(node, continuation)
    if caller_cx:get_call_exprs(node) then
      assert(node:is(ast.typed.stat.Var))
      local call_exprs_map = caller_cx:get_call_exprs(node)

      local region_symbols = data.filter(function(symbol)
        return std.is_region(symbol:gettype()) end, node.symbols)

      local stats = terralib.newlist {node}

      for idx = 1, #region_symbols do
        local bounds_symbol = caller_cx:find_bounds_symbol(region_symbols[idx])
        stats:insert(mk_stat_var(bounds_symbol, nil,
          mk_expr_bounds_access(region_symbols[idx])))
      end

      for idx = 1, #region_symbols do
        -- First, create necessary primary partitions
        for pparam, _ in call_exprs_map:items() do
          local parent_partition = nil
          local region = region_symbols[idx]
          local parent_region = caller_cx:find_parent_region(region)
          while parent_region do
            parent_partition =
              caller_cx:find_primary_partition(pparam, parent_region)
            if parent_partition then break end
            parent_region = caller_cx:find_parent_region(region)
          end
          if parent_partition then
            local partition_symbol, partition_stats =
              create_subset_partition(caller_cx, region, parent_partition, pparam)
            stats:insertall(partition_stats)
          else
            local partition_symbol, partition_stats =
              create_equal_partition(caller_cx, region, pparam)
            stats:insertall(partition_stats)
          end
        end
      end

      local region_symbol_set = {}
      region_symbols:map(function(region)
        region_symbol_set[region] = true
      end)
      -- Second, create ghost partitions
      for pparam, call_exprs in call_exprs_map:items() do
        local global_stencils = terralib.newlist()
        local global_stencil_indicies = {}
        for call_expr, _ in pairs(call_exprs) do
          local task_cx = parallelizable(call_expr).cx
          local param_arg_mapping =
            task_cx:make_param_arg_mapping(caller_cx, call_expr.args)
          for idx1 = 1, #task_cx.stencils do
            local orig_stencil = task_cx.stencils[idx1]
            local range = orig_stencil:range(param_arg_mapping)
            -- Analyze only the stencils that have a range being declared
            if region_symbol_set[range] then
              local stencil1 = orig_stencil:subst(param_arg_mapping)
              global_stencil_indicies[orig_stencil] = -1
              for idx2 = 1, #global_stencils do
                local stencil2 = global_stencils[idx2]
                if stencil2:range() == range then
                  -- If global stencil analysis is requested,
                  -- keep joining stencils whenever possible
                  if std.config["parallelize-global"] then
                    local joined =
                      stencil_analysis.join_stencil(stencil1, stencil2)
                    if joined then
                      global_stencils[idx2] = joined
                      global_stencil_indicies[orig_stencil] = idx2
                    end
                  -- Otherwise, just check the compatibility
                  -- so that partitions of the same shape are de-duplicated
                  else
                    local compatible =
                      stencil_analysis.stencils_compatible(stencil1, stencil2)
                    if compatible then
                      global_stencil_indicies[orig_stencil] = idx2
                    end
                  end
                end
              end
              if global_stencil_indicies[orig_stencil] == -1 then
                global_stencils:insert(stencil1)
                global_stencil_indicies[orig_stencil] = #global_stencils
              end
            end
          end
        end
        -- Now create all ghost partitions
        local ghost_partition_symbols = terralib.newlist()
        for idx = 1, #global_stencils do
          local stencil = global_stencils[idx]
          local range_symbol = stencil:range()
          local partition_symbol =
            caller_cx:find_primary_partition(pparam, range_symbol)
          assert(partition_symbol)
          local partition_symbol, partition_stats =
            create_image_partition(caller_cx, range_symbol, partition_symbol,
                                   stencil:index(), pparam)
          ghost_partition_symbols:insert(partition_symbol)
          stats:insertall(partition_stats)
        end
        -- Finally, record ghost partitions that each task call will use
        for call_expr, _ in pairs(call_exprs) do
          local orig_stencils = parallelizable(call_expr).cx.stencils
          for idx = 1, #orig_stencils do
            local stencil_idx = global_stencil_indicies[orig_stencils[idx]]
            if stencil_idx then
              local symbol = ghost_partition_symbols[stencil_idx]
              caller_cx:add_ghost_partition(call_expr, orig_stencils[idx], symbol)
            end
          end
        end
      end

      return stats
    elseif call_stats[node] then
      return node
    else
      return continuation(node, true)
    end
  end
end

local function transform_task_launches(parallelizable, caller_cx, call_stats)
  return function(node, continuation)
    if call_stats[node] then
      local stats = terralib.newlist()

      if node:is(ast.typed.stat.Expr) then
        stats:insertall(
          create_indexspace_launch(parallelizable, caller_cx, node.expr))
      else
        local expr
        local lhs
        if node:is(ast.typed.stat.Var) then
          expr = node.values[1]
          lhs = mk_expr_id(node.symbols[1],
                           std.rawref(&std.as_read(node.symbols[1]:gettype())))
        elseif node:is(ast.typed.stat.Reduce) then
          expr = node.rhs[1]
          lhs = node.lhs[1]
        else
          assert(false, "unreachable")
        end

        local reduction_op = parallelizable(expr).cx.reduction_info.op
        local lhs_type = std.as_read(lhs.expr_type)
        if node:is(ast.typed.stat.Var) then
          stats:insert(node {
            values = terralib.newlist { mk_expr_constant(
              std.reduction_op_init[reduction_op][lhs_type], lhs_type)}
          })
        end
        stats:insertall(
          create_indexspace_launch(parallelizable, caller_cx, expr, lhs))
      end

      return stats
    else
      return continuation(node, true)
    end
  end
end

function parallelize_task_calls.top_task(global_cx, node)
  local function parallelizable(node)
    if not node:is(ast.typed.expr.Call) then return false end
    local fn = node.fn.value
    return not node.annotations.parallel:is(ast.annotation.Forbid) and
           std.is_task(fn) and global_cx[fn]
  end

  -- Return if there is no parallelizable task call
  local found = false
  ast.traverse_node_continuation(function(node, continuation)
    if parallelizable(node) then found = true
    else continuation(node, true) end end, node)
  if not found then return node end

  -- First, normalize all task calls so that task calls are either
  -- their own statements or single variable declarations.
  local body = node.body
  local call_stats = {}
  local normalized =
    ast.flatmap_node_continuation(
      normalize_calls(parallelizable, call_stats),
      body)

  -- Second, group task calls by reaching region declarations
  local caller_cx = caller_context.new(node.prototype:get_constraints())
  ast.traverse_node_continuation(
    collect_calls(caller_cx, parallelizable),
    normalized)

  -- Third, insert partition creation code when necessary
  local partition_created =
    ast.flatmap_node_continuation(
      insert_partition_creation(parallelizable, caller_cx, call_stats),
      normalized)

  -- Finally, replace single task launches into indexspace launches
  local parallelized =
    ast.flatmap_node_continuation(
      transform_task_launches(parallelizable, caller_cx, call_stats),
      partition_created)

  return node { body = parallelized }
end

-- #####################################
-- ## Normalizer for region accesses
-- #################

local function rewrite_metadata_access(task_cx)
  return function(node, continuation)
    if node:is(ast.typed.expr.FieldAccess) and
       node.field_name == "bounds" and
       std.is_region(std.as_read(node.value.expr_type)) then
      assert(node.value:is(ast.typed.expr.ID))
      local metadata_params =
        task_cx:find_metadata_parameters(node.value.value)
      if metadata_params then
        return mk_expr_id(metadata_params.bounds)
      else
        return continuation(node, true)
      end
    else
      return continuation(node, true)
    end
  end
end

-- normalize field accesses; e.g.
-- a = b.f
-- ~>  t = b.f
--     a = t
-- also, collect all field accesses for stencil analysis
--       and track the return value

local normalizer_context = {}

normalizer_context.__index = normalizer_context

function normalizer_context.new(loop_var)
  local cx = {
    var_decls = terralib.newlist {symbol_table.new_global_scope({})},
    loop_var = loop_var,
  }
  return setmetatable(cx, normalizer_context)
end

function normalizer_context:push_scope()
  self.var_decls:insert(self:current_scope():new_local_scope())
end

function normalizer_context:pop_scope()
  self.var_decls[#self.var_decls] = nil
end

function normalizer_context:current_scope()
  return self.var_decls[#self.var_decls]
end

function normalizer_context:add_decl(symbol, node)
  self:current_scope():insert(nil, symbol, node)
end

function normalizer_context:find_decl(symbol)
  return self:current_scope():safe_lookup(symbol)
end

function normalizer_context:get_loop_var()
  return self.loop_var
end

local function is_centered(node)
  -- TODO: Index expressions might have been casted to index type,
  --       which can demote them to be uncentered, even though they aren't.
  if node:is(ast.typed.expr.IndexAccess) and
     (std.is_bounded_type(node.index.expr_type) or
      std.is_bounded_type(std.as_read(node.index.expr_type))) then
    return true
  elseif node:is(ast.typed.expr.Deref) then
    if std.is_bounded_type(node.value.expr_type) then
      return true
    end
  elseif node:is(ast.typed.expr.FieldAccess) then
    return is_centered(node.value)
  else
    return false
  end
end

local function find_field_accesses(accesses)
  return function(node, continuation)
    if node:is(ast.typed.expr) and
       std.is_ref(node.expr_type) and
       not is_centered(node) then
      accesses:insert(node)
    else
      continuation(node, true)
    end
  end
end

function normalize_accesses.expr(normalizer_cx)
  return function(node, continuation)
    if node:is(ast.typed.expr.ID) then
      return normalizer_cx:find_decl(node.value) or node
    else
      return continuation(node, true)
    end
  end
end

local function lift_all_accesses(task_cx, normalizer_cx, accesses, stat)
  if #accesses == 0 then return stat end
  local stats = terralib.newlist()
  local rewrites = {}
  local loop_var = normalizer_cx:get_loop_var()
  for idx = 1, #accesses do
    local access = accesses[idx]
    -- Make stencil metadata for later steps
    local normalized =
      ast.map_node_continuation(normalize_accesses.expr(normalizer_cx), access)
    local index_access = extract_index_access_expr(normalized)
    local index_expr = index_access.index
		if index_expr:is(ast.typed.expr.Cast) then
		  index_expr = index_expr.arg
		end
    if not (index_expr:is(ast.typed.expr.ID) and
            index_expr.value == loop_var) then
      local region_symbol = index_access.value.value
      local field_path = access.expr_type.field_path
      local stencil = Stencil {
        region = region_symbol,
        index = index_expr,
        range = loop_var:gettype().bounds_symbols[1],
        fields = { [field_path:hash()] = field_path },
      }
      task_cx:add_access(access, stencil)

      local tmp_symbol = get_new_tmp_var(std.as_read(access.expr_type))
      local stat = mk_stat_var(tmp_symbol, nil, access)
      task_cx:record_stat_requires_case_split(stat)
      stats:insert(stat)
      rewrites[access] = mk_expr_id(tmp_symbol)
    end
  end
  stat = ast.map_node_continuation(function(node, continuation)
    if rewrites[node] then
      return rewrites[node]
    else
      return continuation(node, true)
    end
  end, stat)
  stats:insert(stat)
  return stats
end

function normalize_accesses.stat(task_cx, normalizer_cx)
  return function(node, continuation)
    if node:is(ast.typed.stat.Var) then
      for idx = 1, #node.symbols do
        local symbol = node.symbols[idx]
        local symbol_type = symbol:gettype()
        if std.is_index_type(symbol_type) or
           std.is_bounded_type(symbol_type) then
          -- TODO: variables can be assigned later
          assert(node.values[idx])
          normalizer_cx:add_decl(symbol, node.values[idx])
        end
      end
      local accesses = terralib.newlist()
      ast.traverse_node_continuation(
        find_field_accesses(accesses), node.values)
      return lift_all_accesses(task_cx, normalizer_cx, accesses, node)
    elseif node:is(ast.typed.stat.Assignment) or
           node:is(ast.typed.stat.Reduce) then
      local accesses_lhs = terralib.newlist()
      local accesses_rhs = terralib.newlist()
      ast.traverse_node_continuation(
        find_field_accesses(accesses_lhs), node.lhs)
      ast.traverse_node_continuation(
        find_field_accesses(accesses_rhs), node.rhs)
      assert(data.all(unpack(accesses_lhs:map(is_centered))))
      return lift_all_accesses(task_cx, normalizer_cx, accesses_rhs, node)
    else
      return continuation(node, true)
    end
  end
end

function normalize_accesses.stat_for_list(task_cx, node)
  -- Rewrite any region metadata access with reserved parameter symbol
  node = ast.map_node_continuation(
    rewrite_metadata_access(task_cx), node)

  local normalizer_cx = normalizer_context.new(node.symbol)
  node = ast.flatmap_node_continuation(
    normalize_accesses.stat(task_cx, normalizer_cx), node)

  return node
end

function normalize_accesses.top_task_body(task_cx, node)
  return node {
    stats = node.stats:map(function(node)
      if node:is(ast.typed.stat.ForList) then
        return normalize_accesses.stat_for_list(task_cx, node)
      else
        return node
      end
    end),
  }
end

-- #####################################
-- ## Analyzer for scalar reductions
-- #################

function reduction_analysis.top_task(task_cx, node)
  if node.return_type:isunit() then return end
  local return_value = node.body.stats[#node.body.stats].value
  if not return_value:is(ast.typed.expr.ID) then
    assert(return_value:is(ast.typed.expr.Constant))
  end
  local reduction_var = return_value.value
  local init_expr
  local reduction_op
  local decl

  ast.traverse_node_continuation(function(node, continuation)
    if node:is(ast.typed.stat.Var) then
      for idx = 1, #node.symbols do
        if node.symbols[idx] == reduction_var then
          assert(node.values[idx] ~= nil)
          init_expr = node.values[idx]
          decl = {
            stat = node,
            index = idx,
          }
          break
        end
      end
    elseif node:is(ast.typed.stat.Reduce) then
      node.lhs:map(function(expr)
        if expr:is(ast.typed.expr.ID) and expr.value == reduction_var then
          assert(reduction_op == nil or reduction_op == node.op)
          reduction_op = node.op
        end
      end)
      continuation(node.rhs, true)
    elseif node:is(ast.typed.expr.ID) then
      assert(node.value ~= reduction_var)
    elseif node:is(ast.typed.stat.Return) then
    else continuation(node, true) end
  end, node)

  assert(init_expr)
  -- TODO: Task might pass through a scalar value
  assert(reduction_op ~= nil)
  -- TODO: convert reductions with - or / into fold-and-reduces
  assert(reduction_op ~= "-" and reduction_op ~= "/")
  task_cx.reduction_info = {
    op = reduction_op,
    symbol = reduction_var,
    declaration = decl,
  }
end

local function extract_constant_offsets(n)
  assert(n:is(ast.typed.expr.Ctor) and
         data.all(n.fields:map(function(field)
           return field.expr_type.type == "integer"
         end)))
  local num_nonzeros = 0
  local offsets = terralib.newlist()
  for idx = 1, #n.fields do
    if n.fields[idx].value:is(ast.typed.expr.Constant) then
      offsets:insert(n.fields[idx].value.value)
    elseif n.fields[idx].value:is(ast.typed.expr.Unary) and
           n.fields[idx].value.op == "-" and
           n.fields[idx].value.rhs:is(ast.typed.expr.Constant) then
      offsets:insert(-n.fields[idx].value.rhs.value)
    else
      assert(false)
    end
    if offsets[#offsets] ~= 0 then num_nonzeros = num_nonzeros + 1 end
  end
  return offsets, num_nonzeros
end

-- #####################################
-- ## Stencil analyzer
-- #################

-- (a, b, c) -->  (a, 0, 0), (0, b, 0), (0, 0, c),
--                (a, b, 0), (0, b, c), (a, 0, c),
--                (a, b, c)
function stencil_analysis.explode_expr(cx, expr)
  -- Index should be either e +/- c or (e +/- c) % r.bounds
  -- where e is for-list loop symbol and r is a region
  if expr:is(ast.typed.expr.Binary) then
    if expr.op == "%" then
      assert(expr.rhs:is(ast.typed.expr.ID) and
             std.is_rect_type(expr.rhs.expr_type))
      return stencil_analysis.explode_expr(cx, expr.lhs):map(function(lhs)
        return expr { lhs = lhs }
      end)
    elseif expr.op == "+" or expr.op == "-" then
      if expr.rhs:is(ast.typed.expr.Ctor) then
        local convert = function(n) return n end
        if expr.op == "-" then
          convert = function(n)
            if n.value.value == 0 then return n
            else
              return n {
                value = n.value {
                  value = -n.value.value,
                },
              }
            end
          end
        end
        return stencil_analysis.explode_expr(cx, expr.rhs):map(function(rhs)
          return expr {
            op = "+",
            rhs = rhs { fields = rhs.fields:map(convert) },
          }
        end)
      elseif expr.rhs:is(ast.typed.expr.Constant) and
             expr.rhs.expr_type.type == "integer" then
        if expr.op == "-" then
          return terralib.newlist { expr {
            op = "+",
            rhs = expr.rhs { value = -expr.rhs.value }
          }}
        else
          return terralib.newlist { expr }
        end
      else
        assert(false)
      end
    else
      assert(false)
    end
  elseif expr:is(ast.typed.expr.Ctor) then
    local constant_type = expr.fields[1].expr_type
    local offsets, num_nonzeros = extract_constant_offsets(expr)
    local num_exploded_offsets = 2 ^ num_nonzeros - 1
    local exploded_offsets = terralib.newlist()
    for idx = 1, num_exploded_offsets do
      local l = terralib.newlist()
      local enc = idx
      for oidx = 1, #offsets do
        if offsets[oidx] == 0 then
          l:insert(0)
        else
          if enc % 2 == 0 then l:insert(0)
          else l:insert(offsets[oidx]) end
          enc = math.floor(enc / 2)
        end
      end
      exploded_offsets:insert(l)
    end
    return exploded_offsets:map(function(offsets)
      return expr {
        fields = offsets:map(function(offset)
          return mk_expr_ctor_list_field_constant(offset, constant_type)
        end)
      }
    end)
  else
    assert(false)
  end
end

local function arg_join(v, n1, n2, field)
  if v == nil then return nil
  elseif n1[field] == v then return n1
  elseif n2[field] == v then return n2
  else return n1 { [field] = v } end
end

-- Find the lub of two stencils (stencils are partially ordered)
-- returns 1) s1 |_| s2 if s1 and s2 has lub
--         2) nil if s1 <> s2
function stencil_analysis.join_stencil(s1, s2)
  if Stencil.is_stencil(s1) and Stencil.is_stencil(s2) then
    assert(s1:range() == s2:range())
    local binder = s1:index():binder()
    local joined =
      stencil_analysis.join_stencil(s1:index():expr(),
                                    s2:index()(binder))
    if joined then
      -- TODO: region symbols and fields should also be merged here
      return s1:replace_index(Lambda {
        binder = binder,
        expr = joined,
      })
    else
      return nil
    end
  elseif ast.is_node(s1) and ast.is_node(s1) and s1:is(s2:type()) then
    if s1:is(ast.typed.expr.ID) then
      return (s1.value == s2.value) and s1
    elseif s1:is(ast.typed.expr.Binary) then
      if s1.op ~= s2.op then return nil
      elseif s1.op == "%" then
        return arg_join(stencil_analysis.join_stencil(s1.rhs, s2.rhs) and
                        stencil_analysis.join_stencil(s1.lhs, s2.lhs),
                        s1, s2, "lhs")
      elseif s1.op == "+" then
        assert(s1.lhs:is(ast.typed.expr.ID) and
               s2.lhs:is(ast.typed.expr.ID) and
               s1.lhs.value == s2.lhs.value)
        return arg_join(stencil_analysis.join_stencil(s1.rhs, s2.rhs),
                        s1, s2, "rhs")
      else
        assert(false)
      end

    elseif s1:is(ast.typed.expr.Ctor) then
      local constant_type = s1.fields[1].expr_type
      local offsets1 = extract_constant_offsets(s1)
      local offsets2 = extract_constant_offsets(s2)
      local joined_offsets = terralib.newlist()
      -- 0: initial, 1: offsets1 >= offsets2, 2: offsets1 < offsets2,
      -- -1: joined
      local argmax_all = 0
      if #offsets1 ~= #offsets2 then return nil end
      for idx = 1, #offsets1 do
        local o1, o2 = offsets1[idx], offsets2[idx]
        if o1 == o2 and o1 == 0 then joined_offsets:insert(0)
        elseif o1 ~= o2 and (o1 * o2 == 0) then return nil
        elseif o1 * o2 < 0 then return nil
        else
          local argmax
          if math.abs(o1) >= math.abs(o2) then
            argmax = 1
            joined_offsets:insert(o1)
          else
            argmax = 2
            joined_offsets:insert(o2)
          end

          if argmax_all == 0 then argmax_all = argmax
          elseif argmax_all ~= argmax then argmax_all = -1 end
        end
      end
      if argmax_all == 1 then return s1
      elseif argmax_all == 2 then return s2
      else
        assert(argmax_all == -1)
        return s1 {
          fields = joined_offsets:map(function(offset)
            return mk_expr_ctor_list_field_constant(offset, constant_type)
          end)
        }
      end
    elseif s1:is(ast.typed.expr.Constant) then
      local o1 = s1.value
      local o2 = s2.value
      if o1 == o2 and o1 == 0 then return s1
      elseif o1 * o2 < 0 then return nil
      elseif math.abs(o1) > math.abs(o2) then return s1
      else return s2 end
    else
      return nil
    end
  else
    return nil
  end
end

function stencil_analysis.stencils_compatible(s1, s2)
  if Stencil.is_stencil(s1) and Stencil.is_stencil(s2) then
    local binder = s1:index():binder()
    return s1:range() == s2:range() and
           stencil_analysis.stencils_compatible(s1:index():expr(),
                                                s2:index()(binder))
  elseif ast.is_node(s1) and ast.is_node(s1) and s1:is(s2:type()) then
    if s1:is(ast.typed.expr.ID) then
      return s1.value == s2.value
    elseif s1:is(ast.typed.expr.Binary) then
      return s1.op == s2.op and
             stencil_analysis.stencils_compatible(s1.lhs, s2.lhs) and
             stencil_analysis.stencils_compatible(s1.rhs, s2.rhs)
    elseif s1:is(ast.typed.expr.Ctor) then
      local o1 = extract_constant_offsets(s1)
      local o2 = extract_constant_offsets(s2)
      if #o1 ~= #o2 then return false end
      for idx = 1, #o1 do
        if o1[idx] ~= o2[idx] then return false end
      end
      return true
    elseif s1:is(ast.typed.expr.Constant) then
      return s1.value == s2.value
    else
      return false
    end
  else
    return false
  end
end

function stencil_analysis.top(cx)
  for _, access_info in pairs(cx.field_accesses) do
    local stencil = access_info.stencil
    access_info.exploded_stencils:insertall(
      stencil_analysis.explode_expr(cx, stencil:index()):map(function(expr)
        return stencil:replace_index(expr)
      end))

    for i = 1, #access_info.exploded_stencils do
      access_info.ghost_indices:insert(-1)
      for j = 1, #cx.stencils do
        local s1 = access_info.exploded_stencils[i]:index()
        local s2 = cx.stencils[j]:index()
        local joined_stencil = stencil_analysis.join_stencil(s1, s2)
        if joined_stencil then
          cx.stencils[j] = cx.stencils[j]:replace_index(joined_stencil)
          cx.stencils[j]:add_fields(access_info.exploded_stencils[i]:fields())
          access_info.ghost_indices[i] = j
          break
        end
      end
      if access_info.ghost_indices[i] == -1 then
        cx.stencils:insert(access_info.exploded_stencils[i])
        access_info.ghost_indices[i] = #cx.stencils
      end
    end
  end
end

local function make_new_region_access(region_expr, index_expr, field)
  local region_symbol = region_expr.value
  local region_type = std.as_read(region_expr.expr_type)
  local index_type = std.as_read(index_expr.expr_type)
  local expr = mk_expr_index_access(region_expr, index_expr,
    std.ref(index_type(region_type:fspace(), region_symbol)))
  for idx = 1, #field do
    local new_field = expr.expr_type.field_path .. data.newtuple(field[idx])
    expr = mk_expr_field_access(expr, field[idx],
      std.ref(expr.expr_type, unpack(new_field)))
  end
  return expr
end

-- #####################################
-- ## Task parallelizer
-- #################

function parallelize_tasks.stat(task_cx)
  return function(node, continuation)
    if task_cx:stat_requires_case_split(node) then
      assert(node:is(ast.typed.stat.Var) and
             #node.symbols == 1 and #node.types == 1 and
             #node.values == 1 and
             (node.values[1]:is(ast.typed.expr.FieldAccess) or
              node.values[1]:is(ast.typed.expr.Deref) or
              node.values[1]:is(ast.typed.expr.IndexAccess)))

      -- Case split for each region access:
      -- var x = r[f(e)] =>
      --   var x; var p = f(e)
      --   do
      --     if x <= r.bounds then x = r[p]
      --     elseif p <= ghost1.bounds then x = ghost1[p]
      --     elseif p <= ghost2.bounds then x = ghost2[p]
      --     ...

      local stats = terralib.newlist()
      -- Remove RHS of a variable declaration as it depends on case analysis
      stats:insert(node { values = terralib.newlist() })
      -- Cache index calculation for several comparisions later
      local access_info = task_cx.field_accesses[node.values[1]]
      local index_expr = extract_index_expr(node.values[1])
      -- If index expressions is complex, cache it before the comparisons
      if not index_expr:is(ast.typed.expr.ID) then
        local index_symbol = get_new_tmp_var(std.as_read(index_expr.expr_type))
        stats:insert(mk_stat_var(index_symbol, nil, index_expr))
        index_expr =
          mk_expr_id(index_symbol, std.rawref(&index_symbol:gettype()))
      end
      -- Populate body of case analysis
      local result_symbol = node.symbols[1]
      local result_expr =
        mk_expr_id(result_symbol, std.rawref(&result_symbol:gettype()))
      local case_split_if
      local elseif_blocks
      for idx = 0, #access_info.ghost_indices do
        local region_symbol
        local field
        if idx == 0 then
          region_symbol = access_info.stencil:region()
          assert(Stencil.is_singleton(access_info.stencil))
          field = access_info.stencil:fields()[1]
        else
          region_symbol = task_cx.ghost_symbols[access_info.ghost_indices[idx]]
          assert(Stencil.is_singleton(access_info.exploded_stencils[idx]))
          field = access_info.exploded_stencils[idx]:fields()[1]
        end

        local region_type = std.rawref(&region_symbol:gettype())
        local region_id_expr = mk_expr_id(region_symbol, region_type)
        local bounds_expr = mk_expr_bounds_access(region_id_expr)
        local cond = mk_expr_binary("<=", index_expr, bounds_expr)

        local region_access =
          make_new_region_access(region_id_expr, index_expr, field)
        local result_assignment = mk_stat_assignment(result_expr, region_access)
        if idx == 0 then
          case_split_if = mk_stat_if(cond, result_assignment)
          elseif_blocks = case_split_if.elseif_blocks
        else
          elseif_blocks:insert(mk_stat_elseif(cond, result_assignment))
        end
      end
      assert(case_split_if)
      if std.config["debug"] then
        case_split_if.else_block.stats:insertall(terralib.newlist {
          --mk_stat_expr(mk_expr_call(print_point[index_symbol:gettype()],
          --             terralib.newlist {
          --               index_expr
          --             })),
          mk_stat_expr(mk_expr_call(std.assert,
                       terralib.newlist {
                         mk_expr_constant(false, bool),
                         mk_expr_constant("unreachable", rawstring)
                       })),
        })
      end
      stats:insert(case_split_if)
      return stats
    else
      return continuation(node, true)
    end
  end
end

function parallelize_tasks.stat_for_list(task_cx, node)
  return ast.flatmap_node_continuation(parallelize_tasks.stat(task_cx), node)
end

function parallelize_tasks.top_task_body(task_cx, node)
  local stats = data.flatmap(
    function(stat)
      if stat:is(ast.typed.stat.ForList) then
        return parallelize_tasks.stat_for_list(task_cx, stat)
      elseif task_cx.reduction_info ~= nil and
             task_cx.reduction_info.declaration.stat == stat then
        local red_decl_index = task_cx.reduction_info.declaration.index

        local symbols = terralib.newlist()
        local types = terralib.newlist()
        local values = terralib.newlist()

        for idx = 1, #stat.symbols do
          if idx ~= red_decl_index then
            symbols:insert(stat.symbols[idx])
            types:insert(stat.types[idx])
            values:insert(stat.values[idx])
          end
        end

        local stats = terralib.newlist()
        if #symbols > 0 then
          stats:insert(stat {
            symbols = symbols,
            types = types,
            values = values,
          })
        end

        local red_var = stat.symbols[red_decl_index]
        local red_var_expr =
          mk_expr_id(red_var, std.rawref(&red_var:gettype()))
        stats:insert(mk_stat_var(red_var, stat.types[red_decl_index]))

        local cond = mk_expr_binary(
          "==", mk_expr_id(task_cx:get_task_point_symbol()),
          mk_expr_zeros(task_cx:get_task_point_symbol():gettype()))
        local if_stat = mk_stat_if(
          cond, mk_stat_assignment(
            red_var_expr, stat.values[red_decl_index]))
        local init =
          std.reduction_op_init[task_cx.reduction_info.op][red_var:gettype()]
        assert(init ~= nil)
        if_stat.else_block.stats:insert(
          mk_stat_assignment(
            red_var_expr, mk_expr_constant(init, red_var:gettype())))
        stats:insert(if_stat)

        return stats
      else
        return stat
      end
    end, node.stats)

  return node { stats = stats }
end

function parallelize_tasks.top_task(global_cx, node)
  -- Analyze loops in the task
  local task_cx = parallel_task_context.new_task_scope(node.params)
  local normalized = normalize_accesses.top_task_body(task_cx, node.body)
  reduction_analysis.top_task(task_cx, node)
  stencil_analysis.top(task_cx)

  -- Now make a new task AST node
  local task_name = node.name .. data.newtuple("parallelized")
  local prototype = std.newtask(task_name)
  local params = terralib.newlist()
  -- Existing region-typed parameters will now refer to the subregions
  -- passed by indexspace launch. this will avoid rewriting types in AST nodes
  params:insertall(node.params)
  -- each stencil corresponds to one ghost region
  for idx = 1, #task_cx.stencils do
    local ghost_symbol =
      copy_region_symbol(task_cx.stencils[idx]:region(),
                         "__ghost" .. tostring(idx))
    task_cx.ghost_symbols:insert(ghost_symbol)
    params:insert(mk_task_param(task_cx.ghost_symbols[idx]))
  end
  -- Append parameters reserved for the metadata of original region parameters
  task_cx:insert_metadata_parameters(params)

  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), node.return_type, false)
  prototype:settype(task_type)
  prototype:set_param_symbols(
    params:map(function(param) return param.symbol end))
  local region_universe = {}
  local privileges = terralib.newlist()
  local coherence_modes = data.new_recursive_map(1)
  --node.prototype:get_coherence_modes():map_list(function(region, map)
  --    print(region)
  --  map:map_list(function(field_path, v)
  --    coherence_modes[region][field_path] = true
  --  end)
  --end)
  privileges:insertall(node.prototype:getprivileges())
  for region, _ in pairs(node.prototype:get_region_universe()) do
    region_universe[region] = true
  end
  -- FIXME: Workaround for the current limitation in SPMD transformation
  local field_set = {}
  for idx = 1, #task_cx.stencils do
		task_cx.stencils[idx]:fields():map(function(field) field_set[field] = true end)
  end
  local fields = terralib.newlist()
  for field, _ in pairs(field_set) do fields:insert(field) end

  for idx = 1, #task_cx.stencils do
		local region = task_cx.ghost_symbols[idx]
		--local fields = task_cx.stencils[idx]:fields()
    -- TODO: handle reductions on ghost regions
    privileges:insert(fields:map(function(field)
      return std.privilege(std.reads, region, field)
    end))
    --coherence_modes[region][field_path] = std.exclusive
    region_universe[region:gettype()] = true
  end
	prototype:setprivileges(privileges)
  prototype:set_coherence_modes(coherence_modes)
  prototype:set_flags(node.flags)
  prototype:set_conditions(node.conditions)
  prototype:set_param_constraints(node.constraints)
  prototype:set_constraints(node.constraints)
  prototype:set_region_universe(region_universe)

  local parallelized = parallelize_tasks.top_task_body(task_cx, normalized)
  local task_ast = ast.typed.top.Task {
    name = task_name,
    params = params,
    return_type = node.return_type,
    privileges = privileges,
    coherence_modes = coherence_modes,
    flags = node.flags,
    conditions = node.conditions,
    constraints = node.constraints,
    body = parallelized,
    config_options = ast.TaskConfigOptions {
      leaf = false,
      inner = false,
      idempotent = false,
    },
    region_divergence = false,
    prototype = prototype,
    annotations = ast.default_annotations(),
    span = node.span,
  }

  -- Hack: prevents parallelized verions from going through parallelizer again
  global_cx[prototype] = {}
  local task_ast_optimized = passes.optimize(task_ast)
  local task_code = passes.codegen(task_ast_optimized, true)

  return task_code, task_cx
end

function parallelize_tasks.entry(node)
  if node:is(ast.typed.top.Task) then
    if global_context[node.prototype] then return node end
    if node.annotations.parallel:is(ast.annotation.Demand) then
      check_parallelizable.top_task(node)
      local task_name = node.name
      local new_task_code, cx = parallelize_tasks.top_task(global_context, node)
      local info = {
        task = new_task_code,
        cx = cx,
      }
      global_context[node.prototype] = info
      return node
    else
      return parallelize_task_calls.top_task(global_context, node)
    end
  else
    return node
  end
end

parallelize_tasks.pass_name = "parallelize_tasks"

return parallelize_tasks
