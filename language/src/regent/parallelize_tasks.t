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

-- Regent Auto-parallelizer

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local data = require("common/data")
local std = require("regent/std")
local report = require("common/report")
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
local function get_ghost_rect_body(res, sz, root, r, s, polarity, f)
  local acc = function(expr) return `([expr].[f]) end
  if f == nil then acc = function(expr) return expr end end
  return quote
    if [acc(`([polarity].__ptr))] == 0 then
      [acc(`([res].lo.__ptr))] = [acc(`([r].lo.__ptr))]
      [acc(`([res].hi.__ptr))] = [acc(`([r].hi.__ptr))]
    elseif [acc(`([polarity].__ptr))] > 0 then
      if [acc(`([s].lo.__ptr))] > [acc(`([s].hi.__ptr))] then
        [acc(`([res].lo.__ptr))] = ([acc(`([r].hi.__ptr))] + 1) % [acc(`([sz].__ptr))]
        [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
      -- if the stencil access is completely off, then we can just leave it as it is
      elseif [acc(`([r].hi.__ptr))] < [acc(`([s].lo.__ptr))] or
             [acc(`([r].lo.__ptr))] > [acc(`([s].hi.__ptr))] then
        [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
        [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
      else
        [acc(`([res].lo.__ptr))] = [acc(`([r].hi.__ptr))] + 1
        [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
      end
    elseif [acc(`([polarity].__ptr))] < 0 then
      if [acc(`([s].lo.__ptr))] > [acc(`([s].hi.__ptr))] then
        [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
        [acc(`([res].hi.__ptr))] = ([acc(`([r].lo.__ptr))] - 1 + [acc(`([sz].__ptr))]) % [acc(`([sz].__ptr))]
      elseif [acc(`([r].hi.__ptr))] < [acc(`([s].lo.__ptr))] or
             [acc(`([r].lo.__ptr))] > [acc(`([s].hi.__ptr))] then
        [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
        [acc(`([res].hi.__ptr))] = [acc(`([s].hi.__ptr))]
      else
        [acc(`([res].lo.__ptr))] = [acc(`([s].lo.__ptr))]
        [acc(`([res].hi.__ptr))] = [acc(`([r].lo.__ptr))] - 1
      end
    end
  end
end

local terra is_zero(p : c.legion_domain_point_t)
  var dim = p.dim
  if dim == 0 then dim = 1 end
  for i = 0, dim do
    if p.point_data[i] ~= 0 then return false end
  end
  return true
end

-- If all stencil accesses fall into the private region,
-- we do not need to calculate the size of the ghost region
local function clear_unnecessary_polarity(root, r, polarity, f)
  local acc = function(expr) return `([expr].[f]) end
  if f == nil then acc = function(expr) return expr end end
  return quote
    if [acc(`([root].lo.__ptr))] == [acc(`([r].lo.__ptr))] and
       [acc(`([root].hi.__ptr))] == [acc(`([r].hi.__ptr))] then
      [acc(`([polarity].__ptr))] = 0
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
  [std.rect1d] = terra(root : std.rect1d, r : std.rect1d, s : std.rect1d, polarity : std.int1d) : std.rect1d
    var sz = root:size()
    var diff_rect : std.rect1d
    diff_rect.lo.__ptr = 0
    diff_rect.hi.__ptr = -1
    [clear_unnecessary_polarity(root, r, polarity)]
    -- If the ghost region is not necessary at all for this stencil,
    -- make a dummy region with only one element.
    if polarity == [std.int1d:zero()] then return diff_rect end
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity)]
    [bounds_checks(diff_rect)]
    return diff_rect
  end,
  [std.rect2d] = terra(root : std.rect2d, r : std.rect2d, s : std.rect2d, polarity : std.int2d) : std.rect2d
    var sz = root:size()
    var diff_rect : std.rect2d
    diff_rect.lo.__ptr.x, diff_rect.lo.__ptr.y = 0, 0
    diff_rect.hi.__ptr.x, diff_rect.hi.__ptr.y = -1, -1
    [clear_unnecessary_polarity(root, r, polarity, "x")]
    [clear_unnecessary_polarity(root, r, polarity, "y")]
    -- If the ghost region is not necessary at all for this stencil,
    -- make a dummy region with only one element.
    if polarity == [std.int2d:zero()] then return diff_rect end
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity, "x")]
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity, "y")]
    [bounds_checks(diff_rect)]
    return diff_rect
  end,
  [std.rect3d] = terra(root : std.rect3d, r : std.rect3d, s : std.rect3d, polarity : std.int3d) : std.rect3d
    var sz = root:size()
    var diff_rect : std.rect3d
    diff_rect.lo.__ptr.x, diff_rect.lo.__ptr.y, diff_rect.lo.__ptr.z = 0, 0, 0
    diff_rect.hi.__ptr.x, diff_rect.hi.__ptr.y, diff_rect.hi.__ptr.z = -1, -1, -1
    [clear_unnecessary_polarity(root, r, polarity, "x")]
    [clear_unnecessary_polarity(root, r, polarity, "y")]
    [clear_unnecessary_polarity(root, r, polarity, "z")]
    -- If the ghost region is not necessary at all for this stencil,
    -- make a dummy region with only one element.
    if polarity == [std.int3d:zero()] then return diff_rect end
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity, "x")]
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity, "y")]
    [get_ghost_rect_body(diff_rect, sz, root, r, s, polarity, "z")]
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

local function shallow_copy(tbl)
  local new_tbl = {}
  for k, v in pairs(tbl) do
    new_tbl[k] = v
  end
  return new_tbl
end

local function parse_dop(line)
  local factors = terralib.newlist()
  assert(string.len(line) > 0)
  local position = 0
  while position do
    local new_position = string.find(line, ",", position+1)
    factors:insert(tonumber(string.sub(line, position+1, new_position and new_position-1)))
    position = new_position
  end
  return factors
end

-- #####################################
-- ## Utilities for AST node manipulation
-- #################

local tmp_var_id = 0

local function get_new_tmp_var(ty, name)
  assert(type(name) == "string")
  local sym = std.newsymbol(ty, name .. "__t".. tostring(tmp_var_id))
  tmp_var_id = tmp_var_id + 1
  return sym
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
    else
      assert(false)
    end
    if offsets[#offsets] ~= 0 then num_nonzeros = num_nonzeros + 1 end
  end
  return offsets, num_nonzeros
end

local function extract_polarity(node)
  if Lambda.is_lambda(node) then
    return extract_polarity(node:expr())
  elseif node:is(ast.typed.expr.Binary) then
    if node.op == "%" then
      return extract_polarity(node.lhs)
    elseif node.op == "+" then
      return extract_polarity(node.rhs)
    elseif node.op == "-" then
      return extract_polarity(node.rhs):map(function(c)
        return -c
      end)
    else
      assert(false)
    end
  elseif node:is(ast.typed.expr.Ctor) then
    return extract_constant_offsets(node):map(function(c)
      if c > 0 then return 1
      elseif c < 0 then return -1
      else return 0 end
    end)
  elseif node:is(ast.typed.expr.Constant) and
         node.expr_type.type == "integer" then
    return terralib.newlist { node.value }
  elseif std.is_ref(node.expr_type) then
    local dim = std.as_read(node.expr_type).dim
    local polarity = terralib.newlist {}
    for idx = 1, dim do polarity:insert(math.huge) end
    return polarity
  else
    assert(false)
  end
end

-- #####################################
-- ## Stencil and Lambda
-- #################

do
  local lambda = {}

  lambda.__index = lambda

  function lambda.__call(self, arg)
    local arg_type
    if std.is_symbol(arg) then
      arg_type = arg:gettype()
    else
      arg_type = std.as_read(arg.expr_type)
    end
    if std.is_bounded_type(arg_type) then
      arg_type = arg_type.index_type
    end

    local binders = self:all_binders()
    local expr = self:body()

    local binder
    local new_binders = terralib.newlist()
    for idx = 1, #binders do
      local binder_type = binders[idx]:gettype()
      if std.is_bounded_type(binder_type) then
        binder_type = binder_type.index_type
      end
      if std.type_eq(binder_type, arg_type) then
        binder = binders[idx]
      else
        new_binders:insert(binders[idx])
      end
    end
    assert(binder)
    if std.is_symbol(arg) then
      expr = rewrite_symbol(expr, binder, arg)
    else
      expr = rewrite_expr_id(expr, binder, arg)
    end
    if #new_binders == 0 then
      return expr
    else
      return Lambda {
        binders = new_binders,
        expr = expr,
      }
    end
  end

  function lambda:expr()
    return self.__expr
  end

  function lambda:body()
    local expr = self:expr()
    if Lambda.is_lambda(expr) then
      return expr:body()
    else
      return expr
    end
  end

  function lambda:binder()
    return self.__binder
  end

  function lambda:all_binders(l)
    l = l or terralib.newlist()
    l:insert(self:binder())
    if Lambda.is_lambda(self:expr()) then
      return self:expr():all_binders(l)
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
      expr_str = ast_util.render(self:expr())
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
      if Stencil.is_stencil(range) then
        range = range:subst(mapping)
      else
        range = mapping[range] or range
      end
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

  function stencil:polarity()
    return self.__polarity
  end

  function stencil:is_static()
    for idx = 1, #self.__polarity do
      if self.__polarity[idx] == math.huge then
        return false
      end
    end
    return true
  end

  function stencil:depth(d)
    local depth = d or 1
    if Stencil.is_stencil(self:range()) then
      return self:range():depth(depth + 1)
    end
    return depth
  end

  function stencil:is_nested()
    return Stencil.is_stencil(self:range())
  end

  function stencil:has_field(field)
    return self.__fields[field:hash()] ~= nil
  end

  function stencil:add_field(field)
    if not self:has_field(field) then
      self.__fields[field:hash()] = field
    end
    return self
  end

  function stencil:add_fields(fields)
    fields:map(function(field) self:add_field(field) end)
    return self
  end

  function stencil:replace_index(index)
    return Stencil {
      region = self:region(),
      index = index,
      range = self:range(),
      fields = shallow_copy(self.__fields),
    }
  end

  function stencil:replace_range(range)
    return Stencil {
      region = self:region(),
      index = self:index(),
      range = range,
      fields = shallow_copy(self.__fields),
    }
  end

  function stencil:to_expr()
    assert(Stencil.is_singleton(self))
    local region = self:region()
    local expr = self:index()
    expr = ast_util.mk_expr_index_access(
      ast_util.mk_expr_id(region, std.rawref(&region:gettype())),
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
      index_str = ast_util.render(self:index())
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
      __polarity = extract_polarity(args.index),
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
  for idx = 1, #params do
    if std.is_region(params[idx].param_type) then
      local symbol = params[idx].symbol
      local region_type = symbol:gettype()
      local name = symbol:getname()
      local bounds_symbol = false
      if not region_type:is_opaque() then
        bounds_symbol = std.newsymbol(
          std.rect_type(region_type:ispace().index_type),
          "__" .. name .. "_bounds")
      end
      region_params[idx] = {
        region = symbol,
        bounds = bounds_symbol,
      }
      region_param_map[symbol] = region_params[idx]
      region_param_indices:insert(idx)
    end
  end

  local cx = {}
  cx.region_params = region_params
  cx.region_param_map = region_param_map
  cx.region_param_indices = region_param_indices
  cx.task_point_symbol = get_new_tmp_var(c.legion_domain_point_t, "__point")
  cx.field_accesses = {}
  cx.field_access_stats = {}
  cx.stencils = terralib.newlist()
  cx.ghost_symbols = {}
  cx.use_primary = {}
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
    if region_param.bounds then
      params:insert(ast_util.mk_task_param(region_param.bounds))
    end
  end
  params:insert(ast_util.mk_task_param(self:get_task_point_symbol()))
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
      assert(mapping[orig_param.bounds] or orig_param.region:gettype():is_opaque())
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

function parallel_param.new(params)
  local tbl = {
    __dop = false,
    __primary_partitions = false,
    __constraints = false,
    __hash = false,
  }
  if params.dop ~= nil then tbl.__dop = params.dop end
  if params.primary_partitions ~= nil then
    tbl.__primary_partitions = params.primary_partitions
  end
  if params.constraints ~= nil then
    tbl.__constraints = params.constraints
  end
  return setmetatable(tbl, parallel_param)
end

function parallel_param:hash()
  if not self.__hash then
    local str = ""
    if self.__dop then str = str .. "dop" .. tostring(self.__dop) end
    if self.__primary_partitions then
      for idx = 1, #self.__primary_partitions do
        --FIXME: need to distinguish between different partitions for the same region
        str = str .. "#" .. tostring(self.__primary_partitions[idx])
      end
    end
    if self.__constraints then
      for idx = 1, #self.__constraints do
        str = str .. "#" .. ast_util.render(self.__constraints[idx])
      end
    end
    self.__hash = str
  end
  return self.__hash
end

function parallel_param:find_primary_partition_for(region_type)
  if not self.__primary_partitions then return false end
  for idx = 1, #self.__primary_partitions do
    if std.type_eq(self.__primary_partitions[idx]:parent_region(),
                   region_type) then
      return self.__primary_partitions[idx]
    end
  end
  return false
end

function parallel_param:find_ghost_partition_for(region_type)
  if not self.__constraints then return false end
  for idx = 1, #self.__constraints do
    -- XXX: This assumes that the hint is only on an image partition
    assert(self.__constraints[idx].lhs:is(ast.typed.expr.Image))
    assert(self.__constraints[idx].rhs:is(ast.typed.expr.ID))
    local ghost_partition_type = std.as_read(self.__constraints[idx].rhs.expr_type)
    if std.type_eq(ghost_partition_type:parent_region(), region_type) then
      return ghost_partition_type
    end
  end
  return false
end

function parallel_param:dop()
  assert(self.__dop ~= false)
  return self.__dop
end

local PRIMARY = setmetatable({}, { __tostring = function(self) return "PRIMARY" end })
local SUBSET = setmetatable({}, { __tostring = function(self) return "SUBSET" end })

caller_context.__index = caller_context

function caller_context.new(constraints)
  local param = parallel_param.new({ dop = std.config["parallelize-dop"] })
  local cx = {
    __param_stack = terralib.newlist { param },
    -- No need to track scopes for declarations
    -- since symbols are already disambiguated
    __region_decls = {},
    __partition_decls = {},
    -- region type -> region symbol
    __region_symbols = data.newmap(),
    -- partition type -> partition symbol
    __partition_symbols = data.newmap(),
    -- Map from region or partition declaration to sets of call expressions
    -- grouped by parallelization parameters
    __call_exprs_by_decl = {},
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
    -- parallelization parameter -> max dimension
    __max_dim = data.newmap(),
    -- list of hints in a `__parallelize_with` block -> parallelization parameter
    __param_by_hints = {},
    __decl_order = {},
    __current_order = 0,
  }
  return setmetatable(cx, caller_context)
end

function caller_context:push_scope(p)
  -- TODO: this should come from bishop
  local param = p or self.__param_stack[#self.__param_stack] or
                parallel_param.new({ dop = std.config["parallelize-dop"] })
  self.__param_stack:insert(param)
end

function caller_context:pop_scope()
  self.__param_stack[#self.__param_stack] = nil
end

function caller_context:add_region_decl(region_type, stat)
  self.__region_decls[region_type] = stat
  self.__decl_order[stat] = self.__current_order
  self.__current_order = self.__current_order + 1
end

function caller_context:add_region_symbol(region_type, region_symbol)
  self.__region_symbols[region_type] = region_symbol
end

function caller_context:find_region_symbol(region_type)
  assert(self.__region_symbols[region_type] ~= nil)
  return self.__region_symbols[region_type]
end

function caller_context:has_region_symbol(region_type)
  return self.__region_symbols[region_type] ~= nil
end

function caller_context:add_partition_decl(partition_type, stat)
  self.__partition_decls[partition_type] = stat
  self.__decl_order[stat] = self.__current_order
  self.__current_order = self.__current_order + 1
end

function caller_context:find_partition_decl(partition_type)
  assert(self.__partition_decls[partition_type] ~= nil)
  return self.__partition_decls[partition_type]
end

function caller_context:add_partition_symbol(partition_type, partition_symbol)
  self.__partition_symbols[partition_type] = partition_symbol
end

function caller_context:find_partition_symbol(partition_type)
  assert(self.__partition_symbols[partition_type] ~= nil)
  return self.__partition_symbols[partition_type]
end

function caller_context:has_partition_symbol(partition_type)
  return self.__partition_symbols[partition_type] ~= nil
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

function caller_context:add_parallel_param_for_hints(hints, param)
  self.__param_by_hints[hints] = param
end

function caller_context:find_parallel_param_for_hints(hints)
  assert(self.__param_by_hints[hints] ~= nil)
  return self.__param_by_hints[hints]
end

function caller_context:set_max_dim(param, dim)
  if self.__max_dim[param] == nil then
    self.__max_dim[param] = dim
  else
    self.__max_dim[param] = math.max(self.__max_dim[param], dim)
  end
end

function caller_context:find_max_dim(param)
  return self.__max_dim[param] or 0
end

function caller_context:add_call(expr, task_cx)
  local param = self.__param_stack[#self.__param_stack]
  local mapping = task_cx:make_param_arg_mapping(self, expr.args)
  for idx = 1, #expr.args do
    local expr_type = std.as_read(expr.args[idx].expr_type)
    if std.is_region(expr_type) then
      local decl
      local request

      local region_type = std.as_read(expr.args[idx].expr_type)
      local region_symbol = self:find_region_symbol(region_type)

      if self:has_parent_region(region_symbol) then
        local parent_region_symbol = region_symbol
        while self:has_parent_region(parent_region_symbol) do
          parent_region_symbol = self:find_parent_region(parent_region_symbol)
        end
        local primary_partition_type_from_hint =
          param:find_primary_partition_for(parent_region_symbol:gettype())
        if primary_partition_type_from_hint then
          local subregion_decl = self.__region_decls[region_type]
          local partition_decl = self.__partition_decls[primary_partition_type_from_hint]
          -- FIXME: Hacky syntactic check to find a right place to create partition.
          --        Need a more sophisticated def-use analysis.
          --        For example, this can break the code:
          --        var p_equal, s = partition(equal, ...), p_interior[0]
          local function appear_earlier(n1, n2)
            local o1 = self.__decl_order[n1]
            local o2 = self.__decl_order[n2]
            assert(o1 ~= nil)
            assert(o2 ~= nil)
            return o1 < o2
          end
          if appear_earlier(subregion_decl, partition_decl) then
            decl = partition_decl
          else
            decl = subregion_decl
          end

          request = { type = SUBSET, region = region_symbol }
        else
          decl = self.__region_decls[region_type]
          request = { type = PRIMARY, region = region_symbol }
        end
      else
        decl = self.__region_decls[region_type]
        request = { type = PRIMARY, region = region_symbol }

        local primary_partition_type_from_hint =
          param:find_primary_partition_for(expr_type)

        if primary_partition_type_from_hint then
          decl = self.__partition_decls[primary_partition_type_from_hint]

          local stencils_with_no_hints = data.filter(function(stencil)
            local stencil_region_type = stencil:region(mapping):gettype()
            return std.type_eq(stencil_region_type, expr_type) and
                   (stencil:is_static() or
                    not param:find_ghost_partition_for(stencil_region_type))
          end, task_cx.stencils)
          if #stencils_with_no_hints == 0 then decl = nil
          else
            local nested = nil
            local range = nil
            for idx = 1, #stencils_with_no_hints do
              assert(range == nil or range == stencils_with_no_hints[idx]:range())
              assert(nested == nil or nested == stencils_with_no_hints[idx]:is_nested())
              range = stencils_with_no_hints[idx]:range()
              nested = stencils_with_no_hints[idx]:is_nested()
            end
            if nested then
              local ghost_partition_type_from_hint =
                param:find_ghost_partition_for(range:region(mapping))
              if ghost_partition_type_from_hint then
                decl = self.__partition_decls[ghost_partition_type_from_hint]
              end
            end
          end
        end
      end

      if decl ~= nil then
        if self.__call_exprs_by_decl[decl] == nil then
          self.__call_exprs_by_decl[decl] = data.newmap()
        end
        if self.__call_exprs_by_decl[decl][param] == nil then
          self.__call_exprs_by_decl[decl][param] = {}
        end
        -- TODO: This collision case is not yet thoroughly understood
        if self.__call_exprs_by_decl[decl][param][expr] == nil or
           (self.__call_exprs_by_decl[decl][param][expr].type == PRIMARY and
            request.type == SUBSET) then
          self.__call_exprs_by_decl[decl][param][expr] = request
        end
        -- TODO: We don't support multiple structured regions with different dimensions
        assert(self:find_max_dim(param) == 0 or
               expr_type:ispace().dim == 0 or
               self:find_max_dim(param) == expr_type:ispace().dim)
        self:set_max_dim(param, expr_type:ispace().dim)
      end
    end
  end
  if self.__parallel_params[expr] == nil then
    self.__parallel_params[expr] = param
  end
end

function caller_context:get_call_exprs(decl)
  return self.__call_exprs_by_decl[decl]
end

function caller_context:add_primary_partition(region_symbol, param, partition_symbol)
  if self.__primary_partitions[region_symbol] == nil then
    self.__primary_partitions[region_symbol] = data.newmap()
  end
  self.__primary_partitions[region_symbol][param] = partition_symbol
end

function caller_context:add_ghost_partition(call, stencil, partition_symbol)
  if self.__ghost_partitions[call] == nil then
    self.__ghost_partitions[call] = {}
  end
  assert(self.__ghost_partitions[call][stencil] == nil)
  self.__ghost_partitions[call][stencil] = partition_symbol
end

function caller_context:add_parent_region(region_symbol, parent_region_symbol)
  self.__parent_region[region_symbol] = parent_region_symbol
end

function caller_context:has_parent_region(region_symbol)
  return self.__parent_region[region_symbol] ~= nil
end

function caller_context:get_parent_region(region_symbol)
  assert(self.__parent_region[region_symbol] ~= nil)
  return self.__parent_region[region_symbol]
end

function caller_context:add_color_space(param, color)
  assert(self.__color_spaces[param] == nil)
  self.__color_spaces[param] = color
end

function caller_context:find_parallel_parameters(call)
  assert(self.__parallel_params[call] ~= nil)
  return self.__parallel_params[call]
end

function caller_context:has_primary_partition(param, region_symbol)
  return self.__primary_partitions[region_symbol] ~= nil and
         self.__primary_partitions[region_symbol][param] ~= nil
end

function caller_context:find_primary_partition(param, region_symbol)
  assert(self.__primary_partitions[region_symbol][param])
  return self.__primary_partitions[region_symbol][param]
end

function caller_context:find_primary_partition_by_call(call, region_symbol)
  local param = self:find_parallel_parameters(call)
  assert(self.__primary_partitions[region_symbol][param])
  return self.__primary_partitions[region_symbol][param]
end

function caller_context:has_primary_partition_by_call(call, region_symbol)
  local param = self:find_parallel_parameters(call)
  return self.__primary_partitions[region_symbol] ~= nil and
         self.__primary_partitions[region_symbol][param] ~= nil
end

function caller_context:find_ghost_partition(call, stencil)
  assert(self.__ghost_partitions[call][stencil])
  return self.__ghost_partitions[call][stencil]
end

function caller_context:has_ghost_partition(call, stencil)
  return self.__ghost_partitions[call] ~= nil and
         self.__ghost_partitions[call][stencil] ~= nil
end

function caller_context:find_parent_region(region_symbol)
  return self.__parent_region[region_symbol]
end

function caller_context:find_color_space(param)
  return self.__color_spaces[param]
end

function caller_context:find_color_space_by_call(call)
  local param = self:find_parallel_parameters(call)
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
  -- 9. tasks should not have multiple independent region parameters:
  --    e.g. task not_good(r : region(ispace(int2d), ...),
  --                       s : region(ispace(int3d), ...))
  --    however, this can be used:
  --    e.g. task good(r : region(ispace(int2d), ...),
  --                   s : region(ispace(int3d), some_fs(r)))
end

-- #####################################
-- ## Code generation utilities for partition creation and indexspace launch
-- #################

local terra factorize(dop : int, factors : &&int, num_factors : &int)
  @num_factors = 0
  @factors = [&int](std.c.malloc([sizeof(int)] * dop))
  while dop > 1 do
    var factor = 1
    while factor <= dop do
      if factor ~= 1 and dop % factor == 0 then
        (@factors)[@num_factors] = factor
        @num_factors = @num_factors + 1
        dop = dop / factor
        break
      end
      factor = factor + 1
    end
  end

  for i = 0, @num_factors - 1 do
    for j = i + 1, @num_factors do
      if (@factors)[i] < (@factors)[j] then
        (@factors)[j], (@factors)[i]  = (@factors)[i], (@factors)[j]
      end
    end
  end
end

local compute_extent = {
  [std.int2d] = terra(dop : int, rect : std.rect2d) : std.int2d
    var factors : &int, num_factors : int
    factorize(dop, &factors, &num_factors)
    var sz = rect:size()
    var extent : std.int2d = std.int2d { [std.int2d.impl_type] { 1, 1 } }
    for idx = 0, num_factors do
      if sz.__ptr.x <= sz.__ptr.y then
        extent.__ptr.y = extent.__ptr.y * factors[idx]
        sz.__ptr.y = (sz.__ptr.y + factors[idx] - 1) / factors[idx]
      else
        extent.__ptr.x = extent.__ptr.x * factors[idx]
        sz.__ptr.x = (sz.__ptr.x + factors[idx] - 1) / factors[idx]
      end
    end
    std.c.free(factors)
    return extent
  end,
  [std.int3d] = terra(dop : int, rect : std.rect3d) : std.int3d
    var factors : &int, num_factors : int
    factorize(dop, &factors, &num_factors)
    var sz = rect:size()
    var sz_remain : int[3], extent : int[3]
    sz_remain[0], sz_remain[1], sz_remain[2] = sz.__ptr.x, sz.__ptr.y, sz.__ptr.z
    for k = 0, 3 do extent[k] = 1 end
    for idx = 0, num_factors do
      var next_max = 0
	    var max_sz = 0
      for k = 2, -1, -1 do
        if max_sz < sz_remain[k] then
          next_max = k
          max_sz = sz_remain[k]
        end
      end
      extent[next_max] = extent[next_max] * factors[idx]
      sz_remain[next_max] = (sz_remain[next_max] + factors[idx] - 1) / factors[idx]
    end
    std.c.free(factors)
    return std.int3d { [std.int3d.impl_type] { extent[0], extent[1], extent[2] } }
  end,
}

local function create_equal_partition(caller_cx, region_symbol, pparam)
  local region_type = region_symbol:gettype()
  local index_type = region_type:ispace().index_type
  local dim = caller_cx:find_max_dim(pparam)
  local factors = parse_dop(pparam:dop())
  local extent_expr
  local color_type
  if #factors == 1 then
    assert(dim == index_type.dim,
      "degree-of-parallelism must be given for all dimensions if regions don't " ..
      "match in their dimensions")
    if dim <= 1 then
      extent_expr = ast_util.mk_expr_constant(factors[1], int)
      color_type = std.int1d
    else
      extent_expr = ast_util.mk_expr_call(
        compute_extent[index_type], terralib.newlist {
          ast_util.mk_expr_constant(factors[1], int),
          ast_util.mk_expr_bounds_access(ast_util.mk_expr_id(region_symbol))
        })
      color_type = index_type
    end
  else
    assert(dim == 0 or #factors == dim,
      "degree-of-parallelism does not match with the dimensions of regions, " ..
      "expected: " .. tostring(dim) .. ", got " .. tostring(#factors))
    extent_expr = ast_util.mk_expr_ctor(factors:map(function(f)
      return ast_util.mk_expr_constant(f, int)
    end))
    if #factors == 2 then color_type = std.int2d
    elseif #factors == 3 then color_type = std.int3d
    else assert(false) end
  end

  local base_name = (region_symbol:hasname() and region_symbol:getname()) or ""

  local color_space_expr = ast_util.mk_expr_ispace(color_type, extent_expr)
  local partition_type =
    std.partition(disjoint, region_symbol, color_space_expr.expr_type)
  local partition_symbol = get_new_tmp_var(partition_type, base_name .. "__equal")

  local stats = terralib.newlist { ast_util.mk_stat_var(
    partition_symbol,
    nil,
    ast_util.mk_expr_partition_equal(partition_type, color_space_expr))
  }

  caller_cx:add_primary_partition(region_symbol, pparam, partition_symbol)

  local color_space_symbol = caller_cx:find_color_space(pparam)
  if color_space_symbol == nil then
    local base_name = (partition_symbol:hasname() and partition_symbol:getname()) or ""
    color_space_symbol = get_new_tmp_var(partition_type:colors(), base_name .. "__colors")
    local colors_expr = ast_util.mk_expr_colors_access(partition_symbol)
    stats:insert(ast_util.mk_stat_var(color_space_symbol, nil, colors_expr))
    caller_cx:add_color_space(pparam, color_space_symbol)
  elseif std.config["debug"] then
    local bounds = ast_util.mk_expr_bounds_access(color_space_symbol)
    local my_bounds = ast_util.mk_expr_bounds_access(
      ast_util.mk_expr_colors_access(partition_symbol))
    stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(
        std.assert, terralib.newlist {
          ast_util.mk_expr_binary("==", bounds, my_bounds),
          ast_util.mk_expr_constant("color space bounds mismatch", rawstring)
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
  local base_name = (pr:hasname() and pr:getname()) or ""

  local color_space_symbol = caller_cx:find_color_space(pparam)
  assert(color_space_symbol)
  local color_space_expr = ast_util.mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type, base_name .. "__color")
  local color_expr = ast_util.mk_expr_id(color_symbol)

  local pr_type = std.as_read(pr:gettype())
  local pr_index_type = pr_type:ispace().index_type
  local pr_rect_type = std.rect_type(pr_index_type)

  -- TODO: disjointness check
  local gp_type = std.partition(std.aliased, pr, color_space_symbol:gettype())
  local gp_symbol = get_new_tmp_var(
    gp_type,
    base_name .. "__gp" .. "__" .. stencil:polarity():concat("_"))
  local stats = terralib.newlist()

  local coloring_symbol = get_new_tmp_var(
    c.legion_domain_point_coloring_t,
    base_name .. "__coloring")
  local coloring_expr = ast_util.mk_expr_id(coloring_symbol)
  stats:insert(ast_util.mk_stat_var(coloring_symbol, nil,
                           ast_util.mk_expr_call(c.legion_domain_point_coloring_create)))

  local loop_body = terralib.newlist()
  local pr_expr = ast_util.mk_expr_id(pr)
  local pp_expr = ast_util.mk_expr_id(pp)
  local sr_type = pp:gettype():subregion_dynamic()
  local sr_expr = ast_util.mk_expr_index_access(pp_expr, color_expr, sr_type)
  local pr_bounds_expr = ast_util.mk_expr_bounds_access(pr_expr)
  local sr_bounds_expr = ast_util.mk_expr_bounds_access(sr_expr)
  local sr_lo_expr = ast_util.mk_expr_field_access(sr_bounds_expr, "lo", pr_index_type)
  local sr_hi_expr = ast_util.mk_expr_field_access(sr_bounds_expr, "hi", pr_index_type)
  local shift_lo_expr = stencil:index()(sr_lo_expr)
  if Lambda.is_lambda(shift_lo_expr) then
    shift_lo_expr = shift_lo_expr(pr_bounds_expr)
  end
  local shift_hi_expr = stencil:index()(sr_hi_expr)
  if Lambda.is_lambda(shift_hi_expr) then
    shift_hi_expr = shift_hi_expr(pr_bounds_expr)
  end
  local polarity_expr =
    ast_util.mk_expr_ctor(stencil:polarity():map(function(c)
      return ast_util.mk_expr_constant(c, int)
    end))
  local tmp_var = get_new_tmp_var(pr_rect_type, base_name .. "__rect")
  loop_body:insert(ast_util.mk_stat_var(tmp_var, nil,
    ast_util.mk_expr_ctor(terralib.newlist {shift_lo_expr, shift_hi_expr})))
  local ghost_rect_expr =
    ast_util.mk_expr_call(get_ghost_rect[pr_rect_type],
                 terralib.newlist { pr_bounds_expr,
                                    sr_bounds_expr,
                                    ast_util.mk_expr_id(tmp_var),
                                    polarity_expr })
  --loop_body:insert(ast_util.mk_stat_block(ast_util.mk_block(terralib.newlist {
  --  ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[pr_rect_type],
  --                            pr_bounds_expr)),
  --  ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[pr_rect_type],
  --                            sr_bounds_expr)),
  --  ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[pr_rect_type],
  --                            ast_util.mk_expr_id(tmp_var))),
  --  ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[pr_rect_type],
  --                            ghost_rect_expr)),
  --  ast_util.mk_stat_expr(ast_util.mk_expr_call(c.printf,
  --                            ast_util.mk_expr_constant("\n", rawstring)))
  --  })))
  loop_body:insert(ast_util.mk_stat_expr(
    ast_util.mk_expr_call(c.legion_domain_point_coloring_color_domain,
                 terralib.newlist { coloring_expr,
                                    color_expr,
                                    ghost_rect_expr })))

  stats:insert(
    ast_util.mk_stat_for_list(color_symbol, color_space_expr, ast_util.mk_block(loop_body)))
  stats:insert(
    ast_util.mk_stat_var(gp_symbol, nil,
                ast_util.mk_expr_partition(gp_type, color_space_expr, coloring_expr)))

  stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(c.legion_domain_point_coloring_destroy,
                                         coloring_expr)))

  if std.config["debug"] then
    local bounds = ast_util.mk_expr_bounds_access(color_space_symbol)
    local my_bounds = ast_util.mk_expr_bounds_access(
      ast_util.mk_expr_colors_access(gp_symbol))
    stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(
        std.assert, terralib.newlist {
          ast_util.mk_expr_binary("==", bounds, my_bounds),
          ast_util.mk_expr_constant("color space bounds mismatch", rawstring)
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
  local base_name = (sr:hasname() and sr:getname()) or ""

  local color_space_symbol = caller_cx:find_color_space(pparam)
  assert(color_space_symbol)
  local color_space_expr = ast_util.mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type, "__color")
  local color_expr = ast_util.mk_expr_id(color_symbol)

  local sr_type = std.as_read(sr:gettype())
  local sr_index_type = sr_type:ispace().index_type
  local sr_rect_type = std.rect_type(sr_index_type)

  local sp_type =
    std.partition(pp:gettype().disjointness, sr, color_space_symbol:gettype())
  local sp_symbol = get_new_tmp_var(sp_type, base_name .. "__sp")
  local stats = terralib.newlist()

  local coloring_symbol = get_new_tmp_var(c.legion_domain_point_coloring_t, base_name .. "__coloring")
  local coloring_expr = ast_util.mk_expr_id(coloring_symbol)
  stats:insert(ast_util.mk_stat_var(coloring_symbol, nil,
                           ast_util.mk_expr_call(c.legion_domain_point_coloring_create)))

  local loop_body = terralib.newlist()
  local sr_expr = ast_util.mk_expr_id(sr)
  local pp_expr = ast_util.mk_expr_id(pp)
  local srpp_type = pp:gettype():subregion_dynamic()
  local srpp_expr = ast_util.mk_expr_index_access(pp_expr, color_expr, srpp_type)
  local sr_bounds_expr = ast_util.mk_expr_bounds_access(sr_expr)
  local srpp_bounds_expr = ast_util.mk_expr_bounds_access(srpp_expr)
  local intersect_expr =
    ast_util.mk_expr_call(get_intersection(sr_rect_type),
                 terralib.newlist { sr_bounds_expr,
                                    srpp_bounds_expr })
  --loop_body:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[sr_rect_type],
  --                                           sr_bounds_expr)))
  --loop_body:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[sr_rect_type],
  --                                           srpp_bounds_expr)))
  --loop_body:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(print_rect[sr_rect_type],
  --                                           intersect_expr)))
  --loop_body:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(c.printf,
  --                                           ast_util.mk_expr_constant("\n", rawstring))))
  loop_body:insert(ast_util.mk_stat_expr(
    ast_util.mk_expr_call(c.legion_domain_point_coloring_color_domain,
                 terralib.newlist { coloring_expr,
                                    color_expr,
                                    intersect_expr })))

  stats:insert(
    ast_util.mk_stat_for_list(color_symbol, color_space_expr, ast_util.mk_block(loop_body)))
  stats:insert(
    ast_util.mk_stat_var(sp_symbol, nil,
                ast_util.mk_expr_partition(sp_type, color_space_expr, coloring_expr)))

  stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(c.legion_domain_point_coloring_destroy,
                                         coloring_expr)))

  if std.config["debug"] then
    local bounds = ast_util.mk_expr_bounds_access(color_space_symbol)
    local my_bounds = ast_util.mk_expr_bounds_access(
      ast_util.mk_expr_colors_access(sp_symbol))
    stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(
        std.assert, terralib.newlist {
          ast_util.mk_expr_binary("==", bounds, my_bounds),
          ast_util.mk_expr_constant("color space bounds mismatch", rawstring)
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
  local color_space_expr = ast_util.mk_expr_id(color_space_symbol)
  local color_type =
    color_space_symbol:gettype().index_type(color_space_symbol)
  local color_symbol = get_new_tmp_var(color_type, "__color")
  local color_expr = ast_util.mk_expr_id(color_symbol)

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
      local partition_expr = ast_util.mk_expr_id(partition_symbol)
      assert(partition_symbol)
      local subregion_type = partition_symbol:gettype():subregion_dynamic()
      args:insert(
        ast_util.mk_expr_index_access(partition_expr, color_expr, subregion_type))
      caller_cx:update_constraint(args[#args])
    else
      args:insert(expr.args[idx])
    end
  end
  -- Now push subregions of ghost partitions
  for idx = 1, #task_cx.stencils do
    if not task_cx.use_primary[task_cx.stencils[idx]] then
      local gp = caller_cx:find_ghost_partition(expr, task_cx.stencils[idx])
      local pr_type = gp:gettype().parent_region_symbol:gettype()
      local sr_type = gp:gettype():subregion_dynamic()
      args:insert(ast_util.mk_expr_index_access(ast_util.mk_expr_id(gp), color_expr, sr_type))
      caller_cx:update_constraint(args[#args])
    end
  end
  -- Append the original region-typed arguments at the end
  for idx = 1, #expr.args do
    if task_cx:is_region_param(idx) then
      local bounds_symbol =
        caller_cx:find_bounds_symbol(expr.args[idx].value)
      if bounds_symbol then
        args:insert(ast_util.mk_expr_id(bounds_symbol))
      end
    end
  end
  args:insert(color_expr)

  local expr = ast_util.mk_expr_call(task, args)
  local call_stat
  if lhs then
    call_stat = ast_util.mk_stat_reduce(task_cx.reduction_info.op, lhs, expr)
  else
    call_stat = ast_util.mk_stat_expr(expr)
  end
  local index_launch =
    ast_util.mk_stat_for_list(color_symbol, color_space_expr, ast_util.mk_block(call_stat))
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
    local value = node[field]
    if parallelizable(value) then
      local tmp_var = get_new_tmp_var(std.as_read(value.expr_type), "") -- FIXME: need semantic name
      stat_vars:insert(ast_util.mk_stat_var(tmp_var, nil, value))
      value = ast_util.mk_expr_id(tmp_var)
    end
    return stat_vars, node { [field] = value }
  end

  return function(node, continuation)
    if node:is(ast.typed.stat.Var) and parallelizable(node.value) then
      call_stats[node] = true
      return node

    elseif (node:is(ast.typed.stat.Assignment) or node:is(ast.typed.stat.Reduce)) and
           parallelizable(node.rhs) then
      if node:is(ast.typed.stat.Reduce) and
         parallelizable(node.rhs).cx.reduction_info.op == node.op then
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

local function add_metadata_declarations(cx)
  return function(node, continuation)
    if node:is(ast.typed.stat.Var) and
       std.is_region(node.symbol:gettype()) and
       not node.symbol:gettype():is_opaque() then
      local stats = terralib.newlist {node}
      local region_symbol = node.symbol

      -- Reserve symbols for metadata
      local base_name = (region_symbol:hasname() and region_symbol:getname()) or ""
      local bounds_symbol = get_new_tmp_var(
        std.rect_type(region_symbol:gettype():ispace().index_type),
        base_name .. "__bounds")
      stats:insert(ast_util.mk_stat_var(bounds_symbol, nil,
        ast_util.mk_expr_bounds_access(region_symbol)))
      cx:add_bounds_symbol(region_symbol, bounds_symbol)

      return stats
    else
      return continuation(node, true)
    end
  end
end

local function collect_calls(cx, parallelizable)
  return function(node, continuation)
    local is_parallelizable = parallelizable(node)
    if is_parallelizable then
      cx:add_call(node, is_parallelizable.cx)
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
        local symbol_type = node.symbol:gettype()
        if std.is_region(symbol_type) or std.is_partition(symbol_type) then
          local region_type
          if std.is_region(symbol_type) then
            cx:add_region_decl(symbol_type, node)
            cx:add_region_symbol(symbol_type, node.symbol)
            region_type = symbol_type
          elseif std.is_partition(symbol_type) then
            cx:add_partition_decl(symbol_type, node)
            cx:add_partition_symbol(symbol_type, node.symbol)
            region_type = symbol_type:parent_region()
          else
            assert(false, "unreachable")
          end
          assert(region_type ~= nil)

          if node.value:is(ast.typed.expr.IndexAccess) then
            local partition_type =
              std.as_read(node.value.value.expr_type)
            cx:add_parent_region(node.symbol,
              partition_type.parent_region_symbol)
          end
        end
        continuation(node, true)
      elseif node:is(ast.typed.stat.ParallelizeWith) then
        local hints_id = data.filter(function(hint)
          return hint:is(ast.typed.expr.ID) end, node.hints)
        local hints_primary = data.filter(function(hint)
          return std.is_partition(std.as_read(hint.expr_type)) end, hints_id):map(function(hint)
          return std.as_read(hint.expr_type) end)
        local hints_color_space = data.filter(function(hint)
          return std.is_ispace(std.as_read(hint.expr_type)) end, hints_id)

        local hints_constraint = data.filter(function(hint)
          return hint:is(ast.typed.expr.ParallelizerConstraint) end, node.hints)
        local param = parallel_param.new({
          primary_partitions = hints_primary,
          constraints = hints_constraint,
        })
        cx:push_scope(param)
        if #hints_color_space > 0 then
          assert(#hints_color_space == 1, "only one color space should be given")
          cx:add_color_space(param, hints_color_space[#hints_color_space].value)
        end
        cx:add_parallel_param_for_hints(node.hints, param)
        for idx = 1, #hints_primary do
          local partition_type = hints_primary[idx]
          assert(std.is_partition(partition_type))
          cx:add_primary_partition(cx:find_region_symbol(partition_type:parent_region()),
                                   param, cx:find_partition_symbol(partition_type))
        end
        continuation(node, true)
        cx:pop_scope()
      else
        continuation(node, true)
      end
    end
  end
end

local function create_primary_partition(caller_cx, pparam, region_symbol)
  local stats = terralib.newlist()

  local parent_partition_symbol = nil
  local parent_region_symbol = caller_cx:find_parent_region(region_symbol)
  while parent_region_symbol do
    parent_partition_symbol =
      caller_cx:find_primary_partition(pparam, parent_region_symbol)
    if parent_partition_symbol then break end
    parent_region_symbol = caller_cx:find_parent_region(region_symbol)
  end
  if parent_partition_symbol then
    local partition_symbol, partition_stats =
      create_subset_partition(caller_cx, region_symbol, parent_partition_symbol,
                              pparam)
    stats:insertall(partition_stats)
  else
    local partition_symbol, partition_stats =
      create_equal_partition(caller_cx, region_symbol, pparam)
    stats:insertall(partition_stats)
  end

  return stats
end

local function join_stencils(joined_stencils, stencil_indices, stencils, mapping)
  for idx1 = 1, #stencils do
    local stencil1 = stencils[idx1]:subst(mapping)
    local joined = false
    for idx2 = 1, #joined_stencils do
      local stencil2 = joined_stencils[idx2]
      -- If global stencil analysis is requested,
      -- keep joining stencils whenever possible
      if std.config["parallelize-global"] then
        local joined_stencil =
          stencil_analysis.join_stencil(stencil1, stencil2)
        if joined_stencil ~= nil then
          joined_stencils[idx2] = joined_stencil
          stencil_indices[stencils[idx1]] = idx2
          joined = true
        end
      -- Otherwise, just check the compatibility
      -- so that partitions of the same shape are de-duplicated
      else
        local compatible =
          stencil_analysis.stencils_compatible(stencil1, stencil2)
        if compatible then
          stencil_indices[stencils[idx1]] = idx2
          joined = true
        end
      end
      if joined then break end
    end
    if not joined then
      joined_stencils:insert(stencil1)
      stencil_indices[stencils[idx1]] = #joined_stencils
    end
  end
end

local function insert_partition_creation(parallelizable, caller_cx, call_stats)
  return function(node, continuation)
    if caller_cx:get_call_exprs(node) then
      assert(node:is(ast.typed.stat.Var))

      local call_exprs_map = caller_cx:get_call_exprs(node)
      local stats = terralib.newlist {node}
      local symbol_type = node.symbol:gettype()
      local region_symbol

      if std.is_region(symbol_type) then
        if symbol_type:is_opaque() then return node end

        region_symbol = node.symbol

        -- First, create necessary primary partitions
        for pparam, _ in call_exprs_map:items() do
          if not caller_cx:has_primary_partition(pparam, region_symbol) then
            stats:insertall(create_primary_partition(caller_cx, pparam, region_symbol))
          end
        end
      elseif std.is_partition(symbol_type) then
        local partition_symbol = node.symbol
        region_symbol = symbol_type.parent_region_symbol

        for pparam, _ in call_exprs_map:items() do
          local color_space_symbol = caller_cx:find_color_space(pparam)
          if color_space_symbol == nil then
            local base_name = (partition_symbol:hasname() and partition_symbol:getname()) or ""
            color_space_symbol = get_new_tmp_var(symbol_type:colors(), base_name .. "__colors")
            local colors_expr = ast_util.mk_expr_colors_access(partition_symbol)
            stats:insert(ast_util.mk_stat_var(color_space_symbol, nil, colors_expr))
            caller_cx:add_color_space(pparam, color_space_symbol)
          end
        end
        for pparam, map in call_exprs_map:items() do
          for expr, request in pairs(map) do
            if request.type == SUBSET then
              region_symbol = request.region
              if not caller_cx:has_primary_partition(pparam, region_symbol) then
                stats:insertall(create_primary_partition(caller_cx, pparam, region_symbol))
              end
            end
          end
        end
      else
        assert(false, "unreachable")
      end
      assert(region_symbol ~= nil)

      -- Second, create ghost partitions
      for pparam, call_exprs in call_exprs_map:items() do
        local global_stencils = terralib.newlist()
        local global_stencil_indices = {}
        for call_expr, _ in pairs(call_exprs) do
          local task_cx = parallelizable(call_expr).cx
          local param_arg_mapping =
            task_cx:make_param_arg_mapping(caller_cx, call_expr.args)
          local stencils = data.filter(function(stencil)
            return stencil:is_static() and
                   (stencil:range(param_arg_mapping) == region_symbol or
                    (std.is_partition(symbol_type) and stencil:is_nested() and
                     stencil:range():region(param_arg_mapping) == region_symbol))
          end, task_cx.stencils)
          join_stencils(global_stencils, global_stencil_indices,
                        stencils, param_arg_mapping)
        end

        -- Now create all ghost partitions
        local ghost_partition_symbols = terralib.newlist()
        for idx = 1, #global_stencils do
          local stencil = global_stencils[idx]
          local range_symbol
          local partition_symbol
          if not stencil:is_nested() then
            range_symbol = stencil:range()
            partition_symbol =
              caller_cx:find_primary_partition(pparam, range_symbol)
          else
            range_symbol = stencil:range():region()
            partition_symbol = node.symbol
          end
          assert(partition_symbol)
          while caller_cx:has_parent_region(range_symbol) do
            range_symbol = caller_cx:get_parent_region(range_symbol)
          end
          local ghost_partition_symbol, ghost_partition_stats =
            create_image_partition(caller_cx, range_symbol, partition_symbol,
                                   stencil, pparam)
          ghost_partition_symbols:insert(ghost_partition_symbol)
          stats:insertall(ghost_partition_stats)
        end

        -- Finally, record ghost partitions that each task call will use
        for call_expr, _ in pairs(call_exprs) do
          local orig_stencils = parallelizable(call_expr).cx.stencils
          for idx = 1, #orig_stencils do
            local stencil_idx = global_stencil_indices[orig_stencils[idx]]
            if stencil_idx then
              local symbol = ghost_partition_symbols[stencil_idx]
              caller_cx:add_ghost_partition(call_expr, orig_stencils[idx], symbol)
            end
          end
        end
      end

      return stats
    elseif call_stats[node] then
      assert(node:is(ast.typed.stat.Var) or
             node:is(ast.typed.stat.Assignment) or
             node:is(ast.typed.stat.Reduce) or
             node:is(ast.typed.stat.Expr))
      local call
      if node:is(ast.typed.stat.Var) then
        call = node.value
      elseif node:is(ast.typed.stat.Assignment) or
             node:is(ast.typed.stat.Reduce) then
        call = node.rhs
      else
        call = node.expr
      end

      local stats = terralib.newlist()

      local regions_with_no_partition = terralib.newlist()
      local pparam = caller_cx:find_parallel_parameters(call)
      for idx = 1, #call.args do
        local arg_type = std.as_read(call.args[idx].expr_type)
        if std.is_region(arg_type) and arg_type:is_opaque() then
          local region_symbol = caller_cx:find_region_symbol(arg_type)
          if not caller_cx:has_primary_partition_by_call(call, region_symbol) then
            regions_with_no_partition:insert(region_symbol)
          end
        end
      end
      if #regions_with_no_partition ~= 0 then
        for idx = 1, #regions_with_no_partition do
          stats:insertall(
            create_primary_partition(caller_cx, pparam, regions_with_no_partition[idx]))
        end
      end

      local stencils_with_no_ghost_partition = terralib.newlist()
      local task_cx = parallelizable(call).cx
      for idx = 1, #task_cx.stencils do
        if not caller_cx:has_ghost_partition(call, task_cx.stencils[idx]) then
          stencils_with_no_ghost_partition:insert(task_cx.stencils[idx])
        end
      end

      local param_arg_mapping = task_cx:make_param_arg_mapping(caller_cx, call.args)
      local cleanup = terralib.newlist()
      if #stencils_with_no_ghost_partition ~= 0 then
        for idx = 1, #stencils_with_no_ghost_partition do
          local orig_stencil = stencils_with_no_ghost_partition[idx]
          local stencil = orig_stencil:subst(param_arg_mapping)
          local gp_type_from_hint = pparam:find_ghost_partition_for(stencil:region():gettype())
          if gp_type_from_hint then
            local gp_symbol = caller_cx:find_partition_symbol(gp_type_from_hint)
            caller_cx:add_ghost_partition(call, orig_stencil, gp_symbol)
          else
            if not stencil:is_static() then
              local region = stencil:region()
              local range = stencil:range()
              -- TODO: Support nested stencils for image partition
              assert(not Stencil.is_stencil(range))
              local index = stencil:index()
              local field_path = index:body().expr_type.field_path

              local pp = caller_cx:find_primary_partition_by_call(call, range)

              local base_name = (region:getname()) or ""
              local gp_type = std.partition(std.aliased, region, pp:gettype():colors())
              local gp_symbol = get_new_tmp_var(
                gp_type,
                base_name .. "__gp" .. "__image_" .. tostring(field_path))
              stats:insert(ast_util.mk_stat_var(gp_symbol, nil,
                ast.typed.expr.Image {
                  expr_type = gp_type,
                  parent = ast_util.mk_expr_id(region, std.rawref(&region:gettype())),
                  partition = ast_util.mk_expr_id(pp, std.rawref(&pp:gettype())),
                  region = ast.typed.expr.RegionRoot {
                    expr_type = std.as_read(range:gettype()),
                    region = ast_util.mk_expr_id(range, std.rawref(&range:gettype())),
                    fields = terralib.newlist {field_path},
                    span = ast.trivial_span(),
                    annotations = ast.default_annotations(),
                  },
                  span = ast.trivial_span(),
                  annotations = ast.default_annotations(),
                }))
              caller_cx:add_ghost_partition(call, orig_stencil, gp_symbol)
              cleanup:insert(ast.typed.stat.RawDelete {
                value = ast_util.mk_expr_id(gp_symbol, std.rawref(&gp_type)),
                span = ast.trivial_span(),
                annotations = ast.default_annotations(),
              })
            else
              local range_symbol
              local partition_symbol
              if orig_stencil:is_nested() then
                partition_symbol =
                  caller_cx:find_ghost_partition(call, orig_stencil:range())
                range_symbol = stencil:range():region()
              else
                assert(false, "should be unreachable for now")
              end
              assert(partition_symbol)
              while caller_cx:has_parent_region(range_symbol) do
                range_symbol = caller_cx:get_parent_region(range_symbol)
              end
              local partition_symbol, partition_stats =
                create_image_partition(caller_cx, range_symbol, partition_symbol,
                                       stencil, pparam)
              stats:insertall(partition_stats)
              caller_cx:add_ghost_partition(call, orig_stencil, partition_symbol)
              cleanup:insert(ast.typed.stat.RawDelete {
                value = ast_util.mk_expr_id(partition_symbol,
                                   std.rawref(&partition_symbol:gettype())),
                span = ast.trivial_span(),
                annotations = ast.default_annotations(),
              })
            end
          end
        end
      end

      stats:insert(node)
      stats:insertall(cleanup)
      return stats
    elseif node:is(ast.typed.stat.ParallelizeWith) then
      -- HACK: no good way to check equivalence between two AST nodes, so just duplicate
      --       metadata associate with this block whenever I make a new block
      local new_node = continuation(node, true)
      caller_cx:add_parallel_param_for_hints(new_node.hints,
        caller_cx:find_parallel_param_for_hints(node.hints))
      return new_node
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
          expr = node.value
          lhs = ast_util.mk_expr_id(node.symbol,
                           std.rawref(&std.as_read(node.symbol:gettype())))
        elseif node:is(ast.typed.stat.Reduce) then
          expr = node.rhs
          lhs = node.lhs
        else
          assert(false, "unreachable")
        end

        local reduction_op = parallelizable(expr).cx.reduction_info.op
        local lhs_type = std.as_read(lhs.expr_type)
        if node:is(ast.typed.stat.Var) then
          stats:insert(node {
            value = ast_util.mk_expr_constant(
              std.reduction_op_init[reduction_op][lhs_type], lhs_type)}
          )
        end
        stats:insertall(
          create_indexspace_launch(parallelizable, caller_cx, expr, lhs))
      end

      return stats
    elseif node:is(ast.typed.stat.ParallelizeWith) then
      local param = caller_cx:find_parallel_param_for_hints(node.hints)
      local stats = terralib.newlist()
      local function partition_type_from_hint(hint)
        local partition_type = nil
        if hint:is(ast.typed.expr.ID) then
          partition_type = std.as_read(hint.expr_type)
        elseif hint:is(ast.typed.expr.ParallelizerConstraint) then
          -- TODO: Hints might have a different form!
          partition_type = std.as_read(hint.rhs.expr_type)
        else
          assert(false, "unreachable")
        end
        return partition_type
      end
      if caller_cx:find_color_space(param) == nil then
        local partition_type = partition_type_from_hint(node.hints[1])
        local partition_symbol = caller_cx:find_partition_symbol(partition_type)
        local base_name = (partition_symbol:hasname() and partition_symbol:getname()) or ""
        local color_space_symbol = get_new_tmp_var(partition_type:colors(), base_name .. "__colors")
        local colors_expr = ast_util.mk_expr_colors_access(partition_symbol)
        stats:insert(ast_util.mk_stat_var(color_space_symbol, nil, colors_expr))
        caller_cx:add_color_space(param, color_space_symbol)

        if std.config["debug"] and not color_space_symbol:gettype():is_opaque() then
          for idx = 2, #node.hints do
            local bounds = ast_util.mk_expr_bounds_access(color_space_symbol)
            local my_partition_type = partition_type_from_hint(node.hints[idx])
            local my_partition_symbol =
              caller_cx:find_partition_symbol(my_partition_type)
            local my_bounds =
              ast_util.mk_expr_bounds_access(ast_util.mk_expr_colors_access(my_partition_symbol))
            stats:insert(ast_util.mk_stat_expr(ast_util.mk_expr_call(
                std.assert, terralib.newlist {
                  ast_util.mk_expr_binary("==", bounds, my_bounds),
                  ast_util.mk_expr_constant("color space bounds mismatch", rawstring)
                })))
          end
        end
      end
      node = continuation(node, true)
      if #stats > 0 then stats:insertall(node.block.stats)
      else stats = node.block.stats end
      return ast.typed.stat.Block {
        block = node.block {
          stats = stats,
        },
        span = node.span,
        annotations = node.annotations,
      }
    else
      return continuation(node, true)
    end
  end
end

function parallelize_task_calls.top_task(global_cx, node)
  local function parallelizable(node)
    if not node then return false
    elseif not node:is(ast.typed.expr.Call) then return false end
    local fn = node.fn.value
    return not node.annotations.parallel:is(ast.annotation.Forbid) and
           std.is_task(fn) and global_cx[fn]
  end

  -- Return if there is no parallelizable task call
  local found_calls = false
  local found_hints = false
  ast.traverse_node_continuation(function(node, continuation)
    if parallelizable(node) then
      found_calls = true
    elseif node:is(ast.typed.stat.ParallelizeWith) then
      found_hints = true
    end
    continuation(node, true)
  end, node)
  if not found_calls then
    if not found_hints then return node
    else
      return ast.map_node_continuation(function(node, continuation)
        if node:is(ast.typed.stat.ParallelizeWith) then
          return ast.typed.stat.Block {
            block = node.block,
            span = node.span,
            annotations = node.annotations,
          }
        else return continuation(node, true) end
      end, node)
    end
  end

  -- Add declartions for the variables that contain region metadata (e.g. bounds)
  local caller_cx = caller_context.new(node.prototype:get_constraints())
  local body =
    ast.flatmap_node_continuation(
      add_metadata_declarations(caller_cx), node.body)

  -- First, normalize all task calls so that task calls are either
  -- their own statements or single variable declarations.
  local call_stats = {}
  local normalized =
    ast.flatmap_node_continuation(
      normalize_calls(parallelizable, call_stats), body)

  -- Second, group task calls by reaching region declarations
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
        return ast_util.mk_expr_id(metadata_params.bounds)
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
-- ~>  var t = b.f
--     a = t
-- also, collect all field accesses for stencil analysis
--       and track the return value

local access_context = {}

access_context.__index = access_context

function access_context.new(loop_bound)
  local cx = {
    __loop_bound = loop_bound,
    __ref_types = terralib.newlist {opaque},
  }
  return setmetatable(cx, access_context)
end

function access_context:push_scope(ref_type)
  self.__ref_types:insert(ref_type)
end

function access_context:pop_scope()
  self.__ref_types[#self.__ref_types] = nil
end

function access_context:current_scope()
  return self.__ref_types[#self.__ref_types]
end

function access_context:loop_bound()
  return self.__loop_bound
end

function access_context:is_centered(node)
  if self:current_scope() ~= opaque and
     std.type_eq(node.expr_type.pointer_type,
                 self:current_scope().pointer_type) then
    return true
  end
  -- TODO: Index expressions might have been casted to index type,
  --       which can demote them to be uncentered, even though they aren't.
  if node:is(ast.typed.expr.IndexAccess) then
    assert(std.is_ref(node.expr_type))
    local index_type = std.as_read(node.index.expr_type)
    if not std.is_bounded_type(index_type) then return false end
    local bounds = index_type:bounds()
    if #bounds == 1 and bounds[1] == self:loop_bound() then return true
    else return false end
  elseif node:is(ast.typed.expr.Deref) then
    if std.is_bounded_type(node.value.expr_type) then
      local bounds = node.value.expr_type:bounds()
      if #bounds == 1 and bounds[1] == self:loop_bound() then return true
      else return false end
    end
  elseif node:is(ast.typed.expr.FieldAccess) then
    return self:is_centered(node.value)
  else
    return false
  end
end

local function find_field_accesses(context, accesses)
  return function(node, continuation)
    if node:is(ast.typed.expr) and
       std.is_ref(node.expr_type) then
      if not context:is_centered(node) then
        accesses:insert(node)
      end
      context:push_scope(node.expr_type)
      continuation(node, true)
      context:pop_scope()
    else
      continuation(node, true)
    end
  end
end

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

function normalizer_context:get_loop_bound()
  local bounds = self.loop_var:gettype():bounds()
  assert(#bounds == 1)
  return bounds[1]
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

local function simplify_expression(node, minus)
  if node:is(ast.typed.expr.Binary) then
    assert(not minus or node.op ~= "%")
    local lhs = simplify_expression(node.lhs, minus)
    local op = node.op
    local rhs
    if op == "-" then
      op = "+"
      rhs = simplify_expression(node.rhs, not minus)
    else
      rhs = simplify_expression(node.rhs, minus)
    end
    return node {
      op = op,
      lhs = lhs,
      rhs = rhs,
    }
  elseif node:is(ast.typed.expr.Unary) then
    if node.op == "-" then
      return simplify_expression(node.rhs, not minus)
    else
      return node { rhs = simplify_expression(node.rhs, minus) }
    end
  elseif node:is(ast.typed.expr.Constant) then
    if not minus then return node
    else return node { value = -node.value } end
  elseif node:is(ast.typed.expr.Cast) then
    return node { arg = simplify_expression(node.arg, minus) }
  elseif node:is(ast.typed.expr.Ctor) then
    return node {
      fields = node.fields:map(function(field)
        return simplify_expression(field, minus)
      end)
    }
  elseif node:is(ast.typed.expr.CtorListField) or
         node:is(ast.typed.expr.CtorRecField) then
    return node { value = simplify_expression(node.value, minus) }
  else
    return node
  end
end

local function extract_stencil_expr(pointer_type, node)
  if not std.is_ref(node.expr_type) then
    return node
  elseif node.expr_type.pointer_type ~= pointer_type then
    return node
  else
    if node:is(ast.typed.expr.FieldAccess) or
       node:is(ast.typed.expr.Deref) then
      return extract_stencil_expr(pointer_type, node.value)
    elseif node:is(ast.typed.expr.IndexAccess) then
      return extract_stencil_expr(pointer_type, node.index)
    else
      assert(false, "unreachable")
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
      ast.map_node_continuation(
        normalize_accesses.expr(normalizer_cx), access)

    local stencil_expr =
      extract_stencil_expr(normalized.expr_type.pointer_type, normalized)
		if stencil_expr:is(ast.typed.expr.Cast) then
		  stencil_expr = stencil_expr.arg
		end

    if not std.type_eq(std.as_read(loop_var:gettype()),
                       std.as_read(stencil_expr.expr_type)) then
      assert(#access.expr_type.bounds_symbols == 1)
      local region_symbol = access.expr_type.bounds_symbols[1]
      local field_path = access.expr_type.field_path
      stencil_expr = simplify_expression(stencil_expr)
      local stencil = Stencil {
        region = region_symbol,
        index = Lambda {
          binders = extract_symbols(always, stencil_expr),
          expr = stencil_expr,
        },
        range = loop_var:gettype().bounds_symbols[1],
        fields = { [field_path:hash()] = field_path },
      }
      task_cx:add_access(access, stencil)

      local base_name = (
        ((region_symbol:hasname() and region_symbol:getname()) or "")
          .. "__" .. tostring(field_path))
      local tmp_symbol = get_new_tmp_var(std.as_read(access.expr_type), base_name .. "__access")
      local stat = ast_util.mk_stat_var(tmp_symbol, nil, access)
      task_cx:record_stat_requires_case_split(stat)
      stats:insert(stat)
      rewrites[access] = ast_util.mk_expr_id(tmp_symbol)
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
      local symbol_type = node.symbol:gettype()
      if std.is_index_type(symbol_type) or
         std.is_bounded_type(symbol_type) then
        -- TODO: variables can be assigned later
        assert(node.value)
        normalizer_cx:add_decl(node.symbol, node.value)
      end
      local accesses = terralib.newlist()
      local access_cx = access_context.new(normalizer_cx:get_loop_bound())
      ast.traverse_node_continuation(
        find_field_accesses(access_cx, accesses), node.value)
      return lift_all_accesses(task_cx, normalizer_cx, accesses, node)
    elseif node:is(ast.typed.stat.Assignment) or
           node:is(ast.typed.stat.Reduce) then
      local accesses_lhs = terralib.newlist()
      local accesses_rhs = terralib.newlist()
      local access_cx = access_context.new(normalizer_cx:get_loop_bound())
      ast.traverse_node_continuation(
        find_field_accesses(access_cx, accesses_lhs), node.lhs)
      ast.traverse_node_continuation(
        find_field_accesses(access_cx, accesses_rhs), node.rhs)
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
      if node.symbol == reduction_var then
        assert(node.value)
        init_expr = node.value
        decl = node
      end
    elseif node:is(ast.typed.stat.Reduce) then
      if node.lhs:is(ast.typed.expr.ID) and node.lhs.value == reduction_var then
        assert(reduction_op == nil or reduction_op == node.op)
        reduction_op = node.op
      end
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

-- #####################################
-- ## Stencil analyzer
-- #################

-- (a, b, c) -->  (a, 0, 0), (0, b, 0), (0, 0, c),
--                (a, b, 0), (0, b, c), (a, 0, c),
--                (a, b, c)
function stencil_analysis.explode_expr(cx, stencil, expr)
  -- Index should be either e +/- c or (e +/- c) % r.bounds
  -- where e is for-list loop symbol and r is a region
  if expr:is(ast.typed.expr.Binary) then
    if expr.op == "%" then
      assert(std.is_rect_type(expr.rhs.expr_type))
      return stencil_analysis.explode_expr(cx, stencil, expr.lhs):map(
        function(stencil)
          return stencil:replace_index(expr { lhs = stencil:index() })
        end)
    elseif expr.op == "+" then
      local lhs = expr.lhs
      local range = stencil:range()
      if not lhs:is(ast.typed.expr.ID) then
        if lhs:is(ast.typed.expr.FieldAccess) then
          assert(#lhs.expr_type:bounds() == 1)
          range = stencil:replace_index(lhs)
          local index_type = std.as_read(lhs.expr_type)
          if std.is_bounded_type(index_type) then
            index_type = index_type.index_type
          end
          lhs = ast_util.mk_expr_id(get_new_tmp_var(index_type, "e"))
        else
          assert(false)
        end
      end

      if expr.rhs:is(ast.typed.expr.Ctor) then
        local exploded_rhs = stencil_analysis.explode_expr(cx, stencil, expr.rhs)
        return stencil_analysis.explode_expr(cx, stencil, expr.rhs):map(
          function(stencil)
            return stencil:replace_index(expr {
              lhs = lhs,
              rhs = stencil:index(),
            }):replace_range(range)
          end)
      elseif expr.rhs:is(ast.typed.expr.Constant) and
             expr.rhs.expr_type.type == "integer" then
        return terralib.newlist {
          stencil:replace_index(expr):replace_range(range)
        }
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
      return stencil:replace_index(expr {
        fields = offsets:map(function(offset)
          return ast_util.mk_expr_ctor_list_field_constant(offset, constant_type)
        end)
      })
    end)
  elseif std.is_ref(expr.expr_type) then
    return terralib.newlist { stencil:replace_index(expr) }
  else
    assert(false)
  end
end

function stencil_analysis.explode_stencil(cx, stencil)
  assert(Stencil.is_stencil(stencil))
  local index = stencil:index()
  if Lambda.is_lambda(index) then index = index:body() end
  local stencils = stencil_analysis.explode_expr(cx, stencil, index):map(
    function(stencil)
      stencil = stencil:replace_index(Lambda {
        binders = extract_symbols(always, stencil:index()),
        expr = stencil:index(),
      })
      if Stencil.is_stencil(stencil:range()) then
        stencil = stencil:replace_range(
          stencil:range():replace_index(Lambda {
            binders = extract_symbols(always, stencil:range():index()),
            expr = stencil:range():index(),
          }))
      end
      return stencil
    end)
  local range_stencils = terralib.newlist()
  for idx1 = 1, #stencils do
    local range_stencil = stencils[idx1]:range()
    if Stencil.is_stencil(range_stencil) then
      local has_compatible = false
      for idx2 = 1, #range_stencils do
        if stencil_analysis.stencils_compatible(range_stencil,
                                                range_stencils[idx2]) then
          has_compatible = true
          break
        end
      end
      if not has_compatible then
        range_stencils:insert(range_stencil)
      end
    end
  end
  range_stencils:insertall(stencils)
  return range_stencils
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
    if not (s1:range() == s2:range() or
            (Stencil.is_stencil(s1:range()) and
             Stencil.is_stencil(s2:range()) and
             stencil_analysis.stencils_compatible(s1:range(), s2:range()))) then
      return nil
    end
    assert(s1:region() == s2:region())
    local s1_binders = s1:index():all_binders()
    local s2_binders = s2:index():all_binders()
    if #s1_binders ~= #s2_binders then return nil end
    local s2_index = s2:index()
    for idx = 1, #s1_binders do
      s2_index = s2_index(s1_binders[idx])
    end
    assert(not Lambda.is_lambda(s2_index))
    local joined =
      stencil_analysis.join_stencil(s1:index():body(), s2_index)
    if joined then
      return Stencil({
        region = s1:region(),
        range = s1:range(),
        index = Lambda {
          binders = s1_binders,
          expr = joined,
        },
        fields = {},
      }):add_fields(s1:fields()):add_fields(s2:fields())
    else
      return nil
    end
  elseif ast.is_node(s1) and ast.is_node(s1) and s1:is(s2:type()) then
    if s1:is(ast.typed.expr.ID) then
      return (s1.value == s2.value) and s1
    elseif s1:is(ast.typed.expr.Binary) then
      if s1.op ~= s2.op then return nil
      elseif s1.op == "%" then
        return arg_join(stencil_analysis.stencils_compatible(s1.rhs, s2.rhs) and
                        stencil_analysis.join_stencil(s1.lhs, s2.lhs),
                        s1, s2, "lhs")
      elseif s1.op == "+" then
        return arg_join(stencil_analysis.stencils_compatible(s1.lhs, s2.lhs) and
                        stencil_analysis.join_stencil(s1.rhs, s2.rhs),
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
            return ast_util.mk_expr_ctor_list_field_constant(offset, constant_type)
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
    elseif std.is_ref(s1.expr_type) and
           std.type_eq(s1.expr_type, s2.expr_type) then
      return s1
    else
      return nil
    end
  else
    return nil
  end
end

function stencil_analysis.stencils_compatible(s1, s2)
  if Stencil.is_stencil(s1) and Stencil.is_stencil(s2) then
    local binders = s1:index():all_binders()
    local s2_index = s2:index()
    for idx = 1, #binders do
      s2_index = s2_index(binders[idx])
    end
    assert(not Lambda.is_lambda(s2_index))
    return s1:range() == s2:range() and
           stencil_analysis.stencils_compatible(s1:index():body(),
                                                s2_index)
  elseif ast.is_node(s1) and ast.is_node(s1) and s1:is(s2:type()) then
    if s1:is(ast.typed.expr.ID) then
      return s1.value == s2.value
    elseif s1:is(ast.typed.expr.Binary) then
      return s1.op == s2.op and
             stencil_analysis.stencils_compatible(s1.lhs, s2.lhs) and
             stencil_analysis.stencils_compatible(s1.rhs, s2.rhs)
    elseif s1:is(ast.typed.expr.Ctor) then
      if #s1.fields ~= #s2.fields then return false end
      for idx = 1, #s1.fields do
        if not stencil_analysis.stencils_compatible(
            s1.fields[idx].value, s2.fields[idx].value) then
          return false
        end
      end
      return true
    elseif s1:is(ast.typed.expr.Constant) then
      return s1.value == s2.value
    elseif s1:is(ast.typed.expr.Cast) then
      if not std.type_eq(s1.fn.value, s2.fn.value) then return false end
      return stencil_analysis.stencils_compatible(s1.arg, s2.arg)
    elseif std.is_ref(s1.expr_type) and
           std.type_eq(s1.expr_type, s2.expr_type) then
      return true
    else
      return false
    end
  else
    return false
  end
end

function stencil_analysis.top(cx)
  local sorted_accesses = terralib.newlist()
  for access, _ in pairs(cx.field_accesses) do
    sorted_accesses:insert(access)
  end
  sorted_accesses:sort(function(a1, a2)
    if a1.span.start.line < a2.span.start.line then
      return true
    elseif a1.span.start.line > a2.span.start.line then
      return false
    elseif a1.span.start.offset < a2.span.start.offset then
      return true
    elseif a1.span.start.offset > a2.span.start.offset then
      return false
    else
      return ast_util.render(a1) < ast_util.render(a2)
    end
  end)

  for idx = 1, #sorted_accesses do
    local access_info = cx.field_accesses[sorted_accesses[idx]]
    access_info.exploded_stencils:insertall(
      stencil_analysis.explode_stencil(cx, access_info.stencil))
    for i = 1, #access_info.exploded_stencils do
      access_info.ghost_indices:insert(-1)
      for j = 1, #cx.stencils do
        local s1 = access_info.exploded_stencils[i]
        local s2 = cx.stencils[j]
        local joined_stencil = stencil_analysis.join_stencil(s1, s2)
        if joined_stencil then
          cx.stencils[j] = joined_stencil
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

  local sorted_indices = terralib.newlist()
  for i = 1, #cx.stencils do sorted_indices:insert(i) end
  for i = 1, #cx.stencils - 1 do
    for j = 2, #cx.stencils do
      if cx.stencils[i]:depth() > cx.stencils[j]:depth()  then
        cx.stencils[i], cx.stencils[j] = cx.stencils[j], cx.stencils[i]
        sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]
      end
    end
  end
  for idx = 1, #sorted_accesses do
    local access_info = cx.field_accesses[sorted_accesses[idx]]
    access_info.ghost_indices:map(function(idx) return sorted_indices[idx] end)
  end

  for idx1 = 2, #cx.stencils do
    local range_stencil = cx.stencils[idx1]:range()
    if Stencil.is_stencil(range_stencil) then
      for idx2 = 1, idx1 do
        if stencil_analysis.stencils_compatible(cx.stencils[idx2], range_stencil) and
           cx.stencils[idx2] ~= range_stencil then
          cx.stencils[idx1] = cx.stencils[idx1]:replace_range(cx.stencils[idx2])
          break
        end
      end
    end
  end
end

local function make_new_region_access(region_expr, stencil_expr, field)
  local region_symbol = region_expr.value
  local region_type = std.as_read(region_expr.expr_type)
  local index_type = std.as_read(stencil_expr.expr_type)
  if std.is_bounded_type(index_type) then
    index_type = index_type.index_type
  end
  local expr = ast_util.mk_expr_index_access(region_expr, stencil_expr,
    std.ref(index_type(region_type:fspace(), region_symbol)))
  for idx = 1, #field do
    local new_field = expr.expr_type.field_path .. data.newtuple(field[idx])
    expr = ast_util.mk_expr_field_access(expr, field[idx],
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
             (node.value:is(ast.typed.expr.FieldAccess) or
              node.value:is(ast.typed.expr.Deref) or
              node.value:is(ast.typed.expr.IndexAccess)))
      local stats = terralib.newlist()

      -- Remove RHS of a variable declaration as it depends on case analysis
      stats:insert(node { value = false })

      -- Cache index calculation for several comparisions later
      local access = node.value
      local access_info = task_cx.field_accesses[access]
      local stencil_expr =
        extract_stencil_expr(access.expr_type.pointer_type, access)

      -- If index expressions is complex, cache it before the comparisons
      if not stencil_expr:is(ast.typed.expr.ID) then
        local index_symbol = get_new_tmp_var(std.as_read(stencil_expr.expr_type), "__index")
        stats:insert(ast_util.mk_stat_var(index_symbol, nil, stencil_expr))
        stencil_expr =
          ast_util.mk_expr_id(index_symbol, std.rawref(&index_symbol:gettype()))
      end

      -- Make an expression for result to reuse in the case splits
      local result_symbol = node.symbol
      local result_expr =
        ast_util.mk_expr_id(result_symbol, std.rawref(&result_symbol:gettype()))

      -- If the stencil had a bounded type, we're sure we're gonna use a
      -- ghost region that covers the entire access.
      -- FIXME: We should still support case splits between private and ghost
      --        regions.
      if std.is_bounded_type(std.as_read(stencil_expr.expr_type)) then
        assert(#access_info.ghost_indices == 1)
        assert(Stencil.is_singleton(access_info.exploded_stencils[1]))
        local stencil = task_cx.stencils[access_info.ghost_indices[1]]
        local region_symbol = task_cx.ghost_symbols[stencil]
        local field = access_info.exploded_stencils[1]:fields()[1]
        local region_type = std.rawref(&region_symbol:gettype())
        local region_id_expr = ast_util.mk_expr_id(region_symbol, region_type)
        local region_access =
          make_new_region_access(region_id_expr, stencil_expr, field)
        stats:insert(ast_util.mk_stat_assignment(result_expr, region_access))
      else
        -- Case split for each region access:
        -- var x = r[f(e)] =>
        --   var x; var p = f(e)
        --   do
        --     if x <= r.bounds then x = r[p]
        --     elseif p <= ghost1.bounds then x = ghost1[p]
        --     elseif p <= ghost2.bounds then x = ghost2[p]
        --     ...

        -- Populate body of case analysis
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
            local stencil = task_cx.stencils[access_info.ghost_indices[idx]]
            region_symbol = task_cx.ghost_symbols[stencil]
            assert(Stencil.is_singleton(access_info.exploded_stencils[idx]))
            field = access_info.exploded_stencils[idx]:fields()[1]
          end

          local region_type = std.rawref(&region_symbol:gettype())
          local region_id_expr = ast_util.mk_expr_id(region_symbol, region_type)
          local bounds_expr = ast_util.mk_expr_bounds_access(region_id_expr)
          local cond = ast_util.mk_expr_binary("<=", stencil_expr, bounds_expr)

          local region_access =
            make_new_region_access(region_id_expr, stencil_expr, field)
          local result_assignment = ast_util.mk_stat_assignment(result_expr, region_access)
          if idx == 0 then
            case_split_if = ast_util.mk_stat_if(cond, result_assignment)
            elseif_blocks = case_split_if.elseif_blocks
          else
            elseif_blocks:insert(ast_util.mk_stat_elseif(cond, result_assignment))
          end
        end

        assert(case_split_if)
        if std.config["debug"] then
          local index_symbol = get_new_tmp_var(std.as_read(stencil_expr.expr_type), "__index")
          case_split_if.else_block.stats:insertall(terralib.newlist {
            ast_util.mk_stat_var(index_symbol, nil, stencil_expr),
            ast_util.mk_stat_expr(ast_util.mk_expr_call(print_point[index_symbol:gettype()],
                         terralib.newlist {
                           ast_util.mk_expr_id(index_symbol),
                         })),
            ast_util.mk_stat_expr(ast_util.mk_expr_call(std.assert,
                         terralib.newlist {
                           ast_util.mk_expr_constant(false, bool),
                           ast_util.mk_expr_constant("unreachable", rawstring)
                         })),
          })
        end
        stats:insert(case_split_if)
      end
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
             task_cx.reduction_info.declaration == stat then

        local stats = terralib.newlist()

        local red_var = stat.symbol
        local red_var_expr =
          ast_util.mk_expr_id(red_var, std.rawref(&red_var:gettype()))
        stats:insert(ast_util.mk_stat_var(red_var, stat.type))

        local cond = ast_util.mk_expr_call(is_zero,
          ast_util.mk_expr_id(task_cx:get_task_point_symbol()))
        local if_stat = ast_util.mk_stat_if(
          cond, ast_util.mk_stat_assignment(red_var_expr, stat.value))
        local init =
          std.reduction_op_init[task_cx.reduction_info.op][red_var:gettype()]
        assert(init ~= nil)
        if_stat.else_block.stats:insert(
          ast_util.mk_stat_assignment(
            red_var_expr, ast_util.mk_expr_constant(init, red_var:gettype())))
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
  local task = std.new_task(task_name)
  local variant = task:make_variant("primary")
  task:set_primary_variant(variant)
  node.prototype:set_parallel_task(task)

  local params = terralib.newlist()
  -- Existing region-typed parameters will now refer to the subregions
  -- passed by indexspace launch. this will avoid rewriting types in AST nodes
  params:insertall(node.params)
  -- each stencil corresponds to one ghost region
  local orig_privileges = node.prototype:get_privileges()
  local function has_interfering_update(stencil)
    local region = stencil:region()
    local fields = stencil:fields()
    for i = 1, #fields do
      for j = 1, #orig_privileges do
        for k = 1, #orig_privileges[j] do
          local pv = orig_privileges[j][k]
          if pv.privilege == std.writes and
             pv.region == region and
             pv.field_path == fields[i] then
             return true
          end
        end
      end
    end
    return false
  end

  for idx = 1, #task_cx.stencils do
    local stencil = task_cx.stencils[idx]
    local check = has_interfering_update(stencil)
    if check then
      -- FIXME: Ghost accesses can be out of bounds because we force them to use
      --        primary partition here. Task is in general not parallelizable
      --        if it has stencils that conflict with its update set. We assume
      --        that the programmer specified right hints to make it parallelizable.
      task_cx.ghost_symbols[stencil] = stencil:region()
      task_cx.use_primary[stencil] = true
    else
      local ghost_symbol =
        copy_region_symbol(stencil:region(), "__ghost" .. tostring(idx))
      task_cx.ghost_symbols[stencil] = ghost_symbol
      task_cx.use_primary[stencil] = false
      params:insert(ast_util.mk_task_param(task_cx.ghost_symbols[stencil]))
    end
  end
  -- Append parameters reserved for the metadata of original region parameters
  task_cx:insert_metadata_parameters(params)

  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), node.return_type, false)
  task:set_type(task_type)
  task:set_param_symbols(
    params:map(function(param) return param.symbol end))
  local region_universe = node.prototype:get_region_universe():copy()
  local privileges = terralib.newlist()
  local coherence_modes = data.new_recursive_map(1)
  --node.prototype:get_coherence_modes():map_list(function(region, map)
  --    print(region)
  --  map:map_list(function(field_path, v)
  --    coherence_modes[region][field_path] = true
  --  end)
  --end)
  privileges:insertall(orig_privileges)
  -- FIXME: Workaround for the current limitation in SPMD transformation
  local field_set = {}
  for idx = 1, #task_cx.stencils do
		task_cx.stencils[idx]:fields():map(function(field) field_set[field] = true end)
  end
  local fields = terralib.newlist()
  for field, _ in pairs(field_set) do fields:insert(field) end

  for idx = 1, #task_cx.stencils do
    local stencil = task_cx.stencils[idx]
    if not task_cx.use_primary[stencil] then
		  local region = task_cx.ghost_symbols[stencil]
		  --local fields = task_cx.stencils[idx]:fields()
      -- TODO: handle reductions on ghost regions
      privileges:insert(fields:map(function(field)
        return std.privilege(std.reads, region, field)
      end))
      --coherence_modes[region][field_path] = std.exclusive
      region_universe[region:gettype()] = true
    end
  end
  task:set_privileges(privileges)
  task:set_coherence_modes(coherence_modes)
  task:set_flags(node.flags)
  task:set_conditions(node.conditions)
  task:set_param_constraints(node.prototype:get_param_constraints())
  task:set_constraints(node.prototype:get_constraints())
  task:set_region_universe(region_universe)

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
    prototype = task,
    annotations = node.annotations {
      parallel = ast.annotation.Forbid { value = false },
    },
    span = node.span,
  }
  variant:set_ast(task_ast)

  -- Hack: prevents parallelized verions from going through parallelizer again
  global_cx[task] = {}
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
