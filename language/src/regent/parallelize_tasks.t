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
local data = require("regent/data")
local std = require("regent/std")
local log = require("regent/log")
local pretty = require("regent/pretty")
local symbol_table = require("regent/symbol_table")
local passes = require("regent/passes")

local c = std.c

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
local get_ghost_rect = {
  [std.rect1d] = terra(root : std.rect1d, r : std.rect1d, s : std.rect1d) : std.rect1d
    var sz = root:size()
    var diff_rect : std.rect1d

    -- wrapped around, periodic boundary
    if s.lo.__ptr > s.hi.__ptr then
      diff_rect.lo.__ptr = s.lo.__ptr
      diff_rect.hi.__ptr = (r.lo.__ptr - 1 + sz.__ptr) % sz.__ptr
    -- shift left
    elseif s.lo.__ptr < r.lo.__ptr then
      diff_rect.lo.__ptr = s.lo.__ptr
      diff_rect.hi.__ptr = r.lo.__ptr - 1
    -- shift right
    elseif s.lo.__ptr > r.lo.__ptr then
      diff_rect.lo.__ptr = r.hi.__ptr + 1
      diff_rect.hi.__ptr = s.hi.__ptr
    else
      diff_rect.lo.__ptr = r.lo.__ptr
      diff_rect.hi.__ptr = r.hi.__ptr
    end

    return diff_rect
  end,
  [std.rect2d] = terra(root : std.rect2d, r : std.rect2d, s : std.rect2d) : std.rect2d
    var sz = root:size()
    var diff_rect : std.rect2d
    -- wrapped around, periodic boundary
    if s.lo.__ptr.x > s.hi.__ptr.x then
      diff_rect.lo.__ptr.x = s.lo.__ptr.x
      diff_rect.hi.__ptr.x = (r.lo.__ptr.x - 1 + sz.__ptr.x) % sz.__ptr.x
    -- shift left
    elseif s.lo.__ptr.x < r.lo.__ptr.x then
      diff_rect.lo.__ptr.x = s.lo.__ptr.x
      diff_rect.hi.__ptr.x = r.lo.__ptr.x - 1
    -- shift right
    elseif s.lo.__ptr.x > r.lo.__ptr.x then
      diff_rect.lo.__ptr.x = r.hi.__ptr.x + 1
      diff_rect.hi.__ptr.x = s.hi.__ptr.x
    else
      diff_rect.lo.__ptr.x = r.lo.__ptr.x
      diff_rect.hi.__ptr.x = r.hi.__ptr.x
    end

    -- wrapped around, periodic boundary
    if s.lo.__ptr.y > s.hi.__ptr.y then
      diff_rect.lo.__ptr.y = s.lo.__ptr.y
      diff_rect.hi.__ptr.y = (r.lo.__ptr.y - 1 + sz.__ptr.y) % sz.__ptr.y
    -- shift up
    elseif s.lo.__ptr.y < r.lo.__ptr.y then
      diff_rect.lo.__ptr.y = s.lo.__ptr.y
      diff_rect.hi.__ptr.y = r.lo.__ptr.y - 1
    -- shift down
    elseif s.lo.__ptr.y > r.lo.__ptr.y then
      diff_rect.lo.__ptr.y = r.hi.__ptr.y + 1
      diff_rect.hi.__ptr.y = s.hi.__ptr.y
    else
      diff_rect.lo.__ptr.y = r.lo.__ptr.y
      diff_rect.hi.__ptr.y = r.hi.__ptr.y
    end
    return diff_rect
  end,
  [std.rect3d] = terra(root : std.rect3d, r : std.rect3d, s : std.rect3d) : std.rect3d
    var sz = root:size()
    var diff_rect : std.rect3d
    -- wrapped around, periodic boundary
    if s.lo.__ptr.x > s.hi.__ptr.x then
      diff_rect.lo.__ptr.x = s.lo.__ptr.x
      diff_rect.hi.__ptr.x = (r.lo.__ptr.x - 1 + sz.__ptr.x) % sz.__ptr.x
    -- shift left
    elseif s.lo.__ptr.x < r.lo.__ptr.x then
      diff_rect.lo.__ptr.x = s.lo.__ptr.x
      diff_rect.hi.__ptr.x = r.lo.__ptr.x - 1
    -- shift right
    elseif s.lo.__ptr.x > r.lo.__ptr.x then
      diff_rect.lo.__ptr.x = r.hi.__ptr.x + 1
      diff_rect.hi.__ptr.x = s.hi.__ptr.x
    else
      diff_rect.lo.__ptr.x = r.lo.__ptr.x
      diff_rect.hi.__ptr.x = r.hi.__ptr.x
    end

    -- wrapped around, periodic boundary
    if s.lo.__ptr.y > s.hi.__ptr.y then
      diff_rect.lo.__ptr.y = s.lo.__ptr.y
      diff_rect.hi.__ptr.y = (r.lo.__ptr.y - 1 + sz.__ptr.y) % sz.__ptr.y
    -- shift up
    elseif s.lo.__ptr.y < r.lo.__ptr.y then
      diff_rect.lo.__ptr.y = s.lo.__ptr.y
      diff_rect.hi.__ptr.y = r.lo.__ptr.y - 1
    -- shift down
    elseif s.lo.__ptr.y > r.lo.__ptr.y then
      diff_rect.lo.__ptr.y = r.hi.__ptr.y + 1
      diff_rect.hi.__ptr.y = s.hi.__ptr.y
    else
      diff_rect.lo.__ptr.y = r.lo.__ptr.y
      diff_rect.hi.__ptr.y = r.hi.__ptr.y
    end

    -- wrapped around, periodic boundary
    if s.lo.__ptr.z > s.hi.__ptr.z then
      diff_rect.lo.__ptr.z = s.lo.__ptr.z
      diff_rect.hi.__ptr.z = (r.lo.__ptr.z - 1 + sz.__ptr.z) % sz.__ptr.z
    -- shift up
    elseif s.lo.__ptr.z < r.lo.__ptr.z then
      diff_rect.lo.__ptr.z = s.lo.__ptr.z
      diff_rect.hi.__ptr.z = r.lo.__ptr.z - 1
    -- shift down
    elseif s.lo.__ptr.z > r.lo.__ptr.z then
      diff_rect.lo.__ptr.z = r.hi.__ptr.z + 1
      diff_rect.hi.__ptr.z = s.hi.__ptr.z
    else
      diff_rect.lo.__ptr.z = r.lo.__ptr.z
      diff_rect.hi.__ptr.z = r.hi.__ptr.z
    end

    return diff_rect
  end
}

local function render(expr)
  if not expr then return "nil" end
  if expr:is(ast.typed.expr) then
    return pretty.render.entry(nil, pretty.expr(nil, expr))
  elseif expr:is(ast.typed.stat) then
    return pretty.render.entry(nil, pretty.stat(nil, expr))
  end
end

-- utility functions for AST node construction
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
  local index_type =
    std.as_read(value.expr_type):ispace().index_type
  return mk_expr_field_access(value, "bounds", std.rect_type(index_type))
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

local function extract_index_expr(node)
  local index_expr
  extract_expr(node,
    function(node) return node:is(ast.typed.expr.IndexAccess) end,
    function(node) index_expr = node.index end)
  return index_expr
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

local check_parallelizable = {}

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
  --    e +/- c or (e +/- c) % r.bounds where r is the primary region
end

local Stencil = {}

local stencil_meta = {}

stencil_meta.__index = stencil_meta

function stencil_meta.__call(self, arg)
  local expr = self:expr()
  if Stencil.is_stencil(expr) then
    local hole = expr:hole()
    expr.__hole = self:hole()
    return Stencil {
      expr = self:expr()(arg),
      hole = hole,
    }
  else
    if std.is_symbol(arg) then
      return rewrite_symbol(expr, self:hole(), arg)
    else
      return rewrite_expr_id(expr, self:hole(), arg)
    end
  end
end

function stencil_meta:expr()
  return self.__expr
end

function stencil_meta:hole()
  return self.__hole
end

function stencil_meta:field_path()
  if Stencil.is_stencil(self:expr()) then
    return self:expr():field_path()
  else
    if self:expr():is(ast.typed.expr.FieldAccess) then
      return self:expr().expr_type.field_path
    else
      return data.newtuple()
    end
  end
end

function stencil_meta:fmap(fn)
  local expr = self:expr()
  if Stencil.is_stencil(expr) then
    return Stencil {
      expr = expr:fmap(fn),
      hole = self:hole(),
    }
  else
    return Stencil {
      expr = fn(expr),
      hole = self:hole(),
    }
  end
end

function stencil_meta:__tostring()
  local hole_str = tostring(self:hole())
  local expr_str
  if Stencil.is_stencil(self:expr()) then
    expr_str = tostring(self:expr())
  else
    expr_str = render(self:expr())
  end
  return "\\" .. hole_str .. "." .. expr_str
end

local stencil_ctor = {}

stencil_ctor.__index = stencil_ctor

function stencil_ctor.__call(self, args)
  assert(args.expr)
  assert(args.hole)
  return setmetatable({
    __expr = args.expr,
    __hole = args.hole,
  }, stencil_meta)
end

function stencil_ctor.is_stencil(e)
  return getmetatable(e) == stencil_meta
end

Stencil = setmetatable(Stencil, stencil_ctor)

local context = {}
context.__index = context

function context.new_task_scope(primary_region)
  local cx = {}
  cx.primary_region = primary_region
  cx.field_accesses = {}
  cx.field_access_stats = {}
  cx.stencils = terralib.newlist()
  cx.ghost_symbols = terralib.newlist()
  return setmetatable(cx, context)
end

function context:add_field_accesses(accesses, loop_symbol)
  for access, _ in pairs(accesses) do
    self.field_accesses[access] = {
      loop_symbol = loop_symbol,
      ghost_indices = terralib.newlist(),
      stencils = terralib.newlist(),
    }
  end
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
local function create_image_partition(pr, pp, stencil)
  local pr_type = std.as_read(pr:gettype())
  local pr_index_type = pr_type:ispace().index_type
  local pr_rect_type = std.rect_type(pr_index_type)
  local pp_color_space_type = pp:gettype():colors()
  local pp_color_type =
    pp_color_space_type.index_type(std.newsymbol(pp_color_space_type))

  -- TODO: partition can be aliased
  local gp_color_space_type = pp_color_space_type
  local gp_type = std.partition(std.disjoint, pr, gp_color_space_type)
  local gp_symbol = get_new_tmp_var(gp_type)
  local stats = terralib.newlist()

  local color_symbol = get_new_tmp_var(pp_color_type)
  local color_expr = mk_expr_id(color_symbol)

  local coloring_symbol = get_new_tmp_var(c.legion_domain_point_coloring_t)
  local coloring_expr = mk_expr_id(coloring_symbol)
  stats:insert(mk_stat_var(coloring_symbol, nil,
                           mk_expr_call(c.legion_domain_point_coloring_create)))

  local loop_body = terralib.newlist()
  local pr_expr = mk_expr_id(pr)
  local pp_expr = mk_expr_id(pp)
  local sr_expr =
    mk_expr_index_access(pp_expr, color_expr, copy_region_type(pr_type))
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
  loop_body:insert(mk_stat_expr(
    mk_expr_call(c.legion_domain_point_coloring_color_domain,
                 terralib.newlist { coloring_expr,
                                    color_expr,
                                    ghost_rect_expr })))

  local pp_colors_expr =
    mk_expr_field_access(pp_expr, "colors", pp_color_space_type)
  stats:insert(mk_stat_for_list(color_symbol, pp_colors_expr, mk_block(loop_body)))
  stats:insert(
    mk_stat_var(gp_symbol, nil,
                mk_expr_partition(gp_type, pp_colors_expr, coloring_expr)))

  stats:insert(mk_stat_expr(mk_expr_call(c.legion_domain_point_coloring_destroy,
                                         coloring_expr)))

  return gp_symbol, stats
end

local function create_indexspace_launch(pr, pp, gps, task)
  local pp_expr = mk_expr_id(pp)
  local pr_type = std.as_read(pr:gettype())
  local pp_color_space_type = pp:gettype():colors()
  local pp_color_type =
    pp_color_space_type.index_type(std.newsymbol(pp_color_space_type))
  local color_symbol = get_new_tmp_var(pp_color_type)
  local color_expr = mk_expr_id(color_symbol)
  local pp_colors_expr =
    mk_expr_field_access(pp_expr, "colors", pp_color_space_type)

  local args = terralib.newlist()
  args:insert(
    mk_expr_index_access(pp_expr, color_expr, copy_region_type(pr_type)))
  for idx = 1, #gps do
    args:insert(
      mk_expr_index_access(mk_expr_id(gps[idx]), color_expr,
                           copy_region_type(pr_type)))
  end
  args:insert(mk_expr_id(pr))
  return mk_stat_for_list(color_symbol, pp_colors_expr,
                          mk_block(mk_stat_expr(mk_expr_call(task, args))))
end

local parallelize_task_calls = {}

function parallelize_task_calls.stat_expr_call(global_cx, local_cx, node)
  if node.annotations.parallel:is(ast.annotation.Forbid) then return node end
  local info = global_cx[node.expr.fn.value.name]
  if not info then return node end

  local parallel_task = info.task
  local task_cx = info.cx

  local args = terralib.newlist()
  args:insertall(node.expr.args)
  local region_idx
  for idx = 1, #args do
    if std.is_region(std.as_read(args[idx].expr_type)) then
      region_idx = idx
      break
    end
  end
  assert(region_idx)
  assert(args[region_idx]:is(ast.typed.expr.ID))
  local primary_region_symbol = args[region_idx].value
  local primary_partition_symbol = local_cx.primary_partitions[primary_region_symbol]

  -- TODO: handle the case where no primary partition exists
  assert(primary_partition_symbol)

  -- create ghost partitions
  local stats = terralib.newlist()
  local ghost_partition_symbols = terralib.newlist()

  for idx = 1, #task_cx.stencils do
    local stencil = task_cx.stencils[idx](primary_region_symbol)
    stencil = stencil:fmap(extract_index_expr):expr()
    local ghost_partition_symbol, ghost_partition_stats =
      create_image_partition(primary_region_symbol,
                             primary_partition_symbol,
                             stencil)
    ghost_partition_symbols:insert(ghost_partition_symbol)
    stats:insertall(ghost_partition_stats)
  end

  -- create an indexspace launch
  local stat = create_indexspace_launch(primary_region_symbol,
                                        primary_partition_symbol,
                                        ghost_partition_symbols,
                                        parallel_task)
  stats:insert(stat)
  return mk_stat_block(mk_block(stats))
end

function parallelize_task_calls.top_task(global_cx, node)
  local local_cx = { primary_partitions = {} }
  node = ast.map_node_postorder(function(node)
    -- TODO: also handles reduction case
    if node:is(ast.typed.stat.Expr) and
       node.expr:is(ast.typed.expr.Call) and
       std.is_task(node.expr.fn.value) then
      return parallelize_task_calls.stat_expr_call(global_cx, local_cx, node)
    elseif node:is(ast.typed.stat.Var) then
      for idx = 1, #node.symbols do
        if std.is_partition(node.symbols[idx]:gettype()) then
          local partition_type = node.symbols[idx]:gettype()
          if partition_type:is_disjoint() then
            local_cx.primary_partitions[partition_type.parent_region_symbol] =
              node.symbols[idx]
          end
        end
      end
      return node
    else
      return node
    end
  end, node)
  return node
end

-- normalize field accesses; e.g.
-- a = b.f
-- ~>  t = b.f
--     a = t
-- also, collect all field accesses for stencil analysis
local normalize_accesses = {}

function normalize_accesses.expr(cx, node)
  if node:is(ast.typed.expr.Deref) then
    if std.is_bounded_type(node.value.expr_type) and
       std.is_ref(node.expr_type) and
       not node.value.expr_type.index_type:is_opaque() then
       assert(#node.value.expr_type:bounds() == 1)
       local region_sym = node.expr_type.bounds_symbols[1]
       return mk_expr_index_access(
         mk_expr_id(region_sym, std.rawref(&region_sym:gettype())),
         node.value, node.expr_type)
    else
      return node
    end
  elseif node:is(ast.typed.expr.FieldAccess) then
    return node {
      value = normalize_accesses.expr(cx, node.value),
    }
  else
    return node
  end
end

function normalize_accesses.stat_for_list(cx, node)
  local rewrites = {}
  local field_reads = {}
  local field_writes = {}
  local function find_field_access(node, continuation)
    if node:is(ast.typed.expr.FieldAccess) then
      local tmp_symbol = get_new_tmp_var(std.as_read(node.expr_type))
      rewrites[node] = mk_expr_id(tmp_symbol)
      field_reads[normalize_accesses.expr(cx, node)] = rewrites[node]
    elseif node:is(ast.typed.stat.Assignment) or
           node:is(ast.typed.stat.Reduce) then
      node.lhs:map(function(node)
        if node:is(ast.typed.expr.FieldAccess) then
          rewrites[node] = normalize_accesses.expr(cx, node)
          field_writes[rewrites[node]] = true
        end
      end)
      continuation(node.rhs, true)
    elseif node:is(ast.typed.stat.Var) then
      node.values:map(function(node)
        if not node:is(ast.typed.expr.FieldAccess) then
          continuation(node, true)
        end
      end)
    else
      continuation(node, true)
    end
  end
  local function rewrite_field_access(node, continuation)
    if rewrites[node] then return rewrites[node]
    else return continuation(node, true) end
  end
  ast.traverse_node_continuation(find_field_access, node)
  local stats = terralib.newlist()
  table.sort(field_reads)
  for field_access, id_expr in pairs(field_reads) do
    local stat = mk_stat_var(id_expr.value, id_expr.expr_type, field_access)
    cx.field_access_stats[stat] = true
    stats:insert(stat)
  end
  stats:insertall(node.block.stats:map(function(node)
    return ast.map_node_continuation(rewrite_field_access, node)
  end))
  local loop = node {
    block = node.block {
      stats = stats,
    },
  }
  cx:add_field_accesses(field_reads, node.symbol)
  cx:add_field_accesses(field_writes, node.symbol)
  return loop
end

function normalize_accesses.top_task_body(cx, node)
  return node {
    stats = node.stats:map(function(node)
      if node:is(ast.typed.stat.ForList) and
         node.value:is(ast.typed.expr.ID) and
         node.value.value == cx.primary_region then
        return normalize_accesses.stat_for_list(cx, node)
      else
        return node
      end
    end),
  }
end

local stencil_analysis = {}

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

-- (a, b, c) -->  (a, 0, 0), (0, b, 0), (0, 0, c),
--                (a, b, 0), (0, b, c), (a, 0, c),
--                (a, b, c)
function stencil_analysis.expr(cx, expr)
  if expr:is(ast.typed.expr.FieldAccess) then
    return stencil_analysis.expr(cx, expr.value):map(function(value)
      return expr { value = value }
    end)
  elseif expr:is(ast.typed.expr.IndexAccess) then
    if not expr.index:is(ast.typed.expr.Binary) then
      return terralib.newlist()
    else
      return stencil_analysis.expr(cx, expr.index):map(function(index)
        return expr { index = index }
      end)
    end
  -- index should be either e +/- c or (e +/- c) % r.bounds
  -- where e is for-list loop symbol and r is primary region
  elseif expr:is(ast.typed.expr.Binary) then
    if expr.op == "%" then
      assert(expr.rhs:is(ast.typed.expr.FieldAccess) and
             expr.rhs.value.value == cx.primary_region and
             expr.rhs.field_name == "bounds")
      return stencil_analysis.expr(cx, expr.lhs):map(function(lhs)
        return expr { lhs = lhs }
      end)
    elseif expr.op == "+" or expr.op == "-" then
      assert(expr.rhs:is(ast.typed.expr.Ctor))
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
      return stencil_analysis.expr(cx, expr.rhs):map(function(rhs)
        return expr {
          op = "+",
          rhs = rhs { fields = rhs.fields:map(convert) },
        }
      end)
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

-- find the lub of two stencils (stencils are partially ordered)
-- returns 1) s1 |_| s2 if s1 and s2 has lub
--         2) nil if s1 <> s2
function stencil_analysis.join_stencil(cx, s1, s2)
  if s1 and s2 and s1:is(s2:type()) then
    if s1:is(ast.typed.expr.FieldAccess) then
      -- TODO: handle multiple fields for a stencil
      if s1.field_name ~= s2.field_name then
        return nil
      else
        return arg_join(stencil_analysis.join_stencil(cx, s1.value, s2.value),
                        s1, s2, "value")
      end
    elseif s1:is(ast.typed.expr.IndexAccess) then
      assert(s1.value:is(ast.typed.expr.ID) and
             s2.value:is(ast.typed.expr.ID) and
             s1.value.value == s2.value.value)
      return arg_join(stencil_analysis.join_stencil(cx, s1.index, s2.index),
                      s1, s2, "index")
    elseif s1:is(ast.typed.expr.Binary) then
      if s1.op == "%" -- TODO: and stencil_analysis.equiv(s1.rhs, s2.rhs)
      then
        return arg_join(stencil_analysis.join_stencil(cx, s1.lhs, s2.lhs),
                        s1, s2, "lhs")
      elseif s1.op == "+" then
        assert(s1.lhs:is(ast.typed.expr.ID) and
               s2.lhs:is(ast.typed.expr.ID) and
               s1.lhs.value == s2.lhs.value)
        return arg_join(stencil_analysis.join_stencil(cx, s1.rhs, s2.rhs),
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
    else
      return nil
    end
  else
    return nil
  end
end

function stencil_analysis.top(cx)
  for access, access_info in pairs(cx.field_accesses) do
    access_info.stencils:insertall(
      stencil_analysis.expr(cx, access):map(function(expr)
        return Stencil {
          hole = access_info.loop_symbol,
          expr = expr,
        }
      end))
  end

  for _, access_info in pairs(cx.field_accesses) do
    for i = 1, #access_info.stencils do
      access_info.ghost_indices:insert(-1)
      for j = 1, #cx.stencils do
        local hole = cx.stencils[j]:hole()
        local s1 = access_info.stencils[i](hole)
        local s2 = cx.stencils[j]:expr()
        local joined_stencil = stencil_analysis.join_stencil(cx, s1, s2)
        if joined_stencil then
          cx.stencils[j] = Stencil {
            hole = hole,
            expr = joined_stencil,
          }
          access_info.ghost_indices[i] = j
          break
        end
      end
      if access_info.ghost_indices[i] == -1 then
        cx.stencils:insert(access_info.stencils[i])
        access_info.ghost_indices[i] = #cx.stencils
      end
    end
  end

  -- TODO: Stencil objects should be also used in the previous steps
  --       to compare between stencils from two different loops
  local function eta_expansion(stencil)
   local hole = stencil:hole()
   local expr = stencil:expr()
   local primary_region_type = std.rawref(&cx.primary_region:gettype())
   local region_binder = get_new_tmp_var(primary_region_type)
   local root_region_binder = get_new_tmp_var(primary_region_type)
   expr = rewrite_expr(expr,
     function(node) return node:is(ast.typed.expr.IndexAccess) and
                           node.value:is(ast.typed.expr.ID) and
                           node.value.value == cx.primary_region end,
     function(node)
       return node { value = node.value { value = region_binder } }
     end)
   expr = rewrite_symbol(expr, cx.primary_region, root_region_binder)
   local stencil = Stencil {
     hole = root_region_binder,
     expr = Stencil {
       hole = region_binder,
       expr = Stencil {
         hole = hole,
         expr = expr,
       },
     },
   }
   return stencil
  end
  for access, access_info in pairs(cx.field_accesses) do
    access_info.stencils = access_info.stencils:map(eta_expansion)
  end
  cx.stencils = cx.stencils:map(eta_expansion)
end

local parallelize_tasks = {}

function parallelize_tasks.stat_for_list(cx, node)
  local loop_var = node.symbol
  local stats = terralib.newlist()
  for idx = 1, #node.block.stats do
    local stat = node.block.stats[idx]
    if cx.field_access_stats[stat] then
      assert(stat:is(ast.typed.stat.Var) and
             #stat.symbols == 1 and #stat.types == 1 and
             #stat.values == 1 and
             (stat.values[1]:is(ast.typed.expr.FieldAccess) or
              stat.values[1]:is(ast.typed.expr.Deref)))
      if cx.field_accesses[stat.values[1]] and
         #cx.field_accesses[stat.values[1]].stencils > 0 then
        -- case split for each field access:
        -- var x = r[f(e)] =>
        --   var x; var p = f(e)
        --   do
        --     if p <= ghost1.bounds then x = ghost1[p]
        --     elseif p <= ghost2.bounds then x = ghost2[p]
        --     ...
        --     else x = r[p]
        local stencil_info = cx.field_accesses[stat.values[1]]
        local result_symbol = stat.symbols[1]
        local result_expr =
          mk_expr_id(result_symbol, std.rawref(&result_symbol:gettype()))
        local ty = stat.types[1]
        -- stencil has a form of \(root region).\(region).\(loop var).(body)
        local stencil = stencil_info.stencils[#stencil_info.stencils]
        local value_stencil = stencil(cx.root_region)
        local point_expr = value_stencil:expr():fmap(extract_index_expr)(loop_var)
        local point_symbol = get_new_tmp_var(loop_var:gettype().index_type)
        local point_symbol_expr = mk_expr_id(point_symbol)
        stats:insert(stat { values = terralib.newlist() })
        stats:insert(mk_stat_var(point_symbol, point_symbol:gettype(), point_expr))
        value_stencil = value_stencil:fmap(function(node)
          return rewrite_expr(node,
            function(node) return node:is(ast.typed.expr.IndexAccess) and
                                  node.value:is(ast.typed.expr.ID) and
                                  node.value.value == value_stencil:hole() end,
            function(node) return node { index = point_symbol_expr } end)
        end)
        local case_split_if
        local elseif_blocks
        for idx = 0, #stencil_info.ghost_indices do
          local region_symbol
          if idx == 0 then region_symbol = cx.primary_region
          else region_symbol = cx.ghost_symbols[stencil_info.ghost_indices[idx]] end
          local region_type = std.rawref(&region_symbol:gettype())
          local region_id_expr = mk_expr_id(region_symbol, region_type)
          local bounds_expr = mk_expr_bounds_access(region_id_expr)
          local cond = mk_expr_binary("<=", point_symbol_expr, bounds_expr)
          local region_access = value_stencil(region_symbol)(loop_var)
          local result_assignment = mk_stat_assignment(result_expr, region_access)
          if idx == 0 then
            case_split_if = mk_stat_if(cond, result_assignment)
            elseif_blocks = case_split_if.elseif_blocks
          else
            elseif_blocks:insert(mk_stat_elseif(cond, result_assignment))
          end
        end
        assert(case_split_if)
        -- FIXME: adding calls to assertion slows down code considerably,
        --        potentially due to instruction cache misses from error string
        --case_split_if.else_block.stats:insert(mk_stat_expr(mk_expr_call(
        --  std.assert,
        --  terralib.newlist {
        --    mk_expr_constant(false, bool),
        --    mk_expr_constant("unreachable", rawstring)
        --  })))
        stats:insert(case_split_if)
      else
        stats:insert(stat)
      end
    else
      stats:insert(stat)
    end
  end
  return node { block = node.block { stats = stats } }
end

function parallelize_tasks.top_task_body(cx, node)
  return node {
    stats = node.stats:map(function(stat)
      if stat:is(ast.typed.stat.ForList) then
        return parallelize_tasks.stat_for_list(cx, stat)
      else
        return stat
      end
    end),
  }
end

function parallelize_tasks.top_task(node)
  local primary_region
  for idx = 1, #node.params do
    if std.is_region(node.params[idx].param_type) then
      primary_region = node.params[idx].symbol
      break
    end
  end

  -- auto-parallelization procedure
  -- 1. field accesses are normalized; a = @b => t = @b; a = t
  -- 2. find stencil sizes for all field accesses
  -- 3. merge stencil sizes
  -- 4. calculate # of necessary ghost regions and number them
  -- 5. for each field access normalized in 1,
  --    find the right ghost region and rewrite the access
  -- 6. create a new task with the modified task signature

  local cx = context.new_task_scope(primary_region)
  local normalized = normalize_accesses.top_task_body(cx, node.body)
  stencil_analysis.top(cx)

  for idx = 1, #cx.stencils do
    local ghost_symbol =
      copy_region_symbol(primary_region, "__ghost" .. tostring(idx))
    cx.ghost_symbols:insert(ghost_symbol)
  end
  cx.root_region = copy_region_symbol(primary_region, "__root")
  local parallelized = parallelize_tasks.top_task_body(cx, normalized)

  -- make a new task AST node
  local task_name = node.name .. data.newtuple("parallelized")
  local prototype = std.newtask(task_name)
  local params = terralib.newlist()
  params:insertall(node.params)
  for idx = 1, #cx.ghost_symbols do
    params:insert(mk_task_param(cx.ghost_symbols[idx]))
  end
  params:insert(mk_task_param(cx.root_region))
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
  for idx = 1, #cx.stencils do
		local region = cx.ghost_symbols[idx]
		local field_path = cx.stencils[idx]:field_path()
    -- TODO: handle reductions on ghost regions
    privileges:insert(terralib.newlist {
      data.map_from_table {
        node_type = "privilege",
        region = region,
        field_path = field_path,
        privilege = std.reads,
      }})
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

  local task_ast_optimized = passes.optimize(task_ast)
  local task_code = passes.codegen(task_ast_optimized, true)

  return task_code, cx
end

local global_context = {}

function parallelize_tasks.entry(node)
  if node:is(ast.typed.top.Task) then
    if node.annotations.parallel:is(ast.annotation.Demand) then
      check_parallelizable.top_task(node)
      local task_name = node.name
      local new_task_code, cx = parallelize_tasks.top_task(node)
      global_context[task_name] = {}
      global_context[task_name].task = new_task_code
      global_context[task_name].cx = cx
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
