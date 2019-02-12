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

-- Regent Auto-parallelizer

local ast = require("regent/ast")
local ast_util = require("regent/ast_util")
local data = require("common/data")
local std = require("regent/std")
local report = require("common/report")
local symbol_table = require("regent/symbol_table")
local passes = require("regent/passes")
local pretty = require("regent/pretty")

local c = std.c

local prefix = "partition driven auto-parallelization"

-- #####################################
-- ## Partitioning constraint inference
-- #################

-- The infer_constraints pass registers image partitioning constraints to the context
-- whenever the index of a region access is derived from another region access or
-- an analyzable expression.

local function find_or_create(map, key, init)
  local init = init or data.newmap
  local value = map[key]
  if value == nil then
    value = init()
    map[key] = value
  end
  return value
end

local range_complex = std.newsymbol("__top__")

local new_range
do
  local range_idx = 0
  function new_range()
    local range = std.newsymbol("I" .. tostring(range_idx))
    range_idx = range_idx + 1
    return range
  end
end

local graph = {}

graph.__index = graph

function graph.new()
  local g = {
    vertices = data.newmap(),
    edges = data.newmap(),
  }
  return setmetatable(g, graph)
end

function graph:dump()
  print("digraph {")
  local next_id = 1
  local node_ids = {}
  for v, _ in self.vertices:items() do
    node_ids[v] = next_id
    print("    " .. tostring(next_id) .. " [ label = \"" .. tostring(v) .. "\" ];")
    next_id = next_id + 1
  end
  for src, edges in self.edges:items() do
    for dst, key in edges:items() do
      local src_id = node_ids[src]
      local dst_id = node_ids[dst]
      print("    " .. tostring(src_id) .. " -> " .. tostring(dst_id) ..
        " [ label = \"" .. tostring(key) .. "\" ];")
    end
  end
  print("}")
end

function graph:connected(v1, v2)
  local edges = self.edges[v1]
  if edges == nil then return false
  else return edges[v2] end
end

function graph:get_connected_component(source)
  local component = terralib.newlist({source})
  local idx = 1
  while idx <= #component do
    local src = component[idx]
    idx = idx + 1
    if self.edges[src] then
      for dst, _ in self.edges[src]:items() do
        component:insert(dst)
      end
    end
  end
  return component
end

function graph:get_connected_components()
  local sources = self.vertices:copy()
  for src, edges in self.edges:items() do
    for dst, _ in edges:items() do
      sources[dst] = nil
    end
  end
  local connected_components = terralib.newlist()
  for source, _ in sources:items() do
    connected_components:insert(self:get_connected_component(source))
  end
  return connected_components
end

local function create_product_graph(g1, g2)
  local g = graph.new()
  for v1, l1 in g1.vertices:items() do
    for v2, l2 in g2.vertices:items() do
      if l1 == l2 then
        local v = data.newtuple(v1, v2)
        g.vertices[v] = true
      end
    end
  end
  for va, _ in g.vertices:items() do
    for vb, _ in g.vertices:items() do
      local va1, va2 = unpack(va)
      local vb1, vb2 = unpack(vb)
      if va1 ~= vb1 and va2 ~= vb2 then
        local key1 = g1:connected(va1, vb1)
        local key2 = g2:connected(va2, vb2)
        if key1 and key2 and key1 == key2 then
          local edges = find_or_create(g.edges, va)
          edges[vb] = key1 or true
        end
      end
    end
  end
  return g
end

local partitioning_constraints = {}

partitioning_constraints.__index = partitioning_constraints

function partitioning_constraints.new()
  local cx = {
    ranges               = data.newmap(),
    subset_constraints   = data.newmap(),
    image_constraints    = data.newmap(),
    analytic_constraints = data.newmap(),
  }
  return setmetatable(cx, partitioning_constraints)
end

function partitioning_constraints:is_empty()
  return self.ranges:is_empty()
end

function partitioning_constraints:print_constraints()
  print("* ranges:")
  for range, region_symbol in self.ranges:items() do
    print("    " .. tostring(range) .. " : partition(" .. tostring(region_symbol) .. ")")
  end
  print("* subset constraints:")
  for src, all_constraints in self.subset_constraints:items() do
    for region_symbol, dst in all_constraints:items() do
      print("    " .. tostring(src) .. " <= " .. tostring(dst))
    end
  end
  print("* image constraints:")
  for src, all_constraints in self.image_constraints:items() do
    for key, dst in all_constraints:items() do
      local region_symbol, field_path = unpack(key)
      print("    " .. tostring(region_symbol) .. "[" .. tostring(src) .. "]." ..
        field_path:mkstring("", ".", "") .. " <= " .. tostring(dst))
    end
  end
  print("* analytic constraints:")
  for src, constraints in self.analytic_constraints:items() do
    for offset, dst in constraints:items() do
      if std.is_symbol(offset[#offset]) then
        local symbol = offset[#offset]
        local o = offset:slice(1, #offset - 1)
        local divider =
          (std.is_region(symbol:gettype()) and tostring(symbol) .. ".ispace.bounds") or
          (std.is_ispace(symbol:gettype()) and tostring(symbol) .. ".bounds") or
          assert(false)
        print("    (" ..tostring(src) .. " + {" ..
          o:mkstring("",",","") .. "}) % " ..
          divider .. " <= " .. tostring(dst))
      else
        print("    " ..tostring(src) .. " + {" ..
          offset:mkstring("",",","") .. "} <= " .. tostring(dst))
      end
    end
  end
end

function partitioning_constraints:set_domain(range, region_symbol)
  assert(self.ranges[range] == nil or self.ranges[range] == region_symbol)
  self.ranges[range] = region_symbol
end

function partitioning_constraints:get_domain(range)
  return self.ranges[range]
end

function partitioning_constraints:find_or_create_subset_constraint(src_range, region_symbol)
  local constraints = find_or_create(self.subset_constraints, src_range)
  return find_or_create(constraints, region_symbol, new_range)
end

function partitioning_constraints:find_or_create_image_constraint(src_range, region_symbol, field_path)
  local key = data.newtuple(region_symbol, field_path)
  local constraints = find_or_create(self.image_constraints, src_range)
  return find_or_create(constraints, key, new_range)
end

function partitioning_constraints:find_or_create_analytic_constraint(src_range, offset)
  local constraints = find_or_create(self.analytic_constraints, src_range)
  return find_or_create(constraints, offset, new_range)
end

function partitioning_constraints:clone(mapping)
  local map_region = nil
  if mapping then
    map_region = function(region_symbol) return mapping[region_symbol] or region_symbol end
  else
    map_region = function(region_symbol) return region_symbol end
  end
  local result = partitioning_constraints.new()
  for range, region_symbol in self.ranges:items() do
    result:set_domain(range, map_region(region_symbol))
  end
  for src, constraints in self.subset_constraints:items() do
    local result_constraints = find_or_create(result.subset_constraints, src)
    for region_symbol, dst in constraints:items() do
      result_constraints[map_region(region_symbol)] = dst
    end
  end
  for src, constraints in self.image_constraints:items() do
    local result_constraints = find_or_create(result.image_constraints, src)
    for pair, dst in constraints:items() do
      local region_symbol, field_path = unpack(pair)
      local key = data.newtuple(map_region(region_symbol), field_path)
      result_constraints[key] = dst
    end
  end
  for src, constraints in self.analytic_constraints:items() do
    local result_constraints = find_or_create(result.analytic_constraints, src)
    for offset, dst in constraints:items() do
      local key = data.newtuple(unpack(offset))
      if std.is_symbol(key[#key]) then
        key[#key] = map_region(key[#key])
      end
      result_constraints[key] = dst
    end
  end
  return result
end

function partitioning_constraints:create_graph()
  local g = graph.new()
  for range, region_symbol in self.ranges:items() do
    g.vertices[range] = region_symbol
  end
  for src, constraints in self.subset_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for region_symbol, dst in constraints:items() do
      assert(edges[dst] == nil)
      edges[dst] = true
    end
  end
  for src, constraints in self.image_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for key, dst in constraints:items() do
      assert(edges[dst] == nil)
      edges[dst] = key
    end
  end
  for src, constraints in self.analytic_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for key, dst in constraints:items() do
      assert(edges[dst] == nil)
      edges[dst] = key
    end
  end
  return g
end

function partitioning_constraints:join(to_join, mapping)
  for range, region_symbol in to_join.ranges:items() do
    local my_range = mapping[range]
    if my_range == nil then
      self.ranges[range] = region_symbol
    end
  end

  for src, constraints in to_join.subset_constraints:items() do
    local my_src = mapping[src] or src
    local my_constraints = find_or_create(self.subset_constraints, my_src)
    for key, dst in constraints:items() do
      local my_dst = mapping[dst] or dst
      assert(my_constraints[key] == nil or my_constraints[key] == my_dst)
      my_constraints[key] = my_dst
    end
  end

  for src, constraints in to_join.image_constraints:items() do
    local my_src = mapping[src] or src
    local my_constraints = find_or_create(self.image_constraints, my_src)
    for key, dst in constraints:items() do
      local my_dst = mapping[dst] or dst
      assert(my_constraints[key] == nil or my_constraints[key] == my_dst)
      my_constraints[key] = my_dst
    end
  end

  for src, constraints in to_join.analytic_constraints:items() do
    local my_src = mapping[src] or src
    local my_constraints = find_or_create(self.analytic_constraints, my_src)
    for key, dst in constraints:items() do
      local my_dst = mapping[dst] or dst
      assert(my_constraints[key] == nil or my_constraints[key] == my_dst)
      my_constraints[key] = my_dst
    end
  end
end

local parallel_task_context = {}

parallel_task_context.__index = parallel_task_context

local function unify(a, b, map)
  local seta = map[a]
  local setb = map[b]
  if seta == nil and setb == nil then
    local u = data.newmap()
    u[a] = true
    u[b] = true
    map[a] = u
    map[b] = u
  elseif seta == nil then
    local u = map[b]
    u[a] = true
    map[a] = u
  elseif setb == nil then
    local u = map[a]
    u[b] = true
    map[b] = u
  else
    for k, _ in setb:items() do
      seta[k] = true
      map[k] = seta
    end
  end
end

local function map_tostring(map)
  local str = nil
  for k, _ in map:items() do
    if str == nil then
      str = tostring(k)
    else
      str = str .. "," .. tostring(k)
    end
  end
  return str or ""
end

function parallel_task_context.new()
  local cx = {
    env                      = data.newmap(),
    back_edges               = data.newmap(),
    sources                  = data.newmap(),

    field_accesses           = data.newmap(),
    field_accesses_summary   = data.newmap(),

    constraints              = partitioning_constraints.new(),
    completeness_constraints = data.newmap(),
    disjointness_constraints = data.newmap(),
    equality_constraints     = data.newmap(),
    union_constraints        = data.newmap(),

    -- Complexity of the system of constraints is a 4-tuple of
    --  # of image/analytic constraints,
    --  # of disjointness/completeness constraints,
    --  # of equality constraints, and
    --  # of distinct field accesses
    complexity               = false,
  }

  return setmetatable(cx, parallel_task_context)
end

local function cmp_complexity(c1, c2)
  for idx = 1, 4 do
    if c1[idx] > c2[idx] then return true
    elseif c1[idx] < c2[idx] then return false end
  end
  return false
end

local function cmp_tasks(t1, t2)
  return cmp_complexity(
    t1:get_partitioning_constraints().complexity,
    t2:get_partitioning_constraints().complexity)
end

function parallel_task_context:print_all_constraints()
  print("================")
  print("* complexity:")
  print("    " .. self.complexity:mkstring("(", ", ", ")"))
  print("* sources:")
  for range, _ in self.sources:items() do
    print("    " .. tostring(range))
  end
  self.constraints:print_constraints()
  print("* completeness constraints:")
  for range, _ in self.completeness_constraints:items() do
    print("    " .. tostring(range) .. " : complete")
  end
  print("* disjointness constraints:")
  for range, _ in self.disjointness_constraints:items() do
    print("    " .. tostring(range) .. " : disjoint")
  end
  print("* equality constraints:")
  do
    local duplicate = {}
    for range, equivalence_set in self.equality_constraints:items() do
      if duplicate[equivalence_set] == nil then
        print("    {" .. map_tostring(equivalence_set) .. "}")
        duplicate[equivalence_set] = true
      end
    end
  end
  print("* union constraints:")
  do
    local duplicate = {}
    for range, concurrent_set in self.union_constraints:items() do
      if duplicate[concurrent_set] == nil then
        print("    {" .. map_tostring(concurrent_set) .. "}")
        duplicate[concurrent_set] = true
      end
    end
  end
  print("* access summary:")
  for region_symbol, accesses_summary in self.field_accesses_summary:items() do
    for field_path, summary in accesses_summary:items() do
      local ranges, privilege = unpack(summary)
      local range = tostring(ranges[1])
      for idx = 2, #ranges do
        range = range .. "," .. tostring(ranges[idx])
      end
      print("    " .. tostring(region_symbol) .. "[{ " .. range .. " }]." ..
        field_path:mkstring("", ".", "") .. " @ " .. tostring(privilege))
    end
  end
  print("================")
end

function parallel_task_context:add_field_access(region_symbol, range, field_path, privilege)
  local domain = self.constraints:get_domain(range)
  if not (domain == nil or domain == region_symbol) then
    local new_range = self.constraints:find_or_create_subset_constraint(range, region_symbol)
    self.back_edges[new_range] = range
    range = new_range
  end

  local all_field_accesses = find_or_create(self.field_accesses, region_symbol)
  local field_accesses = find_or_create(all_field_accesses, field_path)
  field_accesses[range] = std.meet_privilege(field_accesses[range], privilege)
  self:set_domain(range, region_symbol)
end

function parallel_task_context:summarize_accesses()
  local complexity = data.newtuple(0, 0, 0, 0)

  for region_symbol, all_field_accesses in self.field_accesses:items() do
    local accesses_summary = find_or_create(self.field_accesses_summary, region_symbol)
    for field_path, accesses in all_field_accesses:items() do
      local ranges = terralib.newlist()
      local joined_privilege = nil
      for range, privilege in accesses:items() do
        ranges:insert(range)
        joined_privilege = std.meet_privilege(joined_privilege, privilege)
      end
      accesses_summary[field_path] = { ranges, joined_privilege }

      complexity[4] = complexity[4] + 1

      if (joined_privilege == std.writes or joined_privilege == "reads_writes") then
        self:add_disjointness_constraint(ranges[1])
      end

      if #ranges > 1 then
        local constrain = nil
        if (joined_privilege == std.writes or joined_privilege == "reads_writes") then
          constrain = self.add_equality_constraint
          local back_edge = nil
          ranges:map(function(range) back_edge = back_edge or self.back_edges[range] end)
          ranges:map(function(range) self.back_edges[range] = back_edge end)
        else
          constrain = self.add_union_constraint
        end
        for idx = 1, #ranges - 1 do
          constrain(self, ranges[idx], ranges[idx + 1])
        end
      end
    end
  end

  for range, _ in self.constraints.ranges:items() do
    if self.back_edges[range] == nil then
      self.sources[range] = true
    end
  end

  for _, all_constraints in self.constraints.subset_constraints:items() do
    for _, _ in all_constraints:items() do
      complexity[1] = complexity[1] + 1
    end
  end
  for _, all_constraints in self.constraints.image_constraints:items() do
    for _, _ in all_constraints:items() do
      complexity[1] = complexity[1] + 1
    end
  end
  for _, constraints in self.constraints.analytic_constraints:items() do
    for _, _ in constraints:items() do
      complexity[1] = complexity[1] + 1
    end
  end
  for range, _ in self.completeness_constraints:items() do
    complexity[2] = complexity[2] + 1
  end
  for range, _ in self.disjointness_constraints:items() do
    complexity[2] = complexity[2] + 1
  end
  local duplicate = {}
  for range, equivalence_set in self.equality_constraints:items() do
    if duplicate[equivalence_set] == nil then
      complexity[3] = complexity[3] + 1
      duplicate[equivalence_set] = true
    end
  end
  self.complexity = complexity
end

function parallel_task_context:find_or_create_image_constraint(src_range, region_symbol, field_path)
  local dst_range =
    self.constraints:find_or_create_image_constraint(src_range, region_symbol, field_path)
  self.back_edges[dst_range] = src_range
  return dst_range
end

function parallel_task_context:find_or_create_analytic_constraint(src_range, offset)
  local dst_range =
    self.constraints:find_or_create_analytic_constraint(src_range, offset)
  self.back_edges[dst_range] = src_range
  return dst_range
end

function parallel_task_context:set_range(symbol, range)
  self.env[symbol] = range
end

function parallel_task_context:get_range(symbol)
  return self.env[symbol]
end

function parallel_task_context:set_domain(range, region_symbol)
  if range == range_complex then return end
  self.constraints:set_domain(range, region_symbol)
end

function parallel_task_context:add_completeness_constraint(range)
  self.completeness_constraints[range] = true
end

function parallel_task_context:add_disjointness_constraint(range)
  self.disjointness_constraints[range] = true
end

function parallel_task_context:add_equality_constraint(range1, range2)
  unify(range1, range2, self.equality_constraints)
end

function parallel_task_context:add_union_constraint(range1, range2)
  unify(range1, range2, self.union_constraints)
end

local infer_constraints = {}

local function unreachable(cx, node) node:printpretty(true) assert(false) end

function infer_constraints.expr_id(cx, expr, privilege, field_path)
  return cx:get_range(expr.value) or range_complex
end

function infer_constraints.expr_field_access(cx, expr, privilege, field_path)
  local field_path = field_path
  if std.is_ref(expr.expr_type) then
    field_path = field_path or expr.expr_type.field_path
  end
  return infer_constraints.expr(cx, expr.value, privilege, field_path)
end

function infer_constraints.expr_deref(cx, expr, privilege, field_path)
  assert(std.is_ref(expr.expr_type) and #expr.expr_type:bounds() == 1)
  local region_symbol = expr.expr_type.bounds_symbols[1]
  local region_type = expr.expr_type:bounds()[1]
  local expr = ast.typed.expr.IndexAccess {
    value = ast.typed.expr.ID {
      value = region_symbol,
      expr_type = std.rawref(&region_type),
      span = expr.span,
      annotations = expr.annotations,
    },
    index = expr.value,
    expr_type = expr.expr_type,
    span = expr.span,
    annotations = expr.annotations,
  }
  return infer_constraints.expr(cx, expr, privilege, field_path)
end

function infer_constraints.expr_index_access(cx, expr, privilege, field_path)
  local field_path = field_path or data.newtuple()
  if not std.is_ref(expr.expr_type) then return range_complex end
  assert(expr.value:is(ast.typed.expr.ID) and
         std.is_region(std.as_read(expr.value.expr_type)))
  local index_range = infer_constraints.expr(cx, expr.index)
  local region_symbol = expr.value.value
  local fspace = std.as_read(expr.expr_type)
  local field_type = std.get_field_path(fspace, field_path)
  local field_paths = std.flatten_struct_fields(field_type)
  field_paths:map(function(suffix)
    local field_path = field_path .. suffix
    cx:add_field_access(region_symbol, index_range, field_path, privilege)
  end)

  local field_type = std.get_field_path(std.as_read(expr.expr_type), field_path)
  if std.is_index_type(field_type) or std.is_bounded_type(field_type) then
    return cx:find_or_create_image_constraint(index_range, region_symbol, field_path)
  else
    return range_complex
  end
end

local function extract_offset(cx, expr, positive)
  if expr:is(ast.typed.expr.Binary) then
    local lhs_range, lhs_offset = extract_offset(cx, expr.lhs, positive)
    if expr.op == "-" then positive = not positive end
    local rhs_range, rhs_offset = extract_offset(cx, expr.rhs, positive)
    if (expr.op == "%" or expr.op == "+" or expr.op == "-") and rhs_range == nil then
      return lhs_range, lhs_offset == nil and rhs_offset or lhs_offset .. rhs_offset
    end
  elseif expr:is(ast.typed.expr.FieldAccess) then
    if std.is_rect_type(expr.expr_type) and expr.field_name == "bounds" and
       std.is_ispace(std.as_read(expr.value.expr_type))
    then
      local base = expr.value
      if base:is(ast.typed.expr.ID) then
        return nil, data.newtuple(base.value)
      elseif base:is(ast.typed.expr.FieldAccess) and
             base.field_name == "ispace" and
             base.value:is(ast.typed.expr.ID)
      then
        assert(std.is_region(std.as_read(base.value.expr_type)))
        return nil, data.newtuple(base.value.value)
      end
    end
  elseif expr:is(ast.typed.expr.ID) then
    return cx:get_range(expr.value), nil
  elseif expr:is(ast.typed.expr.Constant) then
    return nil, data.newtuple(expr.value)
  elseif expr:is(ast.typed.expr.Ctor) then
    local all_constant = true
    local values = expr.fields:map(function(field)
      if field.value:is(ast.typed.expr.Constant) then
        if positive then
          return field.value.value
        else
          return -field.value.value
        end
      elseif field.value:is(ast.typed.expr.Unary) and
             field.value.op == "-" and
             field.value.rhs:is(ast.typed.expr.Constant)
      then
        assert(field.value.expr_type:isintegral())
        if positive then
          return -field.value.rhs.value
        else
          return field.value.rhs.value
        end
      else
        all_constant = false
        return nil
      end
    end)
    if all_constant then return nil, data.newtuple(unpack(values)) end
  end
  return range_complex, nil
end

function infer_constraints.expr_binary(cx, expr, privilege, field_path)
  if not std.is_index_type(expr.expr_type) then return range_complex end
  local src_range, offset = extract_offset(cx, expr, true)
  if src_range ~= range_complex and offset ~= nil then
    return cx:find_or_create_analytic_constraint(src_range, offset)
  else
    return range_complex
  end
end

function infer_constraints.expr_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.arg, privilege) or range_complex
end

function infer_constraints.expr_regent_cast(cx, expr, privilege, field_path)
  return infer_constraints.expr(cx, expr.value, privilege) or range_complex
end

function infer_constraints.expr_complex(cx, expr, privilege, field_path)
  return range_complex
end

local infer_constraints_expr_table = {
  [ast.typed.expr.ID]           = infer_constraints.expr_id,
  [ast.typed.expr.FieldAccess]  = infer_constraints.expr_field_access,
  [ast.typed.expr.Deref]        = infer_constraints.expr_deref,
  [ast.typed.expr.IndexAccess]  = infer_constraints.expr_index_access,
  [ast.typed.expr.Binary]       = infer_constraints.expr_binary,
  [ast.typed.expr.Cast]         = infer_constraints.expr_cast,
  [ast.typed.expr.DynamicCast]  = infer_constraints.expr_regent_cast,
  [ast.typed.expr.StaticCast]   = infer_constraints.expr_regent_cast,
  [ast.typed.expr.UnsafeCast]   = infer_constraints.expr_regent_cast,

  [ast.typed.expr.Constant]     = infer_constraints.expr_complex,
  [ast.typed.expr.Call]         = infer_constraints.expr_complex,
  [ast.typed.expr.Null]         = infer_constraints.expr_complex,
  [ast.typed.expr.Isnull]       = infer_constraints.expr_complex,
  [ast.typed.expr.Ctor]         = infer_constraints.expr_complex,
  [ast.typed.expr.Unary]        = infer_constraints.expr_complex,
  [ast.typed.expr]              = unreachable,
}

local infer_constraints_expr = ast.make_single_dispatch(
  infer_constraints_expr_table,
  {ast.typed.expr})

function infer_constraints.expr(cx, node, privilege, field_path)
  return infer_constraints_expr(cx)(node, privilege, field_path)
end

function infer_constraints.stat_for_list(cx, stat)
  if stat.metadata and stat.metadata.parallelizable then
    local range = new_range()
    local region_symbol = stat.value.value
    cx:set_range(stat.symbol, range)
    cx:set_domain(range, region_symbol)
    cx:add_completeness_constraint(range)
    if stat.metadata.reductions and #stat.metadata.reductions > 0 then
      cx:add_disjointness_constraint(range)
    end
  end
  infer_constraints.block(cx, stat.block)
end

function infer_constraints.stat_block(cx, stat)
  infer_constraints.block(cx, stat.block)
end

function infer_constraints.stat_if(cx, stat)
  infer_constraints.block(cx, stat.then_block)
  infer_constraints.block(cx, stat.else_block)
end

function infer_constraints.stat_var(cx, stat)
  if not stat.value then return end

  local range = infer_constraints.expr(cx, stat.value, std.reads)
  cx:set_range(stat.symbol, range)
end

function infer_constraints.stat_assignment_or_reduce(cx, stat)
  assert(not std.is_ref(stat.rhs.expr_type))

  if std.is_ref(stat.lhs.expr_type) then
    local privilege = (stat:is(ast.typed.stat.Assignment) and std.writes) or
                      std.reduces(stat.op)
    infer_constraints.expr(cx, stat.lhs, privilege)
  end
end

function infer_constraints.pass_through_stat(cx, stat) end

local infer_constraints_stat_table = {
  [ast.typed.stat.ForList]    = infer_constraints.stat_for_list,

  [ast.typed.stat.ForNum]     = infer_constraints.stat_block,
  [ast.typed.stat.While]      = infer_constraints.stat_block,
  [ast.typed.stat.Repeat]     = infer_constraints.stat_block,
  [ast.typed.stat.Block]      = infer_constraints.stat_block,

  [ast.typed.stat.If]         = infer_constraints.stat_if,

  [ast.typed.stat.Var]        = infer_constraints.stat_var,
  [ast.typed.stat.Assignment] = infer_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = infer_constraints.stat_assignment_or_reduce,

  [ast.typed.stat.Return]     = infer_constraints.pass_through_stat,
  [ast.typed.stat.Expr]       = infer_constraints.pass_through_stat,

  [ast.typed.stat]            = unreachable,
}

local infer_constraints_stat = ast.make_single_dispatch(
  infer_constraints_stat_table,
  {ast.typed.stat})

function infer_constraints.stat(cx, node)
  infer_constraints_stat(cx)(node)
end

function infer_constraints.block(cx, block)
  block.stats:map(function(stat) infer_constraints.stat(cx, stat) end)
end

function infer_constraints.top_task(node)
  local proto = node.prototype
  assert(proto)

  -- Analyze loops in the task
  local cx = parallel_task_context.new()
  infer_constraints.block(cx, node.body)
  cx:summarize_accesses()
  proto:set_partitioning_constraints(cx)

  --local task_name = node.name .. data.newtuple("parallelized")
  --local task = std.new_task(task_name)
  --local variant = task:make_variant("primary")

  --local params = terralib.newlist()
  --local privileges = terralib.newlist()
  --local region_universe = node.prototype:get_region_universe():copy()
  --local coherence_modes = node.prototype:get_coherence_modes():copy()

  --params:insertall(node.params)
  --params:map(function(param)
  --  if std.is_region(param.param_type) then
  --    local accesses = cx.field_accesses[param.symbol]
  --    if accesses then
  --      local primary_privileges = terralib.newlist()
  --      for field, privilege in accesses:items() do
  --        if privilege == "reads_writes" then
  --          primary_privileges:insert(
  --            std.privilege(std.reads, param.symbol, field))
  --          primary_privileges:insert(
  --            std.privilege(std.writes, param.symbol, field))
  --        else
  --          primary_privileges:insert(
  --            std.privilege(privilege, param.symbol, field))
  --        end
  --      end
  --      privileges:insert(primary_privileges)
  --    end
  --  end
  --end)

  --cx.image_regions:map(function(region_symbol)
  --  local accesses = cx.field_accesses[region_symbol]
  --  if accesses then
  --    params:insert(ast_util.mk_task_param(region_symbol))
  --    local image_privileges = terralib.newlist()
  --    for field, privilege in accesses:items() do
  --      image_privileges:insert(
  --        std.privilege(privilege, region_symbol, field))
  --    end
  --    privileges:insert(image_privileges)
  --    region_universe[region_symbol:gettype()] = true
  --  end
  --end)

  --task:set_type(terralib.types.functype(
  --    params:map(function(param) return param.param_type end),
  --    node.return_type,
  --    false))
  --task:set_param_symbols(params:map(function(param) return param.symbol end))

  --task:set_primary_variant(variant)
  --task:set_privileges(privileges)
  --task:set_coherence_modes(coherence_modes)
  --task:set_flags(node.flags)
  --task:set_conditions(node.conditions)
  --task:set_param_constraints(node.prototype:get_param_constraints())
  --task:set_constraints(node.prototype:get_constraints())
  --task:set_region_universe(region_universe)

  --local task_ast = ast.typed.top.Task {
  --  name = task_name,
  --  params = params,
  --  return_type = node.return_type,
  --  privileges = privileges,
  --  coherence_modes = coherence_modes,
  --  flags = node.flags,
  --  conditions = node.conditions,
  --  constraints = node.constraints,
  --  body = body,
  --  config_options = node.config_options,
  --  region_divergence = false,
  --  metadata = false,
  --  prototype = task,
  --  annotations = node.annotations {
  --    parallel = ast.annotation.Forbid { value = false },
  --  },
  --  span = node.span,
  --}

  --local task_ast_optimized = passes.optimize(task_ast)
  --task = passes.codegen(task_ast_optimized, true)

  --return task, cx
end

-- #####################################
-- ## Partitioning constraint solver
-- #################

-- The collect_constraints pass collects all partitioning constraints

local collector_context = {}

collector_context.__index = collector_context

function collector_context.new()
  local cx = {
    all_tasks = terralib.newlist(),
    task_mappings = {},
  }

  return setmetatable(cx, collector_context)
end

function collector_context:add_task(task, mapping)
  self.all_tasks:insert(task)
  self.task_mappings[task] = mapping
end

function collector_context:get_mapping(task)
  local mapping = self.task_mappings[task]
  assert(mapping ~= nil)
  return mapping
end

local collect_constraints = {}

function collect_constraints.stat_parallelize_with(cx, stat)
  report.error(stat, prefix .. " failed: parallelize_with blocks cannot be nested")
end

function collect_constraints.stat_block(cx, stat)
  collect_constraints.block(cx, stat.block)
end

function collect_constraints.stat_if(cx, stat)
  collect_constraints.block(cx, stat.then_block)
  collect_constraints.block(cx, stat.else_block)
end

local function add_task(cx, call)
  assert(call:is(ast.typed.expr.Call))
  local task = call.fn.value
  if not std.is_task(task) or
     not task:has_partitioning_constraints()
  then
    return
  end
  local params = task:get_param_symbols()
  local region_params = data.filter(
    function(param) return std.is_region(param:gettype()) end,
    params)
  local region_args = data.filter(
    function(arg) return std.is_region(std.as_read(arg.expr_type)) end,
    call.args)
  local region_arg_symbols = region_args:map(function(arg)
    assert(arg:is(ast.typed.expr.ID))
    return arg.value
  end)
  assert(#region_params == #region_arg_symbols)
  local mapping = data.newmap()
  data.zip(region_params, region_arg_symbols):map(function(pair)
    local param, arg = unpack(pair)
    mapping[param] = arg
  end)
  cx:add_task(task, mapping)
end

function collect_constraints.stat_var(cx, stat)
  if not (stat.value and stat.value:is(ast.typed.expr.Call)) then
    return
  end
  add_task(cx, stat.value)
end

function collect_constraints.stat_assignment_or_reduce(cx, stat)
  if not stat.rhs:is(ast.typed.expr.Call) then return end
  add_task(cx, stat.rhs)
end

function collect_constraints.stat_expr(cx, stat)
  if not stat.expr:is(ast.typed.expr.Call) then return end
  add_task(cx, stat.expr)
end

function collect_constraints.pass_through_stat(cx, stat) end

local collect_constraints_stat_table = {
  [ast.typed.stat.ParallelizeWith] = collect_constraints.stat_parallelize_with,

  [ast.typed.stat.Var]        = collect_constraints.stat_var,
  [ast.typed.stat.Assignment] = collect_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = collect_constraints.stat_assignment_or_reduce,
  [ast.typed.stat.Expr]       = collect_constraints.stat_expr,

  [ast.typed.stat.ForList]    = collect_constraints.stat_block,
  [ast.typed.stat.ForNum]     = collect_constraints.stat_block,
  [ast.typed.stat.While]      = collect_constraints.stat_block,
  [ast.typed.stat.Repeat]     = collect_constraints.stat_block,
  [ast.typed.stat.Block]      = collect_constraints.stat_block,
  [ast.typed.stat.MustEpoch]  = collect_constraints.stat_block,

  [ast.typed.stat.If]         = collect_constraints.stat_if,

  [ast.typed.stat]            = collect_constraints.pass_through_stat,
}

local collect_constraints_stat = ast.make_single_dispatch(
  collect_constraints_stat_table,
  {ast.typed.stat})

function collect_constraints.stat(cx, node)
  collect_constraints_stat(cx)(node)
end

function collect_constraints.block(cx, block)
  block.stats:map(function(stat) collect_constraints.stat(cx, stat) end)
end

---------------------

local solver_context = {}

solver_context.__index = solver_context

function solver_context.new()
  local cx = {
    constraints = partitioning_constraints.new(),
  }
  return setmetatable(cx, solver_context)
end

function solver_context:unify(new_constraints, region_mapping)
  if new_constraints.constraints:is_empty() then
    return
  elseif self.constraints:is_empty() then
    self.constraints = new_constraints.constraints:clone(region_mapping)
    return
  end

  local g1 = self.constraints:create_graph()
  local to_unify = new_constraints.constraints:clone(region_mapping)
  new_constraints:print_all_constraints()
  local g2 = to_unify:create_graph()
  local g = create_product_graph(g1, g2)
  local connected_components = g:get_connected_components()
  assert(#connected_components > 0)
  table.sort(connected_components, function(c1, c2) return #c1 > #c2 end)

  local range_mapping = data.newmap()
  connected_components[1]:map(function(pair)
    local dst, src = unpack(pair)
    range_mapping[src] = dst
  end)

  self.constraints:join(to_unify, range_mapping)
  self:print_all_constraints()
end

function solver_context:print_all_constraints()
  print("################")
  self.constraints:print_constraints()
  print("################")
end

---------------------

local parallelization_context = {}

parallelization_context.__index = parallelization_context

function parallelization_context.new()
  local cx = {
    mapping = data.newmap(),
  }

  return setmetatable(cx, parallelization_context)
end

local parallelize_task_calls = {}

function parallelize_task_calls.stat_parallelize_with(cx, stat)
  local collector_cx = collector_context.new()
  collect_constraints.block(collector_cx, stat.block)

  local all_tasks = collector_cx.all_tasks:map(function(task) return task end)
  table.sort(all_tasks, cmp_tasks)

  local solver_cx = solver_context.new()
  all_tasks:map(function(task)
    local constraints = task:get_partitioning_constraints()
    local mapping = collector_cx:get_mapping(task)
    solver_cx:unify(constraints, mapping)
  end)

  return ast.typed.stat.Block {
    block = stat.block,
    span = stat.span,
    annotations = stat.annotations,
  }
end

function parallelize_task_calls.stat_block(cx, stat)
  local block = parallelize_task_calls.block(cx, stat.block)
  return stat { block = block }
end

function parallelize_task_calls.stat_if(cx, stat)
  local then_block = parallelize_task_calls.block(cx, stat.then_block)
  local else_block = parallelize_task_calls.block(cx, stat.else_block)
  return stat {
    then_block = then_block,
    else_block = else_block,
  }
end

function parallelize_task_calls.stat_var(cx, stat)
  return stat
end

function parallelize_task_calls.stat_assignment_or_reduce(cx, stat)
  return stat
end

function parallelize_task_calls.stat_expr(cx, stat)
  return stat
end

function parallelize_task_calls.pass_through_stat(cx, stat)
  return stat
end

local parallelize_task_calls_stat_table = {
  [ast.typed.stat.ParallelizeWith] = parallelize_task_calls.stat_parallelize_with,

  [ast.typed.stat.Var]        = parallelize_task_calls.stat_var,
  [ast.typed.stat.Assignment] = parallelize_task_calls.stat_assignment_or_reduce,
  [ast.typed.stat.Reduce]     = parallelize_task_calls.stat_assignment_or_reduce,
  [ast.typed.stat.Expr]       = parallelize_task_calls.stat_expr,

  [ast.typed.stat.ForList]    = parallelize_task_calls.stat_block,
  [ast.typed.stat.ForNum]     = parallelize_task_calls.stat_block,
  [ast.typed.stat.While]      = parallelize_task_calls.stat_block,
  [ast.typed.stat.Repeat]     = parallelize_task_calls.stat_block,
  [ast.typed.stat.Block]      = parallelize_task_calls.stat_block,
  [ast.typed.stat.MustEpoch]  = parallelize_task_calls.stat_block,

  [ast.typed.stat.If]         = parallelize_task_calls.stat_if,

  [ast.typed.stat]            = parallelize_task_calls.pass_through_stat,
}

local parallelize_task_calls_stat = ast.make_single_dispatch(
  parallelize_task_calls_stat_table,
  {ast.typed.stat})

function parallelize_task_calls.stat(cx, node)
  return parallelize_task_calls_stat(cx)(node)
end

function parallelize_task_calls.block(cx, block)
  local stats = terralib.newlist()
  block.stats:map(function(stat)
    local result = parallelize_task_calls.stat(cx, stat)
    if terralib.islist(result) then
      stats:insertall(result)
    else
      stats:insert(result)
    end
  end)
  return block { stats = stats }
end

function parallelize_task_calls.top_task(node)
  local cx = parallelization_context.new()
  local body = parallelize_task_calls.block(cx, node.body)
  return node { body = body }
end

local parallelize_tasks = {}

function parallelize_tasks.entry(node)
  if node:is(ast.typed.top.Task) then
    if node.annotations.parallel:is(ast.annotation.Demand) then
      assert(node.config_options.leaf)
      assert(node.metadata)
      infer_constraints.top_task(node)
      return node
    elseif not node.config_options.leaf then
      return parallelize_task_calls.top_task(node)
    else
      return node
    end
  else
    return node
  end
end

parallelize_tasks.pass_name = "parallelize_tasks"

return parallelize_tasks
