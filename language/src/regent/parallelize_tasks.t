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

local function intersect(set1, set2)
  local result = data.newmap()
  for k, _ in set1:items() do
    if set2[k] then result[k] = true end
  end
  return result
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
  for v, l in self.vertices:items() do
    node_ids[v] = next_id
    print("    " .. tostring(next_id) .. " [ label = \"" .. tostring(v) .. " : " .. tostring(l) .. "\" ];")
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
      if l1:unifiable(l2) then
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
        local label1 = g1:connected(va1, vb1)
        local label2 = g2:connected(va2, vb2)
        local intersect = label1 and label2 and intersect(label1, label2)
        if intersect and not intersect:is_empty() then
          local edges = find_or_create(g.edges, va)
          edges[vb] = intersect
        end
      end
    end
  end
  return g
end

local partition_info = {}

partition_info.__index = partition_info

function partition_info.new(region_symbol, disjoint, complete)
  local tuple = {
    region = region_symbol,
    disjoint = disjoint or false,
    complete = complete or false,
  }
  return setmetatable(tuple, partition_info)
end

function partition_info:clone(mapping)
  local region = self.region
  if mapping then region = mapping(region) end
  assert(region ~= nil)
  return partition_info.new(region, self.disjoint, self.complete)
end

function partition_info:meet_disjointness(disjoint)
  self.disjoint = self.disjoint or disjoint
end

function partition_info:meet_completeness(complete)
  self.complete = self.complete or complete
end

function partition_info:__tostring()
  local disjoint = (self.disjoint and "D") or "A"
  local complete = (self.complete and "C") or "I"
  return "partition(" .. tostring(self.region) ..  "," ..
    disjoint .. "," .. complete .. ")"
end

function partition_info:unifiable(other)
  return self.region == other.region
end

function partition_info:__eq(other)
  return self.region == other.region and
         self.disjoint == other.disjoint and
         self.complete == other.complete
end

local require_disjoint_partition = {
  [std.writes]     = true,
  ["reads_writes"] = true,
}

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

local function render_subset_constraint(src, dst)
  return tostring(src) .. " <= " .. tostring(dst)
end

local function render_image_constraint(src, key, dst)
  local region_symbol, field_path = unpack(key)
  return tostring(region_symbol) .. "[" .. tostring(src) .. "]." ..
    field_path:mkstring("", ".", "") .. " <= " .. tostring(dst)
end

local function render_analytic_constraint(src, offset, dst)
  if std.is_symbol(offset[#offset]) then
    local symbol = offset[#offset]
    local o = offset:slice(1, #offset - 1)
    local divider =
      (std.is_region(symbol:gettype()) and tostring(symbol) .. ".ispace.bounds") or
      (std.is_ispace(symbol:gettype()) and tostring(symbol) .. ".bounds") or
      assert(false)
    return "(" ..tostring(src) .. " + {" ..
      o:mkstring("",",","") .. "}) % " ..
      divider .. " <= " .. tostring(dst)
  else
    return tostring(src) .. " + {" ..
      offset:mkstring("",",","") .. "} <= " .. tostring(dst)
  end
end

function partitioning_constraints:print_constraints()
  print("* partitions:")
  for range, partition in self.ranges:items() do
    print("    " .. tostring(range) .. " : " .. tostring(partition))
  end
  print("* subset constraints:")
  for src, all_constraints in self.subset_constraints:items() do
    for region_symbol, dst in all_constraints:items() do
      print("    " .. render_subset_constraint(src, dst))
    end
  end
  print("* image constraints:")
  for src, all_constraints in self.image_constraints:items() do
    for key, dst in all_constraints:items() do
      print("    " .. render_image_constraint(src, key, dst))
    end
  end
  print("* analytic constraints:")
  for src, constraints in self.analytic_constraints:items() do
    for offset, dst in constraints:items() do
      print("    " .. render_analytic_constraint(src, offset, dst))
    end
  end
end

function partitioning_constraints:set_partition(range, partition)
  assert(self.ranges[range] == nil)
  self.ranges[range] = partition
end

function partitioning_constraints:get_partition(range)
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
  for range, partition in self.ranges:items() do
    result:set_partition(range, partition:clone(map_region))
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

function partitioning_constraints:unify_ranges(range_mapping)
  local result = partitioning_constraints.new()
  for range, partition in self.ranges:items() do
    local new_range = range_mapping[range] or range
    local new_partition = result:get_partition(new_range)
    if new_partition == nil then
      result:set_partition(new_range, partition)
    else
      new_partition:meet_disjointness(partition.disjoint)
      new_partition:meet_completeness(partition.complete)
    end
  end
  for src, constraints in self.subset_constraints:items() do
    local new_src = range_mapping[src] or src
    local result_constraints = find_or_create(result.subset_constraints, new_src)
    for region_symbol, dst in constraints:items() do
      local new_dst = range_mapping[dst] or dst
      if new_src == new_dst then
        return false, render_subset_constraint(new_src, new_dst)
      end
      result_constraints[region_symbol] = new_dst
    end
  end
  for src, constraints in self.image_constraints:items() do
    local new_src = range_mapping[src] or src
    local result_constraints = find_or_create(result.image_constraints, new_src)
    for key, dst in constraints:items() do
      local new_dst = range_mapping[dst] or dst
      if new_src == new_dst then
        return false, render_image_constraint(new_src, key, new_dst)
      end
      result_constraints[key] = new_dst
    end
  end
  for src, constraints in self.analytic_constraints:items() do
    local new_src = range_mapping[src] or src
    local result_constraints = find_or_create(result.analytic_constraints, new_src)
    for offset, dst in constraints:items() do
      local new_dst = range_mapping[dst] or dst
      if new_src == new_dst then
        return false, render_analytic_constraint(new_src, offset, new_dst)
      end
      result_constraints[offset] = new_dst
    end
  end
  return result
end

function partitioning_constraints:propagate_disjointness()
  local function propagate(all_constraints, render)
    for src, constraints in all_constraints:items() do
      local src_partition = self.ranges[src]
      local disjoint_images = data.newmap()
      local num_disjoint_images = 0
      for key, dst in constraints:items() do
        local dst_partition = self.ranges[dst]
        if dst_partition.disjoint then
          disjoint_images[key] = dst
          num_disjoint_images = num_disjoint_images + 1
          src_partition:meet_disjointness(dst_partition.disjoint)
        end
      end
      if num_disjoint_images > 1 then
        local error_message =
          tostring(src)  .. " : " .. tostring(src_partition) .. ". "
        for _, dst in disjoint_images:items() do
          local dst_partition = self.ranges[dst]
          error_message = error_message ..
            tostring(dst)  .. " : " .. tostring(dst_partition) .. ". "
        end
        for key, dst in disjoint_images:items() do
          error_message = error_message .. render(src, key, dst) .. " /\\ "
        end
        return false, error_message
      end
    end
    return true
  end

  local satisfiable, error_message =
    propagate(self.image_constraints, render_image_constraint)
  if not satisfiable then return satisfiable, error_message end
  local satisfiable, error_message =
    propagate(self.analytic_constraints, render_analytic_constraint)
  if not satisfiable then return satisfiable, error_message end

  return true
end

function partitioning_constraints:create_graph()
  local g = graph.new()
  for range, partition in self.ranges:items() do
    g.vertices[range] = partition
  end
  for src, constraints in self.subset_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for region_symbol, dst in constraints:items() do
      local labels = find_or_create(edges, dst)
      labels[true] = true
    end
  end
  for src, constraints in self.image_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for key, dst in constraints:items() do
      local labels = find_or_create(edges, dst)
      labels[key] = true
    end
  end
  for src, constraints in self.analytic_constraints:items() do
    local edges = find_or_create(g.edges, src)
    for key, dst in constraints:items() do
      local labels = find_or_create(edges, dst)
      labels[key] = true
    end
  end
  return g
end

function partitioning_constraints:join(to_join, mapping)
  for range, partition in to_join.ranges:items() do
    local my_range = mapping[range]
    if my_range == nil then
      self.ranges[range] = partition
    else
      local my_partition = self.ranges[my_range]
      my_partition:meet_disjointness(partition.disjoint)
      my_partition:meet_completeness(partition.complete)
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

function partitioning_constraints:get_complexity()
  local complexity = 0
  for _, all_constraints in self.subset_constraints:items() do
    for _, _ in all_constraints:items() do
      complexity = complexity + 1
    end
  end
  for _, all_constraints in self.image_constraints:items() do
    for _, _ in all_constraints:items() do
      complexity = complexity + 1
    end
  end
  for _, constraints in self.analytic_constraints:items() do
    for _, _ in constraints:items() do
      complexity = complexity + 1
    end
  end
  return complexity
end

local parallel_task_context = {}

parallel_task_context.__index = parallel_task_context

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

function parallel_task_context.new(task)
  local cx = {
    task                     = task,
    env                      = data.newmap(),
    back_edges               = data.newmap(),
    sources                  = data.newmap(),
    sources_by_regions       = data.newmap(),

    field_accesses           = data.newmap(),
    field_accesses_summary   = data.newmap(),

    constraints              = partitioning_constraints.new(),

    -- Complexity of the system of constraints is a pair of
    --  # of subset/image/analytic constraints,
    --  # of distinct field accesses
    complexity               = false,
  }

  return setmetatable(cx, parallel_task_context)
end

local function cmp_complexity(c1, c2)
  if c1[1] > c2[1] then return true
  elseif c1[1] < c2[1] then return false
  else
    if c1[2] > c2[2] then return true
    elseif c1[2] < c2[2] then return false end
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
  print("* sources by regions:")
  for region, source in self.sources_by_regions:items() do
    print("    " .. tostring(region) .. " => " .. tostring(source))
  end
  self.constraints:print_constraints()
  --print("* accesses:")
  --for region_symbol, accesses_summary in self.field_accesses_summary:items() do
  --  for field_path, summary in accesses_summary:items() do
  --    local range, privilege = unpack(summary)
  --    print("    " .. tostring(region_symbol) .. "[" .. tostring(range) .. "]." ..
  --      field_path:mkstring("", ".", "") .. " @ " .. tostring(privilege))
  --  end
  --end
  print("* accesses:")
  for region_symbol, all_accesses in self.field_accesses:items() do
    for field_path, accesses in all_accesses:items() do
      for range, privilege in accesses:items() do
        print("    " .. tostring(region_symbol) .. "[" .. tostring(range) .. "]." ..
          field_path:mkstring("", ".", "") .. " @ " .. tostring(privilege))
      end
    end
  end
  print("================")
end

function parallel_task_context:add_field_access(region_symbol, range, field_path, privilege)
  local partition = self.constraints:get_partition(range)
  if not (partition == nil or partition.region == region_symbol) then
    local new_range =
      self.constraints:find_or_create_subset_constraint(range, region_symbol)
    self.back_edges[new_range] = range
    range = new_range
    partition = nil
  end

  local all_field_accesses = find_or_create(self.field_accesses, region_symbol)
  local field_accesses = find_or_create(all_field_accesses, field_path)
  local join = std.meet_privilege(field_accesses[range], privilege)
  field_accesses[range] = join
  if partition == nil then
    local disjoint = require_disjoint_partition[join]
    self:set_partition(range, partition_info.new(region_symbol, disjoint, false))
  else
    partition:meet_disjointness(require_disjoint_partition[join])
  end
end

function parallel_task_context:summarize_accesses()
  local range_mapping = {}
  local needs_unification = false
  local num_accesses = 0
  for region_symbol, all_field_accesses in self.field_accesses:items() do
    local accesses_summary =
      find_or_create(self.field_accesses_summary, region_symbol)
    for field_path, accesses in all_field_accesses:items() do
      local ranges = terralib.newlist()
      local join = nil
      for range, privilege in accesses:items() do
        ranges:insert(range)
        join = std.meet_privilege(join, privilege)
      end
      local disjoint = require_disjoint_partition[join]
      if disjoint and #ranges > 1 then
        local equivalence_class = ranges[1]
        needs_unification = true
        accesses_summary[field_path] = { equivalence_class, join }
        ranges:map(function(range)
          self.constraints:get_partition(range):meet_disjointness(disjoint)
          range_mapping[range] = equivalence_class
        end)
      else
        accesses_summary[field_path] = { ranges, join }
      end
      num_accesses = num_accesses + 1
    end
  end

  if needs_unification then
    local unified, error_message = self.constraints:unify_ranges(range_mapping)
    if not unified then
      self.constraints:print_constraints()
      report.error(self.task,
        prefix .. " failed: found an unsatisfiable constraint during unification: " ..
        error_message)
    end
    self.constraints = unified
  end

  local satisfiable, error_message = self.constraints:propagate_disjointness()
  if not satisfiable then
    report.error(self.task,
      prefix .. " failed: found an unsatisfiable constraint during unification: " ..
      error_message)
  end

  self.complexity = data.newtuple(self.constraints:get_complexity(), num_accesses)
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

function parallel_task_context:set_partition(range, partition)
  if range == range_complex then return end
  self.constraints:set_partition(range, partition)
end

function parallel_task_context:update_partition(range, partition)
  if range == range_complex then return end
  local my_partition = self.constraints:get_partition(range)
  if my_partition == nil then
    self.constraints:set_partition(range, partition)
  else
    my_partition:meet_disjointness(partition.disjoint)
    my_partition:meet_completeness(partition.complete)
  end
end

function parallel_task_context:find_or_create_source_range(region)
  local range = find_or_create(self.sources_by_regions, region, new_range)
  self.sources[range] = true
  return range
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
    local region_symbol = stat.value.value
    local range = cx:find_or_create_source_range(region_symbol)
    cx:set_range(stat.symbol, range)
    local disjoint = stat.metadata.reductions and #stat.metadata.reductions > 0
    cx:update_partition(range, partition_info.new(region_symbol, disjoint, true))
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
  local cx = parallel_task_context.new(node)
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
    sources = data.newmap(),
    sources_by_regions = data.newmap(),
    constraints = partitioning_constraints.new(),
  }
  return setmetatable(cx, solver_context)
end

local function find_unifiable_ranges(constraints1, constraints2, source1, source2)
  -- TODO: Need to check if the resulting constraints are unsatisfiable
  local mapping = data.newmap()
  local worklist = terralib.newlist({data.newtuple(source1, source2)})
  local idx = 1
  while idx <= #worklist do
    local range1, range2 = unpack(worklist[idx])
    mapping[range1] = range2
    idx = idx + 1
    local all_constraints1 = constraints1.image_constraints[range1]
    local all_constraints2 = constraints2.image_constraints[range2]
    if all_constraints1 ~= nil and all_constraints2 ~= nil then
      for key, dst_range1 in all_constraints1:items() do
        local dst_range2 = all_constraints2[key]
        if dst_range2 ~= nil then
          worklist:insert(data.newtuple(dst_range1, dst_range2))
        end
      end
    end
  end
  return true, mapping
end

function solver_context:unify(new_constraints, region_mapping)
  if new_constraints.constraints:is_empty() then
    return
  elseif self.constraints:is_empty() then
    for source, _ in new_constraints.sources:items() do
      self.sources[source] = true
    end
    for region, source in new_constraints.sources_by_regions:items() do
      local my_region = region_mapping[region]
      self.sources_by_regions[my_region] = source
    end
    self.constraints = new_constraints.constraints:clone(region_mapping)
    return
  end

  local range_mapping = data.newmap()
  local to_unify = new_constraints.constraints:clone(region_mapping)
  for region, source in new_constraints.sources_by_regions:items() do
    local my_region = region_mapping[region]
    local my_source = self.sources_by_regions[my_region]
    if my_source ~= nil then
      local unifiable, mapping =
        find_unifiable_ranges(to_unify, self.constraints, source, my_source)
      if unifiable then
        for src, tgt in mapping:items() do
          range_mapping[src] = tgt
        end
      end
    else
      local partition = to_unify:get_partition(source)
      for my_range, my_partition in self.constraints.ranges:items() do
        if partition == my_partition then
          range_mapping[source] = my_range
          break
        end
      end
    end
  end

  self.constraints:join(to_unify, range_mapping)
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
  if #all_tasks == 0 then
    return ast.typed.stat.Block {
      block = stat.block,
      span = stat.span,
      annotations = stat.annotations,
    }
  end

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
