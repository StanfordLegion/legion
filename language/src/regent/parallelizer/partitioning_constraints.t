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

local data = require("common/data")
local std = require("regent/std")

local partition_info = require("regent/parallelizer/partition_info")
local ranges         = require("regent/parallelizer/ranges")

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

local function set_to_string(map)
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
    print("    " .. tostring(next_id) .. " [ label = \"" .. tostring(v) ..
        " : " .. tostring(l) .. "\" ];")
    next_id = next_id + 1
  end
  for src, edges in self.edges:items() do
    for dst, key in edges:items() do
      local src_id = node_ids[src]
      local dst_id = node_ids[dst]
      print("    " .. tostring(src_id) .. " -> " .. tostring(dst_id) ..
        " [ label = \"" .. set_to_string(key) .. "\" ];")
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
  return find_or_create(constraints, region_symbol, ranges.new)
end

function partitioning_constraints:find_or_create_image_constraint(src_range, region_symbol, field_path)
  local key = data.newtuple(region_symbol, field_path)
  local constraints = find_or_create(self.image_constraints, src_range)
  return find_or_create(constraints, key, ranges.new)
end

function partitioning_constraints:find_or_create_analytic_constraint(src_range, offset)
  local constraints = find_or_create(self.analytic_constraints, src_range)
  return find_or_create(constraints, offset, ranges.new)
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

function partitioning_constraints:remove_unnecessary_constraints()
  local function remove(all_constraints)
    local to_remove = terralib.newlist()
    for src, constraints in all_constraints:items() do
      for key, dst in constraints:items() do
        if self.ranges[dst] == nil then
          to_remove:insert({src, key})
        end
      end
    end
    to_remove:map(function(pair)
      local src, key = unpack(pair)
      all_constraints[src][key] = nil
    end)
  end

  remove(self.subset_constraints)
  remove(self.image_constraints)
  remove(self.analytic_constraints)
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
    if my_range == range then
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

function partitioning_constraints:find_path_to_disjoint_child(range)
  local found_any = false
  local found_path = nil
  local function visit_children(edges)
    if edges == nil then return end
    for key, child in edges:items() do
      local found, path = self:find_path_to_disjoint_child(child)
      assert(not (found_any and found))
      if found then
        found_any = found
        found_path = path
        found_path:insert({key, range})
      end
    end
  end
  visit_children(self.subset_constraints[range])
  visit_children(self.image_constraints[range])
  visit_children(self.analytic_constraints[range])
  if found_any then
    assert(found_path ~= nil)
    return found_any, found_path
  else
    local partition = self:get_partition(range)
    return partition.disjoint, terralib.newlist({{nil, range}})
  end
end

function partitioning_constraints:find_paths_to_disjoint_children(sources)
  local paths = terralib.newlist()
  for source, _ in sources:items() do
    local found, path = self:find_path_to_disjoint_child(source)
    if found then
      assert(path ~= nil)
      paths:insert(path)
    else
      paths:insert(terralib.newlist({{nil, source}}))
    end
  end
  return paths
end

return partitioning_constraints
