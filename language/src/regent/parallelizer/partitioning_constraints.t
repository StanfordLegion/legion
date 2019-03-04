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

local common_ast = require("common/ast")
local data = require("common/data")
local std = require("regent/std")

local hash_set       = require("regent/parallelizer/hash_set")
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

local constraint_type = common_ast.make_factory("constraint_type")
constraint_type:leaf("Subset"):set_memoize():set_print_custom("subset")
constraint_type:leaf("Image"):set_memoize():set_print_custom("image")
constraint_type:leaf("Affine"):set_memoize():set_print_custom("affine")

constraint_type.subset = constraint_type.Subset {}
constraint_type.image  = constraint_type.Image {}
constraint_type.affine = constraint_type.Affine {}

local constraint_info = {}

constraint_info.__index = constraint_info

function constraint_info.new(type, info)
  local cx = {
    type = type,
    info = info,
  }
  return setmetatable(cx, constraint_info)
end

function constraint_info:hash()
  return tostring(self.type) .. self.info:hash()
end

function constraint_info:is_affine()
  return self.type:is(constraint_type.Affine)
end

function constraint_info:is_image()
  return self.type:is(constraint_type.Image)
end

function constraint_info:is_subset()
  return self.type:is(constraint_type.Subset)
end

function constraint_info:clone(mapping)
  local info = self.info
  if self:is_subset() then
    info = mapping(info)
  elseif self:is_image() then
    local region_symbol, field_path = unpack(info)
    info = data.newtuple(mapping(region_symbol), field_path)
  elseif self:is_affine() then
    info = data.newtuple(unpack(info))
    if std.is_symbol(info[#info]) then
      info[#info] = mapping(info[#info])
    end
  else
    assert(false)
  end
  return constraint_info.new(self.type, info)
end

local partitioning_constraints = {}

partitioning_constraints.__index = partitioning_constraints

function partitioning_constraints.new()
  local cx = {
    ranges      = data.newmap(),
    constraints = data.newmap(),
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
    local divider = tostring(symbol)
    return "(" ..tostring(src) .. " + {" ..
      o:mkstring("",",","") .. "}) % " ..
      divider .. " <= " .. tostring(dst)
  else
    return tostring(src) .. " + {" ..
      offset:mkstring("",",","") .. "} <= " .. tostring(dst)
  end
end

local function render_constraint(src, info, dst)
  if info:is_subset() then
    return render_subset_constraint(src, dst)
  elseif info:is_image() then
    return render_image_constraint(src, info.info, dst)
  elseif info:is_affine() then
    return render_analytic_constraint(src, info.info, dst)
  else
    assert(false)
  end
end

function partitioning_constraints:print_constraints()
  print("* partitions:")
  for range, partition in self.ranges:items() do
    print("    " .. tostring(range) .. " : " .. tostring(partition))
  end
  print("* constraints:")
  for src, all_constraints in self.constraints:items() do
    for key, dst in all_constraints:items() do
      print("    " .. render_constraint(src, key, dst))
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

function partitioning_constraints:is_constrained(range)
  return self.constraints[range] ~= nil
end

function partitioning_constraints:find_or_create_subset_constraint(src_range, region_symbol)
  local constraints = find_or_create(self.constraints, src_range)
  local key = constraint_info.new(constraint_type.subset, region_symbol)
  return find_or_create(constraints, key, ranges.new)
end

function partitioning_constraints:find_or_create_image_constraint(src_range, region_symbol, field_path)
  local constraints = find_or_create(self.constraints, src_range)
  local key = constraint_info.new(constraint_type.image, data.newtuple(region_symbol, field_path))
  return find_or_create(constraints, key, ranges.new)
end

function partitioning_constraints:find_or_create_analytic_constraint(src_range, offset)
  local constraints = find_or_create(self.constraints, src_range)
  local key = constraint_info.new(constraint_type.affine, offset)
  return find_or_create(constraints, key, ranges.new)
end

local function lift(tbl)
  if tbl ~= nil then
    return function(symbol) return tbl[symbol] or symbol end
  else
    return function(symbol) return symbol end
  end
end

function partitioning_constraints:join(to_join, mapping)
  local mapping = lift(mapping)

  for range, partition in to_join.ranges:items() do
    local new_range = mapping(range)
    local new_partition = partition:clone(mapping)

    local my_partition = self.ranges[new_range]
    if my_partition == nil then
      self.ranges[new_range] = new_partition
    else
      assert(my_partition.region == new_partition.region)
      my_partition:meet_disjointness(new_partition.disjoint)
      my_partition:meet_completeness(new_partition.complete)
    end
  end

  for src, constraints in to_join.constraints:items() do
    local new_src = mapping(src)
    local my_constraints = find_or_create(self.constraints, new_src)

    for info, dst in constraints:items() do
      local new_info = info:clone(mapping)
      local new_dst = mapping(dst)
      assert(my_constraints[new_info] == nil or my_constraints[new_info] == new_dst)
      my_constraints[new_info] = new_dst
    end
  end
end

function partitioning_constraints:clone(mapping)
  local result = partitioning_constraints.new()
  result:join(self, mapping)
  return result
end

function partitioning_constraints:remove_unnecessary_constraints()
  local to_remove = terralib.newlist()
  for src, constraints in self.constraints:items() do
    for key, dst in constraints:items() do
      if self.ranges[dst] == nil then
        to_remove:insert({src, key})
      end
    end
  end
  to_remove:map(function(pair)
    local src, key = unpack(pair)
    self.constraints[src][key] = nil
  end)
end

function partitioning_constraints:propagate_disjointness()
  for src, constraints in self.constraints:items() do
    local src_partition = self.ranges[src]
    local disjoint_images = data.newmap()
    local num_disjoint_images = 0
    for key, dst in constraints:items() do
      local dst_partition = self.ranges[dst]
      if dst_partition.disjoint then
        disjoint_images[key] = dst
        num_disjoint_images = num_disjoint_images + 1
        src_partition:meet_disjointness(dst_partition.disjoint)
        -- TODO: We assume that we will synthesize an equal partition for this
        dst_partition:meet_completeness(true)
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
        error_message = error_message .. render_constraint(src, key, dst) .. " /\\ "
      end
      return false, error_message
    end
  end
  return true
end

function partitioning_constraints:dump()
  print("digraph {")
  local next_id = 1
  local node_ids = {}
  for range, partition in self.ranges:items() do
    node_ids[range] = next_id
    print("    " .. tostring(next_id) .. " [ label = \"" .. tostring(range) ..
        " : " .. tostring(partition) .. "\" ];")
    next_id = next_id + 1
  end
  for src, constraints in self.constraints:items() do
    for key, dst in constraints:items() do
      local src_id = node_ids[src]
      local dst_id = node_ids[dst]
      print("    " .. tostring(src_id) .. " -> " .. tostring(dst_id) ..
        " [ label = \"" .. tostring(key) .. "\" ];")
    end
  end
  print("}")
end

function partitioning_constraints:get_complexity()
  local complexity = 0
  for _, constraints in self.constraints:items() do
    for _, _ in constraints:items() do
      complexity = complexity + 1
    end
  end
  return complexity
end

function partitioning_constraints:find_path_to_disjoint_child(range)
  local found_any = false
  local found_path = nil

  local edges = self.constraints[range]
  if edges ~= nil then
    for info, child in edges:items() do
      local found, path = self:find_path_to_disjoint_child(child)
      assert(not (found_any and found))
      if found then
        found_any = found
        found_path = path
        found_path:insert({info, range})
      end
    end
  end

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
  sources:foreach(function(source)
    local found, path = self:find_path_to_disjoint_child(source)
    if found then
      assert(path ~= nil)
      paths:insert(path)
    else
      paths:insert(terralib.newlist({{nil, source}}))
    end
  end)
  return paths
end

return partitioning_constraints
