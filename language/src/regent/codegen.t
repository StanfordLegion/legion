-- Copyright 2018 Stanford University, NVIDIA Corporation
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

-- Regent Code Generation

local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")
local codegen_hooks = require("regent/codegen_hooks")
local cudahelper = require("regent/cudahelper")
local openmphelper = require("regent/openmphelper")
local pretty = require("regent/pretty")

-- Configuration Variables

-- Setting this flag to true allows the compiler to emit aligned
-- vector loads and stores, and is safe for use only with the general
-- LLR (because the shared LLR does not properly align instances).
local aligned_instances = std.config["aligned-instances"]

-- Setting this flag to true allows the compiler to use cached index
-- iterators, which are generally faster (they only walk the index
-- space bitmask once) but are only safe when the index space itself
-- is never modified (by allocator or deleting elements).
local cache_index_iterator = std.config["cached-iterators"]

-- Setting this flag to true directs the compiler to emit assertions
-- whenever two regions being placed in different physical regions
-- would require the use of the divergence-safe code path to be used.
local dynamic_branches_assert = std.config["no-dynamic-branches-assert"]

-- Setting this flag directs the compiler to emit bounds checks on all
-- pointer accesses. This is independent from the runtime's bounds
-- checks flag as the compiler does not use standard runtime
-- accessors.
local bounds_checks = std.config["bounds-checks"]

local emergency = std.config["emergency-gc"]

local function manual_gc() end

if emergency then
  local emergency_threshold = 1024 * 1024
  -- Manually call GC whenever the current heap size exceeds threshold
  manual_gc = function()
    if collectgarbage("count") >= emergency_threshold then
      collectgarbage("collect")
      collectgarbage("collect")
    end
  end
end

local codegen = {}

-- load Legion dynamic library
local c = std.c

local context = {}

function context:__index(field)
  local value = context[field]
  if value ~= nil then
    return value
  end
  error("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex(field, value)
  error("context has no field '" .. field .. "' (in assignment)", 2)
end

function context:new_local_scope(divergence, must_epoch, must_epoch_point, break_label)
  assert(not (self.must_epoch and must_epoch))
  divergence = self.divergence or divergence or false
  must_epoch = self.must_epoch or must_epoch or false
  must_epoch_point = self.must_epoch_point or must_epoch_point or false
  if not break_label then
    break_label = self.break_label or false
  end
  assert((must_epoch == false) == (must_epoch_point == false))
  return setmetatable({
    variant = self.variant,
    expected_return_type = self.expected_return_type,
    constraints = self.constraints,
    task = self.task,
    task_meta = self.task_meta,
    leaf = self.leaf,
    divergence = divergence,
    must_epoch = must_epoch,
    must_epoch_point = must_epoch_point,
    break_label = break_label,
    context = self.context,
    runtime = self.runtime,
    ispaces = self.ispaces:new_local_scope(),
    regions = self.regions:new_local_scope(),
    lists_of_regions = self.lists_of_regions:new_local_scope(),
    cleanup_items = terralib.newlist(),
  }, context)
end

function context:new_task_scope(expected_return_type, constraints, leaf, task_meta, task, ctx, runtime)
  assert(expected_return_type and task and ctx and runtime)
  return setmetatable({
    variant = self.variant,
    expected_return_type = expected_return_type,
    constraints = constraints,
    task = task,
    task_meta = task_meta,
    leaf = leaf,
    divergence = false,
    must_epoch = false,
    must_epoch_point = false,
    break_label = false,
    context = ctx,
    runtime = runtime,
    ispaces = symbol_table.new_global_scope({}),
    regions = symbol_table.new_global_scope({}),
    lists_of_regions = symbol_table.new_global_scope({}),
    cleanup_items = terralib.newlist(),
  }, context)
end

function context.new_global_scope(variant)
  variant = variant or false
  return setmetatable({
    variant = variant,
  }, context)
end

function context:check_divergence(region_types, field_paths)
  if not self.divergence then
    return false
  end
  for _, divergence in ipairs(self.divergence) do
    local contained = true
    for _, r in ipairs(region_types) do
      if not divergence.group[r] then
        contained = false
        break
      end
    end
    for _, field_path in ipairs(field_paths) do
      if not divergence.valid_fields[data.hash(field_path)] then
        contained = false
        break
      end
    end
    if contained then
      return true
    end
  end
  return false
end

local ispace = setmetatable({}, { __index = function(t, k) error("ispace has no field " .. tostring(k), 2) end})
ispace.__index = ispace

function context:has_ispace(ispace_type)
  if not rawget(self, "ispaces") then
    error("not in task context", 2)
  end
  return self.ispaces:safe_lookup(ispace_type)
end

function context:ispace(ispace_type)
  if not rawget(self, "ispaces") then
    error("not in task context", 2)
  end
  return self.ispaces:lookup(nil, ispace_type)
end

function context:add_ispace_root(ispace_type, index_space, index_iterator,
                                 bounds)
  if not self.ispaces then
    error("not in task context", 2)
  end
  if self:has_ispace(ispace_type) then
    error("ispace " .. tostring(ispace_type) .. " already defined in this context", 2)
  end

  self.ispaces:insert(
    nil,
    ispace_type,
    setmetatable(
      {
        index_space = index_space,
        index_partition = nil,
        index_iterator = index_iterator,
        root_ispace_type = ispace_type,
        bounds = bounds,
      }, ispace))
end

function context:add_ispace_subispace(ispace_type, index_space, index_iterator,
                                      parent_ispace_type, bounds)
  if not self.ispaces then
    error("not in task context", 2)
  end
  if self:has_ispace(ispace_type) then
    error("ispace " .. tostring(ispace_type) .. " already defined in this context", 2)
  end
  if not self:ispace(parent_ispace_type) then
    error("parent to ispace " .. tostring(ispace_type) .. " not defined in this context", 2)
  end

  self.ispaces:insert(
    nil,
    ispace_type,
    setmetatable(
      {
        index_space = index_space,
        index_iterator = index_iterator,
        root_ispace_type = self:ispace(parent_ispace_type).root_ispace_type,
        bounds = bounds,
      }, ispace))
end

local region = setmetatable({}, { __index = function(t, k) error("region has no field " .. tostring(k), 2) end})
region.__index = region

function context:has_region(region_type)
  if not rawget(self, "regions") then
    error("not in task context", 2)
  end
  return self.regions:safe_lookup(region_type)
end

function context:region(region_type)
  if not rawget(self, "regions") then
    error("not in task context", 2)
  end
  return self.regions:lookup(nil, region_type)
end

function context:add_region_root(region_type, logical_region, field_paths,
                                 privilege_field_paths, field_privileges, field_types,
                                 field_ids, fields_are_scratch, physical_regions,
                                 base_pointers, strides)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  if not self:has_ispace(region_type:ispace()) then
    error("ispace of region " .. tostring(region_type) .. " not defined in this context", 2)
  end
  self.regions:insert(
    nil,
    region_type,
    setmetatable(
      {
        logical_region = logical_region,
        field_paths = field_paths,
        privilege_field_paths = privilege_field_paths,
        field_privileges = field_privileges,
        field_types = field_types,
        field_ids = field_ids,
        fields_are_scratch = fields_are_scratch,
        physical_regions = physical_regions,
        base_pointers = base_pointers,
        strides = strides,
        root_region_type = region_type,
      }, region))
end

function context:add_region_subregion(region_type, logical_region,
                                      parent_region_type)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  if not self:has_ispace(region_type:ispace()) then
    error("ispace of region " .. tostring(region_type) .. " not defined in this context", 2)
  end
  if not self:region(parent_region_type) then
    error("parent to region " .. tostring(region_type) .. " not defined in this context", 2)
  end
  self.regions:insert(
    nil,
    region_type,
    setmetatable(
      {
        logical_region = logical_region,
        field_paths = self:region(parent_region_type).field_paths,
        privilege_field_paths = self:region(parent_region_type).privilege_field_paths,
        field_privileges = self:region(parent_region_type).field_privileges,
        field_types = self:region(parent_region_type).field_types,
        field_ids = self:region(parent_region_type).field_ids,
        fields_are_scratch = self:region(parent_region_type).fields_are_scratch,
        physical_regions = self:region(parent_region_type).physical_regions,
        base_pointers = self:region(parent_region_type).base_pointers,
        strides = self:region(parent_region_type).strides,
        root_region_type = self:region(parent_region_type).root_region_type,
      }, region))
end

function region:field_type(field_path)
  local field_type = self.field_types[field_path:hash()]
  assert(field_type)
  return field_type
end

function region:field_id(field_path)
  local field_id = self.field_ids[field_path:hash()]
  assert(field_id)
  return field_id
end

function region:field_is_scratch(field_path)
  local field_is_scratch = self.fields_are_scratch[field_path:hash()]
  assert(field_is_scratch ~= nil)
  return field_is_scratch
end

function region:physical_region(field_path)
  local physical_region = self.physical_regions[field_path:hash()]
  assert(physical_region)
  return physical_region
end

function region:base_pointer(field_path)
  local base_pointer = self.base_pointers[field_path:hash()]
  assert(base_pointer)
  return base_pointer
end

function region:stride(field_path)
  local stride = self.strides[field_path:hash()]
  assert(stride)
  return stride
end

local list_of_regions = setmetatable({}, { __index = function(t, k) error("list of regions has no field " .. tostring(k), 2) end})
list_of_regions.__index = list_of_regions

function context:has_list_of_regions(list_type)
  if not rawget(self, "lists_of_regions") then
    error("not in task context", 2)
  end
  return self.lists_of_regions:safe_lookup(list_type)
end

function context:list_of_regions(list_type)
  if not rawget(self, "lists_of_regions") then
    error("not in task context", 2)
  end
  return self.lists_of_regions:lookup(nil, list_type)
end

function context:add_list_of_regions(list_type, list_of_logical_regions,
                                     field_paths, privilege_field_paths,
                                     field_privileges, field_types, field_ids, fields_are_scratch)
  if not self.lists_of_regions then
    error("not in task context", 2)
  end
  if self:has_list_of_regions(list_type) then
    error("region " .. tostring(list_type) .. " already defined in this context", 2)
  end
  assert(list_of_logical_regions and
           field_paths and privilege_field_paths and
           field_privileges and field_types and field_ids)
  self.lists_of_regions:insert(
    nil,
    list_type,
    setmetatable(
      {
        list_of_logical_regions = list_of_logical_regions,
        field_paths = field_paths,
        privilege_field_paths = privilege_field_paths,
        field_privileges = field_privileges,
        field_types = field_types,
        field_ids = field_ids,
        fields_are_scratch = fields_are_scratch,
      }, list_of_regions))
end

function list_of_regions:field_type(field_path)
  local field_type = self.field_types[field_path:hash()]
  assert(field_type)
  return field_type
end

function list_of_regions:field_id(field_path)
  local field_id = self.field_ids[field_path:hash()]
  assert(field_id)
  return field_id
end

function list_of_regions:field_is_scratch(field_path)
  local field_is_scratch = self.fields_are_scratch[field_path:hash()]
  assert(field_is_scratch ~= nil)
  return field_is_scratch
end

function context:has_region_or_list(value_type)
  if std.is_region(value_type) then
    return self:has_region(value_type)
  elseif std.is_list(value_type) and value_type:is_list_of_regions() then
    return self:has_list_of_regions(value_type)
  else
    assert(false)
  end
end

function context:region_or_list(value_type)
  if std.is_region(value_type) then
    return self:region(value_type)
  elseif std.is_list(value_type) and value_type:is_list_of_regions() then
    return self:list_of_regions(value_type)
  else
    assert(false)
  end
end

function context:add_cleanup_item(item)
  assert(self.cleanup_items and item)
  self.cleanup_items:insert(item)
end

function context:get_cleanup_items()
  assert(self.cleanup_items)
  local items = terralib.newlist()
  -- Add cleanup items in reverse order.
  for i = #self.cleanup_items, 1, -1 do
    items:insert(self.cleanup_items[i])
  end
  return quote [items] end
end

local function physical_region_get_base_pointer(cx, index_type, field_type, field_id, privilege, physical_region)
  assert(index_type and field_type and field_id and privilege and physical_region)
  local accessor_args = terralib.newlist({physical_region, field_id})

  local base_pointer = terralib.newsymbol(&field_type, "base_pointer")
  do -- Used to be `index_type:is_opaque()`. Now all cases are structured.
    local dim = data.max(index_type.dim, 1)
    local expected_stride = terralib.sizeof(field_type)

    local dims = data.range(2, dim + 1)
    local strides = terralib.newlist()
    strides:insert(expected_stride)
    for i = 2, dim do
      strides:insert(terralib.newsymbol(c.size_t, "stride" .. tostring(i)))
    end

    local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. tostring(dim) .. "d"]
    local destroy_accessor = c["legion_accessor_array_" .. tostring(dim) .. "d_destroy"]
    local raw_rect_ptr = c["legion_accessor_array_" .. tostring(dim) .. "d_raw_rect_ptr"]

    local rect_t = c["legion_rect_" .. tostring(dim) .. "d_t"]
    local domain_get_bounds = c["legion_domain_get_bounds_" .. tostring(dim) .. "d"]

    local actions = quote
      var accessor = [get_accessor]([accessor_args])

      var region = c.legion_physical_region_get_logical_region([physical_region])
      var domain = c.legion_index_space_get_domain([cx.runtime], region.index_space)
      var rect = [domain_get_bounds](domain)

      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[dim]
      var [base_pointer] = [&field_type]([raw_rect_ptr](
          accessor, rect, &subrect, &(offsets[0])))

      -- Sanity check the outputs.
      std.assert(base_pointer ~= nil or
                 c.legion_domain_get_volume(domain) <= 1,
                 "base pointer is nil")
      [data.range(dim):map(
         function(i)
           return quote
             std.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
             std.assert(subrect.hi.x[i] == rect.hi.x[i], "subrect not equal to rect")
           end
         end)]

      -- WARNING: The compiler assumes SOA, so the first stride should
      -- be equal to the element size. However, the runtime gets
      -- confused on instances with only a single element, and may
      -- return a different value. In those cases, force the stride to
      -- its expected value to avoid problems downstream.
      std.assert(offsets[0].offset == [expected_stride] or
                 c.legion_domain_get_volume(domain) <= 1,
                 "stride does not match expected value")
      offsets[0].offset = [expected_stride]

      -- Fix up the base pointer so it points to the origin (zero),
      -- regardless of where rect is located. This allows us to do
      -- pointer arithmetic later oblivious to what sort of a subrect
      -- we are working with.
      [data.range(dim):map(
         function(i)
           return quote
             [base_pointer] = [&field_type](([&int8]([base_pointer])) - rect.lo.x[i] * offsets[i].offset)
           end
         end)]

      [dims:map(
         function(i)
           return quote var [ strides[i] ] = offsets[i-1].offset end
         end)]
      [destroy_accessor](accessor)
    end
    return actions, base_pointer, strides
  end
end

local function index_space_bounds(cx, is, index_type)
  local bounds = terralib.newsymbol(std.rect_type(index_type), "bounds")
  local actions
  if index_type.dim == 1 then
    actions = quote
      var domain = c.legion_index_space_get_domain([cx.runtime], [is])
      [bounds].lo = domain.rect_data[0]
      [bounds].hi = domain.rect_data[1]
    end
  else
    local domain = terralib.newsymbol(c.legion_domain_t)
    actions = quote
      var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
    end
    local idx = 0
    local fields = terralib.newlist { "lo", "hi" }
    fields:map(function(field)
      index_type.impl_type:getentries():map(function(entry)
        actions = quote
          [actions]
          [bounds].[field].__ptr.[entry.field] = [domain].rect_data[ [idx] ]
        end
        idx = idx + 1
      end)
    end)
  end
  actions = quote
    var [bounds]
    do [actions] end
  end
  return actions, bounds
end

local function make_copy(cx, value, value_type)
  value_type = std.as_read(value_type)
  if std.is_future(value_type) then
    return `([value_type]{ __result = c.legion_future_copy([value].__result) })

  else
    return value
  end
end

local function make_cleanup_item(cx, value, value_type)
  value_type = std.as_read(value_type)
  if std.is_future(value_type) then
    -- Futures are reference counted by the Legion runtime, so
    -- deleting a reference is always ok. (If there are other live
    -- references, they won't be invalidated.)
    return quote
      c.legion_future_destroy([value].__result)
    end

  -- FIXME: Currently, Regent leaks nearly everything. This is mostly
  -- ok in a world where objects are allocated during program
  -- initialization and used for the duration of the program. However,
  -- objects with shorter lifetimes will accumulate and can
  -- potentially cause issues.

  -- WARNING: If you intend to add a cleanup for a new type, proceed
  -- with caution. Some data types are reference counted by the Legion
  -- runtime, others are not. Generally speaking, if a type is
  -- reference counted, it is ok to destroy the value as soon as the
  -- reference is no longer needed (this has no impact on other
  -- references, if any). However, if a type is not reference counted,
  -- you must ensure that there can be no other references to the
  -- object, ever. Practically speaking, this means implementing an
  -- escape analysis to determine the object's lifetime.
  else
    return quote end
  end
end

-- A expr is an object which encapsulates a value and some actions
-- (statements) necessary to produce said value.
local expr = {}
expr.__index = function(t, k) error("expr: no such field " .. tostring(k), 2) end

function expr.just(actions, value)
  if not actions or not value then
    error("expr requires actions and value", 2)
  end
  return setmetatable({ actions = actions, value = value }, expr)
end

function expr.once_only(actions, value, value_type)
  if not actions or not value or not value_type then
    error("expr requires actions and value and value_type", 2)
  end
  local value_name = terralib.newsymbol(std.as_read(value_type))
  actions = quote
    [actions]
    var [value_name] = [value]
  end
  return expr.just(actions, value_name)
end

-- A value encapsulates an rvalue or lvalue. Values are unwrapped by
-- calls to read or write, as appropriate for the lvalue-ness of the
-- object.

local values = {}

local function get_source_location(node)
  assert(node.span.source and node.span.start.line)
  return tostring(node.span.source) .. ":" .. tostring(node.span.start.line)
end

local function unpack_region(cx, region_expr, region_type, static_region_type)
  assert(not cx:has_region(region_type))

  local r = terralib.newsymbol(region_type, "r")
  local lr = terralib.newsymbol(c.legion_logical_region_t, "lr") 
  local is = terralib.newsymbol(c.legion_index_space_t, "is")
  local it = false
  if cache_index_iterator then
    it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
  end
  local actions = quote
    [region_expr.actions]
    var [r] = [std.implicit_cast(
                 static_region_type, region_type, region_expr.value)]
    var [lr] = [r].impl
    var [is] = [lr].index_space
  end

  if cache_index_iterator then
    actions = quote
      [actions]
      var [it] = c.legion_terra_cached_index_iterator_create(
        [cx.runtime], [cx.context], [is])
    end
  end

  local parent_region_type = std.search_constraint_predicate(
    cx, region_type, {},
    function(cx, region)
      return cx:has_region(region)
    end)
  if not parent_region_type then
    error("failed to find appropriate for region " .. tostring(region_type) .. " in unpack", 2)
  end

  local bounds
  if not region_type:is_opaque() then
    local bounds_actions
    bounds_actions, bounds = index_space_bounds(cx, is, region_type:ispace().index_type)
    actions = quote [actions]; [bounds_actions] end
  end

  cx:add_ispace_subispace(region_type:ispace(), is, it,
                          parent_region_type:ispace(), bounds)
  cx:add_region_subregion(region_type, r, parent_region_type)

  return expr.just(actions, r)
end

local value = {}
value.__index = value

function values.value(node, value_expr, value_type, field_path)
  if not ast.is_node(node) then
    error("value requires an AST node", 2)
  end
  if getmetatable(value_expr) ~= expr then
    error("value requires an expression", 2)
  end
  if not terralib.types.istype(value_type) then
    error("value requires a type", 2)
  end

  if field_path == nil then
    field_path = data.newtuple()
  elseif not data.is_tuple(field_path) then
    error("value requires a valid field_path", 2)
  end

  return setmetatable(
    {
      node = node,
      expr = value_expr,
      value_type = value_type,
      field_path = field_path,
    },
    value)
end

function value:new(node, value_expr, value_type, field_path)
  return values.value(node, value_expr, value_type, field_path)
end

function value:read(cx)
  local actions = self.expr.actions
  local result = self.expr.value
  for _, field_name in ipairs(self.field_path) do
    result = `([result].[field_name])
  end
  return expr.just(actions, result)
end

function value:write(cx, value)
  error("attempting to write to rvalue", 2)
end

function value:reduce(cx, value, op)
  error("attempting to reduce to rvalue", 2)
end

function value:__get_field(cx, node, value_type, field_name)
  if value_type:ispointer() then
    return values.rawptr(node, self:read(cx), value_type, data.newtuple(field_name))
  elseif std.is_index_type(value_type) then
    return self:new(node, self.expr, self.value_type, self.field_path .. data.newtuple("__ptr", field_name))
  elseif std.is_bounded_type(value_type) then
    assert(std.get_field(value_type.index_type.base_type, field_name))
    return self:new(node, self.expr, self.value_type, self.field_path .. data.newtuple("__ptr", "__ptr", field_name))
  else
    return self:new(
      node, self.expr, self.value_type, self.field_path .. data.newtuple(field_name))
  end
end

function value:get_field(cx, node, field_name, field_type, value_type)
  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, node, value_type, field_name)
end

function value:get_index(cx, node, index, result_type)
  local value_expr = self:read(cx)
  local actions = terralib.newlist({value_expr.actions, index.actions})
  local value_type = std.as_read(self.value_type)
  if bounds_checks and value_type:isarray() then
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  elseif std.is_list(value_type) then -- Enable list bounds checks all the time.
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_expr.value].__size,
          [get_source_location(node) .. ": list access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  end

  local result
  if std.is_list(value_type) then
    result = expr.just(
      quote [actions] end,
      `([value_type:data(value_expr.value)][ [index.value] ]))
  else
    result = expr.just(
      quote [actions] end,
      `([value_expr.value][ [index.value] ]))
  end
  return values.rawref(node, result, &result_type, data.newtuple())
end

function value:unpack(cx, value_type, field_name, field_type)
  assert(not std.is_bounded_type(value_type) or
           std.get_field(value_type.index_type.base_type, field_name))
  local unpack_type = std.as_read(field_type)

  if std.is_region(unpack_type) and not cx:has_region(unpack_type) then
    local static_region_type = std.get_field(value_type, field_name)
    local region_expr = self:__get_field(cx, self.node, value_type, field_name):read(cx)
    region_expr = unpack_region(cx, region_expr, unpack_type, static_region_type)
    region_expr = expr.just(region_expr.actions, self.expr.value)
    return self:new(self.node, region_expr, self.value_type, self.field_path)
  elseif std.is_bounded_type(unpack_type) then
    assert(unpack_type:is_ptr())
    local region_types = unpack_type:bounds()

    do
      local has_all_regions = true
      for _, region_type in ipairs(region_types) do
        if not (cx:has_region(region_type) or region_type == std.wild_type) then
          has_all_regions = false
          break
        end
      end
      if has_all_regions then
        return self
      end
    end

    -- FIXME: What to do about multi-region pointers?
    assert(#region_types == 1)
    local region_type = region_types[1]

    local static_ptr_type = std.get_field(value_type, field_name)
    local static_region_types = static_ptr_type:bounds()
    assert(#static_region_types == 1)
    local static_region_type = static_region_types[1]

    local region_field_name
    for _, entry in pairs(value_type:getentries()) do
      local entry_type = entry[2] or entry.type
      if entry_type == static_region_type then
        region_field_name = entry[1] or entry.field
      end
    end
    assert(region_field_name)

    local region_expr = self:__get_field(cx, self.node, value_type, region_field_name):read(cx)
    region_expr = unpack_region(cx, region_expr, region_type, static_region_type)
    region_expr = expr.just(region_expr.actions, self.expr.value)
    return self:new(self.node, region_expr, self.value_type, self.field_path)
  else
    return self
  end
end

local ref = setmetatable({}, { __index = value })
ref.__index = ref

function values.ref(node, value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or
    not (std.is_bounded_type(value_type) or std.is_vptr(value_type)) then
    error("ref requires a legion ptr type", 2)
  end
  return setmetatable(values.value(node, value_expr, value_type, field_path), ref)
end

function ref:new(node, value_expr, value_type, field_path)
  return values.ref(node, value_expr, value_type, field_path)
end

local function get_element_pointer(cx, node, region_types, index_type, field_type,
                                   base_pointer, strides, index)
  if bounds_checks then
    local terra check(runtime : c.legion_runtime_t,
                      ctx : c.legion_context_t,
                      pointer : index_type,
                      pointer_index : uint32,
                      region : c.legion_logical_region_t,
                      region_index : uint32)
      if region_index == pointer_index then
        var check = c.legion_domain_point_safe_cast(runtime, ctx, pointer:to_domain_point(), region)
        if c.legion_domain_point_is_null(check) then
          std.assert_error(false, [get_source_location(node) .. ": pointer " .. tostring(index_type) .. " is out-of-bounds"])
        end
      end
      return pointer
    end

    local pointer_value = index
    local pointer_index = 1
    if #region_types > 1 then
      pointer_index = `([index].__index)
    end

    for region_index, region_type in ipairs(region_types) do
      assert(cx:has_region(region_type))
      local lr = cx:region(region_type).logical_region
      index = `check(
        [cx.runtime], [cx.context],
        [pointer_value], [pointer_index],
        [lr].impl, [region_index])
    end
  end

  -- Note: This code is performance-critical and tends to be sensitive
  -- to small changes. Please thoroughly performance-test any changes!
  if std.is_bounded_type(index_type) then
    if not index_type.fields then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      return `(@[&field_type](&base_pointer[ [index].__ptr.__ptr ]))
    elseif #index_type.fields == 1 then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      local field = index_type.fields[1]
      return `(@[&field_type](&base_pointer[ [index].__ptr.__ptr.[field] ]))
    else
      local offset
      for i, field in ipairs(index_type.fields) do
        if offset then
          offset = `(offset + [index].__ptr.__ptr.[ field ] * [ strides[i] ])
        else
          offset = `([index].__ptr.__ptr.[ field ] * [ strides[i] ])
        end
      end
      return `(@([&field_type]([&int8](base_pointer) + offset)))
    end
  elseif std.is_index_type(index_type) then
    if not index_type.fields then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      return `(@[&field_type](&base_pointer[ [index].__ptr ]))
    elseif #index_type.fields == 1 then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      local field = index_type.fields[1]
      return `(@[&field_type](&base_pointer[ [index].__ptr.[field] ]))
    else
      local offset
      for i, field in ipairs(index_type.fields) do
        if offset then
          offset = `(offset + [index].__ptr.[ field ] * [ strides[i] ])
        else
          offset = `([index].__ptr.[ field ] * [ strides[i] ])
        end
      end
      return `(@([&field_type]([&int8](base_pointer) + offset)))
    end
  else
    assert(false)
  end
end

function ref:__ref(cx, expr_type)
  local actions = self.expr.actions
  local value = self.expr.value

  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))
  local field_paths, field_types = std.flatten_struct_fields(value_type)
  local absolute_field_paths = field_paths:map(
    function(field_path) return self.field_path .. field_path end)

  local region_types = self.value_type:bounds()
  local base_pointers_by_region = region_types:map(
    function(region_type)
      return absolute_field_paths:map(
        function(field_path)
          return cx:region(region_type):base_pointer(field_path)
        end)
    end)
  local strides_by_region = region_types:map(
    function(region_type)
      return absolute_field_paths:map(
        function(field_path)
          return cx:region(region_type):stride(field_path)
        end)
    end)

  local base_pointers, strides

  if cx.check_divergence(region_types, field_paths) or #region_types == 1 then
    base_pointers = base_pointers_by_region[1]
    strides = strides_by_region[1]
  else
    base_pointers = data.zip(absolute_field_paths, field_types):map(
      function(field)
        local field_path, field_type = unpack(field)
        return terralib.newsymbol(&field_type, "base_pointer_" .. field_path:hash())
      end)
    strides = absolute_field_paths:map(
      function(field_path)
        return cx:region(region_types[1]):stride(field_path):map(
          function(_)
            return terralib.newsymbol(c.size_t, "stride_" .. field_path:hash())
          end)
      end)

    local cases
    for i = #region_types, 1, -1 do
      local region_base_pointers = base_pointers_by_region[i]
      local region_strides = strides_by_region[i]
      local case = data.zip(base_pointers, region_base_pointers, strides, region_strides):map(
        function(pair)
          local base_pointer, region_base_pointer, field_strides, field_region_strides = unpack(pair)
          local setup = quote [base_pointer] = [region_base_pointer] end
          for i, stride in ipairs(field_strides) do
            local region_stride = field_region_strides[i]
            setup = quote [setup]; [stride] = [region_stride] end
          end
          return setup
        end)

      if cases then
        cases = quote
          if [value].__index == [i] then
            [case]
          else
            [cases]
          end
        end
      else
        cases = case
      end
    end

    actions = quote
      [actions];
      [base_pointers:map(
         function(base_pointer) return quote var [base_pointer] end end)];
      [strides:map(
         function(stride) return quote [stride:map(function(s) return quote var [s] end end)] end end)];
      [cases]
    end
  end

  local values
  if not expr_type or std.type_maybe_eq(std.as_read(expr_type), value_type) then
    values = data.zip(field_types, base_pointers, strides):map(
      function(field)
        local field_type, base_pointer, stride = unpack(field)
        return get_element_pointer(cx, self.node, region_types, self.value_type, field_type, base_pointer, stride, value)
      end)
  else
    assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
    values = data.zip(field_types, base_pointers, strides):map(
      function(field)
        local field_type, base_pointer, stride = unpack(field)
        local vec = vector(field_type, std.as_read(expr_type).N)
        return `(@[&vec](&[get_element_pointer(cx, self.node, region_types, self.value_type, field_type, base_pointer, stride, value)]))
      end)
    value_type = expr_type
  end

  return actions, values, value_type, field_paths, field_types
end

function ref:read(cx, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  local actions, values, value_type, field_paths, field_types = self:__ref(cx, expr_type)
  local value = terralib.newsymbol(value_type)
  actions = quote
    [actions];
    var [value]
    [data.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if expr_type and
            (expr_type:isvector() or
             std.is_vptr(expr_type) or
             std.is_sov(expr_type)) then
           if field_type:isvector() then field_type = field_type.type end
           local align = sizeof(field_type)
           if aligned_instances then
             align = sizeof(vector(field_type, expr_type.N))
           end
           return quote
             [result] = terralib.attrload(&[field_value], {align = [align]})
           end
         else
           return quote [result] = [field_value] end
         end
      end)]
  end
  return expr.just(actions, value)
end

function ref:write(cx, value, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  local value_expr = value:read(cx, expr_type)
  local actions, values, value_type, field_paths, field_types = self:__ref(cx, expr_type)
  actions = quote
    [value_expr.actions];
    [actions];
    [data.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if expr_type and
            (expr_type:isvector() or
             std.is_vptr(expr_type) or
             std.is_sov(expr_type)) then
           if field_type:isvector() then field_type = field_type.type end
           local align = sizeof(field_type)
           if aligned_instances then
             align = sizeof(vector(field_type, expr_type.N))
           end
           return quote
             terralib.attrstore(&[field_value], [result], {align = [align]})
           end
         else
          return quote [field_value] = [result] end
        end
      end)]
  end
  return expr.just(actions, quote end)
end

local reduction_fold = {
  ["+"] = "+",
  ["-"] = "-",
  ["*"] = "*",
  ["/"] = "*", -- FIXME: Need to fold with "/" for RW instances.
  ["max"] = "max",
  ["min"] = "min",
}

function ref:reduce(cx, value, op, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  local fold_op = reduction_fold[op]
  assert(fold_op)
  local value_expr = value:read(cx, expr_type)
  local actions, values, value_type, field_paths, field_types = self:__ref(cx, expr_type)
  local function quote_vector_binary_op(fold_op, sym, result, expr_type)
    if fold_op == "min" or fold_op == "max" then
      return `([std["v" .. fold_op](expr_type)](sym, result))
    else
      return std.quote_binary_op(fold_op, sym, result)
    end
  end
  actions = quote
    [value_expr.actions];
    [actions];
    [data.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if expr_type and
            (expr_type:isvector() or
             std.is_vptr(expr_type) or
             std.is_sov(expr_type)) then
           if field_type:isvector() then field_type = field_type.type end
           local align = sizeof(field_type)
           if aligned_instances then
             align = sizeof(vector(field_type, expr_type.N))
           end

           local field_value_load = quote
              terralib.attrload(&[field_value], {align = [align]})
           end
           local sym = terralib.newsymbol(expr_type)
           return quote
             var [sym] =
               terralib.attrload(&[field_value], {align = [align]})
             terralib.attrstore(&[field_value],
               [quote_vector_binary_op(fold_op, sym, result, expr_type)],
               {align = [align]})
           end
         else
           return quote
             [field_value] = [std.quote_binary_op(
                                fold_op, field_value, result)]
           end
         end
      end)]
  end
  return expr.just(actions, quote end)
end

function ref:get_field(cx, node, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, node, value_type, field_name)
end

function ref:get_index(cx, node, index, result_type)
  local value_actions, value = self:__ref(cx)
  -- Arrays are never field-sliced, therefore, an array array access
  -- must be to a single field.
  assert(#value == 1)
  value = value[1]

  local actions = terralib.newlist({value_actions, index.actions})
  local value_type = self.value_type.points_to_type
  if bounds_checks and value_type:isarray() then
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  end
  assert(not std.is_list(value_type)) -- Shouldn't be an l-value anyway.
  local result = expr.just(quote [actions] end, `([value][ [index.value] ]))
  return values.rawref(node, result, &result_type, data.newtuple())
end

local vref = setmetatable({}, { __index = value })
vref.__index = vref

function values.vref(node, value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or not std.is_vptr(value_type) then
    error("vref requires a legion vptr type", 2)
  end
  return setmetatable(values.value(node, value_expr, value_type, field_path), vref)
end

function vref:new(node, value_expr, value_type, field_path)
  return values.vref(node, value_expr, value_type, field_path)
end

function vref:__unpack(cx)
  assert(std.is_vptr(self.value_type))

  local actions = self.expr.actions
  local value = self.expr.value

  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))
  local field_paths, field_types = std.flatten_struct_fields(value_type)
  local absolute_field_paths = field_paths:map(
    function(field_path) return self.field_path .. field_path end)

  local region_types = self.value_type:bounds()
  local base_pointers_by_region = region_types:map(
    function(region_type)
      return absolute_field_paths:map(
        function(field_path)
          return cx:region(region_type):base_pointer(field_path)
        end)
    end)

  return field_paths, field_types, region_types, base_pointers_by_region
end

function vref:read(cx, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
  local actions = self.expr.actions
  local field_paths, field_types, region_types, base_pointers_by_region = self:__unpack(cx)
  -- where the result should go
  local value = terralib.newsymbol(expr_type)
  local vref_value = self.expr.value
  local vector_width = self.value_type.N

  -- make symols to store scalar values from different pointers
  local vars = terralib.newlist()
  for i = 1, vector_width do
    local v = terralib.newsymbol(expr_type.type)
    vars:insert(v)
    actions = quote
      [actions];
      var [v]
    end
  end

  -- if the vptr points to a single region
  if cx.check_divergence(region_types, field_paths) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    data.zip(base_pointers, field_paths):map(
      function(pair)
        local base_pointer, field_path = unpack(pair)
        for i = 1, vector_width do
          local v = vars[i]
          for _, field_name in ipairs(field_path) do
            v = `([v].[field_name])
          end
          actions = quote
            [actions];
            [v] = base_pointer[ [vref_value].__ptr.value[ [i - 1] ] ]
          end
        end
      end)
  -- if the vptr can point to multiple regions
  else
    for field_idx, field_path in pairs(field_paths) do
      for vector_idx = 1, vector_width do
        local v = vars[vector_idx]
        for _, field_name in ipairs(field_path) do
          v = `([v].[field_name])
        end
        local cases
        for region_idx = #base_pointers_by_region, 1, -1 do
          local base_pointer = base_pointers_by_region[region_idx][field_idx]
          local case = quote
              v = base_pointer[ [vref_value].__ptr.value[ [vector_idx - 1] ] ]
          end

          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case]
              else
                [cases]
              end
            end
          else
            cases = case
          end
        end
        actions = quote [actions]; [cases] end
      end
    end
  end

  actions = quote
    [actions];
    var [value]
    [field_paths:map(
       function(field_path)
         local result = value
         local field_accesses = vars:map(
          function(v)
            for _, field_name in ipairs(field_path) do
              v = `([v].[field_name])
            end
            return v
          end)
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         return quote [result] = vector( [field_accesses] ) end
       end)]
  end

  return expr.just(actions, value)
end

function vref:write(cx, value, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
  local actions = self.expr.actions
  local value_expr = value:read(cx, expr_type)
  local field_paths, field_types, region_types, base_pointers_by_region = self:__unpack(cx)

  local vref_value = self.expr.value
  local vector_width = self.value_type.N

  actions = quote
    [value_expr.actions];
    [actions]
  end

  if cx.check_divergence(region_types, field_paths) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    data.zip(base_pointers, field_paths):map(
      function(pair)
        local base_pointer, field_path = unpack(pair)
        for i = 1, vector_width do
          local result = value_expr.value
          local field_value = `base_pointer[ [vref_value].__ptr.value[ [i - 1] ] ]
          for _, field_name in ipairs(field_path) do
            result = `([result].[field_name])
          end
          local assignment
          if value.value_type:isprimitive() or
             (std.is_sov(expr_type) and
              std.type_eq(expr_type.type, value.value_type)) then
            assignment = quote
              [field_value] = [result]
            end
          else
            assignment = quote
              [field_value] = [result][ [i - 1] ]
            end
          end
          actions = quote
            [actions];
            [assignment]
          end
        end
      end)
  else
    for field_idx, field_path in pairs(field_paths) do
      for vector_idx = 1, vector_width do
        local result = value_expr.value
        for _, field_name in ipairs(field_path) do
          result = `([result].[field_name])
        end
        if value.value_type:isvector() then
          result = `(result[ [vector_idx - 1] ])
        end
        local cases
        for region_idx = #base_pointers_by_region, 1, -1 do
          local base_pointer = base_pointers_by_region[region_idx][field_idx]
          local case = quote
            base_pointer[ [vref_value].__ptr.value[ [vector_idx - 1] ] ] =
              result
          end

          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case]
              else
                [cases]
              end
            end
          else
            cases = case
          end
        end
        actions = quote [actions]; [cases] end
      end
    end

  end
  return expr.just(actions, quote end)
end

function vref:reduce(cx, value, op, expr_type)
  if expr_type and (std.is_ref(expr_type) or std.is_rawref(expr_type)) then
    expr_type = std.as_read(expr_type)
  end
  assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
  local actions = self.expr.actions
  local fold_op = reduction_fold[op]
  assert(fold_op)
  local value_expr = value:read(cx, expr_type)
  local field_paths, field_types, region_types, base_pointers_by_region = self:__unpack(cx)

  local vref_value = self.expr.value
  local vector_width = self.value_type.N

  actions = quote
    [value_expr.actions];
    [actions]
  end

  if cx.check_divergence(region_types, field_paths) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    data.zip(base_pointers, field_paths):map(
      function(pair)
        local base_pointer, field_path = unpack(pair)
        local result = value_expr.value
        for i = 1, vector_width do
          local field_value = `base_pointer[ [vref_value].__ptr.value[ [i - 1] ] ]
          for _, field_name in ipairs(field_path) do
            result = `([result].[field_name])
          end
          if value.value_type:isprimitive() then
            actions = quote
              [actions];
              [field_value] =
                [std.quote_binary_op(fold_op, field_value, result)]
            end
          else
            local v = terralib.newsymbol(expr_type.type)
            local assignment = quote
              var [v] = [result][ [i - 1] ]
            end
            actions = quote
              [actions];
              [assignment];
              [field_value] =
                [std.quote_binary_op(fold_op, field_value, v)]
            end
          end
        end
      end)
  else
    for field_idx, field_path in pairs(field_paths) do
      for vector_idx = 1, vector_width do
        local result = value_expr.value
        for _, field_name in ipairs(field_path) do
          result = `([result].[field_name])
        end
        if value.value_type:isvector() then
          result = `result[ [vector_idx - 1] ]
        end
        local cases
        for region_idx = #base_pointers_by_region, 1, -1 do
          local base_pointer = base_pointers_by_region[region_idx][field_idx]
          local field_value = `base_pointer[ [vref_value].__ptr.value[ [vector_idx - 1] ] ]
          local case = quote
            [field_value] =
              [std.quote_binary_op(fold_op, field_value, result)]
          end
          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case]
              else
                [cases]
              end
            end
          else
            cases = case
          end
        end
        actions = quote [actions]; [cases] end
      end
    end

  end

  return expr.just(actions, quote end)
end

function vref:get_field(cx, node, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, node, value_type, field_name)
end

local rawref = setmetatable({}, { __index = value })
rawref.__index = rawref

-- For pointer-typed rvalues, this entry point coverts the pointer
-- to an lvalue by dereferencing the pointer.
function values.rawptr(node, value_expr, value_type, field_path)
  if getmetatable(value_expr) ~= expr then
    error("rawref requires an expression", 2)
  end

  value_expr = expr.just(value_expr.actions, `(@[value_expr.value]))
  return values.rawref(node, value_expr, value_type, field_path)
end

-- This entry point is for lvalues which are already references
-- (e.g. for mutable variables on the stack). Conceptually
-- equivalent to a pointer rvalue which has been dereferenced. Note
-- that value_type is still the pointer type, not the reference
-- type.
function values.rawref(node, value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or not value_type:ispointer() then
    error("rawref requires a pointer type, got " .. tostring(value_type), 2)
  end
  return setmetatable(values.value(node, value_expr, value_type, field_path), rawref)
end

function rawref:new(node, value_expr, value_type, field_path)
  return values.rawref(node, value_expr, value_type, field_path)
end

function rawref:__ref(cx)
  local actions = self.expr.actions
  local result = self.expr.value
  for _, field_name in ipairs(self.field_path) do
    result = `([result].[field_name])
  end
  return expr.just(actions, result)
end

function rawref:read(cx)
  return self:__ref(cx)
end

function rawref:write(cx, value)
  local value_expr = value:read(cx)
  local ref_expr = self:__ref(cx)
  local cleanup = make_cleanup_item(cx, ref_expr.value, self.value_type.type)
  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
    [cleanup];
    [ref_expr.value] = [value_expr.value]
  end
  return expr.just(actions, quote end)
end

function rawref:reduce(cx, value, op)
  local ref_expr = self:__ref(cx)
  local cleanup = make_cleanup_item(cx, ref_expr.value, self.value_type.type)
  local value_expr = value:read(cx)

  local ref_type = std.get_field_path(self.value_type.type, self.field_path)
  local value_type = std.as_read(value.value_type)

  local reduce = ast.typed.expr.Binary {
    op = op,
    lhs = ast.typed.expr.Internal {
      value = values.value(self.node, expr.just(quote end, ref_expr.value), ref_type),
      expr_type = ref_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    },
    rhs = ast.typed.expr.Internal {
      value = values.value(value.node, expr.just(quote end, value_expr.value), value_type),
      expr_type = value_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    },
    expr_type = ref_type,
    annotations = ast.default_annotations(),
    span = ast.trivial_span(),
  }

  local reduce_expr = codegen.expr(cx, reduce):read(cx, ref_type)

  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
    [reduce_expr.actions];
    [cleanup];
    [ref_expr.value] = [reduce_expr.value]
  end
  return expr.just(actions, quote end)
end

function rawref:get_field(cx, node, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, node, value_type, field_name)
end

function rawref:get_index(cx, node, index, result_type)
  local ref_expr = self:__ref(cx)
  local actions = terralib.newlist({ref_expr.actions, index.actions})
  local value_type = self.value_type.type
  if bounds_checks and value_type:isarray() then
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  elseif std.is_list(value_type) then -- Enable list bounds checks all the time.
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [ref_expr.value].__size,
          [get_source_location(node) .. ": list access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  end

  local result
  if std.is_list(value_type) then
    result = expr.just(
      quote [actions] end,
      `([value_type:data(ref_expr.value)][ [index.value] ]))
  else
    result = expr.just(
      quote [actions] end,
      `([ref_expr.value][ [index.value] ]))
  end
  return values.rawref(node, result, &result_type, data.newtuple())
end

-- A helper for capturing debug information.
local function emit_debuginfo(node)
  assert(node.span.source and node.span.start.line)
  if string.len(node.span.source) == 0 then
    return quote end
  end
  return quote
    terralib.debuginfo(node.span.source, node.span.start.line)
  end
end

function codegen.expr_internal(cx, node)
  return node.value
end

function codegen.expr_region_root(cx, node)
  return codegen.expr(cx, node.region)
end

function codegen.expr_condition(cx, node)
  return codegen.expr(cx, node.value):read(
    cx, std.as_read(node.value.expr_type))
end

function codegen.expr_id(cx, node)
  assert(std.is_symbol(node.value))
  local value = node.value:getsymbol()
  if std.is_rawref(node.expr_type) then
    return values.rawref(
      node,
      expr.just(emit_debuginfo(node), value),
      node.expr_type.pointer_type)
  else
    return values.value(
      node,
      expr.just(emit_debuginfo(node), value),
      node.expr_type)
  end
end

function codegen.expr_constant(cx, node)
  local value = node.value
  local value_type = std.as_read(node.expr_type)

  if terralib.isconstant(value) then
    assert(std.type_eq(value.type, value_type))
  else
    value = terralib.constant(value_type, value)
  end

  return values.value(
    node,
    expr.just(emit_debuginfo(node), value),
    value_type)
end

function codegen.expr_function(cx, node)
  local value_type = std.as_read(node.expr_type)
  local value = node.value
  if cx.variant:is_cuda() then
    value = cudahelper.replace_with_builtin(value)
  end
  return values.value(
    node,
    expr.just(emit_debuginfo(node), value),
    value_type)
end

function codegen.expr_field_access(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local field_name = node.field_name
  local field_type = node.expr_type

  if std.is_region(value_type) and field_name == "ispace" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)

    local actions = quote
      [value.actions];
      [emit_debuginfo(node)]
    end

    return values.value(
      node,
      expr.once_only(
        actions,
        `([expr_type] { impl = [value.value].impl.index_space }),
        expr_type),
      expr_type)
  elseif (std.is_ispace(value_type) or std.is_region(value_type)) and
         field_name == "bounds" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)
    assert(expr_type.is_rect_type)
    local bounds
    if std.is_ispace(value_type) then
      bounds = cx:ispace(value_type).bounds
    else
      bounds = cx:ispace(value_type:ispace()).bounds
    end

    local actions = quote
      [value.actions]
      [emit_debuginfo(node)]
    end
    return values.value(
      node,
      expr.once_only(
        actions,
        bounds,
        expr_type),
      expr_type)
  elseif std.is_ispace(value_type) and field_name == "volume" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)

    local volume = terralib.newsymbol(expr_type, "volume")
    local actions = quote
      [value.actions]
      [emit_debuginfo(node)]

      -- Currently doesn't support index spaces with multiple domains.
      regentlib.assert(not c.legion_index_space_has_multiple_domains([cx.runtime], [value.value].impl),
        "\"volume\" field isn't supported on index spaces with multiple domains")
      var [volume] = c.legion_domain_get_volume(
        c.legion_index_space_get_domain([cx.runtime], [value.value].impl))
    end
    return values.value(
      node,
      expr.just(actions, volume),
      expr_type)
  elseif std.is_partition(value_type) and field_name == "colors" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)
    local is = terralib.newsymbol(c.legion_index_space_t, "colors")
    local actions = quote
      [value.actions]
      var color_space =
        c.legion_index_partition_get_color_space([cx.runtime],
                                                 [value.value].impl.index_partition)
      var domain = 
        c.legion_index_space_get_domain([cx.runtime], color_space)
      var [is] = c.legion_index_space_create_domain([cx.runtime], [cx.context],
                                                    domain)
    end
    return values.value(
      node,
      expr.once_only(
        actions,
        `([expr_type]({ impl = [is] })),
        expr_type),
      expr_type)
  else
    return codegen.expr(cx, node.value):get_field(cx, node, field_name, field_type, node.value.expr_type)
  end
end

function codegen.expr_index_access(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local index_type = std.as_read(node.index.expr_type)
  local expr_type = std.as_read(node.expr_type)

  if std.is_partition(value_type) then
    local value = codegen.expr(cx, node.value):read(cx)
    local index = codegen.expr(cx, node.index):read(cx)

    local actions = quote
      [value.actions];
      [index.actions];
      [emit_debuginfo(node)]
    end

    if cx:has_region(expr_type) then
      local lr = cx:region(expr_type).logical_region
      return values.value(node, expr.just(actions, lr), expr_type)
    end

    local parent_region_type = value_type:parent_region()

    local r = terralib.newsymbol(expr_type, "r")
    local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
    local is = terralib.newsymbol(c.legion_index_space_t, "is")
    local it = false
    if cache_index_iterator then
      it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
    end

    local color_type = value_type:colors().index_type
    local color = std.implicit_cast(index_type, color_type, index.value)

    actions = quote
      [actions]
      var dp = [color]:to_domain_point()
      var [lr] = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
        [cx.runtime], [value.value].impl, dp)
      var [is] = [lr].index_space
      var [r] = [expr_type] { impl = [lr] }
    end

    if cache_index_iterator then
      actions = quote
        [actions]
        var [it] = c.legion_terra_cached_index_iterator_create(
          [cx.runtime], [cx.context], [is])
      end
    end

    local bounds
    if not expr_type:is_opaque() then
      local bounds_actions
      bounds_actions, bounds = index_space_bounds(cx, is, expr_type:ispace().index_type)
      actions = quote [actions]; [bounds_actions] end
    end

    cx:add_ispace_subispace(expr_type:ispace(), is, it,
                            parent_region_type:ispace(), bounds)
    cx:add_region_subregion(expr_type, r, parent_region_type)

    return values.value(node, expr.just(actions, r), expr_type)
  elseif std.is_cross_product(value_type) then
    local value = codegen.expr(cx, node.value):read(cx)
    local index = codegen.expr(cx, node.index):read(cx)

    local actions = quote
      [value.actions];
      [index.actions];
      [emit_debuginfo(node)]
    end

    local color_type = value_type:partition():colors().index_type
    local color = std.implicit_cast(index_type, color_type, index.value)

    local region_type = expr_type:parent_region()
    local lr
    if not cx:has_region(region_type) then
      local parent_region_type = value_type:parent_region()

      local r = terralib.newsymbol(region_type, "r")
      lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
      local is = terralib.newsymbol(c.legion_index_space_t, "is")
      local it = false
      if cache_index_iterator then
        it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
      end
      actions = quote
        [actions]
        var dp = [color]:to_domain_point()
        var [lr] = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
          [cx.runtime], [value.value].impl, dp)
        var [is] = [lr].index_space
        var [r] = [region_type] { impl = [lr] }
      end

      if cache_index_iterator then
        actions = quote
          [actions]
          var [it] = c.legion_terra_cached_index_iterator_create(
            [cx.runtime], [cx.context], [is])
        end
      end

      local bounds
      if not region_type:is_opaque() then
        local bounds_actions
        bounds_actions, bounds = index_space_bounds(cx, is, region_type:ispace().index_type)
        actions = quote [actions]; [bounds_actions] end
      end

      cx:add_ispace_subispace(region_type:ispace(), is, it,
                              parent_region_type:ispace(), bounds)
      cx:add_region_subregion(region_type, r, parent_region_type)
    else
      lr = `([cx:region(region_type).logical_region]).impl
    end

    local result = terralib.newsymbol(expr_type, "subpartition")
    local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
    local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
    actions = quote
      [actions]
      var dp = [color]:to_domain_point()
      var [ip] = c.legion_terra_index_cross_product_get_subpartition_by_color_domain_point(
        [cx.runtime], [cx.context],
        [value.value].product, dp)
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [cx.context], [lr], [ip])
    end

    if std.is_partition(expr_type) then
      actions = quote
        [actions]
        var [result] = [expr_type] { impl = [lp] }
      end
    elseif std.is_cross_product(expr_type) then
      actions = quote
        [actions]
        var [result] = [expr_type] {
          impl = [lp],
          product = c.legion_terra_index_cross_product_t {
            partition = [ip],
            other_color = [value.value].colors[2],
          },
          -- FIXME: colors
        }
      end
    end
    return values.value(node, expr.just(actions, result), expr_type)
  elseif std.is_list(value_type) then
    local index = codegen.expr(cx, node.index):read(cx)
    if not std.is_list(index_type) then
      -- Single indexing
      index = expr.just(index.actions, std.implicit_cast(index_type, int, index.value))
      local value = codegen.expr(cx, node.value):get_index(cx, node, index, expr_type)
      if not value_type:is_list_of_regions() then
        return value
      else
        local region = value:read(cx)
        local region_type = std.as_read(node.expr_type)

        if cx:has_region_or_list(region_type) then
          return values.value(node, region, region_type)
        end

        if std.is_region(region_type) then
          -- FIXME: For the moment, iterators, allocators, and physical
          -- regions are inaccessible since we assume lists are always
          -- unmapped.
          local bounds
          if not region_type:is_opaque() then
            local bounds_actions
            bounds_actions, bounds =
              index_space_bounds(cx, `([region.value].impl.index_space),
                                 region_type:ispace().index_type)
            region.actions = quote [region.actions]; [bounds_actions] end
          end
          cx:add_ispace_root(
            region_type:ispace(),
            `([region.value].impl.index_space),
            false,
            bounds)
          cx:add_region_root(
            region_type, region.value,
            cx:list_of_regions(value_type).field_paths,
            cx:list_of_regions(value_type).privilege_field_paths,
            cx:list_of_regions(value_type).field_privileges,
            cx:list_of_regions(value_type).field_types,
            cx:list_of_regions(value_type).field_ids,
            cx:list_of_regions(value_type).fields_are_scratch,
            false,
            false,
            false)
        elseif std.is_list_of_regions(region_type) then
          cx:add_list_of_regions(
            region_type, region.value,
            cx:list_of_regions(value_type).field_paths,
            cx:list_of_regions(value_type).privilege_field_paths,
            cx:list_of_regions(value_type).field_privileges,
            cx:list_of_regions(value_type).field_types,
            cx:list_of_regions(value_type).field_ids,
            cx:list_of_regions(value_type).fields_are_scratch)
        else
          assert(false)
        end
        return values.value(node, region, region_type)
      end
    else
      -- List indexing
      local value = codegen.expr(cx, node.value):read(cx)

      local list_type = node.expr_type
      local list = terralib.newsymbol(list_type, "list")
      local actions = quote
        [value.actions]
        [index.actions]
        [emit_debuginfo(node)]

        var data = c.malloc(
          terralib.sizeof([list_type.element_type]) * [index.value].__size)
        regentlib.assert(data ~= nil, "malloc failed in index_access")
        var [list] = [list_type] {
          __size = [index.value].__size,
          __data = data
        }
        for i = 0, [index.value].__size do
          var idx = [index_type:data(index.value)][i]
          std.assert(idx < [value.value].__size, "slice index out of bounds")
          [list_type:data(list)][i] = [std.implicit_cast(
            value_type.element_type,
            list_type.element_type,
            `([value_type:data(value.value)][idx]))]
        end
      end

      if value_type:is_list_of_regions() then
        cx:add_list_of_regions(
          list_type, list,
          cx:list_of_regions(value_type).field_paths,
          cx:list_of_regions(value_type).privilege_field_paths,
          cx:list_of_regions(value_type).field_privileges,
          cx:list_of_regions(value_type).field_types,
          cx:list_of_regions(value_type).field_ids,
          cx:list_of_regions(value_type).fields_are_scratch)
      end
      return values.value(node, expr.just(actions, list), list_type)
    end
  elseif std.is_region(value_type) then
    local index = codegen.expr(cx, node.index):read(cx, index_type)

    local pointer_type = node.expr_type.pointer_type
    local pointer = index
    if not std.type_eq(index_type, pointer_type) then
      if std.is_vptr(index_type) then
        return values.vref(node, pointer, index_type)
      end
      local point = std.implicit_cast(index_type, pointer_type.index_type, index.value)
      pointer = expr.just(
        index.actions,
        `([pointer_type] { __ptr = [pointer_type.index_type] { __ptr = [point].__ptr }}))
    end
    return values.ref(node, pointer, pointer_type)
  else
    local index = codegen.expr(cx, node.index):read(cx)
    return codegen.expr(cx, node.value):get_index(cx, node, index, expr_type)
  end
end

function codegen.expr_method_call(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local args = node.args:map(
    function(arg) return codegen.expr(cx, arg):read(cx) end)

  local actions = quote
    [value.actions];
    [args:map(function(arg) return arg.actions end)];
    [emit_debuginfo(node)]
  end
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    node,
    expr.once_only(
      actions,
      `([value.value]:[node.method_name](
          [args:map(function(arg) return arg.value end)])),
      expr_type),
    expr_type)
end

local function expr_call_setup_task_args(
    cx, task, args, arg_types, param_types, params_struct_type, params_map_label, params_map_type,
    task_args, task_args_setup, task_args_cleanup)
  local size = terralib.newsymbol(c.size_t, "size")
  local buffer = terralib.newsymbol(&opaque, "buffer")

  task_args_setup:insert(quote
    var [size] = terralib.sizeof(params_struct_type)
  end)

  for i = 1, #args do
    local arg = args[i]
    local arg_type = arg_types[i]
    if not std.is_future(arg_type) then
      local param_type = param_types[i] -- arg has already been cast to param_type
      local size_actions, size_value = std.compute_serialized_size(
        param_type, arg)
      task_args_setup:insert(size_actions)
      task_args_setup:insert(quote [size] = [size] + [size_value] end)
   end
  end

  task_args_setup:insert(quote
    var [buffer] = c.malloc([size])
    std.assert([buffer] ~= nil, "malloc failed in setup task args")
    [task_args].args = [buffer]
    [task_args].arglen = [size]
  end)

  local fixed_ptr = terralib.newsymbol(&params_struct_type, "fixed_ptr")
  local data_ptr = terralib.newsymbol(&uint8, "data_ptr")
  task_args_setup:insert(quote
    var [fixed_ptr] = [&params_struct_type](buffer)
    var [data_ptr] = [&uint8](buffer) + terralib.sizeof(params_struct_type)
  end)

  if params_map_label then
    task_args_setup:insert(quote
      var params_map_value : params_map_type = arrayof(
        uint64,
        [data.range(params_map_type.N):map(function() return 0 end)])
      [data.mapi(
         function(i, arg_type)
           if std.is_future(arg_type) then
             return quote
               params_map_value[ [math.floor((i-1)/64)] ] =
                 params_map_value[ [math.floor((i-1)/64)] ] + [2ULL ^ math.fmod(i-1, 64)]
             end
           end
           return quote end
         end,
         arg_types)]
      [fixed_ptr].[params_map_label] = params_map_value
    end)
  end

  -- Prepare the by-value arguments to the task.
  for i, arg in ipairs(args) do
    local arg_type = arg_types[i]
    if not std.is_future(arg_type) then
      local c_field = params_struct_type:getentries()[i + 1]
      local c_field_name = c_field[1] or c_field.field
      if terralib.issymbol(c_field_name) then
        c_field_name = c_field_name.displayname
      end

      local param_type = param_types[i]
      task_args_setup:insert(
        std.serialize(
          param_type, std.implicit_cast(arg_type, param_type, arg),
          `(&[fixed_ptr].[c_field_name]), `(&[data_ptr])))
    end
  end

  -- Prepare the region arguments to the task.
  -- (Region values have already been copied into task arguments verbatim.)

  -- Pass field IDs by-value to the task.
  local param_field_ids = task:get_field_id_param_labels()
  for _, i in ipairs(std.fn_params_with_privileges_by_index(task:get_type())) do
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    local field_paths, _ = std.flatten_struct_fields(param_type:fspace())
    for j, field_path in pairs(field_paths) do
      local arg_field_id = cx:region_or_list(arg_type):field_id(field_path)
      local param_field_id = param_field_ids[i][j]
      task_args_setup:insert(
        quote [fixed_ptr].[param_field_id] = [arg_field_id] end)
    end
  end

  -- Check that the final sizes line up.
  task_args_setup:insert(quote
    std.assert([data_ptr] - [&uint8]([buffer]) == [size],
      "mismatch in data serialized in setup task args")
  end)

  -- Add cleanup code for buffer.
  task_args_cleanup:insert(quote
    c.free([buffer])
  end)
end

local function expr_call_setup_future_arg(
    cx, task, arg, launcher, index, args_setup)
  local add_future = c.legion_task_launcher_add_future
  if index then
    add_future = c.legion_index_launcher_add_future
  end

  args_setup:insert(quote
    add_future(launcher, [arg].__result)
  end)
end

local function add_phase_barrier_arg_recurse(
  add_barrier, launcher, arg, arg_type)
  if std.is_phase_barrier(arg_type) then
    return quote
      add_barrier(launcher, [arg].impl)
    end
  else
    local index = terralib.newsymbol(uint64)
    local elmt = terralib.newsymbol(arg_type.element_type)
    local loop_body = add_phase_barrier_arg_recurse(add_barrier, launcher, elmt,
      arg_type.element_type)
    return quote
      for [index] = 0, [arg].__size do
        var [elmt] = [arg_type:data(arg)][ [index] ]
        [loop_body]
      end
    end
  end
end

local function expr_call_setup_phase_barrier_arg(
    cx, task, arg, condition, launcher, index, args_setup, arg_type)
  local add_barrier
  if condition == std.arrives then
    if index then
      add_barrier = c.legion_index_launcher_add_arrival_barrier
    else
      add_barrier = c.legion_task_launcher_add_arrival_barrier
    end
  elseif condition == std.awaits then
    if index then
      add_barrier = c.legion_index_launcher_add_wait_barrier
    else
      add_barrier = c.legion_task_launcher_add_wait_barrier
    end
  else
    assert(false)
  end

  args_setup:insert(
    add_phase_barrier_arg_recurse(add_barrier, launcher, arg, arg_type))
end

local function expr_call_setup_ispace_arg(
    cx, task, arg_type, param_type, launcher, index, args_setup)
  local parent_ispace =
    cx:ispace(cx:ispace(arg_type).root_ispace_type).index_space

  local add_requirement
  if index then
      add_requirement = c.legion_index_launcher_add_index_requirement
  else
      add_requirement = c.legion_task_launcher_add_index_requirement
  end
  assert(add_requirement)

  local requirement = terralib.newsymbol(uint, "requirement")
  local requirement_args = terralib.newlist({
      launcher, `([cx:ispace(arg_type).index_space]),
      c.ALL_MEMORY, `([parent_ispace]), false})

  args_setup:insert(
    quote
      var [requirement] = [add_requirement]([requirement_args])
    end)
end

local terra get_root_of_tree(runtime : c.legion_runtime_t,
                             r : c.legion_logical_region_t)
  while c.legion_logical_region_has_parent_logical_partition(runtime, r) do
    r = c.legion_logical_partition_get_parent_logical_region(
      runtime, c.legion_logical_region_get_parent_logical_partition(runtime, r))
  end
  return r
end

local function any_fields_are_scratch(cx, container_type, field_paths)
  assert(cx:has_region_or_list(container_type))
  local are_scratch = field_paths:map(
    function(field_path) return cx:region_or_list(container_type):field_is_scratch(field_path) end)
  return #are_scratch > 0 and data.all(unpack(are_scratch))
end

local function raise_privilege_depth(cx, value, container_type, field_paths, optional)
  -- This method is also used to adjust privilege depth for the
  -- callee's side, so check optional before asserting that
  -- container_type is not defined.
  local scratch = false
  if not optional or cx:has_region_or_list(container_type) then
    scratch = any_fields_are_scratch(cx, container_type, field_paths)
  end

  if scratch then
    value = `(get_root_of_tree([cx.runtime], [value]))
  elseif std.is_region(container_type) then
    local region = cx:region(
      cx:region(container_type).root_region_type).logical_region
    return `([region].impl)
  elseif std.is_list_of_regions(container_type) and container_type.region_root then
    assert(cx:has_region(container_type.region_root))
    local region = cx:region(
      cx:region(container_type.region_root).root_region_type).logical_region
    return `([region].impl)
  elseif std.is_list_of_regions(container_type) and container_type.privilege_depth then
    for i = 1, container_type.privilege_depth do
      value = `(
        c.legion_logical_partition_get_parent_logical_region(
          [cx.runtime],
          c.legion_logical_region_get_parent_logical_partition(
            [cx.runtime], [value])))
    end
  else
    assert(false)
  end
  return value
end

local function make_region_projection_functor(cx, expr)
  -- Right now we never generate any non-trivial region projection functors.

  return 0 -- Identity projection functor.
end

local function strip_casts(node)
  if node:is(ast.typed.expr.Cast) then
    return node.arg
  end
  return node
end

local function make_partition_projection_functor(cx, expr, loop_index, color_space)
  -- We assume that there's only one variable, and that it's the loop index.

  assert(expr:is(ast.typed.expr.IndexAccess))

  local index = strip_casts(expr.index)

  if index:is(ast.typed.expr.ID) then
    assert(index.value == loop_index)
    return 0 -- Identity projection functor.
  end

  local point = terralib.newsymbol(c.legion_domain_point_t, "point")

  local symbol_type = loop_index:gettype()
  local symbol = loop_index:getsymbol()
  local symbol_setup
  if std.is_bounded_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]({ __ptr = [symbol_type.index_type]([point]) })
    end
  else
    -- Otherwise symbol_type has to be some simple integral type.
    assert(symbol_type:isintegral())
    symbol_setup = quote
      var [symbol] = [int1d]([point])
    end
  end

  -- Again, if it's a number it has to be converted back through an index type.
  local index_type = std.as_read(index.expr_type)
  if index_type:isintegral() then
    index_type = int1d
  end

  -- Generate a projection functor that evaluates `expr`.
  local value = codegen.expr(cx, index):read(cx)
  local terra partition_functor(runtime : c.legion_runtime_t,
                                mappable : c.legion_mappable_t,
                                index : uint,
                                parent : c.legion_logical_partition_t,
                                [point])
    [symbol_setup];
    [value.actions];
    var index : index_type = [value.value];
    var subregion = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
      runtime, parent, index)
    return subregion
  end

  return std.register_projection_functor(0, nil, partition_functor)
end

local function expr_call_setup_region_arg(
    cx, task, arg_value, arg_type, param_type, launcher, index, args_setup)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)
  local privilege_modes = privileges:map(std.privilege_mode)
  local coherence_modes = coherences:map(std.coherence_mode)

  local add_field = c.legion_task_launcher_add_field
  if index then
    add_field = c.legion_index_launcher_add_field
  end

  local add_flags = c.legion_task_launcher_add_flags
  if index then
    add_flags = c.legion_index_launcher_add_flags
  end

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local field_types = privilege_field_types[i]
    local privilege_mode = privilege_modes[i]
    local coherence_mode = coherence_modes[i]
    local flag = std.flag_mode(flags[i])

    local region = `([cx:region(arg_type).logical_region].impl)
    local parent_region = raise_privilege_depth(
      cx, region, arg_type, field_paths)

    local reduction_op
    if std.is_reduction_op(privilege) then
      local op = std.get_reduction_op(privilege)
      assert(#field_types == 1)
      local field_type = field_types[1]
      reduction_op = std.reduction_op_ids[op][field_type]
    end

    if privilege_mode == c.REDUCE then
      assert(reduction_op)
    end

    local add_requirement
    if index then
      if reduction_op then
        add_requirement = c.legion_index_launcher_add_region_requirement_logical_region_reduction
     else
        add_requirement = c.legion_index_launcher_add_region_requirement_logical_region
      end
    else
      if reduction_op then
        add_requirement = c.legion_task_launcher_add_region_requirement_logical_region_reduction
      else
        add_requirement = c.legion_task_launcher_add_region_requirement_logical_region
      end
    end
    assert(add_requirement)

    local requirement = terralib.newsymbol(uint, "requirement")
    local requirement_args = terralib.newlist({
        launcher, region})
    if index then
      local projection_functor = make_region_projection_functor(cx, arg_value)
      requirement_args:insert(projection_functor)
    end
    if reduction_op then
      requirement_args:insert(reduction_op)
    else
      requirement_args:insert(privilege_mode)
    end
    requirement_args:insertall(
      {coherence_mode, parent_region, 0, false})

    args_setup:insert(
      quote
        var [requirement] = [add_requirement]([requirement_args])
        [field_paths:map(
           function(field_path)
             local field_id = cx:region(arg_type):field_id(field_path)
             return quote
               add_field(
                 [launcher], [requirement], [field_id], true)
             end
           end)]
        [add_flags]([launcher], [requirement], [flag])
      end)
  end
end

local function setup_list_of_regions_add_region(
    cx, param_type, container_type, value_type, value,
    region, parent, field_paths, add_requirement, get_requirement,
    add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)
  return quote
    var [region] = [raise_privilege_depth(cx, `([value].impl), param_type, field_paths, true)]
    var [parent] = [raise_privilege_depth(cx, `([value].impl), container_type, field_paths)]
    var requirement = [get_requirement]([launcher], [region])
    if requirement == [uint32](-1) then
      requirement = [add_requirement]([requirement_args])
      [add_flags]([launcher], requirement, [flag])
    else
      [intersect_flags]([launcher], requirement, [flag])
    end
    [field_paths:map(
       function(field_path)
         local field_id = cx:list_of_regions(container_type):field_id(field_path)
         return quote
           if not [has_field]([launcher], requirement, [field_id]) then
             [add_field]([launcher], requirement, [field_id], true)
           end
         end
       end)]
    end
end

local function setup_list_of_regions_add_list(
    cx, param_type, container_type, value_type, value,
    region, parent, field_paths, add_requirement, get_requirement,
    add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)
  local element = terralib.newsymbol(value_type.element_type)
  if std.is_list(value_type.element_type) then
    return quote
      for i = 0, [value].__size do
        var [element] = [value_type:data(value)][i]
        [setup_list_of_regions_add_list(
           cx, param_type, container_type, value_type.element_type, element,
           region, parent, field_paths, add_requirement, get_requirement,
           add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)]
      end
    end
  else
    return quote
      for i = 0, [value].__size do
        var [element] = [value_type:data(value)][i]
        [setup_list_of_regions_add_region(
           cx, param_type, container_type, value_type.element_type, element,
           region, parent, field_paths, add_requirement, get_requirement,
           add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)]
      end
    end
  end
end

local function expr_call_setup_list_of_regions_arg(
    cx, task, arg_type, param_type, launcher, index, args_setup)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)
  local privilege_modes = privileges:map(std.privilege_mode)
  local coherence_modes = coherences:map(std.coherence_mode)

  local add_field = c.legion_task_launcher_add_field
  if index then
    add_field = c.legion_index_launcher_add_field
  end

  local has_field = c.legion_terra_task_launcher_has_field
  if index then
    assert(false)
  end

  local add_flags = c.legion_task_launcher_add_flags
  if index then
    add_flags = c.legion_index_launcher_add_flags
  end

  local intersect_flags = c.legion_task_launcher_intersect_flags
  if index then
    intersect_flags = c.legion_index_launcher_intersect_flags
  end

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local field_types = privilege_field_types[i]
    local privilege_mode = privilege_modes[i]
    local coherence_mode = coherence_modes[i]
    local flag = std.flag_mode(flags[i])

    local reduction_op
    if std.is_reduction_op(privilege) then
      local op = std.get_reduction_op(privilege)
      assert(#field_types == 1)
      local field_type = field_types[1]
      reduction_op = std.reduction_op_ids[op][field_type]
    end

    if privilege_mode == c.REDUCE then
      assert(reduction_op)
    end

    local add_requirement
    if index then
      if reduction_op then
        add_requirement = c.legion_index_launcher_add_region_requirement_logical_region_reduction
     else
        add_requirement = c.legion_index_launcher_add_region_requirement_logical_region
      end
    else
      if reduction_op then
        add_requirement = c.legion_task_launcher_add_region_requirement_logical_region_reduction
      else
        add_requirement = c.legion_task_launcher_add_region_requirement_logical_region
      end
    end
    assert(add_requirement)

    local get_requirement
    if index then
      assert(false)
    else
      get_requirement = c.legion_terra_task_launcher_get_region_requirement_logical_region
    end
    assert(get_requirement)

    local list = cx:list_of_regions(arg_type).list_of_logical_regions

    local region = terralib.newsymbol(c.legion_logical_region_t, "region")
    local parent = terralib.newsymbol(c.legion_logical_region_t, "parent")
    local requirement_args = terralib.newlist({launcher, region})
    if index then
      requirement_args:insert(0)
    end
    if reduction_op then
      requirement_args:insert(reduction_op)
    else
      requirement_args:insert(privilege_mode)
    end
    requirement_args:insertall(
      {coherence_mode, parent, 0, false})

    args_setup:insert(
      setup_list_of_regions_add_list(
        cx, param_type, arg_type, arg_type, list,
        region, parent, field_paths, add_requirement, get_requirement,
        add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher))
  end
end

local function expr_call_setup_partition_arg(
    cx, task, arg_value, arg_type, param_type, partition, loop_index, launcher, index, args_setup)
  assert(index)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)
  local privilege_modes = privileges:map(std.privilege_mode)
  local coherence_modes = coherences:map(std.coherence_mode)
  local parent_region =
    cx:region(cx:region(arg_type).root_region_type).logical_region

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local field_types = privilege_field_types[i]
    local privilege_mode = privilege_modes[i]
    local coherence_mode = coherence_modes[i]
    local flag = std.flag_mode(flags[i])

    local reduction_op
    if std.is_reduction_op(privilege) then
      local op = std.get_reduction_op(privilege)
      assert(#field_types == 1)
      local field_type = field_types[1]
      reduction_op = std.reduction_op_ids[op][field_type]
    end

    if privilege_mode == c.REDUCE then
      assert(reduction_op)
    end

    local add_requirement
    if reduction_op then
      add_requirement = c.legion_index_launcher_add_region_requirement_logical_partition_reduction
    else
      add_requirement = c.legion_index_launcher_add_region_requirement_logical_partition
    end
    assert(add_requirement)

    local projection_functor = make_partition_projection_functor(cx, arg_value, loop_index)

    local requirement = terralib.newsymbol(uint, "requirement")
    local requirement_args = terralib.newlist({
        launcher, `([partition].impl), projection_functor})

    if reduction_op then
      requirement_args:insert(reduction_op)
    else
      requirement_args:insert(privilege_mode)
    end
    requirement_args:insertall(
      {coherence_mode, `([parent_region].impl), 0, false})

    args_setup:insert(
      quote
      var [requirement] =
        [add_requirement]([requirement_args])
        [field_paths:map(
           function(field_path)
             local field_id = cx:region(arg_type):field_id(field_path)
             return quote
               c.legion_index_launcher_add_field(
                 [launcher], [requirement], [field_id], true)
             end
           end)]
      c.legion_index_launcher_add_flags([launcher], [requirement], [flag])
      end)
  end
end

function codegen.expr_call(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local args = node.args:map(
    function(arg) return codegen.expr(cx, arg):read(cx, arg.expr_type) end)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)

  local actions = quote
    [fn.actions];
    [args:map(function(arg) return arg.actions end)];
    [conditions:map(function(condition) return condition.actions end)];
    [emit_debuginfo(node)]
  end

  local arg_types = terralib.newlist()
  for i, arg in ipairs(args) do
    arg_types:insert(std.as_read(node.args[i].expr_type))
  end

  local arg_values = terralib.newlist()
  local param_types = node.fn.expr_type.parameters
  for i, arg in ipairs(args) do
    local arg_value = args[i].value
    if i <= #param_types and param_types[i] ~= std.untyped and
      not std.is_future(arg_types[i])
    then
      arg_values:insert(std.implicit_cast(arg_types[i], param_types[i], arg_value))
    else
      arg_values:insert(arg_value)
    end
  end

  local value_type = std.as_read(node.expr_type)
  if std.is_task(fn.value) then
    local params_struct_type = fn.value:get_params_struct()
    local task_args = terralib.newsymbol(c.legion_task_argument_t, "task_args")
    local task_args_setup = terralib.newlist()
    local task_args_cleanup = terralib.newlist()
    expr_call_setup_task_args(
      cx, fn.value, arg_values, arg_types, param_types,
      params_struct_type, fn.value:has_params_map_label(), fn.value:has_params_map_type(),
      task_args, task_args_setup, task_args_cleanup)

    local launcher = terralib.newsymbol(c.legion_task_launcher_t, "launcher")

    -- Pass futures.
    local args_setup = terralib.newlist()
    for i, arg_type in ipairs(arg_types) do
      if std.is_future(arg_type) then
        local arg_value = arg_values[i]
        expr_call_setup_future_arg(
          cx, fn.value, arg_value,
          launcher, false, args_setup)
      end
    end

    -- Pass phase barriers (from annotations on parameters).
    local param_conditions = fn.value:get_conditions()
    for condition, args_enabled in pairs(param_conditions) do
      for i, arg_type in ipairs(arg_types) do
        if args_enabled[i] then
          assert(std.is_phase_barrier(arg_type) or
            (std.is_list(arg_type) and std.is_phase_barrier(arg_type.element_type)))
          local arg_value = arg_values[i]
          expr_call_setup_phase_barrier_arg(
            cx, fn.value, arg_value, condition,
            launcher, false, args_setup, arg_type)
        end
      end
    end

    -- Pass phase barriers (from extra conditions).
    for i, condition in ipairs(node.conditions) do
      local condition_expr = conditions[i]
      for _, condition_kind in ipairs(condition.conditions) do
        expr_call_setup_phase_barrier_arg(
          cx, fn.value, condition_expr.value, condition_kind,
          launcher, false, args_setup, std.as_read(condition.expr_type))
      end
    end

    -- Pass index spaces through index requirements.
    for i, arg_type in ipairs(arg_types) do
      if std.is_ispace(arg_type) then
        local param_type = param_types[i]

        expr_call_setup_ispace_arg(
          cx, fn.value, arg_type, param_type, launcher, false, args_setup)
      end
    end

    -- Pass regions through region requirements.
    local fn_type = fn.value:get_type()
    for _, i in ipairs(std.fn_param_regions_by_index(fn_type)) do
      local arg_value = arg_values[i]
      local arg_type = arg_types[i]
      local param_type = param_types[i]

      expr_call_setup_region_arg(
        cx, fn.value, node.args[i], arg_type, param_type, launcher, false, args_setup)
    end

    -- Pass regions through lists of region requirements.
    for _, i in ipairs(std.fn_param_lists_of_regions_by_index(fn_type)) do
      local arg_type = arg_types[i]
      local param_type = param_types[i]

      expr_call_setup_list_of_regions_arg(
        cx, fn.value, arg_type, param_type, launcher, false, args_setup)
    end

    local future
    if not cx.must_epoch then
      future = terralib.newsymbol(c.legion_future_t, "future")
    end

    local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
    local launcher_setup = quote
      var [task_args]
      [task_args_setup]
      var [tag] = 0
      [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
      var [launcher] = c.legion_task_launcher_create(
        [fn.value:get_task_id()], [task_args],
        c.legion_predicate_true(), 0, [tag])
      [args_setup]
    end

    local launcher_execute
    if not cx.must_epoch then
      launcher_execute = quote
        var [future] = c.legion_task_launcher_execute(
          [cx.runtime], [cx.context], [launcher])
        c.legion_task_launcher_destroy(launcher)
        [task_args_cleanup]
      end
    else
      launcher_execute = quote
        c.legion_must_epoch_launcher_add_single_task(
          [cx.must_epoch],
          [int1d]([cx.must_epoch_point]),
          [launcher])
        [cx.must_epoch_point] = [cx.must_epoch_point] + 1
      end
    end

    actions = quote
      [actions]
      [launcher_setup]
      [launcher_execute]
    end

    local future_type = value_type
    if not std.is_future(future_type) then
      future_type = std.future(value_type)
    end

    local future_value
    if future then
      future_value = values.value(
        node,
        expr.once_only(actions, `([future_type]{ __result = [future] }), future_type),
        value_type)
    end

    if std.is_future(value_type) then
      assert(future_value)
      return future_value
    elseif value_type == terralib.types.unit then
      if future then
        actions = quote
          [actions]
          c.legion_future_destroy(future)
        end
      end

      return values.value(node, expr.just(actions, quote end), terralib.types.unit)
    else
      assert(future_value)
      local value = codegen.expr(
        cx,
        ast.typed.expr.FutureGetResult {
          value = ast.typed.expr.Internal {
            value = future_value,
            expr_type = future_type,
            annotations = node.annotations,
            span = node.span,
          },
          expr_type = value_type,
          annotations = node.annotations,
          span = node.span,
        }):read(cx)
      local actions = quote
        [value.actions]
        c.legion_future_destroy(future)
      end
      return values.value(node, expr.just(actions, value.value), value_type)
    end
  else
    return values.value(
      node,
      expr.once_only(actions, `([fn.value]([arg_values])), value_type),
      value_type)
  end
end

function codegen.expr_cast(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local arg = codegen.expr(cx, node.arg):read(cx, node.arg.expr_type)

  local actions = quote
    [fn.actions];
    [arg.actions];
    [emit_debuginfo(node)]
  end
  local expr_type = std.as_read(node.expr_type)
  return values.value(
    node,
    expr.once_only(actions, `([fn.value]([arg.value])), expr_type),
    expr_type)
end

function codegen.expr_ctor_list_field(cx, node)
  return codegen.expr(cx, node.value):read(cx)
end

function codegen.expr_ctor_rec_field(cx, node)
  return  codegen.expr(cx, node.value):read(cx)
end

function codegen.expr_ctor_field(cx, node)
  if node:is(ast.typed.expr.CtorListField) then
    return codegen.expr_ctor_list_field(cx, node)
  elseif node:is(ast.typed.expr.CtorRecField) then
    return codegen.expr_ctor_rec_field(cx, node)
  else
  end
end

function codegen.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return codegen.expr_ctor_field(cx, field) end)

  local field_values = fields:map(function(field) return field.value end)
  local actions = quote
    [fields:map(function(field) return field.actions end)];
    [emit_debuginfo(node)]
  end
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    node,
    expr.once_only(actions, `([expr_type]{ [field_values] }), expr_type),
    expr_type)
end

function codegen.expr_raw_context(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    node,
    expr.just(emit_debuginfo(node), cx.context),
    value_type)
end

function codegen.expr_raw_fields(cx, node)
  local region = codegen.expr(cx, node.region):read(cx)
  local region_type = std.as_read(node.region.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local region = cx:region(region_type)
  local field_ids = terralib.newlist()
  for i, field_path in ipairs(node.fields) do
    field_ids:insert({i-1, region:field_id(field_path)})
  end

  local result = terralib.newsymbol(expr_type, "raw_fields")
  local actions = quote
    [emit_debuginfo(node)]
    var [result]
    [field_ids:map(
       function(pair)
         local i, field_id = unpack(pair)
         return quote [result][ [i] ] = [field_id] end
       end)]
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_raw_physical(cx, node)
  local region = codegen.expr(cx, node.region):read(cx)
  local region_type = std.as_read(node.region.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local region = cx:region(region_type)
  local physical_regions = terralib.newlist()
  for i, field_path in ipairs(node.fields) do
    physical_regions:insert({i-1, region:physical_region(field_path)})
  end

  local result = terralib.newsymbol(expr_type, "raw_physical")
  local actions = quote
    [emit_debuginfo(node)]
    var [result]
    [physical_regions:map(
       function(pair)
         local i, physical_region = unpack(pair)
         return quote [result][ [i] ] = [physical_region] end
       end)]
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_raw_runtime(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    node,
    expr.just(emit_debuginfo(node), cx.runtime),
    value_type)
end

function codegen.expr_raw_value(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = value.actions
  local result
  if std.is_ispace(value_type) then
    result = `([value.value].impl)
  elseif std.is_region(value_type) then
    result = `([value.value].impl)
  elseif std.is_partition(value_type) then
    result = `([value.value].impl)
  elseif std.is_cross_product(value_type) then
    result = `([value.value].product)
  elseif std.is_bounded_type(value_type) then
    result = `([value.value].__ptr.__ptr)
  else
    assert(false)
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_isnull(cx, node)
  local pointer = codegen.expr(cx, node.pointer):read(cx)
  local index_type = std.as_read(node.pointer.expr_type).index_type
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [pointer.actions];
    [emit_debuginfo(node)]
  end

  if index_type:is_opaque() then
    return values.value(
      node,
      expr.once_only(
        actions,
        `([expr_type](c.legion_ptr_is_null([pointer.value].__ptr.__ptr))),
        expr_type),
      expr_type)
  else
    return values.value(
      node,
      expr.once_only(
        actions,
        `([expr_type]([index_type]([pointer.value]) == [index_type:nil_index()])),
        expr_type),
      expr_type)
  end
end

function codegen.expr_null(cx, node)
  local pointer_type = node.pointer_type
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    node,
    expr.once_only(
      emit_debuginfo(node),
      `([pointer_type]{ __ptr = [ptr] { __ptr = c.legion_ptr_nil() }}),
      expr_type),
    expr_type)
end

function codegen.expr_dynamic_cast(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  if std.is_partition(expr_type) then
    local result = terralib.newsymbol(expr_type)
    actions = quote
      [actions]
      var [result] = [expr_type] { impl = [value.value].impl }
    end
    if expr_type:is_disjoint() and not value_type:is_disjoint() then
      actions = quote
        [actions]
        std.assert_error(
            c.legion_index_partition_is_disjoint([cx.runtime], [result].impl.index_partition),
            [get_source_location(node) .. ": " .. pretty.entry_expr(node.value) ..
            " is not a disjoint partition"])
      end
    end
    return values.value(node, expr.once_only(actions, result, expr_type), expr_type)
  end

  local input = `([std.implicit_cast(value_type, expr_type.index_type, value.value)])

  local result
  local regions = expr_type:bounds()
  if #regions == 1 then
    local region = regions[1]
    assert(cx:has_region(region))
    local lr = `([cx:region(region).logical_region].impl)
    if expr_type.index_type:is_opaque() then
      result = `(
        [expr_type]({
          __ptr = [ptr] {
            __ptr = c.legion_ptr_safe_cast([cx.runtime], [cx.context], [input].__ptr, [lr])
          }
        }))
    else
      result = `(
        [expr_type]({
            __ptr = [expr_type.index_type](
              c.legion_domain_point_safe_cast([cx.runtime], [cx.context], [input], [lr]))
        }))
    end
  else
    result = terralib.newsymbol(expr_type)
    local cases
    if expr_type.index_type:is_opaque() then
      cases = quote
        [result] = [expr_type]({ __ptr = [ptr] { __ptr = c.legion_ptr_nil() }, __index = 0 })
      end
      for i = #regions, 1, -1 do
        local region = regions[i]
        assert(cx:has_region(region))
        local lr = `([cx:region(region).logical_region].impl)
        cases = quote
          var temp = c.legion_ptr_safe_cast([cx.runtime], [cx.context], [input].__ptr, [lr])
          if not c.legion_ptr_is_null(temp) then
            result = [expr_type]({
              __ptr = [ptr] { __ptr = temp },
              __index = [i],
            })
          else
            [cases]
          end
        end
      end
    else
      cases = quote
        [result] = [expr_type]({
          __ptr = [expr_type.index_type:nil_index()],
          __index = 0
        })
      end
      for i = #regions, 1, -1 do
        local region = regions[i]
        assert(cx:has_region(region))
        local lr = `([cx:region(region).logical_region].impl)
        cases = quote
          var temp = c.legion_domain_point_safe_cast([cx.runtime], [cx.context], [input], [lr])
          if temp.dim ~= -1 then
            result = [expr_type]({
              __ptr = [expr_type.index_type](temp),
              __index = [i],
            })
          else
            [cases]
          end
        end
      end
    end

    actions = quote [actions]; var [result]; [cases] end
  end

  return values.value(node, expr.once_only(actions, result, expr_type), expr_type)
end

function codegen.expr_static_cast(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end
  local input = value.value
  local result
  if #(expr_type:bounds()) == 1 then
    result = terralib.newsymbol(expr_type)
    local input_regions = value_type:bounds()
    local result_last = node.parent_region_map[#input_regions]
    local cases
    if result_last then
      cases = quote
        [result] = [expr_type]({ __ptr = [input].__ptr })
      end
    else
      cases = quote
        [result] = [expr_type]({ __ptr = [ptr] { __ptr = c.legion_ptr_nil() }})
      end
    end
    for i = #input_regions - 1, 1, -1 do
      local result_i = node.parent_region_map[i]
      if result_i then
        cases = quote
          if [input].__index == [i] then
            [result] = [expr_type]({ __ptr = [input].__ptr })
          else
            [cases]
          end
        end
      else
        cases = quote
          if [input].__index == [i] then
            [result] = [expr_type]({ __ptr = [ptr] { __ptr = c.legion_ptr_nil() }})
          else
            [cases]
          end
        end
      end
    end

    actions = quote [actions]; var [result]; [cases] end
  else
    result = terralib.newsymbol(expr_type)
    local input_regions = value_type:bounds()
    local result_last = node.parent_region_map[#input_regions]
    local cases
    if result_last then
      cases = quote
        [result] = [expr_type]({
            __ptr = [input].__ptr,
            __index = [result_last],
        })
      end
    else
      cases = quote
        [result] = [expr_type]({
            __ptr = c.legion_ptr_nil(),
            __index = 0,
        })
      end
    end
    for i = #input_regions - 1, 1, -1 do
      local result_i = node.parent_region_map[i]
      if result_i then
        cases = quote
          if [input].__index == [i] then
            [result] = [expr_type]({
              __ptr = [input].__ptr,
              __index = [result_i],
            })
          else
            [cases]
          end
        end
      else
        cases = quote
          if [input].__index == [i] then
            [result] = [expr_type]({
              __ptr = c.legion_ptr_nil(),
              __index = 0,
            })
          else
            [cases]
          end
        end
      end
    end

    actions = quote [actions]; var [result]; [cases] end
  end

  return values.value(node, expr.once_only(actions, result, expr_type), expr_type)
end

function codegen.expr_unsafe_cast(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  if not std.is_vptr(value_type) then
    local input = std.implicit_cast(value_type, expr_type.index_type, value.value)

    if bounds_checks then
      local regions = expr_type:bounds()
      assert(#regions == 1)
      local region = regions[1]
      assert(cx:has_region(region))
      local lr = `([cx:region(region).logical_region].impl)

      if expr_type.index_type:is_opaque() then
        local check = terralib.newsymbol(c.legion_ptr_t, "check")
        actions = quote
          [actions]
          var [check] = c.legion_ptr_safe_cast([cx.runtime], [cx.context], [input].__ptr, [lr])
          if c.legion_ptr_is_null([check]) then
            std.assert_error(false, [get_source_location(node) .. ": pointer " .. tostring(expr_type) .. " is out-of-bounds"])
          end
          var [result] = [expr_type]({ __ptr = [ptr] { __ptr = [check] }})
        end
      else
        local check = terralib.newsymbol(c.legion_domain_point_t, "check")
        actions = quote
          [actions]
          var [check] = c.legion_domain_point_safe_cast([cx.runtime], [cx.context], [input], [lr])
          if c.legion_domain_point_is_null([check]) then
            std.assert_error(false, [get_source_location(node) .. ": pointer " .. tostring(expr_type) .. " is out-of-bounds"])
          end
          var [result] = [expr_type]({ __ptr = [expr_type.index_type]([check]) })
        end
      end
    else
      if expr_type.index_type:is_opaque() then
        actions = quote
          [actions]
          var [result] = [expr_type]({ __ptr = [ptr] { __ptr = [input].__ptr }})
        end
      else
        actions = quote
          [actions]
          var [result] = [expr_type]({ __ptr = [input] })
        end
      end
    end
  else
    -- TODO: bounds checks are ignored for vptr
    assert(#expr_type:bounds() == 1)
    actions = quote
      [actions]
      var [result] = [expr_type]({ __ptr =
        [expr_type.impl_type] {
          value = [value.value].__ptr.value
        }
      })
    end
  end

  return values.value(node, expr.just(actions, result), expr_type)
end

function codegen.expr_ispace(cx, node)
  local index_type = node.index_type
  local extent = codegen.expr(cx, node.extent):read(cx)
  local extent_type = std.as_read(node.extent.expr_type)
  local start = node.start and codegen.expr(cx, node.start):read(cx)
  local ispace_type = std.as_read(node.expr_type)
  local actions = quote
    [extent.actions];
    [start and start.actions or (quote end)];
    [emit_debuginfo(node)]
  end

  local extent_value = `([std.implicit_cast(extent_type, index_type, extent.value)].__ptr)
  if index_type:is_opaque() then
    extent_value = `([extent_value].value)
  end

  local start_value = start and `([std.implicit_cast(start_type, index_type, start.value)].__ptr)
  if index_type:is_opaque() then
    start_value = start and `([start_value].value)
  end

  local is = terralib.newsymbol(c.legion_index_space_t, "is")
  local i = terralib.newsymbol(ispace_type, "i")

  local it = false
  if cache_index_iterator then
    it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
  end

  local bounds, bounds_actions = terralib.newlist()
  if not ispace_type:is_opaque() then
    bounds_actions, bounds = index_space_bounds(cx, is, ispace_type.index_type)
  end
  cx:add_ispace_root(ispace_type, is, it, bounds)

  if ispace_type.dim == 0 then
    if start then
      actions = quote
        [actions]
        std.assert([start_value] == 0, "opaque ispaces must start at 0 right now")
      end
    end
    actions = quote
      [actions]
      var [is] = c.legion_index_space_create([cx.runtime], [cx.context], [extent_value])
    end

    if cache_index_iterator then
      actions = quote
        [actions]
        var [it] = c.legion_terra_cached_index_iterator_create(
          [cx.runtime], [cx.context], [is])
      end
    end
  else
    if not start then
      start_value = index_type:zero()
    end

    local domain_from_bounds = std["domain_from_bounds_" .. tostring(ispace_type.dim) .. "d"]
    actions = quote
      [actions]
      var domain = [domain_from_bounds](
        ([index_type](start_value)):to_point(),
        ([index_type](extent_value)):to_point())
      var [is] = c.legion_index_space_create_domain([cx.runtime], [cx.context], domain)
      [bounds_actions]
    end
  end

  actions = quote
    [actions]
    var [i] = [ispace_type]{ impl = [is] }
  end

  return values.value(node, expr.just(actions, i), ispace_type)
end

function codegen.expr_region(cx, node)
  local fspace_type = node.fspace_type
  local ispace = codegen.expr(cx, node.ispace):read(cx)
  local region_type = std.as_read(node.expr_type)
  local index_type = region_type:ispace().index_type
  local actions = quote
    [ispace.actions];
    [emit_debuginfo(node)]
  end

  local r = terralib.newsymbol(region_type, "r")
  local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
  local is = terralib.newsymbol(c.legion_index_space_t, "is")
  local fs = terralib.newsymbol(c.legion_field_space_t, "fs")
  local pr = terralib.newsymbol(c.legion_physical_region_t, "pr")

  local field_paths, field_types = std.flatten_struct_fields(fspace_type)
  local field_privileges = field_paths:map(function(_) return "reads_writes" end)
  local field_id = 100
  local field_ids = field_paths:map(
    function(_)
      field_id = field_id + 1
      return field_id
    end)
  local fields_are_scratch = field_paths:map(function(_) return false end)
  local physical_regions = field_paths:map(function(_) return pr end)

  local pr_actions, base_pointers, strides = unpack(data.zip(unpack(
    data.zip(field_types, field_ids, field_privileges):map(
      function(field)
        local field_type, field_id, field_privilege = unpack(field)
        return terralib.newlist({
            physical_region_get_base_pointer(cx, index_type, field_type, field_id, field_privilege, pr)})
  end))))
  pr_actions = pr_actions or terralib.newlist()
  base_pointers = base_pointers or terralib.newlist()
  strides = strides or terralib.newlist()

  cx:add_region_root(region_type, r,
                     field_paths,
                     terralib.newlist({field_paths}),
                     data.dict(data.zip(field_paths:map(data.hash), field_privileges)),
                     data.dict(data.zip(field_paths:map(data.hash), field_types)),
                     data.dict(data.zip(field_paths:map(data.hash), field_ids)),
                     data.dict(data.zip(field_paths:map(data.hash), fields_are_scratch)),
                     data.dict(data.zip(field_paths:map(data.hash), physical_regions)),
                     data.dict(data.zip(field_paths:map(data.hash), base_pointers)),
                     data.dict(data.zip(field_paths:map(data.hash), strides)))

  local fs_naming_actions
  if fspace_type:isstruct() then
    fs_naming_actions = quote
      c.legion_field_space_attach_name([cx.runtime], [fs], [fspace_type.name], false)
      [data.zip(field_paths, field_ids):map(
         function(field)
           local field_path, field_id = unpack(field)
           local field_name = field_path:mkstring("", ".", "")
           return `(c.legion_field_id_attach_name(
                      [cx.runtime], [fs], field_id, field_name, false))
         end)]
    end
  else
    fs_naming_actions = quote end
  end

  actions = quote
    [actions]
    var capacity = [ispace.value]
    var [is] = [ispace.value].impl
    var [fs] = c.legion_field_space_create([cx.runtime], [cx.context])
    var fsa = c.legion_field_allocator_create([cx.runtime], [cx.context],  [fs]);
    [data.zip(field_types, field_ids):map(
       function(field)
         local field_type, field_id = unpack(field)
         return `(c.legion_field_allocator_allocate_field(
                    fsa, terralib.sizeof([field_type]), [field_id]))
       end)]
    [fs_naming_actions];
    c.legion_field_allocator_destroy(fsa)
    var [lr] = c.legion_logical_region_create([cx.runtime], [cx.context], [is], [fs], true)
    var [r] = [region_type]{ impl = [lr] }
  end
  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  if not cx.variant:get_config_options().inner then
    actions = quote
      [actions];
      var [tag] = 0
      [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
      var il = c.legion_inline_launcher_create_logical_region(
        [lr], c.READ_WRITE, c.EXCLUSIVE, [lr], 0, false, 0, [tag]);
      [field_ids:map(
         function(field_id)
           return `(c.legion_inline_launcher_add_field(il, [field_id], true))
         end)]
      var [pr] = c.legion_inline_launcher_execute([cx.runtime], [cx.context], il)
      c.legion_inline_launcher_destroy(il)
      c.legion_physical_region_wait_until_valid([pr])
      [pr_actions]
    end
  else -- make sure all regions are unmapped in inner tasks
    actions = quote
      [actions];
      c.legion_runtime_unmap_all_regions([cx.runtime], [cx.context])
      var [pr] -- FIXME: Need to define physical region for detach to work
    end
  end

  return values.value(node, expr.just(actions, r), region_type)
end

function codegen.expr_partition(cx, node)
  local region = codegen.expr(cx, node.region):read(cx)
  local coloring_type = std.as_read(node.coloring.expr_type)
  local coloring = codegen.expr(cx, node.coloring):read(cx)
  local colors = node.colors and codegen.expr(cx, node.colors):read(cx)
  local partition_type = std.as_read(node.expr_type)
  local actions = quote
    [region.actions];
    [coloring.actions];
    [(colors and colors.actions) or (quote end)];
    [emit_debuginfo(node)]
  end

  local index_partition_create
  local args = terralib.newlist({
      cx.runtime, cx.context,
      `([region.value].impl.index_space),
  })

  if colors then
    local color_space = terralib.newsymbol(c.legion_domain_t)
    args:insert(color_space)
    actions = quote
      [actions]
      var [color_space] = c.legion_index_space_get_domain(
        [cx.runtime], [colors.value].impl)
    end
  elseif coloring_type == std.c.legion_domain_coloring_t then
    local color_space = terralib.newsymbol(c.legion_domain_t)
    args:insert(color_space)
    actions = quote
      [actions]
      var [color_space] = c.legion_domain_coloring_get_color_space([coloring.value])
    end
  end

  if coloring_type == std.c.legion_coloring_t then
    index_partition_create = c.legion_index_partition_create_coloring
  elseif coloring_type == std.c.legion_domain_coloring_t then
    index_partition_create = c.legion_index_partition_create_domain_coloring
  elseif coloring_type == std.c.legion_point_coloring_t then
    index_partition_create = c.legion_index_partition_create_point_coloring
  elseif coloring_type == std.c.legion_domain_point_coloring_t then
    index_partition_create = c.legion_index_partition_create_domain_point_coloring
  elseif coloring_type == std.c.legion_multi_domain_point_coloring_t then
    index_partition_create = c.legion_index_partition_create_multi_domain_point_coloring
  else
    assert(false)
  end

  args:insert(coloring.value)


  if coloring_type == std.c.legion_coloring_t or
    coloring_type == std.c.legion_domain_coloring_t
  then
    args:insert(partition_type:is_disjoint())
  elseif coloring_type == std.c.legion_point_coloring_t or
    coloring_type == std.c.legion_domain_point_coloring_t or
    coloring_type == std.c.legion_multi_domain_point_coloring_t
  then
    args:insert(
      (partition_type:is_disjoint() and c.DISJOINT_KIND) or c.ALIASED_KIND)
  else
    assert(false)
  end

  args:insert(-1) -- AUTO_GENERATE_ID


  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  actions = quote
    [actions]
    var [ip] = [index_partition_create]([args])
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], [region.value].impl, [ip])
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_partition_equal(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr(cx, node.region):read(cx)
  local colors_type = std.as_read(node.colors.expr_type)
  local colors = codegen.expr(cx, node.colors):read(cx)
  local partition_type = std.as_read(node.expr_type)
  local actions = quote
    [region.actions];
    [colors.actions];
    [emit_debuginfo(node)]
  end

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")

  if region_type:is_opaque() then
    actions = quote
      [actions]
      var domain = c.legion_index_space_get_domain(
        [cx.runtime], [colors.value].impl)
      var [ip] = c.legion_index_partition_create_equal(
        [cx.runtime], [cx.context], [region.value].impl.index_space,
        [colors.value].impl, 1 --[[ granularity ]], -1 --[[ AUTO_GENERATE_ID ]])
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [cx.context], [region.value].impl, [ip])
    end
  else
    local dim = region_type:ispace().dim
    local blockify = terralib.newsymbol(
      c["legion_blockify_" .. tostring(dim) .. "d_t"], "blockify")
    local domain_get_rect =
      c["legion_domain_get_rect_" .. tostring(dim) .. "d"]
    local create_index_partition =
      c["legion_index_partition_create_blockify_" .. tostring(dim) .. "d"]

    actions = quote
      [actions]
      var region_domain = c.legion_index_space_get_domain(
        [cx.runtime], [region.value].impl.index_space)
      var region_rect = [domain_get_rect](region_domain)
      var color_domain = c.legion_index_space_get_domain(
        [cx.runtime], [colors.value].impl)
      var color_rect = [domain_get_rect](color_domain)

      var [blockify]
      [data.range(dim):map(
         function(i)
           local block = `(region_rect.hi.x[ [i] ] - region_rect.lo.x[ [i] ] + 1)
           local colors = `(color_rect.hi.x[ [i] ] - color_rect.lo.x[ [i] ] + 1)
           return quote
             [blockify].block_size.x[ [i] ] = ([block] + [colors] - 1) / [colors]
             [blockify].offset.x[ [i] ] = region_rect.lo.x[ [i] ]
           end
         end)]
      var [ip] = [create_index_partition](
        [cx.runtime], [cx.context], [region.value].impl.index_space,
        [blockify], -1 --[[ AUTO_GENERATE_ID ]])
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [cx.context], [region.value].impl, [ip])
    end
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_partition_by_field(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx)
  local colors_type = std.as_read(node.colors.expr_type)
  local colors = codegen.expr(cx, node.colors):read(cx)
  local partition_type = std.as_read(node.expr_type)
  local actions = quote
    [region.actions];
    [colors.actions];
    [emit_debuginfo(node)]
  end

  assert(cx:has_region(region_type))
  local parent_region =
    cx:region(cx:region(region_type).root_region_type).logical_region

  local fields = std.flatten_struct_fields(region_type:fspace())
  local field_paths = data.filter(
    function(field) return field:starts_with(node.region.fields[1]) end,
    fields)
  assert(#field_paths == 1)

  local field_id = cx:region(region_type):field_id(field_paths[1])

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  actions = quote
    [actions]
    var [ip] = c.legion_index_partition_create_by_field(
      [cx.runtime], [cx.context], [region.value].impl, [parent_region].impl,
      field_id, [colors.value].impl, -1)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], [region.value].impl, [ip])
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_image(cx, node)
  local parent_type = std.as_read(node.parent.expr_type)
  local parent = codegen.expr(cx, node.parent):read(cx)
  local partition_type = std.as_read(node.partition.expr_type)
  local partition = codegen.expr(cx, node.partition):read(cx)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx)

  local result_type = std.as_read(node.expr_type)
  local actions = quote
    [parent.actions];
    [partition.actions];
    [region.actions];
    [emit_debuginfo(node)]
  end

  assert(cx:has_region(region_type))
  local region_parent =
    cx:region(cx:region(region_type).root_region_type).logical_region

  local field_paths, field_types =
    std.flatten_struct_fields(region_type:fspace())
  local fields_i = data.filteri(
    function(field) return field:starts_with(node.region.fields[1]) end,
    field_paths)
  if #fields_i > 1 then
    fields_i = data.filteri(
      function(field) return field:starts_with(node.region.fields[1] .. data.newtuple("__ptr")) end,
      field_paths)
  end
  assert(#fields_i == 1)
  local index_fields =
    std.flatten_struct_fields(field_types[fields_i[1]])
  local field_path = field_paths[fields_i[1]] .. index_fields[1]

  local field_id = cx:region(region_type):field_id(field_path)

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local create_partition
  if std.is_rect_type(field_types[fields_i[1]]) then
    create_partition = c.legion_index_partition_create_by_image_range
  else
    create_partition = c.legion_index_partition_create_by_image
  end
  actions = quote
    [actions]
    var colors = c.legion_index_partition_get_color_space(
      [cx.runtime], [partition.value].impl.index_partition)
    var [ip] = [create_partition](
      [cx.runtime], [cx.context],
      [parent.value].impl.index_space,
      [partition.value].impl, [region_parent].impl, field_id,
      colors, c.COMPUTE_KIND, -1)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], [parent.value].impl, [ip])
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_preimage(cx, node)
  local parent_type = std.as_read(node.parent.expr_type)
  local parent = codegen.expr(cx, node.parent):read(cx)
  local partition_type = std.as_read(node.partition.expr_type)
  local partition = codegen.expr(cx, node.partition):read(cx)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx)

  local result_type = std.as_read(node.expr_type)
  local actions = quote
    [parent.actions];
    [partition.actions];
    [region.actions];
    [emit_debuginfo(node)]
  end

  assert(cx:has_region(region_type))
  local region_parent =
    cx:region(cx:region(region_type).root_region_type).logical_region

  local field_paths, field_types =
    std.flatten_struct_fields(region_type:fspace())
  local fields_i = data.filteri(
    function(field) return field:starts_with(node.region.fields[1]) end,
    field_paths)
  if #fields_i > 1 then
    fields_i = data.filteri(
      function(field) return field:starts_with(node.region.fields[1] .. data.newtuple("__ptr")) end,
      field_paths)
  end
  assert(#fields_i == 1)
  local index_fields =
    std.flatten_struct_fields(field_types[fields_i[1]])
  local field_path = field_paths[fields_i[1]] .. index_fields[1]

  local field_id = cx:region(region_type):field_id(field_path)

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local create_partition
  if std.is_rect_type(field_types[fields_i[1]]) then
    create_partition = c.legion_index_partition_create_by_preimage_range
  else
    create_partition = c.legion_index_partition_create_by_preimage
  end
  actions = quote
    [actions]
    var colors = c.legion_index_partition_get_color_space(
      [cx.runtime], [partition.value].impl.index_partition)
    var [ip] = [create_partition](
      [cx.runtime], [cx.context], [partition.value].impl.index_partition,
      [parent.value].impl, [region_parent].impl, field_id, colors,
      [(result_type:is_disjoint() and
        result_type:parent_region():is_opaque() and
        c.DISJOINT_KIND) or
       c.COMPUTE_KIND],
      -1)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], [region.value].impl, [ip])
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_cross_product(cx, node)
  local args = node.args:map(function(arg) return codegen.expr(cx, arg):read(cx) end)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [args:map(function(arg) return arg.actions end)]
    [emit_debuginfo(node)]
  end

  local partitions = terralib.newsymbol(
    c.legion_index_partition_t[#args], "partitions")
  local colors = terralib.newsymbol(c.legion_color_t[#args], "colors")
  local product = terralib.newsymbol(
    c.legion_terra_index_cross_product_t, "cross_product")
  local lr = cx:region(expr_type:parent_region()).logical_region
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  actions = quote
    [actions]
    var [partitions]
    [data.zip(data.range(#args), args):map(
       function(pair)
         local i, arg = unpack(pair)
         return quote partitions[i] = [arg.value].impl.index_partition end
       end)]
    var [colors]
    var [product] = c.legion_terra_index_cross_product_create_multi(
      [cx.runtime], [cx.context], &(partitions[0]), &(colors[0]), [#args])
    var ip = c.legion_terra_index_cross_product_get_partition([product])
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], lr.impl, ip)
  end

  return values.value(
    node,
    expr.once_only(
      actions,
      `(expr_type {
          impl = [lp],
          product = [product],
          colors = [colors],
        }),
      expr_type),
    expr_type)
end

function codegen.expr_cross_product_array(cx, node)
  local lhs = codegen.expr(cx, node.lhs):read(cx)
  local colorings = codegen.expr(cx, node.colorings):read(cx)
  local disjoint = node.disjointness:is(ast.disjointness_kind.Disjoint)

  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [lhs.actions];
    [colorings.actions];
    [emit_debuginfo(node)]
  end

  local lr = cx:region(expr_type:parent_region()).logical_region
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local product = terralib.newsymbol(
    c.legion_terra_index_cross_product_t, "cross_product")
  local colors = terralib.newsymbol(c.legion_color_t[2], "colors")
  actions = quote
    var [colors]
    var start : double = c.legion_get_current_time_in_micros()/double(1e6)
    var color_space =
      c.legion_index_partition_get_color_space(
        [cx.runtime], [lhs.value].impl.index_partition)
    var color_domain = 
      c.legion_index_space_get_domain([cx.runtime], color_space)
    regentlib.assert(color_domain.dim == 1, "color domain should be 1D")
    var start_color = color_domain.rect_data[0]
    var end_color = color_domain.rect_data[1]
    regentlib.assert(start_color >= 0 and end_color >= 0,
      "colors should be non-negative")
    var [lp] = [lhs.value].impl
    var lhs_ip = [lp].index_partition
    var [colors]
    [colors][0] = c.legion_index_partition_get_color(
      [cx.runtime], lhs_ip)
    var other_color = -1

    for color = start_color, end_color + 1 do
      var lhs_subregion = c.legion_logical_partition_get_logical_subregion_by_color(
        [cx.runtime], [lp], color)
      var lhs_subspace = c.legion_index_partition_get_index_subspace(
        [cx.runtime], lhs_ip, color)
      var rhs_ip = c.legion_index_partition_create_coloring(
        [cx.runtime], [cx.context], lhs_subspace, [colorings.value] [color],
        [disjoint], other_color)
      var rhs_lp = c.legion_logical_partition_create([cx.runtime], [cx.context],
        lhs_subregion, rhs_ip)
      other_color =
        c.legion_index_partition_get_color([cx.runtime], rhs_ip)
    end

    [colors][1] = other_color
    var [product]
    [product].partition = lhs_ip
    [product].other_color = other_color
    var stop : double = c.legion_get_current_time_in_micros()/double(1e6)
    c.printf("codegen: cross_product_array %e\n", stop - start)
  end

  return values.value(
    node,
    expr.once_only(
      actions,
      `(expr_type {
          impl = [lp],
          product = [product],
          colors = [colors],
        }),
      expr_type),
    expr_type)
end

-- Returns Terra expression for subregion of partition of specified color.
-- The color space can be either structured or unstructured.
local function get_partition_subregion(cx, partition, color)
  local partition_type = partition.value.type
  local part_color_type = partition_type:colors().index_type
  return `(c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
    [cx.runtime], [partition.value].impl, [part_color_type]([color]):to_domain_point()))
end

function codegen.expr_list_slice_partition(cx, node)
  local partition_type = std.as_read(node.partition.expr_type)
  local partition = codegen.expr(cx, node.partition):read(cx, partition_type)
  local indices_type = std.as_read(node.indices.expr_type)
  local indices = codegen.expr(cx, node.indices):read(cx, indices_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [partition.actions]
    [indices.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  local parent_region = partition_type:parent_region()
  assert(cx:has_region(parent_region))

  cx:add_list_of_regions(
    expr_type, result,
    cx:region(parent_region).field_paths,
    cx:region(parent_region).privilege_field_paths,
    cx:region(parent_region).field_privileges,
    cx:region(parent_region).field_types,
    cx:region(parent_region).field_ids,
    cx:region(parent_region).fields_are_scratch)

  actions = quote
    [actions]
    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) * [indices.value].__size)
    regentlib.assert(data ~= nil, "malloc failed in list_slice_partition")
    var [result] = expr_type {
      __size = [indices.value].__size,
      __data = data,
    }

    for i = 0, [indices.value].__size do
      var color = [indices_type:data(indices.value)][i]
      var r = [get_partition_subregion(cx, partition, color)]
      [expr_type:data(result)][i] = [expr_type.element_type] { impl = r }
    end
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_duplicate_partition(cx, node)
  local partition_type = std.as_read(node.partition.expr_type)
  local partition = codegen.expr(cx, node.partition):read(cx, partition_type)
  local indices_type = std.as_read(node.indices.expr_type)
  local indices = codegen.expr(cx, node.indices):read(cx, indices_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [partition.actions]
    [indices.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  local parent_region = partition_type:parent_region()
  assert(cx:has_region(parent_region))

  cx:add_list_of_regions(
    expr_type, result,
    cx:region(parent_region).field_paths,
    cx:region(parent_region).privilege_field_paths,
    cx:region(parent_region).field_privileges,
    cx:region(parent_region).field_types,
    cx:region(parent_region).field_ids,
    cx:region(parent_region).fields_are_scratch)

  actions = quote
    [actions]
    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) * [indices.value].__size)
    regentlib.assert(data ~= nil, "malloc failed in list_duplicate_partition")
    var [result] = expr_type {
      __size = [indices.value].__size,
      __data = data,
    }

    -- Grab the root region to copy semantic info.
    var root = c.legion_logical_partition_get_parent_logical_region(
      [cx.runtime], [partition.value].impl)
    while c.legion_logical_region_has_parent_logical_partition(
      [cx.runtime], root)
    do
      var part = c.legion_logical_region_get_parent_logical_partition(
        [cx.runtime], root)
      root = c.legion_logical_partition_get_parent_logical_region(
        [cx.runtime], part)
    end

    for i = 0, [indices.value].__size do
      var color = [indices_type:data(indices.value)][i]
      var orig_r = [get_partition_subregion(cx, partition, color)]
      var r = c.legion_logical_region_create(
        [cx.runtime], [cx.context], orig_r.index_space, orig_r.field_space, true)
      var new_root = c.legion_logical_partition_get_logical_subregion_by_tree(
        [cx.runtime], orig_r.index_space, orig_r.field_space, r.tree_id)

      -- Attach semantic info.
      var name : &int8
      c.legion_logical_region_retrieve_name([cx.runtime], root, &name)
      regentlib.assert(name ~= nil, "invalid name")
      c.legion_logical_region_attach_name([cx.runtime], new_root, name, false)

      [expr_type:data(result)][i] = [expr_type.element_type] { impl = r }
    end
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_slice_cross_product(cx, node)
  local product_type = std.as_read(node.product.expr_type)
  local product = codegen.expr(cx, node.product):read(cx, product_type)
  local indices_type = std.as_read(node.indices.expr_type)
  local indices = codegen.expr(cx, node.indices):read(cx, indices_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [product.actions]
    [indices.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  actions = quote
    [actions]
    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) * [indices.value].__size)
    regentlib.assert(data ~= nil, "malloc failed in list_slice_cross_product")
    var [result] = expr_type {
      __size = [indices.value].__size,
      __data = data,
    }
    for i = 0, [indices.value].__size do
      var color = c.legion_domain_point_from_point_1d(
        c.legion_point_1d_t { x = arrayof(c.coord_t, [indices_type:data(indices.value)][i]) })
      var ip = c.legion_terra_index_cross_product_get_subpartition_by_color_domain_point(
        [cx.runtime], [cx.context],
        [product.value].product, color)
      var lp = c.legion_logical_partition_create_by_tree(
        [cx.runtime], [cx.context], ip,
        [product.value].impl.field_space, [product.value].impl.tree_id)
      [expr_type:data(result)][i] = [expr_type.element_type] { impl = lp }
    end
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_cross_product(cx, node)
  local lhs_type = std.as_read(node.lhs.expr_type)
  local lhs = codegen.expr(cx, node.lhs):read(cx, lhs_type)
  local rhs_type = std.as_read(node.rhs.expr_type)
  local rhs = codegen.expr(cx, node.rhs):read(cx, rhs_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [lhs.actions]
    [rhs.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  assert(cx:has_list_of_regions(lhs_type))

  cx:add_list_of_regions(
    expr_type, result,
    cx:list_of_regions(lhs_type).field_paths,
    cx:list_of_regions(lhs_type).privilege_field_paths,
    cx:list_of_regions(lhs_type).field_privileges,
    cx:list_of_regions(lhs_type).field_types,
    cx:list_of_regions(lhs_type).field_ids,
    cx:list_of_regions(lhs_type).fields_are_scratch)

  local cross_product_create = c.legion_terra_index_cross_product_create_list
  if node.shallow then
    cross_product_create = c.legion_terra_index_cross_product_create_list_shallow
  end

  actions = quote
    [actions]

    var lhs_ : &c.legion_terra_logical_region_list_t =
      [&c.legion_terra_logical_region_list_t]([&opaque](&[lhs.value]))
    var rhs_ : &c.legion_terra_logical_region_list_t =
      [&c.legion_terra_logical_region_list_t]([&opaque](&[rhs.value]))

    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) * [lhs.value].__size)
    regentlib.assert(data ~= nil, "malloc failed in list_cross_product")
    var [result] = expr_type {
      __size = [lhs.value].__size,
      __data = data,
    }
    for i = 0, [lhs.value].__size do
      var subdata = c.malloc(
        terralib.sizeof([expr_type.element_type.element_type]) * [rhs.value].__size)
      regentlib.assert(subdata ~= nil, "malloc failed in list_cross_product")
      [expr_type:data(result)][i] = [expr_type.element_type] {
        __size = 0, -- will be set to a meaningful value inside the C library
        __data = subdata,
      }
    end

    var result_ : &c.legion_terra_logical_region_list_list_t =
      [&c.legion_terra_logical_region_list_list_t]([&opaque](&[result]))

    var product = [cross_product_create](
      [cx.runtime], [cx.context], lhs_, rhs_, result_)
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_cross_product_complete(cx, node)
  local lhs_type = std.as_read(node.lhs.expr_type)
  local lhs = codegen.expr(cx, node.lhs):read(cx, lhs_type)
  local product_type = std.as_read(node.product.expr_type)
  local product = codegen.expr(cx, node.product):read(cx, product_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [lhs.actions]
    [product.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")

  assert(cx:has_list_of_regions(lhs_type))

  cx:add_list_of_regions(
    expr_type, result,
    cx:list_of_regions(lhs_type).field_paths,
    cx:list_of_regions(lhs_type).privilege_field_paths,
    cx:list_of_regions(lhs_type).field_privileges,
    cx:list_of_regions(lhs_type).field_types,
    cx:list_of_regions(lhs_type).field_ids,
    cx:list_of_regions(lhs_type).fields_are_scratch)

  actions = quote
    [actions]

    regentlib.assert([product.value].__size == [lhs.value].__size,
                     "size mismatch in list_cross_product 1")

    var lhs_list : c.legion_terra_index_space_list_t
    lhs_list.space.tid = 0
    lhs_list.space.id = 0
    lhs_list.count = [lhs.value].__size
    lhs_list.subspaces = [&c.legion_index_space_t](
      c.malloc(
        terralib.sizeof([c.legion_index_space_t]) * [lhs.value].__size))
    regentlib.assert(lhs_list.subspaces ~= nil, "malloc failed in list_cross_product")
    for i = 0, [lhs.value].__size do
      lhs_list.subspaces[i] = [lhs_type:data(lhs.value)][i].impl.index_space
    end

    var shallow_product : c.legion_terra_index_space_list_list_t
    shallow_product.count = [lhs.value].__size
    shallow_product.sublists = [&c.legion_terra_index_space_list_t](
      c.malloc(
        terralib.sizeof([c.legion_terra_index_space_list_t]) * [lhs.value].__size))
    regentlib.assert(shallow_product.sublists ~= nil, "malloc failed in list_cross_product")
    for i = 0, [lhs.value].__size do
      var subsize = [product_type:data(product.value)][i].__size
      shallow_product.sublists[i].space = [lhs_type:data(lhs.value)][i].impl.index_space
      shallow_product.sublists[i].count = subsize
      shallow_product.sublists[i].subspaces = [&c.legion_index_space_t](
        c.malloc(terralib.sizeof([c.legion_index_space_t]) * subsize))
      for j = 0, subsize do
        shallow_product.sublists[i].subspaces[j] =
          [product_type.element_type:data(
             `([product_type:data(product.value)][i]))][j].impl.index_space
      end
    end

    var complete_product = c.legion_terra_index_cross_product_create_list_complete(
      [cx.runtime], [cx.context], [lhs_list], [shallow_product], false)
    regentlib.assert(complete_product.count == [lhs.value].__size,
                     "size mismatch in list_cross_product 2")

    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) * [lhs.value].__size)
    regentlib.assert(data ~= nil, "malloc failed in list_cross_product")
    var [result] = expr_type {
      __size = [lhs.value].__size,
      __data = data,
    }

    for i = 0, [lhs.value].__size do
      var subsize = [product_type:data(product.value)][i].__size
      regentlib.assert(
        subsize == complete_product.sublists[i].count,
        "size mismatch in list_cross_product 3")

      -- Allocate sublist.
      var subdata = c.malloc(
        terralib.sizeof([expr_type.element_type.element_type]) * subsize)
      regentlib.assert(subdata ~= nil, "malloc failed in list_cross_product")
      [expr_type:data(result)][i] = [expr_type.element_type] {
        __size = subsize,
        __data = subdata,
      }

      -- Fill sublist.
      for j = 0, subsize do
        [expr_type.element_type:data(`([expr_type:data(result)][i]))][j] =
          [expr_type.element_type.element_type] {
            impl = c.legion_logical_region_t {
              tree_id = [product_type.element_type:data(
                           `([product_type:data(product.value)][i]))][j].impl.tree_id,
              index_space = complete_product.sublists[i].subspaces[j],
              field_space = [product_type.element_type:data(
                               `([product_type:data(product.value)][i]))][j].impl.field_space,
            }
          }
      end
    end

    c.legion_terra_index_space_list_list_destroy(complete_product)
    c.free(lhs_list.subspaces)
    -- c.free(rhs_list.subspaces)
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

local gen_expr_list_phase_barriers = terralib.memoize(
  function(product_type, expr_type)
    local result = terralib.newsymbol(expr_type, "result")
    local product = terralib.newsymbol(product_type, "product")
    local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
    local context = terralib.newsymbol(c.legion_context_t, "context")
    local terra list_phase_barriers([runtime], [context], [product])
      var data = c.malloc(
        terralib.sizeof([expr_type.element_type]) * [product].__size)
      regentlib.assert(data ~= nil, "malloc failed in list_phase_barriers")
      var [result] = expr_type {
        __size = [product].__size,
        __data = data,
      }
      for i = 0, [product].__size do
        var subsize = [product_type:data(product)][i].__size

        -- Allocate sublist.
        var subdata = c.malloc(
          terralib.sizeof([expr_type.element_type.element_type]) * subsize)
        regentlib.assert(subdata ~= nil, "malloc failed in list_phase_barriers")
        [expr_type:data(result)][i] = [expr_type.element_type] {
          __size = subsize,
          __data = subdata,
        }

        -- Fill sublist.
        for j = 0, subsize do
          [expr_type.element_type:data(`([expr_type:data(result)][i]))][j] =
            [expr_type.element_type.element_type] {
              impl = c.legion_phase_barrier_create([runtime], [context], 1)
            }
        end
      end
      return [result]
    end
    list_phase_barriers:setinlined(false)
    return list_phase_barriers
  end)

function codegen.expr_list_phase_barriers(cx, node)
  local product_type = std.as_read(node.product.expr_type)
  local product = codegen.expr(cx, node.product):read(cx, product_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [product.actions]
    [emit_debuginfo(node)]
  end

  local result = terralib.newsymbol(expr_type, "result")
  local helper = gen_expr_list_phase_barriers(product_type, expr_type)

  actions = quote
    [actions]
    var [result] = [helper]([cx.runtime], [cx.context], [product.value])
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

-- Returns Terra expression for color as an index_type (e.g. int2d).
local function get_logical_region_color(runtime, region, color_type)
  assert(std.is_index_type(color_type))
  local domain_pt_expr =
    `(c.legion_logical_region_get_color_domain_point([runtime], region))
  return `([color_type.from_domain_point](domain_pt_expr))
end

-- Returns Terra expression for offset of `point` in a rectangle.
local function get_offset_in_rect(point, rect_size)
  local point_type = point.type
  local rect_size_type = rect_size.type
  assert(std.is_index_type(point_type))
  assert(std.type_eq(point_type, rect_size_type))

  local fields = point_type.fields
  if fields then
    local offset = `0
    local multiplicand = `1
    for _, field in ipairs(fields) do
      offset = `([offset] + [point].__ptr.[field] * [multiplicand])
      multiplicand = `([multiplicand] * [rect_size].__ptr.[field])
    end
    return offset
  else
    return `([point].__ptr)
  end
end

local gen_expr_list_invert = terralib.memoize(
  function(rhs_type, product_type, expr_type, barriers_type)
    local color_type = rhs_type:partition():colors().index_type
    if color_type:is_opaque() then  -- Treat `ptr` as `int1d`.
      color_type = int1d
    end
    local result = terralib.newsymbol(expr_type, "result")
    local rhs = terralib.newsymbol(rhs_type, "rhs")
    local product = terralib.newsymbol(product_type, "product")
    local barriers = terralib.newsymbol(barriers_type, "barriers")
    local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
    local terra list_invert([runtime], [rhs], [product], [barriers])
      -- 1. Compute an index from colors to rhs index.
      -- 2. Compute sublist sizes.
      -- 3. Allocate sublists.
      -- 3. Fill sublists.

      var data = c.malloc(
        terralib.sizeof([expr_type.element_type]) * [rhs].__size)
      regentlib.assert(data ~= nil, "malloc failed in list_invert")
      var [result] = expr_type {
        __size = [rhs].__size,
        __data = data,
      }

      if [rhs].__size > 0 then
        -- 1. Compute an index from colors to rhs index.

        -- Determine the range of colors in rhs.
        var min_color : color_type
        var max_color : color_type
        for i = 0, [rhs].__size do
          var rhs_elt = [rhs_type:data(rhs)][i].impl
          var rhs_color = [get_logical_region_color(runtime, rhs_elt, color_type)]
          if i == 0 then
            min_color, max_color = rhs_color, rhs_color
          else
            min_color, max_color =
              [color_type.elem_min](min_color, rhs_color),
              [color_type.elem_max](max_color, rhs_color)
          end
        end
        var colors_rect = [std.rect_type(color_type)] { lo = min_color, hi = max_color }
        var colors_size = colors_rect:size()
        var num_colors = colors_rect:volume()
        regentlib.assert(num_colors >= 0, "invalid color range in list_invert")

        -- Build the index.
        var color_to_index = [&int64](c.malloc(
          terralib.sizeof(int64) * num_colors))
        regentlib.assert(color_to_index ~= nil, "malloc failed in list_invert")

        for i = 0, num_colors do
          color_to_index[i] = -1
        end

        for i = 0, [rhs].__size do
          var rhs_elt = [rhs_type:data(rhs)][i].impl
          var rhs_color = [get_logical_region_color(runtime, rhs_elt, color_type)]

          var delta = rhs_color - min_color
          var color_index = [get_offset_in_rect(delta, colors_size)]
          regentlib.assert(
            (0 <= color_index) and (color_index < num_colors),
            "color index out of bounds in list_invert")
          regentlib.assert(
            color_to_index[color_index] == -1,
            "duplicate colors in list_invert")
          color_to_index[color_index] = i
        end

        -- 2. Compute sublists sizes.
        for i = 0, [rhs].__size do
          [expr_type:data(result)][i].__size = 0
        end

        for j = 0, [product].__size do
          for k = 0, [product_type:data(product)][j].__size do
            var leaf = [product_type.element_type:data(
                          `([product_type:data(product)][j]))][k].impl
            var inner = leaf

            if not [product_type.shallow] then
              regentlib.assert(
                c.legion_logical_region_has_parent_logical_partition(
                  [runtime], leaf),
                "missing color in list_invert")
              var part = c.legion_logical_region_get_parent_logical_partition(
                [runtime], leaf)
              var parent = c.legion_logical_partition_get_parent_logical_region(
                [runtime], part)
              inner = parent
            end

            var inner_color = [get_logical_region_color(runtime, inner, color_type)]
            var delta = inner_color - min_color
            var i = color_to_index[ [get_offset_in_rect(delta, colors_size)] ]
            [expr_type:data(result)][i].__size =
              [expr_type:data(result)][i].__size + 1
          end
        end

        -- 3. Allocate sublists.
        for i = 0, [rhs].__size do
          var subsize = [expr_type:data(result)][i].__size
          var subdata = [&expr_type.element_type.element_type](c.malloc(
            terralib.sizeof([expr_type.element_type.element_type]) * subsize))
          regentlib.assert(subdata ~= nil, "malloc failed in list_invert")
          [expr_type:data(result)][i].__data = subdata
        end

        -- 4. Fill sublists.

        -- Create a list to hold the next index.
        var subslots = [&int64](c.malloc(
          terralib.sizeof([int64]) * [rhs].__size))
        for i = 0, [rhs].__size do
          subslots[i] = 0
        end

        for j = 0, [product].__size do
          for k = 0, [product_type:data(product)][j].__size do
            var leaf = [product_type.element_type:data(
                          `([product_type:data(product)][j]))][k].impl
            var inner = leaf

            if not [product_type.shallow] then
              regentlib.assert(
                c.legion_logical_region_has_parent_logical_partition(
                  [runtime], leaf),
                "missing parent in list_invert")
              var part = c.legion_logical_region_get_parent_logical_partition(
                [runtime], leaf)
              var parent = c.legion_logical_partition_get_parent_logical_region(
                [runtime], part)
              inner = parent
            end

            var inner_color = [get_logical_region_color(runtime, inner, color_type)]
            var delta = inner_color - min_color
            var i = color_to_index[ [get_offset_in_rect(delta, colors_size)] ]
            regentlib.assert(subslots[i] < [expr_type:data(result)][i].__size,
                             "overflowed sublist in list_invert")
            [expr_type.element_type:data(`([expr_type:data(result)][i]))][subslots[i]] =
                [barriers_type.element_type:data(
                   `([barriers_type:data(barriers)][j]))][k]
            subslots[i] = subslots[i] + 1
          end
        end

        for i = 0, [rhs].__size do
          regentlib.assert(subslots[i] == [expr_type:data(result)][i].__size, "underflowed sublist in list_invert")
        end

        c.free(subslots)
        c.free(color_to_index)
      end
      return [result]
    end
    list_invert:setinlined(false)
    return list_invert
  end)

function codegen.expr_list_invert(cx, node)
  local rhs_type = std.as_read(node.rhs.expr_type)
  local rhs = codegen.expr(cx, node.rhs):read(cx, rhs_type)
  local product_type = std.as_read(node.product.expr_type)
  local product = codegen.expr(cx, node.product):read(cx, product_type)
  local barriers_type = std.as_read(node.barriers.expr_type)
  local barriers = codegen.expr(cx, node.barriers):read(cx, barriers_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [rhs.actions]
    [product.actions]
    [barriers.actions]
    [emit_debuginfo(node)]
  end

  local color_type = rhs_type:partition():colors().index_type
  if color_type:is_opaque() then  -- Treat `ptr` as `int1d`.
    color_type = int1d
  end

  local result = terralib.newsymbol(expr_type, "result")
  local helper = gen_expr_list_invert(rhs_type, product_type, expr_type, barriers_type)
  actions = quote
    [actions]
    var [result] = [helper]([cx.runtime], [rhs.value], [product.value], [barriers.value])
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_range(cx, node)
  local start_type = std.as_read(node.start.expr_type)
  local start = codegen.expr(cx, node.start):read(cx, start_type)
  local stop_type = std.as_read(node.stop.expr_type)
  local stop = codegen.expr(cx, node.stop):read(cx, stop_type)
  local expr_type = std.as_read(node.expr_type)

  local result = terralib.newsymbol(expr_type, "result")
  local actions = quote
    [start.actions]
    [stop.actions]
    [emit_debuginfo(node)]

    regentlib.assert([stop.value] >= [start.value], "negative size range in list_range")
    var data = c.malloc(
      terralib.sizeof([expr_type.element_type]) *
        ([stop.value] - [start.value]))
    regentlib.assert(data ~= nil, "malloc failed in list_range")
    var [result] = expr_type {
      __size = [stop.value] - [start.value],
      __data = data
    }
    for i = [start.value], [stop.value] do
      [expr_type:data(result)][i - [start.value] ] = i
    end
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_list_ispace(cx, node)
  local ispace_type = std.as_read(node.ispace.expr_type)
  local ispace = codegen.expr(cx, node.ispace):read(cx, ispace_type)
  local expr_type = std.as_read(node.expr_type)

  local result = terralib.newsymbol(expr_type, "result") -- Resulting list.
  local result_len = terralib.newsymbol(uint64, "result_len")

  -- Construct AST that populates the `result` list with indices from `ispace`.
  --
  -- Loop variable: `for p in ispace do`
  local bound
  if node.ispace:is(ast.typed.expr.ID) then
    bound = node.ispace.value
  else
    bound = std.newsymbol(ispace_type)
  end
  local p_symbol = std.newsymbol(
    ispace_type.index_type(bound),
    "p")
  -- Index in list: `result[i] = p; i += 1`.
  local i = terralib.newsymbol(int, "i")

  local loop_body = ast.typed.stat.Internal {
    actions = quote
      regentlib.assert((i >= 0) and (i < [result_len]), "list index out of bounds in list_ispace")
      [expr_type:data(result)][ [i] ] = [p_symbol:getsymbol()];
      [i] = [i] + 1
    end,
    annotations = ast.default_annotations(),
    span = node.span,
  }
  local populate_list_loop = ast.typed.stat.ForList {
    symbol = p_symbol,
    value = node.ispace,
    block = ast.typed.Block {
      stats = terralib.newlist({loop_body}),
      span = node.span,
    },
    annotations = ast.default_annotations(),
    span = node.span,
  }

  local actions = quote
    [ispace.actions]
    [emit_debuginfo(node)]

    -- Compute size of resulting list.
    -- Currently doesn't support index spaces with multiple domains.
    regentlib.assert(not c.legion_index_space_has_multiple_domains([cx.runtime], [ispace.value].impl),
      "list_ispace doesn't support index spaces with multiple domains")
    var ispace_domain = c.legion_index_space_get_domain([cx.runtime], [ispace.value].impl)
    var [result_len] = c.legion_domain_get_volume(ispace_domain)

    -- Allocate list.
    var data = c.malloc(terralib.sizeof([expr_type.element_type]) * [result_len])
    regentlib.assert(data ~= nil, "malloc failed in list_ispace")
    var [result] = expr_type {
      __size = [result_len],
      __data = data
    }

    -- Populate list with indices from index space.
    var [i] = 0;
    [codegen.stat(cx, populate_list_loop)]
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

local function gen_expr_list_from_element(expr_type, result, list, value)
  if not std.is_list(expr_type.element_type) then
    return quote
      var len = [list].__size
      var data = c.malloc(terralib.sizeof([expr_type.element_type]) * len)
      [result] = expr_type {
        __size = len,
        __data = data,
      }
      for i = 0, len do
        [expr_type:data(result)][i] = [value]
      end
    end
  else
    return quote
      var len = [list].__size
      var data = c.malloc(terralib.sizeof([expr_type.element_type]) * len)
      [result] = expr_type {
        __size = len,
        __data = data,
      }
      for i = 0, len do
        [gen_expr_list_from_element(
            expr_type.element_type,
            `([expr_type:data(result)][i]),
            `([expr_type:data(list)][i]),
            value)]
      end
    end
  end
end

function codegen.expr_list_from_element(cx, node)
  local list_type = std.as_read(node.list.expr_type)
  local list = codegen.expr(cx, node.list):read(cx, list_type)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)

  local expr_type = std.as_read(node.expr_type)

  local result = terralib.newsymbol(expr_type, "result")
  local result_len = terralib.newsymbol(uint64, "result_len")

  local actions = quote
    [list.actions]
    [value.actions]
    [emit_debuginfo(node)]
    var [result]
    [gen_expr_list_from_element(expr_type, result, list.value, value.value)]
  end

  return values.value(
    node,
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_phase_barrier(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  return values.value(
    node,
    expr.once_only(
      actions,
      `(expr_type {
          impl = c.legion_phase_barrier_create(
            [cx.runtime], [cx.context], [value.value]),
        }),
      expr_type),
    expr_type)
end

function codegen.expr_dynamic_collective(cx, node)
  local arrivals_type = std.as_read(node.arrivals.expr_type)
  local arrivals = codegen.expr(cx, node.arrivals):read(cx, arrivals_type)
  local value_type = node.value_type
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [arrivals.actions];
    [emit_debuginfo(node)]
  end

  local redop = std.reduction_op_ids[node.op][value_type]
  local init_value = std.reduction_op_init[node.op][value_type]
  assert(redop and init_value)

  local init = terralib.newsymbol(value_type, "init")
  actions = quote
    [actions]
    var [init] = [init_value]
  end

  return values.value(
    node,
    expr.once_only(
      actions,
      `(expr_type {
          impl = c.legion_dynamic_collective_create(
            [cx.runtime], [cx.context], [arrivals.value],
            [redop], &[init], [terralib.sizeof(value_type)]),
        }),
      expr_type),
    expr_type)
end

function codegen.expr_dynamic_collective_get_result(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  local future_type = expr_type
  if not std.is_future(expr_type) then
    future_type = std.future(expr_type)
  end

  local future_value = values.value(
    node,
    expr.once_only(
      actions,
      `(future_type {
          __result = c.legion_dynamic_collective_get_result(
            [cx.runtime], [cx.context], [value.value].impl),
        }),
      future_type),
    future_type)

  if std.is_future(expr_type) then
    return future_value
  else
    return codegen.expr(
      cx,
      ast.typed.expr.FutureGetResult {
        value = ast.typed.expr.Internal {
          value = future_value,
          expr_type = future_type,
          annotations = node.annotations,
          span = node.span,
        },
        expr_type = expr_type,
        annotations = node.annotations,
        span = node.span,
    })
  end
end

local function expr_advance_phase_barrier(runtime, context, value, value_type)
  if std.is_phase_barrier(value_type) then
    return quote end, `(value_type {
      impl = c.legion_phase_barrier_advance(
        [runtime], [context], [value].impl),
    })
  elseif std.is_dynamic_collective(value_type) then
    return quote end, `(value_type {
      impl = c.legion_dynamic_collective_advance(
        [runtime], [context], [value].impl),
    })
  else
    assert(false)
  end
end

local function expr_advance_list_body(runtime, context, value, value_type)
  if std.is_list(value_type) then
    local result = terralib.newsymbol(value_type, "result")
    local element = terralib.newsymbol(value_type.element_type, "element")
    local inner_actions, inner_value = expr_advance_list_body(
      runtime, context, element, value_type.element_type)
    local actions = quote
      var data = c.malloc(
        terralib.sizeof([value_type.element_type]) * [value].__size)
      regentlib.assert(data ~= nil, "malloc failed in index_access")
      var [result] = [value_type] {
        __size = [value].__size,
        __data = data
      }
      for i = 0, [value].__size do
        var [element] = [value_type:data(value)][i]
        [inner_actions]
        [value_type:data(result)][i] = [inner_value]
      end
    end
    return actions, result
  else
    return expr_advance_phase_barrier(runtime, context, value, value_type)
  end
end

local expr_advance_list_helper = terralib.memoize(
  function (value_type)
    local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
    local context = terralib.newsymbol(c.legion_context_t, "context")
    local value = terralib.newsymbol(value_type, "value")
    local result_actions, result =
      expr_advance_list_body(runtime, context, value, value_type)
    local terra advance_barriers([runtime], [context], [value])
      [result_actions]
      return [result]
    end
    advance_barriers:setinlined(false)
    return advance_barriers
  end)

function expr_advance_list(cx, value, value_type)
  if std.is_list(value_type) then
    local helper = expr_advance_list_helper(value_type)
    local result = terralib.newsymbol(value_type, "result")
    local actions = quote
      var [result] = [helper]([cx.runtime], [cx.context], [value])
    end
    return actions, result
  else
    return expr_advance_phase_barrier(cx.runtime, cx.context, value, value_type)
  end
end

function codegen.expr_advance(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  local result_actions, result = expr_advance_list(cx, value.value, value_type)
  actions = quote
    [actions];
    [result_actions]
  end

  return values.value(
    node,
    expr.once_only(actions, result, expr_type),
    expr_type)
end

local function expr_adjust_phase_barrier(cx, barrier, barrier_type,
                                         value, value_type)
  local result = terralib.newsymbol(barrier_type, "result")
  if std.is_phase_barrier(barrier_type) then
    return quote
      var [result] = std.phase_barrier {
        impl = c.legion_phase_barrier_alter_arrival_count(
          [cx.runtime], [cx.context], [barrier].impl, [value])
      }
    end, result
  elseif std.is_dynamic_collective(barrier_type) then
    return quote
      var [result] = std.phase_barrier {
        impl = c.legion_dynamic_collective_alter_arrival_count(
          [cx.runtime], [cx.context], [barrier].impl, [value])
      }
    end, result
  else
    assert(false)
  end
end

local function expr_adjust_list(cx, barrier, barrier_type, value, value_type)
  if std.is_list(barrier_type) then
    local result = terralib.newsymbol(barrier_type, "result")
    local element = terralib.newsymbol(barrier_type.element_type, "element")
    local inner_actions, inner_result = expr_adjust_list(
      cx, element, barrier_type.element_type, value, value_type)
    local actions = quote
      var data = c.malloc(
        terralib.sizeof([barrier_type.element_type]) * [barrier].__size)
      regentlib.assert(data ~= nil, "malloc failed in adjust")
      var [result] = barrier_type {
        __size = [barrier].__size,
        __data = data,
      }

      for i = 0, [barrier].__size do
        var [element] = [barrier_type:data(barrier)][i]
        [inner_actions]
        [barrier_type:data(result)][i] = [inner_result]
      end
    end
    return actions, result
  else
    return expr_adjust_phase_barrier(
      cx, barrier, barrier_type, value, value_type)
  end
end

function codegen.expr_adjust(cx, node)
  local barrier_type = std.as_read(node.barrier.expr_type)
  local barrier = codegen.expr(cx, node.barrier):read(cx, barrier_type)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [barrier.actions];
    [value.actions];
    [emit_debuginfo(node)]
  end

  local result_actions = expr_adjust_list(
    cx, barrier.value, barrier_type, value.value, value_type)
  actions = quote
    [actions];
    [result_actions]
  end

  return values.value(
    node,
    expr.just(actions, barrier.value),
    expr_type)
end

local function gen_expr_arrive(expr_type, cx, value)
  if std.is_phase_barrier(expr_type) then
    return quote
      c.legion_phase_barrier_arrive(
        [cx.runtime], [cx.context], value.impl, 1)
    end
  else
    assert(std.is_list_of_phase_barriers(expr_type))
    return quote
      for i = 0, [value].__size do
        var l = [expr_type:data(value)][i]
        [gen_expr_arrive(expr_type.element_type, cx, `([expr_type:data(value)][i]))]
      end
    end
  end
end

function codegen.expr_arrive(cx, node)
  local barrier_type = std.as_read(node.barrier.expr_type)
  local barrier = codegen.expr(cx, node.barrier):read(cx, barrier_type)
  local value_type = node.value and std.as_read(node.value.expr_type)
  local value = node.value and codegen.expr(cx, node.value):read(cx, value_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [barrier.actions];
    [value and value.actions];
    [emit_debuginfo(node)]
  end

  if std.is_phase_barrier(barrier_type) or std.is_list_of_phase_barriers(barrier_type) then
    actions = quote
      [actions]
      [gen_expr_arrive(barrier_type, cx, barrier.value)]
    end
  elseif std.is_dynamic_collective(barrier_type) then
    if std.is_future(value_type) then
      actions = quote
        [actions]
        c.legion_dynamic_collective_defer_arrival(
          [cx.runtime], [cx.context], [barrier.value].impl,
          [value.value].__result, 1)
      end
    else
      actions = quote
        [actions]
        var buffer : value_type = [value.value]
        c.legion_dynamic_collective_arrive(
          [cx.runtime], [cx.context], [barrier.value].impl,
          &buffer, [terralib.sizeof(value_type)], 1)
      end
    end
  else
    assert(false)
  end

  return values.value(
    node,
    expr.just(actions, barrier.value),
    expr_type)
end

local function gen_expr_await(expr_type, cx, value)
  if std.is_phase_barrier(expr_type) then
    return quote
      c.legion_phase_barrier_wait(
        [cx.runtime], [cx.context], [value].impl)
    end
  else
    assert(std.is_list_of_phase_barriers(expr_type))
    return quote
      for i = 0, [value].__size do
        var l = [expr_type:data(value)][i]
        [gen_expr_await(expr_type.element_type, cx, `([expr_type:data(value)][i]))]
      end
    end
  end
end

function codegen.expr_await(cx, node)
  local barrier_type = std.as_read(node.barrier.expr_type)
  local barrier = codegen.expr(cx, node.barrier):read(cx, barrier_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [barrier.actions];
    [emit_debuginfo(node)]
  end

  if std.is_phase_barrier(barrier_type) or std.is_list_of_phase_barriers(barrier_type) then
    actions = quote
      [actions]
      [gen_expr_await(barrier_type, cx, barrier.value)]
    end
  else
    assert(false)
  end

  return values.value(
    node,
    expr.just(actions, barrier.value),
    expr_type)
end

local function get_container_root(cx, value, container_type, field_paths)
  if std.is_region(container_type) then
    assert(cx:has_region(container_type))
    local root = cx:region(
      cx:region(container_type).root_region_type).logical_region
    return `([root].impl)
  elseif std.is_list_of_regions(container_type) then
    return raise_privilege_depth(cx, value, container_type, field_paths)
  else
    assert(false)
  end
end

local function expand_phase_barriers(value, value_type, thunk)
  if std.is_list(value_type) then
    local element_type = value_type.element_type
    local element = terralib.newsymbol(element_type, "barrier")
    local index = terralib.newsymbol(uint64, "index")
    return quote
      for [index] = 0, [value].__size do
        var [element] = [value_type:data(value)][ [index] ]
        [expand_phase_barriers(element, element_type, thunk)]
      end
    end
  else
    return thunk(value)
  end
end

local function map_phase_barriers(value, value_type, fn)
  if std.is_list(value_type) then
    local result = terralib.newsymbol(value_type, "map_" .. tostring(value))
    local element_type = value_type.element_type
    local element = terralib.newsymbol(element_type, "barrier")
    local index = terralib.newsymbol(uint64, "index")

    local inner_actions, inner_result = map_phase_barriers(
      element, element_type, fn)

    local actions = quote
      var data = c.malloc(
        terralib.sizeof([value_type.element_type]) * [value].__size)
      regentlib.assert(data ~= nil, "malloc failed in copy")
      var [result] = value_type {
        __size = [value].__size,
        __data = data,
      }

      for [index] = 0, [value].__size do
        var [element] = [value_type:data(value)][ [index] ]
        [inner_actions]
        [value_type:data(result)][ [index] ] = [inner_result]
      end
    end
    return actions, result
  else
    return fn(value, value_type)
  end
end

local function expr_copy_issue_phase_barriers(values, types, condition_kinds,
                                              launcher)
  local actions = terralib.newlist()
  for i, value in ipairs(values) do
    local value_type = types[i]
    local conditions = condition_kinds[i]
    for _, condition_kind in ipairs(conditions) do
      local add_barrier
      if condition_kind == std.awaits then
        add_barrier = c.legion_copy_launcher_add_wait_barrier
      elseif condition_kind == std.arrives then
        add_barrier = c.legion_copy_launcher_add_arrival_barrier
      else
        assert(false)
      end

      actions:insert(
        expand_phase_barriers(
          value, value_type,
          function(value)
            return quote
              [add_barrier]([launcher], [value].impl)
            end
          end))
    end
  end
  return actions
end

local function expr_copy_adjust_phase_barriers(cx, values, value_types, condition_kinds,
                                               need_adjust, count)
  local result_actions, results = terralib.newlist(), terralib.newlist()
  for i, value in ipairs(values) do
    local value_type = value_types[i]
    local arrives = data.filter(
      function(kind) return kind == std.arrives end, condition_kinds[i])
    assert(#arrives <= 1)
    if #arrives > 0 and need_adjust[i] then
      local actions, result = map_phase_barriers(
        value, value_type,
        function(value, value_type)
          local result = terralib.newsymbol(value_type, "result")
          local actions = quote
            var [result] = [value]
            -- The barrier is expecting 1 arrival. We're going to be
            -- arriving with [count]. Adjust the expect arrivals to match.
            var adjust = [int64]([count]) - 1
            if adjust > 0 then
              -- Extra arrivals, increase count.
              [result] = std.phase_barrier {
                impl = c.legion_phase_barrier_alter_arrival_count(
                  [cx.runtime], [cx.context], [value].impl, adjust)
              }
            elseif adjust < 0 then
              -- Fewer arrivals, decrement count.
              c.legion_phase_barrier_arrive(
                [cx.runtime], [cx.context], [value].impl, -adjust)
            end
          end
          return actions, result
        end)
      result_actions:insert(actions)
      results:insert(result)
    else
      results:insert(value)
    end
  end
  return result_actions, results
end

local function expr_copy_extract_phase_barriers(index, values, value_types, depth)
  local actions = terralib.newlist()
  local result_values = terralib.newlist()
  local result_types = terralib.newlist()
  local stopped_expansion = terralib.newlist()
  for i, value in ipairs(values) do
    local value_type = value_types[i]

    -- Record stop condition only once, at the exact limit of depth.
    local stop = std.is_list(value_type) and
      value_type.barrier_depth and depth == value_type.barrier_depth

    local result_type, result_value
    if std.is_list(value_type) and
      (not value_type.barrier_depth or depth <= value_type.barrier_depth)
    then
      result_type = value_type.element_type
      result_value = terralib.newsymbol(result_type, "extract_" .. tostring(value))
      actions:insert(quote
          std.assert([index] < [value].__size, "barrier index out of bounds in copy")
          var [result_value] = [value_type:data(value)][ [index] ]
      end)
    else
      result_type = value_type
      result_value = value
    end
    result_values:insert(result_value)
    result_types:insert(result_type)
    stopped_expansion:insert(stop)
  end
  return actions, result_values, result_types, stopped_expansion
end

local function expr_copy_setup_region(
    cx, src_value, src_type, src_container_type, src_fields,
    dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds,
    depth, op, launcher)
  assert(std.is_region(src_type) and std.is_region(dst_type))
  assert(std.type_supports_privileges(src_container_type) and
           std.type_supports_privileges(dst_container_type))
  assert(data.all(condition_types:map(
                    function(condition_type)
                      return std.is_phase_barrier(condition_type)
                    end)))

  local add_src_region =
    c.legion_copy_launcher_add_src_region_requirement_logical_region
  local add_dst_region =
    c.legion_copy_launcher_add_dst_region_requirement_logical_region
  if op then
    add_dst_region =
      c.legion_copy_launcher_add_dst_region_requirement_logical_region_reduction
  end

  local src_all_fields = std.flatten_struct_fields(src_type:fspace())
  local dst_all_fields = std.flatten_struct_fields(dst_type:fspace())
  local actions = terralib.newlist()

  local launcher = terralib.newsymbol(c.legion_copy_launcher_t, "launcher")
  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  actions:insert(quote
    var [tag] = 0
    [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
    var [launcher] = c.legion_copy_launcher_create(
      c.legion_predicate_true(), 0, [tag])
  end)

  local region_src_i, region_dst_i
  for i, src_field in ipairs(src_fields) do
    local dst_field = dst_fields[i]
    local src_copy_fields = data.filter(
      function(field) return field:starts_with(src_field) end,
      src_all_fields)
    local dst_copy_fields = data.filter(
      function(field) return field:starts_with(dst_field) end,
      dst_all_fields)
    assert(#src_copy_fields == #dst_copy_fields)

    local src_parent = get_container_root(
      cx, `([src_value].impl), src_container_type, src_copy_fields)
    local dst_parent = get_container_root(
      cx, `([dst_value].impl), dst_container_type, dst_copy_fields)

    local scratch = any_fields_are_scratch(cx, src_container_type, src_copy_fields) or
      any_fields_are_scratch(cx, dst_container_type, dst_copy_fields)

    for j, src_copy_field in ipairs(src_copy_fields) do
      local dst_copy_field = dst_copy_fields[j]
      local src_field_id = cx:region_or_list(src_container_type):field_id(src_copy_field)
      local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
      local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

      local dst_mode = c.READ_WRITE
      if op then
        dst_mode = std.reduction_op_ids[op][dst_field_type]
      end

      local src_i, dst_i
      local add_new_requirement = false
      if scratch or op then
        src_i = terralib.newsymbol(uint, "src_i")
        dst_i = terralib.newsymbol(uint, "dst_i")
        add_new_requirement = true
      else
        if not region_src_i then
          region_src_i = terralib.newsymbol(uint, "region_src_i")
          region_dst_i = terralib.newsymbol(uint, "region_dst_i")
          add_new_requirement = true
        end
        src_i = region_src_i
        dst_i = region_dst_i
      end

      if add_new_requirement then
        actions:insert(quote
          var [src_i] = add_src_region(
            [launcher], [src_value].impl, c.READ_ONLY, c.EXCLUSIVE,
            [src_parent], 0, false)
          var [dst_i] = add_dst_region(
            [launcher], [dst_value].impl, dst_mode, c.EXCLUSIVE,
            [dst_parent], 0, false)
        end)
      end

      actions:insert(quote
        c.legion_copy_launcher_add_src_field(
          [launcher], [src_i], [src_field_id], true)
        c.legion_copy_launcher_add_dst_field(
          [launcher], [dst_i], [dst_field_id], true)
      end)
    end
  end
  actions:insertall(
    expr_copy_issue_phase_barriers(condition_values, condition_types, condition_kinds, launcher))
  actions:insert(quote
    c.legion_copy_launcher_execute([cx.runtime], [cx.context], [launcher])
  end)
  return actions
end

local function count_nested_list_size(value, value_type)
  if std.is_list(value_type) then
    local element_type = value_type.element_type
    local element = terralib.newsymbol(element_type, "element")
    local index = terralib.newsymbol(uint64, "index")
    local count = terralib.newsymbol(int64, "count")

    local nested_count, nested_actions = count_nested_list_size(
      element, element_type)

    local actions = quote
      var [count] = 0
      for [index] = 0, [value].__size do
        var [element] = [value_type:data(value)][ [index] ]
        [nested_actions]
        [count] = [count] + [nested_count]
      end
    end

    return count, terralib.newlist({actions})
  else
    return 1, terralib.newlist()
  end
end

local function expr_copy_setup_list_one_to_many(
    cx, src_value, src_type, src_container_type, src_fields,
    dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds,
    depth, op, launcher)
  assert(std.is_region(src_type))
  if std.is_list(dst_type) then
    local dst_element_type = dst_type.element_type
    local dst_element = terralib.newsymbol(dst_element_type, "dst_element")
    local index = terralib.newsymbol(uint64, "index")
    local c_actions, c_values, c_types, c_stopped = expr_copy_extract_phase_barriers(
      index, condition_values, condition_types, depth)
    local update_actions = terralib.newlist()
    if data.any(unpack(c_stopped)) then
      local count, count_actions = count_nested_list_size(
        dst_element, dst_element_type)
      update_actions:insertall(count_actions)

      local adjust_actions, adjust_values = expr_copy_adjust_phase_barriers(
          cx, c_values, c_types, condition_kinds, c_stopped, count)
      update_actions:insertall(adjust_actions)
      c_values = adjust_values
    end
    return quote
      for [index] = 0, [dst_value].__size do
        var [dst_element] = [dst_type:data(dst_value)][ [index] ]
        [c_actions]
        [update_actions]
        [expr_copy_setup_region(
           cx, src_value, src_type, src_container_type, src_fields,
           dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds,
           depth + 1, op, launcher)]
      end
    end
  else
    return expr_copy_setup_region(
      cx, src_value, src_type, src_container_type, src_fields,
      dst_value, dst_type, dst_container_type, dst_fields,
      condition_values, condition_types, condition_kinds,
      depth, op, launcher)
  end
end

local function expr_copy_setup_list_one_to_one(
    cx, src_value, src_type, src_container_type, src_fields,
    dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds,
    depth, op, launcher)
  if std.is_list(src_type) then
    local src_element_type = src_type.element_type
    local src_element = terralib.newsymbol(src_element_type, "src_element")
    local dst_element_type = dst_type.element_type
    local dst_element = terralib.newsymbol(dst_element_type, "dst_element")
    local index = terralib.newsymbol(uint64, "index")
    local c_actions, c_values, c_types, c_stopped = expr_copy_extract_phase_barriers(
      index, condition_values, condition_types, depth)
    local update_actions = terralib.newlist()
    if data.any(unpack(c_stopped)) then
      local count, count_actions = count_nested_list_size(
        dst_element, dst_element_type)
      update_actions:insertall(count_actions)

      local adjust_actions, adjust_values = expr_copy_adjust_phase_barriers(
          cx, c_values, c_types, condition_kinds, c_stopped, count)
      update_actions:insertall(adjust_actions)
      c_values = adjust_values
    end
    return quote
      std.assert([src_value].__size == [dst_value].__size, "mismatch in number of regions to copy")
      for [index] = 0, [src_value].__size do
        var [src_element] = [src_type:data(src_value)][ [index] ]
        var [dst_element] = [dst_type:data(dst_value)][ [index] ]
        [c_actions]
        [update_actions]
        [expr_copy_setup_list_one_to_one(
           cx, src_element, src_element_type, src_container_type, src_fields,
           dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds,
           depth + 1, op, launcher)]
      end
    end
  else
    return expr_copy_setup_list_one_to_many(
      cx, src_value, src_type, src_container_type, src_fields,
      dst_value, dst_type, dst_container_type, dst_fields,
      condition_values, condition_types, condition_kinds,
      depth, op, launcher)
  end
end

function codegen.expr_copy(cx, node)
  local src_type = std.as_read(node.src.expr_type)
  local src = codegen.expr_region_root(cx, node.src):read(cx, src_type)
  local dst_type = std.as_read(node.dst.expr_type)
  local dst = codegen.expr_region_root(cx, node.dst):read(cx, dst_type)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)

  local actions = terralib.newlist()
  actions:insert(src.actions)
  actions:insert(dst.actions)
  actions:insertall(
    conditions:map(function(condition) return condition.actions end))
  actions:insert(
    quote
      [emit_debuginfo(node)]
    end)

  actions:insert(
    quote
      [expr_copy_setup_list_one_to_one(
         cx, src.value, src_type, src_type, node.src.fields,
         dst.value, dst_type, dst_type, node.dst.fields,
         conditions:map(function(condition) return condition.value end),
         node.conditions:map(
           function(condition)
             return std.as_read(condition.value.expr_type)
         end),
         node.conditions:map(function(condition) return condition.conditions end),
         1, node.op, launcher)]
    end)

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

local function expr_fill_setup_region(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    value_value, value_type)
  assert(std.is_region(dst_type))
  assert(std.type_supports_privileges(dst_container_type))

  local dst_all_fields = std.flatten_struct_fields(dst_type:fspace())
  local value_fields, value_field_types
  if terralib.types.istype(value_type) then
    value_fields, value_field_types = std.flatten_struct_fields(value_type)
  else
    value_fields = terralib.newlist(data.newtuple())
    value_field_types = value_type
  end

  local actions = terralib.newlist()
  for i, dst_field in ipairs(dst_fields) do
    local dst_copy_fields = data.filter(
      function(field) return field:starts_with(dst_field) end,
      dst_all_fields)
    assert(#dst_copy_fields == #value_fields)

    local dst_parent = get_container_root(
      cx, `([dst_value].impl), dst_container_type, dst_copy_fields)

    for j, dst_copy_field in ipairs(dst_copy_fields) do
      local value_field = value_fields[j]
      local value_field_type = value_field_types[j]
      local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
      local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

      local fill_value = value_value
      for _, field_name in ipairs(value_field) do
        fill_value = `([fill_value].[field_name])
      end
      fill_value = std.implicit_cast(
        value_field_type, dst_field_type, fill_value)

      actions:insert(quote
        var buffer : dst_field_type = [fill_value]
        c.legion_runtime_fill_field(
          [cx.runtime], [cx.context], [dst_value].impl, [dst_parent],
          dst_field_id, &buffer, terralib.sizeof(dst_field_type),
          c.legion_predicate_true())
      end)
    end
  end
  return actions
end

local function expr_fill_setup_list(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    value_value, value_type)
  if std.is_list(dst_type) then
    local dst_element_type = dst_type.element_type
    local dst_element = terralib.newsymbol(dst_element_type, "dst_element")
    return quote
      for i = 0, [dst_value].__size do
        var [dst_element] = [dst_type:data(dst_value)][i]
        [expr_fill_setup_list(
           cx, dst_element, dst_element_type, dst_container_type, dst_fields,
           value_value, value_type)]
      end
    end
  else
    return expr_fill_setup_region(
      cx, dst_value, dst_type, dst_container_type, dst_fields,
      value_value, value_type)
  end
end

function codegen.expr_fill(cx, node)
  local dst_type = std.as_read(node.dst.expr_type)
  local dst = codegen.expr_region_root(cx, node.dst):read(cx, dst_type)
  local value_type = std.as_read(node.value.expr_type)
  local value = codegen.expr(cx, node.value):read(cx, value_type)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)
  assert(#conditions == 0) -- FIXME: Can't issue phase barriers on fills.

  local actions = quote
    [dst.actions]
    [value.actions]
    [conditions:map(
       function(condition) return condition.value.actions end)]
    [emit_debuginfo(node)]

    [expr_fill_setup_list(
       cx, dst.value, dst_type, dst_type, node.dst.fields,
       value.value, value_type)]
  end

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

local function expr_acquire_issue_phase_barriers(values, condition_kinds, launcher)
  local actions = terralib.newlist()
  for i, value in ipairs(values) do
    local conditions = condition_kinds[i]
    for _, condition_kind in ipairs(conditions) do
      local add_barrier
      if condition_kind == std.awaits then
        add_barrier = c.legion_acquire_launcher_add_wait_barrier
      elseif condition_kind == std.arrives then
        add_barrier = c.legion_acquire_launcher_add_arrival_barrier
      else
        assert(false)
      end
      actions:insert(quote [add_barrier]([launcher], [value].impl) end)
    end
  end
  return actions
end

local function expr_acquire_extract_phase_barriers(index, values, value_types)
  local actions = terralib.newlist()
  local result_values = terralib.newlist()
  local result_types = terralib.newlist()
  for i, value in ipairs(values) do
    local value_type = value_types[i]

    local result_type = value_type.element_type
    local result_value = terralib.newsymbol(result_type, "condition_element")
    actions:insert(quote
        std.assert([index] < [value].__size, "barrier index out of bounds in acquire")
        var [result_value] = [value_type:data(value)][ [index] ]
    end)
    result_values:insert(result_value)
    result_types:insert(result_type)
  end
  return actions, result_values, result_types
end

local function expr_acquire_setup_region(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  assert(std.is_region(dst_type))
  assert(std.type_supports_privileges(dst_container_type))

  local dst_all_fields = std.flatten_struct_fields(dst_type:fspace())

  local actions = terralib.newlist()
  for i, dst_field in ipairs(dst_fields) do
    local dst_copy_fields = data.filter(
      function(field) return field:starts_with(dst_field) end,
      dst_all_fields)

    local dst_parent = get_container_root(
      cx, `([dst_value].impl), dst_container_type, dst_copy_fields)

    for j, dst_copy_field in ipairs(dst_copy_fields) do
      local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
      local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

      local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
      actions:insert(quote
        var [tag] = 0
        [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
        var launcher = c.legion_acquire_launcher_create(
          [dst_value].impl, [dst_parent],
          c.legion_predicate_true(), 0, [tag])
        c.legion_acquire_launcher_add_field(
          launcher, dst_field_id)
        [expr_acquire_issue_phase_barriers(condition_values, condition_kinds, launcher)]
        c.legion_acquire_launcher_execute([cx.runtime], [cx.context], [launcher])
      end)
    end
  end
  return actions
end

local function expr_acquire_setup_list(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  if std.is_list(dst_type) then
    local dst_element_type = dst_type.element_type
    local dst_element = terralib.newsymbol(dst_element_type, "dst_element")
    local index = terralib.newsymbol(uint64, "index")
    local c_actions, c_values, c_types = expr_acquire_extract_phase_barriers(
      index, condition_values, condition_types)
    return quote
      for [index] = 0, [dst_value].__size do
        var [dst_element] = [dst_type:data(dst_value)][ [index] ]
        [c_actions]
        [expr_acquire_setup_list(
           cx, dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds)]
      end
    end
  else
    return expr_acquire_setup_region(
      cx, dst_value, dst_type, dst_container_type, dst_fields,
      condition_values, condition_types, condition_kinds)
  end
end

function codegen.expr_acquire(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)

  local actions = quote
    [region.actions]
    [conditions:map(
       function(condition) return condition.value.actions end)]
    [emit_debuginfo(node)]

    [expr_acquire_setup_list(
       cx, region.value, region_type, region_type, node.region.fields,
       conditions:map(function(condition) return condition.value end),
       node.conditions:map(
         function(condition)
           return std.as_read(condition.value.expr_type)
         end),
       node.conditions:map(function(condition) return condition.conditions end))]
  end

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

local function expr_release_issue_phase_barriers(values, condition_kinds, launcher)
  local actions = terralib.newlist()
  for i, value in ipairs(values) do
    local conditions = condition_kinds[i]
    for _, condition_kind in ipairs(conditions) do
      local add_barrier
      if condition_kind == std.awaits then
        add_barrier = c.legion_release_launcher_add_wait_barrier
      elseif condition_kind == std.arrives then
        add_barrier = c.legion_release_launcher_add_arrival_barrier
      else
        assert(false)
      end
      actions:insert(quote [add_barrier]([launcher], [value].impl) end)
    end
  end
  return actions
end

local function expr_release_extract_phase_barriers(index, values, value_types)
  local actions = terralib.newlist()
  local result_values = terralib.newlist()
  local result_types = terralib.newlist()
  for i, value in ipairs(values) do
    local value_type = value_types[i]

    local result_type = value_type.element_type
    local result_value = terralib.newsymbol(result_type, "condition_element")
    actions:insert(quote
        std.assert([index] < [value].__size, "barrier index out of bounds in release")
        var [result_value] = [value_type:data(value)][ [index] ]
    end)
    result_values:insert(result_value)
    result_types:insert(result_type)
  end
  return actions, result_values, result_types
end

local function expr_release_setup_region(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  assert(std.is_region(dst_type))
  assert(std.type_supports_privileges(dst_container_type))

  local dst_all_fields = std.flatten_struct_fields(dst_type:fspace())

  local actions = terralib.newlist()
  for i, dst_field in ipairs(dst_fields) do
    local dst_copy_fields = data.filter(
      function(field) return field:starts_with(dst_field) end,
      dst_all_fields)

    local dst_parent = get_container_root(
      cx, `([dst_value].impl), dst_container_type, dst_copy_fields)

    for j, dst_copy_field in ipairs(dst_copy_fields) do
      local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
      local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

      local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
      actions:insert(quote
        var [tag] = 0
        [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
        var launcher = c.legion_release_launcher_create(
          [dst_value].impl, [dst_parent],
          c.legion_predicate_true(), 0, [tag])
        c.legion_release_launcher_add_field(
          launcher, dst_field_id)
        [expr_release_issue_phase_barriers(condition_values, condition_kinds, launcher)]
        c.legion_release_launcher_execute([cx.runtime], [cx.context], [launcher])
      end)
    end
  end
  return actions
end

local function expr_release_setup_list(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  if std.is_list(dst_type) then
    local dst_element_type = dst_type.element_type
    local dst_element = terralib.newsymbol(dst_element_type, "dst_element")
    local index = terralib.newsymbol(uint64, "index")
    local c_actions, c_values, c_types = expr_release_extract_phase_barriers(
      index, condition_values, condition_types)
    return quote
      for [index] = 0, [dst_value].__size do
        var [dst_element] = [dst_type:data(dst_value)][ [index] ]
        [c_actions]
        [expr_release_setup_list(
           cx, dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds)]
      end
    end
  else
    return expr_release_setup_region(
      cx, dst_value, dst_type, dst_container_type, dst_fields,
      condition_values, condition_types, condition_kinds)
  end
end

function codegen.expr_release(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)

  local actions = quote
    [region.actions]
    [conditions:map(
       function(condition) return condition.value.actions end)]
    [emit_debuginfo(node)]

    [expr_release_setup_list(
       cx, region.value, region_type, region_type, node.region.fields,
       conditions:map(function(condition) return condition.value end),
       node.conditions:map(
         function(condition)
           return std.as_read(condition.value.expr_type)
         end),
       node.conditions:map(function(condition) return condition.conditions end))]
  end

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

function codegen.expr_attach_hdf5(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local filename_type = std.as_read(node.filename.expr_type)
  local filename = codegen.expr(cx, node.filename):read(cx, filename_type)
  local mode_type = std.as_read(node.mode.expr_type)
  local mode = codegen.expr(cx, node.mode):read(cx, mode_type)

  if not cx.variant:get_config_options().inner then
    report.warn(node, "WARNING: Attach invalidates region contents. DO NOT attempt to access region after using attach.")
  end

  assert(cx:has_region(region_type))

  local all_fields = std.flatten_struct_fields(region_type:fspace())
  local full_fields = terralib.newlist()
  for i, region_field in ipairs(node.region.fields) do
    local region_full_fields = data.filter(
      function(field) return field:starts_with(region_field) end,
      all_fields)
    full_fields:insertall(region_full_fields)
  end

  local field_map = terralib.newsymbol(c.legion_field_map_t, "field_map")
  local field_map_setup = quote
    var [field_map] = c.legion_field_map_create()
    [full_fields:map(
       function(field_path)
         return quote
           c.legion_field_map_insert(
               [field_map],
               [cx:region(region_type):field_id(field_path)],
               [field_path:concat(".")])
         end
       end)]
  end
  local field_map_teardown = quote
    c.legion_field_map_destroy([field_map])
  end

  local parent = get_container_root(
    cx, `([region.value].impl), region_type, full_fields)

  local new_pr = terralib.newsymbol(c.legion_physical_region_t, "new_pr")

  local actions = quote
    [region.actions]
    [filename.actions]
    [mode.actions]
    [emit_debuginfo(node)]

    [field_map_setup]
    var [new_pr] = c.legion_runtime_attach_hdf5(
      [cx.runtime], [cx.context],
      [filename.value], [region.value].impl, [parent], [field_map], [mode.value])
    [field_map_teardown]

    [full_fields:map(
       function(field_path)
         return quote
           -- FIXME: This is redundant (since the same physical region
           -- will generally show up more than once. At any rate, it
           -- would be preferable not to have to do this at all.
           [cx:region(region_type):physical_region(field_path)] = [new_pr]
         end
       end)]
  end

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

function codegen.expr_detach_hdf5(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)

  if not cx.variant:get_config_options().inner then
    report.warn(node, "WARNING: Detach invalidates region contents. DO NOT attempt to access region after using detach.")
  end

  assert(cx:has_region(region_type))

  local all_fields = std.flatten_struct_fields(region_type:fspace())
  local full_fields = terralib.newlist()
  for _, region_field in ipairs(node.region.fields) do
    local region_full_fields = data.filter(
      function(field) return field:starts_with(region_field) end,
      all_fields)
    full_fields:insertall(region_full_fields)
  end

  -- Hack: De-duplicate physical regions by symbol.
  local pr_set = {}
  for _, field_path in ipairs(full_fields) do
    pr_set[cx:region(region_type):physical_region(field_path)] = true
  end
  local pr_list = terralib.newlist()
  for pr, _ in pairs(pr_set) do
    pr_list:insert(pr)
  end

  local actions = quote
    [region.actions]
    [emit_debuginfo(node)]

    [pr_list:map(
       function(pr)
         return quote
           c.legion_runtime_detach_hdf5([cx.runtime], [cx.context], [pr])
         end
       end)]
  end

  return values.value(node, expr.just(actions, quote end), terralib.types.unit)
end

function codegen.expr_allocate_scratch_fields(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote
    [region.actions]
    [emit_debuginfo(node)]
  end

  assert(cx:has_region_or_list(region_type))
  local field_space
  if std.is_region(region_type) then
    field_space = `([region.value].impl.field_space)
  elseif std.is_list_of_regions(region_type) then
    field_space = terralib.newsymbol(c.legion_field_space_t, "field_space")
    actions = quote
      regentlib.assert([region.value].__size > 0, "attempting to allocate scratch fields for empty list")
      var r = [region_type:data(region.value)][0]
      var [field_space] = r.impl.field_space
    end
  else
    assert(false)
  end

  local field_ids = terralib.newsymbol(expr_type, "field_ids")
  actions = quote
    [actions]
    var fsa = c.legion_field_allocator_create(
      [cx.runtime], [cx.context], [field_space])
    var [field_ids]
    [data.zip(data.range(#node.region.fields), node.region.fields):map(
       function(field)
         local i, field_path = unpack(field)
         local field_name = field_path:mkstring("", ".", "") .. ".(scratch)"
         local field_type = cx:region_or_list(region_type):field_type(field_path)
         return quote
           [field_ids][i] = c.legion_field_allocator_allocate_local_field(
             fsa, terralib.sizeof(field_type), -1ULL)
           c.legion_field_id_attach_name(
             [cx.runtime], [field_space], [field_ids][i], field_name, false)
         end
       end)]
    c.legion_field_allocator_destroy(fsa)
  end

  return values.value(node, expr.just(actions, field_ids), expr_type)
end

function codegen.expr_with_scratch_fields(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local field_ids_type = std.as_read(node.field_ids.expr_type)
  local field_ids = codegen.expr(cx, node.field_ids):read(cx, field_ids_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote
    [region.actions]
    [field_ids.actions]
    [emit_debuginfo(node)]
  end

  local value = expr.once_only(
    actions,
    std.implicit_cast(region_type, expr_type, region.value),
    expr_type)

  assert(cx:has_region_or_list(region_type))

  local old_field_ids = cx:region_or_list(region_type).field_ids
  local new_field_ids = {}
  for k, v in pairs(old_field_ids) do
    new_field_ids[k] = v
  end
  for i, field_path in pairs(node.region.fields) do
    new_field_ids[field_path:hash()] = `([field_ids.value][ [i-1] ])
  end

  local old_fields_are_scratch = cx:region_or_list(region_type).fields_are_scratch
  local new_fields_are_scratch = {}
  for k, v in pairs(old_fields_are_scratch) do
    new_fields_are_scratch[k] = v
  end
  for i, field_path in pairs(node.region.fields) do
    new_fields_are_scratch[field_path:hash()] = true
  end

  if std.is_region(region_type) then
    cx:add_region_root(
      expr_type, value.value,
      cx:region(region_type).field_paths,
      cx:region(region_type).privilege_field_paths,
      cx:region(region_type).field_privileges,
      cx:region(region_type).field_types,
      new_field_ids,
      new_fields_are_scratch,
      false,
      false,
      false)
  elseif std.is_list_of_regions(region_type) then
    cx:add_list_of_regions(
      expr_type, value.value,
      cx:list_of_regions(region_type).field_paths,
      cx:list_of_regions(region_type).privilege_field_paths,
      cx:list_of_regions(region_type).field_privileges,
      cx:list_of_regions(region_type).field_types,
      new_field_ids,
      new_fields_are_scratch)
  else
    assert(false)
  end

  return values.value(node, value, expr_type)
end

local lift_unary_op_to_futures = terralib.memoize(
  function (op, rhs_type, expr_type)
    assert(terralib.types.istype(rhs_type) and
             terralib.types.istype(expr_type))
    if std.is_future(rhs_type) then
      rhs_type = rhs_type.result_type
    end
    if std.is_future(expr_type) then
      expr_type = expr_type.result_type
    end

    local name = data.newtuple(
      "__unary_" .. tostring(rhs_type) .. "_" .. tostring(op))
    local rhs_symbol = std.newsymbol(rhs_type, "rhs")
    local task = std.new_task(name)
    local variant = task:make_variant("primary")
    task:set_primary_variant(variant)
    local node = ast.typed.top.Task {
      name = name,
      params = terralib.newlist({
          ast.typed.top.TaskParam {
            symbol = rhs_symbol,
            param_type = rhs_type,
            future = false,
            annotations = ast.default_annotations(),
            span = ast.trivial_span(),
          },
      }),
      return_type = expr_type,
      privileges = terralib.newlist(),
      coherence_modes = data.newmap(),
      flags = data.newmap(),
      conditions = {},
      constraints = terralib.newlist(),
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.stat.Return {
              value = ast.typed.expr.Unary {
                op = op,
                rhs = ast.typed.expr.ID {
                  value = rhs_symbol,
                  expr_type = rhs_type,
                  annotations = ast.default_annotations(),
                  span = ast.trivial_span(),
                },
                expr_type = expr_type,
                annotations = ast.default_annotations(),
                span = ast.trivial_span(),
              },
              annotations = ast.default_annotations(),
              span = ast.trivial_span(),
            },
        }),
        span = ast.trivial_span(),
      },
      config_options = ast.TaskConfigOptions {
        leaf = true,
        inner = false,
        idempotent = true,
      },
      region_divergence = false,
      prototype = task,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    task:set_type(
      terralib.types.functype(
        node.params:map(function(param) return param.param_type end),
        node.return_type,
        false))
    task:set_privileges(node.privileges)
    task:set_conditions({})
    task:set_param_constraints(node.constraints)
    task:set_constraints({})
    task:set_region_universe(data.newmap())
    return codegen.entry(node)
  end)

local lift_binary_op_to_futures = terralib.memoize(
  function (op, lhs_type, rhs_type, expr_type)
    assert(terralib.types.istype(lhs_type) and
             terralib.types.istype(rhs_type) and
             terralib.types.istype(expr_type))
    if std.is_future(lhs_type) then
      lhs_type = lhs_type.result_type
    end
    if std.is_future(rhs_type) then
      rhs_type = rhs_type.result_type
    end
    if std.is_future(expr_type) then
      expr_type = expr_type.result_type
    end

    local name = data.newtuple(
      "__binary_" .. tostring(lhs_type) .. "_" ..
        tostring(rhs_type) .. "_" .. tostring(op))
    local lhs_symbol = std.newsymbol(lhs_type, "lhs")
    local rhs_symbol = std.newsymbol(rhs_type, "rhs")
    local task = std.new_task(name)
    local variant = task:make_variant("primary")
    task:set_primary_variant(variant)
    local node = ast.typed.top.Task {
      name = name,
      params = terralib.newlist({
         ast.typed.top.TaskParam {
            symbol = lhs_symbol,
            param_type = lhs_type,
            future = false,
            annotations = ast.default_annotations(),
            span = ast.trivial_span(),
         },
         ast.typed.top.TaskParam {
            symbol = rhs_symbol,
            param_type = rhs_type,
            future = false,
            annotations = ast.default_annotations(),
            span = ast.trivial_span(),
         },
      }),
      return_type = expr_type,
      privileges = terralib.newlist(),
      coherence_modes = data.newmap(),
      flags = data.newmap(),
      conditions = {},
      constraints = terralib.newlist(),
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.stat.Return {
              value = ast.typed.expr.Binary {
                op = op,
                lhs = ast.typed.expr.ID {
                  value = lhs_symbol,
                  expr_type = lhs_type,
                  annotations = ast.default_annotations(),
                  span = ast.trivial_span(),
                },
                rhs = ast.typed.expr.ID {
                  value = rhs_symbol,
                  expr_type = rhs_type,
                  annotations = ast.default_annotations(),
                  span = ast.trivial_span(),
                },
                expr_type = expr_type,
                annotations = ast.default_annotations(),
                span = ast.trivial_span(),
              },
              annotations = ast.default_annotations(),
              span = ast.trivial_span(),
            },
        }),
        span = ast.trivial_span(),
      },
      config_options = ast.TaskConfigOptions {
        leaf = true,
        inner = false,
        idempotent = true,
      },
      region_divergence = false,
      prototype = task,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    task:set_type(
      terralib.types.functype(
        node.params:map(function(param) return param.param_type end),
        node.return_type,
        false))
    task:set_privileges(node.privileges)
    task:set_conditions({})
    task:set_param_constraints(node.constraints)
    task:set_constraints({})
    task:set_region_universe(data.newmap())
    return codegen.entry(node)
  end)

function codegen.expr_unary(cx, node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    local rhs_type = std.as_read(node.rhs.expr_type)
    local task = lift_unary_op_to_futures(node.op, rhs_type, expr_type)

    local call = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = task,
        expr_type = task:get_type(),
        annotations = ast.default_annotations(),
        span = node.span,
      },
      args = terralib.newlist({node.rhs}),
      conditions = terralib.newlist(),
      replicable = false,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
    return codegen.expr(cx, call)
  else
    local rhs = codegen.expr(cx, node.rhs):read(cx, expr_type)
    local actions = quote
      [rhs.actions];
      [emit_debuginfo(node)]
    end
    return values.value(
      node,
      expr.once_only(actions, std.quote_unary_op(node.op, rhs.value), expr_type),
      expr_type)
  end
end

function codegen.expr_binary(cx, node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_partition(expr_type) then
    local lhs = codegen.expr(cx, node.lhs):read(cx, node.lhs.expr_type)
    local rhs = codegen.expr(cx, node.rhs):read(cx, node.rhs.expr_type)
    local actions = quote
      [lhs.actions];
      [rhs.actions];
      [emit_debuginfo(node)]
    end

    local create_partition
    if node.op == "-" then
      create_partition = c.legion_index_partition_create_by_difference
    elseif node.op == "&" then
      create_partition = c.legion_index_partition_create_by_intersection
    elseif node.op == "|" then
      create_partition = c.legion_index_partition_create_by_union
    else
      assert(false)
    end

    local partition_type = std.as_read(node.expr_type)
    local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
    local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
    actions = quote
      [actions]
      var is = c.legion_index_partition_get_parent_index_space(
        [cx.runtime], [lhs.value].impl.index_partition)
      var lhs_colors = c.legion_index_partition_get_color_space(
        [cx.runtime], [lhs.value].impl.index_partition)
      var rhs_colors = c.legion_index_partition_get_color_space(
        [cx.runtime], [rhs.value].impl.index_partition)
      var colors : c.legion_index_space_t
      if lhs_colors.tid ~= rhs_colors.tid or lhs_colors.id ~= rhs_colors.id then
        var color_spaces : c.legion_index_space_t[2]
        color_spaces[0] = lhs_colors
        color_spaces[1] = rhs_colors
        colors = c.legion_index_space_union(
          [cx.runtime], [cx.context], &(color_spaces[0]), 2) -- FIXME: Leaks
      else
        colors = lhs_colors
      end
      var [ip] = [create_partition](
        [cx.runtime], [cx.context],
        is, [lhs.value].impl.index_partition, [rhs.value].impl.index_partition,
        colors,
        [(partition_type:is_disjoint() and c.DISJOINT_KIND) or c.COMPUTE_KIND],
        -1)
      var [lp] = c.legion_logical_partition_create_by_tree(
        [cx.runtime], [cx.context],
        [ip], [lhs.value].impl.field_space, [lhs.value].impl.tree_id)
    end

    return values.value(
      node,
      expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
      partition_type)
  elseif std.is_future(expr_type) then
    local lhs_type = std.as_read(node.lhs.expr_type)
    local rhs_type = std.as_read(node.rhs.expr_type)
    local task = lift_binary_op_to_futures(
      node.op, lhs_type, rhs_type, expr_type)

    local call = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = task,
        expr_type = task:get_type(),
        annotations = ast.default_annotations(),
        span = node.span,
      },
      args = terralib.newlist({node.lhs, node.rhs}),
      conditions = terralib.newlist(),
      replicable = false,
      expr_type = expr_type,
      annotations = node.annotations,
      span = node.span,
    }
    return codegen.expr(cx, call)
  else
    local lhs = codegen.expr(cx, node.lhs):read(cx, node.lhs.expr_type)
    local rhs = codegen.expr(cx, node.rhs):read(cx, node.rhs.expr_type)
    local actions = quote
      [lhs.actions];
      [rhs.actions];
      [emit_debuginfo(node)]
    end

    local expr_type = std.as_read(node.expr_type)
    return values.value(
      node,
      expr.once_only(actions, std.quote_binary_op(node.op, lhs.value, rhs.value), expr_type),
      expr_type)
  end
end

function codegen.expr_deref(cx, node)
  local value = codegen.expr(cx, node.value):read(cx, node.value.expr_type)
  local value_type = std.as_read(node.value.expr_type)

  if value_type:ispointer() then
    return values.rawptr(node, value, value_type)
  elseif std.is_bounded_type(value_type) then
    assert(value_type:is_ptr())
    return values.ref(node, value, value_type)
  elseif std.is_vptr(value_type) then
    return values.vref(node, value, value_type)
  else
    assert(false)
  end
end

function codegen.expr_future(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  local content_type = expr_type.result_type
  local content_value = std.implicit_cast(value_type, content_type, value.value)

  if content_type == terralib.types.unit then
    assert(false)
  else
    local buffer = terralib.newsymbol(&opaque, "buffer")
    local data_ptr = terralib.newsymbol(&uint8, "data_ptr")
    local result = terralib.newsymbol(c.legion_future_t, "result")

    local size_actions, size_value = std.compute_serialized_size(
      content_type, content_value)
    local ser_actions = std.serialize(
      content_type, content_value, buffer, `(&[data_ptr]))
    local actions = quote
      [actions]
      [size_actions]
      var buffer_size = terralib.sizeof(content_type) + [size_value]
      var [buffer] = c.malloc(buffer_size)
      std.assert([buffer] ~= nil, "malloc failed in future")
      var [data_ptr] = [&uint8]([buffer]) + terralib.sizeof(content_type)
      [ser_actions]
      std.assert(
        [data_ptr] - [&uint8]([buffer]) == buffer_size,
        "mismatch in data serialized in future")
      var [result] = c.legion_future_from_untyped_pointer(
        [cx.runtime], [buffer], buffer_size)
      c.free([buffer])
    end

    return values.value(
      node,
      expr.once_only(actions, `([expr_type]{ __result = [result] }), expr_type),
      expr_type)
  end
end

function codegen.expr_future_get_result(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote
    [value.actions];
    [emit_debuginfo(node)]
  end

  if expr_type == terralib.types.unit then
    assert(false)
  else
    local buffer = terralib.newsymbol(&opaque, "buffer")
    local buffer_size = terralib.newsymbol(c.size_t, "buffer_size")
    local result = terralib.newsymbol(expr_type, "result")
    local data_ptr = terralib.newsymbol(&uint8, "data_ptr")

    local deser_actions, deser_value = std.deserialize(
      expr_type, buffer, `(&[data_ptr]))
    local actions = quote
      [actions]
      var [buffer] = c.legion_future_get_untyped_pointer([value.value].__result)
      var [buffer_size] = c.legion_future_get_untyped_size([value.value].__result)
      var [data_ptr] = [&uint8]([buffer]) + terralib.sizeof(expr_type)
      [deser_actions]
      var [result] = [deser_value]
      std.assert([buffer_size] == [data_ptr] - [&uint8]([buffer]),
        "mismatch in data left over in future")
    end
    return values.value(
      node,
      expr.just(actions, result),
      expr_type)
  end
end

function codegen.expr(cx, node)
  if node:is(ast.typed.expr.Internal) then
    return codegen.expr_internal(cx, node)

  elseif node:is(ast.typed.expr.ID) then
    return codegen.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
    return codegen.expr_constant(cx, node)

  elseif node:is(ast.typed.expr.Function) then
    return codegen.expr_function(cx, node)

  elseif node:is(ast.typed.expr.FieldAccess) then
    return codegen.expr_field_access(cx, node)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return codegen.expr_index_access(cx, node)

  elseif node:is(ast.typed.expr.MethodCall) then
    return codegen.expr_method_call(cx, node)

  elseif node:is(ast.typed.expr.Call) then
    return codegen.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    return codegen.expr_cast(cx, node)

  elseif node:is(ast.typed.expr.Ctor) then
    return codegen.expr_ctor(cx, node)

  elseif node:is(ast.typed.expr.RawContext) then
    return codegen.expr_raw_context(cx, node)

  elseif node:is(ast.typed.expr.RawFields) then
    return codegen.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return codegen.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return codegen.expr_raw_runtime(cx, node)

  elseif node:is(ast.typed.expr.RawValue) then
    return codegen.expr_raw_value(cx, node)

  elseif node:is(ast.typed.expr.Isnull) then
    return codegen.expr_isnull(cx, node)

  elseif node:is(ast.typed.expr.Null) then
    return codegen.expr_null(cx, node)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return codegen.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.expr.StaticCast) then
    return codegen.expr_static_cast(cx, node)

  elseif node:is(ast.typed.expr.UnsafeCast) then
    return codegen.expr_unsafe_cast(cx, node)

  elseif node:is(ast.typed.expr.Ispace) then
    return codegen.expr_ispace(cx, node)

  elseif node:is(ast.typed.expr.Region) then
    return codegen.expr_region(cx, node)

  elseif node:is(ast.typed.expr.Partition) then
    return codegen.expr_partition(cx, node)

  elseif node:is(ast.typed.expr.PartitionEqual) then
    return codegen.expr_partition_equal(cx, node)

  elseif node:is(ast.typed.expr.PartitionByField) then
    return codegen.expr_partition_by_field(cx, node)

  elseif node:is(ast.typed.expr.Image) then
    return codegen.expr_image(cx, node)

  elseif node:is(ast.typed.expr.Preimage) then
    return codegen.expr_preimage(cx, node)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return codegen.expr_cross_product(cx, node)

  elseif node:is(ast.typed.expr.CrossProductArray) then
    return codegen.expr_cross_product_array(cx, node)

  elseif node:is(ast.typed.expr.ListSlicePartition) then
    return codegen.expr_list_slice_partition(cx, node)

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return codegen.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.typed.expr.ListSliceCrossProduct) then
    return codegen.expr_list_slice_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProduct) then
    return codegen.expr_list_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProductComplete) then
    return codegen.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.typed.expr.ListPhaseBarriers) then
    return codegen.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.typed.expr.ListInvert) then
    return codegen.expr_list_invert(cx, node)

  elseif node:is(ast.typed.expr.ListRange) then
    return codegen.expr_list_range(cx, node)

  elseif node:is(ast.typed.expr.ListIspace) then
    return codegen.expr_list_ispace(cx, node)

  elseif node:is(ast.typed.expr.ListFromElement) then
    return codegen.expr_list_from_element(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return codegen.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollective) then
    return codegen.expr_dynamic_collective(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return codegen.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return codegen.expr_advance(cx, node)

  elseif node:is(ast.typed.expr.Adjust) then
    return codegen.expr_adjust(cx, node)

  elseif node:is(ast.typed.expr.Arrive) then
    return codegen.expr_arrive(cx, node)

  elseif node:is(ast.typed.expr.Await) then
    return codegen.expr_await(cx, node)

  elseif node:is(ast.typed.expr.Copy) then
    return codegen.expr_copy(cx, node)

  elseif node:is(ast.typed.expr.Fill) then
    return codegen.expr_fill(cx, node)

  elseif node:is(ast.typed.expr.Acquire) then
    return codegen.expr_acquire(cx, node)

  elseif node:is(ast.typed.expr.Release) then
    return codegen.expr_release(cx, node)

  elseif node:is(ast.typed.expr.AttachHDF5) then
    return codegen.expr_attach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.DetachHDF5) then
    return codegen.expr_detach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.AllocateScratchFields) then
    return codegen.expr_allocate_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.WithScratchFields) then
    return codegen.expr_with_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.Unary) then
    return codegen.expr_unary(cx, node)

  elseif node:is(ast.typed.expr.Binary) then
    return codegen.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.Deref) then
    return codegen.expr_deref(cx, node)

  elseif node:is(ast.typed.expr.Future) then
    return codegen.expr_future(cx, node)

  elseif node:is(ast.typed.expr.FutureGetResult) then
    return codegen.expr_future_get_result(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function codegen.expr_list(cx, node)
  return node:map(function(item) return codegen.expr(cx, item) end)
end

function codegen.block(cx, node)
  return node.stats:map(
    function(stat) return codegen.stat(cx, stat) end)
end

local function cleanup_after(cx, block)
  local result = terralib.newlist({quote [block] end})
  result:insert(cx:get_cleanup_items())
  return quote [result] end
end

function codegen.stat_internal(cx, node)
  assert(terralib.isquote(node.actions))
  return node.actions
end

function codegen.stat_if(cx, node)
  local clauses = terralib.newlist()

  -- Insert first clause in chain.
  local cond = codegen.expr(cx, node.cond):read(cx)
  local then_cx = cx:new_local_scope()
  local then_block = cleanup_after(then_cx, codegen.block(then_cx, node.then_block))
  clauses:insert({cond, then_block})

  -- Add rest of clauses.
  for _, elseif_block in ipairs(node.elseif_blocks) do
 cond = codegen.expr(cx, elseif_block.cond):read(cx)
    local elseif_cx = cx:new_local_scope()
    local block = cleanup_after(elseif_cx, codegen.block(elseif_cx, elseif_block.block))
    clauses:insert({cond, block})
  end
  local else_cx = cx:new_local_scope()
  local else_block = cleanup_after(else_cx, codegen.block(else_cx, node.else_block))

  -- Build chain of clauses backwards.
  local tail = else_block
  repeat
    local cond, block = unpack(clauses:remove())
    tail = quote
      if [quote [cond.actions] in [cond.value] end] then
        [block]
      else
        [tail]
      end
    end
  until #clauses == 0
  return tail
end

function codegen.stat_while(cx, node)
  local cond = codegen.expr(cx, node.cond):read(cx)
  local break_label = terralib.newlabel()
  local body_cx = cx:new_local_scope(nil, nil, nil, break_label)
  local block = cleanup_after(body_cx, codegen.block(body_cx, node.block))
  return quote
    while [quote [cond.actions] in [cond.value] end] do
      [block]
    end
    ::[break_label]::
  end
end

function codegen.stat_for_num(cx, node)
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local bounds = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)
  local break_label = terralib.newlabel()
  local cx = cx:new_local_scope(nil, nil, nil, break_label)
  local block = cleanup_after(cx, codegen.block(cx, node.block))

  local v1, v2, v3 = unpack(bounds)
  if #bounds == 2 then
    return quote
      [v1.actions]; [v2.actions]
      for [symbol] = [v1.value], [v2.value] do
        [block]
      end
      ::[break_label]::
    end
  else
    return quote
      [v1.actions]; [v2.actions]; [v3.actions]
      for [symbol] = [v1.value], [v2.value], [v3.value] do
        [block]
      end
      ::[break_label]::
    end
  end
end

function codegen.stat_for_num_vectorized(cx, node)
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local bounds = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)
  local cx = cx:new_local_scope()
  local block = cleanup_after(cx, codegen.block(cx, node.block))
  local orig_block = cleanup_after(cx, codegen.block(cx, node.orig_block))
  local vector_width = node.vector_width

  local v1, v2, v3 = unpack(bounds)
  assert(#bounds == 2)
  return quote
    [v1.actions]; [v2.actions]
    do
      var [symbol] = [v1.value]
      var stop = [v2.value]

      while [symbol] + vector_width - 1 < stop do
        [block]
        [symbol] = [symbol] + vector_width
      end

      while [symbol] < stop do
        [orig_block]
        [symbol] = [symbol] + 1
      end
    end
  end
end

-- Find variables defined from the outer scope
local function collect_symbols(cx, node)
  local result = terralib.newlist()

  local undefined = {}
  local reduction_variables = {}
  local defined = { [node.symbol] = true }
  local accesses = {}
  local function collect_symbol_pre(node)
    if rawget(node, "node_type") then
      if node:is(ast.typed.stat.Var) then
        defined[node.symbol] = true
      elseif node:is(ast.typed.stat.ForNum) or
             node:is(ast.typed.stat.ForList) then
        defined[node.symbol] = true
      end
    end
  end
  local function collect_symbol_post(node)
    if rawget(node, "node_type") then
      if node:is(ast.typed.expr.ID) and
             not defined[node.value] and
             not std.is_region(std.as_read(node.expr_type)) then
        undefined[node.value] = true
      elseif (node:is(ast.typed.expr.FieldAccess) or
              node:is(ast.typed.expr.IndexAccess)) and
             std.is_ref(node.expr_type) then
        accesses[node] = true
        if accesses[node.value] and
           std.is_ref(node.expr_type) and
           std.is_ref(node.value.expr_type) and
           node.expr_type:bounds() == node.value.expr_type:bounds() then
           accesses[node.value] = nil
        end
      elseif node:is(ast.typed.expr.FieldAccess) and
             node.field_name == "bounds" and
             (std.is_region(std.as_read(node.value.expr_type)) or
              std.is_ispace(std.as_read(node.value.expr_type))) then
        local ispace_type = std.as_read(node.value.expr_type)
        if std.is_region(ispace_type) then
          ispace_type = ispace_type:ispace()
        end
        undefined[cx:ispace(ispace_type).bounds] = true
      elseif node:is(ast.typed.expr.Deref) and
             std.is_ref(node.expr_type) and
             (not std.is_ref(node.value.expr_type) or
              node.expr_type:bounds() ~= node.value.expr_type:bounds()) then
        accesses[node] = true
      elseif node:is(ast.typed.stat.Reduce) then
        if node.lhs:is(ast.typed.expr.ID) and undefined[node.lhs.value] then
          reduction_variables[node.lhs.value:getsymbol()] = node.op
        end
      end
    end
  end
  ast.traverse_node_prepostorder(collect_symbol_pre,
                                 collect_symbol_post,
                                 node.block)

  -- Base pointers need a special treatment to find them
  local base_pointers = {}
  local strides = {}
  for node, _ in pairs(accesses) do
    local value_type = std.as_read(node.expr_type)
    node.expr_type:bounds():map(function(region)
      local prefix = node.expr_type.field_path
      local field_paths = std.flatten_struct_fields(value_type)
      local absolute_field_paths = field_paths:map(
        function(field_path) return prefix .. field_path end)
      absolute_field_paths:map(function(field_path)
        base_pointers[cx:region(region):base_pointer(field_path)] = true
        local stride = cx:region(region):stride(field_path)
        for idx = 2, #stride do strides[stride[idx]] = true end
      end)
    end)
  end

  for base_pointer, _ in pairs(base_pointers) do
    result:insert(base_pointer)
  end
  for stride, _ in pairs(strides) do
    result:insert(stride) end
  for symbol, _ in pairs(undefined) do
    if std.is_symbol(symbol) then symbol = symbol:getsymbol() end
    result:insert(symbol)
  end

  return result, reduction_variables
end

function codegen.stat_for_list(cx, node)
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local break_label = terralib.newlabel()
  local cx = cx:new_local_scope(nil, nil, nil, break_label)
  local block = cleanup_after(cx, codegen.block(cx, node.block))

  local ispace_type, is, it
  if std.is_ispace(value_type) then
    ispace_type = value_type
    is = `([value.value].impl)
  elseif std.is_region(value_type) then
    ispace_type = value_type:ispace()
    assert(cx:has_ispace(ispace_type))
    is = `([value.value].impl.index_space)
    it = cx:ispace(ispace_type).index_iterator
  elseif std.is_list(value_type) then
    return quote
      for i = 0, [value.value].__size do
        var [symbol] = [value_type:data(value.value)][i]
        do
          [block]
        end
      end
      ::[break_label]::
    end
  else
    assert(false)
  end

  local actions = quote
    [value.actions]
  end
  local cleanup_actions = quote end

  local iterator_has_next, iterator_next_span -- For unstructured
  local domain -- For structured
  if ispace_type.dim == 0 then
    if it and cache_index_iterator then
      iterator_has_next = c.legion_terra_cached_index_iterator_has_next
      iterator_next_span = c.legion_terra_cached_index_iterator_next_span
      actions = quote
        [actions]
        c.legion_terra_cached_index_iterator_reset(it)
      end
    else
      iterator_has_next = c.legion_index_iterator_has_next
      iterator_next_span = c.legion_index_iterator_next_span
      it = terralib.newsymbol(c.legion_index_iterator_t, "it")
      actions = quote
        [actions]
        var [it] = c.legion_index_iterator_create([cx.runtime], [cx.context], [is])
      end
      cleanup_actions = quote
        c.legion_index_iterator_destroy([it])
      end
    end
  else
    domain = terralib.newsymbol(c.legion_domain_t, "domain")
    actions = quote
      [actions]
      var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
    end
  end

  local cuda = cx.variant:is_cuda()
  local openmp = node.annotations.openmp:is(ast.annotation.Demand) and
                 openmphelper.check_openmp_available()
  if node.annotations.openmp:is(ast.annotation.Demand) and
     not openmphelper.check_openmp_available() then
    report.warn(node,
      "ignoring demand pragma at " .. node.span.source ..
      ":" .. tostring(node.span.start.line) ..
      " since the OpenMP module is unavailable")
  end

  if not cuda then
    if ispace_type.dim == 0 then
      if not openmp then
        return quote
          [actions]
          while iterator_has_next([it]) do
            var count : c.size_t = 0
            var base = iterator_next_span([it], &count, -1).value
            for i = 0, count do
              var [symbol] = [symbol.type]{
                __ptr = ptr {
                  __ptr = c.legion_ptr_t {
                    value = base + i
                  }
                }
              }
              do
                [block]
              end
            end
          end
          ::[break_label]::
          [cleanup_actions]
        end
      else
        local count = terralib.newsymbol(uint64, "count")
        local base = terralib.newsymbol(int64, "base")
        local symbols, reductions = collect_symbols(cx, node)
        symbols:insert(count)
        symbols:insert(base)
        local arg_type, mapping = openmphelper.generate_argument_type(symbols, reductions)
        local arg = terralib.newsymbol(&arg_type, "arg")
        local worker_init, launch_init =
          openmphelper.generate_argument_init(arg, arg_type, mapping, reductions)
        local worker_cleanup =
          openmphelper.generate_worker_cleanup(arg, arg_type, mapping, reductions)
        local launcher_cleanup =
          openmphelper.generate_launcher_cleanup(arg, arg_type, mapping, reductions)

        local terra omp_worker(data : &opaque)
          var [arg] = [&arg_type](data)
          [worker_init]
          var num_threads = [openmphelper.get_num_threads]()
          var thread_id = [openmphelper.get_thread_num]()
          var chunk = (count + num_threads - 1) / num_threads
          if chunk == 0 then chunk = 1 end
          var start_idx = thread_id * chunk + base
          var end_idx = (thread_id + 1) * chunk + base
          if end_idx > base + count then end_idx = base + count end
          for i = start_idx, end_idx do
            var [symbol] = [symbol.type]{ __ptr = ptr { __ptr = c.legion_ptr_t { value = i } } }
            do
              [block]
            end
          end
          [worker_cleanup]
        end

        return quote
          [actions]
          while iterator_has_next([it]) do
            var [count] = 0
            var [base] = iterator_next_span([it], &count, -1).value
            var arg_obj : arg_type
            var [arg] = &arg_obj
            [launch_init]
            [openmphelper.launch]([omp_worker], [arg], [openmphelper.get_max_threads](), 0)
            [launcher_cleanup]
          end
          ::[break_label]::
          [cleanup_actions]
        end
      end
    else
      local fields = ispace_type.index_type.fields
      if fields then
        local domain_get_rect = c["legion_domain_get_rect_" .. tostring(ispace_type.dim) .. "d"]
        local index = fields:map(function(field) return terralib.newsymbol(c.coord_t, tostring(field)) end)
        local body = quote
          var [symbol] = [symbol.type] { __ptr = [symbol.type.index_type.impl_type]{ index } }
          do
            [block]
          end
        end
        if not openmp then
          local rect_type = c["legion_rect_" .. tostring(ispace_type.dim) .. "d_t"]
          local rect = terralib.newsymbol(rect_type, "rect")
          for i = 1, ispace_type.dim do
            local rect_i = i - 1 -- C is zero-based, Lua is one-based
            body = quote
              for [ index[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
                [body]
              end
            end
          end
          return quote
            [actions]
            var [rect] = [domain_get_rect]([domain])
            [body]
            ::[break_label]::
            [cleanup_actions]
          end
        else
          local rect_type = c["legion_rect_" .. tostring(ispace_type.dim) .. "d_t"]
          local rect = terralib.newsymbol(&rect_type, "rect")
          for i = 1, ispace_type.dim do
            local rect_i = i - 1 -- C is zero-based, Lua is one-based
            if i ~= ispace_type.dim then
              body = quote
                for [ index[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
                  [body]
                end
              end
            else
              local start_idx = terralib.newsymbol(int64, "start_idx")
              local end_idx = terralib.newsymbol(int64, "end_idx")
              body = quote
                [openmphelper.generate_preamble_structured(rect, rect_i, start_idx, end_idx)]
                for [ index[i] ] = [start_idx], [end_idx] do
                  [body]
                end
              end
            end
          end
          local symbols, reductions = collect_symbols(cx, node)
          symbols:insert(rect)
          local arg_type, mapping = openmphelper.generate_argument_type(symbols, reductions)
          local arg = terralib.newsymbol(&arg_type, "arg")
          local worker_init, launch_init =
            openmphelper.generate_argument_init(arg, arg_type, mapping, reductions)
          local worker_cleanup =
            openmphelper.generate_worker_cleanup(arg, arg_type, mapping, reductions)
          local launcher_cleanup =
            openmphelper.generate_launcher_cleanup(arg, arg_type, mapping, reductions)
          local terra omp_worker(data : &opaque)
            var [arg] = [&arg_type](data)
            [worker_init]
            [body]
            [worker_cleanup]
          end
          return quote
            [actions]
            var r = [domain_get_rect]([domain])
            var [rect] = &r
            var arg_obj : arg_type
            var [arg] = &arg_obj
            [launch_init]
            [openmphelper.launch]([omp_worker], [arg], [openmphelper.get_max_threads](), 0)
            [launcher_cleanup]
            ::[break_label]::
            [cleanup_actions]
          end
        end
      else
        if not openmp then
          return quote
            [actions]
            var rect = c.legion_domain_get_rect_1d([domain])
            for i = rect.lo.x[0], rect.hi.x[0] + 1 do
              var [symbol] = [symbol.type]{ __ptr = i }
              do
                [block]
              end
            end
            ::[break_label]::
            [cleanup_actions]
          end
        else
          local start_idx = terralib.newsymbol(int64, "start_idx")
          local end_idx = terralib.newsymbol(int64, "end_idx")
          local rect_type = c.legion_rect_1d_t
          local rect = terralib.newsymbol(&rect_type, "rect")
          local symbols, reductions = collect_symbols(cx, node)
          symbols:insert(rect)
          local arg_type, mapping = openmphelper.generate_argument_type(symbols, reductions)
          local arg = terralib.newsymbol(&arg_type, "arg")
          local worker_init, launch_init =
            openmphelper.generate_argument_init(arg, arg_type, mapping, reductions)
          local worker_cleanup =
            openmphelper.generate_worker_cleanup(arg, arg_type, mapping, reductions)
          local launcher_cleanup =
            openmphelper.generate_launcher_cleanup(arg, arg_type, mapping, reductions)
          local terra omp_worker(data : &opaque)
            var [arg] = [&arg_type](data)
            [worker_init]
            [openmphelper.generate_preamble_structured(rect, 0, start_idx, end_idx)]
            for i = [start_idx], [end_idx] do
              var [symbol] = [symbol.type]{ __ptr = i }
              do
                [block]
              end
            end
            [worker_cleanup]
          end
          return quote
            [actions]
            var r = c.legion_domain_get_rect_1d([domain])
            var [rect] = &r
            var arg_obj : arg_type
            var [arg] = &arg_obj
            [launch_init]
            [openmphelper.launch]([omp_worker], [arg], [openmphelper.get_max_threads](), 0)
            [launcher_cleanup]
            ::[break_label]::
            [cleanup_actions]
          end
        end
      end
    end
  else
    -- Reject the loop if the body has external function calls
    ast.traverse_node_postorder(function(node)
      if node:is(ast.typed.expr.Call) then
        local fn = node.fn.value
        if std.is_task(fn) then
          report.error(node, "CUDA task cannot launch other tasks in a for loop")
        elseif cudahelper.replace_with_builtin(fn) == fn and fn ~= array then
          report.error(node, "CUDA task cannot call external functions in a for loop")
        end
      end
    end, node.block)

    -- Now wrap the body as a terra function
    local indices = terralib.newlist()
    local lower_bounds = terralib.newlist()
    local upper_bounds = terralib.newlist()
    local body
    if ispace_type:is_opaque() then
      indices:insert(terralib.newsymbol(c.coord_t, "ptr"))
      lower_bounds:insert(terralib.newsymbol(c.coord_t, "lo1"))
      upper_bounds:insert(terralib.newsymbol(c.coord_t, "hi1"))
      body = quote
        var [symbol] = [symbol.type]{
          __ptr = [ptr]{ __ptr = c.legion_ptr_t { value = [ indices[1] ] } }
        }
        do
          [block]
        end
      end
    else
      for i = 1, ispace_type.dim do
        lower_bounds:insert(terralib.newsymbol(c.coord_t, "lo" .. tostring(i)))
        upper_bounds:insert(terralib.newsymbol(c.coord_t, "hi" .. tostring(i)))
      end
      local fields = ispace_type.index_type.fields
      if fields then
        indices:insertall(
          fields:map(function(field) return terralib.newsymbol(c.coord_t, tostring(field)) end))
        body = quote
          var [symbol] = [symbol.type] { __ptr = [symbol.type.index_type.impl_type]{ [indices] } }
          do
            [block]
          end
        end
      else
        indices:insert(terralib.newsymbol(c.coord_t, "idx"))
        body = quote
          var [symbol] = [symbol.type]{ __ptr = [ indices[1] ] }
          do
            [block]
          end
        end
      end
    end

    local index_inits = terralib.newlist()
    local tid = terralib.newsymbol(c.size_t, "tid")
    local offsets = terralib.newlist()
    local counts = terralib.newlist()
    for idx = 1, #indices do
      offsets:insert(terralib.newsymbol(c.size_t, "offset" .. tostring(idx)))
      counts:insert(terralib.newsymbol(c.size_t, "dim_size_" .. tostring(idx)))
    end
    local count = counts[1]
    for idx = 2, #indices do
      count = `([count] * [ counts[idx] ])
    end

    -- Compute a global tid
    index_inits:insert(quote
      var [tid] = [cudahelper.global_thread_id()]
      if [tid] >= [count] then
        return
      end
    end)

    -- Convert the global tid into a point in an index space
    index_inits:insert(quote var [ offsets[1] ] = 1 end)
    for idx = 2, #indices do
      index_inits:insert(quote
        var [ offsets[idx] ] = [ offsets[idx - 1] ] * [ counts[idx - 1] ]
      end)
    end

    for idx = #indices, 1, -1 do
      index_inits:insert(quote
        var [ indices[idx] ] = [ lower_bounds[idx] ] + [tid] / [ offsets[idx] ]
        [tid] = [tid] % [ offsets[idx] ]
      end)
    end

    body = quote
      [index_inits]
      [body]
    end

    local args, reductions = collect_symbols(cx, node)
    -- Remove reduction variables from kernel argument list as
    -- we will define them in the kernel
    args = data.filter(function(arg) return reductions[arg] == nil end, args)
    local shared_mem_size = cudahelper.compute_reduction_buffer_size(node, reductions)
    local device_ptrs, device_ptrs_map, host_preamble =
      cudahelper.generate_reduction_preamble(reductions)
    local kernel_preamble, kernel_postamble =
      cudahelper.generate_reduction_kernel(reductions, device_ptrs_map)
    local host_postamble =
      cudahelper.generate_reduction_postamble(reductions, device_ptrs_map)
    args:insertall(lower_bounds)
    args:insertall(counts)
    args:insertall(device_ptrs)
    args:sort(function(s1, s2) return sizeof(s1.type) > sizeof(s2.type) end)

    local terra kernel([args])
      [kernel_preamble]
      [body]
      [kernel_postamble]
    end

    -- Register the kernel function to JIT
    local kernel_id = cx.task_meta:get_cuda_variant():add_cuda_kernel(kernel)

    ---- kernel launch
    local count = terralib.newsymbol(c.size_t, "count")
    local kernel_call =
      cudahelper.codegen_kernel_call(kernel_id, count, args, shared_mem_size)

    if ispace_type:is_opaque() then
      return quote
        [actions]
        [host_preamble]
        while iterator_has_next([it]) do
          var [ counts[1] ] = 0
          var [ lower_bounds[1] ] = iterator_next_span([it], &[ counts[1] ], -1).value
          var [count] = [ counts[1] ]
          [kernel_call]
        end
        [host_postamble]
        [cleanup_actions]
      end
    else
      local rect_type = c["legion_rect_" .. tostring(ispace_type.dim) .. "d_t"]
      local domain_get_rect = c["legion_domain_get_rect_" .. tostring(ispace_type.dim) .. "d"]
      local rect = terralib.newsymbol(rect_type, "rect")
      local bounds_setup = terralib.newlist()
      bounds_setup:insert(quote var [count] = 1 end)
      for idx = 1, ispace_type.dim do
        bounds_setup:insert(quote
          var [ lower_bounds[idx] ], [ counts[idx] ] =
            [rect].lo.x[ [idx - 1] ], [rect].hi.x[ [idx - 1] ] - [rect].lo.x[ [idx - 1] ] + 1
          [count] = [count] * [ counts[idx] ]
        end)
      end
      kernel_call = quote
        if [ count ] > 0 then [kernel_call] end
      end
      return quote
        [actions]
        [host_preamble]
        var [rect] = [domain_get_rect]([domain])
        [bounds_setup]
        [kernel_call]
        [host_postamble]
        [cleanup_actions]
      end
    end
  end
end

function codegen.stat_for_list_vectorized(cx, node)
  if cx.variant:is_cuda() or node.annotations.openmp:is(ast.annotation.Demand) then
    return codegen.stat_for_list(cx,
      ast.typed.stat.ForList {
        symbol = node.symbol,
        value = node.value,
        block = node.orig_block,
        span = node.span,
        annotations = node.annotations,
      })
  end
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local cx = cx:new_local_scope()
  local block = cleanup_after(cx, codegen.block(cx, node.block))
  local orig_block_1 = cleanup_after(cx, codegen.block(cx, node.orig_block))
  local orig_block_2 = cleanup_after(cx, codegen.block(cx, node.orig_block))
  local vector_width = node.vector_width

  local ispace_type, is, it
  if std.is_region(value_type) then
    ispace_type = value_type:ispace()
    assert(cx:has_ispace(ispace_type))
    is = `([value.value].impl.index_space)
    it = cx:ispace(ispace_type).index_iterator
  else
    ispace_type = value_type
    is = `([value.value].impl)
  end

  local actions = quote
    [value.actions]
  end
  local cleanup_actions = quote end

  local iterator_has_next, iterator_next_span -- For unstructured
  local domain -- For structured
  if ispace_type.dim == 0 then
    if it and cache_index_iterator then
      iterator_has_next = c.legion_terra_cached_index_iterator_has_next
      iterator_next_span = c.legion_terra_cached_index_iterator_next_span
      actions = quote
        [actions]
        c.legion_terra_cached_index_iterator_reset(it)
      end
    else
      iterator_has_next = c.legion_index_iterator_has_next
      iterator_next_span = c.legion_index_iterator_next_span
      it = terralib.newsymbol(c.legion_index_iterator_t, "it")
      actions = quote
        [actions]
        var [it] = c.legion_index_iterator_create([cx.runtime], [cx.context], [is])
      end
      cleanup_actions = quote
        c.legion_index_iterator_destroy([it])
      end
    end
  else
    domain = terralib.newsymbol(c.legion_domain_t, "domain")
    actions = quote
      [actions]
      var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
    end
  end

  if ispace_type.dim == 0 then
    return quote
      [actions]
      while iterator_has_next([it]) do
        var count : c.size_t = 0
        var base = iterator_next_span([it], &count, -1).value
        var alignment : c.size_t = [vector_width]
        var start = (base + alignment - 1) and not (alignment - 1)
        var stop = (base + count) and not (alignment - 1)
        var final = base + count
        var i = base
        if count >= vector_width then
          while i < start do
            var [symbol] = [symbol.type]{ __ptr = [ptr]{ __ptr = c.legion_ptr_t { value = i }}}
            do
              [orig_block_1]
            end
            i = i + 1
          end
          while i < stop do
            var [symbol] = [symbol.type]{ __ptr = [ptr]{ __ptr = c.legion_ptr_t { value = i }}}
            do
              [block]
            end
            i = i + [vector_width]
          end
        end
        while i < final do
          var [symbol] = [symbol.type]{ __ptr = [ptr]{ __ptr = c.legion_ptr_t { value = i }}}
          do
            [orig_block_2]
          end
          i = i + 1
        end
      end
      [cleanup_actions]
    end
  else
    local fields = ispace_type.index_type.fields
    if fields then
      local rect_type = c["legion_rect_" .. tostring(ispace_type.dim) .. "d_t"]
      local domain_get_rect = c["legion_domain_get_rect_" .. tostring(ispace_type.dim) .. "d"]
      local rect = terralib.newsymbol(rect_type, "rect")
      local index = fields:map(function(field) return terralib.newsymbol(c.coord_t, tostring(field)) end)
      local base = terralib.newsymbol(c.coord_t, "base")
      local count = terralib.newsymbol(c.coord_t, "base")
      local start = terralib.newsymbol(c.coord_t, "base")
      local stop = terralib.newsymbol(c.coord_t, "base")
      local final = terralib.newsymbol(c.coord_t, "base")

      local body = quote
        var [ index[1] ] = base
        if count >= [vector_width] then
          while [ index[1] ] < [start] do
            var [symbol] = [symbol.type] { __ptr = [symbol.type.index_type.impl_type]{ index } }
            do
              [orig_block_1]
            end
            [ index[1] ] = [ index[1] ] + 1
          end
          while [ index[1] ] < [stop] do
            var [symbol] = [symbol.type] { __ptr = [symbol.type.index_type.impl_type]{ index } }
            do
              [block]
            end
            [ index[1] ] = [ index[1] ] + [vector_width]
          end
        end
        while [ index[1] ] < [final] do
          var [symbol] = [symbol.type] { __ptr = [symbol.type.index_type.impl_type]{ index } }
          do
            [orig_block_2]
          end
          [ index[1] ] = [ index[1] ] + 1
        end
      end
      for i = 2, ispace_type.dim do
        local rect_i = i - 1 -- C is zero-based, Lua is one-based
        body = quote
          for [ index[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
            [body]
          end
        end
      end
      return quote
        [actions]
        var [rect] = [domain_get_rect]([domain])
        var alignment = [vector_width]
        var [base] = [rect].lo.x[0]
        var [count] = [rect].hi.x[0] - [rect].lo.x[0] + 1
        var [start] = ([base] + alignment - 1) and not (alignment - 1)
        var [stop] = ([base] + [count]) and not (alignment - 1)
        var [final] = [base] + [count]
        [body]
        [cleanup_actions]
      end
    else
      return quote
        [actions]
        var rect = c.legion_domain_get_rect_1d([domain])
        var alignment = [vector_width]
        var base = rect.lo.x[0]
        var count = rect.hi.x[0] - rect.lo.x[0] + 1
        var start = (base + alignment - 1) and not (alignment - 1)
        var stop = (base + count) and not (alignment - 1)
        var final = base + count

        var i = base
        if count >= [vector_width] then
          while i < start do
            var [symbol] = [symbol.type]{ __ptr = i }
            do
              [orig_block_1]
            end
            i = i + 1
          end
          while i < stop do
            var [symbol] = [symbol.type]{ __ptr = i }
            do
              [block]
            end
            i = i + [vector_width]
          end
        end
        while i < final do
          var [symbol] = [symbol.type]{ __ptr = i }
          do
            [orig_block_2]
          end
          i = i + 1
        end
        [cleanup_actions]
      end
    end
  end
end

function codegen.stat_repeat(cx, node)
  local cx = cx:new_local_scope()
  local block = codegen.block(cx, node.block)
  local until_cond = codegen.expr(cx, node.until_cond):read(cx)
  return quote
    repeat
      [block];
      [until_cond.actions];
      [cx:get_cleanup_items()]
    until [until_cond.value]
  end
end

function codegen.stat_must_epoch(cx, node)
  local must_epoch = terralib.newsymbol(c.legion_must_epoch_launcher_t, "must_epoch")
  local must_epoch_point = terralib.newsymbol(c.coord_t, "must_epoch_point")
  local future_map = terralib.newsymbol(c.legion_future_map_t, "legion_future_map_t")

  local cx = cx:new_local_scope(nil, must_epoch, must_epoch_point)
  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  local actions = quote
    var [tag] = 0
    [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
    var [must_epoch] = c.legion_must_epoch_launcher_create(0, [tag])
    var [must_epoch_point] = 0
    [cleanup_after(cx, codegen.block(cx, node.block))]
    var [future_map] = c.legion_must_epoch_launcher_execute(
      [cx.runtime], [cx.context], [must_epoch])
    c.legion_must_epoch_launcher_destroy([must_epoch])
  end
  actions = quote
    do
      [actions];
      c.legion_future_map_destroy([future_map])
    end
  end
  return actions
end

function codegen.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return quote
    do
      [cleanup_after(cx, codegen.block(cx, node.block))]
    end
  end
end

local function stat_index_launch_setup(cx, node, domain, actions)
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local preamble = node.preamble:map(function(stat) return codegen.stat(cx, stat) end)

  local fn = codegen.expr(cx, node.call.fn):read(cx)
  assert(std.is_task(fn.value))
  local args = terralib.newlist()
  local args_partitions = terralib.newlist()
  for i, arg in ipairs(node.call.args) do
    local partition = false
    if not node.args_provably.projectable[i] then
      args:insert(codegen.expr(cx, arg):read(cx))
    else
      -- Run codegen halfway to get the partition. Note: Remember to
      -- splice the actions back in later.
      partition = codegen.expr(cx, arg.value):read(cx)

      -- Now run codegen the rest of the way to get the region.
      local partition_type = std.as_read(arg.value.expr_type)
      local region = codegen.expr(
        cx,
        ast.typed.expr.IndexAccess {
          value = ast.typed.expr.Internal {
            value = values.value(
              node,
              expr.just(quote end, partition.value),
              partition_type),
            expr_type = partition_type,
            annotations = node.annotations,
            span = node.span,
          },
          index = arg.index,
          expr_type = arg.expr_type,
          annotations = node.annotations,
          span = node.span,
        }):read(cx)
      args:insert(region)
    end
    args_partitions:insert(partition)
  end
  local conditions = node.call.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)

  local actions = quote
    [actions]
    [fn.actions];
    [data.zip(args, args_partitions, node.args_provably.invariant):map(
       function(pair)
         local arg, arg_partition, invariant = unpack(pair)

         -- Here we slice partition actions back in.
         local arg_actions = quote end
         if arg_partition then
           arg_actions = quote [arg_actions]; [arg_partition.actions] end
         end

         -- Normal invariant arg actions.
         if invariant then
           arg_actions = quote [arg_actions]; [arg.actions] end
         end

         return arg_actions
       end)];
    [conditions:map(function(condition) return condition.actions end)]
  end

  local arg_types = terralib.newlist()
  for i, arg in ipairs(args) do
    arg_types:insert(std.as_read(node.call.args[i].expr_type))
  end

  local arg_values = terralib.newlist()
  local param_types = node.call.fn.expr_type.parameters
  for i, arg in ipairs(args) do
    local arg_value = args[i].value
    if i <= #param_types and param_types[i] ~= std.untyped and
      not std.is_future(arg_types[i])
    then
      arg_values:insert(std.implicit_cast(arg_types[i], param_types[i], arg_value))
    else
      arg_values:insert(arg_value)
    end
  end

  local value_type = fn.value:get_type().returntype

  local params_struct_type = fn.value:get_params_struct()
  local task_args = terralib.newsymbol(c.legion_task_argument_t, "task_args")
  local task_args_setup = terralib.newlist()
  local task_args_cleanup = terralib.newlist()
  task_args_setup:insertall(preamble)
  for i, arg in ipairs(args) do
    local invariant = node.args_provably.invariant[i]
    if not invariant then
      task_args_setup:insert(arg.actions)
    end
  end
  expr_call_setup_task_args(
    cx, fn.value, arg_values, arg_types, param_types,
    params_struct_type, fn.value:has_params_map_label(), fn.value:has_params_map_type(),
    task_args, task_args_setup, task_args_cleanup)

  local launcher = terralib.newsymbol(c.legion_index_launcher_t, "launcher")

  -- Pass futures.
  local args_setup = terralib.newlist()
  for i, arg_type in ipairs(arg_types) do
    if std.is_future(arg_type) then
      local arg_value = arg_values[i]
      expr_call_setup_future_arg(
        cx, fn.value, arg_value, launcher, true, args_setup)
    end
  end

  -- Pass phase barriers.
  local param_conditions = fn.value:get_conditions()
  for condition, args_enabled in pairs(param_conditions) do
    for i, arg_type in ipairs(arg_types) do
      if args_enabled[i] then
        assert(std.is_phase_barrier(arg_type) or
          (std.is_list(arg_type) and std.is_phase_barrier(arg_type.element_type)))
        local arg_value = arg_values[i]
        expr_call_setup_phase_barrier_arg(
          cx, fn.value, arg_value, condition,
          launcher, true, args_setup, arg_type)
      end
    end
  end

  -- Pass phase barriers (from extra conditions).
  for i, condition in ipairs(node.call.conditions) do
    local condition_expr = conditions[i]
    for _, condition_kind in ipairs(condition.conditions) do
      expr_call_setup_phase_barrier_arg(
        cx, fn.value, condition_expr.value, condition_kind,
        launcher, false, args_setup, std.as_read(condition.expr_type))
    end
  end

  -- Pass index spaces through index requirements.
  for i, arg_type in ipairs(arg_types) do
    if std.is_ispace(arg_type) then
      local param_type = param_types[i]

      if not node.args_provably.projectable[i] then
        expr_call_setup_ispace_arg(
          cx, fn.value, arg_type, param_type, launcher, true, args_setup)
      else
        assert(false) -- FIXME: Implement index partitions

        -- local partition = args_partitions[i]
        -- assert(partition)
        -- expr_call_setup_ispace_partition_arg(
        --   cx, fn.value, arg_type, param_type, partition.value, launcher, true,
        --   ispace_args_setup)
      end
    end
  end

  -- Pass regions through region requirements.
  for _, i in ipairs(std.fn_param_regions_by_index(fn.value:get_type())) do
    local arg_value = arg_values[i]
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    if not node.args_provably.projectable[i] then
      expr_call_setup_region_arg(
        cx, fn.value, node.call.args[i], arg_type, param_type, launcher, true, args_setup)
    else
      local partition = args_partitions[i]
      assert(partition)
      expr_call_setup_partition_arg(
        cx, fn.value, node.call.args[i], arg_type, param_type, partition.value, node.symbol, launcher, true,
        args_setup)
    end
  end

  local must_epoch_setup = quote end
  if cx.must_epoch then
    -- FIXME: This is totally broken. It is not safe to edit the loop
    -- bounds to avoid collisions with other index launches, and on
    -- top of that this code won't successfully number single task
    -- launches correctly unless they follow very specific patterns.
    must_epoch_setup = quote
      var launch_size = c.legion_domain_get_volume([domain])
      [cx.must_epoch_point] = [cx.must_epoch_point] + launch_size
    end
  end

  local point = terralib.newsymbol(c.legion_domain_point_t, "point")

  local symbol_type = node.symbol:gettype()
  local symbol = node.symbol:getsymbol()
  local symbol_setup
  if std.is_bounded_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]({ __ptr = [symbol_type.index_type]([point]) })
    end
  else
    -- Otherwise symbol_type has to be some simple integral type.
    assert(symbol_type:isintegral())
    symbol_setup = quote
      var [symbol] = [int1d]([point])
    end
  end

  local argument_map = terralib.newsymbol(c.legion_argument_map_t, "argument_map")
  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  local launcher_setup = quote
    [must_epoch_setup]
    var [argument_map] = c.legion_argument_map_create()
    do
      var it = c.legion_domain_point_iterator_create([domain])
      while c.legion_domain_point_iterator_has_next(it) do
        var [point] = c.legion_domain_point_iterator_next(it)
        [symbol_setup]

        var [task_args]
        [task_args_setup]
        c.legion_argument_map_set_point(
          [argument_map], [point], [task_args], true)
        [task_args_cleanup]
      end
      c.legion_domain_point_iterator_destroy(it)
    end

    var g_args : c.legion_task_argument_t
    g_args.args = nil
    g_args.arglen = 0
    var [tag] = 0
    [codegen_hooks.gen_update_mapping_tag(tag, cx.task)]
    var [launcher] = c.legion_index_launcher_create(
      [fn.value:get_task_id()],
      [domain], g_args, [argument_map],
      c.legion_predicate_true(), false, 0, [tag])
    [args_setup]
  end

  local execute_fn = c.legion_index_launcher_execute
  local execute_args = terralib.newlist({
      cx.runtime, cx.context, launcher})
  local reduce_as_type = std.as_read(node.call.expr_type)
  if std.is_future(reduce_as_type) then
    reduce_as_type = reduce_as_type.result_type
  end
  if node.reduce_lhs then
    execute_fn = c.legion_index_launcher_execute_reduction

    local op = std.reduction_op_ids[node.reduce_op][reduce_as_type]
    assert(op)
    execute_args:insert(op)
  end

  local future, launcher_execute
  if not cx.must_epoch then
    if node.reduce_lhs then
      future = terralib.newsymbol(c.legion_future_t, "future")
    else
      future = terralib.newsymbol(c.legion_future_map_t, "future")
    end
    launcher_execute = quote
      var [future] = execute_fn(execute_args)
    end
  else
    launcher_execute = quote
      c.legion_must_epoch_launcher_add_index_task(
        [cx.must_epoch], [launcher])
    end
  end

  if node.reduce_lhs then
    assert(not cx.must_epoch)
    local rhs_type = std.as_read(node.call.expr_type)
    local future_type = rhs_type
    if not std.is_future(rhs_type) then
      future_type = std.future(rhs_type)
    end

    local rh = terralib.newsymbol(future_type)
    local rhs = ast.typed.expr.Internal {
      value = values.value(node, expr.just(quote end, rh), future_type),
      expr_type = future_type,
      annotations = node.annotations,
      span = node.span,
    }

    if not std.is_future(rhs_type) then
      rhs = ast.typed.expr.FutureGetResult {
        value = rhs,
        expr_type = rhs_type,
        annotations = node.annotations,
        span = node.span,
      }
    end

    local reduce = ast.typed.stat.Reduce {
      op = node.reduce_op,
      lhs = node.reduce_lhs,
      rhs = rhs,
      annotations = node.annotations,
      span = node.span,
    }

    launcher_execute = quote
      [launcher_execute]
      var [rh] = [future_type]({ __result = [future] })
      [codegen.stat(cx, reduce)]
    end
  end

  local destroy_future_fn = c.legion_future_map_destroy
  if node.reduce_lhs then
    destroy_future_fn = c.legion_future_destroy
  end

  local launcher_cleanup
  if not cx.must_epoch then
    launcher_cleanup = quote
      c.legion_argument_map_destroy([argument_map])
      destroy_future_fn([future])
      c.legion_index_launcher_destroy([launcher])
    end
  else
    launcher_cleanup = quote
      c.legion_argument_map_destroy([argument_map])
    end
  end

  actions = quote
    [actions];
    [launcher_setup];
    [launcher_execute];
    [launcher_cleanup]
  end
  return actions
end

function codegen.stat_index_launch_num(cx, node)
  local values = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)

  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local actions = quote
    [values:map(function(value) return value.actions end)]
    var [domain] = c.legion_domain_from_rect_1d(
      c.legion_rect_1d_t {
        lo = c.legion_point_1d_t { x = arrayof(c.coord_t, [values[1].value]) },
        hi = c.legion_point_1d_t { x = arrayof(c.coord_t, [values[2].value] - 1) },
      })
  end

  return stat_index_launch_setup(cx, node, domain, actions)
end

function codegen.stat_index_launch_list(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local actions
  if std.is_ispace(value_type) then
    actions = quote
      [value.actions]
      var [domain] = c.legion_index_space_get_domain(
        [cx.runtime], [value.value].impl)
    end
  elseif std.is_region(value_type) then
    actions = quote
      [value.actions]
      var [domain] = c.legion_index_space_get_domain(
        [cx.runtime], [value.value].impl.index_space)
    end
  else
    assert(false)
  end

  return stat_index_launch_setup(cx, node, domain, actions)
end

function codegen.stat_var(cx, node)
  local lhs = node.symbol:getsymbol()

  -- Capture rhs values (copying if necessary).
  local rhs = false
  if node.value then
    local rh = codegen.expr(cx, node.value):read(cx, node.value.expr_type)

    local rh_value = rh.value
    if node.value:is(ast.typed.expr.ID) then
      -- If this is a variable, copy the value to preserve ownership.
      rh_value = make_copy(cx, rh_value, node.value.expr_type)
    end
    rh = expr.just(rh.actions, rh_value)

    rhs = rh
  end

  -- Cast rhs values to lhs types.
  local actions = terralib.newlist()
  local rhs_value = false
  if rhs then
    local rhs_type = std.as_read(node.value.expr_type)
    local lhs_type = node.type
    if lhs_type then
      rhs_value = std.implicit_cast(rhs_type, lhs_type, rhs.value)
    else
      rhs_value = rh.value
    end
    actions:insert(rhs.actions)
  end

  -- Register cleanup items for lhs.
  local lhs_type = node.symbol:gettype()
  cx:add_cleanup_item(make_cleanup_item(cx, lhs, lhs_type))

  local function is_partitioning_expr(node)
    if node:is(ast.typed.expr.Partition) or node:is(ast.typed.expr.PartitionEqual) or
       node:is(ast.typed.expr.PartitionByField) or node:is(ast.typed.expr.Image) or
       node:is(ast.typed.expr.Preimage) or
       (node:is(ast.typed.expr.Binary) and std.is_partition(node.expr_type)) then
      return true
    else
      return false
    end
  end

  local decls = terralib.newlist()
  if node.value then
    if node.value:is(ast.typed.expr.Ispace) then
      actions = quote
        [actions]
        c.legion_index_space_attach_name([cx.runtime], [rhs_value].impl, [lhs.displayname], false)
      end
    elseif node.value:is(ast.typed.expr.Region) then
      actions = quote
        [actions]
        c.legion_logical_region_attach_name([cx.runtime], [rhs_value].impl, [lhs.displayname], false)
      end
    elseif is_partitioning_expr(node.value) then
      actions = quote
        [actions]
        c.legion_logical_partition_attach_name([cx.runtime], [rhs_value].impl, [lhs.displayname], false)
      end
    end
    decls:insert(quote var [lhs] = [rhs_value] end)
  else
    decls:insert(quote var [lhs] end)
  end
  return quote [actions]; [decls] end
end

function codegen.stat_var_unpack(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  local lhs = node.symbols:map(function(symbol) return symbol:getsymbol() end)
  local rhs = terralib.newlist()
  local actions = value.actions
  for i, field_name in ipairs(node.fields) do
    local field_type = node.field_types[i]

    local static_field_type = std.get_field(value_type, field_name)
    local field_value = std.implicit_cast(
      static_field_type, field_type, `([value.value].[field_name]))
    rhs:insert(field_value)

    if std.is_region(field_type) then
      local field_expr = expr.just(actions, field_value)
      field_expr = unpack_region(cx, field_expr, field_type, static_field_type)
      actions = quote [actions]; [field_expr.actions] end
    end
  end

  return quote
    [actions]
    var [lhs] = [rhs]
  end
end

function codegen.stat_return(cx, node)
  if not node.value then
    return quote return end
  end
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local result_type = cx.expected_return_type

  local result = terralib.newsymbol(result_type, "result")
  local actions = quote
    [value.actions]
    var [result] = [std.implicit_cast(value_type, result_type, value.value)]
  end

  if result_type == terralib.types.unit then
    return quote
      [actions]
      return
    end
  else
    -- Force unaligned access because malloc does not provide
    -- blocks aligned for all purposes (e.g. SSE vectors).
    local result_type_alignment = 1 -- data.min(terralib.sizeof(result_type), 8)

    return quote
      [actions]
      var buffer_size = terralib.sizeof([result_type])
      var buffer = c.malloc(buffer_size)
      std.assert(buffer ~= nil, "malloc failed in return")
      terralib.attrstore(
        [&result_type](buffer), result,
        { align = [result_type_alignment] })
      return std.serialized_value {
        value = buffer,
        size = buffer_size,
      }
      -- Task wrapper is responsible for calling free.
    end
  end
end

function codegen.stat_break(cx, node)
  assert(cx.break_label)
  return quote goto [cx.break_label] end
end

function codegen.stat_assignment(cx, node)
  local actions = terralib.newlist()
  local lhs = codegen.expr(cx, node.lhs)
  local rhs = codegen.expr(cx, node.rhs)

  local rhs_expr = rhs:read(cx, node.rhs.expr_type)

  -- Capture the rhs value in a temporary so that it doesn't get
  -- overridden on assignment to the lhs (if lhs and rhs alias).
  do
    local rhs_expr_value = rhs_expr.value
    if node.rhs:is(ast.typed.expr.ID) then
      -- If this is a variable, copy the value to preserve ownership.
      rhs_expr_value = make_copy(cx, rhs_expr_value, node.rhs.expr_type)
    end
    rhs_expr = expr.once_only(rhs_expr.actions, rhs_expr_value, node.rhs.expr_type)
  end

  actions:insert(rhs_expr.actions)
  rhs = values.value(
    node.rhs,
    expr.just(quote end, rhs_expr.value),
    std.as_read(node.rhs.expr_type))

  actions:insert(lhs:write(cx, rhs, node.lhs.expr_type).actions)

  return quote [actions] end
end

function codegen.stat_reduce(cx, node)
  local actions = terralib.newlist()
  local lhs = codegen.expr(cx, node.lhs)
  local rhs = codegen.expr(cx, node.rhs)

  local rhs_expr = rhs:read(cx, node.rhs.expr_type)

  actions:insert(rhs_expr.actions)
  rhs = values.value(
    node,
    expr.just(quote end, rhs_expr.value),
    std.as_read(node.rhs.expr_type))

  actions:insert(lhs:reduce(cx, rhs, node.op, node.lhs.expr_type).actions)

  return quote [actions] end
end

function codegen.stat_expr(cx, node)
  local expr = codegen.expr(cx, node.expr):read(cx)

  -- If the value is stored in a variable, it will be cleaned up at
  -- the end of the variable's lifetime. Otherwise cleanup now.
  if not node.expr:is(ast.typed.expr.ID) then
    local cleanup = make_cleanup_item(cx, expr.value, node.expr.expr_type)
    return quote [expr.actions]; [cleanup] end
  else
    return quote [expr.actions] end
  end
end

function codegen.stat_begin_trace(cx, node)
  local trace_id = codegen.expr(cx, node.trace_id):read(cx)
  return quote
    [trace_id.actions];
    [emit_debuginfo(node)];
    c.legion_runtime_begin_trace([cx.runtime], [cx.context], [trace_id.value])
  end
end

function codegen.stat_end_trace(cx, node)
  local trace_id = codegen.expr(cx, node.trace_id):read(cx)
  return quote
    [trace_id.actions];
    [emit_debuginfo(node)];
    c.legion_runtime_end_trace([cx.runtime], [cx.context], [trace_id.value])
  end
end

local function find_region_roots(cx, region_types)
  local roots_by_type = {}
  for _, region_type in ipairs(region_types) do
    assert(cx:has_region(region_type))
    local root_region_type = cx:region(region_type).root_region_type
    roots_by_type[root_region_type] = true
  end
  local roots = terralib.newlist()
  for region_type, _ in pairs(roots_by_type) do
    roots:insert(region_type)
  end
  return roots
end

local function find_region_roots_physical(cx, region_types)
  local roots = find_region_roots(cx, region_types)
  local result = terralib.newlist()
  for _, region_type in ipairs(roots) do
    local physical_regions = cx:region(region_type).physical_regions
    local privilege_field_paths = cx:region(region_type).privilege_field_paths
    for _, field_paths in ipairs(privilege_field_paths) do
      for _, field_path in ipairs(field_paths) do
        result:insert(physical_regions[field_path:hash()])
      end
    end
  end
  return result
end

function codegen.stat_map_regions(cx, node)
  -- FIXME: For now, assume that lists of regions are NEVER mapped.
  assert(not(data.any(unpack(node.region_types:map(
    function(region_type) return not std.is_region(region_type) end)))))
  local roots = find_region_roots_physical(cx, node.region_types)
  local actions = terralib.newlist()
  for _, pr in ipairs(roots) do
    actions:insert(
      `(c.legion_runtime_remap_region([cx.runtime], [cx.context], [pr])))
  end
  for _, pr in ipairs(roots) do
    actions:insert(
      `(c.legion_physical_region_wait_until_valid([pr])))
  end
  return quote [actions] end
end

function codegen.stat_unmap_regions(cx, node)
  local regions = data.filter(
    function(region_type) return std.is_region(region_type) end,
    node.region_types)
  local roots = find_region_roots_physical(cx, regions)
  local actions = terralib.newlist()
  for _, pr in ipairs(roots) do
    actions:insert(
      `(c.legion_runtime_unmap_region([cx.runtime], [cx.context], [pr])))
  end
  return quote [actions] end
end

function codegen.stat_raw_delete(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  local region_delete_fn, ispace_delete_fn, ispace_getter
  if std.is_region(value_type) then
    region_delete_fn = c.legion_logical_region_destroy
    ispace_delete_fn = c.legion_index_space_destroy
    ispace_getter = function(x) return `([x.value].impl.index_space) end
  else
    region_delete_fn = c.legion_logical_partition_destroy
    ispace_delete_fn = c.legion_index_partition_destroy
    ispace_getter = function(x) return `([x.value].impl.index_partition) end
  end

  return quote
    [value.actions]
    [region_delete_fn]([cx.runtime], [cx.context], [value.value].impl)
    [ispace_delete_fn]([cx.runtime], [cx.context], [ispace_getter(value)])
  end
end

local make_dummy_task = terralib.memoize(
  function()
    local name = data.newtuple("__dummy")
    local task = std.new_task(name)
    local variant = task:make_variant("primary")
    task:set_primary_variant(variant)
    local expr_type = int
    local node = ast.typed.top.Task {
      name = name,
      params = terralib.newlist(),
      return_type = expr_type,
      privileges = terralib.newlist(),
      coherence_modes = data.newmap(),
      flags = data.newmap(),
      conditions = {},
      constraints = terralib.newlist(),
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.stat.Return {
              value = ast.typed.expr.Constant {
                value = 0,
                expr_type = expr_type,
                annotations = ast.default_annotations(),
                span = ast.trivial_span(),
              },
              annotations = ast.default_annotations(),
              span = ast.trivial_span(),
            },
        }),
        span = ast.trivial_span(),
      },
      config_options = ast.TaskConfigOptions {
        leaf = true,
        inner = false,
        idempotent = true,
      },
      region_divergence = false,
      prototype = task,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }
    task:set_type(
      terralib.types.functype(terralib.newlist(), node.return_type, false))
    task:set_privileges(node.privileges)
    task:set_conditions({})
    task:set_param_constraints(node.constraints)
    task:set_constraints({})
    task:set_region_universe(data.newmap())
    return codegen.entry(node)
  end)

function codegen.stat_fence(cx, node)
  local kind = node.kind
  local blocking = node.blocking

  local issue_fence
  if kind:is(ast.fence_kind.Execution) then
    issue_fence = c.legion_runtime_issue_execution_fence
  elseif kind:is(ast.fence_kind.Mapping) then
    issue_fence = c.legion_runtime_issue_mapping_fence
  end

  local actions = terralib.newlist()

  actions:insert(
    quote
      [issue_fence]([cx.runtime], [cx.context])
    end)

  if blocking then
    local task = make_dummy_task()

    local call = ast.typed.expr.Call {
      fn = ast.typed.expr.Function {
        value = task,
        expr_type = task:get_type(),
        annotations = ast.default_annotations(),
        span = node.span,
      },
      args = terralib.newlist(),
      conditions = terralib.newlist(),
      replicable = false,
      expr_type = task:get_type().returntype,
      annotations = node.annotations,
      span = node.span,
    }

    local call_expr = codegen.expr(cx, call):read(cx)
    actions:insert(call_expr.actions)
  end

  return quote
    [actions]
  end
end

function codegen.stat_parallelize_with(cx, node)
  return quote [codegen.block(cx, node.block)] end
end

function codegen.stat(cx, node)
  manual_gc()

  if node:is(ast.typed.stat.Internal) then
    return codegen.stat_internal(cx, node)

  elseif node:is(ast.typed.stat.If) then
    return codegen.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return codegen.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return codegen.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForNumVectorized) then
    return codegen.stat_for_num_vectorized(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return codegen.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.ForListVectorized) then
    return codegen.stat_for_list_vectorized(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return codegen.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return codegen.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return codegen.stat_block(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return codegen.stat_index_launch_num(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return codegen.stat_index_launch_list(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return codegen.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return codegen.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return codegen.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return codegen.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    return codegen.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return codegen.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return codegen.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.BeginTrace) then
    return codegen.stat_begin_trace(cx, node)

  elseif node:is(ast.typed.stat.EndTrace) then
    return codegen.stat_end_trace(cx, node)

  elseif node:is(ast.typed.stat.MapRegions) then
    return codegen.stat_map_regions(cx, node)

  elseif node:is(ast.typed.stat.UnmapRegions) then
    return codegen.stat_unmap_regions(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    return codegen.stat_raw_delete(cx, node)

  elseif node:is(ast.typed.stat.Fence) then
    return codegen.stat_fence(cx, node)

  elseif node:is(ast.typed.stat.ParallelizeWith) then
    return codegen.stat_parallelize_with(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

local function get_params_map_type(params)
  if #params == 0 then
    return false
  else
    return uint64[math.ceil(#params/64)]
  end
end

local function filter_fields(fields, privileges)
  local remove = terralib.newlist()
  for _, field in pairs(fields) do
    local privilege = privileges[data.hash(field)]
    if not privilege or std.is_reduction_op(privilege) then
      remove:insert(field)
    end
  end
  for _, field in ipairs(remove) do
    fields[field] = nil
  end
  return fields
end

local function unpack_param_helper(cx, node, param_type, params_map_type, i)
  -- Inputs/outputs:
  local c_task = terralib.newsymbol(c.legion_task_t, "task")
  local params_map = terralib.newsymbol(params_map_type, "params_map_type")
  local fixed_ptr = terralib.newsymbol(&opaque, "fixed_ptr")
  local data_ptr = terralib.newsymbol(&&uint8, "data_ptr")
  local future_count = terralib.newsymbol(int32, "future_count")
  local future_i = terralib.newsymbol(&int32, "future_i")

  -- Generate code to unpack a future.
  local future = terralib.newsymbol(c.legion_future_t, "future")
  local future_type = std.future(param_type)
  local future_result = codegen.expr(
    cx,
    ast.typed.expr.FutureGetResult {
      value = ast.typed.expr.Internal {
        value = values.value(
          node,
          expr.just(quote end, `([future_type]{ __result = [future] })),
          future_type),
        expr_type = future_type,
        annotations = node.annotations,
        span = node.span,
      },
      expr_type = param_type,
      annotations = node.annotations,
      span = node.span,
  }):read(cx)

  -- Generate code to unpack a non-future.
  local deser_actions, deser_value = std.deserialize(
    param_type, fixed_ptr, data_ptr)

  local terra unpack_param([c_task], [params_map], [fixed_ptr], [data_ptr],
                           [future_count], [future_i])
    if ([params_map][ [math.floor((i-1)/64)] ] and [2ULL ^ math.fmod(i-1, 64)]) == 0 then
      [deser_actions]
      return [deser_value]
    else
      std.assert(@[future_i] < [future_count], "missing future in task param")
      var [future] = c.legion_task_get_future([c_task], @[future_i])
      @[future_i] = @[future_i] + 1
      [future_result.actions]
      var result = [future_result.value]
      c.legion_future_destroy([future])
      return result
    end
  end
  unpack_param:setinlined(false)
  return unpack_param
end

local function setup_regent_calling_convention_metadata(node, task)
  local params_struct_type = terralib.types.newstruct()
  params_struct_type.entries = terralib.newlist()
  task:set_params_struct(params_struct_type)

  -- The param map tracks which parameters are stored in the task
  -- arguments versus futures. The space in the params struct will be
  -- reserved either way, but this tells us where to find a valid copy
  -- of the data. Currently this field has a fixed size to keep the
  -- code here sane, though conceptually it's just a bit vector.
  local params_map_type = get_params_map_type(node.params)
  local params_map_label = false
  local params_map_symbol = false
  if params_map_type then
    params_map_label = terralib.newlabel("__map")
    params_map_symbol = terralib.newsymbol(params_map_type, "__map")
    params_struct_type.entries:insert(
      { field = params_map_label, type = params_map_type })

    task:set_params_map_type(params_map_type)
    task:set_params_map_label(params_map_label)
    task:set_params_map_symbol(params_map_symbol)
  end

  -- Normal arguments are straight out of the param types.
  params_struct_type.entries:insertall(node.params:map(
    function(param)
      local param_label = param.symbol:getlabel()
      return { field = param_label, type = param.param_type }
    end))

  -- Regions require some special handling here. Specifically, field
  -- IDs are going to be passed around dynamically, so we need to
  -- reserve some extra slots in the params struct here for those
  -- field IDs.
  local fn_type = task:get_type()
  local param_types = task:get_type().parameters
  local param_field_id_labels = data.newmap()
  local param_field_id_symbols = data.newmap()
  for _, region_i in pairs(std.fn_params_with_privileges_by_index(fn_type)) do
    local region = param_types[region_i]
    local field_paths, field_types =
      std.flatten_struct_fields(region:fspace())
    local field_id_labels = field_paths:map(
      function(field_path)
        return terralib.newlabel("field_" .. field_path:hash())
      end)
    local field_id_symbols = field_paths:map(
      function(field_path)
        return terralib.newsymbol(c.legion_field_id_t, "field_" .. field_path:hash())
      end)
    param_field_id_labels[region_i] = field_id_labels
    param_field_id_symbols[region_i] = field_id_symbols
    params_struct_type.entries:insertall(
      data.zip(field_id_labels, field_types):map(
        function(field)
          local field_id, field_type = unpack(field)
          return { field = field_id, type = c.legion_field_id_t }
        end))
  end
  task:set_field_id_param_labels(param_field_id_labels)
  task:set_field_id_param_symbols(param_field_id_symbols)
end

function codegen.top_task(cx, node)
  local task = node.prototype
  local variant = cx.variant
  assert(variant)
  assert(task == variant.task)

  -- we temporaily turn off generating two task versions for cuda tasks
  if node.annotations.cuda:is(ast.annotation.Demand) then
    node = node { region_divergence = false }
  end

  variant:set_config_options(node.config_options)

  local c_task = terralib.newsymbol(c.legion_task_t, "task")
  local c_regions = terralib.newsymbol(&c.legion_physical_region_t, "regions")
  local c_num_regions = terralib.newsymbol(uint32, "num_regions")
  local c_context = terralib.newsymbol(c.legion_context_t, "context")
  local c_runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
  local c_params = terralib.newlist({
      c_task, c_regions, c_num_regions, c_context, c_runtime })

  local params = node.params:map(
    function(param) return param.symbol end)

  local param_types = task:get_type().parameters
  local return_type = node.return_type

  local params_struct_type = task:get_params_struct()
  local params_map_type = task:has_params_map_type() and
    task:get_params_map_type()
  local params_map_label = task:has_params_map_label() and
    task:get_params_map_label()
  local params_map_symbol = task:has_params_map_symbol() and
    task:get_params_map_symbol()

  local cx = cx:new_task_scope(return_type,
                               task:get_constraints(),
                               variant:get_config_options().leaf,
                               task, c_task, c_context, c_runtime)

  -- Unpack the by-value parameters to the task.
  local task_setup = terralib.newlist()
  -- FIXME: This is an obnoxious hack to avoid inline mappings in shard tasks.
  --        Will be fixed with a proper handling of list of regions in
  --        the inline mapping optimizer.
  if cx.task_meta:is_shard_task() then
    task_setup:insert(quote
      c.legion_runtime_unmap_all_regions([c_runtime], [c_context])
    end)
  end
  local args = terralib.newsymbol(&params_struct_type, "args")
  local arglen = terralib.newsymbol(c.size_t, "arglen")
  local data_ptr = terralib.newsymbol(&uint8, "data_ptr")
  if #(task:get_params_struct():getentries()) > 0 then
    task_setup:insert(quote
      var [args], [arglen], [data_ptr]
      if c.legion_task_get_is_index_space(c_task) then
        [arglen] = c.legion_task_get_local_arglen(c_task)
        std.assert([arglen] >= terralib.sizeof(params_struct_type),
                   ["arglen mismatch in " .. tostring(task.name) .. " (index task)"])
        args = [&params_struct_type](c.legion_task_get_local_args(c_task))
      else
        [arglen] = c.legion_task_get_arglen(c_task)
        std.assert([arglen] >= terralib.sizeof(params_struct_type),
                   ["arglen mismatch " .. tostring(task.name) .. " (single task)"])
        args = [&params_struct_type](c.legion_task_get_args(c_task))
      end
      var [data_ptr] = [&uint8](args) + terralib.sizeof(params_struct_type)
    end)
    task_setup:insert(quote
      var [params_map_symbol] = args.[params_map_label]
    end)

    local future_count = terralib.newsymbol(int32, "future_count")
    local future_i = terralib.newsymbol(int32, "future_i")
    task_setup:insert(quote
      var [future_count] = c.legion_task_get_futures_size([c_task])
      var [future_i] = 0
    end)
    for i = 1, #params do
      local param = params[i]
      local param_type = node.params[i].param_type
      local param_symbol = param:getsymbol()

      local helper = unpack_param_helper(cx, node, param_type, params_map_type, i)

      local actions = quote
        var [param_symbol] = [helper](
          [c_task], [params_map_symbol], &args.[param:getlabel()], &[data_ptr],
          [future_count], &[future_i])
      end
      if std.is_ispace(param_type) and not cx:has_ispace(param_type) then
        local bounds
        if not param_type:is_opaque() then
          local bounds_actions
          bounds_actions, bounds =
            index_space_bounds(cx, `([param_symbol].impl), param_type.index_type)
          actions = quote [actions]; [bounds_actions] end
        end
        cx:add_ispace_root(param_type, `([param_symbol].impl), false, bounds)
      end
      task_setup:insert(actions)
    end
    task_setup:insert(quote
      std.assert([future_i] == [future_count],
        "extra futures left over in task params")
      std.assert([arglen] == [data_ptr] - [&uint8]([args]),
        "mismatch in data left over in task params")
    end)
  end

  -- Prepare any region parameters to the task.

  -- Unpack field IDs passed by-value to the task.
  local param_field_id_labels = task:get_field_id_param_labels()
  local param_field_id_symbols = task:get_field_id_param_symbols()
  for region, param_fields in param_field_id_labels:items() do
    for i, param_field in ipairs(param_fields) do
      local param_symbol = param_field_id_symbols[region][i]
      task_setup:insert(quote
        var [param_symbol] = args.[param_field]
      end)
    end
  end

  -- Unpack the region requirements.
  local physical_region_i = 0
  local fn_type = task:get_type()
  for _, region_i in ipairs(std.fn_param_regions_by_index(fn_type)) do
    local region_type = param_types[region_i]
    local index_type = region_type:ispace().index_type
    local r = params[region_i]:getsymbol()
    local is = terralib.newsymbol(c.legion_index_space_t, "is")
    local it = false
    if cache_index_iterator then
      it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
    end

    local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
      std.find_task_privileges(region_type, task)

    local privileges_by_field_path = std.group_task_privileges_by_field_path(
      privileges, privilege_field_paths)

    local field_paths, field_types =
      std.flatten_struct_fields(region_type:fspace())
    local field_ids_by_field_path = data.dict(
      data.zip(field_paths:map(data.hash), param_field_id_symbols[region_i]))

    local physical_regions = terralib.newlist()
    local physical_regions_by_field_path = {}
    local physical_regions_index = terralib.newlist()
    local physical_region_actions = terralib.newlist()
    local base_pointers = terralib.newlist()
    local base_pointers_by_field_path = {}
    local strides = terralib.newlist()
    local strides_by_field_path = {}
    for i, field_paths in ipairs(privilege_field_paths) do
      local privilege = privileges[i]
      local field_types = privilege_field_types[i]
      local flag = flags[i]
      local physical_region = terralib.newsymbol(
        c.legion_physical_region_t,
        "pr_" .. tostring(physical_region_i))

      physical_regions:insert(physical_region)
      physical_regions_index:insert(physical_region_i)
      physical_region_i = physical_region_i + 1

      -- we still need physical regions for map/unmap operations to work
      for i, field_path in ipairs(field_paths) do
        physical_regions_by_field_path[field_path:hash()] = physical_region
      end

      if not variant:get_config_options().inner and
         not variant:is_external() and
         flag ~= std.no_access_flag then
        local pr_actions, pr_base_pointers, pr_strides = unpack(data.zip(unpack(
          data.zip(field_paths, field_types):map(
            function(field)
              local field_path, field_type = unpack(field)
              local field_id = field_ids_by_field_path[field_path:hash()]
              return terralib.newlist({
                  physical_region_get_base_pointer(cx, index_type, field_type, field_id, privilege, physical_region)})
        end))))

        physical_region_actions:insertall(pr_actions or {})
        base_pointers:insert(pr_base_pointers)

        for i, field_path in ipairs(field_paths) do
          if privileges_by_field_path[field_path:hash()] ~= "none" then
            base_pointers_by_field_path[field_path:hash()] = pr_base_pointers[i]
            strides_by_field_path[field_path:hash()] = pr_strides[i]
          end
        end
      end
    end

    local actions = quote end

    if not cx.leaf then
      actions = quote
        [actions]
        var [is] = [r].impl.index_space
      end
    end


    if cache_index_iterator then
      actions = quote
        [actions]
        var [it] = c.legion_terra_cached_index_iterator_create(
          [cx.runtime], [cx.context], [r].impl.index_space)
      end
    end

    task_setup:insert(actions)

    for i, field_paths in ipairs(privilege_field_paths) do
      local field_types = privilege_field_types[i]
      local privilege = privileges[i]
      local physical_region = physical_regions[i]
      local physical_region_index = physical_regions_index[i]

      task_setup:insert(quote
        std.assert([physical_region_index] < c_num_regions, "too few physical regions in task setup")
        var [physical_region] = [c_regions][ [physical_region_index] ]
      end)
    end
    task_setup:insertall(physical_region_actions)

    -- Force inner tasks to unmap all regions
    if variant:get_config_options().inner then
      local actions = quote
        c.legion_runtime_unmap_all_regions([cx.runtime], [cx.context])
      end
      task_setup:insertall(actions)
    end

    if not cx:has_ispace(region_type:ispace()) then
      local bounds
      if not region_type:is_opaque() then
        local bounds_actions
        bounds_actions, bounds =
          index_space_bounds(cx, `([r].impl.index_space), region_type:ispace().index_type)
        task_setup:insert(bounds_actions)
      end
      cx:add_ispace_root(region_type:ispace(), is, it, bounds)
    end
    cx:add_region_root(region_type, r,
                       field_paths,
                       privilege_field_paths,
                       privileges_by_field_path,
                       data.dict(data.zip(field_paths:map(data.hash), field_types)),
                       field_ids_by_field_path,
                       data.dict(data.zip(field_paths:map(data.hash), field_types:map(function(_) return false end))),
                       physical_regions_by_field_path,
                       base_pointers_by_field_path,
                       strides_by_field_path)
  end

  for _, list_i in ipairs(std.fn_param_lists_of_regions_by_index(fn_type)) do
    local list_type = param_types[list_i]
    local list = params[list_i]

    local privileges, privilege_field_paths, privilege_field_types =
      std.find_task_privileges(list_type, task)

    local privileges_by_field_path = std.group_task_privileges_by_field_path(
      privileges, privilege_field_paths)

    local field_paths, field_types =
      std.flatten_struct_fields(list_type:fspace())
    local field_ids_by_field_path = data.dict(
      data.zip(field_paths:map(data.hash), param_field_id_symbols[list_i]))

    -- We never actually access physical instances for lists, so don't
    -- build any accessors here.

    cx:add_list_of_regions(list_type, list,
                           field_paths,
                           privilege_field_paths,
                           privileges_by_field_path,
                           data.dict(data.zip(field_paths:map(data.hash), field_types)),
                           field_ids_by_field_path,
                           data.dict(data.zip(field_paths:map(data.hash), field_types:map(function(_) return false end))))
  end

  local preamble = quote
    [emit_debuginfo(node)]
    [task_setup]
  end

  local body
  if node.region_divergence then
    local region_divergence = terralib.newlist()
    local cases
    local diagnostic = quote end
    for _, rs in pairs(node.region_divergence) do
      local r1 = rs[1]
      if cx:has_region(r1) then
        local contained = true
        local rs_cases
        local rs_diagnostic = quote end

        local r1_fields = cx:region(r1).field_paths
        local valid_fields = data.dict(data.zip(r1_fields, r1_fields))
        for _, r in ipairs(rs) do
          if not cx:has_region(r) then
            contained = false
            break
          end
          filter_fields(valid_fields, cx:region(r).field_privileges)
        end

        if contained then
          local r1_bases = cx:region(r1).base_pointers
          for _, r in ipairs(rs) do
            if r1 ~= r then
              local r_base = cx:region(r).base_pointers
              for field, _ in pairs(valid_fields) do
                local r1_base = r1_bases[field:hash()]
                local r_base = r_base[field:hash()]
                assert(r1_base and r_base)
                if rs_cases == nil then
                  rs_cases = `([r1_base] == [r_base])
                else
                  rs_cases = `([rs_cases] and [r1_base] == [r_base])
                  rs_diagnostic = quote
                    [rs_diagnostic]
                    c.printf(["comparing for divergence: regions %s %s field %s bases %p and %p\n"],
                      [tostring(r1)], [tostring(r)], [tostring(field)],
                      [r1_base], [r_base])
                  end
                end
              end
            end
          end

          local group = {}
          for _, r in ipairs(rs) do
            group[r] = true
          end
          region_divergence:insert({group = group, valid_fields = valid_fields})
          if cases == nil then
            cases = rs_cases
          else
            cases = `([cases] and [rs_cases])
          end
          diagnostic = quote
            [diagnostic]
            [rs_diagnostic]
          end
        end
      end
    end

    if cases then
      local div_cx = cx:new_local_scope()
      local body_div = cleanup_after(div_cx, codegen.block(div_cx, node.body))
      local check_div = quote end
      if dynamic_branches_assert then
        check_div = quote
          [diagnostic]
          std.assert(false, ["falling back to slow path in task " .. task.name .. "\n"])
        end
      end

      local nodiv_cx = cx:new_local_scope(region_divergence)
      local body_nodiv = cleanup_after(nodiv_cx, codegen.block(nodiv_cx, node.body))

      body = quote
        if [cases] then
          [body_nodiv]
        else
          [check_div]
          [body_div]
        end
      end
    else
      body = cleanup_after(cx, codegen.block(cx, node.body))
    end
  else
    body = cleanup_after(cx, codegen.block(cx, node.body))
  end

  local result_type = std.type_size_bucket_type(return_type)
  local guard = quote end
  if result_type ~= terralib.types.unit then
    guard = quote
      std.assert_error(false, [get_source_location(node) .. ": missing return statement in task that is expected to return " .. tostring(return_type)])
    end
  end
  local terra proto([c_params]): result_type
    do
      [preamble]; -- Semicolon required. This is not an array access.
      [body]
    end
    [guard]
  end
  proto:setname(tostring(task:get_name()))
  if node.annotations.optimize:is(ast.annotation.Forbid) then
    proto:setoptimized(false)
  end
  variant:set_definition(proto)

  if emergency then proto:compile() end
  manual_gc()

  return task
end

function codegen.top_fspace(cx, node)
  return node.fspace
end

function codegen.top_quote_expr(cx, node)
  return std.newrquote(node)
end

function codegen.top_quote_stat(cx, node)
  return std.newrquote(node)
end

function codegen.top(cx, node)
  if node:is(ast.typed.top.Task) then
    local task = node.prototype

    setup_regent_calling_convention_metadata(node, task)

    if not node.body then return task end

    if not (node.annotations.cuda:is(ast.annotation.Demand) and
            cudahelper.check_cuda_available())
    then
      if node.annotations.cuda:is(ast.annotation.Demand) then
        report.warn(node,
          "ignoring demand pragma at " .. node.span.source ..
          ":" .. tostring(node.span.start.line) ..
          " since the CUDA compiler is unavailable")
      end
      local cpu_variant = task:get_primary_variant()
      task:add_complete_thunk(
        function()
          local cx = context.new_global_scope(cpu_variant)
          return codegen.top_task(cx, node)
      end)
      std.register_variant(cpu_variant)
      return task
    else
      local cpu_variant = task:get_primary_variant()
      task:add_complete_thunk(
        function()
          local cx = context.new_global_scope(cpu_variant)
          return codegen.top_task(cx, node)
      end)
      std.register_variant(cpu_variant)

      local cuda_variant = task:make_variant("cuda")
      cuda_variant:set_is_cuda(true)
      task:add_complete_thunk(
        function()
          local cx = context.new_global_scope(cuda_variant)
          return codegen.top_task(cx, node)
      end)
      std.register_variant(cuda_variant)
      task:set_cuda_variant(cuda_variant)

      return task
    end

  elseif node:is(ast.typed.top.Fspace) then
    return codegen.top_fspace(cx, node)

  elseif node:is(ast.specialized.top.QuoteExpr) then
    return codegen.top_quote_expr(cx, node)

  elseif node:is(ast.specialized.top.QuoteStat) then
    return codegen.top_quote_stat(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function codegen.entry(node)
  local cx = context.new_global_scope()
  return codegen.top(cx, node)
end

return codegen
