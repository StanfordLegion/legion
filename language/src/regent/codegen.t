-- Copyright 2022 Stanford University, NVIDIA Corporation
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
local util = require("regent/ast_util")
local codegen_hooks = require("regent/codegen_hooks")
local gpuhelper = require("regent/gpu/helper")
local data = require("common/data")
local log = require("common/log")
local licm = require("regent/licm")
local openmphelper = require("regent/openmphelper")
local pretty = require("regent/pretty")
local report = require("regent/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local log_codegen = log.make_logger("codegen")
local log_privileges = log.make_logger("privileges")

-- Configuration Variables

-- Setting this flag to true allows the compiler to emit aligned
-- vector loads and stores, and is safe for use only with the general
-- LLR (because the shared LLR does not properly align instances).
local aligned_instances = std.config["aligned-instances"]

-- Setting this flag to true directs the compiler to emit assertions
-- whenever two regions being placed in different physical regions
-- would require the use of the divergence-safe code path to be used.
local dynamic_branches_assert = std.config["no-dynamic-branches-assert"]

-- Setting this flag directs the compiler to emit bounds checks on all
-- pointer accesses. This is independent from the runtime's bounds
-- checks flag as the compiler does not use standard runtime
-- accessors.
local bounds_checks_enabled = std.config["bounds-checks"]
local function needs_bounds_checks(name) return bounds_checks_enabled end
do
  if bounds_checks_enabled then
    local patterns = terralib.newlist()
    for pattern in string.gmatch(std.config["bounds-checks-targets"], "[^,]+") do
      patterns:insert(pattern)
    end
    needs_bounds_checks = function(name)
      return data.any(unpack(patterns:map(function(pattern)
        return string.match(name, pattern) ~= nil
      end)))
    end
  end
end

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

-- Utilities to help generate less garbage from empty quotes.
local empty_quote = quote end

local function as_quote(x)
  if rawequal(x, empty_quote) or (terralib.islist(x) and #x == 0) then
    return empty_quote
  end
  if terralib.isquote(x) then
    return x
  end
  return quote [x] end
end

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

function context:new_local_scope(divergence, must_epoch, must_epoch_point, loop_point, loop_domain, loop_domain_type, break_label)
  assert(not (self.must_epoch and must_epoch))
  divergence = self.divergence or divergence or false
  must_epoch = self.must_epoch or must_epoch or false
  must_epoch_point = self.must_epoch_point or must_epoch_point or false
  loop_point = loop_point or self.loop_point or false
  loop_domain = loop_domain or self.loop_domain or false
  loop_domain_type = loop_domain_type or self.loop_domain_type or false
  if not break_label then
    break_label = self.break_label or false
  end
  assert((must_epoch == false) == (must_epoch_point == false))
  assert((loop_point == false) == (loop_domain == false))
  -- FIXME: we have both loop_point and loop_symbol below, they seem
  -- to mean the same thing but are used by different parts of the compiler.
  return setmetatable({
    variant = self.variant,
    expected_return_type = self.expected_return_type,
    privileges = self.privileges,
    constraints = self.constraints,
    orderings = self.orderings,
    region_usage = self.region_usage,
    task = self.task,
    task_meta = self.task_meta,
    leaf = self.leaf,
    divergence = divergence,
    must_epoch = must_epoch,
    must_epoch_point = must_epoch_point,
    loop_point = loop_point,
    loop_domain = loop_domain,
    loop_domain_type = loop_domain_type,
    break_label = break_label,
    context = self.context,
    runtime = self.runtime,
    result  = self.result,
    ispaces = self.ispaces:new_local_scope(),
    regions = self.regions:new_local_scope(),
    lists_of_regions = self.lists_of_regions:new_local_scope(),
    cleanup_items = terralib.newlist(),
    bounds_checks = self.bounds_checks,
    codegen_contexts = self.codegen_contexts,
    loop_symbol = self.loop_symbol,
    parent = self,
  }, context)
end

function context:new_task_scope(expected_return_type, constraints, orderings, region_usage, leaf,
                                task_meta, task, ctx, runtime, result, bounds_checks)
  assert(expected_return_type and task and ctx and runtime and result)
  return setmetatable({
    variant = self.variant,
    expected_return_type = expected_return_type,
    privileges = data.newmap(),
    constraints = constraints,
    orderings = orderings,
    region_usage = region_usage,
    task = task,
    task_meta = task_meta,
    leaf = leaf,
    divergence = false,
    must_epoch = false,
    must_epoch_point = false,
    loop_point = false,
    loop_domain = false,
    loop_domain_type = false,
    break_label = false,
    context = ctx,
    runtime = runtime,
    result = result,
    ispaces = symbol_table.new_global_scope({}),
    regions = symbol_table.new_global_scope({}),
    lists_of_regions = symbol_table.new_global_scope({}),
    cleanup_items = terralib.newlist(),
    bounds_checks = bounds_checks,
    codegen_contexts = data.newmap(),
    loop_symbol = false,
    parent = false,
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
      if not divergence.valid_fields[field_path] then
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

function context:add_ispace_root(ispace_type, index_space,
                                 domain, bounds)
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
        root_ispace_type = ispace_type,
        domain = domain,
        bounds = bounds,
      }, ispace))
end

function context:add_ispace_subispace(ispace_type, index_space,
                                      parent_ispace_type, domain, bounds)
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
        root_ispace_type = self:ispace(parent_ispace_type).root_ispace_type,
        domain = domain,
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
                                 field_ids, field_id_array, fields_are_scratch, physical_regions,
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
        field_id_array = field_id_array,
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
        field_id_array = self:region(parent_region_type).field_id_array,
        fields_are_scratch = self:region(parent_region_type).fields_are_scratch,
        physical_regions = self:region(parent_region_type).physical_regions,
        base_pointers = self:region(parent_region_type).base_pointers,
        strides = self:region(parent_region_type).strides,
        root_region_type = self:region(parent_region_type).root_region_type,
      }, region))
end

function context:add_codegen_context(name, context)
  self.codegen_contexts[name] = context
end

function context:get_codegen_context(name)
  assert(self.codegen_contexts[name] ~= nil)
  return self.codegen_contexts[name]
end

function context:has_codegen_context(name)
  return self.codegen_contexts[name] ~= nil
end

function context:set_loop_symbol(loop_symbol)
  self.loop_symbol = loop_symbol
end

function region:field_type(field_path)
  local field_type = self.field_types[field_path]
  assert(field_type)
  return field_type
end

function region:field_id(field_path)
  local field_id = self.field_ids[field_path]
  assert(field_id)
  return field_id
end

function region:field_is_scratch(field_path)
  local field_is_scratch = self.fields_are_scratch[field_path]
  assert(field_is_scratch ~= nil)
  return field_is_scratch
end

function region:physical_region(field_path)
  local physical_region = self.physical_regions[field_path]
  assert(physical_region)
  return physical_region
end

function region:base_pointer(field_path)
  local base_pointer = self.base_pointers[field_path]
  assert(base_pointer)
  return base_pointer
end

function region:stride(field_path)
  local stride = self.strides[field_path]
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
                                     field_privileges, field_types,
                                     field_ids, field_id_array, fields_are_scratch)
  if not self.lists_of_regions then
    error("not in task context", 2)
  end
  if self:has_list_of_regions(list_type) then
    error("region " .. tostring(list_type) .. " already defined in this context", 2)
  end
  assert(list_of_logical_regions and
           field_paths and privilege_field_paths and
           field_privileges and field_types and field_ids and field_id_array)
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
        field_id_array = field_id_array,
        fields_are_scratch = fields_are_scratch,
      }, list_of_regions))
end

function list_of_regions:field_type(field_path)
  local field_type = self.field_types[field_path]
  assert(field_type)
  return field_type
end

function list_of_regions:field_id(field_path)
  local field_id = self.field_ids[field_path]
  assert(field_id)
  return field_id
end

function list_of_regions:field_is_scratch(field_path)
  local field_is_scratch = self.fields_are_scratch[field_path]
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
  return as_quote(items)
end

function context:get_all_cleanup_items_for_break()
  local break_label = self.break_label
  assert(break_label)
  local items = terralib.newlist()
  local ctx = self
  while ctx and ctx.break_label == break_label do
    items:insert(ctx:get_cleanup_items())
    ctx = ctx.parent
  end
  return items
end

function context:get_all_cleanup_items_for_return()
  local items = terralib.newlist()
  local ctx = self
  while ctx do
    items:insert(ctx:get_cleanup_items())
    ctx = ctx.parent
  end
  return items
end

local function physical_region_get_base_pointer_setup(index_type, field_type, fastest_index, expected_stride,
                                                      runtime, physical_region, field_id)
  assert(index_type and field_type and runtime and physical_region and field_id)

  local dim = data.max(index_type.dim, 1)
  local elem_type
  if std.is_regent_array(field_type) then
    elem_type = field_type.elem_type
  else
    elem_type = field_type
  end

  local dims = data.range(1, dim + 1)
  local strides = terralib.newlist()
  for i = 1, dim do
    if fastest_index == i then
      strides:insert(expected_stride)
    else
      strides:insert(terralib.newsymbol(c.size_t, "stride" .. tostring(i)))
    end
  end

  local get_accessor = c["legion_physical_region_get_field_accessor_array_" .. tostring(dim) .. "d"]
  local destroy_accessor = c["legion_accessor_array_" .. tostring(dim) .. "d_destroy"]
  local raw_rect_ptr = c["legion_accessor_array_" .. tostring(dim) .. "d_raw_rect_ptr"]

  local rect_t = c["legion_rect_" .. tostring(dim) .. "d_t"]
  local domain_get_bounds = c["legion_domain_get_bounds_" .. tostring(dim) .. "d"]

  local base_pointer
  local p_base_pointer
  local num_fields
  local actions
  if std.is_regent_array(field_type) then
    base_pointer = terralib.newsymbol((&elem_type)[field_type.N], "base_pointer")
    p_base_pointer = terralib.newsymbol(&&elem_type, "p_base_pointer")
    num_fields = field_type.N
    actions = quote
      var [base_pointer]
      var [p_base_pointer] = [&&elem_type]([base_pointer])
    end
  else
    base_pointer = terralib.newsymbol(&field_type, "base_pointer")
    p_base_pointer = terralib.newsymbol(&&field_type, "p_base_pointer")
    num_fields = 1
    actions = quote
      var [base_pointer]
      var [p_base_pointer] = &[base_pointer]
    end
  end

  actions = quote
    [actions];
    [dims:map(function(i)
      if fastest_index ~= i then
        return quote var [ strides[i] ] end
      else
        return empty_quote
      end
    end)];
    for idx = 0, [num_fields] do
      var accessor = [get_accessor](physical_region, field_id + [idx])

      var region = c.legion_physical_region_get_logical_region([physical_region])
      var domain = c.legion_index_space_get_domain([runtime], region.index_space)
      var rect = [domain_get_bounds](domain)

      var subrect : rect_t
      var offsets : c.legion_byte_offset_t[dim]
      [p_base_pointer][idx] =
        [&elem_type]([raw_rect_ptr](accessor, rect, &subrect, &(offsets[0])))

      -- Sanity check the outputs.
      std.assert([p_base_pointer][idx] ~= nil or c.legion_domain_get_volume(domain) <= 1,
                 "base pointer is nil")
      [data.range(dim):map(
         function(i)
           return quote
             std.assert(subrect.lo.x[i] == rect.lo.x[i], "subrect not equal to rect")
             std.assert(subrect.hi.x[i] == rect.hi.x[i], "subrect not equal to rect")
           end
         end)]

      std.assert(offsets[ [fastest_index - 1] ].offset == [expected_stride] or
                 c.legion_domain_get_volume(domain) <= 1,
                 "stride does not match expected value")

      -- Fix up the base pointer so it points to the origin (zero),
      -- regardless of where rect is located. This allows us to do
      -- pointer arithmetic later oblivious to what sort of a subrect
      -- we are working with.
      [data.range(dim):map(
         function(i)
           return quote
             [p_base_pointer][idx] = [&elem_type](([&int8]([p_base_pointer][idx])) - rect.lo.x[i] * offsets[i].offset)
           end
         end)]

      [dims:map(
         function(i)
           if fastest_index ~= i then
             return quote [ strides[i] ] = offsets[i-1].offset end
           else
             return empty_quote
           end
         end)]
      [destroy_accessor](accessor)
    end
  end
  return actions, base_pointer, strides
end

local physical_region_get_base_pointer_thunk = data.weak_memoize(
  function(index_type, field_type, fastest_index, expected_stride)
    assert(index_type and field_type)

    local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
    local physical_region = terralib.newsymbol(c.legion_physical_region_t, "physical_region")
    local field_id = terralib.newsymbol(c.legion_field_id_t, "field_id")

    local actions, base_pointer, strides = physical_region_get_base_pointer_setup(
      index_type, field_type, fastest_index, expected_stride, runtime, physical_region, field_id)

    local terra get_base_pointer([runtime], [physical_region], [field_id])
      [actions]
      return [base_pointer], [strides]
    end
    get_base_pointer:setinlined(false)
    return terralib.newlist({get_base_pointer, strides})
  end)

local function physical_region_get_base_pointer(cx, region_type, index_type, field_type,
                                                field_path, physical_region, field_id)
  local fastest_index = 1
  local expected_stride
  if std.is_regent_array(field_type) then
    expected_stride = terralib.sizeof(field_type.elem_type)
  else
    expected_stride = terralib.sizeof(field_type)
  end
  if cx.orderings[region_type] and cx.orderings[region_type][field_path] then
    local ordering, stride = unpack(cx.orderings[region_type][field_path])
    assert(#ordering > 0)
    fastest_index = ordering[1]
    expected_stride = stride
  end
  -- FIXME: The opt-compile-time code path improves compile time and
  -- has the same runtime performance, but has potential issues on
  -- non-x86 due to its use of an aggregate return value, so we can't
  -- make it the default just yet.
  if std.config["opt-compile-time"] then
    local thunk, expected_strides = unpack(physical_region_get_base_pointer_thunk(
      index_type, field_type, fastest_index, expected_stride))

    local base_pointer
    if std.is_regent_array(field_type) then
      base_pointer = terralib.newsymbol((&field_type.elem_type)[field_type.N], "base_pointer")
    else
      base_pointer = terralib.newsymbol(&field_type, "base_pointer")
    end
    local computed_strides = data.mapi(
      function(i, _)
        return terralib.newsymbol(c.size_t, "stride" .. tostring(i))
      end,
      expected_strides)
    -- In order to ensure constant folding, forward any expected
    -- strides with constant values.
    local result_strides = data.mapi(
      function(i, stride)
        if type(stride) == "number" and terralib.isintegral(stride) then
          return stride
        else
          return computed_strides[i]
        end
      end,
      expected_strides)

    local actions = quote
      var [base_pointer], [computed_strides] = [thunk](
        [cx.runtime], [physical_region], [field_id])
    end

    return actions, base_pointer, result_strides
  else
    return physical_region_get_base_pointer_setup(
      index_type, field_type, fastest_index, expected_stride, cx.runtime, physical_region, field_id)
  end
end

local function index_space_bounds(cx, is, is_type, domain, bounds)
  local index_type = is_type.index_type

  if not domain then
    domain = terralib.newsymbol(c.legion_domain_t, "domain_" .. tostring(is_type))
  end

  if not bounds then
    bounds = false
  end
  local actions = quote
    var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
  end

  if not index_type:is_opaque() then
    if not bounds then
      bounds = terralib.newsymbol(std.rect_type(index_type), "bounds_" .. tostring(is_type))
    end

    local bounds_actions = nil
    if index_type.dim == 1 then
      bounds_actions = quote
        [bounds].lo = [domain].rect_data[0]
        [bounds].hi = [domain].rect_data[1]
      end
    else
      bounds_actions = empty_quote
      local idx = 0
      local fields = terralib.newlist { "lo", "hi" }
      fields:map(function(field)
        index_type.impl_type:getentries():map(function(entry)
          bounds_actions = quote
            [bounds_actions]
            [bounds].[field].__ptr.[entry.field] = [domain].rect_data[ [idx] ]
          end
          idx = idx + 1
        end)
      end)
    end
    actions = quote
      [actions]
      var [bounds]
      do [bounds_actions] end
    end
  end
  return actions, domain, bounds
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
    return empty_quote
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
  local actions = quote
    [region_expr.actions]
    var [r] = [std.implicit_cast(
                 static_region_type, region_type, region_expr.value)]
    var [lr] = [r].impl
    var [is] = [lr].index_space
  end

  local parent_region_type = std.search_constraint_predicate(
    cx, region_type, {},
    function(cx, region)
      return cx:has_region(region)
    end)
  if not parent_region_type then
    error("failed to find appropriate for region " .. tostring(region_type) .. " in unpack", 2)
  end

  local bounds_actions, domain, bounds =
    index_space_bounds(cx, is, region_type:ispace())
  actions = quote [actions]; [bounds_actions] end

  cx:add_ispace_subispace(region_type:ispace(), is,
                          parent_region_type:ispace(), domain, bounds)
  cx:add_region_subregion(region_type, r, parent_region_type)

  return expr.just(actions, r)
end

-- Utilities for importing raw C/C++ handles to Regent

-- == sha256sum("import")[0:7]
local IMPORT_SEMANTIC_TAG   = 0x68e67653
-- == sha256sum("import")[8:15]
local IMPORT_SEMANTIC_VALUE = 0xd93a28e2

local function tag_imported(cx, handle)
  local attach = nil
  local handle_type = handle.type
  if handle_type == std.c.legion_logical_region_t then
    attach = std.c.legion_logical_region_attach_semantic_information
  else
    assert(false, "unreachable")
  end
  return quote
    do
      var result : uint32 = [IMPORT_SEMANTIC_VALUE]
      var result_size : uint64 = [sizeof(uint32)]
      [attach]([cx.runtime], [handle], [IMPORT_SEMANTIC_TAG],
        [&opaque](&result), result_size, false)
    end
  end
end

local function check_imported(cx, node, handle)
  local retrieve = nil
  local handle_type = handle.type
  if handle_type == std.c.legion_logical_region_t then
    retrieve = std.c.legion_logical_region_retrieve_semantic_information
  else
    assert(false, "unreachable")
  end
  return quote
    do
      var result : &opaque
      var result_size : uint64 = 0
      [retrieve]([cx.runtime], [handle], [IMPORT_SEMANTIC_TAG],
        &result, &result_size, true, true)
      std.assert_error(result_size == 0,
        [get_source_location(node) ..
          ": cannot import a handle that is already imported"])
    end
  end
end

local function eq_struct(st, a, b)
  local expr = `(true)
  local entries = st:getentries()
  for _, entry in ipairs(entries) do
    local entry_name = entry[1] or entry.field
    assert(type(entry_name) == "string")
    local entry_type = entry[2] or entry.type
    if entry_type:isstruct() then
      expr = `([expr] and [eq_struct(entry_type, `([a].[entry_name]), `([b].[entry_name]))])
    else
      expr = `([expr] and ([a].[entry_name] == [b].[entry_name]))
    end
  end
  return expr
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

function value:address(cx)
  assert(false)
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
  elseif std.is_index_type(std.as_read(value_type)) then
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
  if cx.bounds_checks and value_type:isarray() then
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
      as_quote(actions),
      `([value_type:data(value_expr.value)][ [index.value] ]))
  else
    result = expr.just(
      as_quote(actions),
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
    for _, entry in ipairs(value_type:getentries()) do
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

local aref = setmetatable({}, { __index = value })
aref.__index = aref

function values.ref(node, value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or
    not (std.is_bounded_type(value_type) or std.is_vptr(value_type)) then
    error("ref requires a legion ptr type", 2)
  end
  local meta
  if std.is_regent_array(std.as_read(node.expr_type)) then
    meta = aref
  else
    meta = ref
  end
  return setmetatable(values.value(node, value_expr, value_type, field_path), meta)
end

function ref:new(node, value_expr, value_type, field_path)
  return values.ref(node, value_expr, value_type, field_path)
end

local function get_element_pointer(cx, node, region_types, index_type, field_type,
                                   base_pointer, strides, field_path, index)
  if cx.bounds_checks then
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

  local ordering
  local expected_stride

  for i = 1, #region_types do
    local region_type = region_types[i]
    if cx.orderings[region_type] and cx.orderings[region_type][field_path] then
      local region_ordering, stride = unpack(cx.orderings[region_type][field_path])
      assert(#region_ordering > 0)
      if ordering then
        --- TODO: all bounds in a bounded type must have the same layout for now
        data.zip(ordering, region_ordering):map(function(pair)
          local o1, o2 = unpack(pair)
          assert(o1 == o2)
        end)
        assert(expected_stride == stride)
      else
        ordering = region_ordering
        expected_stride = stride
      end
    else
      ordering = std.layout.make_index_ordering_from_constraint(
          std.layout.default_layout(region_type:ispace().index_type))
      expected_stride = terralib.sizeof(field_type)
    end
  end
  assert(#ordering == ((not index_type.fields and 1) or #index_type.fields))

  -- Note: This code is performance-critical and tends to be sensitive
  -- to small changes. Please thoroughly performance-test any changes!
  local index_value
  if std.is_bounded_type(index_type) then
    index_value = `([index].__ptr.__ptr)
  elseif std.is_index_type(index_type) then
    index_value = `([index].__ptr)
  else
    assert(false)
  end

  if expected_stride == terralib.sizeof(field_type) then
    if not index_type.fields then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      return `(@[&field_type](&base_pointer[ [index_value] ]))
    elseif #index_type.fields == 1 then
      -- Assumes stride[1] == terralib.sizeof(field_type)
      local field = index_type.fields[ ordering[1] ]
      return `(@[&field_type](&base_pointer[ [index_value].[field] ]))
    else
      local offset
      for i, field in ipairs(index_type.fields) do
        if offset then
          offset = `(offset + [index_value].[ field ] * [ strides[i] ])
        else
          offset = `([index_value].[ field ] * [ strides[i] ])
        end
      end
      return `(@([&field_type]([&int8](base_pointer) + offset)))
    end
  else
    -- Assumes more than one field contiguously locate
    local access_type = int8[expected_stride]
    if not index_type.fields then
      return `(@[&field_type](&([&access_type](base_pointer))[ [index_value] ]))
    elseif #index_type.fields == 1 then
      local field = index_type.fields[ ordering[1] ]
      return `(@[&field_type](&([&access_type](base_pointer))[ [index_value].[field] ]))
    else
      local offset
      for i, field in ipairs(index_type.fields) do
        if offset then
          offset = `(offset + [index_value].[ field ] * [ strides[i] ])
        else
          offset = `([index_value].[ field ] * [ strides[i] ])
        end
      end
      return `(@([&field_type]([&int8](base_pointer) + offset)))
    end
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
        return terralib.newsymbol(&field_type, "base_pointer_" .. tostring(field_path))
      end)
    strides = absolute_field_paths:map(
      function(field_path)
        return cx:region(region_types[1]):stride(field_path):map(
          function(_)
            return terralib.newsymbol(c.size_t, "stride_" .. tostring(field_path))
          end)
      end)

    local cases
    for i = #region_types, 1, -1 do
      local region_base_pointers = base_pointers_by_region[i]
      local region_strides = strides_by_region[i]
      local case_ = data.zip(base_pointers, region_base_pointers, strides, region_strides):map(
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
            [case_]
          else
            [cases]
          end
        end
      else
        cases = case_
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
    values = data.zip(field_types, base_pointers, strides, absolute_field_paths):map(
      function(field)
        local field_type, base_pointer, stride, field_path = unpack(field)
        return get_element_pointer(cx, self.node, region_types, self.value_type, field_type, base_pointer, stride, field_path, value)
      end)
  else
    assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
    values = data.zip(field_types, base_pointers, strides, absolute_field_paths):map(
      function(field)
        local field_type, base_pointer, stride, field_path = unpack(field)
        local vec
        if std.type_eq(field_type, std.ptr) then
          assert(std.is_vptr(expr_type))
          vec = expr_type.vec_type
        else
          vec = vector(field_type, std.as_read(expr_type).N)
        end
        return `(@[&vec](&[get_element_pointer(cx, self.node, region_types, self.value_type, field_type, base_pointer, stride, field_path, value)]))
      end)
    value_type = expr_type
  end

  return actions, values, value_type, field_paths, field_types
end

function ref:address(cx)
  return values.value(self.node, self.expr, self.value_type)
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
           if std.is_vptr(expr_type) and std.type_eq(field_type, std.ptr) then
            result = `([result].value)
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
         if std.is_vptr(expr_type) and std.type_eq(field_type, std.ptr) then
           result = `([result].__ptr.value)
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
  return expr.just(actions, empty_quote)
end

local reduction_fold = {
  ["+"] = "+",
  ["-"] = "-",
  ["*"] = "*",
  ["/"] = "/", -- FIXME: Need to fold with "/" for RW instances.
  ["max"] = "max",
  ["min"] = "min",
}

function ref:reduce(cx, value, op, expr_type, atomic)
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
         -- TODO: Users should be able to override atomic reduction operator for non-primitive types
         elseif expr_type:isarray() then
           local N = expr_type.N
           if cx.variant:is_openmp() and atomic then
             return quote
               for i = 0, N do
                 [openmphelper.generate_atomic_update(fold_op, expr_type.type)](&([field_value][i]), result[i])
               end
             end
           elseif cx.variant:is_cuda() and atomic then
             return quote
               for i = 0, N do
                 [gpuhelper.generate_atomic_update(fold_op, expr_type.type)](&([field_value][i]), result[i])
               end
             end
           else
             return quote
               for i = 0, N do
                 [field_value][i] = [std.quote_binary_op(fold_op, `(field_value[i]), `(result[i]))]
               end
             end
           end
         else
           if cx.variant:is_openmp() and atomic then
             return quote
               [openmphelper.generate_atomic_update(fold_op, value_type)](&[field_value], result)
             end
           elseif cx.variant:is_cuda() and atomic then
             return quote
               [gpuhelper.generate_atomic_update(fold_op, value_type)](&[field_value], result)
             end
           else
             return quote
               [field_value] = [std.quote_binary_op(fold_op, field_value, result)]
             end
           end
         end
      end)]
  end
  return expr.just(actions, empty_quote)
end

function ref:get_field(cx, node, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  if value_type:isstruct() and value_type.__no_field_slicing then
    local value_actions, value = result:__ref(cx)
    assert(#value == 1)
    result = values.rawref(result.node, expr.just(value_actions, value[1]), &value_type)
  end
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
  if cx.bounds_checks and value_type:isarray() then
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  end
  assert(not std.is_list(value_type)) -- Shouldn't be an l-value anyway.
  local result = expr.just(as_quote(actions), `([value][ [index.value] ]))
  return values.rawref(node, result, &result_type, data.newtuple())
end

function aref:__ref(cx, index)
  assert(index ~= nil)
  local actions = self.expr.actions
  local value = self.expr.value

  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))
  local field_type = value_type
  local absolute_field_path = self.field_path

  local region_types = self.value_type:bounds()
  local base_pointers_by_region = region_types:map(
    function(region_type)
      return cx:region(region_type):base_pointer(absolute_field_path)
    end)
  local strides_by_region = region_types:map(
    function(region_type)
      return cx:region(region_type):stride(absolute_field_path)
    end)

  local base_pointer, strides
  local field_paths = terralib.newlist { absolute_field_path }
  if cx.check_divergence(region_types, field_paths) or #region_types == 1 then
    base_pointer = base_pointers_by_region[1]
    strides = strides_by_region[1]
  else
    base_pointer = terralib.newsymbol(&field_type, "base_pointer_" .. tostring(absolute_field_path))
    strides = cx:region(region_types[1]):stride(absolute_field_path):map(
      function(_)
        return terralib.newsymbol(c.size_t, "stride_" .. tostring(field_path))
      end)

    local cases
    for i = #region_types, 1, -1 do
      local region_base_pointer = base_pointers_by_region[i]
      local region_strides = strides_by_region[i]

      local case_ = quote
        [base_pointer] = [region_base_pointer]
      end
      for i, stride in ipairs(strides) do
        local region_stride = region_strides[i]
        setup = quote [case_]; [stride] = [region_stride] end
      end

      if cases then
        cases = quote
          if [value].__index == [i] then
            [case_]
          else
            [cases]
          end
        end
      else
        cases = case_
      end
    end

    actions = quote
      [actions];
      var [base_pointer];
      [strides:map(
         function(stride) return quote var [stride] end end)];
      [cases]
    end
  end

  base_pointer = `([base_pointer][ [index] ])

  value = get_element_pointer(cx, self.node, region_types, self.value_type, field_type.elem_type, base_pointer, strides, absolute_field_path, value)

  return actions, value
end

function aref:get_index(cx, node, index, result_type)
  local value_actions, value = self:__ref(cx, index.value)

  local actions = terralib.newlist({value_actions, index.actions})
  local value_type = std.as_read(self.node.expr_type)
  if cx.bounds_checks then
    actions:insert(
      quote
        std.assert_error([index.value] >= 0 and [index.value] < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end)
  end
  assert(not std.is_list(value_type)) -- Shouldn't be an l-value anyway.
  local result = expr.just(as_quote(actions), value)
  return values.rawref(node, result, &result_type, data.newtuple())
end

function aref:address(cx)
  return values.value(self.node, self.expr, self.value_type)
end

function aref:read(cx, expr_type)
  local value_type = std.as_read(self.node.expr_type)
  assert(std.type_eq(value_type, std.as_read(expr_type)))

  local value = terralib.newsymbol(value_type)
  local index = terralib.newsymbol(int32)
  local elem_actions, elem_value = self:__ref(cx, index)
  local actions = quote
    var [value];
    for [index] = 0, [value_type.N] do
      [elem_actions];
      [value].impl[ [index] ] = [elem_value]
    end
  end

  return expr.just(actions, value)
end

function aref:write(cx, value, expr_type)
  local value_type = std.as_read(self.node.expr_type)
  assert(std.type_eq(value_type, std.as_read(expr_type)))
  local index = terralib.newsymbol(int32)
  local value_expr = value:read(cx, expr_type)
  local elem_actions, elem_value = self:__ref(cx, index)
  local actions = quote
    [value_expr.actions];
    for [index] = 0, [value_type.N] do
      [elem_actions];
      [elem_value] = [value_expr.value].impl[ [index] ]
    end
  end

  return expr.just(actions, empty_quote)
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

function vref:address(cx)
  assert(false)
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
    for field_idx, field_path in ipairs(field_paths) do
      for vector_idx = 1, vector_width do
        local v = vars[vector_idx]
        for _, field_name in ipairs(field_path) do
          v = `([v].[field_name])
        end
        local cases
        for region_idx = #base_pointers_by_region, 1, -1 do
          local base_pointer = base_pointers_by_region[region_idx][field_idx]
          local case_ = quote
              v = base_pointer[ [vref_value].__ptr.value[ [vector_idx - 1] ] ]
          end

          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case_]
              else
                [cases]
              end
            end
          else
            cases = case_
          end
        end
        actions = quote [actions]; [cases] end
      end
    end
  end

  actions = quote
    [actions];
    var [value]
    [data.zip(field_paths, field_types):map(
       function(pair)
         local field_path, field_type = unpack(pair)
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
         if std.is_vptr(expr_type) and std.type_eq(field_type, ptr) then
           result = `([result].value)
           field_accesses = field_accesses:map(function(field_access)
             return `([field_access].__ptr.value)
           end)
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
    for field_idx, field_path in ipairs(field_paths) do
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
          local case_ = quote
            base_pointer[ [vref_value].__ptr.value[ [vector_idx - 1] ] ] =
              result
          end

          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case_]
              else
                [cases]
              end
            end
          else
            cases = case_
          end
        end
        actions = quote [actions]; [cases] end
      end
    end

  end
  return expr.just(actions, empty_quote)
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
    for field_idx, field_path in ipairs(field_paths) do
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
          local case_ = quote
            [field_value] =
              [std.quote_binary_op(fold_op, field_value, result)]
          end
          if cases then
            cases = quote
              if [vref_value].__index[ [vector_idx - 1] ] == [region_idx] then
                [case_]
              else
                [cases]
              end
            end
          else
            cases = case_
          end
        end
        actions = quote [actions]; [cases] end
      end
    end

  end

  return expr.just(actions, empty_quote)
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

function rawref:address(cx)
  local value_expr = self:__ref(cx)
  local actions = value_expr.actions
  local result = `(&[value_expr.value])
  return values.value(self.node, expr.just(actions, result), self.value_type)
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
  return expr.just(actions, empty_quote)
end

function rawref:reduce(cx, value, op, expr_type, atomic)
  local ref_expr = self:__ref(cx)
  local cleanup = make_cleanup_item(cx, ref_expr.value, self.value_type.type)
  local value_expr = value:read(cx)

  local ref_type = std.get_field_path(self.value_type.type, self.field_path)
  local value_type = std.as_read(value.value_type)

  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
  end

  local fold_op = reduction_fold[op]

  if cx.variant:is_openmp() and atomic then
    actions = quote
      [actions];
      [openmphelper.generate_atomic_update(fold_op, self.value_type.type)](&[ref_expr.value], [value_expr.value])
      [cleanup];
    end
  elseif cx.variant:is_cuda() and atomic then
    actions = quote
      [actions];
      [gpuhelper.generate_atomic_update(fold_op, self.value_type.type)](&[ref_expr.value], [value_expr.value])
      [cleanup];
    end
  else
    local reduce = ast.typed.expr.Binary {
      op = op,
      lhs = ast.typed.expr.Internal {
        value = values.value(self.node, expr.just(empty_quote, ref_expr.value), ref_type),
        expr_type = ref_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      rhs = ast.typed.expr.Internal {
        value = values.value(value.node, expr.just(empty_quote, value_expr.value), value_type),
        expr_type = value_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      expr_type = ref_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
    }

    local reduce_expr = codegen.expr(cx, reduce):read(cx, ref_type)

    actions = quote
      [actions];
      [reduce_expr.actions];
      [cleanup];
      [ref_expr.value] = [reduce_expr.value]
    end
  end
  return expr.just(actions, empty_quote)
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
  if cx.bounds_checks and value_type:isarray() then
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
      as_quote(actions),
      `([value_type:data(ref_expr.value)][ [index.value] ]))
  elseif std.is_regent_array(value_type) then
    result = expr.just(
      as_quote(actions),
      `([ref_expr.value].impl[ [index.value] ]))
  else
    result = expr.just(
      as_quote(actions),
      `([ref_expr.value][ [index.value] ]))
  end
  return values.rawref(node, result, &result_type, data.newtuple())
end

-- A helper for capturing debug information.
local function emit_debuginfo(node)
  assert(node.span.source and node.span.start.line)
  if not std.config["debuginfo"] or string.len(node.span.source) == 0 then
    return empty_quote
  end
  return quote
    terralib.debuginfo(node.span.source, node.span.start.line)
  end
end

local function get_provenance(node)
  return node.span.source .. ":" .. node.span.start.line
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

function codegen.expr_global(cx, node)
  local value = node.value
  local value_type = std.as_read(node.expr_type)

  assert(terralib.isglobalvar(value))
  assert(std.type_eq(value.type, value_type))

  return values.value(
    node,
    expr.just(emit_debuginfo(node), value),
    value_type)
end

function codegen.expr_function(cx, node)
  local value_type = std.as_read(node.expr_type)
  local value = node.value
  if std.is_math_fn(value) then
    value = value:get_definition()
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
  elseif std.is_ispace(value_type) and field_name == "bounds" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)
    assert(std.is_rect_type(expr_type))
    -- XXX: This line must run after the code generation for 'node.value' is done,
    --      because the index space is not yet registered to the context until then.
    local bounds = cx:ispace(value_type).bounds

    local actions
    -- If the expression has the form 'region_name.ispace.bounds',
    -- we skip the code generation for the first field access.
    if node.value:is(ast.typed.expr.FieldAccess) and
       node.value.field_name == "ispace" and
       node.value.value:is(ast.typed.expr.ID)
    then
      actions = emit_debuginfo(node)
    -- Otherwise, we fall back to the general case. For example,
    -- if the expression is 'partition_name[color].ispace.bounds',
    -- we must go through the normal generation for the first field access.
    else
      actions = quote
        [value.actions]
        [emit_debuginfo(node)]
      end
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
      std.assert(not c.legion_index_space_has_multiple_domains([cx.runtime], [value.value].impl),
        "\"volume\" field isn't supported on index spaces with multiple domains")
      var [volume] = c.legion_domain_get_volume(
        c.legion_index_space_get_domain([cx.runtime], [value.value].impl))
    end
    return values.value(
      node,
      expr.just(actions, volume),
      expr_type)
  elseif (std.is_partition(value_type) or std.is_cross_product(value_type)) and field_name == "colors" then
    local value = codegen.expr(cx, node.value):read(cx)
    local expr_type = std.as_read(node.expr_type)
    local is = terralib.newsymbol(c.legion_index_space_t, "colors")
    local actions = quote
      [value.actions]
      var [is] =
        c.legion_index_partition_get_color_space([cx.runtime],
                                                 [value.value].impl.index_partition)
    end

    if not cx:has_ispace(expr_type) then
      local bounds_actions, domain, bounds = index_space_bounds(cx, is, expr_type)
      actions = quote
        [actions];
        [bounds_actions]
      end

      cx:add_ispace_root(
        expr_type,
        is,
        domain,
        bounds)
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

    local bounds_actions, domain, bounds =
      index_space_bounds(cx, is, expr_type:ispace())
    actions = quote [actions]; [bounds_actions] end

    cx:add_ispace_subispace(expr_type:ispace(), is,
                            parent_region_type:ispace(), domain, bounds)
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
      actions = quote
        [actions]
        var dp = [color]:to_domain_point()
        var [lr] = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
          [cx.runtime], [value.value].impl, dp)
        var [is] = [lr].index_space
        var [r] = [region_type] { impl = [lr] }
      end

      local bounds_actions, domain, bounds =
        index_space_bounds(cx, is, region_type:ispace())
      actions = quote [actions]; [bounds_actions] end

      cx:add_ispace_subispace(region_type:ispace(), is,
                              parent_region_type:ispace(), domain, bounds)
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
        [cx.runtime],
        [value.value].product, dp)
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [lr], [ip])
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
          local bounds_actions, domain, bounds =
            index_space_bounds(cx, `([region.value].impl.index_space),
                               region_type:ispace())
          region.actions = quote [region.actions]; [bounds_actions] end
          cx:add_ispace_root(
            region_type:ispace(),
            `([region.value].impl.index_space),
            domain,
            bounds)
          cx:add_region_root(
            region_type, region.value,
            cx:list_of_regions(value_type).field_paths,
            cx:list_of_regions(value_type).privilege_field_paths,
            cx:list_of_regions(value_type).field_privileges,
            cx:list_of_regions(value_type).field_types,
            cx:list_of_regions(value_type).field_ids,
            cx:list_of_regions(value_type).field_id_array,
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
            cx:list_of_regions(value_type).field_id_array,
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

        var size = terralib.sizeof([list_type.element_type]) * [index.value].__size
        var data = c.malloc(size)
        std.assert(size == 0 or data ~= nil, "malloc failed in index_access")
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
          cx:list_of_regions(value_type).field_id_array,
          cx:list_of_regions(value_type).fields_are_scratch)
      end
      return values.value(node, expr.just(actions, list), list_type)
    end
  elseif std.is_region(value_type) then
    -- We still need to do codegen for the value to get the metadata correct
    local value = codegen.expr(cx, node.value)
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
    return values.ref(node, pointer, pointer_type, nil, std.is_bounded_type(index_type))
  elseif std.is_transform_type(value_type) then
    local value = codegen.expr(cx, node.value):read(cx, value_type)
    local index = codegen.expr(cx, node.index):read(cx, index_type)
    local actions = quote
      [value.actions];
      [index.actions];
      [emit_debuginfo(node)]
    end
    if cx.bounds_checks then
      actions = quote
        [actions];
        std.assert_error([index].value.__ptr.x >= 0 and [index].value.__ptr.x < [value_type.M],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
        std.assert_error([index].value.__ptr.y >= 0 and [index].value.__ptr.y < [value_type.N],
          [get_source_location(node) .. ": array access to " .. tostring(value_type) .. " is out-of-bounds"])
      end
    end
    return values.rawref(node, expr.just(actions,
          `([value].value.impl.trans[ [index].value.__ptr.x ][ [index].value.__ptr.y ])), &expr_type)
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
      if size_actions then
        task_args_setup:insert(size_actions)
        task_args_setup:insert(quote [size] = [size] + [size_value] end)
      end
   end
  end

  task_args_setup:insert(quote
    -- Note: it's important to use calloc here because otherwise some
    -- of the padding in the argument buffer may be garbage. In most
    -- cases, this is a non-issue, but it can create spurious errors
    -- with Legion's DCR safety check.
    var [buffer] = c.calloc(1, [size])
    std.assert([size] == 0 or [buffer] ~= nil, "calloc failed in setup task args")
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
               params_map_value[(uint64([i])-1)/64] =
                 params_map_value[(uint64([i])-1)/64] + (uint64(1) << ((uint64([i])-1)%64))
             end
           end
           return empty_quote
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

    local arg_field_id_array = cx:region_or_list(arg_type).field_id_array
    local param_field_id_array = param_field_ids[i]

    local arg_fs_type = arg_type:fspace()
    local param_fs_type = param_type:fspace()

    if std.type_eq(arg_fs_type, param_fs_type) then
      task_args_setup:insert(
        quote [fixed_ptr].[param_field_id_array] = [arg_field_id_array] end)
    else
      -- If we hit this branch, it means that the region argument was created by
      -- a projection, so we need to copy field ids individually due to potential
      -- field reordering.

      local arg_field_paths = std.flatten_struct_fields(arg_fs_type)
      local field_id_index = data.dict(data.zip(arg_field_paths,
                                                data.range(0, #arg_field_paths)))
      local param_field_paths = std.flatten_struct_fields(param_fs_type)
      assert(#param_field_paths == #arg_field_paths)
      local field_id_array_type = std.get_field(params_struct_type, param_field_id_array)
      assert(field_id_array_type.N == #arg_field_paths)
      local terra copy_field_ids([fixed_ptr], arg_field_id_array : &field_id_array_type)
        [data.zip(param_field_paths, data.range(0, #param_field_paths)):map(
            function(pair)
              local param_field_path, param_i = unpack(pair)
              local arg_i = field_id_index[param_field_path]
              return
                quote [fixed_ptr].[param_field_id_array][ [param_i] ] =
                      (@arg_field_id_array)[ [arg_i] ]
                end
            end)]
      end
      copy_field_ids:setinlined(false)
      task_args_setup:insert(quote [copy_field_ids]([fixed_ptr], &[arg_field_id_array]) end)
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

local function is_identity_projection(expr, loop_index)
  if expr:is(ast.typed.expr.Projection) then
    expr = expr.region
  end
  assert(expr:is(ast.typed.expr.IndexAccess))

  -- Strip the index for the purpose of checking if this is the
  -- identity projection functor.
  local stripped_index = strip_casts(expr.index)
  return stripped_index:is(ast.typed.expr.ID) and stripped_index.value == loop_index
end

local function wrap_partition_internal(node, parent)
  return node {
    value = ast.typed.expr.Internal {
      value = values.value(
        node.value,
        expr.just(empty_quote, { impl = parent }),
        node.value.expr_type),
      expr_type = node.value.expr_type,
      annotations = node.annotations,
      span = node.span
    }
  }
end

local function make_partition_projection_functor(cx, expr, loop_index, color_space,
                                                 free_vars, free_vars_setup, requirement)
  cx = cx:new_local_scope()

  if expr:is(ast.typed.expr.Projection) then
    expr = expr.region
  end
  assert(expr:is(ast.typed.expr.IndexAccess))

  -- Never return 0 for cross products
  if is_identity_projection(expr, loop_index) and
     std.is_partition(std.as_read(util.get_base_indexed_node(expr).expr_type))
  then
    return 0 -- Identity projection functor.
  end

  local index = expr.index
  local index_type = std.as_read(index.expr_type)

  local point = terralib.newsymbol(c.legion_domain_point_t, "point")

  local symbol_type = loop_index:gettype()
  local symbol = loop_index:getsymbol()
  local symbol_setup
  if std.is_bounded_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]({ __ptr = [symbol_type.index_type]([point]) })
    end
  elseif std.is_index_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]([point])
    end
  else
    -- Otherwise symbol_type has to be some simple integral type.
    assert(symbol_type:isintegral())
    symbol_setup = quote
      var [symbol] = [int1d]([point])
    end
  end

  -- Hack: Rooting any ispaces present manually
  if free_vars then
    for _, symbol in free_vars:keys() do
      local symbol_type = symbol:gettype()
      if cx:has_ispace(symbol_type) then
        local ispace = cx:ispace(symbol_type)
        local bounds_actions, _ignore1, _ignore2 = index_space_bounds(cx, ispace.index_space, symbol_type, ispace.domain, ispace.bounds)
        free_vars_setup:insert(
          quote
            var [ispace.index_space] = [symbol:getsymbol()].impl
            [bounds_actions];
          end)
      end
    end
  end

  -- Generate a projection functor that evaluates `expr`.
  local value = codegen.expr(cx, index):read(cx)

  if requirement and free_vars_setup then
    free_vars_setup:insert(as_quote(value.actions))

    local parent = terralib.newsymbol(c.legion_logical_partition_t, "parent")
    local base_type = std.as_read(util.get_base_indexed_node(expr).expr_type)
    local depth = 0
    if std.is_partition(base_type) then
      expr = wrap_partition_internal(expr, parent)
    else
      -- No wrap_partition_internal in this case because we capture
      -- the cross-product as a closure, rather than getting it
      -- through the projection functor arguments.
      assert(std.is_cross_product(base_type))
      depth = #base_type.partition_symbols - 1
    end

    local index_access = codegen.expr(cx, expr):read(cx)

    local terra partition_functor([cx.runtime],
                                  mappable : c.legion_mappable_t,
                                  idx : uint,
                                  [parent],
                                  [point])
      var [requirement];
      var mappable_type = c.legion_mappable_get_type(mappable)
      if mappable_type == c.TASK_MAPPABLE then
        var task = c.legion_mappable_as_task(mappable)
        [requirement] = c.legion_task_get_requirement(task, idx)
      elseif mappable_type == c.COPY_MAPPABLE then
        var copy = c.legion_mappable_as_copy(mappable)
        [requirement] = c.legion_copy_get_requirement(copy, idx)
      elseif mappable_type == c.FILL_MAPPABLE then
        var fill = c.legion_mappable_as_fill(mappable)
        std.assert(idx == 0, "projection index for fill is not zero")
        [requirement] = c.legion_fill_get_requirement(fill)
      elseif mappable_type == c.INLINE_MAPPABLE then
        var mapping = c.legion_mappable_as_inline_mapping(mappable)
        std.assert(idx == 0, "projection index for inline mapping is not zero")
        [requirement] = c.legion_inline_get_requirement(mapping)
      else
        std.assert(false, "unhandled mappable type")
      end
      [symbol_setup];
      [free_vars_setup];
      [index_access.actions];
      return [index_access.value].impl
    end

    return std.register_projection_functor(false, false, depth, nil, partition_functor)

  -- create fill projection functor without mappable
  -- create projection functors with no preamble or free variables without mappable
  else
    local terra partition_functor(runtime : c.legion_runtime_t,
                                  parent : c.legion_logical_partition_t,
                                  [point],
                                  launch : c.legion_domain_t)
      [symbol_setup];
      [value.actions];
      var index : index_type = [value.value];
      var subregion = c.legion_logical_partition_get_logical_subregion_by_color_domain_point(
        runtime, parent, index)
      return subregion
    end

    return std.register_projection_functor(false, true, 0, nil, partition_functor)
  end
end

local function add_region_fields(cx, arg_type, field_paths, field_types, launcher, index)
  local add_field = c.legion_task_launcher_add_field
  if index then
    add_field = c.legion_index_launcher_add_field
  end

  local field_id_index = data.newmap()
  for i, f in ipairs(cx:region(arg_type).field_paths) do
    field_id_index[f] = i - 1
  end

  -- TODO: Would be good to take field_id_array by pointer, but that
  -- would require that the field ID array always be stored in memory
  -- (i.e. not a constant or r-value expression).
  local terra add_fields([launcher], requirement : uint, field_id_array : &c.legion_field_id_t[#cx:region(arg_type).field_paths])
    [data.zip(field_paths, field_types):map(
       function(field)
         local field_path, field_type = unpack(field)
         local field_id = `(@field_id_array)[ [field_id_index[field_path] ] ]
         if std.is_regent_array(field_type) then
           return quote
             for idx = 0, [field_type.N] do
               add_field(
                 [launcher], requirement, [field_id] + idx, true)
             end
           end
         else
           return quote
             add_field(
               [launcher], requirement, [field_id], true)
           end
         end
       end)]
  end
  add_fields:setinlined(false)

  return add_fields
end

local function expr_call_setup_region_arg(
    cx, task, arg_value, arg_type, param_type, launcher, index, args_setup)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)
  local privilege_modes = privileges:map(std.privilege_mode)
  local coherence_modes = coherences:map(std.coherence_mode)

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
      local field_type
      for _, t in ipairs(field_types) do
        if field_type then
          assert(std.type_eq(field_type, t))
        else
          field_type = t
        end
      end
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

    local add_fields = add_region_fields(cx, arg_type, field_paths, field_types, launcher, index)

    args_setup:insert(
      quote
        var [requirement] = [add_requirement]([requirement_args])
        [add_fields]([launcher], [requirement], &[cx:region(arg_type).field_id_array])
        [add_flags]([launcher], [requirement], [flag])
      end)
  end
end

local function expr_call_setup_region_arg_local(
    cx, task, arg_value, arg_type, param_type, physical_regions)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]

    local physical_region
    for _, field_path in ipairs(field_paths) do
      if not physical_region then
        physical_region = cx:region(arg_type).physical_regions[field_path]
      end
      -- Better hope all the fields are in the same physical region...
    end
    physical_regions:insert(physical_region)
  end
end

local function setup_list_of_regions_add_region(
    cx, param_type, container_type, value_type, value,
    region, parent, field_paths, field_types, add_requirement, get_requirement,
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
    [data.zip(field_paths, field_types):map(
       function(field)
         local field_path, field_type = unpack(field)
         local field_id = cx:list_of_regions(container_type):field_id(field_path)
         if std.is_regent_array(field_type) then
           return quote
             if not [has_field]([launcher], requirement, [field_id]) then
               for idx = 0, [field_type.N] do
                 [add_field]([launcher], requirement, [field_id] + idx, true)
               end
             end
           end
         else
           return quote
             if not [has_field]([launcher], requirement, [field_id]) then
               [add_field]([launcher], requirement, [field_id], true)
             end
           end
         end
       end)]
    end
end

local function setup_list_of_regions_add_list(
    cx, param_type, container_type, value_type, value,
    region, parent, field_paths, field_types, add_requirement, get_requirement,
    add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)
  local element = terralib.newsymbol(value_type.element_type)
  if std.is_list(value_type.element_type) then
    return quote
      for i = 0, [value].__size do
        var [element] = [value_type:data(value)][i]
        [setup_list_of_regions_add_list(
           cx, param_type, container_type, value_type.element_type, element,
           region, parent, field_paths, field_types, add_requirement, get_requirement,
           add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher)]
      end
    end
  else
    return quote
      for i = 0, [value].__size do
        var [element] = [value_type:data(value)][i]
        [setup_list_of_regions_add_region(
           cx, param_type, container_type, value_type.element_type, element,
           region, parent, field_paths, field_types, add_requirement, get_requirement,
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
        region, parent, field_paths, field_types, add_requirement, get_requirement,
        add_field, has_field, add_flags, intersect_flags, requirement_args, flag, launcher))
  end
end

local function index_launch_free_var_setup(free_vars)
  local free_vars_struct = terralib.types.newstruct()
  free_vars_struct.entries = terralib.newlist()
  for _, symbol in free_vars:keys() do
    free_vars_struct.entries:insert({
      field = tostring(symbol),
      type = symbol:gettype(),
    })
  end

  local free_vars_setup = terralib.newlist()
  local get_args = c.legion_index_launcher_get_projection_args
  local proj_args_get = terralib.newsymbol(free_vars_struct, "proj_args")
  local reg_requirement = terralib.newsymbol(c.legion_region_requirement_t, "requirement")
  free_vars_setup:insert(
    quote
      var [proj_args_get] = @[&free_vars_struct]([get_args]([reg_requirement], nil))
    end)
  for _, symbol in free_vars:keys() do
    free_vars_setup:insert(
      quote
        var [symbol:getsymbol()] = [proj_args_get].[tostring(symbol)]
      end)
  end
  return free_vars_setup, free_vars_struct, reg_requirement
end

local function expr_call_setup_partition_arg(
    outer_cx, cx, task, arg_value, arg_type, param_type, partition, loop_index, launcher, index, args_setup, free_vars, loop_vars_setup)
  assert(index)
  local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
    std.find_task_privileges(param_type, task)
  local privilege_modes = privileges:map(std.privilege_mode)
  local coherence_modes = coherences:map(std.coherence_mode)

  local set_args = c.legion_index_launcher_set_projection_args
  local free_vars_setup, free_vars_struct, reg_requirement =
    index_launch_free_var_setup(free_vars)

  free_vars_setup:insertall(loop_vars_setup)

  -- Cross products always need the full-blown partition_functor
  local needs_non_identity_functor = not (
    is_identity_projection(arg_value, loop_index) and
    std.is_partition(
      std.as_read(util.get_base_indexed_node(arg_value).expr_type)))
  local proj_args_set = nil
  if needs_non_identity_functor and not free_vars:is_empty() then
    proj_args_set = terralib.newsymbol(free_vars_struct, "proj_args")
    args_setup:insert(
      quote
        var [proj_args_set]
      end)
    for _, symbol in free_vars:keys() do
      args_setup:insert(
        quote
          [proj_args_set].[tostring(symbol)] = [symbol:getsymbol()]
        end)
    end
  end

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
      local field_type
      for _, t in ipairs(field_types) do
        if field_type then
          assert(std.type_eq(field_type, t))
        else
          field_type = t
        end
      end
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

    local projection_functor = make_partition_projection_functor(outer_cx, arg_value, loop_index, false, free_vars, free_vars_setup, reg_requirement)

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

    local add_fields = add_region_fields(cx, arg_type, field_paths, field_types, launcher, true)

    args_setup:insert(
      quote
        var [requirement] = [add_requirement]([requirement_args])
        [add_fields]([launcher], [requirement], &[cx:region(arg_type).field_id_array])
        c.legion_index_launcher_add_flags([launcher], [requirement], [flag])
      end)
    if proj_args_set ~= nil then
      args_setup:insert(
        quote
          [set_args]([launcher], [requirement], [&opaque](&[proj_args_set]), terralib.sizeof(free_vars_struct), false)
        end)
    end
  end
end

local function expr_call_setup_remote(cx, node, task, arg_values, arg_types, param_types, conditions, launcher, args_setup)
  -- Pass futures.
  for i, arg_type in ipairs(arg_types) do
    if std.is_future(arg_type) then
      local arg_value = arg_values[i]
      expr_call_setup_future_arg(
        cx, task, arg_value,
        launcher, false, args_setup)
    end
  end

  -- Pass phase barriers (from annotations on parameters).
  local param_conditions = task:get_conditions()
  for condition, args_enabled in param_conditions:items() do
    for i, arg_type in ipairs(arg_types) do
      if args_enabled[i] then
        assert(std.is_phase_barrier(arg_type) or
          (std.is_list(arg_type) and std.is_phase_barrier(arg_type.element_type)))
        local arg_value = arg_values[i]
        expr_call_setup_phase_barrier_arg(
          cx, task, arg_value, condition,
          launcher, false, args_setup, arg_type)
      end
    end
  end

  -- Pass phase barriers (from extra conditions).
  for i, condition in ipairs(node.conditions) do
    local condition_expr = conditions[i]
    for _, condition_kind in ipairs(condition.conditions) do
      expr_call_setup_phase_barrier_arg(
        cx, task, condition_expr.value, condition_kind,
        launcher, false, args_setup, std.as_read(condition.expr_type))
    end
  end

  -- Pass index spaces through index requirements.
  for i, arg_type in ipairs(arg_types) do
    if std.is_ispace(arg_type) then
      local param_type = param_types[i]

      expr_call_setup_ispace_arg(
        cx, task, arg_type, param_type, launcher, false, args_setup)
    end
  end

  -- Pass regions through region requirements.
  local fn_type = task:get_type()
  for _, i in ipairs(std.fn_param_regions_by_index(fn_type)) do
    local arg_value = arg_values[i]
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    expr_call_setup_region_arg(
      cx, task, node.args[i], arg_type, param_type, launcher, false, args_setup)
  end

  -- Pass regions through lists of region requirements.
  for _, i in ipairs(std.fn_param_lists_of_regions_by_index(fn_type)) do
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    expr_call_setup_list_of_regions_arg(
      cx, task, arg_type, param_type, launcher, false, args_setup)
  end
end

local function expr_call_setup_local(cx, node, task, arg_values, arg_types, param_types, args_setup)
  -- Pass futures.
  local futures = terralib.newlist()
  for i, arg_type in ipairs(arg_types) do
    if std.is_future(arg_type) then
      local arg_value = arg_values[i]
      futures:insert(`([arg_value].__result))
    end
  end

  -- No phase barriers.
  assert(#task:get_conditions() == 0)
  assert(#node.conditions == 0)

  -- Don't need special setup for index spaces.

  -- Pass regions.
  local physical_regions = terralib.newlist()
  local fn_type = task:get_type()
  for _, i in ipairs(std.fn_param_regions_by_index(fn_type)) do
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    expr_call_setup_region_arg_local(
      cx, task, node.args[i], arg_type, param_type, physical_regions)
  end

  -- Pass lists of regions.
  for _, i in ipairs(std.fn_param_lists_of_regions_by_index(fn_type)) do
    assert(false) -- Not supported.
  end

  return futures, physical_regions
end

local function expr_call_setup_predicate(cx, node, predicate, predicate_else_value, launcher, args_setup, task_args_setup)
  -- Setup predicate.
  local predicate_symbol = terralib.newsymbol(c.legion_predicate_t, "predicate")
  local predicate_value
  if predicate then
    predicate_value = `c.legion_predicate_create([cx.runtime], [cx.context], [predicate.value].__result)
  else
    predicate_value = `c.legion_predicate_true()
  end
  task_args_setup:insert(
    quote
      var [predicate_symbol] = [predicate_value]
    end)

  if predicate_else_value then
    assert(std.is_future(std.as_read(node.predicate_else_value.expr_type)))
    args_setup:insert(
      quote
        c.legion_task_launcher_set_predicate_false_future(
          [launcher], [predicate_else_value.value].__result)
      end)
  end

  return predicate_symbol
end

local function loop_bounds_to_domain_or_index_space(cx, values, value_type)
  if terralib.islist(values) then
    return `((rect1d {
                lo = int1d([values[1]]),
                hi = int1d([values[2]]) - 1,
             }):to_domain()), false
  else
    assert(value_type)
    if std.is_ispace(value_type) then
      return `[values].impl, true
    elseif std.is_region(value_type) then
      return `[values].impl.index_space, true
    elseif std.is_rect_type(value_type) then
      return `([values]:to_domain()), false
    else
      assert(false)
    end
  end
end


local function loop_bounds_to_domain(cx, values, value_type)
  local value, is_index_space = loop_bounds_to_domain_or_index_space(
    cx, values, value_type)
  if is_index_space then
    return `c.legion_index_space_get_domain([cx.runtime], value)
  end
  return value
end

local function loop_bounds_to_index_space(cx, values, value_type)
  local value, is_index_space = loop_bounds_to_domain_or_index_space(
    cx, values, value_type)
  if not is_index_space then
    return `c.legion_index_space_create_domain(
      [cx.runtime], [cx.context], value)
  end
  return value
end

function codegen.expr_call(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local args = node.args:map(
    function(arg) return codegen.expr(cx, arg):read(cx, arg.expr_type) end)
  local conditions = node.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)
  local predicate = node.predicate and codegen.expr(cx, node.predicate):read(cx)
  local predicate_else_value = node.predicate_else_value and
    codegen.expr(cx, node.predicate_else_value):read(cx)

  local actions = quote
    [fn.actions];
    [args:map(function(arg) return arg.actions end)];
    [conditions:map(function(condition) return condition.actions end)];
    [predicate and predicate.actions];
    [predicate_else_value and predicate_else_value.actions];
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

    local is_local = fn.value.is_local
    local launcher
    if not is_local then
      launcher = terralib.newsymbol(c.legion_task_launcher_t, "launcher")
    end

    local args_setup = terralib.newlist()
    local predicate_symbol
    local local_futures, local_regions
    if not is_local then
      expr_call_setup_remote(
        cx, node, fn.value, arg_values, arg_types, param_types, conditions,
        launcher, args_setup)
      predicate_symbol = expr_call_setup_predicate(
        cx, node, predicate, predicate_else_value,
        launcher, args_setup, task_args_setup)
    else
      local_futures, local_regions = expr_call_setup_local(
        cx, node, fn.value, arg_values, arg_types, param_types, args_setup)
      assert(not predicate) -- Predication is not supported.
    end

    local future
    if not cx.must_epoch then
      future = terralib.newsymbol(c.legion_future_t, "future")
    end

    if cx.loop_point and not is_local then
      local point
      if cx.loop_point.type:isintegral() then
        point = `(int1d([cx.loop_point]):to_domain_point())
      else
        point = `([cx.loop_point]:to_domain_point())
      end

      args_setup:insert(
        quote
          c.legion_task_launcher_set_point(
            [launcher], [point])
        end)
    end

    if cx.loop_domain and not is_local and
      -- Skip doing this if we have a stride in the loop.
      (not terralib.islist(cx.loop_domain) or #cx.loop_domain == 2)
    then
      local domain = loop_bounds_to_index_space(
        cx, cx.loop_domain, cx.loop_domain_type)

      args_setup:insert(
        quote
          c.legion_task_launcher_set_sharding_space(
            [launcher], [domain])
        end)
    end

    local launcher_setup
    if not is_local then
      local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
      launcher_setup = quote
        var [task_args]
        [task_args_setup]
        var mapper = [fn.value:has_mapper_id() or 0]
        var [tag] = [fn.value:has_mapping_tag_id() or 0]
        [codegen_hooks.gen_update_mapping_tag(tag, fn.value:has_mapping_tag_id(), cx.task)]
        var [launcher] = c.legion_task_launcher_create(
          [fn.value:get_task_id()], [task_args],
          [predicate_symbol], [mapper], [tag])
        c.legion_task_launcher_set_provenance([launcher], [get_provenance(node)])
        [args_setup]
      end
    else
      launcher_setup = quote
        var [task_args]
        [task_args_setup]
        [args_setup]
      end
    end

    local launcher_execute
    local local_result
    if not is_local and not cx.must_epoch then
      launcher_execute = quote
        var [future] = c.legion_task_launcher_execute(
          [cx.runtime], [cx.context], [launcher])
        c.legion_task_launcher_destroy(launcher)
        [task_args_cleanup]
      end
    elseif not is_local then
      launcher_execute = quote
        c.legion_must_epoch_launcher_add_single_task(
          [cx.must_epoch],
          [int1d]([cx.must_epoch_point]),
          [launcher])
        [cx.must_epoch_point] = [cx.must_epoch_point] + 1
      end
    else
      local region_array = terralib.newsymbol(c.legion_physical_region_t[#local_regions], "physical_regions")
      local result = terralib.newsymbol(std.serialized_value, "result")

      local c_task = terralib.newsymbol(c.legion_task_t, "task")
      local c_regions = terralib.newsymbol(&c.legion_physical_region_t, "regions")
      local c_num_regions = #local_regions
      local c_context = cx.context
      local c_runtime = cx.runtime
      local c_result = `(&result)
      local c_params = terralib.newlist({
        c_task, c_regions, c_num_regions, c_context, c_runtime, c_result })

      local variant
      if cx.variant:is_cuda() then
        variant = fn.value:get_cuda_variant()
        assert(variant)
      elseif cx.variant:is_openmp() then
        for _, v in ipairs(fn.value:get_variants()) do
          if v:is_openmp() then
            variant = v
          end
        end
        assert(variant)
      else
        variant = fn.value:get_primary_variant()
      end
      launcher_execute = quote
        var [region_array]
        [data.mapi(
           function(i, pr)
             return quote
               [region_array][ [i-1] ] = [pr]
             end
           end,
           local_regions)]

        var task_mut = c.legion_task_create_empty()
        c.legion_task_set_args(task_mut, [task_args].args)
        c.legion_task_set_arglen(task_mut, [task_args].arglen)
        [local_futures:map(
           function(future)
             return quote
               c.legion_task_add_future(task_mut, [future])
             end
           end)]
        var [c_task] = c.legion_task_mut_as_task(task_mut)
        var [c_regions] = [region_array]
        var [result] = std.serialized_value { nil, 0 }
        [variant:get_definition()]([c_params])
        c.legion_task_destroy(task_mut)
        [task_args_cleanup]
        -- This is a hack, but it's actually easier to drive the
        -- serialization infrastructure if we go through the same
        -- future code path as remote task launches.
        var [future] = c.legion_future_from_untyped_pointer([cx.runtime], [result].value, [result].size)
        c.free([result].value)
      end
    end

    actions = quote
      [actions]
      [launcher_setup]
      [launcher_execute]
    end

    if predicate then
      actions = quote
        [actions]
        c.legion_predicate_destroy([predicate_symbol])
      end
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

      return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
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
  local expr_type = std.as_read(node.expr_type)
  assert(not std.is_future(expr_type)) -- This is handled in optimize_future now

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
  local region = codegen.expr_region_root(cx, node.region):read(cx)
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

function codegen.expr_raw_future(cx, node)
  local value = codegen.expr(cx, node.value):read(cx, node.value.expr_type)

  local future_type = node.expr_type
  if not std.is_future(future_type) then
    future_type = std.future(node.expr_type)
  end

  local future_value = values.value(
    node,
    expr.once_only(value.actions,
      `([future_type] {__result = [value.value]} ),
      future_type),
    future_type)

  if std.is_future(node.expr_type) then
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
        expr_type = node.expr_type,
        annotations = node.annotations,
        span = node.span,
     })
  end
end

function codegen.expr_raw_physical(cx, node)
  local region = codegen.expr_region_root(cx, node.region):read(cx)
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

function codegen.expr_raw_task(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    node,
    expr.just(emit_debuginfo(node), cx.task),
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
  local pointer_type = std.as_read(node.pointer.expr_type)
  local expr_type = std.as_read(node.expr_type)
  local actions = quote
    [pointer.actions];
    [emit_debuginfo(node)]
  end

  if std.is_bounded_type(pointer_type) then
    local index_type = pointer_type.index_type
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
  elseif pointer_type:ispointer() or pointer_type == niltype then
    return values.value(
      node,
      expr.once_only(
        actions,
        `([expr_type]([pointer.value] == nil)),
        expr_type),
      expr_type)
  else
    assert(false, "unreachable")
  end
end

function codegen.expr_null(cx, node)
  local pointer_type = node.pointer_type
  local expr_type = std.as_read(node.expr_type)

  if std.is_bounded_type(pointer_type) then
    return values.value(
      node,
      expr.once_only(
        emit_debuginfo(node),
        `([pointer_type]{ __ptr = [ptr] { __ptr = c.legion_ptr_nil() }}),
        expr_type),
      expr_type)
  elseif pointer_type:ispointer() or pointer_type == niltype then
    return values.value(
      node,
      expr.once_only(
        emit_debuginfo(node),
        `([pointer_type](nil)),
        expr_type),
      expr_type)
  else
    assert(false, "unreachable")
  end
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
  if std.is_partition(expr_type) then
    local result = terralib.newsymbol(expr_type)
    actions = quote
      [actions];
      var [result] = [expr_type] { impl = [value.value].impl }
    end
    return values.value(node, expr.just(actions, result), expr_type)
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

    if cx.bounds_checks then
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
    [start and start.actions or (empty_quote)];
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

  local bounds_actions, domain, bounds = index_space_bounds(cx, is, ispace_type)
  cx:add_ispace_root(ispace_type, is, domain, bounds)

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

  local source_file = tostring(node.span.source)
  local source_line = tostring(node.span.start.line)

  actions = quote
    [actions]
    c.legion_index_space_attach_semantic_information(
      [cx.runtime], [is], c.SOURCE_FILE_TAG, [source_file], [string.len(source_file)+1], false)
    c.legion_index_space_attach_semantic_information(
      [cx.runtime], [is], c.SOURCE_LINE_TAG, [source_line], [string.len(source_line)+1], false)
    var [i] = [ispace_type]{ impl = [is] }
    [bounds_actions]
  end

  return values.value(node, expr.just(actions, i), ispace_type)
end

local function attach_name_and_type(cx, fs, field_id, field_name, field_type)
  local actions = terralib.newlist()
  actions:insert(
    quote
      c.legion_field_id_attach_name(
        [cx.runtime], [fs], field_id, field_name, false)
  end)
  if std.get_type_id(field_type) then
    actions:insert(
      quote
        var type_id : uint32 = [std.get_type_id(field_type)]
        c.legion_field_id_attach_semantic_information(
          [cx.runtime], [fs], field_id, [std.get_type_semantic_tag()], &type_id, terralib.sizeof(uint32), false)
    end)
  end
  return actions
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
  local field_ids = data.zip(field_paths, field_types):map(
    function(pair)
      field_id = field_id + 1
      local my_field_id = field_id
      local field_path, field_type = unpack(pair)
      if std.is_regent_array(field_type) then
        field_id = field_id + field_type.N - 1
      end
      return my_field_id
    end)

  -- Hack: allocate a buffer here, because we don't want these to live
  -- on the stack and we can't take an address to constant memory.
  local field_id_array_buffer = terralib.newsymbol(&c.legion_field_id_t[#field_ids], "field_ids")
  local field_id_array = `(@[field_id_array_buffer])
  local field_id_array_initializer = terralib.constant(`arrayof(c.legion_field_id_t, [field_ids]))

  local fields_are_scratch = field_paths:map(function(_) return false end)
  local physical_regions = field_paths:map(function(_) return pr end)

  local pr_actions, base_pointers, strides = unpack(data.zip(unpack(
    data.zip(field_types, field_paths, field_ids, field_privileges):map(
      function(field)
        local field_type, field_path, field_id, field_privilege = unpack(field)
        return terralib.newlist({
          physical_region_get_base_pointer(cx, region_type, index_type, field_type, field_path, pr, field_id)})
  end))))
  pr_actions = pr_actions or terralib.newlist()
  base_pointers = base_pointers or terralib.newlist()
  strides = strides or terralib.newlist()

  cx:add_region_root(region_type, r,
                     field_paths,
                     terralib.newlist({field_paths}),
                     data.dict(data.zip(field_paths, field_privileges)),
                     data.dict(data.zip(field_paths, field_types)),
                     data.dict(data.zip(field_paths, field_ids)),
                     field_id_array,
                     data.dict(data.zip(field_paths, fields_are_scratch)),
                     data.dict(data.zip(field_paths, physical_regions)),
                     data.dict(data.zip(field_paths, base_pointers)),
                     data.dict(data.zip(field_paths, strides)))

  local source_file = tostring(node.span.source)
  local source_line = tostring(node.span.start.line)

  local fs_naming_actions = quote
    c.legion_field_space_attach_name([cx.runtime], [fs], [tostring(fspace_type)], false)
    c.legion_field_space_attach_semantic_information(
      [cx.runtime], [fs], c.SOURCE_FILE_TAG, [source_file], [string.len(source_file)+1], false)
    c.legion_field_space_attach_semantic_information(
      [cx.runtime], [fs], c.SOURCE_LINE_TAG, [source_line], [string.len(source_line)+1], false)
  end
  if fspace_type:isstruct() then
    fs_naming_actions = quote
      [fs_naming_actions]
      [data.flatmap(
         function(field)
           local field_path, field_type, field_id = unpack(field)
           local field_name = field_path:mkstring("", ".", "")
           local actions = terralib.newlist()
           if std.is_regent_array(field_type) then
             for idx = 0, field_type.N - 1 do
               local elem_name = field_name .. "[" .. tostring(idx) .. "]"
               actions:insertall(
                 attach_name_and_type(cx, fs, field_id + idx, elem_name, field_type.elem_type))
             end
           else
             actions:insertall(
               attach_name_and_type(cx, fs, field_id, field_name, field_type))
           end
           return actions
         end,
         data.zip(field_paths, field_types, field_ids))]
    end
  else
    assert(#field_ids == 1)
    fs_naming_actions = attach_name_and_type(cx, fs, field_ids[1], "__value", field_types[1])
  end

  local source_file = tostring(node.span.source)
  local source_line = tostring(node.span.start.line)

  actions = quote
    [actions]
    var capacity = [ispace.value]
    var [is] = [ispace.value].impl
    var [fs] = c.legion_field_space_create([cx.runtime], [cx.context])
    var fsa = c.legion_field_allocator_create([cx.runtime], [cx.context],  [fs]);
    var [field_id_array_buffer] = [&c.legion_field_id_t[#field_ids]](c.malloc([#field_ids] * [terralib.sizeof(c.legion_field_id_t)]))
    regentlib.assert([#field_ids] == 0 or [field_id_array_buffer] ~= nil, "failed allocation in field ID array buffer")
    [field_id_array] = [field_id_array_initializer]
    [data.flatmap(
       function(field)
         local field_type, field_id = unpack(field)
         if std.is_regent_array(field_type) then
           local allocate_fields = terralib.newlist()
           return quote
             for idx = 0, [field_type.N] do
               c.legion_field_allocator_allocate_field(
                 fsa, terralib.sizeof([field_type.elem_type]), [field_id] + idx)
             end
           end
         else
           return `(c.legion_field_allocator_allocate_field(
                      fsa, terralib.sizeof([field_type]), [field_id]))
         end
       end,
       data.zip(field_types, field_ids))]
    [fs_naming_actions];
    c.legion_field_allocator_destroy(fsa)
    var [lr] = c.legion_logical_region_create([cx.runtime], [cx.context], [is], [fs], true)
    c.legion_logical_region_attach_semantic_information(
      [cx.runtime], [lr], c.SOURCE_FILE_TAG, [source_file], [string.len(source_file)+1], false)
    c.legion_logical_region_attach_semantic_information(
      [cx.runtime], [lr], c.SOURCE_LINE_TAG, [source_line], [string.len(source_line)+1], false)
    var [r] = [region_type]{ impl = [lr] }
    [tag_imported(cx, lr)]
  end

  cx:add_cleanup_item(quote c.free([field_id_array_buffer]) end)

  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  if not cx.variant:get_config_options().inner and
    (not cx.region_usage or cx.region_usage[region_type])
  then
    actions = quote
      [actions];
      var [tag] = 0
      [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
      -- Note: it's safe to make this unconditionally write-discard
      -- because this is guarranteed to be the first use of the region
      var il = c.legion_inline_launcher_create_logical_region(
        [lr], c.WRITE_DISCARD, c.EXCLUSIVE, [lr], 0, false, 0, [tag]);
      c.legion_inline_launcher_set_provenance(il, [get_provenance(node)])
      [data.zip(field_ids, field_types):map(
         function(field)
           local field_id, field_type = unpack(field)
           if std.is_regent_array(field_type) then
             return quote
               for idx = 0, [field_type.N] do
                 c.legion_inline_launcher_add_field(il, [field_id] + idx, true)
               end
             end
           else
             return `(c.legion_inline_launcher_add_field(il, [field_id], true))
           end
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

local function partition_kind(disjointness, completeness)
  local mapping = {
    [false] = {
      [false] = c.COMPUTE_KIND,
      [std.incomplete] = c.COMPUTE_INCOMPLETE_KIND,
      [std.complete] = c.COMPUTE_COMPLETE_KIND,
    },
    [std.aliased] = {
      [false] = c.ALIASED_KIND,
      [std.incomplete] = c.ALIASED_INCOMPLETE_KIND,
      [std.complete] = c.ALIASED_COMPLETE_KIND,
    },
    [std.disjoint] = {
      [false] = c.DISJOINT_KIND,
      [std.incomplete] = c.DISJOINT_INCOMPLETE_KIND,
      [std.complete] = c.DISJOINT_COMPLETE_KIND,
    },
  }

  return assert(mapping[disjointness][completeness])
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
    [(colors and colors.actions) or (empty_quote)];
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
    args:insert(partition_kind(node.disjointness, node.completeness))
  else
    assert(false)
  end

  args:insert(c.AUTO_GENERATE_ID)


  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  actions = quote
    [actions]
    var [ip] = [index_partition_create]([args])
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [region.value].impl, [ip])
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
        [colors.value].impl, 1 --[[ granularity ]], c.AUTO_GENERATE_ID)
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [region.value].impl, [ip])
    end
  else
    local dim = region_type:ispace().dim
    local colors_dim = colors_type.dim
    local transform = terralib.newsymbol(
      c["legion_transform_" .. tostring(dim) .. "x" .. tostring(dim) .. "_t"], "transform")
    local extent = terralib.newsymbol(
      c["legion_rect_" .. tostring(dim) .. "d_t"], "extent")
    local domain_get_rect =
      c["legion_domain_get_rect_" .. tostring(dim) .. "d"]
    local create_domain_transform =
      c["legion_domain_transform_from_" .. tostring(dim) .. "x" .. tostring(dim)]
    local create_domain =
      c["legion_domain_from_rect_" .. tostring(dim) .. "d"]

    actions = quote
      [actions]
      var [ip]
      var region_domain = c.legion_index_space_get_domain(
        [cx.runtime], [region.value].impl.index_space)
      var color_domain = c.legion_index_space_get_domain(
        [cx.runtime], [colors.value].impl)
      if c.legion_domain_is_dense(region_domain) and c.legion_domain_is_dense(color_domain) and [dim == colors_dim] then
        var region_rect = [domain_get_rect](region_domain)
        var color_rect = [domain_get_rect](color_domain)

        var [transform]
        var [extent]
        [data.range(dim):map(
           function(i)
             local block = `(region_rect.hi.x[ [i] ] - region_rect.lo.x[ [i] ] + 1)
             local colors = `(color_rect.hi.x[ [i] ] - color_rect.lo.x[ [i] ] + 1)
             return quote
               var block_size = [block] / [colors] + ([block] % [colors] + [colors] - 1) / [colors]
               for j = 0, dim do
                 if i == j then
                   [transform].trans[i][j] = block_size
                 else
                   [transform].trans[i][j] = 0
                 end
               end

               [extent].lo.x[ [i] ] =
                 region_rect.lo.x[ [i] ] - color_rect.lo.x[ [i] ] * block_size
               [extent].hi.x[ [i] ] =
                 region_rect.lo.x[ [i] ] - color_rect.lo.x[ [i] ] * block_size + block_size - 1
             end
           end)]
        var dtransform = [create_domain_transform]([transform])
        var dextent = [create_domain]([extent])
        [ip] = c.legion_index_partition_create_by_restriction(
          [cx.runtime], [cx.context], [region.value].impl.index_space,
          [colors.value].impl,
          dtransform, dextent, c.DISJOINT_KIND,
          c.AUTO_GENERATE_ID)
      else
        [ip] = c.legion_index_partition_create_equal(
          [cx.runtime], [cx.context], [region.value].impl.index_space,
          [colors.value].impl, 1,
          c.AUTO_GENERATE_ID)
      end
      var [lp] = c.legion_logical_partition_create(
        [cx.runtime], [region.value].impl, [ip])
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
      field_id, [colors.value].impl, c.AUTO_GENERATE_ID, 0, 0,
      [partition_kind(std.disjoint, node.completeness)])
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [region.value].impl, [ip])
  end

  return values.value(
    node,
    expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_partition_by_restriction(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr(cx, node.region):read(cx)
  local transform_type = std.as_read(node.transform.expr_type)
  local transform = codegen.expr(cx, node.transform):read(cx)
  local extent_type = std.as_read(node.extent.expr_type)
  local extent = codegen.expr(cx, node.extent):read(cx)
  local colors_type = std.as_read(node.colors.expr_type)
  local colors = codegen.expr(cx, node.colors):read(cx)
  local partition_type = std.as_read(node.expr_type)
  local actions = quote
    [region.actions];
    [transform.actions];
    [extent.actions];
    [colors.actions];
    [emit_debuginfo(node)]
  end

  assert(cx:has_region(region_type))

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  actions = quote
    [actions]
    var [ip] = c.legion_index_partition_create_by_restriction(
      [cx.runtime], [cx.context],
      [region.value].impl.index_space,
      [colors.value].impl,
      [transform.value], [extent.value],
      [partition_kind(node.disjointness, node.completeness)],
      c.AUTO_GENERATE_ID)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [region.value].impl, [ip])
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
      [partition.value].impl, [region_parent].impl, field_id, colors,
      [partition_kind(node.disjointness, node.completeness)],
      c.AUTO_GENERATE_ID, 0, 0)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [parent.value].impl, [ip])
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
      [partition_kind(node.disjointness, node.completeness)],
      c.AUTO_GENERATE_ID, 0, 0)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [region.value].impl, [ip])
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
      [cx.runtime], lr.impl, ip)
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
    std.assert(color_domain.dim == 1, "color domain should be 1D")
    var start_color = color_domain.rect_data[0]
    var end_color = color_domain.rect_data[1]
    std.assert(start_color >= 0 and end_color >= 0,
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
      var rhs_lp = c.legion_logical_partition_create([cx.runtime], lhs_subregion, rhs_ip)
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
    cx:region(parent_region).field_id_array,
    cx:region(parent_region).fields_are_scratch)

  actions = quote
    [actions]
    var size = terralib.sizeof([expr_type.element_type]) * [indices.value].__size
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_slice_partition")
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
    cx:region(parent_region).field_id_array,
    cx:region(parent_region).fields_are_scratch)

  local source_file = tostring(node.span.source)
  local source_line = tostring(node.span.start.line)

  actions = quote
    [actions]
    var size = terralib.sizeof([expr_type.element_type]) * [indices.value].__size
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_duplicate_partition")
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
      std.assert(name ~= nil, "invalid name")
      c.legion_logical_region_attach_name([cx.runtime], new_root, name, false)
      c.legion_logical_region_attach_semantic_information(
        [cx.runtime], [r], c.SOURCE_FILE_TAG, [source_file], [string.len(source_file)+1], false)
      c.legion_logical_region_attach_semantic_information(
        [cx.runtime], [r], c.SOURCE_LINE_TAG, [source_line], [string.len(source_line)+1], false)

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
    var size = terralib.sizeof([expr_type.element_type]) * [indices.value].__size
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_slice_cross_product")
    var [result] = expr_type {
      __size = [indices.value].__size,
      __data = data,
    }
    for i = 0, [indices.value].__size do
      var color = c.legion_domain_point_from_point_1d(
        c.legion_point_1d_t { x = arrayof(c.coord_t, [indices_type:data(indices.value)][i]) })
      var ip = c.legion_terra_index_cross_product_get_subpartition_by_color_domain_point(
        [cx.runtime], [product.value].product, color)
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
    cx:list_of_regions(lhs_type).field_id_array,
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

    var size = terralib.sizeof([expr_type.element_type]) * [lhs.value].__size
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_cross_product")
    var [result] = expr_type {
      __size = [lhs.value].__size,
      __data = data,
    }
    for i = 0, [lhs.value].__size do
      var subsize = terralib.sizeof([expr_type.element_type.element_type]) * [rhs.value].__size
      var subdata = c.malloc(subsize)
      std.assert(subsize == 0 or subdata ~= nil, "malloc failed in list_cross_product")
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
    cx:list_of_regions(lhs_type).field_id_array,
    cx:list_of_regions(lhs_type).fields_are_scratch)

  actions = quote
    [actions]

    std.assert([product.value].__size == [lhs.value].__size,
                     "size mismatch in list_cross_product 1")

    var lhs_list : c.legion_terra_index_space_list_t
    lhs_list.space.tid = 0
    lhs_list.space.id = 0
    lhs_list.count = [lhs.value].__size
    var lhs_size = terralib.sizeof([c.legion_index_space_t]) * [lhs.value].__size
    lhs_list.subspaces = [&c.legion_index_space_t](c.malloc(lhs_size))
    std.assert(lhs_size == 0 or lhs_list.subspaces ~= nil, "malloc failed in list_cross_product")
    for i = 0, [lhs.value].__size do
      lhs_list.subspaces[i] = [lhs_type:data(lhs.value)][i].impl.index_space
    end

    var shallow_product : c.legion_terra_index_space_list_list_t
    shallow_product.count = [lhs.value].__size
    var shallow_size = terralib.sizeof([c.legion_terra_index_space_list_t]) * [lhs.value].__size
    shallow_product.sublists = [&c.legion_terra_index_space_list_t](c.malloc(shallow_size))
    std.assert(shallow_size == 0 or shallow_product.sublists ~= nil, "malloc failed in list_cross_product")
    for i = 0, [lhs.value].__size do
      var subsize = [product_type:data(product.value)][i].__size
      shallow_product.sublists[i].space = [lhs_type:data(lhs.value)][i].impl.index_space
      shallow_product.sublists[i].count = subsize
      var shallow_subsize = terralib.sizeof([c.legion_index_space_t]) * subsize
      shallow_product.sublists[i].subspaces = [&c.legion_index_space_t](c.malloc(shallow_subsize))
      std.assert(shallow_subsize == 0 or shallow_product.sublists[i].subspaces ~= nil, "malloc failed in list_cross_product")
      for j = 0, subsize do
        shallow_product.sublists[i].subspaces[j] =
          [product_type.element_type:data(
             `([product_type:data(product.value)][i]))][j].impl.index_space
      end
    end

    var complete_product = c.legion_terra_index_cross_product_create_list_complete(
      [cx.runtime], [cx.context], [lhs_list], [shallow_product], false)
    std.assert(complete_product.count == [lhs.value].__size,
                     "size mismatch in list_cross_product 2")

    var data_size = terralib.sizeof([expr_type.element_type]) * [lhs.value].__size
    var data = c.malloc(data_size)
    std.assert(data_size == 0 or data ~= nil, "malloc failed in list_cross_product")
    var [result] = expr_type {
      __size = [lhs.value].__size,
      __data = data,
    }

    for i = 0, [lhs.value].__size do
      var subsize = [product_type:data(product.value)][i].__size
      std.assert(
        subsize == complete_product.sublists[i].count,
        "size mismatch in list_cross_product 3")

      -- Allocate sublist.
      var subdata_size = terralib.sizeof([expr_type.element_type.element_type]) * subsize
      var subdata = c.malloc(subdata_size)
      std.assert(subdata_size == 0 or subdata ~= nil, "malloc failed in list_cross_product")
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

local gen_expr_list_phase_barriers = data.weak_memoize(
  function(product_type, expr_type)
    local result = terralib.newsymbol(expr_type, "result")
    local product = terralib.newsymbol(product_type, "product")
    local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
    local context = terralib.newsymbol(c.legion_context_t, "context")
    local terra list_phase_barriers([runtime], [context], [product])
      var size = terralib.sizeof([expr_type.element_type]) * [product].__size
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in list_phase_barriers")
      var [result] = expr_type {
        __size = [product].__size,
        __data = data,
      }
      for i = 0, [product].__size do
        var subsize = [product_type:data(product)][i].__size

        -- Allocate sublist.
        var subdata_size = terralib.sizeof([expr_type.element_type.element_type]) * subsize
        var subdata = c.malloc(subdata_size)
        std.assert(subdata_size == 0 or subdata ~= nil, "malloc failed in list_phase_barriers")
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

local gen_expr_list_invert = data.weak_memoize(
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

      var size = terralib.sizeof([expr_type.element_type]) * [rhs].__size
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in list_invert")
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
        std.assert(num_colors >= 0, "invalid color range in list_invert")

        -- Build the index.
        var color_to_index_size = terralib.sizeof(int64) * num_colors
        var color_to_index = [&int64](c.malloc(color_to_index_size))
        std.assert(color_to_index_size == 0 or color_to_index ~= nil, "malloc failed in list_invert")

        for i = 0, num_colors do
          color_to_index[i] = -1
        end

        for i = 0, [rhs].__size do
          var rhs_elt = [rhs_type:data(rhs)][i].impl
          var rhs_color = [get_logical_region_color(runtime, rhs_elt, color_type)]

          var delta = rhs_color - min_color
          var color_index = [get_offset_in_rect(delta, colors_size)]
          std.assert(
            (0 <= color_index) and (color_index < num_colors),
            "color index out of bounds in list_invert")
          std.assert(
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
              std.assert(
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
          var subdata_size = terralib.sizeof([expr_type.element_type.element_type]) * subsize
          var subdata = [&expr_type.element_type.element_type](c.malloc(subdata_size))
          std.assert(subdata_size == 0 or subdata ~= nil, "malloc failed in list_invert")
          [expr_type:data(result)][i].__data = subdata
        end

        -- 4. Fill sublists.

        -- Create a list to hold the next index.
        var subslots_size = terralib.sizeof([int64]) * [rhs].__size
        var subslots = [&int64](c.malloc(subslots_size))
        std.assert(subslots_size == 0 or subslots ~= nil, "malloc failed in list_invert")
        for i = 0, [rhs].__size do
          subslots[i] = 0
        end

        for j = 0, [product].__size do
          for k = 0, [product_type:data(product)][j].__size do
            var leaf = [product_type.element_type:data(
                          `([product_type:data(product)][j]))][k].impl
            var inner = leaf

            if not [product_type.shallow] then
              std.assert(
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
            std.assert(subslots[i] < [expr_type:data(result)][i].__size,
                             "overflowed sublist in list_invert")
            [expr_type.element_type:data(`([expr_type:data(result)][i]))][subslots[i]] =
                [barriers_type.element_type:data(
                   `([barriers_type:data(barriers)][j]))][k]
            subslots[i] = subslots[i] + 1
          end
        end

        for i = 0, [rhs].__size do
          std.assert(subslots[i] == [expr_type:data(result)][i].__size, "underflowed sublist in list_invert")
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

    std.assert([stop.value] >= [start.value], "negative size range in list_range")
    var size = terralib.sizeof([expr_type.element_type]) * ([stop.value] - [start.value])
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_range")
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
      std.assert((i >= 0) and (i < [result_len]), "list index out of bounds in list_ispace")
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
    metadata = false,
    annotations = ast.default_annotations(),
    span = node.span,
  }

  local actions = quote
    [ispace.actions]
    [emit_debuginfo(node)]

    -- Compute size of resulting list.
    -- Currently doesn't support index spaces with multiple domains.
    std.assert(not c.legion_index_space_has_multiple_domains([cx.runtime], [ispace.value].impl),
      "list_ispace doesn't support index spaces with multiple domains")
    var ispace_domain = c.legion_index_space_get_domain([cx.runtime], [ispace.value].impl)
    var [result_len] = c.legion_domain_get_volume(ispace_domain)

    -- Allocate list.
    var size = terralib.sizeof([expr_type.element_type]) * [result_len]
    var data = c.malloc(size)
    std.assert(size == 0 or data ~= nil, "malloc failed in list_ispace")
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
      var size = terralib.sizeof([expr_type.element_type]) * len
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in gen_expr_list_from_element")
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
      var size = terralib.sizeof([expr_type.element_type]) * len
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in gen_expr_list_from_element")
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
    return empty_quote, `(value_type {
      impl = c.legion_phase_barrier_advance(
        [runtime], [context], [value].impl),
    })
  elseif std.is_dynamic_collective(value_type) then
    return empty_quote, `(value_type {
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
      var size = terralib.sizeof([value_type.element_type]) * [value].__size
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in index_access")
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

local expr_advance_list_helper = data.weak_memoize(
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
      var size = terralib.sizeof([barrier_type.element_type]) * [barrier].__size
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in adjust")
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
      var size = terralib.sizeof([value_type.element_type]) * [value].__size
      var data = c.malloc(size)
      std.assert(size == 0 or data ~= nil, "malloc failed in copy")
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
    cx, node, src_value, src_type, src_container_type, src_fields,
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
    [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
    var [launcher] = c.legion_copy_launcher_create(
      c.legion_predicate_true(), 0, [tag])
    c.legion_copy_launcher_set_provenance([launcher], [get_provenance(node)])
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
    c.legion_copy_launcher_destroy([launcher])
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
    cx, node, src_value, src_type, src_container_type, src_fields,
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
           cx, node, src_value, src_type, src_container_type, src_fields,
           dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds,
           depth + 1, op, launcher)]
      end
    end
  else
    return expr_copy_setup_region(
      cx, node, src_value, src_type, src_container_type, src_fields,
      dst_value, dst_type, dst_container_type, dst_fields,
      condition_values, condition_types, condition_kinds,
      depth, op, launcher)
  end
end

local function expr_copy_setup_list_one_to_one(
    cx, node, src_value, src_type, src_container_type, src_fields,
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
           cx, node, src_element, src_element_type, src_container_type, src_fields,
           dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds,
           depth + 1, op, launcher)]
      end
    end
  else
    return expr_copy_setup_list_one_to_many(
      cx, node, src_value, src_type, src_container_type, src_fields,
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
  actions:insert(emit_debuginfo(node))

  actions:insert(
    quote
      [expr_copy_setup_list_one_to_one(
         cx, node, src.value, src_type, src_type, node.src.fields,
         dst.value, dst_type, dst_type, node.dst.fields,
         conditions:map(function(condition) return condition.value end),
         node.conditions:map(
           function(condition)
             return std.as_read(condition.value.expr_type)
         end),
         node.conditions:map(function(condition) return condition.conditions end),
         1, node.op, launcher)]
    end)

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
end

local function expr_fill_setup_region(
    cx, dst_value, dst_type, dst_container_type, dst_fields,
    value_value, value_type, index, domain, projection_functor)
  if index then
    assert(std.is_partition(dst_type))
    assert(domain and projection_functor)
  else
    assert(std.is_region(dst_type))
  end
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

      if index then
        actions:insert(quote
          var buffer : dst_field_type = [fill_value]
          c.legion_runtime_index_fill_field_with_domain(
            [cx.runtime], [cx.context], [domain], [dst_value].impl, [dst_parent],
            dst_field_id, &buffer, [terralib.sizeof(dst_field_type)],
            [projection_functor], c.legion_predicate_true(), 0, 0)
        end)
      else
        actions:insert(quote
          var buffer : dst_field_type = [fill_value]
          c.legion_runtime_fill_field(
            [cx.runtime], [cx.context], [dst_value].impl, [dst_parent],
            dst_field_id, &buffer, terralib.sizeof(dst_field_type),
            c.legion_predicate_true())
        end)
      end
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
      value_value, value_type, false)
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

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
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
    cx, node, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  assert(std.is_region(dst_type))
  assert(std.type_supports_privileges(dst_container_type))

  local actions = terralib.newlist()
  local dst_copy_fields = std.get_absolute_field_paths(
    dst_type:fspace(), dst_fields)
  local dst_parent = get_container_root(
    cx, `([dst_value].impl), dst_container_type, dst_copy_fields)
  local launcher = terralib.newsymbol(c.legion_acquire_launcher_t, "launcher")
  actions:insert(quote
    var tag = 0
    [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
    var [launcher] = c.legion_acquire_launcher_create(
      [dst_value].impl, [dst_parent],
      c.legion_predicate_true(), 0, tag)
    c.legion_acquire_launcher_set_provenance([launcher], [get_provenance(node)])
  end)
  for j, dst_copy_field in ipairs(dst_copy_fields) do
    local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
    local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

    actions:insert(quote
      c.legion_acquire_launcher_add_field([launcher], dst_field_id)
    end)
  end
  actions:insert(quote
    [expr_acquire_issue_phase_barriers(condition_values, condition_kinds, launcher)]
    c.legion_acquire_launcher_execute([cx.runtime], [cx.context], [launcher])
    c.legion_acquire_launcher_destroy([launcher])
  end)
  return actions
end

local function expr_acquire_setup_list(
    cx, node, dst_value, dst_type, dst_container_type, dst_fields,
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
           cx, node, dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds)]
      end
    end
  else
    return expr_acquire_setup_region(
      cx, node, dst_value, dst_type, dst_container_type, dst_fields,
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
       cx, node, region.value, region_type, region_type, node.region.fields,
       conditions:map(function(condition) return condition.value end),
       node.conditions:map(
         function(condition)
           return std.as_read(condition.value.expr_type)
         end),
       node.conditions:map(function(condition) return condition.conditions end))]
  end

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
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
    cx, node, dst_value, dst_type, dst_container_type, dst_fields,
    condition_values, condition_types, condition_kinds)
  assert(std.is_region(dst_type))
  assert(std.type_supports_privileges(dst_container_type))

  local actions = terralib.newlist()
  local dst_copy_fields = std.get_absolute_field_paths(
    dst_type:fspace(), dst_fields)
  local dst_parent = get_container_root(
    cx, `([dst_value].impl), dst_container_type, dst_copy_fields)
  local launcher = terralib.newsymbol(c.legion_release_launcher_t, "launcher")
  actions:insert(quote
    var tag = 0
    [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
    var [launcher] = c.legion_release_launcher_create(
      [dst_value].impl, [dst_parent],
      c.legion_predicate_true(), 0, tag)
    c.legion_release_launcher_set_provenance([launcher], [get_provenance(node)])
  end)
  for j, dst_copy_field in ipairs(dst_copy_fields) do
    local dst_field_id = cx:region_or_list(dst_container_type):field_id(dst_copy_field)
    local dst_field_type = cx:region_or_list(dst_container_type):field_type(dst_copy_field)

    actions:insert(quote
      c.legion_release_launcher_add_field([launcher], dst_field_id)
    end)
  end
  actions:insert(quote
    [expr_release_issue_phase_barriers(condition_values, condition_kinds, launcher)]
    c.legion_release_launcher_execute([cx.runtime], [cx.context], [launcher])
    c.legion_release_launcher_destroy([launcher])
  end)
  return actions
end

local function expr_release_setup_list(
    cx, node, dst_value, dst_type, dst_container_type, dst_fields,
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
           cx, node, dst_element, dst_element_type, dst_container_type, dst_fields,
           c_values, c_types, condition_kinds)]
      end
    end
  else
    return expr_release_setup_region(
      cx, node, dst_value, dst_type, dst_container_type, dst_fields,
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
       cx, node, region.value, region_type, region_type, node.region.fields,
       conditions:map(function(condition) return condition.value end),
       node.conditions:map(
         function(condition)
           return std.as_read(condition.value.expr_type)
         end),
       node.conditions:map(function(condition) return condition.conditions end))]
  end

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
end

function codegen.expr_attach_hdf5(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)
  local filename_type = std.as_read(node.filename.expr_type)
  local filename = codegen.expr(cx, node.filename):read(cx, filename_type)
  local mode_type = std.as_read(node.mode.expr_type)
  local mode = codegen.expr(cx, node.mode):read(cx, mode_type)
  local field_map_type = node.field_map and std.as_read(node.field_map.expr_type)
  local field_map = node.field_map and codegen.expr(cx, node.field_map):read(cx, field_map_type)

  if not cx.variant:get_config_options().inner and
    (not cx.region_usage or cx.region_usage[region_type])
  then
    report.info(node, "WARNING: Attach invalidates region contents. DO NOT attempt to access region after using attach.")
  end

  assert(cx:has_region(region_type))

  local fm = terralib.newsymbol(c.legion_field_map_t, "fm")
  local field_types = node.region.fields:map(function(field_path)
      return std.get_field_path(region_type:fspace(), field_path)
    end)
  local absolute_field_paths = data.flatmap(function(pair)
      local field_type, field_path = unpack(pair)
      return std.flatten_struct_fields(field_type):map(function(suffix)
        return field_path .. suffix
      end)
    end, data.zip(field_types, node.region.fields))
  local fm_setup = quote
    var [fm] = c.legion_field_map_create()
    [data.mapi(
       function(i, field_path)
         return quote
           c.legion_field_map_insert(
               [fm],
               [cx:region(region_type):field_id(field_path)],
               [(field_map and `([field_map.value][ [i-1] ])) or field_path:concat(".")])
         end
       end,
       absolute_field_paths)]
  end
  local fm_teardown = quote
    c.legion_field_map_destroy([fm])
  end

  local parent = get_container_root(
    cx, `([region.value].impl), region_type, node.region.fields)

  local new_pr = terralib.newsymbol(c.legion_physical_region_t, "new_pr")

  local actions = quote
    [region.actions]
    [filename.actions]
    [mode.actions]
    [field_map and field_map.actions or empty_quote]
    [emit_debuginfo(node)]

    [fm_setup]
    var [new_pr] = c.legion_runtime_attach_hdf5(
      [cx.runtime], [cx.context],
      [filename.value], [region.value].impl, [parent], [fm], [mode.value])
    [fm_teardown]

    [absolute_field_paths:map(
       function(field_path)
         return quote
           -- FIXME: This is redundant (since the same physical region
           -- will generally show up more than once. At any rate, it
           -- would be preferable not to have to do this at all.
           [cx:region(region_type):physical_region(field_path)] = [new_pr]
         end
       end)]
  end

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
end

function codegen.expr_detach_hdf5(cx, node)
  local region_type = std.as_read(node.region.expr_type)
  local region = codegen.expr_region_root(cx, node.region):read(cx, region_type)

  if not cx.variant:get_config_options().inner and
    (not cx.region_usage or cx.region_usage[region_type])
  then
    report.info(node, "WARNING: Detach invalidates region contents. DO NOT attempt to access region after using detach.")
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
  local pr_set = data.newmap()
  for _, field_path in ipairs(full_fields) do
    pr_set[cx:region(region_type):physical_region(field_path)] = true
  end
  local pr_list = pr_set:map_keys(function(x) return x end)

  local actions = quote
    [region.actions]
    [emit_debuginfo(node)]

    [pr_list:map(
       function(pr)
         return quote
           c.legion_runtime_detach_hdf5([cx.runtime], [cx.context], [pr])
           c.legion_physical_region_destroy([pr])
         end
       end)]
  end

  return values.value(node, expr.just(actions, empty_quote), terralib.types.unit)
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
      std.assert([region.value].__size > 0, "attempting to allocate scratch fields for empty list")
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
             fsa, terralib.sizeof(field_type), c.AUTO_GENERATE_ID)
           [attach_name_and_type(cx, field_space, `([field_ids][i]), field_name, field_type)]
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
  assert(cx:has_region_or_list(region_type))

  local old_field_ids = cx:region_or_list(region_type).field_ids
  local new_field_ids = old_field_ids:copy()
  for i, field_path in ipairs(node.region.fields) do
    new_field_ids[field_path] = `([field_ids.value][ [i-1] ])
  end

  local field_paths, field_types = std.flatten_struct_fields(region_type:fspace())

  local new_field_id_array = terralib.newsymbol(c.legion_field_id_t[#field_paths], "new_field_ids")
  actions = quote
    [actions]
    var [new_field_id_array] = arrayof(
      c.legion_field_id_t,
      [field_paths:map(function(field_path) return new_field_ids[field_path] end)])
  end

  local old_fields_are_scratch = cx:region_or_list(region_type).fields_are_scratch
  local new_fields_are_scratch = old_fields_are_scratch:copy()
  for i, field_path in ipairs(node.region.fields) do
    new_fields_are_scratch[field_path] = true
  end

  local value = expr.once_only(
    actions,
    std.implicit_cast(region_type, expr_type, region.value),
    expr_type)

  if std.is_region(region_type) then
    cx:add_region_root(
      expr_type, value.value,
      cx:region(region_type).field_paths,
      cx:region(region_type).privilege_field_paths,
      cx:region(region_type).field_privileges,
      cx:region(region_type).field_types,
      new_field_ids,
      new_field_id_array,
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
      new_field_id_array,
      new_fields_are_scratch)
  else
    assert(false)
  end

  return values.value(node, value, expr_type)
end

function codegen.expr_unary(cx, node)
  local expr_type = std.as_read(node.expr_type)
  assert(not std.is_future(expr_type)) -- This is handled in optimize_future now

  local rhs = codegen.expr(cx, node.rhs):read(cx, expr_type)
  local actions = quote
    [rhs.actions];
    [emit_debuginfo(node)]
  end
  if std.as_read(node.rhs.expr_type):isarray() then
    local result = terralib.newsymbol(expr_type, "result")
    actions = quote
      [actions];
      var [result]
      for i = 0, [expr_type.N] do
        [result][i] = [std.quote_unary_op(node.op, `([rhs.value][i]))]
      end
    end
    return values.value(
      node,
      expr.just(actions, result),
      expr_type)
  else
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

    local lhs_type = std.as_read(node.lhs.expr_type)
    local rhs_type = std.as_read(node.rhs.expr_type)
    local create_partition
    if node.op == "-" then
      create_partition = c.legion_index_partition_create_by_difference
    elseif node.op == "&" then
      if std.is_region(lhs_type) then
        create_partition = c.legion_index_partition_create_by_intersection_mirror
      else
        create_partition = c.legion_index_partition_create_by_intersection
      end
    elseif node.op == "|" then
      create_partition = c.legion_index_partition_create_by_union
    else
      assert(false)
    end

    local partition_type = std.as_read(node.expr_type)
    local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
    local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")

    if std.is_region(lhs_type) then
      actions = quote
        [actions]
        var is = [lhs.value].impl.index_space
        var [ip] = [create_partition](
          [cx.runtime], [cx.context],
          is, [rhs.value].impl.index_partition,
          [(partition_type:is_disjoint() and c.DISJOINT_KIND) or c.COMPUTE_KIND],
          c.AUTO_GENERATE_ID, false)
        var [lp] = c.legion_logical_partition_create_by_tree(
          [cx.runtime], [cx.context],
          [ip], [lhs.value].impl.field_space, [lhs.value].impl.tree_id)
      end
    else
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
          c.AUTO_GENERATE_ID)
        var [lp] = c.legion_logical_partition_create_by_tree(
          [cx.runtime], [cx.context],
          [ip], [lhs.value].impl.field_space, [lhs.value].impl.tree_id)
      end
    end

    return values.value(
      node,
      expr.once_only(actions, `(partition_type { impl = [lp] }), partition_type),
      partition_type)
  elseif std.is_ispace(expr_type) then
    local ispace_op = nil

    if node.op == "&" then
      ispace_op = c.legion_index_space_intersection
    elseif node.op == "|" then
      ispace_op = c.legion_index_space_union
    else
      assert(false, "unreachable")
    end

    local lhs = codegen.expr(cx, node.lhs):read(cx, node.lhs.expr_type)
    local rhs = codegen.expr(cx, node.rhs):read(cx, node.rhs.expr_type)
    local result = terralib.newsymbol(expr_type)
    local bounds_actions, domain, bounds = index_space_bounds(cx, `([result].impl), expr_type)
    local actions = quote
      [lhs.actions];
      [rhs.actions];
      [emit_debuginfo(node)]
      var [result]
      do
        var args : c.legion_index_space_t[2]
        args[0] = [lhs.value].impl
        args[1] = [rhs.value].impl
        [result] = [expr_type] { impl = [ispace_op]([cx.runtime], [cx.context], args, 2) }
      end
      [bounds_actions]
    end
    cx:add_ispace_root(expr_type, `([result].impl), domain, bounds)

    return values.value(
      node,
      expr.just(actions, result),
      expr_type)
  else
    assert(not std.is_future(expr_type)) -- This is handled in optimize_future now
    local lhs = codegen.expr(cx, node.lhs):read(cx, node.lhs.expr_type)
    local rhs = codegen.expr(cx, node.rhs):read(cx, node.rhs.expr_type)
    local actions = quote
      [lhs.actions];
      [rhs.actions];
      [emit_debuginfo(node)]
    end

    local expr_type = std.as_read(node.expr_type)
    local lhs_type = std.as_read(node.lhs.expr_type)
    local rhs_type = std.as_read(node.rhs.expr_type)
    if std.is_region(rhs_type) or std.is_ispace(rhs_type) then
      assert(node.op == "<=")
      local domain = nil
      if std.is_ispace(rhs_type) then
        domain = cx:ispace(rhs_type).domain
      elseif std.is_region(rhs_type) then
        domain = cx:ispace(rhs_type:ispace()).domain
      end
      assert(domain ~= nil)
      local result = terralib.newsymbol(bool)
      actions = quote
        [actions]
        var [result] = c.legion_domain_contains(domain, [lhs.value]:to_domain_point())
      end
      return values.value(
        node,
        expr.just(actions, result),
        expr_type)
    else
    end
    if std.as_read(lhs_type):isarray() and
       std.as_read(rhs_type):isarray() and
       node.op ~= "-"
    then
      assert(expr_type:isarray())
      local result = terralib.newsymbol(expr_type, "result")
      actions = quote
        [actions];
        var [result]
        for i = 0, [expr_type.N] do
          [result][i] = [std.quote_binary_op(node.op, `([lhs.value][i]), `([rhs.value][i]))]
        end
      end
      return values.value(
        node,
        expr.just(actions, result),
        expr_type)
    else
      return values.value(
        node,
        expr.once_only(actions, std.quote_binary_op(node.op, lhs.value, rhs.value), expr_type),
        expr_type)
    end
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

function codegen.expr_address_of(cx, node)
  local value = codegen.expr(cx, node.value)
  return value:address(cx)
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
    if not size_actions then
      size_actions = empty_quote
      size_value = 0
    end
    local ser_actions = std.serialize(
      content_type, content_value, buffer, `(&[data_ptr]))
    local actions = quote
      [actions]
      [size_actions]
      var buffer_size = terralib.sizeof(content_type) + [size_value]
      var [buffer] = c.malloc(buffer_size)
      std.assert(buffer_size == 0 or [buffer] ~= nil, "malloc failed in future")
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

function codegen.expr_import_ispace(cx, node)
  local ispace_type = node.expr_type
  local value = codegen.expr(cx, node.value):read(cx)
  local is = terralib.newsymbol(c.legion_index_space_t, "is")
  local i = terralib.newsymbol(ispace_type, "i")

  local bounds_actions, domain, bounds = index_space_bounds(cx, is, ispace_type)
  cx:add_ispace_root(ispace_type, is, domain, bounds)

  local actions = quote
    [value.actions];
    [emit_debuginfo(node)];
    var [is] = [value.value]
    var [i] = [ispace_type] { impl = [is] }
    [bounds_actions];
    -- Consistency check on the imported index space
    do
      std.assert_error([domain].dim == [ispace_type.dim],
        [get_source_location(node) .. ": " .. pretty.entry_expr(node.value) ..
          " is not a " ..  ispace_type.dim .. "D index space"])
      std.assert_error(not
        std.c.legion_index_space_has_parent_index_partition([cx.runtime], [is]),
        [get_source_location(node) .. ": cannot import a subspace"])
    end
  end
  return values.value(
    node,
    expr.just(actions, i),
    ispace_type)
end

function codegen.expr_import_region(cx, node)
  local region_type = std.as_read(node.expr_type)
  local fspace_type = region_type:fspace()
  local index_type = region_type:ispace().index_type

  local ispace = codegen.expr(cx, node.ispace):read(cx)
  local value = codegen.expr(cx, node.value):read(cx)
  local src_field_ids = codegen.expr(cx, node.field_ids):read(cx)

  local actions = quote
    [ispace.actions];
    [value.actions];
    [src_field_ids.actions];
    [emit_debuginfo(node)];
  end

  local r = terralib.newsymbol(region_type, "r")
  local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
  local pr = terralib.newsymbol(c.legion_physical_region_t, "pr")

  local field_ids_type = std.as_read(node.field_ids.expr_type)
  assert(field_ids_type:isarray())

  local field_id_array = terralib.newsymbol(c.legion_field_id_t[field_ids_type.N], "field_ids")
  actions = quote
    [actions];
    var [field_id_array]
    for i = 0, [field_ids_type.N] do
      [field_id_array][i] = [src_field_ids.value][i]
    end
  end

  local field_paths, field_types = std.flatten_struct_fields(fspace_type)
  local field_privileges = field_paths:map(function(_) return "reads_writes" end)
  local field_id_offset = 0
  local field_ids = field_paths:map(function(_)
    local field_id = `([field_id_array][ [field_id_offset] ])
    field_id_offset = field_id_offset + 1
    return field_id
  end)
  local fields_are_scratch = field_paths:map(function(_) return false end)
  local physical_regions = field_paths:map(function(_) return pr end)
  local pr_actions, base_pointers, strides = unpack(data.zip(unpack(
    data.zip(field_types, field_paths, field_ids, field_privileges):map(
      function(field)
        local field_type, field_path, field_id, field_privilege = unpack(field)
        return terralib.newlist({
          physical_region_get_base_pointer(cx, region_type, index_type, field_type, field_path, pr, field_id)})
  end))))
  pr_actions = pr_actions or terralib.newlist()
  base_pointers = base_pointers or terralib.newlist()
  strides = strides or terralib.newlist()

  cx:add_region_root(region_type, r,
                     field_paths,
                     terralib.newlist({field_paths}),
                     data.dict(data.zip(field_paths, field_privileges)),
                     data.dict(data.zip(field_paths, field_types)),
                     data.dict(data.zip(field_paths, field_ids)),
                     field_id_array,
                     data.dict(data.zip(field_paths, fields_are_scratch)),
                     data.dict(data.zip(field_paths, physical_regions)),
                     data.dict(data.zip(field_paths, base_pointers)),
                     data.dict(data.zip(field_paths, strides)))

  -- Consistency check on the imported logical region handle
  local check_actions = quote
    std.assert_error(
      [eq_struct(c.legion_index_space_t, `([ispace.value].impl), `([lr].index_space))],
      [get_source_location(node) .. ": " .. pretty.entry_expr(node.ispace) ..
      " is not the index space of " .. pretty.entry_expr(node.value)])
    std.assert_error(not
      std.c.legion_logical_region_has_parent_logical_partition([cx.runtime], [lr]),
      [get_source_location(node) .. ": cannot import a subregion"])
    std.assert_error(
      std.c.legion_field_space_has_fields([cx.runtime], [cx.context], [lr].field_space,
        [field_id_array], [field_ids_type.N]),
      [get_source_location(node) .. ": found an invalid field id"])
    [data.zip(field_ids, field_types):map(
      function(pair)
        local field_id, field_type = unpack(pair)
        return quote
          std.assert_error(
            std.c.legion_field_id_get_size([cx.runtime], [cx.context], [lr].field_space, [field_id]) ==
            [sizeof(field_type)],
            [get_source_location(node) .. ": field size does not match"])
        end
      end)]
  end

  actions = quote
    [actions];
    var [lr] = [value.value]
    var [r] = [region_type] { impl = [lr] }
    [check_actions];
    [check_imported(cx, node, lr)];
    [tag_imported(cx, lr)]
  end

  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  if not cx.variant:get_config_options().inner and
    (not cx.region_usage or cx.region_usage[region_type])
  then
    actions = quote
      [actions];
      var [tag] = 0
      [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
      var il = c.legion_inline_launcher_create_logical_region(
        [lr], c.READ_WRITE, c.EXCLUSIVE, [lr], 0, false, 0, [tag]);
      c.legion_inline_launcher_set_provenance(il, [get_provenance(node)])
      [data.zip(field_ids, field_types):map(
         function(field)
           local field_id, field_type = unpack(field)
           if std.is_regent_array(field_type) then
             return quote
               for idx = 0, [field_type.N] do
                 c.legion_inline_launcher_add_field(il, [field_id] + idx, true)
               end
             end
           else
             return `(c.legion_inline_launcher_add_field(il, [field_id], true))
           end
         end)]
      var [pr] = c.legion_inline_launcher_execute([cx.runtime], [cx.context], il)
      c.legion_inline_launcher_destroy(il)
      c.legion_physical_region_wait_until_valid([pr])
      [pr_actions]
    end
  else
    actions = quote
      [actions];
      c.legion_runtime_unmap_all_regions([cx.runtime], [cx.context])
      var [pr]
    end
  end

  return values.value(
    node,
    expr.just(actions, r),
    region_type)
end

function codegen.expr_import_partition(cx, node)
  local partition_type = std.as_read(node.expr_type)
  local region = codegen.expr(cx, node.region):read(cx)
  local colors = codegen.expr(cx, node.colors):read(cx)
  local colors_type = std.as_read(node.colors.expr_type)
  local value = codegen.expr(cx, node.value):read(cx)

  local actions = quote
    [region.actions];
    [colors.actions];
    [value.actions];
    [emit_debuginfo(node)];
  end

  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")

  local check_actions = empty_quote
  if partition_type:is_disjoint() then
    check_actions = quote
      std.assert_error(
        c.legion_index_partition_is_disjoint([cx.runtime], [lp].index_partition),
        [get_source_location(node) .. ": " .. pretty.entry_expr(node.value) ..
        " is not a disjoint partition"])
    end
  end
  local dim = math.max(colors_type.dim, 1)
  check_actions = quote
    [check_actions];
    do
      var parent = std.c.legion_logical_partition_get_parent_logical_region([cx.runtime], [lp])
      std.assert_error([eq_struct(c.legion_logical_region_t, `([region.value].impl), parent)],
        [get_source_location(node) .. ": " .. pretty.entry_expr(node.value) ..
        " is not a logical partition of " .. pretty.entry_expr(node.region)])
    end
    do
      var cs = std.c.legion_index_partition_get_color_space([cx.runtime], [lp].index_partition)
      var domain = std.c.legion_index_space_get_domain([cx.runtime], cs)
      std.assert_error(domain.dim == [dim],
        [get_source_location(node) .. ": " .. pretty.entry_expr(node.value) ..
        " does not have a " .. tostring(dim) .. "D color space"])
    end
  end

  actions = quote
    [actions];
    var [lp] = [value.value]
    [check_actions];
  end

  return values.value(
    node,
    expr.once_only(actions, `([partition_type] { impl = [lp] }), partition_type),
    partition_type)
end

function codegen.expr_import_cross_product(cx, node)
  local partitions = node.partitions:map(function(p) return codegen.expr(cx, p):read(cx) end)
  local colors = codegen.expr(cx, node.colors):read(cx)
  local value = codegen.expr(cx, node.value):read(cx)
  local expr_type = std.as_read(node.expr_type)

  local actions = empty_quote

  for _, p in ipairs(partitions) do
    actions = quote
      [actions];
      [p.actions];
    end
  end

  actions = quote
    [actions];
    [colors.actions];
    [value.actions];
  end

  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local lr = cx:region(expr_type:parent_region()).logical_region

  actions = quote
    [actions];
    var ip = c.legion_terra_index_cross_product_get_partition(value.value)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], lr.impl, ip)
  end

  return values.value(node,
    expr.once_only(
      actions,
        `(expr_type { impl = [lp], product = [value.value], colors = [colors.value] }),
         expr_type),
    expr_type)
end

function codegen.expr_projection(cx, node)
  local orig_region_value = codegen.expr(cx, node.region):read(cx)
  local orig_region_type = std.as_read(node.region.expr_type)

  local region_type = std.as_read(node.expr_type)

  local orig_field_paths = std.get_absolute_field_paths(orig_region_type:fspace(),
      node.field_mapping:map(function(entry) return entry[1] end))
  local field_paths = std.get_absolute_field_paths(region_type:fspace(),
      node.field_mapping:map(function(entry) return entry[2] end))

  local orig_region = cx:region(orig_region_type)
  local result = terralib.newsymbol(region_type, "r")
  local field_id_array_buffer =
    terralib.newsymbol(&c.legion_field_id_t[#orig_field_paths], "field_ids")
  local field_id_array = `(@[field_id_array_buffer])
  local index_mapping = data.dict(data.zip(orig_region.field_paths,
                                           data.range(0, #orig_region.field_paths)))
  local actions = quote
    [orig_region_value.actions]
    var [result] = [region_type] { impl = [orig_region_value.value].impl }
    var [field_id_array_buffer] =
      [&c.legion_field_id_t[#orig_field_paths]](c.malloc([#orig_field_paths] *
                                                         [terralib.sizeof(c.legion_field_id_t)]))
    [data.zip(data.range(0, #orig_field_paths), orig_field_paths):map(function(pair)
      local idx, orig_field_path = unpack(pair)
      local orig_idx = index_mapping[orig_field_path]
      assert(orig_idx ~= nil)
      return quote
        [field_id_array][ [idx] ] = [orig_region.field_id_array][ [orig_idx] ]
      end
    end)]
  end
  cx:add_cleanup_item(quote c.free([field_id_array_buffer]) end)

  local field_mapping = data.dict(data.zip(field_paths, orig_field_paths))
  local function map_dict(dict)
    return data.dict(data.zip(field_paths,
                              field_paths:map(function(field_path)
                                return dict[field_mapping[field_path]]
                              end)))
  end

  if std.type_eq(orig_region.root_region_type, orig_region_type) then
    cx:add_region_root(region_type, result,
                       field_paths,
                       terralib.newlist({field_paths}),
                       map_dict(orig_region.field_privileges),
                       map_dict(orig_region.field_types),
                       map_dict(orig_region.field_ids),
                       field_id_array,
                       map_dict(orig_region.fields_are_scratch),
                       map_dict(orig_region.physical_regions),
                       map_dict(orig_region.base_pointers),
                       map_dict(orig_region.strides))
  else
    local root_region_type = orig_region.root_region_type
    local parent_region_type =
      std.region(root_region_type.ispace_symbol, region_type:fspace())
    cx:add_region_root(parent_region_type,
                       cx:region(root_region_type).logical_region,
                       field_paths,
                       terralib.newlist({field_paths}),
                       map_dict(orig_region.field_privileges),
                       map_dict(orig_region.field_types),
                       map_dict(orig_region.field_ids),
                       field_id_array,
                       map_dict(orig_region.fields_are_scratch),
                       map_dict(orig_region.physical_regions),
                       map_dict(orig_region.base_pointers),
                       map_dict(orig_region.strides))
    cx:add_region_subregion(region_type, result, parent_region_type)
  end

  return values.value(
    node,
    expr.once_only(actions, result, region_type),
    region_type)
end

function codegen.expr(cx, node)
  if node:is(ast.typed.expr.Internal) then
    return codegen.expr_internal(cx, node)

  elseif node:is(ast.typed.expr.ID) then
    return codegen.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
    return codegen.expr_constant(cx, node)

  elseif node:is(ast.typed.expr.Global) then
    return codegen.expr_global(cx, node)

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

  elseif node:is(ast.typed.expr.RawFuture) then
    return codegen.expr_raw_future(cx, node)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return codegen.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return codegen.expr_raw_runtime(cx, node)

  elseif node:is(ast.typed.expr.RawTask) then
    return codegen.expr_raw_task(cx, node)

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

  elseif node:is(ast.typed.expr.PartitionByRestriction) then
    return codegen.expr_partition_by_restriction(cx, node)

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

  elseif node:is(ast.typed.expr.AddressOf) then
    return codegen.expr_address_of(cx, node)

  elseif node:is(ast.typed.expr.Future) then
    return codegen.expr_future(cx, node)

  elseif node:is(ast.typed.expr.FutureGetResult) then
    return codegen.expr_future_get_result(cx, node)

  elseif node:is(ast.typed.expr.ImportIspace) then
    return codegen.expr_import_ispace(cx, node)

  elseif node:is(ast.typed.expr.ImportRegion) then
    return codegen.expr_import_region(cx, node)

  elseif node:is(ast.typed.expr.ImportPartition) then
    return codegen.expr_import_partition(cx, node)

  elseif node:is(ast.typed.expr.ImportCrossProduct) then
    return codegen.expr_import_cross_product(cx, node)

  elseif node:is(ast.typed.expr.Projection) then
    return codegen.expr_projection(cx, node)

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

local function cleanup(cx)
  return as_quote(cx:get_cleanup_items())
end

local function cleanup_after(cx, block)
  local result = terralib.newlist({as_quote(block)})
  result:insert(cleanup(cx))
  return as_quote(result)
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
  local body_cx = cx:new_local_scope(nil, nil, nil, nil, nil, nil, break_label)
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
  local cx = cx:new_local_scope(nil, nil, nil, symbol, bounds:map(function(b) return b.value end), nil, break_label)
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
local function collect_symbols(cx, node, cuda)
  local result = terralib.newlist()

  local undefined =  data.newmap()
  local reduction_variables = {}
  local defined =  data.map_from_table({ [node.symbol] = true })
  local accesses = data.newmap()
  local function collect_symbol_pre(node)
    if ast.is_node(node) then
      if node:is(ast.typed.stat.Var) then
        defined[node.symbol] = true
      elseif node:is(ast.typed.stat.ForNum) or
             node:is(ast.typed.stat.ForList) then
        defined[node.symbol] = true
      end
    end
  end
  local function collect_symbol_post(node)
    if ast.is_node(node) then
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
  local base_pointers = data.newmap()
  local strides = data.newmap()
  local lrs = data.newmap()
  for node, _ in accesses:items() do
    local value_type = std.as_read(node.expr_type)
    node.expr_type:bounds():map(function(region)
      local prefix = node.expr_type.field_path
      local field_paths = std.flatten_struct_fields(value_type)
      local absolute_field_paths = field_paths:map(
        function(field_path) return prefix .. field_path end)
      absolute_field_paths:map(function(field_path)
        field_path = std.extract_privileged_prefix(region:fspace(), field_path)
        base_pointers[cx:region(region):base_pointer(field_path)] = true
        local stride = cx:region(region):stride(field_path)
        for idx = 2, #stride do strides[stride[idx]] = true end
        lrs[cx:region(region).logical_region] = true
      end)
    end)
  end

  for base_pointer, _ in base_pointers:items() do
    result:insert(base_pointer)
  end
  for stride, _ in strides:items() do
    result:insert(stride) end
  for symbol, _ in undefined:items() do
    if std.is_symbol(symbol) then symbol = symbol:getsymbol() end
    result:insert(symbol)
  end
  if not cuda then
    result:insert(cx.runtime)
    result:insert(cx.context)
    for lr, _ in lrs:items() do
      result:insert(lr)
    end
  end

  return result, reduction_variables, lrs
end

function codegen.stat_for_list(cx, node)
  local symbol = node.symbol:getsymbol()
  local cx = cx:new_local_scope()
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local break_label = terralib.newlabel()
  local cx = cx:new_local_scope(nil, nil, nil, symbol, value.value, value_type, break_label)

  -- Exit early when the iteration space is a regent list
  if std.is_list(value_type) then
    local block = cleanup_after(cx, codegen.block(cx, node.block))
    return quote
      for i = 0, [value.value].__size do
        var [symbol] = [value_type:data(value.value)][i]
        do
          [block]
        end
      end
      ::[break_label]::
    end
  end

  local index_type = node.symbol:gettype()
  if std.is_bounded_type(index_type) then
    index_type = index_type.index_type
  end

  -- Retrieve dimension-specific handle types and functions from the C API
  local dim = data.max(index_type.dim, 1)
  local rect_type = c["legion_rect_" .. tostring(dim) .. "d_t"]
  local rect_it_type =
    c["legion_rect_in_domain_iterator_" .. tostring(dim) .. "d_t"]
  local rect_it_create =
    c["legion_rect_in_domain_iterator_create_" .. tostring(dim) .. "d"]
  local rect_it_destroy =
    c["legion_rect_in_domain_iterator_destroy_" .. tostring(dim) .. "d"]
  local rect_it_valid =
    c["legion_rect_in_domain_iterator_valid_" .. tostring(dim) .. "d"]
  local rect_it_step =
    c["legion_rect_in_domain_iterator_step_" .. tostring(dim) .. "d"]
  local rect_it_get =
    c["legion_rect_in_domain_iterator_get_rect_" .. tostring(dim) .. "d"]

  -- Create variables for the outer loop that iterates over rectangles in a domain
  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local rect = terralib.newsymbol(rect_type, "rect")
  local rect_it = terralib.newsymbol(rect_it_type, "rect_it")

  -- Check if the loop needs the CUDA or OpenMP code generation
  local cuda = cx.variant:is_cuda() and
               (node.metadata and node.metadata.parallelizable) and
               not node.annotations.cuda:is(ast.annotation.Forbid)
  local openmp = not cx.variant:is_cuda() and
                 openmphelper.check_openmp_available() and
                 node.annotations.openmp:is(ast.annotation.Demand)

  if node.annotations.openmp:is(ast.annotation.Demand) then
    local available, error_message = openmphelper.check_openmp_available()
    if std.config["openmp"] ~= 0 and not available then
      report.warn(node,
        "ignoring pragma since " .. error_message)
    end
  end

  if openmp and not cx.leaf then
    if std.config["openmp"] ~= 0 then
      report.error(node,
        "OpenMP code generation failed since the OpenMP loop is in a non-leaf task")
    end
  end


  -- Code generation for the loop body
  local block = node.block

  if cuda then
    -- If the loop needs the CUDA code generation, we replace calls to CPU math functions
    -- with their GPU counterparts.
    block = ast.map_node_postorder(function(node)
      if node:is(ast.typed.expr.Call) then
        local value = node.fn.value
        if std.is_math_fn(value) then
          return node { fn = node.fn { value = gpuhelper.get_gpu_variant(value) } }
        elseif value == array or value == arrayof then
          return node
        else
          assert(false, "this case should have been catched by the checker")
        end
      else
        return node
      end
    end, block)

    local cuda_cx = gpuhelper.new_kernel_context(node)
    cx:add_codegen_context("cuda", cuda_cx)
    block = gpuhelper.optimize_loop(cuda_cx, node, block)
    if std.config["cuda-licm"] then
      block = licm.entry(node.symbol, block)
    end
  end

  -- If we do either CUDA or OpenMP code generation,
  -- we remember the loop variable of the loop that we parallelize over
  if cuda or openmp then cx:set_loop_symbol(node.symbol) end

  local fields = index_type.fields
  local indices = terralib.newlist()
  if fields then
    indices:insertall(fields:map(function(field)
      return terralib.newsymbol(c.coord_t, tostring(field))
    end))
  else
    indices:insert(terralib.newsymbol(c.coord_t, "x"))
  end

  local symbol_setup = nil
  if index_type.dim == 0 then
    symbol_setup = quote
      var [symbol] = [symbol.type]{
        __ptr = ptr { __ptr = c.legion_ptr_t { value = [ indices[1] ] } }
      }
    end
  elseif index_type.dim == 1 then
    symbol_setup = quote
      var [symbol] = [symbol.type]{ __ptr = [ indices[1] ] }
    end
  else
    symbol_setup = quote
      var [symbol] = [symbol.type] { __ptr = [index_type.impl_type]{ [indices] } }
    end
  end
  local body = quote
    [symbol_setup]
    do
      [cleanup_after(cx, codegen.block(cx, block))]
    end
  end
  local preamble = empty_quote
  local postamble = empty_quote

  if not (cuda or openmp) then
    -- TODO: Need to codegen these dimension loops in the right order
    --       based on the instance layout
    for i = 1, dim do
      local rect_i = i - 1 -- C is zero-based, Lua is one-based
      body = quote
        for [ indices[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
          [body]
        end
      end
    end

  else
    local symbols, reductions, lrs = collect_symbols(cx, node, cuda)
    if openmp then
      symbols:insert(rect)
      local can_change = { [rect] = true }
      local arg_type, mapping = openmphelper.generate_argument_type(symbols, reductions)
      local arg = terralib.newsymbol(&arg_type, "arg")
      local worker_init, launch_init, launch_update =
        openmphelper.generate_argument_init(arg, arg_type, mapping, can_change, reductions)
      local worker_cleanup =
        openmphelper.generate_worker_cleanup(arg, arg_type, mapping, reductions)
      local launcher_cleanup =
        openmphelper.generate_launcher_cleanup(arg, arg_type, mapping, reductions)

      -- TODO: Need to codegen these dimension loops in the right order
      --       based on the instance layout
      for i = 1, dim do
        local rect_i = i - 1 -- C is zero-based, Lua is one-based
        if i ~= dim then
          body = quote
            for [ indices[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
              [body]
            end
          end
        else
          local start_idx = terralib.newsymbol(int64, "start_idx")
          local end_idx = terralib.newsymbol(int64, "end_idx")
          body = quote
            [openmphelper.generate_preamble(rect, rect_i, start_idx, end_idx)]
            for [ indices[i] ] = [start_idx], [end_idx] do
              [body]
            end
          end
        end
      end

      local terra omp_worker(data : &opaque)
        var [arg] = [&arg_type](data)
        [worker_init]
        [body]
        [worker_cleanup]
      end

      body = quote
        [launch_update]
        [openmphelper.launch]([omp_worker], [arg], [openmphelper.get_max_threads](), 0)
      end
      preamble = launch_init
      postamble = launcher_cleanup

      -- Legion's safe cast is not thread-safe, because it might update the internal cache
      -- Here we force the cache to get initialized so all OpenMP worker threads would only
      -- read the cache.
      if std.config["bounds-checks"] then
        preamble = quote
          [preamble];
          [lrs:map_list(function(lr)
            local index_type = lr.type:ispace().index_type
            if index_type:is_opaque() then index_type = std.int1d end
            return quote
              var p = [index_type:zero()]
              c.legion_domain_point_safe_cast([cx.runtime], [cx.context],
                p:to_domain_point(), [lr].impl)
            end
          end)]
        end
      end
    else -- if openmp then
      local cuda_cx = cx:get_codegen_context("cuda")
      assert(cuda)
      assert(not std.config["bounds-checks"], "bounds checks with CUDA are unsupported")
      local lower_bounds = indices:map(function(symbol)
        return terralib.newsymbol(c.coord_t, "lo_" .. symbol.id)
      end)
      local counts = indices:map(function(symbol)
        return terralib.newsymbol(c.coord_t, "cnt_" .. symbol.id)
      end)
      local args = data.filter(function(arg) return reductions[arg] == nil end, symbols)
      local shared_mem_size = gpuhelper.compute_reduction_buffer_size(cuda_cx, node, reductions)
      local device_ptrs, device_ptrs_map, host_ptrs_map, host_preamble, buffer_cleanups =
        gpuhelper.generate_reduction_preamble(cuda_cx, reductions)
      local kernel_preamble, kernel_postamble =
        gpuhelper.generate_reduction_kernel(cuda_cx, reductions, device_ptrs_map)
      local host_postamble =
        gpuhelper.generate_reduction_postamble(cuda_cx, reductions, device_ptrs_map, host_ptrs_map)
      args:insertall(lower_bounds)
      args:insertall(counts)
      args:insertall(device_ptrs)

      local need_spiil = gpuhelper.check_arguments_need_spill(args)

      local kernel_param_pack = empty_quote
      local kernel_param_unpack = empty_quote
      local spill_cleanup = empty_quote
      if need_spiil then
        local arg = nil
        kernel_param_pack, kernel_param_unpack, spill_cleanup, arg =
          gpuhelper.generate_argument_spill(args)
        args = terralib.newlist({arg})
      else
        -- Sort arguments in descending order of sizes to avoid misalignment
        -- (which causes a segfault inside the driver API)
        args:sort(function(s1, s2)
            local t1 = s1.type
            if t1:isarray() then t1 = t1.type end
            local t2 = s2.type
            if t2:isarray() then t2 = t2.type end
            return sizeof(t1) > sizeof(t2)
        end)
      end

      -- Reconstruct indices from the global thread id
      local index_inits = terralib.newlist()
      local tid = terralib.newsymbol(c.size_t, "tid")
      local offsets = indices:map(function(symbol)
        return terralib.newsymbol(c.coord_t, "off_" .. symbol.id)
      end)
      local count = counts[1]
      for idx = 2, #indices do
        count = `([count] * [ counts[idx] ])
      end
      index_inits:insert(quote
        var [tid] = [gpuhelper.global_thread_id()]
        if [tid] >= [count] then return end
      end)
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
      local terra kernel([args])
        [kernel_param_unpack]
        [kernel_preamble]
        [index_inits]
        [body]
        [kernel_postamble]
      end

      -- Register the kernel function to JIT
      cx.task_meta:get_cuda_variant():add_cuda_kernel(kernel)

      if std.config["cuda-pretty-kernels"] then
        io.write("===== CUDA kernel @ " .. node.span.source .. ":" .. node.span.start.line .. " =====\n")
        kernel:printpretty(false)
      end

      local count = terralib.newsymbol(c.size_t, "count")
      local kernel_call =
        gpuhelper.codegen_kernel_call(cuda_cx, kernel, count, args, shared_mem_size, false)

      local bounds_setup = terralib.newlist()
      bounds_setup:insert(quote var [count] = 1 end)
      for idx = 1, dim do
        bounds_setup:insert(quote
          var [ lower_bounds[idx] ], [ counts[idx] ] =
            [rect].lo.x[ [idx - 1] ], [rect].hi.x[ [idx - 1] ] - [rect].lo.x[ [idx - 1] ] + 1
          [count] = [count] * [ counts[idx] ]
        end)
      end

      body = quote
        [bounds_setup]
        [kernel_param_pack]
        [kernel_call]
        [spill_cleanup]
      end

      preamble = host_preamble
      postamble = quote [host_postamble]; [buffer_cleanups]; end
    end  -- if openmp then
  end


  if not std.is_rect_type(value_type) then
    local is = nil
    if std.is_ispace(value_type) then
      is = `([value.value].impl)
    elseif std.is_region(value_type) then
      assert(cx:has_ispace(value_type:ispace()))
      is = `([value.value].impl.index_space)
    else
      assert(false)
    end

    return quote
      [value.actions]
      var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
      var [rect_it] = [rect_it_create]([domain])
      [preamble]
      while [rect_it_valid]([rect_it]) do
        var [rect] = [rect_it_get]([rect_it])
        [body]
        [rect_it_step]([rect_it])
      end
      ::[break_label]::
      [rect_it_destroy]([rect_it])
      [postamble]
    end

  else
    return quote
      [value.actions]
      [preamble]
      var [rect] = [value.value]
      [body]
      ::[break_label]::
      [postamble]
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
        metadata = node.orig_metadata,
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

  local ispace_type, is
  if std.is_region(value_type) then
    ispace_type = value_type:ispace()
    assert(cx:has_ispace(ispace_type))
    is = `([value.value].impl.index_space)
  else
    ispace_type = value_type
    is = `([value.value].impl)
  end
  local index_type = ispace_type.index_type

  -- Retrieve dimension-specific handle types and functions from the C API
  local dim = data.max(ispace_type.dim, 1)
  local rect_type = c["legion_rect_" .. tostring(dim) .. "d_t"]
  local rect_it_type =
    c["legion_rect_in_domain_iterator_" .. tostring(dim) .. "d_t"]
  local rect_it_create =
    c["legion_rect_in_domain_iterator_create_" .. tostring(dim) .. "d"]
  local rect_it_destroy =
    c["legion_rect_in_domain_iterator_destroy_" .. tostring(dim) .. "d"]
  local rect_it_valid =
    c["legion_rect_in_domain_iterator_valid_" .. tostring(dim) .. "d"]
  local rect_it_step =
    c["legion_rect_in_domain_iterator_step_" .. tostring(dim) .. "d"]
  local rect_it_get =
    c["legion_rect_in_domain_iterator_get_rect_" .. tostring(dim) .. "d"]

  -- Create variables for the outer loop that iterates over rectangles in a domain
  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local rect = terralib.newsymbol(rect_type, "rect")
  local rect_it = terralib.newsymbol(rect_it_type, "rect_it")

  local base = terralib.newsymbol(c.coord_t, "base")
  local count = terralib.newsymbol(c.coord_t, "count")
  local start = terralib.newsymbol(c.coord_t, "start")
  local stop = terralib.newsymbol(c.coord_t, "stop")
  local final = terralib.newsymbol(c.coord_t, "final")

  local fields = index_type.fields
  local indices = terralib.newlist()
  if fields then
    indices:insertall(fields:map(function(field)
      return terralib.newsymbol(c.coord_t, tostring(field))
    end))
  else
    indices:insert(terralib.newsymbol(c.coord_t, "x"))
  end

  local symbol_setup = nil
  if ispace_type.dim == 0 then
    symbol_setup = quote
      var [symbol] = [symbol.type]{
        __ptr = ptr { __ptr = c.legion_ptr_t { value = [ indices[1] ] } }
      }
    end
  elseif ispace_type.dim == 1 then
    symbol_setup = quote
      var [symbol] = [symbol.type]{ __ptr = [ indices[1] ] }
    end
  else
    symbol_setup = quote
      var [symbol] = [symbol.type] { __ptr = [index_type.impl_type]{ [indices] } }
    end
  end

  local bounds_setup = quote
    var alignment = [vector_width]
    var [base] = [rect].lo.x[0]
    var [count] = [rect].hi.x[0] - [rect].lo.x[0] + 1
    var [start] = ([base] + alignment - 1) and not (alignment - 1)
    var [stop] = ([base] + [count]) and not (alignment - 1)
    var [final] = [base] + [count]
  end

  local i = indices[1]
  local body = quote
    var [i] = [base]
    if [count] >= [vector_width] then
      while [i] < [start] do
        [symbol_setup]
        do
          [orig_block_1]
        end
        [i] = [i] + 1
      end
      while [i] < [stop] do
        [symbol_setup]
        do
          [block]
        end
        [i] = [i] + [vector_width]
      end
    end
    while [i] < [final] do
      [symbol_setup]
      do
        [orig_block_2]
      end
      [i] = [i] + 1
    end
  end

  for i = 2, dim do
    local rect_i = i - 1 -- C is zero-based, Lua is one-based
    body = quote
      for [ indices[i] ] = [rect].lo.x[rect_i], [rect].hi.x[rect_i] + 1 do
        [body]
      end
    end
  end

  return quote
    [value.actions]
    var [domain] = c.legion_index_space_get_domain([cx.runtime], [is])
    var [rect_it] = [rect_it_create]([domain])
    while [rect_it_valid]([rect_it]) do
      var [rect] = [rect_it_get]([rect_it])
      [bounds_setup]
      [body]
      [rect_it_step]([rect_it])
    end
    [rect_it_destroy]([rect_it])
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
    [codegen_hooks.gen_update_mapping_tag(tag, false, cx.task)]
    var [must_epoch] = c.legion_must_epoch_launcher_create(0, [tag])
    c.legion_must_epoch_launcher_set_provenance([must_epoch], [get_provenance(node)])
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
  local loop_cx = cx:new_local_scope()
  local preamble = node.preamble:map(function(stat) return codegen.stat(cx, stat) end)
  local loop_vars = node.loop_vars:map(function(stat) return codegen.stat(cx, stat) end)
  local has_preamble = #preamble > 0

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
      local region_arg
      if arg:is(ast.typed.expr.Projection) then
        region_arg = arg.region
      else
        region_arg = arg
      end
      local partition_expr = util.get_base_indexed_node(region_arg)
      local partition_type = std.as_read(partition_expr.expr_type)
      partition = codegen.expr(cx, partition_expr):read(cx)

      -- Now run codegen the rest of the way to get the region.
      local region_expr = util.replace_base_indexed_node(
        region_arg,
        ast.typed.expr.Internal {
          value = values.value(
            node,
            expr.just(empty_quote, partition.value),
            partition_type),
          expr_type = partition_type,
          annotations = node.annotations,
          span = node.span,
        })

      if arg:is(ast.typed.expr.Projection) then
        region_expr = arg {
          region = region_expr,
        }
      end
      local region = codegen.expr(loop_cx, region_expr):read(loop_cx)
      args:insert(region)
    end
    args_partitions:insert(partition)
  end
  local conditions = node.call.conditions:map(
    function(condition)
      return codegen.expr_condition(cx, condition)
    end)
  local predicate = node.call.predicate and codegen.expr(cx, node.call.predicate):read(cx)

  local symbol_type = node.symbol:gettype()
  local symbol = node.symbol:getsymbol()

  local actions = quote
    [actions]
    [fn.actions];
    [data.zip(args, args_partitions, node.args_provably.invariant):map(
       function(pair)
         local arg, arg_partition, invariant = unpack(pair)

         -- Here we slice partition actions back in.
         local arg_actions = empty_quote
         if arg_partition then
           arg_actions = quote [arg_actions]; [arg_partition.actions] end
         end

         -- Normal invariant arg actions.
         if not has_preamble and invariant then
           arg_actions = quote [arg_actions]; [arg.actions] end
         end

         return arg_actions
       end)];
    [conditions:map(function(condition) return condition.actions end)]
    [predicate and predicate.actions]
  end

  local arg_types = terralib.newlist()
  for i, arg in ipairs(args) do
    arg_types:insert(std.as_read(node.call.args[i].expr_type))
  end

  local arg_values = terralib.newlist()
  local param_types = node.call.fn.expr_type.parameters
  for i, arg in ipairs(args) do
    local arg_value = arg.value
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
    if has_preamble or not invariant then
      task_args_setup:insert(arg.actions)
    end
  end
  expr_call_setup_task_args(
    loop_cx, fn.value, arg_values, arg_types, param_types,
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
  for condition, args_enabled in param_conditions:items() do
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
        cx, loop_cx, fn.value, node.call.args[i], arg_type, param_type, partition.value, node.symbol, launcher, true,
        args_setup, node.free_vars[i], loop_vars)
    end
  end

  local must_epoch_setup = empty_quote
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

  local symbol_setup
  if std.is_bounded_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]({ __ptr = [symbol_type.index_type]([point]) })
    end
  elseif std.is_index_type(symbol_type) then
    symbol_setup = quote
      var [symbol] = [symbol_type]([point])
    end
  else
    -- Otherwise symbol_type has to be some simple integral type.
    assert(symbol_type:isintegral())
    symbol_setup = quote
      var [symbol] = [int1d]([point])
    end
  end

  -- Setup predicate.
  local predicate_symbol = terralib.newsymbol(c.legion_predicate_t, "predicate")
  local predicate_value
  if predicate then
    predicate_value = `c.legion_predicate_create([cx.runtime], [cx.context], [predicate.value].__result)
  else
    predicate_value = `c.legion_predicate_true()
  end
  local predicate_setup = quote
    var [predicate_symbol] = [predicate_value]
  end

  local argument_map = terralib.newsymbol(c.legion_argument_map_t, "argument_map")
  local task_args_loop_setup, task_args_loop_cleanup
  if node.is_constant_time then
    local global_args = terralib.newsymbol(c.legion_task_argument_t, "global_args")
    -- It turns out the easiest way to do this is to literally run the loop
    -- for one iteration. Some of this work will be immediately thrown away,
    -- but we'll compute all the right things to make sure we get the regions
    -- set up correctly in the process.
    task_args_loop_setup = quote
      var [global_args]
      [global_args].args = nil
      [global_args].arglen = 0
      do
        var it = c.legion_domain_point_iterator_create([domain])
        if c.legion_domain_point_iterator_has_next(it) then
          var [point] = c.legion_domain_point_iterator_next(it)
          [symbol_setup]

          var [task_args]
          [task_args_setup]
          [args_setup]

          [global_args].args = c.malloc([task_args].arglen)
          std.assert([global_args].args ~= nil, "allocation failed in constant time launch args setup")
          c.memcpy([global_args].args, [task_args].args, [task_args].arglen)
          [global_args].arglen = [task_args].arglen

          c.legion_index_launcher_set_global_arg([launcher], [global_args])

          [task_args_cleanup]
          [cleanup(loop_cx)]
        end
        c.legion_domain_point_iterator_destroy(it)
      end
    end
    task_args_loop_cleanup = quote
      c.free([global_args].args)
      [cleanup(cx)]
    end
  else
    task_args_loop_setup = quote
      do
        var it = c.legion_domain_point_iterator_create([domain])
        var args_uninitialized = true
        while c.legion_domain_point_iterator_has_next(it) do
          var [point] = c.legion_domain_point_iterator_next(it)
          [symbol_setup]

          var [task_args]
          [task_args_setup]
          c.legion_argument_map_set_point(
            [argument_map], [point], [task_args], true)

          if args_uninitialized then
            [args_setup];
            args_uninitialized = false
          end

          [task_args_cleanup]
          [cleanup(loop_cx)]
        end
        c.legion_domain_point_iterator_destroy(it)
      end
    end
    task_args_loop_cleanup = as_quote(cleanup(cx))
  end

  local tag = terralib.newsymbol(c.legion_mapping_tag_id_t, "tag")
  local launcher_setup = quote
    [must_epoch_setup]
    var [argument_map] = c.legion_argument_map_create()
    var g_args : c.legion_task_argument_t
    g_args.args = nil
    g_args.arglen = 0
    [predicate_setup]
    var mapper = [fn.value:has_mapper_id() or 0]
    var [tag] = [fn.value:has_mapping_tag_id() or 0]
    [codegen_hooks.gen_update_mapping_tag(tag, fn.value:has_mapping_tag_id(), cx.task)]
    var [launcher] = c.legion_index_launcher_create(
      [fn.value:get_task_id()],
      [domain], g_args, [argument_map],
      [predicate_symbol], false, [mapper], [tag])
    c.legion_index_launcher_set_provenance([launcher], [get_provenance(node)])
    [task_args_loop_setup]
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
      value = values.value(node, expr.just(empty_quote, rh), future_type),
      expr_type = future_type,
      annotations = node.annotations,
      span = node.span,
    }

    local reduce
    if not std.is_future(rhs_type) then
      reduce = ast.typed.stat.Reduce {
        lhs = node.reduce_lhs,
        rhs = ast.typed.expr.FutureGetResult {
          value = rhs,
          expr_type = rhs_type,
          annotations = node.annotations,
          span = node.span,
        },
        op = node.reduce_op,
        metadata = false,
        annotations = node.annotations,
        span = node.span,
      }
    else
      assert(node.reduce_task) -- Set by optimize_futures

      reduce = ast.typed.stat.Assignment {
        lhs = node.reduce_lhs,
        rhs = ast.typed.expr.Call {
          fn = ast.typed.expr.Function {
            value = node.reduce_task,
            expr_type = node.reduce_task:get_type(),
            annotations = ast.default_annotations(),
            span = node.span,
          },
          args = terralib.newlist({
            node.reduce_lhs,
            rhs,
          }),
          conditions = terralib.newlist(),
          predicate = false,
          predicate_else_value = false,
          replicable = false,
          expr_type = std.as_read(node.reduce_lhs.expr_type),
          annotations = node.annotations,
          span = node.span,
        },
        metadata = false,
        annotations = node.annotations,
        span = node.span,
      }
    end

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
      [task_args_loop_cleanup]
    end
  else
    launcher_cleanup = quote
      c.legion_argument_map_destroy([argument_map])
      -- FIXME: we currently leak the task args in a must epoch launch
      -- [task_args_loop_cleanup]
    end
  end

  if predicate then
    launcher_cleanup = quote
      [launcher_cleanup]
      c.legion_predicate_destroy([predicate_symbol])
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

local function stat_index_fill_setup(cx, node, domain, actions)
  local fill = node.call
  local value = codegen.expr(cx, fill.value):read(cx)
  local value_type = std.as_read(fill.value.expr_type)
  local region = fill.dst.region
  local region_type = std.as_read(region.expr_type)
  local partition = region.value
  local partition_value = codegen.expr(cx, partition):read(cx)
  local partition_type = std.as_read(partition.expr_type)
  -- We need a complete codegen for the destination so we can access the metadata
  local _ = codegen.expr(cx, region)
  local parent_region =
    cx:region(cx:region(region_type).root_region_type).logical_region
  local projection_functor =
    make_partition_projection_functor(cx, region, node.symbol)

  return quote
    [actions]
    [value.actions]
    [partition_value.actions]
    [expr_fill_setup_region(
       cx, partition_value.value, partition_type, region_type, fill.dst.fields,
       value.value, value_type, true, domain, projection_functor)]
  end
end

function codegen.stat_index_launch_num(cx, node)
  local values = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)

  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local actions = quote
    [values:map(function(value) return value.actions end)]
    var [domain] = [loop_bounds_to_domain(cx, values:map(function(value) return value.value end))]
  end

  if node.call:is(ast.typed.expr.Call) then
    return stat_index_launch_setup(cx, node, domain, actions)
  elseif node.call:is(ast.typed.expr.Fill) then
    return stat_index_fill_setup(cx, node, domain, actions)
  else
    assert(false)
  end
end

function codegen.stat_index_launch_list(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  local domain = terralib.newsymbol(c.legion_domain_t, "domain")
  local actions = quote
    [value.actions]
    var [domain] = [loop_bounds_to_domain(cx, value.value, value_type)]
  end

  if node.call:is(ast.typed.expr.Call) then
    return stat_index_launch_setup(cx, node, domain, actions)
  elseif node.call:is(ast.typed.expr.Fill) then
    return stat_index_fill_setup(cx, node, domain, actions)
  else
    assert(false)
  end
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
    if node:is(ast.typed.expr.Partition) or
      node:is(ast.typed.expr.PartitionEqual) or
      node:is(ast.typed.expr.PartitionByField) or
      node:is(ast.typed.expr.PartitionByRestriction) or
      node:is(ast.typed.expr.Image) or
      node:is(ast.typed.expr.Preimage) or
      (node:is(ast.typed.expr.Binary) and std.is_partition(node.expr_type))
    then
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
        c.legion_index_partition_attach_name([cx.runtime], [rhs_value].impl.index_partition, [lhs.displayname], false)
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
      [cx:get_all_cleanup_items_for_return()]
      return
    end
  else
    local buffer = terralib.newsymbol(&opaque, "buffer")
    local data_ptr = terralib.newsymbol(&uint8, "data_ptr")

    local size_actions, size_value = std.compute_serialized_size(
      result_type, result)
    if not size_actions then
      size_actions = empty_quote
      size_value = 0
    end
    local ser_actions = std.serialize(
      result_type, result, buffer, `(&[data_ptr]))
    return quote
      [actions]
      [size_actions]
      var buffer_size = terralib.sizeof(result_type) + [size_value]
      var [buffer] = c.malloc(buffer_size)
      std.assert(buffer_size == 0 or [buffer] ~= nil, "malloc failed in return")
      var [data_ptr] = [&uint8]([buffer]) + terralib.sizeof(result_type)
      [ser_actions]
      std.assert(
        [data_ptr] - [&uint8]([buffer]) == buffer_size,
        "mismatch in data serialized in return")
      @[cx.result] = std.serialized_value {
        value = buffer,
        size = buffer_size,
      }
      [cx:get_cleanup_items()]
      return
      -- Task wrapper is responsible for calling free.
    end
  end
end

function codegen.stat_break(cx, node)
  assert(cx.break_label)
  return quote [cx:get_all_cleanup_items_for_break()]; goto [cx.break_label] end
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
    expr.just(empty_quote, rhs_expr.value),
    std.as_read(node.rhs.expr_type))

  actions:insert(lhs:write(cx, rhs, node.lhs.expr_type).actions)

  return as_quote(actions)
end

function codegen.stat_reduce(cx, node)
  local lhs = codegen.expr(cx, node.lhs)
  local rhs = codegen.expr(cx, node.rhs)
  local atomic = std.is_ref(node.lhs.expr_type) and node.metadata and
    -- The 'centers' being false means the value type requires
    -- this reduction to be handled with read-write or overrides
    -- the operator.  This type of reductions are always centered,
    -- enforced by the checker, and thus do not need atomics.
    node.metadata.centers and
    not node.metadata.centers:has(cx.loop_symbol)

  local actions = terralib.newlist()
  local rhs_expr = rhs:read(cx, node.rhs.expr_type)
  actions:insert(rhs_expr.actions)

  local lhs_actions = nil
  if cx:has_codegen_context("cuda") then
    local cuda_cx = cx:get_codegen_context("cuda")
    local function generator(rhs_terra)
      local rhs = values.value(
        node,
        expr.just(empty_quote, rhs_terra),
        std.as_read(node.rhs.expr_type))
      return lhs:reduce(cx, rhs, node.op, node.lhs.expr_type, atomic).actions
    end
    lhs_actions = gpuhelper.generate_region_reduction(cuda_cx, cx.loop_symbol,
        node, rhs_expr.value, node.lhs.expr_type, std.as_read(node.lhs.expr_type),
        generator)
  else
    rhs = values.value(
      node,
      expr.just(empty_quote, rhs_expr.value),
      std.as_read(node.rhs.expr_type))
    lhs_actions = lhs:reduce(cx, rhs, node.op, node.lhs.expr_type, atomic).actions
  end

  actions:insert(lhs_actions)
  return as_quote(actions)
end

function codegen.stat_expr(cx, node)
  local expr = codegen.expr(cx, node.expr):read(cx)

  -- If the value is stored in a variable, it will be cleaned up at
  -- the end of the variable's lifetime. Otherwise cleanup now.
  if not node.expr:is(ast.typed.expr.ID) then
    local cleanup = make_cleanup_item(cx, expr.value, node.expr.expr_type)
    return quote [expr.actions]; [cleanup] end
  else
    return as_quote(expr.actions)
  end
end

function codegen.stat_begin_trace(cx, node)
  local trace_id = codegen.expr(cx, node.trace_id):read(cx)
  return quote
    [trace_id.actions];
    [emit_debuginfo(node)];
    c.legion_runtime_begin_trace([cx.runtime], [cx.context], [trace_id.value], false)
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
  local roots_by_type = data.newmap()
  for _, region_type in ipairs(region_types) do
    assert(cx:has_region(region_type))
    local root_region_type = cx:region(region_type).root_region_type
    roots_by_type[root_region_type] = true
  end
  local roots = terralib.newlist()
  for region_type, _ in roots_by_type:items() do
    roots:insert(region_type)
  end
  return roots
end

local function find_region_roots_physical(cx, region_types)
  local roots = find_region_roots(cx, region_types)
  local result = terralib.newlist()
  for _, region_type in ipairs(roots) do
    if not cx.region_usage or cx.region_usage[region_type] then
      local physical_regions = cx:region(region_type).physical_regions
      local privilege_field_paths = cx:region(region_type).privilege_field_paths
      for _, field_paths in ipairs(privilege_field_paths) do
        for _, field_path in ipairs(field_paths) do
          result:insert(physical_regions[field_path])
        end
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
  return as_quote(actions)
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
  return as_quote(actions)
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

  local actions = quote
    [value.actions]
    [region_delete_fn]([cx.runtime], [cx.context], [value.value].impl)
    [ispace_delete_fn]([cx.runtime], [cx.context], [ispace_getter(value)])
  end

  if std.is_region(value_type) then
    actions = quote
      [actions]
      c.legion_field_space_destroy(
          [cx.runtime], [cx.context], [value.value].impl.field_space)
    end
  end

  return actions
end

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

  local f = terralib.newsymbol(c.legion_future_t)

  actions:insert(
    quote
      var [f] = [issue_fence]([cx.runtime], [cx.context])
    end)

  if blocking then
    actions:insert(
      quote
        c.legion_future_get_void_result([f])
      end)
  end

  actions:insert(
    quote
      c.legion_future_destroy([f])
    end)

  return quote
    [actions]
  end
end

function codegen.stat_parallelize_with(cx, node)
  return as_quote(codegen.block(cx, node.block))
end

local function generate_parallel_prefix_bounds_checks(cx, node, lhs_region, rhs_region)
  if not cx.bounds_checks then
    return empty_quote
  else
    local lhs_r = cx:region(lhs_region)
    local rhs_r = cx:region(rhs_region)
    local lhs_is = terralib.newsymbol(c.legion_index_space_t, "lhs_is")
    local rhs_is = terralib.newsymbol(c.legion_index_space_t, "rhs_is")
    local region_types = terralib.newlist({lhs_region, rhs_region})
    local regions = terralib.newlist({lhs_r, rhs_r})
    local ispaces = terralib.newlist({lhs_is, rhs_is})
    local dense_checks = empty_quote
    local bounds = data.zip(region_types, regions, ispaces):map(function(tuple)
      local region, r, is = unpack(tuple)
      local bounds_actions, domain, bounds = index_space_bounds(cx, is, region:ispace())
      dense_checks = quote
        [dense_checks];
        var [is] = [r.logical_region].impl.index_space
        [bounds_actions];
        std.assert_error(c.legion_domain_is_dense([domain]),
            [get_source_location(node) .. ": parallel prefix operator supports only dense regions"])
      end
      return bounds
    end)
    return quote
      do
        [dense_checks];
        std.assert_error([bounds[1]] == [bounds[2]],
            [get_source_location(node) ..
            ": the source and the target of a parallel prefix operator must have the same size"])
      end
    end
  end
end

local function generate_parallel_prefix_cpu(cx, node)
  -- Generate the following snippet of code:
  --
  --   var start_i = rhs.bounds.lo
  --   var end_i = rhs.bounds.hi
  --   var index, prev_index
  --   if dir > 0 then
  --    index = start_i
  --   else
  --    index = end_i
  --   end
  --   lhs[index] = rhs[index]
  --   prev_index = index
  --   index = index + dir
  --
  --   while start_i <= index and index <= end_i do
  --     lhs[index] = op(rhs[index], lhs[prev_index])
  --     prev_index = index
  --     index = index + dir
  --   end
  --

  assert(#node.lhs.fields == 1)
  assert(#node.rhs.fields == 1)
  local lhs_field = node.lhs.fields[1]
  local rhs_field = node.rhs.fields[1]
  local lhs_type = std.as_read(node.lhs.expr_type)
  local rhs_type = std.as_read(node.rhs.expr_type)
  assert(std.is_region(lhs_type))
  assert(std.is_region(rhs_type))
  local bounds = cx:ispace(rhs_type:ispace()).bounds

  local index_type = rhs_type:ispace().index_type
  local dir = codegen.expr(cx, node.dir):read(cx)
  local start_i = terralib.newsymbol(index_type, "start_i")
  local end_i = terralib.newsymbol(index_type, "end_i")

  local index = std.newsymbol(index_type, "index")
  local prev_index = std.newsymbol(index_type, "prev_index")
  local index_value = index:getsymbol()
  local prev_index_value = prev_index:getsymbol()
  local index_expr = ast.typed.expr.ID {
    value = index,
    expr_type = std.rawref(&index_type),
    span = node.dir.span,
    annotations = node.dir.annotations,
  }
  local prev_index_expr = ast.typed.expr.ID {
    value = prev_index,
    expr_type = std.rawref(&index_type),
    span = node.dir.span,
    annotations = node.dir.annotations,
  }

  local region_roots = terralib.newlist { node.lhs, node.lhs, node.rhs }
  local index_exprs = terralib.newlist { index_expr, prev_index_expr, index_expr }
  local index_access_exprs = data.zip(region_roots, index_exprs):map(function(pair)
    local region_root, index_expr = unpack(pair)
    local region = region_root.region
    local region_type = std.as_read(region.expr_type)
    local region_symbol
    if region:is(ast.typed.expr.ID) then
      region_symbol = region.value
    else
      region_symbol = terralib.newsymbol(region_type)
    end
    return ast.typed.expr.IndexAccess {
      value = region,
      index = index_expr,
      expr_type = std.ref(index_type(region_type:fspace(), region_symbol)),
      span = region.span,
      annotations = region.annotations,
    }
  end)
  local field_access_exprs = data.zip(region_roots, index_access_exprs):map(function(pair)
    local region_root, index_access_expr = unpack(pair)
    local field_path = region_root.fields[1]
    local field_expr = index_access_expr
    for i = 1, #field_path do
      local value_type = field_expr.expr_type
      assert(std.is_ref(value_type))
      field_expr = ast.typed.expr.FieldAccess {
        value = field_expr,
        field_name = field_path[i],
        expr_type = std.get_field(value_type, field_path[i]),
        span = region_root.region.span,
        annotations = region_root.region.annotations,
      }
    end
    return field_expr
  end)

  local lhs_value = codegen.expr(cx, field_access_exprs[1])
  local rhs_value = codegen.expr(cx, field_access_exprs[3])

  local op_expr = ast.typed.expr.Binary {
    op = node.op,
    lhs = field_access_exprs[3],
    rhs = field_access_exprs[2],
    expr_type = std.as_read(field_access_exprs[2].expr_type),
    span = node.span,
    annotations = node.annotations,
  }
  local op_value = codegen.expr(cx, op_expr)

  local lhs_write_init = lhs_value:write(cx, rhs_value)
  local lhs_write_loop = lhs_value:write(cx, op_value)

  local actions = quote
    do
      [generate_parallel_prefix_bounds_checks(cx, node, lhs_type, rhs_type)];
      [dir.actions];
      var [start_i], [end_i] = [bounds].lo, [bounds].hi
      var [index_value], [prev_index_value]
      if [index_type]([dir.value]) >= [index_type:zero()] then
        [index_value] = [start_i]
      else
        [index_value] = [end_i]
      end
      [lhs_write_init.actions];
      [prev_index_value] = [index_value]
      [index_value] = [index_value] + [dir.value]
      while [start_i] <= [index_value] and [index_value] <= [end_i] do
        [lhs_write_loop.actions];
        [prev_index_value] = [index_value]
        [index_value] = [index_value] + [dir.value]
      end
      [emit_debuginfo(node)]
    end
  end
  return actions
end

local function generate_parallel_prefix_gpu(cx, node)
  -- Generate the following snippet of code:
  --
  -- host side:
  --
  --   parallel_prefix<<rhs.bounds:size() / 2>>(lhs, rhs, rhs.bounds:size(), dir)
  --
  -- device side:
  --
  --   terra parallel_prefix(lhs, rhs, n, dir)
  --
  --     var tmp = __shared_memory()
  --     var t = tid()
  --     var lr = [int](dir >= 0)
  --     var oa = 2 * t + 1
  --     var ob = 2 * t + 2 * lr
  --
  --     tmp[2 * t] = rhs[2 * t]
  --     tmp[2 * t + 1] = rhs[2 * t + 1]
  --
  --     var d = n >> 1
  --     var offset = 1
  --     while d > 0 do
  --       __barrier()
  --       if t * dir < d * dir then
  --         var ai = offset * oa - lr
  --         var bi = offset * ob - lr
  --         tmp[bi] = op(tmp[ai], tmp[bi])
  --       end
  --       offset = offset << 1
  --       d = d >> 1
  --     end
  --     if t == 0 then tmp[(n - lr) % n] = identity end
  --     d = 1
  --     while d < n do
  --        offset = offset >> 1
  --       __barrier()
  --        if t * dir < d * dir then
  --          var ai = offset * oa - lr
  --          var bi = offset * ob - lr
  --          var x = tmp[ai]
  --          tmp[ai] = tmp[bi]
  --          tmp[bi] = op(t, tmp[bi])
  --        end
  --        d = d << 1
  --     end
  --     __barrier()
  --
  --     lhs[2 * t] = op(rhs[2 * t], tmp[2 * t])
  --     lhs[2 * t + 1] = op(rhs[2 * t + 1], tmp[2 * t + 1])
  --   end
  --

  assert(#node.lhs.fields == 1)
  assert(#node.rhs.fields == 1)
  local lhs_field = node.lhs.fields[1]
  local rhs_field = node.rhs.fields[1]
  local lhs_type = std.as_read(node.lhs.expr_type)
  local rhs_type = std.as_read(node.rhs.expr_type)
  assert(std.is_region(lhs_type))
  assert(std.is_region(rhs_type))
  local bounds = cx:ispace(rhs_type:ispace()).bounds

  local index_type = rhs_type:ispace().index_type
  local dir_value = codegen.expr(cx, node.dir):read(cx)

  local elem_type = std.get_field_path(lhs_type:fspace(), lhs_field)
  assert(std.type_eq(elem_type, std.get_field_path(rhs_type:fspace(), rhs_field)))

  local idx = std.newsymbol(index_type, "idx")
  local res = std.newsymbol(elem_type, "res")
  local idx_expr = ast.typed.expr.ID {
    value = idx,
    expr_type = std.rawref(&index_type),
    span = node.dir.span,
    annotations = node.dir.annotations,
  }
  local res_expr = ast.typed.expr.ID {
    value = res,
    expr_type = std.rawref(&elem_type),
    span = node.dir.span,
    annotations = node.dir.annotations,
  }

  local region_roots = terralib.newlist { node.lhs, node.rhs }
  local index_access_exprs = region_roots:map(function(region_root)
    local region = region_root.region
    local region_type = std.as_read(region.expr_type)
    local region_symbol
    if region:is(ast.typed.expr.ID) then
      region_symbol = region.value
    else
      region_symbol = terralib.newsymbol(region_type)
    end
    return ast.typed.expr.IndexAccess {
      value = region,
      index = idx_expr,
      expr_type = std.ref(index_type(region_type:fspace(), region_symbol)),
      span = region.span,
      annotations = region.annotations,
    }
  end)
  local field_access_exprs = data.zip(region_roots, index_access_exprs):map(function(pair)
    local region_root, index_access_expr = unpack(pair)
    local field_path = region_root.fields[1]
    local field_expr = index_access_expr
    for i = 1, #field_path do
      local value_type = field_expr.expr_type
      assert(std.is_ref(value_type))
      field_expr = ast.typed.expr.FieldAccess {
        value = field_expr,
        field_name = field_path[i],
        expr_type = std.get_field(value_type, field_path[i]),
        span = region_root.region.span,
        annotations = region_root.region.annotations,
      }
    end
    return field_expr
  end)

  local lhs_value = codegen.expr(cx, field_access_exprs[1])
  local rhs_value = codegen.expr(cx, field_access_exprs[2])
  local res_value = codegen.expr(cx, res_expr)

  local rhs = rhs_value:read(cx)
  local lhs_write = lhs_value:write(cx, res_value)
  local lhs_read = lhs_value:read(cx)

  local lhs_base_pointer = cx:region(lhs_type):base_pointer(lhs_field)
  local rhs_base_pointer = cx:region(rhs_type):base_pointer(rhs_field)

  local dir = terralib.newsymbol(std.as_read(node.dir.expr_type), "dir")
  local total = terralib.newsymbol(uint64, "total")

  local cuda_cx = gpuhelper.new_kernel_context(node)
  local launch_actions =
    gpuhelper.generate_parallel_prefix_op(cuda_cx, cx.task_meta:get_cuda_variant(), total,
                                           lhs_write, lhs_read, rhs, lhs_base_pointer, rhs_base_pointer,
                                           res:getsymbol(), idx:getsymbol(), dir, node.op, elem_type)
  local preamble, postamble
  local lhs_base_pointer_back = terralib.newsymbol(lhs_base_pointer.type)
  local rhs_base_pointer_back = terralib.newsymbol(rhs_base_pointer.type)
  if lhs_base_pointer == rhs_base_pointer then
    preamble = quote
      var [lhs_base_pointer_back] = [lhs_base_pointer]
      [lhs_base_pointer] = &[lhs_base_pointer][ [bounds].lo.__ptr ]
    end
    postamble = quote
      [lhs_base_pointer] = [lhs_base_pointer_back]
    end
  else
    preamble = quote
      var [lhs_base_pointer_back] = [lhs_base_pointer]
      var [rhs_base_pointer_back] = [rhs_base_pointer]
      [lhs_base_pointer] = &[lhs_base_pointer][ [bounds].lo.__ptr ]
      [rhs_base_pointer] = &[rhs_base_pointer][ [bounds].lo.__ptr ]
    end
    postamble = quote
      [lhs_base_pointer] = [lhs_base_pointer_back]
      [rhs_base_pointer] = [rhs_base_pointer_back]
    end
  end

  return quote
    do
      [generate_parallel_prefix_bounds_checks(cx, node, lhs_type, rhs_type)];
      [dir_value.actions]
      var [dir] = [dir_value.value]
      var [total] = [uint64]([bounds]:size())
      do
        [preamble]
        [launch_actions]
        [postamble]
      end
    end
  end
end

function codegen.stat_parallel_prefix(cx, node)
  if cx.variant:is_cuda() then
    return generate_parallel_prefix_gpu(cx, node)
  else
    return generate_parallel_prefix_cpu(cx, node)
  end
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

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    return codegen.stat_parallel_prefix(cx, node)

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
  for _, field in fields:keys() do
    local privilege = privileges[field]
    if not privilege or std.is_reduction_op(privilege) then
      remove:insert(field)
    end
  end
  for _, field in ipairs(remove) do
    fields[field] = nil
  end
  return fields
end

local unpack_param_helper = data.weak_memoize(function(param_type)
  local cx = context.new_global_scope()

  -- Inputs/outputs:
  local c_task = terralib.newsymbol(c.legion_task_t, "task")
  local params_map = terralib.newsymbol(&uint64, "params_map")
  local param_i = terralib.newsymbol(uint64, "param_i")
  local fixed_ptr = terralib.newsymbol(&opaque, "fixed_ptr")
  local data_ptr = terralib.newsymbol(&&uint8, "data_ptr")
  local future_count = terralib.newsymbol(int32, "future_count")
  local future_i = terralib.newsymbol(&int32, "future_i")

  -- Hack: this isn't used, but some APIs require nodes so we have to provide one.
  local dummy_node = ast.typed.expr.Internal {
    value = false,
    expr_type = opaque,
    annotations = ast.default_annotations(),
    span = ast.trivial_span(),
  }

  -- Generate code to unpack a future.
  local future = terralib.newsymbol(c.legion_future_t, "future")
  local future_type = std.future(param_type)
  local future_result = codegen.expr(
    cx,
    ast.typed.expr.FutureGetResult {
      value = ast.typed.expr.Internal {
        value = values.value(
          dummy_node,
          expr.just(empty_quote, `([future_type]{ __result = [future] })),
          future_type),
        expr_type = future_type,
        annotations = ast.default_annotations(),
        span = ast.trivial_span(),
      },
      expr_type = param_type,
      annotations = ast.default_annotations(),
      span = ast.trivial_span(),
  }):read(cx)

  -- Generate code to unpack a non-future.
  local deser_actions, deser_value = std.deserialize(
    param_type, fixed_ptr, data_ptr)

  local terra unpack_param([c_task], [params_map], [param_i], [fixed_ptr], [data_ptr],
                           [future_count], [future_i])
    if ([params_map][([param_i])/64] and (uint64(1) << ([param_i]%64))) == 0 then
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
end)

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
  for _, region_i in ipairs(std.fn_params_with_privileges_by_index(fn_type)) do
    local region = param_types[region_i]
    local field_paths, field_types =
      std.flatten_struct_fields(region:fspace())
    local field_label = terralib.newlabel(tostring(region) .. "_fields")
    param_field_id_labels[region_i] = field_label
    params_struct_type.entries:insert({
      field = field_label,
      type = c.legion_field_id_t[#field_types]
    })
  end
  task:set_field_id_param_labels(param_field_id_labels)
end

function codegen.top_task(cx, node)
  log_codegen:info("%s", "Starting codegen for task " .. tostring(node.name))

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
  local c_result = terralib.newsymbol(&std.serialized_value, "result")
  local c_params = terralib.newlist({
      c_task, c_regions, c_num_regions, c_context, c_runtime, c_result })

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

  local orderings = data.newmap()
  if variant:has_layout_constraints() then
    local region_types = data.newmap()
    data.filter(
      function(symbol)
        return std.is_region(symbol:gettype())
      end, task:get_param_symbols()):map(
      function(symbol)
        region_types[symbol:getname()] = symbol:gettype()
      end)
    variant:get_layout_constraints():map(function(constraint)
      local ordering = std.layout.make_index_ordering_from_constraint(constraint)

      local dimensions = constraint.dimensions
      local field_constraint_i = data.filteri(function(dimension)
        return dimension:is(ast.layout.Field)
      end, dimensions)
      if #field_constraint_i > 1 or #field_constraint_i == 0 then
        error("there must be one field constraint in the annotation")
      end
      local field_constraint = dimensions[field_constraint_i[1]]

      local region_type = region_types[field_constraint.region_name]
      if not orderings[region_type] then
        orderings[region_type] = data.newmap()
      end

      local absolute_field_paths =
        std.get_absolute_field_paths(region_type:fspace(), field_constraint.field_paths)

      if field_constraint_i[1] ~= 1 then
        absolute_field_paths:map(function(field_path)
          local field_type = std.get_field_path(region_type:fspace(), field_path)
          local expected_stride = terralib.sizeof(field_type)
          orderings[region_type][field_path] = data.newtuple(ordering, expected_stride)
        end)
      else
        local struct_size = data.reduce(function(a, b) return a + b end,
          absolute_field_paths:map(function(field_path)
            return terralib.sizeof(std.get_field_path(region_type:fspace(), field_path))
          end),
          0)
        absolute_field_paths:map(function(field_path)
          orderings[region_type][field_path] = data.newtuple(ordering, struct_size)
        end)
      end
    end)
  end

  local task_name = task.name:mkstring("", ".", "")
  local bounds_checks = needs_bounds_checks(task_name)
  if bounds_checks and std.config["bounds-checks-targets"] ~= ".*" then
    report.info(node, "bounds checks are enabled for task " .. task_name)
  end
  local cx = cx:new_task_scope(return_type,
                               task:get_constraints(),
                               orderings,
                               node.region_usage,
                               variant:get_config_options().leaf,
                               task, c_task, c_context, c_runtime, c_result, bounds_checks)

  -- FIXME: This code should be deduplicated with type_check, no
  -- reason to do it twice....
  for _, privilege_list in ipairs(task.privileges) do
    for _, privilege in ipairs(privilege_list) do
      local privilege_type = privilege.privilege
      local region = privilege.region
      local field_path = privilege.field_path
      assert(std.type_supports_privileges(region:gettype()))
      std.add_privilege(cx, privilege_type, region:gettype(), field_path)
    end
  end

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
      if c.legion_task_get_is_index_space(c_task) and c.legion_task_get_local_arglen(c_task) > 0 then
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

      local helper = unpack_param_helper(param_type)

      local actions = quote
        var [param_symbol] = [helper](
          [c_task], [params_map_symbol], [i-1], &args.[param:getlabel()], &[data_ptr],
          [future_count], &[future_i])
      end
      if std.is_ispace(param_type) and not cx:has_ispace(param_type) then
        local bounds_actions, domain, bounds =
          index_space_bounds(cx, `([param_symbol].impl), param_type)
        actions = quote [actions]; [bounds_actions] end
        cx:add_ispace_root(param_type, `([param_symbol].impl), domain, bounds)
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

  log_privileges:info('%s', 'task ' .. tostring(task.name))

  -- Unpack the region requirements.
  local physical_region_i = 0
  local fn_type = task:get_type()
  for _, region_i in ipairs(std.fn_param_regions_by_index(fn_type)) do
    local region_type = param_types[region_i]
    local index_type = region_type:ispace().index_type
    local r = params[region_i]:getsymbol()
    local is = terralib.newsymbol(c.legion_index_space_t, "is")

    local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
      std.find_task_privileges(region_type, task)

    log_privileges:info('  region ' .. tostring(region_i))
    for i, field_paths in ipairs(privilege_field_paths) do
      local privilege = privileges[i]
      log_privileges:info('    physical region ' .. tostring(physical_region_i) .. ' (privilege ' .. tostring(privilege) .. ')')
      for _, field_path in ipairs(field_paths) do
        log_privileges:info('      ' .. tostring(field_path))
      end
    end

    local privileges_by_field_path = std.group_task_privileges_by_field_path(
      privileges, privilege_field_paths)

    local field_paths, field_types =
      std.flatten_struct_fields(region_type:fspace())
    local field_id_array = `(args.[param_field_id_labels[region_i]])
    local field_ids_by_field_path = data.dict(
      data.zip(field_paths, data.mapi(function(field_i, _) return `([field_id_array][field_i - 1]) end, field_paths)))

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
        physical_regions_by_field_path[field_path] = physical_region
      end

      if not cx.variant:get_config_options().inner and
        (not cx.region_usage or cx.region_usage[region_type]) and
        flag ~= std.no_access_flag
      then
        local pr_actions, pr_base_pointers, pr_strides = unpack(data.zip(unpack(
          data.zip(field_paths, field_types):map(
            function(field)
              local field_path, field_type = unpack(field)
              local field_id = field_ids_by_field_path[field_path]
              return terralib.newlist({
                physical_region_get_base_pointer(cx, region_type, index_type, field_type, field_path, physical_region, field_id)})
        end))))

        physical_region_actions:insertall(pr_actions or {})
        base_pointers:insert(pr_base_pointers)

        for i, field_path in ipairs(field_paths) do
          if privileges_by_field_path[field_path] ~= "none" then
            base_pointers_by_field_path[field_path] = pr_base_pointers[i]
            strides_by_field_path[field_path] = pr_strides[i]
          end
        end
      end
    end

    local actions = empty_quote

    -- Hack: This enables Regent tasks to execute with constant time
    -- launches. The value of the region in the task arguments will be
    -- bogus, but we can recover it from the task's region requirement
    -- so we should be good.
    if #privilege_field_paths > 0 then
      local physical_region_index = physical_regions_index[1]
      actions = quote
        [actions]
        std.assert([physical_region_index] < c_num_regions, "too few physical regions in task setup")
        -- Important: local tasks do not have region requirements filled out, so we have to avoid pulling this in that case.
        if c.legion_task_get_is_index_space(c_task) then
          var req = c.legion_task_get_requirement(c_task, [physical_region_index])
          if c.legion_region_requirement_get_handle_type(req) == c.SINGULAR then
            var new_r = c.legion_region_requirement_get_region(req)
            if new_r.tree_id ~= 0 then
              [r].impl = new_r
            else
              std.assert(false, "corrupted tree_id in region argument unpack")
            end
          else
            std.assert(false, "non-singular region requirement in region argument unpack")
          end
        end
      end
    end

    if not cx.leaf then
      actions = quote
        [actions]
        var [is] = [r].impl.index_space
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
    if not (not cx.variant:get_config_options().inner and
              (not cx.region_usage or cx.region_usage[region_type]))
    then
      local actions = quote
        c.legion_runtime_unmap_all_regions([cx.runtime], [cx.context])
      end
      task_setup:insert(actions)
    end

    if not cx:has_ispace(region_type:ispace()) then
      local bounds_actions, domain, bounds =
        index_space_bounds(cx, `([r].impl.index_space), region_type:ispace())
      task_setup:insert(bounds_actions)
      cx:add_ispace_root(region_type:ispace(), is, domain, bounds)
    end

    -- If the region does *NOT* have privileges, look for parents that
    -- might be the root of privilege. In certain cases privileges
    -- might be split across multiple parents; we do not handle this
    -- case right now and such programs will hit runtime errors.
    local has_privileges = data.any(unpack(
      privileges:map(function(privilege) return privilege ~= "none" end)))

    local parent
    if not has_privileges then
      local parent_has_privileges = false
      for _, field_path in ipairs(field_paths) do
        for i = #field_path, 0, -1 do
          parent = std.search_any_privilege(cx, region_type, field_path:slice(1, i), {})
          if parent then break end
        end
        if parent then break end
      end
    end

    if parent and cx:has_region(parent) then
      cx:add_region_subregion(region_type, r, parent)
    else
      cx:add_region_root(region_type, r,
                         field_paths,
                         privilege_field_paths,
                         privileges_by_field_path,
                         data.dict(data.zip(field_paths, field_types)),
                         field_ids_by_field_path,
                         field_id_array,
                         data.dict(data.zip(field_paths, field_types:map(function(_) return false end))),
                         physical_regions_by_field_path,
                         base_pointers_by_field_path,
                         strides_by_field_path)
    end
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
    local field_id_array = `(args.[param_field_id_labels[list_i]])
    local field_ids_by_field_path = data.dict(
      data.zip(field_paths, data.mapi(function(field_i, _) return `([field_id_array][field_i - 1]) end, field_paths)))

    -- We never actually access physical instances for lists, so don't
    -- build any accessors here.

    cx:add_list_of_regions(list_type, list,
                           field_paths,
                           privilege_field_paths,
                           privileges_by_field_path,
                           data.dict(data.zip(field_paths, field_types)),
                           field_ids_by_field_path,
                           field_id_array,
                           data.dict(data.zip(field_paths, field_types:map(function(_) return false end))))
  end

  local preamble = quote
    [emit_debuginfo(node)]
    [task_setup]
  end

  local body
  if node.region_divergence then
    local region_divergence = terralib.newlist()
    local cases
    local diagnostic = empty_quote
    for _, rs in node.region_divergence:values() do
      local r1 = rs[1]
      if cx:has_region(r1) then
        local contained = true
        local rs_cases
        local rs_diagnostic = empty_quote

        local r1_fields = cx:region(r1).field_paths
        local valid_fields = data.set(r1_fields)
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
              for _, field in valid_fields:keys() do
                local r1_base = r1_bases[field]
                local r_base = r_base[field]
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
      local check_div = empty_quote
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

  local guard = empty_quote
  if return_type ~= terralib.types.unit then
    guard = quote
      std.assert_error(false, [get_source_location(node) .. ": missing return statement in task that is expected to return " .. tostring(return_type)])
    end
  end
  local terra proto([c_params])
    do
      [preamble]; -- Semicolon required. This is not an array access.
      [body]
    end
    [guard]
  end
  proto:setinlined(false)
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

    task:set_compile_thunk(
      function(variant)
        local cx = context.new_global_scope(variant)
        return codegen.top_task(cx, node)
    end)

    local cpu_variant = task:get_primary_variant()
    cpu_variant:set_ast(node)
    std.register_variant(cpu_variant)

    -- Mark the variant as OpenMP variant when at least one OpenMP loop exists
    if std.config["openmp"] ~= 0 and openmphelper.check_openmp_available() then
      ast.traverse_node_postorder(
        function(node)
          if node:is(ast.typed.stat) and
             node.annotations.openmp:is(ast.annotation.Demand)
          then
            cpu_variant:set_is_openmp(true)
          end
        end, node)
    end

    if node.annotations.cuda:is(ast.annotation.Demand) then
      local available, error_message = gpuhelper.check_gpu_available()
      if available then
        local cuda_variant = task:make_variant("cuda")
        cuda_variant:set_is_cuda(true)
        std.register_variant(cuda_variant)
        task:set_cuda_variant(cuda_variant)
      elseif gpuhelper.is_gpu_requested() and
             node.annotations.cuda:is(ast.annotation.Demand)
      then
        report.warn(node,
          "ignoring pragma at " .. node.span.source ..
          ":" .. tostring(node.span.start.line) .. " since " .. tostring(error_message))
      end
    end

    return task

  elseif node:is(ast.typed.top.Fspace) then
    return codegen.top_fspace(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function codegen.entry(node)
  local cx = context.new_global_scope()
  return codegen.top(cx, node)
end

return codegen
