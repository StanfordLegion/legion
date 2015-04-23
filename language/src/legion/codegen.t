-- Copyright 2015 Stanford University
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

-- Legion Code Generation

local ast = require("legion/ast")
local log = require("legion/log")
local std = require("legion/std")
local symbol_table = require("legion/symbol_table")

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

local codegen = {}

-- load Legion dynamic library
local c = std.c

local regions = {}
regions.__index = function(t, k) error("context: no such region " .. tostring(k), 2) end

local region = setmetatable({}, { __index = function(t, k) error("region has no field " .. tostring(k), 2) end})
region.__index = region

local context = {}
context.__index = context

function context:new_local_scope(div)
  if div == nil then
    div = self.divergence
  end
  return setmetatable({
    expected_return_type = self.expected_return_type,
    constraints = self.constraints,
    task = self.task,
    leaf = self.leaf,
    divergence = div,
    context = self.context,
    runtime = self.runtime,
    regions = self.regions:new_local_scope()
  }, context)
end

function context:new_task_scope(expected_return_type, constraints, leaf, task, ctx, runtime)
  assert(expected_return_type and task and ctx and runtime)
  return setmetatable({
    expected_return_type = expected_return_type,
    constraints = constraints,
    task = task,
    leaf = leaf,
    divergence = nil,
    context = ctx,
    runtime = runtime,
    regions = symbol_table.new_global_scope({})
  }, context)
end

function context.new_global_scope()
  return setmetatable({
  }, context)
end

function context:check_divergence(region_types)
  if not self.divergence then
    return false
  end
  for _, group in ipairs(self.divergence) do
    local contained = true
    for _, r in ipairs(region_types) do
      if not group[r] then
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
  return self.regions:lookup(region_type)
end

function context:add_region_root(region_type, logical_region, index_allocator,
                                 index_iterator, field_paths,
                                 privilege_field_paths, field_types,
                                 field_ids, physical_regions, accessors,
                                 base_pointers)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  self.regions:insert(
    region_type,
    setmetatable(
      {
        logical_region = logical_region,
        logical_partition = nil,
        index_allocator = index_allocator,
        index_iterator = index_iterator,
        field_paths = field_paths,
        privilege_field_paths = privilege_field_paths,
        field_types = field_types,
        field_ids = field_ids,
        physical_regions = physical_regions,
        accessors = accessors,
        base_pointers = base_pointers,
        root_region_type = region_type,
      }, region))
end

function context:add_region_subregion(region_type, logical_region,
                                      logical_partition, index_allocator,
                                      index_iterator, parent_region_type)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  if not self:region(parent_region_type) then
    error("parent to region " .. tostring(region_type) .. " not defined in this context", 2)
  end
  self.regions:insert(
    region_type,
    setmetatable(
      {
        logical_region = logical_region,
        logical_partition = logical_partition,
        index_allocator = index_allocator,
        index_iterator = index_iterator,
        field_paths = self:region(parent_region_type).field_paths,
        privilege_field_paths = self:region(parent_region_type).privilege_field_paths,
        field_types = self:region(parent_region_type).field_types,
        field_ids = self:region(parent_region_type).field_ids,
        physical_regions = self:region(parent_region_type).physical_regions,
        accessors = self:region(parent_region_type).accessors,
        base_pointers = self:region(parent_region_type).base_pointers,
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

function region:physical_region(field_path)
  local physical_region = self.physical_regions[field_path:hash()]
  assert(physical_region)
  return physical_region
end

function region:accessor(field_path)
  local accessor = self.accessors[field_path:hash()]
  assert(accessor)
  return accessor
end

function region:base_pointer(field_path)
  local base_pointer = self.base_pointers[field_path:hash()]
  assert(base_pointer)
  return base_pointer
end

local function accessor_generic_get_base_pointer(field_type)
  return terra(physical : c.legion_physical_region_t,
             accessor : c.legion_accessor_generic_t)

    var base_pointer : &opaque = nil
    var stride : c.size_t = terralib.sizeof(field_type)
    var ok = c.legion_accessor_generic_get_soa_parameters(
      accessor, &base_pointer, &stride)

    std.assert(ok, "failed to get base pointer")
    std.assert(base_pointer ~= nil, "base pointer is nil")
    std.assert(stride == terralib.sizeof(field_type),
               "stride does not match expected value")

    return [&field_type](base_pointer)
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

function expr.once_only(actions, value)
  if not actions or not value then
    error("expr requires actions and value", 2)
  end
  local value_name = terralib.newsymbol()
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

local function unpack_region(cx, region_expr, region_type, static_region_type)
  assert(not cx:has_region(region_type))

  local r = terralib.newsymbol(region_type, "r")
  local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
  local isa = false
  if not cx.leaf then
    isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
  end
  local it = false
  if cache_index_iterator then
    it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
  end
  local actions = quote
    [region_expr.actions]
    var [r] = [std.implicit_cast(
                 static_region_type, region_type, region_expr.value)]
    var [lr] = [r].impl
  end

  if not cx.leaf then
    actions = quote
      [actions]
      var [isa] = c.legion_index_allocator_create(
        [cx.runtime], [cx.context],
        [lr].index_space)
    end
  end

  if cache_index_iterator then
    actions = quote
      [actions]
      var [it] = c.legion_terra_cached_index_iterator_create([lr].index_space)
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

  cx:add_region_subregion(region_type, r, false, isa, it, parent_region_type)

  return expr.just(actions, r)
end

local value = {}
value.__index = value

function values.value(value_expr, value_type, field_path)
  if getmetatable(value_expr) ~= expr then
    error("value requires an expression", 2)
  end
  if not terralib.types.istype(value_type) then
    error("value requires a type", 2)
  end

  if field_path == nil then
    field_path = std.newtuple()
  elseif not std.is_tuple(field_path) then
    error("value requires a valid field_path", 2)
  end

  return setmetatable(
    {
      expr = value_expr,
      value_type = value_type,
      field_path = field_path,
    },
    value)
end

function value:new(value_expr, value_type, field_path)
  return values.value(value_expr, value_type, field_path)
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

function value:__get_field(cx, value_type, field_name)
  if value_type:ispointer() then
    return values.rawptr(self:read(cx), value_type, std.newtuple(field_name))
  elseif std.is_ptr(value_type) then
    return values.ref(self:read(cx, value_type), value_type, std.newtuple(field_name))
  elseif std.is_vptr(value_type) then
    return values.vref(self:read(cx, value_type), value_type, std.newtuple(field_name))
  else
    return self:new(
      self.expr, self.value_type, self.field_path .. std.newtuple(field_name))
  end
end

function value:get_field(cx, field_name, field_type)
  local value_type = self.value_type

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, value_type, field_name)
end

function value:get_index(cx, index, result_type)
  local value_expr = self:read(cx)
  local result = expr.just(quote [value_expr.actions]; [index.actions] end,
                           `([value_expr.value][ [index.value] ]))
  return values.value(result, result_type, std.newtuple())
end

function value:unpack(cx, value_type, field_name, field_type)
  local unpack_type = std.as_read(field_type)
  if std.is_region(unpack_type) and not cx:has_region(unpack_type) then
    local static_region_type = std.get_field(value_type, field_name)
    local region_expr = self:__get_field(cx, value_type, field_name):read(cx)
    region_expr = unpack_region(cx, region_expr, unpack_type, static_region_type)
    region_expr = expr.just(region_expr.actions, self.expr.value)
    return self:new(region_expr, self.value_type, self.field_path)
  elseif std.is_ptr(unpack_type) then
    local region_types = unpack_type:points_to_regions()

    do
      local has_all_regions = true
      for _, region_type in ipairs(region_types) do
        if not cx:has_region(region_type) then
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
    local static_region_types = static_ptr_type:points_to_regions()
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

    local region_expr = self:__get_field(cx, value_type, region_field_name):read(cx)
    region_expr = unpack_region(cx, region_expr, region_type, static_region_type)
    region_expr = expr.just(region_expr.actions, self.expr.value)
    return self:new(region_expr, self.value_type, self.field_path)
  else
    return self
  end
end

local ref = setmetatable({}, { __index = value })
ref.__index = ref

function values.ref(value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or
    not (std.is_ptr(value_type) or std.is_vptr(value_type)) then
    error("ref requires a legion ptr type", 2)
  end
  return setmetatable(values.value(value_expr, value_type, field_path), ref)
end

function ref:new(value_expr, value_type, field_path)
  return values.ref(value_expr, value_type, field_path)
end

function ref:__ref(cx, expr_type)
  local actions = self.expr.actions
  local value = self.expr.value

  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))
  local field_paths, field_types = std.flatten_struct_fields(value_type)
  local absolute_field_paths = field_paths:map(
    function(field_path) return self.field_path .. field_path end)

  local region_types = self.value_type:points_to_regions()
  local base_pointers_by_region = region_types:map(
    function(region_type)
      return absolute_field_paths:map(
        function(field_path)
          return cx:region(region_type):base_pointer(field_path)
        end)
    end)

  local base_pointers

  if cx.check_divergence(region_types) or #region_types == 1 then
    base_pointers = base_pointers_by_region[1]
  else
    base_pointers = std.zip(absolute_field_paths, field_types):map(
      function(field)
        local field_path, field_type = unpack(field)
        return terralib.newsymbol(&field_type, "base_pointer_" .. field_path:hash())
      end)

    local cases
    for i = #region_types, 1, -1 do
      local region_base_pointers = base_pointers_by_region[i]
      local case = std.zip(base_pointers, region_base_pointers):map(
        function(pair)
          local base_pointer, region_base_pointer = unpack(pair)
          return quote [base_pointer] = [region_base_pointer] end
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
      [cases]
    end
  end

  local values
  if not expr_type or std.as_read(expr_type) == value_type then
    values = base_pointers:map(
      function(base_pointer)
        return `(base_pointer[ [value].__ptr.value ])
      end)
  else
    assert(expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type))
    values = std.zip(base_pointers, field_types):map(
      function(field)
        local base_pointer, field_type = unpack(field)
        local vec = vector(field_type, std.as_read(expr_type).N)
        return `(@[&vec](&base_pointer[ [value].__ptr.value ]))
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
    var [value] : value_type
    [std.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if not aligned_instances and expr_type and
            (expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type)) then
           return quote
             [result] = terralib.attrload(&[field_value],
                                          { align = [ sizeof(field_type) ] })
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
    [std.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if not aligned_instances and expr_type and
            (expr_type:isvector() or std.is_vptr(expr_type) or std.is_sov(expr_type)) then
          return quote
            terralib.attrstore(&[field_value], [result],
                               { align = [ sizeof(field_type) ] })
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
  actions = quote
    [value_expr.actions];
    [actions];
    [std.zip(values, field_paths, field_types):map(
       function(pair)
         local field_value, field_path, field_type = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         if not aligned_instances and expr_type and (expr_type:isvector() or std.is_sov(expr_type)) then
           local field_value_load = quote
              terralib.attrload(&[field_value],
                                { align = [ sizeof(field_type) ] })
           end
           local sym = terralib.newsymbol()
           return quote
             var [sym] : expr_type =
               terralib.attrload(&[field_value],
                                 { align = [ sizeof(field_type) ] })
             terralib.attrstore(
               &[field_value],
               [std.quote_binary_op(fold_op, sym, result)],
               { align = [ sizeof(field_type) ] })
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

function ref:get_field(cx, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, value_type, field_name)
end

function ref:get_index(cx, index, result_type)
  local actions, value = self:__ref(cx)
  -- Arrays are never field-sliced, therefore, an array array access
  -- must be to a single field.
  assert(#value == 1)
  value = value[1]
  local result = expr.just(quote [actions]; [index.actions] end, `([value][ [index.value] ]))
  return values.rawref(result, &result_type, std.newtuple())
end

local vref = setmetatable({}, { __index = value })
vref.__index = vref

function values.vref(value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or not std.is_vptr(value_type) then
    error("vref requires a legion vptr type", 2)
  end
  return setmetatable(values.value(value_expr, value_type, field_path), vref)
end

function vref:new(value_expr, value_type, field_path)
  return values.vref(value_expr, value_type, field_path)
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

  local region_types = self.value_type:points_to_regions()
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
      var [ v ] : expr_type.type
    end
  end

  -- if the vptr points to a single region
  if cx.check_divergence(region_types) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    std.zip(base_pointers, field_paths):map(
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
    var [value] : expr_type
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

  if cx.check_divergence(region_types) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    std.zip(base_pointers, field_paths):map(
      function(pair)
        local base_pointer, field_path = unpack(pair)
        local result = value_expr.value
        for i = 1, vector_width do
          local field_value = `base_pointer[ [vref_value].__ptr.value[ [i - 1] ] ]
          for _, field_name in ipairs(field_path) do
            result = `([result].[field_name])
          end
          local assignment
          if value.value_type:isprimitive() then
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

  if cx.check_divergence(region_types) or #region_types == 1 then
    local base_pointers = base_pointers_by_region[1]

    std.zip(base_pointers, field_paths):map(
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
            local v = terralib.newsymbol()
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

function vref:get_field(cx, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, value_type, field_name)
end

local rawref = setmetatable({}, { __index = value })
rawref.__index = rawref

-- For pointer-typed rvalues, this entry point coverts the pointer
-- to an lvalue by dereferencing the pointer.
function values.rawptr(value_expr, value_type, field_path)
  if getmetatable(value_expr) ~= expr then
    error("rawref requires an expression", 2)
  end

  value_expr = expr.just(value_expr.actions, `(@[value_expr.value]))
  return values.rawref(value_expr, value_type, field_path)
end

-- This entry point is for lvalues which are already references
-- (e.g. for mutable variables on the stack). Conceptually
-- equivalent to a pointer rvalue which has been dereferenced. Note
-- that value_type is still the pointer type, not the reference
-- type.
function values.rawref(value_expr, value_type, field_path)
  if not terralib.types.istype(value_type) or not value_type:ispointer() then
    error("rawref requires a pointer type, got " .. tostring(value_type), 2)
  end
  return setmetatable(values.value(value_expr, value_type, field_path), rawref)
end

function rawref:new(value_expr, value_type, field_path)
  return values.rawref(value_expr, value_type, field_path)
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
  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
    [ref_expr.value] = [value_expr.value]
  end
  return expr.just(actions, quote end)
end

function rawref:reduce(cx, value, op)
  local ref_expr = self:__ref(cx)
  local value_expr = value:read(cx)

  local ref_type = self.value_type.type
  local value_type = std.as_read(value.value_type)

  local reduce = ast.typed.ExprBinary {
    op = op,
    lhs = ast.typed.ExprInternal {
      value = values.value(expr.just(quote end, ref_expr.value), ref_type),
      expr_type = ref_type,
    },
    rhs = ast.typed.ExprInternal {
      value = values.value(expr.just(quote end, value_expr.value), value_type),
      expr_type = value_type,
    },
    expr_type = ref_type,
    span = ast.trivial_span(),
  }

  local reduce_expr = codegen.expr(cx, reduce):read(cx, ref_type)

  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
    [reduce_expr.actions];
    [ref_expr.value] = [reduce_expr.value]
  end
  return expr.just(actions, quote end)
end

function rawref:get_field(cx, field_name, field_type, value_type)
  assert(value_type)
  value_type = std.as_read(value_type)

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, value_type, field_name)
end

function rawref:get_index(cx, index, result_type)
  local ref_expr = self:__ref(cx)
  local result = expr.just(
    quote [ref_expr.actions]; [index.actions] end,
    `([ref_expr.value][ [index.value] ]))
  return values.rawref(result, &result_type, std.newtuple())
end

function codegen.expr_internal(cx, node)
  return node.value
end

function codegen.expr_id(cx, node)
  if std.is_rawref(node.expr_type) then
    return values.rawref(expr.just(quote end, node.value), node.expr_type.pointer_type)
  else
    return values.value(expr.just(quote end, node.value), node.expr_type)
  end
end

function codegen.expr_constant(cx, node)
  local value = node.value
  local value_type = std.as_read(node.expr_type)
  return values.value(
    expr.just(quote end, `([terralib.constant(value_type, value)])),
    value_type)
end

function codegen.expr_function(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    expr.just(quote end, node.value),
    value_type)
end

function codegen.expr_field_access(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  if std.is_region(value_type) and
    value_type:has_default_partition() and
    node.field_name == "partition"
  then
    local value = codegen.expr(cx, node.value):read(cx)
    assert(cx:has_region(value_type))
    local lp = cx:region(value_type).logical_partition
    assert(lp)
    local partition_type = std.as_read(node.expr_type)
    return values.value(
      expr.once_only(value.actions, `([partition_type]({ impl = lp }))),
      node.expr_type)
  else
    local field_name = node.field_name
    local field_type = node.expr_type
    return codegen.expr(cx, node.value):get_field(cx, field_name, field_type, node.value.expr_type)
  end
end

function codegen.expr_index_access(cx, node)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  if std.is_partition(value_type) or std.is_cross_product(value_type) or
    (std.is_region(value_type) and value_type:has_default_partition())
  then
    local value = codegen.expr(cx, node.value):read(cx)
    local index = codegen.expr(cx, node.index):read(cx)

    local actions = quote [value.actions]; [index.actions] end

    if cx:has_region(expr_type) then
      local lr = cx:region(expr_type).logical_region
      if std.is_cross_product(value_type) and
        not cx:region(expr_type).logical_partition
      then
        local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
        actions = quote
          [actions]
          var ip = c.legion_terra_index_cross_product_get_subpartition_by_color(
            [cx.runtime], [cx.context],
            [value.value].product, [index.value])
          var [lp] = c.legion_logical_partition_create(
            [cx.runtime], [cx.context], [lr].impl, ip)
        end
        cx:region(expr_type).logical_partition = lp
      end
      return values.value(expr.just(actions, lr), expr_type)
    end

    local parent_region_type = value_type:parent_region()

    local partition = `([value.value].impl)
    if std.is_region(value_type) then
      partition = cx:region(value_type).logical_partition
    end

    local r = terralib.newsymbol(expr_type, "r")
    local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
    local isa = false
    if not cx.leaf then
      isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
    end
    local it = false
    if cache_index_iterator then
      it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
    end
    actions = quote
      [actions]
      var [lr] = c.legion_logical_partition_get_logical_subregion_by_color(
        [cx.runtime], [cx.context],
        [partition], [index.value])
      var [r] = [expr_type] { impl = [lr] }
    end

    if not cx.leaf then
      actions = quote
        [actions]
        var [isa] = c.legion_index_allocator_create(
          [cx.runtime], [cx.context],
          [lr].index_space)
      end
    end

    if cache_index_iterator then
      actions = quote
        [actions]
        var [it] = c.legion_terra_cached_index_iterator_create([lr].index_space)
      end
    end

    local lp = false
    if std.is_cross_product(value_type) then
      lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
      actions = quote
        [actions]
        var ip = c.legion_terra_index_cross_product_get_subpartition_by_color(
          [cx.runtime], [cx.context],
          [value.value].product, [index.value])
        var [lp] = c.legion_logical_partition_create(
          [cx.runtime], [cx.context], [lr], ip)
      end
    end

    cx:add_region_subregion(expr_type, r, lp, isa, it, parent_region_type)

    return values.value(expr.just(actions, r), expr_type)
  else
    local index = codegen.expr(cx, node.index):read(cx)
    return codegen.expr(cx, node.value):get_index(cx, index, expr_type)
  end
end

function codegen.expr_method_call(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local args = node.args:map(
    function(arg) return codegen.expr(cx, arg):read(cx) end)

  local actions = quote
    [value.actions];
    [args:map(function(arg) return arg.actions end)]
  end
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    expr.once_only(
      actions,
      `([value.value]:[node.method_name](
          [args:map(function(arg) return arg.value end)]))),
    expr_type)
end

function expr_call_setup_task_args(cx, task, args, arg_types, param_types,
                                   params_struct_type, params_map, task_args,
                                   task_args_setup)
  -- This all has to be done in 64-bit integers to avoid imprecision
  -- loss due to Lua's ONLY numeric type being double. Below we use
  -- LuaJIT's uint64_t cdata type as a replacement.

  -- Beware: LuaJIT does not expose bitwise operators at the Lua
  -- level. Below we use plus (instead of bitwise or) and
  -- exponentiation (instead of shift).
  local params_map_value = 0ULL
  for i, arg_type in ipairs(arg_types) do
    if std.is_future(arg_type) then
      params_map_value = params_map_value + (2ULL ^ (i-1))
    end
  end

  if params_map then
    task_args_setup:insert(quote
      [task_args].[params_map] = [params_map_value]
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
      task_args_setup:insert(quote [task_args].[c_field_name] = [arg] end)
    end
  end

  -- Prepare the region arguments to the task.

  -- Pass field IDs by-value to the task.
  local param_field_ids = task:get_field_id_params()
  do
    local param_field_id_i = 1
    for _, i in ipairs(std.fn_param_regions_by_index(task:gettype())) do
      local arg_type = arg_types[i]
      local param_type = param_types[i]

      local field_paths, _ = std.flatten_struct_fields(param_type.element_type)
      for _, field_path in pairs(field_paths) do
        local arg_field_id = cx:region(arg_type):field_id(field_path)
        local param_field_id = param_field_ids[param_field_id_i]
        param_field_id_i = param_field_id_i + 1
        task_args_setup:insert(
          quote [task_args].[param_field_id] = [arg_field_id] end)
      end
    end
  end
  return task_args_setup
end

function expr_call_setup_future_arg(cx, task, arg, arg_type, param_type,
                                    launcher, index, future_args_setup)
  local add_future = c.legion_task_launcher_add_future
  if index then
    add_future = c.legion_index_launcher_add_future
  end

  future_args_setup:insert(quote
    add_future(launcher, [arg].__result)
  end)

  return future_args_setup
end

function expr_call_setup_region_arg(cx, task, arg_type, param_type, launcher,
                                    index, region_args_setup)
  local privileges, privilege_field_paths, privilege_field_types =
    std.find_task_privileges(param_type, task:getprivileges())
  local privilege_modes = privileges:map(std.privilege_mode)
  local parent_region =
    cx:region(cx:region(arg_type).root_region_type).logical_region

  local add_field = c.legion_task_launcher_add_field
  if index then
    add_field = c.legion_index_launcher_add_field
  end

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local field_types = privilege_field_types[i]
    local privilege_mode = privilege_modes[i]

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

    local requirement = terralib.newsymbol("requirement")
    local requirement_args = terralib.newlist({
        launcher, `([cx:region(arg_type).logical_region].impl)})
    if index then
      requirement_args:insert(0)
    end
    if reduction_op then
      requirement_args:insert(reduction_op)
    else
      requirement_args:insert(privilege_mode)
    end
    requirement_args:insertall(
      {c.EXCLUSIVE, `([parent_region].impl), 0, false})

    region_args_setup:insert(
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
      end)
  end
end

function expr_call_setup_partition_arg(cx, task, arg_type, param_type,
                                       partition, launcher, index,
                                       region_args_setup)
  assert(index)
  local privileges, privilege_field_paths, privilege_field_types =
    std.find_task_privileges(param_type, task:getprivileges())
  local privilege_modes = privileges:map(std.privilege_mode)
  local parent_region =
    cx:region(cx:region(arg_type).root_region_type).logical_region

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local field_types = privilege_field_types[i]
    local privilege_mode = privilege_modes[i]

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

    local requirement = terralib.newsymbol("requirement")
    local requirement_args = terralib.newlist({
        launcher, `([partition].impl), 0 --[[ default projection ID ]]})
    if reduction_op then
      requirement_args:insert(reduction_op)
    else
      requirement_args:insert(privilege_mode)
    end
    requirement_args:insertall(
      {c.EXCLUSIVE, `([parent_region].impl), 0, false})

    region_args_setup:insert(
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
      end)
  end
end

function codegen.expr_call(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local args = node.args:map(
    function(arg) return codegen.expr(cx, arg):read(cx, arg.expr_type) end)

  local actions = quote
    [fn.actions];
    [args:map(function(arg) return arg.actions end)]
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
    local task_args = terralib.newsymbol(params_struct_type)
    local task_args_setup = terralib.newlist()
    expr_call_setup_task_args(
      cx, fn.value, arg_values, arg_types, param_types,
      params_struct_type, fn.value:get_params_map(),
      task_args, task_args_setup)

    local launcher = terralib.newsymbol("launcher")

    -- Pass futures.
    local future_args_setup = terralib.newlist()
    for i, arg_type in ipairs(arg_types) do
      if std.is_future(arg_type) then
        local arg_value = arg_values[i]
        local param_type = param_types[i]
        expr_call_setup_future_arg(
          cx, fn.value, arg_value, arg_type, param_type,
          launcher, false, future_args_setup)
      end
    end

    -- Pass regions through region requirements.
    local region_args_setup = terralib.newlist()
    for _, i in ipairs(std.fn_param_regions_by_index(fn.value:gettype())) do
      local arg_type = arg_types[i]
      local param_type = param_types[i]

      expr_call_setup_region_arg(
        cx, fn.value, arg_type, param_type, launcher, false, region_args_setup)
    end

    local future = terralib.newsymbol("future")
    local launcher_setup = quote
      var [task_args]
      [task_args_setup]
      var t_args : c.legion_task_argument_t
      t_args.args = [&opaque](&[task_args])
      t_args.arglen = terralib.sizeof(params_struct_type)
      var [launcher] = c.legion_task_launcher_create(
        [fn.value:gettaskid()], t_args,
        c.legion_predicate_true(), 0, 0)
      [future_args_setup]
      [region_args_setup]
      var [future] = c.legion_task_launcher_execute(
        [cx.runtime], [cx.context], [launcher])
    end
    local launcher_cleanup = quote
      c.legion_task_launcher_destroy(launcher)
    end

    local future_type = value_type
    if not std.is_future(future_type) then
      future_type = std.future(value_type)
    end

    actions = quote
      [actions]
      [launcher_setup]
      [launcher_cleanup]
    end
    local future_value = values.value(
      expr.once_only(actions, `([future_type]{ __result = [future] })),
      value_type)

    if std.is_future(value_type) then
      return future_value
    elseif value_type == terralib.types.unit then
      actions = quote
        [actions]
        c.legion_future_destroy(future)
      end

      return values.value(expr.just(actions, quote end), terralib.types.unit)
    else
      return codegen.expr(
        cx,
        ast.typed.ExprFutureGetResult {
          value = ast.typed.ExprInternal {
            value = future_value,
            expr_type = future_type,
          },
          expr_type = value_type,
          span = node.span,
        })
    end
  else
    return values.value(
      expr.once_only(actions, `([fn.value]([arg_values]))),
      value_type)
  end
end

function codegen.expr_cast(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local arg = codegen.expr(cx, node.arg):read(cx, node.arg.expr_type)

  local actions = quote
    [fn.actions]; [arg.actions]
  end
  local value_type = std.as_read(node.expr_type)
  return values.value(
    expr.once_only(actions, `([fn.value]([arg.value]))),
    value_type)
end

function codegen.expr_ctor_list_field(cx, node)
  return codegen.expr(cx, node.value):read(cx)
end

function codegen.expr_ctor_rec_field(cx, node)
  return  codegen.expr(cx, node.value):read(cx)
end

function codegen.expr_ctor_field(cx, node)
  if node:is(ast.typed.ExprCtorListField) then
    return codegen.expr_ctor_list_field(cx, node)
  elseif node:is(ast.typed.ExprCtorRecField) then
    return codegen.expr_ctor_rec_field(cx, node)
  else
  end
end

function codegen.expr_ctor(cx, node)
  local fields = node.fields:map(
    function(field) return codegen.expr_ctor_field(cx, field) end)

  local field_values = fields:map(function(field) return field.value end)
  local actions = fields:map(function(field) return field.actions end)
  local expr_type = std.as_read(node.expr_type)

  if node.named then
    local st = std.ctor(
      node.fields:map(
        function(field)
          local field_type = std.as_read(field.value.expr_type)
          return { field.name, field_type }
        end))

    return values.value(
      expr.once_only(actions, `([st]({ [field_values] }))),
      expr_type)
  else
    return values.value(
      expr.once_only(actions, `({ [field_values] })),
      expr_type)
  end
end

function codegen.expr_raw_context(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    expr.just(quote end, cx.context),
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

  local result = terralib.newsymbol("raw_fields")
  local actions = quote
    var [result] : expr_type
    [field_ids:map(
       function(pair)
         local i, field_id = unpack(pair)
         return quote [result][ [i] ] = [field_id] end
       end)]
  end

  return values.value(
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

  local result = terralib.newsymbol("raw_physical")
  local actions = quote
    var [result] : expr_type
    [physical_regions:map(
       function(pair)
         local i, physical_region = unpack(pair)
         return quote [result][ [i] ] = [physical_region] end
       end)]
  end

  return values.value(
    expr.just(actions, result),
    expr_type)
end

function codegen.expr_raw_runtime(cx, node)
  local value_type = std.as_read(node.expr_type)
  return values.value(
    expr.just(quote end, cx.runtime),
    value_type)
end

function codegen.expr_isnull(cx, node)
  local pointer = codegen.expr(cx, node.pointer):read(cx)
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    expr.once_only(
      pointer.actions,
      `([expr_type](c.legion_ptr_is_null([pointer.value].__ptr)))),
    expr_type)
end

function codegen.expr_new(cx, node)
  local pointer_type = node.pointer_type
  local region = codegen.expr(cx, node.region):read(cx)
  local region_type = std.as_read(node.region.expr_type)
  local isa = cx:region(region_type).index_allocator

  local expr_type = std.as_read(node.expr_type)

  return values.value(
    expr.once_only(
      region.actions,
      `([pointer_type]{ __ptr = c.legion_index_allocator_alloc([isa], 1) })),
    expr_type)
end

function codegen.expr_null(cx, node)
  local pointer_type = node.pointer_type
  local expr_type = std.as_read(node.expr_type)

  return values.value(
    expr.once_only(
      quote end,
      `([pointer_type]{ __ptr = c.legion_ptr_nil() })),
    expr_type)
end

function codegen.expr_dynamic_cast(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local expr_type = std.as_read(node.expr_type)

  local actions = value.actions
  local input = `([value.value].__ptr)
  local result
  local regions = expr_type:points_to_regions()
  if #regions == 1 then
    local region = regions[1]
    assert(cx:has_region(region))
    local lr = `([cx:region(region).logical_region].impl)
    result = `(
      [expr_type]({
          __ptr = (c.legion_ptr_safe_cast([cx.runtime], [cx.context], [input], [lr]))
      }))
  else
    result = terralib.newsymbol(expr_type)
    local cases = quote
      [result] = [expr_type]({ __ptr = c.legion_ptr_nil(), __index = 0 })
    end
    for i = #regions, 1, -1 do
      local region = regions[i]
      assert(cx:has_region(region))
      local lr = `([cx:region(region).logical_region].impl)
      cases = quote
        var temp = c.legion_ptr_safe_cast([cx.runtime], [cx.context], [input], [lr])
        if not c.legion_ptr_is_null(temp) then
          result = [expr_type]({
            __ptr = temp,
            __index = [i],
          })
        else
          [cases]
        end
      end
    end

    actions = quote [actions]; var [result]; [cases] end
  end

  return values.value(expr.once_only(actions, result), expr_type)
end

function codegen.expr_static_cast(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = value.actions
  local input = value.value
  local result
  if #(expr_type:points_to_regions()) == 1 then
    result = terralib.newsymbol(expr_type)
    local input_regions = value_type:points_to_regions()
    local result_last = node.parent_region_map[#input_regions]
    local cases
    if result_last then
      cases = quote
        [result] = [expr_type]({ __ptr = [input].__ptr })
      end
    else
      cases = quote
        [result] = [expr_type]({ __ptr = c.legion_ptr_nil() })
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
            [result] = [expr_type]({ __ptr = c.legion_ptr_nil() })
          else
            [cases]
          end
        end
      end
    end

    actions = quote [actions]; var [result]; [cases] end
  else
    result = terralib.newsymbol(expr_type)
    local input_regions = value_type:points_to_regions()
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

  return values.value(expr.once_only(actions, result), expr_type)
end

function codegen.expr_region(cx, node)
  local element_type = node.element_type
  local size = codegen.expr(cx, node.size):read(cx)
  local region_type = std.as_read(node.expr_type)

  local r = terralib.newsymbol(region_type, "r")
  local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
  local isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
  local it = false
  if cache_index_iterator then
    it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
  end
  local fsa = terralib.newsymbol(c.legion_field_allocator_t, "fsa")
  local pr = terralib.newsymbol(c.legion_physical_region_t, "pr")

  local field_paths, field_types = std.flatten_struct_fields(element_type)
  local field_id = 100
  local field_ids = field_paths:map(
    function(_)
      field_id = field_id + 1
      return field_id
    end)
  local physical_regions = field_paths:map(function(_) return pr end)
  local accessors = field_paths:map(
    function(field_path)
      return terralib.newsymbol(c.legion_accessor_generic_t, "accessor_" .. field_path:hash())
    end)
  local base_pointers = std.zip(field_paths, field_types):map(
    function(field)
      local field_path, field_type = unpack(field)
      return terralib.newsymbol(&field_type, "base_pointer_" .. field_path:hash())
    end)

  cx:add_region_root(region_type, r, isa, it,
                     field_paths,
                     terralib.newlist({field_paths}),
                     std.dict(std.zip(field_paths:map(std.hash), field_types)),
                     std.dict(std.zip(field_paths:map(std.hash), field_ids)),
                     std.dict(std.zip(field_paths:map(std.hash), physical_regions)),
                     std.dict(std.zip(field_paths:map(std.hash), accessors)),
                     std.dict(std.zip(field_paths:map(std.hash), base_pointers)))

  local actions = quote
    [size.actions]
    var capacity = [size.value]
    var is = c.legion_index_space_create([cx.runtime], [cx.context], capacity)
    var [isa] = c.legion_index_allocator_create([cx.runtime], [cx.context],  is)
    var fs = c.legion_field_space_create([cx.runtime], [cx.context])
    var [fsa] = c.legion_field_allocator_create([cx.runtime], [cx.context],  fs);
    [std.zip(field_types, field_ids):map(
       function(field)
         local field_type, field_id = unpack(field)
         return `(c.legion_field_allocator_allocate_field(
                    [fsa], terralib.sizeof([field_type]), [field_id]))
       end)]
    var [lr] = c.legion_logical_region_create([cx.runtime], [cx.context], is, fs)
    var il = c.legion_inline_launcher_create_logical_region(
      [lr], c.READ_WRITE, c.EXCLUSIVE, [lr], 0, false, 0, 0);
    [field_ids:map(
       function(field_id)
         return `(c.legion_inline_launcher_add_field(il, [field_id], true))
       end)]
    var [pr] = c.legion_inline_launcher_execute([cx.runtime], [cx.context], il)
    c.legion_inline_launcher_destroy(il)
    c.legion_physical_region_wait_until_valid([pr])
    [std.zip(field_ids, field_types, accessors, base_pointers):map(
       function(field)
         local field_id, field_type, accessor, base_pointer = unpack(field)
         return quote
           var [accessor] = c.legion_physical_region_get_field_accessor_generic([pr], [field_id])
           var [base_pointer] = [accessor_generic_get_base_pointer(field_type)]([pr], [accessor])
         end
       end)]
    var [r] = [region_type]{ impl = [lr] }
  end

  if cache_index_iterator then
    actions = quote
      [actions]
      var [it] = c.legion_terra_cached_index_iterator_create([lr].index_space)
    end
  end

  return values.value(expr.just(actions, r), region_type)
end

function codegen.expr_partition(cx, node)
  local region_expr = codegen.expr(cx, node.region):read(cx)
  local coloring_expr = codegen.expr(cx, node.coloring):read(cx)
  local partition_type = std.as_read(node.expr_type)

  local ip = terralib.newsymbol(c.legion_index_partition_t, "ip")
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local actions = quote
    [region_expr.actions];
    [coloring_expr.actions];
    var [ip] = c.legion_index_partition_create_coloring(
      [cx.runtime], [cx.context],
      [region_expr.value].impl.index_space,
      [coloring_expr.value],
      [node.disjointness == std.disjoint],
      -1)
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], [region_expr.value].impl, [ip])
  end

  return values.value(
    expr.once_only(actions, `(partition_type { impl = [lp] })),
    partition_type)
end

function codegen.expr_cross_product(cx, node)
  local lhs = codegen.expr(cx, node.lhs):read(cx)
  local rhs = codegen.expr(cx, node.rhs):read(cx)
  local expr_type = std.as_read(node.expr_type)

  local product = terralib.newsymbol(
    c.legion_terra_index_cross_product_t, "cross_product")
  local lr = cx:region(expr_type:parent_region()).logical_region
  local lp = terralib.newsymbol(c.legion_logical_partition_t, "lp")
  local actions = quote
    [lhs.actions]
    [rhs.actions]
    var [product] = c.legion_terra_index_cross_product_create(
      [cx.runtime], [cx.context],
      [lhs.value].impl.index_partition,
      [rhs.value].impl.index_partition)
    var ip = c.legion_terra_index_cross_product_get_partition([product])
    var [lp] = c.legion_logical_partition_create(
      [cx.runtime], [cx.context], lr.impl, ip)
  end

  return values.value(
    expr.once_only(actions, `(expr_type { impl = [lp], product = [product] })),
    expr_type)
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

    local name = "__unary_" .. tostring(rhs_type) .. "_" .. tostring(op)
    local rhs_symbol = terralib.newsymbol(rhs_type, "rhs")
    local task = std.newtask(name)
    local node = ast.typed.StatTask {
      name = name,
      params = terralib.newlist({
          ast.typed.StatTaskParam {
            symbol = rhs_symbol,
            param_type = rhs_type,
            span = ast.trivial_span(),
          },
      }),
      return_type = expr_type,
      privileges = terralib.newlist(),
      constraints = terralib.newlist(),
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.StatReturn {
              value = ast.typed.ExprUnary {
                op = op,
                rhs = ast.typed.ExprID {
                  value = rhs_symbol,
                  expr_type = rhs_type,
                  span = ast.trivial_span(),
                },
                expr_type = expr_type,
                span = ast.trivial_span(),
              },
              span = ast.trivial_span(),
            },
        }),
        span = ast.trivial_span(),
      },
      config_options = ast.typed.StatTaskConfigOptions {
        leaf = true,
        inner = false,
        idempotent = true,
      },
      region_divergence = false,
      prototype = task,
      span = ast.trivial_span(),
    }
    task:settype(
      terralib.types.functype(
        node.params:map(function(param) return param.param_type end),
        node.return_type,
        false))
    task:setprivileges(node.privileges)
    task:set_param_constraints(node.constraints)
    task:set_constraints({})
    task:set_region_universe({})
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

    local name = ("__binary_" .. tostring(lhs_type) .. "_" ..
                    tostring(rhs_type) .. "_" .. tostring(op))
    local lhs_symbol = terralib.newsymbol(lhs_type, "lhs")
    local rhs_symbol = terralib.newsymbol(rhs_type, "rhs")
    local task = std.newtask(name)
    local node = ast.typed.StatTask {
      name = name,
      params = terralib.newlist({
         ast.typed.StatTaskParam {
            symbol = lhs_symbol,
            param_type = lhs_type,
            span = ast.trivial_span(),
         },
         ast.typed.StatTaskParam {
            symbol = rhs_symbol,
            param_type = rhs_type,
            span = ast.trivial_span(),
         },
      }),
      return_type = expr_type,
      privileges = terralib.newlist(),
      constraints = terralib.newlist(),
      body = ast.typed.Block {
        stats = terralib.newlist({
            ast.typed.StatReturn {
              value = ast.typed.ExprBinary {
                op = op,
                lhs = ast.typed.ExprID {
                  value = lhs_symbol,
                  expr_type = lhs_type,
                  span = ast.trivial_span(),
                },
                rhs = ast.typed.ExprID {
                  value = rhs_symbol,
                  expr_type = rhs_type,
                  span = ast.trivial_span(),
                },
                expr_type = expr_type,
                span = ast.trivial_span(),
              },
              span = ast.trivial_span(),
            },
        }),
        span = ast.trivial_span(),
      },
      config_options = ast.typed.StatTaskConfigOptions {
        leaf = true,
        inner = false,
        idempotent = true,
      },
      region_divergence = false,
      prototype = task,
      span = ast.trivial_span(),
    }
    task:settype(
      terralib.types.functype(
        node.params:map(function(param) return param.param_type end),
        node.return_type,
        false))
    task:setprivileges(node.privileges)
    task:set_param_constraints(node.constraints)
    task:set_constraints({})
    task:set_region_universe({})
    return codegen.entry(node)
  end)

function codegen.expr_unary(cx, node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    local rhs_type = std.as_read(node.rhs.expr_type)
    local task = lift_unary_op_to_futures(node.op, rhs_type, expr_type)

    local call = ast.typed.ExprCall {
      fn = ast.typed.ExprFunction {
        value = task,
        expr_type = task:gettype(),
        span = node.span,
      },
      fn_unspecialized = false,
      args = terralib.newlist({node.rhs}),
      expr_type = expr_type,
      span = node.span,
    }
    return codegen.expr(cx, call)
  else
    local rhs = codegen.expr(cx, node.rhs):read(cx, expr_type)
    return values.value(
      expr.once_only(rhs.actions, std.quote_unary_op(node.op, rhs.value)),
      expr_type)
  end
end

function codegen.expr_binary(cx, node)
  local expr_type = std.as_read(node.expr_type)
  if std.is_future(expr_type) then
    local lhs_type = std.as_read(node.lhs.expr_type)
    local rhs_type = std.as_read(node.rhs.expr_type)
    local task = lift_binary_op_to_futures(
      node.op, lhs_type, rhs_type, expr_type)

    local call = ast.typed.ExprCall {
      fn = ast.typed.ExprFunction {
        value = task,
        expr_type = task:gettype(),
      span = node.span,
      },
      fn_unspecialized = false,
      args = terralib.newlist({node.lhs, node.rhs}),
      expr_type = expr_type,
      span = node.span,
    }
    return codegen.expr(cx, call)
  else
    local lhs = codegen.expr(cx, node.lhs):read(cx, node.lhs.expr_type)
    local rhs = codegen.expr(cx, node.rhs):read(cx, node.rhs.expr_type)

    local actions = quote [lhs.actions]; [rhs.actions] end
    local expr_type = std.as_read(node.expr_type)
    return values.value(
      expr.once_only(actions, std.quote_binary_op(node.op, lhs.value, rhs.value)),
      expr_type)
  end
end

function codegen.expr_deref(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  if value_type:ispointer() then
    return values.rawptr(value, value_type)
  elseif std.is_ptr(value_type) then
    return values.ref(value, value_type)
  else
    assert(false)
  end
end

function codegen.expr_future(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote [value.actions] end

  local result_type = std.type_size_bucket_type(value_type)
  if result_type == terralib.types.unit then
    assert(false)
  elseif result_type == c.legion_task_result_t then
    local result = terralib.newsymbol(c.legion_future_t, "result")
    local actions = quote
      [actions]
      var buffer = [value.value]
      var [result] = c.legion_future_from_buffer(
        [cx.runtime], [&opaque](&buffer), terralib.sizeof(value_type))
    end

    return values.value(
      expr.once_only(actions, `([expr_type]{ __result = [result] })),
      expr_type)
  else
    local result_type_name = std.type_size_bucket_name(result_type)
    local future_from_fn = c["legion_future_from" .. result_type_name]
    local result = terralib.newsymbol(c.legion_future_t, "result")
    local actions = quote
      [actions]
      var buffer = [value.value]
      var [result] = [future_from_fn]([cx.runtime], @[&result_type](&buffer))
    end
    return values.value(
      expr.once_only(actions, `([expr_type]{ __result = [result] })),
      expr_type)
  end
end

function codegen.expr_future_get_result(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local expr_type = std.as_read(node.expr_type)

  local actions = quote [value.actions] end

  local result_type = std.type_size_bucket_type(expr_type)
  if result_type == terralib.types.unit then
    assert(false)
  elseif result_type == c.legion_task_result_t then
    local result_value = terralib.newsymbol(expr_type, "result_value")
    local expr_type_alignment = std.min(terralib.sizeof(expr_type), 8)
    local actions = quote
      [actions]
      var result = c.legion_future_get_result([value.value].__result)
        -- Force unaligned access because malloc does not provide
        -- blocks aligned for all purposes (e.g. SSE vectors).
      var [result_value] = terralib.attrload(
        [&expr_type](result.value),
        { align = [expr_type_alignment] })
      c.legion_task_result_destroy(result)
    end
    return values.value(
      expr.just(actions, result_value),
      expr_type)
  else
    local result_type_name = std.type_size_bucket_name(result_type)
    local get_result_fn = c["legion_future_get_result" .. result_type_name]
    local result_value = terralib.newsymbol(expr_type, "result_value")
    local actions = quote
      [actions]
      var result = [get_result_fn]([value.value].__result)
      var [result_value] = @[&expr_type](&result)
    end
    return values.value(
      expr.just(actions, result_value),
      expr_type)
  end
end

function codegen.expr(cx, node)
  if node:is(ast.typed.ExprInternal) then
    return codegen.expr_internal(cx, node)

  elseif node:is(ast.typed.ExprID) then
    return codegen.expr_id(cx, node)

  elseif node:is(ast.typed.ExprConstant) then
    return codegen.expr_constant(cx, node)

  elseif node:is(ast.typed.ExprFunction) then
    return codegen.expr_function(cx, node)

  elseif node:is(ast.typed.ExprFieldAccess) then
    return codegen.expr_field_access(cx, node)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return codegen.expr_index_access(cx, node)

  elseif node:is(ast.typed.ExprMethodCall) then
    return codegen.expr_method_call(cx, node)

  elseif node:is(ast.typed.ExprCall) then
    return codegen.expr_call(cx, node)

  elseif node:is(ast.typed.ExprCast) then
    return codegen.expr_cast(cx, node)

  elseif node:is(ast.typed.ExprCtor) then
    return codegen.expr_ctor(cx, node)

  elseif node:is(ast.typed.ExprRawContext) then
    return codegen.expr_raw_context(cx, node)

  elseif node:is(ast.typed.ExprRawFields) then
    return codegen.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return codegen.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return codegen.expr_raw_runtime(cx, node)

  elseif node:is(ast.typed.ExprIsnull) then
    return codegen.expr_isnull(cx, node)

  elseif node:is(ast.typed.ExprNew) then
    return codegen.expr_new(cx, node)

  elseif node:is(ast.typed.ExprNull) then
    return codegen.expr_null(cx, node)

  elseif node:is(ast.typed.ExprDynamicCast) then
    return codegen.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.ExprStaticCast) then
    return codegen.expr_static_cast(cx, node)

  elseif node:is(ast.typed.ExprRegion) then
    return codegen.expr_region(cx, node)

  elseif node:is(ast.typed.ExprPartition) then
    return codegen.expr_partition(cx, node)

  elseif node:is(ast.typed.ExprCrossProduct) then
    return codegen.expr_cross_product(cx, node)

  elseif node:is(ast.typed.ExprUnary) then
    return codegen.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return codegen.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return codegen.expr_deref(cx, node)

  elseif node:is(ast.typed.ExprFuture) then
    return codegen.expr_future(cx, node)

  elseif node:is(ast.typed.ExprFutureGetResult) then
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

function codegen.stat_if(cx, node)
  local clauses = terralib.newlist()

  -- Insert first clause in chain.
  local cond = codegen.expr(cx, node.cond):read(cx)
  local then_cx = cx:new_local_scope()
  local then_block = codegen.block(then_cx, node.then_block)
  clauses:insert({cond, then_block})

  -- Add rest of clauses.
  for _, elseif_block in ipairs(node.elseif_blocks) do
    local cond = codegen.expr(cx, elseif_block.cond):read(cx)
    local elseif_cx = cx:new_local_scope()
    local block = codegen.block(elseif_cx, elseif_block.block)
    clauses:insert({cond, block})
  end
  local else_cx = cx:new_local_scope()
  local else_block = codegen.block(else_cx, node.else_block)

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
  local body_cx = cx:new_local_scope()
  local block = codegen.block(body_cx, node.block)
  return quote
    while [quote [cond.actions] in [cond.value] end] do
      [block]
    end
  end
end

function codegen.stat_for_num(cx, node)
  local symbol = node.symbol
  local cx = cx:new_local_scope()
  local bounds = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)
  local cx = cx:new_local_scope()
  local block = codegen.block(cx, node.block)

  local v1, v2, v3 = unpack(bounds)
  if #bounds == 2 then
    return quote
      [v1.actions]; [v2.actions]
      for [symbol] = [v1.value], [v2.value] do
        [block]
      end
    end
  else
    return quote
      [v1.actions]; [v2.actions]; [v3.actions]
      for [symbol] = [v1.value], [v2.value], [v3.value] do
        [block]
      end
    end
  end
end

function codegen.stat_for_list(cx, node)
  local symbol = node.symbol
  local cx = cx:new_local_scope()
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local cx = cx:new_local_scope()
  local block = codegen.block(cx, node.block)

  assert(cx:has_region(value_type))
  local lr = cx:region(value_type).logical_region
  local it = cx:region(value_type).index_iterator

  local actions = quote
    [value.actions]
  end
  local cleanup_actions = quote end

  local iterator_has_next
  local iterator_next_span
  if cache_index_iterator then
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
      var is = [lr].impl.index_space
      var [it] = c.legion_index_iterator_create(is)
    end
    cleanup_actions = quote
      c.legion_index_iterator_destroy([it])
    end
  end

  return quote
    do
      [actions]
      while iterator_has_next([it]) do
        var count : c.size_t = 0
        var base = iterator_next_span([it], &count, -1).value
        for i = 0, count do
          var [symbol] = [symbol.type]{
            __ptr = c.legion_ptr_t {
              value = base + i
            }
          }
          do
            [block]
          end
        end
      end
      [cleanup_actions]
    end
  end
end

function codegen.stat_for_list_vectorized(cx, node)
  local symbol = node.symbol
  local cx = cx:new_local_scope()
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local cx = cx:new_local_scope()
  local block = codegen.block(cx, node.block)
  local orig_block = codegen.block(cx, node.orig_block)
  local vector_width = node.vector_width

  assert(cx:has_region(value_type))
  local lr = cx:region(value_type).logical_region
  local it = cx:region(value_type).index_iterator

  local actions = quote
    [value.actions]
  end
  local cleanup_actions = quote end

  local iterator_has_next
  local iterator_next_span
  if cache_index_iterator then
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
      var is = [lr].impl.index_space
      var [it] = c.legion_index_iterator_create(is)
    end
    cleanup_actions = quote
      c.legion_index_iterator_destroy([it])
    end
  end

  return quote
    do
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
            var [symbol] = [symbol.type]{ __ptr = c.legion_ptr_t { value = i }}
            do
              [orig_block]
            end
            i = i + 1
          end
          while i < stop do
            var [symbol] = [symbol.type]{ __ptr = c.legion_ptr_t { value = i }}
            do
              [block]
            end
            i = i + [vector_width]
          end
        end
        while i < final do
          var [symbol] = [symbol.type]{ __ptr = c.legion_ptr_t { value = i }}
          do
            [orig_block]
          end
          i = i + 1
        end
      end
      [cleanup_actions]
    end
  end
end

function codegen.stat_repeat(cx, node)
  local cx = cx:new_local_scope()
  local block = codegen.block(cx, node.block)
  local until_cond = codegen.expr(cx, node.until_cond):read(cx)
  return quote
    repeat
      [block]
    until [quote [until_cond.actions] in [until_cond.value] end]
  end
end

function codegen.stat_block(cx, node)
  local cx = cx:new_local_scope()
  return quote
    do
      [codegen.block(cx, node.block)]
    end
  end
end

function codegen.stat_index_launch(cx, node)
  local symbol = node.symbol
  local cx = cx:new_local_scope()
  local domain = codegen.expr_list(cx, node.domain):map(function(value) return value:read(cx) end)

  local fn = codegen.expr(cx, node.call.fn):read(cx)
  assert(std.is_task(fn.value))
  local args = terralib.newlist()
  local args_partitions = {}
  for i, arg in ipairs(node.call.args) do
    if not node.args_provably.variant[i] then
      args:insert(codegen.expr(cx, arg):read(cx))
    else
      -- Run codegen halfway to get the partition.
      local partition = codegen.expr(cx, arg.value):read(cx)
      args_partitions[i] = partition

      -- Now run codegen the rest of the way to get the region.
      local partition_type = std.as_read(arg.value.expr_type)
      local region = codegen.expr(
        cx,
        ast.typed.ExprIndexAccess {
          value = ast.typed.ExprInternal {
            value = values.value(
              expr.just(quote end, partition.value),
              partition_type),
            expr_type = partition_type,
          },
          index = arg.index,
          expr_type = arg.expr_type,
          span = node.span,
        }):read(cx)
      args:insert(region)
    end
  end

  local actions = quote
    [domain[1].actions];
    [domain[2].actions];
    -- ignore domain[3] because we know it is a constant
    [fn.actions];
    [std.zip(args, node.args_provably.invariant):map(
       function(pair)
         local arg, invariant = unpack(pair)
         if invariant then
           return arg.actions
         else
           return quote end
         end
       end)]
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

  local value_type = fn.value:gettype().returntype

  local params_struct_type = fn.value:get_params_struct()
  local task_args = terralib.newsymbol(params_struct_type)
  local task_args_setup = terralib.newlist()
  for i, arg in ipairs(args) do
    local invariant = node.args_provably.invariant[i]
    if not invariant then
      task_args_setup:insert(arg.actions)
    end
  end
  expr_call_setup_task_args(
    cx, fn.value, arg_values, arg_types, param_types,
    params_struct_type, fn.value:get_params_map(),
    task_args, task_args_setup)

  local launcher = terralib.newsymbol("launcher")

  -- Pass futures.
  local future_args_setup = terralib.newlist()
  for i, arg_type in ipairs(arg_types) do
    if std.is_future(arg_type) then
      local arg_value = arg_values[i]
      local param_type = param_types[i]
      expr_call_setup_future_arg(
        cx, fn.value, arg_value, arg_type, param_type,
        launcher, true, future_args_setup)
    end
  end

  -- Pass regions through region requirements.
  local region_args_setup = terralib.newlist()
  for _, i in ipairs(std.fn_param_regions_by_index(fn.value:gettype())) do
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    if not node.args_provably.variant[i] then
      expr_call_setup_region_arg(
        cx, fn.value, arg_type, param_type, launcher, true, region_args_setup)
    else
      local partition = args_partitions[i]
      assert(partition)
      expr_call_setup_partition_arg(
        cx, fn.value, arg_type, param_type, partition.value, launcher, true,
        region_args_setup)
    end
  end

  local argument_map = terralib.newsymbol("argument_map")
  local launcher_setup = quote
    var [argument_map] = c.legion_argument_map_create()
    for [node.symbol] = [domain[1].value], [domain[2].value] do
      var [task_args]
      [task_args_setup]
      var t_args : c.legion_task_argument_t
      t_args.args = [&opaque](&[task_args])
      t_args.arglen = terralib.sizeof(params_struct_type)
      c.legion_argument_map_set_point(
        [argument_map],
        c.legion_domain_point_from_point_1d(
          c.legion_point_1d_t { x = arrayof(int32, [node.symbol]) }),
        t_args, true)
    end
    var g_args : c.legion_task_argument_t
    g_args.args = nil
    g_args.arglen = 0
    var [launcher] = c.legion_index_launcher_create(
      [fn.value:gettaskid()],
      c.legion_domain_from_rect_1d(
        c.legion_rect_1d_t {
          lo = c.legion_point_1d_t { x = arrayof(int32, [domain[1].value]) },
          hi = c.legion_point_1d_t { x = arrayof(int32, [domain[2].value] - 1) },
        }),
      g_args, [argument_map],
      c.legion_predicate_true(), false, 0, 0)
    [future_args_setup]
    [region_args_setup]
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

  local future = terralib.newsymbol("future")
  local launcher_execute = quote
    var [future] = execute_fn(execute_args)
  end

  if node.reduce_lhs then
    local rhs_type = std.as_read(node.call.expr_type)
    local future_type = rhs_type
    if not std.is_future(rhs_type) then
      future_type = std.future(rhs_type)
    end

    local rh = terralib.newsymbol(future_type)
    local rhs = ast.typed.ExprInternal {
      value = values.value(expr.just(quote end, rh), future_type),
      expr_type = future_type,
    }

    if not std.is_future(rhs_type) then
      rhs = ast.typed.ExprFutureGetResult {
        value = rhs,
        expr_type = rhs_type,
        span = node.span,
      }
    end

    local reduce = ast.typed.StatReduce {
      op = node.reduce_op,
      lhs = terralib.newlist({node.reduce_lhs}),
      rhs = terralib.newlist({rhs}),
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

  local launcher_cleanup = quote
    c.legion_argument_map_destroy([argument_map])
    destroy_future_fn([future])
    c.legion_index_launcher_destroy([launcher])
  end

  actions = quote
    [actions];
    [launcher_setup];
    [launcher_execute];
    [launcher_cleanup]
  end
  return actions
end

function codegen.stat_var(cx, node)
  local lhs = node.symbols
  local types = node.types
  local rhs = terralib.newlist()
  for i, value in pairs(node.values) do
    local rh = codegen.expr(cx, value)
    rhs:insert(rh:read(cx, value.expr_type))
  end

  local rhs_values = terralib.newlist()
  for i, rh in ipairs(rhs) do
    local rhs_type = std.as_read(node.values[i].expr_type)
    local lhs_type = types[i]
    if lhs_type then
      rhs_values:insert(std.implicit_cast(rhs_type, lhs_type, rh.value))
    else
      rhs_values:insert(rh.value)
    end
  end
  local actions = rhs:map(function(rh) return rh.actions end)

  if #rhs > 0 then
    local decls = terralib.newlist()
    for i, lh in ipairs(lhs) do
      if node.values[i]:is(ast.typed.ExprRegion) then
        actions = quote
          [actions]
          c.legion_logical_region_attach_name([cx.runtime], [ rhs_values[i] ].impl, [lh.displayname])
        end
      end
      decls:insert(quote var [lh] : types[i] = [ rhs_values[i] ] end)
    end
    return quote [actions]; [decls] end
  else
    local decls = terralib.newlist()
    for i, lh in ipairs(lhs) do
      decls:insert(quote var [lh] : types[i] end)
    end
    return quote [decls] end
  end
end

function codegen.stat_var_unpack(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)

  local lhs = node.symbols
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
  local return_type = cx.expected_return_type
  local result_type = std.type_size_bucket_type(return_type)

  local result = terralib.newsymbol("result")
  local actions = quote
    [value.actions]
    var [result] = [std.implicit_cast(value_type, return_type, value.value)]
  end

  if result_type == c.legion_task_result_t then
    return quote
      [actions]
      return c.legion_task_result_create(
        [&opaque](&[result]),
        terralib.sizeof([return_type]))
    end
  else
    return quote
      [actions]
      return @[&result_type](&[result])
    end
  end
end

function codegen.stat_break(cx, node)
  return quote break end
end

function codegen.stat_assignment(cx, node)
  local actions = terralib.newlist()
  local lhs = codegen.expr_list(cx, node.lhs)
  local rhs = codegen.expr_list(cx, node.rhs)
  rhs = std.zip(rhs, node.lhs, node.rhs):map(
    function(pair)
      local rh_value, lh_node, rh_node = unpack(pair)
      local rh_expr = rh_value:read(cx, rh_node.expr_type)
      actions:insert(rh_expr.actions)
      return values.value(
        expr.just(quote end, rh_expr.value),
        std.as_read(rh_node.expr_type))
    end)

  actions:insertall(
    std.zip(lhs, rhs, node.lhs):map(
      function(pair)
        local lh, rh, lh_node = unpack(pair)
        return lh:write(cx, rh, lh_node.expr_type).actions
      end))

  return quote [actions] end
end

function codegen.stat_reduce(cx, node)
  local actions = terralib.newlist()
  local lhs = codegen.expr_list(cx, node.lhs)
  local rhs = codegen.expr_list(cx, node.rhs)
  rhs = std.zip(rhs, node.rhs):map(
    function(pair)
      local rh_value, rh_node = unpack(pair)
      local rh_expr = rh_value:read(cx, rh_node.expr_type)
      actions:insert(rh_expr.actions)
      return values.value(
        expr.just(quote end, rh_expr.value),
        std.as_read(rh_node.expr_type))
    end)

  actions:insertall(
    std.zip(lhs, rhs, node.lhs):map(
      function(pair)
        local lh, rh, lh_node = unpack(pair)
        return lh:reduce(cx, rh, node.op, lh_node.expr_type).actions
      end))

  return quote [actions] end
end

function codegen.stat_expr(cx, node)
  local expr = codegen.expr(cx, node.expr):read(cx)
  return quote [expr.actions] end
end

function find_region_roots(cx, region_types)
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

function find_region_roots_physical(cx, region_types)
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
  local roots = find_region_roots_physical(cx, node.region_types)
  local actions = terralib.newlist()
  for _, pr in ipairs(roots) do
    actions:insert(
      `(c.legion_runtime_unmap_region([cx.runtime], [cx.context], [pr])))
  end
  return quote [actions] end
end

function codegen.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return codegen.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return codegen.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return codegen.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return codegen.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatForListVectorized) then
    return codegen.stat_for_list_vectorized(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return codegen.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return codegen.stat_block(cx, node)

  elseif node:is(ast.typed.StatIndexLaunch) then
    return codegen.stat_index_launch(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return codegen.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return codegen.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return codegen.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return codegen.stat_break(cx, node)

  elseif node:is(ast.typed.StatAssignment) then
    return codegen.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return codegen.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return codegen.stat_expr(cx, node)

  elseif node:is(ast.typed.StatMapRegions) then
    return codegen.stat_map_regions(cx, node)

  elseif node:is(ast.typed.StatUnmapRegions) then
    return codegen.stat_unmap_regions(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function get_params_map_type(params)
  if #params == 0 then
    return false
  elseif #params <= 64 then
    return uint64
  else
    assert(false)
  end
end

function codegen.stat_task(cx, node)
  local task = node.prototype
  std.register_task(task)

  task:set_config_options(node.config_options)

  local params_struct_type = terralib.types.newstruct()
  params_struct_type.entries = terralib.newlist()
  task:set_params_struct(params_struct_type)

  -- The param map tracks which parameters are stored in the task
  -- arguments versus futures. The space in the params struct will be
  -- reserved either way, but this tells us where to find a valid copy
  -- of the data. Currently this field has a fixed size to keep the
  -- code here sane, though conceptually it's just a bit vector.
  local params_map_type = get_params_map_type(node.params)
  local params_map = false
  if params_map_type then
    params_map = terralib.newsymbol(params_map_type, "__map")
    params_struct_type.entries:insert(
      { field = params_map, type = params_map_type })
  end
  task:set_params_map_type(params_map_type)
  task:set_params_map(params_map)

  -- Normal arguments are straight out of the param types.
  params_struct_type.entries:insertall(node.params:map(
    function(param)
      return { field = param.symbol.displayname, type = param.param_type }
    end))

  -- Regions require some special handling here. Specifically, field
  -- IDs are going to be passed around dynamically, so we need to
  -- reserve some extra slots in the params struct here for those
  -- field IDs.
  local param_regions = std.fn_param_regions(task:gettype())
  local param_field_ids = terralib.newlist()
  for _, region in ipairs(param_regions) do
    local field_paths, field_types =
      std.flatten_struct_fields(region.element_type)
    local field_ids = field_paths:map(
      function(field_path)
        return terralib.newsymbol("field_" .. field_path:hash())
      end)
    param_field_ids:insertall(field_ids)
    params_struct_type.entries:insertall(
      std.zip(field_ids, field_types):map(
        function(field)
          local field_id, field_type = unpack(field)
          return { field = field_id, type = c.legion_field_id_t }
        end))
  end
  task:set_field_id_params(param_field_ids)

  local params = node.params:map(
    function(param) return param.symbol end)
  local param_types = task:gettype().parameters
  local return_type = node.return_type

  local c_task = terralib.newsymbol(c.legion_task_t, "task")
  local c_regions = terralib.newsymbol(&c.legion_physical_region_t, "regions")
  local c_num_regions = terralib.newsymbol(uint32, "num_regions")
  local c_context = terralib.newsymbol(c.legion_context_t, "context")
  local c_runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
  local c_params = terralib.newlist({
      c_task, c_regions, c_num_regions, c_context, c_runtime })

  local cx = cx:new_task_scope(return_type, task:get_constraints(), task:get_config_options().leaf, c_task, c_context, c_runtime)

  -- Unpack the by-value parameters to the task.
  local task_args_setup = terralib.newlist()
  local args = terralib.newsymbol(&params_struct_type, "args")
  if #(task:get_params_struct():getentries()) > 0 then
    task_args_setup:insert(quote
      var [args]
      if c.legion_task_get_is_index_space(c_task) then
        var arglen = c.legion_task_get_local_arglen(c_task)
        if arglen ~= terralib.sizeof(params_struct_type) then c.abort() end
        args = [&params_struct_type](c.legion_task_get_local_args(c_task))
      else
        var arglen = c.legion_task_get_arglen(c_task)
        if arglen ~= terralib.sizeof(params_struct_type) then c.abort() end
        args = [&params_struct_type](c.legion_task_get_args(c_task))
      end
    end)
    task_args_setup:insert(quote
      var [params_map] = args.[params_map]
    end)

    local future_count = terralib.newsymbol(int32, "future_count")
    local future_i = terralib.newsymbol(int32, "future_i")
    task_args_setup:insert(quote
      var [future_count] = c.legion_task_get_futures_size([c_task])
      var [future_i] = 0
    end)
    for i, param in ipairs(params) do
      local param_type = node.params[i].param_type
      local param_type_alignment = std.min(terralib.sizeof(param_type), 8)

      local future = terralib.newsymbol("future")
      local future_type = std.future(param_type)
      local future_result = codegen.expr(
        cx,
        ast.typed.ExprFutureGetResult {
          value = ast.typed.ExprInternal {
            value = values.value(
              expr.just(quote end, `([future_type]{ __result = [future] })),
              future_type),
            expr_type = future_type,
          },
          expr_type = param_type,
          span = node.span,
      }):read(cx)

      task_args_setup:insert(quote
        var [param] : param_type
        if ([params_map] and [2ULL ^ (i-1)]) == 0 then
          -- Force unaligned access because malloc does not provide
          -- blocks aligned for all purposes (e.g. SSE vectors).
          [param] = terralib.attrload(
            (&args.[param.displayname]),
            { align = [param_type_alignment] })
        else
          std.assert([future_i] < [future_count], "missing future in task param")
          var [future] = c.legion_task_get_future([c_task], [future_i])
          [future_result.actions]
          [param] = [future_result.value]
          [future_i] = [future_i] + 1
        end
      end)
    end
    task_args_setup:insert(quote
      std.assert([future_i] == [future_count], "extra futures left over in task params")
    end)
  end

  -- Prepare any region parameters to the task.

  -- Unpack field IDs passed by-value to the task.
  local param_field_ids = task:get_field_id_params()
  for _, param in ipairs(param_field_ids) do
    task_args_setup:insert(quote
      var [param] = args.[param]
    end)
  end

  -- Unpack the region requirements.
  local region_args_setup = terralib.newlist()
  do
    local physical_region_i = 0
    local param_field_id_i = 1
    for _, region_i in ipairs(std.fn_param_regions_by_index(task:gettype())) do
      local region_type = param_types[region_i]
      local r = params[region_i]
      local isa = false
      if not cx.leaf then
        isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
      end
      local it = false
      if cache_index_iterator then
        it = terralib.newsymbol(c.legion_terra_cached_index_iterator_t, "it")
      end

      local privileges, privilege_field_paths, privilege_field_types =
        std.find_task_privileges(region_type, task:getprivileges())

      local privileges_by_field_path = std.group_task_privileges_by_field_path(
        privileges, privilege_field_paths)

      local field_paths, field_types =
        std.flatten_struct_fields(region_type.element_type)
      local field_ids_by_field_path = {}
      for _, field_path in ipairs(field_paths) do
        field_ids_by_field_path[field_path:hash()] = param_field_ids[param_field_id_i]
        param_field_id_i = param_field_id_i + 1
      end

      local physical_regions = terralib.newlist()
      local physical_regions_by_field_path = {}
      local physical_regions_index = terralib.newlist()
      local accessors = terralib.newlist()
      local accessors_by_field_path = {}
      local base_pointers = terralib.newlist()
      local base_pointers_by_field_path = {}
      for i, field_paths in ipairs(privilege_field_paths) do
        local field_types = privilege_field_types[i]
        local physical_region = terralib.newsymbol(
          c.legion_physical_region_t,
          "pr_" .. tostring(physical_region_i))

        physical_regions:insert(physical_region)
        physical_regions_index:insert(physical_region_i)
        physical_region_i = physical_region_i + 1

        local physical_region_accessors = field_paths:map(
          function(field_path)
            return terralib.newsymbol(
              c.legion_accessor_generic_t,
              "accessor_" .. field_path:hash())
          end)
        accessors:insert(physical_region_accessors)

        local physical_region_base_pointers = std.zip(field_paths, field_types):map(
          function(field)
            local field_path, field_type = unpack(field)
            return terralib.newsymbol(
              &field_type,
              "base_pointer_" .. field_path:hash())
          end)
        base_pointers:insert(physical_region_base_pointers)

        for i, field_path in ipairs(field_paths) do
          physical_regions_by_field_path[field_path:hash()] = physical_region
          if privileges_by_field_path[field_path:hash()] ~= "none" then
            accessors_by_field_path[field_path:hash()] = physical_region_accessors[i]
            base_pointers_by_field_path[field_path:hash()] = physical_region_base_pointers[i]
          end
        end
      end

      local actions = quote end

      if not cx.leaf then
        actions = quote
          [actions]
          var is = [r].impl.index_space
          var [isa] = c.legion_index_allocator_create([cx.runtime], [cx.context],  is)
        end
      end


      if cache_index_iterator then
        actions = quote
          [actions]
          var [it] = c.legion_terra_cached_index_iterator_create([r].impl.index_space)
        end
      end

      region_args_setup:insert(actions)

      for i, field_paths in ipairs(privilege_field_paths) do
        local field_types = privilege_field_types[i]
        local privilege = privileges[i]
        local physical_region = physical_regions[i]
        local physical_region_index = physical_regions_index[i]

        local get_accessor
        if std.is_reduction_op(privilege) then
          get_accessor = c.legion_physical_region_get_accessor_generic
        else
          get_accessor = c.legion_physical_region_get_field_accessor_generic
        end
        assert(get_accessor)

        region_args_setup:insert(quote
          var [physical_region] = [c_regions][ [physical_region_index] ]
        end)

        for j, field_path in ipairs(field_paths) do
          local field_type = field_types[j]
          local accessor = accessors[i][j]
          local base_pointer = base_pointers[i][j]
          local field_id = field_ids_by_field_path[field_path:hash()]
          assert(accessor and base_pointer and field_id)

          local accessor_args = terralib.newlist({physical_region})
          if not std.is_reduction_op(privilege) then
            accessor_args:insert(field_id)
          end

          if privileges_by_field_path[field_path:hash()] ~= "none" then
            region_args_setup:insert(quote
              var [accessor] = [get_accessor]([accessor_args])
              var [base_pointer] = [accessor_generic_get_base_pointer(field_type)](
                [physical_region], [accessor])
            end)
          end
        end
      end

      cx:add_region_root(region_type, r, isa, it,
                         field_paths,
                         privilege_field_paths,
                         std.dict(std.zip(field_paths:map(std.hash), field_types)),
                         field_ids_by_field_path,
                         physical_regions_by_field_path,
                         accessors_by_field_path,
                         base_pointers_by_field_path)
    end
  end

  local preamble = quote [task_args_setup]; [region_args_setup] end

  local body
  local has_divergence = false
  if node.region_divergence then
    for _, rs in pairs(node.region_divergence) do
      local rs_diverges = #rs > 1
      if rs_diverges then
        for _, r in ipairs(rs) do
          if not cx:has_region(r) then
            rs_diverges = false
            break
          end
        end
      end
      if rs_diverges then
        has_divergence = true
        break
      end
    end
  end
  if has_divergence then
    local region_divergence = terralib.newlist()
    local cases
    for _, rs in pairs(node.region_divergence) do
      local r1 = rs[1]
      if cx:has_region(r1) then
        local contained = true
        local rs_cases

        local r1_fields = cx:region(r1).field_paths
        local r1_bases = cx:region(r1).base_pointers
        for _, r in ipairs(rs) do
          if r1 ~= r then
            if not cx:has_region(r) then
              contained = false
            else
              local r_base = cx:region(r).base_pointers
              for _, field in ipairs(r1_fields) do
                local r1_base = r1_bases[field:hash()]
                local r_base = r_base[field:hash()]
                if r1_base and r_base then
                  if rs_cases == nil then
                    rs_cases = `([r1_base] == [r_base])
                  else
                    rs_cases = `([rs_cases] and [r1_base] == [r_base])
                  end
                end
              end
            end
          end
        end

        if contained then
          local group = {}
          for _, r in ipairs(rs) do
            group[r] = true
          end
          region_divergence:insert(group)
          if cases == nil then
            cases = rs_cases
          else
            cases = `([cases] and [rs_cases])
          end
        end
      end
    end
    assert(cases)

    local div_cx = cx:new_local_scope()
    local body_div = codegen.block(div_cx, node.body)

    local nodiv_cx = cx:new_local_scope(region_divergence)
    local body_nodiv = codegen.block(nodiv_cx, node.body)

    body = quote
      if [cases] then
        [body_nodiv]
      else
        c.printf(["warning: falling back to slow path in task " .. task.name .. "\n"])
        [body_div]
      end
    end
  else
    body = codegen.block(cx, node.body)
  end

  local proto = task:getdefinition()
  local result_type = std.type_size_bucket_type(return_type)
  terra proto([c_params]): result_type
    [preamble]; -- Semicolon required. This is not an array access.
    [body]
  end

  return task
end

function codegen.stat_fspace(cx, node)
  return node.fspace
end

function codegen.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return codegen.stat_task(cx, node)

  elseif node:is(ast.typed.StatFspace) then
    return codegen.stat_fspace(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function codegen.entry(node)
  local cx = context.new_global_scope()
  return codegen.stat_top(cx, node)
end

return codegen
