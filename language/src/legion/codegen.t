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

local codegen = {}

-- load Legion dynamic library
local c = std.c

local regions = {}
regions.__index = function(t, k) error("context: no such region " .. tostring(k), 2) end

local region = setmetatable({}, { __index = function(t, k) error("region has no field " .. tostring(k), 2) end})
region.__index = region

local context = {}
context.__index = context

function context:new_task_scope(expected_return_type, constraints, task, ctx, runtime)
  assert(expected_return_type and task and ctx and runtime)
  return setmetatable({
    expected_return_type = expected_return_type,
    constraints = constraints,
    task = task,
    context = ctx,
    runtime = runtime,
    regions = setmetatable({}, regions),
  }, context)
end

function context.new_global_scope()
  return setmetatable({
  }, context)
end

function context:has_region(region_type)
  if not rawget(self, "regions") then
    error("not in task context", 2)
  end
  return rawget(self.regions, region_type)
end

function context:add_region_root(region_type, logical_region, index_allocator,
                                 field_allocator, field_paths, field_types,
                                 field_ids, physical_regions, accessors)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  self.regions[region_type] = setmetatable(
    {
      logical_region = logical_region,
      index_allocator = index_allocator,
      field_allocator = field_allocator,
      field_paths = field_paths,
      field_types = field_types,
      field_ids = field_ids,
      physical_regions = physical_regions,
      accessors = accessors,
      parent_region_type = region_type,
    }, region)
end

function context:add_region_subregion(region_type, logical_region, index_allocator,
                                      parent_region_type)
  if not self.regions then
    error("not in task context", 2)
  end
  if self:has_region(region_type) then
    error("region " .. tostring(region_type) .. " already defined in this context", 2)
  end
  if not self.regions[parent_region_type] then
    error("parent to region " .. tostring(region_type) .. " not defined in this context", 2)
  end
  self.regions[region_type] = setmetatable(
    {
      logical_region = logical_region,
      index_allocator = index_allocator,
      field_allocator = self.regions[parent_region_type].field_allocator,
      field_paths = self.regions[parent_region_type].field_paths,
      field_types = self.regions[parent_region_type].field_types,
      field_ids = self.regions[parent_region_type].field_ids,
      physical_regions = self.regions[parent_region_type].physical_regions,
      accessors = self.regions[parent_region_type].accessors,
      parent_region_type = parent_region_type,
    }, region)
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
  local isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
  local actions = quote
    [region_expr.actions]
    var [r] = [std.implicit_cast(
                 static_region_type, region_type, region_expr.value)]
    var [lr] = [r].impl
    var [isa] = c.legion_index_allocator_create(
      [cx.runtime], [cx.context],
      [lr].index_space)
  end

  local parent_region_type = std.search_constraint_predicate(
    cx, region_type, {},
    function(cx, region)
      return cx:has_region(region)
    end)
  if not parent_region_type then
    error("failed to find appropriate for region " .. tostring(region_type) .. " in unpack", 2)
  end

  cx:add_region_subregion(region_type, r, isa, parent_region_type)

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
    return values.ref(self:read(cx), value_type, std.newtuple(field_name))
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

function value:unpack(cx, value_type, field_name, field_type)
  if std.is_region(field_type) and not cx:has_region(field_type) then
    local static_region_type = std.get_field(value_type, field_name)
    local region_expr = self:__get_field(cx, value_type, field_name):read(cx)
    region_expr = unpack_region(cx, region_expr, field_type, static_region_type)
    region_expr = expr.just(region_expr.actions, self.expr.value)
    return self:new(region_expr, self.value_type, self.field_path)
  elseif std.is_ptr(field_type) then
    local region_types = field_type:points_to_regions()

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
  if not terralib.types.istype(value_type) or not std.is_ptr(value_type) then
    error("ref requires a legion ptr type", 2)
  end
  return setmetatable(values.value(value_expr, value_type, field_path), ref)
end

function ref:new(value_expr, value_type, field_path)
  return values.ref(value_expr, value_type, field_path)
end

function ref:__ref(cx)
  local actions = self.expr.actions
  local value = self.expr.value

  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))
  local field_paths, field_types = std.flatten_struct_fields(value_type)
  local absolute_field_paths = field_paths:map(
    function(field_path) return self.field_path .. field_path end)

  local region_types = self.value_type:points_to_regions()
  local accessors_by_region = region_types:map(
    function(region_type)
      return absolute_field_paths:map(
        function(field_path)
          return cx.regions[region_type]:accessor(field_path)
        end)
    end)

  local accessors
  if #region_types == 1 then
    accessors = accessors_by_region[1]
  else
    -- elliott
    accessors = absolute_field_paths:map(
      function(field_path)
        return terralib.newsymbol(c.legion_accessor_array_t, "accessor_" .. field_path:hash())
      end)

    local cases = quote c.abort() end
    for i = #region_types, 1, -1 do
      local region_accessors = accessors_by_region[i]
      cases = quote
        if [value].__index == [i] then
          [std.zip(accessors, region_accessors):map(
             function(pair)
               local accessor, region_accessor = unpack(pair)
               return quote [accessor] = [region_accessor] end
             end)]
        else
          [cases]
        end
      end
    end

    actions = quote
      [actions];
      [accessors:map(
         function(accessor) return quote var [accessor] end end)];
      [cases]
    end
  end

  local values = std.zip(accessors, field_types):map(
    function(pair)
      local accessor, field_type = unpack(pair)
      return `(
        @([&field_type](c.legion_accessor_array_ref([accessor], [value].__ptr))))
    end)
  return actions, values, value_type, field_paths, field_types
end

function ref:read(cx)
  local actions, values, value_type, field_paths, field_types = self:__ref(cx)
  local value = terralib.newsymbol(value_type)
  actions = quote
    [actions];
    var [value] : value_type
    [std.zip(values, field_paths):map(
       function(pair)
         local field_value, field_path = unpack(pair)
         local result = value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         return quote [result] = [field_value] end
      end)]
  end
  return expr.just(actions, value)
end

function ref:write(cx, value)
  local value_expr = value:read(cx)
  local actions, values, value_type, field_paths, field_types = self:__ref(cx)
  actions = quote
    [value_expr.actions];
    [actions];
    [std.zip(values, field_paths):map(
       function(pair)
         local field_value, field_path = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         return quote [field_value] = [result] end
      end)]
  end
  return expr.just(actions, quote end)
end

function ref:reduce(cx, value, op)
  local value_expr = value:read(cx)
  local actions, values, value_type, field_paths, field_types = self:__ref(cx)
  actions = quote
    [value_expr.actions];
    [actions];
    [std.zip(values, field_paths):map(
       function(pair)
         local field_value, field_path = unpack(pair)
         local result = value_expr.value
         for _, field_name in ipairs(field_path) do
           result = `([result].[field_name])
         end
         return quote
           [field_value] = [std.quote_binary_op(
                              op, field_value, result)]
         end
      end)]
  end
  return expr.just(actions, quote end)
end

function ref:get_field(cx, field_name, field_type)
  local value_type = std.as_read(
    std.get_field_path(self.value_type.points_to_type, self.field_path))

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
  local value_expr = value:read(cx)
  local ref_expr = self:__ref(cx)
  local actions = quote
    [value_expr.actions];
    [ref_expr.actions];
    [ref_expr.value] = [std.quote_binary_op(
                          op, ref_expr.value, value_expr.value)]
  end
  return expr.just(actions, quote end)
end

function rawref:get_field(cx, field_name, field_type)
  local value_type = std.as_read(
    std.get_field_path(self.value_type.type, self.field_path))

  local result = self:unpack(cx, value_type, field_name, field_type)
  return result:__get_field(cx, value_type, field_name)
end

function codegen.expr_internal(cx, node)
  return node.value
end

function codegen.expr_id(cx, node)
  return values.rawref(expr.just(quote end, node.value), &node.expr_type)
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
  local field_name = node.field_name
  local field_type = node.expr_type
  return codegen.expr(cx, node.value):get_field(cx, field_name, field_type)
end

function codegen.expr_index_access(cx, node)
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local index = codegen.expr(cx, node.index):read(cx)

  local actions = quote [value.actions]; [index.actions] end

  local expr_type = std.as_read(node.expr_type)
  if std.is_partition(value_type) then
    local parent_region_type = value_type:parent_region()

    local r = terralib.newsymbol(expr_type, "r")
    local lr = terralib.newsymbol(c.legion_logical_region_t, "lr")
    local isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
    actions = quote
      [actions];
      var [lr] = c.legion_logical_partition_get_logical_subregion_by_color(
        [cx.runtime], [cx.context],
        [value.value].impl, [index.value])
      var [isa] = c.legion_index_allocator_create(
        [cx.runtime], [cx.context],
        [lr].index_space)
      var [r] = [expr_type] { impl = [lr] }
    end

    cx:add_region_subregion(expr_type, r, isa, parent_region_type)

    return values.value(expr.just(actions, r), expr_type)
  else
    return values.rawref(
      expr.once_only(actions, `([value.value][ [index.value] ])),
      &expr_type)
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
                                   params_struct_type, task_args,
                                   task_args_setup)
  -- Prepare the by-value arguments to the task.
  for i, arg in ipairs(args) do
    local c_field = params_struct_type:getentries()[i]
    local c_field_name = c_field[1] or c_field.field
    if terralib.issymbol(c_field_name) then
      c_field_name = c_field_name.displayname
    end
    task_args_setup:insert(quote [task_args].[c_field_name] = [arg] end)
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
        local arg_field_id = cx.regions[arg_type]:field_id(field_path)
        local param_field_id = param_field_ids[param_field_id_i]
        param_field_id_i = param_field_id_i + 1
        task_args_setup:insert(
          quote [task_args].[param_field_id] = [arg_field_id] end)
      end
    end
  end
  return task_args_setup
end

function expr_call_setup_region_arg(cx, task, arg_type, param_type, launcher,
                                    index, region_args_setup)
  local privileges, privilege_field_paths = std.find_task_privileges(
    param_type, task:getprivileges())
  local privilege_modes = privileges:map(std.privilege_mode)
  local parent_region =
    cx.regions[cx.regions[arg_type].parent_region_type].logical_region

  local add_requirement = c.legion_task_launcher_add_region_requirement_logical_region
  if index then
    add_requirement = c.legion_index_launcher_add_region_requirement_logical_region
  end

  local add_field = c.legion_task_launcher_add_field
  if index then
    add_field = c.legion_index_launcher_add_field
  end

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local privilege_mode = privilege_modes[i]

    local requirement = terralib.newsymbol("requirement")
    region_args_setup:insert(
      quote
      var [requirement] =
          add_requirement(
            [launcher], [cx.regions[arg_type].logical_region].impl,
            [privilege_mode], c.EXCLUSIVE,
            [parent_region].impl,
            0, false)
        [field_paths:map(
           function(field_path)
             local field_id = cx.regions[arg_type]:field_id(field_path)
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
  local privileges, privilege_field_paths = std.find_task_privileges(
    param_type, task:getprivileges())
  local privilege_modes = privileges:map(std.privilege_mode)
  local parent_region =
    cx.regions[cx.regions[arg_type].parent_region_type].logical_region

  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local privilege_mode = privilege_modes[i]

    local requirement = terralib.newsymbol("requirement")
    region_args_setup:insert(
      quote
      var [requirement] =
          c.legion_index_launcher_add_region_requirement_logical_partition(
            [launcher], [partition].impl, 0 --[[ default projection ID ]],
            [privilege_mode], c.EXCLUSIVE,
            [parent_region].impl,
            0, false)
        [field_paths:map(
           function(field_path)
             local field_id = cx.regions[arg_type]:field_id(field_path)
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
    function(arg) return codegen.expr(cx, arg):read(cx) end)

  local actions = quote
    [fn.actions];
    [args:map(function(arg) return arg.actions end)]
  end

  local arg_types = terralib.newlist()
  for i, arg in ipairs(args) do
    arg_types:insert(node.args[i].expr_type)
  end

  local arg_values = terralib.newlist()
  local param_types = node.fn.expr_type.parameters
  for i, arg in ipairs(args) do
    local arg_value = args[i].value
    if i <= #param_types and param_types[i] ~= std.untyped then
      arg_values:insert(std.implicit_cast(arg_types[i], param_types[i], arg_value))
    else
      arg_values:insert(arg_value)
    end
  end

  local value_type = std.as_read(node.expr_type)
  if std.is_task(fn.value) then
    value_type = fn.value:gettype().returntype

    local params_struct_type = fn.value:get_params_struct()
    local task_args = terralib.newsymbol(params_struct_type)
    local task_args_setup = terralib.newlist()
    expr_call_setup_task_args(
      cx, fn.value, arg_values, arg_types, param_types, params_struct_type,
      task_args, task_args_setup)

    -- Pass regions through region requirements.
    local launcher = terralib.newsymbol("launcher")
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
      [region_args_setup]
      var [future] = c.legion_task_launcher_execute(
        [cx.runtime], [cx.context], [launcher])
    end
    local launcher_cleanup = quote
      c.legion_future_destroy(future)
      c.legion_task_launcher_destroy(launcher)
    end

    if value_type == terralib.types.unit then
      actions = quote
        [actions]
        [launcher_setup]
        [launcher_cleanup]
      end
      return values.value(expr.just(actions, quote end), terralib.types.unit)
    else
      local result_value = terralib.newsymbol("result_value")
      local value_type_alignment = std.min(terralib.sizeof(value_type), 8)
      actions = quote
        [actions]
        [launcher_setup]
        var result = c.legion_future_get_result([future])
        -- Force unaligned access because malloc does not provide
        -- blocks aligned for all purposes (e.g. SSE vectors).
        var [result_value] = terralib.attrload(
          [&value_type](result.value),
          { align = [value_type_alignment] })
        c.legion_task_result_destroy(result)
        [launcher_cleanup]
      end
      return values.value(
        expr.just(actions, result_value),
        value_type)
    end
  else
    return values.value(
      expr.once_only(actions, `([fn.value]([arg_values]))),
      value_type)
  end
end

function codegen.expr_cast(cx, node)
  local fn = codegen.expr(cx, node.fn):read(cx)
  local arg = codegen.expr(cx, node.arg):read(cx)

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

  local region = cx.regions[region_type]
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

  local region = cx.regions[region_type]
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
  local isa = cx.regions[region_type].index_allocator

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
    local lr = `([cx.regions[region].logical_region].impl)
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
      local lr = `([cx.regions[region].logical_region].impl)
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
    result = `([expr_type]{ __ptr = [input].__ptr })
  else
    result = terralib.newsymbol(expr_type)
    local input_regions = value_type:points_to_regions()
    local result_last = node.parent_region_map[#input_regions]
    local cases = quote
      [result] = [expr_type]({
          __ptr = [input].__ptr,
          __index = [result_last]
      })
    end
    for i = #input_regions - 1, 1, -1 do
      local result_i = node.parent_region_map[i]
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
      return terralib.newsymbol(c.legion_accessor_array_t, "accessor_" .. field_path:hash())
    end)

  cx:add_region_root(region_type, r, isa, fsa,
                     field_paths,
                     std.dict(std.zip(field_paths:map(std.hash), field_types)),
                     std.dict(std.zip(field_paths:map(std.hash), field_ids)),
                     std.dict(std.zip(field_paths:map(std.hash), physical_regions)),
                     std.dict(std.zip(field_paths:map(std.hash), accessors)))

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
    [std.zip(field_ids, accessors):map(
       function(field)
         local field_id, accessor = unpack(field)
         return quote
           var [accessor] = c.legion_physical_region_get_field_accessor_array([pr], [field_id])
         end
       end)]
    var [r] = [region_type]{ impl = [lr] }
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

function codegen.expr_unary(cx, node)
  local rhs = codegen.expr(cx, node.rhs):read(cx)
  local expr_type = std.as_read(node.expr_type)
  return values.value(
    expr.once_only(rhs.actions, std.quote_unary_op(node.op, rhs.value)),
    expr_type)
end

function codegen.expr_binary(cx, node)
  local lhs = codegen.expr(cx, node.lhs):read(cx)
  local rhs = codegen.expr(cx, node.rhs):read(cx)

  local actions = quote [lhs.actions]; [rhs.actions] end
  local expr_type = std.as_read(node.expr_type)
  return values.value(
    expr.once_only(actions, std.quote_binary_op(node.op, lhs.value, rhs.value)),
    expr_type)
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

  elseif node:is(ast.typed.ExprUnary) then
    return codegen.expr_unary(cx, node)

  elseif node:is(ast.typed.ExprBinary) then
    return codegen.expr_binary(cx, node)

  elseif node:is(ast.typed.ExprDeref) then
    return codegen.expr_deref(cx, node)

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
  local then_block = codegen.block(cx, node.then_block)
  clauses:insert({cond, then_block})

  -- Add rest of clauses.
  for _, elseif_block in ipairs(node.elseif_blocks) do
    local cond = codegen.expr(cx, elseif_block.cond):read(cx)
    local block = codegen.block(cx, elseif_block.block)
    clauses:insert({cond, block})
  end
  local else_block = codegen.block(cx, node.else_block)

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
  local block = codegen.block(cx, node.block)
  return quote
    while [quote [cond.actions] in [cond.value] end] do
      [block]
    end
  end
end

function codegen.stat_for_num(cx, node)
  local symbol = node.symbol
  local bounds = codegen.expr_list(cx, node.values):map(function(value) return value:read(cx) end)
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
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local block = codegen.block(cx, node.block)

  local regions = symbol.type.points_to_region_symbols
  assert(#regions == 1)
  local lr = regions[1]
  return quote
    do
      [value.actions]
      var [lr] = [value.value]
      var is = [lr].impl.index_space
      var it = c.legion_index_iterator_create(is)
      while c.legion_index_iterator_has_next(it) do
        var rawptr = c.legion_index_iterator_next(it)
        var [symbol] = [symbol.type]{ __ptr = rawptr }
        do
          [block]
        end
      end
      c.legion_index_iterator_destroy(it)
    end
  end
end

function codegen.stat_repeat(cx, node)
  local block = codegen.block(cx, node.block)
  local until_cond = codegen.expr(cx, node.until_cond):read(cx)
  return quote
    repeat
      [block]
    until [quote [until_cond.actions] in [until_cond.value] end]
  end
end

function codegen.stat_block(cx, node)
  return quote
    do
      [codegen.block(cx, node.block)]
    end
  end
end

function codegen.stat_index_launch(cx, node)
  local symbol = node.symbol
  local domain = codegen.expr_list(cx, node.domain):map(function(value) return value:read(cx) end)

  local fn = codegen.expr(cx, node.expr.fn):read(cx)
  assert(std.is_task(fn.value))
  local args = terralib.newlist()
  local args_partitions = {}
  for i, arg in ipairs(node.expr.args) do
    if node.args_invariant[i] then
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
        }):read(cx)
      args:insert(region)
    end
  end

  local actions = quote
    [domain[1].actions];
    [domain[2].actions];
    -- ignore domain[3] because we know it is a constant
    [fn.actions];
    [std.zip(args, node.args_invariant):map(
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
    arg_types:insert(node.expr.args[i].expr_type)
  end

  local arg_values = terralib.newlist()
  local param_types = node.expr.fn.expr_type.parameters
  for i, arg in ipairs(args) do
    local arg_value = args[i].value
    if i <= #param_types and param_types[i] ~= std.untyped then
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
    local invariant = node.args_invariant[i]
    if not invariant then
      task_args_setup:insert(arg.actions)
    end
  end
  expr_call_setup_task_args(
    cx, fn.value, arg_values, arg_types, param_types, params_struct_type,
    task_args, task_args_setup)

  -- Pass regions through region requirements.
  local launcher = terralib.newsymbol("launcher")
  local region_args_setup = terralib.newlist()
  for _, i in ipairs(std.fn_param_regions_by_index(fn.value:gettype())) do
    local arg_type = arg_types[i]
    local param_type = param_types[i]

    if node.args_invariant[i] then
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
  local future_map = terralib.newsymbol("future_map")
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
    [region_args_setup]
    var [future_map] = c.legion_index_launcher_execute(
      [cx.runtime], [cx.context], [launcher])
  end
  local launcher_cleanup = quote
  c.legion_argument_map_destroy([argument_map])
  c.legion_future_map_destroy([future_map])
  c.legion_index_launcher_destroy([launcher])
  end

  actions = quote
    [actions];
    [launcher_setup];
    [launcher_cleanup]
  end
  return actions
end

function codegen.stat_var(cx, node)
  local lhs = node.symbols
  local rhs = codegen.expr_list(cx, node.values):map(function(rh) return rh:read(cx) end)

  local rhs_values = terralib.newlist()
  for i, rh in ipairs(rhs) do
    local rhs_type = std.as_read(node.values[i].expr_type)
    local lhs_type = lhs[i].type
    if lhs_type then
      rhs_values:insert(std.implicit_cast(rhs_type, lhs_type, rh.value))
    else
      rhs_values:insert(rh.value)
    end
  end
  local actions = rhs:map(function(rh) return rh.actions end)

  if #rhs > 0 then
    return quote [actions]; var [lhs] = [rhs_values] end
  else
    return quote var [lhs] end
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
  local value = codegen.expr(cx, node.value):read(cx)
  local value_type = std.as_read(node.value.expr_type)
  local result_type = cx.expected_return_type
  return quote
    [value.actions]
    var result = [std.implicit_cast(value_type, result_type, value.value)]
    return c.legion_task_result_create(
      [&opaque](&result),
      terralib.sizeof([result_type]))
  end
end

function codegen.stat_break(cx, node)
  return quote break end
end

function codegen.stat_assignment(cx, node)
  local actions = terralib.newlist()
  local lhs = codegen.expr_list(cx, node.lhs)
  local rhs = codegen.expr_list(cx, node.rhs)
  rhs = std.zip(rhs, node.rhs):map(
    function(pair)
      local rh_value, rh_node = unpack(pair)
      local rh_expr = rh_value:read(cx)
      actions:insert(rh_expr.actions)
      return values.value(
        expr.just(quote end, rh_expr.value),
        std.as_read(rh_node.expr_type))
    end)

  actions:insertall(
    std.zip(lhs, rhs):map(
      function(pair)
        local lh, rh = unpack(pair)
        return lh:write(cx, rh).actions
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
      local rh_expr = rh_value:read(cx)
      actions:insert(rh_expr.actions)
      return values.value(
        expr.just(quote end, rh_expr.value),
        std.as_read(rh_node.expr_type))
    end)

  actions:insertall(
    std.zip(lhs, rhs):map(
      function(pair)
        local lh, rh = unpack(pair)
        return lh:reduce(cx, rh, node.op).actions
      end))

  return quote [actions] end
end

function codegen.stat_expr(cx, node)
  local expr = codegen.expr(cx, node.expr):read(cx)
  return quote [expr.actions] end
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

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function codegen.stat_task(cx, node)
  local task = node.prototype
  std.register_task(task)

  local params_struct_type = terralib.types.newstruct()
  params_struct_type.entries = node.params:map(
    function(param)
      return { field = param.symbol.displayname, type = param.param_type }
    end)
  task:set_params_struct(params_struct_type)

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

  local cx = cx:new_task_scope(return_type, task:get_constraints(), c_task, c_context, c_runtime)

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
    for _, param in ipairs(params) do
      local param_type_alignment = std.min(terralib.sizeof(param.type), 8)
      task_args_setup:insert(quote
        -- Force unaligned access because malloc does not provide
        -- blocks aligned for all purposes (e.g. SSE vectors).
        var [param] = terralib.attrload(
          (&args.[param.displayname]),
          { align = [param_type_alignment] })
      end)
    end
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
      local isa = terralib.newsymbol(c.legion_index_allocator_t, "isa")
      local fsa = terralib.newsymbol(c.legion_field_allocator_t, "fsa")

      local privileges, privilege_field_paths = std.find_task_privileges(
        region_type, task:getprivileges())

      local privileges_by_field_path = {}
      for i, privilege in ipairs(privileges) do
        local field_paths = privilege_field_paths[i]
        for _, field_path in ipairs(field_paths) do
          privileges_by_field_path[field_path:hash()] = privilege
        end
      end

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
      for _, field_paths in ipairs(privilege_field_paths) do
        local physical_region = terralib.newsymbol(
          c.legion_physical_region_t,
          "pr_" .. tostring(physical_region_i))

        physical_regions:insert(physical_region)
        physical_regions_index:insert(physical_region_i)
        physical_region_i = physical_region_i + 1

        local physical_region_accessors = field_paths:map(
          function(field_path)
            return terralib.newsymbol(
              c.legion_accessor_array_t,
              "accessor_" .. field_path:hash())
          end)
        accessors:insert(physical_region_accessors)

        for i, field_path in ipairs(field_paths) do
          physical_regions_by_field_path[field_path:hash()] = physical_region
          if privileges_by_field_path[field_path:hash()] ~= "none" then
            accessors_by_field_path[field_path:hash()] = physical_region_accessors[i]
          end
        end
      end

      region_args_setup:insert(quote
        var is = [r].impl.index_space
        var [isa] = c.legion_index_allocator_create([cx.runtime], [cx.context],  is)
        var fs = [r].impl.field_space
        var [fsa] = c.legion_field_allocator_create([cx.runtime], [cx.context],  fs)
      end)

      for i, field_paths in ipairs(privilege_field_paths) do
        local physical_region = physical_regions[i]
        local physical_region_index = physical_regions_index[i]

        region_args_setup:insert(quote
          var [physical_region] = [c_regions][ [physical_region_index] ]
        end)

        for j, field_path in ipairs(field_paths) do
          local accessor = accessors[i][j]
          local field_id = field_ids_by_field_path[field_path:hash()]
          assert(accessor and field_id)

          if privileges_by_field_path[field_path:hash()] ~= "none" then
            region_args_setup:insert(quote
              var [accessor] = c.legion_physical_region_get_field_accessor_array(
                [physical_region], [field_id])
            end)
          end
        end
      end

      cx:add_region_root(region_type, r, isa, fsa,
                         field_paths,
                         std.dict(std.zip(field_paths:map(std.hash), field_types)),
                         field_ids_by_field_path,
                         physical_regions_by_field_path,
                         accessors_by_field_path)
    end
  end

  local preamble = quote [task_args_setup]; [region_args_setup] end

  local body = codegen.block(cx, node.body)

  local proto = task:getdefinition()
  if return_type == terralib.types.unit then
    terra proto([c_params]): terralib.types.unit
      [preamble]; -- Semicolon required. This is not an array access.
      [body]
    end
  else
    terra proto([c_params]): c.legion_task_result_t
      [preamble]; -- Semicolon required. This is not an array access.
      [body]
    end
  end

  proto:compile()

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
