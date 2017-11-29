-- Copyright 2017 Stanford University
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

-- Helper for Generating C Header Files for Tasks

-- TODO: Features NOT supported by this interface
--   * Futures
--   * Barriers
--   * Index launches
--   * Persisitent launchers (is this an objective?)

local ast = require("regent/ast")
local data = require("common/data")
local base = require("regent/std_base")

local header_helper = {}

local c = base.c

function header_helper.normalize_name(name)
  return string.gsub(
    string.gsub(
      string.gsub(name, ".*/", ""),
      "[<>]", ""),
    "[^A-Za-z0-9]", "_")
end

local function get_helper_name(task)
  return header_helper.normalize_name(tostring(task:get_name())) .. "_call"
end

local function get_task_params(task)
  return task.param_symbols:map(
    function(param)
      local result = terralib.newlist()

      local param_type = param:gettype()

      local primary_type, primary_ctype
      if base.types.is_region(param_type) then
        primary_type, primary_ctype = c.legion_logical_region_t, "legion_logical_region_t"
      elseif base.types.is_ispace(param_type) then
        primary_type, primary_ctype = c.legion_index_space_t, "legion_index_space_t"
      elseif param_type == int32 then
        primary_type, primary_ctype = int32, "int32_t"
      elseif param_type == int64 then
        primary_type, primary_ctype = int64, "int64_t"
      elseif param_type == uint32 then
        primary_type, primary_ctype = uint32, "uint32_t"
      elseif param_type == uint64 then
        primary_type, primary_ctype = uint64, "uint64_t"
      else
        assert(false, "unknown type " .. tostring(param_type))
      end

      assert(primary_type and primary_ctype)
      result:insert({primary_type, primary_ctype, param:getname()})

      -- Add secondary symbols for special types
      if base.types.is_region(param_type) then
        result:insert({c.legion_logical_region_t, "legion_logical_region_t", param:getname() .. "_parent"})
        result:insert({&c.legion_field_id_t, "legion_field_id_t *", param:getname() .. "_fields"})
        result:insert({c.size_t, "size_t", param:getname() .. "_num_fields"})
      end

      return result
    end)
end

local function render_c_params(params)
  local result = terralib.newlist()
  for _, param_list in ipairs(params) do
    for _, param in ipairs(param_list) do
      local param_type, param_ctype, param_name = unpack(param)
      result:insert(param_ctype .. " " .. header_helper.normalize_name(param_name))
    end
  end
  return result:concat(", ")
end

function header_helper.generate_task_interface(task)
  local name = get_helper_name(task)
  local params = render_c_params(get_task_params(task))
  return string.format("legion_future_t %s(legion_runtime_t runtime, legion_context_t context, %s);", name, params)
end

function header_helper.generate_task_implementation(task)
  local name = get_helper_name(task)
  local params = get_task_params(task)

  local param_symbols = params:map(
    function(param_list)
      return param_list:map(
        function(param)
          local param_type, param_ctype, param_name = unpack(param)
          return terralib.newsymbol(param_type, param_name)
        end)
    end)

  local runtime = terralib.newsymbol(c.legion_runtime_t, "runtime")
  local context = terralib.newsymbol(c.legion_context_t, "context")

  local params_struct_type = task:get_params_struct()
  local params_struct = terralib.newsymbol(params_struct_type, "params_struct_type")

  local task_args = terralib.newsymbol(c.legion_task_argument_t, "task_args")
  local task_args_setup = quote
    var [params_struct]
    var [task_args]
    [task_args].args = &[params_struct]
    [task_args].arglen = [terralib.sizeof(params_struct_type)]
  end

  local launcher = terralib.newsymbol(c.legion_task_launcher_t, "launcher")
  local launcher_setup = quote
    var [launcher] = c.legion_task_launcher_create(
      [task:get_task_id()], [task_args],
      c.legion_predicate_true(), 0, 0)
  end

  local param_field_ids = task:get_field_id_param_labels()

  local args_setup = terralib.newlist()
  for i, task_param_symbol in ipairs(task:get_param_symbols()) do
    local param_type = task_param_symbol:gettype()
    if not base.types.is_region(param_type) then
      assert(#param_symbols[i] == 1)
    end

    -- Pack the primary value
    local arg_value = param_symbols[i][1]

    -- Handle conversions from C to Regent types
    local cast_value = arg_value
    if base.types.is_region(param_type) then
      cast_value = `([param_type] { impl = [arg_value] })
    elseif base.types.is_ispace(param_type) then
      cast_value = `([param_type] { impl = [arg_value] })
    end

    local c_field = params_struct_type:getentries()[i + 1]
    local c_field_name = c_field[1] or c_field.field
    if terralib.issymbol(c_field_name) then
      c_field_name = c_field_name.displayname
    end
    args_setup:insert(
      quote
        [params_struct].[c_field_name] = [cast_value]
      end)

    -- Pack secondary values
    if base.types.is_region(param_type) then
      -- Pack region fields
      local parent_region = param_symbols[i][2]
      local field_ids = param_symbols[i][3]
      local field_count = param_symbols[i][4]

      local field_paths, _ = base.types.flatten_struct_fields(param_type:fspace())
      args_setup:insert(
        quote
            base.assert([field_count] == [#field_paths],
              ["wrong number of fields for region " .. tostring(arg_value) .. " (argument " .. i .. ")"])
        end)
      local field_id_by_path = data.newmap()
      for j, field_path in pairs(field_paths) do
        local arg_field_id = `([field_ids][ [j-1] ])
        local param_field_id = param_field_ids[i][j]
        field_id_by_path[field_path] = arg_field_id
        args_setup:insert(
          quote [params_struct].[param_field_id] = [arg_field_id] end)
      end

      -- Add region requirements
      local privileges, privilege_field_paths, privilege_field_types, coherences, flags =
        base.find_task_privileges(param_type, task)
      local privilege_modes = privileges:map(base.privilege_mode)
      local coherence_modes = coherences:map(base.coherence_mode)

      for i, privilege in ipairs(privileges) do
        local field_paths = privilege_field_paths[i]
        local field_types = privilege_field_types[i]
        local privilege_mode = privilege_modes[i]
        local coherence_mode = coherence_modes[i]
        local flag = base.flag_mode(flags[i])

        local reduction_op
        if base.is_reduction_op(privilege) then
          local op = base.get_reduction_op(privilege)
          assert(#field_types == 1)
          local field_type = field_types[1]
          reduction_op = base.reduction_op_ids[op][field_type]
        end

        if privilege_mode == c.REDUCE then
          assert(reduction_op)
        end

        local add_requirement
        if reduction_op then
          add_requirement = c.legion_task_launcher_add_region_requirement_logical_region_reduction
        else
          add_requirement = c.legion_task_launcher_add_region_requirement_logical_region
        end
        assert(add_requirement)

        local add_field = c.legion_task_launcher_add_field

        local add_flags = c.legion_task_launcher_add_flags


        local requirement = terralib.newsymbol(uint, "requirement")
        local requirement_args = terralib.newlist({
            launcher, arg_value})
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
                 local field_id = field_id_by_path[field_path]
                 return quote
                   add_field(
                     [launcher], [requirement], [field_id], true)
                 end
               end)]
            [add_flags]([launcher], [requirement], [flag])
          end)
      end
    end
  end

  local launcher_execute = quote
    var future = c.legion_task_launcher_execute(
      [runtime], [context], [launcher])
    c.legion_task_launcher_destroy([launcher])
    return future
  end

  local actions = terralib.newlist()
  actions:insert(task_args_setup)
  actions:insert(launcher_setup)
  actions:insertall(args_setup)
  actions:insert(launcher_execute)

  local terra helper([runtime], [context], [data.flatten(param_symbols)])
    [actions]
  end
  helper:setname(name)

  return name, helper
end

return header_helper
