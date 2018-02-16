-- Copyright 2018 Stanford University
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

-- C++ Interface:
--
-- my_task_launcher launcher; // optional: mapper, mapper tag
-- launcher.add_argument_x(1234);
-- std::vector<FieldID> fields; // fill fields for r
-- launcher.add_argument_r(region, parent, fields);
-- Future f = launcher.execute(runtime, context);

-- C Interface:
--
-- my_task_launcher_t launcher = my_task_launcher_create(0, 0); // mapper, mapper tag
-- my_task_launcher_add_argument_x(launcher, 1234);
-- legion_field_id_t *fields = ...; // malloc an array of fields and fill
-- size_t num_fields = 5;
-- my_task_launcher_add_argument_r(region, parent, fields, num_fields);
-- legion_future_t f = my_task_launcher_execute(runtime, context, launcher);
-- my_task_launcher_destroy(launcher);

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

local function get_launcher_name(task)
  return header_helper.normalize_name(tostring(task:get_name())) .. "_launcher"
end

local function get_task_params(task)
  return task.param_symbols:map(
    function(param)
      local result = terralib.newlist()

      local param_type = param:gettype()

      local terra_type, c_type, cxx_type
      if base.types.is_region(param_type) then
        terra_type, c_type, cxx_type = c.legion_logical_region_t, "legion_logical_region_t", "Legion::LogicalRegion"
      elseif base.types.is_ispace(param_type) then
        terra_type, c_type, cxx_type = c.legion_index_space_t, "legion_index_space_t", "Legion::IndexSpace"
      elseif param_type == int32 then
        terra_type, c_type, cxx_type = int32, "int32_t",  "int32_t"
      elseif param_type == int64 then
        terra_type, c_type, cxx_type = int64, "int64_t", "int64_t"
      elseif param_type == uint32 then
        terra_type, c_type, cxx_type = uint32, "uint32_t", "uint32_t"
      elseif param_type == uint64 then
        terra_type, c_type, cxx_type = uint64, "uint64_t", "uint64_t"
      else
        assert(false, "unknown type " .. tostring(param_type))
      end

      assert(terra_type and c_type and cxx_type)
      result:insert({terra_type, c_type, cxx_type, param:getname()})

      -- Add secondary symbols for special types
      if base.types.is_region(param_type) then
        result:insert({c.legion_logical_region_t, "legion_logical_region_t", "Legion::LogicalRegion", param:getname() .. "_parent"})
        result:insert({&c.legion_field_id_t, "const legion_field_id_t *", "const std::vector<Legion::FieldID> &", param:getname() .. "_fields"})
        result:insert({c.size_t, "size_t", false, param:getname() .. "_num_fields"})
      end

      return result
    end)
end

local function render_c_params(param_list)
  local result = terralib.newlist()
  for _, param in ipairs(param_list) do
    local terra_type, c_type, cxx_type, param_name = unpack(param)
    result:insert(c_type .. " " .. header_helper.normalize_name(param_name))
  end
  return result
end

local function render_cxx_params(param_list)
  local result = terralib.newlist()
  for _, param in ipairs(param_list) do
    local terra_type, c_type, cxx_type, param_name = unpack(param)
    if cxx_type then
      result:insert(cxx_type .. " " .. header_helper.normalize_name(param_name))
    end
  end
  return result
end

local function render_c_proto(name, args, return_type)
  return string.format(
    "%s %s(%s);",
    return_type, name, args:concat(", "))
end

local function render_c_def(name, args, return_type, body)
  if return_type == nil then return_type = "" end
  local result = terralib.newlist()
  result:insert(string.format(
    "%s %s(%s) {",
    return_type, name, args:concat(", ")))
  result:insertall(body:map(function(line) return "  " .. line end))
  result:insert("}")
  return result
end

function header_helper.generate_task_c_interface(task)
  local name = get_launcher_name(task)

  local launcher_type = name .. "_t"

  local launcher_proto = string.format(
    "typedef struct %s { void *impl; } %s;",
    launcher_type, launcher_type)

  local create_name = name .. "_create"
  local destroy_name = name .. "_destroy"
  local execute_name = name .. "_execute"

  local create_args = terralib.newlist({
    "legion_predicate_t pred /* = legion_predicate_true() */",
    "legion_mapper_id_t id /* = 0 */",
    "legion_mapping_tag_id_t tag /* = 0 */",
  })
  local create_proto = render_c_proto(
    create_name, create_args, launcher_type)

  local destroy_args = terralib.newlist({
    launcher_type .. " launcher",
  })
  local destroy_proto = render_c_proto(
    destroy_name, destroy_args, "void")

  local execute_args = terralib.newlist({
    "legion_runtime_t runtime",
    "legion_context_t context",
    launcher_type .. " launcher",
  })
  local execute_proto = render_c_proto(
    execute_name, execute_args, "legion_future_t")

  local result = terralib.newlist({launcher_proto, create_proto, destroy_proto, execute_proto})

  local params = get_task_params(task)
  for _, param_list in ipairs(params) do
    local param_name = header_helper.normalize_name(param_list[1][4])
    local add_name = name .. "_add_argument_" .. param_name
    local add_args = render_c_params(param_list)
    add_args:insert(1, launcher_type .. " launcher")
    add_args:insert("bool overwrite /* = false */")
    local add_proto = render_c_proto(add_name, add_args, "void")
    result:insert(add_proto)
  end

  return result:concat("\n\n")
end

function header_helper.generate_task_cxx_interface(task)
  local name = get_launcher_name(task)

  local c_launcher_type = name .. "_t"

  local c_create_name = name .. "_create"
  local c_destroy_name = name .. "_destroy"
  local c_execute_name = name .. "_execute"

  local launcher_type = name

  local predicate_field = "predicate"
  local launcher_field = "launcher"

  local ctor_name = launcher_type
  local dtor_name = "~" .. launcher_type
  local execute_name = "execute"

  local ctor_args = terralib.newlist({
    "Legion::Predicate pred = Legion::Predicate::TRUE_PRED",
    "Legion::MapperID id = 0",
    "Legion::MappingTagID tag = 0",
  })
  local ctor_def = render_c_def(
    ctor_name, ctor_args, nil,
    terralib.newlist({
      "predicate = new Legion::Predicate(pred);",
      "launcher = " .. c_create_name .. "(Legion::CObjectWrapper::wrap(predicate), id, tag);",
    }))

  local dtor_args = terralib.newlist()
  local dtor_def = render_c_def(
    dtor_name, dtor_args, nil,
    terralib.newlist({
      c_destroy_name .. "(launcher);",
      "delete predicate;",
    }))

  local execute_args = terralib.newlist({
    "Legion::Runtime *runtime",
    "Legion::Context ctx",
  })
  local execute_def = render_c_def(
    execute_name, execute_args, "Legion::Future",
    terralib.newlist({
        "legion_runtime_t c_runtime = Legion::CObjectWrapper::wrap(runtime);",
        "Legion::CContext c_ctx(ctx);",
        "legion_context_t c_context = Legion::CObjectWrapper::wrap(&c_ctx);",
        "legion_future_t c_future = " .. c_execute_name .. "(c_runtime, c_context, launcher);",
        "Legion::Future future = *Legion::CObjectWrapper::unwrap(c_future);",
        "legion_future_destroy(c_future);",
        "return future;",
    }))

  local body = terralib.newlist()
  body:insertall(ctor_def)
  body:insertall(dtor_def)
  body:insertall(execute_def)

  local task_param_symbols = task:get_param_symbols()
  local params = get_task_params(task)
  for i, param_list in ipairs(params) do
    local param_name = header_helper.normalize_name(param_list[1][4])
    local param_type = task_param_symbols[i]:gettype()

    local c_add_name = name .. "_add_argument_" .. param_name
    local add_name = "add_argument_" .. param_name

    local add_args = render_cxx_params(param_list)
    add_args:insert("bool overwrite = false")

    local actions = terralib.newlist()

    local c_add_args
    if base.types.is_region(param_type) then
      local region = param_list[1][4]
      local parent = param_list[2][4]
      local fields = param_list[3][4]
      actions:insert("legion_logical_region_t c_region = Legion::CObjectWrapper::wrap(" .. region .. ");")
      actions:insert("legion_logical_region_t c_parent = Legion::CObjectWrapper::wrap(" .. parent .. ");")
      actions:insert("const legion_field_id_t *c_fields = &(" .. fields .. ".front());")
      actions:insert("size_t c_num_fields = " .. fields .. ".size();")
      c_add_args = terralib.newlist({"c_region", "c_parent", "c_fields", "c_num_fields"})
    else
      c_add_args = param_list:map(
        function(param)
          local terra_type, c_type, cxx_type, param_name = unpack(param)
          return param_name
        end)
    end

    c_add_args:insert(1, "launcher")
    c_add_args:insert("overwrite")

    actions:insert(
      string.format("%s(%s);", c_add_name, c_add_args:concat(", ")))

    local add_def = render_c_def(add_name, add_args, "void", actions)
    body:insertall(add_def)
  end

  local result = terralib.newlist()
  result:insert(string.format("class %s {", launcher_type))
  result:insert("public:")
  result:insertall(body:map(function(line) return "  " .. line end))
  result:insert("private:")
  result:insert(string.format("  Legion::Predicate *%s;", predicate_field))
  result:insert(string.format("  %s %s;", c_launcher_type, launcher_field))
  result:insert("};")

  return result:concat("\n")
end

local function make_internal_launcher_state_types(params_struct_type)
  local struct wrapper_type {
    impl: &opaque,
  }
  local struct state_type {
    launcher: c.legion_task_launcher_t,
    task_args: &params_struct_type,
    args_provided: &int8,
  }
  return wrapper_type, state_type
end

local function make_create_launcher(task, launcher_name,
                                    wrapper_type, state_type,
                                    params, params_struct_type)
  local helper_name = launcher_name .. "_create"
  local terra helper(pred : c.legion_predicate_t,
                     id : c.legion_mapper_id_t,
                     tag : c.legion_mapping_tag_id_t)
    -- Important: use calloc to ensure the param map is zeroed.
    var params_struct = [&params_struct_type](
      c.calloc(1, [terralib.sizeof(params_struct_type)]))
    base.assert(params_struct ~= nil, ["calloc failed in " .. helper_name])

    var task_args : c.legion_task_argument_t
    task_args.args = params_struct
    task_args.arglen = [terralib.sizeof(params_struct_type)]

    var launcher = c.legion_task_launcher_create(
      [task:get_task_id()], task_args, pred, id, tag)

    var launcher_state = [&state_type](
      c.malloc([terralib.sizeof(state_type)]))
    base.assert(launcher_state ~= nil, ["malloc failed in " .. helper_name])

    -- Important: use calloc to ensure provided map is zeroed.
    var args_provided = [&int8](c.calloc([#params], 1))
    base.assert(args_provided ~= nil, ["calloc failed in " .. helper_name])

    launcher_state.launcher = launcher
    launcher_state.task_args = params_struct
    launcher_state.args_provided = args_provided

    var wrapper : wrapper_type
    wrapper.impl = [&opaque](launcher_state)
    return wrapper
  end
  helper:setname(helper_name)
  return { helper_name, helper }
end

local function make_destroy_launcher(launcher_name, wrapper_type, state_type)
  local helper_name = launcher_name .. "_destroy"
  local terra helper(wrapper : wrapper_type)
    var launcher_state = [&state_type](wrapper.impl)
    c.legion_task_launcher_destroy(launcher_state.launcher)
    c.free(launcher_state.task_args)
    c.free(launcher_state.args_provided)
    c.free(launcher_state)
  end
  helper:setname(helper_name)
  return { helper_name, helper }
end

local function make_execute_launcher(launcher_name, wrapper_type, state_type, params)
  local helper_name = launcher_name .. "_execute"
  local terra helper(runtime : c.legion_runtime_t, context : c.legion_context_t,
                     wrapper : wrapper_type)
    var launcher_state = [&state_type](wrapper.impl)
    [data.mapi(
       function(i, param_list)
         local param_name = param_list[1][4]
         return quote
           base.assert(
               launcher_state.args_provided[i - 1] == 1,
               ["parameter " .. param_name .. " was not supplied in " .. launcher_name])
         end
       end,
       params)]
    var future = c.legion_task_launcher_execute(
      runtime, context, launcher_state.launcher)
    return future
  end
  helper:setname(helper_name)
  return { helper_name, helper }
end

local function make_add_argument(launcher_name, wrapper_type, state_type,
                                 task, params_struct_type,
                                 param_i, param_list, task_param_symbol,
                                 param_field_id)
  local param_name = header_helper.normalize_name(param_list[1][4])

  local param_symbol = param_list:map(
    function(param)
      local terra_type, c_type, cxx_type, param_name = unpack(param)
      return terralib.newsymbol(terra_type, param_name)
    end)

  local param_type = task_param_symbol:gettype()
  if not base.types.is_region(param_type) then
    assert(#param_symbol == 1)
  end

  local launcher_state = terralib.newsymbol(&state_type, "launcher_state")
  local arg_setup = terralib.newlist()

  -- Pack the primary value
  local arg_value = param_symbol[1]

  -- Handle conversions from C to Regent types
  local cast_value = arg_value
  if base.types.is_region(param_type) then
    cast_value = `([param_type] { impl = [arg_value] })
  elseif base.types.is_ispace(param_type) then
    cast_value = `([param_type] { impl = [arg_value] })
  end

  local c_field = params_struct_type:getentries()[param_i + 1]
  local c_field_name = c_field[1] or c_field.field
  if terralib.issymbol(c_field_name) then
    c_field_name = c_field_name.displayname
  end
  arg_setup:insert(
    quote
      [launcher_state].task_args.[c_field_name] = [cast_value]
    end)

  -- Pack secondary values
  if base.types.is_region(param_type) then
    -- Pack region fields
    local parent_region = param_symbol[2]
    local field_ids = param_symbol[3]
    local field_count = param_symbol[4]

    local field_paths, _ = base.types.flatten_struct_fields(param_type:fspace())
    arg_setup:insert(
      quote
          base.assert([field_count] == [#field_paths],
            ["wrong number of fields for region " .. tostring(arg_value) .. " (argument " .. param_i .. ")"])
      end)
    local field_id_by_path = data.newmap()
    for j, field_path in pairs(field_paths) do
      local arg_field_id = `([field_ids][ [j-1] ])
      local param_field_id = param_field_id[j]
      field_id_by_path[field_path] = arg_field_id
      arg_setup:insert(
        quote [launcher_state].task_args.[param_field_id] = [arg_field_id] end)
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
          `([launcher_state].launcher), arg_value})
      if reduction_op then
        requirement_args:insert(reduction_op)
      else
        requirement_args:insert(privilege_mode)
      end
      requirement_args:insertall(
        {coherence_mode, parent_region, 0, false})

      arg_setup:insert(
        quote
          var [requirement] = [add_requirement]([requirement_args])
          [field_paths:map(
             function(field_path)
               local field_id = field_id_by_path[field_path]
               return quote
                 add_field(
                   [launcher_state].launcher, [requirement], [field_id], true)
               end
             end)]
          [add_flags]([launcher_state].launcher, [requirement], [flag])
        end)
    end
  end

  local helper_name = launcher_name .. "_add_argument_" .. param_name
  local terra helper(wrapper : wrapper_type, [param_symbol], overwrite : bool)
    var [launcher_state] = [&state_type](wrapper.impl)

    -- FIXME: Provide support for overwriting region arguments
    if overwrite and [base.types.is_region(param_type)] then
      base.assert(
        false,
        ["overwriting a region argument is not currently suppported in " .. launcher_name])
    end

    if not overwrite then
      base.assert(
        [launcher_state].args_provided[param_i - 1] == 0,
        ["parameter " .. param_name .. " was already supplied in " .. launcher_name])
    end

    [launcher_state].args_provided[param_i - 1] = 1

    [arg_setup]
  end
  helper:setname(helper_name)
  return { helper_name, helper }
end

function header_helper.generate_task_implementation(task)
  local launcher_name = get_launcher_name(task)
  local params = get_task_params(task)

  local params_struct_type = task:get_params_struct()

  local wrapper_type, state_type = make_internal_launcher_state_types(
    params_struct_type)

  local create = make_create_launcher(
    task, launcher_name, wrapper_type, state_type, params, params_struct_type)
  local destroy = make_destroy_launcher(
    launcher_name, wrapper_type, state_type)
  local execute = make_execute_launcher(
    launcher_name, wrapper_type, state_type, params)
  local result = terralib.newlist({create, destroy, execute})

  local task_param_symbols = task:get_param_symbols()
  local param_field_ids = task:get_field_id_param_labels()
  for i, param_list in ipairs(params) do
    local task_param_symbol = task_param_symbols[i]
    local param_field_id = param_field_ids[i]
    result:insert(
      make_add_argument(
        launcher_name, wrapper_type, state_type,
        task, params_struct_type,
        i, param_list, task_param_symbol, param_field_id))
  end

  return result
end

local function OLD_generate_task_implementation(task)
  local name = get_helper_name(task)
  local params = get_task_params(task)

  local param_symbols = params:map(
    function(param_list)
      return param_list:map(
        function(param)
          local terra_type, c_type, cxx_type, param_name = unpack(param)
          return terralib.newsymbol(terra_type, param_name)
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
