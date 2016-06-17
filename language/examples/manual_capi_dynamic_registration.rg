-- Copyright 2016 Stanford University
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

-- Note: The binding library is only being used to load the dynamic
-- library for Legion, not for any actual functionality. All Legion
-- calls happen through the C API.
require('legionlib')
c = terralib.includecstring([[
#include "legion_c.h"
#include <stdio.h>
#include <stdlib.h>
]])

FID_1 = 1
FID_2 = 2

TID_TOP_LEVEL_TASK = 100
TID_SUB_TASK = 101

---------------------------------------------------------------------
--
-- this is task registration code that could/should be in a library
--
---------------------------------------------------------------------

local dlfcn = terralib.includec("dlfcn.h")
terra legion_has_llvm_support() : bool
  return (dlfcn.dlsym([&opaque](0), "legion_runtime_register_task_variant_llvmir") ~= [&opaque](0))
end
use_llvm = legion_has_llvm_support()

function legion_task_wrapper(body)
  -- look at the return type of the task we're wrapping to emit the right postamble code
  local ft = body:gettype()
  local rt = ft.returntype
  local wrapper = nil
  if terralib.sizeof(rt) > 0 then
    wrapper = terra(data : &opaque, datalen : c.size_t, userdata : &opaque, userlen : c.size_t, proc_id : c.legion_lowlevel_id_t)
      var task : c.legion_task_t,
          regions : &c.legion_physical_region_t,
          num_regions : uint32,
          ctx : c.legion_context_t,
          runtime : c.legion_runtime_t
      c.legion_task_preamble(data, datalen, proc_id, &task, &regions, &num_regions, &ctx, &runtime)
      var rv : rt = body(task, regions, num_regions, ctx, runtime)
      c.legion_task_postamble(runtime, ctx, [&opaque](&rv), terralib.sizeof(rt))
    end
  else
    wrapper = terra(data : &opaque, datalen : c.size_t, userdata : &opaque, userlen : c.size_t, proc_id : c.legion_lowlevel_id_t)
      var task : c.legion_task_t,
          regions : &c.legion_physical_region_t,
          num_regions : uint32,
          ctx : c.legion_context_t,
          runtime : c.legion_runtime_t
      c.legion_task_preamble(data, datalen, proc_id, &task, &regions, &num_regions, &ctx, &runtime)
      body(task, regions, num_regions, ctx, runtime)
      c.legion_task_postamble(runtime, ctx, [&opaque](0), 0)
    end
  end
  return wrapper
end

function preregister_task(terrafunc)
  -- either way, we wrap the body with legion preamble and postamble first
  local wrapped = legion_task_wrapper(terrafunc)
  if use_llvm then
    -- if we can register llvmir, ask Terra to generate that
    local ir = terralib.saveobj(nil, "llvmir", { entry=wrapped } )
    local rfunc = terra(id : c.legion_task_id_t,
                        task_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_preregister_task_variant_llvmir(
        id, task_name,
        execution_constraints, layout_constraints, options,
        ir, "entry", userdata, userlen)
    end
    return rfunc
  else
    -- use the terra function directly, which ffi will convert to a (non-portable) function pointer
    local rfunc = terra(id : c.legion_task_id_t,
                        task_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_preregister_task_variant_fnptr(
        id, task_name,
        execution_constraints, layout_constraints, options,
        wrapped, userdata, userlen)
    end
    return rfunc
  end
end

function register_task(terrafunc)
  -- either way, we wrap the body with legion preamble and postamble first
  local wrapped = legion_task_wrapper(terrafunc)
  if use_llvm then
    -- if we can register llvmir, ask Terra to generate that
    local ir = terralib.saveobj(nil, "llvmir", { entry=wrapped } )
    local rfunc = terra(runtime : c.legion_runtime_t,
                        id : c.legion_task_id_t,
                        task_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_register_task_variant_llvmir(
        runtime, id, task_name,
        true, -- global registration possible with llvmir
        execution_constraints, layout_constraints, options,
        ir, "entry", userdata, userlen)
    end
    return rfunc
  else
    -- use the terra function directly, which ffi will convert to a (non-portable) function pointer
    local rfunc = terra(runtime : c.legion_runtime_t,
                        id : c.legion_task_id_t,
                        task_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_register_task_variant_fnptr(
        runtime, id, task_name,
        false, -- global registration not possible with non-portable pointer
        execution_constraints, layout_constraints, options,
        wrapped, userdata, userlen)
    end
    return rfunc
  end
end

---------------------------------------------------------------------
--
-- actual application tasks start here
--
---------------------------------------------------------------------


terra sub_task(task : c.legion_task_t,
               regions : &c.legion_physical_region_t,
               num_regions : uint32,
               ctx : c.legion_context_t,
               runtime : c.legion_runtime_t) : uint32
  var arglen = c.legion_task_get_arglen(task)
  c.printf("in sub_task (%u arglen, %u regions)...\n",
                arglen, num_regions)
  var y = @[&uint32](c.legion_task_get_args(task))
  return y + 1
end

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  c.printf("in top_level_task...\n")

  -- sub task is dynamically registered in the top level task
  -- the Lua escape here constructs the right registration function, picking between a non-portable function
  --  pointer and LLVMIR during the compilation of this task, but the actual registration happens when the
  --  top_level_task is executed
  var execution_constraints = c.legion_execution_constraint_set_create()
  c.legion_execution_constraint_set_add_processor_constraint(execution_constraints, c.LOC_PROC)
  var layout_constraints = c.legion_task_layout_constraint_set_create()
  [ register_task(sub_task) ](
    runtime, TID_SUB_TASK, "sub_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)

  var x : uint32 = 42
  var sub_args = c.legion_task_argument_t {
     args = [&opaque](&x),
     arglen = terralib.sizeof(uint32)
  }
  var launcher = c.legion_task_launcher_create(
    TID_SUB_TASK, sub_args, c.legion_predicate_true(), 0, 0)

  var f = c.legion_task_launcher_execute(runtime, ctx, launcher)
  var rv : uint32 = 99
  c.legion_future_get_result_bytes(f, [&opaque](&rv), terralib.sizeof(uint32))
  c.printf("back in parent (rv = %d)\n", rv)
  if rv ~= 43 then
    c.printf("abort\n")
    c.abort()
  end
end

local args = require("manual_capi_args")

terra main()
  c.printf("in main...\n")

  -- top level task must be "preregistered" (i.e. before we start the runtime)
  var execution_constraints = c.legion_execution_constraint_set_create()
  c.legion_execution_constraint_set_add_processor_constraint(execution_constraints, c.LOC_PROC)
  var layout_constraints = c.legion_task_layout_constraint_set_create()
  [ preregister_task(top_level_task) ](
    TID_TOP_LEVEL_TASK,
    "top_level_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)

  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end

if use_llvm then
  print("LLVM support detected...  tasks will be registered as LLVM IR")
else
  print("LLVM support NOT detected...  tasks will be registered as function pointers")
end
main()
