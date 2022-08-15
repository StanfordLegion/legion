-- Copyright 2022 Stanford University
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

-- This file is not meant to be run directly.

-- runs-with:
-- []

local tasklib = {}

local ffi = require("ffi")
local dlfcn = terralib.includec("dlfcn.h")
local function dlopen_library(library_name)
  -- Right now we do this globally and do not attempt to unload
  -- libraries (and really, there is no safe way to do so because
  -- LuaJIT and LLVM will both get unloaded before we're ready)
  local ok = dlfcn.dlopen(library_name, bit.bor(dlfcn.RTLD_LAZY, dlfcn.RTLD_GLOBAL))
  if ffi.cast("intptr_t", ok) == 0LL then
    assert(false, "dlopen failed: " .. tostring(dlfcn.dlerror()))
  end
end

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  dlopen_library("libregent.dylib")
else
  dlopen_library("libregent.so")
end

local c = terralib.includecstring([[
#include "legion.h"
#include "regent.h"
#include <stdio.h>
#include <stdlib.h>
]])
tasklib.c = c

local terra legion_has_llvm_support() : bool
  return (dlfcn.dlsym([&opaque](0), "legion_runtime_register_task_variant_llvmir") ~= [&opaque](0))
end
local use_llvm = legion_has_llvm_support()

local function legion_task_wrapper(body)
  -- look at the return type of the task we're wrapping to emit the right postamble code
  local ft = body:gettype()
  local rt = ft.returntype
  local wrapper = nil
  if terralib.sizeof(rt) > 0 then
    wrapper = terra(data : &opaque, datalen : c.size_t, userdata : &opaque, userlen : c.size_t, proc_id : c.legion_proc_id_t)
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
    wrapper = terra(data : &opaque, datalen : c.size_t, userdata : &opaque, userlen : c.size_t, proc_id : c.legion_proc_id_t)
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

function tasklib.preregister_task(terrafunc)
  -- either way, we wrap the body with legion preamble and postamble first
  local wrapped = legion_task_wrapper(terrafunc)
  if use_llvm then
    -- if we can register llvmir, ask Terra to generate that
    local ir = terralib.saveobj(nil, "llvmir", { entry=wrapped } )
    local rfunc = terra(id : c.legion_task_id_t,
                        variant_id : c.legion_variant_id_t,
                        task_name : &int8,
                        variant_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_preregister_task_variant_llvmir(
        id, variant_id, task_name,
        execution_constraints, layout_constraints, options,
        ir, "entry", userdata, userlen)
    end
    return rfunc
  else
    -- use the terra function directly, which ffi will convert to a (non-portable) function pointer
    local rfunc = terra(id : c.legion_task_id_t,
                        variant_id : c.legion_variant_id_t,
                        task_name : &int8,
                        variant_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_preregister_task_variant_fnptr(
        id, variant_id, task_name, variant_name,
        execution_constraints, layout_constraints, options,
        wrapped, userdata, userlen)
    end
    return rfunc
  end
end

function tasklib.register_task(terrafunc)
  -- either way, we wrap the body with legion preamble and postamble first
  local wrapped = legion_task_wrapper(terrafunc)
  if use_llvm then
    -- if we can register llvmir, ask Terra to generate that
    local ir = terralib.saveobj(nil, "llvmir", { entry=wrapped } )
    local rfunc = terra(runtime : c.legion_runtime_t,
                        id : c.legion_task_id_t,
                        task_name : &int8,
                        variant_name : &int8,
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
                        variant_name : &int8,
                        execution_constraints : c.legion_execution_constraint_set_t,
                        layout_constraints : c.legion_task_layout_constraint_set_t,
                        options: c.legion_task_config_options_t,
                        userdata : &opaque,
                        userlen : c.size_t)
      return c.legion_runtime_register_task_variant_fnptr(
        runtime, id, task_name, variant_name,
        false, -- global registration not possible with non-portable pointer
        execution_constraints, layout_constraints, options,
        wrapped, userdata, userlen)
    end
    return rfunc
  end
end

if use_llvm then
  print("LLVM support detected...  tasks will be registered as LLVM IR")
else
  print("LLVM support NOT detected...  tasks will be registered as function pointers")
end

return tasklib
