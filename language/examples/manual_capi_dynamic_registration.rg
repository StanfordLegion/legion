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

function legion_task_wrapper(body)
  -- look at the return type of the task we're wrapping to emit the right postamble code
  local ft = body:gettype(false)
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

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  c.printf("in top_level_task...\n")

  c.legion_runtime_register_task_variant_fnptr(
    runtime,
    TID_SUB_TASK,
    c.LOC_PROC,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "sub_task",
    [&opaque](0), 0, 
    [ legion_task_wrapper(sub_task) ])

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
  assert(rv == 43)
end

terra main()
  c.printf("in main...\n")
  c.legion_runtime_preregister_task_variant_fnptr(
    TID_TOP_LEVEL_TASK,
    c.LOC_PROC,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "top_level_task",
    [&opaque](0), 0, 
    [ legion_task_wrapper(top_level_task) ])
  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  c.legion_runtime_start(0, [&rawstring](0), false)
end
main()
