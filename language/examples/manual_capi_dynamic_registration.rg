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

local tasklib = require("manual_capi_tasklib")
local c = tasklib.c

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
  [ tasklib.register_task(sub_task) ](
    runtime, TID_SUB_TASK, "sub_task", "sub_task",
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
  var rv : uint32 = @[&uint32](c.legion_future_get_untyped_pointer(f))
  c.printf("back in parent (rv = %d)\n", rv)
  if rv ~= 43 then
    c.printf("abort\n")
    c.abort()
  end
  c.legion_future_destroy(f)
end

local args = require("manual_capi_args")

terra main()
  c.printf("in main...\n")

  -- top level task must be "preregistered" (i.e. before we start the runtime)
  var execution_constraints = c.legion_execution_constraint_set_create()
  c.legion_execution_constraint_set_add_processor_constraint(execution_constraints, c.LOC_PROC)
  var layout_constraints = c.legion_task_layout_constraint_set_create()
  [ tasklib.preregister_task(top_level_task) ](
    TID_TOP_LEVEL_TASK,
    -1, -- AUTO_GENERATE_ID
    "top_level_task", "top_level_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)

  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end
main()
