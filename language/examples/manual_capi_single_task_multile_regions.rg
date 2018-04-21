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

local tasklib = require("manual_capi_tasklib")
local c = tasklib.c

TID_TOP_LEVEL_TASK = 1
TID_FAKE_TASK = 2

F1R1 = 0
F2R1 = 1
F1R2 = 2
F2R2 = 3

terra fake_task(task : c.legion_task_t,
                regions : &c.legion_physical_region_t,
                num_regions : uint32,
                ctx : c.legion_context_t,
                runtime : c.legion_runtime_t)
  c.printf("Fake task\n")
end

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  c.printf("Start top task!\n")

  c.printf("Define index space ...\n")

  var is1 = c.legion_index_space_create(runtime, ctx, 5)
  var fs1 = c.legion_field_space_create(runtime, ctx)
  do
    var allocator = c.legion_field_allocator_create(
      runtime, ctx, fs1)
    c.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F1R1)
    c.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F2R1)
    c.legion_field_allocator_destroy(allocator)
  end
  var lr1 =
    c.legion_logical_region_create(runtime, ctx, is1, fs1, true)

  var is2 = c.legion_index_space_create(runtime, ctx, 5)
  var fs2 = c.legion_field_space_create(runtime, ctx)
  do
    var allocator = c.legion_field_allocator_create(
      runtime, ctx, fs2)
    c.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F1R2)
    c.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F2R2)
    c.legion_field_allocator_destroy(allocator)
  end
  var lr2 =
    c.legion_logical_region_create(runtime, ctx, is2, fs2, true)

  var arg_nil = c.legion_task_argument_t { args = nil, arglen = 0 }

  do
    var task_launcher = c.legion_task_launcher_create(TID_FAKE_TASK, arg_nil,
                                                       c.legion_predicate_true(),
                                                       0, 0)
    var r1idx = c.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr1, c.READ_WRITE, c.EXCLUSIVE, lr1, 0, false)
    c.legion_task_launcher_add_field(task_launcher, r1idx, F1R1, true)
    c.legion_task_launcher_execute(runtime, ctx, task_launcher)
    c.printf("Launched task with single regions\n")
  end

  do
    var task_launcher = c.legion_task_launcher_create(TID_FAKE_TASK, arg_nil,
                                                       c.legion_predicate_true(),
                                                       0, 0)
    var r1idx = c.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr1, c.READ_WRITE, c.EXCLUSIVE, lr1, 0, false)
    c.legion_task_launcher_add_field(task_launcher, r1idx, F1R1, true)
    var r2idx = c.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr2, c.READ_WRITE, c.EXCLUSIVE, lr2, 0, false)
    c.legion_task_launcher_add_field(task_launcher, r2idx, F1R2, true)
    c.legion_task_launcher_execute(runtime, ctx, task_launcher)
    c.printf("Launched task with multiple regions\n")
  end

  c.printf("End top task!\n")
end

local args = require("manual_capi_args")

terra legion_main()
  var execution_constraints = c.legion_execution_constraint_set_create()
  c.legion_execution_constraint_set_add_processor_constraint(execution_constraints, c.LOC_PROC)
  var layout_constraints = c.legion_task_layout_constraint_set_create()
  [ tasklib.preregister_task(top_level_task) ](
    TID_TOP_LEVEL_TASK,
    "top_level_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)
  [ tasklib.preregister_task(fake_task) ](
    TID_FAKE_TASK,
    "fake_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)
  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end

legion_main()
