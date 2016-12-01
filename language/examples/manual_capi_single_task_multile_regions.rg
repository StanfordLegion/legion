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

-- This tests for the ability of a task to take different numbers of
-- regions. Specifically, the default mapper had at one point made
-- assumptions (during memoization) about the number of region
-- requirements staying the same on every invocation of a task.

require("legionlib")
local C = terralib.includecstring([[
#include "stdio.h"
#include "stdlib.h"
]])
local Lg  = terralib.includecstring([[
#include "legion_c.h"
]])

TOP_TASK_ID = 1
FAKE_TASK_ID = 2

F1R1 = 0
F2R1 = 1
F1R2 = 2
F2R2 = 3

terra fake_task(task : Lg.legion_task_t,
                regions : &Lg.legion_physical_region_t,
                num_regions : uint32,
                ctx : Lg.legion_context_t,
                runtime : Lg.legion_runtime_t)
  C.printf("Fake task\n")
end

terra top_level_task(task : Lg.legion_task_t,
                     regions : &Lg.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : Lg.legion_context_t,
                     runtime : Lg.legion_runtime_t)
  C.printf("Start top task!\n")

  C.printf("Define index space ...\n")

  var is1 = Lg.legion_index_space_create(runtime, ctx, 5)
  var fs1 = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(
      runtime, ctx, fs1)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F1R1)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F2R1)
    Lg.legion_field_allocator_destroy(allocator)
  end
  var lr1 =
    Lg.legion_logical_region_create(runtime, ctx, is1, fs1)

  var is2 = Lg.legion_index_space_create(runtime, ctx, 5)
  var fs2 = Lg.legion_field_space_create(runtime, ctx)
  do
    var allocator = Lg.legion_field_allocator_create(
      runtime, ctx, fs2)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F1R2)
    Lg.legion_field_allocator_allocate_field(
      allocator, sizeof(uint64), F2R2)
    Lg.legion_field_allocator_destroy(allocator)
  end
  var lr2 =
    Lg.legion_logical_region_create(runtime, ctx, is2, fs2)

  var arg_nil = Lg.legion_task_argument_t { args = nil, arglen = 0 }

  do
    var task_launcher = Lg.legion_task_launcher_create(FAKE_TASK_ID, arg_nil,
                                                       Lg.legion_predicate_true(),
                                                       0, 0)
    var r1idx = Lg.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr1, Lg.READ_WRITE, Lg.EXCLUSIVE, lr1, 0, false)
    Lg.legion_task_launcher_add_field(task_launcher, r1idx, F1R1, true)
    Lg.legion_task_launcher_execute(runtime, ctx, task_launcher)
    C.printf("Launched task with single regions\n")
  end

  do
    var task_launcher = Lg.legion_task_launcher_create(FAKE_TASK_ID, arg_nil,
                                                       Lg.legion_predicate_true(),
                                                       0, 0)
    var r1idx = Lg.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr1, Lg.READ_WRITE, Lg.EXCLUSIVE, lr1, 0, false)
    Lg.legion_task_launcher_add_field(task_launcher, r1idx, F1R1, true)
    var r2idx = Lg.legion_task_launcher_add_region_requirement_logical_region(
      task_launcher, lr2, Lg.READ_WRITE, Lg.EXCLUSIVE, lr2, 0, false)
    Lg.legion_task_launcher_add_field(task_launcher, r2idx, F1R2, true)
    Lg.legion_task_launcher_execute(runtime, ctx, task_launcher)
    C.printf("Launched task with multiple regions\n")
  end

  C.printf("End top task!\n")
end

local args = require("manual_capi_args")

terra legion_main()
  Lg.legion_runtime_register_task_void(
    TOP_TASK_ID,
    Lg.LOC_PROC,
    true,
    false,
    1,
    Lg.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "top_level_task",
    top_level_task)
  Lg.legion_runtime_register_task_void(
    FAKE_TASK_ID,
    Lg.LOC_PROC,
    true,
    false,
    1,
    Lg.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "fake_task",
    fake_task)
  Lg.legion_runtime_set_top_level_task_id(TOP_TASK_ID)
  [args.argv_setup]
  Lg.legion_runtime_start(args.argc, args.argv, false)
end

legion_main()
