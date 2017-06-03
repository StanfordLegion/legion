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

-- runs-with:
-- []

-- FIXME: Uses deprecated APIs

-- Note: The binding library is only being used to load the dynamic
-- library for Legion, not for any actual functionality. All Legion
-- calls happen through the C API.
require('legionlib')
c = terralib.includecstring([[
#include "legion_c.h"
#include "legion_terra.h"
#include <stdio.h>
#include <stdlib.h>
]])

RED_PLUS_INT = 1

FID_1 = 1
FID_2 = 2

TID_TOP_LEVEL_TASK = 100
TID_SUB_TASK = 101

terra sub_task(task : c.legion_task_t,
               regions : &c.legion_physical_region_t,
               num_regions : uint32,
               ctx : c.legion_context_t,
               runtime : c.legion_runtime_t)
  var arglen = c.legion_task_get_arglen(task)
  c.printf("in sub_task (%u arglen, %u regions)...\n",
                arglen, num_regions)

  if arglen ~= terralib.sizeof(c.legion_ptr_t) or num_regions ~= 1 then
    c.printf("abort\n")
    c.abort()
  end

  var p1 = @[&c.legion_ptr_t](c.legion_task_get_args(task))
  c.printf("got arg %d\n", p1)

  var a1 = c.legion_physical_region_get_accessor_generic(
    regions[0])

  -- FIMXE: This has privilege problems.
  -- c.reduce_plus_int32(a1, p1, 123)

  c.legion_accessor_generic_destroy(a1)
end

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  c.printf("in top_level_task...\n")

  var is = c.legion_index_space_create(runtime, ctx, 5)
  var fs = c.legion_field_space_create(runtime, ctx)
  var r = c.legion_logical_region_create(runtime, ctx, is, fs)

  var ptr1 : c.legion_ptr_t, ptr2 : c.legion_ptr_t
  var f1 : c.legion_field_id_t, f2 : c.legion_field_id_t
  do
    var isa = c.legion_index_allocator_create(runtime, ctx, is)
    var fsa = c.legion_field_allocator_create(runtime, ctx, fs)

    c.printf("created region (%d,%d,%d)\n",
                  r.tree_id, r.index_space.id, r.field_space.id)

    ptr1 = c.legion_index_allocator_alloc(isa, 1)
    ptr2 = c.legion_index_allocator_alloc(isa, 1)

    c.printf("allocated pointers %d %d\n",
                  ptr1.value, ptr2.value)

    f1 = c.legion_field_allocator_allocate_field(fsa, sizeof(int), FID_1)
    f2 = c.legion_field_allocator_allocate_field(fsa, sizeof(int), FID_2)

    c.printf("allocated fields %d %d\n",
                  f1, f2)

    c.legion_field_allocator_destroy(fsa)
    c.legion_index_allocator_destroy(isa)
  end

  var coloring = c.legion_coloring_create()
  c.legion_coloring_add_point(coloring, 1, ptr1)
  c.legion_coloring_add_point(coloring, 2, ptr1)

  var isp = c.legion_index_partition_create_coloring(
    runtime, ctx, is, coloring, false, -1)
  var lp = c.legion_logical_partition_create(runtime, ctx, r, isp)
  c.legion_coloring_destroy(coloring)

  var r1 = c.legion_logical_partition_get_logical_subregion_by_color(
    runtime, lp, 1)
  var r2 = c.legion_logical_partition_get_logical_subregion_by_color(
    runtime, lp, 2)

  var sub_args_buffer = ptr1
  var sub_args = c.legion_task_argument_t {
    args = [&opaque](&sub_args_buffer),
    arglen = terralib.sizeof(c.legion_ptr_t)
  }
  var launcher = c.legion_task_launcher_create(
    TID_SUB_TASK, sub_args, c.legion_predicate_true(), 0, 0)
  var rr1 = c.legion_task_launcher_add_region_requirement_logical_region_reduction(
    launcher, r1, RED_PLUS_INT, c.EXCLUSIVE, r, 0, false)
  c.legion_task_launcher_add_field(
    launcher, rr1, f1, true --[[ inst ]])

  var f = c.legion_task_launcher_execute(runtime, ctx, launcher)
  c.legion_future_destroy(f)

  c.legion_logical_partition_destroy(runtime, ctx, lp)
  c.legion_index_partition_destroy(runtime, ctx, isp)
  c.legion_logical_region_destroy(runtime, ctx, r)
  c.legion_field_space_destroy(runtime, ctx, fs)
  c.legion_index_space_destroy(runtime, ctx, is)
end

local args = require("manual_capi_args")

terra main()
  c.printf("in main...\n")
  c.register_reduction_plus_int32(RED_PLUS_INT)
  c.legion_runtime_register_task_void(
    TID_TOP_LEVEL_TASK,
    c.LOC_PROC,
    true,
    false,
    1, -- c.AUTO_GENERATE_ID,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "top_level_task",
    top_level_task)
  c.legion_runtime_register_task_void(
    TID_SUB_TASK,
    c.LOC_PROC,
    true,
    false,
    1, -- c.AUTO_GENERATE_ID,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    "sub_task",
    sub_task)
  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end
main()
