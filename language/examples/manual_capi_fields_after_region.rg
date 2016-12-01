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

-- This is an example of a failure in the runtime when allocating
-- fields after the region itself has been created.
local broken = false

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

  if broken then
    var il = c.legion_inline_launcher_create_logical_region(
      r, c.READ_WRITE, c.EXCLUSIVE, r, 0, false, 0, 0)
    c.legion_inline_launcher_add_field(il, f1, true)
    c.legion_inline_launcher_add_field(il, f2, true)
    var pr = c.legion_inline_launcher_execute(runtime, ctx, il)

    c.legion_inline_launcher_destroy(il)
  end

  c.legion_logical_region_destroy(runtime, ctx, r)
  c.legion_field_space_destroy(runtime, ctx, fs)
  c.legion_index_space_destroy(runtime, ctx, is)
end

local args = require("manual_capi_args")

terra main()
  c.printf("in main...\n")
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
  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end
main()
