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

FID_1 = 1

TID_TOP_LEVEL_TASK = 100
TID_SUB_TASK = 101

terra sub_task(task : c.legion_task_t,
               regions : &c.legion_physical_region_t,
               num_regions : uint32,
               ctx : c.legion_context_t,
               runtime : c.legion_runtime_t)
  var r = c.legion_physical_region_get_logical_region(regions[0])
  var is = r.index_space
  var d = c.legion_index_space_get_domain(runtime, is)
  var rect = c.legion_domain_get_rect_1d(d)

  var a1 = c.legion_physical_region_get_field_accessor_array_1d(
    regions[0], FID_1)

  -- This code uses the raw rect ptr API.
  do
    var subrect : c.legion_rect_1d_t
    var offsets : c.legion_byte_offset_t[1]

    var base = [&int8](c.legion_accessor_array_1d_raw_rect_ptr(
                         a1, rect, &subrect, &(offsets[0])))
    if base == nil then
      c.printf("Error: failed to get base ptr\n")
      c.abort()
    end
    if rect.lo.x[0] ~= subrect.lo.x[0] or rect.hi.x[0] ~= subrect.hi.x[0] then
      c.printf("Error: subrect doesn't match rect bounds\n")
      c.abort()
    end

    c.printf("got raw rect pointer:\n")
    c.printf("  rect is lo %d, hi %d\n", rect.lo.x[0], rect.hi.x[0])
    c.printf("  subrect is lo %d, hi %d\n", subrect.lo.x[0], subrect.hi.x[0])
    c.printf("  offset is %d bytes\n", offsets[0].offset)
    c.printf("  base %p\n", base)

    for global_i0 = rect.lo.x[0], rect.hi.x[0]+1 do
      var local_i0 = global_i0 - rect.lo.x[0]
      var ptr = [&int](base + offsets[0].offset*local_i0)
      c.printf("writing value %2d at global %2d local %2d pointer %p\n",
               global_i0, global_i0, local_i0, ptr)
      @ptr = global_i0
    end
  end

  -- Sanity check with generic API.
  for global_i0 = rect.lo.x[0], rect.hi.x[0]+1 do
    var p = c.legion_point_1d_t { x = arrayof(int64, global_i0) }
    -- var p = c.legion_domain_point_t { dim = 1, point_data = arrayof(int64, global_i0, 0, 0) }
    var value : int
    c.legion_accessor_array_1d_read_point(a1, p, &value, sizeof(int))
    c.printf("read value %2d at global %2d\n", value, global_i0)
    if value ~= global_i0 then
      c.abort()
    end
  end

  c.legion_accessor_array_1d_destroy(a1)
end

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  c.printf("in top_level_task...\n")

  var d = c.legion_domain_from_rect_1d(
    c.legion_rect_1d_t {
      lo = c.legion_point_1d_t { x = arrayof(int64, 0) },
      hi = c.legion_point_1d_t { x = arrayof(int64, 15) },
    })
  var is = c.legion_index_space_create_domain(runtime, ctx, d)
  var fs = c.legion_field_space_create(runtime, ctx)
  var r = c.legion_logical_region_create(runtime, ctx, is, fs, true)

  do
    var fsa = c.legion_field_allocator_create(runtime, ctx, fs)
    c.legion_field_allocator_allocate_field(fsa, sizeof(int), FID_1)
    c.legion_field_allocator_destroy(fsa)
  end

  var sub_args = c.legion_task_argument_t { args = nil, arglen = 0 }
  var launcher = c.legion_task_launcher_create(
    TID_SUB_TASK, sub_args, c.legion_predicate_true(), 0, 0)
  var rr1 = c.legion_task_launcher_add_region_requirement_logical_region(
    launcher, r, c.READ_WRITE, c.EXCLUSIVE, r, 0, false)
  c.legion_task_launcher_add_field(
    launcher, rr1, FID_1, true --[[ inst ]])

  var f = c.legion_task_launcher_execute(runtime, ctx, launcher)
  c.legion_future_destroy(f)

  c.legion_logical_region_destroy(runtime, ctx, r)
  c.legion_field_space_destroy(runtime, ctx, fs)
  c.legion_index_space_destroy(runtime, ctx, is)
end

local args = require("manual_capi_args")

terra main()
  c.printf("in main...\n")

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
  [ tasklib.preregister_task(sub_task) ](
    TID_SUB_TASK,
    "sub_task",
    execution_constraints, layout_constraints,
    c.legion_task_config_options_t {
      leaf = false, inner = false, idempotent = false},
    nil, 0)
  c.legion_runtime_set_top_level_task_id(TID_TOP_LEVEL_TASK)
  [args.argv_setup]
  c.legion_runtime_start(args.argc, args.argv, false)
end
main()
