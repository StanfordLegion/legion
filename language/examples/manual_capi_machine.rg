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

TID_TOP_LEVEL_TASK = 100

terra top_level_task(task : c.legion_task_t,
                     regions : &c.legion_physical_region_t,
                     num_regions : uint32,
                     ctx : c.legion_context_t,
                     runtime : c.legion_runtime_t)
  var machine = c.legion_machine_create()

  -- Query processors
  var all_procs = c.legion_processor_query_create(machine)
  c.printf("\nTotal processors: %lu\n", c.legion_processor_query_count(all_procs))
  c.legion_processor_query_destroy(all_procs)

  var loc_procs = c.legion_processor_query_create(machine)
  c.legion_processor_query_only_kind(loc_procs, c.LOC_PROC)
  c.printf("  %lu CPU(s):\n", c.legion_processor_query_count(loc_procs))
  var loc_proc = c.legion_processor_query_first(loc_procs)
  while loc_proc.id ~= 0 do
    var mem_affinity = c.legion_memory_query_create(machine)
    c.legion_memory_query_has_affinity_to_processor(mem_affinity, loc_proc, 0, 0)
    c.printf("    %lx with affinity to %lu memories:\n", loc_proc.id, c.legion_memory_query_count(mem_affinity))
    var mem = c.legion_memory_query_first(mem_affinity)
    while mem.id ~= 0 do
      c.printf("      %lx\n", mem.id)
      mem = c.legion_memory_query_next(mem_affinity, mem)
    end
    c.legion_memory_query_destroy(mem_affinity)
    loc_proc = c.legion_processor_query_next(loc_procs, loc_proc)
  end
  c.legion_processor_query_destroy(loc_procs)

  var toc_procs = c.legion_processor_query_create(machine)
  c.legion_processor_query_only_kind(toc_procs, c.TOC_PROC)
  c.printf("  %lu GPU(s):\n", c.legion_processor_query_count(toc_procs))
  var toc_proc = c.legion_processor_query_first(toc_procs)
  while toc_proc.id ~= 0 do
    var mem_affinity = c.legion_memory_query_create(machine)
    c.legion_memory_query_has_affinity_to_processor(mem_affinity, toc_proc, 0, 0)
    c.printf("    %lx with affinity to %lu memories:\n", toc_proc.id, c.legion_memory_query_count(mem_affinity))
    var mem = c.legion_memory_query_first(mem_affinity)
    while mem.id ~= 0 do
      c.printf("      %lx\n", mem.id)
      mem = c.legion_memory_query_next(mem_affinity, mem)
    end
    c.legion_memory_query_destroy(mem_affinity)
    toc_proc = c.legion_processor_query_next(toc_procs, toc_proc)
  end
  c.legion_processor_query_destroy(toc_procs)

  var util_procs = c.legion_processor_query_create(machine)
  c.legion_processor_query_only_kind(util_procs, c.UTIL_PROC)
  c.printf("  %lu Utility:\n", c.legion_processor_query_count(util_procs))
  var util_proc = c.legion_processor_query_first(util_procs)
  while util_proc.id ~= 0 do
    var mem_affinity = c.legion_memory_query_create(machine)
    c.legion_memory_query_has_affinity_to_processor(mem_affinity, util_proc, 0, 0)
    c.printf("    %lx with affinity to %lu memories:\n", util_proc.id, c.legion_memory_query_count(mem_affinity))
    var mem = c.legion_memory_query_first(mem_affinity)
    while mem.id ~= 0 do
      c.printf("      %lx\n", mem.id)
      mem = c.legion_memory_query_next(mem_affinity, mem)
    end
    c.legion_memory_query_destroy(mem_affinity)
    util_proc = c.legion_processor_query_next(util_procs, util_proc)
  end
  c.legion_processor_query_destroy(util_procs)

  -- Query memories
  var all_mems = c.legion_memory_query_create(machine)
  c.printf("\nTotal memories: %lu\n", c.legion_memory_query_count(all_mems))
  c.legion_memory_query_destroy(all_mems)

  var global_mems = c.legion_memory_query_create(machine)
  c.legion_memory_query_only_kind(global_mems, c.GLOBAL_MEM)
  c.printf("  %lu Global:\n", c.legion_memory_query_count(global_mems))
  var global_mem = c.legion_memory_query_first(global_mems)
  while global_mem.id ~= 0 do
    c.printf("    %lx\n", global_mem.id)
    global_mem = c.legion_memory_query_next(global_mems, global_mem)
  end
  c.legion_memory_query_destroy(global_mems)

  var sys_mems = c.legion_memory_query_create(machine)
  c.legion_memory_query_only_kind(sys_mems, c.SYSTEM_MEM)
  c.printf("  %lu System:\n", c.legion_memory_query_count(sys_mems))
  var sys_mem = c.legion_memory_query_first(sys_mems)
  while sys_mem.id ~= 0 do
    var proc_affinity = c.legion_processor_query_create(machine)
    c.legion_processor_query_best_affinity_to_memory(proc_affinity, sys_mem, 0, 0)
    c.printf("    %lx with affinity to %lu memories:\n", sys_mem.id, c.legion_processor_query_count(proc_affinity))
    var proc = c.legion_processor_query_first(proc_affinity)
    while proc.id ~= 0 do
      c.printf("      %lx\n", proc.id)
      proc = c.legion_processor_query_next(proc_affinity, proc)
    end
    c.legion_processor_query_destroy(proc_affinity)
    sys_mem = c.legion_memory_query_next(sys_mems, sys_mem)
  end
  c.legion_memory_query_destroy(sys_mems)

  var reg_mems = c.legion_memory_query_create(machine)
  c.legion_memory_query_only_kind(reg_mems, c.REGDMA_MEM)
  c.printf("  %lu Registered:\n", c.legion_memory_query_count(reg_mems))
  var reg_mem = c.legion_memory_query_first(reg_mems)
  while reg_mem.id ~= 0 do
    c.printf("    %lx\n", reg_mem.id)
    reg_mem = c.legion_memory_query_next(reg_mems, reg_mem)
  end
  c.legion_memory_query_destroy(reg_mems)

  var zcopy_mems = c.legion_memory_query_create(machine)
  c.legion_memory_query_only_kind(zcopy_mems, c.Z_COPY_MEM)
  c.printf("  %lu Zero Copy:\n", c.legion_memory_query_count(zcopy_mems))
  var zcopy_mem = c.legion_memory_query_first(zcopy_mems)
  while zcopy_mem.id ~= 0 do
    c.printf("    %lx\n", zcopy_mem.id)
    zcopy_mem = c.legion_memory_query_next(zcopy_mems, zcopy_mem)
  end
  c.legion_memory_query_destroy(zcopy_mems)

  var fb_mems = c.legion_memory_query_create(machine)
  c.legion_memory_query_only_kind(fb_mems, c.GPU_FB_MEM)
  c.printf("  %lu Frame Buffer:\n", c.legion_memory_query_count(fb_mems))
  var fb_mem = c.legion_memory_query_first(fb_mems)
  while fb_mem.id ~= 0 do
    c.printf("    %lx\n", fb_mem.id)
    fb_mem = c.legion_memory_query_next(fb_mems, fb_mem)
  end
  c.legion_memory_query_destroy(fb_mems)

  c.legion_machine_destroy(machine)
end

local args = require("manual_capi_args")

terra main()
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
