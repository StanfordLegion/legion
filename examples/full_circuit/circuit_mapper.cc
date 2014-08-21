/* Copyright 2014 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "circuit_mapper.h"

using namespace LegionRuntime::HighLevel;

LegionRuntime::Logger::Category log_mapper("mapper");

CircuitMapper::CircuitMapper(Machine *m, HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p)
{
  const std::set<Processor> &all_procs = machine->get_all_processors();
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    Processor::Kind k = machine->get_processor_kind(*it);
    switch (k)
    {
      case Processor::LOC_PROC:
        all_cpus.push_back(*it);
        break;
      case Processor::TOC_PROC:
        all_gpus.push_back(*it);
        break;
      default:
        break;
    }
  }
  map_to_gpus = !all_gpus.empty();
}

void CircuitMapper::slice_domain(const Task *task, const Domain &domain,
                                 std::vector<DomainSplit> &slices)
{
  if (map_to_gpus && (task->task_id != CHECK_FIELD_TASK_ID))
  {
    decompose_index_space(domain, all_gpus, 1/*splitting factor*/, slices);
  }
  else
  {
    decompose_index_space(domain, all_cpus, 1/*splitting factor*/, slices);
  }
}

bool CircuitMapper::map_task(Task *task)
{
  if (map_to_gpus && (task->task_id != TOP_LEVEL_TASK_ID) &&
      (task->task_id != CHECK_FIELD_TASK_ID))
  {
    // Otherwise do custom mappings for GPU memories
    Memory zc_mem = machine_interface.find_memory_kind(task->target_proc,
                                                       Memory::Z_COPY_MEM);
    assert(zc_mem.exists());
    Memory fb_mem = machine_interface.find_memory_kind(task->target_proc,
                                                       Memory::GPU_FB_MEM);
    assert(zc_mem.exists());
    switch (task->task_id)
    {
      case CALC_NEW_CURRENTS_TASK_ID:
        {
          for (unsigned idx = 0; idx < task->regions.size(); idx++)
          {
            // Wires and pvt nodes in framebuffer, 
            // shared and ghost in zero copy memory
            if (idx < 3)
              task->regions[idx].target_ranking.push_back(fb_mem);
            else
              task->regions[idx].target_ranking.push_back(zc_mem);
            task->regions[idx].virtual_map = false;
            task->regions[idx].enable_WAR_optimization = war_enabled;
            task->regions[idx].reduction_list = false;
            // Make everything SOA
            task->regions[idx].blocking_factor = 
              task->regions[idx].max_blocking_factor;
          }
          break;
        }
      case DISTRIBUTE_CHARGE_TASK_ID:
        {
          for (unsigned idx = 0; idx < task->regions.size(); idx++)
          {
            // Wires and pvt nodes in framebuffer, 
            // shared and ghost in zero copy memory
            if (idx < 2)
              task->regions[idx].target_ranking.push_back(fb_mem);
            else
              task->regions[idx].target_ranking.push_back(zc_mem);
            task->regions[idx].virtual_map = false;
            task->regions[idx].enable_WAR_optimization = war_enabled;
            task->regions[idx].reduction_list = false;
            // Make everything SOA
            task->regions[idx].blocking_factor = 
              task->regions[idx].max_blocking_factor;
          }
          break;
        }
      case UPDATE_VOLTAGES_TASK_ID:
        {
          for (unsigned idx = 0; idx < task->regions.size(); idx++)
          {
            // Only shared write stuff needs to go in zc_mem
            if (idx != 1)
              task->regions[idx].target_ranking.push_back(fb_mem);
            else
              task->regions[idx].target_ranking.push_back(zc_mem);
            task->regions[idx].virtual_map = false;
            task->regions[idx].enable_WAR_optimization = war_enabled;
            task->regions[idx].reduction_list = false;
            // Make everything SOA
            task->regions[idx].blocking_factor = 
              task->regions[idx].max_blocking_factor;
          }
          break;
        }
      default:
        assert(false); // should never get here
    }
  }
  else
  {
    // Put everything in the system memory
    Memory sys_mem = 
      machine_interface.find_memory_kind(task->target_proc,
                                         Memory::SYSTEM_MEM);
    assert(sys_mem.exists());
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem);
      task->regions[idx].virtual_map = false;
      task->regions[idx].enable_WAR_optimization = war_enabled;
      task->regions[idx].reduction_list = false;
      // Make everything SOA
      task->regions[idx].blocking_factor = 
        task->regions[idx].max_blocking_factor;
    }
  }
  // We don't care about the result
  return false;
}

bool CircuitMapper::map_inline(Inline *inline_operation)
{
  // let the default mapper do its thing, and then override the
  //  blocking factor to force SOA
  bool ret = DefaultMapper::map_inline(inline_operation);
  RegionRequirement& req = inline_operation->requirement;
  req.blocking_factor = req.max_blocking_factor;
  return ret;
}

void CircuitMapper::notify_mapping_failed(const Mappable *mappable)
{
  switch (mappable->get_mappable_kind())
  {
    case Mappable::TASK_MAPPABLE:
      {
        Task *task = mappable->as_mappable_task();
        int failed_idx = -1;
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          if (task->regions[idx].mapping_failed)
          {
            failed_idx = idx;
            break;
          }
        }
        log_mapper.error("Failed task mapping for region %d of task %s (%p)\n", 
			 failed_idx, task->variants->name, task); 
        assert(false);
        break;
      }
    case Mappable::COPY_MAPPABLE:
      {
        Copy *copy = mappable->as_mappable_copy();
        int failed_idx = -1;
        for (unsigned idx = 0; idx < copy->src_requirements.size(); idx++)
        {
          if (copy->src_requirements[idx].mapping_failed)
          {
            failed_idx = idx;
            break;
          }
        }
        for (unsigned idx = 0; idx < copy->dst_requirements.size(); idx++)
        {
          if (copy->dst_requirements[idx].mapping_failed)
          {
            failed_idx = copy->src_requirements.size() + idx;
            break;
          }
        }
        log_mapper.error("Failed copy mapping for region %d of copy (%p)\n", 
			 failed_idx, copy);
        assert(false);
        break;
      }
    default:
      assert(false);
  }
}


