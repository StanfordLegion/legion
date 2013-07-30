/* Copyright 2013 Stanford University
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


#include "circuit.h"
#include "circuit_mapper.h"

#include <cstdlib>
#include <cstdio>
#include <cassert>

using namespace LegionRuntime::HighLevel;

LegionRuntime::Logger::Category log_mapper("circuit_mapper");

CircuitMapper::CircuitMapper(Machine *m, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(m, rt, local)
{
  const std::set<Processor> &all_procs = m->get_all_processors();
  // Make a list of the CPU and GPU processors
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    Processor::Kind kind = m->get_processor_kind(*it);
    if (kind == Processor::LOC_PROC)
      cpu_procs.push_back(*it);
    else
      gpu_procs.push_back(*it); // small assumption that there are no other kinds of processors
  }
  // Make sure we have some CPUs and GPUs
  if (cpu_procs.empty())
  {
    log_mapper.error("Circuit Mapper could not find any CPUs");
    exit(1);
  }
  if (gpu_procs.empty())
  {
    log_mapper.error("Circuit Mapper could not find any GPUs");
    exit(1);
  }

  // Now find our set of memories
  if (local_kind == Processor::LOC_PROC)
  {
    std::vector<Memory> &memory_stack = memory_stacks[local_proc];
    unsigned num_mem = memory_stack.size();
    assert(num_mem >= 2);
    gasnet_mem = memory_stack[num_mem-1];
    {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, local_proc, gasnet_mem);
      assert(result.size() == 1);
      log_mapper.info("CPU %x has gasnet memory %x with "
          "bandwidth %u and latency %u",local_proc.id, gasnet_mem.id,
          result[0].bandwidth, result[0].latency);
    }
    zero_copy_mem = memory_stack[num_mem-2];
    {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, local_proc, zero_copy_mem);
      assert(result.size() == 1);
      log_mapper.info("CPU %x has zero copy memory %x with "
          "bandwidth %u and latency %u",local_proc.id, zero_copy_mem.id,
          result[0].bandwidth, result[0].latency);
    }
    framebuffer_mem = Memory::NO_MEMORY;
  }
  else
  {
    std::vector<Memory> &memory_stack = memory_stacks[local_proc];
    unsigned num_mem = memory_stack.size();
    assert(num_mem >= 2);
    zero_copy_mem = memory_stack[num_mem-1];
    {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, local_proc, zero_copy_mem);
      assert(result.size() == 1);
      log_mapper.info("GPU %x has zero copy memory %x with "
          "bandwidth %u and latency %u",local_proc.id, zero_copy_mem.id,
          result[0].bandwidth, result[0].latency);
    }
    framebuffer_mem = memory_stack[num_mem-2];
    {
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, local_proc, framebuffer_mem);
      assert(result.size() == 1);
      log_mapper.info("GPU %x has frame buffer memory %x with "
          "bandwidth %u and latency %u",local_proc.id, framebuffer_mem.id,
          result[0].bandwidth, result[0].latency);
    }
    // Need to compute the gasnet memory
    {
      // Assume the gasnet memory is the one with the smallest bandwidth
      // from any CPU
      assert(!cpu_procs.empty());
      std::vector<ProcessorMemoryAffinity> result;
      m->get_proc_mem_affinity(result, (cpu_procs.front()));
      assert(!result.empty());
      unsigned min_idx = 0;
      unsigned min_bandwidth = result[0].bandwidth;
      for (unsigned idx = 1; idx < result.size(); idx++)
      {
        if (result[idx].bandwidth < min_bandwidth)
        {
          min_bandwidth = result[idx].bandwidth;
          min_idx = idx;
        }
      }
      gasnet_mem = result[min_idx].m;
      log_mapper.info("GPU %x has gasnet memory %x with "
          "bandwidth %u and latency %u",local_proc.id,gasnet_mem.id,
          result[min_idx].bandwidth,result[min_idx].latency);
    }
  }
}

bool CircuitMapper::spawn_task(const Task *task)
{
  if (task->task_id == REGION_MAIN)
    return false;
  return true;
}

Processor CircuitMapper::select_target_processor(const Task *task)
{
  if (task->task_id == REGION_MAIN)
    return local_proc;
  // All other tasks get mapped onto the GPU
  assert(task->is_index_space);

  DomainPoint point = task->get_index_point();
  unsigned proc_id = point.get_index() % gpu_procs.size();
  return gpu_procs[proc_id];
}

Processor CircuitMapper::target_task_steal(const std::set<Processor> &blacklisted)
{
  // No task stealing
  return Processor::NO_PROC;
}

void CircuitMapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal)
{
  // No stealing, so do nothing
}

bool CircuitMapper::map_task_region(const Task *task, Processor target, 
                                MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                const RegionRequirement &req, unsigned index,
                                const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                std::vector<Memory> &target_ranking,
                                std::set<FieldID> &additional_fields, 
                                bool &enable_WAR_optimization)
{
  // CPU mapper should only be called for region main
  if (local_kind == Processor::LOC_PROC)
  {
    assert(task->task_id == REGION_MAIN);
    // Put everything in gasnet here
    target_ranking.push_back(gasnet_mem);
  }
  else 
  {
    switch (task->task_id)
    {
      case CALC_NEW_CURRENTS:
        {
          switch (index)
          {
            case 0:
              {
                // Wires in frame buffer
                target_ranking.push_back(framebuffer_mem);
                // No WAR optimization here, re-use instances
                enable_WAR_optimization = false;
                break;
              }
            case 1:
              {
                // Private nodes in frame buffer
                target_ranking.push_back(framebuffer_mem);
                break;
              }
            case 2:
              {
                // Shared nodes in zero-copy mem
                target_ranking.push_back(zero_copy_mem);
                break;
              }
            case 3:
              {
                // Ghost nodes in zero-copy mem
                target_ranking.push_back(zero_copy_mem);
                break;
              }
            default:
              assert(false);
          }
          break;
        }
      case DISTRIBUTE_CHARGE:
        {
          switch (index)
          {
            case 0:
              {
                // Wires in frame buffer
                target_ranking.push_back(framebuffer_mem);
                break;
              }
            case 1:
              {
                // Private nodes in frame buffer
                target_ranking.push_back(framebuffer_mem);
                // No WAR optimization here
                enable_WAR_optimization = false;
                break;
              }
            case 2:
              {
                // Shared nodes in zero-copy mem
                target_ranking.push_back(zero_copy_mem);
                break;
              }
            case 3:
              {
                // Shared nodes in zero-copy mem
                target_ranking.push_back(zero_copy_mem);
                break;
              }
            default:
              assert(false);
          }
          break;
        }
      case UPDATE_VOLTAGES:
        {
          switch (index)
          {
            case 0:
              {
                // Private nodes in frame buffer
                target_ranking.push_back(framebuffer_mem);
                break;
              }
            case 1:
              {
                // Shared nodes in zero-copy mem
                target_ranking.push_back(zero_copy_mem);
                break;
              }
            case 2:
              {
                // Locator map, always put in our frame buffer
                target_ranking.push_back(framebuffer_mem);
                break;
              }
            default:
              assert(false);
          }
          break;
        }
      default:
        assert(false);
    }
  }
  return true;
}

void CircuitMapper::rank_copy_target(const Task *task, Processor target,
                                      MappingTagID tag, bool inline_mapping,
                                      const RegionRequirement &req, unsigned index,
                                      const std::set<Memory> &current_instances,
                                      std::set<Memory> &to_reuse,
                                      std::vector<Memory> &to_create,
                                      bool &create_one)
{
  // The gasnet memory should already be a valid choice
  assert(current_instances.find(gasnet_mem) != current_instances.end());
  to_reuse.insert(gasnet_mem);
}

void CircuitMapper::slice_index_space(const Task *task, const IndexSpace &index_space,
                                  std::vector<Mapper::DomainSplit> &slices)
{
  DefaultMapper::decompose_index_space(index_space, gpu_procs, 1/*splitting factor*/, slices);
}

// EOF

