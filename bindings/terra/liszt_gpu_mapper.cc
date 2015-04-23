/* Copyright 2015 Stanford University
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


#include "liszt_gpu_mapper.h"



#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;



///
/// Mapper
///

LegionRuntime::Logger::Category log_mapper("mapper");

class LzGpuMapper : public DefaultMapper
{
public:
  LzGpuMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
  virtual bool rank_copy_targets(const Mappable *mappable,
                                 LogicalRegion rebuild_region,
                                 const std::set<Memory> &current_instances,
                                 bool complete,
                                 size_t max_blocking_factor,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one,
                                 size_t &blocking_factor);
private:
  Color get_task_color_by_region(Task *task, const RegionRequirement &requirement);
private:
  Memory local_sysmem;
  Memory local_gpumem;
  std::set<Processor> local_procs;
  std::map<std::string, TaskPriority> task_priorities;
};

LzGpuMapper::LzGpuMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  // the GPU memory is Memory::GPU_FB_MEM
  local_sysmem =
    machine_interface.find_memory_kind(local_proc, Memory::SYSTEM_MEM);
  local_gpumem = 
    machine_interface.find_memory_kind(local_proc, Memory::GPU_FB_MEM);

  machine.get_shared_processors(local_sysmem, local_procs);
  if (!local_procs.empty()) {
    machine_interface.filter_processors(machine, Processor::LOC_PROC, local_procs);
  }



  // SET THE GPU PROC ID HERE ETC?





  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);

  if ((*(all_procs.begin())) == local_proc) {
    printf("There are %ld processors:\n", all_procs.size());
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      // For every processor there is an associated kind
      Processor::Kind kind = it->kind();
      switch (kind)
      {
        // Latency-optimized cores (LOCs) are CPUs
        case Processor::LOC_PROC:
          {
            printf("  Processor ID %x is CPU\n", it->id); 
            break;
          }
        // Throughput-optimized cores (TOCs) are GPUs
        case Processor::TOC_PROC:
          {
            printf("  Processor ID %x is GPU\n", it->id);
            break;
          }
        // Utility processors are helper processors for
        // running Legion runtime meta-level tasks and 
        // should not be used for running application tasks
        case Processor::UTIL_PROC:
          {
            printf("  Processor ID %x is utility\n", it->id);
            break;
          }
        default:
          assert(false);
      }
    }

    std::set<Memory> all_mems;
    machine.get_all_memories(all_mems);
    printf("There are %ld memories:\n", all_mems.size());
    for (std::set<Memory>::const_iterator it = all_mems.begin();
          it != all_mems.end(); it++)
    {
      Memory::Kind kind = it->kind();
      size_t memory_size_in_kb = it->capacity() >> 10;
      switch (kind)
      {
        // RDMA addressable memory when running with GASNet
        case Memory::GLOBAL_MEM:
          {
            printf("  GASNet Global Memory ID %x has %ld KB\n", 
                    it->id, memory_size_in_kb);
            break;
          }
        // DRAM on a single node
        case Memory::SYSTEM_MEM:
          {
            printf("  System Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Pinned memory on a single node
        case Memory::REGDMA_MEM:
          {
            printf("  Pinned Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // A memory associated with a single socket
        case Memory::SOCKET_MEM:
          {
            printf("  Socket Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Zero-copy memory betweeen CPU DRAM and
        // all GPUs on a single node
        case Memory::Z_COPY_MEM:
          {
            printf("  Zero-Copy Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // GPU framebuffer memory for a single GPU
        case Memory::GPU_FB_MEM:
          {
            printf("  GPU Frame Buffer Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Disk memory on a single node
        case Memory::DISK_MEM:
          {
            printf("  Disk Memory ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L3 cache
        case Memory::LEVEL3_CACHE:
          {
            printf("  Level 3 Cache ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L2 cache
        case Memory::LEVEL2_CACHE:
          {
            printf("  Level 2 Cache ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L1 cache
        case Memory::LEVEL1_CACHE:
          {
            printf("  Level 1 Cache ID %x has %ld KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        default:
          assert(false);
      }
    }

    std::set<Memory> vis_mems;
    machine.get_visible_memories(local_proc, vis_mems);
    printf("There are %ld memories visible from processor %x\n",
            vis_mems.size(), local_proc.id);
    for (std::set<Memory>::const_iterator it = vis_mems.begin();
          it != vis_mems.end(); it++)
    {
      // Edges between nodes are called affinities in the
      // machine model.  Affinities also come with approximate
      // indications of the latency and bandwidth between the 
      // two nodes.  Right now these are unit-less measurements,
      // but our plan is to teach the Legion runtime to profile
      // these values on start-up to give them real values
      // and further increase the portability of Legion applications.
      std::vector<ProcessorMemoryAffinity> affinities;
      int results = 
        machine.get_proc_mem_affinity(affinities, local_proc, *it);
      // We should only have found 1 results since we
      // explicitly specified both values.
      assert(results == 1);
      printf("  Memory %x has bandwidth %d and latency %d\n",
              it->id, affinities[0].bandwidth, affinities[0].latency);
    }
  }
}

void LzGpuMapper::select_task_options(Task *task)
{
  // Task options:
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = false;
  task->profile_task = false;

  std::string name(task->variants->name);
  if (task_priorities.count(name)) {
    task->task_priority = task_priorities[name];
  } else {
    task->task_priority = 0;
  }
}

void LzGpuMapper::select_task_variant(Task *task)
{
  // Use the SOA variant for all tasks.
  // task->selected_variant = VARIANT_SOA;
  DefaultMapper::select_task_variant(task);

  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Select SOA layout for all regions.
    req.blocking_factor = req.max_blocking_factor;
  }
}

bool LzGpuMapper::map_task(Task *task)
{
  assert(task->target_proc == local_proc);
  task->additional_procs = local_procs;

  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    // Place all regions in local system memory.
    req.target_ranking.push_back(local_sysmem);
  }

  return false;
}

bool LzGpuMapper::map_inline(Inline *inline_operation)
{
  RegionRequirement &req = inline_operation->requirement;

  // Region options:
  req.virtual_map = false;
  req.enable_WAR_optimization = false;
  req.reduction_list = false;
  req.blocking_factor = req.max_blocking_factor;

  // Place all regions in global memory.
  req.target_ranking.push_back(local_sysmem);

  log_mapper.debug(
    "inline mapping region (%d,%d,%d) target ranking front %d (size %lu)",
    req.region.get_index_space().id,
    req.region.get_field_space().get_id(),
    req.region.get_tree_id(),
    req.target_ranking[0].id,
    req.target_ranking.size());

  return false;
}

void LzGpuMapper::notify_mapping_failed(const Mappable *mappable)
{
  switch (mappable->get_mappable_kind()) {
  case Mappable::TASK_MAPPABLE:
    {
      log_mapper.warning("mapping failed on task");
      break;
    }
  case Mappable::COPY_MAPPABLE:
    {
      log_mapper.warning("mapping failed on copy");
      break;
    }
  case Mappable::INLINE_MAPPABLE:
    {
      Inline *_inline = mappable->as_mappable_inline();
      RegionRequirement &req = _inline->requirement;
      LogicalRegion region = req.region;
      log_mapper.warning(
        "mapping %s on inline region (%d,%d,%d) memory %d",
        (req.mapping_failed ? "failed" : "succeeded"),
        region.get_index_space().id,
        region.get_field_space().get_id(),
        region.get_tree_id(),
        req.selected_memory.id);
      break;
    }
  case Mappable::ACQUIRE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on acquire");
      break;
    }
  case Mappable::RELEASE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on release");
      break;
    }
  }
  assert(0 && "mapping failed");
}

bool LzGpuMapper::rank_copy_targets(const Mappable *mappable,
                                      LogicalRegion rebuild_region,
                                      const std::set<Memory> &current_instances,
                                      bool complete,
                                      size_t max_blocking_factor,
                                      std::set<Memory> &to_reuse,
                                      std::vector<Memory> &to_create,
                                      bool &create_one,
                                      size_t &blocking_factor)
{
  DefaultMapper::rank_copy_targets(mappable, rebuild_region, current_instances,
                                   complete, max_blocking_factor, to_reuse,
                                   to_create, create_one, blocking_factor);
  if (create_one) {
    blocking_factor = max_blocking_factor;
  }
  return true;
}

Color LzGpuMapper::get_task_color_by_region(Task *task, const RegionRequirement &requirement)
{
  if (requirement.handle_type == SINGULAR) {
    return get_logical_region_color(requirement.region);
  }
  return 0;
}

void create_mappers(
  Machine machine,
  HighLevelRuntime *runtime,
  const std::set<Processor> &local_procs
) {
  for (
    std::set<Processor>::const_iterator it = local_procs.begin();
    it != local_procs.end();
    it++
  ) {
    runtime->replace_default_mapper(
      new LzGpuMapper(machine, runtime, *it), *it
    );
  }
}

void register_liszt_gpu_mapper()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}



