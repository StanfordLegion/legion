/* Copyright 2017 Stanford University
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <queue>
#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  COMPUTE_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
};

class GCMapper : public DefaultMapper {
public:
  GCMapper(Machine machine, HighLevelRuntime *runtime, Processor local)
    : DefaultMapper(machine, runtime, local)
  {
    std::set<Memory> all_mem;
    machine.get_all_memories(all_mem);
    for (std::set<Memory>::iterator it = all_mem.begin(); it != all_mem.end(); it++) {
      if (it->kind() == Memory::SYSTEM_MEM)
        sys_mem.push_back(*it);
    }
    assert(sys_mem.size() == 1);
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    for (std::set<Processor>::iterator it = all_procs.begin(); it != all_procs.end(); it++) {
      if (it->kind() == Processor::LOC_PROC) {
        cpu_procs.push_back(*it);
      }
    }
  }

  virtual void select_task_options(Task *task)
  {
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true;
    task->profile_task = false;
    std::set<Processor> all_procs;
    machine.get_all_processors(all_procs);
    if (task->task_id == COMPUTE_TASK_ID) {
      int idx = *((int*)task->args);
      task->target_proc = cpu_procs[idx % cpu_procs.size()];
    }
  }

  virtual bool map_task(Task *task)
  {
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem[0]);
      task->regions[idx].virtual_map = false;
      task->regions[idx].enable_WAR_optimization = true;
      task->regions[idx].reduction_list = false;
      task->regions[idx].blocking_factor =
        task->regions[idx].max_blocking_factor;
    }
    return true;
  }

public:
  std::vector<Memory> sys_mem;
  std::vector<Processor> cpu_procs;
};

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new GCMapper(machine, rt, *it), *it);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int nregions = 1024, size_per_region = 1024 * 1024;
  unsigned nslots = 8;
  std::queue<Future> future_vec;
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_VAL);
  }
  for (int i = 0; i < nregions; i++) {
    while (future_vec.size() >= nslots) {
     future_vec.front().get_void_result();
     future_vec.pop();
    }

    Rect<1> rect(Point<1>(i * size_per_region), Point<1>((i + 1) * size_per_region - 1));
    IndexSpace is_i = runtime->create_index_space(ctx, Domain::from_rect<1>(rect));
    LogicalRegion lr_i = runtime->create_logical_region(ctx, is_i, fs);
    TaskLauncher task_launcher(COMPUTE_TASK_ID, TaskArgument(&i, sizeof(i)));
    task_launcher.add_region_requirement(
        RegionRequirement(lr_i, READ_WRITE, EXCLUSIVE, lr_i));
    task_launcher.add_field(0, FID_VAL);
    future_vec.push(runtime->execute_task(ctx, task_launcher));
    runtime->destroy_logical_region(ctx, lr_i);
    runtime->destroy_index_space(ctx, is_i);
  }
  runtime->destroy_field_space(ctx, fs);
}

void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  int idx = *((int*)task->args);
  FieldID fid = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> acc =
    regions[0].get_field_accessor(fid).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  //int nsize = dom.get_rect<1>().dim_size(0);
  Rect<1> rect = dom.get_rect<1>();
  int sum = 0;
  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    sum += idx;
    acc.write(DomainPoint::from_point<1>(pir.p), sum);
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  // Register custom mappers
  HighLevelRuntime::set_registration_callback(update_mappers);
  return HighLevelRuntime::start(argc, argv);
}
