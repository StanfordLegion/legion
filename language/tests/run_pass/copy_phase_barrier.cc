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

#include "copy_phase_barrier.h"

#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

///
/// Mapper
///

class TestMapper : public DefaultMapper
{
public:
  TestMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual bool rank_copy_targets(const Mappable *mappable,
                                 LogicalRegion rebuild_region,
                                 const std::set<Memory> &current_instances,
                                 bool complete,
                                 size_t max_blocking_factor,
                                 std::set<Memory> &to_reuse,
                                 std::vector<Memory> &to_create,
                                 bool &create_one,
                                 size_t &blocking_factor);
// private:
//   Memory local_sysmem;
//   Memory local_regmem;
//   std::set<Processor> local_procs;
//   std::map<std::string, TaskPriority> task_priorities;
//   std::set<Processor> all_procs;
//   std::map<Processor, Memory> all_sysmem;
//   std::map<Processor, Memory> all_regmem;
};

TestMapper::TestMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  printf("TestMapper::TestMapper\n");
  // local_sysmem =
  //   machine_interface.find_memory_kind(local_proc, Memory::SYSTEM_MEM);
  // local_regmem =
  //   machine_interface.find_memory_kind(local_proc, Memory::REGDMA_MEM);
  // if(!local_regmem.exists()) {
  //   local_regmem = local_sysmem;
  // }

  // machine.get_shared_processors(local_sysmem, local_procs);
  // if (!local_procs.empty()) {
  //   machine_interface.filter_processors(machine, Processor::LOC_PROC, local_procs);
  // }

  // machine.get_all_processors(all_procs);
  // for (std::set<Processor>::iterator it = all_procs.begin(), ie = all_procs.end();
  //      it != ie; ++it) {
  //   all_sysmem[*it] =
  //     machine_interface.find_memory_kind(*it, Memory::SYSTEM_MEM);
  //   all_regmem[*it] =
  //     machine_interface.find_memory_kind(*it, Memory::REGDMA_MEM);
  //   if(!all_regmem[*it].exists()) {
  //     all_regmem[*it] = all_sysmem[*it];
  //   }
  // }
}

bool TestMapper::rank_copy_targets(const Mappable *mappable,
                                   LogicalRegion rebuild_region,
                                   const std::set<Memory> &current_instances,
                                   bool complete,
                                   size_t max_blocking_factor,
                                   std::set<Memory> &to_reuse,
                                   std::vector<Memory> &to_create,
                                   bool &create_one,
                                   size_t &blocking_factor)
{
  printf("TestMapper::rank_copy_targets\n");
  DefaultMapper::rank_copy_targets(mappable, rebuild_region, current_instances,
                                   complete, max_blocking_factor, to_reuse,
                                   to_create, create_one, blocking_factor);
  if (create_one) {
    blocking_factor = max_blocking_factor;
  }
  return false;
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new TestMapper(machine, runtime, *it), *it);
  }
}

void register_mappers()
{
  printf("register_mappers\n");
  HighLevelRuntime::set_registration_callback(create_mappers);
}
