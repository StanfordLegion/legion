/* Copyright 2019 Stanford University
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

///
/// Mapper
///

static LegionRuntime::Logger::Category log_circuit("circuit");

class CircuitMapper : public DefaultMapper
{
public:
  CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
  virtual Memory default_policy_select_target_memory(
                                MapperContext ctx,
                                Processor target_proc,
                                const RegionRequirement &req);
};

CircuitMapper::CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void CircuitMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

Memory CircuitMapper::default_policy_select_target_memory(MapperContext ctx,
                                                          Processor target_proc,
                                                          const RegionRequirement &req)
{
  if (target_proc.kind() != Processor::TOC_PROC ||
      !runtime->has_parent_logical_partition(ctx, req.region))
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
  LogicalRegion parent = runtime->get_parent_logical_region(ctx,
      runtime->get_parent_logical_partition(ctx, req.region));
  if (!runtime->has_parent_logical_partition(ctx, parent))
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
  DomainPoint color = runtime->get_logical_region_color_point(ctx, parent);
  if (color[0] > 0)
  {
    Machine::MemoryQuery visible_memories(machine);
    visible_memories.has_affinity_to(target_proc);
    visible_memories.only_kind(Memory::Z_COPY_MEM);
    assert(visible_memories.count() > 0);
    return *visible_memories.begin();
  }
  else
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req);
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    CircuitMapper* mapper = new CircuitMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "circuit_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
