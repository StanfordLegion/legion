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

#include "pennant.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

///
/// Mapper
///

LegionRuntime::Logger::Category log_mapper("mapper");

class CircuitMapper : public DefaultMapper
{
public:
  CircuitMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
private:
  Color get_task_color_by_region(Task *task, const RegionRequirement &requirement);
private:
  std::map<Processor, Memory> all_sysmems;
};

CircuitMapper::CircuitMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  std::set<Processor> all_procs;
	machine.get_all_processors(all_procs);
  machine_interface.filter_processors(machine, Processor::LOC_PROC, all_procs);

  for (std::set<Processor>::iterator itr = all_procs.begin();
       itr != all_procs.end(); ++itr) {
    Memory sysmem = machine_interface.find_memory_kind(*itr, Memory::SYSTEM_MEM);
    all_sysmems[*itr] = sysmem;
  }
}

void CircuitMapper::select_task_options(Task *task)
{
  // Task options:
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = true;
  task->profile_task = false;
}

void CircuitMapper::select_task_variant(Task *task)
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

bool CircuitMapper::map_task(Task *task)
{
  Memory sysmem = all_sysmems[task->target_proc];
  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Region options:
    req.virtual_map = false;
    req.enable_WAR_optimization = false;
    req.reduction_list = false;

    // Place all regions in local system memory.
    req.target_ranking.push_back(sysmem);
  }

  return false;
}

bool CircuitMapper::map_inline(Inline *inline_operation)
{
  Memory sysmem = all_sysmems[local_proc];
  RegionRequirement &req = inline_operation->requirement;

  // Region options:
  req.virtual_map = false;
  req.enable_WAR_optimization = false;
  req.reduction_list = false;
  req.blocking_factor = req.max_blocking_factor;

  // Place all regions in global memory.
  req.target_ranking.push_back(sysmem);

  log_mapper.debug(
    "inline mapping region (%d,%d,%d) target ranking front %d (size %lu)",
    req.region.get_index_space().id,
    req.region.get_field_space().get_id(),
    req.region.get_tree_id(),
    req.target_ranking[0].id,
    req.target_ranking.size());

  return false;
}

void CircuitMapper::notify_mapping_failed(const Mappable *mappable)
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

Color CircuitMapper::get_task_color_by_region(Task *task, const RegionRequirement &requirement)
{
  if (requirement.handle_type == SINGULAR) {
    return get_logical_region_color(requirement.region);
  }
  return 0;
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new CircuitMapper(machine, runtime, *it), *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
