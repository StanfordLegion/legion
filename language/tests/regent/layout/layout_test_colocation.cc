/* Copyright 2023 Stanford University, Los Alamos National Laboratory
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

#include "layout_test.h"

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

static LegionRuntime::Logger::Category log_layout_test("layout_test");

class layout_testMapper : public DefaultMapper
{
public:
  layout_testMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list,
                std::vector<Memory>* sysmems_list,
                std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
                std::map<Processor, Memory>* proc_sysmems,
                std::map<Processor, Memory>* proc_regmems);
  void map_task(const MapperContext      ctx,
                const Task&              task,
                const MapTaskInput&      input,
                      MapTaskOutput&     output) override;
  LogicalRegion default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &constraints,
                                bool force_new_instances,
                                bool meets_constraints) override;
  // void default_policy_select_instance_fields(
  //                              MapperContext ctx,
  //                              const RegionRequirement &req,
  //                              const std::set<FieldID> &needed_fields,
  //                              std::vector<FieldID> &fields) override;

private:
  // std::vector<Processor>& procs_list;
  // std::vector<Memory>& sysmems_list;
  //std::map<Memory, std::vector<Processor> >& sysmem_local_procs;
  //std::map<Processor, Memory>& proc_sysmems;
  // std::map<Processor, Memory>& proc_regmems;
};

LogicalRegion layout_testMapper::default_policy_select_instance_region(
                              MapperContext ctx, Memory target_memory,
                              const RegionRequirement &req,
                              const LayoutConstraintSet &constraints,
                              bool force_new_instances,
                              bool meets_constraints)
{
  return req.region;
}

//void layout_testMapper::default_policy_select_instance_fields(
//                                MapperContext ctx,
//                                const RegionRequirement &req,
//                                const std::set<FieldID> &needed_fields,
//                                std::vector<FieldID> &fields)
//{
//  fields.insert(fields.end(), needed_fields.begin(), needed_fields.end());
//}

layout_testMapper::layout_testMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                             std::map<Processor, Memory>* _proc_sysmems,
                             std::map<Processor, Memory>* _proc_regmems)
  : DefaultMapper(rt, machine, local, mapper_name)//,
    // procs_list(*_procs_list),
    // sysmems_list(*_sysmems_list),
    //sysmem_local_procs(*_sysmem_local_procs),
    //proc_sysmems(*_proc_sysmems),
    // proc_regmems(*_proc_regmems)
{
}

void layout_testMapper::map_task(const MapperContext      ctx,
                                 const Task&              task,
                                 const MapTaskInput&      input,
                                       MapTaskOutput&     output)
{
  log_layout_test.info("task name %s", task.get_task_name());
  if (strcmp(task.get_task_name(), "colocate") != 0)
  {
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  for (unsigned idx = 0; idx < task.regions.size(); ++idx)
  {
    LogicalPartition part =
      runtime->get_parent_logical_partition(ctx, task.regions[idx].region);
    const char *name;
    runtime->retrieve_name(ctx, part, name);
    log_layout_test.info("req %u name %s priv %d",
        idx, name, task.regions[idx].privilege);
  }

  Processor::Kind target_kind = task.target_proc.kind();
  // Get the variant that we are going to use to map this task
  VariantInfo chosen = default_find_preferred_variant(task, ctx,
                    true/*needs tight bound*/, true/*cache*/, target_kind);
  output.chosen_variant = chosen.variant;
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  output.target_procs.push_back(task.target_proc);

  LayoutConstraintSet constraints;
  std::vector<FieldID> fields;
  fields.push_back(*task.regions[0].privilege_fields.begin());
  Memory target_memory = default_policy_select_target_memory(ctx,
                                             task.target_proc,
                                             task.regions[0]);
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_X;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_Z;
  dimension_ordering[3] = DIM_F;
  constraints.add_constraint(SpecializedConstraint())
    .add_constraint(MemoryConstraint(target_memory.kind()))
    .add_constraint(FieldConstraint(fields, false/*contiguous*/,
                                    false/*inorder*/))
    .add_constraint(OrderingConstraint(dimension_ordering,
                                       false/*contigous*/));

  std::vector<LogicalRegion> target_regions;
  target_regions.push_back(task.regions[0].region);
  target_regions.push_back(task.regions[2].region);
  //target_regions.push_back(runtime->get_parent_logical_region(ctx,
  //      runtime->get_parent_logical_partition(ctx, task.regions[2].region)));
  PhysicalInstance result;
  runtime->create_physical_instance(ctx, target_memory,
                    constraints, target_regions, result);

  output.chosen_instances[0].push_back(result);
  output.chosen_instances[2].push_back(result);

  {
    LayoutConstraintSet constraints;
    std::vector<FieldID> fields;
    fields.push_back(*task.regions[1].privilege_fields.begin());
    Memory target_memory = default_policy_select_target_memory(ctx,
                                               task.target_proc,
                                               task.regions[1]);
    std::vector<DimensionKind> dimension_ordering(4);
    dimension_ordering[0] = DIM_X;
    dimension_ordering[1] = DIM_Y;
    dimension_ordering[2] = DIM_Z;
    dimension_ordering[3] = DIM_F;
    constraints.add_constraint(SpecializedConstraint())
      .add_constraint(MemoryConstraint(target_memory.kind()))
      .add_constraint(FieldConstraint(fields, false/*contiguous*/,
                                      false/*inorder*/))
      .add_constraint(OrderingConstraint(dimension_ordering,
                                         false/*contigous*/));

    std::vector<LogicalRegion> target_regions;
    target_regions.push_back(task.regions[1].region);
    PhysicalInstance result;
    bool created;
    runtime->find_or_create_physical_instance(ctx, target_memory,
                      constraints, target_regions, result, created);
    output.chosen_instances[1].push_back(result);
  }
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_regmems = new std::map<Processor, Memory>();


  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
        if (proc_regmems->find(affinity.p) == proc_regmems->end())
          (*proc_regmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::REGDMA_MEM)
        (*proc_regmems)[affinity.p] = affinity.m;
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    (*sysmem_local_procs)[it->second].push_back(it->first);
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    layout_testMapper* mapper = new layout_testMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "layout_test_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems,
                                              proc_regmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
