/* Copyright 2021 Stanford University
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

#include "external_test.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

static LegionRuntime::Logger::Category log_mapper("external_test_mapper");

class ExternalTestMapper : public DefaultMapper
{
public:
  ExternalTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name,
                std::vector<Processor>* procs_list,
                std::vector<Memory>* sysmems_list,
                std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
                std::map<Processor, Memory>* proc_sysmems);
  void map_task(const MapperContext      ctx,
                const Task&              task,
                const MapTaskInput&      input,
                      MapTaskOutput&     output);
private:
  //std::vector<Processor>& procs_list;
  //std::vector<Memory>& sysmems_list;
  //std::map<Memory, std::vector<Processor> >& sysmem_local_procs;
  std::map<Processor, Memory>& proc_sysmems;
};

ExternalTestMapper::ExternalTestMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                             std::map<Processor, Memory>* _proc_sysmems)
  : DefaultMapper(rt, machine, local, mapper_name),
    //procs_list(*_procs_list),
    //sysmems_list(*_sysmems_list),
    //sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems)
{
}

void ExternalTestMapper::map_task(const MapperContext  ctx,
                                const Task&          task,
                                const MapTaskInput&  input,
                                      MapTaskOutput& output)
{
  const char* task_name = task.get_task_name();

  if (strcmp(task_name, "aos_test") == 0 || strcmp(task_name, "soa_test") == 0) {

    Processor::Kind target_kind = task.target_proc.kind();
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                      true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = 0;
    output.postmap_task = false;
    output.target_procs.push_back(task.target_proc);
    Memory target_memory = proc_sysmems[task.target_proc];

    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      const RegionRequirement& req = task.regions[idx];
      assert((req.privilege != NO_ACCESS) && !req.privilege_fields.empty());
      std::set<FieldID> fields(task.regions[idx].privilege_fields);

      std::vector<LogicalRegion> target_regions(1, req.region);
      bool created = false;
      PhysicalInstance result;
      LayoutConstraintSet constraints;

      std::vector<DimensionKind> dimension_ordering(4);
      if (strcmp(task_name, "aos_test") == 0) {
        dimension_ordering[0] = DIM_F;
        dimension_ordering[1] = DIM_X;
        dimension_ordering[2] = DIM_Y;
        dimension_ordering[3] = DIM_Z;
      }
      else if (strcmp(task_name, "soa_test") == 0) {
        dimension_ordering[0] = DIM_X;
        dimension_ordering[1] = DIM_Y;
        dimension_ordering[2] = DIM_Z;
        dimension_ordering[3] = DIM_F;
      }
      else {
        assert(false);
      }
      constraints.add_constraint(MemoryConstraint(target_memory.kind()))
        .add_constraint(FieldConstraint(fields, true/*contiguous*/, true/*inorder*/))
        .add_constraint(OrderingConstraint(dimension_ordering, false/*contigous*/));
      if (!runtime->find_or_create_physical_instance(ctx, target_memory,
                                  constraints, target_regions, result, created))
        default_report_failed_instance_creation(task, idx, task.target_proc,
                                                target_memory);
      output.chosen_instances[idx].push_back(result);
    }

    return;
  }

  DefaultMapper::map_task(ctx, task, input, output);
}

static void create_mappers(Machine machine, Runtime *runtime,
                                         const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();


  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
      }
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
    ExternalTestMapper* mapper = new ExternalTestMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "external_test_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}

