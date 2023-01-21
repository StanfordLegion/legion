/* Copyright 2023 Stanford University
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

#include "optimize_tracing_invalidate1.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class TracingMapper : public DefaultMapper
{
public:
  TracingMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
public:
  virtual void slice_task(const MapperContext      ctx,
                          const Task&              task,
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output);
  virtual void map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output);
private:
  typedef std::map<std::pair<LogicalRegion,FieldID>,PhysicalInstance> Mapping;
  Mapping inc_instances;
  Mapping step_instances;
  Mapping check_instances;
};

TracingMapper::TracingMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void TracingMapper::slice_task(const MapperContext      ctx,
                               const Task&              task,
                               const SliceTaskInput&    input,
                                     SliceTaskOutput&   output)
{
  DomainT<1,coord_t> point_space = input.domain;
  Point<1,coord_t> num_blocks(local_cpus.size());
  default_decompose_points<1>(point_space, local_cpus,
                              num_blocks, false/*recurse*/,
                              stealing_enabled, output.slices);
}

void TracingMapper::map_task(const MapperContext ctx,
                             const Task &task,
                             const MapTaskInput &input,
                             MapTaskOutput &output)
{
  if (strcmp(task.get_task_name(), "inc") == 0 ||
      strcmp(task.get_task_name(), "step") == 0 ||
      strcmp(task.get_task_name(), "check") == 0)
  {
    Mapping &mapping =
       strcmp(task.get_task_name(), "inc") == 0   ? inc_instances :
      (strcmp(task.get_task_name(), "step") == 0  ? step_instances : check_instances);

    Processor::Kind target_kind = task.target_proc.kind();
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
        true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = default_policy_select_task_priority(ctx, task);
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    bool need_acquire = false;
    for (unsigned idx = 0; idx < task.regions.size(); ++idx)
    {
      const RegionRequirement &req = task.regions[idx];
      for (std::set<FieldID>::const_iterator fit = req.privilege_fields.begin();
           fit != req.privilege_fields.end(); ++fit)
      {
        std::pair<LogicalRegion,FieldID> key(req.region, *fit);
        Mapping::iterator finder = mapping.find(key);
        if (finder == mapping.end())
        {
          PhysicalInstance result;
          Machine::MemoryQuery valid_mems(machine);
          valid_mems.has_affinity_to(task.target_proc);
          valid_mems.only_kind(Memory::SYSTEM_MEM);
          Memory target_memory = *valid_mems.begin();

          std::vector<LogicalRegion> target_regions(1, req.region);

          LayoutConstraintSet constraints;
          std::vector<FieldID> fields(1, *fit);
          std::vector<DimensionKind> dimension_ordering(1, DIM_X);

          constraints.add_constraint(SpecializedConstraint())
            .add_constraint(MemoryConstraint(target_memory.kind()))
            .add_constraint(FieldConstraint(fields, false, false))
            .add_constraint(OrderingConstraint(dimension_ordering, false));

          assert(runtime->create_physical_instance(ctx, target_memory,
                constraints, target_regions, result));
          mapping[key] = result;
          output.chosen_instances[idx].push_back(result);
        }
        else
        {
          output.chosen_instances[idx].push_back(finder->second);
          need_acquire = true;
        }
      }
    }
    if (need_acquire && !runtime->acquire_instances(ctx, output.chosen_instances))
      assert(false);
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    TracingMapper* mapper = new TracingMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "variant_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
