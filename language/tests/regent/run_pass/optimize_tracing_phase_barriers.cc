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

#include "optimize_tracing_phase_barriers.h"

#include "mappers/default_mapper.h"

#include <cstring>

using namespace Legion;
using namespace Legion::Mapping;

class Tester : public DefaultMapper
{
public:
  Tester(MapperRuntime *rt, Machine machine, Processor local,
             const char *mapper_name);
  void map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output) override;
};

Tester::Tester(MapperRuntime *rt, Machine machine, Processor local,
                       const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void Tester::map_task(const MapperContext ctx,
                      const Task &task,
                      const MapTaskInput &input,
                      MapTaskOutput &output)
{
  if (std::strcmp(task.get_task_name(), "f1") == 0 ||
      //std::strcmp(task.get_task_name(), "f2") == 0 ||
      //std::strcmp(task.get_task_name(), "f") == 0 ||
      //std::strcmp(task.get_task_name(), "g1") == 0 ||
      //std::strcmp(task.get_task_name(), "g1") == 0 ||
      std::strcmp(task.get_task_name(), "g1") == 0 )
  {
    Processor::Kind target_kind = task.target_proc.kind();
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
        true/*needs tight bound*/, true/*cache*/, target_kind);
    assert(chosen.variant != -1U);
    output.chosen_variant = chosen.variant;
    output.task_priority = default_policy_select_task_priority(ctx, task);
    output.postmap_task = false;
    output.target_procs.push_back(task.target_proc);

    assert(task.regions.size() == 1);
    const RegionRequirement &req = task.regions[0];
    const LogicalRegion &region = req.region;
    LayoutConstraintSet constraints;
    std::vector<FieldID> fields;
    runtime->get_field_space_fields(ctx, region.get_field_space(), fields);
    Memory target_memory = default_policy_select_target_memory(ctx,
        task.target_proc, task.regions[0]);
    std::vector<DimensionKind> dimension_ordering(3);
    dimension_ordering[0] = DIM_X;
    dimension_ordering[1] = DIM_Y;
    dimension_ordering[2] = DIM_F;
    constraints.add_constraint(SpecializedConstraint())
      .add_constraint(MemoryConstraint(target_memory.kind()))
      .add_constraint(FieldConstraint(fields, false/*contiguous*/,
            false/*inorder*/))
      .add_constraint(OrderingConstraint(dimension_ordering,
            false/*contigous*/));
    std::vector<LogicalRegion> target_regions(1, region);

    PhysicalInstance result;
    if (!runtime->create_physical_instance(ctx, target_memory,
          constraints, target_regions, result))
    {
      default_report_failed_instance_creation(task, 0, task.target_proc,
          target_memory);
    }
    output.chosen_instances[0].push_back(result);
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

static void create_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    Tester* mapper = new Tester(runtime->get_mapper_runtime(),
                                        machine, *it, "test_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  Runtime::add_registration_callback(create_mappers);
}
