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

#include "stencil_mapper.h"

#include "mappers/default_mapper.h"

#define SPMD_SHARD_USE_IO_PROC 1

using namespace Legion;
using namespace Legion::Mapping;

static LegionRuntime::Logger::Category log_stencil("stencil");

class StencilMapper : public DefaultMapper
{
public:
  StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                const char *mapper_name);
  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);
  virtual LogicalRegion default_policy_select_instance_region(
                                MapperContext ctx, Memory target_memory,
                                const RegionRequirement &req,
                                const LayoutConstraintSet &constraints,
                                bool force_new_instances,
                                bool meets_constraints);
  virtual void map_task(const MapperContext ctx,
                        const Task &task,
                        const MapTaskInput &input,
                        MapTaskOutput &output);
};

StencilMapper::StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
}

void StencilMapper::default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs)
{
  target_procs.push_back(task.target_proc);
}

LogicalRegion StencilMapper::default_policy_select_instance_region(
                              MapperContext ctx, Memory target_memory,
                              const RegionRequirement &req,
                              const LayoutConstraintSet &constraints,
                              bool force_new_instances,
                              bool meets_constraints)
{
  return req.region;
}

void StencilMapper::map_task(const MapperContext      ctx,
                             const Task&              task,
                             const MapTaskInput&      input,
                                   MapTaskOutput&     output)
{
  if (task.parent_task != NULL && task.parent_task->must_epoch_task) {
    Processor::Kind target_kind = task.target_proc.kind();
    // Get the variant that we are going to use to map this task
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                                                        true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    // TODO: some criticality analysis to assign priorities
    output.task_priority = 0;
    output.postmap_task = false;
    // Figure out our target processors
    output.target_procs.push_back(task.target_proc);

    for (unsigned idx = 0; idx < task.regions.size(); idx++) {
      const RegionRequirement &req = task.regions[idx];

      // Skip any empty regions
      if ((req.privilege == NO_ACCESS) || (req.privilege_fields.empty()))
        continue;

      assert(input.valid_instances[idx].size() == 1);
      output.chosen_instances[idx] = input.valid_instances[idx];
      bool ok = runtime->acquire_and_filter_instances(ctx, output.chosen_instances);
      if (!ok) {
        log_stencil.error("failed to acquire instances");
        assert(false);
      }
    }
    return;
  }

  if (strcmp(task.get_task_name(), "stencil.parallel") != 0)
  {
    DefaultMapper::map_task(ctx, task, input, output);
    return;
  }

  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task.task_id, variants, task.target_proc.kind());
  VariantID chosen_variant = -1U;
  for (unsigned idx = 0; idx < variants.size(); ++idx)
  {
    const char *variant_name =
      runtime->find_task_variant_name(ctx, task.task_id, variants[idx]);
    if (strcmp(variant_name, "colocation") == 0)
    {
      chosen_variant = variants[idx];
      break;
    }
  }
  assert(chosen_variant != -1U);
  output.chosen_variant = chosen_variant;
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  output.target_procs.push_back(task.target_proc);

  CachedMappingPolicy cache_policy =
    default_policy_select_task_cache_policy(ctx, task);
  const unsigned long long task_hash = compute_task_hash(task);
  std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
  std::map<std::pair<TaskID,Processor>,
           std::list<CachedTaskMapping> >::const_iterator
    finder = cached_task_mappings.find(cache_key);
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE && finder != cached_task_mappings.end())
  {
    bool found = false;
    // Iterate through and see if we can find one with our variant and hash
    for (std::list<CachedTaskMapping>::const_iterator it =
          finder->second.begin(); it != finder->second.end(); it++)
    {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash))
      {
        // Have to copy it before we do the external call which
        // might invalidate our iterator
        output.chosen_instances = it->mapping;
        found = true;
        break;
      }
    }
    if (found)
      if (runtime->acquire_and_filter_instances(ctx, output.chosen_instances))
        return;
  }

  ColocationConstraint colocation;
  {
    const ExecutionConstraintSet &constraints =
      runtime->find_execution_constraints(ctx, task.task_id, chosen_variant);
    assert(constraints.colocation_constraints.size() == 1);
    colocation = constraints.colocation_constraints[0];
  }

  LayoutConstraintSet constraints;
  std::vector<FieldID> fields;
  for (std::set<FieldID>::const_iterator it = colocation.fields.begin();
       it != colocation.fields.end(); ++it)
    fields.push_back(*it);
  Memory target_memory = default_policy_select_target_memory(ctx,
                                             task.target_proc,
                                             task.regions[0]);
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
  std::vector<LogicalRegion> target_regions;
  for (std::set<unsigned>::const_iterator it = colocation.indexes.begin();
       it != colocation.indexes.end(); ++it)
    target_regions.push_back(task.regions[*it].region);

  PhysicalInstance result;
  bool created;
  if (!runtime->find_or_create_physical_instance(ctx, target_memory,
        constraints, target_regions, result, created))
  {
    default_report_failed_instance_creation(task, *colocation.indexes.begin(),
        task.target_proc, target_memory);
  }

  for (std::set<unsigned>::const_iterator it = colocation.indexes.begin();
       it != colocation.indexes.end(); ++it)
    output.chosen_instances[*it].push_back(result);

  for (unsigned idx = 0; idx < task.regions.size(); ++idx)
  {
    if (colocation.indexes.find(idx) != colocation.indexes.end() ||
        task.regions[idx].privilege == NO_ACCESS ||
        task.regions[idx].region == LogicalRegion::NO_REGION)
      continue;
    LayoutConstraintSet constraints;
    std::vector<FieldID> fields;
    fields.push_back(*task.regions[idx].privilege_fields.begin());
    Memory target_memory = default_policy_select_target_memory(ctx,
                                               task.target_proc,
                                               task.regions[idx]);
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

    std::vector<LogicalRegion> target_regions;
    target_regions.push_back(task.regions[idx].region);
    PhysicalInstance result;
    bool created;
    if (!runtime->find_or_create_physical_instance(ctx, target_memory,
          constraints, target_regions, result, created))
    {
      default_report_failed_instance_creation(task, idx, task.target_proc,
          target_memory);
    }
    output.chosen_instances[idx].push_back(result);
  }

  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE)
  {
    // Now that we are done, let's cache the result so we can use it later
    std::list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
    map_list.push_back(CachedTaskMapping());
    CachedTaskMapping &cached_result = map_list.back();
    cached_result.task_hash = task_hash;
    cached_result.variant = output.chosen_variant;
    cached_result.mapping = output.chosen_instances;
    cached_result.has_reductions = false;
  }
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    StencilMapper* mapper = new StencilMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "stencil_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
