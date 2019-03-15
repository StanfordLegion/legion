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
                const char *mapper_name,
                std::vector<Processor>* procs_list);
  virtual void select_task_options(const MapperContext    ctx,
                                   const Task&            task,
                                         TaskOptions&     output);
  virtual void default_policy_rank_processor_kinds(
                                    MapperContext ctx, const Task &task,
                                    std::vector<Processor::Kind> &ranking);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
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
  virtual void map_copy(const MapperContext ctx,
                        const Copy &copy,
                        const MapCopyInput &input,
                        MapCopyOutput &output);
  template<bool IS_SRC>
  void stencil_create_copy_instance(MapperContext ctx, const Copy &copy,
                                    const RegionRequirement &req, unsigned index,
                                    std::vector<PhysicalInstance> &instances);
private:
  std::vector<Processor>& procs_list;
};

StencilMapper::StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list)
  : DefaultMapper(rt, machine, local, mapper_name)
  , procs_list(*_procs_list)
{
}

void StencilMapper::select_task_options(const MapperContext    ctx,
                                        const Task&            task,
                                              TaskOptions&     output)
{
  output.initial_proc = default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  output.stealable = stealing_enabled;
#ifdef MAP_LOCALLY
  output.map_locally = true;
#else
  output.map_locally = false;
#endif
}

void StencilMapper::default_policy_rank_processor_kinds(MapperContext ctx,
                        const Task &task, std::vector<Processor::Kind> &ranking)
{
#if SPMD_SHARD_USE_IO_PROC
  const char* task_name = task.get_task_name();
  const char* prefix = "shard_";
  if (strncmp(task_name, prefix, strlen(prefix)) == 0) {
    // Put shard tasks on IO processors.
    ranking.resize(4);
    ranking[0] = Processor::TOC_PROC;
    ranking[1] = Processor::PROC_SET;
    ranking[2] = Processor::IO_PROC;
    ranking[3] = Processor::LOC_PROC;
  } else {
#endif
    ranking.resize(4);
    ranking[0] = Processor::TOC_PROC;
    ranking[1] = Processor::PROC_SET;
    ranking[2] = Processor::LOC_PROC;
    ranking[3] = Processor::IO_PROC;
#if SPMD_SHARD_USE_IO_PROC
  }
#endif
}

Processor StencilMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
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
  runtime->find_valid_variants(ctx, task.task_id, variants, Processor::LOC_PROC);
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

  LayoutConstraintSet constraints;
  std::vector<FieldID> fields;
  fields.push_back(*task.regions[0].privilege_fields.begin());
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
  target_regions.push_back(task.regions[0].region);
  target_regions.push_back(task.regions[1].region);
  PhysicalInstance result;
  bool created;
  runtime->find_or_create_physical_instance(ctx, target_memory,
                    constraints, target_regions, result, created);

  output.chosen_instances[0].push_back(result);
  output.chosen_instances[1].push_back(result);

  {
    LayoutConstraintSet constraints;
    std::vector<FieldID> fields;
    fields.push_back(*task.regions[2].privilege_fields.begin());
    Memory target_memory = default_policy_select_target_memory(ctx,
                                               task.target_proc,
                                               task.regions[2]);
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
    target_regions.push_back(task.regions[2].region);
    PhysicalInstance result;
    bool created;
    runtime->find_or_create_physical_instance(ctx, target_memory,
                      constraints, target_regions, result, created);
    output.chosen_instances[2].push_back(result);
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

void StencilMapper::map_copy(const MapperContext ctx,
                             const Copy &copy,
                             const MapCopyInput &input,
                             MapCopyOutput &output)
{
  log_stencil.spew("Stencil mapper map_copy");
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
  {
    // Always use a virtual instance for the source.
    output.src_instances[idx].clear();
    output.src_instances[idx].push_back(
      PhysicalInstance::get_virtual_instance());

    // Place the destination instance on the remote node.
    output.dst_instances[idx].clear();
    if (!copy.dst_requirements[idx].is_restricted()) {
      // Call a customized method to create an instance on the desired node.
      stencil_create_copy_instance<false/*is src*/>(ctx, copy, 
        copy.dst_requirements[idx], idx, output.dst_instances[idx]);
    } else {
      // If it's restricted, just take the instance. This will only
      // happen inside the shard task.
      output.dst_instances[idx] = input.dst_instances[idx];
      if (!output.dst_instances[idx].empty())
        runtime->acquire_and_filter_instances(ctx,
                                output.dst_instances[idx]);
    }
  }
}

//--------------------------------------------------------------------------
template<bool IS_SRC>
void StencilMapper::stencil_create_copy_instance(MapperContext ctx,
                     const Copy &copy, const RegionRequirement &req, 
                     unsigned idx, std::vector<PhysicalInstance> &instances)
//--------------------------------------------------------------------------
{
  // This method is identical to the default version except that it
  // chooses an intelligent memory based on the destination of the
  // copy.

  // See if we have all the fields covered
  std::set<FieldID> missing_fields = req.privilege_fields;
  for (std::vector<PhysicalInstance>::const_iterator it = 
        instances.begin(); it != instances.end(); it++)
  {
    it->remove_space_fields(missing_fields);
    if (missing_fields.empty())
      break;
  }
  if (missing_fields.empty())
    return;
  // If we still have fields, we need to make an instance
  // We clearly need to take a guess, let's see if we can find
  // one of our instances to use.

  // ELLIOTT: Get the remote node here.
  Color index = runtime->get_logical_region_color(ctx, copy.src_requirements[idx].region);
  Memory target_memory = default_policy_select_target_memory(ctx,
                           procs_list[index % procs_list.size()],
                           req);
  log_stencil.warning("Building instance for copy of a region with index %u to be in memory %llx",
                      index, target_memory.id);
  bool force_new_instances = false;
  LayoutConstraintID our_layout_id = 
   default_policy_select_layout_constraints(ctx, target_memory, 
                                            req, COPY_MAPPING,
                                            true/*needs check*/, 
                                            force_new_instances);
  LayoutConstraintSet creation_constraints = 
              runtime->find_layout_constraints(ctx, our_layout_id);
  creation_constraints.add_constraint(
      FieldConstraint(missing_fields,
                      false/*contig*/, false/*inorder*/));
  instances.resize(instances.size() + 1);
  if (!default_make_instance(ctx, target_memory, 
        creation_constraints, instances.back(), 
        COPY_MAPPING, force_new_instances, true/*meets*/, req))
  {
    // If we failed to make it that is bad
    log_stencil.error("Stencil mapper failed allocation for "
                   "%s region requirement %d of explicit "
                   "region-to-region copy operation in task %s "
                   "(ID %lld) in memory " IDFMT " for processor "
                   IDFMT ". This means the working set of your "
                   "application is too big for the allotted "
                   "capacity of the given memory under the default "
                   "mapper's mapping scheme. You have three "
                   "choices: ask Realm to allocate more memory, "
                   "write a custom mapper to better manage working "
                   "sets, or find a bigger machine. Good luck!",
                   IS_SRC ? "source" : "destination", idx, 
                   copy.parent_task->get_task_name(),
                   copy.parent_task->get_unique_id(),
		       target_memory.id,
		       copy.parent_task->current_proc.id);
    assert(false);
  }
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();

  Machine::ProcessorQuery procs_query(machine);
  procs_query.only_kind(Processor::LOC_PROC);
  for (Machine::ProcessorQuery::iterator it = procs_query.begin();
        it != procs_query.end(); it++)
    procs_list->push_back(*it);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    StencilMapper* mapper = new StencilMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "stencil_mapper",
                                              procs_list);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::add_registration_callback(create_mappers);
}
