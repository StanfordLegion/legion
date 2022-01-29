/* Copyright 2022 Stanford University
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

#include "circuit_mapper.h"

Logger log_mapper("mapper");

CircuitMapper::CircuitMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                             std::map<Processor, Memory>* _proc_sysmems,
                             std::map<Processor, Memory>* _proc_fbmems,
                             std::map<Processor, Memory>* _proc_zcmems)
  : DefaultMapper(rt, machine, local, mapper_name),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
    proc_fbmems(*_proc_fbmems),
    proc_zcmems(*_proc_zcmems)
{
}

void CircuitMapper::map_task(const MapperContext      ctx,
                             const Task&              task,
                             const MapTaskInput&      input,
                                   MapTaskOutput&     output)
{
  if ((task.task_id != TOP_LEVEL_TASK_ID) &&
      (task.task_id != CHECK_FIELD_TASK_ID))
  {
    Processor::Kind target_kind = task.target_proc.kind();
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                      true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = 0;
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    bool map_to_gpu = task.target_proc.kind() == Processor::TOC_PROC;
    Memory sysmem = proc_sysmems[task.target_proc];
    Memory fbmem = proc_fbmems[task.target_proc];

    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if ((task.regions[idx].privilege == NO_ACCESS) ||
          (task.regions[idx].privilege_fields.empty())) continue;

      Memory target_memory;
      if (!map_to_gpu) 
        target_memory = sysmem;
      else
        target_memory = fbmem;
      const RegionRequirement &req = task.regions[idx];
      map_circuit_region(ctx, req.region, task.target_proc, 
                         target_memory, output.chosen_instances[idx], 
                         req.privilege_fields, req.redop);
    }
    runtime->acquire_instances(ctx, output.chosen_instances);
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

void CircuitMapper::map_inline(const MapperContext    ctx,
                               const InlineMapping&   inline_op,
                               const MapInlineInput&  input,
                                     MapInlineOutput& output)
{
  Memory target_memory =
    proc_sysmems[inline_op.parent_task->current_proc];
  std::map<Processor, Memory>::iterator finder =
    proc_zcmems.find(inline_op.parent_task->current_proc);
  if (finder != proc_zcmems.end())
    target_memory = finder->second;
  bool force_create = false;
  LayoutConstraintID our_layout_id =
    default_policy_select_layout_constraints(ctx, target_memory,
        inline_op.requirement, INLINE_MAPPING, true, force_create);
  LayoutConstraintSet creation_constraints =
    runtime->find_layout_constraints(ctx, our_layout_id);
  std::set<FieldID> fields(inline_op.requirement.privilege_fields);
  creation_constraints.add_constraint(
      FieldConstraint(fields, false/*contig*/, false/*inorder*/));
  output.chosen_instances.resize(output.chosen_instances.size()+1);
  if (!default_make_instance(ctx, target_memory, creation_constraints,
        output.chosen_instances.back(), INLINE_MAPPING,
        force_create, true, inline_op.requirement))
  {
    log_mapper.error("Circuit mapper failed allocation for region "
                 "requirement of inline mapping in task %s (UID %lld) "
                 "in memory " IDFMT "for processor " IDFMT ". This "
                 "means the working set of your application is too big "
                 "for the allotted capacity of the given memory under "
                 "the default mapper's mapping scheme. You have three "
                 "choices: ask Realm to allocate more memory, write a "
                 "custom mapper to better manage working sets, or find "
                 "a bigger machine. Good luck!",
                 inline_op.parent_task->get_task_name(),
                 inline_op.parent_task->get_unique_id(),
                 target_memory.id,
                 inline_op.parent_task->current_proc.id);
  }
}

void CircuitMapper::map_circuit_region(const MapperContext ctx, LogicalRegion region,
                                       Processor target_proc, Memory target,
                                       std::vector<PhysicalInstance> &instances,
                                       const std::set<FieldID> &privilege_fields,
                                       ReductionOpID redop)
{
  const std::pair<LogicalRegion,Memory> key(region, target);
  if (redop > 0) {
    assert(redop == REDUCE_ID);
    std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance>::const_iterator
      finder = reduction_instances.find(key);
    if (finder != reduction_instances.end()) {
      instances.push_back(finder->second);
      return;
    }
  } else {
    std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance>::const_iterator
      finder = local_instances.find(key);
    if (finder != local_instances.end()) {
      instances.push_back(finder->second);
      return;
    }
  }
  // First time through, then we make an instance
  std::vector<LogicalRegion> regions(1, region);  
  LayoutConstraintSet layout_constraints;
  // No specialization
  if (redop > 0)
    layout_constraints.add_constraint(
        SpecializedConstraint(LEGION_AFFINE_REDUCTION_SPECIALIZE, redop));
  else
    layout_constraints.add_constraint(SpecializedConstraint(LEGION_AFFINE_SPECIALIZE));
  // SOA-Fortran dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_X;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_Z;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, 
                                                       false/*contiguous*/));
  // Constrained for the target memory kind
  layout_constraints.add_constraint(MemoryConstraint(target.kind()));
  // Have all the field for the instance available
  std::vector<FieldID> all_fields;
  if (redop > 0)
    all_fields.insert(all_fields.end(), privilege_fields.begin(), privilege_fields.end());
  else
    runtime->get_field_space_fields(ctx, region.get_field_space(), all_fields);
  layout_constraints.add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                                    false/*inorder*/));
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
  if (target_proc.kind() == Processor::LOC_PROC) {
    for (std::vector<FieldID>::const_iterator it =
          all_fields.begin(); it != all_fields.end(); it++)
#if defined(__AVX512F__)
      layout_constraints.add_constraint(AlignmentConstraint(*it, LEGION_EQ_EK, 64/*bytes*/));
#elif defined(__AVX__)
      layout_constraints.add_constraint(AlignmentConstraint(*it, LEGION_EQ_EK, 32/*bytes*/));
#else
      layout_constraints.add_constraint(AlignmentConstraint(*it, LEGION_EQ_EK, 16/*bytes*/));
#endif
  }
#endif

  PhysicalInstance result; bool created;
  if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
        regions, result, created, true/*acquire*/, GC_NEVER_PRIORITY)) {
    log_mapper.error("ERROR: Circuit Mapper failed to allocate instance");
    assert(false);
  }
  instances.push_back(result);
  // Save the result for future use
  if (redop > 0)
    reduction_instances[key] = result;
  else
    local_instances[key] = result;
}

void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_fbmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_zcmems = new std::map<Processor, Memory>();

  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];

    // skip memories with no capacity for creating instances
    if(affinity.m.capacity() == 0)
      continue;

    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::Z_COPY_MEM) {
        (*proc_zcmems)[affinity.p] = affinity.m;
      }
    }
    else if (affinity.p.kind() == Processor::TOC_PROC) {
      if (affinity.m.kind() == Memory::GPU_FB_MEM) {
        (*proc_fbmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::Z_COPY_MEM) {
        (*proc_zcmems)[affinity.p] = affinity.m;
      }
    }
  }

  // do a second pass in which a regdma/socket memory is used as a sysmem if
  //   a processor doesn't have sysmem for some reason
  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];

    // skip memories with no capacity for creating instances
    if(affinity.m.capacity() == 0)
      continue;

    if ((affinity.p.kind() == Processor::LOC_PROC) &&
	((affinity.m.kind() == Memory::SOCKET_MEM) ||
	 (affinity.m.kind() == Memory::REGDMA_MEM)) &&
	(proc_sysmems->count(affinity.p) == 0)) {
      (*proc_sysmems)[affinity.p] = affinity.m;
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
    CircuitMapper* mapper = new CircuitMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "circuit_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems,
                                              proc_fbmems,
                                              proc_zcmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

