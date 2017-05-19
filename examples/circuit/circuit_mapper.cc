/* Copyright 2017 Stanford University
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
    Memory zcmem = proc_zcmems[task.target_proc];

    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if ((task.regions[idx].privilege == NO_ACCESS) ||
          (task.regions[idx].privilege_fields.empty())) continue;

      Memory target_memory;
      if (!map_to_gpu) target_memory = sysmem;
      else {
        switch (task.task_id)
        {
          case CALC_NEW_CURRENTS_TASK_ID:
            {
              if (idx < 3)
                target_memory = fbmem;
              else
                target_memory = zcmem;
              break;
            }

          case DISTRIBUTE_CHARGE_TASK_ID:
            {
              if (idx < 2)
                target_memory = fbmem;
              else
                target_memory = zcmem;
              break;
            }

          case UPDATE_VOLTAGES_TASK_ID:
            {
              if (idx != 1)
                target_memory = fbmem;
              else
                target_memory = zcmem;
              break;
            }

          default:
            {
              assert(false);
              break;
            }
        }
      }
      const TaskLayoutConstraintSet &layout_constraints =
        runtime->find_task_layout_constraints(ctx,
                            task.task_id, output.chosen_variant);
      std::set<FieldID> fields(task.regions[idx].privilege_fields);
      if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, task.regions[idx], idx, fields,
              layout_constraints, true,
              output.chosen_instances[idx]))
      {
        default_report_failed_instance_creation(task, idx,
                                    task.target_proc, target_memory);
      }
    }
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

void update_mappers(Machine machine, HighLevelRuntime *runtime,
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

