/* Copyright 2014 Stanford University
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

#include "legion.h"
#include "shim_mapper.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

// This is the implementation of the shim mapper which provides
// backwards compatibility with the old version of the mapper interface

namespace LegionRuntime {
  namespace HighLevel {

    Logger::Category log_shim("shim_mapper");

    //--------------------------------------------------------------------------
    ShimMapper::ShimMapper(Machine *m, HighLevelRuntime *rt, Processor local)
      : DefaultMapper(m, rt, local)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShimMapper::ShimMapper(const ShimMapper &rhs)
      : DefaultMapper(NULL, NULL, Processor::NO_PROC)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShimMapper::~ShimMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShimMapper& ShimMapper::operator=(const ShimMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_options(Task *task)
    //--------------------------------------------------------------------------
    {
      task->inline_task = false;
      task->spawn_task = spawn_task(task);
      task->map_locally = map_task_locally(task);
      task->target_proc = select_target_processor(task);
      task->profile_task = profile_task_execution(task, task->target_proc);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_tasks_to_schedule(
                                            const std::list<Task*> &ready_tasks)
    //--------------------------------------------------------------------------
    {
      std::vector<bool> ready_mask(ready_tasks.size(),false);
      select_tasks_to_schedule(ready_tasks, ready_mask);
      unsigned idx = 0;
      for (std::list<Task*>::const_iterator it = ready_tasks.begin();
            it != ready_tasks.end(); it++,idx++)
      {
        (*it)->schedule = ready_mask[idx];
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::target_task_steal(const std::set<Processor> &blacklist,
                                       std::set<Processor> &targets)
    //--------------------------------------------------------------------------
    {
      Processor target = target_task_steal(blacklist);
      if (target.exists())
        targets.insert(target);
    }

    // permit task steal interface is the same

    // slice domain interface is the same

    //--------------------------------------------------------------------------
    bool ShimMapper::pre_map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        if (task->regions[idx].must_early_map)
        {
          task->regions[idx].virtual_map = false;
          task->regions[idx].early_map = true;
          task->regions[idx].enable_WAR_optimization = war_enabled;
          if (task->regions[idx].privilege == REDUCE)
          {
            task->regions[idx].reduction_list = 
              select_reduction_layout(task, task->target_proc,
                                      task->regions[idx], idx,
                                      Memory::NO_MEMORY);
            task->regions[idx].blocking_factor = 1;
          }
          else
          {
            task->regions[idx].blocking_factor = 
              select_region_layout(task, task->target_proc,
                                   task->regions[idx], idx,
                                   Memory::NO_MEMORY,
                                   task->regions[idx].max_blocking_factor);
            task->regions[idx].reduction_list = false;
          }
          bool note = map_task_region(task, task->target_proc,
                                      task->regions[idx].tag,
                                      false/*inline mapping*/,
                                      true/*premapping*/,
                                      task->regions[idx], idx,
                                      task->regions[idx].current_instances,
                                      task->regions[idx].target_ranking,
                                      task->regions[idx].additional_fields,
                                    task->regions[idx].enable_WAR_optimization);
          notify = notify || note;
        }
        else
          task->regions[idx].early_map = false;
      }
      return notify;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::select_task_variant(Task *task)
    //--------------------------------------------------------------------------
    {
      task->selected_variant = select_task_variant(task, task->target_proc);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        task->regions[idx].blocking_factor = 
          select_region_layout(task, task->target_proc, 
                               task->regions[idx], idx, Memory::NO_MEMORY,
                               task->regions[idx].max_blocking_factor);
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        task->regions[idx].virtual_map = map_region_virtually(task,
                                    task->target_proc, task->regions[idx], idx);
        if (!task->regions[idx].virtual_map)
        {
          bool note = map_task_region(task, task->target_proc,
                                      task->regions[idx].tag, false/*inline*/,
                                      false/*premap*/, task->regions[idx], idx,
                                      task->regions[idx].current_instances,
                                      task->regions[idx].target_ranking,
                                      task->regions[idx].additional_fields,
                                    task->regions[idx].enable_WAR_optimization);
          task->regions[idx].reduction_list = 
            select_reduction_layout(task, task->target_proc,
                                    task->regions[idx], idx, 
                                    Memory::NO_MEMORY);
          notify = notify || note;
        }
      }
      return notify;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_copy(Copy *copy)
    //--------------------------------------------------------------------------
    {
      assert(copy->src_requirements.size() == copy->dst_requirements.size());
      for (unsigned idx = 0; idx < copy->src_requirements.size(); idx++)
      {
        copy->src_requirements[idx].virtual_map = false; 
        map_task_region(copy->parent_task, local_proc,
                        copy->src_requirements[idx].tag,
                        false/*inline*/, false/*premap*/,
                        copy->src_requirements[idx], idx,
                        copy->src_requirements[idx].current_instances,
                        copy->src_requirements[idx].target_ranking,
                        copy->src_requirements[idx].additional_fields,
                        copy->src_requirements[idx].enable_WAR_optimization);
        copy->src_requirements[idx].blocking_factor = 
          select_region_layout(copy->parent_task, local_proc,
                               copy->src_requirements[idx], idx, 
                               Memory::NO_MEMORY,
                               copy->src_requirements[idx].max_blocking_factor);
        copy->src_requirements[idx].reduction_list = 
          select_reduction_layout(copy->parent_task, local_proc,
                                  copy->src_requirements[idx], idx,
                                  Memory::NO_MEMORY);
        copy->dst_requirements[idx].virtual_map = false; 
        map_task_region(copy->parent_task, local_proc,
                        copy->dst_requirements[idx].tag,
                        false/*inline*/, false/*premap*/,
                        copy->dst_requirements[idx], idx,
                        copy->dst_requirements[idx].current_instances,
                        copy->dst_requirements[idx].target_ranking,
                        copy->dst_requirements[idx].additional_fields,
                        copy->dst_requirements[idx].enable_WAR_optimization);
        copy->dst_requirements[idx].blocking_factor = 
          select_region_layout(copy->parent_task, local_proc,
                               copy->dst_requirements[idx], idx, 
                               Memory::NO_MEMORY,
                               copy->dst_requirements[idx].max_blocking_factor);
        copy->dst_requirements[idx].reduction_list = 
          select_reduction_layout(copy->parent_task, local_proc,
                                  copy->dst_requirements[idx], idx,
                                  Memory::NO_MEMORY);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_inline(Inline *inline_op)
    //--------------------------------------------------------------------------
    {
      inline_op->requirement.virtual_map = false;
      bool notify = map_task_region(inline_op->parent_task, 
                                 local_proc, inline_op->requirement.tag,
                                 true/*inline*/, false/*premap*/,
                                 inline_op->requirement, 0/*idx*/,
                                 inline_op->requirement.current_instances,
                                 inline_op->requirement.target_ranking,
                                 inline_op->requirement.additional_fields,
                               inline_op->requirement.enable_WAR_optimization);
      inline_op->requirement.blocking_factor = 
        select_region_layout(inline_op->parent_task, local_proc,
                             inline_op->requirement, 0/*idx*/, 
                             Memory::NO_MEMORY, 
                             inline_op->requirement.max_blocking_factor);
      inline_op->requirement.reduction_list = 
        select_reduction_layout(inline_op->parent_task, local_proc,
                                inline_op->requirement, 0/*idx*/,
                                Memory::NO_MEMORY);
      return notify;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_mapping_result(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      Task *task = mappable->as_mappable_task();
      if (task != NULL)
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          notify_mapping_result(task, task->target_proc,
                                task->regions[idx], idx, false/*inline*/,
                                task->regions[idx].selected_memory);
        }
      }
      Inline *op = mappable->as_mappable_inline();
      if (op != NULL)
      {
        notify_mapping_result(op->parent_task, local_proc,
                              op->requirement, 0/*idx*/, true/*inline*/,
                              op->requirement.selected_memory);
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_mapping_failed(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      Task *task = mappable->as_mappable_task();
      if (task != NULL)
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          if (task->regions[idx].mapping_failed)
          {
            notify_failed_mapping(task, task->target_proc,
                                  task->regions[idx], idx, false/*inline*/);
          }
        }
      }
      Inline *in_op = mappable->as_mappable_inline();
      if (in_op != NULL)
      {
        notify_failed_mapping(in_op->parent_task, local_proc,
                              in_op->requirement, 0/*idx*/, true/*inline*/);
      }
      Copy *copy_op = mappable->as_mappable_copy();
      if (copy_op != NULL)
      {
        for (unsigned idx = 0; idx < copy_op->src_requirements.size(); idx++)
        {
          if (copy_op->src_requirements[idx].mapping_failed)
          {
            notify_failed_mapping(copy_op->parent_task, local_proc,
                                  copy_op->src_requirements[idx], idx, false);
          }
          if (copy_op->dst_requirements[idx].mapping_failed)
          {
            notify_failed_mapping(copy_op->parent_task, local_proc,
                                  copy_op->dst_requirements[idx], idx, false);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete, size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse, 
                                     std::vector<Memory> &to_create,
                                     bool &create_one, size_t &blocking_factor)
    //--------------------------------------------------------------------------
    {
      Task *task = mappable->as_mappable_task();
      if (task == NULL)
      {
        Inline *in_op = mappable->as_mappable_inline();
        if (in_op != NULL)
        {
          rank_copy_targets(in_op->parent_task, local_proc, 
                            in_op->requirement.tag, true/*inline*/,
                            in_op->requirement, 0/*idx*/,
                            current_instances, to_reuse, to_create, create_one);
          blocking_factor = select_region_layout(in_op->parent_task, local_proc,
                                                 in_op->requirement, 0/*idx*/,
                                                 Memory::NO_MEMORY,
                                                 max_blocking_factor);
        }
        else
        {
          Copy *copy_op = mappable->as_mappable_copy();
          // Find a region with the same tree ID  
          // Not completely safe since we could have multiple fields
          // with different privileges on the same tree, but it is
          // close enough for now
          for (unsigned idx = 0; idx < copy_op->src_requirements.size(); idx++)
          {
            if (copy_op->src_requirements[idx].region.get_tree_id() ==
                rebuild_region.get_tree_id())
            {
              rank_copy_targets(copy_op->parent_task, local_proc, 
                                copy_op->src_requirements[idx].tag,
                                false/*inline*/, 
                                copy_op->src_requirements[idx], idx,
                                current_instances, to_reuse, to_create, 
                                create_one);
              blocking_factor = select_region_layout(copy_op->parent_task, 
                                           local_proc,
                                           copy_op->src_requirements[idx], idx,
                                           Memory::NO_MEMORY, 
                                           max_blocking_factor);
              return false;
            }
            else if (copy_op->dst_requirements[idx].region.get_tree_id() == 
                rebuild_region.get_tree_id())
            {
              rank_copy_targets(copy_op->parent_task, local_proc, 
                                copy_op->dst_requirements[idx].tag,
                                false/*inline*/, 
                                copy_op->dst_requirements[idx], idx,
                                current_instances, to_reuse, to_create, 
                                create_one);
              blocking_factor = select_region_layout(copy_op->parent_task,
                                           local_proc,
                                           copy_op->dst_requirements[idx], idx,
                                           Memory::NO_MEMORY, 
                                           max_blocking_factor);
              return false;
            }
          }
          // should never get here
          assert(false);
        }
      }
      else 
      {
        // Find a region with the same tree ID  
        // Not completely safe since we could have multiple fields
        // with different privileges on the same tree, but it is
        // close enough for now
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          if (((task->regions[idx].handle_type == PART_PROJECTION) &&
               (task->regions[idx].partition.get_tree_id() ==
                rebuild_region.get_tree_id())) ||
              (((task->regions[idx].handle_type == SINGULAR) || 
                (task->regions[idx].handle_type == REG_PROJECTION)) &&
               (task->regions[idx].region.get_tree_id() ==
                rebuild_region.get_tree_id())))
          {
            rank_copy_targets(task, task->target_proc, task->regions[idx].tag,
                              false/*inline*/, task->regions[idx], idx,
                              current_instances, to_reuse, to_create,
                              create_one);
            blocking_factor = select_region_layout(task, task->target_proc,
                                                   task->regions[idx], idx,
                                                   Memory::NO_MEMORY,
                                                   max_blocking_factor);
            return false;
          }
        }
        // should never get here
        assert(false);
      }
      // No composite instances
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::rank_copy_sources(const Mappable *mappable,
                                      const std::set<Memory> &current_instances,
                                      Memory dst_mem, 
                                      std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------
    {
      rank_copy_sources(current_instances, dst_mem, chosen_order);
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_profiling_info(const Task *task)
    //--------------------------------------------------------------------------
    {
      Mapper::ExecutionProfile profiling;
      profiling.start_time = task->start_time;
      profiling.stop_time = task->stop_time;
      notify_profiling_info(task, task->target_proc, profiling);
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::speculate_on_predicate(const Mappable *m, bool &spec_value)
    //--------------------------------------------------------------------------
    {
      return speculate_on_predicate(m->tag, spec_value);
    }


    //--------------------------------------------------------------------------
    void ShimMapper::select_tasks_to_schedule(
            const std::list<Task*> &ready_tasks, std::vector<bool> &ready_mask)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select tasks to schedule in shim mapper for "
                          "processor " IDFMT "", local_proc.id);
      if (breadth_first_traversal)
      {
        // TODO: something with some feedback pressure based on profiling
        unsigned count = 0;
        for (std::vector<bool>::iterator it = ready_mask.begin();
              (count < max_schedule_count) && (it != ready_mask.end()); it++)
        {
          *it = true; 
          count++;
        }
      }
      else
      {
        // Find the deepest task, and mark valid until tasks at that depth
        // until we're done or need to go to the next depth
        unsigned max_depth = 0;
        for (std::list<Task*>::const_iterator it = ready_tasks.begin();
              it != ready_tasks.end(); it++)
        {
          if ((*it)->depth > max_depth)
            max_depth = (*it)->depth;
        }
        unsigned count = 0;
        // Only schedule tasks from the max_depth in any pass
        unsigned idx = 0;
        for (std::list<Task*>::const_iterator it = ready_tasks.begin();
            (count < max_schedule_count) && 
            (it != ready_tasks.end()); it++, idx++)
        {
          if ((*it)->depth == max_depth)
          {
            //printf("Scheduling task %s\n",(*it)->variants->name);
            ready_mask[idx] = true;
            count++;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::spawn_task(const Task *task)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Spawn task %s (ID %lld) in shim mapper "
                          "for processor " IDFMT "",
                           task->variants->name, 
                           task->get_unique_task_id(), local_proc.id);
      return true;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_task_locally(const Task *task)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Map task %s (ID %lld) locally in shim mapper "
                          "for processor " IDFMT "",
                           task->variants->name, 
                           task->get_unique_task_id(), local_proc.id);
      return false;
    }

    //--------------------------------------------------------------------------
    Processor ShimMapper::select_target_processor(const Task *task)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select target processor for task %s (ID %lld) "
                          "in shim mapper for processor " IDFMT "",
                          task->variants->name, 
                          task->get_unique_task_id(), local_proc.id);
      // See if the profiler has profiling results for all the different 
      // kinds of processors that the task has variants for
      if (profiler.profiling_complete(task))
      {
        Processor::Kind best_kind = profiler.best_processor_kind(task);
        // If our local processor is the right kind then do that
        if (best_kind == local_kind)
          return local_proc;
        // Otherwise select a random processor of the right kind
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = 
          select_random_processor(all_procs, best_kind, machine);
        return result;
      }
      else
      {
        // Get the next kind to find
        Processor::Kind next_kind = profiler.next_processor_kind(task);
        if (next_kind == local_kind)
          return local_proc;
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = 
          select_random_processor(all_procs, next_kind, machine);
        return result;
      }
      log_shim(LEVEL_ERROR,"Couldn't find variants for task %s (ID %lld)",
                          task->variants->name, task->get_unique_task_id());
      // Should never get here, this means we have a task that only has
      // variants for processors that don't exist anywhere in the system.
      assert(false);
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------
    Processor ShimMapper::target_task_steal(
                                          const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Target task steal in shim mapper for "
                          "processor " IDFMT "",local_proc.id);
      if (stealing_enabled)
      {
        // Choose a random processor from our group that is 
        // not on the blacklist
        std::set<Processor> diff_procs; 
        std::set<Processor> all_procs = machine->get_all_processors();
        // Remove ourselves
        all_procs.erase(local_proc);
        std::set_difference(all_procs.begin(),all_procs.end(),
                            blacklist.begin(),blacklist.end(),
                            std::inserter(diff_procs,diff_procs.end()));
        if (diff_procs.empty())
        {
          return Processor::NO_PROC;
        }
        unsigned index = (lrand48()) % (diff_procs.size());
        for (std::set<Processor>::const_iterator it = diff_procs.begin();
              it != diff_procs.end(); it++)
        {
          if (!index--)
          {
            log_shim(LEVEL_SPEW,"Attempting a steal from processor " IDFMT " "
                                "on processor " IDFMT "",local_proc.id,it->id);
            return *it;
          }
        }
        // Should never make it here, the runtime shouldn't call us if 
        // the blacklist is all procs
        assert(false);
      }
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------
    VariantID ShimMapper::select_task_variant(const Task *task, 
                                              Processor target)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select task variant for task %s (ID %lld) "
                            "in shim mapper for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = machine->get_processor_kind(target);
      if (!task->variants->has_variant(target_kind, 
            !(task->is_index_space), task->is_index_space))
      {
        log_shim(LEVEL_ERROR,"Mapper unable to find variant for task %s "
                             "(ID %lld)", task->variants->name, 
                             task->get_unique_task_id());
        assert(false);
      }
      return task->variants->get_variant(target_kind, !(task->is_index_space), 
                                          task->is_index_space);
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_region_virtually(const Task *task, Processor target,
                                   const RegionRequirement &req, unsigned index)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Map region virtually for task %s (ID %lld) "
                          "in shim mapper for processor " IDFMT "",
                          task->variants->name, 
                          task->get_unique_task_id(), local_proc.id);
      return false;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::map_task_region(const Task *task, Processor target, 
                                     MappingTagID tag, 
                                     bool inline_mapping, bool pre_mapping,
                                     const RegionRequirement &req, 
                                     unsigned index,
        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                     std::vector<Memory> &target_ranking,
                                     std::set<FieldID> &additional_fields,
                                     bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Map task region in shim mapper for region ? "
                          "of task %s (ID %lld) for processor " IDFMT "", 
                          task->variants->name, 
                          task->get_unique_task_id(), local_proc.id);
      enable_WAR_optimization = war_enabled;
      if (inline_mapping)
      {
        machine_interface.find_memory_stack(target, 
                                            target_ranking, true/*latency*/);
        return false;
      }
      if (pre_mapping) /* premapping means there is no processor */
      {
        Memory global = machine_interface.find_global_memory();
        assert(global.exists());
        target_ranking.push_back(global);
        return false;
      }
      // Check to see if our memoizer already has mapping for us to use
      if (memoizer.has_mapping(target, task, index))
      {
        memoizer.recall_mapping(target, task, index, target_ranking);
        return true;
      }
      // Otherwise, get our processor stack
      machine_interface.find_memory_stack(target, target_ranking, 
                (machine->get_processor_kind(target) == Processor::LOC_PROC));
      memoizer.record_mapping(target, task, index, target_ranking);
      return true;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_mapping_result(const Task *task, Processor target,
                                           const RegionRequirement &req,
                                           unsigned index, bool inline_mapping, 
                                           Memory result)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Task %s mapped region requirement for "
                          "index %d to memory " IDFMT "",
                          task->variants->name, index, result.id);
      memoizer.notify_mapping(target, task, index, result); 
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_failed_mapping(const Task *task, Processor target,
                                           const RegionRequirement &req,
                                           unsigned index, bool inline_mapping)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_ERROR,"Notify failed mapping for task %s (ID %lld) "
                           "in shim mapper for processor " IDFMT "",
                           task->variants->name, 
                           task->get_unique_task_id(), local_proc.id);
      assert(false);
    }

    //--------------------------------------------------------------------------
    size_t ShimMapper::select_region_layout(const Task *task, Processor target,
                                            const RegionRequirement &req, 
                                            unsigned index, 
                                            const Memory &chosen_mem, 
                                            size_t max_blocking_factor)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select region layout for task %s (ID %lld) "
                          "in shim mapper for processor " IDFMT "",
                           task->variants->name, 
                           task->get_unique_task_id(), local_proc.id);
      if(!target.exists()) {
	log_shim(LEVEL_INFO,"Asked to select region layout for NO_PROC - "
                            "using local proc's processor type");
	target = local_proc;
      }
      if (machine->get_processor_kind(target) == Processor::TOC_PROC)
        return max_blocking_factor;
      return 1;
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::select_reduction_layout(const Task *task, 
                                              const Processor target,
                                              const RegionRequirement &req, 
                                              unsigned index,
                                              const Memory &chosen_mem)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select reduction layout for task %s (ID %lld) "
                          "in shim mapper for processor " IDFMT "",
                          task->variants->name, 
                          task->get_unique_task_id(), local_proc.id);
      // Always do foldable reduction instances if we can
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::rank_copy_targets(const Task *task, Processor target,
                                       MappingTagID tag, bool inline_mapping,
                                       const RegionRequirement &req, 
                                       unsigned index,
                                     const std::set<Memory> &current_instances,
                                       std::set<Memory> &to_reuse,
                                       std::vector<Memory> &to_create,
                                       bool &create_one)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Rank copy targets for task %s (ID %lld) in "
                          "shim mapper for processor " IDFMT "",
                           task->variants->name, 
                           task->get_unique_task_id(), local_proc.id);
      if (current_instances.empty())
      {
        // Pick the global memory
        Memory global = machine_interface.find_global_memory();
        assert(global.exists());
        to_create.push_back(global);
        // Only make one new instance
        create_one = true;
      }
      else
      {
        to_reuse.insert(current_instances.begin(),current_instances.end());
      }
    }

    //--------------------------------------------------------------------------
    void ShimMapper::rank_copy_sources(
                                      const std::set<Memory> &current_instances,
                                      const Memory &dst, 
                                      std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Select copy source in shim mapper for "
                          "processor " IDFMT "", local_proc.id);
      // Handle the simple case of having the destination memory 
      // in the set of instances 
      if (current_instances.find(dst) != current_instances.end())
      {
        chosen_order.push_back(dst);
        return;
      }

      machine_interface.find_memory_stack(dst, chosen_order, true/*latency*/);
      if (chosen_order.empty())
      {
        // This is the multi-hop copy because none of 
        // the memories had an affinity
        // SJT: just send the first one
        if(current_instances.size() > 0) {
          chosen_order.push_back(*(current_instances.begin()));
        } else {
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::profile_task_execution(const Task *task, Processor target)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Profile task execution for task %s (UID %lld) "
                          "on processor " IDFMT "",
                          task->variants->name, 
                          task->get_unique_task_id(), target.id);
      if (!profiler.profiling_complete(task))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void ShimMapper::notify_profiling_info(const Task *task, Processor target,
                                           const ExecutionProfile &profiling)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Notify profiling info for task %s (UID %lld) "
                          "on processor " IDFMT "", task->variants->name, 
                          task->get_unique_task_id(), target.id);
      memoizer.commit_mapping(target, task);
      profiler.update_profiling_info(task, target, 
          machine->get_processor_kind(target), profiling);
    }

    //--------------------------------------------------------------------------
    bool ShimMapper::speculate_on_predicate(MappingTagID tag, 
                                               bool &speculative_value)
    //--------------------------------------------------------------------------
    {
      log_shim(LEVEL_SPEW,"Speculate on predicate in shim mapper "
                          "for processor " IDFMT "", local_proc.id);
      return false;
    }

  };
};

