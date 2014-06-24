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
#include "default_mapper.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_SPLIT_FACTOR           2
#define STATIC_BREADTH_FIRST          false
#define STATIC_WAR_ENABLED            false 
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8
#define STATIC_NUM_PROFILE_SAMPLES    1

// This is the default implementation of the mapper interface for 
// the general low level runtime

namespace LegionRuntime {
  namespace HighLevel {

    Logger::Category log_mapper("default_mapper");

    //--------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(Machine *m, HighLevelRuntime *rt, 
                                 Processor local) 
      : Mapper(rt), local_proc(local), 
        local_kind(m->get_processor_kind(local)), machine(m),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        splitting_factor(STATIC_SPLIT_FACTOR),
        breadth_first_traversal(STATIC_BREADTH_FIRST),
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT),
        machine_interface(MappingUtilities::MachineQueryInterface(m))
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default mapper for "
                            "processor " IDFMT "",
                 local_proc.id);
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
        unsigned num_profiling_samples = STATIC_NUM_PROFILE_SAMPLES;
        // Parse the input arguments looking for ones for the default mapper
        for (int i=1; i < argc; i++)
        {
#define INT_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], argname)) {  \
            varname = atoi(argv[++i]);      \
            continue;                       \
          } } while(0);
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], argname)) {    \
            varname = (atoi(argv[++i]) != 0); \
            continue;                         \
          } } while(0);
          INT_ARG("-dm:thefts", max_steals_per_theft);
          INT_ARG("-dm:count", max_steal_count);
          INT_ARG("-dm:split", splitting_factor);
          BOOL_ARG("-dm:war", war_enabled);
          BOOL_ARG("-dm:steal", stealing_enabled);
          BOOL_ARG("-dm:bft", breadth_first_traversal);
          INT_ARG("-dm:sched", max_schedule_count);
          INT_ARG("-dm:prof",num_profiling_samples);
#undef BOOL_ARG
#undef INT_ARG
        }
        profiler.set_needed_profiling_samples(num_profiling_samples);
      }
    }

    //--------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(const DefaultMapper &rhs)
      : Mapper(NULL), local_proc(Processor::NO_PROC),
        local_kind(Processor::LOC_PROC), machine(NULL),
        machine_interface(MappingUtilities::MachineQueryInterface(NULL))
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DefaultMapper::~DefaultMapper(void)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Deleting default mapper for processor " IDFMT "",
                  local_proc.id);
    }

    //--------------------------------------------------------------------------
    DefaultMapper& DefaultMapper::operator=(const DefaultMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_options(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select task options in default mapper " 
                            "for proecessor " IDFMT "", local_proc.id);
      task->inline_task = false;
      task->spawn_task = stealing_enabled;
      task->map_locally = false; 
      task->profile_task = !profiler.profiling_complete(task);
      task->task_priority = 0; // No prioritization
      // For selecting a target processor see if we have finished profiling
      // the given task otherwise send it to a processor of the right kind
      if (profiler.profiling_complete(task))
      {
        Processor::Kind best_kind = profiler.best_processor_kind(task);
        // If our local processor is the right kind then do that
        if (best_kind == local_kind)
          task->target_proc = local_proc;
        else
        {
          // Otherwise select a random processor of the right kind
          const std::set<Processor> &all_procs = machine->get_all_processors();
          task->target_proc = select_random_processor(all_procs, best_kind, 
                                                      machine);
        }
      }
      else
      {
        // Get the next kind to find
        Processor::Kind next_kind = profiler.next_processor_kind(task);
        if (next_kind == local_kind)
          task->target_proc = local_proc;
        else
        {
          const std::set<Processor> &all_procs = machine->get_all_processors();
          task->target_proc = select_random_processor(all_procs, 
                                                      next_kind, machine);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_tasks_to_schedule(
                                            const std::list<Task*> &ready_tasks)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select tasks to schedule in default mapper for "
                            "processor " IDFMT "", local_proc.id);
      if (breadth_first_traversal)
      {
        unsigned count = 0; 
        for (std::list<Task*>::const_iterator it = ready_tasks.begin(); 
              (count < max_schedule_count) && (it != ready_tasks.end()); it++)
        {
          (*it)->schedule = true;
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
        for (std::list<Task*>::const_iterator it = ready_tasks.begin();
              (count < max_schedule_count) && (it != ready_tasks.end()); it++)
        {
          if ((*it)->depth == max_depth)
          {
            (*it)->schedule = true;
            count++;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::target_task_steal(const std::set<Processor> &blacklist,
                                          std::set<Processor> &targets)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Target task steal in default mapper for "
                            "processor " IDFMT "",local_proc.id);
      if (stealing_enabled)
      {
        // Choose a random processor from our group that is not on the blacklist
        std::set<Processor> diff_procs; 
        std::set<Processor> all_procs = machine->get_all_processors();
        // Remove ourselves
        all_procs.erase(local_proc);
        std::set_difference(all_procs.begin(),all_procs.end(),
                            blacklist.begin(),blacklist.end(),
                            std::inserter(diff_procs,diff_procs.end()));
        if (diff_procs.empty())
          return;
        unsigned index = (lrand48()) % (diff_procs.size());
        for (std::set<Processor>::const_iterator it = diff_procs.begin();
              it != diff_procs.end(); it++)
        {
          if (!index--)
          {
            log_mapper(LEVEL_SPEW,"Attempting a steal from processor " IDFMT
                                  " on processor " IDFMT "",
                                  local_proc.id,it->id);
            targets.insert(*it);
            break;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::permit_task_steal(Processor thief,
                                          const std::vector<const Task*> &tasks,
                                          std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Permit task steal in default mapper for "
                            "processor " IDFMT "",local_proc.id);

      if (stealing_enabled)
      {
        // First see if we're even allowed to steal anything
        if (max_steals_per_theft == 0)
          return;
        // We're allowed to steal something, go through and find a task to steal
        unsigned total_stolen = 0;
        for (std::vector<const Task*>::const_iterator it = tasks.begin();
              it != tasks.end(); it++)
        {
          if ((*it)->steal_count < max_steal_count)
          {
            log_mapper(LEVEL_DEBUG,"Task %s (ID %lld) stolen from "
                                   "processor " IDFMT " by processor " IDFMT "",
                                   (*it)->variants->name, 
                                   (*it)->get_unique_task_id(), 
                                   local_proc.id, thief.id);
            to_steal.insert(*it);
            total_stolen++;
            // Check to see if we're done
            if (total_stolen == max_steals_per_theft)
              return;
            // If not, do locality aware task stealing, try to steal other 
            // tasks that use the same logical regions.  Don't need to 
            // worry about all the tasks we've already seen since we 
            // either stole them or decided not for some reason
            for (std::vector<const Task*>::const_iterator inner_it = it;
                  inner_it != tasks.end(); inner_it++)
            {
              // Check to make sure this task hasn't 
              // been stolen too much already
              if ((*inner_it)->steal_count >= max_steal_count)
                continue;
              // Check to make sure it's not one of 
              // the tasks we've already stolen
              if (to_steal.find(*inner_it) != to_steal.end())
                continue;
              // If its not the same check to see if they have 
              // any of the same logical regions
              for (std::vector<RegionRequirement>::const_iterator reg_it1 = 
                    (*it)->regions.begin(); reg_it1 != 
                    (*it)->regions.end(); reg_it1++)
              {
                bool shared = false;
                for (std::vector<RegionRequirement>::const_iterator reg_it2 = 
                      (*inner_it)->regions.begin(); reg_it2 != 
                      (*inner_it)->regions.end(); reg_it2++)
                {
                  // Check to make sure they have the same type of region 
                  // requirement, and that the region (or partition) 
                  // is the same.
                  if (reg_it1->handle_type == reg_it2->handle_type)
                  {
                    if (((reg_it1->handle_type == SINGULAR) || 
                          (reg_it1->handle_type == REG_PROJECTION)) &&
                        (reg_it1->region == reg_it2->region))
                    {
                      shared = true;
                      break;
                    }
                    if ((reg_it1->handle_type == PART_PROJECTION) &&
                        (reg_it1->partition == reg_it2->partition))
                    {
                      shared = true;
                      break;
                    }
                  }
                }
                if (shared)
                {
                  log_mapper(LEVEL_DEBUG,"Task %s (ID %lld) stolen from "
                                         "processor " IDFMT " by processor " 
                                         IDFMT "",
                                         (*inner_it)->variants->name, 
                                         (*inner_it)->get_unique_task_id(), 
                                         local_proc.id, thief.id);
                  // Add it to the list of steals and either return or break
                  to_steal.insert(*inner_it);
                  total_stolen++;
                  if (total_stolen == max_steals_per_theft)
                    return;
                  // Otherwise break, onto the next task
                  break;
                }
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::slice_domain(const Task *task, const Domain &domain,
                                     std::vector<DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Slice index space in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name, 
                            task->get_unique_task_id(), local_proc.id);

      Processor::Kind best_kind;
      if (profiler.profiling_complete(task))
        best_kind = profiler.best_processor_kind(task);
      else
        best_kind = profiler.next_processor_kind(task);
      std::set<Processor> all_procs = machine->get_all_processors();
      machine_interface.filter_processors(machine, best_kind, all_procs);
      std::vector<Processor> procs(all_procs.begin(),all_procs.end());

      DefaultMapper::decompose_index_space(domain, procs, 
                                           splitting_factor, slices);
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::pre_map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Pre-map task in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        if (task->regions[idx].must_early_map)
        {
          task->regions[idx].virtual_map = false;
          task->regions[idx].early_map = true;
          task->regions[idx].enable_WAR_optimization = war_enabled;
          task->regions[idx].reduction_list = false;
          task->regions[idx].make_persistent = false;
          task->regions[idx].blocking_factor = 1;
          Memory global = machine_interface.find_global_memory();
          assert(global.exists());
          task->regions[idx].target_ranking.push_back(global);
        }
        else
          task->regions[idx].early_map = false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::select_task_variant(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select task variant in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = 
        machine->get_processor_kind(task->target_proc);
      if (!task->variants->has_variant(target_kind, 
            !(task->is_index_space), task->is_index_space))
      {
        log_mapper(LEVEL_ERROR,"Mapper unable to find variant for "
                               "task %s (ID %lld)",
                               task->variants->name, 
                               task->get_unique_task_id());
        assert(false);
      }
      task->selected_variant = task->variants->get_variant(target_kind,
                                                      !(task->is_index_space),
                                                      task->is_index_space);
      if (target_kind == Processor::LOC_PROC)
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
          task->regions[idx].blocking_factor = 1;
      }
      else
      {
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
          task->regions[idx].blocking_factor = 
            task->regions[idx].max_blocking_factor;
      }
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::map_task(Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task in default mapper for task %s "
                            "(ID %lld) for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), local_proc.id);
      Processor::Kind target_kind = 
        machine->get_processor_kind(task->target_proc);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        // See if this instance is restricted
        if (!task->regions[idx].restricted)
        {
          // Check to see if our memoizer already has mapping for us to use
          if (memoizer.has_mapping(task->target_proc, task, idx))
          {
            memoizer.recall_mapping(task->target_proc, task, idx,
                                    task->regions[idx].target_ranking);
          }
          else
          {
            machine_interface.find_memory_stack(task->target_proc,
                                                task->regions[idx].target_ranking,
                                  (machine->get_processor_kind(task->target_proc) 
                                                        == Processor::LOC_PROC));
            memoizer.record_mapping(task->target_proc, task, idx,
                                    task->regions[idx].target_ranking);
          }
        }
        else
        {
          assert(task->regions[idx].current_instances.size() == 1);
          Memory target = (task->regions[idx].current_instances.begin())->first;
          task->regions[idx].target_ranking.push_back(target);
        }
        task->regions[idx].virtual_map = false;
        task->regions[idx].enable_WAR_optimization = war_enabled;
        task->regions[idx].reduction_list = false;
        task->regions[idx].make_persistent = false;
        if (target_kind == Processor::LOC_PROC)
          task->regions[idx].blocking_factor = 1;
        else
          task->regions[idx].blocking_factor = 
            task->regions[idx].max_blocking_factor;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::map_copy(Copy *copy)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map copy for copy ID %lld in default mapper "
                            "for processor " IDFMT "", 
                            copy->get_unique_copy_id(), local_proc.id);
      std::vector<Memory> local_stack; 
      machine_interface.find_memory_stack(local_proc, local_stack,
                                          (local_kind == Processor::LOC_PROC)); 
      assert(copy->src_requirements.size() == copy->dst_requirements.size());
      for (unsigned idx = 0; idx < copy->src_requirements.size(); idx++)
      {
        copy->src_requirements[idx].virtual_map = false;
        copy->src_requirements[idx].early_map = false;
        copy->src_requirements[idx].enable_WAR_optimization = war_enabled;
        copy->src_requirements[idx].reduction_list = false;
        copy->src_requirements[idx].make_persistent = false;
        if (!copy->src_requirements[idx].restricted)
        {
          copy->src_requirements[idx].target_ranking = local_stack;
        }
        else
        {
          assert(copy->src_requirements[idx].current_instances.size() == 1);
          Memory target = 
            (copy->src_requirements[idx].current_instances.begin())->first;
          copy->src_requirements[idx].target_ranking.push_back(target);
        }
        copy->dst_requirements[idx].virtual_map = false;
        copy->dst_requirements[idx].early_map = false;
        copy->dst_requirements[idx].enable_WAR_optimization = war_enabled;
        copy->dst_requirements[idx].reduction_list = false;
        copy->dst_requirements[idx].make_persistent = false;
        if (!copy->dst_requirements[idx].restricted)
        {
          copy->dst_requirements[idx].target_ranking = local_stack;
        }
        else
        {
          assert(copy->dst_requirements[idx].current_instances.size() == 1);
          Memory target = 
            (copy->dst_requirements[idx].current_instances.begin())->first;
          copy->dst_requirements[idx].target_ranking.push_back(target);
        }
        if (local_kind == Processor::LOC_PROC)
        {
          copy->src_requirements[idx].blocking_factor = 1;
          copy->dst_requirements[idx].blocking_factor = 1;
        }
        else
        {
          copy->src_requirements[idx].blocking_factor = 
            copy->src_requirements[idx].max_blocking_factor;
          copy->dst_requirements[idx].blocking_factor = 
            copy->dst_requirements[idx].max_blocking_factor;
        } 
      }
      // No profiling on copies yet
      return false;
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::map_inline(Inline *inline_op)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map inline for operation ID %lld in default "
                            "mapper for processor " IDFMT "",
                            inline_op->get_unique_inline_id(), local_proc.id);
      inline_op->requirement.virtual_map = false;
      inline_op->requirement.early_map = false;
      inline_op->requirement.enable_WAR_optimization = war_enabled;
      inline_op->requirement.reduction_list = false;
      inline_op->requirement.make_persistent = false;
      if (!inline_op->requirement.restricted)
      {
        machine_interface.find_memory_stack(local_proc, 
                                          inline_op->requirement.target_ranking,
                                          (local_kind == Processor::LOC_PROC));
      }
      else
      {
        assert(inline_op->requirement.current_instances.size() == 1);
        Memory target = 
          (inline_op->requirement.current_instances.begin())->first;
        inline_op->requirement.target_ranking.push_back(target);
      }
      if (local_kind == Processor::LOC_PROC)
        inline_op->requirement.blocking_factor = 1;
      else
        inline_op->requirement.blocking_factor = 
          inline_op->requirement.max_blocking_factor;
      // No profiling on inline mappings
      return false;
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::map_must_epoch(const std::vector<Task*> &tasks,
                             const std::vector<MappingConstraint> &constraints,
                             MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map must epoch in default mapper for processor"
                            " "IDFMT" ",
                            local_proc.id);
      // First fixup any target processors to ensure that they are all
      // pointed at different processors.  We know for now that all must epoch
      // tasks need to be running on CPUs so get the set of CPU processors.
      const std::set<Processor> &all_cpus = 
        machine_interface.filter_processors(Processor::LOC_PROC);
      assert(all_cpus.size() >= tasks.size());
      // Round robing the tasks onto the processors
      std::set<Processor>::const_iterator proc_it = all_cpus.begin();
      for (std::vector<Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++, proc_it++)
      {
        (*it)->target_proc = *proc_it;
      }
      // Map all the tasks like normal, then go through and fix up the
      // mapping requests based on constraints.
      for (std::vector<Task*>::const_iterator it = tasks.begin();
            it != tasks.end(); it++)
      {
        map_task(*it);
      }
      // For right now, we'll put everything in the global memory
      Memory global_mem = machine_interface.find_global_memory();
      assert(global_mem.exists());
      for (std::vector<MappingConstraint>::const_iterator it =
            constraints.begin(); it != constraints.end(); it++)
      {
        it->t1->regions[it->idx1].target_ranking.clear();
        it->t1->regions[it->idx1].target_ranking.push_back(global_mem);
        it->t2->regions[it->idx2].target_ranking.clear();
        it->t2->regions[it->idx2].target_ranking.push_back(global_mem);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::notify_mapping_result(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      // We should only get this for tasks in the default mapper
      if (mappable->get_mappable_kind() == Mappable::TASK_MAPPABLE)
      {
        const Task *task = mappable->as_mappable_task();
        assert(task != NULL);
        log_mapper(LEVEL_SPEW,"Notify mapping for task %s (ID %lld) in "
                              "default mapper for processor " IDFMT "",
                              task->variants->name,
                              task->get_unique_task_id(), local_proc.id);
        for (unsigned idx = 0; idx < task->regions.size(); idx++)
        {
          memoizer.notify_mapping(task->target_proc, task, idx, 
                                  task->regions[idx].selected_memory);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::notify_mapping_failed(const Mappable *mappable)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Notify failed mapping for operation ID %lld "
                            "in default mapper for processor " IDFMT "",
                            mappable->get_unique_mappable_id(), local_proc.id);
      // Do nothing for now and hope things drain out and free
      // up the necessary memory resources to do the mapping
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one, size_t &blocking_factor)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Rank copy targets for mappable (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            mappable->get_unique_mappable_id(), local_proc.id);
      if (current_instances.empty())
      {
        // Pick the global memory
        Memory global = machine_interface.find_global_memory();
        assert(global.exists());
        to_create.push_back(global);
        // Only make one new instance
        create_one = true;
        blocking_factor = 1;
      }
      else
      {
        to_reuse.insert(current_instances.begin(),current_instances.end());
        create_one = false;
        blocking_factor = 1;
      }
      // Don't make any composite instances since they're 
      // not fully supported yet
      return false;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::rank_copy_sources(const Mappable *mappable,
                                      const std::set<Memory> &current_instances,
                                      Memory dst_mem, 
                                      std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select copy source in default mapper for "
                            "processor " IDFMT "", local_proc.id);
      // Handle the simple case of having the destination 
      // memory in the set of instances 
      if (current_instances.find(dst_mem) != current_instances.end())
      {
        chosen_order.push_back(dst_mem);
        return;
      }

      machine_interface.find_memory_stack(dst_mem, 
                                          chosen_order, true/*latency*/);
      if (chosen_order.empty())
      {
        // This is the multi-hop copy because none 
        // of the memories had an affinity
        // SJT: just send the first one
        if(current_instances.size() > 0) {
          chosen_order.push_back(*(current_instances.begin()));
        } else {
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::notify_profiling_info(const Task *task)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Notify profiling info for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name, 
                            task->get_unique_task_id(), task->target_proc.id);
      memoizer.commit_mapping(task->target_proc, task);
      Mapper::ExecutionProfile profiling;
      profiling.start_time = task->start_time;
      profiling.stop_time = task->stop_time;
      profiler.update_profiling_info(task, task->target_proc,
                   machine->get_processor_kind(task->target_proc), profiling);
    }

    //--------------------------------------------------------------------------
    bool DefaultMapper::speculate_on_predicate(const Task *task, 
                                               bool &spec_value)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Speculate on predicate for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), task->target_proc.id);
      // While the runtime supports speculation, it currently doesn't
      // know how to roll back from mis-speculation, so for the moment
      // we don't speculate.
      return false;
    }

    //--------------------------------------------------------------------------
    int DefaultMapper::get_tunable_value(const Task *task, TunableID tid,
                                         MappingTagID tag)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Get tunable value for task %s (ID %lld) in "
                            "default mapper for processor " IDFMT "",
                            task->variants->name,
                            task->get_unique_task_id(), task->target_proc.id);
      // For right now the default mapper doesn't know how to guess
      // for tunable variables, so instead simply assert.  In the future
      // we might consider employing a performance profiling directed
      // approach to guessing for tunable variables.
      assert(false);
      return 0;
    }

    //--------------------------------------------------------------------------
    void DefaultMapper::handle_message(Processor source,
                                       const void *message, size_t length)
    //--------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Handle message in default mapper for processor " 
                            IDFMT "", local_proc.id);
      // We don't send any messages so we should never receive one
      assert(false);
    }

    //--------------------------------------------------------------------------
    /*static*/ Processor DefaultMapper::select_random_processor(
                                            const std::set<Processor> &options, 
                                      Processor::Kind filter, Machine *machine)
    //--------------------------------------------------------------------------
    {
      std::vector<Processor> valid_options;
      for (std::set<Processor>::const_iterator it = options.begin();
            it != options.end(); it++)
      {
        if (machine->get_processor_kind(*it) == filter)
          valid_options.push_back(*it);
      }
      if (!valid_options.empty())
      {
        if (valid_options.size() == 1)
          return valid_options[0];
        unsigned idx = (lrand48()) % valid_options.size();
        return valid_options[idx];
      }
      return Processor::NO_PROC;
    }

    template <unsigned DIM>
    static void round_robin_point_assign(const Domain &domain, 
                                         const std::vector<Processor> &targets,
					 unsigned splitting_factor, 
                                     std::vector<Mapper::DomainSplit> &slices)
    {
      Arrays::Rect<DIM> r = domain.get_rect<DIM>();

      std::vector<Processor>::const_iterator target_it = targets.begin();
      for(Arrays::GenericPointInRectIterator<DIM> pir(r); pir; pir++) 
      {
        // rect containing a single point
	Arrays::Rect<DIM> subrect(pir.p, pir.p); 
	Mapper::DomainSplit ds(Domain::from_rect<DIM>(subrect), *target_it++, 
                               false /* recurse */, false /* stealable */);
	slices.push_back(ds);
	if(target_it == targets.end())
	  target_it = targets.begin();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void DefaultMapper::decompose_index_space(const Domain &domain, 
                                          const std::vector<Processor> &targets,
                                          unsigned splitting_factor, 
                                      std::vector<Mapper::DomainSplit> &slices)
    //--------------------------------------------------------------------------
    {
      switch(domain.get_dim()) {
      case 2:
	round_robin_point_assign<2>(domain, targets, splitting_factor, slices);
	return;

      case 3:
	round_robin_point_assign<3>(domain, targets, splitting_factor, slices);
	return;

	// cases 0 and 1 fall through to old code for now
      }

      // Only handle these two cases right now
      assert((domain.get_dim() == 0) || (domain.get_dim() == 1));
      if (domain.get_dim() == 0)
      {
        IndexSpace index_space = domain.get_index_space();
        // This assumes the IndexSpace is 1-dimensional and split it according 
        // to the splitting factor.
        LowLevel::ElementMask mask = index_space.get_valid_mask();

        // Count valid elements in mask.
        unsigned num_elts = 0;
        {
          LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
          int position = 0, length = 0;
          while (enabled->get_next(position, length)) {
            num_elts += length;
          }
        }

        // Choose split sizes based on number of elements and processors.
        unsigned num_chunks = targets.size() * splitting_factor;
        if (num_chunks > num_elts) {
          num_chunks = num_elts;
        }
        unsigned num_elts_per_chunk = num_elts / num_chunks;
        unsigned num_elts_extra = num_elts % num_chunks;

        std::vector<LowLevel::ElementMask> chunks(num_chunks, mask);
        for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
          LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
          int position = 0, length = 0;
          while (enabled->get_next(position, length)) {
            chunks[chunk].disable(position, length);
          }
        }

        // Iterate through valid elements again and assign to chunks.
        {
          LowLevel::ElementMask::Enumerator *enabled = mask.enumerate_enabled();
          int position = 0, length = 0;
          unsigned chunk = 0;
          int remaining_in_chunk = num_elts_per_chunk + 
            (chunk < num_elts_extra ? 1 : 0);
          while (enabled->get_next(position, length)) {
            for (; chunk < num_chunks; chunk++,
                   remaining_in_chunk = num_elts_per_chunk + 
                   (chunk < num_elts_extra ? 1 : 0)) {
              if (length <= remaining_in_chunk) {
                chunks[chunk].enable(position, length);
                break;
              }
              chunks[chunk].enable(position, remaining_in_chunk);
              position += remaining_in_chunk;
              length -= remaining_in_chunk;
            }
          }
        }

        for (unsigned chunk = 0; chunk < num_chunks; chunk++) 
        {
          // TODO: Come up with a better way of distributing 
          // work across the processor groups
          slices.push_back(Mapper::DomainSplit(
            Domain(IndexSpace::create_index_space(index_space, chunks[chunk])),
                             targets[(chunk % targets.size())], false, false));
        }
      }
      else
      {
        // Only works for one dimensional rectangles right now
        assert(domain.get_dim() == 1);
        Arrays::Rect<1> rect = domain.get_rect<1>();
        unsigned num_elmts = rect.volume();
        unsigned num_chunks = targets.size()*splitting_factor;
        if (num_chunks > num_elmts)
          num_chunks = num_elmts;
        // Number of elements per chunk rounded up
        // which works because we know that rectangles are contiguous
        unsigned lower_bound = num_elmts/num_chunks;
        unsigned upper_bound = lower_bound+1;
        unsigned number_small = num_chunks - (num_elmts % num_chunks);
        unsigned index = 0;
        for (unsigned idx = 0; idx < num_chunks; idx++)
        {
          unsigned elmts = (idx < number_small) ? lower_bound : upper_bound;
          Arrays::Point<1> lo(index);  
          Arrays::Point<1> hi(index+elmts-1);
          index += elmts;
          Arrays::Rect<1> chunk(rect.lo+lo,rect.lo+hi);
          unsigned proc_idx = idx % targets.size();
          slices.push_back(DomainSplit(
                Domain::from_rect<1>(chunk), targets[proc_idx], false, false));
        }
      }
    }

  };
};
