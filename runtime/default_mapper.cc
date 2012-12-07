/* Copyright 2012 Stanford University
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
#define STATIC_WAR_ENABLED            true
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8

// This is the default implementation of the mapper interface for the general low level runtime

namespace LegionRuntime {
  namespace HighLevel {

    Logger::Category log_mapper("defmapper");

    //--------------------------------------------------------------------------------------------
    DefaultMapper::DefaultMapper(Machine *m, HighLevelRuntime *rt, Processor local) 
      : runtime(rt), local_proc(local), local_kind(m->get_processor_kind(local)), machine(m),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        splitting_factor(STATIC_SPLIT_FACTOR),
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default mapper for processor %x",local_proc.id);
      // Get the kind of processor that this mapper is managing
      const std::set<Processor> &locals = m->get_local_processors(local); 
      for (std::set<Processor>::const_iterator it = locals.begin();
            it != locals.end(); it++)
      {
        local_procs[*it] = machine->get_processor_kind(*it);
      }
      // Check to see if there any input arguments to parse
      {
        int argc = HighLevelRuntime::get_input_args().argc;
        char **argv = HighLevelRuntime::get_input_args().argv;
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
          INT_ARG("-dm:sched", max_schedule_count);
#undef BOOL_ARG
#undef INT_ARG
        }
      }
      // Now we're going to build our memory stacks
      for (std::map<Processor,Processor::Kind>::const_iterator it = local_procs.begin();
            it != local_procs.end(); it++)
      {
        // Optimize CPUs for latency and GPUs for throughput
        compute_memory_stack(it->first, memory_stacks[it->first], machine, (it->second == Processor::TOC_PROC));
        assert(!memory_stacks[it->first].empty());
      }
      // Now build our set of similar processors and our alternative processor map
      {
        const std::set<Processor> &all_procs = machine->get_all_processors();
        for (std::set<Processor>::const_iterator it = all_procs.begin();
              it != all_procs.end(); it++)
        {
          other_procs[*it] = machine->get_processor_kind(*it);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    DefaultMapper::~DefaultMapper(void)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Deleting default mapper for processor %x",local_proc.id);
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::select_tasks_to_schedule(const std::list<Task*> &ready_tasks,
                                                 std::vector<bool> &ready_mask)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select tasks to schedule in default mapper for processor %x",
                 local_proc.id);
      // TODO: something with some feedback pressure based on profiling
      unsigned count = 0;
      for (std::vector<bool>::iterator it = ready_mask.begin();
            (count < max_schedule_count) && (it != ready_mask.end()); it++)
      {
        *it = true; 
      }
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::spawn_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Spawn task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      return true;
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::map_task_locally(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task %s (ID %d) locally in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      return false;
    }

    //--------------------------------------------------------------------------------------------
    Processor DefaultMapper::select_target_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select target processor for task %s (ID %d) in default mapper for processor %x",
                  task->variants->name, task->get_unique_task_id(), local_proc.id);
      // Check to see if we can run a variant of it locally
      if (task->variants->has_variant(local_kind, task->is_index_space))
        return local_proc;
      // Otherwise pick another try a different processor kind, prefer GPUs over CPUs
      if (task->variants->has_variant(Processor::TOC_PROC, task->is_index_space))
      {
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = select_random_processor(all_procs, Processor::TOC_PROC, machine);
        if (result.exists())
          return result;
      }
      // Otherwise try CPUs
      if (task->variants->has_variant(Processor::LOC_PROC, task->is_index_space))
      {
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = select_random_processor(all_procs, Processor::LOC_PROC, machine);
        if (result.exists())
          return result;
      }
      log_mapper(LEVEL_ERROR,"Mapper couldn't find variants for task %s (ID %d)",
                              task->variants->name, task->get_unique_task_id());
      // Should never get here, this means we have a task that only has
      // variants for processors that don't exist anywhere in the system.
      assert(false);
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------------------------
    Processor DefaultMapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Target task steal in default mapper for processor %x",local_proc.id);
      if (stealing_enabled)
      {
        // Choose a random processor from our group that is not on the blacklist
        std::set<Processor> diff_procs; 
        std::set<Processor> all_procs = machine->get_all_processors();
        // Remove ourselves
        all_procs.erase(local_proc);
        std::set_difference(all_procs.begin(),all_procs.end(),
                            blacklist.begin(),blacklist.end(),std::inserter(diff_procs,diff_procs.end()));
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
            log_mapper(LEVEL_SPEW,"Attempting a steal from processor %x on processor %x",local_proc.id,it->id);
            return *it;
          }
        }
        // Should never make it here, the runtime shouldn't call us if the blacklist is all procs
        assert(false);
      }
      return Processor::NO_PROC;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                          std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Permit task steal in default mapper for processor %x",local_proc.id);

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
            log_mapper(LEVEL_DEBUG,"Task %s (ID %d) stolen from processor %x by processor %x",
                       (*it)->variants->name, (*it)->get_unique_task_id(), local_proc.id, thief.id);
            to_steal.insert(*it);
            total_stolen++;
            // Check to see if we're done
            if (total_stolen == max_steals_per_theft)
              return;
            // If not, do locality aware task stealing, try to steal other tasks that use
            // the same logical regions.  Don't need to worry about all the tasks we've already
            // seen since we either stole them or decided not for some reason
            for (std::vector<const Task*>::const_iterator inner_it = it;
                  inner_it != tasks.end(); inner_it++)
            {
              // Check to make sure this task hasn't been stolen too much already
              if ((*inner_it)->steal_count >= max_steal_count)
                continue;
              // Check to make sure it's not one of the tasks we've already stolen
              if (to_steal.find(*inner_it) != to_steal.end())
                continue;
              // If its not the same check to see if they have any of the same logical regions
              for (std::vector<RegionRequirement>::const_iterator reg_it1 = (*it)->regions.begin();
                    reg_it1 != (*it)->regions.end(); reg_it1++)
              {
                bool shared = false;
                for (std::vector<RegionRequirement>::const_iterator reg_it2 = (*inner_it)->regions.begin();
                      reg_it2 != (*inner_it)->regions.end(); reg_it2++)
                {
                  // Check to make sure they have the same type of region requirement, and that
                  // the region (or partition) is the same.
                  if (reg_it1->handle_type == reg_it2->handle_type)
                  {
                    if ((reg_it1->handle_type == SINGULAR) &&
                        (reg_it1->region == reg_it2->region))
                    {
                      shared = true;
                      break;
                    }
                    if ((reg_it1->handle_type == PROJECTION) &&
                        (reg_it1->partition == reg_it2->partition))
                    {
                      shared = true;
                      break;
                    }
                  }
                }
                if (shared)
                {
                  log_mapper(LEVEL_DEBUG,"Task %s (ID %d) stolen from processor %x by processor %x",
                             (*inner_it)->variants->name, (*inner_it)->get_unique_task_id(), local_proc.id,
                             thief.id);
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

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::slice_index_space(const Task *task, const IndexSpace &index_space,
                                   std::vector<Mapper::IndexSplit> &slices)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Slice index space in default mapper for task %s (ID %d) for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);

      const std::set<Processor> &all_procs = machine->get_all_processors();
      std::vector<Processor> procs(all_procs.begin(),all_procs.end());

      DefaultMapper::decompose_index_space(index_space, procs, splitting_factor, slices);
    }

    //--------------------------------------------------------------------------------------------
    VariantID DefaultMapper::select_task_variant(const Task *task, Processor target)
    //--------------------------------------------------------------------------------------------
    {
      Processor::Kind target_kind = machine->get_processor_kind(target);
      if (!task->variants->has_variant(target_kind, task->is_index_space))
      {
        log_mapper(LEVEL_ERROR,"Mapper unable to find variant for task %s (ID %d)",
                                task->variants->name, task->get_unique_task_id());
        assert(false);
      }
      return task->variants->get_variant(target_kind, task->is_index_space);
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::map_region_virtually(const Task *task, Processor target,
                                             const RegionRequirement &req, unsigned index)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map region virtually for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::map_task_region(const Task *task, Processor target, MappingTagID tag, bool inline_mapping,
                                        const RegionRequirement &req, unsigned index,
                                        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                        std::vector<Memory> &target_ranking,
                                        bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task region in default mapper for region ? of task %s (ID %d) "
                 "for processor %x", task->variants->name, task->get_unique_task_id(), local_proc.id);
      // Just give our processor stack
      target_ranking = memory_stacks[target];
      enable_WAR_optimization = war_enabled;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::notify_failed_mapping(const Task *task, Processor target,
                                              const RegionRequirement &req,
                                              unsigned index, bool inline_mapping)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Notify failed mapping for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      assert(false);
    }

    //--------------------------------------------------------------------------------------------
    size_t DefaultMapper::select_region_layout(const Task *task, const Processor target,
                                               const RegionRequirement &req, unsigned index,
                                               const Memory &chosen_mem, size_t max_blocking_factor)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select region layout for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      if (machine->get_processor_kind(target) == Processor::TOC_PROC)
        return max_blocking_factor;
      return 1;
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::select_reduction_layout(const Task *task, const Processor target,
                                                const RegionRequirement &req, unsigned index,
                                                const Memory &chosen_mem)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select reduction layout for task %s (ID %d) in default mapper for processor %x",
                  task->variants->name, task->get_unique_task_id(), local_proc.id);
      // Always do foldable reduction instances if we can
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::rank_copy_targets(const Task *task, Processor target,
                                          MappingTagID tag, bool inline_mapping,
                                          const RegionRequirement &req, unsigned index,
                                          const std::set<Memory> &current_instances,
                                          std::set<Memory> &to_reuse,
                                          std::vector<Memory> &to_create,
                                          bool &create_one)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Rank copy targets for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      if (current_instances.empty())
      {
        assert(memory_stacks.find(target) != memory_stacks.end());
        to_create = memory_stacks[target];
        // Only make one new instance
        create_one = true;
      }
      else
      {
        to_reuse.insert(current_instances.begin(),current_instances.end());
      }
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::rank_copy_sources(const std::set<Memory> &current_instances,
                                          const Memory &dst, std::vector<Memory> &chosen_order)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select copy source in default mapper for processor %x", local_proc.id);
      // Handle the simple case of having the destination memory in the set of instances 
      if (current_instances.find(dst) != current_instances.end())
      {
        chosen_order.push_back(dst);
        return;
      }

      // Pick the one with the best memory-memory bandwidth
      // TODO: handle the case where we need a multi-hop copy
      bool found = false;
      unsigned max_band = 0;
      for (std::set<Memory>::const_iterator it = current_instances.begin();
           it != current_instances.end(); it++)
      {
        std::vector<Machine::MemoryMemoryAffinity> affinities;
        int size = machine->get_mem_mem_affinity(affinities, *it, dst);
        log_mapper(LEVEL_SPEW,"memory %x has %d affinities", it->id, size);
        if (size > 0)
        {
          if (!found)
          {
            found = true;
            max_band = affinities[0].bandwidth;
            chosen_order.push_back(*it);
          }
          else
          {
            if (affinities[0].bandwidth > max_band)
            {
              max_band = affinities[0].bandwidth;
              chosen_order.push_back(*it);
            }
          }
          }
      }
      // Make sure that we always set a value
      if (!found)
      {
        // This is the multi-hop copy because none of the memories had an affinity
        // SJT: just send the first one
        if(current_instances.size() > 0) {
          chosen_order.push_back(*(current_instances.begin()));
        } else {
          assert(false);
        }
      }
    }

    //--------------------------------------------------------------------------------------------
    bool DefaultMapper::speculate_on_predicate(MappingTagID tag, bool &speculative_value)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Speculate on predicate in default mapper for processor %x",
                 local_proc.id);
      return false;
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ void DefaultMapper::compute_memory_stack(Processor target_proc, std::vector<Memory> &result,
                                                        Machine *machine, bool bandwidth /*= true*/)
    //--------------------------------------------------------------------------------------------
    {
      // First get the set of memories that we can see from this processor
      const std::set<Memory> &visible = machine->get_visible_memories(target_proc);
      std::list<std::pair<Memory,unsigned/*bandwidth/latency*/> > temp_stack;
      // Go through each of the memories
      for (std::set<Memory>::const_iterator it = visible.begin();
            it != visible.end(); it++)
      {
        // Insert the memory into our list
        {
          std::vector<Machine::ProcessorMemoryAffinity> local_affin;
          int size = machine->get_proc_mem_affinity(local_affin,target_proc,*it);
          assert(size == 1);
          // Sort the memory into list based on bandwidth 
          bool inserted = false;
          if (bandwidth)
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it = temp_stack.begin();
                  stack_it != temp_stack.end(); stack_it++)
            {
              if (local_affin[0].bandwidth > stack_it->second)
              {
                inserted = true;
                temp_stack.insert(stack_it,std::pair<Memory,unsigned>(*it,local_affin[0].bandwidth));
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(std::pair<Memory,unsigned>(*it,local_affin[0].bandwidth));
          }
          else
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it = temp_stack.begin();
                  stack_it != temp_stack.end(); stack_it++)
            {
              if (local_affin[0].latency < stack_it->second)
              {
                inserted = true;
                temp_stack.insert(stack_it,std::pair<Memory,unsigned>(*it,local_affin[0].latency));
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(std::pair<Memory,unsigned>(*it,local_affin[0].latency));
          }
        }
      }
      // Now dump the temp stack into the actual stack
      for (std::list<std::pair<Memory,unsigned> >::const_iterator it = temp_stack.begin();
            it != temp_stack.end(); it++)
      {
        result.push_back(it->first);
      }
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ Processor DefaultMapper::select_random_processor(const std::set<Processor> &options, 
                                          Processor::Kind filter, Machine *machine)
    //--------------------------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------------------------
    /*static*/ void DefaultMapper::decompose_index_space(const IndexSpace &index_space, const std::vector<Processor> &targets,
                                                         unsigned splitting_factor, std::vector<Mapper::IndexSplit> &slices)
    //--------------------------------------------------------------------------------------------
    {
      // This assumes the IndexSpace is 1-dimensional and split it according to the splitting factor.
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
        int remaining_in_chunk = num_elts_per_chunk + (chunk < num_elts_extra ? 1 : 0);
        while (enabled->get_next(position, length)) {
          for (; chunk < num_chunks; chunk++,
                 remaining_in_chunk = num_elts_per_chunk + (chunk < num_elts_extra ? 1 : 0)) {
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

      for (unsigned chunk = 0; chunk < num_chunks; chunk++) {
        // TODO: Come up with a better way of distributing work across the processor groups
        slices.push_back(Mapper::IndexSplit(IndexSpace::create_index_space(index_space, chunks[chunk]),
                                   targets[(chunk % targets.size())], false, false));
      }
    }

  };
};
