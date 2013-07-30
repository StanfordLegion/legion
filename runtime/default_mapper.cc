/* Copyright 2013 Stanford University
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
        breadth_first_traversal(STATIC_BREADTH_FIRST),
        war_enabled(STATIC_WAR_ENABLED),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT),
        machine_interface(MappingUtilities::MachineQueryInterface(m))
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Initializing the default mapper for processor %x",local_proc.id);
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
            (count < max_schedule_count) && (it != ready_tasks.end()); it++, idx++)
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
      // If this is a leaf task, map it locally, otherwise map it remotely
      if (task->variants->leaf)
        return true;
      return false;
    }

    //--------------------------------------------------------------------------------------------
    Processor DefaultMapper::select_target_processor(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select target processor for task %s (ID %d) in default mapper for processor %x",
                  task->variants->name, task->get_unique_task_id(), local_proc.id);
      // See if the profiler has profiling results for all the different kinds of
      // processors that the task has variants for
      if (profiler.profiling_complete(task))
      {
        Processor::Kind best_kind = profiler.best_processor_kind(task);
        // If our local processor is the right kind then do that
        if (best_kind == local_kind)
          return local_proc;
        // Otherwise select a random processor of the right kind
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = select_random_processor(all_procs, best_kind, machine);
        return result;
      }
      else
      {
        // Get the next kind to find
        Processor::Kind next_kind = profiler.next_processor_kind(task);
        if (next_kind == local_kind)
          return local_proc;
        const std::set<Processor> &all_procs = machine->get_all_processors();
        Processor result = select_random_processor(all_procs, next_kind, machine);
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
                    if (((reg_it1->handle_type == SINGULAR) || (reg_it1->handle_type == REG_PROJECTION)) &&
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
    void DefaultMapper::slice_domain(const Task *task, const Domain &domain,
                                   std::vector<Mapper::DomainSplit> &slices)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Slice index space in default mapper for task %s (ID %d) for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);

      const std::set<Processor> &all_procs = machine->get_all_processors();
      std::vector<Processor> procs(all_procs.begin(),all_procs.end());

      DefaultMapper::decompose_index_space(domain, procs, splitting_factor, slices);
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
    bool DefaultMapper::map_task_region(const Task *task, Processor target, MappingTagID tag, 
                                        bool inline_mapping, bool pre_mapping,
                                        const RegionRequirement &req, unsigned index,
                                        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                        std::vector<Memory> &target_ranking,
                                        std::set<FieldID> &additional_fields,
                                        bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Map task region in default mapper for region ? of task %s (ID %d) "
                 "for processor %x", task->variants->name, task->get_unique_task_id(), local_proc.id);
      enable_WAR_optimization = war_enabled;
      if (inline_mapping)
      {
        machine_interface.find_memory_stack(target, target_ranking, true/*latency*/);
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

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::notify_mapping_result(const Task *task, Processor target,
                                              const RegionRequirement &req,
                                              unsigned index, bool inline_mapping, Memory result)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Task %s mapped region requirement for index %d to memory %x",
                              task->variants->name, index, result.id);
      memoizer.notify_mapping(target, task, index, result); 
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::notify_failed_mapping(const Task *task, Processor target,
                                              const RegionRequirement &req,
                                              unsigned index, bool inline_mapping)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_ERROR,"Notify failed mapping for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      assert(false);
    }

    //--------------------------------------------------------------------------------------------
    size_t DefaultMapper::select_region_layout(const Task *task, Processor target,
                                               const RegionRequirement &req, unsigned index,
                                               const Memory &chosen_mem, size_t max_blocking_factor)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Select region layout for task %s (ID %d) in default mapper for processor %x",
                 task->variants->name, task->get_unique_task_id(), local_proc.id);
      if(!target.exists()) {
	log_mapper.info("Asked to select region layout for NO_PROC - using local proc's processor type");
	target = local_proc;
      }
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

      machine_interface.find_memory_stack(dst, chosen_order, true/*latency*/);
      if (chosen_order.empty())
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
    bool DefaultMapper::profile_task_execution(const Task *task, Processor target)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Profile task execution for task %s (UID %d) on processor %x",
                            task->variants->name, task->get_unique_task_id(), target.id);
      if (!profiler.profiling_complete(task))
        return true;
      return false;
    }

    //--------------------------------------------------------------------------------------------
    void DefaultMapper::notify_profiling_info(const Task *task, Processor target,
                                              const ExecutionProfile &profiling)
    //--------------------------------------------------------------------------------------------
    {
      log_mapper(LEVEL_SPEW,"Notify profiling info for task %s (UID %d) on processor %x",
                            task->variants->name, task->get_unique_task_id(), target.id);
      memoizer.commit_mapping(target, task);
      profiler.update_profiling_info(task, target, machine->get_processor_kind(target), profiling);
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

    template <unsigned DIM>
    static void round_robin_point_assign(const Domain &domain, const std::vector<Processor> &targets,
					 unsigned splitting_factor, std::vector<Mapper::DomainSplit> &slices)
    {
      Arrays::Rect<DIM> r = domain.get_rect<DIM>();

      std::vector<Processor>::const_iterator target_it = targets.begin();
      for(Arrays::GenericPointInRectIterator<DIM> pir(r); pir; pir++) {
	Arrays::Rect<DIM> subrect(pir.p, pir.p); // rect containing a single point
	Mapper::DomainSplit ds(Domain::from_rect<DIM>(subrect), *target_it++, false /* recurse */, false /* stealable */);
	slices.push_back(ds);
	if(target_it == targets.end())
	  target_it = targets.begin();
      }
    }

    //--------------------------------------------------------------------------------------------
    /*static*/ void DefaultMapper::decompose_index_space(const Domain &domain, const std::vector<Processor> &targets,
                                                         unsigned splitting_factor, std::vector<Mapper::DomainSplit> &slices)
    //--------------------------------------------------------------------------------------------
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
          slices.push_back(Mapper::DomainSplit(Domain(IndexSpace::create_index_space(index_space, chunks[chunk])),
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
        unsigned elmts_per_chunk = (num_elmts+(num_chunks-1))/num_chunks;
        for (unsigned idx = 0; idx < num_chunks; idx++)
        {
          Arrays::Point<1> lo(idx*elmts_per_chunk);  
          Arrays::Point<1> hi((((idx+1)*elmts_per_chunk > num_elmts) ? num_elmts : (idx+1)*elmts_per_chunk)-1);
          Arrays::Rect<1> chunk(lo,hi);
          unsigned proc_idx = idx % targets.size();
          slices.push_back(DomainSplit(Domain::from_rect<1>(chunk), targets[proc_idx], false, false));
        }
      }
    }

  };
};
