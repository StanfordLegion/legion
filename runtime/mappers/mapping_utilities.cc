/* Copyright 2022 Stanford University, NVIDIA Corporation
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


#include "mappers/mapping_utilities.h"

#include <algorithm>
#include <limits>

namespace Legion {
  namespace Mapping {
    namespace Utilities {

      /**************************
       * Machine Query Interface
       **************************/

      //------------------------------------------------------------------------
      MachineQueryInterface::MachineQueryInterface(Machine m)
        : machine(m), global_memory(Memory::NO_MEMORY) { }
      //------------------------------------------------------------------------

      //------------------------------------------------------------------------
      Memory MachineQueryInterface::find_global_memory(void)
      //------------------------------------------------------------------------
      {
        if (global_memory.exists())
          return global_memory;
        global_memory = MachineQueryInterface::find_global_memory(machine);
        return global_memory;
      }

      //------------------------------------------------------------------------
      /*static*/ Memory MachineQueryInterface::find_global_memory(
                                                              Machine machine)
      //------------------------------------------------------------------------
      {
        Memory global_memory = Memory::NO_MEMORY;
        std::set<Memory> all_memories;
	machine.get_all_memories(all_memories);
        for (std::set<Memory>::const_iterator it = all_memories.begin();
              it != all_memories.end(); it++)
        {
          if (it->kind() == Memory::GLOBAL_MEM)
          {
            global_memory = *it;
            return global_memory;
          }
        }
        // Otherwise check to see if there is a memory that is visible
        // to all of the processors
        std::set<Processor> all_processors;
	machine.get_all_processors(all_processors);
        for (std::set<Memory>::const_iterator mit = all_memories.begin();
              mit != all_memories.end(); mit++)
        {
          std::set<Processor> vis_processors;
	  machine.get_shared_processors(*mit, vis_processors);
          if (vis_processors.size() == all_processors.size())
          {
            global_memory = *mit;
            return global_memory;
          }
        }
        // Otherwise indicate that it doesn't exist
        return global_memory;
      }

      //------------------------------------------------------------------------
      void MachineQueryInterface::find_memory_stack(Processor proc,
                          std::vector<Memory> &stack, bool latency)
      //------------------------------------------------------------------------
      {
        std::map<Processor,std::vector<Memory> >::iterator finder =
                                                  proc_mem_stacks.find(proc);
        if (finder != proc_mem_stacks.end())
        {
          stack = finder->second;
          if (!latency)
            MachineQueryInterface::sort_memories(machine, proc, stack, latency);
          return;
        }
        MachineQueryInterface::find_memory_stack(machine, proc, stack, latency);
        proc_mem_stacks[proc] = stack;
        if (!latency)
          MachineQueryInterface::sort_memories(machine, proc,
                                               proc_mem_stacks[proc], latency);
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::find_memory_stack(Machine machine,
                                                               Processor proc,
                                      std::vector<Memory> &stack, bool latency)
      //------------------------------------------------------------------------
      {
        std::set<Memory> visible;
	machine.get_visible_memories(proc, visible);
	// memories with no capacity should not go in the stack
	for (std::set<Memory>::const_iterator it = visible.begin();
	     it != visible.end();
	     it++)
	{
	  if ((*it).capacity() > 0)
	    stack.push_back(*it);
	}
        MachineQueryInterface::sort_memories(machine, proc, stack, latency);
      }

      //------------------------------------------------------------------------
      void MachineQueryInterface::find_memory_stack(Memory mem,
                          std::vector<Memory> &stack, bool latency)
      //------------------------------------------------------------------------
      {
        std::map<Memory,std::vector<Memory> >::iterator finder =
                                                      mem_mem_stacks.find(mem);
        if (finder != mem_mem_stacks.end())
        {
          stack = finder->second;
          if (!latency)
            MachineQueryInterface::sort_memories(machine, mem, stack, latency);
          return;
        }
        MachineQueryInterface::find_memory_stack(machine, mem, stack, latency);
        mem_mem_stacks[mem] = stack;
        if (!latency)
          MachineQueryInterface::sort_memories(machine, mem,
                                               mem_mem_stacks[mem], latency);
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::find_memory_stack(Machine machine,
                                                               Memory mem,
                                      std::vector<Memory> &stack, bool latency)
      //------------------------------------------------------------------------
      {
        std::set<Memory> visible;
	machine.get_visible_memories(mem, visible);
	// memories with no capacity should not go in the stack
	for (std::set<Memory>::const_iterator it = visible.begin();
	     it != visible.end();
	     it++)
	{
	  if ((*it).capacity() > 0)
	    stack.push_back(*it);
	}
        MachineQueryInterface::sort_memories(machine, mem, stack, latency);
      }

      //------------------------------------------------------------------------
      Memory MachineQueryInterface::find_memory_kind(Processor proc,
                                                     Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        std::pair<Processor,Memory::Kind> key(proc,kind);
        std::map<std::pair<Processor,Memory::Kind>,Memory>::const_iterator
          finder = proc_mem_table.find(key);
        if (finder != proc_mem_table.end())
          return finder->second;
        Memory result = MachineQueryInterface::find_memory_kind(machine,
                                                                proc, kind);
        proc_mem_table[key] = result;
        return result;
      }

      //------------------------------------------------------------------------
      /*static*/ Memory MachineQueryInterface::find_memory_kind(
                            Machine machine, Processor proc, Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        std::set<Memory> visible_memories;
	machine.get_visible_memories(proc, visible_memories);
        for (std::set<Memory>::const_iterator it = visible_memories.begin();
            it != visible_memories.end(); it++)
        {
          if (it->kind() == kind)
            return *it;
        }
        return Memory::NO_MEMORY;
      }

      //------------------------------------------------------------------------
      Memory MachineQueryInterface::find_memory_kind(Memory mem,
                                                     Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        std::pair<Memory,Memory::Kind> key(mem,kind);
        std::map<std::pair<Memory,Memory::Kind>,Memory>::const_iterator
          finder = mem_mem_table.find(key);
        if (finder != mem_mem_table.end())
          return finder->second;
        Memory result = MachineQueryInterface::find_memory_kind(machine,
                                                                mem, kind);
        mem_mem_table[key] = result;
        return result;
      }

      //------------------------------------------------------------------------
      /*static*/ Memory MachineQueryInterface::find_memory_kind(
                                Machine machine, Memory mem, Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        std::set<Memory> visible_memories;
	machine.get_visible_memories(mem, visible_memories);
        for (std::set<Memory>::const_iterator it = visible_memories.begin();
              it != visible_memories.end(); it++)
        {
          if (it->kind() == kind)
            return *it;
        }
        return Memory::NO_MEMORY;
      }

      //------------------------------------------------------------------------
      Processor MachineQueryInterface::find_processor_kind(Memory mem,
                                                           Processor::Kind kind)
      //------------------------------------------------------------------------
      {
        std::pair<Memory,Processor::Kind> key(mem,kind);
        std::map<std::pair<Memory,Processor::Kind>,Processor>::const_iterator
          finder = mem_proc_table.find(key);
        if (finder != mem_proc_table.end())
          return finder->second;
        Processor result = MachineQueryInterface::find_processor_kind(machine,
                                                                    mem, kind);
        mem_proc_table[key] = result;
        return result;
      }

      //------------------------------------------------------------------------
      /*static*/ Processor MachineQueryInterface::find_processor_kind(
                            Machine machine, Memory mem, Processor::Kind kind)
      //------------------------------------------------------------------------
      {
        std::set<Processor> visible_procs;
	machine.get_shared_processors(mem, visible_procs);
        for (std::set<Processor>::const_iterator it = visible_procs.begin();
              it != visible_procs.end(); it++)
        {
          if (it->kind() == kind)
            return *it;
        }
        return Processor::NO_PROC;
      }

      //------------------------------------------------------------------------
      const std::set<Processor>& MachineQueryInterface::filter_processors(
                                                          Processor::Kind kind)
      //------------------------------------------------------------------------
      {
        std::map<Processor::Kind,std::set<Processor> >::const_iterator
          finder = proc_kinds.find(kind);
        if (finder != proc_kinds.end())
          return finder->second;
        std::set<Processor> &result = proc_kinds[kind];
        MachineQueryInterface::filter_processors(machine, kind, result);
        return result;
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::filter_processors(Machine machine,
                                                           Processor::Kind kind,
                                                     std::set<Processor> &procs)
      //------------------------------------------------------------------------
      {
        if (procs.empty())
        {
          std::set<Processor> all_procs;
	  machine.get_all_processors(all_procs);
          for (std::set<Processor>::const_iterator it = all_procs.begin();
                it != all_procs.end(); it++)
          {
            if (it->kind() == kind)
              procs.insert(*it);
          }
        }
        else
        {
          std::vector<Processor> to_delete;
          for (std::set<Processor>::const_iterator it = procs.begin();
                it != procs.end(); it++)
          {
            if (it->kind() != kind)
              to_delete.push_back(*it);
          }
          for (std::vector<Processor>::const_iterator it = to_delete.begin();
                it != to_delete.end(); it++)
          {
            procs.erase(*it);
          }
        }
      }

      //------------------------------------------------------------------------
      const std::set<Memory>& MachineQueryInterface::filter_memories(
                                                              Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        std::map<Memory::Kind,std::set<Memory> >::const_iterator finder =
          mem_kinds.find(kind);
        if (finder != mem_kinds.end())
          return finder->second;
        std::set<Memory> &result = mem_kinds[kind];
        MachineQueryInterface::filter_memories(machine, kind, result);
        return result;
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::filter_memories(Machine machine,
                                                             Memory::Kind kind,
                                                        std::set<Memory> &mems)
      //------------------------------------------------------------------------
      {
        if (mems.empty())
        {
          std::set<Memory> all_mems;
	  machine.get_all_memories(all_mems);
          for (std::set<Memory>::const_iterator it = all_mems.begin();
                it != all_mems.end(); it++)
          {
            if (it->kind() == kind)
              mems.insert(*it);
          }
        }
        else
        {
          std::vector<Memory> to_delete;
          for (std::set<Memory>::const_iterator it = mems.begin();
                it != mems.end(); it++)
          {
            if (it->kind() != kind)
              to_delete.push_back(*it);
          }
          for (std::vector<Memory>::const_iterator it = to_delete.begin();
                it != to_delete.end(); it++)
          {
            mems.erase(*it);
          }
        }
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::sort_memories(Machine machine,
                                                           Processor proc,
                                                  std::vector<Memory> &memories,
                                                            bool latency)
      //------------------------------------------------------------------------
      {
        std::list<std::pair<Memory,unsigned/*bandwidth or latency*/> >
          temp_stack;
        for (std::vector<Memory>::const_iterator it = memories.begin();
              it != memories.end(); it++)
        {
          std::vector<Machine::ProcessorMemoryAffinity> affinity;
#ifndef NDEBUG
          bool result =
#endif
          machine.get_proc_mem_affinity(affinity, proc, *it);
          assert(result == 1);
          bool inserted = false;
          if (latency)
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it =
                  temp_stack.begin(); stack_it != temp_stack.end(); stack_it++)
            {
              if (affinity[0].latency < stack_it->second)
              {
                temp_stack.insert(stack_it,
                    std::pair<Memory,unsigned>(*it,affinity[0].latency));
                inserted = true;
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(
                  std::pair<Memory,unsigned>(*it,affinity[0].latency));
          }
          else /*bandwidth*/
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it =
                  temp_stack.begin(); stack_it != temp_stack.end(); stack_it++)
            {
              if (affinity[0].bandwidth > stack_it->second)
              {
                temp_stack.insert(stack_it,
                    std::pair<Memory,unsigned>(*it,affinity[0].bandwidth));
                inserted = true;
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(
                  std::pair<Memory,unsigned>(*it,affinity[0].bandwidth));
          }
        }
        // Now put the temp stack onto the real stack
        assert(temp_stack.size() == memories.size());
        unsigned idx = 0;
        for (std::list<std::pair<Memory,unsigned> >::const_iterator it =
              temp_stack.begin(); it != temp_stack.end(); it++, idx++)
        {
          memories[idx] = it->first;
        }
      }

      //------------------------------------------------------------------------
      /*static*/ void MachineQueryInterface::sort_memories(Machine machine,
                                                           Memory mem,
                                                  std::vector<Memory> &memories,
                                                           bool latency)
      //------------------------------------------------------------------------
      {
        std::list<std::pair<Memory,unsigned/*bandwidth or latency*/> >
          temp_stack;
        for (std::vector<Memory>::const_iterator it = memories.begin();
              it != memories.end(); it++)
        {
          std::vector<Machine::MemoryMemoryAffinity> affinity;
          int size = machine.get_mem_mem_affinity(affinity, mem, *it);
	  if(size == 0) {
	    // insert a dummy (bad) affinity for when two memories don't
            // actually have affinity (i.e. a multi-hop copy would be necessary)
	    Machine::MemoryMemoryAffinity mma;
	    mma.m1 = mem;
	    mma.m2 = *it;
	    mma.latency = 1000000;
	    mma.bandwidth = 0;
	    affinity.push_back(mma);
	    size++;
	  }
          assert(size == 1);
          bool inserted = false;
          if (latency)
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it =
                  temp_stack.begin(); stack_it != temp_stack.end(); stack_it++)
            {
              if (affinity[0].latency < stack_it->second)
              {
                temp_stack.insert(stack_it,
                    std::pair<Memory,unsigned>(*it,affinity[0].latency));
                inserted = true;
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(
                  std::pair<Memory,unsigned>(*it,affinity[0].latency));
          }
          else /*bandwidth*/
          {
            for (std::list<std::pair<Memory,unsigned> >::iterator stack_it =
                  temp_stack.begin(); stack_it != temp_stack.end(); stack_it++)
            {
              if (affinity[0].bandwidth > stack_it->second)
              {
                temp_stack.insert(stack_it,
                    std::pair<Memory,unsigned>(*it,affinity[0].bandwidth));
                inserted = true;
                break;
              }
            }
            if (!inserted)
              temp_stack.push_back(
                  std::pair<Memory,unsigned>(*it,affinity[0].bandwidth));
          }
        }
        // Now put the temp stack onto the real stack
        assert(temp_stack.size() == memories.size());
        unsigned idx = 0;
        for (std::list<std::pair<Memory,unsigned> >::const_iterator it =
              temp_stack.begin(); it != temp_stack.end(); it++, idx++)
        {
          memories[idx] = it->first;
        }
      }

      /**********************************
       * Mapping Memoizer
       **********************************/

      //------------------------------------------------------------------------
      MappingMemoizer::MappingMemoizer(void)
      //------------------------------------------------------------------------
      {
      }

      //------------------------------------------------------------------------
      bool MappingMemoizer::has_mapping(Processor target, const Task *task,
                                        unsigned index) const
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::const_iterator finder =
          permanent_mappings.find(key);
        if (finder == permanent_mappings.end())
          return false;
        if (index < finder->second.rankings.size())
          return true;
        return false;
      }

      //------------------------------------------------------------------------
      bool MappingMemoizer::recall_mapping(Processor target, const Task *task,
                                           unsigned index,
                                           std::vector<Memory> &ranking) const
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::const_iterator finder =
          permanent_mappings.find(key);
        if (finder == permanent_mappings.end())
          return false;
        if (index < finder->second.rankings.size())
        {
          ranking = finder->second.rankings[index];
          return true;
        }
        return false;
      }

      //------------------------------------------------------------------------
      Memory MappingMemoizer::recall_chosen(Processor target, const Task *task,
                                            unsigned index) const
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::const_iterator finder =
          permanent_mappings.find(key);
        if (finder == permanent_mappings.end())
          return Memory::NO_MEMORY;
        if (index < finder->second.chosen.size())
          return finder->second.chosen[index];
        return Memory::NO_MEMORY;
      }

      //------------------------------------------------------------------------
      void MappingMemoizer::record_mapping(Processor target, const Task *task,
                                           unsigned index,
                                           const std::vector<Memory> &ranking)
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::iterator finder =
          temporary_mappings.find(key);
        if (finder == temporary_mappings.end())
        {
          temporary_mappings[key] = MemoizedMapping(task->regions.size());
          finder = temporary_mappings.find(key);
        }
        if (index < finder->second.rankings.size())
          finder->second.rankings[index] = ranking;
      }

      //------------------------------------------------------------------------
      void MappingMemoizer::notify_mapping(Processor target, const Task *task,
                                           unsigned index, Memory result)
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::iterator finder =
          temporary_mappings.find(key);
        if (finder == temporary_mappings.end())
        {
          temporary_mappings[key] = MemoizedMapping(task->regions.size());
          finder = temporary_mappings.find(key);
        }
        if (index < finder->second.chosen.size())
          finder->second.chosen[index] = result;
      }

      //------------------------------------------------------------------------
      void MappingMemoizer::commit_mapping(Processor target, const Task *task)
      //------------------------------------------------------------------------
      {
        MappingKey key(target,task->task_id);
        std::map<MappingKey,MemoizedMapping>::const_iterator finder =
          temporary_mappings.find(key);
        if (finder == temporary_mappings.end())
          return;
        permanent_mappings.insert(*finder);
      }

      //------------------------------------------------------------------------
      MappingMemoizer::MemoizedMapping::MemoizedMapping(void)
      //------------------------------------------------------------------------
      {
      }

      //------------------------------------------------------------------------
      MappingMemoizer::MemoizedMapping::MemoizedMapping(size_t num_elmts)
        : chosen(std::vector<Memory>(num_elmts,Memory::NO_MEMORY)),
          rankings(std::vector<std::vector<Memory> >(num_elmts))
      //------------------------------------------------------------------------
      {
      }

      /************************
       * Mapping Profiler
       ************************/

      //------------------------------------------------------------------------
      MappingProfiler::MappingProfiler(void)
        : needed_samples(1), max_samples(32)
      //------------------------------------------------------------------------
      {
      }

      //------------------------------------------------------------------------
      void MappingProfiler::set_needed_profiling_samples(unsigned num_samples)
      //------------------------------------------------------------------------
      {
        if (num_samples > 0)
          needed_samples = num_samples;
      }

      //------------------------------------------------------------------------
      void MappingProfiler::set_needed_profiling_samples(
                            Processor::TaskFuncID task_id, unsigned num_samples)
      //------------------------------------------------------------------------
      {
        if (num_samples > 0)
        {
          OptionMap::iterator finder = profiling_options.find(task_id);
          if (finder == profiling_options.end())
          {
            profiling_options[task_id] =
              ProfilingOption(needed_samples, max_samples);
            finder = profiling_options.find(task_id);
          }
          finder->second.needed_samples = num_samples;
        }
      }

      //------------------------------------------------------------------------
      void MappingProfiler::set_max_profiling_samples(unsigned max)
      //------------------------------------------------------------------------
      {
        if (max > 0)
          max_samples = max;
      }

      //------------------------------------------------------------------------
      void MappingProfiler::set_max_profiling_samples(
                                    Processor::TaskFuncID task_id, unsigned max)
      //------------------------------------------------------------------------
      {
        if (max > 0)
        {
          OptionMap::iterator finder = profiling_options.find(task_id);
          if (finder == profiling_options.end())
          {
            profiling_options[task_id] =
              ProfilingOption(needed_samples, max_samples);
            finder = profiling_options.find(task_id);
          }
          finder->second.max_samples = max;
        }
      }

      void MappingProfiler::set_gather_in_original_processor(
          Processor::TaskFuncID task_id, bool flag)
      {
        OptionMap::iterator finder = profiling_options.find(task_id);
        if (finder == profiling_options.end())
        {
          profiling_options[task_id] =
            ProfilingOption(needed_samples, max_samples);
          finder = profiling_options.find(task_id);
        }

        finder->second.gather_in_orig_proc = flag;
      }

      //------------------------------------------------------------------------
      bool MappingProfiler::profiling_complete(const Task *task) const
      //------------------------------------------------------------------------
      {
        unsigned needed_samples_for_this_task = needed_samples;
        {
          OptionMap::const_iterator finder =
            profiling_options.find(task->task_id);
          if (finder != profiling_options.end())
            needed_samples_for_this_task = finder->second.needed_samples;
        }

        TaskMap::const_iterator finder = task_profiles.find(task->task_id);
        if (finder == task_profiles.end() || finder->second.size() == 0)
          return false;
        for (VariantMap::const_iterator it = finder->second.begin();
              it != finder->second.end(); it++)
        {
          if (it->second.samples.size() < needed_samples_for_this_task)
            return false;
        }
        return true;
      }

      //------------------------------------------------------------------------
      bool MappingProfiler::profiling_complete(
                                   const Task *task, Processor::Kind kind) const
      //------------------------------------------------------------------------
      {
        unsigned needed_samples_for_this_task = needed_samples;
        {
          OptionMap::const_iterator finder =
            profiling_options.find(task->task_id);
          if (finder != profiling_options.end())
            needed_samples_for_this_task = finder->second.needed_samples;
        }

        TaskMap::const_iterator finder = task_profiles.find(task->task_id);
        if (finder == task_profiles.end() || finder->second.size() == 0)
          return false;

        VariantMap::const_iterator var_finder = finder->second.find(kind);
        if (var_finder == finder->second.end() ||
            var_finder->second.samples.size() < needed_samples_for_this_task)
            return false;
        return true;
      }

      //------------------------------------------------------------------------
      Processor::Kind MappingProfiler::best_processor_kind(
                                                        const Task *task) const
      //------------------------------------------------------------------------
      {
        bool best_set = false;
        float best_time = 0.f;
        Processor::Kind best_kind = Processor::LOC_PROC;
        TaskMap::const_iterator finder = task_profiles.find(task->task_id);
        assert(finder != task_profiles.end());
        for (VariantMap::const_iterator it = finder->second.begin();
              it != finder->second.end(); it++)
        {
          if (!best_set)
          {
            best_time = float(it->second.total_time)/
                        float(it->second.samples.size());
            best_kind = it->first;
            best_set = true;
          }
          else
          {
            float time = float(it->second.total_time)/
                         float(it->second.samples.size());
            if (time < best_time)
            {
              best_time = time;
              best_kind = it->first;
            }
          }
        }
        assert(best_set);
        return best_kind;
      }

      //------------------------------------------------------------------------
      void MappingProfiler::add_profiling_sample(Processor::TaskFuncID task_id,
                                         const MappingProfiler::Profile& sample)
      //------------------------------------------------------------------------
      {
        unsigned max_samples_for_this_task = max_samples;
        {
          OptionMap::const_iterator finder = profiling_options.find(task_id);
          if (finder != profiling_options.end())
            max_samples_for_this_task = finder->second.max_samples;
        }

        TaskMap::iterator finder = task_profiles.find(task_id);
        if (finder == task_profiles.end())
        {
          task_profiles[task_id] = VariantMap();
          finder = task_profiles.find(task_id);
        }
        Processor::Kind kind = sample.target_processor.kind();
        VariantMap::iterator var_finder = finder->second.find(kind);
        if (var_finder == finder->second.end())
        {
          finder->second[kind] = VariantProfile();
          var_finder = finder->second.find(kind);
        }
        if (var_finder->second.samples.size() == max_samples_for_this_task)
        {
          var_finder->second.total_time -=
            var_finder->second.samples.front().execution_time;
          var_finder->second.samples.pop_front();
        }
        var_finder->second.total_time += sample.execution_time;
        var_finder->second.samples.push_back(sample);
      }

      //------------------------------------------------------------------------
      MappingProfiler::TaskMap MappingProfiler::get_task_profiles() const
      //------------------------------------------------------------------------
      {
        return task_profiles;
      }

      //------------------------------------------------------------------------
      MappingProfiler::VariantMap MappingProfiler::get_variant_profiles(
                                                Processor::TaskFuncID tid) const
      //------------------------------------------------------------------------
      {
        MappingProfiler::TaskMap::const_iterator finder =
          task_profiles.find(tid);
        if (finder != task_profiles.end())
          return finder->second;
        else
          return VariantMap();
      }

      //------------------------------------------------------------------------
      MappingProfiler::VariantProfile MappingProfiler::get_variant_profile(
                          Processor::TaskFuncID tid, Processor::Kind kind) const
      //------------------------------------------------------------------------
      {
        MappingProfiler::TaskMap::const_iterator finder =
          task_profiles.find(tid);
        if (finder != task_profiles.end())
        {
          MappingProfiler::VariantMap::const_iterator var_finder =
            finder->second.find(kind);
          if (var_finder != finder->second.end())
            return var_finder->second;
          else
            return VariantProfile();
        }
        else
          return VariantProfile();
      }

      //------------------------------------------------------------------------
      MappingProfiler::ProfilingOption MappingProfiler::get_profiling_option(
                                                Processor::TaskFuncID tid) const
      //------------------------------------------------------------------------
      {
        MappingProfiler::OptionMap::const_iterator finder =
          profiling_options.find(tid);
        if (finder != profiling_options.end())
          return finder->second;
        else
          return ProfilingOption();
      }

      //------------------------------------------------------------------------
      void MappingProfiler::clear_samples(Processor::TaskFuncID task_id)
      //------------------------------------------------------------------------
      {
        MappingProfiler::TaskMap::iterator finder = task_profiles.find(task_id);
        if (finder != task_profiles.end())
        {
          for (MappingProfiler::VariantMap::iterator it =
               finder->second.begin(); it != finder->second.end(); ++it)
          {
            it->second.samples.clear();
            it->second.total_time = 0;
          }
        }
      }

      //------------------------------------------------------------------------
      void MappingProfiler::clear_samples(Processor::TaskFuncID task_id,
                                                           Processor::Kind kind)
      //------------------------------------------------------------------------
      {
        MappingProfiler::TaskMap::iterator finder = task_profiles.find(task_id);
        if (finder != task_profiles.end())
        {
          MappingProfiler::VariantMap::iterator var_finder =
            finder->second.find(kind);
          if (var_finder != finder->second.end())
          {
            var_finder->second.samples.clear();
            var_finder->second.total_time = 0;
          }
        }
      }

      template<typename T>
      static bool compare_second(const std::pair<T, double>& pair1,
                                 const std::pair<T, double>& pair2)
      {
        return pair1.second > pair2.second;
      }

      //------------------------------------------------------------------------
      MappingProfiler::AssignmentMap MappingProfiler::get_balanced_assignments(
                      Processor::TaskFuncID task_id, Processor::Kind kind) const
      //------------------------------------------------------------------------
      {
        using namespace std;
        MappingProfiler::TaskMap::const_iterator finder =
          task_profiles.find(task_id);
        if (finder == task_profiles.end())
          return MappingProfiler::AssignmentMap();

        MappingProfiler::VariantMap::const_iterator var_finder =
          finder->second.find(kind);
        if (var_finder == finder->second.end())
          return MappingProfiler::AssignmentMap();

        const VariantProfile& profile = var_finder->second;

        // calculate average execution times
        map<Processor, double> total_exec_times;
        map<DomainPoint, pair<double, int> > exec_times;
        for (list<Profile>::const_iterator it = profile.samples.begin();
             it != profile.samples.end(); ++it)
        {
          if (exec_times.find(it->index_point) == exec_times.end())
          {
            exec_times[it->index_point].first = it->execution_time;
            exec_times[it->index_point].second = 1;
          }
          else
          {
            exec_times[it->index_point].first += it->execution_time;
            exec_times[it->index_point].second += 1;
          }
          if (total_exec_times.find(it->target_processor) ==
              total_exec_times.end())
            total_exec_times[it->target_processor] = 0;
        }

        // sort points by their average execution times
        vector<pair<DomainPoint, double> > avg_exec_times;
        for (map<DomainPoint, pair<double, int> >::iterator it =
             exec_times.begin(); it != exec_times.end(); ++it)
        {
          double avg_exec_time = it->second.first / it->second.second;
          avg_exec_times.push_back(
              pair<DomainPoint, double>(it->first, avg_exec_time));
        }
        sort(avg_exec_times.begin(), avg_exec_times.end(),
             compare_second<DomainPoint>);

        // LPT scheduling
        MappingProfiler::AssignmentMap assignmentMap;
        for (unsigned int i = 0; i < avg_exec_times.size(); ++i)
        {
          map<Processor, double>::iterator finder =
            max_element(total_exec_times.begin(), total_exec_times.end(),
                        compare_second<Processor>);
          pair<DomainPoint, double>& p = avg_exec_times[i];

          assignmentMap[finder->first].push_back(p.first);
          finder->second += p.second;
        }
        return assignmentMap;
      }

      //------------------------------------------------------------------------
      MappingProfiler::AssignmentMap MappingProfiler::get_balanced_assignments(
                                            Processor::TaskFuncID task_id) const
      //------------------------------------------------------------------------
      {
        using namespace std;

        MappingProfiler::TaskMap::const_iterator finder =
          task_profiles.find(task_id);
        if (finder == task_profiles.end())
          return MappingProfiler::AssignmentMap();

        const MappingProfiler::VariantMap& varMap = finder->second;

        set<DomainPoint> points;
        map<Processor::Kind, pair<double, int> > exec_times;
        map<Processor, double> total_exec_times;

        // calculate average execution times
        for (MappingProfiler::VariantMap::const_iterator it = varMap.begin();
             it != varMap.end(); ++it)
        {
          Processor::Kind kind = it->first;
          const VariantProfile& profile = it->second;
          exec_times[kind] = pair<double, int>(0.0, 0);
          for (list<Profile>::const_iterator it2 = profile.samples.begin();
               it2 != profile.samples.end(); ++it2)
          {
            exec_times[kind].first += it2->execution_time;
            exec_times[kind].second++;
            total_exec_times[it2->target_processor] = 0;
            points.insert(it2->index_point);
          }
          exec_times[kind].first /= exec_times[kind].second;
        }

        MappingProfiler::AssignmentMap assignmentMap;
        for (set<DomainPoint>::iterator it = points.begin(); it != points.end();
             it++)
        {
          double min_exec_time = numeric_limits<double>::max();
          map<Processor, double>::iterator finder = total_exec_times.end();
          for (map<Processor, double>::iterator it2 = total_exec_times.begin();
               it2 != total_exec_times.end(); ++it2)
          {
            double exec_time =
              it2->second + exec_times[it2->first.kind()].first;
            if (exec_time < min_exec_time)
            {
              finder = it2;
              min_exec_time = exec_time;
            }
          }

          assert(finder != total_exec_times.end());
          assignmentMap[finder->first].push_back(*it);
          finder->second = min_exec_time;
        }
        return assignmentMap;
      }

      //------------------------------------------------------------------------
      MappingProfiler::VariantProfile::VariantProfile(void)
        : total_time(0)
      //------------------------------------------------------------------------
      {
      }

      //------------------------------------------------------------------------
      MappingProfiler::ProfilingOption::ProfilingOption(void)
        : needed_samples(1), max_samples(32), gather_in_orig_proc(false)
      //------------------------------------------------------------------------
      {
      }

      //------------------------------------------------------------------------
      MappingProfiler::ProfilingOption::ProfilingOption(
                                unsigned needed_samples_, unsigned max_samples_)
        : needed_samples(needed_samples_), max_samples(max_samples_),
          gather_in_orig_proc(false)
      //------------------------------------------------------------------------
      {
      }

      /**********************************
       * Printing functions
       **********************************/

      //------------------------------------------------------------------------
      const char* to_string(Processor::Kind kind)
      //------------------------------------------------------------------------
      {
        switch (kind) {
          case Processor::NO_KIND: return "NO_KIND";
          case Processor::TOC_PROC: return "TOC_PROC";
          case Processor::LOC_PROC: return "LOC_PROC";
          case Processor::UTIL_PROC: return "UTIL_PROC";
          case Processor::IO_PROC: return "IO_PROC";
          case Processor::PROC_GROUP: return "PROC_GROUP";
          case Processor::PROC_SET: return "PROC_SET";
          case Processor::OMP_PROC: return "OMP_PROC";
          case Processor::PY_PROC: return "PY_PROC";
          default: assert(false); return "";
        }
      }

      //------------------------------------------------------------------------
      const char* to_string(Memory::Kind kind)
      //------------------------------------------------------------------------
      {
        switch (kind) {
          case Memory::NO_MEMKIND: return "NO_MEMKIND";
          case Memory::GLOBAL_MEM: return "GLOBAL_MEM";
          case Memory::SYSTEM_MEM: return "SYSTEM_MEM";
          case Memory::REGDMA_MEM: return "REGDMA_MEM";
          case Memory::SOCKET_MEM: return "SOCKET_MEM";
          case Memory::Z_COPY_MEM: return "Z_COPY_MEM";
          case Memory::GPU_FB_MEM: return "GPU_FB_MEM";
          case Memory::DISK_MEM: return "DISK_MEM";
          case Memory::HDF_MEM: return "HDF_MEM";
          case Memory::FILE_MEM: return "FILE_MEM";
          case Memory::LEVEL3_CACHE: return "LEVEL3_CACHE";
          case Memory::LEVEL2_CACHE: return "LEVEL2_CACHE";
          case Memory::LEVEL1_CACHE: return "LEVEL1_CACHE";
          case Memory::GPU_MANAGED_MEM: return "GPU_MANAGED_MEM";
          case Memory::GPU_DYNAMIC_MEM: return "GPU_DYNAMIC_MEM";
          default: assert(false); return "";
        }
      }

      //------------------------------------------------------------------------
      const char* to_string(PrivilegeMode priv)
      //------------------------------------------------------------------------
      {
        switch (priv) {
          case LEGION_NO_ACCESS: return "NO_ACCESS";
          case LEGION_READ_ONLY: return "READ_ONLY";
          case LEGION_WRITE_PRIV: return "WRITE_PRIV";
          case LEGION_REDUCE: return "REDUCE";
          case LEGION_READ_WRITE: return "READ_WRITE";
          case LEGION_WRITE_ONLY: return "WRITE_ONLY";
          case LEGION_WRITE_DISCARD: return "WRITE_DISCARD";
          default: assert(false); return "";
        }
      }

      //------------------------------------------------------------------------
      const char* to_string(CoherenceProperty prop)
      //------------------------------------------------------------------------
      {
        switch (prop) {
          case LEGION_EXCLUSIVE: return "EXCLUSIVE";
          case LEGION_ATOMIC: return "ATOMIC";
          case LEGION_SIMULTANEOUS: return "SIMULTANEOUS";
          case LEGION_RELAXED: return "RELAXED";
          default: assert(false); return "";
        }
      }

      //------------------------------------------------------------------------
      template<int DIM>
      static std::string to_string(const DomainT<DIM>& dom)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        bool past_first = false;
        for (RectInDomainIterator<DIM> it(dom); it.valid(); it.step()) {
          if (past_first) {
            ss << "+";
          } else {
            past_first = true;
          }
          ss << *it;
        }
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Domain& dom)
      //------------------------------------------------------------------------
      {
        switch (dom.get_dim()) {
#define CASE(N) case N: return to_string<N>(dom);
          LEGION_FOREACH_N(CASE)
#undef CASE
          default: assert(false);
        }
        return "";
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            LogicalRegion lr)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << "(" << lr.get_tree_id() << ",("
           << lr.get_index_space().get_id() << ","
           << lr.get_index_space().get_tree_id() << "),"
           << lr.get_field_space().get_id() << ")";
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            IndexSpace is)
      //------------------------------------------------------------------------
      {
        std::vector<Domain> domains;
        runtime->get_index_space_domains(ctx, is, domains);
        std::stringstream ss;
        bool past_first = false;
        for (std::vector<Domain>::iterator it = domains.begin();
             it != domains.end(); ++it) {
          if (past_first) {
            ss << "+";
          } else {
            past_first = true;
          }
          ss << to_string(runtime, ctx, *it);
        }
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const LayoutConstraintSet& constraints)
      //------------------------------------------------------------------------
      {
        const std::vector<DimensionKind>& dims =
          constraints.ordering_constraint.ordering;
        std::stringstream ss;
        if (dims.front() == LEGION_DIM_F) {
          ss << "AoS:";
        } else if (dims.back() == LEGION_DIM_F) {
          ss << "SoA:";
        } else {
          return "other";
        }
        for (std::vector<DimensionKind>::const_reverse_iterator rit =
               dims.rbegin(); rit != dims.rend(); ++rit) {
          switch(*rit) {
            case LEGION_DIM_X: ss << "X"; break;
            case LEGION_DIM_Y: ss << "Y"; break;
            case LEGION_DIM_Z: ss << "Z"; break;
            case LEGION_DIM_F: break;
            default: return "other";
          }
        }
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            FieldSpace fs,
                            const std::set<FieldID>& fields)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        bool past_first = false;
        for (std::set<FieldID>::const_iterator it = fields.begin();
             it != fields.end(); ++it) {
          if (past_first) {
            ss << "+";
          } else {
            past_first = true;
          }
          const void* name;
          size_t name_size;
          if (runtime->retrieve_semantic_information(
                  ctx, fs, *it, LEGION_NAME_SEMANTIC_TAG, name, name_size,
                  true/*can_fail*/, false/*wait_until_ready*/)) {
            ss << static_cast<const char*>(name);
          } else {
            ss << *it;
          }
        }
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            PhysicalInstance inst)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << "Instance";
        if (inst.is_virtual_instance()) {
          ss << "(VIRTUAL)";
          return ss.str();
        }
        ss << "[" << std::hex << inst.get_instance_id() << std::dec << "](";
        if (inst.is_reduction_instance()) {
          ss << "REDUCTION,";
        }
        if (inst.is_external_instance()) {
          ss << "EXTERNAL,";
        }
        if (inst.is_collective_instance()) {
          ss << "COLLECTIVE,";
        }
        ss << "region=(" << inst.get_tree_id() << ",*,"
           << inst.get_field_space().get_id() << ")";
        ss << ",memory=" << inst.get_location();
        ss << ",domain=" << to_string(runtime, ctx, inst.get_instance_domain());
        std::set<FieldID> fields;
        inst.get_fields(fields);
        ss << ",fields=" << to_string(runtime, ctx, inst.get_field_space(), fields);
        const LayoutConstraintSet& constraints =
            runtime->find_layout_constraints(ctx, inst.get_layout_id());
        ss << ",layout=" << to_string(runtime, ctx, constraints);
        ss << ")";
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const RegionRequirement& req,
                            unsigned req_idx)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << "Requirement";
        ss << "[" << req_idx << "]";
        ss << "(privilege=" << to_string(req.privilege);
        if (req.is_restricted()) {
          ss << ",RESTRICTED";
        }
        if (req.prop != LEGION_EXCLUSIVE) {
          ss << ",prop=" << to_string(req.prop);
        }
        LogicalRegion lr =
          req.region.exists() ? req.region
          : runtime->get_parent_logical_region(ctx, req.partition);
        IndexSpace is = lr.get_index_space();
        FieldSpace fs = lr.get_field_space();
        ss << ",region=" << to_string(runtime, ctx, lr);
        ss << ",domain=" << to_string(runtime, ctx, is);
        ss << ",fields=" << to_string(runtime, ctx, fs, req.privilege_fields);
        ss << ")";
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Task& task,
                            bool include_index_point)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << task.get_task_name();
        if (include_index_point && task.is_index_space) {
          ss << "(index_point=" << task.index_point << ")";
        }
        ss << "<" << task.get_unique_id() << ">";
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const InlineMapping& inline_op)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << "InlineMapping" << "<" << inline_op.get_unique_id() << ">";
        return ss.str();
      }

      //------------------------------------------------------------------------
      std::string to_string(MapperRuntime* runtime,
                            const MapperContext ctx,
                            const Copy& copy,
                            bool include_index_point)
      //------------------------------------------------------------------------
      {
        std::stringstream ss;
        ss << "Copy";
        if (include_index_point && copy.is_index_space) {
          ss << "(index_point=" << copy.index_point << ")";
        }
        ss << "<" << copy.get_unique_id() << ">";
        return ss.str();
      }
    }; // namespace Utilities
  }; // namespace Mapping
}; // namespace Legion

// EOF
