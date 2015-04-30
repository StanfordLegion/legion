/* Copyright 2015 Stanford University
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


#include "mapping_utilities.h"

namespace LegionRuntime {
  namespace HighLevel {
    namespace MappingUtilities {

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
        stack.insert(stack.end(),visible.begin(),visible.end());
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
        stack.insert(stack.end(),visible.begin(),visible.end());
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
          assert(machine.get_proc_mem_affinity(affinity, proc, *it) == 1);
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
      void MappingProfiler::set_max_profiling_samples(unsigned max)
      //------------------------------------------------------------------------
      {
        if (max > 0)
          max_samples = max;
      }
      
      //------------------------------------------------------------------------
      bool MappingProfiler::profiling_complete(const Task *task) const
      //------------------------------------------------------------------------
      {
        TaskMap::const_iterator finder = task_profiles.find(task->task_id);
        if (finder == task_profiles.end())
          return false;
        for (VariantMap::const_iterator it = finder->second.begin();
              it != finder->second.end(); it++)
        {
          if (it->second.execution_times.size() < max_samples)
            return false;
        }
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
                        float(it->second.execution_times.size());
            best_kind = it->first;
            best_set = true;
          }
          else
          {
            float time = float(it->second.total_time)/
                         float(it->second.execution_times.size());
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
      Processor::Kind MappingProfiler::next_processor_kind(
                                                        const Task *task) const
      //------------------------------------------------------------------------
      {
        TaskMap::const_iterator finder = task_profiles.find(task->task_id);
        if (finder == task_profiles.end())
          return task->variants->get_all_variants().begin()->second.proc_kind;
        for (VariantMap::const_iterator it = finder->second.begin();
              it != finder->second.end(); it++)
        {
          if (it->second.execution_times.size() < max_samples)
            return it->first;
        }
        return best_processor_kind(task);
      }

      //------------------------------------------------------------------------
      void MappingProfiler::update_profiling_info(const Task *task, 
                                                  Processor target, 
                                                  Processor::Kind kind,
                                        const Mapper::ExecutionProfile &profile)
      //------------------------------------------------------------------------
      {
        TaskMap::iterator finder = task_profiles.find(task->task_id);
        if (finder == task_profiles.end())
        {
	  const std::map<VariantID,TaskVariantCollection::Variant>& variants = 
            task->variants->get_all_variants();
          for (std::map<VariantID,TaskVariantCollection::Variant>::
               const_iterator it = variants.begin(); it != variants.end(); it++)
          {
            task_profiles[task->task_id][it->second.proc_kind] = 
              VariantProfile();
          }
          finder = task_profiles.find(task->task_id);
        }
        VariantMap::iterator var_finder = finder->second.find(kind);
        assert(var_finder != finder->second.end());
        if (var_finder->second.execution_times.size() == max_samples)
        {
          var_finder->second.total_time -= 
            var_finder->second.execution_times.front();
          var_finder->second.execution_times.pop_front();
        }
        long long total_time = profile.stop_time - profile.start_time;
        var_finder->second.total_time += total_time;
        var_finder->second.execution_times.push_back(total_time);
      }

      //------------------------------------------------------------------------
      MappingProfiler::VariantProfile::VariantProfile(void)
        : total_time(0)
      //------------------------------------------------------------------------
      {
      }

    };
  };
};

// EOF

