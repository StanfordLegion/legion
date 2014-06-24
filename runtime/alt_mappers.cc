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


#include "alt_mappers.h"
#include "utilities.h"
#include <cstdlib>

namespace LegionRuntime {
  namespace HighLevel {
    //////////////////////////////////////
    // Debug Mapper
    //////////////////////////////////////

    Logger::Category log_debug("debugmapper");

    //--------------------------------------------------------------------------------------------
    DebugMapper::DebugMapper(Machine *m, HighLevelRuntime *rt, Processor local)
      : DefaultMapper(m,rt,local)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Initializing the debug mapper on processor %x",local_proc.id);
    }

    //--------------------------------------------------------------------------------------------
    bool DebugMapper::spawn_child_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Spawn child task %s (ID %d) in debug mapper on processor %x",
          task->variants->name, task->task_id, local_proc.id);
      // Still need to be spawned so we can choose a processor
      return true; 
    }

    //--------------------------------------------------------------------------------------------
    Processor DebugMapper::target_task_steal(const std::set<Processor> &blacklist)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Target task steal in debug mapper on processor %x",local_proc.id);
      // Don't perform any stealing
      return Processor::NO_PROC; 
    }

    //--------------------------------------------------------------------------------------------
    void DebugMapper::permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                                          std::set<const Task*> &to_steal)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Permit task steal in debug mapper on processor %x",local_proc.id);
      // Do nothing
    }

    //--------------------------------------------------------------------------------------------
    bool DebugMapper::map_task_region(const Task *task, Processor target, 
                                      MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                      const RegionRequirement &req, unsigned index,
                                      const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                      std::vector<Memory> &target_ranking,
                                      std::set<FieldID> &additional_fields,
                                      bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Map task region in debug mapper for region (%x,%x,%d) of task %s (ID %d) "
          "(unique id %d) on processor %x",req.region.get_index_space().id,req.region.get_field_space().get_id(),
          req.region.get_tree_id(), task->variants->name,
          task->task_id, task->get_unique_task_id(), local_proc.id);
      // Always move things into the last memory in our stack
      std::vector<Memory> memory_stack;
      machine_interface.find_memory_stack(target, memory_stack, 
          (machine->get_processor_kind(target) == Processor::LOC_PROC));
      assert(!memory_stack.empty());
      target_ranking.push_back(memory_stack.back());
      enable_WAR_optimization = false;
      return true;
    }

    //--------------------------------------------------------------------------------------------
    void DebugMapper::rank_copy_targets(const Task *task, Processor target,
                                        MappingTagID tag, bool inline_mapping,
                                        const RegionRequirement &req, unsigned index,
                                        const std::set<Memory> &current_instances,
                                        std::set<Memory> &to_reuse,
                                        std::vector<Memory> &to_create,
                                        bool &create_one)
    //--------------------------------------------------------------------------------------------
    {
      log_debug(LEVEL_SPEW,"Rank copy targets in debug mapper for task %s (ID %d) (unique id %d) "
          "on processor %x", task->variants->name, task->task_id, task->get_unique_task_id(), local_proc.id);
      // Always map things into the last memory in our stack
      std::vector<Memory> memory_stack;
      machine_interface.find_memory_stack(target, memory_stack, 
                                          (machine->get_processor_kind(target) == Processor::LOC_PROC));
      Memory last = memory_stack.back();
      if (current_instances.find(last) != current_instances.end())
        to_reuse.insert(last);
      else
      {
        to_create.push_back(last);
        create_one = true; // only need to make one
      }
    }

    //////////////////////////////////////
    // Shared Mapper 
    //////////////////////////////////////

    Logger::Category log_shared("sharedmapper");

    //--------------------------------------------------------------------------------------------
    SharedMapper::SharedMapper(Machine *m, HighLevelRuntime *rt, Processor local)
      : DefaultMapper(m,rt,local)
    //--------------------------------------------------------------------------------------------
    {
      log_shared(LEVEL_SPEW,"Initializing the shared mapper on processor %x",local_proc.id);
      // Find the one memory that is shared by all the processors and report an
      // error if none could be found.
      const std::set<Memory> &all_mems = m->get_all_memories();
      std::list<Memory> common_memories(all_mems.begin(),all_mems.end());
      const std::set<Processor> &processors = m->get_all_processors();
      for (std::set<Processor>::const_iterator pit = processors.begin();
            pit != processors.end(); pit++)
      {
        const std::set<Memory> &visible_mems = m->get_visible_memories(*pit);
        for (std::list<Memory>::iterator it = common_memories.begin();
              it != common_memories.end(); /*nothing*/)
        {
          if (visible_mems.find(*it) != visible_mems.end())
            it++;
          else
            it = common_memories.erase(it);
        }
      }
      if (common_memories.empty())
      {
        log_shared(LEVEL_ERROR,"Shared mapper unable to find common memory for all processors!  Exiting...");
        exit(1);
      }
      else
        shared_memory = *common_memories.begin();
    }

    //--------------------------------------------------------------------------------------------
    bool SharedMapper::map_task_region(const Task *task, Processor target, 
                                       MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                       const RegionRequirement &req, unsigned index,
                                       const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                       std::vector<Memory> &target_ranking,
                                       std::set<FieldID> &additional_fields,
                                       bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      target_ranking.push_back(shared_memory);
      enable_WAR_optimization = war_enabled;
      return true;
    }

    //--------------------------------------------------------------------------------------------
    void SharedMapper::rank_copy_targets(const Task *task, Processor target,
                                         MappingTagID tag, bool inline_mapping,
                                         const RegionRequirement &req, unsigned index,
                                         const std::set<Memory> &current_instances,
                                         std::set<Memory> &to_reuse,
                                         std::vector<Memory> &to_create,
                                         bool &create_one)
    //--------------------------------------------------------------------------------------------
    {
      if (current_instances.find(shared_memory) != current_instances.end())
        to_reuse.insert(shared_memory);
      else
      {
        to_create.push_back(shared_memory);
        create_one = true;
      }
    }

    //////////////////////////////////////
    // Sequoia Mapper
    //////////////////////////////////////

    Logger::Category log_sequoia("sequoiamapper");

    //--------------------------------------------------------------------------------------------
    SequoiaMapper::SequoiaMapper(Machine *m, HighLevelRuntime *rt, Processor local)
      : DefaultMapper(m,rt,local)
    //--------------------------------------------------------------------------------------------
    {
      log_sequoia(LEVEL_SPEW,"Initializing the sequoia mapper on processor %x",local_proc.id);
    }

    //--------------------------------------------------------------------------------------------
    bool SequoiaMapper::spawn_child_task(const Task *task)
    //--------------------------------------------------------------------------------------------
    {
      log_sequoia(LEVEL_SPEW,"Spawn child task in sequoia mapper on processor %x", local_proc.id);
      // Need to be able to select target processor
      return true;
    }

    //--------------------------------------------------------------------------------------------
    bool SequoiaMapper::map_task_region(const Task *task, Processor target, 
                                        MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                        const RegionRequirement &req, unsigned index,
                                        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                        std::vector<Memory> &target_ranking,
                                        std::set<FieldID> &additional_fields,
                                        bool &enable_WAR_optimization)
    //--------------------------------------------------------------------------------------------
    {
      log_sequoia(LEVEL_SPEW,"Map task region in sequoia mapper for region (%x,%x,%d) of task %s (ID %d) "
          "(unique id %d) on processor %x",req.region.get_index_space().id, req.region.get_field_space().get_id(), 
          req.region.get_tree_id(), task->variants->name,
          task->task_id, task->get_unique_task_id(), local_proc.id);
      // Perform a Sequoia-like creation of instances.  If this is the first instance, put
      // it in the global memory, otherwise find the instance closest to the processor and
      // select one memory closer.
      std::vector<Memory> memory_stack;
      machine_interface.find_memory_stack(target, memory_stack,
                              (machine->get_processor_kind(target) == Processor::LOC_PROC));
      if (current_instances.empty())
      {
        log_sequoia(LEVEL_DEBUG,"No prior instances for region (%x,%x,%d) on processor %x",
            req.region.get_index_space().id, req.region.get_field_space().get_id(),
            req.region.get_tree_id(), local_proc.id);
        target_ranking.push_back(memory_stack.back());
      }
      else
      {
        // Find the current instance closest to the processor, list from one closer
        unsigned closest_idx = memory_stack.size() - 1;
        for (unsigned idx = 0; idx < memory_stack.size(); idx++)
        {
          if (current_instances.find(memory_stack[idx]) != current_instances.end())
          {
            closest_idx = idx;
            break;
          }
        }
        log_sequoia(LEVEL_DEBUG,"Closest instance for region (%x,%x,%d) is memory %d on processor %x",
            req.region.get_index_space().id,req.region.get_field_space().get_id(),
            req.region.get_tree_id(), memory_stack[closest_idx].id,local_proc.id);
        // Now make the ranking from one closer to the end of the memory stack
        if (closest_idx > 0)
        {
          target_ranking.push_back(memory_stack[closest_idx-1]);
        }
        for (unsigned idx = closest_idx; idx < memory_stack.size(); idx++)
        {
          target_ranking.push_back(memory_stack[idx]);
        }
      }
      enable_WAR_optimization = war_enabled;
      return true;
    }

    //--------------------------------------------------------------------------------------------
    void SequoiaMapper::rank_copy_targets(const Task *task, Processor target,
                                          MappingTagID tag, bool inline_mapping,
                                          const RegionRequirement &req, unsigned index,
                                          const std::set<Memory> &current_instances,
                                          std::set<Memory> &to_reuse,
                                          std::vector<Memory> &to_create,
                                          bool &create_one)
    //--------------------------------------------------------------------------------------------
    {
      log_sequoia(LEVEL_SPEW,"Rank copy targets in sequoia mapper for task %s (ID %d) (unique id %d) "
          "on processor %x", task->variants->name, task->task_id, task->get_unique_task_id(), local_proc.id);
      // This is also Sequoia-like creation of instances.  Find the least common denominator
      // in our stack and pick that memory followed by any memories after it back to the global memory
      std::vector<Memory> memory_stack;
      machine_interface.find_memory_stack(target, memory_stack, 
                            (machine->get_processor_kind(target) == Processor::LOC_PROC));
      if (current_instances.empty())
      {
        to_create.push_back(memory_stack.back());
        create_one = true;
      }
      else
      {
        unsigned last_idx = memory_stack.size()-1;
        for (unsigned idx = memory_stack.size()-1; idx >= 0; idx--)
        {
          if (current_instances.find(memory_stack[idx]) != current_instances.end())
          {
            last_idx = idx;
            break;
          }
        }
        // Now make the ranking from the last_idx to the end
        for (unsigned idx = last_idx; idx < memory_stack.size(); idx++)
        {
          to_create.push_back(memory_stack[idx]);
        }
        create_one = true;
      }
    }
  }; // namespace HighLevel
}; // namespace LegionRuntime

