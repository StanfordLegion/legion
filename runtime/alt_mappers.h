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


#ifndef __ALTERNATIVE_MAPPERS__
#define __ALTERNATIVE_MAPPERS__

#include "legion.h"
#include "default_mapper.h"

// A collection of natural extensions to the default
// mapper that are useful for various purposes.

namespace LegionRuntime {
  namespace HighLevel {
    // A debug mapper that will always map things
    // into the last memory in it's stack and will
    // turn off all task stealing
    class DebugMapper : public DefaultMapper {
    public:
      DebugMapper(Machine *m, HighLevelRuntime *rt, Processor local);
    public:
      virtual bool spawn_child_task(const Task *task);
      virtual Processor target_task_steal(const std::set<Processor> &blacklist);
      virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);
      virtual bool map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    std::set<FieldID> &additional_fields,
                                    bool &enable_WAR_optimization);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::set<Memory> &to_reuse,
                                    std::vector<Memory> &to_create,
                                    bool &create_one);
    };

    // A mapper that always maps data into a memory that
    // is commonly shared by all processors and reports
    // an error if no such memory can be found.
    class SharedMapper : public DefaultMapper {
    public:
      SharedMapper(Machine *m, HighLevelRuntime *rt, Processor local);
    public:
      virtual bool map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    std::set<FieldID> &additional_fields,
                                    bool &enable_WAR_optimization);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::set<Memory> &to_reuse,
                                    std::vector<Memory> &to_create,
                                    bool &create_one); 
    protected:
      Memory shared_memory;
    };

    // A mapper that makes an assumption about the
    // task-tree that it is reasonably matched to the
    // underlying memory hierarchy.  This mapper will
    // try to move data one level closer to the processor
    // than it was before
    class SequoiaMapper : public DefaultMapper {
    public:
      SequoiaMapper(Machine *m, HighLevelRuntime *rt, Processor local);
    public:
      virtual bool spawn_child_task(const Task *task);
      virtual bool map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    std::set<FieldID> &additional_fields,
                                    bool &enable_WAR_optimization);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::set<Memory> &to_reuse,
                                    std::vector<Memory> &to_create,
                                    bool &create_one);
    };
  };
};

#endif // __ALTERNATIVE_MAPPERS__
