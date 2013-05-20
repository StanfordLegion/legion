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


#ifndef __DEFAULT_MAPPER_H__
#define __DEFAULT_MAPPER_H__

#include "legion.h"
#include "mapping_utilities.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace LegionRuntime {
  namespace HighLevel {

    class DefaultMapper : public Mapper {
    public:
      DefaultMapper(Machine *machine, HighLevelRuntime *rt, Processor local);
      virtual ~DefaultMapper(void);
    public:
      virtual void select_tasks_to_schedule(const std::list<Task*> &ready_tasks, std::vector<bool> &ready_mask);
      virtual bool map_task_locally(const Task *task);
      virtual bool spawn_task(const Task *task);
      virtual Processor select_target_processor(const Task *task);
      virtual Processor target_task_steal(const std::set<Processor> &blacklisted);
      virtual void permit_task_steal(Processor thief, const std::vector<const Task*> &tasks,
                                      std::set<const Task*> &to_steal);
      virtual void slice_domain(const Task *task, const Domain &domain,
                                      std::vector<Mapper::DomainSplit> &slices);
      virtual VariantID select_task_variant(const Task *task, Processor target);
      virtual bool map_region_virtually(const Task *task, Processor target,
                                        const RegionRequirement &req, unsigned index);
      virtual bool map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping, bool pre_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    std::set<FieldID> &additional_fields,
                                    bool &enable_WAR_optimization);
      virtual void notify_mapping_result(const Task *task, Processor target, const RegionRequirement &req,
                                          unsigned index, bool inline_mapping, Memory result);
      virtual void notify_failed_mapping(const Task *task, Processor target,
                                          const RegionRequirement &req, unsigned index, bool inline_mapping);
      virtual size_t select_region_layout(const Task *task, Processor target,
                                          const RegionRequirement &req, unsigned index,
                                          const Memory &chosen_mem, size_t max_blocking_factor); 
      virtual bool select_reduction_layout(const Task *task, Processor target,
                                          const RegionRequirement &req, unsigned index,
                                          const Memory &chosen_mem);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::set<Memory> &current_instances,
                                    std::set<Memory> &to_reuse,
                                    std::vector<Memory> &to_create,
                                    bool &create_one);
      virtual void rank_copy_sources(const std::set<Memory> &current_instances,
                                     const Memory &dst, std::vector<Memory> &chosen_order);
      virtual bool profile_task_execution(const Task *task, Processor target);
      virtual void notify_profiling_info(const Task *task, Processor target, const ExecutionProfile &profile);
      virtual bool speculate_on_predicate(MappingTagID tag, bool &speculative_value);
    public:
      // Helper methods for building other kinds of mappers, made static so they can be used in non-derived classes

      // Pick a random processor of a given kind
      static Processor select_random_processor(const std::set<Processor> &options, Processor::Kind filter, Machine *machine);
      // Break an IndexSpace of tasks into IndexSplits
      static void decompose_index_space(const Domain &domain, const std::vector<Processor> &targets,
                                        unsigned splitting_factor, std::vector<Mapper::DomainSplit> &slice);
    protected:
      HighLevelRuntime *const runtime;
      const Processor local_proc;
      const Processor::Kind local_kind;
      Machine *const machine;
      // The maximum number of tasks a mapper will allow to be stolen at a time
      // Controlled by -dm:thefts
      unsigned max_steals_per_theft;
      // The maximum number of times that a single task is allowed to be stolen
      // Controlled by -dm:count
      unsigned max_steal_count;
      // The splitting factor for breaking index spaces across the machine
      // Mapper will try to break the space into split_factor * num_procs
      // difference pieces
      // Controlled by -dm:split
      unsigned splitting_factor;
      // Do a breadth-first traversal of the task tree, by default we do
      // a depth-first traversal to improve locality
      bool breadth_first_traversal;
      // Whether or not copies can be made to avoid Write-After-Read dependences
      // Controlled by -dm:war
      bool war_enabled;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
      // Utilities for use within the default mapper 
      MappingUtilities::MachineQueryInterface machine_interface;
      MappingUtilities::MappingMemoizer memoizer;
      MappingUtilities::MappingProfiler profiler;
    };

  };
};

#endif // __DEFAULT_MAPPER_H__

// EOF

