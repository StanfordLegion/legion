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


#ifndef __DEFAULT_MAPPER_H__
#define __DEFUALT_MAPPER_H__

#include "legion.h"

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
      virtual void slice_index_space(const Task *task, const IndexSpace &index_space,
                                      std::vector<Mapper::IndexSplit> &slices);
      virtual VariantID select_task_variant(const Task *task, Processor target);
      virtual bool map_region_virtually(const Task *task, Processor target,
                                        const RegionRequirement &req, unsigned index);
      virtual void map_task_region(const Task *task, Processor target, 
                                    MappingTagID tag, bool inline_mapping,
                                    const RegionRequirement &req, unsigned index,
                                    const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
                                    std::vector<Memory> &target_ranking,
                                    bool &enable_WAR_optimization);
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
      virtual bool speculate_on_predicate(MappingTagID tag, bool &speculative_value);
    public:
      // Helper methods for building other kinds of mappers, made static so they can be used in non-derived classes

      // Construct a memory stack for the target processor and put it in the result vector.  By default sorts by
      // bandwidth.  Passing false to bandwidth will result in memories being sorted by latency instead.
      static void compute_memory_stack(Processor target, std::vector<Memory> &result, Machine *machine, bool bandwidth = true);
      // Pick a random processor of a given kind
      static Processor select_random_processor(const std::set<Processor> &options, Processor::Kind filter, Machine *machine);
      // Break an IndexSpace of tasks into IndexSplits
      static void decompose_index_space(const IndexSpace &index_space, const std::vector<Processor> &targets,
                                        unsigned splitting_factor, std::vector<Mapper::IndexSplit> &slice);
    protected:
      HighLevelRuntime *const runtime;
      const Processor local_proc;
      const Processor::Kind local_kind;
      std::map<Processor,Processor::Kind> local_procs;
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
      // Whether or not copies can be made to avoid Write-After-Read dependences
      // Controlled by -dm:war
      bool war_enabled;
      // Track whether stealing is enabled
      bool stealing_enabled;
      // The maximum number of tasks scheduled per step
      unsigned max_schedule_count;
      // The memory stack for each processor in this mapper
      std::map<Processor,std::vector<Memory> > memory_stacks;
      // For every processor group, get the set of types of processors that it has
      std::map<Processor,Processor::Kind> other_procs;
    };

  };
};

#endif // __DEFAULT_MAPPER_H__

// EOF

