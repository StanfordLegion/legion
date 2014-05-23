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

    /**
     * \class DefaultMapper
     * The default mapper class is our base implementation of the
     * mapper interface that relies on some simple heuristics 
     * to perform most of them calls for general purpose Legion
     * applications.  You should feel free to extend this class
     * with your own heuristics by overriding some or all of the
     * methods.  You can also ignore this implementation entirely
     * and perform your own implementation of the mapper interface.
     */
    class DefaultMapper : public Mapper {
    public:
      DefaultMapper(Machine *machine, HighLevelRuntime *rt, Processor local);
      DefaultMapper(const DefaultMapper &rhs);
      virtual ~DefaultMapper(void);
    public:
      DefaultMapper& operator=(const DefaultMapper &rhs);
    public:
      virtual void select_task_options(Task *task);
      virtual void select_tasks_to_schedule(
                      const std::list<Task*> &ready_tasks);
      virtual void target_task_steal(
                            const std::set<Processor> &blacklist,
                            std::set<Processor> &targets);
      virtual void permit_task_steal(Processor thief, 
                                const std::vector<const Task*> &tasks,
                                std::set<const Task*> &to_steal);
      virtual void slice_domain(const Task *task, const Domain &domain,
                                std::vector<DomainSplit> &slices);
      virtual bool pre_map_task(Task *task);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
      virtual bool map_must_epoch(const std::vector<Task*> &tasks,
                            const std::vector<MappingConstraint> &constraints,
                            MappingTagID tag);
      virtual void notify_mapping_result(const Mappable *mappable);
      virtual void notify_mapping_failed(const Mappable *mappable);
      virtual bool rank_copy_targets(const Mappable *mappable,
                                     LogicalRegion rebuild_region,
                                     const std::set<Memory> &current_instances,
                                     bool complete,
                                     size_t max_blocking_factor,
                                     std::set<Memory> &to_reuse,
                                     std::vector<Memory> &to_create,
                                     bool &create_one,
                                     size_t &blocking_factor);
      virtual void rank_copy_sources(const Mappable *mappable,
                      const std::set<Memory> &current_instances,
                      Memory dst_mem, 
                      std::vector<Memory> &chosen_order);
      virtual void notify_profiling_info(const Task *task);
      virtual bool speculate_on_predicate(const Task *task,
                                          bool &spec_value);
      virtual int get_tunable_value(const Task *task, 
                                    TunableID tid,
                                    MappingTagID tag);
      virtual void handle_message(Processor source,
                                  const void *message,
                                  size_t length);
    public:
      // Helper methods for building other kinds of mappers, made static 
      // so they can be used in non-derived classes
      // Pick a random processor of a given kind
      static Processor select_random_processor(
                              const std::set<Processor> &options, 
                              Processor::Kind filter, Machine *machine);
      // Break an IndexSpace of tasks into IndexSplits
      static void decompose_index_space(const Domain &domain, 
                              const std::vector<Processor> &targets,
                              unsigned splitting_factor, 
                              std::vector<Mapper::DomainSplit> &slice);
    protected:
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

