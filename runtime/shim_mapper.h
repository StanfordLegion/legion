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


#ifndef __SHIM_MAPPER_H__
#define __SHIM_MAPPER_H__

#include "legion.h"
#include "mapping_utilities.h"
#include "default_mapper.h"
#include <cstdlib>
#include <cassert>
#include <algorithm>

namespace LegionRuntime {
  namespace HighLevel {

    /**
     * \class ShimMapper
     * The ShimMapper class provides backwards compatibility with an earlier
     * version of the mapping interface.  The new mapper calls are implemented
     * as functions of the earlier mapper calls.  Old mappers can use the
     * new Mapper interface simply by extending the ShimMapper instead
     * of the old DefaultMapper
     */
    class ShimMapper : public DefaultMapper {
    public:
      ShimMapper(Machine *machine, HighLevelRuntime *rt, Processor local);
      ShimMapper(const ShimMapper &rhs);
      virtual ~ShimMapper(void);
    public:
      ShimMapper& operator=(const ShimMapper &rhs);
    public:
      // The new mapping calls
      virtual void select_task_options(Task *task);
      virtual void select_tasks_to_schedule(
                      const std::list<Task*> &ready_tasks);
      virtual void target_task_steal(
                            const std::set<Processor> &blacklist,
                            std::set<Processor> &targets);
      // No need to override permit_task_steal as the interface is unchanged
      //virtual void permit_task_steal(Processor thief, 
      //                          const std::vector<const Task*> &tasks,
      //                          std::set<const Task*> &to_steal);
      // No need to override slice_domain as the interface is unchanged
      //virtual void slice_domain(const Task *task, const Domain &domain,
      //                          std::vector<DomainSplit> &slices);
      virtual bool pre_map_task(Task *task);
      virtual void select_task_variant(Task *task);
      virtual bool map_task(Task *task);
      virtual bool map_copy(Copy *copy);
      virtual bool map_inline(Inline *inline_operation);
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
      virtual bool speculate_on_predicate(const Mappable *mappable,
                                          bool &spec_value);
    protected:
      // Old-style mapping methods
      virtual bool spawn_task(const Task *task);
      virtual bool map_task_locally(const Task *task);
      virtual Processor select_target_processor(const Task *task);
      virtual bool map_region_virtually(const Task *task, Processor target,
					const RegionRequirement &req, 
                                        unsigned index);
      virtual bool map_task_region(const Task *task, Processor target, 
                                   MappingTagID tag, bool inline_mapping, 
                                   bool pre_mapping, 
                                   const RegionRequirement &req, unsigned index,
        const std::map<Memory,bool/*all-fields-up-to-date*/> &current_instances,
				   std::vector<Memory> &target_ranking,
				   std::set<FieldID> &additional_fields,
				   bool &enable_WAR_optimization);
      virtual size_t select_region_layout(const Task *task, Processor target,
					  const RegionRequirement &req, 
                                          unsigned index, 
                                          const Memory &chosen_mem, 
                                          size_t max_blocking_factor);
      virtual bool select_reduction_layout(const Task *task, 
                                           const Processor target,
					   const RegionRequirement &req, 
                                           unsigned index, 
                                           const Memory &chosen_mem);
      virtual void select_tasks_to_schedule(const std::list<Task*> &ready_tasks,
					    std::vector<bool> &ready_mask);
      virtual Processor target_task_steal(const std::set<Processor> &blacklist);
      virtual VariantID select_task_variant(const Task *task, Processor target);
      virtual void notify_mapping_result(const Task *task, Processor target,
					 const RegionRequirement &req,
					 unsigned index, bool inline_mapping, 
                                         Memory result);
      virtual void notify_failed_mapping(const Task *task, Processor target,
					 const RegionRequirement &req,
					 unsigned index, bool inline_mapping);
      virtual void rank_copy_sources(const std::set<Memory> &current_instances,
				     const Memory &dst, 
                                     std::vector<Memory> &chosen_order);
      virtual void rank_copy_targets(const Task *task, Processor target,
                                   MappingTagID tag, bool inline_mapping,
                                   const RegionRequirement &req, unsigned index,
                                   const std::set<Memory> &current_instances,
                                   std::set<Memory> &to_reuse,
                                   std::vector<Memory> &to_create,
                                   bool &create_one);
      virtual bool profile_task_execution(const Task *task, Processor target);
      virtual void notify_profiling_info(const Task *task, Processor target,
					 const ExecutionProfile &profiling);
      virtual bool speculate_on_predicate(MappingTagID tag, 
                                          bool &speculative_value);
    };

  };
};

#endif // __SHIM_MAPPER_H__

// EOF

