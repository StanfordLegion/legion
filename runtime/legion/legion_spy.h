/* Copyright 2015 Stanford University, NVIDIA Corporation
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


#ifndef __LEGION_SPY_H__
#define __LEGION_SPY_H__

#include "lowlevel.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"

/**
 * This file contains calls for logging that are consumed by 
 * the legion_spy tool in the tools directory.
 * To see where these statements get consumed, look in spy_parser.py
 */

namespace LegionRuntime {
  namespace HighLevel {
    namespace LegionSpy {

      typedef LegionRuntime::LowLevel::IDType IDType;

      extern Logger::Category log_spy;

      // Logger calls for the machine architecture
      static inline void log_utility_processor(IDType unique_id)
      {
        log_spy.info("Utility " IDFMT "", 
                unique_id);
      }

      static inline void log_processor(IDType unique_id, unsigned kind)
      {
        log_spy.info("Processor " IDFMT " %u", 
                unique_id, kind);
      }

      static inline void log_memory(IDType unique_id, size_t capacity)
      {
        log_spy.info("Memory " IDFMT " %lu", 
                unique_id, capacity);
      }

      static inline void log_proc_mem_affinity(IDType proc_id, 
            IDType mem_id, unsigned bandwidth, unsigned latency)
      {
        log_spy.info("Processor Memory " IDFMT " " IDFMT " %u %u", 
                  proc_id, mem_id, bandwidth, latency);
      }

      static inline void log_mem_mem_affinity(IDType mem1, 
          IDType mem2, unsigned bandwidth, unsigned latency)
      {
        log_spy.info("Memory Memory " IDFMT " " IDFMT " %u %u", 
                          mem1, mem2, bandwidth, latency);
      }

      // Logger calls for the shape of region trees
      static inline void log_top_index_space(IDType unique_id)
      {
        log_spy.info("Index Space " IDFMT "", unique_id);
      }

      static inline void log_index_space_name(IDType unique_id,
                                              const char* name)
      {
        log_spy.info("Index Space Name " IDFMT " %s",
            unique_id, name);
      }

      static inline void log_index_partition(IDType parent_id, 
                IDType unique_id, bool disjoint, const DomainPoint& point)
      {
        log_spy.info("Index Partition " IDFMT " " IDFMT " %u %u %u %u %u",
                    parent_id, unique_id, disjoint, point.dim, point.point_data[0],
                    point.point_data[1], point.point_data[2]);
      }

      static inline void log_index_partition_name(IDType unique_id,
                                                  const char* name)
      {
        log_spy.info("Index Partition Name " IDFMT " %s",
            unique_id, name);
      }

      static inline void log_index_subspace(IDType parent_id, 
                              IDType unique_id, const DomainPoint& point)
      {
        log_spy.info("Index Subspace " IDFMT " " IDFMT " %u %u %u %u",
                          parent_id, unique_id, point.dim, point.point_data[0],
                          point.point_data[1], point.point_data[2]);
      }

      static inline void log_index_space_independence(IDType parent_id,
                                    IDType unique_id1, IDType unique_id2)
      {
        log_spy.info("Index Space Independence " IDFMT " " IDFMT " " IDFMT "",
                      parent_id, unique_id1, unique_id2);
      }
      
      static inline void log_index_partition_independence(IDType parent_id,
                                       IDType unique_id1, IDType unique_id2)
      {
        log_spy.info("Index Partition Independence " IDFMT " " IDFMT " " 
                      IDFMT "", parent_id, unique_id1, unique_id2);
      }

      static inline void log_field_space(unsigned unique_id)
      {
        log_spy.info("Field Space %u", unique_id);
      }

      static inline void log_field_space_name(unsigned unique_id,
                                              const char* name)
      {
        log_spy.info("Field Space Name %u %s",
            unique_id, name);
      }

      static inline void log_field_creation(unsigned unique_id, 
                                            unsigned field_id)
      {
        log_spy.info("Field Creation %u %u", 
                            unique_id, field_id);
      }

      static inline void log_field_name(unsigned unique_id,
                                        unsigned field_id,
                                        const char* name)
      {
        log_spy.info("Field Name %u %u %s",
            unique_id, field_id, name);
      }

      static inline void log_top_region(IDType index_space, 
                      unsigned field_space, unsigned tree_id)
      {
        log_spy.info("Region " IDFMT " %u %u", 
              index_space, field_space, tree_id);
      }

      static inline void log_logical_region_name(IDType index_space, 
                      unsigned field_space, unsigned tree_id,
                      const char* name)
      {
        log_spy.info("Logical Region Name " IDFMT " %u %u %s", 
              index_space, field_space, tree_id, name);
      }

      static inline void log_logical_partition_name(IDType index_partition,
                      unsigned field_space, unsigned tree_id,
                      const char* name)
      {
        log_spy.info("Logical Partition Name " IDFMT " %u %u %s", 
              index_partition, field_space, tree_id, name);
      }

      // Logger calls for operations 
      static inline void log_top_level_task(Processor::TaskFuncID task_id,
                                            UniqueID unique_id,
                                            const char *name)
      {
        log_spy.info("Top Task %u %llu %s", 
            task_id, unique_id, name);
      }

      static inline void log_individual_task(UniqueID context,
                                             UniqueID unique_id,
                                             Processor::TaskFuncID task_id,
                                             const char *name)
      {
        log_spy.info("Individual Task %llu %u %llu %s", 
            context, task_id, unique_id, name);
      }

      static inline void log_index_task(UniqueID context,
                                        UniqueID unique_id,
                                        Processor::TaskFuncID task_id,
                                        const char *name)
      {
        log_spy.info("Index Task %llu %u %llu %s",
            context, task_id, unique_id, name);
      }

      static inline void log_mapping_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.info("Mapping Operation %llu %llu", context, unique_id);
      }

      static inline void log_fill_operation(UniqueID context,
                                            UniqueID unique_id)
      {
        log_spy.info("Fill Operation %llu %llu", context, unique_id);
      }

      static inline void log_close_operation(UniqueID context,
                                             UniqueID unique_id,
                                             unsigned is_inter_close_op)
      {
        log_spy.info("Close Operation %llu %llu %u",
            context, unique_id, is_inter_close_op);
      }

      static inline void log_fence_operation(UniqueID context,
                                             UniqueID unique_id)
      {
        log_spy.info("Fence Operation %llu %llu",
            context, unique_id);
      }

      static inline void log_copy_operation(UniqueID context,
                                            UniqueID unique_id)
      {
        log_spy.info("Copy Operation %llu %llu",
            context, unique_id);
      }

      static inline void log_acquire_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.info("Acquire Operation %llu %llu",
            context, unique_id);
      }

      static inline void log_release_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.info("Release Operation %llu %llu",
            context, unique_id);
      }

      static inline void log_deletion_operation(UniqueID context,
                                                UniqueID deletion)
      {
        log_spy.info("Deletion Operation %llu %llu",
            context, deletion);
      }

      static inline void log_dependent_partition_operation(UniqueID context,
                                                           UniqueID unique_id,
                                                           IDType pid,
                                                           int kind)
      {
        log_spy.info("Dependent Partition Operation %llu %llu " IDFMT " %d",
            context, unique_id, pid, kind);
      }

      static inline void log_pending_partition_operation(UniqueID context,
                                                         UniqueID unique_id)
      {
        log_spy.info("Pending Partition Operation %llu %llu",
            context, unique_id);
      }

      static inline void log_target_pending_partition(UniqueID unique_id,
                                                      IDType pid,
                                                      int kind)
      {
        log_spy.info("Pending Partition Target %llu " IDFMT " %d", unique_id,
            pid, kind);
      }

      static inline void log_index_slice(UniqueID index_id, UniqueID slice_id)
      {
        log_spy.info("Index Slice %llu %llu", index_id, slice_id);
      }

      static inline void log_slice_slice(UniqueID slice_one, UniqueID slice_two)
      {
        log_spy.info("Slice Slice %llu %llu", slice_one, slice_two);
      }

      static inline void log_slice_point(UniqueID slice_id, UniqueID point_id,
                                         const DomainPoint &point)
      {
        log_spy.info("Slice Point %llu %llu %u %u %u %u", 
            slice_id, point_id,
            point.dim, point.point_data[0],
            point.point_data[1], point.point_data[2]);
      }

      static inline void log_point_point(UniqueID p1, UniqueID p2)
      {
        log_spy.info("Point Point %llu %llu", p1, p2);
      }

      // Logger calls for mapping dependence analysis 
      static inline void log_logical_requirement(UniqueID unique_id, 
          unsigned index, bool region, IDType index_component,
          unsigned field_component, unsigned tree_id, unsigned privilege, 
          unsigned coherence, unsigned redop)
      {
        log_spy.info("Logical Requirement %llu %u %u " IDFMT " %u %u %u %u %u", 
            unique_id, index, region, index_component,
            field_component, tree_id, privilege, coherence, redop);
      }

      static inline void log_requirement_fields(UniqueID unique_id, 
          unsigned index, const std::set<unsigned> &logical_fields)
      {
        for (std::set<unsigned>::const_iterator it = logical_fields.begin();
              it != logical_fields.end(); it++)
        {
          log_spy.info("Logical Requirement Field %llu %u %u", 
                              unique_id, index, *it);
        }
      }

      static inline void log_mapping_dependence(UniqueID context, 
                UniqueID prev_id, unsigned prev_idx, UniqueID next_id, 
                unsigned next_idx, unsigned dep_type)
      {
        log_spy.info("Mapping Dependence %llu %llu %u %llu %u %d", 
            context, prev_id, prev_idx, next_id, next_idx, dep_type);
      }

      // Logger calls for physical dependence analysis
      static inline void log_task_instance_requirement(UniqueID unique_id, 
                                  unsigned idx, unsigned index)
      {
        log_spy.info("Task Instance Requirement %llu %u %u", 
                            unique_id, idx, index);
      }

      // Logger calls for events
      static inline void log_event_dependence(Event one, Event two)
      {
        if (one != two)
          log_spy.info("Event Event " IDFMT " %u " IDFMT " %u", 
                          one.id, one.gen, two.id, two.gen);
      }

      static inline void log_event_dependences(
          const std::set<Event> &preconditions, Event result)
      {
        for (std::set<Event>::const_iterator it = preconditions.begin();
              it != preconditions.end(); it++)
        {
          if (*it != result)
            log_spy.info("Event Event " IDFMT " %u " IDFMT " %u", 
                    it->id, it->gen, result.id, result.gen);
        }
      }

      static inline void log_implicit_dependence(Event one, Event two)
      {
        if (one != two)
          log_spy.info("Implicit Event " IDFMT " %u " IDFMT " %u",
              one.id, one.gen, two.id, two.gen);
      }

      static inline void log_op_events(UniqueID uid, Event start_event,
                                       Event term_event)
      {
        log_spy.info("Op Events %llu " IDFMT " %u " IDFMT " %u",
            uid, start_event.id, start_event.gen, 
            term_event.id, term_event.gen);
      }

      static inline void log_copy_operation(IDType src_inst,
                                            IDType dst_inst,
                                            IDType index_handle,
                                            unsigned field_handle,
                                            unsigned tree_id,
                                            Event start_event,
                                            Event term_event,
                                            unsigned redop,
                                            std::set<FieldID> fields)
                                            //const char *mask)
      {
        log_spy.info("Copy Events " IDFMT " " IDFMT " " IDFMT 
                           " %u %u " IDFMT " %u " IDFMT " %u %u",
            src_inst, dst_inst, index_handle, field_handle,
            tree_id, start_event.id, start_event.gen, term_event.id,
            term_event.gen, redop);
        for (std::set<FieldID>::iterator it = fields.begin();
             it != fields.end(); ++it)
          log_spy.info("Copy Field " IDFMT " %u " IDFMT " %u %u",
              start_event.id, start_event.gen, term_event.id, term_event.gen, *it);
      }

      // Logger calls for physical instances
      static inline void log_physical_instance(IDType inst_id, 
                         IDType mem_id, IDType index_handle, 
                         unsigned field_handle, unsigned tree_id)
      {
        log_spy.info("Physical Instance " IDFMT " " IDFMT " " 
                            IDFMT " %u %u", 
            inst_id, mem_id, index_handle, field_handle, tree_id);
      }

      static inline void log_physical_reduction(IDType inst_id, 
          IDType mem_id, IDType index_handle, unsigned field_handle, 
          unsigned tree_id, bool fold, unsigned indirect_id = 0)
      {
        log_spy.info("Reduction Instance " IDFMT " " IDFMT " " 
                            IDFMT " %u %u %u %u", 
                             inst_id, mem_id, index_handle, field_handle, 
                             tree_id, fold, indirect_id);
      }

      static inline void log_instance_field(IDType inst_id, FieldID field_id)
      {
        log_spy.info("Instance Field " IDFMT " %u", inst_id, field_id);
      }


      static inline void log_op_user(UniqueID user,
                                     unsigned idx, 
                                     IDType inst_id)
      {
        log_spy.info("Op Instance User %llu %u " IDFMT "", 
                              user, idx, inst_id);
      }

      static inline void log_phase_barrier(Barrier barrier)
      {
        log_spy.info("Phase Barrier " IDFMT, barrier.id);
      }

      static inline void log_op_proc_user(UniqueID user,
                                          IDType proc_id)
      {
        log_spy.info("Op Processor User %llu " IDFMT "",
                              user, proc_id);
      }

    };

    class TreeStateLogger {
    public:
      TreeStateLogger(void);
      TreeStateLogger(AddressSpaceID sid, bool verbose,
                      bool logical_only, bool physical_only);
      ~TreeStateLogger(void);
    public:
      void log(const char *fmt, ...);
      void down(void);
      void up(void);
      void start_block(const char *ftm, ...);
      void finish_block(void);
      unsigned get_depth(void) const { return depth; }
    public:
      static void capture_state(Internal *rt, const RegionRequirement *req, 
                                unsigned idx, const char *task_name, 
                                long long uid, RegionTreeNode *node, 
                                ContextID ctx, bool before, 
                                bool pre_map, bool closing, bool logical, 
                                FieldMask capture_mask, FieldMask working_mask);
    private:
      void println(const char *fmt, va_list args);
    public:
      const bool verbose;
      const bool logical_only;
      const bool physical_only;
    private:
      FILE *tree_state_log;
      char block_buffer[128];
      unsigned depth;
      Reservation logger_lock;
    };
  };
};

#endif // __LEGION_SPY_H__

