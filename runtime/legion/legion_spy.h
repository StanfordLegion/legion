/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "realm.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"

/**
 * This file contains calls for logging that are consumed by 
 * the legion_spy tool in the tools directory.
 * To see where these statements get consumed, look in spy_parser.py
 */

namespace Legion {
  namespace Internal {
    namespace LegionSpy {

      typedef ::legion_lowlevel_id_t IDType;

      extern LegionRuntime::Logger::Category log_spy;

      // Logger calls for the machine architecture
      static inline void log_processor_kind(unsigned kind, const char *name)
      {
        log_spy.print("Processor Kind %d %s", kind, name);
      }

      static inline void log_memory_kind(unsigned kind, const char *name)
      {
        log_spy.print("Memory Kind %d %s", kind, name);
      }

      static inline void log_processor(IDType unique_id, unsigned kind)
      {
        log_spy.print("Processor " IDFMT " %u", 
		      unique_id, kind);
      }

      static inline void log_memory(IDType unique_id, size_t capacity,
          unsigned kind)
      {
        log_spy.print("Memory " IDFMT " %lu %u", 
		      unique_id, capacity, kind);
      }

      static inline void log_proc_mem_affinity(IDType proc_id, 
            IDType mem_id, unsigned bandwidth, unsigned latency)
      {
        log_spy.print("Processor Memory " IDFMT " " IDFMT " %u %u", 
		      proc_id, mem_id, bandwidth, latency);
      }

      static inline void log_mem_mem_affinity(IDType mem1, 
          IDType mem2, unsigned bandwidth, unsigned latency)
      {
        log_spy.print("Memory Memory " IDFMT " " IDFMT " %u %u", 
		      mem1, mem2, bandwidth, latency);
      }

      // Logger calls for the shape of region trees
      static inline void log_top_index_space(IDType unique_id)
      {
        log_spy.print("Index Space " IDFMT "", unique_id);
      }

      static inline void log_index_space_name(IDType unique_id,
                                              const char* name)
      {
        log_spy.print("Index Space Name " IDFMT " %s",
		      unique_id, name);
      }

      static inline void log_index_partition(IDType parent_id, 
                IDType unique_id, bool disjoint, const DomainPoint& point)
      {
        log_spy.print("Index Partition " IDFMT " " IDFMT " %u %u %d %d %d",
		      parent_id, unique_id, disjoint, point.dim, 
                    (int)point.point_data[0],
                    (int)point.point_data[1],
                    (int)point.point_data[2]);
      }

      static inline void log_index_partition_name(IDType unique_id,
                                                  const char* name)
      {
        log_spy.print("Index Partition Name " IDFMT " %s",
		      unique_id, name);
      }

      static inline void log_index_subspace(IDType parent_id, 
                              IDType unique_id, const DomainPoint& point)
      {
        log_spy.print("Index Subspace " IDFMT " " IDFMT " %u %d %d %d",
		      parent_id, unique_id, point.dim, 
		      (int)point.point_data[0],
		      (int)point.point_data[1],
		      (int)point.point_data[2]);
      }

      static inline void log_field_space(unsigned unique_id)
      {
        log_spy.print("Field Space %u", unique_id);
      }

      static inline void log_field_space_name(unsigned unique_id,
                                              const char* name)
      {
        log_spy.print("Field Space Name %u %s",
		      unique_id, name);
      }

      static inline void log_field_creation(unsigned unique_id, 
                                            unsigned field_id)
      {
        log_spy.print("Field Creation %u %u", 
		      unique_id, field_id);
      }

      static inline void log_field_name(unsigned unique_id,
                                        unsigned field_id,
                                        const char* name)
      {
        log_spy.print("Field Name %u %u %s",
		      unique_id, field_id, name);
      }

      static inline void log_top_region(IDType index_space, 
                      unsigned field_space, unsigned tree_id)
      {
        log_spy.print("Region " IDFMT " %u %u", 
		      index_space, field_space, tree_id);
      }

      static inline void log_logical_region_name(IDType index_space, 
                      unsigned field_space, unsigned tree_id,
                      const char* name)
      {
        log_spy.print("Logical Region Name " IDFMT " %u %u %s", 
		      index_space, field_space, tree_id, name);
      }

      static inline void log_logical_partition_name(IDType index_partition,
                      unsigned field_space, unsigned tree_id,
                      const char* name)
      {
        log_spy.print("Logical Partition Name " IDFMT " %u %u %s", 
		      index_partition, field_space, tree_id, name);
      }

      // For capturing information about the shape of index spaces
      template<int DIM>
      static inline void log_index_space_point(IDType handle, 
                                               long long int *vals)
      {
        log_spy.print("Index Space Point " IDFMT " %d %lld %lld %lld", handle, 
		      DIM, vals[0],
		      DIM < 2 ? 0 : vals[1],
		      DIM < 3 ? 0 : vals[2]);
      }

      template<int DIM>
      static inline void log_index_space_rect(IDType handle, 
                                              long long int *lower, 
                                              long long int *higher)
      {
        log_spy.print("Index Space Rect " IDFMT " %d "
		      "%lld %lld %lld %lld %lld %lld",
		      handle, DIM, lower[0],
		      DIM < 2 ? 0 : lower[1], 
		      DIM < 3 ? 0 : lower[2], higher[0],
		      DIM < 2 ? 0 : higher[1],
		      DIM < 3 ? 0 : higher[2]);
      }

      static inline void log_empty_index_space(IDType handle)
      {
        log_spy.print("Empty Index Space " IDFMT "", handle);
      }

      // Logger calls for operations 
      static inline void log_task_name(TaskID task_id, const char *name)
      {
        log_spy.print("Task ID Name %d %s", task_id, name);
      }

      static inline void log_top_level_task(Processor::TaskFuncID task_id,
                                            UniqueID unique_id,
                                            const char *name)
      {
        log_spy.print("Top Task %u %llu %s", 
		      task_id, unique_id, name);
      }

      static inline void log_individual_task(UniqueID context,
                                             UniqueID unique_id,
                                             Processor::TaskFuncID task_id,
                                             const char *name)
      {
        log_spy.print("Individual Task %llu %u %llu %s", 
		      context, task_id, unique_id, name);
      }

      static inline void log_index_task(UniqueID context,
                                        UniqueID unique_id,
                                        Processor::TaskFuncID task_id,
                                        const char *name)
      {
        log_spy.print("Index Task %llu %u %llu %s",
		      context, task_id, unique_id, name);
      }

      static inline void log_mapping_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.print("Mapping Operation %llu %llu", context, unique_id);
      }

      static inline void log_fill_operation(UniqueID context,
                                            UniqueID unique_id)
      {
        log_spy.print("Fill Operation %llu %llu", context, unique_id);
      }

      static inline void log_close_operation(UniqueID context,
                                             UniqueID unique_id,
                                             bool is_intermediate_close_op,
                                             bool read_only_close_op)
      {
        log_spy.print("Close Operation %llu %llu %u %u",
		      context, unique_id, is_intermediate_close_op ? 1 : 0,
		      read_only_close_op ? 1 : 0);
      }

      static inline void log_close_op_creator(UniqueID close_op_id,
                                              UniqueID creator_op_id,
                                              int idx)
      {
        log_spy.print("Close Operation Creator %llu %llu %d",
		      close_op_id, creator_op_id, idx);
      }

      static inline void log_fence_operation(UniqueID context,
                                             UniqueID unique_id)
      {
        log_spy.print("Fence Operation %llu %llu",
		      context, unique_id);
      }

      static inline void log_copy_operation(UniqueID context,
                                            UniqueID unique_id)
      {
        log_spy.print("Copy Operation %llu %llu",
		      context, unique_id);
      }

      static inline void log_acquire_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.print("Acquire Operation %llu %llu",
		      context, unique_id);
      }

      static inline void log_release_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.print("Release Operation %llu %llu",
		      context, unique_id);
      }

      static inline void log_deletion_operation(UniqueID context,
                                                UniqueID deletion)
      {
        log_spy.print("Deletion Operation %llu %llu",
		      context, deletion);
      }

      static inline void log_dependent_partition_operation(UniqueID context,
                                                           UniqueID unique_id,
                                                           IDType pid,
                                                           int kind)
      {
        log_spy.print("Dependent Partition Operation %llu %llu " IDFMT " %d",
		      context, unique_id, pid, kind);
      }

      static inline void log_pending_partition_operation(UniqueID context,
                                                         UniqueID unique_id)
      {
        log_spy.print("Pending Partition Operation %llu %llu",
		      context, unique_id);
      }

      static inline void log_target_pending_partition(UniqueID unique_id,
                                                      IDType pid,
                                                      int kind)
      {
        log_spy.print("Pending Partition Target %llu " IDFMT " %d", unique_id,
		      pid, kind);
      }

      static inline void log_index_slice(UniqueID index_id, UniqueID slice_id)
      {
        log_spy.print("Index Slice %llu %llu", index_id, slice_id);
      }

      static inline void log_slice_slice(UniqueID slice_one, UniqueID slice_two)
      {
        log_spy.print("Slice Slice %llu %llu", slice_one, slice_two);
      }

      static inline void log_slice_point(UniqueID slice_id, UniqueID point_id,
                                         const DomainPoint &point)
      {
        log_spy.print("Slice Point %llu %llu %u %d %d %d", 
		      slice_id, point_id,
		      point.dim, (int)point.point_data[0],
		      (int)point.point_data[1], (int)point.point_data[2]);
      }

      static inline void log_point_point(UniqueID p1, UniqueID p2)
      {
        log_spy.print("Point Point %llu %llu", p1, p2);
      }

      // Logger calls for mapping dependence analysis 
      static inline void log_logical_requirement(UniqueID unique_id, 
          unsigned index, bool region, IDType index_component,
          unsigned field_component, unsigned tree_id, unsigned privilege, 
          unsigned coherence, unsigned redop, IDType parent_index)
      {
        log_spy.print("Logical Requirement %llu %u %u " IDFMT " %u %u "
		      "%u %u %u " IDFMT, unique_id, index, region, index_component,
		      field_component, tree_id,
		      privilege, coherence, redop, parent_index);
      }

      static inline void log_requirement_fields(UniqueID unique_id, 
          unsigned index, const std::set<unsigned> &logical_fields)
      {
        for (std::set<unsigned>::const_iterator it = logical_fields.begin();
              it != logical_fields.end(); it++)
        {
          log_spy.print("Logical Requirement Field %llu %u %u", 
			unique_id, index, *it);
        }
      }

      static inline void log_requirement_fields(UniqueID unique_id, 
          unsigned index, const std::vector<FieldID> &logical_fields)
      {
        for (std::vector<FieldID>::const_iterator it = logical_fields.begin();
              it != logical_fields.end(); it++)
        {
          log_spy.print("Logical Requirement Field %llu %u %u", 
			unique_id, index, *it);
        }
      }

      // Logger call for physical instances
      static inline void log_physical_instance(IDType inst_id, IDType mem_id,
                                               ReductionOpID redop)
      {
        log_spy.print("Physical Instance " IDFMT " " IDFMT " %d", 
		      inst_id, mem_id, redop);
      }

      static inline void log_physical_instance_region(IDType inst_id, 
                                                      LogicalRegion handle)
      {
        log_spy.print("Physical Instance Region " IDFMT " %d %d %d",
                      inst_id, handle.get_index_space().get_id(), 
                      handle.get_field_space().get_id(), handle.get_tree_id());
      }

      static inline void log_physical_instance_field(IDType inst_id,
                                                     FieldID field_id)
      {
        log_spy.print("Physical Instance Field " IDFMT " %d", inst_id, field_id);
      }

      // Logger calls for mapping decisions
      static inline void log_mapping_decision(UniqueID unique_id, 
                                  unsigned index, FieldID fid, IDType inst_id)
      {
        log_spy.print("Mapping Decision %llu %d %d " IDFMT "", unique_id,
		      index, fid, inst_id);
      }

      // The calls above this ifdef record the basic information about
      // the execution of an application. It is sufficient to show how
      // an application executed, but is insufficient to actually 
      // validate the execution by the runtime. The calls inside this
      // ifdef are the more expensive logging calls necessary for 
      // checking the correctness of the runtime's behaviour.
#ifdef LEGION_SPY
      // Logger calls for mapping dependences
      static inline void log_mapping_dependence(UniqueID context, 
                UniqueID prev_id, unsigned prev_idx, UniqueID next_id, 
                unsigned next_idx, unsigned dep_type)
      {
        log_spy.print("Mapping Dependence %llu %llu %u %llu %u %d", 
		      context, prev_id, prev_idx,
		      next_id, next_idx, dep_type);
      }

      // Logger calls for realm events
      static inline void log_event_dependence(LgEvent one, LgEvent two)
      {
        if (one != two)
          log_spy.print("Event Event " IDFMT " %u " IDFMT " %u", 
			one.id, one.gen, two.id, two.gen);
      }

      static inline void log_ap_user_event(ApUserEvent event)
      {
        log_spy.print("Ap User Event " IDFMT " %u", event.id, event.gen);
      }

      static inline void log_rt_user_event(RtUserEvent event)
      {
        log_spy.print("Rt User Event " IDFMT " %u", event.id, event.gen);
      }

      static inline void log_ap_user_event_trigger(ApUserEvent event)
      {
        log_spy.print("Ap User Event Trigger " IDFMT " %u",event.id,event.gen);
      }

      static inline void log_rt_user_event_trigger(RtUserEvent event)
      {
        log_spy.print("Rt User Event Trigger " IDFMT " %u",event.id,event.gen);
      }

      static inline void log_operation_events(UniqueID uid,
                                              LgEvent pre, LgEvent post)
      {
        log_spy.print("Operation Events %llu " IDFMT " %u " IDFMT " %u",
		      uid, pre.id, pre.gen, post.id, post.gen);
      }

      static inline void log_copy_events(UniqueID op_unique_id,
                                         LogicalRegion handle,
                                         LgEvent pre, LgEvent post)
      {
        log_spy.print("Copy Events %llu %d %d %d " IDFMT " %u "
                      IDFMT " %u", op_unique_id,
                      handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), handle.get_tree_id(), 
                      pre.id, pre.gen, post.id, post.gen);
      }

      static inline void log_copy_field(LgEvent post, FieldID src_fid,
                                        IDType src, FieldID dst_fid,
                                        IDType dst, ReductionOpID redop)
      {
        log_spy.print("Copy Field " IDFMT " %u %d " IDFMT " %d " IDFMT " %d",
                      post.id, post.gen, src_fid, src, dst_fid, dst, redop);
      }

      static inline void log_copy_intersect(LgEvent post, int is_region,
                                            IDType index, unsigned field,
                                            unsigned tree_id)
      {
        log_spy.print("Copy Intersect " IDFMT " %u %d " IDFMT " %d %d",
                      post.id, post.gen, is_region, index, field, tree_id);
      }

      static inline void log_fill_events(UniqueID op_unique_id,
                                         LogicalRegion handle,
                                         LgEvent pre, LgEvent post)
      {
        log_spy.print("Fill Events %llu %d %d %d " IDFMT " %u " IDFMT " %u",
		      op_unique_id, handle.get_index_space().get_id(),
		      handle.get_field_space().get_id(), handle.get_tree_id(),
		      pre.id, pre.gen, post.id, post.gen);
      }

      static inline void log_fill_field(LgEvent post, FieldID fid, IDType dst)
      {
        log_spy.print("Fill Field " IDFMT " %u %d " IDFMT, 
                      post.id, post.gen, fid, dst);
      }

      static inline void log_fill_intersect(LgEvent post, int is_region,
                                            IDType index, unsigned field,
                                            unsigned tree_id)
      {
        log_spy.print("Fill Intersect " IDFMT " %u %d " IDFMT " %d %d",
		      post.id, post.gen, is_region, index, field, tree_id);
      }

      static inline void log_phase_barrier(ApBarrier barrier)
      {
        log_spy.print("Phase Barrier " IDFMT, barrier.id);
      } 
#endif
    }; // namespace LegionSpy

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
      static void capture_state(Runtime *rt, const RegionRequirement *req, 
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
  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_SPY_H__

