/* Copyright 2018 Stanford University, NVIDIA Corporation
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
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"

/**
 * This file contains calls for logging that are consumed by 
 * the legion_spy tool in the tools directory.
 * To see where these statements get consumed, look in spy_parser.py
 */

namespace Legion {
  namespace Internal {
    namespace LegionSpy {

      typedef ::realm_id_t IDType;

      extern Realm::Logger log_spy;

      // One time logger calls to record what gets logged
      static inline void log_legion_spy_config(void)
      {
#ifdef LEGION_SPY
        log_spy.print("Legion Spy Detailed Logging");
#else
        log_spy.print("Legion Spy Logging");
#endif
      }

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
        log_spy.print("Memory " IDFMT " %zu %u", 
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
                IDType unique_id, bool disjoint, LegionColor point)
      {
        log_spy.print("Index Partition " IDFMT " " IDFMT " %u %lld",
		      parent_id, unique_id, disjoint, point); 
      }

      static inline void log_index_partition_name(IDType unique_id,
                                                  const char* name)
      {
        log_spy.print("Index Partition Name " IDFMT " %s",
		      unique_id, name);
      }

      static inline void log_index_subspace(IDType parent_id, 
                              IDType unique_id, const DomainPoint &point)
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
                                unsigned field_id, size_t size)
      {
        log_spy.print("Field Creation %u %u %ld", 
		      unique_id, field_id, long(size));
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
      template<int DIM, typename T>
      static inline void log_index_space_point(IDType handle,
                                    const Point<DIM,T> &point)
      {
        LEGION_STATIC_ASSERT(DIM <= 3);
        log_spy.print("Index Space Point " IDFMT " %d %lld %lld %lld", handle,
                      DIM, (long long)(point[0]), 
                      (long long)((DIM < 2) ? 0 : point[1]),
                      (long long)((DIM < 3) ? 0 : point[2]));
      }

      template<int DIM, typename T>
      static inline void log_index_space_rect(IDType handle, 
                                              const Rect<DIM,T> &rect)
      {
        LEGION_STATIC_ASSERT(DIM <= 3);
        log_spy.print("Index Space Rect " IDFMT " %d "
                      "%lld %lld %lld %lld %lld %lld", handle, DIM, 
                      (long long)(rect.lo[0]),
                      (long long)((DIM < 2) ? 0 : rect.lo[1]), 
                      (long long)((DIM < 3) ? 0 : rect.lo[2]), 
                      (long long)(rect.hi[0]), 
                      (long long)((DIM < 2) ? 0 : rect.hi[1]), 
                      (long long)((DIM < 3) ? 0 : rect.hi[2]));
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

      static inline void log_task_variant(TaskID task_id, unsigned variant_id,
                                          bool inner, bool leaf, 
                                          bool idempotent, const char *name)
      {
        log_spy.print("Task Variant %d %d %d %d %d %s", task_id, variant_id,
                                               inner, leaf, idempotent, name);
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

      static inline void log_open_operation(UniqueID context,
                                            UniqueID unique_id)
      {
        log_spy.print("Open Operation %llu %llu", context, unique_id);
      }

      static inline void log_advance_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.print("Advance Operation %llu %llu", context, unique_id);
      }

      static inline void log_internal_op_creator(UniqueID internal_op_id,
                                                 UniqueID creator_op_id,
                                                 int idx)
      {
        log_spy.print("Internal Operation Creator %llu %llu %d",
		      internal_op_id, creator_op_id, idx);
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

      static inline void log_attach_operation(UniqueID context,
                                              UniqueID attach)
      {
        log_spy.print("Attach Operation %llu %llu", 
                      context, attach);
      }

      static inline void log_detach_operation(UniqueID context,
                                              UniqueID detach)
      {
        log_spy.print("Detach Operation %llu %llu",
                      context, detach);
      }

      static inline void log_dynamic_collective(UniqueID context, 
                                                UniqueID collective)
      {
        log_spy.print("Dynamic Collective %llu %llu", context, collective);
      }

      static inline void log_timing_operation(UniqueID context, UniqueID timing)
      {
        log_spy.print("Timing Operation %llu %llu", context, timing);
      }

      static inline void log_predicate_operation(UniqueID context, 
                                                 UniqueID pred_op)
      {
        log_spy.print("Predicate Operation %llu %llu", context, pred_op);
      }

      static inline void log_must_epoch_operation(UniqueID context,
                                                  UniqueID must_op)
      {
        log_spy.print("Must Epoch Operation %llu %llu", context, must_op);
      }

      static inline void log_summary_operation(UniqueID context,
                                               UniqueID unique_id)
      {
        log_spy.print("Summary Operation %llu %llu", context, unique_id);
      }

      static inline void log_summary_op_creator(UniqueID internal_op_id,
                                                UniqueID creator_op_id)
      {
        log_spy.print("Summary Operation Creator %llu %llu",
		      internal_op_id, creator_op_id);
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

      static inline void log_index_point(UniqueID index_id, UniqueID point_id,
                                         const DomainPoint &point)
      {
        log_spy.print("Index Point %llu %llu %u %d %d %d", index_id, point_id,
                      point.dim, (int)point.point_data[0],
                      (int)point.point_data[1], (int)point.point_data[2]);
      }

      static inline void log_child_operation_index(UniqueID parent_id, 
                                       unsigned index, UniqueID child_id)
      {
        log_spy.print("Operation Index %llu %d %llu", parent_id,index,child_id);
      }

      static inline void log_close_operation_index(UniqueID parent_id,
                                        unsigned index, UniqueID child_id)
      {
        log_spy.print("Close Index %llu %d %llu", parent_id, index, child_id);
      }

      static inline void log_predicated_false_op(UniqueID unique_id)
      {
        log_spy.print("Predicate False %lld", unique_id);
      }

      // Logger calls for mapping dependence analysis 
      static inline void log_logical_requirement(UniqueID unique_id, 
          unsigned index, bool region, IDType index_component,
          unsigned field_component, unsigned tree_id, unsigned privilege, 
          unsigned coherence, unsigned redop, IDType parent_index)
      {
        log_spy.print("Logical Requirement %llu %u %u " IDFMT " %u %u "
		      "%u %u %u " IDFMT, unique_id, index, region, 
                      index_component, field_component, tree_id,
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

      static inline void log_projection_function(ProjectionID pid,
                                                 int depth)
      {
        log_spy.print("Projection Function %u %d", pid, depth);
      }

      static inline void log_requirement_projection(UniqueID unique_id,
                                      unsigned index, ProjectionID pid)
      {
        log_spy.print("Logical Requirement Projection %llu %u %u", 
                      unique_id, index, pid);
      }

      template<int DIM, typename T>
      static inline void log_launch_index_space_rect(UniqueID unique_id, 
                                                     const Rect<DIM,T> &rect)
      {
        LEGION_STATIC_ASSERT(DIM <= 3);
        log_spy.print() << "Index Launch Rect " << unique_id << " "
                        << DIM << " " << rect.lo[0]
                        << " " << ((DIM < 2) ? 0 : rect.lo[1])
                        << " " << ((DIM < 3) ? 0 : rect.lo[2])
                        << " " << rect.hi[0]
                        << " " << ((DIM < 2) ? 0 : rect.hi[1])
                        << " " << ((DIM < 3) ? 0 : rect.hi[2]);
      }

      // Logger calls for futures
      static inline void log_future_creation(UniqueID creator_id,
                                             ApEvent future_event, 
                                             const DomainPoint &point)
      {
        log_spy.print("Future Creation %llu " IDFMT " %u %d %d %d",
                      creator_id, future_event.id, point.dim,
                      (int)point.point_data[0], 
                      (point.dim > 1) ? (int)point.point_data[1] : 0,
                      (point.dim > 2) ? (int)point.point_data[2] : 0);
      }

      static inline void log_future_use(UniqueID user_id, 
                                        ApEvent future_event)
      {
        log_spy.print("Future Usage %llu " IDFMT "", user_id, future_event.id);
      }

      static inline void log_predicate_use(UniqueID pred_id,
                                           UniqueID previous_predicate)
      {
        log_spy.print("Predicate Use %llu %llu", pred_id, previous_predicate);
      }

      // Logger call for physical instances
      static inline void log_physical_instance(ApEvent inst_event,
                                               IDType inst_id, IDType mem_id,
                                               ReductionOpID redop)
      {
        log_spy.print("Physical Instance " IDFMT " " IDFMT " " IDFMT " %d", 
		      inst_event.id, inst_id, mem_id, redop);
      }

      static inline void log_physical_instance_region(ApEvent inst_event, 
                                                      LogicalRegion handle)
      {
        log_spy.print("Physical Instance Region " IDFMT " %d %d %d",
                      inst_event.id, handle.get_index_space().get_id(), 
                      handle.get_field_space().get_id(), handle.get_tree_id());
      }

      static inline void log_physical_instance_field(ApEvent inst_event,
                                                     FieldID field_id)
      {
        log_spy.print("Physical Instance Field " IDFMT " %d", 
                      inst_event.id, field_id);
      }

      static inline void log_physical_instance_creator(ApEvent inst_event, 
                                           UniqueID creator_id, IDType proc_id)
      {
        log_spy.print("Physical Instance Creator " IDFMT " %lld " IDFMT "",
                      inst_event.id, creator_id, proc_id);
      }

      static inline void log_physical_instance_creation_region(
                                      ApEvent inst_event, LogicalRegion handle)
      {
        log_spy.print("Physical Instance Creation Region " IDFMT " %d %d %d",
                      inst_event.id, handle.get_index_space().get_id(), 
                      handle.get_field_space().get_id(), handle.get_tree_id());
      }

      static inline void log_instance_specialized_constraint(ApEvent inst_event,
                                  SpecializedKind kind, ReductionOpID redop)
      {
        log_spy.print("Instance Specialized Constraint " IDFMT " %d %d",
                      inst_event.id, kind, redop);
      }

      static inline void log_instance_memory_constraint(ApEvent inst_event,
                                                     Memory::Kind kind)
      {
        log_spy.print("Instance Memory Constraint " IDFMT " %d", 
                      inst_event.id, kind);
      }

      static inline void log_instance_field_constraint(ApEvent inst_event,
                      bool contiguous, bool inorder, size_t num_fields)
      {
        log_spy.print("Instance Field Constraint " IDFMT " %d %d %zd",
            inst_event.id, (contiguous ? 1 : 0), (inorder ? 1 : 0), num_fields);
      }

      static inline void log_instance_field_constraint_field(ApEvent inst_event,
                                                             FieldID fid)
      {
        log_spy.print("Instance Field Constraint Field " IDFMT " %d",
                      inst_event.id, fid);
      }

      static inline void log_instance_ordering_constraint(ApEvent inst_event,
                                  bool contiguous, size_t num_dimensions)
      {
        log_spy.print("Instance Ordering Constraint " IDFMT " %d %zd",
                      inst_event.id, (contiguous ? 1 : 0), num_dimensions);
      }

      static inline void log_instance_ordering_constraint_dimension(
                                    ApEvent inst_event, DimensionKind dim)
      {
        log_spy.print("Instance Ordering Constraint Dimension " IDFMT " %d",
                      inst_event.id, dim);
      }

      static inline void log_instance_splitting_constraint(ApEvent inst_event,
                              DimensionKind dim, size_t value, bool chunks)
      {
        log_spy.print("Instance Splitting Constraint " IDFMT " %d %zd %d",
                      inst_event.id, dim, value, (chunks ? 1 : 0));
      }

      static inline void log_instance_dimension_constraint(ApEvent inst_event,
                        DimensionKind dim, EqualityKind eqk, size_t value)
      {
        log_spy.print("Instance Dimension Constraint " IDFMT " %d %d %zd",
                      inst_event.id, dim, eqk, value);
      }

      static inline void log_instance_alignment_constraint(ApEvent inst_event,
                          FieldID fid, EqualityKind eqk, size_t alignment)
      {
        log_spy.print("Instance Alignment Constraint " IDFMT " %d %d %zd",
                      inst_event.id, fid, eqk, alignment);
      }

      static inline void log_instance_offset_constraint(ApEvent inst_event,
                                      FieldID fid, long offset)
      {
        log_spy.print("Instance Offset Constraint " IDFMT " %d %ld",
                      inst_event.id, fid, offset);
      }

      // Logger calls for mapping decisions
      static inline void log_variant_decision(UniqueID unique_id, unsigned vid)
      {
        log_spy.print("Variant Decision %llu %u", unique_id, vid);
      }

      static inline void log_mapping_decision(UniqueID unique_id, 
                                unsigned index, FieldID fid, ApEvent inst_event)
      {
        log_spy.print("Mapping Decision %llu %d %d " IDFMT "", unique_id,
		      index, fid, inst_event.id);
      }

      static inline void log_post_mapping_decision(UniqueID unique_id, 
                                unsigned index, FieldID fid, ApEvent inst_event)
      {
        log_spy.print("Post Mapping Decision %llu %d %d " IDFMT "", unique_id,
		      index, fid, inst_event.id);
      }

      static inline void log_temporary_instance(UniqueID unique_id,
                                unsigned index, FieldID fid, ApEvent inst_event)
      {
        log_spy.print("Temporary Instance %llu %d %d " IDFMT "", unique_id,
                      index, fid, inst_event.id);
      }

      static inline void log_task_priority(UniqueID unique_id, 
                                           TaskPriority priority)
      {
        log_spy.print("Task Priority %llu %d", unique_id, priority);
      }

      static inline void log_task_processor(UniqueID unique_id, IDType proc_id)
      {
        log_spy.print("Task Processor %llu " IDFMT "", unique_id, proc_id);
      }

      static inline void log_task_premapping(UniqueID unique_id, unsigned index)
      {
        log_spy.print("Task Premapping %llu %d", unique_id, index);
      }

      static inline void log_tunable_value(UniqueID unique_id, unsigned index,
                                    const void *value, size_t num_bytes)
      {
        // Build a hex string for the value 
        // For now the result must be a multiple of 4 bytes
        assert((num_bytes % 4) == 0);
        size_t buffer_size = ((8 * num_bytes) / 4) + 1;
        char *buffer = (char*)malloc(buffer_size);
        unsigned *src = (unsigned*)value;
        unsigned byte_index = 0;
        for (unsigned word_idx = 0; word_idx < (num_bytes/4); word_idx++)
        {
          unsigned word = src[word_idx];
          // Every 4 bits get's a hex character 
          for (unsigned i = 0; i < (8*sizeof(word)/4); i++, byte_index++)
          {
            // Get the next four bits
            unsigned offset = (word >> (i*4)) & 0xF; 
            if (offset < 10)
              buffer[byte_index] = '0' + offset;
            else
              buffer[byte_index] = 'A' + (offset-10);
          }
        }
        buffer[byte_index] = '\0';
        log_spy.print("Task Tunable %llu %d %zd %s\n", 
                      unique_id, index, num_bytes, buffer);
        free(buffer);
      }

      static inline void log_phase_barrier_arrival(UniqueID unique_id,
                                                   ApBarrier barrier)
      {
        log_spy.print("Phase Barrier Arrive %llu " IDFMT "",
                      unique_id, barrier.id);
      }

      static inline void log_phase_barrier_wait(UniqueID unique_id,
                                                ApEvent previous)
      {
        log_spy.print("Phase Barrier Wait %llu " IDFMT "",
                      unique_id, previous.id);
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

      // Logger call for disjoint close operations
      static inline void log_disjoint_close_field(UniqueID close_id,
                                                  FieldID fid)
      {
        log_spy.print("Disjoint Close Field %llu %d", close_id, fid);
      }

      // Logger calls for realm events
      static inline void log_event_dependence(LgEvent one, LgEvent two)
      {
        if (one != two)
          log_spy.print("Event Event " IDFMT " " IDFMT, 
			one.id, two.id);
      }

      static inline void log_ap_user_event(ApUserEvent event)
      {
        log_spy.print("Ap User Event " IDFMT, event.id);
      }

      static inline void log_rt_user_event(RtUserEvent event)
      {
        log_spy.print("Rt User Event " IDFMT, event.id);
      }

      static inline void log_pred_event(PredEvent event)
      {
        log_spy.print("Pred Event " IDFMT, event.id);
      }

      static inline void log_ap_user_event_trigger(ApUserEvent event)
      {
        log_spy.print("Ap User Event Trigger " IDFMT, event.id);
      }

      static inline void log_rt_user_event_trigger(RtUserEvent event)
      {
        log_spy.print("Rt User Event Trigger " IDFMT, event.id);
      }

      static inline void log_pred_event_trigger(PredEvent event)
      {
        log_spy.print("Pred Event Trigger " IDFMT, event.id);
      }

      static inline void log_operation_events(UniqueID uid,
                                              LgEvent pre, LgEvent post)
      {
        log_spy.print("Operation Events %llu " IDFMT " " IDFMT,
		      uid, pre.id, post.id);
      }

      static inline void log_copy_events(UniqueID op_unique_id,
                                         LogicalRegion handle,
                                         LgEvent pre, LgEvent post)
      {
        log_spy.print("Copy Events %llu %d %d %d " IDFMT " " IDFMT,
                      op_unique_id,
                      handle.get_index_space().get_id(),
                      handle.get_field_space().get_id(), handle.get_tree_id(), 
                      pre.id, post.id);
      }

      static inline void log_copy_field(LgEvent post, FieldID src_fid,
                                        ApEvent src_event, FieldID dst_fid,
                                        ApEvent dst_event, ReductionOpID redop)
      {
        log_spy.print("Copy Field " IDFMT " %d " IDFMT " %d " IDFMT " %d",
                  post.id, src_fid, src_event.id, dst_fid, dst_event.id, redop);
      }

      static inline void log_copy_intersect(LgEvent post, int is_region,
                                            IDType index, unsigned field,
                                            unsigned tree_id)
      {
        log_spy.print("Copy Intersect " IDFMT " %d " IDFMT " %d %d",
                      post.id, is_region, index, field, tree_id);
      }

      static inline void log_fill_events(UniqueID op_unique_id,
                                         LogicalRegion handle,
                                         LgEvent pre, LgEvent post,
                                         UniqueID fill_unique_id)
      {
        log_spy.print("Fill Events %llu %d %d %d " IDFMT " " IDFMT " %llu",
		      op_unique_id, handle.get_index_space().get_id(),
		      handle.get_field_space().get_id(), handle.get_tree_id(),
		      pre.id, post.id, fill_unique_id);
      }

      static inline void log_fill_field(LgEvent post, 
                                        FieldID fid, ApEvent dst_event)
      {
        log_spy.print("Fill Field " IDFMT " %d " IDFMT, 
                      post.id, fid, dst_event.id);
      }

      static inline void log_fill_intersect(LgEvent post, int is_region,
                                            IDType index, unsigned field,
                                            unsigned tree_id)
      {
        log_spy.print("Fill Intersect " IDFMT " %d " IDFMT " %d %d",
		      post.id, is_region, index, field, tree_id);
      } 

      static inline void log_deppart_events(UniqueID op_unique_id,
                                            IndexSpace handle,
                                            LgEvent pre, LgEvent post)
      {
        // Realm has an optimization where if it can do the deppart op
        // immediately it just returns the precondition as the postcondition
        // which of course breaks Legion Spy's way of logging deppart
        // operations uniquely as their completion event
        assert(pre != post);
        log_spy.print("Deppart Events %llu %d " IDFMT " " IDFMT,
                      op_unique_id, handle.get_id(), pre.id, post.id);
      }

      static inline void log_replay_operation(UniqueID op_unique_id)
      {
        log_spy.print("Replay Operation %llu", op_unique_id);
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

