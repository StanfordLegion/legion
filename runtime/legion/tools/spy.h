/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "legion/tools/types.h"

namespace Legion {
  namespace Internal {
    namespace LegionSpy {

      // One time logger calls to record what gets logged
      static inline void log_legion_spy_config(void)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Legion Spy Detailed Logging");
        else if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Legion Spy Logging");
      }

      // Logger calls for the machine architecture
      static inline void log_processor_kind(unsigned kind, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Processor Kind %d %s", kind, name);
      }

      static inline void log_memory_kind(unsigned kind, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Memory Kind %d %s", kind, name);
      }

      static inline void log_processor(IDType unique_id, unsigned kind)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Processor " IDFMT " %u", unique_id, kind);
      }

      static inline void log_memory(
          IDType unique_id, size_t capacity, unsigned kind)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Memory " IDFMT " %zu %u", unique_id, capacity, kind);
      }

      static inline void log_proc_mem_affinity(
          IDType proc_id, IDType mem_id, unsigned bandwidth, unsigned latency)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Processor Memory " IDFMT " " IDFMT " %u %u", proc_id, mem_id,
              bandwidth, latency);
      }

      static inline void log_mem_mem_affinity(
          IDType mem1, IDType mem2, unsigned bandwidth, unsigned latency)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Memory Memory " IDFMT " " IDFMT " %u %u", mem1, mem2, bandwidth,
              latency);
      }

      // Logger calls for the shape of region trees
      static inline void log_top_index_space(
          DistributedID unique_id, AddressSpaceID owner,
          const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Index Space %llu %u %.*s", unique_id, owner,
              int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_index_space_name(
          DistributedID unique_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Index Space Name %llu %s", unique_id, name);
      }

      static inline void log_index_partition(
          DistributedID parent_id, DistributedID unique_id, int disjoint,
          int complete, LegionColor point, AddressSpaceID owner,
          const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          // Convert ints from -1,0,1 to 0,1,2
          log_spy.print(
              "Index Partition %llu %llu %d %d %lld %u %.*s", parent_id,
              unique_id, disjoint + 1, complete + 1, point, owner,
              int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_index_partition_name(
          DistributedID unique_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Index Partition Name %llu %s", unique_id, name);
      }

      static inline void log_index_subspace(
          DistributedID parent_id, DistributedID unique_id,
          AddressSpaceID owner, const DomainPoint& point)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        legion_assert(point.get_dim() > 0);
        legion_assert(point.get_dim() <= LEGION_MAX_DIM);
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Index Subspace " << parent_id << " " << unique_id << " "
                << owner << " " << point.dim;
        for (int dim = 0; dim < point.dim; dim++)
          message << " " << point.point_data[dim];
      }

      static inline void log_field_space(
          DistributedID unique_id, AddressSpaceID owner,
          const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Field Space %llu %u %.*s", unique_id, owner,
              int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_field_space_name(
          DistributedID unique_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Field Space Name %llu %s", unique_id, name);
      }

      static inline void log_field_creation(
          DistributedID unique_id, unsigned field_id, size_t size,
          const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Field Creation %llu %u %ld %.*s", unique_id, field_id,
              long(size), int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_field_name(
          DistributedID unique_id, unsigned field_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Field Name %llu %u %s", unique_id, field_id, name);
      }

      static inline void log_top_region(
          DistributedID index_space, DistributedID field_space,
          unsigned tree_id, AddressSpaceID owner,
          const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Region %llu %llu %u %u %.*s", index_space, field_space, tree_id,
              owner, int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_logical_region_name(
          DistributedID index_space, DistributedID field_space,
          unsigned tree_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Logical Region Name %llu %llu %u %s", index_space, field_space,
              tree_id, name);
      }

      static inline void log_logical_partition_name(
          DistributedID index_partition, DistributedID field_space,
          unsigned tree_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Logical Partition Name %llu %llu %u %s", index_partition,
              field_space, tree_id, name);
      }

      // For capturing information about the shape of index spaces
      template<int DIM, typename T>
      static inline void log_index_space_point(
          DistributedID handle, const Point<DIM, T>& point)
      {
        static_assert(DIM > 0, "DIM must be positive");
        static_assert(DIM <= LEGION_MAX_DIM, "DIM exceeds LEGION_MAX_DIM");
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Index Space Point " << handle << " " << DIM;
        for (int dim = 0; dim < DIM; dim++) message << " " << point[dim];
      }

      template<int DIM, typename T>
      static inline void log_index_space_rect(
          DistributedID handle, const Rect<DIM, T>& rect)
      {
        static_assert(DIM > 0, "DIM must be positive");
        static_assert(DIM <= LEGION_MAX_DIM, "DIM exceeds LEGION_MAX_DIM");
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Index Space Rect " << handle << " " << DIM;
        for (int dim = 0; dim < DIM; dim++)
          message << " " << rect.lo[dim] << " " << rect.hi[dim];
      }

      static inline void log_empty_index_space(DistributedID handle)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Empty Index Space %llu", handle);
      }

      // Index space expression computations
      static inline void log_index_space_expr(
          DistributedID unique_id, IndexSpaceExprID expr_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Index Space Expression %llu %lld", unique_id, expr_id);
      }

      static inline void log_index_space_union(
          IndexSpaceExprID result_id,
          const std::vector<IndexSpaceExprID>& sources)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        const size_t max_chars = 16;
        char* result = (char*)malloc(sources.size() * max_chars);
        char temp[max_chars];
        unsigned idx = 0;
        for (const IndexSpaceExprID& source : sources)
        {
          if (idx > 0)
          {
            snprintf(temp, max_chars, " %lld", source);
            strncat(result, temp, max_chars);
          }
          else
            snprintf(result, max_chars, "%lld", source);
          idx++;
        }
        log_spy.print(
            "Index Space Union %lld %zd %s", result_id, sources.size(), result);
        free(result);
      }

      static inline void log_index_space_intersection(
          IndexSpaceExprID res_id, const std::vector<IndexSpaceExprID>& sources)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        const size_t max_chars = 16;
        char* result = (char*)malloc(sources.size() * max_chars);
        char temp[max_chars];
        unsigned idx = 0;
        for (const IndexSpaceExprID& source : sources)
        {
          if (idx > 0)
          {
            snprintf(temp, max_chars, " %lld", source);
            strncat(result, temp, max_chars);
          }
          else
            snprintf(result, max_chars, " %lld", source);
          idx++;
        }
        log_spy.print(
            "Index Space Intersection %lld %zd %s", res_id, sources.size(),
            result);
        free(result);
      }

      static inline void log_index_space_difference(
          IndexSpaceExprID result_id, IndexSpaceExprID left,
          IndexSpaceExprID right)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Index Space Difference %lld %lld %lld", result_id, left, right);
      }

      // Logger calls for operations
      static inline void log_task_name(TaskID task_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Task ID Name %d %s", task_id, name);
      }

      static inline void log_task_variant(
          TaskID task_id, unsigned variant_id, bool inner, bool leaf,
          bool idempotent, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Task Variant %d %d %d %d %d %s", task_id, variant_id, inner,
              leaf, idempotent, name);
      }

      static inline void log_top_level_task(
          Processor::TaskFuncID task_id, UniqueID parent_ctx_uid,
          UniqueID unique_id, const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Top Task %u %llu %llu %s", task_id, parent_ctx_uid, unique_id,
              name);
      }

      static inline void log_individual_task(
          UniqueID context, UniqueID unique_id, Processor::TaskFuncID task_id,
          const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Individual Task %llu %u %llu %s", context, task_id, unique_id,
              name);
      }

      static inline void log_index_task(
          UniqueID context, UniqueID unique_id, Processor::TaskFuncID task_id,
          const char* name)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Index Task %llu %u %llu %s", context, task_id, unique_id, name);
      }

      static inline void log_inline_task(UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Inline Task %llu", unique_id);
      }

      static inline void log_mapping_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Mapping Operation %llu %llu", context, unique_id);
      }

      static inline void log_fill_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Fill Operation %llu %llu", context, unique_id);
      }

      static inline void log_discard_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Discard Operation %llu %llu", context, unique_id);
      }

      static inline void log_close_operation(
          UniqueID context, UniqueID unique_id, bool is_intermediate_close_op)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Close Operation %llu %llu %u", context, unique_id,
              is_intermediate_close_op ? 1 : 0);
      }

      static inline void log_refinement_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Refinement Operation %llu %llu", context, unique_id);
      }

      static inline void log_reset_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Reset Operation %llu %llu", context, unique_id);
      }

      static inline void log_internal_op_creator(
          UniqueID internal_op_id, UniqueID creator_op_id, int idx)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Internal Operation Creator %llu %llu %d", internal_op_id,
              creator_op_id, idx);
      }

      static inline void log_fence_operation(
          UniqueID context, UniqueID unique_id, bool execution)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Fence Operation %llu %llu %d", context, unique_id,
              execution ? 1 : 0);
      }

      static inline void log_copy_operation(
          UniqueID context, UniqueID unique_id, unsigned copy_kind,
          bool couple_src_indirect, bool couple_dst_indirect)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Copy Operation %llu %llu %u %d %d", context, unique_id,
              copy_kind, couple_src_indirect ? 1 : 0,
              couple_dst_indirect ? 1 : 0);
      }

      static inline void log_acquire_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Acquire Operation %llu %llu", context, unique_id);
      }

      static inline void log_release_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Release Operation %llu %llu", context, unique_id);
      }

      static inline void log_creation_operation(
          UniqueID context, UniqueID creation)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Creation Operation %llu %llu", context, creation);
      }

      static inline void log_deletion_operation(
          UniqueID context, UniqueID deletion, bool unordered)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Deletion Operation %llu %llu %u", context, deletion,
              unordered ? 1 : 0);
      }

      static inline void log_attach_operation(
          UniqueID context, UniqueID attach, bool restricted)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Attach Operation %llu %llu %u", context, attach,
              restricted ? 1 : 0);
      }

      static inline void log_detach_operation(
          UniqueID context, UniqueID detach, bool unordered)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Detach Operation %llu %llu %u", context, detach,
              unordered ? 1 : 0);
      }

      static inline void log_dynamic_collective(
          UniqueID context, UniqueID collective)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Dynamic Collective %llu %llu", context, collective);
      }

      static inline void log_timing_operation(UniqueID context, UniqueID timing)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Timing Operation %llu %llu", context, timing);
      }

      static inline void log_tunable_operation(
          UniqueID context, UniqueID tunable)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Tunable Operation %llu %llu", context, tunable);
      }

      static inline void log_all_reduce_operation(
          UniqueID context, UniqueID reduce)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("All Reduce Operation %llu %llu", context, reduce);
      }

      static inline void log_predicate_operation(
          UniqueID context, UniqueID pred_op)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Predicate Operation %llu %llu", context, pred_op);
      }

      static inline void log_must_epoch_operation(
          UniqueID context, UniqueID must_op)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Must Epoch Operation %llu %llu", context, must_op);
      }

      static inline void log_summary_op_creator(
          UniqueID internal_op_id, UniqueID creator_op_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Summary Operation Creator %llu %llu", internal_op_id,
              creator_op_id);
      }

      static inline void log_dependent_partition_operation(
          UniqueID context, UniqueID unique_id, IDType pid, int kind)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Dependent Partition Operation %llu %llu " IDFMT " %d", context,
              unique_id, pid, kind);
      }

      static inline void log_pending_partition_operation(
          UniqueID context, UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Pending Partition Operation %llu %llu", context, unique_id);
      }

      static inline void log_target_pending_partition(
          UniqueID unique_id, DistributedID pid, int kind)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Pending Partition Target %llu %llu %d", unique_id, pid, kind);
      }

      static inline void log_index_slice(UniqueID index_id, UniqueID slice_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Index Slice %llu %llu", index_id, slice_id);
      }

      static inline void log_slice_slice(UniqueID slice_one, UniqueID slice_two)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Slice Slice %llu %llu", slice_one, slice_two);
      }

      static inline void log_slice_point(
          UniqueID slice_id, UniqueID point_id, const DomainPoint& point)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        legion_assert(point.get_dim() > 0);
        legion_assert(point.get_dim() <= LEGION_MAX_DIM);
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Slice Point " << slice_id << " " << point_id << " "
                << point.dim;
        for (int dim = 0; dim < point.dim; dim++)
          message << " " << point.point_data[dim];
      }

      static inline void log_point_point(UniqueID p1, UniqueID p2)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Point Point %llu %llu", p1, p2);
      }

      static inline void log_index_point(
          UniqueID index_id, UniqueID point_id, const DomainPoint& point)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        legion_assert(point.get_dim() > 0);
        legion_assert(point.get_dim() <= LEGION_MAX_DIM);
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Index Point " << index_id << " " << point_id << " "
                << point.dim;
        for (int dim = 0; dim < point.dim; dim++)
          message << " " << point.point_data[dim];
      }

      static inline void log_replication(
          UniqueID uid, DistributedID repl_id, bool control_replicated)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Replicate Task %llu %llu %d", uid, repl_id,
              (control_replicated ? 1 : 0));
      }

      static inline void log_shard(
          DistributedID repl_id, ShardID sid, UniqueID uid)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Replicate Shard %llu %d %llu", repl_id, sid, uid);
      }

      static inline void log_owner_shard(UniqueID uid, ShardID sid)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Owner Shard %llu %d", uid, sid);
      }

      static inline void log_intra_space_dependence(
          UniqueID point_id, const DomainPoint& point)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        legion_assert(point.get_dim() > 0);
        legion_assert(point.get_dim() <= LEGION_MAX_DIM);
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Intra Space Dependence " << point_id << " " << point.dim;
        for (int dim = 0; dim < point.dim; dim++)
          message << " " << point.point_data[dim];
      }

      static inline void log_operation_provenance(
          UniqueID unique_id, const std::string_view& provenance)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Operation Provenance %llu %.*s", unique_id,
              int(provenance.length()),
              (provenance.length() == 0) ? "" : provenance.data());
      }

      static inline void log_child_operation_index(
          UniqueID parent_id, size_t index, UniqueID child_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Operation Index %llu %zd %llu", parent_id, index, child_id);
      }

      static inline void log_predicated_false_op(UniqueID unique_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Predicate False %lld", unique_id);
      }

      // Logger calls for mapping dependence analysis
      static inline void log_logical_requirement(
          UniqueID unique_id, unsigned index, bool region,
          DistributedID index_component, DistributedID field_component,
          DistributedID tree_id, unsigned privilege, unsigned coherence,
          unsigned redop, DistributedID parent_index)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Logical Requirement %llu %u %u %llu %llu %llu "
              "%u %u %u %llu",
              unique_id, index, region, index_component, field_component,
              tree_id, privilege, coherence, redop, parent_index);
      }

      static inline void log_requirement_fields(
          UniqueID unique_id, unsigned index,
          const std::set<unsigned>& logical_fields)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          for (const unsigned& field : logical_fields)
          {
            log_spy.print(
                "Logical Requirement Field %llu %u %u", unique_id, index,
                field);
          }
      }

      static inline void log_requirement_fields(
          UniqueID unique_id, unsigned index,
          const std::vector<FieldID>& logical_fields)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          for (const FieldID& field : logical_fields)
          {
            log_spy.print(
                "Logical Requirement Field %llu %u %u", unique_id, index,
                field);
          }
      }

      static inline void log_projection_function(
          ProjectionID pid, unsigned depth, bool invertible)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Projection Function %u %u %d", pid, depth, invertible ? 1 : 0);
      }

      static inline void log_requirement_projection(
          UniqueID unique_id, unsigned index, ProjectionID pid)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Logical Requirement Projection %llu %u %u", unique_id, index,
              pid);
      }

      template<int DIM, typename T>
      static inline void log_launch_index_space_rect(
          UniqueID unique_id, const Rect<DIM, T>& rect)
      {
        static_assert(DIM > 0, "DIM must be positive");
        static_assert(DIM <= LEGION_MAX_DIM, "DIM exceeds LEGION_MAX DIM");
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Index Launch Rect " << unique_id << " " << DIM;
        for (int dim = 0; dim < DIM; dim++)
          message << " " << rect.lo[dim] << " " << rect.hi[dim];
      }

      // Logger calls for futures
      static inline void log_future_creation(
          UniqueID creator_id, DistributedID future_did,
          const DomainPoint& point)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        Realm::LoggerMessage&& message = log_spy.print();
        message << "Future Creation " << creator_id << " " << future_did << " "
                << point.dim;
        for (int dim = 0; dim < point.dim; dim++)
          message << " " << point.point_data[dim];
      }

      static inline void log_future_use(
          UniqueID user_id, DistributedID future_did)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Future Usage %llu %llu", user_id, future_did);
      }

      static inline void log_predicate_use(
          UniqueID pred_id, UniqueID previous_predicate)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Predicate Use %llu %llu", pred_id, previous_predicate);
      }

      // Logger call for physical instances
      static inline void log_physical_instance(
          LgEvent inst_event, IDType inst_id, IDType mem_id,
          IndexSpaceExprID expr_id, FieldSpace handle, RegionTreeID tid,
          ReductionOpID redop)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Physical Instance " IDFMT " " IDFMT " " IDFMT
              " %d %lld %lld %lld",
              inst_event.id, inst_id, mem_id, redop, expr_id, handle.get_id(),
              tid);
      }

      static inline void log_physical_instance_field(
          LgEvent inst_event, FieldID field_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Physical Instance Field " IDFMT " %d", inst_event.id, field_id);
      }

      static inline void log_physical_instance_creator(
          LgEvent inst_event, UniqueID creator_id, IDType proc_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Physical Instance Creator " IDFMT " %lld " IDFMT "",
              inst_event.id, creator_id, proc_id);
      }

      static inline void log_physical_instance_creation_region(
          LgEvent inst_event, LogicalRegion handle)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Physical Instance Creation Region " IDFMT " %lld %lld %lld",
              inst_event.id, handle.get_index_space().get_id(),
              handle.get_field_space().get_id(), handle.get_tree_id());
      }

      static inline void log_instance_specialized_constraint(
          LgEvent inst_event, SpecializedKind kind, ReductionOpID redop)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Specialized Constraint " IDFMT " %d %d", inst_event.id,
              kind, redop);
      }

      static inline void log_instance_memory_constraint(
          LgEvent inst_event, Memory::Kind kind)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Memory Constraint " IDFMT " %d", inst_event.id, kind);
      }

      static inline void log_instance_field_constraint(
          LgEvent inst_event, bool contiguous, bool inorder, size_t num_fields)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Field Constraint " IDFMT " %d %d %zd", inst_event.id,
              (contiguous ? 1 : 0), (inorder ? 1 : 0), num_fields);
      }

      static inline void log_instance_field_constraint_field(
          LgEvent inst_event, FieldID fid)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Field Constraint Field " IDFMT " %d", inst_event.id,
              fid);
      }

      static inline void log_instance_ordering_constraint(
          LgEvent inst_event, bool contiguous, size_t num_dimensions)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Ordering Constraint " IDFMT " %d %zd", inst_event.id,
              (contiguous ? 1 : 0), num_dimensions);
      }

      static inline void log_instance_ordering_constraint_dimension(
          LgEvent inst_event, DimensionKind dim)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Ordering Constraint Dimension " IDFMT " %d",
              inst_event.id, dim);
      }

      static inline void log_instance_tiling_constraint(
          LgEvent inst_event, DimensionKind dim, size_t value, bool tiles)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Splitting Constraint " IDFMT " %d %zd %d",
              inst_event.id, dim, value, (tiles ? 1 : 0));
      }

      static inline void log_instance_dimension_constraint(
          LgEvent inst_event, DimensionKind dim, EqualityKind eqk, size_t value)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Dimension Constraint " IDFMT " %d %d %zd",
              inst_event.id, dim, eqk, value);
      }

      static inline void log_instance_alignment_constraint(
          LgEvent inst_event, FieldID fid, EqualityKind eqk, size_t alignment)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Alignment Constraint " IDFMT " %d %d %zd",
              inst_event.id, fid, eqk, alignment);
      }

      static inline void log_instance_offset_constraint(
          LgEvent inst_event, FieldID fid, long offset)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Offset Constraint " IDFMT " %d %ld", inst_event.id, fid,
              offset);
      }

      static inline void log_instance_deletion(
          LgEvent inst_event, LgEvent deletion)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Instance Deletion " IDFMT " " IDFMT, inst_event.id, deletion.id);
      }

      // Logger calls for mapping decisions
      static inline void log_variant_decision(UniqueID unique_id, unsigned vid)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Variant Decision %llu %u", unique_id, vid);
      }

      static inline void log_mapping_decision(
          UniqueID unique_id, unsigned index, FieldID fid, LgEvent inst_event)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Mapping Decision %llu %d %d " IDFMT "", unique_id, index, fid,
              inst_event.id);
      }

      static inline void log_post_mapping_decision(
          UniqueID unique_id, unsigned index, FieldID fid, LgEvent inst_event)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Post Mapping Decision %llu %d %d " IDFMT "", unique_id, index,
              fid, inst_event.id);
      }

      static inline void log_task_priority(
          UniqueID unique_id, TaskPriority priority)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Task Priority %llu %d", unique_id, priority);
      }

      static inline void log_task_processor(UniqueID unique_id, IDType proc_id)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Task Processor %llu " IDFMT "", unique_id, proc_id);
      }

      static inline void log_task_premapping(UniqueID unique_id, unsigned index)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print("Task Premapping %llu %d", unique_id, index);
      }

      static inline char to_ascii(unsigned value)
      {
        return value < 10 ? '0' + value : 'A' + (value - 10);
      }

      static inline void log_tunable_value(
          UniqueID unique_id, unsigned index, const void* value,
          size_t num_bytes)
      {
        if (spy_logging_level == NO_SPY_LOGGING)
          return;
        // Build a hex string for the value
        size_t buffer_size = ((8 * num_bytes) / 4) + 1;
        char* buffer = (char*)malloc(buffer_size);
        unsigned byte_index = 0;

        {
          const unsigned* src = (const unsigned*)value;
          for (unsigned word_idx = 0; word_idx < (num_bytes / 4); word_idx++)
          {
            unsigned word = src[word_idx];
            // Every 4 bits get's a hex character
            for (unsigned i = 0; i < (8 * sizeof(word) / 4); i++, byte_index++)
              // Get the next four bits
              buffer[byte_index] = to_ascii((word >> (i * 4)) & 0xF);
          }
        }
        // Convert remaining bytes
        {
          const char* src = (const char*)value;
          for (unsigned char_index = (num_bytes / 4) * 4;
               char_index < num_bytes; char_index++)
          {
            unsigned word = src[char_index];
            for (unsigned i = 0; i < 2; i++, byte_index++)
              buffer[byte_index] = to_ascii((word >> (i * 4)) & 0xF);
          }
        }
        buffer[byte_index] = '\0';
        log_spy.print(
            "Task Tunable %llu %d %zd %s\n", unique_id, index, num_bytes,
            buffer);
        free(buffer);
      }

      static inline void log_phase_barrier_arrival(
          UniqueID unique_id, ApBarrier barrier)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Phase Barrier Arrive %llu " IDFMT "", unique_id, barrier.id);
      }

      static inline void log_phase_barrier_wait(
          UniqueID unique_id, ApEvent previous)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Phase Barrier Wait %llu " IDFMT "", unique_id, previous.id);
      }

      static inline void log_collective_rendezvous(
          UniqueID unique_id, unsigned requirement_index,
          unsigned analysis_index)
      {
        if (spy_logging_level > NO_SPY_LOGGING)
          log_spy.print(
              "Collective Rendezvous %llu %u %u", unique_id, requirement_index,
              analysis_index);
      }

      // The calls above this ifdef record the basic information about
      // the execution of an application. It is sufficient to show how
      // an application executed, but is insufficient to actually
      // validate the execution by the runtime. The calls below
      // are the more expensive logging calls necessary for
      // checking the correctness of the runtime's behaviour.
      // Logger calls for mapping dependences
      static inline void log_mapping_dependence(
          UniqueID context, UniqueID prev_id, unsigned prev_idx,
          UniqueID next_id, unsigned next_idx, unsigned dep_type,
          bool pointwise = false)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Mapping Dependence %llu %llu %u %llu %u %d %d", context, prev_id,
              prev_idx, next_id, next_idx, dep_type, pointwise ? 1 : 0);
      }

      static inline void log_future_dependence(
          UniqueID context, UniqueID prev_id, UniqueID next_id,
          bool pointwise = false)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Future Dependence %llu %llu %llu %d", context, prev_id, next_id,
              pointwise ? 1 : 0);
      }

      // Logger calls for realm events
      static inline void log_event_dependence(LgEvent one, LgEvent two)
      {
        if ((spy_logging_level > LIGHT_SPY_LOGGING) && (one != two))
          log_spy.print("Event Event " IDFMT " " IDFMT, one.id, two.id);
      }

      static inline void log_reservation_acquire(
          Reservation r, LgEvent pre, LgEvent post)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Reservation " IDFMT " " IDFMT " " IDFMT, r.id, pre.id, post.id);
      }

      static inline void log_ap_user_event(ApUserEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Ap User Event " IDFMT " %llu", event.id, implicit_unique_op_id);
      }

      static inline void log_rt_user_event(RtUserEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Rt User Event " IDFMT " %llu", event.id, implicit_unique_op_id);
      }

      static inline void log_pred_event(PredEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Pred Event " IDFMT, event.id);
      }

      static inline void log_ap_user_event_trigger(ApUserEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Ap User Event Trigger " IDFMT, event.id);
      }

      static inline void log_rt_user_event_trigger(RtUserEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Rt User Event Trigger " IDFMT, event.id);
      }

      static inline void log_pred_event_trigger(PredEvent event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Pred Event Trigger " IDFMT, event.id);
      }

      // We use this call as a special guard call to know when
      // all the logging calls associated with an operation are
      // done which is useful for knowing when log files are
      // incomplete because a job crashes in the middle of a run
      static inline void log_operation_events(
          UniqueID uid, LgEvent pre, LgEvent post)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Operation Events %llu " IDFMT " " IDFMT, uid, pre.id, post.id);
      }

      static inline void log_copy_events(
          UniqueID op_unique_id, IndexSpaceExprID expr_id,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id, LgEvent pre,
          LgEvent post, CollectiveKind collective)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Copy Events %llu %lld %lld %lld " IDFMT " " IDFMT " %d",
              op_unique_id, expr_id, src_tree_id, dst_tree_id, pre.id, post.id,
              collective);
      }

      static inline void log_copy_field(
          LgEvent post, FieldID src_fid, LgEvent src_event, FieldID dst_fid,
          LgEvent dst_event, ReductionOpID redop)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Copy Field " IDFMT " %d " IDFMT " %d " IDFMT " %d", post.id,
              src_fid, src_event.id, dst_fid, dst_event.id, redop);
      }

      static inline void log_indirect_events(
          UniqueID op_unique_id, IndexSpaceExprID expr_id,
          unsigned indirection_id, LgEvent pre, LgEvent post)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Indirect Events %llu %lld %d " IDFMT " " IDFMT, op_unique_id,
              expr_id, indirection_id, pre.id, post.id);
      }

      static inline void log_indirect_field(
          LgEvent post, FieldID src_fid, LgEvent src_event, int src_indirect,
          FieldID dst_fid, LgEvent dst_event, int dst_indirect,
          ReductionOpID redop)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Indirect Field " IDFMT " %d " IDFMT " %d %d " IDFMT " %d %d",
              post.id, src_fid, src_event.id, src_indirect, dst_fid,
              dst_event.id, dst_indirect, redop);
      }

      static inline void log_indirect_instance(
          unsigned indirection_id, unsigned index, LgEvent inst_event,
          FieldID fid)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Indirect Instance %u %u " IDFMT " %d", indirection_id, index,
              inst_event.id, fid);
      }

      static inline void log_indirect_group(
          unsigned indirection_id, unsigned index, LgEvent inst_event,
          IDType index_space)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Indirect Group %u %u " IDFMT " %llu", indirection_id, index,
              inst_event.id, index_space);
      }

      static inline void log_fill_events(
          UniqueID op_unique_id, IndexSpaceExprID expr_id, FieldSpace handle,
          RegionTreeID tree_id, LgEvent pre, LgEvent post,
          UniqueID fill_unique_id, CollectiveKind collective)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Fill Events %llu %lld %lld %lld " IDFMT " " IDFMT " %llu %d",
              op_unique_id, expr_id, handle.get_id(), tree_id, pre.id, post.id,
              fill_unique_id, collective);
      }

      static inline void log_fill_field(
          LgEvent post, FieldID fid, LgEvent dst_event)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print(
              "Fill Field " IDFMT " %d " IDFMT, post.id, fid, dst_event.id);
      }

      static inline void log_deppart_events(
          UniqueID op_unique_id, IndexSpaceExprID expr_id, LgEvent pre,
          LgEvent post, DepPartOpKind op_kind)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
        {
          // Realm has an optimization where if it can do the deppart op
          // immediately it just returns the precondition as the postcondition
          // which of course breaks Legion Spy's way of logging deppart
          // operations uniquely as their completion event
          legion_assert(pre != post);
          log_spy.print(
              "Deppart Events %llu %lld " IDFMT " " IDFMT " %d", op_unique_id,
              expr_id, pre.id, post.id, op_kind);
        }
      }

      // We use this call as a special guard call to know when
      // all the logging calls associated with an operation are
      // done which is useful for knowing when log files are
      // incomplete because a job crashes in the middle of a run
      static inline void log_replay_operation(UniqueID op_unique_id)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          log_spy.print("Replay Operation %llu", op_unique_id);
      }

      // Logging for equivalence set creation
      static inline void log_equivalence_set(
          DistributedID did, IndexSpaceExprID expr_id, RegionTreeID tid)
      {
        if (spy_logging_level > HEAVY_SPY_LOGGING)
          log_spy.print(
              "Equivalence Set %llx %lld %lld %llu", did, expr_id, tid,
              implicit_unique_op_id);
      }

      static inline void log_equivalence_set_use(
          DistributedID did, UniqueID uid, unsigned index)
      {
        if (spy_logging_level > HEAVY_SPY_LOGGING)
          log_spy.print("Equivalence Use %llx %llu %d", did, uid, index);
      }

    }  // namespace LegionSpy
  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SPY_H__
