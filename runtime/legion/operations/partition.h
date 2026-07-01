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

#ifndef __LEGION_PENDING_PARTITION_H__
#define __LEGION_PENDING_PARTITION_H__

#include "legion/operations/operation.h"
#include "legion/api/future_map_impl.h"

namespace Legion {
  namespace Internal {

    /**
     * \class PendingPartitionOp
     * Pending partition operations are ones that must be deferred
     * in order to move the overhead of computing them off the
     * application cores. In many cases deferring them is also
     * necessary to avoid possible application deadlock with
     * other pending partitions.
     */
    class PendingPartitionOp : public Operation {
    protected:
      enum PendingPartitionKind {
        EQUAL_PARTITION = 0,
        WEIGHT_PARTITION,
        UNION_PARTITION,
        INTERSECTION_PARTITION,
        INTERSECTION_WITH_REGION,
        DIFFERENCE_PARTITION,
        RESTRICTED_PARTITION,
        BY_DOMAIN_PARTITION,
      };
      // Track pending partition operations as thunks
      class PendingPartitionThunk {
      public:
        virtual ~PendingPartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures) = 0;
        virtual void perform_logging(PendingPartitionOp* op) = 0;
        virtual bool is_cross_product(void) const { return false; }
      };
      class EqualPartitionThunk : public PendingPartitionThunk {
      public:
        EqualPartitionThunk(IndexPartition id, size_t g)
          : pid(id), granularity(g)
        { }
        virtual ~EqualPartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_equal_partition(pid, granularity);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        size_t granularity;
      };
      class WeightPartitionThunk : public PendingPartitionThunk {
      public:
        WeightPartitionThunk(IndexPartition id, size_t g)
          : pid(id), granularity(g)
        { }
        virtual ~WeightPartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_weights(pid, futures, granularity);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        size_t granularity;
      };
      class UnionPartitionThunk : public PendingPartitionThunk {
      public:
        UnionPartitionThunk(
            IndexPartition id, IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2)
        { }
        virtual ~UnionPartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_union(pid, handle1, handle2);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class IntersectionPartitionThunk : public PendingPartitionThunk {
      public:
        IntersectionPartitionThunk(
            IndexPartition id, IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2)
        { }
        virtual ~IntersectionPartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_intersection(pid, handle1, handle2);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class IntersectionWithRegionThunk : public PendingPartitionThunk {
      public:
        IntersectionWithRegionThunk(IndexPartition id, IndexPartition p, bool d)
          : pid(id), part(p), dominates(d)
        { }
        virtual ~IntersectionWithRegionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_intersection(pid, part, dominates);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition part;
        const bool dominates;
      };
      class DifferencePartitionThunk : public PendingPartitionThunk {
      public:
        DifferencePartitionThunk(
            IndexPartition id, IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2)
        { }
        virtual ~DifferencePartitionThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_difference(pid, handle1, handle2);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class RestrictedPartitionThunk : public PendingPartitionThunk {
      public:
        RestrictedPartitionThunk(
            IndexPartition id, const void* tran, size_t tran_size,
            const void* ext, size_t ext_size)
          : pid(id), transform(malloc(tran_size)), extent(malloc(ext_size))
        {
          memcpy(transform, tran, tran_size);
          memcpy(extent, ext, ext_size);
        }
        virtual ~RestrictedPartitionThunk(void)
        {
          free(transform);
          free(extent);
        }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_restriction(pid, transform, extent);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        void* const transform;
        void* const extent;
      };
      class FutureMapThunk : public PendingPartitionThunk {
      public:
        FutureMapThunk(IndexPartition id, const FutureMap& fm, bool inter)
          : pid(id), future_map_domain(fm.impl->get_domain()),
            perform_intersections(inter)
        { }
        virtual ~FutureMapThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_partition_by_domain(
              pid, futures, future_map_domain, perform_intersections);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        const Domain future_map_domain;
        bool perform_intersections;
      };
      class CrossProductThunk : public PendingPartitionThunk {
      public:
        CrossProductThunk(
            IndexPartition b, IndexPartition s, LegionColor c, ShardID local,
            const ShardMapping* mapping)
          : base(b), source(s), part_color(c), local_shard(local),
            shard_mapping(mapping)
        { }
        virtual ~CrossProductThunk(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->create_cross_product_partitions(
              base, source, part_color, local_shard, shard_mapping);
        }
        virtual void perform_logging(PendingPartitionOp* op);
        virtual bool is_cross_product(void) const { return true; }
      protected:
        IndexPartition base;
        IndexPartition source;
        LegionColor part_color;
        ShardID local_shard;
        const ShardMapping* shard_mapping;
      };
      class ComputePendingSpace : public PendingPartitionThunk {
      public:
        ComputePendingSpace(
            IndexSpace t, bool is, const std::vector<IndexSpace>& h)
          : is_union(is), is_partition(false), target(t), handles(h)
        { }
        ComputePendingSpace(IndexSpace t, bool is, IndexPartition h)
          : is_union(is), is_partition(true), target(t), handle(h)
        { }
        virtual ~ComputePendingSpace(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          if (is_partition)
            return op->compute_pending_space(target, handle, is_union);
          else
            return op->compute_pending_space(target, handles, is_union);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        bool is_union, is_partition;
        IndexSpace target;
        IndexPartition handle;
        std::vector<IndexSpace> handles;
      };
      class ComputePendingDifference : public PendingPartitionThunk {
      public:
        ComputePendingDifference(
            IndexSpace t, IndexSpace i, const std::vector<IndexSpace>& h)
          : target(t), initial(i), handles(h)
        { }
        virtual ~ComputePendingDifference(void) { }
      public:
        virtual ApEvent perform(
            PendingPartitionOp* op,
            const std::map<DomainPoint, FutureImpl*>& futures)
        {
          return op->compute_pending_space(target, initial, handles);
        }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexSpace target, initial;
        std::vector<IndexSpace> handles;
      };
    public:
      PendingPartitionOp(void);
      PendingPartitionOp(const PendingPartitionOp& rhs) = delete;
      virtual ~PendingPartitionOp(void);
    public:
      PendingPartitionOp& operator=(const PendingPartitionOp& rhs) = delete;
    public:
      void initialize_equal_partition(
          InnerContext* ctx, IndexPartition pid, size_t granularity,
          Provenance* prov);
      void initialize_weight_partition(
          InnerContext* ctx, IndexPartition pid, const FutureMap& weights,
          size_t granularity, Provenance* provenance);
      void initialize_union_partition(
          InnerContext* ctx, IndexPartition pid, IndexPartition handle1,
          IndexPartition handle2, Provenance* provenance);
      void initialize_intersection_partition(
          InnerContext* ctx, IndexPartition pid, IndexPartition handle1,
          IndexPartition handle2, Provenance* provenance);
      void initialize_intersection_partition(
          InnerContext* ctx, IndexPartition pid, IndexPartition part,
          const bool dominates, Provenance* provenance);
      void initialize_difference_partition(
          InnerContext* ctx, IndexPartition pid, IndexPartition handle1,
          IndexPartition handle2, Provenance* provenance);
      void initialize_restricted_partition(
          InnerContext* ctx, IndexPartition pid, const void* transform,
          size_t transform_size, const void* extent, size_t extent_size,
          Provenance* provenance);
      void initialize_by_domain(
          InnerContext* ctx, IndexPartition pid, const FutureMap& future_map,
          bool perform_intersections, Provenance* provenance);
      void initialize_cross_product(
          InnerContext* ctx, IndexPartition base, IndexPartition source,
          LegionColor color, Provenance* provenance, ShardID local_shard = 0,
          const ShardMapping* shard_mapping = nullptr);
      void initialize_index_space_union(
          InnerContext* ctx, IndexSpace target,
          const std::vector<IndexSpace>& handles, Provenance* provenance);
      void initialize_index_space_union(
          InnerContext* ctx, IndexSpace target, IndexPartition handle,
          Provenance* provenance);
      void initialize_index_space_intersection(
          InnerContext* ctx, IndexSpace target,
          const std::vector<IndexSpace>& handles, Provenance* provenance);
      void initialize_index_space_intersection(
          InnerContext* ctx, IndexSpace target, IndexPartition handle,
          Provenance* provenance);
      void initialize_index_space_difference(
          InnerContext* ctx, IndexSpace target, IndexSpace initial,
          const std::vector<IndexSpace>& handles, Provenance* provenance);
      void perform_logging(void);
    public:
      ApEvent create_equal_partition(IndexPartition pid, size_t granularity);
      ApEvent create_partition_by_weights(
          IndexPartition pid, const std::map<DomainPoint, FutureImpl*>& futures,
          size_t granularity);
      ApEvent create_partition_by_union(
          IndexPartition pid, IndexPartition handle1, IndexPartition handle2);
      ApEvent create_partition_by_intersection(
          IndexPartition pid, IndexPartition handle1, IndexPartition handle2);
      ApEvent create_partition_by_intersection(
          IndexPartition pid, IndexPartition part, const bool dominates);
      ApEvent create_partition_by_difference(
          IndexPartition pid, IndexPartition handle1, IndexPartition handle2);
      ApEvent create_partition_by_restriction(
          IndexPartition pid, const void* transform, const void* extent);
      ApEvent create_partition_by_domain(
          IndexPartition pid, const std::map<DomainPoint, FutureImpl*>& futures,
          const Domain& future_map_domain, bool perform_intersections);
      ApEvent create_cross_product_partitions(
          IndexPartition base, IndexPartition source, LegionColor part_color,
          ShardID shard = 0, const ShardMapping* mapping = nullptr);
    public:
      ApEvent compute_pending_space(
          IndexSpace result, const std::vector<IndexSpace>& handles,
          bool is_union);
      ApEvent compute_pending_space(
          IndexSpace result, IndexPartition handle, bool is_union);
      ApEvent compute_pending_space(
          IndexSpace result, IndexSpace initial,
          const std::vector<IndexSpace>& handles);
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_execution(void) override;
      virtual bool is_partition_op(void) const override { return true; }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    protected:
      virtual void populate_sources(
          const FutureMap& fm, IndexPartition pid, bool need_all_futures);
      virtual bool perform_pending_space(
          IndexSpaceNode* target, bool& broadcast);
      void request_future_buffers(
          std::set<RtEvent>& mapped_events, std::set<RtEvent>& ready_events);
    protected:
      PendingPartitionThunk* thunk;
      FutureMap future_map;
      std::map<DomainPoint, FutureImpl*> sources;
    };

    /**
     * \class ReplPendingPartitionOp
     * A pending partition operation that knows that its
     * being executed in a control replication context
     */
    class ReplPendingPartitionOp : public PendingPartitionOp {
    public:
      ReplPendingPartitionOp(void);
      ReplPendingPartitionOp(const ReplPendingPartitionOp& rhs) = delete;
      virtual ~ReplPendingPartitionOp(void);
    public:
      ReplPendingPartitionOp& operator=(const ReplPendingPartitionOp& rhs) =
          delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void populate_sources(
          const FutureMap& fm, IndexPartition pid,
          bool needs_all_futures) override;
      virtual bool perform_pending_space(
          IndexSpaceNode* target, bool& broadcast) override;
      virtual void trigger_execution(void) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PENDING_PARTITION_H__
