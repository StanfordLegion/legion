/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/operations/partition.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Pending Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingPartitionOp::PendingPartitionOp(void) : Operation(), thunk(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PendingPartitionOp::~PendingPartitionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_equal_partition(
        InnerContext* ctx, IndexPartition pid, size_t granularity,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new EqualPartitionThunk(pid, granularity);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_weight_partition(
        InnerContext* ctx, IndexPartition pid, const FutureMap& weights,
        size_t granularity, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new WeightPartitionThunk(pid, granularity);
      // Also save this locally for analysis
      populate_sources(weights, pid, true /*needs all futures*/);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_union_partition(
        InnerContext* ctx, IndexPartition pid, IndexPartition h1,
        IndexPartition h2, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new UnionPartitionThunk(pid, h1, h2);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(
        InnerContext* ctx, IndexPartition pid, IndexPartition h1,
        IndexPartition h2, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, prov);
      legion_assert(thunk == nullptr);
      thunk = new IntersectionPartitionThunk(pid, h1, h2);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_intersection_partition(
        InnerContext* ctx, IndexPartition pid, IndexPartition part,
        const bool dominates, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, prov);
      legion_assert(thunk == nullptr);
      thunk = new IntersectionWithRegionThunk(pid, part, dominates);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_difference_partition(
        InnerContext* ctx, IndexPartition pid, IndexPartition h1,
        IndexPartition h2, Provenance* prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, prov);
      legion_assert(thunk == nullptr);
      thunk = new DifferencePartitionThunk(pid, h1, h2);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_restricted_partition(
        InnerContext* ctx, IndexPartition pid, const void* transform,
        size_t transform_size, const void* extent, size_t extent_size,
        Provenance* prov)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, prov);
      legion_assert(thunk == nullptr);
      thunk = new RestrictedPartitionThunk(
          pid, transform, transform_size, extent, extent_size);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_by_domain(
        InnerContext* ctx, IndexPartition pid, const FutureMap& fm,
        bool perform_intersections, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new FutureMapThunk(pid, fm, perform_intersections);
      // Also save this locally for analysis
      populate_sources(fm, pid, false /*needs all futures*/);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_cross_product(
        InnerContext* ctx, IndexPartition base, IndexPartition source,
        LegionColor part_color, Provenance* provenance, ShardID shard,
        const ShardMapping* mapping)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new CrossProductThunk(base, source, part_color, shard, mapping);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(
        InnerContext* ctx, IndexSpace target,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new ComputePendingSpace(target, true /*union*/, handles);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_union(
        InnerContext* ctx, IndexSpace target, IndexPartition handle,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new ComputePendingSpace(target, true /*union*/, handle);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
        InnerContext* ctx, IndexSpace target,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new ComputePendingSpace(target, false /*union*/, handles);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_intersection(
        InnerContext* ctx, IndexSpace target, IndexPartition handle,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new ComputePendingSpace(target, false /*union*/, handle);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::initialize_index_space_difference(
        InnerContext* ctx, IndexSpace target, IndexSpace initial,
        const std::vector<IndexSpace>& handles, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      legion_assert(thunk == nullptr);
      thunk = new ComputePendingDifference(target, initial, handles);
      perform_logging();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::perform_logging()
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_pending_partition_operation(
          parent_ctx->get_unique_id(), unique_op_id);
      thunk->perform_logging(this);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if ((future_map.impl != nullptr) && (future_map.impl->op != nullptr))
        register_dependence(future_map.impl->op, future_map.impl->op_gen);
      // Recording this as a pending implicit creation
      parent_ctx->update_current_implicit_creation(this);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Give these slightly higher priority since they are likely
      // needed by later operations
      enqueue_ready_operation(
          RtEvent::NO_RT_EVENT, LG_THROUGHPUT_DEFERRED_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::populate_sources(
        const FutureMap& fm, IndexPartition pid, bool needs_all_futures)
    //--------------------------------------------------------------------------
    {
      future_map = fm;
      legion_assert(sources.empty());
      if (future_map.impl != nullptr)
        future_map.impl->get_all_futures(sources);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::request_future_buffers(
        std::set<RtEvent>& mapped_events, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const DomainPoint, FutureImpl*>& it : sources)
      {
        it.second->request_runtime_instance(this);
        const RtEvent ready = it.second->find_runtime_instance_ready();
        if (ready.exists())
          ready_events.insert(ready);
      }
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> mapped_events, ready_events;
      request_future_buffers(mapped_events, ready_events);
      // Can only marked that that this is mapped after we've requested
      // buffers for any futures in the future map we need which may
      // require performing allocations
      if (!mapped_events.empty())
        complete_mapping(Runtime::merge_events(mapped_events));
      else
        complete_mapping();
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists() && !ready.has_triggered())
        {
          parent_ctx->add_to_trigger_execution_queue(this, ready);
          return;
        }
      }
      trigger_execution();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // Perform the partitioning operation
      const ApEvent ready_event = thunk->perform(this, sources);
      if (ready_event.exists())
        record_completion_effect(ready_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (thunk != nullptr)
        delete thunk;
      thunk = nullptr;
      future_map = FutureMap();  // clear any references
      sources.clear();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* PendingPartitionOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[PENDING_PARTITION_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind PendingPartitionOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return PENDING_PARTITION_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::EqualPartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), EQUAL_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::WeightPartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), WEIGHT_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::UnionPartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), UNION_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::IntersectionPartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), INTERSECTION_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::IntersectionWithRegionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), INTERSECTION_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::DifferencePartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), DIFFERENCE_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::RestrictedPartitionThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), RESTRICTED_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::FutureMapThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_target_pending_partition(
          op->unique_op_id, pid.get_id(), BY_DOMAIN_PARTITION);
    }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::CrossProductThunk::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::ComputePendingSpace::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PendingPartitionOp::ComputePendingDifference::perform_logging(
        PendingPartitionOp* op)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_equal_partition(
        IndexPartition pid, size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      return new_part->create_equal_children(this, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_weights(
        IndexPartition pid, const std::map<DomainPoint, FutureImpl*>& weights,
        size_t granularity)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      return new_part->create_by_weights(this, weights, granularity);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_union(
        IndexPartition pid, IndexPartition handle1, IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      IndexPartNode* node1 = runtime->get_node(handle1);
      IndexPartNode* node2 = runtime->get_node(handle2);
      return new_part->create_by_union(this, node1, node2);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_intersection(
        IndexPartition pid, IndexPartition handle1, IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      IndexPartNode* node1 = runtime->get_node(handle1);
      IndexPartNode* node2 = runtime->get_node(handle2);
      return new_part->create_by_intersection(this, node1, node2);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_intersection(
        IndexPartition pid, IndexPartition part, const bool dominates)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      IndexPartNode* node = runtime->get_node(part);
      return new_part->create_by_intersection(this, node, dominates);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_difference(
        IndexPartition pid, IndexPartition handle1, IndexPartition handle2)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      IndexPartNode* node1 = runtime->get_node(handle1);
      IndexPartNode* node2 = runtime->get_node(handle2);
      return new_part->create_by_difference(this, node1, node2);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_restriction(
        IndexPartition pid, const void* transform, const void* extent)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      return new_part->create_by_restriction(transform, extent);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_partition_by_domain(
        IndexPartition pid, const std::map<DomainPoint, FutureImpl*>& weights,
        const Domain& future_map_domain, bool perform_intersections)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* new_part = runtime->get_node(pid);
      return new_part->parent->create_by_domain(
          this, new_part, weights, future_map_domain, perform_intersections);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::create_cross_product_partitions(
        IndexPartition base, IndexPartition source, LegionColor part_color,
        ShardID local_shard, const ShardMapping* shard_mapping)
    //--------------------------------------------------------------------------
    {
      IndexPartNode* base_node = runtime->get_node(base);
      IndexPartNode* source_node = runtime->get_node(source);
      std::set<ApEvent> ready_events;
      if (shard_mapping == nullptr)
      {
        for (ColorSpaceIterator itr(base_node); itr; itr++)
        {
          IndexSpaceNode* child_node = base_node->get_child(*itr);
          IndexPartNode* part_node = child_node->get_child(part_color);
          ApEvent ready =
              child_node->create_by_intersection(this, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      else if (
          ((LegionColor)shard_mapping->size()) <= base_node->total_children)
      {
        for (ColorSpaceIterator itr(
                 base_node, local_shard, shard_mapping->size());
             itr; itr++)
        {
          IndexSpaceNode* child_node = base_node->get_child(*itr);
          IndexPartNode* part_node = child_node->get_child(part_color);
          ApEvent ready =
              child_node->create_by_intersection(this, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      else
      {
        const unsigned color_index = local_shard % base_node->total_children;
        // See if we're the first local shard on this address space
        bool first_local_shard = true;
        for (ShardID shard = color_index; shard < shard_mapping->size();
             shard += base_node->total_children)
        {
          const AddressSpaceID space = (*shard_mapping)[shard];
          if (space != runtime->address_space)
            continue;
          first_local_shard = (shard == local_shard);
          break;
        }
        if (first_local_shard)
        {
          LegionColor child_color = color_index;
          if (base_node->total_children < base_node->max_linearized_color)
          {
            unsigned index = 0;
            for (ColorSpaceIterator itr(base_node); itr; itr++, index++)
            {
              if (index != color_index)
                continue;
              child_color = *itr;
              break;
            }
          }
          IndexSpaceNode* child_node = base_node->get_child(child_color);
          IndexPartNode* part_node = child_node->get_child(part_color);
          ApEvent ready =
              child_node->create_by_intersection(this, part_node, source_node);
          ready_events.insert(ready);
        }
      }
      return Runtime::merge_events(nullptr, ready_events);
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::compute_pending_space(
        IndexSpace target, const std::vector<IndexSpace>& handles,
        bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* child_node = runtime->get_node(target);
      bool broadcast = true;
      if (perform_pending_space(child_node, broadcast))
        return child_node->compute_pending_space(
            this, handles, is_union, broadcast);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::compute_pending_space(
        IndexSpace target, IndexPartition handle, bool is_union)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* child_node = runtime->get_node(target);
      bool broadcast = true;
      if (perform_pending_space(child_node, broadcast))
        return child_node->compute_pending_space(
            this, handle, is_union, broadcast);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent PendingPartitionOp::compute_pending_space(
        IndexSpace target, IndexSpace initial,
        const std::vector<IndexSpace>& handles)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* child_node = runtime->get_node(target);
      bool broadcast = true;
      if (perform_pending_space(child_node, broadcast))
        return child_node->compute_pending_difference(
            this, initial, handles, broadcast);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    bool PendingPartitionOp::perform_pending_space(
        IndexSpaceNode* target, bool& broadcast)
    //--------------------------------------------------------------------------
    {
      legion_assert(broadcast);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Repl Pending Partition Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp::ReplPendingPartitionOp(void) : PendingPartitionOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp::~ReplPendingPartitionOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp::activate();
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      PendingPartitionOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::populate_sources(
        const FutureMap& fm, IndexPartition pid, bool needs_all_futures)
    //--------------------------------------------------------------------------
    {
      future_map = fm;
      legion_assert(sources.empty());
      legion_assert(future_map.impl != nullptr);
      if (future_map.impl != nullptr)
      {
        if (!needs_all_futures)
        {
          IndexPartNode* partition = runtime->get_node(pid);
          const Domain future_map_domain = future_map.impl->get_domain();
          for (ColorSpaceIterator itr(partition, true /*local only*/); itr;
               itr++)
          {
            const DomainPoint point =
                partition->color_space->delinearize_color_to_point(*itr);
            if (!future_map_domain.contains(point))
              continue;
            Future f = future_map.impl->get_future(point, true /*internal*/);
            sources[point] = f.impl;
          }
        }
        else
          future_map.impl->get_all_futures(sources);
      }
    }

    //--------------------------------------------------------------------------
    bool ReplPendingPartitionOp::perform_pending_space(
        IndexSpaceNode* target, bool& broadcast)
    //--------------------------------------------------------------------------
    {
      legion_assert(broadcast);
      // We know we are in a replicate context
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      const CollectiveMapping* shard_collective =
          repl_ctx->shard_manager->collective_mapping;
      legion_assert(shard_collective != nullptr);
      // Check to see if the target is created collectively
      if ((target->collective_mapping == nullptr) ||
          !shard_collective->contains(*target->collective_mapping))
      {
        // Whichever node is closest to the owner is the one that does the work
        const AddressSpaceID closest =
            shard_collective->find_nearest(target->owner_space);
        return (closest == repl_ctx->local_space);
      }
      else
      {
        broadcast = false;
        // Only do the work if we're on one of the nodes for the target
        return target->collective_mapping->contains(repl_ctx->local_space);
      }
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      // We know we are in a replicate context
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Perform the partitioning operation
      ApEvent ready_event;
      // One the first shard will perform the pending partition computations
      if (repl_ctx->shard_manager->is_first_local_shard(repl_ctx->owner_shard))
        ready_event = thunk->perform(this, sources);
      else if (thunk->is_cross_product())
        ready_event = thunk->perform(this, sources);
      if (ready_event.exists())
        record_completion_effect(ready_event);
      complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
