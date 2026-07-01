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

#include "legion/operations/complete.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/shard.h"
#include "legion/operations/recurrent.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceCompleteOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceCompleteOp::TraceCompleteOp(void) : TraceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TraceCompleteOp::~TraceCompleteOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::initialize_complete(
        InnerContext* ctx, LogicalTrace* tr, Provenance* provenance,
        bool remove_reference)
    //--------------------------------------------------------------------------
    {
      initialize(
          ctx, tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false /*need future*/, provenance);
      trace = tr;
      tracing = false;
      has_blocking_call = trace->get_and_clear_blocking_call();
      remove_trace_reference = remove_reference;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceCompleteOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind TraceCompleteOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      trace->end_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        physical->complete_physical_trace(
            this, map_applied_conditions, execution_preconditions,
            has_blocking_call);
      }
      if (remove_trace_reference && trace->remove_reference())
        delete trace;
      TraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    bool TraceCompleteOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceComplete
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    ReplTraceComplete<OP>::ReplTraceComplete(void) : OP()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::initialize_complete(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      // Get a collective ID to use for check all replayable
      replayable_collective_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_86);
      idempotent_collective_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_94);
      sync_compute_frontiers_collective_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_92);
      deduplication_collective_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_67);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::activate(void)
    //--------------------------------------------------------------------------
    {
      OP::activate();
      replayable_collective_id = 0;
      replayable_collective = nullptr;
      idempotent_collective_id = 0;
      idempotent_collective = nullptr;
      sync_compute_frontiers_collective_id = 0;
      deduplication_collective_id = 0;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::deactivate(bool free)
    //--------------------------------------------------------------------------
    {
      legion_assert(!free);
      if (replayable_collective != nullptr)
        delete replayable_collective;
      if (idempotent_collective != nullptr)
        delete idempotent_collective;
      OP::deactivate(free);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::begin_replayable_exchange(
        ReplayableStatus status)
    //--------------------------------------------------------------------------
    {
      legion_assert(replayable_collective == nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      replayable_collective =
          new AllReduceCollective<ProdReduction<bool>, false>(
              repl_ctx, replayable_collective_id);
      if (status == REPLAYABLE)
        replayable_collective->async_all_reduce(true);
      else
        replayable_collective->async_all_reduce(false);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::end_replayable_exchange(
        ReplayableStatus& status)
    //--------------------------------------------------------------------------
    {
      legion_assert(replayable_collective != nullptr);
      if (!replayable_collective->get_result() && (status == REPLAYABLE))
        status = NOT_REPLAYABLE_REMOTE_SHARD;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::begin_idempotent_exchange(
        IdempotencyStatus status)
    //--------------------------------------------------------------------------
    {
      legion_assert(idempotent_collective == nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      idempotent_collective =
          new AllReduceCollective<ProdReduction<bool>, false>(
              repl_ctx, idempotent_collective_id);
      if (status == IDEMPOTENT)
        idempotent_collective->async_all_reduce(true);
      else
        idempotent_collective->async_all_reduce(false);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::end_idempotent_exchange(
        IdempotencyStatus& status)
    //--------------------------------------------------------------------------
    {
      legion_assert(idempotent_collective != nullptr);
      if (!idempotent_collective->get_result() && (status == IDEMPOTENT))
        status = NOT_IDEMPOTENT_REMOTE_SHARD;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::sync_compute_frontiers(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      SlowBarrier pre_sync_barrier(
          repl_ctx, sync_compute_frontiers_collective_id);
      pre_sync_barrier.perform_collective_sync(precondition);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceComplete<OP>::deduplicate_condition_sets(
        std::map<EquivalenceSet*, unsigned>& condition_sets)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      // If this replication doesn't span multiple nodes we don't care
      if (repl_ctx->shard_manager->collective_mapping == nullptr)
        return;
      // If the equivalence set doesn't have a collective map then we know
      // that by definition we're the only ones who can know about it so we
      // don't need to exchange it. If it has a collective mapping then try
      // to exchange so the some shard on each node in the replicate context
      // that is in the collective mapping finds it. Note we don't need to
      // worry about deduplicating between several shards on the same node
      // trying to record the condition. They can race and the first one to
      // record the condition will win. We mainly need to get the asymptotic
      // benefits of deduplicating across lots of nodes here.
      TracingSetDeduplication exchange(repl_ctx, deduplication_collective_id);
      for (std::map<EquivalenceSet*, unsigned>::iterator it =
               condition_sets.begin();
           it != condition_sets.end();
           /*nothing*/)
      {
        if (it->first->collective_mapping != nullptr)
        {
          exchange.record_set(it->first->did, it->second);
          std::map<EquivalenceSet*, unsigned>::iterator delete_it = it++;
          condition_sets.erase(delete_it);
        }
        else
          it++;
      }
      // Do the exchange
      const std::map<DistributedID, unsigned>& collective_sets =
          exchange.all_gather_collective_sets();
      // No need to bother if we're not the first local shard on each node
      // for this next part since we just want one shard doing this part
      if (repl_ctx->shard_manager->is_first_local_shard(repl_ctx->owner_shard))
      {
        // For each of the sets set if there is a copy on this node and we are
        // contained in the collective mapping for the equivalence sets then
        // we're going to participate in the
        const AddressSpaceID local_space = runtime->address_space;
        for (const std::pair<const DistributedID, unsigned>& it :
             collective_sets)
        {
          // See if we can find the equivalence set on this node
          EquivalenceSet* set = static_cast<EquivalenceSet*>(
              runtime->weak_find_distributed_collectable(it.first));
          if (set == nullptr)
            continue;
          // If we don't have a collective mapping then this equivalence set
          // was migrated here after it was initially created somewhere else
          if ((set->collective_mapping != nullptr) &&
              set->collective_mapping->contains(local_space))
          {
            // All the nodes in the collective mapping will be represented
            // by at least one shard because this collective mapping had
            // to have been made in this context
            AddressSpaceID capture_space =
                set->select_collective_trace_capture_space();
            if (capture_space == local_space)
              condition_sets[set] = it.second;
          }
          if (set->remove_base_resource_ref(RUNTIME_REF))
            delete set;
        }
      }
    }

    template class ReplTraceComplete<ReplCompleteOp>;
    template class ReplTraceComplete<ReplRecurrentOp>;

    /////////////////////////////////////////////////////////////
    // ReplTraceCompleteOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp::ReplTraceCompleteOp(void)
      : ReplTraceComplete<ReplCompleteOp>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp::~ReplTraceCompleteOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::initialize_complete(
        ReplicateContext* ctx, LogicalTrace* tr, Provenance* provenance,
        bool remove_reference)
    //--------------------------------------------------------------------------
    {
      initialize(
          ctx, tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false /*need future*/, provenance);
      trace = tr;
      tracing = false;
      has_blocking_call = trace->get_and_clear_blocking_call();
      remove_trace_reference = remove_reference;
      ReplTraceComplete<ReplCompleteOp>::initialize_complete(ctx);
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplTraceComplete<ReplCompleteOp>::activate();
      has_blocking_call = false;
      remove_trace_reference = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ReplTraceComplete<ReplCompleteOp>::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceCompleteOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ReplTraceCompleteOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      trace->end_logical_trace(this);
      ReplTraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        if (physical->is_recording())
        {
          // Have to do the mapping fence on the way in to guarantee that
          // everyone is done mapping befor we try to capture conditions
          runtime->phase_barrier_arrive(mapping_fence_barrier, 1 /*count*/);
          enqueue_ready_operation(mapping_fence_barrier);
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      bool fence_before = false;
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        fence_before = physical->is_recording();
        physical->complete_physical_trace(
            this, map_applied_conditions, execution_preconditions,
            has_blocking_call);
      }
      if (remove_trace_reference && trace->remove_reference())
        delete trace;
      if (fence_before)
      {
        legion_assert(fence_kind == EXECUTION_FENCE);
        // Perform the normal execution fence analysis
        parent_ctx->perform_execution_fence_analysis(
            this, execution_preconditions);
        parent_ctx->update_current_execution_fence(
            this, get_completion_event());
        // Now we wrap up the fence, we already did the mapping fence
        // during the trigger ready stage of the pipeline
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        record_completion_effects(execution_preconditions);
        complete_execution();
      }
      else
        ReplTraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    bool ReplTraceCompleteOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    /////////////////////////////////////////////////////////////
    // Tracing Set Deduplication
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TracingSetDeduplication::TracingSetDeduplication(
        ReplicateContext* ctx, CollectiveID id)
      : AllGatherCollective<false>(ctx, id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TracingSetDeduplication::~TracingSetDeduplication(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TracingSetDeduplication::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(collective_sets.size());
      for (const std::pair<const DistributedID, unsigned>& it : collective_sets)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void TracingSetDeduplication::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_sets;
      derez.deserialize(num_sets);
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        derez.deserialize(collective_sets[did]);
      }
    }

    //--------------------------------------------------------------------------
    void TracingSetDeduplication::record_set(DistributedID did, unsigned index)
    //--------------------------------------------------------------------------
    {
      collective_sets[did] = index;
    }

    //--------------------------------------------------------------------------
    const std::map<DistributedID, unsigned>&
        TracingSetDeduplication::all_gather_collective_sets(void)
    //--------------------------------------------------------------------------
    {
      perform_collective_sync();
      return collective_sets;
    }

  }  // namespace Internal
}  // namespace Legion
