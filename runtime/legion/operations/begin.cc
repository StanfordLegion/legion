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

#include "legion/operations/begin.h"
#include "legion/contexts/replicate.h"
#include "legion/operations/recurrent.h"
#include "legion/tracing/logical.h"
#include "legion/tracing/shard.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceBeginOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceBeginOp::TraceBeginOp(void) : TraceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TraceBeginOp::~TraceBeginOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TraceBeginOp::initialize_begin(
        InnerContext* ctx, LogicalTrace* tr, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize(
          ctx, tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false /*need future*/, provenance);
      trace = tr;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceBeginOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_BEGIN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind TraceBeginOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_BEGIN_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      trace->begin_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // All our mapping dependences are satisfied, check to see if we're
      // doing a physical replay, if we are then we need to refresh the
      // equivalence sets for all the templates
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        legion_assert(!physical->has_current_template());
        std::set<RtEvent> refresh_ready;
        physical->refresh_condition_sets(this, refresh_ready);
        if (!refresh_ready.empty())
        {
          enqueue_ready_operation(Runtime::merge_events(refresh_ready));
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceBeginOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        const bool replaying = physical->begin_physical_trace(
            this, map_applied_conditions, execution_preconditions);
        // Tell the parent context whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      }
      TraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* TraceBeginOp::create_fresh_template(
        PhysicalTrace* physical)
    //--------------------------------------------------------------------------
    {
      return new PhysicalTemplate(physical, get_completion_event());
    }

    //--------------------------------------------------------------------------
    bool TraceBeginOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceBegin
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    ReplTraceBegin<OP>::ReplTraceBegin(void) : OP()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceBegin<OP>::initialize_begin(
        ReplicateContext* ctx, LogicalTrace* trace)
    //--------------------------------------------------------------------------
    {
      // Allocate template status collective IDs if we might be checking
      if (trace->has_physical_trace())
      {
        for (unsigned idx = 0; idx < ctx->get_max_trace_templates(); idx++)
          status_collective_ids.emplace_back(
              ctx->get_next_collective_index(COLLECTIVE_LOC_91));
        slow_barrier_id = ctx->get_next_collective_index(COLLECTIVE_LOC_95);
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceBegin<OP>::activate(void)
    //--------------------------------------------------------------------------
    {
      OP::activate();
      slow_barrier = nullptr;
      slow_barrier_id = 0;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceBegin<OP>::deactivate(bool free)
    //--------------------------------------------------------------------------
    {
      legion_assert(!free);
      status_collective_ids.clear();
      if (slow_barrier != nullptr)
        delete slow_barrier;
      OP::deactivate(free);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    PhysicalTemplate* ReplTraceBegin<OP>::create_fresh_template(
        PhysicalTrace* physical)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      return new ShardedPhysicalTemplate(
          physical, this->get_completion_event(), repl_ctx);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    bool ReplTraceBegin<OP>::allreduce_template_status(
        bool& valid, bool acquired)
    //--------------------------------------------------------------------------
    {
      legion_assert(!status_collective_ids.empty());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      AllReduceCollective<typename OP::StatusReduction, false> allreduce(
          repl_ctx, status_collective_ids.back());
      if (!acquired)
        valid = false;
      typename OP::TemplateStatus status = {valid, !acquired};
      status = allreduce.sync_all_reduce(status);
      valid = status.all_valid;
      status_collective_ids.pop_back();
      return status.any_not_acquired;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void ReplTraceBegin<OP>::perform_template_creation_barrier(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(slow_barrier_id > 0);
      legion_assert(slow_barrier == nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(this->parent_ctx);
      slow_barrier = new SlowBarrier(repl_ctx, slow_barrier_id);
      slow_barrier->perform_collective_async();
      this->map_applied_conditions.insert(
          slow_barrier->perform_collective_wait(false));
    }

    template class ReplTraceBegin<ReplBeginOp>;
    template class ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> >;

    /////////////////////////////////////////////////////////////
    // ReplTraceBeginOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceBeginOp::ReplTraceBeginOp(void) : ReplTraceBegin<ReplBeginOp>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTraceBeginOp::~ReplTraceBeginOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::initialize_begin(
        ReplicateContext* ctx, LogicalTrace* tr, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize(
          ctx, tr->has_physical_trace() ? EXECUTION_FENCE : MAPPING_FENCE,
          false /*need future*/, provenance);
      trace = tr;
      tracing = false;
      ReplTraceBegin<ReplBeginOp>::initialize_begin(ctx, trace);
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplTraceBegin<ReplBeginOp>::activate();
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ReplTraceBegin<ReplBeginOp>::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceBeginOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_BEGIN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ReplTraceBeginOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_BEGIN_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      trace->begin_logical_trace(this);
      ReplTraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // All our mapping dependences are satisfied, check to see if we're
      // doing a physical replay, if we are then we need to refresh the
      // equivalence sets for all the templates
      if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        legion_assert(mapping_fence_barrier.exists());
        legion_assert(!physical->has_current_template());
        std::set<RtEvent> refresh_ready;
        physical->refresh_condition_sets(this, refresh_ready);
        // Have to do the mapping fence on the way in for physical traces
        // since we need to know everything is done mapping before testing
        // preconditions of any templates. Note that this is safe to do in
        // parallel with the refresh since we're not going to look inside
        // the equivalence sets until later and any refinements to the
        // equivalence sets will do their own barriers across the shards
        runtime->phase_barrier_arrive(mapping_fence_barrier, 1 /*count*/);
        if (!refresh_ready.empty())
        {
          refresh_ready.insert(mapping_fence_barrier);
          enqueue_ready_operation(Runtime::merge_events(refresh_ready));
        }
        else
          enqueue_ready_operation(mapping_fence_barrier);
      }
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (trace->has_physical_trace())
      {
        legion_assert(fence_kind == EXECUTION_FENCE);
        PhysicalTrace* physical = trace->get_physical_trace();
        const bool replaying = physical->begin_physical_trace(
            this, map_applied_conditions, execution_preconditions);
        if (!replaying)
          // Have to do the slow barrier here to make sure that
          // all the shards have made their templates for recording
          perform_template_creation_barrier();
        // Tell the parent context whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
        // Do the normal physical fence analysis
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
    bool ReplTraceBeginOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

  }  // namespace Internal
}  // namespace Legion
