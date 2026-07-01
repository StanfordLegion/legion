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

#include "legion/operations/recurrent.h"
#include "legion/contexts/replicate.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceRecurrentOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceRecurrentOp::TraceRecurrentOp(void) : TraceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TraceRecurrentOp::~TraceRecurrentOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::initialize_recurrent(
        InnerContext* ctx, LogicalTrace* tr, LogicalTrace* prev,
        Provenance* prov, bool remove_ref)
    //--------------------------------------------------------------------------
    {
      TraceOp::initialize(
          ctx,
          tr->has_physical_trace() || prev->has_physical_trace() ?
              EXECUTION_FENCE :
              MAPPING_FENCE,
          false /*need future*/, prov);
      trace = tr;
      tracing = false;
      previous = prev;
      has_blocking_call = previous->get_and_clear_blocking_call();
      if (trace == previous)
        has_intermediate_fence = trace->has_intermediate_fence();
      remove_trace_reference = remove_ref;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TraceOp::activate();
      previous = nullptr;
      has_blocking_call = false;
      has_intermediate_fence = false;
      remove_trace_reference = false;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      TraceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* TraceRecurrentOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_RECURRENT_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind TraceRecurrentOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_RECURRENT_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // We don't optimize for recurrent replays of logical analysis
      // at the moment as it doesn't really seem worth it in most cases
      previous->end_logical_trace(this);
      trace->begin_logical_trace(this);
      TraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> ready_events;
      if (trace != previous)
      {
        if (previous->has_physical_trace())
        {
          PhysicalTrace* physical = previous->get_physical_trace();
          if (physical->is_replaying())
            physical->complete_physical_trace(
                this, ready_events, execution_preconditions, has_blocking_call);
        }
        if (trace->has_physical_trace())
        {
          PhysicalTrace* physical = trace->get_physical_trace();
          physical->refresh_condition_sets(this, ready_events);
        }
      }
      else if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        if (physical->is_recording())
          physical->refresh_condition_sets(this, ready_events);
        else if (!physical->get_current_template()->is_idempotent())
        {
          physical->refresh_condition_sets(this, ready_events);
          physical->complete_physical_trace(
              this, ready_events, execution_preconditions, has_blocking_call);
        }
      }
      if (!ready_events.empty())
        enqueue_ready_operation(Runtime::merge_events(ready_events));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void TraceRecurrentOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Check to see if this is a true recurrent replay or not
      if (trace != previous)
      {
        // Not recurrent so complete the previous trace and begin the new one
        if (previous->has_physical_trace())
        {
          PhysicalTrace* physical = previous->get_physical_trace();
          if (physical->is_recording())
            physical->complete_physical_trace(
                this, map_applied_conditions, execution_preconditions,
                has_blocking_call);
        }
        if (trace->has_physical_trace())
        {
          PhysicalTrace* physical = trace->get_physical_trace();
          const bool replaying = physical->begin_physical_trace(
              this, map_applied_conditions, execution_preconditions);
          // Tell the parent whether we are replaying
          parent_ctx->record_physical_trace_replay(mapped_event, replaying);
        }
      }
      else if (trace->has_physical_trace())
      {
        // This is recurrent, so try to do the recurrent replay
        PhysicalTrace* physical = trace->get_physical_trace();
        const bool replaying = physical->replay_physical_trace(
            this, map_applied_conditions, execution_preconditions,
            has_blocking_call, has_intermediate_fence);
        // Tell the parent whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      }
      if (remove_trace_reference && previous->remove_reference())
        delete previous;
      TraceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    bool TraceRecurrentOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate* TraceRecurrentOp::create_fresh_template(
        PhysicalTrace* physical)
    //--------------------------------------------------------------------------
    {
      return new PhysicalTemplate(physical, get_completion_event());
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceRecurrentOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceRecurrentOp::ReplTraceRecurrentOp(void)
      : ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTraceRecurrentOp::~ReplTraceRecurrentOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::initialize_recurrent(
        ReplicateContext* ctx, LogicalTrace* tr, LogicalTrace* prev,
        Provenance* prov, bool remove_ref)
    //--------------------------------------------------------------------------
    {
      ReplTraceOp::initialize(
          ctx,
          tr->has_physical_trace() || prev->has_physical_trace() ?
              EXECUTION_FENCE :
              MAPPING_FENCE,
          false /*need future*/, prov);
      trace = tr;
      tracing = false;
      previous = prev;
      has_blocking_call = previous->get_and_clear_blocking_call();
      if (trace == previous)
        has_intermediate_fence = trace->has_intermediate_fence();
      remove_trace_reference = remove_ref;
      initialize_begin(ctx, trace);
      initialize_complete(ctx);
    }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> >::activate();
      previous = nullptr;
      has_blocking_call = false;
      has_intermediate_fence = false;
      remove_trace_reference = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> >::deactivate(false);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceRecurrentOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_RECURRENT_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ReplTraceRecurrentOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_RECURRENT_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // We don't both optimizing for recurrent replays of logical analysis
      // at the moment as it doesn't really seem worth it in most cases
      previous->end_logical_trace(this);
      trace->begin_logical_trace(this);
      ReplTraceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      bool fence_before = false;
      // This is really subtle so be careful!
      // The ready_events are the local events that we need to trigger before
      // we can move on to trigger_mapping, the fence_events are events that
      // have to be triggered across all the shards before any shard can move
      // on to the trigger_mapping call. The former are for refreshing
      // equivalence sets while the latter are for applying postconditions
      std::set<RtEvent> ready_events, fence_events;
      if (trace != previous)
      {
        if (trace->has_physical_trace())
        {
          PhysicalTrace* physical = trace->get_physical_trace();
          physical->refresh_condition_sets(this, ready_events);
          fence_before = true;
        }
        if (previous->has_physical_trace())
        {
          PhysicalTrace* physical = previous->get_physical_trace();
          if (physical->is_replaying())
            physical->complete_physical_trace(
                this, fence_before ? fence_events : map_applied_conditions,
                execution_preconditions, has_blocking_call);
          else
            fence_before = true;
        }
      }
      else if (trace->has_physical_trace())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        if (physical->is_recording())
        {
          physical->refresh_condition_sets(this, ready_events);
          fence_before = true;
        }
        else if (!physical->get_current_template()->is_idempotent())
        {
          physical->refresh_condition_sets(this, ready_events);
          physical->complete_physical_trace(
              this, fence_events, execution_preconditions, has_blocking_call);
          fence_before = true;
        }
      }
      if (fence_before)
      {
        legion_assert(mapping_fence_barrier.exists());
        if (!fence_events.empty())
          runtime->phase_barrier_arrive(
              mapping_fence_barrier, 1 /*count*/,
              Runtime::merge_events(fence_events));
        else
          runtime->phase_barrier_arrive(mapping_fence_barrier, 1 /*count*/);
        if (!ready_events.empty())
        {
          ready_events.insert(mapping_fence_barrier);
          enqueue_ready_operation(Runtime::merge_events(ready_events));
        }
        else
          enqueue_ready_operation(mapping_fence_barrier);
      }
      else if (!ready_events.empty())
        enqueue_ready_operation(Runtime::merge_events(ready_events));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceRecurrentOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      bool fence_before = false;
      // Check to see if this is a true recurrent replay or not
      if (trace != previous)
      {
        // Not recurrent so complete the previous trace and begin the new one
        if (previous->has_physical_trace())
        {
          PhysicalTrace* physical = previous->get_physical_trace();
          if (physical->is_recording())
          {
            physical->complete_physical_trace(
                this, map_applied_conditions, execution_preconditions,
                has_blocking_call);
            fence_before = true;
          }
        }
        if (trace->has_physical_trace())
        {
          PhysicalTrace* physical = trace->get_physical_trace();
          const bool replaying = physical->begin_physical_trace(
              this, map_applied_conditions, execution_preconditions);
          if (!replaying)
            // have to do the slow barrier here to make sure that
            // all the shards have made their templates for recording
            perform_template_creation_barrier();
          // Tell the parent whether we are replaying
          parent_ctx->record_physical_trace_replay(mapped_event, replaying);
          fence_before = true;
        }
      }
      else if (trace->has_physical_trace())
      {
        // This is recurrent, so try to do the recurrent replay
        PhysicalTrace* physical = trace->get_physical_trace();
        // The only way we no longer have a current template is if it was
        // not idempotent and we had to complete it before the mapping fence
        // If we do have a template and we're recording then we know we also
        // did the mapping fence before this
        fence_before =
            !physical->has_current_template() || physical->is_recording();
        const bool replaying = physical->replay_physical_trace(
            this, map_applied_conditions, execution_preconditions,
            has_blocking_call, has_intermediate_fence);
        if (!replaying && fence_before)
          // Have to do the slow barrier here to make sure that
          // all the shards have made their templates for recording
          perform_template_creation_barrier();
        // Tell the parent whether we are replaying
        parent_ctx->record_physical_trace_replay(mapped_event, replaying);
      }
      if (remove_trace_reference && previous->remove_reference())
        delete previous;
      if (fence_before)
      {
        legion_assert(fence_kind == EXECUTION_FENCE);
        // Perform the normal dexecution fence analysis
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
    bool ReplTraceRecurrentOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      return false;
    }

  }  // namespace Internal
}  // namespace Legion
