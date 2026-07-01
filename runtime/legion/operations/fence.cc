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

#include "legion/operations/fence.h"
#include "legion/analysis/versioning.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/tracing/recognizer.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Fence Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FenceOp::FenceOp(void) : MemoizableOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FenceOp::~FenceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future FenceOp::initialize(
        InnerContext* ctx, FenceKind kind, bool need_future,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      fence_kind = kind;
      if (need_future)
        result = Future(new FutureImpl(
            parent_ctx, true /*register*/,
            runtime->get_available_distributed_id(), get_provenance(), this));
      LegionSpy::log_fence_operation(
          parent_ctx->get_unique_id(), unique_op_id, (kind == EXECUTION_FENCE));
      return result;
    }

    //--------------------------------------------------------------------------
    void FenceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MemoizableOp::activate();
    }

    //--------------------------------------------------------------------------
    void FenceOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.clear();
      execution_preconditions.clear();
      result = Future();  // clear out our future reference
      MemoizableOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* FenceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FENCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind FenceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FENCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->perform_mapping_fence_analysis(this);
      parent_ctx->update_current_mapping_fence(this);
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const TraceInfo trace_info(this);
      switch (fence_kind)
      {
        case MAPPING_FENCE:
          {
            // Still need to get a callback if we're going to be replaying
            if (is_recording())
              trace_info.record_complete_replay(map_applied_conditions);
            break;
          }
        case EXECUTION_FENCE:
          {
            if (is_recording())
              tpl->record_execution_fence(get_trace_local_id());
            parent_ctx->perform_execution_fence_analysis(
                this, execution_preconditions);
            record_completion_effects(execution_preconditions);
            parent_ctx->update_current_execution_fence(
                this, get_completion_event());
            break;
          }
        default:
          std::abort();  // should never get here
      }
      if (!map_applied_conditions.empty())
        complete_mapping(Runtime::merge_events(map_applied_conditions));
      else
        complete_mapping();
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void FenceOp::complete_replay(ApEvent fence_complete_event)
    //--------------------------------------------------------------------------
    {
      if (fence_complete_event.exists())
        // Handle the case for marking when the copy completes
        record_completion_effect(fence_complete_event);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void FenceOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      if (result.impl != nullptr)
        result.impl->set_result(complete);
      complete_operation(complete);
    }

    //--------------------------------------------------------------------------
    bool FenceOp::record_trace_hash(TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      hasher.hash(fence_kind);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    const VersionInfo& FenceOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void FenceOp::perform_measurement(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called on derived classes
      std::abort();
    }

    //--------------------------------------------------------------------------
    void FenceOp::DeferTimingMeasurementArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      op->perform_measurement();
    }

    /////////////////////////////////////////////////////////////
    // Repl Fence Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplFenceOp::ReplFenceOp(void) : FenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplFenceOp::~ReplFenceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplFenceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      FenceOp::activate();
      mapping_fence_barrier = RtBarrier::NO_RT_BARRIER;
      execution_fence_barrier = ApBarrier::NO_AP_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      FenceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      initialize_fence_barriers();
      FenceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::initialize_fence_barriers(ReplicateContext* repl_ctx)
    //--------------------------------------------------------------------------
    {
      legion_assert(!mapping_fence_barrier.exists());
      legion_assert(!execution_fence_barrier.exists());
      if (repl_ctx == nullptr)
        repl_ctx = legion_safe_cast<ReplicateContext*>(parent_ctx);
      // If we get here that means we weren't replayed so make our fences
      mapping_fence_barrier = repl_ctx->get_next_mapping_fence_barrier();
      if (fence_kind == EXECUTION_FENCE)
        execution_fence_barrier = repl_ctx->get_next_execution_fence_barrier();
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const TraceInfo trace_info(this);
      switch (fence_kind)
      {
        case MAPPING_FENCE:
          {
            // Still need to get a callback if we're going to be replaying
            if (is_recording())
              trace_info.record_complete_replay(map_applied_conditions);
            break;
          }
        case EXECUTION_FENCE:
          {
            if (is_recording())
              tpl->record_execution_fence(get_trace_local_id());
            parent_ctx->perform_execution_fence_analysis(
                this, execution_preconditions);
            record_completion_effects(execution_preconditions);
            parent_ctx->update_current_execution_fence(
                this, get_completion_event());
            break;
          }
        default:
          std::abort();  // should never get here
      }
      // Do our arrival
      if (!map_applied_conditions.empty())
        runtime->phase_barrier_arrive(
            mapping_fence_barrier, 1 /*count*/,
            Runtime::merge_events(map_applied_conditions));
      else
        runtime->phase_barrier_arrive(mapping_fence_barrier, 1 /*count*/);
      // We're mapped when everyone is mapped
      complete_mapping(mapping_fence_barrier);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapping_fence_barrier.exists());
      // We don't need the mapping fence barrier
      runtime->phase_barrier_arrive(mapping_fence_barrier, 1 /*count*/);
      FenceOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      if (fence_kind == EXECUTION_FENCE)
      {
        legion_assert(execution_fence_barrier.exists());
        runtime->phase_barrier_arrive(
            execution_fence_barrier, 1 /*count*/, complete);
        FenceOp::trigger_complete(execution_fence_barrier);
      }
      else
        FenceOp::trigger_complete(complete);
    }

  }  // namespace Internal
}  // namespace Legion
