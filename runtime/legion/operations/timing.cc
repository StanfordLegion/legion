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

#include "legion/operations/timing.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Timing Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TimingOp::TimingOp(void) : FenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TimingOp::~TimingOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future TimingOp::initialize(
        InnerContext* ctx, const TimingLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Future f = FenceOp::initialize(
          ctx, EXECUTION_FENCE, true /*need future*/, provenance);
      measurement = launcher.measurement;
      LegionSpy::log_timing_operation(ctx->get_unique_id(), unique_op_id);
      return f;
    }

    //--------------------------------------------------------------------------
    void TimingOp::activate(void)
    //--------------------------------------------------------------------------
    {
      FenceOp::activate();
      measured = RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void TimingOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      FenceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* TimingOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TIMING_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind TimingOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TIMING_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      if (complete.exists() && !complete.has_triggered_faultignorant())
      {
        DeferTimingMeasurementArgs args(this);
        measured = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY,
            Runtime::protect_event(complete));
      }
      else
        perform_measurement();
      complete_operation(complete);
    }

    //--------------------------------------------------------------------------
    void TimingOp::perform_measurement(void)
    //--------------------------------------------------------------------------
    {
      switch (measurement)
      {
        case LEGION_MEASURE_SECONDS:
          {
            double value = Realm::Clock::current_time();
            result.impl->set_local(&value, sizeof(value));
            break;
          }
        case LEGION_MEASURE_MICRO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_microseconds();
            result.impl->set_local(&value, sizeof(value));
            break;
          }
        case LEGION_MEASURE_NANO_SECONDS:
          {
            long long value = Realm::Clock::current_time_in_nanoseconds();
            result.impl->set_local(&value, sizeof(value));
            break;
          }
        default:
          std::abort();  // should never get here
      }
    }

    //--------------------------------------------------------------------------
    void TimingOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/, measured);
    }

    /////////////////////////////////////////////////////////////
    // Repl Timing Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTimingOp::ReplTimingOp(void) : ReplFenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTimingOp::~ReplTimingOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future ReplTimingOp::initialize(
        InnerContext* ctx, const TimingLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      Future f = ReplFenceOp::initialize(
          ctx, EXECUTION_FENCE, true /*need future*/, provenance);
      measurement = launcher.measurement;
      LegionSpy::log_timing_operation(ctx->get_unique_id(), unique_op_id);
      return f;
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplFenceOp::activate();
      measured = RtEvent::NO_RT_EVENT;
      timing_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (timing_collective != nullptr)
      {
        delete timing_collective;
        timing_collective = nullptr;
      }
      ReplFenceOp::deactivate(false /*freeop*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTimingOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TIMING_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind ReplTimingOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TIMING_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::trigger_complete(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      legion_assert(execution_fence_barrier.exists());
      runtime->phase_barrier_arrive(
          execution_fence_barrier, 1 /*count*/, complete);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      DeferTimingMeasurementArgs args(this);
      // Shard 0 will handle the timing operation
      if (repl_ctx->owner_shard->shard_id > 0)
      {
        const RtEvent ready = timing_collective->perform_collective_wait();
        measured = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, ready);
      }
      else
        measured = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY,
            Runtime::protect_event(execution_fence_barrier));
      complete_operation(execution_fence_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::perform_measurement(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Shard 0 will handle the timing operation
      if (repl_ctx->owner_shard->shard_id > 0)
      {
        long long value =
            timing_collective->get_value(false /*already waited*/);
        result.impl->set_local(&value, sizeof(value));
      }
      else
      {
        // Perform the measurement and then arrive on the barrier
        // with the result to broadcast it to the other shards
        switch (measurement)
        {
          case LEGION_MEASURE_SECONDS:
            {
              double value = Realm::Clock::current_time();
              result.impl->set_local(&value, sizeof(value));
              long long alt_value = 0;
              static_assert(sizeof(alt_value) == sizeof(value));
              memcpy(&alt_value, &value, sizeof(value));
              timing_collective->broadcast(alt_value);
              break;
            }
          case LEGION_MEASURE_MICRO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_microseconds();
              result.impl->set_local(&value, sizeof(value));
              timing_collective->broadcast(value);
              break;
            }
          case LEGION_MEASURE_NANO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_nanoseconds();
              result.impl->set_local(&value, sizeof(value));
              timing_collective->broadcast(value);
              break;
            }
          default:
            std::abort();  // should never get here
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      commit_operation(true /*deactivate*/, measured);
    }

  }  // namespace Internal
}  // namespace Legion
