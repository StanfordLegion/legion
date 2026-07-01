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

#include "legion/operations/dynamic.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Dynamic Collective Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::DynamicCollectiveOp(void) : MemoizableOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DynamicCollectiveOp::~DynamicCollectiveOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future DynamicCollectiveOp::initialize(
        InnerContext* ctx, const DynamicCollective& dc, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      future = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      collective = dc;
      const ReductionOp* redop = Runtime::get_reduction_op(collective.redop);
      future.impl->set_future_result_size(
          redop->sizeof_rhs, runtime->address_space);
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        LegionSpy::log_dynamic_collective(ctx->get_unique_id(), unique_op_id);
        DomainPoint empty_point;
        LegionSpy::log_future_creation(
            unique_op_id, future.impl->did, empty_point);
      }
      return future;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MemoizableOp::activate();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Free the future
      future = Future();
      MemoizableOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* DynamicCollectiveOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[DYNAMIC_COLLECTIVE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind DynamicCollectiveOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return DYNAMIC_COLLECTIVE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      std::vector<PhaseBarrier> wait_barriers, no_arrival_barriers;
      wait_barriers.emplace_back(collective);
      parent_ctx->perform_barrier_dependence_analysis(
          this, wait_barriers, no_arrival_barriers);
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      ApEvent barrier = Runtime::get_previous_phase(collective.phase_barrier);
      if (!barrier.has_triggered_faultignorant())
      {
        const RtEvent safe = Runtime::protect_event(barrier);
        if (safe.exists() && !safe.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, safe);
        else
          trigger_execution();
      }
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void DynamicCollectiveOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      const ReductionOp* redop = Runtime::get_reduction_op(collective.redop);
      const size_t result_size = redop->sizeof_lhs;
      void* result_buffer = std::malloc(result_size);
      ApBarrier prev = Runtime::get_previous_phase(collective.phase_barrier);
      legion_no_skip_assert(
          Runtime::get_barrier_result(prev, result_buffer, result_size));
      future.impl->set_local(result_buffer, result_size, true /*own*/);
      complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
