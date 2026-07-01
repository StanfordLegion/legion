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

// Included from memoizable.h - do not include this directly

// Useful for IDEs
#include "legion/operations/memoizable.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Memoizable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void Memoizable<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      this->set_memoizable_state();
      if (this->is_replaying())
      {
        OP::trigger_replay();
        if (this->tpl->can_start_replay())
          this->tpl->start_replay();
      }
      else
        OP::trigger_ready();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    ApEvent Memoizable<OP>::compute_sync_precondition(
        const TraceInfo& trace_info) const
    //--------------------------------------------------------------------------
    {
      // If you get a compiler error here, don't forget that you can statically
      // specialize this method for particular OP types, see FenceOp or
      // AllReduceOp in runtime.cc
      if (!this->wait_barriers.empty() || !this->grants.empty())
      {
        std::vector<ApEvent> sync_preconditions;
        for (const PhaseBarrier& bar : this->wait_barriers)
        {
          ApEvent e = Runtime::get_previous_phase(bar.phase_barrier);
          sync_preconditions.emplace_back(e);
          LegionSpy::log_phase_barrier_wait(this->get_unique_op_id(), e);
        }
        for (const Grant& grant : this->grants)
        {
          ApEvent e = grant.impl->acquire_grant();
          sync_preconditions.emplace_back(e);
        }
        if (this->has_execution_fence_event())
          sync_preconditions.emplace_back(this->get_execution_fence_event());
        ApEvent result = Runtime::merge_events(nullptr, sync_preconditions);
        if (this->is_recording())
          trace_info.record_op_sync_event(result);
        return result;
      }
      else  // nothing to record since we just depend on the fence
        return this->get_execution_fence_event();
    }

  }  // namespace Internal
}  // namespace Legion
