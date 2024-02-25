/* Copyright 2023 Stanford University, NVIDIA Corporation
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

// Included from legion_ops.h - do not include this directly

// Useful for IDEs
#include "legion/legion_ops.h"
#include "legion/legion_trace.h"

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
                                              const TraceInfo &trace_info) const
    //--------------------------------------------------------------------------
    {
      // If you get a compiler error here, don't forget that you can statically
      // specialize this method for particular OP types, see FenceOp or 
      // AllReduceOp in runtime.cc
      if (!this->wait_barriers.empty() || !this->grants.empty())
      {
        std::vector<ApEvent> sync_preconditions;
        for (std::vector<PhaseBarrier>::const_iterator it = 
              this->wait_barriers.begin(); it != 
              this->wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          sync_preconditions.push_back(e);
          if (this->runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_wait(this->get_unique_op_id(), e);
        }
        for (std::vector<Grant>::const_iterator it =
              this->grants.begin(); it != this->grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          sync_preconditions.push_back(e);
        }
        if (this->has_execution_fence_event())
          sync_preconditions.push_back(this->get_execution_fence_event());
        ApEvent result = Runtime::merge_events(NULL, sync_preconditions);
        if (this->is_recording())
          trace_info.record_op_sync_event(result);
        return result;
      }
      else // nothing to record since we just depend on the fence
        return this->get_execution_fence_event();
    }

    /////////////////////////////////////////////////////////////
    // Speculative
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Record a mapping dependence on our predicate
      if (this->predicate != NULL)
        this->register_dependence(this->predicate->creator, 
                                  this->predicate->creator_gen);
      // Then we can do the base initialization
      OP::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!this->true_guard.exists() && !this->false_guard.exists());
#endif
      this->set_memoizable_state();
      if (this->predication_state == OP::PENDING_PREDICATE_STATE)
      {
        if (this->is_recording())
          REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
                "Recording of predicated operations is not yet supported")
#ifdef DEBUG_LEGION
        assert(this->predicate != NULL);
#endif
        bool value = this->predicate->get_predicate(
            this->context_index, this->true_guard, this->false_guard);
        bool ready = !this->false_guard.exists();
#ifdef LEGION_SPY
        // We don't support speculation for legion spy validation runs
        // as it doesn't really understand the event graphs that get
        // generated because of the predication events
        if (!ready)
        {
          // If false was poisoned then predicate resolve true
          this->false_guard.wait_faultaware(value, true/*from application*/);
          ready = true;
        }
#endif
        // We do the mapping if we resolve true or if the predicate isn't ready
        // If it's already resolved false then we can take the easy way out
        if (ready && !value)
          this->predication_state = OP::PREDICATED_FALSE_STATE;
        else
          this->predication_state = OP::PREDICATED_TRUE_STATE;
      }
      if (this->predication_state == OP::PREDICATED_FALSE_STATE)
      {
        if (this->is_recording())
          REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
                "Recording of predicated operations is not yet supported")
        if (this->runtime->legion_spy_enabled)
          LegionSpy::log_predicated_false_op(this->unique_op_id);
        this->predicate_false();
      }
      else
        Memoizable<OP>::trigger_ready();
    }

  }; // namespace Internal
}; // namespace Legion 
