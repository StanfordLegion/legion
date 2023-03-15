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
    void Memoizable<OP>::initialize_memoizable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!this->is_memoizing()); // should always be no memo here
#endif
      PhysicalTrace *const physical_trace = (this->trace == NULL) ? NULL :
          this->trace->get_physical_trace();
      // Only invoke memoization if we are doing physical tracing
      if (physical_trace != NULL)
      {
        this->tpl = physical_trace->get_current_template();
        if ((this->tpl == NULL) || !this->tpl->is_replaying())
          this->invoke_memoize_operation();
        else
          this->memo_state = OP::MEMO_REQ; // replaying so memoization required
      }
#ifdef DEBUG_LEGION
      assert(!this->is_memoizing() || (this->memo_state == OP::MEMO_REQ));
#endif
      if (this->memo_state == OP::MEMO_REQ)
      {
        if (this->tpl == NULL)
        {
          this->trace->set_state_record();
          TaskTreeCoordinates coordinates;
          this->compute_task_tree_coordinates(coordinates);
          this->tpl =
            physical_trace->start_new_template(std::move(coordinates));
#ifdef DEBUG_LEGION
          assert(this->tpl != NULL);
#endif
        }
        if (this->tpl->is_replaying())
        {
#ifdef DEBUG_LEGION
          assert(this->trace->is_replaying());
          assert(this->tpl->is_replaying());
#endif
          this->memo_state = OP::MEMO_REPLAY;
          this->trigger_replay();
        }
        // Check that all the recorders agree on recording
        else if (physical_trace->check_memoize_consensus(this->trace_local_id))
        {
#ifdef DEBUG_LEGION
          assert(this->trace->is_recording());
          assert(this->tpl->is_recording());
#endif
          this->memo_state = OP::MEMO_RECORD;
        }
        else
          this->memo_state = OP::NO_MEMO;
      }
      else if (this->tpl != NULL)
      {
#ifdef DEBUG_LEGION
        assert(this->tpl->is_recording());
#endif
        this->tpl->record_no_consensus();
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Memoizable<OP>::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (!this->is_replaying())
        OP::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Memoizable<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (this->is_recording())
        this->tpl->record_completion_event(this->get_completion_event(),
              this->get_operation_kind(), this->get_trace_local_id());
      OP::trigger_ready();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    ApEvent Memoizable<OP>::compute_sync_precondition(
                                              const TraceInfo &trace_info) const
    //--------------------------------------------------------------------------
    {
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
            LegionSpy::log_phase_barrier_wait(this->unique_op_id, e);
        }
        for (std::vector<Grant>::const_iterator it =
              this->grants.begin(); it != this->grants.end(); it++)
        {
          ApEvent e = it->impl->acquire_grant();
          sync_preconditions.push_back(e);
        }
        if (this->execution_fence_event.exists())
          sync_preconditions.push_back(this->execution_fence_event);
        ApEvent result = Runtime::merge_events(NULL, sync_preconditions);
        if (this->is_recording())
          trace_info.record_op_sync_event(result);
        return result;
      }
      else // nothing to record since we just depend on the fence
        return this->execution_fence_event;
    }

    /////////////////////////////////////////////////////////////
    // Speculative
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (!this->is_replaying())
      {
        // Record a mapping dependence on our predicate
        if (this->predicate != NULL)
        {
          if (this->is_recording())
            REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
                "Recording of predicated operations is not yet supported")
          this->register_dependence(this->predicate->creator, 
                                    this->predicate->creator_gen);
        }
        // Then we can do the base initialization
        OP::trigger_dependence_analysis();
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!this->true_guard.exists() && !this->false_guard.exists());
#endif
      if (this->predication_state == OP::PENDING_PREDICATE_STATE)
      {
#ifdef DEBUG_LEGION
        assert(this->predicate != NULL);
#endif
        bool value =
          this->predicate->get_predicate(this->true_guard, this->false_guard);
        bool ready = !this->false_guard.exists();
#ifdef LEGION_SPY
        // We don't support speculation for legion spy validation runs
        // as it doesn't really understand the event graphs that get
        // generated because of the predication events
        if (!ready)
        {
          // If false was poisoned then predicate resolve true
          this->false_guard.wait_faultaware(value);
          ready = true;
        }
#endif
        // We do the mapping if we resolve true or if the predicate isn't ready
        // If it's already resolved false then we can take the easy way out
        if (ready && !value)
          this->predication_state = OP::RESOLVE_FALSE_STATE;
        else
          this->predication_state = OP::RESOLVE_TRUE_STATE;
      }
      if (this->predication_state == OP::RESOLVE_FALSE_STATE)
      {
        if (this->runtime->legion_spy_enabled)
          LegionSpy::log_predicated_false_op(this->unique_op_id);
        this->predicate_false();
      }
      else
        Memoizable<OP>::trigger_ready();
    }

  }; // namespace Internal
}; // namespace Legion 
