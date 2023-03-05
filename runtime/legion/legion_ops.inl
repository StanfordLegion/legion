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
    void Predicated<OP>::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // Register ourselves as a waiter on the predicate if we have one
      if (this->predicate != NULL)
      {
        bool value;
        if (this->predicate->register_waiter(this,this->get_generation(),value))
        {
          AutoLock o_lock(this->op_lock);
          if (value)
            this->predication_state = OP::RESOLVE_TRUE_STATE;
          else
            this->predication_state = OP::RESOLVE_FALSE_STATE;
        }
      }
      Memoizable<OP>::trigger_prepipeline_stage();
    }

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
          this->register_dependence(this->predicate, 
                                    this->predicate->get_generation());
          this->predicate->get_predicate_guards(this->true_guard,
                                                this->false_guard);
        }
        // Then we can do the base initialization
        OP::trigger_dependence_analysis();
      }
      if (this->predicate != NULL)
        this->predicate->remove_predicate_reference();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      bool speculate = false;
      // We don't support speculation for legion spy validation runs
      // as it doesn't really understand the event graphs that get
      // generated because of the predication events
#ifndef LEGION_SPY
      bool perform_query = false;
      if (this->predicate != NULL)
      {
        AutoLock o_lock(this->op_lock);
        perform_query = 
          (this->predication_state != OP::PENDING_ANALYSIS_STATE);
      }
      if (perform_query && this->query_speculate())
        speculate = true;
#endif
      bool trigger = false;
      bool continue_true = false;
      bool continue_false = false;
      {
        AutoLock o_lock(this->op_lock);
        switch (this->predication_state)
        {
          case OP::PENDING_ANALYSIS_STATE:
            {
              if (speculate)
              {
                trigger = true;
                this->predication_state = OP::SPECULATIVE_MAPPING_STATE;
              }
              else
              {
                this->predication_state = OP::WAITING_MAPPING_STATE;
                // Clear the predicates since they won't matter
                this->true_guard = PredEvent::NO_PRED_EVENT;
                this->false_guard = PredEvent::NO_PRED_EVENT;
              }
              break;
            }
          case OP::RESOLVE_TRUE_STATE:
            {
              trigger = true;
              continue_true = true;
              // Clear the predicates since they won't matter
              this->true_guard = PredEvent::NO_PRED_EVENT;
              this->false_guard = PredEvent::NO_PRED_EVENT;
              break;
            }
          case OP::RESOLVE_FALSE_STATE:
            {
              // If we're recording we still need to map this like normal
              // so that the recording can capture it even with the false
              // predicate resolution in case the replay is not false
              if (this->is_recording())
                trigger = true;
              continue_false = true;
              break;
            }
          default:
            assert(false); // should never make it here
        }
      }
      if (continue_true)
        this->resolve_true(false/*speculated*/, false/*launched*/);
      else if (continue_false)
      {
        if (this->runtime->legion_spy_enabled)
          LegionSpy::log_predicated_false_op(this->unique_op_id);
        this->resolve_false(false/*specualted*/,
                            this->is_recording()/*launched*/);
      }
      if (trigger)
        Memoizable<OP>::trigger_ready();
    }

  }; // namespace Internal
}; // namespace Legion 
