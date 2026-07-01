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

// Included from predicate.h - do not include this directly

// Useful for IDEs
#include "legion/operations/predicate.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Predicated
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Record a mapping dependence on our predicate
      if (this->predicate != nullptr)
        this->register_dependence(
            this->predicate->creator, this->predicate->creator_gen);
      // Then we can do the base initialization
      OP::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void Predicated<OP>::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!this->true_guard.exists() && !this->false_guard.exists());
      if (this->predication_state == OP::PENDING_PREDICATE_STATE)
      {
        if (this->is_recording())
        {
          Fatal err;
          err << "Recording of predicated operations is not yet supported";
          err.raise();
        }
        legion_assert(this->predicate != nullptr);
        bool value = this->predicate->get_predicate(
            this->context_index, this->true_guard, this->false_guard);
        bool ready = !this->false_guard.exists();
        // We don't support speculation for legion spy validation runs
        // as it doesn't really understand the event graphs that get
        // generated because of the predication events
        if ((spy_logging_level > LIGHT_SPY_LOGGING) && !ready)
        {
          // If false was poisoned then predicate resolve true
          this->false_guard.wait_faultaware(value, true /*from application*/);
          ready = true;
        }
        // Hold the lock while doing this to prevent races on checking
        // the predication state
        AutoLock o_lock(this->op_lock);
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
        {
          Fatal err;
          err << "Recording of predicated operations is not yet supported";
          err.raise();
        }
        LegionSpy::log_predicated_false_op(this->unique_op_id);
        this->predicate_false();
      }
      else
        OP::trigger_ready();
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    bool Predicated<OP>::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      // TODO: Right now we don't support tracing of predicated operations
      // so we need to disable auto tracing of operations with predicates
      // We can remove this function once tracing supports predicated ops
      switch (this->predication_state)
      {
        case OP::PREDICATED_TRUE_STATE:
          break;
        case OP::PENDING_PREDICATE_STATE:
        case OP::PREDICATED_FALSE_STATE:
          return Operation::record_trace_hash(recorder, opidx);
      }
      return OP::record_trace_hash(recorder, opidx);
    }

  }  // namespace Internal
}  // namespace Legion
