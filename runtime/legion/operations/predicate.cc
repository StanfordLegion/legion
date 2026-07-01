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

#include "legion/operations/predicate.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Predicated Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicatedOp::PredicatedOp(void) : MemoizableOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PredicatedOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MemoizableOp::activate();
      predication_state = PENDING_PREDICATE_STATE;
      predicate = nullptr;
      true_guard = PredEvent::NO_PRED_EVENT;
      false_guard = PredEvent::NO_PRED_EVENT;
    }

    //--------------------------------------------------------------------------
    void PredicatedOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if ((predicate != nullptr) && predicate->remove_reference())
        delete predicate;
      MemoizableOp::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    void PredicatedOp::initialize_predication(
        InnerContext* ctx, const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      if (p == Predicate::TRUE_PRED)
      {
        predication_state = PREDICATED_TRUE_STATE;
        predicate = nullptr;
      }
      else if (p == Predicate::FALSE_PRED)
      {
        predication_state = PREDICATED_FALSE_STATE;
        predicate = nullptr;
      }
      else
      {
        predication_state = PENDING_PREDICATE_STATE;
        predicate = p.impl;
        predicate->add_reference();
        LegionSpy::log_predicate_use(unique_op_id, predicate->creator_uid);
      }
    }

    //--------------------------------------------------------------------------
    bool PredicatedOp::is_predicated_op(void) const
    //--------------------------------------------------------------------------
    {
      return (predicate != nullptr);
    }

    //--------------------------------------------------------------------------
    bool PredicatedOp::get_predicate_value(size_t index)
    //--------------------------------------------------------------------------
    {
      // This should only be called for inlining operations
      legion_assert(!true_guard.exists() && !false_guard.exists());
      if (predication_state == PENDING_PREDICATE_STATE)
      {
        legion_assert(predicate != nullptr);
        bool value = predicate->get_predicate(index, true_guard, false_guard);
        if (false_guard.exists())
          // Wait for the predicate to resolve
          // If false was poisoned then the predicate resolved true
          false_guard.wait_faultaware(value, true /*from application*/);
        return value;
      }
      else
        return (predication_state == PREDICATED_TRUE_STATE);
    }

  }  // namespace Internal
}  // namespace Legion
