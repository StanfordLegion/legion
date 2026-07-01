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

#include "legion/api/predicate_impl.h"
#include "legion/contexts/replicate.h"
#include "legion/operations/operation.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Predicate
  /////////////////////////////////////////////////////////////

  const Predicate Predicate::TRUE_PRED = Predicate(true);
  const Predicate Predicate::FALSE_PRED = Predicate(false);

  //--------------------------------------------------------------------------
  Predicate::Predicate(void) : impl(nullptr), const_value(true)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Predicate::Predicate(const Predicate& p)
  //--------------------------------------------------------------------------
  {
    const_value = p.const_value;
    impl = p.impl;
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  Predicate::Predicate(Predicate&& p) noexcept
  //--------------------------------------------------------------------------
  {
    const_value = p.const_value;
    impl = p.impl;
    p.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  Predicate::Predicate(bool value) : impl(nullptr), const_value(value)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  Predicate::Predicate(Internal::PredicateImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  Predicate::~Predicate(void)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
  }

  //--------------------------------------------------------------------------
  Predicate& Predicate::operator=(const Predicate& rhs)
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    const_value = rhs.const_value;
    impl = rhs.impl;
    if (impl != nullptr)
      impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  Predicate& Predicate::operator=(Predicate&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    const_value = rhs.const_value;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Predicate Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicateImpl::PredicateImpl(Operation* op)
      : context(op->get_context()), creator(op),
        creator_gen(op->get_generation()), creator_uid(op->get_unique_op_id()),
        value(-1)
    //--------------------------------------------------------------------------
    {
      context->add_base_resource_ref(APPLICATION_REF);
    }

    //--------------------------------------------------------------------------
    PredicateImpl::~PredicateImpl(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(0 <= value);
      if (context->remove_base_resource_ref(APPLICATION_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::get_predicate(
        uint64_t context_index, PredEvent& true_g, PredEvent& false_g)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(predicate_lock);
      if (0 <= value)
        return (0 < value);
      // Not ready yet, make guards if they don't exist yet
      if (!true_guard.exists())
      {
        true_guard = Runtime::create_pred_event();
        false_guard = Runtime::create_pred_event();
      }
      true_g = true_guard;
      false_g = false_guard;
      return false;
    }

    //--------------------------------------------------------------------------
    bool PredicateImpl::get_predicate(RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(predicate_lock);
      if (0 <= value)
        return (0 < value);
      if (!ready_event.exists())
        ready_event = Runtime::create_rt_user_event();
      ready = ready_event;
      return false;
    }

    //--------------------------------------------------------------------------
    void PredicateImpl::set_predicate(bool result)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(predicate_lock);
      legion_assert(value < 0);
      if (result)
        value = 1;
      else
        value = 0;
      if (ready_event.exists())
        Runtime::trigger_event(ready_event);
      if (true_guard.exists())
      {
        if (result)
        {
          Runtime::trigger_event(true_guard);
          Runtime::poison_event(false_guard);
        }
        else
        {
          Runtime::poison_event(true_guard);
          Runtime::trigger_event(false_guard);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Predicate Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PredicateCollective::PredicateCollective(
        ReplPredicateImpl* pred, ReplicateContext* ctx, CollectiveID id)
      : AllReduceCollective<MaxReduction<uint64_t>, false>(ctx, id),
        predicate(pred)
    //--------------------------------------------------------------------------
    {
      predicate->add_reference();
    }

    //--------------------------------------------------------------------------
    RtEvent PredicateCollective::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      const RtEvent result = AllReduceCollective<
          MaxReduction<uint64_t>, false>::post_complete_exchange();
      if (predicate->remove_reference())
        delete predicate;
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Repl Predicate Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplPredicateImpl::ReplPredicateImpl(
        Operation* op, uint64_t coordinate, CollectiveID id)
      : PredicateImpl(op), predicate_coordinate(coordinate), collective_id(id),
        max_observed_index(0), collective(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplPredicateImpl::~ReplPredicateImpl(void)
    //--------------------------------------------------------------------------
    {
      if (collective != nullptr)
        delete collective;
    }

    //--------------------------------------------------------------------------
    bool ReplPredicateImpl::get_predicate(
        uint64_t context_index, PredEvent& true_g, PredEvent& false_g)
    //--------------------------------------------------------------------------
    {
      bool trigger_guards = false;
      AutoLock p_lock(predicate_lock);
      // If the result is true then we can just return that
      if (0 < value)
        return true;
      if (value == 0)
      {
        // For the false case, check to see if we already got the
        // maximum observed false case
        if (collective != nullptr)
          max_observed_index = collective->get_result();
        // Can safely return false here since it's later than the
        // maximum observed index across all the shards so all shards
        // will return the same false decision
        if (max_observed_index < context_index)
          return false;
        // Othewise we fall through and pretend like we don't know that
        // it is false yet so that we align the predication decision
        // across all the shards
        trigger_guards = true;
      }
      // Not ready yet, make guards if they don't exist yet
      if (!true_guard.exists())
      {
        true_guard = Runtime::create_pred_event();
        false_guard = Runtime::create_pred_event();
        if (trigger_guards)
        {
          // We're doing the fall-through case where we know its false
          // but we have to make sure that all the shards do the same
          // thing so we're pretending like we don't know the result yet
          legion_assert(value == 0);
          Runtime::poison_event(true_guard);
          Runtime::trigger_event(false_guard);
        }
      }
      true_g = true_guard;
      false_g = false_guard;
      if (context_index > max_observed_index)
        max_observed_index = context_index;
      return false;
    }

    //--------------------------------------------------------------------------
    void ReplPredicateImpl::set_predicate(bool result)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(predicate_lock);
      legion_assert(value < 0);
      if (!result)  // False
      {
        value = 0;
        if (collective_id > 0)
        {
          ReplicateContext* repl_ctx =
              legion_safe_cast<ReplicateContext*>(context);
          collective = new PredicateCollective(this, repl_ctx, collective_id);
          collective->async_all_reduce(max_observed_index);
        }
      }
      else  // True
        value = 1;
      if (ready_event.exists())
        Runtime::trigger_event(ready_event);
      if (true_guard.exists())
      {
        if (result)
        {
          Runtime::trigger_event(true_guard);
          Runtime::poison_event(false_guard);
        }
        else
        {
          Runtime::poison_event(true_guard);
          Runtime::trigger_event(false_guard);
        }
      }
    }

  }  // namespace Internal
}  // namespace Legion
