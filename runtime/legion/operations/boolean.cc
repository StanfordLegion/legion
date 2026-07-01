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

#include "legion/operations/boolean.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/api/predicate_impl.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Future Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FuturePredOp::FuturePredOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FuturePredOp::~FuturePredOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FuturePredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      future = Future();
      predicate = Predicate();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* FuturePredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FUTURE_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind FuturePredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FUTURE_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    Predicate FuturePredOp::initialize(
        InnerContext* ctx, const Future& f, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      legion_assert(f.impl != nullptr);
      initialize_operation(ctx, provenance);
      future = f;
      predicate = Predicate(ctx->create_predicate_impl(this));
      to_predicate = true;
      LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
      if (future.impl != nullptr)
        LegionSpy::log_future_use(unique_op_id, future.impl->did);
      return predicate;
    }

    //--------------------------------------------------------------------------
    Future FuturePredOp::initialize(
        InnerContext* ctx, const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      legion_assert(p.impl != nullptr);
      initialize_operation(ctx, provenance);
      predicate = p;
      future = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      to_predicate = false;
      LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
      LegionSpy::log_predicate_use(unique_op_id, p.impl->creator_uid);
      return future;
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (to_predicate)
      {
        legion_assert(future.impl != nullptr);
        // Register this operation as dependent on task that
        // generated the future
        future.impl->register_dependence(this);
      }
      else
      {
        legion_assert(predicate.impl != nullptr);
        register_dependence(
            predicate.impl->creator, predicate.impl->creator_gen);
      }
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Mark that we completed mapping this operation
      if (to_predicate)
      {
        future.impl->request_runtime_instance(this);
        complete_mapping();
        const RtEvent ready = future.impl->find_runtime_instance_ready();
        if (ready.exists() && !ready.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, ready);
        else
          trigger_execution();
      }
      else
      {
        complete_mapping();
        RtEvent ready;
        predicate.impl->get_predicate(ready);
        if (ready.exists())
          parent_ctx->add_to_trigger_execution_queue(this, ready);
        else
          trigger_execution();
      }
    }

    //--------------------------------------------------------------------------
    void FuturePredOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      if (!to_predicate)
      {
        RtEvent ready;
        bool value = predicate.impl->get_predicate(ready);
        legion_assert(!ready.exists());
        FutureInstance* result =
            FutureInstance::create_local(&value, sizeof(value), false /*own*/);
        future.impl->set_result(ApEvent::NO_AP_EVENT, result);
      }
      else
        predicate.impl->set_predicate(
            future.impl->get_boolean_value(parent_ctx));
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Not Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    NotPredOp::NotPredOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    NotPredOp::~NotPredOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Predicate NotPredOp::initialize(
        InnerContext* ctx, const Predicate& p, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      initialize_operation(ctx, provenance);
      to_set = Predicate(ctx->create_predicate_impl(this));
      previous = p;
      LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
      LegionSpy::log_predicate_use(unique_op_id, p.impl->creator_uid);
      return to_set;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
    }

    //--------------------------------------------------------------------------
    void NotPredOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      previous = Predicate();
      to_set = Predicate();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* NotPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[NOT_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind NotPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return NOT_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      register_dependence(previous.impl->creator, previous.impl->creator_gen);
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      RtEvent ready;
      bool value = previous.impl->get_predicate(ready);
      if (!ready.exists())
      {
        to_set.impl->set_predicate(!value);
        complete_execution();
      }
      else
        parent_ctx->add_to_trigger_execution_queue(this, ready);
    }

    //--------------------------------------------------------------------------
    void NotPredOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      RtEvent ready;
      bool value = previous.impl->get_predicate(ready);
      legion_assert(!ready.exists());
      to_set.impl->set_predicate(!value);
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // And Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AndPredOp::AndPredOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AndPredOp::~AndPredOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Predicate AndPredOp::initialize(
        InnerContext* ctx, std::vector<Predicate>& predicates,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      initialize_operation(ctx, provenance);
      to_set = Predicate(ctx->create_predicate_impl(this));
      previous.swap(predicates);
      LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
      for (const Predicate& prev : previous)
        LegionSpy::log_predicate_use(unique_op_id, prev.impl->creator_uid);
      return to_set;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
    }

    //--------------------------------------------------------------------------
    void AndPredOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      previous.clear();
      to_set = Predicate();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* AndPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[AND_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind AndPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return AND_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (const Predicate& prev : previous)
        register_dependence(prev.impl->creator, prev.impl->creator_gen);
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      std::vector<RtEvent> ready_events;
      for (const Predicate& prev : previous)
      {
        RtEvent ready;
        prev.impl->get_predicate(ready);
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists() && !ready.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, ready);
        else
          trigger_execution();
      }
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void AndPredOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (const Predicate& prev : previous)
      {
        RtEvent ready;
        bool value = prev.impl->get_predicate(ready);
        legion_assert(!ready.exists());
        if (value)
          continue;
        result = false;
        break;
      }
      to_set.impl->set_predicate(result);
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Or Predicate Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OrPredOp::OrPredOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OrPredOp::~OrPredOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Predicate OrPredOp::initialize(
        InnerContext* ctx, std::vector<Predicate>& predicates,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      legion_assert(ctx != nullptr);
      initialize_operation(ctx, provenance);
      previous.swap(predicates);
      to_set = Predicate(ctx->create_predicate_impl(this));
      LegionSpy::log_predicate_operation(ctx->get_unique_id(), unique_op_id);
      for (const Predicate& prev : previous)
        LegionSpy::log_predicate_use(unique_op_id, prev.impl->creator_uid);
      return to_set;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
    }

    //--------------------------------------------------------------------------
    void OrPredOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      previous.clear();
      to_set = Predicate();
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* OrPredOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[OR_PRED_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind OrPredOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return OR_PRED_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (const Predicate& prev : previous)
        register_dependence(prev.impl->creator, prev.impl->creator_gen);
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      std::vector<RtEvent> ready_events;
      for (const Predicate& prev : previous)
      {
        RtEvent ready;
        prev.impl->get_predicate(ready);
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists() && !ready.has_triggered())
          parent_ctx->add_to_trigger_execution_queue(this, ready);
        else
          trigger_execution();
      }
      else
        trigger_execution();
    }

    //--------------------------------------------------------------------------
    void OrPredOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      bool result = false;
      for (const Predicate& prev : previous)
      {
        RtEvent ready;
        bool value = prev.impl->get_predicate(ready);
        legion_assert(!ready.exists());
        if (!value)
          continue;
        result = true;
        break;
      }
      to_set.impl->set_predicate(result);
      complete_execution();
    }

  }  // namespace Internal
}  // namespace Legion
