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

#include "legion/tracing/template.h"
#include "legion/analysis/overwrite.h"
#include "legion/analysis/valid.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/across.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/region.h"
#include "legion/operations/allreduce.h"
#include "legion/operations/complete.h"
#include "legion/operations/fence.h"
#include "legion/tasks/index.h"
#include "legion/tracing/instructions.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceConditionSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceConditionSet::TraceConditionSet(
        PhysicalTemplate* tpl, unsigned req_index, RegionTreeID tid,
        IndexSpaceExpression* expr, lng::FieldMaskMap<LogicalView>&& vws)
      : EqSetTracker(set_lock), owner(tpl), condition_expr(expr), views(vws),
        tree_id(tid), parent_req_index(req_index), shared(false)
    //--------------------------------------------------------------------------
    {
      condition_expr->add_base_expression_reference(TRACE_REF);
      for (lng::FieldMaskMap<LogicalView>::const_iterator it = views.begin();
           it != views.end(); it++)
        it->first->add_base_gc_ref(TRACE_REF);
      analysis.invalid = nullptr;
    }

    //--------------------------------------------------------------------------
    TraceConditionSet::~TraceConditionSet(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(equivalence_sets.empty());
      legion_assert(analysis.invalid == nullptr);
      if (condition_expr->remove_base_expression_reference(TRACE_REF))
        delete condition_expr;
      for (lng::FieldMaskMap<LogicalView>::const_iterator it = views.begin();
           it != views.end(); it++)
        if (it->first->remove_base_gc_ref(TRACE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::matches(
        IndexSpaceExpression* other_expr,
        const FieldMapView<LogicalView>& other_views) const
    //--------------------------------------------------------------------------
    {
      if (condition_expr != other_expr)
        return false;
      if (views.size() != other_views.size())
        return false;
      for (lng::FieldMaskMap<LogicalView>::const_iterator it = views.begin();
           it != views.end(); it++)
      {
        FieldMapView<LogicalView>::const_iterator finder =
            other_views.find(it->first);
        if (finder == other_views.end())
          return false;
        if (it->second != finder->second)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::invalidate_equivalence_sets(void)
    //--------------------------------------------------------------------------
    {
      lng::FieldMaskMap<EquivalenceSet> to_remove;
      lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree> > to_cancel;
      {
        AutoLock s_lock(set_lock);
        if (current_subscriptions.empty())
        {
          legion_assert(equivalence_sets.empty());
          return;
        }
        // Copy and not remove since we need to see the acknowledgement
        // before we know when it is safe to remove our references
        to_remove.swap(equivalence_sets);
        to_cancel.swap(current_subscriptions);
      }
      for (lng::map<AddressSpaceID, lng::FieldMaskMap<EqKDTree> >::
               const_iterator it = to_cancel.begin();
           it != to_cancel.end(); it++)
        cancel_subscriptions(it->first, FieldMapView(it->second));
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               to_remove.begin();
           it != to_remove.end(); it++)
        if (it->first->remove_base_gc_ref(TRACE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::dump_conditions(void) const
    //--------------------------------------------------------------------------
    {
      TraceViewSet view_set(
          owner->trace->logical_trace->context, 0 /*did*/, condition_expr,
          tree_id);
      for (lng::FieldMaskMap<LogicalView>::const_iterator it = views.begin();
           it != views.end(); it++)
        view_set.insert(it->first, condition_expr, it->second);
      view_set.dump();
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::test_preconditions(
        FenceOp* op, unsigned index, std::vector<RtEvent>& ready_events,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be
      // blocking all other operations from running and changing the
      // equivalence sets while we are here
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      legion_assert(
          !(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
      legion_assert(analysis.invalid == nullptr);
      analysis.invalid =
          new InvalidInstAnalysis(op, index, condition_expr, views);
      analysis.invalid->add_reference();
      std::set<RtEvent> deferral_events;
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               equivalence_sets.begin();
           it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
        if (!overlap)
          continue;
        analysis.invalid->analyze(
            it->first, overlap, deferral_events, applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis.invalid->has_remote_sets())
      {
        const RtEvent ready =
            analysis.invalid->perform_remote(traversal_done, applied_events);
        if (ready.exists() && !ready.has_triggered())
          ready_events.emplace_back(ready);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(analysis.invalid != nullptr);
      const bool result = !analysis.invalid->has_invalid();
      if (analysis.invalid->remove_reference())
        delete analysis.invalid;
      analysis.invalid = nullptr;
      return result;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::test_anticonditions(
        FenceOp* op, unsigned index, std::vector<RtEvent>& ready_events,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be
      // blocking all other operations from running and changing the
      // equivalence sets while we are here
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      legion_assert(
          !(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
      legion_assert(analysis.invalid == nullptr);
      analysis.antivalid =
          new AntivalidInstAnalysis(op, index, condition_expr, views);
      analysis.antivalid->add_reference();
      std::set<RtEvent> deferral_events;
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               equivalence_sets.begin();
           it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
        if (!overlap)
          continue;
        analysis.antivalid->analyze(
            it->first, overlap, deferral_events, applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis.antivalid->has_remote_sets())
      {
        const RtEvent ready =
            analysis.antivalid->perform_remote(traversal_done, applied_events);
        if (ready.exists() && !ready.has_triggered())
          ready_events.emplace_back(ready);
      }
    }

    //--------------------------------------------------------------------------
    bool TraceConditionSet::check_anticonditions(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(analysis.antivalid != nullptr);
      const bool result = !analysis.antivalid->has_antivalid();
      if (analysis.antivalid->remove_reference())
        delete analysis.antivalid;
      analysis.invalid = nullptr;
      return result;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::apply_postconditions(
        FenceOp* op, unsigned index, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence should be
      // blocking all other operations from running and changing the
      // equivalence sets while we are here
      // We should already have refreshed the equivalence sets before we
      // get here so that they should all be up to date
      legion_assert(
          !(views.get_valid_mask() - equivalence_sets.get_valid_mask()));
      // Perform an overwrite analysis for each of the postconditions
      const TraceInfo trace_info(op);
      const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      OverwriteAnalysis* analysis = new OverwriteAnalysis(
          op, index, usage, condition_expr, views,
          PhysicalTraceInfo(trace_info, index), ApEvent::NO_AP_EVENT);
      analysis->add_reference();
      std::set<RtEvent> deferral_events;
      for (lng::FieldMaskMap<EquivalenceSet>::const_iterator it =
               equivalence_sets.begin();
           it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = views.get_valid_mask() & it->second;
        if (!overlap)
          continue;
        analysis->analyze(it->first, overlap, deferral_events, applied_events);
      }
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, applied_events);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    void TraceConditionSet::refresh_equivalence_sets(
        FenceOp* op, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // We should not need the lock here because the fence op should be
      // blocking all other operations from running and changing the
      // equivalence sets while we are here
      const FieldMask invalid_mask =
          views.get_valid_mask() - equivalence_sets.get_valid_mask();
      if (!!invalid_mask)
      {
        AddressSpaceID space = runtime->address_space;
        // Create a user event and store it in equivalence_sets_ready
        const RtUserEvent compute_event = Runtime::create_rt_user_event();
        {
          AutoLock s_lock(set_lock);
          legion_assert(equivalence_sets_ready == nullptr);
          equivalence_sets_ready = new shrt::map<RtUserEvent, FieldMask>();
          equivalence_sets_ready->insert(
              std::make_pair(compute_event, invalid_mask));
        }
        std::vector<EqSetTracker*> targets(1, this);
        std::vector<AddressSpaceID> target_spaces(1, space);
        InnerContext* context = owner->trace->logical_trace->context;
        InnerContext* outermost =
            context->find_parent_physical_context(parent_req_index);
        RtEvent ready = context->compute_equivalence_sets(
            parent_req_index, targets, target_spaces, space, condition_expr,
            invalid_mask);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch a meta-task to finalize this trace condition set
          LgFinalizeEqSetsArgs args(
              this, compute_event, context, outermost, parent_req_index,
              condition_expr);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(
              compute_event, context, outermost, parent_req_index,
              condition_expr, op->get_unique_op_id());
        ready_events.insert(compute_event);
      }
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTemplate::PhysicalTemplate(PhysicalTrace* t, ApEvent fence_event)
      : trace(t), total_replays(1), replayable(REPLAYABLE),
        idempotency(IDEMPOTENT), fence_completion_id(0),
        has_virtual_mapping(false), has_non_leaf_task(false),
        has_variable_return_size(false), has_no_consensus(false),
        last_fence(nullptr), remaining_replays(0), total_logical(0)
    //--------------------------------------------------------------------------
    {
      events.emplace_back(fence_event);
      event_map[fence_event] = fence_completion_id;
      finished_transitive_reduction.store(nullptr);
      instructions.emplace_back(new AssignFenceCompletion(
          *this, fence_completion_id, TraceLocalID()));
    }

    //--------------------------------------------------------------------------
    PhysicalTemplate::~PhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(failure.view == nullptr);
      legion_assert(failure.expr == nullptr);
      {
        AutoLock tpl_lock(template_lock);
        for (TraceConditionSet* pre : preconditions)
        {
          pre->invalidate_equivalence_sets();
          if (pre->remove_reference())
            delete pre;
        }
        for (TraceConditionSet* const & condition_set : anticonditions)
        {
          condition_set->invalidate_equivalence_sets();
          if (condition_set->remove_reference())
            delete condition_set;
        }
        for (TraceConditionSet* const & condition_set : postconditions)
        {
          condition_set->invalidate_equivalence_sets();
          if (condition_set->remove_reference())
            delete condition_set;
        }
        for (Instruction*& instruction : instructions) delete instruction;
        cached_mappings.clear();
      }
      TransitiveReductionState* state = finished_transitive_reduction.load();
      if (state != nullptr)
        delete state;
      for (const std::map<DistributedID, IndividualView*>::value_type&
               view_pair : recorded_views)
        if (view_pair.second->remove_base_gc_ref(TRACE_REF))
          delete view_pair.second;
      for (IndexSpaceExpression* const & expression : recorded_expressions)
        if (expression->remove_base_expression_reference(TRACE_REF))
          delete expression;
      for (PhysicalManager* const & manager : all_instances)
        if (manager->remove_base_gc_ref(TRACE_REF))
          delete manager;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> all_events;
      std::set<ApEvent> local_barriers;
      for (const std::map<ApEvent, BarrierAdvance*>::value_type& barrier_pair :
           managed_barriers)
        local_barriers.insert(barrier_pair.second->get_current_barrier());
      for (const std::map<ApEvent, unsigned>::value_type& event_pair :
           event_map)
        // If this is one of our local barriers then don't use it
        if (local_barriers.find(event_pair.first) == local_barriers.end())
          all_events.insert(event_pair.first);
      return Runtime::merge_events(nullptr, all_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_execution_fence(const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      legion_assert(!events.empty());
      legion_assert(events.size() == instructions.size());
      // This is dumb, in the future we should find the frontiers
      // Scan backwards until we find the previous execution fence (if any)
      // Skip the most recent one as that is going to be our term event
      std::set<unsigned> preconditions;
      for (int idx = events.size() - 1; idx > 0; idx--)
      {
        if (events[idx].exists())
          preconditions.insert(idx);
        if (instructions[idx] == last_fence)
        {
          preconditions.insert(last_fence->complete);
          break;
        }
      }
      if (last_fence == nullptr)
        preconditions.insert(0);
      legion_assert(!preconditions.empty());
      unsigned complete = 0;
      if (preconditions.size() > 1)
      {
        // Record a merge event
        complete = events.size();
        events.emplace_back(ApEvent());
        insert_instruction(
            new MergeEvent(*this, complete, preconditions, tlid));
      }
      else
        complete = *(preconditions.begin());
      events.emplace_back(ApEvent());
      CompleteReplay* fence = new CompleteReplay(*this, tlid, complete);
      insert_instruction(fence);
      // update the last fence
      last_fence = fence;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTemplate::test_preconditions(
        FenceOp* op, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> ready_events;
      for (unsigned idx = 0; idx < preconditions.size(); idx++)
        preconditions[idx]->test_preconditions(
            op, idx, ready_events, applied_events);
      for (unsigned idx = 0; idx < anticonditions.size(); idx++)
        anticonditions[idx]->test_anticonditions(
            op, idx, ready_events, applied_events);
      if (!ready_events.empty())
        return Runtime::merge_events(ready_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::check_preconditions(void)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (TraceConditionSet* const & condition_set : preconditions)
        if (!condition_set->check_preconditions())
          result = false;
      for (TraceConditionSet* const & condition_set : anticonditions)
        if (!condition_set->check_anticonditions())
          result = false;
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::apply_postconditions(
        FenceOp* op, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < postconditions.size(); idx++)
        postconditions[idx]->apply_postconditions(op, idx, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::invalidate_equivalence_sets(void) const
    //--------------------------------------------------------------------------
    {
      for (TraceConditionSet* const & condition_set : preconditions)
        condition_set->invalidate_equivalence_sets();
      for (TraceConditionSet* const & condition_set : anticonditions)
        condition_set->invalidate_equivalence_sets();
      for (TraceConditionSet* const & condition_set : postconditions)
        condition_set->invalidate_equivalence_sets();
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::can_start_replay(void)
    //--------------------------------------------------------------------------
    {
      // This might look racy but its not. We only call this method when we
      // are replaying a physical template which by definition cannot happen
      // on the first trace execution. Therefore the entire logical analysis
      // will have finished recording all the operations before we start
      // replaying a single operation in the physical analysis
      const size_t op_count = trace->logical_trace->get_operation_count();
      const unsigned total = total_logical.fetch_add(1) + 1;
      legion_assert(total <= op_count);
      return (total == op_count);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::register_operation(MemoizableOp* memoizable)
    //--------------------------------------------------------------------------
    {
      legion_assert(memoizable != nullptr);
      const TraceLocalID tid = memoizable->get_trace_local_id();

      AutoLock tpl_lock(template_lock);
      legion_assert(operations.find(tid) == operations.end());
      operations[tid] = memoizable;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::execute_slice(
        unsigned slice_idx, bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(slice_idx < slices.size());
      std::vector<Instruction*>& instructions = slices[slice_idx];
      for (Instruction* instruction : instructions)
        instruction->execute(events, user_events, operations, recurrent_replay);
      unsigned remaining = remaining_replays.fetch_sub(1);
      legion_assert(remaining > 0);
      if (remaining == 1)
      {
        AutoLock tpl_lock(template_lock);
        if (replay_postcondition.exists())
          Runtime::trigger_event(replay_postcondition);
      }
    }

    //--------------------------------------------------------------------------
    ReplayableStatus PhysicalTemplate::finalize(
        CompleteOp* op, bool has_blocking_call)
    //--------------------------------------------------------------------------
    {
      if (has_no_consensus.load())
        replayable = NOT_REPLAYABLE_CONSENSUS;
      else if (has_blocking_call)
        replayable = NOT_REPLAYABLE_BLOCKING;
      else if (has_virtual_mapping)
        replayable = NOT_REPLAYABLE_VIRTUAL;
      else if (has_non_leaf_task)
        replayable = NOT_REPLAYABLE_NON_LEAF;
      else if (has_variable_return_size)
        replayable = NOT_REPLAYABLE_VARIABLE_RETURN;
      op->begin_replayable_exchange(replayable);
      idempotency = capture_conditions(op);
      op->begin_idempotent_exchange(idempotency);
      op->end_replayable_exchange(replayable);
      if (is_replayable())
      {
        // The user can't ask for both no transitive reduction and inlining
        // of the transitive reduction.
        legion_assert(
            !(runtime->no_transitive_reduction &&
              runtime->inline_transitive_reduction));
        // Optimize will sync the idempotency computation
        optimize(op, runtime->inline_transitive_reduction);
        std::fill(events.begin(), events.end(), ApEvent::NO_AP_EVENT);
        event_map.clear();
        // Defer performing the transitive reduction because it might
        // be expensive (see comment above). Note you can only kick off
        // the transitive reduction in the background once all the other
        // optimizations are done so that they don't race on mutating
        // the instruction and event data structures
        if (!runtime->no_trace_optimization &&
            !runtime->no_transitive_reduction &&
            !runtime->inline_transitive_reduction)
        {
          TransitiveReductionState* state =
              new TransitiveReductionState(Runtime::create_rt_user_event());
          transitive_reduction_done = state->done;
          TransitiveReductionArgs args(this, state);
          runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
        }
        // Can dump now if we're not deferring the transitive reduction
        else if (runtime->dump_physical_traces)
          dump_template();
      }
      else
      {
        if (runtime->dump_physical_traces)
        {
          // Optimize will sync the idempotency computation
          optimize(op, !runtime->no_transitive_reduction);
          dump_template();
        }
        else
          op->end_idempotent_exchange(idempotency);
      }
      return replayable;
    }

    //--------------------------------------------------------------------------
    IdempotencyStatus PhysicalTemplate::capture_conditions(CompleteOp* op)
    //--------------------------------------------------------------------------
    {
      // First let's get the equivalence sets with data for these regions
      // We'll use the result to get guide the creation of the trace condition
      // sets. Note we're going to end up recomputing the equivalence sets
      // inside the trace condition sets but that is a small price to pay to
      // minimize the number of conditions that we need to do the capture
      // Next we need to compute the equivalence sets for all these regions
      std::map<EquivalenceSet*, unsigned> current_sets;
      trace->find_condition_sets(current_sets);

      // For cases of control replication we need to deduplicate the
      // condition sets so that each shard only captures one equivalence set
      op->deduplicate_condition_sets(current_sets);

      std::vector<RtEvent> ready_events;
      std::atomic<unsigned> result(IDEMPOTENT);
      for (const std::map<EquivalenceSet*, unsigned>::value_type& set_pair :
           current_sets)
      {
        RtEvent ready = set_pair.first->capture_trace_conditions(
            this, set_pair.first->local_space, set_pair.second, &result);
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      return static_cast<IdempotencyStatus>(result.load());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::receive_trace_conditions(
        TraceViewSet* previews, TraceViewSet* antiviews,
        TraceViewSet* postviews,
        const FieldMapView<IndexSpaceExpression>& unique_dirty_exprs,
        unsigned parent_req_index, RegionTreeID tree_id,
        std::atomic<unsigned>* result)
    //--------------------------------------------------------------------------
    {
      // First check to see if these conditions are idempotent or not
      TraceViewSet::FailedPrecondition fail;
      if ((previews != nullptr) && (postviews != nullptr) &&
          !previews->subsumed_by(*postviews, unique_dirty_exprs, &fail))
      {
        unsigned initial = IDEMPOTENT;
        if (result->compare_exchange_strong(
                initial, NOT_IDEMPOTENT_SUBSUMPTION) &&
            runtime->dump_physical_traces)
        {
          failure = fail;
          if (failure.view != nullptr)
            failure.view->add_base_resource_ref(TRACE_REF);
          if (failure.expr != nullptr)
            failure.expr->add_base_expression_reference(TRACE_REF);
        }
      }
      else if (
          (postviews != nullptr) && (antiviews != nullptr) &&
          !postviews->independent_of(*antiviews, &fail))
      {
        unsigned initial = IDEMPOTENT;
        if (result->compare_exchange_strong(
                initial, NOT_IDEMPOTENT_ANTIDEPENDENT) &&
            runtime->dump_physical_traces)
        {
          failure = fail;
          if (failure.view != nullptr)
            failure.view->add_base_resource_ref(TRACE_REF);
          if (failure.expr != nullptr)
            failure.expr->add_base_expression_reference(TRACE_REF);
        }
      }
      // Now we can convert these views into conditions
      std::vector<TraceConditionSet*> postsets;
      // Create the postconditions first so we can see if we can share
      // them with any of the preconditions or anticonditions
      if (postviews != nullptr)
      {
        local::map<IndexSpaceExpression*, local::FieldMaskMap<LogicalView> >
            expr_views;
        postviews->transpose_uniquely(expr_views);
        postsets.reserve(expr_views.size());
        for (local::map<
                 IndexSpaceExpression*,
                 local::FieldMaskMap<LogicalView> >::iterator it =
                 expr_views.begin();
             it != expr_views.end(); it++)
        {
          TraceConditionSet* set = new TraceConditionSet(
              this, parent_req_index, tree_id, it->first,
              std::move(it->second));
          set->add_reference();
          postsets.emplace_back(set);
        }
        AutoLock tpl_lock(template_lock);
        postconditions.insert(
            postconditions.end(), postsets.begin(), postsets.end());
      }
      // Next do the previews and the antiviews looking for sharing with
      // the postviews so we can minimize the number of EqSetTrackers
      if (previews != nullptr)
      {
        local::map<IndexSpaceExpression*, local::FieldMaskMap<LogicalView> >
            expr_views;
        previews->transpose_uniquely(expr_views);
        for (local::map<
                 IndexSpaceExpression*,
                 local::FieldMaskMap<LogicalView> >::iterator eit =
                 expr_views.begin();
             eit != expr_views.end(); eit++)
        {
          TraceConditionSet* set = nullptr;
          for (std::vector<TraceConditionSet*>::iterator it = postsets.begin();
               it != postsets.end(); it++)
          {
            if (!(*it)->matches(eit->first, eit->second))
              continue;
            set = *it;
            postsets.erase(it);
            break;
          }
          if (set == nullptr)
            set = new TraceConditionSet(
                this, parent_req_index, tree_id, eit->first,
                std::move(eit->second));
          else
            set->mark_shared();
          set->add_reference();
          AutoLock tpl_lock(template_lock);
          preconditions.emplace_back(set);
        }
      }
      if (antiviews != nullptr)
      {
        local::map<IndexSpaceExpression*, local::FieldMaskMap<LogicalView> >
            expr_views;
        antiviews->transpose_uniquely(expr_views);
        for (local::map<
                 IndexSpaceExpression*,
                 local::FieldMaskMap<LogicalView> >::iterator eit =
                 expr_views.begin();
             eit != expr_views.end(); eit++)
        {
          TraceConditionSet* set = nullptr;
          for (std::vector<TraceConditionSet*>::iterator it = postsets.begin();
               it != postsets.end(); it++)
          {
            if (!(*it)->matches(eit->first, eit->second))
              continue;
            set = *it;
            postsets.erase(it);
            break;
          }
          if (set == nullptr)
            set = new TraceConditionSet(
                this, parent_req_index, tree_id, eit->first,
                std::move(eit->second));
          else
            set->mark_shared();
          set->add_reference();
          AutoLock tpl_lock(template_lock);
          anticonditions.emplace_back(set);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::refresh_condition_sets(
        FenceOp* op, std::set<RtEvent>& ready_events) const
    //--------------------------------------------------------------------------
    {
      for (TraceConditionSet* pre : preconditions)
        pre->refresh_equivalence_sets(op, ready_events);
      for (TraceConditionSet* anti : anticonditions)
        anti->refresh_equivalence_sets(op, ready_events);
      for (TraceConditionSet* post : postconditions)
        if (!post->is_shared())
          post->refresh_equivalence_sets(op, ready_events);
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::acquire_instance_references(void) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<PhysicalManager*>::const_iterator it =
               all_instances.begin();
           it != all_instances.end(); it++)
      {
        if (!(*it)->acquire_instance(TRACE_REF))
        {
          // Remove all the references we already added up to now
          // No need to check for deletion, we stil have gc references
          for (std::vector<PhysicalManager*>::const_iterator it2 =
                   all_instances.begin();
               it2 != it; it2++)
            (*it2)->remove_base_valid_ref(TRACE_REF);
          return false;
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::release_instance_references(
        std::set<RtEvent>& map_applied_conditions) const
    //--------------------------------------------------------------------------
    {
      for (PhysicalManager* manager : all_instances)
      {
        // Record the last replay completion event as a user
        // Note the map_applied_conditions are a formality here since we
        // know that we're still holding a valid reference when we do this
        // call so all the work of this operation should be local and no
        // messages should end up being sent
        manager->record_instance_user(replay_complete, map_applied_conditions);
        // No need to check for deletions, we stil hold gc references
        manager->remove_base_valid_ref(TRACE_REF);
      }
      // Also need to release any shadow instances we might have made as
      // part of executing this template
      for (IssueAcross* across : across_copies)
        across->executor->release_shadow_instances();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::optimize(
        CompleteOp* op, bool do_transitive_reduction)
    //--------------------------------------------------------------------------
    {
      legion_assert(instructions.size() == events.size());
      std::vector<RtEvent> frontier_events;
      find_all_last_instance_user_events(frontier_events);
      compute_frontiers(frontier_events);
      // Check to see if the indirection fields for any across copies are
      // mutated during the execution of the trace. If they aren't then we
      // know that we don't need to recompute preimages on back-to-back replays
      // Do this here so we can also use the 'sync_compute_frontiers' barrier
      // to know that all these analyses are done as well
      if (!across_copies.empty())
      {
        for (IssueAcross* across : across_copies)
        {
          std::map<unsigned, InstUsers>::iterator finder =
              src_indirect_insts.find(across->lhs);
          if ((finder != src_indirect_insts.end()) &&
              are_read_only_users(finder->second))
            across->executor->record_trace_immutable_indirection(true /*src*/);
          finder = dst_indirect_insts.find(across->lhs);
          if ((finder != dst_indirect_insts.end()) &&
              are_read_only_users(finder->second))
            across->executor->record_trace_immutable_indirection(false /*dst*/);
        }
      }
      std::vector<unsigned> gen;
      // Sync the idempotency computation
      op->end_idempotent_exchange(idempotency);
      // Fence elision can only be performed if the template is idempotent.
      if (!is_idempotent() || !trace->perform_fence_elision)
      {
        gen.resize(events.size(), 0 /*fence instruction*/);
        for (unsigned idx = 0; idx < instructions.size(); ++idx) gen[idx] = idx;
      }
      else
        elide_fences(gen, frontier_events);
      // Sync the frontier computation so we know that all our frontier data
      // structures such as 'local_frontiers' and 'remote_frontiers' are ready
      sync_compute_frontiers(op, frontier_events);
      if (!runtime->no_trace_optimization)
      {
        propagate_merges(gen);
        if (do_transitive_reduction)
        {
          TransitiveReductionState state(RtUserEvent::NO_RT_USER_EVENT);
          transitive_reduction(&state, false /*deferred*/);
        }
        propagate_copies(&gen);
        eliminate_dead_code(gen);
      }
      prepare_parallel_replay(gen);
      push_complete_replays();
      // After elide fences we can clear these views
      op_insts.clear();
      copy_insts.clear();
      mutated_insts.clear();
      src_indirect_insts.clear();
      dst_indirect_insts.clear();
      instance_last_users.clear();
      // We don't need the expression or view references anymore
      // We do need to translate the recorded views into a vector of instances
      // that need to be acquired in order for the template to be replayed
      all_instances.reserve(recorded_views.size());
      for (const std::pair<const DistributedID, IndividualView*>& it :
           recorded_views)
      {
        PhysicalManager* manager = it.second->get_manager();
        manager->add_base_gc_ref(TRACE_REF);
        all_instances.emplace_back(manager);
        if (it.second->remove_base_gc_ref(TRACE_REF))
          delete it.second;
      }
      recorded_views.clear();
      for (IndexSpaceExpression* expr : recorded_expressions)
        if (expr->remove_base_expression_reference(TRACE_REF))
          delete expr;
      recorded_expressions.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_all_last_instance_user_events(
        std::vector<RtEvent>& frontier_events)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const TraceLocalID, InstUsers>& it : op_insts)
        find_last_instance_events(it.second, frontier_events);
      for (const std::pair<const unsigned, InstUsers>& it : copy_insts)
        find_last_instance_events(it.second, frontier_events);
      for (const std::pair<const unsigned, InstUsers>& it : src_indirect_insts)
        find_last_instance_events(it.second, frontier_events);
      for (const std::pair<const unsigned, InstUsers>& it : dst_indirect_insts)
        find_last_instance_events(it.second, frontier_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_last_instance_events(
        const InstUsers& users, std::vector<RtEvent>& frontier_events)
    //--------------------------------------------------------------------------
    {
      for (InstUsers::const_iterator uit = users.begin(); uit != users.end();
           uit++)
      {
        std::deque<LastUserResult>& results =
            instance_last_users[uit->instance];
        // Scan through all the queries we've done so far for this instance
        // and see if we've already done one for these parameters
        bool found = false;
        for (const LastUserResult& result : results)
        {
          if (!result.user.matches(*uit))
            continue;
          found = true;
          break;
        }
        if (!found)
        {
          results.emplace_back(LastUserResult(*uit));
          LastUserResult& result = results.back();
          std::map<DistributedID, IndividualView*>::const_iterator finder =
              recorded_views.find(uit->instance.view_did);
          legion_assert(finder != recorded_views.end());
          PhysicalManager* manager = finder->second->get_manager();
          legion_assert(manager->did == uit->instance.inst_did);
          const RegionUsage usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
          finder->second->find_last_users(
              manager, result.events, usage, uit->mask, uit->expr,
              frontier_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::compute_frontiers(
        std::vector<RtEvent>& frontier_events)
    //--------------------------------------------------------------------------
    {
      // We need to wait for all the last user instance events to be ready
      if (!frontier_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(frontier_events);
        frontier_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Now we can convert all the results to frontiers
      std::map<ApEvent, unsigned> frontier_map;
      for (std::pair<const UniqueInst, std::deque<LastUserResult> >& lit :
           instance_last_users)
      {
        for (LastUserResult& uit : lit.second)
        {
          // For each event convert it into a frontier
          for (const ApEvent& it : uit.events)
          {
            std::map<ApEvent, unsigned>::const_iterator finder =
                frontier_map.find(it);
            if (finder == frontier_map.end())
            {
              unsigned index = find_frontier_event(it, frontier_events);
              uit.frontiers.emplace_back(index);
              frontier_map[it] = index;
            }
            else
              uit.frontiers.emplace_back(finder->second);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    unsigned PhysicalTemplate::find_frontier_event(
        ApEvent event, std::vector<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // Check to see if it is an event we know about
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      // If it's not an event we recognize we can just return the start event
      if (finder == event_map.end())
        return 0;
      legion_assert(frontiers.find(finder->second) == frontiers.end());
      // Make a new frontier event
      const unsigned next_event_id = events.size();
      frontiers[finder->second] = next_event_id;
      events.resize(next_event_id + 1);
      return next_event_id;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::elide_fences(
        std::vector<unsigned>& gen, std::vector<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // Reserve some events for merges to be added during fence elision
      unsigned num_merges = 0;
      for (Instruction* inst : instructions) switch (inst->get_kind())
        {
          case ISSUE_COPY:
            {
              unsigned precondition_idx =
                  inst->as_issue_copy()->precondition_idx;
              InstructionKind generator_kind =
                  instructions[precondition_idx]->get_kind();
              num_merges += generator_kind != MERGE_EVENT;
              break;
            }
          case ISSUE_FILL:
            {
              unsigned precondition_idx =
                  inst->as_issue_fill()->precondition_idx;
              InstructionKind generator_kind =
                  instructions[precondition_idx]->get_kind();
              num_merges += generator_kind != MERGE_EVENT;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross* across = inst->as_issue_across();
              if (across->collective_precondition == 0)
              {
                InstructionKind generator_kind =
                    instructions[across->copy_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              else
              {
                InstructionKind generator_kind =
                    instructions[across->collective_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              if (across->src_indirect_precondition != 0)
              {
                InstructionKind generator_kind =
                    instructions[across->src_indirect_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              if (across->dst_indirect_precondition != 0)
              {
                InstructionKind generator_kind =
                    instructions[across->dst_indirect_precondition]->get_kind();
                num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              }
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay* complete = inst->as_complete_replay();
              InstructionKind generator_kind =
                  instructions[complete->complete]->get_kind();
              num_merges += (generator_kind != MERGE_EVENT) ? 1 : 0;
              break;
            }
          default:
            {
              break;
            }
        }

      unsigned merge_starts = events.size();
      events.resize(events.size() + num_merges);

      // We are now going to break the invariant that
      // the generator of events[idx] is instructions[idx].
      // After fence elision, the generator of events[idx] is
      // instructions[gen[idx]].
      gen.resize(events.size(), 0 /*fence instruction*/);
      std::vector<Instruction*> new_instructions;

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        switch (kind)
        {
          case COMPLETE_REPLAY:
            {
              CompleteReplay* replay = inst->as_complete_replay();
              std::map<TraceLocalID, InstUsers>::iterator finder =
                  op_insts.find(replay->owner);
              if (finder == op_insts.end())
                break;
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(
                  replay->complete, users, instructions, new_instructions, gen,
                  merge_starts);
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy* copy = inst->as_issue_copy();
              std::map<unsigned, InstUsers>::iterator finder =
                  copy_insts.find(copy->lhs);
              legion_assert(finder != copy_insts.end());
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(
                  copy->precondition_idx, users, instructions, new_instructions,
                  gen, merge_starts);
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill* fill = inst->as_issue_fill();
              std::map<unsigned, InstUsers>::iterator finder =
                  copy_insts.find(fill->lhs);
              legion_assert(finder != copy_insts.end());
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              rewrite_preconditions(
                  fill->precondition_idx, users, instructions, new_instructions,
                  gen, merge_starts);
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross* across = inst->as_issue_across();
              std::map<unsigned, InstUsers>::iterator finder =
                  copy_insts.find(across->lhs);
              legion_assert(finder != copy_insts.end());
              std::set<unsigned> users;
              find_all_last_users(finder->second, users);
              // This is super subtle: for indirections that are
              // working collectively together on a set of indirect
              // source or destination instances, we actually have
              // a fan-in event construction. The indirect->copy_precondition
              // contains the result of that fan-in tree which is not
              // what we want to update here. We instead want to update
              // the set of preconditions to that collective fan-in for this
              // part of the indirect which feed into the collective event
              // tree construction. The local fan-in event is stored at
              // indirect->collective_precondition so use that instead for this
              if (across->collective_precondition == 0)
                rewrite_preconditions(
                    across->copy_precondition, users, instructions,
                    new_instructions, gen, merge_starts);
              else
                rewrite_preconditions(
                    across->collective_precondition, users, instructions,
                    new_instructions, gen, merge_starts);
              // Also do the rewrites for any indirection preconditions
              if (across->src_indirect_precondition != 0)
              {
                users.clear();
                finder = src_indirect_insts.find(across->lhs);
                legion_assert(finder != src_indirect_insts.end());
                find_all_last_users(finder->second, users);
                rewrite_preconditions(
                    across->src_indirect_precondition, users, instructions,
                    new_instructions, gen, merge_starts);
              }
              if (across->dst_indirect_precondition != 0)
              {
                users.clear();
                finder = dst_indirect_insts.find(across->lhs);
                legion_assert(finder != dst_indirect_insts.end());
                find_all_last_users(finder->second, users);
                rewrite_preconditions(
                    across->dst_indirect_precondition, users, instructions,
                    new_instructions, gen, merge_starts);
              }
              break;
            }
          default:
            {
              break;
            }
        }
        gen[idx] = new_instructions.size();
        new_instructions.emplace_back(inst);
      }
      instructions.swap(new_instructions);
      new_instructions.clear();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::rewrite_preconditions(
        unsigned& precondition, std::set<unsigned>& users,
        const std::vector<Instruction*>& instructions,
        std::vector<Instruction*>& new_instructions, std::vector<unsigned>& gen,
        unsigned& merge_starts)
    //--------------------------------------------------------------------------
    {
      if (users.empty())
        return;
      Instruction* generator_inst = instructions[precondition];
      if (generator_inst->get_kind() == MERGE_EVENT)
      {
        MergeEvent* merge = generator_inst->as_merge_event();
        merge->rhs.insert(users.begin(), users.end());
      }
      else
      {
        unsigned merging_event_idx = merge_starts++;
        gen[merging_event_idx] = new_instructions.size();
        if (precondition != fence_completion_id)
          users.insert(precondition);
        new_instructions.emplace_back(new MergeEvent(
            *this, merging_event_idx, users, generator_inst->owner));
        precondition = merging_event_idx;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::propagate_merges(std::vector<unsigned>& gen)
    //--------------------------------------------------------------------------
    {
      std::vector<bool> used(instructions.size(), false);

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        used[idx] = kind != MERGE_EVENT;
        switch (kind)
        {
          case MERGE_EVENT:
            {
              MergeEvent* merge = inst->as_merge_event();
              std::set<unsigned> new_rhs;
              bool changed = false;
              for (const unsigned& it : merge->rhs)
              {
                Instruction* generator = instructions[gen[it]];
                if (generator->get_kind() == MERGE_EVENT)
                {
                  MergeEvent* to_splice = generator->as_merge_event();
                  new_rhs.insert(to_splice->rhs.begin(), to_splice->rhs.end());
                  changed = true;
                }
                else
                  new_rhs.insert(it);
              }
              if (changed)
                merge->rhs.swap(new_rhs);
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent* trigger = inst->as_trigger_event();
              used[gen[trigger->rhs]] = true;
              break;
            }
          case BARRIER_ARRIVAL:
            {
              BarrierArrival* arrival = inst->as_barrier_arrival();
              used[gen[arrival->rhs]] = true;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy* copy = inst->as_issue_copy();
              used[gen[copy->precondition_idx]] = true;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross* across = inst->as_issue_across();
              used[gen[across->copy_precondition]] = true;
              if (across->collective_precondition != 0)
                used[gen[across->collective_precondition]] = true;
              if (across->src_indirect_precondition != 0)
                used[gen[across->src_indirect_precondition]] = true;
              if (across->dst_indirect_precondition != 0)
                used[gen[across->dst_indirect_precondition]] = true;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill* fill = inst->as_issue_fill();
              used[gen[fill->precondition_idx]] = true;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay* complete = inst->as_complete_replay();
              used[gen[complete->complete]] = true;
              break;
            }
          case REPLAY_MAPPING:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
          case BARRIER_ADVANCE:
            {
              break;
            }
          default:
            {
              // unreachable
              std::abort();
            }
        }
      }
      record_used_frontiers(used, gen);

      std::vector<unsigned> inv_gen(instructions.size(), -1U);
      for (unsigned idx = 0; idx < gen.size(); ++idx)
      {
        unsigned g = gen[idx];
        legion_assert(inv_gen[g] == -1U || g == fence_completion_id);
        if (g != -1U && g < instructions.size() && inv_gen[g] == -1U)
          inv_gen[g] = idx;
      }
      std::vector<Instruction*> to_delete;
      std::vector<unsigned> new_gen(gen.size(), -1U);
      initialize_generators(new_gen);
      std::vector<Instruction*> new_instructions;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (used[idx])
        {
          Instruction* inst = instructions[idx];
          // Fence elision-style operations can only be performed if the
          // trace is idempotent.
          if (trace->perform_fence_elision && is_idempotent())
          {
            if (inst->get_kind() == MERGE_EVENT)
            {
              MergeEvent* merge = inst->as_merge_event();
              if (merge->rhs.size() > 1)
                merge->rhs.erase(fence_completion_id);
            }
          }
          unsigned e = inv_gen[idx];
          legion_assert(e == -1U || (e < new_gen.size() && new_gen[e] == -1U));
          if (e != -1U)
            new_gen[e] = new_instructions.size();
          new_instructions.emplace_back(inst);
        }
        else
          to_delete.emplace_back(instructions[idx]);
      instructions.swap(new_instructions);
      gen.swap(new_gen);
      for (unsigned idx = 0; idx < to_delete.size(); ++idx)
        delete to_delete[idx];
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_used_frontiers(
        std::vector<bool>& used, const std::vector<unsigned>& gen) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
        used[gen[it.first]] = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::sync_compute_frontiers(
        CompleteOp* op, const std::vector<RtEvent>& frontier_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(frontier_events.empty());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_generators(std::vector<unsigned>& new_gen)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
        new_gen[it.second] = 0;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_eliminate_dead_code_frontiers(
        const std::vector<unsigned>& gen, std::vector<bool>& used)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
      {
        unsigned g = gen[it.first];
        if (g != -1U && g < instructions.size())
          used[g] = true;
      }
      // Don't eliminate the last fence instruction
      if (last_fence != nullptr)
        used[gen[last_fence->complete]] = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::prepare_parallel_replay(
        const std::vector<unsigned>& gen)
    //--------------------------------------------------------------------------
    {
      const size_t replay_parallelism = runtime->max_replay_parallelism;
      slices.resize(replay_parallelism);
      std::map<TraceLocalID, unsigned> slice_indices_by_owner;
      std::vector<unsigned> slice_indices_by_inst;
      slice_indices_by_inst.resize(instructions.size());

      for (unsigned idx = 1; idx < instructions.size(); ++idx)
        slice_indices_by_inst[idx] = -1U;
      bool round_robin_for_tasks = false;

      std::set<Processor> distinct_targets;
      for (CachedMappings::iterator it = cached_mappings.begin();
           it != cached_mappings.end(); ++it)
        distinct_targets.insert(it->second.target_procs[0]);
      round_robin_for_tasks = distinct_targets.size() < replay_parallelism;

      unsigned next_slice_id = 0;
      for (const std::pair<const TraceLocalID, std::pair<unsigned, unsigned> >&
               it : memo_entries)
      {
        unsigned slice_index = -1U;
        if (!round_robin_for_tasks && (it.second.second == TASK_OP_KIND) &&
            (it.first.index_point.get_dim() > 0))
        {
          CachedMappings::iterator finder = cached_mappings.find(it.first);
          legion_assert(finder != cached_mappings.end());
          legion_assert(finder->second.target_procs.size() > 0);
          slice_index = finder->second.target_procs[0].id % replay_parallelism;
        }
        else
        {
          legion_assert(
              slice_indices_by_owner.find(it.first) ==
              slice_indices_by_owner.end());
          slice_index = next_slice_id;
          next_slice_id = (next_slice_id + 1) % replay_parallelism;
        }

        legion_assert(slice_index != -1U);
        slice_indices_by_owner[it.first] = slice_index;
      }
      // Make sure that event creations and triggers are in the same slice
      std::map<unsigned /*user event*/, unsigned /*slice*/> user_event_slices;
      // Keep track of these so that we don't end up leaking them
      std::vector<Instruction*> crossing_instructions;
      std::map<unsigned, std::pair<unsigned, unsigned> > crossing_counts;
      for (unsigned idx = 1; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        const TraceLocalID& owner = inst->owner;
        std::map<TraceLocalID, unsigned>::iterator finder =
            slice_indices_by_owner.find(owner);
        unsigned slice_index = -1U;
        const InstructionKind kind = inst->get_kind();
        if (finder != slice_indices_by_owner.end())
          slice_index = finder->second;
        else if (kind == TRIGGER_EVENT)
        {
          // Find the slice where the event creation was assigned
          // and make sure that we end up on the same slice
          TriggerEvent* trigger = inst->as_trigger_event();
          std::map<unsigned, unsigned>::iterator finder =
              user_event_slices.find(trigger->lhs);
          legion_assert(finder != user_event_slices.end());
          slice_index = finder->second;
          user_event_slices.erase(finder);
        }
        else
        {
          slice_index = next_slice_id;
          next_slice_id = (next_slice_id + 1) % replay_parallelism;
          if (kind == CREATE_AP_USER_EVENT)
          {
            // Save which slice this is on so the later trigger will
            // get recorded on the same slice
            CreateApUserEvent* create = inst->as_create_ap_user_event();
            legion_assert(
                user_event_slices.find(create->lhs) == user_event_slices.end());
            user_event_slices[create->lhs] = slice_index;
          }
        }
        slices[slice_index].emplace_back(inst);
        slice_indices_by_inst[idx] = slice_index;

        if (inst->get_kind() == MERGE_EVENT)
        {
          MergeEvent* merge = inst->as_merge_event();
          unsigned crossing_found = false;
          std::set<unsigned> new_rhs;
          for (const unsigned& rh : merge->rhs)
          {
            // Don't need to worry about crossing events for the fence
            // initialization as we know it's always set before any
            // slices executes (rh == 0)
            if ((rh == 0) || (gen[rh] == 0))
              new_rhs.insert(rh);
            else
            {
              legion_assert(gen[rh] != -1U);
              unsigned generator_slice = slice_indices_by_inst[gen[rh]];
              legion_assert(generator_slice != -1U);
              if (generator_slice != slice_index)
              {
                crossing_found = true;
                std::map<unsigned, std::pair<unsigned, unsigned> >::iterator
                    finder = crossing_counts.find(rh);
                if (finder != crossing_counts.end())
                {
                  new_rhs.insert(finder->second.first);
                  finder->second.second += 1;
                }
                else
                {
                  unsigned new_crossing_event = events.size();
                  events.resize(events.size() + 1);
                  crossing_counts[rh] = std::pair<unsigned, unsigned>(
                      new_crossing_event, 1 /*count*/);
                  new_rhs.insert(new_crossing_event);
                  TriggerEvent* crossing = new TriggerEvent(
                      *this, new_crossing_event, rh,
                      instructions[gen[rh]]->owner);
                  slices[generator_slice].emplace_back(crossing);
                  crossing_instructions.emplace_back(crossing);
                }
              }
              else
                new_rhs.insert(rh);
            }
          }

          if (crossing_found)
            merge->rhs.swap(new_rhs);
        }
        else
        {
          switch (inst->get_kind())
          {
            case TRIGGER_EVENT:
              {
                parallelize_replay_event(
                    inst->as_trigger_event()->rhs, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                break;
              }
            case BARRIER_ARRIVAL:
              {
                parallelize_replay_event(
                    inst->as_barrier_arrival()->rhs, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                break;
              }
            case ISSUE_COPY:
              {
                parallelize_replay_event(
                    inst->as_issue_copy()->precondition_idx, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                break;
              }
            case ISSUE_FILL:
              {
                parallelize_replay_event(
                    inst->as_issue_fill()->precondition_idx, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                break;
              }
            case ISSUE_ACROSS:
              {
                IssueAcross* across = inst->as_issue_across();
                parallelize_replay_event(
                    across->copy_precondition, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                if (across->collective_precondition != 0)
                  parallelize_replay_event(
                      across->collective_precondition, slice_index, gen,
                      slice_indices_by_inst, crossing_counts,
                      crossing_instructions);
                if (across->src_indirect_precondition != 0)
                  parallelize_replay_event(
                      across->src_indirect_precondition, slice_index, gen,
                      slice_indices_by_inst, crossing_counts,
                      crossing_instructions);
                if (across->dst_indirect_precondition != 0)
                  parallelize_replay_event(
                      across->dst_indirect_precondition, slice_index, gen,
                      slice_indices_by_inst, crossing_counts,
                      crossing_instructions);
                break;
              }
            case COMPLETE_REPLAY:
              {
                parallelize_replay_event(
                    inst->as_complete_replay()->complete, slice_index, gen,
                    slice_indices_by_inst, crossing_counts,
                    crossing_instructions);
                break;
              }
            default:
              {
                break;
              }
          }
        }
      }
      legion_assert(user_event_slices.empty());
      // Update the crossing events and their counts
      if (!crossing_counts.empty())
      {
        for (const std::pair<const unsigned, std::pair<unsigned, unsigned> >&
                 it : crossing_counts)
          crossing_events.insert(it.second);
      }
      // Append any new crossing instructions to the list of instructions
      // so that they will still be deleted when the template is
      if (!crossing_instructions.empty())
        instructions.insert(
            instructions.end(), crossing_instructions.begin(),
            crossing_instructions.end());
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::parallelize_replay_event(
        unsigned& event_to_check, unsigned slice_index,
        const std::vector<unsigned>& gen,
        const std::vector<unsigned>& slice_indices_by_inst,
        std::map<unsigned, std::pair<unsigned, unsigned> >& crossing_counts,
        std::vector<Instruction*>& crossing_instructions)
    //--------------------------------------------------------------------------
    {
      // If this is the zero event, then don't even bother, we know the
      // fence event is set before all the slices replay anyway
      if (event_to_check == 0)
        return;
      unsigned g = gen[event_to_check];
      legion_assert(g != -1U && g < instructions.size());
      unsigned generator_slice = slice_indices_by_inst[g];
      legion_assert(generator_slice != -1U);
      if (generator_slice != slice_index)
      {
        std::map<unsigned, std::pair<unsigned, unsigned> >::iterator finder =
            crossing_counts.find(event_to_check);
        if (finder != crossing_counts.end())
        {
          event_to_check = finder->second.first;
          finder->second.second += 1;
        }
        else
        {
          unsigned new_crossing_event = events.size();
          events.resize(events.size() + 1);
          crossing_counts[event_to_check] =
              std::pair<unsigned, unsigned>(new_crossing_event, 1 /*count*/);
          TriggerEvent* crossing = new TriggerEvent(
              *this, new_crossing_event, event_to_check,
              instructions[g]->owner);
          event_to_check = new_crossing_event;
          slices[generator_slice].emplace_back(crossing);
          crossing_instructions.emplace_back(crossing);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_transitive_reduction_frontiers(
        std::vector<unsigned>& topo_order,
        std::vector<unsigned>& inv_topo_order)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
      {
        inv_topo_order[it.second] = topo_order.size();
        topo_order.emplace_back(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::transitive_reduction(
        TransitiveReductionState* state, bool deferred)
    //--------------------------------------------------------------------------
    {
      // Transitive reduction inspired by Klaus Simon,
      // "An improved algorithm for transitive closure on acyclic digraphs"

      // The transitive reduction can be a really long computation and we
      // don't want it monopolizing an entire processor while it's running
      // so we time-slice as background tasks until it is done. We pick the
      // somewhat arbitrary timeslice of 2ms since that's around the right
      // order of magnitude for other meta-tasks on most machines while still
      // being large enough to warm-up caches and make forward progress
      constexpr long long TIMEOUT = 2000;  // in microseconds
      unsigned long long running_time = 0;
      unsigned long long previous_time =
          Realm::Clock::current_time_in_microseconds();
      std::vector<unsigned>& topo_order = state->topo_order;
      std::vector<unsigned>& inv_topo_order = state->inv_topo_order;
      std::vector<std::vector<unsigned> >& incoming = state->incoming;
      std::vector<std::vector<unsigned> >& outgoing = state->outgoing;
      if (state->stage == 0)
      {
        topo_order.reserve(instructions.size());
        inv_topo_order.resize(events.size(), -1U);
        incoming.resize(events.size());
        outgoing.resize(events.size());

        initialize_transitive_reduction_frontiers(topo_order, inv_topo_order);
        state->stage++;
      }

      // First, build a DAG and find nodes with no incoming edges
      if (state->stage == 1)
      {
        std::map<TraceLocalID, ReplayMapping*>& replay_insts =
            state->replay_insts;
        for (unsigned idx = state->iteration; idx < instructions.size(); ++idx)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time =
                Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state);
              runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
              return;
            }
            else
              previous_time = current_time;
          }
          Instruction* inst = instructions[idx];
          switch (inst->get_kind())
          {
            case REPLAY_MAPPING:
              {
                ReplayMapping* replay = inst->as_replay_mapping();
                replay_insts[inst->owner] = replay;
                break;
              }
            case CREATE_AP_USER_EVENT:
              {
                break;
              }
            case TRIGGER_EVENT:
              {
                TriggerEvent* trigger = inst->as_trigger_event();
                incoming[trigger->lhs].emplace_back(trigger->rhs);
                outgoing[trigger->rhs].emplace_back(trigger->lhs);
                break;
              }
            case BARRIER_ARRIVAL:
              {
                BarrierArrival* arrival = inst->as_barrier_arrival();
                incoming[arrival->lhs].emplace_back(arrival->rhs);
                outgoing[arrival->rhs].emplace_back(arrival->lhs);
                break;
              }
            case MERGE_EVENT:
              {
                MergeEvent* merge = inst->as_merge_event();
                for (const unsigned& it : merge->rhs)
                {
                  incoming[merge->lhs].emplace_back(it);
                  outgoing[it].emplace_back(merge->lhs);
                }
                break;
              }
            case ISSUE_COPY:
              {
                IssueCopy* copy = inst->as_issue_copy();
                incoming[copy->lhs].emplace_back(copy->precondition_idx);
                outgoing[copy->precondition_idx].emplace_back(copy->lhs);
                break;
              }
            case ISSUE_FILL:
              {
                IssueFill* fill = inst->as_issue_fill();
                incoming[fill->lhs].emplace_back(fill->precondition_idx);
                outgoing[fill->precondition_idx].emplace_back(fill->lhs);
                break;
              }
            case ISSUE_ACROSS:
              {
                IssueAcross* across = inst->as_issue_across();
                incoming[across->lhs].emplace_back(across->copy_precondition);
                outgoing[across->copy_precondition].emplace_back(across->lhs);
                if (across->collective_precondition != 0)
                {
                  incoming[across->lhs].emplace_back(
                      across->collective_precondition);
                  outgoing[across->collective_precondition].emplace_back(
                      across->lhs);
                }
                if (across->src_indirect_precondition != 0)
                {
                  incoming[across->lhs].emplace_back(
                      across->src_indirect_precondition);
                  outgoing[across->src_indirect_precondition].emplace_back(
                      across->lhs);
                }
                if (across->dst_indirect_precondition != 0)
                {
                  incoming[across->lhs].emplace_back(
                      across->dst_indirect_precondition);
                  outgoing[across->dst_indirect_precondition].emplace_back(
                      across->lhs);
                }
                break;
              }
            case SET_OP_SYNC_EVENT:
              {
                SetOpSyncEvent* sync = inst->as_set_op_sync_event();
                inv_topo_order[sync->lhs] = topo_order.size();
                topo_order.emplace_back(sync->lhs);
                break;
              }
            case BARRIER_ADVANCE:
              {
                BarrierAdvance* advance = inst->as_barrier_advance();
                inv_topo_order[advance->lhs] = topo_order.size();
                topo_order.emplace_back(advance->lhs);
                break;
              }
            case ASSIGN_FENCE_COMPLETION:
              {
                inv_topo_order[fence_completion_id] = topo_order.size();
                topo_order.emplace_back(fence_completion_id);
                break;
              }
            case COMPLETE_REPLAY:
              {
                CompleteReplay* replay = inst->as_complete_replay();
                // Check to see if we can find a replay instruction to match
                std::map<TraceLocalID, ReplayMapping*>::iterator replay_finder =
                    replay_insts.find(replay->owner);
                if (replay_finder != replay_insts.end())
                {
                  incoming[replay_finder->second->lhs].emplace_back(
                      replay->complete);
                  outgoing[replay->complete].emplace_back(
                      replay_finder->second->lhs);
                  replay_insts.erase(replay_finder);
                }
                break;
              }
            default:
              std::abort();
          }
        }
        // should have seen a complete replay instruction for every replay
        // mapping
        legion_assert(replay_insts.empty());
        state->stage++;
        state->iteration = 0;
        replay_insts.clear();
      }

      // Second, do a toposort on nodes via BFS
      if (state->stage == 2)
      {
        std::vector<unsigned>& remaining_edges = state->remaining_edges;
        if (remaining_edges.empty())
        {
          remaining_edges.resize(incoming.size());
          for (unsigned idx = 0; idx < incoming.size(); ++idx)
            remaining_edges[idx] = incoming[idx].size();
        }

        unsigned idx = state->iteration;
        while (idx < topo_order.size())
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time =
                Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state);
              runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
              return;
            }
            else
              previous_time = current_time;
          }
          unsigned node = topo_order[idx];
          legion_assert(remaining_edges[node] == 0);
          const std::vector<unsigned>& out = outgoing[node];
          for (unsigned oidx = 0; oidx < out.size(); ++oidx)
          {
            unsigned next = out[oidx];
            if (--remaining_edges[next] == 0)
            {
              inv_topo_order[next] = topo_order.size();
              topo_order.emplace_back(next);
            }
          }
          ++idx;
        }
        for (unsigned idx = 0; idx < incoming.size(); idx++)
        {
          legion_assert(remaining_edges[idx] == 0);
        }
        state->stage++;
        state->iteration = 0;
        remaining_edges.clear();
      }

      // Third, construct a chain decomposition
      if (state->stage == 3)
      {
        std::vector<unsigned>& chain_indices = state->chain_indices;
        if (chain_indices.empty())
        {
          chain_indices.resize(topo_order.size(), -1U);
          state->pos = chain_indices.size() - 1;
        }

        int pos = state->pos;
        unsigned num_chains = state->num_chains;
        while (true)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time =
                Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->pos = pos;
              state->num_chains = num_chains;
              TransitiveReductionArgs args(this, state);
              runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
              return;
            }
            else
              previous_time = current_time;
          }
          while (pos >= 0 && chain_indices[pos] != -1U) --pos;
          if (pos < 0)
            break;
          unsigned curr = topo_order[pos];
          while (incoming[curr].size() > 0)
          {
            chain_indices[inv_topo_order[curr]] = num_chains;
            const std::vector<unsigned>& in = incoming[curr];
            bool found = false;
            for (unsigned iidx = 0; iidx < in.size(); ++iidx)
            {
              unsigned next = in[iidx];
              if (chain_indices[inv_topo_order[next]] == -1U)
              {
                found = true;
                curr = next;
                chain_indices[inv_topo_order[curr]] = num_chains;
                break;
              }
            }
            if (!found)
              break;
          }
          chain_indices[inv_topo_order[curr]] = num_chains;
          ++num_chains;
        }
        state->stage++;
        state->num_chains = num_chains;
      }

      // Fourth, find the frontiers of chains that are connected to each node
      if (state->stage == 4)
      {
        const unsigned num_chains = state->num_chains;
        const std::vector<unsigned>& chain_indices = state->chain_indices;
        std::vector<std::vector<int> >& all_chain_frontiers =
            state->all_chain_frontiers;
        if (all_chain_frontiers.empty())
          all_chain_frontiers.resize(topo_order.size());
        std::vector<std::vector<unsigned> >& incoming_reduced =
            state->incoming_reduced;
        if (incoming_reduced.empty())
          incoming_reduced.resize(topo_order.size());
        for (unsigned idx = state->iteration; idx < topo_order.size(); idx++)
        {
          // Check for timeout
          if (deferred)
          {
            unsigned long long current_time =
                Realm::Clock::current_time_in_microseconds();
            running_time += (current_time - previous_time);
            if (TIMEOUT <= running_time)
            {
              // Hit the timeout so launch a continuation
              state->iteration = idx;
              TransitiveReductionArgs args(this, state);
              runtime->issue_runtime_meta_task(args, LG_LOW_PRIORITY);
              return;
            }
            else
              previous_time = current_time;
          }
          std::vector<int> chain_frontiers(num_chains, -1);
          const std::vector<unsigned>& in = incoming[topo_order[idx]];
          std::vector<unsigned>& in_reduced = incoming_reduced[idx];
          for (unsigned iidx = 0; iidx < in.size(); ++iidx)
          {
            int rank = inv_topo_order[in[iidx]];
            legion_assert((unsigned)rank < idx);
            const std::vector<int>& pred_chain_frontiers =
                all_chain_frontiers[rank];
            for (unsigned k = 0; k < num_chains; ++k)
              chain_frontiers[k] =
                  std::max(chain_frontiers[k], pred_chain_frontiers[k]);
          }
          for (unsigned iidx = 0; iidx < in.size(); ++iidx)
          {
            int rank = inv_topo_order[in[iidx]];
            unsigned chain_idx = chain_indices[rank];
            if (chain_frontiers[chain_idx] < rank)
            {
              in_reduced.emplace_back(in[iidx]);
              chain_frontiers[chain_idx] = rank;
            }
          }
          legion_assert(in.size() == 0 || in_reduced.size() > 0);
          all_chain_frontiers[idx].swap(chain_frontiers);
        }
        state->stage++;
        state->iteration = 0;
        all_chain_frontiers.clear();
      }

      // Lastly, suppress transitive dependences using chains
      if (deferred)
      {
        const RtUserEvent to_trigger = state->done;
        finished_transitive_reduction.store(state);
        Runtime::trigger_event(to_trigger);
      }
      else
        finalize_transitive_reduction(
            state->inv_topo_order, state->incoming_reduced);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finalize_transitive_reduction(
        const std::vector<unsigned>& inv_topo_order,
        const std::vector<std::vector<unsigned> >& incoming_reduced)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
        if (instructions[idx]->get_kind() == MERGE_EVENT)
        {
          MergeEvent* merge = instructions[idx]->as_merge_event();
          unsigned order = inv_topo_order[merge->lhs];
          legion_assert(order != -1U);
          const std::vector<unsigned>& in_reduced = incoming_reduced[order];
          if (in_reduced.size() == merge->rhs.size())
          {
            for (unsigned iidx = 0; iidx < in_reduced.size(); ++iidx)
            {
              legion_assert(
                  merge->rhs.find(in_reduced[iidx]) != merge->rhs.end());
            }
            continue;
          }
          std::set<unsigned> new_rhs;
          for (unsigned iidx = 0; iidx < in_reduced.size(); ++iidx)
          {
            legion_assert(
                merge->rhs.find(in_reduced[iidx]) != merge->rhs.end());
            new_rhs.insert(in_reduced[iidx]);
          }
          // Remove any references to crossing events which are no longer needed
          if (!crossing_events.empty())
          {
            for (const unsigned& it : merge->rhs)
            {
              std::map<unsigned, unsigned>::iterator finder =
                  crossing_events.find(it);
              if ((finder != crossing_events.end()) &&
                  (new_rhs.find(it) == new_rhs.end()))
              {
                legion_assert(finder->second > 0);
                finder->second--;
              }
            }
          }
          merge->rhs.swap(new_rhs);
        }
      // Remove any crossing instructions from the slices that are no
      // longer needed because the transitive reduction eliminated the
      // need for the edge
      for (std::map<unsigned, unsigned>::iterator it = crossing_events.begin();
           it != crossing_events.end();
           /*nothing*/)
      {
        if (it->second == 0)
        {
          // No more references to this crossing instruction so remove it
          bool found = false;
          for (std::vector<Instruction*>& sit : slices)
          {
            for (std::vector<Instruction*>::iterator iit = sit.begin();
                 iit != sit.end(); iit++)
            {
              TriggerEvent* trigger = (*iit)->as_trigger_event();
              if (trigger == nullptr)
                continue;
              if (trigger->lhs == it->first)
              {
                sit.erase(iit);
                found = true;
                break;
              }
            }
            if (found)
              break;
          }
          std::map<unsigned, unsigned>::iterator to_delete = it++;
          crossing_events.erase(to_delete);
        }
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::check_finalize_transitive_reduction(void)
    //--------------------------------------------------------------------------
    {
      TransitiveReductionState* state =
          finished_transitive_reduction.exchange(nullptr);
      if (state != nullptr)
      {
        finalize_transitive_reduction(
            state->inv_topo_order, state->incoming_reduced);
        delete state;
        // We also need to rerun the propagate copies analysis to
        // remove any mergers which contain only a single input
        propagate_copies(nullptr /*don't need the gen out*/);
        if (runtime->dump_physical_traces)
          dump_template();
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::propagate_copies(std::vector<unsigned>* gen)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned, unsigned> substitutions;
      std::vector<Instruction*> new_instructions;
      new_instructions.reserve(instructions.size());
      std::set<Instruction*> to_prune;
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        if (instructions[idx]->get_kind() == MERGE_EVENT)
        {
          MergeEvent* merge = instructions[idx]->as_merge_event();
          legion_assert(merge->rhs.size() > 0);
          if (merge->rhs.size() == 1)
          {
            substitutions[merge->lhs] = *merge->rhs.begin();
            legion_assert(merge->lhs != substitutions[merge->lhs]);
            if (gen == nullptr)
              to_prune.insert(inst);
            else
              delete inst;
          }
          else
            new_instructions.emplace_back(inst);
        }
        else
          new_instructions.emplace_back(inst);
      }

      if (instructions.size() == new_instructions.size())
        return;

      // Rewrite the frontiers first
      rewrite_frontiers(substitutions);

      // Then rewrite the instructions
      instructions.swap(new_instructions);

      std::vector<unsigned> new_gen((gen == nullptr) ? 0 : gen->size(), -1U);
      if (gen != nullptr)
        initialize_generators(new_gen);

      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        int lhs = -1;
        switch (inst->get_kind())
        {
          case REPLAY_MAPPING:
            {
              ReplayMapping* replay = inst->as_replay_mapping();
              lhs = replay->lhs;
              break;
            }
          case CREATE_AP_USER_EVENT:
            {
              CreateApUserEvent* create = inst->as_create_ap_user_event();
              lhs = create->lhs;
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent* trigger = inst->as_trigger_event();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(trigger->rhs);
              if (finder != substitutions.end())
                trigger->rhs = finder->second;
              break;
            }
          case BARRIER_ARRIVAL:
            {
              BarrierArrival* arrival = inst->as_barrier_arrival();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(arrival->rhs);
              if (finder != substitutions.end())
                arrival->rhs = finder->second;
              lhs = arrival->lhs;
              break;
            }
          case MERGE_EVENT:
            {
              MergeEvent* merge = inst->as_merge_event();
              std::set<unsigned> new_rhs;
              for (const unsigned& it : merge->rhs)
              {
                std::map<unsigned, unsigned>::const_iterator finder =
                    substitutions.find(it);
                if (finder != substitutions.end())
                  new_rhs.insert(finder->second);
                else
                  new_rhs.insert(it);
              }
              merge->rhs.swap(new_rhs);
              lhs = merge->lhs;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy* copy = inst->as_issue_copy();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(copy->precondition_idx);
              if (finder != substitutions.end())
                copy->precondition_idx = finder->second;
              lhs = copy->lhs;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill* fill = inst->as_issue_fill();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(fill->precondition_idx);
              if (finder != substitutions.end())
                fill->precondition_idx = finder->second;
              lhs = fill->lhs;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross* across = inst->as_issue_across();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(across->copy_precondition);
              if (finder != substitutions.end())
                across->copy_precondition = finder->second;
              if (across->collective_precondition != 0)
              {
                finder = substitutions.find(across->collective_precondition);
                if (finder != substitutions.end())
                  across->collective_precondition = finder->second;
              }
              if (across->src_indirect_precondition != 0)
              {
                finder = substitutions.find(across->src_indirect_precondition);
                if (finder != substitutions.end())
                  across->src_indirect_precondition = finder->second;
              }
              if (across->dst_indirect_precondition != 0)
              {
                finder = substitutions.find(across->dst_indirect_precondition);
                if (finder != substitutions.end())
                  across->dst_indirect_precondition = finder->second;
              }
              lhs = across->lhs;
              break;
            }
          case SET_OP_SYNC_EVENT:
            {
              SetOpSyncEvent* sync = inst->as_set_op_sync_event();
              lhs = sync->lhs;
              break;
            }
          case BARRIER_ADVANCE:
            {
              BarrierAdvance* advance = inst->as_barrier_advance();
              lhs = advance->lhs;
              break;
            }
          case ASSIGN_FENCE_COMPLETION:
            {
              lhs = fence_completion_id;
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay* replay = inst->as_complete_replay();
              std::map<unsigned, unsigned>::const_iterator finder =
                  substitutions.find(replay->complete);
              if (finder != substitutions.end())
                replay->complete = finder->second;
              break;
            }
          default:
            {
              break;
            }
        }
        if ((lhs != -1) && (gen != nullptr))
          new_gen[lhs] = idx;
      }
      if (gen != nullptr)
        gen->swap(new_gen);
      if (!to_prune.empty())
      {
        legion_assert(!slices.empty());
        // Remove these instructions from any slices and then delete them
        for (unsigned idx = 0; idx < slices.size(); idx++)
        {
          std::vector<Instruction*>& slice = slices[idx];
          for (std::vector<Instruction*>::iterator it = slice.begin();
               it != slice.end();
               /*nothing*/)
          {
            std::set<Instruction*>::iterator finder = to_prune.find(*it);
            if (finder != to_prune.end())
            {
              it = slice.erase(it);
              delete *finder;
              to_prune.erase(finder);
              if (to_prune.empty())
                break;
            }
            else
              it++;
          }
          if (to_prune.empty())
            break;
        }
        legion_assert(to_prune.empty());
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::rewrite_frontiers(
        std::map<unsigned, unsigned>& substitutions)
    //--------------------------------------------------------------------------
    {
      std::vector<std::pair<unsigned, unsigned> > to_add;
      for (std::map<unsigned, unsigned>::iterator it = frontiers.begin();
           it != frontiers.end();
           /*nothing*/)
      {
        std::map<unsigned, unsigned>::const_iterator finder =
            substitutions.find(it->first);
        if (finder != substitutions.end())
        {
          to_add.emplace_back(std::make_pair(finder->second, it->second));
          std::map<unsigned, unsigned>::iterator to_delete = it++;
          frontiers.erase(to_delete);
        }
        else
          it++;
      }
      for (const std::pair<unsigned, unsigned>& it : to_add)
      {
        std::map<unsigned, unsigned>::const_iterator finder =
            frontiers.find(it.first);
        if (finder != frontiers.end())
        {
          // Handle the case where we recorded two different frontiers
          // but they are now being merged together from the same source
          // and we can therefore substitute the first one for the second
          legion_assert(substitutions.find(it.second) == substitutions.end());
          substitutions[it.second] = finder->second;
        }
        else
          frontiers.insert(it);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::eliminate_dead_code(std::vector<unsigned>& gen)
    //--------------------------------------------------------------------------
    {
      std::vector<bool> used(instructions.size(), false);
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        Instruction* inst = instructions[idx];
        InstructionKind kind = inst->get_kind();
        // We only eliminate two kinds of instructions currently:
        // SetOpSyncEvent
        used[idx] = (kind != SET_OP_SYNC_EVENT);
        switch (kind)
        {
          case MERGE_EVENT:
            {
              MergeEvent* merge = inst->as_merge_event();
              for (const unsigned& it : merge->rhs)
              {
                legion_assert(gen[it] != -1U);
                used[gen[it]] = true;
              }
              break;
            }
          case TRIGGER_EVENT:
            {
              TriggerEvent* trigger = inst->as_trigger_event();
              legion_assert(gen[trigger->rhs] != -1U);
              used[gen[trigger->rhs]] = true;
              break;
            }
          case ISSUE_COPY:
            {
              IssueCopy* copy = inst->as_issue_copy();
              legion_assert(gen[copy->precondition_idx] != -1U);
              used[gen[copy->precondition_idx]] = true;
              break;
            }
          case ISSUE_FILL:
            {
              IssueFill* fill = inst->as_issue_fill();
              legion_assert(gen[fill->precondition_idx] != -1U);
              used[gen[fill->precondition_idx]] = true;
              break;
            }
          case ISSUE_ACROSS:
            {
              IssueAcross* across = inst->as_issue_across();
              legion_assert(gen[across->copy_precondition] != -1U);
              used[gen[across->copy_precondition]] = true;
              if (across->collective_precondition != 0)
              {
                legion_assert(gen[across->collective_precondition] != -1U);
                used[gen[across->collective_precondition]] = true;
              }
              if (across->src_indirect_precondition != 0)
              {
                legion_assert(gen[across->src_indirect_precondition] != -1U);
                used[gen[across->src_indirect_precondition]] = true;
              }
              if (across->dst_indirect_precondition != 0)
              {
                legion_assert(gen[across->dst_indirect_precondition] != -1U);
                used[gen[across->dst_indirect_precondition]] = true;
              }
              break;
            }
          case COMPLETE_REPLAY:
            {
              CompleteReplay* complete = inst->as_complete_replay();
              legion_assert(gen[complete->complete] != -1U);
              used[gen[complete->complete]] = true;
              break;
            }
          case BARRIER_ARRIVAL:
            {
              BarrierArrival* arrival = inst->as_barrier_arrival();
              legion_assert(gen[arrival->rhs] != -1U);
              used[gen[arrival->rhs]] = true;
              break;
            }
          case REPLAY_MAPPING:
          case CREATE_AP_USER_EVENT:
          case SET_OP_SYNC_EVENT:
          case ASSIGN_FENCE_COMPLETION:
          case BARRIER_ADVANCE:
            {
              break;
            }
          default:
            // unreachable
            std::abort();
        }
      }
      initialize_eliminate_dead_code_frontiers(gen, used);

      std::vector<unsigned> inv_gen(instructions.size(), -1U);
      for (unsigned idx = 0; idx < gen.size(); ++idx)
      {
        unsigned g = gen[idx];
        if (g != -1U && g < instructions.size() && inv_gen[g] == -1U)
          inv_gen[g] = idx;
      }

      std::vector<Instruction*> new_instructions;
      std::vector<Instruction*> to_delete;
      std::vector<unsigned> new_gen(gen.size(), -1U);
      initialize_generators(new_gen);
      for (unsigned idx = 0; idx < instructions.size(); ++idx)
      {
        if (used[idx])
        {
          unsigned e = inv_gen[idx];
          legion_assert(e == -1U || (e < new_gen.size() && new_gen[e] == -1U));
          if (e != -1U)
            new_gen[e] = new_instructions.size();
          new_instructions.emplace_back(instructions[idx]);
        }
        else
          to_delete.emplace_back(instructions[idx]);
      }

      instructions.swap(new_instructions);
      gen.swap(new_gen);
      for (unsigned idx = 0; idx < to_delete.size(); ++idx)
        delete to_delete[idx];
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::push_complete_replays(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < slices.size(); ++idx)
      {
        std::vector<Instruction*>& instructions = slices[idx];
        std::vector<Instruction*> new_instructions;
        new_instructions.reserve(instructions.size());
        std::vector<Instruction*> complete_replays;
        for (unsigned iidx = 0; iidx < instructions.size(); ++iidx)
        {
          Instruction* inst = instructions[iidx];
          if (inst->get_kind() == COMPLETE_REPLAY)
            complete_replays.emplace_back(inst);
          else
            new_instructions.emplace_back(inst);
        }
        new_instructions.insert(
            new_instructions.end(), complete_replays.begin(),
            complete_replays.end());
        instructions.swap(new_instructions);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_template(void) const
    //--------------------------------------------------------------------------
    {
      InnerContext* ctx = trace->logical_trace->context;
      log_tracing.info() << "#### Replayable: " << replayable
                         << ", Idempotent: " << idempotency << " " << this
                         << " Trace " << trace->logical_trace->tid << " for "
                         << ctx->get_task_name() << " (UID "
                         << ctx->get_unique_id() << ") ####";
      if (idempotency == NOT_IDEMPOTENT_SUBSUMPTION)
      {
        log_tracing.info() << "Non-subsumed condition: "
                           << failure.to_string(trace->logical_trace->context);
        if ((failure.view != nullptr) &&
            failure.view->remove_base_resource_ref(TRACE_REF))
          delete failure.view;
        failure.view = nullptr;
        if ((failure.expr != nullptr) &&
            failure.expr->remove_base_expression_reference(TRACE_REF))
          delete failure.expr;
        failure.expr = nullptr;
      }
      else if (idempotency == NOT_IDEMPOTENT_ANTIDEPENDENT)
      {
        log_tracing.info() << "Anti-dependent condition: "
                           << failure.to_string(trace->logical_trace->context);
        if ((failure.view != nullptr) &&
            failure.view->remove_base_resource_ref(TRACE_REF))
          delete failure.view;
        failure.view = nullptr;
        if ((failure.expr != nullptr) &&
            failure.expr->remove_base_expression_reference(TRACE_REF))
          delete failure.expr;
        failure.expr = nullptr;
      }
      const size_t replay_parallelism = trace->get_replay_targets().size();
      for (unsigned sidx = 0; sidx < replay_parallelism; ++sidx)
      {
        log_tracing.info() << "[Slice " << sidx << "]";
        dump_instructions(slices[sidx]);
      }
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
        log_tracing.info() << "  events[" << it.second << "] = events["
                           << it.first << "]";
      dump_sharded_template();

      log_tracing.info() << "[Precondition]";
      for (TraceConditionSet* pre : preconditions) pre->dump_conditions();

      log_tracing.info() << "[Anticondition]";
      for (TraceConditionSet* anti : anticonditions) anti->dump_conditions();

      log_tracing.info() << "[Postcondition]";
      for (TraceConditionSet* post : postconditions) post->dump_conditions();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::dump_instructions(
        const std::vector<Instruction*>& instructions) const
    //--------------------------------------------------------------------------
    {
      for (Instruction* instruction : instructions)
        log_tracing.info() << "  " << instruction->to_string(memo_entries);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::pack_recorder(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(runtime->address_space);
      rez.serialize(this);
      rez.serialize<DistributedID>(0);  // no coll
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_premap_output(
        MemoizableOp* memo, const Mapper::PremapTaskOutput& output,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID op_key = memo->get_trace_local_id();
      AutoLock t_lock(template_lock);
      legion_assert(is_recording());
      legion_assert(!output.reduction_futures.empty());
      legion_assert(
          cached_premappings.find(op_key) == cached_premappings.end());
      CachedPremapping& premapping = cached_premappings[op_key];
      premapping.future_locations = output.reduction_futures;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_premap_output(
        IndexTask* task, std::vector<Memory>& future_locations)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, false /*exclusive*/);
      legion_assert(is_replaying());
      CachedPremappings::const_iterator finder =
          cached_premappings.find(op_key);
      legion_assert(finder != cached_premappings.end());
      future_locations = finder->second.future_locations;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mapper_output(
        const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
        const std::deque<InstanceSet>& physical_instances, bool is_leaf,
        bool has_return_size, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
      legion_assert(is_recording());
      legion_assert(cached_mappings.find(tlid) == cached_mappings.end());
      CachedMapping& mapping = cached_mappings[tlid];
      // If you change the things recorded from output here then
      // you also need to change RemoteTraceRecorder::record_mapper_output
      mapping.target_procs = output.target_procs;
      mapping.chosen_variant = output.chosen_variant;
      mapping.task_priority = output.task_priority;
      mapping.postmap_task = output.postmap_task;
      mapping.future_locations = output.future_locations;
      mapping.physical_instances = physical_instances;
      for (const std::pair<const Memory, PoolBounds>& it :
           output.leaf_pool_bounds)
      {
        // Check to see if it is is bounded, if it is we can save it, if not
        // then we already issued a warning in the task that this is going
        // to invalidate the trace replay so do that now
        if (it.second.is_bounded())
          mapping.pool_bounds.emplace(std::make_pair(
              std::make_pair(it.first, true /*escaping*/), it.second));
        else
          record_no_consensus();
      }
      for (const std::pair<const Memory, PoolBounds>& it :
           output.non_escaping_leaf_pool_bounds)
      {
        // We should only be saving bounded pools here anyway
        if (it.second.is_bounded())
          mapping.pool_bounds.emplace(std::make_pair(
              std::make_pair(it.first, false /*escaping*/), it.second));
        else
          record_no_consensus();
      }
      for (InstanceSet& it : mapping.physical_instances)
      {
        for (unsigned idx = 0; idx < it.size(); idx++)
        {
          const InstanceRef& ref = it[idx];
          if (ref.is_virtual_ref())
            has_virtual_mapping = true;
        }
      }
      if (!is_leaf)
        has_non_leaf_task = true;
      if (!has_return_size)
        has_variable_return_size = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_mapper_output(
        SingleTask* task, VariantID& chosen_variant,
        TaskPriority& task_priority, bool& postmap_task,
        std::vector<Processor>& target_procs,
        std::vector<Memory>& future_locations,
        std::map<std::pair<Memory, bool>, PoolBounds>& pool_bounds,
        std::deque<InstanceSet>& physical_instances) const
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, false /*exclusive*/);
      legion_assert(is_replaying());
      CachedMappings::const_iterator finder = cached_mappings.find(op_key);
      legion_assert(finder != cached_mappings.end());
      chosen_variant = finder->second.chosen_variant;
      task_priority = finder->second.task_priority;
      postmap_task = finder->second.postmap_task;
      target_procs = finder->second.target_procs;
      future_locations = finder->second.future_locations;
      pool_bounds = finder->second.pool_bounds;
      physical_instances = finder->second.physical_instances;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_allreduce_mapping(
        AllReduceOp* allreduce, std::vector<Memory>& target_memories,
        size_t& future_size)
    //--------------------------------------------------------------------------
    {
      TraceLocalID op_key = allreduce->get_trace_local_id();
      AutoLock t_lock(template_lock, false /*exclusive*/);
      legion_assert(is_replaying());
      std::map<TraceLocalID, CachedAllreduce>::const_iterator finder =
          cached_allreduces.find(op_key);
      legion_assert(finder != cached_allreduces.end());
      target_memories = finder->second.target_memories;
      future_size = finder->second.future_size;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_replay_mapping(
        ApEvent lhs, unsigned op_kind, const TraceLocalID& tlid,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      unsigned lhs_ = convert_event(lhs);
      record_memo_entry(tlid, lhs_, op_kind);
      insert_instruction(new ReplayMapping(*this, lhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::request_term_event(ApUserEvent& term_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          !term_event.exists() || term_event.has_triggered_faultignorant());
      term_event = Runtime::create_ap_user_event(nullptr);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_create_ap_user_event(
        ApUserEvent& lhs, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      // Make the event here so it is on our local node
      // Note this is important for control replications where the
      // convert_event method will check this property
      lhs = Runtime::create_ap_user_event(nullptr);
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());

      unsigned lhs_ = convert_event(lhs);
      user_events[lhs_] = lhs;
      insert_instruction(new CreateApUserEvent(*this, lhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_trigger_event(
        ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs.exists());
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      // Do this first in case it gets pre-empted
      const unsigned rhs_ =
          rhs.exists() ? find_event(rhs, tpl_lock) : fence_completion_id;
#ifdef LEGION_DEBUG
      // Make sure we're always recording user events on the same shard
      // where the create user event is recorded
      unsigned lhs_ = std::numeric_limits<unsigned>::max();
      for (const std::pair<const unsigned, ApUserEvent>& it : user_events)
      {
        if (it.second != lhs)
          continue;
        lhs_ = it.first;
        break;
      }
      legion_assert(lhs_ != std::numeric_limits<unsigned>::max());
#else
      unsigned lhs_ = find_event(lhs, tpl_lock);
#endif
      events.emplace_back(ApEvent());
      insert_instruction(new TriggerEvent(*this, lhs_, rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(
        ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
        const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());

      std::set<unsigned> rhs_;
      for (unsigned idx = 0; idx < num_rhs; idx++)
      {
        std::map<ApEvent, unsigned>::const_iterator finder =
            event_map.find(rhs[idx]);
        if (finder != event_map.end())
          rhs_.insert(finder->second);
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

      if (lhs.exists())
      {
        // Check for reuse
        for (unsigned idx = 0; idx < num_rhs; idx++)
        {
          if (lhs != rhs[idx])
            continue;
          Runtime::rename_event(lhs);
          break;
        }
      }
      else
        Runtime::rename_event(lhs);

      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_merge_events(
        PredEvent& lhs, PredEvent e1, PredEvent e2, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      // need support for predicated execution with tracing
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_collective_barrier(
        ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      // should only be called on sharded physical templates
      std::abort();
    }

    //--------------------------------------------------------------------------
    ShardID PhysicalTemplate::record_barrier_creation(
        ApBarrier& bar, size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(!bar.exists());
      bar = runtime->create_ap_barrier(total_arrivals);
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned lhs = convert_event(bar);
      BarrierAdvance* advance =
          new BarrierAdvance(*this, bar, lhs, total_arrivals, true /*owner*/);
      insert_instruction(advance);
      // Save this as one of the barriers that we're managing
      managed_barriers[bar] = advance;
      return 0;  // No bothering with shards here
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_barrier_arrival(
        ApBarrier bar, ApEvent pre, size_t arrivals,
        std::set<RtEvent>& applied_events, ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
      // Should only be seeing things from ourself here
      legion_assert(owner_shard == 0);
      AutoLock tpl_lock(template_lock);
      legion_assert(bar.exists());
      legion_assert(is_recording());
      const unsigned rhs =
          pre.exists() ? find_event(pre, tpl_lock) : fence_completion_id;
      const unsigned lhs = events.size();
      events.emplace_back(ApEvent());
      BarrierArrival* arrival =
          new BarrierArrival(*this, bar, lhs, rhs, arrivals, true /*managed*/);
      insert_instruction(arrival);
      managed_arrivals[bar].emplace_back(arrival);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_copy(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, int priority,
        CollectiveKind collective, bool record_effect)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
        Runtime::rename_event(lhs);

      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      // Do this first in case it gets preempted
      const unsigned rhs_ = find_event(precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueCopy(
          *this, lhs_, expr, tlid, src_fields, dst_fields, reservations,
          src_tree_id, dst_tree_id, rhs_, src_unique, dst_unique, priority,
          collective, record_effect));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_fill(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, int priority, CollectiveKind collective,
        bool record_effect)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
        Runtime::rename_event(lhs);

      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      // Do this first in case it gets preempted
      const unsigned rhs_ = find_event(precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      insert_instruction(new IssueFill(
          *this, lhs_, expr, tlid, fields, fill_value, fill_size, fill_uid,
          handle, tree_id, rhs_, unique_event, priority, collective,
          record_effect));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_issue_across(
        const TraceLocalID& tlid, ApEvent& lhs, ApEvent collective_precondition,
        ApEvent copy_precondition, ApEvent src_indirect_precondition,
        ApEvent dst_indirect_precondition, CopyAcrossExecutor* executor)
    //--------------------------------------------------------------------------
    {
      if (!lhs.exists())
        Runtime::rename_event(lhs);

      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      unsigned copy_pre = find_event(copy_precondition, tpl_lock);
      unsigned collective_pre = 0, src_indirect_pre = 0, dst_indirect_pre = 0;
      if (collective_precondition.exists())
        collective_pre = find_event(collective_precondition, tpl_lock);
      if (src_indirect_precondition.exists())
        src_indirect_pre = find_event(src_indirect_precondition, tpl_lock);
      if (dst_indirect_precondition.exists())
        dst_indirect_pre = find_event(dst_indirect_precondition, tpl_lock);
      unsigned lhs_ = convert_event(lhs);
      IssueAcross* across = new IssueAcross(
          *this, lhs_, copy_pre, collective_pre, src_indirect_pre,
          dst_indirect_pre, tlid, executor);
      across_copies.emplace_back(across);
      insert_instruction(across);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_op_inst(
        const TraceLocalID& tlid, unsigned parent_req_index,
        const UniqueInst& inst, RegionNode* node, const RegionUsage& usage,
        const FieldMask& user_mask, bool update_validity,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      if (update_validity)
        record_instance_user(
            op_insts[tlid], inst, usage, node->row_source, user_mask, applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_instance_user(
        InstUsers& users, const UniqueInst& instance, const RegionUsage& usage,
        IndexSpaceExpression* expr, const FieldMask& mask,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      if (!IS_READ_ONLY(usage))
        record_mutated_instance(instance, expr, mask, applied);
      for (InstUsers::iterator it = users.begin(); it != users.end(); it++)
      {
        if (!it->matches(instance, usage, expr))
          continue;
        it->mask |= mask;
        return;
      }
      users.emplace_back(InstanceUser(instance, usage, expr, mask));
      if (recorded_views.find(instance.view_did) == recorded_views.end())
      {
        RtEvent ready;
        IndividualView* view = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(instance.view_did, ready));
        recorded_views[instance.view_did] = view;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_base_gc_ref(TRACE_REF);
      }
      if (recorded_expressions.insert(expr).second)
        expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_mutated_instance(
        const UniqueInst& inst, IndexSpaceExpression* expr,
        const FieldMask& mask, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      shrt::FieldMaskMap<IndexSpaceExpression>& insts = mutated_insts[inst];
      if (insts.empty() &&
          (recorded_views.find(inst.view_did) == recorded_views.end()))
      {
        RtEvent ready;
        IndividualView* view = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(inst.view_did, ready));
        recorded_views[inst.view_did] = view;
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        view->add_base_gc_ref(TRACE_REF);
      }
      if (insts.insert(expr, mask) && recorded_expressions.insert(expr).second)
        expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_fill_inst(
        ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& inst,
        const FieldMask& fill_mask, std::set<RtEvent>& applied_events,
        const bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage usage(LEGION_WRITE_ONLY, LEGION_EXCLUSIVE, 0);
      record_instance_user(
          copy_insts[lhs_], inst, usage, expr, fill_mask, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_copy_insts(
        ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
        unsigned dst_idx, IndexSpaceExpression* expr,
        const UniqueInst& src_inst, const UniqueInst& dst_inst,
        const FieldMask& src_mask, const FieldMask& dst_mask,
        PrivilegeMode src_mode, PrivilegeMode dst_mode, ReductionOpID redop,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage src_usage(src_mode, LEGION_EXCLUSIVE, 0);
      const RegionUsage dst_usage(dst_mode, LEGION_EXCLUSIVE, redop);
      record_instance_user(
          copy_insts[lhs_], src_inst, src_usage, expr, src_mask,
          applied_events);
      record_instance_user(
          copy_insts[lhs_], dst_inst, dst_usage, expr, dst_mask,
          applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_across_insts(
        ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
        unsigned dst_idx, IndexSpaceExpression* expr,
        const AcrossInsts& src_insts, const AcrossInsts& dst_insts,
        PrivilegeMode src_mode, PrivilegeMode dst_mode, bool src_indirect,
        bool dst_indirect, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned lhs_ = find_event(lhs, tpl_lock);
      const RegionUsage src_usage(src_mode, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it = src_insts.begin();
           it != src_insts.end(); it++)
        record_instance_user(
            src_indirect ? src_indirect_insts[lhs_] : copy_insts[lhs_],
            it->first, src_usage, expr, it->second, applied_events);
      const RegionUsage dst_usage(dst_mode, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it = dst_insts.begin();
           it != dst_insts.end(); it++)
        record_instance_user(
            dst_indirect ? dst_indirect_insts[lhs_] : copy_insts[lhs_],
            it->first, dst_usage, expr, it->second, applied_events);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_indirect_insts(
        ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
        const AcrossInsts& insts, std::set<RtEvent>& applied,
        PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned indirect = find_event(indirect_done, tpl_lock);
      const RegionUsage usage(privilege, LEGION_EXCLUSIVE, 0);
      for (AcrossInsts::const_iterator it = insts.begin(); it != insts.end();
           it++)
        record_instance_user(
            copy_insts[indirect], it->first, usage, expr, it->second, applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_set_op_sync_event(
        ApEvent& lhs, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      // Always make a fresh event here for these
      ApUserEvent rename = Runtime::create_ap_user_event(nullptr);
      Runtime::trigger_event_untraced(rename, lhs);
      lhs = rename;
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());

      insert_instruction(new SetOpSyncEvent(*this, convert_event(lhs), tlid));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_complete_replay(
        const TraceLocalID& tlid, ApEvent complete,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      // Do this first in case it gets preempted
      const unsigned complete_ =
          complete.exists() ? find_event(complete, tpl_lock) : 0;
      events.emplace_back(ApEvent());
      insert_instruction(new CompleteReplay(*this, tlid, complete_));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_reservations(
        const TraceLocalID& tlid,
        const std::map<Reservation, bool>& reservations,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      legion_assert(
          cached_reservations.find(tlid) == cached_reservations.end());
      cached_reservations[tlid] = reservations;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_future_allreduce(
        const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
        size_t future_size)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      legion_assert(cached_allreduces.find(tlid) == cached_allreduces.end());
      CachedAllreduce& allreduce = cached_allreduces[tlid];
      allreduce.target_memories = target_memories;
      allreduce.future_size = future_size;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_concurrent_group(
        IndexTask* task, Color color, size_t local, size_t global,
        RtBarrier barrier, const std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(!shards.empty());
      legion_assert(std::is_sorted(shards.begin(), shards.end()));
      const TraceLocalID tlid = task->get_trace_local_id();
      AutoLock tpl_lock(template_lock);
      concurrent_groups[tlid].emplace_back(
          ConcurrentGroup(color, local, global, barrier, shards));
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_concurrent_groups(IndexTask* task)
    //--------------------------------------------------------------------------
    {
      const TraceLocalID tlid = task->get_trace_local_id();
      // No need for a lock here, this data structur is read-only while
      // we're doing the trace replay other than the barrier advances
      // which don't race with each other
      std::map<TraceLocalID, std::vector<ConcurrentGroup> >::iterator finder =
          concurrent_groups.find(tlid);
      legion_assert(finder != concurrent_groups.end());
      for (ConcurrentGroup& group : finder->second)
      {
        task->initialize_concurrent_group(
            group.color, group.local, group.global, group.barrier,
            group.shards);
        Runtime::advance_barrier(group.barrier);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::get_task_reservations(
        SingleTask* task, std::map<Reservation, bool>& reservations) const
    //--------------------------------------------------------------------------
    {
      const TraceLocalID key = task->get_trace_local_id();
      AutoLock t_lock(template_lock, false /*exclusive*/);
      legion_assert(is_replaying());
      std::map<TraceLocalID, std::map<Reservation, bool> >::const_iterator
          finder = cached_reservations.find(key);
      legion_assert(finder != cached_reservations.end());
      reservations = finder->second;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_owner_shard(unsigned tid, ShardID owner)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_local_space(unsigned tid, IndexSpace sp)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_sharding_function(
        unsigned tid, ShardingFunction* function)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    ShardID PhysicalTemplate::find_owner_shard(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    IndexSpace PhysicalTemplate::find_local_space(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    ShardingFunction* PhysicalTemplate::find_sharding_function(unsigned tid)
    //--------------------------------------------------------------------------
    {
      // Only called on sharded physical template
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::initialize_replay(ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
      legion_assert(operations.empty());
      legion_assert(remaining_replays.load() == 0);
      legion_assert(!replay_postcondition.exists());
      if (total_replays++ == Realm::Barrier::MAX_PHASES)
      {
        replay_precondition = refresh_managed_barriers();
        // Reset it back to one after updating our barriers
        total_replays = 1;
      }
      else
        replay_precondition = RtEvent::NO_RT_EVENT;
      remaining_replays.store(slices.size());
      total_logical.store(0);
      // Check to see if we have a finished transitive reduction result
      check_finalize_transitive_reduction();

      if (recurrent)
      {
        if (last_fence != nullptr)
          events[fence_completion_id] = events[last_fence->complete];
        for (const std::pair<const unsigned, unsigned>& it : frontiers)
          events[it.second] = events[it.first];
      }
      else
      {
        events[fence_completion_id] = completion;
        for (const std::pair<const unsigned, unsigned>& it : frontiers)
          events[it.second] = completion;
      }

      for (const std::pair<const unsigned, unsigned>& it : crossing_events)
      {
        ApUserEvent ev = Runtime::create_ap_user_event(nullptr);
        events[it.first] = ev;
        user_events[it.first] = ev;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::start_replay(void)
    //--------------------------------------------------------------------------
    {
      const std::vector<Processor>& replay_targets =
          trace->get_replay_targets();
      legion_assert(remaining_replays.load() == slices.size());
      for (unsigned idx = 0; idx < slices.size(); ++idx)
      {
        ReplaySliceArgs args(this, idx, trace->is_recurrent());
        if (runtime->replay_on_cpus)
          runtime->issue_application_processor_task(
              args, LG_LOW_PRIORITY,
              replay_targets[idx % replay_targets.size()], replay_precondition);
        else
          runtime->issue_runtime_meta_task(
              args, LG_THROUGHPUT_WORK_PRIORITY, replay_precondition,
              replay_targets[idx % replay_targets.size()]);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalTemplate::refresh_managed_barriers(void)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID, std::map<ApEvent, ApBarrier> > notifications;
      for (std::pair<const ApEvent, BarrierAdvance*>& it : managed_barriers)
        it.second->refresh_barrier(it.first, notifications);
      if (!notifications.empty())
      {
        legion_assert(notifications.size() == 1);
        std::map<ShardID, std::map<ApEvent, ApBarrier> >::const_iterator local =
            notifications.begin();
        legion_assert(local->first == 0);
        legion_assert(local->second.size() == managed_arrivals.size());
        for (const std::pair<const ApEvent, ApBarrier>& it : local->second)
        {
          std::map<ApEvent, std::vector<BarrierArrival*> >::iterator finder =
              managed_arrivals.find(it.first);
          legion_assert(finder != managed_arrivals.end());
          for (unsigned idx = 0; idx < finder->second.size(); idx++)
            finder->second[idx]->set_managed_barrier(it.second);
        }
      }
      for (std::pair<const TraceLocalID, std::vector<ConcurrentGroup> >& cit :
           concurrent_groups)
      {
        for (ConcurrentGroup& group : cit.second)
        {
          legion_assert(group.shards.size() == 1);
          legion_assert(group.shards.back() == 0);
          group.barrier.destroy_barrier();
          group.barrier = runtime->create_rt_barrier(group.global);
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::finish_replay(
        FenceOp* fence, std::set<ApEvent>& postconditions)
    //--------------------------------------------------------------------------
    {
      if (remaining_replays.load() > 0)
      {
        RtEvent wait_on;
        {
          AutoLock tpl_lock(template_lock);
          if (remaining_replays.load() > 0)
          {
            legion_assert(!replay_postcondition.exists());
            replay_postcondition = Runtime::create_rt_user_event();
            wait_on = replay_postcondition;
          }
        }
        if (wait_on.exists())
        {
          wait_on.wait();
          replay_postcondition = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      for (const std::pair<const unsigned, unsigned>& it : frontiers)
        postconditions.insert(events[it.first]);
      if (last_fence != nullptr)
        postconditions.insert(events[last_fence->complete]);
      operations.clear();
      replay_complete = fence->get_completion_event();
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::defer_template_deletion(
        ApEvent& pending_deletion, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      pending_deletion = get_completion_for_deletion();
      if (!pending_deletion.exists() &&
          transitive_reduction_done.has_triggered())
      {
        check_finalize_transitive_reduction();
        return false;
      }
      RtEvent precondition = Runtime::protect_event(pending_deletion);
      if (transitive_reduction_done.exists() &&
          !transitive_reduction_done.has_triggered())
      {
        if (precondition.exists())
          precondition =
              Runtime::merge_events(precondition, transitive_reduction_done);
        else
          precondition = transitive_reduction_done;
      }
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeleteTemplateArgs args(this);
        applied_events.insert(runtime->issue_runtime_meta_task(
            args, LG_LOW_PRIORITY, precondition));
        return true;
      }
      else
      {
        check_finalize_transitive_reduction();
        return false;
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::ReplaySliceArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      tpl->execute_slice(slice_index, recurrent_replay);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::TransitiveReductionArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      tpl->transitive_reduction(state, true /*deferred*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::DeleteTemplateArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      tpl->check_finalize_transitive_reduction();
      delete tpl;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::record_memo_entry(
        const TraceLocalID& tlid, unsigned entry, unsigned op_kind)
    //--------------------------------------------------------------------------
    {
      legion_assert(memo_entries.find(tlid) == memo_entries.end());
      memo_entries[tlid] = std::pair<unsigned, unsigned>(entry, op_kind);
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG
    unsigned PhysicalTemplate::convert_event(const ApEvent& event, bool check)
#else
    unsigned PhysicalTemplate::convert_event(const ApEvent& event)
#endif
    //--------------------------------------------------------------------------
    {
      unsigned event_ = events.size();
      events.emplace_back(event);
      legion_assert(event_map.find(event) == event_map.end());
      event_map[event] = event_;
      return event_;
    }

    //--------------------------------------------------------------------------
    unsigned PhysicalTemplate::find_event(
        const ApEvent& event, AutoLock& tpl_lock)
    //--------------------------------------------------------------------------
    {
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      legion_assert(finder != event_map.end());
      legion_assert(finder->second != NO_INDEX);
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::insert_instruction(Instruction* inst)
    //--------------------------------------------------------------------------
    {
      legion_assert(instructions.size() + 1 == events.size());
      instructions.emplace_back(inst);
    }

    //--------------------------------------------------------------------------
    void PhysicalTemplate::find_all_last_users(
        const InstUsers& inst_users, std::set<unsigned>& users) const
    //--------------------------------------------------------------------------
    {
      for (InstUsers::const_iterator uit = inst_users.begin();
           uit != inst_users.end(); uit++)
      {
        std::map<UniqueInst, std::deque<LastUserResult> >::const_iterator
            finder = instance_last_users.find(uit->instance);
        legion_assert(finder != instance_last_users.end());
        for (const LastUserResult& it : finder->second)
        {
          if (!it.user.matches(*uit))
            continue;
          legion_assert(it.events.size() == it.frontiers.size());
          users.insert(it.frontiers.begin(), it.frontiers.end());
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalTemplate::are_read_only_users(InstUsers& inst_users)
    //--------------------------------------------------------------------------
    {
      for (InstUsers::const_iterator vit = inst_users.begin();
           vit != inst_users.end(); vit++)
      {
        // Scan through the other users and look for anything overlapping
        shrt::map<UniqueInst, shrt::FieldMaskMap<IndexSpaceExpression> >::
            const_iterator finder = mutated_insts.find(vit->instance);
        if (finder == mutated_insts.end())
          continue;
        if (vit->mask * finder->second.get_valid_mask())
          continue;
        for (shrt::FieldMaskMap<IndexSpaceExpression>::const_iterator it =
                 finder->second.begin();
             it != finder->second.end(); it++)
        {
          if (vit->mask * it->second)
            continue;
          IndexSpaceExpression* intersect =
              runtime->intersect_index_spaces(vit->expr, it->first);
          if (intersect->is_empty())
            continue;
          // Not immutable
          return false;
        }
      }
      return true;
    }

  }  // namespace Internal
}  // namespace Legion
