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

#include "legion/analysis/valid.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/expression.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Valid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(
        Operation* o, unsigned idx, IndexSpaceExpression* expr,
        ReductionOpID red)
      : PhysicalAnalysis(o, idx, expr, false /*on heap*/, true /*immutable*/),
        redop(red), target_analysis(this)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        IndexSpaceExpression* expr, ValidInstAnalysis* t, ReductionOpID red)
      : PhysicalAnalysis(src, prev, o, idx, expr, true /*on heap*/), redop(red),
        target_analysis(t)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::~ValidInstAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool ValidInstAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_valid_instances(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // No migration check for reading
      return false;
    }

    //--------------------------------------------------------------------------
    RtEvent ValidInstAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        EquivalenceSetRequestInstances rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit.second.size());
          for (const std::pair<EquivalenceSet* const, FieldMask>& it :
               rit.second)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(redop);
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        rez.dispatch(target);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent ValidInstAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_updates(perform_precondition, applied_events);
      if (recorded_instances != nullptr)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          EquivalenceSetRemoteInstances rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (const std::pair<LogicalView* const, FieldMask>& it :
                 *recorded_instances)
            {
              rez.serialize(it.first->did);
              rez.serialize(it.second);
            }
            rez.serialize<bool>(restricted);
          }
          rez.dispatch(original_source);
          return response_event;
        }
        else
          target_analysis->process_local_instances(
              *recorded_instances, restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetRequestInstances::handle(
        Deserializer& derez, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;

      std::vector<EquivalenceSet*> eq_sets(num_eq_sets);
      op::vector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, previous);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      ReductionOpID redop;
      derez.deserialize(redop);
      ValidInstAnalysis* target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      ValidInstAnalysis* analysis = new ValidInstAnalysis(
          original_source, previous, op, index, expr, target, redop);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->analyze(
            eq_sets[idx], eq_masks[idx], deferral_events, applied_events,
            ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready =
            analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready =
          analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Invalid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(
        Operation* o, unsigned idx, IndexSpaceExpression* expr,
        const lng::FieldMaskMap<LogicalView>& valid_insts)
      : PhysicalAnalysis(o, idx, expr, true /*on heap*/, true /*immutable*/),
        valid_instances(valid_insts), target_analysis(this)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        IndexSpaceExpression* expr, InvalidInstAnalysis* t,
        const op::FieldMaskMap<LogicalView>& valid_insts)
      : PhysicalAnalysis(src, prev, o, idx, expr, true /*on heap*/),
        valid_instances(valid_insts), target_analysis(t)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::~InvalidInstAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool InvalidInstAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_invalid_instances(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // No migration check for reading
      return false;
    }

    //--------------------------------------------------------------------------
    RtEvent InvalidInstAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        EquivalenceSetRequestInvalid rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit.second.size());
          for (const std::pair<EquivalenceSet* const, FieldMask>& it :
               rit.second)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(valid_instances.size());
          for (const std::pair<LogicalView* const, FieldMask>& it :
               valid_instances)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
          }
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        rez.dispatch(target);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent InvalidInstAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_updates(perform_precondition, applied_events);
      if (recorded_instances != nullptr)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          EquivalenceSetRemoteInstances rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (const std::pair<LogicalView* const, FieldMask>& it :
                 *recorded_instances)
            {
              rez.serialize(it.first->did);
              rez.serialize(it.second);
            }
            rez.serialize<bool>(restricted);
          }
          rez.dispatch(original_source);
          return response_event;
        }
        else
          target_analysis->process_local_instances(
              *recorded_instances, restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetRequestInvalid::handle(
        Deserializer& derez, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, nullptr);
      op::vector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, previous);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      op::FieldMaskMap<LogicalView> valid_instances;
      size_t num_valid_instances;
      derez.deserialize<size_t>(num_valid_instances);
      for (unsigned idx = 0; idx < num_valid_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask view_mask;
        derez.deserialize(view_mask);
        valid_instances.insert(view, view_mask);
      }
      InvalidInstAnalysis* target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      InvalidInstAnalysis* analysis = new InvalidInstAnalysis(
          original_source, previous, op, index, expr, target, valid_instances);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->analyze(
            eq_sets[idx], eq_masks[idx], deferral_events, applied_events,
            ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready =
            analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready =
          analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Antivalid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::AntivalidInstAnalysis(
        Operation* o, unsigned idx, IndexSpaceExpression* expr,
        const op::FieldMaskMap<LogicalView>& anti_insts)
      : PhysicalAnalysis(o, idx, expr, true /*on heap*/, true /*immutable*/),
        antivalid_instances(anti_insts), target_analysis(this)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::AntivalidInstAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        IndexSpaceExpression* expr, AntivalidInstAnalysis* a,
        const op::FieldMaskMap<LogicalView>& anti_insts)
      : PhysicalAnalysis(src, prev, o, idx, expr, true /*on heap*/),
        antivalid_instances(anti_insts), target_analysis(a)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::~AntivalidInstAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool AntivalidInstAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_antivalid_instances(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // No migration check for reading
      return false;
    }

    //--------------------------------------------------------------------------
    RtEvent AntivalidInstAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        EquivalenceSetRequestAntivalid rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit.second.size());
          for (const std::pair<EquivalenceSet* const, FieldMask>& it :
               rit.second)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(antivalid_instances.size());
          for (const std::pair<LogicalView* const, FieldMask>& it :
               antivalid_instances)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
          }
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        rez.dispatch(target);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent AntivalidInstAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_updates(perform_precondition, applied_events);
      if (recorded_instances != nullptr)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          EquivalenceSetRemoteInstances rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (const std::pair<LogicalView* const, FieldMask>& it :
                 *recorded_instances)
            {
              rez.serialize(it.first->did);
              rez.serialize(it.second);
            }
            rez.serialize<bool>(restricted);
          }
          rez.dispatch(original_source);
          return response_event;
        }
        else
          target_analysis->process_local_instances(
              *recorded_instances, restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSetRequestAntivalid::handle(
        Deserializer& derez, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, nullptr);
      op::vector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, previous);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      op::FieldMaskMap<LogicalView> antivalid_instances;
      size_t num_antivalid_instances;
      derez.deserialize<size_t>(num_antivalid_instances);
      for (unsigned idx = 0; idx < num_antivalid_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask view_mask;
        derez.deserialize(view_mask);
        antivalid_instances.insert(view, view_mask);
      }
      AntivalidInstAnalysis* target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      AntivalidInstAnalysis* analysis = new AntivalidInstAnalysis(
          original_source, previous, op, index, expr, target,
          antivalid_instances);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->analyze(
            eq_sets[idx], eq_masks[idx], deferral_events, applied_events,
            ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready =
            analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready =
          analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

  }  // namespace Internal
}  // namespace Legion
