/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/analysis/update.h"
#include "legion/analysis/aggregator.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/nodes/region.h"
#include "legion/operations/remote.h"
#include "legion/utilities/collectives.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Update Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(
        Operation* o, unsigned idx, const RegionRequirement& req,
        RegionNode* rn, const PhysicalTraceInfo& t_info, const ApEvent pre,
        const ApEvent term, const bool check, const bool record)
      : CollectiveCopyFillAnalysis(
            o, idx, rn, true /*on heap*/, t_info, IS_WRITE(req)),
        usage(req), precondition(pre), term_event(term),
        // Don't support checking initialized for simultaneous because of
        // must epoch operations which need a total order on mapping points
        check_initialized(
            check && !IS_WRITE_DISCARD(usage) && !IS_SIMULT(usage)),
        record_valid(record), output_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        const RegionUsage& use, RegionNode* rn,
        std::vector<PhysicalManager*>&& target_insts,
        op::vector<op::FieldMaskMap<InstanceView>>&& target_vws,
        std::vector<IndividualView*>&& source_vws,
        const PhysicalTraceInfo& info, CollectiveMapping* mapping,
        const RtEvent user_reg, const ApEvent pre, const ApEvent term,
        const bool check, const bool record, const bool first_local)
      : CollectiveCopyFillAnalysis(
            src, prev, o, idx, rn, true /*on heap*/, std::move(target_insts),
            std::move(target_vws), std::move(source_vws), info, mapping,
            first_local, IS_WRITE(use)),
        usage(use), precondition(pre), term_event(term),
        check_initialized(check), record_valid(record),
        output_aggregator(nullptr), remote_user_registered(user_reg)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UpdateAnalysis::~UpdateAnalysis(void)
    //--------------------------------------------------------------------------
    {
      // If we didn't perform a registration and someone wanted to know that
      // the registration was done then we need to trigger that
      if (user_registered.exists())
        Runtime::trigger_event(user_registered);
    }

    //--------------------------------------------------------------------------
    void UpdateAnalysis::record_uninitialized(
        const FieldMask& uninit, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (!uninitialized)
      {
        legion_assert(!uninitialized_reported.exists());
        uninitialized_reported = Runtime::create_rt_user_event();
        applied_events.insert(uninitialized_reported);
      }
      uninitialized |= uninit;
    }

    //--------------------------------------------------------------------------
    bool UpdateAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->update_set(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // Perform a check for migration
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent UpdateAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // Easy out if we don't have any remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      legion_assert(!target_instances.empty());
      legion_assert(target_instances.size() == target_views.size());
      if (!remote_user_registered.exists())
      {
        legion_assert(original_source == runtime->address_space);
        legion_assert(!user_registered.exists());
        user_registered = Runtime::create_rt_user_event();
        remote_user_registered = user_registered;
      }
      std::set<RtEvent> remote_events;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent updated = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteUpdateAnalysis rez;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(region->handle);
          rez.serialize(usage);
          rez.serialize<size_t>(target_instances.size());
          for (unsigned idx = 0; idx < target_instances.size(); idx++)
          {
            rez.serialize(target_instances[idx]->did);
            rez.serialize<size_t>(target_views[idx].size());
            for (const std::pair<InstanceView* const, FieldMask>& it :
                 target_views[idx])
            {
              rez.serialize(it.first->did);
              rez.serialize(it.second);
            }
          }
          rez.serialize<size_t>(source_views.size());
          for (unsigned idx = 0; idx < source_views.size(); idx++)
            rez.serialize(source_views[idx]->did);
          trace_info.pack_trace_info(rez);
          // We only need to pack the collective mapping once when going
          // from the origin space to the next space
          CollectiveMapping* mapping = get_replicated_mapping();
          if ((mapping != nullptr) &&
              (original_source == runtime->address_space))
          {
            mapping->pack(rez);
            rez.serialize<bool>(is_collective_first_local());
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize(precondition);
          rez.serialize(term_event);
          rez.serialize(updated);
          rez.serialize(remote_user_registered);
          rez.serialize(applied);
          rez.serialize<bool>(check_initialized);
          rez.serialize<bool>(record_valid);
        }
        rez.dispatch(target);
        remote_events.insert(updated);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(remote_events);
    }

    //--------------------------------------------------------------------------
    RtEvent UpdateAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_updates(perform_precondition, applied_events);
      // Report any uninitialized data now that we know the traversal is done
      if (!!uninitialized)
      {
        legion_assert(check_initialized);
        legion_assert(uninitialized_reported.exists());
        region->report_uninitialized_usage(
            op, index, uninitialized, uninitialized_reported);
      }
      if (!input_aggregators.empty())
      {
#ifndef NON_AGGRESSIVE_AGGREGATORS
        const bool is_local = (original_source == runtime->address_space);
#endif
        for (const std::pair<const RtEvent, CopyFillAggregator*>& it :
             input_aggregators)
        {
          it.second->issue_updates(trace_info, precondition);
#ifdef NON_AGGRESSIVE_AGGREGATORS
          if (!it.second->effects_applied.has_triggered())
            guard_events.insert(it.second->effects_applied);
#else
          if (!it.second->effects_applied.has_triggered())
          {
            if (is_local)
            {
              if (!it.second->guard_postcondition.has_triggered())
                guard_events.insert(it.second->guard_postcondition);
              applied_events.insert(it.second->effects_applied);
            }
            else
              guard_events.insert(it.second->effects_applied);
          }
#endif
          if (it.second->release_guards(applied_events))
            delete it.second;
        }
      }
      if (!guard_events.empty())
        return Runtime::merge_events(guard_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent UpdateAnalysis::perform_registration(
        RtEvent precondition, const RegionUsage& usage,
        std::set<RtEvent>& applied_events, ApEvent init_precondition,
        ApEvent termination, ApEvent& instances_ready, bool symbolic)
    //--------------------------------------------------------------------------
    {
      legion_assert(usage == this->usage);
      // Should always be local
      legion_assert(original_source == runtime->address_space);
      if (precondition.exists() && !precondition.has_triggered())
        return defer_registration(
            precondition, usage, applied_events, trace_info, init_precondition,
            termination, instances_ready, symbolic);

      // Invoke the base implementation to see if we actually do it now
      const RtEvent registered =
          CollectiveCopyFillAnalysis::perform_registration(
              precondition, usage, applied_events, init_precondition,
              termination, instances_ready, symbolic);
      // If we're doing a collective read-write then check to make sure that
      // we have exactly one arrival on each of the instances, if we don't
      // then we're not going to have the isolation that we expect. We make
      // an exception for this in the case of write-only where we allow for
      // multiple point tasks to stomp on the same instance since we know
      // that they are all writing the same values in the end anyway.
      if (!collective_arrivals.empty() && IS_WRITE(usage) && HAS_READ(usage))
      {
        legion_assert(IS_COLLECTIVE(usage));
        for (const std::pair<InstanceView* const, size_t>& it :
             collective_arrivals)
          if ((it.second > 1) &&
              (it.first->is_individual_view() ||
               (it.first->as_collective_view()->local_views.size() <
                it.second)))
          {
            Error err(LEGION_MAPPER_EXCEPTION);
            err << "Illegal mapper output: detected multiple write-collective "
                << "users of the same instance on region requirement " << index
                << " of " << op->get_logging_name() << " (UID "
                << op->get_unique_op_id() << "). "
                << "For read-write collectives it is mandatory "
                << "that every point map to a separate instance.";
            err.raise();
          }
      }
      if (user_registered.exists())
      {
        Runtime::trigger_event(user_registered, registered);
        user_registered = RtUserEvent::NO_RT_USER_EVENT;
      }
      return registered;
    }

    //--------------------------------------------------------------------------
    ApEvent UpdateAnalysis::perform_output(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_output(
            perform_precondition, trace_info, false /*track*/, applied_events);
      if (output_aggregator != nullptr)
      {
        output_aggregator->issue_updates(
            trace_info, term_event, true /*restricted output*/);
        if (output_aggregator->effects_applied.has_triggered())
          applied_events.insert(output_aggregator->effects_applied);
        if (output_aggregator->release_guards(applied_events))
          delete output_aggregator;
      }
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteUpdateAnalysis::handle(
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
      FieldMask user_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        user_mask |= eq_masks[idx];
      }
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionUsage usage;
      derez.deserialize(usage);
      size_t num_targets;
      derez.deserialize(num_targets);
      std::vector<PhysicalManager*> targets(num_targets);
      op::vector<op::FieldMaskMap<InstanceView>> target_views(num_targets);
      for (unsigned idx1 = 0; idx1 < num_targets; idx1++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        targets[idx1] = runtime->find_or_request_instance_manager(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        size_t num_views;
        derez.deserialize(num_views);
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          derez.deserialize(did);
          LogicalView* view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          target_views[idx1].insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_sources;
      derez.deserialize(num_sources);
      std::vector<IndividualView*> source_views(num_sources);
      for (unsigned idx = 0; idx < num_sources; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        source_views[idx] = static_cast<IndividualView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      std::set<RtEvent> deferral_events, applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      bool first_local = true;
      size_t collective_mapping_size;
      derez.deserialize(collective_mapping_size);
      CollectiveMapping* collective_mapping =
          ((collective_mapping_size) > 0) ?
              new CollectiveMapping(derez, collective_mapping_size) :
              nullptr;
      if (collective_mapping != nullptr)
        derez.deserialize<bool>(first_local);
      ApEvent precondition;
      derez.deserialize(precondition);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtUserEvent updated;
      derez.deserialize(updated);
      RtEvent remote_user_registered;
      derez.deserialize(remote_user_registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      bool check_initialized;
      derez.deserialize(check_initialized);
      bool record_valid;
      derez.deserialize(record_valid);

      RegionNode* node = runtime->get_node(handle);
      // This takes ownership of the remote operation
      UpdateAnalysis* analysis = new UpdateAnalysis(
          original_source, previous, op, index, usage, node, std::move(targets),
          std::move(target_views), std::move(source_views), trace_info,
          collective_mapping, remote_user_registered, precondition, term_event,
          check_initialized, record_valid, first_local);
      analysis->add_reference();
      // Make sure that all our pointers are ready
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
      std::set<RtEvent> update_events;
      // If we have remote messages to send do that now
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready =
            analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          update_events.insert(remote_ready);
      }
      // Then perform the updates
      // Note that we need to capture all the effects of these updates
      // before we can consider them applied, so we can't use the
      // applied_events data structure here
      const RtEvent updates_ready =
          analysis->perform_updates(traversal_done, update_events);
      if (updates_ready.exists())
        update_events.insert(updates_ready);
      // We can trigger our updated event done when all the guards are done
      if (!update_events.empty())
        Runtime::trigger_event(updated, Runtime::merge_events(update_events));
      else
        Runtime::trigger_event(updated);
      // If we have outputs we need for the user to be registered
      // before we can apply the output copies
      analysis->perform_output(remote_user_registered, applied_events);
      // Do the rest of the triggers
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

  }  // namespace Internal
}  // namespace Legion
