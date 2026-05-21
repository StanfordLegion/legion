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

#include "legion/analysis/release.h"
#include "legion/analysis/aggregator.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/region.h"
#include "legion/operations/remote.h"
#include "legion/instances/physical.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Release Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(
        Operation* o, unsigned idx, ApEvent pre, RegionNode* node,
        const PhysicalTraceInfo& t_info)
      : CollectiveCopyFillAnalysis(
            o, idx, node, true /*on heap*/, t_info, true /*exclusive*/),
        precondition(pre), target_analysis(this), release_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        RegionNode* node, ApEvent pre, ReleaseAnalysis* t,
        std::vector<PhysicalManager*>&& target_insts,
        op::vector<op::FieldMaskMap<InstanceView>>&& target_vws,
        std::vector<IndividualView*>&& source_vws,
        const PhysicalTraceInfo& info, CollectiveMapping* mapping,
        const bool first)
      : CollectiveCopyFillAnalysis(
            src, prev, o, idx, node, true /*on heap*/, std::move(target_insts),
            std::move(target_vws), std::move(source_vws), info, mapping, first,
            true /*exclusive*/),
        precondition(pre), target_analysis(t), release_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReleaseAnalysis::~ReleaseAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool ReleaseAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->release_restrictions(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // Perform a check for migration after this
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent ReleaseAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // Easy out if there is nothing to do
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> remote_events;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent returned = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteReleaseAnalysis rez;
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
          for (IndividualView* const it : source_views) rez.serialize(it->did);
          rez.serialize(region->handle);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(precondition);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(target_analysis);
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
        }
        rez.dispatch(target);
        applied_events.insert(applied);
        remote_events.insert(returned);
      }
      return Runtime::merge_events(remote_events);
    }

    //--------------------------------------------------------------------------
    RtEvent ReleaseAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Defer this if necessary
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // See if we have any instance names to send back
      if ((target_analysis != this) && (recorded_instances != nullptr))
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
          applied_events.insert(response_event);
        }
        else
          target_analysis->process_local_instances(
              *recorded_instances, restricted);
      }
      if (release_aggregator != nullptr)
      {
        std::set<RtEvent> guard_events;
        release_aggregator->issue_updates(trace_info, precondition);
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (release_aggregator->effects_applied.has_triggered())
          guard_events.insert(release_aggregator->effects_applied);
#else
        if (!release_aggregator->effects_applied.has_triggered())
        {
          if (original_source == runtime->address_space)
          {
            if (!release_aggregator->guard_postcondition.has_triggered())
              guard_events.insert(release_aggregator->guard_postcondition);
            applied_events.insert(release_aggregator->effects_applied);
          }
          else
            guard_events.insert(release_aggregator->effects_applied);
        }
#endif
        if (release_aggregator->release_guards(applied_events))
          delete release_aggregator;
        if (!guard_events.empty())
          return Runtime::merge_events(guard_events);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteReleaseAnalysis::handle(
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
        source_views[idx] = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(did, ready));
        if (ready.exists())
          ready_events.insert(ready);
      }
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionNode* region = runtime->get_node(handle);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      ApEvent precondition;
      derez.deserialize(precondition);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent applied;
      derez.deserialize(applied);
      ReleaseAnalysis* target;
      derez.deserialize(target);
      std::set<RtEvent> deferral_events, applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      size_t collective_mapping_size;
      derez.deserialize(collective_mapping_size);
      CollectiveMapping* mapping =
          ((collective_mapping_size) > 0) ?
              new CollectiveMapping(derez, collective_mapping_size) :
              nullptr;
      bool first_local = true;
      if (mapping != nullptr)
        derez.deserialize<bool>(first_local);

      ReleaseAnalysis* analysis = new ReleaseAnalysis(
          original_source, previous, op, index, region, precondition, target,
          std::move(targets), std::move(target_views), std::move(source_views),
          trace_info, mapping, first_local);
      analysis->add_reference();
      RtEvent ready_event;
      // Make sure that all our pointers are ready
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
      // Note that we use the ready events here for applied so that
      // we can know that all our updates are done before we tell
      // the original source node that we've returned
      const RtEvent local_ready = analysis->perform_updates(
          traversal_done, (original_source == runtime->address_space) ?
                              applied_events :
                              ready_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(returned, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(returned);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

  }  // namespace Internal
}  // namespace Legion
