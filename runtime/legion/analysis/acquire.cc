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

#include "legion/analysis/acquire.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/kernel/runtime.h"
#include "legion/managers/message.h"
#include "legion/nodes/region.h"
#include "legion/operations/remote.h"
#include "legion/views/logical.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Acquire Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(
        Operation* o, unsigned idx, RegionNode* node,
        const PhysicalTraceInfo& t_info)
      : RegistrationAnalysis(
            o, idx, node, true /*on heap*/, t_info, true /*exclusive*/),
        target_analysis(this)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        RegionNode* node, AcquireAnalysis* t, const PhysicalTraceInfo& t_info,
        CollectiveMapping* mapping, bool first_local)
      : RegistrationAnalysis(
            src, prev, o, idx, node, true /*on heap*/, t_info, mapping,
            first_local, true /*exclusive*/),
        target_analysis(t)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AcquireAnalysis::~AcquireAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool AcquireAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->acquire_restrictions(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // Perform a check for migration after this
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent AcquireAnalysis::perform_remote(
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
        RemoteAcquireAnalysis rez;
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
          rez.serialize(region->handle);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
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
    RtEvent AcquireAnalysis::perform_updates(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //-------------------------------------------------------------------------
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
    /*static*/ void RemoteAcquireAnalysis::handle(
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
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionNode* region = runtime->get_node(handle);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent applied;
      derez.deserialize(applied);
      AcquireAnalysis* target;
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

      // This takes ownership of the operation
      AcquireAnalysis* analysis = new AcquireAnalysis(
          original_source, previous, op, index, region, target, trace_info,
          mapping, first_local);
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
