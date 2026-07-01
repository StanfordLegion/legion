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

#include "legion/analysis/filter.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/region.h"
#include "legion/operations/remote.h"
#include "legion/views/instance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Filter Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(
        Operation* o, unsigned idx, RegionNode* node,
        const PhysicalTraceInfo& t_info, const bool remove_restrict)
      : RegistrationAnalysis(
            o, idx, node, true /*on heap*/, t_info, true /*exclusive*/),
        remove_restriction(remove_restrict)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        RegionNode* node, const PhysicalTraceInfo& t_info,
        const op::FieldMaskMap<InstanceView>& views, CollectiveMapping* mapping,
        const bool first, const bool remove_restrict,
        std::map<InstanceView*, LamportClock>&& clocks)
      : RegistrationAnalysis(
            src, prev, o, idx, node, true /*on heap*/, t_info, mapping, first,
            true /*exclusive*/),
        filter_views(views), remote_lamport_clocks(std::move(clocks)),
        remove_restriction(remove_restrict)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FilterAnalysis::~FilterAnalysis(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<InstanceView* const, LamportClock>& it :
           remote_lamport_clocks)
        it.first->unpack_global_ref(it.second);
    }

    //--------------------------------------------------------------------------
    RtEvent FilterAnalysis::perform_traversal(
        RtEvent precondition, const VersionInfo& info,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
        return defer_traversal(precondition, info, applied_events);
      if (!target_views.empty())
      {
        legion_assert(target_views.size() == 1);
        filter_views = target_views.back();
      }
      return RegistrationAnalysis::perform_traversal(
          precondition, info, applied_events);
    }

    //--------------------------------------------------------------------------
    bool FilterAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->filter_set(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // Perform a check for migration after this
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent FilterAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteFilterAnalysis rez;
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
          rez.serialize<size_t>(filter_views.size());
          for (const std::pair<InstanceView* const, FieldMask>& it :
               filter_views)
          {
            rez.serialize(it.first->did);
            rez.serialize(it.second);
            it.first->pack_global_ref(rez);
          }
          rez.serialize(remove_restriction);
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
          rez.serialize(applied);
        }
        rez.dispatch(target);
        applied_events.insert(applied);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteFilterAnalysis::handle(
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
      op::FieldMaskMap<InstanceView> filter_views;
      size_t num_views;
      derez.deserialize(num_views);
      std::map<InstanceView*, LamportClock> lamport_clocks;
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent view_ready;
        InstanceView* inst_view = static_cast<InstanceView*>(
            runtime->find_or_request_logical_view(view_did, view_ready));
        FieldMask mask;
        derez.deserialize(mask);
        filter_views.insert(inst_view, mask);
        if (view_ready.exists())
          ready_events.insert(view_ready);
        lamport_clocks[inst_view] =
            InstanceView::unpack_global_ref_clock(derez);
      }
      bool remove_restriction;
      derez.deserialize(remove_restriction);
      std::set<RtEvent> deferral_events, applied_events;
      const PhysicalTraceInfo trace_info =
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
      RtUserEvent applied;
      derez.deserialize(applied);

      // This takes ownership of the remote operation
      FilterAnalysis* analysis = new FilterAnalysis(
          original_source, previous, op, index, region, trace_info,
          filter_views, collective_mapping, first_local, remove_restriction,
          std::move(lamport_clocks));
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
        analysis->perform_remote(traversal_done, applied_events);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

  }  // namespace Internal
}  // namespace Legion
