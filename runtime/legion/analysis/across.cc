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

#include "legion/analysis/across.h"
#include "legion/analysis/aggregator.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/managers/message.h"
#include "legion/nodes/region.h"
#include "legion/nodes/across.h"
#include "legion/operations/remote.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Copy Across Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::CopyAcrossAnalysis(
        Operation* o, unsigned src_idx, unsigned dst_idx,
        const RegionRequirement& src_req, const RegionRequirement& dst_req,
        const InstanceSet& target_insts,
        const op::vector<op::FieldMaskMap<InstanceView>>& target_vws,
        const std::vector<IndividualView*>& source_vws, const ApEvent pre,
        const ApEvent dst_ready, const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned>& src_idxes,
        const std::vector<unsigned>& dst_idxes, const PhysicalTraceInfo& t_info,
        const bool perf)
      : PhysicalAnalysis(
            o, dst_idx, runtime->get_node(dst_req.region)->row_source,
            true /*on heap*/, false /*immutable*/),
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)),
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), src_usage(src_req),
        dst_usage(dst_req), src_region(src_req.region),
        dst_region(dst_req.region), targets_ready(dst_ready),
        target_instances(convert_instances(target_insts)),
        target_views(target_vws), source_views(source_vws), precondition(pre),
        pred_guard(pred), redop(red), src_indexes(src_idxes),
        dst_indexes(dst_idxes),
        across_helper(
            perf ? nullptr :
                   create_across_helper(
                       src_mask, dst_mask, target_instances, target_views,
                       src_indexes, dst_indexes)),
        trace_info(t_info), perfect(perf), across_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::CopyAcrossAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned src_idx,
        unsigned dst_idx, const RegionUsage& src_use,
        const RegionUsage& dst_use, const LogicalRegion src_reg,
        const LogicalRegion dst_reg, const ApEvent dst_ready,
        std::vector<PhysicalManager*>&& target_insts,
        op::vector<op::FieldMaskMap<InstanceView>>&& target_vws,
        std::vector<IndividualView*>&& source_vws, const ApEvent pre,
        const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned>& src_idxes,
        const std::vector<unsigned>& dst_idxes, const PhysicalTraceInfo& t_info,
        const bool perf)
      : PhysicalAnalysis(
            src, prev, o, dst_idx, runtime->get_node(dst_reg)->row_source,
            true /*on heap*/),
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)),
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), src_usage(src_use),
        dst_usage(dst_use), src_region(src_reg), dst_region(dst_reg),
        targets_ready(dst_ready), target_instances(target_insts),
        target_views(target_vws), source_views(source_vws), precondition(pre),
        pred_guard(pred), redop(red), src_indexes(src_idxes),
        dst_indexes(dst_idxes),
        across_helper(
            perf ? nullptr :
                   create_across_helper(
                       src_mask, dst_mask, target_instances, target_views,
                       src_indexes, dst_indexes)),
        trace_info(t_info), perfect(perf), across_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::~CopyAcrossAnalysis(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          !aggregator_guard.exists() || aggregator_guard.has_triggered());
      if (across_helper != nullptr)
        delete across_helper;
    }

    //--------------------------------------------------------------------------
    void CopyAcrossAnalysis::record_uninitialized(
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
    CopyFillAggregator* CopyAcrossAnalysis::get_across_aggregator(void)
    //--------------------------------------------------------------------------
    {
      if (across_aggregator == nullptr)
      {
        legion_assert(!aggregator_guard.exists());
        aggregator_guard = Runtime::create_rt_user_event();
        across_aggregator = new CopyFillAggregator(
            this, src_index, dst_index, nullptr /*no previous guard*/,
            true /*track*/, pred_guard, aggregator_guard);
      }
      return across_aggregator;
    }

    //--------------------------------------------------------------------------
    bool CopyAcrossAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->issue_across_copies(
          *this, mask, expr, expr_covers, applied_events, already_deferred);
      // Perform a check for migration after this
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent CopyAcrossAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      legion_assert(target_instances.size() == target_views.size());
      legion_assert(src_indexes.size() == dst_indexes.size());
      for (const std::pair<
               const AddressSpaceID, op::FieldMaskMap<EquivalenceSet>>& rit :
           remote_sets)
      {
        legion_assert(!rit.second.empty());
        const AddressSpaceID target = rit.first;
        const ApUserEvent copy = Runtime::create_ap_user_event(&trace_info);
        const RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteCopyAcrossAnalysis rez;
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
          rez.serialize(src_index);
          rez.serialize(dst_index);
          rez.serialize(src_usage);
          rez.serialize(dst_usage);
          rez.serialize(targets_ready);
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
          rez.serialize(src_region);
          rez.serialize(dst_region);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(redop);
          rez.serialize<bool>(perfect);
          if (!perfect)
          {
            rez.serialize<size_t>(src_indexes.size());
            for (unsigned idx = 0; idx < src_indexes.size(); idx++)
            {
              rez.serialize(src_indexes[idx]);
              rez.serialize(dst_indexes[idx]);
            }
          }
          rez.serialize(applied);
          rez.serialize(copy);
          trace_info.pack_trace_info(rez);
        }
        rez.dispatch(target);
        applied_events.insert(applied);
        copy_events.emplace_back(copy);
      }
      // Filter all the remote expressions from the local ones here
      filter_remote_expressions(local_exprs);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent CopyAcrossAnalysis::perform_updates(
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
        legion_assert(uninitialized_reported.exists());
        RegionNode* src_node = runtime->get_node(src_region);
        src_node->report_uninitialized_usage(
            op, src_index, uninitialized, uninitialized_reported);
      }
      if (across_aggregator != nullptr)
      {
        legion_assert(aggregator_guard.exists());
        legion_assert(across_aggregator->track_events);
        // Trigger the guard event for the aggregator once all the
        // actual guard events are done. Note that this is safe for
        // copy across aggregators because unlike other aggregators
        // they are moving data from one field to another so it is
        // safe to create entanglements between fields since they are
        // all going to be subsumed by the same completion event for
        // the copy-across operation anyway
        if (!guard_events.empty())
          Runtime::trigger_event(
              aggregator_guard, Runtime::merge_events(guard_events));
        else
          Runtime::trigger_event(aggregator_guard);
        // Record the event field preconditions for each view
        std::map<InstanceView*, std::vector<ApEvent>> dst_events;
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          for (const std::pair<InstanceView* const, FieldMask>& it :
               target_views[idx])
          {
            // Always instantiate the entry in the map
            dst_events[it.first].emplace_back(targets_ready);
          }
        }
        // This is a copy-across aggregator so the destination events
        // are being handled by the copy operation that mapped the
        // target instance for us
        const ApEvent effect = across_aggregator->issue_updates(
            trace_info, precondition, false /*restricted*/,
            false /*manage dst events*/, &dst_events);
        if (effect.exists())
          copy_events.emplace_back(effect);
        if (!across_aggregator->effects_applied.has_triggered())
          applied_events.insert(across_aggregator->effects_applied);
        if (across_aggregator->release_guards(applied_events))
          delete across_aggregator;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyAcrossAnalysis::perform_output(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_output(
            perform_precondition, trace_info, true /*track*/, applied_events);
      if (!copy_events.empty())
        return Runtime::merge_events(&trace_info, copy_events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteCopyAcrossAnalysis::handle(
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
      FieldMask src_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        src_mask |= eq_masks[idx];
      }
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned src_index, dst_index;
      derez.deserialize(src_index);
      derez.deserialize(dst_index);
      RegionUsage src_usage, dst_usage;
      derez.deserialize(src_usage);
      derez.deserialize(dst_usage);
      ApEvent dst_ready;
      derez.deserialize(dst_ready);
      size_t num_dsts;
      derez.deserialize(num_dsts);
      std::vector<PhysicalManager*> dst_instances(num_dsts);
      op::vector<op::FieldMaskMap<InstanceView>> dst_views(num_dsts);
      for (unsigned idx1 = 0; idx1 < num_dsts; idx1++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        dst_instances[idx1] =
            runtime->find_or_request_instance_manager(did, ready);
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
          dst_views[idx1].insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_srcs;
      derez.deserialize(num_srcs);
      std::vector<IndividualView*> src_views(num_srcs, nullptr);
      for (unsigned idx = 0; idx < num_srcs; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        src_views[idx] = static_cast<IndividualView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      LogicalRegion src_handle, dst_handle;
      derez.deserialize(src_handle);
      derez.deserialize(dst_handle);
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      ReductionOpID redop;
      derez.deserialize(redop);
      bool perfect;
      derez.deserialize(perfect);
      std::vector<unsigned> src_indexes, dst_indexes;
      if (!perfect)
      {
        size_t num_indexes;
        derez.deserialize(num_indexes);
        src_indexes.resize(num_indexes);
        dst_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          derez.deserialize(src_indexes[idx]);
          derez.deserialize(dst_indexes[idx]);
        }
      }
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent copy;
      derez.deserialize(copy);
      std::set<RtEvent> deferral_events, applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);

      RegionNode* dst_node = runtime->get_node(dst_handle);
      IndexSpaceExpression* dst_expr = dst_node->row_source;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      // If we're not perfect we need to wait on the ready event here
      // because we need the dst_instances to be valid to construct
      // the copy-across helpers
      if (!perfect && ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      // This takes ownership of the op and the across helpers
      CopyAcrossAnalysis* analysis = new CopyAcrossAnalysis(
          original_source, previous, op, src_index, dst_index, src_usage,
          dst_usage, src_handle, dst_handle, dst_ready,
          std::move(dst_instances), std::move(dst_views), std::move(src_views),
          precondition, pred_guard, redop, src_indexes, dst_indexes, trace_info,
          perfect);
      analysis->add_reference();
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->analyze(
            eq_sets[idx], eq_masks[idx], deferral_events, applied_events,
            ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      // Start with the source mask here in case we need to filter which
      // is all done on the source fields
      analysis->local_exprs.insert(dst_expr, src_mask);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = analysis->perform_remote(traversal_done, applied_events);
      RtEvent updates_ready;
      // Chain these so we get the local_exprs set correct
      if (remote_ready.exists() || analysis->has_across_updates())
        updates_ready = analysis->perform_updates(remote_ready, applied_events);
      const ApEvent result =
          analysis->perform_output(updates_ready, applied_events);
      Runtime::trigger_event(copy, result, trace_info, applied_events);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      // Clean up our analysis
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    /*static*/ CopyAcrossHelper* CopyAcrossAnalysis::create_across_helper(
        const FieldMask& src_mask, const FieldMask& dst_mask,
        const std::vector<PhysicalManager*>& dst_instances,
        const op::vector<op::FieldMaskMap<InstanceView>>& dst_views,
        const std::vector<unsigned>& src_indexes,
        const std::vector<unsigned>& dst_indexes)
    //--------------------------------------------------------------------------
    {
      legion_assert(!dst_instances.empty());
      CopyAcrossHelper* result =
          new CopyAcrossHelper(src_mask, src_indexes, dst_indexes);
      for (unsigned idx = 0; idx < dst_instances.size(); idx++)
        dst_instances[idx]->initialize_across_helper(
            result, dst_views[idx].get_valid_mask(), src_indexes, dst_indexes);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::vector<PhysicalManager*>
        CopyAcrossAnalysis::convert_instances(const InstanceSet& instances)
    //--------------------------------------------------------------------------
    {
      std::vector<PhysicalManager*> result(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
        result[idx] = instances[idx].get_physical_manager();
      return result;
    }

  }  // namespace Internal
}  // namespace Legion
