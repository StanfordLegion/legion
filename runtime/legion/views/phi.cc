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

#include "legion/views/phi.h"
#include "legion/kernel/runtime.h"
#include "legion/utilities/serdez.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // PhiView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhiView::PhiView(
        DistributedID did, PredEvent tguard, PredEvent fguard,
        shrt::FieldMaskMap<DeferredView>&& true_vws,
        shrt::FieldMaskMap<DeferredView>&& false_vws, bool register_now)
      : DeferredView(encode_phi_did(did), register_now), true_guard(tguard),
        false_guard(fguard), true_views(true_vws), false_views(false_vws)
    //--------------------------------------------------------------------------
    {
      legion_assert(true_guard.exists());
      legion_assert(false_guard.exists());
      legion_assert(
          true_views.get_valid_mask() == false_views.get_valid_mask());
      if (register_now)
      {
        shrt::map<DeferredView*, LamportClock> empty_clocks;
        add_initial_references(empty_clocks);
      }
#ifdef LEGION_GC
      log_garbage.info(
          "GC Phi View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    PhiView::~PhiView(void)
    //--------------------------------------------------------------------------
    {
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_local(void)
    //--------------------------------------------------------------------------
    {
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
        it->first->remove_nested_gc_ref(did);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
        it->first->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void PhiView::pack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      if (view_lamport_clocks.find(this) != view_lamport_clocks.end())
        return;
      pack_global_ref(view_lamport_clocks[this]);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
        it->first->pack_valid_ref(view_lamport_clocks, inst_lamport_clocks);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
        it->first->pack_valid_ref(view_lamport_clocks, inst_lamport_clocks);
    }

    //--------------------------------------------------------------------------
    void PhiView::unpack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      shrt::map<LogicalView*, LamportClock>::iterator finder =
          view_lamport_clocks.find(this);
      if (finder == view_lamport_clocks.end())
        return;
      unpack_global_ref(finder->second);
      view_lamport_clocks.erase(finder);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
        it->first->unpack_valid_ref(view_lamport_clocks, inst_lamport_clocks);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
        it->first->unpack_valid_ref(view_lamport_clocks, inst_lamport_clocks);
    }

    //--------------------------------------------------------------------------
    void PhiView::add_initial_references(
        shrt::map<DeferredView*, LamportClock>& lamport_clocks)
    //--------------------------------------------------------------------------
    {
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
      {
        it->first->add_nested_resource_ref(did);
        it->first->add_nested_gc_ref(did);
        shrt::map<DeferredView*, LamportClock>::iterator finder =
            lamport_clocks.find(it->first);
        if (finder != lamport_clocks.end())
        {
          it->first->unpack_global_ref(finder->second);
          lamport_clocks.erase(finder);
        }
      }
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
      {
        it->first->add_nested_resource_ref(did);
        it->first->add_nested_gc_ref(did);
        shrt::map<DeferredView*, LamportClock>::iterator finder =
            lamport_clocks.find(it->first);
        if (finder != lamport_clocks.end())
        {
          it->first->unpack_global_ref(finder->second);
          lamport_clocks.erase(finder);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      legion_assert(collective_mapping == nullptr);
      PhiViewMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(true_guard);
        rez.serialize(false_guard);
        rez.serialize<size_t>(true_views.size());
        for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
                 true_views.begin();
             it != true_views.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          it->first->pack_global_ref(rez);
        }
        rez.serialize<size_t>(false_views.size());
        for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
                 false_views.begin();
             it != false_views.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          // Need to deduplicate
          if (true_views.find(it->first) == true_views.end())
            it->first->pack_global_ref(rez);
        }
      }
      rez.dispatch(target);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void PhiView::flatten(
        CopyFillAggregator& aggregator, InstanceView* dst_view,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* helper)
    //--------------------------------------------------------------------------
    {
      legion_assert(!(src_mask - true_views.get_valid_mask()));
      legion_assert(!(src_mask - false_views.get_valid_mask()));
      const PredEvent next_true =
          !pred_guard.exists() ?
              true_guard :
              Runtime::merge_events(&trace_info, pred_guard, true_guard);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               true_views.begin();
           it != true_views.end(); it++)
      {
        const FieldMask overlap = src_mask & it->second;
        if (!overlap)
          continue;
        it->first->flatten(
            aggregator, dst_view, overlap, expr, next_true, trace_info,
            tracing_eq, helper);
      }
      const PredEvent next_false =
          !pred_guard.exists() ?
              false_guard :
              Runtime::merge_events(&trace_info, pred_guard, false_guard);
      for (shrt::FieldMaskMap<DeferredView>::const_iterator it =
               false_views.begin();
           it != false_views.end(); it++)
      {
        const FieldMask overlap = src_mask & it->second;
        if (!overlap)
          continue;
        it->first->flatten(
            aggregator, dst_view, overlap, expr, next_false, trace_info,
            tracing_eq, helper);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiViewMessage::handle(Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PredEvent true_guard, false_guard;
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      std::set<RtEvent> ready_events;
      shrt::FieldMaskMap<DeferredView> true_views, false_views;
      size_t num_true_views;
      derez.deserialize(num_true_views);
      shrt::map<DeferredView*, LamportClock> lamport_clocks;
      for (unsigned idx = 0; idx < num_true_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        DeferredView* view = static_cast<DeferredView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        FieldMask mask;
        derez.deserialize(mask);
        true_views.insert(view, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
        lamport_clocks[view] = DeferredView::unpack_global_ref_clock(derez);
      }
      size_t num_false_views;
      derez.deserialize(num_false_views);
      for (unsigned idx = 0; idx < num_false_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        DeferredView* view = static_cast<DeferredView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        FieldMask mask;
        derez.deserialize(mask);
        false_views.insert(view, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
        if (lamport_clocks.find(view) == lamport_clocks.end())
          lamport_clocks[view] = DeferredView::unpack_global_ref_clock(derez);
      }
      // Make the phi view but don't register it yet
      void* location =
          runtime->find_or_create_pending_collectable_location<PhiView>(did);
      PhiView* view = new (location) PhiView(
          did, true_guard, false_guard, std::move(true_views),
          std::move(false_views), false /*register_now*/);
      if (!ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(ready_events);
        PhiView::DeferPhiViewRegistrationArgs args(view, lamport_clocks);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, wait_on);
      }
      else
      {
        // Add the resource references
        view->add_initial_references(lamport_clocks);
        view->register_with_runtime();
        legion_assert(lamport_clocks.empty());
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::DeferPhiViewRegistrationArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      view->add_initial_references(*lamport_clocks);
      view->register_with_runtime();
      legion_assert(lamport_clocks->empty());
      delete lamport_clocks;
    }

  }  // namespace Internal
}  // namespace Legion
