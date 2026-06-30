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

#include "legion/views/collective.h"
#include "legion/analysis/collective.h"
#include "legion/contexts/inner.h"
#include "legion/instances/physical.h"
#include "legion/nodes/index.h"
#include "legion/operations/remote.h"
#include "legion/views/allreduce.h"
#include "legion/views/fill.h"
#include "legion/views/individual.h"
#include "legion/views/reduction.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // CollectiveView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveView::CollectiveView(
        DistributedID id, DistributedID ctx_did,
        const std::vector<IndividualView*>& views,
        const std::vector<DistributedID>& insts, bool register_now,
        CollectiveMapping* mapping)
      : InstanceView(id, register_now, mapping), context_did(ctx_did),
        instances(insts), local_views(views), valid_state(NOT_VALID_STATE),
        collect_lamport_clock(0), pending_collect_lamport_clock(0),
        bump_collect_lamport_clock(false), sent_valid_references(0),
        received_valid_references(0), deletion_notified(false),
        multiple_local_memories(has_multiple_local_memories(local_views))
    //--------------------------------------------------------------------------
    {
      for (IndividualView* view : local_views)
      {
        // For collective instances we always want the logical analysis
        // node for the view to be on the same node as the owner for actual
        // physical instance to aid in our ability to do the analysis
        // See the get_analysis_space function for why we check this
        legion_assert(view->logical_owner == view->get_manager()->owner_space);
        //(*it)->add_nested_resource_ref(did);
        view->add_nested_gc_ref(did);
        // Record ourselves with each of our local views so they can
        // notify us when they are deleted
        PhysicalManager* manager = view->get_manager();
        legion_no_skip_assert(manager->register_deletion_subscriber(this));
      }
    }

    //--------------------------------------------------------------------------
    CollectiveView::~CollectiveView(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    /*static*/ bool CollectiveView::has_multiple_local_memories(
        const std::vector<IndividualView*>& local_views)
    //--------------------------------------------------------------------------
    {
      if (local_views.size() < 2)
        return false;
      Memory first = local_views.front()->get_manager()->memory_manager->memory;
      for (unsigned idx = 1; idx < local_views.size(); idx++)
        if (local_views[idx]->get_manager()->memory_manager->memory != first)
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!deletion_notified.exchange(true))
      {
        // Unregister ourselves with all our local instances
        for (IndividualView* view : local_views)
        {
          PhysicalManager* manager = view->get_manager();
          manager->unregister_deletion_subscriber(this);
        }
      }
      for (IndividualView* view : local_views)
        if (view->remove_nested_gc_ref(did))
          delete view;
      for (const std::pair<PhysicalManager* const, IndividualView*>& it :
           remote_instances)
        if (it.second->remove_nested_gc_ref(did))
          delete it.second;
      remote_instances.clear();
    }

    //--------------------------------------------------------------------------
    void CollectiveView::pack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      // Deduplicate so that a view packed more than once in a single message
      // only contributes a single sent valid reference and a single stamp
      if (view_lamport_clocks.find(this) != view_lamport_clocks.end())
        return;
      AutoLock v_lock(view_lock);
      legion_assert(valid_state == FULL_VALID_STATE);
      // If we've committed our counts to an invalidation round then the first
      // pack afterwards must advance the clock so that a reference packed after
      // the commit is detectable as a causality violation when it is unpacked
      if (bump_collect_lamport_clock)
      {
        collect_lamport_clock++;
        bump_collect_lamport_clock = false;
      }
      view_lamport_clocks[this] = collect_lamport_clock;
      sent_valid_references++;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::unpack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      shrt::map<LogicalView*, LamportClock>::iterator finder =
          view_lamport_clocks.find(this);
      if (finder == view_lamport_clocks.end())
        return;
      {
        AutoLock v_lock(view_lock);
        legion_assert(valid_state == FULL_VALID_STATE);
        received_valid_references++;
        collect_lamport_clock = std::max(collect_lamport_clock, finder->second);
      }
      view_lamport_clocks.erase(finder);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        legion_assert(
            (valid_state == NOT_VALID_STATE) ||
            (valid_state == PENDING_INVALID_STATE));
        // If we're not in a pending invalid state then send out the
        // notifications to all the nodes in the collective that we are
        // now valid they should hold valid references on all their local views
        if (valid_state != PENDING_INVALID_STATE)
          make_valid(false /*need lock*/);
        else  // We can promote ourselves up to fully valid again
          valid_state = FULL_VALID_STATE;
      }
      else
      {
        if (valid_state == NOT_VALID_STATE)
          make_valid(false /*need lock*/);
        else  // restore ourselves back to full valid state
          valid_state = FULL_VALID_STATE;
        // Not the owner so need to send a message on down the chain
        // to make the owner valid and ensure all the nodes are keeping
        // a valid reference
        CollectiveViewAddRemoteReference rez;
        rez.serialize(did);
        if ((collective_mapping != nullptr) &&
            collective_mapping->contains(local_space))
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        else
          rez.dispatch(owner_space);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveView::make_valid(bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock v_lock(view_lock);
        make_valid(false /*need lock*/);
      }
      else
      {
        // If we're already fully valid then there is nothing more to do
        if (valid_state == FULL_VALID_STATE)
          return;
        // Send the messages to the children to get them in flight
        if ((collective_mapping != nullptr) &&
            collective_mapping->contains(local_space))
        {
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          if (!children.empty())
          {
            CollectiveViewMakeValid rez;
            rez.serialize(did);
            for (const AddressSpaceID& child : children) rez.dispatch(child);
          }
        }
        // Only need to add the references if we're in the not-valid state
        // as if we are in the pending invalid state then we haven't actually
        // remove the reference yet
        if (valid_state == NOT_VALID_STATE)
        {
          for (IndividualView* view : local_views)
            view->add_nested_valid_ref(did);
          add_base_gc_ref(INTERNAL_VALID_REF);
        }
        valid_state = FULL_VALID_STATE;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewMakeValid::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);

      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      view->make_valid(true /*need lock*/);
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::make_invalid(bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock v_lock(view_lock);
        return make_invalid(false /*need lock*/);
      }
      else
      {
        legion_assert(
            (valid_state == FULL_VALID_STATE) ||
            (valid_state == PENDING_INVALID_STATE));
        legion_assert(is_owner() || collective_mapping->contains(local_space));
        if (valid_state == FULL_VALID_STATE)
        {
          // This is a potential race with adding a valid reference for
          // mapping and a previous invalidation. These races should be
          // mostly benign so we'll igonore them for now but we might
          // need to do something about them in the future
          return false;
        }
        // Send it upstream to any children
        if (collective_mapping != nullptr)
        {
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          if (!children.empty())
          {
            CollectiveViewMakeInvalid rez;
            rez.serialize(did);
            for (const AddressSpaceID& child : children) rez.dispatch(child);
          }
        }
        valid_state = NOT_VALID_STATE;
        for (IndividualView* view : local_views)
          view->remove_nested_valid_ref(did);
        return remove_base_gc_ref(INTERNAL_VALID_REF);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewMakeInvalid::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);

      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      if (view->make_invalid(true /*need lock*/))
        delete view;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::perform_invalidate_request(
        LamportClock snapshot, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock v_lock(view_lock);
        return perform_invalidate_request(snapshot, false /*need lock*/);
      }
      else
      {
        legion_assert((pending_collect_lamport_clock < snapshot) || is_owner());
        // See if we're going to fail right away
        if (valid_state == FULL_VALID_STATE)
        {
          CollectiveViewInvalidateResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(snapshot);
            rez.serialize<bool>(true /*fail*/);
            // Carry our clock even on failure so the owner stays monotonic
            rez.serialize(collect_lamport_clock);
          }
          if ((collective_mapping != nullptr) &&
              collective_mapping->contains(local_space))
            rez.dispatch(
                collective_mapping->get_parent(owner_space, local_space));
          else
            rez.dispatch(owner_space);
          return false;
        }
        invalidation_failed = false;
        // Commit to this invalidation round: record the round's snapshot, fold
        // it into our clock, and arm the bump so that the first valid reference
        // packed after this point advances our clock past the snapshot and is
        // detectable as a causality violation.
        pending_collect_lamport_clock = snapshot;
        collect_lamport_clock = std::max(collect_lamport_clock, snapshot);
        bump_collect_lamport_clock = true;
        total_valid_sent = 0;
        total_valid_received = 0;
        remaining_invalidation_responses = 1;
        // Send out messages to all our copies to check if there are still
        // valid references anywhere that we need to be aware of
        if ((collective_mapping != nullptr) &&
            collective_mapping->contains(local_space))
        {
          std::vector<AddressSpaceID> children;
          collective_mapping->get_children(owner_space, local_space, children);
          if (!children.empty())
          {
            CollectiveViewInvalidateRequest rez;
            rez.serialize(did);
            rez.serialize(pending_collect_lamport_clock);
            for (const AddressSpaceID& child : children) rez.dispatch(child);
            remaining_invalidation_responses += children.size();
          }
        }
        if (is_owner() && has_remote_instances())
        {
          CollectiveViewInvalidateRequest rez;
          rez.serialize(did);
          rez.serialize(pending_collect_lamport_clock);
          struct InvalidFunctor {
            InvalidFunctor(CollectiveViewInvalidateRequest& z, unsigned& cnt)
              : rez(z), count(cnt)
            { }
            inline void apply(AddressSpaceID target)
            {
              if (target == runtime->address_space)
                return;
              rez.dispatch(target);
              count++;
            }
            CollectiveViewInvalidateRequest& rez;
            unsigned& count;
          };
          InvalidFunctor functor(rez, remaining_invalidation_responses);
          map_over_remote_instances(functor);
        }
        // Now we can perform our local arrival
        return perform_invalidate_response(
            snapshot, sent_valid_references, received_valid_references,
            collect_lamport_clock, false /*fail*/, false /*need lock*/);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewInvalidateRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      LamportClock snapshot;
      derez.deserialize(snapshot);

      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      if (view->perform_invalidate_request(snapshot, true /*need lock*/))
        delete view;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::perform_invalidate_response(
        LamportClock snapshot, uint64_t total_sent, uint64_t total_received,
        LamportClock clock, bool failed, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock v_lock(view_lock);
        return perform_invalidate_response(
            snapshot, total_sent, total_received, clock, failed,
            false /*need lock*/);
      }
      else
      {
        // If this response is stale then we don't need to do anything
        if (snapshot < pending_collect_lamport_clock)
          return false;
        // Always merge the responder's clock: this lets us detect a valid
        // reference packed newer than this round's snapshot, and keeps our
        // clock monotonic so the next round's snapshot exceeds any reference
        // that caused this round to fail (guaranteeing forward progress).
        collect_lamport_clock = std::max(collect_lamport_clock, clock);
        if (!failed)
        {
          total_valid_sent += total_sent;
          total_valid_received += total_received;
        }
        else
          invalidation_failed = true;
        legion_assert(remaining_invalidation_responses > 0);
        if (--remaining_invalidation_responses == 0)
        {
          // Check that we are still not valid
          if (!invalidation_failed && (valid_state == FULL_VALID_STATE))
            invalidation_failed = true;
          if (!is_owner())
          {
            // Send the response back to the owner
            CollectiveViewInvalidateResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(snapshot);
              rez.serialize(invalidation_failed);
              // Carry our clock unconditionally so the owner can both detect
              // post-snapshot packs and stay monotonic across rounds
              rez.serialize(collect_lamport_clock);
              if (!invalidation_failed)
              {
                rez.serialize(total_valid_sent);
                rez.serialize(total_valid_received);
              }
            }
            if ((collective_mapping != nullptr) &&
                collective_mapping->contains(local_space))
              rez.dispatch(
                  collective_mapping->get_parent(owner_space, local_space));
            else
              rez.dispatch(owner_space);
          }
          // Only invalidate if no node re-validated, the packed and unpacked
          // valid reference counts balance (no reference in flight), and no
          // node observed a valid reference packed after this round's snapshot
          // (a causality violation that could otherwise hide behind aliased
          // counts).
          else if (
              !invalidation_failed &&
              (total_valid_sent == total_valid_received) &&
              (collect_lamport_clock <= pending_collect_lamport_clock))
            return make_invalid(false /*need lock*/);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewInvalidateResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      LamportClock snapshot, clock;
      uint64_t total_sent = 0, total_received = 0;
      derez.deserialize(snapshot);
      bool failed;
      derez.deserialize(failed);
      derez.deserialize(clock);
      if (!failed)
      {
        derez.deserialize(total_sent);
        derez.deserialize(total_received);
      }
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      if (view->perform_invalidate_response(
              snapshot, total_sent, total_received, clock, failed,
              true /*need lock*/))
        delete view;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewAddRemoteReference::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);

      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      view->add_base_valid_ref(REMOTE_DID_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewRemoveRemoteReference::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);

      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      if (view->remove_base_valid_ref(REMOTE_DID_REF))
        delete view;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      valid_state = PENDING_INVALID_STATE;
      if (is_owner())
      {
        // We're the owner so we need to see if it is safe to actually
        // downgrade the valid state of this collective view which means
        // checking that none of our copies are valid on any node
        // Start a new round using a fresh, strictly-increasing Lamport-clock
        // snapshot (this replaces the old monotonic generation counter)
        return perform_invalidate_request(
            ++collect_lamport_clock, false /*need lock*/);
      }
      else
      {
        // Not the owner so send a message down the chain to remove the
        // valid reference that we added when we became valid
        CollectiveViewRemoveRemoteReference rez;
        rez.serialize(did);
        if ((collective_mapping != nullptr) &&
            collective_mapping->contains(local_space))
          rez.dispatch(
              collective_mapping->get_parent(owner_space, local_space));
        else
          rez.dispatch(owner_space);
        // Nodes which aren't part of the collective won't be getting a
        // make_invalid call so they can remove their reference now
        if ((collective_mapping == nullptr) ||
            !collective_mapping->contains(local_space))
        {
          valid_state = NOT_VALID_STATE;
          return remove_base_gc_ref(INTERNAL_VALID_REF);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveView::get_analysis_space(
        PhysicalManager* instance) const
    //--------------------------------------------------------------------------
    {
      return instance->owner_space;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::aliases(InstanceView* other) const
    //--------------------------------------------------------------------------
    {
      if (other == this)
        return true;
      if (other->is_individual_view())
      {
        IndividualView* individual = other->as_individual_view();
        return std::binary_search(
            instances.begin(), instances.end(), individual->get_manager()->did);
      }
      else
      {
        CollectiveView* collective = other->as_collective_view();
        if (instances.size() < collective->instances.size())
        {
          for (const DistributedID& it : instances)
            if (std::binary_search(
                    collective->instances.begin(), collective->instances.end(),
                    it))
              return true;
          return false;
        }
        else
        {
          for (const DistributedID& it : collective->instances)
            if (std::binary_search(instances.begin(), instances.end(), it))
              return true;
          return false;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveView::notify_instance_deletion(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      notify_instance_deletion(manager->tree_id);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::add_subscriber_reference(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      add_nested_resource_ref(manager->did);
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::remove_subscriber_reference(PhysicalManager* manager)
    //--------------------------------------------------------------------------
    {
      return remove_nested_resource_ref(manager->did);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::notify_instance_deletion(RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      // Check to see if we're the first deletion arrival
      if (deletion_notified.exchange(true))
        return;
      if (is_owner())
      {
        // Notify the context that this can be deleted
        // See if the context is local or not
        const AddressSpaceID context_space =
            runtime->determine_owner(context_did);
        if (context_space != local_space)
        {
          CollectiveViewDeletion rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(tid);
            rez.serialize(context_did);
          }
          rez.dispatch(context_space);
        }
        else
        {
          InnerContext* context = static_cast<InnerContext*>(
              runtime->weak_find_distributed_collectable(context_did));
          if (context != nullptr)
          {
            context->notify_collective_deletion(tid, did);
            if (context->remove_base_resource_ref(RUNTIME_REF))
              delete context;
          }
        }
      }
      else
      {
        legion_assert(collective_mapping != nullptr);
        legion_assert(collective_mapping->contains(local_space));
        // Send the notification down to the parent
        CollectiveViewNotification rez;
        rez.serialize(did);
        rez.serialize(tid);
        rez.dispatch(collective_mapping->get_parent(owner_space, local_space));
      }
      // Unregister ourselves with all our local instances
      if (!deletion_notified.exchange(true))
      {
        for (IndividualView* view : local_views)
        {
          PhysicalManager* manager = view->get_manager();
          manager->unregister_deletion_subscriber(this);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveViewNotification::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      RegionTreeID tid;
      derez.deserialize(tid);
      DistributedCollectable* dc =
          runtime->weak_find_distributed_collectable(did);
      if (dc != nullptr)
      {
        CollectiveView* view = static_cast<CollectiveView*>(dc);
        view->notify_instance_deletion(tid);
        if (view->remove_base_resource_ref(RUNTIME_REF))
          delete view;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::fill_from(
        FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* fill_expression, Operation* op,
        const unsigned index, const IndexSpace collective_match_space,
        const FieldMask& fill_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool fill_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
      // Should never have a copy-across with a collective manager as the target
      legion_assert(manage_dst_events);
      legion_assert(across_helper == nullptr);
      legion_assert(collective_mapping != nullptr);
      // This one is easy, just tree broadcast out to all the nodes and
      // perform the fill operation on each one of them
      ApEvent result;
      if (need_valid_return)
        result = Runtime::create_ap_user_event(&trace_info);
      if (!collective_mapping->contains(local_space))
      {
        // This node doesn't have any instances, so start at one that
        // is contained within the collective mapping
        AddressSpaceID origin = collective_mapping->find_nearest(local_space);
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeFill rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(fill_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          fill_expression->pack_expression(rez, origin);
          rez.serialize<bool>(fill_restricted);
          if (fill_restricted)
            op->pack_remote_operation(rez, origin, applied_events);
          rez.serialize(index);
          rez.serialize(collective_match_space);
          rez.serialize(op->get_context_index());
          rez.serialize(fill_mask);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            ApBarrier bar;
            ShardID sid = 0;
            if (need_valid_return)
            {
              sid = trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
              result = bar;
            }
            rez.serialize(bar);
            if (bar.exists())
              rez.serialize(sid);
          }
          else
          {
            ApUserEvent to_trigger;
            if (need_valid_return)
            {
              to_trigger = Runtime::create_ap_user_event(&trace_info);
              result = to_trigger;
            }
            rez.serialize(to_trigger);
          }
          rez.serialize(origin);
        }
        rez.dispatch(origin);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      else
      {
        ApUserEvent to_trigger;
        if (need_valid_return)
        {
          to_trigger = Runtime::create_ap_user_event(&trace_info);
          result = to_trigger;
        }
        perform_collective_fill(
            fill_view, precondition, predicate_guard, fill_expression, op,
            index, collective_match_space, op->get_context_index(), fill_mask,
            trace_info, recorded_events, applied_events, to_trigger,
            local_space, fill_restricted);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::copy_from(
        InstanceView* src_view, ApEvent precondition, PredEvent predicate_guard,
        ReductionOpID reduction_op_id, IndexSpaceExpression* copy_expression,
        Operation* op, const unsigned index,
        const IndexSpace collective_match_space, const FieldMask& copy_mask,
        PhysicalManager* src_point, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        CopyAcrossHelper* across_helper, const bool manage_dst_events,
        const bool copy_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
      // Should never have a copy-across with a collective manager as the target
      legion_assert(manage_dst_events);
      legion_assert(across_helper == nullptr);
      legion_assert(collective_mapping != nullptr);
      legion_assert(reduction_op_id == src_view->get_redop());
      // Several cases here:
      // 1. The source is a normal individual manager - in this case we'll issue
      //    the copy/reduction from the source to an instance on the closest
      //    node and then build the broadcast tree from there
      // 2. The source is another normal collective manager - we'll do a
      //    broadcast out to all the nodes and have each of them pick a
      //    source instance to copy from and then do the copy
      // 3. The source is a reduction collective instance with the same
      //      collective mapping as the destination - broadcast control
      //    out to all the nodes and then perform the all-reduce between the
      //    instances of the source, then do the reduction the same as the
      //    case for copies with a normal collective manager
      // 4. The source is a reduction manager that is either an individual
      //      instance or a collective instance with a different mapping
      //      than the destination - Build a reduction tree down to a
      //    single instance if necessary and then broadcast out the
      //    reduction data to all the other instances
      ApUserEvent all_done;
      if (need_valid_return)
        all_done = Runtime::create_ap_user_event(&trace_info);
      if (!src_view->is_collective_view())
      {
        // Case 1: the source is an individual manager
        // Copy to one of our instances and then broadcast it
        IndividualView* source_view = src_view->as_individual_view();
        const UniqueID op_id = op->get_unique_op_id();
        // Get the precondition as well
        const ApEvent src_pre = source_view->find_copy_preconditions(
            true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&trace_info, precondition, src_pre);
          else
            precondition = src_pre;
        }
        PhysicalManager* source_manager = source_view->get_manager();
        std::vector<CopySrcDstField> src_fields;
        source_manager->compute_copy_offsets(copy_mask, src_fields);
        // We have to follow the tree for other kinds of operations here
        const AddressSpaceID origin = select_origin_space();
        ApUserEvent copy_done = Runtime::create_ap_user_event(&trace_info);
        // Record the copy done event on the source view
        source_view->add_copy_user(
            true /*reading*/, 0 /*redop*/, copy_done, copy_mask,
            copy_expression, collective_match_space, op_id, index,
            recorded_events, trace_info.recording, runtime->address_space);
        ApBarrier all_bar;
        ShardID owner_shard = 0;
        if (trace_info.recording &&
            (all_done.exists() || (source_view->get_redop() > 0)))
        {
          const size_t arrivals = collective_mapping->size();
          owner_shard = trace_info.record_barrier_creation(all_bar, arrivals);
          // Tracing copy-optimization will eliminate this when
          // the trace gets optimized
          if (all_done.exists())
            Runtime::trigger_event(
                all_done, all_bar, trace_info, applied_events);
          if (source_view->get_redop() > 0)
          {
            Runtime::trigger_event(
                copy_done, all_bar, trace_info, applied_events);
            copy_done = ApUserEvent::NO_AP_USER_EVENT;
          }
        }
        const UniqueInst src_inst(source_view);
        if (origin != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          if (reduction_op_id == 0)
          {
            CollectiveDistributeBroadcast rez;
            {
              RezCheck z(rez);
              rez.serialize(this->did);
              source_view->pack_fields(rez, src_fields);
              src_inst.serialize(rez);
              rez.serialize(precondition);
              rez.serialize(predicate_guard);
              copy_expression->pack_expression(rez, origin);
              rez.serialize<bool>(copy_restricted);
              if (copy_restricted)
                op->pack_remote_operation(rez, origin, applied_events);
              rez.serialize(index);
              rez.serialize(collective_match_space);
              rez.serialize(op->get_context_index());
              rez.serialize(copy_mask);
              trace_info.pack_trace_info(rez);
              rez.serialize(recorded);
              rez.serialize(applied);
              if (trace_info.recording)
              {
                // If this is a reducecast case, then the barrier is for
                // all of the different reductions
                if (source_view->get_redop() == 0)
                {
                  ApBarrier copy_bar;
                  ShardID sid = trace_info.record_barrier_creation(
                      copy_bar, 1 /*arrivals*/);
                  Runtime::trigger_event(
                      copy_done, copy_bar, trace_info, applied_events);
                  rez.serialize(copy_bar);
                  rez.serialize(sid);
                }
                rez.serialize(all_bar);
                if (all_bar.exists())
                  rez.serialize(owner_shard);
              }
              else
              {
                rez.serialize(copy_done);
                if (source_view->get_redop() == 0)
                  rez.serialize(all_done);
              }
              rez.serialize(origin);
              rez.serialize(COLLECTIVE_BROADCAST);
            }
            rez.dispatch(origin);
          }
          else
          {
            CollectiveDistributeReducecast rez;
            {
              RezCheck z(rez);
              rez.serialize(this->did);
              rez.serialize(source_view->did);
              source_view->pack_fields(rez, src_fields);
              src_inst.serialize(rez);
              rez.serialize(precondition);
              rez.serialize(predicate_guard);
              copy_expression->pack_expression(rez, origin);
              rez.serialize<bool>(copy_restricted);
              if (copy_restricted)
                op->pack_remote_operation(rez, origin, applied_events);
              rez.serialize(index);
              rez.serialize(collective_match_space);
              rez.serialize(op->get_context_index());
              rez.serialize(copy_mask);
              trace_info.pack_trace_info(rez);
              rez.serialize(recorded);
              rez.serialize(applied);
              if (trace_info.recording)
              {
                // If this is a reducecast case, then the barrier is for
                // all of the different reductions
                if (source_view->get_redop() == 0)
                {
                  ApBarrier copy_bar;
                  ShardID sid = trace_info.record_barrier_creation(
                      copy_bar, 1 /*arrivals*/);
                  Runtime::trigger_event(
                      copy_done, copy_bar, trace_info, applied_events);
                  rez.serialize(copy_bar);
                  rez.serialize(sid);
                }
                rez.serialize(all_bar);
                if (all_bar.exists())
                  rez.serialize(owner_shard);
              }
              else
              {
                rez.serialize(copy_done);
                if (source_view->get_redop() == 0)
                  rez.serialize(all_done);
              }
              rez.serialize(origin);
            }
            rez.dispatch(origin);
          }
          recorded_events.insert(recorded);
          applied_events.insert(applied);
        }
        else
        {
          if (reduction_op_id > 0)
            perform_collective_reducecast(
                source_view->as_reduction_view(), src_fields, precondition,
                predicate_guard, copy_expression, op, index,
                collective_match_space, op->get_context_index(), copy_mask,
                src_inst, source_manager->get_unique_event(), trace_info,
                recorded_events, applied_events, copy_done, all_bar,
                owner_shard, origin, copy_restricted);
          else
            perform_collective_broadcast(
                src_fields, precondition, predicate_guard, copy_expression, op,
                index, collective_match_space, op->get_context_index(),
                copy_mask, src_inst, source_manager->get_unique_event(),
                trace_info, recorded_events, applied_events, copy_done,
                all_done, all_bar, owner_shard, origin, copy_restricted,
                COLLECTIVE_BROADCAST);
        }
      }
      else
      {
        CollectiveView* collective = src_view->as_collective_view();
        const AddressSpaceID origin = select_origin_space();
        // If the source is a reduction collective instance then we need
        // to see if we can go down the point-wise route based on performing
        // an all-reduce, or whether we have to do a tree reduction followed
        // by a tree broadcast. To do the all-reduce path we need all the
        // collective mappings for both collective instances to be the same
        uint64_t allreduce_tag = 0;
        if (collective->is_allreduce_view())
        {
          // Case 3: this is conceptually an all-reduce
          // We'll handle two separate cases here depending on whether
          // the two collective instances have matching collective mappings
          if ((collective_mapping != collective->collective_mapping) &&
              (*collective_mapping != *(collective->collective_mapping)))
          {
            // The two collective mappings do not align, which should
            // be fairly uncommon, but we'll handle it anyway
            // In this case we'll do a reduction down to a single
            // instance in the source collective manager and then
            // broadcast back out to all the destination instances
            // For correctness, the reduce cast must start whereever
            // a comparable broadcast or fill would have started
            // on the destination collective instance
            perform_collective_hourglass(
                collective->as_allreduce_view(), precondition, predicate_guard,
                copy_expression, op, index, collective_match_space, copy_mask,
                (src_point != nullptr) ? src_point->did : 0, trace_info,
                recorded_events, applied_events, all_done, origin,
                copy_restricted);
            return all_done;
          }
          // Otherwise we can fall through and do the allreduce as part
          // of the pointwise copy, get a tag through for unique identification
          if (origin == local_space)
          {
            AllreduceView* allreduce = collective->as_allreduce_view();
            allreduce_tag = allreduce->generate_unique_allreduce_tag();
          }
        }
        ApBarrier all_bar;
        ShardID owner_shard;
        if (all_done.exists() && trace_info.recording)
        {
          const size_t arrivals = collective_mapping->size();
          owner_shard = trace_info.record_barrier_creation(all_bar, arrivals);
          // Tracing copy-optimization will eliminate this when
          // the trace gets optimized
          Runtime::trigger_event(all_done, all_bar, trace_info, applied_events);
        }
        // Case 2 and 3 (all-reduce): Broadcast out the point-wise command
        if (origin != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          CollectiveDistributePointwise rez;
          {
            RezCheck z(rez);
            rez.serialize(this->did);
            rez.serialize(collective->did);
            rez.serialize(precondition);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, origin);
            rez.serialize<bool>(copy_restricted);
            if (copy_restricted)
              op->pack_remote_operation(rez, origin, applied_events);
            rez.serialize(index);
            rez.serialize(collective_match_space);
            rez.serialize(op->get_context_index());
            rez.serialize(copy_mask);
            if (src_point != nullptr)
              rez.serialize(src_point->did);
            else
              rez.serialize<DistributedID>(0);
            rez.serialize(op->get_unique_op_id());
            trace_info.pack_trace_info(rez);
            rez.serialize(recorded);
            rez.serialize(applied);
            if (trace_info.recording)
            {
              rez.serialize(all_bar);
              if (all_bar.exists())
                rez.serialize(owner_shard);
            }
            else
              rez.serialize(all_done);
            rez.serialize(origin);
            rez.serialize(allreduce_tag);
          }
          rez.dispatch(origin);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
        }
        else
          perform_collective_pointwise(
              collective, precondition, predicate_guard, copy_expression, op,
              index, collective_match_space, op->get_context_index(), copy_mask,
              (src_point != nullptr) ? src_point->did : 0,
              op->get_unique_op_id(), trace_info, recorded_events,
              applied_events, all_done, all_bar, owner_shard, origin,
              allreduce_tag, copy_restricted);
      }
      return all_done;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::collective_fuse_gather(
        const std::map<IndividualView*, IndexSpaceExpression*>& sources,
        ApEvent precondition, PredEvent predicate_guard, Operation* op,
        const unsigned index, const IndexSpace collective_match_space,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        const bool copy_restricted, const bool need_valid_return)
    //--------------------------------------------------------------------------
    {
      legion_assert(!sources.empty());
      const AddressSpaceID origin = select_origin_space();
      if (origin != local_space)
      {
        // Forward this on to a node with an instance to target
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        ApUserEvent all_done;
        if (need_valid_return)
          all_done = Runtime::create_ap_user_event(&trace_info);
        CollectiveFuseGather rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize<size_t>(sources.size());
          for (const std::pair<IndividualView* const, IndexSpaceExpression*>&
                   it : sources)
          {
            rez.serialize(it.first->did);
            it.second->pack_expression(rez, origin);
          }
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          op->pack_remote_operation(rez, origin, applied_events);
          rez.serialize(index);
          rez.serialize(collective_match_space);
          rez.serialize(copy_mask);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          rez.serialize(all_done);
          rez.serialize<bool>(copy_restricted);
        }
        rez.dispatch(origin);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
        return all_done;
      }
      else
      {
        // We're on a node with a local instance, perform copies to
        // one of the local instances and then either broadcast or reduce
        legion_assert(!local_views.empty());
        legion_assert(collective_mapping != nullptr);
        legion_assert(collective_mapping->contains(local_space));
        legion_assert((op != nullptr) || !copy_restricted);
        const size_t op_ctx_index = op->get_context_index();
        CollectiveAnalysis* first_local_analysis = nullptr;
        if (!copy_restricted && ((op == nullptr) || trace_info.recording))
        {
          // If this is not a copy-out to a restricted collective instance
          // then we should be able to find our local analyses to use for
          // performing operations
          first_local_analysis = local_views.front()->find_collective_analysis(
              op_ctx_index, index, collective_match_space);
          legion_assert(first_local_analysis != nullptr);
          if (op == nullptr)
          {
            op = first_local_analysis->get_operation();
            // Don't need the analysis anymore if we're not tracing
            if (!trace_info.recording)
              first_local_analysis = nullptr;
          }
        }
        legion_assert(op != nullptr);
        const PhysicalTraceInfo& local_info =
            (first_local_analysis == nullptr) ?
                trace_info :
                first_local_analysis->get_trace_info();
        const UniqueID op_id = op->get_unique_op_id();
        // Do the copies to our local instance first
        IndividualView* local_view = local_views.front();
        // Get the dst_fields for performing the local broadcasts
        std::vector<CopySrcDstField> local_fields;
        PhysicalManager* local_manager = local_view->get_manager();
        local_manager->compute_copy_offsets(copy_mask, local_fields);
        const std::vector<Reservation> no_reservations;
        const UniqueInst local_inst(local_view);
        const LgEvent local_unique = local_manager->get_unique_event();
        // Iterate over all the sources and issue copies for each of them
        // to our target local view for their particular expression
        std::set<IndexSpaceExpression*> to_fuse;
        for (const std::pair<IndividualView* const, IndexSpaceExpression*>& it :
             sources)
        {
          // Get the destination precondition
          const ApEvent dst_pre = local_view->find_copy_preconditions(
              false /*reading*/, 0 /*redop*/, copy_mask, it.second, op_id,
              index, applied_events, local_info);
          const ApEvent src_pre = it.first->find_copy_preconditions(
              true /*reading*/, 0 /*redop*/, copy_mask, it.second, op_id, index,
              applied_events, local_info);
          std::vector<CopySrcDstField> src_fields;
          PhysicalManager* source_manager = it.first->get_manager();
          source_manager->compute_copy_offsets(copy_mask, src_fields);
          const ApEvent copy_post = it.second->issue_copy(
              op, local_info, local_fields, src_fields, no_reservations,
              source_manager->tree_id, local_manager->tree_id,
              Runtime::merge_events(
                  &local_info, src_pre, dst_pre, precondition),
              predicate_guard, source_manager->get_unique_event(), local_unique,
              COLLECTIVE_BROADCAST, copy_restricted);
          if (local_info.recording)
          {
            const UniqueInst src_inst(it.first);
            local_info.record_copy_insts(
                copy_post, it.second, src_inst, local_inst, copy_mask,
                copy_mask, 0 /*redop*/, applied_events);
          }
          if (copy_post.exists())
          {
            local_view->add_copy_user(
                false /*reading*/, 0 /*redop*/, copy_post, copy_mask, it.second,
                collective_match_space, op_id, index, recorded_events,
                local_info.recording, runtime->address_space);
            it.first->add_copy_user(
                true /*reading*/, 0 /*redop*/, copy_post, copy_mask, it.second,
                collective_match_space, op_id, index, recorded_events,
                local_info.recording, runtime->address_space);
          }
          // Save this expression for fusing
          to_fuse.insert(it.second);
        }
        // Fuse together the expressions for the group copy
        IndexSpaceExpression* fused_expression =
            runtime->union_index_spaces(to_fuse);
        ApUserEvent all_done;
        ApBarrier all_bar;
        ShardID owner_shard = 0;
        if (need_valid_return)
        {
          all_done = Runtime::create_ap_user_event(&trace_info);
          if (trace_info.recording)
          {
            const size_t arrivals = collective_mapping->size();
            owner_shard = trace_info.record_barrier_creation(all_bar, arrivals);
            // Tracing copy-optimization will eliminate this when
            // the trace gets optimized
            Runtime::trigger_event(
                all_done, all_bar, trace_info, applied_events);
          }
        }
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(origin, local_space, children);
        perform_local_broadcast(
            local_view, local_fields, children, first_local_analysis,
            precondition, predicate_guard, fused_expression, op, index,
            collective_match_space, op_ctx_index, copy_mask, local_inst,
            local_unique, local_info, recorded_events, applied_events, all_done,
            all_bar, owner_shard, local_space, copy_restricted,
            COLLECTIVE_BROADCAST);
        return all_done;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveFuseGather::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent ready;
      CollectiveView* collective = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(view_did, ready));
      std::set<RtEvent> ready_events;
      if (ready.exists())
        ready_events.insert(ready);
      size_t num_sources;
      derez.deserialize(num_sources);
      std::map<IndividualView*, IndexSpaceExpression*> sources;
      for (unsigned idx = 0; idx < num_sources; idx++)
      {
        derez.deserialize(view_did);
        IndividualView* view = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        if (ready.exists())
          ready_events.insert(ready);
        sources[view] = IndexSpaceExpression::unpack_expression(derez, source);
      }
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      Operation* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent all_done;
      derez.deserialize(all_done);
      bool copy_restricted;
      derez.deserialize<bool>(copy_restricted);

      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      const ApEvent done = collective->collective_fuse_gather(
          sources, precondition, predicate_guard, op, index, match_space,
          copy_mask, trace_info, recorded_events, applied_events,
          copy_restricted, all_done.exists());

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (all_done.exists())
        Runtime::trigger_event(all_done, done, trace_info, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != nullptr)
        delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::register_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* user_expr, const UniqueID op_id,
        const size_t op_ctx_index, const unsigned index, ApEvent term_event,
        PhysicalManager* target, CollectiveMapping* analysis_mapping,
        size_t local_collective_arrivals, std::vector<RtEvent>& registered,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const AddressSpaceID source, const bool symbolic /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (local_collective_arrivals > 0)
      {
        // Check to see if we're on the right node for this
        if (!target->is_owner())
        {
          ApUserEvent ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent registered_event = Runtime::create_rt_user_event();
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          ViewRegisterUser rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target->did);
            rez.serialize(usage);
            rez.serialize(user_mask);
            rez.serialize(user_expr->handle);
            rez.serialize(op_id);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(term_event);
            rez.serialize(local_collective_arrivals);
            rez.serialize(ready_event);
            rez.serialize(registered_event);
            rez.serialize(applied_event);
            trace_info.pack_trace_info(rez);
          }
          rez.dispatch(target->owner_space);
          registered.emplace_back(registered_event);
          applied_events.insert(applied_event);
          return ready_event;
        }
        else
          return register_collective_user(
              usage, user_mask, user_expr, op_id, op_ctx_index, index,
              term_event, target, local_collective_arrivals, registered,
              applied_events, trace_info, symbolic);
      }
      legion_assert(target->is_owner());
      legion_assert(analysis_mapping == nullptr);
      // Iterate through our local views and find the view for the target
      for (unsigned idx = 0; idx < local_views.size(); idx++)
        if (local_views[idx]->get_manager() == target)
          return local_views[idx]->register_user(
              usage, user_mask, user_expr, op_id, op_ctx_index, index,
              term_event, target, analysis_mapping, local_collective_arrivals,
              registered, applied_events, trace_info, source, symbolic);
      // Should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::contains(PhysicalManager* manager) const
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID manager_space = get_analysis_space(manager);
      if (manager_space != local_space)
      {
        if ((collective_mapping == nullptr) ||
            !collective_mapping->contains(manager_space))
          return false;
        // Check all the current
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          std::map<PhysicalManager*, IndividualView*>::const_iterator finder =
              remote_instances.find(manager);
          if (finder != remote_instances.end())
            return true;
          // If we already have all the managers from that node then
          // we don't need to check again
          if (remote_instance_responses.contains(manager_space))
            return false;
        }
        // Send the request and wait for the result
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        CollectiveRemoteInstancesRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(ready_event);
        }
        rez.dispatch(manager_space);
        if (!ready_event.has_triggered())
          ready_event.wait();
        AutoLock v_lock(view_lock, false /*exclusive*/);
        return (remote_instances.find(manager) != remote_instances.end());
      }
      else
      {
        for (unsigned idx = 0; idx < local_views.size(); idx++)
          if (local_views[idx]->get_manager() == manager)
            return true;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::meets_regions(
        const std::vector<LogicalRegion>& regions, bool tight_bounds) const
    //--------------------------------------------------------------------------
    {
      if (!local_views.empty())
        return local_views.front()->get_manager()->meets_regions(
            regions, tight_bounds);
      legion_assert(
          (collective_mapping == nullptr) ||
          !collective_mapping->contains(local_space));
      PhysicalManager* manager = nullptr;
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        if (!remote_instances.empty())
          manager = remote_instances.begin()->first;
      }
      if (manager == nullptr)
      {
        const AddressSpaceID target_space =
            (collective_mapping == nullptr) ?
                owner_space :
                collective_mapping->find_nearest(local_space);
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        CollectiveRemoteInstancesRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(ready_event);
        }
        rez.dispatch(target_space);
        if (!ready_event.has_triggered())
          ready_event.wait();
        AutoLock v_lock(view_lock, false /*exclusive*/);
        legion_assert(!remote_instances.empty());
        manager = remote_instances.begin()->first;
      }
      return manager->meets_regions(regions, tight_bounds);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::find_instances_in_memory(
        Memory memory, std::vector<PhysicalManager*>& instances)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID memory_space = memory.address_space();
      if (memory_space != local_space)
      {
        // No point checking if we know that it won't have it
        if ((collective_mapping == nullptr) ||
            !collective_mapping->contains(memory_space))
          return;
        {
          AutoLock v_lock(view_lock, false /*exclusive*/);
          // See if we need the check
          if (remote_instance_responses.contains(memory_space))
          {
            for (const std::pair<PhysicalManager* const, IndividualView*>& it :
                 remote_instances)
              if (it.first->memory_manager->memory == memory)
                instances.emplace_back(it.first);
            return;
          }
        }
        const RtUserEvent ready_event = Runtime::create_rt_user_event();
        CollectiveRemoteInstancesRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(ready_event);
        }
        rez.dispatch(memory_space);
        if (!ready_event.has_triggered())
          ready_event.wait();
        AutoLock v_lock(view_lock, false /*exclusive*/);
        for (const std::pair<PhysicalManager* const, IndividualView*>& it :
             remote_instances)
          if (it.first->memory_manager->memory == memory)
            instances.emplace_back(it.first);
      }
      else
      {
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          PhysicalManager* manager = local_views[idx]->get_manager();
          if (manager->memory_manager->memory == memory)
            instances.emplace_back(manager);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveRemoteInstancesRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, ready));
      RtUserEvent done;
      derez.deserialize(done);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      legion_assert(!view->local_views.empty());
      CollectiveRemoteInstancesResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize<size_t>(view->local_views.size());
        for (unsigned idx = 0; idx < view->local_views.size(); idx++)
          rez.serialize(view->local_views[idx]->did);
        rez.serialize(done);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::process_remote_instances_response(
        AddressSpaceID src, const std::vector<IndividualView*>& views)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      // Deduplicate cases where we already received this response
      if (remote_instance_responses.contains(src))
        return;
      for (IndividualView* const view : views)
      {
        PhysicalManager* manager = view->get_manager();
        if (remote_instances.insert(std::make_pair(manager, view)).second)
          view->add_nested_gc_ref(did);
      }
      remote_instance_responses.add(src);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::record_remote_instances(
        const std::vector<IndividualView*>& views)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      for (IndividualView* const view : views)
      {
        PhysicalManager* manager = view->get_manager();
        if (remote_instances.insert(std::make_pair(manager, view)).second)
          view->add_nested_gc_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveRemoteInstancesResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, ready));
      std::vector<RtEvent> ready_events;
      if (ready.exists())
        ready_events.emplace_back(ready);
      size_t num_instances;
      derez.deserialize(num_instances);
      std::vector<IndividualView*> instances(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
      {
        derez.deserialize(did);
        instances[idx] = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(did, ready));
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      RtUserEvent done;
      derez.deserialize(done);

      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      view->process_remote_instances_response(source, instances);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::find_instances_nearest_memory(
        Memory memory, std::vector<PhysicalManager*>& instances, bool bandwidth)
    //--------------------------------------------------------------------------
    {
      constexpr size_t size_max = std::numeric_limits<size_t>::max();
      size_t best = bandwidth ? 0 : size_max;
      if (collective_mapping != nullptr)
      {
        std::atomic<size_t> atomic_best(best);
        const AddressSpaceID origin = select_origin_space();
        std::vector<DistributedID> best_instances;
        RtEvent ready = find_instances_nearest_memory(
            memory, local_space, &best_instances, &atomic_best, origin, best,
            bandwidth);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        std::vector<RtEvent> ready_events;
        for (const DistributedID& it : best_instances)
        {
          instances.emplace_back(
              runtime->find_or_request_instance_manager(it, ready));
          if (ready.exists())
            ready_events.emplace_back(ready);
        }
        if (!ready_events.empty())
        {
          ready = Runtime::merge_events(ready_events);
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
        }
      }
      else
      {
        if (!is_owner())
        {
          const RtUserEvent ready_event = Runtime::create_rt_user_event();
          CollectiveRemoteInstancesRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(ready_event);
          }
          rez.dispatch(owner_space);
          if (!ready_event.has_triggered())
            ready_event.wait();
          std::map<Memory, size_t> searches;
          AutoLock v_lock(view_lock, false /*exclusive*/);
          for (const std::pair<PhysicalManager* const, IndividualView*>& it :
               remote_instances)
          {
            const Memory local = it.first->memory_manager->memory;
            std::map<Memory, size_t>::const_iterator finder =
                searches.find(local);
            if (finder == searches.end())
            {
              Realm::Machine::AffinityDetails affinity;
              if (runtime->machine.has_affinity(memory, local, &affinity))
              {
                legion_assert(0 < affinity.bandwidth);
#ifndef __clang__  // clang's idea of size_max is off by one
                legion_assert(affinity.bandwidth < size_max);
#endif
                if (bandwidth)
                {
                  searches[local] = affinity.bandwidth;
                  if (affinity.bandwidth >= best)
                  {
                    if (affinity.bandwidth > best)
                    {
                      instances.clear();
                      best = affinity.bandwidth;
                    }
                    instances.emplace_back(it.first);
                  }
                }
                else
                {
                  legion_assert(0 < affinity.latency);
#ifndef __clang__  // clang's idea of size_max is off by one
                  legion_assert(affinity.latency < size_max);
#endif
                  searches[local] = affinity.latency;
                  if (affinity.latency <= best)
                  {
                    if (affinity.latency < best)
                    {
                      instances.clear();
                      best = affinity.latency;
                    }
                    instances.emplace_back(it.first);
                  }
                }
              }
              else
                searches[local] = bandwidth ? 0 : size_max;
            }
            else if (finder->second == best)
              instances.emplace_back(it.first);
          }
        }
        else
          find_nearest_local_instances(memory, best, instances, bandwidth);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveView::find_instances_nearest_memory(
        Memory memory, AddressSpaceID source,
        std::vector<DistributedID>* instances, std::atomic<size_t>* target,
        AddressSpaceID origin, size_t best, bool bandwidth)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      const AddressSpaceID space = memory.address_space();
      if ((space != local_space) || !collective_mapping->contains(local_space))
      {
        if (collective_mapping->contains(space))
        {
          legion_assert(source == local_space);
          // Assume that all memmories in the same space are always inherently
          // closer to the target memory than any others, so we can send the
          // request straight to that node and do the lookup
          const RtUserEvent done = Runtime::create_rt_user_event();
          CollectiveNearestInstancesRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(memory);
            rez.serialize(source);
            rez.serialize(instances);
            rez.serialize(target);
            rez.serialize(origin);
            rez.serialize(best);
            rez.serialize<bool>(bandwidth);
            rez.serialize(done);
            pack_global_ref(rez);
          }
          rez.dispatch(space);
          return done;
        }
        else
        {
          if (collective_mapping->contains(local_space))
          {
            // Do our local check and update the best
            std::vector<PhysicalManager*> local_results;
            find_nearest_local_instances(
                memory, best, local_results, bandwidth);
            std::vector<RtEvent> done_events;
            std::vector<AddressSpaceID> children;
            collective_mapping->get_children(origin, local_space, children);
            for (const AddressSpaceID& child : children)
            {
              const RtUserEvent done = Runtime::create_rt_user_event();
              CollectiveNearestInstancesRequest rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(memory);
                rez.serialize(source);
                rez.serialize(instances);
                rez.serialize(target);
                rez.serialize(origin);
                rez.serialize(best);
                rez.serialize<bool>(bandwidth);
                rez.serialize(done);
                pack_global_ref(rez);
              }
              rez.dispatch(child);
              done_events.emplace_back(done);
            }
            if (!local_results.empty())
            {
              if (source != local_space)
              {
                const RtUserEvent done = Runtime::create_rt_user_event();
                CollectiveNearestInstancesResponse rez;
                {
                  RezCheck z(rez);
                  rez.serialize(instances);
                  rez.serialize(target);
                  rez.serialize(best);
                  rez.serialize<size_t>(local_results.size());
                  for (PhysicalManager* manager : local_results)
                    rez.serialize(manager->did);
                  rez.serialize<bool>(bandwidth);
                  rez.serialize(done);
                }
                rez.dispatch(source);
                done_events.emplace_back(done);
              }
              else
              {
                std::vector<DistributedID> results(local_results.size());
                for (unsigned idx = 0; idx < local_results.size(); idx++)
                  results[idx] = local_results[idx]->did;
                process_nearest_instances(
                    target, instances, best, results, bandwidth);
              }
            }
            if (!done_events.empty())
              return Runtime::merge_events(done_events);
          }
          else
          {
            legion_assert(source == local_space);
            // Send to the origin to start
            const RtUserEvent done = Runtime::create_rt_user_event();
            CollectiveNearestInstancesRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(memory);
              rez.serialize(source);
              rez.serialize(instances);
              rez.serialize(target);
              rez.serialize(origin);
              rez.serialize(best);
              rez.serialize<bool>(bandwidth);
              rez.serialize(done);
              pack_global_ref(rez);
            }
            rez.dispatch(origin);
            return done;
          }
        }
      }
      else
      {
        // Assume that all memories in the same space are always inherently
        // closer to the target memory than any others
        // See if we find the memory itself
        std::vector<PhysicalManager*> results;
        find_nearest_local_instances(memory, best, results, bandwidth);
        if (source != local_space)
        {
          if (!results.empty())
          {
            const RtUserEvent done = Runtime::create_rt_user_event();
            CollectiveNearestInstancesResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(instances);
              rez.serialize(target);
              rez.serialize(best);
              rez.serialize<size_t>(results.size());
              for (PhysicalManager* manager : results)
                rez.serialize(manager->did);
              rez.serialize<bool>(bandwidth);
              rez.serialize(done);
            }
            rez.dispatch(source);
            return done;
          }
        }
        else
        {
          // This is the local case, so there's no atomicity required
          for (PhysicalManager* manager : results)
            instances->emplace_back(manager->did);
          target->store(best);
        }
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::find_nearest_local_instances(
        Memory memory, size_t& best, std::vector<PhysicalManager*>& results,
        bool bandwidth) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        PhysicalManager* manager = local_views[idx]->get_manager();
        if (manager->memory_manager->memory == memory)
          results.emplace_back(manager);
      }
      constexpr size_t size_max = std::numeric_limits<size_t>::max();
      if (results.empty())
      {
        // Nothing in the memory itself, so see which of our memories
        // are closer to anything else
        std::map<Memory, size_t> searches;
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          PhysicalManager* manager = local_views[idx]->get_manager();
          const Memory local = manager->memory_manager->memory;
          std::map<Memory, size_t>::const_iterator finder =
              searches.find(local);
          if (finder == searches.end())
          {
            Realm::Machine::AffinityDetails affinity;
            if (runtime->machine.has_affinity(memory, local, &affinity))
            {
              legion_assert(0 < affinity.bandwidth);
#ifndef __clang__  // clang's idea of size_max is off by one
              legion_assert(affinity.bandwidth < size_max);
#endif
              if (bandwidth)
              {
                searches[local] = affinity.bandwidth;
                if (affinity.bandwidth >= best)
                {
                  if (affinity.bandwidth > best)
                  {
                    results.clear();
                    best = affinity.bandwidth;
                  }
                  results.emplace_back(manager);
                }
              }
              else
              {
                legion_assert(0 < affinity.latency);
#ifndef __clang__  // clang's idea of size_max is off by one
                legion_assert(affinity.latency < size_max);
#endif
                searches[local] = affinity.latency;
                if (affinity.latency <= best)
                {
                  if (affinity.latency < best)
                  {
                    results.clear();
                    best = affinity.latency;
                  }
                  results.emplace_back(manager);
                }
              }
            }
            else
              searches[local] = bandwidth ? 0 : size_max;
          }
          else if (finder->second == best)
            results.emplace_back(manager);
        }
        if (results.empty())
        {
          // If none of them have affinity then they are all
          // equally bad so we can just pass in all of them
          for (unsigned idx = 0; idx < local_views.size(); idx++)
            results.emplace_back(local_views[idx]->get_manager());
        }
      }
      else
        best = bandwidth ? size_max - 1 : 1;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveNearestInstancesRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Memory memory;
      derez.deserialize(memory);
      AddressSpaceID source;
      derez.deserialize(source);
      std::vector<DistributedID>* instances;
      derez.deserialize(instances);
      std::atomic<size_t>* target;
      derez.deserialize(target);
      AddressSpaceID origin;
      derez.deserialize(origin);
      size_t best;
      derez.deserialize(best);
      bool bandwidth;
      derez.deserialize(bandwidth);
      RtUserEvent done;
      derez.deserialize(done);

      CollectiveView* manager = static_cast<CollectiveView*>(
          runtime->find_distributed_collectable(did));
      Runtime::trigger_event(
          done,
          manager->find_instances_nearest_memory(
              memory, source, instances, target, origin, best, bandwidth));
      manager->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveNearestInstancesResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::vector<DistributedID>* instances;
      derez.deserialize(instances);
      std::atomic<size_t>* target;
      derez.deserialize(target);
      size_t best;
      derez.deserialize(best);
      size_t num_instances;
      derez.deserialize(num_instances);
      std::vector<DistributedID> results(num_instances);
      for (unsigned idx = 0; idx < num_instances; idx++)
        derez.deserialize(results[idx]);
      bool bandwidth;
      derez.deserialize(bandwidth);
      CollectiveView::process_nearest_instances(
          target, instances, best, results, bandwidth);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveView::process_nearest_instances(
        std::atomic<size_t>* target, std::vector<DistributedID>* instances,
        size_t best, const std::vector<DistributedID>& results, bool bandwidth)
    //--------------------------------------------------------------------------
    {
      // spin until we can get safely set the guard to add our entries
      const size_t guard = bandwidth ? std::numeric_limits<size_t>::max() : 0;
      size_t current = target->load();
      while ((current == guard) || (bandwidth && (current <= best)) ||
             (!bandwidth && (best <= current)))
      {
        if (!target->compare_exchange_weak(current, guard))
          continue;
        // If someone else still holds the guard then keep trying
        if (current == guard)
          continue;
        if (bandwidth)
        {
          if (current < best)
            instances->clear();
          for (unsigned idx = 0; idx < results.size(); idx++)
            instances->emplace_back(results[idx]);
        }
        else
        {
          if (best < current)
            instances->clear();
          for (unsigned idx = 0; idx < results.size(); idx++)
            instances->emplace_back(results[idx]);
        }
        target->store(best);
        break;
      }
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveView::select_source_space(
        AddressSpaceID destination) const
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      // 1. If the collective manager has instances on the same node
      //    as the destination then we'll use one of them
      if (collective_mapping->contains(destination))
        return destination;
      // 2. If the collective manager has instances on the local node
      //    then we'll use one of them
      if (collective_mapping->contains(local_space))
        return local_space;
      // 3. Pick the node closest to the destination in the collective
      //    manager and use that to issue copies
      return collective_mapping->find_nearest(destination);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::pack_fields(
        Serializer& rez, const std::vector<CopySrcDstField>& fields,
        LgEvent inst_uid) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(inst_uid);
      rez.serialize<size_t>(fields.size());
      for (unsigned idx = 0; idx < fields.size(); idx++)
        rez.serialize(fields[idx]);
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        // Pack the instance points for these instances so we can check to
        // see if we already fetched them on the remote node
        std::map<DistributedID, IndividualView*> to_send;
        for (const CopySrcDstField& field : fields)
        {
          bool found = false;
          for (unsigned idx = 0; idx < local_views.size(); idx++)
          {
            PhysicalManager* manager = local_views[idx]->get_manager();
            if (manager->instance != field.inst)
              continue;
            to_send.emplace(local_views[idx]->did, local_views[idx]);
            found = true;
            break;
          }
          if (!found)
          {
            AutoLock v_lock(view_lock, false /*exclusive*/);
            for (const std::map<PhysicalManager*, IndividualView*>::value_type&
                     remote_entry : remote_instances)
            {
              if (remote_entry.first->instance != field.inst)
                continue;
              to_send.emplace(remote_entry.second->did, remote_entry.second);
              found = true;
              break;
            }
            legion_assert(found);
          }
        }
        legion_assert(!to_send.empty());
        rez.serialize<size_t>(to_send.size());
        for (const std::pair<const DistributedID, IndividualView*>&
                 distributed_id : to_send)
          rez.serialize(distributed_id.first);
        for (const std::pair<const DistributedID, IndividualView*>& view :
             to_send)
          view.second->pack_global_ref(rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ LgEvent CollectiveView::unpack_fields(
        std::vector<CopySrcDstField>& fields, Deserializer& derez,
        std::set<RtEvent>& ready_events, CollectiveView* view,
        RtEvent view_ready)
    //--------------------------------------------------------------------------
    {
      LgEvent inst_uid;
      derez.deserialize(inst_uid);
      size_t num_fields;
      derez.deserialize(num_fields);
      fields.resize(num_fields);
      const Processor local_proc = Processor::get_executing_processor();
      for (unsigned idx = 0; idx < fields.size(); idx++)
      {
        CopySrcDstField& field = fields[idx];
        derez.deserialize(field);
        // Only need to check this on the first iteration since all the
        // fields should share the same instance
        if (idx == 0)
        {
          // Check to see if we fetched the metadata for this instance
          RtEvent ready(field.inst.fetch_metadata(local_proc));
          if (implicit_profiler != nullptr)
            implicit_profiler->record_fetch_metadata(ready, inst_uid);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
        }
      }
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        // Legion Spy is a bit dumb currently and needs to have logged every
        // instance on every node where it might be used currently, so check
        // to make sure we've logged it by loading the individual view and
        // therefore the manager for each instances that we need
        size_t num_views;
        derez.deserialize(num_views);
        if (num_views > 0)
        {
          std::vector<RtEvent> wait_events;
          std::vector<IndividualView*> views(num_views);
          for (unsigned idx = 0; idx < num_views; idx++)
          {
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready;
            views[idx] = static_cast<IndividualView*>(
                runtime->find_or_request_logical_view(did, ready));
            if (ready.exists())
              wait_events.emplace_back(ready);
          }
          if (!wait_events.empty())
          {
            if (view_ready.exists())
              wait_events.emplace_back(view_ready);
            const RtEvent wait_on = Runtime::merge_events(wait_events);
            if (wait_on.exists() && !wait_on.has_triggered())
              wait_on.wait();
          }
          else if (view_ready.exists() && !view_ready.has_triggered())
            view_ready.wait();
          view->record_remote_instances(views);
          for (IndividualView* const view : views)
            view->unpack_global_ref(derez);
        }
        else
        {
          // These fields are from an individual manager so just
          // load a copy of it here
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          runtime->find_or_request_logical_view(did, ready);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      return inst_uid;
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveView::find_local_index(PhysicalManager* target) const
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < local_views.size(); idx++)
        if (local_views[idx]->get_manager() == target)
          return idx;
      // We should always find it
      std::abort();
    }

    //--------------------------------------------------------------------------
    void CollectiveView::register_collective_analysis(
        PhysicalManager* target, CollectiveAnalysis* analysis,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // First check to see if we are on the right node for this target
      const AddressSpaceID analysis_space = get_analysis_space(target);
      if (analysis_space != local_space)
      {
        const RtEvent applied = Runtime::create_rt_user_event();
        CollectiveRemoteRegistration rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target->did);
          analysis->pack_collective_analysis(
              rez, analysis_space, applied_events);
          rez.serialize(applied);
        }
        rez.dispatch(analysis_space);
        applied_events.insert(applied);
        return;
      }
      else
      {
        const unsigned local_index = find_local_index(target);
        local_views[local_index]->register_collective_analysis(this, analysis);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveRemoteRegistration::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent view_ready, manager_ready;
      CollectiveView* collective_view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, view_ready));
      derez.deserialize(did);
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, manager_ready);
      std::set<RtEvent> applied_events;
      RemoteCollectiveAnalysis* analysis =
          RemoteCollectiveAnalysis::unpack(derez);
      analysis->add_reference();
      RtUserEvent applied;
      derez.deserialize(applied);

      if (view_ready.exists() && !view_ready.has_triggered())
        view_ready.wait();
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();
      collective_view->register_collective_analysis(
          manager, analysis, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::register_collective_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
        const unsigned index, ApEvent term_event, PhysicalManager* target,
        size_t local_collective_arrivals,
        std::vector<RtEvent>& registered_events,
        std::set<RtEvent>& applied_events, const PhysicalTraceInfo& trace_info,
        const bool symbolic)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      legion_assert(
          ((collective_mapping != nullptr) &&
           collective_mapping->contains(local_space)) ||
          is_owner());
      const unsigned target_index = find_local_index(target);
      // We performing a collective analysis, this function performs a
      // parallel rendezvous to ensure several important invariants.
      // 1. SUBTLE!!! Make sure that all the participants have arrived
      //    at this function before performing any view analysis. This
      //    is required to ensure that any copies that need to be issued
      //    have had a chance to record their view users first before we
      //    attempt to look for preconditions for this user.
      // 2. Similarly make sure that the applied events reflects the case
      //    where all the users have been recorded across the views on
      //    each node to ensure that any downstream copies or users will
      //    observe all the most recent users.
      // 3. Deduplicate across all the participants on the same node since
      //    there is always just a single view on each node. This function
      //    call will always return the local user precondition for the
      //    local instances. Make sure to merge together all the partcipant
      //    postconditions for the local instances to reflect in the view
      //    that the local instances are ready when they are all ready.
      // 4. Do NOT block in this function call or you can risk deadlock because
      //    we might be doing several of these calls for a region requirement
      //    on different instances and the orders might vary on each node.

      // The unique tag for the rendezvous is our context ID which will be
      // the same across all points and the index of our region requirement
      PhysicalTraceInfo* result_info;
      RtUserEvent local_registered, global_registered;
      RtUserEvent local_applied, global_applied;
      std::vector<RtEvent> remote_registered, remote_applied;
      std::vector<ApUserEvent> local_ready_events;
      std::vector<std::vector<ApEvent> > local_term_events;
      const RendezvousKey key(op_ctx_index, index, expr->handle);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          // Count how many expected arrivals we have
          // If we're doing collective per space
          rendezvous.remaining_local_arrivals = local_collective_arrivals;
          rendezvous.local_initialized = true;
          rendezvous.remaining_remote_arrivals =
              (collective_mapping == nullptr) ?
                  0 :
                  collective_mapping->count_children(owner_space, local_space);
          rendezvous.local_term_events.resize(local_views.size());
          rendezvous.ready_events.resize(local_views.size());
          for (unsigned idx = 0; idx < local_views.size(); idx++)
            rendezvous.ready_events[idx] =
                Runtime::create_ap_user_event(&trace_info);
          rendezvous.trace_info = new PhysicalTraceInfo(trace_info);
          rendezvous.expr = expr;
          expr->add_nested_expression_reference(did);
          rendezvous.local_registered = Runtime::create_rt_user_event();
          rendezvous.global_registered = Runtime::create_rt_user_event();
          rendezvous.local_applied = Runtime::create_rt_user_event();
          rendezvous.global_applied = Runtime::create_rt_user_event();
          // This is very subtle!
          // For a collective view to become valid, we need to know
          // that a valid reference is added on all the nodes where
          // there are local views. It's unclear whether the analysis
          // performing this registration will actually try to record
          // the collective view with an equivalence set that will add
          // a nested valid reference or not, but to be safe we make
          // sure that all collective views are valid at the point of
          // registration such that the follow-on decrement after the
          // registration will only invalidate the collective view if
          // the view wasn't registered with a collective analysis
#ifdef LEGION_DEBUG_GC
          if (valid_references++ == 0)
#else
          if (valid_references.fetch_add(1) == 0)
#endif
            notify_valid();
        }
        else if (!finder->second.local_initialized)
        {
          // First local arrival, but rendezvous was made by a remote
          // arrival so we need to make the ready event
          legion_assert(finder->second.ready_events.empty());
          legion_assert(finder->second.local_term_events.empty());
          legion_assert(finder->second.trace_info == nullptr);
          finder->second.local_term_events.resize(local_views.size());
          finder->second.ready_events.resize(local_views.size());
          for (unsigned idx = 0; idx < local_views.size(); idx++)
            finder->second.ready_events[idx] =
                Runtime::create_ap_user_event(&trace_info);
          finder->second.trace_info = new PhysicalTraceInfo(trace_info);
          finder->second.expr = expr;
          expr->add_nested_expression_reference(did);
          finder->second.remaining_local_arrivals = local_collective_arrivals;
          finder->second.local_initialized = true;
          // This is very subtle!
          // See similar comment above for what we are doing here
#ifdef LEGION_DEBUG_GC
          if (valid_references++ == 0)
#else
          if (valid_references.fetch_add(1) == 0)
#endif
            notify_valid();
        }
        if (term_event.exists())
          finder->second.local_term_events[target_index].emplace_back(
              term_event);
        // Record the applied events
        registered_events.emplace_back(finder->second.global_registered);
        applied_events.insert(finder->second.global_applied);
        // The result will be the ready event
        ApEvent result = finder->second.ready_events[target_index];
        result_info = finder->second.trace_info;
        expr = finder->second.expr;
        legion_assert(finder->second.local_initialized);
        legion_assert(finder->second.remaining_local_arrivals > 0);
        // See if we've seen all the arrivals
        if (--finder->second.remaining_local_arrivals == 0)
        {
          // If we're going to need to defer this then save
          // all of our local state needed to perform registration
          // for when it is safe to do so
          if (!is_owner() || (finder->second.remaining_remote_arrivals > 0))
          {
            // Save the state that we need for finalization later
            finder->second.usage = usage;
            finder->second.mask =
                new HeapifyBox<FieldMask, OPERATION_LIFETIME>(user_mask);
            finder->second.op_id = op_id;
            finder->second.symbolic = symbolic;
          }
          if (finder->second.remaining_remote_arrivals == 0)
          {
            if (!is_owner())
            {
              // Not the owner so send the message to the parent
              RtEvent registered = finder->second.local_registered;
              if (!finder->second.remote_registered.empty())
              {
                finder->second.remote_registered.emplace_back(registered);
                registered =
                    Runtime::merge_events(finder->second.remote_registered);
              }
              RtEvent applied = finder->second.local_applied;
              if (!finder->second.remote_applied.empty())
              {
                finder->second.remote_applied.emplace_back(applied);
                applied = Runtime::merge_events(finder->second.remote_applied);
              }
              const AddressSpaceID parent =
                  collective_mapping->get_parent(owner_space, local_space);
              CollectiveRegisterUserRequest rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(op_ctx_index);
                rez.serialize(index);
                rez.serialize(expr->handle);
                rez.serialize(registered);
                rez.serialize(applied);
              }
              rez.dispatch(parent);
              return result;
            }
            else
            {
              // We're going to fall through so grab the state
              // that we need to do the finalization now
              remote_registered.swap(finder->second.remote_registered);
              remote_applied.swap(finder->second.remote_applied);
              local_registered = finder->second.local_registered;
              global_registered = finder->second.global_registered;
              local_applied = finder->second.local_applied;
              global_applied = finder->second.global_applied;
              local_ready_events.swap(finder->second.ready_events);
              local_term_events.swap(finder->second.local_term_events);
              // We can erase this from the data structure now
              rendezvous_users.erase(finder);
            }
          }
          else  // Still waiting for remote arrivals
            return result;
        }
        else  // Not the last local arrival so we can just return the result
          return result;
      }
      legion_assert(is_owner());
      finalize_collective_user(
          usage, user_mask, expr, op_id, op_ctx_index, index, local_registered,
          global_registered, local_applied, global_applied, local_ready_events,
          local_term_events, result_info, symbolic);
      RtEvent all_registered = local_registered;
      if (!remote_registered.empty())
      {
        remote_registered.emplace_back(all_registered);
        all_registered = Runtime::merge_events(remote_registered);
      }
      Runtime::trigger_event(global_registered, all_registered);
      RtEvent all_applied = local_applied;
      if (!remote_applied.empty())
      {
        remote_applied.emplace_back(all_applied);
        all_applied = Runtime::merge_events(remote_applied);
      }
      Runtime::trigger_event(global_applied, all_applied);
      return local_ready_events[target_index];
    }

    //--------------------------------------------------------------------------
    void CollectiveView::process_register_user_request(
        const size_t op_ctx_index, const unsigned index,
        const IndexSpace match_space, RtEvent registered, RtEvent applied)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      UserRendezvous to_perform;
      const RendezvousKey key(op_ctx_index, index, match_space);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        if (finder == rendezvous_users.end())
        {
          // If we are then make the record for knowing when we've seen
          // all the expected arrivals
          finder =
              rendezvous_users.insert(std::make_pair(key, UserRendezvous()))
                  .first;
          UserRendezvous& rendezvous = finder->second;
          rendezvous.local_initialized = false;
          rendezvous.remaining_remote_arrivals =
              collective_mapping->count_children(owner_space, local_space);
          rendezvous.local_registered = Runtime::create_rt_user_event();
          rendezvous.global_registered = Runtime::create_rt_user_event();
          rendezvous.local_applied = Runtime::create_rt_user_event();
          rendezvous.global_applied = Runtime::create_rt_user_event();
        }
        finder->second.remote_registered.emplace_back(registered);
        finder->second.remote_applied.emplace_back(applied);
        legion_assert(finder->second.remaining_remote_arrivals > 0);
        // If we're not the last arrival then we're done
        if ((--finder->second.remaining_remote_arrivals > 0) ||
            !finder->second.local_initialized ||
            (finder->second.remaining_local_arrivals > 0))
          return;
        if (!is_owner())
        {
          // Continue sending the message up the tree to the parent
          registered = finder->second.local_registered;
          if (!finder->second.remote_registered.empty())
          {
            finder->second.remote_registered.emplace_back(registered);
            registered =
                Runtime::merge_events(finder->second.remote_registered);
          }
          applied = finder->second.local_applied;
          if (!finder->second.remote_applied.empty())
          {
            finder->second.remote_applied.emplace_back(applied);
            applied = Runtime::merge_events(finder->second.remote_applied);
          }
          const AddressSpaceID parent =
              collective_mapping->get_parent(owner_space, local_space);
          CollectiveRegisterUserRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(op_ctx_index);
            rez.serialize(index);
            rez.serialize(match_space);
            rez.serialize(registered);
            rez.serialize(applied);
          }
          rez.dispatch(parent);
          return;
        }
        legion_assert(finder->second.remaining_analyses == 0);
        // We're the owner so we can start doing the user registration
        // Grab everything we need to call finalize_collective_user
        to_perform = std::move(finder->second);
        // Then we can erase the entry
        rendezvous_users.erase(finder);
      }
      legion_assert(is_owner());
      finalize_collective_user(
          to_perform.usage, *(to_perform.mask), to_perform.expr,
          to_perform.op_id, op_ctx_index, index, to_perform.local_registered,
          to_perform.global_registered, to_perform.local_applied,
          to_perform.global_applied, to_perform.ready_events,
          to_perform.local_term_events, to_perform.trace_info,
          to_perform.symbolic);
      RtEvent all_registered = to_perform.local_registered;
      if (!to_perform.remote_registered.empty())
      {
        to_perform.remote_registered.emplace_back(all_registered);
        all_registered = Runtime::merge_events(to_perform.remote_registered);
      }
      Runtime::trigger_event(to_perform.global_registered, all_registered);
      RtEvent all_applied = to_perform.local_applied;
      if (!to_perform.remote_applied.empty())
      {
        to_perform.remote_applied.emplace_back(all_applied);
        all_applied = Runtime::merge_events(to_perform.remote_applied);
      }
      Runtime::trigger_event(to_perform.global_applied, all_applied);
      delete to_perform.mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveRegisterUserRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, ready));
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      RtEvent registered, applied;
      derez.deserialize(registered);
      derez.deserialize(applied);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->process_register_user_request(
          op_ctx_index, index, match_space, registered, applied);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::process_register_user_response(
        const size_t op_ctx_index, const unsigned index,
        const IndexSpace match_space, const RtEvent registered,
        const RtEvent applied)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      legion_assert(!local_views.empty());
      UserRendezvous to_perform;
      const RendezvousKey key(op_ctx_index, index, match_space);
      {
        AutoLock v_lock(view_lock);
        // Check to see if we're the first one to arrive on this node
        std::map<RendezvousKey, UserRendezvous>::iterator finder =
            rendezvous_users.find(key);
        legion_assert(finder != rendezvous_users.end());
        legion_assert(finder->second.remaining_analyses == 0);
        to_perform = std::move(finder->second);
        // Can now remove this from the data structure
        rendezvous_users.erase(finder);
      }
      // Now we can perform the user registration
      finalize_collective_user(
          to_perform.usage, *(to_perform.mask), to_perform.expr,
          to_perform.op_id, op_ctx_index, index, to_perform.local_registered,
          to_perform.global_registered, to_perform.local_applied,
          to_perform.global_applied, to_perform.ready_events,
          to_perform.local_term_events, to_perform.trace_info,
          to_perform.symbolic);
      Runtime::trigger_event(to_perform.global_registered, registered);
      Runtime::trigger_event(to_perform.global_applied, applied);
      delete to_perform.mask;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveRegisterUserResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, ready));
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      RtEvent registered, applied;
      derez.deserialize(registered);
      derez.deserialize(applied);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      view->process_register_user_response(
          op_ctx_index, index, match_space, registered, applied);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::finalize_collective_user(
        const RegionUsage& usage, const FieldMask& user_mask,
        IndexSpaceNode* expr, const UniqueID op_id, const size_t op_ctx_index,
        const unsigned index, RtUserEvent local_registered,
        RtEvent global_registered, RtUserEvent local_applied,
        RtEvent global_applied, std::vector<ApUserEvent>& ready_events,
        std::vector<std::vector<ApEvent> >& term_events,
        const PhysicalTraceInfo* trace_info, const bool symbolic)
    //--------------------------------------------------------------------------
    {
      // First send out any messages to the children so they can start
      // their own registrations
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      if (!children.empty())
      {
        CollectiveRegisterUserResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(op_ctx_index);
          rez.serialize(index);
          rez.serialize(expr->handle);
          rez.serialize(global_registered);
          rez.serialize(global_applied);
        }
        for (const AddressSpaceID& address_space : children)
          rez.dispatch(address_space);
      }
      legion_assert(local_views.size() == term_events.size());
      legion_assert(local_views.size() == ready_events.size());
      // Perform the registration on the local views
      std::vector<RtEvent> registered_events;
      std::set<RtEvent> applied_events;
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        const ApEvent term_event =
            Runtime::merge_events(trace_info, term_events[idx]);
        const ApEvent ready = local_views[idx]->register_user(
            usage, user_mask, expr, op_id, op_ctx_index, index, term_event,
            local_views[idx]->get_manager(), nullptr /*analysis mapping*/,
            0 /*no collective arrivals*/, registered_events, applied_events,
            *trace_info, runtime->address_space, symbolic);
        Runtime::trigger_event(
            ready_events[idx], ready, *trace_info, applied_events);
        // Also unregister the collective analyses
        local_views[idx]->unregister_collective_analysis(
            this, op_ctx_index, index, expr->handle);
      }
      if (!registered_events.empty())
        Runtime::trigger_event(
            local_registered, Runtime::merge_events(registered_events));
      else
        Runtime::trigger_event(local_registered);
      if (!applied_events.empty())
        Runtime::trigger_event(
            local_applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(local_applied);
      if (expr->remove_nested_expression_reference(did))
        delete expr;
      delete trace_info;
      // Remove the valid reference that we added for registration
      AutoLock v_lock(view_lock);
      legion_assert(valid_references > 0);
#ifdef LEGION_DEBUG_GC
      if ((--valid_references) == 0)
#else
      if (valid_references.fetch_sub(1) == 1)
#endif
        notify_invalid();
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_collective_fill(
        FillView* fill_view, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* fill_expression, Operation* op,
        const unsigned index, const IndexSpace match_space,
        const size_t op_context_index, const FieldMask& fill_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent ready_event,
        AddressSpaceID origin, const bool fill_restricted)
    //--------------------------------------------------------------------------
    {
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      legion_assert((op != nullptr) || !fill_restricted);
      CollectiveAnalysis* first_local_analysis = nullptr;
      if (!fill_restricted && ((op == nullptr) || trace_info.recording))
      {
        // If this is not a fill-out to a restricted collective instance
        // then we should be able to find our local analyses to use for
        // performing operations
        first_local_analysis = local_views.front()->find_collective_analysis(
            op_context_index, index, match_space);
        legion_assert(first_local_analysis != nullptr);
        if (op == nullptr)
        {
          op = first_local_analysis->get_operation();
          // Don't need the analysis anymore if we're not tracing
          if (!trace_info.recording)
            first_local_analysis = nullptr;
        }
      }
      legion_assert(op != nullptr);
      const PhysicalTraceInfo& local_info =
          (first_local_analysis == nullptr) ?
              trace_info :
              first_local_analysis->get_trace_info();
      legion_assert(local_info.recording == trace_info.recording);
      // Send it on to any children in the broadcast tree first
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      std::vector<ApEvent> ready_events;
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeFill rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(fill_view->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          fill_expression->pack_expression(rez, child);
          rez.serialize<bool>(fill_restricted);
          if (fill_restricted)
            op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(op_context_index);
          rez.serialize(fill_mask);
          local_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            if (ready_event.exists() && !trace_barrier.exists())
            {
              trace_shard = local_info.record_barrier_creation(
                  trace_barrier, children.size());
              ready_events.emplace_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            if (trace_barrier.exists())
              rez.serialize(trace_shard);
          }
          else
          {
            ApUserEvent child_ready;
            if (ready_event.exists())
            {
              child_ready = Runtime::create_ap_user_event(&local_info);
              ready_events.emplace_back(child_ready);
            }
            rez.serialize(child_ready);
          }
          rez.serialize(origin);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Now we can perform the fills for our instances
      const UniqueID op_id = op->get_unique_op_id();
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        const PhysicalTraceInfo& inst_info =
            (first_local_analysis == nullptr) ?
                trace_info :
                local_views[idx]
                    ->find_collective_analysis(
                        op_context_index, index, match_space)
                    ->get_trace_info();
        IndividualView* local_view = local_views[idx];
        ApEvent dst_precondition = local_view->find_copy_preconditions(
            false /*reading*/, 0 /*redop*/, fill_mask, fill_expression, op_id,
            index, applied_events, inst_info);
        if (dst_precondition.exists())
        {
          if (precondition.exists())
            dst_precondition = Runtime::merge_events(
                &inst_info, precondition, dst_precondition);
        }
        else
          dst_precondition = precondition;
        PhysicalManager* local_manager = local_view->get_manager();
        std::vector<CopySrcDstField> dst_fields;
        local_manager->compute_copy_offsets(fill_mask, dst_fields);
        const ApEvent result = fill_view->issue_fill(
            op, fill_expression, local_view, fill_mask, inst_info, dst_fields,
            applied_events, local_manager, dst_precondition, predicate_guard,
            COLLECTIVE_FILL, fill_restricted);
        if (result.exists())
        {
          if (ready_event.exists())
            ready_events.emplace_back(result);
          local_view->add_copy_user(
              false /*reading*/, 0 /*redop*/, result, fill_mask,
              fill_expression, match_space, op_id, index, recorded_events,
              inst_info.recording, runtime->address_space);
        }
      }
      // Use the trace info for doing the trigger if necessary
      if (!ready_events.empty())
      {
        legion_assert(ready_event.exists());
        Runtime::trigger_event(
            ready_event, Runtime::merge_events(&local_info, ready_events),
            trace_info, applied_events);
      }
      else if (ready_event.exists())
        Runtime::trigger_event(
            ready_event, ApEvent::NO_AP_EVENT, trace_info, applied_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeFill::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did, fill_did;
      derez.deserialize(view_did);
      RtEvent view_ready, fill_ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      derez.deserialize(fill_did);
      FillView* fill_view = static_cast<FillView*>(
          runtime->find_or_request_logical_view(fill_did, fill_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* fill_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      bool fill_restricted;
      derez.deserialize<bool>(fill_restricted);
      Operation* op = nullptr;
      std::set<RtEvent> ready_events;
      if (fill_restricted)
        op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask fill_mask;
      derez.deserialize(fill_mask);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        if (bar.exists())
        {
          ShardID sid;
          derez.deserialize(sid);
          // Copy-elmination will take care of this for us
          // when the trace is optimized
          ready = Runtime::create_ap_user_event(&trace_info);
          runtime->phase_barrier_arrive(bar, 1 /*count*/, ready);
          trace_info.record_barrier_arrival(
              bar, ready, 1 /*count*/, applied_events, sid);
        }
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);

      // Make sure all the distributed collectables are ready
      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (fill_ready.exists() && !fill_ready.has_triggered())
        ready_events.insert(fill_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      view->perform_collective_fill(
          fill_view, precondition, predicate_guard, fill_expression, op, index,
          match_space, op_ctx_index, fill_mask, trace_info, recorded_events,
          applied_events, ready, origin, fill_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != nullptr)
        delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent CollectiveView::perform_collective_point(
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, ApEvent precondition,
        PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, Operation* op, const unsigned index,
        const FieldMask& copy_mask, const FieldMask& dst_mask,
        const Memory location, const UniqueInst& dst_inst,
        const LgEvent dst_unique_event, const DistributedID src_inst_did,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, CollectiveKind collective)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      // Figure out which instance we're going to use for the copy
      unsigned instance_index = 0;
      if (src_inst_did > 0)
      {
        instance_index = std::numeric_limits<unsigned>::max();
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          PhysicalManager* manager = local_views[idx]->get_manager();
          if (manager->did != src_inst_did)
            continue;
          instance_index = idx;
          break;
        }
        legion_assert(instance_index != std::numeric_limits<unsigned>::max());
      }
      else if (instances.size() > 1)
      {
        int best_bandwidth = -1;
        const Machine& machine = runtime->machine;
        Machine::AffinityDetails details;
        if (machine.has_affinity(
                location, local_views[0]->get_manager()->memory_manager->memory,
                &details))
          best_bandwidth = details.bandwidth;
        for (unsigned idx = 1; idx < local_views.size(); idx++)
        {
          const Memory memory =
              local_views[idx]->get_manager()->memory_manager->memory;
          if (machine.has_affinity(location, memory, &details))
          {
            if ((best_bandwidth < 0) ||
                (int(details.bandwidth) > best_bandwidth))
            {
              best_bandwidth = details.bandwidth;
              instance_index = idx;
            }
          }
        }
      }
      // Compute the src_fields
      IndividualView* local_view = local_views[instance_index];
      // Compute the source precondition to get that in flight
      const UniqueID op_id = op->get_unique_op_id();
      const ApEvent src_pre = local_view->find_copy_preconditions(
          true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, trace_info);
      if (src_pre.exists())
      {
        if (precondition.exists())
          precondition =
              Runtime::merge_events(&trace_info, precondition, src_pre);
        else
          precondition = src_pre;
      }
      PhysicalManager* local_manager = local_view->get_manager();
      std::vector<CopySrcDstField> src_fields;
      local_manager->compute_copy_offsets(copy_mask, src_fields);
      // Issue the copy
      const ApEvent copy_post = copy_expression->issue_copy(
          op, trace_info, dst_fields, src_fields, reservations,
          local_manager->tree_id, dst_inst.tid, precondition, predicate_guard,
          local_manager->get_unique_event(), dst_unique_event, collective,
          false /*copy restricted*/);
      // Record the user
      if (copy_post.exists())
        local_view->add_copy_user(
            true /*reading*/, 0 /*redop*/, copy_post, copy_mask,
            copy_expression, upper_bound, op_id, index, recorded_events,
            trace_info.recording, runtime->address_space);
      if (trace_info.recording)
      {
        const UniqueInst src_inst(local_view);
        trace_info.record_copy_insts(
            copy_post, copy_expression, src_inst, dst_inst, copy_mask, dst_mask,
            get_redop(), applied_events);
      }
      return copy_post;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributePoint::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent view_ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      std::vector<CopySrcDstField> dst_fields;
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      LgEvent dst_unique_event = CollectiveView::unpack_fields(
          dst_fields, derez, ready_events, view, view_ready);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexSpace upper_bound;
      derez.deserialize(upper_bound);
      Operation* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      Memory location;
      derez.deserialize(location);
      UniqueInst dst_inst;
      dst_inst.deserialize(derez);
      DistributedID src_inst_did;
      derez.deserialize(src_inst_did);
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      CollectiveKind collective_kind;
      derez.deserialize(collective_kind);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      derez.deserialize(ready);

      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      const ApEvent result = view->perform_collective_point(
          dst_fields, reservations, precondition, predicate_guard,
          copy_expression, upper_bound, op, index, copy_mask, dst_mask,
          location, dst_inst, dst_unique_event, src_inst_did, trace_info,
          recorded_events, applied_events, collective_kind);

      Runtime::trigger_event(ready, result, trace_info, applied_events);
      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_collective_broadcast(
        const std::vector<CopySrcDstField>& src_fields, ApEvent precondition,
        PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
        Operation* op, const unsigned index, const IndexSpace match_space,
        const size_t op_ctx_index, const FieldMask& copy_mask,
        const UniqueInst& src_inst, const LgEvent src_unique_event,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent copy_done,
        ApUserEvent all_done, ApBarrier all_bar, ShardID owner_shard,
        AddressSpaceID origin, const bool copy_restricted,
        const CollectiveKind collective_kind)
    //--------------------------------------------------------------------------
    {
      legion_assert(copy_done.exists());
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      legion_assert((op != nullptr) || !copy_restricted);
      CollectiveAnalysis* first_local_analysis = nullptr;
      if (!copy_restricted && ((op == nullptr) || trace_info.recording))
      {
        // If this is not a copy-out to a restricted collective instance
        // then we should be able to find our local analyses to use for
        // performing operations
        first_local_analysis = local_views.front()->find_collective_analysis(
            op_ctx_index, index, match_space);
        legion_assert(first_local_analysis != nullptr);
        if (op == nullptr)
        {
          op = first_local_analysis->get_operation();
          // Don't need the analysis anymore if we're not tracing
          if (!trace_info.recording)
            first_local_analysis = nullptr;
        }
      }
      legion_assert(op != nullptr);
      const PhysicalTraceInfo& local_info =
          (first_local_analysis == nullptr) ?
              trace_info :
              first_local_analysis->get_trace_info();
      const UniqueID op_id = op->get_unique_op_id();
      // Do the copy to our local instance first
      IndividualView* local_view = local_views.front();
      ApEvent local_pre = local_view->find_copy_preconditions(
          false /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, local_info);
      // Get the precondition for the local copy
      if (precondition.exists())
      {
        if (local_pre.exists())
          local_pre =
              Runtime::merge_events(&local_info, precondition, local_pre);
        else
          local_pre = precondition;
      }
      // Get the dst_fields for performing the local broadcasts
      std::vector<CopySrcDstField> local_fields;
      PhysicalManager* local_manager = local_view->get_manager();
      local_manager->compute_copy_offsets(copy_mask, local_fields);
      const std::vector<Reservation> no_reservations;
      const ApEvent copy_post = copy_expression->issue_copy(
          op, local_info, local_fields, src_fields, no_reservations,
          src_inst.tid, local_manager->tree_id, local_pre, predicate_guard,
          src_unique_event, local_manager->get_unique_event(), collective_kind,
          copy_restricted);
      if (local_info.recording)
      {
        const UniqueInst dst_inst(local_view);
        local_info.record_copy_insts(
            copy_post, copy_expression, src_inst, dst_inst, copy_mask,
            copy_mask, 0 /*redop*/, applied_events);
      }
      Runtime::trigger_event(copy_done, copy_post, trace_info, applied_events);
      // Always record the writer to ensure later reads catch it
      local_view->add_copy_user(
          false /*reading*/, 0 /*redop*/, copy_post, copy_mask, copy_expression,
          match_space, op_id, index, recorded_events, local_info.recording,
          runtime->address_space);
      // Broadcast out the copy events to any children
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      // See if we're done
      if (children.empty() && (instances.size() == 1))
      {
        if (all_done.exists())
          Runtime::trigger_event(
              all_done, copy_post, trace_info, applied_events);
        return;
      }
      perform_local_broadcast(
          local_view, local_fields, children, first_local_analysis,
          precondition, predicate_guard, copy_expression, op, index,
          match_space, op_ctx_index, copy_mask, src_inst, src_unique_event,
          local_info, recorded_events, applied_events, all_done, all_bar,
          owner_shard, origin, copy_restricted, collective_kind);
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_local_broadcast(
        IndividualView* local_view,
        const std::vector<CopySrcDstField>& local_fields,
        const std::vector<AddressSpaceID>& children,
        CollectiveAnalysis* first_local_analysis, ApEvent precondition,
        PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
        Operation* op, const unsigned index, const IndexSpace match_space,
        const size_t op_ctx_index, const FieldMask& copy_mask,
        const UniqueInst& src_inst, const LgEvent src_unique_event,
        const PhysicalTraceInfo& local_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent all_done,
        ApBarrier all_bar, ShardID owner_shard, AddressSpaceID origin,
        const bool copy_restricted, const CollectiveKind collective_kind)
    //--------------------------------------------------------------------------
    {
      const UniqueID op_id = op->get_unique_op_id();
      const PhysicalManager* local_manager = local_view->get_manager();
      ApEvent local_pre = local_view->find_copy_preconditions(
          true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, local_info);
      ApBarrier broadcast_bar;
      ShardID broadcast_shard = 0;
      std::vector<ApEvent> read_events, done_events;
      const UniqueInst local_inst(local_view);
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeBroadcast rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          pack_fields(rez, local_fields, local_manager->get_unique_event());
          local_inst.serialize(rez);
          rez.serialize(local_pre);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, child);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          local_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            if (!broadcast_bar.exists())
            {
              broadcast_shard = local_info.record_barrier_creation(
                  broadcast_bar, children.size());
              read_events.emplace_back(broadcast_bar);
            }
            rez.serialize(broadcast_bar);
            rez.serialize(broadcast_shard);
            rez.serialize(all_bar);
            if (all_bar.exists())
              rez.serialize(owner_shard);
          }
          else
          {
            const ApUserEvent broadcast =
                Runtime::create_ap_user_event(&local_info);
            rez.serialize(broadcast);
            read_events.emplace_back(broadcast);
            ApUserEvent done;
            if (all_done.exists())
            {
              done = Runtime::create_ap_user_event(&local_info);
              done_events.emplace_back(done);
            }
            rez.serialize(done);
          }
          rez.serialize(origin);
          rez.serialize(collective_kind);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Now broadcast out to the rest of our local instances
      broadcast_local(
          local_manager, 0, op, index, copy_expression, match_space, copy_mask,
          local_pre, predicate_guard, local_fields, local_inst, local_info,
          collective_kind, read_events, recorded_events, applied_events,
          false /*has preconditions*/, (first_local_analysis != nullptr),
          op_ctx_index);
      if (!read_events.empty())
      {
        ApEvent read_done = Runtime::merge_events(&local_info, read_events);
        if (read_done.exists())
        {
          local_view->add_copy_user(
              true /*reading*/, 0 /*redop*/, read_done, copy_mask,
              copy_expression, match_space, op_id, index, recorded_events,
              local_info.recording, runtime->address_space);
          if (all_bar.exists() || all_done.exists())
            done_events.emplace_back(all_done);
        }
      }
      if (all_bar.exists())
      {
        ApEvent arrival;
        if (!done_events.empty())
          arrival = Runtime::merge_events(&local_info, done_events);
        runtime->phase_barrier_arrive(all_bar, 1 /*count*/, arrival);
        local_info.record_barrier_arrival(
            all_bar, arrival, 1 /*count*/, applied_events, owner_shard);
      }
      else if (all_done.exists())
      {
        Runtime::trigger_event(
            all_done, Runtime::merge_events(&local_info, done_events),
            local_info, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveView::broadcast_local(
        const PhysicalManager* src_manager, const unsigned src_index,
        Operation* op, const unsigned index,
        IndexSpaceExpression* copy_expression, IndexSpace match_space,
        const FieldMask& copy_mask, ApEvent precondition,
        PredEvent predicate_guard,
        const std::vector<CopySrcDstField>& src_fields,
        const UniqueInst& src_inst, const PhysicalTraceInfo& trace_info,
        const CollectiveKind collective_kind,
        std::vector<ApEvent>& destination_events,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        const bool has_instance_events, const bool first_local_analysis,
        const size_t op_ctx_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      legion_assert(
          !has_instance_events ||
          (destination_events.size() == local_views.size()));
      if (local_views.size() == 1)
        return;
      const UniqueID op_id = op->get_unique_op_id();
      const std::vector<Reservation> no_reservations;
      if (multiple_local_memories)
      {
        // If there are multiple local instances on this node, then
        // we need to get a spanning tree to use for issuing the
        // broadcast copies across the local views
        const std::vector<std::pair<unsigned, unsigned> >& spanning_copies =
            find_spanning_broadcast_copies(src_index);
        unsigned destination_events_offset = 0;
        if (!has_instance_events)
        {
          destination_events_offset = destination_events.size();
          destination_events.resize(
              destination_events_offset + local_views.size(),
              ApEvent::NO_AP_EVENT);
        }
        ApEvent* local_events = &destination_events[destination_events_offset];
        local_events[src_index] = precondition;
        std::vector<std::vector<CopySrcDstField> > local_fields(
            local_views.size());
        local_fields[src_index] = src_fields;
        // Forward order copies <source,destination>
        for (const std::pair<unsigned, unsigned>& it : spanning_copies)
        {
          legion_assert(it.first != it.second);
          legion_assert(it.second != src_index);
          legion_assert(
              has_instance_events || !local_events[it.second].exists());
          legion_assert(!local_fields[it.first].empty());
          IndividualView* local_view = local_views[it.first];
          PhysicalManager* local_manager = local_view->get_manager();
          IndividualView* dst_view = local_views[it.second];
          PhysicalManager* dst_manager = dst_view->get_manager();
          dst_manager->compute_copy_offsets(copy_mask, local_fields[it.second]);
          const PhysicalTraceInfo& inst_info =
              !first_local_analysis ? trace_info :
                                      local_views[it.second]
                                          ->find_collective_analysis(
                                              op_ctx_index, index, match_space)
                                          ->get_trace_info();
          ApEvent dst_pre =
              has_instance_events ?
                  local_events[it.second] :
                  dst_view->find_copy_preconditions(
                      false /*reading*/, 0 /*redop*/, copy_mask,
                      copy_expression, op_id, index, applied_events, inst_info);
          // Merge in the source precondition
          if (local_events[it.first].exists())
          {
            if (dst_pre.exists())
              dst_pre = Runtime::merge_events(
                  &inst_info, dst_pre, local_events[it.first]);
            else
              dst_pre = local_events[it.first];
          }
          const ApEvent dst_post = copy_expression->issue_copy(
              op, inst_info, local_fields[it.second], local_fields[it.first],
              no_reservations, local_manager->tree_id, dst_manager->tree_id,
              dst_pre, predicate_guard, local_manager->get_unique_event(),
              dst_manager->get_unique_event(), collective_kind,
              false /*copy restricted*/);
          if (dst_post.exists())
          {
            // Keep the reads in order to to prevent contention on
            // egress bandwidth since these will mostly be on switched
            // networks like the front-side bus or NVLink
            local_events[it.first] = dst_post;
            local_events[it.second] = dst_post;
          }
          if (inst_info.recording)
          {
            const UniqueInst local_inst(local_view);
            const UniqueInst dst_inst(dst_view);
            inst_info.record_copy_insts(
                dst_post, copy_expression, local_inst, dst_inst, copy_mask,
                copy_mask, 0 /*redop*/, applied_events);
          }
        }
        // Go through and save the results on the views
        for (unsigned idx = 0; idx < local_views.size(); idx++)
          if ((idx != src_index) && local_events[idx].exists())
            local_views[idx]->add_copy_user(
                false /*reading*/, 0 /*redop*/, local_events[idx], copy_mask,
                copy_expression, match_space, op_id, index, recorded_events,
                trace_info.recording, runtime->address_space);
        if (!has_instance_events)
        {
          // Prune out the empty entries
          for (std::vector<ApEvent>::iterator it = destination_events.begin();
               it != destination_events.end();
               /*nothing*/)
            if (it->exists())
              it++;
            else
              it = destination_events.erase(it);
        }
      }
      else
      {
        // If all the local instances are in the same memory then we
        // might as well just issue copies from the source to all the
        // memories since they'll all be fighting over the same
        // bandwidth anyway for the copies to be performed
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          if (idx == src_index)
            continue;
          IndividualView* dst_view = local_views[idx];
          PhysicalManager* dst_manager = dst_view->get_manager();
          std::vector<CopySrcDstField> dst_fields;
          dst_manager->compute_copy_offsets(copy_mask, dst_fields);
          const PhysicalTraceInfo& inst_info =
              !first_local_analysis ? trace_info :
                                      local_views[idx]
                                          ->find_collective_analysis(
                                              op_ctx_index, index, match_space)
                                          ->get_trace_info();
          ApEvent dst_pre =
              has_instance_events ?
                  destination_events[idx] :
                  dst_view->find_copy_preconditions(
                      false /*reading*/, 0 /*redop*/, copy_mask,
                      copy_expression, op_id, index, applied_events, inst_info);
          if (precondition.exists())
          {
            if (dst_pre.exists())
              dst_pre =
                  Runtime::merge_events(&inst_info, dst_pre, precondition);
            else
              dst_pre = precondition;
          }
          const ApEvent dst_post = copy_expression->issue_copy(
              op, inst_info, dst_fields, src_fields, no_reservations,
              src_manager->tree_id, dst_manager->tree_id, dst_pre,
              predicate_guard, src_manager->get_unique_event(),
              dst_manager->get_unique_event(), collective_kind,
              false /*copy restricted*/);
          if (dst_post.exists())
          {
            if (has_instance_events)
              destination_events[idx] = dst_post;
            else
              destination_events.emplace_back(dst_post);
            dst_view->add_copy_user(
                false /*reading*/, 0 /*redop*/, dst_post, copy_mask,
                copy_expression, match_space, op_id, index, recorded_events,
                trace_info.recording, runtime->address_space);
          }
          if (inst_info.recording)
          {
            const UniqueInst dst_inst(dst_view);
            inst_info.record_copy_insts(
                dst_post, copy_expression, src_inst, dst_inst, copy_mask,
                copy_mask, 0 /*redop*/, applied_events);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    const std::vector<std::pair<unsigned, unsigned> >&
        CollectiveView::find_spanning_broadcast_copies(unsigned root_index)
    //--------------------------------------------------------------------------
    {
      // Check to see if we already have an answer
      {
        AutoLock v_lock(view_lock, false /*exclusive*/);
        std::map<unsigned, std::vector<std::pair<unsigned, unsigned> > >::
            const_iterator finder = spanning_copies.find(root_index);
        if (finder != spanning_copies.end())
          return finder->second;
      }
      // We don't have an answer so we need to compute one
      // First find all the memories and track the first instance set in them
      std::map<Memory, unsigned> first_in_memory;
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        Memory memory = local_views[idx]->get_manager()->memory_manager->memory;
        std::map<Memory, unsigned>::iterator finder =
            first_in_memory.find(memory);
        if (finder != first_in_memory.end())
        {
          if (idx == root_index)
            finder->second = root_index;
        }
        else
        {
          if (idx == root_index)
            first_in_memory[memory] = root_index;
          else
            first_in_memory[memory] =
                std::numeric_limits<unsigned>::max();  // not set yet
        }
      }
      legion_assert(first_in_memory.size() > 1);  // should be multiple memories
      const size_t total_memories = first_in_memory.size();
      // Figure out what the index is of the root memory
      std::map<Memory, unsigned>::iterator root_finder = first_in_memory.find(
          local_views[root_index]->get_manager()->memory_manager->memory);
      // Better be able to find it
      legion_assert(root_finder != first_in_memory.end());
      const unsigned root_memory_index =
          std::distance(first_in_memory.begin(), root_finder);
      // Next construct an adjacency matrix between the memories with edges
      // assigned costs of 1/bandwidth so that higher bandwidth gives a
      // lower edge cost. For connectivity if any memories have no path from
      // the root we'll give them a cost of 2 for multi-hop copies
      // Any cost less than zero is considered a missing edge
      std::vector<float> adjacency_matrix(
          total_memories * total_memories, -1.f);
      const bool same_bandwidth = construct_spanning_adjacency_matrix(
          root_memory_index, first_in_memory, adjacency_matrix);
      // Check for the special case here where all the edges have the
      // same bandwidth in which case we can do this with BFS.
      // The case of all having the same bandwidth happens with switched
      // systems like the front-side bus and NVLink with NVSwitch.
      // We represent the spanning tree using a vector that maps each node
      // to it's "parent" node (e.g. the one that produces it) in the
      // spanning tree (the root has no parent)
      std::vector<unsigned> previous(
          total_memories, std::numeric_limits<unsigned>::max());
      if (same_bandwidth)
        compute_spanning_tree_same_bandwidth(
            root_memory_index, adjacency_matrix, previous, first_in_memory);
      else
        compute_spanning_tree_diff_bandwidth(
            root_memory_index, adjacency_matrix, previous, first_in_memory);
      // Next we compute the actual order of spanning copies, we do this
      // by first computing a depth-first search to determine which
      // children have the deepest sub-trees so that we can use that to
      // say what order children should be traversed in from each node
      std::vector<std::pair<unsigned, bool> > dfs_stack;
      dfs_stack.emplace_back(
          std::pair<unsigned, bool>(root_memory_index, true));
      std::vector<unsigned> max_subtree_depth(
          total_memories, std::numeric_limits<unsigned>::max());
      while (!dfs_stack.empty())
      {
        std::pair<unsigned, bool>& next = dfs_stack.back();
        if (next.second)
        {
          // Pre-traversal
          next.second = false;
          // Need the value because the reference might be invalidated
          // by the vector changing size when we push children onto it
          const unsigned current = next.first;
          // Push all of our children onto the stack
          for (unsigned child = 0; child < total_memories; child++)
            if (previous[child] == current)
              dfs_stack.emplace_back(std::pair<unsigned, bool>(child, true));
        }
        else
        {
          // Post-traversal
          // Compute our max sub-tree
          unsigned max_depth = 0;
          for (unsigned child = 0; child < total_memories; child++)
          {
            if (previous[child] != next.first)
              continue;
            legion_assert(
                max_subtree_depth[child] !=
                std::numeric_limits<unsigned>::max());
            unsigned depth = max_subtree_depth[child] + 1;
            if (max_depth < depth)
              max_depth = depth;
          }
          max_subtree_depth[next.first] = max_depth;
          dfs_stack.pop_back();
        }
      }
      // Now we've got the max depth of each sub-tree, so we can compute
      // the order of the spanning copies from each node to maximize getting
      // the ones with the maximum depth out first, need to do this with bfs
      // <memory index,view index>
      std::deque<std::pair<unsigned, unsigned> > bfs_queue;
      bfs_queue.emplace_back(std::make_pair(root_memory_index, root_index));
      std::vector<std::pair<unsigned, unsigned> > spanning;
      while (!bfs_queue.empty())
      {
        const std::pair<unsigned, unsigned>& next = bfs_queue.front();
        // Track the <depth,child> pairs so we can sort them
        // and then traverse them in order
        std::vector<std::pair<unsigned, unsigned> > child_depths;
        unsigned child = 0;
        for (const std::pair<const Memory, unsigned>& it : first_in_memory)
        {
          if (previous[child] != next.first)
          {
            child++;
            continue;
          }
          legion_assert(it.second != std::numeric_limits<unsigned>::max());
          child_depths.emplace_back(
              std::make_pair(max_subtree_depth[child], it.second));
          // Add the child to the queue to traverse next
          bfs_queue.emplace_back(std::make_pair(child++, it.second));
        }
        if (!child_depths.empty())
        {
          std::sort(child_depths.begin(), child_depths.end());
          // Reverse order traverse so we do the max depth ones first
          for (std::vector<std::pair<unsigned, unsigned> >::reverse_iterator
                   it = child_depths.rbegin();
               it != child_depths.rend(); it++)
            // Add it to the spanning
            spanning.emplace_back(std::make_pair(next.second, it->second));
        }
        bfs_queue.pop_front();
      }
      // Should have a copy into every memory except the root one
      legion_assert(spanning.size() == (total_memories - 1));
      if (total_memories < local_views.size())
      {
        // Record copies to all instances that share memories
        // from the first instance to that memory. The motivation
        // for this is that intra-memory bandwidth should always
        // be much higher than bandwidth between different memories
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          const Memory memory =
              local_views[idx]->get_manager()->memory_manager->memory;
          std::map<Memory, unsigned>::const_iterator finder =
              first_in_memory.find(memory);
          legion_assert(finder != first_in_memory.end());
          legion_assert(finder->second < local_views.size());
          // If this view is not the first one to be copied to in
          // this memory then we need to issue a copy from the
          // first one to be assigned in this memory
          if (finder->second != idx)
            spanning.emplace_back(
                std::pair<unsigned, unsigned>(finder->second, idx));
        }
      }
      // Should have a copy into every view except the root one
      legion_assert(spanning.size() == (local_views.size() - 1));
      // Save the result if it doesn't exist yet
      AutoLock v_lock(view_lock);
      std::vector<std::pair<unsigned, unsigned> >& result =
          spanning_copies[root_index];
      // Check to see if we're the first ones to get here
      if (result.empty())
        result.swap(spanning);
      return result;
    }

    //--------------------------------------------------------------------------
    bool CollectiveView::construct_spanning_adjacency_matrix(
        unsigned root_index, const std::map<Memory, unsigned>& first_in_memory,
        std::vector<float>& adjacency_matrix) const
    //--------------------------------------------------------------------------
    {
      const size_t total_memories = first_in_memory.size();
      std::vector<Realm::Machine::MemoryMemoryAffinity> affinity(1);
      unsigned row = 0, same_bandwidth = 0;
      for (std::map<Memory, unsigned>::const_iterator it1 =
               first_in_memory.begin();
           it1 != first_in_memory.end(); it1++, row++)
      {
        unsigned col = row + 1;
        for (std::map<Memory, unsigned>::const_iterator it2 = std::next(it1);
             it2 != first_in_memory.end(); it2++, col++)
        {
          unsigned count = runtime->machine.get_mem_mem_affinity(
              affinity, it1->first, it2->first);
          if (count == 0)
            continue;
          legion_assert(count == 1);
          legion_assert(affinity.front().bandwidth > 0);
          legion_assert(
              affinity.front().bandwidth <
              std::numeric_limits<unsigned>::max());
          unsigned bandwidth = affinity.front().bandwidth;
          float cost = 1.f / bandwidth;
          // Assume symmetric bandwidth here
          adjacency_matrix[row * total_memories + col] = cost;
          adjacency_matrix[col * total_memories + row] = cost;
          // Keep track of whether we are the same bandwidth everywhere
          if (same_bandwidth != std::numeric_limits<unsigned>::max())
          {
            if (same_bandwidth == 0)  // First time we've seen any bandwidth
              same_bandwidth = bandwidth;
            else if (same_bandwidth != bandwidth)  // Check if they are the same
              same_bandwidth = std::numeric_limits<unsigned>::max();
          }
        }
      }
      // Check to see if we can reach all the memories and if not put in
      // edges between all the reachable and all the non-reachable memories
      // to represent potential multi-hop copies
      std::vector<bool> reachable(total_memories, false);
      reachable[root_index] = true;
      std::vector<unsigned> dfs_stack(1, root_index);
      unsigned total_reachable = 1;
      while (!dfs_stack.empty())
      {
        unsigned next = dfs_stack.back();
        dfs_stack.pop_back();
        for (unsigned idx = 0; idx < total_memories; idx++)
        {
          if (idx == next)
            continue;
          // Check if there is an edge to the next memory
          if (adjacency_matrix[next * total_memories + idx] < 0.f)
            continue;
          // See if this child is already reachable
          if (reachable[idx])
            continue;
          // Mark it as reachable and add it to the stack
          reachable[idx] = true;
          total_reachable++;
          dfs_stack.emplace_back(idx);
        }
      }
      legion_assert(total_reachable <= total_memories);
      // Handle the case where not all the memories are reachable with
      // direct copies from the source memory
      if (total_reachable < total_memories)
      {
        if (same_bandwidth == 0)
        {
          // Did not have any edges!
          // They will all have the same bandwidth in this case
          same_bandwidth = 1;
        }
        else if (same_bandwidth != std::numeric_limits<unsigned>::max())
        {
          // No longer true that they all have the same bandwidth
          same_bandwidth = std::numeric_limits<unsigned>::max();
        }
        // Go through and add in multi-hop copy edges with cost 2
        for (unsigned row = 0; row < total_memories; row++)
        {
          if (reachable[row])
            continue;
          for (unsigned col = 0; col < total_memories; col++)
          {
            if ((row == col) || !reachable[col])
              continue;
            adjacency_matrix[row * total_memories + col] = 2.f;
            adjacency_matrix[col * total_memories + row] = 2.f;
          }
        }
      }
      legion_assert(same_bandwidth != 0);
      return (same_bandwidth != std::numeric_limits<unsigned>::max());
    }

    //--------------------------------------------------------------------------
    void CollectiveView::compute_spanning_tree_same_bandwidth(
        unsigned root_index, const std::vector<float>& adjacency_matrix,
        std::vector<unsigned>& previous,
        std::map<Memory, unsigned>& first_in_memory) const
    //--------------------------------------------------------------------------
    {
      // All edge have the same bandwidth
      // Run BFS but only have each node add one child each time they
      // get pulled off the queue until they have exhausted their children
      // so that we can order copies for maximum bandwidth
      const size_t total_memories = first_in_memory.size();
      // <current node,next child to search>
      std::deque<std::pair<unsigned, unsigned> > bfs_queue;
      bfs_queue.emplace_back(std::pair<unsigned, unsigned>(root_index, 0));
      while (!bfs_queue.empty())
      {
        const std::pair<unsigned, unsigned>& next = bfs_queue.front();
        for (unsigned child = next.second; child < total_memories; child++)
        {
          // Skip going to ourself or back to the root
          if ((child == next.first) || (child == root_index))
            continue;
          // Check to see if the next child is already reached
          if (previous[child] != std::numeric_limits<unsigned>::max())
            continue;
          // Check to see if we have an edge to the next child
          if (adjacency_matrix[next.first * total_memories + child] < 0.f)
            continue;
          // Find the first local view in this memory
          std::map<Memory, unsigned>::iterator finder = first_in_memory.begin();
          std::advance(finder, child);
          legion_assert(finder->second == std::numeric_limits<unsigned>::max());
          for (unsigned idx = 0; idx < local_views.size(); idx++)
          {
            if (local_views[idx]->get_manager()->memory_manager->memory !=
                finder->first)
              continue;
            finder->second = idx;
            break;
          }
          legion_assert(finder->second != std::numeric_limits<unsigned>::max());
          // Record it in the spanning
          previous[child] = next.first;
          // Add it the child to list to search
          bfs_queue.emplace_back(std::pair<unsigned, unsigned>(child, 0));
          // Add ourself back on the list if there are more children to search
          if (++child < total_memories)
            bfs_queue.emplace_back(
                std::pair<unsigned, unsigned>(next.first, child));
          break;
        }
        bfs_queue.pop_front();
      }
    }

    //--------------------------------------------------------------------------
    void CollectiveView::compute_spanning_tree_diff_bandwidth(
        unsigned root_index, const std::vector<float>& adjacency_matrix,
        std::vector<unsigned>& previous,
        std::map<Memory, unsigned>& first_in_memory) const
    //--------------------------------------------------------------------------
    {
      const size_t total_memories = first_in_memory.size();
      // Use a greedy algorithm here to try to get as much parallelism
      // in copies going as quickly as possible by having each node
      // always choose the next child with the lowest cost to copy
      // to next and then continue the process
      // <current node,next child to search>
      std::deque<unsigned> bfs_queue;
      bfs_queue.emplace_back(root_index);
      while (!bfs_queue.empty())
      {
        const unsigned next = bfs_queue.front();
        bfs_queue.pop_front();
        // Iterate over all the children and find the next one that
        // hasn't been traversed with the lowest cost to get to
        unsigned total_children = 0;
        float lowest_cost = -1.f;
        unsigned lowest_child = std::numeric_limits<unsigned>::max();
        for (unsigned child = 0; child < total_memories; child++)
        {
          // Skip going to ourself or back to the root
          if ((child == next) || (child == root_index))
            continue;
          // Check to see if the next child is already reached
          if (previous[child] != std::numeric_limits<unsigned>::max())
            continue;
          // Check to see if we have an edge to the next child
          float cost = adjacency_matrix[next * total_memories + child];
          // No edge if cost is negative
          if (cost < 0.f)
            continue;
          // See if we're the first child or the lowest cost one
          if ((total_children++ == 0) || (cost < lowest_cost))
          {
            lowest_cost = cost;
            lowest_child = child;
          }
        }
        // If we have a next child then want to issue the copy it
        if (total_children > 0)
        {
          // Found a next child so record it
          // Find the first local view in this memory
          std::map<Memory, unsigned>::iterator finder = first_in_memory.begin();
          std::advance(finder, lowest_child);
          legion_assert(finder->second == std::numeric_limits<unsigned>::max());
          for (unsigned idx = 0; idx < local_views.size(); idx++)
          {
            if (local_views[idx]->get_manager()->memory_manager->memory !=
                finder->first)
              continue;
            finder->second = idx;
            break;
          }
          legion_assert(finder->second != std::numeric_limits<unsigned>::max());
          legion_assert(
              previous[lowest_child] == std::numeric_limits<unsigned>::max());
          // Record it in the spanning
          previous[lowest_child] = next;
          // Add the child to list to search
          bfs_queue.emplace_back(lowest_child);
          // If we still have more children we could copy to then
          // we put ourselves back on the list for the next round
          if (total_children > 1)
            bfs_queue.emplace_back(next);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeBroadcast::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent view_ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      std::vector<CopySrcDstField> src_fields;
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      LgEvent src_unique_event = CollectiveView::unpack_fields(
          src_fields, derez, ready_events, view, view_ready);
      UniqueInst src_inst;
      src_inst.deserialize(derez);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation* op = nullptr;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready, all_done;
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      if (trace_info.recording)
      {
        ApBarrier broadcast_bar;
        derez.deserialize(broadcast_bar);
        ShardID broadcast_shard;
        derez.deserialize(broadcast_shard);
        // Copy-elmination will take care of this for us
        // when the trace is optimized
        ready = Runtime::create_ap_user_event(&trace_info);
        runtime->phase_barrier_arrive(broadcast_bar, 1 /*count*/, ready);
        trace_info.record_barrier_arrival(
            broadcast_bar, ready, 1 /*count*/, applied_events, broadcast_shard);
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
      {
        derez.deserialize(ready);
        derez.deserialize(all_done);
      }
      AddressSpaceID origin;
      derez.deserialize(origin);
      CollectiveKind collective_kind;
      derez.deserialize(collective_kind);

      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      view->perform_collective_broadcast(
          src_fields, precondition, predicate_guard, copy_expression, op, index,
          match_space, op_ctx_index, copy_mask, src_inst, src_unique_event,
          trace_info, recorded_events, applied_events, ready, all_done, all_bar,
          owner_shard, origin, copy_restricted, collective_kind);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != nullptr)
        delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_collective_reducecast(
        ReductionView* source, const std::vector<CopySrcDstField>& src_fields,
        ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, Operation* op,
        const unsigned index, const IndexSpace match_space,
        const size_t op_ctx_index, const FieldMask& copy_mask,
        const UniqueInst& src_inst, const LgEvent src_unique_event,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent reduce_done,
        ApBarrier all_bar, ShardID owner_shard, AddressSpaceID origin,
        const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
      ReductionOpID src_redop = source->get_redop();
      legion_assert(src_redop > 0);
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      legion_assert((op != nullptr) || !copy_restricted);
      // Only one of these should be valid
      legion_assert(reduce_done.exists() != all_bar.exists());
      // If we have any children, broadcast this out to the first in parallel
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      std::vector<ApEvent> reduce_events;
      if (!children.empty() && !trace_info.recording)
      {
        // Help out with broadcasting the precondition event
        // In the tracing case the precondition is a barrier
        // so there's no need for us to do this
        const ApUserEvent local_precondition =
            Runtime::create_ap_user_event(&trace_info);
        Runtime::trigger_event(
            local_precondition, precondition, trace_info, applied_events);
        precondition = local_precondition;
      }
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeReducecast rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(source->did);
          source->pack_fields(rez, src_fields);
          src_inst.serialize(rez);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, child);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            rez.serialize(all_bar);
            rez.serialize(owner_shard);
          }
          else
          {
            const ApUserEvent reduced =
                Runtime::create_ap_user_event(&trace_info);
            rez.serialize(reduced);
            reduce_events.emplace_back(reduced);
          }
          rez.serialize(origin);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      CollectiveAnalysis* first_local_analysis = nullptr;
      if (!copy_restricted && ((op == nullptr) || trace_info.recording))
      {
        // If this is not a copy-out to a restricted collective instance
        // then we should be able to find our local analyses to use for
        // performing operations
        first_local_analysis = local_views.front()->find_collective_analysis(
            op_ctx_index, index, match_space);
        legion_assert(first_local_analysis != nullptr);
        if (op == nullptr)
        {
          op = first_local_analysis->get_operation();
          // Don't need the analysis anymore if we're not tracing
          if (!trace_info.recording)
            first_local_analysis = nullptr;
        }
      }
      legion_assert(op != nullptr);
      const PhysicalTraceInfo& local_info =
          (first_local_analysis == nullptr) ?
              trace_info :
              first_local_analysis->get_trace_info();
      const UniqueID op_id = op->get_unique_op_id();
      std::vector<ApEvent> local_done_events;
      std::vector<CopySrcDstField> local_fields;
      std::vector<Reservation> local_reservations;
      // Issue the reductions to our local instances
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        const PhysicalTraceInfo& inst_info =
            (first_local_analysis == nullptr) ?
                trace_info :
                local_views[idx]
                    ->find_collective_analysis(op_ctx_index, index, match_space)
                    ->get_trace_info();
        IndividualView* dst_view = local_views[idx];
        // Compute the reducing precondition for our local instances
        ApEvent reduce_pre = dst_view->find_copy_preconditions(
            false /*reading*/, src_redop, copy_mask, copy_expression, op_id,
            index, applied_events, inst_info);
        if (precondition.exists())
        {
          if (reduce_pre.exists())
            reduce_pre =
                Runtime::merge_events(&inst_info, precondition, reduce_pre);
          else
            reduce_pre = precondition;
        }
        PhysicalManager* dst_manager = dst_view->get_manager();
        dst_manager->compute_copy_offsets(copy_mask, local_fields);
        for (CopySrcDstField& field : local_fields)
          field.set_redop(src_redop, (get_redop() > 0), true /*exclusive*/);
        dst_view->find_field_reservations(copy_mask, local_reservations);
        const ApEvent reduce_done = copy_expression->issue_copy(
            op, inst_info, local_fields, src_fields, local_reservations,
            src_inst.tid, dst_manager->tree_id, reduce_pre, predicate_guard,
            src_unique_event, dst_manager->get_unique_event(),
            COLLECTIVE_REDUCECAST, copy_restricted);
        if (reduce_done.exists())
        {
          local_done_events.emplace_back(reduce_done);
          dst_view->add_copy_user(
              false /*reading*/, src_redop, reduce_done, copy_mask,
              copy_expression, match_space, op_id, index, recorded_events,
              inst_info.recording, runtime->address_space);
        }
        if (inst_info.recording)
        {
          const UniqueInst dst_inst(dst_view);
          inst_info.record_copy_insts(
              reduce_done, copy_expression, src_inst, dst_inst, copy_mask,
              copy_mask, src_redop, applied_events);
        }
        local_fields.clear();
        local_reservations.clear();
      }
      if (all_bar.exists())
      {
        ApEvent local_done;
        if (!local_done_events.empty())
          local_done = Runtime::merge_events(&local_info, local_done_events);
        runtime->phase_barrier_arrive(all_bar, 1 /*count*/, local_done);
        local_info.record_barrier_arrival(
            all_bar, local_done, 1 /*count*/, applied_events, owner_shard);
      }
      else
      {
        if (!local_done_events.empty())
          reduce_events.insert(
              reduce_events.end(), local_done_events.begin(),
              local_done_events.end());
        Runtime::trigger_event(
            reduce_done, Runtime::merge_events(&local_info, reduce_events),
            local_info, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeReducecast::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did, src_did;
      derez.deserialize(view_did);
      RtEvent view_ready, src_ready;
      CollectiveView* view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      derez.deserialize(src_did);
      ReductionView* src_view = static_cast<ReductionView*>(
          runtime->find_or_request_logical_view(src_did, src_ready));
      std::vector<CopySrcDstField> src_fields;
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      LgEvent src_unique_event = CollectiveView::unpack_fields(
          src_fields, derez, ready_events, view, view_ready);
      UniqueInst src_inst;
      src_inst.deserialize(derez);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation* op = nullptr;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      if (trace_info.recording)
      {
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);

      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (src_ready.exists() && !src_ready.has_triggered())
        ready_events.insert(src_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      view->perform_collective_reducecast(
          src_view, src_fields, precondition, predicate_guard, copy_expression,
          op, index, match_space, op_ctx_index, copy_mask, src_inst,
          src_unique_event, trace_info, recorded_events, applied_events, ready,
          all_bar, owner_shard, origin, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != nullptr)
        delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_collective_hourglass(
        AllreduceView* source, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, Operation* op,
        const unsigned index, const IndexSpace match_space,
        const FieldMask& copy_mask, const DistributedID src_inst_did,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent all_done,
        AddressSpaceID target, const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
      legion_assert(op != nullptr);
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(target));
      if (target != local_space)
      {
        // Send this to where the target address space is
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeHourglass rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(source->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(copy_mask);
          rez.serialize(src_inst_did);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          rez.serialize(all_done);
          rez.serialize(copy_restricted);
        }
        rez.dispatch(target);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
        return;
      }
      legion_assert(!instances.empty());
      const UniqueID op_id = op->get_unique_op_id();
      IndividualView* local_view = local_views.front();
      // Perform the collective reduction first on the source
      ApEvent reduce_pre = local_view->find_copy_preconditions(
          false /*reding*/, source->redop, copy_mask, copy_expression, op_id,
          index, applied_events, trace_info);
      if (precondition.exists())
      {
        if (reduce_pre.exists())
          reduce_pre =
              Runtime::merge_events(&trace_info, precondition, reduce_pre);
        else
          reduce_pre = precondition;
      }
      PhysicalManager* local_manager = local_view->get_manager();
      // We'll just use the first instance for the target
      std::vector<CopySrcDstField> local_fields;
      local_manager->compute_copy_offsets(copy_mask, local_fields);
      std::vector<Reservation> reservations;
      local_view->find_field_reservations(copy_mask, reservations);
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        local_fields[idx].set_redop(
            source->redop, false /*fold*/, true /*exclusive*/);
      // Build the reduction tree down to our first instance
      const AddressSpaceID origin =
          (src_inst_did > 0) ? runtime->determine_owner(src_inst_did) :
                               source->select_source_space(local_space);
      ApEvent reduced;
      const UniqueInst local_inst(local_view);
      // Note that there is something subtle going on here!
      // If the copy aggregator needs to issue multiple reduction copies
      // to this collective instance, they might all need to do an
      // hourglass or a reducecast case, to keep things correct with
      // the broadcast, we rely on the ordering of collective messages
      // going out from the same source node to all the other nodes to
      // ensure that copies are done in the right order
      if (origin != local_space)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeReduction rez;
        {
          RezCheck z(rez);
          rez.serialize(source->did);
          pack_fields(rez, local_fields, local_manager->get_unique_event());
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
          rez.serialize(reduce_pre);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, origin);
          rez.serialize(match_space);
          op->pack_remote_operation(rez, origin, applied_events);
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(copy_mask);
          rez.serialize(src_inst_did);
          local_inst.serialize(rez);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            ApBarrier bar;
            const ShardID sid =
                trace_info.record_barrier_creation(bar, 1 /*arrivals*/);
            rez.serialize(bar);
            rez.serialize(sid);
            reduced = bar;
          }
          else
          {
            const ApUserEvent to_trigger =
                Runtime::create_ap_user_event(&trace_info);
            rez.serialize(to_trigger);
            reduced = to_trigger;
          }
          rez.serialize(origin);
          rez.serialize(COLLECTIVE_HOURGLASS_ALLREDUCE);
        }
        rez.dispatch(origin);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      else
      {
        const ApUserEvent to_trigger =
            Runtime::create_ap_user_event(&trace_info);
        source->perform_collective_reduction(
            local_fields, reservations, reduce_pre, predicate_guard,
            copy_expression, match_space, op, index, copy_mask, copy_mask,
            src_inst_did, local_inst, local_manager->get_unique_event(),
            trace_info, COLLECTIVE_HOURGLASS_ALLREDUCE, recorded_events,
            applied_events, to_trigger, origin);
        reduced = to_trigger;
      }
      // Record the write
      if (reduced.exists())
        local_view->add_copy_user(
            false /*reading*/, source->redop, reduced, copy_mask,
            copy_expression, match_space, op_id, index, recorded_events,
            trace_info.recording, runtime->address_space);
      // Do the broadcast out, remove the redop from local fields
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        local_fields[idx].set_redop(0 /*redop*/, false /*fold*/);
      // Start with any children
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(local_space, local_space, children);
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      std::vector<ApEvent> all_done_events;
      if (!children.empty() || (local_views.size() > 1))
      {
        ApEvent broadcast_pre = local_view->find_copy_preconditions(
            true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        if (precondition.exists())
        {
          if (broadcast_pre.exists())
            broadcast_pre =
                Runtime::merge_events(&trace_info, precondition, broadcast_pre);
          else
            broadcast_pre = precondition;
        }
        ApBarrier broadcast_bar;
        ShardID broadcast_shard = 0;
        std::vector<ApEvent> broadcast_events;
        if (all_done.exists() && trace_info.recording)
        {
          const size_t arrivals = collective_mapping->size();
          owner_shard = trace_info.record_barrier_creation(all_bar, arrivals);
        }
        for (const AddressSpaceID& child : children)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          CollectiveDistributeBroadcast rez;
          {
            RezCheck z(rez);
            rez.serialize(this->did);
            pack_fields(rez, local_fields, local_manager->get_unique_event());
            local_inst.serialize(rez);
            rez.serialize(broadcast_pre);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, child);
            rez.serialize<bool>(copy_restricted);
            if (copy_restricted)
              op->pack_remote_operation(rez, origin, applied_events);
            rez.serialize(index);
            rez.serialize(match_space);
            rez.serialize(op->get_context_index());
            rez.serialize(copy_mask);
            trace_info.pack_trace_info(rez);
            rez.serialize(recorded);
            rez.serialize(applied);
            if (trace_info.recording)
            {
              if (!broadcast_bar.exists())
              {
                broadcast_shard = trace_info.record_barrier_creation(
                    broadcast_bar, children.size());
                broadcast_events.emplace_back(broadcast_bar);
              }
              rez.serialize(broadcast_bar);
              rez.serialize(broadcast_shard);
              rez.serialize(all_bar);
              if (all_bar.exists())
                rez.serialize(owner_shard);
            }
            else
            {
              const ApUserEvent done =
                  Runtime::create_ap_user_event(&trace_info);
              rez.serialize(done);
              broadcast_events.emplace_back(done);
              ApUserEvent all;
              if (all_done.exists())
              {
                all = Runtime::create_ap_user_event(&trace_info);
                all_done_events.emplace_back(all);
              }
              rez.serialize(all);
            }
            rez.serialize(origin);
            rez.serialize(COLLECTIVE_HOURGLASS_ALLREDUCE);
          }
          rez.dispatch(child);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
        }
        // Then do our local broadcast
        broadcast_local(
            local_manager, 0, op, index, copy_expression, match_space,
            copy_mask, broadcast_pre, predicate_guard, local_fields, local_inst,
            trace_info, COLLECTIVE_HOURGLASS_ALLREDUCE, broadcast_events,
            recorded_events, applied_events);
        if (!broadcast_events.empty())
        {
          const ApEvent broadcast_done =
              Runtime::merge_events(&trace_info, broadcast_events);
          if (broadcast_done.exists())
          {
            local_view->add_copy_user(
                true /*reading*/, 0 /*redop*/, broadcast_done, copy_mask,
                copy_expression, match_space, op_id, index, recorded_events,
                trace_info.recording, runtime->address_space);
            if (all_done.exists())
              all_done_events.emplace_back(broadcast_done);
          }
        }
      }
      if (all_done.exists())
      {
        if (all_bar.exists())
        {
          ApEvent arrival;
          if (!all_done_events.empty())
            arrival = Runtime::merge_events(&trace_info, all_done_events);
          runtime->phase_barrier_arrive(all_bar, 1 /*count*/, arrival);
          trace_info.record_barrier_arrival(
              all_bar, arrival, 1 /*count*/, applied_events, owner_shard);
          Runtime::trigger_event(all_done, all_bar, trace_info, applied_events);
        }
        else
        {
          Runtime::trigger_event(
              all_done, Runtime::merge_events(&trace_info, all_done_events),
              trace_info, applied_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeHourglass::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent dst_view_ready, src_view_ready;
      CollectiveView* target = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, dst_view_ready));
      derez.deserialize(did);
      AllreduceView* src_view = static_cast<AllreduceView*>(
          runtime->find_or_request_logical_view(did, src_view_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      std::set<RtEvent> ready_events;
      Operation* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      DistributedID src_inst_did;
      derez.deserialize(src_inst_did);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent all_done;
      derez.deserialize(all_done);
      bool copy_restricted;
      derez.deserialize<bool>(copy_restricted);

      if (src_view_ready.exists() && !src_view_ready.has_triggered())
        ready_events.insert(src_view_ready);
      if (dst_view_ready.exists() && !dst_view_ready.has_triggered())
        ready_events.insert(dst_view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      target->perform_collective_hourglass(
          src_view, precondition, predicate_guard, copy_expression, op, index,
          match_space, copy_mask, src_inst_did, trace_info, recorded_events,
          applied_events, all_done, runtime->address_space, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    void CollectiveView::perform_collective_pointwise(
        CollectiveView* source, ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, Operation* op,
        const unsigned index, const IndexSpace match_space,
        const size_t op_ctx_index, const FieldMask& copy_mask,
        const DistributedID src_inst_did, const UniqueID src_inst_did_op,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, ApUserEvent all_done,
        ApBarrier all_bar, ShardID owner_shard, AddressSpaceID origin,
        const uint64_t allreduce_tag, const bool copy_restricted)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping->contains(local_space));
      legion_assert((op != nullptr) || !copy_restricted);
      CollectiveAnalysis* first_local_analysis = nullptr;
      if (!copy_restricted && ((op == nullptr) || trace_info.recording))
      {
        // If this is not a copy-out to a restricted collective instance
        // then we should be able to find our local analyses to use for
        // performing operations
        first_local_analysis = local_views.front()->find_collective_analysis(
            op_ctx_index, index, match_space);
        legion_assert(first_local_analysis != nullptr);
        if (op == nullptr)
        {
          op = first_local_analysis->get_operation();
          // Don't need the analysis anymore if we're not tracing
          if (!trace_info.recording)
            first_local_analysis = nullptr;
        }
      }
      legion_assert(op != nullptr);
      const PhysicalTraceInfo& local_info =
          (first_local_analysis == nullptr) ?
              trace_info :
              first_local_analysis->get_trace_info();
      // First distribute this off to all the child nodes
      std::vector<ApEvent> done_events;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributePointwise rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          rez.serialize(source->did);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, child);
          rez.serialize<bool>(copy_restricted);
          if (copy_restricted)
            op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(match_space);
          rez.serialize(op_ctx_index);
          rez.serialize(copy_mask);
          rez.serialize(src_inst_did);
          rez.serialize(src_inst_did_op);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (local_info.recording)
          {
            rez.serialize(all_bar);
            if (all_bar.exists())
              rez.serialize(owner_shard);
          }
          else
          {
            ApUserEvent done;
            if (all_done.exists())
            {
              done = Runtime::create_ap_user_event(&local_info);
              done_events.emplace_back(done);
            }
            rez.serialize(done);
          }
          rez.serialize(origin);
          rez.serialize(allreduce_tag);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      CollectiveKind collective_kind = COLLECTIVE_POINT_TO_POINT;
      const UniqueID op_id = op->get_unique_op_id();
      // If the source is a reduction manager, this is where we need
      // to perform the all-reduce before issuing the pointwise copies
      if (source->is_allreduce_view())
      {
        // Better have the same collective mappings if we're doing all-reduce
        legion_assert(
            (collective_mapping == source->collective_mapping) ||
            ((*collective_mapping) == (*(source->collective_mapping))));
        legion_assert(source->is_reduction_kind());
        AllreduceView* allreduce = source->as_allreduce_view();
        allreduce->perform_collective_allreduce(
            precondition, predicate_guard, copy_expression, match_space, op,
            index, copy_mask, local_info, recorded_events, applied_events,
            allreduce_tag);
        collective_kind = COLLECTIVE_BUTTERFLY_ALLREDUCE;
      }
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        const PhysicalTraceInfo& inst_info =
            (first_local_analysis == nullptr) ?
                trace_info :
                local_views[idx]
                    ->find_collective_analysis(op_ctx_index, index, match_space)
                    ->get_trace_info();
        IndividualView* local_view = local_views[idx];
        // Find the precondition for all our local copies
        const ApEvent dst_pre = local_view->find_copy_preconditions(
            false /*reading*/, source->get_redop(), copy_mask, copy_expression,
            op_id, index, applied_events, inst_info);
        if (dst_pre.exists())
        {
          if (precondition.exists())
            precondition =
                Runtime::merge_events(&local_info, precondition, dst_pre);
          else
            precondition = dst_pre;
        }
        PhysicalManager* local_manager = local_view->get_manager();
        // Get our dst_fields
        std::vector<CopySrcDstField> dst_fields;
        local_manager->compute_copy_offsets(copy_mask, dst_fields);
        std::vector<Reservation> reservations;
        if (source->get_redop() > 0)
        {
          local_view->find_field_reservations(copy_mask, reservations);
          for (unsigned idx = 0; idx < dst_fields.size(); idx++)
            dst_fields[idx].set_redop(
                source->get_redop(), false /*fold*/, true /*exclusive*/);
        }
        const Memory location = local_manager->memory_manager->memory;
        // Now we need to pick the source point for this copy if it hasn't
        // already been picked by the mapper
        DistributedID local_src_inst_did = 0;
        if (!copy_restricted)
        {
          CollectiveAnalysis* analysis =
              local_views[idx]->find_collective_analysis(
                  op_ctx_index, index, match_space);
          // See if this is the same analysis that already had a change to
          // pick the source instance because it was the one issuing this
          // copy in the first place. If not then we give the mapper a
          // chance to pick the source now
          Operation* analysis_op = analysis->get_operation();
          if (analysis_op->get_unique_op_id() != src_inst_did_op)
          {
            // invoke the mapper to pick the source point in this case
            std::vector<InstanceView*> src_views(1, source);
            std::vector<unsigned> ranking;
            std::map<unsigned, PhysicalManager*> points;
            analysis_op->select_sources(
                analysis->get_requirement_index(), local_manager, src_views,
                ranking, points);
            std::map<unsigned, PhysicalManager*>::const_iterator finder =
                points.find(0);
            if (finder != points.end())
              local_src_inst_did = finder->second->did;
          }
          else  // mapper already had a chance to pick the source point
            local_src_inst_did = src_inst_did;
        }
        // TODO: how to let the mapper pick in copy-out cases
        // If the mapper didn't pick a source point then we can
        const AddressSpaceID src =
            (local_src_inst_did > 0) ?
                runtime->determine_owner(local_src_inst_did) :
                source->select_source_space(local_space);
        const UniqueInst dst_inst(local_view);
        ApEvent local_done;
        if (src != local_space)
        {
          const RtUserEvent recorded = Runtime::create_rt_user_event();
          const RtUserEvent applied = Runtime::create_rt_user_event();
          ApUserEvent done = Runtime::create_ap_user_event(&inst_info);
          CollectiveDistributePoint rez;
          {
            RezCheck z(rez);
            rez.serialize(source->did);
            pack_fields(rez, dst_fields, local_manager->get_unique_event());
            rez.serialize<size_t>(reservations.size());
            for (unsigned idx2 = 0; idx2 < reservations.size(); idx2++)
              rez.serialize(reservations[idx2]);
            rez.serialize(precondition);
            rez.serialize(predicate_guard);
            copy_expression->pack_expression(rez, src);
            rez.serialize(match_space);
            op->pack_remote_operation(rez, src, applied_events);
            rez.serialize(index);
            rez.serialize(copy_mask);
            rez.serialize(copy_mask);  // again for dst mask
            rez.serialize(location);
            dst_inst.serialize(rez);
            rez.serialize(local_src_inst_did);
            inst_info.pack_trace_info(rez);
            rez.serialize(collective_kind);
            rez.serialize(recorded);
            rez.serialize(applied);
            rez.serialize(done);
          }
          rez.dispatch(src);
          recorded_events.insert(recorded);
          applied_events.insert(applied);
          local_done = done;
        }
        else
          local_done = source->perform_collective_point(
              dst_fields, reservations, precondition, predicate_guard,
              copy_expression, match_space, op, index, copy_mask, copy_mask,
              location, dst_inst, local_manager->get_unique_event(),
              local_src_inst_did, inst_info, recorded_events, applied_events,
              collective_kind);
        if (local_done.exists())
        {
          done_events.emplace_back(local_done);
          local_view->add_copy_user(
              false /*reading*/, source->get_redop(), local_done, copy_mask,
              copy_expression, match_space, op_id, index, recorded_events,
              inst_info.recording, runtime->address_space);
        }
      }
      if (all_bar.exists())
      {
        ApEvent arrival;
        if (!done_events.empty())
          arrival = Runtime::merge_events(&local_info, done_events);
        runtime->phase_barrier_arrive(all_bar, 1 /*count*/, arrival);
        local_info.record_barrier_arrival(
            all_bar, arrival, 1 /*count*/, applied_events, owner_shard);
      }
      else if (all_done.exists())
      {
        Runtime::trigger_event(
            all_done, Runtime::merge_events(&local_info, done_events),
            local_info, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributePointwise::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent dst_view_ready, src_view_ready;
      CollectiveView* dst_view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, dst_view_ready));
      derez.deserialize(did);
      CollectiveView* src_view = static_cast<CollectiveView*>(
          runtime->find_or_request_logical_view(did, src_view_ready));
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      bool copy_restricted;
      derez.deserialize(copy_restricted);
      Operation* op = nullptr;
      std::set<RtEvent> ready_events;
      if (copy_restricted)
        op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      size_t op_ctx_index;
      derez.deserialize(op_ctx_index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      DistributedID src_inst_did;
      derez.deserialize(src_inst_did);
      UniqueID src_inst_did_op;
      derez.deserialize(src_inst_did_op);
      std::set<RtEvent> recorded_events, applied_events;
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApBarrier all_bar;
      ShardID owner_shard = 0;
      ApUserEvent all_done;
      if (trace_info.recording)
      {
        derez.deserialize(all_bar);
        if (all_bar.exists())
          derez.deserialize(owner_shard);
      }
      else
        derez.deserialize(all_done);
      AddressSpaceID origin;
      derez.deserialize(origin);
      uint64_t allreduce_tag;
      derez.deserialize(allreduce_tag);

      if (src_view_ready.exists() && !src_view_ready.has_triggered())
        ready_events.insert(src_view_ready);
      if (dst_view_ready.exists() && !dst_view_ready.has_triggered())
        ready_events.insert(dst_view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Check if this is the first invocation for allreduce on a
      // node where we can get a tag
      if ((allreduce_tag == 0) && src_view->is_allreduce_view())
      {
        AllreduceView* allreduce = src_view->as_allreduce_view();
        allreduce_tag = allreduce->generate_unique_allreduce_tag();
      }

      dst_view->perform_collective_pointwise(
          src_view, precondition, predicate_guard, copy_expression, op, index,
          match_space, op_ctx_index, copy_mask, src_inst_did, src_inst_did_op,
          trace_info, recorded_events, applied_events, all_done, all_bar,
          owner_shard, origin, allreduce_tag, copy_restricted);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (op != nullptr)
        delete op;
    }

  }  // namespace Internal
}  // namespace Legion
