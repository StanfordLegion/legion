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

#include "legion/instances/physical.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/across.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/region.h"
#include "legion/tools/spy.h"
#include "legion/utilities/privileges.h"
#include "legion/views/materialized.h"
#include "legion/views/reduction.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // PhysicalManager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalManager::PhysicalManager(
        DistributedID did, MemoryManager* memory, PhysicalInstance inst,
        IndexSpaceExpression* inst_domain, const void* pl, size_t pl_size,
        FieldSpaceNode* node, RegionTreeID tree_id, LayoutDescription* layout,
        ReductionOpID redop_id, bool register_now, size_t footprint,
        ApEvent u_event, LgEvent unique, InstanceKind k,
        const ReductionOp* op /*= nullptr*/,
        CollectiveMapping* mapping /*=nullptr*/,
        ApEvent p_event /*= ApEvent::NO_AP_EVENT*/,
        GarbageCollectionState init /*COLLECTABLE_GC_STATE*/)
      : InstanceManager(
            encode_instance_did(
                did, (k == EXTERNAL_ATTACHED_INSTANCE_KIND), (redop_id > 0)),
            layout, node,
            // If we're on the owner node we need to produce the expression
            // that actually describes this points in this space
            // On remote nodes we'll already have it from the owner
            (runtime->determine_owner(did) == runtime->address_space) &&
                    (k != UNBOUND_INSTANCE_KIND) ?
                inst_domain->create_layout_expression(pl, pl_size) :
                inst_domain,
            tree_id, register_now, mapping),
        memory_manager(memory), unique_event(unique),
        instance_footprint(footprint),
        reduction_op(
            (redop_id == 0) ? nullptr : runtime->get_reduction(redop_id)),
        redop(redop_id), piece_list(pl), piece_list_size(pl_size),
        instance(inst), use_event(Runtime::create_ap_user_event(nullptr)),
        instance_ready(
            (k == UNBOUND_INSTANCE_KIND) ? Runtime::create_rt_user_event() :
                                           RtUserEvent::NO_RT_USER_EVENT),
        kind(k), producer_event(p_event), gc_state(init), pending_changes(0),
        failed_collection_count(0), min_gc_priority(0), added_gc_events(0),
        valid_references(0), sent_valid_references(0),
        received_valid_references(0), collect_lamport_clock(0),
        pending_collect_lamport_clock(0), bump_collect_lamport_clock(false),
        padded_reservations(nullptr)
    //--------------------------------------------------------------------------
    {
      // If the manager was initialized with a valid Realm instance,
      // trigger the use event with the ready event of the instance metadata
      if (kind != UNBOUND_INSTANCE_KIND)
      {
        legion_assert(instance.exists());
        Runtime::trigger_event_untraced(use_event, fetch_metadata(u_event));
        if (is_owner() && (runtime->profiler != nullptr))
          implicit_profiler->register_physical_instance_spaces(
              unique_event, inst_domain->record_profiler_expression(),
              (pl_size > 0) ? instance_domain->record_profiler_expression() :
                              0);
        if (spy_logging_level > NO_SPY_LOGGING)
        {
          legion_assert(unique_event.exists());
          LegionSpy::log_physical_instance(
              unique_event, inst.id, memory->memory.id, inst_domain->expr_id,
              field_space_node->handle, tree_id, redop);
          layout->log_instance_layout(unique_event);
        }
      }
      else  // add a resource reference to remove once this manager is set
        add_base_valid_ref(PENDING_UNBOUND_REF);
      // If we're in a pending collectable state, then add a reference
      if (gc_state == PENDING_COLLECTED_GC_STATE)
      {
        legion_assert(!is_owner());
        add_base_resource_ref(PENDING_COLLECTIVE_REF);
      }
      if (!is_owner() && !is_external_instance())
        memory_manager->register_remote_instance(this);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Instance Manager %lld %d " IDFMT " " IDFMT " ",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, inst.id,
          memory->memory.id);
#endif
    }

    //--------------------------------------------------------------------------
    PhysicalManager::~PhysicalManager(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(subscribers.empty());
      legion_assert(valid_references == 0);
      // Remote references removed by DistributedCollectable destructor
      if (!is_owner() && !is_external_instance())
        memory_manager->unregister_remote_instance(this);
      if (padded_reservations != nullptr)
      {
        // If this is the owner view, delete any atomic reservations
        if (is_owner())
        {
          for (std::pair<const unsigned, Reservation>& it :
               *padded_reservations)
            it.second.destroy_reservation();
        }
        delete padded_reservations;
      }
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalManager::get_use_event(ApEvent user) const
    //--------------------------------------------------------------------------
    {
      if (kind != UNBOUND_INSTANCE_KIND)
        return use_event;
      else
        // If the user is the one that is going to bind an instance
        // to this manager, return a no event
        return (user == producer_event) ? ApEvent::NO_AP_EVENT : use_event;
    }

    //--------------------------------------------------------------------------
    PointerConstraint PhysicalManager::get_pointer_constraint(void) const
    //--------------------------------------------------------------------------
    {
      if (use_event.exists() && !use_event.has_triggered_faultignorant())
        use_event.wait_faultignorant();
      void* inst_ptr = instance.pointer_untyped(0 /*offset*/, 0 /*elem size*/);
      return PointerConstraint(memory_manager->memory, uintptr_t(inst_ptr));
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::log_instance_creation(
        UniqueID creator_id, Processor proc,
        const std::vector<LogicalRegion>& regions) const
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      const LgEvent inst_event = get_unique_event();
      const LayoutConstraints* constraints = layout->constraints;
      LegionSpy::log_physical_instance_creator(inst_event, creator_id, proc.id);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        LegionSpy::log_physical_instance_creation_region(
            inst_event, regions[idx]);
      LegionSpy::log_instance_specialized_constraint(
          inst_event, constraints->specialized_constraint.kind,
          constraints->specialized_constraint.redop);
      legion_assert(constraints->memory_constraint.has_kind);
      if (constraints->memory_constraint.is_valid())
        LegionSpy::log_instance_memory_constraint(
            inst_event, constraints->memory_constraint.kind);
      LegionSpy::log_instance_field_constraint(
          inst_event, constraints->field_constraint.contiguous,
          constraints->field_constraint.inorder,
          constraints->field_constraint.field_set.size());
      for (FieldID fid : constraints->field_constraint.field_set)
        LegionSpy::log_instance_field_constraint_field(inst_event, fid);
      LegionSpy::log_instance_ordering_constraint(
          inst_event, constraints->ordering_constraint.contiguous,
          constraints->ordering_constraint.ordering.size());
      for (DimensionKind kind : constraints->ordering_constraint.ordering)
        LegionSpy::log_instance_ordering_constraint_dimension(inst_event, kind);
      for (const TilingConstraint& constraint : constraints->tiling_constraints)
        LegionSpy::log_instance_tiling_constraint(
            inst_event, constraint.dim, constraint.value, constraint.tiles);
      for (const DimensionConstraint& constraint :
           constraints->dimension_constraints)
        LegionSpy::log_instance_dimension_constraint(
            inst_event, constraint.kind, constraint.eqk, constraint.value);
      for (const AlignmentConstraint& constraint :
           constraints->alignment_constraints)
        LegionSpy::log_instance_alignment_constraint(
            inst_event, constraint.fid, constraint.eqk, constraint.alignment);
      for (const OffsetConstraint& constraint : constraints->offset_constraints)
        LegionSpy::log_instance_offset_constraint(
            inst_event, constraint.fid, constraint.offset);
    }

    //--------------------------------------------------------------------------
    PhysicalInstance PhysicalManager::get_instance(void) const
    //--------------------------------------------------------------------------
    {
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
      return instance;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::compute_copy_offsets(
        const FieldMask& copy_mask, std::vector<CopySrcDstField>& fields)
    //--------------------------------------------------------------------------
    {
      // Make sure the instance is ready before we compute the offsets
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
      legion_assert(layout != nullptr);
      legion_assert(instance.exists());
      // Pass in our physical instance so the layout knows how to specialize
      layout->compute_copy_offsets(copy_mask, instance, fields);
    }

    //--------------------------------------------------------------------------
    IndividualView* PhysicalManager::construct_top_view(
        AddressSpaceID logical_owner, DistributedID view_did,
        InnerContext* own_ctx, CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      if (redop > 0)
      {
        if (mapping != nullptr)
        {
          // Handle the case where we already requested this view on this
          // node from an unrelated meta-task execution
          void* location =
              runtime
                  ->find_or_create_pending_collectable_location<ReductionView>(
                      view_did);
          return new (location) ReductionView(
              view_did, logical_owner, this, true /*register now*/, mapping);
        }
        else
          return new ReductionView(
              view_did, logical_owner, this, true /*register now*/, mapping);
      }
      else
      {
        if (mapping != nullptr)
        {
          // Handle the case where we already requested this view on this
          // node from an unrelated meta-task execution
          void* location = runtime->find_or_create_pending_collectable_location<
              MaterializedView>(view_did);
          return new (location) MaterializedView(
              view_did, logical_owner, this, true /*register now*/, mapping);
        }
        else
          return new MaterializedView(
              view_did, logical_owner, this, true /*register now*/, mapping);
      }
    }

    //--------------------------------------------------------------------------
    IndividualView* PhysicalManager::find_or_create_instance_top_view(
        InnerContext* own_ctx, AddressSpaceID logical_owner,
        CollectiveMapping* mapping)
    //--------------------------------------------------------------------------
    {
      // If we're a replicate context then we want to ignore the specific
      // context DID since there might be several shards on this node
      bool replicated = false;
      DistributedID key = own_ctx->get_replication_id();
      if (key == 0)
        key = own_ctx->did;
      else
        replicated = true;
      RtEvent wait_for;
      {
        AutoLock i_lock(inst_lock);
        // All contexts should always be new since they should be deduplicating
        // on their side before calling this method
        legion_assert(subscribers.find(own_ctx) == subscribers.end());
        std::map<DistributedID, ViewEntry>::iterator finder =
            context_views.find(key);
        if (finder != context_views.end())
        {
          // This should only happen with control replication because normal
          // contexts should be deduplicating on their side
          legion_assert(replicated);
          // This better be a new context so bump the reference count
          if (subscribers.insert(own_ctx).second)
            own_ctx->add_subscriber_reference(this);
          finder->second.second++;
          return finder->second.first;
        }
        // Check to see if someone else from this context is making the view
        if (replicated)
        {
          // Only need to do this for control replication, otherwise the
          // context will have deduplicated for us
          std::map<DistributedID, RtUserEvent>::iterator pending_finder =
              pending_views.find(key);
          if (pending_finder != pending_views.end())
          {
            if (!pending_finder->second.exists())
              pending_finder->second = Runtime::create_rt_user_event();
            wait_for = pending_finder->second;
          }
          else
            pending_views[key] = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      if (wait_for.exists())
      {
        if (!wait_for.has_triggered())
          wait_for.wait();
        AutoLock i_lock(inst_lock);
        std::map<DistributedID, ViewEntry>::iterator finder =
            context_views.find(key);
        legion_assert(replicated);
        legion_assert(finder != context_views.end());
        // This better be a new context so bump the reference count
        if (subscribers.insert(own_ctx).second)
          own_ctx->add_subscriber_reference(this);
        finder->second.second++;
        return finder->second.first;
      }
      // At this point we're repsonsibile for doing the work to make the view
      IndividualView* result = nullptr;
      // Check to see if we're the owner
      if (is_owner())
      {
        // We're going to construct the view no matter what, see which
        // node is going to be the logical owner
        DistributedID view_did = runtime->get_available_distributed_id();
        result = construct_top_view(
            (mapping == nullptr) ? logical_owner : owner_space, view_did,
            own_ctx, mapping);
      }
      else if ((mapping != nullptr) && mapping->contains(local_space))
      {
        // If we're collectively making this view then we're just going to
        // do that and use the owner node as the logical owner for the view
        // We still need to get the distributed ID from the next node down
        // in the collective mapping though
        std::atomic<DistributedID> view_did(0);
        RtUserEvent ready = Runtime::create_rt_user_event();
        CreateTopViewRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          own_ctx->pack_inner_context(rez);
          rez.serialize(owner_space);
          mapping->pack(rez);
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        AddressSpaceID target = mapping->get_parent(owner_space, local_space);
        rez.dispatch(target);
        ready.wait();
        // For collective instances each node of the instance serves as its
        // own logical owner view
        result = construct_top_view(
            runtime->address_space, view_did.load(), own_ctx, mapping);
      }
      else
      {
        // We're not collective and not the owner so send the request
        // to the owner to make the logical view and send back the result
        std::atomic<DistributedID> view_did(0);
        RtUserEvent ready = Runtime::create_rt_user_event();
        CreateTopViewRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          own_ctx->pack_inner_context(rez);
          rez.serialize(logical_owner);
          rez.serialize<size_t>(0);  // no mapping
          rez.serialize(&view_did);
          rez.serialize(ready);
        }
        rez.dispatch(owner_space);
        ready.wait();
        RtEvent view_ready;
        result = static_cast<IndividualView*>(
            runtime->find_or_request_logical_view(view_did.load(), view_ready));
        if (view_ready.exists() && !view_ready.has_triggered())
          view_ready.wait();
      }
      // Retake the lock, save the view, and signal any other waiters
      AutoLock i_lock(inst_lock);
      legion_assert(context_views.find(key) == context_views.end());
      ViewEntry& entry = context_views[key];
      entry.first = result;
      entry.second = 1 /*only a single initial reference*/;
      if (subscribers.insert(own_ctx).second)
        own_ctx->add_subscriber_reference(this);
      if (replicated)
      {
        std::map<DistributedID, RtUserEvent>::iterator finder =
            pending_views.find(key);
        legion_assert(finder != pending_views.end());
        if (finder->second.exists())
          Runtime::trigger_event(finder->second);
        pending_views.erase(finder);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::register_deletion_subscriber(
        InstanceDeletionSubscriber* subscriber, bool allow_duplicates)
    //--------------------------------------------------------------------------
    {
      bool result = false;
      subscriber->add_subscriber_reference(this);
      {
        AutoLock inst(inst_lock);
        if (gc_state != COLLECTED_GC_STATE)
        {
          if (subscribers.insert(subscriber).second)
            return true;
          legion_assert(allow_duplicates);
          result = true;
          // Fall through to remove the duplicate reference on the subscriber
        }
      }
      if (subscriber->remove_subscriber_reference(this))
        delete subscriber;
      return result;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_deletion_subscriber(
        InstanceDeletionSubscriber* subscriber)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock inst(inst_lock);
        std::set<InstanceDeletionSubscriber*>::iterator finder =
            subscribers.find(subscriber);
        if (finder == subscribers.end())
          return;
        subscribers.erase(finder);
      }
      if (subscriber->remove_subscriber_reference(this))
        delete subscriber;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unregister_active_context(InnerContext* own_ctx)
    //--------------------------------------------------------------------------
    {
      // If we're a replicate context then we want to ignore the specific
      // context UID since there might be several shards on this node
      DistributedID key = own_ctx->get_replication_id();
      if (key == 0)
        key = own_ctx->did;
      {
        AutoLock inst(inst_lock);
        std::set<InstanceDeletionSubscriber*>::iterator finder =
            subscribers.find(own_ctx);
        // We could already have removed this context if this
        // physical instance was deleted
        if (finder == subscribers.end())
          return;
        subscribers.erase(finder);
        // Remove the reference on the view entry and remove it from our
        // manager if it no longer has anymore active contexts
        std::map<DistributedID, ViewEntry>::iterator view_finder =
            context_views.find(key);
        legion_assert(view_finder != context_views.end());
        legion_assert(view_finder->second.second > 0);
        if (--view_finder->second.second == 0)
          context_views.erase(view_finder);
      }
      if (own_ctx->remove_subscriber_reference(this))
        delete own_ctx;
    }

    //--------------------------------------------------------------------------
    PieceIteratorImpl* PhysicalManager::create_piece_iterator(
        IndexSpaceNode* privilege_node)
    //--------------------------------------------------------------------------
    {
      return instance_domain->create_piece_iterator(
          piece_list, piece_list_size, privilege_node);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::record_instance_user(
        ApEvent user_event, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock);
      legion_assert(gc_state != COLLECTED_GC_STATE);
      legion_assert(added_gc_events < runtime->gc_epoch_size);
      if (is_owner() || (gc_state != PENDING_COLLECTED_GC_STATE))
      {
        if (gc_events.insert(user_event).second &&
            (++added_gc_events == runtime->gc_epoch_size))
        {
          // We don't prune these when doing detailed legion spy so that we
          // can check that there are no use-after-delete errors
          if (spy_logging_level <= LIGHT_SPY_LOGGING)
          {
            // Go through and prune out any events that have triggered
            for (std::set<ApEvent>::iterator it = gc_events.begin();
                 it != gc_events.end();
                 /*nothing*/)
            {
              if (it->has_triggered_faultignorant())
              {
                std::set<ApEvent>::iterator to_delete = it++;
                gc_events.erase(to_delete);
              }
              else
                it++;
            }
          }
          added_gc_events = 0;
        }
      }
      else
      {
        const RtEvent applied = Runtime::create_rt_user_event();
        GarbageCollectionRecordEvent rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(user_event);
          rez.serialize(applied);
          pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
        applied_events.insert(applied);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionRecordEvent::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ApEvent user_event;
      derez.deserialize(user_event);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      std::set<RtEvent> applied;
      manager->record_instance_user(user_event, applied);
      manager->unpack_global_ref(derez);
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_shutdown_preconditions(
        std::set<ApEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock inst(inst_lock, false /*exclusive*/);
      // Only need to get these if we didn't already delete the manager
      // If we already deleted the manager there is already a meta-task in
      // flight that summarizes these events and makes sure we can't shutdown
      // without it running, see perform_deletion
      if (gc_state != COLLECTED_GC_STATE)
      {
        for (const ApEvent& event : gc_events)
          if (!event.has_triggered_faultignorant())
            preconditions.insert(event);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_regions(
        const std::vector<LogicalRegion>& regions, bool tight_region_bounds,
        const Domain* padding_delta) const
    //--------------------------------------------------------------------------
    {
      legion_assert(tree_id > 0);  // only happens with VirtualManager
      legion_assert(!regions.empty());
      std::set<IndexSpaceExpression*> region_exprs;
      for (const LogicalRegion& region : regions)
      {
        // If the region tree IDs don't match that is bad
        if (region.get_tree_id() != tree_id)
          return false;
        RegionNode* node = runtime->get_node(region);
        region_exprs.insert(node->row_source);
      }
      IndexSpaceExpression* space_expr =
          (region_exprs.size() == 1) ?
              *(region_exprs.begin()) :
              runtime->union_index_spaces(region_exprs);
      return meets_expression(space_expr, tight_region_bounds, padding_delta);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::meets_expression(
        IndexSpaceExpression* space_expr, bool tight_bounds,
        const Domain* padding_delta) const
    //--------------------------------------------------------------------------
    {
      return instance_domain->meets_layout_expression(
          space_expr, tight_bounds, piece_list, piece_list_size, padding_delta);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_local(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_valid_ref(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      LamportClock lamport_clock = 0;
      pack_valid_ref(lamport_clock);
      rez.serialize(lamport_clock);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_valid_ref(LamportClock& lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      // We should always be holding a valid reference when we
      // pack a valid reference so the state should always be valid
      legion_assert(gc_state == VALID_GC_STATE);
      // If we've committed our counts to a collection round then the first
      // pack afterwards must advance the clock so that a reference packed
      // after the commit is detectable when it is unpacked elsewhere
      if (bump_collect_lamport_clock)
      {
        collect_lamport_clock++;
        bump_collect_lamport_clock = false;
      }
      lamport_clock = collect_lamport_clock;
      sent_valid_references++;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unpack_valid_ref(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);
      unpack_valid_ref(lamport_clock);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::unpack_valid_ref(LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);

      received_valid_references++;
      collect_lamport_clock = std::max(collect_lamport_clock, lamport_clock);
    }

#ifdef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void PhysicalManager::add_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      valid_references += cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid(true /*need check*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::add_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      valid_references += cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
      if (valid_references == cnt)
        notify_valid(true /*need check*/);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      legion_assert(finder != detailed_base_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid(i_lock);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      legion_assert(finder != detailed_nested_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_valid_references.erase(finder);
      if (valid_references == 0)
        return notify_invalid(i_lock);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::add_valid_reference(int cnt, bool need_check)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      if (valid_references == 0)
        notify_valid(need_check);
      valid_references += cnt;
    }
#else   // LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void PhysicalManager::add_valid_reference(int cnt, bool need_check)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      if (valid_references.fetch_add(cnt) == 0)
        notify_valid(need_check);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      legion_assert(valid_references.load() >= cnt);
      if (valid_references.fetch_sub(cnt) == cnt)
        return notify_invalid(i_lock);
      else
        return false;
    }
#endif  // !defined LEGION_DEBUG_GC

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_valid(bool need_check)
    //--------------------------------------------------------------------------
    {
      // No need for the lock, it is held by the caller
      legion_assert(gc_state != VALID_GC_STATE);
      legion_assert(gc_state != COLLECTED_GC_STATE);
#ifdef LEGION_DEBUG
      // In debug mode we eagerly add valid references such that the owner
      // is valid as long as a copy of the manager on one node is valid
      // This way we can easily check that acquires are being done safely
      // if instance isn't already valid somewhere
      if (need_check && (kind != UNBOUND_INSTANCE_KIND) &&
          (!is_external_instance() || !is_owner()))
      {
        // Should never be here if we're the owner as it indicates that
        // we tried to add a valid reference without first doing an acquire
        legion_assert(!is_owner());
        // Send a message to check that we can safely do the acquire
        const RtUserEvent done = Runtime::create_rt_user_event();
        std::atomic<bool> result(true);
        GarbageCollectionDebugRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(&result);
          rez.serialize(done);
          pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
        if (!done.has_triggered())
          done.wait();
        if (!result.load())
        {
          Fatal err;
          err << "Found an internal garbage collection race. Please "
              << "run with -lg:safe_mapper and see if it reports any "
              << "errors. If not, then please report this as a bug.";
          err.raise();
        }
      }
#else
      if (gc_state == COLLECTED_GC_STATE)
      {
        Fatal err;
        err << "Found an internal garbage collection race. Please "
            << "run with -lg:safe_mapper and see if it reports any "
            << "errors. If not, then please report this as a bug.";
        err.raise();
      }
#endif
      gc_state = VALID_GC_STATE;
      add_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionDebugRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<bool>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      // Should be guaranteed to be able to acquire this
      if (manager->acquire_instance(REMOTE_DID_REF))
      {
        Runtime::trigger_event(done);
        // Remove the reference that we just got
        manager->remove_base_valid_ref(REMOTE_DID_REF);
      }
      else
      {
        // If we get here, we failed so send the response
        GarbageCollectionDebugResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize(done);
        }
        rez.dispatch(source);
      }
      manager->unpack_global_ref(derez);
#else
      std::abort();  // should never get this in release mode
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionDebugResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      DerezCheck z(derez);
      std::atomic<bool>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      target->store(false);
      Runtime::trigger_event(done);
#else
      std::abort();  // should never get this in release mode
#endif
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::notify_invalid(AutoLock& i_lock)
    //--------------------------------------------------------------------------
    {
      // No need for the lock it is held by the caller
      legion_assert(kind != UNBOUND_INSTANCE_KIND);
      legion_assert((gc_state == VALID_GC_STATE) || is_external_instance());
      // If we're an external instance that has already been detached and
      // therfore deleted then we don't ever want to go back to collectable
      if (!is_external_instance() || (gc_state != COLLECTED_GC_STATE))
      {
        gc_state = COLLECTABLE_GC_STATE;
        // If this instance is set to eager collection priority
        // then we try to do that now
        if (min_gc_priority == LEGION_GC_EAGER_PRIORITY)
        {
          RtEvent dummy_ready;
          collect(dummy_ready, nullptr, &i_lock);
        }
      }
      return remove_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_GC
    template<typename T>
    bool PhysicalManager::acquire_internal(
        T source, std::map<T, int>& detailed_valid_references)
#else
    bool PhysicalManager::acquire_internal(void)
#endif
    //--------------------------------------------------------------------------
    {
      {
        bool success = false;
        AutoLock i_lock(inst_lock);
        // Check our current state
        switch (gc_state.load())
        {
          case VALID_GC_STATE:
            {
              legion_assert(valid_references > 0);
              success = true;
              break;
            }
          case COLLECTABLE_GC_STATE:
            {
              notify_valid(false /*need check*/);
              success = true;
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
            {
              // Hurry the garbage collector is trying to eat it!
              if (is_owner())
              {
                // We're the owner so we can save this
                notify_valid(false /*need check*/);
                success = true;
              }
              // Not the owner so we need to send a message to the
              // owner to have it try to do the acquire
              break;
            }
          case COLLECTED_GC_STATE:
            return false;
          default:
            std::abort();
        }
        if (success)
        {
#ifdef LEGION_DEBUG_GC
          valid_references++;
          typename std::map<T, int>::iterator finder =
              detailed_valid_references.find(source);
          if (finder == detailed_valid_references.end())
            detailed_valid_references[source] = 1;
          else
            finder->second++;
#else
          valid_references.fetch_add(1);
#endif
          return true;
        }
      }
      legion_assert(!is_owner());
      std::atomic<bool> result(false);
      const RtUserEvent ready = Runtime::create_rt_user_event();
      GarbageCollectionAcquireRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(this);
        rez.serialize(&result);
        rez.serialize(ready);
      }
      rez.dispatch(owner_space);
      ready.wait();
      if (result.load())
      {
#ifdef LEGION_DEBUG_GC
        AutoLock i_lock(inst_lock);
        typename std::map<T, int>::iterator finder =
            detailed_valid_references.find(source);
        if (finder == detailed_valid_references.end())
          detailed_valid_references[source] = 1;
        else
          finder->second++;
#endif
        return true;
      }
      else
      {
        std::set<InstanceDeletionSubscriber*> to_notify;
        {
          AutoLock i_lock(inst_lock);
          legion_assert(
              (gc_state == PENDING_COLLECTED_GC_STATE) ||
              (gc_state == COLLECTED_GC_STATE));
          gc_state = COLLECTED_GC_STATE;
          to_notify.swap(subscribers);
        }
        for (InstanceDeletionSubscriber* subscriber : to_notify)
        {
          subscriber->notify_instance_deletion(this);
          if (subscriber->remove_subscriber_reference(this))
            delete subscriber;
        }
        return false;
      }
    }

#ifdef LEGION_DEBUG_GC
    // Explicit template instantiations
    template bool PhysicalManager::acquire_internal<ReferenceSource>(
        ReferenceSource, std::map<ReferenceSource, int>&);
    template bool PhysicalManager::acquire_internal<DistributedID>(
        DistributedID, std::map<DistributedID, int>&);
#endif

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionAcquireRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PhysicalManager* remote;
      derez.deserialize(remote);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      if (manager->acquire_instance(REMOTE_DID_REF))
      {
        // We succeeded so send the response back with the reference
        GarbageCollectionAcquireResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(remote);
          rez.serialize(result);
          rez.serialize(ready);
        }
        rez.dispatch(source);
        // Wait for the result to be applied and then remove
        // the reference that we acquired on this node
        ready.wait();
        manager->remove_base_valid_ref(REMOTE_DID_REF);
      }
      else
      {
        // We failed, so the flag is already set, just trigger the event
        Runtime::trigger_event(ready);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionAcquireResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalManager* manager;
      derez.deserialize(manager);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      // Just add the reference for now
      manager->add_valid_reference(1 /*count*/, false /*need check*/);
      result->store(true);
      // Triggering the event removes the reference we added on the remote node
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::can_collect(bool& already_collected) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      // This is a lightweight test that shouldn't involve any communication
      // or commitment to performing a collection. It's just for finding
      // instances that we know are locally collectable
      already_collected = false;
      AutoLock i_lock(inst_lock, false /*exclusive*/);
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
      {
        already_collected = true;
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::acquire_collect(
        std::set<ApEvent>& remote_events, uint64_t& sent_valid,
        uint64_t& received_valid, LamportClock lamport_clock,
        LamportClock& collect_clock)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      AutoLock i_lock(inst_lock);
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
      legion_assert(gc_state != COLLECTED_GC_STATE);
      gc_state = PENDING_COLLECTED_GC_STATE;
      remote_events.swap(gc_events);
      sent_valid = sent_valid_references;
      received_valid = received_valid_references;
      // Commit our contribution to this collection round: fold in the round's
      // snapshot clock and arm the bump so that any valid reference packed
      // after this point advances our clock and is therefore detectable as a
      // causality violation. Report our clock back to the owner so it can tell
      // when we observed a post-snapshot packed reference (collect_clock >
      // round snapshot) even in the case where the sent/received counts alias.
      collect_lamport_clock = std::max(collect_lamport_clock, lamport_clock);
      bump_collect_lamport_clock = true;
      collect_clock = collect_lamport_clock;
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtEvent* target;
      derez.deserialize(target);
      PhysicalInstance* hole;
      derez.deserialize(hole);
      RtUserEvent done;
      derez.deserialize(done);

      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      RtEvent ready;
      PhysicalInstance hole_instance = PhysicalInstance::NO_INST;
      if (manager->collect(ready, (hole == nullptr) ? nullptr : &hole_instance))
      {
        GarbageCollectionResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(result);
          rez.serialize(target);
          rez.serialize(ready);
          rez.serialize(hole);
          if (hole != nullptr)
            rez.serialize(hole_instance);
          rez.serialize(done);
        }
        rez.dispatch(source);
      }
      else  // Couldn't collect so we are done
        Runtime::trigger_event(done);
      manager->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtEvent* target;
      derez.deserialize(target);
      derez.deserialize(*target);
      PhysicalInstance* hole;
      derez.deserialize(hole);
      if (hole != nullptr)
        derez.deserialize(*hole);
      RtUserEvent done;
      derez.deserialize(done);

      result->store(true);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionAcquire::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::atomic<unsigned>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);

      RtEvent ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, ready);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      std::set<ApEvent> gc_events;
      const AddressSpaceID owner = manager->owner_space;
      uint64_t sent_valid = 0, received_valid = 0;
      LamportClock collect_clock = 0;
      if (!manager->acquire_collect(
              gc_events, sent_valid, received_valid, lamport_clock,
              collect_clock))
      {
        GarbageCollectionFailed rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(done);
        }
        rez.dispatch(owner);
      }
      else
      {
        std::set<RtEvent> ready_events;
        // Send the gc events back to the owner if we have any, merge
        // them all back together first so there is just one remote
        // event on this node
        if (!gc_events.empty())
        {
          const ApEvent remote = Runtime::merge_events(nullptr, gc_events);
          if (remote.exists())
            manager->record_instance_user(remote, ready_events);
        }
        // If we have different numbers of sent and received valid
        // references then we need to tell the owner that too. We must also
        // report when we observed a packed valid reference newer than this
        // round's snapshot (collect_clock > lamport_clock), since that is a
        // causality violation that can otherwise hide behind aliased counts.
        if ((sent_valid != received_valid) || (collect_clock > lamport_clock))
        {
          const RtUserEvent notified = Runtime::create_rt_user_event();
          GarbageCollectionMismatch rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(sent_valid);
            rez.serialize(received_valid);
            rez.serialize(collect_clock);
            rez.serialize(notified);
          }
          rez.dispatch(owner);
          ready_events.insert(notified);
        }
        const AddressSpaceID local = manager->local_space;
        // Check to see if we need to broadcast this out to more places
        if ((manager->collective_mapping != nullptr) &&
            manager->collective_mapping->contains(local))
        {
          // Broadcast this out to all our children
          std::vector<AddressSpaceID> children;
          manager->collective_mapping->get_children(owner, local, children);
          if (!children.empty())
          {
            for (const AddressSpaceID& child : children)
            {
              const RtUserEvent child_done = Runtime::create_rt_user_event();
              GarbageCollectionAcquire rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(target);
                rez.serialize(child_done);
                rez.serialize(lamport_clock);
              }
              rez.dispatch(child);
              ready_events.insert(child_done);
            }
          }
        }
        if (!ready_events.empty())
          Runtime::trigger_event(done, Runtime::merge_events(ready_events));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionFailed::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<unsigned>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);

      target->fetch_add(1);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::process_remote_reference_mismatch(
        uint64_t sent, uint64_t received, LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock i_lock(inst_lock);
      legion_assert(is_owner());
      legion_assert(gc_state != COLLECTED_GC_STATE);
      sent_valid_references += sent;
      received_valid_references += received;
      // Aggregate the remote node's clock so the owner's collection decision
      // can detect a packed valid reference newer than this round's snapshot
      collect_lamport_clock = std::max(collect_lamport_clock, lamport_clock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionMismatch::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      uint64_t remote_sent, remote_received;
      LamportClock remote_clock;
      derez.deserialize(remote_sent);
      derez.deserialize(remote_received);
      derez.deserialize(remote_clock);
      RtUserEvent done;
      derez.deserialize(done);
      // Should still be able to find this manager here
      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      manager->process_remote_reference_mismatch(
          remote_sent, remote_received, remote_clock);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionNotification::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);

      // Should still be able to find this manager here
      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));
      manager->notify_remote_deletion();
      manager->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::notify_remote_deletion(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      // Forward on the deletion notification to any children
      if ((collective_mapping != nullptr) &&
          collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        if (!children.empty())
        {
          GarbageCollectionNotification rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            pack_global_ref(rez, children.size());
          }
          for (const AddressSpaceID& child : children) rez.dispatch(child);
        }
      }
      std::set<InstanceDeletionSubscriber*> to_notify;
      {
        AutoLock i_lock(inst_lock);
        legion_assert(
            (gc_state == COLLECTED_GC_STATE) ||
            (gc_state == PENDING_COLLECTED_GC_STATE));
        gc_state = COLLECTED_GC_STATE;
        to_notify.swap(subscribers);
      }
      if (!to_notify.empty())
      {
        for (InstanceDeletionSubscriber* subscriber : to_notify)
        {
          subscriber->notify_instance_deletion(this);
          if (subscriber->remove_subscriber_reference(this))
            delete subscriber;
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::pack_garbage_collection_state(
        Serializer& rez, AddressSpaceID target, bool need_lock)
    //--------------------------------------------------------------------------
    {
      // We have to atomically get the current collection state and
      // update the set of remote instances, note that it can be read-only
      // since we're just reading the state and the `update-remote_instaces'
      // call will take its own exclusive lock
      if (need_lock)
      {
        AutoLock i_lock(inst_lock, false /*exclusive*/);
        pack_garbage_collection_state(rez, target, false /*need lock*/);
      }
      else
      {
        switch (gc_state.load())
        {
          case VALID_GC_STATE:
          case COLLECTABLE_GC_STATE:
            {
              rez.serialize(COLLECTABLE_GC_STATE);
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
          case COLLECTED_GC_STATE:
            {
              rez.serialize(gc_state);
              break;
            }
          default:
            std::abort();
        }
        update_remote_instances(target);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::is_collected(void) const
    //--------------------------------------------------------------------------
    {
      return (gc_state == COLLECTED_GC_STATE);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::collect(
        RtEvent& ready, PhysicalInstance* hole, AutoLock* i_lock)
    //--------------------------------------------------------------------------
    {
      legion_assert((hole == nullptr) || !hole->exists());
      if (i_lock == nullptr)
      {
        AutoLock i2_lock(inst_lock);
        return collect(ready, hole, &i2_lock);
      }
      // Do a quick to check to see if we can do a collection on the local node
      if (gc_state == VALID_GC_STATE)
        return false;
      // If it's already collected then we're done
      if (gc_state == COLLECTED_GC_STATE)
        return true;
      bool has_local_references = false;
      uint64_t local_valid_sent = 0, local_valid_received = 0;
      if (is_owner())
      {
        // Check to see if anyone is already performing a deletion
        // on this manager, if so then deduplicate
        if (gc_state == COLLECTABLE_GC_STATE)
        {
          gc_state = PENDING_COLLECTED_GC_STATE;
          failed_collection_count.store(0);
          // Start a new collection round: advance our clock to obtain this
          // round's snapshot and arm the bump so that any valid reference we
          // pack after freezing our counts advances the clock past the
          // snapshot and is therefore detectable as a causality violation.
          pending_collect_lamport_clock = ++collect_lamport_clock;
          bump_collect_lamport_clock = true;
          // Pull a copy of these onto the stack in case we fail to
          // collect and we need to restore them
          local_valid_sent = sent_valid_references;
          local_valid_received = received_valid_references;
          has_local_references = true;
          std::vector<RtEvent> ready_events;
          if (collective_mapping != nullptr)
          {
            // We're the owner so it should contain ourselves
            legion_assert(collective_mapping->contains(local_space));
            std::vector<AddressSpaceID> children;
            collective_mapping->get_children(
                owner_space, local_space, children);
            for (const AddressSpaceID& child : children)
            {
              const RtUserEvent ready_event = Runtime::create_rt_user_event();
              GarbageCollectionAcquire rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(&failed_collection_count);
                rez.serialize(ready_event);
                rez.serialize(pending_collect_lamport_clock);
              }
              rez.dispatch(child);
              ready_events.emplace_back(ready_event);
            }
          }
          if (has_remote_instances())
          {
            struct AcquireFunctor {
              AcquireFunctor(
                  DistributedID d, std::vector<RtEvent>& r,
                  std::atomic<unsigned>* c, LamportClock clk)
                : did(d), ready_events(r), count(c), clock(clk)
              { }
              inline void apply(AddressSpaceID target)
              {
                if (target == runtime->address_space)
                  return;
                const RtUserEvent ready_event = Runtime::create_rt_user_event();
                GarbageCollectionAcquire rez;
                {
                  RezCheck z(rez);
                  rez.serialize(did);
                  rez.serialize(count);
                  rez.serialize(ready_event);
                  rez.serialize(clock);
                }
                rez.dispatch(target);
                ready_events.emplace_back(ready_event);
              }
              const DistributedID did;
              std::vector<RtEvent>& ready_events;
              std::atomic<unsigned>* const count;
              const LamportClock clock;
            };
            AcquireFunctor functor(
                did, ready_events, &failed_collection_count,
                pending_collect_lamport_clock);
            map_over_remote_instances(functor);
          }
          if (!ready_events.empty())
            collection_ready = Runtime::merge_events(ready_events);
        }
        else
        {
          legion_assert(gc_state == PENDING_COLLECTED_GC_STATE);
          // Should alaready have outstanding changes for this deletion
          legion_assert(pending_changes > 0);
        }
        pending_changes++;
        const RtEvent wait_on = collection_ready;
        if (!wait_on.has_triggered())
        {
          i_lock->release();
          wait_on.wait();
          i_lock->reacquire();
        }
        legion_assert(pending_changes > 0);
        switch (gc_state.load())
        {
          // Anything in these states means the collection attempt failed
          // because something else acquired a valid reference while
          // the collection was in progress
          case VALID_GC_STATE:
          case COLLECTABLE_GC_STATE:
            {
              // Restore our local sent/received counts
              if (has_local_references)
              {
                sent_valid_references = local_valid_sent;
                received_valid_references = local_valid_received;
              }
              break;
            }
          case PENDING_COLLECTED_GC_STATE:
            {
              // Precondition should have triggered if we're here
              legion_assert(collection_ready.has_triggered());
              // Check to see if there were any collection guards we
              // were unable to acquire on remote nodes or whether there
              // are still packed valid reference outstanding
              if ((failed_collection_count.load() > 0) ||
                  (sent_valid_references != received_valid_references) ||
                  (collect_lamport_clock > pending_collect_lamport_clock))
              {
                // Restore our local sent/received counts (but not the clock,
                // which stays monotonic so the next round's snapshot exceeds
                // any reference that caused this round to fail, guaranteeing
                // forward progress once the traffic quiesces).
                if (has_local_references)
                {
                  sent_valid_references = local_valid_sent;
                  received_valid_references = local_valid_received;
                }
                // See if we're the last release, if not then we
                // keep it in this state
                if (--pending_changes == 0)
                  gc_state = COLLECTABLE_GC_STATE;
              }
              else
              {
                // Deletion success and we're the first ones to discover it
                // Move to the deletion state and send the deletion messages
                // to mark that we successfully performed the deletion
                // Grab the set of active contexts to notify
                std::set<InstanceDeletionSubscriber*> to_notify;
                // Notify the subscribers if we've been collected
                to_notify.swap(subscribers);
                // Now we can perform the deletion which will release the lock
                RtEvent hole_ready =
                    perform_deletion(runtime->address_space, hole, i_lock);
                // Only save the event for the whole being ready if we have one
                if ((hole != nullptr) && hole->exists())
                  ready = hole_ready;
                // Send notification messages to the remote nodes to tell
                // them that this instance has been deleted, this is needed
                // so that we can invalidate any subscribers on those nodes
                if (collective_mapping != nullptr)
                {
                  // We're the owner so it should contain ourselves
                  legion_assert(collective_mapping->contains(local_space));
                  std::vector<AddressSpaceID> children;
                  collective_mapping->get_children(
                      owner_space, local_space, children);
                  if (!children.empty())
                  {
                    GarbageCollectionNotification rez;
                    {
                      RezCheck z(rez);
                      rez.serialize(did);
                      pack_global_ref(rez, children.size());
                    }
                    for (const AddressSpaceID& child : children)
                      rez.dispatch(child);
                  }
                }
                if (has_remote_instances())
                {
                  struct NotifyFunctor {
                    NotifyFunctor(DistributedID d, PhysicalManager* m)
                      : did(d), manager(m)
                    { }
                    inline void apply(AddressSpaceID target)
                    {
                      if (target == runtime->address_space)
                        return;
                      GarbageCollectionNotification rez;
                      {
                        RezCheck z(rez);
                        rez.serialize(did);
                        manager->pack_global_ref(rez);
                      }
                      rez.dispatch(target);
                    }
                    const DistributedID did;
                    PhysicalManager* manager;
                    unsigned count;
                  };
                  NotifyFunctor functor(did, this);
                  map_over_remote_instances(functor);
                }
                // Now that the lock is released we can notify the subscribers
                if (!to_notify.empty())
                {
                  for (InstanceDeletionSubscriber* subscriber : to_notify)
                  {
                    subscriber->notify_instance_deletion(this);
                    if (subscriber->remove_subscriber_reference(this))
                      delete subscriber;
                  }
                }
                return true;
              }
              break;
            }
          case COLLECTED_GC_STATE:
            {
              // Save the event for when the collection is done
              ready = collection_ready;
              return true;
            }
          default:
            std::abort();  // should not be in any other state
        }
        return false;
      }
      else
      {
        // No longer need the lock here since we're just sending a message
        i_lock->release();
        // Send it to the owner to check
        std::atomic<bool> result(false);
        const RtUserEvent done = Runtime::create_rt_user_event();
        GarbageCollectionRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(&result);
          rez.serialize(&ready);
          rez.serialize(hole);
          rez.serialize(done);
          pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
        done.wait();
        i_lock->reacquire();
        return result.load();
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::set_garbage_collection_priority(
        MapperID mapper_id, Processor p, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_external_instance());
      RtUserEvent done_event;
      RtEvent wait_on, updated;
      bool remove_never_reference = false;
      bool broadcast_priority_update = false;
      {
        const std::pair<MapperID, Processor> key(mapper_id, p);
        AutoLock i_lock(inst_lock);
        // If this thing is already deleted then there is nothing to do
        if (gc_state == COLLECTED_GC_STATE)
          return RtEvent::NO_RT_EVENT;
        if (mapper_gc_priorities.empty())
        {
          // First mapper priority to be set which means we need to update
          // the min_gc_priority to be the initial value
          mapper_gc_priorities.emplace(std::make_pair(key, priority));
          // Always fall through to send the update because we were
          // effectively in an uninitialized state before
        }
        else
        {
          std::map<std::pair<MapperID, Processor>, GCPriority>::iterator
              finder = mapper_gc_priorities.find(key);
          if (finder == mapper_gc_priorities.end())
          {
            mapper_gc_priorities[key] = priority;
            if (min_gc_priority <= priority)
              return RtEvent::NO_RT_EVENT;
          }
          else
          {
            // See if we're the minimum priority
            if (min_gc_priority < finder->second)
            {
              // We weren't one of the minimum priorities before
              finder->second = priority;
              if (min_gc_priority <= priority)
                return RtEvent::NO_RT_EVENT;
              // Otherwise fall through and update the min priority
            }
            else
            {
              // We were one of the minimum priorities before
              legion_assert(finder->second == min_gc_priority);
              // If things don't change then there is nothing to do
              if (finder->second == priority)
                return RtEvent::NO_RT_EVENT;
              finder->second = priority;
              if (min_gc_priority < priority)
              {
                // Raising one of the old minimum priorities
                // See what the new min priority is
                for (const std::pair<
                         const std::pair<MapperID, Processor>, GCPriority>& it :
                     mapper_gc_priorities)
                {
                  // If the new minimum priority is still the same we're done
                  if (it.second == min_gc_priority)
                    return RtEvent::NO_RT_EVENT;
                  if (it.second < priority)
                    priority = it.second;
                }
                // If we get here then we're increasing the minimum priority
                legion_assert(min_gc_priority < priority);
              }
              // Else lowering the minimum priority
            }
          }
          // If we get here then we're changing the minimum priority
          legion_assert(priority != min_gc_priority);
        }
        // Only deal with never collection refs on the owner node where
        // the ultimate garbage collection decisions are to be made
        if (is_owner())
        {
          if (priority < min_gc_priority)
          {
            // Transitioning to a smaller priority
            legion_assert(LEGION_GC_NEVER_PRIORITY < min_gc_priority);
            if (priority == LEGION_GC_NEVER_PRIORITY)
            {
              // Check the garbage collection state because this is going
              // to be like an acquire operation
              switch (gc_state.load())
              {
                case VALID_GC_STATE:
                  break;
                case COLLECTABLE_GC_STATE:
                // Garbage collector is trying to eat it, save it!
                case PENDING_COLLECTED_GC_STATE:
                  {
                    gc_state = VALID_GC_STATE;
                    break;
                  }
                default:
                  std::abort();
              }
                // Update the references
#ifdef LEGION_GC
              log_base_ref<true>(
                  VALID_REF_KIND, did, local_space, NEVER_GC_REF, 1);
#endif
#ifdef LEGION_DEBUG_GC
              valid_references++;
              std::map<ReferenceSource, int>::iterator finder =
                  detailed_base_valid_references.find(NEVER_GC_REF);
              if (finder == detailed_base_valid_references.end())
                detailed_base_valid_references[NEVER_GC_REF] = 1;
              else
                finder->second++;
#else
              valid_references.fetch_add(1);
#endif
            }
            if (min_gc_priority == LEGION_GC_EAGER_PRIORITY)
              // Tell the remote nodes they no longer need to
              // check for each deletion
              broadcast_priority_update = true;
          }
          else if (min_gc_priority < priority)
          {
            // Transitioning to a larger priority
            if (priority == LEGION_GC_EAGER_PRIORITY)
            {
              // If we're eagerly collectable then we try
              // to delete this now, otherwise eager priority
              // needs to be broadcasted out to all other nodes
              // in case they become locally valid and then
              // invalid they need to know to check for that
              // as soon as they see it
              if (!collect(updated, nullptr, &i_lock))
                broadcast_priority_update = true;
            }
            if (min_gc_priority == LEGION_GC_NEVER_PRIORITY)
              remove_never_reference = true;
          }
          // Else we were uninitialize before and this is the
          // first time a mapper has set the priority
        }
        min_gc_priority = priority;
        // Make an event for when the priority updates are done
        wait_on = priority_update_done;
        done_event = Runtime::create_rt_user_event();
        priority_update_done = done_event;
      }
      // If we make it here then we need to do the update
      if (wait_on.exists() && !wait_on.has_triggered())
        wait_on.wait();
      // Perform any updates for this priority
      if (is_owner())
      {
        memory_manager->set_garbage_collection_priority(this, priority);
        if (broadcast_priority_update)
          updated = broadcast_garbage_collection_priority_update(priority);
      }
      else
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        GarbageCollectionPriorityUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(priority);
          rez.serialize(done);
          rez.serialize<bool>(false);  // broadcast
          pack_global_ref(rez);
        }
        rez.dispatch(owner_space);
        updated = done;
      }
      if (remove_never_reference && remove_base_valid_ref(NEVER_GC_REF))
        std::abort();  // should never end up deleting ourselves
      Runtime::trigger_event(done_event, updated);
      return done_event;
    }

    //--------------------------------------------------------------------------
    /*static*/ void GarbageCollectionPriorityUpdate::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      GCPriority priority;
      derez.deserialize(priority);
      RtUserEvent done;
      derez.deserialize(done);
      bool broadcast;
      derez.deserialize<bool>(broadcast);

      PhysicalManager* manager = static_cast<PhysicalManager*>(
          runtime->find_distributed_collectable(did));

      if (!broadcast)
      {
        // To avoid collisiions with existing local mappers which could lead
        // to aliasing of priority updates, we use "invalid" processor IDs
        // here that will never conflict with existing processor IDs
        // Note that the NO_PROC is a valid processor ID for mappers in the
        // case where the mapper handles all the processors in a node. We
        // therefore always add the owner address space to the source to
        // produce a non-zero processor ID. Note that this formulation also
        // avoid conflicts from different remote sources.
        const Processor fake_proc(source + manager->owner_space);
        legion_assert(fake_proc.id != 0);
        Runtime::trigger_event(
            done, manager->set_garbage_collection_priority(
                      0 /*default mapper ID*/, fake_proc, priority));
      }
      else
        Runtime::trigger_event(
            done,
            manager->broadcast_garbage_collection_priority_update(priority));
      manager->unpack_global_ref(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ManagerRequestMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      PhysicalManager* manager = legion_safe_cast<PhysicalManager*>(dc);
      manager->send_manager(source);
    }

    //--------------------------------------------------------------------------
    size_t PhysicalManager::get_instance_size(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock lock(inst_lock, false /*exclusive*/);
      return instance_footprint;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalManager::fetch_metadata(ApEvent use_event) const
    //--------------------------------------------------------------------------
    {
      ApEvent ready(
          instance.fetch_metadata(Processor::get_executing_processor()));
      if (!use_event.exists())
        return ready;
      if (implicit_profiler != nullptr)
        implicit_profiler->record_fetch_metadata(ready, unique_event);
      if (!ready.exists())
        return use_event;
      return Runtime::merge_events(nullptr, ready, use_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateTopViewRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent man_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, man_ready);
      InnerContext* context = InnerContext::unpack_inner_context(derez);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      CollectiveMapping* mapping = nullptr;
      size_t total_spaces;
      derez.deserialize(total_spaces);
      if (total_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, total_spaces);
        mapping->add_reference();
      }
      std::atomic<DistributedID>* target;
      derez.deserialize(target);
      RtUserEvent done;
      derez.deserialize(done);
      // See if we're ready or we need to defer this until later
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        PhysicalManager::RemoteCreateViewArgs args(
            manager, context, logical_owner, mapping, target, source, done);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, man_ready);
        return;
      }
      PhysicalManager::process_top_view_request(
          manager, context, logical_owner, mapping, target, source, done);
      if ((mapping != nullptr) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::process_top_view_request(
        PhysicalManager* manager, InnerContext* context, AddressSpaceID logical,
        CollectiveMapping* mapping, std::atomic<DistributedID>* target,
        AddressSpaceID source, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // Get the view from the context
      InstanceView* view =
          context->create_instance_top_view(manager, logical, mapping);
      CreateTopViewResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(view->did);
        rez.serialize(done_event);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CreateTopViewResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::atomic<DistributedID>* target;
      derez.deserialize(target);
      DistributedID did;
      derez.deserialize(did);
      target->store(did);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::RemoteCreateViewArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      process_top_view_request(
          manager, context, logical_owner, mapping, target, source, done_event);
      if ((mapping != nullptr) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::initialize_across_helper(
        CopyAcrossHelper* helper, const FieldMask& dst_mask,
        const std::vector<unsigned>& src_indexes,
        const std::vector<unsigned>& dst_indexes)
    //--------------------------------------------------------------------------
    {
      // Make sure the instance is ready before we compute the offsets
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
      legion_assert(src_indexes.size() == dst_indexes.size());
      legion_assert(helper->offsets.size() == dst_indexes.size());
      std::vector<CopySrcDstField> dst_fields;
      layout->compute_copy_offsets(dst_mask, instance, dst_fields);
      // This is a double translation that we need to do, for each
      // field in the destination mask, we need to find out its index
      // in the dst_indexes and then use that same index in the src_indexes
      // to specify where to put the CopySrcDstField data
      const unsigned pop_count = dst_mask.pop_count();
      int next_start = 0;
      for (unsigned idx = 0; idx < pop_count; idx++)
      {
        const int index = dst_mask.find_next_set(next_start);
        legion_assert(index >= 0);
        const unsigned dst_index = index;
        // Now we need to find the location of index in the dst_indexes
        for (unsigned loc = 0; loc < dst_indexes.size(); loc++)
        {
          if (dst_index != dst_indexes[loc])
            continue;
          helper->offsets[loc] = dst_fields[idx];
          next_start = index + 1;
          break;
        }
        // Should have been found during the traversal, if not this will fail
        legion_assert(next_start == (index + 1));
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      legion_assert(
          (collective_mapping == nullptr) ||
          !collective_mapping->contains(target));
      InstanceManagerMessage rez;
      {
        AutoLock lock(inst_lock, false /*exclusive*/);
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(memory_manager->memory);
        rez.serialize(instance);
        rez.serialize(instance_footprint);
        // No need for a reference here since we know we'll continue holding it
        instance_domain->pack_expression(rez, target);
        rez.serialize(piece_list_size);
        if (piece_list_size > 0)
          rez.serialize(piece_list, piece_list_size);
        rez.serialize(field_space_node->handle);
        rez.serialize(tree_id);
        rez.serialize(unique_event);
        if (kind != UNBOUND_INSTANCE_KIND)
          rez.serialize<ApEvent>(use_event);
        else
          rez.serialize(producer_event);
        layout->pack_layout_description(rez, target);
        rez.serialize(redop);
        rez.serialize(kind);
        pack_garbage_collection_state(rez, target, false /*need lock*/);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceManagerMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Memory mem;
      derez.deserialize(mem);
      PhysicalInstance inst;
      derez.deserialize(inst);
      size_t inst_footprint;
      derez.deserialize(inst_footprint);
      IndexSpaceExpression* inst_domain =
          IndexSpaceExpression::unpack_expression(derez, source);
      size_t piece_list_size;
      derez.deserialize(piece_list_size);
      void* piece_list = nullptr;
      if (piece_list_size > 0)
      {
        piece_list = malloc(piece_list_size);
        derez.deserialize(piece_list, piece_list_size);
      }
      FieldSpace handle;
      derez.deserialize(handle);
      RtEvent fs_ready;
      FieldSpaceNode* space_node = runtime->get_node(handle, &fs_ready);
      RegionTreeID tree_id;
      derez.deserialize(tree_id);
      LgEvent unique_event;
      derez.deserialize(unique_event);
      ApEvent use_event;
      derez.deserialize(use_event);
      LayoutConstraintID layout_id;
      derez.deserialize(layout_id);
      RtEvent layout_ready;
      LayoutConstraints* constraints = runtime->find_layout_constraints(
          layout_id, false /*can fail*/, &layout_ready);
      ReductionOpID redop;
      derez.deserialize(redop);
      PhysicalManager::InstanceKind kind;
      derez.deserialize(kind);
      PhysicalManager::GarbageCollectionState gc_state;
      derez.deserialize(gc_state);

      if (fs_ready.exists() || layout_ready.exists())
      {
        const RtEvent precondition =
            Runtime::merge_events(fs_ready, layout_ready);
        if (precondition.exists() && !precondition.has_triggered())
        {
          // We need to defer this instance creation
          PhysicalManager::DeferPhysicalManagerArgs args(
              did, mem, inst, inst_footprint, inst_domain, handle, tree_id,
              layout_id, use_event, unique_event, kind, redop, piece_list,
              piece_list_size, gc_state);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_RESPONSE_PRIORITY, precondition);
          return;
        }
        // If we fall through we need to refetch things that we didn't get
        if (fs_ready.exists())
          space_node = runtime->get_node(handle);
        if (layout_ready.exists())
          constraints =
              runtime->find_layout_constraints(layout_id, false /*can fail*/);
      }
      // If we fall through here we can create the manager now
      PhysicalManager::create_remote_manager(
          did, mem, inst, inst_footprint, inst_domain, piece_list,
          piece_list_size, space_node, tree_id, constraints, use_event,
          unique_event, kind, redop, gc_state);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferPhysicalManagerArgs::DeferPhysicalManagerArgs(
        DistributedID d, Memory m, PhysicalInstance i, size_t f,
        IndexSpaceExpression* lx, FieldSpace fs, RegionTreeID tid,
        LayoutConstraintID l, ApEvent use, LgEvent unique, InstanceKind k,
        ReductionOpID r, const void* pl, size_t pl_size,
        GarbageCollectionState gc)
      : LgTaskArgs<DeferPhysicalManagerArgs>(false, true), did(d), mem(m),
        inst(i), footprint(f), local_expr(lx), space(fs), tree_id(tid),
        layout_id(l), use_event(use), unique_event(unique), kind(k), redop(r),
        piece_list(pl), piece_list_size(pl_size), state(gc)
    //--------------------------------------------------------------------------
    {
      local_expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalManager::DeferDeletePhysicalManager ::DeferDeletePhysicalManager(
        PhysicalManager* manager_)
      : LgTaskArgs<DeferDeletePhysicalManager>(true, true), manager(manager_)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PhysicalManager::DeferPhysicalManagerArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode* space_node = runtime->get_node(space);
      LayoutConstraints* constraints =
          runtime->find_layout_constraints(layout_id);
      create_remote_manager(
          did, mem, inst, footprint, local_expr, piece_list, piece_list_size,
          space_node, tree_id, constraints, use_event, unique_event, kind,
          redop, state);
      // Remove the local expression reference if necessary
      if (local_expr->remove_base_expression_reference(META_TASK_REF))
        delete local_expr;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::DeferDeletePhysicalManager::execute(void) const
    //--------------------------------------------------------------------------
    {
      manager->memory_manager->unregister_deleted_instance(manager);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManager::create_remote_manager(
        DistributedID did, Memory mem, PhysicalInstance inst,
        size_t inst_footprint, IndexSpaceExpression* inst_domain,
        const void* piece_list, size_t piece_list_size,
        FieldSpaceNode* space_node, RegionTreeID tree_id,
        LayoutConstraints* constraints, ApEvent use_event, LgEvent unique_event,
        InstanceKind kind, ReductionOpID redop, GarbageCollectionState state)
    //--------------------------------------------------------------------------
    {
      LayoutDescription* layout =
          LayoutDescription::handle_unpack_layout_description(
              constraints, space_node, inst_domain->get_num_dims());
      MemoryManager* memory = runtime->find_memory_manager(mem);
      const ReductionOp* op =
          (redop == 0) ? nullptr : runtime->get_reduction(redop);
      void* location =
          runtime->find_or_create_pending_collectable_location<PhysicalManager>(
              did);
      PhysicalManager* man = new (location) PhysicalManager(
          did, memory, inst, inst_domain, piece_list, piece_list_size,
          space_node, tree_id, layout, redop, false /*reg now*/, inst_footprint,
          use_event, unique_event, kind, op, nullptr, ApEvent::NO_AP_EVENT,
          state);
      // Hold-off doing the registration until construction is complete
      man->register_with_runtime();
      // Remove the reference we got back on the layout description
      if (layout->remove_reference())
        delete layout;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::perform_deletion(
        AddressSpaceID source, PhysicalInstance* hole,
        AutoLock* i_lock /* = nullptr*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      legion_assert(source == local_space);
      if (i_lock == nullptr)
      {
        AutoLock instance_lock(inst_lock);
        return perform_deletion(source, hole, &instance_lock);
      }
      legion_assert(pending_views.empty());
      legion_assert(gc_state != COLLECTED_GC_STATE);
      gc_state = COLLECTED_GC_STATE;
      log_garbage.spew(
          "Deleting physical instance " IDFMT " in memory " IDFMT "",
          instance.id, memory_manager->memory.id);
      RtEvent deferred_deletion;
#ifndef LEGION_DISABLE_GC
      // Get the deferred deletion event from the gc events
      if (!gc_events.empty())
        deferred_deletion = Runtime::protect_merge_events(gc_events);
      // Now we can release the lock since we're done with the atomic updates
      i_lock->release();
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
      // Handle a small race where the instance name is not ready yet
      if (instance_ready.exists() && !instance_ready.has_triggered())
        instance_ready.wait();
#ifdef LEGION_MALLOC_INSTANCES
      if (kind == INTERNAL_INSTANCE_KIND)
        memory_manager->free_legion_instance(this, deferred_deletion);
#else
      // We can't escape instances with serdez fields since we need to
      // delete them explicity but everything else we can escape
      if (!serdez_fields.empty())
        instance.destroy(serdez_fields, deferred_deletion);
      else if (hole != nullptr)
        *hole = instance;  // escape the hole to use for redistricting
      else
        instance.destroy(deferred_deletion);
#endif
#else
      // Release the i_lock since we're done with the atomic updates
      i_lock->release();
#endif
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (!deferred_deletion.exists())
          Runtime::rename_event(deferred_deletion);
        for (const ApEvent& event : gc_events)
          LegionSpy::log_event_dependence(event, deferred_deletion);
        LegionSpy::log_instance_deletion(unique_event, deferred_deletion);
      }
      // Once the deletion is actually done then we can tell the memory
      // manager that the deletion is finished and it is safe to remove
      // this manager for its list of current instances
      if (deferred_deletion.exists() && !deferred_deletion.has_triggered())
      {
        DeferDeletePhysicalManager args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LOW_PRIORITY, deferred_deletion);
      }
      else
        memory_manager->unregister_deleted_instance(this);
      return deferred_deletion;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::force_deletion(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      // If we already deleted this then it's deferred deletion task should
      // have run and pruned it out of the memory manager to prevent calling
      // force_deletion on this physical manager
      legion_assert(gc_state != COLLECTED_GC_STATE);
      // There are no races by the time we get here so we don't need the lock
      log_garbage.spew(
          "Force deleting physical instance " IDFMT " in memory " IDFMT "",
          instance.id, memory_manager->memory.id);
#ifndef LEGION_DISABLE_GC
      std::vector<PhysicalInstance::DestroyedField> serdez_fields;
      layout->compute_destroyed_fields(serdez_fields);
#ifdef LEGION_MALLOC_INSTANCES
      if (kind == INTERNAL_INSTANCE_KIND)
        memory_manager->free_legion_instance(this, RtEvent::NO_RT_EVENT);
#else
      if (!serdez_fields.empty())
        instance.destroy(serdez_fields);
      else
        instance.destroy();
#endif
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::broadcast_garbage_collection_priority_update(
        GCPriority priority)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> done_events;
      // Send out the messages to perform the broadcast
      if ((collective_mapping != nullptr) &&
          collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(owner_space, local_space, children);
        for (const AddressSpaceID& child : children)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          GarbageCollectionPriorityUpdate rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(priority);
            rez.serialize(done);
            rez.serialize<bool>(true);  // broadcast
            pack_global_ref(rez);
          }
          rez.dispatch(child);
          done_events.emplace_back(done);
        }
      }
      if (is_owner() && has_remote_instances())
      {
        struct UpdateFunctor {
          UpdateFunctor(
              PhysicalManager* m, std::vector<RtEvent>& d, GCPriority p)
            : manager(m), done_events(d), priority(p)
          { }
          inline void apply(AddressSpaceID target)
          {
            if (target == runtime->address_space)
              return;
            const RtUserEvent done = Runtime::create_rt_user_event();
            GarbageCollectionPriorityUpdate rez;
            {
              RezCheck z(rez);
              rez.serialize(manager->did);
              rez.serialize(priority);
              rez.serialize(done);
              rez.serialize<bool>(true);  // broadcast
              manager->pack_global_ref(rez);
            }
            rez.dispatch(target);
            done_events.emplace_back(done);
          }
          PhysicalManager* const manager;
          std::vector<RtEvent>& done_events;
          const GCPriority priority;
        };
        UpdateFunctor functor(this, done_events, priority);
        map_over_remote_instances(functor);
      }
      RtEvent result;
      if (!done_events.empty())
        result = Runtime::merge_events(done_events);
      // Take the lock and perform our local update
      AutoLock i_lock(inst_lock);
      if (priority != LEGION_GC_EAGER_PRIORITY)
      {
        // If we have mapper opinions we can reset this to whatever their
        // current opinions are, otherwise we set it to whatever the new
        // priority is
        if (!mapper_gc_priorities.empty())
        {
          for (const std::pair<
                   const std::pair<MapperID, Processor>, GCPriority>& it :
               mapper_gc_priorities)
            if (it.second < min_gc_priority)
              min_gc_priority = it.second;
        }
        else
          min_gc_priority = priority;
      }
      else if (mapper_gc_priorities.empty())
      {
        // Only need to set the to eager priority if there are no mapper
        // opinions because if there are mapper opinions either they
        // already set us to eager priority or they've changed their
        // minds in between and we've lost the race
        min_gc_priority = LEGION_GC_EAGER_PRIORITY;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalManager::attach_external_instance(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_external_instance());
      return memory_manager->attach_external_instance(this);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::detach_external_instance(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_external_instance());
      memory_manager->detach_external_instance(this);
    }

    //--------------------------------------------------------------------------
    uintptr_t PhysicalManager::get_instance_pointer(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      if (use_event.exists() && !use_event.has_triggered_faultignorant())
        use_event.wait_faultignorant();
      void* inst_ptr = instance.pointer_untyped(0 /*offset*/, 0 /*elem size*/);
      return uintptr_t(inst_ptr);
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::has_visible_from(const std::set<Memory>& mems) const
    //--------------------------------------------------------------------------
    {
      return (mems.find(memory_manager->memory) != mems.end());
    }

    //--------------------------------------------------------------------------
    bool PhysicalManager::update_physical_instance(
        PhysicalInstance new_instance, RtEvent ready, size_t new_footprint)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock lock(inst_lock);
        legion_assert(kind == UNBOUND_INSTANCE_KIND);
        legion_assert(instance_footprint == -1U);
        legion_assert(piece_list_size == 0);
        instance = new_instance;
        kind = INTERNAL_INSTANCE_KIND;
        instance_footprint = new_footprint;

        Runtime::trigger_event(instance_ready, ready);

        if (spy_logging_level > NO_SPY_LOGGING)
        {
          LegionSpy::log_physical_instance(
              unique_event, instance.id, memory_manager->memory.id,
              instance_domain->expr_id, field_space_node->handle, tree_id,
              redop);
          layout->log_instance_layout(unique_event);
        }
        if (is_owner() && (runtime->profiler != nullptr))
          implicit_profiler->register_physical_instance_spaces(
              unique_event, instance_domain->record_profiler_expression(),
              0 /*no pieces*/);

        if (is_owner() && has_remote_instances())
          broadcast_manager_update();

        Runtime::trigger_event_untraced(
            use_event, fetch_metadata(producer_event));
      }
      return remove_base_valid_ref(PENDING_UNBOUND_REF);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::broadcast_manager_update(void)
    //--------------------------------------------------------------------------
    {
      PhysicalManagerUpdate rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(instance);
        rez.serialize(instance_ready);
        rez.serialize(instance_footprint);
      }
      struct BroadcastFunctor {
        BroadcastFunctor(PhysicalManagerUpdate& r) : rez(r) { }
        inline void apply(AddressSpaceID target) { rez.dispatch(target); }
        PhysicalManagerUpdate& rez;
      };
      BroadcastFunctor functor(rez);
      map_over_remote_instances(functor);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalManagerUpdate::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      PhysicalInstance instance;
      derez.deserialize(instance);
      RtEvent ready;
      derez.deserialize(ready);
      size_t footprint;
      derez.deserialize(footprint);

      RtEvent manager_ready;
      PhysicalManager* manager =
          runtime->find_or_request_instance_manager(did, manager_ready);
      if (manager_ready.exists() && !manager_ready.has_triggered())
        manager_ready.wait();

      if (manager->update_physical_instance(instance, ready, footprint))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_padded_reservations(
        const FieldMask& mask, Operation* op, unsigned index)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> reservations(mask.pop_count());
      find_field_reservations(mask, reservations);
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        op->update_atomic_locks(index, reservations[idx], true /*exclusive*/);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::find_field_reservations(
        const FieldMask& mask, std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
      legion_assert(mask.pop_count() == reservations.size());
      unsigned offset = 0;
      if (is_owner())
      {
        AutoLock i_lock(inst_lock);
        if (padded_reservations == nullptr)
          padded_reservations = new std::map<unsigned, Reservation>();
        for (int idx = mask.find_first_set(); idx >= 0;
             idx = mask.find_next_set(idx + 1))
        {
          std::map<unsigned, Reservation>::const_iterator finder =
              padded_reservations->find(idx);
          if (finder == padded_reservations->end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            padded_reservations->insert(std::make_pair(idx, handle));
            reservations[offset++] = handle;
          }
          else
            reservations[offset++] = finder->second;
        }
      }
      else
      {
        // Figure out which fields we need requests for and send them
        FieldMask needed_fields;
        {
          AutoLock i_lock(inst_lock, false /*exclusive*/);
          if (padded_reservations == nullptr)
          {
            for (int idx = mask.find_first_set(); idx >= 0;
                 idx = mask.find_next_set(idx + 1))
              needed_fields.set_bit(idx);
          }
          else
          {
            for (int idx = mask.find_first_set(); idx >= 0;
                 idx = mask.find_next_set(idx + 1))
            {
              std::map<unsigned, Reservation>::const_iterator finder =
                  padded_reservations->find(idx);
              if (finder == padded_reservations->end())
                needed_fields.set_bit(idx);
              else
                reservations[offset++] = finder->second;
            }
          }
        }
        if (!!needed_fields)
        {
          RtUserEvent wait_on = Runtime::create_rt_user_event();
          PaddedReservationRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(needed_fields);
            rez.serialize(wait_on);
          }
          rez.dispatch(owner_space);
          wait_on.wait();
          // Now retake the lock and get the remaining reservations
          AutoLock i_lock(inst_lock, false /*exclusive*/);
          legion_assert(padded_reservations != nullptr);
          for (int idx = needed_fields.find_first_set(); idx >= 0;
               idx = needed_fields.find_next_set(idx + 1))
          {
            std::map<unsigned, Reservation>::const_iterator finder =
                padded_reservations->find(idx);
            legion_assert(finder != padded_reservations->end());
            reservations[offset++] = finder->second;
          }
        }
      }
      legion_assert(offset == reservations.size());
      // Sort them before returning
      if (reservations.size() > 1)
        std::sort(reservations.begin(), reservations.end());
    }

    //--------------------------------------------------------------------------
    /*static*/ void PaddedReservationRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask needed_fields;
      derez.deserialize(needed_fields);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      PhysicalManager* target = legion_safe_cast<PhysicalManager*>(dc);
      std::vector<Reservation> reservations(needed_fields.pop_count());
      target->find_field_reservations(needed_fields, reservations);
      PaddedReservationResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(needed_fields);
        for (unsigned idx = 0; idx < reservations.size(); idx++)
          rez.serialize(reservations[idx]);
        rez.serialize(to_trigger);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    void PhysicalManager::update_field_reservations(
        const FieldMask& mask, const std::vector<Reservation>& reservations)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      legion_assert(mask.pop_count() == reservations.size());
      unsigned offset = 0;
      AutoLock i_lock(inst_lock);
      if (padded_reservations == nullptr)
        padded_reservations = new std::map<unsigned, Reservation>();
      for (int idx = mask.find_first_set(); idx >= 0;
           idx = mask.find_next_set(idx + 1))
        padded_reservations->insert(
            std::make_pair(idx, reservations[offset++]));
    }

    //--------------------------------------------------------------------------
    /*static*/ void PaddedReservationResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask mask;
      derez.deserialize(mask);
      std::vector<Reservation> reservations(mask.pop_count());
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        derez.deserialize(reservations[idx]);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      PhysicalManager* target = legion_safe_cast<PhysicalManager*>(dc);
      target->update_field_reservations(mask, reservations);
      Runtime::trigger_event(to_trigger);
    }

  }  // namespace Internal
}  // namespace Legion
