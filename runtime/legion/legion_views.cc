/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "region_tree.h"
#include "legion_spy.h"
#include "legion_profiling.h"
#include "legion_instances.h"
#include "legion_views.h"
#include "legion_analysis.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // LogicalView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalView::LogicalView(RegionTreeForest *ctx, DistributedID did,
                             AddressSpaceID own_addr, AddressSpace loc_space,
                             RegionTreeNode *node, bool register_now)
      : DistributedCollectable(ctx->runtime, did, own_addr, 
                               loc_space, register_now), 
        context(ctx), logical_node(node), 
        view_lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
      if (register_now)
        logical_node->register_logical_view(this); 
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
      view_lock.destroy_reservation();
      view_lock = Reservation::NO_RESERVATION;
      logical_node->unregister_logical_view(this);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::delete_logical_view(LogicalView *view)
    //--------------------------------------------------------------------------
    {
      if (view->is_instance_view())
      {
        InstanceView *inst_view = view->as_instance_view();
        if (inst_view->is_materialized_view())
          legion_delete(inst_view->as_materialized_view());
        else if (inst_view->is_reduction_view())
          legion_delete(inst_view->as_reduction_view());
        else
          assert(false);
      }
      else if (view->is_deferred_view())
      {
        DeferredView *deferred_view = view->as_deferred_view();
        if (deferred_view->is_composite_view())
          legion_delete(deferred_view->as_composite_view());
        else if (deferred_view->is_fill_view())
          legion_delete(deferred_view->as_fill_view());
        else
          assert(false);
      }
      else
        assert(false);
    }

    //--------------------------------------------------------------------------
    void LogicalView::send_remote_registration(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(destruction_event);
        const bool is_region = logical_node->is_region();
        rez.serialize<bool>(is_region);
        if (is_region)
          rez.serialize(logical_node->as_region_node()->handle);
        else
          rez.serialize(logical_node->as_partition_node()->handle);
      }
      runtime->send_view_remote_registration(owner_space, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_view_remote_registration(
           RegionTreeForest *forest, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Event destroy_event;
      derez.deserialize(destroy_event);
      bool is_region;
      derez.deserialize<bool>(is_region);
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        forest->get_node(handle)->find_view(did)->register_remote_instance(
                                                      source, destroy_event);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        forest->get_node(handle)->find_view(did)->register_remote_instance(
                                                      source, destroy_event);
      }
    }

    //--------------------------------------------------------------------------
    DistributedID LogicalView::send_view(AddressSpaceID target,
                                         const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      DistributedID result = send_view_base(target);
      send_view_updates(target, update_mask);
      return result;
    }

    //--------------------------------------------------------------------------
    void LogicalView::defer_collect_user(Event term_event) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, DEFER_COLLECT_USER_CALL);
#endif
      // The runtime will add the gc reference to this view when necessary
      runtime->defer_collect_user(this, term_event);
    }
 
    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_deferred_collect(LogicalView *view,
                                             const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      view->collect_users(term_events);
      // Then remove the gc reference on the object
      if (view->remove_base_gc_ref(PENDING_GC_REF))
        delete_logical_view(view);
    }

    /////////////////////////////////////////////////////////////
    // InstanceView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp, AddressSpaceID local_sp,
                               RegionTreeNode *node, bool register_now)
      : LogicalView(ctx, did, owner_sp, local_sp, node, register_now)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool InstanceView::is_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceView::is_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    InstanceView* InstanceView::as_instance_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<InstanceView*>(this);
    }

    //--------------------------------------------------------------------------
    DeferredView* InstanceView::as_deferred_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // MaterializedView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
                               RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID own_addr, AddressSpaceID loc_addr,
                               RegionTreeNode *node, InstanceManager *man,
                               MaterializedView *par, unsigned dep,
                               bool register_now, UniqueID context_uid)
      : InstanceView(ctx, did, own_addr, loc_addr, node, register_now), 
        manager(man), parent(par), depth(dep)
    //--------------------------------------------------------------------------
    {
      // Otherwise the instance lock will get filled in when we are unpacked
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      // If we are either not a parent or we are a remote parent add 
      // a resource reference to avoid being collected
      if (parent != NULL)
        add_nested_resource_ref(did);
      else 
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(context_uid != 0);
#endif
        manager->add_nested_resource_ref(did);
        manager->register_logical_top_view(context_uid, this);
        // Do remote registration for the top of each remote tree
        if (!is_owner())
        {
          add_base_resource_ref(REMOTE_DID_REF);
          send_remote_registration();
        }
      }
#ifdef LEGION_GC
      log_garbage.info("GC Materialized View %ld %ld", did, manager->did); 
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL, false),
        manager(NULL), parent(NULL), depth(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    {
      // Remove our resource references on our children
      // Capture their recycle events in the process
      for (std::map<ColorPoint,MaterializedView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        recycle_events.insert(it->second->get_destruction_event());
        if (it->second->remove_nested_resource_ref(did))
          legion_delete(it->second);
      }
      if (parent == NULL)
      {
        manager->unregister_logical_top_view(this);
        if (manager->remove_nested_resource_ref(did))
          delete manager;
        if (is_owner())
        {
          UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
          map_over_remote_instances(functor);
        }
      }
      if (!atomic_reservations.empty())
      {
        // If this is the owner view, delete any atomic reservations
        if (is_owner())
        {
          for (std::map<FieldID,Reservation>::iterator it = 
                atomic_reservations.begin(); it != 
                atomic_reservations.end(); it++)
          {
            it->second.destroy_reservation();
          }
        }
        atomic_reservations.clear();
      }
      if (!initial_user_events.empty())
      {
        for (std::set<Event>::const_iterator it = initial_user_events.begin();
              it != initial_user_events.end(); it++)
          filter_local_users(*it);
      }
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE) && \
      defined(DEBUG_HIGH_LEVEL)
      // Don't forget to remove the initial user if there was one
      // before running these checks
      assert(current_epoch_users.empty());
      assert(previous_epoch_users.empty());
      assert(outstanding_gc_events.empty());
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView& MaterializedView::operator=(const MaterializedView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    Memory MaterializedView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->memory;
    }

    //--------------------------------------------------------------------------
    const FieldMask& MaterializedView::get_physical_mask(void) const
    //--------------------------------------------------------------------------
    {
      return manager->layout->allocated_fields;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::is_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return true; 
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return false; 
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<MaterializedView*>(this);
    }

    //--------------------------------------------------------------------------
    ReductionView* MaterializedView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_space(const FieldMask &space_mask) const
    //--------------------------------------------------------------------------
    {
      return !(space_mask - manager->layout->allocated_fields);
    }

    //--------------------------------------------------------------------------
    LogicalView* MaterializedView::get_subview(const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
      return get_materialized_subview(c);
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_subview(
                                                           const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, GET_SUBVIEW_CALL);
#endif
      // This is the common case
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<ColorPoint,MaterializedView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      RegionTreeNode *child_node = logical_node->get_tree_child(c);
      MaterializedView *child_view = legion_new<MaterializedView>(context,
                                            did, owner_space, local_space,
                                            child_node, manager, this, 
                                            depth, false/*don't register*/);
      // Retake the lock and try and add it, see if
      // someone else added the child in the meantime
      bool free_child_view = false;
      MaterializedView *result = child_view;
      {
        AutoLock v_lock(view_lock);
        std::map<ColorPoint,MaterializedView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
        {
          // Guaranteed to succeed
          if (child_view->remove_nested_resource_ref(did))
            free_child_view = true;
          // Change the result
          result = finder->second;
        }
        else
        {
          children[c] = child_view;
          child_node->register_logical_view(child_view);
        }
      }
      if (free_child_view)
        legion_delete(child_view);
      return result;
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_field(FieldID fid,
                              std::vector<Domain::CopySrcDstField> &copy_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_FIELD_CALL);
#endif
      std::vector<FieldID> local_fields(1,fid);
      manager->compute_copy_offsets(local_fields, copy_fields); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_to(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_TO_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, dst_fields);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_from(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, COPY_FROM_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::reduce_to(ReductionOpID redop, 
                                     const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_TO_CALL);
#endif
      manager->compute_copy_offsets(copy_mask, dst_fields);
      return false; // not a fold
    }

    //--------------------------------------------------------------------------
    void MaterializedView::reduce_from(ReductionOpID redop,
                                       const FieldMask &reduce_mask,
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_FROM_CALL);
#endif
      manager->compute_copy_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_war_dependence(const RegionUsage &usage,
                                              const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, HAS_WAR_DEPENDENCE_CALL);
#endif
      // No WAR dependences for read-only or reduce 
      if (IS_READ_ONLY(usage) || IS_REDUCE(usage))
        return false;
      const ColorPoint &local_color = logical_node->get_color();
      if (has_local_war_dependence(usage, user_mask, ColorPoint(), local_color))
        return true;
      if (parent != NULL)
        return parent->has_war_dependence_above(usage, user_mask, local_color);
      return false;
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::accumulate_events(std::set<Event> &all_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ACCUMULATE_EVENTS_CALL);
#endif
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      all_events.insert(outstanding_gc_events.begin(),
                        outstanding_gc_events.end());
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user(ReductionOpID redop, Event copy_term,
                                         const VersionInfo &version_info,
                                     const FieldMask &copy_mask, bool reading)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_COPY_USER_CALL);
#endif
      // Quick test, we only need to do this if the copy_term event
      // exists, otherwise the user is already done
      if (copy_term.exists())
      {
        RegionUsage usage;
        usage.redop = redop;
        usage.prop = EXCLUSIVE;
        if (reading)
          usage.privilege = READ_ONLY;
        else if (redop > 0)
          usage.privilege = REDUCE;
        else
          usage.privilege = READ_WRITE;
        if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
        {
          const ColorPoint &local_color = logical_node->get_color();
          parent->add_copy_user_above(usage, copy_term, local_color,
                                      version_info, copy_mask);
        }
        add_local_copy_user(usage, copy_term, true/*base*/, ColorPoint(),
                            version_info, copy_mask);
      }
    }

    //--------------------------------------------------------------------------
    Event MaterializedView::add_user(const RegionUsage &usage, Event term_event,
                                     const FieldMask &user_mask, Operation *op,
                                     const VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_CALL);
#endif
      std::set<Event> wait_on_events;
      Event start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color,
                               version_info, user_mask, wait_on_events);
      }
      const bool issue_collect = add_local_user(usage, term_event, true/*base*/,
                         ColorPoint(), version_info, user_mask, wait_on_events);
      // Launch the garbage collection task, if it doesn't exist
      // then the user wasn't registered anyway, see add_local_user
      if (issue_collect)
        defer_collect_user(term_event);
      // At this point tasks shouldn't be allowed to wait on themselves
#ifdef DEBUG_HIGH_LEVEL
      if (term_event.exists())
        assert(wait_on_events.find(term_event) == wait_on_events.end());
#endif
      if (IS_ATOMIC(usage))
        find_atomic_reservations(user_mask, op, IS_WRITE(usage));
      // Return the merge of the events
      return Runtime::merge_events<false>(wait_on_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_initial_user(Event term_event,
                                            const RegionUsage &usage,
                                            const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // No need to take the lock since we are just initializing
      PhysicalUser *user = legion_new<PhysicalUser>(usage, ColorPoint());
      user->add_reference();
      add_current_user(user, term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::notify_active(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
        manager->add_nested_gc_ref(did);
      else
        parent->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      // No need to worry about handling the deletion case since
      // we know we also hold a resource reference and therefore
      // the manager won't be deleted until we are deleted at
      // the earliest
      if (parent == NULL)
        manager->remove_nested_gc_ref(did);
      else if (parent->remove_nested_gc_ref(did))
        delete parent;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      // If we are at the top of the tree add a valid reference
      // Otherwise add our valid reference on our parent
      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, true/*add*/);
        manager->add_nested_valid_ref(did);
      }
      else
        parent->add_nested_valid_ref(did);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      // If we are at the top of the tree add a valid reference
      // Otherwise add our valid reference on the parent
      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, false/*add*/);
        manager->remove_nested_valid_ref(did);
      }
      else if (parent->remove_nested_valid_ref(did))
        legion_delete(parent);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::collect_users(const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock v_lock(view_lock);
        // Remove any event users from the current and previous users
        for (std::set<Event>::const_iterator it = term_events.begin();
              it != term_events.end(); it++)
        {
          filter_local_users(*it); 
        }
      }
      if (parent != NULL)
        parent->collect_users(term_events);
    } 

    //--------------------------------------------------------------------------
    DistributedID MaterializedView::send_view_base(AddressSpaceID target) 
    //--------------------------------------------------------------------------
    {
      // See if we already have it
      if (!has_remote_instance(target))
      {
        if (parent == NULL)
        {
          // If we are the parent we have to do the send
          // Send the physical manager first
          DistributedID manager_did = manager->send_manager(target);
#ifdef DEBUG_HIGH_LEVEL
          assert(logical_node->is_region()); // Always regions at the top
#endif
          // Don't take the lock, it's alright to have duplicate sends
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(manager_did);
            rez.serialize(logical_node->as_region_node()->handle);
            rez.serialize(owner_space);
            rez.serialize(depth);
            // Find the context UID since we don't store it
            // Must be the root view (parent == NULL) for this to work
            UniqueID context_uid = manager->find_context_uid(this);
            rez.serialize(context_uid);
          }
          runtime->send_materialized_view(target, rez);
        }
        else // Ask our parent to do the send
          parent->send_view_base(target);
        // We've now done the send so record it 
        update_remote_instances(target);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::send_view_updates(AddressSpaceID target,
                                             const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      std::map<PhysicalUser*,int/*index*/> needed_users;  
      Serializer current_rez, previous_rez;
      unsigned current_events = 0, previous_events = 0;
      // Take the lock in read-only mode
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        for (LegionMap<Event,EventUsers>::aligned::const_iterator cit = 
              current_epoch_users.begin(); cit != 
              current_epoch_users.end(); cit++)
        {
          FieldMask overlap = cit->second.user_mask & update_mask;
          if (!overlap)
            continue;
          current_events++;
          current_rez.serialize(cit->first);
          const EventUsers &event_users = cit->second;
          if (event_users.single)
          {
            int index = needed_users.size();
            needed_users[event_users.users.single_user] = index;
            event_users.users.single_user->add_reference();
            current_rez.serialize(index);
            current_rez.serialize(overlap);
          }
          else
          {
            Serializer event_rez;
            int count = -1; // start this at negative one
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = event_users.users.multi_users->begin(); it != 
                  event_users.users.multi_users->end(); it++)
            {
              FieldMask overlap2 = it->second & overlap;
              if (!overlap2)
                continue;
              count--; // Make the count negative to disambiguate
              int index = needed_users.size();
              needed_users[it->first] = index;
              it->first->add_reference();
              event_rez.serialize(index);
              event_rez.serialize(overlap2);
            }
            // If there was only one, we can take the normal path
            if ((count == -1) || (count < -2))
              current_rez.serialize(count);
            size_t event_rez_size = event_rez.get_used_bytes();
            current_rez.serialize(event_rez.get_buffer(), event_rez_size);
          }
        }
        for (LegionMap<Event,EventUsers>::aligned::const_iterator pit = 
              previous_epoch_users.begin(); pit != 
              previous_epoch_users.end(); pit++)
        {
          FieldMask overlap = pit->second.user_mask & update_mask;
          if (!overlap)
            continue;
          previous_events++;
          previous_rez.serialize(pit->first);
          const EventUsers &event_users = pit->second;
          if (event_users.single)
          {
            std::map<PhysicalUser*,int>::const_iterator finder = 
              needed_users.find(event_users.users.single_user);
            if (finder == needed_users.end())
            {
              int index = needed_users.size();
              previous_rez.serialize(index);
              needed_users[event_users.users.single_user] = index;
              event_users.users.single_user->add_reference();
            }
            else
              previous_rez.serialize(finder->second);
            previous_rez.serialize(overlap);
          }
          else 
          {
            Serializer event_rez;
            int count = -1; // start this at negative one
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              FieldMask overlap2 = it->second & overlap;
              if (!overlap2)
                continue;
              count--; // Make the count negative to disambiguate
              std::map<PhysicalUser*,int>::const_iterator finder = 
                needed_users.find(it->first);
              if (finder == needed_users.end())
              {
                int index = needed_users.size();
                needed_users[it->first] = index;
                event_rez.serialize(index);
                it->first->add_reference();
              }
              else
                event_rez.serialize(finder->second);
              event_rez.serialize(overlap2);
            }
            // If there was only one user, we can take the normal path
            if ((count == -1) || (count < -2))
              previous_rez.serialize(count);
            size_t event_rez_size = event_rez.get_used_bytes();
            previous_rez.serialize(event_rez.get_buffer(), event_rez_size); 
          }
        }
      }
      // Now build our buffer and send the result
      Serializer rez;
      {
        RezCheck z(rez);
        bool is_region = logical_node->is_region();
        rez.serialize(is_region);
        if (is_region)
          rez.serialize(logical_node->as_region_node()->handle);
        else
          rez.serialize(logical_node->as_partition_node()->handle);
        rez.serialize(did);
        // Pack the needed users first
        rez.serialize<size_t>(needed_users.size());
        for (std::map<PhysicalUser*,int>::const_iterator it = 
              needed_users.begin(); it != needed_users.end(); it++)
        {
          rez.serialize(it->second);
          it->first->pack_user(rez);
          if (it->first->remove_reference())
            legion_delete(it->first);
        }
        // Then pack the current and previous events
        rez.serialize(current_events);
        size_t current_size = current_rez.get_used_bytes();
        rez.serialize(current_rez.get_buffer(), current_size);
        rez.serialize(previous_events);
        size_t previous_size = previous_rez.get_used_bytes();
        rez.serialize(previous_rez.get_buffer(), previous_size);
      }
      runtime->send_materialized_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_update(Deserializer &derez,
                                          AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t num_users;
      derez.deserialize(num_users);
      std::vector<PhysicalUser*> users(num_users);
      FieldSpaceNode *field_node = logical_node->column_source;
      for (unsigned idx = 0; idx < num_users; idx++)
      {
        int index;
        derez.deserialize(index);
        users[index] = PhysicalUser::unpack_user(derez, field_node, 
                                                 source, true/*add ref*/); 
      }
      // We've already added a reference for all users since we'll know
      // that we'll be adding them at least once
      std::vector<bool> need_reference(num_users, false);
      std::deque<Event> collect_events;
      {
        // Hold the lock when updating the view
        AutoLock v_lock(view_lock); 
        unsigned num_current;
        derez.deserialize(num_current);
        for (unsigned idx = 0; idx < num_current; idx++)
        {
          Event current_event;
          derez.deserialize(current_event);
          int index;
          derez.deserialize(index);
          if (index < 0)
          {
            int count = (-index) - 1;
            for (int i = 0; i < count; i++)
            {
              derez.deserialize(index);
#ifdef DEBUG_HIGH_LEVEL
              assert(unsigned(index) < num_users);
#endif
              FieldMask user_mask;
              derez.deserialize(user_mask);
              field_node->transform_field_mask(user_mask, source);
              if (need_reference[index])
                users[index]->add_reference();
              else
                need_reference[index] = true;
              add_current_user(users[index], current_event, user_mask);
            }
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(unsigned(index) < num_users);
#endif
            // Just one user
            FieldMask user_mask;
            derez.deserialize(user_mask);
            field_node->transform_field_mask(user_mask, source);
            if (need_reference[index])
              users[index]->add_reference();
            else
              need_reference[index] = true;
            add_current_user(users[index], current_event, user_mask);
          }
          if (outstanding_gc_events.find(current_event) ==
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(current_event);
            collect_events.push_back(current_event);
          }
        }
        unsigned num_previous;
        derez.deserialize(num_previous);
        for (unsigned idx = 0; idx < num_previous; idx++)
        {
          Event previous_event;
          derez.deserialize(previous_event);
          int index;
          derez.deserialize(index);
          if (index < 0)
          {
            int count = (-index) - 1;
            for (int i = 0; i < count; i++)
            {
              derez.deserialize(index);
#ifdef DEBUG_HIGH_LEVEL
              assert(unsigned(index) < num_users);
#endif
              FieldMask user_mask;
              derez.deserialize(user_mask);
              field_node->transform_field_mask(user_mask, source);
              if (need_reference[index])
                users[index]->add_reference();
              else
                need_reference[index] = true;
              add_previous_user(users[index], previous_event, user_mask);
            }
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(unsigned(index) < num_users);
#endif
            // Just one user
            FieldMask user_mask;
            derez.deserialize(user_mask);
            field_node->transform_field_mask(user_mask, source);
            if (need_reference[index])
              users[index]->add_reference();
            else
              need_reference[index] = true;
            add_previous_user(users[index], previous_event, user_mask);
          }
          if (outstanding_gc_events.find(previous_event) ==
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(previous_event);
            collect_events.push_back(previous_event);
          }
        }
      }
      if (!collect_events.empty())
      {
        if (parent != NULL)
          parent->update_gc_events(collect_events);
        for (std::deque<Event>::const_iterator it = 
              collect_events.begin(); it != collect_events.end(); it++)
        {
          defer_collect_user(*it); 
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      for (unsigned idx = 0; idx < need_reference.size(); idx++)
        assert(need_reference[idx]);
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::update_gc_events(const std::deque<Event> &gc_events)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->update_gc_events(gc_events);
      AutoLock v_lock(view_lock);
      for (std::deque<Event>::const_iterator it = gc_events.begin();
            it != gc_events.end(); it++)
      {
        outstanding_gc_events.insert(*it);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above(const RegionUsage &usage, 
                                          Event term_event,
                                          const ColorPoint &child_color,
                                          const VersionInfo &version_info,
                                          const FieldMask &user_mask,
                                          std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_ABOVE_CALL);
#endif
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color,
                               version_info, user_mask, preconditions);
      }
      add_local_user(usage, term_event, false/*base*/, child_color,
                     version_info, user_mask, preconditions);
      // No need to launch a collect user task, the child takes care of that
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user_above(const RegionUsage &usage, 
                                               Event copy_term, 
                                               const ColorPoint &child_color,
                                               const VersionInfo &version_info,
                                               const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color,
                                    version_info, copy_mask);
      }
      add_local_copy_user(usage, copy_term, false/*base*/, child_color, 
                          version_info, copy_mask);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_local_copy_user(const RegionUsage &usage, 
                                               Event copy_term, bool base_user,
                                               const ColorPoint &child_color,
                                               const VersionInfo &version_info,
                                               const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
      PhysicalUser *user;
      // We currently only use the version information for avoiding
      // WAR dependences on the same version number, so we don't need
      // it if we aren't only reading
      if (IS_READ_ONLY(usage))
        user = legion_new<PhysicalUser>(usage, child_color,
                                        version_info.get_versions(logical_node));
      else
        user = legion_new<PhysicalUser>(usage, child_color);
      user->add_reference();
      bool issue_collect = false;
      {
        AutoLock v_lock(view_lock);
        add_current_user(user, copy_term, copy_mask); 
        if (base_user)
          issue_collect = (outstanding_gc_events.find(copy_term) ==
                            outstanding_gc_events.end());
        outstanding_gc_events.insert(copy_term);
      }
      if (issue_collect)
        defer_collect_user(copy_term);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_local_user(const RegionUsage &usage,
                                          Event term_event, bool base_user,
                                          const ColorPoint &child_color,
                                          const VersionInfo &version_info,
                                          const FieldMask &user_mask,
                                          std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_LOCAL_USER_CALL);
#endif
      std::set<Event> dead_events;
      LegionMap<Event,FieldMask>::aligned filter_previous;
      FieldMask dominated;
      // Hold the lock in read-only mode when doing this part of the analysis
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        for (LegionMap<Event,EventUsers>::aligned::const_iterator cit = 
              current_epoch_users.begin(); cit != 
              current_epoch_users.end(); cit++)
        {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
          // We're about to do a bunch of expensive tests, 
          // so first do something cheap to see if we can 
          // skip all the tests.
          if (cit->first.has_triggered())
          {
            dead_events.insert(cit->first);
            continue;
          }
#endif
          // No need to check for dependences on ourselves
          if (cit->first == term_event)
            continue;
          // If we arleady recorded a dependence, then we are done
          if (preconditions.find(cit->first) != preconditions.end())
            continue;
          const EventUsers &event_users = cit->second;
          if (event_users.single)
          {
            find_current_preconditions(cit->first, 
                                       event_users.users.single_user,
                                       event_users.user_mask,
                                       usage, user_mask, child_color,
                                       preconditions, observed, non_dominated);
          }
          else
          {
            // Otherwise do a quick test for non-interference on the
            // summary mask and iterate the users if needed
            if (!(user_mask * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                // Unlike with the copy analysis, once we record a dependence
                // on an event, we are done, so we can keep going
                if (find_current_preconditions(cit->first,
                                               it->first, it->second,
                                               usage, user_mask, child_color,
                                               preconditions, observed, 
                                               non_dominated))
                  break;
              }
            }
          }
        }
        // See if we have any fields for which we need to do an analysis
        // on the previous fields
        // It's only safe to dominate fields that we observed
        dominated = (observed & (user_mask - non_dominated));
        // Update the non-dominated mask with what we
        // we're actually not-dominated by
        non_dominated = user_mask - dominated;
        const bool skip_analysis = !non_dominated;
        for (LegionMap<Event,EventUsers>::aligned::const_iterator pit = 
              previous_epoch_users.begin(); pit != 
              previous_epoch_users.end(); pit++)
        {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
          // We're about to do a bunch of expensive tests, 
          // so first do something cheap to see if we can 
          // skip all the tests.
          if (pit->first.has_triggered())
          {
            dead_events.insert(pit->first);
            continue;
          }
#endif
          // No need to check for dependences on ourselves
          if (pit->first == term_event)
            continue;
          // If we arleady recorded a dependence, then we are done
          if (preconditions.find(pit->first) != preconditions.end())
            continue;
          const EventUsers &event_users = pit->second;
          if (!!dominated)
          {
            FieldMask dom_overlap = event_users.user_mask & dominated;
            if (!!dom_overlap)
              filter_previous[pit->first] = dom_overlap;
          }
          // If we don't have any non-dominated fields we can skip the
          // rest of the analysis because we dominated everything
          if (skip_analysis)
            continue;
          if (event_users.single)
          {
            find_previous_preconditions(pit->first,
                                        event_users.users.single_user,
                                        event_users.user_mask,
                                        usage, non_dominated,
                                        child_color, preconditions);
          }
          else
          {
            if (!(non_dominated * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                // Once we find a dependence we are can skip the rest
                if (find_previous_preconditions(pit->first,
                                                it->first, it->second,
                                                usage, non_dominated,
                                                child_color, preconditions))
                  break;
              }
            }
          }
        }
      }
      PhysicalUser *new_user = NULL;
      if (term_event.exists())
      {
        // Only need to record version info if we are read-only
        // because we only use the version info for avoiding WAR dependences
        if (IS_READ_ONLY(usage))
          new_user = legion_new<PhysicalUser>(usage, child_color,
                                      version_info.get_versions(logical_node));
        else
          new_user = legion_new<PhysicalUser>(usage, child_color);
        new_user->add_reference();
      }
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      if (!dead_events.empty())
      {
        for (std::set<Event>::const_iterator it = dead_events.begin();
              it != dead_events.end(); it++)
        {
          filter_local_users(*it);
        }
      }
      if (!filter_previous.empty())
        filter_previous_users(filter_previous);
      if (!!dominated)
        filter_current_users(dominated);
      // Finally add our user and return if we need to issue a GC meta-task
      if (term_event.exists())
      {
        add_current_user(new_user, term_event, user_mask);
        if (outstanding_gc_events.find(term_event) == 
            outstanding_gc_events.end())
        {
          outstanding_gc_events.insert(term_event);
          return base_user;
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::find_current_preconditions(Event test_event,
                                                 const PhysicalUser *prev_user,
                                                 const FieldMask &prev_mask,
                                                 const RegionUsage &next_user,
                                                 const FieldMask &next_mask,
                                                 const ColorPoint &child_color,
                                                 std::set<Event> &preconditions,
                                                 FieldMask &observed,
                                                 FieldMask &non_dominated)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = prev_mask & next_mask;
      if (!overlap)
        return false;
      else
        observed |= overlap;
      if (child_color.is_valid())
      {
        // Same child, already done the analysis
        if (child_color == prev_user->child)
        {
          non_dominated |= overlap;
          return false;
        }
        // Disjoint children, keep going
        if (prev_user->child.is_valid() &&
            logical_node->are_children_disjoint(child_color,
                                                prev_user->child))
        {
          non_dominated |= overlap;
          return false;
        }
      }
      // Now do a dependence analysis
      DependenceType dt = check_dependence_type(prev_user->usage, next_user);
      switch (dt)
      {
        case NO_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            // No actual dependence
            non_dominated |= overlap;
            return false;
          }
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
          {
            // Actual dependence
            preconditions.insert(test_event);
            break;
          }
        default:
          assert(false); // should never get here
      }
      // If we made it to the end we recorded a dependence so return true
      return true;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::find_previous_preconditions(Event test_event,
                                                 const PhysicalUser *prev_user,
                                                 const FieldMask &prev_mask,
                                                 const RegionUsage &next_user,
                                                 const FieldMask &next_mask,
                                                 const ColorPoint &child_color,
                                                 std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      if (child_color.is_valid())
      {
        // Same child: did analysis below
        if (child_color == prev_user->child)
          return false;
        if (prev_user->child.is_valid() &&
            logical_node->are_children_disjoint(child_color,
                                          prev_user->child))
          return false;
      }
      FieldMask overlap = prev_mask & next_mask;
      if (!overlap)
        return false;
      // Now do a dependence analysis
      DependenceType dt = check_dependence_type(prev_user->usage, next_user);
      switch (dt)
      {
        case NO_DEPENDENCE:
        case ATOMIC_DEPENDENCE:
        case SIMULTANEOUS_DEPENDENCE:
          {
            // No actual dependence
            return false;
          }
        case TRUE_DEPENDENCE:
        case ANTI_DEPENDENCE:
          {
            // Actual dependence
            preconditions.insert(test_event);
            break;
          }
        default:
          assert(false); // should never get here
      }
      // If we make it here, we recorded a dependence
      return true;
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions(ReductionOpID redop, 
                                                   bool reading, 
                                                   const FieldMask &copy_mask,
                                                const VersionInfo &version_info,
                             LegionMap<Event,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_CALL);
#endif
      Event start_use_event = manager->get_use_event();
      if (start_use_event.exists())
      {
        LegionMap<Event,FieldMask>::aligned::iterator finder = 
          preconditions.find(start_use_event);
        if (finder == preconditions.end())
          preconditions[start_use_event] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, copy_mask,
                                      local_point, version_info, preconditions);
      }
      find_local_copy_preconditions(redop, reading, copy_mask, 
                                    ColorPoint(), version_info, preconditions);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                                  const ColorPoint &child_color,
                                                const VersionInfo &version_info,
                             LegionMap<Event,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_COPY_PRECONDITIONS_ABOVE_CALL);
#endif
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, copy_mask,
                                      local_point, version_info, preconditions);
      }
      find_local_copy_preconditions(redop, reading, copy_mask, 
                                    child_color, version_info, preconditions);
    }
    
    //--------------------------------------------------------------------------
    void MaterializedView::find_local_copy_preconditions(ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                                  const ColorPoint &child_color,
                                                const VersionInfo &version_info,
                             LegionMap<Event,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#endif
      // First get our set of version data in case we need it, it's really
      // only safe to do this if we are at the bottom of our set of versions
      const FieldVersions *versions = 
        child_color.is_valid() ? NULL : version_info.get_versions(logical_node);
      std::set<Event> dead_events;
      LegionMap<Event,FieldMask>::aligned filter_previous;
      FieldMask dominated;
      {
        // Hold the lock in read-only mode when doing this analysis
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        for (LegionMap<Event,EventUsers>::aligned::const_iterator cit = 
              current_epoch_users.begin(); cit != 
              current_epoch_users.end(); cit++)
        {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
          // We're about to do a bunch of expensive tests, 
          // so first do something cheap to see if we can 
          // skip all the tests.
          if (cit->first.has_triggered())
          {
            dead_events.insert(cit->first);
            continue;
          }
#endif
          const EventUsers &event_users = cit->second;
          if (event_users.single)
          {
            find_current_copy_preconditions(cit->first,
                                            event_users.users.single_user,
                                            event_users.user_mask,
                                            redop, reading, copy_mask,
                                            child_color, versions,
                                            preconditions, observed, 
                                            non_dominated);
          }
          else
          {
            // Otherwise do a quick test for non-interference on the
            // summary mask and iterate the users if needed
            if (!(copy_mask * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                // You might think after we record one event dependence that
                // would be enough to skip the other users for the same event,
                // but we actually do need precise field information for each
                // event to properly issue dependent copies
                find_current_copy_preconditions(cit->first,it->first,it->second,
                                                redop, reading, copy_mask,
                                                child_color, versions,
                                                preconditions, observed,
                                                non_dominated);
              }
            }
          }
        }
        // See if we have any fields for which we need to do an analysis
        // on the previous fields
        // It's only safe to dominate fields that we observed
        dominated = (observed & (copy_mask - non_dominated));
        // Update the non-dominated mask with what we
        // we're actually not-dominated by
        non_dominated = copy_mask - dominated;
        const bool skip_analysis = !non_dominated;
        for (LegionMap<Event,EventUsers>::aligned::const_iterator pit = 
              previous_epoch_users.begin(); pit != 
              previous_epoch_users.end(); pit++)
        {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
          // We're about to do a bunch of expensive tests, 
          // so first do something cheap to see if we can 
          // skip all the tests.
          if (pit->first.has_triggered())
          {
            dead_events.insert(pit->first);
            continue;
          }
#endif
          const EventUsers &event_users = pit->second;
          if (!!dominated)
          {
            FieldMask dom_overlap = event_users.user_mask & dominated;
            if (!!dom_overlap)
              filter_previous[pit->first] = dom_overlap;
          }
          // If we don't have any non-dominated fields we can skip the
          // rest of the analysis because we dominated everything
          if (skip_analysis)
            continue;
          if (event_users.single)
          {
            find_previous_copy_preconditions(pit->first,
                                             event_users.users.single_user,
                                             event_users.user_mask,
                                             redop, reading, non_dominated,
                                             child_color, versions,
                                             preconditions);
          }
          else
          {
            if (!(non_dominated * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                find_previous_copy_preconditions(pit->first,
                                                 it->first, it->second,
                                                 redop, reading, 
                                                 non_dominated, child_color,
                                                 versions, preconditions);
              }
            }
          }
        }
      }
      // Release the lock, if we have any modifications to make, then
      // retake the lock in exclusive mode
      if (!dead_events.empty() || !filter_previous.empty() || !!dominated)
      {
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
        {
          for (std::set<Event>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
          {
            filter_local_users(*it);
          }
        }
        if (!filter_previous.empty())
          filter_previous_users(filter_previous);
        if (!!dominated)
          filter_current_users(dominated);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_current_copy_preconditions(Event test_event,
                                              const PhysicalUser *user,
                                              const FieldMask &user_mask,
                                              ReductionOpID redop, bool reading,
                                              const FieldMask &copy_mask,
                                              const ColorPoint &child_color,
                                              const FieldVersions *versions,
                             LegionMap<Event,FieldMask>::aligned &preconditions,
                                              FieldMask &observed,
                                              FieldMask &non_dominated)
    //--------------------------------------------------------------------------
    {
      FieldMask overlap = copy_mask & user_mask;
      if (!overlap)
        return;
      else
        observed |= overlap;
      if (child_color.is_valid())
      {
        // Same child, already done the analysis
        if (child_color == user->child)
        {
          non_dominated |= overlap;
          return;
        }
        // Disjoint children, keep going
        if (user->child.is_valid() &&
            logical_node->are_children_disjoint(child_color,
                                                user->child))
        {
          non_dominated |= overlap;
          return;
        }
      }
      // Now do a dependence analysis
      if (reading && IS_READ_ONLY(user->usage))
      {
        non_dominated |= overlap;
        return;
      }
      if ((redop > 0) && (user->usage.redop == redop))
      {
        non_dominated |= overlap;
        return;
      }
      // Check for WAR and WAW dependences, if we have one we
      // can see if we are writing the same version number
      // in which case there is no need for a dependence, thank
      // you wonchan and mini-aero for raising this case
      if (!reading && (redop == 0) && (versions != NULL) &&
          !IS_REDUCE(user->usage) && user->same_versions(overlap, versions))
      {
        non_dominated |= overlap;
        return;
      }
      // If we make it here, then we have a dependence, so record it 
      LegionMap<Event,FieldMask>::aligned::iterator finder = 
        preconditions.find(test_event);
      if (finder == preconditions.end())
        preconditions[test_event] = overlap;
      else
        finder->second |= overlap;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_copy_preconditions(Event test_event,
                                              const PhysicalUser *user,
                                              const FieldMask &user_mask,
                                              ReductionOpID redop, bool reading,
                                              const FieldMask &copy_mask,
                                              const ColorPoint &child_color,
                                              const FieldVersions *versions,
                        LegionMap<Event,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    { 
      if (child_color.is_valid())
      {
        // Same child: did analysis below
        if (child_color == user->child)
          return;
        if (user->child.is_valid() &&
            logical_node->are_children_disjoint(child_color,
                                                user->child))
          return;
      }
      FieldMask overlap = user_mask & copy_mask;
      if (!overlap)
        return;
      if (reading && IS_READ_ONLY(user->usage))
        return;
      if ((redop > 0) && (user->usage.redop == redop))
        return;
      if (!reading && (redop == 0) && (versions != NULL) &&
          !IS_REDUCE(user->usage) && user->same_versions(overlap, versions))
        return;
      // Otherwise record the dependence
      LegionMap<Event,FieldMask>::aligned::iterator finder = 
        preconditions.find(test_event);
      if (finder == preconditions.end())
        preconditions[test_event] = overlap;
      else
        finder->second |= overlap;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_previous_users(
                     const LegionMap<Event,FieldMask>::aligned &filter_previous)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<Event,FieldMask>::aligned::const_iterator fit = 
            filter_previous.begin(); fit != filter_previous.end(); fit++)
      {
        LegionMap<Event,EventUsers>::aligned::iterator finder = 
          previous_epoch_users.find(fit->first);
        // Someone might have already removed it
        if (finder == previous_epoch_users.end())
          continue;
        finder->second.user_mask -= fit->second;
        if (!finder->second.user_mask)
        {
          // We can delete the whole entry
          if (finder->second.single)
          {
            PhysicalUser *user = finder->second.users.single_user;
            if (user->remove_reference())
              legion_delete(user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = finder->second.users.multi_users->begin(); it !=
                  finder->second.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                legion_delete(it->first);
            }
            // Delete the map too
            delete finder->second.users.multi_users;
          }
          previous_epoch_users.erase(finder);
        }
        else if (!finder->second.single) // only need to filter for non-single
        {
          // Filter out the users for the dominated fields
          std::vector<PhysicalUser*> to_delete;
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                finder->second.users.multi_users->begin(); it !=
                finder->second.users.multi_users->end(); it++)
          {
            it->second -= fit->second; 
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<PhysicalUser*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              finder->second.users.multi_users->erase(*it);
              if ((*it)->remove_reference())
                legion_delete(*it);
            }
            // See if we can shrink this back down
            if (finder->second.users.multi_users->size() == 1)
            {
              LegionMap<PhysicalUser*,FieldMask>::aligned::iterator first_it =
                            finder->second.users.multi_users->begin();     
#ifdef DEBUG_HIGH_LEVEL
              // This summary mask should dominate
              assert(!(first_it->second - finder->second.user_mask));
#endif
              PhysicalUser *user = first_it->first;
              finder->second.user_mask = first_it->second;
              delete finder->second.users.multi_users;
              finder->second.users.single_user = user;
              finder->second.single = true;
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_current_users(const FieldMask &dominated)
    //--------------------------------------------------------------------------
    {
      std::vector<Event> events_to_delete;
      for (LegionMap<Event,EventUsers>::aligned::iterator cit = 
            current_epoch_users.begin(); cit !=
            current_epoch_users.end(); cit++)
      {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
        if (cit->first.has_triggered())
        {
          EventUsers &current_users = cit->second;
          if (current_users.single)
          {
            if (current_users.users.single_user->remove_reference())
              legion_delete(current_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                  current_users.users.multi_users->begin(); it !=
                  current_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                legion_delete(it->first);
            }
            delete current_users.users.multi_users;
          }
          events_to_delete.push_back(cit->first);
          continue;
        }
#endif
        EventUsers &current_users = cit->second;
        FieldMask summary_overlap = current_users.user_mask & dominated;
        if (!summary_overlap)
          continue;
        current_users.user_mask -= summary_overlap;
        EventUsers &prev_users = previous_epoch_users[cit->first];
        if (current_users.single)
        {
          PhysicalUser *user = current_users.users.single_user;
          if (prev_users.single)
          {
            // Single, see if something exists there yet
            if (prev_users.users.single_user == NULL)
            {
              prev_users.users.single_user = user; 
              prev_users.user_mask = summary_overlap;
              if (!current_users.user_mask) // reference flows back
                events_to_delete.push_back(cit->first); 
              else
                user->add_reference(); // add a reference
            }
            else if (prev_users.users.single_user == user)
            {
              // Same user, update the fields 
              prev_users.user_mask |= summary_overlap;
              if (!current_users.user_mask)
              {
                events_to_delete.push_back(cit->first);
                user->remove_reference(); // remove unnecessary reference
              }
            }
            else
            {
              // Go to multi mode
              LegionMap<PhysicalUser*,FieldMask>::aligned *new_map = 
                            new LegionMap<PhysicalUser*,FieldMask>::aligned();
              (*new_map)[prev_users.users.single_user] = prev_users.user_mask;
              (*new_map)[user] = summary_overlap;
              if (!current_users.user_mask) // reference flows back
                events_to_delete.push_back(cit->first); 
              else
                user->add_reference();
              prev_users.user_mask |= summary_overlap;
              prev_users.users.multi_users = new_map;
              prev_users.single = false;
            }
          }
          else
          {
            // Already multi
            prev_users.user_mask |= summary_overlap;
            // See if we can find it in the multi-set
            LegionMap<PhysicalUser*,FieldMask>::aligned::iterator finder = 
              prev_users.users.multi_users->find(user);
            if (finder == prev_users.users.multi_users->end())
            {
              // Couldn't find it
              (*prev_users.users.multi_users)[user] = summary_overlap;
              if (!current_users.user_mask) // reference flows back
                events_to_delete.push_back(cit->first); 
              else
                user->add_reference();
            }
            else
            {
              // Found it, update it 
              finder->second |= summary_overlap;
              if (!current_users.user_mask)
              {
                events_to_delete.push_back(cit->first);
                user->remove_reference(); // remove redundant reference
              }
            }
          }
        }
        else
        {
          // Many things, filter them and move them back
          if (!current_users.user_mask)
          {
            // Moving the whole set back, see what the previous looks like
            if (prev_users.single)
            {
              if (prev_users.users.single_user != NULL)
              {
                // Merge the one user into this map so we can move 
                // the whole map back
                PhysicalUser *user = prev_users.users.single_user;  
                LegionMap<PhysicalUser*,FieldMask>::aligned::iterator finder =
                  current_users.users.multi_users->find(user);
                if (finder == current_users.users.multi_users->end())
                {
                  // Add it reference is already there
                  (*current_users.users.multi_users)[user] = 
                    prev_users.user_mask;
                }
                else
                {
                  // Already there, update it and remove duplicate reference
                  finder->second |= prev_users.user_mask;
                  user->remove_reference();
                }
              }
              // Now just move the map back
              prev_users.user_mask |= summary_overlap;
              prev_users.users.multi_users = current_users.users.multi_users;
              prev_users.single = false;
            }
            else
            {
              // merge the two sets
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                    it = current_users.users.multi_users->begin();
                    it != current_users.users.multi_users->end(); it++)
              {
                // See if we can find it
                LegionMap<PhysicalUser*,FieldMask>::aligned::iterator finder = 
                  prev_users.users.multi_users->find(it->first);
                if (finder == prev_users.users.multi_users->end())
                {
                  // Didn't find it, just move it back, reference moves back
                  prev_users.users.multi_users->insert(*it);
                }
                else
                {
                  finder->second |= it->second; 
                  // Remove the duplicate reference
                  it->first->remove_reference();
                }
              }
              prev_users.user_mask |= summary_overlap;
              // Now delete the set
              delete current_users.users.multi_users;
            }
            events_to_delete.push_back(cit->first);
          }
          else
          {
            // Only send back filtered users
            std::vector<PhysicalUser*> to_delete;
            if (prev_users.single)
            {
              // Make a new map to send back  
              LegionMap<PhysicalUser*,FieldMask>::aligned *new_map = 
                            new LegionMap<PhysicalUser*,FieldMask>::aligned();
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                    current_users.users.multi_users->begin(); it !=
                    current_users.users.multi_users->end(); it++)
              {
                FieldMask overlap = summary_overlap & it->second;
                if (!overlap)
                  continue;
                // Can move without checking
                (*new_map)[it->first] = overlap;
                it->second -= overlap;
                if (!it->second)
                  to_delete.push_back(it->first); // reference flows back
                else
                  it->first->add_reference(); // need new reference
              }
              // Also capture the existing previous user if there is one
              if (prev_users.users.single_user != NULL)
              {
                LegionMap<PhysicalUser*,FieldMask>::aligned::iterator finder = 
                  new_map->find(prev_users.users.single_user);
                if (finder == new_map->end())
                {
                  (*new_map)[prev_users.users.single_user] = 
                    prev_users.user_mask;
                }
                else
                {
                  finder->second |= prev_users.user_mask;
                  // Remove redundant reference
                  finder->first->remove_reference();
                }
              }
              // Make the new map the previous set
              prev_users.user_mask |= summary_overlap;
              prev_users.users.multi_users = new_map;
              prev_users.single = false;
            }
            else
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it =
                    current_users.users.multi_users->begin(); it !=
                    current_users.users.multi_users->end(); it++)
              {
                FieldMask overlap = summary_overlap & it->second; 
                if (!overlap)
                  continue;
                it->second -= overlap;
                LegionMap<PhysicalUser*,FieldMask>::aligned::iterator finder = 
                  prev_users.users.multi_users->find(it->first);
                // See if it already exists
                if (finder == prev_users.users.multi_users->end())
                {
                  // Doesn't exist yet, so add it 
                  (*prev_users.users.multi_users)[it->first] = overlap;
                  if (!it->second) // reference flows back
                    to_delete.push_back(it->first);
                  else
                    it->first->add_reference();
                }
                else
                {
                  // Already exists so update it
                  finder->second |= overlap;
                  if (!it->second)
                  {
                    to_delete.push_back(it->first);
                    // Remove redundant reference
                    it->first->remove_reference();
                  }
                }
              }
              prev_users.user_mask |= summary_overlap;
            }
            // See if we can collapse this map back down
            if (!to_delete.empty())
            {
              for (std::vector<PhysicalUser*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                current_users.users.multi_users->erase(*it);
              }
              if (current_users.users.multi_users->size() == 1)
              {
                LegionMap<PhysicalUser*,FieldMask>::aligned::iterator 
                  first_it = current_users.users.multi_users->begin();
#ifdef DEBUG_HIGH_LEVEL
                // Should dominate as an upper bound
                assert(!(first_it->second - current_users.user_mask));
#endif
                PhysicalUser *user = first_it->first;
                current_users.user_mask = first_it->second;
                delete current_users.users.multi_users;
                current_users.users.single_user = user;   
                current_users.single = true;
              }
            }
          }
        }
      }
      // Delete any events
      if (!events_to_delete.empty())
      {
        for (std::vector<Event>::const_iterator it = events_to_delete.begin();
              it != events_to_delete.end(); it++)
        {
          current_epoch_users.erase(*it); 
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_current_user(PhysicalUser *user, 
                                            Event term_event,
                                            const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Reference should already have been added
      EventUsers &event_users = current_epoch_users[term_event];
      if (event_users.single)
      {
        if (event_users.users.single_user == NULL)
        {
          // make it the entry
          event_users.users.single_user = user;
          event_users.user_mask = user_mask;
        }
        else
        {
          // convert to multi
          LegionMap<PhysicalUser*,FieldMask>::aligned *new_map = 
                           new LegionMap<PhysicalUser*,FieldMask>::aligned();
          (*new_map)[event_users.users.single_user] = event_users.user_mask;
          (*new_map)[user] = user_mask;
          event_users.user_mask |= user_mask;
          event_users.users.multi_users = new_map;
          event_users.single = false;
        }
      }
      else
      {
        // Add it to the set 
        (*event_users.users.multi_users)[user] = user_mask;
        event_users.user_mask |= user_mask;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_previous_user(PhysicalUser *user, 
                                             Event term_event,
                                             const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Reference should already have been added
      EventUsers &event_users = previous_epoch_users[term_event];
      if (event_users.single)
      {
        if (event_users.users.single_user == NULL)
        {
          // make it the entry
          event_users.users.single_user = user;
          event_users.user_mask = user_mask;
        }
        else
        {
          // convert to multi
          LegionMap<PhysicalUser*,FieldMask>::aligned *new_map = 
                           new LegionMap<PhysicalUser*,FieldMask>::aligned();
          (*new_map)[event_users.users.single_user] = event_users.user_mask;
          (*new_map)[user] = user_mask;
          event_users.user_mask |= user_mask;
          event_users.users.multi_users = new_map;
          event_users.single = false;
        }
      }
      else
      {
        // Add it to the set 
        (*event_users.users.multi_users)[user] = user_mask;
        event_users.user_mask |= user_mask;
      }
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_war_dependence_above(const RegionUsage &usage,
                                                    const FieldMask &user_mask,
                                                  const ColorPoint &child_color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, HAS_WAR_DEPENDENCE_ABOVE_CALL);
#endif
      const ColorPoint &local_color = logical_node->get_color();
      if (has_local_war_dependence(usage, user_mask, child_color, local_color))
        return true;
      if (parent != NULL)
        return parent->has_war_dependence_above(usage, user_mask, local_color);
      return false;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_local_war_dependence(const RegionUsage &usage,
                                                    const FieldMask &user_mask,
                                                  const ColorPoint &child_color,
                                                  const ColorPoint &local_color)
    //--------------------------------------------------------------------------
    {
      // Do the local analysis
      FieldMask observed;
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      for (LegionMap<Event,EventUsers>::aligned::const_iterator cit = 
            current_epoch_users.begin(); cit != 
            current_epoch_users.end(); cit++)
      {
        const EventUsers &event_users = cit->second;
        FieldMask overlap = user_mask & event_users.user_mask;
        if (!overlap)
          continue;
        else
          observed |= overlap;
        if (event_users.single)
        {
          if (IS_READ_ONLY(event_users.users.single_user->usage))
            return true;
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator it =
                event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            FieldMask overlap2 = user_mask & it->second;
            if (!overlap2)
              continue;
            if (IS_READ_ONLY(it->first->usage))
              return true;
          }
        }
      }
      FieldMask not_observed = user_mask - observed;
      // If we had fields that were not observed, check the previous list
      if (!!not_observed)
      {
        for (LegionMap<Event,EventUsers>::aligned::const_iterator pit = 
              previous_epoch_users.begin(); pit != 
              previous_epoch_users.end(); pit++)
        {
          const EventUsers &event_users = pit->second;
          if (event_users.single)
          {
            FieldMask overlap = user_mask & event_users.user_mask;
            if (!overlap)
              continue;
            if (IS_READ_ONLY(event_users.users.single_user->usage))
              return true;
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              FieldMask overlap = user_mask & it->second;
              if (!overlap)
                continue;
              if (IS_READ_ONLY(it->first->usage))
                return true;
            }
          }
        }
      }
      return false;
    }
    
#if 0
    //--------------------------------------------------------------------------
    void MaterializedView::update_versions(const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, UPDATE_VERSIONS_CALL);
#endif
      std::vector<VersionID> to_delete;
      LegionMap<VersionID,FieldMask>::aligned new_versions;
      for (LegionMap<VersionID,FieldMask>::aligned::iterator it = 
            current_versions.begin(); it != current_versions.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!!overlap)
        {
          new_versions[(it->first+1)] = overlap; 
          it->second -= update_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
      }
      for (std::vector<VersionID>::const_iterator it = to_delete.begin();
            it != to_delete.end(); it++)
      {
        current_versions.erase(*it);
      }
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            new_versions.begin(); it != new_versions.end(); it++)
      {
        LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
          current_versions.find(it->first);
        if (finder == current_versions.end())
          current_versions.insert(*it);
        else
          finder->second |= it->second;
      }
    }
#endif

    //--------------------------------------------------------------------------
    void MaterializedView::filter_local_users(Event term_event) 
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      std::set<Event>::iterator event_finder = 
        outstanding_gc_events.find(term_event); 
      if (event_finder != outstanding_gc_events.end())
      {
        LegionMap<Event,EventUsers>::aligned::iterator current_finder = 
          current_epoch_users.find(term_event);
        if (current_finder != current_epoch_users.end())
        {
          EventUsers &event_users = current_finder->second;
          if (event_users.single)
          {
            if (event_users.users.single_user->remove_reference())
              legion_delete(event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                legion_delete(it->first);
            }
            delete event_users.users.multi_users;
          }
          current_epoch_users.erase(current_finder);
        }
        LegionMap<Event,EventUsers>::aligned::iterator previous_finder = 
          previous_epoch_users.find(term_event);
        if (previous_finder != previous_epoch_users.end())
        {
          EventUsers &event_users = previous_finder->second; 
          if (event_users.single)
          {
            if (event_users.users.single_user->remove_reference())
              legion_delete(event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                legion_delete(it->first);
            }
            delete event_users.users.multi_users;
          }
          previous_epoch_users.erase(previous_finder);
        }
        outstanding_gc_events.erase(event_finder);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_atomic_reservations(const FieldMask &mask,
                                                    Operation *op, bool excl)
    //--------------------------------------------------------------------------
    {
      // Keep going up the tree until we get to the root
      if (parent == NULL)
      {
        // Compute the field set
        std::set<FieldID> atomic_fields;
        logical_node->column_source->get_field_set(mask, atomic_fields);
        std::vector<std::pair<FieldID,Reservation> > to_send_back;
        // Take our lock and lookup the needed fields in order
        {
          AutoLock v_lock(view_lock);
          for (std::set<FieldID>::const_iterator it = atomic_fields.begin();
                it != atomic_fields.end(); it++)
          {
            std::map<FieldID,Reservation>::const_iterator finder = 
              atomic_reservations.find(*it);
            if (finder == atomic_reservations.end())
            {
              // Make a new reservation and add it to the set
              Reservation handle = Reservation::create_reservation();
              atomic_reservations[*it] = handle;
              op->update_atomic_locks(handle, excl);
              if (!is_owner())
                to_send_back.push_back(
                        std::pair<FieldID,Reservation>(*it, handle));
            }
            else
              op->update_atomic_locks(finder->second, excl);
          }
        }
        if (!to_send_back.empty())
          send_back_atomic_reservations(to_send_back);
      }
      else
        parent->find_atomic_reservations(mask, op, excl);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::set_descriptor(FieldDataDescriptor &desc,
                                          unsigned fid_idx) const
    //--------------------------------------------------------------------------
    {
      // Get the low-level index space
      const Domain &dom = logical_node->get_domain_no_wait();
      desc.index_space = dom.get_index_space();
      // Then ask the manager to fill in the rest of the information
      manager->set_descriptor(desc, fid_idx);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::send_back_atomic_reservations(
                  const std::vector<std::pair<FieldID,Reservation> > &send_back)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        bool is_region = logical_node->is_region();
        rez.serialize<bool>(is_region);
        if (is_region)
          rez.serialize(logical_node->as_region_node()->handle);
        else
          rez.serialize(logical_node->as_partition_node()->handle);
        rez.serialize<size_t>(send_back.size());
        for (std::vector<std::pair<FieldID,Reservation> >::const_iterator it =
              send_back.begin(); it != send_back.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      context->runtime->send_back_atomic(owner_space, rez);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_atomic_reservations(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_handles;
      derez.deserialize(num_handles);
      AutoLock v_lock(view_lock);
      for (unsigned idx = 0; idx < num_handles; idx++)
      {
        FieldID fid;
        derez.deserialize(fid);
        Reservation handle;
        derez.deserialize(handle);
        // TODO: If we ever hit this assertion then we need to serialize
        // atomic mappings occurring on different nodes at the same time.
        // This might occur because two tasks with atomic read-only
        // privileges mapped on different nodes and generated different
        // reservations for the same instance. We can either serialize
        // the mapping process or deal with sets of reservations for some
        // fields. Either way we defer this for later work.
        assert(atomic_reservations.find(fid) == atomic_reservations.end());
        atomic_reservations[fid] = handle;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_back_atomic(
                                                          RegionTreeForest *ctx,
                                                          Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        node = ctx->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        node = ctx->get_node(handle);
      }
      LogicalView *target_view = node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL 
      assert(target_view->is_instance_view());
#endif
      InstanceView *inst_view = target_view->as_instance_view();
#ifdef DEBUG_HIGH_LEVEL
      assert(inst_view->is_materialized_view()); 
#endif
      inst_view->as_materialized_view()->process_atomic_reservations(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_materialized_view(
                  Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      LogicalRegion handle;
      derez.deserialize(handle);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      unsigned depth;
      derez.deserialize(depth);
      UniqueID context_uid;
      derez.deserialize(context_uid);

      RegionNode *target_node = runtime->forest->get_node(handle); 
      PhysicalManager *phy_man = target_node->find_manager(manager_did);
#ifdef DEBUG_HIGH_LEVEL
      assert(!phy_man->is_reduction_manager());
#endif
      InstanceManager *inst_manager = phy_man->as_instance_manager();

      MaterializedView *new_view = legion_new<MaterializedView>(runtime->forest,
                                      did, owner_space, runtime->address_space,
                                      target_node, inst_manager, 
                                      (MaterializedView*)NULL/*parent*/,
                                      depth, false/*don't register yet*/,
                                      context_uid);
      if (!target_node->register_logical_view(new_view))
      {
        if (new_view->remove_base_resource_ref(REMOTE_DID_REF))
          legion_delete(new_view);
      }
      else
      {
        new_view->register_with_runtime();
        new_view->update_remote_instances(source);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_update(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *target_node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      DistributedID did;
      derez.deserialize(did);
      LogicalView *view = target_node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(view->is_instance_view());
      assert(view->as_instance_view()->is_materialized_view());
#endif
      MaterializedView *mat_view = 
        view->as_instance_view()->as_materialized_view();
      mat_view->process_update(derez, source);
    }

    /////////////////////////////////////////////////////////////
    // DeferredView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredView::DeferredView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp, AddressSpaceID local_sp,
                               RegionTreeNode *node, bool register_now)
      : LogicalView(ctx, did, owner_sp, local_sp, node, register_now)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    {
      // Remove resource references on our valid reductions
      for (std::deque<ReductionEpoch>::iterator rit = reduction_epochs.begin();
            rit != reduction_epochs.end(); rit++)
      {
        ReductionEpoch &epoch = *rit;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          if ((*it)->remove_nested_resource_ref(did))
            legion_delete(*it);
        }
      }
      reduction_epochs.clear();
    }

    //--------------------------------------------------------------------------
    void DeferredView::update_reduction_views(ReductionView *view,
                                              const FieldMask &valid,
                                              bool update_parent /*= true*/)
    //--------------------------------------------------------------------------
    {
      // First if we have a parent, we have to update its valid reduciton views
      if (update_parent && has_parent())
      {
        DeferredView *parent_view = get_parent()->as_deferred_view();
        parent_view->update_reduction_views_above(view, valid, this);
      }
      if ((logical_node == view->logical_node) ||  
          logical_node->intersects_with(view->logical_node))
      {
        // If it intersects, then we need to update our local reductions
        // and then also update any child reductions
        update_local_reduction_views(view, valid);
        update_child_reduction_views(view, valid);
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::update_reduction_epochs(const ReductionEpoch &epoch)
    //--------------------------------------------------------------------------
    {
      // This should be the parent and have no children
#ifdef DEBUG_HIGH_LEVEL
      assert(get_parent() == NULL);
#endif
      // No need to hold the lock since this is only called when 
      // the deferred view is being constructed
      reduction_epochs.push_back(epoch);
      // Don't forget to update the reduction mask
      reduction_mask |= epoch.valid_fields;
    }

    //--------------------------------------------------------------------------
    void DeferredView::update_reduction_views_above(ReductionView *view,
                                                    const FieldMask &valid,
                                                    DeferredView *from_child)
    //--------------------------------------------------------------------------
    {
      // Keep going up if necessary
      if (has_parent())
      {
        DeferredView *parent_view = get_parent()->as_deferred_view();
        parent_view->update_reduction_views_above(view, valid, this);
      }
      if ((logical_node == view->logical_node) ||
          logical_node->intersects_with(view->logical_node))
      {
        update_local_reduction_views(view, valid);
        update_child_reduction_views(view, valid, from_child);
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::update_local_reduction_views(ReductionView *view,
                                                    const FieldMask &valid_mask)
    //--------------------------------------------------------------------------
    {
      // We can do this before taking the lock
      ReductionOpID redop = view->get_redop();
      bool added = false;
      AutoLock v_lock(view_lock);
      reduction_mask |= valid_mask;
      // Iterate backwards and add to the first epoch that matches
      for (std::deque<ReductionEpoch>::reverse_iterator it = 
            reduction_epochs.rbegin(); it != reduction_epochs.rend(); it++)
      {
        if (redop != it->redop)
          continue;
        if (valid_mask * it->valid_fields)
          continue;
#ifdef DEBUG_HIGH_LEVEL
        assert(valid_mask == it->valid_fields);
#endif
        if (it->views.find(view) == it->views.end())
        {
          // Look at our state to see how to add the reduction view
#ifdef DEBUG_HIGH_LEVEL
          assert((current_state == INACTIVE_STATE) ||
                 (current_state == ACTIVE_INVALID_STATE) || 
                 (current_state == VALID_STATE));
#endif
          view->add_nested_resource_ref(did);
          if (current_state != INACTIVE_STATE)
            view->add_nested_gc_ref(did);
          if (current_state == VALID_STATE)
            view->add_nested_valid_ref(did);
          it->views.insert(view);
        }
        added = true;
        break;
      }
      if (!added)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((current_state == INACTIVE_STATE) ||
               (current_state == ACTIVE_INVALID_STATE) || 
               (current_state == VALID_STATE));
#endif
        view->add_nested_resource_ref(did);
        if (current_state != INACTIVE_STATE)
          view->add_nested_gc_ref(did);
        if (current_state == VALID_STATE)
          view->add_nested_valid_ref(did);
        reduction_epochs.push_back(ReductionEpoch(view, redop, valid_mask));
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::flush_reductions(const TraversalInfo &info,
                                        MaterializedView *dst,
                                        const FieldMask &reduce_mask,
                                LegionMap<Event,FieldMask>::aligned &conditions)
    //--------------------------------------------------------------------------
    {
      // Iterate over all the reduction epochs and issue reductions
      LegionDeque<ReductionEpoch>::aligned to_issue;  
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        for (LegionDeque<ReductionEpoch>::aligned::const_iterator it = 
              reduction_epochs.begin(); it != reduction_epochs.end(); it++)
        {
          if (reduce_mask * it->valid_fields)
            continue;
          to_issue.push_back(*it);
        }
      }
      for (LegionDeque<ReductionEpoch>::aligned::const_iterator rit = 
            to_issue.begin(); rit != to_issue.end(); rit++)
      {
        const ReductionEpoch &epoch = *rit; 
        // Compute the per-field preconditions
        std::set<Event> preconditions;
        for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
              conditions.begin(); it != conditions.end(); it++)
        {
          if (it->second * epoch.valid_fields)
            continue;
          preconditions.insert(it->first);
        }
        // Now issue the reductions from all the views
        std::set<Event> postconditions;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          std::set<Domain> component_domains;
          Event dom_pre = find_component_domains(*it, dst, component_domains);
          if (!component_domains.empty())
          {
            Event result = (*it)->perform_deferred_reduction(dst,
                                    epoch.valid_fields, info.version_info, 
                                    preconditions, component_domains, 
                                    dom_pre, info.op);
            if (result.exists())
              postconditions.insert(result);
          }
        }
        // Merge the post-conditions together and add them to results
        Event result = Runtime::merge_events<false>(postconditions);
        if (result.exists())
          conditions[result] = epoch.valid_fields;
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::flush_reductions_across(const TraversalInfo &info,
                                               MaterializedView *dst,
                                               FieldID src_field, 
                                               FieldID dst_field,
                                               Event dst_precondition,
                                               std::set<Event> &conditions)
    //--------------------------------------------------------------------------
    {
      // Find the reductions to perform
      unsigned src_index = 
        logical_node->column_source->get_field_index(src_field);
      std::deque<ReductionEpoch> to_issue;
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        for (std::deque<ReductionEpoch>::const_iterator it = 
              reduction_epochs.begin(); it != reduction_epochs.end(); it++)
        {
          if (it->valid_fields.is_set(src_index))
            to_issue.push_back(*it);
        }
      }
      if (!to_issue.empty())
      {
        std::set<Event> preconditions = conditions;
        preconditions.insert(dst_precondition);
        for (std::deque<ReductionEpoch>::const_iterator rit = 
              to_issue.begin(); rit != to_issue.end(); rit++)
        {
          const ReductionEpoch &epoch = *rit;
          std::set<Event> postconditions;
          for (std::set<ReductionView*>::const_iterator it = 
                epoch.views.begin(); it != epoch.views.end(); it++)
          {
            // Get the domains for this reduction view
            std::set<Domain> component_domains;
            Event dom_pre = find_component_domains(*it, dst, component_domains);
            if (!component_domains.empty())
            {
              Event result = (*it)->perform_deferred_across_reduction(dst,
                                              dst_field, src_field, src_index,
                                              info.version_info,
                                              preconditions, component_domains,
                                              dom_pre, info.op);
              if (result.exists())
                postconditions.insert(result);
            }
          }
          // Merge the postconditions
          Event result = Runtime::merge_events<false>(postconditions);
          if (result.exists())
          {
            conditions.insert(result);
            preconditions.insert(result);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    Event DeferredView::find_component_domains(ReductionView *reduction_view,
                                               MaterializedView *dst_view,
                                            std::set<Domain> &component_domains)
    //--------------------------------------------------------------------------
    {
      Event result = Event::NO_EVENT;
      if (dst_view->logical_node == reduction_view->logical_node)
      {
        if (dst_view->logical_node->has_component_domains())
          component_domains = 
            dst_view->logical_node->get_component_domains(result);
        else
          component_domains.insert(dst_view->logical_node->get_domain(result));
      }
      else
        component_domains = dst_view->logical_node->get_intersection_domains(
                                                reduction_view->logical_node);
      return result;
    }

    //--------------------------------------------------------------------------
    void DeferredView::activate_deferred(void)
    //--------------------------------------------------------------------------
    {
      // Add gc references to all our reduction views
      // Have to hold the lock when accessing this data structure 
      AutoLock v_lock(view_lock, 1, false/*exclusive*/);
      for (LegionDeque<ReductionEpoch>::aligned::const_iterator rit = 
            reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
      {
        const ReductionEpoch &epoch = *rit;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          (*it)->add_nested_gc_ref(did);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::deactivate_deferred(void)
    //--------------------------------------------------------------------------
    {
      // Hold the lock when accessing the reduction views
      AutoLock v_lock(view_lock, 1, false/*exclusive*/);
      for (LegionDeque<ReductionEpoch>::aligned::const_iterator rit = 
            reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
      {
        const ReductionEpoch &epoch = *rit;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          // No need to check for deletion condition since we hold resource refs
          (*it)->remove_nested_gc_ref(did);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::validate_deferred(void)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock, 1, false/*exclusive*/);
      for (LegionDeque<ReductionEpoch>::aligned::const_iterator rit = 
            reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
      {
        const ReductionEpoch &epoch = *rit;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          (*it)->add_nested_valid_ref(did);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::invalidate_deferred(void)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock, 1, false/*exclusive*/);
      for (LegionDeque<ReductionEpoch>::aligned::const_iterator rit = 
            reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
      {
        const ReductionEpoch &epoch = *rit;
        for (std::set<ReductionView*>::const_iterator it = 
              epoch.views.begin(); it != epoch.views.end(); it++)
        {
          // No need to check for deletion condition since we hold resource refs
          (*it)->remove_nested_valid_ref(did);
        }
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::send_deferred_view_updates(AddressSpaceID target,
                                                  const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<unsigned/*idx*/,ReductionEpoch>::aligned to_send;
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Quick check for being done
        if (update_mask * reduction_mask)
          return;
        unsigned idx = 0;
        for (LegionDeque<ReductionEpoch>::aligned::const_iterator it = 
             reduction_epochs.begin(); it != reduction_epochs.end(); it++,idx++)
        {
          if (update_mask * it->valid_fields)
            continue;
          to_send[idx] = *it;
        }
      }
      if (to_send.empty())
        return;
      // Now pack up the results and send the views in the process
      Serializer rez;
      {
        RezCheck z(rez);  
        bool is_region = logical_node->is_region();
        rez.serialize(is_region);
        if (is_region)
        {
          LogicalRegion handle = logical_node->as_region_node()->handle;
          rez.serialize(handle);
        }
        else
        {
          LogicalPartition handle = logical_node->as_partition_node()->handle;
          rez.serialize(handle);
        }
        rez.serialize(did);
        rez.serialize<size_t>(to_send.size());
        for (LegionMap<unsigned,ReductionEpoch>::aligned::const_iterator sit =
              to_send.begin(); sit != to_send.end(); sit++)
        {
          rez.serialize(sit->first);
          rez.serialize(sit->second.redop);
          rez.serialize(sit->second.valid_fields);
          rez.serialize<size_t>(sit->second.views.size());
          for (std::set<ReductionView*>::const_iterator it = 
                sit->second.views.begin(); it != sit->second.views.end(); it++)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((*it)->logical_node->is_region());
#endif
            rez.serialize((*it)->logical_node->as_region_node()->handle);
            DistributedID red_did = (*it)->send_view(target, update_mask); 
            rez.serialize(red_did);
          }
        }
      }
      runtime->send_deferred_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void DeferredView::process_deferred_view_update(Deserializer &derez, 
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t num_epochs;
      derez.deserialize(num_epochs);
      FieldSpaceNode *field_node = logical_node->column_source;
      AutoLock v_lock(view_lock);
      for (unsigned idx = 0; idx < num_epochs; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        if (index >= reduction_epochs.size())
          reduction_epochs.resize(index+1);
        ReductionEpoch &epoch = reduction_epochs[index]; 
        derez.deserialize(epoch.redop);
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        field_node->transform_field_mask(valid_fields, source);
        epoch.valid_fields |= valid_fields;
        reduction_mask |= valid_fields;
        size_t num_reductions;
        derez.deserialize(num_reductions);
        for (unsigned idx2 = 0; idx2 < num_reductions; idx2++)
        {
          LogicalRegion handle;
          derez.deserialize(handle);
          RegionTreeNode *node = context->get_node(handle);
          DistributedID red_did;
          derez.deserialize(red_did);
          LogicalView *red_view = node->find_view(red_did);
          red_view->add_nested_resource_ref(did);
          if (current_state != INACTIVE_STATE)
            red_view->add_nested_gc_ref(did);
          if (current_state == VALID_STATE)
            red_view->add_nested_valid_ref(did);
#ifdef DEBUG_HIGH_LEVEL
          assert(red_view->is_instance_view());
          assert(red_view->as_instance_view()->is_reduction_view());
#endif
          epoch.views.insert(red_view->as_instance_view()->as_reduction_view());
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void DeferredView::handle_deferred_update(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *target_node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      DistributedID did;
      derez.deserialize(did);
      LogicalView *view = target_node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(view->is_deferred_view());
#endif
      view->as_deferred_view()->process_deferred_view_update(derez, source);
    }

    /////////////////////////////////////////////////////////////
    // CompositeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(RegionTreeForest *ctx, DistributedID did,
                              AddressSpaceID owner_proc, RegionTreeNode *node,
                              AddressSpaceID local_proc, const FieldMask &mask,
                              bool register_now, CompositeView *par/*= NULL*/)
      : DeferredView(ctx, did, owner_proc, local_proc, node, register_now), 
        parent(par), valid_mask(mask)
    //--------------------------------------------------------------------------
    {
      // If we are either not a parent or we are a remote parent add 
      // a resource reference to avoid being collected
      if (parent != NULL)
        add_nested_resource_ref(did);
      else 
      {
        // Do remote registration for the top of each remote tree
        if (!is_owner())
        {
          add_base_resource_ref(REMOTE_DID_REF);
          send_remote_registration();
        }
      }
#ifdef LEGION_GC
      log_garbage.info("GC Composite View %ld", did);
#endif
    }
    
    //--------------------------------------------------------------------------
    CompositeView::CompositeView(const CompositeView &rhs)
      : DeferredView(NULL, 0, 0, 0, NULL, false), parent(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeView::~CompositeView(void)
    //--------------------------------------------------------------------------
    {
      // Remove any resource references that we hold on child views
      // Capture our child destruction events in the process
      for (std::map<ColorPoint,CompositeView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        recycle_events.insert(it->second->get_destruction_event());
        if (it->second->remove_nested_resource_ref(did))
          legion_delete(it->second);
      }
      children.clear();
      if ((parent == NULL) && is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
      // Remove any references we have to our roots
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        if (it->first->remove_reference())
          legion_delete(it->first);
      }
      roots.clear(); 
    }

    //--------------------------------------------------------------------------
    CompositeView& CompositeView::operator=(const CompositeView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* CompositeView::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<CompositeView,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void CompositeView::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_active(void)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->add_nested_gc_ref(did);

      activate_deferred();

      // If we are the top level view, add gc references to all our instances
      if (parent == NULL)
      {
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
              roots.begin(); it != roots.end(); it++)
        {
          it->first->add_gc_references();
        }
      } 
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
              roots.begin(); it != roots.end(); it++)
        {
          it->first->remove_gc_references();
        }
      }

      deactivate_deferred();

      if ((parent != NULL) && parent->remove_nested_gc_ref(did))
        legion_delete(parent); 
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, true/*add*/);
      }
      else
        parent->add_nested_valid_ref(did);

      for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->add_valid_references();
      }

      validate_deferred(); 
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->remove_valid_references();
      }

      invalidate_deferred(); 

      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, false/*add*/);
      }
      else if (parent->remove_nested_valid_ref(did))
        legion_delete(parent);
    }

    //--------------------------------------------------------------------------
    DistributedID CompositeView::send_view_base(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (!has_remote_instance(target))
      {
        if (parent == NULL)
        {
          // Don't take the lock, it's alright to have duplicate sends
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(owner_space);
            rez.serialize(valid_mask);
            bool is_region = logical_node->is_region();
            rez.serialize(is_region);
            if (is_region)
              rez.serialize(logical_node->as_region_node()->handle);
            else
              rez.serialize(logical_node->as_partition_node()->handle);
            rez.serialize<size_t>(roots.size());
            for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator 
                  it = roots.begin(); it != roots.end(); it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(it->first->logical_node == this->logical_node);
#endif
              // Pack the version info and then pack the composite tree
              // We know that the physical states have already been 
              // created for this version info in order to do the capture
              // of the tree, so we can pass in dummy context and 
              // local space parameters.
              VersionInfo &info = it->first->version_info->get_version_info();
              info.pack_version_info(rez, 0, 0);
              it->first->pack_composite_tree(rez, target); 
              rez.serialize(it->second);
            }
          }
          runtime->send_composite_view(target, rez);
        }
        else // Ask our parent to do the send
          parent->send_view_base(target);
        // We've done the send so record it
        update_remote_instances(target);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    void CompositeView::unpack_composite_view(Deserializer &derez, 
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t num_roots;
      derez.deserialize(num_roots);
      for (unsigned idx = 0; idx < num_roots; idx++)
      {
        CompositeVersionInfo *version_info = new CompositeVersionInfo();
        VersionInfo &info = version_info->get_version_info();
        info.unpack_version_info(derez);
        CompositeNode *new_node = legion_new<CompositeNode>(logical_node,
                            (CompositeNode*)NULL/*parent*/, version_info);
        new_node->unpack_composite_tree(derez, source);
        new_node->add_reference();
        FieldMask &mask = roots[new_node];
        derez.deserialize(mask);
        logical_node->column_source->transform_field_mask(mask, source);
        new_node->set_owner_did(did);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::make_local(std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      // This might not be the top of the view tree, but that is alright
      // because all the composite nodes share the same VersionInfo
      // data structure so we'll end up waiting for the right set of
      // version states to be local. It might be a little bit of an 
      // over approximation for sub-views, but that is ok for now.
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        VersionInfo &info = it->first->version_info->get_version_info();
        // If we are getting this call, we know we are on a remote node
        // so we know the physical states are already unpacked and therefore
        // we can pass in a dummy context ID
        info.make_local(preconditions, context, 0/*dummy ctx*/);
        // Now check the sub-tree for recursive composite views
        std::set<DistributedID> checked_views;
        it->first->make_local(preconditions, checked_views);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_send_composite_view(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner;
      derez.deserialize(owner);
      FieldMask valid_mask;
      derez.deserialize(valid_mask);
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *target_node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        target_node = runtime->forest->get_node(handle);
      }
      // Transform the fields mask 
      target_node->column_source->transform_field_mask(valid_mask, source);
      CompositeView *new_view = legion_new<CompositeView>(runtime->forest,
                            did, owner, target_node, runtime->address_space,
                            valid_mask, false/*register now*/);
      new_view->unpack_composite_view(derez, source);
      if (!target_node->register_logical_view(new_view))
      {
        if (new_view->remove_base_resource_ref(REMOTE_DID_REF))
          legion_delete(new_view);
      }
      else
      {
        new_view->register_with_runtime();
        new_view->update_remote_instances(source);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::send_view_updates(AddressSpaceID target,
                                          const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      // For composite views, we only need to send the view structures
      // of our actual instances, the enclosing version state will check
      // to see if our version infos are up to date. We do still need to
      // send updates for our constituent reduction views.
      send_deferred_view_updates(target, update_mask); 
    }

    //--------------------------------------------------------------------------
    void CompositeView::add_root(CompositeNode *root, 
                                 const FieldMask &valid, bool top)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid - valid_mask));
      // There should always be at most one root for each field
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        assert(it->second * valid);
      }
#endif
      LegionMap<CompositeNode*,FieldMask>::aligned::iterator finder = 
                                                            roots.find(root);
      if (finder == roots.end())
      {
        roots[root] = valid;
        // Add a reference for when we go to delete it
        root->add_reference();
      }
      else
        finder->second |= valid;
      if (top)
        root->set_owner_did(did);
    }

    //--------------------------------------------------------------------------
    void CompositeView::update_child_reduction_views(ReductionView *view,
                                                    const FieldMask &valid_mask,
                                                    DeferredView *to_skip)
    //--------------------------------------------------------------------------
    {
      // Make a copy of the child views and update them
      std::map<ColorPoint,CompositeView*> to_handle;
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        to_handle = children;
      }
      std::set<ColorPoint> handled;
      // Keep iterating until we've handled all the children
      while (!to_handle.empty())
      {
        for (std::map<ColorPoint,CompositeView*>::const_iterator it = 
              to_handle.begin(); it != to_handle.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(handled.find(it->first) == handled.end());
#endif
          handled.insert(it->first);
          if (it->second == to_skip)
            continue;
          it->second->update_reduction_views(view, valid_mask, false/*parent*/);
        }
        to_handle.clear();
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(handled.size() <= children.size());
#endif
        if (handled.size() == children.size())
          break;
        // Otherwise figure out what additional children to handle
        for (std::map<ColorPoint,CompositeView*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if (handled.find(it->first) == handled.end())
            to_handle.insert(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    LogicalView* CompositeView::get_subview(const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
      // This is the common case
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<ColorPoint,CompositeView*>::const_iterator finder = 
                                                          children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      RegionTreeNode *child_node = logical_node->get_tree_child(c); 
      CompositeView *child_view = legion_new<CompositeView>(context, did, 
                                                    owner_space, child_node,
                                                    local_space, valid_mask,
                                                    false/*register*/, this);
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        it->first->find_bounding_roots(child_view, it->second); 
      }
      
      // Retake the lock and try and add the child, see if
      // someone else added the child in the meantime
      bool free_child_view = false;
      CompositeView *result = child_view;
      {
        AutoLock v_lock(view_lock);
        std::map<ColorPoint,CompositeView*>::const_iterator finder = 
                                                          children.find(c);
        if (finder != children.end())
        {
          // Guaranteed to succeed
          if (child_view->remove_nested_resource_ref(did))
            free_child_view = true;
          result = finder->second;
        }
        else
        {
          children[c] = child_view;
          // Update the subviews while holding the lock
          for (std::deque<ReductionEpoch>::const_iterator rit = 
                reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
          {
            const ReductionEpoch &epoch = *rit;
            for (std::set<ReductionView*>::const_iterator it = 
                  epoch.views.begin(); it != epoch.views.end(); it++)
            {
              child_view->update_reduction_views(*it, epoch.valid_fields,
                                                 false/*update parent*/);
            }
          }
          child_node->register_logical_view(child_view);
        }
      }
      if (free_child_view)
        legion_delete(child_view);
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeView::update_valid_mask(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      valid_mask |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeView::flatten_composite_view(FieldMask &global_dirt,
                                               const FieldMask &flatten_mask, 
                                               CompositeCloser &closer, 
                                               CompositeNode *target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(valid_mask * flatten_mask)); // A little sanity check
#endif
      // Try to flatten this composite view, first make sure there are no
      // reductions which cannot be flattened
      LegionDeque<ReductionEpoch>::aligned flat_reductions;
      {
        // Hold the lock when 
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        for (LegionDeque<ReductionEpoch>::aligned::const_iterator it1 = 
              reduction_epochs.begin(); it1 != reduction_epochs.end(); it1++)
        {
          FieldMask overlap = it1->valid_fields & flatten_mask; 
          if (!overlap)
            continue;
          const ReductionEpoch &epoch = *it1; 
          flat_reductions.push_back(ReductionEpoch());
          ReductionEpoch &next = flat_reductions.back();
          for (std::set<ReductionView*>::const_iterator it = 
                epoch.views.begin(); it != epoch.views.end(); it++)
          {
            FieldMask temp = overlap;
            closer.filter_capture_mask((*it)->logical_node, temp);
            if (!!temp)
              next.views.insert(*it);
          }
          if (!next.views.empty())
          {
            next.valid_fields = overlap;
            next.redop = epoch.redop;
          }
          else // Actually empty so we can ignore it
            flat_reductions.pop_back();
        }
      }
      // Now see if we can flatten any roots
      LegionMap<CompositeNode*,FieldMask>::aligned new_roots;
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = flatten_mask & it->second;
        if (!overlap)
          continue;
        // Check to see if we already captured this root
        closer.filter_capture_mask(it->first->logical_node, overlap);
        if (!overlap)
          continue;
        // If we can't flatten the reductions, then we also 
        // can't flatten the roots
        CompositeNode *new_root = it->first->flatten(overlap, closer, 
                                         NULL/*parent*/, global_dirt, 
                                     (flat_reductions.empty() ? target : NULL));
        if (new_root != NULL)
          new_roots[new_root] = overlap;
      }
      // If we have no new roots and we have no reductions then 
      // we successfully flattened into the target
      if (!flat_reductions.empty() || !new_roots.empty())
      {
#ifdef DEBUG_HIGH_LEVEL
        // Sanity check that we always have something to apply reductions to
        if (!flat_reductions.empty())
          assert(!new_roots.empty()); 
#endif
        // Make a new composite view and then iterate over the roots
        DistributedID flat_did = 
          context->runtime->get_available_distributed_id(false);
        CompositeView *result = legion_new<CompositeView>(context, flat_did,
                                              context->runtime->address_space,
                                              logical_node,
                                              context->runtime->address_space,
                                              flatten_mask, true/*reg now*/);
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
              new_roots.begin(); it != new_roots.end(); it++)
        {
          result->add_root(it->first, it->second, true/*top*/);
        }
        for (std::deque<ReductionEpoch>::const_iterator it = 
              flat_reductions.begin(); it != flat_reductions.end(); it++)
        {
          result->update_reduction_epochs(*it);
        }
        // Add the new view to the target
        target->update_instance_views(result, flatten_mask); 
        // TODO: As an optimization, send the new composite view to all the 
        // known locations of this view so we can avoid duplicate sends of the 
        // meta-data for all the constituent views
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_field_descriptors(Event term_event, 
                                               const RegionUsage &usage, 
                                               const FieldMask &user_mask,
                                               unsigned fid_idx, Operation *op,
                                   std::vector<FieldDataDescriptor> &field_data,
                                   std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      // Iterate over all the roots and find the one for our event
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            roots.begin(); it != roots.end(); it++)
      {
        if (it->second.is_set(fid_idx))
        {
          Event target_pre;
          const Domain &target = 
            it->first->logical_node->get_domain(target_pre);
          std::vector<Realm::IndexSpace> already_handled;
          std::set<Event> already_preconditions;
          it->first->find_field_descriptors(term_event, usage,
                                            user_mask, fid_idx, op, 
                                            target.get_index_space(),
                                            target_pre, field_data, 
                                            preconditions, already_handled,
                                            already_preconditions);
          return;
        }
      }
      // We should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool CompositeView::find_field_descriptors(Event term_event,
                                               const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               unsigned fid_idx, Operation *op,
                                               Realm::IndexSpace target,
                                               Event target_precondition,
                                   std::vector<FieldDataDescriptor> &field_data,
                                   std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                             std::set<Event> &already_preconditions)
    //--------------------------------------------------------------------------
    {
      // Iterate over all the roots and find the one for our event
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            roots.begin(); it != roots.end(); it++)
      {
        if (it->second.is_set(fid_idx))
        {
          return it->first->find_field_descriptors(term_event, usage, user_mask,
                                                   fid_idx, op,
                                                   target, target_precondition,
                                                   field_data, preconditions,
                                                   already_handled, 
                                                   already_preconditions);
        }
      }
      // We should never get here
      assert(false);
      return false;
    }
    
    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              const FieldMask &copy_mask,
                                              CopyTracker *tracker /* = NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(copy_mask - valid_mask));
#endif
      LegionMap<Event,FieldMask>::aligned preconditions;
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   copy_mask, info.version_info, preconditions);
      // Iterate over all the roots and issue copies to update the 
      // target instance from this particular view
      LegionMap<Event,FieldMask>::aligned postconditions;
#ifdef DEBUG_HIGH_LEVEL
      FieldMask accumulate_mask;
#endif
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        it->first->issue_update_copies(info, dst, overlap, overlap,
                                       preconditions, postconditions, tracker);
#ifdef DEBUG_HIGH_LEVEL
        assert(overlap * accumulate_mask);
        accumulate_mask |= overlap;
#endif
      }
      // Now that we've issued all our copies, flush any reductions
      FieldMask reduce_overlap = reduction_mask & copy_mask;
      if (!!reduce_overlap)
        flush_reductions(info, dst, reduce_overlap, postconditions); 
      // Fun trick here, use the precondition set routine to get the
      // sets of fields which all have the same precondition events
      LegionList<EventSet>::aligned postcondition_sets;
      RegionTreeNode::compute_event_sets(copy_mask, postconditions,
                                         postcondition_sets);
      // Now add all the post conditions for each of the
      // writes for fields with the same set of post condition events
      for (LegionList<EventSet>::aligned::iterator pit = 
            postcondition_sets.begin(); pit !=
            postcondition_sets.end(); pit++)
      {
        EventSet &post_set = *pit;
        // Don't need to record anything if empty
        if (post_set.preconditions.empty())
          continue;
        // Compute the merge event
        Event postcondition = 
          Runtime::merge_events<false>(post_set.preconditions);
        if (postcondition.exists())
        {
          dst->add_copy_user(0/*redop*/, postcondition, info.version_info,
                             post_set.set_mask, false/*reading*/);
          if (tracker != NULL)
            tracker->add_copy_event(postcondition);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              const FieldMask &copy_mask,
                     const LegionMap<Event,FieldMask>::aligned &preconditions,
                           LegionMap<Event,FieldMask>::aligned &postconditions,
                                              CopyTracker *tracker /* = NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!(copy_mask - valid_mask));
#endif
      LegionMap<Event,FieldMask>::aligned local_postconditions;
#ifdef DEBUG_HIGH_LEVEL
      FieldMask accumulate_mask;
#endif
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        it->first->issue_update_copies(info, dst, overlap, overlap,
                                 preconditions, local_postconditions, tracker);
#ifdef DEBUG_HIGH_LEVEL
        assert(overlap * accumulate_mask);
        accumulate_mask |= overlap;
#endif
      }
      FieldMask reduce_overlap = reduction_mask & copy_mask;
      // Finally see if we have any reductions to flush
      if (!!reduce_overlap)
        flush_reductions(info, dst, reduce_overlap, postconditions);
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies_across(const TraversalInfo &info,
                                                     MaterializedView *dst,
                                                     FieldID src_field,
                                                     FieldID dst_field,
                                                     Event precondition,
                                                std::set<Event> &postconditions)
    //--------------------------------------------------------------------------
    {
      unsigned src_index = 
        logical_node->column_source->get_field_index(src_field);
      std::set<Event> preconditions;
      // This includes the destination precondition
      preconditions.insert(precondition);
      // Keep track of the local postconditions
      std::set<Event> local_postconditions;
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            roots.begin(); it != roots.end(); it++)
      {
        if (it->second.is_set(src_index))
        {
          it->first->issue_across_copies(info, dst, src_index, src_field, 
                                         dst_field, true/*need field*/,
                                         preconditions, local_postconditions);
          // We know there is at most one root here so
          // once we find it then we are done
          break;
        }
      }
      if (!reduction_epochs.empty() && reduction_mask.is_set(src_index))
      {
        // Merge our local postconditions to generate a new precondition
        Event local_postcondition = 
          Runtime::merge_events<false>(local_postconditions);
        flush_reductions_across(info, dst, src_field, dst_field,
                                local_postcondition, postconditions);
      }
      else // Otherwise we can just add locally
        postconditions.insert(local_postconditions.begin(),
                              local_postconditions.end());
    }

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeVersionInfo::CompositeVersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeVersionInfo::CompositeVersionInfo(const CompositeVersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeVersionInfo::~CompositeVersionInfo(void)
    //--------------------------------------------------------------------------
    {
      version_info.release();
    }

    //--------------------------------------------------------------------------
    CompositeVersionInfo& CompositeVersionInfo::operator=(
                                                const CompositeVersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(RegionTreeNode *logical, CompositeNode *par,
                                 CompositeVersionInfo *ver_info)
      : Collectable(), context(logical->context), logical_node(logical), 
        parent(par), version_info(ver_info), owner_did(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(version_info != NULL);
#endif
      version_info->add_reference();
    }

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(const CompositeNode &rhs)
      : Collectable(), context(NULL), logical_node(NULL), 
        parent(NULL), version_info(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeNode::~CompositeNode(void)
    //--------------------------------------------------------------------------
    {
      // Free up all our children
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        if (it->first->remove_reference())
          legion_delete(it->first);
      }
      open_children.clear();
      // Remove our resource references
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_base_resource_ref(COMPOSITE_NODE_REF))
          LogicalView::delete_logical_view(it->first);
      }
      valid_views.clear();
      if (version_info->remove_reference())
        delete version_info;
    }

    //--------------------------------------------------------------------------
    CompositeNode& CompositeNode::operator=(const CompositeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void* CompositeNode::operator new(size_t count)
    //--------------------------------------------------------------------------
    {
      return legion_alloc_aligned<CompositeNode,true/*bytes*/>(count);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::operator delete(void *ptr)
    //--------------------------------------------------------------------------
    {
      free(ptr);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::capture_physical_state(RegionTreeNode *tree_node,
                                               PhysicalState *state,
                                               const FieldMask &capture_mask,
                                               CompositeCloser &closer,
                                               FieldMask &global_dirt,
                                             const FieldMask &other_dirty_mask,
            const LegionMap<LogicalView*,FieldMask,
                            VALID_VIEW_ALLOC>::track_aligned &other_valid_views)
    //--------------------------------------------------------------------------
    {
      // Capture the global dirt we are passing back
      global_dirt |= (capture_mask & other_dirty_mask);
      // Also track the fields that we have dirty data for
      FieldMask local_dirty = capture_mask & other_dirty_mask;
      // Record the local dirty fields
      dirty_mask |= local_dirty;
      LegionMap<CompositeView*,FieldMask>::aligned to_flatten;
      FieldMask need_flatten = capture_mask;
      // Only need to pull down valid views if we are at the top
      if ((state != NULL) && (parent == NULL))
      {
        LegionMap<LogicalView*,FieldMask>::aligned instances;
        tree_node->find_valid_instance_views(closer.ctx, state, capture_mask,
            capture_mask, closer.version_info, false/*needs space*/, instances);
        capture_instances(capture_mask, need_flatten, to_flatten, instances);  
      }
      else
      {
        capture_instances(capture_mask, need_flatten, 
                          to_flatten, other_valid_views); 
      }
      // This is a very important optimization! We can't just blindly capture
      // all valid views because there might be some composite views in here.
      // If we continue doing this, we may end up with a chain of composite
      // views reaching all the way back to the start of the task which would
      // be bad. Instead we do a flattening procedure which prevents duplicate
      // captures of the same logical nodes for the same fields. This way we
      // only capture the most recent necessary data. Note there can still be
      // nested composite views, but none with overlapping information.
      if (!!need_flatten)
      {
        for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
              to_flatten.begin(); it != to_flatten.end(); it++)
        {
          FieldMask overlap = need_flatten & it->second;
          if (!overlap)
            continue;
          it->first->flatten_composite_view(global_dirt, overlap, closer, this);
          need_flatten -= overlap;
          if (!need_flatten)
            break;
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename MAP_TYPE>
    void CompositeNode::capture_instances(const FieldMask &capture_mask,
                                          FieldMask &need_flatten,
                    LegionMap<CompositeView*,FieldMask>::aligned &to_flatten,
                                          const MAP_TYPE &instances)
    //--------------------------------------------------------------------------
    {
      // Capture as many non-composite instances as we can, as long as we
      // have one for each field then we are good, otherwise, we need to 
      // flatten a composite instance to capture the necessary state
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        // Figure out what kind of view we have
        if (it->first->is_deferred_view())
        {
          DeferredView *def_view = it->first->as_deferred_view();
          if (def_view->is_composite_view())
          {
            CompositeView *comp_view = def_view->as_composite_view();
            to_flatten[comp_view] = overlap;
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(def_view->is_fill_view());
#endif
            update_instance_views(def_view, overlap);
            need_flatten -= overlap;
          }
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(it->first->is_instance_view());
          assert(it->first->as_instance_view()->is_materialized_view());
#endif
          update_instance_views(it->first, overlap);
          need_flatten -= overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeNode::flatten(const FieldMask &flatten_mask,
                                          CompositeCloser &closer,
                                          CompositeNode *parent,
                                          FieldMask &global_dirt,
                                          CompositeNode *target)
    //--------------------------------------------------------------------------
    {
      CompositeNode *result = NULL;
      // If we don't have a target, go ahead and make the clone
      // We're also only allowed to flatten to a node of the same kind
      // so go ahead make a clone if we aren't the same
      if ((target == NULL) || (target->logical_node != logical_node))
        result = create_clone_node(parent, closer); 
      // First capture down the tree  
      for (LegionMap<CompositeNode*,ChildInfo>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = flatten_mask & it->second.open_fields;
        if (!overlap)
          continue;
        // Check to see if it has already been handled
        closer.filter_capture_mask(it->first->logical_node, overlap);
        if (!overlap)
          continue;
        CompositeNode *flat_child = it->first->flatten(overlap, closer, 
                                       result, global_dirt, NULL/*no target*/);
        // Make the result if we haven't yet
        if (result == NULL)
          result = create_clone_node(parent, closer);
        result->update_child_info(flat_child, overlap);
      }
      // Then capture ourself if we don't have anyone below, we can capture
      // directly into the target, otherwise we have to capture to our result
      if (result == NULL)
        target->capture_physical_state(logical_node, NULL/*state*/, 
            flatten_mask, closer, global_dirt, dirty_mask, valid_views);
      else
        result->capture_physical_state(logical_node, NULL/*state*/, 
            flatten_mask, closer, global_dirt, dirty_mask, valid_views);
      // Finally update the closer with the capture fields
      closer.update_capture_mask(logical_node, flatten_mask);
      return result;
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeNode::create_clone_node(CompositeNode *parent,
                                                    CompositeCloser &closer)
    //--------------------------------------------------------------------------
    {
      // If we're making a copy and we don't have a parent, make a new
      // version info and then capture the needed version infos
      CompositeNode *result;
      if (parent == NULL)
      {
        CompositeVersionInfo *new_info = new CompositeVersionInfo();
        result = legion_new<CompositeNode>(logical_node, parent, new_info);
        // We need to capture the version info for all the nodes 
        // except the ones we know we've already captured
        VersionInfo &new_version_info = new_info->get_version_info();
        const VersionInfo &old_info = version_info->get_version_info();
        new_version_info.clone_from(old_info, closer);
      }
      else
        result = legion_new<CompositeNode>(logical_node, parent,
                                           parent->version_info);
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_parent_info(const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->update_child_info(this, capture_mask);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_child_info(CompositeNode *child, 
                                          const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      std::map<CompositeNode*,ChildInfo>::iterator finder = 
                                                    open_children.find(child); 
      if (finder == open_children.end())
      {
        // If we didn't find it, we have to make it
        // and determine if it is complete
        bool complete = child->logical_node->is_complete();
        open_children[child] = ChildInfo(complete, mask);
        // Add a reference to it
        child->add_reference();
      }
      else
        finder->second.open_fields |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_instance_views(LogicalView *view,
                                              const FieldMask &valid_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(logical_node == view->logical_node);
#endif
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        valid_views.find(view);
      if (finder == valid_views.end())
      {
        view->add_base_resource_ref(COMPOSITE_NODE_REF);
        valid_views[view] = valid_mask;
      }
      else
        finder->second |= valid_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_update_copies(const TraversalInfo &info,
                                            MaterializedView *dst,
                                            FieldMask traversal_mask,
                                            const FieldMask &copy_mask,
                            const LegionMap<Event,FieldMask>::aligned &preconds,
                            LegionMap<Event,FieldMask>::aligned &postconditions,
                                            CopyTracker *tracker /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      // First check to see if any of our children are complete
      // If they are then we can skip issuing any copies from this level
      LegionMap<Event,FieldMask>::aligned dst_preconditions = preconds;
      if (!valid_views.empty())
      {
        // The fields we need to update are any in the traversal
        // mask plus any from the original copy for which we have dirty data
        FieldMask incomplete_mask = traversal_mask | (dirty_mask & copy_mask);
        // Otherwise, if none are complete, we need to issue update copies
        // from this level assuming we have instances that intersect
        bool already_valid = false;
        // Do a quick check to see if we are done early
        {
          LegionMap<LogicalView*,FieldMask>::aligned::const_iterator finder = 
            valid_views.find(dst);
          if ((finder != valid_views.end()) && 
              !(incomplete_mask - finder->second))
            already_valid = true;
        }
        if (!already_valid && !!incomplete_mask)
        {
          RegionTreeNode *target = dst->logical_node;
          LegionMap<LogicalView*,FieldMask>::aligned valid_instances;
          for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
                valid_views.begin(); it != valid_views.end(); it++)
          {
            FieldMask overlap = incomplete_mask & it->second;
            if (!overlap)
              continue;
            valid_instances[it->first] = overlap;
          }
          LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
          LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
          // Note that this call destroys valid_instances 
          // and updates incomplete_mask
          target->sort_copy_instances(info, dst, incomplete_mask, 
                      valid_instances, src_instances, deferred_instances);
          if (!src_instances.empty())
          {
            // Use our version info for the sources
            const VersionInfo &src_info = version_info->get_version_info();
            LegionMap<Event,FieldMask>::aligned update_preconditions;
            FieldMask update_mask;
            for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator
                  it = src_instances.begin(); it != src_instances.end(); it++)
            {
#ifdef DEBUG_HIGH_LEVEL
              assert(!!it->second);
#endif
              it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                it->second, src_info, update_preconditions);
              update_mask |= it->second;
            }
            // Also get the set of destination preconditions
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                  preconds.begin(); it != preconds.end(); it++)
            {
              FieldMask overlap = update_mask & it->second;
              if (!overlap)
                continue;
              LegionMap<Event,FieldMask>::aligned::iterator finder = 
                update_preconditions.find(it->first);
              if (finder == update_preconditions.end())
                update_preconditions[it->first] = overlap;
              else
                finder->second |= overlap;
            }
            
            // Now we have our preconditions so we can issue our copy
            LegionMap<Event,FieldMask>::aligned update_postconditions;
            RegionTreeNode::issue_grouped_copies(context, info, dst, 
                         update_preconditions, update_mask, Event::NO_EVENT,
                         find_intersection_domains(dst->logical_node),
                         src_instances,src_info,update_postconditions,tracker);
            // If we dominate the target, then we can remove
            // the update_mask fields from the traversal_mask
            if (dominates(dst->logical_node))
              traversal_mask -= update_mask;
            // Add all our updates to both the dst_preconditions
            // as well as the actual postconditions.  No need to
            // check for duplicates as we know all these events
            // are brand new and can't be anywhere else.
            if (!update_postconditions.empty())
            {
#ifdef DEBUG_HIGH_LEVEL
              for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                    update_postconditions.begin(); it != 
                    update_postconditions.end(); it++)
              {
                assert(dst_preconditions.find(it->first) == 
                       dst_preconditions.end());
                assert(postconditions.find(it->first) == 
                       postconditions.end());
              }
#endif
              dst_preconditions.insert(update_postconditions.begin(),
                                       update_postconditions.end());
              postconditions.insert(update_postconditions.begin(),
                                    update_postconditions.end());
            }
          }
          // Now if we still have fields which aren't
          // updated then we need to see if we have composite
          // views for those fields
          if (!deferred_instances.empty())
          {
            FieldMask update_mask;
            for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator
                  it = deferred_instances.begin(); it !=
                  deferred_instances.end(); it++)
            {
              LegionMap<Event,FieldMask>::aligned postconds;
              it->first->issue_deferred_copies(info, dst, it->second,
                                               preconds, postconds, tracker);
              update_mask |= it->second;
              if (!postconds.empty())
              {
#ifdef DEBUG_HIGH_LEVEL
                for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                      postconds.begin(); it != postconds.end(); it++)
                {
                  assert(dst_preconditions.find(it->first) ==
                         dst_preconditions.end());
                  assert(postconditions.find(it->first) ==
                         postconditions.end());
                }
#endif
                dst_preconditions.insert(postconds.begin(), postconds.end());
                postconditions.insert(postconds.begin(), postconds.end());
              }
            }
            // If we dominate the logical node we can remove the
            // updated fields from the traversal mask
            if (dominates(dst->logical_node))
              traversal_mask -= update_mask;
          }
        }
      }

      // Now traverse any open children that intersect with the destination
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = copy_mask & it->second.open_fields;
        // If we have no fields in common or we don't intersect with
        // the child then we can skip traversing this child
        if (!overlap || !it->first->intersects_with(dst->logical_node))
          continue;
        // If we make it here then we need to traverse the child
        it->first->issue_update_copies(info, dst, traversal_mask, 
                                       overlap, dst_preconditions, 
                                       postconditions, tracker);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_across_copies(const TraversalInfo &info,
                                            MaterializedView *dst,
                                            unsigned src_index,
                                            FieldID  src_field,
                                            FieldID  dst_field,
                                            bool    need_field,
                                            std::set<Event> &preconditions,
                                            std::set<Event> &postconditions)
    //--------------------------------------------------------------------------
    {
      std::set<Event> dst_preconditions = preconditions;
      if (!valid_views.empty())
      {
        bool incomplete = need_field || dirty_mask.is_set(src_index);
        if (incomplete)
        {
          FieldMask src_mask; src_mask.set_bit(src_index);
          LegionMap<LogicalView*,FieldMask>::aligned valid_instances;
          for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
                valid_views.begin(); it != valid_views.end(); it++)
          {
            if (it->second.is_set(src_index))
              valid_instances[it->first] = src_mask;
          }
          LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
          LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
          dst->logical_node->sort_copy_instances(info, dst, src_mask,
                      valid_instances, src_instances, deferred_instances);
          if (!src_instances.empty())
          {
            // There should be at most one of these
#ifdef DEBUG_HIGH_LEVEL
            assert(src_instances.size() == 1);
#endif
            MaterializedView *src = (src_instances.begin())->first;
            LegionMap<Event,FieldMask>::aligned src_preconditions;
            src->find_copy_preconditions(0/*redop*/, true/*reading*/,
                          src_mask, version_info->get_version_info(), 
                          src_preconditions);
            for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
                  src_preconditions.begin(); it != 
                  src_preconditions.end(); it++)
            {
              preconditions.insert(it->first);
            }
            Event copy_pre = Runtime::merge_events<false>(preconditions);
            std::set<Event> result_events;
            std::vector<Domain::CopySrcDstField> src_fields, dst_fields;
            src->copy_field(src_field, src_fields);
            dst->copy_field(dst_field, dst_fields);
            const std::set<Domain> &overlap_domains = 
              find_intersection_domains(dst->logical_node);
            for (std::set<Domain>::const_iterator it = overlap_domains.begin();
                  it != overlap_domains.end(); it++)
            {
              result_events.insert(context->issue_copy(*it, info.op, src_fields,
                                                       dst_fields, copy_pre));
            }
            Event copy_post = Runtime::merge_events<false>(result_events);
            if (copy_post.exists())
            {
              // Only need to record the source user as the destination
              // user will be recorded by the copy across operation
              src->add_copy_user(0/*redop*/, copy_post,
                                 version_info->get_version_info(),
                                 src_mask, true/*reading*/);
              // Also add the event to the dst_preconditions and 
              // our post conditions
              dst_preconditions.insert(copy_post);
              postconditions.insert(copy_post);
            }
            // If we dominate then we no longer need to get
            // updates unless they are dirty
            if (dominates(dst->logical_node))
              need_field = false;
          }
          else if (!deferred_instances.empty())
          {
            // There should be at most one of these
#ifdef DEBUG_HIGH_LEVEL
            assert(deferred_instances.size() == 1); 
#endif
            DeferredView *src = (deferred_instances.begin())->first; 
            std::set<Event> postconds;
            Event pre = Runtime::merge_events<false>(preconditions);
            src->issue_deferred_copies_across(info, dst, src_field,
                                              dst_field, pre, postconds);
            if (!postconds.empty())
            {
              dst_preconditions.insert(postconds.begin(), postconds.end());
              postconditions.insert(postconds.begin(), postconds.end());
            }
            // If we dominate then we no longer need to get
            // updates unless they are dirty
            if (dominates(dst->logical_node))
              need_field = false;
          }
        }
      }
      // Now traverse any open children that intersect with the destination
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        if ((it->second.open_fields.is_set(src_index)) && 
            it->first->intersects_with(dst->logical_node))
        {
          it->first->issue_across_copies(info, dst, src_index,
                                         src_field, dst_field, need_field,
                                         dst_preconditions, postconditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool CompositeNode::intersects_with(RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->intersects_with(dst);
    }

    //--------------------------------------------------------------------------
    const std::set<Domain>& CompositeNode::find_intersection_domains(
                                                            RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->get_intersection_domains(dst);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::find_bounding_roots(CompositeView *target,
                                            const FieldMask &bounding_mask)
    //--------------------------------------------------------------------------
    {
      // See if we can fields with exactly one child that dominates
      FieldMask single, multi;
      LegionMap<CompositeNode*,FieldMask>::aligned dominators;
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second.open_fields & bounding_mask;
        if (!overlap)
          continue;
        if (!it->first->dominates(target->logical_node))
          continue;
        LegionMap<CompositeNode*,FieldMask>::aligned::iterator finder = 
          dominators.find(it->first);
        if (finder == dominators.end())
          dominators[it->first] = overlap;
        else
          finder->second |= overlap;
        // Update the multi mask first 
        multi |= (single & overlap);
        // Now update the single mask
        single |= overlap;
      }
      // Subtract any fields from the multi mask from the single mask
      if (!!multi)
        single -= multi;
      // If we still have any single fields then go and issue them
      if (!!single)
      {
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
              dominators.begin(); it != dominators.end(); it++)
        {
          FieldMask overlap = single & it->second;
          if (!overlap)
            continue;
          it->first->find_bounding_roots(target, overlap);
        }
        // Now see if we have any leftover fields here
        FieldMask local_mask = bounding_mask - single;
        if (!!local_mask)
          target->add_root(this, local_mask, false/*top*/);
      }
      else
      {
        // There were no single fields so add ourself
        target->add_root(this, bounding_mask, false/*top*/);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::set_owner_did(DistributedID own_did)
    //--------------------------------------------------------------------------
    {
      owner_did = own_did;
      for (LegionMap<CompositeNode*,ChildInfo>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->set_owner_did(owner_did);
      }
    } 

    //--------------------------------------------------------------------------
    bool CompositeNode::find_field_descriptors(Event term_event,
                                               const RegionUsage &usage, 
                                               const FieldMask &user_mask,
                                               unsigned fid_idx, Operation *op,
                                               Realm::IndexSpace target,
                                               Event target_precondition,
                                   std::vector<FieldDataDescriptor> &field_data,
                                   std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                             std::set<Event> &already_preconditions)
    //--------------------------------------------------------------------------
    {
      // We need to find any field descriptors in our children  
      // If any of the children are complete then we are done here too 
      // and can continue on, otherwise, we also need to register at least
      // one local instance if it exists.

      // Keep track of all the index spaces we've handled below
      std::vector<Realm::IndexSpace> handled_index_spaces;
      // Keep track of the preconditions for using the handled index spaces
      std::set<Event> handled_preconditions;
      unsigned done_children = 0;
      Event domain_precondition;
      const Domain &local_domain = 
        logical_node->get_domain(domain_precondition);
      bool need_child_intersect = (target != local_domain.get_index_space());
      for (LegionMap<CompositeNode*,ChildInfo>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        if (it->second.open_fields.is_set(fid_idx))
        {
          bool done;
          // Compute the low-level index space to ask for from the child
          Event child_precondition;
          const Domain &child_domain = 
            it->first->logical_node->get_domain(child_precondition);
          if (need_child_intersect)
          {
            // Compute the intersection of our target with the child
            std::vector<Realm::IndexSpace::BinaryOpDescriptor> ops(1);
            ops[0].op = Realm::IndexSpace::ISO_INTERSECT;
            ops[0].parent = local_domain.get_index_space();
            ops[0].left_operand = target;
            ops[0].right_operand = child_domain.get_index_space();
            Event pre = Runtime::merge_events<true>(target_precondition, 
                                      child_precondition, domain_precondition);
            Event child_ready = Realm::IndexSpace::compute_index_spaces(ops,
                                                        false/*mutable*/, pre);
            done = it->first->find_field_descriptors(term_event, usage,
                                                   user_mask,fid_idx, op,
                                                   ops[0].result, child_ready,
                                                   field_data, preconditions,
                                                   handled_index_spaces,
                                                   handled_preconditions);
            // We can also issue the deletion for the child index space
            ops[0].result.destroy(term_event);
          }
          else
            done = it->first->find_field_descriptors(term_event, usage,
                                                user_mask, fid_idx, op,
                                                child_domain.get_index_space(), 
                                                child_precondition,
                                                field_data, preconditions,
                                                handled_index_spaces,
                                                handled_preconditions);
          // If it is complete and we handled everything, then we are done
          if (done)
          {
            done_children++;
            if (it->second.complete && done)
              return true;
          }
        }
      }
      // If we're complete and we closed all the children, then we are
      //also done
      if (logical_node->is_complete() && 
          (done_children == logical_node->get_num_children()))
        return true;
      // If we make it here, we weren't able to cover ourselves, so make an 
      // index space for the remaining set of points we need to handle
      // First compute what we did handle
      Realm::IndexSpace local_handled = Realm::IndexSpace::NO_SPACE;
      Event local_precondition = Event::NO_EVENT;
      if (handled_index_spaces.size() == 1)
      {
        local_handled = handled_index_spaces.front();
        if (!handled_preconditions.empty())
          local_precondition = *(handled_preconditions.begin());
      }
      else if (handled_index_spaces.size() > 1)
      {
        Event parent_precondition;
        const Domain &parent_dom = 
          logical_node->get_domain(parent_precondition);
        if (parent_precondition.exists())
          handled_preconditions.insert(parent_precondition);
        // Compute the union of all our handled index spaces
        Event handled_pre = Runtime::merge_events<true>(handled_preconditions);
        local_precondition = Realm::IndexSpace::reduce_index_spaces( 
                              Realm::IndexSpace::ISO_UNION,
                              handled_index_spaces, local_handled,
                              false/*not mutable*/, 
                              parent_dom.get_index_space(), handled_pre);
        // We can also emit the destruction for this temporary index space now
        local_handled.destroy(term_event);
      }
      // Now we can compute the remaining part of the index space
      Realm::IndexSpace remaining_space = target;
      Event remaining_precondition = target_precondition;
      if (local_handled.exists())
      {
        // Compute the set difference
        std::vector<Realm::IndexSpace::BinaryOpDescriptor> ops(1);
        ops[0].op = Realm::IndexSpace::ISO_SUBTRACT;
        ops[0].parent = local_domain.get_index_space();
        ops[0].left_operand = target;
        ops[0].right_operand = local_handled;
        Event pre = Runtime::merge_events<true>(target_precondition,
                                        local_precondition,domain_precondition);
        remaining_precondition = Realm::IndexSpace::compute_index_spaces(ops,
                                                        false/*mutable*/, pre);
        remaining_space = ops[0].result;
        // We also emit the destruction for this temporary index space
        remaining_space.destroy(term_event);
      }
      // If we make it here we need to register at least one instance
      // from ourself if there are any
      DeferredView *deferred_view = NULL;
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        // Check to see if the instance is valid for our target field
        if (it->second.is_set(fid_idx))
        {
          // See if this is a composite view
          if (it->first->is_instance_view())
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(it->first->as_instance_view()->is_materialized_view());
#endif
            MaterializedView *view = 
              it->first->as_instance_view()->as_materialized_view();
            // Record the instance and its information
            field_data.push_back(FieldDataDescriptor());
            view->set_descriptor(field_data.back(), fid_idx);
              
            field_data.back().index_space = remaining_space;
            // Register ourselves as a user of this instance
            Event ref_ready = view->add_user(usage, term_event, user_mask, op,
                                            version_info->get_version_info());
            Event ready_event = Runtime::merge_events<true>(
                                        ref_ready, remaining_precondition);
            if (ready_event.exists())
              preconditions.insert(ready_event);
            // Record that we handled the remaining space
            already_handled.push_back(remaining_space);
            if (remaining_precondition.exists())
              already_preconditions.insert(remaining_precondition);
            // We found an actual instance, so we are done
            return true;
          }
          else
          {
            // Save it as a composite view and keep going
#ifdef DEBUG_HIGH_LEVEL
            assert(it->first->is_deferred_view());
            assert(deferred_view == NULL);
#endif
            deferred_view = it->first->as_deferred_view();
          }
        }
      }
      // If we made it here, we're not sure if we covered everything
      // or not, so record what we have handled
      if (local_handled.exists())
      {
        already_handled.push_back(local_handled);
        if (local_precondition.exists())
          already_preconditions.insert(local_precondition);
      }
      // If we still have a composite view, then register that
      if (deferred_view != NULL)
        return deferred_view->find_field_descriptors(term_event, usage, 
                                                     user_mask, fid_idx, op, 
                                                     remaining_space, 
                                                     remaining_precondition,
                                                     field_data, preconditions,
                                                     already_handled,
                                                     already_preconditions);
      return false;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::add_gc_references(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_gc_ref(owner_did);
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->add_gc_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::remove_gc_references(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.end(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_nested_gc_ref(owner_did);
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->remove_gc_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::add_valid_references(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_valid_ref(owner_did);
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->add_valid_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::remove_valid_references(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.end(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->add_nested_valid_ref(owner_did);
      }
      for (std::map<CompositeNode*,ChildInfo>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->remove_valid_references();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::pack_composite_tree(Serializer &rez, 
                                            AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(dirty_mask);
      rez.serialize<size_t>(valid_views.size());
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        // Only need to send the structure for now, we'll check for
        // updates when we unpack and request anything we need later
        DistributedID did = it->first->send_view_base(target); 
        rez.serialize(did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(open_children.size());
      for (LegionMap<CompositeNode*,ChildInfo>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        rez.serialize(it->first->logical_node->get_color());
        rez.serialize<bool>(it->second.complete);
        rez.serialize(it->second.open_fields);
        it->first->pack_composite_tree(rez, target);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::unpack_composite_tree(Deserializer &derez,
                                              AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      FieldSpaceNode *field_node = logical_node->column_source;
      derez.deserialize(dirty_mask); 
      field_node->transform_field_mask(dirty_mask, source);
      size_t num_valid_views;
      derez.deserialize(num_valid_views);
      for (unsigned idx = 0; idx < num_valid_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        LogicalView *view = logical_node->find_view(did);
        view->add_base_resource_ref(COMPOSITE_NODE_REF);
        FieldMask &mask = valid_views[view];
        derez.deserialize(mask);
        field_node->transform_field_mask(mask, source);
      }
      size_t num_open_children;
      derez.deserialize(num_open_children);
      for (unsigned idx = 0; idx < num_open_children; idx++)
      {
        ColorPoint child_color;
        derez.deserialize(child_color);
        RegionTreeNode *child_node = logical_node->get_tree_child(child_color);
        CompositeNode *new_node = legion_new<CompositeNode>(child_node, this,
                                                            version_info);
        ChildInfo &info = open_children[new_node];
        new_node->add_reference();
        derez.deserialize<bool>(info.complete);
        derez.deserialize(info.open_fields);
        field_node->transform_field_mask(info.open_fields, source);
        new_node->unpack_composite_tree(derez, source);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::make_local(std::set<Event> &preconditions,
                                   std::set<DistributedID> &checked_views)
    //--------------------------------------------------------------------------
    {
      // Check all our views for composite instances so we do any
      // recursive checking for up-to-date views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        // If we already checked this view, we are good
        if (checked_views.find(it->first->did) != checked_views.end())
          continue;
        checked_views.insert(it->first->did);
        if (it->first->is_deferred_view())
        {
          DeferredView *def_view = it->first->as_deferred_view();
          if (def_view->is_composite_view())
          {
            def_view->as_composite_view()->make_local(preconditions);
          }
        }
      }
      // Then traverse any children
      for (LegionMap<CompositeNode*,ChildInfo>::aligned::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        it->first->make_local(preconditions, checked_views);
      }
    }
    
    //--------------------------------------------------------------------------
    bool CompositeNode::dominates(RegionTreeNode *dst)
    //--------------------------------------------------------------------------
    {
      return logical_node->dominates(dst);
    }

    /////////////////////////////////////////////////////////////
    // FillView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, AddressSpaceID local_proc,
                       RegionTreeNode *node, bool reg_now, 
                       FillViewValue *val, FillView *par)
      : DeferredView(ctx, did, owner_proc, local_proc, node, reg_now), 
        parent(par), value(val)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(value != NULL);
#endif
      value->add_reference();
      if (parent != NULL)
        add_nested_resource_ref(did);
      else if (!is_owner())
      {
        add_base_resource_ref(REMOTE_DID_REF);
        send_remote_registration();
      }
#ifdef LEGION_GC
      log_garbage.info("GC Fill View %ld", did);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::FillView(const FillView &rhs)
      : DeferredView(NULL, 0, 0, 0, NULL, false), parent(NULL), value(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    FillView::~FillView(void)
    //--------------------------------------------------------------------------
    {
      if (value->remove_reference())
        delete value;
      // Clean up our children and capture their destruction events
      for (std::map<ColorPoint,FillView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        recycle_events.insert(it->second->get_destruction_event());
        if (it->second->remove_nested_resource_ref(did))
          legion_delete(it->second);
      }
      if ((parent == NULL) && is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
    }

    //--------------------------------------------------------------------------
    FillView& FillView::operator=(const FillView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FillView::notify_active(void)
    //--------------------------------------------------------------------------
    {
      activate_deferred();
    }

    //--------------------------------------------------------------------------
    void FillView::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      deactivate_deferred();
    }
    
    //--------------------------------------------------------------------------
    void FillView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, true/*add*/);
      }
      else 
        parent->add_nested_valid_ref(did);
      validate_deferred();
    }

    //--------------------------------------------------------------------------
    void FillView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      invalidate_deferred();
      if (parent == NULL)
      {
        if (!is_owner())
          send_remote_valid_update(owner_space, 1/*count*/, false/*add*/);
      }
      else if (parent->remove_nested_valid_ref(did))
        legion_delete(parent);
    }

    //--------------------------------------------------------------------------
    DistributedID FillView::send_view_base(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (!has_remote_instance(target))
      {
        if (parent == NULL)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(logical_node->is_region()); // Always regions at the top
#endif
          // Don't take the lock, it's alright to have duplicate sends
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(owner_space);
            rez.serialize(logical_node->as_region_node()->handle);
            rez.serialize(value->value_size);
            rez.serialize(value->value, value->value_size);
          }
          runtime->send_fill_view(target, rez);
        }
        else // Ask our parent to do the send
          parent->send_view_base(target);
        // We've now done the send so record it
        update_remote_instances(target);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    void FillView::send_view_updates(AddressSpaceID target, 
                                     const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      // We only need to send updates for our constituent reduction views
      send_deferred_view_updates(target, update_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillView::handle_send_fill_view(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      LogicalRegion handle;
      derez.deserialize(handle);
      size_t value_size;
      derez.deserialize(value_size);
      void *value = malloc(value_size);
      derez.deserialize(value, value_size);
      
      RegionNode *target_node = runtime->forest->get_node(handle);
      FillView::FillViewValue *fill_value = 
                      new FillView::FillViewValue(value, value_size);
      FillView *new_view = legion_new<FillView>(runtime->forest, did, 
                                  owner_space, runtime->address_space,
                                  target_node, false/*register now*/,
                                  fill_value);
      if (!target_node->register_logical_view(new_view))
      {
        if (new_view->remove_base_resource_ref(REMOTE_DID_REF))
          legion_delete(new_view);
      }
      else
      {
        new_view->register_with_runtime();
        new_view->update_remote_instances(source);
      }
    }

    //--------------------------------------------------------------------------
    LogicalView* FillView::get_subview(const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
      // See if we already have this child
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<ColorPoint,FillView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      RegionTreeNode *child_node = logical_node->get_tree_child(c);
      FillView *child_view = legion_new<FillView>(context, did,
                                                  owner_space, local_space,
                                                  child_node, false/*register*/,
                                                  value, this/*parent*/);
      // Retake the lock and try and add the child, see if someone else added
      // the child in the meantime
      bool free_child_view = false;
      FillView *result = child_view;
      {
        AutoLock v_lock(view_lock);
        std::map<ColorPoint,FillView*>::const_iterator finder = 
                                                          children.find(c);
        if (finder != children.end())
        {
          if (child_view->remove_nested_resource_ref(did))
            free_child_view = true;
          result = finder->second;
        }
        else
        {
          children[c] = child_view;
          // Update the subviews while holding the lock
          for (std::deque<ReductionEpoch>::const_iterator rit = 
                reduction_epochs.begin(); rit != reduction_epochs.end(); rit++)
          {
            const ReductionEpoch &epoch = *rit;
            for (std::set<ReductionView*>::const_iterator it = 
                  epoch.views.begin(); it != epoch.views.end(); it++)
            {
              child_view->update_reduction_views(*it, epoch.valid_fields,
                                                 false/*update parent*/);
            }
          }
          // Register the child node
          child_node->register_logical_view(child_view);
        }
      }
      if (free_child_view)
        legion_delete(child_view);
      return result;
    }

    //--------------------------------------------------------------------------
    void FillView::update_child_reduction_views(ReductionView *view,
                                                const FieldMask &valid_mask,
                                                DeferredView *to_skip/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      std::map<ColorPoint,FillView*> to_handle;
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        to_handle = children;
      }
      std::set<ColorPoint> handled;
      // Keep iterating until we've handled all the children
      while (!to_handle.empty())
      {
        for (std::map<ColorPoint,FillView*>::const_iterator it = 
              to_handle.begin(); it != to_handle.end(); it++)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(handled.find(it->first) == handled.end());
#endif
          handled.insert(it->first);
          if (it->second == to_skip)
            continue;
          it->second->update_reduction_views(view, valid_mask, false/*parent*/);
        }
        to_handle.clear();
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
#ifdef DEBUG_HIGH_LEVEL
        assert(handled.size() <= children.size());
#endif
        if (handled.size() == children.size())
          break;
        // Otherwise figure out what additional children to handle
        for (std::map<ColorPoint,FillView*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if (handled.find(it->first) == handled.end())
            to_handle.insert(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                                         CopyTracker *tracker)
    //--------------------------------------------------------------------------
    {
      LegionMap<Event,FieldMask>::aligned preconditions;
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   copy_mask, info.version_info, preconditions);
      // Compute the precondition sets
      LegionList<EventSet>::aligned precondition_sets;
      RegionTreeNode::compute_event_sets(copy_mask, preconditions,
                                         precondition_sets);
      // Iterate over the precondition sets
      for (LegionList<EventSet>::aligned::iterator pit = 
            precondition_sets.begin(); pit !=
            precondition_sets.end(); pit++)
      {
        EventSet &pre_set = *pit;
        // Build the src and dst fields vectors
        std::vector<Domain::CopySrcDstField> dst_fields;
        dst->copy_to(pre_set.set_mask, dst_fields);
        Event fill_pre = Runtime::merge_events<false>(pre_set.preconditions);
#ifdef LEGION_SPY
        if (!fill_pre.exists())
        {
          UserEvent new_fill_pre = UserEvent::create_user_event();
          new_fill_pre.trigger();
          fill_pre = new_fill_pre;
        }
        LegionSpy::log_event_dependences(pre_set.preconditions, fill_pre);
#endif
        // Issue the fill commands
        Event fill_post;
        if (dst->logical_node->has_component_domains())
        {
          std::set<Event> post_events; 
          Event dom_pre;
          const std::set<Domain> &fill_domains = 
            dst->logical_node->get_component_domains(dom_pre);
          if (dom_pre.exists())
            fill_pre = Runtime::merge_events<false>(fill_pre, dom_pre);
          UniqueID op_id = (info.op == NULL) ? 0 : info.op->get_unique_op_id();
          for (std::set<Domain>::const_iterator it = fill_domains.begin();
                it != fill_domains.end(); it++)
          {
            post_events.insert(context->issue_fill(*it, op_id,
                                                   dst_fields, value->value,
                                                   value->value_size,fill_pre));
          }
          fill_post = Runtime::merge_events<false>(post_events);
        }
        else
        {
          Event dom_pre;
          const Domain &dom = dst->logical_node->get_domain(dom_pre);
          if (dom_pre.exists())
            fill_pre = Runtime::merge_events<false>(fill_pre, dom_pre);
          UniqueID op_id = (info.op == NULL) ? 0 : info.op->get_unique_op_id();
          fill_post = context->issue_fill(dom, op_id, dst_fields,
                                          value->value, value->value_size, 
                                          fill_pre);
        }
#ifdef LEGION_SPY
        if (!fill_post.exists())
        {
          UserEvent new_fill_post = UserEvent::create_user_event();
          new_fill_post.trigger();
          fill_post = new_fill_post;
        }
#endif
        // Now see if there are any reductions to apply
        FieldMask reduce_overlap = reduction_mask & pre_set.set_mask;
        if (!!reduce_overlap)
        {
          // See if we have any reductions to flush
          LegionMap<Event,FieldMask>::aligned reduce_conditions;
          if (fill_post.exists())
            reduce_conditions[fill_post] = pre_set.set_mask;
          flush_reductions(info, dst, reduce_overlap, reduce_conditions);
          // Sort out the post-conditions into different groups 
          LegionList<EventSet>::aligned postcondition_sets;
          RegionTreeNode::compute_event_sets(pre_set.set_mask, 
                                             reduce_conditions,
                                             postcondition_sets);
          // Add each of the different postconditions separately
          for (LegionList<EventSet>::aligned::iterator it = 
               postcondition_sets.begin(); it !=
               postcondition_sets.end(); it++)
          {
            Event reduce_post = Runtime::merge_events<false>(it->preconditions);
            if (reduce_post.exists())
            {
              if (tracker != NULL)
                tracker->add_copy_event(reduce_post);
              dst->add_copy_user(0/*redop*/, reduce_post, info.version_info,
                                 it->set_mask, false/*reading*/);
            }
          }
        }
        else if (fill_post.exists())
        {
          if (tracker != NULL)
            tracker->add_copy_event(fill_post);
          dst->add_copy_user(0/*redop*/, fill_post, info.version_info,
                             pre_set.set_mask, false/*reading*/);
        }
      }
    }
    
    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
             const LegionMap<Event,FieldMask>::aligned &preconditions,
                   LegionMap<Event,FieldMask>::aligned &postconditions,
                                         CopyTracker *tracker)
    //--------------------------------------------------------------------------
    {
      // Do the same thing as above, but no need to add ourselves as user
      // or compute the destination preconditions as they are already included
      LegionList<EventSet>::aligned precondition_sets;
      RegionTreeNode::compute_event_sets(copy_mask, preconditions,
                                         precondition_sets);
      // Iterate over the precondition sets
      for (LegionList<EventSet>::aligned::iterator pit = 
            precondition_sets.begin(); pit !=
            precondition_sets.end(); pit++)
      {
        EventSet &pre_set = *pit;
        // Build the src and dst fields vectors
        std::vector<Domain::CopySrcDstField> dst_fields;
        dst->copy_to(pre_set.set_mask, dst_fields);
        Event fill_pre = Runtime::merge_events<false>(pre_set.preconditions);
#ifdef LEGION_SPY
        if (!fill_pre.exists())
        {
          UserEvent new_fill_pre = UserEvent::create_user_event();
          new_fill_pre.trigger();
          fill_pre = new_fill_pre;
        }
        LegionSpy::log_event_dependences(pre_set.preconditions, fill_pre);
#endif
        // Issue the fill commands
        Event fill_post;
        if (dst->logical_node->has_component_domains())
        {
          std::set<Event> post_events; 
          Event dom_pre;
          const std::set<Domain> &fill_domains = 
            dst->logical_node->get_component_domains(dom_pre);
          if (dom_pre.exists())
            fill_pre = Runtime::merge_events<false>(fill_pre, dom_pre);
          UniqueID op_id = (info.op == NULL) ? 0 : info.op->get_unique_op_id();
          for (std::set<Domain>::const_iterator it = fill_domains.begin();
                it != fill_domains.end(); it++)
          {
            post_events.insert(context->issue_fill(*it, op_id, dst_fields,
                                                   value->value, 
                                                   value->value_size,fill_pre));
          }
          fill_post = Runtime::merge_events<false>(post_events);
        }
        else
        {
          Event dom_pre;
          const Domain &dom = dst->logical_node->get_domain(dom_pre);
          if (dom_pre.exists())
            fill_pre = Runtime::merge_events<false>(fill_pre, dom_pre);
          UniqueID op_id = (info.op == NULL) ? 0 : info.op->get_unique_op_id();
          fill_post = context->issue_fill(dom, op_id, dst_fields,
                                          value->value, value->value_size, 
                                          fill_pre);
        }
#ifdef LEGION_SPY
        if (!fill_post.exists())
        {
          UserEvent new_fill_post = UserEvent::create_user_event();
          new_fill_post.trigger();
          fill_post = new_fill_post;
        }
#endif
        FieldMask reduce_overlap = reduction_mask & pre_set.set_mask;
        if (!!reduce_overlap)
          flush_reductions(info, dst, reduce_overlap, postconditions);
      }
    }

    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies_across(const TraversalInfo &info,
                                                MaterializedView *dst,
                                                FieldID src_field,
                                                FieldID dst_field,
                                                Event precondition,
                                          std::set<Event> &postconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<Domain::CopySrcDstField> dst_fields;   
      dst->copy_field(dst_field, dst_fields);
      // Issue the copy to the low-level runtime and get back the event
      std::set<Event> post_events;
      const std::set<Domain> &overlap_domains = 
        logical_node->get_intersection_domains(dst->logical_node);
      UniqueID op_id = (info.op == NULL) ? 0 : info.op->get_unique_op_id();
      for (std::set<Domain>::const_iterator it = overlap_domains.begin();
            it != overlap_domains.end(); it++)
      {
        post_events.insert(context->issue_fill(*it, op_id, dst_fields,
                                              value->value, value->value_size, 
                                              precondition));
      }
      Event post_event = Runtime::merge_events<false>(post_events); 
      // If we're going to issue a reduction then we can just flush reductions
      // and the precondition will translate naturally
      if (!!reduction_mask)
        flush_reductions_across(info, dst, src_field, dst_field,
                                post_event, postconditions);
      else
        postconditions.insert(post_event);
    }

    //--------------------------------------------------------------------------
    void FillView::find_field_descriptors(Event term_event, 
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx, Operation *op,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions)
    //--------------------------------------------------------------------------
    {
      // We should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool FillView::find_field_descriptors(Event term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          unsigned fid_idx, Operation *op,
                                          Realm::IndexSpace target,
                                          Event target_precondition,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<Event> &preconditions,
                             std::vector<Realm::IndexSpace> &already_handled,
                                       std::set<Event> &already_preconditions)
    //--------------------------------------------------------------------------
    {
      // We should never get here
      assert(false);
      return false;
    }
    
    /////////////////////////////////////////////////////////////
    // ReductionView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(RegionTreeForest *ctx, DistributedID did,
                                 AddressSpaceID own_sp, AddressSpaceID loc_sp,
                                 RegionTreeNode *node, ReductionManager *man,
                                 bool register_now, UniqueID context_uid)
      : InstanceView(ctx, did, own_sp, loc_sp, node, register_now), manager(man)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(manager != NULL);
#endif
      manager->add_nested_resource_ref(did);
      manager->register_logical_top_view(context_uid, this);
      if (!is_owner())
      {
        add_base_resource_ref(REMOTE_DID_REF);
        send_remote_registration();
      }
#ifdef LEGION_GC
      log_garbage.info("GC Reduction View %ld %ld", did, manager->did);
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL, false), manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> functor(this);
        map_over_remote_instances(functor);
      }
      manager->unregister_logical_top_view(this);
      if (manager->remove_nested_resource_ref(did))
      {
        if (manager->is_list_manager())
          legion_delete(manager->as_list_manager());
        else
          legion_delete(manager->as_fold_manager());
      }
      // Remove any initial users as well
      if (!initial_user_events.empty())
      {
        for (std::set<Event>::const_iterator it = initial_user_events.begin();
              it != initial_user_events.end(); it++)
          filter_local_users(*it);
      }
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE) && \
      defined(DEBUG_HIGH_LEVEL)
      assert(reduction_users.empty());
      assert(reading_users.empty());
      assert(outstanding_gc_events.empty());
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView& ReductionView::operator=(const ReductionView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReductionView::perform_reduction(InstanceView *target,
                                          const FieldMask &reduce_mask,
                                          const VersionInfo &version_info,
                                          Operation *op,
                                          CopyTracker *tracker /*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, reduce_mask, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, src_fields);

      LegionMap<Event,FieldMask>::aligned preconditions;
      target->find_copy_preconditions(manager->redop, false/*reading*/, 
                                      reduce_mask, version_info, preconditions);
      this->find_copy_preconditions(manager->redop, true/*reading*/, 
                                    reduce_mask, version_info, preconditions);
      std::set<Event> event_preconds;
      for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
      {
        event_preconds.insert(it->first);
      }
      Event reduce_pre = Runtime::merge_events<false>(event_preconds); 
#ifdef LEGION_SPY
      IndexSpace reduce_index_space;
#endif
      Event reduce_post; 
      if (logical_node->has_component_domains())
      {
        std::set<Event> post_events;
        Event dom_pre;
        const std::set<Domain> &component_domains = 
          logical_node->get_component_domains(dom_pre);
        if (dom_pre.exists())
          reduce_pre = Runtime::merge_events<false>(reduce_pre, dom_pre);
        for (std::set<Domain>::const_iterator it = 
              component_domains.begin(); it != component_domains.end(); it++)
        {
          Event post = manager->issue_reduction(op, src_fields, dst_fields,
                                                *it, reduce_pre, fold, 
                                                true/*precise*/);
          post_events.insert(post);
        }
        reduce_post = Runtime::merge_events<false>(post_events);
#ifdef LEGION_SPY
        reduce_index_space = logical_node->as_region_node()->row_source->handle;
#endif
      }
      else
      {
        Event dom_pre;
        Domain domain = logical_node->get_domain(dom_pre);
        if (dom_pre.exists())
          reduce_pre = Runtime::merge_events<false>(reduce_pre, dom_pre);
        reduce_post = manager->issue_reduction(op, src_fields, dst_fields,
                                               domain, reduce_pre, fold,
                                               true/*precise*/);
#ifdef LEGION_SPY
        reduce_index_space = logical_node->as_region_node()->row_source->handle;
#endif
      }
#ifdef LEGION_SPY
      if (!reduce_post.exists())
      {
        UserEvent new_reduce_post = UserEvent::create_user_event();
        new_reduce_post.trigger();
        reduce_post = new_reduce_post;
      }
#endif
      target->add_copy_user(manager->redop, reduce_post, version_info,
                            reduce_mask, false/*reading*/);
      this->add_copy_user(manager->redop, reduce_post, version_info,
                          reduce_mask, true/*reading*/);
      if (tracker != NULL)
        tracker->add_copy_event(reduce_post);
#ifdef LEGION_SPY
      {
        std::vector<FieldID> fids;
        manager->region_node->column_source->get_field_ids(reduce_mask,
            fids);
        LegionSpy::log_copy_events(manager->get_instance().id,
            target->get_manager()->get_instance().id, true,
            reduce_index_space.get_id(),
            manager->region_node->column_source->handle.id,
            manager->region_node->handle.tree_id, reduce_pre, reduce_post,
            manager->redop, fids);
      }
#endif
    } 

    //--------------------------------------------------------------------------
    Event ReductionView::perform_deferred_reduction(MaterializedView *target,
                                                    const FieldMask &red_mask,
                                                const VersionInfo &version_info,
                                                    const std::set<Event> &pre,
                                         const std::set<Domain> &reduce_domains,
                                                    Event dom_precondition,
                                                    Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, red_mask, dst_fields);
      this->reduce_from(manager->redop, red_mask, src_fields);

      LegionMap<Event,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/,
                              red_mask, version_info, src_pre);
      std::set<Event> preconditions = pre;
      if (dom_precondition.exists())
        preconditions.insert(dom_precondition);
      for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      Event reduce_pre = Runtime::merge_events<false>(preconditions); 
      std::set<Event> post_events;
      for (std::set<Domain>::const_iterator it = reduce_domains.begin();
            it != reduce_domains.end(); it++)
      {
        Event post = manager->issue_reduction(op, src_fields, dst_fields,
                                              *it, reduce_pre, fold,
                                              false/*precise*/);
        post_events.insert(post);
      }
      Event reduce_post = Runtime::merge_events<false>(post_events);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, version_info,
                    red_mask, true/*reading*/);
#ifdef LEGION_SPY
      IndexSpace reduce_index_space =
              target->logical_node->as_region_node()->row_source->handle;
      {
        std::vector<FieldID> fids;
        manager->region_node->column_source->get_field_ids(red_mask, fids);
        LegionSpy::log_copy_events(manager->get_instance().id,
            target->get_manager()->get_instance().id, true,
            reduce_index_space.get_id(),
            manager->region_node->column_source->handle.id,
            manager->region_node->handle.tree_id, reduce_pre, reduce_post,
            manager->redop, fids);
      }
#endif
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    Event ReductionView::perform_deferred_across_reduction(
                              MaterializedView *target, FieldID dst_field, 
                              FieldID src_field, unsigned src_index, 
                              const VersionInfo &version_info,
                              const std::set<Event> &preconds,
                              const std::set<Domain> &reduce_domains,
                              Event dom_precondition, Operation *op)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, PERFORM_REDUCTION_CALL);
#endif
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      const bool fold = false;
      target->copy_field(dst_field, dst_fields);
      FieldMask red_mask; red_mask.set_bit(src_index);
      this->reduce_from(manager->redop, red_mask, src_fields);

      LegionMap<Event,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/,
                              red_mask, version_info, src_pre);
      std::set<Event> preconditions = preconds;
      if (dom_precondition.exists())
        preconditions.insert(dom_precondition);
      for (LegionMap<Event,FieldMask>::aligned::const_iterator it = 
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      Event reduce_pre = Runtime::merge_events<false>(preconditions); 
      std::set<Event> post_events;
      for (std::set<Domain>::const_iterator it = reduce_domains.begin();
            it != reduce_domains.end(); it++)
      {
        Event post = manager->issue_reduction(op, src_fields, dst_fields,
                                              *it, reduce_pre, fold,
                                              false/*precise*/);
        post_events.insert(post);
      }
      Event reduce_post = Runtime::merge_events<false>(post_events);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, version_info,
                    red_mask, true/*reading*/);
#ifdef LEGION_SPY
      IndexSpace reduce_index_space =
              target->logical_node->as_region_node()->row_source->handle;
      {
        std::vector<FieldID> fids;
        manager->region_node->column_source->get_field_ids(red_mask, fids);
        LegionSpy::log_copy_events(manager->get_instance().id,
            target->get_manager()->get_instance().id, true,
            reduce_index_space.get_id(),
            manager->region_node->column_source->handle.id,
            manager->region_node->handle.tree_id, reduce_pre, reduce_post,
            manager->redop, fids);
      }
#endif
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    bool ReductionView::is_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool ReductionView::is_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    MaterializedView* ReductionView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      return NULL;
    }

    //--------------------------------------------------------------------------
    ReductionView* ReductionView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      return const_cast<ReductionView*>(this);
    }

    //--------------------------------------------------------------------------
    PhysicalManager* ReductionView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      return manager;
    }

    //--------------------------------------------------------------------------
    LogicalView* ReductionView::get_subview(const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
      // Right now we don't make sub-views for reductions
      return this;
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_copy_preconditions(ReductionOpID redop,
                                                bool reading,
                                                const FieldMask &copy_mask,
                                                const VersionInfo &version_info,
                             LegionMap<Event,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    {
      Event use_event = manager->get_use_event();
      if (use_event.exists())
      {
        LegionMap<Event,FieldMask>::aligned::iterator finder = 
            preconditions.find(use_event);
        if (finder == preconditions.end())
          preconditions[use_event] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      if (reading)
      {
        // Register dependences on any reducers
        for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
              reduction_users.begin(); rit != reduction_users.end(); rit++)
        {
          const EventUsers &event_users = rit->second;
          if (event_users.single)
          {
            FieldMask overlap = copy_mask & event_users.user_mask;
            if (!overlap)
              continue;
            LegionMap<Event,FieldMask>::aligned::iterator finder = 
              preconditions.find(rit->first);
            if (finder == preconditions.end())
              preconditions[rit->first] = overlap;
            else
              finder->second |= overlap;
          }
          else
          {
            if (!(copy_mask * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                FieldMask overlap = copy_mask & it->second;
                if (!overlap)
                  continue;
                LegionMap<Event,FieldMask>::aligned::iterator finder = 
                  preconditions.find(rit->first);
                if (finder == preconditions.end())
                  preconditions[rit->first] = overlap;
                else
                  finder->second |= overlap;
              }
            }
          }
        }
      }
      else
      {
        // Register dependences on any readers
        for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
              reading_users.begin(); rit != reading_users.end(); rit++)
        {
          const EventUsers &event_users = rit->second;
          if (event_users.single)
          {
            FieldMask overlap = copy_mask & event_users.user_mask;
            if (!overlap)
              continue;
            LegionMap<Event,FieldMask>::aligned::iterator finder = 
              preconditions.find(rit->first);
            if (finder == preconditions.end())
              preconditions[rit->first] = overlap;
            else
              finder->second |= overlap;
          }
          else
          {
            if (!(copy_mask * event_users.user_mask))
            {
              for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                    it = event_users.users.multi_users->begin(); it !=
                    event_users.users.multi_users->end(); it++)
              {
                FieldMask overlap = copy_mask & it->second;
                if (!overlap)
                  continue;
                LegionMap<Event,FieldMask>::aligned::iterator finder = 
                  preconditions.find(rit->first);
                if (finder == preconditions.end())
                  preconditions[rit->first] = overlap;
                else
                  finder->second |= overlap;
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_copy_user(ReductionOpID redop, Event copy_term,
                                      const VersionInfo &version_info,
                                      const FieldMask &mask, bool reading)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_COPY_USER_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      
      // Quick test: only need to do this if copy term exists
      bool issue_collect = false;
      if (copy_term.exists())
      {
        PhysicalUser *user;
        // We don't use field versions for doing interference 
        // tests on reductions so no need to record it
        if (reading)
        {
          RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
          user = legion_new<PhysicalUser>(usage, ColorPoint());
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = legion_new<PhysicalUser>(usage, ColorPoint());
        }
        AutoLock v_lock(view_lock);
        add_physical_user(user, reading, copy_term, mask);
        // Update the reference users
        if (outstanding_gc_events.find(copy_term) ==
            outstanding_gc_events.end())
        {
          outstanding_gc_events.insert(copy_term);
          issue_collect = true;
        }
      }
      // Launch the garbage collection task if necessary
      if (issue_collect)
        defer_collect_user(copy_term);
    }

    //--------------------------------------------------------------------------
    Event ReductionView::add_user(const RegionUsage &usage, Event term_event,
                                  const FieldMask &user_mask, Operation *op,
                                  const VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, ADD_USER_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      if (IS_REDUCE(usage))
        assert(usage.redop == manager->redop);
      else
        assert(IS_READ_ONLY(usage));
#endif
      const bool reading = IS_READ_ONLY(usage);
      std::set<Event> wait_on;
      Event use_event = manager->get_use_event();
      if (use_event.exists())
        wait_on.insert(use_event);
      // Who cares just hold the lock in exlcusive mode, this analysis
      // shouldn't be too expensive for reduction views
      bool issue_collect = false;
      PhysicalUser *new_user;
      // We don't use field versions for doing interference 
      // tests on reductions so no need to record it
      if (reading)
        new_user = legion_new<PhysicalUser>(usage, ColorPoint());
      else
        new_user = legion_new<PhysicalUser>(usage, ColorPoint());
      {
        AutoLock v_lock(view_lock);
        if (!reading)
        {
          // Reducing
          for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
                reading_users.begin(); rit != reading_users.end(); rit++)
          {
            const EventUsers &event_users = rit->second;
            if (event_users.single)
            {
              FieldMask overlap = user_mask & event_users.user_mask;
              if (!overlap)
                continue;
              wait_on.insert(rit->first);
            }
            else
            {
              if (!(user_mask * event_users.user_mask))
              {
                for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                      it = event_users.users.multi_users->begin(); it !=
                      event_users.users.multi_users->end(); it++)
                {
                  FieldMask overlap = user_mask & it->second;
                  if (!overlap)
                    continue;
                  // Once we have one event precondition we are done
                  wait_on.insert(rit->first);
                  break;
                }
              }
            }
          }
          add_physical_user(new_user, false/*reading*/, term_event, user_mask);
        }
        else // We're reading so wait on any reducers
        {
          for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
                reduction_users.begin(); rit != reduction_users.end(); rit++)
          {
            const EventUsers &event_users = rit->second;
            if (event_users.single)
            {
              FieldMask overlap = user_mask & event_users.user_mask;
              if (!overlap)
                continue;
              wait_on.insert(rit->first);
            }
            else
            {
              if (!(user_mask * event_users.user_mask))
              {
                for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                      it = event_users.users.multi_users->begin(); it !=
                      event_users.users.multi_users->end(); it++)
                {
                  FieldMask overlap = user_mask & it->second;
                  if (!overlap)
                    continue;
                  // Once we have one event precondition we are done
                  wait_on.insert(rit->first);
                  break;
                }
              }
            }
          }
          add_physical_user(new_user, true/*reading*/, term_event, user_mask);
        }
        // Only need to do this if we actually have a term event
        if (outstanding_gc_events.find(term_event) ==
            outstanding_gc_events.end())
        {
          outstanding_gc_events.insert(term_event);
          issue_collect = true;
        }
      }
      // Launch the garbage collection task if we need to
      if (issue_collect)
        defer_collect_user(term_event);
      // Return our result
      return Runtime::merge_events<false>(wait_on);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_physical_user(PhysicalUser *user, bool reading,
                                          Event term_event, 
                                          const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Better already be holding the lock
      EventUsers *event_users;
      if (reading)
        event_users = &(reading_users[term_event]);
      else
        event_users = &(reduction_users[term_event]);
      if (event_users->single)
      {
        if (event_users->users.single_user == NULL)
        {
          // make it the entry
          event_users->users.single_user = user;
          event_users->user_mask = user_mask;
        }
        else
        {
          // convert to multi
          LegionMap<PhysicalUser*,FieldMask>::aligned *new_map = 
                           new LegionMap<PhysicalUser*,FieldMask>::aligned();
          (*new_map)[event_users->users.single_user] = event_users->user_mask;
          (*new_map)[user] = user_mask;
          event_users->user_mask |= user_mask;
          event_users->users.multi_users = new_map;
          event_users->single = false;
        }
      }
      else
      {
        // Add it to the set 
        (*event_users->users.multi_users)[user] = user_mask;
        event_users->user_mask |= user_mask;
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::filter_local_users(Event term_event)
    //--------------------------------------------------------------------------
    {
      // Better be holding the lock before calling this
      std::set<Event>::iterator event_finder = 
        outstanding_gc_events.find(term_event);
      if (event_finder != outstanding_gc_events.end())
      {
        LegionMap<Event,EventUsers>::aligned::iterator finder = 
          reduction_users.find(term_event);
        if (finder != reduction_users.end())
        {
          EventUsers &event_users = finder->second;
          if (event_users.single)
          {
            legion_delete(event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator it
                  = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              legion_delete(it->first);
            }
            delete event_users.users.multi_users;
          }
          reduction_users.erase(finder);
        }
        finder = reading_users.find(term_event);
        if (finder != reading_users.end())
        {
          EventUsers &event_users = finder->second;
          if (event_users.single)
          {
            legion_delete(event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              legion_delete(it->first);
            }
            delete event_users.users.multi_users;
          }
          reading_users.erase(finder);
        }
        outstanding_gc_events.erase(event_finder);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_initial_user(Event term_event, 
                                         const RegionUsage &usage,
                                         const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // We don't use field versions for doing interference tests on
      // reductions so there is no need to record it
      PhysicalUser *user = legion_new<PhysicalUser>(usage, ColorPoint()); 
      add_physical_user(user, IS_READ_ONLY(usage), term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }
 
    //--------------------------------------------------------------------------
    bool ReductionView::reduce_to(ReductionOpID redop, 
                                  const FieldMask &reduce_mask,
                              std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_TO_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      // Get the destination fields for this copy
      manager->find_field_offsets(reduce_mask, dst_fields);
      return manager->is_foldable();
    }

    //--------------------------------------------------------------------------
    void ReductionView::reduce_from(ReductionOpID redop,
                                    const FieldMask &reduce_mask,
                              std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_PERF
      PerfTracer tracer(context, REDUCE_FROM_CALL);
#endif
#ifdef DEBUG_HIGH_LEVEL
      assert(redop == manager->redop);
#endif
      manager->find_field_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_to(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_from(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool ReductionView::has_war_dependence(const RegionUsage &usage,
                                           const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    } 

    //--------------------------------------------------------------------------
    void ReductionView::notify_active(void)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_inactive(void)
    //--------------------------------------------------------------------------
    {
      // No need to check for deletion of the manager since
      // we know that we also hold a resource reference
      manager->remove_nested_gc_ref(did);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_valid_ref(did);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_valid_ref(did);
    }

    //--------------------------------------------------------------------------
    void ReductionView::collect_users(const std::set<Event> &term_events)
    //--------------------------------------------------------------------------
    {
      // Do not do this if we are in LegionSpy so we can see 
      // all of the dependences
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      AutoLock v_lock(view_lock);
      for (std::set<Event>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
      {
        filter_local_users(*it); 
      }
#endif
    }

    //--------------------------------------------------------------------------
    DistributedID ReductionView::send_view_base(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (!has_remote_instance(target))
      {
        // Send the physical manager first
        DistributedID manager_did = manager->send_manager(target);
#ifdef DEBUG_HIGH_LEVEL
        assert(logical_node->is_region()); // Always regions at the top
#endif
        // Don't take the lock, it's alright to have duplicate sends
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(manager_did);
          rez.serialize(logical_node->as_region_node()->handle);
          rez.serialize(owner_space);
          // We only store this UID in the manager, so look it up here
          UniqueID context_uid = manager->find_context_uid(this);
          rez.serialize(context_uid);
        }
        runtime->send_reduction_view(target, rez);
        update_remote_instances(target);
      }
      return did;
    }

    //--------------------------------------------------------------------------
    void ReductionView::send_view_updates(AddressSpaceID target,
                                          const FieldMask &update_mask)
    //--------------------------------------------------------------------------
    {
      Serializer reduction_rez, reading_rez;
      std::deque<PhysicalUser*> red_users, read_users;
      unsigned reduction_events = 0, reading_events = 0;
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
              reduction_users.begin(); rit != reduction_users.end(); rit++)
        {
          FieldMask overlap = rit->second.user_mask & update_mask;
          if (!overlap)
            continue;
          reduction_events++;
          const EventUsers &event_users = rit->second;
          reduction_rez.serialize(rit->first);
          if (event_users.single)
          {
            reduction_rez.serialize<size_t>(1);
            reduction_rez.serialize(overlap);
            red_users.push_back(event_users.users.single_user);
          }
          else
          {
            reduction_rez.serialize<size_t>(
                                      event_users.users.multi_users->size());
            // Just send them all
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                  it = event_users.users.multi_users->begin(); it != 
                  event_users.users.multi_users->end(); it++)
            {
              reduction_rez.serialize(it->second);
              red_users.push_back(it->first);
            }
          }
        }
        for (LegionMap<Event,EventUsers>::aligned::const_iterator rit = 
              reading_users.begin(); rit != reading_users.end(); rit++)
        {
          FieldMask overlap = rit->second.user_mask & update_mask;
          if (!overlap)
            continue;
          reading_events++;
          const EventUsers &event_users = rit->second;
          reading_rez.serialize(rit->first);
          if (event_users.single)
          {
            reading_rez.serialize<size_t>(1);
            reading_rez.serialize(overlap);
            read_users.push_back(event_users.users.single_user);
          }
          else
          {
            reading_rez.serialize<size_t>(
                                      event_users.users.multi_users->size());
            // Just send them all
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                  it = event_users.users.multi_users->begin(); it != 
                  event_users.users.multi_users->end(); it++)
            {
              reading_rez.serialize(it->second);
              read_users.push_back(it->first);
            }
          }
        }
      }
      // We've released the lock, so reassemble the message
      Serializer rez;
      {
        RezCheck z(rez);
#ifdef DEBUG_HIGH_LEVEL
        assert(logical_node->is_region());
#endif
        rez.serialize(logical_node->as_region_node()->handle);
        rez.serialize(did);
        rez.serialize<size_t>(red_users.size());
        for (std::deque<PhysicalUser*>::const_iterator it = 
              red_users.begin(); it != red_users.end(); it++)
        {
          (*it)->pack_user(rez);
        }
        rez.serialize<size_t>(read_users.size());
        for (std::deque<PhysicalUser*>::const_iterator it = 
              read_users.begin(); it != read_users.end(); it++)
        {
          (*it)->pack_user(rez);
        }
        rez.serialize(reduction_events);
        size_t reduction_size = reduction_rez.get_used_bytes(); 
        rez.serialize(reduction_rez.get_buffer(), reduction_size);
        rez.serialize(reading_events);
        size_t reading_size = reading_rez.get_used_bytes();
        rez.serialize(reading_rez.get_buffer(), reading_size);
      }
      runtime->send_reduction_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_update(Deserializer &derez, 
                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      size_t num_reduction_users;
      derez.deserialize(num_reduction_users);
      std::vector<PhysicalUser*> red_users(num_reduction_users);
      FieldSpaceNode *field_node = logical_node->column_source;
      for (unsigned idx = 0; idx < num_reduction_users; idx++)
        red_users[idx] = PhysicalUser::unpack_user(derez, field_node, 
                                                   source, true/*add ref*/);
      size_t num_reading_users;
      derez.deserialize(num_reading_users);
      std::deque<PhysicalUser*> read_users(num_reading_users);
      for (unsigned idx = 0; idx < num_reading_users; idx++)
        read_users[idx] = PhysicalUser::unpack_user(derez, field_node, 
                                                    source, true/*add ref*/);
      std::deque<Event> collect_events;
      {
        unsigned reduction_index = 0, reading_index = 0;
        unsigned num_reduction_events;
        derez.deserialize(num_reduction_events);
        AutoLock v_lock(view_lock);
        for (unsigned idx = 0; idx < num_reduction_events; idx++)
        {
          Event red_event;
          derez.deserialize(red_event);
          size_t num_users;
          derez.deserialize(num_users);
          for (unsigned idx2 = 0; idx2 < num_users; idx2++)
          {
            FieldMask user_mask;
            derez.deserialize(user_mask);
            field_node->transform_field_mask(user_mask, source);
            add_physical_user(red_users[reduction_index++], false/*reading*/,
                              red_event, user_mask);
          }
          if (outstanding_gc_events.find(red_event) == 
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(red_event);
            collect_events.push_back(red_event);
          }
        }
        unsigned num_reading_events;
        derez.deserialize(num_reading_events);
        for (unsigned idx = 0; idx < num_reading_events; idx++)
        {
          Event read_event;
          derez.deserialize(read_event);
          size_t num_users;
          derez.deserialize(num_users);
          for (unsigned idx2 = 0; idx2 < num_users; idx2++)
          {
            FieldMask user_mask;
            derez.deserialize(user_mask);
            field_node->transform_field_mask(user_mask, source);
            add_physical_user(read_users[reading_index++], true/*reading*/,
                              read_event, user_mask);
          }
          if (outstanding_gc_events.find(read_event) ==
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(read_event);
            collect_events.push_back(read_event);
          }
        }
      }
      if (!collect_events.empty())
      {
        for (std::deque<Event>::const_iterator it = collect_events.begin();
              it != collect_events.end(); it++)
        {
          defer_collect_user(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    Memory ReductionView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->memory;
    }

    //--------------------------------------------------------------------------
    ReductionOpID ReductionView::get_redop(void) const
    //--------------------------------------------------------------------------
    {
      return manager->redop;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_send_reduction_view(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez); 
      DistributedID did;
      derez.deserialize(did);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      LogicalRegion handle;
      derez.deserialize(handle);
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      UniqueID context_uid;
      derez.deserialize(context_uid);

      RegionNode *target_node = runtime->forest->get_node(handle);
      PhysicalManager *phy_man = target_node->find_manager(manager_did);
#ifdef DEBUG_HIGH_LEVEL
      assert(phy_man->is_reduction_manager());
#endif
      ReductionManager *red_manager = phy_man->as_reduction_manager();

      ReductionView *new_view = legion_new<ReductionView>(runtime->forest,
                                   did, owner_space, runtime->address_space,
                                   target_node, red_manager,
                                   false/*don't register yet*/, context_uid);
      if (!target_node->register_logical_view(new_view))
      {
        if (new_view->remove_base_resource_ref(REMOTE_DID_REF))
          legion_delete(new_view);
      }
      else
      {
        new_view->register_with_runtime();
        new_view->update_remote_instances(source);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_send_update(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionTreeNode *node = runtime->forest->get_node(handle);
      DistributedID did;
      derez.deserialize(did);
      LogicalView *view = node->find_view(did);
#ifdef DEBUG_HIGH_LEVEL
      assert(view->is_instance_view());
      assert(view->as_instance_view()->is_reduction_view());
#endif
      ReductionView *red_view = view->as_instance_view()->as_reduction_view();
      red_view->process_update(derez, source);
    }

  }; // namespace Internal 
}; // namespace Legion

