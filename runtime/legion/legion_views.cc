/* Copyright 2018 Stanford University, NVIDIA Corporation
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
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_analysis.h"
#include "legion/legion_context.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // LogicalView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalView::LogicalView(RegionTreeForest *ctx, DistributedID did,
                             AddressSpaceID own_addr,
                             RegionTreeNode *node, bool register_now)
      : DistributedCollectable(ctx->runtime, did, own_addr, register_now), 
        context(ctx), logical_node(node)
    //--------------------------------------------------------------------------
    {
      logical_node->add_base_gc_ref(LOGICAL_VIEW_REF);
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
      if (logical_node->remove_base_gc_ref(LOGICAL_VIEW_REF))
        delete logical_node;
    }

    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_view_request(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      LogicalView *view = dynamic_cast<LogicalView*>(dc);
      assert(view != NULL);
#else
      LogicalView *view = static_cast<LogicalView*>(dc);
#endif
      view->send_view(source);
    }

    //--------------------------------------------------------------------------
    void LogicalView::defer_collect_user(ApEvent term_event,
                                         ReferenceMutator *mutator) 
    //--------------------------------------------------------------------------
    {
      // The runtime will add the gc reference to this view when necessary
      runtime->defer_collect_user(this, term_event, mutator);
    }
 
    //--------------------------------------------------------------------------
    /*static*/ void LogicalView::handle_deferred_collect(LogicalView *view,
                                           const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      view->collect_users(term_events);
      // Then remove the gc reference on the object
      if (view->remove_base_gc_ref(PENDING_GC_REF))
        delete view;
    }

    /////////////////////////////////////////////////////////////
    // InstanceView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp,
                               AddressSpaceID log_own, RegionTreeNode *node, 
                               UniqueID own_ctx, bool register_now)
      : LogicalView(ctx, did, owner_sp, node, register_now), 
        owner_context(own_ctx), logical_owner(log_own)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_update_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_update_request(source, done_event, derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_update_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_update_response(derez, done_event, runtime->forest);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_remote_update(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_remote_update(derez, source, runtime->forest);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_remote_invalidate(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask invalid_mask;
      derez.deserialize(invalid_mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_remote_invalidate(invalid_mask, done_event);
    }

    /////////////////////////////////////////////////////////////
    // MaterializedView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
                               RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID own_addr,
                               AddressSpaceID log_own, RegionTreeNode *node, 
                               InstanceManager *man, MaterializedView *par, 
                               UniqueID own_ctx, bool register_now)
      : InstanceView(ctx, encode_materialized_did(did, par == NULL), own_addr, 
                     log_own, node, own_ctx, register_now), 
        manager(man), parent(par), 
        disjoint_children(node->are_all_children_disjoint())
    //--------------------------------------------------------------------------
    {
      // Otherwise the instance lock will get filled in when we are unpacked
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      logical_node->register_instance_view(manager, owner_context, this);
      // If we are either not a parent or we are a remote parent add 
      // a resource reference to avoid being collected
      if (parent != NULL)
        add_nested_resource_ref(parent->did);
      else 
        manager->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Materialized View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, 
          LEGION_DISTRIBUTED_ID_FILTER(manager->did)); 
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL, 0, false),
        manager(NULL), parent(NULL), disjoint_children(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    {
      // Always unregister ourselves with the region tree node
      logical_node->unregister_instance_view(manager, owner_context);
      for (std::map<LegionColor,MaterializedView*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->second->remove_nested_resource_ref(did))
          delete it->second;
      }
      if ((parent == NULL) && manager->remove_nested_resource_ref(did))
        delete manager;
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
        for (std::set<ApEvent>::const_iterator it = initial_user_events.begin();
              it != initial_user_events.end(); it++)
          filter_local_users(*it);
      }
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE) && \
      defined(DEBUG_LEGION)
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
    void MaterializedView::add_remote_child(MaterializedView *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager == child->manager);
#endif
      LegionColor c = child->logical_node->get_color();
      AutoLock v_lock(view_lock);
      children[c] = child;
    }

    //--------------------------------------------------------------------------
    Memory MaterializedView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->get_memory();
    }

    //--------------------------------------------------------------------------
    const FieldMask& MaterializedView::get_physical_mask(void) const
    //--------------------------------------------------------------------------
    {
      return manager->layout->allocated_fields;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::has_space(const FieldMask &space_mask) const
    //--------------------------------------------------------------------------
    {
      return !(space_mask - manager->layout->allocated_fields);
    }

    //--------------------------------------------------------------------------
    LogicalView* MaterializedView::get_subview(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      return get_materialized_subview(c);
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_subview(
                                                            const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // This is the common case we should already have it
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<LegionColor,MaterializedView*>::const_iterator finder = 
                                                            children.find(c);
        if (finder != children.end())
          return finder->second;
      }
      // If we don't have it, we have to make it
      if (is_owner())
      {
        RegionTreeNode *child_node = logical_node->get_tree_child(c);
        // Allocate the DID eagerly
        DistributedID child_did = 
          context->runtime->get_available_distributed_id();
        bool free_child_did = false;
        MaterializedView *child_view = NULL;
        {
          // Retake the lock and see if we lost the race
          AutoLock v_lock(view_lock);
          std::map<LegionColor,MaterializedView*>::const_iterator finder = 
                                                              children.find(c);
          if (finder != children.end())
          {
            child_view = finder->second;
            free_child_did = true;
          }
          else
          {
            // Otherwise we get to make it
            child_view = new MaterializedView(context, child_did, 
                                              owner_space, logical_owner, 
                                              child_node, manager, this, 
                                              owner_context, true/*reg now*/);
            children[c] = child_view;
          }
          if (free_child_did)
            context->runtime->recycle_distributed_id(child_did,
                                                     RtEvent::NO_RT_EVENT);
          return child_view;
        }
      }
      else
      {
        // Find the distributed ID for this child view
        volatile DistributedID child_did;
        RtUserEvent wait_on = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(c);
          rez.serialize(&child_did);
          rez.serialize(wait_on);
        }
        runtime->send_subview_did_request(owner_space, rez); 
        wait_on.wait();
        RtEvent ready;
        LogicalView *child_view = 
          context->runtime->find_or_request_logical_view(child_did, ready);
        if (ready.exists())
          ready.wait();
#ifdef DEBUG_LEGION
        assert(child_view->is_materialized_view());
#endif
        return child_view->as_materialized_view();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_subview_did_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID parent_did;
      derez.deserialize(parent_did);
      LegionColor color;
      derez.deserialize(color);
      DistributedID *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = 
        runtime->find_distributed_collectable(parent_did);
#ifdef DEBUG_LEGION
      MaterializedView *parent_view = dynamic_cast<MaterializedView*>(dc);
      assert(parent_view != NULL);
#else
      MaterializedView *parent_view = static_cast<MaterializedView*>(dc);
#endif
      MaterializedView *child_view = 
        parent_view->get_materialized_subview(color);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(child_view->did);
        rez.serialize(target);
        rez.serialize(to_trigger);
      }
      runtime->send_subview_did_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_subview_did_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID result;
      derez.deserialize(result);
      DistributedID *target;
      derez.deserialize(target);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      (*target) = result;
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    MaterializedView* MaterializedView::get_materialized_parent_view(void) const
    //--------------------------------------------------------------------------
    {
      return parent;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_field(FieldID fid,
                                      std::vector<CopySrcDstField> &copy_fields)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> local_fields(1,fid);
      manager->compute_copy_offsets(local_fields, copy_fields); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_to(const FieldMask &copy_mask,
                                   std::vector<CopySrcDstField> &dst_fields,
                                   CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      if (across_helper == NULL)
        manager->compute_copy_offsets(copy_mask, dst_fields);
      else
        across_helper->compute_across_offsets(copy_mask, dst_fields);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_from(const FieldMask &copy_mask,
                                     std::vector<CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::reduce_to(ReductionOpID redop, 
                                     const FieldMask &copy_mask,
                                     std::vector<CopySrcDstField> &dst_fields,
                                     CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      if (across_helper == NULL)
        manager->compute_copy_offsets(copy_mask, dst_fields);
      else
        across_helper->compute_across_offsets(copy_mask, dst_fields);
      return false; // not a fold
    }

    //--------------------------------------------------------------------------
    void MaterializedView::reduce_from(ReductionOpID redop,
                                       const FieldMask &reduce_mask, 
                                       std::vector<CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      manager->compute_copy_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::accumulate_events(std::set<ApEvent> &all_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      all_events.insert(outstanding_gc_events.begin(),
                        outstanding_gc_events.end());
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions(ReductionOpID redop, 
                                                   bool reading, 
                                                   bool single_copy,
                                                   bool restrict_out,
                                                   const FieldMask &copy_mask,
                                                   VersionTracker *versions,
                                                   const UniqueID creator_op_id,
                                                   const unsigned index,
                                                   const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                             std::set<RtEvent> &applied_events, bool can_filter)
    //--------------------------------------------------------------------------
    {
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
      {
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          preconditions.find(start_use_event);
        if (finder == preconditions.end())
          preconditions[start_use_event] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      RegionNode *origin_node = logical_node->is_region() ? 
        logical_node->as_region_node() : 
        logical_node->as_partition_node()->parent;
      // If we can filter we can do the normal case, otherwise
      // we do the above case where we don't filter
      if (can_filter)
        find_local_copy_preconditions(redop, reading, single_copy, restrict_out,
                                      copy_mask, INVALID_COLOR, origin_node, 
                                      versions, creator_op_id, index, source, 
                                      preconditions, applied_events);
      else
        find_local_copy_preconditions_above(redop, reading, single_copy, 
                                      restrict_out, copy_mask, INVALID_COLOR, 
                                      origin_node, versions,creator_op_id,index,
                                      source, preconditions, applied_events,
                                      false/*actually above*/);
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, single_copy,
                   restrict_out, copy_mask, local_point, origin_node, versions, 
                   creator_op_id, index, source, preconditions, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(ReductionOpID redop,
                                                         bool reading,
                                                         bool single_copy,
                                                         bool restrict_out,
                                                     const FieldMask &copy_mask,
                                                  const LegionColor child_color,
                                                  RegionNode *origin_node,
                                                  VersionTracker *versions,
                                                  const UniqueID creator_op_id,
                                                  const unsigned index,
                                                  const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      find_local_copy_preconditions_above(redop, reading, single_copy, 
                  restrict_out, copy_mask, child_color, origin_node, versions, 
                  creator_op_id, index, source, preconditions, applied_events);
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, single_copy, 
                  restrict_out, copy_mask, local_point, origin_node, versions, 
                  creator_op_id, index, source, preconditions, applied_events);
      }
    }
    
    //--------------------------------------------------------------------------
    void MaterializedView::find_local_copy_preconditions(ReductionOpID redop,
                                                         bool reading,
                                                         bool single_copy,
                                                         bool restrict_out,
                                                     const FieldMask &copy_mask,
                                                  const LegionColor child_color,
                                                  RegionNode *origin_node,
                                                  VersionTracker *versions,
                                                  const UniqueID creator_op_id,
                                                  const unsigned index,
                                                  const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
      // If we are not the logical owner, we need to see if we are up to date 
      if (!is_logical_owner())
      {
        // We are also reading if we are doing a reductions
        perform_remote_valid_check(copy_mask, versions,reading || (redop != 0));
      }
      FieldMask filter_mask;
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users, 
                                           filter_previous_users;
      LegionMap<VersionID,FieldMask>::aligned advance_versions, add_versions;
      if (reading)
      {
        RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
        FieldMask observed, non_dominated;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_current_preconditions<true/*track*/>(copy_mask, usage, child_color,
                               origin_node, creator_op_id, index, preconditions,
                               dead_events, filter_current_users,
                               observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      origin_node, creator_op_id, index, 
                                      preconditions, dead_events);
      }
      else
      {
        RegionUsage usage((redop > 0) ? REDUCE : WRITE_DISCARD,EXCLUSIVE,redop);
        FieldMask observed, non_dominated, write_skip_mask;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Find any version updates as well our write skip mask
        find_copy_version_updates(copy_mask, versions, write_skip_mask, 
            filter_mask, advance_versions, add_versions, redop > 0,
            restrict_out, true/*base*/);
        // Can only do the write-skip optimization if this is a single copy
        if (single_copy && !!write_skip_mask)
        {
          // If we have a write skip mask we know we won't interfere with
          // any users in the list of current users so we can skip them
          const FieldMask current_mask = copy_mask - write_skip_mask;
          if (!!current_mask)
            find_current_preconditions<true/*track*/>(current_mask, usage, 
                                       child_color, origin_node, creator_op_id,
                                       index, preconditions, dead_events, 
                                       filter_current_users,
                                       observed, non_dominated);
        }
        else // the normal case with no write-skip
          find_current_preconditions<true/*track*/>(copy_mask, usage, 
                                     child_color, origin_node, creator_op_id,
                                     index, preconditions, dead_events, 
                                     filter_current_users, 
                                     observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      origin_node, creator_op_id, index, 
                                      preconditions, dead_events);
      }
      if (!dead_events.empty() || 
          !filter_previous_users.empty() || !filter_current_users.empty() ||
          !advance_versions.empty() || !add_versions.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
        if (!filter_previous_users.empty())
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                filter_previous_users.begin(); it != 
                filter_previous_users.end(); it++)
            filter_previous_user(it->first, it->second);
        if (!filter_current_users.empty())
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                filter_current_users.begin(); it !=
                filter_current_users.end(); it++)
            filter_current_user(it->first, it->second);
        if (!advance_versions.empty() || !add_versions.empty())
          apply_version_updates(filter_mask, advance_versions, 
                                add_versions, source, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_local_copy_preconditions_above(
                                                  ReductionOpID redop, 
                                                  bool reading,
                                                  bool single_copy,
                                                  bool restrict_out,
                                                  const FieldMask &copy_mask,
                                                  const LegionColor child_color,
                                                  RegionNode *origin_node,
                                                  VersionTracker *versions,
                                                  const UniqueID creator_op_id,
                                                  const unsigned index,
                                                  const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                                  const bool actually_above)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
      // If we are not the logical owner and we're not actually already above
      // the base level, then we need to see if we are up to date 
      // Otherwise we did this check at the base level and it went all
      // the way up the tree so we are already good
      if (!is_logical_owner() && !actually_above)
      {
        // We are also reading if we are doing reductions
        perform_remote_valid_check(copy_mask, versions,reading || (redop != 0));
      }
      FieldMask filter_mask;
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users; 
      LegionMap<VersionID,FieldMask>::aligned advance_versions, add_versions;
      if (reading)
      {
        RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
        FieldMask observed, non_dominated;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_current_preconditions<false/*track*/>(copy_mask, usage, 
                                   child_color, origin_node, creator_op_id, 
                                   index, preconditions, dead_events, 
                                   filter_current_users,observed,non_dominated);
        // No domination above
        find_previous_preconditions(copy_mask, usage, child_color,
                                    origin_node, creator_op_id, index, 
                                    preconditions, dead_events);
      }
      else
      {
        RegionUsage usage((redop > 0) ? REDUCE : WRITE_DISCARD,EXCLUSIVE,redop);
        FieldMask observed, non_dominated, write_skip_mask;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Find any version updates as well our write skip mask
        find_copy_version_updates(copy_mask, versions, write_skip_mask, 
            filter_mask, advance_versions, add_versions, redop > 0,
            restrict_out, false/*base*/);
        // Can only do the write skip optimization if this is a single copy
        if (single_copy && !!write_skip_mask)
        {
          // If we have a write skip mask we know we won't interfere with
          // any users in the list of current users so we can skip them
          const FieldMask current_mask = copy_mask - write_skip_mask;
          if (!!current_mask)
            find_current_preconditions<false/*track*/>(current_mask, usage, 
                                       child_color, origin_node, creator_op_id,
                                       index, preconditions, dead_events, 
                                       filter_current_users,
                                       observed, non_dominated);
        }
        else // the normal case with no write-skip
          find_current_preconditions<false/*track*/>(copy_mask, usage, 
                                     child_color, origin_node, creator_op_id,
                                     index, preconditions, dead_events, 
                                     filter_current_users, 
                                     observed, non_dominated);
        // No domination above
        find_previous_preconditions(copy_mask, usage, child_color,
                                    origin_node, creator_op_id, index, 
                                    preconditions, dead_events);
      }
#ifdef DEBUG_LEGION
      assert(filter_current_users.empty());
#endif
      if (!dead_events.empty() || 
          !advance_versions.empty() || !add_versions.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
        if (!advance_versions.empty() || !add_versions.empty())
          apply_version_updates(filter_mask, advance_versions, 
                                add_versions, source, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                         VersionTracker *versions,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const FieldMask &copy_mask, 
                                         bool reading, bool restrict_out,
                                         const AddressSpaceID source,
                                         std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
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
      RegionNode *origin_node = logical_node->is_region() ? 
        logical_node->as_region_node() : 
        logical_node->as_partition_node()->parent;
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color,
                               origin_node, versions, creator_op_id, index,
                               restrict_out, copy_mask, source, applied_events);
      }
      add_local_copy_user(usage, copy_term, true/*base*/, restrict_out,
          INVALID_COLOR, origin_node, versions, creator_op_id, index, 
          copy_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user_above(const RegionUsage &usage, 
                                               ApEvent copy_term, 
                                               const LegionColor child_color,
                                               RegionNode *origin_node,
                                               VersionTracker *versions,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const bool restrict_out,
                                               const FieldMask &copy_mask,
                                               const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color, origin_node,
                                  versions, creator_op_id, index, restrict_out, 
                                  copy_mask, source, applied_events);
      }
      add_local_copy_user(usage, copy_term, false/*base*/, restrict_out,
                      child_color, origin_node, versions, creator_op_id, 
                      index, copy_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_local_copy_user(const RegionUsage &usage, 
                                               ApEvent copy_term,
                                               bool base_user,bool restrict_out,
                                               const LegionColor child_color,
                                               RegionNode *origin_node,
                                               VersionTracker *versions,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const FieldMask &copy_mask,
                                               const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // If we are not the owner we have to send the user back to the owner
        RtUserEvent remote_update_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_update_event);
          rez.serialize<bool>(true); // is copy
          rez.serialize<bool>(restrict_out);
          rez.serialize(usage);
          rez.serialize(copy_mask);
          rez.serialize(child_color);
          rez.serialize(origin_node->handle);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize(copy_term);
          // Figure out which version infos we need
          LegionMap<VersionID,FieldMask>::aligned needed_versions;
          FieldVersions field_versions;
          // We don't need to worry about split fields here, that
          // will be taken care of on the handler side, just pack
          // up the version infos as they are currently stored
          versions->get_field_versions(logical_node, false/*split prev*/,
                                       copy_mask, field_versions);
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                field_versions.begin(); it != field_versions.end(); it++)
          {
            FieldMask overlap = it->second & copy_mask;
            if (!overlap)
              continue;
            needed_versions[it->first] = overlap;
          }
          rez.serialize<size_t>(needed_versions.size());
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                needed_versions.begin(); it != needed_versions.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          FieldMask local_split;
          versions->get_split_mask(logical_node, copy_mask, local_split);
          rez.serialize(local_split);
          rez.serialize<bool>(base_user);
        }
        runtime->send_view_remote_update(logical_owner, rez);
        // Tell the operation it has to wait for this event
        // to trigger before it can be considered mapped
        applied_events.insert(remote_update_event);
      }
      PhysicalUser *user = new PhysicalUser(usage, child_color, 
                                    creator_op_id, index, origin_node);
      user->add_reference();
      bool issue_collect = false;
      {
        AutoLock v_lock(view_lock);
        add_current_user(user, copy_term, copy_mask); 
        if (base_user)
          issue_collect = (outstanding_gc_events.find(copy_term) ==
                            outstanding_gc_events.end());
        outstanding_gc_events.insert(copy_term);
        // See if we need to check for read only invalidates
        // Don't worry about writing copies, their invalidations
        // will be sent if they update the version number
        if (!valid_remote_instances.empty() && IS_READ_ONLY(usage))
          perform_read_invalidations(copy_mask, versions, 
                                     source, applied_events);
      }
      if (issue_collect)
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(copy_term, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::find_user_precondition(
                          const RegionUsage &usage, ApEvent term_event,
                          const FieldMask &user_mask, Operation *op,
                          const unsigned index, VersionTracker *versions,
                          std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> wait_on_events;
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      UniqueID op_id = op->get_unique_op_id();
      RegionNode *origin_node = logical_node->is_region() ? 
        logical_node->as_region_node() : 
        logical_node->as_partition_node()->parent;
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, INVALID_COLOR, 
          origin_node, versions, op_id, index, user_mask, 
          wait_on_events, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
                              origin_node, versions, op_id, index, user_mask, 
                              wait_on_events, applied_events);
      }
      return Runtime::merge_events(wait_on_events); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_user_preconditions_above(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                RegionNode *origin_node,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Do the precondition analysis on the way up
      find_local_user_preconditions_above(usage, term_event, child_color, 
                          origin_node, versions, op_id, index, user_mask, 
                          preconditions, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
                              origin_node, versions, op_id, index, user_mask, 
                              preconditions, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_local_user_preconditions(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                RegionNode *origin_node,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
      // If we are not the logical owner, we need to see if we are up to date 
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        assert(!IS_REDUCE(usage)); // no user reductions currently, might change
#endif
        // Only writing if we are overwriting, otherwise we are also reading
        perform_remote_valid_check(user_mask, versions, 
                                   !HAS_WRITE_DISCARD(usage));
      }
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users, 
                                           filter_previous_users;
      if (IS_READ_ONLY(usage))
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<true/*track*/>(user_mask, usage, child_color,
                                   origin_node, term_event, op_id, index, 
                                   preconditions, dead_events, 
                                   filter_current_users,observed,non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      origin_node, term_event, op_id, index,
                                      preconditions, dead_events);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<true/*track*/>(user_mask, usage, child_color,
                                   origin_node, term_event, op_id, index, 
                                   preconditions, dead_events, 
                                   filter_current_users,observed,non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      origin_node, term_event, op_id, index,
                                      preconditions, dead_events);
      }
      if (!dead_events.empty() || 
          !filter_previous_users.empty() || !filter_current_users.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
        if (!filter_previous_users.empty())
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                filter_previous_users.begin(); it != 
                filter_previous_users.end(); it++)
            filter_previous_user(it->first, it->second);
        if (!filter_current_users.empty())
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                filter_current_users.begin(); it !=
                filter_current_users.end(); it++)
            filter_current_user(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_local_user_preconditions_above(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                RegionNode *origin_node,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                                const bool actually_above)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
      // If we are not the logical owner and we are not actually above the
      // base level, then we need to see if we are up to date 
      // If we are actually above then we already did this check at the
      // base level and went all the way up the tree so there is no need
      // to do it again here
      if (!is_logical_owner() && !actually_above)
      {
#ifdef DEBUG_LEGION
        assert(!IS_REDUCE(usage)); // no reductions for now, might change
#endif
        // We are reading if we are not overwriting
        perform_remote_valid_check(user_mask, versions, 
                                   !HAS_WRITE_DISCARD(usage));
      }
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users;
      if (IS_READ_ONLY(usage))
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<false/*track*/>(user_mask, usage, 
                                   child_color, origin_node,
                                   term_event, op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated);
        // No domination above
        find_previous_preconditions(user_mask, usage, child_color, 
                                    origin_node, term_event, op_id, index,
                                    preconditions, dead_events);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<false/*track*/>(user_mask, usage, 
                                   child_color, origin_node,
                                   term_event, op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated);
        // No domination above
        find_previous_preconditions(user_mask, usage, child_color, 
                                    origin_node, term_event, op_id, index,
                                    preconditions, dead_events);
      }
#ifdef DEBUG_LEGION
      assert(filter_current_users.empty());
#endif
      if (!dead_events.empty())
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user(const RegionUsage &usage,ApEvent term_event,
                                    const FieldMask &user_mask, Operation *op,
                                    const unsigned index, AddressSpaceID source,
                                    VersionTracker *versions,
                                    std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      UniqueID op_id = op->get_unique_op_id();
      bool need_version_update = false;
      if (IS_WRITE(usage))
      {
        FieldVersions advance_versions;
        versions->get_advance_versions(logical_node, true/*base*/,
                                       user_mask, advance_versions);
        need_version_update = update_version_numbers(user_mask,advance_versions,
                                                     source, applied_events);
      }
      RegionNode *origin_node = logical_node->is_region() ? 
        logical_node->as_region_node() : 
        logical_node->as_partition_node()->parent;
      // Go up the tree if necessary 
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color, origin_node,
            versions, op_id, index, user_mask, need_version_update, 
            source, applied_events);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                         INVALID_COLOR, origin_node, true/*base*/, versions,
                         op_id, index, user_mask, source, applied_events);
      // Launch the garbage collection task, if it doesn't exist
      // then the user wasn't registered anyway, see add_local_user
      if (issue_collect)
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
      }
      if (IS_ATOMIC(usage))
        find_atomic_reservations(user_mask, op, IS_WRITE(usage));
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above(const RegionUsage &usage,
                                          ApEvent term_event,
                                          const LegionColor child_color,
                                          RegionNode *origin_node,
                                          VersionTracker *versions,
                                          const UniqueID op_id,
                                          const unsigned index,
                                          const FieldMask &user_mask,
                                          const bool need_version_update,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      bool need_update_above = false;
      if (need_version_update)
      {
        FieldVersions advance_versions;
        versions->get_advance_versions(logical_node, false/*base*/,
                                       user_mask, advance_versions);
        need_update_above = update_version_numbers(user_mask, advance_versions,
                                                   source, applied_events);
      }
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color, origin_node,
            versions, op_id, index, user_mask, need_update_above, 
            source, applied_events);
      }
      add_local_user(usage, term_event, child_color, origin_node, false/*base*/,
                     versions, op_id, index, user_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_local_user(const RegionUsage &usage,
                                          ApEvent term_event,
                                          const LegionColor child_color,
                                          RegionNode *origin_node,
                                          const bool base_user,
                                          VersionTracker *versions,
                                          const UniqueID op_id,
                                          const unsigned index,
                                          const FieldMask &user_mask,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (!term_event.exists())
        return false;
      if (!is_logical_owner())
      {
        RegionNode *origin_node = logical_node->is_region() ? 
          logical_node->as_region_node() : 
          logical_node->as_partition_node()->parent;
        // If we are no the owner, we have to send the user back
        RtUserEvent remote_update_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_update_event);
          rez.serialize<bool>(false); // is copy
          rez.serialize<bool>(false); // restrict out
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize(child_color);
          rez.serialize(origin_node->handle);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(term_event);
          // Figure out which version infos we need
          LegionMap<VersionID,FieldMask>::aligned needed_versions;
          FieldVersions field_versions;
          // We don't need to worry about split fields here, that
          // will be taken care of on the handler side, just pack
          // up the version infos as they are currently stored
          versions->get_field_versions(logical_node, false/*split previous*/,
                                       user_mask, field_versions);
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                field_versions.begin(); it != field_versions.end(); it++)
          {
            FieldMask overlap = it->second & user_mask;
            if (!overlap)
              continue;
            needed_versions[it->first] = overlap;
          }
          rez.serialize<size_t>(needed_versions.size());
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                needed_versions.begin(); it != needed_versions.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          FieldMask local_split;
          versions->get_split_mask(logical_node, user_mask, local_split);
          rez.serialize(local_split);
          rez.serialize<bool>(base_user);
        }
        runtime->send_view_remote_update(logical_owner, rez);
        // Tell the operation it has to wait for this event to
        // trigger before it can be considered mapped
        applied_events.insert(remote_update_event);
      }
      PhysicalUser *new_user = 
        new PhysicalUser(usage, child_color, op_id, index, origin_node);
      new_user->add_reference();
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      // Finally add our user and return if we need to issue a GC meta-task
      add_current_user(new_user, term_event, user_mask);
      // See if we need to check for read only invalidates
      // Don't worry about read-write, the invalidations will
      // be sent automatically if the version number is advanced
      if (!valid_remote_instances.empty() && IS_READ_ONLY(usage))
        perform_read_invalidations(user_mask, versions, source, applied_events);
      if (outstanding_gc_events.find(term_event) == 
          outstanding_gc_events.end())
      {
        outstanding_gc_events.insert(term_event);
        return (child_color == INVALID_COLOR);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::add_user_fused(const RegionUsage &usage, 
                                             ApEvent term_event,
                                             const FieldMask &user_mask, 
                                             Operation *op,const unsigned index,
                                             VersionTracker *versions,
                                             const AddressSpaceID source,
                                             std::set<RtEvent> &applied_events,
                                             bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> wait_on_events;
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      UniqueID op_id = op->get_unique_op_id();
      RegionNode *origin_node = logical_node->is_region() ? 
        logical_node->as_region_node() : 
        logical_node->as_partition_node()->parent;
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, INVALID_COLOR, 
                     origin_node, versions, op_id, index, user_mask, 
                     wait_on_events, applied_events);
      bool need_version_update = false;
      if (IS_WRITE(usage) && update_versions)
      {
        FieldVersions advance_versions;
        versions->get_advance_versions(logical_node, true/*base*/,
                                       user_mask, advance_versions);
        need_version_update = update_version_numbers(user_mask,advance_versions,
                                                     source, applied_events);
      }
      // Go up the tree if necessary
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_user_above_fused(usage, term_event, local_color, 
                              origin_node, versions, op_id, index, 
                              user_mask, source, wait_on_events, 
                              applied_events, need_version_update);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                         INVALID_COLOR, origin_node, true/*base*/, versions,
                         op_id, index, user_mask, source, applied_events);
      // Launch the garbage collection task, if it doesn't exist
      // then the user wasn't registered anyway, see add_local_user
      if (issue_collect)
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
      }
      // At this point tasks shouldn't be allowed to wait on themselves
#ifdef DEBUG_LEGION
      if (term_event.exists())
        assert(wait_on_events.find(term_event) == wait_on_events.end());
#endif
      if (IS_ATOMIC(usage))
        find_atomic_reservations(user_mask, op, IS_WRITE(usage));
      // Return the merge of the events
      return Runtime::merge_events(wait_on_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above_fused(const RegionUsage &usage, 
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                RegionNode *origin_node,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                                const AddressSpaceID source,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                                const bool need_version_update)
    //--------------------------------------------------------------------------
    {
      // Do the precondition analysis on the way up
      find_local_user_preconditions_above(usage, term_event, child_color, 
                          origin_node, versions, op_id, index, user_mask, 
                          preconditions, applied_events);
      bool need_update_above = false;
      if (need_version_update)
      {
        FieldVersions advance_versions;
        versions->get_advance_versions(logical_node, false/*base*/, 
                                       user_mask, advance_versions);
        need_update_above = update_version_numbers(user_mask, advance_versions,
                                                   source, applied_events);
      }
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_user_above_fused(usage, term_event, local_color,origin_node,
                              versions, op_id, index, user_mask, source,
                              preconditions, applied_events, need_update_above);
      }
      // Add the user on the way back down
      add_local_user(usage, term_event, child_color, origin_node, false/*base*/,
                     versions, op_id, index, user_mask, source, applied_events);
      // No need to launch a collect user task, the child takes care of that
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_initial_user(ApEvent term_event,
                                            const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            const UniqueID op_id,
                                            const unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(logical_node->is_region());
#endif
      // No need to take the lock since we are just initializing
      PhysicalUser *user = new PhysicalUser(usage, INVALID_COLOR, op_id, index, 
                                            logical_node->as_region_node());
      user->add_reference();
      add_current_user(user, term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        if (is_owner())
          manager->add_nested_gc_ref(did, mutator);
        else
          send_remote_gc_update(owner_space, mutator, 1, true/*add*/);
      }
      else
        parent->add_nested_gc_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
      {
        // we have a resource reference on the manager so no need to check
        if (is_owner())
          manager->remove_nested_gc_ref(did, mutator);
        else
          send_remote_gc_update(owner_space, mutator, 1, false/*add*/);
      }
      else if (parent->remove_nested_gc_ref(did, mutator))
        delete parent;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL)
        manager->add_nested_valid_ref(did, mutator);
      else
        parent->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (parent == NULL) 
        // we have a resource reference on the manager so no need to check
        manager->remove_nested_valid_ref(did, mutator);
      else if (parent->remove_nested_valid_ref(did, mutator))
        delete parent;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::collect_users(const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock v_lock(view_lock);
        // Remove any event users from the current and previous users
        for (std::set<ApEvent>::const_iterator it = term_events.begin();
              it != term_events.end(); it++)
          filter_local_users(*it); 
      }
      if (parent != NULL)
        parent->collect_users(term_events);
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(manager->did);
        if (parent == NULL)
          rez.serialize<DistributedID>(0);
        else
          rez.serialize<DistributedID>(parent->did);
        if (logical_node->is_region())
        {
          rez.serialize<bool>(true);
          rez.serialize(logical_node->as_region_node()->handle);
        }
        else
        {
          rez.serialize<bool>(false);
          rez.serialize(logical_node->as_partition_node()->handle);
        }
        rez.serialize(owner_space);
        rez.serialize(logical_owner);
        rez.serialize(owner_context);
      }
      runtime->send_materialized_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::update_gc_events(
                                           const std::deque<ApEvent> &gc_events)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->update_gc_events(gc_events);
      AutoLock v_lock(view_lock);
      for (std::deque<ApEvent>::const_iterator it = gc_events.begin();
            it != gc_events.end(); it++)
      {
        outstanding_gc_events.insert(*it);
      }
    }    

    //--------------------------------------------------------------------------
    void MaterializedView::filter_invalid_fields(FieldMask &to_filter,
                                                 VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      // If we're not the parent then keep going up
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
        parent->filter_invalid_fields(to_filter, version_info);
      // If we still have fields to filter, then do that now
      if (!!to_filter)
      {
        // If we're not the owner then make sure that we are up to date
        if (!is_logical_owner())
          perform_remote_valid_check(to_filter, &version_info, true/*reading*/);
        // Get the version numbers that we need 
        FieldVersions needed_versions;
        version_info.get_field_versions(logical_node, false/*split prev*/,
                                        to_filter, needed_versions);
        // Need the lock in read only mode while touching current versions
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        for (FieldVersions::iterator it = needed_versions.begin();
              it != needed_versions.end(); it++)
        {
          std::map<VersionID,FieldMask>::const_iterator finder =  
            current_versions.find(it->first);
          if (finder != current_versions.end())
          {
            const FieldMask to_remove = it->second - finder->second;
            if (!!to_remove)
            {
              to_filter -= to_remove;
              if (!to_filter)
                return;
            }
          }
          else
          {
            // None of the fields are at the right version number
            to_filter -= it->second;
            if (!to_filter)
              return;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_version_updates(const FieldMask &copy_mask,
                                                     VersionTracker *versions,
                                                     FieldMask &write_skip_mask,
                                                     FieldMask &filter_mask,
                              LegionMap<VersionID,FieldMask>::aligned &advance,
                              LegionMap<VersionID,FieldMask>::aligned &add_only,
                                bool is_reduction, bool restrict_out, bool base)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
      // These are updates for a copy, so we are never going to the
      // next version number, we only go to the current versions
      FieldVersions update_versions;
      // If we're doing a restrict out copy, the version we are updating
      // is the advance version, otherwise it is the current version
      // before the operation. We want split previous here because we
      // haven't actually done the write yet, so if we're writing then
      // we're just copying the previous version over.
      if (restrict_out)
        versions->get_advance_versions(logical_node, base, 
                                       copy_mask, update_versions);
      else
        versions->get_field_versions(logical_node, true/*split previous*/,
                                     copy_mask, update_versions);
      FieldMask split_mask;
      versions->get_split_mask(logical_node, copy_mask, split_mask);
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            update_versions.begin(); it != update_versions.end(); it++)
      {
        FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        if (it->first == 0)
        {
          filter_mask |= overlap;
          add_only[it->first] = overlap;
          continue;
        }
        // We're trying to get this to current version number, check to
        // see if we're already at one of the most common values of
        // either the previous version number or the current one
        // otherwise we will have to add this to the set to filter later
        const VersionID previous_number = it->first - 1;
        const VersionID next_number = it->first;
        LegionMap<VersionID,FieldMask>::aligned::const_iterator finder = 
          current_versions.find(previous_number);
        if (finder != current_versions.end())
        {
          const FieldMask intersect = overlap & finder->second;
          if (!!intersect)
          {
            advance[previous_number] = intersect;
            overlap -= intersect;
            if (!overlap)
              continue;
          }
          // Bump the iterator to the next entry, hopefully 
          // it is the next version number, but if not we'll figure it out
          finder++;
        }
        else
          finder = current_versions.find(next_number);
        // Check if the finder is good and the right version number
        if ((finder != current_versions.end()) && 
            (finder->first == next_number))
        {
          const FieldMask intersect = overlap & finder->second;
          if (!!intersect)
          {
            // This is a write skip field since we're already
            // at the version number at this view, but we're only
            // really at the version number if we're not reducing
            // We can't count split fields here because they might
            // contain users from many versions
#ifndef LEGION_SPY 
            if (!is_reduction)
            {
              if (!!split_mask)
                write_skip_mask |= (intersect - split_mask);
              else
                write_skip_mask |= intersect;
            }
#endif
            overlap -= intersect;
            if (!overlap)
              continue;
          }
        }
        // If we still have fields, then record we need to filter them
        filter_mask |= overlap;
        // Record the version number and fields to add after the filter
        add_only[next_number] = overlap;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::apply_version_updates(FieldMask &filter_mask,
                      const LegionMap<VersionID,FieldMask>::aligned &advance,
                      const LegionMap<VersionID,FieldMask>::aligned &add_only,
                      AddressSpaceID source, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
      // If we have remote instances, we need to check to see 
      // if we need to send any invalidations
      if (!valid_remote_instances.empty())
      {
#ifdef DEBUG_LEGION
        assert(is_logical_owner());
#endif
        // Keep track of any invalidations that we have to apply 
        // make a copy here before filter gets destroyed by the call
        FieldMask invalidate_mask = filter_mask;
        if (!!filter_mask)
        {
          // See if any of them are already up to date so we don't have
          // to send invalidations, this is expensive enough that it is
          // worth the extra analysis cost here to just do it
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                add_only.begin(); it != add_only.end(); it++)
          {
            LegionMap<VersionID,FieldMask>::aligned::const_iterator finder = 
              current_versions.find(it->first);
            if (finder == current_versions.end())
              continue;
            FieldMask overlap = finder->second & it->second;
            if (!!overlap)
              invalidate_mask -= overlap;
          }
          filter_and_add(filter_mask, add_only);
        }
        if (!advance.empty())
        {
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                advance.begin(); it != advance.end(); it++)
          {
            LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
              current_versions.find(it->first);
            // Someone else could already have advanced this
            if (finder == current_versions.end())
              continue;
            FieldMask overlap = finder->second & it->second;
            if (!overlap)
              continue;
            finder->second -= overlap;
            if (!finder->second)
              current_versions.erase(finder);
            current_versions[it->first+1] |= overlap;
            invalidate_mask |= overlap;
          }
        }
        if (!!invalidate_mask)
          send_invalidations(invalidate_mask, source, applied_events);
      }
      else
      {
        // This is the common path
        if (!!filter_mask)
          filter_and_add(filter_mask, add_only);
        if (!advance.empty())
        {
          for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
                advance.begin(); it != advance.end(); it++)
          {
            LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
              current_versions.find(it->first);
            // Someone else could already have advanced this
            if (finder == current_versions.end())
              continue;
            finder->second -= it->second;
            if (!finder->second)
              current_versions.erase(finder);
            current_versions[it->first+1] |= it->second;
          }
        }
      }
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_and_add(FieldMask &filter_mask,
                    const LegionMap<VersionID,FieldMask>::aligned &add_versions)
    //--------------------------------------------------------------------------
    {
      std::vector<VersionID> to_delete; 
      for (LegionMap<VersionID,FieldMask>::aligned::iterator it = 
            current_versions.begin(); it != current_versions.end(); it++)
      {
        FieldMask overlap = it->second & filter_mask;
        if (!overlap)
          continue;
        it->second -= overlap;
        if (!it->second)
          to_delete.push_back(it->first);
        filter_mask -= overlap;
        if (!filter_mask)
          break;
      }
      // Delete the old entries
      if (!to_delete.empty())
      {
        for (std::vector<VersionID>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
          current_versions.erase(*it);
      }
      // Then add the new entries
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            add_versions.begin(); it != add_versions.end(); it++)
        current_versions[it->first] |= it->second;
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::update_version_numbers(const FieldMask &user_mask,
                                           const FieldVersions &target_versions,
                                           const AddressSpaceID source,
                                           std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      FieldMask filter_mask, invalidate_mask;
      LegionMap<VersionID,FieldMask>::aligned update_versions;
      bool need_check_above = false;
      // Need the lock in exclusive mode to do the update
      AutoLock v_lock(view_lock);
      // If we are logical owner and we have remote valid instances
      // we need to track which version numbers get updated so we can
      // send invalidates
      const bool need_invalidates = is_logical_owner() && 
          !valid_remote_instances.empty() && !(user_mask * remote_valid_mask);
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            target_versions.begin(); it != target_versions.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first > 0);
#endif
        FieldMask overlap = it->second & user_mask;
        if (!overlap)
          continue;
        // We are always trying to advance the version numbers here
        // since these are writing users and are therefore going from
        // the current version number to the next one. We'll check for
        // the most common cases here, and only filter if we don't find them.
        const VersionID previous_number = it->first - 1; 
        const VersionID next_number = it->first; 
        LegionMap<VersionID,FieldMask>::aligned::iterator finder = 
          current_versions.find(previous_number);
        if (finder != current_versions.end())
        {
          FieldMask intersect = overlap & finder->second;
          if (!!intersect)
          {
            need_check_above = true;
            finder->second -= intersect;
            if (need_invalidates)
              invalidate_mask |= intersect;
            if (!finder->second)
            {
              current_versions.erase(finder);
              // We just deleted the iterator so we need a new one
              finder = current_versions.find(next_number);
            }
            else // We didn't delete the iterator so trying bumping it
              finder++;
            if (finder != current_versions.end())
            {
              if (finder->first != next_number) 
              {
                current_versions[next_number] = intersect;
                // Set it to end since we know there is no point in checking
                finder = current_versions.end();
              }
              else // finder points to the right place
                finder->second |= intersect;
            }
            else // no valid iterator so just put in the value
              current_versions[next_number] = intersect;
            overlap -= intersect;
            if (!overlap)
              continue;
          }
          else // Try the next element, hopefully it is version number+1
            finder++;
        }
        else
          finder = current_versions.find(next_number);
        // Check if the finder is good and the right version number
        if ((finder != current_versions.end()) && 
            (finder->first == next_number))
        {
          FieldMask intersect = overlap & finder->second;
          if (!!intersect)
          {
            finder->second |= intersect;
            overlap -= intersect;
            if (!overlap)
              continue;
          }
        }
        // If we still have fields, then record we need to filter them
        filter_mask |= overlap;
        // Record the version number and fields to add after the filter
        update_versions[next_number] = overlap;
        if (need_invalidates)
          invalidate_mask |= overlap;
      }
      // If we need to filter, let's do that now
      if (!!filter_mask)
      {
        need_check_above = true;
        filter_and_add(filter_mask, update_versions);  
      }
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
      if (!!invalidate_mask)
        send_invalidations(invalidate_mask, source, applied_events);
      return need_check_above;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void MaterializedView::sanity_check_versions(void)
    //--------------------------------------------------------------------------
    {
      FieldMask version_mask;
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            current_versions.begin(); it != current_versions.end(); it++)
      {
        assert(version_mask * it->second);
        version_mask |= it->second;
      }
    }
#endif

    //--------------------------------------------------------------------------
    void MaterializedView::add_current_user(PhysicalUser *user, 
                                            ApEvent term_event,
                                            const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Must be called while holding the lock
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
    void MaterializedView::filter_local_users(ApEvent term_event) 
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FILTER_LOCAL_USERS_CALL);
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      std::set<ApEvent>::iterator event_finder = 
        outstanding_gc_events.find(term_event); 
      if (event_finder != outstanding_gc_events.end())
      {
        LegionMap<ApEvent,EventUsers>::aligned::iterator current_finder = 
          current_epoch_users.find(term_event);
        if (current_finder != current_epoch_users.end())
        {
          EventUsers &event_users = current_finder->second;
          if (event_users.single)
          {
            if (event_users.users.single_user->remove_reference())
              delete (event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                delete (it->first);
            }
            delete event_users.users.multi_users;
          }
          current_epoch_users.erase(current_finder);
        }
        LegionMap<ApEvent,EventUsers>::aligned::iterator previous_finder = 
          previous_epoch_users.find(term_event);
        if (previous_finder != previous_epoch_users.end())
        {
          EventUsers &event_users = previous_finder->second; 
          if (event_users.single)
          {
            if (event_users.users.single_user->remove_reference())
              delete (event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                delete (it->first);
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
    void MaterializedView::filter_local_users(const FieldMask &filter_mask,
                      LegionMap<ApEvent,EventUsers>::aligned &local_epoch_users)
    //--------------------------------------------------------------------------
    {
      // lock better be held by caller
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FILTER_LOCAL_USERS_CALL);
      std::vector<ApEvent> to_delete;
      for (LegionMap<ApEvent,EventUsers>::aligned::iterator lit = 
           local_epoch_users.begin(); lit != local_epoch_users.end(); lit++)
      {
        const FieldMask overlap = lit->second.user_mask & filter_mask;
        if (!overlap)
          continue;
        EventUsers &local_users = lit->second;
        local_users.user_mask -= overlap;
        if (!local_users.user_mask)
        {
          // Total removal of the entry
          to_delete.push_back(lit->first);
          if (local_users.single)
          {
            PhysicalUser *user = local_users.users.single_user;
            if (user->remove_reference())
              delete (user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = local_users.users.multi_users->begin(); it !=
                  local_users.users.multi_users->end(); it++)
            {
              if (it->first->remove_reference())
                delete (it->first);
            }
            // Delete the map too
            delete local_users.users.multi_users;
          }
        }
        else if (!local_users.single) // only need to filter for non-single
        {
          // Partial removal of the entry
          std::vector<PhysicalUser*> to_erase;
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                local_users.users.multi_users->begin(); it !=
                local_users.users.multi_users->end(); it++)
          {
            it->second -= overlap;
            if (!it->second)
              to_erase.push_back(it->first);
          }
          if (!to_erase.empty())
          {
            for (std::vector<PhysicalUser*>::const_iterator it = 
                  to_erase.begin(); it != to_erase.end(); it++)
            {
              local_users.users.multi_users->erase(*it);
              if ((*it)->remove_reference())
                delete (*it);
            }
            // See if we can shrink this back down
            if (local_users.users.multi_users->size() == 1)
            {
              LegionMap<PhysicalUser*,FieldMask>::aligned::iterator first_it =
                            local_users.users.multi_users->begin();     
#ifdef DEBUG_LEGION
              // This summary mask should dominate
              assert(!(first_it->second - local_users.user_mask));
#endif
              PhysicalUser *user = first_it->first;
              local_users.user_mask = first_it->second;
              delete local_users.users.multi_users;
              local_users.users.single_user = user;
              local_users.single = true;
            }
          }
        }
      }
      if (!to_delete.empty())
      {
        for (std::vector<ApEvent>::const_iterator it = to_delete.begin();
              it != to_delete.end(); it++)
          local_epoch_users.erase(*it);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_current_user(ApEvent user_event, 
                                               const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
      // lock better be held by caller
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FILTER_CURRENT_USERS_CALL);
      LegionMap<ApEvent,EventUsers>::aligned::iterator cit = 
        current_epoch_users.find(user_event);
      // Some else might already have moved it back or it could have
      // been garbage collected already
      if (cit == current_epoch_users.end())
        return;
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      if (cit->first.has_triggered_faultignorant())
      {
        EventUsers &current_users = cit->second;
        if (current_users.single)
        {
          if (current_users.users.single_user->remove_reference())
            delete (current_users.users.single_user);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                current_users.users.multi_users->begin(); it !=
                current_users.users.multi_users->end(); it++)
          {
            if (it->first->remove_reference())
              delete (it->first);
          }
          delete current_users.users.multi_users;
        }
        current_epoch_users.erase(cit);
        return;
      }
#endif
      EventUsers &current_users = cit->second;
      FieldMask summary_overlap = current_users.user_mask & filter_mask;
      if (!summary_overlap)
        return;
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
              current_epoch_users.erase(cit);
            else
              user->add_reference(); // add a reference
          }
          else if (prev_users.users.single_user == user)
          {
            // Same user, update the fields 
            prev_users.user_mask |= summary_overlap;
            if (!current_users.user_mask)
            {
              current_epoch_users.erase(cit);
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
              current_epoch_users.erase(cit);
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
              current_epoch_users.erase(cit);
            else
              user->add_reference();
          }
          else
          {
            // Found it, update it 
            finder->second |= summary_overlap;
            if (!current_users.user_mask)
            {
              current_epoch_users.erase(cit);
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
          current_epoch_users.erase(cit);
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
#ifdef DEBUG_LEGION
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

    //--------------------------------------------------------------------------
    void MaterializedView::filter_previous_user(ApEvent user_event,
                                                const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
      // lock better be held by caller
      DETAILED_PROFILER(context->runtime,
                        MATERIALIZED_VIEW_FILTER_PREVIOUS_USERS_CALL);
      LegionMap<ApEvent,EventUsers>::aligned::iterator pit = 
        previous_epoch_users.find(user_event);
      // This might already have been filtered or garbage collected
      if (pit == previous_epoch_users.end())
        return;
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      if (pit->first.has_triggered_faultignorant())
      {
        EventUsers &previous_users = pit->second;
        if (previous_users.single)
        {
          if (previous_users.users.single_user->remove_reference())
            delete (previous_users.users.single_user);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                previous_users.users.multi_users->begin(); it !=
                previous_users.users.multi_users->end(); it++)
          {
            if (it->first->remove_reference())
              delete (it->first);
          }
          delete previous_users.users.multi_users;
        }
        previous_epoch_users.erase(pit);
        return;
      }
#endif
      EventUsers &previous_users = pit->second;
      FieldMask summary_overlap = previous_users.user_mask & filter_mask;
      if (!summary_overlap)
        return;
      previous_users.user_mask -= summary_overlap;
      if (!previous_users.user_mask)
      {
        // We can delete the whole entry
        if (previous_users.single)
        {
          PhysicalUser *user = previous_users.users.single_user;
          if (user->remove_reference())
            delete (user);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = previous_users.users.multi_users->begin(); it !=
                previous_users.users.multi_users->end(); it++)
          {
            if (it->first->remove_reference())
              delete (it->first);
          }
          // Delete the map too
          delete previous_users.users.multi_users;
        }
        previous_epoch_users.erase(pit);
      }
      else if (!previous_users.single) // only need to filter for non-single
      {
        // Filter out the users for the dominated fields
        std::vector<PhysicalUser*> to_delete;
        for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
              previous_users.users.multi_users->begin(); it !=
              previous_users.users.multi_users->end(); it++)
        {
          it->second -= summary_overlap; 
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<PhysicalUser*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            previous_users.users.multi_users->erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
          // See if we can shrink this back down
          if (previous_users.users.multi_users->size() == 1)
          {
            LegionMap<PhysicalUser*,FieldMask>::aligned::iterator first_it =
                          previous_users.users.multi_users->begin();     
#ifdef DEBUG_LEGION
            // This summary mask should dominate
            assert(!(first_it->second - previous_users.user_mask));
#endif
            PhysicalUser *user = first_it->first;
            previous_users.user_mask = first_it->second;
            delete previous_users.users.multi_users;
            previous_users.users.single_user = user;
            previous_users.single = true;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<bool TRACK_DOM>
    void MaterializedView::find_current_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const LegionColor child_color,
                                                 RegionNode *origin_node,
                                                 ApEvent term_event,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events,
                           LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                                 FieldMask &observed,
                                                 FieldMask &non_dominated)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator cit = 
           current_epoch_users.begin(); cit != current_epoch_users.end(); cit++)
      {
        if (cit->first == term_event)
          continue;
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (cit->first.has_triggered_faultignorant())
        {
          dead_events.insert(cit->first);
          continue;
        }
        if (preconditions.find(cit->first) != preconditions.end())
          continue;
#endif
        const EventUsers &event_users = cit->second;
        const FieldMask overlap = event_users.user_mask & user_mask;
        if (!overlap)
          continue;
        else if (TRACK_DOM)
          observed |= overlap;
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, origin_node))
          {
            preconditions.insert(cit->first);
            if (TRACK_DOM)
              filter_events[cit->first] = overlap;
          }
          else if (TRACK_DOM)
            non_dominated |= overlap;
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            const FieldMask user_overlap = user_mask & it->second;
            if (!user_overlap)
              continue;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, origin_node))
            {
              preconditions.insert(cit->first);
              if (TRACK_DOM)
                filter_events[cit->first] |= user_overlap;
            }
            else if (TRACK_DOM)
              non_dominated |= user_overlap;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const LegionColor child_color,
                                                 RegionNode *origin_node,
                                                 ApEvent term_event,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator pit = 
            previous_epoch_users.begin(); pit != 
            previous_epoch_users.end(); pit++)
      {
        if (pit->first == term_event)
          continue;
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (pit->first.has_triggered_faultignorant())
        {
          dead_events.insert(pit->first);
          continue;
        }
        if (preconditions.find(pit->first) != preconditions.end())
          continue;
#endif
        const EventUsers &event_users = pit->second;
        if (user_mask * event_users.user_mask)
          continue;
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, origin_node))
            preconditions.insert(pit->first);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            if (user_mask * it->second)
              continue;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, origin_node))
              preconditions.insert(pit->first);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<bool TRACK_DOM>
    void MaterializedView::find_current_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const LegionColor child_color,
                                                 RegionNode *origin_node,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                                 std::set<ApEvent> &dead_events,
                           LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                                 FieldMask &observed,
                                                 FieldMask &non_dominated)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator cit = 
           current_epoch_users.begin(); cit != current_epoch_users.end(); cit++)
      {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (cit->first.has_triggered_faultignorant())
        {
          dead_events.insert(cit->first);
          continue;
        }
#endif
        const EventUsers &event_users = cit->second;
        FieldMask overlap = event_users.user_mask & user_mask;
        if (!overlap)
          continue;
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          preconditions.find(cit->first);
#ifndef LEGION_SPY
        if (finder != preconditions.end())
        {
          overlap -= finder->second;
          if (!overlap)
            continue;
        }
#endif
        if (TRACK_DOM)
          observed |= overlap;
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, origin_node))
          {
            if (finder == preconditions.end())
              preconditions[cit->first] = overlap;
            else
              finder->second |= overlap;
            if (TRACK_DOM)
              filter_events[cit->first] = overlap;
          }
          else if (TRACK_DOM)
            non_dominated |= overlap;
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            const FieldMask user_overlap = user_mask & it->second;
            if (!user_overlap)
              continue;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, origin_node))
            {
              if (finder == preconditions.end())
                preconditions[cit->first] = user_overlap;
              else
                finder->second |= user_overlap;
              if (TRACK_DOM)
                filter_events[cit->first] |= user_overlap;
            }
            else if (TRACK_DOM)
              non_dominated |= user_overlap;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const LegionColor child_color,
                                                 RegionNode *origin_node,
                                                 const UniqueID op_id,
                                                 const unsigned index,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                                 std::set<ApEvent> &dead_events)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator pit = 
            previous_epoch_users.begin(); pit != 
            previous_epoch_users.end(); pit++)
      {
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (pit->first.has_triggered_faultignorant())
        {
          dead_events.insert(pit->first);
          continue;
        }
#endif
        const EventUsers &event_users = pit->second;
        FieldMask overlap = user_mask & event_users.user_mask;
        if (!overlap)
          continue;
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          preconditions.find(pit->first);
#ifndef LEGION_SPY
        if (finder != preconditions.end())
        {
          overlap -= finder->second;
          if (!overlap)
            continue;
        }
#endif
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, origin_node))
          {
            if (finder == preconditions.end())
              preconditions[pit->first] = overlap;
            else
              finder->second |= overlap;
          }
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            const FieldMask user_overlap = overlap & it->second;
            if (!user_overlap)
              continue;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, origin_node))
            {
              if (finder == preconditions.end())
              {
                preconditions[pit->first] = user_overlap;
                // Needed for when we go around the loop again
                finder = preconditions.find(pit->first);
              }
              else
                finder->second |= user_overlap;
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_filter_users(const FieldMask &dom_mask,
                            LegionMap<ApEvent,FieldMask>::aligned &filter_users)
    //--------------------------------------------------------------------------
    {
      // Lock better be held by caller
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator it = 
           previous_epoch_users.begin(); it != previous_epoch_users.end(); it++)
      {
        FieldMask overlap = it->second.user_mask & dom_mask;
        if (!overlap)
          continue;
        filter_users[it->first] = overlap;
      }
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
        std::vector<FieldID> atomic_fields;
        logical_node->column_source->get_field_set(mask, atomic_fields);
        // If we are the owner we can do this here
        if (is_owner())
        {
          std::vector<Reservation> reservations(atomic_fields.size());
          find_field_reservations(atomic_fields, reservations);
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            op->update_atomic_locks(reservations[idx], excl);
        }
        else
        {
          // Figure out which fields we need requests for and send them
          std::vector<FieldID> needed_fields;
          {
            AutoLock v_lock(view_lock, 1, false);
            for (std::vector<FieldID>::const_iterator it = 
                  atomic_fields.begin(); it != atomic_fields.end(); it++)
            {
              std::map<FieldID,Reservation>::const_iterator finder = 
                atomic_reservations.find(*it);
              if (finder == atomic_reservations.end())
                needed_fields.push_back(*it);
              else
                op->update_atomic_locks(finder->second, excl);
            }
          }
          if (!needed_fields.empty())
          {
            RtUserEvent wait_on = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize<size_t>(needed_fields.size());
              for (unsigned idx = 0; idx < needed_fields.size(); idx++)
                rez.serialize(needed_fields[idx]);
              rez.serialize(wait_on);
            }
            runtime->send_atomic_reservation_request(owner_space, rez);
            wait_on.wait();
            // Now retake the lock and get the remaining reservations
            AutoLock v_lock(view_lock, 1, false);
            for (std::vector<FieldID>::const_iterator it = 
                  needed_fields.begin(); it != needed_fields.end(); it++)
            {
              std::map<FieldID,Reservation>::const_iterator finder =
                atomic_reservations.find(*it);
#ifdef DEBUG_LEGION
              assert(finder != atomic_reservations.end());
#endif
              op->update_atomic_locks(finder->second, excl);
            }
          }
        }
      }
      else
        parent->find_atomic_reservations(mask, op, excl);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_field_reservations(
                                    const std::vector<FieldID> &needed_fields, 
                                    std::vector<Reservation> &results)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(needed_fields.size() == results.size());
#endif
      AutoLock v_lock(view_lock);
      for (unsigned idx = 0; idx < needed_fields.size(); idx++)
      {
        std::map<FieldID,Reservation>::const_iterator finder = 
          atomic_reservations.find(needed_fields[idx]);
        if (finder == atomic_reservations.end())
        {
          // Make a new reservation and add it to the set
          Reservation handle = Reservation::create_reservation();
          atomic_reservations[needed_fields[idx]] = handle;
          results[idx] = handle;
        }
        else
          results[idx] = finder->second;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_atomic_reservation_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
        derez.deserialize(fields[idx]);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      MaterializedView *target = dynamic_cast<MaterializedView*>(dc);
      assert(target != NULL);
#else
      MaterializedView *target = static_cast<MaterializedView*>(dc);
#endif
      std::vector<Reservation> reservations(num_fields);
      target->find_field_reservations(fields, reservations);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(num_fields);
        for (unsigned idx = 0; idx < num_fields; idx++)
        {
          rez.serialize(fields[idx]);
          rez.serialize(reservations[idx]);
        }
        rez.serialize(to_trigger);
      }
      runtime->send_atomic_reservation_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::update_field_reservations(
                                  const std::vector<FieldID> &fields, 
                                  const std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(fields.size() == reservations.size());
#endif
      AutoLock v_lock(view_lock);
      for (unsigned idx = 0; idx < fields.size(); idx++)
        atomic_reservations[fields[idx]] = reservations[idx];
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_send_atomic_reservation_response(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      size_t num_fields;
      derez.deserialize(num_fields);
      std::vector<FieldID> fields(num_fields);
      std::vector<Reservation> reservations(num_fields);
      for (unsigned idx = 0; idx < num_fields; idx++)
      {
        derez.deserialize(fields[idx]);
        derez.deserialize(reservations[idx]);
      }
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      MaterializedView *target = dynamic_cast<MaterializedView*>(dc);
      assert(target != NULL);
#else
      MaterializedView *target = static_cast<MaterializedView*>(dc);
#endif
      target->update_field_reservations(fields, reservations);
      Runtime::trigger_event(to_trigger);
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
      DistributedID parent_did;
      derez.deserialize(parent_did);
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
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      RtEvent man_ready;
      PhysicalManager *phy_man = 
        runtime->find_or_request_physical_manager(manager_did, man_ready);
      MaterializedView *parent = NULL;
      if (parent_did != 0)
      {
        RtEvent par_ready;
        LogicalView *par_view = 
          runtime->find_or_request_logical_view(parent_did, par_ready);
        if (par_ready.exists() && !par_ready.has_triggered())
        {
          // Need to avoid virtual channel deadlock here so defer it
          DeferMaterializedViewArgs args;
          args.did = did;
          args.owner_space = owner_space;
          args.logical_owner = logical_owner;
          args.target_node = target_node;
          args.manager = phy_man;
          // Have to static cast this since it might not be ready
          args.parent = static_cast<MaterializedView*>(par_view);
          args.context_uid = context_uid;
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                  Runtime::merge_events(par_ready, man_ready));
          return;
        }
#ifdef DEBUG_LEGION
        assert(par_view->is_materialized_view());
#endif
        parent = par_view->as_materialized_view();
      }
      if (man_ready.exists())
        man_ready.wait();
#ifdef DEBUG_LEGION
      assert(phy_man->is_instance_manager());
#endif
      create_remote_materialized_view(runtime, did, owner_space, logical_owner,
                                    target_node, phy_man, parent, context_uid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_deferred_materialized_view(
                                             Runtime *runtime, const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMaterializedViewArgs *margs = 
        (const DeferMaterializedViewArgs*)args;
      create_remote_materialized_view(runtime, margs->did, margs->owner_space,
          margs->logical_owner, margs->target_node, margs->manager,
          margs->parent, margs->context_uid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::create_remote_materialized_view(
       Runtime *runtime, DistributedID did, AddressSpaceID owner_space,
       AddressSpaceID logical_owner, RegionTreeNode *target_node, 
       PhysicalManager *phy_man, MaterializedView *parent, UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
      InstanceManager *inst_manager = phy_man->as_instance_manager();
      void *location;
      MaterializedView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) MaterializedView(runtime->forest,
                                              did, owner_space, 
                                              logical_owner, 
                                              target_node, inst_manager,
                                              parent, context_uid,
                                              false/*register now*/);
      else
        view = new MaterializedView(runtime->forest, did, owner_space,
                                    logical_owner, target_node, inst_manager, 
                                    parent, context_uid,false/*register now*/);
      if (parent != NULL)
        parent->add_remote_child(view);
      // Register only after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::perform_remote_valid_check(
                  const FieldMask &check_mask, VersionTracker *versions,
                  bool reading, std::set<RtEvent> *wait_on)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      FieldMask need_valid_update;
      std::set<RtEvent> local_wait_on;
      RtUserEvent request_event;
      if (reading)
      {
        // If we are reading we need to check to see if we are at
        // the right version number and whether we have done the read
        // request yet for our given version number
        FieldVersions field_versions;
        versions->get_field_versions(logical_node, true/*split prev*/,
                                     check_mask, field_versions);
        FieldMask split_mask;
        versions->get_split_mask(logical_node, check_mask, split_mask);
        const bool has_split_mask = !!split_mask;
        need_valid_update = check_mask;
        AutoLock v_lock(view_lock);
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              field_versions.begin(); it != field_versions.end(); it++)
        {
          FieldMask overlap = it->second & check_mask;
          if (!overlap)
            continue;
          // See if we can find it as the current version number
          LegionMap<VersionID,FieldMask>::aligned::const_iterator finder = 
            current_versions.find(it->first);
          if (finder != current_versions.end())
          {
            const FieldMask version_overlap = overlap & finder->second;
            if (!!version_overlap)
            {
              need_valid_update -= version_overlap;
              if (!need_valid_update)
                break;
            }
          }
          // If we have a split mask, it's also alright if the current
          // versions are at the next version number
          if (has_split_mask)
          {
            const FieldMask split_overlap = overlap & split_mask;
            if (!split_overlap)
              continue;
            finder = current_versions.find(it->first + 1);
            if (finder != current_versions.end())
            {
              const FieldMask version_overlap = split_overlap & finder->second;
              if (!!version_overlap)
              {
                need_valid_update -= version_overlap;
                if (!need_valid_update)
                  break;
              }
            }
          }
        }
        // Also look for any pending requests that overlap since they
        // will bring the result up to date for us too
        if (!remote_update_requests.empty())
        {
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it =
                remote_update_requests.begin(); it != 
                remote_update_requests.end(); it++)
          {
            if (it->second * check_mask)
              continue;
            local_wait_on.insert(it->first);
            need_valid_update -= it->second;
          }
        }
        // Figure out what we need to send
        if (!!need_valid_update)
        {
          request_event = Runtime::create_rt_user_event();
          remote_update_requests[request_event] = need_valid_update;
        }
      }
      else
      {
        // If we're writing all we need to do is check that we are valid,
        // if we're not valid we have to send a request
        AutoLock v_lock(view_lock);
        need_valid_update = check_mask - remote_valid_mask;
        if (!remote_update_requests.empty())
        {
          // See which fields we already have requests for
          for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
                remote_update_requests.begin(); it != 
                remote_update_requests.end(); it++)
          {
            FieldMask overlap = check_mask & it->second;
            if (!overlap)
              continue;
            if (wait_on != NULL)
              wait_on->insert(it->first);
            else
              local_wait_on.insert(it->first);
            need_valid_update -= overlap;
          }
        }
        if (!!need_valid_update)
        {
          request_event = Runtime::create_rt_user_event();
          remote_update_requests[request_event] = need_valid_update;
          // We also have to filter out the current and previous epoch 
          // user lists so that when we get the update then we know we
          // won't have polluting users in the list to start
          filter_local_users(need_valid_update, current_epoch_users);
          filter_local_users(need_valid_update, previous_epoch_users);
        }
      }
      // If we have a request event, send the request now
      if (request_event.exists())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(request_event);
          rez.serialize(need_valid_update);
        }
        context->runtime->send_view_update_request(logical_owner, rez);
        local_wait_on.insert(request_event);
      }
      // If we have a parent, see if it needs to send requests too so 
      // we can get as many in flight as possible in parallel
      if (parent != NULL)
      {
        if (wait_on != NULL)
          parent->perform_remote_valid_check(check_mask, versions,
                                             reading, wait_on);
        else
          parent->perform_remote_valid_check(check_mask, versions,
                                             reading, &local_wait_on);
      }
      // If we have any events to wait on do the right thing with them
      if (!local_wait_on.empty())
      {
        if (wait_on == NULL)
        {
          // If we are the base caller, then we do the wait
          RtEvent wait_for = Runtime::merge_events(local_wait_on);
          wait_for.wait();
        }
        else // Otherwise add the events to the set to wait on
          wait_on->insert(local_wait_on.begin(), local_wait_on.end());
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::perform_read_invalidations(
                 const FieldMask &check_mask, VersionTracker *versions,
                 const AddressSpaceID source, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Must be called while holding the view lock in exclusive mode
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // Quick test for intersection here to see if we are done early
      if (check_mask * remote_valid_mask)
        return;
      FieldMask invalidate_mask;
      // Check to see if we have a split mask, any fields which are
      // not split have to be invalidated since we're directly reading
      // the current version number (we know we're reading the current
      // version number or else something else is broken). In the
      // case of split version numbers the advance of the version
      // number already invalidated the remote leases so we don't
      // have to worry about it.
      FieldMask split_mask;
      versions->get_split_mask(logical_node, check_mask, split_mask);
      if (!!split_mask)
      {
        const FieldMask invalidate_mask = check_mask - split_mask;
        if (!!invalidate_mask)
          send_invalidations(invalidate_mask, source, applied_events);
      }
      else // Reading at the base invalidates all remote leases
        send_invalidations(check_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::send_invalidations(const FieldMask &invalidate_mask,
              const AddressSpaceID can_skip, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Must be called while holding the view lock in exclusive mode
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // No overlapping fields means we are done
      if (invalidate_mask * remote_valid_mask)
        return;
      std::vector<AddressSpaceID> to_delete;
      bool has_skip = false;
      for (LegionMap<AddressSpaceID,FieldMask>::aligned::iterator it = 
            valid_remote_instances.begin(); it != 
            valid_remote_instances.end(); it++)
      {
        // If the update was from this node we don't need to send
        // an invalidate because clearly it is still up to date
        if (it->first == can_skip)
        {
          has_skip = true;
          continue;
        }
        FieldMask overlap = it->second & invalidate_mask;
        if (!overlap)
          continue;
        RtUserEvent invalidate_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(overlap);
          rez.serialize(invalidate_event);
        }
        context->runtime->send_view_remote_invalidate(it->first, rez);
        applied_events.insert(invalidate_event);
        it->second -= overlap;
        if (!it->second)
          to_delete.push_back(it->first);
      }
      if (!to_delete.empty())
      {
        for (std::vector<AddressSpaceID>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
          valid_remote_instances.erase(*it);
      }
      // Filter the remote valid mask and add back in any fields that
      // were skipped
      remote_valid_mask -= invalidate_mask;
      if (has_skip)
        remote_valid_mask |= valid_remote_instances[can_skip];
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_update_request(AddressSpaceID source,
                                    RtUserEvent done_event, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      FieldMask request_mask;
      derez.deserialize(request_mask);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_event);
        rez.serialize(request_mask);
        // Hold the view lock when building up the information to send back
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Package up the information to send back
        // First figure out which views to send back
        LegionMap<VersionID,FieldMask>::aligned response_versions;
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              current_versions.begin(); it != current_versions.end(); it++)
        {
          FieldMask overlap = it->second & request_mask;
          if (!overlap)
            continue;
          response_versions[it->first] = overlap;
        }
        rez.serialize<size_t>(response_versions.size());
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              response_versions.begin(); it != response_versions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        std::vector<ApEvent> current_events, previous_events;
        for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator it = 
             current_epoch_users.begin(); it != current_epoch_users.end(); it++)
        {
          if (it->second.user_mask * request_mask)
            continue;
          current_events.push_back(it->first);
        }
        for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator it = 
              previous_epoch_users.begin(); it != 
              previous_epoch_users.end(); it++)
        {
          if (it->second.user_mask * request_mask)
            continue;
          previous_events.push_back(it->first);
        }
        rez.serialize<size_t>(current_events.size());
        for (std::vector<ApEvent>::const_iterator it = current_events.begin();
              it != current_events.end(); it++)
        {
          rez.serialize(*it);
          const EventUsers &users = current_epoch_users[*it];
          if (users.single)
          {
            rez.serialize<size_t>(1);
            users.users.single_user->pack_user(rez);
            rez.serialize(users.user_mask);
          }
          else
          {
            // Figure out how many to send
            std::vector<PhysicalUser*> to_send;
            LegionVector<FieldMask>::aligned send_masks;
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = users.users.multi_users->begin(); uit !=
                  users.users.multi_users->end(); uit++)
            {
              const FieldMask user_mask = uit->second & request_mask;
              if (!user_mask)
                continue;
              to_send.push_back(uit->first);
              send_masks.push_back(user_mask);
            }
            rez.serialize<size_t>(to_send.size());
            for (unsigned idx = 0; idx < to_send.size(); idx++)
            {
              to_send[idx]->pack_user(rez);
              rez.serialize(send_masks[idx]);
            }
          }
        }
        rez.serialize<size_t>(previous_events.size());
        for (std::vector<ApEvent>::const_iterator it = previous_events.begin();
              it != previous_events.end(); it++)
        {
          rez.serialize(*it);
          const EventUsers &users = previous_epoch_users[*it];
          if (users.single)
          {
            rez.serialize<size_t>(1);
            users.users.single_user->pack_user(rez);
            rez.serialize(users.user_mask);
          }
          else
          {
            // Figure out how many to send
            std::vector<PhysicalUser*> to_send;
            LegionVector<FieldMask>::aligned send_masks;
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = users.users.multi_users->begin(); uit !=
                  users.users.multi_users->end(); uit++)
            {
              const FieldMask user_mask = uit->second & request_mask;
              if (!user_mask)
                continue;
              to_send.push_back(uit->first);
              send_masks.push_back(user_mask);
            }
            rez.serialize<size_t>(to_send.size());
            for (unsigned idx = 0; idx < to_send.size(); idx++)
            {
              to_send[idx]->pack_user(rez);
              rez.serialize(send_masks[idx]);
            }
          }
        }
      }
      // Send the message back to get it on the wire before an 
      // invalidate might be issued
      runtime->send_view_update_response(source, rez);
      // Retake the lock in exlcusive mode to update our
      // set of remote instances
      AutoLock v_lock(view_lock);
      valid_remote_instances[source] |= request_mask;
      remote_valid_mask |= request_mask;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_update_response(Deserializer &derez,
                                                   RtUserEvent done_event,
                                                   RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      FieldMask response_mask;
      derez.deserialize(response_mask);
      std::set<ApEvent> collect_events;
      // Take the lock in exclusive mode and update all our state
      {
        AutoLock v_lock(view_lock);
        LegionMap<VersionID,FieldMask>::aligned version_updates;
        size_t num_versions;
        derez.deserialize(num_versions);
        for (unsigned idx = 0; idx < num_versions; idx++)
        {
          VersionID vid;
          derez.deserialize(vid);
          derez.deserialize(version_updates[vid]);
        }
        filter_and_add(response_mask, version_updates);
        // Current users
        size_t num_current;
        derez.deserialize(num_current);
        for (unsigned idx = 0; idx < num_current; idx++)
        {
          ApEvent current_event;
          derez.deserialize(current_event);
          size_t num_users;
          derez.deserialize(num_users);
          // See if we already have a users for this event
          LegionMap<ApEvent,EventUsers>::aligned::iterator finder = 
            current_epoch_users.find(current_event);
          if (finder != current_epoch_users.end())
          {
            // Convert to multi users if we haven't already 
            EventUsers &current_users = finder->second;
            if (current_users.single)
            {
              LegionMap<PhysicalUser*,FieldMask>::aligned *new_users = 
                new LegionMap<PhysicalUser*,FieldMask>::aligned();
              (*new_users)[current_users.users.single_user] = 
                current_users.user_mask;
              current_users.users.multi_users = new_users;
              current_users.single = false;
            }
            LegionMap<PhysicalUser*,FieldMask>::aligned &local = 
                *(current_users.users.multi_users);
            for (unsigned idx2 = 0; idx2 < num_users; idx2++)
            {
              PhysicalUser *new_user = 
                PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
              FieldMask &new_mask = local[new_user];
              derez.deserialize(new_mask);
              current_users.user_mask |= new_mask;
            }
          }
          else
          {
            EventUsers &current_users = current_epoch_users[current_event];
            if (num_users == 1)
            {
              current_users.users.single_user = 
                PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
              derez.deserialize(current_users.user_mask);
            }
            else
            {
              current_users.single = false;
              current_users.users.multi_users = 
                new LegionMap<PhysicalUser*,FieldMask>::aligned();
              LegionMap<PhysicalUser*,FieldMask>::aligned &local = 
                *(current_users.users.multi_users);
              for (unsigned idx2 = 0; idx2 < num_users; idx2++)
              {
                PhysicalUser *new_user = 
                  PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
                FieldMask &new_mask = local[new_user];
                derez.deserialize(new_mask);
                current_users.user_mask |= new_mask;
              }
            }
            // Didn't have it before so update the collect events
            if (outstanding_gc_events.find(current_event) == 
                  outstanding_gc_events.end())
            {
              outstanding_gc_events.insert(current_event);
              collect_events.insert(current_event);
            }
          }
        }
        // Previous users
        size_t num_previous;
        derez.deserialize(num_previous);
        for (unsigned idx = 0; idx < num_previous; idx++)
        {
          ApEvent previous_event;
          derez.deserialize(previous_event);
          size_t num_users;
          derez.deserialize(num_users);
          // See if we already have a users for this event
          LegionMap<ApEvent,EventUsers>::aligned::iterator finder = 
            previous_epoch_users.find(previous_event);
          if (finder != previous_epoch_users.end())
          {
            // Convert to multi users if we haven't already 
            EventUsers &previous_users = finder->second;
            if (previous_users.single)
            {
              LegionMap<PhysicalUser*,FieldMask>::aligned *new_users = 
                new LegionMap<PhysicalUser*,FieldMask>::aligned();
              (*new_users)[previous_users.users.single_user] = 
                previous_users.user_mask;
              previous_users.users.multi_users = new_users;
              previous_users.single = false;
            }
            LegionMap<PhysicalUser*,FieldMask>::aligned &local = 
                *(previous_users.users.multi_users);
            for (unsigned idx2 = 0; idx2 < num_users; idx2++)
            {
              PhysicalUser *new_user = 
                PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
              FieldMask &new_mask = local[new_user];
              derez.deserialize(new_mask);
              previous_users.user_mask |= new_mask;
            }
          }
          else
          {
            EventUsers &previous_users = previous_epoch_users[previous_event];
            if (num_users == 1)
            {
              previous_users.users.single_user = 
                PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
              derez.deserialize(previous_users.user_mask);
            }
            else
            {
              previous_users.single = false;
              previous_users.users.multi_users = 
                new LegionMap<PhysicalUser*,FieldMask>::aligned();
              LegionMap<PhysicalUser*,FieldMask>::aligned &local = 
                *(previous_users.users.multi_users);
              for (unsigned idx2 = 0; idx2 < num_users; idx2++)
              {
                PhysicalUser *new_user = 
                  PhysicalUser::unpack_user(derez, true/*add ref*/, forest);
                FieldMask &new_mask = local[new_user];
                derez.deserialize(new_mask);
                previous_users.user_mask |= new_mask;
              }
            }
            // Didn't have it before so update the collect events
            if (outstanding_gc_events.find(previous_event) == 
                  outstanding_gc_events.end())
            {
              outstanding_gc_events.insert(previous_event);
              collect_events.insert(previous_event);
            }
          }
        }
        // Update our remote valid mask
        remote_valid_mask |= response_mask;
        // Prune out the request event
#ifdef DEBUG_LEGION
        assert(remote_update_requests.find(done_event) != 
                remote_update_requests.end());
#endif
        remote_update_requests.erase(done_event);
      }
      
      if (!collect_events.empty())
      {
        std::set<RtEvent> applied_events;
        WrapperReferenceMutator mutator(applied_events);
        for (std::set<ApEvent>::const_iterator it = collect_events.begin();
              it != collect_events.end(); it++)
          defer_collect_user(*it, &mutator);
        if (!applied_events.empty())
        {
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(applied_events));
          return;
        }
        // Otherwise fall through to the normal trigger path
      }
      // Trigger our request saying everything is up to date
      // Issue any defferred collections that we might have
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_remote_update(Deserializer &derez,
                                                 AddressSpaceID source,
                                                 RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      RtUserEvent update_event;
      derez.deserialize(update_event);
      bool is_copy;
      derez.deserialize(is_copy);
      bool restrict_out;
      derez.deserialize(restrict_out);
      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      LegionColor child_color;
      derez.deserialize(child_color);
      LogicalRegion origin_handle;
      derez.deserialize(origin_handle);
      RegionNode *origin_node = forest->get_node(origin_handle);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      size_t num_versions;
      derez.deserialize(num_versions);
      FieldVersions field_versions;
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        derez.deserialize(field_versions[vid]);
      }
      FieldMask split_mask;
      derez.deserialize(split_mask);
      bool base_user;
      derez.deserialize(base_user);
      
      // Make a dummy version info for doing the analysis calls
      // and put our split mask in it
      VersionInfo dummy_version_info;
      dummy_version_info.resize(logical_node->get_depth());
      dummy_version_info.record_split_fields(logical_node, split_mask);

      std::set<RtEvent> applied_conditions;
      if (is_copy)
      {
        // Do analysis and register the user
        LegionMap<ApEvent,FieldMask>::aligned dummy_preconditions;
        // Do different things depending on whether we are a base
        // user or a user being registered above in the tree
        if (base_user)
          // Always safe to assume single copy here since we don't
          // actually use the results and assuming single copy means
          // that fewer users will potentially be filtered
          find_local_copy_preconditions(usage.redop, IS_READ_ONLY(usage),
                                      true/*single copy*/, restrict_out,
                                      user_mask, child_color, origin_node,
                                      &dummy_version_info, op_id, index, source,
                                      dummy_preconditions, applied_conditions);
        else
          find_local_copy_preconditions_above(usage.redop, IS_READ_ONLY(usage),
                                      true/*single copy*/, restrict_out,
                                      user_mask, child_color, origin_node,
                                      &dummy_version_info, op_id, index, source,
                                      dummy_preconditions, applied_conditions);
        add_local_copy_user(usage, term_event, base_user, 
                            restrict_out, child_color, origin_node,
                            &dummy_version_info, op_id, index,
                            user_mask, source, applied_conditions);
      }
      else
      {
        // Do analysis and register the user
        std::set<ApEvent> dummy_preconditions;
        // We do different things depending on whether we are the base
        // user or whether we are being registered above in the tree
        if (base_user)
          find_local_user_preconditions(usage, term_event, child_color,
                                        origin_node, &dummy_version_info, op_id,
                                        index,user_mask, dummy_preconditions, 
                                        applied_conditions);
        else
          find_local_user_preconditions_above(usage, term_event, child_color,
                                        origin_node, &dummy_version_info, op_id,
                                        index,user_mask, dummy_preconditions, 
                                        applied_conditions);
        if (IS_WRITE(usage))
          update_version_numbers(user_mask, field_versions,
                                 source, applied_conditions);
        if (add_local_user(usage, term_event, child_color, origin_node, 
                           base_user, &dummy_version_info, op_id, index, 
                           user_mask, source, applied_conditions))
        {
          WrapperReferenceMutator mutator(applied_conditions);
          defer_collect_user(term_event, &mutator);
        }
      }
      // Chain the update events
      if (!applied_conditions.empty())
        Runtime::trigger_event(update_event,
                               Runtime::merge_events(applied_conditions));
      else
        Runtime::trigger_event(update_event);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_remote_invalidate(
                          const FieldMask &invalid_mask, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock v_lock(view_lock);
        remote_valid_mask -= invalid_mask;
      }
      Runtime::trigger_event(done_event);
    }

    /////////////////////////////////////////////////////////////
    // DeferredView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredView::DeferredView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp,
                               RegionTreeNode *node, bool register_now)
      : LogicalView(ctx, did, owner_sp, node, register_now)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DeferredView::issue_deferred_copies_across(const TraversalInfo &info,
                                                     MaterializedView *dst,
                                      const std::vector<unsigned> &src_indexes,
                                      const std::vector<unsigned> &dst_indexes,
                                         ApEvent precondition, PredEvent guard,
                                         std::set<ApEvent> &postconditions)
    //--------------------------------------------------------------------------
    {
      bool perfect = true;
      FieldMask src_mask, dst_mask;
      for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
      {
        src_mask.set_bit(src_indexes[idx]);
        dst_mask.set_bit(dst_indexes[idx]);
        if (perfect && (src_indexes[idx] != dst_indexes[idx]))
          perfect = false;
      }
      // Initialize the preconditions
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      preconditions[precondition] = src_mask;
      // A seemingly common case but not the general one, if the fields
      // are in the same locations for the source and destination then
      // we can just do the normal deferred copy routine
      LegionMap<ApEvent,FieldMask>::aligned local_postconditions;
      if (perfect)
      {
        issue_deferred_copies(info, dst, src_mask, 
                              preconditions, local_postconditions, guard);
      }
      else
      {
        // Initialize the across copy helper
        CopyAcrossHelper across_helper(src_mask);
        dst->manager->initialize_across_helper(&across_helper, dst_mask, 
                                               src_indexes, dst_indexes);
        issue_deferred_copies(info, dst, src_mask, preconditions, 
                              local_postconditions, guard, &across_helper);
      }
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            local_postconditions.begin(); it != 
            local_postconditions.end(); it++)
        postconditions.insert(it->first);
    }

    /////////////////////////////////////////////////////////////
    // DeferredVersionInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredVersionInfo::DeferredVersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredVersionInfo::DeferredVersionInfo(const DeferredVersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeferredVersionInfo::~DeferredVersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredVersionInfo& DeferredVersionInfo::operator=(
                                                const DeferredVersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    /////////////////////////////////////////////////////////////
    // CompositeCopyNode
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeCopyNode::CompositeCopyNode(RegionTreeNode *node, CompositeView *v)
      : logical_node(node), view_node(v)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    CompositeCopyNode::CompositeCopyNode(const CompositeCopyNode &rhs)
      : logical_node(NULL), view_node(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeCopyNode::~CompositeCopyNode(void)
    //--------------------------------------------------------------------------
    {
      // Delete our recursive nodes 
      for (LegionMap<CompositeCopyNode*,FieldMask>::aligned::const_iterator it =
            child_nodes.begin(); it != child_nodes.end(); it++)
        delete it->first;
      child_nodes.clear();
      for (LegionMap<CompositeCopyNode*,FieldMask>::aligned::const_iterator it =
            nested_nodes.begin(); it != nested_nodes.end(); it++)
        delete it->first;
      nested_nodes.clear();
    }

    //--------------------------------------------------------------------------
    CompositeCopyNode& CompositeCopyNode::operator=(
                                                   const CompositeCopyNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::add_child_node(CompositeCopyNode *child,
                                           const FieldMask &child_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child_nodes.find(child) == child_nodes.end());
#endif
      child_nodes[child] = child_mask; 
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::add_nested_node(CompositeCopyNode *nested,
                                            const FieldMask &nested_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(nested->view_node != NULL);
      assert(nested_nodes.find(nested) == nested_nodes.end());
#endif
      nested_nodes[nested] = nested_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::add_source_view(LogicalView *source_view,
                                            const FieldMask &source_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(source_views.find(source_view) == source_views.end());
#endif
      source_views[source_view] = source_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::add_reduction_view(ReductionView *reduction_view,
                                               const FieldMask &reduction_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_views.find(reduction_view) == reduction_views.end());
#endif
      reduction_views[reduction_view] = reduction_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::issue_copies(const TraversalInfo &traversal_info,
                              MaterializedView *dst, const FieldMask &copy_mask,
                              VersionTracker *src_version_tracker,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                        PredEvent pred_guard, CopyAcrossHelper *helper) const
    //--------------------------------------------------------------------------
    {
      // We're doing the painter's algorithm
      // First traverse any nested composite instances and issue
      // copies from them since they are older than us
      const LegionMap<ApEvent,FieldMask>::aligned *local_preconditions = 
                                                          &preconditions;
      // Temporary data structures in case we need them
      LegionMap<ApEvent,FieldMask>::aligned temp_local;
      bool temp_copy = false;
      if (!nested_nodes.empty())
      {
        LegionMap<ApEvent,FieldMask>::aligned nested_postconditions;
        issue_nested_copies(traversal_info, dst, copy_mask, src_version_tracker,
                      preconditions, nested_postconditions, pred_guard, helper);
        // Add the nested postconditions to our postconditions
        postconditions.insert(nested_postconditions.begin(),
                              nested_postconditions.end());
        // See if we need to update our local or child preconditions
        if (!source_views.empty() || !child_nodes.empty() || 
            !reduction_views.empty())
        {
          // Makes new local_preconditions
          local_preconditions = &temp_local;
          temp_local = preconditions;
          temp_copy = true;
          temp_local.insert(nested_postconditions.begin(),
                            nested_postconditions.end());
        }
      }
      // Next issue copies from any of our source views 
      if (!source_views.empty())
      {
        // Uses local_preconditions
        LegionMap<ApEvent,FieldMask>::aligned local_postconditions;
        issue_local_copies(traversal_info, dst, copy_mask, src_version_tracker,
                           *local_preconditions, local_postconditions, 
                           pred_guard, helper);
        postconditions.insert(local_postconditions.begin(),
                              local_postconditions.end());
        // Makes new local_preconditions
        if (!child_nodes.empty() || !reduction_views.empty())
        {
          if (!temp_copy)
          {
            local_preconditions = &temp_local;
            temp_local = preconditions;
            temp_copy = true;
          }
          temp_local.insert(local_postconditions.begin(),
                            local_postconditions.end());
        }
      }
      // Traverse our children and issue any copies to them
      if (!child_nodes.empty())
      {
        // Uses local_preconditions
        LegionMap<ApEvent,FieldMask>::aligned child_postconditions;
        issue_child_copies(traversal_info, dst, copy_mask, src_version_tracker,
            *local_preconditions, child_postconditions, postreductions, 
            pred_guard, helper);
        postconditions.insert(child_postconditions.begin(),
                              child_postconditions.end());
        // Makes new local_preconditions
        if (!reduction_views.empty())
        {
          if (!temp_copy)
          {
            local_preconditions = &temp_local;
            temp_local = preconditions;
            temp_copy = true;
          }
          temp_local.insert(child_postconditions.begin(),
                            child_postconditions.end());
        }
      }
      // Finally apply any reductions that we have on the way back up
      if (!reduction_views.empty())
      {
        // Uses local_preconditions
        issue_reductions(traversal_info, dst, copy_mask, src_version_tracker,
                   *local_preconditions, postreductions, pred_guard, helper);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::copy_to_temporary(const TraversalInfo &info,
                            MaterializedView *dst, const FieldMask &copy_mask,
                            VersionTracker *src_version_tracker,
                const LegionMap<ApEvent,FieldMask>::aligned &dst_preconditions,
                      LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                      PredEvent pred_guard, AddressSpaceID local_space, 
                      bool restrict_out)
    //--------------------------------------------------------------------------
    {
      // Make a temporary instance and issue copies to it
      // then copy from the temporary instance to the target
      MaterializedView *temporary_dst = 
        info.op->create_temporary_instance(dst->manager,info.index, copy_mask);
      // Get the corresponding sub_view to the destination
      if (temporary_dst->logical_node != dst->logical_node)
      {
        std::vector<LegionColor> colors;
        RegionTreeNode *dst_node = dst->logical_node;
        do 
        {
#ifdef DEBUG_LEGION
          assert(dst_node->get_depth() > 
                  temporary_dst->logical_node->get_depth());
#endif
          colors.push_back(dst_node->get_color());
          dst_node = dst_node->get_parent();
        } 
        while (dst_node != temporary_dst->logical_node);
#ifdef DEBUG_LEGION
        assert(!colors.empty());
#endif
        while (!colors.empty())
        {
          temporary_dst = 
            temporary_dst->get_materialized_subview(colors.back());
          colors.pop_back();
        }
#ifdef DEBUG_LEGION
        assert(temporary_dst->logical_node == dst->logical_node);
#endif
      }
      LegionMap<ApEvent,FieldMask>::aligned empty_pre, local_pre, local_reduce;
      issue_copies(info, temporary_dst, copy_mask, src_version_tracker,
         empty_pre, local_pre, local_reduce, pred_guard, NULL/*across helper*/);
      // Now that we've done all the copies to the temporary instance
      // we can compute the rest of the destination preconditions
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/, 
                                   false/*single copy*/, restrict_out,
                                   copy_mask, &info.version_info,
                                   info.op->get_unique_op_id(), info.index,
                                   local_space, local_pre, 
                                   info.map_applied_events);
      // Merge the destination preconditions
      if (!dst_preconditions.empty())
        local_pre.insert(dst_preconditions.begin(), dst_preconditions.end());
      // Also merge any local reduces into the preconditons
      if (!local_reduce.empty())
        local_pre.insert(local_reduce.begin(), local_reduce.end());
      // Compute the event sets
      LegionList<EventSet>::aligned event_sets;
      RegionTreeNode::compute_event_sets(copy_mask, local_pre, event_sets);
      // Iterate over the event sets, for each event set, record a user
      // on the temporary for being done with copies there, issue a copy
      // from the temporary to the original destination, and then record
      // users on both instances, put the done event in the postcondition set
      for (LegionList<EventSet>::aligned::const_iterator it = 
            event_sets.begin(); it != event_sets.end(); it++)
      {
        ApEvent copy_pre;
        if (it->preconditions.size() == 1)
          copy_pre = *(it->preconditions.begin());
        else if (!it->preconditions.empty())
          copy_pre = Runtime::merge_events(it->preconditions);
        // Make a user for when the destination is up to date
        if (copy_pre.exists())
          temporary_dst->add_copy_user(0/*redop*/, copy_pre, 
              &info.version_info, info.op->get_unique_op_id(), info.index,
              it->set_mask, false/*reading*/, false/*restrict out*/, 
              local_space, info.map_applied_events); 
        // Perform the copy
        std::vector<CopySrcDstField> src_fields;
        std::vector<CopySrcDstField> dst_fields;
        temporary_dst->copy_from(it->set_mask, src_fields);
        dst->copy_to(it->set_mask, dst_fields);
#ifdef DEBUG_LEGION
        assert(!src_fields.empty());
        assert(!dst_fields.empty());
        assert(src_fields.size() == dst_fields.size());
#endif
        ApEvent copy_post = dst->logical_node->issue_copy(info.op, src_fields,
                              dst_fields, copy_pre, pred_guard, logical_node);
        if (copy_post.exists())
        {
          dst->add_copy_user(0/*redop*/, copy_post, &info.version_info,
                             info.op->get_unique_op_id(), info.index,
                             it->set_mask, false/*reading*/, 
                             false/*restrict out*/, local_space,
                             info.map_applied_events);
          temporary_dst->add_copy_user(0/*redop*/, copy_post, 
                             &info.version_info, info.op->get_unique_op_id(),
                             info.index, it->set_mask, true/*reading*/,
                             false/*restrict out*/, local_space, 
                             info.map_applied_events);
          postconditions[copy_post] = it->set_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::issue_nested_copies(
                              const TraversalInfo &traversal_info,
                              MaterializedView *dst, const FieldMask &copy_mask,
                              VersionTracker *src_version_tracker,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                        PredEvent pred_guard, CopyAcrossHelper *helper) const
    //--------------------------------------------------------------------------
    {
      FieldMask nested_mask;
      LegionMap<ApEvent,FieldMask>::aligned postreductions;
      for (LegionMap<CompositeCopyNode*,FieldMask>::aligned::const_iterator it =
            nested_nodes.begin(); it != nested_nodes.end(); it++)
      {
        const FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        nested_mask |= overlap;
#ifdef DEBUG_LEGION
        assert(it->first->view_node != NULL);
#endif
        it->first->issue_copies(traversal_info, dst, overlap, 
            it->first->view_node, preconditions, postconditions,
            postreductions, pred_guard, helper);
      }
      // We have to merge everything back together into postconditions here
      // including the reductions because they have to finish before we issue
      // any more copies to the destination
      if (!!nested_mask)
      {
        if (!postreductions.empty())
          postconditions.insert(postreductions.begin(), postreductions.end());
        // Compute the event sets
        LegionList<EventSet>::aligned event_sets;
        RegionTreeNode::compute_event_sets(nested_mask, 
                                           postconditions, event_sets);
        // Clear out the post conditions and put the merge
        // of the event sets there
        postconditions.clear();
        for (LegionList<EventSet>::aligned::const_iterator it = 
              event_sets.begin(); it != event_sets.end(); it++)
        {
          ApEvent post;
          if (it->preconditions.size() == 1)
            post = *(it->preconditions.begin());
          else if (!it->preconditions.empty())
            post = Runtime::merge_events(it->preconditions);
          if (post.exists())
          {
            // post is not guaranteed to be unique!
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
              postconditions.find(post);
            if (finder == postconditions.end())
              postconditions[post] = it->set_mask;
            else
              finder->second |= it->set_mask;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::issue_local_copies(const TraversalInfo &info,
                              MaterializedView *dst, FieldMask copy_mask,
                              VersionTracker *src_version_tracker,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                        PredEvent guard, CopyAcrossHelper *across_helper) const
    //--------------------------------------------------------------------------
    {
      // First check to see if the target is already valid
      {
        PhysicalManager *dst_manager = dst->get_manager();
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              source_views.begin(); it != source_views.end(); it++)
        {
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_LEGION
          assert(it->first->is_materialized_view());
#endif
          if (it->first->as_materialized_view()->manager == dst_manager)
          {
            copy_mask -= it->second;
            if (!copy_mask)
              return;
          }
        }
      }
      LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
      LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
      // Sort the instances
      dst->logical_node->sort_copy_instances(info, dst, copy_mask, 
                            source_views, src_instances, deferred_instances);
      if (!src_instances.empty())
      {
        // This has all our destination preconditions
        // Only issue copies from fields which have values
        FieldMask actual_copy_mask;
        LegionMap<ApEvent,FieldMask>::aligned src_preconditions;
        const AddressSpaceID local_space = 
          logical_node->context->runtime->address_space;
        for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
              it = src_instances.begin(); it != src_instances.end(); it++)
        {
          it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                             true/*single copy*/,
                                             false/*restrict out*/,
                                             it->second, src_version_tracker,
                                             info.op->get_unique_op_id(),
                                             info.index, local_space, 
                                             src_preconditions,
                                             info.map_applied_events);
          actual_copy_mask |= it->second;
        }
        // Move in any preconditions that overlap with our set of fields
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              preconditions.begin(); it != preconditions.end(); it++)
        {
          FieldMask overlap = it->second & actual_copy_mask;
          if (!overlap)
            continue;
          LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            src_preconditions.find(it->first);
          if (finder == src_preconditions.end())
            src_preconditions[it->first] = overlap;
          else
            finder->second |= overlap;
        }
        // issue the grouped copies and put the result in the postconditions
        // We are the intersect
        dst->logical_node->issue_grouped_copies(info, dst,false/*restrict out*/,
                                 guard, src_preconditions, actual_copy_mask, 
                                 src_instances, src_version_tracker, 
                                 postconditions, across_helper, logical_node);
      }
      if (!deferred_instances.empty())
      {
        for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator it = 
              deferred_instances.begin(); it != deferred_instances.end(); it++)
        {
          LegionMap<ApEvent,FieldMask>::aligned deferred_preconditions;
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pre_it =
                preconditions.begin(); pre_it != preconditions.end(); pre_it++)
          {
            const FieldMask overlap = pre_it->second & it->second;
            if (!overlap)
              continue;
            deferred_preconditions[pre_it->first] = overlap;
          }
          it->first->issue_deferred_copies(info, dst, it->second,
              deferred_preconditions, postconditions, guard, across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::issue_child_copies(
                              const TraversalInfo &traversal_info,
                              MaterializedView *dst, const FieldMask &copy_mask,
                              VersionTracker *src_version_tracker,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                        PredEvent pred_guard, CopyAcrossHelper *helper) const
    //--------------------------------------------------------------------------
    {
      bool multiple_children = false;
      FieldMask single_child_mask;
      for (LegionMap<CompositeCopyNode*,FieldMask>::aligned::const_iterator it =
            child_nodes.begin(); it != child_nodes.end(); it++)
      {
        const FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        if (!multiple_children)
          multiple_children = !!(single_child_mask & overlap);
        single_child_mask |= overlap;
        it->first->issue_copies(traversal_info, dst, overlap, 
            src_version_tracker, preconditions, postconditions,
            postreductions, pred_guard, helper);
      }
      // Merge the postconditions from all the children to build a 
      // common output event for each field if there were multiple children
      // No need to merge the reduction postconditions, they just continue
      // flowing up the tree
      if (!postconditions.empty() && multiple_children)
      {
        LegionList<EventSet>::aligned event_sets;
        // Have to do the merge for all fields
        RegionTreeNode::compute_event_sets(single_child_mask,
                                           postconditions, event_sets);
        // Clear out the post conditions and put the merge
        // of the event sets there
        postconditions.clear();
        for (LegionList<EventSet>::aligned::const_iterator it = 
              event_sets.begin(); it != event_sets.end(); it++)
        {
          ApEvent post;
          if (it->preconditions.size() == 1)
            post = *(it->preconditions.begin());
          else if (!it->preconditions.empty())
            post = Runtime::merge_events(it->preconditions);
          if (post.exists())
          {
            // post is not guaranteed to be unique!
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
              postconditions.find(post);
            if (finder == postconditions.end())
              postconditions[post] = it->set_mask;
            else
              finder->second |= it->set_mask;
          }
        }  
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopyNode::issue_reductions(const TraversalInfo &info,
                              MaterializedView *dst, const FieldMask &copy_mask,
                              VersionTracker *src_version_tracker,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                    PredEvent pred_guard, CopyAcrossHelper *across_helper) const
    //--------------------------------------------------------------------------
    {
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        const FieldMask overlap = copy_mask & it->second;
        if (!overlap)
          continue;
        // This is precise but maybe unecessary
        std::set<ApEvent> local_preconditions;
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pre_it = 
              preconditions.begin(); pre_it != preconditions.end(); pre_it++)
        {
          FieldMask pre_overlap = overlap & pre_it->second;
          if (!pre_overlap)
            continue;
          local_preconditions.insert(pre_it->first);
        }
        ApEvent reduce_event = it->first->perform_deferred_reduction(dst,
            overlap, src_version_tracker, local_preconditions, info.op,
            info.index, pred_guard, across_helper, 
            (dst->logical_node == it->first->logical_node) ?
              NULL : it->first->logical_node, info.map_applied_events);
        if (reduce_event.exists())
          postreductions[reduce_event] = overlap;
      }
    }

    /////////////////////////////////////////////////////////////
    // CompositeCopier 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeCopier::CompositeCopier(const FieldMask &copy_mask)
      : destination_valid(copy_mask)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCopier::CompositeCopier(const CompositeCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeCopier::~CompositeCopier(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCopier& CompositeCopier::operator=(const CompositeCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeCopier::filter_written_fields(RegionTreeNode *node,
                                                FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      // Traverse up the tree and remove any writen fields 
      // that have been written at this level or a parent
      LegionMap<RegionTreeNode*,FieldMask>::aligned::const_iterator finder = 
        written_nodes.find(node);
      while (finder != written_nodes.end())
      {
        mask -= finder->second;
        if (!mask)
          return;
        node = node->get_parent();
        if (node == NULL)
          break;
        finder = written_nodes.find(node);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeCopier::and_written_fields(RegionTreeNode *node,
                                             FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      LegionMap<RegionTreeNode*,FieldMask>::aligned::const_iterator finder = 
        written_nodes.find(node);
      FieldMask written_mask;
      while (finder != written_nodes.end())
      {
        written_mask |= finder->second;
        node = node->get_parent();
        if (node == NULL)
          break;
        finder = written_nodes.find(node);
      }
      mask &= written_mask;
    }

    //--------------------------------------------------------------------------
    void CompositeCopier::record_written_fields(RegionTreeNode *node,
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<RegionTreeNode*,FieldMask>::aligned::iterator finder = 
        written_nodes.find(node);
      if (finder == written_nodes.end())
        written_nodes[node] = mask;
      else
        finder->second |= mask;
    }

    /////////////////////////////////////////////////////////////
    // CompositeBase 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeBase::CompositeBase(LocalLock &r)
      : base_lock(r)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeBase::~CompositeBase(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeCopyNode* CompositeBase::construct_copy_tree(MaterializedView *dst,
                                                   RegionTreeNode *logical_node,
                                                   FieldMask &copy_mask,
                                                   FieldMask &locally_complete,
                                                   FieldMask &dominate_capture,
                                                   CompositeCopier &copier,
                                                   CompositeView *owner)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!copy_mask);
#endif
      // First check to see if we've already written to this node
      copier.filter_written_fields(logical_node, copy_mask);
      // If we've already written all our fields, no need to traverse here
      if (!copy_mask)
        return NULL;
      // If we get here, we're going to return something
      CompositeCopyNode *result = new CompositeCopyNode(logical_node, owner);
      // Do the ready check first
      perform_ready_check(copy_mask);
      // Figure out which children we need to traverse because they intersect
      // with the dst instance and any reductions that will need to be applied
      LegionMap<CompositeNode*,FieldMask>::aligned children_to_traverse;
      // We have to capture any local dirty data here
      FieldMask local_capture, local_dominate;
      const bool tested_all_children = perform_construction_analysis(dst,
                                    logical_node, copy_mask, local_capture,
                                    dominate_capture, local_dominate,
                                    copier, result, children_to_traverse);
      // Traverse all the children and see if our children make us
      // complete in any way so we can avoid capturing locally
      if (!children_to_traverse.empty())
      {
        // We can be locally complete in two ways:
        // 1. We can have a child that is complete for us (e.g. we are region
        //      and one of our partitions is complete)
        FieldMask complete_child;
        // 2. We are ourselves complete and all our children are written (note
        //      that this also is alright if the children were written earlier)
        //      Note we can actually only worry about this for interfering
        //      children if we know the composite instance contains all the
        //      interfering children we intersect with for the target region.
        const bool is_complete = tested_all_children && 
                                  logical_node->is_complete();
        if (is_complete)
          locally_complete = copy_mask;
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
              children_to_traverse.begin(); it != 
              children_to_traverse.end(); it++)
        {
          CompositeCopyNode *child = it->first->construct_copy_tree(dst,
              it->first->logical_node, it->second, complete_child,
              dominate_capture, copier);
          if (child != NULL)
          {
            result->add_child_node(child, it->second);
            if (is_complete && !!locally_complete)
              copier.and_written_fields(it->first->logical_node, 
                                        locally_complete);
          }
          else if (is_complete && !!locally_complete)
            locally_complete.clear();
        }
        // Check to see if we are complete in any way, if we are then we
        // can filter our local_capture mask and report up the tree that
        // we are complete
        if (!!complete_child)
        {
          local_capture -= complete_child;
          copier.record_written_fields(logical_node, complete_child);
        }
        if (is_complete && !!locally_complete)
        {
          local_capture -= locally_complete;
          copier.record_written_fields(logical_node, locally_complete);
        }
      }
      // If we have to capture any data locally, do that now
      if (!!local_capture)
      {
        LegionMap<LogicalView*,FieldMask>::aligned local_valid_views;
        find_valid_views(local_capture, local_dominate, 
                         local_valid_views, true/*need lock*/);
        LegionMap<CompositeView*,FieldMask>::aligned nested_composite_views;
        // Track which fields we see dirty updates for other instances
        // versus for the destination instance
        FieldMask other_dirty, destination_dirty;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              local_valid_views.begin(); it != local_valid_views.end(); it++)
        {
          if (it->first->is_composite_view())
          {
            nested_composite_views[it->first->as_composite_view()] = it->second;
            continue;
          }
          // Record that we captured a non-composite view for these fields
          if (!!local_capture)
            local_capture -= it->second; 
          result->add_source_view(it->first, it->second); 
          if (it->first->is_materialized_view())
          {
            MaterializedView *mat_view = it->first->as_materialized_view();
            // Check to see if this is the destination instance or not
            if (mat_view->get_manager() == dst->get_manager())
              destination_dirty |= it->second;
            else
              other_dirty |= it->second;
          }
          else
            other_dirty |= it->second;
        }
        // Record that we have writes from any of the fields we have 
        // actualy source instances (e.g. materialized or fills)
        copier.record_written_fields(logical_node, 
                                     other_dirty | destination_dirty);
        // Also tell the copier whether the destination instance is
        // no longer valid or whether it contains dirty data
        if (!!destination_dirty)
        {
          copier.update_destination_dirty_fields(destination_dirty);
          // No need to worry about issuing copies for these
          // fields because the destination is already valid
          other_dirty -= destination_dirty;
        }
        // Filter any fields which were not valid for the target
        if (!!other_dirty)
          copier.filter_destination_valid_fields(other_dirty);
        // If there are any fields we didn't capture then see if we have
        // a nested composite instance we need to traverse
        if (!nested_composite_views.empty() && !!local_capture)
        {
          for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
                nested_composite_views.begin(); it != 
                nested_composite_views.end(); it++)
          {
            FieldMask overlap = it->second & local_capture;
            if (!overlap)
              continue;
            local_capture -= overlap;
            FieldMask dummy_complete_below;
            FieldMask dominate = overlap;
            CompositeCopyNode *nested = it->first->construct_copy_tree(dst,
                                    it->first->logical_node, overlap, 
                                    dummy_complete_below, dominate, 
                                    copier, it->first); 
            if (nested != NULL)
              result->add_nested_node(nested, overlap);
            if (!local_capture)
              break;
          }
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool CompositeBase::perform_construction_analysis(MaterializedView *dst,
                                                  RegionTreeNode *logical_node,
                                                  const FieldMask &copy_mask,
                                                  FieldMask &local_capture,
                                                  FieldMask &dominate_capture,
                                                  FieldMask &local_dominate,
                                                  CompositeCopier &copier,
                                                  CompositeCopyNode *result,
             LegionMap<CompositeNode*,FieldMask>::aligned &children_to_traverse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!local_capture);
      assert(!local_dominate);
      assert(children_to_traverse.empty());
#endif
      // need this in read only to touch the children data structure
      // and the list of reduction views
      const size_t all_children = logical_node->get_num_children(); 
      AutoLock b_lock(base_lock,1,false/*exclusive*/);
      const bool tested_all_children = (children.size() == all_children);
      if (children.empty())
      {
        if (!!dominate_capture)
        {
          local_dominate = copy_mask & dominate_capture;
          if (!!local_dominate)
            dominate_capture -= local_dominate;
        }
      }
      else if (dst->logical_node == logical_node)
      {
        // Handle a common case of target and logical region 
        // being the same, e.g. closing to this root
        // We know all the children interfere and are dominated by us
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          const FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          children_to_traverse[it->first] = overlap;
        }
        if (!!dominate_capture)
        {
          local_dominate = copy_mask & dominate_capture;
          if (!!local_dominate)
            dominate_capture -= local_dominate;
        }
      }
      else if ((dst->logical_node->get_parent() == logical_node) &&
                !logical_node->is_region() &&
                logical_node->as_partition_node()->row_source->is_disjoint())
      {
        // If we're at a partition node and the destination is one of our
        // children and we know the partition is disjoint then we just
        // have to look for the destination node in the set of a 
        // composite nodes
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if (it->first->logical_node != dst->logical_node)
            continue;
          // Once we find the node we can break out no matter what
          const FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            break;
          children_to_traverse[it->first] = overlap;
          break;
        }
      }
      else if (!!dominate_capture && !(dominate_capture * copy_mask))
      {
        // See if we have exactly one dominating child that allows us
        // to continue passing dominated fields down to that child so 
        // we don't have to do the dominate analysis at this node
        FieldMask single_dominated;
        FieldMask invalid_dominated;
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          const FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          // Skip any nodes that don't even intersect, they don't matter
          if (!it->first->logical_node->intersects_with(dst->logical_node,
                                                        false/*computes*/))
            continue;
          children_to_traverse[it->first] = overlap;
          FieldMask dom_overlap = overlap & dominate_capture;
          // If this doesn't overlap with the dominating capture fields
          // then we don't need to worry about it anymore
          if (!dom_overlap)
            continue;
          // We can remove any fields that have already been invalidated
          if (!!invalid_dominated)
          {
            dom_overlap -= invalid_dominated;
            if (!dom_overlap)
              continue;
          }
          if (it->first->logical_node->dominates(dst->logical_node))
          {
            // See if we are the first or duplicate dominating child 
            FieldMask duplicate_dom = single_dominated & dom_overlap;
            if (!!duplicate_dom)
            {
              invalid_dominated |= duplicate_dom;
              single_dominated |= (dom_overlap - duplicate_dom);
            }
            else
              single_dominated |= dom_overlap; 
          }
          else // we intersect but don't dominate, so any fields are invalidated
            invalid_dominated |= dom_overlap; 
        }
        // Remove any invalid fields from the single dominated
        if (!!single_dominated && !!invalid_dominated)
          single_dominated -= invalid_dominated;
        // Our local dominate are the ones for this copy initially
        local_dominate = dominate_capture & copy_mask;
        // Remove these from the dominate mask
        dominate_capture -= copy_mask;
        if (!!single_dominated)
        {
          // We have to handle any fields that are not single dominated
          local_dominate -= single_dominated;
          // Put the single dominated fields back into the dominate capture mask
          dominate_capture |= single_dominated;
        }
      } 
      else
      {
        // There are no remaining dominate fields, so we just need to 
        // look for interfering children
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          const FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          // Skip any nodes that don't even intersect, they don't matter
          if (!it->first->logical_node->intersects_with(dst->logical_node,
                                                        false/*computes*/))
            continue;
          children_to_traverse[it->first] = overlap;
        }
      }
      // Any dirty fields that we have here also need to be captured locally
      // This includes any reduction fields which we are going to be reducing to
      local_capture = (dirty_mask | reduction_mask) & copy_mask;
      // Always include our local dominate in the local capture
      if (!!local_dominate)
        local_capture |= local_dominate;
      if (!!reduction_mask)
      {
        FieldMask reduc_overlap = reduction_mask & copy_mask;
        if (!!reduc_overlap)
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            const FieldMask overlap = it->second & copy_mask; 
            if (!overlap)
              continue;
            result->add_reduction_view(it->first, overlap);
            copier.update_reduction_fields(overlap);
          }
        }
      }
      return tested_all_children;
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeBase::find_child_node(RegionTreeNode *child)
    //--------------------------------------------------------------------------
    {
      AutoLock b_lock(base_lock,1,false/*exclusive*/);
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->first->logical_node == child)
          return it->first;
      }
      // should never get here
      assert(false);
      return NULL;
    }

    /////////////////////////////////////////////////////////////
    // CompositeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(RegionTreeForest *ctx, DistributedID did,
                              AddressSpaceID owner_proc, RegionTreeNode *node,
                              DeferredVersionInfo *info, ClosedNode *tree, 
                              InnerContext *context, bool register_now)
      : DeferredView(ctx, encode_composite_did(did), owner_proc, 
                     node, register_now), CompositeBase(view_lock),
        version_info(info), closed_tree(tree), owner_context(context)
    {
      // Add our references
      version_info->add_reference();
      closed_tree->add_reference();
      owner_context->add_reference();
#ifdef DEBUG_LEGION
      assert(owner_context != NULL);
      assert(closed_tree != NULL);
      assert(closed_tree->node == node);
#endif
#ifdef LEGION_GC
      log_garbage.info("GC Composite View %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(const CompositeView &rhs)
      : DeferredView(NULL, 0, 0, NULL, false), CompositeBase(view_lock),
        version_info(NULL), closed_tree(NULL), owner_context(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeView::~CompositeView(void)
    //--------------------------------------------------------------------------
    {
      // Delete our children
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        delete (it->first);
      children.clear();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      }
      valid_views.clear();
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete (it->first);
      }
      nested_composite_views.clear();
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete (it->first);
      }
      reduction_views.clear();
      // Remove our references and delete if necessary
      if (version_info->remove_reference())
        delete version_info;
      // Remove our references and delete if necessary
      if (closed_tree->remove_reference())
        delete closed_tree;
      // Remove the reference on our context
      if (owner_context->remove_reference())
        delete owner_context;
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
    CompositeView* CompositeView::clone(const FieldMask &clone_mask,
         const LegionMap<CompositeView*,FieldMask>::aligned &replacements) const
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = context->runtime; 
      DistributedID result_did = runtime->get_available_distributed_id();
      CompositeView *result = new CompositeView(context, result_did,
          runtime->address_space, logical_node, version_info, closed_tree, 
          owner_context, true/*register now*/);
      // Clone the children
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask overlap = it->second & clone_mask;
        if (!overlap)
          continue;
        it->first->clone(result, overlap);
      }
      FieldMask dirty_overlap = dirty_mask & clone_mask;
      if (!!dirty_overlap)
      {
        result->record_dirty_fields(dirty_overlap);
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator 
              it = valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = it->second & dirty_overlap;
          if (!overlap)
            continue;
          result->record_valid_view(it->first, overlap);
        }
      }
      // Can just insert the replacements directly
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
            replacements.begin(); it != replacements.end(); it++)
        result->record_valid_view(it->first, it->second);
      FieldMask reduc_overlap = reduction_mask & clone_mask;
      if (!!reduc_overlap)
      {
        result->record_reduction_fields(reduc_overlap);
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
              reduction_views.begin(); it != reduction_views.end(); it++)
        {
          FieldMask overlap = it->second & reduc_overlap;
          if (!overlap)
            continue;
          result->record_reduction_view(it->first, overlap);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // No need to do anything
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->first->notify_valid(mutator, true/*root*/);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->first->notify_invalid(mutator, true/*root*/);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Don't take the lock, it's alright to have duplicate sends
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize<UniqueID>(owner_context->get_context_uid());
        bool is_region = logical_node->is_region();
        rez.serialize(is_region);
        if (is_region)
          rez.serialize(logical_node->as_region_node()->handle);
        else
          rez.serialize(logical_node->as_partition_node()->handle);
        version_info->pack_version_numbers(rez);
        closed_tree->pack_closed_node(rez);
        pack_composite_view(rez);
      }
      runtime->send_composite_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    LogicalView* CompositeView::get_subview(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Composite views don't need subviews
      return this;
    }

    //--------------------------------------------------------------------------
    void CompositeView::prune(ClosedNode *new_tree, FieldMask &valid_mask,
                     LegionMap<CompositeView*,FieldMask>::aligned &replacements, 
                     unsigned prune_depth)
    //--------------------------------------------------------------------------
    {
      if (prune_depth >= LEGION_PRUNE_DEPTH_WARNING)
        REPORT_LEGION_WARNING(LEGION_WARNING_PRUNE_DEPTH_EXCEEDED,
                        "WARNING: Composite View Tree has depth %d which "
                        "is larger than LEGION_PRUNE_DEPTH_WARNING of %d. "
                        "Please report this use case to the Legion developers "
                        "mailing list as it could be an important "
                        "runtime performance bug.", prune_depth,
                        LEGION_PRUNE_DEPTH_WARNING)
      // Figure out which fields are not dominated
      FieldMask non_dominated = valid_mask;
      new_tree->filter_dominated_fields(closed_tree, non_dominated);
      FieldMask dominated = valid_mask - non_dominated;
      if (!!dominated)
      {
        // If we had any dominated fields then we try to prune our
        // deferred valid views and put the results directly into 
        // the replacements
        for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
        {
          FieldMask overlap = it->second & dominated;
          if (!overlap)
            continue;
          it->first->prune(new_tree, overlap, replacements, prune_depth+1);
          if (!!overlap)
          {
            // Some fields are still valid so add them to the replacements
            LegionMap<CompositeView*,FieldMask>::aligned::iterator finder =
              replacements.find(it->first);
            if (finder == replacements.end())
              replacements[it->first] = overlap;
            else
              finder->second |= overlap;
          }
        }
        // Any fields that were dominated are no longer valid
        valid_mask -= dominated;
        // If all fields were dominated then we are done
        if (!valid_mask)
          return;
      }
      // For any non-dominated fields, see if any of our composite views change
      FieldMask changed_mask;
      LegionMap<CompositeView*,FieldMask>::aligned local_replacements;
      for (LegionMap<CompositeView*,FieldMask>::aligned::iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
      {
        FieldMask overlap = it->second & non_dominated;
        if (!overlap)
          continue;
        FieldMask still_valid = overlap;
        it->first->prune(new_tree, still_valid, 
                         local_replacements, prune_depth+1);
        // See if any fields were pruned, if so they are changed
        FieldMask changed = overlap - still_valid;
        if (!!changed)
          changed_mask |= changed;
      }
      if (!local_replacements.empty())
      {
        for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
              local_replacements.begin(); it != local_replacements.end(); it++)
          changed_mask |= it->second;
      }
      if (!!changed_mask)
      {
        CompositeView *view = clone(changed_mask, local_replacements);
        view->finalize_capture(false/*need prune*/);
        replacements[view] = changed_mask;
        // Any fields that changed are no longer valid
        valid_mask -= changed_mask;
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              FieldMask copy_mask,
                                              const RestrictInfo &restrict_info,
                                              bool restrict_out)
    //--------------------------------------------------------------------------
    {
      CompositeCopier copier(copy_mask);
      FieldMask top_locally_complete;
      FieldMask dominate_capture(copy_mask);
      CompositeCopyNode *copy_tree = construct_copy_tree(dst, logical_node,
          copy_mask, top_locally_complete, dominate_capture, copier, this);
#ifdef DEBUG_LEGION
      assert(copy_tree != NULL);
#endif
      copy_mask -= copier.get_already_valid_fields();       
      // If we have any reduction fields though we still need to 
      copy_mask |= copier.get_reduction_fields();
      // issue copies for them
      if (!copy_mask)
      {
        delete copy_tree;
        return;
      }
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      LegionMap<ApEvent,FieldMask>::aligned postconditions;
      
      if (restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        restrict_mask &= copy_mask;
        if (!!restrict_mask)
        {
          ApEvent restrict_pre = info.op->get_restrict_precondition();
          preconditions[restrict_pre] = restrict_mask;
        }
      }
      // We have to do the copy for the remaining fields, see if we
      // need to make a temporary instance to avoid overwriting data
      // in the destination instance as part of the painter's algorithm
      if (copier.has_dirty_destination_fields())
      {
        // We need to make a temporary instance 
        copy_tree->copy_to_temporary(info, dst, copy_mask, this,
                                     preconditions, postconditions, 
                                     PredEvent::NO_PRED_EVENT,
                                     local_space, restrict_out);
      }
      else
      {
        // Compute the rest of the destination preconditions now
        dst->find_copy_preconditions(0/*redop*/, false/*reading*/, 
                                     false/*single copy*/, restrict_out,
                                     copy_mask, &info.version_info, 
                                     info.op->get_unique_op_id(), info.index,
                                     local_space, preconditions, 
                                     info.map_applied_events);  
        LegionMap<ApEvent,FieldMask>::aligned postreductions;
        // No temporary instance necessary here
        copy_tree->issue_copies(info, dst, copy_mask, this, 
            preconditions, postconditions, postreductions, 
            PredEvent::NO_PRED_EVENT, NULL);
        if (!postreductions.empty())
          postconditions.insert(postreductions.begin(),
                                postreductions.end());
      }
      delete copy_tree;
      // If we have no postconditions, then we are done
      if (postconditions.empty())
        return;
      // Sort these into event sets and add a destination user for each merge
      LegionList<EventSet>::aligned postcondition_sets;
      RegionTreeNode::compute_event_sets(copy_mask, postconditions,
                                         postcondition_sets);
      
      // Now we can register our dependences on the target
      for (LegionList<EventSet>::aligned::const_iterator it = 
            postcondition_sets.begin(); it != postcondition_sets.end(); it++)
      {
        if (it->preconditions.empty())
          continue;
        ApEvent done_event;
        if (it->preconditions.size() == 1)
          done_event = *(it->preconditions.begin());
        else
          done_event = Runtime::merge_events(it->preconditions);
        dst->add_copy_user(0/*redop*/, done_event, &info.version_info,
                           info.op->get_unique_op_id(), info.index,
                           it->set_mask, false/*reading*/, restrict_out,
                           local_space, info.map_applied_events);
      }
      if (restrict_out && restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              postconditions.begin(); it != postconditions.end(); it++)
        {
          if (it->second * restrict_mask)
            continue;
          info.op->record_restrict_postcondition(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              FieldMask copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                          PredEvent pred_guard, CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        COMPOSITE_VIEW_ISSUE_DEFERRED_COPIES_CALL);
      LegionMap<ApEvent,FieldMask>::aligned postreductions;
      CompositeCopier copier(copy_mask);
      FieldMask dummy_locally_complete;
      FieldMask dominate_capture(copy_mask);
      CompositeCopyNode *copy_tree = construct_copy_tree(dst, logical_node,
          copy_mask, dummy_locally_complete, dominate_capture, copier, this);
#ifdef DEBUG_LEGION
      assert(copy_tree != NULL);
#endif
      copy_tree->issue_copies(info, dst, copy_mask, this, preconditions, 
              postconditions, postreductions, pred_guard, across_helper);
      delete copy_tree;
      if (!postreductions.empty())
      {
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              postreductions.begin(); it != postreductions.end(); it++)
          postconditions.insert(*it);
      }
    } 

    //--------------------------------------------------------------------------
    bool CompositeView::is_upper_bound_node(RegionTreeNode *node) const
    //--------------------------------------------------------------------------
    {
      return version_info->is_upper_bound_node(node);
    }

    //--------------------------------------------------------------------------
    void CompositeView::get_field_versions(RegionTreeNode *node,bool split_prev,
                                           const FieldMask &needed_fields,
                                           FieldVersions &field_versions)
    //--------------------------------------------------------------------------
    {
      // Check to see if this is at the depth of our root node or above it
      // if it is then we can just ask our version info for the results
      if ((node == logical_node) || 
          (node->get_depth() <= logical_node->get_depth()))
      {
        version_info->get_field_versions(node, split_prev,
                                         needed_fields, field_versions);
        return;
      }
      // See if we've already cached the result
      FieldMask still_needed;
      {
        AutoLock v_lock(view_lock,1,false/*exlcusive*/);
        LegionMap<RegionTreeNode*,NodeVersionInfo>::aligned::const_iterator
          finder = node_versions.find(node);
        if (finder != node_versions.end())
        {
          still_needed = needed_fields - finder->second.valid_fields;
          if (!still_needed)
          {
            // We have to make a copy here since these versions could change
            field_versions = finder->second.versions;
            return;
          }
        }
        else
          still_needed = needed_fields; // we still need all the fields
      }
      CompositeNode *capture_node = capture_above(node, still_needed);
      // Result wasn't cached, retake the lock in exclusive mode and compute it
      AutoLock v_lock(view_lock);
      NodeVersionInfo &result = node_versions[node];
      capture_node->capture_field_versions(result.versions, still_needed);
      result.valid_fields |= still_needed;
      field_versions = result.versions;
    }

    //--------------------------------------------------------------------------
    void CompositeView::get_advance_versions(RegionTreeNode *node, bool base,
                                             const FieldMask &needed_fields,
                                             FieldVersions &field_versions)
    //--------------------------------------------------------------------------
    {
      // This should never be called here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CompositeView::get_split_mask(RegionTreeNode *node, 
                                       const FieldMask &needed_fields,
                                       FieldMask &split)
    //--------------------------------------------------------------------------
    {
      // Check to see if this is at the depth of our root node or above it
      // if it is above then we can just ask our version info for the results
      if (node->get_depth() < logical_node->get_depth())
        version_info->get_split_mask(node, needed_fields, split);
      // Nothing at or below here is considered to be split because it is 
      // closed so there is no need for us to do anything
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeView::capture_above(RegionTreeNode *node,
                                                const FieldMask &needed_fields)
    //--------------------------------------------------------------------------
    {
      // Recurse up the tree to get the parent version state
      RegionTreeNode *parent = node->get_parent();
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      if (parent == logical_node)
      {
        // We've reached the top, no need to capture, return the proper child
        return find_child_node(node);
      }
      // Otherwise continue up the tree 
      CompositeNode *parent_node = capture_above(parent, needed_fields);
      // Now make sure that this node has captured for all subregions
      // Do this on the way back down to know that the parent node is good
      parent_node->perform_ready_check(needed_fields);
      return parent_node->find_child_node(node);
    }

    //--------------------------------------------------------------------------
    InnerContext* CompositeView::get_owner_context(void) const
    //--------------------------------------------------------------------------
    {
      return owner_context;
    }

    //--------------------------------------------------------------------------
    void CompositeView::perform_ready_check(FieldMask mask)
    //--------------------------------------------------------------------------
    {
      // Nothing to do here
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_valid_views(const FieldMask &update_mask,
                                         const FieldMask &up_mask,
                       LegionMap<LogicalView*,FieldMask>::aligned &result_views,
                                         bool needs_lock)
    //--------------------------------------------------------------------------
    {
      // Never need the lock here anyway
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        FieldMask overlap = update_mask & it->second;
        if (!overlap)
          continue;
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          result_views.find(it->first);
        if (finder == result_views.end())
          result_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
      {
        FieldMask overlap = update_mask & it->second;
        if (!overlap)
          continue;
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          result_views.find(it->first);
        if (finder == result_views.end())
          result_views[it->first] = overlap;
        else
          finder->second |= overlap;
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
      UniqueID owner_uid;
      derez.deserialize(owner_uid);
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
      DeferredVersionInfo *version_info = new DeferredVersionInfo();
      version_info->unpack_version_numbers(derez, runtime->forest);
      ClosedNode *closed_tree = 
        ClosedNode::unpack_closed_node(derez, runtime, is_region);
      InnerContext *owner_context = runtime->find_context(owner_uid);
      // Make the composite view, but don't register it yet
      void *location;
      CompositeView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) CompositeView(runtime->forest, 
                                           did, owner, target_node, 
                                           version_info, closed_tree,
                                           owner_context,
                                           false/*register now*/);
      else
        view = new CompositeView(runtime->forest, did, owner, 
                           target_node, version_info, closed_tree, 
                           owner_context, false/*register now*/);
      // Unpack all the internal data structures
      std::set<RtEvent> ready_events;
      view->unpack_composite_view(derez, ready_events);
      if (!ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(ready_events);
        DeferCompositeViewRegistrationArgs args;
        args.view = view;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                         wait_on);
        // Not ready to perform registration yet
        return;
      }
      // If we get here, we are ready to perform the registration
      view->register_with_runtime(NULL/*remote registration not needed*/);
    } 

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_deferred_view_registration(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeViewRegistrationArgs *vargs = 
        (const DeferCompositeViewRegistrationArgs*)args;
      // Register only after construction
      vargs->view->register_with_runtime(NULL/*no remote registration*/);
    }

    //--------------------------------------------------------------------------
    void CompositeView::record_dirty_fields(const FieldMask &dirty)
    //--------------------------------------------------------------------------
    {
      dirty_mask |= dirty; 
    }

    //--------------------------------------------------------------------------
    void CompositeView::record_valid_view(LogicalView *view, const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      // For now we'll just record it, we'll add references later
      // during the call to finalize_capture
      if (view->is_instance_view())
      {
#ifdef DEBUG_LEGION
        assert(view->is_materialized_view());
#endif
        MaterializedView *mat_view = view->as_materialized_view();
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          valid_views.find(mat_view);
        if (finder == valid_views.end())
          valid_views[mat_view] = m;
        else
          finder->second |= m;
      }
      else
      {
        DeferredView *def_view = view->as_deferred_view();
        if (def_view->is_composite_view())
        {
          CompositeView *composite_view = def_view->as_composite_view();
          // See if it is a nested on or from above
          if (composite_view->logical_node == logical_node)
          {
            // nested
            LegionMap<CompositeView*,FieldMask>::aligned::iterator finder = 
              nested_composite_views.find(composite_view);
            if (finder == nested_composite_views.end())
              nested_composite_views[composite_view] = m;
            else
              finder->second |= m;
          }
          else
          {
            // not nested
#ifdef DEBUG_LEGION
            assert(composite_view->logical_node->get_depth() < 
                    logical_node->get_depth()); // should be above us
#endif
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid_views.find(composite_view);
            if (finder == valid_views.end())
              valid_views[composite_view] = m;
            else
              finder->second |= m;
          }
        }
        else
        {
          // Just add it like normal
          LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
            valid_views.find(def_view);
          if (finder == valid_views.end())
            valid_views[def_view] = m;
          else
            finder->second |= m;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::record_reduction_fields(const FieldMask &reduc)
    //--------------------------------------------------------------------------
    {
      reduction_mask |= reduc;
    }

    //--------------------------------------------------------------------------
    void CompositeView::record_reduction_view(ReductionView *view, 
                                              const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // For now just add it, we'll record references 
      // during finalize_capture
      LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
        reduction_views.find(view);
      if (finder == reduction_views.end())
        reduction_views[view] = mask;
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeView::record_child_version_state(const LegionColor color, 
          VersionState *state, const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *child_node = logical_node->get_tree_child(color);
      for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->first->logical_node == child_node)
        {
          it->first->record_version_state(state, mask, mutator, true/*root*/);
          it->second |= mask;
          return;
        }
      }
      // Didn't find it so make it
      CompositeNode *child = 
        new CompositeNode(child_node, this, did); 
      child->record_version_state(state, mask, mutator, true/*root*/);
      children[child] = mask;
    }

    //--------------------------------------------------------------------------
    void CompositeView::finalize_capture(bool need_prune)
    //--------------------------------------------------------------------------
    {
      // We add base resource references to all our views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
        it->first->add_nested_resource_ref(did);
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        it->first->add_nested_resource_ref(did);
      // For the deferred views, we try to prune them 
      // based on our closed tree if they are the same we keep them 
      if (need_prune)
      {
        std::vector<CompositeView*> to_erase;
        LegionMap<CompositeView*,FieldMask>::aligned replacements;
        for (LegionMap<CompositeView*,FieldMask>::aligned::iterator it = 
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
        {
          // If the composite view is above in the tree we don't
          // need to worry about pruning it for resource reasons
          if (it->first->logical_node != logical_node)
          {
#ifdef DEBUG_LEGION
            // Should be above us in the region tree
            assert(logical_node->get_depth() > 
                    it->first->logical_node->get_depth());
#endif
            it->first->add_nested_resource_ref(did);
            continue;
          }
          it->first->prune(closed_tree, it->second, replacements, 0/*depth*/);
          if (!it->second)
            to_erase.push_back(it->first);
          else
            it->first->add_nested_resource_ref(did);
        }
        if (!to_erase.empty())
        {
          for (std::vector<CompositeView*>::const_iterator it = 
                to_erase.begin(); it != to_erase.end(); it++)
            nested_composite_views.erase(*it);
        }
        if (!replacements.empty())
        {
          for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
                replacements.begin(); it != replacements.end(); it++)
          {
            LegionMap<CompositeView*,FieldMask>::aligned::iterator finder =
              nested_composite_views.find(it->first);
            if (finder == nested_composite_views.end())
            {
              it->first->add_nested_resource_ref(did);
              nested_composite_views.insert(*it);
            }
            else
              finder->second |= it->second;
          }
        }
      }
      else
      {
        for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
          it->first->add_nested_resource_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::pack_composite_view(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(dirty_mask);
      rez.serialize<size_t>(valid_views.size());
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(nested_composite_views.size());
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize(reduction_mask);
      rez.serialize<size_t>(reduction_views.size());
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(children.size());
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        it->first->pack_composite_node(rez);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::unpack_composite_view(Deserializer &derez,
                                              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(dirty_mask);
      size_t num_mat_views;
      derez.deserialize(num_mat_views);
      for (unsigned idx = 0; idx < num_mat_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        LogicalView *view = static_cast<LogicalView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        derez.deserialize(valid_views[view]);
        if (ready.exists() && !ready.has_triggered())
          preconditions.insert(defer_add_reference(view, ready));
        else // Otherwise we can add the reference now
          view->add_nested_resource_ref(did);
      }
      size_t num_nested_views;
      derez.deserialize(num_nested_views);
      for (unsigned idx = 0; idx < num_nested_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        CompositeView *view = static_cast<CompositeView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        derez.deserialize(nested_composite_views[view]);
        if (ready.exists() && !ready.has_triggered())
          preconditions.insert(defer_add_reference(view, ready));
        else
          view->add_nested_resource_ref(did);
      }
      derez.deserialize(reduction_mask);
      size_t num_reduc_views;
      derez.deserialize(num_reduc_views);
      for (unsigned idx = 0; idx < num_reduc_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        ReductionView *reduc_view = static_cast<ReductionView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        derez.deserialize(reduction_views[reduc_view]);
        if (ready.exists() && !ready.has_triggered())
          preconditions.insert(defer_add_reference(reduc_view, ready));
        else
          reduc_view->add_nested_resource_ref(did);
      }
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        CompositeNode *child = CompositeNode::unpack_composite_node(derez, 
                              this, context->runtime, did, preconditions);
        derez.deserialize(children[child]);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent CompositeView::defer_add_reference(DistributedCollectable *dc, 
                                               RtEvent precondition) const
    //--------------------------------------------------------------------------
    {
      DeferCompositeViewRefArgs args;
      args.dc = dc;
      args.did = did;
      return context->runtime->issue_runtime_meta_task(args, 
          LG_LATENCY_DEFERRED_PRIORITY, precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_deferred_view_ref(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeViewRefArgs *ref_args = 
        (const DeferCompositeViewRefArgs*)args;
      ref_args->dc->add_nested_resource_ref(ref_args->did);
    }

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(RegionTreeNode* node, CompositeBase *p,
                                 DistributedID own_did)
      : CompositeBase(node_lock), logical_node(node), parent(p), 
        owner_did(own_did), currently_valid(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(const CompositeNode &rhs)
      : CompositeBase(node_lock), logical_node(NULL), parent(NULL), owner_did(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeNode::~CompositeNode(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
            version_states.begin(); it != version_states.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(owner_did))
          delete (it->first);
      }
      version_states.clear();
      // Free up all our children 
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        delete (it->first);
      }
      children.clear();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(owner_did))
          delete it->first;
      }
      valid_views.clear();
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(owner_did))
          delete (it->first);
      }
      reduction_views.clear();
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
    InnerContext* CompositeNode::get_owner_context(void) const
    //--------------------------------------------------------------------------
    {
      return parent->get_owner_context();
    }

    //--------------------------------------------------------------------------
    void CompositeNode::perform_ready_check(FieldMask mask)
    //--------------------------------------------------------------------------
    {
      // Do a quick test with read-only lock first
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Remove any fields that are already valid
        mask -= valid_fields;
        if (!mask)
          return;
      }
      RtUserEvent capture_event;
      std::set<RtEvent> preconditions; 
      LegionMap<VersionState*,FieldMask>::aligned needed_states;
      {
        AutoLock n_lock(node_lock);
        // Retest to see if we lost the race
        mask -= valid_fields;
        if (!mask)
          return;
        for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it = 
              pending_captures.begin(); it != pending_captures.end(); it++)
        {
          if (it->second * mask)
            continue;
          preconditions.insert(it->first);
          mask -= it->second;
          if (!mask)
            break;
        }
        // If we still have fields, we're going to do a pending capture
        if (!!mask)
        {
          capture_event = Runtime::create_rt_user_event();
          pending_captures[capture_event] = mask;
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator 
                it = version_states.begin(); it != version_states.end(); it++)
          {
            FieldMask overlap = it->second & mask;
            if (!overlap)
              continue;
            needed_states[it->first] = overlap;
          }
        }
      }
      if (!needed_states.empty())
      {
        // Request final states for all the version states and then either
        // launch a task to do the capture, or do it now
        std::set<RtEvent> capture_preconditions;
        InnerContext *owner_context = parent->get_owner_context();
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              needed_states.begin(); it != needed_states.end(); it++)
          it->first->request_final_version_state(owner_context, it->second,
                                                 capture_preconditions);
        if (!capture_preconditions.empty())
        {
          RtEvent capture_precondition = 
            Runtime::merge_events(capture_preconditions);
          DeferCaptureArgs args;
          args.proxy_this = this;
          args.capture_event = capture_event;
          Runtime *runtime = logical_node->context->runtime;
          RtEvent precondition = 
            runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                             capture_precondition);
          preconditions.insert(precondition);
        }
        else // We can do the capture now!
        {
          WrapperReferenceMutator mutator(preconditions);
          capture(capture_event, &mutator);
        }
      }
      if (!preconditions.empty())
      {
        RtEvent wait_on = Runtime::merge_events(preconditions);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::find_valid_views(const FieldMask &update_mask,
                                         const FieldMask &up_mask,
                       LegionMap<LogicalView*,FieldMask>::aligned &result_views,
                                         bool needs_lock)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        find_valid_views(update_mask, up_mask, result_views, false);
        return;
      }
      // Insert anything we have here
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        FieldMask overlap = it->second & update_mask;
        if (!overlap)
          continue;
        LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
          result_views.find(it->first);
        if (finder == result_views.end())
          result_views[it->first] = overlap;
        else
          finder->second |= overlap;
      }
      // See if we need to keep going up the view tree
      if (!!up_mask)
      {
        // Only go up if necessary and only for fields which are not dirty here
        // because those fields above are not valid
        if (!!dirty_mask)
        {
          FieldMask local_up = up_mask - dirty_mask;
          if (!!local_up)
            parent->find_valid_views(local_up, local_up, result_views);
        }
        else
          parent->find_valid_views(up_mask, up_mask, result_views);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::capture(RtUserEvent capture_event, 
                                ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock n_lock(node_lock);
        LegionMap<RtUserEvent,FieldMask>::aligned::iterator finder = 
          pending_captures.find(capture_event);
#ifdef DEBUG_LEGION
        assert(finder != pending_captures.end());
#endif
        // Perform the capture of each of our overlapping version states
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              version_states.begin(); it != version_states.end(); it++)
        {
          FieldMask overlap = it->second & finder->second;
          if (!overlap)
            continue;
          it->first->capture(this, overlap, mutator);
        }
        valid_fields |= finder->second;
        pending_captures.erase(finder); 
      }
      Runtime::trigger_event(capture_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeNode::handle_deferred_capture(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCaptureArgs *dargs = (const DeferCaptureArgs*)args;
      LocalReferenceMutator mutator;
      dargs->proxy_this->capture(dargs->capture_event, &mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::clone(CompositeView *target,
                              const FieldMask &clone_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_valid);
#endif
      const LegionColor color = logical_node->get_color();
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =  
            version_states.begin(); it != version_states.end(); it++)
      {
        FieldMask overlap = it->second & clone_mask;
        if (!overlap)
          continue;
        // We already hold a reference here so we can pass a NULL mutator
        target->record_child_version_state(color, it->first, 
                                           overlap, NULL/*mutator*/);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::pack_composite_node(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      if (logical_node->is_region())
      {
        rez.serialize<bool>(true);
        rez.serialize(logical_node->as_region_node()->handle);
      }
      else
      {
        rez.serialize<bool>(false);
        rez.serialize(logical_node->as_partition_node()->handle);
      }
      rez.serialize<size_t>(version_states.size());
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
            version_states.begin(); it != version_states.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ CompositeNode* CompositeNode::unpack_composite_node(
        Deserializer &derez, CompositeView *parent, Runtime *runtime, 
        DistributedID owner_did, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      bool is_region;
      derez.deserialize(is_region);
      RegionTreeNode *node;
      if (is_region)
      {
        LogicalRegion handle;
        derez.deserialize(handle);
        node = runtime->forest->get_node(handle);
      }
      else
      {
        LogicalPartition handle;
        derez.deserialize(handle);
        node = runtime->forest->get_node(handle);
      }
      CompositeNode *result = 
        new CompositeNode(node, parent, owner_did);
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        VersionState *state = 
          runtime->find_or_request_version_state(did, ready); 
        derez.deserialize(result->version_states[state]);
        if (ready.exists() && !ready.has_triggered())
        {
          DeferCompositeNodeRefArgs args;
          args.state = state;
          args.owner_did = owner_did;
          RtEvent precondition = 
            runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                             ready);
          preconditions.insert(precondition);
        }
        else
          state->add_nested_resource_ref(owner_did);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeNode::handle_deferred_node_ref(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeNodeRefArgs *nargs = 
        (const DeferCompositeNodeRefArgs*)args;
      nargs->state->add_nested_resource_ref(nargs->owner_did);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_valid(ReferenceMutator *mutator, bool root)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!currently_valid);
#endif
      if (root)
      {
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              version_states.begin(); it != version_states.end(); it++)
          it->first->add_nested_valid_ref(owner_did, mutator);
      }
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->first->notify_valid(mutator, false/*root*/);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
        it->first->add_nested_valid_ref(owner_did, mutator);
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        it->first->add_nested_valid_ref(owner_did, mutator);
      currently_valid = true;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_invalid(ReferenceMutator *mutator, bool root)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(currently_valid);
#endif
      if (root)
      {
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              version_states.begin(); it != version_states.end(); it++)
          it->first->remove_nested_valid_ref(owner_did, mutator);
      }
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->first->notify_invalid(mutator, false/*root*/);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
        it->first->remove_nested_valid_ref(owner_did, mutator);
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        it->first->remove_nested_valid_ref(owner_did, mutator);
      currently_valid = false;
    } 

    //--------------------------------------------------------------------------
    void CompositeNode::record_dirty_fields(const FieldMask &dirty)
    //--------------------------------------------------------------------------
    {
      // should already hold the lock from the caller
      dirty_mask |= dirty;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_valid_view(LogicalView *view, const FieldMask &m)
    //--------------------------------------------------------------------------
    {
      // should already hold the lock from the caller
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        valid_views.find(view);
      if (finder == valid_views.end())
      {
        // Add both a resource and a valid reference
        // No need for a mutator since these must be valid if we are capturing
        view->add_nested_resource_ref(owner_did);
        if (currently_valid)
          view->add_nested_valid_ref(owner_did);
        valid_views[view] = m;
      }
      else
        finder->second |= m;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_reduction_fields(const FieldMask &reduc)
    //--------------------------------------------------------------------------
    {
      // should already hold the lock from the caller
      reduction_mask |= reduc;
    }
    
    //--------------------------------------------------------------------------
    void CompositeNode::record_reduction_view(ReductionView *view,
                                              const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // should already hold the lock from the caller
      LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
        reduction_views.find(view);
      if (finder == reduction_views.end())
      {
        // Add both a resource and a valid reference
        // No need for a mutator since these must be valid if we are capturing
        view->add_nested_resource_ref(owner_did);
        if (currently_valid)
          view->add_nested_valid_ref(owner_did);
        reduction_views[view] = mask;
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_child_version_state(const LegionColor color,
          VersionState *state, const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *child_node = logical_node->get_tree_child(color);
      for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->first->logical_node == child_node)
        {
          it->first->record_version_state(state, mask, mutator, false/*root*/); 
          it->second |= mask;
          return;
        }
      }
      // Didn't find it so make it
      CompositeNode *child = 
        new CompositeNode(child_node, this, owner_did); 
      child->record_version_state(state, mask, mutator, false/*root*/);
      children[child] = mask;
      if (currently_valid)
        child->notify_valid(mutator, false/*root*/);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_version_state(VersionState *state, 
                    const FieldMask &mask, ReferenceMutator *mutator, bool root)
    //--------------------------------------------------------------------------
    {
      LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
        version_states.find(state);
      if (finder == version_states.end())
      {
        state->add_nested_resource_ref(owner_did);
        version_states[state] = mask;
        if (root && currently_valid)
          state->add_nested_valid_ref(owner_did, mutator);
      }
      else
        finder->second |= mask;
    } 

    //--------------------------------------------------------------------------
    void CompositeNode::capture_field_versions(FieldVersions &versions,
                                            const FieldMask &capture_mask) const
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock,1,false/*exclusive*/);
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
            version_states.begin(); it != version_states.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        FieldVersions::iterator finder = 
          versions.find(it->first->version_number);
        if (finder == versions.end())
          versions[it->first->version_number] = overlap;
        else
          finder->second |= overlap;
      }
    }

    /////////////////////////////////////////////////////////////
    // FillView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, RegionTreeNode *node, 
                       FillViewValue *val, bool register_now
#ifdef LEGION_SPY
                       , UniqueID op_uid
#endif
                       )
      : DeferredView(ctx, encode_fill_did(did), owner_proc,
                     node, register_now), value(val)
#ifdef LEGION_SPY
        , fill_op_uid(op_uid)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(value != NULL);
#endif
      value->add_reference();
#ifdef LEGION_GC
      log_garbage.info("GC Fill View %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::FillView(const FillView &rhs)
      : DeferredView(NULL, 0, 0, NULL, false), value(NULL)
#ifdef LEGION_SPY
        , fill_op_uid(0)
#endif
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
    LogicalView* FillView::get_subview(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Fill views don't need subviews
      return this;
    }

    //--------------------------------------------------------------------------
    void FillView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void FillView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, false/*add*/);
    }
    
    //--------------------------------------------------------------------------
    void FillView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void FillView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void FillView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(logical_node->is_region());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize(logical_node->as_region_node()->handle);
        rez.serialize(value->value_size);
        rez.serialize(value->value, value->value_size);
#ifdef LEGION_SPY
        rez.serialize(fill_op_uid);
#endif
      }
      runtime->send_fill_view(target, rez);
      // We've now done the send so record it
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                                         const RestrictInfo &restrict_info,
                                         bool restrict_out)
    //--------------------------------------------------------------------------
    {
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      // We know we're going to write all these fields so we can filter
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   true/*single copy*/, restrict_out,
                                   copy_mask, &info.version_info, 
                                   info.op->get_unique_op_id(),
                                   info.index, local_space, 
                                   preconditions, info.map_applied_events);
      if (restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        restrict_mask &= copy_mask;
        if (!!restrict_mask)
        {
          ApEvent restrict_pre = info.op->get_restrict_precondition();
          preconditions[restrict_pre] = restrict_mask;
        }
      }
      LegionMap<ApEvent,FieldMask>::aligned postconditions;
      issue_deferred_copies(info, dst, copy_mask, preconditions,
                            postconditions, PredEvent::NO_PRED_EVENT);
      // We know there is at most one event per field so no need
      // to sort into event sets here
      // Register the resulting events as users of the destination
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            postconditions.begin(); it != postconditions.end(); it++)
      {
        dst->add_copy_user(0/*redop*/, it->first, &info.version_info, 
                           info.op->get_unique_op_id(), info.index,
                           it->second, false/*reading*/, restrict_out,
                           local_space, info.map_applied_events);
      }
      if (restrict_out && restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              postconditions.begin(); it != postconditions.end(); it++)
        {
          if (it->second * restrict_mask)
            continue;
          info.op->record_restrict_postcondition(it->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         FieldMask copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                          PredEvent pred_guard, CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
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
        std::vector<CopySrcDstField> dst_fields;
        dst->copy_to(pre_set.set_mask, dst_fields, across_helper);
        ApEvent fill_pre = Runtime::merge_events(pre_set.preconditions);
        // Issue the fill command
        // Only apply an intersection if the destination logical node
        // is different than our logical node
        // If the intersection is empty we can skip the fill all together
        if ((logical_node != dst->logical_node) && 
            (!logical_node->intersects_with(dst->logical_node)))
          continue;
        ApEvent fill_post = dst->logical_node->issue_fill(info.op, dst_fields,
                        value->value, value->value_size, fill_pre, pred_guard, 
#ifdef LEGION_SPY
                        fill_op_uid,
#endif
                  (logical_node == dst->logical_node) ? NULL : logical_node);
        if (fill_post.exists())
          postconditions[fill_post] = pre_set.set_mask;
      }
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
#ifdef LEGION_SPY
      UniqueID op_uid;
      derez.deserialize(op_uid);
#endif
      
      RegionNode *target_node = runtime->forest->get_node(handle);
      FillView::FillViewValue *fill_value = 
                      new FillView::FillViewValue(value, value_size);
      void *location;
      FillView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) FillView(runtime->forest, did,
                                      owner_space, target_node, fill_value,
                                      false/*register now*/
#ifdef LEGION_SPY
                                      , op_uid
#endif
                                      );
      else
        view = new FillView(runtime->forest, did, owner_space,
                            target_node,fill_value,false/*register now*/
#ifdef LEGION_SPY
                            , op_uid
#endif
                            );
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    /////////////////////////////////////////////////////////////
    // PhiView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhiView::PhiView(RegionTreeForest *ctx, DistributedID did, 
                     AddressSpaceID owner_space,
                     DeferredVersionInfo *info, RegionTreeNode *node, 
                     PredEvent tguard, PredEvent fguard, 
                     InnerContext *owner, bool register_now) 
      : DeferredView(ctx, encode_phi_did(did), owner_space, node, 
                     register_now), true_guard(tguard), false_guard(fguard), 
        version_info(info), owner_context(owner)
    //--------------------------------------------------------------------------
    {
      version_info->add_reference();
#ifdef LEGION_GC
      log_garbage.info("GC Phi View %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    PhiView::PhiView(const PhiView &rhs)
      : DeferredView(NULL, 0, 0, NULL, false),
        true_guard(PredEvent::NO_PRED_EVENT),
        false_guard(PredEvent::NO_PRED_EVENT), 
        version_info(NULL), owner_context(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhiView::~PhiView(void)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      }
      true_views.clear();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            false_views.begin(); it != false_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      }
      false_views.clear();
      if (version_info->remove_reference())
        delete version_info;
    }

    //--------------------------------------------------------------------------
    PhiView& PhiView::operator=(const PhiView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    LogicalView* PhiView::get_subview(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Phi views don't need subviews
      return this;
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::issue_deferred_copies(const TraversalInfo &info,
                                        MaterializedView *dst,
                                        FieldMask copy_mask,
                                        const RestrictInfo &restrict_info,
                                        bool restrict_out)
    //--------------------------------------------------------------------------
    {
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      // We know we're going to write all these fields so we can filter
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   true/*single copy*/, restrict_out,
                                   copy_mask, &info.version_info, 
                                   info.op->get_unique_op_id(),
                                   info.index, local_space, 
                                   preconditions, info.map_applied_events);
      if (restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        restrict_mask &= copy_mask;
        if (!!restrict_mask)
        {
          ApEvent restrict_pre = info.op->get_restrict_precondition();
          preconditions[restrict_pre] = restrict_mask;
        }
      }
      LegionMap<ApEvent,FieldMask>::aligned postconditions;
      // Now issue copies for both cases 
      issue_guarded_update_copies(info, dst, copy_mask, true_guard,
                                  true_views, restrict_info, restrict_out,
                                  preconditions, postconditions);
      issue_guarded_update_copies(info, dst, copy_mask, false_guard,
                                  false_views, restrict_info, restrict_out,
                                  preconditions, postconditions);
      // Now merge the postconditions and register them with the destination
      LegionList<EventSet>::aligned event_sets;
      RegionTreeNode::compute_event_sets(copy_mask, postconditions, event_sets);
      FieldMask restrict_mask;
      if (restrict_out && restrict_info.has_restrictions())
        restrict_info.populate_restrict_fields(restrict_mask);
      for (LegionList<EventSet>::aligned::const_iterator it = 
            event_sets.begin(); it != event_sets.end(); it++)
      {
        ApEvent post = Runtime::merge_events(it->preconditions);
        if (post.exists())
          dst->add_copy_user(0/*redop*/, post, &info.version_info,
                             info.op->get_unique_op_id(), info.index,
                             it->set_mask, false/*reading*/, restrict_out,
                             local_space, info.map_applied_events);
        if (restrict_out && !(it->set_mask * restrict_mask))
          info.op->record_restrict_postcondition(post);
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::issue_deferred_copies(const TraversalInfo &info,
                                        MaterializedView *dst,
                                        FieldMask copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                 PredEvent pred_guard, CopyAcrossHelper *helper)
    //--------------------------------------------------------------------------
    {
      RestrictInfo dummy_restrict_info;
      LegionMap<ApEvent,FieldMask>::aligned local_postconditions;
      // Issue copies for both cases 
      // We can skip the cases where the polarity is different for the guard
      if (pred_guard != false_guard)
        issue_guarded_update_copies(info, dst, copy_mask, true_guard,
                                    true_views, dummy_restrict_info, 
                                    false/*restrict out*/, preconditions, 
                                    local_postconditions, helper);
      if (pred_guard != true_guard)
        issue_guarded_update_copies(info, dst, copy_mask, false_guard,
                                    false_views, dummy_restrict_info,
                                    false/*restrict out*/, preconditions,
                                    local_postconditions, helper);
      // Now merge the postconditions and protect them for when we are done
      LegionList<EventSet>::aligned event_sets;
      RegionTreeNode::compute_event_sets(copy_mask, postconditions, event_sets);
      for (LegionList<EventSet>::aligned::const_iterator it = 
            event_sets.begin(); it != event_sets.end(); it++)
      {
        ApEvent post = Runtime::merge_events(it->preconditions);
        if (post.exists())
        {
          // post is not guaranteed to be unique
          LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            postconditions.find(post);
          if (finder == postconditions.end())
            postconditions[post] = it->set_mask;
          else
            finder->second |= it->set_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::issue_guarded_update_copies(const TraversalInfo &info,
                                       MaterializedView *dst,
                                       FieldMask copy_mask,
                                       PredEvent predicate_guard,
                  const LegionMap<LogicalView*,FieldMask>::aligned &valid_views,
                                       const RestrictInfo &restrict_info,
                                       bool restrict_out,
                  const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                        LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                       CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // First check to see if the target is already valid
      {
        PhysicalManager *dst_manager = dst->get_manager();
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          if (it->first->is_deferred_view())
            continue;
#ifdef DEBUG_LEGION
          assert(it->first->is_materialized_view());
#endif
          if (it->first->as_materialized_view()->manager == dst_manager)
          {
            copy_mask -= it->second;
            if (!copy_mask)
              return;
          }
        }
      }
      LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
      LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
      // Sort the instances
      dst->logical_node->sort_copy_instances(info, dst, copy_mask, 
                            valid_views, src_instances, deferred_instances);
      if (!src_instances.empty())
      {
        // This has all our destination preconditions
        // Only issue copies from fields which have values
        FieldMask actual_copy_mask;
        LegionMap<ApEvent,FieldMask>::aligned src_preconditions;
        const AddressSpaceID local_space = context->runtime->address_space;
        for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
              it = src_instances.begin(); it != src_instances.end(); it++)
        {
          it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                             true/*single copy*/,
                                             false/*restrict out*/,
                                             it->second, this,
                                             info.op->get_unique_op_id(),
                                             info.index, local_space, 
                                             src_preconditions,
                                             info.map_applied_events);
          actual_copy_mask |= it->second;
        }
        // Move in any preconditions that overlap with our set of fields
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              preconditions.begin(); it != preconditions.end(); it++)
        {
          FieldMask overlap = it->second & actual_copy_mask;
          if (!overlap)
            continue;
          LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            src_preconditions.find(it->first);
          if (finder == src_preconditions.end())
            src_preconditions[it->first] = overlap;
          else
            finder->second |= overlap;
        }
        // issue the grouped copies and put the result in the postconditions
        dst->logical_node->issue_grouped_copies(info, dst,false/*restrict out*/,
                                 predicate_guard, src_preconditions, 
                                 actual_copy_mask, src_instances, this, 
                                 postconditions, across_helper, logical_node);
      }
      if (!deferred_instances.empty())
      {
        for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator it = 
              deferred_instances.begin(); it != deferred_instances.end(); it++)
        {
          LegionMap<ApEvent,FieldMask>::aligned deferred_preconditions;
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pre_it =
                preconditions.begin(); pre_it != preconditions.end(); pre_it++)
          {
            const FieldMask overlap = pre_it->second & it->second;
            if (!overlap)
              continue;
            deferred_preconditions[pre_it->first] = overlap;
          }
          it->first->issue_deferred_copies(info, dst, it->second, 
              deferred_preconditions, postconditions, 
              predicate_guard, across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::record_true_view(LogicalView *view, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        true_views.find(view);
      if (finder == true_views.end())
      {
        true_views[view] = mask;
        view->add_nested_resource_ref(did);
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void PhiView::record_false_view(LogicalView *view, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        false_views.find(view);
      if (finder == false_views.end())
      {
        false_views[view] = mask;
        view->add_nested_resource_ref(did);
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void PhiView::pack_phi_view(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(true_views.size());
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(false_views.size());
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::unpack_phi_view(Deserializer &derez, 
                                  std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      size_t num_true_views;
      derez.deserialize(num_true_views);
      for (unsigned idx = 0; idx < num_true_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        LogicalView *view = static_cast<LogicalView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        derez.deserialize(true_views[view]);
        if (ready.exists() && !ready.has_triggered())
          preconditions.insert(defer_add_reference(view, ready));
        else // Otherwise we can add the reference now
          view->add_nested_resource_ref(did);
      }
      size_t num_false_views;
      derez.deserialize(num_false_views);
      for (unsigned idx = 0; idx < num_false_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        LogicalView *view = static_cast<LogicalView*>(
            runtime->find_or_request_logical_view(view_did, ready));
        derez.deserialize(false_views[view]);
        if (ready.exists() && !ready.has_triggered())
          preconditions.insert(defer_add_reference(view, ready));
        else // Otherwise we can add the reference now
          view->add_nested_resource_ref(did);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PhiView::defer_add_reference(DistributedCollectable *dc,
                                         RtEvent precondition) const
    //--------------------------------------------------------------------------
    {
      DeferPhiViewRefArgs args;
      args.dc = dc;
      args.did = did;
      return context->runtime->issue_runtime_meta_task(args,
          LG_LATENCY_DEFERRED_PRIORITY, precondition);
    }

    //--------------------------------------------------------------------------
    void PhiView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        bool is_region = logical_node->is_region();
        rez.serialize(is_region);
        if (is_region)
          rez.serialize(logical_node->as_region_node()->handle);
        else
          rez.serialize(logical_node->as_partition_node()->handle);
        rez.serialize(true_guard);
        rez.serialize(false_guard);
        version_info->pack_version_numbers(rez);
        rez.serialize<UniqueID>(owner_context->get_context_uid());
        pack_phi_view(rez);
      }
      runtime->send_phi_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    bool PhiView::is_upper_bound_node(RegionTreeNode *node) const
    //--------------------------------------------------------------------------
    {
      return version_info->is_upper_bound_node(node);
    }

    //--------------------------------------------------------------------------
    void PhiView::get_field_versions(RegionTreeNode *node, bool split_prev,
                                     const FieldMask &needed_fields,
                                     FieldVersions &field_versions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node->get_depth() <= logical_node->get_depth());
#endif
      version_info->get_field_versions(node, split_prev,
                                       needed_fields, field_versions);
    }

    //--------------------------------------------------------------------------
    void PhiView::get_advance_versions(RegionTreeNode *node, bool base,
                                       const FieldMask &needed_fields,
                                       FieldVersions &field_versions)
    //--------------------------------------------------------------------------
    {
      // This should never be called here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhiView::get_split_mask(RegionTreeNode *node, 
                                 const FieldMask &needed_fields,
                                 FieldMask &split_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node->get_depth() <= logical_node->get_depth());
#endif
      version_info->get_split_mask(node, needed_fields, split_mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiView::handle_send_phi_view(Runtime *runtime,
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID owner;
      derez.deserialize(owner);
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
      PredEvent true_guard, false_guard;
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      DeferredVersionInfo *version_info = new DeferredVersionInfo();
      version_info->unpack_version_numbers(derez, runtime->forest);
      UniqueID owner_uid;
      derez.deserialize(owner_uid);
      InnerContext *owner_context = runtime->find_context(owner_uid);
      // Make the phi view but don't register it yet
      void *location;
      PhiView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) PhiView(runtime->forest,
                                     did, owner, version_info, target_node, 
                                     true_guard, false_guard, owner_context, 
                                     false/*register_now*/);
      else
        view = new PhiView(runtime->forest, did, owner,
                           version_info, target_node, true_guard, 
                           false_guard, owner_context, false/*register now*/);
      // Unpack all the internal data structures
      std::set<RtEvent> ready_events;
      view->unpack_phi_view(derez, ready_events);
      if (!ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(ready_events);
        DeferPhiViewRegistrationArgs args;
        args.view = view;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                         wait_on);
        return;
      }
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiView::handle_deferred_view_ref(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPhiViewRefArgs *rargs = (const DeferPhiViewRefArgs*)args;
      rargs->dc->add_nested_resource_ref(rargs->did); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhiView::handle_deferred_view_registration(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPhiViewRegistrationArgs *pargs = 
        (const DeferPhiViewRegistrationArgs*)args;
      pargs->view->register_with_runtime(NULL/*no remote registration*/);
    }

    /////////////////////////////////////////////////////////////
    // ReductionView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(RegionTreeForest *ctx, DistributedID did,
                                 AddressSpaceID own_sp,
                                 AddressSpaceID log_own, RegionTreeNode *node, 
                                 ReductionManager *man, UniqueID own_ctx, 
                                 bool register_now)
      : InstanceView(ctx, encode_reduction_did(did), own_sp, log_own, 
                     node, own_ctx, register_now), 
        manager(man), remote_request_event(RtEvent::NO_RT_EVENT)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      logical_node->register_instance_view(manager, owner_context, this);
      manager->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Reduction View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : InstanceView(NULL, 0, 0, 0, NULL, 0, false), manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
      // Always unregister ourselves with the region tree node
      logical_node->unregister_instance_view(manager, owner_context);
      if (manager->remove_nested_resource_ref(did))
      {
        if (manager->is_list_manager())
          delete (manager->as_list_manager());
        else
          delete (manager->as_fold_manager());
      }
      // Remove any initial users as well
      if (!initial_user_events.empty())
      {
        for (std::set<ApEvent>::const_iterator it = initial_user_events.begin();
              it != initial_user_events.end(); it++)
          filter_local_users(*it);
      }
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE) && \
      defined(DEBUG_LEGION)
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
                                          VersionTracker *versions,
                                          Operation *op, unsigned index,
                                          std::set<RtEvent> &map_applied_events,
                                          PredEvent pred_guard, 
                                          bool restrict_out)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime,REDUCTION_VIEW_PERFORM_REDUCTION_CALL);
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;

      bool fold = target->reduce_to(manager->redop, reduce_mask, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, src_fields);

      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      target->find_copy_preconditions(manager->redop, false/*reading*/, 
            false/*single copy*/, restrict_out, reduce_mask, versions, 
            op->get_unique_op_id(), index, local_space, preconditions, 
            map_applied_events);
      this->find_copy_preconditions(manager->redop, true/*reading*/, 
           true/*single copy*/, restrict_out, reduce_mask, versions, 
           op->get_unique_op_id(), index, local_space, preconditions, 
           map_applied_events);
      std::set<ApEvent> event_preconds;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
      {
        event_preconds.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(event_preconds); 
      ApEvent reduce_post = manager->issue_reduction(op, 
                                                     src_fields, dst_fields,
                                                     target->logical_node,
                                                     reduce_pre, pred_guard,
                                                     fold, true/*precise*/,
                                                     NULL/*intersect*/);
      target->add_copy_user(manager->redop, reduce_post, versions,
                           op->get_unique_op_id(), index, reduce_mask, 
                           false/*reading*/, restrict_out, local_space, 
                           map_applied_events);
      this->add_copy_user(manager->redop, reduce_post, versions,
                         op->get_unique_op_id(), index, reduce_mask, 
                         true/*reading*/, restrict_out, local_space, 
                         map_applied_events);
      if (restrict_out)
        op->record_restrict_postcondition(reduce_post);
    } 

    //--------------------------------------------------------------------------
    ApEvent ReductionView::perform_deferred_reduction(MaterializedView *target,
                                                    const FieldMask &red_mask,
                                                    VersionTracker *versions,
                                                   const std::set<ApEvent> &pre,
                                                  Operation *op, unsigned index,
                                                    PredEvent predicate_guard,
                                                    CopyAcrossHelper *helper,
                                                    RegionTreeNode *intersect,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_CALL);
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, red_mask, 
                                    dst_fields, helper);
      this->reduce_from(manager->redop, red_mask, src_fields);
      LegionMap<ApEvent,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/, 
                              true/*single copy*/, false/*restrict out*/,
                              red_mask, versions, op->get_unique_op_id(), index,
                              local_space, src_pre, map_applied_events);
      std::set<ApEvent> preconditions = pre;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(preconditions); 
      ApEvent reduce_post = target->logical_node->issue_copy(op, 
                             src_fields, dst_fields, reduce_pre, 
                             predicate_guard, intersect, manager->redop, fold);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, versions,
                    op->get_unique_op_id(), index, red_mask, 
                    true/*reading*/, false/*restrict out*/,
                    local_space, map_applied_events);
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::perform_deferred_across_reduction(
                              MaterializedView *target, FieldID dst_field, 
                              FieldID src_field, unsigned src_index, 
                              VersionTracker *versions,
                              const std::set<ApEvent> &preconds,
                              Operation *op, unsigned index,
                              PredEvent predicate_guard,
                              RegionTreeNode *intersect,
                              std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_ACROSS_CALL);
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      const bool fold = false;
      target->copy_field(dst_field, dst_fields);
      FieldMask red_mask; red_mask.set_bit(src_index);
      this->reduce_from(manager->redop, red_mask, src_fields);
      LegionMap<ApEvent,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/, 
                              true/*singe copy*/, false/*restrict out*/,
                              red_mask, versions, op->get_unique_op_id(), index,
                              local_space, src_pre, map_applied_events);
      std::set<ApEvent> preconditions = preconds;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(preconditions); 
      ApEvent reduce_post = manager->issue_reduction(op, 
                                             src_fields, dst_fields,
                                             intersect, reduce_pre,
                                             predicate_guard,
                                             fold, false/*precise*/,
                                             target->logical_node);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, versions,
                    op->get_unique_op_id(), index, red_mask, 
                    true/*reading*/, false/*restrict out*/,
                    local_space, map_applied_events);
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    PhysicalManager* ReductionView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      return manager;
    }

    //--------------------------------------------------------------------------
    LogicalView* ReductionView::get_subview(const LegionColor c)
    //--------------------------------------------------------------------------
    {
      // Right now we don't make sub-views for reductions
      return this;
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_copy_preconditions(ReductionOpID redop,
                                                bool reading,
                                                bool single_copy,
                                                bool restrict_out,
                                                const FieldMask &copy_mask,
                                                VersionTracker *versions,
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                             std::set<RtEvent> &applied_events, bool can_filter)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_FIND_COPY_PRECONDITIONS_CALL);
#ifdef DEBUG_LEGION
      assert(can_filter); // should always be able to filter reductions
#endif
      ApEvent use_event = manager->get_use_event();
      if (use_event.exists())
      {
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            preconditions.find(use_event);
        if (finder == preconditions.end())
          preconditions[use_event] = copy_mask;
        else
          finder->second |= copy_mask;
      }
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      if (reading)
      {
        // Register dependences on any reducers
        for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator rit = 
              reduction_users.begin(); rit != reduction_users.end(); rit++)
        {
          const EventUsers &event_users = rit->second;
          if (event_users.single)
          {
            FieldMask overlap = copy_mask & event_users.user_mask;
            if (!overlap)
              continue;
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
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
                LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
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
        for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator rit =
              reading_users.begin(); rit != reading_users.end(); rit++)
        {
          const EventUsers &event_users = rit->second;
          if (event_users.single)
          {
            FieldMask overlap = copy_mask & event_users.user_mask;
            if (!overlap)
              continue;
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder =
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
                LegionMap<ApEvent,FieldMask>::aligned::iterator finder =
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
    void ReductionView::add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                      VersionTracker *versions,
                                      const UniqueID creator_op_id,
                                      const unsigned index,
                                      const FieldMask &mask, 
                                      bool reading, bool restrict_out,
                                      const AddressSpaceID source,
                                      std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == manager->redop);
#endif
      if (!is_logical_owner())
      {
        // If we are not the logical owner we have to send our result back
        RtUserEvent remote_applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_applied_event);
          rez.serialize(copy_term);
          rez.serialize(mask);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize<bool>(true); // is_copy
          rez.serialize(redop);
          rez.serialize<bool>(reading);
        }
        runtime->send_view_remote_update(logical_owner, rez);
        applied_events.insert(remote_applied_event);
      }
      // Quick test: only need to do this if copy term exists
      bool issue_collect = false;
      if (copy_term.exists())
      {
        PhysicalUser *user;
        // We don't use field versions for doing interference 
        // tests on reductions so no need to record it
#ifdef DEBUG_LEGION
        assert(logical_node->is_region());
#endif
        if (reading)
        {
          RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
          user = new PhysicalUser(usage, INVALID_COLOR, creator_op_id, index, 
                                  logical_node->as_region_node());
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = new PhysicalUser(usage, INVALID_COLOR, creator_op_id, index, 
                                  logical_node->as_region_node());
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
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(copy_term, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::find_user_precondition(const RegionUsage &usage,
                                                  ApEvent term_event,
                                                  const FieldMask &user_mask,
                                                  Operation *op, 
                                                  const unsigned index,
                                                  VersionTracker *versions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_FIND_USER_PRECONDITIONS_CALL);
#ifdef DEBUG_LEGION
      if (IS_REDUCE(usage))
        assert(usage.redop == manager->redop);
      else
        assert(IS_READ_ONLY(usage));
#endif
      const bool reading = IS_READ_ONLY(usage);
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
      std::set<ApEvent> wait_on;
      ApEvent use_event = manager->get_use_event();
      if (use_event.exists())
        wait_on.insert(use_event);
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        if (!reading)
          find_reducing_preconditions(user_mask, term_event, wait_on);
        else
          find_reading_preconditions(user_mask, term_event, wait_on);
      }
      return Runtime::merge_events(wait_on);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_user(const RegionUsage &usage, ApEvent term_event,
                                 const FieldMask &user_mask, Operation *op,
                                 const unsigned index, AddressSpaceID source,
                                 VersionTracker *versions,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (IS_REDUCE(usage))
        assert(usage.redop == manager->redop);
      else
        assert(IS_READ_ONLY(usage));
#endif
      if (!term_event.exists())
        return;
      UniqueID op_id = op->get_unique_op_id();
      if (!is_logical_owner())
      {
        // Send back the results to the logical owner node
        RtUserEvent remote_applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_applied_event);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize<bool>(false); // is copy
          rez.serialize(usage);
        }
        runtime->send_view_remote_update(logical_owner, rez);
        applied_events.insert(remote_applied_event);
      }
#ifdef DEBUG_LEGION
      assert(logical_node->is_region());
#endif
      const bool reading = IS_READ_ONLY(usage);
      PhysicalUser *new_user = new PhysicalUser(usage, INVALID_COLOR, op_id, 
                                      index, logical_node->as_region_node());
      bool issue_collect = false;
      {
        AutoLock v_lock(view_lock);
        add_physical_user(new_user, reading, term_event, user_mask);
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
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::add_user_fused(const RegionUsage &usage, 
                                          ApEvent term_event,
                                          const FieldMask &user_mask, 
                                          Operation *op, const unsigned index,
                                          VersionTracker *versions,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events,
                                          bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (IS_REDUCE(usage))
        assert(usage.redop == manager->redop);
      else
        assert(IS_READ_ONLY(usage));
#endif
      UniqueID op_id = op->get_unique_op_id();
      if (!is_logical_owner())
      {
        // Send back the results to the logical owner node
        RtUserEvent remote_applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_applied_event);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize<bool>(false); // is copy
          rez.serialize(usage);
        }
        runtime->send_view_remote_update(logical_owner, rez);
        applied_events.insert(remote_applied_event);
      }
      const bool reading = IS_READ_ONLY(usage);
      std::set<ApEvent> wait_on;
      ApEvent use_event = manager->get_use_event();
      if (use_event.exists())
        wait_on.insert(use_event);
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
#ifdef DEBUG_LEGION
      assert(logical_node->is_region());
#endif
      // Who cares just hold the lock in exlcusive mode, this analysis
      // shouldn't be too expensive for reduction views
      bool issue_collect = false;
      PhysicalUser *new_user = new PhysicalUser(usage, INVALID_COLOR, op_id, 
                                      index, logical_node->as_region_node());
      {
        AutoLock v_lock(view_lock);
        if (!reading) // Reducing
          find_reducing_preconditions(user_mask, term_event, wait_on);
        else // We're reading so wait on any reducers
          find_reading_preconditions(user_mask, term_event, wait_on);  
        // Only need to do this if we actually have a term event
        if (term_event.exists())
        {
          add_physical_user(new_user, reading, term_event, user_mask);
          if (outstanding_gc_events.find(term_event) ==
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(term_event);
            issue_collect = true;
          }
        }
      }
      // Launch the garbage collection task if we need to
      if (issue_collect)
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
      }
      // Return our result
      return Runtime::merge_events(wait_on);
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reducing_preconditions(const FieldMask &user_mask,
                                                    ApEvent term_event,
                                                    std::set<ApEvent> &wait_on)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator rit = 
            reading_users.begin(); rit != reading_users.end(); rit++)
      {
        if (rit->first == term_event)
          continue;
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
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reading_preconditions(const FieldMask &user_mask,
                                                   ApEvent term_event,
                                                   std::set<ApEvent> &wait_on)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator rit = 
            reduction_users.begin(); rit != reduction_users.end(); rit++)
      {
        if (rit->first == term_event)
          continue;
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
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_physical_user(PhysicalUser *user, bool reading,
                                          ApEvent term_event, 
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
    void ReductionView::filter_local_users(ApEvent term_event)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_FILTER_LOCAL_USERS_CALL);
      // Better be holding the lock before calling this
      std::set<ApEvent>::iterator event_finder = 
        outstanding_gc_events.find(term_event);
      if (event_finder != outstanding_gc_events.end())
      {
        LegionMap<ApEvent,EventUsers>::aligned::iterator finder = 
          reduction_users.find(term_event);
        if (finder != reduction_users.end())
        {
          EventUsers &event_users = finder->second;
          if (event_users.single)
          {
            delete (event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator it
                  = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              delete (it->first);
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
            delete (event_users.users.single_user);
          }
          else
          {
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  it = event_users.users.multi_users->begin(); it !=
                  event_users.users.multi_users->end(); it++)
            {
              delete (it->first);
            }
            delete event_users.users.multi_users;
          }
          reading_users.erase(finder);
        }
        outstanding_gc_events.erase(event_finder);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_initial_user(ApEvent term_event, 
                                         const RegionUsage &usage,
                                         const FieldMask &user_mask,
                                         const UniqueID op_id,
                                         const unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(logical_node->is_region());
#endif
      // We don't use field versions for doing interference tests on
      // reductions so there is no need to record it
      PhysicalUser *user = new PhysicalUser(usage, INVALID_COLOR, op_id, index,
                                            logical_node->as_region_node());
      add_physical_user(user, IS_READ_ONLY(usage), term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }
 
    //--------------------------------------------------------------------------
    bool ReductionView::reduce_to(ReductionOpID redop, 
                                  const FieldMask &reduce_mask,
                                  std::vector<CopySrcDstField> &dst_fields,
                                  CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == manager->redop);
#endif
      // Get the destination fields for this copy
      if (across_helper == NULL)
        manager->find_field_offsets(reduce_mask, dst_fields);
      else
        across_helper->compute_across_offsets(reduce_mask, dst_fields);
      return manager->is_foldable();
    }

    //--------------------------------------------------------------------------
    void ReductionView::reduce_from(ReductionOpID redop,
                                    const FieldMask &reduce_mask,
                                    std::vector<CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == manager->redop);
      assert(FieldMask::pop_count(reduce_mask) == 1); // only one field
#endif
      manager->find_field_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_to(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &dst_fields,
                                CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_from(const FieldMask &copy_mask,
                                  std::vector<CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        manager->add_nested_gc_ref(did, mutator);
      else
        send_remote_gc_update(owner_space, mutator, 1, true/*add*/);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        manager->remove_nested_gc_ref(did, mutator);
      else
        send_remote_gc_update(owner_space, mutator, 1, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // No need to check for deletion of the manager since
      // we know that we also hold a resource reference
      manager->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::collect_users(const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      // Do not do this if we are in LegionSpy so we can see 
      // all of the dependences
#if !defined(LEGION_SPY) && !defined(EVENT_GRAPH_TRACE)
      AutoLock v_lock(view_lock);
      for (std::set<ApEvent>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
      {
        filter_local_users(*it); 
      }
#endif
    }

    //--------------------------------------------------------------------------
    void ReductionView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(logical_node->is_region()); // Always regions at the top
#endif
      // Don't take the lock, it's alright to have duplicate sends
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(manager->did);
        rez.serialize(logical_node->as_region_node()->handle);
        rez.serialize(owner_space);
        rez.serialize(logical_owner);
        rez.serialize(owner_context);
      }
      runtime->send_reduction_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    Memory ReductionView::get_location(void) const
    //--------------------------------------------------------------------------
    {
      return manager->get_memory();
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
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      UniqueID context_uid;
      derez.deserialize(context_uid);

      RegionNode *target_node = runtime->forest->get_node(handle);
      RtEvent man_ready;
      PhysicalManager *phy_man = 
        runtime->find_or_request_physical_manager(manager_did, man_ready);
      if (man_ready.exists())
        man_ready.wait();
#ifdef DEBUG_LEGION
      assert(phy_man->is_reduction_manager());
#endif
      ReductionManager *red_manager = phy_man->as_reduction_manager();
      void *location;
      ReductionView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) ReductionView(runtime->forest,
                                           did, owner_space, logical_owner,
                                           target_node, red_manager,
                                           context_uid, false/*register now*/);
      else
        view = new ReductionView(runtime->forest, did, owner_space,
                                 logical_owner, target_node, red_manager, 
                                 context_uid, false/*register now*/);
      // Only register after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    //--------------------------------------------------------------------------
    void ReductionView::perform_remote_valid_check(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      bool send_request = false;
      // If we don't have any registered readers, we have to ask
      // the owner for all the current reducers
      {
        AutoLock v_lock(view_lock);  
        if (!remote_request_event.exists())
        {
          remote_request_event = Runtime::create_rt_user_event();
          send_request = true;
        }
        // else request was already sent
      }
      // If we made the event send the request
      if (send_request)
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_request_event);
        }
        context->runtime->send_view_update_request(logical_owner, rez);
      }
      if (!remote_request_event.has_triggered())
        remote_request_event.wait();
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_update_request(AddressSpaceID source,
                                    RtUserEvent done_event, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // Send back all the reduction users
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_event);
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        rez.serialize<size_t>(reduction_users.size());
        for (LegionMap<ApEvent,EventUsers>::aligned::const_iterator it = 
              reduction_users.begin(); it != reduction_users.end(); it++)
        {
          rez.serialize(it->first);
          if (it->second.single)
          {
            rez.serialize<size_t>(1);
            it->second.users.single_user->pack_user(rez);
            rez.serialize(it->second.user_mask);
          }
          else
          {
            rez.serialize<size_t>(it->second.users.multi_users->size());
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = it->second.users.multi_users->begin(); uit != 
                  it->second.users.multi_users->end(); uit++)
            {
              uit->first->pack_user(rez);
              rez.serialize(uit->second);
            }
          }
        }
      }
      runtime->send_view_update_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_update_response(Deserializer &derez,
                                                RtUserEvent done_event,
                                                RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      std::set<ApEvent> deferred_collections;
      {
        // Take the lock in exclusive mode and start unpacking things
        size_t num_events;
        derez.deserialize(num_events);
        AutoLock v_lock(view_lock);
        for (unsigned idx1 = 0; idx1 < num_events; idx1++)
        {
          ApEvent term_event;
          derez.deserialize(term_event);
          outstanding_gc_events.insert(term_event);
#ifdef DEBUG_LEGION
          // should never have this event before now
          assert(reduction_users.find(term_event) == reduction_users.end());
#endif
          EventUsers &current_users = reduction_users[term_event];
          size_t num_users;
          derez.deserialize(num_users);
          if (num_users == 1)
          {
            current_users.users.single_user = 
              PhysicalUser::unpack_user(derez, false/*add ref*/, forest);
            derez.deserialize(current_users.user_mask);
          }
          else
          {
            current_users.single = false;
            current_users.users.multi_users = 
              new LegionMap<PhysicalUser*,FieldMask>::aligned();
            LegionMap<PhysicalUser*,FieldMask>::aligned &multi = 
              *(current_users.users.multi_users);
            for (unsigned idx2 = 0; idx2 < num_users; idx2++)
            {
              PhysicalUser *user =
                PhysicalUser::unpack_user(derez, false/*add ref*/, forest);
              derez.deserialize(multi[user]);
            }
          }
          deferred_collections.insert(term_event);
        }
      }
      // Defer all the event collections
      if (!deferred_collections.empty())
      {
        std::set<RtEvent> applied_events;
        WrapperReferenceMutator mutator(applied_events);
        for (std::set<ApEvent>::const_iterator it = 
              deferred_collections.begin(); it != 
              deferred_collections.end(); it++)
          defer_collect_user(*it, &mutator);
        if (!applied_events.empty())
        {
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(applied_events));
          return;
        }
      }
      // Trigger the done event
      Runtime::trigger_event(done_event); 
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_remote_update(Deserializer &derez,
                                              AddressSpaceID source,
                                              RegionTreeForest *forest)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      RtUserEvent done_event;
      derez.deserialize(done_event);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      bool is_copy;
      derez.deserialize(is_copy);
      bool issue_collect = false;
      if (is_copy)
      {
        ReductionOpID redop;
        derez.deserialize(redop);
        bool reading;
        derez.deserialize(reading);
        PhysicalUser *user = NULL;
#ifdef DEBUG_LEGION
        assert(logical_node->is_region());
#endif
        // We don't use field versions for doing interference 
        // tests on reductions so no need to record it
        if (reading)
        {
          RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
          user = new PhysicalUser(usage, INVALID_COLOR, op_id, index,
                                  logical_node->as_region_node());
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = new PhysicalUser(usage, INVALID_COLOR, op_id, index,
                                  logical_node->as_region_node());
        }
        AutoLock v_lock(view_lock);
        add_physical_user(user, reading, term_event, user_mask);
        // Update the reference users
        if (outstanding_gc_events.find(term_event) ==
            outstanding_gc_events.end())
        {
          outstanding_gc_events.insert(term_event);
          issue_collect = true;
        }
      }
      else
      {
        RegionUsage usage;
        derez.deserialize(usage);
        const bool reading = IS_READ_ONLY(usage);
#ifdef DEBUG_LEGION
        assert(logical_node->is_region());
#endif
        PhysicalUser *new_user = 
          new PhysicalUser(usage, INVALID_COLOR, op_id, index,
                           logical_node->as_region_node());
        AutoLock v_lock(view_lock);
        add_physical_user(new_user, reading, term_event, user_mask);
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
      {
        std::set<RtEvent> applied_events;
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
        if (!applied_events.empty())
        {
          Runtime::trigger_event(done_event,
              Runtime::merge_events(applied_events));
          return;
        }
      }
      // Now we can trigger our done event
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_remote_invalidate(const FieldMask &invalid_mask,
                                                  RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // Should never be called, there are no invalidates for reduction views
      assert(false);
    }

  }; // namespace Internal 
}; // namespace Legion

