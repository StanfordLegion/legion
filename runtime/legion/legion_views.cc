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
                             RegionTreeNode *node, RtUserEvent destroy_event,
                             bool register_now)
      : DistributedCollectable(ctx->runtime, did, own_addr, loc_space, 
                               destroy_event, register_now), 
        context(ctx), logical_node(node), 
        view_lock(Reservation::create_reservation()) 
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
      view_lock.destroy_reservation();
      view_lock = Reservation::NO_RESERVATION;
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
        delete_logical_view(view);
    }

    /////////////////////////////////////////////////////////////
    // InstanceView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp, AddressSpaceID local_sp,
                               AddressSpaceID log_own, RegionTreeNode *node, 
                               UniqueID own_ctx, RtUserEvent destroy_event,
                               bool register_now)
      : LogicalView(ctx, did, owner_sp, local_sp, node, destroy_event,
          register_now), owner_context(own_ctx), logical_owner(log_own)
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
      inst_view->process_update_response(derez, done_event);
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
      inst_view->process_remote_update(derez, source);
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
                               AddressSpaceID own_addr, AddressSpaceID loc_addr,
                               AddressSpaceID log_own, RegionTreeNode *node, 
                               InstanceManager *man, MaterializedView *par, 
                               UniqueID own_ctx, RtUserEvent destroy_event,
                               bool register_now)
      : InstanceView(ctx, encode_materialized_did(did, par == NULL), own_addr, 
         loc_addr, log_own, node, own_ctx, destroy_event, register_now), 
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
      {
        manager->add_nested_resource_ref(did);
        // If we are the root and remote add a resource reference from
        // the owner node
        if (!is_owner())
          add_base_resource_ref(REMOTE_DID_REF);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Materialized View %ld %d %ld", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, manager->did); 
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, 0, 0, 0, NULL, 0, 
          RtUserEvent::NO_RT_USER_EVENT, false),
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
        if (manager->remove_nested_resource_ref(did))
          delete manager;
        if (is_owner())
        {
          UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> 
            functor(this, NULL);
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
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
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
      // This is the common case we should already have it
      {
        AutoLock v_lock(view_lock, 1, false/*exclusive*/);
        std::map<ColorPoint,MaterializedView*>::const_iterator finder = 
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
          context->runtime->get_available_distributed_id(false);
        bool free_child_did = false;
        MaterializedView *child_view = NULL;
        {
          // Retake the lock and see if we lost the race
          AutoLock v_lock(view_lock);
          std::map<ColorPoint,MaterializedView*>::const_iterator finder = 
                                                              children.find(c);
          if (finder != children.end())
          {
            child_view = finder->second;
            free_child_did = true;
          }
          else
          {
            // Otherwise we get to make it
            child_view = legion_new<MaterializedView>(context, child_did, 
                                              owner_space, local_space,
                                              logical_owner, child_node, 
                                              manager, this, owner_context,
                                              RtUserEvent::NO_RT_USER_EVENT,
                                              true/*reg now*/);
            children[c] = child_view;
          }
          if (free_child_did)
            context->runtime->free_distributed_id(child_did);
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
        MaterializedView *mat_child = child_view->as_materialized_view();
        // Retake the lock and add the child
        AutoLock v_lock(view_lock);
        children[c] = mat_child;
        return mat_child;
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
      ColorPoint color;
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
                              std::vector<Domain::CopySrcDstField> &copy_fields)
    //--------------------------------------------------------------------------
    {
      std::vector<FieldID> local_fields(1,fid);
      manager->compute_copy_offsets(local_fields, copy_fields); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::copy_to(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields,
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
                               std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::reduce_to(ReductionOpID redop, 
                                     const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields,
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
                               std::vector<Domain::CopySrcDstField> &src_fields)
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
                                                   const FieldMask &copy_mask,
                                                const VersionInfo &version_info,
                                                   const UniqueID creator_op_id,
                                                   const unsigned index,
                                                   const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events)
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
      find_local_copy_preconditions(redop, reading, copy_mask, ColorPoint(), 
                                    version_info, creator_op_id, index, source,
                                    preconditions, applied_events);
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, copy_mask,
                        local_point, version_info, creator_op_id, index,
                        source, preconditions, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                                  const ColorPoint &child_color,
                                                const VersionInfo &version_info,
                                                  const UniqueID creator_op_id,
                                                  const unsigned index,
                                                  const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      find_local_copy_preconditions(redop, reading, copy_mask, child_color, 
                                    version_info, creator_op_id, index, source,
                                    preconditions, applied_events);
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, copy_mask,
                        local_point, version_info, creator_op_id, index, 
                        source, preconditions, applied_events);
      }
    }
    
    //--------------------------------------------------------------------------
    void MaterializedView::find_local_copy_preconditions(ReductionOpID redop,
                                                         bool reading,
                                                     const FieldMask &copy_mask,
                                                  const ColorPoint &child_color,
                                                const VersionInfo &version_info,
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
        perform_remote_valid_check(copy_mask, version_info, reading);  
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
        find_current_preconditions(copy_mask, usage, child_color, 
                                   creator_op_id, index, preconditions,
                                   dead_events, filter_current_users,
                                   observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      creator_op_id, index, preconditions,
                                      dead_events);
      }
      else
      {
        RegionUsage usage((redop > 0) ? REDUCE : WRITE_DISCARD,EXCLUSIVE,redop);
        FieldMask observed, non_dominated, write_skip_mask;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Find any version updates as well our write skip mask
        find_version_updates(copy_mask, version_info, write_skip_mask, 
            filter_mask, advance_versions, add_versions, redop > 0);
        if (!!write_skip_mask)
        {
          // If we have a write skip mask we know we won't interfere with
          // any users in the list of current users so we can skip them
          const FieldMask current_mask = copy_mask - write_skip_mask;
          if (!!current_mask)
            find_current_preconditions(current_mask, usage, child_color,
                                       creator_op_id, index, preconditions,
                                       dead_events, filter_current_users,
                                       observed, non_dominated);
        }
        else // the normal case with no write-skip
          find_current_preconditions(copy_mask, usage, child_color, 
                                     creator_op_id, index, preconditions,
                                     dead_events, filter_current_users,
                                     observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      creator_op_id, index, preconditions,
                                      dead_events);
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
    void MaterializedView::add_copy_user(ReductionOpID redop, ApEvent copy_term,
                                         const VersionInfo &version_info,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                     const FieldMask &copy_mask, bool reading,
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
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color,
                               version_info, creator_op_id, index, 
                               copy_mask, source, applied_events);
      }
      add_local_copy_user(usage, copy_term, true/*base*/, ColorPoint(),
       version_info, creator_op_id, index, copy_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user_above(const RegionUsage &usage, 
                                               ApEvent copy_term, 
                                               const ColorPoint &child_color,
                                               const VersionInfo &version_info,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const FieldMask &copy_mask,
                                               const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color,
         version_info, creator_op_id, index, copy_mask, source, applied_events);
      }
      add_local_copy_user(usage, copy_term, false/*base*/, child_color, 
        version_info, creator_op_id, index, copy_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_local_copy_user(const RegionUsage &usage, 
                                               ApEvent copy_term,bool base_user,
                                               const ColorPoint &child_color,
                                               const VersionInfo &version_info,
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
          rez.serialize(usage);
          rez.serialize(copy_mask);
          rez.serialize(child_color);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize(copy_term);
          // Figure out which version infos we need
          LegionMap<VersionID,FieldMask>::aligned needed_versions;
          const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
            version_info.get_versions(logical_node)->get_field_versions();
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
        }
        runtime->send_view_remote_update(logical_owner, rez);
        // Tell the operation it has to wait for this event
        // to trigger before it can be considered mapped
        applied_events.insert(remote_update_event);
      }
      PhysicalUser *user = legion_new<PhysicalUser>(usage, child_color, 
                                                    creator_op_id, index);
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
        if (!valid_remote_instances.empty() && IS_READ_ONLY(usage))
        {
          // We only actually have to do the invalidations if we are not split
          bool is_split = false;
          version_info.get_advance_mask(logical_node, is_split);
          if (!is_split)
            perform_read_invalidations(copy_mask, version_info, 
                                       source, applied_events);
        }
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
                          const unsigned index, const VersionInfo &version_info,
                          std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> wait_on_events;
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      UniqueID op_id = op->get_unique_op_id();
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, ColorPoint(), 
         version_info, op_id, index, user_mask, wait_on_events, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
         version_info, op_id, index, user_mask, wait_on_events, applied_events);
      }
      return Runtime::merge_events(wait_on_events); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_user_preconditions_above(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const ColorPoint &child_color,
                                                const VersionInfo &version_info,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // Do the precondition analysis on the way up
      find_local_user_preconditions(usage, term_event, child_color, 
          version_info, op_id, index, user_mask, preconditions, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
          version_info, op_id, index, user_mask, preconditions, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_local_user_preconditions(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const ColorPoint &child_color,
                                                const VersionInfo &version_info,
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
      const bool read_only = IS_READ_ONLY(usage);
      if (!is_logical_owner())
        perform_remote_valid_check(user_mask, version_info, read_only);
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users, 
                                           filter_previous_users;
      if (read_only)
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions(user_mask, usage, child_color, term_event,
                                   op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      term_event, op_id, index,
                                      preconditions, dead_events);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions(user_mask, usage, child_color, term_event,
                                   op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      term_event, op_id, index,
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
    void MaterializedView::add_user(const RegionUsage &usage,ApEvent term_event,
                                    const FieldMask &user_mask, Operation *op,
                                    const unsigned index, AddressSpaceID source,
                                    const VersionInfo &version_info,
                                    std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      UniqueID op_id = op->get_unique_op_id();
      bool need_version_update = false;
      if (IS_WRITE(usage))
        need_version_update = update_version_numbers(user_mask, version_info,
                                                     source, applied_events);
      // Go up the tree if necessary 
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color, version_info, 
         op_id, index, user_mask, need_version_update, source, applied_events);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                         ColorPoint(), version_info, op_id, index, 
                         user_mask, source, applied_events);
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
                                          const ColorPoint &child_color,
                                          const VersionInfo &version_info,
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
        need_update_above = update_version_numbers(user_mask, version_info,
                                                   source, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color, version_info,
         op_id, index, user_mask, need_update_above, source, applied_events);
      }
      add_local_user(usage, term_event, child_color, version_info,
                     op_id, index, user_mask, source, applied_events);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_local_user(const RegionUsage &usage,
                                          ApEvent term_event,
                                          const ColorPoint &child_color,
                                          const VersionInfo &version_info,
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
        // If we are no the owner, we have to send the user back
        RtUserEvent remote_update_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(remote_update_event);
          rez.serialize<bool>(false); // is copy
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize(child_color);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(term_event);
          // Figure out which version infos we need
          LegionMap<VersionID,FieldMask>::aligned needed_versions;
          const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
            version_info.get_versions(logical_node)->get_field_versions();
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
        }
        runtime->send_view_remote_update(logical_owner, rez);
        // Tell the operation it has to wait for this event to
        // trigger before it can be considered mapped
        applied_events.insert(remote_update_event);
      }
      PhysicalUser *new_user = 
        legion_new<PhysicalUser>(usage, child_color, op_id, index);
      new_user->add_reference();
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      // Finally add our user and return if we need to issue a GC meta-task
      add_current_user(new_user, term_event, user_mask);
      // See if we need to check for read only invalidates
      if (!valid_remote_instances.empty() && IS_READ_ONLY(usage))
      {
        // We only actually have to do the invalidations if we are not split
        bool is_split = false;
        version_info.get_advance_mask(logical_node, is_split);
        if (!is_split)
          perform_read_invalidations(user_mask, version_info, 
                                     source, applied_events);
      }
      if (outstanding_gc_events.find(term_event) == 
          outstanding_gc_events.end())
      {
        outstanding_gc_events.insert(term_event);
        return !child_color.is_valid();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::add_user_fused(const RegionUsage &usage, 
                                             ApEvent term_event,
                                             const FieldMask &user_mask, 
                                             Operation *op,const unsigned index,
                                             const VersionInfo &version_info,
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
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, ColorPoint(), 
         version_info, op_id, index, user_mask, wait_on_events, applied_events);
      bool need_version_update = false;
      if (IS_WRITE(usage) && update_versions)
        need_version_update = update_version_numbers(user_mask, version_info,
                                                     source, applied_events);
      // Go up the tree if necessary
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above_fused(usage, term_event, local_color, 
                              version_info, op_id, index, user_mask, source, 
                              wait_on_events, applied_events, 
                              need_version_update);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                         ColorPoint(), version_info, op_id, index, 
                         user_mask, source, applied_events);
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
                                                const ColorPoint &child_color,
                                                const VersionInfo &version_info,
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
      find_local_user_preconditions(usage, term_event, child_color, 
          version_info, op_id, index, user_mask, preconditions, applied_events);
      bool need_update_above = false;
      if (need_version_update)
        need_update_above = update_version_numbers(user_mask, version_info,
                                                   source, applied_events);
      // Go up the tree if we have to
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
      {
        const ColorPoint &local_color = logical_node->get_color();
        parent->add_user_above_fused(usage, term_event, local_color,
                              version_info, op_id, index, user_mask, source,
                              preconditions, applied_events, need_update_above);
      }
      // Add the user on the way back down
      add_local_user(usage, term_event, child_color, version_info,
                     op_id, index, user_mask, source, applied_events);
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
      // No need to take the lock since we are just initializing
      PhysicalUser *user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                                    op_id, index);
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
      else if(parent->remove_nested_gc_ref(did, mutator))
        legion_delete(parent);
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
        legion_delete(parent);
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
      RtUserEvent destroy_event = Runtime::create_rt_user_event();
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
        rez.serialize(destroy_event);
      }
      runtime->send_materialized_view(target, rez);
      register_remote_instance(target, destroy_event);
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
    void MaterializedView::find_version_updates(const FieldMask &user_mask,
                                                const VersionInfo &version_info,
                                                FieldMask &write_skip_mask,
                                                FieldMask &filter_mask,
                              LegionMap<VersionID,FieldMask>::aligned &advance,
                              LegionMap<VersionID,FieldMask>::aligned &add_only,
                                                bool is_reduction)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      sanity_check_versions();
#endif
      FieldVersions *versions = version_info.get_versions(logical_node); 
#ifdef DEBUG_LEGION
      assert(versions != NULL);
#endif
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        versions->get_field_versions();
      bool is_split = false;
#ifndef LEGION_SPY
      const FieldMask &advance_mask = 
#endif
        version_info.get_advance_mask(logical_node, is_split);
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            field_versions.begin(); it != field_versions.end(); it++)
      {
        FieldMask overlap = it->second & user_mask;
        if (!overlap)
          continue;
        // Special case for the zero version number
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
          FieldMask intersect = overlap & finder->second;
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
          FieldMask intersect = overlap & finder->second;
          if (!!intersect)
          {
            // This is a write skip field since we're already
            // at the version number at this view, but we're only
            // really at the version number if we are not a split
            // version number and we're not reducing
            // We skip this optimization if we are doing Legion Spy
            // because Legion Spy doesn't currently understand
            // version numbers and so it can't do the same check
#ifndef LEGION_SPY
            if (!is_reduction)
            {
              if (is_split)
                write_skip_mask |= (intersect - advance_mask);
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
            // If we are remote update our read masks
            if (!is_logical_owner())
            {
              previous_remote_read_requests -= overlap;
              previous_remote_read_requests |= 
                (current_remote_read_requests & overlap);
              current_remote_read_requests -= overlap;
            }
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
      // If we are remote we need to filter our read request masks
      if (!is_logical_owner())
      {
        previous_remote_read_requests -= filter_mask;
        current_remote_read_requests -= filter_mask;
      }
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::update_version_numbers(const FieldMask &user_mask,
                                                const VersionInfo &version_info,
                                                const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      FieldVersions *versions = version_info.get_versions(logical_node); 
#ifdef DEBUG_LEGION
      assert(versions != NULL);
#endif
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        versions->get_field_versions();
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
            field_versions.begin(); it != field_versions.end(); it++)
      {
        FieldMask overlap = it->second & user_mask;
        if (!overlap)
          continue;
        // Special case for the zero version number
        if (it->first == 0)
        {
          filter_mask |= overlap;
          update_versions[1] = overlap;
          continue;
        }
        // We are always trying to advance the version numbers here
        // since these are writing users and are therefore going from
        // the current version number to the next one. We'll check for
        // the most common cases here, and only filter if we don't find them.
        const VersionID previous_number = it->first; 
        const VersionID next_number = it->first + 1; 
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
            // If we are remote, filter back our previous read requests
            if (!is_logical_owner())
            {
              previous_remote_read_requests -= intersect;
              previous_remote_read_requests |= 
                (current_remote_read_requests & intersect);
              current_remote_read_requests -= intersect;
            }
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
        LegionMap<ApEvent,EventUsers>::aligned::iterator previous_finder = 
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
      if (pit->first.has_triggered())
      {
        EventUsers &previous_users = pit->second;
        if (previous_users.single)
        {
          if (previous_users.users.single_user->remove_reference())
            legion_delete(previous_users.users.single_user);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::iterator it = 
                previous_users.users.multi_users->begin(); it !=
                previous_users.users.multi_users->end(); it++)
          {
            if (it->first->remove_reference())
              legion_delete(it->first);
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
            legion_delete(user);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = previous_users.users.multi_users->begin(); it !=
                previous_users.users.multi_users->end(); it++)
          {
            if (it->first->remove_reference())
              legion_delete(it->first);
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
              legion_delete(*it);
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
    void MaterializedView::find_current_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const ColorPoint &child_color,
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
        if (cit->first.has_triggered())
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
        else
          observed |= overlap;
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index))
          {
            preconditions.insert(cit->first);
            filter_events[cit->first] = overlap;
          }
          else
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
            if (has_local_precondition(it->first, usage,
                                       child_color, op_id, index))
            {
              preconditions.insert(cit->first);
              filter_events[cit->first] |= user_overlap;
            }
            else
              non_dominated |= user_overlap;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const ColorPoint &child_color,
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
        if (pit->first.has_triggered())
        {
          dead_events.insert(pit->first);
          continue;
        }
#endif
        const EventUsers &event_users = pit->second;
        if (user_mask * event_users.user_mask)
          continue;
#ifndef LEGION_SPY
        if (preconditions.find(pit->first) != preconditions.end())
          continue;
#endif
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index))
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
            if (has_local_precondition(it->first, usage,
                                       child_color, op_id, index))
              preconditions.insert(pit->first);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_current_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const ColorPoint &child_color,
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
        if (cit->first.has_triggered())
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
        else
          observed |= overlap;
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index))
          {
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
              preconditions.find(cit->first);
            if (finder == preconditions.end())
              preconditions[cit->first] = overlap;
            else
              finder->second |= overlap;
            filter_events[cit->first] = overlap;
          }
          else
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
            if (has_local_precondition(it->first, usage,
                                       child_color, op_id, index))
            {
              LegionMap<ApEvent,FieldMask>::aligned::iterator finder =
                preconditions.find(cit->first);
              if (finder == preconditions.end())
                preconditions[cit->first] = overlap;
              else
                finder->second |= overlap;
              filter_events[cit->first] |= user_overlap;
            }
            else
              non_dominated |= user_overlap;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                                 const FieldMask &user_mask,
                                                 const RegionUsage &usage,
                                                 const ColorPoint &child_color,
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
        if (pit->first.has_triggered())
        {
          dead_events.insert(pit->first);
          continue;
        }
#endif
        const EventUsers &event_users = pit->second;
        const FieldMask overlap = user_mask & event_users.user_mask;
        if (!overlap)
          continue;
#ifndef LEGION_SPY
        if (preconditions.find(pit->first) != preconditions.end())
          continue;
#endif
        if (event_users.single)
        {
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index))
          {
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
              preconditions.find(pit->first);
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
            const FieldMask user_overlap = user_mask & it->second;
            if (!user_overlap)
              continue;
            if (has_local_precondition(it->first, usage,
                                       child_color, op_id, index))
            {
              LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
                preconditions.find(pit->first);
              if (finder == preconditions.end())
                preconditions[pit->first] = user_overlap;
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
    void MaterializedView::set_descriptor(FieldDataDescriptor &desc,
                                          FieldID field_id) const
    //--------------------------------------------------------------------------
    {
      // Get the low-level index space
      const Domain &dom = logical_node->get_domain_no_wait();
      desc.index_space = dom.get_index_space();
      // Then ask the manager to fill in the rest of the information
      manager->set_descriptor(desc, field_id);
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
      RtUserEvent destroy_event;
      derez.deserialize(destroy_event);
      RtEvent man_ready;
      PhysicalManager *phy_man = 
        runtime->find_or_request_physical_manager(manager_did, man_ready);
      MaterializedView *parent = NULL;
      if (parent_did != 0)
      {
        RtEvent par_ready;
        LogicalView *par_view = 
          runtime->find_or_request_logical_view(parent_did, par_ready);
        if (par_ready.exists())
          par_ready.wait();
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
      InstanceManager *inst_manager = phy_man->as_instance_manager();
      void *location;
      MaterializedView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = legion_new_in_place<MaterializedView>(location, runtime->forest,
                                              did, owner_space, 
                                              runtime->address_space,
                                              logical_owner, 
                                              target_node, inst_manager,
                                              parent, context_uid,
                                              destroy_event,
                                              false/*register now*/);
      else
        view = legion_new<MaterializedView>(runtime->forest, did, owner_space,
                                     runtime->address_space, logical_owner,
                                     target_node, inst_manager, parent, 
                                     context_uid, destroy_event,
                                     false/*register now*/);
      // Register only after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::perform_remote_valid_check(
                  const FieldMask &check_mask, const VersionInfo &version_info,
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
        FieldVersions *versions = version_info.get_versions(logical_node); 
#ifdef DEBUG_LEGION
        assert(versions != NULL);
#endif
        const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
          versions->get_field_versions();
        need_valid_update = check_mask;
        AutoLock v_lock(view_lock);
        for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
              field_versions.begin(); it != field_versions.end(); it++)
        {
          FieldMask overlap = it->second & check_mask;
          if (!overlap)
            continue;
          // See if we can find it as either the current or the next
          // version number
          LegionMap<VersionID,FieldMask>::aligned::const_iterator finder = 
            current_versions.find(it->first);
          if (finder != current_versions.end())
          {
            FieldMask version_overlap = overlap & finder->second;
            if (!!version_overlap)
            {
              // If we've already done the read check for these fields
              // then we are done
              FieldMask valid_fields = 
                version_overlap & current_remote_read_requests;
              if (!!valid_fields)
              {
                need_valid_update -= valid_fields;
                current_remote_read_requests |= version_overlap;
              }
            }
          }
          finder = current_versions.find(it->first+1);
          if (finder != current_versions.end())
          {
            FieldMask version_overlap = overlap & finder->second;
            if (!!version_overlap)
            {
              FieldMask valid_fields = 
                version_overlap & previous_remote_read_requests;
              if (!!valid_fields)
              {
                need_valid_update -= valid_fields;
                previous_remote_read_requests |= version_overlap;
              }
            }
          }
        }
        // When these requests are done the current version number
        // will have all its field read requests done
        if (!!need_valid_update)
          current_remote_read_requests |= need_valid_update;
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
        else if (local_wait_on.empty())
          return; // no updates need here
      }
      else
      {
        // If we're writing all we need to do is check that we are valid,
        // if we're not valid we have to send a request
        AutoLock v_lock(view_lock);
        need_valid_update = check_mask - remote_valid_mask;
        if (!need_valid_update)
          return; // We're done if all our fields are valid
        // See which fields we already have requests for
        for (LegionMap<RtEvent,FieldMask>::aligned::const_iterator it = 
              remote_update_requests.begin(); it != 
              remote_update_requests.end(); it++)
        {
          FieldMask overlap = need_valid_update & it->second;
          if (!overlap)
            continue;
          if (wait_on != NULL)
            wait_on->insert(it->first);
          else
            local_wait_on.insert(it->first);
          need_valid_update -= overlap;
          if (!need_valid_update)
            break;
        }
        if (!!need_valid_update)
        {
          request_event = Runtime::create_rt_user_event();
          remote_update_requests[request_event] = need_valid_update;
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
      }
      // If we have a parent, see if it needs to send requests too so 
      // we can get as many in flight as possible in parallel
      if (parent != NULL)
      {
        if (wait_on != NULL)
          parent->perform_remote_valid_check(check_mask, version_info,
                                             reading, wait_on);
        else
          parent->perform_remote_valid_check(check_mask, version_info,
                                             reading, &local_wait_on);
      }
      // If we are the base caller, then we do the wait
      if ((wait_on == NULL) && !local_wait_on.empty())
      {
        RtEvent wait_for = Runtime::merge_events(local_wait_on);
        wait_for.wait();
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::perform_read_invalidations(
                 const FieldMask &check_mask, const VersionInfo &version_info,
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
      // If we are reading exactly the current version number for any of
      // these fields, then we need to send invalidations to remote nodes
      // for those fields.
      FieldMask invalidate_mask;
      FieldVersions *versions = version_info.get_versions(logical_node); 
#ifdef DEBUG_LEGION
      assert(versions != NULL);
#endif
      const LegionMap<VersionID,FieldMask>::aligned &field_versions = 
        versions->get_field_versions();
      for (LegionMap<VersionID,FieldMask>::aligned::const_iterator it = 
            field_versions.begin(); it != field_versions.end(); it++)
      {
        FieldMask overlap = it->second & check_mask;
        if (!overlap)
          continue;
        LegionMap<VersionID,FieldMask>::aligned::const_iterator finder = 
          current_versions.find(it->first);
        if (finder == current_versions.end())
          continue;
        FieldMask version_overlap = overlap & finder->second;
        if (!version_overlap)
          continue;
        invalidate_mask |= version_overlap; 
      }
      if (!!invalidate_mask)
        send_invalidations(invalidate_mask, source, applied_events);
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
            rez.serialize<size_t>(users.users.multi_users->size());
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = users.users.multi_users->begin(); uit !=
                  users.users.multi_users->end(); uit++)
            {
              uit->first->pack_user(rez);
              rez.serialize(uit->second);
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
            rez.serialize<size_t>(users.users.multi_users->size());
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator
                  uit = users.users.multi_users->begin(); uit !=
                  users.users.multi_users->end(); uit++)
            {
              uit->first->pack_user(rez);
              rez.serialize(uit->second);
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
                                                   RtUserEvent done_event)
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
                PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                  PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                  PhysicalUser::unpack_user(derez, true/*add ref*/);
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
                                                 AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      RtUserEvent update_event;
      derez.deserialize(update_event);
      bool is_copy;
      derez.deserialize(is_copy);
      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      ColorPoint child_color;
      derez.deserialize(child_color);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      size_t num_versions;
      derez.deserialize(num_versions);
      FieldVersions *field_versions = new FieldVersions();
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        FieldMask mask;
        derez.deserialize(mask);
        field_versions->add_field_version(vid, mask);
      }
      // Make a temporary version info just for these field versions
      VersionInfo temp_version_info;
      VersionInfo::NodeInfo &node_info = 
        temp_version_info.find_tree_node_info(logical_node);
      field_versions->add_reference();
      node_info.field_versions = field_versions;
      std::set<RtEvent> applied_conditions;
      if (is_copy)
      {
        // Do analysis and register the user
        LegionMap<ApEvent,FieldMask>::aligned dummy_preconditions;
        find_local_copy_preconditions(usage.redop, IS_READ_ONLY(usage),
                                    user_mask, child_color,
                                    temp_version_info, op_id, index, source,
                                    dummy_preconditions, applied_conditions);
        add_local_copy_user(usage, term_event, true/*base user*/,
                            child_color, temp_version_info, op_id, index,
                            user_mask, source, applied_conditions);
      }
      else
      {
        // Do analysis and register the user
        std::set<ApEvent> dummy_preconditions;
        find_local_user_preconditions(usage, term_event, child_color,
                                      temp_version_info, op_id, index,user_mask,
                                      dummy_preconditions, applied_conditions);
        if (IS_WRITE(usage))
          update_version_numbers(user_mask, temp_version_info,
                                 source, applied_conditions);
        if (add_local_user(usage, term_event, child_color, temp_version_info,
                           op_id, index, user_mask, source, applied_conditions))
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
                               AddressSpaceID owner_sp, AddressSpaceID local_sp,
                               RegionTreeNode *node, RtUserEvent destroy_event,
                               bool register_now)
      : LogicalView(ctx, did, owner_sp, local_sp, node, 
                    destroy_event, register_now)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DeferredView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
      // Find the destination preconditions first 
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                                   copy_mask, info.version_info, 
                                   info.op->get_unique_op_id(),
                                   info.index, local_space, 
                                   preconditions, info.map_applied_events);
      LegionMap<ApEvent,FieldMask>::aligned postconditions;
      issue_deferred_copies(info, dst, copy_mask, 
                            preconditions, postconditions);
      // Register the resulting events as users of the destination
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            postconditions.begin(); it != postconditions.end(); it++)
      {
        dst->add_copy_user(0/*redop*/, it->first, info.version_info, 
                           info.op->get_unique_op_id(), info.index,
                           it->second, false/*reading*/, local_space, 
                           info.map_applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::issue_deferred_copies_across(const TraversalInfo &info,
                                                     MaterializedView *dst,
                                      const std::vector<unsigned> &src_indexes,
                                      const std::vector<unsigned> &dst_indexes,
                                                     ApEvent precondition,
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
      LegionMap<ApEvent,FieldMask>::aligned local_postconditions;
      // A seemingly common case but not the general one, if the fields
      // are in the same locations for the source and destination then
      // we can just do the normal deferred copy routine
      if (perfect)
      {
        issue_deferred_copies(info, dst, src_mask, preconditions, 
                              local_postconditions);
      }
      else
      {
        // Initialize the across copy helper
        CopyAcrossHelper across_helper(src_mask);
        dst->manager->initialize_across_helper(&across_helper, dst_mask, 
                                               src_indexes, dst_indexes);
        issue_deferred_copies(info, dst, src_mask, preconditions, 
                              local_postconditions, &across_helper);
      }
      // Put the local postconditions in the result
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
           local_postconditions.begin(); it != local_postconditions.end(); it++)
      {
        postconditions.insert(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void DeferredView::find_field_descriptors(ApEvent term_event,
                                          const RegionUsage &usage,
                                          const FieldMask &user_mask,
                                          FieldID field_id, Operation *op,
                                          const unsigned index,
                                  std::vector<FieldDataDescriptor> &field_data,
                                          std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // TODO: reimplement this for dependent partitioning
      assert(false);
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
    // CompositeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(RegionTreeForest *ctx, DistributedID did,
                              AddressSpaceID owner_proc, RegionTreeNode *node,
                              AddressSpaceID local_proc, CompositeNode *r,
                              CompositeVersionInfo *info, 
                              RtUserEvent destroy_event, bool register_now)
      : DeferredView(ctx, encode_composite_did(did), owner_proc, local_proc, 
             node, destroy_event, register_now), root(r), version_info(info)
    {
      version_info->add_reference();
      root->set_owner_did(did);
      // Do remote registration if necessary
      if (!is_owner())
        add_base_resource_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Composite View %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(const CompositeView &rhs)
      : DeferredView(NULL, 0, 0, 0, NULL, RtUserEvent::NO_RT_USER_EVENT, false),
        root(NULL), version_info(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeView::~CompositeView(void)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> 
          functor(this, NULL);
        map_over_remote_instances(functor);
      }
      // Delete our root
      legion_delete(root);
      // See if we can delete our version info
      if (version_info->remove_reference())
        delete version_info;
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
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
    void CompositeView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, true/*add*/);
      root->notify_active(mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1, false/*add*/);
      root->notify_inactive(mutator); 
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      root->notify_valid(mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      root->notify_invalid(mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Don't take the lock, it's alright to have duplicate sends
      RtUserEvent destroy_event = Runtime::create_rt_user_event();
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
        rez.serialize(destroy_event);
        VersionInfo &info = version_info->get_version_info();
        info.pack_version_numbers(rez);
        root->pack_composite_tree(rez, target);
      }
      runtime->send_composite_view(target, rez);
      register_remote_instance(target, destroy_event);
    }

    //--------------------------------------------------------------------------
    LogicalView* CompositeView::get_subview(const ColorPoint &c)
    //--------------------------------------------------------------------------
    {
      // Composite views don't need subviews
      return this;
    }

    //--------------------------------------------------------------------------
    DeferredView* CompositeView::simplify(CompositeCloser &closer,
                                          const FieldMask &capture_mask)
    //-------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, COMPOSITE_VIEW_SIMPLIFY_CALL);
      CompositeNode *new_root = legion_new<CompositeNode>(logical_node, 
                                                          (CompositeNode*)NULL);
      FieldMask captured_mask = capture_mask;
      if (root->simplify(closer, captured_mask, new_root))
      {
        DistributedID new_did = 
          context->runtime->get_available_distributed_id(false);
        // TODO: simplify the version info here too
        // to avoid moving around extra state
        // Make a new composite view
        return legion_new<CompositeView>(context, new_did, 
            context->runtime->address_space, logical_node, 
            context->runtime->address_space, new_root, 
            version_info, RtUserEvent::NO_RT_USER_EVENT, true/*register now*/);
      }
      else // didn't change so we can delete the new root and return ourself
      {
        legion_delete(new_root);
        return this;
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              const FieldMask &copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                              CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        COMPOSITE_VIEW_ISSUE_DEFERRED_COPIES_CALL);
      LegionMap<ApEvent,FieldMask>::aligned postreductions;
      root->issue_deferred_copies(info, dst, copy_mask, 
                                  version_info->get_version_info(), 
                                  preconditions, postconditions, 
                                  postreductions, across_helper);
      if (!postreductions.empty())
      {
        // We need to merge the two post sets
        postreductions.insert(postconditions.begin(), postconditions.end());
        // Clear this out since this is where we will put the results
        postconditions.clear();
        // Now sort them and reduce them
        LegionList<EventSet>::aligned event_sets; 
        RegionTreeNode::compute_event_sets(copy_mask, 
                                           postreductions, event_sets);
        for (LegionList<EventSet>::aligned::const_iterator it = 
              event_sets.begin(); it != event_sets.end(); it++)
        {
          if (it->preconditions.size() == 1)
          {
            ApEvent post = *(it->preconditions.begin());
            if (!post.exists())
              continue;
            postconditions[post] = it->set_mask;
          }
          else
          {
            ApEvent post = Runtime::merge_events(it->preconditions);
            if (!post.exists())
              continue;
            postconditions[post] = it->set_mask;
          }
        }
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
      RtUserEvent destroy_event;
      derez.deserialize(destroy_event);
      CompositeVersionInfo *version_info = new CompositeVersionInfo();
      VersionInfo &info = version_info->get_version_info();
      info.unpack_version_numbers(derez, runtime->forest);
      CompositeNode *root = legion_new<CompositeNode>(target_node, 
                                                      (CompositeNode*)NULL);
      std::map<LogicalView*,std::pair<RtEvent,unsigned> > pending_refs;
      root->unpack_composite_tree(derez, source, runtime, pending_refs);
      if (!pending_refs.empty())
      {
        // Defer the adding of the references and the registration
        // of the view itself until it is actually ready, do not 
        // wait here to avoid blocking the virtual channel
        std::set<RtEvent> ready_events;
        for (std::map<LogicalView*,std::pair<RtEvent,unsigned> >::const_iterator
              it = pending_refs.begin(); it != pending_refs.end(); it++)
        {
          if (it->second.first.has_triggered())
          {
            // If it's already triggered we can do it now
            it->first->add_base_resource_ref(COMPOSITE_NODE_REF, 
                                             it->second.second);
            continue;
          }
          DeferCompositeNodeRefArgs args;
          args.hlr_id = HLR_DEFER_COMPOSITE_NODE_TASK_ID;
          args.view = it->first;
          args.refs = it->second.second;
          RtEvent ready = runtime->issue_runtime_meta_task(&args, sizeof(args),
              HLR_DEFER_COMPOSITE_NODE_TASK_ID, HLR_LATENCY_PRIORITY,
              NULL/*op*/, it->second.first);
          ready_events.insert(ready);
        }
        if (!ready_events.empty())
        {
          RtEvent wait_on = Runtime::merge_events(ready_events);
          DeferCompositeViewCreationArgs args;
          args.hlr_id = HLR_DEFER_CREATE_COMPOSITE_VIEW_TASK_ID;
          args.did = did;
          args.owner = owner;
          args.target_node = target_node;
          args.root = root;
          args.version_info = version_info;
          args.destroy_event = destroy_event;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
              HLR_DEFER_CREATE_COMPOSITE_VIEW_TASK_ID, HLR_LATENCY_PRIORITY,
              NULL/*op*/, wait_on);
          return;
        }
        // Otherwise fall through and do the normal case
      }
      // Nothing to wait for so we are good to go
      void *location;
      CompositeView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = legion_new_in_place<CompositeView>(location, runtime->forest, 
                                           did, owner, target_node, 
                                           runtime->address_space,
                                           root, version_info, destroy_event,
                                           false/*register now*/);
      else
        view = legion_new<CompositeView>(runtime->forest, did, owner, 
                           target_node, runtime->address_space, root, 
                           version_info, destroy_event, 
                           false/*register now*/);
      // Register only after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_deferred_node_refs(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeNodeRefArgs *ref_args = 
        (const DeferCompositeNodeRefArgs*)args;
      ref_args->view->add_base_resource_ref(COMPOSITE_NODE_REF, ref_args->refs);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_deferred_view_creation(Runtime *rt,
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeViewCreationArgs *vargs = 
        (const DeferCompositeViewCreationArgs*)args;
      void *location;
      CompositeView *view = NULL;
      if (rt->find_pending_collectable_location(vargs->did, location))
        view = legion_new_in_place<CompositeView>(location, rt->forest,
                           vargs->did, vargs->owner, vargs->target_node,
                           rt->address_space, vargs->root, vargs->version_info,
                           vargs->destroy_event, false/*register now*/);
      else
        view = legion_new<CompositeView>(rt->forest, vargs->did,
                           vargs->owner, vargs->target_node, 
                           rt->address_space, vargs->root,
                           vargs->version_info, vargs->destroy_event,
                           false/*register now*/);
      // Register only after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(RegionTreeNode* node, CompositeNode *p)
      : logical_node(node), parent(p), owner_did(0)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->add_child(this);
    }

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(const CompositeNode &rhs)
      : logical_node(NULL), parent(NULL)
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
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        legion_delete(it->first);
      }
      // Remove our resource references
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_base_resource_ref(COMPOSITE_NODE_REF))
          LogicalView::delete_logical_view(it->first);
      }
      valid_views.clear();
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_base_resource_ref(COMPOSITE_NODE_REF))
          legion_delete(it->first);
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
    void CompositeNode::add_child(CompositeNode *child)
    //--------------------------------------------------------------------------
    {
      // Referencing it should instantiate it
      children[child];
    }

    //--------------------------------------------------------------------------
    void CompositeNode::update_child(CompositeNode *child,const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.find(child) != children.end());
#endif
      children[child] |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::finalize(FieldMask &final_mask)
    //--------------------------------------------------------------------------
    {
      if (!children.empty())
      {
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it =
              children.begin(); it != children.end(); it++)
        {
          it->first->finalize(it->second);
          final_mask |= it->second;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::set_owner_did(DistributedID own_did)
    //--------------------------------------------------------------------------
    {
      owner_did = own_did;
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->set_owner_did(own_did);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::capture_physical_state(CompositeCloser &closer,
                                               PhysicalState *state,
                                               const FieldMask &close_mask,
                                               const FieldMask &capture_dirty,
                                               const FieldMask &capture_reduc)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        COMPOSITE_NODE_CAPTURE_PHYSICAL_STATE_CALL);
      // Check to see if this is the root, if it is, we need to pull
      // the valid instance views from the state
      if (parent == NULL)
      {
        // Capture any dirty fields that we need here
        dirty_mask = close_mask;
        if (!!dirty_mask)
        {
          LegionMap<LogicalView*,FieldMask>::aligned instances;
          logical_node->find_valid_instance_views(closer.ctx, state, dirty_mask,
              dirty_mask, closer.version_info, false/*needs space*/, instances);
          if (!instances.empty())
            capture_instances(closer, dirty_mask, &instances);
        }
      }
      else
      {
        // Tell the parent about our capture for all the fields
        // for which we are good regardless of what we capture
        parent->update_child(this, close_mask);
        if (!state->valid_views.empty())
        {
          dirty_mask = state->dirty_mask & capture_dirty;
          if (!!dirty_mask)
          {
            // C++ sucks sometimes
            LegionMap<LogicalView*,FieldMask>::aligned *valid_views = 
              reinterpret_cast<LegionMap<LogicalView*,FieldMask>::aligned*>(
                  &(state->valid_views));
            capture_instances(closer, dirty_mask, valid_views);
          }
        }
      }
      // Always capture any reductions that we need
      if (!state->reduction_views.empty() && !!capture_reduc)
      {
        reduction_mask = state->reduction_mask & capture_reduc;
        if (!!reduction_mask)
        {
          // More C++ suckiness
          LegionMap<ReductionView*,FieldMask>::aligned *reduction_views =
            reinterpret_cast<LegionMap<ReductionView*,FieldMask>::aligned*>(
                &(state->reduction_views));
          capture_reductions(reduction_mask, reduction_views);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool CompositeNode::capture_instances(CompositeCloser &closer,
                                          const FieldMask &capture_mask,
                        const LegionMap<LogicalView*,FieldMask>::aligned *views)
    //--------------------------------------------------------------------------
    {
      bool changed = false;
      LegionMap<DeferredView*,FieldMask>::aligned deferred_views;
      const UniqueID target_context_uid = closer.target_ctx->get_context_uid();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            views->begin(); it != views->end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        // Figure out what kind of view we have
        if (it->first->is_deferred_view())
        {
          deferred_views[it->first->as_deferred_view()] = overlap; 
        }
        else
        { 
          InstanceView *inst_view = it->first->as_instance_view();
          // Check to see if it is the same context as our target context
          if (inst_view->owner_context == target_context_uid)
          {
            // Same context so we can use the same view
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid_views.find(it->first);
            if (finder == valid_views.end())
            {
              it->first->add_base_resource_ref(COMPOSITE_NODE_REF);
              valid_views[it->first] = overlap;
            }
            else
              finder->second |= overlap;
          }
          else
          {
            // Different context, so we need the translated view
            InstanceView *alt_view = logical_node->find_context_view(
                inst_view->get_manager(), closer.target_ctx);
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid_views.find(alt_view);
            if (finder == valid_views.end())
            {
              alt_view->add_base_resource_ref(COMPOSITE_NODE_REF);
              valid_views[alt_view] = overlap;
            }
            else
              finder->second |= overlap;
            // This definitely changed
            changed = true;
          }
        }
      }
      if (!deferred_views.empty())
      {
        // Get a mask for all the fields that we did capture
        FieldMask captured;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          captured |= it->second;
        }
        // If we captured a real instance for all the fields then we are good
        if (!(capture_mask - captured))
          return changed;
        // Otherwise capture deferred instances for the rest
        for (LegionMap<DeferredView*,FieldMask>::aligned::iterator it = 
              deferred_views.begin(); it != deferred_views.end(); it++)
        {
          if (!!captured)
          {
            it->second -= captured;
            if (!it->second)
              continue;
          }
          // Simplify the composite instance
          DeferredView *simple_view = it->first->simplify(closer, it->second);
          if (simple_view != it->first)
            changed = true;
          LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
            valid_views.find(simple_view);
          if (finder == valid_views.end())
          {
            simple_view->add_base_resource_ref(COMPOSITE_NODE_REF);
            valid_views[simple_view] = it->second; 
          }
          else
            finder->second |= it->second;
        }
      }
      return changed;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::capture_reductions(const FieldMask &capture_mask,
                      const LegionMap<ReductionView*,FieldMask>::aligned *views)
    //--------------------------------------------------------------------------
    {
      // Don't need to translate reductions, 
      // they are used once and not re-used
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            views->begin(); it != views->end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
          reduction_views.find(it->first);
        if (finder == reduction_views.end())
        {
          it->first->add_base_resource_ref(COMPOSITE_NODE_REF);
          reduction_views[it->first] = overlap;
        }
        else
          finder->second |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    bool CompositeNode::simplify(CompositeCloser &closer,
                                 FieldMask &capture_mask,
                                 CompositeNode *new_parent)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        COMPOSITE_NODE_SIMPLIFY_CALL);
      // Filter the capture mask
      bool changed = closer.filter_capture_mask(logical_node, capture_mask);
      // If the set of captured nodes changed then we changed
      if (!capture_mask)
        return true;
      CompositeNode *new_node = legion_new<CompositeNode>(logical_node, 
                                                          new_parent);
      new_parent->update_child(new_node, capture_mask);
      // Simplify any of our children
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask child_capture = it->second & capture_mask;
        if (!child_capture)
        {
          // If the set of nodes captured changes, then we changed
          if (!changed)
            changed = true;
          continue;
        }
        if (it->first->simplify(closer, child_capture, new_node)) 
          changed = true;
      }
      // Now do our capture and update the closer
      if (new_node->capture_instances(closer, capture_mask, &valid_views))
        changed = true;
      new_node->capture_reductions(capture_mask, &reduction_views);
      closer.update_capture_mask(logical_node, capture_mask);
      return changed;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_deferred_copies(const TraversalInfo &info,
                                              MaterializedView *dst,
                                              const FieldMask &copy_mask,
                                            const VersionInfo &src_version_info,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                                              CopyAcrossHelper *across_helper,
                                              bool check_root) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime, 
                        COMPOSITE_NODE_ISSUE_DEFERRED_COPIES_CALL);
      // The invariant that we want to maintain for this function is that
      // it places no more than one event in the postconditions data structure
      // for any field.
      LegionMap<ApEvent,FieldMask>::aligned local_postconditions;
      // First see if we are at the root of the tree for this particular copy
      bool traverse_children = true;
      if (check_root)
      {
        CompositeNode *child = find_next_root(dst->logical_node);
        if (child != NULL)
        {
          // If we have another child, we can continue the traversal
          // If we have reductions here we need to do something special
          if (!reduction_views.empty())
          {
            // Have this path fall through to catch the reductions   
            // but don't traverse the children since we're already doing it
            child->issue_deferred_copies(info, dst, copy_mask, src_version_info,
                                  preconditions, local_postconditions, 
                                  postreductions, across_helper,
                                  true/*check root*/);
            traverse_children = false;
          }
          else // This is the common case
          {
            child->issue_deferred_copies(info, dst, copy_mask, src_version_info,
                                  preconditions, postconditions, postreductions,
                                  across_helper, true/*check root*/);
            return;
          }
        }
        else
        {
          // Otherwise we fall through and do the actual update copies
          LegionMap<LogicalView*,FieldMask>::aligned all_valid_views;
          // We have to pull down any valid views to make sure we are issuing
          // copies to all the possibly overlapping locations
          find_valid_views(copy_mask, all_valid_views);
          if (!all_valid_views.empty())
          {
            // If we have no children we can just put the results
            // straight into the postcondition otherwise put it
            // in our local postcondition
            if (children.empty() && reduction_views.empty())
            {
              issue_update_copies(info, dst, copy_mask, src_version_info,
                          preconditions, postconditions, all_valid_views, 
                          across_helper);
              return;
            }
            else
              issue_update_copies(info, dst, copy_mask, src_version_info,
                    preconditions, local_postconditions, all_valid_views, 
                    across_helper);
          }
        }
      }
      else
      {
        // Issue update copies just from this level that are needed 
        if (!valid_views.empty())
        {
          FieldMask update_mask = dirty_mask & copy_mask;
          if (!!update_mask)
          {
            // If we have no children we can just put the results
            // straight into the postcondition otherwise put it
            // in our local postcondition
            if (children.empty() && reduction_views.empty())
            {
              issue_update_copies(info, dst, update_mask, src_version_info,
                  preconditions, postconditions, valid_views, across_helper);
              return;
            }
            else
              issue_update_copies(info, dst, update_mask, src_version_info,
               preconditions, local_postconditions, valid_views, across_helper);
          }
        }
      }
      LegionMap<ApEvent,FieldMask>::aligned temp_preconditions;
      const LegionMap<ApEvent,FieldMask>::aligned *local_preconditions = NULL;
      if (traverse_children)
      {
        // Defer initialization until we find the first interfering child
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          if (!it->first->logical_node->intersects_with(dst->logical_node))
            continue;
          if (local_preconditions == NULL)
          {
            // Do the initialization
            // The preconditions going down are anything from above
            // as well as anything that we generated
            if (!local_postconditions.empty())
            {
              // Move over the local postconditions and keep track of
              // which fields we wrote to, then move over any preconditions
              // for fields that we didn't write to
              FieldMask written = copy_mask;
              temp_preconditions = local_postconditions;
              for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
                    local_postconditions.begin(); it != 
                    local_postconditions.end(); it++)
              {
                written -= it->second;
                if (!written)
                  break;
              }
              // If we had fields we didn't write to, then move over
              // the necessary preconditions for those fields
              if (!!written)
              {
                for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
                      preconditions.begin(); it != preconditions.end(); it++)
                {
                  FieldMask overlap = written & it->second;
                  if (!overlap)
                    continue;
#ifdef DEBUG_LEGION
                  assert(temp_preconditions.find(it->first) == 
                         temp_preconditions.end());
#endif
                  temp_preconditions[it->first] = overlap;
                }
              }
              local_preconditions = &temp_preconditions; 
            }
            else
              local_preconditions = &preconditions;
          }
          // Now traverse the child
          it->first->issue_deferred_copies(info, dst, overlap, src_version_info,
                            *local_preconditions, local_postconditions, 
                            postreductions, across_helper, false/*check root*/);
        }
      }
      // Handle any reductions we might have
      if (!reduction_views.empty())
      {
        if (local_preconditions != NULL)
          issue_update_reductions(info, dst, copy_mask, src_version_info,
              *local_preconditions, postreductions, across_helper);
        else if (!local_postconditions.empty())
        {
          temp_preconditions = local_postconditions;
          temp_preconditions.insert(preconditions.begin(),
                                    preconditions.end());
          issue_update_reductions(info, dst, copy_mask, src_version_info,
              temp_preconditions, postreductions, across_helper);
        }
        else
          issue_update_reductions(info, dst, copy_mask, src_version_info,
              preconditions, postreductions, across_helper);
      }
      // Quick out if we don't have any postconditions
      if (local_postconditions.empty())
        return;
      // See if we actually traversed any children
      if (local_preconditions != NULL)
      {
        // We traversed some children so we need to do a merge of our
        // local_postconditions to deduplicate events across fields
        LegionList<EventSet>::aligned event_sets; 
        RegionTreeNode::compute_event_sets(copy_mask, local_postconditions,
                                           event_sets);
        for (LegionList<EventSet>::aligned::const_iterator it = 
              event_sets.begin(); it != event_sets.end(); it++)
        {
          if (it->preconditions.size() == 1)
          {
            ApEvent post = *(it->preconditions.begin());
            if (!post.exists())
              continue;
            postconditions[post] = it->set_mask;
          }
          else
          {
            ApEvent post = Runtime::merge_events(it->preconditions);
            if (!post.exists())
              continue;
            postconditions[post] = it->set_mask;
          }
        }
      }
      else
      {
        // We didn't traverse any children so we can just copy our
        // local_postconditions into the postconditions set
        postconditions.insert(local_postconditions.begin(),
                              local_postconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    CompositeNode* CompositeNode::find_next_root(RegionTreeNode *target) const
    //--------------------------------------------------------------------------
    {
      if (children.empty())
        return NULL;
      if (children.size() == 1)
      {
        CompositeNode *child = children.begin()->first;
        if (child->logical_node->dominates(target))
          return child;
      }
      else if (logical_node->are_all_children_disjoint())
      {
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          if (it->first->logical_node->dominates(target))
            return it->first;
        }
      }
      else
      {
        CompositeNode *child = NULL;
        // Check to see if we have one child that dominates and none
        // that intersect
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          if (it->first->logical_node->dominates(target))
          {
            // Having multiple dominating children is not allowed
            if (child != NULL)
              return NULL;
            child = it->first;
            continue;
          }
          // If it doesn't dominate, but it does intersect that is not allowed
          if (it->first->logical_node->intersects_with(target))
            return NULL;
        }
        return child;
      }
      return NULL;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::find_valid_views(const FieldMask &search_mask,
                        LegionMap<LogicalView*,FieldMask>::aligned &valid) const
    //--------------------------------------------------------------------------
    {
      bool need_check = false;
      if (parent != NULL)
      {
        FieldMask up_mask = search_mask - dirty_mask;
        if (!!up_mask)
        {
          LegionMap<LogicalView*,FieldMask>::aligned valid_up;
          parent->find_valid_views(up_mask, valid_up);
          if (!valid_up.empty())
          {
            need_check = true;
            const ColorPoint &local_color = logical_node->get_color();
            for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
                  valid_up.begin(); it != valid_up.end(); it++)
            {
              LogicalView *local_view = it->first->get_subview(local_color);
              valid[local_view] = it->second;
            }
          }
        }
      }
      // Now figure out which of our views we can add
      if (!valid_views.empty())
      {
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
        {
          FieldMask overlap = search_mask & it->second;
          if (!overlap)
            continue;
          if (need_check)
          {
            LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
              valid.find(it->first);
            if (finder == valid.end())
              valid[it->first] = overlap;
            else
              finder->second |= overlap;
          }
          else
            valid[it->first] = overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_update_copies(const TraversalInfo &info, 
                                            MaterializedView *dst,
                                            FieldMask copy_mask,
                                            const VersionInfo &src_version_info,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                    const LegionMap<LogicalView*,FieldMask>::aligned &views,
                          CopyAcrossHelper *across_helper) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        COMPOSITE_NODE_ISSUE_UPDATE_COPIES_CALL);
      // This is similar to the version of this call in RegionTreeNode
      // but different in that it knows how to deal with intersections
      // See if the target manager is already valid at this level for any fields
      {
        PhysicalManager *dst_manager = dst->get_manager();
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              views.begin(); it != views.end(); it++)
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
      dst->logical_node->sort_copy_instances(info, dst, copy_mask, views,
                                             src_instances, deferred_instances);
      // Now we can issue the copy operations
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
                                             it->second, src_version_info,
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
        dst->logical_node->issue_grouped_copies(info, dst, src_preconditions,
                                 actual_copy_mask, src_instances, 
                                 src_version_info, postconditions, 
                                 across_helper, logical_node);
      }
      if (!deferred_instances.empty())
      {
        // If we have any deferred instances, issue copies to them as well
        for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator it = 
              deferred_instances.begin(); it != deferred_instances.end(); it++)
        {
          it->first->issue_deferred_copies(info, dst, it->second,
                        preconditions, postconditions, across_helper);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::issue_update_reductions(const TraversalInfo &info,
                                                MaterializedView *dst,
                                                const FieldMask &copy_mask,
                                            const VersionInfo &src_version_info,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postreductions,
                          CopyAcrossHelper *across_helper) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(logical_node->context->runtime,
                        COMPOSITE_NODE_ISSUE_UPDATE_REDUCTIONS_CALL);
      FieldMask reduce_mask = copy_mask & reduction_mask;
      if (!reduce_mask)
        return;
      std::set<ApEvent> local_preconditions;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
      {
        if (it->second * reduce_mask)
          continue;
        local_preconditions.insert(it->first);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        FieldMask overlap = reduce_mask & it->second;
        if (!overlap)
          continue;
        // Perform the reduction
        ApEvent reduce_event = it->first->perform_deferred_reduction(dst,
            overlap, src_version_info, local_preconditions, info.op,
            info.index, across_helper, 
            (dst->logical_node == it->first->logical_node) ?
              NULL : it->first->logical_node, info.map_applied_events);
        if (reduce_event.exists())
          postreductions[reduce_event] = overlap;
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::pack_composite_tree(Serializer &rez, 
                                            AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(dirty_mask);
      rez.serialize(reduction_mask);
      rez.serialize<size_t>(valid_views.size());
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(reduction_views.size());
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        // Same as above 
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(children.size());
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        rez.serialize(it->first->logical_node->get_color());
        rez.serialize(it->second);
        it->first->pack_composite_tree(rez, target);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::unpack_composite_tree(Deserializer &derez,
                                              AddressSpaceID source,
                                              Runtime *runtime,
                                              std::map<LogicalView*,
                                    std::pair<RtEvent,unsigned> > &pending_refs)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(dirty_mask);
      derez.deserialize(reduction_mask);
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(view_did, ready);
        derez.deserialize(valid_views[view]);
        if (ready.exists())
        {
          std::map<LogicalView*,std::pair<RtEvent,unsigned> >::iterator finder =
            pending_refs.find(view);
          if (finder == pending_refs.end())
            pending_refs[view] = std::pair<RtEvent,unsigned>(ready,1);
          else
            finder->second.second++;
          continue;
        }
        view->add_base_resource_ref(COMPOSITE_NODE_REF);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        DistributedID reduc_did;
        derez.deserialize(reduc_did);
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(reduc_did, ready);
        // Have to static cast since it might not be ready yet
        ReductionView *red_view = static_cast<ReductionView*>(view);
        derez.deserialize(reduction_views[red_view]);
        if (ready.exists())
        {
          std::map<LogicalView*,std::pair<RtEvent,unsigned> >::iterator finder =
            pending_refs.find(view);
          if (finder == pending_refs.end())
            pending_refs[view] = std::pair<RtEvent,unsigned>(ready,1);
          else
            finder->second.second++;
          continue;
        }
        red_view->add_base_resource_ref(COMPOSITE_NODE_REF);
      }
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        ColorPoint child_color;
        derez.deserialize(child_color);
        RegionTreeNode *child_node = logical_node->get_tree_child(child_color);
        CompositeNode *child = legion_new<CompositeNode>(child_node, this);
        derez.deserialize(children[child]);
        child->unpack_composite_tree(derez, source, runtime, pending_refs);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_gc_ref(owner_did, mutator);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->add_nested_gc_ref(owner_did, mutator);
      }
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->notify_active(mutator);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_nested_gc_ref(owner_did, mutator);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_nested_gc_ref(owner_did, mutator);
      }
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->notify_inactive(mutator);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        it->first->add_nested_valid_ref(owner_did, mutator);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        it->first->add_nested_valid_ref(owner_did, mutator);
      }
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->notify_valid(mutator);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_nested_valid_ref(owner_did, mutator);
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        // Don't worry about deletion condition since we own resource refs
        it->first->remove_nested_valid_ref(owner_did, mutator);
      }
      for (std::map<CompositeNode*,FieldMask>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->notify_invalid(mutator);
      }
    }

    /////////////////////////////////////////////////////////////
    // FillView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc, AddressSpaceID local_proc,
                       RegionTreeNode *node, FillViewValue *val, 
                       RtUserEvent destroy_event, bool register_now)
      : DeferredView(ctx, encode_fill_did(did), owner_proc, local_proc, 
                     node, destroy_event, register_now), value(val)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(value != NULL);
#endif
      value->add_reference();
      if (!is_owner())
        add_base_resource_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Fill View %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::FillView(const FillView &rhs)
      : DeferredView(NULL, 0, 0, 0, NULL, RtUserEvent::NO_RT_USER_EVENT, false),
        value(NULL)
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
      if (is_owner())
      {
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> 
          functor(this, NULL);
        map_over_remote_instances(functor);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
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
    LogicalView* FillView::get_subview(const ColorPoint &c)
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
      RtUserEvent destroy_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
        rez.serialize(logical_node->as_region_node()->handle);
        rez.serialize(value->value_size);
        rez.serialize(value->value, value->value_size);
        rez.serialize(destroy_event);
      }
      runtime->send_fill_view(target, rez);
      // We've now done the send so record it
      register_remote_instance(target, destroy_event);
    }

    //--------------------------------------------------------------------------
    DeferredView* FillView::simplify(CompositeCloser &closer, 
                                     const FieldMask &capture_mask)
    //--------------------------------------------------------------------------
    {
      // Fill views simplify easily
      return this;
    }

    //--------------------------------------------------------------------------
    void FillView::issue_deferred_copies(const TraversalInfo &info,
                                         MaterializedView *dst,
                                         const FieldMask &copy_mask,
                    const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                          LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                         CopyAcrossHelper *across_helper)
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
        std::vector<Domain::CopySrcDstField> dst_fields;
        dst->copy_to(pre_set.set_mask, dst_fields, across_helper);
        ApEvent fill_pre = Runtime::merge_events(pre_set.preconditions);
        // Issue the fill command
        // Only apply an intersection if the destination logical node
        // is different than our logical node
        ApEvent fill_post = dst->logical_node->issue_fill(info.op, dst_fields,
                                  value->value, value->value_size, fill_pre, 
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
      RtUserEvent destroy_event;
      derez.deserialize(destroy_event);
      
      RegionNode *target_node = runtime->forest->get_node(handle);
      FillView::FillViewValue *fill_value = 
                      new FillView::FillViewValue(value, value_size);
      void *location;
      FillView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = legion_new_in_place<FillView>(location, runtime->forest, did,
                                      owner_space, runtime->address_space,
                                      target_node, fill_value, destroy_event,
                                      false/*register now*/);
      else
        view = legion_new<FillView>(runtime->forest, did, owner_space,
                                    runtime->address_space, target_node, 
                                    fill_value, destroy_event, 
                                    false/*register now*/);
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    /////////////////////////////////////////////////////////////
    // ReductionView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(RegionTreeForest *ctx, DistributedID did,
                                 AddressSpaceID own_sp, AddressSpaceID loc_sp,
                                 AddressSpaceID log_own, RegionTreeNode *node, 
                                 ReductionManager *man, UniqueID own_ctx, 
                                 RtUserEvent destroy_event, bool register_now)
      : InstanceView(ctx, encode_reduction_did(did), own_sp, loc_sp, log_own, 
          node, own_ctx, destroy_event, register_now), 
        manager(man), remote_request_event(RtEvent::NO_RT_EVENT)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      logical_node->register_instance_view(manager, owner_context, this);
      manager->add_nested_resource_ref(did);
      if (!is_owner())
        add_base_resource_ref(REMOTE_DID_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Reduction View %ld %d %ld", did, local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : InstanceView(NULL, 0, 0, 0, 0, NULL, 0, 
          RtUserEvent::NO_RT_USER_EVENT, false), manager(NULL)
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
      if (is_owner())
      {
        // If we're the owner, remove our valid references on remote nodes
        UpdateReferenceFunctor<RESOURCE_REF_KIND,false/*add*/> 
          functor(this, NULL);
        map_over_remote_instances(functor);
      }
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
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %ld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
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
                                          Operation *op, unsigned index,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime,REDUCTION_VIEW_PERFORM_REDUCTION_CALL);
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, reduce_mask, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, src_fields);
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      target->find_copy_preconditions(manager->redop, false/*reading*/, 
              reduce_mask, version_info, op->get_unique_op_id(), index, 
              local_space, preconditions, map_applied_events);
      this->find_copy_preconditions(manager->redop, true/*reading*/, 
           reduce_mask, version_info, op->get_unique_op_id(), index, 
           local_space, preconditions, map_applied_events);
      std::set<ApEvent> event_preconds;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
      {
        event_preconds.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(event_preconds); 
      ApEvent reduce_post = manager->issue_reduction(op, src_fields, dst_fields,
                                                   target->logical_node, 
                                                   reduce_pre,
                                                   fold, true/*precise*/,
                                                   NULL/*intersect*/);
      target->add_copy_user(manager->redop, reduce_post, version_info,
                           op->get_unique_op_id(), index, reduce_mask, 
                           false/*reading*/, local_space, map_applied_events);
      this->add_copy_user(manager->redop, reduce_post, version_info,
                         op->get_unique_op_id(), index, reduce_mask, 
                         true/*reading*/, local_space, map_applied_events);
    } 

    //--------------------------------------------------------------------------
    ApEvent ReductionView::perform_deferred_reduction(MaterializedView *target,
                                                    const FieldMask &red_mask,
                                                const VersionInfo &version_info,
                                                   const std::set<ApEvent> &pre,
                                                  Operation *op, unsigned index,
                                                    CopyAcrossHelper *helper,
                                                    RegionTreeNode *intersect,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_CALL);
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, red_mask, 
                                    dst_fields, helper);
      this->reduce_from(manager->redop, red_mask, src_fields);

      LegionMap<ApEvent,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/, red_mask, 
                              version_info, op->get_unique_op_id(), index, 
                              local_space, src_pre, map_applied_events);
      std::set<ApEvent> preconditions = pre;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(preconditions); 
      ApEvent reduce_post = target->logical_node->issue_copy(op,
                                            src_fields, dst_fields,
                                            reduce_pre, intersect,
                                            manager->redop, fold);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, version_info,
                    op->get_unique_op_id(), index, red_mask, 
                    true/*reading*/, local_space, map_applied_events);
      return reduce_post;
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::perform_deferred_across_reduction(
                              MaterializedView *target, FieldID dst_field, 
                              FieldID src_field, unsigned src_index, 
                              const VersionInfo &version_info,
                              const std::set<ApEvent> &preconds,
                              Operation *op, unsigned index,
                              RegionTreeNode *intersect,
                              std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_ACROSS_CALL);
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      const bool fold = false;
      target->copy_field(dst_field, dst_fields);
      FieldMask red_mask; red_mask.set_bit(src_index);
      this->reduce_from(manager->redop, red_mask, src_fields);
      LegionMap<ApEvent,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/, red_mask, 
                              version_info, op->get_unique_op_id(), index, 
                              local_space, src_pre, map_applied_events);
      std::set<ApEvent> preconditions = preconds;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(preconditions); 
      ApEvent reduce_post = manager->issue_reduction(op, src_fields, dst_fields,
                                                   intersect, reduce_pre,
                                                   fold, false/*precise*/,
                                                   target->logical_node);
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, version_info,
                    op->get_unique_op_id(), index, red_mask, 
                    true/*reading*/, local_space, map_applied_events);
      return reduce_post;
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
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_FIND_COPY_PRECONDITIONS_CALL);
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
                                      const VersionInfo &version_info,
                                      const UniqueID creator_op_id,
                                      const unsigned index,
                                      const FieldMask &mask, bool reading,
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
        if (reading)
        {
          RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
          user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                          creator_op_id, index);
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                          creator_op_id, index);
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
                                              const VersionInfo &version_info,
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
                                 const VersionInfo &version_info,
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
      const bool reading = IS_READ_ONLY(usage);
      PhysicalUser *new_user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                                        op_id, index);
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
                                          const VersionInfo &version_info,
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
      // Who cares just hold the lock in exlcusive mode, this analysis
      // shouldn't be too expensive for reduction views
      bool issue_collect = false;
      PhysicalUser *new_user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                                        op_id, index);
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
    void ReductionView::add_initial_user(ApEvent term_event, 
                                         const RegionUsage &usage,
                                         const FieldMask &user_mask,
                                         const UniqueID op_id,
                                         const unsigned index)
    //--------------------------------------------------------------------------
    {
      // We don't use field versions for doing interference tests on
      // reductions so there is no need to record it
      PhysicalUser *user = legion_new<PhysicalUser>(usage, ColorPoint(), 
                                                    op_id, index);
      add_physical_user(user, IS_READ_ONLY(usage), term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }
 
    //--------------------------------------------------------------------------
    bool ReductionView::reduce_to(ReductionOpID redop, 
                                  const FieldMask &reduce_mask,
                              std::vector<Domain::CopySrcDstField> &dst_fields,
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
                              std::vector<Domain::CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == manager->redop);
#endif
      manager->find_field_offsets(reduce_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_to(const FieldMask &copy_mask,
                               std::vector<Domain::CopySrcDstField> &dst_fields,
                                CopyAcrossHelper *across_helper)
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
      RtUserEvent destroy_event = Runtime::create_rt_user_event();
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
        rez.serialize(destroy_event);
      }
      runtime->send_reduction_view(target, rez);
      register_remote_instance(target, destroy_event);
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
      RtUserEvent destroy_event;
      derez.deserialize(destroy_event);

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
        view = legion_new_in_place<ReductionView>(location, runtime->forest,
                                           did, owner_space,
                                           runtime->address_space,
                                           logical_owner,
                                           target_node, red_manager,
                                           context_uid, destroy_event,
                                           false/*register now*/);
      else
        view = legion_new<ReductionView>(runtime->forest, did, owner_space,
                                  runtime->address_space, logical_owner,
                                  target_node, red_manager, context_uid,
                                  destroy_event, false/*register now*/);
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
      RtUserEvent request_event;
      RtEvent wait_on;
      // If we don't have any registered readers, we have to ask
      // the owner for all the current reducers
      {
        AutoLock v_lock(view_lock);  
        if (!remote_request_event.exists())
        {
          request_event = Runtime::create_rt_user_event();
          wait_on = request_event;
        }
        else // request was already sent
          wait_on = remote_request_event;
      }
      // If we made the event send the request
      if (request_event.exists())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(request_event);
        }
        context->runtime->send_view_update_request(logical_owner, rez);
      }
      if (!wait_on.has_triggered())
        wait_on.wait();
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
            rez.serialize(*(it->second.users.single_user));
            rez.serialize(it->second.user_mask);
          }
          else
          {
            rez.serialize<size_t>(it->second.users.multi_users->size());
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = it->second.users.multi_users->begin(); uit != 
                  it->second.users.multi_users->end(); uit++)
            {
              rez.serialize(*(uit->first));
              rez.serialize(uit->second);
            }
          }
        }
      }
      runtime->send_view_update_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void ReductionView::process_update_response(Deserializer &derez,
                                                RtUserEvent done_event)
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
#ifdef DEBUG_LEGION
          // should never have this event before now
          assert(reduction_users.find(term_event) == reduction_users.end());
#endif
          EventUsers current_users = reduction_users[term_event];
          size_t num_users;
          derez.deserialize(num_users);
          if (num_users == 1)
          {
            current_users.users.single_user = legion_new<PhysicalUser>();
            derez.deserialize(*(current_users.users.single_user));
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
              PhysicalUser *user = legion_new<PhysicalUser>();
              derez.deserialize(*user);
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
                                              AddressSpaceID source)
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
        PhysicalUser *user;
        // We don't use field versions for doing interference 
        // tests on reductions so no need to record it
        if (reading)
        {
          RegionUsage usage(READ_ONLY, EXCLUSIVE, 0);
          user = legion_new<PhysicalUser>(usage, ColorPoint(), op_id, index);
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = legion_new<PhysicalUser>(usage, ColorPoint(), op_id, index);
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
        PhysicalUser *new_user = 
          legion_new<PhysicalUser>(usage, ColorPoint(), op_id, index);
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

