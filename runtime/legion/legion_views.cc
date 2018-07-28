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
#include "legion/legion_trace.h"
#include "legion/legion_context.h"
#include "legion/legion_replication.h"

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

#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
      inst_view->process_update_response(derez, done_event, 
                                         source, runtime->forest);
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
#else // DISTRIBUTED_INSTANCE_VIEWS
    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_copy_preconditions(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      ReductionOpID redop;
      derez.deserialize(redop);
      bool reading, single_copy, restrict_out, can_filter;
      derez.deserialize<bool>(reading);
      derez.deserialize<bool>(single_copy);
      derez.deserialize<bool>(restrict_out);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression *copy_expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      VersionInfo version_info;
      if (!reading)
        version_info.unpack_version_numbers(derez, runtime->forest);
      else
        version_info.unpack_upper_bound_node(derez, runtime->forest);
      UniqueID creator_op_id;
      derez.deserialize(creator_op_id);
      unsigned index;
      derez.deserialize(index);
      derez.deserialize<bool>(can_filter);
      int pop_count = FieldMask::pop_count(copy_mask);
      std::map<int,ApUserEvent> trigger_events;
      for (int idx = 0; idx < pop_count; idx++)
      {
        int field_index;
        derez.deserialize(field_index);
        derez.deserialize(trigger_events[field_index]);
      }
      RtUserEvent applied_event;
      derez.deserialize(applied_event);

      // Do the call
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      inst_view->find_copy_preconditions(redop, reading, single_copy, 
          restrict_out, copy_mask, copy_expr, &version_info, creator_op_id, 
          index, source, preconditions, applied_events, trace_info, can_filter);

      // Sort the event preconditions into equivalence sets
      LegionList<FieldSet<ApEvent> >::aligned event_sets;
      compute_field_sets<ApEvent>(copy_mask, preconditions, event_sets);
      for (LegionList<FieldSet<ApEvent> >::aligned::const_iterator it = 
            event_sets.begin(); it != event_sets.end(); it++)
      {
        int next_start = 0;
        pop_count = FieldMask::pop_count(it->set_mask);
        ApEvent precondition = Runtime::merge_events(&trace_info, it->elements);
#ifndef LEGION_SPY
        if (!trace_info.recording && precondition.has_triggered_faultignorant())
          continue;
#endif
        for (int idx = 0; idx < pop_count; idx++)
        {
          int field_index = it->set_mask.find_next_set(next_start);
          std::map<int,ApUserEvent>::iterator finder = 
            trigger_events.find(field_index); 
#ifdef DEBUG_LEGION
          assert(finder != trigger_events.end());
#endif
          Runtime::trigger_event(finder->second, precondition);
          trigger_events.erase(finder);
          next_start = field_index + 1;
        }
      }
      // Trigger the remaining events that have no preconditions
      while (!trigger_events.empty())
      {
        std::map<int,ApUserEvent>::iterator first = trigger_events.begin();
        Runtime::trigger_event(first->second);
        trigger_events.erase(first);
      }
      // Trigger the applied event with the preconditions
      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
                               Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_add_copy(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      ReductionOpID redop;
      derez.deserialize(redop);
      ApEvent copy_term;
      derez.deserialize(copy_term);
      UniqueID creator_op_id;
      derez.deserialize(creator_op_id);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression *copy_expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      bool reading, restrict_out;
      derez.deserialize<bool>(reading);
      derez.deserialize<bool>(restrict_out);
      VersionInfo version_info;
      if (!reading)
        version_info.unpack_version_numbers(derez, runtime->forest);
      else
        version_info.unpack_upper_bound_node(derez, runtime->forest);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);

      // Do the base call
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      inst_view->add_copy_user(redop, copy_term, &version_info, copy_expr,
                               creator_op_id, index, copy_mask, reading, 
                               restrict_out, source, applied_events,trace_info);

      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_user_preconditions(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      RegionUsage usage;
      derez.deserialize(usage);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      VersionInfo version_info;
      if (IS_WRITE(usage))
        version_info.unpack_version_numbers(derez, runtime->forest);
      else
        version_info.unpack_upper_bound_node(derez, runtime->forest);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      ApUserEvent result_event;
      derez.deserialize(result_event);

      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      ApEvent result = inst_view->find_user_precondition(usage, term_event, 
          user_mask, op_id, index, &version_info, applied_events, trace_info);

      Runtime::trigger_event(result_event, result);
      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
                               Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_add_user(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      RegionUsage usage;
      derez.deserialize(usage);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      VersionInfo version_info;
      if (IS_WRITE(usage))
        version_info.unpack_version_numbers(derez, runtime->forest);
      else
        version_info.unpack_upper_bound_node(derez, runtime->forest);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);

      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      inst_view->add_user_base(usage, term_event, user_mask, op_id, index,
                       source, &version_info, applied_events, trace_info);

      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
                               Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_add_user_fused(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      RegionUsage usage;
      derez.deserialize(usage);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      VersionInfo version_info;
      if (IS_WRITE(usage))
        version_info.unpack_version_numbers(derez, runtime->forest);
      else
        version_info.unpack_upper_bound_node(derez, runtime->forest);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      bool update_versions;
      derez.deserialize<bool>(update_versions);
      ApUserEvent result_event;
      derez.deserialize(result_event);

      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      ApEvent result = inst_view->add_user_fused_base(usage, term_event, 
                         user_mask, op_id, index, &version_info, source, 
                         applied_events, trace_info, update_versions);

      Runtime::trigger_event(result_event, result);
      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
                               Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }
#endif // not DISTRIBUTED_INSTANCE_VIEWS

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
                                               IndexSpaceExpression *copy_expr,
                                               VersionTracker *versions,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                              bool can_filter)
    //--------------------------------------------------------------------------
    {
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        // If this is not the logical owner send a message to the 
        // logical owner to perform the analysis
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(redop);
          rez.serialize<bool>(reading);
          rez.serialize<bool>(single_copy);
          rez.serialize<bool>(restrict_out);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          if (!reading)
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize<bool>(can_filter);
          // Make 1 Ap user event per field and add it to the set
          int pop_count = FieldMask::pop_count(copy_mask);
          int next_start = 0;
          for (int idx = 0; idx < pop_count; idx++)
          {
            int field_index = copy_mask.find_next_set(next_start);
#ifdef DEBUG_LEGION
            assert(field_index >= 0);
#endif
            ApUserEvent field_ready = Runtime::create_ap_user_event();
            rez.serialize(field_index);
            rez.serialize(field_ready);
            preconditions[field_ready].set_bit(field_index); 
            // We'll start looking again at the next index after this one
            next_start = field_index + 1;
          }
          // Make a Rt user event to signal when we are done
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          rez.serialize(applied_event);
          applied_events.insert(applied_event);
        }
        runtime->send_instance_view_find_copy_preconditions(logical_owner, rez);
        return;
      }
#endif
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
      // If we can filter we can do the normal case, otherwise
      // we do the above case where we don't filter
      if (can_filter)
        find_local_copy_preconditions(redop, reading, single_copy, restrict_out,
                                      copy_mask, INVALID_COLOR, copy_expr, 
                                      versions, creator_op_id, index, source, 
                                      preconditions, applied_events,trace_info);
      else
        find_local_copy_preconditions_above(redop, reading, single_copy, 
                                      restrict_out, copy_mask, INVALID_COLOR, 
                                      copy_expr, versions,creator_op_id,index,
                                      source, preconditions, applied_events,
                                      trace_info, false/*actually above*/);
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, single_copy,
          restrict_out, copy_mask, local_point, copy_expr, versions, 
          creator_op_id, index, source,preconditions,applied_events,trace_info);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_above(ReductionOpID redop,
                                                         bool reading,
                                                         bool single_copy,
                                                         bool restrict_out,
                                                const FieldMask &copy_mask,
                                                const LegionColor child_color,
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      find_local_copy_preconditions_above(redop, reading, single_copy, 
        restrict_out, copy_mask, child_color, user_expr, versions, 
        creator_op_id, index, source, preconditions, applied_events,trace_info);
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_point = logical_node->get_color();
        parent->find_copy_preconditions_above(redop, reading, single_copy, 
          restrict_out, copy_mask, local_point, user_expr, versions, 
          creator_op_id, index, source,preconditions,applied_events,trace_info);
      }
    }
    
    //--------------------------------------------------------------------------
    void MaterializedView::find_local_copy_preconditions(ReductionOpID redop,
                                                         bool reading,
                                                         bool single_copy,
                                                         bool restrict_out,
                                                const FieldMask &copy_mask,
                                                const LegionColor child_color,
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      // If we are not the logical owner, we need to see if we are up to date 
      if (!is_logical_owner())
      {
        // We are also reading if we are doing a reductions
        perform_remote_valid_check(copy_mask, versions,reading || (redop != 0));
      }
#elif defined(DEBUG_LEGION)
      assert(is_logical_owner());
#endif
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
                               user_expr, creator_op_id, index, preconditions,
                               dead_events, filter_current_users,
                               observed, non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      user_expr, creator_op_id, index, 
                                      preconditions, dead_events, trace_info);
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
                                       child_color, user_expr, creator_op_id,
                                       index, preconditions, dead_events, 
                                       filter_current_users,
                                       observed, non_dominated, trace_info);
        }
        else // the normal case with no write-skip
          find_current_preconditions<true/*track*/>(copy_mask, usage, 
                                     child_color, user_expr, creator_op_id,
                                     index, preconditions, dead_events, 
                                     filter_current_users, 
                                     observed, non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color,
                                      user_expr, creator_op_id, index, 
                                      preconditions, dead_events, trace_info);
      }
      if ((!dead_events.empty() || 
           !filter_previous_users.empty() || !filter_current_users.empty() ||
           !advance_versions.empty() || !add_versions.empty()) &&
          !trace_info.recording)
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
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                                  const bool actually_above)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      // If we are not the logical owner and we're not actually already above
      // the base level, then we need to see if we are up to date 
      // Otherwise we did this check at the base level and it went all
      // the way up the tree so we are already good
      if (!is_logical_owner() && !actually_above)
      {
        // We are also reading if we are doing reductions
        perform_remote_valid_check(copy_mask, versions,reading || (redop != 0));
      }
#elif defined(DEBUG_LEGION)
      assert(is_logical_owner());
#endif
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
                                   child_color, user_expr, creator_op_id, 
                                   index, preconditions, dead_events, 
                                   filter_current_users,observed,
                                   non_dominated, trace_info);
        // No domination above
        find_previous_preconditions(copy_mask, usage, child_color,
                                    user_expr, creator_op_id, index, 
                                    preconditions, dead_events, trace_info);
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
                                       child_color, user_expr, creator_op_id,
                                       index, preconditions, dead_events, 
                                       filter_current_users, observed, 
                                       non_dominated, trace_info);
        }
        else // the normal case with no write-skip
          find_current_preconditions<false/*track*/>(copy_mask, usage, 
                                     child_color, user_expr, creator_op_id,
                                     index, preconditions, dead_events, 
                                     filter_current_users, observed, 
                                     non_dominated, trace_info);
        // No domination above
        find_previous_preconditions(copy_mask, usage, child_color,
                                    user_expr, creator_op_id, index, 
                                    preconditions, dead_events, trace_info);
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
                                         IndexSpaceExpression *copy_expr,
                                         const UniqueID creator_op_id,
                                         const unsigned index,
                                         const FieldMask &copy_mask, 
                                         bool reading, bool restrict_out,
                                         const AddressSpaceID source,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        // If we're not the logical owner, send a message to the
        // owner to add the copy user
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(redop);
          rez.serialize(copy_term);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize<bool>(reading);
          rez.serialize<bool>(restrict_out);
          if (!reading)
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
        }
        runtime->send_instance_view_add_copy_user(logical_owner, rez);
        return;
      }
#endif
      RegionUsage usage;
      usage.redop = redop;
      usage.prop = EXCLUSIVE;
      if (reading)
        usage.privilege = READ_ONLY;
      else if (redop > 0)
        usage.privilege = REDUCE;
      else
        usage.privilege = READ_WRITE;
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color, copy_expr, 
                                versions, creator_op_id, index, restrict_out,
                                copy_mask, source, applied_events, trace_info);
      }
      add_local_copy_user(usage, copy_term, true/*base*/, restrict_out,
          INVALID_COLOR, copy_expr, versions, creator_op_id, index, 
          copy_mask, source, applied_events, trace_info);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user_above(const RegionUsage &usage, 
                                               ApEvent copy_term, 
                                               const LegionColor child_color,
                                               IndexSpaceExpression *user_expr,
                                               VersionTracker *versions,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const bool restrict_out,
                                               const FieldMask &copy_mask,
                                               const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_copy_user_above(usage, copy_term, local_color, user_expr,
                                  versions, creator_op_id, index, restrict_out, 
                                  copy_mask, source, applied_events, trace_info);
      }
      add_local_copy_user(usage, copy_term, false/*base*/, restrict_out,
                      child_color, user_expr, versions, creator_op_id, 
                      index, copy_mask, source, applied_events, trace_info);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_local_copy_user(const RegionUsage &usage, 
                                               ApEvent copy_term,
                                               bool base_user,bool restrict_out,
                                               const LegionColor child_color,
                                               IndexSpaceExpression *user_expr,
                                               VersionTracker *versions,
                                               const UniqueID creator_op_id,
                                               const unsigned index,
                                               const FieldMask &copy_mask,
                                               const AddressSpaceID source,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
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
          user_expr->pack_expression(rez, logical_owner);
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
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      else
      {
        PhysicalUser *user = new PhysicalUser(usage, child_color, 
                                      creator_op_id, index, user_expr);
        user->add_reference();
        bool issue_collect = false;
        {
          AutoLock v_lock(view_lock);
          add_current_user(user, copy_term, copy_mask); 
          if (trace_info.recording)
          {
#ifdef DEBUG_LEGION
            assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
            trace_info.tpl->record_outstanding_gc_event(this, copy_term);
          }
          else
          {
            if (base_user)
              issue_collect = (outstanding_gc_events.find(copy_term) ==
                                outstanding_gc_events.end());
            outstanding_gc_events.insert(copy_term);
          }
        }
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(copy_term, &mutator);
        }
      }
#else
      PhysicalUser *user = new PhysicalUser(usage, child_color, creator_op_id,
                                            index, user_expr);
      user->add_reference();
      bool issue_collect = false;
      {
        AutoLock v_lock(view_lock);
        add_current_user(user, copy_term, copy_mask); 
        if (trace_info.recording)
        {
#ifdef DEBUG_LEGION
          assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
          trace_info.tpl->record_outstanding_gc_event(this, copy_term);
        }
        else
        {
          if (base_user)
            issue_collect = (outstanding_gc_events.find(copy_term) ==
                              outstanding_gc_events.end());
          outstanding_gc_events.insert(copy_term);
        }
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
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::find_user_precondition(
                          const RegionUsage &usage, ApEvent term_event,
                          const FieldMask &user_mask, const UniqueID op_id,
                          const unsigned index, VersionTracker *versions,
                          std::set<RtEvent> &applied_events,
                          const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        ApUserEvent result = Runtime::create_ap_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          if (IS_WRITE(usage))
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
          rez.serialize(result);
        }
        runtime->send_instance_view_find_user_preconditions(logical_owner, rez);
        return result;
      }
#endif
      std::set<ApEvent> wait_on_events;
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      IndexSpaceExpression *user_expr = 
        logical_node->get_index_space_expression();
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, INVALID_COLOR, 
          user_expr, versions, op_id, index, user_mask, 
          wait_on_events, applied_events, trace_info);
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
                              user_expr, versions, op_id, index, user_mask, 
                              wait_on_events, applied_events, trace_info);
      }
      return Runtime::merge_events(&trace_info, wait_on_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_user_preconditions_above(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      // Do the precondition analysis on the way up
      find_local_user_preconditions_above(usage, term_event, child_color, 
                          user_expr, versions, op_id, index, user_mask, 
                          preconditions, applied_events, trace_info);
      // Go up the tree if we have to
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->find_user_preconditions_above(usage, term_event, local_color, 
                              user_expr, versions, op_id, index, user_mask, 
                              preconditions, applied_events, trace_info);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_local_user_preconditions(
                                                const RegionUsage &usage,
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
#elif defined(DEBUG_LEGION)
      assert(is_logical_owner());
#endif
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users, 
                                           filter_previous_users;
      if (IS_READ_ONLY(usage))
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<true/*track*/>(user_mask, usage, child_color,
                                   user_expr, term_event, op_id, index, 
                                   preconditions, dead_events, 
                                   filter_current_users, observed,
                                   non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      user_expr, term_event, op_id, index,
                                      preconditions, dead_events, trace_info);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<true/*track*/>(user_mask, usage, child_color,
                                   user_expr, term_event, op_id, index, 
                                   preconditions, dead_events, 
                                   filter_current_users, observed,
                                   non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, filter_previous_users);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(previous_mask, usage, child_color, 
                                      user_expr, term_event, op_id, index,
                                      preconditions, dead_events, trace_info);
      }
      if ((!dead_events.empty() || 
           !filter_previous_users.empty() || !filter_current_users.empty()) &&
          !trace_info.recording)
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
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                                const bool actually_above)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
#elif defined(DEBUG_LEGION)
      assert(is_logical_owner());
#endif
      std::set<ApEvent> dead_events;
      LegionMap<ApEvent,FieldMask>::aligned filter_current_users;
      if (IS_READ_ONLY(usage))
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<false/*track*/>(user_mask, usage, 
                                   child_color, user_expr,
                                   term_event, op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated, trace_info);
        // No domination above
        find_previous_preconditions(user_mask, usage, child_color, 
                                    user_expr, term_event, op_id, index,
                                    preconditions, dead_events, trace_info);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions<false/*track*/>(user_mask, usage, 
                                   child_color, user_expr,
                                   term_event, op_id, index, preconditions, 
                                   dead_events, filter_current_users, 
                                   observed, non_dominated, trace_info);
        // No domination above
        find_previous_preconditions(user_mask, usage, child_color, 
                                    user_expr, term_event, op_id, index,
                                    preconditions, dead_events, trace_info);
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
                                    const FieldMask &user_mask, 
                                    Operation *op, const unsigned index,
                                    VersionTracker *versions,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      const UniqueID op_id = op->get_unique_op_id(); 
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          if (IS_WRITE(usage))
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
        }
        runtime->send_instance_view_add_user(logical_owner, rez);
        if (IS_ATOMIC(usage))
          find_atomic_reservations(user_mask, op, IS_WRITE(usage));
        return;
      }
#endif
      add_user_base(usage, term_event, user_mask, op_id, index,
        context->runtime->address_space, versions, applied_events, trace_info);
      if (IS_ATOMIC(usage))
        find_atomic_reservations(user_mask, op, IS_WRITE(usage));
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_base(const RegionUsage &usage, 
                                         ApEvent term_event,
                                         const FieldMask &user_mask, 
                                         const UniqueID op_id, 
                                         const unsigned index, 
                                         const AddressSpaceID source,
                                         VersionTracker *versions,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      bool need_version_update = false;
      if (IS_WRITE(usage))
      {
        FieldVersions advance_versions;
        versions->get_advance_versions(logical_node, true/*base*/,
                                       user_mask, advance_versions);
        need_version_update = update_version_numbers(user_mask,advance_versions,
                                                     source, applied_events);
      }
      IndexSpaceExpression *user_expr = 
        logical_node->get_index_space_expression();
      // Go up the tree if necessary 
      if ((parent != NULL) && !versions->is_upper_bound_node(logical_node))
      {
        const LegionColor local_color = logical_node->get_color();
        parent->add_user_above(usage, term_event, local_color, user_expr,
            versions, op_id, index, user_mask, need_version_update, 
            source, applied_events, trace_info);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                   INVALID_COLOR, user_expr, true/*base*/, versions,
                   op_id, index, user_mask, source, applied_events, trace_info);
      // Launch the garbage collection task, if it doesn't exist
      // then the user wasn't registered anyway, see add_local_user
      if (issue_collect)
      {
        WrapperReferenceMutator mutator(applied_events);
        defer_collect_user(term_event, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above(const RegionUsage &usage,
                                          ApEvent term_event,
                                          const LegionColor child_color,
                                          IndexSpaceExpression *user_expr,
                                          VersionTracker *versions,
                                          const UniqueID op_id,
                                          const unsigned index,
                                          const FieldMask &user_mask,
                                          const bool need_version_update,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events,
                                          const PhysicalTraceInfo &trace_info)
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
        parent->add_user_above(usage, term_event, local_color, user_expr,
            versions, op_id, index, user_mask, need_update_above, 
            source, applied_events, trace_info);
      }
      add_local_user(usage, term_event, child_color, user_expr, false/*base*/,
        versions, op_id, index, user_mask, source, applied_events, trace_info);
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_local_user(const RegionUsage &usage,
                                          ApEvent term_event,
                                          const LegionColor child_color,
                                          IndexSpaceExpression *user_expr,
                                          const bool base_user,
                                          VersionTracker *versions,
                                          const UniqueID op_id,
                                          const unsigned index,
                                          const FieldMask &user_mask,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events,
                                          const PhysicalTraceInfo &trace_info)
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
          rez.serialize<bool>(false); // restrict out
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize(child_color);
          user_expr->pack_expression(rez, logical_owner);
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
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      else
      {
        PhysicalUser *new_user = 
          new PhysicalUser(usage, child_color, op_id,index, user_expr);
        new_user->add_reference();
        // No matter what, we retake the lock in exclusive mode so we
        // can handle any clean-up and add our user
        AutoLock v_lock(view_lock);
        // Finally add our user and return if we need to issue a GC meta-task
        add_current_user(new_user, term_event, user_mask);
        if (trace_info.recording)
        {
#ifdef DEBUG_LEGION
          assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
          trace_info.tpl->record_outstanding_gc_event(this, term_event);
          return false;
        }
        else if (outstanding_gc_events.find(term_event) == 
                  outstanding_gc_events.end())
        {
          outstanding_gc_events.insert(term_event);
          return (child_color == INVALID_COLOR);
        }
      }
#else
      PhysicalUser *new_user = 
        new PhysicalUser(usage, child_color, op_id, index, user_expr);
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
      if (trace_info.recording)
      {
#ifdef DEBUG_LEGION
        assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
        trace_info.tpl->record_outstanding_gc_event(this, term_event);
        return false;
      }
      else if (outstanding_gc_events.find(term_event) == 
               outstanding_gc_events.end())
      {
        outstanding_gc_events.insert(term_event);
        return (child_color == INVALID_COLOR);
      }
#endif
      return false;
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::add_user_fused(const RegionUsage &usage, 
                                           ApEvent term_event,
                                           const FieldMask &user_mask, 
                                           Operation *op,const unsigned index,
                                           VersionTracker *versions,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info,
                                           bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
      const UniqueID op_id = op->get_unique_op_id();
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        ApUserEvent result = Runtime::create_ap_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          if (IS_WRITE(usage))
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
          rez.serialize(update_versions);
          rez.serialize(result);
        }
        runtime->send_instance_view_add_user_fused(logical_owner, rez);
        if (IS_ATOMIC(usage))
          find_atomic_reservations(user_mask, op, IS_WRITE(usage));
        return result;
      }
#endif
      ApEvent result = add_user_fused_base(usage, term_event, user_mask, op_id, 
                              index, versions, context->runtime->address_space,
                              applied_events, trace_info, update_versions);
      if (IS_ATOMIC(usage))
        find_atomic_reservations(user_mask, op, IS_WRITE(usage));
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::add_user_fused_base(const RegionUsage &usage, 
                                           ApEvent term_event,
                                           const FieldMask &user_mask, 
                                           const UniqueID op_id,
                                           const unsigned index,
                                           VersionTracker *versions,
                                           const AddressSpaceID source,
                                           std::set<RtEvent> &applied_events,
                                           const PhysicalTraceInfo &trace_info,
                                           bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> wait_on_events;
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        wait_on_events.insert(start_use_event);
      IndexSpaceExpression *user_expr = 
        logical_node->get_index_space_expression();
      // Find our local preconditions
      find_local_user_preconditions(usage, term_event, INVALID_COLOR, 
                     user_expr, versions, op_id, index, user_mask, 
                     wait_on_events, applied_events, trace_info);
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
                              user_expr, versions, op_id, index, 
                              user_mask, source, wait_on_events, 
                              applied_events, trace_info,
                              need_version_update);
      }
      // Add our local user
      const bool issue_collect = add_local_user(usage, term_event, 
                   INVALID_COLOR, user_expr, true/*base*/, versions, op_id, 
                   index, user_mask, source, applied_events, trace_info);
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
      // Return the merge of the events
      return Runtime::merge_events(&trace_info, wait_on_events);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_user_above_fused(const RegionUsage &usage, 
                                                ApEvent term_event,
                                                const LegionColor child_color,
                                                IndexSpaceExpression *user_expr,
                                                VersionTracker *versions,
                                                const UniqueID op_id,
                                                const unsigned index,
                                                const FieldMask &user_mask,
                                                const AddressSpaceID source,
                                              std::set<ApEvent> &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                                const bool need_version_update)
    //--------------------------------------------------------------------------
    {
      // Do the precondition analysis on the way up
      find_local_user_preconditions_above(usage, term_event, child_color, 
                          user_expr, versions, op_id, index, user_mask, 
                          preconditions, applied_events, trace_info);
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
        parent->add_user_above_fused(usage, term_event, local_color, user_expr,
                              versions, op_id, index, user_mask, source,
                              preconditions, applied_events, trace_info,
                              need_update_above);
      }
      // Add the user on the way back down
      add_local_user(usage, term_event, child_color, user_expr, false/*base*/,
         versions, op_id, index, user_mask, source, applied_events, trace_info);
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
                                    logical_node->get_index_space_expression());
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
    void MaterializedView::update_gc_events(
                                           const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      if (parent != NULL)
        parent->update_gc_events(term_events);
      AutoLock v_lock(view_lock);
      for (std::set<ApEvent>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
      {
        outstanding_gc_events.insert(*it);
      }
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
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        // Send a message to the logical owner to do the analysis
        RtUserEvent wait_on = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(to_filter);
          version_info.pack_writing_version_numbers(rez);
          rez.serialize(&to_filter);
          rez.serialize(wait_on);
        }
        runtime->send_view_filter_invalid_fields_request(logical_owner, rez);
        // Wait for the result to be ready
        wait_on.wait();
        return;
      }
#endif
      // If we're not the parent then keep going up
      if ((parent != NULL) && !version_info.is_upper_bound_node(logical_node))
        parent->filter_invalid_fields(to_filter, version_info);
      // If we still have fields to filter, then do that now
      if (!!to_filter)
      {
#ifdef DISTRIBUTED_INSTANCE_VIEWS
        // If we're not the owner then make sure that we are up to date
        if (!is_logical_owner())
          perform_remote_valid_check(to_filter, &version_info, true/*reading*/);
#endif
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
#endif // DISTRIBUTED_INSTANCE_VIEWS
      {
        // This is the common path
        if (!!filter_mask || !add_only.empty())
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
            const FieldMask overlap = finder->second & it->second;
            if (!overlap)
              continue;
            finder->second -= overlap;
            if (!finder->second)
              current_versions.erase(finder);
            current_versions[it->first+1] |= overlap;
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
      FieldMask filter_mask;
      LegionMap<VersionID,FieldMask>::aligned update_versions;
      bool need_check_above = false;
      // Need the lock in exclusive mode to do the update
      AutoLock v_lock(view_lock);
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      // If we are logical owner and we have remote valid instances
      // we need to track which version numbers get updated so we can
      // send invalidates
      const bool need_invalidates = is_logical_owner() && 
          !valid_remote_instances.empty() && !(user_mask * remote_valid_mask);
      FieldMask invalidate_mask;
#endif
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
            if (need_invalidates)
              invalidate_mask |= intersect;
#endif
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
        if (need_invalidates)
          invalidate_mask |= overlap;
#endif
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      if (!!invalidate_mask)
        send_invalidations(invalidate_mask, source, applied_events);
#endif
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
                                               IndexSpaceExpression *user_expr,
                                               ApEvent term_event,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events,
                           LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                               FieldMask &observed,
                                               FieldMask &non_dominated,
                                            const PhysicalTraceInfo &trace_info)
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
        if (!trace_info.recording && cit->first.has_triggered_faultignorant())
        {
          dead_events.insert(cit->first);
          continue;
        }
        if (!trace_info.recording &&
            preconditions.find(cit->first) != preconditions.end())
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
          ApEvent pre = cit->first;
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, user_expr, pre))
          {
            preconditions.insert(pre);
            if (TRACK_DOM && (pre == cit->first))
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
            ApEvent pre = cit->first;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, user_expr, pre))
            {
              preconditions.insert(pre);
              if (TRACK_DOM && (pre == cit->first))
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
                                               IndexSpaceExpression *user_expr,
                                               ApEvent term_event,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events,
                                            const PhysicalTraceInfo &trace_info)
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
        if (!trace_info.recording && pit->first.has_triggered_faultignorant())
        {
          dead_events.insert(pit->first);
          continue;
        }
        if (!trace_info.recording &&
            preconditions.find(pit->first) != preconditions.end())
          continue;
#endif
        const EventUsers &event_users = pit->second;
        if (user_mask * event_users.user_mask)
          continue;
        if (event_users.single)
        {
          ApEvent pre = pit->first;
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, user_expr, pre))
            preconditions.insert(pre);
        }
        else
        {
          for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                it = event_users.users.multi_users->begin(); it !=
                event_users.users.multi_users->end(); it++)
          {
            if (user_mask * it->second)
              continue;
            ApEvent pre = pit->first;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, user_expr, pre))
              preconditions.insert(pre);
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
                                               IndexSpaceExpression *user_expr,
                                               const UniqueID op_id,
                                               const unsigned index,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                               std::set<ApEvent> &dead_events,
                           LegionMap<ApEvent,FieldMask>::aligned &filter_events,
                                               FieldMask &observed,
                                               FieldMask &non_dominated,
                                            const PhysicalTraceInfo &trace_info)
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
        if (!trace_info.recording && cit->first.has_triggered_faultignorant())
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
        if (!trace_info.recording && finder != preconditions.end())
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
          ApEvent pre = cit->first;
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, user_expr, pre))
          {
            if (pre != cit->first)
              preconditions[pre] |= overlap;
            else if (finder == preconditions.end())
              preconditions[cit->first] = overlap;
            else
              finder->second |= overlap;
            if (TRACK_DOM && (pre == cit->first))
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
            ApEvent pre = cit->first;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, user_expr, pre))
            {
              if (pre != cit->first)
                preconditions[pre] |= user_overlap;
              else if (finder == preconditions.end())
                preconditions[cit->first] = user_overlap;
              else
                finder->second |= user_overlap;
              if (TRACK_DOM && (pre == cit->first))
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
                                               IndexSpaceExpression *user_expr,
                                               const UniqueID op_id,
                                               const unsigned index,
                         LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                               std::set<ApEvent> &dead_events,
                                            const PhysicalTraceInfo &trace_info)
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
        if (!trace_info.recording && pit->first.has_triggered_faultignorant())
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
        if (!trace_info.recording && finder != preconditions.end())
        {
          overlap -= finder->second;
          if (!overlap)
            continue;
        }
#endif
        if (event_users.single)
        {
          ApEvent pre = pit->first;
          if (has_local_precondition(event_users.users.single_user, usage,
                                     child_color, op_id, index, user_expr, pre))
          {
            if (pre != pit->first)
              preconditions[pre] |= overlap;
            else if (finder == preconditions.end())
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
            ApEvent pre = pit->first;
            if (has_local_precondition(it->first, usage, child_color, 
                                       op_id, index, user_expr, pre))
            {
              if (pre != pit->first)
                preconditions[pre] |= user_overlap;
              else if (finder == preconditions.end())
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

#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
            users.users.single_user->pack_user(rez, source);
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
              to_send[idx]->pack_user(rez, source);
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
            users.users.single_user->pack_user(rez, source);
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
              to_send[idx]->pack_user(rez, source);
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
                                                   AddressSpaceID source,
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
                PhysicalUser::unpack_user(derez, true/*add ref*/,forest,source);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/,forest,source);
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
                  PhysicalUser::unpack_user(derez, true/*add ref*/, 
                                            forest, source);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/,forest,source);
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
                PhysicalUser::unpack_user(derez, true/*add ref*/,forest,source);
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
                  PhysicalUser::unpack_user(derez, true/*add ref*/, 
                                            forest, source);
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
      IndexSpaceExpression *user_expr = 
        IndexSpaceExpression::unpack_expression(derez, forest, source);
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
      PhysicalTraceInfo trace_info(NULL/*op*/);
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
              true/*single copy*/, restrict_out, user_mask, child_color, 
              user_expr, &dummy_version_info, op_id, index, source,
              dummy_preconditions, applied_conditions, trace_info);
        else
          find_local_copy_preconditions_above(usage.redop, IS_READ_ONLY(usage),
              true/*single copy*/, restrict_out, user_mask, child_color, 
              user_expr, &dummy_version_info, op_id, index, source,
              dummy_preconditions, applied_conditions, trace_info);
        add_local_copy_user(usage, term_event, base_user, restrict_out, 
                            child_color, user_expr, &dummy_version_info, 
                            op_id, index, user_mask, source, 
                            applied_conditions, trace_info);
      }
      else
      {
        // Do analysis and register the user
        std::set<ApEvent> dummy_preconditions;
        // We do different things depending on whether we are the base
        // user or whether we are being registered above in the tree
        if (base_user)
          find_local_user_preconditions(usage, term_event, child_color,
                                        user_expr, &dummy_version_info, op_id,
                                        index,user_mask, dummy_preconditions, 
                                        applied_conditions, trace_info);
        else
          find_local_user_preconditions_above(usage, term_event, child_color,
                                        user_expr, &dummy_version_info, op_id,
                                        index,user_mask, dummy_preconditions, 
                                        applied_conditions, trace_info);
        if (IS_WRITE(usage))
          update_version_numbers(user_mask, field_versions,
                                 source, applied_conditions);
        if (add_local_user(usage, term_event, child_color, user_expr, 
                           base_user, &dummy_version_info, op_id, index, 
                           user_mask, source, applied_conditions, trace_info))
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
#else // DISTRIBUTED_INSTANCE_VIEWS
    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_filter_invalid_fields_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask to_filter;
      derez.deserialize(to_filter);
      VersionInfo version_info;
      version_info.unpack_version_numbers(derez, runtime->forest);
      FieldMask *remote_mask;
      derez.deserialize(remote_mask);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);

      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (!ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_materialized_view());
#endif
      MaterializedView *mat_view = view->as_materialized_view(); 
      mat_view->filter_invalid_fields(to_filter, version_info);

      // Now send the response back
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_mask);
        rez.serialize(to_filter);
        rez.serialize(to_trigger);
      }
      runtime->send_view_filter_invalid_fields_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_filter_invalid_fields_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      FieldMask *target;
      derez.deserialize(target);
      derez.deserialize(*target);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }
#endif // DISTRIBUTED_INSTANCE_VIEWS

    /////////////////////////////////////////////////////////////
    // ShardedWriteTracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardedWriteTracker::ShardedWriteTracker(unsigned fidx, RegionTreeForest *f,
                                             IndexSpaceExpression *bound,
                                             ShardedWriteTracker *remote,
                                             RtUserEvent event,
                                             AddressSpaceID target)
      : field_index(fidx), forest(f), upper_bound(bound), pending_expr((remote 
            == NULL) ? new PendingIndexSpaceExpression(upper_bound,f) : NULL), 
        remote_tracker(remote), remote_event(event), remote_target(target)
    //--------------------------------------------------------------------------
    {
      if (pending_expr != NULL)
        pending_expr->add_expression_reference(); 
    }

    //--------------------------------------------------------------------------
    ShardedWriteTracker::ShardedWriteTracker(const ShardedWriteTracker &rhs)
      : field_index(0), forest(NULL), upper_bound(NULL), pending_expr(NULL),
        remote_tracker(NULL), remote_event(RtUserEvent::NO_RT_USER_EVENT),
        remote_target(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardedWriteTracker::~ShardedWriteTracker(void) 
    //--------------------------------------------------------------------------
    {
      if ((pending_expr != NULL) && pending_expr->remove_expression_reference())
        delete pending_expr;
    }

    //--------------------------------------------------------------------------
    ShardedWriteTracker& ShardedWriteTracker::operator=(
                                                 const ShardedWriteTracker &rhs) 
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ShardedWriteTracker::pack_for_remote_shard(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RtUserEvent result = Runtime::create_rt_user_event();
      rez.serialize(this);
      rez.serialize(result);
      // Can do this without the lock since we know adding this is sequential
      remote_events.insert(result);
    }

    //--------------------------------------------------------------------------
    void ShardedWriteTracker::record_valid_expression(IndexSpaceExpression *exp)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
      valid_expressions.insert(exp);
    }

    //--------------------------------------------------------------------------
    void ShardedWriteTracker::record_sub_expression(IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      AutoLock e_lock(expr_lock);
      sub_expressions.insert(expr);
    }

    //--------------------------------------------------------------------------
    bool ShardedWriteTracker::arm(void)
    //--------------------------------------------------------------------------
    {
      // Add a reference to ourselves first
      add_reference();
      if (!remote_events.empty())
      {
        RtEvent wait_for = Runtime::merge_events(remote_events);
        if (wait_for.exists() && !wait_for.has_triggered())
        {
          ShardedWriteTrackerArgs args;
          args.tracker = this;
          forest->runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, wait_for);
          return false;
        }
      }
      // If we make it here, then we can do the computation
      evaluate();
      return remove_reference();
    }

    //--------------------------------------------------------------------------
    void ShardedWriteTracker::evaluate(void)
    //--------------------------------------------------------------------------
    {
      // Don't need the lock here since we know everything is ready
      // and won't be changing
      if (remote_tracker != NULL)
      {
#ifdef DEBUG_LEGION
        assert(pending_expr == NULL);
#endif
        // Remote case
        if (!valid_expressions.empty())
        {
          IndexSpaceExpression *valid = 
            forest->union_index_spaces(valid_expressions);
          if (!sub_expressions.empty())
          {
            IndexSpaceExpression *sub = 
              forest->union_index_spaces(sub_expressions);
            valid = forest->subtract_index_spaces(valid, sub);
          }
          if (!valid->is_empty())
            send_shard_valid(forest->runtime, remote_tracker,
                             remote_target, valid, remote_event);
          else
            Runtime::trigger_event(remote_event);
        }
        else // We can just trigger our event since we're done
          Runtime::trigger_event(remote_event);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(pending_expr != NULL);
#endif
        // Owner case
        if (!valid_expressions.empty())
        {
          IndexSpaceExpression *valid = 
            forest->union_index_spaces(valid_expressions);
          if (!sub_expressions.empty())
          {
            IndexSpaceExpression *sub = 
              forest->union_index_spaces(sub_expressions);
            valid = forest->subtract_index_spaces(valid, sub);
          }
          pending_expr->set_result(
              forest->subtract_index_spaces(upper_bound, valid));
        }
        else
          pending_expr->set_result(upper_bound); // same as the upper bound
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedWriteTracker::handle_evaluate(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardedWriteTrackerArgs *sargs = 
        (const ShardedWriteTrackerArgs*)args;
      sargs->tracker->evaluate();
      if (sargs->tracker->remove_reference())
        delete sargs->tracker;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedWriteTracker::unpack_tracker(Deserializer &derez,
                              ShardedWriteTracker *&tracker, RtUserEvent &event)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(tracker);
      derez.deserialize(event);
    }

    //--------------------------------------------------------------------------
    /*static*/ ShardedWriteTracker* ShardedWriteTracker::unpack_tracker(
         unsigned fidx, AddressSpaceID source, Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      ShardedWriteTracker *remote;
      derez.deserialize(remote);
      RtUserEvent remote_done;
      derez.deserialize(remote_done);
      return new ShardedWriteTracker(fidx, rt->forest, NULL,
                                     remote, remote_done, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedWriteTracker::send_shard_valid(Runtime *runtime,
                                                  ShardedWriteTracker *tracker, 
                                                  AddressSpaceID target, 
                                                  IndexSpaceExpression *expr,
                                                  RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      if (target == runtime->address_space)
      {
        tracker->record_valid_expression(expr);
        Runtime::trigger_event(done_event);
      }
      else
      {
        Serializer rez;
        rez.serialize(tracker);
        expr->pack_expression(rez, target);
        rez.serialize(done_event);
        rez.serialize<bool>(true); // valid
        runtime->send_control_replicate_composite_view_write_summary(target, 
                                                                     rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedWriteTracker::send_shard_sub(Runtime *runtime,
                                                  ShardedWriteTracker *tracker, 
                                                  AddressSpaceID target, 
                                                  IndexSpaceExpression *expr,
                                                  RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      if (target == runtime->address_space)
      {
        tracker->record_sub_expression(expr);
        Runtime::trigger_event(done_event);
      }
      else
      {
        Serializer rez;
        rez.serialize(tracker);
        expr->pack_expression(rez, target);
        rez.serialize(done_event);
        rez.serialize<bool>(false); // sub
        runtime->send_control_replicate_composite_view_write_summary(target,
                                                                     rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardedWriteTracker::process_shard_summary(
           Deserializer &derez, RegionTreeForest *forest, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ShardedWriteTracker *tracker;
      derez.deserialize(tracker);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, forest, source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool valid;
      derez.deserialize<bool>(valid);
      if (valid)
        tracker->record_valid_expression(expr);
      else
        tracker->record_sub_expression(expr);
      Runtime::trigger_event(done_event);
    }

    /////////////////////////////////////////////////////////////
    // DeferredCopier
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredCopier::DeferredCopier(const TraversalInfo *in, InnerContext *ctx,
       MaterializedView *d, const FieldMask &m, const RestrictInfo &res, bool r)
      : info(in), shard_context(ctx), dst(d), across_helper(NULL), 
        restrict_info(&res), restrict_out(r), deferred_copy_mask(m), 
        current_reduction_epoch(0)
#ifdef DEBUG_LEGION
        , finalized(false)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredCopier::DeferredCopier(const TraversalInfo *in, InnerContext *ctx,
       MaterializedView *d, const FieldMask &m, ApEvent p, CopyAcrossHelper *h)
      : info(in), shard_context(ctx), dst(d), across_helper(h), 
        restrict_info(NULL), restrict_out(false), deferred_copy_mask(m), 
        current_reduction_epoch(0)
#ifdef DEBUG_LEGION
        , finalized(false)
#endif
    //--------------------------------------------------------------------------
    {
      dst_preconditions[p] = m;
    }

    //--------------------------------------------------------------------------
    DeferredCopier::DeferredCopier(const DeferredCopier &rhs)
      : info(rhs.info), shard_context(NULL), dst(rhs.dst), 
        across_helper(rhs.across_helper), 
        restrict_info(rhs.restrict_info), restrict_out(rhs.restrict_out)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeferredCopier::~DeferredCopier(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(finalized);
#endif
    }

    //--------------------------------------------------------------------------
    DeferredCopier& DeferredCopier::operator=(const DeferredCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::merge_destination_preconditions(
     const FieldMask &mask,LegionMap<ApEvent,FieldMask>::aligned &preconditions)
    //--------------------------------------------------------------------------
    {
      const FieldMask needed_mask = mask - dst_precondition_mask;
      if (!!needed_mask)
        compute_dst_preconditions(needed_mask);
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            dst_preconditions.begin(); it != dst_preconditions.end(); it++)
      {
        const FieldMask overlap = it->second & mask;
        if (!overlap)
          continue;
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          preconditions.find(it->first);
        if (finder == preconditions.end())
          preconditions[it->first] = overlap;
        else
          finder->second |= overlap;
      }
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::buffer_reductions(VersionTracker *version_tracker,
                               PredEvent pred_guard, RegionTreeNode *intersect,
                               const WriteMasks &write_masks,
              LegionMap<ReductionView*,FieldMask>::aligned &source_reductions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_epochs.size() == reduction_epoch_masks.size());
#endif
      if (current_reduction_epoch >= reduction_epochs.size())
      {
        reduction_epochs.resize(current_reduction_epoch + 1);
        reduction_epoch_masks.resize(current_reduction_epoch + 1);
#ifndef DISABLE_CVOPT
        reduction_shards.resize(current_reduction_epoch + 1);
#endif
      }
      PendingReductions &pending_reductions = 
        reduction_epochs[current_reduction_epoch];
      bool has_prev_write_mask = (source_reductions.size() == 1);
      FieldMask prev_write_mask;
      FieldMask &epoch_mask = reduction_epoch_masks[current_reduction_epoch];
      for (LegionMap<ReductionView*,FieldMask>::aligned::iterator rit = 
            source_reductions.begin(); rit != source_reductions.end(); rit++)
      {
        LegionList<PendingReduction>::aligned &pending_reduction_list = 
          pending_reductions[rit->first];
        // First look for any write masks that we need to filter by
        if (!has_prev_write_mask || !(prev_write_mask * rit->second))
        {
          for (WriteMasks::const_iterator it = write_masks.begin(); 
                it != write_masks.end(); it++)
          {
            // If this is the first time through we're computing the summary
            if (!has_prev_write_mask)
              prev_write_mask |= it->second;
            const FieldMask overlap = rit->second & it->second;
            if (!overlap)
              continue;
            pending_reduction_list.push_back(PendingReduction(overlap, 
                  version_tracker, pred_guard, intersect, it->first));
            epoch_mask |= overlap;
            rit->second -= overlap;
            // Can break out if we're done and we already have a summary mask
            if (!rit->second && has_prev_write_mask)
              break;
          }
          if (!rit->second)
            continue;
        }
        pending_reduction_list.push_back(PendingReduction(rit->second, 
              version_tracker, pred_guard, intersect, NULL/*mask*/));
        epoch_mask |= rit->second;
      }
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void DeferredCopier::buffer_reduction_shards(PredEvent pred_guard,
        ReplicationID repl_id, RtEvent shard_invalid_barrier,
        const LegionMap<ShardID,WriteMasks>::aligned &shard_reductions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_shards.size() == reduction_epoch_masks.size());
#endif
      if (current_reduction_epoch >= reduction_shards.size())
      {
        reduction_epochs.resize(current_reduction_epoch + 1);
        reduction_epoch_masks.resize(current_reduction_epoch + 1);
        reduction_shards.resize(current_reduction_epoch + 1);
      }
      PendingReductionShards &pending_shards = 
        reduction_shards[current_reduction_epoch];
      FieldMask &epoch_mask = reduction_epoch_masks[current_reduction_epoch];
      for (LegionMap<ShardID,WriteMasks>::aligned::const_iterator sit = 
            shard_reductions.begin(); sit != shard_reductions.end(); sit++)
      {
        const ShardInfo key(sit->first, repl_id, shard_invalid_barrier);
        LegionDeque<ReductionShard>::aligned &reductions = pending_shards[key];
        for (WriteMasks::const_iterator it = sit->second.begin();
              it != sit->second.end(); it++)
        {
          reductions.push_back(
              ReductionShard(it->second, pred_guard, it->first));
          epoch_mask |= it->second;
        }
      }
    }
#endif

    //--------------------------------------------------------------------------
    void DeferredCopier::begin_guard_protection(void)
    //--------------------------------------------------------------------------
    {
      protected_copy_posts.resize(protected_copy_posts.size() + 1);
      copy_postconditions.swap(protected_copy_posts.back());
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::end_guard_protection(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!protected_copy_posts.empty());
#endif
      LegionMap<ApEvent,FieldMask>::aligned &target = 
        protected_copy_posts.back();
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            copy_postconditions.begin(); it != copy_postconditions.end(); it++)
      {
        ApEvent protect = Runtime::ignorefaults(it->first);
        if (!protect.exists())
          continue;
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          target.find(protect); 
        if (finder == target.end())
          target.insert(*it);
        else
          finder->second |= it->second;
      }
      copy_postconditions.clear();
      copy_postconditions.swap(target);
      protected_copy_posts.pop_back();
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::begin_reduction_epoch(void)
    //--------------------------------------------------------------------------
    {
      // Increase the depth
      current_reduction_epoch++; 
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::end_reduction_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_reduction_epoch > 0);
#endif
      current_reduction_epoch--; 
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::record_previously_valid(IndexSpaceExpression *expr,
                                                 const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // No need to record this if we're doing an across copy
      if (across_helper != NULL)
        return;
      WriteSet::iterator finder = dst_previously_valid.find(expr);
      if (finder == dst_previously_valid.end())
        dst_previously_valid.insert(expr, mask);
      else
        finder.merge(mask);
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::finalize(DeferredView *src_view,
                                  std::set<ApEvent> *postconditions/*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_reduction_epoch == 0);
      assert(!finalized);
      finalized = true;
#endif
      WriteSet reduce_exprs;
      // Apply any pending reductions using the proper preconditions
      if (!reduction_epoch_masks.empty())
        apply_reduction_epochs(reduce_exprs);
      // If we have no copy post conditions at this point we're done
      // Note that if we have remote shards we sent messages too then 
      // this is guaranteed not to be empty
      if (copy_postconditions.empty())
        return;
      // Either apply or record our postconditions
      if (postconditions == NULL)
      {
        // Need to uniquify our copy_postconditions before doing this
        uniquify_copy_postconditions();
        const AddressSpaceID local_space = dst->context->runtime->address_space;
        // Handle any restriction cases
        FieldMask restrict_mask;
        if (restrict_out && (restrict_info != NULL) && 
            restrict_info->has_restrictions())
          restrict_info->populate_restrict_fields(restrict_mask);
        // Compute the performed write expression for each of these
        // fields. 
        IndexSpaceExpression *dst_expr = 
          dst->logical_node->get_index_space_expression();
        // If we have any write trackers, the first thing we want to
        // do is update them to get them in flight and to filter out
        // any fields we don't need to worry about for the rest of
        // this computation
        if (!write_trackers.empty())
          arm_write_trackers(reduce_exprs, true/*add reference*/);
        // Next we can compute the performed write expressions for
        // any fields for which we still have previsously valid data
        // This is the destination expression minus any any expressions 
        // for where the instance was already valid
        std::vector<IndexSpaceExpression*> actual_dst_exprs;
        LegionVector<FieldMask>::aligned previously_valid_masks;
        if (!dst_previously_valid.empty())
          compute_actual_dst_exprs(dst_expr, reduce_exprs,
                                   actual_dst_exprs, previously_valid_masks);
        // Apply the destination users
        for (LegionMap<ApEvent,FieldMask>::aligned::iterator it = 
             copy_postconditions.begin(); it != copy_postconditions.end(); it++)
        {
          // handle any restricted postconditions
          if (restrict_out && !(it->second * restrict_mask))
            info->op->record_restrict_postcondition(it->first);
          // If we have any write trackers for the fields for this event
          // then we use them as the write expressions for the copy user
          if (!write_trackers.empty())
          {
            // Iterate over the fields in the field mask
            int index = it->second.find_first_set();
            while (index >= 0)
            {
              std::map<unsigned,ShardedWriteTracker*>::const_iterator finder =
                write_trackers.find(index);
              if (finder != write_trackers.end())
              {
                FieldMask write_mask;
                write_mask.set_bit(index);
                dst->add_copy_user(0/*redop*/, it->first, &info->version_info,
                         finder->second->pending_expr->get_ready_expression(),
                         info->op->get_unique_op_id(), info->index,
                         write_mask, false/*reading*/, restrict_out,
                         local_space, info->map_applied_events,*info);
                // Unset the bit since we handled it
                it->second.unset_bit(index);
              }
              // Do the next field
              index = it->second.find_next_set(index+1);
            }
          }
          // Compute the performed write expression for each of these
          // fields. This is the destination expression minus any 
          // any expressions for where the instance was already valid
          if (!previously_valid_masks.empty())
          {
            for (unsigned idx = 0; idx < previously_valid_masks.size(); idx++)
            {
              const FieldMask overlap = 
                previously_valid_masks[idx] & it->second;
              if (!overlap)
                continue;
              dst->add_copy_user(0/*redop*/, it->first, &info->version_info,
                                 actual_dst_exprs[idx], 
                                 info->op->get_unique_op_id(), info->index,
                                 overlap, false/*reading*/, restrict_out,
                                 local_space, info->map_applied_events, *info);
              // Tell the recorder about any empty copies
              if (info->recording && actual_dst_exprs[idx]->is_empty())
                info->record_empty_copy(src_view, overlap, dst);
              it->second -= overlap;
              if (!it->second)
                break;
            }
          }
          // Easy case, we can just use the normal destination expression
          // for any remaining fields which were not previously valid
          if (!!it->second)
            dst->add_copy_user(0/*redop*/, it->first, &info->version_info, 
                           dst_expr, info->op->get_unique_op_id(), info->index,
                           it->second, false/*reading*/, restrict_out,
                           local_space, info->map_applied_events, *info);
        }
        // Once we're done with all the registrations then we can remove
        // the references that we added to the write trackers
        if (!write_trackers.empty())
        {
          for (std::map<unsigned,ShardedWriteTracker*>::const_iterator it =
                write_trackers.begin(); it != write_trackers.end(); it++)
            if (it->second->remove_reference())
              delete it->second;
          write_trackers.clear();
        }
      }
      else
      {
        // Arm our write trackers so they are ready to go
        if (!write_trackers.empty())
          arm_write_trackers(reduce_exprs, false/*add reference*/);
        // Just merge in the copy postcondition events
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              copy_postconditions.begin(); it != 
              copy_postconditions.end(); it++)
          postconditions->insert(it->first);
      }
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void DeferredCopier::pack_copier(Serializer &rez,const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
      rez.serialize(dst->did);
      info->pack(rez);
      if (across_helper != NULL)
      {
        rez.serialize<bool>(true); // across
        across_helper->pack(rez, copy_mask);
      }
      else
        rez.serialize<bool>(false); // across
      // Compute preconditions for any fields we might need
      const FieldMask needed_fields = copy_mask - dst_precondition_mask;
      if (!!needed_fields)
        compute_dst_preconditions(needed_fields);
      LegionMap<ApEvent,FieldMask>::aligned needed_preconditions;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            dst_preconditions.begin(); it != dst_preconditions.end(); it++)
      {
        const FieldMask overlap = it->second & copy_mask;
        if (!overlap)
          continue;
        needed_preconditions[it->first] = overlap;
      }
      rez.serialize<size_t>(needed_preconditions.size());
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
           needed_preconditions.begin(); it != needed_preconditions.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void DeferredCopier::pack_sharded_write_tracker(unsigned field_index,
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,ShardedWriteTracker*>::const_iterator finder = 
        write_trackers.find(field_index);
      if (finder == write_trackers.end())
      {
        ShardedWriteTracker *result = new ShardedWriteTracker(field_index,
          dst->context, dst->logical_node->get_index_space_expression());
        result->pack_for_remote_shard(rez);
        write_trackers[field_index] = result;
      }
      else
        finder->second->pack_for_remote_shard(rez);
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::uniquify_copy_postconditions(void)
    //--------------------------------------------------------------------------
    {
      LegionList<FieldSet<ApEvent> >::aligned copy_postcondition_sets;
      compute_field_sets<ApEvent>(FieldMask(), copy_postconditions,
                                  copy_postcondition_sets);
      copy_postconditions.clear();
      for (LegionList<FieldSet<ApEvent> >::aligned::const_iterator it = 
            copy_postcondition_sets.begin(); it != 
            copy_postcondition_sets.end(); it++)
      {
        ApEvent result = Runtime::merge_events(info, it->elements);
        LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
          copy_postconditions.find(result);
        if (finder == copy_postconditions.end())
          copy_postconditions[result] = it->set_mask;
        else
          finder->second |= it->set_mask;
      }
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::compute_dst_preconditions(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mask * dst_precondition_mask);
#endif
      const AddressSpaceID local_space = dst->context->runtime->address_space;
      // Note here that we set 'can_filter' to false because this is an
      // over-approximation for the copy that we're eventually going to
      // be issuing, so we can't be sure we're writing all of it
      // The only exception here is in the case where we're doing a copy
      // across in which case we know we're going to write everything
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                             false/*single copy*/, restrict_out, mask, 
                             dst->logical_node->get_index_space_expression(), 
                             &info->version_info, info->op->get_unique_op_id(),
                             info->index, local_space, dst_preconditions,
                             info->map_applied_events, *info,
                             (across_helper != NULL)/*can filter*/);
      if ((restrict_info != NULL) && restrict_info->has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info->populate_restrict_fields(restrict_mask);
        restrict_mask &= mask;
        if (!!restrict_mask)
        {
          ApEvent restrict_pre = info->op->get_restrict_precondition(*info);
          LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            dst_preconditions.find(restrict_pre);
          if (finder == dst_preconditions.end())
            dst_preconditions[restrict_pre] = restrict_mask;
          else
            finder->second |= restrict_mask;
        }
      }
      dst_precondition_mask |= mask;
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::apply_reduction_epochs(WriteSet &reduce_exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_epochs.size() == reduction_epoch_masks.size());
#ifndef DISABLE_CVOPT
      assert(reduction_shards.size() == reduction_epoch_masks.size());
#endif
#endif
      FieldMask summary_mask = reduction_epoch_masks.front();
      for (unsigned idx = 1; idx < reduction_epoch_masks.size(); idx++)
        summary_mask |= reduction_epoch_masks[idx];
      // Check to see if we need any more destination preconditions
      const FieldMask needed_mask = summary_mask - dst_precondition_mask;
      if (!!needed_mask)
      {
        // We're not going to use the existing destination preconditions
        // for anything else, so go ahead and clear it
        dst_preconditions.clear();
        compute_dst_preconditions(needed_mask);
        if (!dst_preconditions.empty())
        {
          // Then merge the destination preconditions for these fields
          // into the copy_postconditions, they'll be collapsed in the
          // next stage of the computation
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
                dst_preconditions.begin(); it != dst_preconditions.end(); it++)
          {
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
              copy_postconditions.find(it->first);
            if (finder == copy_postconditions.end())
              copy_postconditions.insert(*it);
            else
              finder->second |= it->second;
          }
        }
      }
      // Uniquify our copy postconditions since they will be our 
      // preconditions to the reductions
      uniquify_copy_postconditions();
      // Iterate over the reduction epochs from back to front since
      // the ones in the back are the ones that should be applied first
      for (int epoch = reduction_epoch_masks.size()-1; epoch >= 0; epoch--)
      {
        FieldMask &epoch_mask = reduction_epoch_masks[epoch];
        // Some level might not have any reductions
        if (!epoch_mask)
          continue;
        LegionMap<ApEvent,FieldMask>::aligned reduction_postconditions;
#ifndef DISABLE_CVOPT
        // Send anything to remote shards first to get things in flight
        PendingReductionShards &pending_shards = reduction_shards[epoch];
        for (PendingReductionShards::const_iterator pit = 
              pending_shards.begin(); pit != pending_shards.end(); pit++)
        {
#ifdef DEBUG_LEGION
          assert(shard_context != NULL);
#endif
          const LegionDeque<ReductionShard>::aligned &reductions = pit->second;
          Serializer rez;
          rez.serialize(pit->first.repl_id);
          rez.serialize(pit->first.shard);
          rez.serialize<RtEvent>(pit->first.shard_invalid_barrier);
          rez.serialize(dst->did);
          info->pack(rez);
          if (across_helper != NULL)
          {
            rez.serialize<bool>(true);
            rez.serialize(epoch_mask);
            across_helper->pack(rez, epoch_mask);
          }
          else
            rez.serialize<bool>(false);
          rez.serialize<size_t>(reductions.size());
          for (LegionDeque<ReductionShard>::aligned::const_iterator it =
                reductions.begin(); it != reductions.end(); it++)
          {
            rez.serialize(it->reduction_mask);
            rez.serialize(it->pred_guard);
            it->mask->pack_expression(rez,
                shard_context->find_shard_space(pit->first.shard));
            // We need one completion event for each field  
            const size_t pop_count = it->reduction_mask.pop_count();
#ifdef DEBUG_LEGION
            assert(pop_count > 0);
#endif
            int index = it->reduction_mask.find_first_set(); 
            for (unsigned idx = 0; idx < pop_count; idx++)
            {
              if (idx > 0)
                index = it->reduction_mask.find_next_set(index+1);
#ifdef DEBUG_LEGION
              assert(index >= 0);
#endif
              ApUserEvent field_done = Runtime::create_ap_user_event();
              if (pop_count > 1)
                rez.serialize<unsigned>(index);
              rez.serialize(field_done);
#ifdef DEBUG_LEGION
              assert(reduction_postconditions.find(field_done) ==
                      reduction_postconditions.end());
#endif
              reduction_postconditions[field_done].set_bit(index);
              if (across_helper != NULL)
                pack_sharded_write_tracker(index, rez);
            }
            // We also need destination preconditions for the reductions
            if (pop_count > 1)
            {
              LegionMap<ApEvent,FieldMask>::aligned reduce_preconditions;
              for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pit =
                    copy_postconditions.begin(); pit != 
                    copy_postconditions.end(); pit++)
              {
                const FieldMask overlap = pit->second & it->reduction_mask;
                if (!overlap)
                  continue;
                reduce_preconditions[pit->first] = overlap;
              }
              rez.serialize<size_t>(reduce_preconditions.size());
              for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator rit =
                    reduce_preconditions.begin(); rit != 
                    reduce_preconditions.end(); rit++)
              {
                rez.serialize(rit->first);
                rez.serialize(rit->second);
              }
            }
            else
            {
              // Special case where we only have a single field
              std::set<ApEvent> reduce_preconditions;
              for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator pit =
                    copy_postconditions.begin(); pit != 
                    copy_postconditions.end(); pit++)
              {
                if (pit->second * it->reduction_mask)
                  continue;
                reduce_preconditions.insert(pit->first);
              }
              if (!reduce_preconditions.empty())
              {
                ApEvent pre = Runtime::merge_events(info, reduce_preconditions);
                rez.serialize(pre);
              }
              else
                rez.serialize(ApEvent::NO_AP_EVENT);
            }
          }
          shard_context->send_composite_view_shard_reduction_request(
                                                pit->first.shard, rez);
        }
#endif
        // Iterate over all the postconditions and issue reductions
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              copy_postconditions.begin(); it != 
              copy_postconditions.end(); it++)
        {
          // See if we interfere with any reduction fields
          const FieldMask overlap = it->second & epoch_mask;
          if (!overlap)
            continue;
          if (issue_reductions(epoch, it->first, overlap, 
                               reduce_exprs, reduction_postconditions))
            break;
          // We've now done all these fields for a reduction
          epoch_mask -= it->second;
          // Can break out if we have no more reduction fields
          if (!epoch_mask)
            break;
        }
        // Issue any copies for which we have no precondition
        if (!!epoch_mask)
          issue_reductions(epoch, ApEvent::NO_AP_EVENT, epoch_mask,
                           reduce_exprs, reduction_postconditions);
        // Fold the reduction post conditions into the copy postconditions
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
              reduction_postconditions.begin(); it != 
              reduction_postconditions.end(); it++)
        {
          LegionMap<ApEvent,FieldMask>::aligned::iterator finder = 
            copy_postconditions.find(it->first);
          if (finder == copy_postconditions.end())
            copy_postconditions.insert(*it);
          else
            finder->second |= it->second;
        }
        // Only uniquify if we have to do more epochs
        if (epoch > 0)
          uniquify_copy_postconditions();
      }
    }

    //--------------------------------------------------------------------------
    bool DeferredCopier::issue_reductions(const int epoch,ApEvent reduction_pre,
                        const FieldMask &reduction_mask, WriteSet &reduce_exprs,
                LegionMap<ApEvent,FieldMask>::aligned &reduction_postconditions)
    //--------------------------------------------------------------------------
    {
      std::vector<ReductionView*> to_remove;
      PendingReductions &pending_reductions = reduction_epochs[epoch];
      // Iterate over reduction views
      for (PendingReductions::iterator pit = pending_reductions.begin();
            pit != pending_reductions.end(); pit++)
      {
        // Iterate over the pending reductions for this reduction view
        for (LegionList<PendingReduction>::aligned::iterator it = 
              pit->second.begin(); it != pit->second.end(); /*nothing*/)
        {
          const FieldMask &overlap = it->reduction_mask & reduction_mask;
          if (!overlap)
          {
            it++;
            continue;
          }
          IndexSpaceExpression *reduce_expr = NULL;
          // Issue the deferred reduction
          ApEvent reduction_post = pit->first->perform_deferred_reduction(
              dst, overlap, it->version_tracker, reduction_pre, info->op, 
              info->index, it->pred_guard, across_helper, it->intersect, 
              it->mask, info->map_applied_events, *info, reduce_expr);
          if (reduction_post.exists())
          {
            LegionMap<ApEvent,FieldMask>::aligned::iterator finder =
              reduction_postconditions.find(reduction_post);
            if (finder == reduction_postconditions.end())
              reduction_postconditions[reduction_post] = overlap;
            else
              finder->second |= overlap;
          }
          // Record any reduce expressions we may need to remove
          // from our previously valid set
          if (!dst_previously_valid.empty())
            reduce_exprs.insert(reduce_expr, overlap);
          // Remove these fields from the pending reduction record
          it->reduction_mask -= overlap;
          if (!it->reduction_mask)
            it = pit->second.erase(it);
          else
            it++;
        }
        // If we've don't have any more pending reductions this view is done
        if (pit->second.empty())
          to_remove.push_back(pit->first);
      }
      if (!to_remove.empty())
      {
        for (std::vector<ReductionView*>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          pending_reductions.erase(*it);
        // Can break out if we don't have any more pending reductions
        if (pending_reductions.empty())
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::arm_write_trackers(WriteSet &reduce_exprs, 
                                            bool add_reference)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!write_trackers.empty());
#endif
      FieldMask tracker_mask;
      // Build a mask for the trackers
      if (!dst_previously_valid.empty() || !reduce_exprs.empty())
      {
        for (std::map<unsigned,ShardedWriteTracker*>::const_iterator it = 
              write_trackers.begin(); it != write_trackers.end(); it++)
          tracker_mask.set_bit(it->first);
      }
      if (!dst_previously_valid.empty())
      {
        std::vector<IndexSpaceExpression*> to_delete;
        for (WriteSet::iterator it = dst_previously_valid.begin();
              it != dst_previously_valid.end(); it++)
        {
          const FieldMask overlap = it->second & tracker_mask;
          if (!overlap)
            continue;
          int index = overlap.find_first_set();
          while (index >= 0)
          {
#ifdef DEBUG_LEGION
            assert(write_trackers.find(index) != write_trackers.end());
#endif
            write_trackers[index]->record_valid_expression(it->first); 
          }
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
          for (std::vector<IndexSpaceExpression*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            dst_previously_valid.erase(*it);
      }
      if (!reduce_exprs.empty())
      {
        std::vector<IndexSpaceExpression*> to_delete;
        for (WriteSet::iterator it = reduce_exprs.begin();
              it != reduce_exprs.end(); it++)
        {
          const FieldMask overlap = it->second & tracker_mask;
          if (!overlap)
            continue;
          int index = overlap.find_first_set();
          while (index >= 0)
          {
#ifdef DEBUG_LEGION
            assert(write_trackers.find(index) != write_trackers.end());
#endif
            write_trackers[index]->record_sub_expression(it->first); 
          }
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
          for (std::vector<IndexSpaceExpression*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            reduce_exprs.erase(*it);
      }
      // Now we can arm all the write trackers
      for (std::map<unsigned,ShardedWriteTracker*>::const_iterator it = 
            write_trackers.begin(); it != write_trackers.end(); it++)
      {
        if (add_reference)
          it->second->add_reference();
        if (it->second->arm())
          delete it->second;
      }
      if (!add_reference)
        write_trackers.clear();
    }

    //--------------------------------------------------------------------------
    void DeferredCopier::compute_actual_dst_exprs(
                       IndexSpaceExpression *dst_expr, WriteSet &reduce_exprs,
                       std::vector<IndexSpaceExpression*> &actual_dst_exprs,
                       LegionVector<FieldMask>::aligned &previously_valid_masks)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = dst->context;
      LegionList<FieldSet<IndexSpaceExpression*> >::aligned 
        valid_sets, reduction_sets;
      dst_previously_valid.compute_field_sets(FieldMask(), valid_sets);
      if (!reduce_exprs.empty())
        reduce_exprs.compute_field_sets(FieldMask(), reduction_sets);
      for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::iterator 
            it = valid_sets.begin(); it != valid_sets.end(); it++)
      {
        IndexSpaceExpression *union_expr = 
            forest->union_index_spaces(it->elements);
        // See if we have any reduction sets that need to be subtracted
        if (!reduction_sets.empty())
        {
          for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
                const_iterator rit = reduction_sets.begin(); 
                rit != reduction_sets.end(); rit++)
          {
            const FieldMask overlap = it->set_mask & rit->set_mask;
            if (!overlap)
              continue;
            previously_valid_masks.push_back(overlap);
            IndexSpaceExpression *reduce_expr = 
              forest->union_index_spaces(rit->elements);
            IndexSpaceExpression *diff_expr = 
              forest->subtract_index_spaces(union_expr, reduce_expr);
            actual_dst_exprs.push_back(
                forest->subtract_index_spaces(dst_expr, diff_expr));
            it->set_mask -= overlap;
            if (!it->set_mask)
              break;
          }
        }
        // Handle any remaining fields
        if (!!it->set_mask)
        {
          previously_valid_masks.push_back(it->set_mask);
          actual_dst_exprs.push_back(
            forest->subtract_index_spaces(dst_expr, union_expr));
        }
      }
    }

#ifndef DISABLE_CVOPT
    /////////////////////////////////////////////////////////////
    // RemoteDeferredCopier
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDeferredCopier::RemoteDeferredCopier(RemoteTraversalInfo *info,
                                               InnerContext *ctx,
                                               MaterializedView *dst,
                                               const FieldMask &copy_mask,
                                               CopyAcrossHelper *helper)
      : DeferredCopier(info, ctx, dst, copy_mask, ApEvent::NO_AP_EVENT, helper),
        remote_info(info)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDeferredCopier::RemoteDeferredCopier(const RemoteDeferredCopier &rhs)
      : DeferredCopier(rhs), remote_info(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteDeferredCopier::~RemoteDeferredCopier(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(finalized); // we should already have been finalized
      assert(restrict_info == NULL);
#endif
      // clean up the things that we own
      delete remote_info;
      if (across_helper != NULL)
        delete across_helper;
    }

    //--------------------------------------------------------------------------
    RemoteDeferredCopier& RemoteDeferredCopier::operator=(
                                                const RemoteDeferredCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredCopier::unpack(Deserializer &derez, 
                                      const FieldMask &copy_mask)
    //--------------------------------------------------------------------------
    {
      size_t num_preconditions;
      derez.deserialize(num_preconditions);
      for (unsigned idx = 0; idx < num_preconditions; idx++)
      {
        ApEvent precondition;
        derez.deserialize(precondition);
        derez.deserialize(dst_preconditions[precondition]);
      }
      // We now have updates for all dst preconditions
      dst_precondition_mask = copy_mask;
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredCopier::unpack_write_tracker(unsigned field_index,
                   AddressSpaceID source, Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(write_trackers.find(field_index) == write_trackers.end());
#endif
      write_trackers[field_index] = 
       ShardedWriteTracker::unpack_tracker(field_index, source, runtime, derez);
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredCopier::finalize(
                                    std::map<unsigned,ApUserEvent> &done_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!finalized);
      assert(current_reduction_epoch == 0);
      finalized = true;
#endif
      // Apply any reductions that we might have
      WriteSet reduce_exprs;
      if (!reduction_epoch_masks.empty())
        apply_reduction_epochs(reduce_exprs);
      // Arm our sharded write trackers so they can send back their results
      arm_write_trackers(reduce_exprs, false/*add reference*/);
      if (!copy_postconditions.empty())
      {
        // Need to uniquify our copy_postconditions before doing this
        uniquify_copy_postconditions();
        for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
             copy_postconditions.begin(); it != copy_postconditions.end(); it++)
        {
          // Iterate over the fields and issue the triggers
          const int pop_count = it->second.pop_count();
          int next_start = 0;
          for (int idx = 0; idx < pop_count; idx++)
          {
            int field_index = it->second.find_next_set(next_start);
#ifdef DEBUG_LEGION
            assert(field_index >= 0);
#endif
            std::map<unsigned,ApUserEvent>::iterator finder = 
              done_events.find(field_index);
#ifdef DEBUG_LEGION
            assert(finder != done_events.end());
#endif
            Runtime::trigger_event(finder->second, it->first);
            // Remove it now that we've done the trigger
            done_events.erase(finder);
            next_start = field_index+1;
          }
        }
      }
      // If we have any remaining untriggered events we can trigger them now
      for (std::map<unsigned,ApUserEvent>::const_iterator it = 
            done_events.begin(); it != done_events.end(); it++)
        Runtime::trigger_event(it->second);
      done_events.clear();
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteDeferredCopier* RemoteDeferredCopier::unpack_copier(
                                  Deserializer &derez, Runtime *runtime, 
                                  const FieldMask &copy_mask, InnerContext *ctx)
    //--------------------------------------------------------------------------
    {
      DistributedID dst_did;
      derez.deserialize(dst_did);
      RtEvent dst_ready;
      LogicalView *dst = 
        runtime->find_or_request_logical_view(dst_did, dst_ready);
      RemoteTraversalInfo *info = RemoteTraversalInfo::unpack(derez, runtime); 
      bool across;
      derez.deserialize(across);
      CopyAcrossHelper *across_helper = NULL;
      if (across)
        across_helper = CopyAcrossHelper::unpack(derez, copy_mask); 
      if (dst_ready.exists() && !dst_ready.has_triggered())
        dst_ready.wait();
#ifdef DEBUG_LEGION
      assert(dst->is_materialized_view());
#endif
      MaterializedView *dst_view = dst->as_materialized_view();
      RemoteDeferredCopier *copier =
        new RemoteDeferredCopier(info, ctx, dst_view, copy_mask, across_helper);
      copier->unpack(derez, copy_mask);
      return copier;
    }
#endif

    /////////////////////////////////////////////////////////////
    // DeferredSingleCopier
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredSingleCopier::DeferredSingleCopier(const TraversalInfo *in,
        InnerContext *ctx, MaterializedView *d, const FieldMask &m, 
        const RestrictInfo &res,bool r)
      : field_index(m.find_first_set()), copy_mask(m), info(in), 
        shard_context(ctx), dst(d), across_helper(NULL), restrict_info(&res), 
        restrict_out(r), current_reduction_epoch(0), 
        write_tracker(NULL), has_dst_preconditions(false)
#ifdef DEBUG_LEGION
        , finalized(false)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredSingleCopier::DeferredSingleCopier(const TraversalInfo *in, 
        InnerContext *ctx, MaterializedView *d, const FieldMask &m, 
        ApEvent p, CopyAcrossHelper *h)
      : field_index(m.find_first_set()), copy_mask(m), info(in), 
        shard_context(ctx), dst(d), across_helper(h), restrict_info(NULL), 
        restrict_out(false), current_reduction_epoch(0), 
        write_tracker(NULL), has_dst_preconditions(true)
#ifdef DEBUG_LEGION
        , finalized(false)
#endif
    //--------------------------------------------------------------------------
    {
      dst_preconditions.insert(p);
    }

    //--------------------------------------------------------------------------
    DeferredSingleCopier::DeferredSingleCopier(const DeferredSingleCopier &rhs)
      : field_index(rhs.field_index), copy_mask(rhs.copy_mask), info(rhs.info),
        shard_context(NULL), dst(rhs.dst), across_helper(rhs.across_helper), 
        restrict_info(rhs.restrict_info), restrict_out(rhs.restrict_out)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeferredSingleCopier::~DeferredSingleCopier(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(finalized);
#endif
    }

    //--------------------------------------------------------------------------
    DeferredSingleCopier& DeferredSingleCopier::operator=(
                                                const DeferredSingleCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::merge_destination_preconditions(
                                               std::set<ApEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      if (!has_dst_preconditions)
        compute_dst_preconditions();
      preconditions.insert(dst_preconditions.begin(), dst_preconditions.end());
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::buffer_reductions(VersionTracker *tracker,
        PredEvent pred_guard, RegionTreeNode *intersect, 
        IndexSpaceExpression *mask, std::vector<ReductionView*> &src_reductions)
    //--------------------------------------------------------------------------
    {
      if (current_reduction_epoch >= reduction_epochs.size())
      {
        reduction_epochs.resize(current_reduction_epoch + 1);
#ifndef DISABLE_CVOPT
        reduction_shards.resize(current_reduction_epoch + 1);
#endif
      }
      PendingReductions &pending_reductions = 
        reduction_epochs[current_reduction_epoch];
      for (std::vector<ReductionView*>::const_iterator it = 
            src_reductions.begin(); it != src_reductions.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(pending_reductions.find(*it) == pending_reductions.end());
#endif
        pending_reductions[*it] = 
          PendingReduction(tracker, pred_guard, intersect, mask);
      }
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void DeferredSingleCopier::buffer_reduction_shards(PredEvent pred_guard,
        ReplicationID repl_id, RtEvent shard_invalid_barrier, 
        const std::map<ShardID,IndexSpaceExpression*> &source_reductions)
    //--------------------------------------------------------------------------
    {
      if (current_reduction_epoch >= reduction_epochs.size())
      {
        reduction_epochs.resize(current_reduction_epoch + 1);
        reduction_shards.resize(current_reduction_epoch + 1);
      }
      PendingReductionShards &pending_shards = 
        reduction_shards[current_reduction_epoch];
      for (std::map<ShardID,IndexSpaceExpression*>::const_iterator it = 
            source_reductions.begin(); it != source_reductions.end(); it++)
      {
        const ShardInfo key(it->first, repl_id, shard_invalid_barrier);
#ifdef DEBUG_LEGION
        assert(pending_shards.find(key) == pending_shards.end());
#endif
        pending_shards[key] = ReductionShard(pred_guard, it->second);
      }
    }
#endif

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::begin_guard_protection(void)
    //--------------------------------------------------------------------------
    {
      protected_copy_posts.resize(protected_copy_posts.size() + 1);
      copy_postconditions.swap(protected_copy_posts.back());
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::end_guard_protection(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!protected_copy_posts.empty());
#endif
      std::set<ApEvent> &target = protected_copy_posts.back();
      for (std::set<ApEvent>::const_iterator it = 
            copy_postconditions.begin(); it != copy_postconditions.end(); it++)
      {
        ApEvent protect = Runtime::ignorefaults(*it);
        if (!protect.exists())
          continue;
        target.insert(protect);
      }
      copy_postconditions.clear();
      copy_postconditions.swap(target);
      protected_copy_posts.pop_back();
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::begin_reduction_epoch(void)
    //--------------------------------------------------------------------------
    {
      current_reduction_epoch++;
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::end_reduction_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_reduction_epoch > 0);
#endif
      current_reduction_epoch--;
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::record_previously_valid(IndexSpaceExpression *ex)
    //--------------------------------------------------------------------------
    {
      // No need to record this if we're doing an across copy
      if (across_helper != NULL)
        return;
      dst_previously_valid.insert(ex);
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::finalize(DeferredView *src_view,
                                        std::set<ApEvent> *postconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!finalized);
      finalized = true;
#endif
      // Track the reduction expressions so we can update our write expr
      std::set<IndexSpaceExpression*> reduce_exprs;
      // Apply any pending reductions using the proper preconditions
#ifndef DISABLE_CVOPT
      if (!reduction_epochs.empty() || !reduction_shards.empty())
#else
      if (!reduction_epochs.empty())
#endif
        apply_reduction_epochs(reduce_exprs);
      // If we have no copy post conditions at this point we're done
      // Note that if we have remote shards we sent messages too then 
      // this is guaranteed not to be empty
      if (copy_postconditions.empty())
        return;
      // Either apply or record our postconditions
      if (postconditions == NULL)
      {
        ApEvent copy_done = Runtime::merge_events(info, copy_postconditions);
        if (copy_done.exists())
        {
          const AddressSpaceID local_space = 
            dst->context->runtime->address_space;
          // Compute our actual write condition
          // Check to see if we have a write tracker that helps
          // with the computation of the expression
          if (write_tracker != NULL)
          {
            // Arm our write tracker
            arm_write_tracker(reduce_exprs, true/*add reference*/);
            // Get whatever the tightest ready expression is
            IndexSpaceExpression *dst_expr = 
              write_tracker->pending_expr->get_ready_expression();
            dst->add_copy_user(0/*redop*/, copy_done, &info->version_info,
                               dst_expr, info->op->get_unique_op_id(), 
                               info->index, copy_mask, false/*reading*/, 
                               restrict_out, local_space, 
                               info->map_applied_events, *info);
            // Remove our reference and clean things up
            if (write_tracker->remove_reference())
              delete write_tracker;
            write_tracker = NULL;
          }
          else
          {
            IndexSpaceExpression *dst_expr = 
              dst->logical_node->get_index_space_expression();
            // Subtract out any previously valid index space expressions
            if (!dst_previously_valid.empty())
            {
              RegionTreeForest *forest = dst->context;
              IndexSpaceExpression *prev_expr = 
                forest->union_index_spaces(dst_previously_valid);
              // If we had any reductions though that overlapped then
              // we don't include those in the prev_expr
              if (!reduce_exprs.empty())
              {
                IndexSpaceExpression *reduce_expr = 
                  forest->union_index_spaces(reduce_exprs);
                prev_expr = 
                  forest->subtract_index_spaces(prev_expr, reduce_expr);
              }
              dst_expr = forest->subtract_index_spaces(dst_expr, prev_expr);
              // Tell the recorder about any empty composite views
              if (info->recording && dst_expr->is_empty())
                info->record_empty_copy(src_view, copy_mask, dst);
            }
            // Use the actually performed write to record what we wrote
            // If we're not precise we get performance bugs and deadlocks
            dst->add_copy_user(0/*redop*/, copy_done, &info->version_info,
                               dst_expr, info->op->get_unique_op_id(), 
                               info->index, copy_mask, false/*reading*/, 
                               restrict_out, local_space, 
                               info->map_applied_events, *info);
          }
          // Handle any restriction cases
          if (restrict_out && (restrict_info != NULL))
          {
            FieldMask restrict_mask;
            restrict_info->populate_restrict_fields(restrict_mask);
            if (restrict_mask.is_set(field_index))
              info->op->record_restrict_postcondition(copy_done);
          }
        }
      }
      else
      {
        // Arm our write tracker if we have one
        if (write_tracker != NULL)
          arm_write_tracker(reduce_exprs, false/*reference*/);
        postconditions->insert(copy_postconditions.begin(),
                               copy_postconditions.end());
      }
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::arm_write_tracker(
        const std::set<IndexSpaceExpression*> &reduce_exprs, bool add_reference)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(write_tracker != NULL);
#endif
      // We have a write tracker so we save any updates for it
      if (!dst_previously_valid.empty())
      {
        for (std::set<IndexSpaceExpression*>::const_iterator it = 
              dst_previously_valid.begin(); it != 
              dst_previously_valid.end(); it++)
          write_tracker->record_valid_expression(*it);
      }
      if (!reduce_exprs.empty())
      {
        for (std::set<IndexSpaceExpression*>::const_iterator it = 
              reduce_exprs.begin(); it != reduce_exprs.end(); it++)
          write_tracker->record_sub_expression(*it);
      }
      if (add_reference)
        write_tracker->add_reference();
      if (write_tracker->arm())
        delete write_tracker;
      if (!add_reference)
        write_tracker = NULL;
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void DeferredSingleCopier::pack_copier(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(dst->did);
      info->pack(rez);
      if (across_helper != NULL)
      {
        rez.serialize<bool>(true); // across
        across_helper->pack(rez, copy_mask);
      }
      else
        rez.serialize<bool>(false); // across
      if (!has_dst_preconditions)
        compute_dst_preconditions();
      rez.serialize<size_t>(dst_preconditions.size());
      for (std::set<ApEvent>::const_iterator it = 
            dst_preconditions.begin(); it != dst_preconditions.end(); it++)
        rez.serialize(*it);
    }
#endif

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::pack_sharded_write_tracker(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      if (write_tracker == NULL)
        write_tracker = new ShardedWriteTracker(field_index,
            dst->context, dst->logical_node->get_index_space_expression());
      write_tracker->pack_for_remote_shard(rez);
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::compute_dst_preconditions(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!has_dst_preconditions);
#endif
      has_dst_preconditions = true;
      const AddressSpaceID local_space = dst->context->runtime->address_space;
      LegionMap<ApEvent,FieldMask>::aligned temp_preconditions;
      // Note here that we set 'can_filter' to false because this is an
      // over-approximation for the copy that we're eventually going to
      // be issuing, so we can't be sure we're writing all of it
      // The only exception here is in the case where we're doing a copy
      // across in which case we know we're going to write everything
      dst->find_copy_preconditions(0/*redop*/, false/*reading*/,
                               false/*single copy*/, restrict_out, copy_mask,
                               dst->logical_node->get_index_space_expression(),
                               &info->version_info,
                               info->op->get_unique_op_id(), info->index,
                               local_space, temp_preconditions,
                               info->map_applied_events, *info,
                               (across_helper != NULL)/*can filter*/);
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            temp_preconditions.begin(); it != temp_preconditions.end(); it++)
        dst_preconditions.insert(it->first);
      if ((restrict_info != NULL) && restrict_info->has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info->populate_restrict_fields(restrict_mask); 
        if (restrict_mask.is_set(field_index))
          dst_preconditions.insert(info->op->get_restrict_precondition(*info));
      }
    }

    //--------------------------------------------------------------------------
    void DeferredSingleCopier::apply_reduction_epochs(
                                  std::set<IndexSpaceExpression*> &reduce_exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
#ifndef DISABLE_CVOPT
      assert(reduction_epochs.size() == reduction_shards.size());
#endif
#endif
      if (!has_dst_preconditions)
        compute_dst_preconditions();
      // We'll just use the destination preconditions here for accumulation
      // We're not going to need it again later anyway
      if (!copy_postconditions.empty())
        dst_preconditions.insert(copy_postconditions.begin(),
                                 copy_postconditions.end());
      ApEvent reduction_pre = Runtime::merge_events(info, dst_preconditions);
      // Iterate epochs in reverse order as the deepest ones are the
      // ones that should be issued first
      for (int epoch = reduction_epochs.size()-1; epoch >= 0; epoch--)
      {
        std::set<ApEvent> reduction_postconditions;
#ifndef DISABLE_CVOPT
        PendingReductionShards &pending_shards = reduction_shards[epoch];
        // First send messages to any shards to get things in flight
        for (PendingReductionShards::const_iterator it = 
              pending_shards.begin(); it != pending_shards.end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(shard_context != NULL);
#endif
          Serializer rez;
          rez.serialize(it->first.repl_id);
          rez.serialize(it->first.shard);
          rez.serialize<RtEvent>(it->first.shard_invalid_barrier);
          rez.serialize(dst->did);
          info->pack(rez);
          if (across_helper != NULL)
          {
            rez.serialize<bool>(true);
            rez.serialize(copy_mask);
            across_helper->pack(rez, copy_mask);
          }
          else
            rez.serialize<bool>(false);
          rez.serialize<size_t>(1); // number of pending
#ifdef DEBUG_LEGION
          assert(copy_mask.pop_count() == 1);
#endif
          rez.serialize(copy_mask);
          rez.serialize(it->second.pred_guard);
          it->second.mask->pack_expression(rez,
              shard_context->find_shard_space(it->first.shard));
          ApUserEvent done_event = Runtime::create_ap_user_event();
          reduction_postconditions.insert(done_event);
          rez.serialize(done_event);
          // Also need to send the preconditions for the reductions
          rez.serialize(reduction_pre);
          if (across_helper != NULL)
            pack_sharded_write_tracker(rez);
          shard_context->send_composite_view_shard_reduction_request(
                                                it->first.shard, rez);
        }
#endif
        PendingReductions &pending_reductions = reduction_epochs[epoch];
        // Issue all the reductions
        for (PendingReductions::const_iterator it = 
              pending_reductions.begin(); it != pending_reductions.end(); it++)
        {
          IndexSpaceExpression *reduce_expr = NULL;
          ApEvent reduction_post = it->first->perform_deferred_reduction(
              dst, copy_mask, it->second.version_tracker, reduction_pre,
              info->op, info->index, it->second.pred_guard, across_helper,
              it->second.intersect, it->second.mask,
              info->map_applied_events, *info, reduce_expr);
          if (reduction_post.exists())
            reduction_postconditions.insert(reduction_post);
          if ((reduce_expr != NULL) && !dst_previously_valid.empty())
            reduce_exprs.insert(reduce_expr);
        }
        if (reduction_postconditions.empty())
          continue;
        if (epoch > 0)
        {
          reduction_pre = Runtime::merge_events(info, reduction_postconditions);
          // In case we don't end up using it later
          copy_postconditions.insert(reduction_pre);
        }
        else
          copy_postconditions.insert(reduction_postconditions.begin(),
                                     reduction_postconditions.end());
      }
    }

#ifndef DISABLE_CVOPT
    /////////////////////////////////////////////////////////////
    // RemoteDeferredSingleCopier
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteDeferredSingleCopier::RemoteDeferredSingleCopier(
                                          RemoteTraversalInfo *info, 
                                          InnerContext *ctx,
                                          MaterializedView *dst, 
                                          const FieldMask &copy_mask,
                                          CopyAcrossHelper *helper)
      : DeferredSingleCopier(info, ctx, dst, copy_mask, 
                             ApEvent::NO_AP_EVENT, helper), remote_info(info)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteDeferredSingleCopier::RemoteDeferredSingleCopier(
                                          const RemoteDeferredSingleCopier &rhs)
      : DeferredSingleCopier(rhs), remote_info(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteDeferredSingleCopier::~RemoteDeferredSingleCopier(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(finalized); // we should already have been finalized
      assert(restrict_info == NULL);
#endif
      // clean up the things we own
      delete remote_info;
      if (across_helper != NULL)
        delete across_helper;
    }

    //--------------------------------------------------------------------------
    RemoteDeferredSingleCopier& RemoteDeferredSingleCopier::operator=(
                                          const RemoteDeferredSingleCopier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredSingleCopier::unpack(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_preconditions;
      derez.deserialize(num_preconditions);
      for (unsigned idx = 0; idx < num_preconditions; idx++)
      {
        ApEvent precondition;
        derez.deserialize(precondition);
        dst_preconditions.insert(precondition);
      }
      has_dst_preconditions = true;
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredSingleCopier::unpack_write_tracker(AddressSpaceID source,
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(write_tracker == NULL);
#endif
      write_tracker = 
       ShardedWriteTracker::unpack_tracker(field_index, source, runtime, derez);
    }

    //--------------------------------------------------------------------------
    void RemoteDeferredSingleCopier::finalize(ApUserEvent done_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!finalized);
      finalized = true;
#endif
      std::set<IndexSpaceExpression*> reduce_exprs;
      // Apply any pending reductions using the proper preconditions
      if (!reduction_epochs.empty() || !reduction_shards.empty())
        apply_reduction_epochs(reduce_exprs);
      // Arm our sharded write tracker so that it can send back its result
      arm_write_tracker(reduce_exprs, false/*add reference*/);
      // Trigger our dependent events
      if (!copy_postconditions.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(info, copy_postconditions));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteDeferredSingleCopier* 
      RemoteDeferredSingleCopier::unpack_copier(Deserializer &derez, 
                Runtime *runtime, const FieldMask &copy_mask, InnerContext *ctx)
    //--------------------------------------------------------------------------
    {
      DistributedID dst_did;
      derez.deserialize(dst_did);
      RtEvent dst_ready;
      LogicalView *dst = 
        runtime->find_or_request_logical_view(dst_did, dst_ready);
      RemoteTraversalInfo *info = RemoteTraversalInfo::unpack(derez, runtime);
      bool across;
      derez.deserialize(across);
      CopyAcrossHelper *across_helper = NULL;
      if (across)
        across_helper = CopyAcrossHelper::unpack(derez, copy_mask); 
      if (dst_ready.exists() && !dst_ready.has_triggered())
        dst_ready.wait();
#ifdef DEBUG_LEGION
      assert(dst->is_materialized_view());
#endif
      MaterializedView *dst_view = dst->as_materialized_view();
      RemoteDeferredSingleCopier *copier =
        new RemoteDeferredSingleCopier(info, ctx, dst_view,
                                       copy_mask, across_helper);
      copier->unpack(derez);
      return copier;
    }
#endif

    /////////////////////////////////////////////////////////////
    // DeferredView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredView::DeferredView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp,
                               RegionTreeNode *node, bool register_now)
      : LogicalView(ctx, did, owner_sp, node, register_now)
#ifdef DEBUG_LEGION
        , currently_active(true), currently_valid(true)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DeferredView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
       if (!is_owner())
        send_remote_gc_update(owner_space, mutator, 1/*count*/, true/*add*/);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(currently_active);
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void DeferredView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
#ifdef DEBUG_LEGION
        assert(currently_active); // should be montonic
        currently_active = true;
#endif
        notify_owner_inactive(mutator);
      }
      else
        send_remote_gc_update(owner_space, mutator, 1/*count*/, false/*add*/);
    }

    //--------------------------------------------------------------------------
    void DeferredView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_valid_update(owner_space, mutator, 1/*count*/, true/*add*/);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
        assert(currently_valid);
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void DeferredView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
      {
#ifdef DEBUG_LEGION
        assert(currently_valid); // should be monotonic
        currently_valid = false;
#endif
        notify_owner_invalid(mutator);
      }
      else
        send_remote_valid_update(owner_space, mutator, 1/*count*/,false/*add*/);
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
#ifdef DEBUG_LEGION
      assert(src_indexes.size() == dst_indexes.size());
#endif
      bool perfect = true;
      FieldMask src_mask, dst_mask;
      for (unsigned idx = 0; idx < dst_indexes.size(); idx++)
      {
        src_mask.set_bit(src_indexes[idx]);
        dst_mask.set_bit(dst_indexes[idx]);
        if (perfect && (src_indexes[idx] != dst_indexes[idx]))
          perfect = false;
      }
      if (src_indexes.size() == 1)
      {
        IndexSpaceExpression *write_performed = NULL;
        if (perfect)
        {
          DeferredSingleCopier copier(&info, get_shard_context(), dst, 
                                      src_mask, precondition);
          issue_deferred_copies_single(copier, NULL/*write mask*/,
                                       write_performed, guard);
          copier.finalize(this, &postconditions);
        }
        else
        {
          // Initialize the across copy helper
          CopyAcrossHelper across_helper(src_mask);
          dst->manager->initialize_across_helper(&across_helper, dst_mask, 
                                                 src_indexes, dst_indexes);
          DeferredSingleCopier copier(&info, get_shard_context(), dst, src_mask,
                                      precondition, &across_helper);
          issue_deferred_copies_single(copier, NULL/*write mask*/,
                                       write_performed, guard);
          copier.finalize(this, &postconditions);
        }
      }
      else
      {
        WriteMasks write_masks; 
        WriteSet performed_writes;
        if (perfect)
        {
          DeferredCopier copier(&info, get_shard_context(), dst, 
                                src_mask, precondition);
          issue_deferred_copies(copier, src_mask, write_masks, 
                                performed_writes, guard);
          copier.finalize(this, &postconditions);
        }
        else
        {
          // Initialize the across copy helper
          CopyAcrossHelper across_helper(src_mask);
          dst->manager->initialize_across_helper(&across_helper, dst_mask, 
                                                 src_indexes, dst_indexes);
          DeferredCopier copier(&info, get_shard_context(), dst,
                                src_mask, precondition, &across_helper);
          issue_deferred_copies(copier, src_mask, write_masks,
                                performed_writes, guard);
          copier.finalize(this, &postconditions);
        }
      }
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
    // CompositeReducer
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeReducer::CompositeReducer(TraversalInfo *in, InnerContext *ctx,
                                       MaterializedView *d, const FieldMask &r,
                                       CopyAcrossHelper *h)
      : info(in), context(ctx), dst(d), reduction_mask(r), across_helper(h)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeReducer::CompositeReducer(const CompositeReducer &rhs)
      : info(rhs.info), context(rhs.context), dst(rhs.dst), 
        reduction_mask(rhs.reduction_mask), across_helper(rhs.across_helper)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeReducer::~CompositeReducer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeReducer& CompositeReducer::operator=(const CompositeReducer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeReducer::unpack_write_tracker(unsigned field_index,
                                                Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_trackers.find(field_index) == remote_trackers.end());
#endif
      std::pair<ShardedWriteTracker*,RtUserEvent> &tracker = 
        remote_trackers[field_index];
      ShardedWriteTracker::unpack_tracker(derez, tracker.first, tracker.second);
    }

    //--------------------------------------------------------------------------
    void CompositeReducer::unpack(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_preconditions;
      derez.deserialize(num_preconditions);
      for (unsigned idx = 0; idx < num_preconditions; idx++)
      {
        ApEvent pre;
        derez.deserialize(pre);
        derez.deserialize(reduce_preconditions[pre]);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent CompositeReducer::find_precondition(const FieldMask &mask) const
    //--------------------------------------------------------------------------
    {
      std::set<ApEvent> preconditions;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
           reduce_preconditions.begin(); it != reduce_preconditions.end(); it++)
      {
        if (it->second * mask)
          continue;
        preconditions.insert(it->first);
      }
      if (preconditions.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(info, preconditions);
    }

    //--------------------------------------------------------------------------
    void CompositeReducer::record_postcondition(ApEvent done, 
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduce_postconditions.find(done) == reduce_postconditions.end());
#endif
      reduce_postconditions[done] = mask;
    }

    //--------------------------------------------------------------------------
    void CompositeReducer::record_expression(IndexSpaceExpression *expr,
                                             const FieldMask &expr_mask)
    //--------------------------------------------------------------------------
    {
      WriteSet::iterator finder = reduce_expressions.find(expr);
      if (finder == reduce_expressions.end())
        reduce_expressions.insert(expr, expr_mask);
      else
        finder.merge(expr_mask);
    }

    //--------------------------------------------------------------------------
    void CompositeReducer::finalize(std::map<unsigned,ApUserEvent> &done_events,
                                    Runtime *runtime, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Short our postconditions into event sets
      if (!reduce_postconditions.empty())
      {
        LegionList<FieldSet<ApEvent> >::aligned event_sets;
        compute_field_sets<ApEvent>(reduction_mask, 
                                    reduce_postconditions, event_sets);
        for (LegionList<FieldSet<ApEvent> >::aligned::const_iterator it =
              event_sets.begin(); it != event_sets.end(); it++)
        {
          ApEvent done = Runtime::merge_events(info, it->elements);
          // Iterate through the field indexes
          int next_start = 0;
          const size_t pop_count = it->set_mask.pop_count();
          for (unsigned idx = 0; idx < pop_count; idx++)
          {
            int field_index = it->set_mask.find_next_set(next_start);
#ifdef DEBUG_LEGION
            assert(field_index >= 0);
#endif
            std::map<unsigned,ApUserEvent>::iterator finder = 
              done_events.find(field_index);
#ifdef DEBUG_LEGION
            assert(finder != done_events.end());
#endif
            Runtime::trigger_event(finder->second, done);
            done_events.erase(finder);
            next_start = field_index + 1;
          }
        }
      }
      // Trigger any remaining events
      if (!done_events.empty())
      {
        for (std::map<unsigned,ApUserEvent>::const_iterator it = 
              done_events.begin(); it != done_events.end(); it++)
          Runtime::trigger_event(it->second);
        done_events.clear();
      }
      if (!remote_trackers.empty())
      {
        LegionList<FieldSet<IndexSpaceExpression*> >::aligned reduce_sets;
        reduce_expressions.compute_field_sets(reduction_mask, reduce_sets);
        for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
              const_iterator it = reduce_sets.begin(); 
              it != reduce_sets.end(); it++)
        {
          if (it->elements.empty())
          {
            int field_index = it->set_mask.find_first_set();
            while (field_index >= 0)
            {
              std::map<unsigned,
                       std::pair<ShardedWriteTracker*,RtUserEvent> >::iterator
                  finder = remote_trackers.find(field_index);
#ifdef DEBUG_LEGION
              assert(finder != remote_trackers.end());
#endif
              Runtime::trigger_event(finder->second.second);
              remote_trackers.erase(finder);
              // find the next field to handle
              field_index = it->set_mask.find_next_set(field_index+1);
            }
          }
          else
          {
            IndexSpaceExpression *union_expr = 
              runtime->forest->union_index_spaces(it->elements);
            int field_index = it->set_mask.find_first_set();
            while (field_index >= 0)
            {
               std::map<unsigned,
                       std::pair<ShardedWriteTracker*,RtUserEvent> >::iterator
                  finder = remote_trackers.find(field_index);
#ifdef DEBUG_LEGION
              assert(finder != remote_trackers.end());
#endif
              ShardedWriteTracker::send_shard_sub(runtime, finder->second.first,
                                     target, union_expr, finder->second.second);
              remote_trackers.erase(finder);
              // find the next field to handle
              field_index = it->set_mask.find_next_set(field_index+1);
            }
          }
        }
#ifdef DEBUG_LEGION
        // We should have handled all the remote trackers
        assert(remote_trackers.empty());
#endif
      }
    }

    /////////////////////////////////////////////////////////////
    // CompositeSingleReducer
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeSingleReducer::CompositeSingleReducer(TraversalInfo *in, 
                                       InnerContext *ctx, MaterializedView *d,
                                       const FieldMask &r, ApEvent p,
                                       CopyAcrossHelper *h)
      : info(in), context(ctx), dst(d), reduction_mask(r), 
        field_index(r.find_first_set()), reduce_pre(p), across_helper(h),
        remote_tracker(NULL), remote_event(RtUserEvent::NO_RT_USER_EVENT)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeSingleReducer::CompositeSingleReducer(
                                              const CompositeSingleReducer &rhs)
      : info(rhs.info), context(rhs.context), dst(rhs.dst), 
        reduction_mask(rhs.reduction_mask), field_index(rhs.field_index),
        reduce_pre(rhs.reduce_pre), across_helper(rhs.across_helper)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeSingleReducer::~CompositeSingleReducer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeSingleReducer& CompositeSingleReducer::operator=(
                                              const CompositeSingleReducer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CompositeSingleReducer::unpack_write_tracker(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_tracker == NULL);
#endif
      ShardedWriteTracker::unpack_tracker(derez, remote_tracker, remote_event);
    }

    //--------------------------------------------------------------------------
    void CompositeSingleReducer::finalize(ApUserEvent done_event,
                                        Runtime *runtime, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (!reduce_postconditions.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(info, reduce_postconditions));
      else
        Runtime::trigger_event(done_event);
      if (remote_tracker != NULL)
      {
        if (!reduce_expressions.empty())
        {
          IndexSpaceExpression *union_expr = 
            runtime->forest->union_index_spaces(reduce_expressions);
          ShardedWriteTracker::send_shard_sub(runtime, remote_tracker, 
                                              target, union_expr, remote_event);
        }
        else
          Runtime::trigger_event(remote_event);
      }
    }

    /////////////////////////////////////////////////////////////
    // CompositeBase 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeBase::CompositeBase(LocalLock &r, bool shard)
      : base_lock(r), composite_shard(shard)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeBase::~CompositeBase(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void CompositeBase::issue_composite_updates(DeferredCopier &copier,
                                                RegionTreeNode *logical_node,
                                          const FieldMask &local_copy_mask,
                                          VersionTracker *src_version_tracker,
                                          PredEvent pred_guard, 
                                          const WriteMasks &write_masks,
                                                WriteSet &performed_writes,
                                                bool need_shard_check/*=true*/)
    //--------------------------------------------------------------------------
    { 
      FieldMask &global_copy_mask = copier.deferred_copy_mask;
#ifdef DEBUG_LEGION
      assert(!(local_copy_mask - global_copy_mask)); // only true at beginning
#endif
      // If this is a composite shard, check to see if we need to issue any
      // updates from shards on remote nodes, no need to do this if we are
      // already on a remote node as someone else already did the check
      if (need_shard_check && composite_shard && !copier.is_remote()) 
      {
        WriteMasks shard_writes;
        // Perform any shard writes and then update our write masks if necessary
        issue_shard_updates(copier, logical_node, local_copy_mask, 
                            pred_guard, write_masks, shard_writes);
        if (!shard_writes.empty())
        {
          // Add our shard writes to the performed write set
          performed_writes.merge(shard_writes);
          // Now build a new write set for going down
          if (!write_masks.empty())
            shard_writes.merge(write_masks);
          combine_writes(shard_writes, copier);
          // Check to see if we still have fields to issue copies for
          if (!!global_copy_mask)
          {
            const FieldMask new_local_mask = local_copy_mask & global_copy_mask;
            issue_composite_updates(copier, logical_node, new_local_mask,
                           src_version_tracker, pred_guard, shard_writes, 
                           performed_writes, false/*don't need shard check*/);
          }
          // Return because we already traversed down
          return;
        }
        // Otherwise we can fall through and keep going
      }
      // First check to see if we have all the valid meta-data
#ifndef DISABLE_CVOPT
      perform_ready_check(local_copy_mask);
#else
      perform_ready_check(local_copy_mask, logical_node);
#endif
      // Find any children that need to be traversed as well as any instances
      // or reduction views that we may need to issue copies from
      LegionMap<CompositeNode*,FieldMask>::aligned children_to_traverse;
      LegionMap<LogicalView*,FieldMask>::aligned source_views;
      LegionMap<ReductionView*,FieldMask>::aligned source_reductions;
      {
        AutoLock b_lock(base_lock,1,false/*exclusive*/);
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          const FieldMask child_mask = it->second & local_copy_mask;
          if (!child_mask)
            continue;
          // Skip any nodes that don't even intersect, they don't matter
          if (!it->first->logical_node->intersects_with(
                copier.dst->logical_node, false/*computes*/))
            continue;
          children_to_traverse[it->first] = child_mask;
        }
        // Get any valid views that we need for this level
        find_valid_views(local_copy_mask, source_views, false/*needs lock*/);
        // Also get any reduction views that we need for this level
        if (!(reduction_mask * local_copy_mask))
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            const FieldMask overlap = it->second & local_copy_mask;
            if (!overlap)
              continue;
            source_reductions[it->first] = overlap;
          }
        }
      }
      if (!children_to_traverse.empty())
      {
        // Do the child traversals and record any writes that we do
        WriteSet child_writes;
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it =
              children_to_traverse.begin(); it != 
              children_to_traverse.end(); it++)
        {
          // Filter by the global copy mask in case it has changed
          it->second &= global_copy_mask;
          if (!it->second)
            continue;
          it->first->issue_composite_updates(copier, it->first->logical_node,
                                it->second, src_version_tracker, pred_guard,
                                write_masks, child_writes);
          // Special case if it any point our global copy mask becomes empty
          // then we can break out because we've issue all our writes, still
          // need to check for any reductions though
          if (!global_copy_mask)
          {
            // If we don't have any source reductions we can just return
            if (source_reductions.empty())
            {
              // Record any merges on the way up
              if (!child_writes.empty())
                performed_writes.merge(child_writes);
              return;
            }
            else 
              // If we have source reductions then we still need to record them
              break;
          }
        }
        if (!source_views.empty())
        {
          WriteMasks previous_writes;
          const WriteMasks *all_previous_writes = NULL;
          // We need to build a local set of source views based on writes
          // we did locally and writes we did below and also update the 
          // global copy mask if necessary
          if (child_writes.empty())
            // Assume these are already write-combined
            all_previous_writes = &write_masks;
          else if (write_masks.empty())
          {
            combine_writes(child_writes, copier);
            all_previous_writes = &child_writes;
          }
          else
          {
            previous_writes = child_writes;
            previous_writes.merge(write_masks);
            // Do write combining which can reduce the global copy mask
            combine_writes(previous_writes, copier);
            all_previous_writes = &previous_writes;
          }
          // Issue our writes from our physical instances
          // If global copy mask is empty though we don't need to 
          // do this as we've already done all the writes
          WriteSet local_writes;
          if (!!global_copy_mask)
            issue_update_copies(copier, logical_node, 
                global_copy_mask & local_copy_mask, src_version_tracker,
                pred_guard, source_views, *all_previous_writes, local_writes);
          // Finally merge our write updates with our child writes
          // if necessary and propagate them back up the tree
          // and propagate up the tree
          if (child_writes.empty())
          {
            // No need to write combine ourselves, just merge
            if (!local_writes.empty())
              performed_writes.merge(local_writes);
          }
          else
          {
            if (!local_writes.empty())
            {
              // Need to merge everything together and write combine them
              local_writes.merge(child_writes);
              combine_writes(local_writes, copier);
              performed_writes.merge(local_writes);
            }
            else // children are already write combined, no need to do it again
              performed_writes.merge(child_writes);
          }
        }
        else if (!child_writes.empty())
          // Propagate child writes up the tree
          performed_writes.merge(child_writes);
      }
      else if (!source_views.empty())
      {
        // We didn't do any child traversals so things are a little easier
        // Issue our writes from our physical instances
        WriteSet local_writes;
        issue_update_copies(copier, logical_node, local_copy_mask, 
                           src_version_tracker, pred_guard, source_views, 
                           write_masks, local_writes);
        // Finally merge our write updates into the performed set
        if (!local_writes.empty())
          performed_writes.merge(local_writes);
      }
      // Lastly no matter what we do, we have to record our reductions
      // to be performed after all the updates to the instance
      if (!source_reductions.empty())
        copier.buffer_reductions(src_version_tracker, pred_guard,
            (copier.dst->logical_node == logical_node) ? NULL : logical_node,
            write_masks, source_reductions);
    }

    //--------------------------------------------------------------------------
    bool CompositeBase::issue_composite_updates_single(
                                   DeferredSingleCopier &copier,
                                   RegionTreeNode *logical_node,
                                   VersionTracker *src_version_tracker,
                                   PredEvent pred_guard,
                                   IndexSpaceExpression *write_mask,
                                   IndexSpaceExpression *&performed_write,
                                   bool need_shard_check/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(performed_write == NULL); // should be NULL on the way in
#endif
      RegionTreeForest *context = logical_node->context;
      // If this is a composite shard, check to see if we need to issue any
      // updates from shards on remote nodes, no need to do this if we are
      // already on a remote node as someone else already did the check
      if (need_shard_check && composite_shard && !copier.is_remote())
      {
        IndexSpaceExpression *shard_write = issue_shard_updates_single(copier,
                                        logical_node, pred_guard, write_mask);
        if (shard_write != NULL)
        {
          IndexSpaceExpression *local_performed = NULL;
          if (write_mask != NULL)
          {
            // Make a new write mask
            IndexSpaceExpression *new_write_mask = 
              context->union_index_spaces(write_mask, shard_write);
            issue_composite_updates_single(copier, logical_node,
                src_version_tracker, pred_guard, new_write_mask,
                local_performed, false/*need shard check*/);
          }
          else
            issue_composite_updates_single(copier, logical_node, 
                  src_version_tracker, pred_guard, shard_write, 
                  local_performed, false/*need shard_check*/);
          if (local_performed != NULL)
            // Merge together the two writes
            performed_write = 
              context->union_index_spaces(shard_write, local_performed);
          else
            performed_write = shard_write;
          // Return at this point since we already recursed
          if (performed_write != NULL)
            return test_done(copier, performed_write, write_mask);
          return false;
        }
        // Otherwise we can fall through and keep going since we did no
        // shard writes on remote nodes
      }
      // First check to see if we have all the valid meta-data
#ifndef DISABLE_CVOPT
      perform_ready_check(copier.copy_mask);
#else
      perform_ready_check(copier.copy_mask, logical_node);
#endif
      // Find any children that need to be traversed as well as any instances
      // or reduction views that we may need to issue copies from
      std::vector<CompositeNode*> children_to_traverse;
      LegionMap<LogicalView*,FieldMask>::aligned temp_views;
      std::vector<ReductionView*> source_reductions;
      {
        AutoLock b_lock(base_lock,1,false/*exclusive*/);
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if (!it->second.is_set(copier.field_index))
            continue;
          // Skip any nodes that don't even intersect, they don't matter
          if (!it->first->logical_node->intersects_with(
                copier.dst->logical_node, false/*computes*/))
            continue;
          children_to_traverse.push_back(it->first);
        }
        // Get any valid views that we need for this level
        find_valid_views(copier.copy_mask, temp_views, false/*needs lock*/);
        if (reduction_mask.is_set(copier.field_index))
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            if (!it->second.is_set(copier.field_index))
              continue;
            source_reductions.push_back(it->first);
          }
        }
      }
      std::vector<LogicalView*> source_views;
      if (!temp_views.empty())
      {
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              temp_views.begin(); it != temp_views.end(); it++)
          source_views.push_back(it->first);
      }
      bool done = false; 
      if (!children_to_traverse.empty())
      {
        // Do the child traversals and record any writes that we do
        std::set<IndexSpaceExpression*> child_writes;
        for (std::vector<CompositeNode*>::const_iterator it = 
              children_to_traverse.begin(); it != 
              children_to_traverse.end(); it++)
        {
          IndexSpaceExpression *child_write = NULL;
          done = (*it)->issue_composite_updates_single(
              copier, (*it)->logical_node, src_version_tracker,
              pred_guard, write_mask, child_write);
          if (child_write != NULL)
            child_writes.insert(child_write);
          if (done)
            break;
        }
        if (!child_writes.empty())
        {
          if (child_writes.size() == 1)
            performed_write = *child_writes.begin();
          else
            performed_write = 
              context->union_index_spaces(child_writes);
          done = test_done(copier, performed_write, write_mask);
        }
        if (!done && !source_views.empty())
        {
          if (performed_write != NULL)
          {
            IndexSpaceExpression *our_write = NULL;
            if (write_mask != NULL)
            {
              // Compute a new write mask for the 
              write_mask = 
                context->union_index_spaces(write_mask, performed_write);
              our_write = issue_update_copies_single(copier, logical_node,
                src_version_tracker, pred_guard, source_views, write_mask);
            }
            else
              our_write = issue_update_copies_single(copier, logical_node,
                src_version_tracker, pred_guard, source_views, performed_write);
            if (our_write != NULL)
              performed_write = 
                context->union_index_spaces(performed_write, our_write);
          }
          else
            performed_write = issue_update_copies_single(
                copier, logical_node, src_version_tracker,
                pred_guard, source_views, write_mask);
        }
      }
      else if (!source_views.empty())
        // We didn't do any child traversals so things are a little easier
        performed_write = issue_update_copies_single(copier, logical_node,
              src_version_tracker, pred_guard, source_views, write_mask);
      // Lastly no matter what we do, we have to record our reductions
      // to be performed after all the updates to the instance
      if (!source_reductions.empty())
        copier.buffer_reductions(src_version_tracker, pred_guard,
            (copier.dst->logical_node == logical_node) ? NULL : logical_node,
            write_mask, source_reductions);
      if (!done && (performed_write != NULL))
        done = test_done(copier, performed_write, write_mask);
      return done;
    }

    //--------------------------------------------------------------------------
    void CompositeBase::issue_composite_reductions(CompositeReducer &reducer,
                                            const FieldMask &local_mask,
                                            const WriteMasks &needed_exprs,
                                            RegionTreeNode *logical_node,
                                            PredEvent pred_guard,
                                            VersionTracker *src_version_tracker)
    //--------------------------------------------------------------------------
    {
      // Issue any reductions at this level and then traverse the children
#ifndef DISABLE_CVOPT
      perform_ready_check(local_mask);
#else
      perform_ready_check(local_mask, reducer.dst->logical_node);
#endif
      LegionMap<ReductionView*,FieldMask>::aligned to_perform;
      LegionMap<CompositeNode*,FieldMask>::aligned to_traverse;
      {
        AutoLock b_lock(base_lock,1,false/*exclusive*/);
        if (!(reduction_mask * reducer.reduction_mask))
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            const FieldMask overlap = it->second & local_mask;
            if (!overlap)
              continue;
            to_perform[it->first] = overlap;
          }
        }
        if (!children.empty())
        {
          for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator 
                it = children.begin(); it != children.end(); it++)
          {
            const FieldMask overlap = it->second & local_mask;
            if (!overlap)
              continue;
            to_traverse[it->first] = overlap;
          }
        }
      }
      // Issue our reductions
      if (!to_perform.empty())
      {
        RegionTreeForest *context = logical_node->context;
        std::map<IndexSpaceExpression*,IndexSpaceExpression*> masks;
        for (LegionMap<ReductionView*,FieldMask>::aligned::iterator it =
              to_perform.begin(); it != to_perform.end(); it++)
        {
          // Iterate over the needed exprs and find the interfering ones
          for (WriteMasks::const_iterator mit = needed_exprs.begin();
                mit != needed_exprs.end(); mit++)
          {
            const FieldMask overlap = it->second & mit->second;
            if (!overlap)
              continue;
            const ApEvent reduce_pre = reducer.find_precondition(overlap);
            // See if we already computed the mask
            std::map<IndexSpaceExpression*,IndexSpaceExpression*>::
              const_iterator finder = masks.find(mit->first);
            IndexSpaceExpression *mask = NULL;
            if (finder == masks.end())
            {
              // Compute the mask for this needed reduction
              mask = context->subtract_index_spaces(
                  reducer.dst->logical_node->get_index_space_expression(), 
                  mit->first);
              masks[mit->first] = mask;
            }
            else
              mask = finder->second;
            IndexSpaceExpression *reduce_expr = NULL;
            ApEvent reduce_done = it->first->perform_deferred_reduction(
                reducer.dst, overlap, src_version_tracker,
                reduce_pre, reducer.info->op, reducer.info->index,
                pred_guard, reducer.across_helper, logical_node, mask, 
                reducer.info->map_applied_events, *reducer.info, reduce_expr);
            if (reduce_done.exists())
              reducer.record_postcondition(reduce_done, overlap);
            // Record our reduce expression if it exists
            if ((reduce_expr != NULL) && !reduce_expr->is_empty())
              reducer.record_expression(reduce_expr, overlap);
            // Can remove these fields from the set
            it->second -= overlap;
            // Once we've done all the fields then we are done
            if (!it->second)
              break;
          }
        }
      }
      // Then traverse our children
      if (to_traverse.empty())
        return;
      RegionTreeForest *context = reducer.dst->logical_node->context;
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        // Build the needed reduction expressions for the child
        WriteMasks needed_child_reductions;
        for (WriteMasks::const_iterator mit = needed_exprs.begin();
              mit != needed_exprs.end(); mit++)
        {
          const FieldMask overlap = mit->second & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *child_mask = 
            context->intersect_index_spaces(mit->first, 
              it->first->logical_node->get_index_space_expression());
          if (child_mask->is_empty())
            continue;
#ifdef DEBUG_LEGION
          assert(needed_child_reductions.find(child_mask) == 
                  needed_child_reductions.end());
#endif
          needed_child_reductions.insert(child_mask, overlap);
        }
        it->first->issue_composite_reductions(reducer, it->second,
              needed_child_reductions, it->first->logical_node,
              pred_guard, src_version_tracker);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeBase::issue_composite_reductions_single(
                                          CompositeSingleReducer &reducer,
                                          IndexSpaceExpression *needed_expr,
                                          RegionTreeNode *logical_node,
                                          PredEvent pred_guard,
                                          VersionTracker *src_version_tracker)
    //--------------------------------------------------------------------------
    {
      // Issue any reductions at this level and then traverse the children
#ifndef DISABLE_CVOPT
      perform_ready_check(reducer.reduction_mask);
#else
      perform_ready_check(reducer.reduction_mask, reducer.dst->logical_node);
#endif
      std::vector<ReductionView*> to_perform;
      std::vector<CompositeNode*> to_traverse;
      {
        AutoLock b_lock(base_lock,1,false/*exclusive*/);
        if (reduction_mask.is_set(reducer.field_index))
        {
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it != reduction_views.end(); it++)
          {
            if (it->second.is_set(reducer.field_index))
              to_perform.push_back(it->first);
          }
        }
        if (!children.empty())
        {
          for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator 
                it = children.begin(); it != children.end(); it++)
          {
            if (it->second.is_set(reducer.field_index))
              to_traverse.push_back(it->first);
          }
        }
      }
      // Issue our reductions
      if (!to_perform.empty())
      {
        // Compute our mask since that is what is needed to apply the reductions
        RegionTreeForest *context = logical_node->context;
        IndexSpaceExpression *reduce_mask = context->subtract_index_spaces(
            reducer.dst->logical_node->get_index_space_expression(),
            needed_expr);
        for (std::vector<ReductionView*>::const_iterator it = 
              to_perform.begin(); it != to_perform.end(); it++)
        {
          IndexSpaceExpression *reduce_expr = NULL;
          ApEvent reduce_done = (*it)->perform_deferred_reduction(
              reducer.dst, reducer.reduction_mask, src_version_tracker,
              reducer.reduce_pre, reducer.info->op, reducer.info->index,
              pred_guard, reducer.across_helper, logical_node, reduce_mask,
              reducer.info->map_applied_events, *reducer.info, reduce_expr);
          if (reduce_done.exists())
            reducer.record_postcondition(reduce_done);
          if ((reduce_expr != NULL) && !reduce_expr->is_empty())
            reducer.record_expression(reduce_expr);
        }
      }
      // Then traverse our children
      if (to_traverse.empty())
        return;
      RegionTreeForest *context = reducer.dst->logical_node->context;
      for (std::vector<CompositeNode*>::const_iterator it = 
            to_traverse.begin(); it != to_traverse.end(); it++)
      {
        IndexSpaceExpression *needed_child = 
          context->intersect_index_spaces(needed_expr, 
              (*it)->logical_node->get_index_space_expression());
        if (needed_child->is_empty())
          continue;
        (*it)->issue_composite_reductions_single(reducer, needed_child,
                  (*it)->logical_node, pred_guard, src_version_tracker);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeBase::issue_update_copies(
                                            DeferredCopier &copier,
                                            RegionTreeNode *logical_node,
                                            FieldMask copy_mask,
                                            VersionTracker *src_version_tracker,
                                            PredEvent predicate_guard,
              const LegionMap<LogicalView*,FieldMask>::aligned &source_views,
                                            const WriteMasks &previous_writes,
                                            WriteSet &performed_writes)
    //--------------------------------------------------------------------------
    {
      MaterializedView *dst = copier.dst;
      const TraversalInfo &info = *copier.info;
#ifdef DEBUG_LEGION
      // The previous writes data structure should be field unique,
      // if it's not then we are going to have big problems
      FieldMask previous_writes_mask;
      for (WriteMasks::const_iterator it = previous_writes.begin();
            it != previous_writes.end(); it++)
      {
        assert(previous_writes_mask * it->second);
        previous_writes_mask |= it->second;
      }
#endif
      // In some cases we might already be done
      if (!copy_mask)
        return;
      RegionTreeForest *context = dst->logical_node->context;
      IndexSpaceExpression *dst_is = 
        dst->logical_node->get_index_space_expression();
      IndexSpaceExpression *local_is = 
        logical_node->get_index_space_expression(); 
      IndexSpaceExpression *intersect_is = 
        (dst_is->expr_id == local_is->expr_id) ? local_is : 
        context->intersect_index_spaces(dst_is, local_is);
      // First check to see if the target is already valid
      if (copier.across_helper == NULL)
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
            FieldMask overlap = copy_mask & it->second;
            copy_mask -= overlap;
            // Find out if we had any prior writes that we need to record
            for (WriteMasks::const_iterator pit = previous_writes.begin();
                  pit != previous_writes.end(); pit++)
            {
              const FieldMask prev_overlap = overlap & pit->second;
              if (!prev_overlap)
                continue;
              // Construct the expression, intersect then subtract
              IndexSpaceExpression *expr = 
                context->subtract_index_spaces(intersect_is, pit->first);
              // Record this as previously valid
              copier.record_previously_valid(expr, prev_overlap);
              WriteMasks::iterator finder = performed_writes.find(expr);
              if (finder == performed_writes.end())
                performed_writes.insert(expr, prev_overlap);
              else
                finder.merge(prev_overlap);
              overlap -= prev_overlap;
              if (!overlap)
                break;
            }
            if (!!overlap)
            {
              // Record this as previously valid
              copier.record_previously_valid(intersect_is, overlap);
              // No prior writes so we can just record the overlap
              WriteMasks::iterator finder = performed_writes.find(intersect_is);
              if (finder == performed_writes.end())
                performed_writes.insert(intersect_is, overlap);
              else
                finder.merge(overlap);
            }
            if (!copy_mask)
              return;
            break;
          }
        }
      }
      // Sort the instances by preferences
      LegionMap<MaterializedView*,FieldMask>::aligned src_instances;
      LegionMap<DeferredView*,FieldMask>::aligned deferred_instances;
      dst->logical_node->sort_copy_instances(info, dst, copy_mask,
                    source_views, src_instances, deferred_instances);
      if (!src_instances.empty())
      {
        // This has all our destination preconditions
        // Only issue copies from fields which have values
        FieldMask actual_copy_mask;
        LegionMap<ApEvent,FieldMask>::aligned copy_preconditions;
        const AddressSpaceID local_space = context->runtime->address_space;
        // We're going to need this no matter what we do
        IndexSpaceExpression *user_expr = 
              logical_node->context->intersect_index_spaces(
                  logical_node->get_index_space_expression(),
                  copier.dst->logical_node->get_index_space_expression());
        for (LegionMap<MaterializedView*,FieldMask>::aligned::const_iterator 
              it = src_instances.begin(); it != src_instances.end(); it++)
        {
                    
          if (!previous_writes.empty())
          {
            // We need to find all the potential write masks that might mask
            // these fields so we can accurately find any interfering updates  
            FieldMask remaining = it->second;
            for (WriteMasks::const_iterator mit = previous_writes.begin();
                  mit != previous_writes.end(); mit++)
            {
              const FieldMask overlap = remaining & mit->second;
              if (!overlap)
                continue;
              // Subtract off the write mask
              IndexSpaceExpression *masked_user_expr = 
                logical_node->context->subtract_index_spaces(user_expr, 
                                                             mit->first);
              it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                               true/*single copy*/,
                                               false/*restrict out*/,
                                               overlap, masked_user_expr,
                                               src_version_tracker,
                                               info.op->get_unique_op_id(),
                                               info.index, local_space, 
                                               copy_preconditions,
                                               info.map_applied_events, info);
              remaining -= overlap;
              if (!remaining)
                break;
            }
            if (!!remaining)
              it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                               true/*single copy*/,
                                               false/*restrict out*/,
                                               remaining, user_expr,
                                               src_version_tracker,
                                               info.op->get_unique_op_id(),
                                               info.index, local_space, 
                                               copy_preconditions,
                                               info.map_applied_events, info);
          }
          else
            // This just requires the basic intersection
            it->first->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                               true/*single copy*/,
                                               false/*restrict out*/,
                                               it->second, user_expr,
                                               src_version_tracker,
                                               info.op->get_unique_op_id(),
                                               info.index, local_space, 
                                               copy_preconditions,
                                               info.map_applied_events, info);
          actual_copy_mask |= it->second;
        }
        copier.merge_destination_preconditions(actual_copy_mask, 
                                               copy_preconditions);
        // Issue the grouped copies and put the results in the postconditions
        dst->logical_node->issue_grouped_copies(*copier.info, dst,
            false/*restrict out*/, predicate_guard, copy_preconditions, 
            actual_copy_mask, src_instances, src_version_tracker,
            copier.copy_postconditions, copier.across_helper, 
            logical_node, &previous_writes, &performed_writes);
      }
      if (!deferred_instances.empty())
      {
        // These ones are easy, we just get to recurse, no need to do any
        // write combining since we know that we didn't actually do any
        // writes for these fields locally or we wouldn't even be traversing
        // the composite views in the first place
        for (LegionMap<DeferredView*,FieldMask>::aligned::const_iterator it =
              deferred_instances.begin(); it != deferred_instances.end(); it++)
          it->first->issue_deferred_copies(copier, it->second, 
                previous_writes, performed_writes, predicate_guard);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeBase::combine_writes(WriteMasks &write_masks,
                                      DeferredCopier &copier, bool prune_global)
    //--------------------------------------------------------------------------
    {
      MaterializedView *dst = copier.dst;
      FieldMask &global_copy_mask = copier.deferred_copy_mask;
      LegionList<FieldSet<IndexSpaceExpression*> >::aligned write_sets;
      // Compute the write sets for different fields
      // We use an empty universe mask since we don't care about fields
      // that we don't find in the input set
      write_masks.compute_field_sets(FieldMask(), write_sets);
      // Clear out the write masks set since we're rebuilding it
      write_masks.clear();
      IndexSpaceExpression *dst_is = 
        dst->logical_node->get_index_space_expression();
      RegionTreeForest *context = dst->logical_node->context;
      // Compute the unions of all the writes
      for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::const_iterator
            it = write_sets.begin(); it != write_sets.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->elements.empty());
#endif
        IndexSpaceExpression *union_is = NULL;
        if (it->elements.size() > 1)
          union_is = context->union_index_spaces(it->elements); 
        else
          union_is = *(it->elements.begin()); 
        write_masks.insert(union_is, it->set_mask);
        // Compute the pending difference so we can do the check
        // to see if it is done writing to this index space or not
        // If it's done writing we can remove the fields from the
        // global copy mask, otherwise just keep the union_is
        if (prune_global)
        {
          IndexSpaceExpression *diff_is = 
            context->subtract_index_spaces(dst_is, union_is);
          if (diff_is->is_empty())
            global_copy_mask -= it->set_mask;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ IndexSpaceExpression* CompositeBase::issue_update_copies_single(
                                      DeferredSingleCopier &copier,
                                      RegionTreeNode *logical_node,
                                      VersionTracker *src_version_tracker,
                                      PredEvent pred_guard,
                                      std::vector<LogicalView*> &source_views,
                                      IndexSpaceExpression *write_mask)
    //--------------------------------------------------------------------------
    {
      MaterializedView *dst = copier.dst;
      const TraversalInfo &info = *copier.info;
      RegionTreeForest *context = dst->logical_node->context;
      IndexSpaceExpression *dst_is = 
        dst->logical_node->get_index_space_expression();
      IndexSpaceExpression *local_is = 
        logical_node->get_index_space_expression(); 
      IndexSpaceExpression *intersect_is = 
        (dst_is->expr_id == local_is->expr_id) ? local_is : 
        context->intersect_index_spaces(dst_is, local_is);
      MaterializedView *src_instance = NULL;
      DeferredView *deferred_instance = NULL;
      if (dst->logical_node->sort_copy_instances_single(info, dst,
            copier.copy_mask, source_views, src_instance, deferred_instance))
      {
        // If we get here then the destination is already valid
        // Construct the write expression, intersect then subtract
        if (write_mask != NULL)
        {
          IndexSpaceExpression *result_expr = 
            context->subtract_index_spaces(intersect_is, write_mask);
          copier.record_previously_valid(result_expr);
          return result_expr;
        }
        else
        {
          copier.record_previously_valid(intersect_is);
          return intersect_is;
        }
      }
      // Easy case if we are just copying from one or more instances
      else if (src_instance != NULL)
      {
        IndexSpaceExpression *copy_expr = intersect_is;
        if (write_mask != NULL)
          copy_expr = context->subtract_index_spaces(copy_expr, write_mask);
        // Check to see if they are the same instance, if they are
        // then we don't actually have to do the copy ourself
        if ((src_instance != dst) && (src_instance->manager->get_instance() !=
              dst->manager->get_instance()))
        {
          LegionMap<ApEvent,FieldMask>::aligned temp_preconditions;
          src_instance->find_copy_preconditions(0/*redop*/, true/*reading*/,
                                                true/*single copy*/,
                                                false/*restrict out*/,
                                                copier.copy_mask, copy_expr,
                                                src_version_tracker,
                                                info.op->get_unique_op_id(),
                                                info.index,
                                                context->runtime->address_space,
                                                temp_preconditions,
                                                info.map_applied_events, info);
          std::set<ApEvent> copy_preconditions;
          for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
               temp_preconditions.begin(); it != temp_preconditions.end(); it++)
            copy_preconditions.insert(it->first);
          copier.merge_destination_preconditions(copy_preconditions);
          // Issue the copy
          ApEvent copy_pre = Runtime::merge_events(copier.info, 
                                                   copy_preconditions);
          ApEvent copy_post = dst->logical_node->issue_single_copy(*copier.info,
              dst, false/*restrict out*/, pred_guard, copy_pre,
              copier.copy_mask, src_instance, src_version_tracker,
              copier.across_helper, logical_node, write_mask);
          if (copy_post.exists())
            copier.record_postcondition(copy_post);
        }
        else // Have to record any previously valid expressions
          copier.record_previously_valid(copy_expr);
        return copy_expr;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(deferred_instance != NULL); // better have one of these
#endif
        IndexSpaceExpression *performed_write = NULL;
        deferred_instance->issue_deferred_copies_single(copier, 
                              write_mask, performed_write, pred_guard);
        return performed_write;
      }
    } 

    //--------------------------------------------------------------------------
    /*static*/ bool CompositeBase::test_done(DeferredSingleCopier &copier,
                     IndexSpaceExpression *write1, IndexSpaceExpression *write2)
    //--------------------------------------------------------------------------
    {
      MaterializedView *dst = copier.dst;
      IndexSpaceExpression *dst_is = 
        dst->logical_node->get_index_space_expression();
      RegionTreeForest *context = dst->logical_node->context;
      IndexSpaceExpression *diff = context->subtract_index_spaces(
          dst_is, (write2 == NULL) ? write1 : 
            context->union_index_spaces(write1, write2));
      return diff->is_empty();
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

    //--------------------------------------------------------------------------
    void CompositeBase::print_view_state(const FieldMask &capture_mask,
                                         TreeStateLogger* logger,
                                         int current_nesting,
                                         int max_nesting)
    //--------------------------------------------------------------------------
    {
      {
        char *mask_string = dirty_mask.to_string();
        logger->log("Dirty Mask: %s", mask_string);
        free(mask_string);
      }
      {
        char *mask_string = reduction_mask.to_string();
        logger->log("Reduction Mask: %s", mask_string);
        free(mask_string);
      }
      {
        unsigned num_valid = 0;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
              valid_views.begin(); it != valid_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        if (num_valid > 0)
        {
          logger->log("Valid Instances (%d)", num_valid);
          logger->down();
          for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
                valid_views.begin(); it != valid_views.end(); it++)
          {
            FieldMask overlap = it->second & capture_mask;
            if (!overlap)
              continue;
            if (it->first->is_deferred_view())
            {
              if (it->first->is_composite_view())
              {
                CompositeView *composite_view = it->first->as_composite_view();
                if (composite_view != NULL)
                {
                  logger->log("=== Composite Instance ===");
                  logger->down();
                  // We go only two levels down into the nested composite views
                  composite_view->print_view_state(capture_mask, logger, 0, 2);
                  logger->up();
                  logger->log("==========================");
                }
              }
              continue;
            }
            assert(it->first->as_instance_view()->is_materialized_view());
            MaterializedView *current = 
              it->first->as_instance_view()->as_materialized_view();
            char *valid_mask = overlap.to_string();
            logger->log("Instance " IDFMT "   Memory " IDFMT "   Mask %s",
                        current->manager->get_instance().id, 
                        current->manager->get_memory().id, valid_mask);
            free(valid_mask);
          }
          logger->up();
        }
      }
      {
        unsigned num_valid = 0;
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              reduction_views.begin(); it != 
              reduction_views.end(); it++)
        {
          if (it->second * capture_mask)
            continue;
          num_valid++;
        }
        if (num_valid > 0)
        {
          logger->log("Valid Reduction Instances (%d)", num_valid);
          logger->down();
          for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
                reduction_views.begin(); it !=
                reduction_views.end(); it++)
          {
            FieldMask overlap = it->second & capture_mask;
            if (!overlap)
              continue;
            char *valid_mask = overlap.to_string();
            logger->log("Reduction Instance " IDFMT "   Memory " IDFMT
                        "  Mask %s",
                        it->first->manager->get_instance().id, 
                        it->first->manager->get_memory().id, valid_mask);
            free(valid_mask);
          }
          logger->up();
        }
      }
      if (!children.empty())
      {
        logger->log("Children (%lu):", children.size());
        logger->down();
        for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it =
              children.begin(); it !=
              children.end(); it++)
        {
          it->first->logical_node->print_context_header(logger);
          {
            char *mask_string = it->second.to_string();
            logger->log("Field Mask: %s", mask_string);
            free(mask_string);
          }
          logger->down();
          it->first->print_view_state(
                              capture_mask, logger, current_nesting, max_nesting);
          logger->up();
        }
        logger->up();
      }
    }

    /////////////////////////////////////////////////////////////
    // CompositeView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(RegionTreeForest *ctx, DistributedID did,
                          AddressSpaceID owner_proc, RegionTreeNode *node,
                          DeferredVersionInfo *info, CompositeViewSummary &sum,
                          InnerContext *context, bool register_now, 
                          ReplicationID repl/*=0*/,
                          RtBarrier invalid_bar/*= NO_RT_BARRIER*/,
                          ShardID origin/*=0*/) 
      : DeferredView(ctx, encode_composite_did(did), owner_proc, 
                     node, register_now), 
        CompositeBase(view_lock, invalid_bar.exists()),
        version_info(info), summary(sum), owner_context(context), repl_id(repl),
        shard_invalid_barrier(invalid_bar), origin_shard(origin)
#ifdef DISABLE_CVOPT
        , packed_shard(NULL)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_context != NULL);
#endif
      // Add our references
      version_info->add_reference();
      owner_context->add_reference();
      // See if we have any partial writes
      if (!summary.partial_writes.empty())
      {
        // Add our expression references
        for (WriteSet::const_iterator it = summary.partial_writes.begin();
              it != summary.partial_writes.end(); it++)
          it->first->add_expression_reference();
      }
      // If we are the owner in a control replicated context then we add a 
      // GC reference that will be removed once all the shards are done 
      // with the view, no mutator since we know we're the owner
      if (shard_invalid_barrier.exists() && is_owner())
        add_base_gc_ref(COMPOSITE_SHARD_REF);
#ifdef LEGION_GC
      log_garbage.info("GC Composite View %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    CompositeView::CompositeView(const CompositeView &rhs)
      : DeferredView(NULL, 0, 0, NULL, false), CompositeBase(view_lock, false),
        version_info(NULL), 
        summary(*const_cast<CompositeViewSummary*>(&rhs.summary)),
        owner_context(NULL), repl_id(0), 
        shard_invalid_barrier(RtBarrier::NO_RT_BARRIER), origin_shard(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CompositeView::~CompositeView(void)
    //--------------------------------------------------------------------------
    {
      // Remove our references and delete if necessary
      if (version_info->remove_reference())
        delete version_info;
      // Remove the reference on our context
      if (owner_context->remove_reference())
        delete owner_context;
#ifdef DISABLE_CVOPT
      if (packed_shard != NULL)
        delete packed_shard;
#endif
      // Remove any resource references we still have
      if (!is_owner())
      {
        for (NestedViewMap::const_iterator it =
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
              children.begin(); it != children.end(); it++)
          delete it->first;
        for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
              valid_views.begin(); it != valid_views.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
        for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it =
              reduction_views.begin(); it != reduction_views.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
      }
#ifdef DEBUG_LEGION
      else
      {
        assert(nested_composite_views.empty());
        assert(children.empty());
        assert(valid_views.empty());
        assert(reduction_views.empty());
      }
#endif
      // Remove any expression references
      if (!summary.partial_writes.empty())
      {
        for (WriteSet::const_iterator it = summary.partial_writes.begin();
              it != summary.partial_writes.end(); it++)
          if (it->first->remove_expression_reference())
            delete it->first;
      }
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
                                        const NestedViewMap &replacements,
                              ReferenceMutator *mutator, InterCloseOp *op) const
    //--------------------------------------------------------------------------
    {
      // See if we need to compute a new partial write set 
      CompositeViewSummary clone_summary;
      clone_summary.complete_writes = summary.complete_writes & clone_mask;
      if (!summary.partial_writes.empty())
      {
        for (WriteSet::const_iterator it = summary.partial_writes.begin();
              it != summary.partial_writes.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          clone_summary.partial_writes.insert(it->first, overlap);
        }
      }
      // Make copies of any sharding summaries if necessary
      if (!summary.write_projections.empty() &&
          !(clone_mask * summary.write_projections.get_valid_mask()))
      {
        for (FieldMaskSet<ShardingSummary>::const_iterator it = 
              summary.write_projections.begin(); it != 
              summary.write_projections.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          clone_summary.write_projections.insert(
              new ShardingSummary(*(it->first)), overlap);
        }
      }
      if (!summary.reduce_projections.empty() &&
          !(clone_mask & summary.reduce_projections.get_valid_mask()))
      {
        for (FieldMaskSet<ShardingSummary>::const_iterator it = 
              summary.reduce_projections.begin(); it != 
              summary.reduce_projections.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          clone_summary.reduce_projections.insert(
              new ShardingSummary(*(it->first)), overlap);
        }
      }
      CompositeView *result = owner_context->create_composite_view(logical_node,
                                version_info, op, true/*clone*/, clone_summary);
      // Clone the children
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        FieldMask overlap = it->second & clone_mask;
        if (!overlap)
          continue;
        it->first->clone(result, overlap, mutator);
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
          result->record_valid_view(it->first, overlap, mutator);
        }
      }
      // Can just insert the replacements directly
      for (LegionMap<CompositeView*,FieldMask>::aligned::const_iterator it =
            replacements.begin(); it != replacements.end(); it++)
        result->record_valid_view(it->first, it->second, mutator);
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
          result->record_reduction_view(it->first, overlap, mutator);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_owner_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        it->first->release_gc_references(mutator);
        delete it->first;
      }
      children.clear();
      // Only GC references for nested views here
      for (NestedViewMap::const_iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
        if (it->first->remove_nested_gc_ref(did, mutator))
          delete it->first;
      nested_composite_views.clear();
      // Remove GC references from deferred views and valid references
      // from normal instance views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->is_deferred_view())
        {
          if (it->first->remove_nested_gc_ref(did, mutator))
            delete it->first;
        }
        else
        {
          if (it->first->remove_nested_valid_ref(did, mutator))
            delete it->first;
        }
      }
      valid_views.clear();
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
        if (it->first->remove_nested_valid_ref(did, mutator))
          delete it->first;
      reduction_views.clear();
      // Unregister the view when we can be safely collected
      if (shard_invalid_barrier.exists() && is_owner())
      {
#ifdef DEBUG_LEGION
        ReplicateContext *ctx = dynamic_cast<ReplicateContext*>(owner_context);
        assert(ctx != NULL);
#else
        ReplicateContext *ctx = static_cast<ReplicateContext*>(owner_context);
#endif
        ctx->unregister_composite_view(this, shard_invalid_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::notify_owner_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      if (shard_invalid_barrier.exists())
      {
        // Do the arrival on our barrier and then launch a task to
        // remove our reference once everyone is done with the view
        Runtime::phase_barrier_arrive(shard_invalid_barrier, 1/*count*/);
        // Launch the task to do the removal of the reference
        // when the shard invalid barrier has triggered
        DeferInvalidateArgs args;
        args.view = this;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                         shard_invalid_barrier);
      }
      // Tell the nested views that they can be potentially collected too
      for (NestedViewMap::const_iterator it =
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
      // Remove valid references from any other deferred views too
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (!it->first->is_deferred_view())
          continue;
        it->first->remove_nested_valid_ref(did, mutator);
      }
      // Also tell the same to the composite nodes so they 
      // can inform their version states
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
        it->first->release_valid_references(mutator);
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
        rez.serialize(shard_invalid_barrier);
        if (shard_invalid_barrier.exists())
        {
          rez.serialize(repl_id);
          rez.serialize(origin_shard);
        }
        summary.pack(rez, target);
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
    void CompositeView::prune(
                   const WriteMasks &partial_write_masks, FieldMask &valid_mask,
                              NestedViewMap &replacements, unsigned prune_depth,
                              ReferenceMutator *mutator, InterCloseOp *op)
    //--------------------------------------------------------------------------
    {
      if (prune_depth >= LEGION_PRUNE_DEPTH_WARNING)
        REPORT_LEGION_WARNING(LEGION_WARNING_PRUNE_DEPTH_EXCEEDED,
                        "WARNING: Composite View Tree has depth %d which "
                        "is larger than LEGION_PRUNE_DEPTH_WARNING of %d. "
                        "Please report this use case to the Legion developers "
                        "mailing list as it is a highly unusual case or could "
                        "be a runtime performance bug.", prune_depth,
                        LEGION_PRUNE_DEPTH_WARNING)
      // First check to see if we can be pruned      
      FieldMask dominated; 
      RegionTreeForest *forest = logical_node->context;
      if (!summary.partial_writes.empty() && 
          !(summary.partial_writes.get_valid_mask() * valid_mask))
      {
        for (WriteSet::const_iterator pit = summary.partial_writes.begin();
              pit != summary.partial_writes.end(); pit++)
        {
          FieldMask remaining = pit->second & valid_mask;
          if (!remaining)
            continue;
          for (WriteMasks::const_iterator it = partial_write_masks.begin();
                it != partial_write_masks.end(); it++)
          {
            const FieldMask overlap = it->second & remaining;
            if (!overlap)
              continue;
            IndexSpaceExpression *diff_expr = 
              forest->subtract_index_spaces(pit->first, it->first);
            if (diff_expr->is_empty())
              dominated |= overlap;
            remaining -= overlap;
            if (!remaining)
              break;
          }
        }
      }
      if (!!dominated)
      {
        // If we had any dominated fields then we try to prune our
        // deferred valid views and put the results directly into 
        // the replacements
        for (NestedViewMap::const_iterator it = 
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
        {
          FieldMask overlap = it->second & dominated;
          if (!overlap)
            continue;
          it->first->prune(partial_write_masks, overlap, replacements, 
                           prune_depth+1, mutator, op);
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
      NestedViewMap local_replacements;
      for (NestedViewMap::iterator it = 
            nested_composite_views.begin(); it != 
            nested_composite_views.end(); it++)
      {
        const FieldMask overlap = it->second & valid_mask;
        if (!overlap)
          continue;
        FieldMask still_valid = overlap;
        it->first->prune(partial_write_masks, still_valid, 
                         local_replacements, prune_depth+1, mutator, op);
        // See if any fields were pruned, if so they are changed
        FieldMask changed = overlap - still_valid;
        if (!!changed)
          changed_mask |= changed;
      }
      if (!local_replacements.empty())
      {
        for (NestedViewMap::const_iterator it =
              local_replacements.begin(); it != local_replacements.end(); it++)
          changed_mask |= it->second;
      }
      if (!!changed_mask)
      {
        CompositeView *view = clone(changed_mask,local_replacements,mutator,op);
        view->finalize_capture(false/*need prune*/, mutator, op);
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
      if (copy_mask.pop_count() == 1)
      {
        DeferredSingleCopier copier(&info, owner_context, dst, copy_mask, 
                                    restrict_info, restrict_out);
        IndexSpaceExpression *write_performed = NULL;
        issue_deferred_copies_single(copier, NULL/*write mask*/,
                                     write_performed, PredEvent::NO_PRED_EVENT);
        copier.finalize(this);
      }
      else
      {
        DeferredCopier copier(&info, owner_context, dst, copy_mask, 
                              restrict_info, restrict_out);
        WriteMasks write_masks;
        WriteSet performed_writes;
        issue_deferred_copies(copier, copy_mask, write_masks, 
                              performed_writes, PredEvent::NO_PRED_EVENT);
        copier.finalize(this);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_deferred_copies(DeferredCopier &copier,
                                              const FieldMask &local_copy_mask,
                                              const WriteMasks &write_masks,
                                              WriteSet &performed_writes,
                                              PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      // Each composite view depth is its own reduction epoch
      copier.begin_reduction_epoch();
      issue_composite_updates(copier, logical_node, local_copy_mask,
                              this, pred_guard, write_masks, performed_writes);
      copier.end_reduction_epoch();
    }

    //--------------------------------------------------------------------------
    bool CompositeView::issue_deferred_copies_single(
                                            DeferredSingleCopier &copier,
                                            IndexSpaceExpression *write_mask,
                                            IndexSpaceExpression *&write_perf,
                                            PredEvent pred_guard)      
    //--------------------------------------------------------------------------
    {
      copier.begin_reduction_epoch();
      bool result = issue_composite_updates_single(copier, logical_node, this,
                                         pred_guard, write_mask, write_perf);
      copier.end_reduction_epoch();
      return result;
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
#ifndef DISABLE_CVOPT
      CompositeNode *capture_node = capture_above(node, still_needed);
#else
      CompositeNode *capture_node = capture_above(node, still_needed, node);
#endif
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
    void CompositeView::pack_writing_version_numbers(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      // Should never be writing to a composite view
      assert(false);
    }

    //--------------------------------------------------------------------------
    void CompositeView::pack_upper_bound_node(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      version_info->pack_upper_bound_node(rez);
    }

#ifndef DISABLE_CVOPT
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
        perform_ready_check(needed_fields);
        return find_child_node(node);
      }
      // Otherwise continue up the tree 
      CompositeNode *parent_node = capture_above(parent, needed_fields);
      // Now make sure that this node has captured for all subregions
      // Do this on the way back down to know that the parent node is good
      parent_node->perform_ready_check(needed_fields);
      return parent_node->find_child_node(node);
    }
#else
    //--------------------------------------------------------------------------
    CompositeNode* CompositeView::capture_above(RegionTreeNode *node,
                                                const FieldMask &needed_fields,
                                                RegionTreeNode *target)
    //--------------------------------------------------------------------------
    {
      // Recurse up the tree to get the parent version state
      RegionTreeNode *parent = node->get_parent();
#ifdef DEBUG_LEGION
      assert(parent != NULL);
#endif
      if (parent == logical_node)
      {
        perform_ready_check(needed_fields, target);
        return find_child_node(node);
      }
      // Otherwise continue up the tree 
      CompositeNode *parent_node = capture_above(parent, needed_fields, target);
      // Now make sure that this node has captured for all subregions
      // Do this on the way back down to know that the parent node is good
      parent_node->perform_ready_check(needed_fields, target);
      return parent_node->find_child_node(node);
    }
#endif

    //--------------------------------------------------------------------------
    void CompositeView::unpack_composite_view_response(Deserializer &derez,
                                                       Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      size_t num_children;
      derez.deserialize(num_children);
      DistributedID owner_did = get_owner_did();
      std::set<RtEvent> ready_events;
      {
        AutoLock v_lock(view_lock);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          CompositeNode *child = CompositeNode::unpack_composite_node(derez,
                this, runtime, owner_did, ready_events, children, is_owner());
          FieldMask child_mask;
          derez.deserialize(child_mask);
          // Have to do a merge of field masks here
          children[child] |= child_mask;
        }
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void CompositeView::issue_shard_updates(DeferredCopier &copier,
                                            RegionTreeNode *logical_node,
                                            const FieldMask &local_copy_mask,
                                            PredEvent pred_guard,
                                            const WriteMasks &write_masks,
                                                  WriteMasks &performed_writes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_invalid_barrier.exists());
#endif
#ifndef DISABLE_CVOPT
      RegionTreeForest *context = logical_node->context;
      LegionMap<ShardID,WriteMasks>::aligned needed_shards, reduction_shards; 
      IndexSpaceExpression *dst_expr = context->intersect_index_spaces(
          logical_node->get_index_space_expression(),
          copier.dst->logical_node->get_index_space_expression());
#ifdef DEBUG_LEGION
      assert(!dst_expr->is_empty());
#endif
      find_needed_shards(local_copy_mask, origin_shard, 
          dst_expr, write_masks, needed_shards, reduction_shards);
      for (std::map<ShardID,WriteMasks>::iterator it = 
            needed_shards.begin(); it != needed_shards.end(); it++)
      {
        Serializer rez;
        rez.serialize(repl_id);
        rez.serialize(it->first);
        rez.serialize<RtEvent>(shard_invalid_barrier);
        rez.serialize(pred_guard);
        rez.serialize<bool>(false); // single
#ifdef DEBUG_LEGION
        assert(!it->second.empty());
#endif
        // Before doing anything, write combine so we get
        // exactly one expression for each field
        LegionList<FieldSet<IndexSpaceExpression*> >::aligned write_sets;
        // Compute the common field sets and combine them, don't
        // give a universal mask since we only care about the
        // fields we have.
        it->second.compute_field_sets(FieldMask(), write_sets);
        it->second.clear();
        FieldMask shard_copy_mask;
        for (LegionList<FieldSet<IndexSpaceExpression*> >::aligned::
              const_iterator wit = write_sets.begin(); 
              wit != write_sets.end(); wit++)
        {
#ifdef DEBUG_LEGION
          assert(!wit->elements.empty());
#endif
          IndexSpaceExpression *union_is = (wit->elements.size() > 1) ?
            context->union_index_spaces(wit->elements) :
            *(wit->elements.begin());
          it->second.insert(union_is, wit->set_mask);
          shard_copy_mask |= wit->set_mask;
        }
#ifdef DEBUG_LEGION
        assert(!it->second.empty());
        assert(!(shard_copy_mask - local_copy_mask));
#endif
        rez.serialize(shard_copy_mask);
        copier.pack_copier(rez, shard_copy_mask);
        rez.serialize<size_t>(it->second.size());
        for (WriteMasks::const_iterator wit = it->second.begin();
              wit != it->second.end(); wit++)
        {
          // The expression we actually want to pack are the
          // write masks and not the writes that are going to
          // be performed so do the difference
          IndexSpaceExpression *write_mask = 
            context->subtract_index_spaces(dst_expr,wit->first);
          write_mask->pack_expression(rez,
              owner_context->find_shard_space(it->first));
          // We need a separate completion event for each field
          // in order to avoid unnecessary serialization, we don't
          // know which fields are going to be grouped together or
          // not yet
          const size_t pop_count = wit->second.pop_count();
#ifdef DEBUG_LEGION
          assert(pop_count > 0);
#endif
          rez.serialize(pop_count);
          int index = wit->second.find_first_set(); 
          for (unsigned idx = 0; idx < pop_count; idx++)
          {
            if (idx > 0)
              index = wit->second.find_next_set(index+1);
#ifdef DEBUG_LEGION
            assert(index >= 0);
#endif
            rez.serialize<unsigned>(index);
            ApUserEvent field_done = Runtime::create_ap_user_event();
            rez.serialize(field_done);
#ifdef DEBUG_LEGION
            assert(copier.copy_postconditions.find(field_done) ==
                    copier.copy_postconditions.end());
#endif
            // This puts the event in the set and sets the bit for the field
            copier.copy_postconditions[field_done].set_bit(index);
            // Find the sharded write tracker we'll use to tell about writes
            // Don't need a write-tracker though if this is a copy-across
            if (copier.across_helper == NULL)
              copier.pack_sharded_write_tracker(index, rez);
          }
        }
        // Now we can send the copy message to the remote node
        owner_context->send_composite_view_shard_copy_request(it->first, rez);
      }
      // Buffer up any reduction shards we might have
      if (!reduction_shards.empty())
        copier.buffer_reduction_shards(pred_guard, repl_id,
            shard_invalid_barrier, reduction_shards);
#endif
    }

    //--------------------------------------------------------------------------
    IndexSpaceExpression* CompositeView::issue_shard_updates_single(
                                               DeferredSingleCopier &copier,
                                               RegionTreeNode *logical_node,
                                               PredEvent pred_guard,
                                               IndexSpaceExpression *write_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_invalid_barrier.exists());
#endif
#ifndef DISABLE_CVOPT
      RegionTreeForest *context = logical_node->context;
      IndexSpaceExpression *target_expr = context->intersect_index_spaces(
        logical_node->get_index_space_expression(), 
        copier.dst->logical_node->get_index_space_expression());
#ifdef DEBUG_LEGION
      assert(!target_expr->is_empty());
#endif
      if (write_mask != NULL)
      {
        target_expr = context->subtract_index_spaces(target_expr, write_mask);
        if (target_expr->is_empty())
          return NULL;
      }
      std::map<ShardID,IndexSpaceExpression*> needed_shards, reduction_shards;
      find_needed_shards_single(copier.field_index, origin_shard,
                                target_expr, needed_shards, reduction_shards);
      std::set<IndexSpaceExpression*> shard_writes;
      for (std::map<ShardID,IndexSpaceExpression*>::const_iterator it = 
            needed_shards.begin(); it != needed_shards.end(); it++)
      {
        Serializer rez;
        rez.serialize(repl_id);
        rez.serialize(it->first);
        rez.serialize<RtEvent>(shard_invalid_barrier);
        rez.serialize(pred_guard);
        rez.serialize<bool>(true); // single
        rez.serialize<unsigned>(copier.field_index);
        copier.pack_copier(rez);
        // The expression that we actually want to pack is the mask
        // which is the difference between the destination and the
        // write that we are going to perform
        IndexSpaceExpression *mask = 
         logical_node->context->subtract_index_spaces(target_expr,it->second);
        mask->pack_expression(rez,
            owner_context->find_shard_space(it->first));
        ApUserEvent field_done = Runtime::create_ap_user_event();
        rez.serialize(field_done);
#ifdef DEBUG_LEGION
        assert(copier.copy_postconditions.find(field_done) ==
                copier.copy_postconditions.end());
#endif
        copier.record_postcondition(field_done);
        // Pack the sharded write tracker if this is not a copy-across
        // Don't need one for copy-across since we know we always write it all
        if (copier.across_helper == NULL)
          copier.pack_sharded_write_tracker(rez);
        // Now we can send the copy message to the remote node
        owner_context->send_composite_view_shard_copy_request(it->first, rez);
      }
      if (!reduction_shards.empty())
        copier.buffer_reduction_shards(pred_guard, repl_id,
            shard_invalid_barrier, reduction_shards);
      if (!shard_writes.empty())
        return context->union_index_spaces(shard_writes);
#endif
      return NULL;
    }

    //--------------------------------------------------------------------------
    InnerContext* CompositeView::get_owner_context(void) const
    //--------------------------------------------------------------------------
    {
      return owner_context;
    }

    //--------------------------------------------------------------------------
#ifndef DISABLE_CVOPT
    void CompositeView::perform_ready_check(FieldMask check_mask)
#else
    void CompositeView::perform_ready_check(FieldMask check_mask, 
                                            RegionTreeNode *target)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef DISABLE_CVOPT
      // See if we need to do a sharding test for control replication
      if (shard_invalid_barrier.exists() && (target != NULL))
      {
        // First check to see if we've already done a check for this target
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          LegionMap<RegionTreeNode*,FieldMask>::aligned::const_iterator finder =
            shard_checks.find(target);
          if (finder != shard_checks.end())
          {
            check_mask -= finder->second;
            if (!check_mask)
              return;
          }
        }
        // Compute the set of shards that we need locally 
        std::set<ShardID> needed_shards;
        find_needed_shards(check_mask, target, needed_shards);
        if (!needed_shards.empty())
        {
          std::set<RtEvent> wait_on;
          for (std::set<ShardID>::const_iterator it = needed_shards.begin();
                it != needed_shards.end(); it++)
          {
            // We can always skip the shard on which we were made
            if ((*it) == origin_shard)
              continue;
            RtUserEvent shard_ready;
            {
              AutoLock v_lock(view_lock);
              std::map<ShardID,RtEvent>::const_iterator finder = 
                requested_shards.find(*it);
              if (finder != requested_shards.end())
              {
                // Already sent the request
                wait_on.insert(finder->second);
                continue;
              }
              else
              {
                shard_ready = Runtime::create_rt_user_event();
                requested_shards[*it] = shard_ready;
              }
            }
            Serializer rez;
            rez.serialize(repl_id);
            rez.serialize(*it);
            rez.serialize<RtEvent>(shard_invalid_barrier);
            rez.serialize(shard_ready);
            rez.serialize(this);
            rez.serialize(context->runtime->address_space);
            owner_context->send_composite_view_shard_request(*it, rez);
            wait_on.insert(shard_ready);
          }
          // Wait for the result to be valid
          if (!wait_on.empty())
          {
            RtEvent wait_for = Runtime::merge_events(wait_on);
            if (wait_for.exists())
              wait_for.wait();
          }
        }
        // Now we can add this to the set of checks that we've performed
        AutoLock v_lock(view_lock);
        shard_checks[target] |= check_mask;
      }
#endif
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void CompositeView::find_needed_shards(const FieldMask &mask,ShardID origin,
                IndexSpaceExpression *target, const WriteMasks &write_masks,
                LegionMap<ShardID,WriteMasks>::aligned &needed_shards,
                LegionMap<ShardID,WriteMasks>::aligned &reduction_shards) const
    //--------------------------------------------------------------------------
    {
      // See if we have any write projections to test against first
      if (!summary.write_projections.empty())
      {
        const FieldMask overlap = mask & 
          summary.write_projections.get_valid_mask();
        if (!!overlap)
          find_interfering_shards(overlap, origin, target, write_masks, 
                                  summary.write_projections, needed_shards);
      }
      // Then see if we have any reduction projections
      if (!summary.reduce_projections.empty())
      {
        const FieldMask overlap = 
          mask & summary.reduce_projections.get_valid_mask();
        if (!!overlap)
          find_interfering_shards(overlap, origin, target, write_masks,
                                  summary.reduce_projections, reduction_shards);
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_needed_shards_single(const unsigned field_index,
        const ShardID origin_shard, IndexSpaceExpression *write_mask,
        std::map<ShardID,IndexSpaceExpression*> &needed_shards,
        std::map<ShardID,IndexSpaceExpression*> &reduction_shards) const
    //--------------------------------------------------------------------------
    {
      // See if we have any write projections first
      if (!summary.write_projections.empty() && 
          summary.write_projections.get_valid_mask().is_set(field_index))
        find_interfering_shards_single(field_index, origin_shard, write_mask,
                                       summary.write_projections,needed_shards);
      // Also check for any reduction projections
      if (!summary.reduce_projections.empty() &&
          summary.reduce_projections.get_valid_mask().is_set(field_index))
        find_interfering_shards_single(field_index, origin_shard, write_mask,
                               summary.reduce_projections, reduction_shards);
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_interfering_shards(FieldMask mask,
            const ShardID origin_shard, IndexSpaceExpression *target_expr,
            const WriteMasks &write_masks,
            const FieldMaskSet<ShardingSummary> &projections,
                  LegionMap<ShardID,WriteMasks>::aligned &needed_shards) const
    //--------------------------------------------------------------------------
    {
      // Iterate over the write masks and find ones that we care about
      for (WriteMasks::const_iterator wit = write_masks.begin();
            wit != write_masks.end(); wit++)
      {
        const FieldMask mask_overlap = wit->second & mask;
        if (!mask_overlap)
          continue;
        mask -= mask_overlap;
        IndexSpaceExpression *mask_expr = 
          logical_node->context->subtract_index_spaces(target_expr, wit->first);
        if (mask_expr->is_empty())
        {
          if (!mask)
            return;
          continue;
        }
        // We have an interesting write mask so find the interfering projections
        for (FieldMaskSet<ShardingSummary>::const_iterator
              pit = projections.begin(); pit != projections.end(); pit++)
        {
          // More intersection testing
          const FieldMask overlap = pit->second & mask_overlap;
          if (!overlap)
            continue;
          RegionTreeNode *node = pit->first->node;
          Domain full_space, shard_space;
          pit->first->domain->get_launch_space_domain(full_space);
          if (pit->first->sharding_domain != pit->first->domain)
            pit->first->sharding_domain->get_launch_space_domain(shard_space);
          else
            shard_space = full_space;
          // Invert the projection function to find the interfering points
          std::map<DomainPoint,IndexSpaceExpression*> interfering_points;
          pit->first->projection->find_interfering_points(node->context, node,
                                        pit->first->domain->handle, full_space,
                                        mask_expr, interfering_points);
          if (!interfering_points.empty())
          {
            for (std::map<DomainPoint,IndexSpaceExpression*>::const_iterator 
                  dit = interfering_points.begin(); 
                  dit != interfering_points.end(); dit++)
            {
              const ShardID shard = 
                pit->first->sharding->find_owner(dit->first, shard_space);
              // Skip our origin shard since we know about that
              if (shard == origin_shard)
                continue;
              // Now we have to insert it into all the right places
              std::map<ShardID,WriteMasks>::iterator shard_finder =
                needed_shards.find(shard);
              if (shard_finder != needed_shards.end())
              {
                WriteMasks::iterator finder = 
                  shard_finder->second.find(dit->second);
                if (finder != shard_finder->second.end())
                  finder.merge(overlap);
                else
                  shard_finder->second.insert(dit->second, overlap);
              }
              else
                needed_shards[shard].insert(dit->second, overlap);
            }
          }
        }
        // If we did all the fields we are done
        if (!mask)
          return;
      }
      // should only get here if we still have fields to do
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      // We have no interesting write mask
      for (FieldMaskSet<ShardingSummary>::const_iterator
            pit = projections.begin(); pit != projections.end(); pit++)
      {
        // More intersection testing
        const FieldMask overlap = pit->second & mask;
        if (!overlap)
          continue;
        Domain full_space, shard_space;
        pit->first->domain->get_launch_space_domain(full_space);
        if (pit->first->sharding_domain != pit->first->domain)
          pit->first->sharding_domain->get_launch_space_domain(shard_space);
        else
          shard_space = full_space;
        RegionTreeNode *node = pit->first->node;
        // Invert the projection function to find the interfering points
        std::map<DomainPoint,IndexSpaceExpression*> interfering_points;
        pit->first->projection->find_interfering_points(node->context, node,
                                    pit->first->domain->handle, full_space,
                                    target_expr, interfering_points);
        if (!interfering_points.empty())
        {
          for (std::map<DomainPoint,IndexSpaceExpression*>::const_iterator 
                dit = interfering_points.begin(); 
                dit != interfering_points.end(); dit++)
          {
            const ShardID shard = 
              pit->first->sharding->find_owner(dit->first, shard_space);
            // Skip our origin shard since we know about that
            if (shard == origin_shard)
              continue;
            // Now we have to insert it into all the right places
            std::map<ShardID,WriteMasks>::iterator shard_finder =
              needed_shards.find(shard);
            if (shard_finder != needed_shards.end())
            {
              WriteMasks::iterator finder = 
                shard_finder->second.find(dit->second);
              if (finder != shard_finder->second.end())
                finder.merge(overlap);
              else
                shard_finder->second.insert(dit->second, overlap);
            }
            else
              needed_shards[shard].insert(dit->second, overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::find_interfering_shards_single(const unsigned fidx,
        const ShardID origin_shard, IndexSpaceExpression *target_expr,
        const FieldMaskSet<ShardingSummary> &projections,
        std::map<ShardID,IndexSpaceExpression*> &needed_shards) const
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<ShardingSummary>::const_iterator
            pit = projections.begin(); pit != projections.end(); pit++)
      {
#ifdef DEBUG_LEGION
        // Should always have a sharding function
        assert(pit->first->sharding != NULL);
#endif
        if (!pit->second.is_set(fidx))
          continue;
        Domain full_space, shard_space;
        pit->first->domain->get_launch_space_domain(full_space);
        if (pit->first->sharding_domain != pit->first->domain)
          pit->first->sharding_domain->get_launch_space_domain(shard_space);
        else
          shard_space = full_space;
        RegionTreeNode *node = pit->first->node;
        // Invert the projection function to find the interfering points
        std::map<DomainPoint,IndexSpaceExpression*> interfering_points;
        pit->first->projection->find_interfering_points(node->context, node,
                                     pit->first->domain->handle, full_space, 
                                     target_expr, interfering_points);
        if (!interfering_points.empty())
        {
          for (std::map<DomainPoint,IndexSpaceExpression*>::const_iterator 
                dit = interfering_points.begin(); 
                dit != interfering_points.end(); dit++)
          {
            const ShardID shard = 
              pit->first->sharding->find_owner(dit->first, shard_space);
            // Skip our origin shard since we know about that
            if (shard == origin_shard)
              continue;
            std::map<ShardID,IndexSpaceExpression*>::iterator finder = 
              needed_shards.find(shard);
            if (finder != needed_shards.end())
              // Union the index space expressions together
              finder->second = node->context->union_index_spaces(
                                      finder->second, dit->second);
            else
              needed_shards[shard] = dit->second;
          }
        }
      }
    }
#else
    //--------------------------------------------------------------------------
    void CompositeView::find_needed_shards(FieldMask mask, 
                 RegionTreeNode *target, std::set<ShardID> &needed_shards) const
    //--------------------------------------------------------------------------
    {
      // See if we have any projections that we need to find other shards for
      if (!summary.write_projections.empty())
      {
        for (FieldMaskSet<ShardingSummary>::const_iterator
              pit = summary.write_projections.begin(); 
              pit != summary.write_projections.end(); pit++)
        {
#ifdef DEBUG_LEGION
          // Should always have a sharding function
          assert(pit->first->sharding != NULL);
#endif
          if (pit->second * mask)
            continue;
          Domain full_space;
          pit->first->domain->get_launch_space_domain(full_space);
          RegionTreeNode *node = pit->first->node;
          // Invert the projection function to find the interfering points
          std::set<DomainPoint> interfering_points;
          pit->first->projection->find_interfering_points(target->context, node,
            pit->first->domain->handle, full_space, target, interfering_points);
          if (!interfering_points.empty())
          {
            for (std::set<DomainPoint>::const_iterator dit = 
                  interfering_points.begin(); dit != 
                  interfering_points.end(); dit++)
            {
              ShardID shard = pit->first->sharding->find_owner(*dit, full_space);
              needed_shards.insert(shard);
            }
          }
        }
      }
      if (!summary.reduce_projections.empty())
      {
        for (FieldMaskSet<ShardingSummary>::const_iterator
              pit = summary.reduce_projections.begin(); 
              pit != summary.reduce_projections.end(); pit++)
        {
#ifdef DEBUG_LEGION
          // Should always have a sharding function
          assert(pit->first->sharding != NULL);
#endif
          if (pit->second * mask)
            continue;
          Domain full_space;
          pit->first->domain->get_launch_space_domain(full_space);
          RegionTreeNode *node = pit->first->node;
          // Invert the projection function to find the interfering points
          std::set<DomainPoint> interfering_points;
          pit->first->projection->find_interfering_points(target->context, node,
            pit->first->domain->handle, full_space, target, interfering_points);
          if (!interfering_points.empty())
          {
            for (std::set<DomainPoint>::const_iterator dit = 
                  interfering_points.begin(); dit != 
                  interfering_points.end(); dit++)
            {
              ShardID shard = pit->first->sharding->find_owner(*dit,full_space);
              needed_shards.insert(shard);
            }
          }
        }
      }
    }
#endif

    //--------------------------------------------------------------------------
    void CompositeView::find_valid_views(const FieldMask &update_mask,
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
      for (NestedViewMap::const_iterator it = 
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
      InnerContext *owner_context = runtime->find_context(owner_uid);
      RtBarrier shard_invalid_barrier;
      derez.deserialize(shard_invalid_barrier);
      ReplicationID repl_id = 0;
      ShardID origin_shard = 0;
      if (shard_invalid_barrier.exists())
      {
        derez.deserialize(repl_id);
        derez.deserialize(origin_shard);
      }
      CompositeViewSummary summary;
      summary.unpack(derez, runtime->forest, source, owner_context);
      // Make the composite view, but don't register it yet
      void *location;
      CompositeView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) CompositeView(runtime->forest, 
                                           did, owner, target_node, 
                                           version_info, summary,
                                           owner_context, false/*register now*/,
                                           repl_id, shard_invalid_barrier,
                                           origin_shard);
      else
        view = new CompositeView(runtime->forest, did, owner, 
                           target_node, version_info, summary, 
                           owner_context, false/*register now*/,
                           repl_id, shard_invalid_barrier, origin_shard);
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
    void CompositeView::record_valid_view(LogicalView *view, FieldMask mask,
                                          ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // If our composite view represents a complete write of this
      // logical region for any fields then there is no need to capture
      // this view for any of those fields
      if (!!summary.complete_writes)
      {
        mask -= summary.complete_writes;
        if (!mask)
          return;
      }
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
        {
          valid_views[mat_view] = mask;
          // Just need valid references for materialized views
          mat_view->add_nested_valid_ref(did, mutator);
        }
        else
          finder->second |= mask;
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
            NestedViewMap::iterator finder = 
              nested_composite_views.find(composite_view);
            if (finder == nested_composite_views.end())
            {
              nested_composite_views[composite_view] = mask;
              // Need gc and valid references for deferred things
              composite_view->add_nested_gc_ref(did, mutator);
              composite_view->add_nested_valid_ref(did, mutator);
            }
            else
              finder->second |= mask;
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
            {
              valid_views[composite_view] = mask;
              // Need gc and valid references for deferred things
              composite_view->add_nested_gc_ref(did, mutator);
              composite_view->add_nested_valid_ref(did, mutator);
            }
            else
              finder->second |= mask;
          }
        }
        else
        {
          // Just add it like normal
          LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
            valid_views.find(def_view);
          if (finder == valid_views.end())
          {
            valid_views[def_view] = mask;
            // Need gc and valid references for deferred things
            def_view->add_nested_gc_ref(did, mutator);
            def_view->add_nested_valid_ref(did, mutator);
          }
          else
            finder->second |= mask;
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
                               const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // For now just add it, we'll record references 
      // during finalize_capture
      LegionMap<ReductionView*,FieldMask>::aligned::iterator finder = 
        reduction_views.find(view);
      if (finder == reduction_views.end())
      {
        reduction_views[view] = mask;
        // just need valid references for reduction views
        view->add_nested_valid_ref(did, mutator);
      }
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
          it->first->record_version_state(state, mask, mutator);
          it->second |= mask;
          return;
        }
      }
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Didn't find it so make it
      CompositeNode *child = 
        new CompositeNode(child_node, this, did, true/*root owner*/); 
      child->record_version_state(state, mask, mutator);
      children[child] = mask;
    }

    //--------------------------------------------------------------------------
    void CompositeView::finalize_capture(bool need_prune, 
                                    ReferenceMutator *mutator, InterCloseOp *op)
    //--------------------------------------------------------------------------
    {
      // For the deferred views, we try to prune them 
      // based on our closed tree if they are the same we keep them 
      if (need_prune)
      {
        std::vector<CompositeView*> to_erase;
        NestedViewMap replacements;
        for (NestedViewMap::iterator it = 
              nested_composite_views.begin(); it != 
              nested_composite_views.end(); it++)
        {
#ifdef DEBUG_LEGION
          // Should be the same node in the region tree
          assert(logical_node == it->first->logical_node);
#endif
          it->first->prune(summary.partial_writes, it->second, replacements, 
                           0/*depth*/, mutator, op);
          if (!it->second)
            to_erase.push_back(it->first);
        }
        if (!to_erase.empty())
        {
          for (std::vector<CompositeView*>::const_iterator it = 
                to_erase.begin(); it != to_erase.end(); it++)
          {
            nested_composite_views.erase(*it);
            // Remove our nested valid and gc references
            (*it)->remove_nested_valid_ref(did);
            if ((*it)->remove_nested_gc_ref(did))
              delete (*it);
          }
        }
        if (!replacements.empty())
        {
          for (NestedViewMap::const_iterator it =
                replacements.begin(); it != replacements.end(); it++)
          {
            NestedViewMap::iterator finder =
              nested_composite_views.find(it->first);
            if (finder == nested_composite_views.end())
            {
              nested_composite_views.insert(*it);
              // We need both gc and valid references on nested things
              it->first->add_nested_gc_ref(did, mutator);
              it->first->add_nested_valid_ref(did, mutator);
            }
            else
              finder->second |= it->second;
          }
        }
      }
      // If we have a shard invalid barrier then this is the point
      // where it is safe to register ourselves as a composite view
      if (shard_invalid_barrier.exists())
      {
#ifdef DISABLE_CVOPT
        // Compute an initial packing of shard
#ifdef DEBUG_LEGION
        assert(packed_shard == NULL);
#endif
        packed_shard = new Serializer;
        packed_shard->serialize<size_t>(children.size());
        for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator 
              it = children.begin(); it != children.end(); it++)
        {
          it->first->pack_composite_node(*packed_shard);
          packed_shard->serialize(it->second);
        }
#endif
        // Then do our registration
#ifdef DEBUG_LEGION
        ReplicateContext *ctx = dynamic_cast<ReplicateContext*>(owner_context);
        assert(ctx != NULL);
#else
        ReplicateContext *ctx = static_cast<ReplicateContext*>(owner_context);
#endif
        ctx->register_composite_view(this, shard_invalid_barrier);
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
      for (NestedViewMap::const_iterator it = 
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

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_deferred_view_invalidation(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferInvalidateArgs *iargs = (const DeferInvalidateArgs*)args;
      if (iargs->view->remove_base_gc_ref(COMPOSITE_SHARD_REF))
        delete iargs->view;
    }

    //--------------------------------------------------------------------------
    void CompositeView::print_view_state(const FieldMask &capture_mask,
                                         TreeStateLogger* logger,
                                         int current_nesting,
                                         int max_nesting)
    //--------------------------------------------------------------------------
    {
      CompositeBase::print_view_state(
                            capture_mask, logger, current_nesting, max_nesting);
      int num_nested_views = 0;
      for (LegionMap<CompositeView*,FieldMask>::aligned::iterator it =
            nested_composite_views.begin(); it !=
            nested_composite_views.end(); it++)
      {
        if (it->second * capture_mask)
          continue;
        num_nested_views++;
      }
      if (num_nested_views > 0)
      {
        if (current_nesting < max_nesting)
        {
          logger->log("---- Nested Instances (Depth: %d) ----",
              current_nesting + 1);
          for (LegionMap<CompositeView*,FieldMask>::aligned::iterator it =
                nested_composite_views.begin(); it !=
                nested_composite_views.end(); it++)
          {
            {
              char *mask_string = it->second.to_string();
              logger->log("Field Mask: %s", mask_string);
              free(mask_string);
            }
            logger->down();
            it->first->print_view_state(
                          capture_mask, logger, current_nesting + 1, max_nesting);
            logger->up();
          }
          logger->log("--------------------------------------");
        }
        else
        {
          logger->log("--- Instances of Depth > %d Elided ---",
              current_nesting);
        }
      }
    }

#ifndef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void CompositeView::handle_sharding_copy_request(Deserializer &derez,
                          Runtime *rt, InnerContext *ctx, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      bool single;
      derez.deserialize(single);
      if (single)
      {
        unsigned field_index;
        derez.deserialize(field_index);
        FieldMask copy_mask;
        copy_mask.set_bit(field_index);
        RemoteDeferredSingleCopier *copier = 
          RemoteDeferredSingleCopier::unpack_copier(derez, rt, copy_mask, ctx);
        IndexSpaceExpression *write_mask = 
          IndexSpaceExpression::unpack_expression(derez, rt->forest, source);
        write_mask->add_expression_reference();
        ApUserEvent done_event;
        derez.deserialize(done_event);
        // Also unpack the write tracker
        if (copier->across_helper == NULL)
          copier->unpack_write_tracker(source, rt, derez);
        // Now we can perform the copies
        IndexSpaceExpression *performed_write = NULL;
        issue_deferred_copies_single(*copier, write_mask, 
                                     performed_write, pred_guard);
        // Do the finalization for the copy
        copier->finalize(done_event);
        // Lastly do the clean-up
        delete copier;
        if (write_mask->remove_expression_reference())
          delete write_mask;
      }
      else
      {
        FieldMask copy_mask;
        derez.deserialize(copy_mask);
        RemoteDeferredCopier *copier = 
          RemoteDeferredCopier::unpack_copier(derez, rt, copy_mask, ctx);
        size_t write_expressions;
        derez.deserialize(write_expressions);
        WriteMasks write_masks;
        std::map<unsigned/*index*/,ApUserEvent> done_events;
        for (unsigned idx = 0; idx < write_expressions; idx++)
        {
          IndexSpaceExpression *expression = 
            IndexSpaceExpression::unpack_expression(derez, 
                                  rt->forest, source);
          expression->add_expression_reference();
          FieldMask expr_mask;
          size_t field_count;
          derez.deserialize(field_count);
          for (unsigned fidx = 0; fidx < field_count; fidx++)
          {
            unsigned field_index;
            derez.deserialize(field_index);
            expr_mask.set_bit(field_index);
#ifdef DEBUG_LEGION
            assert(done_events.find(field_index) == done_events.end());
#endif
            derez.deserialize(done_events[field_index]);
            // Also unpack the write tracker
            if (copier->across_helper == NULL)
              copier->unpack_write_tracker(field_index, source, rt, derez);
          }
          if (!!expr_mask)
            write_masks.insert(expression, expr_mask);
        }
        // Now we can perform our copies
        WriteSet performed_writes;
        issue_deferred_copies(*copier, copy_mask, write_masks,
                              performed_writes, pred_guard);
        // Do the finalization
        copier->finalize(done_events);
        // Lastly do the clean-up
        delete copier;
        for (WriteMasks::const_iterator it = write_masks.begin();
              it != write_masks.end(); it++)
          if (it->first->remove_expression_reference())
            delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void CompositeView::handle_sharding_reduction_request(Deserializer &derez,
                     Runtime *runtime, InnerContext *ctx, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID dst_did;
      derez.deserialize(dst_did);
      RtEvent dst_ready;
      LogicalView *dst = 
        runtime->find_or_request_logical_view(dst_did, dst_ready);
      MaterializedView *dst_view = NULL;
      RemoteTraversalInfo *info = RemoteTraversalInfo::unpack(derez, runtime); 
      bool across;
      derez.deserialize(across);
      CopyAcrossHelper *across_helper = NULL;
      if (across)
      {
        FieldMask reduce_mask;
        derez.deserialize(reduce_mask);
        across_helper = CopyAcrossHelper::unpack(derez, reduce_mask);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        FieldMask reduction_mask;
        derez.deserialize(reduction_mask);
        PredEvent pred_guard;
        derez.deserialize(pred_guard);
        IndexSpaceExpression *expression = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        const size_t pop_count = reduction_mask.pop_count();
        if (pop_count == 1)
        {
          ApUserEvent done_event;
          derez.deserialize(done_event);
          ApEvent reduce_pre;
          derez.deserialize(reduce_pre);
          if (dst_view == NULL)
          {
            if (!dst_ready.has_triggered())
              dst_ready.wait();
#ifdef DEBUG_LEGION
            assert(dst->is_materialized_view());
#endif
            dst_view = dst->as_materialized_view();
          }
          CompositeSingleReducer reducer(info, ctx, dst_view, reduction_mask,
                                         reduce_pre, across_helper);
          if (across_helper != NULL)
            reducer.unpack_write_tracker(derez);
          issue_composite_reductions_single(reducer, expression, logical_node, 
                                            pred_guard, this);
          reducer.finalize(done_event, runtime, source);
        }
        else
        {
          if (dst_view == NULL)
          {
            if (!dst_ready.has_triggered())
              dst_ready.wait();
#ifdef DEBUG_LEGION
            assert(dst->is_materialized_view());
#endif
            dst_view = dst->as_materialized_view();
          }
          CompositeReducer reducer(info, ctx, dst_view,
                                   reduction_mask, across_helper);
          std::map<unsigned,ApUserEvent> done_events;
          for (unsigned idx = 0; idx < pop_count; idx++)
          {
            unsigned index;
            derez.deserialize(index);
#ifdef DEBUG_LEGION
            assert(done_events.find(index) == done_events.end());
#endif
            derez.deserialize(done_events[index]);
            if (across_helper != NULL)
              reducer.unpack_write_tracker(index, derez);
          }
          reducer.unpack(derez);
          WriteMasks reduce_masks;
          reduce_masks.insert(expression, reduction_mask);
          issue_composite_reductions(reducer, reduction_mask, reduce_masks, 
                                     logical_node, pred_guard, this);
          reducer.finalize(done_events, runtime, source);
        }
      }
      // Cleanup the resources that we own
      delete info;
      if (across_helper != NULL)
        delete across_helper;
    }
#else
    //--------------------------------------------------------------------------
    void CompositeView::handle_sharding_update_request(Deserializer &derez,
                                                       Runtime *runtime)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(packed_shard != NULL);
#endif
      RtUserEvent done_event;
      derez.deserialize(done_event);
      CompositeView *target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);

      Serializer rez;
      if (runtime->address_space != source)
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(packed_shard->get_buffer(),
                      packed_shard->get_used_bytes());
        rez.serialize(done_event); 
      }
      else
      {
        // No check or target here since we know where we're going
        rez.serialize(packed_shard->get_buffer(),
                      packed_shard->get_used_bytes());
        rez.serialize(done_event);
      }
      if (runtime->address_space == source)
      {
        // same node so can send it directly
        Deserializer local_derez(rez.get_buffer(), rez.get_used_bytes());
        target->unpack_composite_view_response(local_derez, runtime);
      }
      else
        runtime->send_control_replicate_composite_view_response(source, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeView::handle_composite_view_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      CompositeView *view;
      derez.deserialize(view);
      view->unpack_composite_view_response(derez, runtime);
    }
#endif

    /////////////////////////////////////////////////////////////
    // CompositeNode 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(RegionTreeNode* node, CompositeBase *p,
                                 DistributedID own_did, bool root_own)
      : CompositeBase(node_lock, false), logical_node(node), parent(p), 
        owner_did(own_did), root_owner(root_own)
#ifdef DISABLE_CVOPT
        , valid_version_states(NULL)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CompositeNode::CompositeNode(const CompositeNode &rhs)
      : CompositeBase(node_lock, false), logical_node(NULL), parent(NULL), 
        owner_did(0), root_owner(false)
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
      // Free up all our children 
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        delete (it->first);
      }
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            valid_views.begin(); it != valid_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(owner_did))
          delete it->first;
      }
      for (LegionMap<ReductionView*,FieldMask>::aligned::const_iterator it = 
            reduction_views.begin(); it != reduction_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(owner_did))
          delete (it->first);
      }
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
    void CompositeNode::unpack_composite_view_response(Deserializer &derez,
                                                       Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      size_t num_children;
      derez.deserialize(num_children);
      DistributedID owner_did = get_owner_did();
      std::set<RtEvent> ready_events;
      {
        AutoLock n_lock(node_lock);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          CompositeNode *child = CompositeNode::unpack_composite_node(derez,
           this, runtime, owner_did, ready_events, children, false/*root own*/);
          FieldMask child_mask;
          derez.deserialize(child_mask);
          // Have to do a merge of field masks here
          children[child] |= child_mask;
        }
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    InnerContext* CompositeNode::get_owner_context(void) const
    //--------------------------------------------------------------------------
    {
      return parent->get_owner_context();
    }

    //--------------------------------------------------------------------------
#ifndef DISABLE_CVOPT
    void CompositeNode::perform_ready_check(FieldMask mask)
#else
    void CompositeNode::perform_ready_check(FieldMask mask,
                                            RegionTreeNode *target)
#endif
    //--------------------------------------------------------------------------
    {
      RtUserEvent capture_event;
      std::set<RtEvent> preconditions;
#ifndef DISABLE_CVOPT
      // Do a quick test with read-only lock first
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Remove any fields that have already been captured
        mask -= captured_fields;
        if (!mask)
          return;
        // See if there are any pending captures to wait for also
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
      }
       
      LegionMap<VersionState*,FieldMask>::aligned needed_states;
      // If we still have fields we have to do more work
      if (!!mask)
      {
        AutoLock n_lock(node_lock);
        // Retest to see if we lost the race
        mask -= captured_fields;
        if (!mask)
          return;
        // See if there are any pending captures which we can wait for
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
        // If we still have fields then we are going to do a pending capture
        if (!!mask)
        {
          for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
                version_states.begin(); it != version_states.end(); it++)
          {
            const FieldMask overlap = it->second & mask;
            if (!overlap)
              continue;
            needed_states[it->first] = overlap;
          }
          if (!needed_states.empty())
          {
            capture_event = Runtime::create_rt_user_event();
            pending_captures[capture_event] = mask;
          }
          else // Nothing to capture so these fields are "captured"
            captured_fields |= mask;
        }
      }
#else
      LegionMap<VersionState*,FieldMask>::aligned needed_states;
      bool have_capture_fields = true;
      // Do a quick test with read-only lock first
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        // Check to see if we are disjoint with the need capture fields
        if (mask * uncaptured_fields)
        {
          have_capture_fields = false;
          // If there are any pending captures then we also need
          // the preconditions for those fields since we aren't going
          // to do the later test
          for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it =
                pending_captures.begin(); it != pending_captures.end(); it++)
          {
            if (it->second * mask)
              continue;
            preconditions.insert(it->first);
          }
        }
      }
      if (have_capture_fields) 
      {
        AutoLock n_lock(node_lock);
        // Check for any pending captures
        for (LegionMap<RtUserEvent,FieldMask>::aligned::const_iterator it =
              pending_captures.begin(); it != pending_captures.end(); it++)
        {
          if (it->second * mask)
            continue;
          preconditions.insert(it->first);
        }
        // Also see if there are any capture fields we need to perform
        const FieldMask capture_mask = uncaptured_fields & mask;
        if (!!capture_mask)
        {
          // Find the set of VersionState objects that need capturing
          std::vector<VersionState*> to_delete;
          for (LegionMap<VersionState*,FieldMask>::aligned::iterator it = 
                uncaptured_states.begin(); it != uncaptured_states.end(); it++)
          {
            const FieldMask needed_fields = it->second & capture_mask;
            if (!needed_fields)
              continue;
            needed_states[it->first] = needed_fields;
            it->second -= needed_fields;
            if (!it->second)
              to_delete.push_back(it->first);
          }
#ifdef DEBUG_LEGION
          assert(!needed_states.empty());
#endif
          if (!to_delete.empty())
          {
            for (std::vector<VersionState*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              uncaptured_states.erase(*it);
          }
          // Save a pending capture mask
          capture_event = Runtime::create_rt_user_event();
          pending_captures[capture_event] = capture_mask;
          // Then we can remove the capture mask from the capture fields
          uncaptured_fields -= capture_mask;
        }
      }
#endif
      if (capture_event.exists())
      {
        // Request final states for all the version states and then either
        // launch a task to do the capture, or do it now
        std::set<RtEvent> capture_preconditions;
        WrapperReferenceMutator mutator(capture_preconditions);
        InnerContext *owner_context = parent->get_owner_context();
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it = 
              needed_states.begin(); it != needed_states.end(); it++)
        {
          std::set<RtEvent> version_preconditions;
          it->first->request_final_version_state(owner_context, it->second,
                                                 version_preconditions);
          if (!version_preconditions.empty())
          {
            RtEvent version_precondition = 
              Runtime::merge_events(version_preconditions);
            if (version_precondition.exists())
            {
              DeferCaptureArgs args;
              args.proxy_this = this;
              args.version_state = it->first;
              // Little scary, but safe since we're going to
              // wait inside this scope and needed_states will
              // not be changing while this is occurring
              args.capture_mask = &it->second;
              capture_preconditions.insert(
                owner_context->runtime->issue_runtime_meta_task(args,
                  LG_LATENCY_WORK_PRIORITY, version_precondition));
              continue;
            }
          }
          // If we get here then we can just do the capture now
          capture(it->first, it->second, &mutator);
        }
        // Now we can trigger our capture event with the preconditions
        if (!capture_preconditions.empty())
        {
          Runtime::trigger_event(capture_event,
              Runtime::merge_events(capture_preconditions));
          // Wait for it to be ready
          capture_event.wait();
        }
        else
          Runtime::trigger_event(capture_event);
        // Now we can remove the capture event from the set
        AutoLock n_lock(node_lock);
        pending_captures.erase(capture_event);
#ifndef DISABLE_CVOPT
        // Once we're done with the capture then we can record 
        // the fields as having been captured
        captured_fields |= mask;
#endif
      }
      // Wait for anything else that we need to trigger
      if (!preconditions.empty())
      {
        RtEvent wait_on = Runtime::merge_events(preconditions);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void CompositeNode::find_valid_views(const FieldMask &update_mask,
                       LegionMap<LogicalView*,FieldMask>::aligned &result_views,
                                         bool needs_lock)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock n_lock(node_lock,1,false/*exclusive*/);
        find_valid_views(update_mask, result_views, false);
        return;
      }
      if (dirty_mask * update_mask)
        return;
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
    }


    //--------------------------------------------------------------------------
    void CompositeNode::capture(VersionState *to_capture, 
                                const FieldMask &capture_mask,
                                ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Need the lock while doing this for the callbacks that
      // will mutate the state of the composite node
      AutoLock n_lock(node_lock);
      to_capture->capture(this, capture_mask, mutator);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CompositeNode::handle_deferred_capture(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCaptureArgs *dargs = (const DeferCaptureArgs*)args;
      LocalReferenceMutator mutator;
      dargs->proxy_this->capture(dargs->version_state, 
                      *(dargs->capture_mask), &mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::clone(CompositeView *target,
                   const FieldMask &clone_mask, ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    {
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
                                           overlap, mutator);
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
        new CompositeNode(node, parent, owner_did, false/*root owner*/);
      result->unpack_version_states(derez, runtime, 
                                    preconditions, false/*need lock*/);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ CompositeNode* CompositeNode::unpack_composite_node(
              Deserializer &derez, CompositeBase *parent, Runtime *runtime, 
              DistributedID owner_did, std::set<RtEvent> &preconditions,
              const LegionMap<CompositeNode*,FieldMask>::aligned &existing,
              bool root_owner)
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
      CompositeNode *result = NULL; 
      // Check for it in the existing nodes
      for (LegionMap<CompositeNode*,FieldMask>::aligned::const_iterator it =
            existing.begin(); it != existing.end(); it++)
      {
        if (it->first->logical_node != node)
          continue;
        result = it->first;
        break;
      }
      // If we didn't find it then we get to make it
      if (result == NULL)
      {
        result = new CompositeNode(node, parent, owner_did, root_owner);
        result->unpack_version_states(derez, runtime, 
                                      preconditions, false/*need lock*/);
      }
      else // already exists so we need a lock when updating it
        result->unpack_version_states(derez, runtime,
                                      preconditions, true/*need lock*/);
      return result;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::unpack_version_states(Deserializer &derez, 
             Runtime *runtime, std::set<RtEvent> &preconditions, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock n_lock(node_lock);
        unpack_version_states(derez, runtime, preconditions,false/*need lock*/);
        return;
      }
      size_t num_versions;
      derez.deserialize(num_versions);
      WrapperReferenceMutator mutator(preconditions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        VersionState *state = 
          runtime->find_or_request_version_state(did, ready); 
        std::map<VersionState*,FieldMask>::iterator finder = 
          version_states.find(state);
#ifndef DISABLE_CVOPT
        if (finder != version_states.end())
        {
          FieldMask state_mask;
          derez.deserialize(state_mask);
          finder->second |= state_mask;
        }
        else
          derez.deserialize(version_states[state]);
        if (ready.exists() && !ready.has_triggered())
        {
          DeferCompositeNodeStateArgs args;
          args.proxy_this = this;
          args.state = state;
          args.owner_did = owner_did;
          args.root_owner = root_owner;
          RtEvent precondition = 
            runtime->issue_runtime_meta_task(args, 
                LG_LATENCY_DEFERRED_PRIORITY, ready);
          preconditions.insert(precondition);
        }
        else
        {
          state->add_nested_resource_ref(owner_did);
          // If we're the root owner then we also have to add our
          // gc ref, but no valid ref since we don't own it
          if (root_owner)
            state->add_nested_gc_ref(owner_did, &mutator);
        }
#else
        if (finder != version_states.end())
        {
          FieldMask state_mask;
          derez.deserialize(state_mask);
          const FieldMask diff_mask = state_mask - finder->second;
          if (!!diff_mask)
          {
            uncaptured_states[state] |= diff_mask;
            uncaptured_fields |= diff_mask;
          }
          finder->second |= state_mask;
          // No need to add any references since it was already captured
#ifdef DEBUG_LEGION
          assert(!ready.exists() || ready.has_triggered());
#endif
        }
        else
        {
          // It's not safe to actually add this to our data structures
          // until we know that the state is valid, so defer it if necessary
          if (ready.exists() && !ready.has_triggered())
          {
            // Defer adding this state
            DeferCompositeNodeStateArgs args;
            args.proxy_this = this;
            args.state = state;
            args.owner_did = owner_did;
            args.root_owner = root_owner;
            args.mask = new FieldMask();
            derez.deserialize(*args.mask);
            RtEvent precondition = 
              runtime->issue_runtime_meta_task(args, 
                  LG_LATENCY_DEFERRED_PRIORITY, ready);
            preconditions.insert(precondition);
          }
          else
          { 
            // Version State is ready now, so we can unpack it directly
            FieldMask &state_mask = version_states[state];
            derez.deserialize(state_mask);
            uncaptured_states[state] = state_mask;
            uncaptured_fields |= state_mask;
            state->add_nested_resource_ref(owner_did);
            // If we're the root owner then we also have to add our
            // gc ref, but no valid ref since we don't own it
            if (root_owner)
              state->add_nested_gc_ref(owner_did, &mutator);
          }
        }
#endif
      }
    }

#ifdef DISABLE_CVOPT
    //--------------------------------------------------------------------------
    void CompositeNode::add_uncaptured_state(VersionState *state, 
                                             const FieldMask &state_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock n_lock(node_lock);
      std::map<VersionState*,FieldMask>::iterator finder = 
          version_states.find(state);
      if (finder != version_states.end())
      {
        const FieldMask diff_mask = state_mask - finder->second;
        if (!!diff_mask)
        {
          uncaptured_states[state] |= diff_mask;
          uncaptured_fields |= diff_mask;
        }
        finder->second |= state_mask;
      }
      else
      {
        uncaptured_states[state] = state_mask;
        uncaptured_fields |= state_mask;
        version_states[state] = state_mask;
      }
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ void CompositeNode::handle_deferred_node_state(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferCompositeNodeStateArgs *nargs = 
        (const DeferCompositeNodeStateArgs*)args;
      // Add our references
      nargs->state->add_nested_resource_ref(nargs->owner_did);
      LocalReferenceMutator mutator;
      if (nargs->root_owner)
        nargs->state->add_nested_gc_ref(nargs->owner_did, &mutator);
#ifdef DISABLE_CVOPT
      // Add the state to the view
      nargs->proxy_this->add_uncaptured_state(nargs->state, *nargs->mask);
      // Free up the mask that we allocated
      delete nargs->mask;
#endif
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
        // Add a valid reference
        view->add_nested_resource_ref(owner_did);
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
        // Add a valid reference
        view->add_nested_resource_ref(owner_did);
        reduction_views[view] = mask;
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_child_version_state(const LegionColor color,
                                     VersionState *state, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      RegionTreeNode *child_node = logical_node->get_tree_child(color);
      for (LegionMap<CompositeNode*,FieldMask>::aligned::iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if (it->first->logical_node == child_node)
        {
          it->first->record_version_state(state, mask, NULL/*mutator*/); 
          it->second |= mask;
          return;
        }
      }
      // Didn't find it so make it
      CompositeNode *child = 
        new CompositeNode(child_node, this, owner_did, false/*root owner*/);
      child->record_version_state(state, mask, NULL/*mutator*/);
      children[child] = mask;
    }

    //--------------------------------------------------------------------------
    void CompositeNode::record_version_state(VersionState *state, 
                               const FieldMask &mask, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      LegionMap<VersionState*,FieldMask>::aligned::iterator finder = 
        version_states.find(state);
      if (finder == version_states.end())
      {
        version_states[state] = mask;
#ifdef DISABLE_CVOPT
        // Also need to update the uncaptured state
        uncaptured_states[state] = mask;
        uncaptured_fields |= mask;
#endif
        // Root owners need gc and valid references on the tree
        // otherwise everyone else just needs a resource reference
        state->add_nested_resource_ref(owner_did);
        if (root_owner)
        {
          state->add_nested_gc_ref(owner_did, mutator);
#ifdef DISABLE_CVOPT
          if (valid_version_states == NULL)
            valid_version_states = new std::vector<VersionState*>();
          valid_version_states->push_back(state);
#endif
          state->add_nested_valid_ref(owner_did, mutator);
        }
      }
      else
      {
        // Update the uncaptured data structure with any missing fileds
        const FieldMask uncaptured_mask = mask - finder->second;
        if (!!uncaptured_mask)
        {
#ifdef DISABLE_CVOPT
          uncaptured_states[state] |= uncaptured_mask;
          uncaptured_fields |= uncaptured_mask;
#endif
          // Only need to update this if the fields aren't already valid
          finder->second |= uncaptured_mask;
        }
      }
    } 

    //--------------------------------------------------------------------------
    void CompositeNode::release_gc_references(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(root_owner); // should only be called on the root owner
#endif
      // No need to check for deletion, we know we're also holding gc refs
      for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
            version_states.begin(); it != version_states.end(); it++)
        it->first->remove_nested_gc_ref(owner_did, mutator);
    }

    //--------------------------------------------------------------------------
    void CompositeNode::release_valid_references(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(root_owner); // should only be called on the root owner
#endif
#ifndef DISABLE_CVOPT
      if (root_owner)
      {
        for (LegionMap<VersionState*,FieldMask>::aligned::const_iterator it =
              version_states.begin(); it != version_states.end(); it++)
          it->first->remove_nested_valid_ref(owner_did, mutator);
      }
#else
      // No need to check for deletion, we know we're also holding gc refs
      if (valid_version_states != NULL)
      {
        for (std::vector<VersionState*>::const_iterator it = 
              valid_version_states->begin(); it != 
              valid_version_states->end(); it++)
          (*it)->remove_nested_valid_ref(owner_did, mutator);
        delete valid_version_states;
        valid_version_states = NULL;
      }
#endif
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
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void FillView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
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
                               true/*single copy*/, restrict_out, copy_mask, 
                               dst->logical_node->get_index_space_expression(),
                               &info.version_info, info.op->get_unique_op_id(),
                               info.index, local_space, 
                               preconditions, info.map_applied_events, info);
      if (restrict_info.has_restrictions())
      {
        FieldMask restrict_mask;
        restrict_info.populate_restrict_fields(restrict_mask);
        restrict_mask &= copy_mask;
        if (!!restrict_mask)
        {
          ApEvent restrict_pre = info.op->get_restrict_precondition(info);
          preconditions[restrict_pre] = restrict_mask;
        }
      }
      LegionMap<ApEvent,FieldMask>::aligned postconditions;
      issue_internal_fills(info, dst, copy_mask, preconditions,
                           postconditions, PredEvent::NO_PRED_EVENT);
      // We know there is at most one event per field so no need
      // to sort into event sets here
      // Register the resulting events as users of the destination
      IndexSpaceExpression *dst_expr = 
        dst->logical_node->get_index_space_expression();
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            postconditions.begin(); it != postconditions.end(); it++)
      {
        dst->add_copy_user(0/*redop*/, it->first, &info.version_info, 
                           dst_expr, info.op->get_unique_op_id(), info.index,
                           it->second, false/*reading*/, restrict_out,
                           local_space, info.map_applied_events, info);
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
    void FillView::issue_deferred_copies(DeferredCopier &copier,
                                         const FieldMask &local_copy_mask,
                                         const WriteMasks &write_masks,
                                         WriteSet &performed_writes,
                                         PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // The previous writes data structure should be field unique,
      // if it's not then we are going to have big problems
      FieldMask previous_writes_mask;
#endif
      FieldMask remaining_fill_mask = local_copy_mask;
      // First issue any masked fills for things that have been written before
      for (WriteMasks::const_iterator it = write_masks.begin(); 
            it != write_masks.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(previous_writes_mask * it->second);
        previous_writes_mask |= it->second;
#endif
        const FieldMask overlap = it->second & remaining_fill_mask;
        if (!overlap)
          continue;
        // Issue our update fills
        issue_update_fills(copier, overlap, it->first, 
                           performed_writes, pred_guard);
        remaining_fill_mask -= overlap;
        if (!remaining_fill_mask)
          return;
      }
#ifdef DEBUG_LEGION
      assert(!!remaining_fill_mask);
#endif
      // Then issue a remaining fill for any remainder events
      issue_update_fills(copier, remaining_fill_mask, 
                         NULL/*mask*/, performed_writes, pred_guard);
    }

    //--------------------------------------------------------------------------
    bool FillView::issue_deferred_copies_single(DeferredSingleCopier &copier,
                                         IndexSpaceExpression *write_mask,
                                         IndexSpaceExpression *&write_perf,
                                         PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      MaterializedView *dst = copier.dst;
      // Only apply an intersection if the destination logical node
      // is different than our logical node
      // If the intersection is empty we can skip the fill all together
      if ((logical_node != dst->logical_node) && 
          (!logical_node->intersects_with(dst->logical_node)))
        return false;
      // Build the dst fields vector
      std::vector<CopySrcDstField> dst_fields;
      dst->copy_to(copier.copy_mask, dst_fields, copier.across_helper);
      std::set<ApEvent> dst_preconditions;
      copier.merge_destination_preconditions(dst_preconditions);
      ApEvent fill_pre = Runtime::merge_events(copier.info, dst_preconditions);
      WriteSet fill_writes;
      // Issue the fill command
      ApEvent fill_post = dst->logical_node->issue_fill(copier.info, 
          dst_fields, value->value, value->value_size, fill_pre, pred_guard,
#ifdef LEGION_SPY
                      fill_op_uid,
#endif
              (logical_node == dst->logical_node) ? NULL : logical_node, 
              write_mask, &fill_writes, &copier.copy_mask);
      if (fill_post.exists())
        copier.record_postcondition(fill_post);
      if (!fill_writes.empty())
      {
#ifdef DEBUG_LEGION
        assert(fill_writes.size() == 1);
#endif
        write_perf = fill_writes.begin()->first;
        return CompositeBase::test_done(copier, write_perf, write_mask);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void FillView::issue_update_fills(DeferredCopier &copier,
                                      const FieldMask &fill_mask,
                                      IndexSpaceExpression *mask,
                                      WriteSet &performed_writes,
                                      PredEvent pred_guard) const
    //--------------------------------------------------------------------------
    {
      // Get the common set of events for these fields and issue the fills 
      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      copier.merge_destination_preconditions(fill_mask, preconditions);
      issue_internal_fills(*copier.info, copier.dst, fill_mask, preconditions,
                           copier.copy_postconditions, pred_guard, 
                           copier.across_helper, mask, &performed_writes);
    }

    //--------------------------------------------------------------------------
    void FillView::issue_internal_fills(const TraversalInfo &info,
                                        MaterializedView *dst,
                                        const FieldMask &fill_mask,
            const LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                  LegionMap<ApEvent,FieldMask>::aligned &postconditions,
                                        PredEvent pred_guard,
                                        CopyAcrossHelper *across_helper,
                                        IndexSpaceExpression *mask,
                                        WriteSet *performed_writes) const
    //--------------------------------------------------------------------------
    {
      LegionList<FieldSet<ApEvent> >::aligned precondition_sets;
      compute_field_sets<ApEvent>(fill_mask, preconditions, precondition_sets);
      // Iterate over the precondition sets
      for (LegionList<FieldSet<ApEvent> >::aligned::iterator pit = 
            precondition_sets.begin(); pit !=
            precondition_sets.end(); pit++)
      {
        FieldSet<ApEvent> &pre_set = *pit;
        // Build the dst fields vector
        std::vector<CopySrcDstField> dst_fields;
        dst->copy_to(pre_set.set_mask, dst_fields, across_helper);
        ApEvent fill_pre = Runtime::merge_events(&info, pre_set.elements);
        // Issue the fill command
        // Only apply an intersection if the destination logical node
        // is different than our logical node
        // If the intersection is empty we can skip the fill all together
        if ((logical_node != dst->logical_node) && 
            (!logical_node->intersects_with(dst->logical_node)))
        {
          if (info.recording)
          {
#ifdef DEBUG_LEGION
            assert(info.logical_ctx != -1U);
#endif
            info.tpl->record_empty_copy_from_fill_view(
                dst, pre_set.set_mask, info.logical_ctx, info.ctx);
          }
          continue;
        }
        ApEvent fill_post = dst->logical_node->issue_fill(&info, 
            dst_fields, value->value, value->value_size, fill_pre, pred_guard,
#ifdef LEGION_SPY
                        fill_op_uid,
#endif
                    (logical_node == dst->logical_node) ? NULL : logical_node,
                    mask, performed_writes, &pre_set.set_mask);
        if (info.recording)
        {
#ifdef DEBUG_LEGION
          assert(info.logical_ctx != -1U);
#endif
          info.tpl->record_deferred_copy_from_fill_view(
              const_cast<FillView*>(this), dst, pre_set.set_mask, 
              info.logical_ctx, info.ctx);
        }
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
                     register_now), CompositeBase(view_lock, false),
        true_guard(tguard), false_guard(fguard), version_info(info),
        owner_context(owner)
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
      : DeferredView(NULL, 0, 0, NULL, false), CompositeBase(view_lock, false),
        true_guard(PredEvent::NO_PRED_EVENT), 
        false_guard(PredEvent::NO_PRED_EVENT), version_info(NULL),
        owner_context(NULL)
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
    void PhiView::notify_owner_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Remove gc references from deferred views and 
      // valid references from materialized views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
      {
        if (it->first->is_deferred_view())
        {
          if (it->first->remove_nested_gc_ref(did, mutator))
            delete it->first;
        }
        else
        {
          if (it->first->remove_nested_valid_ref(did, mutator))
            delete it->first;
        }
      }
      true_views.clear();
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            false_views.begin(); it != false_views.end(); it++)
      {
        if (it->first->is_deferred_view())
        {
          if (it->first->remove_nested_gc_ref(did, mutator))
            delete it->first;
        }
        else
        {
          if (it->first->remove_nested_valid_ref(did, mutator))
            delete it->first;
        }
      }
      false_views.clear();
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_owner_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      // Remove valid references from any deferred views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
      {
        if (!it->first->is_deferred_view())
          continue;
        it->first->remove_nested_valid_ref(did, mutator);
      }
      // Remove valid references from any deferred views
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
      {
        if (!it->first->is_deferred_view())
          continue;
        it->first->remove_nested_valid_ref(did, mutator);
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::issue_deferred_copies(const TraversalInfo &info,
                                        MaterializedView *dst,
                                        FieldMask copy_mask,
                                        const RestrictInfo &restrict_info,
                                        bool restrict_out)
    //--------------------------------------------------------------------------
    {
      if (copy_mask.pop_count() == 1)
      {
        DeferredSingleCopier copier(&info, owner_context, dst, copy_mask, 
                                    restrict_info, restrict_out);
        IndexSpaceExpression *performed_write;
        issue_deferred_copies_single(copier, NULL/*write mask*/,
                                     performed_write, PredEvent::NO_PRED_EVENT);
        copier.finalize(this);
      }
      else
      {
        DeferredCopier copier(&info, owner_context, dst,
                              copy_mask, restrict_info, restrict_out);
        WriteMasks write_masks;
        WriteSet performed_writes;
        issue_deferred_copies(copier, copy_mask, write_masks, 
                              performed_writes, PredEvent::NO_PRED_EVENT);
        copier.finalize(this);
      }
    }

    //--------------------------------------------------------------------------
    void PhiView::issue_deferred_copies(DeferredCopier &copier,
                                        const FieldMask &local_copy_mask,
                                        const WriteMasks &write_masks,
                                        WriteSet &performed_writes,
                                        PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      WriteSet true_writes, false_writes;
      copier.begin_guard_protection();
      issue_update_copies(copier, logical_node, local_copy_mask, this,
                          true_guard, true_views, write_masks, true_writes);
      issue_update_copies(copier, logical_node, local_copy_mask, this,
                          false_guard, false_views, write_masks, false_writes);
      copier.end_guard_protection();
      // Built write combined sets and merge them to update 
      combine_writes(true_writes, copier, false/*prune*/);
      combine_writes(false_writes, copier, false/*prune*/);
      // Iterate over the two sets and check for equivalence of expressions 
      for (WriteSet::iterator true_it = true_writes.begin(); 
            true_it != true_writes.end(); true_it++)
      {
        for (WriteSet::iterator false_it = false_writes.begin(); 
              false_it != false_writes.end(); false_it++)
        {
          const FieldMask overlap = true_it->second & false_it->second;
          if (!overlap)
            continue;
          // Check to make sure that the sets are equal for now since
          // we won't handle the case correctly if they aren't
          IndexSpaceExpression *diff1 = 
            context->subtract_index_spaces(true_it->first, false_it->first);
          IndexSpaceExpression *diff2 = 
            context->subtract_index_spaces(false_it->first, true_it->first);
          if (!diff1->is_empty() || !diff2->is_empty())
            REPORT_LEGION_FATAL(LEGION_FATAL_INCONSISTENT_PHI_VIEW,
                "Legion Internal Fatal Error: Phi View has different "
                "write sets for true and false cases. This is currently "
                "unsupported. Please report this use case to the Legion "
                "developers mailing list.")
          // For now we'll add the true expression to the write set
          WriteSet::iterator finder = performed_writes.find(true_it->first);
          if (finder == performed_writes.end())
            performed_writes.insert(true_it->first, overlap);
          else
            finder.merge(overlap);
          // Filter out anything we can
          true_it.filter(overlap);
          false_it.filter(overlap);
          if (!true_it->second)
          {
            // Can prune in this case since we're about to break
            if (!false_it->second)
              false_writes.erase(false_it);
            break;
          }
        }
#ifdef DEBUG_LEGION
        assert(!true_it->second); // should have no fields left
#endif
      }
    }

    //--------------------------------------------------------------------------
    bool PhiView::issue_deferred_copies_single(DeferredSingleCopier &copier,
                                           IndexSpaceExpression *write_mask,
                                           IndexSpaceExpression *&write_perf,
                                           PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
      copier.begin_guard_protection();
      std::vector<LogicalView*> true_instances;
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
        if (it->second.is_set(copier.field_index))
          true_instances.push_back(it->first);
      IndexSpaceExpression *true_write = 
        issue_update_copies_single(copier, logical_node, this, 
                                   true_guard, true_instances, write_mask);
      std::vector<LogicalView*> false_instances;
      for (LegionMap<LogicalView*,FieldMask>::aligned::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
        if (it->second.is_set(copier.field_index))
          false_instances.push_back(it->first);
      IndexSpaceExpression *false_write = 
        issue_update_copies_single(copier, logical_node, this,
                                   false_guard, false_instances, write_mask);
      copier.end_guard_protection();
      if (true_write != false_write)
      {
        if ((true_write == NULL) || (false_write == NULL))
          REPORT_LEGION_FATAL(LEGION_FATAL_INCONSISTENT_PHI_VIEW,
                  "Legion Internal Fatal Error: Phi View has different "
                  "write sets for true and false cases. This is currently "
                  "unsupported. Please report this use case to the Legion "
                  "developers mailing list.")
        // Check to make sure that the sets are equal for now since
        // we won't handle the case correctly if they aren't
        IndexSpaceExpression *diff1 = 
            context->subtract_index_spaces(true_write, false_write);
        IndexSpaceExpression *diff2 = 
            context->subtract_index_spaces(false_write, true_write);
        if (!diff1->is_empty() || !diff2->is_empty())
          REPORT_LEGION_FATAL(LEGION_FATAL_INCONSISTENT_PHI_VIEW,
                  "Legion Internal Fatal Error: Phi View has different "
                  "write sets for true and false cases. This is currently "
                  "unsupported. Please report this use case to the Legion "
                  "developers mailing list.")
      }
      // Just save the true write expression for right now
      if (true_write != NULL)
      {
        write_perf = true_write;
        return test_done(copier, write_perf, write_mask); 
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void PhiView::record_true_view(LogicalView *view, const FieldMask &mask,
                                   ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        true_views.find(view);
      if (finder == true_views.end())
      {
        true_views[view] = mask;
        if (view->is_deferred_view())
        {
          // Deferred views need valid and gc references
          view->add_nested_gc_ref(did, mutator);
          view->add_nested_valid_ref(did, mutator);
        }
        else // Otherwise we just need the valid reference
          view->add_nested_resource_ref(did);
      }
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void PhiView::record_false_view(LogicalView *view, const FieldMask &mask,
                                    ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      LegionMap<LogicalView*,FieldMask>::aligned::iterator finder = 
        false_views.find(view);
      if (finder == false_views.end())
      {
        false_views[view] = mask;
        if (view->is_deferred_view())
        {
          // Deferred views need valid and gc references
          view->add_nested_gc_ref(did, mutator);
          view->add_nested_valid_ref(did, mutator);
        }
        else // Otherwise we just need the valid reference
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
    void PhiView::pack_writing_version_numbers(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      // Should never be writing to a Phi view
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PhiView::pack_upper_bound_node(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      version_info->pack_upper_bound_node(rez);
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
          node, own_ctx, register_now), manager(man)
#ifdef DISTRIBUTED_INSTANCE_VIEWS
        , remote_request_event(RtEvent::NO_RT_EVENT)
#endif
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
                                          const PhysicalTraceInfo &trace_info,
                                          bool restrict_out)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime,REDUCTION_VIEW_PERFORM_REDUCTION_CALL);
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;

      bool fold = target->reduce_to(manager->redop, reduce_mask, dst_fields);
      this->reduce_from(manager->redop, reduce_mask, src_fields);

      LegionMap<ApEvent,FieldMask>::aligned preconditions;
      IndexSpaceExpression *reduce_expr = 
        target->logical_node->get_index_space_expression();
      target->find_copy_preconditions(manager->redop, false/*reading*/, 
            false/*single copy*/, restrict_out, reduce_mask, reduce_expr,
            versions, op->get_unique_op_id(), index, local_space, 
            preconditions, map_applied_events, trace_info);
      this->find_copy_preconditions(manager->redop, true/*reading*/, 
           true/*single copy*/, restrict_out, reduce_mask, reduce_expr,
           versions, op->get_unique_op_id(), index, local_space, 
           preconditions, map_applied_events, trace_info);
      std::set<ApEvent> event_preconds;
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it = 
            preconditions.begin(); it != preconditions.end(); it++)
      {
        event_preconds.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(&trace_info, event_preconds); 
      if (trace_info.recording)
      {
#ifdef DEBUG_LEGION
        assert(trace_info.tpl != NULL && trace_info.tpl->is_recording());
#endif
        ContextID logical_ctx =
          op->find_logical_context(index)->get_context().get_id();
        ContextID physical_ctx =
          op->find_physical_context(index)->get_context().get_id();
        trace_info.tpl->record_copy_views(this, reduce_mask,
            logical_ctx, physical_ctx, target, reduce_mask,
            logical_ctx, physical_ctx);
      }
      ApEvent reduce_post = manager->issue_reduction(trace_info, 
                                                     src_fields, dst_fields,
                                                     target->logical_node,
                                                     reduce_pre, pred_guard,
                                                     fold, true/*precise*/,
                                                     NULL/*intersect*/,
                                                     NULL/*mask*/);
      
      target->add_copy_user(manager->redop, reduce_post, versions, reduce_expr,
                           op->get_unique_op_id(), index, reduce_mask, 
                           false/*reading*/, restrict_out, local_space, 
                           map_applied_events, trace_info);
      this->add_copy_user(manager->redop, reduce_post, versions, reduce_expr,
                         op->get_unique_op_id(), index, reduce_mask, 
                         true/*reading*/, restrict_out, local_space, 
                         map_applied_events, trace_info);
      if (restrict_out)
        op->record_restrict_postcondition(reduce_post);
    } 

    //--------------------------------------------------------------------------
    ApEvent ReductionView::perform_deferred_reduction(MaterializedView *target,
                                                      const FieldMask &red_mask,
                                                      VersionTracker *versions,
                                                      ApEvent dst_precondition,
                                                      Operation *op, 
                                                      unsigned index,
                                                      PredEvent pred_guard,
                                                      CopyAcrossHelper *helper,
                                                      RegionTreeNode *intersect,
                                                     IndexSpaceExpression *mask,
                                          std::set<RtEvent> &map_applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                             IndexSpaceExpression *&reduce_expr)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_PERFORM_DEFERRED_REDUCTION_CALL);
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      bool fold = target->reduce_to(manager->redop, red_mask, 
                                    dst_fields, helper);
      this->reduce_from(manager->redop, red_mask, src_fields);
      RegionTreeForest *context = target->logical_node->context;
      reduce_expr = target->logical_node->get_index_space_expression();
      if (intersect != NULL)
        reduce_expr = context->intersect_index_spaces(reduce_expr,
            intersect->get_index_space_expression());
      if (mask != NULL)
        reduce_expr = context->subtract_index_spaces(reduce_expr, mask);
      LegionMap<ApEvent,FieldMask>::aligned src_pre;
      // Don't need to ask the target for preconditions as they 
      // are included as part of the pre set
      find_copy_preconditions(manager->redop, true/*reading*/, 
                              true/*single copy*/, false/*restrict out*/,
                              red_mask, reduce_expr, versions, 
                              op->get_unique_op_id(), index,
                              local_space, src_pre, 
                              map_applied_events, trace_info);
      std::set<ApEvent> preconditions;
      preconditions.insert(dst_precondition);
      for (LegionMap<ApEvent,FieldMask>::aligned::const_iterator it =
            src_pre.begin(); it != src_pre.end(); it++)
      {
        preconditions.insert(it->first);
      }
      ApEvent reduce_pre = Runtime::merge_events(&trace_info, preconditions);
      ApEvent reduce_post = manager->issue_reduction(trace_info,
                                                     src_fields, dst_fields,
                                                     target->logical_node,
                                                     reduce_pre, pred_guard,
                                                     fold, true/*precise*/,
                                                     intersect, mask);
      if (trace_info.recording)
      {
        // TODO: this is not dead code
        assert(false); // XXX: This seems to be a dead code
      }
      // No need to add the user to the destination as that will
      // be handled by the caller using the reduce post event we return
      add_copy_user(manager->redop, reduce_post, versions, reduce_expr,
                    op->get_unique_op_id(), index, red_mask, 
                    true/*reading*/, false/*restrict out*/,
                    local_space, map_applied_events, trace_info);
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
                                                IndexSpaceExpression *copy_expr,
                                                VersionTracker *versions,
                                                const UniqueID creator_op_id,
                                                const unsigned index,
                                                const AddressSpaceID source,
                           LegionMap<ApEvent,FieldMask>::aligned &preconditions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                                bool can_filter)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(context->runtime, 
                        REDUCTION_VIEW_FIND_COPY_PRECONDITIONS_CALL);
#ifdef DEBUG_LEGION
      assert(can_filter); // should always be able to filter reductions
#endif
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
#else
      if (!is_logical_owner())
      {
        // If this is not the logical owner send a message to the 
        // logical owner to perform the analysis
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(redop);
          rez.serialize<bool>(reading);
          rez.serialize<bool>(single_copy);
          rez.serialize<bool>(restrict_out);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          if (!reading)
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          rez.serialize(creator_op_id);
          rez.serialize(index);
          rez.serialize<bool>(can_filter);
          // Make 1 Ap user event per field and add it to the set
          int pop_count = FieldMask::pop_count(copy_mask);
          int next_start = 0;
          for (int idx = 0; idx < pop_count; idx++)
          {
            int field_index = copy_mask.find_next_set(next_start);
            ApUserEvent field_ready = Runtime::create_ap_user_event();
            rez.serialize(field_index);
            rez.serialize(field_ready);
            preconditions[field_ready].set_bit(field_index); 
            // We'll start looking again at the next index after this one
            next_start = field_index + 1;
          }
          // Make a Rt user event to signal when we are done
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          rez.serialize(applied_event);
          applied_events.insert(applied_event);
        }
        runtime->send_instance_view_find_copy_preconditions(logical_owner, rez);
        return;
      }
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
                                      IndexSpaceExpression *copy_expr,
                                      const UniqueID creator_op_id,
                                      const unsigned index,
                                      const FieldMask &mask, 
                                      bool reading, bool restrict_out,
                                      const AddressSpaceID source,
                                      std::set<RtEvent> &applied_events,
                                      const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop == manager->redop);
#endif
#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
#endif
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
                                  logical_node->get_index_space_expression());
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = new PhysicalUser(usage, INVALID_COLOR, creator_op_id, index, 
                                  logical_node->get_index_space_expression());
        }
        AutoLock v_lock(view_lock);
        add_physical_user(user, reading, copy_term, mask);
        if (trace_info.recording)
        {
          trace_info.tpl->record_outstanding_gc_event(this, copy_term);
        }
        else
        {
          // Update the reference users
          if (outstanding_gc_events.find(copy_term) ==
              outstanding_gc_events.end())
          {
            outstanding_gc_events.insert(copy_term);
            issue_collect = true;
          }
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
                                                  const UniqueID op_id,
                                                  const unsigned index,
                                                  VersionTracker *versions,
                                              std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
#else
      if (!is_logical_owner())
      {
        ApUserEvent result = Runtime::create_ap_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
#ifdef DEBUG_LEGION
          assert(!IS_WRITE(usage));
#endif
          versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
          rez.serialize(result);
        }
        runtime->send_instance_view_find_user_preconditions(logical_owner, rez);
        return result;
      }
#endif
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
      return Runtime::merge_events(&trace_info, wait_on);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_user(const RegionUsage &usage, ApEvent term_event,
                                 const FieldMask &user_mask, Operation *op,
                                 const unsigned index, VersionTracker *versions,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info)
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
      const UniqueID op_id = op->get_unique_op_id();
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          if (IS_WRITE(usage))
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
        }
        runtime->send_instance_view_add_user(logical_owner, rez);
        return;
      }
#endif
      add_user_base(usage, term_event, user_mask, op_id, index, 
        context->runtime->address_space, versions, applied_events, trace_info);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_user_base(const RegionUsage &usage, 
                                 ApEvent term_event, const FieldMask &user_mask,
                                 const UniqueID op_id, const unsigned index, 
                                 const AddressSpaceID source, 
                                 VersionTracker *versions,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
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
                          index, logical_node->get_index_space_expression());
      bool issue_collect = false;
      {
        AutoLock v_lock(view_lock);
        add_physical_user(new_user, reading, term_event, user_mask);
        if (trace_info.recording)
        {
          trace_info.tpl->record_outstanding_gc_event(this, term_event);
        }
        else
        {
          // Only need to do this if we actually have a term event
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
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::add_user_fused(const RegionUsage &usage, 
                                          ApEvent term_event,
                                          const FieldMask &user_mask, 
                                          Operation *op, const unsigned index,
                                          VersionTracker *versions,
                                          std::set<RtEvent> &applied_events,
                                          const PhysicalTraceInfo &trace_info,
                                          bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (IS_REDUCE(usage))
        assert(usage.redop == manager->redop);
      else
        assert(IS_READ_ONLY(usage));
#endif
      const UniqueID op_id = op->get_unique_op_id();
#ifndef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner())
      {
        if (trace_info.recording)
          assert(false);
        ApUserEvent result = Runtime::create_ap_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(term_event);
          rez.serialize(user_mask);
          rez.serialize(op_id);
          rez.serialize(index);
          if (IS_WRITE(usage))
            versions->pack_writing_version_numbers(rez);
          else
            versions->pack_upper_bound_node(rez);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          applied_events.insert(applied_event);
          rez.serialize(applied_event);
          rez.serialize(update_versions);
          rez.serialize(result);
        }
        runtime->send_instance_view_add_user_fused(logical_owner, rez);
        return result;
      }
#endif
      return add_user_fused_base(usage, term_event, user_mask, op_id, index,
                                 versions, context->runtime->address_space, 
                                 applied_events, trace_info, update_versions);
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::add_user_fused_base(const RegionUsage &usage, 
                                          ApEvent term_event,
                                          const FieldMask &user_mask, 
                                          const UniqueID op_id, 
                                          const unsigned index,
                                          VersionTracker *versions,
                                          const AddressSpaceID source,
                                          std::set<RtEvent> &applied_events,
                                          const PhysicalTraceInfo &trace_info,
                                          bool update_versions/*=true*/)
    //--------------------------------------------------------------------------
    {
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
#ifdef DISTRIBUTED_INSTANCE_VIEWS
      if (!is_logical_owner() && reading)
        perform_remote_valid_check();
#elif defined(DEBUG_LEGION)
      assert(is_logical_owner());
#endif
#ifdef DEBUG_LEGION
      assert(logical_node->is_region());
#endif
      // Who cares just hold the lock in exlcusive mode, this analysis
      // shouldn't be too expensive for reduction views
      bool issue_collect = false;
      PhysicalUser *new_user = new PhysicalUser(usage, INVALID_COLOR, op_id, 
                          index, logical_node->get_index_space_expression());
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
          if (trace_info.recording)
          {
            trace_info.tpl->record_outstanding_gc_event(this, term_event);
          }
          else
          {
            if (outstanding_gc_events.find(term_event) ==
                outstanding_gc_events.end())
            {
              outstanding_gc_events.insert(term_event);
              issue_collect = true;
            }
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
      return Runtime::merge_events(&trace_info, wait_on);
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
                                    logical_node->get_index_space_expression());
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
    void ReductionView::update_gc_events(const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      for (std::set<ApEvent>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
      {
        outstanding_gc_events.insert(*it);
      }
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

#ifdef DISTRIBUTED_INSTANCE_VIEWS
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
            it->second.users.single_user->pack_user(rez, source);
            rez.serialize(it->second.user_mask);
          }
          else
          {
            rez.serialize<size_t>(it->second.users.multi_users->size());
            for (LegionMap<PhysicalUser*,FieldMask>::aligned::const_iterator 
                  uit = it->second.users.multi_users->begin(); uit != 
                  it->second.users.multi_users->end(); uit++)
            {
              uit->first->pack_user(rez, source);
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
                                                AddressSpaceID source,
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
              PhysicalUser::unpack_user(derez, false/*add ref*/,forest,source);
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
                PhysicalUser::unpack_user(derez,false/*add ref*/,forest,source);
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
                          logical_node->get_index_space_expression());
        }
        else
        {
          RegionUsage usage(REDUCE, EXCLUSIVE, redop);
          user = new PhysicalUser(usage, INVALID_COLOR, op_id, index,
                          logical_node->get_index_space_expression());
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
                   logical_node->get_index_space_expression());
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
#endif // DISTRIBUTED_INSTANCE_VIEWS

  }; // namespace Internal 
}; // namespace Legion

