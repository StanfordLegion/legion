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
                             AddressSpaceID own_addr, bool register_now)
      : DistributedCollectable(ctx->runtime, did, own_addr, register_now), 
        context(ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalView::~LogicalView(void)
    //--------------------------------------------------------------------------
    {
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

    /////////////////////////////////////////////////////////////
    // InstanceView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void InstanceView::defer_collect_user(ApEvent term_event,
                                          ReferenceMutator *mutator) 
    //--------------------------------------------------------------------------
    {
      // The runtime will add the gc reference to this view when necessary
      PhysicalManager *manager = get_manager(); 
      std::set<ApEvent> to_collect;
      if (manager->defer_collect_user(this, term_event, to_collect))
        add_base_gc_ref(PENDING_GC_REF, mutator);
      if (!to_collect.empty())
        collect_users(to_collect); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_deferred_collect(InstanceView *view,
                                            const std::set<ApEvent> &to_collect)
    //--------------------------------------------------------------------------
    {
      view->collect_users(to_collect);
      // Then remove the gc reference on the object
      if (view->remove_base_gc_ref(PENDING_GC_REF))
        delete view;
    }

    //--------------------------------------------------------------------------
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp,
                               AddressSpaceID log_own,
                               UniqueID own_ctx, bool register_now)
      : LogicalView(ctx, did, owner_sp, register_now), 
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
    /*static*/ void InstanceView::handle_view_register_user(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      IndexSpaceExpression *user_expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      ApEvent pre = inst_view->register_user(usage, user_mask, user_expr,
                                             op_id, index, term_event,
                                             applied_events, trace_info);
      Runtime::trigger_event(ready_event, pre);
      if (!applied_events.empty())
      {
        const RtEvent precondition = Runtime::merge_events(applied_events);
        Runtime::trigger_event(applied_event, precondition);
        // Send back a response to the source removing the remote valid ref
        inst_view->send_remote_valid_decrement(source, precondition);
      }
      else
      {
        Runtime::trigger_event(applied_event);
        // Send back a response to the source removing the remote valid ref
        inst_view->send_remote_valid_decrement(source, RtEvent::NO_RT_EVENT);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_find_copy_pre_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize<bool>(reading);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression *copy_expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      CopyFillAggregator *remote_aggregator;
      derez.deserialize(remote_aggregator);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      const PhysicalTraceInfo trace_info(NULL);
      EventFieldExprs preconditions;
      inst_view->find_copy_preconditions_remote(reading, copy_mask, copy_expr, 
                                      op_id, index, preconditions, trace_info);
      // Send back the response unless the preconditions are empty in
      // which case we can just trigger the done event
      if (!preconditions.empty())
      {
        // Pack up the response and send it back
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(did);
          rez.serialize<size_t>(preconditions.size());
          for (EventFieldExprs::const_iterator eit = preconditions.begin();
                eit != preconditions.end(); eit++)
          {
            rez.serialize(eit->first);
            const FieldMaskSet<IndexSpaceExpression> &exprs = eit->second;
            rez.serialize<size_t>(exprs.size());
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
                  exprs.begin(); it != exprs.end(); it++)
            {
              it->first->pack_expression(rez, source);
              rez.serialize(it->second);
            }
          }
          rez.serialize(remote_aggregator);
          rez.serialize<bool>(reading);
          rez.serialize(done_event);
        }
        runtime->send_view_find_copy_preconditions_response(source, rez);
      }
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_find_copy_pre_response(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      EventFieldExprs preconditions;
      size_t num_events;
      derez.deserialize(num_events);
      for (unsigned idx1 = 0; idx1 < num_events; idx1++)
      {
        ApEvent event;
        derez.deserialize(event);
        size_t num_exprs;
        derez.deserialize(num_exprs);
        FieldMaskSet<IndexSpaceExpression> &exprs = preconditions[event];
        for (unsigned idx2 = 0; idx2 < num_exprs; idx2++)
        {
          IndexSpaceExpression *expr = IndexSpaceExpression::unpack_expression(
                                                derez, runtime->forest, source);
          FieldMask expr_mask;
          derez.deserialize(expr_mask);
          exprs.insert(expr, expr_mask);
        }
      }
      CopyFillAggregator *local_aggregator;
      derez.deserialize(local_aggregator);
      bool reading;
      derez.deserialize(reading);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      local_aggregator->record_preconditions(inst_view, reading, preconditions);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_add_copy_user(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      bool reading;
      derez.deserialize(reading);
      ApEvent term_event;
      derez.deserialize(term_event);
      FieldMask copy_mask;
      derez.deserialize(copy_mask);
      IndexSpaceExpression *copy_expr =
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      std::set<RtEvent> applied_events;
      const PhysicalTraceInfo trace_info(NULL);
      inst_view->add_copy_user(reading, term_event, copy_mask, copy_expr,
                               op_id, index, applied_events, trace_info);
      if (!applied_events.empty())
      {
        const RtEvent precondition = Runtime::merge_events(applied_events);
        Runtime::trigger_event(applied_event, precondition);
        // Send back a response to the source removing the remote valid ref
        inst_view->send_remote_valid_decrement(source, precondition);
      }
      else
      {
        Runtime::trigger_event(applied_event);
        // Send back a response to the source removing the remote valid ref
        inst_view->send_remote_valid_decrement(source, RtEvent::NO_RT_EVENT);
      }
    }

    /////////////////////////////////////////////////////////////
    // MaterializedView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
                               RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID own_addr,
                               AddressSpaceID log_own, InstanceManager *man,
                               UniqueID own_ctx, bool register_now)
      : InstanceView(ctx, encode_materialized_did(did), own_addr,
                     log_own, own_ctx, register_now), manager(man)
    //--------------------------------------------------------------------------
    {
      // Otherwise the instance lock will get filled in when we are unpacked
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      // Keep the manager from being collected
      manager->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Materialized View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space, 
          LEGION_DISTRIBUTED_ID_FILTER(manager->did)); 
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, 0, 0, 0, false), manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    {
      if (manager->remove_nested_resource_ref(did))
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
    void MaterializedView::accumulate_events(std::set<ApEvent> &all_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock,1,false/*exclusive*/);
      all_events.insert(outstanding_gc_events.begin(),
                        outstanding_gc_events.end());
    } 

    //--------------------------------------------------------------------------
    void MaterializedView::add_initial_user(ApEvent term_event,
                                            const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            IndexSpaceExpression *user_expr,
                                            const UniqueID op_id,
                                            const unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // No need to take the lock since we are just initializing
      PhysicalUser *user = 
        new PhysicalUser(usage, user_expr, op_id, index, false/*copy*/); 
      user->add_reference();
      add_current_user(user, term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::register_user(const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            IndexSpaceExpression *user_expr,
                                            const UniqueID op_id,
                                            const unsigned index,
                                            ApEvent term_event,
                                            std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        // If we're not the logical owner send a message there 
        // to do the analysis and provide a user event to trigger
        // with the precondition
        ApUserEvent ready_event = Runtime::create_ap_user_event();
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(user_mask);
          user_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(term_event);
          rez.serialize(ready_event);
          rez.serialize(applied_event);
        }
        // Add a remote valid reference that will be removed by 
        // the receiver once the changes have been applied
        WrapperReferenceMutator mutator(applied_events);
        add_base_valid_ref(REMOTE_DID_REF, &mutator);
        runtime->send_view_register_user(logical_owner, rez);
        applied_events.insert(applied_event);
        return ready_event;
      }
      else
      {
        std::set<ApEvent> wait_on_events;
        ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          wait_on_events.insert(start_use_event);
        // Find the preconditions
        find_user_preconditions(usage, user_expr, user_mask, term_event, 
                                op_id, index, wait_on_events, trace_info);
        // Add our local user
        const bool issue_collect = add_user(usage, user_expr, user_mask,
          term_event, op_id, index, false/*copy*/, applied_events, trace_info);
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
        if (!wait_on_events.empty())
          return Runtime::merge_events(&trace_info, wait_on_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent MaterializedView::find_copy_preconditions(bool reading,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            CopyFillAggregator &aggregator,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(&aggregator);
          rez.serialize(ready_event);
        }
        runtime->send_view_find_copy_preconditions_request(logical_owner, rez);
        return ready_event;
      }
      else
      {
        EventFieldExprs preconditions;
        const RegionUsage usage(reading ? READ_ONLY : READ_WRITE, EXCLUSIVE, 0);
        find_copy_preconditions(usage, copy_expr, copy_mask, 
                                op_id, index, preconditions, trace_info); 
        // Return any preconditions we found to the aggregator
        if (!preconditions.empty())
          aggregator.record_preconditions(this, reading, preconditions);
        // We're done with the analysis
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_remote(bool reading,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            EventFieldExprs &preconditions,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      const RegionUsage usage(reading ? READ_ONLY : READ_WRITE, EXCLUSIVE, 0);
      find_copy_preconditions(usage, copy_expr, copy_mask, 
                              op_id, index, preconditions, trace_info);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user(bool reading, ApEvent term_event,
                                         const FieldMask &copy_mask,
                                         IndexSpaceExpression *copy_expr,
                                         UniqueID op_id, unsigned index,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(term_event);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(applied_event);
        }
        // Add a remote valid reference that will be removed by 
        // the receiver once the changes have been applied
        WrapperReferenceMutator mutator(applied_events);
        add_base_valid_ref(REMOTE_DID_REF, &mutator);
        runtime->send_view_add_copy_user(logical_owner, rez);
        applied_events.insert(applied_event);
      }
      else
      {
        const RegionUsage usage(reading ? READ_ONLY : READ_WRITE, EXCLUSIVE, 0);
        const bool issue_collect = add_user(usage, copy_expr, copy_mask,
            term_event, op_id, index, true/*copy*/, applied_events, trace_info);
        // Launch the garbage collection task, if it doesn't exist
        // then the user wasn't registered anyway, see add_local_user
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(term_event, &mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_user_preconditions(const RegionUsage &usage,
                                            IndexSpaceExpression *user_expr,
                                            const FieldMask &user_mask,
                                            ApEvent term_event,
                                            UniqueID op_id, unsigned index,
                                            std::set<ApEvent> &preconditions,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      std::set<ApEvent> dead_events;
      EventFieldUsers current_to_filter, previous_to_filter;
      // Perform the analysis with a read-only lock
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        FieldMask observed, non_dominated;
        find_current_preconditions(usage, user_mask, user_expr, 
                                   term_event, op_id, index, 
                                   preconditions, dead_events, 
                                   current_to_filter, observed,
                                   non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, user_expr, previous_to_filter);
        const FieldMask previous_mask = user_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(usage, previous_mask, user_expr,
                                      term_event, op_id, index,
                                      preconditions, dead_events, trace_info);
      }
      if ((!dead_events.empty() || 
           !previous_to_filter.empty() || !current_to_filter.empty()) &&
          !trace_info.recording)
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions(const RegionUsage &usage,
                                            IndexSpaceExpression *copy_expr,
                                            const FieldMask &copy_mask,
                                            UniqueID op_id, unsigned index,
                                            EventFieldExprs &preconditions,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(preconditions.empty());
#endif
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        preconditions[start_use_event].insert(copy_expr, copy_mask);
      FieldMask filter_mask;
      std::set<ApEvent> dead_events;
      EventFieldUsers current_to_filter, previous_to_filter;
      // Do the first pass with a read-only lock on the events
      {
        FieldMask observed, non_dominated;
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_current_preconditions(usage, copy_mask, copy_expr, op_id, index,
                                   preconditions, dead_events,
                                   current_to_filter, observed,
                                   non_dominated, trace_info);
        const FieldMask dominated = observed - non_dominated;
        if (!!dominated)
          find_previous_filter_users(dominated, copy_expr, previous_to_filter);
        const FieldMask previous_mask = copy_mask - dominated;
        if (!!previous_mask)
          find_previous_preconditions(usage, previous_mask, copy_expr, op_id, 
                              index, preconditions, dead_events, trace_info);
      }
      if ((!dead_events.empty() || !previous_to_filter.empty() || 
            !current_to_filter.empty()) && !trace_info.recording)
      {
        // Need exclusive permissions to modify data structures
        AutoLock v_lock(view_lock);
        if (!dead_events.empty())
          for (std::set<ApEvent>::const_iterator it = dead_events.begin();
                it != dead_events.end(); it++)
            filter_local_users(*it); 
        if (!previous_to_filter.empty())
          filter_previous_users(previous_to_filter);
        if (!current_to_filter.empty())
          filter_current_users(current_to_filter);
      }
    }

    //--------------------------------------------------------------------------
    bool MaterializedView::add_user(const RegionUsage &usage,
                                    IndexSpaceExpression *user_expr,
                                    const FieldMask &user_mask,
                                    ApEvent term_event, UniqueID op_id, 
                                    unsigned index, bool copy_user,
                                    std::set<RtEvent> &applied_events,
                                    const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      PhysicalUser *new_user = 
        new PhysicalUser(usage, user_expr, op_id, index, copy_user);
      new_user->add_reference();
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      // Finally add our user and return if we need to issue a GC meta-task
      add_current_user(new_user, term_event, user_mask);
      // If we're tracing we defer adding these events until the trace
      // capture is complete so we can get a full set of preconditions
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
        return true;
      }
      else
        return false;
    }
 
    //--------------------------------------------------------------------------
    void MaterializedView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        manager->add_nested_gc_ref(did, mutator);
      else
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // we have a resource reference on the manager so no need to check
      if (is_owner())
        manager->remove_nested_gc_ref(did, mutator);
      else
        send_remote_gc_decrement(owner_space, RtEvent::NO_RT_EVENT, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // we have a resource reference on the manager so no need to check
      manager->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::collect_users(const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      // Remove any event users from the current and previous users
      for (std::set<ApEvent>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
        filter_local_users(*it); 
    }

    //--------------------------------------------------------------------------
    void MaterializedView::update_gc_events(
                                           const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      for (std::set<ApEvent>::const_iterator it = term_events.begin();
            it != term_events.end(); it++)
        outstanding_gc_events.insert(*it);
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
      AutoLock v_lock(view_lock);
      for (std::deque<ApEvent>::const_iterator it = gc_events.begin();
            it != gc_events.end(); it++)
        outstanding_gc_events.insert(*it);
    }    

    //--------------------------------------------------------------------------
    void MaterializedView::add_current_user(PhysicalUser *user, 
                                            ApEvent term_event,
                                            const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Must be called while holding the lock
      // Reference should already have been added
      EventUsers &event_users = current_epoch_users[term_event];
      EventUsers::iterator finder = event_users.find(user);
      if (finder == event_users.end())
        event_users.insert(user, user_mask);
      else
        finder.merge(user_mask);
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
        EventFieldUsers::iterator current_finder = 
          current_epoch_users.find(term_event);
        if (current_finder != current_epoch_users.end())
        {
          for (EventUsers::const_iterator it = current_finder->second.begin();
                it != current_finder->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
          current_epoch_users.erase(current_finder);
        }
        LegionMap<ApEvent,EventUsers>::aligned::iterator previous_finder = 
          previous_epoch_users.find(term_event);
        if (previous_finder != previous_epoch_users.end())
        {
          for (EventUsers::const_iterator it = previous_finder->second.begin();
                it != previous_finder->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
          previous_epoch_users.erase(previous_finder);
        }
        outstanding_gc_events.erase(event_finder);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_current_users(
                                               const EventFieldUsers &to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller 
      for (EventFieldUsers::const_iterator fit = to_filter.begin();
            fit != to_filter.end(); fit++)
      {
        EventFieldUsers::iterator event_finder = 
          current_epoch_users.find(fit->first);
        // If it's already been pruned out then either it was filtered
        // because it finished or someone else moved it already, either
        // way we don't need to do anything about it
        if (event_finder == current_epoch_users.end())
          continue;
        EventFieldUsers::iterator target_finder = 
          previous_epoch_users.find(fit->first);
        for (EventUsers::const_iterator it = fit->second.begin();
              it != fit->second.end(); it++)
        {
          EventUsers::iterator finder = event_finder->second.find(it->first);
          // Might already have been pruned out again, either way there is
          // nothing for us to do here if it was already moved
          if (finder == event_finder->second.end())
            continue;
          const FieldMask overlap = finder->second & it->second;
          if (!overlap)
            continue;
          finder.filter(overlap);
          bool needs_reference = true;
          if (!finder->second)
          {
            // Have the reference flow back with the user
            needs_reference = false;
            event_finder->second.erase(finder);
          }
          // Now add the user to the previous set
          if (target_finder == previous_epoch_users.end())
          {
            if (needs_reference)
              it->first->add_reference();
            previous_epoch_users[fit->first].insert(it->first, overlap);
            target_finder = previous_epoch_users.find(fit->first);
          }
          else
          {
            finder = target_finder->second.find(it->first);
            if (finder == target_finder->second.end())
            {
              if (needs_reference)
                it->first->add_reference();
              target_finder->second.insert(it->first, overlap);
            }
            else
            {
              finder.merge(overlap);
              // Remove any extra references we might be trying to send back
              if (!needs_reference && it->first->remove_reference())
                delete it->first;
            }
          }
        }
        if (event_finder->second.empty())
          current_epoch_users.erase(event_finder);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::filter_previous_users(
                                               const EventFieldUsers &to_filter)
    //--------------------------------------------------------------------------
    {
      // Lock needs to be held by caller
      for (EventFieldUsers::const_iterator fit = to_filter.begin();
            fit != to_filter.end(); fit++)
      {
        EventFieldUsers::iterator event_finder = 
          previous_epoch_users.find(fit->first);
        // Might already have been pruned out
        if (event_finder == previous_epoch_users.end())
          continue;
        for (EventUsers::const_iterator it = fit->second.begin();
              it != fit->second.end(); it++)
        {
          EventUsers::iterator finder = event_finder->second.find(it->first);
          // Might already have been pruned out again
          if (finder == event_finder->second.end())
            continue;
          finder.filter(it->second);
          if (!finder->second)
          {
            if (finder->first->remove_reference())
              delete finder->first;
            event_finder->second.erase(finder);
          }
        }
        if (event_finder->second.empty())
          previous_epoch_users.erase(event_finder);
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_current_preconditions(
                                               const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               ApEvent term_event,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events,
                                               EventFieldUsers &filter_users,
                                               FieldMask &observed,
                                               FieldMask &non_dominated,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (EventFieldUsers::const_iterator cit = current_epoch_users.begin(); 
            cit != current_epoch_users.end(); cit++)
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
        const FieldMask overlap = event_users.get_valid_mask() & user_mask;
        if (!overlap)
          continue;
        EventFieldUsers::iterator to_filter = filter_users.find(cit->first);
        for (EventUsers::const_iterator it = event_users.begin();
              it != event_users.end(); it++)
        {
          const FieldMask user_overlap = user_mask & it->second;
          if (!user_overlap)
            continue;
          observed |= user_overlap;
          IndexSpaceExpression *overlap_expr = NULL;
          if (has_local_precondition<false>(it->first, usage, op_id, index,
                                            user_expr, &overlap_expr))
          {
            preconditions.insert(cit->first);
            if (overlap_expr->get_volume() == it->first->expr_volume)
            {

              if (to_filter == filter_users.end())
              {
                filter_users[cit->first].insert(it->first, user_overlap);
                to_filter = filter_users.find(cit->first);
              }
              else
              {
#ifdef DEBUG_LEGION
                assert(to_filter->second.find(it->first) == 
                        to_filter->second.end());
#endif
                to_filter->second.insert(it->first, user_overlap);
              }
            }
            else
              non_dominated |= user_overlap;
          }
          else
            non_dominated |= user_overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                               const RegionUsage &usage,
                                               const FieldMask &user_mask,
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
      for (EventFieldUsers::const_iterator pit = previous_epoch_users.begin(); 
            pit != previous_epoch_users.end(); pit++)
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
        if (user_mask * event_users.get_valid_mask())
          continue;
        for (EventUsers::const_iterator it = event_users.begin();
              it != event_users.end(); it++)
        {
          if (user_mask * it->second)
            continue;
          if (has_local_precondition<false>(it->first, usage,
                                            op_id, index, user_expr))
            preconditions.insert(pit->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_current_preconditions(
                                               const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               EventFieldExprs &preconditions,
                                               std::set<ApEvent> &dead_events,
                                               EventFieldUsers &filter_events,
                                               FieldMask &observed,
                                               FieldMask &non_dominated,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (EventFieldUsers::const_iterator cit = current_epoch_users.begin(); 
            cit != current_epoch_users.end(); cit++)
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
        FieldMask overlap = event_users.get_valid_mask() & user_mask;
        if (!overlap)
          continue;
        EventFieldExprs::iterator finder = preconditions.find(cit->first);
#ifndef LEGION_SPY
        if (!trace_info.recording && finder != preconditions.end())
        {
          overlap -= finder->second.get_valid_mask();
          if (!overlap)
            continue;
        }
#endif
        EventFieldUsers::iterator to_filter = filter_events.find(cit->first);
        for (EventUsers::const_iterator it = event_users.begin();
              it != event_users.end(); it++)
        {
          const FieldMask user_overlap = user_mask & it->second;
          if (!user_overlap)
            continue;
          observed |= user_overlap;
          IndexSpaceExpression *overlap_expr = NULL;
          if (has_local_precondition<true>(it->first, usage, op_id, index, 
                                           user_expr, &overlap_expr))
          {
            if (finder == preconditions.end())
            {
              preconditions[cit->first].insert(overlap_expr, user_overlap);
              finder = preconditions.find(cit->first);
            }
            else
              finder->second.insert(overlap_expr, user_overlap);
            if (overlap_expr->get_volume() == it->first->expr_volume)
            {
              if (to_filter == filter_events.end())
              {
                filter_events[cit->first].insert(it->first, user_overlap);
                to_filter = filter_events.find(cit->first);
              }
              else
                to_filter->second.insert(it->first, user_overlap);
            }
            else
              non_dominated |= user_overlap;
          }
          else
            non_dominated |= user_overlap;
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_preconditions(
                                               const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               EventFieldExprs &preconditions,
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
        FieldMask overlap = user_mask & event_users.get_valid_mask();
        if (!overlap)
          continue;
        EventFieldExprs::iterator finder = preconditions.find(pit->first);
#ifndef LEGION_SPY
        if (!trace_info.recording && finder != preconditions.end())
        {
          overlap -= finder->second.get_valid_mask();
          if (!overlap)
            continue;
        }
#endif
        for (EventUsers::const_iterator it = event_users.begin();
              it != event_users.end(); it++)
        {
          const FieldMask user_overlap = overlap & it->second;
          if (!user_overlap)
            continue;
          IndexSpaceExpression *overlap_expr = NULL;
          if (has_local_precondition<true>(it->first, usage, op_id, index, 
                                           user_expr, &overlap_expr))
          {
            if (finder == preconditions.end())
            {
              preconditions[pit->first].insert(overlap_expr, user_overlap);
              // Needed for when we go around the loop again
              finder = preconditions.find(pit->first);
            }
            else
              finder->second.insert(overlap_expr, user_overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_previous_filter_users(const FieldMask &dom_mask,
                                              IndexSpaceExpression *filter_expr,
                                                  EventFieldUsers &filter_users)
    //--------------------------------------------------------------------------
    {
      // Lock better be held by caller
      for (EventFieldUsers::const_iterator pit = previous_epoch_users.begin(); 
            pit != previous_epoch_users.end(); pit++)
      {
        FieldMask event_overlap = pit->second.get_valid_mask() & dom_mask;
        if (!event_overlap)
          continue;
        EventFieldUsers::iterator to_filter = filter_users.find(pit->first);
        for (EventUsers::const_iterator it = pit->second.begin();
              it != pit->second.end(); it++)
        {
          const FieldMask user_overlap = it->second & event_overlap;
          if (!user_overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(it->first->expr, filter_expr);
          if (expr_overlap->is_empty())
            continue;
          else if (expr_overlap->get_volume() == it->first->expr_volume)
          {
            if (to_filter == filter_users.end())
            {
              filter_users[pit->first].insert(it->first, user_overlap);
              to_filter = filter_users.find(pit->first);
            }
            else
              to_filter->second.insert(it->first, user_overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_atomic_reservations(const FieldMask &mask,
                                                    Operation *op, bool excl)
    //--------------------------------------------------------------------------
    {
      // Compute the field set
      std::vector<FieldID> atomic_fields;
      manager->field_space_node->get_field_set(mask, atomic_fields);
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
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      UniqueID context_uid;
      derez.deserialize(context_uid);
      RtEvent man_ready;
      PhysicalManager *phy_man = 
        runtime->find_or_request_physical_manager(manager_did, man_ready);
      if (man_ready.exists())
        man_ready.wait();
#ifdef DEBUG_LEGION
      assert(phy_man->is_instance_manager());
#endif
      InstanceManager *inst_manager = phy_man->as_instance_manager();
      void *location;
      MaterializedView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) MaterializedView(runtime->forest,
                                              did, owner_space, 
                                              logical_owner, inst_manager,
                                              context_uid,
                                              false/*register now*/);
      else
        view = new MaterializedView(runtime->forest, did, owner_space,
                                    logical_owner, inst_manager, 
                                    context_uid, false/*register now*/);
      // Register only after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

    /////////////////////////////////////////////////////////////
    // DeferredView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredView::DeferredView(RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID owner_sp, bool register_now)
      : LogicalView(ctx, did, owner_sp, register_now)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // FillView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(RegionTreeForest *ctx, DistributedID did,
                       AddressSpaceID owner_proc,
                       FillViewValue *val, bool register_now
#ifdef LEGION_SPY
                       , UniqueID op_uid
#endif
                       )
      : DeferredView(ctx, encode_fill_did(did), owner_proc, register_now), 
        value(val)
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
      : DeferredView(NULL, 0, 0, false), value(NULL)
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
    void FillView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void FillView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_decrement(owner_space, RtEvent::NO_RT_EVENT, mutator);
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
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(owner_space);
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
    void FillView::flatten(CopyFillAggregator &aggregator, Operation *op,
                           InstanceView *dst_view, const FieldMask &src_mask,
                           IndexSpaceExpression *expr, CopyAcrossHelper *helper)
    //--------------------------------------------------------------------------
    {
      aggregator.record_fill(dst_view, this, src_mask, expr, helper);
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
      size_t value_size;
      derez.deserialize(value_size);
      void *value = malloc(value_size);
      derez.deserialize(value, value_size);
#ifdef LEGION_SPY
      UniqueID op_uid;
      derez.deserialize(op_uid);
#endif
      
      FillView::FillViewValue *fill_value = 
                      new FillView::FillViewValue(value, value_size);
      void *location;
      FillView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) FillView(runtime->forest, did,
                                      owner_space, fill_value,
                                      false/*register now*/
#ifdef LEGION_SPY
                                      , op_uid
#endif
                                      );
      else
        view = new FillView(runtime->forest, did, owner_space,
                            fill_value, false/*register now*/
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
                     PredEvent tguard, PredEvent fguard, 
                     InnerContext *owner, bool register_now) 
      : DeferredView(ctx, encode_phi_did(did), owner_space, register_now),
        true_guard(tguard), false_guard(fguard), owner_context(owner)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Phi View %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    PhiView::PhiView(const PhiView &rhs)
      : DeferredView(NULL, 0, 0, false),
        true_guard(PredEvent::NO_PRED_EVENT), 
        false_guard(PredEvent::NO_PRED_EVENT), owner_context(NULL)
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
    void PhiView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (!is_owner())
        send_remote_gc_decrement(owner_space, RtEvent::NO_RT_EVENT, mutator);
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
      DeferPhiViewRefArgs args(dc, did);
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
        rez.serialize(true_guard);
        rez.serialize(false_guard);
        rez.serialize<UniqueID>(owner_context->get_context_uid());
        pack_phi_view(rez);
      }
      runtime->send_phi_view(target, rez);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    void PhiView::flatten(CopyFillAggregator &aggregator, Operation *op,
                          InstanceView *dst_view, const FieldMask &src_mask,
                          IndexSpaceExpression *expr, CopyAcrossHelper *helper)
    //--------------------------------------------------------------------------
    {
      // TODO: implement this
      assert(false);
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
      PredEvent true_guard, false_guard;
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      UniqueID owner_uid;
      derez.deserialize(owner_uid);
      InnerContext *owner_context = runtime->find_context(owner_uid);
      // Make the phi view but don't register it yet
      void *location;
      PhiView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) PhiView(runtime->forest, did, owner,
                                     true_guard, false_guard, owner_context, 
                                     false/*register_now*/);
      else
        view = new PhiView(runtime->forest, did, owner, true_guard, 
                           false_guard, owner_context, false/*register now*/);
      // Unpack all the internal data structures
      std::set<RtEvent> ready_events;
      view->unpack_phi_view(derez, ready_events);
      if (!ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(ready_events);
        DeferPhiViewRegistrationArgs args(view);
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
                                 AddressSpaceID log_own,
                                 ReductionManager *man, UniqueID own_ctx, 
                                 bool register_now)
      : InstanceView(ctx, encode_reduction_did(did), own_sp, log_own, 
                     own_ctx, register_now), manager(man)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      manager->add_nested_resource_ref(did);
#ifdef LEGION_GC
      log_garbage.info("GC Reduction View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : InstanceView(NULL, 0, 0, 0, 0, false), manager(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    {
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
    PhysicalManager* ReductionView::get_manager(void) const
    //--------------------------------------------------------------------------
    {
      return manager;
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_initial_user(ApEvent term_event, 
                                         const RegionUsage &usage,
                                         const FieldMask &user_mask,
                                         IndexSpaceExpression *user_expr,
                                         const UniqueID op_id,
                                         const unsigned index)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // We don't use field versions for doing interference tests on
      // reductions so there is no need to record it
      PhysicalUser *user = 
        new PhysicalUser(usage, user_expr, op_id, index, false/*copy*/);
      user->add_reference();
      add_physical_user(user, IS_READ_ONLY(usage), term_event, user_mask);
      initial_user_events.insert(term_event);
      // Don't need to actual launch a collection task, destructor
      // will handle this case
      outstanding_gc_events.insert(term_event);
    }

    //--------------------------------------------------------------------------
    ApEvent ReductionView::register_user(const RegionUsage &usage,
                                         const FieldMask &user_mask,
                                         IndexSpaceExpression *user_expr,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         ApEvent term_event,
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
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        // If we're not the logical owner send a message there 
        // to do the analysis and provide a user event to trigger
        // with the precondition
        ApUserEvent ready_event = Runtime::create_ap_user_event();
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(user_mask);
          user_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(term_event);
          rez.serialize(ready_event);
          rez.serialize(applied_event);
        }
        // Add a remote valid reference that will be removed by 
        // the receiver once the changes have been applied
        WrapperReferenceMutator mutator(applied_events);
        add_base_valid_ref(REMOTE_DID_REF, &mutator);
        runtime->send_view_register_user(logical_owner, rez);
        applied_events.insert(applied_event);
        return ready_event;
      }
      else
      {
        std::set<ApEvent> wait_on_events;
        ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          wait_on_events.insert(start_use_event);
        if (IS_READ_ONLY(usage))
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reading_preconditions(user_mask, user_expr,op_id,wait_on_events);
        }
        else
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reducing_preconditions(user_mask,user_expr,op_id,wait_on_events);
        }
        // Add our local user
        const bool issue_collect = add_user(usage, user_expr, user_mask,
           term_event, op_id, index, false/*copy*/, applied_events, trace_info);
        // Launch the garbage collection task, if it doesn't exist
        // then the user wasn't registered anyway, see add_local_user
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(term_event, &mutator);
        }
        if (!wait_on_events.empty())
          return Runtime::merge_events(&trace_info, wait_on_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReductionView::find_copy_preconditions(bool reading,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            CopyFillAggregator &aggregator,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(&aggregator);
          rez.serialize(ready_event);
        }
        runtime->send_view_find_copy_preconditions_request(logical_owner, rez);
        return ready_event;
      }
      else
      {
        EventFieldExprs preconditions;
        ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          preconditions[start_use_event].insert(copy_expr, copy_mask);
        if (reading)
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reading_preconditions(copy_mask, copy_expr, op_id,preconditions);
        }
        else
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reducing_preconditions(copy_mask, copy_expr,op_id,preconditions);
        }
        // Return any preconditions we found to the aggregator
        if (!preconditions.empty())
          aggregator.record_preconditions(this, reading, preconditions);
        // We're done with the analysis
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_copy_preconditions_remote(bool reading,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            EventFieldExprs &preconditions,
                                            const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(preconditions.empty());
#endif
      ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        preconditions[start_use_event].insert(copy_expr, copy_mask);
      if (reading)
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_reading_preconditions(copy_mask, copy_expr, op_id, preconditions);
      }
      else
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_reducing_preconditions(copy_mask, copy_expr, op_id, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_copy_user(bool reading, ApEvent term_event, 
                                      const FieldMask &copy_mask,
                                      IndexSpaceExpression *copy_expr,
                                      UniqueID op_id, unsigned index,
                                      std::set<RtEvent> &applied_events,
                                      const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
#ifdef DEBUG_LEGION
        // Don't support tracing for this case yet
        assert(!trace_info.recording);
#endif
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(term_event);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(applied_event);
        }
        // Add a remote valid reference that will be removed by 
        // the receiver once the changes have been applied
        WrapperReferenceMutator mutator(applied_events);
        add_base_valid_ref(REMOTE_DID_REF, &mutator);
        runtime->send_view_add_copy_user(logical_owner, rez);
        applied_events.insert(applied_event);
      }
      else
      {
        const RegionUsage usage(reading ? READ_ONLY : REDUCE, 
                                EXCLUSIVE, manager->redop);
        const bool issue_collect = add_user(usage, copy_expr, copy_mask,
            term_event, op_id, index, true/*copy*/, applied_events, trace_info);
        // Launch the garbage collection task, if it doesn't exist
        // then the user wasn't registered anyway, see add_local_user
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(term_event, &mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reducing_preconditions(const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               UniqueID op_id,
                                               std::set<ApEvent> &wait_on)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (EventFieldUsers::const_iterator uit = reading_users.begin();
            uit != reading_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          if (it->first->op_id == op_id)
            continue;
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          // Once we have one event precondition we are done
          wait_on.insert(uit->first);
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reading_preconditions(const FieldMask &user_mask,
                                              IndexSpaceExpression *user_expr,
                                              UniqueID op_id,
                                              std::set<ApEvent> &wait_on)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (EventFieldUsers::const_iterator uit = reduction_users.begin();
            uit != reduction_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          if (it->first->op_id == op_id)
            continue;
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          // Once we have one event precondition we are done
          wait_on.insert(uit->first);
          break;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reducing_preconditions(const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               UniqueID op_id,
                                               EventFieldExprs &preconditions)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (EventFieldUsers::const_iterator uit = reading_users.begin();
            uit != reading_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        EventFieldExprs::iterator event_finder = preconditions.find(uit->first);
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          if (it->first->op_id == op_id)
            continue;
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          // Have a precondition so we need to record it
          if (event_finder == preconditions.end())
          {
            preconditions[uit->first].insert(expr_overlap, overlap);
            event_finder = preconditions.find(uit->first);
          }
          else
          {
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              event_finder->second.find(expr_overlap);
            if (finder == event_finder->second.end())
              event_finder->second.insert(expr_overlap, overlap);
            else
              finder.merge(overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reading_preconditions(const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               UniqueID op_id,
                                               EventFieldExprs &preconditions)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      for (EventFieldUsers::const_iterator uit = reduction_users.begin();
            uit != reduction_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        EventFieldExprs::iterator event_finder = preconditions.find(uit->first);
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          if (it->first->op_id == op_id)
            continue;
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          // Have a precondition so we need to record it
          if (event_finder == preconditions.end())
          {
            preconditions[uit->first].insert(expr_overlap, overlap);
            event_finder = preconditions.find(uit->first);
          }
          else
          {
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              event_finder->second.find(expr_overlap);
            if (finder == event_finder->second.end())
              event_finder->second.insert(expr_overlap, overlap);
            else
              finder.merge(overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool ReductionView::add_user(const RegionUsage &usage,
                                 IndexSpaceExpression *user_expr,
                                 const FieldMask &user_mask, ApEvent term_event,
                                 UniqueID op_id, unsigned index, bool copy_user,
                                 std::set<RtEvent> &applied_events,
                                 const PhysicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      PhysicalUser *new_user = 
        new PhysicalUser(usage, user_expr, op_id, index, copy_user);
      new_user->add_reference();
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      add_physical_user(new_user, IS_READ_ONLY(usage), term_event, user_mask);
      // If we're tracing we defer adding these events until the trace
      // capture is complete so we can get a full set of preconditions
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
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_physical_user(PhysicalUser *user, bool reading,
                                          ApEvent term_event, 
                                          const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // Better already be holding the lock
      EventUsers &event_users = reading ? 
        reading_users[term_event] : reduction_users[term_event];
#ifdef DEBUG_LEGION
      assert(event_users.find(user) == event_users.end());
#endif
      event_users.insert(user, user_mask);
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
        EventFieldUsers::iterator finder = reduction_users.find(term_event);
        if (finder != reduction_users.end())
        {
          for (EventUsers::const_iterator it = finder->second.begin();
                it != finder->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
          reduction_users.erase(finder);
        }
        finder = reading_users.find(term_event);
        if (finder != reading_users.end())
        {
          for (EventUsers::const_iterator it = finder->second.begin();
                it != finder->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
          reading_users.erase(finder);
        }
        outstanding_gc_events.erase(event_finder);
      }
    } 
 
    //--------------------------------------------------------------------------
    void ReductionView::copy_to(const FieldMask &copy_mask,
                                std::vector<CopySrcDstField> &dst_fields,
                                CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // Get the destination fields for this copy
      if (across_helper == NULL)
        manager->find_field_offsets(copy_mask, dst_fields);
      else
        across_helper->compute_across_offsets(copy_mask, dst_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::copy_from(const FieldMask &copy_mask,
                                  std::vector<CopySrcDstField> &src_fields)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(FieldMask::pop_count(copy_mask) == 1); // only one field
#endif
      manager->find_field_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        manager->add_nested_gc_ref(did, mutator);
      else
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_owner())
        manager->remove_nested_gc_ref(did, mutator);
      else
        send_remote_gc_decrement(owner_space, RtEvent::NO_RT_EVENT, mutator);
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
#endif
      // Don't take the lock, it's alright to have duplicate sends
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(manager->did);
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
      AddressSpaceID owner_space;
      derez.deserialize(owner_space);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      UniqueID context_uid;
      derez.deserialize(context_uid);

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
        view = new(location) ReductionView(runtime->forest, did, owner_space, 
                                           logical_owner, red_manager,
                                           context_uid, false/*register now*/);
      else
        view = new ReductionView(runtime->forest, did, owner_space,
                                 logical_owner, red_manager, 
                                 context_uid, false/*register now*/);
      // Only register after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

  }; // namespace Internal 
}; // namespace Legion

