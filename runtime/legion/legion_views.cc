/* Copyright 2022 Stanford University, NVIDIA Corporation
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
    InstanceView::InstanceView(RegionTreeForest *ctx, DistributedID did,
                               PhysicalManager *man, AddressSpaceID owner_sp,
                               AddressSpaceID log_own,
                               UniqueID own_ctx, bool register_now)
      : LogicalView(ctx, did, owner_sp, register_now), manager(man),
        owner_context(own_ctx), logical_owner(log_own)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      // Keep the manager from being collected
      manager->add_nested_resource_ref(did);
    }

    //--------------------------------------------------------------------------
    InstanceView::~InstanceView(void)
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
    }

#ifdef ENABLE_VIEW_REPLICATION
    //--------------------------------------------------------------------------
    void InstanceView::process_replication_request(AddressSpaceID source,
                                                  const FieldMask &request_mask,
                                                  RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InstanceView::process_replication_response(RtUserEvent done_event,
                                                    Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void InstanceView::process_replication_removal(AddressSpaceID source,
                                                  const FieldMask &removal_mask)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }
#endif // ENABLE_VIEW_REPLICATION

    //--------------------------------------------------------------------------
    void InstanceView::find_atomic_reservations(const FieldMask &mask,
                                 Operation *op, const unsigned index, bool excl)
    //--------------------------------------------------------------------------
    {
      std::vector<Reservation> reservations(mask.pop_count());
      find_field_reservations(mask, reservations);
      for (unsigned idx = 0; idx < reservations.size(); idx++)
        op->update_atomic_locks(index, reservations[idx], excl);
    } 

    //--------------------------------------------------------------------------
    void InstanceView::find_field_reservations(const FieldMask &mask,
                                         std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mask.pop_count() == reservations.size());
#endif
      unsigned offset = 0;
      if (is_owner())
      {
        AutoLock v_lock(view_lock);
        for (int idx = mask.find_first_set(); idx >= 0;
              idx = mask.find_next_set(idx+1))
        {
          std::map<unsigned,Reservation>::const_iterator finder = 
            atomic_reservations.find(idx);
          if (finder == atomic_reservations.end())
          {
            // Make a new reservation and add it to the set
            Reservation handle = Reservation::create_reservation();
            atomic_reservations[idx] = handle;
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
          AutoLock v_lock(view_lock, 1, false);
          for (int idx = mask.find_first_set(); idx >= 0;
                idx = mask.find_next_set(idx+1))
          {
            std::map<unsigned,Reservation>::const_iterator finder = 
              atomic_reservations.find(idx);
            if (finder == atomic_reservations.end())
              needed_fields.set_bit(idx);
            else
              reservations[offset++] = finder->second;
          }
        }
        if (!!needed_fields)
        {
          RtUserEvent wait_on = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(needed_fields);
            rez.serialize(wait_on);
          }
          runtime->send_atomic_reservation_request(owner_space, rez);
          wait_on.wait();
          // Now retake the lock and get the remaining reservations
          AutoLock v_lock(view_lock, 1, false);
          for (int idx = needed_fields.find_first_set(); idx >= 0;
                idx = needed_fields.find_next_set(idx+1))
          {
            std::map<unsigned,Reservation>::const_iterator finder =
              atomic_reservations.find(idx);
#ifdef DEBUG_LEGION
            assert(finder != atomic_reservations.end());
#endif
            reservations[offset++] = finder->second;
          }
        }
      }
#ifdef DEBUG_LEGION
      assert(offset == reservations.size());
#endif
      // Sort them before returning
      if (reservations.size() > 1)
        std::sort(reservations.begin(), reservations.end());
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_send_atomic_reservation_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask needed_fields;
      derez.deserialize(needed_fields);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      InstanceView *target = dynamic_cast<InstanceView*>(dc);
      assert(target != NULL);
#else
      InstanceView *target = static_cast<InstanceView*>(dc);
#endif
      std::vector<Reservation> reservations(needed_fields.pop_count());
      target->find_field_reservations(needed_fields, reservations);
      Serializer rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(needed_fields);
        for (unsigned idx = 0; idx < reservations.size(); idx++)
          rez.serialize(reservations[idx]);
        rez.serialize(to_trigger);
      }
      runtime->send_atomic_reservation_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void InstanceView::update_field_reservations(const FieldMask &mask,
                                   const std::vector<Reservation> &reservations)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(mask.pop_count() == reservations.size());
#endif
      unsigned offset = 0;
      AutoLock v_lock(view_lock);
      for (int idx = mask.find_first_set(); idx >= 0;
            idx = mask.find_next_set(idx+1))
        atomic_reservations[idx] = reservations[offset++];
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_send_atomic_reservation_response(
                                          Runtime *runtime, Deserializer &derez)
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
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      InstanceView *target = dynamic_cast<InstanceView*>(dc);
      assert(target != NULL);
#else
      InstanceView *target = static_cast<InstanceView*>(dc);
#endif
      target->update_field_reservations(mask, reservations);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_register_user(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      std::set<RtEvent> ready_events;
      RtEvent ready;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);
      if (ready.exists())
        ready_events.insert(ready);

      RegionUsage usage;
      derez.deserialize(usage);
      FieldMask user_mask;
      derez.deserialize(user_mask);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode *user_expr = runtime->forest->get_node(handle);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtEvent collect_event;
      derez.deserialize(collect_event);
      ApUserEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      const PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime, ready_events);

      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      std::set<RtEvent> applied_events;
      ApEvent pre = inst_view->register_user(usage, user_mask, user_expr,
                                             op_id, index, term_event,
                                             collect_event, applied_events, 
                                             trace_info, source);
      if (ready_event.exists())
        Runtime::trigger_event(&trace_info, ready_event, pre);
      if (!applied_events.empty())
      {
        const RtEvent precondition = Runtime::merge_events(applied_events);
        Runtime::trigger_event(applied_event, precondition);
        // Send back a response to the source removing the remote valid ref
        if (inst_view->is_logical_owner())
          inst_view->send_remote_valid_decrement(source, NULL, precondition);
      }
      else
      {
        Runtime::trigger_event(applied_event);
        // Send back a response to the source removing the remote valid ref
        if (inst_view->is_logical_owner())
          inst_view->send_remote_valid_decrement(source);
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
      ReductionOpID redop;
      derez.deserialize(redop);
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
      bool trace_recording;
      derez.deserialize(trace_recording);

#ifdef NON_AGGRESSIVE_AGGREGATORS
      // There are often many parallel copy requests coming from remote nodes
      // which could be handled in parallel, so if we always launch a meta task 
      // to handle these since they are often expensive and we don't want to
      // block the virtual channel. Note that we can't do this with methods
      // that add users, but only non-modifying calls like this
      DeferFindCopyPreconditionArgs args(view, reading, trace_recording, redop,
                                         copy_mask, copy_expr, op_id, index, 
                                         source, remote_aggregator, done_event);
      // One-up the message priority here to keep us ahead of any other
      // messages which might have been enqueued.
      runtime->issue_runtime_meta_task(args,LG_LATENCY_RESPONSE_PRIORITY,ready);
#else
      // This blocks the virtual channel, but keeps queries in-order 
      // with respect to updates from the same node which is necessary
      // for preventing cycles in the realm event graph
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      InstanceView *inst_view = view->as_instance_view();
      EventFieldExprs preconditions;
      inst_view->find_copy_preconditions_remote(reading, redop, copy_mask, 
          copy_expr, op_id, index, preconditions, trace_recording, source);
      // Send back the response unless the preconditions are empty in
      // which case we can just trigger the done event
      if (!preconditions.empty())
      {
        // Pack up the response and send it back
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(inst_view->did);
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
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_find_copy_pre_request(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferFindCopyPreconditionArgs *dargs = 
        (const DeferFindCopyPreconditionArgs*)args;
#ifdef DEBUG_LEGION
      assert(dargs->view->is_instance_view());
#endif
      InstanceView *inst_view = dargs->view->as_instance_view();
      EventFieldExprs preconditions;
      inst_view->find_copy_preconditions_remote(dargs->reading, dargs->redop,
          *dargs->copy_mask, dargs->copy_expr, dargs->op_id, dargs->index, 
          preconditions, dargs->trace_recording, dargs->source);
      // Send back the response unless the preconditions are empty in
      // which case we can just trigger the done event
      if (!preconditions.empty())
      {
        // Pack up the response and send it back
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize(inst_view->did);
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
              it->first->pack_expression(rez, dargs->source);
              rez.serialize(it->second);
            }
          }
          rez.serialize(dargs->aggregator);
          rez.serialize<bool>(dargs->reading);
          rez.serialize(dargs->done_event);
        }
        runtime->send_view_find_copy_preconditions_response(dargs->source, rez);
      }
      else
        Runtime::trigger_event(dargs->done_event);
      // Clean up the mask we allocated
      delete dargs->copy_mask;
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
      ReductionOpID redop;
      derez.deserialize(redop);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtEvent collect_event;
      derez.deserialize(collect_event);
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
      bool trace_recording;
      derez.deserialize(trace_recording);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();

      std::set<RtEvent> applied_events;
      inst_view->add_copy_user(reading,redop,term_event,collect_event,copy_mask,
          copy_expr, op_id, index, applied_events, trace_recording, source);
      if (!applied_events.empty())
      {
        const RtEvent precondition = Runtime::merge_events(applied_events);
        Runtime::trigger_event(applied_event, precondition);
        // Send back a response to the source removing the remote valid ref
        if (inst_view->is_logical_owner())
          inst_view->send_remote_valid_decrement(source, NULL, precondition);
      }
      else
      {
        Runtime::trigger_event(applied_event);
        // Send back a response to the source removing the remote valid ref
        if (inst_view->is_logical_owner())
          inst_view->send_remote_valid_decrement(source);
      }
    }

#ifdef ENABLE_VIEW_REPLICATION
    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_replication_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      FieldMask request_mask;
      derez.deserialize(request_mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_replication_request(source, request_mask, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_replication_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      RtUserEvent done_event;
      derez.deserialize(done_event);
      
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_replication_response(done_event, derez);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void InstanceView::handle_view_replication_removal(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready = RtEvent::NO_RT_EVENT;
      LogicalView *view = runtime->find_or_request_logical_view(did, ready);

      FieldMask removal_mask;
      derez.deserialize(removal_mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
#ifdef DEBUG_LEGION
      assert(view->is_instance_view());
#endif
      InstanceView *inst_view = view->as_instance_view();
      inst_view->process_replication_removal(source, removal_mask);
      // Trigger the done event now that we are done
      Runtime::trigger_event(done_event);
    }
#endif // ENABLE_VIEW_REPLICATION

    /////////////////////////////////////////////////////////////
    // CollectableView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void CollectableView::defer_collect_user(PhysicalManager *manager,
                 ApEvent term_event, RtEvent collect, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // The runtime will add the gc reference to this view when necessary
      std::set<ApEvent> to_collect;
      bool add_ref = false;
      bool remove_ref = false;
      manager->defer_collect_user(this, term_event, collect,
                                  to_collect, add_ref, remove_ref);
      if (add_ref)
        add_collectable_reference(mutator);
      if (!to_collect.empty())
        collect_users(to_collect); 
      if (remove_ref && remove_collectable_reference(mutator))
        delete this;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectableView::handle_deferred_collect(
                                            CollectableView *view, 
                                            const std::set<ApEvent> &to_collect)
    //--------------------------------------------------------------------------
    {
      view->collect_users(to_collect);
      // Then remove the gc reference on the object
      if (view->remove_collectable_reference(NULL))
        delete view;
    }

    /////////////////////////////////////////////////////////////
    // ExprView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExprView::ExprView(RegionTreeForest *ctx, PhysicalManager *man, 
                       InstanceView *view, IndexSpaceExpression *exp) 
      : context(ctx), manager(man), inst_view(view),
        view_expr(exp), view_volume(view_expr->get_volume()),
#if defined(DEBUG_LEGION_GC) || defined(LEGION_GC)
        view_did(view->did),
#endif
        invalid_fields(FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES))
    //--------------------------------------------------------------------------
    {
      view_expr->add_nested_expression_reference(view->did);
    }

    //--------------------------------------------------------------------------
    ExprView::ExprView(const ExprView &rhs)
      : context(rhs.context), manager(rhs.manager), inst_view(rhs.inst_view),
        view_expr(rhs.view_expr), view_volume(rhs.view_volume)
#if defined(DEBUG_LEGION_GC) || defined(LEGION_GC)
        , view_did(rhs.view_did)
#endif
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ExprView::~ExprView(void)
    //--------------------------------------------------------------------------
    {
#if defined(DEBUG_LEGION_GC) || defined(LEGION_GC)
      if (view_expr->remove_nested_expression_reference(view_did))
        delete view_expr;
#else
      // We can lie about the did here since its not actually used
      if (view_expr->remove_nested_expression_reference(0/*bogus did*/))
        delete view_expr;
#endif
      if (!subviews.empty())
      {
        for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
              it != subviews.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
      }
      // If we have any current or previous users filter them out now
      if (!current_epoch_users.empty())
      {
        for (EventFieldUsers::const_iterator eit = current_epoch_users.begin();
              eit != current_epoch_users.end(); eit++)
        {
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
        }
        current_epoch_users.clear();
      }
      if (!previous_epoch_users.empty())
      {
        for (EventFieldUsers::const_iterator eit = previous_epoch_users.begin();
              eit != previous_epoch_users.end(); eit++)
        {
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
        }
        previous_epoch_users.clear();
      }
    }

    //--------------------------------------------------------------------------
    ExprView& ExprView::operator=(const ExprView &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExprView::verify_current_to_filter(
                 const FieldMask &dominated, EventFieldUsers &current_to_filter)
    //--------------------------------------------------------------------------
    {
      if (!!dominated)
      {
        for (EventFieldUsers::iterator eit = current_to_filter.begin();
              eit != current_to_filter.end(); /*nothing*/)
        {
          const FieldMask non_dominated = 
            eit->second.get_valid_mask() - dominated;
          // If everything was actually dominated we can keep going
          if (!non_dominated)
          {
            eit++;
            continue;
          }
          // If no fields were dominated we can just remove this
          if (non_dominated == eit->second.get_valid_mask())
          {
            EventFieldUsers::iterator to_delete = eit++;
            current_to_filter.erase(to_delete);
            continue;
          }
          // Otherwise do the actuall overlapping test
          std::vector<PhysicalUser*> to_delete; 
          for (FieldMaskSet<PhysicalUser>::iterator it =
                eit->second.begin(); it != eit->second.end(); it++)
          {
            it.filter(non_dominated);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!eit->second.tighten_valid_mask())
          {
            EventFieldUsers::iterator to_delete = eit++;
            current_to_filter.erase(to_delete);
          }
          else
          {
            for (std::vector<PhysicalUser*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              eit->second.erase(*it);
            eit++;
          }
        }
      }
      else
        current_to_filter.clear();
    }

    //--------------------------------------------------------------------------
    void ExprView::find_user_preconditions(const RegionUsage &usage,
                                           IndexSpaceExpression *user_expr,
                                           const bool user_dominates,
                                           const FieldMask &user_mask,
                                           ApEvent term_event,
                                           UniqueID op_id, unsigned index,
                                           std::set<ApEvent> &preconditions,
                                           const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(Internal::implicit_runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_PRECONDITIONS_CALL);
      FieldMask dominated;
      std::set<ApEvent> dead_events; 
      EventFieldUsers current_to_filter, previous_to_filter;
      // Perform the analysis with a read-only lock
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (user_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(usage, user_mask, user_expr,
                                       term_event, op_id, index, 
                                       user_dominates, preconditions, 
                                       dead_events, current_to_filter, 
                                       observed, non_dominated,trace_recording);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = user_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(usage, previous_mask, user_expr,
                                          term_event, op_id, index, 
                                          user_dominates, preconditions,
                                          dead_events, trace_recording);
          }
        }
        else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(usage, user_mask, user_expr,
                                       term_event, op_id, index, 
                                       user_dominates, preconditions, 
                                       dead_events, current_to_filter, 
                                       observed, non_dominated,trace_recording);
#ifdef DEBUG_LEGION
            assert(!observed);
            assert(current_to_filter.empty());
#endif
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(usage, user_mask, user_expr,
                                        term_event, op_id, index,
                                        user_dominates, preconditions,
                                        dead_events, trace_recording);
        }
      } 
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!trace_recording && (!dead_events.empty() || 
           !previous_to_filter.empty() || !current_to_filter.empty()))
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
      // Then see if there are any users below that we need to traverse
      if (!subviews.empty() && 
          !(subviews.get_valid_mask() * user_mask))
      {
        FieldMaskSet<ExprView> to_traverse;
        std::map<ExprView*,IndexSpaceExpression*> traverse_exprs;
        for (FieldMaskSet<ExprView>::const_iterator it = 
              subviews.begin(); it != subviews.end(); it++)
        {
          FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          // If we've already determined the user dominates
          // then we don't even have to do this test
          if (user_dominates)
          {
            to_traverse.insert(it->first, overlap);
            continue;
          }
          if (it->first->view_expr == user_expr)
          {
            to_traverse.insert(it->first, overlap);
            traverse_exprs[it->first] = user_expr;
            continue;
          }
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(it->first->view_expr, user_expr);
          if (!expr_overlap->is_empty())
          {
            to_traverse.insert(it->first, overlap);
            traverse_exprs[it->first] = expr_overlap;
          }
        }
        if (!to_traverse.empty())
        {
          if (user_dominates)
          {
            for (FieldMaskSet<ExprView>::const_iterator it = 
                  to_traverse.begin(); it != to_traverse.end(); it++)
              it->first->find_user_preconditions(usage, it->first->view_expr,
                                    true/*dominate*/, it->second, term_event,
                                    op_id, index,preconditions,trace_recording);
          }
          else
          {
            for (FieldMaskSet<ExprView>::const_iterator it = 
                  to_traverse.begin(); it != to_traverse.end(); it++)
            {
              IndexSpaceExpression *intersect = traverse_exprs[it->first];
              const bool user_dominates = 
                (intersect->expr_id == it->first->view_expr->expr_id) ||
                (intersect->get_volume() == it->first->view_volume);
              it->first->find_user_preconditions(usage, intersect, 
                            user_dominates, it->second, term_event, 
                            op_id, index, preconditions, trace_recording);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_copy_preconditions(const RegionUsage &usage,
                                           IndexSpaceExpression *copy_expr,
                                           const bool copy_dominates,
                                           const FieldMask &copy_mask,
                                           UniqueID op_id, unsigned index,
                                           EventFieldExprs &preconditions,
                                           const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(Internal::implicit_runtime, 
                        MATERIALIZED_VIEW_FIND_LOCAL_COPY_PRECONDITIONS_CALL);
      FieldMask dominated;
      std::set<ApEvent> dead_events; 
      EventFieldUsers current_to_filter, previous_to_filter;
      // Do the first pass with a read-only lock on the events
      {
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Check to see if we dominate when doing this analysis and
        // can therefore filter or whether we are just intersecting
        // Do the local analysis
        if (copy_dominates)
        {
          // We dominate in this case so we can do filtering
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(usage, copy_mask, copy_expr, 
                                       op_id, index, copy_dominates,
                                       preconditions, dead_events, 
                                       current_to_filter, observed, 
                                       non_dominated, trace_recording);
            if (!!observed)
              dominated = observed - non_dominated;
          }
          if (!previous_epoch_users.empty())
          {
            if (!!dominated)
              find_previous_filter_users(dominated, previous_to_filter);
            const FieldMask previous_mask = copy_mask - dominated;
            if (!!previous_mask)
              find_previous_preconditions(usage, previous_mask, copy_expr,
                                          op_id, index, copy_dominates,
                                          preconditions, dead_events,
                                          trace_recording);
          }
        }
        else
        {
          if (!current_epoch_users.empty())
          {
            FieldMask observed, non_dominated;
            find_current_preconditions(usage, copy_mask, copy_expr, 
                                       op_id, index, copy_dominates,
                                       preconditions, dead_events, 
                                       current_to_filter, observed, 
                                       non_dominated, trace_recording);
#ifdef DEBUG_LEGION
            assert(!observed);
            assert(current_to_filter.empty());
#endif
          }
          if (!previous_epoch_users.empty())
            find_previous_preconditions(usage, copy_mask, copy_expr,
                                        op_id, index, copy_dominates,
                                        preconditions, dead_events,
                                        trace_recording);
        }
      }
      // It's possible that we recorded some users for fields which
      // are not actually fully dominated, if so we need to prune them
      // otherwise we can get into issues of soundness
      if (!current_to_filter.empty())
        verify_current_to_filter(dominated, current_to_filter);
      if (!trace_recording && (!dead_events.empty() || 
           !previous_to_filter.empty() || !current_to_filter.empty()))
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
      // Then see if there are any users below that we need to traverse
      if (!subviews.empty() && 
          !(subviews.get_valid_mask() * copy_mask))
      {
        for (FieldMaskSet<ExprView>::const_iterator it = 
              subviews.begin(); it != subviews.end(); it++)
        {
          FieldMask overlap = it->second & copy_mask;
          if (!overlap)
            continue;
          // If the copy dominates then we don't even have
          // to do the intersection test
          if (copy_dominates)
          {
            it->first->find_copy_preconditions(usage, it->first->view_expr,
                                    true/*dominate*/, overlap, op_id, index,
                                    preconditions, trace_recording);
            continue;
          }
          if (it->first->view_expr == copy_expr)
          {
            it->first->find_copy_preconditions(usage, copy_expr,
                                    true/*dominate*/, overlap, op_id, index,
                                    preconditions, trace_recording);
            continue;
          }
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(it->first->view_expr, copy_expr);
          if (!expr_overlap->is_empty())
          {
            const bool copy_dominates = 
              (expr_overlap->expr_id == it->first->view_expr->expr_id) ||
              (expr_overlap->get_volume() == it->first->view_volume);
            it->first->find_copy_preconditions(usage, expr_overlap, 
                              copy_dominates, overlap, op_id, 
                              index, preconditions, trace_recording);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    ExprView* ExprView::find_congruent_view(IndexSpaceExpression *expr) const
    //--------------------------------------------------------------------------
    {
      // Handle the base case first
      if ((expr == view_expr) || (expr->get_volume() == view_volume))
        return const_cast<ExprView*>(this);
      for (FieldMaskSet<ExprView>::const_iterator it = 
            subviews.begin(); it != subviews.end(); it++)
      {
        if (it->first->view_expr == expr)
          return it->first;
        IndexSpaceExpression *overlap =
          context->intersect_index_spaces(expr, it->first->view_expr);
        const size_t overlap_volume = overlap->get_volume();
        if (overlap_volume == 0)
          continue;
        // See if we dominate or just intersect
        if (overlap_volume == expr->get_volume())
        {
          // See if we strictly dominate or whether they are equal
          if (overlap_volume < it->first->view_volume)
          {
            ExprView *result = it->first->find_congruent_view(expr);
            if (result != NULL)
              return result;
          }
          else // Otherwise we're the same 
            return it->first;
        }
      }
      return NULL;
    }

    //--------------------------------------------------------------------------
    void ExprView::insert_subview(ExprView *subview, FieldMask &subview_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(this != subview);
#endif
      // Iterate over all subviews and see which ones we dominate and which
      // ones dominate the subview
      if (!subviews.empty() && !(subviews.get_valid_mask() * subview_mask))
      {
        bool need_tighten = true;
        std::vector<ExprView*> to_delete;
        FieldMaskSet<ExprView> dominating_subviews;
        for (FieldMaskSet<ExprView>::iterator it = 
              subviews.begin(); it != subviews.end(); it++)
        {
          // See if we intersect on fields
          FieldMask overlap_mask = it->second & subview_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression *overlap =
            context->intersect_index_spaces(subview->view_expr,
                                            it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == 0)
            continue;
          // See if we dominate or just intersect
          if (overlap_volume == subview->view_volume)
          {
#ifdef DEBUG_LEGION
            // Should only strictly dominate if they were congruent
            // then we wouldn't be inserting in the first place
            assert(overlap_volume < it->first->view_volume);
#endif
            // Dominator so we can just continue traversing
            dominating_subviews.insert(it->first, overlap_mask);
          }
          else if (overlap_volume == it->first->view_volume)
          {
#ifdef DEBUG_LEGION
            assert(overlap_mask * dominating_subviews.get_valid_mask());
#endif
            // We dominate this view so we can just pull it 
            // in underneath of us now
            it.filter(overlap_mask);
            subview->insert_subview(it->first, overlap_mask);
            need_tighten = true;
            // See if we need to remove this subview
            if (!it->second)
              to_delete.push_back(it->first);
          }
          // Otherwise it's just a normal intersection
        }
        // See if we had any dominators
        if (!dominating_subviews.empty())
        {
          if (dominating_subviews.size() > 1)
          {
            // We need to deduplicate finding or making the new ExprView
            // First check to see if we have it already in one sub-tree
            // If not, we'll pick the one with the smallest bounding volume
            LegionMap<std::pair<size_t/*volume*/,ExprView*>,FieldMask>
              sorted_subviews;
            for (FieldMaskSet<ExprView>::const_iterator it = 
                  dominating_subviews.begin(); it != 
                  dominating_subviews.end(); it++)
            {
              FieldMask overlap = it->second;
              // Channeling Tuco here
              it->first->find_tightest_subviews(subview->view_expr, overlap,
                                                sorted_subviews);
            }
            for (LegionMap<std::pair<size_t,ExprView*>,FieldMask>::
                  const_iterator it = sorted_subviews.begin(); it !=
                  sorted_subviews.end(); it++)
            {
              FieldMask overlap = it->second & subview_mask;
              if (!overlap)
                continue;
              subview_mask -= overlap;
              it->first.second->insert_subview(subview, overlap);
              if (!subview_mask || 
                  (subview_mask * dominating_subviews.get_valid_mask()))
                break;
            }
#ifdef DEBUG_LEGION
            assert(subview_mask * dominating_subviews.get_valid_mask());
#endif
          }
          else
          {
            FieldMaskSet<ExprView>::const_iterator first = 
              dominating_subviews.begin();
            FieldMask dominated_mask = first->second; 
            subview_mask -= dominated_mask;
            first->first->insert_subview(subview, dominated_mask);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<ExprView*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            subviews.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
        }
        if (need_tighten)
          subviews.tighten_valid_mask();
      }
      // If we make it here and there are still fields then we need to 
      // add it locally
      if (!!subview_mask && subviews.insert(subview, subview_mask))
        subview->add_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::find_tightest_subviews(IndexSpaceExpression *expr,
                                          FieldMask &expr_mask,
                                          LegionMap<std::pair<size_t,ExprView*>,
                                                     FieldMask> &bounding_views)
    //--------------------------------------------------------------------------
    {
      if (!subviews.empty() && !(expr_mask * subviews.get_valid_mask()))
      {
        FieldMask dominated_mask;
        for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
              it != subviews.end(); it++)
        {
          // See if we intersect on fields
          FieldMask overlap_mask = it->second & expr_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression *overlap =
            context->intersect_index_spaces(expr, it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == 0)
            continue;
          // See if we dominate or just intersect
          if (overlap_volume == expr->get_volume())
          {
#ifdef DEBUG_LEGION
            // Should strictly dominate otherwise we'd be congruent
            assert(overlap_volume < it->first->view_volume);
#endif
            dominated_mask |= overlap_mask;
            // Continute the traversal
            it->first->find_tightest_subviews(expr,overlap_mask,bounding_views);
          }
        }
        // Remove any dominated fields from below
        if (!!dominated_mask)
          expr_mask -= dominated_mask;
      }
      // If we still have fields then record ourself
      if (!!expr_mask)
      {
        std::pair<size_t,ExprView*> key(view_volume, this);
        bounding_views[key] |= expr_mask;
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::add_partial_user(const RegionUsage &usage,
                                    UniqueID op_id, unsigned index,
                                    FieldMask user_mask,
                                    const ApEvent term_event,
                                    const RtEvent collect_event,
                                    IndexSpaceExpression *user_expr,
                                    const size_t user_volume,
                                    const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // We're going to try to put this user as far down the ExprView tree
      // as we can in order to avoid doing unnecessary intersection tests later
      {
        // Find all the intersecting subviews to see if we can 
        // continue the traversal
        // No need for the view lock anymore since we're protected
        // by the expr_lock at the top of the tree
        //AutoLock v_lock(view_lock,1,false/*exclusive*/); 
        for (FieldMaskSet<ExprView>::const_iterator it = 
              subviews.begin(); it != subviews.end(); it++)
        {
          // If the fields don't overlap then we don't care
          const FieldMask overlap_mask = it->second & user_mask;
          if (!overlap_mask)
            continue;
          IndexSpaceExpression *overlap =
            context->intersect_index_spaces(user_expr, it->first->view_expr);
          const size_t overlap_volume = overlap->get_volume();
          if (overlap_volume == user_volume)
          {
            // Check for the cases where we dominated perfectly
            if (overlap_volume == it->first->view_volume)
            {
#ifdef ENABLE_VIEW_REPLICATION
              PhysicalUser *dominate_user = new PhysicalUser(usage,
                  it->first->view_expr, op_id, index, collect_event,
                  true/*copy*/, true/*covers*/);
#else
              PhysicalUser *dominate_user = new PhysicalUser(usage,
                  it->first->view_expr,op_id,index,true/*copy*/,true/*covers*/);
#endif
              it->first->add_current_user(dominate_user, term_event, 
                      collect_event, overlap_mask, trace_recording);
            }
            else
            {
              // Continue the traversal on this node
              it->first->add_partial_user(usage, op_id, index, overlap_mask,
                                          term_event, collect_event, user_expr,
                                          user_volume, trace_recording);
            }
            // We only need to record the partial user in one sub-tree
            // where it is dominated in order to be sound
            user_mask -= overlap_mask;
            if (!user_mask)
              break;
          }
          // Otherwise for all other cases we're going to record it here
          // because they don't dominate the user to be recorded
        }
      }
      // If we still have local fields, make a user and record it here
      if (!!user_mask)
      {
#ifdef ENABLE_VIEW_REPLICATION
        PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index,
                                collect_event, true/*copy*/, false/*covers*/);
#else
        PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index,
                                              true/*copy*/, false/*covers*/);
#endif
        add_current_user(user, term_event, collect_event, 
                         user_mask, trace_recording);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::add_current_user(PhysicalUser *user,const ApEvent term_event,
                              RtEvent collect_event, const FieldMask &user_mask,
                              const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      bool issue_collect = true;
      {
        AutoLock v_lock(view_lock);
        EventUsers &event_users = current_epoch_users[term_event];
        if (event_users.insert(user, user_mask))
          user->add_reference();
        else
          issue_collect = false;
      }
      if (issue_collect)
        defer_collect_user(manager, term_event, collect_event);
    }

    //--------------------------------------------------------------------------
    void ExprView::clean_views(FieldMask &valid_mask, 
                               FieldMaskSet<ExprView> &clean_set)
    //--------------------------------------------------------------------------
    {
      // Handle the base case if we already did it
      FieldMaskSet<ExprView>::const_iterator finder = clean_set.find(this);
      if (finder != clean_set.end())
      {
        valid_mask = finder->second;
        return;
      }
      // No need to hold the lock for this part we know that no one
      // is going to be modifying this data structure at the same time
      FieldMaskSet<ExprView> new_subviews;
      std::vector<ExprView*> to_delete;
      for (FieldMaskSet<ExprView>::iterator it = subviews.begin();
            it != subviews.end(); it++)
      {
        FieldMask new_mask;
        it->first->clean_views(new_mask, clean_set);
        // Save this as part of the valid mask without filtering
        valid_mask |= new_mask;
        // Have to make sure to filter this by the previous set of fields 
        // since we could get more than we initially had
        // We also need update the invalid fields if we remove a path
        // to the subview
        if (!!new_mask)
        {
          new_mask &= it->second;
          const FieldMask new_invalid = it->second - new_mask;
          if (!!new_invalid)
          {
#ifdef DEBUG_LEGION
            // Should only have been one path here
            assert(it->first->invalid_fields * new_invalid);
#endif
            it->first->invalid_fields |= new_invalid;
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          // Should only have been one path here
          assert(it->first->invalid_fields * it->second);
#endif
          it->first->invalid_fields |= it->second;
        }
        if (!!new_mask)
          new_subviews.insert(it->first, new_mask);
        else
          to_delete.push_back(it->first);
      }
      subviews.swap(new_subviews);
      if (!to_delete.empty())
      {
        for (std::vector<ExprView*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
          if ((*it)->remove_reference())
            delete (*it);
      }
      AutoLock v_lock(view_lock);
      if (!current_epoch_users.empty())
      {
        for (EventFieldUsers::const_iterator it = 
              current_epoch_users.begin(); it != 
              current_epoch_users.end(); it++)
          valid_mask |= it->second.get_valid_mask();
      }
      if (!previous_epoch_users.empty())
      {
        for (EventFieldUsers::const_iterator it = 
              previous_epoch_users.begin(); it != 
              previous_epoch_users.end(); it++)
          valid_mask |= it->second.get_valid_mask();
      }
      // Save this for the future so we don't need to compute it again
      if (clean_set.insert(this, valid_mask))
        add_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::pack_replication(Serializer &rez,
                                    std::map<PhysicalUser*,unsigned> &indexes,
                                    const FieldMask &pack_mask,
                                    const AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      {
        // Need a read-only lock here to protect against garbage collection
        // tasks coming back through and pruning out current epoch users
        // but we know there are no other modifications happening in parallel
        // because the replicated lock at the top prevents any new users
        // from being added while we're doing this pack
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // Pack the current users
        EventFieldUsers needed_current; 
        for (EventFieldUsers::const_iterator eit = current_epoch_users.begin();
              eit != current_epoch_users.end(); eit++)
        {
          if (eit->second.get_valid_mask() * pack_mask)
            continue;
          FieldMaskSet<PhysicalUser> &needed = needed_current[eit->first];
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            const FieldMask overlap = it->second & pack_mask;
            if (!overlap)
              continue;
            needed.insert(it->first, overlap);
          }
        }
        rez.serialize<size_t>(needed_current.size());
        for (EventFieldUsers::const_iterator eit = needed_current.begin();
              eit != needed_current.end(); eit++)
        {
          rez.serialize(eit->first);
          rez.serialize<size_t>(eit->second.size());
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            // See if we already packed this before or not
            std::map<PhysicalUser*,unsigned>::const_iterator finder = 
              indexes.find(it->first);
            if (finder == indexes.end())
            {
              const unsigned index = indexes.size();
              rez.serialize(index);
              it->first->pack_user(rez, target);
              indexes[it->first] = index;
            }
            else
              rez.serialize(finder->second);
            rez.serialize(it->second);
          }
        }
        // Pack the previous users
        EventFieldUsers needed_previous; 
        for (EventFieldUsers::const_iterator eit = previous_epoch_users.begin();
              eit != previous_epoch_users.end(); eit++)
        {
          if (eit->second.get_valid_mask() * pack_mask)
            continue;
          FieldMaskSet<PhysicalUser> &needed = needed_previous[eit->first];
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            const FieldMask overlap = it->second & pack_mask;
            if (!overlap)
              continue;
            needed.insert(it->first, overlap);
          }
        }
        rez.serialize<size_t>(needed_previous.size());
        for (EventFieldUsers::const_iterator eit = needed_previous.begin();
              eit != needed_previous.end(); eit++)
        {
          rez.serialize(eit->first);
          rez.serialize<size_t>(eit->second.size());
          for (FieldMaskSet<PhysicalUser>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            // See if we already packed this before or not
            std::map<PhysicalUser*,unsigned>::const_iterator finder = 
              indexes.find(it->first);
            if (finder == indexes.end())
            {
              const unsigned index = indexes.size();
              rez.serialize(index);
              it->first->pack_user(rez, target);
              indexes[it->first] = index;
            }
            else
              rez.serialize(finder->second);
            rez.serialize(it->second);
          }
        }
      }
      // Pack the needed subviews no need for a lock here
      // since we know that we're protected by the expr_lock
      // at the top of the tree
      FieldMaskSet<ExprView> needed_subviews;
      for (FieldMaskSet<ExprView>::const_iterator it = 
            subviews.begin(); it != subviews.end(); it++)
      {
        const FieldMask overlap = it->second & pack_mask;
        if (!overlap)
          continue;
        needed_subviews.insert(it->first, overlap);
      }
      rez.serialize<size_t>(needed_subviews.size());
      for (FieldMaskSet<ExprView>::const_iterator it = 
            needed_subviews.begin(); it != needed_subviews.end(); it++)
      {
        it->first->view_expr->pack_expression(rez, target);
        rez.serialize(it->second);
        it->first->pack_replication(rez, indexes, it->second, target);
      }
    }
    
    //--------------------------------------------------------------------------
    void ExprView::unpack_replication(Deserializer &derez, ExprView *root,
                              const AddressSpaceID source,
                              std::map<IndexSpaceExprID,ExprView*> &expr_cache,
                              std::vector<PhysicalUser*> &users)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      std::map<ApEvent,RtEvent> to_collect;
      // Need a read-write lock since we're going to be mutating the structures
      {
        AutoLock v_lock(view_lock);
        size_t num_current;
        derez.deserialize(num_current);
        for (unsigned idx1 = 0; idx1 < num_current; idx1++)
        {
          ApEvent user_event;
          derez.deserialize(user_event);
          FieldMaskSet<PhysicalUser> &current_users = 
            current_epoch_users[user_event];
#ifndef ENABLE_VIEW_REPLICATION
          if (current_users.empty())
            to_collect[user_event] = RtEvent::NO_RT_EVENT;
#endif
          size_t num_users;
          derez.deserialize(num_users);
          for (unsigned idx2 = 0; idx2 < num_users; idx2++)
          {
            unsigned user_index;
            derez.deserialize(user_index);
            if (user_index >= users.size())
            {
#ifdef DEBUG_LEGION
              assert(user_index == users.size());
#endif
              users.push_back(PhysicalUser::unpack_user(derez, context,source));
              // Add a reference to prevent this being deleted
              // before we're done unpacking
              users.back()->add_reference();
#ifdef ENABLE_VIEW_REPLICATION
              to_collect[user_event] = users.back()->collect_event;
#endif
            }
            FieldMask user_mask;
            derez.deserialize(user_mask);
            if (current_users.insert(users[user_index], user_mask))
              users[user_index]->add_reference();
          }
        }
        size_t num_previous;
        derez.deserialize(num_previous);
        for (unsigned idx1 = 0; idx1 < num_previous; idx1++)
        {
          ApEvent user_event;
          derez.deserialize(user_event);
          FieldMaskSet<PhysicalUser> &previous_users = 
            previous_epoch_users[user_event];
#ifndef ENABLE_VIEW_REPLICATION
          if (previous_users.empty())
            to_collect[user_event] = RtEvent::NO_RT_EVENT;
#endif
          size_t num_users;
          derez.deserialize(num_users);
          for (unsigned idx2 = 0; idx2 < num_users; idx2++)
          {
            unsigned user_index;
            derez.deserialize(user_index);
            if (user_index >= users.size())
            {
#ifdef DEBUG_LEGION
              assert(user_index == users.size());
#endif
              users.push_back(PhysicalUser::unpack_user(derez, context,source));
              // Add a reference to prevent this being deleted
              // before we're done unpacking
              users.back()->add_reference();
#ifdef ENABLE_VIEW_REPLICATION
              to_collect[user_event] = users.back()->collect_event;
#endif
            }
            FieldMask user_mask;
            derez.deserialize(user_mask);
            if (previous_users.insert(users[user_index], user_mask))
              users[user_index]->add_reference();
          }
        }
      }
      size_t num_subviews;
      derez.deserialize(num_subviews);
      if (num_subviews > 0)
      {
        for (unsigned idx = 0; idx < num_subviews; idx++)
        {
          IndexSpaceExpression *subview_expr = 
            IndexSpaceExpression::unpack_expression(derez, context, source);
          FieldMask subview_mask;
          derez.deserialize(subview_mask);
          // See if we already have it in the cache
          std::map<IndexSpaceExprID,ExprView*>::const_iterator finder = 
            expr_cache.find(subview_expr->expr_id);
          ExprView *subview = NULL;
          if (finder == expr_cache.end())
          {
            // See if we can find this view in the tree before we make it
            subview = root->find_congruent_view(subview_expr);
            // If it's still NULL then we can make it
            if (subview == NULL)
              subview = new ExprView(context, manager, inst_view, subview_expr);
            expr_cache[subview_expr->expr_id] = subview;
          }
          else
            subview = finder->second;
#ifdef DEBUG_LEGION
          assert(subview != NULL);
#endif
          // Check to see if it needs to be inserted
          if (subview != root)
          {
            FieldMask insert_mask = subview->invalid_fields & subview_mask;
            if (!!insert_mask)
            {
              subview->invalid_fields -= insert_mask;
              root->insert_subview(subview, insert_mask);
            }
          }
          // Continue the unpacking
          subview->unpack_replication(derez, root, source, expr_cache, users);
        }
      }
      if (!to_collect.empty())
      {
        for (std::map<ApEvent,RtEvent>::const_iterator it = 
              to_collect.begin(); it != to_collect.end(); it++)
          defer_collect_user(manager, it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::deactivate_replication(const FieldMask &deactivate_mask)
    //--------------------------------------------------------------------------
    {
      // Traverse any subviews and do the deactivates in those nodes first
      // No need to get the lock here since we're protected by the 
      // exclusive expr_lock at the top of the tree
      // Don't worry about pruning, when we clean the cache after doing
      // this pass then that will also go through and prune out any 
      // expr views which no longer have users in any subtrees
      for (FieldMaskSet<ExprView>::const_iterator it = 
            subviews.begin(); it != subviews.end(); it++)
      {
        const FieldMask overlap = it->second & deactivate_mask;
        if (!overlap)
          continue;
        it->first->deactivate_replication(overlap);
      }
      // Need a read-write lock since we're going to be mutating the structures
      AutoLock v_lock(view_lock);
      // Prune out the current epoch users
      if (!current_epoch_users.empty())
      {
        std::vector<ApEvent> events_to_delete;
        for (EventFieldUsers::iterator eit = current_epoch_users.begin();
              eit != current_epoch_users.end(); eit++)
        {
          if (eit->second.get_valid_mask() * deactivate_mask)
            continue;
          bool need_tighten = false;
          std::vector<PhysicalUser*> to_delete;
          for (FieldMaskSet<PhysicalUser>::iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            if (it->second * deactivate_mask)
              continue;
            need_tighten = true;
            it.filter(deactivate_mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<PhysicalUser*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              eit->second.erase(*it);
              if ((*it)->remove_reference())
                delete (*it);
            }
            if (eit->second.empty())
            {
              events_to_delete.push_back(eit->first);
              continue;
            }
          }
          if (need_tighten)
            eit->second.tighten_valid_mask();
        }
        if (!events_to_delete.empty())
        {
          for (std::vector<ApEvent>::const_iterator it = 
                events_to_delete.begin(); it != events_to_delete.end(); it++)
            current_epoch_users.erase(*it);
        }
      }
      // Prune out the previous epoch users
      if (!previous_epoch_users.empty())
      {
        std::vector<ApEvent> events_to_delete;
        for (EventFieldUsers::iterator eit = previous_epoch_users.begin();
              eit != previous_epoch_users.end(); eit++)
        {
          if (eit->second.get_valid_mask() * deactivate_mask)
            continue;
          bool need_tighten = false;
          std::vector<PhysicalUser*> to_delete;
          for (FieldMaskSet<PhysicalUser>::iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            if (it->second * deactivate_mask)
              continue;
            need_tighten = true;
            it.filter(deactivate_mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<PhysicalUser*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              eit->second.erase(*it);
              if ((*it)->remove_reference())
                delete (*it);
            }
            if (eit->second.empty())
            {
              events_to_delete.push_back(eit->first);
              continue;
            }
          }
          if (need_tighten)
            eit->second.tighten_valid_mask();
        }
        if (!events_to_delete.empty())
        {
          for (std::vector<ApEvent>::const_iterator it = 
                events_to_delete.begin(); it != events_to_delete.end(); it++)
            previous_epoch_users.erase(*it);
        }
      } 
    }

    //--------------------------------------------------------------------------
    void ExprView::add_collectable_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      add_reference();
      // Only the logical owner adds the full GC reference as this is where
      // the actual garbage collection algorithm will take place and we know
      // that we have all the valid gc event users
      if (inst_view->is_logical_owner())
        inst_view->add_base_gc_ref(PENDING_GC_REF, mutator);
      else
        inst_view->add_base_resource_ref(PENDING_GC_REF);
    }

    //--------------------------------------------------------------------------
    bool ExprView::remove_collectable_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (inst_view->is_logical_owner())
      {
        if (inst_view->remove_base_gc_ref(PENDING_GC_REF, mutator))
          delete inst_view;
      }
      else
      {
        if (inst_view->remove_base_resource_ref(PENDING_GC_REF))
          delete inst_view;
      }
      return remove_reference();
    }

    //--------------------------------------------------------------------------
    void ExprView::collect_users(const std::set<ApEvent> &to_collect)
    //--------------------------------------------------------------------------
    {
      AutoLock v_lock(view_lock);
      for (std::set<ApEvent>::const_iterator it = 
            to_collect.begin(); it != to_collect.end(); it++)
        filter_local_users(*it);
    }

    //--------------------------------------------------------------------------
    void ExprView::filter_local_users(ApEvent term_event) 
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      DETAILED_PROFILER(context->runtime, 
                        MATERIALIZED_VIEW_FILTER_LOCAL_USERS_CALL);
      // Don't do this if we are in Legion Spy since we want to see
      // all of the dependences on an instance
#ifndef LEGION_DISABLE_EVENT_PRUNING
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
      LegionMap<ApEvent,EventUsers>::iterator previous_finder = 
        previous_epoch_users.find(term_event);
      if (previous_finder != previous_epoch_users.end())
      {
        for (EventUsers::const_iterator it = previous_finder->second.begin();
              it != previous_finder->second.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        previous_epoch_users.erase(previous_finder);
      }
#endif
    }

    //--------------------------------------------------------------------------
    void ExprView::filter_current_users(const EventFieldUsers &to_filter)
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
            if (target_finder->second.insert(it->first, overlap))
            {
              // Added a new user to the previous users
              if (needs_reference)
                it->first->add_reference();
            }
            else
            {
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
    void ExprView::filter_previous_users(const EventFieldUsers &to_filter)
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
    void ExprView::find_current_preconditions(const RegionUsage &usage,
                                              const FieldMask &user_mask,
                                              IndexSpaceExpression *user_expr,
                                              ApEvent term_event,
                                              const UniqueID op_id,
                                              const unsigned index,
                                              const bool user_covers,
                                              std::set<ApEvent> &preconditions,
                                              std::set<ApEvent> &dead_events,
                                              EventFieldUsers &filter_users,
                                              FieldMask &observed,
                                              FieldMask &non_dominated,
                                              const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (EventFieldUsers::const_iterator cit = current_epoch_users.begin(); 
            cit != current_epoch_users.end(); cit++)
      {
        if (cit->first == term_event)
          continue;
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (!trace_recording && cit->first.has_triggered_faultignorant())
        {
          dead_events.insert(cit->first);
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(cit->first) != preconditions.end())
          continue;
#endif
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
          bool dominates = true;
          if (has_local_precondition<false>(it->first, usage, user_expr, 
                                  op_id, index, user_covers, dominates))
          {
            preconditions.insert(cit->first);
            if (dominates)
            {
              observed |= user_overlap;
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
    void ExprView::find_previous_preconditions(const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               ApEvent term_event,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               const bool user_covers,
                                               std::set<ApEvent> &preconditions,
                                               std::set<ApEvent> &dead_events,
                                               const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (EventFieldUsers::const_iterator pit = previous_epoch_users.begin(); 
            pit != previous_epoch_users.end(); pit++)
      {
        if (pit->first == term_event)
          continue;
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (!trace_recording && pit->first.has_triggered_faultignorant())
        {
          dead_events.insert(pit->first);
          continue;
        }
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording &&
            preconditions.find(pit->first) != preconditions.end())
          continue;
#endif
#endif
        const EventUsers &event_users = pit->second;
        if (user_mask * event_users.get_valid_mask())
          continue;
        for (EventUsers::const_iterator it = event_users.begin();
              it != event_users.end(); it++)
        {
          if (user_mask * it->second)
            continue;
          if (has_local_precondition<false>(it->first, usage, user_expr, 
                                            op_id, index, user_covers))
            preconditions.insert(pit->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_current_preconditions(const RegionUsage &usage,
                                              const FieldMask &user_mask,
                                              IndexSpaceExpression *user_expr,
                                              const UniqueID op_id,
                                              const unsigned index,
                                              const bool user_covers,
                                              EventFieldExprs &preconditions,
                                              std::set<ApEvent> &dead_events,
                                              EventFieldUsers &filter_events,
                                              FieldMask &observed,
                                              FieldMask &non_dominated,
                                              const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (EventFieldUsers::const_iterator cit = current_epoch_users.begin(); 
            cit != current_epoch_users.end(); cit++)
      {
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (!trace_recording && cit->first.has_triggered_faultignorant())
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
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording && finder != preconditions.end())
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
          bool dominated = true;
          if (has_local_precondition<true>(it->first, usage, user_expr,
                                 op_id, index, user_covers, dominated)) 
          {
            if (finder == preconditions.end())
            {
              preconditions[cit->first].insert(user_expr, user_overlap);
              finder = preconditions.find(cit->first);
            }
            else
              finder->second.insert(user_expr, user_overlap);
            if (dominated)
            {
              observed |= user_overlap;
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
    void ExprView::find_previous_preconditions(const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               const UniqueID op_id,
                                               const unsigned index,
                                               const bool user_covers,
                                               EventFieldExprs &preconditions,
                                               std::set<ApEvent> &dead_events,
                                               const bool trace_recording)
    //--------------------------------------------------------------------------
    {
      // Caller must be holding the lock
      for (LegionMap<ApEvent,EventUsers>::const_iterator pit = 
            previous_epoch_users.begin(); pit != 
            previous_epoch_users.end(); pit++)
      {
#ifndef LEGION_DISABLE_EVENT_PRUNING
        // We're about to do a bunch of expensive tests, 
        // so first do something cheap to see if we can 
        // skip all the tests.
        if (!trace_recording && pit->first.has_triggered_faultignorant())
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
#if 0
        // You might think you can optimize things like this, but you can't
        // because we still need the correct epoch users for every ExprView
        // when we go to add our user later
        if (!trace_recording && finder != preconditions.end())
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
          if (has_local_precondition<true>(it->first, usage, user_expr, 
                                           op_id, index, user_covers))
          {
            if (finder == preconditions.end())
            {
              preconditions[pit->first].insert(user_expr, user_overlap);
              // Needed for when we go around the loop again
              finder = preconditions.find(pit->first);
            }
            else
              finder->second.insert(user_expr, user_overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ExprView::find_previous_filter_users(const FieldMask &dom_mask,
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

    /////////////////////////////////////////////////////////////
    // PendingTaskUser
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingTaskUser::PendingTaskUser(const RegionUsage &u, const FieldMask &m,
                                     IndexSpaceNode *expr, const UniqueID id,
                                     const unsigned idx, const ApEvent term,
                                     const RtEvent collect)
      : usage(u), user_mask(m), user_expr(expr), op_id(id), 
        index(idx), term_event(term), collect_event(collect)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PendingTaskUser::~PendingTaskUser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PendingTaskUser::apply(MaterializedView *view, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      const FieldMask overlap = user_mask & mask;
      if (!overlap)
        return false;
      view->add_internal_task_user(usage, user_expr, overlap, term_event, 
                                   collect_event, op_id,index,false/*tracing*/);
      user_mask -= overlap;
      return !user_mask;
    }

    /////////////////////////////////////////////////////////////
    // PendingCopyUser
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingCopyUser::PendingCopyUser(const bool read, const FieldMask &mask,
                                     IndexSpaceExpression *e, const UniqueID id,
                                     const unsigned idx, const ApEvent term,
                                     const RtEvent collect)
      : reading(read), copy_mask(mask), copy_expr(e), op_id(id), 
        index(idx), term_event(term), collect_event(collect)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    PendingCopyUser::~PendingCopyUser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool PendingCopyUser::apply(MaterializedView *view, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      const FieldMask overlap = copy_mask & mask;
      if (!overlap)
        return false;
      const RegionUsage usage(reading ? LEGION_READ_ONLY : LEGION_READ_WRITE, 
                              LEGION_EXCLUSIVE, 0);
      view->add_internal_copy_user(usage, copy_expr, overlap, term_event,
                       collect_event, op_id, index, false/*trace recording*/);
      copy_mask -= overlap;
      return !copy_mask;
    }

    /////////////////////////////////////////////////////////////
    // MaterializedView 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(
                               RegionTreeForest *ctx, DistributedID did,
                               AddressSpaceID own_addr,
                               AddressSpaceID log_own, PhysicalManager *man,
                               UniqueID own_ctx, bool register_now)
      : InstanceView(ctx, encode_materialized_did(did), man, own_addr,
                     log_own, own_ctx, register_now), 
        expr_cache_uses(0), outstanding_additions(0)
#ifdef ENABLE_VIEW_REPLICATION
        , remote_added_users(0), remote_pending_users(NULL)
#endif
    //--------------------------------------------------------------------------
    {
#ifdef ENABLE_VIEW_REPLICATION
      repl_ptr.replicated_copies = NULL;
#endif
      if (is_logical_owner())
      {
        current_users = new ExprView(ctx,manager,this,manager->instance_domain);
        current_users->add_reference();
      }
      else
        current_users = NULL;
#ifdef LEGION_GC
      log_garbage.info("GC Materialized View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space, 
          LEGION_DISTRIBUTED_ID_FILTER(manager->did)); 
#endif
    }

    //--------------------------------------------------------------------------
    MaterializedView::MaterializedView(const MaterializedView &rhs)
      : InstanceView(NULL, 0, NULL, 0, 0, 0, false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MaterializedView::~MaterializedView(void)
    //--------------------------------------------------------------------------
    {
      if ((current_users != NULL) && current_users->remove_reference())
        delete current_users;
#ifdef ENABLE_VIEW_REPLICATION
      if (repl_ptr.replicated_copies != NULL)
      {
#ifdef DEBUG_LEGION
        assert(is_logical_owner());
#endif
        // We should only have replicated copies here
        // If there are replicated requests that is very bad
        delete repl_ptr.replicated_copies;
      }
#ifdef DEBUG_LEGION
      assert(remote_pending_users == NULL);
#endif
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
      assert(current_users != NULL);
#endif
#ifdef ENABLE_VIEW_REPLICATION
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                term_event, false/*copy user*/, true/*covers*/);
#else
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                            false/*copy user*/, true/*covers*/);
#endif
      // No need to take the lock since we are just initializing
      // If it's the root this is easy
      if (user_expr == current_users->view_expr)
      {
        current_users->add_current_user(user, term_event, RtEvent::NO_RT_EVENT,
                                        user_mask, false);
        return;
      }
      // See if we have it in the cache
      std::map<IndexSpaceExprID,ExprView*>::const_iterator finder = 
        expr_cache.find(user_expr->expr_id);
      if (finder == expr_cache.end() || 
          !(finder->second->invalid_fields * user_mask))
      {
        // No need for expr_lock since this is initialization
        if (finder == expr_cache.end())
        {
          ExprView *target_view = current_users->find_congruent_view(user_expr);
          // Couldn't find a congruent view so we need to make one
          if (target_view == NULL)
            target_view = new ExprView(context, manager, this, user_expr);
          expr_cache[user_expr->expr_id] = target_view;
          finder = expr_cache.find(user_expr->expr_id);
        }
        if (finder->second != current_users)
        {
          // Now insert it for the invalid fields
          FieldMask insert_mask = user_mask & finder->second->invalid_fields;
          // Mark that we're removing these fields from the invalid fields
          // first since we're later going to destroy the insert mask
          finder->second->invalid_fields -= insert_mask;
          // Then insert the subview into the tree
          current_users->insert_subview(finder->second, insert_mask);
        }
      }
      // Now that the view is valid we can add the user to it
      finder->second->add_current_user(user, term_event, RtEvent::NO_RT_EVENT,
                                       user_mask, false);
      // No need to launch a collection task as the destructor will handle it 
    }

    //--------------------------------------------------------------------------
    ApEvent MaterializedView::register_user(const RegionUsage &usage,
                                            const FieldMask &user_mask,
                                            IndexSpaceNode *user_expr,
                                            const UniqueID op_id,
                                            const unsigned index,
                                            ApEvent term_event,
                                            RtEvent collect_event,
                                            std::set<RtEvent> &applied_events,
                                            const PhysicalTraceInfo &trace_info,
                                            const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Quick test for empty index space expressions
      if (user_expr->is_empty())
        return manager->get_use_event();
      if (!is_logical_owner())
      {
        ApUserEvent ready_event;
        // Check to see if this user came from somewhere that wasn't
        // the logical owner, if so we need to send the update back 
        // to the owner to be handled
        if (source != logical_owner)
        {
          // If we're not the logical owner send a message there 
          // to do the analysis and provide a user event to trigger
          // with the precondition
          ready_event = Runtime::create_ap_user_event(&trace_info);
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(usage);
            rez.serialize(user_mask);
            rez.serialize(user_expr->handle);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(term_event);
            rez.serialize(collect_event);
            rez.serialize(ready_event);
            rez.serialize(applied_event);
            trace_info.pack_trace_info<true/*pack operation*/>(rez, 
                                    applied_events, logical_owner);
          }
          // Add a remote valid reference that will be removed by 
          // the receiver once the changes have been applied
          WrapperReferenceMutator mutator(applied_events);
          add_base_valid_ref(REMOTE_DID_REF, &mutator);
          runtime->send_view_register_user(logical_owner, rez);
          applied_events.insert(applied_event);
        }
#ifdef ENABLE_VIEW_REPLICATION
        // If we have any local fields then we also need to update
        // them here too since the owner isn't going to send us any
        // updates itself, Do this after sending the message to make
        // sure that we see a sound set of local fields
        AutoLock r_lock(replicated_lock);
        // Only need to add it if it's still replicated
        const FieldMask local_mask = user_mask & replicated_fields;
        if (!!local_mask)
        {
          // See if we need to make the current users data structure
          if (current_users == NULL)
          {
            // Prevent races between multiple added users at the same time
            AutoLock v_lock(view_lock);
            // See if we lost the race
            if (current_users == NULL)
            {
              current_users = 
               new ExprView(context, manager, this, manager->instance_domain);
              current_users->add_reference();
            }
          }
          // Add our local user
          add_internal_task_user(usage, user_expr, local_mask, term_event,
                       collect_event, op_id, index, trace_info.recording);
          // Increment the number of remote added users
          remote_added_users++;
        }
        // If we have outstanding requests to be made a replicated
        // copy then we need to buffer this user so it can be applied
        // later once we actually do get the update from the owner
        // This only applies to updates from the local node though since
        // any remote updates will be sent to us again by the owner
        if ((repl_ptr.replicated_requests != NULL) && (source == local_space))
        {
#ifdef DEBUG_LEGION
          assert(!repl_ptr.replicated_requests->empty());
#endif
          FieldMask buffer_mask;
          for (LegionMap<RtUserEvent,FieldMask>::const_iterator
                it = repl_ptr.replicated_requests->begin();
                it != repl_ptr.replicated_requests->end(); it++)
          {
            const FieldMask overlap = user_mask & it->second;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            assert(overlap * buffer_mask);
#endif
            buffer_mask |= overlap;
            // This user isn't fully applied until the request comes
            // back to make this view valid and the user gets applied
            applied_events.insert(it->first);
          }
          if (!!buffer_mask)
          {
            // Protected by exclusive replicated lock
            if (remote_pending_users == NULL)
              remote_pending_users = new std::list<RemotePendingUser*>();
            remote_pending_users->push_back(
                new PendingTaskUser(usage, buffer_mask, user_expr, op_id,
                                    index, term_event, collect_event));
          }
        }
        if (remote_added_users >= user_cache_timeout)
          update_remote_replication_state(applied_events);
#endif // ENABLE_VIEW_REPLICATION
        return ready_event;
      }
      else
      {
#ifdef ENABLE_VIEW_REPLICATION
        // We need to hold a read-only copy of the replicated lock when
        // doing this in order to make sure it's atomic with any 
        // replication requests that arrive
        AutoLock r_lock(replicated_lock,1,false/*exclusive*/);
        // Send updates to any remote copies to get them in flight
        if (repl_ptr.replicated_copies != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!repl_ptr.replicated_copies->empty());
#endif
          const FieldMask repl_mask = replicated_fields & user_mask;
          if (!!repl_mask)
          {
            for (LegionMap<AddressSpaceID,FieldMask>::const_iterator
                  it = repl_ptr.replicated_copies->begin(); 
                  it != repl_ptr.replicated_copies->end(); it++)
            {
              if (it->first == source)
                continue;
              const FieldMask overlap = it->second & repl_mask;
              if (!overlap)
                continue;
              // Send the update to the remote node
              RtUserEvent applied_event = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(usage);
                rez.serialize(overlap);
                rez.serialize(user_expr->handle);
                rez.serialize(op_id);
                rez.serialize(index);
                rez.serialize(term_event);
                rez.serialize(collect_event);
                rez.serialize(ApUserEvent::NO_AP_USER_EVENT);
                rez.serialize(applied_event);
                trace_info.pack_trace_info<true/*pack operation*/>(rez, 
                                            applied_events, it->first);
              }
              runtime->send_view_register_user(it->first, rez);
              applied_events.insert(applied_event);
            }
          }
        }
#endif // ENABLE_VIEW_REPLICATION
        // Now we can do our local analysis
        std::set<ApEvent> wait_on_events;
        ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          wait_on_events.insert(start_use_event);
        // Find the preconditions
        const bool user_dominates = 
          (user_expr->expr_id == current_users->view_expr->expr_id) ||
          (user_expr->get_volume() == current_users->view_volume);
        {
          // Traversing the tree so need the expr_view lock
          AutoLock e_lock(expr_lock,1,false/*exclusive*/);
          current_users->find_user_preconditions(usage, user_expr, 
                            user_dominates, user_mask, term_event, 
                            op_id, index, wait_on_events, trace_info.recording);
        }
        // Add our local user
        add_internal_task_user(usage, user_expr, user_mask, term_event, 
                               collect_event, op_id,index,trace_info.recording);
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
                                            ReductionOpID redop,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            CopyFillAggregator &aggregator,
                                            const bool trace_recording,
                                            const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // Check to see if there are any replicated fields here which we
        // can handle locally so we don't have to send a message to the owner
        RtUserEvent ready_event;
#ifdef ENABLE_VIEW_REPLICATION
        FieldMask new_remote_fields;
#endif
        FieldMask request_mask(copy_mask);
#ifdef ENABLE_VIEW_REPLICATION
        // See if we can handle this now while all the fields are local
        {
          AutoLock r_lock(replicated_lock,1,false/*exclusive*/);
          if (!!replicated_fields)
          {
            request_mask -= replicated_fields;
            if (!request_mask)
            {
              // All of our fields are local here so we can do the
              // analysis now without waiting for anything
              // We do this while holding the read-only lock on
              // replication to prevent invalidations of the
              // replication state while we're doing this analysis
#ifdef DEBUG_LEGION
              assert(current_users != NULL);
#endif
              EventFieldExprs preconditions;
              ApEvent start_use_event = manager->get_use_event();
              if (start_use_event.exists())
                preconditions[start_use_event].insert(copy_expr, copy_mask);
              const RegionUsage usage(reading ? LEGION_READ_ONLY : (redop > 0) ?
                  LEGION_REDUCE : LEGION_READ_WRITE, LEGION_EXCLUSIVE, redop);
              const bool copy_dominates = 
                (copy_expr->expr_id == current_users->view_expr->expr_id) ||
                (copy_expr->get_volume() == current_users->view_volume);
              {
                // Need a read-only copy of the expr_view lock to 
                // traverse the tree
                AutoLock e_lock(expr_lock,1,false/*exclusive*/);
                current_users->find_copy_preconditions(usage, copy_expr, 
                                       copy_dominates, copy_mask, op_id, 
                                       index, preconditions, trace_recording);
              }
              // Return any preconditions we found to the aggregator
              if (!preconditions.empty())
                aggregator.record_preconditions(this, reading, preconditions);
              // See if there are any new fields we need to record
              // as having been used for copy precondition testing
              // We'll have to update them later with the lock in
              // exclusive mode, this is technically unsafe, but in
              // the worst case it will just invalidate the cache
              // and we'll have to make it valid again later
              new_remote_fields = copy_mask - remote_copy_pre_fields;
            }
          }
        }
        if (!!request_mask)
#endif // ENABLE_VIEW_REPLICATION
        {
          // All the fields are not local, first send the request to 
          // the owner to do the analysis since we're going to need 
          // to do that anyway, then issue any request for replicated
          // fields to be moved to this node and record it as a 
          // precondition for the mapping
          ready_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(&aggregator);
            rez.serialize(ready_event);
            rez.serialize<bool>(trace_recording);
          }
          runtime->send_view_find_copy_preconditions_request(logical_owner,rez);
#ifdef ENABLE_VIEW_REPLICATION
#ifndef DISABLE_VIEW_REPLICATION
          // Need the lock for this next part
          AutoLock r_lock(replicated_lock);
          // Record these fields as being sampled
          remote_copy_pre_fields |= (new_remote_fields & replicated_fields);
          // Recompute this to make sure we didn't lose any races
          request_mask = copy_mask - replicated_fields;
          if (!!request_mask && (repl_ptr.replicated_requests != NULL))
          {
            for (LegionMap<RtUserEvent,FieldMask>::const_iterator it = 
                  repl_ptr.replicated_requests->begin(); it !=
                  repl_ptr.replicated_requests->end(); it++)
            {
              request_mask -= it->second;
              if (!request_mask)
                break;
            }
          }
          if (!!request_mask)
          {
            // Send the request to the owner to make these replicated fields
            const RtUserEvent request_event = Runtime::create_rt_user_event();
            Serializer rez2;
            {
              RezCheck z2(rez2);
              rez2.serialize(did);
              rez2.serialize(request_mask);
              rez2.serialize(request_event);
            }
            runtime->send_view_replication_request(logical_owner, rez2);
            if (repl_ptr.replicated_requests == NULL)
              repl_ptr.replicated_requests =
                new LegionMap<RtUserEvent,FieldMask>();
            (*repl_ptr.replicated_requests)[request_event] = request_mask;
            // Make sure this is done before things are considered "applied"
            // in order to prevent dangling requests
            aggregator.record_reference_mutation_effect(request_event);
          }
#endif
#endif
        }
#ifdef ENABLE_VIEW_REPLICATION
        else if (!!new_remote_fields)
        {
          AutoLock r_lock(replicated_lock);
          // Record any new fields which are still replicated
          remote_copy_pre_fields |= (new_remote_fields & replicated_fields);
          // Then fall through like normal
        }
#endif
        return ready_event;
      }
      else
      {
        // In the case where we're the owner we can just handle
        // this without needing to do anything
        EventFieldExprs preconditions;
        const ApEvent start_use_event = manager->get_use_event();
        if (start_use_event.exists())
          preconditions[start_use_event].insert(copy_expr, copy_mask);
        const RegionUsage usage(reading ? LEGION_READ_ONLY : (redop > 0) ?
            LEGION_REDUCE : LEGION_READ_WRITE, LEGION_EXCLUSIVE, redop);
        const bool copy_dominates = 
          (copy_expr->expr_id == current_users->view_expr->expr_id) ||
          (copy_expr->get_volume() == current_users->view_volume);
        {
          // Need a read-only copy of the expr_lock to traverse the tree
          AutoLock e_lock(expr_lock,1,false/*exclusive*/);
          current_users->find_copy_preconditions(usage,copy_expr,copy_dominates,
                        copy_mask, op_id, index,preconditions, trace_recording);
        }
        // Return any preconditions we found to the aggregator
        if (!preconditions.empty())
          aggregator.record_preconditions(this, reading, preconditions);
        // We're done with the analysis
        return RtEvent::NO_RT_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::find_copy_preconditions_remote(bool reading,
                                                ReductionOpID redop,
                                                const FieldMask &copy_mask,
                                                IndexSpaceExpression *copy_expr,
                                                UniqueID op_id, unsigned index,
                                                EventFieldExprs &preconditions,
                                                const bool trace_recording,
                                                const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      const ApEvent start_use_event = manager->get_use_event();
      if (start_use_event.exists())
        preconditions[start_use_event].insert(copy_expr, copy_mask);
      const RegionUsage usage(reading ? LEGION_READ_ONLY : (redop > 0) ? 
          LEGION_REDUCE : LEGION_READ_WRITE, LEGION_EXCLUSIVE, redop);
      const bool copy_dominates = 
          (copy_expr->expr_id == current_users->view_expr->expr_id) ||
          (copy_expr->get_volume() == current_users->view_volume);
      // Need a read-only copy of the expr_lock to traverse the tree
      AutoLock e_lock(expr_lock,1,false/*exclusive*/);
      current_users->find_copy_preconditions(usage, copy_expr, copy_dominates,
                      copy_mask, op_id, index, preconditions, trace_recording);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_copy_user(bool reading, ReductionOpID redop,
                                         ApEvent term_event,
                                         RtEvent collect_event,
                                         const FieldMask &copy_mask,
                                         IndexSpaceExpression *copy_expr,
                                         UniqueID op_id, unsigned index,
                                         std::set<RtEvent> &applied_events,
                                         const bool trace_recording,
                                         const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        // Check to see if this update came from some place other than the
        // source in which case we need to send it back to the source
        if (source != logical_owner)
        {
          RtUserEvent applied_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<bool>(reading);
            rez.serialize(redop);
            rez.serialize(term_event);
            rez.serialize(collect_event);
            rez.serialize(copy_mask);
            copy_expr->pack_expression(rez, logical_owner);
            rez.serialize(op_id);
            rez.serialize(index);
            rez.serialize(applied_event);
            rez.serialize<bool>(trace_recording);
          }
          // Add a remote valid reference that will be removed by 
          // the receiver once the changes have been applied
          WrapperReferenceMutator mutator(applied_events);
          add_base_valid_ref(REMOTE_DID_REF, &mutator);
          runtime->send_view_add_copy_user(logical_owner, rez);
          applied_events.insert(applied_event);
        }
#ifdef ENABLE_VIEW_REPLICATION
        AutoLock r_lock(replicated_lock);
        // Only need to add it if it's still replicated
        const FieldMask local_mask = copy_mask & replicated_fields;
        // If we have local fields to handle do that here
        if (!!local_mask)
        {
          // See if we need to make the current users data structure
          if (current_users == NULL)
          {
            // Prevent races between multiple added users at the same time
            AutoLock v_lock(view_lock);
            // See if we lost the race
            if (current_users == NULL)
            {
              current_users = 
               new ExprView(context, manager, this, manager->instance_domain);
              current_users->add_reference();
            }
          }
          const RegionUsage usage(reading ? LEGION_READ_ONLY: (redop > 0) ? 
              LEGION_REDUCE : LEGION_READ_WRITE, LEGION_EXCLUSIVE, redop);
          add_internal_copy_user(usage, copy_expr, local_mask, term_event, 
                                 collect_event, op_id, index, trace_recording);
          // Increment the remote added users count
          remote_added_users++;
        }
        // If we have pending replicated requests that overlap with this
        // user then we need to record this as a pending user to be applied
        // once we receive the update from the owner node
        // This only applies to updates from the local node though since
        // any remote updates will be sent to us again by the owner
        if ((repl_ptr.replicated_requests != NULL) && (source == local_space))
        {
#ifdef DEBUG_LEGION
          assert(!repl_ptr.replicated_requests->empty());
#endif
          FieldMask buffer_mask;
          for (LegionMap<RtUserEvent,FieldMask>::const_iterator
                it = repl_ptr.replicated_requests->begin();
                it != repl_ptr.replicated_requests->end(); it++)
          {
            const FieldMask overlap = copy_mask & it->second;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            assert(overlap * buffer_mask);
#endif
            buffer_mask |= overlap;
            // This user isn't fully applied until the request comes
            // back to make this view valid and the user gets applied
            applied_events.insert(it->first);
          }
          if (!!buffer_mask)
          {
            // Protected by exclusive replicated lock
            if (remote_pending_users == NULL)
              remote_pending_users = new std::list<RemotePendingUser*>();
            remote_pending_users->push_back(
                new PendingCopyUser(reading, buffer_mask, copy_expr, op_id,
                                    index, term_event, collect_event));
          }
        }
        if (remote_added_users >= user_cache_timeout)
          update_remote_replication_state(applied_events);
#endif // ENABLE_VIEW_REPLICATION
      }
      else
      {
#ifdef ENABLE_VIEW_REPLICATION
        // We need to hold this lock in read-only mode to properly
        // synchronize this with any replication requests that arrive
        AutoLock r_lock(replicated_lock,1,false/*exclusive*/);
        // Send updates to any remote copies to get them in flight
        if (repl_ptr.replicated_copies != NULL)
        {
#ifdef DEBUG_LEGION
          assert(!repl_ptr.replicated_copies->empty());
#endif
          const FieldMask repl_mask = replicated_fields & copy_mask;
          if (!!repl_mask)
          {
            for (LegionMap<AddressSpaceID,FieldMask>::const_iterator
                  it = repl_ptr.replicated_copies->begin(); 
                  it != repl_ptr.replicated_copies->end(); it++)
            {
              if (it->first == source)
                continue;
              const FieldMask overlap = it->second & repl_mask;
              if (!overlap)
                continue;
              RtUserEvent applied_event = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize<bool>(reading);
                rez.serialize(redop);
                rez.serialize(term_event);
                rez.serialize(collect_event);
                rez.serialize(copy_mask);
                copy_expr->pack_expression(rez, it->first);
                rez.serialize(op_id);
                rez.serialize(index);
                rez.serialize(applied_event);
                rez.serialize<bool>(trace_recording);
              }
              runtime->send_view_add_copy_user(it->first, rez);
              applied_events.insert(applied_event);
            }
          }
        }
#endif
        // Now we can do our local analysis
        const RegionUsage usage(reading ? LEGION_READ_ONLY : (redop > 0) ?
            LEGION_REDUCE : LEGION_READ_WRITE, LEGION_EXCLUSIVE, redop);
        add_internal_copy_user(usage, copy_expr, copy_mask, term_event, 
                               collect_event, op_id, index, trace_recording);
      }
    }

#ifdef ENABLE_VIEW_REPLICATION
    //--------------------------------------------------------------------------
    void MaterializedView::process_replication_request(AddressSpaceID source,
                                                  const FieldMask &request_mask,
                                                  RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // Atomically we need to package up the response and send it back
      AutoLock r_lock(replicated_lock); 
      if (repl_ptr.replicated_copies == NULL)
        repl_ptr.replicated_copies = 
          new LegionMap<AddressSpaceID,FieldMask>();
      LegionMap<AddressSpaceID,FieldMask>::iterator finder = 
        repl_ptr.replicated_copies->find(source);
      if (finder != repl_ptr.replicated_copies->end())
      {
#ifdef DEBUG_LEGION
        assert(finder->second * request_mask);
#endif
        finder->second |= request_mask; 
      }
      else
        (*repl_ptr.replicated_copies)[source] = request_mask;
      // Update the summary as well
      replicated_fields |= request_mask;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_event);
        std::map<PhysicalUser*,unsigned> indexes;
        // Make sure no one else is mutating the state of the tree
        // while we are doing the packing
        AutoLock e_lock(expr_lock,1,false/*exclusive*/);
        current_users->pack_replication(rez, indexes, request_mask, source);
      }
      runtime->send_view_replication_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_replication_response(RtUserEvent done,
                                                        Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      AutoLock r_lock(replicated_lock);
      {
        // Take the view lock so we can modify the cache as well
        // as part of our unpacking
        AutoLock v_lock(view_lock);
        if (current_users == NULL)
        {
          current_users = 
            new ExprView(context, manager, this, manager->instance_domain);
          current_users->add_reference();
        }
        // We need to hold the expr lock here since we might have to 
        // make ExprViews and we need this to be atomic with other
        // operations that might also try to mutate the tree 
        AutoLock e_lock(expr_lock);
        std::vector<PhysicalUser*> users;
        // The source is always from the logical owner space
        current_users->unpack_replication(derez, current_users, 
                                          logical_owner, expr_cache, users);
        // Remove references from all our users
        for (unsigned idx = 0; idx < users.size(); idx++)
          if (users[idx]->remove_reference())
            delete users[idx]; 
      }
#ifdef DEBUG_LEGION
      assert(repl_ptr.replicated_requests != NULL);
#endif
      LegionMap<RtUserEvent,FieldMask>::iterator finder = 
        repl_ptr.replicated_requests->find(done);
#ifdef DEBUG_LEGION
      assert(finder != repl_ptr.replicated_requests->end());
#endif
      // Go through and apply any pending remote users we've recorded 
      if (remote_pending_users != NULL)
      {
        for (std::list<RemotePendingUser*>::iterator it = 
              remote_pending_users->begin(); it != 
              remote_pending_users->end(); /*nothing*/)
        {
          if ((*it)->apply(this, finder->second))
          {
            delete (*it);
            it = remote_pending_users->erase(it);
          }
          else
            it++;
        }
        if (remote_pending_users->empty())
        {
          delete remote_pending_users;
          remote_pending_users = NULL;
        }
      }
      // Record that these fields are now replicated
      replicated_fields |= finder->second;
      repl_ptr.replicated_requests->erase(finder);
      if (repl_ptr.replicated_requests->empty())
      {
        delete repl_ptr.replicated_requests;
        repl_ptr.replicated_requests = NULL;
      }
    }

    //--------------------------------------------------------------------------
    void MaterializedView::process_replication_removal(AddressSpaceID source,
                                                  const FieldMask &removal_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock r_lock(replicated_lock);
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(repl_ptr.replicated_copies != NULL);
#endif
      LegionMap<AddressSpaceID,FieldMask>::iterator finder = 
        repl_ptr.replicated_copies->find(source);
#ifdef DEBUG_LEGION
      assert(finder != repl_ptr.replicated_copies->end());
      // We should know about all the fields being removed
      assert(!(removal_mask - finder->second));
#endif
      finder->second -= removal_mask;
      if (!finder->second)
      {
        repl_ptr.replicated_copies->erase(finder);
        if (repl_ptr.replicated_copies->empty())
        {
          delete repl_ptr.replicated_copies;
          repl_ptr.replicated_copies = NULL;
          replicated_fields.clear();
          return;
        }
        // Otherwise fall through and rebuild the replicated fields
      }
      // Rebuild the replicated fields so they are precise
      if (repl_ptr.replicated_copies->size() > 1)
      {
        replicated_fields.clear();
        for (LegionMap<AddressSpaceID,FieldMask>::const_iterator it =
              repl_ptr.replicated_copies->begin(); it !=
              repl_ptr.replicated_copies->end(); it++)
          replicated_fields |= finder->second;
      }
      else
        replicated_fields = repl_ptr.replicated_copies->begin()->second;
    }
#endif // ENABLE_VIEW_REPLICATION
 
    //--------------------------------------------------------------------------
    void MaterializedView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_gc_ref(did, mutator);
      // If we're the logical owner, but not the original owner
      // then we use a gc reference on the original owner to 
      // keep all the views allive until we're done
      if (is_logical_owner() && !is_owner())
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // we have a resource reference on the manager so no need to check
      manager->remove_nested_gc_ref(did, mutator);
      // If we're the logical owner but not the original owner
      // then we remove the gc reference that we added
      if (is_logical_owner() && !is_owner())
        send_remote_gc_decrement(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // The logical owner is where complete set of users is and is therefore
      // where garbage collection will take place so we need to send our 
      // valid update there if we're not the owner, otherwise we send it 
      // down to the manager
      if (is_logical_owner())
        manager->add_nested_valid_ref(did, mutator);
      else
        send_remote_valid_increment(logical_owner, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_logical_owner())
        // we have a resource reference on the manager so no need to check
        manager->remove_nested_valid_ref(did, mutator);
      else
        send_remote_valid_decrement(logical_owner, mutator);
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_internal_task_user(const RegionUsage &usage,
                                            IndexSpaceExpression *user_expr,
                                            const FieldMask &user_mask,
                                            ApEvent term_event, 
                                            RtEvent collect_event, 
                                            UniqueID op_id,
                                            const unsigned index,
                                            const bool trace_recording)
    //--------------------------------------------------------------------------
    {
#ifdef ENABLE_VIEW_REPLICATION
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                             collect_event, false/*copy user*/, true/*covers*/);
#else
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                            false/*copy user*/, true/*covers*/);
#endif
      // Hold a reference to this in case it finishes before we're done
      // with the analysis and its get pruned/deleted
      user->add_reference();
      ExprView *target_view = NULL;
      bool has_target_view = false;
      // Handle an easy case first, if the user_expr is the same as the 
      // view_expr for the root then this is easy
      bool update_count = true;
      bool update_cache = false;
      if (user_expr != current_users->view_expr)
      {
        // Hard case where we will have subviews
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // See if we can find the entry in the cache and it's valid 
        // for all of our fields
        LegionMap<IndexSpaceExprID,ExprView*>::const_iterator
          finder = expr_cache.find(user_expr->expr_id);
        if (finder != expr_cache.end())
        {
          target_view = finder->second;
          AutoLock e_lock(expr_lock,1,false/*exclusive*/);
          if (finder->second->invalid_fields * user_mask)
            has_target_view = true;
        }
        else
          update_cache = true;
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
      }
      else // This is just going to add at the top so never needs to wait
      {
        target_view = current_users;
        update_count = false;
        has_target_view = true;
      }
      if (!has_target_view)
      {
        // This could change the shape of the view tree so we need
        // exclusive privilege son the expr lock to serialize it
        // with everything else traversing the tree
        AutoLock e_lock(expr_lock);
        // If we don't have a target view see if there is a 
        // congruent one already in the tree
        if (target_view == NULL)
        {
          target_view = current_users->find_congruent_view(user_expr);
          if (target_view == NULL)
            target_view = new ExprView(context, manager, this, user_expr);
        }
        if (target_view != current_users)
        {
          // Now see if we need to insert it
          FieldMask insert_mask = user_mask & target_view->invalid_fields;
          if (!!insert_mask)
          {
            // Remove these fields from being invalid before we
            // destroy the insert mask
            target_view->invalid_fields -= insert_mask;
            // Do the insertion into the tree
            current_users->insert_subview(target_view, insert_mask);
          }
        }
      }
      // Now we know the target view and it's valid for all fields
      // so we can add it to the expr view
      target_view->add_current_user(user, term_event, collect_event,
                                    user_mask, trace_recording);
      if (user->remove_reference())
        delete user;
      AutoLock v_lock(view_lock);
      if (update_count)
      {
#ifdef DEBUG_LEGION
        assert(outstanding_additions.load() > 0);
#endif
        if ((outstanding_additions.fetch_sub(1) == 1) && clean_waiting.exists())
        {
          // Wake up the clean waiter
          Runtime::trigger_event(clean_waiting);
          clean_waiting = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      if (!update_cache)
      {
        // Update the timeout and see if we need to clear the cache
        if (!expr_cache.empty())
        {
          expr_cache_uses++;
          // Check for equality guarantees only one thread in here at a time
          if (expr_cache_uses == user_cache_timeout)
          {
            // Wait until there are are no more outstanding additions
            while (outstanding_additions.load() > 0)
            {
#ifdef DEBUG_LEGION
              assert(!clean_waiting.exists());
#endif
              clean_waiting = Runtime::create_rt_user_event();
              const RtEvent wait_on = clean_waiting;
              v_lock.release();
              wait_on.wait();
              v_lock.reacquire();
            }
            clean_cache<true/*need expr lock*/>();
          }
        }
      }
      else
        expr_cache[user_expr->expr_id] = target_view;
    }

    //--------------------------------------------------------------------------
    void MaterializedView::add_internal_copy_user(const RegionUsage &usage,
                                            IndexSpaceExpression *user_expr,
                                            const FieldMask &user_mask,
                                            ApEvent term_event, 
                                            RtEvent collect_event, 
                                            UniqueID op_id,
                                            const unsigned index,
                                            const bool trace_recording)
    //--------------------------------------------------------------------------
    { 
      // First we're going to check to see if we can add this directly to 
      // an existing ExprView with the same expresssion in which case
      // we'll be able to mark this user as being precise
      ExprView *target_view = NULL;
      bool has_target_view = false;
      // Handle an easy case first, if the user_expr is the same as the 
      // view_expr for the root then this is easy
      bool update_count = false;
      bool update_cache = false;
      if (user_expr != current_users->view_expr)
      {
        // Hard case where we will have subviews
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        // See if we can find the entry in the cache and it's valid 
        // for all of our fields
        LegionMap<IndexSpaceExprID,ExprView*>::const_iterator
          finder = expr_cache.find(user_expr->expr_id);
        if (finder != expr_cache.end())
        {
          target_view = finder->second;
          AutoLock e_lock(expr_lock,1,false/*exclusive*/);
          if (finder->second->invalid_fields * user_mask)
            has_target_view = true;
        }
        // increment the number of outstanding additions
        outstanding_additions.fetch_add(1);
        update_count = true;
      }
      else // This is just going to add at the top so never needs to wait
      {
        target_view = current_users;
        has_target_view = true;
      }
      if (!has_target_view)
      {
        // Do a quick test to see if we can find a target view
        AutoLock e_lock(expr_lock);
        // If we haven't found it yet, see if we can find it
        if (target_view == NULL)
        {
          target_view = current_users->find_congruent_view(user_expr);
          if (target_view != NULL)
            update_cache = true;
        }
        // Don't make it though if we don't already have it
        if (target_view != NULL)
        {
          // No need to insert this if it's the root
          if (target_view != current_users)
          {
            FieldMask insert_mask = target_view->invalid_fields & user_mask;
            if (!!insert_mask)
            {
              target_view->invalid_fields -= insert_mask;
              current_users->insert_subview(target_view, insert_mask);
            }
          }
          has_target_view = true;
        }
      }
      if (has_target_view)
      {
        // If we have a target view, then we know we cover it because
        // the expressions match directly
#ifdef ENABLE_VIEW_REPLICATION
        PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                               collect_event, true/*copy user*/,true/*covers*/);
#else
        PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                              true/*copy user*/,true/*covers*/);
#endif
        // Hold a reference to this in case it finishes before we're done
        // with the analysis and its get pruned/deleted
        user->add_reference();
        // We already know the view so we can just add the user directly
        // there and then do any updates that we need to
        target_view->add_current_user(user, term_event, collect_event, 
                                      user_mask, trace_recording);
        if (user->remove_reference())
          delete user;
        if (update_count || update_cache)
        {
          AutoLock v_lock(view_lock);
          if (update_cache)
            expr_cache[user_expr->expr_id] = target_view;
          if (update_count)
          {
#ifdef DEBUG_LEGION
            assert(outstanding_additions.load() > 0);
#endif
            if ((outstanding_additions.fetch_sub(1) == 1) && 
                clean_waiting.exists())
            {
              // Wake up the clean waiter
              Runtime::trigger_event(clean_waiting);
              clean_waiting = RtUserEvent::NO_RT_USER_EVENT;
            }
          }
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(update_count); // this should always be true
        assert(!update_cache); // this should always be false
#endif
        // This is a case where we don't know where to add the copy user
        // so we need to traverse down and find one, 
        {
          // We're traversing the view tree but not modifying it so 
          // we need a read-only copy of the expr_lock
          AutoLock e_lock(expr_lock,1,false/*exclusive*/);
          current_users->add_partial_user(usage, op_id, index,
                                          user_mask, term_event, 
                                          collect_event, user_expr, 
                                          user_expr->get_volume(), 
                                          trace_recording);
        }
        AutoLock v_lock(view_lock);
#ifdef DEBUG_LEGION
        assert(outstanding_additions.load() > 0);
#endif
        if ((outstanding_additions.fetch_sub(1) == 1) && clean_waiting.exists())
        {
          // Wake up the clean waiter
          Runtime::trigger_event(clean_waiting);
          clean_waiting = RtUserEvent::NO_RT_USER_EVENT;
        }
      } 
    }

    //--------------------------------------------------------------------------
    template<bool NEED_EXPR_LOCK>
    void MaterializedView::clean_cache(void)
    //--------------------------------------------------------------------------
    {
      // Clear the cache
      expr_cache.clear();
      // Reset the cache use counter
      expr_cache_uses = 0;
      // Anytime we clean the cache, we also traverse the 
      // view tree and see if there are any views we can 
      // remove because they no longer have live users
      FieldMask dummy_mask; 
      FieldMaskSet<ExprView> clean_set;
      if (NEED_EXPR_LOCK)
      {
        // Take the lock in exclusive mode since we might be modifying the tree
        AutoLock e_lock(expr_lock);
        current_users->clean_views(dummy_mask, clean_set);
        // We can safely repopulate the cache with any view expressions which
        // are still valid, remove all references for views in the clean set 
        for (FieldMaskSet<ExprView>::const_iterator it = 
              clean_set.begin(); it != clean_set.end(); it++)
        {
          if (!!(~(it->first->invalid_fields)))
            expr_cache[it->first->view_expr->expr_id] = it->first;
          if (it->first->remove_reference())
            delete it->first;
        }
      }
      else
      {
        // Same as above, but without needing to acquire the lock
        // because the caller promised that they already have it
        current_users->clean_views(dummy_mask, clean_set);
        // We can safely repopulate the cache with any view expressions which
        // are still valid, remove all references for views in the clean set 
        for (FieldMaskSet<ExprView>::const_iterator it = 
              clean_set.begin(); it != clean_set.end(); it++)
        {
          if (!!(~(it->first->invalid_fields)))
            expr_cache[it->first->view_expr->expr_id] = it->first;
          if (it->first->remove_reference())
            delete it->first;
        }
      }
    }

#ifdef ENABLE_VIEW_REPLICATION
    //--------------------------------------------------------------------------
    void MaterializedView::update_remote_replication_state(
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
      assert(!!replicated_fields);
      assert(current_users != NULL);
      assert(remote_added_users >= user_cache_timeout);
#endif
      // We can reset the counter now
      remote_added_users = 0;
      // See what fields haven't been sampled recently and therefore
      // we should stop maintaining as remote duplicates
      const FieldMask deactivate_mask = 
        replicated_fields - remote_copy_pre_fields; 
      // We can clear this now for the next epoch
      remote_copy_pre_fields.clear();
      // If we have any outstanding requests though keep those
      if (repl_ptr.replicated_requests != NULL)
      {
        for (LegionMap<RtUserEvent,FieldMask>::const_iterator it = 
              repl_ptr.replicated_requests->begin(); it !=
              repl_ptr.replicated_requests->end(); it++)
        {
#ifdef DEBUG_LEGION
          assert(it->second * deactivate_mask);
#endif
          remote_copy_pre_fields |= it->second;
        }
      }
      // If we don't have any fields to deactivate then we're done
      if (!deactivate_mask)
        return;
      // Send the message to do the deactivation on the owner node
      RtUserEvent done_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(deactivate_mask);
        rez.serialize(done_event);
      }
      runtime->send_view_replication_removal(logical_owner, rez);
      applied_events.insert(done_event);
      // Perform it locally
      {
        // Anytime we do a deactivate that can influence the valid
        // set of ExprView objects so we need to clean the cache
        AutoLock v_lock(view_lock);
#ifdef DEBUG_LEGION
        // There should be no outstanding_additions when we're here
        // because we're already protected by the replication lock
        assert(outstanding_additions.load() == 0);
#endif
        // Go through and remove any users for the deactivate mask
        // Need an exclusive copy of the expr_lock to do this
        AutoLock e_lock(expr_lock);
        current_users->deactivate_replication(deactivate_mask);
        // Then clean the cache since we likely invalidated some
        // things. This will also go through and remove any views
        // that no longer have any active users
        clean_cache<false/*need expr lock*/>();
      }
      // Record that these fields are no longer replicated 
      replicated_fields -= deactivate_mask;
    }
#endif // ENABLE_VIEW_REPLICATION

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
      PhysicalManager *manager =
        runtime->find_or_request_instance_manager(manager_did, man_ready);
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        // Defer this until the manager is ready
        DeferMaterializedViewArgs args(did, manager, owner_space,
                                       logical_owner, context_uid);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_RESPONSE_PRIORITY, man_ready);
      }
      else
        create_remote_view(runtime, did, manager, owner_space, 
                           logical_owner, context_uid); 
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::handle_defer_materialized_view(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferMaterializedViewArgs *dargs = 
        (const DeferMaterializedViewArgs*)args; 
      create_remote_view(runtime, dargs->did, dargs->manager, 
          dargs->owner_space, dargs->logical_owner, dargs->context_uid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MaterializedView::create_remote_view(Runtime *runtime,
                            DistributedID did, PhysicalManager *manager,
                            AddressSpaceID owner_space,
                            AddressSpaceID logical_owner, UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager->is_physical_manager());
#endif
      PhysicalManager *inst_manager = manager->as_physical_manager();
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
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space);
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
        send_remote_gc_decrement(owner_space, mutator);
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
    void FillView::flatten(CopyFillAggregator &aggregator,
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
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space);
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
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
      {
        if (it->first->remove_nested_resource_ref(did))
          delete it->first;
      }
      true_views.clear();
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it =
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
        send_remote_gc_decrement(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
        it->first->add_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it =
            true_views.begin(); it != true_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it = 
            false_views.begin(); it != false_views.end(); it++)
        it->first->remove_nested_valid_ref(did, mutator);
    }

    //--------------------------------------------------------------------------
    void PhiView::record_true_view(LogicalView *view, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      LegionMap<LogicalView*,FieldMask>::iterator finder = 
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
      LegionMap<LogicalView*,FieldMask>::iterator finder = 
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
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it = 
            true_views.begin(); it != true_views.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(false_views.size());
      for (LegionMap<LogicalView*,FieldMask>::const_iterator it = 
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
    void PhiView::flatten(CopyFillAggregator &aggregator,
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
      std::set<RtEvent> ready_events;
      RtEvent ctx_ready;
      InnerContext *owner_context = 
        runtime->find_context(owner_uid, false, &ctx_ready);
      if (ctx_ready.exists())
        ready_events.insert(ctx_ready);
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
                                 PhysicalManager *man, UniqueID own_ctx, 
                                 bool register_now)
      : InstanceView(ctx, encode_reduction_did(did), man, own_sp, log_own, 
                     own_ctx, register_now),
        fill_view(runtime->find_or_create_reduction_fill_view(manager->redop))
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info("GC Reduction View %lld %d %lld", 
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space,
          LEGION_DISTRIBUTED_ID_FILTER(manager->did));
#endif
    }

    //--------------------------------------------------------------------------
    ReductionView::ReductionView(const ReductionView &rhs)
      : InstanceView(NULL, 0, NULL, 0, 0, 0, false), fill_view(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReductionView::~ReductionView(void)
    //--------------------------------------------------------------------------
    { 
      if (!initial_user_events.empty())
      {
        for (std::set<ApEvent>::const_iterator it = initial_user_events.begin();
              it != initial_user_events.end(); it++)
          filter_local_users(*it);
      }
#if !defined(LEGION_DISABLE_EVENT_PRUNING) && defined(DEBUG_LEGION)
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
      assert(IS_READ_ONLY(usage) || IS_REDUCE(usage));
#endif
      // We don't use field versions for doing interference tests on
      // reductions so there is no need to record it
#ifdef ENABLE_VIEW_REPLICATION
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                term_event, false/*copy*/, true/*covers*/);
#else
      PhysicalUser *user = new PhysicalUser(usage, user_expr, op_id, index, 
                                            false/*copy*/, true/*covers*/);
#endif
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
                                         IndexSpaceNode *user_expr,
                                         const UniqueID op_id,
                                         const unsigned index,
                                         ApEvent term_event,
                                         RtEvent collect_event,
                                         std::set<RtEvent> &applied_events,
                                         const PhysicalTraceInfo &trace_info,
                                         const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(usage.redop == manager->redop);
#endif
      // Quick test for empty index space expressions
      if (user_expr->is_empty())
        return manager->get_use_event();
      if (!is_logical_owner())
      {
        // If we're not the logical owner send a message there 
        // to do the analysis and provide a user event to trigger
        // with the precondition
        ApUserEvent ready_event = Runtime::create_ap_user_event(&trace_info);
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize(user_expr->handle);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(term_event);
          rez.serialize(collect_event);
          rez.serialize(ready_event);
          rez.serialize(applied_event);
          trace_info.pack_trace_info<true/*pack operation*/>(rez,
                                  applied_events, logical_owner);
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
        {
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reducing_preconditions(usage, user_mask, user_expr,
                                      op_id, wait_on_events);
        }
        // Add our local user
        const bool issue_collect = add_user(usage, user_expr, user_mask, 
                                      term_event, collect_event, 
                                      op_id, index, false/*copy*/, 
                                      applied_events, trace_info.recording);
        // Launch the garbage collection task, if it doesn't exist
        // then the user wasn't registered anyway, see add_local_user
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(get_manager(), term_event, collect_event,&mutator);
        }
        if (!wait_on_events.empty())
          return Runtime::merge_events(&trace_info, wait_on_events);
        else
          return ApEvent::NO_AP_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReductionView::find_copy_preconditions(bool reading,
                                            ReductionOpID redop,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            CopyFillAggregator &aggregator,
                                            const bool trace_recording,
                                            const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(redop);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(&aggregator);
          rez.serialize(ready_event);
          rez.serialize<bool>(trace_recording);
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
        else if (redop > 0)
        {
#ifdef DEBUG_LEGION
          assert(redop == manager->redop);
#endif
          AutoLock v_lock(view_lock,1,false/*exclusive*/);
          find_reducing_preconditions(copy_mask, copy_expr,op_id,preconditions);
        }
        else
        {
          AutoLock v_lock(view_lock);
          find_initializing_preconditions(copy_mask, copy_expr, 
                                          op_id, preconditions);
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
                                            ReductionOpID redop,
                                            const FieldMask &copy_mask,
                                            IndexSpaceExpression *copy_expr,
                                            UniqueID op_id, unsigned index,
                                            EventFieldExprs &preconditions,
                                            const bool trace_recording,
                                            const AddressSpaceID source)
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
      else if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(redop == manager->redop);
#endif
        AutoLock v_lock(view_lock,1,false/*exclusive*/);
        find_reducing_preconditions(copy_mask, copy_expr, op_id, preconditions);
      }
      else
      {
        AutoLock v_lock(view_lock);
        find_initializing_preconditions(copy_mask, copy_expr, 
                                        op_id, preconditions);
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_copy_user(bool reading, ReductionOpID redop,
                                      ApEvent term_event, RtEvent collect_event,
                                      const FieldMask &copy_mask,
                                      IndexSpaceExpression *copy_expr,
                                      UniqueID op_id, unsigned index,
                                      std::set<RtEvent> &applied_events,
                                      const bool trace_recording,
                                      const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
      {
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize<bool>(reading);
          rez.serialize(redop);
          rez.serialize(term_event);
          rez.serialize(collect_event);
          rez.serialize(copy_mask);
          copy_expr->pack_expression(rez, logical_owner);
          rez.serialize(op_id);
          rez.serialize(index);
          rez.serialize(applied_event);
          rez.serialize<bool>(trace_recording);
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
        // Reduction copy users behave like simultaneous
        const RegionUsage usage(reading ? LEGION_READ_ONLY : (redop > 0) ?
            LEGION_REDUCE : LEGION_READ_WRITE, (redop > 0) ?
            LEGION_SIMULTANEOUS : LEGION_EXCLUSIVE, redop);
        const bool issue_collect = add_user(usage, copy_expr, copy_mask,
            term_event, collect_event, op_id, index, true/*copy*/,
            applied_events, trace_recording);
        // Launch the garbage collection task, if it doesn't exist
        // then the user wasn't registered anyway, see add_local_user
        if (issue_collect)
        {
          WrapperReferenceMutator mutator(applied_events);
          defer_collect_user(get_manager(), term_event, collect_event,&mutator);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reducing_preconditions(const RegionUsage &usage,
                                               const FieldMask &user_mask,
                                               IndexSpaceExpression *user_expr,
                                               UniqueID op_id,
                                               std::set<ApEvent> &wait_on) const
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      // we know that fills are always done between readers and reducers so
      // we just need to check the initialization users and not readers
      for (EventFieldUsers::const_iterator uit = initialization_users.begin();
            uit != initialization_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          wait_on.insert(uit->first);
        }
      }
      // check for coherence dependences on previous reduction users
      for (EventFieldUsers::const_iterator uit = reduction_users.begin();
            uit != reduction_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          // If they are both simultaneous then we can skip
          if (IS_SIMULT(usage) && IS_SIMULT(it->first->usage))
            continue;
          // If they are both atomic then we can skip
          if (IS_ATOMIC(usage) && IS_ATOMIC(it->first->usage))
            continue;
          // Otherwise we need to check for dependences
          IndexSpaceExpression *expr_overlap = 
            context->intersect_index_spaces(user_expr, it->first->expr);
          if (expr_overlap->is_empty())
            continue;
          wait_on.insert(uit->first);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_initializing_preconditions(
                                                const FieldMask &user_mask,
                                                IndexSpaceExpression *user_expr,
                                                UniqueID op_id,
                                                EventFieldExprs &preconditions)
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      // we know that reduces dominate earlier fills so we don't need to check
      // those, but we do need to check both reducers and readers since it is
      // possible there were no readers of reduction instance
      for (EventFieldUsers::iterator uit = reduction_users.begin();
            uit != reduction_users.end(); /*nothing*/)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
        {
          uit++;
          continue;
        }
        std::vector<PhysicalUser*> to_delete;
        EventFieldExprs::iterator event_finder = preconditions.find(uit->first);
        for (EventUsers::iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
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
          // See if we can prune out this user because it is dominated
          if (expr_overlap->get_volume() == it->first->expr->get_volume())
          {
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<PhysicalUser*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            uit->second.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
          if (uit->second.empty())
          {
            EventFieldUsers::iterator to_erase = uit++;
            reduction_users.erase(to_erase);
          }
          else
          {
            uit->second.tighten_valid_mask();
            uit++;
          }
        }
        else
          uit++;
      }
      for (EventFieldUsers::iterator uit = reading_users.begin();
            uit != reading_users.end(); /*nothing*/)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
        {
          uit++;
          continue;
        }
        std::vector<PhysicalUser*> to_delete;
        EventFieldExprs::iterator event_finder = preconditions.find(uit->first);
        for (EventUsers::iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
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
          // See if we can prune out this user because it is dominated
          if (expr_overlap->get_volume() == it->first->expr->get_volume())
          {
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
        }
        if (!to_delete.empty())
        {
          for (std::vector<PhysicalUser*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            uit->second.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
          if (uit->second.empty())
          {
            EventFieldUsers::iterator to_erase = uit++;
            reading_users.erase(to_erase);
          }
          else
          {
            uit->second.tighten_valid_mask();
            uit++;
          }
        }
        else
          uit++;
      }
    }

    //--------------------------------------------------------------------------
    void ReductionView::find_reducing_preconditions(const FieldMask &user_mask,
                                           IndexSpaceExpression *user_expr,
                                           UniqueID op_id,
                                           EventFieldExprs &preconditions) const
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      // we know that fills are always done between readers and reducers so
      // we just need to check the initialization users for reader deps
      for (EventFieldUsers::const_iterator uit = initialization_users.begin();
            uit != initialization_users.end(); uit++)
      {
        const FieldMask event_mask = uit->second.get_valid_mask() & user_mask;
        if (!event_mask)
          continue;
        EventFieldExprs::iterator event_finder = preconditions.find(uit->first);
        for (EventUsers::const_iterator it = uit->second.begin();
              it != uit->second.end(); it++)
        {
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
      // reduction copies into reduction instances operate atomically so 
      // we just need to check for dependences on other exclusive and atomic
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
          const FieldMask overlap = event_mask & it->second;
          if (!overlap)
            continue;
          // We can run in parallel with simultaneous users
          if (IS_SIMULT(it->first->usage))
            continue;
          // Otherwise we need to check for dependences
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
                                           EventFieldExprs &preconditions) const
    //--------------------------------------------------------------------------
    {
      // lock must be held by caller
      // readers only need to check reducers because we know that the only way
      // to get an initialization is for there to be a reducer that dominates
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
                                 const FieldMask &user_mask, 
                                 ApEvent term_event, RtEvent collect_event,
                                 UniqueID op_id, unsigned index, bool copy_user,
                                 std::set<RtEvent> &applied_events,
                                 const bool trace_recording)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
#ifdef ENABLE_VIEW_REPLICATION
      PhysicalUser *new_user = new PhysicalUser(usage, user_expr, op_id, index, 
                                     collect_event, copy_user, true/*covers*/);
#else
      PhysicalUser *new_user = new PhysicalUser(usage, user_expr, op_id, index, 
                                                copy_user, true/*covers*/);
#endif
      new_user->add_reference();
      // No matter what, we retake the lock in exclusive mode so we
      // can handle any clean-up and add our user
      AutoLock v_lock(view_lock);
      add_physical_user(new_user, IS_READ_ONLY(usage), term_event, user_mask);

      if (outstanding_gc_events.find(term_event) == outstanding_gc_events.end())
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
      EventUsers &event_users = reading ? reading_users[term_event] : 
                 IS_REDUCE(user->usage) ? reduction_users[term_event] : 
                                          initialization_users[term_event];
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
        EventFieldUsers::iterator finder = 
          initialization_users.find(term_event);
        if (finder != initialization_users.end())
        {
          for (EventUsers::const_iterator it = finder->second.begin();
                it != finder->second.end(); it++)
            if (it->first->remove_reference())
              delete it->first;
          initialization_users.erase(finder);
        }
        finder = reduction_users.find(term_event);
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
        manager->compute_copy_offsets(copy_mask, dst_fields);
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
      manager->compute_copy_offsets(copy_mask, src_fields);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      manager->add_nested_gc_ref(did, mutator);
      // If we're the logical owner, but not the original owner
      // then we use a gc reference on the original owner to 
      // keep all the views allive until we're done
      if (is_logical_owner() && !is_owner())
        send_remote_gc_increment(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      manager->remove_nested_gc_ref(did, mutator);
      // If we're the logical owner but not the original owner
      // then we remove the gc reference that we added
      if (is_logical_owner() && !is_owner())
        send_remote_gc_decrement(owner_space, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // The logical owner is where complete set of users is and is therefore
      // where garbage collection will take place so we need to send our 
      // valid update there if we're not the owner, otherwise we send it 
      // down to the manager
      if (is_logical_owner())
        manager->add_nested_valid_ref(did, mutator);
      else
        send_remote_valid_increment(logical_owner, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_logical_owner())
        // we have a resource reference on the manager so no need to check
        manager->remove_nested_valid_ref(did, mutator);
      else
        send_remote_valid_decrement(logical_owner, mutator);
    }

    //--------------------------------------------------------------------------
    void ReductionView::add_collectable_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mutator != NULL);
#endif
      // Only the logical owner adds the full GC reference as this is where
      // the actual garbage collection algorithm will take place and we know
      // that we have all the valid gc event users
      if (is_logical_owner())
        add_base_gc_ref(PENDING_GC_REF, mutator);
      else
        add_base_resource_ref(PENDING_GC_REF);
    }

    //--------------------------------------------------------------------------
    bool ReductionView::remove_collectable_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      if (is_logical_owner())
        return remove_base_gc_ref(PENDING_GC_REF, mutator);
      else
        return remove_base_resource_ref(PENDING_GC_REF);
    }

    //--------------------------------------------------------------------------
    void ReductionView::collect_users(const std::set<ApEvent> &term_events)
    //--------------------------------------------------------------------------
    {
      // Do not do this if we are in LegionSpy so we can see 
      // all of the dependences
#ifndef LEGION_DISABLE_EVENT_PRUNING
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
      PhysicalManager *manager =
        runtime->find_or_request_instance_manager(manager_did, man_ready);
      if (man_ready.exists() && !man_ready.has_triggered())
      {
        // Defer this until the manager is ready
        DeferReductionViewArgs args(did, manager, owner_space,
                                    logical_owner, context_uid);
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_RESPONSE_PRIORITY, man_ready);
      }
      else
        create_remote_view(runtime, did, manager, owner_space, 
                           logical_owner, context_uid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::handle_defer_reduction_view(
                                             const void *args, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferReductionViewArgs *dargs = 
        (const DeferReductionViewArgs*)args; 
      create_remote_view(runtime, dargs->did, dargs->manager, 
          dargs->owner_space, dargs->logical_owner, dargs->context_uid);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReductionView::create_remote_view(Runtime *runtime,
                            DistributedID did, PhysicalManager *manager,
                            AddressSpaceID owner_space, 
                            AddressSpaceID logical_owner, UniqueID context_uid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager->is_reduction_manager());
#endif
      void *location;
      ReductionView *view = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        view = new(location) ReductionView(runtime->forest, did, owner_space, 
                                           logical_owner, manager,
                                           context_uid, false/*register now*/);
      else
        view = new ReductionView(runtime->forest, did, owner_space,
                                 logical_owner, manager, 
                                 context_uid, false/*register now*/);
      // Only register after construction
      view->register_with_runtime(NULL/*remote registration not needed*/);
    }

  }; // namespace Internal 
}; // namespace Legion

