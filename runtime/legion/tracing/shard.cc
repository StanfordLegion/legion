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

#include "realm/id.h"  // TODO: remove this hackiness
#include "legion/tracing/shard.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/shard.h"
#include "legion/nodes/expression.h"
#include "legion/operations/complete.h"
#include "legion/tracing/instructions.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ShardedPhysicalTemplate
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::ShardedPhysicalTemplate(
        PhysicalTrace* trace, ApEvent fence_event, ReplicateContext* ctx)
      : HeapifyMixin<
            ShardedPhysicalTemplate, PhysicalTemplate, CONTEXT_LIFETIME>(
            trace, fence_event),
        repl_ctx(ctx), local_shard(repl_ctx->owner_shard->shard_id),
        total_shards(repl_ctx->shard_manager->total_shards),
        template_index(repl_ctx->register_trace_template(this)),
        refreshed_barriers(0), next_deferral_precondition(0),
        recurrent_replays(0), updated_frontiers(0)
    //--------------------------------------------------------------------------
    {
      repl_ctx->add_base_resource_ref(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::~ShardedPhysicalTemplate(void)
    //--------------------------------------------------------------------------
    {
      for (std::pair<const unsigned, ApBarrier>& it : local_frontiers)
        it.second.destroy_barrier();
      // Unregister ourselves from the context and then remove our reference
      repl_ctx->unregister_trace_template(template_index);
      if (repl_ctx->remove_base_resource_ref(TRACE_REF))
        delete repl_ctx;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_trigger_event(
        ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs.exists());
      const AddressSpaceID event_space = find_event_space(lhs);
      if ((event_space == runtime->address_space) &&
          record_shard_event_trigger(lhs, rhs, tlid))
        return;
      RtEvent done = repl_ctx->shard_manager->send_trace_event_trigger(
          trace->logical_trace->tid, event_space, lhs, rhs, tlid);
      if (done.exists())
        applied.insert(done);
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::record_shard_event_trigger(
        ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs.exists());
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      std::map<ApEvent, unsigned>::const_iterator finder = event_map.find(lhs);
      if (finder == event_map.end())
        return false;
      legion_assert(finder->second != NO_INDEX);
      const unsigned rhs_ =
          rhs.exists() ? find_event(rhs, tpl_lock) : fence_completion_id;
      events.emplace_back(ApEvent());
      insert_instruction(new TriggerEvent(*this, finder->second, rhs_, tlid));
      return true;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_merge_events(
        ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
        const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      std::set<unsigned> rhs_;
      std::set<RtEvent> wait_for;
      std::vector<ApEvent> pending_events;
      std::map<ApEvent, RtUserEvent> request_events;
      for (unsigned idx = 0; idx < num_rhs; idx++)
      {
        const ApEvent event = rhs[idx];
        if (!event.exists())
          continue;
        std::map<ApEvent, unsigned>::const_iterator finder =
            event_map.find(event);
        if (finder == event_map.end())
        {
          // We're going to need to check this event later
          pending_events.emplace_back(event);
          // See if anyone else has requested this event yet
          std::map<ApEvent, RtEvent>::const_iterator request_finder =
              pending_event_requests.find(event);
          if (request_finder == pending_event_requests.end())
          {
            const RtUserEvent request_event = Runtime::create_rt_user_event();
            pending_event_requests[event] = request_event;
            wait_for.insert(request_event);
            request_events[event] = request_event;
          }
          else
            wait_for.insert(request_finder->second);
        }
        else if (finder->second != NO_INDEX)
          rhs_.insert(finder->second);
      }
      // If we have anything to wait for we need to do that
      if (!wait_for.empty())
      {
        tpl_lock.release();
        // Send any request messages first
        if (!request_events.empty())
        {
          for (const std::pair<const ApEvent, RtUserEvent>& it : request_events)
            request_remote_shard_event(it.first, it.second);
        }
        // Do the wait
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
        tpl_lock.reacquire();
        // All our pending events should be here now
        for (const ApEvent& pending : pending_events)
        {
          std::map<ApEvent, unsigned>::const_iterator finder =
              event_map.find(pending);
          legion_assert(finder != event_map.end());
          if (finder->second != NO_INDEX)
            rhs_.insert(finder->second);
        }
      }
      if (rhs_.size() == 0)
        rhs_.insert(fence_completion_id);

      if (!lhs.exists())
        Runtime::rename_event(lhs);
      else if (find_event_space(lhs) != runtime->address_space)
      {
        // If the lhs event wasn't made on this node then we need to rename it
        // because we need all events to go back to a node where we know that
        // we have a shard that can answer queries about it
        // Need to make this relationship explicit to Legion Spy
        const ApEvent previous = lhs;
        Runtime::rename_event(lhs);
        LegionSpy::log_event_dependence(previous, lhs);
      }
      else
      {
        for (unsigned idx = 0; idx < num_rhs; idx++)
        {
          if (lhs != rhs[idx])
            continue;
          Runtime::rename_event(lhs);
          break;
        }
      }
      insert_instruction(new MergeEvent(*this, convert_event(lhs), rhs_, tlid));
    }

#ifdef LEGION_DEBUG
    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::convert_event(
        const ApEvent& event, bool check)
    //--------------------------------------------------------------------------
    {
      // We should only be recording events made on our node
      legion_assert(
          !check || (find_event_space(event) == runtime->address_space));
      return PhysicalTemplate::convert_event(event, check);
    }
#endif

    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::find_event(
        const ApEvent& event, AutoLock& tpl_lock)
    //--------------------------------------------------------------------------
    {
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      // If we've already got it then we're done
      if (finder != event_map.end())
      {
        legion_assert(finder->second != NO_INDEX);
        return finder->second;
      }
      // If we don't have it then we need to request it
      // See if someone else already sent the request
      RtEvent wait_for;
      RtUserEvent request_event;
      std::map<ApEvent, RtEvent>::const_iterator request_finder =
          pending_event_requests.find(event);
      if (request_finder == pending_event_requests.end())
      {
        // We're the first ones so send the request
        request_event = Runtime::create_rt_user_event();
        wait_for = request_event;
        pending_event_requests[event] = wait_for;
      }
      else
        wait_for = request_finder->second;
      // Can't be holding the lock while we wait
      tpl_lock.release();
      // Send the request if necessary
      if (request_event.exists())
        request_remote_shard_event(event, request_event);
      if (wait_for.exists())
        wait_for.wait();
      tpl_lock.reacquire();
      // Once we get here then there better be an answer
      finder = event_map.find(event);
      legion_assert(finder != event_map.end());
      legion_assert(finder->second != NO_INDEX);
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_collective_barrier(
        ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(bar.exists());
      AutoLock tpl_lock(template_lock);
      legion_assert(is_recording());
      const unsigned pre_ = pre.exists() ? find_event(pre, tpl_lock) : 0;
#ifdef LEGION_DEBUG
      const unsigned bar_ = convert_event(bar, false /*check*/);
#else
      const unsigned bar_ = convert_event(bar);
#endif
      BarrierArrival* arrival = new BarrierArrival(
          *this, bar, bar_, pre_, arrivals, false /*managed*/);
      insert_instruction(arrival);
      legion_assert(collective_barriers.find(key) == collective_barriers.end());
      // Save this collective barrier
      collective_barriers[key] = arrival;
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::record_barrier_creation(
        ApBarrier& bar, size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::record_barrier_creation(bar, total_arrivals);
      return local_shard;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_barrier_arrival(
        ApBarrier bar, ApEvent pre, size_t arrival_count,
        std::set<RtEvent>& applied, ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(bar.exists());
      legion_assert(is_recording());
      // Find the pre event first
      unsigned rhs = find_event(pre, tpl_lock);
      events.emplace_back(ApEvent());
      BarrierArrival* arrival = new BarrierArrival(
          *this, bar, events.size() - 1, rhs, arrival_count, true /*managed*/);
      insert_instruction(arrival);
      if (owner_shard != local_shard)
      {
        // Check to see if we've already made a barrier arrival instruction
        // for this barrier or not
        std::map<ApEvent, std::vector<BarrierArrival*> >::iterator finder =
            managed_arrivals.find(bar);
        if (finder == managed_arrivals.end())
        {
          // Need to request a subscription to this barrier on the owner shard
          // We need to tell the owner shard that we are going to
          // subscribe to its updates for this barrier
          RtEvent subscribed = Runtime::create_rt_user_event();
          ShardManager* manager = repl_ctx->shard_manager;
          ReplTraceUpdateMessage rez;
          rez.serialize(manager->did);
          rez.serialize(owner_shard);
          rez.serialize(template_index);
          rez.serialize(REMOTE_BARRIER_SUBSCRIBE);
          rez.serialize(bar);
          rez.serialize(local_shard);
          rez.serialize(subscribed);
          manager->send_trace_update(owner_shard, rez);
          applied.insert(subscribed);
          managed_arrivals[bar].emplace_back(arrival);
        }
        else
          finder->second.emplace_back(arrival);
      }
      else
        managed_arrivals[bar].emplace_back(arrival);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_copy(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, int priority,
        CollectiveKind collective, bool record_effect)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != runtime->address_space)
          Runtime::rename_event(lhs);
      }
      // Then do the base call
      PhysicalTemplate::record_issue_copy(
          tlid, lhs, expr, src_fields, dst_fields, reservations, src_tree_id,
          dst_tree_id, precondition, pred_guard, src_unique, dst_unique,
          priority, collective, record_effect);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_fill(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, int priority, CollectiveKind collective,
        bool record_effect)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != runtime->address_space)
          Runtime::rename_event(lhs);
      }
      // Then do the base call
      PhysicalTemplate::record_issue_fill(
          tlid, lhs, expr, fields, fill_value, fill_size, fill_uid, handle,
          tree_id, precondition, pred_guard, unique_event, priority, collective,
          record_effect);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_issue_across(
        const TraceLocalID& tlid, ApEvent& lhs, ApEvent collective_precondition,
        ApEvent copy_precondition, ApEvent src_indirect_precondition,
        ApEvent dst_indirect_precondition, CopyAcrossExecutor* executor)
    //--------------------------------------------------------------------------
    {
      // Make sure the lhs event is local to our shard
      if (lhs.exists())
      {
        const AddressSpaceID event_space = find_event_space(lhs);
        if (event_space != runtime->address_space)
          Runtime::rename_event(lhs);
      }
      // Then do the base call
      PhysicalTemplate::record_issue_across(
          tlid, lhs, collective_precondition, copy_precondition,
          src_indirect_precondition, dst_indirect_precondition, executor);
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardedPhysicalTemplate::find_trace_shard_event(
        ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      // Check to see if we made this event
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      // If we didn't make this event then we don't do anything
      if (finder == event_map.end() || (finder->second == NO_INDEX))
        return ApBarrier::NO_AP_BARRIER;
      // If we did make it then see if we have a remote barrier for it yet
      std::map<ApEvent, BarrierAdvance*>::const_iterator barrier_finder =
          managed_barriers.find(event);
      if (barrier_finder == managed_barriers.end())
      {
        // Make a new barrier and record it in the events
        ApBarrier barrier = runtime->create_ap_barrier(1 /*arrival count*/);
        // The first generation of each barrier should be triggered when
        // it is recorded in a barrier arrival instruction
        runtime->phase_barrier_arrive(barrier, 1 /*count*/);
        // Record this in the instruction stream
#ifdef LEGION_DEBUG
        const unsigned lhs = convert_event(barrier, false /*check*/);
#else
        const unsigned lhs = convert_event(barrier);
#endif
        // First record the barrier advance for this new barrier
        BarrierAdvance* advance = new BarrierAdvance(
            *this, barrier, lhs, 1 /*arrival count*/, true /*owner*/);
        insert_instruction(advance);
        managed_barriers[event] = advance;
        // Next make the arrival instruction for this barrier
        events.emplace_back(ApEvent());
        BarrierArrival* arrival = new BarrierArrival(
            *this, barrier, events.size() - 1, finder->second, 1 /*count*/,
            true /*managed*/);
        insert_instruction(arrival);
        managed_arrivals[event].emplace_back(arrival);
        // Record our local shard too
        advance->record_subscribed_shard(local_shard);
        return advance->record_subscribed_shard(remote_shard);
      }
      else
        return barrier_finder->second->record_subscribed_shard(remote_shard);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_trace_shard_event(
        ApEvent event, ApBarrier barrier)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(event.exists());
      legion_assert(event_map.find(event) == event_map.end());
      if (barrier.exists())
      {
        legion_assert(local_advances.find(event) == local_advances.end());
#ifdef LEGION_DEBUG
        const unsigned index = convert_event(event, false /*check*/);
#else
        const unsigned index = convert_event(event);
#endif
        BarrierAdvance* advance = new BarrierAdvance(
            *this, barrier, index, 1 /*count*/, false /*owner*/);
        insert_instruction(advance);
        local_advances[event] = advance;
        // Don't remove it, just set it to NO_EVENT so we can tell the names
        // of the remote events that we got from other shards
        // See get_completion_for_deletion for where we use this
        std::map<ApEvent, RtEvent>::iterator finder =
            pending_event_requests.find(event);
        legion_assert(finder != pending_event_requests.end());
        finder->second = RtEvent::NO_RT_EVENT;
      }
      else  // no barrier means it's not part of the trace
      {
        event_map[event] = NO_INDEX;
        // In this case we can remove it since we're not tracing it
        std::map<ApEvent, RtEvent>::iterator finder =
            pending_event_requests.find(event);
        legion_assert(finder != pending_event_requests.end());
        pending_event_requests.erase(finder);
      }
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardedPhysicalTemplate::find_trace_shard_frontier(
        ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      // Check to see if we made this event
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      // If we didn't make this event then we don't do anything
      if (finder == event_map.end() || (finder->second == NO_INDEX))
        return ApBarrier::NO_AP_BARRIER;
      std::map<unsigned, ApBarrier>::const_iterator barrier_finder =
          local_frontiers.find(finder->second);
      if (barrier_finder == local_frontiers.end())
      {
        // Make a barrier and record it
        const ApBarrier result =
            runtime->create_ap_barrier(1 /*arrival count*/);
        barrier_finder =
            local_frontiers.insert(std::make_pair(finder->second, result))
                .first;
      }
      // Record that this shard depends on this event
      local_subscriptions[finder->second].insert(remote_shard);
      return barrier_finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_trace_shard_frontier(
        unsigned frontier, ApBarrier barrier)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      remote_frontiers.emplace_back(std::make_pair(barrier, frontier));
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::handle_trace_update(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      UpdateKind kind;
      derez.deserialize(kind);
      RtUserEvent done;
      std::set<RtEvent> applied;
      switch (kind)
      {
        case UPDATE_MUTATED_INST:
          {
            derez.deserialize(done);
            UniqueInst inst;
            inst.deserialize(derez);
            IndexSpaceExpression* user_expr =
                IndexSpaceExpression::unpack_expression(derez, source);
            if (handle_update_mutated_inst(
                    inst, user_expr, derez, applied, done))
              return;
            break;
          }
        case READ_ONLY_USERS_REQUEST:
          {
            ShardID source_shard;
            derez.deserialize(source_shard);
            legion_assert(source_shard != repl_ctx->owner_shard->shard_id);
            size_t num_users;
            derez.deserialize(num_users);
            InstUsers inst_users(num_users);
            for (unsigned vidx = 0; vidx < num_users; vidx++)
            {
              InstanceUser& user = inst_users[vidx];
              user.instance.deserialize(derez);
              user.expr =
                  IndexSpaceExpression::unpack_expression(derez, source);
              derez.deserialize(user.mask);
            }
            std::atomic<bool>* result;
            derez.deserialize(result);
            derez.deserialize(done);
            ShardManager* manager = repl_ctx->shard_manager;
            if (!PhysicalTemplate::are_read_only_users(inst_users))
            {
              ReplTraceUpdateMessage rez;
              rez.serialize(manager->did);
              rez.serialize(source_shard);
              rez.serialize(template_index);
              rez.serialize(READ_ONLY_USERS_RESPONSE);
              rez.serialize(result);
              rez.serialize(done);
              manager->send_trace_update(source_shard, rez);
              // Make sure we don't double trigger
              done = RtUserEvent::NO_RT_USER_EVENT;
            }
            // Otherwise we can just fall through and trigger the event
            break;
          }
        case READ_ONLY_USERS_RESPONSE:
          {
            std::atomic<bool>* result;
            derez.deserialize(result);
            result->store(false);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        case TEMPLATE_BARRIER_REFRESH:
          {
            size_t num_barriers;
            derez.deserialize(num_barriers);
            AutoLock tpl_lock(template_lock);
            if (update_advances_ready.exists())
            {
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApEvent key;
                derez.deserialize(key);
                ApBarrier bar;
                derez.deserialize(bar);
                std::map<ApEvent, BarrierAdvance*>::const_iterator finder =
                    local_advances.find(key);
                if (finder == local_advances.end())
                {
                  std::map<ApEvent, std::vector<BarrierArrival*> >::
                      const_iterator finder2 = managed_arrivals.find(key);
                  legion_assert(finder2 != managed_arrivals.end());
                  for (BarrierArrival* arrival : finder2->second)
                    arrival->set_managed_barrier(bar);
                }
                else
                  finder->second->remote_refresh_barrier(bar);
              }
              size_t num_concurrent;
              derez.deserialize(num_concurrent);
              for (unsigned idx = 0; idx < num_concurrent; idx++)
              {
                TraceLocalID tlid;
                tlid.deserialize(derez);
                std::map<TraceLocalID, std::vector<ConcurrentGroup> >::iterator
                    finder = concurrent_groups.find(tlid);
                legion_assert(finder != concurrent_groups.end());
                size_t num_colors;
                derez.deserialize(num_colors);
                for (unsigned idx2 = 0; idx2 < num_colors; idx2++)
                {
                  Color color;
                  derez.deserialize(color);
                  for (ConcurrentGroup& group : finder->second)
                  {
                    if (group.color != color)
                      continue;
                    derez.deserialize(group.barrier);
                    break;
                  }
                }
                num_barriers += num_colors;
              }
              refreshed_barriers += num_barriers;
              size_t expected = local_advances.size() + managed_arrivals.size();
              for (const std::pair<
                       const TraceLocalID, std::vector<ConcurrentGroup> >& it :
                   concurrent_groups)
                expected += it.second.size();
              legion_assert(refreshed_barriers <= expected);
              // See if the wait has already been done by the local shard
              // If so, trigger it, otherwise do nothing so it can come
              // along and see that everything is done
              if (refreshed_barriers == expected)
              {
                done = update_advances_ready;
                // We're done so reset everything for the next refresh
                update_advances_ready = RtUserEvent::NO_RT_USER_EVENT;
                refreshed_barriers = 0;
              }
            }
            else
            {
              // Buffer these for later until we know it is safe to apply them
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApEvent key;
                derez.deserialize(key);
                legion_assert(
                    pending_refresh_barriers.find(key) ==
                    pending_refresh_barriers.end());
                derez.deserialize(pending_refresh_barriers[key]);
              }
              size_t num_concurrent;
              derez.deserialize(num_concurrent);
              for (unsigned idx = 0; idx < num_concurrent; idx++)
              {
                TraceLocalID tlid;
                tlid.deserialize(derez);
                std::vector<std::pair<Color, RtBarrier> >& pending =
                    pending_concurrent_barriers[tlid];
                size_t num_colors;
                derez.deserialize(num_colors);
                const unsigned offset = pending.size();
                pending.resize(offset + num_colors);
                for (unsigned idx2 = 0; idx2 < num_colors; idx2++)
                {
                  std::pair<Color, RtBarrier>& next = pending[offset + idx2];
                  derez.deserialize(next.first);
                  derez.deserialize(next.second);
                }
              }
            }
            break;
          }
        case FRONTIER_BARRIER_REFRESH:
          {
            size_t num_barriers;
            derez.deserialize(num_barriers);
            AutoLock tpl_lock(template_lock);
            if (update_frontiers_ready.exists())
            {
              // Unpack these barriers and refresh the frontiers
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApBarrier oldbar, newbar;
                derez.deserialize(oldbar);
                derez.deserialize(newbar);
                [[maybe_unused]] bool found = false;
                for (std::pair<ApBarrier, unsigned>& it : remote_frontiers)
                {
                  if (it.first != oldbar)
                    continue;
                  it.first = newbar;
                  found = true;
                  break;
                }
                legion_assert(found);
              }
              updated_frontiers += num_barriers;
              legion_assert(updated_frontiers <= remote_frontiers.size());
              if (updated_frontiers == remote_frontiers.size())
              {
                done = update_frontiers_ready;
                // We're done so reset everything for the next stage
                update_frontiers_ready = RtUserEvent::NO_RT_USER_EVENT;
                updated_frontiers = 0;
              }
            }
            else
            {
              // Buffer these barriers for later until it is safe
              for (unsigned idx = 0; idx < num_barriers; idx++)
              {
                ApBarrier oldbar;
                derez.deserialize(oldbar);
                legion_assert(
                    pending_refresh_frontiers.find(oldbar) ==
                    pending_refresh_frontiers.end());
                derez.deserialize(pending_refresh_frontiers[oldbar]);
              }
            }
            break;
          }
        case REMOTE_BARRIER_SUBSCRIBE:
          {
            ApBarrier bar;
            derez.deserialize(bar);
            ShardID remote_shard;
            derez.deserialize(remote_shard);
            derez.deserialize(done);

            AutoLock tpl_lock(template_lock);
            std::map<ApEvent, BarrierAdvance*>::const_iterator finder =
                managed_barriers.find(bar);
            legion_assert(finder != managed_barriers.end());
            finder->second->record_subscribed_shard(remote_shard);
            break;
          }
        default:
          std::abort();
      }
      if (done.exists())
      {
        if (!applied.empty())
          Runtime::trigger_event(done, Runtime::merge_events(applied));
        else
          Runtime::trigger_event(done);
      }
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
        ShardedPhysicalTemplate* t, UpdateKind k, RtUserEvent d,
        const UniqueInst& i, Deserializer& derez, IndexSpaceExpression* x,
        RtUserEvent u)
      : LgTaskArgs<DeferTraceUpdateArgs>(false, false), target(t), kind(k),
        done(d), inst(i), expr(x), buffer_size(derez.get_remaining_bytes()),
        buffer(malloc(buffer_size)), deferral_event(u)
    //--------------------------------------------------------------------------
    {
      memcpy(buffer, derez.get_current_pointer(), buffer_size);
      derez.advance_pointer(buffer_size);
      expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    ShardedPhysicalTemplate::DeferTraceUpdateArgs::DeferTraceUpdateArgs(
        const DeferTraceUpdateArgs& rhs, RtUserEvent d, IndexSpaceExpression* e)
      : LgTaskArgs<DeferTraceUpdateArgs>(false, false), target(rhs.target),
        kind(rhs.kind), done(rhs.done), inst(rhs.inst), expr(e),
        buffer_size(rhs.buffer_size), buffer(rhs.buffer), deferral_event(d)
    //--------------------------------------------------------------------------
    {
      // Expression reference rolls over unless its new and we need a reference
      expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::DeferTraceUpdateArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> applied;
      Deserializer derez(buffer, buffer_size);
      switch (kind)
      {
        case UPDATE_MUTATED_INST:
          {
            if (target->handle_update_mutated_inst(
                    inst, expr, derez, applied, done, this))
              return;
            break;
          }
        default:
          std::abort();  // should never get here
      }
      legion_assert(done.exists());
      if (!applied.empty())
        Runtime::trigger_event(done, Runtime::merge_events(applied));
      else
        Runtime::trigger_event(done);
      if (deferral_event.exists())
        Runtime::trigger_event(deferral_event);
      if (expr->remove_base_expression_reference(META_TASK_REF))
        delete expr;
      free(buffer);
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::handle_update_mutated_inst(
        const UniqueInst& inst, IndexSpaceExpression* user_expr,
        Deserializer& derez, std::set<RtEvent>& applied, RtUserEvent done,
        const DeferTraceUpdateArgs* dargs)
    //--------------------------------------------------------------------------
    {
      AutoTryLock tpl_lock(template_lock);
      if (!tpl_lock.has_lock())
      {
        RtUserEvent deferral;
        if (dargs != nullptr)
          deferral = dargs->deferral_event;
        RtEvent pre;
        if (!deferral.exists())
        {
          deferral = Runtime::create_rt_user_event();
          pre = chain_deferral_events(deferral);
        }
        else
          pre = tpl_lock.try_next();
        if (dargs == nullptr)
        {
          DeferTraceUpdateArgs args(
              this, UPDATE_MUTATED_INST, done, inst, derez, user_expr,
              deferral);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_MESSAGE_PRIORITY, pre);
        }
        else
        {
          DeferTraceUpdateArgs args(*dargs, deferral, user_expr);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_MESSAGE_PRIORITY, pre);
#ifdef LEGION_DEBUG
          // Keep the deserializer happy since we didn't use it
          derez.advance_pointer(derez.get_remaining_bytes());
#endif
        }
        return true;
      }
      FieldMask user_mask;
      derez.deserialize(user_mask);
      PhysicalTemplate::record_mutated_instance(
          inst, user_expr, user_mask, applied);
      return false;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::request_remote_shard_event(
        ApEvent event, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(event.exists());
      const AddressSpaceID event_space = find_event_space(event);
      repl_ctx->shard_manager->send_trace_event_request(
          this, repl_ctx->owner_shard->shard_id, runtime->address_space,
          template_index, event, event_space, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID ShardedPhysicalTemplate::find_event_space(
        ApEvent event)
    //--------------------------------------------------------------------------
    {
      if (!event.exists())
        return 0;
      // TODO: Remove hack include at top of file when we fix this
      const Realm::ID id(event.id);
      if (id.is_barrier())
        return id.barrier_creator_node();
      legion_assert(id.is_event());
      return id.event_creator_node();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::pack_recorder(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(runtime->address_space);
      rez.serialize(this);
      rez.serialize(repl_ctx->shard_manager->did);
      rez.serialize(trace->logical_trace->tid);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_replay(
        ApEvent completion, bool recurrent)
    //--------------------------------------------------------------------------
    {
      legion_assert(pending_collectives.empty());
      PhysicalTemplate::initialize_replay(completion, recurrent);
      // Now update all of our barrier information
      if (recurrent)
      {
        // If we've run out of generations update the local barriers and
        // send out the updates to everyone
        if (recurrent_replays == Realm::Barrier::MAX_PHASES)
        {
          std::map<ShardID, std::map<ApBarrier /*old**/, ApBarrier /*new*/> >
              notifications;
          // Update our barriers and record which updates to send out
          for (std::pair<const unsigned, ApBarrier>& it : local_frontiers)
          {
            const ApBarrier new_barrier =
                runtime->create_ap_barrier(1 /*arrival count*/);
            legion_assert(
                local_subscriptions.find(it.first) !=
                local_subscriptions.end());
            const std::set<ShardID>& shards = local_subscriptions[it.first];
            for (const ShardID& shard : shards)
              notifications[shard][it.second] = new_barrier;
            // destroy the old barrier and replace it with the new one
            it.second.destroy_barrier();
            it.second = new_barrier;
          }
          // Send out the notifications to all the remote shards
          ShardManager* manager = repl_ctx->shard_manager;
          for (const std::pair<const ShardID, std::map<ApBarrier, ApBarrier> >&
                   nit : notifications)
          {
            ReplTraceUpdateMessage rez;
            rez.serialize(manager->did);
            rez.serialize(nit.first);
            rez.serialize(template_index);
            rez.serialize(FRONTIER_BARRIER_REFRESH);
            rez.serialize<size_t>(nit.second.size());
            for (const std::pair<const ApBarrier, ApBarrier>& it : nit.second)
            {
              rez.serialize(it.first);
              rez.serialize(it.second);
            }
            manager->send_trace_update(nit.first, rez);
          }
          // Now we wait to see that we get all of our remote barriers updated
          RtEvent remote_frontiers_ready;
          {
            AutoLock tpl_lock(template_lock);
            legion_assert(!update_frontiers_ready.exists());
            // Apply any pending refresh frontiers
            if (!pending_refresh_frontiers.empty())
            {
              for (const std::pair<const ApBarrier, ApBarrier>& pit :
                   pending_refresh_frontiers)
              {
                [[maybe_unused]] bool found = false;
                for (std::pair<ApBarrier, unsigned>& it : remote_frontiers)
                {
                  if (it.first != pit.first)
                    continue;
                  it.first = pit.second;
                  found = true;
                  break;
                }
                legion_assert(found);
              }
              updated_frontiers += pending_refresh_frontiers.size();
              legion_assert(updated_frontiers <= remote_frontiers.size());
              pending_refresh_frontiers.clear();
            }
            if (updated_frontiers < remote_frontiers.size())
            {
              update_frontiers_ready = Runtime::create_rt_user_event();
              remote_frontiers_ready = update_frontiers_ready;
            }
            else  // Reset this back to zero for the next round
              updated_frontiers = 0;
          }
          // Wait for the remote frontiers to be updated
          if (remote_frontiers_ready.exists() &&
              !remote_frontiers_ready.has_triggered())
            remote_frontiers_ready.wait();
          // Reset this back to zero after barrier updates
          recurrent_replays = 0;
        }
        // Now we can do the normal update of events based on our barriers
        // Don't advance on last generation to avoid setting barriers back to 0
        const bool advance_barriers =
            ((++recurrent_replays) < Realm::Barrier::MAX_PHASES);
        for (std::pair<const unsigned, ApBarrier>& it : local_frontiers)
        {
          runtime->phase_barrier_arrive(
              it.second, 1 /*count*/, events[it.first]);
          if (advance_barriers)
            Runtime::advance_barrier(it.second);
        }
        for (std::pair<ApBarrier, unsigned>& it : remote_frontiers)
        {
          events[it.second] = it.first;
          if (advance_barriers)
            Runtime::advance_barrier(it.first);
        }
      }
      else
      {
        for (const std::pair<ApBarrier, unsigned>& it : remote_frontiers)
          events[it.second] = completion;
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::start_replay(void)
    //--------------------------------------------------------------------------
    {
      if (!pending_collectives.empty())
      {
        for (const std::pair<const std::pair<size_t, size_t>, ApBarrier>& it :
             pending_collectives)
        {
          // This data structure should be read-only at this point
          // so we shouldn't need the lock to access it
          std::map<std::pair<size_t, size_t>, BarrierArrival*>::const_iterator
              finder = collective_barriers.find(it.first);
          legion_assert(finder != collective_barriers.end());
          finder->second->set_collective_barrier(it.second);
        }
        pending_collectives.clear();
      }
      // Now call the base version of this
      PhysicalTemplate::start_replay();
    }

    //--------------------------------------------------------------------------
    RtEvent ShardedPhysicalTemplate::refresh_managed_barriers(void)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID, std::map<ApEvent, ApBarrier> > notifications;
      // Need to update all our barriers since we're out of generations
      for (const std::pair<const ApEvent, BarrierAdvance*>& it :
           managed_barriers)
        it.second->refresh_barrier(it.first, notifications);
      // Also see if we have any concurrent barriers to update
      size_t local_refreshed = 0;
      std::map<
          ShardID,
          std::map<TraceLocalID, std::vector<std::pair<Color, RtBarrier> > > >
          concurrent_updates;
      for (std::pair<const TraceLocalID, std::vector<ConcurrentGroup> >& cit :
           concurrent_groups)
      {
        for (ConcurrentGroup& it : cit.second)
        {
          legion_assert(!it.shards.empty());
          legion_assert(std::binary_search(
              it.shards.begin(), it.shards.end(), local_shard));
          if (local_shard == it.shards.front())
          {
            it.barrier.destroy_barrier();
            it.barrier = runtime->create_rt_barrier(it.global);
            for (unsigned idx = 1; idx < it.shards.size(); idx++)
            {
              ShardID shard = it.shards[idx];
              notifications[shard];  // instantiate so it is there
              concurrent_updates[shard][cit.first].emplace_back(
                  std::make_pair(it.color, it.barrier));
            }
            local_refreshed++;
          }
        }
      }
      // Send out the notifications to all the shards
      ShardManager* manager = repl_ctx->shard_manager;
      for (const std::pair<const ShardID, std::map<ApEvent, ApBarrier> >& nit :
           notifications)
      {
        if (nit.first != local_shard)
        {
          ReplTraceUpdateMessage rez;
          rez.serialize(manager->did);
          rez.serialize(nit.first);
          rez.serialize(template_index);
          rez.serialize(TEMPLATE_BARRIER_REFRESH);
          rez.serialize<size_t>(nit.second.size());
          for (const std::pair<const ApEvent, ApBarrier>& it : nit.second)
          {
            rez.serialize(it.first);
            rez.serialize(it.second);
          }
          std::map<
              ShardID,
              std::map<
                  TraceLocalID, std::vector<std::pair<Color, RtBarrier> > > >::
              const_iterator finder = concurrent_updates.find(nit.first);
          if (finder != concurrent_updates.end())
          {
            rez.serialize<size_t>(finder->second.size());
            for (const std::pair<
                     const TraceLocalID,
                     std::vector<std::pair<Color, RtBarrier> > >& tit :
                 finder->second)
            {
              tit.first.serialize(rez);
              rez.serialize<size_t>(tit.second.size());
              for (const std::pair<Color, RtBarrier>& it : tit.second)
              {
                rez.serialize(it.first);
                rez.serialize(it.second);
              }
            }
          }
          else
            rez.serialize<size_t>(0);
          manager->send_trace_update(nit.first, rez);
        }
        else
        {
          local_refreshed = nit.second.size();
          for (const std::pair<const ApEvent, ApBarrier>& it : nit.second)
          {
            std::map<ApEvent, std::vector<BarrierArrival*> >::iterator finder =
                managed_arrivals.find(it.first);
            legion_assert(finder != managed_arrivals.end());
            for (unsigned idx = 0; idx < finder->second.size(); idx++)
              finder->second[idx]->set_managed_barrier(it.second);
          }
        }
      }
      // Then wait for all our advances to be updated from other shards
      RtEvent replay_precondition;
      {
        AutoLock tpl_lock(template_lock);
        legion_assert(!update_advances_ready.exists());
        if (local_refreshed > 0)
          refreshed_barriers += local_refreshed;
        if (!pending_refresh_barriers.empty())
        {
          for (const std::pair<const ApEvent, ApBarrier>& it :
               pending_refresh_barriers)
          {
            std::map<ApEvent, BarrierAdvance*>::const_iterator finder =
                local_advances.find(it.first);
            if (finder == local_advances.end())
            {
              std::map<ApEvent, std::vector<BarrierArrival*> >::const_iterator
                  finder2 = managed_arrivals.find(it.first);
              legion_assert(finder2 != managed_arrivals.end());
              for (unsigned idx = 0; idx < finder2->second.size(); idx++)
                finder2->second[idx]->set_managed_barrier(it.second);
            }
            else
              finder->second->remote_refresh_barrier(it.second);
          }
          refreshed_barriers += pending_refresh_barriers.size();
          pending_refresh_barriers.clear();
        }
        if (!pending_concurrent_barriers.empty())
        {
          for (const std::pair<
                   const TraceLocalID,
                   std::vector<std::pair<Color, RtBarrier> > >& bit :
               pending_concurrent_barriers)
          {
            std::map<TraceLocalID, std::vector<ConcurrentGroup> >::iterator
                finder = concurrent_groups.find(bit.first);
            legion_assert(finder != concurrent_groups.end());
            for (const std::pair<Color, RtBarrier>& cit : bit.second)
            {
              for (ConcurrentGroup& it : finder->second)
              {
                if (cit.first != it.color)
                  continue;
                it.barrier = cit.second;
                break;
              }
            }
            refreshed_barriers += bit.second.size();
          }
          pending_concurrent_barriers.clear();
        }
        size_t expected = local_advances.size() + managed_arrivals.size();
        for (const std::pair<const TraceLocalID, std::vector<ConcurrentGroup> >&
                 it : concurrent_groups)
          expected += it.second.size();
        legion_assert(refreshed_barriers <= expected);
        if (refreshed_barriers < expected)
        {
          update_advances_ready = Runtime::create_rt_user_event();
          replay_precondition = update_advances_ready;
        }
        else  // Reset this back to zero for the next round
          refreshed_barriers = 0;
      }
      return replay_precondition;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::finish_replay(
        FenceOp* fence, std::set<ApEvent>& postconditions)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::finish_replay(fence, postconditions);
      // Also need to do any local frontiers that we have here as well
      for (const std::pair<const unsigned, ApBarrier>& it : local_frontiers)
        postconditions.insert(events[it.first]);
    }

    //--------------------------------------------------------------------------
    ApEvent ShardedPhysicalTemplate::get_completion_for_deletion(void) const
    //--------------------------------------------------------------------------
    {
      // Skip the any events that are from remote shards since we
      std::set<ApEvent> all_events;
      std::set<ApEvent> local_barriers;
      for (const std::pair<const ApEvent, BarrierAdvance*>& it :
           managed_barriers)
        local_barriers.insert(it.second->get_current_barrier());
      for (const std::pair<const ApEvent, unsigned>& it : event_map)
      {
        // If this is a remote event or one of our barriers then don't use it
        if ((local_barriers.find(it.first) == local_barriers.end()) &&
            (pending_event_requests.find(it.first) ==
             pending_event_requests.end()))
          all_events.insert(it.first);
      }
      return Runtime::merge_events(nullptr, all_events);
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_mutated_instance(
        const UniqueInst& inst, IndexSpaceExpression* user_expr,
        const FieldMask& user_mask, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      const ShardID target_shard = find_inst_owner(inst);
      // Check to see if we're on the right shard, if not send the message
      if (target_shard != repl_ctx->owner_shard->shard_id)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        ShardManager* manager = repl_ctx->shard_manager;
        ReplTraceUpdateMessage rez;
        rez.serialize(manager->did);
        rez.serialize(target_shard);
        rez.serialize(template_index);
        rez.serialize(UPDATE_MUTATED_INST);
        rez.serialize(done);
        inst.serialize(rez);
        user_expr->pack_expression(rez, manager->get_shard_space(target_shard));
        rez.serialize(user_mask);
        manager->send_trace_update(target_shard, rez);
        applied.insert(done);
      }
      else
        PhysicalTemplate::record_mutated_instance(
            inst, user_expr, user_mask, applied);
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::find_inst_owner(const UniqueInst& inst)
    //--------------------------------------------------------------------------
    {
      // Figure out where the owner for this instance is and then send it to
      // the appropriate shard trace. The algorithm we use for determining
      // the right shard trace is to send a instance to a shard trace on the
      // node that owns the instance. If there is no shard on that node we
      // round-robin views based on their owner node mod the number of nodes
      // where there are shards. Once on the correct node, then we pick the
      // shard corresponding to their instance ID mod the number of shards on
      // that node. This algorithm guarantees that all the related instances
      // end up on the same shard for analysis to determine if the trace is
      // replayable or not.
      const AddressSpaceID inst_owner = inst.get_analysis_space();
      std::vector<ShardID> owner_shards;
      find_owner_shards(inst_owner, owner_shards);
      legion_assert(!owner_shards.empty());
      // Round-robin based on the distributed IDs for the views in the
      // case where there are multiple shards, this should relatively
      // balance things out
      if (owner_shards.size() > 1)
        return owner_shards[inst.view_did % owner_shards.size()];
      else  // If there's only one shard then there is only one choice
        return owner_shards.front();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::find_owner_shards(
        AddressSpaceID owner, std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      // See if we already computed it or not
      std::map<AddressSpaceID, std::vector<ShardID> >::const_iterator finder =
          did_shard_owners.find(owner);
      if (finder != did_shard_owners.end())
      {
        shards = finder->second;
        return;
      }
      // If we haven't computed it yet, then we need to do that now
      const ShardMapping& shard_spaces = repl_ctx->shard_manager->get_mapping();
      for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
        if (shard_spaces[idx] == owner)
          shards.emplace_back(idx);
      // If we didn't find any then take the owner mod the number of total
      // spaces and then send it to the shards on that space
      if (shards.empty())
      {
        std::set<AddressSpaceID> unique_spaces;
        for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
          unique_spaces.insert(shard_spaces[idx]);
        const unsigned count = owner % unique_spaces.size();
        std::set<AddressSpaceID>::const_iterator target_space =
            unique_spaces.begin();
        for (unsigned idx = 0; idx < count; idx++) target_space++;
        for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
          if (shard_spaces[idx] == *target_space)
            shards.emplace_back(idx);
      }
      legion_assert(!shards.empty());
      // Save the result so we don't have to do this again for this space
      did_shard_owners[owner] = shards;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_owner_shard(
        unsigned tid, ShardID owner)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(owner_shards.find(tid) == owner_shards.end());
      owner_shards[tid] = owner;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_local_space(
        unsigned tid, IndexSpace sp)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(local_spaces.find(tid) == local_spaces.end());
      local_spaces[tid] = sp;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_sharding_function(
        unsigned tid, ShardingFunction* function)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      legion_assert(sharding_functions.find(tid) == sharding_functions.end());
      sharding_functions[tid] = function;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::dump_sharded_template(void) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<ApBarrier, unsigned>& it : remote_frontiers)
        log_tracing.info() << "events[" << it.second
                           << "] = Runtime::barrier_advance(" << std::hex
                           << it.first.id << std::dec << ")";
      for (const std::pair<const unsigned, ApBarrier>& it : local_frontiers)
        log_tracing.info() << "Runtime::phase_barrier_arrive(" << std::hex
                           << it.second.id << std::dec << ", events["
                           << it.first << "])";
    }

    //--------------------------------------------------------------------------
    ShardID ShardedPhysicalTemplate::find_owner_shard(unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      std::map<unsigned, ShardID>::const_iterator finder =
          owner_shards.find(tid);
      legion_assert(finder != owner_shards.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    IndexSpace ShardedPhysicalTemplate::find_local_space(unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      std::map<unsigned, IndexSpace>::const_iterator finder =
          local_spaces.find(tid);
      legion_assert(finder != local_spaces.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ShardedPhysicalTemplate::find_sharding_function(
        unsigned tid)
    //--------------------------------------------------------------------------
    {
      AutoLock tpl_lock(template_lock);
      std::map<unsigned, ShardingFunction*>::const_iterator finder =
          sharding_functions.find(tid);
      legion_assert(finder != sharding_functions.end());
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::prepare_collective_barrier_replay(
        const std::pair<size_t, size_t>& key, ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(template_lock);
      // Save the barrier until it's safe to update the instruction
      pending_collectives[key] = newbar;
    }

    //--------------------------------------------------------------------------
    unsigned ShardedPhysicalTemplate::find_frontier_event(
        ApEvent event, std::vector<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // Check to see which shard should own this event
      std::map<ApEvent, unsigned>::const_iterator finder =
          event_map.find(event);
      if (finder != event_map.end())
      {
        if (finder->second == NO_INDEX)
          return 0;  // start fence event
        else
          return PhysicalTemplate::find_frontier_event(event, ready_events);
      }
      const AddressSpaceID event_space = find_event_space(event);
      // Allocate a slot for this event though we might not use it
      const unsigned next_event_id = events.size();
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      repl_ctx->shard_manager->send_trace_frontier_request(
          this, repl_ctx->owner_shard->shard_id, runtime->address_space,
          template_index, event, event_space, next_event_id, done_event);
      events.resize(next_event_id + 1);
      ready_events.emplace_back(done_event);
      return next_event_id;
    }

    //--------------------------------------------------------------------------
    bool ShardedPhysicalTemplate::are_read_only_users(InstUsers& inst_users)
    //--------------------------------------------------------------------------
    {
      std::map<ShardID, InstUsers> shard_inst_users;
      for (InstUsers::iterator vit = inst_users.begin();
           vit != inst_users.end(); vit++)
      {
        const ShardID owner_shard = find_inst_owner(vit->instance);
        shard_inst_users[owner_shard].emplace_back(*vit);
      }
      std::atomic<bool> result(true);
      std::vector<RtEvent> done_events;
      ShardManager* manager = repl_ctx->shard_manager;
      const ShardID local_shard = repl_ctx->owner_shard->shard_id;
      for (std::pair<const ShardID, InstUsers>& sit : shard_inst_users)
      {
        if (sit.first != local_shard)
        {
          const RtUserEvent done = Runtime::create_rt_user_event();
          const AddressSpaceID target = manager->get_shard_space(sit.first);
          ReplTraceUpdateMessage rez;
          rez.serialize(manager->did);
          rez.serialize(sit.first);
          rez.serialize(template_index);
          rez.serialize(READ_ONLY_USERS_REQUEST);
          rez.serialize(local_shard);
          rez.serialize<size_t>(sit.second.size());
          for (InstUsers::const_iterator vit = sit.second.begin();
               vit != sit.second.end(); vit++)
          {
            vit->instance.serialize(rez);
            vit->expr->pack_expression(rez, target);
            rez.serialize(vit->mask);
          }
          rez.serialize(&result);
          rez.serialize(done);
          manager->send_trace_update(sit.first, rez);
          done_events.emplace_back(done);
        }
        else if (!PhysicalTemplate::are_read_only_users(sit.second))
        {
          // Still need to wait for anyone else to write to result if
          // they end up finding out that they are not read-only
          result.store(false);
          break;
        }
      }
      if (!done_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(done_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      return result.load();
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::sync_compute_frontiers(
        CompleteOp* op, const std::vector<RtEvent>& frontier_events)
    //--------------------------------------------------------------------------
    {
      if (!frontier_events.empty())
        op->sync_compute_frontiers(Runtime::merge_events(frontier_events));
      else
        op->sync_compute_frontiers(RtEvent::NO_RT_EVENT);
      // Check for any empty remote frontiers which were not actually
      // contained in the trace and therefore need to be pruned out of
      // any event mergers
      std::vector<unsigned> to_filter;
      for (std::vector<std::pair<ApBarrier, unsigned> >::iterator it =
               remote_frontiers.begin();
           it != remote_frontiers.end();
           /*nothing*/)
      {
        if (!it->first.exists())
        {
          to_filter.emplace_back(it->second);
          it = remote_frontiers.erase(it);
        }
        else
          it++;
      }
      if (!to_filter.empty())
      {
        for (Instruction* instruction : instructions)
        {
          if (instruction->get_kind() != MERGE_EVENT)
            continue;
          MergeEvent* merge = instruction->as_merge_event();
          for (unsigned idx = 0; idx < to_filter.size(); idx++)
          {
            std::set<unsigned>::iterator finder =
                merge->rhs.find(to_filter[idx]);
            if (finder == merge->rhs.end())
              continue;
            // Found one, filter it out from the set
            merge->rhs.erase(finder);
            // Handle a weird case where we pruned them all out
            // Go back to the case of just pointing at the completion event
            if (merge->rhs.empty())
              merge->rhs.insert(0 /*fence completion id*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_generators(
        std::vector<unsigned>& new_gen)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_generators(new_gen);
      for (const std::pair<ApBarrier, unsigned>& it : remote_frontiers)
        new_gen[it.second] = 0;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_eliminate_dead_code_frontiers(
        const std::vector<unsigned>& gen, std::vector<bool>& used)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_eliminate_dead_code_frontiers(gen, used);
      for (const std::pair<const unsigned, ApBarrier>& it : local_frontiers)
      {
        unsigned g = gen[it.first];
        if (g != -1U && g < instructions.size())
          used[g] = true;
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::initialize_transitive_reduction_frontiers(
        std::vector<unsigned>& topo_order,
        std::vector<unsigned>& inv_topo_order)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::initialize_transitive_reduction_frontiers(
          topo_order, inv_topo_order);
      for (const std::pair<ApBarrier, unsigned>& it : remote_frontiers)
      {
        inv_topo_order[it.second] = topo_order.size();
        topo_order.emplace_back(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::record_used_frontiers(
        std::vector<bool>& used, const std::vector<unsigned>& gen) const
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::record_used_frontiers(used, gen);
      for (const std::pair<const unsigned, ApBarrier>& it : local_frontiers)
        used[gen[it.first]] = true;
    }

    //--------------------------------------------------------------------------
    void ShardedPhysicalTemplate::rewrite_frontiers(
        std::map<unsigned, unsigned>& substitutions)
    //--------------------------------------------------------------------------
    {
      PhysicalTemplate::rewrite_frontiers(substitutions);
      std::vector<std::pair<unsigned, ApBarrier> > to_add;
      for (std::map<unsigned, ApBarrier>::iterator it = local_frontiers.begin();
           it != local_frontiers.end();
           /*nothing*/)
      {
        std::map<unsigned, unsigned>::const_iterator finder =
            substitutions.find(it->first);
        if (finder != substitutions.end())
        {
          to_add.emplace_back(std::make_pair(finder->second, it->second));
          // Also need to update the local subscriptions data structure
          std::map<unsigned, std::set<ShardID> >::iterator subscription_finder =
              local_subscriptions.find(it->first);
          legion_assert(subscription_finder != local_subscriptions.end());
          std::map<unsigned, std::set<ShardID> >::iterator local_finder =
              local_subscriptions.find(finder->second);
          if (local_finder != local_subscriptions.end())
            local_finder->second.insert(
                subscription_finder->second.begin(),
                subscription_finder->second.end());
          else
            local_subscriptions[finder->second].swap(
                subscription_finder->second);
          local_subscriptions.erase(subscription_finder);
          std::map<unsigned, ApBarrier>::iterator to_delete = it++;
          local_frontiers.erase(to_delete);
        }
        else
          it++;
      }
      for (const std::pair<unsigned, ApBarrier>& it : to_add)
        local_frontiers.insert(it);
    }

  }  // namespace Internal
}  // namespace Legion
