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

#include "legion/utilities/collectives.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/operations/attach.h"
#include "legion/operations/copy.h"
#include "legion/operations/trace.h"
#include "legion/tasks/index.h"

namespace Legion {
  namespace Internal {

#ifdef LEGION_DEBUG_COLLECTIVES
    /////////////////////////////////////////////////////////////
    // Collective Check Reduction
    /////////////////////////////////////////////////////////////

    /*static*/ const long CollectiveCheckReduction::IDENTITY = -1;
    /*static*/ const long CollectiveCheckReduction::identity = IDENTITY;
    /*static*/ const long CollectiveCheckReduction::BAD = -2;
    /*static*/ const ReductionOpID CollectiveCheckReduction::REDOP =
        LEGION_MAX_APPLICATION_REDOP_ID + 1;

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::apply<true>(LHS& lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs > IDENTITY);
      if (lhs != IDENTITY)
      {
        if (lhs != rhs)
          lhs = BAD;
      }
      else
        lhs = rhs;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::apply<false>(LHS& lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      LHS* ptr = &lhs;
      LHS temp = *ptr;
      while ((temp != BAD) && (temp != rhs))
      {
        if (temp != IDENTITY)
          temp = __sync_val_compare_and_swap(ptr, temp, BAD);
        else
          temp = __sync_val_compare_and_swap(ptr, temp, rhs);
      }
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::fold<true>(RHS& rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      legion_assert(rhs2 > IDENTITY);
      if (rhs1 != IDENTITY)
      {
        if (rhs1 != rhs2)
          rhs1 = BAD;
      }
      else
        rhs1 = rhs2;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::fold<false>(RHS& rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      RHS* ptr = &rhs1;
      RHS temp = *ptr;
      while ((temp != BAD) && (temp != rhs2))
      {
        if (temp != IDENTITY)
          temp = __sync_val_compare_and_swap(ptr, temp, BAD);
        else
          temp = __sync_val_compare_and_swap(ptr, temp, rhs2);
      }
    }

    /////////////////////////////////////////////////////////////
    // Check Reduction
    /////////////////////////////////////////////////////////////

    /*static*/ const CloseCheckReduction::CloseCheckValue
        CloseCheckReduction::IDENTITY = CloseCheckReduction::CloseCheckValue();
    /*static*/ const CloseCheckReduction::CloseCheckValue
        CloseCheckReduction::identity = IDENTITY;
    /*static*/ const ReductionOpID CloseCheckReduction::REDOP =
        LEGION_MAX_APPLICATION_REDOP_ID + 2;

    //--------------------------------------------------------------------------
    CloseCheckReduction::CloseCheckValue::CloseCheckValue(void)
      : operation_index(0), barrier(RtBarrier::NO_RT_BARRIER),
        region(LogicalRegion::NO_REGION), partition(LogicalPartition::NO_PART),
        is_region(true), read_only(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CloseCheckReduction::CloseCheckValue::CloseCheckValue(
        Operation* op, RtBarrier bar, RegionTreeNode* node, bool read)
      : operation_index(op->get_context_index()), barrier(bar),
        is_region(node->is_region()), read_only(read)
    //--------------------------------------------------------------------------
    {
      if (is_region)
        region = node->as_region_node()->handle;
      else
        partition = node->as_partition_node()->handle;
    }

    //--------------------------------------------------------------------------
    bool CloseCheckReduction::CloseCheckValue::operator==(
        const CloseCheckValue& rhs) const
    //--------------------------------------------------------------------------
    {
      if (operation_index != rhs.operation_index)
        return false;
      if (barrier != rhs.barrier)
        return false;
      if (read_only != rhs.read_only)
        return false;
      if (is_region != rhs.is_region)
        return false;
      if (is_region)
      {
        if (region != rhs.region)
          return false;
      }
      else
      {
        if (partition != rhs.partition)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::apply<true>(LHS& lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      // Only copy over if LHS is the identity
      // This will effectively do a broadcast of one value
      if (lhs == IDENTITY)
        lhs = rhs;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::apply<false>(LHS& lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      // Not supported at the moment
      std::abort();
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::fold<true>(RHS& rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      // Only copy over if RHS1 is the identity
      // This will effectively do a broadcast of one value
      if (rhs1 == IDENTITY)
        rhs1 = rhs2;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::fold<false>(RHS& rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      // Not supported at the moment
      std::abort();
    }
#endif  // LEGION_DEBUG_COLLECTIVES

    /////////////////////////////////////////////////////////////
    // Shard Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : manager(ctx->shard_manager), context(ctx),
        local_shard(ctx->owner_shard->shard_id),
        collective_index(ctx->get_next_collective_index(loc))
    //--------------------------------------------------------------------------
    {
      context->add_base_resource_ref(COLLECTIVE_REF);
    }

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(ReplicateContext* ctx, CollectiveID id)
      : manager(ctx->shard_manager), context(ctx),
        local_shard(ctx->owner_shard->shard_id), collective_index(id)
    //--------------------------------------------------------------------------
    {
      context->add_base_resource_ref(COLLECTIVE_REF);
    }

    //--------------------------------------------------------------------------
    ShardCollective::~ShardCollective(void)
    //--------------------------------------------------------------------------
    {
      // Unregister this with the context
      context->unregister_collective(this);
      if (context->remove_base_resource_ref(COLLECTIVE_REF))
        delete context;
    }

    //--------------------------------------------------------------------------
    void ShardCollective::perform_collective_sync(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      perform_collective_async(precondition);
      perform_collective_wait(true /*block*/);
    }

    //--------------------------------------------------------------------------
    void ShardCollective::DeferCollectiveArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      collective->perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool ShardCollective::defer_collective_async(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(precondition.exists());
      DeferCollectiveArgs args(this);
      if (precondition.has_triggered())
        return false;
      runtime->issue_runtime_meta_task(
          args, LG_LATENCY_DEFERRED_PRIORITY, precondition);
      return true;
    }

    //--------------------------------------------------------------------------
    int ShardCollective::convert_to_index(ShardID id, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // shift everything so that the target shard is at index 0
      const int result =
          ((id + (manager->total_shards - origin)) % manager->total_shards);
      return result;
    }

    //--------------------------------------------------------------------------
    ShardID ShardCollective::convert_to_shard(int index, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // Add target then take the modulus
      const ShardID result = (index + origin) % manager->total_shards;
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Gather Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BroadcastCollective::BroadcastCollective(
        CollectiveIndexLocation loc, ReplicateContext* ctx, ShardID o)
      : ShardCollective(loc, ctx), origin(o),
        shard_collective_radix(ctx->get_shard_collective_radix())
    //--------------------------------------------------------------------------
    {
      if (local_shard != origin)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    BroadcastCollective::BroadcastCollective(
        ReplicateContext* ctx, CollectiveID id, ShardID o)
      : ShardCollective(ctx, id), origin(o),
        shard_collective_radix(ctx->get_shard_collective_radix())
    //--------------------------------------------------------------------------
    {
      if (local_shard != origin)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    BroadcastCollective::~BroadcastCollective(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void BroadcastCollective::perform_collective_async(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_shard == origin);
      if (precondition.exists() && defer_collective_async(precondition))
        return;
      // Register this with the context
      context->register_collective(this);
      send_messages();
    }

    //--------------------------------------------------------------------------
    RtEvent BroadcastCollective::perform_collective_wait(bool block /*=true*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_shard != origin);
      // Pull a copy of the done event onto the stack in case the
      // collective ends up being deleted immediately after it is registered
      const RtEvent result = done_event;
      // Register this with the context
      context->register_collective(this);
      if (!result.has_triggered())
      {
        if (block)
          result.wait();
        else
          return result;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::handle_collective_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_shard != origin);
      // No need for the lock since this is only written to once
      unpack_collective(derez);
      // Send our messages
      send_messages();
      // Pull the done event onto the stack in case post_broadcast
      // ends up deleting the object, see CreateCollectiveFillView
      const RtUserEvent done = done_event;
      // Then trigger our event to indicate that we are ready
      Runtime::trigger_event(done, post_broadcast());
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      if (done_event.exists())
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent BroadcastCollective::get_done_event(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(local_shard != origin);
      return done_event;
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::send_messages(void) const
    //--------------------------------------------------------------------------
    {
      const MessageKind message = get_message_kind();
      const int local_index = convert_to_index(local_shard, origin);
      for (int idx = 1; idx <= shard_collective_radix; idx++)
      {
        const int target_index = local_index * shard_collective_radix + idx;
        if (target_index >= int(manager->total_shards))
          break;
        ShardID target = convert_to_shard(target_index, origin);
        ShardCollectiveMessage rez(message);
        {
          rez.serialize(manager->did);
          rez.serialize(target);
          rez.serialize(collective_index);
          pack_collective(rez);
        }
        manager->send_collective_message(target, rez);
      }
    }

    /////////////////////////////////////////////////////////////
    // Gather Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GatherCollective::GatherCollective(
        CollectiveIndexLocation loc, ReplicateContext* ctx, ShardID t)
      : ShardCollective(loc, ctx), target(t),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        expected_notifications(compute_expected_notifications()),
        received_notifications(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    GatherCollective::GatherCollective(
        ReplicateContext* ctx, CollectiveID id, ShardID t)
      : ShardCollective(ctx, id), target(t),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        expected_notifications(compute_expected_notifications()),
        received_notifications(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    GatherCollective::~GatherCollective(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void GatherCollective::perform_collective_async(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && defer_collective_async(precondition))
        return;
      // Register this with the context
      context->register_collective(this);
      bool done = false;
      {
        AutoLock c_lock(collective_lock);
        legion_assert(received_notifications < expected_notifications);
        done = (++received_notifications == expected_notifications);
        // This is a bit tricky but if we're done increment the expected
        // notifications by one to make sure something calling get_done_event
        // doesn't think we're done too early
        if (done)
          received_notifications--;
      }
      if (done)
      {
        if (local_shard != target)
          send_message();
        RtEvent postcondition = post_gather();
        RtUserEvent to_trigger;
        {
          AutoLock c_lock(collective_lock);
          legion_assert((received_notifications + 1) == expected_notifications);
          // remove the guard
          received_notifications++;
          if (done_event.to_trigger.exists())
          {
            to_trigger = done_event.to_trigger;
            done_event.postcondition = to_trigger;
          }
          else
            done_event.postcondition = postcondition;
        }
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger, postcondition);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent GatherCollective::get_done_event(void)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      if (received_notifications < expected_notifications)
      {
        if (!done_event.to_trigger.exists())
          done_event.to_trigger = Runtime::create_rt_user_event();
        return done_event.to_trigger;
      }
      else
        return done_event.postcondition;
    }

    //--------------------------------------------------------------------------
    RtEvent GatherCollective::perform_collective_wait(bool block /*=true*/)
    //--------------------------------------------------------------------------
    {
      const RtEvent result = get_done_event();
      if (!block)
        return result;
      if (!result.exists())
        return result;
      if (!result.has_triggered())
        result.wait();
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void GatherCollective::handle_collective_message(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        // Hold the lock while doing these operations
        AutoLock c_lock(collective_lock);
        // Unpack the result
        unpack_collective(derez);
        legion_assert(received_notifications < expected_notifications);
        done = (++received_notifications == expected_notifications);
        // This is a bit tricky but if we're done increment the expected
        // notifications by one to make sure something calling get_done_event
        // doesn't think we're done too early
        if (done)
          received_notifications--;
      }
      if (done)
      {
        if (local_shard != target)
          send_message();
        RtEvent postcondition = post_gather();
        RtUserEvent to_trigger;
        {
          AutoLock c_lock(collective_lock);
          legion_assert((received_notifications + 1) == expected_notifications);
          // Remove the guard
          received_notifications++;
          if (done_event.to_trigger.exists())
          {
            to_trigger = done_event.to_trigger;
            done_event.postcondition = to_trigger;
          }
          else
            done_event.postcondition = postcondition;
        }
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger, postcondition);
      }
    }

    //--------------------------------------------------------------------------
    void GatherCollective::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      legion_assert(received_notifications == 0);
      received_notifications = expected_notifications;
      if (done_event.to_trigger.exists())
      {
        RtUserEvent to_trigger = done_event.to_trigger;
        Runtime::trigger_event(to_trigger);
        done_event.postcondition = RtEvent::NO_RT_EVENT;
      }
      else
        done_event.postcondition = RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void GatherCollective::send_message(void)
    //--------------------------------------------------------------------------
    {
      // Convert to our local index
      const int local_index = convert_to_index(local_shard, target);
      legion_assert(local_index > 0);  // should never be here for zero
      // Subtract by 1 and then divide to get the target (truncate)
      const int target_index = (local_index - 1) / shard_collective_radix;
      // Then convert back to the target
      ShardID next = convert_to_shard(target_index, target);
      ShardCollectiveMessage rez(get_message_kind());
      {
        rez.serialize(manager->did);
        rez.serialize(next);
        rez.serialize(collective_index);
        AutoLock c_lock(collective_lock, false /*exclusive*/);
        pack_collective(rez);
      }
      manager->send_collective_message(next, rez);
    }

    //--------------------------------------------------------------------------
    int GatherCollective::compute_expected_notifications(void) const
    //--------------------------------------------------------------------------
    {
      int result = 1;  // always have one arriver for ourself
      const int index = convert_to_index(local_shard, target);
      for (int idx = 1; idx <= shard_collective_radix; idx++)
      {
        const int source_index = index * shard_collective_radix + idx;
        if (source_index >= int(manager->total_shards))
          break;
        result++;
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // All Gather Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::AllGatherCollective(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : ShardCollective(loc, ctx), participants(nullptr),
        total_shards(manager->total_shards), local_index(local_shard),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        participating(local_index < shard_collective_participating_shards),
        reorder_stages(nullptr), pending_send_ready_stages(0),
        done_triggered(false)
    //--------------------------------------------------------------------------
    {
      initialize_collective();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::AllGatherCollective(
        ReplicateContext* ctx, CollectiveID id)
      : ShardCollective(ctx, id), participants(nullptr),
        total_shards(manager->total_shards), local_index(local_shard),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        participating(local_index < shard_collective_participating_shards),
        reorder_stages(nullptr), pending_send_ready_stages(0),
        done_triggered(false)
    //--------------------------------------------------------------------------
    {
      initialize_collective();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::AllGatherCollective(
        ReplicateContext* ctx, CollectiveID id,
        const std::vector<ShardID>& parts)
      : ShardCollective(ctx, id), participants(&parts),
        total_shards(parts.size()), reorder_stages(nullptr),
        pending_send_ready_stages(0), done_triggered(false)
    //--------------------------------------------------------------------------
    {
      legion_assert(std::is_sorted(parts.begin(), parts.end()));
      legion_assert(
          std::binary_search(parts.begin(), parts.end(), local_shard));
      std::vector<ShardID>::const_iterator finder =
          std::lower_bound(parts.begin(), parts.end(), local_shard);
      legion_assert(finder != parts.end());
      legion_assert(*finder == local_shard);
      local_index = std::distance(parts.begin(), finder);
      shard_collective_radix = runtime->legion_collective_radix;
      participating = configure_collective_settings(
          parts.size(), local_index, shard_collective_radix,
          shard_collective_log_radix, shard_collective_stages,
          shard_collective_participating_shards, shard_collective_last_radix);
      initialize_collective();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::initialize_collective(void)
    //--------------------------------------------------------------------------
    {
      if (total_shards > 1)
      {
        // We already have our contributions for each stage so
        // we can set the inditial participants to 1
        if (participating)
        {
          legion_assert(shard_collective_stages > 0);
          sent_stages.resize(shard_collective_stages, false);
          stage_notifications.resize(shard_collective_stages, 1);
          // Stage 0 always starts with 0 notifications since we'll
          // explictcly arrive on it
          stage_notifications[0] = 0;
        }
        done_event = Runtime::create_rt_user_event();
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::~AllGatherCollective(void)
    //--------------------------------------------------------------------------
    {
      if (reorder_stages != nullptr)
      {
        legion_assert(reorder_stages->empty());
        delete reorder_stages;
      }
      if (participating)
      {
        // We should have sent all our stages before being deleted
        for (unsigned idx = 0; idx < sent_stages.size(); idx++)
        {
          legion_assert(sent_stages[idx]);
        }
        legion_assert(done_triggered);
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::perform_collective_async(RtEvent pre)
    //--------------------------------------------------------------------------
    {
      if (pre.exists() && defer_collective_async(pre))
        return;
      // Register this with the context
      context->register_collective(this);
      if (total_shards <= 1)
      {
        post_complete_exchange().wait();
        done_triggered = true;
        return;
      }
      // See if we are a participating shard or not
      if (participating)
      {
        // We are a participating shard
        // See if we are waiting for an initial notification
        // if not we can just send our message now
        if ((int(total_shards) == shard_collective_participating_shards) ||
            (local_index >=
             (int(total_shards) - shard_collective_participating_shards)))
        {
          const bool all_stages_done = initiate_collective();
          if (all_stages_done)
            complete_exchange();
        }
      }
      else
      {
        // We are not a participating shard
        // so we just have to send notification to one shard
        send_remainder_stage();
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    RtEvent AllGatherCollective<INORDER>::perform_collective_wait(
        bool block /*=true*/)
    //--------------------------------------------------------------------------
    {
      if (total_shards <= 1)
        return RtEvent::NO_RT_EVENT;
      if (!done_event.has_triggered())
      {
        if (block)
          done_event.wait();
        else
          return done_event;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::handle_collective_message(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      int stage;
      derez.deserialize(stage);
      legion_assert(participating || (stage == -1));
      unpack_stage(stage, derez);
      bool all_stages_done = false;
      if (stage == -1)
      {
        if (!participating)
          all_stages_done = true;
        else  // we can now initiate the collective
          all_stages_done = initiate_collective();
      }
      else
        all_stages_done = send_ready_stages();
      if (all_stages_done)
        complete_exchange();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      // make it look like we sent all the stages
      for (unsigned idx = 0; idx < sent_stages.size(); idx++)
        sent_stages[idx] = true;
      legion_assert(!done_triggered);
      legion_assert(!done_event.has_triggered());
      // Trigger the user event
      Runtime::trigger_event(done_event);
      done_triggered = true;
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::construct_message(
        ShardID target, int stage, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->did);
      rez.serialize(target);
      rez.serialize(collective_index);
      rez.serialize(stage);
      AutoLock c_lock(collective_lock, false /*exclusive*/);
      pack_collective_stage(target, rez, stage);
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    bool AllGatherCollective<INORDER>::initiate_collective(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          participating);  // should only get this for participating shards
      {
        AutoLock c_lock(collective_lock);
        legion_assert(!sent_stages.empty());
        legion_assert(!sent_stages[0]);  // stage 0 shouldn't be sent yet
        legion_assert(!stage_notifications.empty());
        if (shard_collective_stages == 1)
        {
          legion_assert(stage_notifications[0] < shard_collective_last_radix);
        }
        else
        {
          legion_assert(stage_notifications[0] < shard_collective_radix);
        }
        stage_notifications[0]++;
        // Increment our guard to prevent deletion of the collective
        // object while we are still traversing
        pending_send_ready_stages++;
      }
      return send_ready_stages(0 /*start stage*/);
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::send_remainder_stage(void)
    //--------------------------------------------------------------------------
    {
      const MessageKind message = get_message_kind();
      if (participating)
      {
        // Send back to the shards that are not participating
        ShardID target =
            (participants == nullptr) ?
                (local_shard + shard_collective_participating_shards) :
                participants->at(
                    local_index + shard_collective_participating_shards);
        legion_assert(target < manager->total_shards);
        ShardCollectiveMessage rez(message);
        construct_message(target, -1 /*stage*/, rez);
        manager->send_collective_message(target, rez);
      }
      else
      {
        // Send to a node that is participating
        ShardID target =
            (participants == nullptr) ?
                local_shard % shard_collective_participating_shards :
                participants->at(
                    local_index % shard_collective_participating_shards);
        ShardCollectiveMessage rez(message);
        construct_message(target, -1 /*stage*/, rez);
        manager->send_collective_message(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    bool AllGatherCollective<INORDER>::send_ready_stages(const int start_stage)
    //--------------------------------------------------------------------------
    {
      legion_assert(participating);
      // Iterate through the stages and send any that are ready
      // Remember that stages have to be done in order
      bool sent_previous_stage = false;
      const MessageKind message = get_message_kind();
      for (int stage = start_stage; stage < shard_collective_stages; stage++)
      {
        {
          AutoLock c_lock(collective_lock);
          if (sent_previous_stage)
          {
            legion_assert(!sent_stages[stage - 1]);
            sent_stages[stage - 1] = true;
            sent_previous_stage = false;
          }
          // If this stage has already been sent then we can keep going
          if (sent_stages[stage])
            continue;
          legion_assert(pending_send_ready_stages > 0);
          // Check to see if we're sending this stage
          // We need all the notifications from the previous stage before
          // we can send this stage
          if (stage > 0)
          {
            // We can't have multiple threads doing sends at the same time
            // so make sure that only the last one is going through doing work
            // but stage 0 is because it is always sent by the initiator so
            // don't check this until we're past the first stage
            if ((stage_notifications[stage - 1] < shard_collective_radix) ||
                (pending_send_ready_stages > 1))
            {
              // Remove our guard before exiting early
              pending_send_ready_stages--;
              return false;
            }
            else if (INORDER && (reorder_stages != nullptr))
            {
              // Check to see if we have any unhandled messages for
              // the previous stage that we need to handle before sending
              std::map<int, std::vector<std::pair<void*, size_t> > >::iterator
                  finder = reorder_stages->find(stage - 1);
              if (finder != reorder_stages->end())
              {
                // Perform the handling for the buffered messages now
                for (const std::pair<void*, size_t>& buffered_msg :
                     finder->second)
                {
                  Deserializer derez(buffered_msg.first, buffered_msg.second);
                  unpack_collective_stage(derez, finder->first);
                  free(buffered_msg.first);
                }
                reorder_stages->erase(finder);
              }
            }
          }
          // If we get here then we can send the stage
        }
        // Now we can do the send
        if (stage == (shard_collective_stages - 1))
        {
          for (int r = 1; r < shard_collective_last_radix; r++)
          {
            const ShardID target =
                (participants == nullptr) ?
                    local_shard ^ (r << (stage * shard_collective_log_radix)) :
                    participants->at(
                        local_index ^
                        (r << (stage * shard_collective_log_radix)));
            ShardCollectiveMessage rez(message);
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
        else
        {
          for (int r = 1; r < shard_collective_radix; r++)
          {
            const ShardID target =
                (participants == nullptr) ?
                    local_shard ^ (r << (stage * shard_collective_log_radix)) :
                    participants->at(
                        local_index ^
                        (r << (stage * shard_collective_log_radix)));
            ShardCollectiveMessage rez(message);
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
        sent_previous_stage = true;
      }
      // If we make it here, then we sent the last stage, check to see
      // if we've seen all the notifications for it
      AutoLock c_lock(collective_lock);
      if (sent_previous_stage)
      {
        legion_assert(!sent_stages[shard_collective_stages - 1]);
        sent_stages[shard_collective_stages - 1] = true;
      }
      // Remove our pending guard and then check to see if we are done
      legion_assert(pending_send_ready_stages > 0);
      if (((--pending_send_ready_stages) == 0) &&
          (stage_notifications.back() == shard_collective_last_radix))
      {
        legion_assert(!done_triggered);
        done_triggered = true;
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::unpack_stage(
        int stage, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      // Do the unpack first while holding the lock
      if (INORDER && (stage >= 0))
      {
        // Check to see if we can handle this message now or whether we
        // need to buffer it for the future because we have not finished
        // sending the current stage yet or not
        if (!sent_stages[stage])
        {
          // Buffer this message until the stage is sent as well
          const size_t buffer_size = derez.get_remaining_bytes();
          void* buffer = malloc(buffer_size);
          memcpy(buffer, derez.get_current_pointer(), buffer_size);
          derez.advance_pointer(buffer_size);
          if (reorder_stages == nullptr)
            reorder_stages =
                new std::map<int, std::vector<std::pair<void*, size_t> > >();
          (*reorder_stages)[stage].emplace_back(
              std::pair<void*, size_t>(buffer, buffer_size));
        }
        else
          unpack_collective_stage(derez, stage);
      }
      else  // Just do the unpack here immediately
        unpack_collective_stage(derez, stage);
      if (stage >= 0)
      {
        legion_assert(stage < int(stage_notifications.size()));
        if (stage < (shard_collective_stages - 1))
        {
          legion_assert(stage_notifications[stage] < shard_collective_radix);
        }
        else
        {
          legion_assert(
              stage_notifications[stage] < shard_collective_last_radix);
        }
        stage_notifications[stage]++;
        // Increment our guard to prevent deletion of the collective
        // object while we are still traversing
        pending_send_ready_stages++;
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      if ((reorder_stages != nullptr) && !reorder_stages->empty())
      {
        legion_assert(reorder_stages->size() == 1);
        std::map<int, std::vector<std::pair<void*, size_t> > >::iterator
            remaining = reorder_stages->begin();
        for (const std::pair<void*, size_t>& buffered_msg : remaining->second)
        {
          Deserializer derez(buffered_msg.first, buffered_msg.second);
          unpack_collective_stage(derez, remaining->first);
          free(buffered_msg.first);
        }
        reorder_stages->erase(remaining);
      }
      // See if we have to send a message back to a non-participating shard
      if ((int(total_shards) > shard_collective_participating_shards) &&
          (local_index <
           int(total_shards - shard_collective_participating_shards)))
        send_remainder_stage();
      // Pull this onto the stack in case post_complete_exchange ends up
      // deleting the object
      const RtUserEvent to_trigger = done_event;
      const RtEvent precondition = post_complete_exchange();
      // Only after we send the message and do the post can we signal we're done
      Runtime::trigger_event(to_trigger, precondition);
    }

    template class AllGatherCollective<true>;
    template class AllGatherCollective<false>;

    /////////////////////////////////////////////////////////////
    // All Reduce Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    AllReduceCollective<REDOP, INORDER>::AllReduceCollective(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : AllGatherCollective<INORDER>(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    AllReduceCollective<REDOP, INORDER>::AllReduceCollective(
        ReplicateContext* ctx, CollectiveID id)
      : AllGatherCollective<INORDER>(ctx, id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    AllReduceCollective<REDOP, INORDER>::~AllReduceCollective(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    void AllReduceCollective<REDOP, INORDER>::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(value);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    void AllReduceCollective<REDOP, INORDER>::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      if (this->participating)
      {
        typename REDOP::RHS next;
        derez.deserialize(next);
        REDOP::template fold<true /*exclusive*/>(value, next);
      }
      else  // Just overwrite in this case since we're not participating
        derez.deserialize(value);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    void AllReduceCollective<REDOP, INORDER>::async_all_reduce(
        typename REDOP::RHS val)
    //--------------------------------------------------------------------------
    {
      value = val;
      this->perform_collective_async();
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    RtEvent AllReduceCollective<REDOP, INORDER>::wait_all_reduce(bool block)
    //--------------------------------------------------------------------------
    {
      return this->perform_collective_wait(block);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    typename REDOP::RHS AllReduceCollective<REDOP, INORDER>::sync_all_reduce(
        typename REDOP::RHS val)
    //--------------------------------------------------------------------------
    {
      async_all_reduce(val);
      return get_result();
    }

    //--------------------------------------------------------------------------
    template<typename REDOP, bool INORDER>
    typename REDOP::RHS AllReduceCollective<REDOP, INORDER>::get_result(void)
    //--------------------------------------------------------------------------
    {
      // Wait for the results to be ready
      wait_all_reduce(true);
      return value;
    }

    // Instantiate this for a common use case
    template class AllReduceCollective<SumReduction<bool>, false>;
    template class AllReduceCollective<ProdReduction<bool>, false>;
    template class AllReduceCollective<MaxReduction<uint32_t>, false>;
    template class AllReduceCollective<MaxReduction<uint64_t>, false>;
    template class AllReduceCollective<MinReduction<unsigned>, false>;
    template class AllReduceCollective<ReplTraceOp::StatusReduction, false>;

    /////////////////////////////////////////////////////////////
    // Buffer Broadcast
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BufferBroadcast::BufferBroadcast(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : BroadcastCollective(loc, ctx, ctx->owner_shard->shard_id),
        buffer(nullptr), size(0), own(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    BufferBroadcast::BufferBroadcast(CollectiveID id, ReplicateContext* ctx)
      : BroadcastCollective(ctx, id, ctx->owner_shard->shard_id),
        buffer(nullptr), size(0), own(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void BufferBroadcast::broadcast(void* b, size_t s, bool copy)
    //--------------------------------------------------------------------------
    {
      legion_assert(buffer == nullptr);
      if (copy)
      {
        size = s;
        buffer = malloc(size);
        memcpy(buffer, b, size);
        own = true;
      }
      else
      {
        buffer = b;
        size = s;
        own = false;
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    const void* BufferBroadcast::get_buffer(size_t& s, bool wait)
    //--------------------------------------------------------------------------
    {
      if (wait)
        perform_collective_wait();
      s = size;
      return buffer;
    }

    //--------------------------------------------------------------------------
    void BufferBroadcast::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(size);
      if (size > 0)
        rez.serialize(buffer, size);
    }

    //--------------------------------------------------------------------------
    void BufferBroadcast::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(size);
      if (size > 0)
      {
        legion_assert(buffer == nullptr);
        buffer = malloc(size);
        derez.deserialize(buffer, size);
        own = true;
      }
    }

    /////////////////////////////////////////////////////////////
    // Buffer Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BufferExchange::BufferExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    BufferExchange::~BufferExchange(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const ShardID, std::pair<void*, size_t> >& result :
           results)
        if (result.second.second > 0)
          free(result.second.first);
    }

    //--------------------------------------------------------------------------
    void BufferExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (const std::pair<const ShardID, std::pair<void*, size_t> >& result :
           results)
      {
        rez.serialize(result.first);
        rez.serialize(result.second.second);
        if (result.second.second > 0)
          rez.serialize(result.second.first, result.second.second);
      }
    }

    //--------------------------------------------------------------------------
    void BufferExchange::unpack_collective_stage(Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize(num_results);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        ShardID shard;
        derez.deserialize(shard);
        size_t size;
        derez.deserialize(size);
        if (results.find(shard) != results.end())
        {
          derez.advance_pointer(size);
          continue;
        }
        if (size > 0)
        {
          void* buffer = malloc(size);
          derez.deserialize(buffer, size);
          results[shard] = std::make_pair(buffer, size);
        }
        else
          results[shard] = std::make_pair<void*, size_t>(nullptr, 0);
      }
    }

    //--------------------------------------------------------------------------
    const std::map<ShardID, std::pair<void*, size_t> >&
        BufferExchange::exchange_buffers(
            void* value, size_t size, bool keep_self)
    //--------------------------------------------------------------------------
    {
      // Can put this in without the lock since we haven't started yet
      results[local_shard] = std::make_pair(value, size);
      perform_collective_sync();
      // Remove ourselves after we're done
      if (!keep_self)
        results.erase(local_shard);
      return results;
    }

    //--------------------------------------------------------------------------
    RtEvent BufferExchange::exchange_buffers_async(
        void* value, size_t size, bool keep_self)
    //--------------------------------------------------------------------------
    {
      // Can put this in without the lock since we haven't started yet
      results[local_shard] = std::make_pair(value, size);
      perform_collective_async();
      return perform_collective_wait(false /*block*/);
    }

    //--------------------------------------------------------------------------
    const std::map<ShardID, std::pair<void*, size_t> >&
        BufferExchange::sync_buffers(bool keep_self)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait(true /*block*/);
      if (!keep_self)
        results.erase(local_shard);
      return results;
    }

    /////////////////////////////////////////////////////////////
    // Future All Reduce Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureAllReduceCollective::FutureAllReduceCollective(
        Operation* o, CollectiveIndexLocation loc, ReplicateContext* ctx,
        ReductionOpID id, const ReductionOp* op)
      : AllGatherCollective(loc, ctx), op(o), redop(op), redop_id(id),
        finished(Runtime::create_ap_user_event(nullptr)), instance(nullptr),
        shadow_instance(nullptr), current_stage(-1), pack_shadow(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FutureAllReduceCollective::FutureAllReduceCollective(
        Operation* o, ReplicateContext* ctx, CollectiveID rid, ReductionOpID id,
        const ReductionOp* op)
      : AllGatherCollective(ctx, id), op(o), redop(op), redop_id(rid),
        finished(Runtime::create_ap_user_event(nullptr)), instance(nullptr),
        shadow_instance(nullptr), current_stage(-1), pack_shadow(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FutureAllReduceCollective::~FutureAllReduceCollective(void)
    //--------------------------------------------------------------------------
    {
      if (shadow_instance != nullptr)
      {
        if (!shadow_reads.empty())
        {
          // Reads always dominate the most recent write
          const ApEvent precondition =
              Runtime::merge_events(nullptr, shadow_reads);
          if (!shadow_instance->defer_deletion(precondition))
            delete shadow_instance;
        }
        else if (!shadow_instance->defer_deletion(shadow_ready))
          delete shadow_instance;
      }
    }

    //--------------------------------------------------------------------------
    void FutureAllReduceCollective::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      // The first time we pack a stage we merge any values that we had
      // unpacked earlier as they are needed for sending this stage for
      // the first time.
      if (stage != current_stage)
      {
        bool check_for_shadow = true;
        if (!pending_reductions.empty())
        {
          std::map<int, std::map<ShardID, PendingReduction> >::iterator next =
              pending_reductions.begin();
          if (next->first == current_stage)
          {
            // Apply all of these to the destination instance
            ApEvent new_instance_ready = perform_reductions(next->second);
            // Check to see if we'll be able to pack up instance by value
            if (new_instance_ready.exists() || !instance->can_pack_by_value())
            {
              if (stage == -1)
              {
                legion_assert(current_stage == (shard_collective_stages - 1));
                instance_ready = new_instance_ready;
                // No need for packing the shadow on the way out
                pack_shadow = false;
              }
              else
              {
                // Have to copy this to the shadow instance because we can't
                // do this in-place without support from Realm
                if (shadow_instance != nullptr)
                {
                  // Handle WAR dependences which dominate previous write
                  if (!shadow_reads.empty())
                  {
                    shadow_ready = Runtime::merge_events(nullptr, shadow_reads);
                    shadow_reads.clear();
                  }
                }
                else
                  create_shadow_instance();
                // Copy to the shadow instance, note this incorporates
                // any of the read postconditions from the previous stage
                // so we know it's safe to write here
                shadow_ready = shadow_instance->copy_from(
                    instance, op,
                    Runtime::merge_events(
                        nullptr, shadow_ready, new_instance_ready));
                instance_ready = shadow_ready;
                pack_shadow = true;
              }
            }
            else
            {
              instance_ready = new_instance_ready;
              pack_shadow = false;
            }
            pending_reductions.erase(next);
            // No need for the check
            check_for_shadow = false;
          }
        }
        if (check_for_shadow)
        {
          // should be stage 0 (first stage) or final stage 0
          legion_assert((stage == 0) || (stage == -1));
          if (stage == -1)
          {
            legion_assert(current_stage == (shard_collective_stages - 1));
            // No need for packing the shadow on the way out
            pack_shadow = false;
          }
          else if (instance_ready.exists() || !instance->can_pack_by_value())
          {
            legion_assert(current_stage == -1);
            // Have to make a copy in this case
            if (shadow_instance != nullptr)
            {
              // Handle WAR dependences which dominate previous write
              if (!shadow_reads.empty())
              {
                shadow_ready = Runtime::merge_events(nullptr, shadow_reads);
                shadow_reads.clear();
              }
            }
            else
              create_shadow_instance();
            shadow_ready = shadow_instance->copy_from(
                instance, op,
                Runtime::merge_events(nullptr, shadow_ready, instance_ready));
            instance_ready = shadow_ready;
            pack_shadow = true;
          }
        }
        current_stage = stage;
      }
      rez.serialize(local_shard);
      if (pack_shadow)
      {
        if (!shadow_instance->pack_instance(
                rez, shadow_ready, false /*pack ownership*/))
        {
          rez.serialize(shadow_ready);
          ApUserEvent reduction_done = Runtime::create_ap_user_event(nullptr);
          rez.serialize(reduction_done);
          shadow_reads.emplace_back(reduction_done);
        }
      }
      else
      {
        if (!instance->pack_instance(
                rez, instance_ready, false /*pack ownership*/))
        {
          rez.serialize(instance_ready);
          ApUserEvent reduction_done = Runtime::create_ap_user_event(nullptr);
          rez.serialize(reduction_done);
          // This happens in the case where we have a stage=-1 copy of
          // the instance to a participating shard, so we can just make
          // this the new precondition for copies coming back
          instance_ready = reduction_done;
        }
      }
    }

    //--------------------------------------------------------------------------
    void FutureAllReduceCollective::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      // We never eagerly do reductions as they can arrive out of order
      // and we can't apply them too early or we'll get duplicate
      // applications of reductions
      ShardID shard;
      derez.deserialize(shard);
      PendingReduction& pending = pending_reductions[stage][shard];
      pending.instance = FutureInstance::unpack_instance(derez);
      if (!pending.instance->is_meta_visible)
      {
        derez.deserialize(pending.precondition);
        derez.deserialize(pending.postcondition);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FutureAllReduceCollective::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Should be exactly one stage left
      legion_assert((pending_reductions.size() == 1) || (current_stage == -1));
      if (!pending_reductions.empty())
      {
        std::map<int, std::map<ShardID, PendingReduction> >::iterator last =
            pending_reductions.begin();
        if (last->first == -1)
        {
          // Copy-in last stage which includes our value so we just overwrite
          legion_assert(last->second.size() == 1);
          PendingReduction& pending = last->second.begin()->second;
          instance_ready = instance->copy_from(
              pending.instance, op,
              Runtime::merge_events(
                  nullptr, instance_ready, pending.precondition));
          if (pending.postcondition.exists())
            Runtime::trigger_event_untraced(
                pending.postcondition, instance_ready);
          delete pending.instance;
        }
        else
          instance_ready = perform_reductions(last->second);
        pending_reductions.erase(last);
      }
      legion_assert(finished.exists());
      // Trigger the finish event for the collective
      Runtime::trigger_event_untraced(finished, instance_ready);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void FutureAllReduceCollective::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(finished.exists());
      // Clean up the finished event we aren't going to trigger
      Runtime::trigger_event_untraced(finished);
      // elide the collective for the base class
      AllGatherCollective<false>::elide_collective();
    }

    //--------------------------------------------------------------------------
    void FutureAllReduceCollective::set_shadow_instance(FutureInstance* shadow)
    //--------------------------------------------------------------------------
    {
      legion_assert(shadow != nullptr);
      legion_assert(shadow_instance == nullptr);
      shadow_instance = shadow;
    }

    //--------------------------------------------------------------------------
    RtEvent FutureAllReduceCollective::async_reduce(
        FutureInstance* inst, ApEvent& ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(instance == nullptr);
      // Should be meta-visible unless it is a large instance
      legion_assert(
          inst->is_meta_visible || (inst->size > LEGION_MAX_RETURN_SIZE));
      // We should either have a shadow instance at this point or the nature
      // of the instance is that it is small enough and on system memory so
      // we will be able to do everything ourselves locally.
      legion_assert(
          (shadow_instance != nullptr) ||
          ((inst->is_meta_visible) && (inst->size <= LEGION_MAX_RETURN_SIZE)));
      instance = inst;
      instance_ready = ready;
      // Record that this is the event that will trigger when finished
      ready = finished;
      // This is a small, but important optimization:
      // For futures that are meta visible and less than the size of the
      // maximum pass-by-value size that are not ready yet, delay starting
      // the collective until they are ready so that we can do as much
      // as possible passing the data by value rather than having to defer
      // to Realm too much.
      if (inst->is_meta_visible && (inst->size <= LEGION_MAX_RETURN_SIZE) &&
          instance_ready.exists() &&
          !instance_ready.has_triggered_faultignorant())
        perform_collective_async(Runtime::protect_event(instance_ready));
      else
        perform_collective_async();
      return perform_collective_wait(false /*block*/);
    }

    //--------------------------------------------------------------------------
    void FutureAllReduceCollective::create_shadow_instance(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(shadow_instance == nullptr);
      legion_assert(instance->is_meta_visible);
      legion_assert(instance->size <= LEGION_MAX_RETURN_SIZE);
      // We're past the mapping stage of the pipeline at this point so
      // it is too late to be making instances the normal way through
      // eager allocation, so we need to just call malloc and make an
      // external allocation. This should only be happening for small
      // instances in system memory so it should not be a problem.
#ifdef __GNUC__
#if __GNUC__ >= 11
      // GCC is dumb and thinks we need to initialize this buffer
      // before we pass it into the create local call, which we
      // obviously don't need to do, so tell the compiler to shut up
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
      void* buffer = malloc(instance->size);
      shadow_instance =
          FutureInstance::create_local(buffer, instance->size, true /*own*/);
#ifdef __GNUC__
#if __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif
#endif
    }

    //--------------------------------------------------------------------------
    ApEvent FutureAllReduceCollective::perform_reductions(
        const std::map<ShardID, PendingReduction>& pending_reductions)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> postconditions;
      for (const std::pair<const ShardID, PendingReduction>& pending_reduction :
           pending_reductions)
      {
        ApEvent post = instance->reduce_from(
            pending_reduction.second.instance, op, redop_id, redop,
            false /*exclusive*/,
            Runtime::merge_events(
                nullptr, instance_ready,
                pending_reduction.second.precondition));
        if (pending_reduction.second.postcondition.exists())
          Runtime::trigger_event_untraced(
              pending_reduction.second.postcondition, post);
        delete pending_reduction.second.instance;
        if (post.exists())
          postconditions.emplace_back(post);
      }
      if (!postconditions.empty())
        return Runtime::merge_events(nullptr, postconditions);
      else
        return ApEvent::NO_AP_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Future Broadcast Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureBroadcastCollective::FutureBroadcastCollective(
        ReplicateContext* ctx, CollectiveIndexLocation loc, ShardID orig,
        Operation* o)
      : BroadcastCollective(loc, ctx, orig), op(o),
        finished(Runtime::create_ap_user_event(nullptr)), instance(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FutureBroadcastCollective::~FutureBroadcastCollective(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FutureBroadcastCollective::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      if (!instance->pack_instance(rez, write_event, false /*pack ownership*/))
      {
        rez.serialize(write_event);
        ApUserEvent remote_read_done = Runtime::create_ap_user_event(nullptr);
        rez.serialize(remote_read_done);
        read_events.emplace_back(remote_read_done);
      }
    }

    //--------------------------------------------------------------------------
    void FutureBroadcastCollective::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      FutureInstance* source = FutureInstance::unpack_instance(derez);
      if (!source->is_meta_visible)
      {
        ApEvent pre;
        derez.deserialize(pre);
        write_event = instance->copy_from(source, op, pre);
        ApUserEvent post;
        derez.deserialize(post);
        Runtime::trigger_event_untraced(post, write_event);
      }
      else
        write_event = instance->copy_from(source, op, ApEvent::NO_AP_EVENT);
      delete source;
    }

    //--------------------------------------------------------------------------
    RtEvent FutureBroadcastCollective::post_broadcast(void)
    //--------------------------------------------------------------------------
    {
      if (!read_events.empty())
      {
        if (write_event.exists())
          read_events.emplace_back(write_event);
        write_event = Runtime::merge_events(nullptr, read_events);
      }
      Runtime::trigger_event_untraced(finished, write_event);
      return postcondition;
    }

    //--------------------------------------------------------------------------
    void FutureBroadcastCollective::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      Runtime::trigger_event_untraced(finished);
      BroadcastCollective::elide_collective();
    }

    //--------------------------------------------------------------------------
    RtEvent FutureBroadcastCollective::async_broadcast(
        FutureInstance* inst, ApEvent precondition, RtEvent post)
    //--------------------------------------------------------------------------
    {
      legion_assert(instance == nullptr);
      // Should be meta-visible unless it is large
      legion_assert(
          inst->is_meta_visible || (inst->size > LEGION_MAX_RETURN_SIZE));
      instance = inst;
      if (is_origin())
      {
        legion_assert(!post.exists());
        write_event = precondition;
        perform_collective_async();
        return post_broadcast();
      }
      else
      {
        legion_assert(!precondition.exists());
        postcondition = post;
        return perform_collective_wait(false /*block*/);
      }
    }

    /////////////////////////////////////////////////////////////
    // Future Reduction Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureReductionCollective::FutureReductionCollective(
        ReplicateContext* ctx, CollectiveIndexLocation loc, ShardID orig,
        Operation* o, FutureBroadcastCollective* broad, const ReductionOp* red,
        ReductionOpID redid)
      : GatherCollective(loc, ctx, orig), op(o), broadcast(broad), redop(red),
        redop_id(redid), instance(nullptr)
    //--------------------------------------------------------------------------
    {
      legion_assert(target == broadcast->origin);
    }

    //--------------------------------------------------------------------------
    FutureReductionCollective::~FutureReductionCollective(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const ShardID, std::pair<FutureInstance*, ApEvent> >&
               pending_reduction : pending_reductions)
        delete pending_reduction.second.first;
    }

    //--------------------------------------------------------------------------
    void FutureReductionCollective::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      if (!pending_reductions.empty())
        perform_reductions();
      rez.serialize(local_shard);
      if (!instance->pack_instance(rez, ready, false /*pack ownership*/))
        rez.serialize(ready);
      // Note there is no need to track the remote reads here since we know
      // that the result is going come back to these instances in the
      // corresponding broadcast operation and that won't be able to happen
      // until the reduction reads are done anyway
    }

    //--------------------------------------------------------------------------
    void FutureReductionCollective::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ShardID shard;
      derez.deserialize(shard);
      FutureInstance* instance = FutureInstance::unpack_instance(derez);
      ApEvent ready;
      if (!instance->is_meta_visible)
        derez.deserialize(ready);
      pending_reductions[shard] = std::make_pair(instance, ready);
    }

    //--------------------------------------------------------------------------
    RtEvent FutureReductionCollective::post_gather(void)
    //--------------------------------------------------------------------------
    {
      if (is_target())
      {
        perform_reductions();
        return broadcast->async_broadcast(instance, ready);
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void FutureReductionCollective::async_reduce(
        FutureInstance* inst, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(instance == nullptr);
      // Should be meta-visible unless it is large
      legion_assert(
          inst->is_meta_visible || (inst->size > LEGION_MAX_RETURN_SIZE));
      instance = inst;
      ready = precondition;
      // This is a small, but important optimization:
      // For futures that are meta visible and less than the size of the
      // maximum pass-by-value size that are not ready yet, delay starting
      // the collective until they are ready so that we can do as much
      // as possible passing the data by value rather than having to defer
      // to Realm too much.
      if (inst->is_meta_visible && (inst->size <= LEGION_MAX_RETURN_SIZE) &&
          precondition.exists() && !precondition.has_triggered_faultignorant())
        perform_collective_async(Runtime::protect_event(precondition));
      else
        perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void FutureReductionCollective::perform_reductions(void) const
    //--------------------------------------------------------------------------
    {
      // Do these in order for determinism
      for (const std::pair<const ShardID, std::pair<FutureInstance*, ApEvent> >&
               pending_reduction : pending_reductions)
        ready = instance->reduce_from(
            pending_reduction.second.first, op, redop_id, redop,
            true /*exclusive*/,
            Runtime::merge_events(
                nullptr, ready, pending_reduction.second.second));
    }

    /////////////////////////////////////////////////////////////
    // Shard Participants Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardParticipantsExchange::ShardParticipantsExchange(
        ReplicateContext* ctx, CollectiveIndexLocation loc)
      : AllGatherCollective<false>(loc, ctx)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardParticipantsExchange::~ShardParticipantsExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ShardParticipantsExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(participants.size());
      for (const ShardID& shard_id : participants) rez.serialize(shard_id);
    }

    //--------------------------------------------------------------------------
    void ShardParticipantsExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_participants;
      derez.deserialize(num_participants);
      for (unsigned idx = 0; idx < num_participants; idx++)
      {
        ShardID shard;
        derez.deserialize(shard);
        participants.insert(shard);
      }
    }

    //--------------------------------------------------------------------------
    void ShardParticipantsExchange::exchange(bool participating)
    //--------------------------------------------------------------------------
    {
      if (participating)
        participants.insert(local_shard);
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool ShardParticipantsExchange::find_shard_participants(
        std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(shards.empty());
      perform_collective_wait();
      if (participants.size() < manager->total_shards)
      {
        shards.insert(shards.end(), participants.begin(), participants.end());
        return false;
      }
      else
        return true;
    }

    /////////////////////////////////////////////////////////////
    // Interfering Point Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    InterferingPointExchange<T>::InterferingPointExchange(
        ReplicateContext* ctx, CollectiveID id, T* o)
      : AllGatherCollective<true>(ctx, id), op(o)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    InterferingPointExchange<T>::~InterferingPointExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<typename T>
    void InterferingPointExchange<T>::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(domain_points.size());
      for (const std::pair<
               const unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
               domain_point : domain_points)
      {
        rez.serialize(domain_point.first);
        rez.serialize<size_t>(domain_point.second.size());
        for (const std::pair<DomainPoint, Domain>& point : domain_point.second)
        {
          rez.serialize(point.first);
          rez.serialize(point.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void InterferingPointExchange<T>::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_reqs;
      derez.deserialize(num_reqs);
      for (unsigned idx1 = 0; idx1 < num_reqs; idx1++)
      {
        unsigned req_idx;
        derez.deserialize(req_idx);
        size_t num_points;
        derez.deserialize(num_points);
        std::vector<std::pair<DomainPoint, Domain> >& points =
            domain_points[req_idx];
        if (!participating)
        {
          legion_assert(stage == -1);
          points.clear();
        }
        const unsigned offset = points.size();
        points.resize(offset + num_points);
        for (unsigned idx2 = 0; idx2 < num_points; idx2++)
        {
          derez.deserialize(points[offset + idx2].first);
          derez.deserialize(points[offset + idx2].second);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    RtEvent InterferingPointExchange<T>::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      op->finish_check_point_requirements(domain_points);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void InterferingPointExchange<T>::exchange_domain_points(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
            points)
    //--------------------------------------------------------------------------
    {
      domain_points.swap(points);
      perform_collective_async();
    }

    template class InterferingPointExchange<ReplIndexTask>;
    template class InterferingPointExchange<ReplIndexCopyOp>;
    template class InterferingPointExchange<ReplIndexAttachOp>;

    /////////////////////////////////////////////////////////////
    // Sharding Gather Collective
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardingGatherCollective::ShardingGatherCollective(
        ReplicateContext* ctx, ShardID target, CollectiveIndexLocation loc)
      : GatherCollective(loc, ctx, target)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardingGatherCollective::~ShardingGatherCollective(void)
    //--------------------------------------------------------------------------
    {
      // Make sure that we wait in case we still have messages to pass on
      perform_collective_wait();
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (const std::pair<const ShardID, ShardingID>& result : results)
      {
        rez.serialize(result.first);
        rez.serialize(result.second);
      }
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize(num_results);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        ShardID shard;
        derez.deserialize(shard);
        derez.deserialize(results[shard]);
      }
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::contribute(ShardingID value)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        legion_assert(results.find(local_shard) == results.end());
        results[local_shard] = value;
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool ShardingGatherCollective::validate(ShardingID value)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_target());
      // Wait for the results
      perform_collective_wait();
      for (const std::pair<const ShardID, ShardingID>& result : results)
      {
        if (result.second != value)
          return false;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Collective Mapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(
        const std::vector<AddressSpaceID>& spaces, size_t r)
      : total_spaces(spaces.size()), radix(r)
    //--------------------------------------------------------------------------
    {
      for (const AddressSpaceID& space : spaces)
        unique_sorted_spaces.add(space);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(const ShardMapping& mapping, size_t r)
      : radix(r)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < mapping.size(); idx++)
        unique_sorted_spaces.add(mapping[idx]);
      total_spaces = unique_sorted_spaces.size();
    }

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(Deserializer& derez, size_t total)
      : total_spaces(total)
    //--------------------------------------------------------------------------
    {
      legion_assert(total_spaces > 0);
      derez.deserialize(unique_sorted_spaces);
      legion_assert(total_spaces == unique_sorted_spaces.size());
      derez.deserialize(radix);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping::CollectiveMapping(const CollectiveMapping& rhs)
      : Collectable(), unique_sorted_spaces(rhs.unique_sorted_spaces),
        total_spaces(rhs.total_spaces), radix(rhs.radix)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::operator==(const CollectiveMapping& rhs) const
    //--------------------------------------------------------------------------
    {
      if (radix != rhs.radix)
        return false;
      return unique_sorted_spaces == rhs.unique_sorted_spaces;
    }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::operator!=(const CollectiveMapping& rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveMapping::get_parent(
        const AddressSpaceID origin, const AddressSpaceID local) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
      legion_assert(local_index < total_spaces);
      legion_assert(origin_index < total_spaces);
      const unsigned offset = convert_to_offset(local_index, origin_index);
      const unsigned index =
          convert_to_index((offset - 1) / radix, origin_index);
      const int result = unique_sorted_spaces.get_index(index);
      legion_assert(result >= 0);
      return result;
    }

    //--------------------------------------------------------------------------
    size_t CollectiveMapping::count_children(
        const AddressSpaceID origin, const AddressSpaceID local) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
      legion_assert(local_index < total_spaces);
      legion_assert(origin_index < total_spaces);
      const unsigned offset =
          radix * convert_to_offset(local_index, origin_index);
      size_t result = 0;
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_spaces)
          result++;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveMapping::get_children(
        const AddressSpaceID origin, const AddressSpaceID local,
        std::vector<AddressSpaceID>& children) const
    //--------------------------------------------------------------------------
    {
      const unsigned local_index = find_index(local);
      const unsigned origin_index = find_index(origin);
      legion_assert(local_index < total_spaces);
      legion_assert(origin_index < total_spaces);
      const unsigned offset =
          radix * convert_to_offset(local_index, origin_index);
      for (unsigned idx = 1; idx <= radix; idx++)
      {
        const unsigned child_offset = offset + idx;
        if (child_offset < total_spaces)
        {
          const unsigned index = convert_to_index(child_offset, origin_index);
          const int child = unique_sorted_spaces.get_index(index);
          legion_assert(child >= 0);
          children.emplace_back(child);
        }
      }
    }

    //--------------------------------------------------------------------------
    AddressSpaceID CollectiveMapping::find_nearest(AddressSpaceID search) const
    //--------------------------------------------------------------------------
    {
      unsigned first = 0;
      unsigned last = size() - 1;
      if (search < (*this)[first])
        return (*this)[first];
      if (search > (*this)[last])
        return (*this)[last];
      // Contained somewhere in the middle so binary
      // search for the two nearest options
      unsigned mid = 0;
      while (first <= last)
      {
        mid = (first + last) / 2;
        const AddressSpaceID midval = (*this)[mid];
        // If we find it just return it
        if (search == midval)
          return search;
        if (search < midval)
          last = mid - 1;
        else if (midval < search)
          first = mid + 1;
        else
          break;
      }
      legion_assert(first != last);
      const unsigned diff_low = search - (*this)[first];
      const unsigned diff_high = (*this)[last] - search;
      if (diff_low < diff_high)
        return (*this)[first];
      else
        return (*this)[last];
    }

    //--------------------------------------------------------------------------
    bool CollectiveMapping::contains(const CollectiveMapping& rhs) const
    //--------------------------------------------------------------------------
    {
      return !(rhs.unique_sorted_spaces - unique_sorted_spaces);
    }

    //--------------------------------------------------------------------------
    CollectiveMapping* CollectiveMapping::clone_with(AddressSpaceID space) const
    //--------------------------------------------------------------------------
    {
      CollectiveMapping* result = new CollectiveMapping(*this);
      result->unique_sorted_spaces.insert(space);
      result->total_spaces = result->unique_sorted_spaces.size();
      return result;
    }

    //--------------------------------------------------------------------------
    void CollectiveMapping::pack(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      legion_assert(total_spaces > 0);
      rez.serialize(total_spaces);
      rez.serialize(unique_sorted_spaces);
      rez.serialize(radix);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveMapping::pack_null(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveMapping::convert_to_offset(
        unsigned index, unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
      legion_assert(index < total_spaces);
      legion_assert(origin_index < total_spaces);
      if (index < origin_index)
      {
        // Modulus arithmetic here
        return ((index + total_spaces) - origin_index);
      }
      else
        return (index - origin_index);
    }

    //--------------------------------------------------------------------------
    unsigned CollectiveMapping::convert_to_index(
        unsigned offset, unsigned origin_index) const
    //--------------------------------------------------------------------------
    {
      legion_assert(offset < total_spaces);
      legion_assert(origin_index < total_spaces);
      unsigned result = origin_index + offset;
      if (result >= total_spaces)
        result -= total_spaces;
      return result;
    }

  }  // namespace Internal
}  // namespace Legion
