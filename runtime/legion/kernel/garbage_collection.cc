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

#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/runtime.h"
#include "legion/nodes/expression.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ImplicitReferenceTracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ImplicitReferenceTracker::~ImplicitReferenceTracker(void)
    //--------------------------------------------------------------------------
    {
      for (IndexSpaceExpression* const & expr_ptr : live_expressions)
        if (expr_ptr->remove_base_expression_reference(LIVE_EXPR_REF))
          delete expr_ptr;
    }

    /////////////////////////////////////////////////////////////
    // DistributedCollectable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(
        DistributedID id, bool do_registration, CollectiveMapping* mapping,
        State initial_state)
      : did(id), owner_space(runtime->determine_owner(did)),
        local_space(runtime->address_space), collective_mapping(mapping),
        current_state(initial_state), gc_references(0), resource_references(0),
        downgrade_owner(owner_space), notready_owner(owner_space),
        sent_global_references(0), received_global_references(0),
        total_sent_references(0), total_received_references(0),
        remaining_responses(0), registered_with_runtime(false)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping != nullptr)
      {
        legion_assert(collective_mapping->contains(owner_space));
        collective_mapping->add_reference();
      }
      if (do_registration)
        register_with_runtime();
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::~DistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(gc_references == 0);
      legion_assert(resource_references == 0);
      if ((collective_mapping != nullptr) &&
          collective_mapping->remove_reference())
        delete collective_mapping;
#ifdef LEGION_GC
      log_garbage.info(
          "GC Deletion %lld %d", LEGION_DISTRIBUTED_ID_FILTER(did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    template<bool NEED_LOCK>
    bool DistributedCollectable::is_global(void) const
    //--------------------------------------------------------------------------
    {
      if (NEED_LOCK)
      {
        AutoLock gc(gc_lock, false /*exclusive*/);
        return (current_state == VALID_REF_STATE) ||
               (current_state == GLOBAL_REF_STATE) ||
               (current_state == PENDING_LOCAL_REF_STATE) ||
               (current_state == PENDING_GLOBAL_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE) ||
               (current_state == GLOBAL_REF_STATE) ||
               (current_state == PENDING_LOCAL_REF_STATE) ||
               (current_state == PENDING_GLOBAL_REF_STATE);
    }

    template bool DistributedCollectable::is_global<true>(void) const;
    template bool DistributedCollectable::is_global<false>(void) const;

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_LOCAL_REF_STATE)
        current_state = GLOBAL_REF_STATE;
#ifdef LEGION_DEBUG_GC
      gc_references += cnt;
#else
      gc_references.fetch_add(cnt);
#endif
    }

#ifndef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      legion_assert(gc_references.load() >= cnt);
      if (gc_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      resource_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      legion_assert(resource_references.load() >= cnt);
      if (resource_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }
#endif  // not defined LEGION_DEBUG_GC

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_GC
    template<typename T>
    bool DistributedCollectable::acquire_global(
        int cnt, T source, std::map<T, int>& detailed_gc_references)
#else
    bool DistributedCollectable::acquire_global(int cnt)
#endif
    //--------------------------------------------------------------------------
    {
      AddressSpaceID current_owner;
      {
        AutoLock gc(gc_lock);
        // Check to see if we lost the race and somebody else already
        // added the references in which case we are done
        if (gc_references > 0)
        {
#ifdef LEGION_DEBUG_GC
          gc_references += cnt;
          typename std::map<T, int>::iterator finder =
              detailed_gc_references.find(source);
          if (finder == detailed_gc_references.end())
            detailed_gc_references[source] = cnt;
          else
            finder->second += cnt;
#else
          gc_references.fetch_add(cnt);
#endif
          return true;
        }
        switch (current_state)
        {
          case GLOBAL_REF_STATE:
          case VALID_REF_STATE:
          case PENDING_GLOBAL_REF_STATE:
            {
              // No downgrade in progress so we can just add the references
              // Can only be in a pending state if we're not the owner
              legion_assert(
                  (current_state != PENDING_GLOBAL_REF_STATE) ||
                  (downgrade_owner != local_space));
#ifdef LEGION_DEBUG_GC
              gc_references += cnt;
              typename std::map<T, int>::iterator finder =
                  detailed_gc_references.find(source);
              if (finder == detailed_gc_references.end())
                detailed_gc_references[source] = cnt;
              else
                finder->second += cnt;
#else
              gc_references.fetch_add(cnt);
#endif
              return true;
            }
          case PENDING_LOCAL_REF_STATE:
            {
              // Can only be in a pending state if we're not the owner
              legion_assert(downgrade_owner != local_space);
              // Not safe to increment the references since we might
              // race with the downgrade request, so we need to send
              // a message to the downgrade owner to see if we can
              break;
            }
          case LOCAL_REF_STATE:
          case DELETED_REF_STATE:
            {
              return false;
            }
          default:
            std::abort();
        }
        current_owner = downgrade_owner;
      }
      // Send the message to the downgrade owner to try to acquire the reference
      std::atomic<bool> result(false);
      const RtUserEvent ready = Runtime::create_rt_user_event();
      DistributedGlobalAcquireRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(this);
        rez.serialize(local_space);
        rez.serialize(cnt);
        rez.serialize(&result);
        rez.serialize(ready);
      }
      rez.dispatch(current_owner);
      ready.wait();
      if (result.load())
      {
#ifdef LEGION_DEBUG_GC
        AutoLock gc(gc_lock);
        typename std::map<T, int>::iterator finder =
            detailed_gc_references.find(source);
        if (finder == detailed_gc_references.end())
          detailed_gc_references[source] = cnt;
        else
          finder->second += cnt;
#endif
        return true;
      }
      else
        return false;
    }

#ifdef LEGION_DEBUG_GC
    template bool DistributedCollectable::acquire_global<ReferenceSource>(
        int, ReferenceSource, std::map<ReferenceSource, int>&);
    template bool DistributedCollectable::acquire_global<DistributedID>(
        int, DistributedID, std::map<DistributedID, int>&);
#endif

    //--------------------------------------------------------------------------
    bool DistributedCollectable::acquire_global_remote(
        AddressSpaceID& current, int count, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (is_global<false /*need lock*/>())
      {
        if (downgrade_owner == local_space)
        {
          // We succeeded
          if (source == local_space)
          {
            // If we're local we can add the references now
#ifdef LEGION_DEBUG_GC
            gc_references += count;
#else
            gc_references.fetch_add(count);
#endif
          }
          else  // Otherwise pack a reference to send back
            sent_global_references++;
          return true;
        }
        else
          current = downgrade_owner;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedGlobalAcquireRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable* remote;
      derez.deserialize(remote);
      AddressSpaceID source;
      derez.deserialize(source);
      int count;
      derez.deserialize(count);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      DistributedCollectable* dc =
          runtime->weak_find_distributed_collectable(did);
      if (dc != nullptr)
      {
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_global_remote(current_owner, count, source))
        {
          // Successfully acquired (packed) a global reference
          if (source != dc->local_space)
          {
            DistributedGlobalAcquireResponse rez;
            {
              RezCheck z2(rez);
              rez.serialize(remote);
              rez.serialize(count);
              rez.serialize(result);
              rez.serialize(ready);
            }
            rez.dispatch(source);
          }
          else
          {
            // Might have been sent back to ourself eventually
            result->store(true);
            Runtime::trigger_event(ready);
          }
        }
        else if (current_owner != dc->local_space)
        {
          // Not the owner anymore, so forward and keep chasing
          DistributedGlobalAcquireRequest rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(remote);
            rez.serialize(source);
            rez.serialize(count);
            rez.serialize(result);
            rez.serialize(ready);
          }
          rez.dispatch(current_owner);
        }
        else
          // Failed so trigger the event
          Runtime::trigger_event(ready);
        if (dc->remove_base_resource_ref(RUNTIME_REF))
          delete dc;
      }
      else
        // Failed so trigger the event
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedGlobalAcquireResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedCollectable* local;
      derez.deserialize(local);
      int count;
      derez.deserialize(count);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      // Just add the valid reference for now
      local->add_gc_reference(count);
      // Unpack the global reference added by acquire_global_remote
      local->unpack_global_ref();
      result->store(true);
      Runtime::trigger_event(ready);
    }

#ifdef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_gc_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_LOCAL_REF_STATE)
        current_state = GLOBAL_REF_STATE;
      gc_references += cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_gc_references.find(source);
      if (finder == detailed_base_gc_references.end())
        detailed_base_gc_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_gc_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_LOCAL_REF_STATE)
        current_state = GLOBAL_REF_STATE;
      gc_references += cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_gc_references.find(source);
      if (finder == detailed_nested_gc_references.end())
        detailed_nested_gc_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_gc_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      legion_assert(gc_references >= cnt);
      gc_references -= cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_gc_references.find(source);
      legion_assert(finder != detailed_base_gc_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (gc_references == 0)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_gc_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      legion_assert(gc_references >= cnt);
      gc_references -= cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_gc_references.find(source);
      legion_assert(finder != detailed_nested_gc_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (gc_references == 0)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_resource_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      resource_references += cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_resource_references.find(source);
      if (finder == detailed_base_resource_references.end())
        detailed_base_resource_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_resource_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      resource_references += cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_resource_references.find(source);
      if (finder == detailed_nested_resource_references.end())
        detailed_nested_resource_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_resource_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      legion_assert(resource_references >= cnt);
      resource_references -= cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_resource_references.find(source);
      legion_assert(finder != detailed_base_resource_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (resource_references == 0)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_resource_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(current_state != DELETED_REF_STATE);
      legion_assert(resource_references >= cnt);
      resource_references -= cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_resource_references.find(source);
      legion_assert(finder != detailed_nested_resource_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (resource_references == 0)
        return can_delete(gc);
      else
        return false;
    }
#endif  // LEGION_DEBUG_GC

    //--------------------------------------------------------------------------
    bool DistributedCollectable::has_remote_instance(
        AddressSpaceID remote_inst) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock, false /*exclusive*/);
      return remote_instances.contains(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_instances(
        AddressSpaceID remote_inst)
    //--------------------------------------------------------------------------
    {
      // Should not be recording things we already know about
      legion_assert(remote_inst != owner_space);
      legion_assert(remote_inst != local_space);
      // should not be recording things in the collective mapping
      legion_assert(
          (collective_mapping == nullptr) ||
          !collective_mapping->contains(remote_inst));
      // should only be recording on the owner or one of the
      // nodes in the collective mapping
      legion_assert(
          is_owner() || ((collective_mapping != nullptr) &&
                         collective_mapping->contains(local_space)));
      AutoLock gc(gc_lock);
      // Handle a very unusual case here were we weren't able to perform the
      // deletion because there was a packed reference, but we didn't know
      // where to send it to yet
      if (is_owner() && remote_instances.empty() &&
          (collective_mapping == nullptr) &&
          (sent_global_references != received_global_references))
      {
        legion_assert(downgrade_owner == local_space);
        legion_assert(
            (current_state == VALID_REF_STATE) ||
            (current_state == GLOBAL_REF_STATE));
        DistributedDowngradeUpdate rez;
        rez.serialize(did);
        rez.serialize(current_state);
        rez.dispatch(remote_inst);
        downgrade_owner = remote_inst;
      }
      else if (remaining_responses > 0)
      {
        // Another hairy case: if we receive a notification of a new remote
        // instance and we're in the middle of a downgrade check, we can't
        // trust the results of our downgrade attempt anymore without also
        // querying the new instance that has just been added.
        notready_owner = remote_inst;
      }
      remote_instances.add(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::filter_remote_instances(
        AddressSpaceID remote_inst)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      remote_instances.remove(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::register_with_runtime(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!registered_with_runtime);
      registered_with_runtime = true;
      runtime->register_distributed_collectable(did, this);
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_registration(
        bool has_global_reference)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_owner());
      legion_assert(registered_with_runtime);
      RtUserEvent registered_event;
      if (has_global_reference)
        pack_global_ref();
      else
        registered_event = Runtime::create_rt_user_event();
      DistributedRemoteRegistration rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(registered_event);
      }
      rez.dispatch(owner_space);
      return registered_event;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedRemoteRegistration::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable* target =
          runtime->find_distributed_collectable(did);
      target->update_remote_instances(source);
      if (done_event.exists())
        Runtime::trigger_event(done_event);
      else
        target->unpack_global_ref();
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::pack_global_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef LEGION_DEBUG
      // Sometimes we're holindg a global reference on a remote node or have
      // another packed global ref that ensures this is safe even though
      // a downgrade attempt is in progress, we handle that case in debug
      // mode by falling back to doing an acquire
      bool remove_reference = false;
      if (current_state == PENDING_LOCAL_REF_STATE)
      {
        remove_reference = true;
        legion_assert(gc_references == 0);
        legion_assert(downgrade_owner != local_space);
        gc.release();
        // We should always succeed in acquiring this or there is a runtime
        // bug somewhere that says were not handling references correctly
        if (!check_global_and_increment(RUNTIME_REF))
          std::abort();
        gc.reacquire();
      }
      else
#endif
        // Must be in a global state when packing a reference
        legion_assert(
            (current_state == VALID_REF_STATE) ||
            (current_state == GLOBAL_REF_STATE) ||
            (current_state == PENDING_GLOBAL_REF_STATE));
      sent_global_references += cnt;
#ifdef LEGION_DEBUG
      gc.release();
      // Should never have to delete this because we just packed a global ref
      if (remove_reference && remove_base_gc_ref(RUNTIME_REF))
        std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unpack_global_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      received_global_references += cnt;
      // No need to send any notifications if a downgrade is in process
      if (remaining_responses == 0)
      {
        if (downgrade_owner == local_space)
        {
          // We're the downgrade owner so check to see if we can resume
          // doing collections or we can wait for the next removal
          check_for_downgrade_restart(local_space);
        }
        else if (
            (current_state == PENDING_LOCAL_REF_STATE) || (gc_references == 0))
        {
          // Send a notification to the downgrade owner to check if it
          // needs to resume collections now that this reference has
          // been unpacked. This is not just a performance optimization.
          // It is vital to forward progress as it removes the necessity
          // to poll on downgrades while we are waiting for references
          // to be unpacked from whatever message they are in. Note that
          // you can't even check whether it is safe to downgrade this
          // node or not since we're not the downgrade owner so we have
          // to notify the downgrade owner about the unpacked references
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.dispatch(downgrade_owner);
        }
        else
        {
          // We have outstanding gc_references (state was promoted from
          // PENDING_LOCAL_REF_STATE back to GLOBAL_REF_STATE by an
          // add_gc_reference, or we never went pending). The downgrade
          // owner can't act on this notification yet anyway, so defer
          // it until our gc_references returns to zero so we send one
          // consolidated DowngradeRestart per release cycle.
          pending_downgrade_restart = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_delete(AutoLock& gc)
    //--------------------------------------------------------------------------
    {
      if (pending_downgrade_restart)
      {
        // Clear unconditionally so the flag doesn't outlive
        // a release cycle, even when the conditions to actually send aren't
        // met (e.g., we became the downgrade owner via DowngradeUpdate).
        pending_downgrade_restart = false;
        if ((downgrade_owner != local_space) && (remaining_responses == 0))
        {
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.dispatch(downgrade_owner);
        }
      }
      switch (current_state)
      {
        case VALID_REF_STATE:
        case GLOBAL_REF_STATE:
        case PENDING_LOCAL_REF_STATE:
        case PENDING_GLOBAL_REF_STATE:
          {
            if (!can_downgrade())
              return false;
            // If we're not the downgrade owner then nothing for us to do
            if (downgrade_owner != local_space)
              return false;
            // We're the downgrade owner, so start the process to check to
            // see if all the nodes are ready to perform the deletion
            if (!is_owner() || !remote_instances.empty() ||
                ((collective_mapping != nullptr) &&
                 (collective_mapping->size() > 1)) ||
                (sent_global_references != received_global_references))
            {
              // If we're already checking for a downgrade but are awaiting
              // responses, then there is nothing to do
              if (remaining_responses > 0)
                return false;
              // Send messages to see if we can perform the deletion
              check_for_downgrade(downgrade_owner);
              return false;
            }
            else
            {
              // No messages to send so we can downgrade the state now
              return perform_downgrade(gc);
            }
          }
        case LOCAL_REF_STATE:
          {
            if (resource_references == 0)
            {
              current_state = DELETED_REF_STATE;
              return true;
            }
            break;
          }
        default:
          std::abort();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_downgrade(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (current_state == GLOBAL_REF_STATE) ||
          (current_state == PENDING_LOCAL_REF_STATE));
      return (gc_references == 0);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_downgrade_notifications(State downgrade)
    //--------------------------------------------------------------------------
    {
      // Ready to downgrade, send the messages
      if (is_owner() || ((collective_mapping != nullptr) &&
                         collective_mapping->contains(local_space)))
      {
        if (collective_mapping != nullptr)
        {
          std::vector<AddressSpaceID> children;
          if (collective_mapping->contains(downgrade_owner))
            collective_mapping->get_children(
                downgrade_owner, local_space, children);
          else
            collective_mapping->get_children(
                owner_space, local_space, children);
          if (!children.empty())
          {
            DistributedDowngradeSuccess rez;
            rez.serialize(did);
            rez.serialize(downgrade);
            for (const AddressSpaceID& child_id : children)
              rez.dispatch(child_id);
          }
        }
        if (!remote_instances.empty())
        {
          DistributedDowngradeSuccess rez;
          rez.serialize(did);
          rez.serialize(downgrade);
          struct {
            void apply(AddressSpaceID space)
            {
              if (space != owner)
                rez->dispatch(space);
            }
            DistributedDowngradeSuccess* rez;
            AddressSpaceID owner;
          } downgrade_functor;
          downgrade_functor.rez = &rez;
          downgrade_functor.owner = downgrade_owner;
          remote_instances.map(downgrade_functor);
        }
      }
      else if (downgrade_owner == local_space)
      {
        // If we're the owner then we have to send it to the owner_space
        // to get all the remote instances
        DistributedDowngradeSuccess rez;
        rez.serialize(did);
        rez.serialize(downgrade);
        rez.dispatch(owner_space);
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::perform_downgrade(AutoLock& gc)
    //--------------------------------------------------------------------------
    {
      legion_assert(gc_references == 0);
      // Should be in the GLOBAL_REF_STATE on the owner and
      // PENDING_LOCAL_REF_STATE if we're not the downgrade owner
      legion_assert(
          ((current_state == GLOBAL_REF_STATE) &&
           (downgrade_owner == local_space)) ||
          ((current_state == PENDING_LOCAL_REF_STATE) &&
           (downgrade_owner != local_space)));
      // Downgrade the state first so that we don't duplicate the callback
      current_state = LOCAL_REF_STATE;
      // Add a resource reference here to prevent collection while we
      // release the lock to perform the callback
#ifdef LEGION_DEBUG_GC
      resource_references++;
#else
      resource_references.fetch_add(1);
#endif
      gc.release();
      // Can do this without holding the lock as the remote_instances data
      // structure should no longer be changing
      send_downgrade_notifications(GLOBAL_REF_STATE);
      notify_local();
      // Unregister this with the runtime
      if (registered_with_runtime)
        runtime->unregister_distributed_collectable(did);
      gc.reacquire();
      legion_assert(resource_references > 0);
      // Remove the guard resource reference that we added before
#ifdef LEGION_DEBUG_GC
      if (--resource_references == 0)
#else
      if (resource_references.fetch_sub(1) == 1)
#endif
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::check_for_downgrade(AddressSpaceID owner)
    //--------------------------------------------------------------------------
    {
      legion_assert(remaining_responses == 0);
      // Update the downgrade owner
      downgrade_owner = owner;
      if (can_downgrade())
      {
        // We're ready to be downgraded
        // Send messages and count how many responses we expect to see
        if (is_owner() || ((collective_mapping != nullptr) &&
                           collective_mapping->contains(local_space)))
        {
          if (collective_mapping != nullptr)
          {
            std::vector<AddressSpaceID> children;
            if (collective_mapping->contains(owner))
              collective_mapping->get_children(owner, local_space, children);
            else
              collective_mapping->get_children(
                  owner_space, local_space, children);
            if (!children.empty())
            {
              DistributedDowngradeRequest rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                // If we're in a pending state send the downgrade
                // for the non-pending version of this state
                if ((current_state == PENDING_LOCAL_REF_STATE) ||
                    (current_state == PENDING_GLOBAL_REF_STATE))
                  rez.serialize(current_state + 1);
                else
                  rez.serialize(current_state);
                rez.serialize(owner);
              }
              for (const AddressSpaceID& child_id : children)
                rez.dispatch(child_id);
              remaining_responses += children.size();
            }
          }
          if (!remote_instances.empty())
          {
            DistributedDowngradeRequest rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              // If we're in a pending state send the downgrade
              // for the non-pending version of this state
              if ((current_state == PENDING_LOCAL_REF_STATE) ||
                  (current_state == PENDING_GLOBAL_REF_STATE))
                rez.serialize(current_state + 1);
              else
                rez.serialize(current_state);
              rez.serialize(owner);
            }
            struct {
              void apply(AddressSpaceID space)
              {
                if (space != owner)
                  rez->dispatch(space);
                else
                  skipped++;
              }
              DistributedDowngradeRequest* rez;
              AddressSpaceID owner;
              unsigned skipped;
            } downgrade_functor;
            downgrade_functor.rez = &rez;
            downgrade_functor.owner = downgrade_owner;
            downgrade_functor.skipped = 0;
            remote_instances.map(downgrade_functor);
            remaining_responses +=
                (remote_instances.size() - downgrade_functor.skipped);
          }
        }
        else if (owner == local_space)
        {
          // Should be in a non-pending state if we're the owner
          legion_assert(
              (current_state == GLOBAL_REF_STATE) ||
              (current_state == VALID_REF_STATE));
          // If we're the owner then we have to send it to the owner_space
          // to get all the remote instances
          DistributedDowngradeRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(current_state);
            rez.serialize(owner);
          }
          rez.dispatch(owner_space);
          remaining_responses++;
        }
        // Initialize the downgrade state
        notready_owner = owner;
        total_sent_references = 0;
        total_received_references = 0;
        if (remaining_responses == 0)
        {
          // Send the response now
          if (owner != local_space)
          {
            // Mark that we're in the pending downgrade state
            accumulate_local_references();
            const AddressSpaceID target = get_downgrade_target(owner);
            DistributedDowngradeResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(owner);  // owner is special bottom value
              rez.serialize(total_sent_references);
              rez.serialize(total_received_references);
            }
            rez.dispatch(target);
            record_pending_downgrade();
          }
          else
          {
            // We only get here if we're the owner and we don't know
            // about any remote instances yet. The only way that
            // should happen is if we have some sent global references
            // There's nothing to do yet since we know we can't be
            // deleted yet
            legion_assert(sent_global_references > 0);
            legion_assert(sent_global_references != received_global_references);
          }
        }
      }
      else if (local_space != owner)
      {
        const AddressSpaceID target = get_downgrade_target(owner);
        DistributedDowngradeResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize<uint64_t>(0);  // sent global references
          rez.serialize<uint64_t>(0);  // received global references
        }
        rez.dispatch(target);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::check_for_downgrade_restart(
        AddressSpaceID new_owner)
    //--------------------------------------------------------------------------
    {
      // If we're no longer the downgrade owner there is nothing to do
      if (downgrade_owner != local_space)
        return;
      // If there is a downgrade in progress there's nothing for us to do
      if (remaining_responses > 0)
        return;
      // If we can't downgrade then we'll restart the downgrade proces
      if ((current_state == LOCAL_REF_STATE) || !can_downgrade())
        return;
      // A deferred DowngradeRestart from a remote can arrive after the
      // previous cycle finished with notready_owner already pointing
      // elsewhere. That cycle is already over; drop this stale nudge
      // rather than asserting.
      if (notready_owner != local_space)
        return;
      // The DowngradeRestart can race with the sender's
      // DistributedRemoteRegistration and arrive first. If we don't
      // know about this remote yet (it's not in remote_instances or
      // the collective_mapping), drop the restart; the upcoming
      // registration will trigger the proper ownership transfer in
      // update_remote_instances.
      if ((new_owner != local_space) && !remote_instances.contains(new_owner) &&
          ((collective_mapping == nullptr) ||
           !collective_mapping->contains(new_owner)))
        return;
      legion_assert(
          (current_state == VALID_REF_STATE) ||
          (current_state == GLOBAL_REF_STATE));
      // Restart the downgrade process
      if (new_owner != local_space)
      {
        downgrade_owner = new_owner;
        DistributedDowngradeUpdate rez;
        rez.serialize(did);
        rez.serialize(current_state);
        rez.dispatch(new_owner);
      }
      else
        check_for_downgrade(new_owner);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::accumulate_local_references(void)
    //--------------------------------------------------------------------------
    {
      total_sent_references += sent_global_references;
      total_received_references += received_global_references;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::record_pending_downgrade(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(downgrade_owner != local_space);
      legion_assert(
          (current_state == GLOBAL_REF_STATE) ||
          (current_state == PENDING_LOCAL_REF_STATE));
      current_state = PENDING_LOCAL_REF_STATE;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable::State to_check;
      derez.deserialize(to_check);
      AddressSpaceID downgrade_owner;
      derez.deserialize(downgrade_owner);

      // It's possible for this to race with the creation of this
      // distributed collectable so wait until it is ready
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      dc->process_downgrade_request(downgrade_owner, to_check);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_request(
        AddressSpaceID owner, State to_check)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner != local_space);  // we should be remote here
      legion_assert(
          (to_check == GLOBAL_REF_STATE) || (to_check == VALID_REF_STATE));
      AutoLock gc(gc_lock);
      // If the owner is asking us to downgrade a state that is less than
      // our current state then that is because the downgrade from our
      // current state has already been done on the owner and we should
      // perform our local down grade to reflect that first
      while (to_check < current_state) perform_downgrade(gc);
      legion_assert(LOCAL_REF_STATE < current_state);
      check_for_downgrade(owner);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID DistributedCollectable::get_downgrade_target(
        AddressSpaceID owner) const
    //--------------------------------------------------------------------------
    {
      legion_assert(owner != local_space);
      if (collective_mapping == nullptr)
      {
        if (local_space == owner_space)
          return owner;
        else
          return owner_space;
      }
      if (!collective_mapping->contains(local_space))
      {
        legion_assert(!is_owner());
        return collective_mapping->find_nearest(local_space);
      }
      if (!collective_mapping->contains(owner))
      {
        if (is_owner())
          return owner;
        else
          return collective_mapping->get_parent(owner_space, local_space);
      }
      return collective_mapping->get_parent(owner, local_space);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::process_downgrade_response(
        AddressSpaceID notready, uint64_t total_sent, uint64_t total_received)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(remaining_responses > 0);
      if (notready != downgrade_owner)
        notready_owner = notready;
      else if (notready_owner == downgrade_owner)
      {
        // Everything still ready for downgrade
        total_sent_references += total_sent;
        total_received_references += total_received;
      }
      if (--remaining_responses == 0)
      {
        // Accumulate our local sent and received references
        accumulate_local_references();
        if (downgrade_owner == local_space)
        {
          legion_assert(
              (current_state == VALID_REF_STATE) ||
              (current_state == GLOBAL_REF_STATE));
          // See if it safe to downgrade
          // Make sure to check ourselves again to handle any
          // check_*_and_increment methods
          if (can_downgrade() && (notready_owner == downgrade_owner) &&
              (total_sent_references == total_received_references))
          {
            // Then perform our local downgrade
            return perform_downgrade(gc);
          }
          else
          {
            // Not ready to downgrade
            if (notready_owner != downgrade_owner)
            {
              // Update the new owner responsible for checking for downgrades
              downgrade_owner = notready_owner;
              DistributedDowngradeUpdate rez;
              rez.serialize(did);
              rez.serialize(current_state);
              rez.dispatch(notready_owner);
            }
            // else: we used to do this, but the polling aspect of continuing
            // to check for downgrades can cause priority inversions in the
            // network traffic (despite setting low priorities because the
            // networking hardware ignores our priorities). Therefore we stopped
            // doing this and now we instead send notifications whenever we
            // do an unpack that might need to restart this process. The first
            // one to get here will restart the downgrade process.
            // See the calls to check_for_downgrade_restart to see where
            // progress comes from now
            //{
            // This is a strange case: all the nodes are ready to downgrade
            // but there are still packed references in flight so we need
            // to keep trying to perform the downgrade until we find one
            // of these nodes and find the reference
            // check_for_downgrade(downgrade_owner);
            //}
          }
        }
        else
        {
          const AddressSpaceID target = get_downgrade_target(downgrade_owner);
          // We had to release the lock to send the requests to our upstream
          // nodes so we need to check again to see if it is still safe to
          // perform the downgrade on this node or not atomically with
          // accumulating our sent and received references
          DistributedDowngradeResponse rez;
          if (can_downgrade())
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(notready_owner);
            rez.serialize(total_sent_references);
            rez.serialize(total_received_references);
            // Record that we're in the pending downgrade state
            record_pending_downgrade();
          }
          else
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
            rez.serialize<uint64_t>(0);  // sent global references
            rez.serialize<uint64_t>(0);  // received global references
          }
          rez.dispatch(target);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID notready;
      derez.deserialize(notready);
      uint64_t total_sent, total_received;
      derez.deserialize(total_sent);
      derez.deserialize(total_received);

      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      if (dc->process_downgrade_response(notready, total_sent, total_received))
        delete dc;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_success(State to_downgrade)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      // Check to see if this state has already been downgraded already
      // because a check_for_downgrade got here first
      if ((to_downgrade == current_state) ||
          ((current_state + 1) == to_downgrade))
        perform_downgrade(gc);
      else
        legion_assert(current_state < to_downgrade);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeSuccess::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable::State to_downgrade;
      derez.deserialize(to_downgrade);

      // These can race with checks for downgrades from other states and
      // therefore it's possible for these to arrive even after the object
      // itself has been deleted so we need a weak find here
      DistributedCollectable* dc =
          runtime->weak_find_distributed_collectable(did);
      if (dc != nullptr)
      {
        dc->process_downgrade_success(to_downgrade);
        if (dc->remove_base_resource_ref(RUNTIME_REF))
          delete dc;
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_update(
        AutoLock& gc, State to_check)
    //--------------------------------------------------------------------------
    {
      legion_assert(downgrade_owner != local_space);
      legion_assert(to_check == GLOBAL_REF_STATE);
      // It's possible we get this notification before the update saying
      // that the downgrade from the previous state has been successful
      // so make sure to update accordingly
      while (to_check < current_state) perform_downgrade(gc);
      if (current_state < to_check)
        current_state = to_check;
      downgrade_owner = local_space;
      if (gc_references == 0)
        check_for_downgrade(downgrade_owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeUpdate::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable::State state;
      derez.deserialize(state);

      // It's possible for this to race with the creation and registration
      // of this distributed collectable so wait for it to be ready
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      AutoLock gc(dc->gc_lock);
      dc->process_downgrade_update(gc, state);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeRestart::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      // It's possible for these messages to race with actual downgrades and
      // destruction of the collectable object so we have to check to see if
      // it is still here, if it's not then it's already been cleaned up and
      // there is nothing more for us to do
      DistributedCollectable* dc =
          runtime->weak_find_distributed_collectable(did);
      if (dc != nullptr)
      {
        {
          AutoLock gc(dc->gc_lock);
          dc->check_for_downgrade_restart(source);
        }
        if (dc->remove_base_resource_ref(RUNTIME_REF))
          delete dc;
      }
    }

    /////////////////////////////////////////////////////////////
    // ValidDistributedCollectable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::ValidDistributedCollectable(
        DistributedID id, bool do_registration, CollectiveMapping* map,
        bool start_in_valid_state)
      : DistributedCollectable(
            id, do_registration, map,
            start_in_valid_state ? VALID_REF_STATE : GLOBAL_REF_STATE),
        valid_references(0), sent_valid_references(0),
        received_valid_references(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::~ValidDistributedCollectable(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<bool NEED_LOCK>
    bool ValidDistributedCollectable::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (NEED_LOCK)
      {
        AutoLock gc(gc_lock, false /*exclusive*/);
        return (current_state == VALID_REF_STATE) ||
               (current_state == PENDING_GLOBAL_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE) ||
               (current_state == PENDING_GLOBAL_REF_STATE);
    }

    template bool ValidDistributedCollectable::is_valid<true>(void) const;
    template bool ValidDistributedCollectable::is_valid<false>(void) const;

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_GLOBAL_REF_STATE)
        current_state = VALID_REF_STATE;
#ifdef LEGION_DEBUG_GC
      valid_references += cnt;
#else
      valid_references.fetch_add(cnt);
#endif
    }

#ifndef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      legion_assert(valid_references.load() >= cnt);
      if (valid_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }
#else  // ifndef LEGION_DEBUG_GC
    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_GLOBAL_REF_STATE)
        current_state = VALID_REF_STATE;
      valid_references += cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      // Promote the current state back up if we had a pending downgrade
      if (current_state == PENDING_GLOBAL_REF_STATE)
        current_state = VALID_REF_STATE;
      valid_references += cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::remove_base_valid_ref_internal(
        ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<ReferenceSource, int>::iterator finder =
          detailed_base_valid_references.find(source);
      legion_assert(finder != detailed_base_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (valid_references == 0)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::remove_nested_valid_ref_internal(
        DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      legion_assert(valid_references >= cnt);
      valid_references -= cnt;
      std::map<DistributedID, int>::iterator finder =
          detailed_nested_valid_references.find(source);
      legion_assert(finder != detailed_nested_valid_references.end());
      legion_assert(finder->second >= cnt);
      finder->second -= cnt;
      if (valid_references == 0)
        return can_delete(gc);
      else
        return false;
    }
#endif

    //--------------------------------------------------------------------------
#ifdef LEGION_DEBUG_GC
    template<typename T>
    bool ValidDistributedCollectable::acquire_valid(
        int cnt, T source, std::map<T, int>& detailed_valid_references)
#else
    bool ValidDistributedCollectable::acquire_valid(int cnt)
#endif
    //--------------------------------------------------------------------------
    {
      AddressSpaceID current_owner;
      {
        AutoLock gc(gc_lock);
        // Check to see if we lost the race and somebody else already
        // added the references in which case we are done
        if (valid_references > 0)
        {
#ifdef LEGION_DEBUG_GC
          valid_references += cnt;
          typename std::map<T, int>::iterator finder =
              detailed_valid_references.find(source);
          if (finder == detailed_valid_references.end())
            detailed_valid_references[source] = cnt;
          else
            finder->second += cnt;
#else
          valid_references.fetch_add(cnt);
#endif
          return true;
        }
        switch (current_state)
        {
          case VALID_REF_STATE:
            {
              // No downgrade in progress so we can just add the references
#ifdef LEGION_DEBUG_GC
              valid_references += cnt;
              typename std::map<T, int>::iterator finder =
                  detailed_valid_references.find(source);
              if (finder == detailed_valid_references.end())
                detailed_valid_references[source] = cnt;
              else
                finder->second += cnt;
#else
              valid_references.fetch_add(cnt);
#endif
              return true;
            }
          case PENDING_GLOBAL_REF_STATE:
            {
              // Can only be in a pending state if we're not the owner
              legion_assert(downgrade_owner != local_space);
              // Not safe to increment the references since we might
              // race with the downgrade request, so we need to send
              // a message to the downgrade owner to see if we can
              break;
            }
          case GLOBAL_REF_STATE:
          case PENDING_LOCAL_REF_STATE:
          case LOCAL_REF_STATE:
          case DELETED_REF_STATE:
            {
              return false;
            }
          default:
            std::abort();
        }
        current_owner = downgrade_owner;
      }
      // Send the message to the downgrade owner to try to acquire the reference
      std::atomic<bool> result(false);
      const RtUserEvent ready = Runtime::create_rt_user_event();
      DistributedValidAcquireRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(this);
        rez.serialize(local_space);
        rez.serialize(cnt);
        rez.serialize(&result);
        rez.serialize(ready);
      }
      rez.dispatch(current_owner);
      ready.wait();
      if (result.load())
      {
#ifdef LEGION_DEBUG_GC
        AutoLock gc(gc_lock);
        typename std::map<T, int>::iterator finder =
            detailed_valid_references.find(source);
        if (finder == detailed_valid_references.end())
          detailed_valid_references[source] = cnt;
        else
          finder->second += cnt;
#endif
        return true;
      }
      else
        return false;
    }

#ifdef LEGION_DEBUG_GC
    template bool ValidDistributedCollectable::acquire_valid<ReferenceSource>(
        int, ReferenceSource, std::map<ReferenceSource, int>&);
    template bool ValidDistributedCollectable::acquire_valid<DistributedID>(
        int, DistributedID, std::map<DistributedID, int>&);
#endif

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::acquire_valid_remote(
        AddressSpaceID& current, int count, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (is_valid<false /*need lock*/>())
      {
        if (downgrade_owner == local_space)
        {
          // We succeeded
          if (source == local_space)
          {
            // If we're local we can add the references now
#ifdef LEGION_DEBUG_GC
            valid_references += count;
#else
            valid_references.fetch_add(count);
#endif
          }
          else  // Otherwise pack a reference to send back
            sent_valid_references++;
          return true;
        }
        else
          current = downgrade_owner;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedValidAcquireRequest::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ValidDistributedCollectable* remote;
      derez.deserialize(remote);
      AddressSpaceID source;
      derez.deserialize(source);
      int count;
      derez.deserialize(count);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      ValidDistributedCollectable* dc =
          static_cast<ValidDistributedCollectable*>(
              runtime->weak_find_distributed_collectable(did));
      if (dc != nullptr)
      {
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_valid_remote(current_owner, count, source))
        {
          if (source != dc->local_space)
          {
            // Successfully acquired (packed) a valid reference
            DistributedValidAcquireResponse rez;
            {
              RezCheck z2(rez);
              rez.serialize(remote);
              rez.serialize(count);
              rez.serialize(result);
              rez.serialize(ready);
            }
            rez.dispatch(source);
          }
          else
          {
            // Might have been sent back to ourself eventually
            result->store(true);
            Runtime::trigger_event(ready);
          }
        }
        else if (current_owner != dc->local_space)
        {
          // Not the owner anymore, so forward and keep chasing
          DistributedValidAcquireRequest rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(remote);
            rez.serialize(source);
            rez.serialize(count);
            rez.serialize(result);
            rez.serialize(ready);
          }
          rez.dispatch(current_owner);
        }
        else
          // Failed so trigger the event
          Runtime::trigger_event(ready);
        if (dc->remove_base_resource_ref(RUNTIME_REF))
          delete dc;
      }
      else
        // Failed so trigger the event
        Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedValidAcquireResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ValidDistributedCollectable* local;
      derez.deserialize(local);
      int count;
      derez.deserialize(count);
      std::atomic<bool>* result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      // Just add the valid reference for now
      local->add_valid_reference(count);
      // Unpack the valid reference packed by acquire_valid_remote
      local->unpack_valid_ref();
      result->store(true);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::pack_valid_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef LEGION_DEBUG
      // Sometimes we're holindg a valid reference on a remote node or have
      // another packed valid ref that ensures this is safe even though
      // a downgrade attempt is in progress, we handle that case in debug
      // mode by falling back to doing an acquire
      bool remove_reference = false;
      if (current_state == PENDING_GLOBAL_REF_STATE)
      {
        remove_reference = true;
        legion_assert(gc_references == 0);
        legion_assert(downgrade_owner != local_space);
        gc.release();
        // We should always succeed in acquiring this or there is a runtime
        // bug somewhere that says were not handling references correctly
        if (!check_valid_and_increment(RUNTIME_REF))
          std::abort();
        gc.reacquire();
      }
      else
#endif
        // Must be valid when packing a reference
        legion_assert(current_state == VALID_REF_STATE);
      sent_valid_references += cnt;
#ifdef LEGION_DEBUG
      gc.release();
      // Should never have to delete this because we just packed a valid ref
      if (remove_reference && remove_base_valid_ref(RUNTIME_REF))
        std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::unpack_valid_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      received_valid_references += cnt;
      // No need to send any notifications if a downgrade is in process
      if (remaining_responses == 0)
      {
        if (downgrade_owner == local_space)
        {
          // We're the downgrade owner so check to see if we can resume
          // doing collections or we can wait for the next removal
          check_for_downgrade_restart(local_space);
        }
        else if (
            (current_state == PENDING_GLOBAL_REF_STATE) ||
            (valid_references == 0))
        {
          // Send a notification to the downgrade owner to check if it
          // needs to resume collections now that this reference has
          // been unpacked. This is not just a performance optimization.
          // It is vital to forward progress as it removes the necessity
          // to poll on downgrades while we are waiting for references
          // to be unpacked from whatever message they are in. Note that
          // you can't even check whether it is safe to downgrade this
          // node or not since we're not the downgrade owner so we have
          // to notify the downgrade owner about the unpacked references
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.dispatch(downgrade_owner);
        }
        else
        {
          // We have outstanding valid_references (state was promoted from
          // PENDING_GLOBAL_REF_STATE back to VALID_REF_STATE by an
          // add_valid_reference, or we never went pending). Defer the
          // notification until our valid_references returns to zero.
          pending_downgrade_restart = true;
        }
      }
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::can_downgrade(void) const
    //--------------------------------------------------------------------------
    {
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
        return (valid_references == 0);
      else
        return DistributedCollectable::can_downgrade();
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::perform_downgrade(AutoLock& gc)
    //--------------------------------------------------------------------------
    {
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
      {
        legion_assert(valid_references == 0);
        // Should be in the GLOBAL_REF_STATE on the owner and
        // PENDING_LOCAL_REF_STATE if we're not the downgrade owner
        legion_assert(
            ((current_state == VALID_REF_STATE) &&
             (downgrade_owner == local_space)) ||
            ((current_state == PENDING_GLOBAL_REF_STATE) &&
             (downgrade_owner != local_space)));
        // Send messages while holding the lock because the remote_instances
        // data structure might still be changing
        send_downgrade_notifications(VALID_REF_STATE);
        // Downgrade the state first so that we don't duplicate the callback
        current_state = GLOBAL_REF_STATE;
        // Add a gc reference here prevent downgrades from the global ref
        // state until we are done performing the callback
#ifdef LEGION_DEBUG_GC
        gc_references++;
#else
        gc_references.fetch_add(1);
#endif
        gc.release();
        notify_invalid();
        gc.reacquire();
        legion_assert(gc_references > 0);
        // Remove the guard reference that we added before
#ifdef LEGION_DEBUG_GC
        if (--gc_references == 0)
#else
        if (gc_references.fetch_sub(1) == 1)
#endif
          return can_delete(gc);
        else
          return false;
      }
      else
        return DistributedCollectable::perform_downgrade(gc);
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::process_downgrade_update(
        AutoLock& gc, State to_check)
    //--------------------------------------------------------------------------
    {
      legion_assert(downgrade_owner != local_space);
      legion_assert(
          (to_check == VALID_REF_STATE) || (to_check == GLOBAL_REF_STATE));
      // It's possible we get this notification before the update saying
      // that the downgrade from the previous state has been successful
      // so make sure to update accordingly
      while (to_check < current_state) perform_downgrade(gc);
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
      {
        downgrade_owner = local_space;
        current_state = VALID_REF_STATE;
        if (valid_references == 0)
          check_for_downgrade(downgrade_owner);
      }
      else
        DistributedCollectable::process_downgrade_update(gc, to_check);
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::accumulate_local_references(void)
    //--------------------------------------------------------------------------
    {
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
      {
        total_sent_references += sent_valid_references;
        total_received_references += received_valid_references;
      }
      else
        DistributedCollectable::accumulate_local_references();
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::record_pending_downgrade(void)
    //--------------------------------------------------------------------------
    {
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
      {
        legion_assert(downgrade_owner != local_space);
        current_state = PENDING_GLOBAL_REF_STATE;
      }
      else
        DistributedCollectable::record_pending_downgrade();
    }

  }  // namespace Internal
}  // namespace Legion
