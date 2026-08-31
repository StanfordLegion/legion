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
    ImplicitReferenceTracker::ImplicitReferenceTracker(bool push)
    //--------------------------------------------------------------------------
    {
      if (push)
        push_reference_tracker();
    }

    //--------------------------------------------------------------------------
    ImplicitReferenceTracker::~ImplicitReferenceTracker(void)
    //--------------------------------------------------------------------------
    {
      if (implicit_reference_tracker == this)
        pop_reference_tracker();
      legion_assert(live_expressions.empty());
      legion_assert(invalid_operations.empty());
    }

    //--------------------------------------------------------------------------
    void ImplicitReferenceTracker::push_reference_tracker(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_reference_tracker != this);
      previous = implicit_reference_tracker;
      implicit_reference_tracker = this;
    }

    //--------------------------------------------------------------------------
    void ImplicitReferenceTracker::pop_reference_tracker(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_reference_tracker == this);
      drain_tracked_references();
      implicit_reference_tracker = previous;
    }

    //--------------------------------------------------------------------------
    void ImplicitReferenceTracker::handoff_reference_tracker(
        ImplicitReferenceTracker& next)
    //--------------------------------------------------------------------------
    {
      legion_assert(implicit_reference_tracker == this);
      legion_assert(&next != this);
      legion_assert(next.live_expressions.empty());
      legion_assert(next.invalid_operations.empty());
      legion_assert(next.previous == nullptr);
      legion_assert(!next.pending_invalidation);
      drain_tracked_references();
      next.previous = previous;
      previous = nullptr;
      implicit_reference_tracker = &next;
    }

    //--------------------------------------------------------------------------
    void ImplicitReferenceTracker::drain_tracked_references(void)
    //--------------------------------------------------------------------------
    {
      for (IndexSpaceExpression* const & expr_ptr : live_expressions)
        if (expr_ptr->remove_base_expression_reference(LIVE_EXPR_REF))
          delete expr_ptr;
      live_expressions.clear();
      invalidate_operations();
    }

    //--------------------------------------------------------------------------
    void ImplicitReferenceTracker::invalidate_operations(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if something higher up the stack is already doing it
      // We don't want to recurse and cause a stack overflow
      if (pending_invalidation)
        return;
      struct MarkPending {
        MarkPending(bool& pending) : pending_invalidation(pending)
        {
          pending_invalidation = true;
        }
        ~MarkPending(void) { pending_invalidation = false; }
      private:
        bool& pending_invalidation;
      } pending(pending_invalidation);
      // Iterate this until converged
      // Note that because making an invalid operation local by removing
      // this reference then we can also cause other operations to become
      // invalid and get added to this list hence this assertion ensuring
      // that we're still the implicit_reference_tracker
      while (!invalid_operations.empty())
      {
        IndexSpaceOperation* next = invalid_operations.back();
        invalid_operations.pop_back();
        if (next->remove_base_gc_ref(REGION_TREE_REF))
          delete next;
      }
    }

    //--------------------------------------------------------------------------
    size_t ImplicitReferenceTracker::count_invalid_operations(void) const
    //--------------------------------------------------------------------------
    {
      return invalid_operations.size();
    }

    //--------------------------------------------------------------------------
    IndexSpaceOperation* ImplicitReferenceTracker::get_invalid_operation(
        unsigned index) const
    //--------------------------------------------------------------------------
    {
      legion_assert(index < invalid_operations.size());
      return invalid_operations[index];
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
        downgrade_owner(owner_space), round_owner(owner_space),
        round_parent(owner_space), notready_owner(owner_space),
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
        AddressSpaceID& current, int count, AddressSpaceID source,
        LamportClock& lamport_clock)
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
          {
            // Under stamp-counting every counted pack is an event at the
            // stamped level, so the clock bump applies unconditionally
            if (bump_downgrade_lamport_clock)
            {
              downgrade_lamport_clock++;
              bump_downgrade_lamport_clock = false;
            }
            // Encode the valid-level stamp in the low bit of the clock
            // token so it travels with every packed global reference
            // (see pack_global_ref for the encoding rationale)
            lamport_clock = downgrade_lamport_clock << 1;
            if (record_valid_stamp(1 /*count*/))
              lamport_clock |= 1;
            sent_global_references++;
          }
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
        LamportClock lamport_clock = 0;
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_global_remote(
                current_owner, count, source, lamport_clock))
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
              rez.serialize(lamport_clock);
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
      local->unpack_global_ref(derez);
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
    void DistributedCollectable::finalize_remote_iterators(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if ((remote_iterators == 0) && remote_iteration_waiter.exists())
      {
        Runtime::trigger_event(remote_iteration_waiter);
        remote_iteration_waiter = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_instances(
        AddressSpaceID remote_inst, bool registration,
        LamportClock* registration_clock)
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
      // Wait for any remote iterators to finish before updating the remote
      // instances. This can be delayed -- or, under a steady stream of
      // iterators, starved -- for the duration of the in-flight iterations.
      // Acceptable since registration is far rarer than these broadcasts, but
      // worth revisiting if it shows up in profiling.
      while (remote_iterators > 0)
      {
        if (!remote_iteration_waiter.exists())
          remote_iteration_waiter = Runtime::create_rt_user_event();
        const RtEvent wait_on = remote_iteration_waiter;
        gc.release();
        wait_on.wait();
        gc.reacquire();
      }
      // Handle a very unusual case here were we weren't able to perform the
      // deletion because there was a packed reference, but we didn't know
      // where to send it to yet
      if (is_owner() && remote_instances.empty() &&
          (collective_mapping == nullptr) && has_packed_references())
      {
        legion_assert(downgrade_owner == local_space);
        legion_assert(
            (current_state == VALID_REF_STATE) ||
            (current_state == GLOBAL_REF_STATE));
        downgrade_owner = remote_inst;
        downgrade_owner_version++;
        DistributedDowngradeUpdate rez;
        rez.serialize(did);
        rez.serialize(current_state);
        rez.serialize(downgrade_owner_version);
        rez.serialize(downgrade_lamport_clock);
        rez.dispatch(remote_inst);
      }
      else if (remaining_responses > 0)
      {
        // Another hairy case: if we receive a notification of a new remote
        // instance and we're in the middle of a downgrade check, we can't
        // trust the results of our downgrade attempt anymore without also
        // querying the new instance that has just been added.
        notready_owner = remote_inst;
      }
      else if (registration)
      {
        // Registration nudge: a downgrade round that already completed
        // (or an idle downgrade owner) has no other way to learn that a
        // new instance now exists, so re-run the check. Without this a
        // round at the next level can commit over an instance list that
        // is missing the registrant (see the registration-gate finding
        // in the TLA+ model). Note the nudge is only needed for actual
        // registrations: instances we are creating ourselves send back
        // their own notification when they unpack their first reference.
        //
        // F16: with our aggregation CLOSED, the mid-round poison above
        // cannot protect a round rooted elsewhere in the collective tree
        // whose responses do not pass through us, and the nudge restart
        // below races that round's decision on the wire. Bump our clock
        // past every round we have ready-voted in (accumulate folds the
        // round's clock into ours at each ready vote): the registration
        // response carries the bump to the registrant, whose future
        // packed references propagate it, so any voter receiving one
        // fails that round's causality check and votes not-ready. A
        // round we voted NOT-ready in is beyond the bump's reach but is
        // already doomed by our vote. See TLA+ finding F16.
        downgrade_lamport_clock++;
        remote_instances.add(remote_inst);
        if (downgrade_owner == local_space)
          check_for_downgrade_restart(
              gc, local_space, downgrade_owner_version,
              downgrade_lamport_clock);
        else
        {
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.serialize(downgrade_owner);  // candidate: the owner itself
          rez.serialize(downgrade_owner_version);
          rez.serialize(downgrade_lamport_clock);
          rez.dispatch(downgrade_owner);
        }
        if (registration_clock != nullptr)
          *registration_clock = downgrade_lamport_clock;
        return;
      }
      remote_instances.add(remote_inst);
      if (registration_clock != nullptr)
        *registration_clock = downgrade_lamport_clock;
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
    void DistributedCollectable::send_remote_registration(void)
    //--------------------------------------------------------------------------
    {
      // This function must be invoked by the caller before the distributed
      // collectable is discoverable by anybody else so we don't need to
      // take a lock to handle races with packed valid references
      legion_assert(!is_owner());
      legion_assert(registered_with_runtime);
      legion_assert(remote_registered);
      legion_assert(sent_global_references == 0);
      remote_registered = false;
      DistributedRemoteRegistration rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
      }
      rez.dispatch(owner_space);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedRemoteRegistration::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable* target =
          runtime->find_distributed_collectable(did);
      LamportClock registration_clock = 0;
      target->update_remote_instances(
          source, true /*registration*/, &registration_clock);
      // Send the acknowledgement through a response message carrying our
      // clock so an F16 bump reaches the registrant BEFORE its
      // registration event triggers (its packs must carry the bump);
      // the registrant triggers the event after folding the clock
      DistributedRegistrationResponse rez;
      {
        RezCheck z2(rez);
        rez.serialize(did);
        rez.serialize(registration_clock);
      }
      rez.dispatch(source);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::process_registration_response(
        LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(!remote_registered);
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      remote_registered = true;
      if (pending_remote_registered.exists())
      {
        Runtime::trigger_event(pending_remote_registered);
        pending_remote_registered = RtUserEvent::NO_RT_USER_EVENT;
      }
      if (pending_downgrade_restart)
        return can_delete(gc);
      else
        return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedRegistrationResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);
      // An unregistered replica cannot be deleted before this response
      // arrives, so we can use a blocking find and assert the result.
      // Deletion requires reaching LOCAL_REF_STATE, which requires
      // performing the GLOBAL-level downgrade locally, and every path to
      // that is closed while our registration is outstanding:
      //  1. Voting ready in a GLOBAL-level round: the registration gate
      //     in check_for_downgrade answers not-ready (or parks the
      //     self-check) until remote_registered is set -- by this
      //     handler.
      //  2. Applying a downgrade success: successes only apply to a
      //     replica that voted in the round, and voting is gate-blocked
      //     per (1).
      //  3. The catch-up in process_downgrade_request: a commit proof
      //     only applies the VALID-level downgrade, reaching
      //     GLOBAL_REF_STATE at most, never LOCAL_REF_STATE.
      // (Registration with the runtime's DID table happened before
      // send_remote_registration, so the find cannot block either.)
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      legion_assert(dc != nullptr);
      if (dc->process_registration_response(lamport_clock))
        delete dc;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::pack_global_ref(Serializer& rez, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      LamportClock lamport_clock = 0;
      pack_global_ref(lamport_clock, cnt);
      rez.serialize(lamport_clock);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::pack_global_ref(
        LamportClock& lamport_clock, unsigned cnt)
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
#endif
      legion_assert(
          (current_state == VALID_REF_STATE) ||
          (current_state == GLOBAL_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE));
      // Only permitted to pack a global reference if we've registered
      // with the owner node otherwise we need to wait for that
      // registration to succeed before we can pack in order to avoid
      // spurious successful downgrades
      if (!remote_registered)
      {
        if (!pending_remote_registered.exists())
          pending_remote_registered = Runtime::create_rt_user_event();
        const RtEvent wait_on = pending_remote_registered;
        gc.release();
        wait_on.wait();
        gc.reacquire();
        // Should still be global state
        legion_assert(
            (current_state == VALID_REF_STATE) ||
            (current_state == GLOBAL_REF_STATE) ||
            (current_state == PENDING_GLOBAL_REF_STATE));
      }
      // Under stamp-counting every counted pack is an event at the stamped
      // level, so the clock bump applies unconditionally: a global reference
      // packed from a valid-level node carries a valid-level count and can
      // otherwise mask an in-flight valid reference from the round tallies
      if (bump_downgrade_lamport_clock)
      {
        downgrade_lamport_clock++;
        bump_downgrade_lamport_clock = false;
      }
      // Encode the valid-level stamp in the low bit of the clock token so
      // it travels with every packed global reference: replicas are created
      // in the creator's known state and the send is counted at the stamped
      // level too, so a valid-level round cannot commit while a global
      // reference packed at the valid level is still in flight.
      // The shift caps the clock at 2^63-1 on the wire, which is safe: the
      // clock is per-object and only advances with downgrade rounds (each
      // at least one network round trip for this object), so it cannot
      // approach 2^63 in any object's lifetime. We use the low bit rather
      // than the high bit so the encoding transforms EVERY token (not just
      // valid-stamped ones): any path that forgets to decode is then wrong
      // for every message and fails loudly in the causality checks instead
      // of lurking until the first valid-stamped reference.
      lamport_clock = downgrade_lamport_clock << 1;
      if (record_valid_stamp(cnt))
        lamport_clock |= 1;
      sent_global_references += cnt;
#ifdef LEGION_DEBUG
      gc.release();
      // Should never have to delete this because we just packed a global ref
      if (remove_reference && remove_base_gc_ref(RUNTIME_REF))
        std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unpack_global_ref(
        Deserializer& derez, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);
      unpack_global_ref(lamport_clock, cnt);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unpack_global_ref(
        LamportClock lamport_clock, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      // Decode the valid-level stamp from the low bit of the clock token
      const bool valid_stamp = ((lamport_clock & 1) != 0);
      lamport_clock >>= 1;
      AutoLock gc(gc_lock);
      legion_assert(is_global<false /*need lock*/>());
      received_global_references += cnt;
      if (valid_stamp)
        apply_valid_stamp(cnt);
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      // No need to send any notifications if a downgrade is in process,
      // but we do need to record the veto so the in-flight round cannot
      // commit a decision that this unpacked reference invalidates
      if (remaining_responses == 0)
      {
        if (downgrade_owner == local_space)
        {
          // We're the downgrade owner so check to see if we can resume
          // doing collections or we can wait for the next removal
          check_for_downgrade_restart(
              gc, local_space, downgrade_owner_version,
              downgrade_lamport_clock);
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
          rez.serialize(local_space);  // propose ourselves as the candidate
          rez.serialize(downgrade_owner_version);
          rez.serialize(downgrade_lamport_clock);
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
      else
      {
        // A downgrade round is in flight through this node (we are the
        // round owner or an aggregating relay). The unpacked reference
        // must veto the round's decision: the counts we already reported
        // could otherwise cancel against this receipt and hide a live
        // reference from the tallies
        pending_downgrade_restart = true;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ LamportClock DistributedCollectable::unpack_global_ref_clock(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Note this returns the encoded clock token (clock plus the
      // valid-level stamp in the low bit); callers must treat it as
      // opaque and only feed it back to unpack_global_ref
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);
      return lamport_clock;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::has_packed_references(void) const
    //--------------------------------------------------------------------------
    {
      return (sent_global_references != received_global_references);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::record_valid_stamp(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      // Base distributed collectables have no valid level so their
      // packed references are never stamped with it
      return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::apply_valid_stamp(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      // A valid-level stamp can only be produced by an object with a
      // valid level, and both ends of a packed reference are the same
      // kind of object, so this should never be called on the base class
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_delete(AutoLock& gc)
    //--------------------------------------------------------------------------
    {
      if (pending_downgrade_restart && (remaining_responses == 0))
      {
        // Drain the deferred restart notification now that our references
        // have returned to zero. If a downgrade round is still in flight
        // through this node we must NOT clear the flag: it is the veto
        // that keeps the in-flight round from committing a decision that
        // the deferred traffic invalidates; the round's decision path
        // clears it.
        pending_downgrade_restart = false;
        if (downgrade_owner != local_space)
        {
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.serialize(local_space);  // propose ourselves as the candidate
          rez.serialize(downgrade_owner_version);
          rez.serialize(downgrade_lamport_clock);
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
                has_packed_references())
            {
              // If we're already checking for a downgrade but are awaiting
              // responses, then there is nothing to do
              if (remaining_responses > 0)
                return false;
              // Send messages to see if we can perform the deletion
              check_for_downgrade(
                  gc, downgrade_owner, downgrade_lamport_clock + 1);
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
      // GLOBAL is legal for any ownership here (finding F17): a remote
      // replica that ready-voted in the committing round can be silently
      // promoted back to GLOBAL by a covered (uncounted) add_gc_reference
      // before the round decides; the covering discipline guarantees the
      // commit waited for the cover chain to quiesce, so applying the
      // success at GLOBAL is exactly right (and gc_references == 0 above
      // still holds at delivery because no cover can exist post-commit).
      legion_assert(
          (current_state == GLOBAL_REF_STATE) ||
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
    void DistributedCollectable::check_for_downgrade(
        AutoLock& gc, AddressSpaceID owner, LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      legion_assert(remaining_responses == 0);
      // Registration gate: we must never vote in (or start) a downgrade
      // round until our registration with the owner space has been
      // acknowledged. Tallies only protect within a level: an instance
      // whose counts were retired by a committed level is invisible to
      // the next level's arithmetic, so the owner space's instance list
      // is the only thing carrying our existence across levels and it is
      // only complete once our registration has been processed.
      // The gate is an ENABLING CONDITION, exactly as in the TLA+ model:
      // it must NEVER block. This path is reached from
      // remove_*_reference calls where the caller has just surrendered
      // its only reference to this object, so releasing the lock to wait
      // here is a use-after-free window (finding F15) -- and F12/F14
      // were both races through the same window. If our registration has
      // not been acknowledged yet we are simply not ready: answer
      // not-ready when asked to vote, or defer the self-check until the
      // acknowledgement arrives (the model's RecvRegResp continuation).
      if (!remote_registered)
      {
        if (owner != local_space)
        {
          // Somebody asked us to vote: answer not-ready so the round
          // owner retries. Note we do NOT clear
          // pending_downgrade_restart here: it may be the veto
          // protecting a later round's decision. If responsibility for
          // this object later transfers to us, the self-check path
          // below defers and picks things up once we are registered.
          const AddressSpaceID target = round_parent;
          DistributedDowngradeResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
            rez.serialize<uint64_t>(0);  // sent global references
            rez.serialize<uint64_t>(0);  // received global references
            rez.serialize(downgrade_lamport_clock);
          }
          if (!pending_remote_registered.exists())
            pending_remote_registered = Runtime::create_rt_user_event();
          // As a performance optimization delay our response until
          // the registration is done so we don't poll unnecessarily
          rez.dispatch(target, pending_remote_registered);
        }
        else
          // We are checking ourselves as the downgrade owner: re-run
          // the check when the registration acknowledgement arrives
          pending_downgrade_restart = true;
        return;
      }
      // Record the owner of the round we are participating in; note we
      // do NOT adopt it as the downgrade owner here: adoption is gated
      // on the ownership version by our callers
      round_owner = owner;
      // A parked restart or deferred unpack notification vetoes any
      // ready vote we would cast in somebody else's round; the owner
      // itself may still start a round (its decision checks the veto)
      if (can_downgrade() && (downgrade_lamport_clock <= lamport_clock) &&
          ((owner == local_space) || !pending_downgrade_restart))
      {
        pending_downgrade_lamport_clock = lamport_clock;
        // Don't need to bump this new lamport clock until we do the accumulate
        bump_downgrade_lamport_clock = false;
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
                rez.serialize(downgrade_owner_version);
                rez.serialize(pending_downgrade_lamport_clock);
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
              rez.serialize(downgrade_owner_version);
              rez.serialize(pending_downgrade_lamport_clock);
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
            downgrade_functor.owner = owner;
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
            rez.serialize(downgrade_owner_version);
            rez.serialize(pending_downgrade_lamport_clock);
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
            const AddressSpaceID target = round_parent;
            DistributedDowngradeResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(owner);  // owner is special bottom value
              rez.serialize(total_sent_references);
              rez.serialize(total_received_references);
              rez.serialize(downgrade_lamport_clock);
            }
            rez.dispatch(target);
            record_pending_downgrade();
          }
          else
          {
            // We only get here if we're the owner and we don't know
            // about any remote instances yet. The only way that
            // should happen is if we have some packed references at
            // the current level. There's nothing to do yet since we
            // know we can't be deleted yet: the eventual unpack will
            // send us a restart notification.
            legion_assert(has_packed_references());
          }
        }
      }
      else if (local_space != owner)
      {
        // Our not-ready answer forces the owner to retry, which subsumes
        // any parked restart or deferred unpack notification we hold
        pending_downgrade_restart = false;
        const AddressSpaceID target = round_parent;
        DistributedDowngradeResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize<uint64_t>(0);              // sent global references
          rez.serialize<uint64_t>(0);              // received global references
          rez.serialize(downgrade_lamport_clock);  // doesn't matter
        }
        rez.dispatch(target);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::check_for_downgrade_restart(
        AutoLock& gc, AddressSpaceID candidate, uint64_t candidate_version,
        LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      // We can always safely update the lamport clock
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      // The object is already dead here; the restart is vestigial
      if ((current_state == LOCAL_REF_STATE) ||
          (current_state == DELETED_REF_STATE))
        return;
      if (downgrade_owner != local_space)
      {
        // We're not the downgrade owner. Forward the restart when we
        // provably know a fresher owner than the sender did (re-tagged
        // with our version so the chain terminates); otherwise park it:
        // the parked restart vetoes our next ready vote so it cannot be
        // lost while its cause is unresolved. Restarts must never be
        // silently dropped: a dropped restart is a lost wakeup that
        // leaks the object.
        if (downgrade_owner_version > candidate_version)
        {
          DistributedDowngradeRestart rez;
          rez.serialize(did);
          rez.serialize(candidate);
          rez.serialize(downgrade_owner_version);
          rez.serialize(downgrade_lamport_clock);
          rez.dispatch(downgrade_owner);
        }
        else
          pending_downgrade_restart = true;
        return;
      }
      // If there is a downgrade round in flight, park the restart as a
      // veto; the round's decision will observe it and retry
      if (remaining_responses > 0)
      {
        pending_downgrade_restart = true;
        return;
      }
      // Registration gate, mirroring check_for_downgrade: we must not
      // transfer ownership or start a round before our registration is
      // acknowledged. This also closes fuzzer finding F14: a thread gate-
      // waiting in check_for_downgrade validated its ownership belief
      // BEFORE releasing the lock, and a transfer through its window arms
      // a round rooted under a stale belief. Park the restart as a veto;
      // the gated round's decision or can_delete drains it.
      if (!remote_registered)
      {
        // Make sure the parked veto has a guaranteed wakeup: the
        // deferred registration check drains it (can_delete) once our
        // registration is acknowledged
        pending_downgrade_restart = true;
        return;
      }
      // If we can't downgrade then the removal of our own references
      // will restart the downgrade process
      if (!can_downgrade())
        return;
      legion_assert(
          (current_state == VALID_REF_STATE) ||
          (current_state == GLOBAL_REF_STATE));
      // Restart the downgrade process
      if (candidate != local_space)
      {
        // Transfer ownership to the candidate under a new version. Note
        // we don't require the candidate to be in remote_instances: this
        // can race with the candidate's registration, and the update
        // handler on the far side blocks in find_distributed_collectable
        // until the instance exists.
        downgrade_owner = candidate;
        downgrade_owner_version++;
        DistributedDowngradeUpdate rez;
        rez.serialize(did);
        rez.serialize(current_state);
        rez.serialize(downgrade_owner_version);
        rez.serialize(downgrade_lamport_clock);
        rez.dispatch(candidate);
      }
      else
        check_for_downgrade(gc, local_space, downgrade_lamport_clock + 1);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::accumulate_local_references(void)
    //--------------------------------------------------------------------------
    {
      total_sent_references += sent_global_references;
      total_received_references += received_global_references;
      // Incorporate the pending downgrade clock
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, pending_downgrade_lamport_clock);
      // Once we accumulate references then we need to bump the downgrade
      // lamport clock for any pack to detect them
      bump_downgrade_lamport_clock = true;
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
      uint64_t owner_version;
      derez.deserialize(owner_version);
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);

      // It's possible for this to race with the creation of this
      // distributed collectable so wait until it is ready
      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      dc->process_downgrade_request(
          source, downgrade_owner, owner_version, to_check, lamport_clock);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_request(
        AddressSpaceID source, AddressSpaceID owner, uint64_t owner_version,
        State to_check, LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner != local_space);  // we should be remote here
      legion_assert(
          (to_check == GLOBAL_REF_STATE) || (to_check == VALID_REF_STATE));
      AutoLock gc(gc_lock);
      // Our response is owed to the node that sent this request (it
      // counted us in its remaining_responses); remember it for every
      // response path of this round
      round_parent = source;
      // Adopt the round's downgrade owner only if its version is strictly
      // newer than the one we know; we still vote in the round either way
      if (owner_version > downgrade_owner_version)
      {
        downgrade_owner = owner;
        downgrade_owner_version = owner_version;
      }
      // If the owner is asking us to downgrade from a level below our
      // current state then this request is a commit proof: rounds for a
      // level are only started once the downgrade of the level above has
      // committed everywhere, so we can (and must) apply our own pending
      // downgrade of the level above before voting at the round's level.
      // We must have voted at the level above for it to have committed,
      // so we can only be in a pending state here; a node that is still
      // fully at the level above (e.g. holding valid references) seeing
      // a lower-level round is a protocol violation.
      if (to_check < current_state)
      {
        legion_assert(current_state == PENDING_GLOBAL_REF_STATE);
        perform_downgrade(gc);
        // perform_downgrade releases the lock for its invalidation
        // callback; another thread can have started a round through us
        // in that window (e.g. an ownership transfer making us the
        // downgrade owner). Answer not-ready so the round owner
        // retries; our own round subsumes this one's interest in us.
        if (remaining_responses > 0)
        {
          const AddressSpaceID target = round_parent;
          DistributedDowngradeResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
            rez.serialize<uint64_t>(0);  // sent global references
            rez.serialize<uint64_t>(0);  // received global references
            rez.serialize(downgrade_lamport_clock);
          }
          rez.dispatch(target);
          return;
        }
      }
      legion_assert(LOCAL_REF_STATE < current_state);
      // We must now be at the round's level: stamped creations guarantee
      // no replica can exist below the object's committed level, so a
      // node below the round's level would mean mixed-level counting
      legion_assert(
          (to_check != VALID_REF_STATE) || (current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE));
      check_for_downgrade(gc, owner, lamport_clock);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::process_downgrade_response(
        AddressSpaceID notready, uint64_t total_sent, uint64_t total_received,
        LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(remaining_responses > 0);
      // Merge in the lamport clock
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      if (notready != round_owner)
        notready_owner = notready;
      else if (notready_owner == round_owner)
      {
        // Everything still ready for downgrade
        total_sent_references += total_sent;
        total_received_references += total_received;
      }
      if (--remaining_responses == 0)
      {
        // Accumulate our local sent and received references
        accumulate_local_references();
        if (round_owner == local_space)
        {
          legion_assert(downgrade_owner == local_space);
          legion_assert(
              (current_state == VALID_REF_STATE) ||
              (current_state == GLOBAL_REF_STATE));
          // See if it safe to downgrade
          // Make sure to check ourselves again to handle any
          // check_*_and_increment methods; a pending restart (deferred
          // unpack traffic or a parked restart) vetoes the decision
          if (can_downgrade() && (notready_owner == round_owner) &&
              (total_sent_references == total_received_references) &&
              (downgrade_lamport_clock <= pending_downgrade_lamport_clock) &&
              !pending_downgrade_restart)
          {
            // Then perform our local downgrade
            return perform_downgrade(gc);
          }
          else
          {
            // Not ready to downgrade
            if (notready_owner != round_owner)
            {
              // Update the new owner responsible for checking for
              // downgrades; every ownership transfer mints a new version
              downgrade_owner = notready_owner;
              downgrade_owner_version++;
              DistributedDowngradeUpdate rez;
              rez.serialize(did);
              rez.serialize(current_state);
              rez.serialize(downgrade_owner_version);
              rez.serialize(downgrade_lamport_clock);
              rez.dispatch(notready_owner);
              if (pending_downgrade_restart)
              {
                // Hand the veto we were holding to the new owner as a
                // restart proposing us: it must not conclude we're quiet
                pending_downgrade_restart = false;
                DistributedDowngradeRestart rez2;
                rez2.serialize(did);
                rez2.serialize(local_space);
                rez2.serialize(downgrade_owner_version);
                rez2.serialize(downgrade_lamport_clock);
                rez2.dispatch(notready_owner);
              }
            }
            // else: we used to do this, but the polling aspect of continuing
            // to check for downgrades can cause priority inversions in the
            // network traffic (despite setting low priorities because the
            // networking hardware ignores our priorities). Therefore we stopped
            // doing this and now we instead send notifications whenever we
            // do an unpack that might need to restart this process. The first
            // one to get here will restart the downgrade process.
            // See the calls to check_for_downgrade_restart to see where
            // progress comes from now. The exceptions are a causality
            // violation and a vetoed decision (a restart arrived or
            // reference traffic passed through mid-round), where the
            // retry responsibility is ours
            else
            {
              const bool retry =
                  (pending_downgrade_lamport_clock < downgrade_lamport_clock) ||
                  pending_downgrade_restart;
              pending_downgrade_restart = false;
              if (retry)
                check_for_downgrade(
                    gc, downgrade_owner, downgrade_lamport_clock);
            }
          }
        }
        else
        {
          const AddressSpaceID target = round_parent;
          // We had to release the lock to send the requests to our upstream
          // nodes so we need to check again to see if it is still safe to
          // perform the downgrade on this node or not atomically with
          // accumulating our sent and received references; a pending
          // restart vetoes our ready vote just like at a leaf
          DistributedDowngradeResponse rez;
          if (can_downgrade() &&
              (downgrade_lamport_clock <= pending_downgrade_lamport_clock) &&
              !pending_downgrade_restart)
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(notready_owner);
            rez.serialize(total_sent_references);
            rez.serialize(total_received_references);
            rez.serialize(downgrade_lamport_clock);
            // Record that we're in the pending downgrade state
            record_pending_downgrade();
          }
          else
          {
            // Our not-ready answer forces the round owner to retry,
            // which subsumes any parked restart we hold
            pending_downgrade_restart = false;
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(local_space);
            rez.serialize<uint64_t>(0);  // sent global references
            rez.serialize<uint64_t>(0);  // received global references
            rez.serialize(downgrade_lamport_clock);
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
      LamportClock lamport_clock;
      derez.deserialize(total_sent);
      derez.deserialize(total_received);
      derez.deserialize(lamport_clock);

      DistributedCollectable* dc = runtime->find_distributed_collectable(did);
      if (dc->process_downgrade_response(
              notready, total_sent, total_received, lamport_clock))
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
        AutoLock& gc, State to_check, LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      legion_assert(to_check == GLOBAL_REF_STATE);
      // We're the new downgrade owner (the version was already adopted
      // by the handler before calling us)
      downgrade_owner = local_space;
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      // If we voted in the round that just failed, roll our vote back;
      // ownership transfers never move the state downward: downgrades
      // only ever happen through committed rounds or their success
      // notifications
      if (current_state == PENDING_LOCAL_REF_STATE)
        current_state = GLOBAL_REF_STATE;
      if (remaining_responses == 0)
      {
        // We're the owner now so we act on (and thereby drain) any
        // parked restart or deferred notification directly; if a round
        // is still aggregating through this node the flag stays as the
        // veto protecting that round's decision
        pending_downgrade_restart = false;
        if ((current_state == GLOBAL_REF_STATE) && (gc_references == 0))
          check_for_downgrade(gc, downgrade_owner, downgrade_lamport_clock + 1);
      }
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
      uint64_t owner_version;
      derez.deserialize(owner_version);
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);

      // This transfer can race with the creation of the target instance:
      // it can name a node whose instance's creating message is still in
      // flight on another channel. We must NOT block waiting for it: this
      // handler runs on the ordered acquire virtual channel and blocking
      // wedges everything queued behind it -- including the acquire
      // grant whose waiter may be the very handler unpacking the payload
      // that creates this instance (finding F18: livelocked CI runs with
      // a self-sustaining restart storm at 300M+ scheduler iterations).
      // Instead the runtime parks the transfer and applies it inline at
      // registration, before the pending-collectable event triggers, so
      // every handler that unblocks on that event observes the transfer
      // already applied -- preserving the ordering this channel provides
      // between updates and the acquire requests behind them.
      DistributedCollectable* dc = runtime->find_or_park_downgrade_update(
          did, static_cast<unsigned>(state), owner_version, lamport_clock);
      if (dc == nullptr)
        return;
      DistributedCollectable::process_downgrade_update_message(
          dc, state, owner_version, lamport_clock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::process_downgrade_update_message(
        DistributedCollectable* dc, State state, uint64_t owner_version,
        LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(dc->gc_lock);
      // Ownership is only ever adopted from strictly newer versions; a
      // stale transfer (e.g. one that raced with a fresher one through
      // another node) must be dropped or it can resurrect a dead owner
      if (owner_version <= dc->downgrade_owner_version)
        return;
      dc->downgrade_owner_version = owner_version;
      dc->process_downgrade_update(gc, state, lamport_clock);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedDowngradeRestart::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID candidate;
      derez.deserialize(candidate);
      uint64_t candidate_version;
      derez.deserialize(candidate_version);
      LamportClock lamport_clock;
      derez.deserialize(lamport_clock);
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
          dc->check_for_downgrade_restart(
              gc, candidate, candidate_version, lamport_clock);
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
        AddressSpaceID& current, int count, AddressSpaceID source,
        LamportClock& lamport_clock)
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
          {
            if (bump_downgrade_lamport_clock)
            {
              downgrade_lamport_clock++;
              bump_downgrade_lamport_clock = false;
            }
            lamport_clock = downgrade_lamport_clock;
            sent_valid_references++;
          }
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
        LamportClock lamport_clock = 0;
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_valid_remote(
                current_owner, count, source, lamport_clock))
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
              rez.serialize(lamport_clock);
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
      local->unpack_valid_ref(derez);
      result->store(true);
      Runtime::trigger_event(ready);
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::pack_valid_ref(
        Serializer& rez, unsigned cnt)
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
#endif
      // Must be valid when packing a reference
      legion_assert(current_state == VALID_REF_STATE);
      // Only permitted to pack a valid reference if we've registered
      // with the owner node otherwise we need to wait for that
      // registration to succeed before we can pack in order to avoid
      // spurious successful downgrades
      if (!remote_registered)
      {
        if (!pending_remote_registered.exists())
          pending_remote_registered = Runtime::create_rt_user_event();
        const RtEvent wait_on = pending_remote_registered;
        gc.release();
        wait_on.wait();
        gc.reacquire();
        // Should still be valid
        legion_assert(current_state == VALID_REF_STATE);
      }
      if (bump_downgrade_lamport_clock)
      {
        downgrade_lamport_clock++;
        bump_downgrade_lamport_clock = false;
      }
      rez.serialize(downgrade_lamport_clock);
      sent_valid_references += cnt;
#ifdef LEGION_DEBUG
      gc.release();
      // Should never have to delete this because we just packed a valid ref
      if (remove_reference && remove_base_valid_ref(RUNTIME_REF))
        std::abort();
#endif
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::unpack_valid_ref(
        Deserializer& derez, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      legion_assert(is_valid<false /*need lock*/>());
      received_valid_references += cnt;
      LamportClock prev_clock;
      derez.deserialize(prev_clock);
      downgrade_lamport_clock = std::max(downgrade_lamport_clock, prev_clock);
      // No need to send any notifications if a downgrade is in process,
      // but we do need to record the veto so the in-flight round cannot
      // commit a decision that this unpacked reference invalidates
      if (remaining_responses == 0)
      {
        if (downgrade_owner == local_space)
        {
          // We're the downgrade owner so check to see if we can resume
          // doing collections or we can wait for the next removal
          check_for_downgrade_restart(
              gc, local_space, downgrade_owner_version,
              downgrade_lamport_clock);
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
          rez.serialize(local_space);  // propose ourselves as the candidate
          rez.serialize(downgrade_owner_version);
          rez.serialize(downgrade_lamport_clock);
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
      else
      {
        // A downgrade round is in flight through this node: the unpacked
        // reference must veto the round's decision so the counts we
        // already reported cannot cancel against this receipt
        pending_downgrade_restart = true;
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
        // VALID is legal for any ownership here: the valid-level analog
        // of the covered promotion (add_valid_reference bumps
        // PENDING_GLOBAL back to VALID without notifying the round
        // owner); see finding F17
        legion_assert(
            (current_state == VALID_REF_STATE) ||
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
        AutoLock& gc, State to_check, LamportClock lamport_clock)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          (to_check == VALID_REF_STATE) || (to_check == GLOBAL_REF_STATE));
      downgrade_lamport_clock =
          std::max(downgrade_lamport_clock, lamport_clock);
      // Ownership transfers are stamped with the sender's level; only
      // roll back our pending vote if the failed round was at our
      // pending level. An update stamped with the global level arriving
      // while we are pending-global is a commit proof for the valid
      // level (the sender had already left it, which can only happen
      // once the valid level committed everywhere), so we apply our
      // pending valid-level downgrade instead of rolling it back:
      // resurrecting a committed valid-level vote would let this node
      // return to the valid level after the object has died there.
      if (current_state == PENDING_GLOBAL_REF_STATE)
      {
        if (to_check == VALID_REF_STATE)
          current_state = VALID_REF_STATE;
        else
        {
          // Do this before adopting the downgrade ownership below so
          // perform_downgrade still sees us as a remote voter
          perform_downgrade(gc);
          // perform_downgrade releases the lock for its invalidation
          // callback; a fresher ownership transfer can have raced
          // through the window and started a round through us. Its
          // round subsumes ours, and pending_downgrade_restart may be
          // the veto protecting its decision, so leave it alone.
          if (remaining_responses > 0)
          {
            downgrade_owner = local_space;
            return;
          }
        }
      }
      else if (current_state == PENDING_LOCAL_REF_STATE)
      {
        legion_assert(to_check == GLOBAL_REF_STATE);
        current_state = GLOBAL_REF_STATE;
      }
      // We're the new downgrade owner (the version was already adopted
      // by the handler before calling us)
      downgrade_owner = local_space;
      if (remaining_responses == 0)
      {
        // We're the owner now so we act on (and thereby drain) any
        // parked restart or deferred notification directly; if a round
        // is still aggregating through this node the flag stays as the
        // veto protecting that round's decision
        pending_downgrade_restart = false;
        if (((current_state == VALID_REF_STATE) && (valid_references == 0)) ||
            ((current_state == GLOBAL_REF_STATE) && (gc_references == 0)))
          check_for_downgrade(gc, downgrade_owner, downgrade_lamport_clock + 1);
      }
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
        // Incorporate the pending downgrade clock
        downgrade_lamport_clock =
            std::max(downgrade_lamport_clock, pending_downgrade_lamport_clock);
        // Once we accumulate references then we need to bump the downgrade
        // lamport clock for any pack to detect them
        bump_downgrade_lamport_clock = true;
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

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::has_packed_references(void) const
    //--------------------------------------------------------------------------
    {
      // At the valid level we have to account for the valid-level counts
      // as well: creations are counted at the level of their stamp, so a
      // packed reference at either level can be carrying our valid level
      // to a new replica
      if ((current_state == VALID_REF_STATE) ||
          (current_state == PENDING_GLOBAL_REF_STATE))
        return (sent_valid_references != received_valid_references) ||
               (sent_global_references != received_global_references);
      return DistributedCollectable::has_packed_references();
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::record_valid_stamp(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      // Global references packed while we are at the valid level are
      // stamped with it and counted at the valid level as well: the
      // receiver may be a brand new replica that must be born in our
      // known state, and the valid-level round tallies must see the
      // creation while it is in flight
      if ((current_state != VALID_REF_STATE) &&
          (current_state != PENDING_GLOBAL_REF_STATE))
        return false;
      sent_valid_references += cnt;
      return true;
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::apply_valid_stamp(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      received_valid_references += cnt;
      // Replicas are born in the creator's known state: if a creation
      // site constructed us at the global level while the creator was
      // still at the valid level, promote to the stamped level. For a
      // replica that already existed, receiving a valid-level stamp
      // below the valid level is unreachable: the stamped count blocks
      // the valid-level round from committing while the reference is in
      // flight, so nothing can have moved us below the valid level.
      legion_assert(current_state != PENDING_LOCAL_REF_STATE);
      legion_assert(current_state != LOCAL_REF_STATE);
      if (current_state == GLOBAL_REF_STATE)
        current_state = VALID_REF_STATE;
    }

  }  // namespace Internal
}  // namespace Legion
