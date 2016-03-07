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
#include "garbage_collection.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // DistributedCollectable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(Runtime *rt,
                                                   DistributedID id,
                                                   AddressSpaceID own_space,
                                                   AddressSpaceID loc_space,
                                                   bool do_registration)
      : runtime(rt), did(id), owner_space(own_space), 
        local_space(loc_space), current_state(INACTIVE_STATE),
        gc_lock(Reservation::create_reservation()), gc_references(0), 
        valid_references(0), resource_references(0), 
        destruction_event(UserEvent::create_user_event()),
        registered_with_runtime(do_registration)
    //--------------------------------------------------------------------------
    {
      if (do_registration)
      {
        runtime->register_distributed_collectable(did, this);
        if (!is_owner())
          send_remote_registration();
      }
      if (!is_owner())
        remote_instances.add(owner_space);
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::~DistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(gc_references == 0);
      assert(valid_references == 0);
      assert(resource_references == 0);
#endif
      destruction_event.trigger(Runtime::merge_events<true>(recycle_events));
      if (registered_with_runtime)
      {
        runtime->unregister_distributed_collectable(did);
        if (is_owner())
        {
          // We can only recycle the distributed ID on the owner
          // node since the ID is the same across all the nodes.
          // We have to defer the collection of the ID until
          // after all of the remote nodes notify us that they
          // have finished collecting it.
          runtime->recycle_distributed_id(did, destruction_event);
        }
      }
      gc_lock.destroy_reservation();
      gc_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
          gc_references += cnt;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
      {
        // If we get here it is probably a race in reference counting
        // scheme above, so mark it is as such
        assert(false);
        delete this;
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(gc_references >= cnt);
#endif
          gc_references -= cnt;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
      assert(current_state != ACTIVE_DELETED_STATE);
      assert(current_state != PENDING_ACTIVE_DELETED_STATE);
      assert(current_state != PENDING_INVALID_DELETED_STATE);
      assert(current_state != PENDING_INACTIVE_DELETED_STATE);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
          valid_references += cnt;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
      {
        // This probably indicates a race in reference counting algorithm
        assert(false);
        delete this;
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
      assert(current_state != ACTIVE_DELETED_STATE);
      assert(current_state != PENDING_ACTIVE_DELETED_STATE);
      assert(current_state != PENDING_INVALID_DELETED_STATE);
      assert(current_state != PENDING_INACTIVE_DELETED_STATE);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(valid_references >= cnt);
#endif
          valid_references -= cnt;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::try_add_valid_reference(bool must_be_valid,
                                                        unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        // Check our state and see if this is going to work
        if (must_be_valid && (current_state != VALID_STATE))
          return false;
        // If we are in any of the deleted states, it is no good
        if ((current_state == DELETED_STATE) || 
            (current_state == ACTIVE_DELETED_STATE) ||
            (current_state == PENDING_ACTIVE_DELETED_STATE) ||
            (current_state == PENDING_INVALID_DELETED_STATE) ||
            (current_state == PENDING_INACTIVE_DELETED_STATE))
          return false;
        if (first)
        {
          valid_references += cnt;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
      {
        // This probably indicates a race in reference counting algorithm
        assert(false);
        delete this;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::try_active_deletion(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      // We can only do this from five states
      // Note this also prevents duplicate deletions
      if (current_state == INACTIVE_STATE)
      {
        current_state = DELETED_STATE;
        return true;
      }
      if (current_state == ACTIVE_INVALID_STATE)
      {
        current_state = ACTIVE_DELETED_STATE;
        return true;
      }
      if (current_state == PENDING_INACTIVE_STATE)
      {
        current_state = PENDING_INACTIVE_DELETED_STATE;
        return true;
      }
      if (current_state == PENDING_ACTIVE_STATE)
      {
        current_state = PENDING_ACTIVE_DELETED_STATE;
        return true;
      }
      if (current_state == PENDING_INVALID_STATE)
      {
        current_state = PENDING_INVALID_DELETED_STATE;
        return true;
      }
      return false; 
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
#endif
      AutoLock gc(gc_lock);
      resource_references += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state != DELETED_STATE);
#endif
      AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(resource_references >= cnt);
#endif
      resource_references -= cnt;
      return can_delete();
    }

#ifdef USE_REMOTE_REFERENCES
    //--------------------------------------------------------------------------
    bool DistributedCollectable::add_create_reference(AddressSpaceID source,
                                      AddressSpaceID target, ReferenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
      assert(source != owner_space);
      assert(current_state != DELETED_STATE);
      assert((kind == GC_REF_KIND) || (kind == VALID_REF_KIND));
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
          std::pair<AddressSpaceID,AddressSpaceID> key(source, target);
          if (kind == VALID_REF_KIND)
          {
            std::map<std::pair<AddressSpaceID,AddressSpaceID>,int>::iterator 
              finder = create_valid_refs.find(key);
            if (finder != create_valid_refs.end())
            {
              finder->second += 1;
              if (finder->second == 0)
                create_valid_refs.erase(finder);
            }
            else
              create_valid_refs[key] = 1;
          }
          else
          {
            std::map<std::pair<AddressSpaceID,AddressSpaceID>,int>::iterator 
              finder = create_gc_refs.find(key);
            if (finder != create_gc_refs.end())
            {
              finder->second += 1;
              if (finder->second == 0)
                create_gc_refs.erase(finder);
            }
            else
              create_gc_refs[key] = 1;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_create_reference(AddressSpaceID source,
                                      AddressSpaceID target, ReferenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner());
      assert(source != owner_space);
      assert(current_state != DELETED_STATE);
      assert((kind == GC_REF_KIND) || (kind == VALID_REF_KIND));
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool do_deletion = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_active();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          notify_inactive();
        AutoLock gc(gc_lock);
        if (first)
        {
          std::pair<AddressSpaceID,AddressSpaceID> key(source, target);
          if (kind == VALID_REF_KIND)
          {
            std::map<std::pair<AddressSpaceID,AddressSpaceID>,int>::iterator 
              finder = create_valid_refs.find(key);
            if (finder != create_valid_refs.end())
            {
              finder->second -= 1;
              if (finder->second == 0)
                create_valid_refs.erase(finder);
            }
            else
              create_valid_refs[key] = -1;
          }
          else
          {
            std::map<std::pair<AddressSpaceID,AddressSpaceID>,int>::iterator 
              finder = create_gc_refs.find(key);
            if (finder != create_gc_refs.end())
            {
              finder->second -= 1;
              if (finder->second == 0)
                create_gc_refs.erase(finder);
            }
            else
              create_gc_refs[key] = -1;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }
#endif // USE_REMOTE_REFERENCES

    //--------------------------------------------------------------------------
    bool DistributedCollectable::has_remote_instance(
                                               AddressSpaceID remote_inst) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return remote_instances.contains(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_instances(AddressSpaceID remote)
    //--------------------------------------------------------------------------
    {
      if (remote != owner_space)
      {
        AutoLock gc(gc_lock);
        remote_instances.add(remote);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::register_remote_instance(AddressSpaceID source,
                                                          Event destroyed)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(is_owner()); // This should only happen on the owner node
#endif
      AutoLock gc(gc_lock);
      remote_instances.add(source);
      recycle_events.insert(destroyed);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::register_with_runtime(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!registered_with_runtime);
#endif
      registered_with_runtime = true;
      runtime->register_distributed_collectable(did, this);
      if (!is_owner())
        send_remote_registration();
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_registration(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!is_owner());
      assert(registered_with_runtime);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(destruction_event);
      }
      runtime->send_did_remote_registration(owner_space, rez);     
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_valid_update(AddressSpaceID target,
                                                       unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(registered_with_runtime);
#endif
      int signed_count = count;
      if (!add)
        signed_count = -signed_count;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
      }
      runtime->send_did_remote_valid_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_gc_update(AddressSpaceID target,
                                                       unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(registered_with_runtime);
#endif
      int signed_count = count;
      if (!add)
        signed_count = -signed_count;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
      }
      runtime->send_did_remote_gc_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_resource_update(
                                AddressSpaceID target, unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(registered_with_runtime);
#endif
      int signed_count = count;
      if (!add)
        signed_count = -signed_count;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
      }
      runtime->send_did_remote_resource_update(target, rez);
    }

#ifdef USE_REMOTE_REFERENCES
    //--------------------------------------------------------------------------
    ReferenceKind DistributedCollectable::send_create_reference(
                                                          AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // If the target is the owner then no need to do anything
      if (target == owner_space)
        return GC_REF_KIND;
      // Sample our current state, it's up to the caller to make sure
      // this is a good state for this object
      State copy = current_state;
      // We better be holding a gc reference or a valid reference when
      // we are doing this operation
#ifdef DEBUG_HIGH_LEVEL
      assert((copy == ACTIVE_INVALID_STATE) || (copy == VALID_STATE) ||
             (copy == PENDING_VALID_STATE) || (copy == PENDING_INVALID_STATE));
#endif
      ReferenceKind result; 
      if (copy == VALID_STATE)
        result = VALID_REF_KIND;
      else
        result = GC_REF_KIND;
      if (is_owner())
      {
        if (result == VALID_REF_KIND)
          add_base_valid_ref(REMOTE_CREATE_REF);
        else
          add_base_gc_ref(REMOTE_CREATE_REF);
      }
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize(target);
          rez.serialize(result);
        }
        runtime->send_did_add_create_reference(owner_space, rez);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::post_create_reference(ReferenceKind kind,
                                              AddressSpaceID target, bool flush)
    //--------------------------------------------------------------------------
    {
      // No need to do anything if we are sending stuff to the owner
      if (target == owner_space)
        return;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(local_space);
        rez.serialize(target);
        rez.serialize(owner_space);
        rez.serialize(kind);
      }
      runtime->send_did_remove_create_reference(target, rez, flush);
    }
#endif // USE_REMOTE_REFERENCES

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_registration(
                  Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Event destroy_event;
      derez.deserialize(destroy_event);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      target->register_remote_instance(source, destroy_event);
    }
    
    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_valid_update(
                                         Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      int count;
      derez.deserialize(count);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (count > 0)
        target->add_base_valid_ref(REMOTE_DID_REF, unsigned(count));
      else if (target->remove_base_valid_ref(REMOTE_DID_REF, unsigned(-count)))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_gc_update(
                                         Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      int count;
      derez.deserialize(count);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (count > 0)
        target->add_base_gc_ref(REMOTE_DID_REF, unsigned(count));
      else if (target->remove_base_gc_ref(REMOTE_DID_REF, unsigned(-count)))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_resource_update(
                                         Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      int count;
      derez.deserialize(count);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (count > 0)
        target->add_base_resource_ref(REMOTE_DID_REF, unsigned(count));
      else if (target->remove_base_resource_ref(REMOTE_DID_REF, 
                                                unsigned(-count)))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_add_create(
                                         Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef USE_REMOTE_REFERENCES
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      AddressSpaceID target;
      derez.deserialize(target);
      ReferenceKind kind;
      derez.deserialize(kind);
      DistributedCollectable *dist = 
        runtime->find_distributed_collectable(did);
      if (dist->add_create_reference(source, target, kind))
        delete dist;
#else
      assert(false);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remove_create(
                                         Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef USE_REMOTE_REFERENCES
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      AddressSpaceID target;
      derez.deserialize(target);
      AddressSpaceID owner;
      derez.deserialize(owner);
      ReferenceKind kind;
      derez.deserialize(kind);
      // Check to see if we are on the owner node or whether we should
      // keep forwarding this message onto the owner
      if (runtime->address_space == owner)
      {
        // We're the owner so handle it
        DistributedCollectable *dist = 
          runtime->find_distributed_collectable(did);
        if (source == owner)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert((kind == GC_REF_KIND) || (kind == VALID_REF_KIND));
#endif
          if (kind == VALID_REF_KIND)
          {
            if (dist->remove_base_valid_ref(REMOTE_CREATE_REF))
              delete dist;
          }
          else
          {
            if (dist->remove_base_gc_ref(REMOTE_CREATE_REF))
              delete dist;
          }
        }
        else
        {
          if (dist->remove_create_reference(source, target, kind))
            delete dist;
        }
      }
      else
      {
        // Keep forwarding on to the owner
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(source);
          rez.serialize(target);
          rez.serialize(owner);
          rez.serialize(kind);
        }
        runtime->send_did_remove_create_reference(owner, rez);
      }
#else
      assert(false);
#endif
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::update_state(bool &need_activate, 
                                              bool &need_validate,
                                              bool &need_invalidate,
                                              bool &need_deactivate,
                                              bool &do_deletion)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
      switch (current_state)
      {
        case INACTIVE_STATE:
          {
            // See if we have any reason to be active
#ifdef USE_REMOTE_REFERENCES
            if ((valid_references > 0) || (!create_valid_refs.empty()))
#else
            if (valid_references > 0)
#endif
            {
              current_state = PENDING_ACTIVE_VALID_STATE;
              need_activate = true;
            }
#ifdef USE_REMOTE_REFERENCES
            else if ((gc_references > 0) || !create_gc_refs.empty())
#else
            else if (gc_references > 0)  
#endif
            {
              current_state = PENDING_ACTIVE_STATE;
              need_activate = true;
            }
            need_validate = false;
            need_invalidate = false;
            need_deactivate = false;
            break;
          }
        case ACTIVE_INVALID_STATE:
          {
            // See if we have a reason to be valid
#ifdef USE_REMOTE_REFERENCES
            if ((valid_references > 0) || !create_valid_refs.empty())
#else
            if (valid_references > 0)
#endif
            {
              // Move to a pending valid state
              current_state = PENDING_VALID_STATE;
              need_validate = true;
              need_deactivate = false;
            }
            // See if we have a reason to be inactive
#ifdef USE_REMOTE_REFERENCES
            else if ((gc_references == 0) && create_gc_refs.empty())
#else
            else if (gc_references == 0)
#endif
            {
              current_state = PENDING_INACTIVE_STATE;
              need_validate = false;
              need_deactivate = true;
            }
            else
            {
              need_validate = false;
              need_deactivate = false;
            }
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case ACTIVE_DELETED_STATE:
          {
            // We should never move to a valid state from here
#ifdef USE_REMOTE_REFERENCES
            if ((valid_references > 0) || !create_valid_refs.empty())
#else
            if (valid_references > 0)
#endif           
              assert(false);
            // See if we have a reason to move towards deletion
#ifdef USE_REMOTE_REFERENCES
            else if ((gc_references == 0) && create_gc_refs.empty())
#else
            else if (gc_references == 0)
#endif
            {
              current_state = PENDING_INACTIVE_DELETED_STATE;
              need_deactivate = true;
            }
            else
              need_deactivate = false;
            need_validate = false;
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case VALID_STATE:
          {
            // See if we have a reason to be invalid
#ifdef USE_REMOTE_REFERENCES
            if ((valid_references == 0) && create_valid_refs.empty())
#else
            if (valid_references == 0)
#endif
            {
              current_state = PENDING_INVALID_STATE;
              need_invalidate = true;
            }
            else
              need_invalidate = false;
            need_activate = false;
            need_validate = false;
            need_deactivate = false;
            break;
          }
        case DELETED_STATE:
          {
            // Hitting this is universally bad
            assert(false);
            break;
          }
        case PENDING_ACTIVE_STATE:
          {
            // See if we were the ones doing the activating
            if (need_activate)
            {
              // See if we are still active
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // Now we need a validate
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references > 0) || !create_gc_refs.empty())
#else
              else if (gc_references > 0)
#endif
              {
                // Nothing more to do
                current_state = ACTIVE_INVALID_STATE;
                need_validate = false;
                need_deactivate = false;
              }
              else
              {
                // Not still valid, go to pending inactive 
                current_state = PENDING_INACTIVE_STATE;
                need_validate = false;
                need_deactivate = true;
              }
            }
            else
            {
              // We weren't the ones doing the activate so 
              // we can't help, just keep going
              need_validate = false;
              need_deactivate = false;
            }
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_INACTIVE_STATE:
          {
            // See if we were doing the deactivate
            if (need_deactivate)
            {
              // See if we are still inactive
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                current_state = PENDING_ACTIVE_VALID_STATE;
                need_activate = true;
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references == 0) && create_gc_refs.empty())
#else
              else if (gc_references == 0)
#endif
              {
                current_state = INACTIVE_STATE;
                need_activate = false;
              }
              else
              {
                current_state = PENDING_ACTIVE_STATE;
                need_activate = true;
              }
            }
            else
            {
              // We weren't the ones doing the deactivate
              // so we can't help, just keep going
              need_activate = false;
            }
            need_validate = false;
            need_invalidate = false;
            need_deactivate = false;
            break;
          }
        case PENDING_VALID_STATE:
          {
            // See if we were the ones doing the validate
            if (need_validate)
            {
              // Check to see if we are still valid
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                current_state = VALID_STATE;
                need_invalidate = false;
              }
              else
              {
                current_state = PENDING_INVALID_STATE;
                need_invalidate = true;
              }
            }
            else
            {
              // We weren't the ones doing the validate
              // so we can't help, just keep going
              need_invalidate = false;
            }
            need_activate = false;
            need_validate = false;
            need_deactivate = false;
            break;
          }
        case PENDING_INVALID_STATE:
          {
            // See if we were doing the invalidate
            if (need_invalidate)
            {
              // Check to see if we are still valid
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // Now we are valid again
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references == 0) && create_gc_refs.empty())
#else
              else if (gc_references == 0)
#endif
              {
                // No longer active either
                current_state = PENDING_INACTIVE_STATE;
                need_validate = false;
                need_deactivate = true;
              }
              else
              {
                current_state = ACTIVE_INVALID_STATE;
                need_validate = false;
                need_deactivate = false;
              }
            }
            else
            {
              // We weren't the ones doing the invalidate
              // so we can't help
              need_validate = false;
              need_deactivate = false;
            }
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_ACTIVE_VALID_STATE:
          {
            // See if we were the ones doing the action
            if (need_activate)
            {
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // Still going to valid
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references == 0) && create_gc_refs.empty())
#else
              else if (gc_references == 0)
#endif
              {
                // all our references disappeared
                current_state = PENDING_INACTIVE_STATE;
                need_validate = false;
                need_deactivate = true;
              }
              else
              {
                // our valid references disappeared, but we at least
                // have some gc references now
                current_state = ACTIVE_INVALID_STATE;
                need_validate = false;
                need_deactivate = false;
              }
            }
            else
            {
              // We weren't the ones doing the activate so we are of no help
              need_validate = false;
              need_deactivate = false;
            }
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_ACTIVE_DELETED_STATE:
          {
            // See if were the ones doing the work
            if (need_activate)
            {
// See if we are still active
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // This is really bad if it happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references > 0) || !create_gc_refs.empty())
#else
              else if (gc_references > 0)
#endif
              {
                // Nothing more to do
                current_state = ACTIVE_DELETED_STATE;
                need_deactivate = false;
              }
              else
              {
                // Not still active, go to pending inactive 
                current_state = PENDING_INACTIVE_DELETED_STATE;
                need_deactivate = true;
              }
            }
            else
            {
              // We weren't the ones doing the activate so 
              // we can't help, just keep going
              need_deactivate = false;
            }
            need_activate = false;
            need_validate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_INVALID_DELETED_STATE:
          {
            // See if we were doing the invalidate
            if (need_invalidate)
            {
              // Check to see if we are still valid
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // Really bad if this happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references == 0) && create_gc_refs.empty())
#else
              else if (gc_references == 0)
#endif
              {
                // No longer active either
                current_state = PENDING_INACTIVE_DELETED_STATE;
                need_deactivate = true;
              }
              else
              {
                current_state = ACTIVE_DELETED_STATE;
                need_deactivate = false;
              }
            }
            else
            {
              // We weren't the ones doing the invalidate
              // so we can't help
              need_deactivate = false;
            }
            need_validate = false;
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_INACTIVE_DELETED_STATE:
          {
            // See if we were doing the deactivate
            if (need_deactivate)
            {
              // See if we are still inactive
#ifdef USE_REMOTE_REFERENCES
              if ((valid_references > 0) || !create_valid_refs.empty())
#else
              if (valid_references > 0)
#endif
              {
                // This is really bad if it happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if ((gc_references == 0) && create_gc_refs.empty())
#else
              else if (gc_references == 0)
#endif
              {
                current_state = DELETED_STATE;
                need_activate = false;
              }
              else
              {
                current_state = PENDING_ACTIVE_DELETED_STATE;
                need_activate = true;
              }
            }
            else
            {
              // We weren't the ones doing the deactivate
              // so we can't help, just keep going
              need_activate = false;
            }
            need_validate = false;
            need_invalidate = false;
            need_deactivate = false;
            break;
          }
        default:
          assert(false); // should never get here
      }
      const bool done = !(need_activate || need_validate || 
                          need_invalidate || need_deactivate);
      if (done)
        do_deletion = can_delete();
      else
        do_deletion = false;
      return done;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_delete(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef USE_REMOTE_REFERENCES
      bool result = ((resource_references == 0) && (gc_references == 0) &&
              (valid_references == 0) && create_gc_refs.empty() && 
              create_valid_refs.empty() && 
              ((current_state == INACTIVE_STATE) || 
               (current_state == DELETED_STATE)));
#else
      bool result = ((resource_references == 0) && (gc_references == 0) &&
              (valid_references == 0) && 
              ((current_state == INACTIVE_STATE) || 
               (current_state == DELETED_STATE)));
#endif
      return result;
    }

  }; // namespace Internal 
}; // namespace Legion

// EOF

