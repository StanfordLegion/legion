/* Copyright 2015 Stanford University, NVIDIA Corporation
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

namespace LegionRuntime {
  namespace HighLevel {


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
      assert(current_state == DELETED_STATE);
#endif
      destruction_event.trigger(Event::merge_events(recycle_events));
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
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(count);
        rez.serialize(add);
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
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(count);
        rez.serialize(add);
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
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(count);
        rez.serialize(add);
      }
      runtime->send_did_remote_resource_update(target, rez);
    }

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
      unsigned count;
      derez.deserialize(count);
      bool add;
      derez.deserialize(add);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (add)
        target->add_base_valid_ref(REMOTE_DID_REF, count);
      else if (target->remove_base_valid_ref(REMOTE_DID_REF, count))
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
      unsigned count;
      derez.deserialize(count);
      bool add;
      derez.deserialize(add);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (add)
        target->add_base_gc_ref(REMOTE_DID_REF, count);
      else if (target->remove_base_gc_ref(REMOTE_DID_REF, count))
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
      unsigned count;
      derez.deserialize(count);
      bool add;
      derez.deserialize(add);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      if (add)
        target->add_base_resource_ref(REMOTE_DID_REF, count);
      else if (target->remove_base_resource_ref(REMOTE_DID_REF, count))
        delete target;
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
            if ((gc_references > 0) || (valid_references > 0))
            {
              // Move to the pending active state
              current_state = PENDING_ACTIVE_STATE;
              need_activate = true;
            }
            else
              need_activate = false;
            need_validate = false;
            need_invalidate = false;
            need_deactivate = false;
            break;
          }
        case ACTIVE_INVALID_STATE:
          {
            // See if we have a reason to be valid
            if (valid_references > 0)
            {
              // Move to a pending valid state
              current_state = PENDING_VALID_STATE;
              need_validate = true;
              need_deactivate = false;
            }
            // See if we have a reason to be inactive
            else if (gc_references == 0)
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
        case VALID_STATE:
          {
            // See if we have a reason to be invalid
            if (valid_references == 0)
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
        case PENDING_ACTIVE_STATE:
          {
            // See if we were the ones doing the activating
            if (need_activate)
            {
              // See if we are still active
              if (valid_references > 0)
              {
                // Now we need a validate
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
              else if (gc_references > 0)
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
              if ((gc_references == 0) && (valid_references == 0))
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
              if (valid_references > 0)
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
              if (valid_references > 0)
              {
                // Now we are valid again
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
              else if (gc_references == 0)
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
      bool result = ((resource_references == 0) && (gc_references == 0) &&
              (valid_references == 0) && (current_state == INACTIVE_STATE));
      if (result)
        current_state = DELETED_STATE;
      return result;
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

