/* Copyright 2014 Stanford University
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
    // CollectableState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectableState::CollectableState(void)
      : current_state(INACTIVE_STATE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CollectableState::~CollectableState(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(current_state == INACTIVE_STATE);
#endif
    }

    //--------------------------------------------------------------------------
    bool CollectableState::update_state(bool has_gc_refs,
                                        bool has_remote_refs,
                                        bool has_valid_refs,
                                        bool has_resource_refs,
                                        bool &need_activate,
                                        bool &need_validate,
                                        bool &need_invalidate,
                                        bool &need_deactivate,
                                        bool &do_delete)
    //--------------------------------------------------------------------------
    {
      switch (current_state)
      {
        case INACTIVE_STATE:
          {
            // See if we have any reason to be active
            if (has_gc_refs || has_valid_refs || has_remote_refs)
            {
              // Move to a pending active state
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
            if (has_valid_refs || has_remote_refs)
            {
              // Move to a pending valid state
              current_state = PENDING_VALID_STATE;
              need_validate = true;
              need_deactivate = false;
            }
            // See if we have a reason to be deactive
            else if (!has_gc_refs)
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
            // Se if we have a reason to be invalid
            if (!has_valid_refs && !has_remote_refs)
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
              // see if we are still active
              if (has_valid_refs || has_remote_refs)
              {
                // Now we need a validate
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
              else if (has_gc_refs)
              {
                current_state = ACTIVE_INVALID_STATE; 
                need_validate = false;
                need_deactivate = false;
              }
              else
              {
                // Not still valid, go to pending invalid
                current_state = PENDING_INACTIVE_STATE;
                need_deactivate = true;
                need_validate = false;
              }
            }
            else
            {
              // We weren't the ones doing the activate
              // so we can't help, just keep going
              need_validate = false;
              need_deactivate = false;
            }
            need_activate = false;
            need_invalidate = false;
            break;
          }
        case PENDING_INACTIVE_STATE:
          {
            // See if we were doing the inactivate
            if (need_deactivate)
            {
              // see if we are still inactive
              if (!has_gc_refs && !has_valid_refs && !has_remote_refs)
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
            // See if we were doing the validate
            if (need_validate)
            {
              // check to see if we are still valid
              if (has_valid_refs || has_remote_refs)
              {
                current_state = VALID_STATE;
                need_invalidate = false;
              }
              else
              {
                current_state = PENDING_INVALID_STATE;
                need_invalidate = false;
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
              // check to see if we are still invalid
              if (has_valid_refs || has_remote_refs)
              {
                // Now we are valid again
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
              else if (!has_gc_refs)
              {
                // No longer should be active either
                current_state = PENDING_INACTIVE_STATE;
                need_validate = false;
                need_deactivate = true;
              }
              else
              {
                // We're not active, but invalid
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
      do_delete = can_delete(has_gc_refs, has_remote_refs,
                             has_valid_refs, has_resource_refs);
      return !(need_activate || need_validate || 
               need_invalidate || need_deactivate);
    }

    //--------------------------------------------------------------------------
    bool CollectableState::can_delete(bool has_gc_refs,
                                      bool has_remote_refs,
                                      bool has_valid_refs,
                                      bool has_resource_refs)
    //--------------------------------------------------------------------------
    {
      return (current_state == INACTIVE_STATE) &&
             (!has_gc_refs) && (!has_remote_refs) &&
             (!has_valid_refs) && (!has_resource_refs);
    }

    /////////////////////////////////////////////////////////////
    // DistributedCollectable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(Runtime *rt,
                                                   DistributedID id,
                                                   AddressSpaceID own_space,
                                                   AddressSpaceID loc_space)
      : CollectableState(), runtime(rt), did(id), owner_space(own_space), 
        local_space(loc_space), owner(owner_space == local_space), 
        gc_lock(Reservation::create_reservation()), gc_references(0), 
        valid_references(0), resource_references(0), held_remote_references(0)
    //--------------------------------------------------------------------------
    {
      runtime->register_distributed_collectable(did, this);
      // We always know there is an instance on the owner node
      remote_spaces.insert(owner_space);
      // If we are not the owner node then set our resource reference
      // count to one reflecting the fact that we can be collected
      // only once the manager on the owner node is collected.
      if (!owner)
      {
        resource_references = 1;
        // Make a user event for telling our owner node when we
        // have been deleted
        destruction_event = UserEvent::create_user_event();
      }
      else
        destruction_event = UserEvent::NO_USER_EVENT; // make a no-user event
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::~DistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(gc_references == 0);
      assert(valid_references == 0);
      assert(remote_references.empty());
      assert(resource_references == 0);
#endif
      runtime->unregister_distributed_collectable(did);
      gc_lock.destroy_reservation();
      gc_lock = Reservation::NO_RESERVATION;
      // Remove references on any remote nodes
      if (owner)
      {
        for (std::set<AddressSpaceID>::const_iterator it = 
              remote_spaces.begin(); it != remote_spaces.end(); it++)
        {
          // We can skip ourselves
          if (owner_space == (*it))
            continue;
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
          }
          runtime->send_remove_distributed_resource(*it, rez);
        }
        remote_spaces.clear();
        // We can only recycle the distributed ID on the owner
        // node since the ID is the same across all the nodes.
        // We have to defer the collection of the ID until
        // after all of the remote nodes notify us that they
        // have finished collecting it.
        Event recycle_event = Event::merge_events(recycle_events);
        runtime->recycle_distributed_id(did, recycle_event);
#ifdef DEBUG_HIGH_LEVEL
        assert(!destruction_event.exists());
#endif
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(destruction_event.exists());
#endif
        // Trigger our destruction event marking that this has been collected
        destruction_event.trigger();
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          gc_references += cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (owner && !remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();  
      }
      if (result)
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
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(gc_references >= cnt);
#endif
          gc_references -= cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (owner && !remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();  
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          valid_references += cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (owner && !remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate, 
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      if (result)
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
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(valid_references >= cnt);
#endif
          valid_references -= cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (owner && !remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      resource_references += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(resource_references >= cnt);
#endif
      resource_references -= cnt;
      return can_delete((gc_references > 0),
                        (owner && !remote_references.empty()),
                        (valid_references > 0),
                        (resource_references > 0));
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::add_remote_reference(AddressSpaceID sid,
                                                      unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(owner);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          std::map<AddressSpaceID,int>::iterator finder = 
            remote_references.find(sid);
          if (finder == remote_references.end())
            remote_references[sid] = int(cnt);
          else
          {
            finder->second += cnt;
            if (finder->second == 0)
              remote_references.erase(finder);
          } 
          first = false;
        }
        done = update_state((gc_references > 0),
                            (!remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_remote_reference(AddressSpaceID sid,
                                                         Event dest_event,
                                                         unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(owner);
#endif
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          // Add the destruction event to the set of recycle events
          recycle_events.insert(dest_event);
          std::map<AddressSpaceID,int>::iterator finder = 
            remote_references.find(sid);
          if (finder == remote_references.end())
            remote_references[sid] = -(int(cnt));
          else
          {
            finder->second -= cnt;
            if (finder->second == 0)
              remote_references.erase(finder);
          }
          first = false;
        }
        done = update_state((gc_references > 0),
                            (!remote_references.empty()),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_held_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!owner);
#endif
      AutoLock gc(gc_lock);
      held_remote_references++;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::send_remote_reference(AddressSpaceID sid,
                                                       unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      // If we're sending back to the owner, then there is
      // no need to send a remote reference
      if (sid == owner_space)
        return false;
      else if (owner)
        add_remote_reference(sid, cnt);
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          // Need to send the sid since it might not be
          // the same as the sender noder
          rez.serialize(sid);
          rez.serialize(cnt);
        }
        runtime->send_add_distributed_remote(owner_space, rez);
      }
      // Mark that we know there is an instance at sid
      update_remote_spaces(sid);
      return true;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_spaces(AddressSpaceID sid)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (remote_spaces.find(sid) == remote_spaces.end())
      {
        notify_new_remote(sid);
        remote_spaces.insert(sid);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::return_held_references(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(!owner);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        // Always send back the destruction event
        rez.serialize(destruction_event);
        rez.serialize(held_remote_references);
      }
      runtime->send_remove_distributed_remote(owner_space, rez);
      // Set the references back to zero since we sent them back
      held_remote_references = 0;
    }
 
    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::process_remove_resource_reference(
                                    Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      
      DistributedCollectable *target = rt->find_distributed_collectable(did);
      if (target->remove_resource_reference())
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::process_remove_remote_reference(
              Runtime *rt, AddressSpaceID source, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      Event destruction_event;
      derez.deserialize(destruction_event);
      unsigned cnt;
      derez.deserialize(cnt);

      DistributedCollectable *target = rt->find_distributed_collectable(did);
      if (target->remove_remote_reference(source, destruction_event, cnt))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::process_add_remote_reference(
                                    Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID source;
      derez.deserialize(source);
      unsigned cnt;
      derez.deserialize(cnt);

      DistributedCollectable *target = rt->find_distributed_collectable(did);
      if (target->add_remote_reference(source, cnt))
        delete target;
    }

    /////////////////////////////////////////////////////////////
    // HierarchicalCollectable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    HierarchicalCollectable::HierarchicalCollectable(Runtime *rt, 
                                                     DistributedID d,
                                                     AddressSpaceID own_addr, 
                                                     DistributedID own_did)
      : CollectableState(), runtime(rt), did(d), 
        gc_lock(Reservation::create_reservation()), 
        gc_references(0), valid_references(0), remote_references(0), 
        resource_references(0), owner_addr(own_addr), 
        owner_did(own_did), held_remote_references(0), free_distributed_id(true)
    //--------------------------------------------------------------------------
    {
      runtime->register_hierarchical_collectable(did, this);
      // note we set resource references to 1 so a remote collectable
      // can only be deleted once its owner has been deleted
      if (owner_did != did)
      {
        resource_references = 1;
#ifdef DEBUG_HIGH_LEVEL
        assert(owner_addr != runtime->address_space);
#endif
      }
    }

    //--------------------------------------------------------------------------
    HierarchicalCollectable::~HierarchicalCollectable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(gc_references == 0);
      assert(valid_references == 0);
      assert(remote_references == 0);
      assert(resource_references == 0);
#endif
      gc_lock.destroy_reservation();
      gc_lock = Reservation::NO_RESERVATION;
      // Remove our references from any remote collectables
      if (!subscribers.empty())
      {
        for (std::map<AddressSpaceID,DistributedID>::const_iterator it = 
              subscribers.begin(); it != subscribers.end(); it++)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(it->second);
          }
          runtime->send_remove_hierarchical_resource(it->first, rez);
        }
      }
      // Free up our distributed id
      runtime->unregister_hierarchical_collectable(did);
      if (free_distributed_id)
        runtime->free_distributed_id(did);
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          gc_references += cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();  
      }
      if (result)
      {
        // If we get here it is probably a race in reference counting
        // scheme above, so mark it is as such
        assert(false);
        delete this;
      }
    }
    
    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(gc_references >= cnt);
#endif
          gc_references -= cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();  
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          valid_references += cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate, 
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      if (result)
      {
        // This probably indicates a race in reference counting algorithm
        assert(false);
        delete this;
      }
    }

    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(valid_references >= cnt);
#endif
          valid_references -= cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      resource_references += cnt;
    }

    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_resource_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(resource_references >= cnt);
#endif
      resource_references -= cnt;
      return can_delete((gc_references > 0),
                        (remote_references > 0),
                        (valid_references > 0),
                        (resource_references > 0));
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
          remote_references += cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      if (result)
      {
        // Probably indicates a race in reference counting algorithm
        assert(false);
        delete this;
      }
    }

    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate = false;
      bool need_validate = false;
      bool need_invalidate = false;
      bool need_deactivate = false;
      bool result = false;
      bool done = false;
      bool first = true;
      while (!done)
      {
        if (need_activate)
          notify_activate();
        if (need_validate)
          notify_valid();
        if (need_invalidate)
          notify_invalid();
        if (need_deactivate)
          garbage_collect();
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(remote_references >= cnt);
#endif
          remote_references -= cnt;
          first = false;
        }
        done = update_state((gc_references > 0),
                            (remote_references > 0),
                            (valid_references > 0),
                            (resource_references > 0),
                            need_activate, need_validate,
                            need_invalidate, need_deactivate, result);
        if (need_deactivate && (held_remote_references > 0))
          return_held_references();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_subscriber(AddressSpaceID target, 
                                                DistributedID subscriber)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
      assert(target != runtime->address_space);
      if (subscribers.find(target) != subscribers.end())
        assert(subscribers[target] == subscriber);
#endif
      subscribers[target] = subscriber;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_held_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      held_remote_references++;
    }

    //--------------------------------------------------------------------------
    DistributedID HierarchicalCollectable::find_distributed_id(
                                                        AddressSpaceID id) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      std::map<AddressSpaceID,DistributedID>::const_iterator finder = 
        subscribers.find(id);
#ifdef DEBUG_HIGH_LEVEL
      assert(finder != subscribers.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::set_no_free_did(void)
    //--------------------------------------------------------------------------
    {
      free_distributed_id = false;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::return_held_references(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(did != owner_did);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(owner_did);
        rez.serialize(held_remote_references);
      }
      runtime->send_remove_hierarchical_remote(owner_addr, rez);
      // Set the references back to zero since we sent them back
      held_remote_references = 0;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HierarchicalCollectable::process_remove_resource_reference(
                                    Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      HierarchicalCollectable *target = rt->find_hierarchical_collectable(did);
      if (target->remove_resource_reference())
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void HierarchicalCollectable::process_remove_remote_reference(
                                     Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      unsigned num_remote_references;
      derez.deserialize(num_remote_references);
      HierarchicalCollectable *target = rt->find_hierarchical_collectable(did);
      if (target->remove_remote_reference(num_remote_references))
        delete target;
    }

  }; // namespace HighLevel
}; // namespace LegionRuntime

// EOF

