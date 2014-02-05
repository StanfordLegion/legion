/* Copyright 2013 Stanford University
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
                                                   AddressSpaceID loc_space)
      : runtime(rt), did(id), owner_space(own_space), local_space(loc_space),
        owner(owner_space == local_space), 
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
        resource_references = 1;
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::~DistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
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
        // Can only do this on the owner node since it
        // is the same across all the nodes
        runtime->free_distributed_id(did);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate;
      {
        AutoLock gc(gc_lock);
        need_activate = (gc_references == 0) && (valid_references == 0) &&
                        (remote_references.empty());
        gc_references += cnt;
      }
      if (need_activate)
        notify_activate();
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool result;
      bool need_gc;
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(gc_references >= cnt);
#endif
        gc_references -= cnt;
        need_gc = (gc_references == 0) && (valid_references == 0) &&
                  (remote_references.empty());
        result = need_gc && (resource_references == 0);
        // If we're garbage collecting and we have held remote
        // references then send them back to the owner
        if (need_gc && (held_remote_references > 0))
          return_held_references();
      }
      if (need_gc)
        garbage_collect();
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate;
      bool need_validate;
      {
        AutoLock gc(gc_lock);
        need_validate = (valid_references == 0);
        need_activate = need_validate && (gc_references == 0) &&
                        (remote_references.empty());
        valid_references += cnt;
      }
      if (need_activate)
        notify_activate();
      if (need_validate)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool result;
      bool need_invalidate;
      bool need_gc;
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(valid_references >= cnt);
#endif
        valid_references -= cnt;
        need_invalidate = (valid_references == 0);
        need_gc = need_invalidate && (gc_references == 0) &&
                  (remote_references.empty());
        result = need_gc && (resource_references == 0);
        if (need_gc && (held_remote_references > 0))
          return_held_references();
      }
      if (need_invalidate)
        notify_invalid();
      if (need_gc)
        garbage_collect();
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
      return ((gc_references == 0) && (remote_references.empty()) && 
              (resource_references == 0));
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::add_remote_reference(AddressSpaceID sid,
                                                      unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(owner);
#endif
      bool need_activate;
      bool need_gc;
      bool result;
      {
        AutoLock gc(gc_lock);
        need_activate = (gc_references == 0) && (remote_references.empty());
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
        need_gc = (gc_references == 0) && (remote_references.empty());
        result = ((gc_references == 0) && (remote_references.empty()) && 
                  (resource_references == 0));
        // finally update the list of nodes we know about
        if (remote_spaces.find(sid) == remote_spaces.end())
        {
          notify_new_remote(sid);
          remote_spaces.insert(sid);
        }
      }
      if (need_activate)
        notify_activate();
      if (need_gc)
        garbage_collect();
      return result;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_remote_reference(AddressSpaceID sid,
                                                         unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(owner);
#endif
      bool need_activate;
      bool need_gc;
      bool result;
      {
        AutoLock gc(gc_lock);
        need_activate = (gc_references == 0) && (remote_references.empty());
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
        need_gc = (gc_references == 0) && (remote_references.empty());
        result = ((gc_references == 0) && (remote_references.empty()) && 
                  (resource_references == 0));
        // update the list of nodes we know about
        if (remote_spaces.find(sid) == remote_spaces.end())
        {
          notify_new_remote(sid);
          remote_spaces.insert(sid);
        }
      }
      if (need_activate)
        notify_activate();
      if (need_gc)
        garbage_collect();
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
      unsigned cnt;
      derez.deserialize(cnt);

      DistributedCollectable *target = rt->find_distributed_collectable(did);
      if (target->remove_remote_reference(source, cnt))
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
      : runtime(rt), did(d), gc_lock(Reservation::create_reservation()), 
        gc_references(0), valid_references(0), remote_references(0), 
        resource_references(0), owner_addr(own_addr), 
        owner_did(own_did), held_remote_references(0)
    //--------------------------------------------------------------------------
    {
      runtime->register_hierarchical_collectable(did, this);
      // note we set resource references to 1 so a remote collectable
      // can only be deleted once its owner has been deleted
      if (owner_did != did)
        resource_references = 1;
    }

    //--------------------------------------------------------------------------
    HierarchicalCollectable::~HierarchicalCollectable(void)
    //--------------------------------------------------------------------------
    {
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
      runtime->free_distributed_id(did);
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate;
      {
        AutoLock gc(gc_lock);
        need_activate = (gc_references == 0) && (valid_references == 0) &&
                        (remote_references == 0);
        gc_references += cnt;
      }
      if (need_activate)
        notify_activate();
    }
    
    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_gc_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool result;
      bool need_gc;
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(gc_references >= cnt);
#endif
        gc_references -= cnt;
        need_gc = (gc_references == 0) && (remote_references == 0) &&
                  (valid_references == 0);
        result = need_gc && (resource_references == 0);
        // If we're garbage collecting and we have held remote
        // references then send them back to the owner
        if (need_gc && (held_remote_references > 0))
          return_held_references();
      }
      if (need_gc)
        garbage_collect();
      return result;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate;
      bool need_validate;
      {
        AutoLock gc(gc_lock);
        need_validate = (valid_references == 0);
        need_activate = need_validate && (gc_references == 0) &&
                        (remote_references == 0);
        valid_references += cnt;
      }
      if (need_activate)
        notify_activate();
      if (need_validate)
        notify_valid();
    }

    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_valid_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool result;
      bool need_invalidate;
      bool need_gc;
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(valid_references >= cnt);
#endif
        valid_references -= cnt;
        need_invalidate = (valid_references == 0);
        need_gc = need_invalidate && (gc_references == 0) &&
                  (remote_references == 0);
        result = need_gc && (resource_references == 0);
        // If we're garbage collecting and we have held remote
        // references then send them back to the owner
        if (need_gc && (held_remote_references > 0))
          return_held_references();
      }
      if (need_invalidate)
        notify_invalid();
      if (need_gc)
        garbage_collect();
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
      return ((gc_references == 0) && (remote_references == 0) && 
              (resource_references == 0) && (valid_references == 0));
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool need_activate;
      {
        AutoLock gc(gc_lock);
        need_activate = (gc_references == 0) && (remote_references == 0) &&
                        (valid_references == 0);
        remote_references += cnt;
      }
      if (need_activate)
        notify_activate();
    }

    //--------------------------------------------------------------------------
    bool HierarchicalCollectable::remove_remote_reference(unsigned cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      bool result;
      bool need_gc;
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
        assert(remote_references >= cnt);
#endif
        remote_references -= cnt;
        need_gc = (gc_references == 0) && (remote_references == 0) &&
                  (valid_references == 0);
        result = need_gc && (resource_references == 0);
        // No need to send back remote references since we are the owner
      }
      if (need_gc)
        garbage_collect();
      return result;
    }

    //--------------------------------------------------------------------------
    void HierarchicalCollectable::add_subscriber(AddressSpaceID target, 
                                                DistributedID subscriber)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_HIGH_LEVEL
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

