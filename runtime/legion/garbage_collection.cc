/* Copyright 2023 Stanford University, NVIDIA Corporation
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
#include "legion/garbage_collection.h"
#include "legion/legion_replication.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // ImplicitReferenceTracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ImplicitReferenceTracker::~ImplicitReferenceTracker(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<IndexSpaceExpression*>::const_iterator it =
            live_expressions.begin(); it != live_expressions.end(); it++)
        if ((*it)->remove_base_expression_reference(LIVE_EXPR_REF))
          delete (*it);
    }

    /////////////////////////////////////////////////////////////
    // DistributedCollectable 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(Runtime *rt,
                                                   DistributedID id,
                                                   bool do_registration,
                                                   CollectiveMapping *mapping,
                                                   State initial_state)
      : runtime(rt), did(id), owner_space(runtime->determine_owner(did)),
        local_space(rt->address_space), collective_mapping(mapping),
        current_state(initial_state), gc_references(0),
        resource_references(0), downgrade_owner(owner_space),
        notready_owner(owner_space), sent_global_references(0),
        received_global_references(0), total_sent_references(0),
        total_received_references(0), remaining_responses(0),
        registered_with_runtime(false)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping != NULL)
      {
#ifdef DEBUG_LEGION
        assert(collective_mapping->contains(owner_space));
#endif
        collective_mapping->add_reference();
      }
      if (do_registration)
        register_with_runtime();
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(
                                              const DistributedCollectable &rhs)
      : runtime(NULL), did(0), owner_space(0), local_space(0), 
        collective_mapping(NULL)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::~DistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(gc_references == 0);
      assert(resource_references == 0);
#endif
      if ((collective_mapping != NULL) && 
          collective_mapping->remove_reference())
        delete collective_mapping;
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    } 

    //--------------------------------------------------------------------------
    template<bool NEED_LOCK>
    bool DistributedCollectable::is_global(void) const
    //--------------------------------------------------------------------------
    {
      if (NEED_LOCK)
      {
        AutoLock gc(gc_lock);
        return (current_state == VALID_REF_STATE) || 
                (current_state == GLOBAL_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE) || 
                (current_state == GLOBAL_REF_STATE);
    }

    template bool DistributedCollectable::is_global<true>(void) const;
    template bool DistributedCollectable::is_global<false>(void) const;

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
#endif
#ifdef DEBUG_LEGION_GC
      gc_references += cnt;
#else
      gc_references.fetch_add(cnt);
#endif
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
      assert(gc_references.load() >= cnt);
#endif
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
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
#endif
      resource_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
      assert(resource_references.load() >= cnt);
#endif
      if (resource_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }
#endif // not defined DEBUG_LEGION_GC 

    //--------------------------------------------------------------------------
#ifdef DEBUG_LEGION_GC
    template<typename T>
    bool DistributedCollectable::acquire_global(int cnt,
                              T source, std::map<T,int> &detailed_gc_references)
#else
    bool DistributedCollectable::acquire_global(int cnt)
#endif
    //--------------------------------------------------------------------------
    {
      AddressSpaceID current_owner;
      {
        AutoLock gc(gc_lock);
        // Check to see if we're on the downgrade owner which is the only
        // place where it is safe to perform this check
        if (downgrade_owner == local_space)
        {
          // If we're on the downgrade owner we can do the check here
          switch (current_state)
          {
            case GLOBAL_REF_STATE:
            case VALID_REF_STATE:
              {
#ifdef DEBUG_LEGION_GC
                gc_references += cnt;
                typename std::map<T,int>::iterator finder =
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
            case LOCAL_REF_STATE:
            case DELETED_REF_STATE:
              {
                return false;
              }
            default:
              assert(false);
          }
        }
        else if (!is_global<false/*need lock*/>())
          return false;
#ifdef DEBUG_LEGION_GC
        else if (gc_references > 0)
        {
          gc_references += cnt;
          typename std::map<T,int>::iterator finder =
            detailed_gc_references.find(source);
          if (finder == detailed_gc_references.end())
            detailed_gc_references[source] = cnt;
          else
            finder->second += cnt;
          return true;
        }
#endif
        current_owner = downgrade_owner;
      }
      // Send the message to the downgrade owner to try to acquire the reference
      std::atomic<bool> result(false);
      const RtUserEvent ready = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(this);
        rez.serialize(local_space);
        rez.serialize(cnt);
        rez.serialize(&result);
        rez.serialize(ready);
      }
      runtime->send_did_acquire_global_request(current_owner, rez);
      ready.wait();
      if (result.load())
      {
#ifdef DEBUG_LEGION_GC
        AutoLock gc(gc_lock);
        typename std::map<T,int>::iterator finder =
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

#ifdef DEBUG_LEGION_GC
    template bool DistributedCollectable::acquire_global<ReferenceSource>(
        int, ReferenceSource, std::map<ReferenceSource,int>&);
    template bool DistributedCollectable::acquire_global<DistributedID>(
        int, DistributedID, std::map<DistributedID,int>&);
#endif

    //--------------------------------------------------------------------------
    bool DistributedCollectable::acquire_global_remote(AddressSpaceID &current,
                                               int count, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (is_global<false/*need lock*/>())
      {
        if (downgrade_owner == local_space)
        {
          // We succeeded
          if (source == local_space)
          {
            // If we're local we can add the references now
#ifdef DEBUG_LEGION_GC
            gc_references += count;
#else
            gc_references.fetch_add(count);
#endif
          }
          else // Otherwise pack a reference to send back
            sent_global_references++;
          return true;
        }
        else
          current = downgrade_owner;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_global_acquire_request(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *remote;
      derez.deserialize(remote);
      AddressSpaceID source;
      derez.deserialize(source);
      int count;
      derez.deserialize(count);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      DistributedCollectable *dc = 
        runtime->weak_find_distributed_collectable(did);
      if (dc != NULL)
      {
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_global_remote(current_owner, count, source))
        {
          // Successfully acquired (packed) a global reference
          if (source != dc->local_space)
          {
            Serializer rez;
            {
              RezCheck z2(rez);
              rez.serialize(remote);
              rez.serialize(count);
              rez.serialize(result);
              rez.serialize(ready);
            }
            runtime->send_did_acquire_global_response(source, rez);
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
          Serializer rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(remote);
            rez.serialize(source);
            rez.serialize(count);
            rez.serialize(result);
            rez.serialize(ready);
          }
          runtime->send_did_acquire_global_request(current_owner, rez);
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
    /*static*/ void DistributedCollectable::handle_global_acquire_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedCollectable *local;
      derez.deserialize(local);
      int count;
      derez.deserialize(count);
      std::atomic<bool> *result;
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

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_gc_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
#endif
      gc_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
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
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
#endif
      gc_references++;
      std::map<DistributedID,int>::iterator finder = 
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
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
      assert(gc_references >= cnt);
#endif
      gc_references -= cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_gc_references.find(source);
      assert(finder != detailed_base_gc_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_gc_references.erase(finder);
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
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
      assert(gc_references >= cnt);
#endif
      gc_references--;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_gc_references.find(source);
      assert(finder != detailed_nested_gc_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_gc_references.erase(finder);
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
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
#endif
      resource_references++;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_resource_references.find(source);
      if (finder == detailed_base_resource_references.end())
        detailed_base_resource_references[source] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_resource_ref_internal (
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
#endif
      resource_references++;
      std::map<DistributedID,int>::iterator finder = 
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
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
      assert(resource_references >= cnt);
#endif
      resource_references--;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_resource_references.find(source);
      assert(finder != detailed_base_resource_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_resource_references.erase(finder);
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
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_REF_STATE);
      assert(resource_references >= cnt);
#endif
      resource_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_resource_references.find(source);
      assert(finder != detailed_nested_resource_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_resource_references.erase(finder);
      if (resource_references == 0)
        return can_delete(gc);
      else
        return false;
    }
#endif // DEBUG_LEGION_GC

    //--------------------------------------------------------------------------
    bool DistributedCollectable::has_remote_instance(
                                               AddressSpaceID remote_inst) const
    //--------------------------------------------------------------------------
    {
      if ((collective_mapping != NULL) && 
          collective_mapping->contains(remote_inst))
        return true;
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return remote_instances.contains(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_instances(
                                                     AddressSpaceID remote_inst)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should not be recording remote things in the collective mapping
      assert((collective_mapping == NULL) || 
              !collective_mapping->contains(remote_inst));
#endif
      AutoLock gc(gc_lock);
      // Handle a very unusual case here were we weren't able to perform the
      // deletion because there was a packed reference, but we didn't know
      // where to send it to yet
      if (is_owner() && remote_instances.empty() && 
          (collective_mapping == NULL) && 
          (sent_global_references != received_global_references))
      {
#ifdef DEBUG_LEGION
        assert(downgrade_owner == local_space);
#endif
        Serializer rez;
        rez.serialize(did);
        rez.serialize(current_state);
        runtime->send_did_downgrade_update(remote_inst, rez);
        downgrade_owner = remote_inst;
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
#ifdef DEBUG_LEGION
      assert(!registered_with_runtime);
#endif
      registered_with_runtime = true;
      if (!is_owner())
        remote_instances.add(owner_space);
      runtime->register_distributed_collectable(did, this);
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_registration(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(registered_with_runtime);
#endif
      RtUserEvent registered_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(registered_event);
      }
      runtime->send_did_remote_registration(owner_space, rez);     
      return registered_event;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_registration(
                  Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      target->update_remote_instances(source);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::pack_global_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
#endif
      sent_global_references += cnt;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unpack_global_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global<false/*need lock*/>());
#endif
      received_global_references += cnt;
    }
    
    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_delete(AutoLock &gc)
    //--------------------------------------------------------------------------
    {
      switch (current_state)
      {
        case VALID_REF_STATE:
        case GLOBAL_REF_STATE:
          {
            if (!can_downgrade())
              return false;
            // If we're not the downgrade owner then nothing for us to do
            if (downgrade_owner != local_space)
              return false;
            // We're the downgrade owner, so start the process to check to
            // see if all the nodes are ready to perform the deletion
            if (!is_owner() || !remote_instances.empty() || 
                ((collective_mapping != NULL) && 
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
          assert(false);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_downgrade(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state == GLOBAL_REF_STATE);
#endif
      return (gc_references == 0);  
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_downgrade_notifications(State downgrade)
    //--------------------------------------------------------------------------
    {
      // Ready to downgrade, send the messages and then do our local one
      if ((collective_mapping != NULL) && 
          collective_mapping->contains(local_space))
      {
        std::vector<AddressSpaceID> children;
        if (collective_mapping->contains(downgrade_owner))
          collective_mapping->get_children(downgrade_owner, local_space,
                                           children);
        else
          collective_mapping->get_children(owner_space, local_space,
                                           children);
        if (!children.empty())
        {
          Serializer rez;
          rez.serialize(did);
          rez.serialize(downgrade);
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            runtime->send_did_downgrade_success(*it, rez);
        }
      }
      if (is_owner())
      {
        if (!remote_instances.empty())
        {
          Serializer rez;
          rez.serialize(did);
          rez.serialize(downgrade);
          struct {
            void apply(AddressSpaceID space)
            { 
              if (space != owner) 
                runtime->send_did_downgrade_success(space, *rez); 
            }
            Serializer *rez;
            Runtime *runtime;
            AddressSpaceID owner;
          } downgrade_functor;
          downgrade_functor.rez = &rez;
          downgrade_functor.runtime = runtime;
          downgrade_functor.owner = downgrade_owner;
          remote_instances.map(downgrade_functor);
        }
      }
      else if ((downgrade_owner == local_space) && ((collective_mapping == NULL) 
                                 || !collective_mapping->contains(local_space)))
      {
        // If we're the owner then we have to send it to the owner_space
        // to get all the remote instances
        Serializer rez;
        rez.serialize(did);
        rez.serialize(downgrade);
        runtime->send_did_downgrade_success(owner_space, rez);
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::perform_downgrade(AutoLock &gc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(gc_references == 0);
      assert(current_state == GLOBAL_REF_STATE);
#endif
      // Downgrade the state first so that we don't duplicate the callback
      current_state = LOCAL_REF_STATE;
      // Add a resource reference here to prevent collection while we
      // release the lock to perform the callback
#ifdef DEBUG_LEGION_GC
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
#ifdef DEBUG_LEGION
      assert(resource_references > 0);
#endif
      // Remove the guard resource reference that we added before 
#ifdef DEBUG_LEGION_GC
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
#ifdef DEBUG_LEGION
      assert(remaining_responses == 0);
#endif
      // Update the downgrade owner
      downgrade_owner = owner;
      if (can_downgrade())
      {
        // We're ready to be downgraded
        // Send messages and count how many responses we expect to see
        if ((collective_mapping != NULL) && 
            collective_mapping->contains(local_space))
        {
          std::vector<AddressSpaceID> children;
          if (collective_mapping->contains(owner))
            collective_mapping->get_children(owner, local_space, children);
          else
            collective_mapping->get_children(owner_space, local_space,children);
          if (!children.empty())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(current_state);
              rez.serialize(owner);
            }
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
              runtime->send_did_downgrade_request(*it, rez);
            remaining_responses += children.size();
          }
        }
        if (is_owner())
        {
          if (!remote_instances.empty())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(current_state);
              rez.serialize(owner);
            }
            struct {
              void apply(AddressSpaceID space)
              { 
                if (space != owner) 
                  runtime->send_did_downgrade_request(space, *rez);
                else
                  skipped++;
              }
              Serializer *rez;
              Runtime *runtime;
              AddressSpaceID owner;
              unsigned skipped;
            } downgrade_functor;
            downgrade_functor.rez = &rez;
            downgrade_functor.runtime = runtime;
            downgrade_functor.owner = downgrade_owner;
            downgrade_functor.skipped = 0;
            remote_instances.map(downgrade_functor);
            remaining_responses += 
              (remote_instances.size() - downgrade_functor.skipped);
          }
        }
        else if ((owner == local_space) && ((collective_mapping == NULL) || 
                                !collective_mapping->contains(local_space)))
        {
          // If we're the owner then we have to send it to the owner_space
          // to get all the remote instances
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(current_state);
            rez.serialize(owner);
          }
          runtime->send_did_downgrade_request(owner_space, rez);
          remaining_responses++;
        }
        initialize_downgrade_state(owner);
        if (remaining_responses == 0)
        {
          // Send the response now
          if (owner != local_space)
          {
            const AddressSpaceID target = get_downgrade_target(owner);
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(owner); // owner is special bottom value
              rez.serialize(total_sent_references);
              rez.serialize(total_received_references);
            }
            runtime->send_did_downgrade_response(target, rez);
          }
#ifdef DEBUG_LEGION
          else
          {
            // We only get here if we're the owner and we don't know
            // about any remote instances yet. The only way that 
            // should happen is if we have some sent global references
            // There's nothing to do yet since we know we can't be
            // deleted yet
            assert(sent_global_references > 0);
            assert(sent_global_references != received_global_references);
          }
#endif
        }
      }
      else if (local_space != owner)
      {
        const AddressSpaceID target = get_downgrade_target(owner);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize<uint64_t>(0); // sent global references
          rez.serialize<uint64_t>(0); // received global references
        }
        runtime->send_did_downgrade_response(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::initialize_downgrade_state(AddressSpaceID own)
    //--------------------------------------------------------------------------
    {
      notready_owner = own;
      total_sent_references = sent_global_references;
      total_received_references = received_global_references;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_downgrade_request(
                   Runtime *runtime, Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      State to_check;
      derez.deserialize(to_check);
      AddressSpaceID downgrade_owner;
      derez.deserialize(downgrade_owner);

      // It's possible for this to race with the creation of this
      // distributed collectable so wait until it is ready
      DistributedCollectable *dc = 
        runtime->find_distributed_collectable(did, true/*wait*/);
      dc->process_downgrade_request(downgrade_owner, to_check);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_request(AddressSpaceID owner,
                                                           State to_check)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner != local_space); // we should be remote here
#endif
      AutoLock gc(gc_lock);
      // If the owner is asking us to downgrade a state that is less than
      // our current state then that is because the downgrade from our 
      // current state has already been done on the owner and we should
      // perform our local down grade to reflect that first
      while (current_state != to_check)
      {
#ifdef DEBUG_LEGION
        assert(to_check < current_state);
#endif
        perform_downgrade(gc);
      }
#ifdef DEBUG_LEGION
      assert(LOCAL_REF_STATE < current_state);
#endif
      check_for_downgrade(owner);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID DistributedCollectable::get_downgrade_target(
                                                     AddressSpaceID owner) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner != local_space);
#endif
      if (collective_mapping == NULL)
      {
        if (local_space == owner_space)
          return owner;
        else
          return owner_space;
      }
      if (!collective_mapping->contains(local_space))
      {
#ifdef DEBUG_LEGION
        assert(!is_owner());
#endif
        return owner_space;
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
#ifdef DEBUG_LEGION
      assert(remaining_responses > 0);
#endif
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
        if (downgrade_owner == local_space)
        {
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
              Serializer rez;
              rez.serialize(did);
              rez.serialize(current_state);
              runtime->send_did_downgrade_update(notready_owner, rez);
            }
            else
            {
              // This is a strange case: all the nodes are ready to downgrade
              // but there are still packed references in flight so we need
              // to keep trying to perform the downgrade until we find one
              // of these nodes and find the reference
              check_for_downgrade(downgrade_owner);
            }
          }
        }
        else
        {
          const AddressSpaceID target = get_downgrade_target(downgrade_owner);
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(notready_owner);
            rez.serialize(total_sent_references);
            rez.serialize(total_received_references);
          }
          runtime->send_did_downgrade_response(target, rez);
        }
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_downgrade_response(
                                          Runtime *runtime, Deserializer &derez)
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

      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
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
      if (to_downgrade == current_state)
        perform_downgrade(gc);
#ifdef DEBUG_LEGION
      else
        assert(current_state < to_downgrade);
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_downgrade_success(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      State to_downgrade;
      derez.deserialize(to_downgrade);

      // These can race with checks for downgrades from other states and
      // therefore it's possible for these to arrive even after the object
      // itself has been deleted so we need a weak find here
      DistributedCollectable *dc =
        runtime->weak_find_distributed_collectable(did);
      if (dc != NULL)
      {
        dc->process_downgrade_success(to_downgrade);
        if (dc->remove_base_resource_ref(RUNTIME_REF))
          delete dc;
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_update(AutoLock &gc,
                                                          State to_check)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(downgrade_owner != local_space);
#endif
      // It's possible we get this notification before the update saying
      // that the downgrade from the previous state has been successful
      // so make sure to update accordingly
      while (to_check != current_state)
      {
#ifdef DEBUG_LEGION
        assert(to_check < current_state);
#endif
        perform_downgrade(gc);
      }
      downgrade_owner = local_space;
      if (gc_references == 0)
        check_for_downgrade(downgrade_owner);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_downgrade_update(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      State state;
      derez.deserialize(state);

      // It's possible for this to race with the creation and registration
      // of this distributed collectable so wait for it to be ready
      DistributedCollectable *dc =
        runtime->find_distributed_collectable(did, true/*wait*/);
      AutoLock gc(dc->gc_lock);
      dc->process_downgrade_update(gc, state);
    }

    /////////////////////////////////////////////////////////////
    // ValidDistributedCollectable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::ValidDistributedCollectable(Runtime *rt,
                                                  DistributedID id, 
                                                  bool do_registration, 
                                                  CollectiveMapping *map,
                                                  bool start_in_valid_state)
      : DistributedCollectable(rt, id, do_registration, map,
          start_in_valid_state ? VALID_REF_STATE : GLOBAL_REF_STATE),
        valid_references(0), sent_valid_references(0),
        received_valid_references(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::ValidDistributedCollectable(
                                         const ValidDistributedCollectable &rhs)
      : DistributedCollectable(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::~ValidDistributedCollectable(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<bool NEED_LOCK>
    bool ValidDistributedCollectable::is_valid(void) const
    //--------------------------------------------------------------------------
    {
      if (NEED_LOCK)
      {
        AutoLock gc(gc_lock);
        return (current_state == VALID_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE);
    }

    template bool ValidDistributedCollectable::is_valid<true>(void) const;
    template bool ValidDistributedCollectable::is_valid<false>(void) const;

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
#endif
#ifdef DEBUG_LEGION_GC
      valid_references += cnt;
#else
      valid_references.fetch_add(cnt);
#endif
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
      assert(valid_references.load() >= cnt);
#endif
      if (valid_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }
#else // ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_base_valid_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
#endif
      valid_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
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
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
#endif
      valid_references += cnt;
      std::map<DistributedID,int>::iterator finder = 
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
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      assert(finder != detailed_base_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_base_valid_references.erase(finder);
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
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      assert(finder != detailed_nested_valid_references.end());
      assert(finder->second >= cnt);
      finder->second -= cnt;
      if (finder->second == 0)
        detailed_nested_valid_references.erase(finder);
      if (valid_references == 0)
        return can_delete(gc);
      else
        return false;
    }
#endif

    //--------------------------------------------------------------------------
#ifdef DEBUG_LEGION_GC
    template<typename T>
    bool ValidDistributedCollectable::acquire_valid(int cnt,
                           T source, std::map<T,int> &detailed_valid_references)
#else
    bool ValidDistributedCollectable::acquire_valid(int cnt)
#endif
    //--------------------------------------------------------------------------
    {
      AddressSpaceID current_owner;
      {
        AutoLock gc(gc_lock);
        // Check to see if we're on the downgrade owner which is the only
        // place where it is safe to perform this check
        if (downgrade_owner == local_space)
        {
          // If we're on the downgrade owner we can do the check here
          switch (current_state)
          {
            case VALID_REF_STATE:
              {
#ifdef DEBUG_LEGION_GC
                valid_references += cnt;
                typename std::map<T,int>::iterator finder =
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
            case GLOBAL_REF_STATE:
            case LOCAL_REF_STATE:
            case DELETED_REF_STATE:
              {
                return false;
              }
            default:
              assert(false);
          }
        }
        else if (!is_valid<false/*need lock*/>())
          return false;
#ifdef DEBUG_LEGION_GC
        else if (valid_references > 0)
        {
          valid_references += cnt;
          typename std::map<T,int>::iterator finder =
            detailed_valid_references.find(source);
          if (finder == detailed_valid_references.end())
            detailed_valid_references[source] = cnt;
          else
            finder->second += cnt;
          return true;
        }
#endif
        current_owner = downgrade_owner;
      }
      // Send the message to the downgrade owner to try to acquire the reference
      std::atomic<bool> result(false);
      const RtUserEvent ready = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(this);
        rez.serialize(local_space);
        rez.serialize(cnt);
        rez.serialize(&result);
        rez.serialize(ready);
      }
      runtime->send_did_acquire_valid_request(current_owner, rez);
      ready.wait();
      if (result.load())
      {
#ifdef DEBUG_LEGION_GC
        AutoLock gc(gc_lock);
        typename std::map<T,int>::iterator finder =
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

#ifdef DEBUG_LEGION_GC
    template bool ValidDistributedCollectable::acquire_valid<ReferenceSource>(
        int, ReferenceSource, std::map<ReferenceSource,int>&);
    template bool ValidDistributedCollectable::acquire_valid<DistributedID>(
        int, DistributedID, std::map<DistributedID,int>&);
#endif

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::acquire_valid_remote(
                      AddressSpaceID &current, int count, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (is_valid<false/*need lock*/>())
      {
        if (downgrade_owner == local_space)
        {
          // We succeeded
          if (source == local_space)
          {
            // If we're local we can add the references now
#ifdef DEBUG_LEGION_GC
            valid_references += count;
#else
            valid_references.fetch_add(count);
#endif
          }
          else // Otherwise pack a reference to send back
            sent_valid_references++;
          return true;
        }
        else
          current = downgrade_owner;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ValidDistributedCollectable::handle_valid_acquire_request(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      ValidDistributedCollectable *remote;
      derez.deserialize(remote);
      AddressSpaceID source;
      derez.deserialize(source);
      int count;
      derez.deserialize(count);
      std::atomic<bool> *result;
      derez.deserialize(result);
      RtUserEvent ready;
      derez.deserialize(ready);

      ValidDistributedCollectable *dc = 
        static_cast<ValidDistributedCollectable*>(
            runtime->weak_find_distributed_collectable(did));
      if (dc != NULL)
      {
        AddressSpaceID current_owner = dc->local_space;
        if (dc->acquire_valid_remote(current_owner, count, source))
        {
          if (source != dc->local_space)
          {
            // Successfully acquired (packed) a valid reference
            Serializer rez;
            {
              RezCheck z2(rez);
              rez.serialize(remote);
              rez.serialize(count);
              rez.serialize(result);
              rez.serialize(ready);
            }
            runtime->send_did_acquire_valid_response(source, rez);
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
          Serializer rez;
          {
            RezCheck z2(rez);
            rez.serialize(did);
            rez.serialize(remote);
            rez.serialize(source);
            rez.serialize(count);
            rez.serialize(result);
            rez.serialize(ready);
          }
          runtime->send_did_acquire_valid_request(current_owner, rez);
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
    /*static*/ void ValidDistributedCollectable::handle_valid_acquire_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ValidDistributedCollectable *local;
      derez.deserialize(local);
      int count;
      derez.deserialize(count);
      std::atomic<bool> *result;
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
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
#endif
      sent_valid_references += cnt;
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::unpack_valid_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid<false/*need lock*/>());
#endif
      received_valid_references += cnt;
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::can_downgrade(void) const
    //--------------------------------------------------------------------------
    {
      if (current_state == VALID_REF_STATE)
        return (valid_references == 0);
      else
        return DistributedCollectable::can_downgrade();
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::perform_downgrade(AutoLock &gc)
    //--------------------------------------------------------------------------
    {
      if (current_state == VALID_REF_STATE)
      {
#ifdef DEBUG_LEGION
        assert(valid_references == 0);
#endif
        // Send messages while holding the lock because the remote_instances
        // data structure might still be changing
        send_downgrade_notifications(VALID_REF_STATE);
        // Downgrade the state first so that we don't duplicate the callback
        current_state = GLOBAL_REF_STATE;
        // Add a gc reference here prevent downgrades from the global ref
        // state until we are done performing the callback
#ifdef DEBUG_LEGION_GC
        gc_references++;
#else
        gc_references.fetch_add(1);
#endif
        gc.release();
        notify_invalid();
        gc.reacquire();
#ifdef DEBUG_LEGION
        assert(gc_references > 0);
#endif
        // Remove the guard reference that we added before
#ifdef DEBUG_LEGION_GC
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
    void ValidDistributedCollectable::process_downgrade_update(AutoLock &gc,
                                                               State to_check)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(downgrade_owner != local_space);
#endif
      // It's possible we get this notification before the update saying
      // that the downgrade from the previous state has been successful
      // so make sure to update accordingly
      while (to_check != current_state)
      {
#ifdef DEBUG_LEGION
        assert(to_check < current_state);
#endif
        perform_downgrade(gc);
      }
      if (current_state == VALID_REF_STATE)
      {
        downgrade_owner = local_space;
        if (valid_references == 0)
          check_for_downgrade(downgrade_owner);
      }
      else
        DistributedCollectable::process_downgrade_update(gc, to_check);
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::initialize_downgrade_state(
                                                           AddressSpaceID owner)
    //--------------------------------------------------------------------------
    {
      if (current_state == VALID_REF_STATE)
      {
        notready_owner = owner;
        total_sent_references = sent_valid_references;
        total_received_references = received_valid_references;
      }
      else
        DistributedCollectable::initialize_downgrade_state(owner);
    }

  }; // namespace Internal 
}; // namespace Legion

// EOF

