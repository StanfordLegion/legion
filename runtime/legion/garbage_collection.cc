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
        collective_mapping->add_reference();
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
    bool DistributedCollectable::is_global(bool need_lock) const
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock gc(gc_lock);
        return (current_state == VALID_REF_STATE) || 
                (current_state == GLOBAL_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE) || 
                (current_state == GLOBAL_REF_STATE);
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
#endif
      gc_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
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
    bool DistributedCollectable::check_active_and_increment(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifndef DEBUG_LEGION_GC
      // Check to see if we can do the add without the lock first
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
#endif
      AutoLock gc(gc_lock);
      if (!is_global(false/*need lock*/))
        return false;
#ifdef LEGION_GC
      log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      gc_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_gc_references.find(source);
      if (finder == detailed_base_gc_references.end())
        detailed_base_gc_references[source] = cnt;
      else
        finder->second += cnt;
#else
      gc_references.fetch_add(cnt);
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::check_active_and_increment(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifndef DEBUG_LEGION_GC
      // Check to see if we can do the add without the lock first
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
#endif
      AutoLock gc(gc_lock);
      if (!is_global(false/*need lock*/))
        return false;
#ifdef LEGION_GC
      log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      gc_references += cnt;
      source = LEGION_DISTRIBUTED_ID_FILTER(source);
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_gc_references.find(source);
      if (finder == detailed_nested_gc_references.end())
        detailed_nested_gc_references[source] = cnt;
      else
        finder->second += cnt;
#else
      gc_references.fetch_add(cnt);
#endif
      return true;
    }

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_gc_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
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
      assert(is_global(false/*need lock*/));
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
      assert(is_global(false/*need lock*/));
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
      assert(is_global(false/*need lock*/));
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
                                     AddressSpaceID remote_inst, bool need_lock)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should not be recording remote things in the collective mapping
      assert((collective_mapping == NULL) || 
              !collective_mapping->contains(remote_inst));
#endif
      if (need_lock)
      {
        AutoLock gc(gc_lock);
        remote_instances.add(remote_inst);
      }
      else
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
      {
        remote_instances.add(owner_space);
        // Add a base resource ref that will be held until
        // the owner node removes it with an unregister message
        add_base_resource_ref(REMOTE_DID_REF);
      }
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
      assert(is_global(false/*need lock*/));
#endif
      sent_global_references += cnt;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unpack_global_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
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
                (collective_mapping != NULL))
            {
              // Send messages to see if we can perform the deletion
              check_for_downgrade(downgrade_owner, false/*need lock*/);
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
    void DistributedCollectable::send_downgrade_notifications(void)
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
          for (std::vector<AddressSpaceID>::const_iterator it =
                children.begin(); it != children.end(); it++)
            runtime->send_did_downgrade_success(*it, rez);
        }
      }
      if (is_owner() && !remote_instances.empty())
      {
        Serializer rez;
        rez.serialize(did);
        struct {
          void apply(AddressSpaceID space)
            { runtime->send_did_downgrade_success(space, *rez); }
          Serializer *rez;
          Runtime *runtime;
        } downgrade_functor;
        downgrade_functor.rez = &rez;
        downgrade_functor.runtime = runtime;
        remote_instances.map(downgrade_functor);
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::perform_downgrade(AutoLock &gc)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
      send_downgrade_notifications();
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
    void DistributedCollectable::check_for_downgrade(AddressSpaceID owner,
                                                     bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock gc(gc_lock);
        check_for_downgrade(owner, false/*need lock*/);
        return;
      }
      // Update the downgrade owner
      downgrade_owner = owner;
      if (can_downgrade())
      {
        // We're ready to be downgraded
        // Send messages and count how many responses we expect to see
        remaining_responses = 0;
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
              rez.serialize(owner);
            }
            for (std::vector<AddressSpaceID>::const_iterator it =
                  children.begin(); it != children.end(); it++)
              runtime->send_did_downgrade_request(*it, rez);
            remaining_responses += children.size();
          }
        }
        if (is_owner() && !remote_instances.empty())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(owner);
          }
          struct {
            void apply(AddressSpaceID space)
              { runtime->send_did_downgrade_request(space, *rez); }
            Serializer *rez;
            Runtime *runtime;
          } downgrade_functor;
          downgrade_functor.rez = &rez;
          downgrade_functor.runtime = runtime;
          remote_instances.map(downgrade_functor);
          remaining_responses += remote_instances.size();
        }
        initialize_downgrade_state(owner);
        if (remaining_responses == 0)
        {
          // Send the response now
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
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(local_space != owner);
#endif
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
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID downgrade_owner;
      derez.deserialize(downgrade_owner);

      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
      dc->check_for_downgrade(downgrade_owner);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID DistributedCollectable::get_downgrade_target(
                                                     AddressSpaceID owner) const
    //--------------------------------------------------------------------------
    {
      if (collective_mapping == NULL)
        return owner_space;
      if (!collective_mapping->contains(local_space))
        return owner_space;
      if (!collective_mapping->contains(owner))
        return collective_mapping->get_parent(owner_space, local_space);
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
          if ((notready_owner == downgrade_owner) && 
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
              runtime->send_did_downgrade_update(notready_owner, rez);
            }
            else
            {
              // This is a strange case: all the nodes are ready to downgrade
              // but there are still packed references in flight so we need
              // to keep trying to perform the downgrade until we find one
              // of these nodes and find the reference
              check_for_downgrade(downgrade_owner, false/*need lock*/);
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
    bool DistributedCollectable::process_downgrade_success(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      return perform_downgrade(gc);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_downgrade_success(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);

      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
      if (dc->process_downgrade_success())
        delete dc;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::process_downgrade_update(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(downgrade_owner != local_space);
#endif
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

      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
      AutoLock gc(dc->gc_lock);
      dc->process_downgrade_update();
    }

    /////////////////////////////////////////////////////////////
    // ValidDistributedCollectable
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ValidDistributedCollectable::ValidDistributedCollectable(Runtime *rt,
                 DistributedID id, bool do_registration, CollectiveMapping *map)
      : DistributedCollectable(rt, id, do_registration, map, VALID_REF_STATE),
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
    bool ValidDistributedCollectable::is_valid(bool need_lock) const
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock gc(gc_lock);
        return (current_state == VALID_REF_STATE);
      }
      else
        return (current_state == VALID_REF_STATE);
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
#endif
      valid_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::remove_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
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
      assert(is_valid(false/*need lock*/));
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
      assert(is_valid(false/*need lock*/));
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
      assert(is_valid(false/*need lock*/));
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
      assert(is_valid(false/*need lock*/));
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
    bool ValidDistributedCollectable::check_valid_and_increment(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifndef DEBUG_LEGION_GC
      // Check to see if we can do the add without the lock first
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
#endif
      AutoLock gc(gc_lock);
      if (!is_valid(false/*need lock*/))
        return false;
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      valid_references += cnt;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_valid_references.find(source);
      if (finder == detailed_base_valid_references.end())
        detailed_base_valid_references[source] = cnt;
      else
        finder->second += cnt;
#else
      valid_references.fetch_add(cnt);
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    bool ValidDistributedCollectable::check_valid_and_increment(
                                                  DistributedID source, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(cnt >= 0);
#endif
#ifndef DEBUG_LEGION_GC
      // Check to see if we can do the add without the lock first
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
        {
#ifdef LEGION_GC
          log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
          return true;
        }
      }
#endif
      AutoLock gc(gc_lock);
      if (!is_valid(false/*need lock*/))
        return false;
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION_GC
      valid_references += cnt;
      source = LEGION_DISTRIBUTED_ID_FILTER(source);
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(source);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[source] = cnt;
      else
        finder->second += cnt;
#else
      valid_references.fetch_add(cnt);
#endif
      return true;
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::pack_valid_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
#endif
      sent_valid_references += cnt;
    }

    //--------------------------------------------------------------------------
    void ValidDistributedCollectable::unpack_valid_ref(unsigned cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
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
        // Send messages while holding the lock because the remote_instances
        // data structure might still be changing
        send_downgrade_notifications();
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
    void ValidDistributedCollectable::process_downgrade_update(void)
    //--------------------------------------------------------------------------
    {
      if (current_state == VALID_REF_STATE)
      {
#ifdef DEBUG_LEGION
        assert(downgrade_owner != local_space);
#endif
        downgrade_owner = local_space;
        if (valid_references == 0)
          check_for_downgrade(downgrade_owner);
      }
      else
        DistributedCollectable::process_downgrade_update();
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

