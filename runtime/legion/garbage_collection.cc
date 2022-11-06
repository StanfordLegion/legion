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
                                                   State starting_state,
                                                   bool do_registration,
                                                   CollectiveMapping *mapping)
      : runtime(rt), did(id), owner_space(runtime->determine_owner(did)),
        local_space(rt->address_space), collective_mapping(mapping),
        current_state(starting_state), gc_references(0), valid_references(0), 
        resource_references(0), registered_with_runtime(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should be either global or second state after initialization
      assert(is_global(false/*need lock*/));
#endif
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
      assert(valid_references == 0);
      assert(resource_references == 0);
#endif
      if ((collective_mapping != NULL) && 
          collective_mapping->remove_reference())
        delete collective_mapping;
#ifdef LEGION_GC
      if (registered_with_runtime)
        log_garbage.info("GC Deletion %lld %d", 
            LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::is_valid(bool need_lock) const
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
    void DistributedCollectable::add_valid_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
#endif
      valid_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_valid_reference(int cnt)
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

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(state != DELETED_STATE);
#endif
      resource_references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(state != DELETED_STATE);
      assert(resource_references.load() >= cnt);
#endif
      if (resource_references.fetch_sub(cnt) == cnt)
        return can_delete(gc);
      else
        return false;
    }
#endif // not defined DEBUG_LEGION_GC

    //--------------------------------------------------------------------------
    bool DistributedCollectable::check_valid_and_increment(
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
    bool DistributedCollectable::check_valid_and_increment(
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
      if (gc_references == 0)
        has_gc_references = true;
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
      if (gc_references == 0)
        has_gc_references = true;
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
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
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
    void DistributedCollectable::add_nested_gc_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
#endif
      gc_references++;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_gc_references.find(did);
      if (finder == detailed_nested_gc_references.end())
        detailed_nested_gc_references[did] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_gc_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
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
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_global(false/*need lock*/));
      assert(gc_references >= cnt);
#endif
      gc_references--;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_gc_references.find(did);
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
    void DistributedCollectable::add_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
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
    void DistributedCollectable::add_nested_valid_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
#endif
      valid_references += cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(did);
      if (finder == detailed_nested_valid_references.end())
        detailed_nested_valid_references[did] = cnt;
      else
        finder->second += cnt;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
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
    bool DistributedCollectable::remove_nested_valid_ref_internal(
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(is_valid(false/*need lock*/));
      assert(valid_references >= cnt);
#endif
      valid_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_valid_references.find(did);
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

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_resource_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
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
                                                     DistributedID did, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      resource_references++;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_resource_references.find(did);
      if (finder == detailed_nested_resource_references.end())
        detailed_nested_resource_references[did] = cnt;
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
      assert(current_state != DELETED_STATE);
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
                                                     DistributedID did, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
      assert(resource_references >= cnt);
#endif
      resource_references -= cnt;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_resource_references.find(did);
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
    bool DistributedCollectable::unregister_with_runtime(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(registered_with_runtime);
#endif
      if (runtime->unregister_distributed_collectable(did))
      {
        // Only need to send these messages from the owner node
        if (is_owner() && !remote_instances.empty())
          send_unregister_messages();
        if ((collective_mapping != NULL) &&
            (collective_mapping->contains(local_space)))
          send_unregister_mapping();
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::confirm_deletion(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (!can_delete())
        return false;
      current_state = DELETED_STATE;
      return true;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::UnregisterFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      const RtEvent precondition = dc->find_unregister_precondition(target);
      if (precondition.exists() && ! precondition.has_triggered())
      {
        DistributedCollectable::DeferRemoteUnregisterArgs args(dc->did, target);
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, precondition); 
      }
      else
      {
        Serializer rez;
        rez.serialize(dc->did);
        runtime->send_did_remote_unregister(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_unregister_messages(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(!remote_instances.empty() || (collective_mapping != NULL));
#endif
      UnregisterFunctor functor(runtime, this);
      // No need for the lock since we're being destroyed
      remote_instances.map(functor);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_unregister_mapping(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
      assert(collective_mapping->contains(local_space));
#endif
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(owner_space, local_space, children);
      if (children.empty())
        return;
      Serializer rez;
      rez.serialize(did);
      for (std::vector<AddressSpaceID>::const_iterator it =
            children.begin(); it != children.end(); it++)
        runtime->send_did_remote_unregister(*it, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_unregister_collectable(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
      // Now remove the resource reference we were holding
      if (dc->remove_base_resource_ref(REMOTE_DID_REF))
        delete dc;
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
    /*static*/ void DistributedCollectable::handle_defer_remote_unregister(
                                             Runtime *runtime, const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoteUnregisterArgs *dargs = 
        (const DeferRemoteUnregisterArgs*)args;
      Serializer rez;
      rez.serialize(dargs->did);
      runtime->send_did_remote_unregister(dargs->target, rez);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::can_delete(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the lock
#ifdef USE_REMOTE_REFERENCES
      bool result = (!has_resource_references && !has_gc_references &&
              !has_valid_references && create_gc_refs.empty() && 
              create_valid_refs.empty() && 
              ((current_state == INACTIVE_STATE) || 
               (current_state == DELETED_STATE)));
#else
      bool result = (!has_resource_references && !has_gc_references &&
              !has_valid_references && 
              ((current_state == INACTIVE_STATE) || 
               (current_state == DELETED_STATE)));
#endif
      return result;
    }

  }; // namespace Internal 
}; // namespace Legion

// EOF

