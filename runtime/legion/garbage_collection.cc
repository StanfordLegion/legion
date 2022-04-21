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

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LocalReferenceMutator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LocalReferenceMutator::LocalReferenceMutator(
                                               const LocalReferenceMutator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LocalReferenceMutator::~LocalReferenceMutator(void)
    //--------------------------------------------------------------------------
    {
      if (!mutation_effects.empty())
      {
        RtEvent wait_on = Runtime::merge_events(mutation_effects);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    LocalReferenceMutator& LocalReferenceMutator::operator=(
                                               const LocalReferenceMutator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LocalReferenceMutator::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      mutation_effects.push_back(event);
    }

    //--------------------------------------------------------------------------
    RtEvent LocalReferenceMutator::get_done_event(void)
    //--------------------------------------------------------------------------
    {
      if (mutation_effects.empty())
        return RtEvent::NO_RT_EVENT;
      RtEvent result = Runtime::merge_events(mutation_effects);
      // Can clear this since the user caller takes responsibility for waiting
      mutation_effects.clear();
      return result;
    }

    /////////////////////////////////////////////////////////////
    // WrapperReferenceMutator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    WrapperReferenceMutator::WrapperReferenceMutator(
                                             const WrapperReferenceMutator &rhs)
      : mutation_effects(rhs.mutation_effects)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    WrapperReferenceMutator& WrapperReferenceMutator::operator=(
                                             const WrapperReferenceMutator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void WrapperReferenceMutator::record_reference_mutation_effect(RtEvent ev)
    //--------------------------------------------------------------------------
    {
      mutation_effects.insert(ev);
    }

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
                                                   AddressSpaceID own_space,
                                                   bool do_registration)
      : runtime(rt), did(id), owner_space(own_space), 
        local_space(rt->address_space), 
        current_state(INACTIVE_STATE), has_gc_references(false),
        has_valid_references(false), has_resource_references(false), 
        reentrant_update(false), gc_references(0), valid_references(0), 
        resource_references(0), registered_with_runtime(do_registration)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (did > 0)
        assert(runtime->determine_owner(did) == owner_space);
#endif
      if (do_registration)
        runtime->register_distributed_collectable(did, this);
      if (!is_owner())
      {
        remote_instances.add(owner_space);
        // Add a base resource ref that will be held until
        // the owner node removes it with an unregister message
        add_base_resource_ref(REMOTE_DID_REF);
      }
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::DistributedCollectable(
                                              const DistributedCollectable &rhs)
      : runtime(NULL), did(0), owner_space(0), local_space(0)
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
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(
                                             ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          // Wait for any state transitions to be finished
          // before we attempt to update the state
          wait_for = check_for_transition_event(reentrant);
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(in_stable_state() || reentrant);
#endif
          // See if we lost the race to update the references
          if (gc_references.fetch_add(cnt) > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            break;
          }
#ifdef DEBUG_LEGION
          assert(!has_gc_references);
#endif
          has_gc_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
#ifdef DEBUG_LEGION
      // Probably a race in the reference counting scheme above
      assert(!do_deletion);
#endif
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_gc_reference(
                                             ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(in_stable_state() || reentrant);
#endif
          const int previous = gc_references.fetch_sub(cnt);
#ifdef DEBUG_LEGION
          assert(has_gc_references);
          assert(previous >= cnt);
#endif
          if (previous == cnt)
            has_gc_references = false;
          else
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            break;
          }
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_valid_reference(
                                             ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(in_stable_state() || reentrant);
#endif
          // See if we lost the race to update the references
          if (valid_references.fetch_add(cnt) > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            break;
          }
#ifdef DEBUG_LEGION
          assert(!has_valid_references);
#endif
          has_valid_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
#ifdef DEBUG_LEGION
      // Probably a race in the reference counting scheme above
      assert(!do_deletion);
#endif
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_valid_reference(
                                             ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(in_stable_state() || reentrant);
#endif
          const int previous = valid_references.fetch_sub(cnt);
#ifdef DEBUG_LEGION
          assert(has_valid_references);
          assert(previous >= cnt);
#endif
          if (previous == cnt)
            has_valid_references = false;
          else
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            break;
          }
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }
#endif // !DEBUG_LEGION_GC

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    bool DistributedCollectable::check_valid(void)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_for;
      bool result = false;
      bool reentrant = false;
      do
      {
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock); 
        wait_for = check_for_transition_event(reentrant);
        if (wait_for.exists())
          continue; 
        assert(in_stable_state());
        result = (current_state == VALID_STATE);
        if (!reentrant)
          reentrant_event = RtEvent::NO_RT_EVENT;
        break;
      } while (true);
      return result;
    }
#endif

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
      // Need to wait until all transitions are done 
      RtEvent wait_for;
      bool reentrant = false;
      do
      {
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock); 
        wait_for = check_for_transition_event(reentrant);
        if (wait_for.exists())
          continue;
#ifdef DEBUG_LEGION
        assert(in_stable_state());
#endif
        if (current_state != VALID_STATE)
        {
          if (!reentrant)
            reentrant_event = RtEvent::NO_RT_EVENT;
          break;
        }
#ifdef LEGION_GC
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION
        assert(has_valid_references);
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
        if (!reentrant)
          reentrant_event = RtEvent::NO_RT_EVENT;
        return true;
      } while (true);
      return false;
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
      // Need to wait until all transitions are done 
      RtEvent wait_for;
      bool reentrant = false;
      do
      {
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock); 
        wait_for = check_for_transition_event(reentrant);
        if (wait_for.exists())
          continue;
#ifdef DEBUG_LEGION
        assert(in_stable_state());
#endif
        if (current_state != VALID_STATE)
        {
          if (!reentrant)
            reentrant_event = RtEvent::NO_RT_EVENT;
          break;
        }
#ifdef LEGION_GC
        log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef DEBUG_LEGION
        assert(has_valid_references);
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
        if (!reentrant)
          reentrant_event = RtEvent::NO_RT_EVENT;
        return true;
      } while (true);
      return false;
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
      // Need to wait until all transitions are done 
      RtEvent wait_for;
      bool reentrant = false;
      do
      {
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock); 
        wait_for = check_for_transition_event(reentrant);
        if (wait_for.exists())
          continue;
#ifdef DEBUG_LEGION
        assert(in_stable_state());
#endif
        if ((current_state != ACTIVE_INVALID_STATE) && 
            (current_state != VALID_STATE))
        {
          if (!reentrant)
            reentrant_event = RtEvent::NO_RT_EVENT;
          break;
        }
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
        if (gc_references.fetch_add(cnt) == 0)
          has_gc_references = true;
#endif
        if (!reentrant)
          reentrant_event = RtEvent::NO_RT_EVENT;
        return true;
      } while (true);
      return false;
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
      // Need to wait until all transitions are done 
      RtEvent wait_for;
      bool reentrant = false;
      do
      {
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock); 
        wait_for = check_for_transition_event(reentrant);
        if (wait_for.exists())
          continue;
#ifdef DEBUG_LEGION
        assert(in_stable_state());
#endif
        if ((current_state != ACTIVE_INVALID_STATE) && 
            (current_state != VALID_STATE))
        {
          if (!reentrant)
            reentrant_event = RtEvent::NO_RT_EVENT;
          break;
        }
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
        if (gc_references.fetch_add(cnt) == 0)
          has_gc_references = true;
#endif
        if (!reentrant)
          reentrant_event = RtEvent::NO_RT_EVENT;
        return true;
      } while (true);
      return false;
    }

#ifndef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      if (resource_references.fetch_add(cnt) == 0)
      {
#ifdef DEBUG_LEGION
        assert(!has_resource_references);
#endif
        has_resource_references = true;
      }
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(int cnt)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock gc(gc_lock);
        int previous = resource_references.fetch_sub(cnt);
#ifdef DEBUG_LEGION
        assert(previous >= cnt);
        assert(has_resource_references);
#endif
        if (previous == cnt)
        {
          has_resource_references = false;
          if (!can_delete())
            return false;
        }
        else
          return false;
      }
      return try_unregister();
    }
#endif

#ifdef USE_REMOTE_REFERENCES
    //--------------------------------------------------------------------------
    bool DistributedCollectable::add_create_reference(AddressSpaceID source,
           ReferenceMutator *mutator, AddressSpaceID target, ReferenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source != owner_space);
      assert(current_state != DELETED_STATE);
      assert((kind == GC_REF_KIND) || (kind == VALID_REF_KIND));
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
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
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_create_reference(AddressSpaceID source,
           ReferenceMutator *mutator, AddressSpaceID target, ReferenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source != owner_space);
      assert((kind == GC_REF_KIND) || (kind == VALID_REF_KIND));
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
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
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }
#endif // USE_REMOTE_REFERENCES

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_gc_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
          gc_references += cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_gc_references.find(source);
          if (finder == detailed_base_gc_references.end())
            detailed_base_gc_references[source] = cnt;
          else
            finder->second += cnt;
          if (gc_references > cnt)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return;
          }
#ifdef DEBUG_LEGION
          assert(!has_gc_references);
#endif
          has_gc_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      // If we get here it is probably a race in reference counting
      // scheme above, so mark it is as such
      assert(!do_deletion);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_gc_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
          gc_references++;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_gc_references.find(did);
          if (finder == detailed_nested_gc_references.end())
            detailed_nested_gc_references[did] = cnt;
          else
            finder->second += cnt;
          if (gc_references > cnt)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return;
          }
#ifdef DEBUG_LEGION
          assert(!has_gc_references);
#endif
          has_gc_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      // If we get here it is probably a race in reference counting
      // scheme above, so mark it is as such
      assert(!do_deletion);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_gc_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(gc_references >= cnt);
          assert(has_gc_references);
#endif
          gc_references -= cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_gc_references.find(source);
          assert(finder != detailed_base_gc_references.end());
          assert(finder->second >= cnt);
          finder->second -= cnt;
          if (finder->second == 0)
            detailed_base_gc_references.erase(finder);
          if (gc_references > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return false;
          }
          has_gc_references = false;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_gc_ref_internal(
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(gc_references >= cnt);
          assert(has_gc_references);
#endif
          gc_references--;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_gc_references.find(did);
          assert(finder != detailed_nested_gc_references.end());
          assert(finder->second >= cnt);
          finder->second -= cnt;
          if (finder->second == 0)
            detailed_nested_gc_references.erase(finder);
          if (gc_references > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return false;
          }
          has_gc_references = false;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
          valid_references += cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_valid_references.find(source);
          if (finder == detailed_base_valid_references.end())
            detailed_base_valid_references[source] = cnt;
          else
            finder->second += cnt;
          if (valid_references > cnt)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return;
          }
#ifdef DEBUG_LEGION
          assert(!has_valid_references);
#endif
          has_valid_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      // This probably indicates a race in reference counting algorithm
      assert(!do_deletion);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_valid_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
          valid_references += cnt;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_valid_references.find(did);
          if (finder == detailed_nested_valid_references.end())
            detailed_nested_valid_references[did] = cnt;
          else
            finder->second += cnt;
          if (valid_references > cnt)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return;
          }
#ifdef DEBUG_LEGION
          assert(!has_valid_references);
#endif
          has_valid_references = true;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      // This probably indicates a race in reference counting algorithm
      assert(!do_deletion);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(valid_references >= cnt);
          assert(has_valid_references);
#endif
          valid_references -= cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_valid_references.find(source);
          assert(finder != detailed_base_valid_references.end());
          assert(finder->second >= cnt);
          finder->second -= cnt;
          if (finder->second == 0)
            detailed_base_valid_references.erase(finder);
          if (valid_references > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return false;
          }
          has_valid_references = false;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_valid_ref_internal(
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current_state != DELETED_STATE);
#endif
      RtEvent wait_for;
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
        {
          notify_active(mutator);
          need_activate = false;
        }
        if (need_validate)
        {
          notify_valid(mutator);
          need_validate = false;
        }
        if (need_invalidate)
        {
          notify_invalid(mutator);
          need_invalidate = false;
        }
        if (need_deactivate)
        {
          notify_inactive(mutator);
          need_deactivate = false;
        }
        if (wait_for.exists() && !wait_for.has_triggered())
          wait_for.wait();
        AutoLock gc(gc_lock);
        if (first)
        {
          bool reentrant = false;
          wait_for = check_for_transition_event(reentrant); 
          if (wait_for.exists())
            continue;
#ifdef DEBUG_LEGION
          assert(valid_references >= cnt);
          assert(has_valid_references);
#endif
          valid_references -= cnt;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_valid_references.find(did);
          assert(finder != detailed_nested_valid_references.end());
          assert(finder->second >= cnt);
          finder->second -= cnt;
          if (finder->second == 0)
            detailed_nested_valid_references.erase(finder);
          if (valid_references > 0)
          {
            if (!reentrant)
              reentrant_event = RtEvent::NO_RT_EVENT;
            return false;
          }
          has_valid_references = false;
          if (reentrant)
          {
            reentrant_update = true;
            break;
          }
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      if (do_deletion)
        return try_unregister();
      return false;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_resource_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      resource_references++;
      std::map<ReferenceSource,int>::iterator finder = 
        detailed_base_resource_references.find(source);
      if (finder == detailed_base_resource_references.end())
        detailed_base_resource_references[source] = cnt;
      else
        finder->second += cnt;
      if (resource_references > cnt)
        return;
#ifdef DEBUG_LEGION
      assert(!has_resource_references);
#endif
      has_resource_references = true;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_nested_resource_ref_internal (
                                                     DistributedID did, int cnt)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      resource_references++;
      std::map<DistributedID,int>::iterator finder = 
        detailed_nested_resource_references.find(did);
      if (finder == detailed_nested_resource_references.end())
        detailed_nested_resource_references[did] = cnt;
      else
        finder->second += cnt;
      if (resource_references > cnt)
        return;
#ifdef DEBUG_LEGION
      assert(!has_resource_references);
#endif
      has_resource_references = true;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_base_resource_ref_internal(
                                                ReferenceSource source, int cnt)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
        assert(resource_references >= cnt);
        assert(has_resource_references);
#endif
        resource_references--;
        std::map<ReferenceSource,int>::iterator finder = 
          detailed_base_resource_references.find(source);
        assert(finder != detailed_base_resource_references.end());
        assert(finder->second >= cnt);
        finder->second -= cnt;
        if (finder->second == 0)
          detailed_base_resource_references.erase(finder);
        if (resource_references > 0)
          return false;
        has_resource_references = false;
        if (!can_delete())
          return false;
      }
      return try_unregister();
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_resource_ref_internal(
                                                     DistributedID did, int cnt)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
        assert(resource_references >= cnt);
        assert(has_resource_references);
#endif
        resource_references -= cnt;
        std::map<DistributedID,int>::iterator finder = 
          detailed_nested_resource_references.find(did);
        assert(finder != detailed_nested_resource_references.end());
        assert(finder->second >= cnt);
        finder->second -= cnt;
        if (finder->second == 0)
          detailed_nested_resource_references.erase(finder);
        if (resource_references > 0)
          return false;
        has_resource_references = false;
        if (!can_delete())
          return false;
      }
      return try_unregister();
    }
#endif // DEBUG_LEGION_GC

    //--------------------------------------------------------------------------
    bool DistributedCollectable::has_remote_instance(
                                               AddressSpaceID remote_inst) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock,1,false/*exclusive*/);
      return remote_instances.contains(remote_inst);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::update_remote_instances(
                                                     AddressSpaceID remote_inst)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
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
    void DistributedCollectable::register_with_runtime(
                                                      ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!registered_with_runtime);
#endif
      registered_with_runtime = true;
      runtime->register_distributed_collectable(did, this);
      if (!is_owner() && (mutator != NULL))
        send_remote_registration(mutator);
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::try_unregister(void)
    //--------------------------------------------------------------------------
    {
      if (!registered_with_runtime)
      {
        // If we were never registered with the runtime then we're done
        AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
        assert(can_delete());
#endif
        current_state = DELETED_STATE;
        return true;
      }
      else 
        return unregister_with_runtime();
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
      runtime->send_did_remote_unregister(target, rez);
    }

    //--------------------------------------------------------------------------
    DistributedCollectable::DeferRemoteUnregisterArgs::
      DeferRemoteUnregisterArgs(DistributedID id, const NodeSet &n)
      : LgTaskArgs<DeferRemoteUnregisterArgs>(implicit_provenance),
        did(id), nodes(new NodeSet(n))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_unregister_messages(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(!remote_instances.empty());
#endif
      // handle the unusual case where the derived type has some precondition
      // on sending out any unregsiter messages, see the comment on the
      // 'reentrant_event' member for why we are using it here to detect
      // this particular event precondition
      if (reentrant_event.exists() && !reentrant_event.has_triggered())
      {
        DeferRemoteUnregisterArgs args(did, remote_instances);
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, reentrant_event);
        return;
      }
      Serializer rez;
      rez.serialize(did);
      UnregisterFunctor functor(runtime, rez);
      // No need for the lock since we're being destroyed
      remote_instances.map(functor);
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
    void DistributedCollectable::send_remote_registration(
                                                      ReferenceMutator *mutator)
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
      mutator->record_reference_mutation_effect(registered_event);
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_valid_increment(
                               AddressSpaceID target, ReferenceMutator *mutator,
                               RtEvent precondition, unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif
      RtUserEvent done_event;
      if (mutator != NULL)
      {
        done_event = Runtime::create_rt_user_event();
        mutator->record_reference_mutation_effect(done_event);
      }
      const int signed_count = count;
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferRemoteReferenceUpdateArgs args(this, target, done_event,
                                            signed_count, VALID_REF_KIND);
        return runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_MESSAGE_PRIORITY, precondition);
      }
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        rez.serialize(done_event);
      }
      runtime->send_did_remote_valid_update(target, rez);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_valid_decrement(
                               AddressSpaceID target, ReferenceMutator *mutator,
                               RtEvent precondition, unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif 
      RtUserEvent done_event;
      if (mutator != NULL)
      {
        done_event = Runtime::create_rt_user_event();
        mutator->record_reference_mutation_effect(done_event);
      }
      const int signed_count = -(int(count));
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferRemoteReferenceUpdateArgs args(this, target, done_event,
                                            signed_count, VALID_REF_KIND);
        return runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, precondition);
      }
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        rez.serialize(done_event);
      }
      runtime->send_did_remote_valid_update(target, rez);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_gc_increment(
                               AddressSpaceID target, ReferenceMutator *mutator,
                               RtEvent precondition, unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif
      RtUserEvent done_event;
      if (mutator != NULL)
      {
        done_event = Runtime::create_rt_user_event();
        mutator->record_reference_mutation_effect(done_event);
      }
      const int signed_count = count;
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferRemoteReferenceUpdateArgs args(this, target, done_event,
                                            signed_count, GC_REF_KIND);
        return runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, precondition);
      }
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        rez.serialize(done_event);
      }
      runtime->send_did_remote_gc_update(target, rez);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_remote_gc_decrement(
                               AddressSpaceID target, ReferenceMutator *mutator,
                               RtEvent precondition, unsigned count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif
      RtUserEvent done_event;
      if (mutator != NULL)
      {
        done_event = Runtime::create_rt_user_event();
        mutator->record_reference_mutation_effect(done_event);
      }
      const int signed_count = -(int(count));
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferRemoteReferenceUpdateArgs args(this, target, done_event,
                                            signed_count, GC_REF_KIND);
        return runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, precondition);
      }
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        rez.serialize(done_event);
      }
      runtime->send_did_remote_gc_update(target, rez);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_resource_decrement(
                    AddressSpaceID target, RtEvent precondition, unsigned count)
    //--------------------------------------------------------------------------
    {
      const int signed_count = -(int(count));
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferRemoteReferenceUpdateArgs args(this, target,
            RtUserEvent::NO_RT_USER_EVENT, signed_count, RESOURCE_REF_KIND);
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_MESSAGE_PRIORITY, precondition);
      }
      else
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(signed_count);
          rez.serialize<bool>(target == owner_space);
        }
        runtime->send_did_remote_resource_update(target, rez);
      }
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
#ifdef DEBUG_LEGION
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
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did);
      target->update_remote_instances(source);
      Runtime::trigger_event(done_event);
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
      bool is_owner;
      derez.deserialize(is_owner);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (done_event.exists())
      {
        std::set<RtEvent> mutator_events;
        WrapperReferenceMutator mutator(mutator_events);
        if (count > 0)
          target->add_base_valid_ref(REMOTE_DID_REF, &mutator, unsigned(count));
        else if (target->remove_base_valid_ref(REMOTE_DID_REF, &mutator, 
                                               unsigned(-count)))
          delete target;
        if (!mutator_events.empty())
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(mutator_events));
        else
          Runtime::trigger_event(done_event);
      }
      else
      {
        if (count > 0)
          target->add_base_valid_ref(REMOTE_DID_REF, NULL, unsigned(count));
        else if(target->remove_base_valid_ref(REMOTE_DID_REF, NULL, 
                                              unsigned(-count)))
          delete target;
      }
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
      bool is_owner;
      derez.deserialize(is_owner);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (done_event.exists())
      {
        std::set<RtEvent> mutator_events;
        WrapperReferenceMutator mutator(mutator_events);
        if (count > 0)
          target->add_base_gc_ref(REMOTE_DID_REF, &mutator, unsigned(count));
        else if (target->remove_base_gc_ref(REMOTE_DID_REF, &mutator, 
                                            unsigned(-count)))
          delete target;
        if (!mutator_events.empty())
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(mutator_events));
        else
          Runtime::trigger_event(done_event);
      }
      else
      {
        if (count > 0)
          target->add_base_gc_ref(REMOTE_DID_REF, NULL, unsigned(count));
        else if(target->remove_base_gc_ref(REMOTE_DID_REF, NULL, 
                                           unsigned(-count)))
          delete target;
      }
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
      bool is_owner;
      derez.deserialize(is_owner);
      RtUserEvent done_event;
      derez.deserialize(done_event);
#ifdef DEBUG_LEGION
      assert(!done_event.exists());
      assert(count <= 0);
#endif
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (count > 0)
        target->add_base_resource_ref(REMOTE_DID_REF, unsigned(count));
      else if(target->remove_base_resource_ref(REMOTE_DID_REF,unsigned(-count)))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void 
      DistributedCollectable::handle_defer_remote_reference_update(
                                             Runtime *runtime, const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoteReferenceUpdateArgs *dargs = 
        (const DeferRemoteReferenceUpdateArgs*)args;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(dargs->did);
        rez.serialize(dargs->count);
        rez.serialize<bool>(dargs->owner);
        rez.serialize(dargs->done_event);
      }
      switch (dargs->kind)
      {
        case GC_REF_KIND:
          {
            runtime->send_did_remote_gc_update(dargs->target, rez);
            break;
          }
        case VALID_REF_KIND:
          {
            runtime->send_did_remote_valid_update(dargs->target, rez);
            break;
          }
        case RESOURCE_REF_KIND:
          {
            runtime->send_did_remote_resource_update(dargs->target, rez);
            break;
          }
        default:
          assert(false);
      }
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
      UnregisterFunctor functor(runtime, rez);
      dargs->nodes->map(functor);
      delete dargs->nodes;
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
#ifdef DEBUG_LEGION
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
    RtEvent DistributedCollectable::check_for_transition_event(bool &reentrant)
    //--------------------------------------------------------------------------
    {
      // If we already have a transition event then return it
      if (transition_event.exists())
      {
        // external tasks can't handle reentrant cases
        if (!Processor::get_executing_processor().exists())
          return transition_event;
        // Check for whether we are reentrant
        const RtEvent finish_event(Processor::get_current_finish_event());
        if (finish_event == reentrant_event)
        {
          reentrant = true;
          return RtEvent::NO_RT_EVENT;
        }
        else
          return transition_event;
      }
      // Otherwise check to see if we are in the middle of a transition
      if ((current_state == PENDING_ACTIVE_STATE) ||
          (current_state == PENDING_INACTIVE_STATE) ||
          (current_state == PENDING_VALID_STATE) ||
          (current_state == PENDING_INVALID_STATE) || 
          (current_state == PENDING_ACTIVE_VALID_STATE) ||
          (current_state == PENDING_INACTIVE_INVALID_STATE))
      {
        // external implicit tasks can't handle being reentrant
        if (!Processor::get_executing_processor().exists())
        {
          transition_event = Runtime::create_rt_user_event();
          return transition_event;
        }
        // Check to see if we are reentrant
        const RtEvent finish_event(Processor::get_current_finish_event());
        if (finish_event == reentrant_event)
        {
          reentrant = true;
          return RtEvent::NO_RT_EVENT;
        }
        else
          transition_event = Runtime::create_rt_user_event();
      }
      else if (Processor::get_executing_processor().exists())
      {
#ifdef DEBUG_LEGION
        assert(!reentrant_event.exists());
#endif
        // Save the reentrant event since we're going to iterate
        reentrant_event = RtEvent(Processor::get_current_finish_event());
      }
      return transition_event;
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
#ifdef DEBUG_LEGION
            assert(!reentrant_update);
#endif
            // See if we have any reason to be active
#ifdef USE_REMOTE_REFERENCES
            if (has_valid_references || (!create_valid_refs.empty()))
#else
            if (has_valid_references)
#endif
            {
              current_state = PENDING_ACTIVE_VALID_STATE;
              need_activate = true;
              need_validate = true;
            }
#ifdef USE_REMOTE_REFERENCES
            else if (has_gc_references || !create_gc_refs.empty())
#else
            else if (has_gc_references)
#endif
            {
              current_state = PENDING_ACTIVE_STATE;
              need_activate = true;
            }
            break;
          }
        case ACTIVE_INVALID_STATE:
          {
#ifdef DEBUG_LEGION
            assert(!reentrant_update);
#endif
            // See if we have a reason to be valid
#ifdef USE_REMOTE_REFERENCES
            if (has_valid_references || !create_valid_refs.empty())
#else
            if (has_valid_references)
#endif
            {
              // Move to a pending valid state
              current_state = PENDING_VALID_STATE;
              need_validate = true;
            }
            // See if we have a reason to be inactive
#ifdef USE_REMOTE_REFERENCES
            else if (!has_gc_references  && create_gc_refs.empty())
#else
            else if (!has_gc_references)
#endif
            {
              current_state = PENDING_INACTIVE_STATE;
              need_deactivate = true;
            }
            break;
          }
        case VALID_STATE:
          {
#ifdef DEBUG_LEGION
            assert(!reentrant_update);
#endif
            // See if we have a reason to be invalid
#ifdef USE_REMOTE_REFERENCES
            if (!has_valid_references && create_valid_refs.empty())
#else
            if (!has_valid_references)
#endif
            {
              need_invalidate = true;
#ifdef USE_REMOTE_REFERENCES
              if (!has_gc_references && create_gc_refs.empty())
#else
              if (!has_gc_references)
#endif
              {
                current_state = PENDING_INACTIVE_INVALID_STATE;
                need_deactivate = true;
              }
              else 
                current_state = PENDING_INVALID_STATE;
            }
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
            current_state = ACTIVE_INVALID_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef DEBUG_LEGION
#ifdef USE_REMOTE_REFERENCES
            assert(!has_valid_references && create_valid_refs.empty());
            assert(has_gc_references || !create_gc_refs.empty());
#else
            assert(!has_valid_references);
            assert(has_gc_references);
#endif
#endif
            break;
          }
        case PENDING_INACTIVE_STATE:
          {
            current_state = INACTIVE_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef DEBUG_LEGION
#ifdef USE_REMOTE_REFERENCES
            assert(!has_valid_references && create_valid_refs.empty());
            assert(!has_gc_references && create_gc_refs.empty());
#else
            assert(!has_valid_references);
            assert(!has_gc_references);
#endif
#endif
            break;
          }
        case PENDING_VALID_STATE:
          {
            current_state = VALID_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef DEBUG_LEGION
#ifdef USE_REMOTE_REFERENCES
            assert(has_valid_references || create_valid_refs.empty());
#else
            assert(has_valid_references);
#endif
#endif
            break;
          }
        case PENDING_INVALID_STATE:
          {
            current_state = ACTIVE_INVALID_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef DEBUG_LEGION
#ifdef USE_REMOTE_REFERENCES
            assert(!has_valid_references && create_valid_refs.empty());
            assert(has_gc_references || !create_gc_refs.empty());
#else
            assert(!has_valid_references);
            assert(has_gc_references);
#endif
#endif
            break;
          }
        case PENDING_ACTIVE_VALID_STATE:
          {
            current_state = VALID_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef USE_REMOTE_REFERENCES
            assert(has_valid_references || !create_valid_refs.empty());
#else
            assert(has_valid_references);
#endif
            break;
          }
        case PENDING_INACTIVE_INVALID_STATE:
          {
            current_state = INACTIVE_STATE;
            if (reentrant_update)
            {
              reentrant_update = false;
              return update_state(need_activate, need_validate,
                  need_invalidate, need_deactivate, do_deletion); 
            }
#ifdef DEBUG_LEGION
#ifdef USE_REMOTE_REFERENCES
            assert(!has_valid_references && !create_valid_refs.empty());
            assert(!has_gc_references && !create_gc_refs.empty());
#else
            assert(!has_valid_references);
            assert(!has_gc_references);
#endif
#endif
            break;
          }
        default:
          assert(false); // should never get here
      }
      const bool done = !(need_activate || need_validate || 
                          need_invalidate || need_deactivate);
      if (done)
      {
#ifdef DEBUG_LEGION
        assert(in_stable_state());
#endif
        do_deletion = can_delete();
        reentrant_event = RtEvent::NO_RT_EVENT;
        if (transition_event.exists())
        {
          Runtime::trigger_event(transition_event);
          transition_event = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
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

