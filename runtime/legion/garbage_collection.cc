/* Copyright 2018 Stanford University, NVIDIA Corporation
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
        wait_on.lg_wait();
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
      mutation_effects.insert(event);
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
        gc_references(0), valid_references(0), resource_references(0), 
        registered_with_runtime(do_registration)
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
      if (is_owner() && registered_with_runtime)
        unregister_with_runtime();
#ifdef LEGION_GC
      log_garbage.info("GC Deletion %lld %d", 
          LEGION_DISTRIBUTED_ID_FILTER(did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_gc_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_LEGION
          // Should have at least one reference here
          assert(__sync_fetch_and_add(&gc_references, 0) > 0);
#endif
          has_gc_references = true;
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
    bool DistributedCollectable::remove_gc_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          // Check to see if we lost the race for changing state
          if (has_gc_references && 
              (__sync_fetch_and_add(&gc_references, 0) == 0))
            has_gc_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_valid_reference(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
#ifdef DEBUG_LEGION
          // Should have at least one reference here
          assert(__sync_fetch_and_add(&valid_references, 0) > 0);
#endif
          has_valid_references = true;
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
    bool DistributedCollectable::remove_valid_reference(
                                                      ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          // Check to see if we lost the race for changing state
          if (has_valid_references &&
              (__sync_fetch_and_add(&valid_references, 0) == 0))
            has_valid_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::try_add_valid_reference(
                  ReferenceMutator *mutator, bool must_be_valid, int cnt /*=1*/)
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
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
          // Still have to use an atomic here
          unsigned previous = __sync_fetch_and_add(&valid_references, cnt);
          if (previous > 0)
            return true;
          has_valid_references = true;
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
      // We can only do this from four states
      // The fifth state is a little weird in that we have to continue
      // polling until it is safely out of the valid state
      // Note this also prevents duplicate deletions
      bool result = false;
      while (true)
      {
        AutoLock gc(gc_lock);
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
        // If we're in PENDING_INVALID_STATE go to the deleted
        // version and then keep going around the loop again until
        // we end up in the ACTIVE_DELETED_STATE
        if (current_state == PENDING_INVALID_STATE)
        { 
          current_state = PENDING_INVALID_DELETED_STATE;
          result = true;
        }
        else if (current_state != PENDING_INVALID_DELETED_STATE)
          break;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_resource_reference(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(!has_resource_references);
#endif
      has_resource_references = true;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_resource_reference(void)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
#ifdef DEBUG_LEGION
      assert(has_resource_references);
#endif
      has_resource_references = false;
      return can_delete();
    }

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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
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
           ReferenceMutator *mutator, AddressSpaceID target, ReferenceKind kind)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(source != owner_space);
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
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

#ifdef DEBUG_LEGION_GC
    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_gc_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          gc_references += cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_gc_references.find(source);
          if (finder == detailed_base_gc_references.end())
            detailed_base_gc_references[source] = cnt;
          else
            finder->second += cnt;
          if (gc_references > cnt)
            return;
#ifdef DEBUG_LEGION
          assert(!has_gc_references);
#endif
          has_gc_references = true;
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
    void DistributedCollectable::add_nested_gc_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          gc_references++;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_gc_references.find(did);
          if (finder == detailed_nested_gc_references.end())
            detailed_nested_gc_references[did] = cnt;
          else
            finder->second += cnt;
          if (gc_references > cnt)
            return;
#ifdef DEBUG_LEGION
          assert(!has_gc_references);
#endif
          has_gc_references = true;
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
    bool DistributedCollectable::remove_base_gc_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
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
            return false;
          has_gc_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_gc_ref_internal(
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
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
            return false;
          has_gc_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::add_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          valid_references += cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_valid_references.find(source);
          if (finder == detailed_base_valid_references.end())
            detailed_base_valid_references[source] = cnt;
          else
            finder->second += cnt;
          if (valid_references > cnt)
            return;
#ifdef DEBUG_LEGION
          assert(!has_valid_references);
#endif
          has_valid_references = true;
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
    void DistributedCollectable::add_nested_valid_ref_internal (
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
          valid_references += cnt;
          std::map<DistributedID,int>::iterator finder = 
            detailed_nested_valid_references.find(did);
          if (finder == detailed_nested_valid_references.end())
            detailed_nested_valid_references[did] = cnt;
          else
            finder->second += cnt;
          if (valid_references > cnt)
            return;
#ifdef DEBUG_LEGION
          assert(!has_valid_references);
#endif
          has_valid_references = true;
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
    bool DistributedCollectable::remove_base_valid_ref_internal(
                     ReferenceSource source, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
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
            return false;
          has_valid_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_valid_ref_internal(
                          DistributedID did, ReferenceMutator *mutator, int cnt)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
        AutoLock gc(gc_lock);
        if (first)
        {
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
            return false;
          has_valid_references = false;
          first = false;
        }
        done = update_state(need_activate, need_validate,
                            need_invalidate, need_deactivate, do_deletion);
      }
      return do_deletion;
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::try_add_valid_reference_internal(
                            ReferenceSource source, ReferenceMutator *mutator, 
                            bool must_be_valid, int cnt)
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
          notify_active(mutator);
        if (need_validate)
          notify_valid(mutator);
        if (need_invalidate)
          notify_invalid(mutator);
        if (need_deactivate)
          notify_inactive(mutator);
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
          unsigned previous = valid_references;
          valid_references += cnt;
          std::map<ReferenceSource,int>::iterator finder = 
            detailed_base_valid_references.find(source);
          if (finder == detailed_base_valid_references.end())
            detailed_base_valid_references[source] = cnt;
          else
            finder->second += cnt;
          if (previous > 0)
            return true;
          has_valid_references = true;
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
      return can_delete();
    }

    //--------------------------------------------------------------------------
    bool DistributedCollectable::remove_nested_resource_ref_internal(
                                                     DistributedID did, int cnt)
    //--------------------------------------------------------------------------
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
      return can_delete();
    }
#endif // DEBUG_LEGION_GC

    //--------------------------------------------------------------------------
    void DistributedCollectable::notify_remote_inactive(ReferenceMutator *m)
    //--------------------------------------------------------------------------
    {
      // Should only called for classes that override this method
      assert(false);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::notify_remote_invalid(ReferenceMutator *m)
    //--------------------------------------------------------------------------
    {
      // Should only called for classes that override this method
      assert(false);
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
    void DistributedCollectable::update_remote_instances(
                                                     AddressSpaceID remote_inst)
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock);
      remote_instances.add(remote_inst);
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
    void DistributedCollectable::unregister_with_runtime(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(registered_with_runtime);
#endif
      runtime->unregister_distributed_collectable(did);
      if (!remote_instances.empty())
        runtime->recycle_distributed_id(did, 
                     send_unregister_messages(REFERENCE_VIRTUAL_CHANNEL));
      else
        runtime->recycle_distributed_id(did, RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::UnregisterFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RtUserEvent done_event = Runtime::create_rt_user_event();
      Serializer rez;
      rez.serialize(did);
      rez.serialize(done_event); 
      runtime->send_did_remote_unregister(target, rez, vc);
      done_events.insert(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent DistributedCollectable::send_unregister_messages(
                                                    VirtualChannelKind vc) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      assert(!remote_instances.empty());
#endif
      std::set<RtEvent> done_events;
      UnregisterFunctor functor(runtime, did, vc, done_events); 
      // No need for the lock since we're being destroyed
      remote_instances.map(functor);
      return Runtime::merge_events(done_events);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::unregister_collectable(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_owner());
      assert(registered_with_runtime);
#endif
      runtime->unregister_distributed_collectable(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_unregister_collectable(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
      dc->unregister_collectable();
      Runtime::trigger_event(done_event);
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
    void DistributedCollectable::send_remote_valid_update(AddressSpaceID target,
                            ReferenceMutator *mutator, unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif
      int signed_count = count;
      RtUserEvent done_event = RtUserEvent::NO_RT_USER_EVENT;
      if (!add)
        signed_count = -signed_count;
      else
        done_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        if (add)
          rez.serialize(done_event);
      }
      runtime->send_did_remote_valid_update(target, rez);
      if (add && (mutator != NULL))
        mutator->record_reference_mutation_effect(done_event);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_gc_update(AddressSpaceID target,
                            ReferenceMutator *mutator, unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
      assert(registered_with_runtime);
#endif
      int signed_count = count;
      RtUserEvent done_event = RtUserEvent::NO_RT_USER_EVENT;
      if (!add)
        signed_count = -signed_count;
      else
        done_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(signed_count);
        rez.serialize<bool>(target == owner_space);
        if (add)
          rez.serialize(done_event);
      }
      runtime->send_did_remote_gc_update(target, rez);
      if (add && (mutator != NULL))
        mutator->record_reference_mutation_effect(done_event);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_resource_update(
                                AddressSpaceID target, unsigned count, bool add)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count != 0);
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
        rez.serialize<bool>(target == owner_space);
      }
      runtime->send_did_remote_resource_update(target, rez);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_invalidate(AddressSpaceID target,
                                                      ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      if (mutator == NULL)
      {
        rez.serialize(did);
        rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
      }
      else
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        rez.serialize(did);
        rez.serialize(done);
        mutator->record_reference_mutation_effect(done);
      }
      runtime->send_did_remote_invalidate(target, rez);
    }

    //--------------------------------------------------------------------------
    void DistributedCollectable::send_remote_deactivate(AddressSpaceID target,
                                                      ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
#endif
      Serializer rez;
      if (mutator == NULL)
      {
        rez.serialize(did);
        rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
      }
      else
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        rez.serialize(did);
        rez.serialize(done);
        mutator->record_reference_mutation_effect(done);
      }
      runtime->send_did_remote_deactivate(target, rez);
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
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.lg_wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (count > 0)
      {
        std::set<RtEvent> mutator_events;
        WrapperReferenceMutator mutator(mutator_events);
        target->add_base_valid_ref(REMOTE_DID_REF, &mutator, unsigned(count));
        RtUserEvent done_event;
        derez.deserialize(done_event);
        if (!mutator_events.empty())
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(mutator_events));
        else
          Runtime::trigger_event(done_event);
      }
      else if (target->remove_base_valid_ref(REMOTE_DID_REF, NULL,
                                             unsigned(-count)))
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
      bool is_owner;
      derez.deserialize(is_owner);
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.lg_wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (count > 0)
      {
        std::set<RtEvent> mutator_events;
        WrapperReferenceMutator mutator(mutator_events);
        target->add_base_gc_ref(REMOTE_DID_REF, &mutator, unsigned(count));
        RtUserEvent done_event;
        derez.deserialize(done_event);
        if (!mutator_events.empty())
          Runtime::trigger_event(done_event,
              Runtime::merge_events(mutator_events));
        else
          Runtime::trigger_event(done_event);
      }
      else if (target->remove_base_gc_ref(REMOTE_DID_REF, NULL,
                                          unsigned(-count)))
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
      bool is_owner;
      derez.deserialize(is_owner);
      DistributedCollectable *target = NULL;
      if (!is_owner)
      {
        RtEvent ready;
        target = runtime->find_distributed_collectable(did, ready);
        if (ready.exists() && !ready.has_triggered())
          ready.lg_wait();
      }
      else
        target = runtime->find_distributed_collectable(did);
      if (count > 0)
        target->add_base_resource_ref(REMOTE_DID_REF, unsigned(count));
      else if (target->remove_base_resource_ref(REMOTE_DID_REF, 
                                                unsigned(-count)))
        delete target;
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_invalidate(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done;
      derez.deserialize(done);
      // We know we are not the owner so we might have to wait 
      RtEvent ready;
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did, ready);
      if (ready.exists() && !ready.has_triggered())
        ready.lg_wait();
      if (done.exists())
      {
        std::set<RtEvent> preconditions;
        WrapperReferenceMutator mutator(preconditions);
        target->notify_remote_invalid(&mutator);
        if (!preconditions.empty())
          Runtime::trigger_event(done, Runtime::merge_events(preconditions));
        else
          Runtime::trigger_event(done);
      }
      else
        target->notify_remote_invalid(NULL);
    }

    //--------------------------------------------------------------------------
    /*static*/ void DistributedCollectable::handle_did_remote_deactivate(
                                          Runtime *runtime, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent done;
      derez.deserialize(done);
      // We know we are not the owner so we might have to wait 
      RtEvent ready;
      DistributedCollectable *target = 
        runtime->find_distributed_collectable(did, ready);
      if (ready.exists() && !ready.has_triggered())
        ready.lg_wait();
      if (done.exists())
      {
        std::set<RtEvent> preconditions;
        WrapperReferenceMutator mutator(preconditions);
        target->notify_remote_inactive(&mutator);
        if (!preconditions.empty())
          Runtime::trigger_event(done, Runtime::merge_events(preconditions));
        else
          Runtime::trigger_event(done);
      }
      else
        target->notify_remote_inactive(NULL);
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
            if (has_valid_references || (!create_valid_refs.empty()))
#else
            if (has_valid_references)
#endif
            {
              current_state = PENDING_ACTIVE_VALID_STATE;
              need_activate = true;
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
            need_validate = false;
            need_invalidate = false;
            need_deactivate = false;
            break;
          }
        case ACTIVE_INVALID_STATE:
          {
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
              need_deactivate = false;
            }
            // See if we have a reason to be inactive
#ifdef USE_REMOTE_REFERENCES
            else if (!has_gc_references  && create_gc_refs.empty())
#else
            else if (!has_gc_references)
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
            if (has_valid_references || !create_valid_refs.empty())
#else
            if (has_valid_references)
#endif           
              assert(false);
            // See if we have a reason to move towards deletion
#ifdef USE_REMOTE_REFERENCES
            else if (!has_gc_references && create_gc_refs.empty())
#else
            else if (!has_gc_references)
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
            if (!has_valid_references && create_valid_refs.empty())
#else
            if (!has_valid_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // Now we need a validate
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if (has_gc_references || !create_gc_refs.empty())
#else
              else if (has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                current_state = PENDING_ACTIVE_VALID_STATE;
                need_activate = true;
              }
#ifdef USE_REMOTE_REFERENCES
              else if (!has_gc_references && create_gc_refs.empty())
#else
              else if (!has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // Now we are valid again
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if (!has_gc_references && create_gc_refs.empty())
#else
              else if (!has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // Still going to valid
                current_state = PENDING_VALID_STATE;
                need_validate = true;
                need_deactivate = false;
              }
#ifdef USE_REMOTE_REFERENCES
              else if (!has_gc_references && create_gc_refs.empty())
#else
              else if (!has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // This is really bad if it happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if (has_gc_references || !create_gc_refs.empty())
#else
              else if (has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // Really bad if this happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if (!has_gc_references  && create_gc_refs.empty())
#else
              else if (!has_gc_references)
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
              if (has_valid_references || !create_valid_refs.empty())
#else
              if (has_valid_references)
#endif
              {
                // This is really bad if it happens
                assert(false);
              }
#ifdef USE_REMOTE_REFERENCES
              else if (!has_gc_references && create_gc_refs.empty())
#else
              else if (!has_gc_references)
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

