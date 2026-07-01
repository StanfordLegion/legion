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

// Included from garbage_collection.h - do not include this directly

// Useful for IDEs
#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    /*static*/ inline void ImplicitReferenceTracker::record_live_expression(
        IndexSpaceExpression* expr)
    //--------------------------------------------------------------------------
    {
      // We should always be inside of a meta-task or an API call for this
      // so there should be no need to create an implicit reference tracker
      // on the fly in this case
      legion_assert(implicit_reference_tracker != nullptr);
      implicit_reference_tracker->live_expressions.emplace_back(expr);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void ImplicitReferenceTracker::record_invalid_operation(
        IndexSpaceOperation* op)
    //--------------------------------------------------------------------------
    {
      // Should always be inside a Legion/Realm task for this
      // If we're not we might not check at the end of the task
      // to clean up these references so this avoids leaking
      legion_assert(implicit_reference_tracker != nullptr);
      implicit_reference_tracker->invalid_operations.emplace_back(op);
    }

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_base_ref(
        ReferenceKind kind, DistributedID did, AddressSpaceID local_space,
        ReferenceSource src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      if (ADD)
        log_garbage.info(
            "GC Add Base Ref %d %lld %d %d %d", kind, did, local_space, src,
            cnt);
      else
        log_garbage.info(
            "GC Remove Base Ref %d %lld %d %d %d", kind, did, local_space, src,
            cnt);
    }

    //--------------------------------------------------------------------------
    template<bool ADD>
    static inline void log_nested_ref(
        ReferenceKind kind, DistributedID did, AddressSpaceID local_space,
        DistributedID src, unsigned cnt)
    //--------------------------------------------------------------------------
    {
      did = LEGION_DISTRIBUTED_ID_FILTER(did);
      src = LEGION_DISTRIBUTED_ID_FILTER(src);
      if (ADD)
        log_garbage.info(
            "GC Add Nested Ref %d %lld %d %lld %d", kind, did, local_space, src,
            cnt);
      else
        log_garbage.info(
            "GC Remove Nested Ref %d %lld %d %lld %d", kind, did, local_space,
            src, cnt);
    }

    //--------------------------------------------------------------------------
    inline void Collectable::add_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      references.fetch_add(cnt);
    }

    //--------------------------------------------------------------------------
    inline bool Collectable::remove_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      unsigned prev = references.fetch_sub(cnt);
      legion_assert(prev >= cnt);  // check for underflow
      // If previous is equal to count, the value is now
      // zero so it is safe to reclaim this object
      return (prev == cnt);
    }

    //--------------------------------------------------------------------------
    inline bool Collectable::check_add_reference(unsigned cnt /*= 1*/)
    //--------------------------------------------------------------------------
    {
      unsigned current = references.load();
      while (current > 0)
      {
        unsigned next = current + cnt;
        if (references.compare_exchange_weak(current, next))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::has_remote_instances(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock gc(gc_lock, false /*exclusive*/);
      return !remote_instances.empty();
    }

    //--------------------------------------------------------------------------
    template<typename FUNCTOR>
    void DistributedCollectable::map_over_remote_instances(FUNCTOR& functor)
    //--------------------------------------------------------------------------
    {
      // We can't iterate the remote_instances data structure while holding
      // the lock since the functor might call back in here so we need to
      // synchronize with updates to the remote instances separately
      // INVARIANT: the functor must never (transitively) call
      // update_remote_instances on this object. update_remote_instances waits
      // for remote_iterators to drain to zero, but this iteration keeps the
      // count > 0 until the functor returns, so such a call self-deadlocks.
      // Calling back in to pack references (pack_global_ref, etc.) is fine.
      //
      // Track the in-flight iteration with an RAII guard so the count can
      // never leak -- a leaked count would permanently wedge
      // update_remote_instances -- if the functor or map() exits non-locally.
      struct IterationGuard {
        IterationGuard(DistributedCollectable* d) : dc(d)
        {
          AutoLock gc(dc->gc_lock, false /*exclusive*/);
          dc->remote_iterators++;
        }
        ~IterationGuard(void)
        {
          if (--dc->remote_iterators == 0)
            dc->finalize_remote_iterators();
        }
        DistributedCollectable* const dc;
      } guard(this);
      remote_instances.map(functor);
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_gc_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_base_gc_ref_internal(source, cnt);
#else
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return;
      }
      add_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_gc_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_nested_gc_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = gc_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return;
      }
      add_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_gc_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_base_gc_ref_internal(source, cnt);
#else
      int current = gc_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_gc_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<false>(GC_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_nested_gc_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = gc_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (gc_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_gc_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_base_resource_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<true>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_base_resource_ref_internal(source, cnt);
#else
      int current = resource_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return;
      }
      add_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void DistributedCollectable::add_nested_resource_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<true>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = resource_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return;
      }
      add_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_base_resource_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<false>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_base_resource_ref_internal(source, cnt);
#else
      int current = resource_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::remove_nested_resource_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<false>(RESOURCE_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_nested_resource_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = resource_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (resource_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_resource_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::has_gc_reference(void) const
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG_GC
      AutoLock gc(gc_lock, false /*exclusive*/);
#endif
      return (gc_references > 0);
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::check_global_and_increment(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt > 0);
#ifndef LEGION_DEBUG_GC
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
      bool result = acquire_global(cnt);
#else
      bool result = acquire_global(cnt, source, detailed_base_gc_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool DistributedCollectable::check_global_and_increment(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt > 0);
#ifndef LEGION_DEBUG_GC
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
      bool result = acquire_global(cnt);
#else
      bool result = acquire_global(
          cnt, LEGION_DISTRIBUTED_ID_FILTER(source),
          detailed_nested_gc_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(GC_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline void ValidDistributedCollectable::add_base_valid_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline void ValidDistributedCollectable::add_nested_valid_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      add_nested_valid_ref_internal(LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
      while (current > 0)
      {
        int next = current + cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return;
      }
      add_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::remove_base_valid_ref(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_base_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_base_valid_ref_internal(source, cnt);
#else
      int current = valid_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::remove_nested_valid_ref(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt >= 0);
#ifdef LEGION_GC
      log_nested_ref<false>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
#ifdef LEGION_DEBUG_GC
      return remove_nested_valid_ref_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source), cnt);
#else
      int current = valid_references.load();
      legion_assert(current >= cnt);
      while (current > cnt)
      {
        int next = current - cnt;
        if (valid_references.compare_exchange_weak(current, next))
          return false;
      }
      return remove_valid_reference(cnt);
#endif
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::has_valid_reference(void) const
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG_GC
      AutoLock gc(gc_lock, false /*exclusive*/);
#endif
      return (valid_references > 0);
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::check_valid_and_increment(
        ReferenceSource source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt > 0);
#ifndef LEGION_DEBUG_GC
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
      bool result = acquire_valid(cnt);
#else
      bool result = acquire_valid(cnt, source, detailed_base_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool ValidDistributedCollectable::check_valid_and_increment(
        DistributedID source, int cnt /*=1*/)
    //--------------------------------------------------------------------------
    {
      legion_assert(cnt > 0);
#ifndef LEGION_DEBUG_GC
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
      bool result = acquire_valid(cnt);
#else
      bool result = acquire_valid(
          cnt, LEGION_DISTRIBUTED_ID_FILTER(source),
          detailed_nested_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, cnt);
#endif
      return result;
    }

  }  // namespace Internal
}  // namespace Legion
