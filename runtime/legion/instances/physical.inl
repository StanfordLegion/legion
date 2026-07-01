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

// Included from physical.h - do not include this directly

// Useful for IDEs
#include "legion/instances/physical.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    inline PhysicalManager* InstanceManager::as_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_physical_manager());
      return static_cast<PhysicalManager*>(const_cast<InstanceManager*>(this));
    }

    //--------------------------------------------------------------------------
    inline void PhysicalManager::add_base_valid_ref(
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
    inline void PhysicalManager::add_nested_valid_ref(
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
    inline bool PhysicalManager::acquire_instance(ReferenceSource source)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DEBUG_GC
      // Note that we cannot do this for external instances as they might
      // have been detached while still holding valid references so they
      // have to go through the full path every time
      if (!is_external_instance())
      {
        // Check to see if we can do the add without the lock first
        int current = valid_references.load();
        while (current > 0)
        {
          int next = current + 1;
          if (valid_references.compare_exchange_weak(current, next))
          {
#ifdef LEGION_GC
            log_base_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
            return true;
          }
        }
      }
      bool result = acquire_internal();
#else
      bool result = acquire_internal(source, detailed_base_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_base_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::acquire_instance(DistributedID source)
    //--------------------------------------------------------------------------
    {
#ifndef LEGION_DEBUG_GC
      // Note that we cannot do this for external instances as they might
      // have been detached while still holding valid references so they
      // have to go through the full path every time
      if (!is_external_instance())
      {
        // Check to see if we can do the add without the lock first
        int current = valid_references.load();
        while (current > 0)
        {
          int next = current + 1;
          if (valid_references.compare_exchange_weak(current, next))
          {
#ifdef LEGION_GC
            log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
            return true;
          }
        }
      }
      bool result = acquire_internal();
#else
      bool result = acquire_internal(
          LEGION_DISTRIBUTED_ID_FILTER(source),
          detailed_nested_valid_references);
#endif
#ifdef LEGION_GC
      if (result)
        log_nested_ref<true>(VALID_REF_KIND, did, local_space, source, 1);
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    inline bool PhysicalManager::remove_base_valid_ref(
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
    inline bool PhysicalManager::remove_nested_valid_ref(
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

  }  // namespace Internal
}  // namespace Legion
