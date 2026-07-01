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

// Included from instance.h - do not include this directly

// Useful for IDEs
#include "legion/instances/instance.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    /*static*/ inline DistributedID InstanceManager::encode_instance_did(
        DistributedID did, bool external, bool reduction)
    //--------------------------------------------------------------------------
    {
      return LEGION_DISTRIBUTED_HELP_ENCODE(
          did, PHYSICAL_MANAGER_DC | (external ? EXTERNAL_CODE : 0) |
                   (reduction ? REDUCTION_CODE : 0));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_physical_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      return (
          (LEGION_DISTRIBUTED_HELP_DECODE(did) & (DIST_TYPE_LAST_DC - 1)) ==
          PHYSICAL_MANAGER_DC);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_reduction_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & (DIST_TYPE_LAST_DC - 1)) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & REDUCTION_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool InstanceManager::is_external_did(DistributedID did)
    //--------------------------------------------------------------------------
    {
      const unsigned decode = LEGION_DISTRIBUTED_HELP_DECODE(did);
      if ((decode & (DIST_TYPE_LAST_DC - 1)) != PHYSICAL_MANAGER_DC)
        return false;
      return ((decode & EXTERNAL_CODE) != 0);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_reduction_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_reduction_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
      return is_physical_did(did);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
      return (did == 0);
    }

    //--------------------------------------------------------------------------
    inline bool InstanceManager::is_external_instance(void) const
    //--------------------------------------------------------------------------
    {
      return is_external_did(did);
    }

  }  // namespace Internal
}  // namespace Legion
