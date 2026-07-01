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

#include "legion/instances/virtual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Virtual Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VirtualManager::VirtualManager(
        DistributedID did, LayoutDescription* desc, CollectiveMapping* mapping)
      : InstanceManager(
            did, desc, nullptr /*field space node*/,
            nullptr /*index space expression*/, 0 /*tree id*/,
            true /*register now*/, mapping)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_GC
      log_garbage.info(
          "GC Virtual Manager %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    VirtualManager::~VirtualManager(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointerConstraint VirtualManager::get_pointer_constraint(void) const
    //--------------------------------------------------------------------------
    {
      return PointerConstraint(Memory::NO_MEMORY, 0);
    }

    //--------------------------------------------------------------------------
    void VirtualManager::send_manager(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

  }  // namespace Internal
}  // namespace Legion
