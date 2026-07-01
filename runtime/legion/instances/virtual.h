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

#ifndef __LEGION_VIRTUAL_INSTANCE_H__
#define __LEGION_VIRTUAL_INSTANCE_H__

#include "legion/instances/instance.h"

namespace Legion {
  namespace Internal {

    /**
     * \class VirtualManager
     * This is a singleton class of which there will be exactly one
     * on every node in the machine. The virtual manager class will
     * represent all the virtual instances.
     */
    class VirtualManager : public InstanceManager,
                           public Heapify<VirtualManager, RUNTIME_LIFETIME> {
    public:
      VirtualManager(
          DistributedID did, LayoutDescription* layout,
          CollectiveMapping* mapping);
      VirtualManager(const VirtualManager& rhs) = delete;
      virtual ~VirtualManager(void);
    public:
      VirtualManager& operator=(const VirtualManager& rhs) = delete;
    public:
      virtual void notify_local(void) override { }
    public:
      virtual PointerConstraint get_pointer_constraint(void) const override;
      virtual void send_manager(AddressSpaceID target);
    };

    //--------------------------------------------------------------------------
    inline VirtualManager* InstanceManager::as_virtual_manager(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_virtual_manager());
      return static_cast<VirtualManager*>(const_cast<InstanceManager*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_VIRTUAL_INSTANCE_H__
