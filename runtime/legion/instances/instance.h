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

#ifndef __LEGION_INSTANCE_H__
#define __LEGION_INSTANCE_H__

#include "legion/instances/layout.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class InstanceManager
     * This is the abstract base class for all instances of a physical
     * resource manager for memory.
     */
    class InstanceManager : public DistributedCollectable {
    public:
      enum {
        EXTERNAL_CODE = 0x20,
        REDUCTION_CODE = 0x40,
      };
    public:
      InstanceManager(
          DistributedID did, LayoutDescription* layout, FieldSpaceNode* node,
          IndexSpaceExpression* domain, RegionTreeID tree_id, bool register_now,
          CollectiveMapping* mapping = nullptr);
      virtual ~InstanceManager(void);
    public:
      virtual PointerConstraint get_pointer_constraint(void) const = 0;
    public:
      inline bool is_reduction_manager(void) const;
      inline bool is_physical_manager(void) const;
      inline bool is_virtual_manager(void) const;
      inline bool is_external_instance(void) const;
      inline PhysicalManager* as_physical_manager(void) const;
      inline VirtualManager* as_virtual_manager(void) const;
    public:
      static inline DistributedID encode_instance_did(
          DistributedID did, bool external, bool reduction);
      static inline bool is_physical_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_external_did(DistributedID did);
    public:
      // Interface to the mapper for layouts
      inline void get_fields(std::set<FieldID>& fields) const
      {
        if (layout != nullptr)
          layout->get_fields(fields);
      }
      inline bool has_field(FieldID fid) const
      {
        if (layout != nullptr)
          return layout->has_field(fid);
        return false;
      }
      inline void has_fields(std::map<FieldID, bool>& fields) const
      {
        if (layout != nullptr)
          layout->has_fields(fields);
        else
          for (std::pair<const FieldID, bool>& field_pair : fields)
            field_pair.second = false;
      }
      inline void remove_space_fields(std::set<FieldID>& fields) const
      {
        if (layout != nullptr)
          layout->remove_space_fields(fields);
        else
          fields.clear();
      }
    public:
      bool entails(
          LayoutConstraints* constraints,
          const LayoutConstraint** failed_constraint) const;
      bool entails(
          const LayoutConstraintSet& constraints,
          const LayoutConstraint** failed_constraint) const;
      bool conflicts(
          LayoutConstraints* constraints,
          const LayoutConstraint** conflict_constraint) const;
      bool conflicts(
          const LayoutConstraintSet& constraints,
          const LayoutConstraint** conflict_constraint) const;
    public:
      LayoutDescription* const layout;
      FieldSpaceNode* const field_space_node;
      IndexSpaceExpression* const instance_domain;
      const RegionTreeID tree_id;
    };

    /**
     * A small interface for subscribing to notifications for
     * when an instance is deleted
     */
    class InstanceDeletionSubscriber {
    public:
      virtual ~InstanceDeletionSubscriber(void) { }
      virtual void notify_instance_deletion(PhysicalManager* manager) = 0;
      virtual void add_subscriber_reference(PhysicalManager* manager) = 0;
      virtual bool remove_subscriber_reference(PhysicalManager* manager) = 0;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/instances/instance.inl"

#endif  // __LEGION_INSTANCE_H__
