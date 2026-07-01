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

#ifndef __LEGION_INSTANCE_SET_H__
#define __LEGION_INSTANCE_SET_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/api/types.h"
#include "legion/utilities/bitmask.h"

namespace Legion {
  namespace Internal {

    /**
     * \class InstanceRef
     * A class for keeping track of references to physical instances
     */
    class InstanceRef : public Heapify<InstanceRef, OPERATION_LIFETIME> {
    public:
      InstanceRef(bool composite = false);
      InstanceRef(const InstanceRef& rhs);
      InstanceRef(InstanceManager* manager, const FieldMask& valid_fields);
      ~InstanceRef(void);
    public:
      InstanceRef& operator=(const InstanceRef& rhs);
    public:
      bool operator==(const InstanceRef& rhs) const;
      bool operator!=(const InstanceRef& rhs) const;
    public:
      inline bool has_ref(void) const { return (manager != nullptr); }
      inline InstanceManager* get_manager(void) const { return manager; }
      inline const FieldMask& get_valid_fields(void) const
      {
        return valid_fields;
      }
      inline void update_fields(const FieldMask& update)
      {
        valid_fields |= update;
      }
    public:
      inline bool is_local(void) const { return local; }
      MappingInstance get_mapping_instance(void) const;
      bool is_virtual_ref(void) const;
    public:
      void add_resource_reference(ReferenceSource source) const;
      void remove_resource_reference(ReferenceSource source) const;
      bool acquire_valid_reference(ReferenceSource source) const;
      void add_valid_reference(ReferenceSource source) const;
      void remove_valid_reference(ReferenceSource source) const;
    public:
      Memory get_memory(void) const;
      PhysicalManager* get_physical_manager(void) const;
    public:
      bool is_field_set(FieldID fid) const;
    public:
      void pack_reference(Serializer& rez) const;
      void unpack_reference(Deserializer& derez, RtEvent& ready);
    protected:
      FieldMask valid_fields;
      InstanceManager* manager;
      bool local;
    };

    /**
     * \class InstanceSet
     * This class is an abstraction for representing one or more
     * instance references. It is designed to be light-weight and
     * easy to copy by value. It maintains an internal copy-on-write
     * data structure to avoid unnecessary premature copies.
     */
    class InstanceSet {
    public:
      struct CollectableRef : public Collectable,
                              public InstanceRef {
      public:
        CollectableRef(void) : Collectable(), InstanceRef() { }
        CollectableRef(const InstanceRef& ref) : Collectable(), InstanceRef(ref)
        { }
        CollectableRef(const CollectableRef& rhs)
          : Collectable(), InstanceRef(rhs)
        { }
        ~CollectableRef(void) { }
      public:
        CollectableRef& operator=(const CollectableRef& rhs);
      };
      struct InternalSet : public Collectable {
      public:
        InternalSet(size_t size = 0)
        {
          if (size > 0)
            vector.resize(size);
        }
        InternalSet(const InternalSet& rhs) : vector(rhs.vector) { }
        ~InternalSet(void) { }
      public:
        InternalSet& operator=(const InternalSet& rhs) = delete;
      public:
        inline bool empty(void) const { return vector.empty(); }
      public:
        op::vector<InstanceRef> vector;
      };
    public:
      InstanceSet(size_t init_size = 0);
      InstanceSet(const InstanceSet& rhs);
      ~InstanceSet(void);
    public:
      InstanceSet& operator=(const InstanceSet& rhs);
      bool operator==(const InstanceSet& rhs) const;
      bool operator!=(const InstanceSet& rhs) const;
    public:
      InstanceRef& operator[](unsigned idx);
      const InstanceRef& operator[](unsigned idx) const;
    public:
      bool empty(void) const;
      size_t size(void) const;
      void resize(size_t new_size);
      void clear(void);
      void swap(InstanceSet& rhs);
      void add_instance(const InstanceRef& ref);
      bool is_virtual_mapping(void) const;
    public:
      void pack_references(Serializer& rez) const;
      void unpack_references(
          Deserializer& derez, std::set<RtEvent>& ready_events);
    public:
      void add_resource_references(ReferenceSource source) const;
      void remove_resource_references(ReferenceSource source) const;
      bool acquire_valid_references(ReferenceSource source) const;
      void add_valid_references(ReferenceSource source) const;
      void remove_valid_references(ReferenceSource source) const;
    protected:
      void make_copy(void);
    protected:
      union {
        CollectableRef* single;
        InternalSet* multi;
      } refs;
      bool single;
      mutable bool shared;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INSTANCE_SET_H__
