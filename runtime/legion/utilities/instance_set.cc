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

#include "legion/utilities/instance_set.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/api/mapping.h"
#include "legion/nodes/field.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // InstanceRef
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(bool comp) : manager(nullptr), local(true)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(const InstanceRef& rhs)
      : valid_fields(rhs.valid_fields), manager(rhs.manager), local(rhs.local)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(InstanceManager* man, const FieldMask& m)
      : valid_fields(m), manager(man), local(true)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InstanceRef::~InstanceRef(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceRef::operator=(const InstanceRef& rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      local = rhs.local;
      manager = rhs.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator==(const InstanceRef& rhs) const
    //--------------------------------------------------------------------------
    {
      if (valid_fields != rhs.valid_fields)
        return false;
      if (manager != rhs.manager)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator!=(const InstanceRef& rhs) const
    //--------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //--------------------------------------------------------------------------
    MappingInstance InstanceRef::get_mapping_instance(void) const
    //--------------------------------------------------------------------------
    {
      return MappingInstance(manager);
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_virtual_ref(void) const
    //--------------------------------------------------------------------------
    {
      if (manager == nullptr)
        return true;
      return manager->is_virtual_manager();
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_resource_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      manager->add_base_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_resource_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      if (manager->remove_base_resource_ref(source))
        delete manager;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::acquire_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      return manager->as_physical_manager()->acquire_instance(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      manager->as_physical_manager()->add_base_valid_ref(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      if (manager->as_physical_manager()->remove_base_valid_ref(source))
        delete manager;
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      if (!manager->is_physical_manager())
        return Memory::NO_MEMORY;
      return manager->as_physical_manager()->get_memory();
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceRef::get_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      legion_assert(manager->is_physical_manager());
      return manager->as_physical_manager();
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_field_set(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      legion_assert(manager != nullptr);
      unsigned index = manager->field_space_node->get_field_index(fid);
      return valid_fields.is_set(index);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(valid_fields);
      if (manager != nullptr)
        rez.serialize(manager->did);
      else
        rez.serialize<DistributedID>(0);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::unpack_reference(Deserializer& derez, RtEvent& ready)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid_fields);
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      manager = runtime->find_or_request_instance_manager(did, ready);
      local = false;
    }

    /////////////////////////////////////////////////////////////
    // InstanceSet
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceSet::CollectableRef& InstanceSet::CollectableRef::operator=(
        const InstanceSet::CollectableRef& rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      local = rhs.local;
      manager = rhs.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(size_t init_size /*=0*/)
      : single((init_size <= 1)), shared(false)
    //--------------------------------------------------------------------------
    {
      if (init_size == 0)
        refs.single = nullptr;
      else if (init_size == 1)
      {
        refs.single = new CollectableRef();
        refs.single->add_reference();
      }
      else
      {
        refs.multi = new InternalSet(init_size);
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(const InstanceSet& rhs) : single(rhs.single)
    //--------------------------------------------------------------------------
    {
      // Mark that the other one is sharing too
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single == nullptr)
        {
          shared = false;
          return;
        }
        shared = true;
        rhs.shared = true;
        refs.single->add_reference();
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::~InstanceSet(void)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((refs.single != nullptr) && refs.single->remove_reference())
          delete (refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet& InstanceSet::operator=(const InstanceSet& rhs)
    //--------------------------------------------------------------------------
    {
      // See if we need to delete our current one
      if (single)
      {
        if ((refs.single != nullptr) && refs.single->remove_reference())
          delete (refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
      // Now copy over the other one
      single = rhs.single;
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single != nullptr)
        {
          shared = true;
          rhs.shared = true;
          refs.single->add_reference();
        }
        else
          shared = false;
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::make_copy(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(shared);
      if (single)
      {
        if (refs.single != nullptr)
        {
          CollectableRef* next = new CollectableRef(*refs.single);
          next->add_reference();
          if (refs.single->remove_reference())
            delete (refs.single);
          refs.single = next;
        }
      }
      else
      {
        InternalSet* next = new InternalSet(*refs.multi);
        next->add_reference();
        if (refs.multi->remove_reference())
          delete refs.multi;
        refs.multi = next;
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator==(const InstanceSet& rhs) const
    //--------------------------------------------------------------------------
    {
      if (single != rhs.single)
        return false;
      if (single)
      {
        if (refs.single == rhs.refs.single)
          return true;
        if (((refs.single == nullptr) && (rhs.refs.single != nullptr)) ||
            ((refs.single != nullptr) && (rhs.refs.single == nullptr)))
          return false;
        return ((*refs.single) == (*rhs.refs.single));
      }
      else
      {
        if (refs.multi->vector.size() != rhs.refs.multi->vector.size())
          return false;
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          if (refs.multi->vector[idx] != rhs.refs.multi->vector[idx])
            return false;
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator!=(const InstanceSet& rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceSet::operator[](unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (shared)
        make_copy();
      if (single)
      {
        legion_assert(idx == 0);
        legion_assert(refs.single != nullptr);
        return *(refs.single);
      }
      legion_assert(idx < refs.multi->vector.size());
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    const InstanceRef& InstanceSet::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // No need to make a copy if shared here since this is read-only
      if (single)
      {
        legion_assert(idx == 0);
        legion_assert(refs.single != nullptr);
        return *(refs.single);
      }
      legion_assert(idx < refs.multi->vector.size());
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      if (single && (refs.single == nullptr))
        return true;
      else if (!single && refs.multi->empty())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    size_t InstanceSet::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == nullptr)
          return 0;
        return 1;
      }
      if (refs.multi == nullptr)
        return 0;
      return refs.multi->vector.size();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::resize(size_t new_size)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (new_size == 0)
        {
          if ((refs.single != nullptr) && refs.single->remove_reference())
            delete (refs.single);
          refs.single = nullptr;
          shared = false;
        }
        else if (new_size > 1)
        {
          // Switch to multi
          InternalSet* next = new InternalSet(new_size);
          if (refs.single != nullptr)
          {
            next->vector[0] = *(refs.single);
            if (refs.single->remove_reference())
              delete (refs.single);
          }
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else if (refs.single == nullptr)
        {
          // New size is 1 but we were empty before
          CollectableRef* next = new CollectableRef();
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
      }
      else
      {
        if (new_size == 0)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.single = nullptr;
          single = true;
          shared = false;
        }
        else if (new_size == 1)
        {
          CollectableRef* next = new CollectableRef(refs.multi->vector[0]);
          if (refs.multi->remove_reference())
            delete (refs.multi);
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
        else
        {
          size_t current_size = refs.multi->vector.size();
          if (current_size != new_size)
          {
            if (shared)
            {
              // Make a copy
              InternalSet* next = new InternalSet(new_size);
              // Copy over the elements
              for (unsigned idx = 0;
                   idx < ((current_size < new_size) ? current_size : new_size);
                   idx++)
                next->vector[idx] = refs.multi->vector[idx];
              if (refs.multi->remove_reference())
                delete refs.multi;
              next->add_reference();
              refs.multi = next;
              shared = false;
            }
            else
            {
              // Resize our existing vector
              refs.multi->vector.resize(new_size);
            }
          }
          // Size is the same so there is no need to do anything
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::clear(void)
    //--------------------------------------------------------------------------
    {
      // No need to copy since we are removing our references and not mutating
      if (single)
      {
        if ((refs.single != nullptr) && refs.single->remove_reference())
          delete (refs.single);
        refs.single = nullptr;
      }
      else
      {
        if (shared)
        {
          // Small optimization here, if we're told to delete it, we know
          // that means we were the last user so we can re-use it
          if (refs.multi->remove_reference())
          {
            // Put a reference back on it since we're reusing it
            refs.multi->add_reference();
            refs.multi->vector.clear();
          }
          else
          {
            // Go back to single
            refs.multi = nullptr;
            single = true;
          }
        }
        else
          refs.multi->vector.clear();
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::swap(InstanceSet& rhs)
    //--------------------------------------------------------------------------
    {
      // Swap references
      {
        InternalSet* other = rhs.refs.multi;
        rhs.refs.multi = refs.multi;
        refs.multi = other;
      }
      // Swap single
      {
        bool other = rhs.single;
        rhs.single = single;
        single = other;
      }
      // Swap shared
      {
        bool other = rhs.shared;
        rhs.shared = shared;
        shared = other;
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_instance(const InstanceRef& ref)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        // No need to check for shared, we're going to make new things anyway
        if (refs.single != nullptr)
        {
          // Make the new multi version
          InternalSet* next = new InternalSet(2);
          next->vector[0] = *(refs.single);
          next->vector[1] = ref;
          if (refs.single->remove_reference())
            delete (refs.single);
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else
        {
          refs.single = new CollectableRef(ref);
          refs.single->add_reference();
        }
      }
      else
      {
        if (shared)
          make_copy();
        refs.multi->vector.emplace_back(ref);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::is_virtual_mapping(void) const
    //--------------------------------------------------------------------------
    {
      if (empty())
        return true;
      if (size() > 1)
        return false;
      return refs.single->is_virtual_ref();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::pack_references(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == nullptr)
        {
          rez.serialize<size_t>(0);
          return;
        }
        rez.serialize<size_t>(1);
        refs.single->pack_reference(rez);
      }
      else
      {
        rez.serialize<size_t>(refs.multi->vector.size());
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].pack_reference(rez);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::unpack_references(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_refs;
      derez.deserialize(num_refs);
      if (num_refs == 0)
      {
        // No matter what, we can just clear out any references we have
        if (single)
        {
          if ((refs.single != nullptr) && refs.single->remove_reference())
            delete (refs.single);
          refs.single = nullptr;
        }
        else
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          single = true;
        }
      }
      else if (num_refs == 1)
      {
        // If we're in multi, go back to single
        if (!single)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.multi = nullptr;
          single = true;
        }
        // Now we can unpack our reference, see if we need to make one
        if (refs.single == nullptr)
        {
          refs.single = new CollectableRef();
          refs.single->add_reference();
        }
        RtEvent ready;
        refs.single->unpack_reference(derez, ready);
        if (ready.exists())
          ready_events.insert(ready);
      }
      else
      {
        // If we're in single, go to multi
        // otherwise resize our multi for the appropriate number of references
        if (single)
        {
          if ((refs.single != nullptr) && refs.single->remove_reference())
            delete (refs.single);
          refs.multi = new InternalSet(num_refs);
          refs.multi->add_reference();
          single = false;
        }
        else
          refs.multi->vector.resize(num_refs);
        // Now do the unpacking
        for (unsigned idx = 0; idx < num_refs; idx++)
        {
          RtEvent ready;
          refs.multi->vector[idx].unpack_reference(derez, ready);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      // We are always not shared when we are done
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_resource_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != nullptr)
          refs.single->add_resource_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_resource_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_resource_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != nullptr)
          refs.single->remove_resource_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_resource_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::acquire_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != nullptr)
          return refs.single->acquire_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          if (!refs.multi->vector[idx].acquire_valid_reference(source))
            return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != nullptr)
          refs.single->add_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_valid_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != nullptr)
          refs.single->remove_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_valid_reference(source);
      }
    }

  }  // namespace Internal
}  // namespace Legion
