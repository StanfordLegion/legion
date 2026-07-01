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

#include "legion/utilities/resources.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Resource Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(void) : provenance(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(
        LogicalRegion r, Provenance* p)
      : region(r), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(const DeletedRegion& rhs)
      : region(rhs.region), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(DeletedRegion&& rhs) noexcept
      : region(rhs.region), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::~DeletedRegion(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion& ResourceTracker::DeletedRegion::operator=(
        const DeletedRegion& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      region = rhs.region;
      provenance = rhs.provenance;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion& ResourceTracker::DeletedRegion::operator=(
        DeletedRegion&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      region = rhs.region;
      provenance = rhs.provenance;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedRegion::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(region);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedRegion::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(region);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(void)
      : fid(0), provenance(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(
        FieldSpace sp, FieldID f, Provenance* p)
      : space(sp), fid(f), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(const DeletedField& rhs)
      : space(rhs.space), fid(rhs.fid), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(DeletedField&& rhs) noexcept
      : space(rhs.space), fid(rhs.fid), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::~DeletedField(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField& ResourceTracker::DeletedField::operator=(
        const DeletedField& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      fid = rhs.fid;
      provenance = rhs.provenance;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField& ResourceTracker::DeletedField::operator=(
        DeletedField&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      fid = rhs.fid;
      provenance = rhs.provenance;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedField::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      rez.serialize(fid);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedField::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      derez.deserialize(fid);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(void)
      : provenance(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(
        FieldSpace sp, Provenance* p)
      : space(sp), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(
        const DeletedFieldSpace& rhs)
      : space(rhs.space), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(
        DeletedFieldSpace&& rhs) noexcept
      : space(rhs.space), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::~DeletedFieldSpace(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace&
        ResourceTracker::DeletedFieldSpace::operator=(
            const DeletedFieldSpace& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      provenance = rhs.provenance;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace&
        ResourceTracker::DeletedFieldSpace::operator=(
            DeletedFieldSpace&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      provenance = rhs.provenance;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedFieldSpace::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedFieldSpace::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(void)
      : provenance(nullptr), recurse(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(
        IndexSpace sp, bool r, Provenance* p)
      : space(sp), provenance(p), recurse(r)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(
        const DeletedIndexSpace& rhs)
      : space(rhs.space), provenance(rhs.provenance), recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(
        DeletedIndexSpace&& rhs) noexcept
      : space(rhs.space), provenance(rhs.provenance), recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::~DeletedIndexSpace(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace&
        ResourceTracker::DeletedIndexSpace::operator=(
            const DeletedIndexSpace& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace&
        ResourceTracker::DeletedIndexSpace::operator=(
            DeletedIndexSpace&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedIndexSpace::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      rez.serialize<bool>(recurse);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedIndexSpace::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      derez.deserialize<bool>(recurse);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(void)
      : provenance(nullptr), recurse(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(
        IndexPartition ip, bool r, Provenance* p)
      : partition(ip), provenance(p), recurse(r)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(
        const DeletedPartition& rhs)
      : partition(rhs.partition), provenance(rhs.provenance),
        recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      if (provenance != nullptr)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(
        DeletedPartition&& rhs) noexcept
      : partition(rhs.partition), provenance(rhs.provenance),
        recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = nullptr;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::~DeletedPartition(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition&
        ResourceTracker::DeletedPartition::operator=(
            const DeletedPartition& rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      partition = rhs.partition;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      if (provenance != nullptr)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition&
        ResourceTracker::DeletedPartition::operator=(
            DeletedPartition&& rhs) noexcept
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      partition = rhs.partition;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      rhs.provenance = nullptr;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedPartition::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(partition);
      rez.serialize<bool>(recurse);
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedPartition::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != nullptr) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(partition);
      derez.deserialize<bool>(recurse);
      // Reference comes back from deserialize
      provenance = Provenance::deserialize(derez);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::ResourceTracker(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ResourceTracker::~ResourceTracker(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ResourceTracker::return_resources(
        ResourceTracker* target, uint64_t return_index,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      if (created_regions.empty() && deleted_regions.empty() &&
          created_fields.empty() && deleted_fields.empty() &&
          created_field_spaces.empty() && latent_field_spaces.empty() &&
          deleted_field_spaces.empty() && created_index_spaces.empty() &&
          deleted_index_spaces.empty() && created_index_partitions.empty() &&
          deleted_index_partitions.empty())
        return;
      target->receive_resources(
          return_index, created_regions, deleted_regions, created_fields,
          deleted_fields, created_field_spaces, latent_field_spaces,
          deleted_field_spaces, created_index_spaces, deleted_index_spaces,
          created_index_partitions, deleted_index_partitions, preconditions);
      created_regions.clear();
      deleted_regions.clear();
      created_fields.clear();
      deleted_fields.clear();
      created_field_spaces.clear();
      latent_field_spaces.clear();
      deleted_field_spaces.clear();
      created_index_spaces.clear();
      deleted_index_spaces.clear();
      created_index_partitions.clear();
      deleted_index_partitions.clear();
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::pack_resources_return(
        Serializer& rez, uint64_t return_index)
    //--------------------------------------------------------------------------
    {
      // Shouldn't need the lock here since we only do this
      // while there is no one else executing
      RezCheck z(rez);
      rez.serialize(return_index);
      rez.serialize<size_t>(created_regions.size());
      if (!created_regions.empty())
      {
        for (const std::pair<const LogicalRegion, unsigned>& region :
             created_regions)
        {
          rez.serialize(region.first);
          rez.serialize(region.second);
        }
        created_regions.clear();
      }
      rez.serialize<size_t>(deleted_regions.size());
      if (!deleted_regions.empty())
      {
        for (const DeletedRegion& region : deleted_regions)
          region.serialize(rez);
        deleted_regions.clear();
      }
      rez.serialize<size_t>(created_fields.size());
      if (!created_fields.empty())
      {
        for (const std::pair<FieldSpace, FieldID>& field : created_fields)
        {
          rez.serialize(field.first);
          rez.serialize(field.second);
        }
        created_fields.clear();
      }
      rez.serialize<size_t>(deleted_fields.size());
      if (!deleted_fields.empty())
      {
        for (const DeletedField& field : deleted_fields) field.serialize(rez);
        deleted_fields.clear();
      }
      rez.serialize<size_t>(created_field_spaces.size());
      if (!created_field_spaces.empty())
      {
        for (const std::pair<const FieldSpace, unsigned>& field_space :
             created_field_spaces)
        {
          rez.serialize(field_space.first);
          rez.serialize(field_space.second);
        }
        created_field_spaces.clear();
      }
      rez.serialize<size_t>(latent_field_spaces.size());
      if (!latent_field_spaces.empty())
      {
        for (const std::pair<const FieldSpace, std::set<LogicalRegion> >&
                 latent_field_space : latent_field_spaces)
        {
          rez.serialize(latent_field_space.first);
          rez.serialize<size_t>(latent_field_space.second.size());
          for (const LogicalRegion& region : latent_field_space.second)
            rez.serialize(region);
        }
        latent_field_spaces.clear();
      }
      rez.serialize<size_t>(deleted_field_spaces.size());
      if (!deleted_field_spaces.empty())
      {
        for (const DeletedFieldSpace& field_space : deleted_field_spaces)
          field_space.serialize(rez);
        deleted_field_spaces.clear();
      }
      rez.serialize<size_t>(created_index_spaces.size());
      if (!created_index_spaces.empty())
      {
        for (const std::pair<const IndexSpace, unsigned>& index_space :
             created_index_spaces)
        {
          rez.serialize(index_space.first);
          rez.serialize(index_space.second);
        }
        created_index_spaces.clear();
      }
      rez.serialize<size_t>(deleted_index_spaces.size());
      if (!deleted_index_spaces.empty())
      {
        for (const DeletedIndexSpace& index_space : deleted_index_spaces)
          index_space.serialize(rez);
        deleted_index_spaces.clear();
      }
      rez.serialize<size_t>(created_index_partitions.size());
      if (!created_index_partitions.empty())
      {
        for (const std::pair<const IndexPartition, unsigned>& index_partition :
             created_index_partitions)
        {
          rez.serialize(index_partition.first);
          rez.serialize(index_partition.second);
        }
        created_index_partitions.clear();
      }
      rez.serialize<size_t>(deleted_index_partitions.size());
      if (!deleted_index_partitions.empty())
      {
        for (const DeletedPartition& index_partition : deleted_index_partitions)
          index_partition.serialize(rez);
        deleted_index_partitions.clear();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ResourceTracker::pack_empty_resources(
        Serializer& rez, uint64_t return_index)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(return_index);
      rez.serialize<size_t>(0);  // created regions
      rez.serialize<size_t>(0);  // deleted regions
      rez.serialize<size_t>(0);  // created fields
      rez.serialize<size_t>(0);  // deleted fields
      rez.serialize<size_t>(0);  // created field spaces
      rez.serialize<size_t>(0);  // latent field spaces
      rez.serialize<size_t>(0);  // deleted field spaces
      rez.serialize<size_t>(0);  // created index spaces
      rez.serialize<size_t>(0);  // deleted index spaces
      rez.serialize<size_t>(0);  // created index partitions
      rez.serialize<size_t>(0);  // deleted index partitions
    }

    //--------------------------------------------------------------------------
    /*static*/ RtEvent ResourceTracker::unpack_resources_return(
        Deserializer& derez, ResourceTracker* target)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while doing the unpack to avoid conflicting
      // with anyone else returning state
      DerezCheck z(derez);
      uint64_t return_index;
      derez.deserialize(return_index);
      size_t num_created_regions;
      derez.deserialize(num_created_regions);
      std::map<LogicalRegion, unsigned> created_regions;
      for (unsigned idx = 0; idx < num_created_regions; idx++)
      {
        LogicalRegion reg;
        derez.deserialize(reg);
        derez.deserialize(created_regions[reg]);
      }
      size_t num_deleted_regions;
      derez.deserialize(num_deleted_regions);
      std::vector<DeletedRegion> deleted_regions(num_deleted_regions);
      for (unsigned idx = 0; idx < num_deleted_regions; idx++)
        deleted_regions[idx].deserialize(derez);
      size_t num_created_fields;
      derez.deserialize(num_created_fields);
      std::set<std::pair<FieldSpace, FieldID> > created_fields;
      for (unsigned idx = 0; idx < num_created_fields; idx++)
      {
        std::pair<FieldSpace, FieldID> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        created_fields.insert(key);
      }
      size_t num_deleted_fields;
      derez.deserialize(num_deleted_fields);
      std::vector<DeletedField> deleted_fields(num_deleted_fields);
      for (unsigned idx = 0; idx < num_deleted_fields; idx++)
        deleted_fields[idx].deserialize(derez);
      size_t num_created_field_spaces;
      derez.deserialize(num_created_field_spaces);
      std::map<FieldSpace, unsigned> created_field_spaces;
      for (unsigned idx = 0; idx < num_created_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_field_spaces[sp]);
      }
      size_t num_latent_field_spaces;
      derez.deserialize(num_latent_field_spaces);
      std::map<FieldSpace, std::set<LogicalRegion> > latent_field_spaces;
      for (unsigned idx = 0; idx < num_latent_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        std::set<LogicalRegion>& regions = latent_field_spaces[sp];
        size_t num_regions;
        derez.deserialize(num_regions);
        for (unsigned idx2 = 0; idx2 < num_regions; idx2++)
        {
          LogicalRegion region;
          derez.deserialize(region);
          regions.insert(region);
        }
      }
      size_t num_deleted_field_spaces;
      derez.deserialize(num_deleted_field_spaces);
      std::vector<DeletedFieldSpace> deleted_field_spaces(
          num_deleted_field_spaces);
      for (unsigned idx = 0; idx < num_deleted_field_spaces; idx++)
        deleted_field_spaces[idx].deserialize(derez);
      size_t num_created_index_spaces;
      derez.deserialize(num_created_index_spaces);
      std::map<IndexSpace, unsigned> created_index_spaces;
      for (unsigned idx = 0; idx < num_created_index_spaces; idx++)
      {
        IndexSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_index_spaces[sp]);
      }
      size_t num_deleted_index_spaces;
      derez.deserialize(num_deleted_index_spaces);
      std::vector<DeletedIndexSpace> deleted_index_spaces(
          num_deleted_index_spaces);
      for (unsigned idx = 0; idx < num_deleted_index_spaces; idx++)
        deleted_index_spaces[idx].deserialize(derez);
      size_t num_created_index_partitions;
      derez.deserialize(num_created_index_partitions);
      std::map<IndexPartition, unsigned> created_index_partitions;
      for (unsigned idx = 0; idx < num_created_index_partitions; idx++)
      {
        IndexPartition ip;
        derez.deserialize(ip);
        derez.deserialize(created_index_partitions[ip]);
      }
      size_t num_deleted_index_partitions;
      derez.deserialize(num_deleted_index_partitions);
      std::vector<DeletedPartition> deleted_index_partitions(
          num_deleted_index_partitions);
      for (unsigned idx = 0; idx < num_deleted_index_partitions; idx++)
        deleted_index_partitions[idx].deserialize(derez);
      std::set<RtEvent> preconditions;
      target->receive_resources(
          return_index, created_regions, deleted_regions, created_fields,
          deleted_fields, created_field_spaces, latent_field_spaces,
          deleted_field_spaces, created_index_spaces, deleted_index_spaces,
          created_index_partitions, deleted_index_partitions, preconditions);
      if (!preconditions.empty())
        return Runtime::merge_events(preconditions);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::merge_received_resources(
        std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID> >& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion> >& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions)
    //--------------------------------------------------------------------------
    {
      if (!created_regs.empty())
      {
        if (!latent_field_spaces.empty())
        {
          for (const std::pair<const LogicalRegion, unsigned>& created_region :
               created_regs)
          {
            std::map<FieldSpace, std::set<LogicalRegion> >::iterator finder =
                latent_field_spaces.find(
                    created_region.first.get_field_space());
            if (finder != latent_field_spaces.end())
              finder->second.insert(created_region.first);
          }
        }
        if (!created_regions.empty())
        {
          for (const std::pair<const LogicalRegion, unsigned>& created_region :
               created_regs)
          {
            std::map<LogicalRegion, unsigned>::iterator finder =
                created_regions.find(created_region.first);
            if (finder == created_regions.end())
              created_regions.insert(created_region);
            else
              finder->second += created_region.second;
          }
        }
        else
          created_regions.swap(created_regs);
      }
      if (!deleted_regs.empty())
      {
        if (!deleted_regions.empty())
          deleted_regions.insert(
              deleted_regions.end(), deleted_regs.begin(), deleted_regs.end());
        else
          deleted_regions.swap(deleted_regs);
      }
      if (!created_fids.empty())
      {
        if (!created_fields.empty())
        {
          for (const std::pair<FieldSpace, FieldID>& created_field :
               created_fids)
          {
            legion_assert(
                created_fields.find(created_field) == created_fields.end());
            created_fields.insert(created_field);
          }
        }
        else
          created_fields.swap(created_fids);
      }
      if (!deleted_fids.empty())
      {
        if (!deleted_fields.empty())
          deleted_fields.insert(
              deleted_fields.end(), deleted_fids.begin(), deleted_fids.end());
        else
          deleted_fields.swap(deleted_fids);
      }
      if (!created_fs.empty())
      {
        if (!latent_field_spaces.empty())
        {
          // Remove any latent field spaces we have ownership for
          for (const std::pair<const FieldSpace, unsigned>&
                   created_field_space : created_fs)
          {
            std::map<FieldSpace, std::set<LogicalRegion> >::iterator finder =
                latent_field_spaces.find(created_field_space.first);
            if (finder != latent_field_spaces.end())
              latent_field_spaces.erase(finder);
          }
        }
        if (!created_field_spaces.empty())
        {
          for (const std::pair<const FieldSpace, unsigned>&
                   created_field_space : created_fs)
          {
            std::map<FieldSpace, unsigned>::iterator finder =
                created_field_spaces.find(created_field_space.first);
            if (finder == created_field_spaces.end())
              created_field_spaces.insert(created_field_space);
            else
              finder->second += created_field_space.second;
          }
        }
        else
          created_field_spaces.swap(created_fs);
      }
      if (!latent_fs.empty())
      {
        if (!created_field_spaces.empty())
        {
          // Remote any latent field spaces we already have ownership on
          for (std::map<FieldSpace, std::set<LogicalRegion> >::iterator it =
                   latent_fs.begin();
               it != latent_fs.end();
               /*nothing*/)
          {
            if (created_field_spaces.find(it->first) !=
                created_field_spaces.end())
            {
              std::map<FieldSpace, std::set<LogicalRegion> >::iterator
                  to_delete = it++;
              latent_fs.erase(to_delete);
            }
            else
              it++;
          }
        }
        if (!created_regions.empty())
        {
          // See if any of these regions are copies of our latent spaces
          for (const std::pair<const LogicalRegion, unsigned>& created_region :
               created_regions)
          {
            std::map<FieldSpace, std::set<LogicalRegion> >::iterator finder =
                latent_fs.find(created_region.first.get_field_space());
            if (finder != latent_fs.end())
              finder->second.insert(created_region.first);
          }
        }
        // Now we can do the merge
        if (!latent_field_spaces.empty())
        {
          for (const std::pair<const FieldSpace, std::set<LogicalRegion> >&
                   latent_field_space : latent_fs)
          {
            std::map<FieldSpace, std::set<LogicalRegion> >::iterator finder =
                latent_field_spaces.find(latent_field_space.first);
            if (finder != latent_field_spaces.end())
              finder->second.insert(
                  latent_field_space.second.begin(),
                  latent_field_space.second.end());
            else
              latent_field_spaces.insert(latent_field_space);
          }
        }
        else
          latent_field_spaces.swap(latent_fs);
      }
      if (!deleted_fs.empty())
      {
        if (!deleted_field_spaces.empty())
          deleted_field_spaces.insert(
              deleted_field_spaces.end(), deleted_fs.begin(), deleted_fs.end());
        else
          deleted_field_spaces.swap(deleted_fs);
      }
      if (!created_is.empty())
      {
        if (!created_index_spaces.empty())
        {
          for (const std::pair<const IndexSpace, unsigned>&
                   created_index_space : created_is)
          {
            std::map<IndexSpace, unsigned>::iterator finder =
                created_index_spaces.find(created_index_space.first);
            if (finder == created_index_spaces.end())
              created_index_spaces.insert(created_index_space);
            else
              finder->second += created_index_space.second;
          }
        }
        else
          created_index_spaces.swap(created_is);
      }
      if (!deleted_is.empty())
      {
        if (!deleted_index_spaces.empty())
          deleted_index_spaces.insert(
              deleted_index_spaces.end(), deleted_is.begin(), deleted_is.end());
        else
          deleted_index_spaces.swap(deleted_is);
      }
      if (!created_partitions.empty())
      {
        if (!created_index_partitions.empty())
        {
          for (const std::pair<const IndexPartition, unsigned>&
                   created_partition : created_partitions)
          {
            std::map<IndexPartition, unsigned>::iterator finder =
                created_index_partitions.find(created_partition.first);
            if (finder == created_index_partitions.end())
              created_index_partitions.insert(created_partition);
            else
              finder->second += created_partition.second;
          }
        }
        else
          created_index_partitions.swap(created_partitions);
      }
      if (!deleted_partitions.empty())
      {
        if (!deleted_index_partitions.empty())
          deleted_index_partitions.insert(
              deleted_index_partitions.end(), deleted_partitions.begin(),
              deleted_partitions.end());
        else
          deleted_index_partitions.swap(deleted_partitions);
      }
    }

  }  // namespace Internal
}  // namespace Legion
