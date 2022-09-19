/* Copyright 2022 Stanford University, NVIDIA Corporation
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


#include "legion/region_tree.h"
#include "legion/legion_tasks.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_analysis.h"
#include "legion/legion_views.h"

#include <algorithm>

#define PRINT_REG(reg) (reg).index_space.id,(reg).field_space.id, (reg).tree_id

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Resource Tracker 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(void)
      : provenance(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(LogicalRegion r,
                                                  Provenance *p)
      : region(r), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(const DeletedRegion &rhs)
      : region(rhs.region), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::DeletedRegion(DeletedRegion &&rhs)
      : region(rhs.region), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion::~DeletedRegion(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion& ResourceTracker::DeletedRegion::operator=(
                                                       const DeletedRegion &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      region = rhs.region;
      provenance = rhs.provenance;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedRegion& ResourceTracker::DeletedRegion::operator=(
                                                            DeletedRegion &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      region = rhs.region;
      provenance = rhs.provenance;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedRegion::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(region);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedRegion::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(region);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(void)
      : fid(0), provenance(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(FieldSpace sp, FieldID f,
                                                Provenance *p)
      : space(sp), fid(f), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(const DeletedField&rhs)
      : space(rhs.space), fid(rhs.fid), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::DeletedField(DeletedField &&rhs)
      : space(rhs.space), fid(rhs.fid), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField::~DeletedField(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField& ResourceTracker::DeletedField::operator=(
                                                        const DeletedField &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      fid = rhs.fid;
      provenance = rhs.provenance;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedField& ResourceTracker::DeletedField::operator=(
                                                             DeletedField &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      fid = rhs.fid;
      provenance = rhs.provenance;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedField::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      rez.serialize(fid);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedField::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      derez.deserialize(fid);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(void)
      : provenance(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(FieldSpace sp,
                                                          Provenance *p)
      : space(sp), provenance(p)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(
                                                   const DeletedFieldSpace &rhs)
      : space(rhs.space), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::DeletedFieldSpace(
                                                        DeletedFieldSpace &&rhs)
      : space(rhs.space), provenance(rhs.provenance)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace::~DeletedFieldSpace(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace& 
      ResourceTracker::DeletedFieldSpace::operator=(
                                                   const DeletedFieldSpace &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      provenance = rhs.provenance;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedFieldSpace& 
      ResourceTracker::DeletedFieldSpace::operator=(DeletedFieldSpace &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      provenance = rhs.provenance;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedFieldSpace::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedFieldSpace::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(void)
      : provenance(NULL), recurse(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(IndexSpace sp,
                                                          bool r, Provenance *p)
      : space(sp), provenance(p), recurse(r)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(
                                                   const DeletedIndexSpace &rhs)
      : space(rhs.space), provenance(rhs.provenance), recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::DeletedIndexSpace(
                                                        DeletedIndexSpace &&rhs)
      : space(rhs.space), provenance(rhs.provenance), recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace::~DeletedIndexSpace(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace& 
      ResourceTracker::DeletedIndexSpace::operator=(
                                                   const DeletedIndexSpace &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedIndexSpace& 
          ResourceTracker::DeletedIndexSpace::operator=(DeletedIndexSpace &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      space = rhs.space;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedIndexSpace::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(space);
      rez.serialize<bool>(recurse);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedIndexSpace::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(space);
      derez.deserialize<bool>(recurse);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(void)
      : provenance(NULL), recurse(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(IndexPartition ip,
                                                        bool r, Provenance *p)
      : partition(ip), provenance(p), recurse(r)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(
                                                     const DeletedPartition&rhs)
      : partition(rhs.partition),provenance(rhs.provenance),recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::DeletedPartition(DeletedPartition&&rhs)
      : partition(rhs.partition),provenance(rhs.provenance),recurse(rhs.recurse)
    //--------------------------------------------------------------------------
    {
      rhs.provenance = NULL;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition::~DeletedPartition(void)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition& 
      ResourceTracker::DeletedPartition::operator=(const DeletedPartition &rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      partition = rhs.partition;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      if (provenance != NULL)
        provenance->add_reference();
      return *this;
    }

    //--------------------------------------------------------------------------
    ResourceTracker::DeletedPartition& 
            ResourceTracker::DeletedPartition::operator=(DeletedPartition &&rhs)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      partition = rhs.partition;
      recurse = rhs.recurse;
      provenance = rhs.provenance;
      rhs.provenance = NULL;
      return *this;
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedPartition::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(partition);
      rez.serialize<bool>(recurse);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
    }

    //--------------------------------------------------------------------------
    void ResourceTracker::DeletedPartition::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      derez.deserialize(partition);
      derez.deserialize<bool>(recurse);
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
    }

    //--------------------------------------------------------------------------
    ResourceTracker::ResourceTracker(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker::ResourceTracker(const ResourceTracker &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ResourceTracker::~ResourceTracker(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ResourceTracker& ResourceTracker::operator=(const ResourceTracker&rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    } 

    //--------------------------------------------------------------------------
    void ResourceTracker::return_resources(ResourceTracker *target, 
                          size_t return_index, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      if (created_regions.empty() && deleted_regions.empty() && 
          created_fields.empty() && deleted_fields.empty() &&
          created_field_spaces.empty() && latent_field_spaces.empty() &&
          deleted_field_spaces.empty() && created_index_spaces.empty() &&
          deleted_index_spaces.empty() && created_index_partitions.empty() &&
          deleted_index_partitions.empty())
        return;
      target->receive_resources(return_index, created_regions, deleted_regions,
          created_fields, deleted_fields, created_field_spaces, 
          latent_field_spaces, deleted_field_spaces, created_index_spaces,
          deleted_index_spaces, created_index_partitions, 
          deleted_index_partitions, preconditions); 
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
    void ResourceTracker::pack_resources_return(Serializer &rez, 
                                                size_t return_index)
    //--------------------------------------------------------------------------
    {
      // Shouldn't need the lock here since we only do this
      // while there is no one else executing
      RezCheck z(rez);
      rez.serialize(return_index);
      rez.serialize<size_t>(created_regions.size());
      if (!created_regions.empty())
      {
        for (std::map<LogicalRegion,unsigned>::const_iterator it =
              created_regions.begin(); it != created_regions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        created_regions.clear();
      }
      rez.serialize<size_t>(deleted_regions.size());
      if (!deleted_regions.empty())
      {
        for (std::vector<DeletedRegion>::const_iterator it = 
              deleted_regions.begin(); it != deleted_regions.end(); it++)
          it->serialize(rez);
        deleted_regions.clear();
      }
      rez.serialize<size_t>(created_fields.size());
      if (!created_fields.empty())
      {
        for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator 
              it = created_fields.begin(); it != created_fields.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        created_fields.clear();
      }
      rez.serialize<size_t>(deleted_fields.size());
      if (!deleted_fields.empty())
      {
        for (std::vector<DeletedField>::const_iterator it =
              deleted_fields.begin(); it != deleted_fields.end(); it++)
          it->serialize(rez);
        deleted_fields.clear();
      }
      rez.serialize<size_t>(created_field_spaces.size());
      if (!created_field_spaces.empty())
      {
        for (std::map<FieldSpace,unsigned>::const_iterator it = 
              created_field_spaces.begin(); it != 
              created_field_spaces.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        created_field_spaces.clear();
      } 
      rez.serialize<size_t>(latent_field_spaces.size());
      if (!latent_field_spaces.empty())
      {
        for (std::map<FieldSpace,std::set<LogicalRegion> >::const_iterator it =
              latent_field_spaces.begin(); it != 
              latent_field_spaces.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize<size_t>(it->second.size());
          for (std::set<LogicalRegion>::const_iterator it2 = 
                it->second.begin(); it2 != it->second.end(); it2++)
            rez.serialize(*it2);
        }
        latent_field_spaces.clear();
      }
      rez.serialize<size_t>(deleted_field_spaces.size());
      if (!deleted_field_spaces.empty())
      {
        for (std::vector<DeletedFieldSpace>::const_iterator it = 
              deleted_field_spaces.begin(); it != 
              deleted_field_spaces.end(); it++)
          it->serialize(rez);
        deleted_field_spaces.clear();
      }
      rez.serialize<size_t>(created_index_spaces.size());
      if (!created_index_spaces.empty())
      {
        for (std::map<IndexSpace,unsigned>::const_iterator it = 
              created_index_spaces.begin(); it != 
              created_index_spaces.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        created_index_spaces.clear();
      }
      rez.serialize<size_t>(deleted_index_spaces.size());
      if (!deleted_index_spaces.empty())
      {
        for (std::vector<DeletedIndexSpace>::const_iterator it = 
              deleted_index_spaces.begin(); it != 
              deleted_index_spaces.end(); it++)
          it->serialize(rez);
        deleted_index_spaces.clear();
      }
      rez.serialize<size_t>(created_index_partitions.size());
      if (!created_index_partitions.empty())
      {
        for (std::map<IndexPartition,unsigned>::const_iterator it = 
              created_index_partitions.begin(); it !=
              created_index_partitions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        created_index_partitions.clear();
      }
      rez.serialize<size_t>(deleted_index_partitions.size());
      if (!deleted_index_partitions.empty())
      {
        for (std::vector<DeletedPartition>::const_iterator it = 
              deleted_index_partitions.begin(); it !=
              deleted_index_partitions.end(); it++)
          it->serialize(rez);
        deleted_index_partitions.clear();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ RtEvent ResourceTracker::unpack_resources_return(
                                   Deserializer &derez, ResourceTracker *target)
    //--------------------------------------------------------------------------
    {
      // Hold the lock while doing the unpack to avoid conflicting
      // with anyone else returning state
      DerezCheck z(derez);
      size_t return_index;
      derez.deserialize(return_index);
      size_t num_created_regions;
      derez.deserialize(num_created_regions);
      std::map<LogicalRegion,unsigned> created_regions;
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
      std::set<std::pair<FieldSpace,FieldID> > created_fields;
      for (unsigned idx = 0; idx < num_created_fields; idx++)
      {
        std::pair<FieldSpace,FieldID> key;
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
      std::map<FieldSpace,unsigned> created_field_spaces;
      for (unsigned idx = 0; idx < num_created_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_field_spaces[sp]);
      }
      size_t num_latent_field_spaces;
      derez.deserialize(num_latent_field_spaces);
      std::map<FieldSpace,std::set<LogicalRegion> > latent_field_spaces;
      for (unsigned idx = 0; idx < num_latent_field_spaces; idx++)
      {
        FieldSpace sp;
        derez.deserialize(sp);
        std::set<LogicalRegion> &regions = latent_field_spaces[sp];
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
      std::vector<DeletedFieldSpace> 
        deleted_field_spaces(num_deleted_field_spaces);
      for (unsigned idx = 0; idx < num_deleted_field_spaces; idx++)
        deleted_field_spaces[idx].deserialize(derez);
      size_t num_created_index_spaces;
      derez.deserialize(num_created_index_spaces);
      std::map<IndexSpace,unsigned> created_index_spaces;
      for (unsigned idx = 0; idx < num_created_index_spaces; idx++)
      {
        IndexSpace sp;
        derez.deserialize(sp);
        derez.deserialize(created_index_spaces[sp]);
      }
      size_t num_deleted_index_spaces;
      derez.deserialize(num_deleted_index_spaces);
      std::vector<DeletedIndexSpace> 
          deleted_index_spaces(num_deleted_index_spaces);
      for (unsigned idx = 0; idx < num_deleted_index_spaces; idx++)
        deleted_index_spaces[idx].deserialize(derez);
      size_t num_created_index_partitions;
      derez.deserialize(num_created_index_partitions);
      std::map<IndexPartition,unsigned> created_index_partitions;
      for (unsigned idx = 0; idx < num_created_index_partitions; idx++)
      {
        IndexPartition ip;
        derez.deserialize(ip);
        derez.deserialize(created_index_partitions[ip]);
      }
      size_t num_deleted_index_partitions;
      derez.deserialize(num_deleted_index_partitions);
      std::vector<DeletedPartition> 
        deleted_index_partitions(num_deleted_index_partitions);
      for (unsigned idx = 0; idx < num_deleted_index_partitions; idx++)
        deleted_index_partitions[idx].deserialize(derez);
      std::set<RtEvent> preconditions;
      target->receive_resources(return_index, created_regions, deleted_regions,
          created_fields, deleted_fields, created_field_spaces,
          latent_field_spaces, deleted_field_spaces, created_index_spaces,
          deleted_index_spaces, created_index_partitions,
          deleted_index_partitions, preconditions);
      if (!preconditions.empty())
        return Runtime::merge_events(preconditions);
      else
        return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // External Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalTask::ExternalTask(void)
      : Task(), arg_manager(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalTask::pack_external_task(Serializer &rez,
                                          AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(task_id);
      rez.serialize(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        pack_index_space_requirement(indexes[idx], rez);
      rez.serialize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        pack_region_requirement(regions[idx], rez);
      rez.serialize(futures.size());
      for (std::vector<Future>::const_iterator it =
            futures.begin(); it != futures.end(); it++) 
        if (it->impl != NULL)
        {
          rez.serialize(it->impl->did);
          rez.serialize(it->impl->context->get_unique_id());
        }
        else
          rez.serialize<DistributedID>(0);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize<bool>((arg_manager != NULL));
      rez.serialize(arglen);
      rez.serialize(args,arglen);
      pack_mappable(*this, rez);
      rez.serialize(is_index_space);
      rez.serialize(must_epoch_task);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      rez.serialize(sharding_space);
      rez.serialize(local_arglen);
      rez.serialize(local_args,local_arglen);
      rez.serialize(orig_proc);
      // No need to pack current proc, it will get set when we unpack
      rez.serialize(steal_count);
      // No need to pack remote, it will get set
      rez.serialize(speculated);
      // No need to pack local function, it's not if we're sending this remote
      rez.serialize<size_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalTask::unpack_external_task(Deserializer &derez,
                                    Runtime *runtime, ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(task_id);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      indexes.resize(num_indexes);
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        unpack_index_space_requirement(indexes[idx], derez);
      size_t num_regions;
      derez.deserialize(num_regions);
      regions.resize(num_regions);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        unpack_region_requirement(regions[idx], derez); 
      size_t num_futures;
      derez.deserialize(num_futures);
      futures.resize(num_futures);
      for (unsigned idx = 0; idx < futures.size(); idx++)
      {
        DistributedID future_did;
        derez.deserialize(future_did);
        if (future_did == 0)
          continue;
        UniqueID uid;
        derez.deserialize(uid);
        FutureImpl *impl = 
          runtime->find_or_create_future(future_did, uid, mutator);
        impl->add_base_gc_ref(FUTURE_HANDLE_REF, mutator);
        futures[idx] = Future(impl, false/*need reference*/);
      }
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      bool has_arg_manager;
      derez.deserialize(has_arg_manager);
      derez.deserialize(arglen);
      if (arglen > 0)
      {
        if (has_arg_manager)
        {
#ifdef DEBUG_LEGION
          assert(arg_manager == NULL);
#endif
          arg_manager = new AllocManager(arglen);
          arg_manager->add_reference();
          args = arg_manager->get_allocation();
        }
        else
          args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        derez.deserialize(args,arglen);
      }
      unpack_mappable(*this, derez); 
      derez.deserialize(is_index_space);
      derez.deserialize(must_epoch_task);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      derez.deserialize(sharding_space);
      derez.deserialize(local_arglen);
      if (local_arglen > 0)
      {
        local_args = malloc(local_arglen);
        derez.deserialize(local_args,local_arglen);
      }
      derez.deserialize(orig_proc);
      derez.deserialize(steal_count);
      derez.deserialize(speculated);
      size_t index;
      derez.deserialize(index);
      set_context_index(index);
    } 

    /////////////////////////////////////////////////////////////
    // Task Operation 
    /////////////////////////////////////////////////////////////
  
    //--------------------------------------------------------------------------
    TaskOp::TaskOp(Runtime *rt)
      : ExternalTask(), MemoizableOp<SpeculativeOp>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskOp::~TaskOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UniqueID TaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int TaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
      return parent_ctx->get_depth() + 1;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* TaskOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_reducing_future(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                       std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }
    
    //--------------------------------------------------------------------------
    void TaskOp::pack_profiling_requests(Serializer &rez,
                                         std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_remote(void) const
    //--------------------------------------------------------------------------
    {
      if (local_cached)
        return !is_local;
      if (!orig_proc.exists())
        is_local = runtime->is_local(parent_ctx->get_executing_processor());
      else
        is_local = runtime->is_local(orig_proc);
      local_cached = true;
      return !is_local;
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_current_proc(Processor current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current.exists());
      assert(runtime->is_local(current));
#endif
      // Always clear target_proc and the mapper when setting a new current proc
      mapper = NULL;
      current_proc = current;
      target_proc = current;
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate_task(void)
    //--------------------------------------------------------------------------
    {
      activate_speculative();
      activate_memoizable();
      complete_received = false;
      commit_received = false;
      children_complete = false;
      children_commit = false;
      stealable = false;
      options_selected = false;
      map_origin = false;
      request_valid_instances = false;
      true_guard = PredEvent::NO_PRED_EVENT;
      false_guard = PredEvent::NO_PRED_EVENT;
      local_cached = false;
      arg_manager = NULL;
      target_proc = Processor::NO_PROC;
      mapper = NULL;
      must_epoch = NULL;
      must_epoch_task = false;
      local_function = false;
      orig_proc = Processor::NO_PROC; // for is_remote
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate_task(void)
    //--------------------------------------------------------------------------
    {
      deactivate_speculative();
      indexes.clear();
      regions.clear();
      futures.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (args != NULL)
      {
        if (arg_manager != NULL)
        {
          // If the arg manager is not NULL then we delete the
          // argument manager and just zero out the arguments
          if (arg_manager->remove_reference())
            delete (arg_manager);
          arg_manager = NULL;
        }
        else
          legion_free(TASK_ARGS_ALLOC, args, arglen);
        args = NULL;
        arglen = 0;
      }
      if (local_args != NULL)
      {
        free(local_args);
        local_args = NULL;
        local_arglen = 0;
      }
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      early_mapped_regions.clear();
      atomic_locks.clear(); 
      parent_req_indexes.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_must_epoch(MustEpochOp *epoch, unsigned index,
                                bool do_registration)
    //--------------------------------------------------------------------------
    {
      Operation::set_must_epoch(epoch, do_registration);
      must_epoch_index = index;
      must_epoch_task = true;
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_BASE_TASK_CALL);
      // pack all the user facing data first
      pack_external_task(rez, target); 
      RezCheck z(rez);
      rez.serialize(parent_req_indexes.size());
      for (unsigned idx = 0; idx < parent_req_indexes.size(); idx++)
        rez.serialize(parent_req_indexes[idx]);
      rez.serialize(map_origin);
      if (map_origin)
      {
        rez.serialize<size_t>(atomic_locks.size());
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      else
      {
        rez.serialize(memo_state);
        if (memo_state == MEMO_RECORD)
        {
          rez.serialize(tpl);
          rez.serialize(trace_local_id);
        }
      }
      rez.serialize(request_valid_instances);
      rez.serialize(execution_fence_event);
      rez.serialize(true_guard);
      rez.serialize(false_guard);
      rez.serialize(early_mapped_regions.size());
      for (std::map<unsigned,InstanceSet>::iterator it = 
            early_mapped_regions.begin(); it != 
            early_mapped_regions.end(); it++)
      {
        rez.serialize(it->first);
        it->second.pack_references(rez);
      } 
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_base_task(Deserializer &derez,
                                  std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_BASE_TASK_CALL);
      // unpack all the user facing data
      unpack_external_task(derez, runtime, this); 
      DerezCheck z(derez);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      if (num_indexes > 0)
      {
        parent_req_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
          derez.deserialize(parent_req_indexes[idx]);
      }
      derez.deserialize(map_origin);
      if (map_origin)
      {
        size_t num_atomic;
        derez.deserialize(num_atomic);
        for (unsigned idx = 0; idx < num_atomic; idx++)
        {
          Reservation lock;
          derez.deserialize(lock);
          derez.deserialize(atomic_locks[lock]);
        }
      }
      else
      {
        derez.deserialize(memo_state);
        if (memo_state == MEMO_RECORD)
        {
          derez.deserialize(tpl);
          derez.deserialize(trace_local_id);
        }
      }
      derez.deserialize(request_valid_instances);
      derez.deserialize(execution_fence_event);
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      size_t num_early;
      derez.deserialize(num_early);
      for (unsigned idx = 0; idx < num_early; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        early_mapped_regions[index].unpack_references(runtime, derez, 
                                                      ready_events);
      }
      // Already had our options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::process_unpack_task(Runtime *rt,Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      Processor current;
      derez.deserialize(current);
      TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case INDIVIDUAL_TASK_KIND:
          {
            IndividualTask *task = rt->get_available_individual_task();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Origin mapped tasks can go straight to launching 
              // themselves since they are already mapped
              if (task->is_origin_mapped())
              {
                TriggerTaskArgs trigger_args(task);
                rt->issue_runtime_meta_task(trigger_args, 
                      LG_THROUGHPUT_WORK_PRIORITY, ready);
              }
              else
                task->enqueue_ready_task(false/*target*/, ready);
            }
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Origin mapped tasks can go straight to launching 
              // themselves since they are already mapped
              if (task->is_origin_mapped())
              {
                TriggerTaskArgs trigger_args(task);
                rt->issue_runtime_meta_task(trigger_args, 
                      LG_THROUGHPUT_WORK_PRIORITY, ready);
              }
              else
                task->enqueue_ready_task(false/*target*/, ready);
            }
            break;
          }
        case POINT_TASK_KIND:
        case INDEX_TASK_KIND:
        default:
          assert(false); // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::process_remote_replay(Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      ApEvent instance_ready;
      derez.deserialize(instance_ready);
      Processor target_proc;
      derez.deserialize(target_proc);
      TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case INDIVIDUAL_TASK_KIND:
          {
            IndividualTask *task = rt->get_available_individual_task();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready);
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready);
            break;
          }
        case POINT_TASK_KIND:
        case INDEX_TASK_KIND:
        default:
          assert(false); // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::mark_stolen(void)
    //--------------------------------------------------------------------------
    {
      steal_count++;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_base_task(InnerContext *ctx, bool track, 
                const std::vector<StaticDependence> *dependences,
                const Predicate &p, Processor::TaskFuncID tid, const char *prov)
    //--------------------------------------------------------------------------
    {
      initialize_speculation(ctx, track, regions.size(), dependences, p, prov);
      initialize_memoizable();
      parent_task = ctx->get_task(); // initialize the parent task
      // Fill in default values for all of the Task fields
      orig_proc = ctx->get_executing_processor();
      current_proc = orig_proc;
      steal_count = 0;
      speculated = false;
      local_function = false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_base_task(InnerContext *ctx, bool track, 
                const std::vector<StaticDependence> *dependences,
                const Predicate &p, Processor::TaskFuncID tid, Provenance *prov)
    //--------------------------------------------------------------------------
    {
      initialize_speculation(ctx, track, regions.size(), dependences, p, prov);
      initialize_memoizable();
      parent_task = ctx->get_task(); // initialize the parent task
      // Fill in default values for all of the Task fields
      orig_proc = ctx->get_executing_processor();
      current_proc = orig_proc;
      steal_count = 0;
      speculated = false;
      local_function = false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::check_empty_field_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].privilege != LEGION_NO_ACCESS && 
            regions[idx].privilege_fields.empty())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENT_TASK,
                           "REGION REQUIREMENT %d OF "
                           "TASK %s (ID %lld) HAS NO PRIVILEGE "
                           "FIELDS! DID YOU FORGET THEM?!?",
                           idx, get_task_name(), get_unique_id());
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::check_future_size(FutureImpl *impl)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(impl != NULL);
#endif
      return impl->get_untyped_size(true);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::select_task_options(bool prioritize)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!options_selected);
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::TaskOptions options;
      options.initial_proc = current_proc;
      options.inline_task = false;
      options.stealable = false;
      options.map_locally = false;
      options.valid_instances = mapper->request_valid_instances;
      options.memoize = false;
      options.replicate = false;
      const TaskPriority parent_priority = parent_ctx->is_priority_mutable() ?
        parent_ctx->get_current_priority() : 0;
      options.parent_priority = parent_priority;
      mapper->invoke_select_task_options(this, &options, &prioritize);
      options_selected = true;
      if (options.initial_proc.kind() == Processor::UTIL_PROC)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output. Mapper %s requested that task %s (UID %lld)"
            " initially be assigned to a utility processor in "
            "'select_task_options.' Only application processor kinds are "
            "permitted to be the target processor for tasks.",
            mapper->get_mapper_name(), get_task_name(), get_unique_id())
      target_proc = options.initial_proc;
      if (local_function && !runtime->is_local(target_proc))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output. Mapper %s requested that local function "
            "task %s (UID %lld) be assigned to processor " IDFMT " which is "
            "not local to address space %d. Local function tasks must be "
            "assigned to local processors.", mapper->get_mapper_name(),
            get_task_name(), get_unique_id(),
            target_proc.id, runtime->address_space)
      stealable = options.stealable;
      map_origin = options.map_locally;
      request_valid_instances = options.valid_instances;
      if (parent_priority != options.parent_priority)
      {
        // Request for priority change see if it is legal or not
        if (parent_ctx->is_priority_mutable())
          parent_ctx->set_current_priority(options.parent_priority);
        else
          REPORT_LEGION_WARNING(LEGION_WARNING_INVALID_PRIORITY_CHANGE,
                                "Mapper %s requested change of priority "
                                "for parent task %s (UID %lld) when launching "
                                "child task %s (UID %lld), but the parent "
                                "context does not support parent task priority "
                                "mutation", mapper->get_mapper_name(),
                                parent_ctx->get_task_name(),
                                parent_ctx->get_unique_id(), 
                                get_task_name(), get_unique_id())
      }
      return options.inline_task;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return get_task_name();
    }

    //--------------------------------------------------------------------------
    Operation::OpKind TaskOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return regions.size();
    }

    //--------------------------------------------------------------------------
    Mappable* TaskOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_complete(void) 
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!complete_received);
        assert(!commit_received);
#endif
        complete_received = true;
        // If all our children are also complete then we are done
        task_complete = children_complete;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false; 
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(complete_received);
        assert(!commit_received);
#endif
        commit_received = true;
        // If we already received the child commit then we
        // are ready to commit this task
        task_commit = children_commit;
      }
      if (task_commit)
        trigger_task_commit();
    } 

    //--------------------------------------------------------------------------
    bool TaskOp::query_speculate(bool &value, bool &mapping_only)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)  
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::SpeculativeOutput output;
      output.speculate = false;
      output.speculate_mapping_only = true;
      mapper->invoke_task_speculate(this, &output);
      if (output.speculate)
      {
        value = output.speculative_value;
        mapping_only = output.speculate_mapping_only;
        if (!mapping_only)
        {
          REPORT_LEGION_ERROR(ERROR_MAPPER_REQUESTED_EXECUTION,
                         "Mapper requested execution speculation for task %s "
                         "(UID %lld). Full execution speculation is a planned "
                         "feature but is not currently supported.",
                         get_task_name(), get_unique_id());
          assert(false);
        }
#ifdef DEBUG_LEGION
        assert(!true_guard.exists());
        assert(!false_guard.exists());
#endif
        predicate->get_predicate_guards(true_guard, false_guard);
        // Switch any write-discard privileges back to read-write
        // so we can make sure we get the right data if we end up
        // predicating false
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          RegionRequirement &req = regions[idx];
          if (HAS_WRITE_DISCARD(req))
            req.privilege &= ~LEGION_DISCARD_MASK;
        }
      }
      return output.speculate;
    }

    //--------------------------------------------------------------------------
    void TaskOp::resolve_true(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void TaskOp::select_sources(const unsigned index,
                                const InstanceRef &target,
                                const InstanceSet &sources,
                                std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < regions.size());
#endif
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances);
      input.region_req_index = index;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_select_task_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_atomic_locks(const unsigned index,
                                     Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<Reservation,bool>::iterator finder = atomic_locks.find(lock);
      if (finder != atomic_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        atomic_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    unsigned TaskOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < parent_req_indexes.size());
#endif
      return parent_req_indexes[idx];
    }

    //--------------------------------------------------------------------------
    VersionInfo& TaskOp::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return (*(new VersionInfo()));
    }

    //--------------------------------------------------------------------------
    const VersionInfo& TaskOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return (*(new VersionInfo()));
    }

    //--------------------------------------------------------------------------
    RegionTreePath& TaskOp::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // This should never be called
      assert(false);
      return (*(new RegionTreePath()));
    }

    //--------------------------------------------------------------------------
    ApEvent TaskOp::compute_sync_precondition(const TraceInfo *info) const
    //--------------------------------------------------------------------------
    {
      ApEvent result;
      if (!wait_barriers.empty() || !grants.empty())
      {
        std::set<ApEvent> sync_preconditions;
        if (!wait_barriers.empty())
        {
          for (std::vector<PhaseBarrier>::const_iterator it = 
                wait_barriers.begin(); it != wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            sync_preconditions.insert(e);
            if (runtime->legion_spy_enabled)
              LegionSpy::log_phase_barrier_wait(unique_op_id, e);
          }
        }
        if (!grants.empty())
        {
          for (std::vector<Grant>::const_iterator it = grants.begin();
                it != grants.end(); it++)
          {
            ApEvent e = it->impl->acquire_grant();
            sync_preconditions.insert(e);
          }
        }
        // For some reason we don't trace these, not sure why
        result = Runtime::merge_events(NULL, sync_preconditions);
        if (!result.exists() ||
            sync_preconditions.find(result) != sync_preconditions.end())
        {
          ApUserEvent rename = Runtime::create_ap_user_event(NULL);
          Runtime::trigger_event(NULL, rename, result);
          result = rename;
        }
      }
      if ((info != NULL) && info->recording)
        info->record_op_sync_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
                                        TaskOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void TaskOp::defer_distribute_task(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      parent_ctx->add_to_distribute_task_queue(this, precondition);
    }

    //--------------------------------------------------------------------------
    RtEvent TaskOp::defer_perform_mapping(RtEvent precondition, MustEpochOp *op,
                                          const DeferMappingArgs *defer_args,
                                          unsigned invocation_count,
                                          std::vector<unsigned> *performed,
                                          std::vector<ApEvent> *effects)
    //--------------------------------------------------------------------------
    {
      const RtUserEvent done_event = (defer_args == NULL) ? 
        Runtime::create_rt_user_event() : defer_args->done_event;
      DeferMappingArgs args(this, op, done_event, invocation_count,
                            performed, effects);
      runtime->issue_runtime_meta_task(args,
          LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return done_event;
    }

    //--------------------------------------------------------------------------
    void TaskOp::defer_launch_task(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      parent_ctx->add_to_launch_task_queue(this, precondition);
    }

    //--------------------------------------------------------------------------
    void TaskOp::enqueue_ready_task(bool use_target_processor,
                                    RtEvent wait_on /*=RtEvent::NO_RT_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (use_target_processor)
        set_current_proc(target_proc);
      if (!wait_on.exists() || wait_on.has_triggered())
      {
        // Need to invoke select task options here for top-level tasks
        if (!options_selected)
          select_task_options(false/*prioritize*/);
        runtime->add_to_ready_queue(current_proc, this);
      }
      else
        parent_ctx->add_to_task_queue(this, wait_on);
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->increment_outstanding();
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->decrement_outstanding();
    } 

    //--------------------------------------------------------------------------
    void TaskOp::perform_privilege_checks(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TASK_PRIVILEGE_CHECK_CALL);
      // First check the index privileges
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        LegionErrorType et = parent_ctx->check_privilege(indexes[idx]);
        switch (et)
        {
          case LEGION_NO_ERROR:
            break;
          case ERROR_BAD_PARENT_INDEX:
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                              "Parent task %s (ID %lld) of task %s "
                              "(ID %lld) "
                              "does not have an index requirement for "
                              "index space %x as a parent of "
                              "child task's index requirement index %d",
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id(), get_task_name(),
                              get_unique_id(), indexes[idx].parent.id, idx)
              break;
            }
          case ERROR_BAD_INDEX_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_NOTSUBSPACE,
                              "Index space %x is not a sub-space "
                              "of parent index space %x for index "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].handle.id,
                              indexes[idx].parent.id, idx,
                              get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_INDEX_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_INDEX_SPACE,
                              "Privileges %x for index space %x "
                              " are not a subset of privileges of parent "
                              "task's privileges for index space "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].privilege,
                              indexes[idx].handle.id, idx,
                              get_task_name(), get_unique_id())
              break;
            }
          default:
            assert(false); // Should never happen
        }
      }
      // Now check the region requirement privileges
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Verify that the requirement is self-consistent
        FieldID bad_field = LEGION_AUTO_GENERATE_ID;
        int bad_index = -1;
        LegionErrorType et = runtime->verify_requirement(regions[idx], 
                                                         bad_field); 
        if ((et == LEGION_NO_ERROR) && !is_index_space && 
            ((regions[idx].handle_type == LEGION_PARTITION_PROJECTION) || 
             (regions[idx].handle_type == LEGION_REGION_PROJECTION)))
          et = ERROR_BAD_PROJECTION_USE;
        // If that worked, then check the privileges with the parent context
        if (et == LEGION_NO_ERROR)
          et = parent_ctx->check_privilege(regions[idx], bad_field, bad_index);
        switch (et)
        {
          case LEGION_NO_ERROR:
            break;
          case ERROR_INVALID_REGION_HANDLE:
            {
              REPORT_LEGION_ERROR(ERROR_INVALID_REGION_HANDLE,
                               "Invalid region handle (%x,%d,%d)"
                               " for region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_INVALID_PARTITION_HANDLE:
            {
              REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_HANDLE,
                               "Invalid partition handle (%x,%d,%d) "
                               "for partition requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PROJECTION_USE:
            {
              REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENT,
                               "Projection region requirement %d used "
                               "in non-index space task %s",
                               idx, get_task_name())
              break;
            }
          case ERROR_NON_DISJOINT_PARTITION:
            {
              REPORT_LEGION_ERROR(ERROR_NONDISJOINT_PARTITION_SELECTED,
                               "Non disjoint partition selected for "
                               "writing region requirement %d of task "
                               "%s.  All projection partitions "
                               "which are not read-only and not reduce "
                               "must be disjoint",
                               idx, get_task_name())
              break;
            }
          case ERROR_FIELD_SPACE_FIELD_MISMATCH:
            {
              FieldSpace sp = 
                (regions[idx].handle_type == LEGION_SINGULAR_PROJECTION) ||
                (regions[idx].handle_type == LEGION_REGION_PROJECTION) ? 
                  regions[idx].region.field_space :
                  regions[idx].partition.field_space;
              REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                               "Field %d is not a valid field of field "
                               "space %d for region %d of task %s "
                               "(ID %lld)",
                               bad_field, sp.id, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_INVALID_INSTANCE_FIELD:
            {
              REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                               "Instance field %d is not one of the "
                               "privilege fields for region %d of "
                               "task %s (ID %lld)",
                               bad_field, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_DUPLICATE_INSTANCE_FIELD:
            {
              REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                               "Instance field %d is a duplicate for "
                               "region %d of task %s (ID %lld)",
                               bad_field, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_BAD_PARENT_REGION:
            {
              if (bad_index < 0) 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "no 'parent' region had that name.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx)
              else if (bad_field == LEGION_AUTO_GENERATE_ID) 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "parent requirement %d did not have "
                                 "sufficient privileges.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx, bad_index)
              else 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "parent requirement %d was missing field %d.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx,
                                 bad_index, bad_field)
              break;
            }
          case ERROR_BAD_REGION_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                               "Region (%x,%x,%x) is not a "
                               "sub-region of parent region "
                               "(%x,%x,%x) for region requirement %d of "
                               "task %s (ID %lld)",
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id,
                               PRINT_REG(regions[idx].parent), idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PARTITION_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_PARTITION_NOT_SUBPARTITION,
                               "Partition (%x,%x,%x) is not a "
                               "sub-partition of parent region "
                               "(%x,%x,%x) for region "
                               "requirement %d task %s (ID %lld)",
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id,
                               PRINT_REG(regions[idx].parent), idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_REGION_TYPE:
            {
              REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_TASK,
                               "Region requirement %d of task %s "
                               "(ID %lld) "
                               "cannot find privileges for field %d in "
                               "parent task",
                               idx, get_task_name(),
                               get_unique_id(), bad_field)
              break;
            }
          case ERROR_BAD_REGION_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_REGION_NOTSUBSET,
                               "Privileges %x for region "
                               "(%x,%x,%x) are not a subset of privileges "
                               "of parent task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege,
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PARTITION_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_PARTITION_NOTSUBSET,
                               "Privileges %x for partition (%x,%x,%x) "
                               "are not a subset of privileges of parent "
                               "task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege,
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          default:
            assert(false); // Should never happen
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::find_early_mapped_region(unsigned idx, InstanceSet &ref)
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,InstanceSet>::const_iterator finder =
        early_mapped_regions.find(idx);
      if (finder != early_mapped_regions.end())
        ref = finder->second;
    }

    //--------------------------------------------------------------------------
    void TaskOp::clone_task_op_from(TaskOp *rhs, Processor p, 
                                    bool can_steal, bool duplicate_args)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CLONE_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(p.exists());
#endif
      // From Operation
      this->parent_ctx = rhs->parent_ctx;
      this->context_index = rhs->context_index;
      this->execution_fence_event = rhs->get_execution_fence_event();
      // Don't register this an operation when setting the must epoch info
      if (rhs->must_epoch != NULL)
        this->set_must_epoch(rhs->must_epoch, rhs->must_epoch_index,
                             false/*do registration*/);
      // From Memoizable
      this->trace_local_id = rhs->trace_local_id;
      // From Task
      this->task_id = rhs->task_id;
      this->indexes = rhs->indexes;
      this->regions = rhs->regions;
      this->futures = rhs->futures;
      this->grants = rhs->grants;
      this->wait_barriers = rhs->wait_barriers;
      this->arrive_barriers = rhs->arrive_barriers;
      this->arglen = rhs->arglen;
      if (rhs->arg_manager != NULL)
      {
        if (duplicate_args)
        {
#ifdef DEBUG_LEGION
          assert(arg_manager == NULL);
#endif
          this->arg_manager = new AllocManager(this->arglen); 
          this->arg_manager->add_reference();
          this->args = this->arg_manager->get_allocation();
          memcpy(this->args, rhs->args, this->arglen);
        }
        else
        {
          // No need to actually do the copy in this case
          this->arg_manager = rhs->arg_manager; 
          this->arg_manager->add_reference();
          this->args = arg_manager->get_allocation();
        }
      }
      else if (arglen > 0)
      {
        // If there is no argument manager then we do the copy no matter what
        this->args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,rhs->args,arglen);
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      if (rhs->mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(rhs->mapper_data != NULL);
#endif
        this->mapper_data_size = rhs->mapper_data_size;
        this->mapper_data = malloc(this->mapper_data_size);
        memcpy(this->mapper_data, rhs->mapper_data, this->mapper_data_size);
      }
      this->is_index_space = rhs->is_index_space;
      this->orig_proc = rhs->orig_proc;
      this->current_proc = rhs->current_proc;
      this->steal_count = rhs->steal_count;
      this->stealable = can_steal;
      this->speculated = rhs->speculated;
      this->parent_task = rhs->parent_task;
      this->map_origin = rhs->map_origin;
      this->sharding_space = rhs->sharding_space;
      this->request_valid_instances = rhs->request_valid_instances;
      // From TaskOp
      this->atomic_locks = rhs->atomic_locks;
      this->early_mapped_regions = rhs->early_mapped_regions;
      this->parent_req_indexes = rhs->parent_req_indexes;
      this->current_proc = rhs->current_proc;
      this->target_proc = p;
      this->true_guard = rhs->true_guard;
      this->false_guard = rhs->false_guard;
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_grants(const std::vector<Grant> &requested_grants)
    //--------------------------------------------------------------------------
    {
      grants = requested_grants;
      const ApEvent grant_pre = get_completion_event();
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(grant_pre);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_arrival_barriers(
                                const std::vector<PhaseBarrier> &phase_barriers)
    //--------------------------------------------------------------------------
    {
      const ApEvent arrive_pre = get_completion_event();
      for (std::vector<PhaseBarrier>::const_iterator it = 
            phase_barriers.begin(); it != phase_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
        Runtime::phase_barrier_arrive(*it, 1/*count*/, arrive_pre);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_phase_barrier_arrival(unique_op_id, it->phase_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_point_region_requirements(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, COMPUTE_POINT_REQUIREMENTS_CALL);
      // Update the region requirements for this point
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type != LEGION_SINGULAR_PROJECTION)
        {
          ProjectionFunction *function = 
            runtime->find_projection_function(regions[idx].projection);
          if (function->is_invertible)
            assert(false); // TODO: implement dependent launches for inline
          regions[idx].region = 
            function->project_point(this, idx, runtime, index_point);
          // Update the region requirement kind 
          regions[idx].handle_type = LEGION_SINGULAR_PROJECTION;
        }
        // Check to see if the region is a NO_REGION,
        // if it is then switch the privilege to NO_ACCESS
        if (regions[idx].region == LogicalRegion::NO_REGION)
        {
          regions[idx].privilege = LEGION_NO_ACCESS;
          continue;
        }
      }
      complete_point_projection(); 
    }

    //--------------------------------------------------------------------------
    void TaskOp::complete_point_projection(void)
    //--------------------------------------------------------------------------
    {
      SingleTask *single_task = dynamic_cast<SingleTask*>(this);
      if (single_task != NULL)
        single_task->update_no_access_regions();
      // Log our requirements that we computed
      if (runtime->legion_spy_enabled)
      {
        UniqueID our_uid = get_unique_id();
        for (unsigned idx = 0; idx < regions.size(); idx++)
          log_requirement(our_uid, idx, regions[idx]);
      }
#ifdef DEBUG_LEGION
      {
        std::vector<RegionTreePath> privilege_paths(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
          initialize_privilege_path(privilege_paths[idx], regions[idx]);
        perform_intra_task_alias_analysis(false/*tracing*/, NULL/*trace*/,
                                          privilege_paths);
      }
#endif
    } 

    //--------------------------------------------------------------------------
    bool TaskOp::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      if (is_origin_mapped())
        return false;
      if (!is_remote())
        early_map_task();
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskOp::perform_intra_task_alias_analysis(bool is_tracing,
               LegionTrace *trace, std::vector<RegionTreePath> &privilege_paths)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INTRA_TASK_ALIASING_CALL);
#ifdef DEBUG_LEGION
      assert(regions.size() == privilege_paths.size());
#endif
      // Quick out if we've already traced this
      if (!is_tracing && (trace != NULL))
      {
        trace->replay_aliased_children(privilege_paths);
        return;
      }
      std::map<RegionTreeID,std::vector<unsigned> > tree_indexes;
      // Find the indexes of requirements with the same tree
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        tree_indexes[regions[idx].parent.get_tree_id()].push_back(idx);
      }
      // Iterate over the trees with multiple requirements
      for (std::map<RegionTreeID,std::vector<unsigned> >::const_iterator 
            tree_it = tree_indexes.begin(); 
            tree_it != tree_indexes.end(); tree_it++)
      {
        const std::vector<unsigned> &indexes = tree_it->second;
        if (indexes.size() <= 1)
          continue;
        // Get the field masks for each of the requirements
        LegionVector<FieldMask> field_masks(indexes.size());
        std::vector<IndexTreeNode*> index_nodes(indexes.size());
        {
          FieldSpaceNode *field_space_node = 
           runtime->forest->get_node(regions[indexes[0]].parent)->column_source;
          for (unsigned idx = 0; idx < indexes.size(); idx++)
          {
            field_masks[idx] = field_space_node->get_field_mask(
                                        regions[indexes[idx]].privilege_fields);
            if (regions[indexes[idx]].handle_type == 
                LEGION_PARTITION_PROJECTION)
              index_nodes[idx] = runtime->forest->get_node(
                        regions[indexes[idx]].partition.get_index_partition());
            else
              index_nodes[idx] = runtime->forest->get_node(
                        regions[indexes[idx]].region.get_index_space());
          }
        }
        // Find the sets of fields which are interfering
        for (unsigned i = 1; i < indexes.size(); i++)
        {
          RegionUsage usage1(regions[indexes[i]]);
          for (unsigned j = 0; j < i; j++)
          {
            FieldMask overlap = field_masks[i] & field_masks[j];
            // No field overlap, so there is nothing to do
            if (!overlap)
              continue;
            // No check for region overlap
            IndexTreeNode *common_ancestor = NULL;
            if (runtime->forest->are_disjoint_tree_only(index_nodes[i],
                  index_nodes[j], common_ancestor))
              continue;
#ifdef DEBUG_LEGION
            assert(common_ancestor != NULL); // should have a counterexample
#endif
            // Get the interference kind and report it if it is bad
            RegionUsage usage2(regions[indexes[j]]);
            DependenceType dtype = check_dependence_type<false>(usage1, usage2);
            // We can only reporting interfering requirements precisely
            // if at least one of these is not a projection requireemnts
            if (((dtype == LEGION_TRUE_DEPENDENCE) || 
                 (dtype == LEGION_ANTI_DEPENDENCE)) &&
                ((regions[indexes[i]].handle_type == 
                  LEGION_SINGULAR_PROJECTION) ||
                 (regions[indexes[j]].handle_type == 
                  LEGION_SINGULAR_PROJECTION)))
              report_interfering_requirements(indexes[j], indexes[i]);
            // Special case, if the parents are not the same,
            // then we don't have to do anything cause their
            // path will not overlap
            if (regions[indexes[i]].parent != regions[indexes[j]].parent)
              continue;
            // Record it in the earlier path as the latter path doesn't matter
            privilege_paths[indexes[j]].record_aliased_children(
                                    common_ancestor->depth, overlap);
            // If we have a trace, record the aliased requirements
            if (trace != NULL)
              trace->record_aliased_children(indexes[j], 
                                             common_ancestor->depth, overlap);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::validate_variant_selection(MapperManager *local_mapper,
                              VariantImpl *impl, Processor::Kind kind, 
                              const std::deque<InstanceSet> &physical_instances,
                              const char *mapper_call_name) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, VALIDATE_VARIANT_SELECTION_CALL);
      // Check the layout constraints first
      const TaskLayoutConstraintSet &layout_constraints = 
        impl->get_layout_constraints();
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.begin(); it != 
            layout_constraints.layouts.end(); it++)
      {
        // Might have constraints for extra region requirements
        if (it->first >= physical_instances.size())
          continue;
        const InstanceSet &instances = physical_instances[it->first]; 
        if (IS_NO_ACCESS(regions[it->first]))
          continue;
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(it->second);
        // If we don't have any fields then this constraint isn't
        // going to apply to any actual instances
        const std::vector<FieldID> &field_vec = 
          constraints->field_constraint.field_set;
        FieldMask constraint_mask;
        if (!field_vec.empty())
        {
          FieldSpaceNode *field_node = runtime->forest->get_node(
                              regions[it->first].region.get_field_space());
          std::set<FieldID> field_set(field_vec.begin(), field_vec.end());
          constraint_mask = field_node->get_field_mask(field_set);
        }
        else
          constraint_mask = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        const LayoutConstraint *conflict_constraint = NULL;
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          const InstanceRef &ref = instances[idx];
          // Check to see if we have any fields which overlap
          const FieldMask overlap = constraint_mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          InstanceManager *manager = ref.get_manager();
          if (manager->conflicts(constraints, index_point,&conflict_constraint))
            break;
          // Check to see if we need an exact match on the layouts
          if (constraints->specialized_constraint.is_exact())
          {
            std::vector<LogicalRegion> regions_to_check(1, 
                                regions[it->first].region);
            PhysicalManager *phy = manager->as_physical_manager();
            if (!phy->meets_regions(regions_to_check,true/*tight*/))
            {
              conflict_constraint = &constraints->specialized_constraint;
              break;
            }
          }
        }
        if (conflict_constraint != NULL)
        {
          const char *constraint_names[] = {
#define CONSTRAINT_NAMES(name, desc) desc,
            LEGION_LAYOUT_CONSTRAINT_KINDS(CONSTRAINT_NAMES)
#undef CONSTRAINT_NAMES
          };
          const char *constraint_name = 
            constraint_names[conflict_constraint->get_constraint_kind()];
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s selected variant "
                        "%d for task %s (ID %lld). But instance selected "
                        "for region requirement %d fails to satisfy the "
                        "corresponding %s layout constraint.", 
                        local_mapper->get_mapper_name(), impl->vid,
                        get_task_name(), get_unique_id(), it->first,
                        constraint_name)
        }
      }
      // Now we can test against the execution constraints
      const ExecutionConstraintSet &execution_constraints = 
        impl->get_execution_constraints();
      // TODO: Check ISA, resource, and launch constraints
      // First check the processor constraint
      if (execution_constraints.processor_constraint.is_valid())
      {
        // If the constraint is a no processor constraint we can ignore it
        if (!execution_constraints.processor_constraint.can_use(kind))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output. Mapper %s selected variant %d "
                      "for task %s (ID %lld). However, this variant does not "
                      "permit running on processors of kind %s.",
                      local_mapper->get_mapper_name(),
                      impl->vid, get_task_name(), get_unique_id(),
                      Processor::get_kind_name(kind))
      }
      // Then check the colocation constraints
      for (std::vector<ColocationConstraint>::const_iterator con_it = 
            execution_constraints.colocation_constraints.begin(); con_it !=
            execution_constraints.colocation_constraints.end(); con_it++)
      {
        if (con_it->indexes.size() < 2)
          continue;
        unsigned idx = 0;
        bool first = true;
        RegionTreeID tree_id = 0;
        FieldSpaceNode *field_space_node = NULL;
        std::map<unsigned/*field index*/,
          std::pair<InstanceManager*,unsigned> > colocation_instances;
        for (std::set<unsigned>::const_iterator iit = con_it->indexes.begin();
              iit != con_it->indexes.end(); iit++, idx++)
        {
          const RegionRequirement &req = regions[*iit];
          if (first)
          {
            first = false;
            tree_id = req.parent.get_tree_id();
            field_space_node = runtime->forest->get_node(
                                req.parent.get_field_space());
            const InstanceSet &insts = physical_instances[*iit];
            FieldMask colocation_mask;
            if (con_it->fields.empty())
            {
              // If there are no explicit fields then we are
              // just going through and checking all of them
              for (std::set<FieldID>::const_iterator it = 
                    req.privilege_fields.begin(); it != 
                    req.privilege_fields.end(); it++)
              {
                unsigned index = field_space_node->get_field_index(*it);
                colocation_instances[index] = 
                  std::pair<InstanceManager*,unsigned>(NULL, *iit);
                colocation_mask.set_bit(index);
              }
            }
            else
            {
              for (std::set<FieldID>::const_iterator it = 
                    con_it->fields.begin(); it != con_it->fields.end(); it++)
              {
                if (req.privilege_fields.find(*it) == 
                    req.privilege_fields.end())
                  continue;
                unsigned index = field_space_node->get_field_index(*it);
                colocation_instances[index] = 
                  std::pair<InstanceManager*,unsigned>(NULL, *iit);
                colocation_mask.set_bit(index);
              }
            }
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef &ref = insts[idx];
              const FieldMask overlap = 
                colocation_mask & ref.get_valid_fields();
              if (!overlap)
                continue;
              InstanceManager *manager = ref.get_manager();
              if (manager->is_virtual_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output. Mapper %s selected a virtual "
                    "instance for region requirement %d of task %s (UID %lld), "
                    "but also selected variant %d which contains a colocation "
                    "constraint for this region requirement. It is illegal to "
                    "request a virtual mapping for a region requirement with a "
                    "colocation constraint.", local_mapper->get_mapper_name(),
                    *iit, get_task_name(), get_unique_id(), impl->vid)
              int index = overlap.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned,
                  std::pair<InstanceManager*,unsigned> >::iterator finder = 
                    colocation_instances.find(index);
#ifdef DEBUG_LEGION
                assert(finder != colocation_instances.end());
                assert(finder->second.first == NULL);
                assert(finder->second.second == *iit);
#endif
                finder->second.first = manager;
                index = overlap.find_next_set(index+1);
              }
            }
          }
          else
          {
            // check to make sure that all these region requirements have
            // the same region tree ID.
            if (req.parent.get_tree_id() != tree_id)
              REPORT_LEGION_ERROR(ERROR_INVALID_LOCATION_CONSTRAINT,
                            "Invalid location constraint. Location constraint "
                            "specified on region requirements %d and %d of "
                            "variant %d of task %s, but region requirements "
                            "contain regions that from different region trees "
                            "(%d and %d). Colocation constraints must always "
                            "be specified on region requirements with regions "
                            "from the same region tree.", 
                            *(con_it->indexes.begin()), *iit, impl->vid,
                            get_task_name(), tree_id, 
                            req.parent.get_tree_id())
            const InstanceSet &insts = physical_instances[*iit];
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef &ref = insts[idx];
              InstanceManager *manager = ref.get_manager();
              if (manager->is_virtual_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output. Mapper %s selected a virtual "
                    "instance for region requirement %d of task %s (UID %lld), "
                    "but also selected variant %d which contains a colocation "
                    "constraint for this region requirement. It is illegal to "
                    "request a virtual mapping for a region requirement with a "
                    "colocation constraint.", local_mapper->get_mapper_name(),
                    *iit, get_task_name(), get_unique_id(), impl->vid)
              const FieldMask &inst_mask = ref.get_valid_fields();
              std::vector<FieldID> field_names;
              field_space_node->get_field_set(inst_mask,parent_ctx,field_names);
              unsigned name_index = 0;
              int index = inst_mask.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned,
                  std::pair<InstanceManager*,unsigned> >::const_iterator
                    finder = colocation_instances.find(index);
                if (finder != colocation_instances.end())
                {
                  if (finder->second.first->get_instance(index_point) != 
                      manager->get_instance(index_point))
                    REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output. Mapper %s selected variant "
                          "%d for task %s (ID %lld). However, this variant "
                          "requires that field %d of region requirements %d be "
                          "co-located with prior requirement %d but it is not. "
                          "Requirement %d mapped to instance " IDFMT " while "
                          "prior requirement %d mapped to instance " IDFMT "",
                          local_mapper->get_mapper_name(), impl->vid, 
                          get_task_name(), get_unique_id(), 
                          field_names[name_index], *iit, finder->second.second,
                          *iit, manager->get_instance(index_point).id, 
                          finder->second.second,
                          finder->second.first->get_instance(index_point).id)
                }
                else
                {
                  if (!con_it->fields.empty())
                  {
                    if (con_it->fields.find(field_names[name_index]) !=
                        con_it->fields.end())
                      colocation_instances[index] = 
                        std::make_pair(manager, *iit);
                  }
                  else
                    colocation_instances[index] = std::make_pair(manager, *iit);
                }
                index = inst_mask.find_next_set(index+1);
                name_index++;
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_parent_indexes(TaskContext *alt_context/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      parent_req_indexes.resize(regions.size());
      TaskContext *use_ctx = (alt_context == NULL) ? parent_ctx : alt_context;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        int parent_index = use_ctx->find_parent_region_req(regions[idx]);
        if (parent_index < 0)
          REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                           "Parent task %s (ID %lld) of task %s "
                           "(ID %lld) does not have a region "
                           "requirement for region "
                           "(%x,%x,%x) as a parent of child task's "
                           "region requirement index %d",
                           use_ctx->get_task_name(), 
                           use_ctx->get_unique_id(),
                           get_task_name(), get_unique_id(),
                           regions[idx].parent.index_space.id,
                           regions[idx].parent.field_space.id, 
                           regions[idx].parent.tree_id, idx)
        parent_req_indexes[idx] = parent_index;
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_complete(void)
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock); 
#ifdef DEBUG_LEGION
        assert(!children_complete);
        // Small race condition here which is alright as
        // long as we haven't committed yet
        assert(!children_commit || !commit_received);
#endif
        children_complete = true;
        task_complete = complete_received;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_committed(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        // There is a small race condition here which is alright
        // as long as we haven't committed yet
        assert(children_complete || !commit_received);
        assert(!children_commit);
#endif
        children_commit = true;
        task_commit = commit_received;
      }
      if (task_commit)
        trigger_task_commit();
    } 

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::log_requirement(UniqueID uid, unsigned idx,
                                            const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                       (req.handle_type == LEGION_REGION_PROJECTION);
      const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                        (req.handle_type == LEGION_PARTITION_PROJECTION); 

      LegionSpy::log_logical_requirement(uid, idx, reg,
          reg ? req.region.index_space.id :
                req.partition.index_partition.id,
          reg ? req.region.field_space.id :
                req.partition.field_space.id,
          reg ? req.region.tree_id : 
                req.partition.tree_id,
          req.privilege, req.prop, req.redop, req.parent.index_space.id);
      LegionSpy::log_requirement_fields(uid, idx, req.privilege_fields);
      if (proj)
        LegionSpy::log_requirement_projection(uid, idx, req.projection);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Task Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTaskOp::RemoteTaskOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : ExternalTask(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp::RemoteTaskOp(const RemoteTaskOp &rhs)
      : ExternalTask(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp::~RemoteTaskOp(void)
    //--------------------------------------------------------------------------
    {
      if (args != NULL)
      {
        if (arg_manager != NULL)
        {
          // If the arg manager is not NULL then we delete the
          // argument manager and just zero out the arguments
          if (arg_manager->remove_reference())
            delete (arg_manager);
          arg_manager = NULL;
        }
        else
          legion_free(TASK_ARGS_ALLOC, args, arglen);
      }
      if (local_args != NULL)
        free(local_args);
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp& RemoteTaskOp::operator=(const RemoteTaskOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteTaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    size_t RemoteTaskOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::set_context_index(size_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteTaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    bool RemoteTaskOp::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteTaskOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_task();
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    Domain RemoteTaskOp::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TASK_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteTaskOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::select_sources(const unsigned index,
                                      const InstanceRef &target,
                                      const InstanceSet &sources,
                                      std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking);
        return;
      }
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(sources, input.source_instances); 
      prepare_for_mapping(target, input.target);
      input.region_req_index = index;
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_task_sources(this, &input, &output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::unpack(Deserializer &derez,
                              ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      unpack_external_task(derez, runtime, &mutator);
      unpack_profiling_requests(derez);
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(Runtime *rt)
      : TaskOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SingleTask::activate_single(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_SINGLE_CALL);
      activate_task();
      task_effects_complete = ApEvent::NO_AP_EVENT;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      selected_variant = 0;
      task_priority = 0;
      perform_postmap = false;
      first_mapping = true;
      execution_context = NULL;
      remote_trace_recorder = NULL;
      leaf_cached = false;
      inner_cached = false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate_single(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_SINGLE_CALL);
      deactivate_task();
      target_processors.clear();
      physical_instances.clear();
      virtual_mapped.clear();
      no_access_regions.clear();
      version_infos.clear();
      intra_space_mapping_dependences.clear();
      map_applied_conditions.clear();
      task_profiling_requests.clear();
      copy_profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
      untracked_valid_regions.clear();
      if ((execution_context != NULL) && execution_context->remove_reference())
        delete execution_context;
#ifdef DEBUG_LEGION
      premapped_instances.clear();
      assert(!deferred_complete_mapping.exists());
      assert(!single_task_termination.exists());
      assert(remote_trace_recorder == NULL); 
#endif
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_leaf(void) const
    //--------------------------------------------------------------------------
    {
      if (!leaf_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_leaf_result = var->is_leaf();
        leaf_cached = true;
      }
      return is_leaf_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_inner(void) const
    //--------------------------------------------------------------------------
    {
      if (!inner_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_inner_result = var->is_inner();
        inner_cached = true;
      }
      return is_inner_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_created_region(unsigned index) const
    //--------------------------------------------------------------------------
    {
      return (index >= regions.size());
    }

    //--------------------------------------------------------------------------
    void SingleTask::update_no_access_regions(void)
    //--------------------------------------------------------------------------
    {
      no_access_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        no_access_regions[idx] = IS_NO_ACCESS(regions[idx]) || 
                                  regions[idx].privilege_fields.empty();
    } 

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_SINGLE_TASK_CALL);
      RezCheck z(rez);
      pack_base_task(rez, target);
      if (is_origin_mapped())
      {
        rez.serialize(selected_variant);
        rez.serialize<size_t>(target_processors.size());
        for (unsigned idx = 0; idx < target_processors.size(); idx++)
          rez.serialize(target_processors[idx]);
        for (unsigned idx = 0; idx < regions.size(); idx++)
          rez.serialize<bool>(virtual_mapped[idx]);
        rez.serialize(deferred_complete_mapping);
        deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        rez.serialize(single_task_termination);
        single_task_termination = ApUserEvent::NO_AP_USER_EVENT;
        rez.serialize<size_t>(physical_instances.size());
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
          physical_instances[idx].pack_references(rez);
        rez.serialize<size_t>(task_profiling_requests.size());
        for (unsigned idx = 0; idx < task_profiling_requests.size(); idx++)
          rez.serialize(task_profiling_requests[idx]);
        rez.serialize<size_t>(copy_profiling_requests.size());
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        if (!task_profiling_requests.empty() || !copy_profiling_requests.empty())
          rez.serialize(profiling_priority);
      }
      else
      { 
        if (!deferred_complete_mapping.exists())
        {
#ifdef DEBUG_LEGION
          assert(!is_remote()); // should only happen on the owner
#endif
          // Make a user event to send remotely to serve as the 
          // mapping completion trigger
          RtUserEvent remote_deferred_complete_mapping = 
            Runtime::create_rt_user_event();
          rez.serialize(remote_deferred_complete_mapping);
          // We can do the trigger now and defer it
          complete_mapping(remote_deferred_complete_mapping);
        }
        else
        {
          rez.serialize(deferred_complete_mapping);
          // Clear it once we've packed it up
          deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        }
        if (memo_state == MEMO_RECORD)
        {
#ifdef DEBUG_LEGION
          assert(tpl != NULL);
#endif
          if (is_remote())
            rez.serialize(remote_collect_event);
          else
            rez.serialize(tpl->get_collect_event());
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez,
                                        std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_SINGLE_TASK_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events);
#ifdef DEBUG_LEGION
      assert(!deferred_complete_mapping.exists());
#endif
      if (map_origin)
      {
        derez.deserialize(selected_variant);
        size_t num_target_processors;
        derez.deserialize(num_target_processors);
        target_processors.resize(num_target_processors);
        for (unsigned idx = 0; idx < num_target_processors; idx++)
          derez.deserialize(target_processors[idx]);
        virtual_mapped.resize(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          bool result;
          derez.deserialize(result);
          virtual_mapped[idx] = result;
        }
        derez.deserialize(deferred_complete_mapping);
        complete_mapping(deferred_complete_mapping);
        derez.deserialize(single_task_termination);
        size_t num_phy;
        derez.deserialize(num_phy);
        physical_instances.resize(num_phy);
        for (unsigned idx = 0; idx < num_phy; idx++)
          physical_instances[idx].unpack_references(runtime,
                                                    derez, ready_events);
        size_t num_task_requests;
        derez.deserialize(num_task_requests);
        if (num_task_requests > 0)
        {
          task_profiling_requests.resize(num_task_requests);
          for (unsigned idx = 0; idx < num_task_requests; idx++)
            derez.deserialize(task_profiling_requests[idx]);
        }
        size_t num_copy_requests;
        derez.deserialize(num_copy_requests);
        if (num_copy_requests > 0)
        {
          copy_profiling_requests.resize(num_copy_requests);
          for (unsigned idx = 0; idx < num_copy_requests; idx++)
            derez.deserialize(copy_profiling_requests[idx]);
        }
        if (!task_profiling_requests.empty() || 
            !copy_profiling_requests.empty())
          derez.deserialize(profiling_priority);
      }
      else
      {
        derez.deserialize(deferred_complete_mapping);
        if (memo_state == MEMO_RECORD)
        {
#ifdef DEBUG_LEGION
          assert(tpl != NULL);
#endif
          derez.deserialize(remote_collect_event);
        }
      }
      update_no_access_regions();
    } 

    //--------------------------------------------------------------------------
    void SingleTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TRIGGER_SINGLE_CALL);
      if (is_remote())
      {
        if (distribute_task())
        {
          // Still local
          if (is_origin_mapped())
          {
            // Remote and origin mapped means
            // we were already mapped so we can
            // just launch the task
            launch_task();
          }
          else
          {
            // Remote but still need to map
            const RtEvent done_mapping = perform_mapping();
            if (done_mapping.exists() && !done_mapping.has_triggered())
              defer_launch_task(done_mapping);
            else
              launch_task();
          }
        }
        // otherwise it was sent away
      }
      else
      {
        // Not remote
        early_map_task();
        // See if we have a must epoch in which case
        // we can simply record ourselves and we are done
        if (must_epoch == NULL)
        {
#ifdef DEBUG_LEGION
          assert(target_proc.exists());
#endif
          // See if this task is going to be sent
          // remotely in which case we need to do the
          // mapping now, otherwise we can defer it
          // until the task ends up on the target processor
          if (is_origin_mapped())
          {
            if (first_mapping)
            {
              first_mapping = false;
              const RtEvent done_mapping = perform_mapping();
              if (!done_mapping.exists() || done_mapping.has_triggered())
              {
                if (distribute_task())
                  launch_task();
              }
              else
                defer_distribute_task(done_mapping);
            }
            else if (distribute_task())
              launch_task();
          }
          else
          {
            if (distribute_task())
            {
              // Still local so try mapping and launching
              const RtEvent done_mapping = perform_mapping();
              if (!done_mapping.exists() || done_mapping.has_triggered())
                launch_task();
              else
                defer_launch_task(done_mapping);
            }
          }
        }
        else
          must_epoch->register_single_task(this, must_epoch_index);
      }
    } 

    //--------------------------------------------------------------------------
    void SingleTask::perform_inlining(VariantImpl *variant,
                                const std::deque<InstanceSet> &parent_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_instances.size() == regions.size());
#endif
      selected_variant = variant->vid;
      target_processors.push_back(current_proc);
      physical_instances = parent_instances;
      virtual_mapped.resize(regions.size());
      no_access_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        virtual_mapped[idx] = false;
        no_access_regions[idx] = IS_NO_ACCESS(regions[idx]);
      }
      complete_mapping();
      // Now we can launch this task right inline in this thread
      launch_task(true/*inline*/); 
    }

    //--------------------------------------------------------------------------
    RtEvent SingleTask::perform_versioning_analysis(const bool post_mapper)
    //--------------------------------------------------------------------------
    {
      if (is_replaying())
        return RtEvent::NO_RT_EVENT;
      // If we're remote and origin mapped, then we are already done
      if (is_remote() && is_origin_mapped())
        return RtEvent::NO_RT_EVENT;
#ifdef DEBUG_LEGION
      assert(version_infos.empty() || (version_infos.size() == regions.size()));
#endif
      version_infos.resize(regions.size());
      std::set<RtEvent> ready_events;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (no_access_regions[idx] || 
            (post_mapper && virtual_mapped[idx]) ||
            (early_mapped_regions.find(idx) != early_mapped_regions.end()))
          continue;
        VersionInfo &version_info = version_infos[idx];
        if (version_info.has_version_info())
          continue;
        runtime->forest->perform_versioning_analysis(this, idx, regions[idx],
                                                 version_info, ready_events);
      }
      if (!ready_events.empty())
        return Runtime::merge_events(ready_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_map_task_input(Mapper::MapTaskInput &input,
                                               Mapper::MapTaskOutput &output,
                                               MustEpochOp *must_epoch_owner,
                                               std::vector<InstanceSet> &valid)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INITIALIZE_MAP_TASK_CALL);
      // Do the traversals for all the non-early mapped regions and find
      // their valid instances, then fill in the mapper input structure
      valid.resize(regions.size());
      input.valid_instances.resize(regions.size());
      output.chosen_instances.resize(regions.size());
      // If we have must epoch owner, we have to check for any 
      // constrained mappings which must be heeded
      if (must_epoch_owner != NULL)
        must_epoch_owner->must_epoch_map_task_callback(this, input, output);
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any early mapped regions
        std::map<unsigned,InstanceSet>::const_iterator early_mapped_finder = 
          early_mapped_regions.find(idx);
        if (early_mapped_finder != early_mapped_regions.end())
        {
          input.premapped_regions.push_back(idx);
          // Still fill in the valid regions so that mappers can use
          // the instance names for constraints
          prepare_for_mapping(early_mapped_finder->second, 
                              input.valid_instances[idx]);
          // We can also copy them over to the output too
          output.chosen_instances[idx] = input.valid_instances[idx];
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
          continue;
        InstanceSet &current_valid = valid[idx];
        if (request_valid_instances)
          runtime->forest->physical_premap_region(this, idx, regions[idx],
                version_infos[idx], current_valid, map_applied_conditions);
        // See if we've already got an output from a must-epoch mapping
        if (!output.chosen_instances[idx].empty())
        {
#ifdef DEBUG_LEGION
          assert(must_epoch_owner != NULL);
#endif
          // We can skip this since we already know the result
          continue;
        }
        // Now we can prepare this for mapping,
        // filter for visible memories if necessary
        if (regions[idx].is_no_access())
          prepare_for_mapping(current_valid, input.valid_instances[idx]);
        // There are no valid instances for reduction-only cases
        else if (regions[idx].privilege != LEGION_REDUCE)
          prepare_for_mapping(current_valid, visible_memories,
                              input.valid_instances[idx]);
      }
#ifdef DEBUG_LEGION
      // Save the inputs for premapped regions so we can check them later
      if (!input.premapped_regions.empty())
      {
        for (std::vector<unsigned>::const_iterator it = 
              input.premapped_regions.begin(); it !=
              input.premapped_regions.end(); it++)
          premapped_instances[*it] = output.chosen_instances[*it];
      }
#endif
      // Prepare the output too
      output.chosen_instances.resize(regions.size());
      output.chosen_variant = 0;
      output.postmap_task = false;
      output.task_priority = 0;
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_map_task_output(Mapper::MapTaskInput &input,
                                              Mapper::MapTaskOutput &output,
                                              MustEpochOp *must_epoch_owner,
                                              std::vector<InstanceSet> &valid)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FINALIZE_MAP_TASK_CALL);
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      // first check the processors to make sure they are all on the
      // same node and of the same kind, if we know we have a must epoch
      // owner then we also know there is only one valid choice
      if (must_epoch_owner == NULL)
      {
        if (output.target_procs.empty())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_EMPTY_OUTPUT_TARGET,
                          "Empty output target_procs from call to 'map_task' "
                          "by mapper %s for task %s (ID %lld). Adding the "
                          "'target_proc' " IDFMT " as the default.",
                          mapper->get_mapper_name(), get_task_name(),
                          get_unique_id(), this->target_proc.id);
          output.target_procs.push_back(this->target_proc);
        }
        else if (runtime->separate_runtime_instances && 
                  (output.target_procs.size() > 1))
        {
          // Ignore additional processors in separate runtime instances
          output.target_procs.resize(1);
        }
        if (!runtime->unsafe_mapper)
          validate_target_processors(output.target_procs);
        // Save the target processors from the output
        target_processors = output.target_procs;
        target_proc = target_processors.front();
      }
      else
      {
        if (output.target_procs.size() > 1)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_SPURIOUS_TARGET,
                          "Ignoring spurious additional target processors "
                          "requested in 'map_task' for task %s (ID %lld) "
                          "by mapper %s because task is part of a must "
                          "epoch launch.", get_task_name(), get_unique_id(),
                          mapper->get_mapper_name());
        }
        if (!output.target_procs.empty() && 
                 (output.target_procs[0] != this->target_proc))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_PROCESSOR_REQUEST,
                          "Ignoring processor request of " IDFMT " for "
                          "task %s (ID %lld) by mapper %s because task "
                          "has already been mapped to processor " IDFMT
                          " as part of a must epoch launch.", 
                          output.target_procs[0].id, get_task_name(), 
                          get_unique_id(), mapper->get_mapper_name(),
                          this->target_proc.id);
        }
        // Only one valid choice in this case, ignore everything else
        target_processors.push_back(this->target_proc);
      }
      // See whether the mapper picked a variant or a generator
      VariantImpl *variant_impl = NULL;
      if (output.chosen_variant > 0)
      {
        variant_impl = runtime->find_variant_impl(task_id, 
                                output.chosen_variant, true/*can fail*/);
      }
      else // TODO: invoke a generator if one exists
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper specified an invalid task variant "
                      "of ID 0 for task %s (ID %lld), but Legion does not yet "
                      "support task generators.", "map_task", 
                      mapper->get_mapper_name(), 
                      get_task_name(), get_unique_id())
      if (variant_impl == NULL)
        // If we couldn't find or make a variant that is bad
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper failed to specify a valid "
                      "task variant or generator capable of create a variant "
                      "implementation of task %s (ID %lld).",
                      "map_task", mapper->get_mapper_name(), get_task_name(),
                      get_unique_id())
      // Save variant validation until we know which instances we'll be using 
#ifdef DEBUG_LEGION
      // Check to see if any premapped region mappings changed
      if (!premapped_instances.empty())
      {
        for (std::map<unsigned,std::vector<Mapping::PhysicalInstance> >::
              const_iterator it = premapped_instances.begin(); it !=
              premapped_instances.end(); it++)
        {
          if (it->second.size() != output.chosen_instances[it->first].size())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper modified the premapped output "
                        "for region requirement %d of task %s (ID %lld).",
                        "map_task", mapper->get_mapper_name(), it->first,
                        get_task_name(), get_unique_id())
          for (unsigned idx = 0; idx < it->second.size(); idx++)
            if (it->second[idx] != output.chosen_instances[it->first][idx])
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper modified the premapped output "
                        "for region requirement %d of task %s (ID %lld).",
                        "map_task", mapper->get_mapper_name(), it->first,
                        get_task_name(), get_unique_id())
        }
      }
#endif
      // fill in virtual_mapped
      virtual_mapped.resize(regions.size(),false);
      // Convert all the outputs into our set of physical instances and
      // validate them by checking the following properites:
      // - all are either pure virtual or pure physical 
      // - no missing fields
      // - all satisfy the region requirement
      // - all are visible from all the target processors
      physical_instances.resize(regions.size());
      // If we're doing safety checks, we need the set of memories
      // visible from all the target processors
      std::set<Memory> visible_memories;
      if (!runtime->unsafe_mapper)
        runtime->find_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // If it was early mapped then it is easy
        std::map<unsigned,InstanceSet>::const_iterator finder = 
          early_mapped_regions.find(idx);
        if (finder != early_mapped_regions.end())
        {
          physical_instances[idx] = finder->second;
          // Check to see if it is visible or not from the target processors
          if (!runtime->unsafe_mapper && !regions[idx].is_no_access())
          {
            InstanceSet &req_instances = physical_instances[idx];
            for (unsigned idx2 = 0; idx2 < req_instances.size(); idx2++)
            {
              const Memory mem = req_instances[idx2].get_memory();
              if (visible_memories.find(mem) == visible_memories.end())
              {
                // Not visible from all target processors
                // Different error messages depending on the cause
                if (regions[idx].is_restricted()) 
                  REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                                "Invalid mapper output from invocation of '%s' "
                                "on mapper %s. Mapper selected processor(s) "
                                "which restricted instance of region "
                                "requirement %d in memory " IDFMT " is not "
                                "visible for task %s (ID %lld).",
                                "map_task", mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id())
                else 
                  REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                                "Invalid mapper output from invocation of '%s' "
                                "on mapper %s. Mapper selected processor(s) "
                                "for which premapped instance of region "
                                "requirement %d in memory " IDFMT " is not "
                                "visible for task %s (ID %lld).",
                                "map_task", mapper->get_mapper_name(), idx,
                                mem.id, get_task_name(), get_unique_id())
              }
            }
          }
          if (runtime->legion_spy_enabled)
            runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                                  idx, regions[idx],
                                                  physical_instances[idx]);
          continue;
        }
        // Skip any NO_ACCESS or empty privilege field regions
        if (no_access_regions[idx])
          continue;
        // Do the conversion
        InstanceSet &result = physical_instances[idx];
        RegionTreeID bad_tree = 0;
        std::vector<FieldID> missing_fields;
        std::vector<PhysicalManager*> unacquired;
        bool free_acquired = false;
        std::map<PhysicalManager*,unsigned> *acquired = NULL;
        // Get the acquired instances only if we are checking
        if (!runtime->unsafe_mapper)
        {
          if (this->must_epoch != NULL)
          {
            acquired = new std::map<PhysicalManager*,unsigned>(
                                  *get_acquired_instances_ref());
            free_acquired = true;
            // Merge the must epoch owners acquired instances too 
            // if we need to check for all our instances being acquired
            std::map<PhysicalManager*,unsigned> *epoch_acquired = 
              this->must_epoch->get_acquired_instances_ref();
            if (epoch_acquired != NULL)
              acquired->insert(epoch_acquired->begin(), epoch_acquired->end());
          }
          else
            acquired = get_acquired_instances_ref();
        }
        int composite_idx = 
          runtime->forest->physical_convert_mapping(this, regions[idx],
                output.chosen_instances[idx], result, bad_tree, missing_fields,
                acquired, unacquired, !runtime->unsafe_mapper);
        if (free_acquired)
          delete acquired;
        if (bad_tree > 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper specified an instance from region "
                        "tree %d for use with region requirement %d of task "
                        "%s (ID %lld) whose region is from region tree %d.",
                        "map_task",mapper->get_mapper_name(), bad_tree,
                        idx, get_task_name(), get_unique_id(),
                        regions[idx].region.get_tree_id())
        if (!missing_fields.empty())
        {
          for (std::vector<FieldID>::const_iterator it = 
                missing_fields.begin(); it != missing_fields.end(); it++)
          {
            const void *name; size_t name_size;
            if (!runtime->retrieve_semantic_information(
                regions[idx].region.get_field_space(), *it, 
                LEGION_NAME_SEMANTIC_TAG, name, name_size, 
                true/*can fail*/, false))
	          name = "(no name)";
              log_run.error("Missing instance for field %s (FieldID: %d)",
                          static_cast<const char*>(name), *it);
          }
          REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper failed to specify an instance for "
                        "%zd fields of region requirement %d on task %s "
                        "(ID %lld). The missing fields are listed below.",
                        "map_task", mapper->get_mapper_name(), 
                        missing_fields.size(), idx, get_task_name(), 
                        get_unique_id())
          
        }
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,unsigned> *acquired_instances = 
            get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator it = 
                unacquired.begin(); it != unacquired.end(); it++)
          {
            if (acquired_instances->find(*it) == acquired_instances->end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from 'map_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
          }
          // Event if we did successfully acquire them, still issue the warning
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                          "mapper %s failed to acquire instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'map_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
        }
        // See if they want a virtual mapping
        if (composite_idx >= 0)
        {
          // Everything better be all virtual or all real
          if (result.size() > 1)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Mapper specified mixed composite and "
                          "concrete instances for region requirement %d of "
                          "task %s (ID %lld). Only full concrete instances "
                          "or a single composite instance is supported.",
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
          if (IS_REDUCE(regions[idx]))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Illegal composite mapping requested on "
                          "region requirement %d of task %s (UID %lld) which "
                          "has only reduction privileges.", 
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
          if (!IS_EXCLUSIVE(regions[idx]))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Illegal composite instance requested "
                          "on region requirement %d of task %s (ID %lld) "
                          "which has a relaxed coherence mode. Virtual "
                          "mappings are only permitted for exclusive "
                          "coherence.", "map_task", mapper->get_mapper_name(),
                          idx, get_task_name(), get_unique_id())
          virtual_mapped[idx] = true;
        }
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                                idx, regions[idx],
                                                physical_instances[idx]);
        // Skip checks if the mapper promises it is safe
        if (runtime->unsafe_mapper)
          continue;
        // If this is anything other than a virtual mapping, check that
        // the instances align with the privileges
        if (!virtual_mapped[idx])
        {
          std::vector<LogicalRegion> regions_to_check(1, regions[idx].region);
          for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
          {
            PhysicalManager *manager = result[idx2].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
              // Doesn't satisfy the region requirement
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of '%s' on "
                            "mapper %s. Mapper specified instance that does "
                            "not meet region requirement %d for task %s "
                            "(ID %lld). The index space for the instance has "
                            "insufficient space for the requested logical "
                            "region.", "map_task", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
            if (manager->is_collective_manager())
            {
              CollectiveManager *collective_manager = 
                manager->as_collective_manager();
              if (!collective_manager->point_space->contains_point(index_point))
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of '%s' on "
                            "mapper %s. Mapper selected a collective instance "
                            "for region requirement %d that does not contain "
                            "the point for task %s (ID %lld).", "map_task",
                            mapper->get_mapper_name(), idx, get_task_name(),
                            get_unique_id())
            }
          }
          if (!regions[idx].is_no_access() &&
              !variant_impl->is_no_access_region(idx))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              const Memory mem = result[idx2].get_memory();
              if (visible_memories.find(mem) == visible_memories.end())
                // Not visible from all target processors
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected an instance for "
                              "region requirement %d in memory " IDFMT " "
                              "which is not visible from the target processors "
                              "for task %s (ID %lld).", "map_task", 
                              mapper->get_mapper_name(), idx, mem.id, 
                              get_task_name(), get_unique_id())
            }
          }
          // If this is a reduction region requirement make sure all the 
          // managers are reduction instances with the right reduction ops
          if (IS_REDUCE(regions[idx]))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              PhysicalManager *manager = result[idx2].get_physical_manager();
              if (!manager->is_reduction_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper failed to choose a "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which has "
                              "reduction privileges.", "map_task", 
                              mapper->get_mapper_name(), idx,
                              get_task_name(), get_unique_id())
              else if (manager->redop != regions[idx].redop)
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper failed selected a "
                              "specialized reduction instance with reduction "
                              "operator %d for region requirement %d of task "
                              "%s (ID %lld) which has reduction privileges "
                              "on a different reduction operator %d.", 
                              "map_task", mapper->get_mapper_name(), 
                              manager->redop, idx, get_task_name(), 
                              get_unique_id(), regions[idx].redop)
            }
          }
          else
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
              if (result[idx2].get_manager()->is_reduction_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected illegal "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which "
                              "does not have reduction privileges.", "map_task",
                              mapper->get_mapper_name(), idx, 
                              get_task_name(), get_unique_id())
          }
        }
      }
      // Now that we have our physical instances we can validate the variant
      if (!runtime->unsafe_mapper)
      {
#ifdef DEBUG_LEGION
        assert(!target_processors.empty());
#endif
        validate_variant_selection(mapper, variant_impl, 
            target_processors.front().kind(), physical_instances, "map_task");
      }
      // Record anything else that needs to be recorded 
      selected_variant = output.chosen_variant;
      task_priority = output.task_priority;
      perform_postmap = output.postmap_task;
      if (!output.untracked_valid_regions.empty())
      {
        untracked_valid_regions.swap(output.untracked_valid_regions);
        for (std::set<unsigned>::iterator it = untracked_valid_regions.begin();
              it != untracked_valid_regions.end(); /*nothing*/)
        {
          // Remove it if it is too big or 
          if ((*it >= regions.size()) || !IS_READ_ONLY(regions[*it]))
          {
            std::set<unsigned>::iterator to_remove = it++;
            if (*to_remove < regions.size())
              REPORT_LEGION_WARNING(LEGION_WARNING_NON_READ_ONLY_UNTRACK_VALID,
                  "Ignoring request by mapper %s to not track valid instances "
                  "for region requirement %d of task %s (UID %lld) because "
                  "region requirement does not have read-only privileges.",
                  mapper->get_mapper_name(), *to_remove, 
                  get_task_name(), unique_op_id)
            untracked_valid_regions.erase(to_remove);
          }
          else
            it++;
        }
      }
    }

    //--------------------------------------------------------------------------
    ApEvent SingleTask::get_memo_completion(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(single_task_termination.exists());
#endif
      return single_task_termination;
    }

    //--------------------------------------------------------------------------
    void SingleTask::replay_map_task_output(void)
    //--------------------------------------------------------------------------
    {
      std::vector<Processor> procs;
      tpl->get_mapper_output(this, selected_variant,
          task_priority, perform_postmap, procs, physical_instances);

      if (runtime->separate_runtime_instances)
      {
        target_processors.resize(1);
        target_processors[0] = this->target_proc;
      }
      else // the common case
        target_processors = procs;

      virtual_mapped.resize(regions.size(), false);
      bool needs_reservations = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        InstanceSet &instances = physical_instances[idx];
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        if (IS_ATOMIC(regions[idx]))
          needs_reservations = true;
        if (instances.is_virtual_mapping())
          virtual_mapped[idx] = true;
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
                                                idx, regions[idx],
                                                instances);
      }
      if (needs_reservations)
        // We group all reservations together anyway
        tpl->get_task_reservations(this, atomic_locks);
#ifdef DEBUG_LEGION
      assert(!single_task_termination.exists());
      assert(!task_effects_complete.exists());
#endif
      single_task_termination = Runtime::create_ap_user_event(NULL);
      task_effects_complete = single_task_termination;
      set_origin_mapped(true); // it's like this was origin mapped
    }

    //--------------------------------------------------------------------------
    void SingleTask::set_effects_postcondition(ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(task_effects_complete == single_task_termination);
#endif
      task_effects_complete = postcondition;
    }

    //--------------------------------------------------------------------------
    InnerContext* SingleTask::create_implicit_context(void)
    //--------------------------------------------------------------------------
    {
      InnerContext *inner_ctx = new InnerContext(runtime, this, 
          get_depth(), false/*is inner*/, regions, parent_req_indexes,
          virtual_mapped, unique_op_id, ApEvent::NO_AP_EVENT,
          false/*remote*/, false/*inline*/, true/*implicit*/);
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      inner_ctx->configure_context(mapper, task_priority);
      execution_context = inner_ctx;
      execution_context->add_reference();
      return inner_ctx;
    }

    //--------------------------------------------------------------------------
    void SingleTask::validate_target_processors(
                                 const std::vector<Processor> &processors) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!processors.empty());
#endif
      // Make sure that they are all on the same node and of the same kind
      const Processor &first = processors.front();
      const Processor::Kind kind = first.kind();
      const AddressSpace space = first.address_space();
      for (unsigned idx = 0; idx < processors.size(); idx++)
      {
        const Processor &proc = processors[idx];
        if (!proc.exists())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested an illegal "
                        "NO_PROC for a target processor when mapping task %s "
                        "(ID %lld).", mapper->get_mapper_name(), 
                        get_task_name(), get_unique_id())
        else if (proc.kind() != kind)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is of kind %s when mapping task %s "
                        "(ID %lld), but the target processor " IDFMT " has "
                        "kind %s. Only one kind of processor is permitted.",
                        mapper->get_mapper_name(), proc.id, 
                        Processor::get_kind_name(proc.kind()), get_task_name(),
                        get_unique_id(), this->target_proc.id, 
                        Processor::get_kind_name(kind))
        if (proc.address_space() != space)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is in address space %d when mapping "
                        "task %s (ID %lld) but the target processor " IDFMT 
                        "is in address space %d. All target processors must "
                        "be in the same address space.", 
                        mapper->get_mapper_name(), proc.id,
                        proc.address_space(), get_task_name(), get_unique_id(), 
                        this->target_proc.id, space)
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invoke_mapper(MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      Mapper::MapTaskInput input;
      Mapper::MapTaskOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      // Initialize the mapping input which also does all the traversal
      // down to the target nodes
      std::vector<InstanceSet> valid_instances(regions.size());
      initialize_map_task_input(input, output, must_epoch_owner, 
                                valid_instances);
      // Now we can invoke the mapper to do the mapping
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_map_task(this, &input, &output);
      // Now we can convert the mapper output into our physical instances
      finalize_map_task_output(input, output, must_epoch_owner,valid_instances);
      // Sort out any profiling requests that we need to perform
      if (!output.task_prof_requests.empty())
      {
        profiling_priority = output.profiling_priority;
        // If we do any legion specific checks, make sure we ask
        // Realm for the proc profiling info so that we can get
        // a callback to report our profiling information
        bool has_proc_request = false;
        // Filter profiling requests into those for copies and the actual task
        for (std::set<ProfilingMeasurementID>::const_iterator it = 
              output.task_prof_requests.requested_measurements.begin(); it !=
              output.task_prof_requests.requested_measurements.end(); it++)
        {
          if ((*it) > Mapping::PMID_LEGION_FIRST)
          {
            // If we haven't seen a proc usage yet, then add it
            // to the realm requests to ensure we get a callback
            // for this task. We know we'll see it before this
            // because the measurement IDs are in order
            if (!has_proc_request)
              task_profiling_requests.push_back(
                  (ProfilingMeasurementID)Realm::PMID_OP_PROC_USAGE);
            // These are legion profiling requests and currently
            // are only profiling task information
            task_profiling_requests.push_back(*it);
            continue;
          }
          switch ((Realm::ProfilingMeasurementID)*it)
          {
            case Realm::PMID_OP_PROC_USAGE:
              has_proc_request = true; // Then fall through
            case Realm::PMID_OP_STATUS:
            case Realm::PMID_OP_BACKTRACE:
            case Realm::PMID_OP_TIMELINE:
	    case Realm::PMID_OP_TIMELINE_GPU:
            case Realm::PMID_PCTRS_CACHE_L1I:
            case Realm::PMID_PCTRS_CACHE_L1D:
            case Realm::PMID_PCTRS_CACHE_L2:
            case Realm::PMID_PCTRS_CACHE_L3:
            case Realm::PMID_PCTRS_IPC:
            case Realm::PMID_PCTRS_TLB:
            case Realm::PMID_PCTRS_BP:
              {
                // Just task
                task_profiling_requests.push_back(*it);
                break;
              }
            default:
              {
                REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_PROFILING,
                              "Mapper %s requested a profiling "
                    "measurement of type %d which is not applicable to "
                    "task %s (UID %lld) and will be ignored.",
                    mapper->get_mapper_name(), *it, get_task_name(),
                    get_unique_id());
              }
          }
        }
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
        assert(outstanding_profiling_requests.load() == 0);
#endif
        profiling_reported = Runtime::create_rt_user_event();
        // Increment the number of profiling responses here since we
        // know that we're going to get one for launching the task
        // No need for the lock since no outstanding physical analyses
        // can be running yet
        outstanding_profiling_requests.store(1);
      }
      if (!output.copy_prof_requests.empty())
      {
        filter_copy_request_kinds(mapper, 
            output.copy_prof_requests.requested_measurements,
            copy_profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
        if (!profiling_reported.exists())
          profiling_reported = Runtime::create_rt_user_event();
      } 

      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((remote_trace_recorder != NULL) ||
                ((tpl != NULL) && tpl->is_recording()));
#endif
        const TraceLocalID tlid = get_trace_local_id();
        if (remote_trace_recorder != NULL)
          remote_trace_recorder->record_mapper_output(tlid, output,
              physical_instances, map_applied_conditions);
        else
          tpl->record_mapper_output(tlid, output, 
              physical_instances, map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent SingleTask::map_all_regions(MustEpochOp *must_epoch_op,
                                        const DeferMappingArgs *defer_args)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MAP_ALL_REGIONS_CALL);
      // Check to see if we need to make a remote trace recorder
      if (is_remote() && is_recording() && (remote_trace_recorder == NULL))
      {
        const RtUserEvent remote_applied = Runtime::create_rt_user_event();
        remote_trace_recorder = new RemoteTraceRecorder(runtime,
            orig_proc.address_space(), runtime->address_space, this, tpl,
            remote_applied, remote_collect_event);
        remote_trace_recorder->add_recorder_reference();
        map_applied_conditions.insert(remote_applied);
      }
      // Only do this the first or second time through
      if ((defer_args == NULL) || (defer_args->invocation_count < 3))
      {
        if ((defer_args == NULL) || (defer_args->invocation_count < 2))
        {
          if (request_valid_instances)
          {
            // If the mapper wants valid instances we first need to do our
            // versioning analysis and then call the mapper
            if ((defer_args == NULL/*first invocation*/) || 
                (defer_args->invocation_count == 0))
            {
              const RtEvent version_ready_event = 
                perform_versioning_analysis(false/*post mapper*/);
              if (version_ready_event.exists() && 
                  !version_ready_event.has_triggered())
                return defer_perform_mapping(version_ready_event, must_epoch_op,
                                             defer_args, 1/*invocation count*/);
            }
            // Now do the mapping call
            invoke_mapper(must_epoch_op);
          }
          else
          {
            // If the mapper doesn't need valid instances, we do the mapper
            // call first and then see if we need to do any versioning analysis
            if ((defer_args == NULL/*first invocation*/) ||
                (defer_args->invocation_count == 0))
            {
              invoke_mapper(must_epoch_op);
              const RtEvent version_ready_event = 
                perform_versioning_analysis(true/*post mapper*/);
              if (version_ready_event.exists() && 
                  !version_ready_event.has_triggered())
                return defer_perform_mapping(version_ready_event, must_epoch_op,
                                             defer_args, 1/*invocation count*/);
            }
          }
        }
        // If we have any intra-space mapping dependences that haven't triggered
        // then we need to defer ourselves until they have occurred
        if (!intra_space_mapping_dependences.empty())
        {
          const RtEvent ready = 
            Runtime::merge_events(intra_space_mapping_dependences);
          intra_space_mapping_dependences.clear();
          if (ready.exists() && !ready.has_triggered())
            return defer_perform_mapping(ready, must_epoch_op,
                                         defer_args, 2/*invocation count*/);
        }
        // Create our task termination event at this point
#ifdef DEBUG_LEGION
        assert(!single_task_termination.exists());
#endif
        // Note that tracing doesn't track this as a user event, it is just
        // a name we're making for the termination event
        single_task_termination = Runtime::create_ap_user_event(NULL); 
        // See if we have a remote trace info to use, if we don't then make
        // our trace info and do the initialization
        const TraceInfo trace_info = is_remote() ?
          TraceInfo(this, remote_trace_recorder, true/*initialize*/) :
          TraceInfo(this, true/*initialize*/);
        ApEvent init_precondition = compute_init_precondition(trace_info);
        // After we've got our results, apply the state to the region tree
        if (!regions.empty())
        {
          if (regions.size() == 1)
          {
            if (early_mapped_regions.empty() && 
                !no_access_regions[0] && !virtual_mapped[0])
            {
              const bool record_valid = (untracked_valid_regions.find(0) == 
                                         untracked_valid_regions.end());
#ifdef DEBUG_LEGION
              assert(!task_effects_complete.exists());
#endif
              task_effects_complete =
                runtime->forest->physical_perform_updates_and_registration(
                    regions[0], version_infos[0], this, 0, 
                    init_precondition, single_task_termination,
                    physical_instances[0], PhysicalTraceInfo(trace_info, 0),
                                          map_applied_conditions,
#ifdef DEBUG_LEGION
                                          get_logging_name(),
                                          unique_op_id,
#endif
                                          record_valid);
              if (task_effects_complete.exists())
                task_effects_complete = Runtime::merge_events(&trace_info, 
                    task_effects_complete, single_task_termination);
              else
                task_effects_complete = single_task_termination;
              if (is_recording() && 
                  (task_effects_complete != single_task_termination))
                trace_info.record_set_effects(this, task_effects_complete);
#ifdef DEBUG_LEGION
              dump_physical_state(&regions[0], 0);
#endif
            }
          }
          else
          {
            unsigned read_only_count = 0;
            std::vector<unsigned> performed_regions;
            performed_regions.reserve(regions.size());
            std::set<RtEvent> registration_postconditions;
            std::vector<UpdateAnalysis*> analyses(regions.size(), NULL);
            std::vector<ApEvent> effects(regions.size(), ApEvent::NO_AP_EVENT);
            std::vector<RtEvent> reg_pre(regions.size(), RtEvent::NO_RT_EVENT);
            for (unsigned idx = 0; idx < regions.size(); idx++)
            {
              if (early_mapped_regions.find(idx) != early_mapped_regions.end())
              {
                if (runtime->legion_spy_enabled)
                  LegionSpy::log_task_premapping(unique_op_id, idx);
                continue;
              }
              if (no_access_regions[idx])
                continue;
              VersionInfo &local_info = get_version_info(idx);
              // If we virtual mapped it, there is nothing to do
              if (virtual_mapped[idx])
                continue;
              performed_regions.push_back(idx);
              const bool record_valid = (untracked_valid_regions.find(idx) ==
                                         untracked_valid_regions.end());
              // apply the results of the mapping to the tree
              reg_pre[idx] = runtime->forest->physical_perform_updates(
                                          regions[idx], local_info, 
                                          this, idx, init_precondition,
                                          single_task_termination,
                                          physical_instances[idx],
                                          PhysicalTraceInfo(trace_info, idx),
                                          map_applied_conditions,
                                          analyses[idx],
#ifdef DEBUG_LEGION
                                          get_logging_name(),
                                          unique_op_id,
#endif
                                          record_valid);
              if (IS_READ_ONLY(regions[idx]))
                read_only_count++;
            }
            // In order to avoid cycles when mapping multiple tasks in parallel
            // with read-only requirements, we need to guarantee that all read-only
            // copies are issued before we can perform any registrations for the
            // task that will be using their results.
            if (read_only_count > 1)
            {
              std::vector<RtEvent> read_only_preconditions;
              read_only_preconditions.reserve(read_only_count);
              std::vector<unsigned> read_only_regions;
              read_only_regions.reserve(read_only_count);
              for (std::vector<unsigned>::const_iterator it =
                    performed_regions.begin(); it !=
                    performed_regions.end(); it++)
              {
                if (!IS_READ_ONLY(regions[*it]))
                  continue;
                read_only_regions.push_back(*it);
                const RtEvent precondition = reg_pre[*it];
                if (precondition.exists())
                  read_only_preconditions.push_back(precondition);
              }
              if (!read_only_preconditions.empty())
              {
                const RtEvent read_only_precondition =
                  Runtime::merge_events(read_only_preconditions);
                if (read_only_precondition.exists())
                {
                  for (std::vector<unsigned>::const_iterator it =
                        read_only_regions.begin(); it !=
                        read_only_regions.end(); it++)
                    reg_pre[*it] = read_only_precondition;
                }
              }
            }
            for (std::vector<unsigned>::const_iterator it = 
                 performed_regions.begin(); it != performed_regions.end(); it++)
            {
              // If we have updates for either copy launcher then defer it
              // in order to avoid blocking here, otherwise we can just do
              // it here as we know that we won't block
              if (reg_pre[*it].exists() || analyses[*it]->has_output_updates())
              {
                const RtEvent registration_post = 
                  runtime->forest->defer_physical_perform_registration(
                                          reg_pre[*it], analyses[*it],
                                          physical_instances[*it],
                                          map_applied_conditions, effects[*it],
                                          PhysicalTraceInfo(trace_info, *it));
                registration_postconditions.insert(registration_post);
              }
              else
                effects[*it] = runtime->forest->physical_perform_registration(
                                          analyses[*it],physical_instances[*it],
                                          PhysicalTraceInfo(trace_info, *it),
                                          map_applied_conditions);
            }
            // Wait for all the registrations to be done
            if (!registration_postconditions.empty())
            {
              const RtEvent registration_post = 
                Runtime::merge_events(registration_postconditions);
              if (registration_post.exists() && 
                  !registration_post.has_triggered())
              {
                std::vector<unsigned> *performed_copy = 
                  new std::vector<unsigned>();
                performed_copy->swap(performed_regions);
                std::vector<ApEvent> *effects_copy = 
                  new std::vector<ApEvent>();
                effects_copy->swap(effects);
                // We'll restart down below with the third possible invocation
                return defer_perform_mapping(registration_post, must_epoch_op,
                                            defer_args, 3/*invocation count*/, 
                                            performed_copy, effects_copy);
              }
            }
            // Now we can do the registrations
            std::set<ApEvent> effects_postconditions;
            for (std::vector<unsigned>::const_iterator it =
                 performed_regions.begin(); it != performed_regions.end(); it++)
            {
              if (effects[*it].exists())
                effects_postconditions.insert(effects[*it]);
#ifdef DEBUG_LEGION
              dump_physical_state(&regions[*it], *it);
#endif
            }
#ifdef DEBUG_LEGION
            assert(!task_effects_complete.exists());
#endif
            if (!effects_postconditions.empty())
            {
              effects_postconditions.insert(single_task_termination);
              task_effects_complete =
                Runtime::merge_events(&trace_info, effects_postconditions);
              if (is_recording())
                trace_info.record_set_effects(this, task_effects_complete);
            }
            else
              task_effects_complete = single_task_termination;
          }
          if (perform_postmap)
            perform_post_mapping(trace_info);
        } // if (!regions.empty())
        else
          task_effects_complete = single_task_termination;
      }
      else // third invocation
      {
#ifdef DEBUG_LEGION
        assert(defer_args->invocation_count == 3);
        assert(defer_args->performed_regions != NULL);
        assert(defer_args->effects != NULL);
#endif
        // This is in case we had to defer the second part of the invocation
        std::set<ApEvent> effects_postconditions;
        for (std::vector<unsigned>::const_iterator it =
              defer_args->performed_regions->begin(); it !=
              defer_args->performed_regions->end(); it++)
        {
          if ((*(defer_args->effects))[*it].exists())
            effects_postconditions.insert((*(defer_args->effects))[*it]);
#ifdef DEBUG_LEGION
          dump_physical_state(&regions[*it], *it);
#endif
        }
#ifdef DEBUG_LEGION
        assert(!task_effects_complete.exists());
#endif
        if (!effects_postconditions.empty())
        {
          const TraceInfo trace_info = is_remote() ?
            TraceInfo(this, remote_trace_recorder) : TraceInfo(this);
          effects_postconditions.insert(single_task_termination);
          task_effects_complete =
            Runtime::merge_events(&trace_info, effects_postconditions);
          if (is_recording())
            trace_info.record_set_effects(this, task_effects_complete);
        }
        else
          task_effects_complete = single_task_termination;
        delete defer_args->performed_regions;
        delete defer_args->effects;
        if (perform_postmap)
        {
          const TraceInfo trace_info = is_remote() ?
            TraceInfo(this, remote_trace_recorder) : TraceInfo(this);
          perform_post_mapping(trace_info);
        }
      }
      if (is_recording())
      {
        const TraceInfo trace_info = is_remote() ?
          TraceInfo(this, remote_trace_recorder) : TraceInfo(this);
        std::set<ApEvent> ready_events;
        if (execution_fence_event.exists())
          ready_events.insert(execution_fence_event);
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx] && !no_access_regions[idx])
            physical_instances[idx].update_wait_on_events(ready_events);
        }
        ApEvent ready_event = Runtime::merge_events(&trace_info, ready_events);
        if (!atomic_locks.empty())
        {
          const TraceLocalID tlid = get_trace_local_id();
          trace_info.record_reservations(tlid, atomic_locks,
                                         map_applied_conditions);
        }
        trace_info.record_complete_replay(this, ready_event);
      }
      if (remote_trace_recorder != NULL)
      {
        if (remote_trace_recorder->remove_recorder_reference())
          delete remote_trace_recorder;
        remote_trace_recorder = NULL;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_post_mapping(const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      Mapper::PostMapInput input;
      Mapper::PostMapOutput output;
      input.mapped_regions.resize(regions.size());
      input.valid_instances.resize(regions.size());
      output.chosen_instances.resize(regions.size());
      std::vector<InstanceSet> postmap_valid(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
          continue;
        if (no_access_regions[idx] || virtual_mapped[idx])
          continue;
        // Don't need to actually traverse very far, but we do need the
        // valid instances for all the regions
        RegionTreePath path;
        initialize_mapping_path(path, regions[idx], regions[idx].region);
        if (request_valid_instances)
          runtime->forest->physical_premap_region(this, idx, regions[idx], 
                                                  get_version_info(idx),
                                                  postmap_valid[idx],
                                                  map_applied_conditions);
        // No need to filter these because they are on the way out
        prepare_for_mapping(postmap_valid[idx], input.valid_instances[idx]);  
        prepare_for_mapping(physical_instances[idx], input.mapped_regions[idx]);
      }
      // Now we can do the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_post_map_task(this, &input, &output);
      // Check and register the results
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (early_mapped_regions.find(idx) != early_mapped_regions.end())
          continue;
        if (no_access_regions[idx] || virtual_mapped[idx])
          continue;
        if (output.chosen_instances[idx].empty())
          continue;
        RegionRequirement &req = regions[idx];
        if (req.is_restricted())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "is restricted. The request is being ignored.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_NO_ACCESS(req))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has NO_ACCESS privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_REDUCE(req))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has REDUCE privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        // Convert the post-mapping  
        InstanceSet result;
        RegionTreeID bad_tree = 0;
        std::vector<PhysicalManager*> unacquired;
        bool had_composite = 
          runtime->forest->physical_convert_postmapping(this, req,
                              output.chosen_instances[idx], result, bad_tree,
                              runtime->unsafe_mapper ? NULL : 
                                get_acquired_instances_ref(),
                              unacquired, !runtime->unsafe_mapper);
        if (bad_tree > 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'postmap_task' invocation "
                        "on mapper %s. Mapper provided an instance from region "
                        "tree %d for use in satisfying region requirement %d "
                        "of task %s (ID %lld) whose region is from region tree "
                        "%d.", mapper->get_mapper_name(), bad_tree,
                        idx, get_task_name(), get_unique_id(), 
                        regions[idx].region.get_tree_id())
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,unsigned> *acquired_instances = 
            get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator uit = 
                unacquired.begin(); uit != unacquired.end(); uit++)
          {
            if (acquired_instances->find(*uit) == acquired_instances->end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from 'postmap_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
          }
          // If we did successfully acquire them, still issue the warning
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                          "mapper %s failed to acquires instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'postmap_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
        }
        if (had_composite)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_COMPOSITE,
                          "Mapper %s requested a composite "
                          "instance be created for region requirement %d "
                          "of task %s (ID %lld) for a post mapping. The "
                          "request is being ignored.",
                          mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        if (!runtime->unsafe_mapper)
        {
          std::vector<LogicalRegion> regions_to_check(1, 
                                        regions[idx].region);
          for (unsigned check_idx = 0; check_idx < result.size(); check_idx++)
          {
            PhysicalManager *manager = result[check_idx].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of "
                            "'postmap_task' on mapper %s. Mapper specified an "
                            "instance region requirement %d of task %s "
                            "(ID %lld) that does not meet the logical region "
                            "requirement.", mapper->get_mapper_name(), idx, 
                            get_task_name(), get_unique_id())
          }
        }
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                                idx, regions[idx], result,
                                                true/*postmapping*/);
        // TODO: Implement physical tracing for postmapped regions
        if (is_memoizing())
          assert(false);
        // Register this with a no-event so that the instance can
        // be used as soon as it is valid from the copy to it
        // We also use read-only privileges to ensure that it doesn't
        // invalidate the other valid instances
        const PrivilegeMode mode = regions[idx].privilege;
        regions[idx].privilege = LEGION_READ_ONLY; 
        VersionInfo &local_version_info = get_version_info(idx);
        runtime->forest->physical_perform_updates_and_registration(
                          regions[idx], local_version_info, this, idx,
                          completion_event/*wait for task to be done*/,
                          ApEvent::NO_AP_EVENT/*done immediately*/, 
                          result, PhysicalTraceInfo(trace_info, idx), 
                          map_applied_conditions,
#ifdef DEBUG_LEGION
                          get_logging_name(), unique_op_id,
#endif
                          false/*track effects*/);
        regions[idx].privilege = mode; 
      }
    } 

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, LAUNCH_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(regions.size() == physical_instances.size());
      assert(regions.size() == no_access_regions.size());
#endif 
      // If we haven't computed our virtual mapping information
      // yet (e.g. because we origin mapped) then we have to
      // do that now
      if (virtual_mapped.size() != regions.size())
      {
        virtual_mapped.resize(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
          virtual_mapped[idx] = physical_instances[idx].is_virtual_mapping();
      }
      VariantImpl *variant = 
        runtime->find_variant_impl(task_id, selected_variant);
      // STEP 1: Compute the precondition for the task launch
      std::set<ApEvent> wait_on_events;
      if (execution_fence_event.exists())
        wait_on_events.insert(execution_fence_event);
#ifdef LEGION_SPY
      // TODO: teach legion spy how to check the inner task optimization
      // for now we'll just turn it off whenever we are going to be
      // validating the runtime analysis
      const bool do_inner_task_optimization = false;
#else
      const bool do_inner_task_optimization = variant->is_inner();
#endif
      // Get the event to wait on unless we are 
      // doing the inner task optimization
      if (!do_inner_task_optimization)
      {
        std::set<ApEvent> ready_events;
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (!virtual_mapped[idx] && !no_access_regions[idx])
            physical_instances[idx].update_wait_on_events(ready_events);
        }
        wait_on_events.insert(Runtime::merge_events(NULL, ready_events));
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          FutureImpl *impl = futures[idx].impl; 
          if (impl != NULL)
            wait_on_events.insert(impl->subscribe(false/*need local data*/));
        }
      }
      // Now add get all the other preconditions for the launch
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        GrantImpl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	ApEvent e = 
          Runtime::get_previous_phase(wait_barriers[idx].phase_barrier);
        wait_on_events.insert(e);
      }

      // STEP 2: Set up the task's context
      {
        if (!variant->is_leaf())
        {
          InnerContext *inner_ctx = new InnerContext(runtime, this, 
              get_depth(), variant->is_inner(), regions, parent_req_indexes,
              virtual_mapped, unique_op_id, execution_fence_event, 
              false/*remote*/, inline_task);
          if (mapper == NULL)
            mapper = runtime->find_mapper(current_proc, map_id);
          inner_ctx->configure_context(mapper, task_priority);
          execution_context = inner_ctx;
        }
        else
          execution_context = new LeafContext(runtime, this, inline_task);
        // Add a reference to our execution context
        execution_context->add_reference();
        std::vector<ApUserEvent> unmap_events(regions.size());
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(regions[idx].handle_type == LEGION_SINGULAR_PROJECTION);
#endif
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx] || no_access_regions[idx])
          {
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(clone_requirements[idx],
                         false/*mapped*/, map_id, tag, unmap_events[idx], 
                         true/*virtual mapped*/, physical_instances[idx]);
            // Don't switch coherence modes since we virtually
            // mapped it which means we will map in the parent's context
          }
          else if (do_inner_task_optimization)
          {
            // If this is an inner task then we don't map
            // the region with a physical region, but instead
            // we mark that the unmap event which marks when
            // the region can be used by child tasks should
            // be the ready event.
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            // Also make the region requirement read-write to force
            // people to wait on the value
            if (!IS_REDUCE(regions[idx]))
              clone_requirements[idx].privilege = LEGION_READ_WRITE;
            execution_context->add_physical_region(clone_requirements[idx],
                    false/*mapped*/, map_id, tag, unmap_events[idx],
                    false/*virtual mapped*/, physical_instances[idx]);
#ifdef DEBUG_LEGION
            assert(unmap_events[idx].exists());
#endif
            // Trigger the user event when the region is 
            // actually ready to be used
            std::set<ApEvent> ready_events;
            physical_instances[idx].update_wait_on_events(ready_events);
            ApEvent precondition = Runtime::merge_events(NULL, ready_events);
            Runtime::trigger_event(NULL, unmap_events[idx], precondition);
          }
          else
          { 
            // If this is not virtual mapped, here is where we
            // switch coherence modes from whatever they are in
            // the enclosing context to exclusive within the
            // context of this task
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(clone_requirements[idx],
                    true/*mapped*/, map_id, tag, unmap_events[idx],
                    false/*virtual mapped*/, physical_instances[idx]);
            // We reset the reference below after we've
            // initialized the local contexts and received
            // back the local instance references
          }
        }
        // Initialize any region tree contexts
        execution_context->initialize_region_tree_contexts(clone_requirements,
                                        unmap_events, map_applied_conditions);
      }
      // Merge together all the events for the start condition 
      ApEvent start_condition = Runtime::merge_events(NULL, wait_on_events);
      // Take all the locks in order in the proper way
      if (!atomic_locks.empty())
      {
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          start_condition = Runtime::acquire_ap_reservation(it->first, 
                                          it->second, start_condition);
          // We can also issue the release now dependent on this
          // task being complete, this way we do it before we launch
          // the task and the atomic_locks might be cleaned up
          Runtime::release_reservation(it->first, single_task_termination);
        }
      }
      // STEP 3: Finally we get to launch the task
      // Mark that we have an outstanding task in this context 
      if (!inline_task)
        parent_ctx->increment_pending();
      // Note there is a potential scary race condition to be aware of here: 
      // once we launch this task it's possible for this task to run and 
      // clean up before we finish the execution of this function thereby
      // invalidating this SingleTask object's fields.  This means
      // that we need to save any variables we need for after the task
      // launch here on the stack before they can be invalidated.
#ifdef DEBUG_LEGION
      assert(!target_processors.empty());
#endif
      Processor launch_processor = target_processors[0];
      if (target_processors.size() > 1)
      {
        // Find the processor group for all the target processors
        launch_processor = runtime->find_processor_group(target_processors);
      }
      Realm::ProfilingRequestSet profiling_requests;
      // If the mapper requested profiling add that now too
      if (!task_profiling_requests.empty())
      {
        // See if we have any realm requests
        std::set<Realm::ProfilingMeasurementID> realm_measurements;
        for (std::vector<ProfilingMeasurementID>::const_iterator it = 
              task_profiling_requests.begin(); it != 
              task_profiling_requests.end(); it++)
        {
          if ((*it) < Mapping::PMID_LEGION_FIRST)
            realm_measurements.insert((Realm::ProfilingMeasurementID)(*it));
          else if ((*it) == Mapping::PMID_RUNTIME_OVERHEAD)
            execution_context->initialize_overhead_tracker();
          else
            assert(false); // should never get here
        }
        if (!realm_measurements.empty())
        {
          OpProfilingResponse response(this, 0, 0, false/*fill*/, true/*task*/);
          Realm::ProfilingRequest &request = profiling_requests.add_request(
              runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
              &response, sizeof(response));
          request.add_measurements(realm_measurements);
          // No need to increment the number of outstanding profiling
          // requests here since it was already done when we invoked
          // the mapper (see SingleTask::invoke_mapper)
          // The exeception is for origin-mapped remote tasks on which
          // we're going to need to send a message back to the owner
          if (is_remote() && is_origin_mapped())
          {
#ifdef DEBUG_LEGION
            assert(outstanding_profiling_requests.load() == 0);
            assert(!profiling_reported.exists());
#endif
            outstanding_profiling_requests.store(1);
            profiling_reported = Runtime::create_rt_user_event();
          }
        }
      }
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_variant_decision(unique_op_id, selected_variant);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, start_condition,
                                        single_task_termination);
#endif
        LegionSpy::log_task_priority(unique_op_id, task_priority);
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          FutureImpl *impl = futures[idx].impl;
          if (impl == NULL)
            continue;
          if (impl->get_ready_event().exists())
            LegionSpy::log_future_use(unique_op_id, impl->get_ready_event());
        }
      }
      // If this is a leaf task variant, then we can immediately trigger
      // the single_task_termination event dependent on the task_launch_event
      // because we know there will be no child operations we need to wait for
      // We have to pull it onto the stack here though to avoid the race 
      // condition with us getting pre-empted and the task running to completion
      // before we get a chance to trigger the event
      ApEvent chain_precondition;
      const ApUserEvent chain_task_termination = single_task_termination;
      if (!variant->is_leaf())
      {
        single_task_termination = Runtime::create_ap_user_event(NULL);
        chain_precondition = single_task_termination;
      }
      else // We're going to trigger this right now with no precondition
        single_task_termination = ApUserEvent::NO_AP_USER_EVENT;
      ApEvent task_launch_event;
      if (inline_task)
      {
        bool poisoned = false;
        if (start_condition.exists() && 
            !start_condition.has_triggered_faultaware(poisoned))
          start_condition.wait_faultaware(poisoned);
        if (poisoned)
          execution_context->raise_poison_exception();
        variant->dispatch_inline(launch_processor, execution_context);
      }
      else
        task_launch_event = variant->dispatch_task(launch_processor, this,
                            execution_context, start_condition, true_guard,
                            task_priority, profiling_requests);
      if (chain_task_termination.exists())
      {
        if (chain_precondition.exists())
          Runtime::trigger_event(NULL, chain_task_termination,
              Runtime::merge_events(NULL, chain_precondition, task_launch_event));
        else
          Runtime::trigger_event(NULL, chain_task_termination, task_launch_event);
      }
      // Finally if this is a predicated task and we have a speculative
      // guard then we need to launch a meta task to handle the case
      // where the task misspeculates
      if (false_guard.exists())
      {
        MisspeculationTaskArgs args(this);
        // Make sure this runs on an application processor where the
        // original task was going to go 
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                         RtEvent(false_guard));
        // Fun little trick here: decrement the outstanding meta-task
        // counts for the mis-speculation task in case it doesn't run
        // If it does run, we'll increment the counts again
#ifdef DEBUG_LEGION
        runtime->decrement_total_outstanding_tasks(
            MisspeculationTaskArgs::TASK_ID, true/*meta*/);
#else
        runtime->decrement_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
        runtime->outstanding_counts[
          MisspeculationTaskArgs::TASK_ID].fetch_sub(1);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_profiling_requests(Serializer &rez,
                                             std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request(
        runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
        &response, sizeof(response));
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            copy_profiling_requests.begin(); it != 
            copy_profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id); 
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(base);
      // First see if this is a task response for an origin-mapped task
      // on a remote node that needs to be sent back to the origin node
      if (task_prof->task && is_origin_mapped() && is_remote())
      {
        // We need to send this response back to the owner node along
        // with the overhead tracker
        SingleTask *orig_task = get_origin_task();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(orig_length);
          rez.serialize(orig, orig_length);
          if (execution_context->overhead_tracker)
          {
            rez.serialize<bool>(true);
            rez.serialize(*execution_context->overhead_tracker);
          }
          else
            rez.serialize<bool>(false);
          
        }
        runtime->send_remote_task_profiling_response(orig_proc, rez);
      }
      else
      {
        // Check to see if we are done mapping, if not then we need to defer
        // this until we are done mapping so we know how many
        if (!mapped_event.has_triggered())
        {
          // Take the lock and see if we lost the race
          AutoLock o_lock(op_lock);
          if (!mapped_event.has_triggered())
          {
            // Save this profiling response for later until we know the
            // full count of profiling responses
            profiling_info.resize(profiling_info.size() + 1);
            SingleProfilingInfo &info = profiling_info.back();
            info.task_response = task_prof->task; 
            info.region_requirement_index = task_prof->src;
            info.fill_response = task_prof->fill;
            info.buffer_size = orig_length;
            info.buffer = malloc(orig_length);
            memcpy(info.buffer, orig, orig_length);
            if (info.task_response)
            {
              // If we had an overhead tracker 
              // see if this is the callback for the task
              if (execution_context->overhead_tracker != NULL)
                // This is the callback for the task itself
                info.profiling_responses.attach_overhead(
                    execution_context->overhead_tracker);
            }
            return;
          }
        }
        // If we get here then we can handle the response now
        Mapping::Mapper::TaskProfilingInfo info;
        info.profiling_responses.attach_realm_profiling_response(response);
        info.task_response = task_prof->task; 
        info.region_requirement_index = task_prof->src;
        info.total_reports = outstanding_profiling_requests.load();
        info.fill_response = task_prof->fill;
        if (info.task_response)
        {
          // If we had an overhead tracker 
          // see if this is the callback for the task
          if (execution_context->overhead_tracker != NULL)
            // This is the callback for the task itself
            info.profiling_responses.attach_overhead(
                execution_context->overhead_tracker);
        }
        mapper->invoke_task_report_profiling(this, &info);
      }
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    } 

    //--------------------------------------------------------------------------
    void SingleTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_single_task_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      if (outstanding_profiling_requests.load() > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapped_event.has_triggered());
#endif
        std::vector<SingleProfilingInfo> to_perform;
        {
          AutoLock o_lock(op_lock);
          to_perform.swap(profiling_info);
        }
        if (!to_perform.empty())
        {
          for (unsigned idx = 0; idx < to_perform.size(); idx++)
          {
            SingleProfilingInfo &info = to_perform[idx];
            const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
            info.total_reports = outstanding_profiling_requests.load();
            info.profiling_responses.attach_realm_profiling_response(resp);
            mapper->invoke_task_report_profiling(this, &info);
            free(info.buffer);
          }
          const int count = to_perform.size() + 
              outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
          assert(count <= outstanding_profiling_requests.load());
#endif
          if (count == outstanding_profiling_requests.load())
            Runtime::trigger_event(profiling_reported);
        }
      }
      else
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::TaskProfilingInfo info;
        info.total_reports = 0;
        info.task_response = true;
        info.region_requirement_index = 0;
        info.fill_response = false; // make valgrind happy
        mapper->invoke_task_report_profiling(this, &info);    
        Runtime::trigger_event(profiling_reported);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_remote_profiling_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t buffer_size;
      derez.deserialize(buffer_size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(buffer_size);
#ifdef DEBUG_LEGION
      // Realm needs this buffer to have 8-byte alignment so check that it does
      assert((uintptr_t(buffer) % 8) == 0);
#endif
      bool has_tracker;
      derez.deserialize(has_tracker);
      Mapping::ProfilingMeasurements::RuntimeOverhead tracker;
      if (has_tracker)
        derez.deserialize(tracker);
      const Realm::ProfilingResponse response(buffer, buffer_size);
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(response.user_data());
      Mapping::Mapper::TaskProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.task_response = task_prof->task; 
      info.region_requirement_index = task_prof->src;
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = task_prof->fill;
      if (has_tracker)
        info.profiling_responses.attach_overhead(&tracker);
      mapper->invoke_task_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::process_remote_profiling_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SingleTask *target;
      derez.deserialize(target);
      target->handle_remote_profiling_response(derez);
    }

    //--------------------------------------------------------------------------
    void SingleTask::trigger_children_complete(ApEvent all_children_complete)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!all_children_complete.exists() || 
              single_task_termination.exists() || is_implicit_top_level_task());
#endif
      if (single_task_termination.exists())
      {
        Runtime::trigger_event(NULL, 
            single_task_termination, all_children_complete); 
        single_task_termination = ApUserEvent::NO_AP_USER_EVENT;
      }
      else if (all_children_complete.exists())
      {
#ifdef DEBUG_LEGION
        assert(is_implicit_top_level_task());
        assert(!task_effects_complete.exists());
#endif
        task_effects_complete = all_children_complete;
      }
      TaskOp::trigger_children_complete();
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(Runtime *rt)
      : TaskOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MultiTask::activate_multi(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_MULTI_CALL);
      activate_task();
      launch_space = NULL;
      future_handles = NULL;
      internal_space = IndexSpace::NO_SPACE;
      sliced = false;
      redop = 0;
      deterministic_redop = false;
      reduction_op = NULL;
      serdez_redop_fns = NULL;
      reduction_state_size = 0;
      reduction_state = NULL;
      first_mapping = true;
      children_complete_invoked = false;
      children_commit_invoked = false;
      predicate_false_result = NULL;
      predicate_false_size = 0;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate_multi(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_MULTI_CALL);
      if (runtime->profiler != NULL)
        runtime->profiler->register_multi_task(this, task_id);
      deactivate_task();
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      if ((future_handles != NULL) && future_handles->remove_reference())
        delete future_handles;
      // Remove our reference to the future map
      future_map = FutureMap();
      if (reduction_state != NULL)
      {
        legion_free(REDUCTION_ALLOC, reduction_state, reduction_state_size);
        reduction_state = NULL;
        reduction_state_size = 0;
      }
      if (!temporary_futures.empty())
      {
        for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator it =
              temporary_futures.begin(); it != temporary_futures.end(); it++)
        {
          legion_free(FUTURE_RESULT_ALLOC, it->second.first, it->second.second);
        }
        temporary_futures.clear();
      }
      // Remove our reference to the point arguments 
      point_arguments = FutureMap();
      point_futures.clear();
      slices.clear(); 
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      predicate_false_future = Future();
      intra_space_dependences.clear();
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void) const
    //--------------------------------------------------------------------------
    {
      return sliced;
    }

    //--------------------------------------------------------------------------
    void MultiTask::slice_index_space(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_INDEX_SPACE_CALL);
#ifdef DEBUG_LEGION
      assert(!sliced);
#endif
      sliced = true;
      stealable = false; // cannot steal something that has been sliced
      Mapper::SliceTaskInput input;
      Mapper::SliceTaskOutput output;
      input.domain_is = internal_space;
      runtime->forest->find_launch_space_domain(internal_space, input.domain);
      output.verify_correctness = false;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_slice_task(this, &input, &output);
      if (output.slices.empty())
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'slice_task' "
                      "call on mapper %s. Mapper failed to specify an slices "
                      "for task %s (ID %lld).", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id())

#ifdef DEBUG_LEGION
      size_t total_points = 0;
#endif
      for (unsigned idx = 0; idx < output.slices.size(); idx++)
      {
        Mapper::TaskSlice &slice = output.slices[idx]; 
        if (!slice.proc.exists())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned a slice for task "
                        "%s (ID %lld) with an invalid processor " IDFMT ".",
                        mapper->get_mapper_name(), get_task_name(),
                        get_unique_id(), slice.proc.id)
        // Check to see if we need to get an index space for this domain
        if (!slice.domain_is.exists() && (slice.domain.get_volume() > 0))
          slice.domain_is = 
            runtime->find_or_create_index_slice_space(slice.domain,
                                    internal_space.get_type_tag(), provenance);
        if (slice.domain_is.get_type_tag() != internal_space.get_type_tag())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned slice index space %d "
                        "for task %s (UID %lld) with a different type than "
                        "original index space to be sliced.",
                        mapper->get_mapper_name(), slice.domain_is.get_id(),
                        get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        // Check to make sure the domain is not empty
        Domain &d = slice.domain;
        if ((d == Domain::NO_DOMAIN) && slice.domain_is.exists())
          runtime->forest->find_launch_space_domain(slice.domain_is, d);
        bool empty = false;
	size_t volume = d.get_volume();
	if (volume == 0)
	  empty = true;
	else
	  total_points += volume;
        if (empty)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned an empty slice for task "
                        "%s (ID %lld).", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
#endif
        SliceTask *new_slice = this->clone_as_slice_task(slice.domain_is,
                                                         slice.proc,
                                                         slice.recurse,
                                                         slice.stealable);
        slices.push_back(new_slice);
      }
#ifdef DEBUG_LEGION
      // If the volumes don't match, then something bad happend in the mapper
      if (total_points != input.domain.get_volume())
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'slice_task' "
                      "on mapper %s. Mapper returned slices with a total "
                      "volume %ld that does not match the expected volume of "
                      "%zd when slicing task %s (ID %lld).", 
                      mapper->get_mapper_name(), long(total_points),
                      input.domain.get_volume(), 
                      get_task_name(), get_unique_id())
#endif
      if (output.verify_correctness)
      {
        std::vector<IndexSpace> slice_spaces(slices.size());
        for (unsigned idx = 0; idx < output.slices.size(); idx++)
          slice_spaces[idx] = output.slices[idx].domain_is;
        runtime->forest->validate_slicing(internal_space, slice_spaces,
                                          this, mapper);
      }
      trigger_slices(); 
      // If we succeeded and this is an intermediate slice task
      // then we can reclaim it, otherwise, if it is the original
      // index task then we want to keep it around. Note it is safe
      // to call get_task_kind here despite the cleanup race because
      // it is a static property of the object.
      if (get_task_kind() == SLICE_TASK_KIND)
        deactivate();
    }

    //--------------------------------------------------------------------------
    void MultiTask::trigger_slices(void)
    //--------------------------------------------------------------------------
    {
      // Add our slices back into the queue of things that are ready to map
      // or send it to its remote node if necessary
      // Watch out for the cleanup race with some acrobatics here
      // to handle the case where the iterator is invalidated
      std::set<RtEvent> wait_for;
      std::list<SliceTask*>::const_iterator it = slices.begin();
      while (true)
      {
        SliceTask *slice = *it;
        // Have to update this before launching the task to avoid 
        // the clean-up race
        it++;
        const bool done = (it == slices.end());
        // Dumb case for must epoch operations, we need these to 
        // be mapped immediately, mapper be damned
        if (must_epoch != NULL)
        {
          TriggerTaskArgs trigger_args(slice);
          RtEvent done = runtime->issue_runtime_meta_task(trigger_args, 
                                           LG_THROUGHPUT_WORK_PRIORITY);
          wait_for.insert(done);
        }
        // If we're replaying this for for a trace then don't even
        // bother asking the mapper about when to map this
        else if (is_replaying())
          slice->enqueue_ready_operation();
        // Figure out whether this task is local or remote
        else if (!runtime->is_local(slice->target_proc))
        {
          // We can only send it away if it is not origin mapped
          // otherwise it has to stay here until it is fully mapped
          if (!slice->is_origin_mapped())
            runtime->send_task(slice);
          else
            slice->enqueue_ready_task(false/*use target*/);
        }
        else
          slice->enqueue_ready_task(true/*use target*/);
        if (done)
          break;
      }
      // Must-epoch operations are nasty little beasts and have
      // to wait for the effects to finish before returning
      if (!wait_for.empty())
      {
        RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(MultiTask *rhs, IndexSpace is,
                                     Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CLONE_MULTI_CALL);
#ifdef DEBUG_LEGION
      assert(this->launch_space == NULL);
      assert(this->future_handles == NULL);
#endif
      this->clone_task_op_from(rhs, p, stealable, false/*duplicate*/);
      this->index_domain = rhs->index_domain;
      this->launch_space = rhs->launch_space;
      add_launch_space_reference(this->launch_space);
      this->future_handles = rhs->future_handles;
      if (this->future_handles != NULL)
        this->future_handles->add_reference();
      this->internal_space = is;
      this->future_map = rhs->future_map;
      this->must_epoch_task = rhs->must_epoch_task;
      this->sliced = !recurse;
      this->redop = rhs->redop;
      if (this->redop != 0)
      {
        this->deterministic_redop = rhs->deterministic_redop;
        if (!this->deterministic_redop)
        {
          // Only need to initialize this if we're not doing a 
          // deterministic reduction operation
          this->reduction_op = rhs->reduction_op;
          this->serdez_redop_fns = rhs->serdez_redop_fns;
          initialize_reduction_state();
        }
      }
      this->point_arguments = rhs->point_arguments;
      if (!rhs->point_futures.empty())
        this->point_futures = rhs->point_futures;
      this->predicate_false_future = rhs->predicate_false_future;
      this->predicate_false_size = rhs->predicate_false_size;
      if (this->predicate_false_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(this->predicate_false_result == NULL);
#endif
        this->predicate_false_result = malloc(this->predicate_false_size);
        memcpy(this->predicate_false_result, rhs->predicate_false_result,
               this->predicate_false_size);
      }
    }

    //--------------------------------------------------------------------------
    Domain MultiTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(internal_space.exists());
#endif
      Domain result;
      runtime->forest->find_launch_space_domain(internal_space, result);
      return result; 
    }

    //--------------------------------------------------------------------------
    void MultiTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MULTI_TRIGGER_EXECUTION_CALL);
      if (is_remote())
      {
        // distribute, slice, then map/launch
        if (distribute_task())
        {
          // Still local
          if (is_sliced())
          {
            if (is_origin_mapped())
              launch_task();
            else
              map_and_launch();
          }
          else
            slice_index_space();
        }
      }
      else
      {
        // Not remote
        // If we're doing a must epoch launch then we don't
        // need to early map any regions because any interfering
        // regions that would be handled by this will be handled
        // by the map_must_epoch call
        if (must_epoch == NULL)
          early_map_task();
        if (is_origin_mapped())
        {
          if (is_sliced())
          {
            if (must_epoch == NULL)
            {
              // See if we've done our first mapping yet or not
              if (first_mapping)
              {
                first_mapping = false;
                const RtEvent done_mapping = perform_mapping();
                if (!done_mapping.exists() || done_mapping.has_triggered())
                {
                  if (distribute_task())
                    launch_task();
                }
                else
                  defer_distribute_task(done_mapping);
              }
              else
              {
                // We know that it is staying on one
                // of our local processors.  If it is
                // still this processor then map and run it
                if (distribute_task())
                  // Still local so we can launch it
                  launch_task();
              }
            }
            else
              register_must_epoch();
          }
          else
            slice_index_space();
        }
        else
        {
          if (distribute_task())
          {
            // Still local try slicing, mapping, and launching
            if (is_sliced())
              map_and_launch();
            else
              slice_index_space();
          }
        }
      }
    } 

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_MULTI_CALL);
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(launch_space->handle);
      rez.serialize(sliced);
      rez.serialize(redop);
      if (redop > 0)
        rez.serialize<bool>(deterministic_redop);
      else if (future_handles != NULL)
      {
        // Only pack the IDs for our local points
        IndexSpaceNode *node = runtime->forest->get_node(internal_space);
        Domain local_domain;
        node->get_launch_space_domain(local_domain);
        size_t local_size = local_domain.get_volume();
        rez.serialize(local_size);
        const std::map<DomainPoint,DistributedID> &handles =
          future_handles->handles;
#ifdef DEBUG_LEGION
        assert(local_size <= handles.size());
#endif
        if (local_size < handles.size())
        {
          for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
          {
            std::map<DomainPoint,DistributedID>::const_iterator finder = 
              handles.find(itr.p);
#ifdef DEBUG_LEGION
            assert(finder != handles.end());
#endif
            rez.serialize(finder->first);
            rez.serialize(finder->second);
          }
        }
        else
        {
          for (std::map<DomainPoint,DistributedID>::const_iterator it =
                handles.begin(); it != handles.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
        }
      }
      else
        rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez,
                                      std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_MULTI_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events); 
      IndexSpace launch_handle;
      derez.deserialize(launch_handle);
#ifdef DEBUG_LEGION
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_handle);
      WrapperReferenceMutator mutator(ready_events);
      launch_space->add_base_valid_ref(CONTEXT_REF, &mutator);
      derez.deserialize(sliced);
      derez.deserialize(redop);
      if (redop > 0)
      {
        derez.deserialize(deterministic_redop);
        // Only need to fill these in if we're not doing a 
        // deterministic reduction operation
        if (!deterministic_redop)
        {
          reduction_op = Runtime::get_reduction_op(redop);
          serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
          initialize_reduction_state();
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(future_handles == NULL);
#endif
        size_t num_handles;
        derez.deserialize(num_handles);
        if (num_handles > 0)
        {
          future_handles = new FutureHandles;
          future_handles->add_reference();
          std::map<DomainPoint,DistributedID> &handles = 
            future_handles->handles;
          for (unsigned idx = 0; idx < num_handles; idx++)
          {
            DomainPoint point;
            derez.deserialize(point);
            derez.deserialize(handles[point]);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::initialize_reduction_state(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_op != NULL);
      assert(reduction_op->identity);
      assert(reduction_state == NULL);
#endif
      reduction_state_size = reduction_op->sizeof_rhs;
      reduction_state = legion_malloc(REDUCTION_ALLOC, reduction_state_size);
      // If we need to initialize specially, then we do that with a serdez fn
      if (serdez_redop_fns != NULL)
        (*(serdez_redop_fns->init_fn))(reduction_op, reduction_state, 
                                       reduction_state_size);
      else
        memcpy(reduction_state, reduction_op->identity,
               reduction_op->sizeof_rhs);
    }

    //--------------------------------------------------------------------------
    void MultiTask::fold_reduction_future(const void *result, 
                                          size_t result_size, 
                                          bool owner, bool exclusive)
    //--------------------------------------------------------------------------
    {
      // Apply the reduction operation
#ifdef DEBUG_LEGION
      assert(reduction_op != NULL);
      assert(reduction_op->cpu_fold_excl_fn);
      assert(reduction_op->cpu_fold_nonexcl_fn);
      assert(reduction_state != NULL);
#endif
      // Perform the reduction, see if we have to do serdez reductions
      if (serdez_redop_fns != NULL)
      {
        // Need to hold the lock to make the serialize/deserialize
        // process atomic
        AutoLock o_lock(op_lock);
        (*(serdez_redop_fns->fold_fn))(reduction_op, reduction_state,
                                       reduction_state_size, result);
      }
      else
      {
        if (exclusive)
          (reduction_op->cpu_fold_excl_fn)(reduction_state, 0,
                                           result, 0,
                                           1, reduction_op->userdata);
        else
          (reduction_op->cpu_fold_nonexcl_fn)(reduction_state, 0,
                                              result, 0,
                                              1, reduction_op->userdata);
      }

      // If we're the owner, then free the memory
      if (owner)
        free(const_cast<void*>(result));
    } 

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(const IndividualTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask& IndividualTask::operator=(const IndividualTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_INDIVIDUAL_CALL);
      activate_single();
      predicate_false_result = NULL;
      predicate_false_size = 0;
      orig_task = this;
      remote_owner_uid = 0;
      remote_unique_id = get_unique_id();
      sent_remotely = false;
      top_level_task = false;
      implicit_top_level_task = false;
      need_intra_task_alias_analysis = true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_INDIVIDUAL_CALL);
      deactivate_single();
      if (!remote_instances.empty())
      {
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
        }
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          runtime->send_remote_context_free(it->first, rez);
        }
        remote_instances.clear();
      }
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      // Remove our reference on the future
      result = Future();
      predicate_false_future = Future();
      privilege_paths.clear(); 
      runtime->free_individual_task(this);
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::initialize_task(InnerContext *ctx,
                                           const TaskLauncher &launcher,
                                           bool track /*=true*/,
                                           bool top_level /*=false*/,
                                           bool implicit_top_level /*=false*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.argument.get_size();
      if (arglen > 0)
      {
        args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,launcher.argument.get_ptr(),arglen);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      index_point = launcher.point;
      index_domain = Domain(index_point, index_point);
      sharding_space = launcher.sharding_space;
      is_index_space = false;
      initialize_base_task(ctx, track, launcher.static_dependences,
          launcher.predicate, task_id, launcher.provenance.c_str());
      remote_owner_uid = ctx->get_unique_id();
      need_intra_task_alias_analysis = !launcher.independent_requirements;
      if (launcher.predicate != Predicate::TRUE_PRED)
      {
        if (launcher.predicate_false_future.impl != NULL)
          predicate_false_future = launcher.predicate_false_future;
        else
        {
          predicate_false_size = launcher.predicate_false_result.get_size();
          if (predicate_false_size > 0)
          {
#ifdef DEBUG_LEGION
            assert(predicate_false_result == NULL);
#endif
            predicate_false_result = 
              legion_malloc(PREDICATE_ALLOC, predicate_false_size);
            memcpy(predicate_false_result, 
                   launcher.predicate_false_result.get_ptr(),
                   predicate_false_size);
          }
        }
      } 
      if (launcher.local_function_task)
      {
        if (!regions.empty())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_LOCAL_FUNCTION_TASK_LAUNCH, 
              "Local function task launch for task %s in parent task %s "
              "(UID %lld) has %zd region requirements. Local function tasks "
              "are not permitted to have any region requirements.", 
              get_task_name(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id(), regions.size())
        local_function = true;
      }
      // Get a future from the parent context to use as the result
      result = Future(new FutureImpl(parent_ctx, runtime, true/*register*/,
            runtime->get_available_distributed_id(), 
            runtime->address_space, get_completion_event(), this));
      check_empty_field_requirements(); 
      // If this is the top-level task we can record some extra properties
      if (top_level)
      {
        this->top_level_task = true;
        this->implicit_top_level_task = implicit_top_level;
        // Top-level tasks never do dependence analysis, so we
        // need to complete those stages now
        resolve_speculation();
      }
      if (runtime->legion_spy_enabled)
      {
        if (top_level)
          LegionSpy::log_top_level_task(task_id, parent_ctx->get_unique_id(),
                                        unique_op_id, get_task_name());
        // Tracking as long as we are not part of a must epoch operation
        if (track || top_level)
        {
          LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                         unique_op_id, task_id,
                                         context_index, get_task_name());
          for (std::vector<PhaseBarrier>::const_iterator it = 
                launcher.wait_barriers.begin(); it !=
                launcher.wait_barriers.end(); it++)
          {
            ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
            LegionSpy::log_phase_barrier_wait(unique_op_id, e);
          }
          LegionSpy::log_future_creation(unique_op_id, 
                result.impl->get_ready_event(), index_point);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::initialize_must_epoch(MustEpochOp *epoch, 
                                           unsigned index, bool do_registration)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                       unique_op_id, task_id,
                                       epoch->get_ctx_index(), get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        LegionSpy::log_future_creation(unique_op_id, 
              result.impl->get_ready_event(), index_point);
      }
      set_must_epoch(epoch, index, do_registration);
      FutureMap map = epoch->get_future_map();
#ifdef DEBUG_LEGION
      map.impl->add_valid_point(index_point);
#endif
      result = map.impl->get_future(index_point);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false/*prioritize*/);
        if (inline_task) 
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_INLINE,
                          "Mapper %s requested to inline task %s "
                          "(UID %lld) but the 'enable_inlining' option was "
                          "not set on the task launcher so the request is "
                          "being ignored", mapper->get_mapper_name(),
                          get_task_name(), get_unique_id());
        }
      }
      // local function tasks have no region requirements so nothing below
      if (local_function)
        return;
      // First compute the parent indexes
      compute_parent_indexes();
      privilege_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        initialize_privilege_path(privilege_paths[idx], regions[idx]);
      update_no_access_regions();
      // If we have a trace, it is unsound to do this until the dependence
      // analysis stage when all the operations are serialized in order
      if (need_intra_task_alias_analysis)
      {
        LegionTrace *local_trace = get_trace();
        if (local_trace == NULL)
          perform_intra_task_alias_analysis(false/*tracing*/, NULL/*trace*/,
                                            privilege_paths);
      }
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
          log_requirement(unique_op_id, idx, regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo_state != MEMO_REQ);
      assert(privilege_paths.size() == regions.size());
#endif
      if (runtime->check_privileges && 
          !is_top_level_task() && !local_function)
        perform_privilege_checks();
      // If we have a trace we do our alias analysis now
      if (need_intra_task_alias_analysis && !local_function)
      {
        LegionTrace *local_trace = get_trace();
        if (local_trace != NULL)
          perform_intra_task_alias_analysis(is_tracing(), local_trace,
                                            privilege_paths);
      }
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
        if (it->impl != NULL)
          it->impl->register_dependence(this);
      if (predicate_false_future.impl != NULL)
        predicate_false_future.impl->register_dependence(this);
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                  wait_barriers, arrive_barriers, must_epoch);
      // Also have to register any dependences on our predicate
      register_predicate_dependence();
      ProjectionInfo projection_info;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        runtime->forest->perform_dependence_analysis(this, idx, regions[idx], 
                                                     projection_info,
                                                     privilege_paths[idx],
                                                     map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Dumb case for must epoch operations, we need these to 
      // be mapped immediately, mapper be damned
      if (must_epoch != NULL)
      {
        TriggerTaskArgs trigger_args(this);
        runtime->issue_runtime_meta_task(trigger_args, 
                                         LG_THROUGHPUT_WORK_PRIORITY);
      }
      // If we're replaying this for for a trace then don't even
      // bother asking the mapper about when to map this
      else if (is_replaying() || local_function)
        enqueue_ready_operation();
      // Figure out whether this task is local or remote
      else if (!runtime->is_local(target_proc))
      {
        // We can only send it away if it is not origin mapped
        // otherwise it has to stay here until it is fully mapped
        if (!is_origin_mapped())
          runtime->send_task(this);
        else
          enqueue_ready_task(false/*use target*/);
      }
      else
        enqueue_ready_task(true/*use target*/);
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::report_interfering_requirements(unsigned idx1, 
                                                         unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 1
      REPORT_LEGION_ERROR(ERROR_ALIASED_INTERFERING_REGION,
                    "Aliased and interfering region requirements for "
                    "individual tasks are not permitted. Region requirements "
                    "%d and %d of task %s (UID %lld) in parent task %s "
                    "(UID %lld) are interfering.", idx1, idx2, get_task_name(),
                    get_unique_id(), parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id())
#else
      REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENTS_INDIVIDUAL,
                      "Region requirements %d and %d of individual task "
                      "%s (UID %lld) in parent task %s (UID %lld) are "
                      "interfering.  This behavior is currently "
                      "undefined. You better really know what you are "
                      "doing.", idx1, idx2, get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
#endif
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we already launched, then return, otherwise continue
      // through and do the work to clean up the task 
      if (launched)
        return;
      // Set the future to the false result
      RtEvent execution_condition;
      if (predicate_false_future.impl != NULL)
      {
        ApEvent wait_on = predicate_false_future.impl->get_ready_event();
        if (wait_on.has_triggered_faultignorant())
        {
          const size_t result_size = 
            check_future_size(predicate_false_future.impl);
          if (result_size > 0)
            result.impl->set_result(
                predicate_false_future.impl->get_untyped_result(true,NULL,true),
                result_size, false/*own*/);
          else
            result.impl->set_result(NULL, 0, false/*own*/);
        }
        else
        {
          // Add references so they aren't garbage collected
          result.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
          predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
          DeferredFutureSetArgs args(result.impl, 
                predicate_false_future.impl, this);
          execution_condition = 
            runtime->issue_runtime_meta_task(args,LG_LATENCY_WORK_PRIORITY,
                                             Runtime::protect_event(wait_on));
        }
      }
      else
      {
        if (predicate_false_size > 0)
          result.impl->set_result(predicate_false_result,
                                  predicate_false_size, false/*own*/);
        else
          result.impl->set_result(NULL, 0, false/*own*/);
      }
      // Then clean up this task instance
      complete_mapping();
      complete_execution(execution_condition);
      resolve_speculation();
      trigger_children_complete(ApEvent::NO_AP_EVENT);
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // Nothing to do for now
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualTask::perform_mapping(
                                        MustEpochOp *must_epoch_owner/*=NULL*/, 
                                        const DeferMappingArgs *args/* =NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PERFORM_MAPPING_CALL);
      const RtEvent deferred = map_all_regions(must_epoch_owner, args);
      if (deferred.exists())
        return deferred; 
      // If we mapped, then we are no longer stealable
      stealable = false;
      // If we succeeded in mapping and it's a leaf task
      // then we get to mark that we are done mapping
      RtEvent applied_condition;
      if (is_leaf())
      {
        if (!map_applied_conditions.empty())
        {
          applied_condition = Runtime::merge_events(map_applied_conditions);
          map_applied_conditions.clear();
        }
        // If we mapped remotely we might have a deferred complete mapping
        // that we can trigger now
        if (deferred_complete_mapping.exists())
        {
#ifdef DEBUG_LEGION
          assert(is_remote());
#endif
          Runtime::trigger_event(deferred_complete_mapping, applied_condition);
          applied_condition = deferred_complete_mapping;
          deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      else if (!is_remote())
      {
        // We did this mapping on the owner
#ifdef DEBUG_LEGION
        assert(!deferred_complete_mapping.exists());
#endif
        deferred_complete_mapping = Runtime::create_rt_user_event();
        applied_condition = deferred_complete_mapping;
      }
      else
      {
        // We did this mapping remotely so there better be an event
#ifdef DEBUG_LEGION
        assert(deferred_complete_mapping.exists());
#endif
        applied_condition = deferred_complete_mapping;
      } 
      // Mark that we have completed mapping
      if (!acquired_instances.empty())
        applied_condition = release_nonempty_acquired_instances(
                          applied_condition, acquired_instances);
      complete_mapping(applied_condition);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_inlining(VariantImpl *variant,
                                  const std::deque<InstanceSet> &parent_regions)
    //--------------------------------------------------------------------------
    {
      SingleTask::perform_inlining(variant, parent_regions);
      // We also need to resolve speculation which PointTasks do not
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
    }

    //--------------------------------------------------------------------------
    VersionInfo& IndividualTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const VersionInfo& IndividualTask::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndividualTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < privilege_paths.size());
#endif
      return privilege_paths[idx];
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndividualTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDIVIDUAL_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::send_remote_context(AddressSpaceID remote_instance,
                                             RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_instance != runtime->address_space);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        execution_context->pack_remote_context(rez, remote_instance);
      }
      runtime->send_remote_context_response(remote_instance, rez);
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMPLETE_CALL);
      // Invalidate any state that we had if we didn't already
      // Do this before sending the complete message to avoid the
      // race condition in the remote case where the top-level
      // context cleans on the owner node while we still need it
      if (execution_context != NULL)
      {
        execution_context->invalidate_region_tree_contexts();
        if (runtime->legion_spy_enabled)
          execution_context->log_created_requirements();
      }
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      std::set<RtEvent> completion_preconditions;
      if (!is_remote())
      {
        // Pass back our created and deleted operations
        if (execution_context != NULL)
        {
          if (top_level_task)
            execution_context->report_leaks_and_duplicates(
                                  completion_preconditions);
          else
            execution_context->return_resources(parent_ctx, 
                   context_index, completion_preconditions);
        }
      }
      else
      {
        Serializer rez;
        pack_remote_complete(rez);
        runtime->send_individual_remote_complete(orig_proc,rez);
      }
      // See if we need to trigger that our children are complete
      // Note it is only safe to do this if we were not sent remotely
      bool need_commit = false;
      if (!sent_remotely && (execution_context != NULL))
        need_commit = execution_context->attempt_children_commit();
      if (must_epoch != NULL)
      {
        RtEvent precondition;
        if (!completion_preconditions.empty())
          precondition = Runtime::merge_events(completion_preconditions);
        must_epoch->notify_subop_complete(this, precondition);
        complete_operation();
      }
      else
      {
        // We can only do an early completer here if there are
        // no other completion preconditions, otherwise actual
        // application execution needs to wait for all the other
        // resources to be returned.
        if (!completion_preconditions.empty())
        {
          completion_preconditions.insert(
              Runtime::protect_event(task_effects_complete));
          complete_operation(Runtime::merge_events(completion_preconditions));
        }
        else
        {
          if (!request_early_complete(task_effects_complete))
            complete_operation(Runtime::protect_event(task_effects_complete));
          else
            complete_operation();
        }
      }
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMMIT_CALL);
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez);
        runtime->send_individual_remote_commit(orig_proc,rez);
      }
      if (profiling_reported.exists())
        finalize_single_task_profiling();
      if (must_epoch != NULL)
      {
        must_epoch->notify_subop_commit(this, profiling_reported);
        commit_operation(true/*deactivate*/, profiling_reported);
      }
      else
        commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future(const void *res, size_t res_size,
                      bool owned, FutureFunctor *functor, Processor future_proc)
    //--------------------------------------------------------------------------
    {
      if (functor != NULL)
        result.impl->set_result(functor, owned, future_proc);
      else
        result.impl->set_result(res, res_size, owned);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_post_mapped(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_POST_MAPPED_CALL);
      if (deferred_complete_mapping.exists())
      {
        if (mapped_precondition.exists())
          map_applied_conditions.insert(mapped_precondition);
        // Little race condition here so pull it on the stack first
        RtUserEvent to_trigger = deferred_complete_mapping;
        deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        if (!map_applied_conditions.empty())
          Runtime::trigger_event(to_trigger, 
              Runtime::merge_events(map_applied_conditions)); 
        else
          Runtime::trigger_event(to_trigger);
      }
      // If we're an implicit top-level we do our complete mapping call here
      else if (top_level_task && implicit_top_level_task)
        complete_mapping(mapped_precondition);
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
      {
        assert(!mapped_precondition.exists());
        assert(map_applied_conditions.empty());
      }
#endif
#endif
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::handle_misspeculation(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
#ifdef DEBUG_LEGION
      runtime->increment_total_outstanding_tasks(
          MisspeculationTaskArgs::TASK_ID, true/*meta*/);
#else
      runtime->increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MisspeculationTaskArgs::TASK_ID].fetch_add(1);
#endif
      // Pretend like we executed the task
      execution_context->begin_misspeculation();
      if (predicate_false_future.impl != NULL)
      {
        // Wait for the future to be ready
        ApEvent wait_on = predicate_false_future.impl->get_ready_event();
        bool poisoned = false;
        wait_on.wait_faultaware(poisoned);
        if (poisoned)
          execution_context->raise_poison_exception();
        void *ptr = 
          predicate_false_future.impl->get_untyped_result(true, NULL, true);
        size_t size = predicate_false_future.impl->get_untyped_size(true);
        execution_context->end_misspeculation(ptr, size); 
      }
      else
        execution_context->end_misspeculation(predicate_false_result,
                                              predicate_false_size);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_TASK_CALL);
      // Check to see if we are stealable, if not and we have not
      // yet been sent remotely, then send the state now
      RezCheck z(rez);
      pack_single_task(rez, target);
      rez.serialize(orig_task);
      rez.serialize(remote_unique_id);
      rez.serialize(remote_owner_uid);
      rez.serialize(top_level_task);
      rez.serialize(result.impl->did);
      if (predicate_false_future.impl != NULL)
        rez.serialize(predicate_false_future.impl->did);
      else
        rez.serialize<DistributedID>(0);
      rez.serialize(predicate_false_size);
      if (predicate_false_size > 0)
        rez.serialize(predicate_false_result, predicate_false_size);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      // Mark that we sent this task remotely
      sent_remotely = true;
      // If this task is remote, then deactivate it, otherwise
      // we're local so we don't want to be deactivated for when
      // return messages get sent back.
      return is_remote();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::unpack_task(Deserializer &derez, Processor current,
                                     std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      derez.deserialize(orig_task);
      derez.deserialize(remote_unique_id);
      set_current_proc(current);
      derez.deserialize(remote_owner_uid);
      derez.deserialize(top_level_task);
      // Quick check to see if we've been sent back to our original node
      if (!is_remote())
      {
#ifdef DEBUG_LEGION
        // Need to make the deserializer happy in debug mode
        // 2 *sizeof(size_t) since we're two DerezChecks deep
        derez.advance_pointer(derez.get_remaining_bytes() - 2*sizeof(size_t));
#endif
        // If we were sent back then mark that we are no longer remote
        orig_task->sent_remotely = false;
        if (deferred_complete_mapping.exists())
        {
          orig_task->deferred_complete_mapping = deferred_complete_mapping;
          deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        }
        // Put the original instance back on the mapping queue and
        // deactivate this version of the task
        orig_task->enqueue_ready_task(false/*target*/);
        deactivate();
        return false;
      }
      DistributedID future_did;
      derez.deserialize(future_did);
      {
        WrapperReferenceMutator mutator(ready_events);
        FutureImpl *impl = 
          runtime->find_or_create_future(future_did, remote_owner_uid,&mutator);
        impl->add_base_gc_ref(FUTURE_HANDLE_REF, &mutator);
        result = Future(impl, false/*need reference*/);
      }
      // Unpack the predicate false infos
      DistributedID pred_false_did;
      derez.deserialize(pred_false_did);
      if (pred_false_did != 0)
      {
        WrapperReferenceMutator mutator(ready_events);
        FutureImpl *impl = runtime->find_or_create_future(pred_false_did,
                                              remote_owner_uid, &mutator);
        impl->add_base_gc_ref(FUTURE_HANDLE_REF, &mutator);
        predicate_false_future = Future(impl, false/*need reference*/);
      }
      derez.deserialize(predicate_false_size);
      if (predicate_false_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(predicate_false_result == NULL);
#endif
        predicate_false_result = malloc(predicate_false_size);
        derez.deserialize(predicate_false_result, predicate_false_size);
      }
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
      // Figure out what our parent context is
      RtEvent ctx_ready;
      parent_ctx = runtime->find_context(remote_owner_uid, false, &ctx_ready);
      if (ctx_ready.exists() && !ctx_ready.has_triggered())
      {
        // Wait if the profiler is going to ask for the context
        if (runtime->profiler != NULL)
          ctx_ready.wait();
        else
          ready_events.insert(ctx_ready);
      }
      // Set our parent task for the user
      parent_task = parent_ctx->get_task();
      // Have to do this before resolving speculation in case
      // we get cleaned up after the resolve speculation call
      if (runtime->legion_spy_enabled)
        LegionSpy::log_point_point(remote_unique_id, get_unique_id());
      // If we're remote, we've already resolved speculation for now
      resolve_speculation();
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      // Return true to add ourselves to the ready queue
      return true;
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer &rez) 
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_REMOTE_COMPLETE_CALL);
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      if (execution_context->has_created_requirements())
        execution_context->send_back_created_state(target); 
      // Send back the pointer to the task instance, then serialize
      // everything else that needs to be sent back
      rez.serialize(orig_task);
      RezCheck z(rez);
      // Pack the privilege state
      execution_context->pack_resources_return(rez, context_index);
      if (!is_origin_mapped())
        rez.serialize(task_effects_complete);
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_REMOTE_COMPLETE_CALL);
      DerezCheck z(derez);
      // First unpack the privilege state
      const RtEvent resources_returned = 
        ResourceTracker::unpack_resources_return(derez, parent_ctx);
      if (!is_origin_mapped())
        derez.deserialize(task_effects_complete);
      // Mark that we have both finished executing and that our
      // children are complete
      complete_execution(resources_returned);
      TaskOp::trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_commit(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
      }
      tpl->register_operation(this);
      complete_mapping();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!target_processors.empty());
#endif
      const AddressSpaceID target_space = 
        runtime->find_address_space(target_processors.front());
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        // This is the remote case, pack it up and ship it over
        // Mark that we are effecitvely mapping this at the origin
        map_origin = true;
        // Pack this task up and send it to the remote node
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(target_processors.front());
          rez.serialize(INDIVIDUAL_TASK_KIND);
          pack_task(rez, target_space);
        }
        runtime->send_remote_task_replay(target_space, rez);
      }
      else
      { 
#ifdef DEBUG_LEGION
        assert(is_leaf());
#endif
        for (std::deque<InstanceSet>::iterator it = physical_instances.begin();
             it != physical_instances.end(); ++it)
          for (unsigned idx = 0; idx < it->size(); ++idx)
            (*it)[idx].set_ready_event(instance_ready_event);
        execution_fence_event = instance_ready_event;
        update_no_access_regions();
        launch_task();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_complete(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_commit(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_commit(derez);
    }

    /////////////////////////////////////////////////////////////
    // Point Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointTask::PointTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask::PointTask(const PointTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask& PointTask::operator=(const PointTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_ACTIVATE_CALL);
      activate_single();
      // Point tasks never have to resolve speculation
      resolve_speculation();
      orig_task = this;
      slice_owner = NULL;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_DEACTIVATE_CALL);
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(
            this->slice_owner->get_unique_op_id(),
            this->get_unique_op_id());
      deactivate_single();
      if (!remote_instances.empty())
      {
        UniqueID local_uid = get_unique_id();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(local_uid);
        }
        for (std::map<AddressSpaceID,RemoteTask*>::const_iterator it = 
              remote_instances.begin(); it != remote_instances.end(); it++)
        {
          runtime->send_remote_context_free(it->first, rez);
        }
        remote_instances.clear();
      }
      runtime->free_point_task(this);
    } 

    //--------------------------------------------------------------------------
    bool PointTask::is_reducing_future(void) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->is_reducing_future();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::report_interfering_requirements(unsigned idx1,
                                                    unsigned idx2)
    //--------------------------------------------------------------------------
    {
      switch (index_point.get_dim())
      {
        case 1:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point %lld of index space task %s (UID %lld) "
                    "in parent task %s (UID %lld) are interfering.", 
                    idx1, idx2, index_point[0], get_task_name(),
                    get_unique_id(), parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id());
            break;
          }
#if LEGION_MAX_DIM >= 2
        case 2:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld) of index space task %s "
                    "(UID %lld) in parent task %s (UID %lld) are interfering.",
                    idx1, idx2, index_point[0], index_point[1], 
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
        case 3:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld) of index space task %s"
                    " (UID %lld) in parent task %s (UID %lld) are interfering.",
                    idx1, idx2, index_point[0], index_point[1], 
                    index_point[2], get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
        case 4:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld) of index space " 
                    "task %s (UID %lld) in parent task %s (UID %lld) are "
                    "interfering.", idx1, idx2, index_point[0], index_point[1],
                    index_point[2], index_point[3], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
        case 5:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld) of index "
                    "space task %s (UID %lld) in parent task %s (UID %lld) are "
                    "interfering.", idx1, idx2, index_point[0], index_point[1],
                    index_point[2], index_point[3], index_point[4], 
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
        case 6:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld) of "
                    "index space task %s (UID %lld) in parent task %s (UID "
                    "%lld) are interfering.", idx1, idx2, index_point[0], 
                    index_point[1], index_point[2], index_point[3], 
                    index_point[4], index_point[5], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
        case 7:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld) of"
                    " index space task %s (UID %lld) in parent task %s (UID "
                    "%lld) are interfering.", idx1, idx2, index_point[0], 
                    index_point[1], index_point[2], index_point[3], 
                    index_point[4], index_point[5], index_point[6],
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
        case 8:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                    "%lld) of index space task %s (UID %lld) in parent task "
                    "%s (UID %lld) are interfering.", idx1, idx2, 
                    index_point[0], index_point[1], index_point[2], 
                    index_point[3], index_point[4], index_point[5], 
                    index_point[6], index_point[7], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
        case 9:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                    "%lld,%lld) of index space task %s (UID %lld) in parent "
                    "task %s (UID %lld) are interfering.", idx1, idx2, 
                    index_point[0], index_point[1], index_point[2], 
                    index_point[3], index_point[4], index_point[5], 
                    index_point[6], index_point[7], index_point[8],
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are always done with early mapping
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are never sent anywhere
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::perform_mapping(MustEpochOp *must_epoch_owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // For point tasks we use the point termination event which as the
      // end event for this task since point tasks can be moved and
      // the completion event is therefore not guaranteed to survive
      // the length of the task's execution
      const RtEvent deferred = map_all_regions(must_epoch_owner, args);
      if (deferred.exists())
        return deferred;
      RtEvent applied_condition;
      // If we succeeded in mapping and we're a leaf so we are done mapping
      if (is_leaf()) 
      {
        if (!map_applied_conditions.empty())
        {
          applied_condition = Runtime::merge_events(map_applied_conditions);
          map_applied_conditions.clear();
        }
        // If we mapped remotely we might have a deferred complete mapping
        // that we can trigger now
        if (deferred_complete_mapping.exists())
        {
#ifdef DEBUG_LEGION
          assert(is_remote());
#endif
          Runtime::trigger_event(deferred_complete_mapping, applied_condition);
          applied_condition = deferred_complete_mapping;
          deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(!deferred_complete_mapping.exists());
#endif
        deferred_complete_mapping = Runtime::create_rt_user_event();
        applied_condition = deferred_complete_mapping;
      }
      slice_owner->record_point_mapped(applied_condition, task_effects_complete,
                                       acquired_instances);
      complete_mapping(applied_condition);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    VersionInfo& PointTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // See if we've copied over the versions from our slice
      // if not we can just use our slice owner
      if (idx < version_infos.size())
        return version_infos[idx];
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    const VersionInfo& PointTask::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // See if we've copied over the versions from our slice
      // if not we can just use our slice owner
      if (idx < version_infos.size())
        return version_infos[idx];
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind PointTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POINT_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void PointTask::send_remote_context(AddressSpaceID remote_instance,
                                        RemoteTask *remote_ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_instance != runtime->address_space);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(remote_ctx);
        execution_context->pack_remote_context(rez, remote_instance);
      }
      runtime->send_remote_context_response(remote_instance, rez);
      AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
      assert(remote_instances.find(remote_instance) == remote_instances.end());
#endif
      remote_instances[remote_instance] = remote_ctx;
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_COMPLETE_CALL);
      // Pass back our created and deleted operations 
      std::set<RtEvent> preconditions;
      if (execution_context != NULL)
      {
        slice_owner->return_privileges(execution_context, preconditions);
        if (!preconditions.empty())
          slice_owner->record_point_complete(
              Runtime::merge_events(preconditions));
        else
          slice_owner->record_point_complete(RtEvent::NO_RT_EVENT);
        if (runtime->legion_spy_enabled)
          execution_context->log_created_requirements();
        // Invalidate any context that we had so that the child
        // operations can begin committing
        execution_context->invalidate_region_tree_contexts(); 
        // See if we need to trigger that our children are complete
        const bool need_commit = execution_context->attempt_children_commit();
        // Mark that this operation is now complete
        complete_operation();
        if (need_commit)
          trigger_children_committed();
      }
      else
      {
        if (!preconditions.empty())
          slice_owner->record_point_complete(
              Runtime::merge_events(preconditions));
        else
          slice_owner->record_point_complete(RtEvent::NO_RT_EVENT);

        complete_operation();
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_COMMIT_CALL);
      if (profiling_reported.exists())
        finalize_single_task_profiling();
      // A little strange here, but we don't directly commit this
      // operation, instead we just tell our slice that we are commited
      // In the deactivation of the slice task is when we will actually
      // have our commit call done
      slice_owner->record_point_committed(profiling_reported);
    }

    //--------------------------------------------------------------------------
    bool PointTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_PACK_TASK_CALL);
      RezCheck z(rez);
      pack_single_task(rez, target);
      rez.serialize(orig_task);
#ifdef DEBUG_LEGION
      assert(is_origin_mapped()); // should be origin mapped if we're here
#endif
      // Return false since point tasks should always be deactivated
      // once they are sent to a remote node
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      derez.deserialize(orig_task);
      set_current_proc(current);
      // Get the context information from our slice owner
      parent_ctx = slice_owner->get_context();
      parent_task = parent_ctx->get_task();
      provenance = slice_owner->get_provenance();
      if (provenance != NULL)
        provenance->add_reference();
      // We should always just apply these things now since we were mapped 
      // on the owner node
#ifdef DEBUG_LEGION
      assert(is_origin_mapped());
#endif
      slice_owner->record_point_mapped(deferred_complete_mapping,
                         ApEvent::NO_AP_EVENT, acquired_instances);
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future(const void *res, size_t res_size, 
                      bool owner, FutureFunctor *functor, Processor future_proc)
    //--------------------------------------------------------------------------
    {
      slice_owner->handle_future(index_point, res, res_size,
                                 owner, functor, future_proc);
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_post_mapped(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_POST_MAPPED_CALL);
      if (deferred_complete_mapping.exists())
      {
        if (mapped_precondition.exists())
          map_applied_conditions.insert(mapped_precondition);
        // Little race condition here so pull it on the stack first
        RtUserEvent to_trigger = deferred_complete_mapping;
        deferred_complete_mapping = RtUserEvent::NO_RT_USER_EVENT;
        if (!map_applied_conditions.empty())
          Runtime::trigger_event(to_trigger, 
              Runtime::merge_events(map_applied_conditions)); 
        else
          Runtime::trigger_event(to_trigger);
      }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
      else
      {
        assert(!mapped_precondition.exists());
        assert(map_applied_conditions.empty());
      }
#endif
#endif
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_misspeculation(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
#ifdef DEBUG_LEGION
      runtime->increment_total_outstanding_tasks(
          MisspeculationTaskArgs::TASK_ID, true/*meta*/);
#else
      runtime->increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MisspeculationTaskArgs::TASK_ID].fetch_add(1);
#endif
      // Pretend like we executed the task
      execution_context->begin_misspeculation();
      size_t result_size;
      const void *result = slice_owner->get_predicate_false_result(result_size);
      execution_context->end_misspeculation(result, result_size);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointTask::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointTask::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < regions.size());
#endif
      RegionRequirement &req = regions[idx];
#ifdef DEBUG_LEGION
      assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
#endif
      req.region = result;
      req.handle_type = LEGION_SINGULAR_PROJECTION;
      // Check to see if the region is a NO_REGION,
      // if it is then switch the privilege to NO_ACCESS
      if (req.region == LogicalRegion::NO_REGION)
        req.privilege = LEGION_NO_ACCESS;
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(SliceTask *owner, const DomainPoint &point,
                                    const FutureMap &point_arguments,
                                    const std::vector<FutureMap> &point_futures)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      // Get our point
      index_point = point;
      // Get our argument
      if (point_arguments.impl != NULL)
      {
        Future f = point_arguments.impl->get_future(point);
        if (f.impl != NULL)
        {
          ApEvent ready = f.impl->get_ready_event();
          ready.wait_faultignorant();
          local_arglen = f.impl->get_untyped_size(true/*internal*/);
          // Have to make a local copy since the point takes ownership
          if (local_arglen > 0)
          {
            local_args = malloc(local_arglen);
            memcpy(local_args, 
                f.impl->get_untyped_result(true, NULL, true), local_arglen);
          }
        }
      }
      if (!point_futures.empty())
      {
        for (std::vector<FutureMap>::const_iterator it = 
              point_futures.begin(); it != point_futures.end(); it++)
          this->futures.push_back(it->impl->get_future(point));
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::send_back_created_state(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (execution_context->has_created_requirements())
        execution_context->send_back_created_state(target);
    } 

    //--------------------------------------------------------------------------
    void PointTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      complete_mapping();
    }

    //--------------------------------------------------------------------------
    void PointTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_leaf());
      assert(is_origin_mapped());
      assert(!target_processors.empty());
#endif
      const AddressSpaceID target_space = 
        runtime->find_address_space(target_processors.front());
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        slice_owner->record_point_mapped(RtEvent::NO_RT_EVENT,
                    task_effects_complete, acquired_instances);
        // Record this slice as an origin-mapped slice so that it
        // will be deactivated accordingly
        slice_owner->index_owner->record_origin_mapped_slice(slice_owner);
        // This is the remote case, pack it up and ship it over
        // Update our target_proc so that the sending code is correct 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(target_processors.front());
          rez.serialize(SLICE_TASK_KIND);
          slice_owner->pack_task(rez, target_space);
        }
        runtime->send_remote_task_replay(target_space, rez);
      }
      else
      {
        // This is the local case
        if (!is_remote())
          slice_owner->record_point_mapped(RtEvent::NO_RT_EVENT,
                      task_effects_complete, acquired_instances);
        for (std::deque<InstanceSet>::iterator it = physical_instances.begin();
             it != physical_instances.end(); ++it)
          for (unsigned idx = 0; idx < it->size(); ++idx)
            (*it)[idx].set_ready_event(instance_ready_event);
        update_no_access_regions();
        launch_task();
      }
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointTask::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, get_domain_point());
    }

    //--------------------------------------------------------------------------
    CollectiveManager* PointTask::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index_point.get_dim() > 0);
#endif
      collective_point = index_point;
      return slice_owner->find_or_create_collective_instance(mapper_call, index,
          constraints, regions, kind, footprint, unsat_kind, unsat_index, 
          collective_point);
    }

    //--------------------------------------------------------------------------
    bool PointTask::finalize_collective_instance(MappingCallKind call_kind,
                                                 unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      return slice_owner->finalize_collective_instance(call_kind,index,success);
    }

    //--------------------------------------------------------------------------
    void PointTask::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      slice_owner->report_total_collective_instance_calls(mapper_call, 
                                                          total_calls);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_intra_space_dependences(unsigned index,
                                    const std::vector<DomainPoint> &dependences)
    //--------------------------------------------------------------------------
    {
      // Scan through the list until we find ourself
      for (unsigned idx = 0; idx < dependences.size(); idx++)
      {
        if (dependences[idx] == index_point)
        {
          // If we've got a prior dependence then record it
          if (idx > 0)
          {
            const DomainPoint &prev = dependences[idx-1];
            const RtEvent pre = slice_owner->find_intra_space_dependence(prev);
            intra_space_mapping_dependences.insert(pre);
            if (runtime->legion_spy_enabled)
            {
              // We know we only need a dependence on the previous point but
              // Legion Spy is stupid, so log everything we have a
              // precondition on even if it is transitively implied
              for (unsigned idx2 = 0; idx2 < idx; idx2++)
                LegionSpy::log_intra_space_dependence(unique_op_id,
                                                      dependences[idx2]);
            }
          }
          // If we're not the last dependence, then send our mapping event
          // so that others can record a dependence on us
          if (idx < (dependences.size()-1))
            slice_owner->record_intra_space_dependence(index_point,
                                                       get_mapped_event());
          return;
        }
      }
      // We should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::has_remaining_inlining_dependences(
                   std::map<PointTask*,unsigned> &remaining,
                   std::map<RtEvent,std::vector<PointTask*> > &event_deps) const
    //--------------------------------------------------------------------------
    {
      if (intra_space_mapping_dependences.empty())
        return false;
      unsigned count = 0;
      for (std::set<RtEvent>::const_iterator it =
            intra_space_mapping_dependences.begin(); it !=
            intra_space_mapping_dependences.end(); it++)
      {
        if (it->has_triggered())
          continue;
        count++;
        event_deps[*it].push_back(const_cast<PointTask*>(this));
      }
      if (count > 0)
      {
        remaining[const_cast<PointTask*>(this)] = count;
        return true;
      }
      else
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Index Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(Runtime *rt)
      : CollectiveInstanceCreator<MultiTask>(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(const IndexTask &rhs)
      : CollectiveInstanceCreator<MultiTask>(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask& IndexTask::operator=(const IndexTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_ACTIVATE_CALL);
      activate_multi();
      serdez_redop_fns = NULL;
      total_points = 0;
      mapped_points = 0;
      complete_points = 0;
      committed_points = 0;
      need_intra_task_alias_analysis = true;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_DEACTIVATE_CALL);
      deactivate_multi();
      privilege_paths.clear();
      if (!origin_mapped_slices.empty())
      {
        for (std::set<SliceTask*>::const_iterator it = 
              origin_mapped_slices.begin(); it != 
              origin_mapped_slices.end(); it++)
        {
          (*it)->deactivate();
        }
        origin_mapped_slices.clear();
      } 
      if (future_map_ready.exists() && !future_map_ready.has_triggered())
        Runtime::trigger_event(future_map_ready);
      // Remove our reference to the reduction future
      reduction_future = Future();
      map_applied_conditions.clear();
      complete_effects.clear();
      complete_preconditions.clear();
      commit_preconditions.clear();
      version_infos.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
#ifdef DEBUG_LEGION
      interfering_requirements.clear();
      point_requirements.clear();
      assert(pending_intra_space_dependences.empty());
#endif 
      runtime->free_index_task(this);
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(InnerContext *ctx,
                                         const IndexTaskLauncher &launcher,
                                         IndexSpace launch_sp, 
                                         bool track /*= true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments = launcher.argument_map.impl->freeze(parent_ctx);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] = 
            launcher.point_futures[idx].impl->freeze(parent_ctx);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_launch_space_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      need_intra_task_alias_analysis = !launcher.independent_requirements;
      initialize_base_task(ctx, track, launcher.static_dependences,
          launcher.predicate, task_id, launcher.provenance.c_str());
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(launcher.predicate_false_future,
                             launcher.predicate_false_result);
      future_map_ready = Runtime::create_rt_user_event();
      future_map = FutureMap(new FutureMapImpl(ctx, this, future_map_ready,
            runtime, runtime->get_available_distributed_id(),
            runtime->address_space));
#ifdef DEBUG_LEGION
      future_map.impl->add_valid_domain(index_domain);
#endif
      check_empty_field_requirements(); 
 
      if (runtime->legion_spy_enabled && track)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  context_index, get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              launcher.wait_barriers.begin(); it !=
              launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      return future_map;
    }

    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(InnerContext *ctx,
                                      const IndexTaskLauncher &launcher,
                                      IndexSpace launch_sp,
                                      ReductionOpID redop_id, 
                                      bool deterministic,
                                      bool track /*= true*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments = launcher.argument_map.impl->freeze(parent_ctx);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] = 
            launcher.point_futures[idx].impl->freeze(parent_ctx);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_launch_space_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      need_intra_task_alias_analysis = !launcher.independent_requirements;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      deterministic_redop = deterministic;
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->identity)
        REPORT_LEGION_ERROR(ERROR_REDUCTION_OPERATION_INDEX,
                      "Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id())
      else
        initialize_reduction_state();
      initialize_base_task(ctx, track, launcher.static_dependences,
          launcher.predicate, task_id, launcher.provenance.c_str());
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(launcher.predicate_false_future,
                             launcher.predicate_false_result);
      reduction_future = Future(new FutureImpl(parent_ctx, runtime,
            true/*register*/, runtime->get_available_distributed_id(), 
            runtime->address_space, get_completion_event(), this));
      check_empty_field_requirements();
      if (runtime->legion_spy_enabled && track)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  context_index, get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              launcher.wait_barriers.begin(); it !=
              launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        LegionSpy::log_future_creation(unique_op_id, 
              reduction_future.impl->get_ready_event(), index_point);
      }
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_predicate(const Future &pred_future,
                                         const UntypedBuffer &pred_arg)
    //--------------------------------------------------------------------------
    {
      if (pred_future.impl != NULL)
        predicate_false_future = pred_future;
      else
      {
        predicate_false_size = pred_arg.get_size();
        if (predicate_false_size > 0)
        {
#ifdef DEBUG_LEGION
          assert(predicate_false_result == NULL);
#endif
          predicate_false_result = 
            legion_malloc(PREDICATE_ALLOC, predicate_false_size);
          memcpy(predicate_false_result, pred_arg.get_ptr(),
                 predicate_false_size);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_must_epoch(MustEpochOp *epoch, 
                                          unsigned index, bool do_registration)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id,
                                  epoch->get_ctx_index(), get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              wait_barriers.begin(); it != wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      set_must_epoch(epoch, index, do_registration);
      future_map = epoch->get_future_map();
#ifdef DEBUG_LEGION
      Domain launch_domain;
      launch_space->get_launch_space_domain(launch_domain);
      future_map.impl->add_valid_domain(launch_domain);
#endif 
      // Enumerate the futures in the future map
      enumerate_futures(index_domain);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent indexes
      compute_parent_indexes(); 
      // Count how many total points we need for this index space task
      total_points = index_domain.get_volume();
      // Annotate any regions which are going to need to be early mapped
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement &req = regions[idx];
        // any region requirements which were marked singular need to
        // be promoted up to being a region projection with depth 0
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;
        }
        if (!IS_WRITE(req))
          continue;
        if (req.handle_type == LEGION_REGION_PROJECTION)
        {
          if (req.projection > 0)
          {
            ProjectionFunction *function =
              runtime->find_projection_function(req.projection);
            if (function->depth == 0)
              req.flags |= LEGION_MUST_PREMAP_FLAG;
          }
          else
            req.flags |= LEGION_MUST_PREMAP_FLAG;
        }
      }
      // Initialize the privilege paths
      privilege_paths.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        initialize_privilege_path(privilege_paths[idx], regions[idx]);
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false/*prioritize*/);
        if (inline_task) 
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_INLINE,
                          "Mapper %s requested to inline task %s "
                          "(UID %lld) but the 'enable_inlining' option was "
                          "not set on the task launcher so the request is "
                          "being ignored", mapper->get_mapper_name(),
                          get_task_name(), get_unique_id());
        }
      }
      if (need_intra_task_alias_analysis)
      {
        // If we don't have a trace, we do our alias analysis now
        LegionTrace *local_trace = get_trace();
        if (local_trace == NULL)
          perform_intra_task_alias_analysis(false/*tracing*/, NULL/*trace*/,
                                            privilege_paths);
      }
      if (runtime->legion_spy_enabled)
      { 
        for (unsigned idx = 0; idx < regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo_state != MEMO_REQ);
      assert(privilege_paths.size() == regions.size());
#endif 
      if (runtime->check_privileges)
        perform_privilege_checks();
      if (need_intra_task_alias_analysis)
      {
        // If we have a trace we do our alias analysis now
        LegionTrace *local_trace = get_trace();
        if (local_trace != NULL)
          perform_intra_task_alias_analysis(is_tracing(), local_trace,
                                            privilege_paths);
      }
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
        if (it->impl != NULL)
          it->impl->register_dependence(this);
      if (predicate_false_future.impl != NULL)
        predicate_false_future.impl->register_dependence(this);
      // Register mapping dependences on any future maps also
      if (point_arguments.impl != NULL)
        point_arguments.impl->register_dependence(this);
      for (std::vector<FutureMap>::const_iterator it = 
            point_futures.begin(); it != point_futures.end(); it++)
        it->impl->register_dependence(this);
      // Also have to register any dependences on our predicate
      register_predicate_dependence();
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                  wait_barriers, arrive_barriers, must_epoch);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        ProjectionInfo projection_info(runtime, regions[idx], launch_space);
        runtime->forest->perform_dependence_analysis(this, idx, regions[idx], 
                                                     projection_info,
                                                     privilege_paths[idx],
                                                     map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_requirements(unsigned idx1,unsigned idx2)
    //--------------------------------------------------------------------------
    {
#if 0
      REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                          "Aliased region requirements for index tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are interfering.", idx1, idx2, get_task_name(),
                          get_unique_id(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
#else
      REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENTS_INDEX,
                      "Region requirements %d and %d of index task %s "
                      "(UID %lld) in parent task %s (UID %lld) are potentially "
                      "interfering.  It's possible that this is a false "
                      "positive if there are projection region requirements "
                      "and each of the point tasks are non-interfering. "
                      "If the runtime is built in debug mode then it will "
                      "check that the region requirements of all points are "
                      "actually non-interfering. If you see no further error "
                      "messages for this index task launch then everything "
                      "is good.", idx1, idx2, get_task_name(), get_unique_id(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
#endif
#ifdef DEBUG_LEGION
      interfering_requirements.insert(std::pair<unsigned,unsigned>(idx1,idx2));
#endif
    }

    //--------------------------------------------------------------------------
    RegionTreePath& IndexTask::get_privilege_path(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < privilege_paths.size());
#endif
      return privilege_paths[idx];
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do a quick test for empty index space launches
      if (launch_space->is_empty())
      {
        // Clean up this task execution if there are no points
        complete_mapping();
        complete_execution();
        trigger_children_complete();
        trigger_children_committed();
      }
      else
      {
        // Enumerate the futures in the future map
        if (redop == 0)
          enumerate_futures(index_domain);
        Operation::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::enumerate_futures(const Domain &domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(future_handles == NULL);
#endif
      future_handles = new FutureHandles;
      future_handles->add_reference();
      std::map<DomainPoint,DistributedID> &handles = future_handles->handles;
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        Future f = future_map.impl->get_future(itr.p);
        handles[itr.p] = f.impl->did;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we already launched, then we can just return
      // otherwise continue through to do the cleanup work
      if (launched)
        return;
      RtEvent execution_condition;
      // Fill in the index task map with the default future value
      if (redop == 0)
      {
        // Handling the future map case
        if (predicate_false_future.impl != NULL)
        {
          ApEvent wait_on = predicate_false_future.impl->get_ready_event();
          if (wait_on.has_triggered_faultignorant())
          {
            const size_t result_size = 
              check_future_size(predicate_false_future.impl);
            const void *result = 
              predicate_false_future.impl->get_untyped_result(true, NULL, true);
            for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
            {
              Future f = future_map.impl->get_future(itr.p);
              if (result_size > 0)
                f.impl->set_result(result, result_size, false/*own*/);
              else
                f.impl->set_result(NULL, 0, false/*own*/);
            }
          }
          else
          {
            // Add references so things won't be prematurely collected
            future_map.impl->add_base_resource_ref(DEFERRED_TASK_REF);
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF,
                                                         this);
            DeferredFutureMapSetArgs args(future_map.impl,
                  predicate_false_future.impl, index_domain, this);
            execution_condition = 
              runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                               Runtime::protect_event(wait_on));
          }
        }
        else
        {
          for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
          {
            Future f = future_map.impl->get_future(itr.p);
            if (predicate_false_size > 0)
              f.impl->set_result(predicate_false_result,
                                 predicate_false_size, false/*own*/);
            else
              f.impl->set_result(NULL, 0, false/*own*/);
          }
        }
      }
      else
      {
        // Handling a reduction case
        if (predicate_false_future.impl != NULL)
        {
          ApEvent wait_on = predicate_false_future.impl->get_ready_event();
          if (wait_on.has_triggered_faultignorant())
          {
            const size_t result_size = 
                        check_future_size(predicate_false_future.impl);
            if (result_size > 0)
              reduction_future.impl->set_result(
                predicate_false_future.impl->get_untyped_result(true,NULL,true),
                result_size, false/*own*/);
            else
              reduction_future.impl->set_result(NULL, 0, false/*own*/);
          }
          else
          {
            // Add references so they aren't garbage collected 
            reduction_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, this);
            predicate_false_future.impl->add_base_gc_ref(DEFERRED_TASK_REF, 
                                                         this);
            DeferredFutureSetArgs args(reduction_future.impl,
                                    predicate_false_future.impl, this);
            execution_condition = 
              runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY,
                                               Runtime::protect_event(wait_on));
          }
        }
        else
        {
          if (predicate_false_size > 0)
            reduction_future.impl->set_result(predicate_false_result,
                                  predicate_false_size, false/*own*/);
          else
            reduction_future.impl->set_result(NULL, 0, false/*own*/);
        }
      }
      // Then clean up this task execution
      complete_mapping();
      complete_execution(execution_condition);
      resolve_speculation();
      trigger_children_complete();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_EARLY_MAP_TASK_CALL);
      std::vector<unsigned> early_map_indexes;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &req = regions[idx];
        if (req.must_premap())
          early_map_indexes.push_back(idx);
      }
      if (!early_map_indexes.empty())
      {
        early_map_regions(map_applied_conditions, early_map_indexes);
        if (!acquired_instances.empty())
        {
          RtEvent precondition;
          if (!map_applied_conditions.empty())
          {
            precondition = Runtime::merge_events(map_applied_conditions);
            map_applied_conditions.clear();
          }
          precondition = release_nonempty_acquired_instances(precondition, 
                                                      acquired_instances);
          if (precondition.exists())
            map_applied_conditions.insert(precondition);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::early_map_regions(std::set<RtEvent> &applied_conditions,
                                      const std::vector<unsigned> &must_premap)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, EARLY_MAP_REGIONS_CALL);
      // This always happens on the owner node so we can just do the 
      // normal trace info creation here without needing to check
      // whether we have a remote trace info
      const TraceInfo trace_info(this);
      ApEvent init_precondition = compute_init_precondition(trace_info);;
      // A little bit of suckinesss here, it's unclear if we have
      // our version infos with the proper versioning information
      // so we might need to "page" it in now.  We'll overlap it as
      // much as possible, but it will still suck. The common case is that
      // we don't have anything to premap though so we shouldn't be
      // doing this all that often.
      std::set<RtEvent> version_ready_events;
      for (std::vector<unsigned>::const_iterator it = must_premap.begin();
            it != must_premap.end(); it++)
      {
        VersionInfo &version_info = get_version_info(*it); 
        if (version_info.has_version_info())
          continue;
        runtime->forest->perform_versioning_analysis(this, *it, regions[*it],
                                         version_info, version_ready_events);
      }
      Mapper::PremapTaskInput input;
      Mapper::PremapTaskOutput output;
      // Initialize this to not have a new target processor
      output.new_target_proc = Processor::NO_PROC;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      // Set up the inputs and outputs 
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      // At this point if we have any version ready events we need to wait
      if (!version_ready_events.empty())
      {
        RtEvent wait_on = Runtime::merge_events(version_ready_events);
        // This wait sucks but whatever for now
        wait_on.wait();
      }
      for (std::vector<unsigned>::const_iterator it = must_premap.begin();
            it != must_premap.end(); it++)
      {
        InstanceSet valid;    
        VersionInfo &version_info = get_version_info(*it);
        // Do the premapping
        if (request_valid_instances)
          runtime->forest->physical_premap_region(this, *it, regions[*it],
                                  version_info, valid, applied_conditions);
        // If we need visible instances, filter them as part of the conversion
        if (regions[*it].is_no_access())
          prepare_for_mapping(valid, input.valid_instances[*it]);
        else
          prepare_for_mapping(valid, visible_memories, 
                              input.valid_instances[*it]);
      }
      // Now invoke the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, &input, &output);
      // See if we need to update the new target processor
      if (output.new_target_proc.exists())
        this->target_proc = output.new_target_proc;
      // See if we have any profiling requests to handle
      if (!output.copy_prof_requests.empty())
      {
        filter_copy_request_kinds(mapper, 
            output.copy_prof_requests.requested_measurements,
            copy_profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
#endif
        profiling_reported = Runtime::create_rt_user_event();
      }
      // Now do the registration
      for (std::vector<unsigned>::const_iterator it = must_premap.begin();
            it != must_premap.end(); it++)
      {
        VersionInfo &version_info = get_version_info(*it);
        InstanceSet &chosen_instances = early_mapped_regions[*it];
        std::map<unsigned,std::vector<MappingInstance> >::const_iterator 
          finder = output.premapped_instances.find(*it);
        if (finder == output.premapped_instances.end())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'premap_task' invocation "
                        "on mapper %s. Mapper failed to map required premap "
                        "region requirement %d of task %s (ID %lld) launched "
                        "in parent task %s (ID %lld).", 
                        mapper->get_mapper_name(), *it, 
                        get_task_name(), get_unique_id(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
        RegionTreeID bad_tree = 0;
        std::vector<FieldID> missing_fields;
        std::vector<PhysicalManager*> unacquired;
        int composite_index = runtime->forest->physical_convert_mapping(
            this, regions[*it], finder->second, 
            chosen_instances, bad_tree, missing_fields,
            runtime->unsafe_mapper ? NULL : get_acquired_instances_ref(),
            unacquired, !runtime->unsafe_mapper);
        if (bad_tree > 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'premap_task' invocation "
                        "on mapper %s. Mapper provided an instance from "
                        "region tree %d for use in satisfying region "
                        "requirement %d of task %s (ID %lld) whose region "
                        "is from region tree %d.", mapper->get_mapper_name(),
                        bad_tree, *it,get_task_name(),get_unique_id(),
                        regions[*it].region.get_tree_id())
        if (!missing_fields.empty())
        {
          for (std::vector<FieldID>::const_iterator fit = 
                missing_fields.begin(); fit != missing_fields.end(); fit++)
          {
            const void *name; size_t name_size;
            if (!runtime->retrieve_semantic_information(
                regions[*it].region.get_field_space(), *fit,
                LEGION_NAME_SEMANTIC_TAG, name, name_size, true, false))
              name = "(no name)";
            log_run.error("Missing instance for field %s (FieldID: %d)",
                          static_cast<const char*>(name), *it);
          }
          REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                        "Invalid mapper output from 'premap_task' invocation "
                        "on mapper %s. Mapper failed to specify instances "
                        "for %zd fields of region requirement %d of task %s "
                        "(ID %lld) launched in parent task %s (ID %lld). "
                        "The missing fields are listed below.",
                        mapper->get_mapper_name(), missing_fields.size(),
                        *it, get_task_name(), get_unique_id(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id())
          
        }
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,unsigned> *acquired_instances = 
            get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator uit = 
                unacquired.begin(); uit != unacquired.end(); uit++)
          {
            if (acquired_instances->find(*uit) == acquired_instances->end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from 'premap_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            (*it), get_task_name(), get_unique_id())
          }
          // If we did successfully acquire them, still issue the warning
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                          "mapper %s failed to acquire instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'premap_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), *it, 
                          get_task_name(), get_unique_id());
        }
        if (composite_index >= 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'premap_task' invocation "
                        "on mapper %s. Mapper requested composite instance "
                        "creation on region requirement %d of task %s "
                        "(ID %lld) launched in parent task %s (ID %lld).",
                        mapper->get_mapper_name(), *it,
                        get_task_name(), get_unique_id(),
                        parent_ctx->get_task_name(),
                        parent_ctx->get_unique_id())
        if (runtime->legion_spy_enabled)
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx, 
                                                *it, regions[*it],
                                                chosen_instances);
        if (!runtime->unsafe_mapper)
        {
          std::vector<LogicalRegion> regions_to_check(1, 
                                        regions[*it].region);
          for (unsigned check_idx = 0; 
                check_idx < chosen_instances.size(); check_idx++)
          {
            PhysicalManager *manager = 
              chosen_instances[check_idx].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of "
                            "'premap_task' on mapper %s. Mapper specified an "
                            "instance region requirement %d of task %s "
                            "(ID %lld) that does not meet the logical region "
                            "requirement. Task was launched in task %s "
                            "(ID %lld).", mapper->get_mapper_name(), *it, 
                            get_task_name(), get_unique_id(), 
                            parent_ctx->get_task_name(), 
                            parent_ctx->get_unique_id())
            if (manager->is_collective_manager())
            {
              CollectiveManager *collective_manager = 
                manager->as_collective_manager();
              if (collective_manager->point_space->handle != 
                  launch_space->handle)
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT, 
                                "Invalid mapper output from invocation of "
                                "'premap_task' on mapper %s. Mapper selected "
                                "a collective instance with domain of index "
                                "space %d for region requirement %d that does "
                                "not match the launch index space %d for index "
                                "task %s (UID %lld).",mapper->get_mapper_name(),
                                collective_manager->point_space->handle.id,
                                *it, launch_space->handle.id, get_task_name(),
                                get_unique_id())
            }
          }
        }
        // TODO: Implement physical tracing for premapped regions
        if (is_memoizing())
          assert(false);
        // Passed all the error checking tests so register it
        // Always defer the users, the point tasks will do that
        // for themselves when they map their regions
        ApEvent effects_done = 
          runtime->forest->physical_perform_updates_and_registration(
                              regions[*it], version_info, 
                              this, *it, init_precondition, completion_event,
                              chosen_instances, 
                              PhysicalTraceInfo(trace_info, *it), 
                              applied_conditions
#ifdef DEBUG_LEGION
                              , get_logging_name(), unique_op_id
#endif
                              );
        if (effects_done.exists())
          complete_effects.insert(effects_done);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_DISTRIBUTE_CALL);
      if (is_origin_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
#ifdef DEBUG_LEGION
        assert(slices.empty());
#endif
        // We're never actually run
        return false;
      }
      else
      {
        if (!is_sliced() && target_proc.exists() && 
            (target_proc != current_proc))
        {
          // Make a slice copy and send it away
          SliceTask *clone = clone_as_slice_task(launch_space->handle, 
                                                 target_proc,
                                                 true/*needs slice*/,
                                                 stealable);
          runtime->send_task(clone);
          return false; // We have now been sent away
        }
        else
          return true; // Still local so we can be sliced
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::perform_mapping(MustEpochOp *owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_PERFORM_MAPPING_CALL);
      // This will only get called if we had slices that failed to origin map 
#ifdef DEBUG_LEGION
      assert(!slices.empty());
      // Should never get duplicate invocations here
      assert(args == NULL);
#endif
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        (*it)->trigger_mapping();
        it = slices.erase(it);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // Index space tasks are never stealable, they must first be
      // split into slices which can then be stolen.  Note that slicing
      // always happens after premapping so we know stealing is safe.
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // This should only ever be called if we had slices which failed to map
#ifdef DEBUG_LEGION
      assert(is_sliced());
      assert(!slices.empty());
#endif
      trigger_slices();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndexTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDEX_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_COMPLETE_CALL);
      // Trigger all the futures or set the reduction future result
      // and then trigger it
      if (redop != 0)
      {
        // Set the future if we actually ran the task or we speculated
        if ((speculation_state != RESOLVE_FALSE_STATE) || false_guard.exists())
        {
          // If we're doing a deterministic reduction this is the point
          // at which we can collapse all the futures down to a single
          // value since we know we have them all in the temporary futures
          if (deterministic_redop)
          {
            for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator
                  it = temporary_futures.begin();
                  it != temporary_futures.end(); it++)
              fold_reduction_future(it->second.first, it->second.second,
                                    false/*owner*/, true/*exclusive*/);
          }
          reduction_future.impl->set_result(reduction_state,
                                            reduction_state_size, 
                                            false/*owner*/);
        }
      }
      else
        Runtime::trigger_event(future_map_ready);
      if (must_epoch != NULL)
      {
        RtEvent precondition;
        if (!complete_preconditions.empty())
          precondition = Runtime::merge_events(complete_preconditions);
        must_epoch->notify_subop_complete(this, precondition);
      } 
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      completion_event);
#endif
      if (!complete_preconditions.empty())
        complete_operation(Runtime::merge_events(complete_preconditions));
      else
        complete_operation();
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_COMMIT_CALL);
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<IndexProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              IndexProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_task_report_profiling(this, &info);
              free(info.buffer);
            }
            const int count = to_perform.size() +
                outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::TaskProfilingInfo info;
          info.total_reports = 0;
          info.task_response = true;
          info.region_requirement_index = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_task_report_profiling(this, &info);    
          Runtime::trigger_event(profiling_reported);
        }
        commit_preconditions.insert(profiling_reported);
      }
      // If we have an origin-mapped slices then we need to check to see
      // if we're waiting on any profiling reports from them
      if (!origin_mapped_slices.empty())
      {
        for (std::set<SliceTask*>::const_iterator it = 
              origin_mapped_slices.begin(); it != 
              origin_mapped_slices.end(); it++)
          (*it)->find_commit_preconditions(commit_preconditions);
      }
      if (must_epoch != NULL)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        must_epoch->notify_subop_commit(this, commit_precondition);
        commit_operation(true/*deactivate*/, commit_precondition);
      }
      else
      {
        // Mark that this operation is now committed
        if (!commit_preconditions.empty())
          commit_operation(true/*deactivate*/, 
              Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true/*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_inlining(VariantImpl *variant,
                                  const std::deque<InstanceSet> &parent_regions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_PERFORM_INLINING_CALL);
      total_points = launch_space->get_volume();
      if (redop == 0)
        enumerate_futures(index_domain);
      SliceTask *slice = clone_as_slice_task(launch_space->handle,
                current_proc, false/*recurse*/, false/*stealable*/);
      slice->enumerate_points();
      slice->perform_inlining(variant, parent_regions);
      // Record that we had our speculation resolved too
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    VersionInfo& IndexTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const VersionInfo& IndexTask::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,VersionInfo>::const_iterator finder = 
        version_infos.find(idx);
#ifdef DEBUG_LEGION
      assert(finder != version_infos.end());
#endif
      return finder->second;
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(IndexSpace is, Processor p,
                                              bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_CLONE_AS_SLICE_CALL);
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx, false/*track*/, NULL/*deps*/,
          Predicate::TRUE_PRED, this->task_id, provenance);
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this;
      result->remote_owner_uid = parent_ctx->get_unique_id();
      result->tpl = tpl;
      result->memo_state = memo_state;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_index_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
                                                result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_future(const DomainPoint &point,
                            const void *result, size_t result_size, bool owner, 
                            FutureFunctor *functor, Processor future_proc)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_HANDLE_FUTURE);
#ifdef DEBUG_LEGION
      assert(reduction_op != NULL);
      assert(functor == NULL);
#endif
      // If we're doing a deterministic reduction then we need to 
      // buffer up these future values until we get all of them so
      // that we can fold them in a deterministic way
      if (deterministic_redop)
      {
        // Store it in our temporary futures
        if (owner)
        {
          // Hold the lock to protect the data structure
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          temporary_futures[point] = 
            std::pair<void*,size_t>(const_cast<void*>(result),result_size);
        }
        else
        {
          void *copy = legion_malloc(FUTURE_RESULT_ALLOC, result_size);
          memcpy(copy,result,result_size);
          // Hold the lock to protect the data structure
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          temporary_futures[point] = 
            std::pair<void*,size_t>(copy,result_size);
        }
      }
      else
        fold_reduction_future(result, result_size, owner, false/*exclusive*/);
    }

    //--------------------------------------------------------------------------
    void IndexTask::pack_profiling_requests(Serializer &rez,
                                            std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request(
        runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
        &response, sizeof(response));
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            copy_profiling_requests.begin(); it != 
            copy_profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          IndexProfilingInfo &info = profiling_info.back();
          info.task_response = task_prof->task; 
          info.region_requirement_index = task_prof->src;
          info.fill_response = task_prof->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::TaskProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.task_response = task_prof->task; 
      info.region_requirement_index = task_prof->src;
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = task_prof->fill;
      mapper->invoke_task_report_profiling(this, &info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    } 

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void IndexTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      // Check to see if we already have it
      std::map<DomainPoint,RtEvent>::const_iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
        return finder->second;
      // Otherwise make a temporary one and record it for now
      const RtUserEvent pending_event = Runtime::create_rt_user_event();
      intra_space_dependences[point] = pending_event;
      pending_intra_space_dependences[point] = pending_event;
      return pending_event;
    }
    
    //--------------------------------------------------------------------------
    void IndexTask::record_intra_space_dependence(const DomainPoint &point,
                                                  RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<DomainPoint,RtEvent>::iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
      {
#ifdef DEBUG_LEGION
        assert(finder->second != point_mapped);
#endif
        std::map<DomainPoint,RtUserEvent>::iterator pending_finder = 
          pending_intra_space_dependences.find(point);
#ifdef DEBUG_LEGION
        assert(pending_finder != pending_intra_space_dependences.end());
#endif
        Runtime::trigger_event(pending_finder->second, point_mapped);
        pending_intra_space_dependences.erase(pending_finder);
        finder->second = point_mapped;
      }
      else
        intra_space_dependences[point] = point_mapped;
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_origin_mapped_slice(SliceTask *local_slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      origin_mapped_slices.insert(local_slice);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_mapped(unsigned points,
                              RtEvent applied_condition, ApEvent slice_complete)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_MAPPED_CALL);
      bool need_trigger = false;
      bool trigger_children_completed = false;
      bool trigger_children_commit = false;
      {
        AutoLock o_lock(op_lock);
        mapped_points += points;
        if (applied_condition.exists())
          map_applied_conditions.insert(applied_condition);
        if (slice_complete.exists())
          complete_effects.insert(slice_complete);
        // Already know that mapped points is the same as total points
        if (mapped_points == total_points)
        {
          need_trigger = true;
          if ((complete_points == total_points) &&
              !children_complete_invoked)
          {
            trigger_children_completed = true;
            children_complete_invoked = true;
          }
          if ((committed_points == total_points) &&
              !children_commit_invoked)
          {
            trigger_children_commit = true;
            children_commit_invoked = true;
          }
        }
      }
      if (need_trigger)
      {
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          RtEvent map_condition = Runtime::merge_events(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
        const ApEvent all_slices_complete = 
          Runtime::merge_events(NULL, complete_effects);
        if (!request_early_complete(all_slices_complete))
          complete_execution(Runtime::protect_event(all_slices_complete));
        else
          complete_execution();
      }
      if (trigger_children_completed)
        trigger_children_complete();
      if (trigger_children_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(unsigned points, 
                                          RtEvent slice_complete)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMPLETE_CALL);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (slice_complete.exists())
          complete_preconditions.insert(slice_complete);
        complete_points += points;
#ifdef DEBUG_LEGION
        assert(complete_points <= total_points);
#endif
        if ((complete_points == total_points) && !children_complete_invoked)
        {
          need_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_commit(unsigned points, 
                                        RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMMIT_CALL);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        committed_points += points;
#ifdef DEBUG_LEGION
        assert(committed_points <= total_points);
#endif
        if ((committed_points == total_points) && !children_commit_invoked)
        {
          need_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_committed();
    } 

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_mapped(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent applied_condition;
      derez.deserialize(applied_condition);
      ApEvent restrict_postcondition;
      derez.deserialize(restrict_postcondition);
#ifdef DEBUG_LEGION
      if (!is_origin_mapped())
      {
        std::map<DomainPoint,std::vector<LogicalRegion> > local_requirements;
        for (unsigned idx = 0; idx < points; idx++)
        {
          DomainPoint point;
          derez.deserialize(point);
          std::vector<LogicalRegion> &reqs = local_requirements[point];
          reqs.resize(regions.size());
          for (unsigned idx2 = 0; idx2 < regions.size(); idx2++)
            derez.deserialize(reqs[idx2]);
        }
        check_point_requirements(local_requirements);
      }
#endif
      return_slice_mapped(points, applied_condition, restrict_postcondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent complete_precondition;
      derez.deserialize(complete_precondition);
      const RtEvent resources_returned =
        ResourceTracker::unpack_resources_return(derez, parent_ctx);
      if (redop > 0)
      {
        if (deterministic_redop)
        {
#ifdef DEBUG_LEGION
          assert(reduction_op != NULL);
#endif
          // Unpack these futures and save them so we can do a
          // deterministic reduction fold operation later
          for (unsigned idx = 0; idx < points; idx++)
          {
            DomainPoint p;
            derez.deserialize(p);
            size_t size;
            derez.deserialize(size);
            const void *ptr = derez.get_current_pointer();
            handle_future(p, ptr, size, false/*owner*/, 
                          NULL/*functor*/, Processor::NO_PROC);
            derez.advance_pointer(size);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(reduction_op != NULL);
#endif
          size_t reduc_size;
          derez.deserialize(reduc_size);
          const void *reduc_ptr = derez.get_current_pointer();
          fold_reduction_future(reduc_ptr, reduc_size,
                                false /*owner*/, false/*exclusive*/);
          // Advance the pointer on the deserializer
          derez.advance_pointer(reduc_size);
        }
      }
      if (resources_returned.exists())
      {
        if (complete_precondition.exists())
          return_slice_complete(points,
              Runtime::merge_events(complete_precondition, resources_returned));
        else
          return_slice_complete(points, resources_returned);
      }
      else
        return_slice_complete(points, complete_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent commit_precondition;
      derez.deserialize(commit_precondition);
      return_slice_commit(points, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replaying());
      assert(current_proc.exists());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
      // Count how many total points we need for this index space task
      total_points = index_domain.get_volume();
      if (redop == 0)
        enumerate_futures(index_domain);
      // Mark that this is origin mapped effectively in case we
      // have any remote tasks, do this before we clone it
      map_origin = true;
      SliceTask *new_slice = this->clone_as_slice_task(internal_space,
                                                       current_proc,
                                                       false, false);
      new_slice->enumerate_points();
      // We need to make one slice per point here in case we need to move
      // points to remote nodes. The way we do slicing right now prevents
      // us from knowing which point tasks are going remote until later in
      // the replay so we have to be pessimistic here
      new_slice->expand_replay_slices(slices);
      // Then do the replay on all the slices
      for (std::list<SliceTask*>::const_iterator it = 
            slices.begin(); it != slices.end(); it++)
        (*it)->trigger_replay();
      resolve_speculation();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_mapped(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_find_intra_dependence(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *task;
      derez.deserialize(task);
      DomainPoint point;
      derez.deserialize(point);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      const RtEvent result = task->find_intra_space_dependence(point);
      Runtime::trigger_event(to_trigger, result);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_record_intra_dependence(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *task;
      derez.deserialize(task);
      DomainPoint point;
      derez.deserialize(point);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      task->record_intra_space_dependence(point, mapped_event);
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void IndexTask::check_point_requirements(
            const std::map<DomainPoint,std::vector<LogicalRegion> > &point_reqs)
    //--------------------------------------------------------------------------
    {
      std::set<std::pair<unsigned,unsigned> > local_interfering = 
        interfering_requirements;
      // Handle any region requirements that interfere with itself
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &req = regions[idx];
        if (!IS_WRITE(req))
          continue;
        // If the projection functions are invertible then we don't have to 
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          ProjectionFunction *func = 
            runtime->find_projection_function(req.projection);   
          if (func->is_invertible)
            continue;
        }
        local_interfering.insert(std::pair<unsigned,unsigned>(idx,idx));
      }
      // Nothing to do if there are no interfering requirements
      if (local_interfering.empty())
        return;
      // Make sure that all the slices coming back here are serialized
      AutoLock o_lock(op_lock);
      for (std::map<DomainPoint,std::vector<LogicalRegion> >::const_iterator 
            pit = point_reqs.begin(); pit != point_reqs.end(); pit++)
      { 
        const std::vector<LogicalRegion> &point_reqs = pit->second;
        for (std::map<DomainPoint,std::vector<LogicalRegion> >::const_iterator
              oit = point_requirements.begin(); 
              oit != point_requirements.end(); oit++)
        {
          const std::vector<LogicalRegion> &other_reqs = oit->second;
          const bool same_point = (pit->first == oit->first);
          // Now check for interference with any other points
          for (std::set<std::pair<unsigned,unsigned> >::const_iterator it =
                local_interfering.begin(); it !=
                local_interfering.end(); it++)
          {
            // Skip same region requireemnt for same point
            if (same_point && (it->first == it->second))
              continue;
            // If either one are the NO_REGION then there is no interference
            if (!point_reqs[it->first].exists() || 
                !other_reqs[it->second].exists())
              continue;
            if (!runtime->forest->are_disjoint(
                  point_reqs[it->first].get_index_space(), 
                  other_reqs[it->second].get_index_space()))
            {
              switch (pit->first.get_dim())
              {
                case 1:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point %lld and region "
                              "requirement %d of point %lld of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, pit->first[0], it->second,
                              oit->first[0], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#if LEGION_MAX_DIM >= 2
                case 2:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point (%lld,%lld) and "
                              "region requirement %d of point (%lld,%lld) of "
                              "%s (UID %lld) in parent task %s (UID %lld) are "
                              "interfering.", it->first, pit->first[0],
                              pit->first[1], it->second, oit->first[0],
                              oit->first[1], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 3
                case 3:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point (%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld) of %s (UID %lld) in parent "
                              "task %s (UID %lld) are interfering.", it->first,
                              pit->first[0], pit->first[1], pit->first[2],
                              it->second, oit->first[0], oit->first[1],
                              oit->first[2], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 4
                case 4:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld) of %s (UID %lld) in parent"
                              " task %s (UID %lld) are interfering.", it->first,
                              pit->first[0], pit->first[1], pit->first[2],
                              pit->first[3], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 5
                case 5:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld) of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 6
                case 6:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld) of %s " 
                              "(UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3], 
                              oit->first[4], oit->first[5],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 7
                case 7:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld) of %s "
                              "(UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], it->second, 
                              oit->first[0], oit->first[1], oit->first[2], 
                              oit->first[3], oit->first[4], oit->first[5],
                              oit->first[6], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 8
                case 8:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], pit->first[7],
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4], 
                              oit->first[5], oit->first[6], oit->first[7],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 9
                case 9:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], pit->first[7],
                              pit->first[8], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3], 
                              oit->first[4], oit->first[5], oit->first[6], 
                              oit->first[7], oit->first[8],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
                default:
                  assert(false);
              }
            }
          }
        }
        // Add it to the set of point requirements
        point_requirements.insert(*pit);
      }
    }
#endif

    /////////////////////////////////////////////////////////////
    // Slice Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(Runtime *rt)
      : MultiTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(const SliceTask &rhs)
      : MultiTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask& SliceTask::operator=(const SliceTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void SliceTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_ACTIVATE_CALL);
      activate_multi();
      // Slice tasks never have to resolve speculation
      resolve_speculation();
      num_unmapped_points = 0;
      num_uncomplete_points = 0;
      num_uncommitted_points = 0;
      index_owner = NULL;
      remote_owner_uid = 0;
      remote_unique_id = get_unique_id();
      origin_mapped = false;
      origin_mapped_complete = RtUserEvent::NO_RT_USER_EVENT;
      // Slice tasks always already have their options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DEACTIVATE_CALL);
      deactivate_multi();
      // Deactivate all our points 
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        // Check to see if we are origin mapped or not which 
        // determines whether we should commit this operation or
        // just deactivate it like normal
        if (is_origin_mapped() && !is_remote())
          (*it)->deactivate();
        else
          (*it)->commit_operation(true/*deactivate*/);
      }
      points.clear(); 
#ifdef DEBUG_LEGION
      assert(local_regions.empty());
      assert(local_fields.empty());
      assert(!origin_mapped_complete.exists() ||
              origin_mapped_complete.has_triggered());
#endif
      map_applied_conditions.clear();
      point_completions.clear();
      complete_preconditions.clear();
      commit_preconditions.clear();
      created_regions.clear();
      created_fields.clear();
      created_field_spaces.clear();
      created_index_spaces.clear();
      created_index_partitions.clear();
      runtime->free_slice_task(this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::early_map_task(void)
    //--------------------------------------------------------------------------
    {
      // Slices are already done with early mapping 
    }

    //--------------------------------------------------------------------------
    void SliceTask::check_target_processors(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      if (points.size() == 1)
        return;
      const AddressSpaceID target_space = 
        runtime->find_address_space(points[0]->target_proc);
      for (unsigned idx = 1; idx < points.size(); idx++)
      {
        if (target_space != 
            runtime->find_address_space(points[idx]->target_proc))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output: two different points in one "
                      "slice of %s (UID %lld) mapped to processors in two"
                      "different address spaces (%d and %d) which is illegal.",
                      get_task_name(), get_unique_id(), target_space,
                      runtime->find_address_space(points[idx]->target_proc))
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::update_target_processor(void)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        return;
#ifdef DEBUG_LEGION
      check_target_processors();
#endif
      this->target_proc = points[0]->target_proc;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DISTRIBUTE_CALL);
      update_target_processor();
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        // The runtime will deactivate this task
        // after it has been sent
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::perform_mapping(MustEpochOp *epoch_owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_PERFORM_MAPPING_CALL);
#ifdef DEBUG_LEGION
      // Should never get duplicate invocations here
      assert(args == NULL);
#endif
      // Check to see if we already enumerated all the points, if
      // not then do so now
      if (points.empty())
        enumerate_points();
      // Once we start mapping then we are no longer stealable
      stealable = false;
      std::set<RtEvent> mapped_events;
      for (std::vector<PointTask*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
      {
        // Now that we support collective instance creation, we need to 
        // enable all the point tasks to be mapping in parallel with
        // each other in case they need to synchronize to create 
        // collective instances
        mapped_events.insert((*it)->defer_perform_mapping(RtEvent::NO_RT_EVENT,
                                    epoch_owner, args, 0/*invocation count*/));
      }
      return Runtime::merge_events(mapped_events);
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_LAUNCH_CALL);
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      // Launch all of our child points
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->launch_task(inline_task);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
    }

    //--------------------------------------------------------------------------
    void SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_MAP_AND_LAUNCH_CALL);
      // First enumerate all of our points if we haven't already done so
      if (points.empty())
        enumerate_points();
      // Mark that this task is no longer stealable.  Once we start
      // executing things onto a specific processor slices cannot move.
      stealable = false;
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      const size_t num_points = points.size();
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = points[idx];
        // Now that we support collective instance creation, we need to 
        // enable all the point tasks to be mapping in parallel with
        // each other in case they need to synchronize to create 
        // collective instances
        const RtEvent map_event = point->defer_perform_mapping(
            RtEvent::NO_RT_EVENT, NULL/*must epoch*/, 
            NULL/*defer args*/, 0/*invocation count*/);
        if (map_event.exists() && !map_event.has_triggered())
          point->defer_launch_task(map_event);
        else
          point->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind SliceTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return SLICE_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_PACK_TASK_CALL);
      // Check to see if we are stealable or not yet fully sliced,
      // if both are false and we're not remote, then we can send the state
      // now or check to see if we are remotely mapped
      RezCheck z(rez);
      // Preamble used in TaskOp::unpack
      rez.serialize(points.size());
      pack_multi_task(rez, target);
      rez.serialize(index_owner);
      rez.serialize(remote_unique_id);
      rez.serialize(origin_mapped);
      rez.serialize(remote_owner_uid);
      rez.serialize(internal_space);
      if (redop == 0)
      {
#ifdef DEBUG_LEGION
        assert(future_map.impl != NULL);
#endif
        rez.serialize(future_map.impl->did);
        rez.serialize(future_map.impl->get_ready_event());
      }
      if (predicate_false_future.impl != NULL)
        rez.serialize(predicate_false_future.impl->did);
      else
        rez.serialize<DistributedID>(0);
      rez.serialize(predicate_false_size);
      if (predicate_false_size > 0)
        rez.serialize(predicate_false_result, predicate_false_size);
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->pack_task(rez, target);
      // If we don't have any points, we have to pack up the argument map
      // and any trace info that we need for doing remote tracing
      if (points.empty())
      {
        if (point_arguments.impl != NULL)
        {
          rez.serialize(point_arguments.impl->did);
          rez.serialize(point_arguments.impl->get_ready_event());
        }
        else
          rez.serialize<DistributedID>(0);
        rez.serialize<size_t>(point_futures.size());
        for (unsigned idx = 0; idx < point_futures.size(); idx++)
        {
          FutureMapImpl *impl = point_futures[idx].impl;
          rez.serialize(impl->did);
          rez.serialize(impl->get_ready_event());
        }
      }
      if (is_origin_mapped() && !is_remote())
      {
#ifdef DEBUG_LEGION
        assert(!origin_mapped_complete.exists());
#endif
        // Similarly for slices being removed remotely but are
        // origin mapped we may need to receive profiling feedback
        // to this node so also hold onto these slices until the
        // index space is done
        index_owner->record_origin_mapped_slice(this);
        return false;
      }
      // Always return true for slice tasks since they should
      // always be deactivated after they are sent somewhere else
      return true;
    }
    
    //--------------------------------------------------------------------------
    bool SliceTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez, ready_events);
      set_current_proc(current);
      derez.deserialize(index_owner);
      derez.deserialize(remote_unique_id); 
      derez.deserialize(origin_mapped);
      derez.deserialize(remote_owner_uid);
      derez.deserialize(internal_space);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_slice(remote_unique_id, get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(remote_unique_id,
            get_unique_op_id());
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      // Check to see if we ended up back on the original node
      // We have to do this before unpacking the points
      if (is_remote())
      {
        RtEvent ctx_ready;
        parent_ctx = runtime->find_context(remote_owner_uid, false, &ctx_ready);
        if (ctx_ready.exists() && !ctx_ready.has_triggered())
        {
          // Need to wait if the profiler is going to want to check this
          if (runtime->profiler != NULL)
            ctx_ready.wait();
          else
            ready_events.insert(ctx_ready);
        }
      }
      else
        parent_ctx = index_owner->parent_ctx;
      if (redop == 0)
      {
        DistributedID future_map_did;
        derez.deserialize(future_map_did);
        RtEvent ready_event;
        derez.deserialize(ready_event);
        WrapperReferenceMutator mutator(ready_events);
        future_map = FutureMap(
            runtime->find_or_create_future_map(future_map_did, parent_ctx, 
                                               ready_event, &mutator)); 
      }
      // Unpack the predicate false infos
      DistributedID pred_false_did;
      derez.deserialize(pred_false_did);
      if (pred_false_did != 0)
      {
        WrapperReferenceMutator mutator(ready_events);
        FutureImpl *impl = runtime->find_or_create_future(pred_false_did,
                                              remote_owner_uid, &mutator);
        impl->add_base_gc_ref(FUTURE_HANDLE_REF, &mutator);
        predicate_false_future = Future(impl, false/*need reference*/);
      }
      derez.deserialize(predicate_false_size);
      if (predicate_false_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(predicate_false_result == NULL);
#endif
        predicate_false_result = malloc(predicate_false_size);
        derez.deserialize(predicate_false_result, predicate_false_size);
      }
      // Unpack the provenance before unpacking any point tasks so
      // that they can pick it up as well
      provenance = Provenance::deserialize(derez);
      if (provenance != NULL)
        provenance->add_reference();
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = runtime->get_available_point_task(); 
        point->slice_owner = this;
        point->unpack_task(derez, current, ready_events);
        point->parent_ctx = parent_ctx;
        points.push_back(point);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_slice_point(get_unique_id(), 
                                     point->get_unique_id(),
                                     point->index_point);
      }
      if (num_points == 0)
      {
        DistributedID future_map_did;
        derez.deserialize(future_map_did);
        if (future_map_did > 0)
        {
          RtEvent ready_event;
          derez.deserialize(ready_event);
          WrapperReferenceMutator mutator(ready_events);
          FutureMapImpl *impl = runtime->find_or_create_future_map(
                  future_map_did, parent_ctx, ready_event, &mutator);
          impl->add_base_gc_ref(FUTURE_HANDLE_REF, &mutator);
          point_arguments = FutureMap(impl, false/*need reference*/);
        }
        size_t num_point_futures;
        derez.deserialize(num_point_futures);
        if (num_point_futures > 0)
        {
          RtEvent ready_event;
          point_futures.resize(num_point_futures);
          WrapperReferenceMutator mutator(ready_events);
          for (unsigned idx = 0; idx < num_point_futures; idx++)
          {
            derez.deserialize(future_map_did);
            derez.deserialize(ready_event);
            FutureMapImpl *impl = runtime->find_or_create_future_map(
                    future_map_did, parent_ctx, ready_event, &mutator);
            impl->add_base_gc_ref(FUTURE_HANDLE_REF, &mutator);
            point_futures[idx] = FutureMap(impl, false/*need reference*/);
          }
        }
      }
      else // Set the first mapping to false since we know things are mapped
        first_mapping = false;
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      // Return true to add this to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_inlining(VariantImpl *variant,
                                const std::deque<InstanceSet> &parent_instances)
    //--------------------------------------------------------------------------
    {
      // Need to handle inter-space dependences correctly here
      std::map<PointTask*,unsigned> remaining;
      std::map<RtEvent,std::vector<PointTask*> > event_deps;
      for (std::vector<PointTask*>::const_iterator it =
            points.begin(); it != points.end(); it++)
        if (!(*it)->has_remaining_inlining_dependences(remaining, event_deps))
          (*it)->perform_inlining(variant, parent_instances);
      while (!remaining.empty())
      {
#ifdef DEBUG_LEGION
        bool found = false; // should find at least one each iteration
#endif
        for (std::map<PointTask*,unsigned>::iterator it =
              remaining.begin(); it != remaining.end(); /*nothing*/)
        {
          if (it->second == 0)
          {
            const RtEvent mapped = it->first->get_mapped_event();
            it->first->perform_inlining(variant, parent_instances);
#ifdef DEBUG_LEGION
            found = true;
            assert(mapped.has_triggered());
#endif
            std::map<RtEvent,std::vector<PointTask*> >::const_iterator finder =
              event_deps.find(mapped);
            if (finder != event_deps.end())
            {
              for (unsigned idx = 0; idx < finder->second.size(); idx++)
              {
                std::map<PointTask*,unsigned>::iterator point_finder =
                  remaining.find(finder->second[idx]);
#ifdef DEBUG_LEGION
                assert(point_finder != remaining.end());
                assert(point_finder->second > 0);
#endif
                point_finder->second--;
              }
              event_deps.erase(finder);
            }
            std::map<PointTask*,unsigned>::iterator to_delete = it++;
            remaining.erase(to_delete);
          }
          else
            it++;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      // Record that we've mapped and executed this slice
      trigger_slice_mapped();
    }

    //--------------------------------------------------------------------------
    SliceTask* SliceTask::clone_as_slice_task(IndexSpace is, Processor p,
                                              bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_SLICE_CALL);
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx,  false/*track*/, NULL/*deps*/,
          Predicate::TRUE_PRED, this->task_id, provenance);
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this->index_owner;
      result->remote_owner_uid = this->remote_owner_uid;
      result->tpl = tpl;
      result->memo_state = memo_state;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
            result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(const DomainPoint &point,
                             const void *result, size_t result_size, bool owner, 
                             FutureFunctor *functor, Processor future_proc)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_HANDLE_FUTURE_CALL);
      if (redop > 0)
      {
        if (is_remote())
        {
#ifdef DEBUG_LEGION
          assert(functor == NULL);
#endif
          // Store the future result in our temporary futures unless we're 
          // doing a non-deterministic reduction in which case we can eagerly
          // fold this now into our reduction buffer
          if (deterministic_redop)
          {
            // Store it in our temporary futures
            if (owner)
            {
              // Hold the lock to protect the data structure
              AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
              assert(temporary_futures.find(point) == temporary_futures.end());
#endif
              temporary_futures[point] = 
                std::pair<void*,size_t>(const_cast<void*>(result),result_size);
            }
            else
            {
              void *copy = legion_malloc(FUTURE_RESULT_ALLOC, result_size);
              memcpy(copy,result,result_size);
              // Hold the lock to protect the data structure
              AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
              assert(temporary_futures.find(point) == temporary_futures.end());
#endif
              temporary_futures[point] = 
                std::pair<void*,size_t>(copy,result_size);
            }
          }
          else
            fold_reduction_future(result, result_size,owner,false/*exclusive*/);
        }
        else
          index_owner->handle_future(point, result, result_size,
                                     owner, functor, future_proc);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(future_handles != NULL);
#endif
        std::map<DomainPoint,DistributedID>::const_iterator finder = 
          future_handles->handles.find(point);
#ifdef DEBUG_LEGION
        assert(finder != future_handles->handles.end());
#endif
        LocalReferenceMutator mutator;
        FutureImpl *impl = runtime->find_or_create_future(finder->second, 
                                parent_ctx->get_context_uid(), &mutator);
        if (functor != NULL)
          impl->set_result(functor, owner, future_proc);
        else
          impl->set_result(result, result_size, owner);
        const RtEvent applied = mutator.get_done_event();
        if (applied.exists() && !applied.has_triggered())
        {
          AutoLock o_lock(op_lock);
          complete_preconditions.insert(applied);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
#endif
      if (points.empty())
        enumerate_points();
      must_epoch->register_slice_task(this);
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        PointTask *point = points[idx];
        must_epoch->register_single_task(point, must_epoch_index);
      }
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_POINT_CALL);
      PointTask *result = runtime->get_available_point_task();
      result->initialize_base_task(parent_ctx, false/*track*/, NULL/*deps*/,
          Predicate::TRUE_PRED, this->task_id, provenance);
      result->clone_task_op_from(this, this->target_proc, 
                                 false/*stealable*/, true/*duplicate*/);
      result->is_index_space = true;
      result->must_epoch_task = this->must_epoch_task;
      result->index_domain = this->index_domain;
      result->tpl = tpl;
      result->memo_state = memo_state;
      // Now figure out our local point information
      result->initialize_point(this, point, point_arguments, point_futures);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_point(get_unique_id(), 
                                   result->get_unique_id(),
                                   result->index_point);
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::enumerate_points(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_ENUMERATE_POINTS_CALL);
      Domain internal_domain;
      runtime->forest->find_launch_space_domain(internal_space,internal_domain);
      size_t num_points = internal_domain.get_volume();
#ifdef DEBUG_LEGION
      assert(num_points > 0);
#endif
      unsigned point_idx = 0;
      points.resize(num_points);
      // Enumerate all the points in our slice and make point tasks
      for (Domain::DomainPointIterator itr(internal_domain); 
            itr; itr++, point_idx++)
        points[point_idx] = clone_as_point_task(itr.p);
      // Compute any projection region requirements
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(regions[idx].projection);
        function->project_points(regions[idx], idx, runtime, 
                                 points, launch_space);
      }
      // Update the no access regions
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->complete_point_projection();
      // Mark how many points we have
      num_unmapped_points = points.size();
      num_uncomplete_points = points.size();
      num_uncommitted_points = points.size();
    } 

    //--------------------------------------------------------------------------
    const void* SliceTask::get_predicate_false_result(size_t &result_size)
    //--------------------------------------------------------------------------
    {
      if (predicate_false_future.impl != NULL)
      {
        // Wait for the future to be ready
        ApEvent wait_on = predicate_false_future.impl->get_ready_event();
        wait_on.wait_faultignorant(); 
        result_size = predicate_false_future.impl->get_untyped_size(true);
        return predicate_false_future.impl->get_untyped_result(true,NULL,true);
      }
      else
      {
        result_size = predicate_false_size;
        return predicate_false_result;
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // This is mostly the same as the base TaskOp::trigger_complete
      // except we also check to see if we have an origin_mapped_complete
      // event that we need to trigger once we're done
      RtUserEvent to_trigger;
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!complete_received);
        assert(!commit_received);
#endif
        complete_received = true;
        // If all our children are also complete then we are done
        task_complete = children_complete;
        if (origin_mapped_complete.exists())
          to_trigger = origin_mapped_complete;
      }
      if (task_complete)
        trigger_task_complete();
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      trigger_slice_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      trigger_slice_commit();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_reference_mutation_effect(RtEvent event)
    //--------------------------------------------------------------------------
    {
      map_applied_conditions.insert(event);
    }

    //--------------------------------------------------------------------------
    void SliceTask::return_privileges(TaskContext *point_context,
                                      std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // If we're remote, pass our privileges back to ourself
      // otherwise pass them directly back to the index owner
      if (is_remote())
        point_context->return_resources(this, context_index, preconditions);
      else
        point_context->return_resources(parent_ctx,context_index,preconditions);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_mapped(RtEvent child_mapped,
     ApEvent child_complete,std::map<PhysicalManager*,unsigned> &child_acquired)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (child_mapped.exists())
          map_applied_conditions.insert(child_mapped);
        if (child_complete.exists())
          point_completions.insert(child_complete);
        if (!child_acquired.empty())
        {
          if (!acquired_instances.empty())
          {
            while (!child_acquired.empty())
            {
              std::map<PhysicalManager*,unsigned>::iterator next =
                child_acquired.begin();
              std::map<PhysicalManager*,unsigned>::iterator finder = 
                acquired_instances.find(next->first);
              if (finder == acquired_instances.end())
                acquired_instances.insert(*next);
              else
                finder->second += next->second;
              child_acquired.erase(next);
            }
          }
          else
            acquired_instances.swap(child_acquired);
        }
#ifdef DEBUG_LEGION
        assert(num_unmapped_points > 0);
#endif
        num_unmapped_points--;
        if (num_unmapped_points == 0)
          needs_trigger = true;
      }
      if (needs_trigger)
        trigger_slice_mapped();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_complete(RtEvent child_complete)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (child_complete.exists())
          complete_preconditions.insert(child_complete);
#ifdef DEBUG_LEGION
        assert(num_uncomplete_points > 0);
#endif
        num_uncomplete_points--;
        if ((num_uncomplete_points == 0) && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_committed(RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(num_uncommitted_points > 0);
#endif
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        num_uncommitted_points--;
        if ((num_uncommitted_points == 0) && !children_commit_invoked)
        {
          needs_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_mapped(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_MAPPED_CALL);
      RtEvent applied_condition;
      if (!map_applied_conditions.empty())
        applied_condition = Runtime::merge_events(map_applied_conditions);
      ApEvent all_points_complete;
      if (!point_completions.empty())
        all_points_complete = Runtime::merge_events(NULL, point_completions);
      if (!acquired_instances.empty())
        applied_condition = release_nonempty_acquired_instances(
                          applied_condition, acquired_instances);
      if (is_remote())
      {
        // Only need to send something back if this wasn't origin mapped 
        if (!is_origin_mapped())
        {
          Serializer rez;
          pack_remote_mapped(rez, applied_condition, all_points_complete);
          runtime->send_slice_remote_mapped(orig_proc, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        // In debug mode, get all our point region requirements and
        // then pass them back to the index space task
        std::map<DomainPoint,std::vector<LogicalRegion> > local_requirements;
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          std::vector<LogicalRegion> &reqs = 
            local_requirements[(*it)->index_point];
          reqs.resize(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            reqs[idx] = (*it)->regions[idx].region;
        }
        index_owner->check_point_requirements(local_requirements);
#endif
        index_owner->return_slice_mapped(points.size(), applied_condition, 
                                         all_points_complete);
      }
      complete_mapping(applied_condition); 
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_COMPLETE_CALL);
      RtEvent complete_precondition;
      if (!complete_preconditions.empty())
        complete_precondition = Runtime::merge_events(complete_preconditions);
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      if (is_remote())
      {
        // Send back the message saying that this slice is complete
        Serializer rez;
        pack_remote_complete(rez, complete_precondition);
        runtime->send_slice_remote_complete(orig_proc, rez);
      }
      else
      {
        index_owner->return_slice_complete(points.size(),complete_precondition);
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_COMMIT_CALL);
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez, commit_precondition);
        runtime->send_slice_remote_commit(orig_proc, rez);
      }
      else
      {
        // created and deleted privilege information already passed back
        // futures already sent back
        index_owner->return_slice_commit(points.size(), commit_precondition);
      }
      commit_operation(true/*deactivate*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_mapped(Serializer &rez, 
                         RtEvent applied_condition, ApEvent all_points_complete)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(applied_condition);
      rez.serialize(all_points_complete);
#ifdef DEBUG_LEGION
      if (!is_origin_mapped())
      {
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          rez.serialize((*it)->index_point);
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize((*it)->regions[idx].region);
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_complete(Serializer &rez, 
                                         RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      // Send back any created state that our point tasks made
      AddressSpaceID target = runtime->find_address_space(orig_proc);
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
        (*it)->send_back_created_state(target);
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize<size_t>(points.size());
      rez.serialize(applied_condition);
      // Serialize the privilege state
      pack_resources_return(rez, context_index); 
      // Now pack up the future results
      if (redop > 0)
      {
        if (deterministic_redop)
        {
          // Same as above but without the extra rez check
#ifdef DEBUG_LEGION
          assert(temporary_futures.size() == points.size());
#endif
          for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator 
                it = temporary_futures.begin(); 
                it != temporary_futures.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second.second);
            rez.serialize(it->second.first,it->second.second);
          }
        }
        else
        {
          rez.serialize<size_t>(reduction_state_size);
          rez.serialize(reduction_state,reduction_state_size);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_commit(Serializer &rez, 
                                       RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(applied_condition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_slice_return(Runtime *rt, 
                                                   Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      Runtime::trigger_event(ready_event);
    }

    //--------------------------------------------------------------------------
    void SliceTask::receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regs,
              std::vector<DeletedRegion> &deleted_regs,
              std::set<std::pair<FieldSpace,FieldID> > &created_fids,
              std::vector<DeletedField> &deleted_fids,
              std::map<FieldSpace,unsigned> &created_fs,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_fs,
              std::vector<DeletedFieldSpace> &deleted_fs,
              std::map<IndexSpace,unsigned> &created_is,
              std::vector<DeletedIndexSpace> &deleted_is,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (!created_regs.empty())
      {
        if (!latent_field_spaces.empty())
        {
          for (std::map<LogicalRegion,unsigned>::const_iterator it = 
                created_regs.begin(); it != created_regs.end(); it++)
          {
            std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder =
              latent_field_spaces.find(it->first.get_field_space());
            if (finder != latent_field_spaces.end())
              finder->second.insert(it->first);
          }
        }
        if (!created_regions.empty())
        {
          for (std::map<LogicalRegion,unsigned>::const_iterator it = 
                created_regs.begin(); it != created_regs.end(); it++)
          {
            std::map<LogicalRegion,unsigned>::iterator finder = 
              created_regions.find(it->first);
            if (finder == created_regions.end())
              created_regions.insert(*it);
            else
              finder->second += it->second;
          }
        }
        else
          created_regions.swap(created_regs);
      }
      if (!deleted_regs.empty())
      {
        if (!deleted_regions.empty())
          deleted_regions.insert(deleted_regions.end(), 
              deleted_regs.begin(), deleted_regs.end());
        else
          deleted_regions.swap(deleted_regs);
      }
      if (!created_fids.empty())
      {
        if (!created_fields.empty())
        {
#ifdef DEBUG_LEGION
          for (std::set<std::pair<FieldSpace,FieldID> >::const_iterator it = 
                created_fids.begin(); it != created_fids.end(); it++)
          {
            assert(created_fields.find(*it) == created_fields.end());
            created_fields.insert(*it);
          }
#else
          created_fields.insert(created_fids.begin(), created_fids.end());
#endif
        }
        else
          created_fields.swap(created_fids);
      }
      if (!deleted_fids.empty())
      {
        if (!deleted_fields.empty())
          deleted_fields.insert(deleted_fields.end(), 
                                deleted_fids.begin(), deleted_fids.end());
        else
          deleted_fields.swap(deleted_fids);
      }
      if (!created_fs.empty())
      {
        if (!latent_field_spaces.empty())
        {
          // Remove any latent field spaces we have ownership for
          for (std::map<FieldSpace,unsigned>::const_iterator it =
                created_fs.begin(); it != created_fs.end(); it++)
          {
            std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
              latent_field_spaces.find(it->first);
            if (finder != latent_field_spaces.end())
              latent_field_spaces.erase(finder);
          }
        }
        if (!created_field_spaces.empty())
        {
          for (std::map<FieldSpace,unsigned>::const_iterator it = 
                created_fs.begin(); it != created_fs.end(); it++)
          {
            std::map<FieldSpace,unsigned>::iterator finder = 
              created_field_spaces.find(it->first);
            if (finder == created_field_spaces.end())
              created_field_spaces.insert(*it);
            else
              finder->second += it->second;
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
          for (std::map<FieldSpace,std::set<LogicalRegion> >::iterator it =
                latent_fs.begin(); it != latent_fs.end(); /*nothing*/)
          {
            if (created_field_spaces.find(it->first) != 
                  created_field_spaces.end())
            {
              std::map<FieldSpace,std::set<LogicalRegion> >::iterator 
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
          for (std::map<LogicalRegion,unsigned>::const_iterator it = 
                created_regions.begin(); it != created_regions.end(); it++)
          {
            std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
              latent_fs.find(it->first.get_field_space());
            if (finder != latent_fs.end())
              finder->second.insert(it->first);
          }
        }
        // Now we can do the merge
        if (!latent_field_spaces.empty())
        {
          for (std::map<FieldSpace,std::set<LogicalRegion> >::const_iterator 
                it = latent_fs.begin(); it != latent_fs.end(); it++)
          {
            std::map<FieldSpace,std::set<LogicalRegion> >::iterator finder = 
              latent_field_spaces.find(it->first);
            if (finder != latent_field_spaces.end())
              finder->second.insert(it->second.begin(), it->second.end());
            else
              latent_field_spaces.insert(*it);
          }
        }
        else
          latent_field_spaces.swap(latent_fs);
      }
      if (!deleted_fs.empty())
      {
        if (!deleted_field_spaces.empty())
          deleted_field_spaces.insert(deleted_field_spaces.end(), 
                                      deleted_fs.begin(), deleted_fs.end());
        else
          deleted_field_spaces.swap(deleted_fs);
      }
      if (!created_is.empty())
      {
        if (!created_index_spaces.empty())
        {
          for (std::map<IndexSpace,unsigned>::const_iterator it = 
                created_is.begin(); it != created_is.end(); it++)
          {
            std::map<IndexSpace,unsigned>::iterator finder = 
              created_index_spaces.find(it->first);
            if (finder == created_index_spaces.end())
              created_index_spaces.insert(*it);
            else
              finder->second += it->second;
          }
        }
        else
          created_index_spaces.swap(created_is);
      }
      if (!deleted_is.empty())
      {
        if (!deleted_index_spaces.empty())
          deleted_index_spaces.insert(deleted_index_spaces.end(), 
                                      deleted_is.begin(), deleted_is.end());
        else
          deleted_index_spaces.swap(deleted_is);
      }
      if (!created_partitions.empty())
      {
        if (!created_index_partitions.empty())
        {
          for (std::map<IndexPartition,unsigned>::const_iterator it = 
                created_partitions.begin(); it != 
                created_partitions.end(); it++)
          {
            std::map<IndexPartition,unsigned>::iterator finder = 
              created_index_partitions.find(it->first);
            if (finder == created_index_partitions.end())
              created_index_partitions.insert(*it);
            else
              finder->second += it->second;
          }
        }
        else
          created_index_partitions.swap(created_partitions);
      }
      if (!deleted_partitions.empty())
      {
        if (!deleted_index_partitions.empty())
          deleted_index_partitions.insert(deleted_index_partitions.end(),
              deleted_partitions.begin(), deleted_partitions.end());
        else
          deleted_index_partitions.swap(deleted_partitions);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::expand_replay_slices(std::list<SliceTask*> &slices)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!points.empty());
      assert(is_origin_mapped());
#endif
      // For each point give it its own slice owner in case we need to
      // to move it remotely as part of the replay
      while (points.size() > 1)
      {
        PointTask *point = points.back();
        points.pop_back();
        SliceTask *new_owner = clone_as_slice_task(internal_space,
                current_proc, false/*recurse*/, false/*stealable*/);
        point->slice_owner = new_owner;
        new_owner->points.push_back(point);
        new_owner->num_unmapped_points = 1;
        new_owner->num_uncomplete_points = 1;
        new_owner->num_uncommitted_points = 1;
        slices.push_back(new_owner);
      }
      // Always add ourselves as the last point
      slices.push_back(this);
      num_unmapped_points = points.size();
      num_uncomplete_points = points.size();
      num_uncommitted_points = points.size();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->trigger_replay();
    }

    //--------------------------------------------------------------------------
    void SliceTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->complete_replay(instance_ready_event);
    }

    //--------------------------------------------------------------------------
    void SliceTask::find_commit_preconditions(std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // See if any of our point tasks have any profiling report events
      for (std::vector<PointTask*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
      {
        const RtEvent profiling_reported = (*it)->get_profiling_reported();
        if (profiling_reported.exists())
          preconditions.insert(profiling_reported);
      }
      // See if we haven't completed it yet, if so make an event to track it
      AutoLock o_lock(op_lock);
      if (!complete_received)
      {
        origin_mapped_complete = Runtime::create_rt_user_event();
        preconditions.insert(origin_mapped_complete);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      // See if we can find or make it
      {
        AutoLock o_lock(op_lock);
        std::map<DomainPoint,RtEvent>::const_iterator finder = 
          intra_space_dependences.find(point);
        // If we've already got it then we're done
        if (finder != intra_space_dependences.end())
          return finder->second;
#ifdef DEBUG_LEGION
        assert(!points.empty());
#endif
        // Next see if it is one of our local points
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          if ((*it)->index_point != point)
            continue;
          // Don't save this in our intra_space_dependences data structure!
          // Doing so could mess up our optimization for detecting when 
          // we need to send dependences back to the origin
          // See SliceTask::record_intra_space_dependence
          return (*it)->get_mapped_event();
        }
        // If we're remote, make up an event and send a message to go find it
        if (is_remote())
        {
          const RtUserEvent temp_event = Runtime::create_rt_user_event();
          // Send the message to the owner to go find it
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize(point);
            rez.serialize(temp_event);
          }
          runtime->send_slice_find_intra_space_dependence(orig_proc, rez);
          // Save this is for ourselves
          intra_space_dependences[point] = temp_event;
          return temp_event;
        }
      }
      // If we make it down here then we're on the same node as the 
      // index_owner so we can just as it what the answer and save it
      const RtEvent result = index_owner->find_intra_space_dependence(point);
      AutoLock o_lock(op_lock);
      intra_space_dependences[point] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_intra_space_dependence(const DomainPoint &point,
                                                  RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
      // Check to see if we already sent it already
      {
        AutoLock o_lock(op_lock);
        std::map<DomainPoint,RtEvent>::const_iterator finder = 
          intra_space_dependences.find(point);
        if (finder != intra_space_dependences.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second == point_mapped);
#endif
          return;
        }
        // Otherwise save it and then let it flow back to the index owner
        intra_space_dependences[point] = point_mapped;
      }
      if (is_remote())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(point);
          rez.serialize(point_mapped);
        }
        runtime->send_slice_record_intra_space_dependence(orig_proc, rez);
      }
      else
        index_owner->record_intra_space_dependence(point, point_mapped);
    }
    
    //--------------------------------------------------------------------------
    CollectiveManager* SliceTask::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        std::atomic<CollectiveManager*> result(NULL);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(mapper_call);
          rez.serialize(SLICE_COLLECTIVE_FIND_OR_CREATE);
          rez.serialize(index);
          constraints.serialize(rez);
          rez.serialize<size_t>(regions.size());
          for (std::vector<LogicalRegion>::const_iterator it = 
                regions.begin(); it != regions.end(); it++)
            rez.serialize(*it);
          rez.serialize(kind);
          rez.serialize(footprint);
          rez.serialize(unsat_kind);
          rez.serialize(unsat_index);
          rez.serialize(collective_point);
          rez.serialize(&result);
          rez.serialize(ready_event);
        }
        runtime->send_slice_collective_instance_request(orig_proc, rez);
        ready_event.wait();
        return result;
      }
      else
        return index_owner->find_or_create_collective_instance(mapper_call, 
            index, constraints, regions, kind, footprint, unsat_kind, 
            unsat_index, collective_point);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::finalize_collective_instance(MappingCallKind call_kind,
                                                 unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        std::atomic<bool> result(success);
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(call_kind);
          rez.serialize(SLICE_COLLECTIVE_FINALIZE);
          rez.serialize(index);
          rez.serialize<bool>(success);
          rez.serialize(&result);
          rez.serialize(ready_event);
        }
        runtime->send_slice_collective_instance_request(orig_proc, rez);
        ready_event.wait();
        return result;
      }
      else
        return index_owner->finalize_collective_instance(call_kind, index, 
                                                         success);
    }

    //--------------------------------------------------------------------------
    void SliceTask::report_total_collective_instance_calls(
                              MappingCallKind mapper_call, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        RtUserEvent applied_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(mapper_call);
          rez.serialize(SLICE_COLLECTIVE_REPORT);
          rez.serialize(total_calls);
          rez.serialize(applied_event);
        }
        runtime->send_slice_collective_instance_request(orig_proc, rez);
        AutoLock o_lock(op_lock);
        map_applied_conditions.insert(applied_event);
      }
      else
        index_owner->report_total_collective_instance_calls(mapper_call,
                                                            total_calls);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_collective_instance_request(
                   Deserializer &derez, AddressSpaceID source, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *index_owner;
      derez.deserialize(index_owner);
      MappingCallKind mapper_call;
      derez.deserialize(mapper_call);
      CollectiveInstMessage message;
      derez.deserialize(message);
      switch (message)
      {
        case SLICE_COLLECTIVE_FIND_OR_CREATE:
          {
            unsigned index;
            derez.deserialize(index);
            LayoutConstraintSet constraints;
            constraints.deserialize(derez);
            size_t num_regions;
            derez.deserialize(num_regions);
            std::vector<LogicalRegion> regions(num_regions);
            for (unsigned idx = 0; idx < num_regions; idx++)
              derez.deserialize(regions[idx]);
            Memory::Kind mem_kind;
            derez.deserialize(mem_kind);
            size_t *remote_footprint;
            derez.deserialize(remote_footprint);
            LayoutConstraintKind *remote_unsat_kind;
            derez.deserialize(remote_unsat_kind);
            unsigned *remote_unsat_index;
            derez.deserialize(remote_unsat_index);
            DomainPoint point;
            derez.deserialize(point);
            std::atomic<CollectiveManager*> *target;
            derez.deserialize(target);
            RtUserEvent done_event;
            derez.deserialize(done_event);

            size_t footprint = 0;
            LayoutConstraintKind unsat_kind = LEGION_SPECIALIZED_CONSTRAINT;
            unsigned unsat_index = 0;
            CollectiveManager *result = 
              index_owner->find_or_create_collective_instance(mapper_call,
                  index, constraints, regions, mem_kind, &footprint,
                  &unsat_kind, &unsat_index, point);
            if ((result != NULL) || (remote_footprint != NULL) || 
                (remote_unsat_kind != NULL) || (remote_unsat_index != NULL))
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(message);
                if (result != NULL)
                {
                  rez.serialize(result->did);
                  rez.serialize(target);
                }
                else
                  rez.serialize<DistributedID>(0);
                rez.serialize(remote_footprint);
                if (remote_footprint != NULL)
                  rez.serialize(footprint);
                rez.serialize(remote_unsat_kind);
                if (remote_unsat_kind != NULL)
                  rez.serialize(unsat_kind);
                rez.serialize(remote_unsat_index);
                if (remote_unsat_index != NULL)
                  rez.serialize(unsat_index);
                rez.serialize(done_event);
              }
              runtime->send_slice_collective_instance_response(source, rez);
            }
            else // Nothing to send back so just trigger the event
              Runtime::trigger_event(done_event);
            break;
          }
        case SLICE_COLLECTIVE_FINALIZE:
          {
            unsigned index;
            derez.deserialize(index);
            bool success;
            derez.deserialize(success);
            std::atomic<bool> *target;
            derez.deserialize(target);
            RtUserEvent done_event;
            derez.deserialize(done_event);
            const bool result = index_owner->finalize_collective_instance(
                                              mapper_call, index, success);
            if (result != success)
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(message);
                rez.serialize(target);
                rez.serialize(result);
                rez.serialize(done_event);
              }
              runtime->send_slice_collective_instance_response(source, rez);
            }
            else // Nothing to send back so just trigger the event
              Runtime::trigger_event(done_event);
            break;
          }
        case SLICE_COLLECTIVE_REPORT:
          {
            unsigned total_calls;
            derez.deserialize(total_calls);
            RtUserEvent done_event;
            derez.deserialize(done_event);
            index_owner->report_total_collective_instance_calls(mapper_call, 
                                                                total_calls);            
            Runtime::trigger_event(done_event);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_collective_instance_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      CollectiveInstMessage message;
      derez.deserialize(message);
      switch (message)
      {
        case SLICE_COLLECTIVE_FIND_OR_CREATE:
          {
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready_event;
            if (did > 0)
            {
              std::atomic<CollectiveManager*> *target;
              derez.deserialize(target);
              target->store(static_cast<CollectiveManager*>(
                  runtime->find_or_request_instance_manager(did, ready_event)));
            }
            size_t *footprint;
            derez.deserialize(footprint);
            if (footprint != NULL)
              derez.deserialize(*footprint);
            LayoutConstraintKind *unsat_kind;
            derez.deserialize(unsat_kind);
            if (unsat_kind != NULL)
              derez.deserialize(*unsat_kind);
            unsigned *unsat_index;
            derez.deserialize(unsat_index);
            if (unsat_index != NULL)
              derez.deserialize(*unsat_index);
            RtUserEvent done_event;
            derez.deserialize(done_event);
            Runtime::trigger_event(done_event, ready_event);
            break;
          }
        case SLICE_COLLECTIVE_FINALIZE:
          {
            std::atomic<bool> *target;
            derez.deserialize(target);
            bool result;
            derez.deserialize<bool>(result);
            target->store(result);
            RtUserEvent done_event;
            derez.deserialize(done_event);
            Runtime::trigger_event(done_event);
          }
        default:
          assert(false);
      }
    }

  }; // namespace Internal 
}; // namespace Legion 

#undef PRINT_REG

// EOF

