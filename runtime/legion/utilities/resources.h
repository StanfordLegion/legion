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

#ifndef __LEGION_RESOURCE_TRACKER_H__
#define __LEGION_RESOURCE_TRACKER_H__

#include "legion/api/data.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ResourceTracker
     * A helper class for tracking which privileges an
     * operation owns. This is inherited by multi-tasks
     * for aggregating the privilege results of their
     * children as well as task contexts for tracking
     * which privileges have been accrued or deleted
     * as part of the execution of the task.
     */
    class ResourceTracker {
    public:
      struct DeletedRegion {
      public:
        DeletedRegion(void);
        DeletedRegion(LogicalRegion r, Provenance* provenance = nullptr);
        DeletedRegion(const DeletedRegion& rhs);
        DeletedRegion(DeletedRegion&& rhs) noexcept;
        ~DeletedRegion(void);
      public:
        DeletedRegion& operator=(const DeletedRegion& rhs);
        DeletedRegion& operator=(DeletedRegion&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        LogicalRegion region;
        Provenance* provenance;
      };
      struct DeletedField {
      public:
        DeletedField(void);
        DeletedField(
            FieldSpace sp, FieldID f, Provenance* provenance = nullptr);
        DeletedField(const DeletedField& rhs);
        DeletedField(DeletedField&& rhs) noexcept;
        ~DeletedField(void);
      public:
        DeletedField& operator=(const DeletedField& rhs);
        DeletedField& operator=(DeletedField&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        FieldSpace space;
        FieldID fid;
        Provenance* provenance;
      };
      struct DeletedFieldSpace {
      public:
        DeletedFieldSpace(void);
        DeletedFieldSpace(FieldSpace sp, Provenance* provenance = nullptr);
        DeletedFieldSpace(const DeletedFieldSpace& rhs);
        DeletedFieldSpace(DeletedFieldSpace&& rhs) noexcept;
        ~DeletedFieldSpace(void);
      public:
        DeletedFieldSpace& operator=(const DeletedFieldSpace& rhs);
        DeletedFieldSpace& operator=(DeletedFieldSpace&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        FieldSpace space;
        Provenance* provenance;
      };
      struct DeletedIndexSpace {
      public:
        DeletedIndexSpace(void);
        DeletedIndexSpace(
            IndexSpace sp, bool recurse, Provenance* provenance = nullptr);
        DeletedIndexSpace(const DeletedIndexSpace& rhs);
        DeletedIndexSpace(DeletedIndexSpace&& rhs) noexcept;
        ~DeletedIndexSpace(void);
      public:
        DeletedIndexSpace& operator=(const DeletedIndexSpace& rhs);
        DeletedIndexSpace& operator=(DeletedIndexSpace&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        IndexSpace space;
        Provenance* provenance;
        bool recurse;
      };
      struct DeletedPartition {
      public:
        DeletedPartition(void);
        DeletedPartition(
            IndexPartition p, bool recurse, Provenance* provenance = nullptr);
        DeletedPartition(const DeletedPartition& rhs);
        DeletedPartition(DeletedPartition&& rhs) noexcept;
        ~DeletedPartition(void);
      public:
        DeletedPartition& operator=(const DeletedPartition& rhs);
        DeletedPartition& operator=(DeletedPartition&& rhs) noexcept;
      public:
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        IndexPartition partition;
        Provenance* provenance;
        bool recurse;
      };
    public:
      ResourceTracker(void);
      ResourceTracker(const ResourceTracker& rhs) = delete;
      virtual ~ResourceTracker(void);
    public:
      ResourceTracker& operator=(const ResourceTracker& rhs) = delete;
    public:
      // Delete this function once MustEpochOps are gone
      void return_resources(
          ResourceTracker* target, uint64_t return_index,
          std::set<RtEvent>& preconditions);
      virtual void receive_resources(
          uint64_t return_index,
          std::map<LogicalRegion, unsigned>& created_regions,
          std::vector<DeletedRegion>& deleted_regions,
          std::set<std::pair<FieldSpace, FieldID> >& created_fields,
          std::vector<DeletedField>& deleted_fields,
          std::map<FieldSpace, unsigned>& created_field_spaces,
          std::map<FieldSpace, std::set<LogicalRegion> >& latent_spaces,
          std::vector<DeletedFieldSpace>& deleted_field_spaces,
          std::map<IndexSpace, unsigned>& created_index_spaces,
          std::vector<DeletedIndexSpace>& deleted_index_spaces,
          std::map<IndexPartition, unsigned>& created_partitions,
          std::vector<DeletedPartition>& deleted_partitions,
          std::set<RtEvent>& preconditions) = 0;
      void pack_resources_return(Serializer& rez, uint64_t return_index);
      static void pack_empty_resources(Serializer& rez, uint64_t return_index);
      static RtEvent unpack_resources_return(
          Deserializer& derez, ResourceTracker* target);
    protected:
      void merge_received_resources(
          std::map<LogicalRegion, unsigned>& created_regions,
          std::vector<DeletedRegion>& deleted_regions,
          std::set<std::pair<FieldSpace, FieldID> >& created_fields,
          std::vector<DeletedField>& deleted_fields,
          std::map<FieldSpace, unsigned>& created_field_spaces,
          std::map<FieldSpace, std::set<LogicalRegion> >& latent_spaces,
          std::vector<DeletedFieldSpace>& deleted_field_spaces,
          std::map<IndexSpace, unsigned>& created_index_spaces,
          std::vector<DeletedIndexSpace>& deleted_index_spaces,
          std::map<IndexPartition, unsigned>& created_partitions,
          std::vector<DeletedPartition>& deleted_partitions);
    protected:
      std::map<LogicalRegion, unsigned> created_regions;
      std::map<LogicalRegion, bool> local_regions;
      std::set<std::pair<FieldSpace, FieldID> > created_fields;
      std::map<std::pair<FieldSpace, FieldID>, bool> local_fields;
      std::map<FieldSpace, unsigned> created_field_spaces;
      std::map<IndexSpace, unsigned> created_index_spaces;
      std::map<IndexPartition, unsigned> created_index_partitions;
    protected:
      std::vector<DeletedRegion> deleted_regions;
      std::vector<DeletedField> deleted_fields;
      std::vector<DeletedFieldSpace> deleted_field_spaces;
      std::map<FieldSpace, std::set<LogicalRegion> > latent_field_spaces;
      std::vector<DeletedIndexSpace> deleted_index_spaces;
      std::vector<DeletedPartition> deleted_index_partitions;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_RESOURCE_TRACKER_H__
