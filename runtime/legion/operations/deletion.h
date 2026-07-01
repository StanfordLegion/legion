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

#ifndef __LEGION_DELETION_H__
#define __LEGION_DELETION_H__

#include "legion/operations/collective.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /**
     * \class DeletionOp
     * In keeping with the deferred execution model, deletions
     * must be deferred until all other operations that were
     * issued earlier are done using the regions that are
     * going to be deleted.  Deletion operations defer deletions
     * until they are safe to be committed.
     */
    class DeletionOp : public Operation {
    public:
      enum DeletionKind {
        INDEX_SPACE_DELETION,
        INDEX_PARTITION_DELETION,
        FIELD_SPACE_DELETION,
        FIELD_DELETION,
        LOGICAL_REGION_DELETION,
      };
    public:
      DeletionOp(void);
      DeletionOp(const DeletionOp& rhs) = delete;
      virtual ~DeletionOp(void);
    public:
      DeletionOp& operator=(const DeletionOp& rhs) = delete;
    public:
      void set_deletion_preconditions(
          const std::map<Operation*, GenerationID>& dependences);
    public:
      void initialize_index_space_deletion(
          InnerContext* ctx, IndexSpace handle,
          std::vector<IndexPartition>& sub_partitions, const bool unordered,
          Provenance* provenance);
      void initialize_index_part_deletion(
          InnerContext* ctx, IndexPartition part,
          std::vector<IndexPartition>& sub_partitions, const bool unordered,
          Provenance* provenance);
      void initialize_field_space_deletion(
          InnerContext* ctx, FieldSpace handle, const bool unordered,
          Provenance* provenance);
      void initialize_field_deletion(
          InnerContext* ctx, FieldSpace handle, FieldID fid,
          const bool unordered, FieldAllocatorImpl* allocator,
          Provenance* provenance, const bool non_owner_shard);
      void initialize_field_deletions(
          InnerContext* ctx, FieldSpace handle,
          const std::set<FieldID>& to_free, const bool unordered,
          FieldAllocatorImpl* allocator, Provenance* provenance,
          const bool non_owner_shard);
      void initialize_logical_region_deletion(
          InnerContext* ctx, LogicalRegion handle, const bool unordered,
          Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override
      {
        return deletion_requirements.size();
      }
      virtual const RegionRequirement& get_requirement(
          unsigned idx) const override
      {
        return deletion_requirements[idx];
      }
    protected:
      void create_deletion_requirements(void);
      void log_deletion_requirements(void);
      void invalidate_fields(
          unsigned index, const RegionRequirement& req,
          const VersionInfo& version_info, const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping,
          const bool collective_first_local);
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    protected:
      DeletionKind kind;
      IndexSpace index_space;
      IndexPartition index_part;
      std::vector<IndexPartition> sub_partitions;
      FieldSpace field_space;
      FieldAllocatorImpl* allocator;
      LogicalRegion logical_region;
      std::set<FieldID> free_fields;
      std::vector<FieldID> local_fields;
      std::vector<FieldID> global_fields;
      std::vector<unsigned> local_field_indexes;
      std::vector<unsigned> parent_req_indexes;
      std::vector<unsigned> deletion_req_indexes;
      std::vector<bool> returnable_privileges;
      std::vector<RegionRequirement> deletion_requirements;
      op::vector<VersionInfo> version_infos;
      std::set<RtEvent> map_applied_conditions;
      std::map<Operation*, GenerationID> dependences;
      bool has_preconditions;
    };

    /**
     * \class ReplDeletionOp
     * A deletion operation that is aware that it is
     * being executed in a control replication context.
     */
    class ReplDeletionOp
      : public ReplCollectiveVersioning<CollectiveVersioning<DeletionOp> > {
    public:
      struct DeferDeletionCommitArgs
        : public LgTaskArgs<DeferDeletionCommitArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_DELETION_COMMIT_TASK_ID;
      public:
        DeferDeletionCommitArgs(void) = default;
        DeferDeletionCommitArgs(ReplDeletionOp* o)
          : LgTaskArgs(false, false), op(o)
        { }
        void execute(void) const;
      public:
        ReplDeletionOp* op;
      };
    public:
      ReplDeletionOp(void);
      ReplDeletionOp(const ReplDeletionOp& rhs) = delete;
      virtual ~ReplDeletionOp(void);
    public:
      ReplDeletionOp& operator=(const ReplDeletionOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
    public:
      void initialize_replication(
          ReplicateContext* ctx, bool is_first,
          RtBarrier* ready_barrier = nullptr,
          RtBarrier* mapping_barrier = nullptr,
          RtBarrier* commit_barrier = nullptr);
      // Help for handling unordered deletions
      void record_unordered_kind(
          std::map<IndexSpace, ReplDeletionOp*>& index_space_deletions,
          std::map<IndexPartition, ReplDeletionOp*>& index_partition_deletions,
          std::map<FieldSpace, ReplDeletionOp*>& field_space_deletions,
          std::map<std::pair<FieldSpace, FieldID>, ReplDeletionOp*>&
              field_deletions,
          std::map<LogicalRegion, ReplDeletionOp*>& logical_region_deletions);
    protected:
      RtBarrier ready_barrier;
      RtBarrier mapping_barrier;
      RtBarrier commit_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class RemoteDeletionOp
     * This is a remote copy of a DeletionOp to be used for
     * mapper calls and other operations
     */
    class RemoteDeletionOp
      : public RemoteOp,
        public Heapify<RemoteDeletionOp, OPERATION_LIFETIME> {
    public:
      RemoteDeletionOp(Operation* ptr, AddressSpaceID src);
      RemoteDeletionOp(const RemoteDeletionOp& rhs) = delete;
      virtual ~RemoteDeletionOp(void);
    public:
      RemoteDeletionOp& operator=(const RemoteDeletionOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DELETION_H__
