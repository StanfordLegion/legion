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

#ifndef __LEGION_DEPENDENT_PARTITION_H__
#define __LEGION_DEPENDENT_PARTITION_H__

#include "legion/analysis/versioning.h"
#include "legion/api/functors_impl.h"
#include "legion/operations/collective.h"
#include "legion/operations/remote.h"
#include "legion/nodes/index.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalPartition
     * An extension of the external-facing Partition to help
     * with packing and unpacking them
     */
    class ExternalPartition : public Partition,
                              public ExternalMappable {
    public:
      ExternalPartition(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_partition(
          Serializer& rez, AddressSpaceID target) const;
      void unpack_external_partition(Deserializer& derez);
    };

    /**
     * \class DependentPartitionOp
     * An operation for creating different kinds of partitions
     * which are dependent on mapping a region in order to compute
     * the resulting partition.
     */
    class DependentPartitionOp : public ExternalPartition,
                                 public Operation {
    protected:
      // Track dependent partition operations as thunks
      class DepPartThunk {
      public:
        virtual ~DepPartThunk(void) { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr) = 0;
        virtual PartitionKind get_kind(void) const = 0;
        virtual IndexPartition get_partition(void) const = 0;
        virtual IndexPartition get_projection(void) const = 0;
        virtual bool safe_projection(IndexPartition p) const { return false; }
        virtual bool is_image(void) const { return false; }
        virtual bool is_preimage(void) const { return false; }
      };
      class ByFieldThunk : public DepPartThunk {
      public:
        ByFieldThunk(IndexPartition p) : pid(p) { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_FIELD; }
        virtual IndexPartition get_partition(void) const { return pid; }
        virtual IndexPartition get_projection(void) const
        {
          return IndexPartition::NO_PART;
        }
      protected:
        IndexPartition pid;
      };
      class ByImageThunk : public DepPartThunk {
      public:
        ByImageThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj)
        { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_IMAGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
        virtual IndexPartition get_projection(void) const { return projection; }
        virtual bool safe_projection(IndexPartition p) const
        {
          return (p == projection);
        }
        virtual bool is_image(void) const { return true; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByImageRangeThunk : public DepPartThunk {
      public:
        ByImageRangeThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj)
        { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_IMAGE_RANGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
        virtual IndexPartition get_projection(void) const { return projection; }
        virtual bool safe_projection(IndexPartition p) const
        {
          return (p == projection);
        }
        virtual bool is_image(void) const { return true; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByPreimageThunk : public DepPartThunk {
      public:
        ByPreimageThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj)
        { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_PREIMAGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
        virtual IndexPartition get_projection(void) const { return projection; }
        virtual bool is_preimage(void) const { return true; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByPreimageRangeThunk : public DepPartThunk {
      public:
        ByPreimageRangeThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj)
        { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_PREIMAGE_RANGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
        virtual IndexPartition get_projection(void) const { return projection; }
        virtual bool is_preimage(void) const { return true; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class AssociationThunk : public DepPartThunk {
      public:
        AssociationThunk(IndexSpace d, IndexSpace r) : domain(d), range(r) { }
      public:
        virtual ApEvent perform(
            DependentPartitionOp* op, FieldID fid, ApEvent instances_ready,
            std::vector<FieldDataDescriptor>& instances,
            const std::map<DomainPoint, Domain>* remote_targets = nullptr,
            std::vector<DeppartResult>* results = nullptr);
        virtual PartitionKind get_kind(void) const { return BY_ASSOCIATION; }
        virtual IndexPartition get_partition(void) const
        {
          return IndexPartition::NO_PART;
        }
        virtual IndexPartition get_projection(void) const
        {
          return IndexPartition::NO_PART;
        }
      protected:
        IndexSpace domain;
        IndexSpace range;
      };
    public:
      DependentPartitionOp(void);
      DependentPartitionOp(const DependentPartitionOp& rhs) = delete;
      virtual ~DependentPartitionOp(void);
    public:
      DependentPartitionOp& operator=(const DependentPartitionOp& rhs) = delete;
    public:
      void initialize_by_field(
          InnerContext* ctx, IndexPartition pid, LogicalRegion handle,
          LogicalRegion parent, IndexSpace color_space, FieldID fid,
          MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance);
      void initialize_by_image(
          InnerContext* ctx, IndexPartition pid, IndexSpace handle,
          LogicalPartition projection, LogicalRegion parent, FieldID fid,
          MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance);
      void initialize_by_image_range(
          InnerContext* ctx, IndexPartition pid, IndexSpace handle,
          LogicalPartition projection, LogicalRegion parent, FieldID fid,
          MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance);
      void initialize_by_preimage(
          InnerContext* ctx, IndexPartition pid, IndexPartition projection,
          LogicalRegion handle, LogicalRegion parent, FieldID fid, MapperID id,
          MappingTagID tag, const UntypedBuffer& marg, Provenance* provenance);
      void initialize_by_preimage_range(
          InnerContext* ctx, IndexPartition pid, IndexPartition projection,
          LogicalRegion handle, LogicalRegion parent, FieldID fid, MapperID id,
          MappingTagID tag, const UntypedBuffer& marg, Provenance* provenance);
      void initialize_by_association(
          InnerContext* ctx, LogicalRegion domain, LogicalRegion domain_parent,
          FieldID fid, IndexSpace range, MapperID id, MappingTagID tag,
          const UntypedBuffer& marg, Provenance* provenance);
      void perform_logging(void) const;
      void log_requirement(void) const;
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      // A method for override with control replication
      virtual void finalize_mapping(void);
      virtual ApEvent trigger_thunk(
          IndexSpace handle, ApEvent insts_ready,
          const InstanceSet& mapped_instances, const PhysicalTraceInfo& info,
          const DomainPoint& color);
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual bool is_partition_op(void) const override { return true; }
      virtual void select_partition_projection(void);
    public:
      virtual PartitionKind get_partition_kind(void) const override;
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual Mappable* get_mappable(void) override;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual void trigger_commit(void) override;
      virtual IndexSpaceNode* get_shard_points(void) const
      {
        return launch_space;
      }
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    public:
      void activate_dependent(void);
      void deactivate_dependent(void);
    public:
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      // Report a profiling result for this operation
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
      virtual void handle_profiling_update(int count) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    public:
      virtual size_t get_collective_points(void) const override;
    protected:
      bool invoke_mapper(
          InstanceSet& mapped_instances,
          std::vector<PhysicalManager*>& source_instances);
      void activate_dependent_op(void);
      void deactivate_dependent_op(void);
      void finalize_partition_profiling(void);
      void find_open_complete_partitions(
          std::vector<LogicalPartition>& partitions) const;
    public:
      void handle_point_complete(ApEvent effects);
      void handle_point_commit(RtEvent point_committed);
    public:
      VersionInfo version_info;
      unsigned parent_req_index;
      std::map<PhysicalManager*, unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
      DepPartThunk* thunk;
    protected:
      MapperManager* mapper;
    protected:
      // For index versions of this operation
      IndexSpaceNode* launch_space;
      std::vector<FieldDataDescriptor> instances;
      std::vector<ApEvent> index_preconditions;
      std::vector<PointDepPartOp*> points;
      std::atomic<int> points_completed;
      unsigned points_committed;
      bool commit_request;
      std::set<RtEvent> commit_preconditions;
      ApUserEvent intermediate_index_event;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    };

    /**
     * \class PointDepPartOp
     * This is a point class for mapping a particular
     * subregion of a partition for a dependent partitioning
     * operation.
     */
    class PointDepPartOp : public DependentPartitionOp,
                           public ProjectionPoint {
    public:
      PointDepPartOp(void);
      PointDepPartOp(const PointDepPartOp& rhs) = delete;
      virtual ~PointDepPartOp(void);
    public:
      PointDepPartOp& operator=(const PointDepPartOp& rhs) = delete;
    public:
      void initialize(DependentPartitionOp* owner, const DomainPoint& point);
      void launch(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual ApEvent trigger_thunk(
          IndexSpace handle, ApEvent insts_ready,
          const InstanceSet& mapped_instances,
          const PhysicalTraceInfo& trace_info,
          const DomainPoint& color) override;
      virtual void trigger_complete(ApEvent effect) override;
      virtual void trigger_commit(void) override;
      virtual PartitionKind get_partition_kind(void) const override;
      virtual unsigned find_parent_index(unsigned idx) override
      {
        return owner->find_parent_index(idx);
      }
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      virtual size_t get_collective_points(void) const override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    public:
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const override;
      virtual void set_projection_result(
          unsigned idx, LogicalRegion result) override;
      virtual void record_intra_space_dependences(
          unsigned idx, const std::vector<DomainPoint>& region_deps) override;
      virtual void record_pointwise_dependence(
          uint64_t previous_context_index, const DomainPoint& previous_point,
          ShardID shard) override;
      virtual const Operation* as_operation(void) const override
      {
        return this;
      }
    public:
      DependentPartitionOp* owner;
    };

    /**
     * \class DeppartResultScatter
     * Scatter the results of a dependent partitioning operation
     * back across the shards so they can fill in their nodes
     */
    class DeppartResultScatter : public BroadcastCollective {
    public:
      DeppartResultScatter(
          ReplicateContext* ctx, CollectiveID id,
          std::vector<DeppartResult>& results);
      DeppartResultScatter(const DeppartResultScatter& rhs) = delete;
      virtual ~DeppartResultScatter(void);
    public:
      DeppartResultScatter& operator=(const DeppartResultScatter& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_DEPPART_RESULT_SCATTER;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    public:
      void broadcast_results(ApEvent done_event);
      inline ApEvent get_done_event(void) { return done_event; }
    public:
      std::vector<DeppartResult>& results;
      const ApUserEvent done_event;
    };

    /**
     * \class FieldDescriptorExchange
     * A class for doing an all-gather of field descriptors for
     * doing dependent partitioning operations. This will also build
     * a butterfly tree of user events that will be used to know when
     * all of the constituent shards are done with the operation they
     * are collectively performing together.
     */
    class FieldDescriptorExchange : public AllGatherCollective<true> {
    public:
      FieldDescriptorExchange(
          ReplicateContext* ctx, CollectiveID id,
          std::vector<FieldDataDescriptor>& descriptors);
      FieldDescriptorExchange(const FieldDescriptorExchange& rhs) = delete;
      virtual ~FieldDescriptorExchange(void);
    public:
      FieldDescriptorExchange& operator=(const FieldDescriptorExchange& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    protected:
      std::vector<FieldDataDescriptor>& descriptors;
    };

    /**
     * \class FieldDescriptorGather
     * Gather all of the field descriptors to a particular shard and
     * track the merge of all the ready events
     */
    class FieldDescriptorGather : public GatherCollective {
    public:
      FieldDescriptorGather(
          ReplicateContext* ctx, CollectiveID id,
          std::vector<FieldDataDescriptor>& descriptors,
          std::map<DomainPoint, Domain>& remote_targets);
      FieldDescriptorGather(const FieldDescriptorGather& rhs) = delete;
      virtual ~FieldDescriptorGather(void);
    public:
      FieldDescriptorGather& operator=(const FieldDescriptorGather& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_FIELD_DESCRIPTOR_GATHER;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    public:
      void contribute_instances(ApEvent instances_ready);
      ApEvent get_ready_event(void);
    protected:
      std::vector<FieldDataDescriptor>& descriptors;
      std::map<DomainPoint, Domain>& remote_targets;
      std::vector<ApEvent> ready_events;
    };

    /**
     * \class ReplDependentPartitionOp
     * A dependent partitioning operation that knows that it
     * is being executed in a control replication context
     */
    class ReplDependentPartitionOp
      : public ReplCollectiveViewCreator<
            CollectiveViewCreator<DependentPartitionOp> > {
    public:
      ReplDependentPartitionOp(void);
      ReplDependentPartitionOp(const ReplDependentPartitionOp& rhs) = delete;
      virtual ~ReplDependentPartitionOp(void);
    public:
      ReplDependentPartitionOp& operator=(const ReplDependentPartitionOp& rhs) =
          delete;
    public:
      void initialize_replication(ReplicateContext* context);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      // Need to pick our sharding functor
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void finalize_mapping(void) override;
      virtual ApEvent trigger_thunk(
          IndexSpace handle, ApEvent insts_ready,
          const InstanceSet& mapped_instances, const PhysicalTraceInfo& info,
          const DomainPoint& color) override;
      virtual void trigger_execution(void) override;
      virtual void select_partition_projection(void) override;
      virtual IndexSpaceNode* get_shard_points(void) const override
      {
        return shard_points;
      }
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
    protected:
      void select_sharding_function(void);
      void find_remote_targets(
          std::vector<ApEvent>& preconditions, ApUserEvent& to_trigger);
    protected:
      ShardingFunction* sharding_function;
      IndexSpaceNode* shard_points;
      RtBarrier mapping_barrier;
      FieldDescriptorGather* gather;
      DeppartResultScatter* scatter;
      FieldDescriptorExchange* exchange;
      ApBarrier collective_ready;
      ApBarrier collective_done;
      std::map<DomainPoint, Domain> remote_targets;
      std::vector<DeppartResult> deppart_results;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

    /**
     * \class RemotePartitionOp
     * This is a remote copy of a DependentPartitionOp to be
     * used for mapper calls and other operations
     */
    class RemotePartitionOp
      : public ExternalPartition,
        public RemoteOp,
        public Heapify<RemotePartitionOp, OPERATION_LIFETIME> {
    public:
      RemotePartitionOp(Operation* ptr, AddressSpaceID src);
      RemotePartitionOp(const RemotePartitionOp& rhs) = delete;
      virtual ~RemotePartitionOp(void);
    public:
      RemotePartitionOp& operator=(const RemotePartitionOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual PartitionKind get_partition_kind(void) const override;
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    protected:
      PartitionKind part_kind;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(
        std::ostream& os, const DependentPartitionOp& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_logging_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DEPENDENT_PARTITION_H__
