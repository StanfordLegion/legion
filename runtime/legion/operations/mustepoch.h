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

#ifndef __LEGION_MUST_EPOCH_H__
#define __LEGION_MUST_EPOCH_H__

#include "legion/kernel/runtime.h"
#include "legion/operations/operation.h"
#include "legion/api/mapping.h"
#include "legion/api/redop.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/resources.h"

namespace Legion {
  namespace Internal {

    /**
     * \class MustEpochOp
     * This operation is actually a meta-operation that
     * represents a collection of operations which all
     * must be guaranteed to be run in parallel.  It
     * mediates all the various stages of performing
     * these operations and ensures that they can all
     * be run in parallel or it reports an error.
     */
    class MustEpochOp : public Operation,
                        public MustEpoch,
                        public ResourceTracker {
    public:
      struct DependenceRecord {
      public:
        inline void add_entry(unsigned op_idx, unsigned req_idx)
        {
          op_indexes.emplace_back(op_idx);
          req_indexes.emplace_back(req_idx);
        }
      public:
        std::vector<unsigned> op_indexes;
        std::vector<unsigned> req_indexes;
      };
    public:
      MustEpochOp(void);
      MustEpochOp(const MustEpochOp& rhs) = delete;
      virtual ~MustEpochOp(void);
    public:
      MustEpochOp& operator=(const MustEpochOp& rhs) = delete;
    public:
      inline FutureMap get_future_map(void) const { return result_map; }
    public:
      // From MustEpoch
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
    public:
      FutureMap initialize(
          InnerContext* ctx, const MustEpochLauncher& launcher,
          Provenance* provenance);
      // Make this a virtual method so it can be overridden for
      // control replicated version of must epoch op
      virtual FutureMap create_future_map(
          TaskContext* ctx, IndexSpace domain, IndexSpace shard_space);
      // Another virtual method to override for control replication
      virtual void instantiate_tasks(
          InnerContext* ctx, const MustEpochLauncher& launcher);
      void find_conflicted_regions(std::vector<PhysicalRegion>& unmapped);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual const char* get_logging_name(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
    public:
      void verify_dependence(
          Operation* source_op, GenerationID source_gen, Operation* target_op,
          GenerationID target_gen);
      bool record_dependence(
          Operation* source_op, GenerationID source_gen, Operation* target_op,
          GenerationID target_gen, unsigned source_idx, unsigned target_idx,
          DependenceType dtype);
      bool record_intra_must_epoch_dependence(
          unsigned src_index, unsigned src_idx, unsigned dst_index,
          unsigned dst_idx, DependenceType dtype);
      void record_mapped_event(const DomainPoint& point, RtEvent mapped);
      void must_epoch_map_task_callback(
          SingleTask* task, Mapper::MapTaskInput& input,
          Mapper::MapTaskOutput& output);
      // Get a reference to our data structure for tracking acquired instances
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
    public:
      // Make this a virtual method to override it for control replication
      virtual MapperManager* invoke_mapper(void);
    public:
      // From ResourceTracker
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
          std::set<RtEvent>& preconditions) override;
    public:
      virtual uint64_t collective_lamport_allreduce(
          uint64_t lamport_clock, bool need_result);
    public:
      void rendezvous_concurrent_mapped(RtEvent precondition);
      virtual void finalize_concurrent_mapped(void);
    public:
      void concurrent_allreduce(
          IndividualTask* task, AddressSpaceID space, uint64_t lamport_clock,
          bool poisoned);
      void concurrent_allreduce(
          SliceTask* slice, AddressSpaceID source, size_t total_points,
          uint64_t lamport_clock, bool poisoned);
      virtual void finish_concurrent_allreduce(void);
    public:
      void register_single_task(SingleTask* single, unsigned index);
      void register_slice_task(SliceTask* slice);
    public:
      // Methods for keeping track of when we can complete and commit
      void register_subop(Operation* op);
      void notify_subop_complete(Operation* op, ApEvent effect);
      void notify_subop_commit(Operation* op, RtEvent precondition);
    public:
      RtUserEvent find_slice_versioning_event(UniqueID slice_id, bool& first);
      int find_operation_index(Operation* op, GenerationID generation);
      TaskOp* find_task_by_index(int index);
    protected:
      static bool single_task_sorter(const Task* t1, const Task* t2);
    protected:
      // Have a virtual function that we can override to for doing the
      // mapping and distribution of the point tasks, we'll override
      // this for control replication
      virtual RtEvent map_and_distribute(void);
    protected:
      IndexSpace compute_launch_space(
          const MustEpochLauncher& launcher, Provenance* provenance);
    protected:
      std::vector<IndividualTask*> indiv_tasks;
      std::vector<IndexTask*> index_tasks;
    protected:
      // The component slices for distribution
      std::set<SliceTask*> slice_tasks;
      // The actual base operations
      // Use a deque to keep everything in order
      std::vector<SingleTask*> single_tasks;
      std::atomic<unsigned> remaining_single_tasks;
      RtUserEvent single_tasks_ready;
      std::atomic<unsigned> remaining_mapped_events;
      std::map<DomainPoint, RtUserEvent> mapped_events;
    protected:
      Mapper::MapMustEpochInput input;
      Mapper::MapMustEpochOutput output;
    protected:
      size_t remaining_collective_unbound_points;
      uint64_t collective_lamport_clock;
      RtUserEvent collective_lamport_clock_ready;
    protected:
      // For the barrier before doing the lamport all-reduce
      RtUserEvent concurrent_mapped;
      size_t remaining_concurrent_mapped;
      std::vector<RtEvent> concurrent_preconditions;
    protected:
      // For doing the lamport clock all-reduce
      size_t remaining_concurrent_points;
      uint64_t concurrent_lamport_clock;
      bool concurrent_poisoned;
      std::vector<std::pair<IndividualTask*, AddressSpaceID> > concurrent_tasks;
      std::vector<std::pair<SliceTask*, AddressSpaceID> > concurrent_slices;
    protected:
      FutureMap result_map;
      unsigned remaining_resource_returns;
      unsigned remaining_subop_completes;
      unsigned remaining_subop_commits;
    protected:
      // Used for computing the constraints
      std::vector<std::set<SingleTask*> > task_sets;
      // Track the physical instances that we've acquired
      std::map<PhysicalManager*, unsigned> acquired_instances;
    protected:
      std::map<
          std::pair<unsigned /*task index*/, unsigned /*req index*/>,
          unsigned /*dependence index*/>
          dependence_map;
      std::vector<DependenceRecord*> dependences;
      std::map<
          std::pair<Operation*, GenerationID>,
          std::vector<std::pair<unsigned /*op idx*/, unsigned /*req idx*/> > >
          internal_dependences;
      std::map<SingleTask*, unsigned /*single task index*/> single_task_map;
      std::vector<std::set<unsigned /*single task index*/> >
          mapping_dependences;
    protected:
      std::map<UniqueID, RtUserEvent> slice_version_events;
    protected:
      std::set<RtEvent> commit_preconditions;
    };

    /**
     * \class MustEpochMappingBroadcast
     * A class for broadcasting the results of the mapping decisions
     * for a map must epoch call on a single node
     */
    class MustEpochMappingBroadcast : public BroadcastCollective {
    public:
      MustEpochMappingBroadcast(
          ReplicateContext* ctx, ShardID origin, CollectiveID collective_id);
      MustEpochMappingBroadcast(const MustEpochMappingBroadcast& rhs) = delete;
      virtual ~MustEpochMappingBroadcast(void);
    public:
      MustEpochMappingBroadcast& operator=(
          const MustEpochMappingBroadcast& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_BROADCAST;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    public:
      void broadcast(
          const std::vector<Processor>& processor_mapping,
          const std::vector<std::vector<Mapping::PhysicalInstance> >& mappings);
      void receive_results(
          std::vector<Processor>& processor_mapping,
          const std::vector<unsigned>& constraint_indexes,
          std::vector<std::vector<Mapping::PhysicalInstance> >& mappings,
          std::map<PhysicalManager*, unsigned>& acquired);
    protected:
      std::vector<Processor> processors;
      std::vector<std::vector<DistributedID> > instances;
    protected:
      RtUserEvent local_done_event;
      mutable std::set<RtEvent> done_events;
      std::set<PhysicalManager*> held_references;
    };

    /**
     * \class MustEpochMappingExchange
     * A class for exchanging the mapping decisions for
     * specific constraints for a must epoch launch
     */
    class MustEpochMappingExchange : public AllGatherCollective<false> {
    public:
      struct ConstraintInfo {
        std::vector<DistributedID> instances;
        ShardID origin_shard;
        int weight;
      };
    public:
      MustEpochMappingExchange(
          ReplicateContext* ctx, CollectiveID collective_id);
      MustEpochMappingExchange(const MustEpochMappingExchange& rhs) = delete;
      virtual ~MustEpochMappingExchange(void);
    public:
      MustEpochMappingExchange& operator=(const MustEpochMappingExchange& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_MUST_EPOCH_MAPPING_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange_must_epoch_mappings(
          ShardID shard_id, size_t total_shards, size_t total_constraints,
          const std::vector<const Task*>& local_tasks,
          const std::vector<const Task*>& all_tasks,
          std::vector<Processor>& processor_mapping,
          const std::vector<unsigned>& constraint_indexes,
          std::vector<std::vector<Mapping::PhysicalInstance> >& mappings,
          const std::vector<int>& mapping_weights,
          std::map<PhysicalManager*, unsigned>& acquired);
    protected:
      std::map<DomainPoint, Processor> processors;
      std::map<unsigned /*constraint index*/, ConstraintInfo> constraints;
    protected:
      RtUserEvent local_done_event;
      std::set<RtEvent> done_events;
      std::set<PhysicalManager*> held_references;
    };

    /**
     * \class MustEpochDependenceExchange
     * A class for exchanging the mapping dependence events for all
     * the single tasks in a must epoch launch so we can know which
     * order the point tasks are being mapped in.
     */
    class MustEpochDependenceExchange : public AllGatherCollective<false> {
    public:
      MustEpochDependenceExchange(
          CollectiveID id, ReplicateContext* ctx,
          std::map<DomainPoint, RtUserEvent>& mapped_events);
      MustEpochDependenceExchange(const MustEpochDependenceExchange& rhs) =
          delete;
      virtual ~MustEpochDependenceExchange(void);
    public:
      MustEpochDependenceExchange& operator=(
          const MustEpochDependenceExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_MUST_EPOCH_DEPENDENCE_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    protected:
      std::map<DomainPoint, RtUserEvent>& mapped_events;
    };

    /**
     * \class MustEpochCompletionExchange
     * A class for exchanging the local mapping and completion events
     * for all the tasks in a must epoch operation
     */
    class MustEpochCompletionExchange : public AllGatherCollective<false> {
    public:
      MustEpochCompletionExchange(
          CollectiveID id, ReplicateContext* ctx,
          std::vector<RtEvent>& local_mapped_events,
          std::vector<ApEvent>& local_completion_events);
      MustEpochCompletionExchange(const MustEpochCompletionExchange& rhs) =
          delete;
      virtual ~MustEpochCompletionExchange(void);
    public:
      MustEpochCompletionExchange& operator=(
          const MustEpochCompletionExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_MUST_EPOCH_COMPLETION_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      RtEvent finish_exchange(ReplMustEpochOp* op);
    protected:
      std::vector<RtEvent>& local_mapped_events;
      std::vector<ApEvent>& local_complete_events;
    };

    /**
     * \class ReplMustEpochOp
     * A must epoch operation that is aware that it is
     * being executed in a control replication context
     */
    class ReplMustEpochOp : public MustEpochOp {
    public:
      struct DeferMustEpochReturnResourcesArgs
        : public LgTaskArgs<DeferMustEpochReturnResourcesArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_MUST_EPOCH_RETURN_TASK_ID;
      public:
        DeferMustEpochReturnResourcesArgs(void) = default;
        DeferMustEpochReturnResourcesArgs(ReplMustEpochOp* o)
          : LgTaskArgs<DeferMustEpochReturnResourcesArgs>(false, false), op(o),
            done(Runtime::create_rt_user_event())
        { }
        void execute(void) const;
      public:
        ReplMustEpochOp* op;
        RtUserEvent done;
      };
    public:
      ReplMustEpochOp(void);
      ReplMustEpochOp(const ReplMustEpochOp& rhs) = delete;
      virtual ~ReplMustEpochOp(void);
    public:
      ReplMustEpochOp& operator=(const ReplMustEpochOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual FutureMap create_future_map(
          TaskContext* ctx, IndexSpace domain, IndexSpace shard_space) override;
      virtual void instantiate_tasks(
          InnerContext* ctx, const MustEpochLauncher& launcher) override;
      virtual MapperManager* invoke_mapper(void) override;
      virtual RtEvent map_and_distribute(void) override;
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
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
          std::set<RtEvent>& preconditions) override;
    public:
      virtual uint64_t collective_lamport_allreduce(
          uint64_t lamport_clock, bool need_result) override;
      virtual void finalize_concurrent_mapped(void) override;
      virtual void finish_concurrent_allreduce(void) override;
    public:
      void initialize_replication(ReplicateContext* ctx);
      Domain get_shard_domain(void) const;
    public:
      bool has_return_resources(void) const;
    protected:
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
      CollectiveID mapping_collective_id;
      bool collective_map_must_epoch_call;
      MustEpochMappingBroadcast* mapping_broadcast;
      MustEpochMappingExchange* mapping_exchange;
      CollectiveID collective_exchange_id;
      AllReduceCollective<MaxReduction<uint64_t>, false>* collective_exchange;
      ConcurrentAllreduce* concurrent_exchange;
      CollectiveID dependence_exchange_id;
      CollectiveID completion_exchange_id;
      std::set<SingleTask*> shard_single_tasks;
      RtBarrier concurrent_mapped_barrier;
      RtBarrier resource_return_barrier;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_MUST_EPOCH_H__
