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

#ifndef __LEGION_INDEX_TASK_H__
#define __LEGION_INDEX_TASK_H__

#include "legion/tasks/multi.h"
#include "legion/api/redop.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class IndexTask
     * An index task is used to represent an index space task
     * launch performed by the runtime.  It will only live
     * on the node on which it was created.  Eventually the
     * mapper will slice the index space, and the corresponding
     * slice tasks for the index space will be distributed around
     * the machine and eventually returned to this index space task.
     */
    class IndexTask : public MultiTask,
                      public Heapify<IndexTask, RUNTIME_LIFETIME> {
    public:
      struct OutputRegionState {
        std::map<DomainPoint, DomainPoint> points;
        // Indexed by [dim][color[dim]]
        std::vector<std::vector<coord_t> > extents;
        std::vector<std::vector<coord_t> > offsets;
        // for control replication
        std::vector<std::set<DomainPoint> > dim_pending_points;
        std::map<DomainPoint, unsigned> pending_points;
        // Only valid for global indexing requirements
        Domain global_color_space;
        ProjectionFunction* projection = nullptr;
        bool has_hi_point = false;
      };
    private:
      struct OutputRegionTagCreator {
      public:
        OutputRegionTagCreator(TypeTag* _type_tag, int _color_ndim)
          : type_tag(_type_tag), color_ndim(_color_ndim)
        { }
        template<typename DIM, typename COLOR_T>
        static inline void demux(OutputRegionTagCreator* creator)
        {
          switch (DIM::N + creator->color_ndim)
          {
#define DIMFUNC(DIM)                                                      \
  case DIM:                                                               \
    {                                                                     \
      *creator->type_tag = NT_TemplateHelper::encode_tag<DIM, COLOR_T>(); \
      break;                                                              \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
        }
      private:
        TypeTag* type_tag;
        int color_ndim;
      };
    public:
      IndexTask(void);
      IndexTask(const IndexTask& rhs) = delete;
      virtual ~IndexTask(void);
    public:
      IndexTask& operator=(const IndexTask& rhs) = delete;
    public:
      FutureMap initialize_task(
          InnerContext* ctx, const IndexTaskLauncher& launcher,
          IndexSpace launch_space, Provenance* provenance, bool track,
          std::vector<OutputRequirement>* outputs = nullptr);
      Future initialize_task(
          InnerContext* ctx, const IndexTaskLauncher& launcher,
          IndexSpace launch_space, Provenance* provenance, ReductionOpID redop,
          bool deterministic, bool track,
          std::vector<OutputRequirement>* outputs = nullptr);
      void initialize_regions(const std::vector<RegionRequirement>& regions);
      void initialize_predicate(
          const Future& pred_future, const UntypedBuffer& pred_arg);
      void perform_base_dependence_analysis(void);
    protected:
      void create_output_regions(
          std::vector<OutputRequirement>& outputs, IndexSpace launch_space);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void prepare_map_must_epoch(void);
    public:
      // Virtual for overload with control replication
      virtual void record_output_extent(
          unsigned index, const DomainPoint& color, const DomainPoint& extent);
      virtual void validate_output_extents(unsigned index);
      virtual void record_output_registered(RtEvent registered);
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    public:
      virtual void trigger_ready(void) override;
      virtual void predicate_false(void) override;
      virtual void premap_task(void) override;
      virtual bool distribute_task(void) override;
      virtual bool perform_mapping(
          MustEpochOp* owner = nullptr,
          const DeferMappingArgs* args = nullptr) override;
      virtual void launch_task(bool inline_task = false) override;
      virtual bool is_stealable(void) const override;
      virtual void trigger_complete(ApEvent effects) override;
    public:
      virtual TaskKind get_task_kind(void) const override;
    protected:
      virtual void trigger_task_commit(void) override;
    public:
      virtual bool pack_task(Serializer& rez, AddressSpaceID target) override;
      virtual bool unpack_task(
          Deserializer& derez, Processor current, AddressSpaceID source,
          std::set<RtEvent>& ready_events) override;
      virtual void perform_inlining(
          VariantImpl* variant,
          const std::deque<InstanceSet>& parent_regions) override;
    public:
      virtual SliceTask* clone_as_slice_task(
          IndexSpace is, Processor p, bool recurse, bool stealable) override;
    public:
      virtual void reduce_future(
          const DomainPoint& point, FutureInstance* instance,
          ApEvent effects) override;
    public:
      virtual void pack_profiling_requests(
          Serializer& rez, std::set<RtEvent>& applied) const override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
      virtual void handle_profiling_update(int count) override;
    public:
      virtual void register_must_epoch(void) override;
    public:
      virtual size_t get_collective_points(void) const override;
    public:
      // Make this a virtual method so for control replication we can
      // create a different type of future map for the task
      virtual FutureMap create_future_map(
          TaskContext* ctx, IndexSpace launch_space, IndexSpace shard_space);
      void rendezvous_concurrent_mapped(
          const DomainPoint& point, Processor target, Color color,
          RtEvent precondition);
      void rendezvous_concurrent_mapped(Deserializer& derez);
      // Also virtual for control replication override
      virtual void finalize_concurrent_mapped(void);
      virtual void initialize_concurrent_group(
          Color color, size_t local, size_t global, RtBarrier barrier,
          const std::vector<ShardID>& shards);
      virtual void concurrent_allreduce(
          Color color, SliceTask* slice, AddressSpaceID slice_space,
          size_t points, uint64_t lamport_clock, VariantID vid, bool poisoned);
      virtual uint64_t collective_lamport_allreduce(
          uint64_t lamport_clock, size_t points, bool need_result);
    public:
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_region =
              std::optional<unsigned>()) override;
    public:
      void record_origin_mapped_slice(SliceTask* local_slice);
      void initialize_must_epoch_concurrent_group(
          Color color, RtUserEvent precondition);
    protected:
      // Virtual so can be overridden by ReplIndexTask
      virtual void create_future_instances(std::vector<Memory>& target_mems);
      // Callback for control replication to perform reduction for sizes
      // and provide an event for when the result is ready
      virtual void finish_index_task_reduction(void);
    public:
      void return_point_mapped(const DomainPoint& point, RtEvent mapped);
      void return_slice_complete(
          unsigned points, ApEvent effects, void* metadata = nullptr,
          size_t metasize = 0);
      void return_slice_commit(unsigned points, RtEvent applied_condition);
    public:
      void unpack_point_mapped(Deserializer& derez, AddressSpaceID source);
      void unpack_slice_complete(Deserializer& derez);
      void unpack_slice_commit(Deserializer& derez);
      void unpack_slice_collective_versioning_rendezvous(
          Deserializer& derez, unsigned index, size_t total_points);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      void enumerate_futures(const Domain& domain);
      void start_check_point_requirements(void);
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              domain_points);
    protected:
      friend class SliceTask;
      Future reduction_future;
      std::optional<size_t> reduction_future_size;
      unsigned total_points;
      unsigned mapped_points;
      unsigned completed_points;
      unsigned committed_points;
    protected:
      std::vector<SliceTask*> origin_mapped_slices;
      std::vector<FutureInstance*> reduction_instances;
      std::vector<Memory> serdez_redop_targets;
    protected:
      std::set<RtEvent> map_applied_conditions;
      std::vector<RtEvent> output_preconditions;
      std::set<RtEvent> commit_preconditions;
    protected:
      std::map<DomainPoint, RtUserEvent> pending_pointwise_dependences;
      // Note the DomainPoints here are colors and not index points!
      std::vector<std::map<DomainPoint, RtEvent> > output_pointwise_dependences;
      std::vector<OutputRegionState> output_region_states;
    protected:
      std::vector<ProfilingMeasurementID> task_profiling_requests;
      std::vector<ProfilingMeasurementID> copy_profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    protected:
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned, unsigned> > interfering_requirements;
      std::map<DomainPoint, std::vector<LogicalRegion> > point_requirements;
    };

    /**
     * \class OutputExtentExchange
     * This class exchanges sizes of output subregions that are globally
     * indexed.
     */
    class OutputExtentExchange : public AllGatherCollective<false> {
    public:
      typedef IndexTask::OutputRegionState OutputRegionState;
    public:
      OutputExtentExchange(
          ReplicateContext* ctx, ReplIndexTask* owner,
          CollectiveIndexLocation loc,
          std::vector<OutputRegionState>& all_output_states);
      OutputExtentExchange(const OutputExtentExchange& rhs) = delete;
      virtual ~OutputExtentExchange(void);
    public:
      OutputExtentExchange& operator=(const OutputExtentExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_OUTPUT_SIZE_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    public:
      ReplIndexTask* const owner;
      std::vector<OutputRegionState>& all_output_states;
    };

    /**
     * \class ConcurrentMappingRendezvous
     * This collective helps to validate the safety of the execution of
     * concurrent index space task launches to ensure that all the point
     * tasks have been mapped to different processors.
     */
    class ConcurrentMappingRendezvous : public AllGatherCollective<true> {
    public:
      ConcurrentMappingRendezvous(
          ReplIndexTask* owner, CollectiveIndexLocation loc,
          ReplicateContext* ctx,
          std::map<Color, MultiTask::ConcurrentGroup>& groups);
      virtual ~ConcurrentMappingRendezvous(void) { }
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CONCURRENT_MAPPING_RENDEZVOUS;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    public:
      void set_trace_barrier(Color color, RtBarrier barrier, size_t arrivals);
      void perform_rendezvous(void);
    public:
      ReplIndexTask* const owner;
    protected:
      std::map<Color, MultiTask::ConcurrentGroup>& groups;
      std::map<Color, std::pair<RtBarrier, size_t> > trace_barriers;
    };

    /**
     * \class ConcurrentAllreduce
     * This class helps with the process of performing an all-reduce between
     * the shards of a concurrent index space task launch to get their
     * maximum lamport clock and whether any inputs were poisoned
     * Do this in order so we can count the total points we've seen and
     * send out the results as soon as possible in the case that we can
     * short-circuit the result when all points come from just a subset
     * of the shards.
     */
    class ConcurrentAllreduce : public AllGatherCollective<false> {
    public:
      // For ReplIndexTask
      ConcurrentAllreduce(
          ReplicateContext* ctx, CollectiveID id, Color color,
          const std::vector<ShardID>& shards);
      // For ReplMustEpochOp
      ConcurrentAllreduce(CollectiveIndexLocation loc, ReplicateContext* ctx);
      ConcurrentAllreduce(const ConcurrentAllreduce& rhs) = delete;
      virtual ~ConcurrentAllreduce(void);
    public:
      ConcurrentAllreduce& operator=(const ConcurrentAllreduce& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CONCURRENT_ALLREDUCE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      // for ReplIndextask
      void perform_concurrent_allreduce(MultiTask::ConcurrentGroup& group);
      // for ReplMustEpochOp
      void perform_concurrent_allreduce(
          std::vector<std::pair<IndividualTask*, AddressSpaceID> >&
              single_tasks,
          std::vector<std::pair<SliceTask*, AddressSpace> >& slice_tasks,
          uint64_t lamport_clock, bool poisoned);
    protected:
      virtual RtEvent post_complete_exchange(void) override;
    protected:
      const Color color;
      std::vector<std::pair<IndividualTask*, AddressSpaceID> > single_tasks;
      std::vector<std::pair<SliceTask*, AddressSpace> > slice_tasks;
      RtBarrier task_barrier;
      uint64_t lamport_clock;
      VariantID variant;
      bool poisoned;
    };

    /**
     * \class ReplIndexTask
     * An individual task that is aware that it is
     * being executed in a control replication context.
     */
    class ReplIndexTask : public ReplCollectiveViewCreator<IndexTask> {
    public:
      ReplIndexTask(void);
      ReplIndexTask(const ReplIndexTask& rhs) = delete;
      virtual ~ReplIndexTask(void);
    public:
      ReplIndexTask& operator=(const ReplIndexTask& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
    protected:
      virtual void create_future_instances(
          std::vector<Memory>& target_mems) override;
      virtual void finish_index_task_reduction(void) override;
    public:
      // Have to override this too for doing output in the
      // case that we misspeculate
      virtual void predicate_false(void) override;
      virtual void prepare_map_must_epoch(void) override;
    public:
      void initialize_replication(ReplicateContext* ctx);
      void set_sharding_function(
          ShardingID functor, ShardingFunction* function);
      virtual FutureMap create_future_map(
          TaskContext* ctx, IndexSpace launch_space,
          IndexSpace shard_space) override;
      virtual void finalize_concurrent_mapped(void) override;
      void finish_concurrent_mapped(
          const std::map<Color, std::pair<RtBarrier, size_t> >& trace_barriers);
      virtual void initialize_concurrent_group(
          Color color, size_t local, size_t global, RtBarrier barrier,
          const std::vector<ShardID>& shards) override;
      virtual void concurrent_allreduce(
          Color color, SliceTask* slice, AddressSpaceID slice_space,
          size_t points, uint64_t lamport_clock, VariantID vid,
          bool poisoned) override;
      virtual uint64_t collective_lamport_allreduce(
          uint64_t lamport_clock, size_t points, bool need_result) override;
      void select_sharding_function(ReplicateContext* repl_ctx);
      void finalize_output_regions(bool first_invocation);
    public:
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_region =
              std::optional<unsigned>()) override;
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              domain_points) override;
    public:
      // Output regions
      virtual void record_output_extent(
          unsigned index, const DomainPoint& color,
          const DomainPoint& extent) override;
      virtual void validate_output_extents(unsigned index) override;
      virtual void record_output_registered(RtEvent registered) override;
    public:
      virtual size_t get_collective_points(void) const override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    public:
      void record_output_offset(
          unsigned index, unsigned dim, size_t color_index, size_t offset);
    protected:
      void compute_output_extent_shards(
          unsigned index, unsigned dim, size_t color_index,
          const Domain& color_space, ProjectionFunction* projection,
          std::set<ShardID>& shards) const;
      void send_output_offset_messages(
          unsigned index, unsigned dim, size_t color_index, size_t offset,
          const std::set<ShardID>& previous_shards,
          const std::set<ShardID>& next_shards) const;
    protected:
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
      BufferExchange* serdez_redop_collective;
      FutureAllReduceCollective* all_reduce_collective;
      FutureReductionCollective* reduction_collective;
      FutureBroadcastCollective* broadcast_collective;
      OutputExtentExchange* output_size_collective;
      CollectiveID collective_check_id;
      CollectiveID interfering_check_id;
      InterferingPointExchange<ReplIndexTask>* interfering_exchange;
      RtBarrier output_bar;
      std::map<Color, CollectiveID> concurrent_exchange_ids;
    protected:
      std::set<std::pair<DomainPoint, ShardID> > unique_intra_space_deps;
    protected:
      // For setting up concurrent execution
      ConcurrentMappingRendezvous* concurrent_mapping_rendezvous;
      CollectiveID collective_exchange_id;
      AllReduceCollective<MaxReduction<uint64_t>, false>* collective_exchange;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    protected:
      bool slice_sharding_output;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INDEX_TASK_H__
