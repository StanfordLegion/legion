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

#ifndef __LEGION_COPY_H__
#define __LEGION_COPY_H__

#include "legion/analysis/versioning.h"
#include "legion/api/functors_impl.h"
#include "legion/operations/pointwise.h"
#include "legion/operations/predicate.h"
#include "legion/operations/remote.h"
#include "legion/nodes/across.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalCopy
     * An extension of the external-facing Copy to help
     * with packing and unpacking them
     */
    class ExternalCopy : public Copy,
                         public ExternalMappable {
    public:
      ExternalCopy(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_copy(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_copy(Deserializer& derez);
    };

    /**
     * \class CopyOp
     * The copy operation provides a mechanism for applications
     * to directly copy data between pairs of fields possibly
     * from different region trees in an efficient way by
     * using the low-level runtime copy facilities.
     */
    class CopyOp : public ExternalCopy,
                   public PredicatedOp {
    public:
      enum ReqType {
        SRC_REQ = 0,
        DST_REQ = 1,
        GATHER_REQ = 2,
        SCATTER_REQ = 3,
        REQ_COUNT = 4,
      };
    public:
      struct DeferredCopyAcross : public LgTaskArgs<DeferredCopyAcross> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFERRED_COPY_ACROSS_TASK_ID;
      public:
        DeferredCopyAcross(void) = default;
        DeferredCopyAcross(
            CopyOp* op, const PhysicalTraceInfo& info, unsigned idx,
            ApEvent init, ApEvent sready, ApEvent dready, ApEvent gready,
            ApEvent cready, ApUserEvent local_pre, ApUserEvent local_post,
            ApEvent collective_pre, ApEvent collective_post, PredEvent g,
            RtUserEvent a, InstanceSet* src, InstanceSet* dst,
            InstanceSet* gather, InstanceSet* scatter, const bool preimages,
            const bool shadow)
          : LgTaskArgs<DeferredCopyAcross>(false, false), copy(op),
            trace_info(new PhysicalTraceInfo(info)), index(idx),
            init_precondition(init), src_ready(sready), dst_ready(dready),
            gather_ready(gready), scatter_ready(cready),
            local_precondition(local_pre), local_postcondition(local_post),
            collective_precondition(collective_pre),
            collective_postcondition(collective_post), guard(g), applied(a),
            src_targets(src), dst_targets(dst), gather_targets(gather),
            scatter_targets(scatter), compute_preimages(preimages),
            shadow_indirections(shadow)
        { }
        void execute(void) const;
      public:
        CopyOp* copy;
        PhysicalTraceInfo* trace_info;
        unsigned index;
        ApEvent init_precondition;
        ApEvent src_ready;
        ApEvent dst_ready;
        ApEvent gather_ready;
        ApEvent scatter_ready;
        ApUserEvent local_precondition;
        ApUserEvent local_postcondition;
        ApEvent collective_precondition;
        ApEvent collective_postcondition;
        PredEvent guard;
        RtUserEvent applied;
        InstanceSet* src_targets;
        InstanceSet* dst_targets;
        InstanceSet* gather_targets;
        InstanceSet* scatter_targets;
        bool compute_preimages;
        bool shadow_indirections;
      };
    public:
      CopyOp(void);
      CopyOp(const CopyOp& rhs) = delete;
      virtual ~CopyOp(void);
    public:
      CopyOp& operator=(const CopyOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const CopyLauncher& launcher,
          Provenance* provenance);
      void log_copy_requirements(void) const;
      void perform_base_dependence_analysis(bool permit_projection);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual Mappable* get_mappable(void) override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_complete(ApEvent complete) override;
      virtual void trigger_commit(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2) override;
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent& collective_pre,
          ApEvent& collective_post, const TraceInfo& trace_info,
          const InstanceSet& instances, const RegionRequirement& req,
          std::vector<IndirectRecord>& records, const bool sources);
    public:
      virtual void predicate_false(void) override;
    public:
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
      virtual void update_atomic_locks(
          const unsigned index, Reservation lock, bool exclusive) override;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
    protected:
      void perform_type_checking(void) const;
      void perform_copy_across(
          const unsigned index, const ApEvent init_precondition,
          const ApEvent src_ready, const ApEvent dst_ready,
          const ApEvent gather_ready, const ApEvent scatter_ready,
          const ApUserEvent local_precondition,
          const ApUserEvent local_postcondition,
          const ApEvent collective_precondition,
          const ApEvent collective_postcondition,
          const PredEvent predication_guard, const InstanceSet& src_targets,
          const InstanceSet& dst_targets, const InstanceSet* gather_targets,
          const InstanceSet* scatter_targets,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& applied_conditions, const bool compute_preimages,
          const bool shadow_indirections);
      void finalize_copy_profiling(void);
    protected:
      static void req_vector_reduce_to_readwrite(
          std::vector<RegionRequirement>& reqs,
          std::vector<unsigned>& changed_idxs);
      static void req_vector_reduce_restore(
          std::vector<RegionRequirement>& reqs,
          const std::vector<unsigned>& changed_idxs);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      // From Memoizable
      virtual void complete_replay(ApEvent copy_complete_event) override;
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(
          unsigned idx) const override;
    protected:
      unsigned get_requirement_offset(unsigned idx) const;
      const char* get_requirement_name(unsigned idx) const;
      template<ReqType REQ_TYPE>
      static const char* get_req_type_name(void);
      template<ReqType REQ_TYPE>
      int perform_conversion(
          unsigned idx, const RegionRequirement& req,
          std::vector<MappingInstance>& output,
          std::vector<MappingInstance>& input,
          std::vector<PhysicalManager*>& sources, InstanceSet& targets,
          bool is_reduce = false);
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc) override;
      virtual void handle_profiling_update(int count) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      // Separate function for this so it can be called by derived classes
      RtEvent perform_local_versioning_analysis(void);
    protected:
      ApEvent copy_across(
          const RegionRequirement& src_req, const RegionRequirement& dst_req,
          const VersionInfo& src_version_info,
          const VersionInfo& dst_version_info, const InstanceSet& src_targets,
          const InstanceSet& dst_targets,
          const std::vector<PhysicalManager*>& sources, unsigned src_index,
          unsigned dst_index, ApEvent precondition, ApEvent src_ready,
          ApEvent dst_ready, PredEvent pred_guard,
          const std::map<Reservation, bool>& reservations,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events);
      ApEvent gather_across(
          const RegionRequirement& src_req, const RegionRequirement& idx_req,
          const RegionRequirement& dst_req,
          std::vector<IndirectRecord>& records, const InstanceSet& src_targets,
          const InstanceSet& idx_targets, const InstanceSet& dst_targets,
          unsigned src_index, unsigned idx_index, unsigned dst_index,
          const bool gather_is_range, const ApEvent init_precondition,
          const ApEvent src_ready, const ApEvent dst_ready,
          const ApEvent idx_ready, const PredEvent pred_guard,
          const ApEvent collective_precondition,
          const ApEvent collective_postcondition,
          const ApUserEvent local_precondition,
          const std::map<Reservation, bool>& reservations,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events,
          const bool possible_src_out_of_range, const bool compute_preimages,
          const bool shadow_indirections);
      ApEvent scatter_across(
          const RegionRequirement& src_req, const RegionRequirement& idx_req,
          const RegionRequirement& dst_req, const InstanceSet& src_targets,
          const InstanceSet& idx_targets, const InstanceSet& dst_targets,
          std::vector<IndirectRecord>& records, unsigned src_index,
          unsigned idx_index, unsigned dst_index, const bool scatter_is_range,
          const ApEvent init_precondition, const ApEvent src_ready,
          const ApEvent dst_ready, const ApEvent idx_ready,
          const PredEvent pred_guard, const ApEvent collective_precondition,
          const ApEvent collective_postcondition,
          const ApUserEvent local_precondition,
          const std::map<Reservation, bool>& reservations,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events,
          const bool possible_dst_out_of_range,
          const bool possible_dst_aliasing, const bool compute_preimages,
          const bool shadow_indirections);
      ApEvent indirect_across(
          const RegionRequirement& src_req,
          const RegionRequirement& src_idx_req,
          const RegionRequirement& dst_req,
          const RegionRequirement& dst_idx_req, const InstanceSet& src_targets,
          const InstanceSet& dst_targets,
          std::vector<IndirectRecord>& src_records,
          const InstanceSet& src_idx_target,
          std::vector<IndirectRecord>& dst_records,
          const InstanceSet& dst_idx_target, unsigned src_index,
          unsigned dst_index, unsigned src_idx_index, unsigned dst_idx_index,
          const bool both_are_range, const ApEvent init_precondition,
          const ApEvent src_ready, const ApEvent dst_ready,
          const ApEvent src_idx_ready, const ApEvent dst_idx_ready,
          const PredEvent pred_guard, const ApEvent collective_precondition,
          const ApEvent collective_postcondition,
          const ApUserEvent local_precondition,
          const std::map<Reservation, bool>& reservations,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events,
          const bool possible_src_out_of_range,
          const bool possible_dst_out_of_range,
          const bool possible_dst_aliasing, const bool compute_preimages,
          const bool shadow_indirections);
    protected:
      template<typename T>
      void initialize_copy_from_launcher(const T& launcher);
    protected:
      std::vector<unsigned> src_parent_indexes;
      std::vector<unsigned> dst_parent_indexes;
      op::vector<VersionInfo> src_versions;
      op::vector<VersionInfo> dst_versions;
    public:  // These are only used for indirect copies
      std::vector<unsigned> gather_parent_indexes;
      std::vector<unsigned> scatter_parent_indexes;
      std::vector<bool> gather_is_range;
      std::vector<bool> scatter_is_range;
      op::vector<VersionInfo> gather_versions;
      op::vector<VersionInfo> scatter_versions;
      std::vector<std::vector<IndirectRecord> > src_indirect_records;
      std::vector<std::vector<IndirectRecord> > dst_indirect_records;
      std::vector<std::map<Reservation, bool> > atomic_locks;
    protected:  // for support with mapping
      MapperManager* mapper;
    protected:
      std::vector<PhysicalManager*> across_sources;
      std::map<PhysicalManager*, unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    public:
      bool possible_src_indirect_out_of_range;
      bool possible_dst_indirect_out_of_range;
      bool possible_dst_indirect_aliasing;
    };

    /**
     * \class IndexCopyOp
     * An index copy operation is the same as a copy operation
     * except it is an index space operation for performing
     * multiple copies with projection functions
     */
    class IndexCopyOp : public PointwiseAnalyzable<CopyOp> {
    public:
      IndexCopyOp(void);
      IndexCopyOp(const IndexCopyOp& rhs) = delete;
      virtual ~IndexCopyOp(void);
    public:
      IndexCopyOp& operator=(const IndexCopyOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const IndexCopyLauncher& launcher,
          IndexSpace launch_space, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
      virtual void predicate_false(void) override;
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2) override;
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent& collective_pre,
          ApEvent& collective_post, const TraceInfo& trace_info,
          const InstanceSet& instances, const RegionRequirement& req,
          std::vector<IndirectRecord>& records, const bool sources) override;
      virtual RtEvent finalize_exchange(
          const unsigned index, const bool source);
    public:
      virtual RtEvent find_intra_space_dependence(const DomainPoint& point);
      virtual bool is_pointwise_analyzable(void) const override;
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      virtual size_t get_collective_points(void) const override;
    public:
      virtual IndexSpaceNode* get_shard_points(void) const
      {
        return launch_space;
      }
      void enumerate_points(void);
      void handle_point_complete(ApEvent effects);
      void handle_point_commit(RtEvent point_committed);
      void start_check_point_requirements(void);
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              point_domains);
      void log_index_copy_requirements(void);
    public:
      IndexSpaceNode* launch_space;
    protected:
      std::vector<PointCopyOp*> points;
      struct IndirectionExchange {
        std::set<ApEvent> local_preconditions;
        std::set<ApEvent> local_postconditions;
        std::vector<std::vector<IndirectRecord>*> src_records;
        std::vector<std::vector<IndirectRecord>*> dst_records;
        ApUserEvent collective_pre;
        ApUserEvent collective_post;
        RtUserEvent src_ready;
        RtUserEvent dst_ready;
      };
      std::vector<IndirectionExchange> collective_exchanges;
      std::atomic<unsigned> points_completed;
      unsigned points_committed;
      bool collective_src_indirect_points;
      bool collective_dst_indirect_points;
      bool commit_request;
      std::set<RtEvent> commit_preconditions;
    protected:
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned, unsigned> > interfering_requirements;
      std::map<DomainPoint, RtUserEvent> pending_pointwise_dependences;
    };

    /**
     * \class PointCopyOp
     * A point copy operation is used for executing the
     * physical part of the analysis for an index copy
     * operation.
     */
    class PointCopyOp : public CopyOp,
                        public ProjectionPoint {
    public:
      friend class IndexCopyOp;
      PointCopyOp(void);
      PointCopyOp(const PointCopyOp& rhs) = delete;
      virtual ~PointCopyOp(void);
    public:
      PointCopyOp& operator=(const PointCopyOp& rhs) = delete;
    public:
      void initialize(IndexCopyOp* owner, const DomainPoint& point);
      void launch(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
      // trigger_mapping same as base class
      virtual void trigger_complete(ApEvent effects) override;
      virtual void trigger_commit(void) override;
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent& collective_pre,
          ApEvent& collective_post, const TraceInfo& trace_info,
          const InstanceSet& instances, const RegionRequirement& req,
          std::vector<IndirectRecord>& records, const bool sources) override;
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
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const override;
    protected:
      IndexCopyOp* owner;
      std::vector<RtEvent> pointwise_mapping_dependences;
    };

    /**
     * \class IndirectRecordExchange
     * A class for doing an all-gather of indirect records for
     * doing gather/scatter/full-indirect copy operations.
     */
    class IndirectRecordExchange : public AllGatherCollective<true> {
    public:
      IndirectRecordExchange(ReplicateContext* ctx, CollectiveID id);
      IndirectRecordExchange(const IndirectRecordExchange& rhs) = delete;
      virtual ~IndirectRecordExchange(void);
    public:
      IndirectRecordExchange& operator=(const IndirectRecordExchange& rhs) =
          delete;
    public:
      RtEvent exchange_records(
          std::vector<std::vector<IndirectRecord>*>& targets,
          std::vector<IndirectRecord>& local_records);
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_INDIRECT_COPY_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    protected:
      std::vector<std::vector<IndirectRecord>*> local_targets;
      std::vector<IndirectRecord> all_records;
    };

    /**
     * \class ReplCopyOp
     * A copy operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplCopyOp : public CopyOp {
    public:
      ReplCopyOp(void);
      ReplCopyOp(const ReplCopyOp& rhs) = delete;
      virtual ~ReplCopyOp(void);
    public:
      ReplCopyOp& operator=(const ReplCopyOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* ctx);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
    protected:
      IndexSpaceNode* launch_space;
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

    /**
     * \class ReplIndexCopyOp
     * An index fill operation that is aware that it is
     * being executed in a control replication context.
     */
    class ReplIndexCopyOp : public IndexCopyOp {
    public:
      ReplIndexCopyOp(void);
      ReplIndexCopyOp(const ReplIndexCopyOp& rhs) = delete;
      virtual ~ReplIndexCopyOp(void);
    public:
      ReplIndexCopyOp& operator=(const ReplIndexCopyOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
      virtual IndexSpaceNode* get_shard_points(void) const override
      {
        return shard_points;
      }
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    protected:
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent& collective_pre,
          ApEvent& collective_post, const TraceInfo& trace_info,
          const InstanceSet& instances, const RegionRequirement& req,
          std::vector<IndirectRecord>& records, const bool sources) override;
      virtual RtEvent finalize_exchange(
          const unsigned index, const bool source) override;
    public:
      virtual RtEvent find_intra_space_dependence(
          const DomainPoint& point) override;
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              point_domains) override;
    public:
      void initialize_replication(ReplicateContext* ctx);
    protected:
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
      IndexSpaceNode* shard_points;
      std::vector<ApBarrier> pre_indirection_barriers;
      std::vector<ApBarrier> post_indirection_barriers;
      std::vector<IndirectRecordExchange*> src_collectives;
      std::vector<IndirectRecordExchange*> dst_collectives;
      std::set<std::pair<DomainPoint, ShardID> > unique_intra_space_deps;
      CollectiveID interfering_check_id;
      InterferingPointExchange<ReplIndexCopyOp>* interfering_exchange;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

    /**
     * \class RemoteCopyOp
     * This is a remote copy of a CopyOp to be used
     * for mapper calls and other operations
     */
    class RemoteCopyOp : public ExternalCopy,
                         public RemoteOp,
                         public Heapify<RemoteCopyOp, OPERATION_LIFETIME> {
    public:
      RemoteCopyOp(Operation* ptr, AddressSpaceID src);
      RemoteCopyOp(const RemoteCopyOp& rhs) = delete;
      virtual ~RemoteCopyOp(void);
    public:
      RemoteCopyOp& operator=(const RemoteCopyOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
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
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const CopyOp& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_logging_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COPY_H__
