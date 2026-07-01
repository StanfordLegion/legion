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

#ifndef __LEGION_FILL_H__
#define __LEGION_FILL_H__

#include "legion/analysis/versioning.h"
#include "legion/api/functors_impl.h"
#include "legion/operations/collective.h"
#include "legion/operations/pointwise.h"
#include "legion/operations/predicate.h"
#include "legion/operations/remote.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalFill
     * An extension of the external-facing Fill to help
     * with packing and unpacking them
     */
    class ExternalFill : public Fill,
                         public ExternalMappable {
    public:
      ExternalFill(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_fill(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_fill(Deserializer& derez);
    };

    /**
     * \class FillOp
     * Fill operations are used to initialize a field to a
     * specific value for a particular logical region.
     */
    class FillOp : public PredicatedOp,
                   public ExternalFill {
    public:
      FillOp(void);
      FillOp(const FillOp& rhs) = delete;
      virtual ~FillOp(void);
    public:
      FillOp& operator=(const FillOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const FillLauncher& launcher,
          Provenance* provenance);
      void perform_base_dependence_analysis(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual Mappable* get_mappable(void) override;
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void) override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual FillView* get_fill_view(void) const;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_complete(ApEvent effects_done) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    public:
      // This is a helper method for ReplFillOp
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
    public:
      virtual void predicate_false(void) override;
    public:
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void trigger_commit(void) override;
    public:
      void log_fill_requirement(void) const;
    public:
      // From Memoizable
      virtual const VersionInfo& get_version_info(unsigned idx) const
      {
        return version_info;
      }
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
      virtual void complete_replay(ApEvent fill_complete_event) override;
    public:
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    protected:
      void fill_fields(
          FillView* fill_view, ApEvent precondition,
          const PhysicalTraceInfo& trace_info);
    public:
      VersionInfo version_info;
      unsigned parent_req_index;
      RtEvent fill_view_ready;
      FillView* fill_view;
      Future future;
      void* value;
      size_t value_size;
      bool set_view;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class IndexFillOp
     * This is the same as a fill operation except for
     * applying a number of fill operations over an
     * index space of points with projection functions.
     */
    class IndexFillOp : public PointwiseAnalyzable<FillOp> {
    public:
      IndexFillOp(void);
      IndexFillOp(const IndexFillOp& rhs) = delete;
      virtual ~IndexFillOp(void);
    public:
      IndexFillOp& operator=(const IndexFillOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, const IndexFillLauncher& launcher,
          IndexSpace launch_space, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    protected:
      void activate_index_fill(void);
      void deactivate_index_fill(void);
    public:
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_commit(void) override;
      virtual void predicate_false(void) override;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      virtual size_t get_collective_points(void) const override;
      virtual IndexSpaceNode* get_shard_points(void) const
      {
        return launch_space;
      }
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
      void enumerate_points(void);
      void handle_point_complete(ApEvent effect);
      void handle_point_commit(void);
    protected:
      void log_index_fill_requirement(void);
    public:
      IndexSpaceNode* launch_space;
    protected:
      std::vector<PointFillOp*> points;
      std::map<DomainPoint, RtUserEvent> pending_pointwise_dependences;
      std::atomic<unsigned> points_completed;
      unsigned points_committed;
      bool commit_request;
    };

    /**
     * \class PointFillOp
     * A point fill op is used for executing the
     * physical part of the analysis for an index
     * fill operation.
     */
    class PointFillOp : public FillOp,
                        public ProjectionPoint {
    public:
      PointFillOp(void);
      PointFillOp(const PointFillOp& rhs) = delete;
      virtual ~PointFillOp(void);
    public:
      PointFillOp& operator=(const PointFillOp& rhs) = delete;
    public:
      void initialize(IndexFillOp* owner, const DomainPoint& point);
      void launch(RtEvent view_ready);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
      // trigger_mapping same as base class
      virtual void trigger_complete(ApEvent effect) override;
      virtual void trigger_commit(void) override;
      virtual FillView* get_fill_view(void) const override;
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
      IndexFillOp* owner;
      std::vector<RtEvent> pointwise_mapping_dependences;
    };

    /**
     * \class ReplFillOp
     * A copy operation that is aware that it is being
     * executed in a control replication context.
     */
    class ReplFillOp
      : public ReplCollectiveVersioning<CollectiveVersioning<FillOp> > {
    public:
      ReplFillOp(void);
      ReplFillOp(const ReplFillOp& rhs) = delete;
      virtual ~ReplFillOp(void);
    public:
      ReplFillOp& operator=(const ReplFillOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* ctx, bool is_first_local);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_replay(void) override;
      virtual bool is_collective_first_local_shard(void) const
      {
        return is_first_local_shard;
      }
      virtual RtEvent finalize_complete_mapping(RtEvent event) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      virtual void predicate_false(void) override;
    public:
      RtBarrier collective_map_barrier;
      bool is_first_local_shard;
    };

    /**
     * \class ReplIndexFillOp
     * An index fill operation that is aware that it is
     * being executed in a control replication context.
     */
    class ReplIndexFillOp : public IndexFillOp {
    public:
      ReplIndexFillOp(void);
      ReplIndexFillOp(const ReplIndexFillOp& rhs) = delete;
      virtual ~ReplIndexFillOp(void);
    public:
      ReplIndexFillOp& operator=(const ReplIndexFillOp& rhs) = delete;
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
      ShardingID sharding_functor;
      ShardingFunction* sharding_function;
      IndexSpaceNode* shard_points;
      MapperManager* mapper;
    public:
      inline void set_sharding_collective(ShardingGatherCollective* collective)
      {
        sharding_collective = collective;
      }
    protected:
      ShardingGatherCollective* sharding_collective;
    };

    /**
     * \class RemoteFillOp
     * This is a remote copy of a FillOp to be used
     * for mapper calls and other operations
     */
    class RemoteFillOp : public ExternalFill,
                         public RemoteOp,
                         public Heapify<RemoteFillOp, OPERATION_LIFETIME> {
    public:
      RemoteFillOp(Operation* ptr, AddressSpaceID src);
      RemoteFillOp(const RemoteFillOp& rhs) = delete;
      virtual ~RemoteFillOp(void);
    public:
      RemoteFillOp& operator=(const RemoteFillOp& rhs) = delete;
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
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const FillOp& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_logging_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FILL_H__
