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

#ifndef __LEGION_DETACH_H__
#define __LEGION_DETACH_H__

#include "legion/analysis/versioning.h"
#include "legion/operations/collective.h"
#include "legion/operations/pointwise.h"
#include "legion/operations/remote.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/coordinates.h"

namespace Legion {
  namespace Internal {

    /**
     * \class DetachOp
     * Operation for detaching a file from a physical instance
     */
    class DetachOp : public Operation {
    public:
      DetachOp(void);
      DetachOp(const DetachOp& rhs) = delete;
      virtual ~DetachOp(void);
    public:
      DetachOp& operator=(const DetachOp& rhs) = delete;
    public:
      Future initialize_detach(
          InnerContext* ctx, PhysicalRegion region, const bool flush,
          const bool unordered, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual unsigned find_parent_index(unsigned idx) override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_complete(ApEvent effects_done) override;
      virtual void trigger_commit(void) override;
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points) override;
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
      virtual void detach_external_instance(PhysicalManager* manager);
      virtual bool is_point_detach(void) const { return false; }
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
    protected:
      void log_requirement(void);
      ApEvent detach_external(
          const InstanceSet& target_instances, const ApEvent termination_event,
          const PhysicalTraceInfo& trace_info, RtEvent filter_precondition,
          const bool second_analysis);
    public:
      PhysicalRegion region;
      RegionRequirement requirement;
      VersionInfo version_info;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
      ApEvent detach_event;
      Future result;
      bool flush;
    };

    /**
     * \class IndexDetachOp
     * This is an index space detach operation for performing many detaches
     */
    class IndexDetachOp
      : public PointwiseAnalyzable<CollectiveViewCreator<Operation> > {
    public:
      IndexDetachOp(void);
      IndexDetachOp(const IndexDetachOp& rhs) = delete;
      virtual ~IndexDetachOp(void);
    public:
      IndexDetachOp& operator=(const IndexDetachOp& rhs) = delete;
    public:
      Future initialize_detach(
          InnerContext* ctx, LogicalRegion parent, RegionTreeNode* upper_bound,
          IndexSpaceNode* launch_bounds, ExternalResourcesImpl* external,
          const std::vector<FieldID>& privilege_fields,
          const std::vector<PhysicalRegion>& regions, bool flush,
          bool unordered, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual bool has_prepipeline_stage(void) const override { return true; }
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_complete(ApEvent effects_done) override;
      virtual void trigger_commit(void) override;
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual size_t get_collective_points(void) const override;
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
    public:
      // Override for control replication
      void handle_point_complete(ApEvent effects);
      void handle_point_commit(void);
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
    protected:
      void log_requirement(void);
    protected:
      RegionRequirement requirement;
      ExternalResources resources;
      IndexSpaceNode* launch_space;
      std::vector<PointDetachOp*> points;
      std::set<RtEvent> map_applied_conditions;
      Future result;
      unsigned parent_req_index;
      std::atomic<unsigned> points_completed;
      unsigned points_committed;
      bool commit_request;
      bool flush;
    };

    /**
     * \class PointDetachOp
     * Indvidiual detach operations for an index space detach
     */
    class PointDetachOp : public DetachOp {
    public:
      PointDetachOp(void);
      PointDetachOp(const PointDetachOp& rhs) = delete;
      virtual ~PointDetachOp(void);
    public:
      PointDetachOp& operator=(const PointDetachOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      void initialize_detach(
          IndexDetachOp* owner, InnerContext* ctx, const PhysicalRegion& region,
          const DomainPoint& point, bool flush);
    public:
      virtual void trigger_complete(ApEvent effects_done) override;
      virtual void trigger_commit(void) override;
      virtual size_t get_collective_points(void) const override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
      virtual RtEvent convert_collective_views(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, const InstanceSet& targets,
          InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
          bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      virtual unsigned find_parent_index(unsigned idx) override
      {
        return owner->find_parent_index(idx);
      }
      virtual bool is_point_detach(void) const override { return true; }
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      DomainPoint index_point;
    protected:
      IndexDetachOp* owner;
    };

    /**
     * \class ReplDetachOp
     * An detach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplDetachOp
      : public ReplCollectiveViewCreator<CollectiveViewCreator<DetachOp> > {
    public:
      ReplDetachOp(void);
      ReplDetachOp(const ReplDetachOp& rhs) = delete;
      virtual ~ReplDetachOp(void);
    public:
      ReplDetachOp& operator=(const ReplDetachOp& rhs) = delete;
    public:
      void initialize_replication(
          ReplicateContext* ctx, bool collective_instances,
          bool first_local_shard);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual RtEvent finalize_complete_mapping(RtEvent event) override;
      virtual void detach_external_instance(PhysicalManager* manager) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    public:
      // Help for unordered detachments
      void record_unordered_kind(
          std::map<std::pair<LogicalRegion, FieldID>, Operation*>& detachments);
    protected:
      RtBarrier collective_map_barrier;
      ApBarrier effects_barrier;
      size_t exchange_index;
      bool collective_instances;
      bool is_first_local_shard;
    };

    /**
     * \class ReplIndexDetachOp
     * An index space detach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexDetachOp : public ReplCollectiveViewCreator<IndexDetachOp> {
    public:
      ReplIndexDetachOp(void);
      ReplIndexDetachOp(const ReplIndexDetachOp& rhs) = delete;
      virtual ~ReplIndexDetachOp(void);
    public:
      ReplIndexDetachOp& operator=(const ReplIndexDetachOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    public:
      void initialize_replication(ReplicateContext* ctx);
      void record_unordered_kind(
          std::map<std::pair<LogicalRegion, FieldID>, Operation*>&
              region_detachments,
          std::map<std::pair<LogicalPartition, FieldID>, Operation*>&
              partition_detachments);
    protected:
      ShardingFunction* sharding_function;
      ShardParticipantsExchange* participants;
      ApBarrier effects_barrier;
    };

    /**
     * \class RemoteDetachOp
     * This is a remote copy of a DetachOp to be used for
     * mapper calls and other operations
     */
    class RemoteDetachOp : public RemoteOp,
                           public Heapify<RemoteDetachOp, OPERATION_LIFETIME> {
    public:
      RemoteDetachOp(Operation* ptr, AddressSpaceID src);
      RemoteDetachOp(const RemoteDetachOp& rhs) = delete;
      virtual ~RemoteDetachOp(void);
    public:
      RemoteDetachOp& operator=(const RemoteDetachOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
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
      DomainPoint index_point;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DETACH_H__
