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

#ifndef __LEGION_ATTACH_H__
#define __LEGION_ATTACH_H__

#include "legion/analysis/versioning.h"
#include "legion/operations/collective.h"
#include "legion/operations/pointwise.h"
#include "legion/operations/remote.h"
#include "legion/utilities/collectives.h"
#include "legion/utilities/coordinates.h"
#include "legion/utilities/instance_set.h"

namespace Legion {
  namespace Internal {

    /**
     * \class AttachOp
     * Operation for attaching a file to a physical instance
     */
    class AttachOp : public Operation {
    public:
      AttachOp(void);
      AttachOp(const AttachOp& rhs) = delete;
      virtual ~AttachOp(void);
    public:
      AttachOp& operator=(const AttachOp& rhs) = delete;
    public:
      PhysicalRegion initialize(
          InnerContext* ctx, const AttachLauncher& launcher,
          Provenance* provenance, unsigned requirement_index);
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
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
      virtual void trigger_mapping(void) override;
      virtual unsigned find_parent_index(unsigned idx) override;
      virtual void trigger_commit(void) override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual bool is_point_attach(void) const { return false; }
    public:
      void create_external_instance(void);
      virtual PhysicalManager* create_manager(
          RegionNode* node, const std::vector<FieldID>& field_set,
          const std::vector<size_t>& field_sizes,
          const std::vector<unsigned>& mask_index_map,
          const std::vector<CustomSerdezID>& serez,
          const FieldMask& external_mask);
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
    protected:
      void log_requirement(void);
      void attach_ready(bool point);
      InstanceRef create_external_instance(
          const RegionRequirement& req, const std::vector<FieldID>& field_set);
      ApEvent create_external(
          RegionNode* node, const std::vector<FieldID>& field_set,
          const std::vector<size_t>& sizes, PhysicalInstance& instance,
          LgEvent& unique_event, size_t& footprint);
      ApEvent attach_external(
          const ApEvent termination_event, const PhysicalTraceInfo& trace_info);
    public:
      ExternalResource resource;
      RegionRequirement requirement;
      VersionInfo version_info;
      PhysicalRegion region;
      unsigned parent_req_index;
      InstanceSet external_instances;
      std::set<RtEvent> map_applied_conditions;
      LayoutConstraintSet layout_constraint_set;
      Realm::ExternalInstanceResource* external_resource;
      std::vector<std::string> hdf5_field_files;
      ApEvent termination_event;
      bool restricted;
    };

    /**
     * \class IndexAttachOp
     * This provides support for doing index space attach
     * operations where we are attaching external resources
     * to many subregions of a region tree with a single operation
     */
    class IndexAttachOp
      : public PointwiseAnalyzable<CollectiveViewCreator<Operation> > {
    public:
      IndexAttachOp(void);
      IndexAttachOp(const IndexAttachOp& rhs) = delete;
      virtual ~IndexAttachOp(void);
    public:
      IndexAttachOp& operator=(const IndexAttachOp& rhs) = delete;
    public:
      ExternalResources initialize(
          InnerContext* ctx, RegionTreeNode* upper_bound,
          IndexSpaceNode* launch_bounds, const IndexAttachLauncher& launcher,
          const std::vector<unsigned>& indexes, Provenance* provenance,
          const bool replicated, unsigned requirement_offset);
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
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
      virtual bool are_all_direct_children(bool local) { return local; }
      virtual size_t get_collective_points(void) const override;
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
    public:
      void handle_point_complete(ApEvent effects);
      void handle_point_commit(void);
      void start_check_point_requirements(void);
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              point_domains);
    protected:
      void log_requirement(void);
    protected:
      RegionRequirement requirement;
      ExternalResources resources;
      IndexSpaceNode* launch_space;
      std::vector<PointAttachOp*> points;
      std::set<RtEvent> map_applied_conditions;
      std::set<RtEvent> commit_preconditions;
      unsigned parent_req_index;
      std::atomic<unsigned> points_completed;
      unsigned points_committed;
      bool commit_request;
    };

    /**
     * \class PointAttachOp
     * An individual attach operation inside of an index attach operation
     */
    class PointAttachOp : public AttachOp {
    public:
      PointAttachOp(void);
      PointAttachOp(const PointAttachOp& rhs) = delete;
      virtual ~PointAttachOp(void);
    public:
      PointAttachOp& operator=(const PointAttachOp& rhs) = delete;
    public:
      inline const DomainPoint& get_index_point(void) const
      {
        return index_point;
      }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      PhysicalRegionImpl* initialize(
          IndexAttachOp* owner, InnerContext* ctx,
          const IndexAttachLauncher& launcher, const DomainPoint& point,
          unsigned index, unsigned requirement_offset);
    public:
      virtual void trigger_complete(ApEvent effect) override;
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
      virtual bool is_point_attach(void) const override { return true; }
      virtual ContextCoordinate get_task_tree_coordinate(void) const override
      {
        return ContextCoordinate(context_index, index_point);
      }
    public:
      DomainPoint index_point;
    protected:
      IndexAttachOp* owner;
    };

    /**
     * \class IndexAttachLaunchSpace
     * This collective computes the number of points in each
     * shard of a replicated index attach collective in order
     * to help compute the index launch space
     */
    class IndexAttachLaunchSpace : public AllGatherCollective<false> {
    public:
      IndexAttachLaunchSpace(
          ReplicateContext* ctx, CollectiveIndexLocation loc);
      IndexAttachLaunchSpace(const IndexAttachLaunchSpace& rhs) = delete;
      virtual ~IndexAttachLaunchSpace(void);
    public:
      IndexAttachLaunchSpace& operator=(const IndexAttachLaunchSpace& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_INDEX_ATTACH_LAUNCH_SPACE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange_counts(size_t count);
      IndexSpaceNode* get_launch_space(Provenance* provenance);
    protected:
      std::vector<size_t> sizes;
      unsigned nonzeros;
    };

    /**
     * \class IndexAttachUpperBound
     * This computes the upper bound node in the region
     * tree for an index space attach operation
     */
    class IndexAttachUpperBound : public AllGatherCollective<false> {
    public:
      IndexAttachUpperBound(ReplicateContext* ctx, CollectiveIndexLocation loc);
      IndexAttachUpperBound(const IndexAttachUpperBound& rhs) = delete;
      virtual ~IndexAttachUpperBound(void);
    public:
      IndexAttachUpperBound& operator=(const IndexAttachUpperBound& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_INDEX_ATTACH_UPPER_BOUND;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      RegionTreeNode* find_upper_bound(RegionTreeNode* node);
    protected:
      RegionTreeNode* node;
    };

    /**
     * \class IndexAttachExchange
     * This class is used to exchange the needed metadata for
     * replicated index space attach operations
     */
    class IndexAttachExchange : public AllGatherCollective<false> {
    public:
      IndexAttachExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      IndexAttachExchange(const IndexAttachExchange& rhs) = delete;
      virtual ~IndexAttachExchange(void);
    public:
      IndexAttachExchange& operator=(const IndexAttachExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_INDEX_ATTACH_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange_spaces(std::vector<IndexSpace>& spaces);
      size_t get_spaces(std::vector<IndexSpace>& spaces, unsigned& local_start);
      IndexSpaceNode* get_launch_space(void);
    protected:
      std::map<ShardID, std::vector<IndexSpace> > shard_spaces;
    };

    /**
     * \class ReplAttachOp
     * An attach operation that is aware that it is being
     * executed in a control replicated context.
     */
    class ReplAttachOp
      : public ReplCollectiveViewCreator<CollectiveViewCreator<AttachOp> > {
    public:
      ReplAttachOp(void);
      ReplAttachOp(const ReplAttachOp& rhs) = delete;
      virtual ~ReplAttachOp(void);
    public:
      ReplAttachOp& operator=(const ReplAttachOp& rhs) = delete;
    public:
      void initialize_replication(
          ReplicateContext* ctx, bool collective_instances,
          bool deduplicate_across_shards, bool first_local_shard);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
    public:
      virtual PhysicalManager* create_manager(
          RegionNode* node, const std::vector<FieldID>& field_set,
          const std::vector<size_t>& field_sizes,
          const std::vector<unsigned>& mask_index_map,
          const std::vector<CustomSerdezID>& serez,
          const FieldMask& external_mask) override;
      virtual RtEvent finalize_complete_mapping(RtEvent event) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    protected:
      RtBarrier collective_map_barrier;
      size_t exchange_index;
      bool collective_instances;
      bool deduplicate_across_shards;
      bool is_first_local_shard;
      // individual insts: whether at least one shard lives on the local process
      bool contains_individual;
    protected:
      RtBarrier resource_barrier;
      ValueBroadcast<DistributedID>* did_broadcast;
      // Need this because std::pair<PhysicalInstance,ApEvent> is not
      // trivially copyable for reasons passing understanding
      struct InstanceEvents {
        PhysicalInstance instance;
        ApEvent ready_event;
        LgEvent unique_event;
      };
      ValueBroadcast<InstanceEvents>* single_broadcast;
    };

    /**
     * \class ReplIndexAttachOp
     * An index space attach operation that is aware
     * that it is executing in a control replicated context
     */
    class ReplIndexAttachOp : public ReplCollectiveViewCreator<IndexAttachOp> {
    public:
      ReplIndexAttachOp(void);
      ReplIndexAttachOp(const ReplIndexAttachOp& rhs) = delete;
      virtual ~ReplIndexAttachOp(void);
    public:
      ReplIndexAttachOp& operator=(const ReplIndexAttachOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_prepipeline_stage(void) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual bool are_all_direct_children(bool local) override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
      virtual void finish_check_point_requirements(
          std::map<unsigned, std::vector<std::pair<DomainPoint, Domain> > >&
              point_domains) override;
    public:
      void initialize_replication(ReplicateContext* ctx);
    protected:
      IndexAttachExchange* collective;
      ShardParticipantsExchange* participants;
      ShardingFunction* sharding_function;
      CollectiveID interfering_check_id;
      InterferingPointExchange<ReplIndexAttachOp>* interfering_exchange;
    };

    /**
     * \class RemoteAttachOp
     * This is a remote copy of a AttachOp to be used for
     * mapper calls and other operations
     */
    class RemoteAttachOp : public RemoteOp,
                           public Heapify<RemoteAttachOp, OPERATION_LIFETIME> {
    public:
      RemoteAttachOp(Operation* ptr, AddressSpaceID src);
      RemoteAttachOp(const RemoteAttachOp& rhs) = delete;
      virtual ~RemoteAttachOp(void);
    public:
      RemoteAttachOp& operator=(const RemoteAttachOp& rhs) = delete;
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
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    protected:
      DomainPoint index_point;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ATTACH_H__
