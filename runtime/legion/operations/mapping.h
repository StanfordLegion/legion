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

#ifndef __LEGION_INLINE_MAPPING_H__
#define __LEGION_INLINE_MAPPING_H__

#include "legion/analysis/versioning.h"
#include "legion/api/mapping.h"
#include "legion/operations/collective.h"
#include "legion/operations/remote.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalMapping
     * An extension of the external-facing InlineMapping to help
     * with packing and unpacking them
     */
    class ExternalMapping : public InlineMapping,
                            public ExternalMappable {
    public:
      ExternalMapping(void);
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      void pack_external_mapping(Serializer& rez, AddressSpaceID target) const;
      void unpack_external_mapping(Deserializer& derez);
    };

    /**
     * \class MapOp
     * Mapping operations are used for computing inline mapping
     * operations.  Mapping operations will always update a
     * physical region once they have finished mapping.  They
     * then complete and commit immediately, possibly even
     * before the physical region is ready to be used.  This
     * also reflects that mapping operations cannot be rolled
     * back because once they have mapped, then information
     * has the ability to escape back to the application's
     * domain and can no longer be tracked by Legion.  Any
     * attempt to roll back an inline mapping operation
     * will result in the entire enclosing task context
     * being restarted.
     */
    class MapOp : public ExternalMapping,
                  public Operation {
    public:
      MapOp(void);
      MapOp(const MapOp& rhs) = delete;
      virtual ~MapOp(void);
    public:
      MapOp& operator=(const MapOp& rhs) = delete;
    public:
      PhysicalRegion initialize(
          InnerContext* ctx, const InlineLauncher& launcher,
          Provenance* provenance, unsigned requirement_index);
      void initialize(
          InnerContext* ctx, const PhysicalRegion& region,
          Provenance* provenance);
      virtual const RegionRequirement& get_requirement(
          unsigned idx = 0) const override
      {
        return requirement;
      }
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
      virtual void trigger_commit(void) override;
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
      virtual bool invoke_mapper(
          InstanceSet& mapped_instances,
          std::vector<PhysicalManager*>& source_instances);
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
      virtual RtEvent finalize_complete_mapping(RtEvent event) { return event; }
    protected:
      bool remap_region;
      ApUserEvent ready_event;
      ApEvent termination_event;
      PhysicalRegion region;
      unsigned parent_req_index;
      VersionInfo version_info;
      std::map<PhysicalManager*, unsigned> acquired_instances;
      std::map<Reservation, bool> atomic_locks;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager* mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      RtUserEvent profiling_reported;
      int profiling_priority;
      int copy_fill_priority;
      std::atomic<int> outstanding_profiling_requests;
      std::atomic<int> outstanding_profiling_reported;
    };

    /**
     * \class ReplMapOp
     * An inline mapping operation that is aware that it is being
     * executed in a control replicated context. We require that
     * any inline mapping be mapped on all shards before we consider
     * it mapped on any shard. The reason for this is that inline
     * mappings can act like a kind of communication between shards
     * where they are all reading/writing to the same logical region.
     */
    class ReplMapOp
      : public ReplCollectiveViewCreator<CollectiveViewCreator<MapOp> > {
    public:
      ReplMapOp(void);
      ReplMapOp(const ReplMapOp& rhs) = delete;
      virtual ~ReplMapOp(void);
    public:
      ReplMapOp& operator=(const ReplMapOp& rhs) = delete;
    public:
      void initialize_replication(ReplicateContext* ctx);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual bool invoke_mapper(
          InstanceSet& mapped_instances,
          std::vector<PhysicalManager*>& source_instances) override;
      virtual bool supports_collective_instances(void) const { return true; }
      virtual RtEvent finalize_complete_mapping(RtEvent precondition) override;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local) override;
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index) override;
      virtual bool find_shard_participants(
          std::vector<ShardID>& shards) override;
    protected:
      CollectiveID mapping_check, sources_check;
      RtBarrier collective_map_barrier;
    };

    /**
     * \class CheckCollectiveMapping
     * A class for exchanging the names of instances used for collective mapping
     */
    class CheckCollectiveMapping
      : public AllGatherCollective<true /*inorder*/> {
    public:
      CheckCollectiveMapping(ReplicateContext* ctx, CollectiveID id);
      CheckCollectiveMapping(const CheckCollectiveMapping&) = delete;
      virtual ~CheckCollectiveMapping(void);
    public:
      CheckCollectiveMapping& operator=(const CheckCollectiveMapping&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_MAPPING;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void verify(const InstanceSet& instances, MapperManager* mapper);
    protected:
      typedef op::vector<std::pair<ShardID, FieldMask> > ShardFields;
      std::map<PhysicalInstance, ShardFields> mapped_instances;
    };

    /**
     * \class CheckCollectiveSources
     * A class for exchanging the names of source instances for confirming
     * that all shards have listed the same instances for mapping
     */
    class CheckCollectiveSources : public BroadcastCollective {
    public:
      CheckCollectiveSources(ReplicateContext* ctx, CollectiveID id);
      CheckCollectiveSources(const CheckCollectiveSources&) = delete;
      virtual ~CheckCollectiveSources(void);
    public:
      CheckCollectiveSources& operator=(const CheckCollectiveSources&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CHECK_COLLECTIVE_SOURCES;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
    public:
      bool verify(const std::vector<PhysicalManager*>& instances);
    protected:
      std::vector<DistributedID> source_instances;
    };

    /**
     * \class RemoteMapOp
     * This is a remote copy of a MapOp to be used
     * for mapper calls and other operations
     */
    class RemoteMapOp : public ExternalMapping,
                        public RemoteOp,
                        public Heapify<RemoteMapOp, OPERATION_LIFETIME> {
    public:
      RemoteMapOp(Operation* ptr, AddressSpaceID src);
      RemoteMapOp(const RemoteMapOp& rhs) = delete;
      virtual ~RemoteMapOp(void);
    public:
      RemoteMapOp& operator=(const RemoteMapOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const override;
      virtual uint64_t get_context_index(void) const override;
      virtual void set_context_index(uint64_t index) override;
      virtual int get_depth(void) const override;
      virtual const Task* get_parent_task(void) const override;
      virtual const std::string_view& get_provenance_string(
          bool human = true) const override;
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
    inline std::ostream& operator<<(std::ostream& os, const MapOp& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_logging_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INLINE_MAPPING_H__
