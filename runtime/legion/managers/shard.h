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

#ifndef __LEGION_SHARD_MANAGER_H__
#define __LEGION_SHARD_MANAGER_H__

#include "legion/contexts/replicate.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/operations/collective.h"
#include "legion/tasks/shard.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ShardMapping
     * A mapping from the shard IDs to their address spaces
     */
    class ShardMapping : public Collectable {
    public:
      ShardMapping(void);
      ShardMapping(const ShardMapping& rhs) = delete;
      ShardMapping(const std::vector<AddressSpaceID>& spaces);
      ~ShardMapping(void);
    public:
      ShardMapping& operator=(const ShardMapping& rhs) = delete;
      AddressSpaceID operator[](unsigned idx) const;
      AddressSpaceID& operator[](unsigned idx);
    public:
      inline bool empty(void) const { return address_spaces.empty(); }
      inline size_t size(void) const { return address_spaces.size(); }
      inline void resize(size_t size) { address_spaces.resize(size); }
    public:
      void pack_mapping(Serializer& rez) const;
      void unpack_mapping(Deserializer& derez);
      static void pack_empty(Serializer& rez);
    protected:
      std::vector<AddressSpaceID> address_spaces;
    };

    /**
     * \class ShardManager
     * This is a class that manages the execution of one or
     * more shards for a given control replication context on
     * a single node. It provides support for doing broadcasts,
     * reductions, and exchanges of information between the
     * variaous shard tasks.
     */
    class ShardManager : public CollectiveViewCreator<CollectiveHelperOp>,
                         public Mapper::SelectShardingFunctorInput {
    public:
      enum BroadcastMessageKind {
        RESOURCE_UPDATE_KIND,
        CREATED_REGION_UPDATE_KIND,
      };
    public:
      struct AttachDeduplication {
      public:
        AttachDeduplication(void) : done_count(0) { }
      public:
        RtUserEvent pending;
        std::vector<const IndexAttachLauncher*> launchers;
        std::map<LogicalRegion, const IndexAttachLauncher*> owners;
        unsigned done_count;
      };
    public:
      ShardManager(
          DistributedID did, CollectiveMapping* mapping, unsigned local_count,
          const Mapper::ContextConfigOutput& config, bool top,
          bool isomorphic_points, bool cr, const Domain& shard_domain,
          std::vector<DomainPoint>&& shard_points,
          std::vector<DomainPoint>&& sorted_points,
          std::vector<ShardID>&& shard_lookup, SingleTask* original = nullptr,
          RtBarrier callback_bar = RtBarrier::NO_RT_BARRIER);
      ShardManager(const ShardManager& rhs) = delete;
      ~ShardManager(void);
    public:
      ShardManager& operator=(const ShardManager& rhs) = delete;
    public:
      void notify_local(void) override;
    public:
      inline ShardMapping& get_mapping(void) const { return *address_spaces; }
      inline CollectiveMapping& get_collective_mapping(void) const
      {
        return *collective_mapping;
      }
      inline AddressSpaceID get_shard_space(ShardID sid) const
      {
        return (*address_spaces)[sid];
      }
      inline bool is_first_local_shard(ShardTask* task) const
      {
        return (local_shards[0] == task);
      }
      inline ReplicateContext* find_local_context(void) const
      {
        return local_shards[0]->get_replicate_context();
      }
      inline size_t count_local_shards(void) const
      {
        return local_shards.size();
      }
      inline unsigned find_local_index(ShardTask* task) const
      {
        for (unsigned idx = 0; idx < local_shards.size(); idx++)
          if (local_shards[idx] == task)
            return idx;
        std::abort();
      }
      inline ContextID get_first_shard_tree_context(void) const
      {
        return local_shards.front()
            ->get_replicate_context()
            ->get_logical_tree_context();
      }
    public:  // From CollectiveHelperOp
      virtual InnerContext* get_context(void) override;
      virtual InnerContext* find_physical_context(unsigned index) override;
      virtual size_t get_collective_points(void) const override;
    public:
      void distribute_explicit(
          SingleTask* task, VariantID chosen_variant,
          std::vector<Processor>& target_processors,
          std::vector<VariantID>& leaf_variants);
      void distribute_implicit(
          TaskID top_task_id, MapperID mapper_id, Processor::Kind kind,
          unsigned shards_per_space, TopLevelContext* top_context);
      void pack_shard_manager(Serializer& rez) const;
      void set_shard_mapping(std::vector<Processor>& shard_mapping);
      ShardTask* create_shard(
          ShardID id, Processor target, VariantID variant,
          InnerContext* parent_ctx, SingleTask* source);
      ShardTask* create_shard(
          ShardID id, Processor target, VariantID variant,
          InnerContext* parent_ctx, Deserializer& derez);
    public:
      virtual void finalize_collective_versioning_analysis(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform) override;
      virtual void construct_collective_mapping(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous) override;
      void finalize_replicate_collective_versioning(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, CollectiveVersioningBase::RegionVersioning>&
              to_perform);
      void finalize_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey& key,
          std::map<
              LogicalRegion, CollectiveViewCreatorBase::CollectiveRendezvous>&
              rendezvous);
      void rendezvous_check_virtual_mappings(
          ShardID shard, MapperManager* mapper,
          const std::vector<bool>& virtual_mappings);
      RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index = std::optional<unsigned>());
    public:
      EquivalenceSet* get_initial_equivalence_set(
          unsigned idx, LogicalRegion region, InnerContext* context,
          bool first_shard);
      // If the creating shards are nullptr we'll assume that they are all
      // participating in the creation of the index space
      EquivalenceSet* deduplicate_equivalence_set_creation(
          RegionNode* node, size_t op_ctx_index, unsigned refinement_number,
          InnerContext* context, bool first_shard,
          const std::vector<ShardID>* creating_shards = nullptr);
      void* deduplicate_fill_view_allocation(size_t key, bool* first = nullptr);
      FillView* deduplicate_fill_view_creation(
          void* ptr, DistributedID fresh_did, UniqueID op_uid,
          bool* first = nullptr);
      void deduplicate_attaches(
          const IndexAttachLauncher& launcher, std::vector<unsigned>& indexes);
      Future deduplicate_future_creation(
          ReplicateContext* ctx, DistributedID did, Operation* op,
          const DomainPoint& point);
      FutureMap deduplicate_future_map_creation(
          ReplicateContext* ctx, Operation* op, IndexSpaceNode* domain,
          IndexSpaceNode* shard_domain, DistributedID did,
          Provenance* provenance);
      FutureMap deduplicate_future_map_creation(
          ReplicateContext* ctx, IndexSpaceNode* domain,
          IndexSpaceNode* shard_domain, DistributedID did,
          Provenance* provenance);
      // Return true if we have a shard on every address space
      bool is_total_sharding(void);
      template<typename T>
      inline void exchange_shard_local_op_data(
          uint64_t context_index, size_t exchange_index, const T& data)
      {
        static_assert(std::is_trivially_copyable<T>());
        exchange_shard_local_op_data(
            context_index, exchange_index, &data, sizeof(data));
      }
      void exchange_shard_local_op_data(
          uint64_t context_index, size_t exchange_index, const void* data,
          size_t size);
      template<typename T>
      inline T find_shard_local_op_data(
          uint64_t context_index, size_t exchange_index)
      {
        static_assert(std::is_trivially_copyable<T>());
        T result;
        find_shard_local_op_data(
            context_index, exchange_index, &result, sizeof(result));
        return result;
      }
      void find_shard_local_op_data(
          uint64_t context_index, size_t exchange_index, void* data,
          size_t size);
      void barrier_shard_local(uint64_t context_index, size_t exchange_index);
    public:
      RtEvent complete_startup_initialization(bool local = true);
      void handle_post_mapped(bool local, RtEvent precondition);
      bool handle_future(
          ApEvent effects, FutureInstance* instance, const void* metadata,
          size_t metasize);
      ApEvent trigger_task_complete(bool local, ApEvent effects_done);
      void trigger_task_commit(bool local, RtEvent precondition);
    public:
      void send_collective_message(
          ShardID target, const ShardCollectiveMessage& message);
      void handle_collective_message(Deserializer& derez);
    public:
      void send_rendezvous_message(
          ShardID target, const ReplicateRendezvousMessage& rez);
      void handle_rendezvous_message(Deserializer& derez);
    public:
      void send_compute_equivalence_sets(
          ShardID target, const ReplComputeEquivalenceSets& rez);
      void handle_compute_equivalence_sets(Deserializer& derez);
      void handle_equivalence_set_notification(Deserializer& derez);
    public:
      void send_output_equivalence_set(
          ShardID target, const ReplOutputEquivalenceSet& rez);
      void handle_output_equivalence_set(Deserializer& derez);
    public:
      void send_output_offset(ShardID target, const ReplOutputOffset& rez);
      void handle_output_offset(Deserializer& derez);
    public:
      void send_refine_equivalence_sets(
          ShardID target, const ReplRefineEquivalenceSets& rez);
      void handle_refine_equivalence_sets(Deserializer& derez);
    public:
      void broadcast_resource_update(
          ShardTask* source, const Serializer& rez,
          std::set<RtEvent>& applied_events);
    public:
      void broadcast_created_region_contexts(
          ShardTask* source, const Serializer& rez,
          std::set<RtEvent>& applied_events);
      void send_created_region_contexts(
          ShardID target, ReplCreatedRegions& rez,
          std::set<RtEvent>& applied_events);
      void handle_created_region_contexts(
          Deserializer& derez, std::set<RtEvent>& applied_events);
    public:
      bool has_empty_shard_subtree(
          AddressSpaceID space, ShardingFunction* sharding,
          IndexSpaceNode* full_space, IndexSpace sharding_space);
    public:
      void broadcast_message(
          ShardTask* source, const Serializer& rez, BroadcastMessageKind kind,
          std::set<RtEvent>& applied_events);
      void handle_broadcast(Deserializer& derez);
    public:
      void send_trace_event_request(
          ShardedPhysicalTemplate* physical_template, ShardID shard_source,
          AddressSpaceID template_source, size_t template_index, ApEvent event,
          AddressSpaceID event_space, RtUserEvent done_event);
      void send_trace_event_response(
          ShardedPhysicalTemplate* physical_template,
          AddressSpaceID template_source, ApEvent event, ApBarrier result,
          RtUserEvent done_event);
      RtEvent send_trace_event_trigger(
          TraceID trace_id, AddressSpaceID target, ApUserEvent lhs, ApEvent rhs,
          const TraceLocalID& tlid);
      void send_trace_frontier_request(
          ShardedPhysicalTemplate* physical_template, ShardID shard_source,
          AddressSpaceID template_source, size_t template_index, ApEvent event,
          AddressSpaceID event_space, unsigned frontier,
          RtUserEvent done_event);
      void send_trace_frontier_response(
          ShardedPhysicalTemplate* physical_template,
          AddressSpaceID template_source, unsigned frontier, ApBarrier result,
          RtUserEvent done_event);
      void send_trace_update(ShardID target, const ReplTraceUpdateMessage& rez);
      void handle_trace_update(Deserializer& derez, AddressSpaceID source);
      void send_find_trace_local_sets(
          ShardID target, const ReplFindTraceSets& rez);
      void handle_find_trace_local_sets(
          Deserializer& derez, AddressSpaceID source);
    public:
      ShardID find_collective_owner(RegionTreeID tid) const;
      void send_find_or_create_collective_view(
          ShardID target, const ReplFindCollectiveView& rez);
      void handle_find_or_create_collective_view(Deserializer& derez);
    public:
      ShardingFunction* find_sharding_function(
          ShardingID sid, bool skip_check = false);
    public:
#ifdef LEGION_USE_LIBDL
      void perform_global_registration_callbacks(
          Realm::DSOReferenceImplementation* dso, const void* buffer,
          size_t buffer_size, bool withargs, size_t dedup_tag,
          RtEvent local_done, RtEvent global_done,
          std::set<RtEvent>& preconditions);
#endif
      bool perform_semantic_attach(void);
    public:
      const std::vector<DomainPoint> shard_points;
      const std::vector<DomainPoint> sorted_points;
      const std::vector<ShardID> shard_lookup;
      const Domain shard_domain;
      const size_t total_shards;
      SingleTask* const original_task;
      const unsigned local_constituents;
      const unsigned remote_constituents;
      // Only valid if control replicated
      const Mapper::ContextConfigOutput context_configuration;
      const bool top_level_task;
      const bool isomorphic_points;
      const bool control_replicated;
    protected:
      mutable LocalLock manager_lock;
      // Inheritted from Mapper::SelectShardingFunctorInput
      // std::vector<Processor>        shard_mapping;
      ShardMapping* address_spaces;
      std::vector<ShardTask*> local_shards;
    protected:
      // There are five kinds of signals that come back from
      // the execution of the shards:
      // - startup complete
      // - mapping complete
      // - future result
      // - task complete
      // - task commit
      RtUserEvent startup_complete;
      // The owner applies these to the original task object only
      // after they have occurred for all the shards
      unsigned local_startup_complete, remote_startup_complete;
      unsigned local_mapping_complete, remote_mapping_complete;
      unsigned trigger_local_complete, trigger_remote_complete;
      unsigned trigger_local_commit, trigger_remote_commit;
      unsigned semantic_attach_counter;
      size_t future_size;
      std::set<RtEvent> mapping_preconditions;
    protected:
      // This barrier is only needed for control replicated tasks
      RtBarrier callback_barrier;
    protected:
      std::map<ShardingID, ShardingFunction*> sharding_functions;
    protected:
      // We need a triple here to uniquely identify creations and
      // make sure the right equivalence set gets hooked up with
      // in the right way.
      // 1. index of the creator op in the context
      // 2. number of the refinement for that creator op
      // 3. logical region being refined
      struct EquivalenceSetKey {
      public:
        inline EquivalenceSetKey(void)
          : handle(LogicalRegion::NO_REGION), op_ctx_index(0),
            refinement_number(0)
        { }
        inline EquivalenceSetKey(size_t op, unsigned number, LogicalRegion h)
          : handle(h), op_ctx_index(op), refinement_number(number)
        { }
      public:
        inline bool operator<(const EquivalenceSetKey& rhs) const
        {
          if (op_ctx_index < rhs.op_ctx_index)
            return true;
          if (op_ctx_index > rhs.op_ctx_index)
            return false;
          if (refinement_number < rhs.refinement_number)
            return true;
          if (refinement_number > rhs.refinement_number)
            return false;
          return (handle < rhs.handle);
        }
      public:
        LogicalRegion handle;
        size_t op_ctx_index;
        unsigned refinement_number;
      };
      struct NewEquivalenceSet {
      public:
        EquivalenceSet* new_set;
        DistributedID did;
        CollectiveMapping* mapping;
        RtUserEvent ready_event;
        size_t remaining;
      };
      std::map<EquivalenceSetKey, NewEquivalenceSet> created_equivalence_sets;
      std::map<DistributedID, std::pair<FutureImpl*, size_t> > created_futures;
      std::map<DistributedID, std::pair<ReplFutureMapImpl*, size_t> >
          created_future_maps;
      std::map<size_t, std::pair<void*, size_t> > fill_view_allocations;
      std::map<void*, size_t> fill_view_creations;
      // ApEvents describing the completion of each shard
      ApUserEvent all_shards_complete;
      std::set<ApEvent> shard_effects;
      std::set<RtEvent> commit_preconditions;
#ifdef LEGION_USE_LIBDL
    protected:
      std::set<Runtime::RegistrationKey> unique_registration_callbacks;
#endif
    protected:
      struct ShardLocalData {
      public:
        ShardLocalData(void) : buffer(nullptr), size(0), remaining(0) { }
      public:
        void* buffer;
        size_t size;
        RtUserEvent pending;
        unsigned remaining;
      };
      std::map<std::pair<size_t, size_t>, ShardLocalData> shard_local_data;
    protected:
      AttachDeduplication* attach_deduplication;
    protected:
      struct VirtualMappingRendezvous {
        std::vector<bool> virtual_mappings;
        MapperManager* mapper;
        ShardID shard;
        unsigned remaining_arrivals;
      };
      VirtualMappingRendezvous* virtual_mapping_rendezvous;
    };

    /**
     * \class ImplicitShardManager
     * This is a class for helping to construct implicitly
     * control replicated top-level tasks from external threads.
     * It helps to setup tasks just as though they had been
     * control replicated, except everything was already control
     * replicated remotely.
     */
    class ImplicitShardManager : public Collectable {
    public:
      ImplicitShardManager(
          TaskID tid, MapperID mid, Processor::Kind k,
          unsigned shards_per_address_space);
      ImplicitShardManager(const ImplicitShardManager& rhs) = delete;
      ~ImplicitShardManager(void);
    public:
      ImplicitShardManager& operator=(const ImplicitShardManager& rhs) = delete;
    public:
      ShardTask* create_shard(
          int shard_id, const DomainPoint& shard_point, Processor proxy,
          const char* task_name);
    protected:
      void create_shard_manager(void);
      void request_shard_manager(void);
    public:
      void process_implicit_rendezvous(Deserializer& derez);
      RtUserEvent set_shard_manager(
          ShardManager* manager, TopLevelContext* context);
    public:
      const TaskID task_id;
      const MapperID mapper_id;
      const Processor::Kind kind;
      const unsigned shards_per_address_space;
    protected:
      mutable LocalLock manager_lock;
      unsigned remaining_local_arrivals;
      unsigned remaining_remote_arrivals;
      unsigned local_shard_id;
      TopLevelContext* top_context;
      ShardManager* shard_manager;
      CollectiveMapping* collective_mapping;
      RtUserEvent manager_ready;
      Processor local_proxy;
      const char* local_task_name;
      std::map<DomainPoint, std::pair<ShardID, Processor> > shard_points;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SHARD_MANAGER_H__
