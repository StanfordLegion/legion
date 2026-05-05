/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_REPLICATE_CONTEXT_H__
#define __LEGION_REPLICATE_CONTEXT_H__

#include "legion/contexts/inner.h"
#include "legion/api/redop.h"
#include "legion/tasks/shard.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ReplicateContext
     * A replicate context is a special kind of inner context for
     * executing control-replicated tasks.
     */
    class ReplicateContext
      : public HeapifyMixin<ReplicateContext, InnerContext, CONTEXT_LIFETIME> {
    public:
      struct ISBroadcast {
      public:
        ISBroadcast(void) : did(0), expr_id(0), double_buffer(false) { }
        ISBroadcast(DistributedID d, IndexTreeID t, IndexSpaceExprID e, bool db)
          : did(d), tid(t), expr_id(e), double_buffer(db)
        { }
      public:
        DistributedID did;
        IndexTreeID tid;
        IndexSpaceExprID expr_id;
        bool double_buffer;
      };
      struct IPBroadcast {
      public:
        IPBroadcast(void) : did(0), double_buffer(false) { }
        IPBroadcast(DistributedID d, bool db) : did(d), double_buffer(db) { }
      public:
        DistributedID did;
        bool double_buffer;
      };
      struct FSBroadcast {
      public:
        FSBroadcast(void) : did(0), double_buffer(false) { }
        FSBroadcast(DistributedID d, bool db) : did(d), double_buffer(db) { }
      public:
        DistributedID did;
        bool double_buffer;
      };
      struct FIDBroadcast {
      public:
        FIDBroadcast(void) : field_id(0), double_buffer(false) { }
        FIDBroadcast(FieldID fid, bool db)
          : field_id(fid), double_buffer(db) { }
      public:
        FieldID field_id;
        bool double_buffer;
      };
      struct LRBroadcast {
      public:
        LRBroadcast(void) : tid(0), double_buffer(false) { }
        LRBroadcast(RegionTreeID t, DistributedID d, bool db)
          : tid(t), did(d), double_buffer(db)
        { }
      public:
        RegionTreeID tid;
        DistributedID did;
        bool double_buffer;
      };
      struct DIDBroadcast {
        DIDBroadcast(void) : did(0), double_buffer(false) { }
        DIDBroadcast(DistributedID d, bool db) : did(d), double_buffer(db) { }
      public:
        DistributedID did;
        bool double_buffer;
      };
      struct IntraSpaceDeps {
      public:
        std::map<ShardID, RtEvent> ready_deps;
        std::map<ShardID, RtUserEvent> pending_deps;
      };
    public:
      template<typename T, bool LOGICAL, bool SINGLE = false>
      class ReplBarrier {
      public:
        ReplBarrier(void) : owner(false) { }
        ReplBarrier(const ReplBarrier& rhs) = delete;
        ReplBarrier(ReplBarrier&& rhs) noexcept
          : barrier(rhs.barrier), owner(rhs.owner)
        {
          rhs.owner = false;
        }
        ~ReplBarrier(void)
        {
          if (owner && barrier.exists())
            barrier.destroy_barrier();
        }
      public:
        ReplBarrier& operator=(const ReplBarrier& rhs) = delete;
        inline ReplBarrier& operator=(ReplBarrier&& rhs) noexcept
        {
          if (owner && barrier.exists())
            barrier.destroy_barrier();
          barrier = rhs.barrier;
          owner = rhs.owner;
          rhs.owner = false;
          return *this;
        }
      public:
#ifdef LEGION_DEBUG_COLLECTIVES
        inline T next(
            ReplicateContext* ctx, ReductionOpID redop = 0,
            const void* init_value = nullptr, size_t init_size = 0)
#else
        inline T next(ReplicateContext* ctx)
#endif
        {
          if (!barrier.exists())
          {
            if (LOGICAL)
              owner = ctx->create_new_logical_barrier(
                  barrier,
#ifdef LEGION_DEBUG_COLLECTIVES
                  redop, init_value, init_size,
#endif
                  SINGLE ? 1 : ctx->total_shards);
            else
              owner = ctx->create_new_replicate_barrier(
                  barrier,
#ifdef LEGION_DEBUG_COLLECTIVES
                  redop, init_value, init_size,
#endif
                  SINGLE ? 1 : ctx->total_shards);
          }
          const T result = barrier;
          Runtime::advance_barrier(barrier);
          return result;
        }
      private:
        T barrier;
        bool owner;
      };
    public:
      enum ReplicateAPICall {
        REPLICATE_PERFORM_REGISTRATION_CALLBACK,
        REPLICATE_CONSENSUS_MATCH,
        REPLICATE_REGISTER_TASK_VARIANT,
        REPLICATE_GENERATE_DYNAMIC_TRACE_ID,
        REPLICATE_GENERATE_DYNAMIC_MAPPER_ID,
        REPLICATE_GENERATE_DYNAMIC_PROJECTION_ID,
        REPLICATE_GENERATE_DYNAMIC_SHARDING_ID,
        REPLICATE_GENERATE_DYNAMIC_CONCURRENT_ID,
        REPLICATE_GENERATE_DYNAMIC_EXCEPTION_HANDLER_ID,
        REPLICATE_GENERATE_DYNAMIC_TASK_ID,
        REPLICATE_GENERATE_DYNAMIC_REDUCTION_ID,
        REPLICATE_GENERATE_DYNAMIC_SERDEZ_ID,
        REPLICATE_CREATE_INDEX_SPACE,
        REPLICATE_CREATE_UNBOUND_INDEX_SPACE,
        REPLICATE_UNION_INDEX_SPACES,
        REPLICATE_INTERSECT_INDEX_SPACES,
        REPLICATE_SUBTRACT_INDEX_SPACES,
        REPLICATE_CREATE_SHARED_OWNERSHIP,
        REPLICATE_DESTROY_INDEX_SPACE,
        REPLICATE_DESTROY_INDEX_PARTITION,
        REPLICATE_CREATE_EQUAL_PARTITION,
        REPLICATE_CREATE_PARTITION_BY_WEIGHTS,
        REPLICATE_CREATE_PARTITION_BY_UNION,
        REPLICATE_CREATE_PARTITION_BY_INTERSECTION,
        REPLICATE_CREATE_PARTITION_BY_DIFFERENCE,
        REPLICATE_CREATE_CROSS_PRODUCT_PARTITIONS,
        REPLICATE_CREATE_ASSOCIATION,
        REPLICATE_CREATE_RESTRICTED_PARTITION,
        REPLICATE_CREATE_PARTITION_BY_DOMAIN,
        REPLICATE_CREATE_PARTITION_BY_FIELD,
        REPLICATE_CREATE_PARTITION_BY_IMAGE,
        REPLICATE_CREATE_PARTITION_BY_IMAGE_RANGE,
        REPLICATE_CREATE_PARTITION_BY_PREIMAGE,
        REPLICATE_CREATE_PARTITION_BY_PREIMAGE_RANGE,
        REPLICATE_CREATE_PENDING_PARTITION,
        REPLICATE_CREATE_INDEX_SPACE_UNION,
        REPLICATE_CREATE_INDEX_SPACE_INTERSECTION,
        REPLICATE_CREATE_INDEX_SPACE_DIFFERENCE,
        REPLICATE_CREATE_FIELD_SPACE,
        REPLICATE_DESTROY_FIELD_SPACE,
        REPLICATE_ALLOCATE_FIELD,
        REPLICATE_FREE_FIELD,
        REPLICATE_ALLOCATE_FIELDS,
        REPLICATE_FREE_FIELDS,
        REPLICATE_CREATE_LOGICAL_REGION,
        REPLICATE_DESTROY_LOGICAL_REGION,
        REPLICATE_RESET_EQUIVALENCE_SETS,
        REPLICATE_CREATE_FIELD_ALLOCATOR,
        REPLICATE_EXECUTE_TASK,
        REPLICATE_EXECUTE_INDEX_SPACE,
        REPLICATE_REDUCE_FUTURE_MAP,
        REPLICATE_CONSTRUCT_FUTURE_MAP,
        REPLICATE_FUTURE_WAIT,
        REPLICATE_FUTURE_MAP_GET_ALL_FUTURES,
        REPLICATE_FUTURE_MAP_WAIT_ALL_FUTURES,
        REPLICATE_MAP_REGION,
        REPLICATE_REMAP_REGION,
        REPLICATE_FILL_FIELDS,
        REPLICATE_DISCARD_FIELDS,
        REPLICATE_ISSUE_COPY,
        REPLICATE_ATTACH_RESOURCE,
        REPLICATE_DETACH_RESOURCE,
        REPLICATE_INDEX_ATTACH_RESOURCE,
        REPLICATE_INDEX_DETACH_RESOURCE,
        REPLICATE_ACQUIRE,
        REPLICATE_RELEASE,
        REPLICATE_MUST_EPOCH,
        REPLICATE_TIMING_MEASUREMENT,
        REPLICATE_TUNABLE_SELECTION,
        REPLICATE_MAPPING_FENCE,
        REPLICATE_EXECUTION_FENCE,
        REPLICATE_BEGIN_TRACE,
        REPLICATE_END_TRACE,
        REPLICATE_CREATE_PHASE_BARRIER,
        REPLICATE_DESTROY_PHASE_BARRIER,
        REPLICATE_ADVANCE_PHASE_BARRIER,
        REPLICATE_ADVANCE_DYNAMIC_COLLECTIVE,
        REPLICATE_END_TASK,
        REPLICATE_FUTURE_FROM_VALUE,
        REPLICATE_ATTACH_TASK_INFO,
        REPLICATE_ATTACH_INDEX_SPACE_INFO,
        REPLICATE_ATTACH_INDEX_PARTITION_INFO,
        REPLICATE_ATTACH_FIELD_SPACE_INFO,
        REPLICATE_ATTACH_FIELD_INFO,
        REPLICATE_ATTACH_LOGICAL_REGION_INFO,
        REPLICATE_ATTACH_LOGICAL_PARTITION_INFO,
        REPLICATE_PUSH_EXCEPTION_HANDLER,
        REPLICATE_POP_EXCEPTION_HANDLER,
      };
    public:
      class AttachDetachShardingFunctor : public ShardingFunctor {
      public:
        AttachDetachShardingFunctor(void) { }
        virtual ~AttachDetachShardingFunctor(void) { }
      public:
        virtual ShardID shard(
            const DomainPoint& point, const Domain& full_space,
            const size_t total_shards);
      };
      /**
       * \class UniversalShardingFunctor
       * This is a special sharding functor only used during the logical
       * analysis and has no bearing on the actual computed sharding. For
       * some operations we need to have a way to say that an individual
       * operation will be analyzed collectively on all the shards. This
       * sharding function accomplishes this by mapping all the points to
       * the non-shard UINT_MAX which will be non-interfering with
       * This maps all the points to the non-shard UINT_MAX which means that
       * it will interfere with any normally mapped projections but not with
       * any other projections which will be analyzed on all the nodes.
       */
      class UniversalShardingFunctor : public ShardingFunctor {
      public:
        UniversalShardingFunctor(void) { }
        virtual ~UniversalShardingFunctor(void) { }
      public:
        virtual ShardID shard(
            const DomainPoint& point, const Domain& full_space,
            const size_t total_shards)
        {
          return std::numeric_limits<ShardID>::max();
        }
      };
    public:
      class HashVerifier : protected Murmur3Hasher {
      public:
        HashVerifier(
            ReplicateContext* ctx, bool p, bool every_call,
            Provenance* prov = nullptr)
          : Murmur3Hasher(), context(ctx), provenance(prov), precise(p),
            verify_every_call(every_call)
        { }
        HashVerifier(const HashVerifier& rhs) = delete;
        HashVerifier& operator=(const HashVerifier& rhs) = delete;
      public:
        template<typename T>
        inline void hash(const T& value, const char* description)
        {
          if (precise)
            Murmur3Hasher::hash<T, true>(value);
          else
            Murmur3Hasher::hash<T, false>(value);
          if (verify_every_call)
            verify(description, true /*verify every call*/);
        }
        inline void hash(
            const void* value, size_t size, const char* description)
        {
          Murmur3Hasher::hash(value, size);
          if (verify_every_call)
            verify(description, true /*verify every call*/);
        }
        inline bool verify(const char* description, bool every_call = false)
        {
          uint64_t hash[2];
          finalize(hash);
          return context->verify_hash(
              hash, description, provenance, every_call);
        }
      public:
        ReplicateContext* const context;
        Provenance* const provenance;
        const bool precise;
        const bool verify_every_call;
      };
    public:
      ReplicateContext(
          const Mapper::ContextConfigOutput& config, ShardTask* owner, int d,
          bool full_inner, const std::vector<RegionRequirement>& reqs,
          const std::vector<OutputRequirement>& output_reqs,
          const std::vector<unsigned>& parent_indexes,
          const std::vector<bool>& virt_mapped, TaskPriority priority,
          ApEvent execution_fence_event, ShardManager* manager,
          bool inline_task, bool implicit_task = false,
          bool concurrent = false);
      ReplicateContext(const ReplicateContext& rhs) = delete;
      virtual ~ReplicateContext(void);
    public:
      ReplicateContext& operator=(const ReplicateContext& rhs) = delete;
    public:
      inline int get_shard_collective_radix(void) const
      {
        return shard_collective_radix;
      }
      inline int get_shard_collective_log_radix(void) const
      {
        return shard_collective_log_radix;
      }
      inline int get_shard_collective_stages(void) const
      {
        return shard_collective_stages;
      }
      inline int get_shard_collective_participating_shards(void) const
      {
        return shard_collective_participating_shards;
      }
      inline int get_shard_collective_last_radix(void) const
      {
        return shard_collective_last_radix;
      }
      virtual ShardID get_shard_id(void) const override
      {
        return owner_shard->shard_id;
      }
      virtual DistributedID get_replication_id(void) const override;
      virtual size_t get_total_shards(void) const override
      {
        return total_shards;
      }
      virtual ContextID get_physical_tree_context(void) const override;
    public:  // Privilege tracker methods
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
    public:  // HashVerifier method
      bool verify_hash(
          const uint64_t hash[2], const char* description,
          Provenance* provenance, bool every);
    protected:
      void receive_replicate_resources(
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
          std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
          RtBarrier& mapped_barrier, RtBarrier& execution_barrier);
      void register_region_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedRegion>& regions, std::set<RtEvent>& preconditions,
          RtBarrier& ready_barrier, RtBarrier& mapped_barrier,
          RtBarrier& execution_barrier);
      void register_field_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedField>& fields, std::set<RtEvent>& preconditions,
          RtBarrier& ready_barrier, RtBarrier& mapped_barrier,
          RtBarrier& execution_barrier);
      void register_field_space_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedFieldSpace>& spaces,
          std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
          RtBarrier& mapped_barrier, RtBarrier& execution_barrier);
      void register_index_space_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedIndexSpace>& spaces,
          std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
          RtBarrier& mapped_barrier, RtBarrier& execution_barrier);
      void register_index_partition_deletions(
          const std::map<Operation*, GenerationID>& dependences,
          std::vector<DeletedPartition>& parts,
          std::set<RtEvent>& preconditions, RtBarrier& ready_barrier,
          RtBarrier& mapped_barrier, RtBarrier& execution_barrier);
    public:
      void perform_replicated_region_deletions(
          std::vector<LogicalRegion>& regions,
          std::set<RtEvent>& preconditions);
      void perform_replicated_field_deletions(
          std::vector<std::pair<FieldSpace, FieldID> >& fields,
          std::set<RtEvent>& preconditions);
      void perform_replicated_field_space_deletions(
          std::vector<FieldSpace>& spaces, std::set<RtEvent>& preconditions);
      void perform_replicated_index_space_deletions(
          std::vector<IndexSpace>& spaces, std::set<RtEvent>& preconditions);
      void perform_replicated_index_partition_deletions(
          std::vector<IndexPartition>& parts, std::set<RtEvent>& preconditions);
    public:
#ifdef LEGION_USE_LIBDL
      virtual void perform_global_registration_callbacks(
          Realm::DSOReferenceImplementation* dso, const void* buffer,
          size_t buffer_size, bool withargs, size_t dedup_tag,
          RtEvent local_done, RtEvent global_done,
          std::set<RtEvent>& preconditions) override;
#endif
      virtual void print_once(FILE* f, const char* message) const override;
      virtual void log_once(Realm::LoggerMessage& message) const override;
      virtual Future from_value(
          const void* value, size_t value_size, bool owned,
          Provenance* provenance, bool shard_local) override;
      virtual Future from_value(
          const void* buffer, size_t size, bool owned,
          const Realm::ExternalInstanceResource& resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          Provenance* provenance, bool shard_local) override;
      virtual Future consensus_match(
          const void* input, void* output, size_t num_elements,
          size_t element_size, Provenance* provenance) override;
    public:
      virtual VariantID register_variant(
          const TaskVariantRegistrar& registrar, const void* user_data,
          size_t user_data_size, const CodeDescriptor& desc, size_t ret_size,
          bool has_ret_size, VariantID vid, bool check_task_id) override;
      virtual VariantImpl* select_inline_variant(
          TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
          std::deque<InstanceSet>& physical_instances) override;
      virtual TraceID generate_dynamic_trace_id(void) override;
      virtual MapperID generate_dynamic_mapper_id(void) override;
      virtual ProjectionID generate_dynamic_projection_id(void) override;
      virtual ShardingID generate_dynamic_sharding_id(void) override;
      virtual ConcurrentID generate_dynamic_concurrent_id(void) override;
      virtual ExceptionHandlerID generate_dynamic_exception_handler_id(
          void) override;
      virtual TaskID generate_dynamic_task_id(void) override;
      virtual ReductionOpID generate_dynamic_reduction_id(void) override;
      virtual CustomSerdezID generate_dynamic_serdez_id(void) override;
      virtual bool perform_semantic_attach(
          const char* func, unsigned kind, const void* arg, size_t arglen,
          SemanticTag tag, const void* buffer, size_t size, bool is_mutable,
          bool& global, const void* arg2 = nullptr,
          size_t arg2len = 0) override;
      virtual void post_semantic_attach(void) override;
      virtual void push_exception_handler(ExceptionHandlerID handler) override;
      virtual Future pop_exception_handler(Provenance* provenance) override;
    public:
      virtual EquivalenceSet* create_initial_equivalence_set(
          unsigned idx1, const RegionRequirement& req) override;
      virtual void refine_equivalence_sets(
          unsigned req_index, IndexSpaceNode* node,
          const FieldMask& refinement_mask,
          std::vector<RtEvent>& applied_events, bool sharded = false,
          bool first = true,
          const CollectiveMapping* mapping = nullptr) override;
      virtual void find_trace_local_sets(
          unsigned req_index, const FieldMask& mask,
          std::map<EquivalenceSet*, unsigned>& current_sets,
          IndexSpaceNode* node = nullptr,
          const CollectiveMapping* mapping = nullptr) override;
      virtual void receive_created_region_contexts(
          const std::vector<RegionNode*>& created_regions,
          const std::vector<EqKDTree*>& created_trees,
          std::set<RtEvent>& applied_events, const ShardMapping* mapping,
          ShardID source_shard) override;
      bool compute_shard_to_shard_mapping(
          const ShardMapping& src_mapping,
          std::multimap<ShardID, ShardID>& src_to_dst_mapping) const;
      void handle_created_region_contexts(
          Deserializer& derez, std::set<RtEvent>& applied_events);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(
          const Domain& domain, bool take_ownership, TypeTag type_tag,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space(
          const Future& future, TypeTag type_tag,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space(
          const std::vector<DomainPoint>& points,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space(
          const std::vector<Domain>& rects, Provenance* provenance) override;
      virtual IndexSpace create_unbound_index_space(
          TypeTag type_tag, Provenance* provenance) override;
    protected:
      IndexSpace create_index_space_replicated(
          const Domain& bounds, TypeTag type_tag, Provenance* provenance,
          bool take_ownership);
    public:
      virtual IndexSpace union_index_spaces(
          const std::vector<IndexSpace>& spaces,
          Provenance* provenance) override;
      virtual IndexSpace intersect_index_spaces(
          const std::vector<IndexSpace>& spaces,
          Provenance* provenance) override;
      virtual IndexSpace subtract_index_spaces(
          IndexSpace left, IndexSpace right, Provenance* provenance) override;
      virtual void create_shared_ownership(IndexSpace handle) override;
      virtual void destroy_index_space(
          IndexSpace handle, const bool unordered, const bool recurse,
          Provenance* provenance) override;
      virtual void create_shared_ownership(IndexPartition handle) override;
      virtual void destroy_index_partition(
          IndexPartition handle, const bool unordered, const bool recurse,
          Provenance* provenance) override;
      virtual IndexPartition create_equal_partition(
          IndexSpace parent, IndexSpace color_space, size_t granularity,
          Color color, Provenance* provenance) override;
      virtual IndexPartition create_partition_by_weights(
          IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
          size_t granularity, Color color, Provenance* provenance) override;
      virtual IndexPartition create_partition_by_union(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition partition, PartitionKind kind,
          Color color, bool dominates, Provenance* provenance) override;
      virtual IndexPartition create_partition_by_difference(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) override;
      virtual Color create_cross_product_partitions(
          IndexPartition handle1, IndexPartition handle2,
          std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
          Color color, Provenance* provenance) override;
      virtual void create_association(
          LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
          IndexSpace range, MapperID id, MappingTagID tag,
          const UntypedBuffer& marg, Provenance* provenance) override;
      virtual IndexPartition create_restricted_partition(
          IndexSpace parent, IndexSpace color_space, const void* transform,
          size_t transform_size, const void* extent, size_t extent_size,
          PartitionKind part_kind, Color color,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_domain(
          IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
          bool perform_intersections, PartitionKind part_kind, Color color,
          Provenance* provenance, bool skip_check = false) override;
      virtual IndexPartition create_partition_by_field(
          LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
          IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
          PartitionKind part_kind, const UntypedBuffer& marg,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_image(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_image_range(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_preimage(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance) override;
      virtual IndexPartition create_partition_by_preimage_range(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* provenance) override;
      virtual IndexPartition create_pending_partition(
          IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
          Color color, Provenance* provenance, bool trust = false) override;
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle,
          Provenance* provenance) override;
      virtual IndexSpace create_index_space_difference(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexSpace initial,
          const std::vector<IndexSpace>& handles,
          Provenance* provenance) override;
      virtual void verify_partition(
          IndexPartition pid, PartitionKind kind,
          const char* function_name) override;
      virtual FieldSpace create_field_space(Provenance* provenance) override;
      virtual FieldSpace create_field_space(
          const std::vector<size_t>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance) override;
      virtual FieldSpace create_field_space(
          const std::vector<Future>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance) override;
      FieldSpace create_replicated_field_space(
          Provenance* provenance, ShardID* creator_shard = nullptr);
      virtual void create_shared_ownership(FieldSpace handle) override;
      virtual void destroy_field_space(
          FieldSpace handle, const bool unordered,
          Provenance* provenance) override;
      virtual FieldID allocate_field(
          FieldSpace space, size_t field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) override;
      virtual FieldID allocate_field(
          FieldSpace space, const Future& field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) override;
      virtual void free_field(
          FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
          const bool unordered, Provenance* provenance) override;
      virtual void allocate_fields(
          FieldSpace space, const std::vector<size_t>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) override;
      virtual void allocate_fields(
          FieldSpace space, const std::vector<Future>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) override;
      virtual void free_fields(
          FieldAllocatorImpl* allocator, FieldSpace space,
          const std::set<FieldID>& to_free, const bool unordered,
          Provenance* provenance) override;
      virtual LogicalRegion create_logical_region(
          IndexSpace index_space, FieldSpace field_space, const bool task_local,
          Provenance* provenance, const bool output_region = false) override;
      virtual void create_shared_ownership(LogicalRegion handle) override;
      virtual void destroy_logical_region(
          LogicalRegion handle, const bool unordered,
          Provenance* provenance) override;
      virtual void reset_equivalence_sets(
          LogicalRegion parent, LogicalRegion region,
          const std::set<FieldID>& fields) override;
    public:
      virtual FieldAllocatorImpl* create_field_allocator(
          FieldSpace handle, bool unordered) override;
      virtual void destroy_field_allocator(FieldSpaceNode* node) override;
    public:
      void initialize_unordered_collective(void);
      void finalize_unordered_collective(AutoLock& d_lock);
      virtual void insert_unordered_ops(AutoLock& d_lock) override;
      virtual void progress_unordered_operations(
          bool end_task = false) override;
      virtual unsigned minimize_repeat_results(
          unsigned ready, bool& double_wait_interval) override;
      virtual Future execute_task(
          const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
          Provenance* provenance) override;
      virtual FutureMap execute_index_space(
          const IndexTaskLauncher& launcher,
          std::vector<OutputRequirement>* outputs,
          Provenance* provenance) override;
      virtual Future execute_index_space(
          const IndexTaskLauncher& launcher, ReductionOpID redop,
          bool deterministic, std::vector<OutputRequirement>* outputs,
          Provenance* provenance) override;
      virtual Future reduce_future_map(
          const FutureMap& future_map, ReductionOpID redop, bool deterministic,
          MapperID map_id, MappingTagID tag, Provenance* provenance,
          Future initial_value) override;
      using InnerContext::construct_future_map;
      virtual FutureMap construct_future_map(
          IndexSpace space, const std::map<DomainPoint, UntypedBuffer>& data,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true) override;
      virtual FutureMap construct_future_map(
          IndexSpace space, const std::map<DomainPoint, Future>& futures,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true) override;
      virtual PhysicalRegion map_region(
          const InlineLauncher& launcher, Provenance* provenance) override;
      virtual ApEvent remap_region(
          const PhysicalRegion& region, Provenance* provenance,
          bool internal = false) override;
      // Unmapping region is the same as for an inner context
      virtual void fill_fields(
          const FillLauncher& launcher, Provenance* provenance) override;
      virtual void fill_fields(
          const IndexFillLauncher& launcher, Provenance* provenance) override;
      virtual void discard_fields(
          const DiscardLauncher& launcher, Provenance* provenance) override;
      virtual void issue_copy(
          const CopyLauncher& launcher, Provenance* provenance) override;
      virtual void issue_copy(
          const IndexCopyLauncher& launcher, Provenance* provenance) override;
      virtual void issue_acquire(
          const AcquireLauncher& launcher, Provenance* provenance) override;
      virtual void issue_release(
          const ReleaseLauncher& launcher, Provenance* provenance) override;
      virtual PhysicalRegion attach_resource(
          const AttachLauncher& launcher, Provenance* provenance) override;
      virtual ExternalResources attach_resources(
          const IndexAttachLauncher& launcher, Provenance* provenance) override;
      virtual RegionTreeNode* compute_index_attach_upper_bound(
          const IndexAttachLauncher& launcher,
          const std::vector<unsigned>& indexes) override;
      virtual Future detach_resource(
          PhysicalRegion region, const bool flush, const bool unordered,
          Provenance* provenance = nullptr) override;
      virtual Future detach_resources(
          ExternalResources resources, const bool flush, const bool unordered,
          Provenance* provenance) override;
      virtual FutureMap execute_must_epoch(
          const MustEpochLauncher& launcher, Provenance* provenance) override;
      virtual Future issue_timing_measurement(
          const TimingLauncher& launcher, Provenance* provenance) override;
      virtual Future select_tunable_value(
          const TunableLauncher& launcher, Provenance* provenance) override;
      virtual Future issue_mapping_fence(Provenance* provenance) override;
      virtual Future issue_execution_fence(Provenance* provenance) override;
      virtual void begin_trace(
          TraceID tid, bool logical_only, bool static_trace,
          const std::set<RegionTreeID>* managed, bool dep,
          Provenance* provenance) override;
      virtual void end_trace(
          TraceID tid, bool deprecated, Provenance* provenance) override;
      virtual void wait_on_future(FutureImpl* future, RtEvent ready) override;
      virtual void wait_on_future_map(
          FutureMapImpl* map, RtEvent ready) override;
      virtual void end_task(
          const void* res, size_t res_size, bool owned, PhysicalInstance inst,
          FutureFunctor* callback_future,
          const Realm::ExternalInstanceResource* resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          const void* metadataptr, size_t metadatasize,
          ApEvent effects) override;
      virtual void post_end_task(void) override;
      virtual bool add_to_dependence_queue(
          Operation* op,
          const std::vector<StaticDependence>* dependences = nullptr,
          bool unordered = false, bool outermost = true) override;
      virtual FenceOp* initialize_trace_completion(Provenance* prov) override;
      virtual PredicateImpl* create_predicate_impl(Operation* op) override;
      virtual CollectiveResult* find_or_create_collective_view(
          RegionTreeID tid, const std::vector<DistributedID>& instances,
          RtEvent& ready) override;
    public:
      virtual ProjectionSummary* construct_projection_summary(
          Operation* op, unsigned index, const RegionRequirement& req,
          LogicalState* owner, const ProjectionInfo& proj_info) override;
      virtual bool has_interfering_shards(
          ProjectionSummary* one, ProjectionSummary* two,
          bool& dominates) override;
      virtual bool match_timeouts(
          std::vector<LogicalUser*>& timeouts,
          TimeoutMatchExchange*& exchange) override;
    public:
      virtual std::pair<bool, bool> has_pointwise_dominance(
          ProjectionSummary* one, ProjectionSummary* two) override;
      virtual RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_index =
              std::optional<unsigned>()) override;
    public:
      virtual FillView* find_or_create_fill_view(
          FillOp* op, const void* value, size_t value_size,
          RtEvent& ready) override;
      virtual FillView* find_or_create_fill_view(
          FillOp* op, const Future& future, bool& set_value,
          RtEvent& ready) override;
      void finalize_collective_fill_view(
          DistributedID view_did, void* allocation, UniqueID op_uid,
          const void* value, size_t value_size,
          CreateCollectiveFillView* collective);
    public:
      virtual Lock create_lock(void) override;
      virtual void destroy_lock(Lock l) override;
      virtual Grant acquire_grant(
          const std::vector<LockRequest>& requests) override;
      virtual void release_grant(Grant grant) override;
    public:
      virtual PhaseBarrier create_phase_barrier(unsigned arrivals) override;
      virtual void destroy_phase_barrier(PhaseBarrier pb) override;
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier pb) override;
    public:
      virtual DynamicCollective create_dynamic_collective(
          unsigned arrivals, ReductionOpID redop, const void* init_value,
          size_t init_size) override;
      virtual void destroy_dynamic_collective(DynamicCollective dc) override;
      virtual void arrive_dynamic_collective(
          DynamicCollective dc, const void* buffer, size_t size,
          unsigned count) override;
      virtual void defer_dynamic_collective_arrival(
          DynamicCollective dc, const Future& future, unsigned count) override;
      virtual Future get_dynamic_collective_result(
          DynamicCollective dc, Provenance* provenance) override;
      virtual DynamicCollective advance_dynamic_collective(
          DynamicCollective dc) override;
    public:
#ifdef LEGION_DEBUG_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(
          Operation* op, RegionTreeNode* node) override;
      virtual RefinementOp* get_refinement_op(
          Operation* op, RegionTreeNode* node) override;
#else
      virtual MergeCloseOp* get_merge_close_op(void) override;
      virtual RefinementOp* get_refinement_op(void) override;
#endif
    public:
      virtual void pack_task_context(Serializer& rez) const;
    public:
      virtual void pack_remote_context(
          Serializer& rez, AddressSpaceID target,
          bool replicate = false) override;
    public:
      void handle_collective_message(Deserializer& derez);
      void register_rendezvous(ShardRendezvous* rendezvous);
      void handle_rendezvous_message(Deserializer& derez);
      void handle_resource_update(
          Deserializer& derez, std::set<RtEvent>& applied);
      void handle_trace_update(Deserializer& derez, AddressSpaceID source);
      void handle_find_trace_local_sets(
          Deserializer& derez, AddressSpaceID source);
      ApBarrier handle_find_trace_shard_event(
          size_t temp_index, ApEvent event, ShardID remote_shard);
      ApBarrier handle_find_trace_shard_frontier(
          size_t temp_index, ApEvent event, ShardID remote_shard);
      void record_intra_space_dependence(
          uint64_t context_index, const DomainPoint& point,
          RtEvent point_mapped, ShardID next_shard);
      void handle_intra_space_dependence(Deserializer& derez);
    public:
      void increase_pending_index_spaces(unsigned count, bool double_buffer);
      void increase_pending_partitions(unsigned count, bool double_buffer);
      void increase_pending_field_spaces(unsigned count, bool double_buffer);
      void increase_pending_fields(unsigned count, bool double_buffer);
      void increase_pending_region_trees(unsigned count, bool double_buffer);
      void increase_pending_distributed_ids(unsigned count, bool double_buffer);
      DistributedID get_next_distributed_id(void);
      bool create_shard_partition(
          Operation* op, IndexPartition& pid, IndexSpace parent,
          IndexSpace color_space, Provenance* provenance,
          PartitionKind part_kind, LegionColor partition_color,
          bool color_generated);
    public:
      // Collective methods
      CollectiveID get_next_collective_index(
          CollectiveIndexLocation loc, bool logical = false);
      void register_collective(ShardCollective* collective);
      ShardCollective* find_or_buffer_collective(Deserializer& derez);
      void unregister_collective(ShardCollective* collective);
      ShardRendezvous* find_or_buffer_rendezvous(Deserializer& derez);
    public:
      // Physical template methods
      size_t register_trace_template(ShardedPhysicalTemplate* phy_template);
      ShardedPhysicalTemplate* find_or_buffer_trace_update(
          Deserializer& derez, AddressSpaceID source);
      void unregister_trace_template(size_t template_index);
    public:
      // Support for making equivalence sets (logical analysis stage only)
      ShardID get_next_equivalence_set_origin(void);
      virtual RtEvent compute_equivalence_sets(
          unsigned req_index, const std::vector<EqSetTracker*>& targets,
          const std::vector<AddressSpaceID>& target_spaces,
          AddressSpaceID creation_target_space, IndexSpaceExpression* expr,
          const FieldMask& mask) override;
      virtual RtEvent record_output_equivalence_set(
          EqSetTracker* source, AddressSpaceID source_space, unsigned req_index,
          EquivalenceSet* set, const FieldMask& mask) override;
      virtual EqKDTree* create_equivalence_set_kd_tree(
          IndexSpaceNode* node) override;
      void handle_compute_equivalence_sets(Deserializer& derez);
      void handle_output_equivalence_set(Deserializer& derez);
      void handle_refine_equivalence_sets(Deserializer& derez);
    public:
      // Support for output region offset computations
      void handle_output_offset(Deserializer& derez);
      void find_pending_output_offsets(
          uint64_t context_index, unsigned index,
          std::vector<std::optional<std::pair<size_t, size_t> > >& offsets);
    public:
      // Fence barrier methods
      inline RtBarrier get_next_mapping_fence_barrier(void)
      {
        return mapping_fence_barrier.next(this);
      }
      inline ApBarrier get_next_execution_fence_barrier(void)
      {
        return execution_fence_barrier.next(this);
      }
      inline RtBarrier get_next_must_epoch_mapped_barrier(void)
      {
        return must_epoch_mapped_barrier.next(this);
      }
      inline RtBarrier get_next_resource_return_barrier(void)
      {
        return resource_return_barrier.next(this);
      }
      inline RtBarrier get_next_summary_fence_barrier(void)
      {
        return summary_fence_barrier.next(this);
      }
      inline RtBarrier get_next_deletion_ready_barrier(void)
      {
        return deletion_ready_barrier.next(this);
      }
      inline RtBarrier get_next_deletion_mapping_barrier(void)
      {
        return deletion_mapping_barrier.next(this);
      }
      inline RtBarrier get_next_deletion_execution_barrier(void)
      {
        return deletion_execution_barrier.next(this);
      }
      inline ApBarrier get_next_detach_effects_barrier(void)
      {
        return detach_effects_barrier.next(this);
      }
      inline RtBarrier get_next_future_wait_barrier(void)
      {
        return future_wait_barrier.next(this);
      }
      inline RtBarrier get_next_dependent_partition_mapping_barrier(void)
      {
        return dependent_partition_mapping_barrier.next(this);
      }
      inline ApBarrier get_next_dependent_partition_execution_barrier(void)
      {
        return dependent_partition_execution_barrier.next(this);
      }
      inline RtBarrier get_next_attach_resource_barrier(void)
      {
        return attach_resource_barrier.next(this);
      }
      inline RtBarrier get_next_output_regions_barrier(void)
      {
        return output_regions_barrier.next(this);
      }
      inline RtBarrier get_next_close_mapped_barrier(void)
      {
        const RtBarrier result =
            close_mapped_barriers[next_close_mapped_bar_index++].next(this);
        if (next_close_mapped_bar_index == close_mapped_barriers.size())
          next_close_mapped_bar_index = 0;
        return result;
      }
      inline RtBarrier get_next_refinement_mapped_barrier(void)
      {
        const RtBarrier result =
            refinement_mapped_barriers[next_refinement_mapped_bar_index++].next(
                this);
        if (next_refinement_mapped_bar_index ==
            refinement_mapped_barriers.size())
          next_refinement_mapped_bar_index = 0;
        return result;
      }
      inline RtBarrier get_next_refinement_barrier(void)
      {
        const RtBarrier result =
            refinement_ready_barriers[next_refinement_ready_bar_index++].next(
                this);
        if (next_refinement_ready_bar_index == refinement_ready_barriers.size())
          next_refinement_ready_bar_index = 0;
        return result;
      }
      // Note this method always returns two barrier generations
      inline RtBarrier get_next_collective_map_barriers(void)
      {
        // Realm phase barriers do not have an even number of maximum
        // phases so we need to handle the case where the names for the
        // two barriers are not the same. If that occurs then we need
        // finish off the old barrier and use the next one
        RtBarrier result =
            collective_map_barriers[next_collective_map_bar_index].next(this);
        RtBarrier next =
            collective_map_barriers[next_collective_map_bar_index].next(this);
        if (result != Runtime::get_previous_phase(next))
        {
          // Finish off the old barrier
          runtime->phase_barrier_arrive(result, 1);
          result = next;
          next =
              collective_map_barriers[next_collective_map_bar_index].next(this);
          legion_assert(result == Runtime::get_previous_phase(next));
        }
        if (++next_collective_map_bar_index == collective_map_barriers.size())
          next_collective_map_bar_index = 0;
        return result;
      }
      // Note this method always returns two barrier generations
      inline ApBarrier get_next_indirection_barriers(void)
      {
        // Realm phase barriers do not have an even number of maximum
        // phases so we need to handle the case where the names for the
        // two barriers are not the same. If that occurs then we need
        // finish off the old barrier and use the next one
        ApBarrier result =
            indirection_barriers[next_indirection_bar_index].next(this);
        ApBarrier next =
            indirection_barriers[next_indirection_bar_index].next(this);
        if (result != Runtime::get_previous_phase(next))
        {
          // Finish off the old barrier
          runtime->phase_barrier_arrive(result, 1);
          result = next;
          next = indirection_barriers[next_indirection_bar_index].next(this);
          legion_assert(result == Runtime::get_previous_phase(next));
        }
        if (++next_indirection_bar_index == indirection_barriers.size())
          next_indirection_bar_index = 0;
        return result;
      }
    protected:
#ifdef LEGION_DEBUG_COLLECTIVES
      // Versions of the methods below but with reduction initialization
      bool create_new_replicate_barrier(
          RtBarrier& bar, ReductionOpID redop, const void* init,
          size_t init_size, size_t arrivals);
      bool create_new_replicate_barrier(
          ApBarrier& bar, ReductionOpID redop, const void* init,
          size_t init_size, size_t arrivals);
      // This one can only be called inside the logical dependence analysis
      bool create_new_logical_barrier(
          RtBarrier& bar, ReductionOpID redop, const void* init,
          size_t init_size, size_t arrivals);
      bool create_new_logical_barrier(
          ApBarrier& bar, ReductionOpID redop, const void* init,
          size_t init_size, size_t arrivals);
#else
      // These can only be called inside the task for this context
      // since they assume that all the shards are aligned and doing
      // the same calls for the same operations in the same order
      bool create_new_replicate_barrier(RtBarrier& bar, size_t arrivals);
      bool create_new_replicate_barrier(ApBarrier& bar, size_t arrivals);
      // This one can only be called inside the logical dependence analysis
      bool create_new_logical_barrier(RtBarrier& bar, size_t arrivals);
      bool create_new_logical_barrier(ApBarrier& bar, size_t arrivals);
#endif
    public:
      const DomainPoint& get_shard_point(void) const;
      ShardedPhysicalTemplate* find_current_shard_template(TraceID tid) const;
    public:
      static void register_attach_detach_sharding_functor(void);
      ShardingFunction* get_attach_detach_sharding_function(void);
      IndexSpaceNode* compute_index_attach_launch_spaces(
          std::vector<size_t>& shard_sizes, Provenance* provenance);
      static void register_universal_sharding_functor(void);
      ShardingFunction* get_universal_sharding_function(void);
    public:
      void hash_future(
          HashVerifier& hasher, const unsigned safe_level, const Future& future,
          const char* description) const;
      static void hash_future_map(
          HashVerifier& hasher, const FutureMap& map, const char* description);
      static void hash_index_space_requirements(
          HashVerifier& hasher,
          const std::vector<IndexSpaceRequirement>& index_requirements);
      static void hash_region_requirements(
          HashVerifier& hasher,
          const std::vector<RegionRequirement>& region_requirements);
      static void hash_output_requirements(
          HashVerifier& hasher,
          const std::vector<OutputRequirement>& output_requirements);
      static void hash_grants(
          HashVerifier& hasher, const std::vector<Grant>& grants);
      static void hash_phase_barriers(
          HashVerifier& hasher,
          const std::vector<PhaseBarrier>& phase_barriers);
      static void hash_argument(
          HashVerifier& hasher, const unsigned safe_level,
          const UntypedBuffer& arg, const char* description);
      static void hash_predicate(
          HashVerifier& hasher, const Predicate& pred, const char* description);
      static void hash_static_dependences(
          HashVerifier& hasher,
          const std::vector<StaticDependence>* dependences);
      void hash_task_launcher(
          HashVerifier& hasher, const unsigned safe_level,
          const TaskLauncher& launcher) const;
      void hash_index_launcher(
          HashVerifier& hasher, const unsigned safe_level,
          const IndexTaskLauncher& launcher);
      void hash_execution_constraints(
          HashVerifier& hasher, const ExecutionConstraintSet& constraints);
      void hash_layout_constraints(
          HashVerifier& hasher, const LayoutConstraintSet& constraints,
          bool hash_pointers);
    public:
      ShardTask* const owner_shard;
      ShardManager* const shard_manager;
      const size_t total_shards;
    protected:
      // Need an extra trace lock here for when we go to look-up traces
      // to find the current template
      mutable LocalLock trace_lock;
    protected:
      typedef ReplBarrier<RtBarrier, false> RtReplBar;
      typedef ReplBarrier<ApBarrier, false> ApReplBar;
      typedef ReplBarrier<ApBarrier, false, true> ApReplSingleBar;
      typedef ReplBarrier<RtBarrier, false, true> RtReplSingleBar;
      typedef ReplBarrier<RtBarrier, true> RtLogicalBar;
      typedef ReplBarrier<ApBarrier, true> ApLogicalBar;
      // These barriers are used to identify when close operations are mapped
      std::vector<RtLogicalBar> close_mapped_barriers;
      unsigned next_close_mapped_bar_index;
      // These barriers are used to identify when refinement ops are ready
      std::vector<RtLogicalBar> refinement_ready_barriers;
      unsigned next_refinement_ready_bar_index;
      // These barriers are used to identify when refinement ops are mapped
      std::vector<RtLogicalBar> refinement_mapped_barriers;
      unsigned next_refinement_mapped_bar_index;
      // These barriers are for signaling when indirect copies are done
      std::vector<ApReplBar> indirection_barriers;
      unsigned next_indirection_bar_index;
      // These barriers are used to identify pre and post conditions for
      // exclusive collective mapping operations
      std::vector<RtLogicalBar> collective_map_barriers;
      unsigned next_collective_map_bar_index;
    protected:
      std::map<std::pair<uint64_t, DomainPoint>, IntraSpaceDeps>
          intra_space_deps;
    protected:
      // Store the global owner shard and local owner shard for allocation
      std::map<FieldSpace, std::pair<ShardID, bool> >
          field_allocator_owner_shards;
    protected:
      ShardID distributed_id_allocator_shard;
      ShardID index_space_allocator_shard;
      ShardID index_partition_allocator_shard;
      ShardID field_space_allocator_shard;
      ShardID field_allocator_shard;
      ShardID logical_region_allocator_shard;
      ShardID dynamic_id_allocator_shard;
      ShardID equivalence_set_allocator_shard;
    protected:
      RtReplBar creation_barrier;
      RtLogicalBar deletion_ready_barrier;
      RtLogicalBar deletion_mapping_barrier;
      RtLogicalBar deletion_execution_barrier;
      RtReplBar attach_resource_barrier;
      ApLogicalBar detach_effects_barrier;
      RtLogicalBar mapping_fence_barrier;
      RtReplBar must_epoch_mapped_barrier;
      RtReplBar resource_return_barrier;
      RtLogicalBar summary_fence_barrier;
      ApLogicalBar execution_fence_barrier;
      RtReplBar dependent_partition_mapping_barrier;
      ApLogicalBar dependent_partition_execution_barrier;
      RtReplBar semantic_attach_barrier;
      RtReplBar future_wait_barrier;
      RtReplBar inorder_barrier;
      RtReplBar output_regions_barrier;
#ifdef LEGION_DEBUG_COLLECTIVES
    protected:
      RtReplBar collective_check_barrier;
      RtLogicalBar logical_check_barrier;
      RtLogicalBar close_check_barrier;
      RtLogicalBar refinement_check_barrier;
      bool collective_guard_reentrant;
      bool logical_guard_reentrant;
#endif
    protected:
      // local barriers to this context for handling returned
      // resources from sub-tasks
      RtBarrier returned_resource_ready_barrier;
      RtBarrier returned_resource_mapped_barrier;
      RtBarrier returned_resource_execution_barrier;
    protected:
      int shard_collective_radix;
      int shard_collective_log_radix;
      int shard_collective_stages;
      int shard_collective_participating_shards;
      int shard_collective_last_radix;
    protected:
      mutable LocalLock replication_lock;
      CollectiveID next_available_collective_index;
      // We also need to create collectives in the logical dependence
      // analysis stage of the pipeline. We'll have those count on the
      // odd numbers of the collective IDs whereas the ones from the
      // application task will be the even numbers.
      CollectiveID next_logical_collective_index;
      std::map<CollectiveID, ShardCollective*> collectives;
      std::map<CollectiveID, std::vector<std::pair<void*, size_t> > >
          pending_collective_updates;
    protected:
      std::map<ShardID, ShardRendezvous*> shard_rendezvous;
      std::map<ShardID, std::vector<std::pair<void*, size_t> > >
          pending_rendezvous_updates;
    protected:
      // Pending allocations of various resources
      std::deque<std::pair<ValueBroadcast<ISBroadcast>*, bool> >
          pending_index_spaces;
      std::deque<std::pair<ValueBroadcast<IPBroadcast>*, ShardID> >
          pending_index_partitions;
      std::deque<std::pair<ValueBroadcast<FSBroadcast>*, bool> >
          pending_field_spaces;
      std::deque<std::pair<ValueBroadcast<FIDBroadcast>*, bool> >
          pending_fields;
      std::deque<std::pair<ValueBroadcast<LRBroadcast>*, bool> >
          pending_region_trees;
      std::deque<std::pair<ValueBroadcast<DIDBroadcast>*, bool> >
          pending_distributed_ids;
      unsigned pending_index_space_check;
      unsigned pending_index_partition_check;
      unsigned pending_field_space_check;
      unsigned pending_field_check;
      unsigned pending_region_tree_check;
      unsigned pending_distributed_id_check;
    protected:
      std::map<size_t, ShardedPhysicalTemplate*> physical_templates;
      struct PendingTemplateUpdate {
      public:
        PendingTemplateUpdate(void) : ptr(nullptr), size(0), source(0) { }
        PendingTemplateUpdate(void* p, size_t s, AddressSpaceID src)
          : ptr(p), size(s), source(src)
        { }
      public:
        void* ptr;
        size_t size;
        AddressSpaceID source;
      };
      std::map<size_t /*template index*/, std::vector<PendingTemplateUpdate> >
          pending_template_updates;
      size_t next_physical_template_index;
    protected:
      // Different from pending_top_views as this applies to our requests
      std::map<PhysicalManager*, RtUserEvent> pending_request_views;
      std::map<RegionTreeID, RtUserEvent> pending_tree_requests;
    protected:
      std::map<std::pair<unsigned, unsigned>, RtBarrier> ready_clone_barriers;
      std::map<std::pair<unsigned, unsigned>, RtUserEvent>
          pending_clone_barriers;
    protected:
      // Pending output offsets for tasks that haven't been registered
      // Should be at most one entry for each point task on each dimension
      std::map<
          std::pair<uint64_t, unsigned>,
          std::vector<std::optional<std::pair<size_t, size_t> > > >
          pending_output_offsets;
    protected:
      struct AttachLaunchSpace {
      public:
        AttachLaunchSpace(IndexSpaceNode* node) : launch_space(node) { }
      public:
        IndexSpaceNode* const launch_space;
        std::vector<size_t> shard_sizes;
      };
      std::vector<AttachLaunchSpace*> index_attach_launch_spaces;
    protected:
      unsigned next_replicate_bar_index;
      unsigned next_logical_bar_index;
    protected:
      static const unsigned MIN_UNORDERED_OPS_EPOCH = 32;
      static const unsigned MAX_UNORDERED_OPS_EPOCH = 32768;
      unsigned unordered_ops_counter;
      unsigned unordered_ops_epoch;
      UnorderedExchange* unordered_collective;
    protected:
      // Collective for auto-tracing to determine the number of minimum
      // number of repeat jobs that are ready across all the shards
      AllReduceCollective<MinReduction<unsigned>, false>*
          minimize_repeats_collective;
    protected:
      // Extra data structures for the fill view cache
      mutable LocalLock fill_view_lock;
      std::map<FillView*, size_t> pending_fill_views;
    };

    /**
     * \class ConsensusMatchExchange
     * This is collective for performing a consensus exchange between
     * the shards for a collection of values.
     */
    class ConsensusMatchExchange : public AllGatherCollective<false> {
    public:
      struct ElementComparator {
      public:
        ElementComparator(size_t size) : element_size(size) { }
      public:
        inline bool operator()(const void* lhs, const void* rhs) const
        {
          return (std::memcmp(lhs, rhs, element_size) < 0);
        }
      public:
        const size_t element_size;
      };
    public:
      ConsensusMatchExchange(
          ReplicateContext* ctx, CollectiveIndexLocation loc,
          FutureImpl* to_complete, const void* input, void* output,
          size_t element_size, size_t num_elements);
      ConsensusMatchExchange(const ConsensusMatchExchange& rhs) = delete;
      virtual ~ConsensusMatchExchange(void);
    public:
      ConsensusMatchExchange& operator=(const ConsensusMatchExchange& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CONSENSUS_MATCH;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    protected:
      FutureImpl* to_complete;
      const size_t total_elements;
      const void* const input;
      void* const output;
      const size_t element_size;
      std::vector<const void*> valid_elements;
    };

    /**
     * \class VerifyReplicableExchange
     * This class exchanges hash values of all the inputs for calls
     * into control replication contexts in order to ensure that they
     * all are the same.
     */
    class VerifyReplicableExchange : public AllGatherCollective<false> {
    public:
      VerifyReplicableExchange(
          CollectiveIndexLocation loc, ReplicateContext* ctx);
      VerifyReplicableExchange(const VerifyReplicableExchange& rhs) = delete;
      virtual ~VerifyReplicableExchange(void);
    public:
      VerifyReplicableExchange& operator=(const VerifyReplicableExchange& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_VERIFY_CONTROL_REPLICATION_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      typedef std::map<std::pair<uint64_t, uint64_t>, ShardID> ShardHashes;
      const ShardHashes& exchange(const uint64_t hash[2]);
    public:
      ShardHashes unique_hashes;
    };

    /**
     * \class CrossProductCollective
     * A class for exchanging the names of partitions created by
     * a call for making cross-product partitions
     */
    class CrossProductCollective : public AllGatherCollective<false> {
    public:
      CrossProductCollective(
          ReplicateContext* ctx, CollectiveIndexLocation loc);
      CrossProductCollective(const CrossProductCollective& rhs) = delete;
      virtual ~CrossProductCollective(void);
    public:
      CrossProductCollective& operator=(const CrossProductCollective& rhs) =
          delete;
    public:
      void exchange_partitions(std::map<IndexSpace, IndexPartition>& handles);
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CROSS_PRODUCT_PARTITION;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    protected:
      std::map<IndexSpace, IndexPartition> non_empty_handles;
    };

    /**
     * \class UnorderedExchange
     * This is a class that exchanges information about unordered operations
     * that are ready to execute on each shard so that we can determine which
     * operations can be inserted into a task stream
     */
    class UnorderedExchange : public AllGatherCollective<true> {
    public:
      UnorderedExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      UnorderedExchange(const UnorderedExchange& rhs) = delete;
      virtual ~UnorderedExchange(void);
    public:
      UnorderedExchange& operator=(const UnorderedExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_UNORDERED_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void start_unordered_exchange(const std::vector<Operation*>& operations);
      void find_ready_operations(std::vector<Operation*>& ready_operations);
    protected:
      template<typename T>
      void update_future_counts(
          const int stage, std::map<int, std::map<T, unsigned> >& future_counts,
          std::map<T, unsigned>& counts);
      template<typename T>
      void pack_counts(Serializer& rez, const std::map<T, unsigned>& counts);
      template<typename T>
      void unpack_counts(
          const int stage, Deserializer& derez,
          std::map<T, unsigned>& future_counts);
      template<typename T>
      void pack_field_counts(
          Serializer& rez,
          const std::map<std::pair<T, FieldID>, unsigned>& counts);
      template<typename T>
      void unpack_field_counts(
          const int stage, Deserializer& derez,
          std::map<std::pair<T, FieldID>, unsigned>& future_counts);
      template<typename T, typename OP>
      void initialize_counts(
          const std::map<T, OP*>& ops, std::map<T, unsigned>& counts);
      template<typename T, typename OP>
      void find_ready_ops(
          const size_t total_shards, const std::map<T, unsigned>& final_counts,
          const std::map<T, OP*>& ops, std::vector<Operation*>& ready_ops);
    protected:
      std::map<IndexSpace, unsigned> index_space_counts;
      std::map<IndexPartition, unsigned> index_partition_counts;
      std::map<FieldSpace, unsigned> field_space_counts;
      // Use the lowest field ID here as the key
      std::map<std::pair<FieldSpace, FieldID>, unsigned> field_counts;
      std::map<LogicalRegion, unsigned> logical_region_counts;
      // Use the lowest field ID here as the key
      std::map<std::pair<LogicalRegion, FieldID>, unsigned>
          region_detach_counts;
      std::map<std::pair<LogicalPartition, FieldID>, unsigned>
          partition_detach_counts;
    protected:
      std::map<IndexSpace, ReplDeletionOp*> index_space_deletions;
      std::map<IndexPartition, ReplDeletionOp*> index_partition_deletions;
      std::map<FieldSpace, ReplDeletionOp*> field_space_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<FieldSpace, FieldID>, ReplDeletionOp*> field_deletions;
      std::map<LogicalRegion, ReplDeletionOp*> logical_region_deletions;
      // Use the lowest field ID here as the key
      std::map<std::pair<LogicalRegion, FieldID>, Operation*>
          region_detachments;
      std::map<std::pair<LogicalPartition, FieldID>, Operation*>
          partition_detachments;
    };

    /**
     * \class ImplicitShardingFunctor
     * Support the computation of an implicit sharding function for
     * the creation of replicated future maps
     */
    class ImplicitShardingFunctor : public AllGatherCollective<false>,
                                    public ShardingFunctor {
    public:
      ImplicitShardingFunctor(
          ReplicateContext* ctx, CollectiveIndexLocation loc,
          ReplFutureMapImpl* map);
      ImplicitShardingFunctor(const ImplicitShardingFunctor& rhs) = delete;
      virtual ~ImplicitShardingFunctor(void);
    public:
      ImplicitShardingFunctor& operator=(const ImplicitShardingFunctor& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_IMPLICIT_SHARDING_FUNCTOR;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      virtual ShardID shard(
          const DomainPoint& point, const Domain& full_space,
          const size_t total_shards) override;
    protected:
      virtual RtEvent post_complete_exchange(void) override;
    public:
      template<typename T>
      void compute_sharding(const std::map<DomainPoint, T>& points)
      {
        for (typename std::map<DomainPoint, T>::const_iterator it =
                 points.begin();
             it != points.end(); it++)
          implicit_sharding[it->first] = local_shard;
        this->perform_collective_async();
      }
    public:
      ReplFutureMapImpl* const map;
    protected:
      std::map<DomainPoint, ShardID> implicit_sharding;
    };

    /**
     * \class PointwiseAllreduce
     * This class performs an all-reduce on a pair of booleans to wtih
     * an and-allreduce on each entry for supporting pointwise analysis.
     */
    class PointwiseAllreduce : public AllGatherCollective<false> {
    public:
      PointwiseAllreduce(
          ReplicateContext* ctx, CollectiveID id, std::pair<bool, bool>& local);
      PointwiseAllreduce(const PointwiseAllreduce& rhs) = delete;
      virtual ~PointwiseAllreduce(void);
    public:
      PointwiseAllreduce& operator=(const PointwiseAllreduce& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_POINTWISE_ALLREDUCE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    private:
      std::pair<bool, bool>& local;
    };

    /**
     * \class ShardSyncTree
     * A synchronization tree allows one shard to be notified when
     * all the other shards have reached a certain point in the
     * execution of the program.
     */
    class ShardSyncTree : public GatherCollective {
    public:
      ShardSyncTree(
          ReplicateContext* ctx, ShardID origin, CollectiveIndexLocation loc);
      ShardSyncTree(const ShardSyncTree& rhs) = delete;
      virtual ~ShardSyncTree(void);
    public:
      ShardSyncTree& operator=(const ShardSyncTree& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_SHARD_SYNC_TREE;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual RtEvent post_gather(void) override;
    protected:
      std::vector<RtEvent> postconditions;
      const RtEvent done;
    };

    /**
     * \class ShardRendezvous
     * A sharded rendezvous class is similar to a shard collective, but it
     * instead of performing collective operations between all the shards
     * in a control replicated context, it will support doing parallel
     * rendezvous between a subset of shards in the context. Callers must
     * provide a unique key for performing the rendezvous.
     */
    class ShardRendezvous {
    public:
      ShardRendezvous(
          ReplicateContext* ctx, ShardID origin,
          const std::vector<ShardID>& participants);
      ShardRendezvous(const ShardRendezvous& rhs) = delete;
      virtual ~ShardRendezvous(void) { }
    public:
      ShardRendezvous& operator=(const ShardRendezvous& rhs) = delete;
    public:
      virtual bool receive_message(Deserializer& derez) = 0;
    public:
      void prefix_message(Serializer& rez, ShardID target) const;
      void register_rendezvous(void);
      size_t get_total_participants(void) const;
      ShardID get_parent(void) const;
      size_t count_children(void) const;
      void get_children(std::vector<ShardID>& children) const;
    protected:
      unsigned find_index(ShardID shard) const;
      ShardID get_index(unsigned index) const;
      unsigned convert_to_offset(unsigned index, unsigned origin) const;
      unsigned convert_to_index(unsigned offset, unsigned origin) const;
    public:
      ReplicateContext* const context;
      const ShardID origin_shard;
      const ShardID local_shard;
      const std::vector<ShardID>& participants;
      const bool all_shards_participating;
    protected:
      mutable LocalLock rendezvous_lock;
    };

    /**
     * \class TimeoutMatchExchange
     * This class helps perform all all-reduce exchange between the shards
     * to see which logical users that have timed out on their analyses
     * can be collected across all the shards. To be collected all the
     * shards must agree on what they are pruning.
     */
    class TimeoutMatchExchange : public AllGatherCollective<false> {
    public:
      TimeoutMatchExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      TimeoutMatchExchange(const TimeoutMatchExchange& rhs) = delete;
      virtual ~TimeoutMatchExchange(void);
    public:
      TimeoutMatchExchange& operator=(const TimeoutMatchExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_TIMEOUT_MATCH_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void perform_exchange(
          std::vector<std::pair<size_t, unsigned> >& timeouts, bool ready);
      bool complete_exchange(
          std::vector<LogicalUser*>& timeouts,
          std::vector<std::pair<size_t, unsigned> >& remaining);
    protected:
      // Pair represents <context index,region requirement index> for each user
      std::vector<std::pair<size_t, unsigned> > all_timeouts;
      bool double_latency;
    };

    /**
     * \class CrossProductExchange
     * This all-gather exchanges IDs for the creation of replicated
     * partitions when performing a cross-product partition
     */
    class CrossProductExchange : public AllGatherCollective<false> {
    public:
      CrossProductExchange(ReplicateContext* ctx, CollectiveIndexLocation loc);
      CrossProductExchange(const CrossProductExchange& rhs) = delete;
      virtual ~CrossProductExchange(void) { }
    public:
      CrossProductExchange& operator=(const CrossProductExchange& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CROSS_PRODUCT_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
    public:
      void exchange_ids(LegionColor color, IndexPartition pid);
      void sync_child_ids(LegionColor color, IndexPartition& pid);
    protected:
      std::map<LegionColor, IndexPartition> child_ids;
    };

    /**
     * \class CreateCollectiveFillView
     * Broadcast out the distributed ID for a new fill view
     */
    class CreateCollectiveFillView : public BroadcastCollective {
    public:
      CreateCollectiveFillView(
          ReplicateContext* ctx, CollectiveID id, void* allocation,
          UniqueID op_uid, const void* value = nullptr, size_t size = 0);
      CreateCollectiveFillView(const CreateCollectiveFillView& rhs) = delete;
      virtual ~CreateCollectiveFillView(void);
    public:
      CreateCollectiveFillView& operator=(const CreateCollectiveFillView& rhs) =
          delete;
    public:
      inline void broadcast_fill_view_did(DistributedID did)
      {
        view_did = did;
        perform_collective_async();
      }
      bool matches(const void* value, size_t value_size) const;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_CREATE_FILL_VIEW;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual RtEvent post_broadcast(void) override;
    protected:
      void* const allocation;
      void* const value;
      const size_t value_size;
      const UniqueID op_uid;
      DistributedID view_did;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REPLICATE_CONTEXT_H__
