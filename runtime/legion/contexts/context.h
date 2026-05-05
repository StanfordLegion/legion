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

#ifndef __LEGION_CONTEXT_H__
#define __LEGION_CONTEXT_H__

#include "legion/api/exception.h"
#include "legion/kernel/garbage_collection.h"
#include "legion/kernel/metatask.h"
#include "legion/kernel/runtime.h"
#include "legion/api/argument_map.h"
#include "legion/api/constraints.h"
#include "legion/api/functors.h"
#include "legion/api/future_map_impl.h"
#include "legion/api/launchers.h"
#include "legion/api/mapping.h"
#include "legion/api/output_region_impl.h"
#include "legion/tools/profiler.h"
#include "legion/utilities/resources.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TaskContext
     * The base class for all task contexts which
     * provide all the methods for handling the
     * execution of a task at runtime.
     */
    class TaskContext : public DistributedCollectable {
    public:
      // This is a no-op task for yield operations
      struct YieldArgs : public LgTaskArgs<YieldArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_YIELD_TASK_ID;
        static constexpr bool IS_APPLICATION_TASK = true;
      public:
        YieldArgs(void) : LgTaskArgs<YieldArgs>(false, false) { }
        inline void execute(void) const { /*nothing to do*/ }
      };
    public:
      TaskContext(
          SingleTask* owner, int depth,
          const std::vector<RegionRequirement>& reqs,
          const std::vector<OutputRequirement>& output_reqs, DistributedID did,
          bool perform_registration, bool inline_task,
          bool implicit_ctx = false, CollectiveMapping* mapping = nullptr);
      virtual ~TaskContext(void);
    public:
      // This is used enough that we want it inlined
      inline Processor get_executing_processor(void) const
      {
        return executing_processor;
      }
      inline void set_executing_processor(Processor p)
      {
        executing_processor = p;
      }
      inline const char* get_task_name(void) const
      {
        return get_task()->get_task_name();
      }
      inline const std::vector<PhysicalRegion>& get_physical_regions(void) const
      {
        return physical_regions;
      }
      inline SingleTask* get_owner_task(void) const { return owner_task; }
      inline bool is_priority_mutable(void) const { return mutable_priority; }
      inline int get_depth(void) const { return depth; }
      inline uint64_t get_tunable_index(void) { return total_tunable_count++; }
    public:
      virtual ShardID get_shard_id(void) const { return 0; }
      virtual DistributedID get_replication_id(void) const { return 0; }
      virtual size_t get_total_shards(void) const { return 1; }
    public:
      // Interface for task contexts
      virtual ContextID get_logical_tree_context(void) const = 0;
      virtual ContextID get_physical_tree_context(void) const = 0;
      virtual const Task* get_task(void) const;
      virtual UniqueID get_unique_id(void) const;
      virtual InnerContext* find_parent_context(void);
      virtual void compute_task_tree_coordinates(
          TaskTreeCoordinates& coords) const = 0;
      virtual VariantImpl* select_inline_variant(
          TaskOp* child, const std::vector<PhysicalRegion>& parent_regions,
          std::deque<InstanceSet>& physical_instances);
      virtual bool is_leaf_context(void) const;
      virtual bool is_inner_context(void) const;
#ifdef LEGION_USE_LIBDL
      virtual void perform_global_registration_callbacks(
          Realm::DSOReferenceImplementation* dso, const void* buffer,
          size_t buffer_size, bool withargs, size_t dedup_tag,
          RtEvent local_done, RtEvent global_done,
          std::set<RtEvent>& preconditions);
#endif
      virtual void print_once(FILE* f, const char* message) const;
      virtual void log_once(Realm::LoggerMessage& message) const;
      virtual Future from_value(
          const void* value, size_t value_size, bool owned,
          Provenance* provenance, bool shard_local);
      virtual Future from_value(
          const void* value, size_t size, bool owned,
          const Realm::ExternalInstanceResource& resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          Provenance* provenance, bool shard_local);
      virtual Future consensus_match(
          const void* input, void* output, size_t num_elements,
          size_t element_size, Provenance* provenance);
    public:
      virtual VariantID register_variant(
          const TaskVariantRegistrar& registrar, const void* user_data,
          size_t user_data_size, const CodeDescriptor& desc, size_t ret_size,
          bool has_ret_size, VariantID vid, bool check_task_id);
      virtual TraceID generate_dynamic_trace_id(void);
      virtual MapperID generate_dynamic_mapper_id(void);
      virtual ProjectionID generate_dynamic_projection_id(void);
      virtual ShardingID generate_dynamic_sharding_id(void);
      virtual ConcurrentID generate_dynamic_concurrent_id(void);
      virtual ExceptionHandlerID generate_dynamic_exception_handler_id(void);
      virtual TaskID generate_dynamic_task_id(void);
      virtual ReductionOpID generate_dynamic_reduction_id(void);
      virtual CustomSerdezID generate_dynamic_serdez_id(void);
      virtual bool perform_semantic_attach(
          const char* func, unsigned kind, const void* arg, size_t arglen,
          SemanticTag tag, const void* buffer, size_t size, bool is_mutable,
          bool& global, const void* arg2 = nullptr, size_t arg2len = 0);
      virtual void post_semantic_attach(void);
    public:
      virtual void push_exception_handler(ExceptionHandlerID handler);
      virtual Future pop_exception_handler(Provenance* provenance);
      ExceptionHandlerID get_current_exception_handler(void) const;
      void record_task_tree_trace(Exception& exception, Operation* op) const;
    public:
      virtual RtEvent find_pointwise_dependence(
          uint64_t context_index, const DomainPoint& point, ShardID shard,
          bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_region =
              std::optional<unsigned>()) = 0;
      virtual void return_resources(
          ResourceTracker* target, uint64_t return_index,
          std::set<RtEvent>& preconditions) = 0;
      virtual void pack_return_resources(
          Serializer& rez, uint64_t return_index) = 0;
      virtual void log_created_requirements(void) const = 0;
      virtual void report_leaks_and_duplicates(
          std::set<RtEvent>& preconditions) = 0;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(
          const Domain& bounds, bool take_ownership, TypeTag type_tag,
          Provenance* provenance) = 0;
      virtual IndexSpace create_index_space(
          const Future& future, TypeTag type_tag, Provenance* provenance) = 0;
      virtual IndexSpace create_index_space(
          const std::vector<DomainPoint>& points, Provenance* provenance) = 0;
      virtual IndexSpace create_index_space(
          const std::vector<Domain>& rects, Provenance* provenance) = 0;
      // This variant creates an uninitialized index space
      // that later is set by a task
      virtual IndexSpace create_unbound_index_space(
          TypeTag type_tag, Provenance* provenance) = 0;
    public:
      virtual IndexSpace union_index_spaces(
          const std::vector<IndexSpace>& spaces, Provenance* provenance) = 0;
      virtual IndexSpace intersect_index_spaces(
          const std::vector<IndexSpace>& spaces, Provenance* provenance) = 0;
      virtual IndexSpace subtract_index_spaces(
          IndexSpace left, IndexSpace right, Provenance* provenance) = 0;
      virtual void create_shared_ownership(IndexSpace handle) = 0;
      virtual void destroy_index_space(
          IndexSpace handle, const bool unordered, const bool recurse,
          Provenance* provenance) = 0;
      virtual void create_shared_ownership(IndexPartition handle) = 0;
      virtual void destroy_index_partition(
          IndexPartition handle, const bool unordered, const bool recurse,
          Provenance* provenance) = 0;
      virtual IndexPartition create_equal_partition(
          IndexSpace parent, IndexSpace color_space, size_t granularity,
          Color color, Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_weights(
          IndexSpace parent, const FutureMap& weights, IndexSpace color_space,
          size_t granularity, Color color, Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_union(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_intersection(
          IndexSpace parent, IndexPartition partition, PartitionKind kind,
          Color color, bool dominates, Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_difference(
          IndexSpace parent, IndexPartition handle1, IndexPartition handle2,
          IndexSpace color_space, PartitionKind kind, Color color,
          Provenance* provenance) = 0;
      virtual Color create_cross_product_partitions(
          IndexPartition handle1, IndexPartition handle2,
          std::map<IndexSpace, IndexPartition>& handles, PartitionKind kind,
          Color color, Provenance* provenance) = 0;
      virtual void create_association(
          LogicalRegion domain, LogicalRegion domain_parent, FieldID domain_fid,
          IndexSpace range, MapperID id, MappingTagID tag,
          const UntypedBuffer& marg, Provenance* prov) = 0;
      virtual IndexPartition create_restricted_partition(
          IndexSpace parent, IndexSpace color_space, const void* transform,
          size_t transform_size, const void* extent, size_t extent_size,
          PartitionKind part_kind, Color color, Provenance* provenance) = 0;
      virtual IndexPartition create_partition_by_domain(
          IndexSpace parent, const FutureMap& domains, IndexSpace color_space,
          bool perform_intersections, PartitionKind part_kind, Color color,
          Provenance* provenance, bool skip_check = false) = 0;
      virtual IndexPartition create_partition_by_field(
          LogicalRegion handle, LogicalRegion parent_priv, FieldID fid,
          IndexSpace color_space, Color color, MapperID id, MappingTagID tag,
          PartitionKind part_kind, const UntypedBuffer& marg,
          Provenance* prov) = 0;
      virtual IndexPartition create_partition_by_image(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov) = 0;
      virtual IndexPartition create_partition_by_image_range(
          IndexSpace handle, LogicalPartition projection, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov) = 0;
      virtual IndexPartition create_partition_by_preimage(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov) = 0;
      virtual IndexPartition create_partition_by_preimage_range(
          IndexPartition projection, LogicalRegion handle, LogicalRegion parent,
          FieldID fid, IndexSpace color_space, PartitionKind part_kind,
          Color color, MapperID id, MappingTagID tag, const UntypedBuffer& marg,
          Provenance* prov) = 0;
      virtual IndexPartition create_pending_partition(
          IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
          Color color, Provenance* prov, bool trust = false) = 0;
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance) = 0;
      virtual IndexSpace create_index_space_union(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle, Provenance* provenance) = 0;
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, const std::vector<IndexSpace>& handles,
          Provenance* provenance) = 0;
      virtual IndexSpace create_index_space_intersection(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexPartition handle, Provenance* provenance) = 0;
      virtual IndexSpace create_index_space_difference(
          IndexPartition parent, const void* realm_color, size_t color_size,
          TypeTag type_tag, IndexSpace initial,
          const std::vector<IndexSpace>& handles, Provenance* provenance) = 0;
      virtual FieldSpace create_field_space(Provenance* provenance) = 0;
      virtual FieldSpace create_field_space(
          const std::vector<size_t>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance) = 0;
      virtual FieldSpace create_field_space(
          const std::vector<Future>& sizes,
          std::vector<FieldID>& resulting_fields, CustomSerdezID serdez_id,
          Provenance* provenance) = 0;
      virtual void create_shared_ownership(FieldSpace handle) = 0;
      virtual void destroy_field_space(
          FieldSpace handle, const bool unordered, Provenance* provenance) = 0;
      virtual FieldID allocate_field(
          FieldSpace space, size_t field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) = 0;
      virtual FieldID allocate_field(
          FieldSpace space, const Future& field_size, FieldID fid, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) = 0;
      virtual void allocate_local_field(
          FieldSpace space, size_t field_size, FieldID fid,
          CustomSerdezID serdez_id, std::set<RtEvent>& done_events,
          Provenance* provenance) = 0;
      virtual void free_field(
          FieldAllocatorImpl* allocator, FieldSpace space, FieldID fid,
          const bool unordered, Provenance* provenance) = 0;
      virtual void allocate_fields(
          FieldSpace space, const std::vector<size_t>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) = 0;
      virtual void allocate_fields(
          FieldSpace space, const std::vector<Future>& sizes,
          std::vector<FieldID>& resuling_fields, bool local,
          CustomSerdezID serdez_id, Provenance* provenance) = 0;
      virtual void allocate_local_fields(
          FieldSpace space, const std::vector<size_t>& sizes,
          const std::vector<FieldID>& resuling_fields, CustomSerdezID serdez_id,
          std::set<RtEvent>& done_events, Provenance* provenance) = 0;
      virtual void free_fields(
          FieldAllocatorImpl* allocator, FieldSpace space,
          const std::set<FieldID>& to_free, const bool unordered,
          Provenance* provenance) = 0;
      virtual LogicalRegion create_logical_region(
          IndexSpace index_space, FieldSpace field_space, const bool task_local,
          Provenance* provenance, const bool output_region = false) = 0;
      virtual void create_shared_ownership(LogicalRegion handle) = 0;
      virtual void destroy_logical_region(
          LogicalRegion handle, const bool unordered,
          Provenance* provenance) = 0;
      virtual void reset_equivalence_sets(
          LogicalRegion parent, LogicalRegion region,
          const std::set<FieldID>& fields) = 0;
      virtual FieldAllocatorImpl* create_field_allocator(
          FieldSpace handle, bool unordered) = 0;
      virtual void destroy_field_allocator(FieldSpaceNode* node) = 0;
      virtual void get_local_field_set(
          const FieldSpace handle, const std::set<unsigned>& indexes,
          std::set<FieldID>& to_set) const = 0;
      virtual void get_local_field_set(
          const FieldSpace handle, const std::set<unsigned>& indexes,
          std::vector<FieldID>& to_set) const = 0;
    public:
      virtual void add_physical_region(
          const RegionRequirement& req, bool mapped, MapperID mid,
          MappingTagID tag, ApUserEvent& unmap_event, bool virtual_mapped,
          const InstanceSet& physical_instances) = 0;
      virtual Future execute_task(
          const TaskLauncher& launcher, std::vector<OutputRequirement>* outputs,
          Provenance* provenance) = 0;
      virtual FutureMap execute_index_space(
          const IndexTaskLauncher& launcher,
          std::vector<OutputRequirement>* outputs, Provenance* provenance) = 0;
      virtual Future execute_index_space(
          const IndexTaskLauncher& launcher, ReductionOpID redop,
          bool deterministic, std::vector<OutputRequirement>* outputs,
          Provenance* provenance) = 0;
      virtual Future reduce_future_map(
          const FutureMap& future_map, ReductionOpID redop, bool deterministic,
          MapperID map_id, MappingTagID tag, Provenance* provenance,
          Future initial_value) = 0;
      virtual FutureMap construct_future_map(
          IndexSpace domain, const std::map<DomainPoint, UntypedBuffer>& data,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true) = 0;
      virtual FutureMap construct_future_map(
          const Domain& domain,
          const std::map<DomainPoint, UntypedBuffer>& data,
          bool collective = false, ShardingID sid = 0,
          bool implicit = false) = 0;
      virtual FutureMap construct_future_map(
          IndexSpace domain, const std::map<DomainPoint, Future>& futures,
          Provenance* provenance, bool collective = false, ShardingID sid = 0,
          bool implicit = false, bool check_space = true) = 0;
      virtual FutureMap construct_future_map(
          const Domain& domain, const std::map<DomainPoint, Future>& futures,
          bool collective = false, ShardingID sid = 0,
          bool implicit = false) = 0;
      virtual FutureMap transform_future_map(
          const FutureMap& fm, IndexSpace new_domain,
          PointTransformFunctor* functor, bool own_functor,
          Provenance* provenance) = 0;
      virtual PhysicalRegion map_region(
          const InlineLauncher& launcher, Provenance* provenance) = 0;
      virtual ApEvent remap_region(
          const PhysicalRegion& region, Provenance* provenance,
          bool internal = false) = 0;
      virtual void unmap_region(PhysicalRegion region) = 0;
      virtual void unmap_all_regions(bool external = true) = 0;
      virtual void fill_fields(
          const FillLauncher& launcher, Provenance* provenance) = 0;
      virtual void fill_fields(
          const IndexFillLauncher& launcher, Provenance* provenance) = 0;
      virtual void discard_fields(
          const DiscardLauncher& launcher, Provenance* provenance) = 0;
      virtual void issue_copy(
          const CopyLauncher& launcher, Provenance* provenance) = 0;
      virtual void issue_copy(
          const IndexCopyLauncher& launcher, Provenance* provenance) = 0;
      virtual void issue_acquire(
          const AcquireLauncher& launcher, Provenance* provenance) = 0;
      virtual void issue_release(
          const ReleaseLauncher& launcher, Provenance* provenance) = 0;
      virtual PhysicalRegion attach_resource(
          const AttachLauncher& launcher, Provenance* provenance) = 0;
      virtual ExternalResources attach_resources(
          const IndexAttachLauncher& launcher, Provenance* provenance) = 0;
      virtual Future detach_resource(
          PhysicalRegion region, const bool flush, const bool unordered,
          Provenance* provenance = nullptr) = 0;
      virtual Future detach_resources(
          ExternalResources resources, const bool flush, const bool unordered,
          Provenance* provenance) = 0;
      virtual void progress_unordered_operations(bool end_task = false) = 0;
      virtual FutureMap execute_must_epoch(
          const MustEpochLauncher& launcher, Provenance* provenance) = 0;
      virtual Future issue_timing_measurement(
          const TimingLauncher& launcher, Provenance* provenance) = 0;
      virtual Future select_tunable_value(
          const TunableLauncher& launcher, Provenance* provenance) = 0;
      virtual Future issue_mapping_fence(Provenance* provenance) = 0;
      virtual Future issue_execution_fence(Provenance* provenance) = 0;
      virtual void complete_frame(Provenance* provenance) = 0;
      virtual Predicate create_predicate(
          const Future& f, Provenance* provenance) = 0;
      virtual Predicate predicate_not(
          const Predicate& p, Provenance* provenance) = 0;
      virtual Predicate create_predicate(
          const PredicateLauncher& launcher, Provenance* provenance) = 0;
      virtual Future get_predicate_future(
          const Predicate& p, Provenance* provenance) = 0;
    public:
      virtual void begin_trace(
          TraceID tid, bool logical_only, bool static_trace,
          const std::set<RegionTreeID>* managed, bool dep,
          Provenance* provenance) = 0;
      virtual void end_trace(
          TraceID tid, bool deprecated, Provenance* provenance) = 0;
      virtual void record_blocking_call(
          uint64_t future_coordinate, bool invalidate_trace = true) = 0;
      virtual void wait_on_future(FutureImpl* future, RtEvent ready) = 0;
      virtual void wait_on_future_map(FutureMapImpl* map, RtEvent ready) = 0;
    public:
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_top_context(
          InnerContext* previous = nullptr) = 0;
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement>& clone_requirements,
          const std::vector<ApUserEvent>& unmap_events) = 0;
      virtual void invalidate_logical_context(void) = 0;
      virtual void invalidate_region_tree_contexts(
          const bool is_top_level_task, std::set<RtEvent>& applied,
          const ShardMapping* mapping = nullptr, ShardID source_shard = 0) = 0;
    public:
      virtual FutureInstance* create_task_local_future(
          Memory memory, size_t size, bool silence_warnings = false,
          const char* warning_string = nullptr) = 0;
      virtual PhysicalInstance create_task_local_instance(
          Memory memory, const Realm::InstanceLayoutGeneric& layout,
          bool can_fail, bool escaping, RtEvent& use_event) = 0;
      virtual void destroy_task_local_instance(
          PhysicalInstance instance, RtEvent precondition) = 0;
      virtual size_t query_available_memory(Memory target, bool escaping) = 0;
      virtual void release_memory_pool(Memory target) = 0;
    public:
      const std::vector<PhysicalRegion>& begin_task(Processor proc);
      virtual void end_task(
          const void* res, size_t res_size, bool owned, PhysicalInstance inst,
          FutureFunctor* callback_functor,
          const Realm::ExternalInstanceResource* resource,
          void (*freefunc)(const Realm::ExternalInstanceResource&),
          const void* metadataptr, size_t metadatasize, ApEvent effects);
      virtual void post_end_task(void) = 0;
      virtual RtEvent escape_task_local_instance(
          PhysicalInstance instance, RtEvent effects, size_t num_results,
          PhysicalInstance* results, LgEvent* unique_events,
          const Realm::InstanceLayoutGeneric** layouts = nullptr,
          bool redistrict_only = false);
      virtual void release_task_local_instances(
          ApEvent effects, RtEvent safe_effects);
      FutureInstance* copy_to_future_inst(const void* value, size_t size);
      virtual void handle_mispredication(void);
    public:
      virtual Lock create_lock(void);
      virtual void destroy_lock(Lock l) = 0;
      virtual Grant acquire_grant(const std::vector<LockRequest>& requests) = 0;
      virtual void release_grant(Grant grant) = 0;
    public:
      virtual PhaseBarrier create_phase_barrier(unsigned arrivals);
      virtual void destroy_phase_barrier(PhaseBarrier pb) = 0;
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier pb);
    public:
      virtual DynamicCollective create_dynamic_collective(
          unsigned arrivals, ReductionOpID redop, const void* init_value,
          size_t init_size) = 0;
      virtual void destroy_dynamic_collective(DynamicCollective dc) = 0;
      virtual void arrive_dynamic_collective(
          DynamicCollective dc, const void* buffer, size_t size,
          unsigned count) = 0;
      virtual void defer_dynamic_collective_arrival(
          DynamicCollective dc, const Future& future, unsigned count) = 0;
      virtual Future get_dynamic_collective_result(
          DynamicCollective dc, Provenance* provenance) = 0;
      virtual DynamicCollective advance_dynamic_collective(
          DynamicCollective dc) = 0;
    public:
      virtual TaskPriority get_current_priority(void) const = 0;
      virtual void set_current_priority(TaskPriority priority) = 0;
    public:
      PhysicalRegion get_physical_region(unsigned idx);
      void get_physical_references(unsigned idx, InstanceSet& refs);
    public:
      OutputRegion get_output_region(unsigned idx) const;
      const std::vector<OutputRegion> get_output_regions(void) const
      {
        return output_regions;
      }
    public:
      virtual void raise_poison_exception(void);
      virtual void raise_region_exception(PhysicalRegion region, bool nuclear);
    public:
      bool safe_cast(
          IndexSpace handle, const void* realm_point, TypeTag type_tag);
      bool is_region_mapped(unsigned idx);
      void record_padded_fields(VariantImpl* variant);
    protected:
      bool check_region_dependence(
          RegionTreeID tid, IndexSpace space, const RegionRequirement& our_req,
          const RegionUsage& our_usage, const RegionRequirement& req,
          bool check_privileges = true) const;
    public:
      void add_output_region(
          const OutputRequirement& req, const InstanceSet& instances,
          bool global_indexing, bool bounded, bool grouped);
      void finalize_output_regions(
          RtEvent safe_effects, std::vector<RtEvent>& output_regions_finalized);
      void initialize_overhead_profiler(void);
      bool begin_runtime_call(RuntimeCallKind kind, Provenance* provenance);
      void end_runtime_call(
          RuntimeCallKind kind, Provenance* provenance,
          unsigned long long start, unsigned long long stop);
      inline void begin_wait(LgEvent event, bool from_application);
      inline void end_wait(LgEvent event, bool from_application);
      void start_profiling_range(void);
      void stop_profiling_range(const char* provenance);
    public:
      void* get_local_task_variable(LocalVariableID id);
      void set_local_task_variable(
          LocalVariableID id, const void* value, void (*destructor)(void*));
    public:
      void yield(void);
      void record_asynchronous_effect(ApEvent effect, const char* provenance);
      void concurrent_task_barrier(void);
    public:
      void increment_inlined(void);
      void decrement_inlined(void);
      void wait_for_inlined(void);
    protected:
      Future predicate_task_false(
          const TaskLauncher& launcher, Provenance* provenance);
      FutureMap predicate_index_task_false(
          IndexSpace launch_space, const IndexTaskLauncher& launcher,
          Provenance* provenance);
      Future predicate_index_task_reduce_false(
          const IndexTaskLauncher& launch, IndexSpace launch_space,
          ReductionOpID redop, Provenance* provenance);
    public:
      SingleTask* const owner_task;
      const std::vector<RegionRequirement>& regions;
      const std::vector<OutputRequirement>& output_reqs;
    protected:
      // For profiling information
      friend class SingleTask;
    protected:
      // Event encapsulating all effects for this task
      ApEvent realm_done_event;
      int depth;
      // This data structure doesn't need a lock becaue
      // it is only mutated by the application task
      std::vector<PhysicalRegion> physical_regions;
    protected:
      std::vector<OutputRegion> output_regions;
    protected:
      Processor executing_processor;
    public:
      // Support for inlining
      mutable LocalLock inline_lock;
      unsigned inlined_tasks;
      RtUserEvent inlining_done;
    protected:
      uint64_t total_tunable_count;
    protected:
      class OverheadProfiler
        : public Mapping::ProfilingMeasurements::RuntimeOverhead {
      public:
        OverheadProfiler(void) : inside_runtime_call(false) { }
      public:
        long long previous_profiling_time;
        bool inside_runtime_call;
      };
      OverheadProfiler* overhead_profiler;
    protected:
      class ImplicitTaskProfiler {
      public:
        std::deque<LegionProfInstance::WaitInfo> waits;
        long long start_time;
      };
      ImplicitTaskProfiler* implicit_task_profiler;
      std::vector<ApEvent>* implicit_effects;
    protected:
      std::map<LocalVariableID, std::pair<void*, void (*)(void*)> >
          task_local_variables;
    protected:
      // Cache for accelerating safe casts
      std::map<IndexSpace, IndexSpaceNode*> safe_cast_spaces;
      std::atomic<int> safe_cast_semaphore;
    protected:
      // Map of task local instances including their unique events
      // from the profilters perspective
      std::map<PhysicalInstance, std::pair<LgEvent, bool /*escaping*/> >
          task_local_instances;
    protected:
      std::vector<ExceptionHandlerID> exception_handler_stack;
      std::vector<long long> user_profiling_ranges;
    protected:
      bool task_executed;
      bool mutable_priority;
    public:
      const bool inline_task;
      const bool implicit_task;
    public:
      // Needed for Legion Spy
      // See comment in PhysicalRegionImpl::unmap_region
      // to understand what these members are for
      inline ApEvent get_tracing_replay_event(void) const
      {
        return tracing_replay_event;
      }
    protected:
      ApEvent tracing_replay_event;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/contexts/context.inl"

#endif  // __LEGION_CONTEXT_H__
