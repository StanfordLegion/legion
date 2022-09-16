/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "legion/legion_tasks.h"
#include "legion/legion_mapping.h"
#include "legion/legion_instances.h"
#include "legion/legion_allocation.h"

namespace Legion {
  namespace Internal {
   
    /**
     * \class TaskContext
     * The base class for all task contexts which 
     * provide all the methods for handling the 
     * execution of a task at runtime.
     */
    class TaskContext : public ResourceTracker, public Collectable {
    public:
      class AutoRuntimeCall {
      public:
        AutoRuntimeCall(TaskContext *c) : ctx(c) { ctx->begin_runtime_call(); }
        ~AutoRuntimeCall(void) { ctx->end_runtime_call(); }
      public:
        TaskContext *const ctx;
      };
      // This is a no-op task for yield operations
      struct YieldArgs : public LgTaskArgs<YieldArgs> {
      public:
        static const LgTaskID TASK_ID = LG_YIELD_TASK_ID;
      public:
        YieldArgs(UniqueID uid) : LgTaskArgs<YieldArgs>(uid) { }
      };
    public:
      TaskContext(Runtime *runtime, SingleTask *owner, int depth,
                  const std::vector<RegionRequirement> &reqs,
                  bool inline_ctx, bool implicit_ctx = false);
      TaskContext(const TaskContext &rhs);
      virtual ~TaskContext(void);
    public:
      TaskContext& operator=(const TaskContext &rhs);
    public:
      // This is used enough that we want it inlined
      inline Processor get_executing_processor(void) const
        { return executing_processor; }
      inline void set_executing_processor(Processor p)
        { executing_processor = p; }
      inline size_t get_tunable_index(void)
        { return total_tunable_count++; }
      inline UniqueID get_unique_id(void) const 
        { return get_context_uid(); }
      inline const char* get_task_name(void)
        { return get_task()->get_task_name(); }
      inline const std::vector<PhysicalRegion>& get_physical_regions(void) const
        { return physical_regions; }
      inline bool has_created_requirements(void) const
        { return !created_requirements.empty(); }
      inline SingleTask* get_owner_task(void) const { return owner_task; }
      inline bool is_priority_mutable(void) const { return mutable_priority; }
      inline int get_depth(void) const { return depth; }
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const = 0;
      virtual ContextID get_context_id(void) const = 0;
      virtual UniqueID get_context_uid(void) const;
      virtual Task* get_task(void); 
      virtual TaskContext* find_parent_context(void);
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target) = 0;
      virtual void compute_task_tree_coordinates(
          std::vector<std::pair<size_t,DomainPoint> > &coordinates) = 0;
      virtual bool attempt_children_complete(void) = 0;
      virtual bool attempt_children_commit(void) = 0;
      virtual VariantImpl* select_inline_variant(TaskOp *child,
                const std::vector<PhysicalRegion> &parent_regions,
                std::deque<InstanceSet> &physical_instances);
      virtual bool is_leaf_context(void) const;
      virtual bool is_inner_context(void) const;
#ifdef LEGION_USE_LIBDL
      virtual void perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, const void *buffer,
                     size_t buffer_size, bool withargs, size_t dedup_tag,
                     RtEvent local_done, RtEvent global_done, 
                     std::set<RtEvent> &preconditions);
#endif
    public:
      virtual VariantID register_variant(const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  const CodeDescriptor &desc, bool ret, 
                                  VariantID vid, bool check_task_id);
      virtual TraceID generate_dynamic_trace_id(void);
      virtual MapperID generate_dynamic_mapper_id(void);
      virtual ProjectionID generate_dynamic_projection_id(void);
      virtual TaskID generate_dynamic_task_id(void);
      virtual ReductionOpID generate_dynamic_reduction_id(void);
      virtual CustomSerdezID generate_dynamic_serdez_id(void);
      virtual bool perform_semantic_attach(bool &global);
      virtual void post_semantic_attach(void);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Domain &bounds,
                                            TypeTag type_tag,
                                            const char *provenance);
      virtual IndexSpace create_index_space(const Future &future,
                                            TypeTag type_tag,
                                            const char *provenance) = 0;
      virtual IndexSpace create_index_space(
                           const std::vector<DomainPoint> &points,
                           const char *provenance);
      virtual IndexSpace create_index_space(
                           const std::vector<Domain> &rects,
                           const char *provenance);
    protected:
      IndexSpace create_index_space_internal(const Domain &bounds,
                                             TypeTag type_tag,
                                             const char *provenance);
    public:
      virtual IndexSpace union_index_spaces(
                           const std::vector<IndexSpace> &spaces,
                           const char *provenance);
      virtual IndexSpace intersect_index_spaces(
                           const std::vector<IndexSpace> &spaces,
                           const char *provenance);
      virtual IndexSpace subtract_index_spaces(
                           IndexSpace left, IndexSpace right,
                           const char *provenance);
      virtual void create_shared_ownership(IndexSpace handle);
      virtual void destroy_index_space(IndexSpace handle,
                                       const bool unordered,
                                       const bool recurse,
                                       const char *provenance) = 0;
      virtual void create_shared_ownership(IndexPartition handle);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered,
                                           const bool recurse,
                                           const char *provenance) = 0;
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color, 
                                            bool dominates,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance) = 0;
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov) = 0;
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *prov) = 0;
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance) = 0;
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance) = 0;
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance) = 0;
      virtual FieldSpace create_field_space(const char *provenance);
      virtual FieldSpace create_field_space(const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id,
                                        const char *provenance);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id,
                                        const char *provenance) = 0;
      virtual void create_shared_ownership(FieldSpace handle);
      virtual void destroy_field_space(FieldSpace handle,
                                       const bool unordered,
                                       const char *provenance) = 0;
      virtual FieldID allocate_field(FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id,
                                     const char *provenance);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id,
                                     const char *provenance) = 0;
      virtual void allocate_local_field(
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events,
                                     const char *provenance) = 0;
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered,
                              const char *provenance) = 0;
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id,
                                   const char *provenance);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id,
                                   const char *provenance) = 0;
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events,
                                   const char *provenance) = 0;
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space, 
                               const std::set<FieldID> &to_free,
                               const bool unordered,
                               const char *provenance) = 0; 
      virtual LogicalRegion create_logical_region(
                                            IndexSpace index_space,
                                            FieldSpace field_space,
                                            bool task_local,
                                            const char *provenance);
      virtual void create_shared_ownership(LogicalRegion handle);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered,
                                          const char *provenance) = 0;
      virtual FieldAllocatorImpl* create_field_allocator(FieldSpace handle);
      virtual void destroy_field_allocator(FieldSpaceNode *node);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const = 0;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const = 0;
    public:
      virtual void add_physical_region(const RegionRequirement &req, 
          bool mapped, MapperID mid, MappingTagID tag, ApUserEvent &unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances) = 0;
      virtual Future execute_task(const TaskLauncher &launcher) = 0;
      virtual FutureMap execute_index_space(
                                         const IndexTaskLauncher &launcher) = 0;
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                   ReductionOpID redop, bool deterministic) = 0; 
      virtual Future reduce_future_map(const FutureMap &future_map,
                                   ReductionOpID redop, bool deterministic,
                                   const char *prov) = 0;
      virtual FutureMap construct_future_map(IndexSpace domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false) = 0;
      virtual FutureMap construct_future_map(const Domain &domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false) = 0;
      virtual FutureMap construct_future_map(IndexSpace domain,
                               const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL) = 0;
      virtual FutureMap construct_future_map(const Domain &domain,
                               const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL) = 0;
      virtual PhysicalRegion map_region(const InlineLauncher &launcher) = 0;
      virtual ApEvent remap_region(const PhysicalRegion &region,
                                   const char *provenance) = 0;
      virtual void unmap_region(PhysicalRegion region) = 0;
      virtual void unmap_all_regions(bool external) = 0;
      virtual void fill_fields(const FillLauncher &launcher) = 0;
      virtual void fill_fields(const IndexFillLauncher &launcher) = 0;
      virtual void issue_copy(const CopyLauncher &launcher) = 0;
      virtual void issue_copy(const IndexCopyLauncher &launcher) = 0;
      virtual void issue_acquire(const AcquireLauncher &launcher) = 0;
      virtual void issue_release(const ReleaseLauncher &launcher) = 0;
      virtual PhysicalRegion attach_resource(
                                  const AttachLauncher &launcher) = 0;
      virtual ExternalResources attach_resources(
                                  const IndexAttachLauncher &launcher) = 0;
      virtual Future detach_resource(PhysicalRegion region, 
                                     const bool flush,const bool unordered,
                                     const char *provenance = NULL) = 0;
      virtual Future detach_resources(ExternalResources resources,
                                    const bool flush, const bool unordered,
                                    const char *provenance) = 0;
      virtual void progress_unordered_operations(void) = 0;
      virtual FutureMap execute_must_epoch(
                                 const MustEpochLauncher &launcher) = 0;
      virtual Future issue_timing_measurement(
                                    const TimingLauncher &launcher) = 0;
      virtual Future select_tunable_value(const TunableLauncher &launcher) = 0;
      virtual Future issue_mapping_fence(const char *provenance) = 0;
      virtual Future issue_execution_fence(const char *provenance) = 0;
      virtual void complete_frame(const char *provenance) = 0;
      virtual Predicate create_predicate(const Future &f,
                                         const char *provenance) = 0;
      virtual Predicate predicate_not(const Predicate &p,
                                      const char *provenance) = 0;
      virtual Predicate create_predicate(const PredicateLauncher &launcher) = 0;
      virtual Future get_predicate_future(const Predicate &p) = 0;
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
               const std::vector<StaticDependence> *dependences) = 0;
      virtual size_t register_new_close_operation(CloseOp *op) = 0;
      virtual size_t register_new_summary_operation(TraceSummaryOp *op) = 0;
      virtual bool add_to_dependence_queue(Operation *op, 
                                           bool unordered = false) = 0;
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
                                          PhysicalInstance instance = 
                                            PhysicalInstance::NO_INST,
                                          FutureFunctor *callback_functor=NULL,
                                          bool own_functor = false) = 0;
      virtual void register_executing_child(Operation *op) = 0;
      virtual void register_child_executed(Operation *op) = 0;
      virtual void register_child_complete(Operation *op) = 0;
      virtual void register_child_commit(Operation *op) = 0; 
      virtual ApEvent register_implicit_dependences(Operation *op) = 0;
    public:
      virtual RtEvent get_current_mapping_fence_event(void) = 0;
      virtual ApEvent get_current_execution_fence_event(void) = 0;
      // Break this into two pieces since we know that there are some
      // kinds of operations (like deletions) that want to act like 
      // one-sided fences (e.g. waiting on everything before) but not
      // preventing re-ordering for things afterwards
      virtual void perform_fence_analysis(Operation *op, 
          std::set<ApEvent> &preconditions, bool mapping, bool execution) = 0;
      virtual void update_current_fence(FenceOp *op, 
                                        bool mapping, bool execution) = 0;
      virtual void update_current_implicit(Operation *op) = 0;
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
        bool static_trace, const std::set<RegionTreeID> *managed, bool dep,
        const char *provenance) = 0;
      virtual void end_trace(TraceID tid, bool deprecated,
                             const char *provenance) = 0;
      virtual void record_previous_trace(LegionTrace *trace) = 0;
      virtual void invalidate_trace_cache(LegionTrace *trace,
                                          Operation *invalidator) = 0;
      virtual void record_blocking_call(void) = 0;
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination) = 0;
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination) = 0;
      virtual void finish_frame(ApEvent frame_termination) = 0;
    public:
      virtual void increment_outstanding(void) = 0;
      virtual void decrement_outstanding(void) = 0;
      virtual void increment_pending(void) = 0;
      virtual void decrement_pending(TaskOp *child) = 0;
      virtual void decrement_pending(bool need_deferral) = 0;
      virtual void increment_frame(void) = 0;
      virtual void decrement_frame(void) = 0;
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index) = 0;
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent) = 0;
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL) = 0;
      virtual InnerContext* find_top_context(InnerContext *previous = NULL) = 0;
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events) = 0;
      virtual void invalidate_region_tree_contexts(void) = 0;
      virtual void send_back_created_state(AddressSpaceID target) = 0;
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL) = 0;
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(
                                                   Legion::Runtime *&runtime);
      virtual PhysicalInstance create_task_local_instance(Memory memory,
                                        Realm::InstanceLayoutGeneric *layout);
      virtual void end_task(const void *res, size_t res_size, bool owned,
                    PhysicalInstance inst, FutureFunctor *callback_functor) = 0;
      virtual void post_end_task(const void *res, size_t res_size, 
                               bool owned, FutureFunctor *callback_functor) = 0;
      void begin_misspeculation(void);
      void end_misspeculation(const void *res, size_t res_size);
    public:
      virtual Lock create_lock(void);
      virtual void destroy_lock(Lock l) = 0;
      virtual Grant acquire_grant(const std::vector<LockRequest> &requests) = 0;
      virtual void release_grant(Grant grant) = 0;
    public:
      virtual PhaseBarrier create_phase_barrier(unsigned arrivals);
      virtual void destroy_phase_barrier(PhaseBarrier pb) = 0;
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier pb) = 0;
    public:
      virtual DynamicCollective create_dynamic_collective(
                                                  unsigned arrivals,
                                                  ReductionOpID redop,
                                                  const void *init_value,
                                                  size_t init_size) = 0;
      virtual void destroy_dynamic_collective(DynamicCollective dc) = 0;
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                        const void *buffer, size_t size, unsigned count) = 0;
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &future,
                                                    unsigned count) = 0;
      virtual Future get_dynamic_collective_result(DynamicCollective dc,
                                                   const char *provenance) = 0;
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc) = 0;
    public:
      virtual TaskPriority get_current_priority(void) const = 0;
      virtual void set_current_priority(TaskPriority priority) = 0;
    public:
      PhysicalRegion get_physical_region(unsigned idx);
      void get_physical_references(unsigned idx, InstanceSet &refs);
    public:
      void add_created_region(LogicalRegion handle, bool task_local);
      // for logging created region requirements
      void log_created_requirements(void); 
    public:
      void register_region_creation(LogicalRegion handle, bool task_local);
    public:
      void register_field_creation(FieldSpace space, FieldID fid, bool local);
      void register_all_field_creations(FieldSpace space, bool local,
                                        const std::vector<FieldID> &fields);
    public:
      void register_field_space_creation(FieldSpace space);
    public:
      bool has_created_index_space(IndexSpace space) const;
      void register_index_space_creation(IndexSpace space);
    public:
      void register_index_partition_creation(IndexPartition handle);
    public:
      void report_leaks_and_duplicates(std::set<RtEvent> &preconditions);
    public:
      void analyze_destroy_fields(FieldSpace handle,
                                  const std::set<FieldID> &to_delete,
                                  std::vector<RegionRequirement> &delete_reqs,
                                  std::vector<unsigned> &parent_req_indexes,
                                  std::vector<FieldID> &global_to_free,
                                  std::vector<FieldID> &local_to_free,
                                  std::vector<FieldID> &local_field_indexes,
                                  std::vector<unsigned> &deletion_req_indexes);
      void analyze_destroy_logical_region(LogicalRegion handle,
                                  std::vector<RegionRequirement> &delete_reqs,
                                  std::vector<unsigned> &parent_req_indexes,
                                  std::vector<bool> &returnable_privileges);
      virtual void analyze_free_local_fields(FieldSpace handle,
                                  const std::vector<FieldID> &local_to_free,
                                  std::vector<unsigned> &local_field_indexes);
      void remove_deleted_requirements(const std::vector<unsigned> &indexes,
                                  std::vector<LogicalRegion> &to_delete);
      void remove_deleted_fields(const std::set<FieldID> &to_free,
                                 const std::vector<unsigned> &indexes);
      virtual void remove_deleted_local_fields(FieldSpace space,
                                 const std::vector<FieldID> &to_remove); 
    public:
      virtual void raise_poison_exception(void);
      virtual void raise_region_exception(PhysicalRegion region, bool nuclear);
    public:
      bool safe_cast(RegionTreeForest *forest, IndexSpace handle, 
                     const void *realm_point, TypeTag type_tag);
      bool is_region_mapped(unsigned idx);
      void clone_requirement(unsigned idx, RegionRequirement &target);
      int find_parent_region_req(const RegionRequirement &req, 
                                 bool check_privilege = true);
      LegionErrorType check_privilege(const IndexSpaceRequirement &req) const;
      LegionErrorType check_privilege(const RegionRequirement &req, 
                                      FieldID &bad_field, int &bad_index, 
                                      bool skip_privileges = false) const; 
      LogicalRegion find_logical_region(unsigned index);
    protected:
      LegionErrorType check_privilege_internal(const RegionRequirement &req,
                                      const RegionRequirement &parent_req,
                                      std::set<FieldID>& privilege_fields,
                                      FieldID &bad_field, int local, int &bad,
                                      bool skip_privileges) const;
      bool check_region_dependence(RegionTreeID tid, IndexSpace space,
                                  const RegionRequirement &our_req,
                                  const RegionUsage &our_usage,
                                  const RegionRequirement &req,
                                  bool check_privileges = true) const;
    public:
      void initialize_overhead_tracker(void);
      inline void begin_runtime_call(void);
      inline void end_runtime_call(void);
      inline void begin_task_wait(bool from_runtime);
      inline void end_task_wait(void); 
      void remap_unmapped_regions(LegionTrace *current_trace,
                           const std::vector<PhysicalRegion> &unmapped_regions,
                           const char *provenance);
    public:
      void* get_local_task_variable(LocalVariableID id);
      void set_local_task_variable(LocalVariableID id, const void *value,
                                   void (*destructor)(void*));
    public:
      void yield(void);
      void release_task_local_instances(PhysicalInstance return_inst);
#ifdef LEGION_MALLOC_INSTANCES
      void release_future_local_instance(PhysicalInstance return_inst);
#endif
    protected:
      Future predicate_task_false(const TaskLauncher &launcher);
      FutureMap predicate_index_task_false(const IndexTaskLauncher &launcher);
      Future predicate_index_task_reduce_false(const IndexTaskLauncher &launch);
    public:
      // Find an index space name for a concrete launch domain
      IndexSpace find_index_launch_space(const Domain &domain,
                                         const std::string &provenance);
    public:
      Runtime *const runtime;
      SingleTask *const owner_task;
      const std::vector<RegionRequirement> &regions;
    protected:
      // For profiling information
      friend class SingleTask;
    protected:
      mutable LocalLock                         privilege_lock;
      int                                       depth;
      unsigned                                  next_created_index;
      // Application tasks can manipulate these next two data
      // structure by creating regions and fields, make sure you are
      // holding the operation lock when you are accessing them
      // We use a region requirement with an empty privilege_fields
      // set to indicate regions on which we have privileges for 
      // all fields because this is a created region instead of
      // a created field.
      std::map<unsigned,RegionRequirement>      created_requirements;
      std::map<unsigned,bool>                   returnable_privileges;
      // Number of outstanding deletions using this created requirement
      // The last one to send the count to zero actually gets to delete
      // the requirement and the logical region
      std::map<unsigned,unsigned>               deletion_counts;
    protected:
      // This data structure doesn't need a lock becaue
      // it is only mutated by the application task 
      std::vector<PhysicalRegion>               physical_regions; 
    protected:
      Processor                             executing_processor;
      size_t                                total_tunable_count;
    protected:
      Mapping::ProfilingMeasurements::RuntimeOverhead *overhead_tracker;
      long long                                previous_profiling_time; 
    protected:
      std::map<LocalVariableID,
               std::pair<void*,void (*)(void*)> > task_local_variables;
    protected:
      // Cache for accelerating safe casts
      std::map<IndexSpace,IndexSpaceNode*> safe_cast_spaces;
    protected:
      // Field allocation data
      std::map<FieldSpace,FieldAllocatorImpl*> field_allocators;
    protected:
      // Our cached set of index spaces for immediate domains
      std::map<Domain,IndexSpace> index_launch_spaces;
    protected:
#ifdef LEGION_MALLOC_INSTANCES
      std::vector<std::pair<PhysicalInstance,uintptr_t> > task_local_instances;
#else
      std::vector<PhysicalInstance> task_local_instances;
#endif
    protected:
      bool task_executed;
      bool has_inline_accessor;
      bool mutable_priority;
    protected: 
      bool children_complete_invoked;
      bool children_commit_invoked;
    public:
      const bool inline_task;
      const bool implicit_task; 
#ifdef LEGION_SPY
    protected:
      UniqueID current_fence_uid;
#endif
    };

    class InnerContext : public TaskContext,
                         public LegionHeapify<InnerContext> {
    public:
      // Prepipeline stages need to hold a reference since the
      // logical analysis could clean the context up before it runs
      struct PrepipelineArgs : public LgTaskArgs<PrepipelineArgs> {
      public:
        static const LgTaskID TASK_ID = LG_PRE_PIPELINE_ID;
      public:
        PrepipelineArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<PrepipelineArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DependenceArgs : public LgTaskArgs<DependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_DEPENDENCE_ID;
      public:
        DependenceArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<DependenceArgs>(op->get_unique_op_id()), 
            context(ctx) { }
      public:
        InnerContext *const context;
      }; 
      struct TriggerReadyArgs : public LgTaskArgs<TriggerReadyArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_READY_ID;
      public:
        TriggerReadyArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<TriggerReadyArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredEnqueueTaskArgs : 
        public LgTaskArgs<DeferredEnqueueTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_ENQUEUE_TASK_ID;
      public:
        DeferredEnqueueTaskArgs(TaskOp *t, InnerContext *ctx)
          : LgTaskArgs<DeferredEnqueueTaskArgs>(t->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredDistributeTaskArgs : 
        public LgTaskArgs<DeferredDistributeTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_DISTRIBUTE_TASK_ID;
      public:
        DeferredDistributeTaskArgs(TaskOp *op, InnerContext *ctx)
          : LgTaskArgs<DeferredDistributeTaskArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredLaunchTaskArgs :
        public LgTaskArgs<DeferredLaunchTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_LAUNCH_TASK_ID;
      public:
        DeferredLaunchTaskArgs(TaskOp *op, InnerContext *ctx)
          : LgTaskArgs<DeferredLaunchTaskArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct TriggerResolutionArgs : public LgTaskArgs<TriggerResolutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_RESOLUTION_ID;
      public:
        TriggerResolutionArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<TriggerResolutionArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct TriggerExecutionArgs : public LgTaskArgs<TriggerExecutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_EXECUTION_ID;
      public:
        TriggerExecutionArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<TriggerExecutionArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredExecutionArgs : public LgTaskArgs<DeferredExecutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_EXECUTION_ID;
      public:
        DeferredExecutionArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<DeferredExecutionArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct TriggerCompletionArgs : public LgTaskArgs<TriggerCompletionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_COMPLETION_ID;
      public:
        TriggerCompletionArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<TriggerCompletionArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredCompletionArgs : 
        public LgTaskArgs<DeferredCompletionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMPLETION_ID;
      public:
        DeferredCompletionArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<DeferredCompletionArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct TriggerCommitArgs : public LgTaskArgs<TriggerCommitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_COMMIT_ID; 
      public:
        TriggerCommitArgs(Operation *op, InnerContext *ctx)
          : LgTaskArgs<TriggerCommitArgs>(op->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct DeferredCommitArgs : public LgTaskArgs<DeferredCommitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMMIT_ID;
      public:
        DeferredCommitArgs(const std::pair<Operation*,bool> &op,
                           InnerContext *ctx)
          : LgTaskArgs<DeferredCommitArgs>(op.first->get_unique_op_id()),
            context(ctx) { }
      public:
        InnerContext *const context;
      };
      struct PostEndArgs : public LgTaskArgs<PostEndArgs> {
      public:
        static const LgTaskID TASK_ID = LG_POST_END_ID;
      public:
        PostEndArgs(TaskOp *owner, InnerContext *ctx)
          : LgTaskArgs<PostEndArgs>(owner->get_unique_op_id()),
            proxy_this(ctx) { }
      public:
        InnerContext *const proxy_this;
      };
      struct PostTaskArgs {
      public:
        PostTaskArgs(TaskContext *ctx, size_t idx, const void *r, size_t s,
                     PhysicalInstance i, RtEvent w, FutureFunctor *f, bool o)
          : context(ctx), index(idx), result(r), size(s), 
            instance(i), wait_on(w), functor(f), owned(o) { }
      public:
        inline bool operator<(const PostTaskArgs &rhs) const
          { return index < rhs.index; }
      public:
        TaskContext *context;
        size_t index;
        const void *result;
        size_t size;
        PhysicalInstance instance;
        RtEvent wait_on;
        FutureFunctor *functor;
        bool owned;
      };
      struct IssueFrameArgs : public LgTaskArgs<IssueFrameArgs> {
      public:
        static const LgTaskID TASK_ID = LG_ISSUE_FRAME_TASK_ID;
      public:
        IssueFrameArgs(TaskOp *owner, InnerContext *ctx,
                       FrameOp *f, ApEvent term)
          : LgTaskArgs<IssueFrameArgs>(owner->get_unique_op_id()),
            parent_ctx(ctx), frame(f), frame_termination(term) { }
      public:
        InnerContext *const parent_ctx;
        FrameOp *const frame;
        const ApEvent frame_termination;
      };
      struct RemoteCreateViewArgs : public LgTaskArgs<RemoteCreateViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_VIEW_CREATION_TASK_ID;
      public:
        RemoteCreateViewArgs(InnerContext *proxy, PhysicalManager *man,
               InstanceView **tar, RtUserEvent trig, AddressSpaceID src)
          : LgTaskArgs<RemoteCreateViewArgs>(implicit_provenance),
            proxy_this(proxy), manager(man), target(tar), 
            to_trigger(trig), source(src) { }
      public:
        InnerContext *const proxy_this;
        PhysicalManager *const manager;
        InstanceView **target;
        const RtUserEvent to_trigger;
        const AddressSpaceID source;
      };
      struct VerifyPartitionArgs : public LgTaskArgs<VerifyPartitionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_VERIFY_PARTITION_TASK_ID;
      public:
        VerifyPartitionArgs(InnerContext *proxy, IndexPartition p, 
                            PartitionKind k, const char *f)
          : LgTaskArgs<VerifyPartitionArgs>(proxy->get_unique_id()), 
            proxy_this(proxy), pid(p), kind(k), func(f) { }
      public:
        InnerContext *const proxy_this;
        const IndexPartition pid;
        const PartitionKind kind;
        const char *const func;
      };
      template<typename T>
      struct QueueEntry {
      public:
        QueueEntry(void) { op = {}; }
        QueueEntry(T o, RtEvent r) : op(o), ready(r) { }
      public:
        T op;
        RtEvent ready;
      };
      struct LocalFieldInfo {
      public:
        LocalFieldInfo(void)
          : fid(0), size(0), serdez(0), index(0), ancestor(false) { }
        LocalFieldInfo(FieldID f, size_t s, CustomSerdezID z, 
                       unsigned idx, bool a)
          : fid(f), size(s), serdez(z), index(idx), ancestor(a) { }
      public:
        FieldID fid;
        size_t size;
        CustomSerdezID serdez;
        unsigned index;
        bool ancestor;
      };
      class AttachProjectionFunctor : public ProjectionFunctor {
      public:
        AttachProjectionFunctor(ProjectionID pid,
                                std::vector<IndexSpace> &&spaces);
        virtual ~AttachProjectionFunctor(void) { }
      public:
        using ProjectionFunctor::project;
        virtual LogicalRegion project(LogicalRegion upper_bound,
                                      const DomainPoint &point,
                                      const Domain &launch_domain);
        virtual LogicalRegion project(LogicalPartition upper_bound,
                                      const DomainPoint &point,
                                      const Domain &launch_domain);
      public:
        virtual bool is_functional(void) const { return true; }
        // Some depth >0 means the runtime can't analyze it
        virtual unsigned get_depth(void) const { return 1; }
      public:
        const std::vector<IndexSpace> handles;
        const ProjectionID pid;
      };
    public:
      InnerContext(Runtime *runtime, SingleTask *owner, int depth, 
                   bool full_inner, const std::vector<RegionRequirement> &reqs,
                   const std::vector<unsigned> &parent_indexes,
                   const std::vector<bool> &virt_mapped, UniqueID context_uid, 
                   ApEvent execution_fence, bool remote = false, 
                   bool inline_task = false, bool implicit_task = false);
      InnerContext(const InnerContext &rhs);
      virtual ~InnerContext(void);
    public:
      InnerContext& operator=(const InnerContext &rhs);
    public:
      inline unsigned get_max_trace_templates(void) const
        { return context_configuration.max_templates_per_trace; }
      void record_physical_trace_replay(RtEvent ready, bool replay);
      bool is_replaying_physical_trace(void);
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<DeletedRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<DeletedField> &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<DeletedFieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<DeletedIndexSpace> &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions);
    protected:
      // Deletions are virtual so they can be overridden for control replication
      void register_region_creations(
                     std::map<LogicalRegion,unsigned> &regions);
      virtual void register_region_deletions(ApEvent precondition,
                     const std::map<Operation*,GenerationID> &dependences,
                     std::vector<DeletedRegion> &regions,
                     std::set<RtEvent> &preconditions);
      void register_field_creations(
            std::set<std::pair<FieldSpace,FieldID> > &fields);
      virtual void register_field_deletions(ApEvent precondition,
            const std::map<Operation*,GenerationID> &dependences,
            std::vector<DeletedField> &fields,
            std::set<RtEvent> &preconditions);
      void register_field_space_creations(
                          std::map<FieldSpace,unsigned> &spaces);
      void register_latent_field_spaces(
            std::map<FieldSpace,std::set<LogicalRegion> > &spaces);
      virtual void register_field_space_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<DeletedFieldSpace> &spaces,
                          std::set<RtEvent> &preconditions);
      void register_index_space_creations(
                          std::map<IndexSpace,unsigned> &spaces);
      virtual void register_index_space_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<DeletedIndexSpace> &spaces,
                          std::set<RtEvent> &preconditions);
      void register_index_partition_creations(
                          std::map<IndexPartition,unsigned> &parts);
      virtual void register_index_partition_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<DeletedPartition> &parts,
                          std::set<RtEvent> &preconditions);
      ApEvent compute_return_deletion_dependences(size_t return_index,
                          std::map<Operation*,GenerationID> &dependences);
    public:
      int has_conflicting_regions(MapOp *map, bool &parent_conflict,
                                  bool &inline_conflict);
      int has_conflicting_regions(AttachOp *attach, bool &parent_conflict,
                                  bool &inline_conflict);
      int has_conflicting_internal(const RegionRequirement &req, 
                                   bool &parent_conflict,
                                   bool &inline_conflict);
      void find_conflicting_regions(TaskOp *task,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(CopyOp *copy,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(AcquireOp *acquire,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(ReleaseOp *release,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(DependentPartitionOp *partition,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_internal(const RegionRequirement &req,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(FillOp *fill,
                                    std::vector<PhysicalRegion> &conflicting); 
      void register_inline_mapped_region(const PhysicalRegion &region);
      void unregister_inline_mapped_region(const PhysicalRegion &region);
    public:
      void print_children(void);
      void perform_window_wait(void);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual UniqueID get_context_uid(void) const;
      virtual bool is_inner_context(void) const;
      virtual void pack_remote_context(Serializer &rez, AddressSpaceID target);
      virtual void unpack_remote_context(Deserializer &derez,
                                         std::set<RtEvent> &preconditions);
      virtual void compute_task_tree_coordinates(
          std::vector<std::pair<size_t,DomainPoint> > &coordinates);
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle,
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      bool inline_child_task(TaskOp *child);
      virtual void analyze_free_local_fields(FieldSpace handle,
                                  const std::vector<FieldID> &local_to_free,
                                  std::vector<unsigned> &local_field_indexes);
      virtual void remove_deleted_local_fields(FieldSpace space,
                                 const std::vector<FieldID> &to_remove);
    public:
      using TaskContext::create_index_space;
      using TaskContext::create_field_space;
      using TaskContext::allocate_field;
      using TaskContext::allocate_fields;
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Future &future, TypeTag tag,
                                            const char *provenance);
      virtual void destroy_index_space(IndexSpace handle, const bool unordered,
                                       const bool recurse,
                                       const char *provenance);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered,
                                           const bool recurse,
                                           const char *provenance);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *prov);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual void verify_partition(IndexPartition pid, PartitionKind kind,
                                    const char *function_name);
      static void handle_partition_verification(const void *args);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id,
                                        const char *provenance);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered,
                                       const char *provenance);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id,
                                     const char *provenance);
      virtual void allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events,
                                     const char *provenance);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id,
                                   const char *provenance);
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events,
                                   const char *provenance);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered,
                              const char *provenance);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space,
                               const std::set<FieldID> &to_free,
                               const bool unordered, const char *provenance);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered,
                                          const char *provenance);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const;
    public:
      virtual void add_physical_region(const RegionRequirement &req, 
          bool mapped, MapperID mid, MappingTagID tag, ApUserEvent &unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances);
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic,
                                       const char *prov);
      virtual FutureMap construct_future_map(IndexSpace domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false);
      virtual FutureMap construct_future_map(const Domain &domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false);
      virtual FutureMap construct_future_map(IndexSpace domain,
                                   const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL);
      virtual FutureMap construct_future_map(const Domain &domain,
                                   const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(const PhysicalRegion &region,
                                   const char *provenance);
      virtual void unmap_region(PhysicalRegion region);
      virtual void unmap_all_regions(bool external);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual ExternalResources attach_resources(
                                        const IndexAttachLauncher &launcher);
      virtual RegionTreeNode* compute_index_attach_upper_bound(
                                        const IndexAttachLauncher &launcher,
                                        const std::vector<unsigned> &indexes);
      virtual ProjectionID compute_index_attach_projection(IndexTreeNode *node,
                                        std::vector<IndexSpace> &spaces);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered,
                                     const char *provenance = NULL);
      virtual Future detach_resources(ExternalResources resources,
                                      const bool flush, const bool unordered,
                                      const char *provenance);
      virtual void progress_unordered_operations(void);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future select_tunable_value(const TunableLauncher &launcher);
      virtual Future issue_mapping_fence(const char *provenance);
      virtual Future issue_execution_fence(const char *provenance);
      virtual void complete_frame(const char *provenance);
      virtual Predicate create_predicate(const Future &f,
                                         const char *provenance);
      virtual Predicate predicate_not(const Predicate &p,
                                      const char *provenance);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual size_t register_new_close_operation(CloseOp *op);
      virtual size_t register_new_summary_operation(TraceSummaryOp *op);
    public:
      void add_to_prepipeline_queue(Operation *op);
      bool process_prepipeline_stage(void);
    public:
      virtual bool add_to_dependence_queue(Operation *op, 
                                           bool unordered = false);
      void process_dependence_stage(void);
    public:
      template<typename T, typename ARGS, bool HAS_BOUNDS>
      void add_to_queue(QueueEntry<T> entry, LocalLock &lock,
                        std::list<QueueEntry<T> > &queue,
                        CompletionQueue &comp_queue);
      template<typename T>
      T process_queue(LocalLock &lock, RtEvent &next_ready,
                      std::list<QueueEntry<T> > &queue,
                      CompletionQueue &comp_queue,
                      std::vector<T> &to_perform) const;
    public:
      void add_to_ready_queue(Operation *op, RtEvent ready);
      bool process_ready_queue(void);
    public:
      void add_to_task_queue(TaskOp *op, RtEvent ready);
      bool process_enqueue_task_queue(void);
    public:
      void add_to_distribute_task_queue(TaskOp *op, RtEvent ready);
      bool process_distribute_task_queue(void);
    public:
      void add_to_launch_task_queue(TaskOp *op, RtEvent ready);
      bool process_launch_task_queue(void);
    public:
      void add_to_resolution_queue(Operation *op, RtEvent ready);
      bool process_resolution_queue(void);
    public:
      void add_to_trigger_execution_queue(Operation *op, RtEvent ready);
      bool process_trigger_execution_queue(void);
    public:
      void add_to_deferred_execution_queue(Operation *op, RtEvent ready);
      bool process_deferred_execution_queue(void);
    public:
      void add_to_trigger_completion_queue(Operation *op, RtEvent ready);
      bool process_trigger_completion_queue(void);
    public:
      void add_to_deferred_completion_queue(Operation *op, RtEvent ready);
      bool process_deferred_completion_queue(void);
    public:
      void add_to_trigger_commit_queue(Operation *op, RtEvent ready); 
      bool process_trigger_commit_queue(void);
    public:
      void add_to_deferred_commit_queue(Operation *op, RtEvent ready,
                                        bool deactivate);
      bool process_deferred_commit_queue(void);
    public:
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
                                          PhysicalInstance instance =
                                            PhysicalInstance::NO_INST,
                                          FutureFunctor *callback_functor=NULL,
                                          bool own_functor = false);
      bool process_post_end_tasks(void);
    public:
      virtual void register_executing_child(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual ApEvent register_implicit_dependences(Operation *op);
    public:
      virtual RtEvent get_current_mapping_fence_event(void);
      virtual ApEvent get_current_execution_fence_event(void);
      virtual void perform_fence_analysis(Operation *op, 
          std::set<ApEvent> &preconditions, bool mapping, bool execution);
      virtual void update_current_fence(FenceOp *op,
                                        bool mapping, bool execution);
      virtual void update_current_implicit(Operation *op);
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep,
          const char *provenance);
      virtual void end_trace(TraceID tid, bool deprecated,
                             const char *provenance);
      virtual void record_previous_trace(LegionTrace *trace);
      virtual void invalidate_trace_cache(LegionTrace *trace,
                                          Operation *invalidator);
      virtual void record_blocking_call(void);
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination);
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination);
      virtual void finish_frame(ApEvent frame_termination);
    public:
      virtual void increment_outstanding(void);
      virtual void decrement_outstanding(void);
      virtual void increment_pending(void);
      virtual void decrement_pending(TaskOp *child);
      virtual void decrement_pending(bool need_deferral);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
    public:
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(InnerContext *previous = NULL);
    public:
      void configure_context(MapperManager *mapper, TaskPriority priority);
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void invalidate_remote_tree_contexts(Deserializer &derez);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL);
      virtual FillView* find_or_create_fill_view(FillOp *op, 
                             std::set<RtEvent> &map_applied_events,
                             const void *value, const size_t value_size,
                             bool &took_ownership);
      static void handle_remote_view_creation(const void *args);
      void notify_instance_deletion(PhysicalManager *deleted); 
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(
                                                    Legion::Runtime *&runtime);
      virtual void end_task(const void *res, size_t res_size, bool owned,
                        PhysicalInstance inst, FutureFunctor *callback_functor);
      virtual void post_end_task(const void *res, size_t res_size, 
                                 bool owned, FutureFunctor *callback_functor);
    public:
      virtual void destroy_lock(Lock l);
      virtual Grant acquire_grant(const std::vector<LockRequest> &requests);
      virtual void release_grant(Grant grant);
    public:
      virtual void destroy_phase_barrier(PhaseBarrier pb);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier pb);
    public:
      void perform_barrier_dependence_analysis(Operation *op,
            const std::vector<PhaseBarrier> &wait_barriers,
            const std::vector<PhaseBarrier> &arrive_barriers,
            MustEpochOp *must_epoch = NULL);
    protected:
      void analyze_barrier_dependences(Operation *op,
            const std::vector<PhaseBarrier> &barriers,
            MustEpochOp *must_epoch, bool previous_gen);
    public:
      virtual DynamicCollective create_dynamic_collective(
                                                  unsigned arrivals,
                                                  ReductionOpID redop,
                                                  const void *init_value,
                                                  size_t init_size);
      virtual void destroy_dynamic_collective(DynamicCollective dc);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                        const void *buffer, size_t size, unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &future,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc,
                                                   const char *provenance);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority); 
    public:
      static void handle_compute_equivalence_sets_request(Deserializer &derez,
                                     Runtime *runtime, AddressSpaceID source);
    public:
      static void handle_prepipeline_stage(const void *args);
      static void handle_dependence_stage(const void *args);
      static void handle_ready_queue(const void *args);
      static void handle_enqueue_task_queue(const void *args);
      static void handle_distribute_task_queue(const void *args);
      static void handle_launch_task_queue(const void *args);
      static void handle_resolution_queue(const void *args);
      static void handle_trigger_execution_queue(const void *args);
      static void handle_deferred_execution_queue(const void *args);
      static void handle_trigger_completion_queue(const void *args);
      static void handle_deferred_completion_queue(const void *args);
      static void handle_trigger_commit_queue(const void *args);
      static void handle_deferred_commit_queue(const void *args);
      static void handle_post_end_task(const void *args);
    public:
      void free_remote_contexts(void);
      void send_remote_context(AddressSpaceID remote_instance, 
                               RemoteContext *target);
    public:
      void convert_target_views(const InstanceSet &targets, 
                                std::vector<InstanceView*> &target_views);
      // I hate the container problem, same as previous except MaterializedView
      void convert_target_views(const InstanceSet &targets, 
                                std::vector<MaterializedView*> &target_views); 
    protected:
      void execute_task_launch(TaskOp *task, bool index, 
                               LegionTrace *current_trace, 
                               bool silence_warnings, bool inlining_enabled);
      // Must be called while holding the dependence lock
      void insert_unordered_ops(AutoLock &d_lock);
    public:
      void clone_local_fields(
          std::map<FieldSpace,std::vector<LocalFieldInfo> > &child_local) const;
#ifdef DEBUG_LEGION
      // This is a helpful debug method that can be useful when called from
      // a debugger to find the earliest operation that hasn't mapped yet
      // which is especially useful when debugging scheduler hangs
      Operation* get_earliest(void) const;
#endif
#ifdef LEGION_SPY
      void register_implicit_replay_dependence(Operation *op);
#endif
    public:
      const RegionTreeContext tree_context; 
      const UniqueID context_uid;
      const bool remote_context;
      const bool full_inner_context;
    protected:
      bool finished_execution;
    protected:
      Mapper::ContextConfigOutput           context_configuration;
      std::vector<std::pair<size_t,DomainPoint> > context_coordinates;
    protected:
      const std::vector<unsigned>           &parent_req_indexes;
      const std::vector<bool>               &virtual_mapped;
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts, note that accessing
      // this data structure requires the inline lock because
      // unordered detach operations can touch it without synchronizing
      // with the executing task
      mutable LocalLock inline_lock;
      LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC> inline_regions;
    protected:
      mutable LocalLock                     child_op_lock;
      // Track whether this task has finished executing
      size_t total_children_count; // total number of sub-operations
      size_t total_close_count; 
      size_t total_summary_count;
      std::atomic<size_t> outstanding_children_count;
      LegionMap<Operation*,GenerationID,
                EXECUTING_CHILD_ALLOC> executing_children;
      LegionMap<Operation*,GenerationID,
                EXECUTED_CHILD_ALLOC> executed_children;
      LegionMap<Operation*,GenerationID,
                COMPLETE_CHILD_ALLOC> complete_children; 
      // For tracking any operations that come from outside the
      // task like a garbage collector that need to be inserted
      // into the stream of operations from the task
      std::vector<Operation*> unordered_ops;
#ifdef DEBUG_LEGION
      // In debug mode also keep track of them in context order so
      // we can see what the longest outstanding operation is which
      // is often useful when things hang
      std::map<unsigned,Operation*> outstanding_children;
#endif
#ifdef LEGION_SPY
      // Some help for Legion Spy for validating fences
      std::deque<UniqueID> ops_since_last_fence;
      std::set<ApEvent> previous_completion_events;
#endif
    protected: // Queues for fusing together small meta-tasks
      mutable LocalLock                               prepipeline_lock;
      std::deque<std::pair<Operation*,GenerationID> > prepipeline_queue;
      unsigned                                        outstanding_prepipeline;
    protected:
      mutable LocalLock                               dependence_lock;
      std::deque<Operation*>                          dependence_queue;
      RtEvent                                         dependence_precondition;
      // Only one of these ever to keep things in order
      bool                                            outstanding_dependence;
    protected: 
      mutable LocalLock                               ready_lock;
      std::list<QueueEntry<Operation*> >              ready_queue;
      CompletionQueue                                 ready_comp_queue;
    protected:
      mutable LocalLock                               enqueue_task_lock;
      std::list<QueueEntry<TaskOp*> >                 enqueue_task_queue;
      CompletionQueue                                 enqueue_task_comp_queue;
    protected:
      mutable LocalLock                               distribute_task_lock;
      std::list<QueueEntry<TaskOp*> >                 distribute_task_queue;
      CompletionQueue                                distribute_task_comp_queue;
    protected:
      mutable LocalLock                               launch_task_lock;
      std::list<QueueEntry<TaskOp*> >                 launch_task_queue;
      CompletionQueue                                 launch_task_comp_queue;
    protected:
      mutable LocalLock                               resolution_lock;
      std::list<QueueEntry<Operation*> >              resolution_queue;
      CompletionQueue                                 resolution_comp_queue;
    protected:
      mutable LocalLock                               trigger_execution_lock;
      std::list<QueueEntry<Operation*> >              trigger_execution_queue;
      CompletionQueue                             trigger_execution_comp_queue;
    protected:
      mutable LocalLock                               deferred_execution_lock;
      std::list<QueueEntry<Operation*> >              deferred_execution_queue;
      CompletionQueue                             deferred_execution_comp_queue;
    protected:
      mutable LocalLock                               trigger_completion_lock;
      std::list<QueueEntry<Operation*> >              trigger_completion_queue;
      CompletionQueue                             trigger_completion_comp_queue;
    protected:
      mutable LocalLock                               deferred_completion_lock;
      std::list<QueueEntry<Operation*> >              deferred_completion_queue;
      CompletionQueue                            deferred_completion_comp_queue;
    protected:
      mutable LocalLock                               trigger_commit_lock;
      std::list<QueueEntry<Operation*> >              trigger_commit_queue;
      CompletionQueue                                 trigger_commit_comp_queue;
    protected:
      mutable LocalLock                               deferred_commit_lock;
      std::list<QueueEntry<std::pair<Operation*,bool> > > deferred_commit_queue;
      CompletionQueue                                deferred_commit_comp_queue;
    protected:
      mutable LocalLock                               post_task_lock;
      std::list<PostTaskArgs>                         post_task_queue;
      CompletionQueue                                 post_task_comp_queue;
    protected:
      // Traces for this task's execution
      LegionMap<TraceID,LegionTrace*,TASK_TRACES_ALLOC> traces;
      LegionTrace *current_trace;
      LegionTrace *previous_trace;
      // ID is either 0 for not replaying, 1 for replaying, or
      // the event id for signaling that the status isn't ready 
      std::atomic<realm_id_t> physical_trace_replay_status;
      bool valid_wait_event;
      RtUserEvent window_wait;
      std::deque<ApEvent> frame_events;
      RtEvent last_registration; 
    protected:
      // Number of sub-tasks ready to map
      unsigned outstanding_subtasks;
      // Number of mapped sub-tasks that are yet to run
      unsigned pending_subtasks;
      // Number of pending_frames
      unsigned pending_frames;
      // Track whether this context is current active for scheduling
      // indicating that it is no longer far enough ahead
      bool currently_active_context;
    protected:
      FenceOp *current_mapping_fence;
      GenerationID mapping_fence_gen;
      unsigned current_mapping_fence_index;
      ApEvent current_execution_fence_event;
      unsigned current_execution_fence_index;
      // We currently do not track dependences for dependent partitioning
      // operations on index partitions and their subspaces directly, so 
      // we instead use this to ensure mapping dependence ordering with 
      // any operations which might need downstream information about 
      // partitions or subspaces. Note that this means that all dependent
      // partitioning operations are guaranteed to map in order currently
      // We've not extended this to include creation operations as well
      // for similar reasons, so now this is a general operation class
      Operation *last_implicit;
      GenerationID last_implicit_gen;
    protected:
      // For managing changing task priorities
      ApEvent realm_done_event;
      TaskPriority current_priority;
    protected: // Instance top view data structures
      mutable LocalLock                         instance_view_lock;
      std::map<PhysicalManager*,InstanceView*>  instance_top_views;
      std::map<PhysicalManager*,RtUserEvent>    pending_top_views;
    protected:
      mutable LocalLock                         tree_set_lock;
      std::map<RegionTreeID,EquivalenceSet*>    tree_equivalence_sets;
      std::map<std::pair<RegionTreeID,
        IndexSpaceExprID>,EquivalenceSet*>      empty_equivalence_sets;
    protected:
      mutable LocalLock                       remote_lock;
      std::map<AddressSpaceID,RemoteContext*> remote_instances;
    protected:
      // Dependence tracking information for phase barriers
      mutable LocalLock                                   phase_barrier_lock;
      struct BarrierContribution {
      public:
        BarrierContribution(void) : op(NULL), gen(0), uid(0), muid(0) { }
        BarrierContribution(Operation *o, GenerationID g, 
                            UniqueID u, UniqueID m, size_t bg)
          : op(o), gen(g), uid(u), muid(m), bargen(bg) { }
      public:
        Operation *op;
        GenerationID gen;
        UniqueID uid;
        UniqueID muid; // must epoch uid
        size_t bargen; // the barrier generation
      };
      std::map<size_t,std::list<BarrierContribution> > barrier_contributions;
    protected:
      // Track information for locally allocated fields
      mutable LocalLock                                 local_field_lock;
      std::map<FieldSpace,std::vector<LocalFieldInfo> > local_field_infos;
    protected:
      // Cache for fill views
      mutable LocalLock     fill_view_lock;            
      std::list<FillView*>  fill_view_cache;
      static const size_t MAX_FILL_VIEW_CACHE_SIZE = 64;
    protected:
      // This data structure should only be accessed during the logical
      // analysis stage of the pipeline and therefore no lock is needed
      std::map<IndexTreeNode*,
        std::vector<AttachProjectionFunctor*> > attach_functions;
    protected:
      // Resources that can build up over a task's lifetime
      LegionDeque<Reservation,TASK_RESERVATION_ALLOC> context_locks;
      LegionDeque<ApBarrier,TASK_BARRIER_ALLOC> context_barriers;
    };

    /**
     * \class TopLevelContext
     * This is the top-level task context that
     * exists at the root of a task tree. In
     * general there will only be one of these
     * per application unless mappers decide to
     * create their own tasks for performing
     * computation.
     */
    class TopLevelContext : public InnerContext {
    public:
      TopLevelContext(Runtime *runtime, UniqueID ctx_uid);
      TopLevelContext(const TopLevelContext &rhs);
      virtual ~TopLevelContext(void);
    public:
      TopLevelContext& operator=(const TopLevelContext &rhs);
    public:
      virtual void pack_remote_context(Serializer &rez, AddressSpaceID target);
      virtual TaskContext* find_parent_context(void);
    public:
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(InnerContext *previous = NULL);
    public:
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle, 
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
    protected:
      std::vector<RegionRequirement>       dummy_requirements;
      std::vector<unsigned>                dummy_indexes;
      std::vector<bool>                    dummy_mapped;
    };

    /**
     * \class RemoteTask
     * A small helper class for giving application
     * visibility to this remote context
     */
    class RemoteTask : public ExternalTask {
    public:
      RemoteTask(RemoteContext *owner);
      RemoteTask(const RemoteTask &rhs);
      virtual ~RemoteTask(void);
    public:
      RemoteTask& operator=(const RemoteTask &rhs);
    public:
      virtual int get_depth(void) const;
      virtual UniqueID get_unique_id(void) const;
      virtual Domain get_slice_domain(void) const;
      virtual size_t get_context_index(void) const; 
      virtual void set_context_index(size_t index);
      virtual bool has_parent_task(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual const char* get_task_name(void) const;
      virtual bool has_trace(void) const;
    public:
      RemoteContext *const owner;
      unsigned context_index;
    };

    /**
     * \class RemoteContext
     * A remote copy of a TaskContext for the 
     * execution of sub-tasks on remote notes.
     */
    class RemoteContext : public InnerContext {
    public:
      struct RemotePhysicalRequestArgs :
        public LgTaskArgs<RemotePhysicalRequestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_PHYSICAL_REQUEST_TASK_ID;
      public:
        RemotePhysicalRequestArgs(UniqueID uid, RemoteContext *ctx,
                                  InnerContext *loc, unsigned idx, 
                                  AddressSpaceID src, RtUserEvent trig,
                                  LogicalRegion par)
          : LgTaskArgs<RemotePhysicalRequestArgs>(implicit_provenance), 
            context_uid(uid), target(ctx), local(loc), index(idx), 
            source(src), to_trigger(trig), parent(par) { }
      public:
        const UniqueID context_uid;
        RemoteContext *const target;
        InnerContext *const local;
        const unsigned index;
        const AddressSpaceID source;
        const RtUserEvent to_trigger;
        const LogicalRegion parent;
      };
      struct RemotePhysicalResponseArgs : 
        public LgTaskArgs<RemotePhysicalResponseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_PHYSICAL_RESPONSE_TASK_ID;
      public:
        RemotePhysicalResponseArgs(RemoteContext *ctx, InnerContext *res, 
                                   unsigned idx)
          : LgTaskArgs<RemotePhysicalResponseArgs>(implicit_provenance), 
            target(ctx), result(res), index(idx) { }
      public:
        RemoteContext *const target;
        InnerContext *const result;
        const unsigned index;
      };
    public:
      RemoteContext(Runtime *runtime, UniqueID context_uid);
      RemoteContext(const RemoteContext &rhs);
      virtual ~RemoteContext(void);
    public:
      RemoteContext& operator=(const RemoteContext &rhs);
    public:
      virtual Task* get_task(void);
      virtual void unpack_remote_context(Deserializer &derez,
                                         std::set<RtEvent> &preconditions);
      virtual TaskContext* find_parent_context(void);
    public:
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(InnerContext *previous = NULL);
    public:
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle,
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
      virtual void invalidate_region_tree_contexts(void);
      virtual void invalidate_remote_tree_contexts(Deserializer &derez);
    public:
      const Task* get_parent_task(void);
    public:
      void unpack_local_field_update(Deserializer &derez);
      static void handle_local_field_update(Deserializer &derez);
    public:
      static void handle_physical_request(Deserializer &derez,
                      Runtime *runtime, AddressSpaceID source);
      static void defer_physical_request(const void *args, Runtime *runtime);
      void set_physical_context_result(unsigned index, 
                                       InnerContext *result);
      static void handle_physical_response(Deserializer &derez, 
                                           Runtime *runtime);
      static void defer_physical_response(const void *args);
    protected:
      UniqueID parent_context_uid;
      TaskContext *parent_ctx;
    protected:
      bool top_level_context;
      RemoteTask remote_task;
    protected:
      std::vector<unsigned> local_parent_req_indexes;
      std::vector<bool> local_virtual_mapped;
    protected:
      // Cached physical contexts recorded from the owner
      std::map<unsigned/*index*/,InnerContext*> physical_contexts;
      std::map<unsigned,RtEvent> pending_physical_contexts;
      std::set<LogicalRegion> local_physical_contexts;
    };

    /**
     * \class LeafContext
     * A context for the execution of a leaf task
     */
    class LeafContext : public TaskContext,
                        public LegionHeapify<LeafContext> {
    public:
      LeafContext(Runtime *runtime, SingleTask *owner,bool inline_task = false);
      LeafContext(const LeafContext &rhs);
      virtual ~LeafContext(void);
    public:
      LeafContext& operator=(const LeafContext &rhs);
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<DeletedRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<DeletedField> &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<DeletedFieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<DeletedIndexSpace> &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target);
      virtual void compute_task_tree_coordinates(
          std::vector<std::pair<size_t,DomainPoint> > &coordinates);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child,
                const std::vector<PhysicalRegion> &parent_regions,
                std::deque<InstanceSet> &physical_instances);
      virtual bool is_leaf_context(void) const;
    public:
      using TaskContext::create_index_space;
      using TaskContext::create_field_space;
      using TaskContext::allocate_field;
      using TaskContext::allocate_fields;
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Future &future, TypeTag tag,
                                            const char *provenance);
      virtual void destroy_index_space(IndexSpace handle, 
                                       const bool unordered,
                                       const bool recurse,
                                       const char *provenance);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered, 
                                           const bool recurse,
                                           const char *provenance);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color,
                                            const char *provenance);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *provenance);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            const UntypedBuffer &marg,
                                            const char *prov);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            const char *prov);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle,
                                            const char *provenance);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles,
                                            const char *provenance);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id,
                                        const char *provenance);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered,
                                       const char *provenance);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id,
                                     const char *provenance);
      virtual void allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events,
                                     const char *provenance);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id,
                                   const char *provenance);
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events,
                                   const char *provenance);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered,
                              const char *provenance);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space,
                               const std::set<FieldID> &to_free,
                               const bool unordered, const char *provenance);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered,
                                          const char *provenance);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const;
    public:
      virtual void add_physical_region(const RegionRequirement &req, 
          bool mapped, MapperID mid, MappingTagID tag, ApUserEvent &unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances);
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic,
                                       const char *prov);
      virtual FutureMap construct_future_map(IndexSpace domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false);
      virtual FutureMap construct_future_map(const Domain &domain,
                               const std::map<DomainPoint,UntypedBuffer> &data,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false);
      virtual FutureMap construct_future_map(IndexSpace domain,
                                   const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL);
      virtual FutureMap construct_future_map(const Domain &domain,
                                   const std::map<DomainPoint,Future> &futures,
                                             bool internal = false,
                                             bool collective = false,
                                             ShardingID sid = 0,
                                             bool implicit = false,
                                             const char *provenance = NULL);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(const PhysicalRegion &region,
                                   const char *provenance);
      virtual void unmap_region(PhysicalRegion region);
      virtual void unmap_all_regions(bool external);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual ExternalResources attach_resources(
                                          const IndexAttachLauncher &launcher);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered,
                                     const char *provenance = NULL);
      virtual Future detach_resources(ExternalResources resources,
                                      const bool flush, const bool unordered,
                                      const char *provenance);
      virtual void progress_unordered_operations(void);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future select_tunable_value(const TunableLauncher &launcher);
      virtual Future issue_mapping_fence(const char *provenance);
      virtual Future issue_execution_fence(const char *provenance);
      virtual void complete_frame(const char *provenance);
      virtual Predicate create_predicate(const Future &f,
                                         const char *provenance);
      virtual Predicate predicate_not(const Predicate &p,
                                      const char *provenance);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual size_t register_new_close_operation(CloseOp *op);
      virtual size_t register_new_summary_operation(TraceSummaryOp *op);
      virtual bool add_to_dependence_queue(Operation *op, 
                                           bool unordered = false);
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
                                          PhysicalInstance instance =
                                            PhysicalInstance::NO_INST,
                                          FutureFunctor *callback_functor=NULL,
                                          bool own_functor = false);
      virtual void register_executing_child(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual ApEvent register_implicit_dependences(Operation *op);
    public:
      virtual RtEvent get_current_mapping_fence_event(void);
      virtual ApEvent get_current_execution_fence_event(void);
      virtual void perform_fence_analysis(Operation *op,
          std::set<ApEvent> &preconditions, bool mapping, bool execution);
      virtual void update_current_fence(FenceOp *op,
                                        bool mapping, bool execution);
      virtual void update_current_implicit(Operation *op);
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep,
          const char *provenance);
      virtual void end_trace(TraceID tid, bool deprecated,
                             const char *provenance);
      virtual void record_previous_trace(LegionTrace *trace);
      virtual void invalidate_trace_cache(LegionTrace *trace,
                                          Operation *invalidator);
      virtual void record_blocking_call(void);
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination);
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination);
      virtual void finish_frame(ApEvent frame_termination);
    public:
      virtual void increment_outstanding(void);
      virtual void decrement_outstanding(void);
      virtual void increment_pending(void);
      virtual void decrement_pending(TaskOp *child);
      virtual void decrement_pending(bool need_deferral);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(InnerContext *context = NULL);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL);
    public:
      virtual void end_task(const void *res, size_t res_size, bool owned,
                        PhysicalInstance inst, FutureFunctor *callback_functor);
      virtual void post_end_task(const void *res, size_t res_size, 
                                 bool owned, FutureFunctor *callback_functor);
    public:
      virtual void destroy_lock(Lock l);
      virtual Grant acquire_grant(const std::vector<LockRequest> &requests);
      virtual void release_grant(Grant grant);
    public:
      virtual void destroy_phase_barrier(PhaseBarrier pb);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier pb);
    public: 
      virtual DynamicCollective create_dynamic_collective(
                                                  unsigned arrivals,
                                                  ReductionOpID redop,
                                                  const void *init_value,
                                                  size_t init_size);
      virtual void destroy_dynamic_collective(DynamicCollective dc);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                        const void *buffer, size_t size, unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &future,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc,
                                                   const char *provenance);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    protected:
      mutable LocalLock                            leaf_lock;
      std::set<RtEvent>                            execution_events;
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority);
    };

    //--------------------------------------------------------------------------
    inline void TaskContext::begin_runtime_call(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(implicit_reference_tracker == NULL);
#endif
      if (overhead_tracker == NULL)
        return;
      const long long current = Realm::Clock::current_time_in_nanoseconds();
      const long long diff = current - previous_profiling_time;
      overhead_tracker->application_time += diff;
      previous_profiling_time = current;
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::end_runtime_call(void)
    //--------------------------------------------------------------------------
    {
      if (implicit_reference_tracker != NULL)
      {
        delete implicit_reference_tracker;
        implicit_reference_tracker = NULL;
      }
      if (overhead_tracker == NULL)
        return;
      const long long current = Realm::Clock::current_time_in_nanoseconds();
      const long long diff = current - previous_profiling_time;
      overhead_tracker->runtime_time += diff;
      previous_profiling_time = current;
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::begin_task_wait(bool from_runtime)
    //--------------------------------------------------------------------------
    {
      if (overhead_tracker == NULL)
        return;
      const long long current = Realm::Clock::current_time_in_nanoseconds();
      const long long diff = current - previous_profiling_time;
      if (from_runtime)
        overhead_tracker->runtime_time += diff;
      else
        overhead_tracker->application_time += diff;
      previous_profiling_time = current;
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::end_task_wait(void)
    //--------------------------------------------------------------------------
    {
      if (overhead_tracker == NULL)
        return;
      const long long current = Realm::Clock::current_time_in_nanoseconds();
      const long long diff = current - previous_profiling_time;
      overhead_tracker->wait_time += diff;
      previous_profiling_time = current;
    }

  };
};


#endif // __LEGION_CONTEXT_H__

