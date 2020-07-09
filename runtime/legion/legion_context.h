/* Copyright 2020 Stanford University, NVIDIA Corporation
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
      TaskContext(Runtime *runtime, TaskOp *owner, int depth,
                  const std::vector<RegionRequirement> &reqs);
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
      inline unsigned get_tunable_index(void)
        { return total_tunable_count++; }
      inline UniqueID get_unique_id(void) const 
        { return get_context_uid(); }
      inline const char* get_task_name(void)
        { return get_task()->get_task_name(); }
      inline const std::vector<PhysicalRegion>& get_physical_regions(void) const
        { return physical_regions; }
      inline bool has_created_requirements(void) const
        { return !created_requirements.empty(); }
      inline TaskOp* get_owner_task(void) const { return owner_task; }
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
                                       AddressSpaceID target,
                                       bool replicate = false) = 0;
      virtual bool attempt_children_complete(void) = 0;
      virtual bool attempt_children_commit(void) = 0;
      virtual void inline_child_task(TaskOp *child) = 0;
      virtual VariantImpl* select_inline_variant(TaskOp *child) const = 0;
      virtual bool is_leaf_context(void) const;
      virtual bool is_inner_context(void) const;
      virtual void perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, RtEvent local_done,
                     RtEvent global_done, std::set<RtEvent> &preconditions);
      virtual void handle_registration_callback_effects(RtEvent effects) = 0;
      virtual void print_once(FILE *f, const char *message) const;
      virtual void log_once(Realm::LoggerMessage &message) const;
      virtual ShardID get_shard_id(void) const;
      virtual size_t get_num_shards(void) const;
      virtual Future consensus_match(const void *input, void *output,
                                     size_t num_elements, size_t element_size);
    public:
      virtual VariantID register_variant(const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  const CodeDescriptor &desc, bool ret, 
                                  VariantID vid, bool check_task_id);
      virtual TraceID generate_dynamic_trace_id(void);
      virtual MapperID generate_dynamic_mapper_id(void);
      virtual ProjectionID generate_dynamic_projection_id(void);
      virtual ShardingID generate_dynamic_sharding_id(void);
      virtual TaskID generate_dynamic_task_id(void);
      virtual ReductionOpID generate_dynamic_reduction_id(void);
      virtual CustomSerdezID generate_dynamic_serdez_id(void);
      virtual bool perform_semantic_attach(bool &global);
      virtual void post_semantic_attach(void);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Domain &bounds,
                                            TypeTag type_tag);
      virtual IndexSpace create_index_space(const Future &future,
                                            TypeTag type_tag) = 0;
      virtual IndexSpace union_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace intersect_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace subtract_index_spaces(
                           IndexSpace left, IndexSpace right);
      virtual void create_shared_ownership(IndexSpace handle);
      virtual void destroy_index_space(IndexSpace handle,
                                       const bool unordered,
                                       const bool recurse) = 0;
      virtual void create_shared_ownership(IndexPartition handle);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered,
                                           const bool recurse) = 0;
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color, 
                                            bool dominates) = 0;
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color) = 0;
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color) = 0;
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag) = 0;
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                  const std::map<DomainPoint,Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color) = 0;
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind) = 0;
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag) = 0;
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag) = 0;
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag) = 0;
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag) = 0;
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color) = 0;
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle) = 0;
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual FieldSpace create_field_space(void);
      virtual FieldSpace create_field_space(const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id) = 0;
      virtual void create_shared_ownership(FieldSpace handle);
      virtual void destroy_field_space(FieldSpace handle,
                                       const bool unordered) = 0;
      virtual FieldID allocate_field(FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id) = 0;
      virtual void allocate_local_field(
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events) = 0;
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered) = 0;
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id) = 0;
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events) = 0;
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space, 
                               const std::set<FieldID> &to_free,
                               const bool unordered) = 0; 
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space,
                                            bool task_local);
      virtual void create_shared_ownership(LogicalRegion handle);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered) = 0;
      virtual FieldAllocatorImpl* create_field_allocator(FieldSpace handle,
                                                         bool unordered);
      virtual void destroy_field_allocator(FieldSpace handle);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const = 0;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const = 0;
    public:
      virtual Future execute_task(const TaskLauncher &launcher) = 0;
      virtual FutureMap execute_index_space(
                                         const IndexTaskLauncher &launcher) = 0;
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                   ReductionOpID redop, bool deterministic) = 0; 
      virtual Future reduce_future_map(const FutureMap &future_map,
                                   ReductionOpID redop, bool deterministic) = 0;
      virtual FutureMap construct_future_map(const Domain &domain,
                    const std::map<DomainPoint,Future> &futures,
                    RtUserEvent domain_deletion = RtUserEvent::NO_RT_USER_EVENT,
                                             bool internal = false) = 0;
      virtual PhysicalRegion map_region(const InlineLauncher &launcher) = 0;
      virtual ApEvent remap_region(PhysicalRegion region) = 0;
      virtual void unmap_region(PhysicalRegion region) = 0;
      virtual void fill_fields(const FillLauncher &launcher) = 0;
      virtual void fill_fields(const IndexFillLauncher &launcher) = 0;
      virtual void issue_copy(const CopyLauncher &launcher) = 0;
      virtual void issue_copy(const IndexCopyLauncher &launcher) = 0;
      virtual void issue_acquire(const AcquireLauncher &launcher) = 0;
      virtual void issue_release(const ReleaseLauncher &launcher) = 0;
      virtual PhysicalRegion attach_resource(
                                  const AttachLauncher &launcher) = 0;
      virtual Future detach_resource(PhysicalRegion region, 
                                     const bool flush,const bool unordered) = 0;
      virtual void progress_unordered_operations(void) = 0;
      virtual FutureMap execute_must_epoch(
                                 const MustEpochLauncher &launcher) = 0;
      virtual Future issue_timing_measurement(
                                    const TimingLauncher &launcher) = 0;
      virtual Future issue_mapping_fence(void) = 0;
      virtual Future issue_execution_fence(void) = 0;
      virtual void complete_frame(void) = 0;
      virtual Predicate create_predicate(const Future &f) = 0;
      virtual Predicate predicate_not(const Predicate &p) = 0;
      virtual Predicate create_predicate(const PredicateLauncher &launcher) = 0;
      virtual Future get_predicate_future(const Predicate &p) = 0;
    public:
      // Calls for barriers and dynamic collectives
      virtual ApBarrier create_phase_barrier(unsigned arrivals,
                                                ReductionOpID redop = 0,
                                                const void *init_value = NULL,
                                                size_t init_size = 0) = 0;
      virtual void destroy_phase_barrier(ApBarrier bar) = 0;
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier bar) = 0;
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                                             const void *buffer, 
                                             size_t size,
                                             unsigned count) = 0;
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &f,
                                                    unsigned count) = 0;
      virtual Future get_dynamic_collective_result(DynamicCollective dc) = 0;
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc) = 0;
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
               const std::vector<StaticDependence> *dependences) = 0;
      virtual void register_new_internal_operation(InternalOp *op) = 0;
      virtual size_t register_new_close_operation(CloseOp *op) = 0;
      virtual size_t register_new_summary_operation(TraceSummaryOp *op) = 0;
      virtual ApEvent add_to_dependence_queue(Operation *op, 
                                              bool unordered = false,
                                              bool outermost = true) = 0;
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
#ifdef LEGION_MALLOC_INSTANCES
                                          uintptr_t allocation = 0,
#endif
                                          PhysicalInstance instance = 
                                            PhysicalInstance::NO_INST) = 0;
      virtual void register_executing_child(Operation *op) = 0;
      virtual void register_child_executed(Operation *op) = 0;
      virtual void register_child_complete(Operation *op) = 0;
      virtual void register_child_commit(Operation *op) = 0; 
      virtual void unregister_child_operation(Operation *op) = 0;
      virtual ApEvent register_implicit_dependences(Operation *op) = 0;
    public:
      virtual RtEvent get_current_mapping_fence_event(void) = 0;
      virtual ApEvent get_current_execution_fence_event(void) = 0;
      // Break this into two pieces since we know that there are some
      // kinds of operations (like deletions) that want to act like 
      // one-sided fences (e.g. waiting on everything before) but not
      // preventing re-ordering for things afterwards
      virtual ApEvent perform_fence_analysis(Operation *op, 
                                        bool mapping, bool execution) = 0;
      virtual void update_current_fence(FenceOp *op, 
                                        bool mapping, bool execution) = 0;
      virtual void update_current_implicit(Operation *op) = 0;
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
        bool static_trace, const std::set<RegionTreeID> *managed, bool dep) = 0;
      virtual void end_trace(TraceID tid, bool deprecated) = 0;
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
      virtual RtEvent decrement_pending(TaskOp *child) = 0;
      virtual RtEvent decrement_pending(bool need_deferral) = 0;
      virtual void increment_frame(void) = 0;
      virtual void decrement_frame(void) = 0;
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(const LogicalUser &user,
                                               RegionTreeNode *node) = 0;
#else
      virtual MergeCloseOp* get_merge_close_op(void) = 0;
#endif
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index) = 0;
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent) = 0;
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL) = 0;
      virtual InnerContext* find_top_context(void) = 0;
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events) = 0;
      virtual void invalidate_region_tree_contexts(void) = 0;
      virtual void send_back_created_state(AddressSpaceID target) = 0;
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source) = 0;
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(
                                                   Legion::Runtime *&runtime);
      virtual void end_task(const void *res, size_t res_size, bool owned,
#ifdef LEGION_MALLOC_INSTANCES
                    uintptr_t allocation = 0,
#endif
                    PhysicalInstance inst = PhysicalInstance::NO_INST) = 0;
      virtual void post_end_task(const void *res, 
                                 size_t res_size, bool owned) = 0;
      void begin_misspeculation(void);
      void end_misspeculation(const void *res, size_t res_size);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f) = 0;
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures) = 0;
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
      void destroy_user_lock(Reservation r);
      void destroy_user_barrier(ApBarrier b);
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
      bool check_region_dependence(RegionTreeID tid, IndexSpace space,
                                  const RegionRequirement &our_req,
                                  const RegionUsage &our_usage,
                                  const RegionRequirement &req,
                                  bool check_privileges = true) const;
      void register_inline_mapped_region(PhysicalRegion &region);
      void unregister_inline_mapped_region(PhysicalRegion &region);
    public:
      bool safe_cast(RegionTreeForest *forest, IndexSpace handle, 
                     const void *realm_point, TypeTag type_tag);
      bool is_region_mapped(unsigned idx);
      void clone_requirement(unsigned idx, RegionRequirement &target);
      int find_parent_region_req(const RegionRequirement &req, 
                                 bool check_privilege = true);
      unsigned find_parent_region(unsigned idx, TaskOp *task);
      unsigned find_parent_index_region(unsigned idx, TaskOp *task);
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
    public:
      void add_physical_region(const RegionRequirement &req, bool mapped,
          MapperID mid, MappingTagID tag, ApUserEvent unmap_event,
          bool virtual_mapped, const InstanceSet &physical_instances);
      void initialize_overhead_tracker(void);
      void unmap_all_regions(void); 
      inline void begin_runtime_call(void);
      inline void end_runtime_call(void);
      inline void begin_task_wait(bool from_runtime);
      inline void end_task_wait(void); 
      void remap_unmapped_regions(LegionTrace *current_trace,
                           const std::vector<PhysicalRegion> &unmapped_regions);
    public:
      void* get_local_task_variable(LocalVariableID id);
      void set_local_task_variable(LocalVariableID id, const void *value,
                                   void (*destructor)(void*));
    public:
      void yield(void);
    protected:
      Future predicate_task_false(const TaskLauncher &launcher);
      FutureMap predicate_index_task_false(const IndexTaskLauncher &launcher);
      Future predicate_index_task_reduce_false(const IndexTaskLauncher &launch);
    public:
      // Find an index space name for a concrete launch domain
      IndexSpace find_index_launch_space(const Domain &domain);
    public:
      Runtime *const runtime;
      TaskOp *const owner_task;
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
      // These next two data structure don't need a lock becaue
      // they are only mutated by the application task 
      std::vector<PhysicalRegion>               physical_regions;
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts
      LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::tracked
                                                inline_regions; 
    protected:
      Processor                             executing_processor;
      unsigned                              total_tunable_count;
    protected:
      Mapping::ProfilingMeasurements::RuntimeOverhead *overhead_tracker;
      long long                                previous_profiling_time;
    protected:
      // Resources that can build up over a task's lifetime
      LegionDeque<Reservation,TASK_RESERVATION_ALLOC>::tracked context_locks;
      LegionDeque<ApBarrier,TASK_BARRIER_ALLOC>::tracked context_barriers;
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
      RtEvent pending_done;
      bool task_executed;
      bool has_inline_accessor;
      bool mutable_priority;
    protected: 
      bool children_complete_invoked;
      bool children_commit_invoked;
#ifdef LEGION_SPY
    protected:
      UniqueID current_fence_uid;
#endif
    };

    class InnerContext : public TaskContext {
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
#ifdef LEGION_MALLOC_INSTANCES
                     uintptr_t alloc,
#endif
                     PhysicalInstance i, RtEvent w)
          : context(ctx), index(idx), result(r), size(s), 
#ifdef LEGION_MALLOC_INSTANCES
            allocation(alloc),
#endif
            instance(i), wait_on(w) { }
      public:
        inline bool operator<(const PostTaskArgs &rhs) const
          { return index < rhs.index; }
      public:
        TaskContext *context;
        size_t index;
        const void *result;
        size_t size;
#ifdef LEGION_MALLOC_INSTANCES
        uintptr_t allocation;
#endif
        PhysicalInstance instance;
        RtEvent wait_on;
      };
      struct PostDecrementArgs : public LgTaskArgs<PostDecrementArgs> {
      public:
        static const LgTaskID TASK_ID = LG_POST_DECREMENT_TASK_ID;
      public:
        PostDecrementArgs(InnerContext *ctx)
          : LgTaskArgs<PostDecrementArgs>(ctx->get_context_uid()),
            parent_ctx(ctx) { }
      public:
        InnerContext *const parent_ctx;
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
    public:
      InnerContext(Runtime *runtime, TaskOp *owner, int depth, bool full_inner,
                   const std::vector<RegionRequirement> &reqs,
                   const std::vector<unsigned> &parent_indexes,
                   const std::vector<bool> &virt_mapped, UniqueID context_uid, 
                   ApEvent execution_fence, bool remote = false);
      InnerContext(const InnerContext &rhs);
      virtual ~InnerContext(void);
    public:
      InnerContext& operator=(const InnerContext &rhs);
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<LogicalRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<std::pair<FieldSpace,FieldID> > &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<FieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<std::pair<IndexSpace,bool> > &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<std::pair<IndexPartition,bool> > &deleted_partitions,
              std::set<RtEvent> &preconditions);
    protected:
      void register_region_creations(
                     std::map<LogicalRegion,unsigned> &regions);
      void register_region_deletions(ApEvent precondition,
                     const std::map<Operation*,GenerationID> &dependences,
                     std::vector<LogicalRegion> &regions,
                     std::set<RtEvent> &preconditions);
      void register_field_creations(
            std::set<std::pair<FieldSpace,FieldID> > &fields);
      void register_field_deletions(ApEvent precondition,
            const std::map<Operation*,GenerationID> &dependences,
            std::vector<std::pair<FieldSpace,FieldID> > &fields,
            std::set<RtEvent> &preconditions);
      void register_field_space_creations(
                          std::map<FieldSpace,unsigned> &spaces);
      void register_latent_field_spaces(
            std::map<FieldSpace,std::set<LogicalRegion> > &spaces);
      void register_field_space_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<FieldSpace> &spaces,
                          std::set<RtEvent> &preconditions);
      void register_index_space_creations(
                          std::map<IndexSpace,unsigned> &spaces);
      void register_index_space_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<std::pair<IndexSpace,bool> > &spaces,
                          std::set<RtEvent> &preconditions);
      void register_index_partition_creations(
                          std::map<IndexPartition,unsigned> &parts);
      void register_index_partition_deletions(ApEvent precondition,
                          const std::map<Operation*,GenerationID> &dependences,
                          std::vector<std::pair<IndexPartition,bool> > &parts,
                          std::set<RtEvent> &preconditions);
      ApEvent compute_return_deletion_dependences(size_t return_index,
                          std::map<Operation*,GenerationID> &dependences);
    public:
      void print_children(void);
      void perform_window_wait(void);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual UniqueID get_context_uid(void) const;
      virtual bool is_inner_context(void) const;
      virtual void pack_remote_context(Serializer &rez, 
          AddressSpaceID target, bool replicate = false);
      virtual void unpack_remote_context(Deserializer &derez,
                                         std::set<RtEvent> &preconditions);
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle,
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
      virtual void handle_registration_callback_effects(RtEvent effects);
      virtual void analyze_free_local_fields(FieldSpace handle,
                                  const std::vector<FieldID> &local_to_free,
                                  std::vector<unsigned> &local_field_indexes);
      virtual void remove_deleted_local_fields(FieldSpace space,
                                 const std::vector<FieldID> &to_remove);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Future &future, TypeTag tag);
      virtual void destroy_index_space(IndexSpace handle, const bool unordered,
                                       const bool recurse);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered,
                                           const bool recurse);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                  const std::map<DomainPoint,Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual void verify_partition(IndexPartition pid, PartitionKind kind,
                                    const char *function_name);
      static void handle_partition_verification(const void *args);
      virtual FieldSpace create_field_space(void);
      virtual FieldSpace create_field_space(const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space,
                               const std::set<FieldID> &to_free,
                               const bool unordered);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const;
    public:
      // Find an index space name for a concrete launch domain
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic);
      virtual FutureMap construct_future_map(const Domain &domain,
                    const std::map<DomainPoint,Future> &futures,
                    RtUserEvent domain_deletion = RtUserEvent::NO_RT_USER_EVENT,
                                             bool internal = false);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered);
      virtual void progress_unordered_operations(void);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future issue_mapping_fence(void);
      virtual Future issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // Calls for barriers and dynamic collectives
      virtual ApBarrier create_phase_barrier(unsigned arrivals,
                                                ReductionOpID redop = 0,
                                                const void *init_value = NULL,
                                                size_t init_size = 0);
      virtual void destroy_phase_barrier(ApBarrier bar);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier bar);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                                             const void *buffer, 
                                             size_t size,
                                             unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &f,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual void register_new_internal_operation(InternalOp *op);
      // Must be called while holding the dependence lock
      virtual void insert_unordered_ops(AutoLock &d_lock, const bool end_task,
                                        const bool progress);
      virtual size_t register_new_close_operation(CloseOp *op);
      virtual size_t register_new_summary_operation(TraceSummaryOp *op);
      void add_to_prepipeline_queue(Operation *op);
      bool process_prepipeline_stage(void);
      virtual ApEvent add_to_dependence_queue(Operation *op, 
                                              bool unordered = false,
                                              bool outermost = true);
      void process_dependence_stage(void);
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
#ifdef LEGION_MALLOC_INSTANCES
                                          uintptr_t allocation = 0,
#endif
                                          PhysicalInstance instance =
                                            PhysicalInstance::NO_INST);
      bool process_post_end_tasks(void);
      virtual void register_executing_child(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual ApEvent register_implicit_dependences(Operation *op);
    public:
      virtual RtEvent get_current_mapping_fence_event(void);
      virtual ApEvent get_current_execution_fence_event(void);
      virtual ApEvent perform_fence_analysis(Operation *op,
                                             bool mapping, bool execution);
      virtual void update_current_fence(FenceOp *op,
                                        bool mapping, bool execution);
      virtual void update_current_implicit(Operation *op);
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep);
      virtual void end_trace(TraceID tid, bool deprecated);
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
      virtual RtEvent decrement_pending(TaskOp *child);
      virtual RtEvent decrement_pending(bool need_deferral);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(const LogicalUser &user,
                                               RegionTreeNode *node);
#else
      virtual MergeCloseOp* get_merge_close_op(void);
#endif
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
    public:
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      void configure_context(MapperManager *mapper, TaskPriority priority);
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      void invalidate_created_requirement_contexts(void);
      virtual void invalidate_remote_tree_contexts(Deserializer &derez);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source);
      virtual FillView* find_or_create_fill_view(FillOp *op, 
                             std::set<RtEvent> &map_applied_events,
                             const void *value, const size_t value_size);
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
#ifdef LEGION_MALLOC_INSTANCES
                            uintptr_t allocation = 0,
#endif
                            PhysicalInstance inst = PhysicalInstance::NO_INST);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                       std::vector<Future> &contributions);
    public:
      virtual ShardingFunction* find_sharding_function(ShardingID sid);
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority);
    public:
      static void handle_compute_equivalence_sets_request(Deserializer &derez,
                                     Runtime *runtime, AddressSpaceID source);
    public:
      void invalidate_remote_contexts(void);
      void clear_instance_top_views(void); 
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
      EquivalenceSet* find_or_create_top_equivalence_set(RegionTreeID tree_id); 
    public:
      void clone_local_fields(
          std::map<FieldSpace,std::vector<LocalFieldInfo> > &child_local) const;
#ifdef DEBUG_LEGION
      // This is a helpful debug method that can be useful when called from
      // a debugger to find the earliest operation that hasn't mapped yet
      // which is especially useful when debugging scheduler hangs
      Operation* get_earliest(void) const;
#endif
    public:
      static void handle_prepipeline_stage(const void *args);
      static void handle_dependence_stage(const void *args);
      static void handle_post_end_task(const void *args);
    public:
      const RegionTreeContext tree_context; 
      const UniqueID context_uid;
      const bool remote_context;
      const bool full_inner_context;
    protected:
      Mapper::ContextConfigOutput           context_configuration;
    protected:
      const std::vector<unsigned>           &parent_req_indexes;
      const std::vector<bool>               &virtual_mapped;
    protected:
      mutable LocalLock                     child_op_lock;
      // Track whether this task has finished executing
      size_t total_children_count; // total number of sub-operations
      size_t total_close_count; 
      size_t total_summary_count;
      size_t outstanding_children_count;
      LegionMap<Operation*,GenerationID,
                EXECUTING_CHILD_ALLOC>::tracked executing_children;
      LegionMap<Operation*,GenerationID,
                EXECUTED_CHILD_ALLOC>::tracked executed_children;
      LegionMap<Operation*,GenerationID,
                COMPLETE_CHILD_ALLOC>::tracked complete_children; 
      // For tracking any operations that come from outside the
      // task like a garbage collector that need to be inserted
      // into the stream of operations from the task
      std::list<Operation*> unordered_ops;
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
      mutable LocalLock                               post_task_lock;
      std::list<PostTaskArgs>                         post_task_queue;
      CompletionQueue                                 post_task_comp_queue;
    protected:
      // Traces for this task's execution
      LegionMap<TraceID,LegionTrace*,TASK_TRACES_ALLOC>::tracked traces;
      LegionTrace *current_trace;
      LegionTrace *previous_trace;
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
      // Event used to order operations to the runtime
      RtEvent context_order_event;
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
      // Tracking information for dynamic collectives
      mutable LocalLock                       collective_lock;
      std::map<ApEvent,std::vector<Future> >  collective_contributions;
    protected:
      // Track information for locally allocated fields
      mutable LocalLock                                 local_field_lock;
      std::map<FieldSpace,std::vector<LocalFieldInfo> > local_field_infos;
    protected:
      // Cache for fill views
      mutable LocalLock     fill_view_lock;            
      std::list<FillView*>  fill_view_cache;
      static const size_t MAX_FILL_VIEW_CACHE_SIZE = 64;
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
      virtual void pack_remote_context(Serializer &rez, 
          AddressSpaceID target, bool replicate = false);
      virtual TaskContext* find_parent_context(void);
    public:
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
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
     * \class ReplicateContext
     * A replicate context is a special kind of inner context for
     * executing control-replicated tasks.
     */
    class ReplicateContext : public InnerContext {
    public: 
      struct ISBroadcast {
      public:
        ISBroadcast(void) : expr_id(0), did(0), double_buffer(false) { }
        ISBroadcast(IndexSpaceID i, IndexTreeID t, IndexSpaceExprID e, 
                    DistributedID d, bool db)
          : space_id(i), tid(t), expr_id(e), did(d), double_buffer(db) { }
      public:
        IndexSpaceID space_id;
        IndexTreeID tid;
        IndexSpaceExprID expr_id;
        DistributedID did;
        bool double_buffer;
      };
      struct IPBroadcast {
      public:
        IPBroadcast(void) : did(0), double_buffer(false) { }
        IPBroadcast(IndexPartitionID p, DistributedID d, bool db) 
          : pid(p), did(d), double_buffer(db) { }
      public:
        IndexPartitionID pid;
        DistributedID did;
        bool double_buffer;
      };
      struct FSBroadcast { 
      public:
        FSBroadcast(void) : did(0), double_buffer(false) { }
        FSBroadcast(FieldSpaceID i, DistributedID d, bool db) 
          : space_id(i), did(d), double_buffer(db) { }
      public:
        FieldSpaceID space_id;
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
        LRBroadcast(void) : tid(0), double_buffer(0) { }
        LRBroadcast(RegionTreeID t, bool db) :
          tid(t), double_buffer(db) { }
      public:
        RegionTreeID tid;
        bool double_buffer;
      };
      struct IntraSpaceDeps {
      public:
        std::map<ShardID,RtEvent> ready_deps;
        std::map<ShardID,RtUserEvent> pending_deps;
      };
      enum ReplicateAPICall {
        REPLICATE_PERFORM_REGISTRATION_CALLBACK,
        REPLICATE_CONSENSUS_MATCH,
        REPLICATE_REGISTER_TASK_VARIANT,
        REPLICATE_GENERATE_DYNAMIC_TRACE_ID,
        REPLICATE_GENERATE_DYNAMIC_MAPPER_ID,
        REPLICATE_GENERATE_DYNAMIC_PROJECTION_ID,
        REPLICATE_GENERATE_DYNAMIC_SHARDING_ID,
        REPLICATE_GENERATE_DYNAMIC_TASK_ID,
        REPLICATE_GENERATE_DYNAMIC_REDUCTION_ID,
        REPLICATE_GENERATE_DYNAMIC_SERDEZ_ID,
        REPLICATE_CREATE_INDEX_SPACE,
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
      };
    public:
      ReplicateContext(Runtime *runtime, ShardTask *owner,int d,bool full_inner,
                       const std::vector<RegionRequirement> &reqs,
                       const std::vector<unsigned> &parent_indexes,
                       const std::vector<bool> &virt_mapped,
                       UniqueID context_uid, ApEvent execution_fence_event,
                       ShardManager *manager);
      ReplicateContext(const ReplicateContext &rhs);
      virtual ~ReplicateContext(void);
    public:
      ReplicateContext& operator=(const ReplicateContext &rhs);
    public:
      inline int get_shard_collective_radix(void) const
        { return shard_collective_radix; }
      inline int get_shard_collective_log_radix(void) const
        { return shard_collective_log_radix; }
      inline int get_shard_collective_stages(void) const
        { return shard_collective_stages; }
      inline int get_shard_collective_participating_shards(void) const
        { return shard_collective_participating_shards; }
      inline int get_shard_collective_last_radix(void) const
        { return shard_collective_last_radix; }
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<LogicalRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<std::pair<FieldSpace,FieldID> > &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<FieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<std::pair<IndexSpace,bool> > &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<std::pair<IndexPartition,bool> > &deleted_partitions,
              std::set<RtEvent> &preconditions);
    protected:
      void receive_replicate_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<LogicalRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<std::pair<FieldSpace,FieldID> > &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<FieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<std::pair<IndexSpace,bool> > &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<std::pair<IndexPartition,bool> > &deleted_partitions,
              std::set<RtEvent> &preconditions, RtBarrier &ready_barrier, 
              RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
      void register_region_deletions(ApEvent precondition,
                     const std::map<Operation*,GenerationID> &dependences,
                     std::vector<LogicalRegion> &regions,
                     std::set<RtEvent> &preconditions, RtBarrier &ready_barrier,
                     RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
      void register_field_deletions(ApEvent precondition,
            const std::map<Operation*,GenerationID> &dependences,
            std::vector<std::pair<FieldSpace,FieldID> > &fields,
            std::set<RtEvent> &preconditions, RtBarrier &ready_barrier,
            RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
      void register_field_space_deletions(ApEvent precondition,
                    const std::map<Operation*,GenerationID> &dependences,
                    std::vector<FieldSpace> &spaces,
                    std::set<RtEvent> &preconditions, RtBarrier &ready_barrier,
                    RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
      void register_index_space_deletions(ApEvent precondition,
                    const std::map<Operation*,GenerationID> &dependences,
                    std::vector<std::pair<IndexSpace,bool> > &spaces,
                    std::set<RtEvent> &preconditions, RtBarrier &ready_barrier,
                    RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
      void register_index_partition_deletions(ApEvent precondition,
                    const std::map<Operation*,GenerationID> &dependences,
                    std::vector<std::pair<IndexPartition,bool> > &parts,
                    std::set<RtEvent> &preconditions, RtBarrier &ready_barrier,
                    RtBarrier &mapped_barrier, RtBarrier &execution_barrier);
    public:
      void perform_replicated_region_deletions(
                     std::vector<LogicalRegion> &regions,
                     std::set<RtEvent> &preconditions);
      void perform_replicated_field_deletions(
            std::vector<std::pair<FieldSpace,FieldID> > &fields,
            std::set<RtEvent> &preconditions);
      void perform_replicated_field_space_deletions(
                          std::vector<FieldSpace> &spaces,
                          std::set<RtEvent> &preconditions);
      void perform_replicated_index_space_deletions(
                          std::vector<IndexSpace> &spaces,
                          std::set<RtEvent> &preconditions);
      void perform_replicated_index_partition_deletions(
                          std::vector<IndexPartition> &parts,
                          std::set<RtEvent> &preconditions);
    public:
      virtual void perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, RtEvent local_done,
                     RtEvent global_done, std::set<RtEvent> &preconditions);
      virtual void handle_registration_callback_effects(RtEvent effects);
      virtual void print_once(FILE *f, const char *message) const;
      virtual void log_once(Realm::LoggerMessage &message) const;
      virtual ShardID get_shard_id(void) const;
      virtual size_t get_num_shards(void) const;
      virtual Future consensus_match(const void *input, void *output,
                                     size_t num_elements, size_t element_size);
    public:
      virtual VariantID register_variant(const TaskVariantRegistrar &registrar,
                                  const void *user_data, size_t user_data_size,
                                  const CodeDescriptor &desc, bool ret, 
                                  VariantID vid, bool check_task_id);
      virtual TraceID generate_dynamic_trace_id(void);
      virtual MapperID generate_dynamic_mapper_id(void);
      virtual ProjectionID generate_dynamic_projection_id(void);
      virtual ShardingID generate_dynamic_sharding_id(void);
      virtual TaskID generate_dynamic_task_id(void);
      virtual ReductionOpID generate_dynamic_reduction_id(void);
      virtual CustomSerdezID generate_dynamic_serdez_id(void);
      virtual bool perform_semantic_attach(bool &global);
      virtual void post_semantic_attach(void);
    public:
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion handle);
      virtual void invalidate_region_tree_contexts(void);
    public:
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle,
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Domain &domain, 
                                            TypeTag type_tag);
      virtual IndexSpace create_index_space(const Future &future, 
                                            TypeTag type_tag);
      virtual IndexSpace union_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace intersect_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace subtract_index_spaces(
                           IndexSpace left, IndexSpace right);
      virtual void create_shared_ownership(IndexSpace handle);
      virtual void destroy_index_space(IndexSpace handle, 
                                       const bool unordered,
                                       const bool recurse);
      virtual void create_shared_ownership(IndexPartition handle);
      virtual void destroy_index_partition(IndexPartition handle, 
                                           const bool unordered,
                                           const bool recurse);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                  const std::map<DomainPoint,Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual void verify_partition(IndexPartition pid, PartitionKind kind,
                                    const char *function_name);
      virtual FieldSpace create_field_space(void);
      virtual FieldSpace create_field_space(const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual void create_shared_ownership(FieldSpace handle);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered);
      virtual FieldID allocate_field(FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space, 
                               const std::set<FieldID> &to_free,
                               const bool unordered);
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space,
                                            bool task_local);
      virtual void create_shared_ownership(LogicalRegion handle);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered);
    public:
      virtual FieldAllocatorImpl* create_field_allocator(FieldSpace handle,
                                                         bool unordered);
      virtual void destroy_field_allocator(FieldSpace handle);
    public:
      virtual void insert_unordered_ops(AutoLock &d_lock, const bool end_task,
                                        const bool progress);
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(PhysicalRegion region);
      // Unmapping region is the same as for an inner context
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future issue_mapping_fence(void);
      virtual Future issue_execution_fence(void);
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep);
      virtual void end_trace(TraceID tid, bool deprecated);
      virtual ApEvent add_to_dependence_queue(Operation *op, 
                                              bool unordered = false,
                                              bool outermost = true);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                       std::vector<Future> &contributions);
    public:
      // Calls for barriers and dynamic collectives
      virtual ApBarrier create_phase_barrier(unsigned arrivals,
                                             ReductionOpID redop = 0,
                                             const void *init_value = NULL,
                                             size_t init_size = 0);
      virtual void destroy_phase_barrier(ApBarrier bar);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier bar);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                                             const void *buffer, 
                                             size_t size,
                                             unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &f,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(const LogicalUser &user,
                                               RegionTreeNode *node);
#else
      virtual MergeCloseOp* get_merge_close_op(void);
#endif
    public:
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target,
                                       bool replicate = false);
    public:
      virtual ShardingFunction* find_sharding_function(ShardingID sid);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source);
      InstanceView* create_replicate_instance_top_view(PhysicalManager *manager,
                                                       AddressSpaceID source);
      void record_replicate_instance_top_view(PhysicalManager *manager, 
                                              InstanceView *result);
    public:
      void exchange_common_resources(void);
      void handle_collective_message(Deserializer &derez);
      void handle_future_map_request(Deserializer &derez);
      void handle_equivalence_set_request(Deserializer &derez);
      void handle_equivalence_set_response(RegionTreeID tree_id, 
                                           EquivalenceSet *result);
      static void handle_eq_response(Deserializer &derez, Runtime *rt);
      void handle_resource_update(Deserializer &derez,
                                  std::set<RtEvent> &applied);
      void handle_trace_update(Deserializer &derez, AddressSpaceID source);
      ApBarrier handle_find_trace_shard_event(size_t temp_index, ApEvent event,
                                              ShardID remote_shard);
      void record_intra_space_dependence(size_t context_index, 
          const DomainPoint &point, RtEvent point_mapped, ShardID next_shard);
      void handle_intra_space_dependence(Deserializer &derez);
    public:
      void increase_pending_index_spaces(unsigned count, bool double_buffer);
      void increase_pending_partitions(unsigned count, bool double_buffer);
      void increase_pending_field_spaces(unsigned count, bool double_buffer);
      void increase_pending_fields(unsigned count, bool double_buffer);
      void increase_pending_region_trees(unsigned count, bool double_buffer);
      bool create_shard_partition(IndexPartition &pid,
          IndexSpace parent, IndexSpace color_space, PartitionKind part_kind,
          LegionColor partition_color, bool color_generated,
          ValueBroadcast<bool> *disjoint_result = NULL,
          ApBarrier partition_ready = ApBarrier::NO_AP_BARRIER);
    public:
      // Collective methods
      CollectiveID get_next_collective_index(CollectiveIndexLocation loc);
      void register_collective(ShardCollective *collective);
      ShardCollective* find_or_buffer_collective(Deserializer &derez);
      void unregister_collective(ShardCollective *collective);
    public:
      // Future map methods
      unsigned peek_next_future_map_barrier_index(void) const;
      RtBarrier get_next_future_map_barrier(void);
      void register_future_map(ReplFutureMapImpl *map);
      ReplFutureMapImpl* find_or_buffer_future_map_request(Deserializer &derez);
      void unregister_future_map(ReplFutureMapImpl *map);
    public:
      // Physical template methods
      size_t register_trace_template(ShardedPhysicalTemplate *phy_template);
      ShardedPhysicalTemplate* find_or_buffer_trace_update(Deserializer &derez,
                                                         AddressSpaceID source);
      void unregister_trace_template(size_t template_index);
    public:
      // Fence barrier methods
      RtBarrier get_next_mapping_fence_barrier(void);
      ApBarrier get_next_execution_fence_barrier(void);
      RtBarrier get_next_trace_recording_barrier(void);
      RtBarrier get_next_summary_fence_barrier(void);
      inline void advance_replicate_barrier(RtBarrier &bar, size_t arrivals)
        {
          Runtime::advance_barrier(bar);
          if (!bar.exists())
            create_new_replicate_barrier(bar, arrivals);
        }
      inline void advance_replicate_barrier(ApBarrier &bar, size_t arrivals)
        {
          Runtime::advance_barrier(bar);
          if (!bar.exists())
            create_new_replicate_barrier(bar, arrivals);
        }
    protected:
      // These can only be called inside the task for this context
      // since they assume that all the shards are aligned and doing
      // the same calls for the same operations in the same order
      void create_new_replicate_barrier(RtBarrier &bar, size_t arrivals);
      void create_new_replicate_barrier(ApBarrier &bar, size_t arrivals);
    public:
      void verify_replicable(Murmur3Hasher &hasher, const char *func_name);
    public:
      // A little help for ConsensusMatchExchange since it is templated
      static void help_complete_future(Future &f, const void *ptr,
                                       size_t size, bool own);
    public:
      ShardTask *const owner_shard;
      ShardManager *const shard_manager;
      const size_t total_shards;
    protected:
      // These barriers are used to identify when close operations are mapped
      std::vector<RtBarrier>  close_mapped_barriers;
      unsigned                next_close_mapped_bar_index;
      // These barriers are for signaling when indirect copies are done
      std::vector<ApBarrier>  indirection_barriers;
      unsigned                next_indirection_bar_index;
      // These barriers are used for signaling when future maps can be reclaimed
      std::vector<RtBarrier>  future_map_barriers;
      unsigned                next_future_map_bar_index;
    protected:
      std::map<std::pair<size_t,DomainPoint>,IntraSpaceDeps> intra_space_deps;
    protected:
      // Store the global owner shard and local owner shard for allocation
      std::map<FieldSpace,
               std::pair<ShardID,bool> > field_allocator_owner_shards;
    protected:
      ShardID index_space_allocator_shard;
      ShardID index_partition_allocator_shard;
      ShardID field_space_allocator_shard;
      ShardID field_allocator_shard;
      ShardID logical_region_allocator_shard;
      ShardID dynamic_id_allocator_shard;
    protected:
      ApBarrier pending_partition_barrier;
      RtBarrier creation_barrier;
      RtBarrier deletion_ready_barrier;
      RtBarrier deletion_mapping_barrier;
      RtBarrier deletion_execution_barrier;
      RtBarrier inline_mapping_barrier;
      RtBarrier external_resource_barrier;
      RtBarrier mapping_fence_barrier;
      RtBarrier trace_recording_barrier;
      RtBarrier summary_fence_barrier;
      ApBarrier execution_fence_barrier;
      ApBarrier attach_broadcast_barrier;
      ApBarrier attach_reduce_barrier;
      RtBarrier dependent_partition_barrier;
      RtBarrier semantic_attach_barrier;
      ApBarrier inorder_barrier;
#ifdef DEBUG_LEGION_COLLECTIVES
    protected:
      RtBarrier collective_check_barrier;
      RtBarrier close_check_barrier;
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
      std::map<CollectiveID,ShardCollective*> collectives;
      std::map<CollectiveID,std::vector<
                std::pair<void*,size_t> > > pending_collective_updates;
      // Use this for creating new summary barriers in the dependence
      // analsyis stage of the pipeline. Our use of this variable is only
      // safe as long as we know we have more generations of phase barriers
      // than we can have outstanding replays, which is usually very true
      volatile CollectiveID trace_recording_collective_id;
      volatile CollectiveID summary_collective_id;
    protected:
      // Pending allocations of various resources
      std::deque<std::pair<ValueBroadcast<ISBroadcast>*,bool> > 
                                            pending_index_spaces;
      std::deque<std::pair<ValueBroadcast<IPBroadcast>*,ShardID> >
                                            pending_index_partitions;
      std::deque<std::pair<ValueBroadcast<FSBroadcast>*,bool> >
                                            pending_field_spaces;
      std::deque<std::pair<ValueBroadcast<FIDBroadcast>*,bool> >
                                            pending_fields;
      std::deque<std::pair<ValueBroadcast<LRBroadcast>*,bool> >
                                            pending_region_trees;
    protected:
      std::map<RtEvent,ReplFutureMapImpl*> future_maps;
      std::map<RtEvent,std::vector<
                std::pair<void*,size_t> > > pending_future_map_requests;
    protected:
      std::map<size_t,ShardedPhysicalTemplate*> physical_templates;
      struct PendingTemplateUpdate {
      public:
        PendingTemplateUpdate(void)
          : ptr(NULL), size(0), source(0) { }
        PendingTemplateUpdate(void *p, size_t s, AddressSpaceID src)
          : ptr(p), size(s), source(src) { }
      public:
        void *ptr;
        size_t size;
        AddressSpaceID source;
      };
      std::map<size_t/*template index*/,
        std::vector<PendingTemplateUpdate> > pending_template_updates;
      size_t next_physical_template_index;
    protected:
      // Different from pending_top_views as this applies to our requests
      std::map<PhysicalManager*,RtUserEvent> pending_request_views;
      std::map<RegionTreeID,RtUserEvent> pending_tree_requests;
    protected:
      std::map<std::pair<unsigned,unsigned>,RtBarrier> ready_clone_barriers;
      std::map<std::pair<unsigned,unsigned>,RtUserEvent> pending_clone_barriers;
    protected:
      unsigned next_replicate_bar_index;
      unsigned next_trace_bar_index;
      unsigned next_summary_bar_index;
    protected:
      static const unsigned MIN_UNORDERED_OPS_EPOCH = 32;
      static const unsigned MAX_UNORDERED_OPS_EPOCH = 32768;
      unsigned unordered_ops_counter;
      unsigned unordered_ops_epoch;
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
      virtual size_t get_context_index(void) const; 
      virtual void set_context_index(size_t index);
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
      virtual InnerContext* find_top_context(void);
    public:
      virtual RtEvent compute_equivalence_sets(VersionManager *manager,
                        RegionTreeID tree_id, IndexSpace handle,
                        IndexSpaceExpression *expr, const FieldMask &mask,
                        AddressSpaceID source);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
      virtual void record_using_physical_context(LogicalRegion handle);
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source);
      virtual void invalidate_region_tree_contexts(void);
      virtual void invalidate_remote_tree_contexts(Deserializer &derez);
    public:
      virtual ShardingFunction* find_sharding_function(ShardingID sid);
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
      ShardManager *shard_manager; // if we're lucky and one is already here
    protected:
      ApEvent remote_completion_event;
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
    protected:
      // For remote replicate contexts
      size_t total_shards;
      ReplicationID repl_id;
      std::map<ShardingID,ShardingFunction*> sharding_functions;
    };

    /**
     * \class LeafContext
     * A context for the execution of a leaf task
     */
    class LeafContext : public TaskContext {
    public:
      LeafContext(Runtime *runtime, TaskOp *owner);
      LeafContext(const LeafContext &rhs);
      virtual ~LeafContext(void);
    public:
      LeafContext& operator=(const LeafContext &rhs);
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<LogicalRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<std::pair<FieldSpace,FieldID> > &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<FieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<std::pair<IndexSpace,bool> > &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<std::pair<IndexPartition,bool> > &deleted_partitions,
              std::set<RtEvent> &preconditions);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual void pack_remote_context(Serializer &rez, 
          AddressSpaceID target, bool replicate = false);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
      virtual void handle_registration_callback_effects(RtEvent effects);
      virtual bool is_leaf_context(void) const;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Future &future, TypeTag tag);
      virtual void destroy_index_space(IndexSpace handle, 
                                       const bool unordered,
                                       const bool recurse);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered, 
                                           const bool recurse);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                  const std::map<DomainPoint,Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space,
                               const std::set<FieldID> &to_free,
                               const bool unordered);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const;
    public:
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic);
      virtual FutureMap construct_future_map(const Domain &domain,
                    const std::map<DomainPoint,Future> &futures,
                    RtUserEvent domain_deletion = RtUserEvent::NO_RT_USER_EVENT,
                                             bool internal = false);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered);
      virtual void progress_unordered_operations(void);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future issue_mapping_fence(void);
      virtual Future issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // Calls for barriers and dynamic collectives
      virtual ApBarrier create_phase_barrier(unsigned arrivals,
                                                ReductionOpID redop = 0,
                                                const void *init_value = NULL,
                                                size_t init_size = 0);
      virtual void destroy_phase_barrier(ApBarrier bar);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier bar);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                                             const void *buffer, 
                                             size_t size,
                                             unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &f,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual void register_new_internal_operation(InternalOp *op);
      virtual size_t register_new_close_operation(CloseOp *op);
      virtual size_t register_new_summary_operation(TraceSummaryOp *op);
      virtual ApEvent add_to_dependence_queue(Operation *op, 
                                              bool unordered = false,
                                              bool outermost = true);
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
#ifdef LEGION_MALLOC_INSTANCES
                                          uintptr_t allocation = 0,
#endif
                                          PhysicalInstance instance =
                                            PhysicalInstance::NO_INST);
      virtual void register_executing_child(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual ApEvent register_implicit_dependences(Operation *op);
    public:
      virtual RtEvent get_current_mapping_fence_event(void);
      virtual ApEvent get_current_execution_fence_event(void);
      virtual ApEvent perform_fence_analysis(Operation *op,
                                             bool mapping, bool execution);
      virtual void update_current_fence(FenceOp *op,
                                        bool mapping, bool execution);
      virtual void update_current_implicit(Operation *op);
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep);
      virtual void end_trace(TraceID tid, bool deprecated);
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
      virtual RtEvent decrement_pending(TaskOp *child);
      virtual RtEvent decrement_pending(bool need_deferral);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(const LogicalUser &user,
                                               RegionTreeNode *node);
#else
      virtual MergeCloseOp* get_merge_close_op(void);
#endif
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source);
    public:
      virtual void end_task(const void *res, size_t res_size, bool owned,
#ifdef LEGION_MALLOC_INSTANCES
                            uintptr_t allocation = 0,
#endif
                            PhysicalInstance inst = PhysicalInstance::NO_INST);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures);
    protected:
      mutable LocalLock                            leaf_lock;
      std::set<RtEvent>                            execution_events;
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority);
    };

    /**
     * \class InlineContext
     * A context for performing the inline execution
     * of a task inside of a parent task.
     */
    class InlineContext : public TaskContext {
    public:
      InlineContext(Runtime *runtime, TaskContext *enclosing, TaskOp *child);
      InlineContext(const InlineContext &rhs);
      virtual ~InlineContext(void);
    public:
      InlineContext& operator=(const InlineContext &rhs);
    public: // Privilege tracker methods
      virtual void receive_resources(size_t return_index,
              std::map<LogicalRegion,unsigned> &created_regions,
              std::vector<LogicalRegion> &deleted_regions,
              std::set<std::pair<FieldSpace,FieldID> > &created_fields,
              std::vector<std::pair<FieldSpace,FieldID> > &deleted_fields,
              std::map<FieldSpace,unsigned> &created_field_spaces,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_spaces,
              std::vector<FieldSpace> &deleted_field_spaces,
              std::map<IndexSpace,unsigned> &created_index_spaces,
              std::vector<std::pair<IndexSpace,bool> > &deleted_index_spaces,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<std::pair<IndexPartition,bool> > &deleted_partitions,
              std::set<RtEvent> &preconditions);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual UniqueID get_context_uid(void) const;
      virtual void pack_remote_context(Serializer &rez, 
          AddressSpaceID target, bool replicate);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
      virtual void handle_registration_callback_effects(RtEvent effects);
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(const Domain &domain, TypeTag tag);
      virtual IndexSpace create_index_space(const Future &future, TypeTag tag);
      virtual IndexSpace union_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace intersect_index_spaces(
                           const std::vector<IndexSpace> &spaces);
      virtual IndexSpace subtract_index_spaces(
                           IndexSpace left, IndexSpace right);
      virtual void create_shared_ownership(IndexSpace handle);
      virtual void destroy_index_space(IndexSpace handle, 
                                       const bool unordered,
                                       const bool recurse);
      virtual void create_shared_ownership(IndexPartition handle);
      virtual void destroy_index_partition(IndexPartition handle,
                                           const bool unordered,
                                           const bool recurse);
      virtual IndexPartition create_equal_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            size_t granularity,
                                            Color color);
      virtual IndexPartition create_partition_by_weights(IndexSpace parent,
                                            const FutureMap &weights,
                                            IndexSpace color_space,
                                            size_t granularity, 
                                            Color color);
      virtual IndexPartition create_partition_by_union(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual IndexPartition create_partition_by_intersection(
                                            IndexSpace parent,
                                            IndexPartition partition,
                                            PartitionKind kind,
                                            Color color,
                                            bool dominates);
      virtual IndexPartition create_partition_by_difference(
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            IndexSpace color_space,
                                            PartitionKind kind,
                                            Color color);
      virtual Color create_cross_product_partitions(
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<IndexSpace,IndexPartition> &handles,
                                            PartitionKind kind,
                                            Color color);
      virtual void create_association(      LogicalRegion domain,
                                            LogicalRegion domain_parent,
                                            FieldID domain_fid,
                                            IndexSpace range,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_restricted_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            const void *transform,
                                            size_t transform_size,
                                            const void *extent,
                                            size_t extent_size,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                  const std::map<DomainPoint,Domain> &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_domain(
                                            IndexSpace parent,
                                            const FutureMap &domains,
                                            IndexSpace color_space,
                                            bool perform_intersections,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexPartition create_partition_by_field(
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            Color color,
                                            MapperID id, MappingTagID tag,
                                            PartitionKind part_kind);
      virtual IndexPartition create_partition_by_image(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_image_range(
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_partition_by_preimage_range(
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color,
                                            MapperID id, MappingTagID tag);
      virtual IndexPartition create_pending_partition(
                                            IndexSpace parent,
                                            IndexSpace color_space,
                                            PartitionKind part_kind,
                                            Color color);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            IndexPartition parent,
                                            const void *realm_color,
                                            TypeTag type_tag,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual FieldSpace create_field_space(void);
      virtual FieldSpace create_field_space(const std::vector<size_t> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual void create_shared_ownership(FieldSpace handle);
      virtual FieldSpace create_field_space(const std::vector<Future> &sizes,
                                        std::vector<FieldID> &resulting_fields,
                                        CustomSerdezID serdez_id);
      virtual void destroy_field_space(FieldSpace handle, const bool unordered);
      virtual FieldID allocate_field(FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual FieldID allocate_field(FieldSpace space, const Future &field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void free_field(FieldAllocatorImpl *allocator, FieldSpace space, 
                              FieldID fid, const bool unordered);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void free_fields(FieldAllocatorImpl *allocator, FieldSpace space,
                               const std::set<FieldID> &to_free,
                               const bool unordered);
      virtual void allocate_local_field(FieldSpace space, size_t field_size,
                                     FieldID fid, CustomSerdezID serdez_id,
                                     std::set<RtEvent> &done_events);
      virtual void allocate_fields(FieldSpace space,
                                   const std::vector<Future> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void allocate_local_fields(FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   const std::vector<FieldID> &resuling_fields,
                                   CustomSerdezID serdez_id,
                                   std::set<RtEvent> &done_events);
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space,
                                            bool task_local);
      virtual void create_shared_ownership(LogicalRegion handle);
      virtual void destroy_logical_region(LogicalRegion handle,
                                          const bool unordered);
      virtual FieldAllocatorImpl* create_field_allocator(FieldSpace handle,
                                                         bool unordered);
      virtual void destroy_field_allocator(FieldSpace handle);
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::set<FieldID> &to_set) const;
      virtual void get_local_field_set(const FieldSpace handle,
                                       const std::set<unsigned> &indexes,
                                       std::vector<FieldID> &to_set) const;
    public:
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                      ReductionOpID redop, bool deterministic);
      virtual Future reduce_future_map(const FutureMap &future_map,
                                       ReductionOpID redop, bool deterministic);
      virtual FutureMap construct_future_map(const Domain &domain,
                    const std::map<DomainPoint,Future> &futures,
                    RtUserEvent domain_deletion = RtUserEvent::NO_RT_USER_EVENT,
                                             bool internal = false);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual ApEvent remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual Future detach_resource(PhysicalRegion region, const bool flush,
                                     const bool unordered);
      virtual void progress_unordered_operations(void);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual Future issue_mapping_fence(void);
      virtual Future issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // Calls for barriers and dynamic collectives
      virtual ApBarrier create_phase_barrier(unsigned arrivals,
                                                ReductionOpID redop = 0,
                                                const void *init_value = NULL,
                                                size_t init_size = 0);
      virtual void destroy_phase_barrier(ApBarrier bar);
      virtual PhaseBarrier advance_phase_barrier(PhaseBarrier bar);
      virtual void arrive_dynamic_collective(DynamicCollective dc,
                                             const void *buffer, 
                                             size_t size,
                                             unsigned count);
      virtual void defer_dynamic_collective_arrival(DynamicCollective dc,
                                                    const Future &f,
                                                    unsigned count);
      virtual Future get_dynamic_collective_result(DynamicCollective dc);
      virtual DynamicCollective advance_dynamic_collective(
                                                   DynamicCollective dc);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual size_t register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual void register_new_internal_operation(InternalOp *op);
      virtual size_t register_new_close_operation(CloseOp *op);
      virtual size_t register_new_summary_operation(TraceSummaryOp *op);
      virtual ApEvent add_to_dependence_queue(Operation *op, 
                                              bool unordered = false,
                                              bool outermost = true);
      virtual void add_to_post_task_queue(TaskContext *ctx, RtEvent wait_on,
                                          const void *result, size_t size, 
#ifdef LEGION_MALLOC_INSTANCES
                                          uintptr_t allocation = 0,
#endif
                                          PhysicalInstance instance =
                                            PhysicalInstance::NO_INST);
      virtual void register_executing_child(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual ApEvent register_implicit_dependences(Operation *op);
    public:
      virtual RtEvent get_current_mapping_fence_event(void);
      virtual ApEvent get_current_execution_fence_event(void);
      virtual ApEvent perform_fence_analysis(Operation *op,
                                             bool mapping, bool execution);
      virtual void update_current_fence(FenceOp *op,
                                        bool mapping, bool execution);
      virtual void update_current_implicit(Operation *op);
    public:
      virtual void begin_trace(TraceID tid, bool logical_only,
          bool static_trace, const std::set<RegionTreeID> *managed, bool dep);
      virtual void end_trace(TraceID tid, bool deprecated);
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
      virtual RtEvent decrement_pending(TaskOp *child);
      virtual RtEvent decrement_pending(bool need_deferral);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
#ifdef DEBUG_LEGION_COLLECTIVES
      virtual MergeCloseOp* get_merge_close_op(const LogicalUser &user,
                                               RegionTreeNode *node);
#else
      virtual MergeCloseOp* get_merge_close_op(void);
#endif
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index,
                                                  LogicalRegion parent);
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                                                     AddressSpaceID source);
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(
                                                    Legion::Runtime *&runtime);
      virtual void end_task(const void *res, size_t res_size, bool owned,
#ifdef LEGION_MALLOC_INSTANCES
                            uintptr_t allocation = 0,
#endif
                            PhysicalInstance inst = PhysicalInstance::NO_INST);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures);
    public:
      virtual TaskPriority get_current_priority(void) const;
      virtual void set_current_priority(TaskPriority priority);
    protected:
      TaskContext *const enclosing;
      TaskOp *const inline_task;
    protected:
      std::vector<unsigned> parent_req_indexes;
    };

    //--------------------------------------------------------------------------
    inline void TaskContext::begin_runtime_call(void)
    //--------------------------------------------------------------------------
    {
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

