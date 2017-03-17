/* Copyright 2017 Stanford University, NVIDIA Corporation
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
#include "legion_tasks.h"
#include "legion_mapping.h"
#include "legion_instances.h"
#include "legion_allocation.h"

namespace Legion {
  namespace Internal {
   
    /**
     * \class TaskContext
     * The base class for all task contexts which 
     * provide all the methods for handling the 
     * execution of a task at runtime.
     */
    class TaskContext : public ContextInterface, 
                        public ResourceTracker, public Collectable {
    public:
      struct ReclaimLocalFieldArgs : public LgTaskArgs<ReclaimLocalFieldArgs> {
      public:
        static const LgTaskID TASK_ID = LG_RECLAIM_LOCAL_FIELD_ID;
      public:
        FieldSpace handle;
        FieldID fid;
      };
      struct PostEndArgs : public LgTaskArgs<PostEndArgs> {
      public:
        static const LgTaskID TASK_ID = LG_POST_END_ID;
      public:
        TaskContext *proxy_this;
        void *result;
        size_t result_size;
      };
      class AutoRuntimeCall {
      public:
        AutoRuntimeCall(TaskContext *c) : ctx(c) { ctx->begin_runtime_call(); }
        ~AutoRuntimeCall(void) { ctx->end_runtime_call(); }
      public:
        TaskContext *const ctx;
      };
    public:
      TaskContext(Runtime *runtime, TaskOp *owner,
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
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const = 0;
      virtual ContextID get_context_id(void) const = 0;
      virtual UniqueID get_context_uid(void) const;
      virtual int get_depth(void) const;
      virtual Task* get_task(void); 
      virtual TaskContext* find_parent_context(void);
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target) = 0;
      virtual bool attempt_children_complete(void) = 0;
      virtual bool attempt_children_commit(void) = 0;
      virtual void inline_child_task(TaskOp *child) = 0;
      virtual VariantImpl* select_inline_variant(TaskOp *child) const = 0;
      virtual bool is_leaf_context(void) const;
      virtual bool is_inner_context(void) const;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            size_t max_num_elmts) = 0; 
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const Domain &domain) = 0;
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                           const std::set<Domain> &domains) = 0;
      virtual void destroy_index_space(IndexSpace handle) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint, int part_color) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind kind, int color) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                      PartitionKind part_kind, int color) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color) = 0;
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                              LegionRuntime::Accessor::RegionAccessor<
                LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                            int part_color) = 0;
      virtual void destroy_index_partition(IndexPartition handle) = 0;
      virtual IndexPartition create_equal_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            size_t granularity,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_weighted_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                              const std::map<DomainPoint,int> &weights,
                                            size_t granularity,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_union(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_intersection(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_difference(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable) = 0;
      virtual void create_cross_product_partition(
                                            RegionTreeForest *forest,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<DomainPoint,IndexPartition> &handles,
                                            PartitionKind kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_field(
                                            RegionTreeForest *forest,
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            const Domain &color_space,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_image(
                                            RegionTreeForest *forest,
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_partition_by_preimage(
                                            RegionTreeForest *forest,
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable) = 0;
      virtual IndexPartition create_pending_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable) = 0;
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle) = 0;
      virtual IndexSpace create_index_space_difference(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles) = 0;
      virtual FieldSpace create_field_space(RegionTreeForest *forest) = 0;
      virtual void destroy_field_space(FieldSpace handle) = 0;
      virtual FieldID allocate_field(RegionTreeForest *forest,
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id) = 0;
      virtual void free_field(FieldSpace space, FieldID fid) = 0;
      virtual void allocate_fields(RegionTreeForest *forest,
                                   FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id) = 0;
      virtual void free_fields(FieldSpace space, 
                               const std::set<FieldID> &to_free) = 0; 
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space) = 0;
      virtual void destroy_logical_region(LogicalRegion handle) = 0;
      virtual void destroy_logical_partition(LogicalPartition handle) = 0;
      virtual IndexAllocator create_index_allocator(RegionTreeForest *forest,
                                                    IndexSpace handle) = 0;
      virtual FieldAllocator create_field_allocator(Legion::Runtime *external,
                                                    FieldSpace handle) = 0;
    public:
      virtual Future execute_task(const TaskLauncher &launcher) = 0;
      virtual FutureMap execute_index_space(
                                         const IndexTaskLauncher &launcher) = 0;
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                         ReductionOpID redop) = 0; 
      virtual PhysicalRegion map_region(const InlineLauncher &launcher) = 0;
      virtual void remap_region(PhysicalRegion region) = 0;
      virtual void unmap_region(PhysicalRegion region) = 0;
      virtual void fill_fields(const FillLauncher &launcher) = 0;
      virtual void fill_fields(const IndexFillLauncher &launcher) = 0;
      virtual void issue_copy(const CopyLauncher &launcher) = 0;
      virtual void issue_copy(const IndexCopyLauncher &launcher) = 0;
      virtual void issue_acquire(const AcquireLauncher &launcher) = 0;
      virtual void issue_release(const ReleaseLauncher &launcher) = 0;
      virtual PhysicalRegion attach_resource(
                                  const AttachLauncher &launcher) = 0;
      virtual void detach_resource(PhysicalRegion region) = 0;
      virtual FutureMap execute_must_epoch(
                                 const MustEpochLauncher &launcher) = 0;
      virtual Future issue_timing_measurement(
                                    const TimingLauncher &launcher) = 0;
      virtual void issue_mapping_fence(void) = 0;
      virtual void issue_execution_fence(void) = 0;
      virtual void complete_frame(void) = 0;
      virtual Predicate create_predicate(const Future &f) = 0;
      virtual Predicate predicate_not(const Predicate &p) = 0;
      virtual Predicate create_predicate(const PredicateLauncher &launcher) = 0;
      virtual Future get_predicate_future(const Predicate &p) = 0;
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual unsigned register_new_child_operation(Operation *op,
               const std::vector<StaticDependence> *dependences) = 0;
      virtual unsigned register_new_close_operation(CloseOp *op) = 0;
      virtual void add_to_dependence_queue(Operation *op, bool has_lock,
                                           RtEvent op_precondition) = 0;
      virtual void register_child_executed(Operation *op) = 0;
      virtual void register_child_complete(Operation *op) = 0;
      virtual void register_child_commit(Operation *op) = 0; 
      virtual void unregister_child_operation(Operation *op) = 0;
      virtual void register_fence_dependence(Operation *op) = 0;
    public:
      virtual void perform_fence_analysis(FenceOp *op) = 0;
      virtual void update_current_fence(FenceOp *op) = 0;
    public:
      virtual void begin_trace(TraceID tid) = 0;
      virtual void end_trace(TraceID tid) = 0;
      virtual void begin_static_trace(
                                     const std::set<RegionTreeID> *managed) = 0;
      virtual void end_static_trace(void) = 0;
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination) = 0;
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination) = 0;
      virtual void finish_frame(ApEvent frame_termination) = 0;
    public:
      virtual void increment_outstanding(void) = 0;
      virtual void decrement_outstanding(void) = 0;
      virtual void increment_pending(void) = 0;
      virtual RtEvent decrement_pending(TaskOp *child) const = 0;
      virtual void decrement_pending(void) = 0;
      virtual void increment_frame(void) = 0;
      virtual void decrement_frame(void) = 0;
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index) = 0;
      virtual InnerContext* find_parent_physical_context(unsigned index) = 0;
      virtual void find_parent_version_info(unsigned index, unsigned depth, 
                  const FieldMask &version_mask, VersionInfo &version_info) = 0;
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL) = 0;
      virtual InnerContext* find_top_context(void) = 0;
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<ApEvent> &preconditions,
          std::set<RtEvent> &applied_events) = 0;
      virtual void invalidate_region_tree_contexts(void) = 0;
      virtual void send_back_created_state(AddressSpaceID target) = 0;
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL) = 0;
      static void handle_remote_view_creation(const void *args);
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(void);
      virtual void end_task(const void *res, size_t res_size, bool owned) = 0;
      virtual void post_end_task(const void *res, 
                                 size_t res_size, bool owned) = 0;
    public:
      virtual void add_acquisition(AcquireOp *op, 
                                   const RegionRequirement &req) = 0;
      virtual void remove_acquisition(ReleaseOp *op, 
                                      const RegionRequirement &req) = 0;
      virtual void add_restriction(AttachOp *op, InstanceManager *instance,
                                   const RegionRequirement &req) = 0;
      virtual void remove_restriction(DetachOp *op, 
                                      const RegionRequirement &req) = 0;
      virtual void release_restrictions(void) = 0;
      virtual bool has_restrictions(void) const = 0; 
      virtual void perform_restricted_analysis(const RegionRequirement &req, 
                                               RestrictInfo &restrict_info) = 0;
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f) = 0;
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures) = 0;
    public:
      PhysicalRegion get_physical_region(unsigned idx);
      void get_physical_references(unsigned idx, InstanceSet &refs);
    public:
      void add_created_region(LogicalRegion handle);
      // for logging created region requirements
      void log_created_requirements(void);
    public: // Privilege tracker methods
      virtual void register_region_creations(
                          const std::set<LogicalRegion> &regions);
      virtual void register_region_deletions(
                          const std::set<LogicalRegion> &regions);
    public:
      virtual void register_field_creations(
            const std::map<std::pair<FieldSpace,FieldID>,bool> &fields);
      virtual void register_field_deletions(
                const std::set<std::pair<FieldSpace,FieldID> > &fields);
    public:
      virtual void register_field_space_creations(
                          const std::set<FieldSpace> &spaces);
      virtual void register_field_space_deletions(
                          const std::set<FieldSpace> &spaces);
    public:
      virtual void register_index_space_creations(
                          const std::set<IndexSpace> &spaces);
      virtual void register_index_space_deletions(
                          const std::set<IndexSpace> &spaces);
    public:
      virtual void register_index_partition_creations(
                          const std::set<IndexPartition> &parts);
      virtual void register_index_partition_deletions(
                          const std::set<IndexPartition> &parts);
    public:
      void register_region_creation(LogicalRegion handle);
      void register_region_deletion(LogicalRegion handle);
    public:
      void register_field_creation(FieldSpace space, FieldID fid, bool local);
      void register_field_creations(FieldSpace space, bool local,
                                    const std::vector<FieldID> &fields);
      void register_field_deletions(FieldSpace space,
                                    const std::set<FieldID> &to_free);
    public:
      void register_field_space_creation(FieldSpace space);
      void register_field_space_deletion(FieldSpace space);
    public:
      bool has_created_index_space(IndexSpace space) const;
      void register_index_space_creation(IndexSpace space);
      void register_index_space_deletion(IndexSpace space);
    public:
      void register_index_partition_creation(IndexPartition handle);
      void register_index_partition_deletion(IndexPartition handle);
    public:
      bool was_created_requirement_deleted(const RegionRequirement &req) const;
    public:
      void destroy_user_lock(Reservation r);
      void destroy_user_barrier(ApBarrier b);
    public:
      ptr_t perform_safe_cast(IndexSpace is, ptr_t pointer);
      DomainPoint perform_safe_cast(IndexSpace is, const DomainPoint &point);
    public:
      void analyze_destroy_index_space(IndexSpace handle, 
                                   std::vector<RegionRequirement> &delete_reqs,
                                   std::vector<unsigned> &parent_req_indexes);
      void analyze_destroy_index_partition(IndexPartition handle, 
                                   std::vector<RegionRequirement> &delete_reqs,
                                   std::vector<unsigned> &parent_req_indexes);
      void analyze_destroy_field_space(FieldSpace handle, 
                                   std::vector<RegionRequirement> &delete_reqs,
                                   std::vector<unsigned> &parent_req_indexes);
      void analyze_destroy_fields(FieldSpace handle,
                                  const std::set<FieldID> &to_delete,
                                  std::vector<RegionRequirement> &delete_reqs,
                                  std::vector<unsigned> &parent_req_indexes);
      void analyze_destroy_logical_region(LogicalRegion handle, 
                                  std::vector<RegionRequirement> &delete_reqs,
                                  std::vector<unsigned> &parent_req_indexes);
      void analyze_destroy_logical_partition(LogicalPartition handle,
                                  std::vector<RegionRequirement> &delete_reqs,
                                  std::vector<unsigned> &parent_req_indexes);
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
                                  const RegionRequirement &req);
      void register_inline_mapped_region(PhysicalRegion &region);
      void unregister_inline_mapped_region(PhysicalRegion &region);
    public:
      bool is_region_mapped(unsigned idx);
      void clone_requirement(unsigned idx, RegionRequirement &target);
      int find_parent_region_req(const RegionRequirement &req, 
                                 bool check_privilege = true);
      unsigned find_parent_region(unsigned idx, TaskOp *task);
      unsigned find_parent_index_region(unsigned idx, TaskOp *task);
      PrivilegeMode find_parent_privilege_mode(unsigned idx);
      LegionErrorType check_privilege(const IndexSpaceRequirement &req) const;
      LegionErrorType check_privilege(const RegionRequirement &req, 
                                      FieldID &bad_field, 
                                      bool skip_privileges = false) const; 
      LogicalRegion find_logical_region(unsigned index);
    protected:
      LegionErrorType check_privilege_internal(const RegionRequirement &req,
                                      const RegionRequirement &parent_req,
                                      std::set<FieldID>& privilege_fields,
                                      FieldID &bad_field, 
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
      void execute_task_launch(TaskOp *task, bool index, 
                             LegionTrace *current_trace, bool silence_warnings);
      void remap_unmapped_regions(LegionTrace *current_trace,
                           const std::vector<PhysicalRegion> &unmapped_regions);
    public:
      void perform_inlining(TaskContext *ctx, VariantImpl *variant); 
    public:
      Runtime *const runtime;
      TaskOp *const owner_task;
      const std::vector<RegionRequirement> &regions;
    protected:
      friend class SingleTask;
      Reservation context_lock;
    protected:
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts
      LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::tracked
                                                   inline_regions; 
      // Application tasks can manipulate these next two data
      // structures by creating regions and fields, make sure you are
      // holding the operation lock when you are accessing them
      std::deque<RegionRequirement>             created_requirements;
      // Track whether the created region requirements have
      // privileges to be returned or not
      std::vector<bool>                         returnable_privileges;
    protected:
      std::vector<PhysicalRegion>               physical_regions;
    protected: // Instance top view data structures
      std::map<PhysicalManager*,InstanceView*>  instance_top_views;
      std::map<PhysicalManager*,RtUserEvent>    pending_top_views;
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
      // Some help for performing fast safe casts
      std::map<IndexSpace,Domain> safe_cast_domains;   
    protected:
      RtEvent pending_done;
      bool task_executed;
    protected: 
      bool children_complete_invoked;
      bool children_commit_invoked;
#ifdef LEGION_SPY
    public:
      RtEvent update_previous_mapped_event(RtEvent next);
    protected:
      UniqueID current_fence_uid;
      RtEvent previous_mapped_event;
#endif
    };

    class InnerContext : public TaskContext {
    public:
      struct DeferredDependenceArgs : 
        public LgTaskArgs<DeferredDependenceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_DEPENDENCE_ID;
      public:
        Operation *op;
      }; 
      struct DecrementArgs : public LgTaskArgs<DecrementArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DECREMENT_PENDING_TASK_ID;
      public:
        InnerContext *parent_ctx;
      };
      struct WindowWaitArgs : public LgTaskArgs<WindowWaitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_WINDOW_WAIT_TASK_ID;
      public:
        InnerContext *parent_ctx;
      };
      struct IssueFrameArgs : public LgTaskArgs<IssueFrameArgs> {
      public:
        static const LgTaskID TASK_ID = LG_ISSUE_FRAME_TASK_ID;
      public:
        InnerContext *parent_ctx;
        FrameOp *frame;
        ApEvent frame_termination;
      };
      struct AddToDepQueueArgs : public LgTaskArgs<AddToDepQueueArgs> {
      public:
        static const LgTaskID TASK_ID = LG_ADD_TO_DEP_QUEUE_TASK_ID;
      public:
        InnerContext *proxy_this;
        Operation *op;
        RtEvent op_pre;
      };
      struct RemoteCreateViewArgs : public LgTaskArgs<RemoteCreateViewArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REMOTE_VIEW_CREATION_TASK_ID;
      public:
        InnerContext *proxy_this;
        PhysicalManager *manager;
        InstanceView **target;
        RtUserEvent to_trigger;
        AddressSpaceID source;
      };
    public:
      InnerContext(Runtime *runtime, TaskOp *owner, bool full_inner,
                   const std::vector<RegionRequirement> &reqs,
                   const std::vector<unsigned> &parent_indexes,
                   const std::vector<bool> &virt_mapped,
                   UniqueID context_uid, bool remote = false);
      InnerContext(const InnerContext &rhs);
      virtual ~InnerContext(void);
    public:
      InnerContext& operator=(const InnerContext &rhs);
    public:
      void print_children(void);
      void perform_window_wait(void);
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual UniqueID get_context_uid(void) const;
      virtual int get_depth(void) const;
      virtual bool is_inner_context(void) const;
      virtual void pack_remote_context(Serializer &rez, AddressSpaceID target);
      virtual void unpack_remote_context(Deserializer &derez,
                                         std::set<RtEvent> &preconditions);
      virtual AddressSpaceID get_version_owner(RegionTreeNode *node,
                                               AddressSpaceID source);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            size_t max_num_elmts);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const Domain &domain);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const std::set<Domain> &domains);
      virtual void destroy_index_space(IndexSpace handle);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                            PartitionKind part_kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                              LegionRuntime::Accessor::RegionAccessor<
                LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                            int part_color);
      virtual void destroy_index_partition(IndexPartition handle);
      virtual IndexPartition create_equal_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_weighted_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                              const std::map<DomainPoint,int> &weights,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_union(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_intersection(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_difference(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual void create_cross_product_partition(
                                            RegionTreeForest *forest,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<DomainPoint,IndexPartition> &handles,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_field(
                                            RegionTreeForest *forest,
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            const Domain &color_space,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_image(
                                            RegionTreeForest *forest,
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_preimage(
                                            RegionTreeForest *forest,
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_pending_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual FieldSpace create_field_space(RegionTreeForest *forest);
      virtual void destroy_field_space(FieldSpace handle);
      virtual FieldID allocate_field(RegionTreeForest *forest,
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void free_field(FieldSpace space, FieldID fid);
      virtual void allocate_fields(RegionTreeForest *forest,
                                   FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void free_fields(FieldSpace space, 
                               const std::set<FieldID> &to_free);
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space);
      virtual void destroy_logical_region(LogicalRegion handle);
      virtual void destroy_logical_partition(LogicalPartition handle);
      virtual IndexAllocator create_index_allocator(RegionTreeForest *forest,
                                                    IndexSpace handle);
      virtual FieldAllocator create_field_allocator(Legion::Runtime *external,
                                                    FieldSpace handle);
    public:
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                         ReductionOpID redop);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual void remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual void detach_resource(PhysicalRegion region);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual void issue_mapping_fence(void);
      virtual void issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual unsigned register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual unsigned register_new_close_operation(CloseOp *op);
      virtual void add_to_dependence_queue(Operation *op, bool has_lock,
                                           RtEvent op_precondition);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      virtual void perform_fence_analysis(FenceOp *op);
      virtual void update_current_fence(FenceOp *op);
    public:
      virtual void begin_trace(TraceID tid);
      virtual void end_trace(TraceID tid);
      virtual void begin_static_trace(const std::set<RegionTreeID> *managed);
      virtual void end_static_trace(void);
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination);
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination);
      virtual void finish_frame(ApEvent frame_termination);
    public:
      virtual void increment_outstanding(void);
      virtual void decrement_outstanding(void);
      virtual void increment_pending(void);
      virtual RtEvent decrement_pending(TaskOp *child) const;
      virtual void decrement_pending(void);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index);
      virtual void find_parent_version_info(unsigned index, unsigned depth, 
                  const FieldMask &version_mask, VersionInfo &version_info);
    public:
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      void configure_context(MapperManager *mapper);
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<ApEvent> &preconditions,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL);
      static void handle_remote_view_creation(const void *args);
      void notify_instance_deletion(PhysicalManager *deleted); 
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    public:
      virtual void end_task(const void *res, size_t res_size, bool owned);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void add_acquisition(AcquireOp *op, 
                                   const RegionRequirement &req);
      virtual void remove_acquisition(ReleaseOp *op, 
                                      const RegionRequirement &req);
      virtual void add_restriction(AttachOp *op, InstanceManager *instance,
                                   const RegionRequirement &req);
      virtual void remove_restriction(DetachOp *op, 
                                      const RegionRequirement &req);
      virtual void release_restrictions(void);
      virtual bool has_restrictions(void) const; 
      virtual void perform_restricted_analysis(const RegionRequirement &req, 
                                               RestrictInfo &restrict_info);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                       std::vector<Future> &contributions);
    public:
      static void handle_version_owner_request(Deserializer &derez,
                            Runtime *runtime, AddressSpaceID source);
      void process_version_owner_response(RegionTreeNode *node, 
                                          AddressSpaceID result);
      static void handle_version_owner_response(Deserializer &derez,
                                                Runtime *runtime);
    public:
      void invalidate_remote_contexts(void);
      void send_remote_context(AddressSpaceID remote_instance, 
                               RemoteContext *target);
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
      // Track whether this task has finished executing
      unsigned total_children_count; // total number of sub-operations
      unsigned total_close_count; 
      unsigned outstanding_children_count;
      LegionSet<Operation*,EXECUTING_CHILD_ALLOC>::tracked executing_children;
      LegionSet<Operation*,EXECUTED_CHILD_ALLOC>::tracked executed_children;
      LegionSet<Operation*,COMPLETE_CHILD_ALLOC>::tracked complete_children; 
#ifdef DEBUG_LEGION
      // In debug mode also keep track of them in context order so
      // we can see what the longest outstanding operation is which
      // is often useful when things hang
      std::map<unsigned,Operation*> outstanding_children;
#endif
    protected:
      // Traces for this task's execution
      LegionMap<TraceID,DynamicTrace*,TASK_TRACES_ALLOC>::tracked traces;
      LegionTrace *current_trace;
      // Event for waiting when the number of mapping+executing
      // child operations has grown too large.
      bool valid_wait_event;
      RtUserEvent window_wait;
      std::deque<ApEvent> frame_events;
      RtEvent last_registration;
      RtEvent dependence_precondition;
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
      FenceOp *current_fence;
      GenerationID fence_gen;
    protected:
      // For tracking restricted coherence
      std::list<Restriction*> coherence_restrictions;
    protected:
      std::map<RegionTreeNode*,
        std::pair<AddressSpaceID,bool/*remote only*/> > region_tree_owners;
    protected:
      std::map<RegionTreeNode*,RtUserEvent> pending_version_owner_requests;
    protected:
      std::map<AddressSpaceID,RemoteContext*> remote_instances;
    protected:
      // Tracking information for dynamic collectives
      std::map<unsigned long/*ID*/,std::map<unsigned/*gen*/,
               std::vector<Future> > > collective_contributions;
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
      virtual int get_depth(void) const;
      virtual void pack_remote_context(Serializer &rez, AddressSpaceID target);
      virtual TaskContext* find_parent_context(void);
    public:
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual AddressSpaceID get_version_owner(RegionTreeNode *node,
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
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const; 
      virtual void set_context_index(unsigned index);
      virtual int get_depth(void) const;
      virtual const char* get_task_name(void) const;
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
      RemoteContext(Runtime *runtime, UniqueID context_uid);
      RemoteContext(const RemoteContext &rhs);
      virtual ~RemoteContext(void);
    public:
      RemoteContext& operator=(const RemoteContext &rhs);
    public:
      virtual int get_depth(void) const;
      virtual Task* get_task(void);
      virtual void unpack_remote_context(Deserializer &derez,
                                         std::set<RtEvent> &preconditions);
      virtual TaskContext* find_parent_context(void);
    public:
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual AddressSpaceID get_version_owner(RegionTreeNode *node,
                                               AddressSpaceID source);
      virtual void find_parent_version_info(unsigned index, unsigned depth, 
                  const FieldMask &version_mask, VersionInfo &version_info);
    protected:
      UniqueID parent_context_uid;
      TaskContext *parent_ctx;
    protected:
      int depth;
      ApEvent remote_completion_event;
      std::vector<VersionInfo> version_infos;
      bool top_level_context;
      RemoteTask remote_task;
    protected:
      std::vector<unsigned> local_parent_req_indexes;
      std::vector<bool> local_virtual_mapped;
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
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
      virtual bool is_leaf_context(void) const;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            size_t max_num_elmts);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const Domain &domain);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const std::set<Domain> &domains);
      virtual void destroy_index_space(IndexSpace handle);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                            PartitionKind part_kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                              LegionRuntime::Accessor::RegionAccessor<
                LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                            int part_color);
      virtual void destroy_index_partition(IndexPartition handle);
      virtual IndexPartition create_equal_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_weighted_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                              const std::map<DomainPoint,int> &weights,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_union(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_intersection(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_difference(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual void create_cross_product_partition(
                                            RegionTreeForest *forest,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<DomainPoint,IndexPartition> &handles,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_field(
                                            RegionTreeForest *forest,
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            const Domain &color_space,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_image(
                                            RegionTreeForest *forest,
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_preimage(
                                            RegionTreeForest *forest,
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_pending_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual FieldSpace create_field_space(RegionTreeForest *forest);
      virtual void destroy_field_space(FieldSpace handle);
      virtual FieldID allocate_field(RegionTreeForest *forest,
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void free_field(FieldSpace space, FieldID fid);
      virtual void allocate_fields(RegionTreeForest *forest,
                                   FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void free_fields(FieldSpace space, 
                               const std::set<FieldID> &to_free);
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space);
      virtual void destroy_logical_region(LogicalRegion handle);
      virtual void destroy_logical_partition(LogicalPartition handle);
      virtual IndexAllocator create_index_allocator(RegionTreeForest *forest,
                                                    IndexSpace handle);
      virtual FieldAllocator create_field_allocator(Legion::Runtime *external,
                                                    FieldSpace handle);
    public:
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                         ReductionOpID redop);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual void remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual void detach_resource(PhysicalRegion region);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual void issue_mapping_fence(void);
      virtual void issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual unsigned register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual unsigned register_new_close_operation(CloseOp *op);
      virtual void add_to_dependence_queue(Operation *op, bool has_lock,
                                           RtEvent op_precondition);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      virtual void perform_fence_analysis(FenceOp *op);
      virtual void update_current_fence(FenceOp *op);
    public:
      virtual void begin_trace(TraceID tid);
      virtual void end_trace(TraceID tid);
      virtual void begin_static_trace(const std::set<RegionTreeID> *managed);
      virtual void end_static_trace(void);
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination);
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination);
      virtual void finish_frame(ApEvent frame_termination);
    public:
      virtual void increment_outstanding(void);
      virtual void decrement_outstanding(void);
      virtual void increment_pending(void);
      virtual RtEvent decrement_pending(TaskOp *child) const;
      virtual void decrement_pending(void);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index);
      virtual void find_parent_version_info(unsigned index, unsigned depth, 
                  const FieldMask &version_mask, VersionInfo &version_info);
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<ApEvent> &preconditions,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL);
      static void handle_remote_view_creation(const void *args);
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    public:
      virtual void end_task(const void *res, size_t res_size, bool owned);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void add_acquisition(AcquireOp *op, 
                                   const RegionRequirement &req);
      virtual void remove_acquisition(ReleaseOp *op, 
                                      const RegionRequirement &req);
      virtual void add_restriction(AttachOp *op, InstanceManager *instance,
                                   const RegionRequirement &req);
      virtual void remove_restriction(DetachOp *op, 
                                      const RegionRequirement &req);
      virtual void release_restrictions(void);
      virtual bool has_restrictions(void) const; 
      virtual void perform_restricted_analysis(const RegionRequirement &req, 
                                               RestrictInfo &restrict_info);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures);
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
    public:
      // Interface for task contexts
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
      virtual UniqueID get_context_uid(void) const;
      virtual int get_depth(void) const;
      virtual void pack_remote_context(Serializer &rez, 
                                       AddressSpaceID target);
      virtual bool attempt_children_complete(void);
      virtual bool attempt_children_commit(void);
      virtual void inline_child_task(TaskOp *child);
      virtual VariantImpl* select_inline_variant(TaskOp *child) const;
    public:
      // Interface to operations performed by a context
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            size_t max_num_elmts);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const Domain &domain);
      virtual IndexSpace create_index_space(RegionTreeForest *forest,
                                            const std::set<Domain> &domains);
      virtual void destroy_index_space(IndexSpace handle);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent, 
                                            const Domain &color_space,
                                            const PointColoring &coloring,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Coloring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainPointColoring &coloring,
                                            PartitionKind kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const DomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                      const MultiDomainPointColoring &coloring,
                                            PartitionKind part_kind, int color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            const MultiDomainColoring &coloring,
                                            bool disjoint, int part_color);
      virtual IndexPartition create_index_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                              LegionRuntime::Accessor::RegionAccessor<
                LegionRuntime::Accessor::AccessorType::Generic> field_accessor,
                                            int part_color);
      virtual void destroy_index_partition(IndexPartition handle);
      virtual IndexPartition create_equal_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_weighted_partition(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                              const std::map<DomainPoint,int> &weights,
                                            size_t granularity,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_union(RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_intersection(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_difference(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual void create_cross_product_partition(
                                            RegionTreeForest *forest,
                                            IndexPartition handle1,
                                            IndexPartition handle2,
                              std::map<DomainPoint,IndexPartition> &handles,
                                            PartitionKind kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_field(
                                            RegionTreeForest *forest,
                                            LogicalRegion handle,
                                            LogicalRegion parent_priv,
                                            FieldID fid,
                                            const Domain &color_space,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_image(
                                            RegionTreeForest *forest,
                                            IndexSpace handle,
                                            LogicalPartition projection,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_partition_by_preimage(
                                            RegionTreeForest *forest,
                                            IndexPartition projection,
                                            LogicalRegion handle,
                                            LogicalRegion parent,
                                            FieldID fid,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexPartition create_pending_partition(
                                            RegionTreeForest *forest,
                                            IndexSpace parent,
                                            const Domain &color_space,
                                            PartitionKind part_kind,
                                            int color, bool allocable);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_union(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                const std::vector<IndexSpace> &handles);
      virtual IndexSpace create_index_space_intersection(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexPartition handle);
      virtual IndexSpace create_index_space_difference(
                                            RegionTreeForest *forest,
                                            IndexPartition parent,
                                            const DomainPoint &color,
                                            IndexSpace initial,
                                const std::vector<IndexSpace> &handles);
      virtual FieldSpace create_field_space(RegionTreeForest *forest);
      virtual void destroy_field_space(FieldSpace handle);
      virtual FieldID allocate_field(RegionTreeForest *forest,
                                     FieldSpace space, size_t field_size,
                                     FieldID fid, bool local,
                                     CustomSerdezID serdez_id);
      virtual void free_field(FieldSpace space, FieldID fid);
      virtual void allocate_fields(RegionTreeForest *forest,
                                   FieldSpace space,
                                   const std::vector<size_t> &sizes,
                                   std::vector<FieldID> &resuling_fields,
                                   bool local, CustomSerdezID serdez_id);
      virtual void free_fields(FieldSpace space, 
                               const std::set<FieldID> &to_free);
      virtual LogicalRegion create_logical_region(RegionTreeForest *forest,
                                            IndexSpace index_space,
                                            FieldSpace field_space);
      virtual void destroy_logical_region(LogicalRegion handle);
      virtual void destroy_logical_partition(LogicalPartition handle);
      virtual IndexAllocator create_index_allocator(RegionTreeForest *forest,
                                                    IndexSpace handle);
      virtual FieldAllocator create_field_allocator(Legion::Runtime *external,
                                                    FieldSpace handle);
    public:
      virtual Future execute_task(const TaskLauncher &launcher);
      virtual FutureMap execute_index_space(const IndexTaskLauncher &launcher);
      virtual Future execute_index_space(const IndexTaskLauncher &launcher,
                                         ReductionOpID redop);
      virtual PhysicalRegion map_region(const InlineLauncher &launcher);
      virtual void remap_region(PhysicalRegion region);
      virtual void unmap_region(PhysicalRegion region);
      virtual void fill_fields(const FillLauncher &launcher);
      virtual void fill_fields(const IndexFillLauncher &launcher);
      virtual void issue_copy(const CopyLauncher &launcher);
      virtual void issue_copy(const IndexCopyLauncher &launcher);
      virtual void issue_acquire(const AcquireLauncher &launcher);
      virtual void issue_release(const ReleaseLauncher &launcher);
      virtual PhysicalRegion attach_resource(const AttachLauncher &launcher);
      virtual void detach_resource(PhysicalRegion region);
      virtual FutureMap execute_must_epoch(const MustEpochLauncher &launcher);
      virtual Future issue_timing_measurement(const TimingLauncher &launcher);
      virtual void issue_mapping_fence(void);
      virtual void issue_execution_fence(void);
      virtual void complete_frame(void);
      virtual Predicate create_predicate(const Future &f);
      virtual Predicate predicate_not(const Predicate &p);
      virtual Predicate create_predicate(const PredicateLauncher &launcher);
      virtual Future get_predicate_future(const Predicate &p);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual unsigned register_new_child_operation(Operation *op,
                const std::vector<StaticDependence> *dependences);
      virtual unsigned register_new_close_operation(CloseOp *op);
      virtual void add_to_dependence_queue(Operation *op, bool has_lock,
                                           RtEvent op_precondition);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      virtual void perform_fence_analysis(FenceOp *op);
      virtual void update_current_fence(FenceOp *op);
    public:
      virtual void begin_trace(TraceID tid);
      virtual void end_trace(TraceID tid);
      virtual void begin_static_trace(const std::set<RegionTreeID> *managed);
      virtual void end_static_trace(void);
    public:
      virtual void issue_frame(FrameOp *frame, ApEvent frame_termination);
      virtual void perform_frame_issue(FrameOp *frame, 
                                       ApEvent frame_termination);
      virtual void finish_frame(ApEvent frame_termination);
    public:
      virtual void increment_outstanding(void);
      virtual void decrement_outstanding(void);
      virtual void increment_pending(void);
      virtual RtEvent decrement_pending(TaskOp *child) const;
      virtual void decrement_pending(void);
      virtual void increment_frame(void);
      virtual void decrement_frame(void);
    public:
      virtual InnerContext* find_parent_logical_context(unsigned index);
      virtual InnerContext* find_parent_physical_context(unsigned index);
      virtual void find_parent_version_info(unsigned index, unsigned depth, 
                  const FieldMask &version_mask, VersionInfo &version_info);
      // Override by RemoteTask and TopLevelTask
      virtual InnerContext* find_outermost_local_context(
                          InnerContext *previous = NULL);
      virtual InnerContext* find_top_context(void);
    public:
      virtual void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<ApEvent> &preconditions,
          std::set<RtEvent> &applied_events);
      virtual void invalidate_region_tree_contexts(void);
      virtual void send_back_created_state(AddressSpaceID target);
    public:
      virtual InstanceView* create_instance_top_view(PhysicalManager *manager,
                             AddressSpaceID source, RtEvent *ready = NULL);
      static void handle_remote_view_creation(const void *args);
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    public:
      virtual const std::vector<PhysicalRegion>& begin_task(void);
      virtual void end_task(const void *res, size_t res_size, bool owned);
      virtual void post_end_task(const void *res, size_t res_size, bool owned);
    public:
      virtual void add_acquisition(AcquireOp *op, 
                                   const RegionRequirement &req);
      virtual void remove_acquisition(ReleaseOp *op, 
                                      const RegionRequirement &req);
      virtual void add_restriction(AttachOp *op, InstanceManager *instance,
                                   const RegionRequirement &req);
      virtual void remove_restriction(DetachOp *op, 
                                      const RegionRequirement &req);
      virtual void release_restrictions(void);
      virtual bool has_restrictions(void) const; 
      virtual void perform_restricted_analysis(const RegionRequirement &req, 
                                               RestrictInfo &restrict_info);
    public:
      virtual void record_dynamic_collective_contribution(DynamicCollective dc,
                                                          const Future &f);
      virtual void find_collective_contributions(DynamicCollective dc,
                                             std::vector<Future> &futures);
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

