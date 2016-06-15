/* Copyright 2016 Stanford University, NVIDIA Corporation
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


#ifndef __LEGION_TASKS_H__
#define __LEGION_TASKS_H__

#include "legion.h"
#include "runtime.h"
#include "legion_ops.h"
#include "region_tree.h"
#include "legion_mapping.h"
#include "legion_utilities.h"
#include "legion_allocation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TaskOp
     * This is the base task operation class for all
     * kinds of tasks in the system.  
     */
    class TaskOp : public Task, public SpeculativeOp {
    public:
      enum TaskKind {
        INDIVIDUAL_TASK_KIND,
        POINT_TASK_KIND,
        REMOTE_TASK_KIND,
        INDEX_TASK_KIND,
        SLICE_TASK_KIND,
      };
    public:
      TaskOp(Runtime *rt);
      virtual ~TaskOp(void);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual int get_depth(void) const;
      virtual const char* get_task_name(void) const;
    public:
      bool is_remote(void) const;
      inline bool is_stolen(void) const { return (steal_count > 0); }
      inline bool is_locally_mapped(void) const { return map_locally; }
    public:
      void set_current_proc(Processor current);
      inline void set_locally_mapped(bool local) { map_locally = local; }
      inline void set_target_proc(Processor next) { target_proc = next; }
    protected:
      void activate_task(void);
      void deactivate_task(void);
    public:
      void set_must_epoch(MustEpochOp *epoch, unsigned index, 
                          bool do_registration);
    protected:
      void pack_base_task(Serializer &rez, AddressSpaceID target);
      void unpack_base_task(Deserializer &derez, 
                            std::set<RtEvent> &ready_events);
      void pack_base_external_task(Serializer &rez, AddressSpaceID target);
      void unpack_base_external_task(Deserializer &derez); 
    public:
      void mark_stolen(void);
      void initialize_base_task(SingleTask *ctx, bool track, 
                                const Predicate &p, 
                                Processor::TaskFuncID tid);
      void check_empty_field_requirements(void);
      size_t check_future_size(FutureImpl *impl);
    public:
      bool select_task_options(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void) = 0;
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void) = 0;
      virtual bool speculate(bool &value);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      virtual unsigned find_parent_index(unsigned idx);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
      virtual void recapture_version_info(unsigned idx);
    public:
      virtual bool early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(MustEpochOp *owner = NULL) = 0;
      virtual void launch_task(void) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle) = 0;
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      // Returns true if the task should be deactivated
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant) = 0;
    public:
      virtual bool is_inline_task(void) const;
      virtual const std::vector<PhysicalRegion>& begin_inline_task(void);
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
    public:
      RegionTreeContext get_parent_context(unsigned idx);
    protected:
      void pack_version_infos(Serializer &rez,
                              std::vector<VersionInfo> &infos,
                              const std::vector<bool> &full_version_info);
      void unpack_version_infos(Deserializer &derez,
                                std::vector<VersionInfo> &infos);
    protected:
      void pack_restrict_infos(Serializer &rez, 
                               std::vector<RestrictInfo> &infos);
      void unpack_restrict_infos(Deserializer &derez,
                                 std::vector<RestrictInfo> &infos);
    public:
      // Tell the parent context that this task is in a ready queue
      void activate_outstanding_task(void);
      void deactivate_outstanding_task(void);
    public:
      void register_region_creation(LogicalRegion handle);
      void register_region_deletion(LogicalRegion handle);
      void register_region_creations(const std::set<LogicalRegion> &regions);
      void register_region_deletions(const std::set<LogicalRegion> &regions);
    public:
      void register_field_creation(FieldSpace space, FieldID fid);
      void register_field_creations(FieldSpace space, 
                                    const std::vector<FieldID> &fields);
      void register_field_deletions(FieldSpace space,
                                    const std::set<FieldID> &to_free);
      void register_field_creations(
                const std::set<std::pair<FieldSpace,FieldID> > &fields);
      void register_field_deletions(
                const std::set<std::pair<FieldSpace,FieldID> > &fields);
    public:
      void register_field_space_creation(FieldSpace space);
      void register_field_space_deletion(FieldSpace space);
      void register_field_space_creations(const std::set<FieldSpace> &spaces);
      void register_field_space_deletions(const std::set<FieldSpace> &spaces);
    public:
      bool has_created_index_space(IndexSpace space) const;
      void register_index_space_creation(IndexSpace space);
      void register_index_space_deletion(IndexSpace space);
      void register_index_space_creations(const std::set<IndexSpace> &spaces);
      void register_index_space_deletions(const std::set<IndexSpace> &spaces);
    public:
      void register_index_partition_creation(IndexPartition handle);
      void register_index_partition_deletion(IndexPartition handle);
      void register_index_partition_creations(
                                        const std::set<IndexPartition> &parts);
      void register_index_partition_deletions(
                                        const std::set<IndexPartition> &parts);
    public:
      virtual void add_created_region(LogicalRegion handle) = 0;
      virtual void add_created_field(FieldSpace handle, FieldID fid) = 0;
    public:
      void return_privilege_state(TaskOp *target);
      void pack_privilege_state(Serializer &rez, AddressSpaceID target);
      void unpack_privilege_state(Deserializer &derez);
      void perform_privilege_checks(void);
    public:
      void find_early_mapped_region(unsigned idx, InstanceSet &ref);
      void clone_task_op_from(TaskOp *rhs, Processor p, 
                              bool stealable, bool duplicate_args);
      void update_grants(const std::vector<Grant> &grants);
      void update_arrival_barriers(const std::vector<PhaseBarrier> &barriers);
      bool compute_point_region_requirements(MinimalPoint *mp = NULL);
      bool early_map_regions(std::set<RtEvent> &applied_conditions,
                             const std::vector<unsigned> &must_premap);
      bool prepare_steal(void);
    public:
      void compute_parent_indexes(void);
      void record_aliased_region_requirements(LegionTrace *trace);
    protected:
      // These methods get called once the task has executed
      // and all the children have either mapped, completed,
      // or committed.
      void trigger_children_complete(void);
      void trigger_children_committed(void);
    protected:
      // Tasks have two requirements to complete:
      // - all speculation must be resolved
      // - all children must be complete
      virtual void trigger_task_complete(void) = 0;
      // Tasks have two requirements to commit:
      // - all commit dependences must be satisfied (trigger_commit)
      // - all children must commit (children_committed)
      virtual void trigger_task_commit(void) = 0;
    protected:
      // Early mapped regions
      std::map<unsigned/*idx*/,InstanceSet>     early_mapped_regions;
      // A map of any locks that we need to take for this task
      std::map<Reservation,bool/*exclusive*/>   atomic_locks;
    protected:
      std::vector<unsigned>                     parent_req_indexes;
    protected:
      std::set<LogicalRegion>                   created_regions;
      std::set<std::pair<FieldSpace,FieldID> >  created_fields;
      std::set<FieldSpace>                      created_field_spaces;
      std::set<IndexSpace>                      created_index_spaces;
      std::set<IndexPartition>                  created_index_partitions;
      std::set<LogicalRegion>                   deleted_regions;
      std::set<std::pair<FieldSpace,FieldID> >  deleted_fields;
      std::set<FieldSpace>                      deleted_field_spaces;
      std::set<IndexSpace>                      deleted_index_spaces;
      std::set<IndexPartition>                  deleted_index_partitions;
    protected:
      bool complete_received;
      bool commit_received;
      bool children_complete;
      bool children_commit;
      bool children_complete_invoked;
      bool children_commit_invoked;
    protected:
      bool map_locally;
    private:
      mutable bool is_local;
      mutable bool local_cached;
    protected:
      AllocManager *arg_manager;
    protected:
      MapperManager *mapper;
    private:
      unsigned current_mapping_index;
    public:
      inline void set_current_mapping_index(unsigned index) 
        { current_mapping_index = index; }
    public:
      // Index for this must epoch op
      unsigned must_epoch_index;
    public:
      // Static methods
      static void process_unpack_task(Runtime *rt,
                                      Deserializer &derez);
    public:
      static void pack_index_space_requirement(
          const IndexSpaceRequirement &req, Serializer &rez);
      static void pack_region_requirement(
          const RegionRequirement &req, Serializer &rez);
      static void pack_grant(
          const Grant &grant, Serializer &rez);
      static void pack_phase_barrier(
          const PhaseBarrier &barrier, Serializer &rez);
    public:
      static void unpack_index_space_requirement(
          IndexSpaceRequirement &req, Deserializer &derez);
      static void unpack_region_requirement(
          RegionRequirement &req, Deserializer &derez);
      static void unpack_grant(
          Grant &grant, Deserializer &derez);
      static void unpack_phase_barrier(
          PhaseBarrier &barrier, Deserializer &derez);
    public:
      static void pack_point(Serializer &rez, const DomainPoint &p);
      static void unpack_point(Deserializer &derez, DomainPoint &p);
    public:
      static void log_requirement(UniqueID uid, unsigned idx,
                                 const RegionRequirement &req);
    };

    /**
     * \class SingleTask
     * This is the parent type for each of the single class
     * kinds of classes.  It also serves as the type that
     * represents a context for each application level task.
     */
    class SingleTask : public TaskOp {
    public:
      // A struct for keeping track of local field information
      struct LocalFieldInfo {
      public:
        LocalFieldInfo(void)
          : handle(FieldSpace::NO_SPACE), fid(0),
            field_size(0), reclaim_event(RtEvent::NO_RT_EVENT),
            serdez_id(0) { }
        LocalFieldInfo(FieldSpace sp, FieldID f,
                       size_t size, RtEvent reclaim,
                       CustomSerdezID sid)
          : handle(sp), fid(f), field_size(size),
            reclaim_event(reclaim), serdez_id(sid) { }
      public:
        FieldSpace     handle;
        FieldID        fid;
        size_t         field_size;
        RtEvent        reclaim_event;
        CustomSerdezID serdez_id;
      };
      struct DeferredDependenceArgs {
      public:
        HLRTaskID hlr_id;
        Operation *op;
      };
      struct PostEndArgs {
      public:
        HLRTaskID hlr_id;
        SingleTask *proxy_this;
        void *result;
        size_t result_size;
      };
      struct DecrementArgs {
        HLRTaskID hlr_id;
        SingleTask *parent_ctx;
      };
      struct WindowWaitArgs {
        HLRTaskID hlr_id;
        SingleTask *parent_ctx;
      };
      struct IssueFrameArgs {
        HLRTaskID hlr_id;
        SingleTask *parent_ctx;
        FrameOp *frame;
        ApEvent frame_termination;
      };
      struct AddToDepQueueArgs {
        HLRTaskID hlr_id;
        SingleTask *proxy_this;
        Operation *op;
      };
      struct DeferredPostMappedArgs {
        HLRTaskID hlr_id;
        SingleTask *task;
      };
      struct MapperProfilingInfo {
        SingleTask *task;
        RtUserEvent profiling_done;
      };
    public:
      SingleTask(Runtime *rt);
      virtual ~SingleTask(void);
    protected:
      void activate_single(void);
      void deactivate_single(void);
    public:
      virtual void trigger_dependence_analysis(void) = 0;
    public:
      // This is used enough that we want it inlined
      inline Processor get_executing_processor(void) const
        { return executing_processor; }
      inline void set_executing_processor(Processor p)
        { executing_processor = p; }
    public:
      // These two functions are only safe to call after
      // the task has had its variant selected
      bool is_leaf(void) const;
      bool is_inner(void) const;
      bool has_virtual_instances(void) const;
      bool is_created_region(unsigned index) const;
      void update_no_access_regions(void);
    public:
      void assign_context(RegionTreeContext ctx);
      RegionTreeContext release_context(void);
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const; 
      virtual UniqueID get_context_uid(void) const;
    public:
      void destroy_user_lock(Reservation r);
      void destroy_user_barrier(ApBarrier b);
    public:
      PhysicalRegion get_physical_region(unsigned idx);
      void get_physical_references(unsigned idx, InstanceSet &refs);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual void register_new_child_operation(Operation *op);
      virtual void add_to_dependence_queue(Operation *op, 
                                           bool has_lock);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      bool has_executing_operation(Operation *op);
      bool has_executed_operation(Operation *op);
      void print_children(void);
      void perform_window_wait(void);
    public:
      void perform_fence_analysis(FenceOp *op);
      virtual void update_current_fence(FenceOp *op);
    public:
      void begin_trace(TraceID tid);
      void end_trace(TraceID tid);
    public:
      void issue_frame(FrameOp *frame, ApEvent frame_termination);
      void perform_frame_issue(FrameOp *frame, ApEvent frame_termination);
      void finish_frame(ApEvent frame_termination);
    public:
      void increment_outstanding(void);
      void decrement_outstanding(void);
      void increment_pending(void);
      RtEvent decrement_pending(SingleTask *child) const;
      void decrement_pending(void);
      void increment_frame(void);
      void decrement_frame(void);
    public:
      void add_local_field(FieldSpace handle, FieldID fid, 
                           size_t size, CustomSerdezID serdez_id);
      void add_local_fields(FieldSpace handle, 
                            const std::vector<FieldID> &fields,
                            const std::vector<size_t> &fields_sizes,
                            CustomSerdezID serdez_id);
      void allocate_local_field(const LocalFieldInfo &info);
    public:
      ptr_t perform_safe_cast(IndexSpace is, ptr_t pointer);
      DomainPoint perform_safe_cast(IndexSpace is, const DomainPoint &point);
    public:
      virtual void add_created_region(LogicalRegion handle);
      virtual void add_created_field(FieldSpace handle, FieldID fid);
      // for logging created region requirements
      void log_created_requirement(unsigned index);
    public:
      void get_top_regions(std::map<LogicalRegion,
                                    RegionTreeContext> &top_regions);
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
      bool has_created_region(LogicalRegion handle) const;
      bool has_created_field(FieldSpace handle, FieldID fid) const;
    public:
      bool has_tree_restriction(RegionTreeID tid, const FieldMask &mask);
      void add_tree_restriction(RegionTreeID tid, const FieldMask &mask);
    public:
      void initialize_map_task_input(Mapper::MapTaskInput &input,
                                     Mapper::MapTaskOutput &output,
                                     MustEpochOp *must_epoch_owner,
                               const std::vector<RegionTreeContext> &enclosing,
                                     std::vector<InstanceSet> &valid_instances);
      void finalize_map_task_output(Mapper::MapTaskInput &input,
                                    Mapper::MapTaskOutput &output,
                                    MustEpochOp *must_epoch_owner,
                              const std::vector<RegionTreeContext> &enclosing,
                                    std::vector<InstanceSet> &valid_instances);
    protected: // mapper helper calls
      void validate_target_processors(const std::vector<Processor> &prcs) const;
      void validate_variant_selection(MapperManager *local_mapper,
                    VariantImpl *impl, const char *call_name) const;
    protected:
      void invoke_mapper(MustEpochOp *must_epoch_owner,
          const std::vector<RegionTreeContext> &enclosing_contexts);
      bool map_all_regions(ApEvent user_event,
                           MustEpochOp *must_epoch_owner = NULL); 
      void perform_post_mapping(void);
      void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<ApUserEvent> &unmap_events,
          std::set<ApEvent> &preconditions);
      void invalidate_region_tree_contexts(void);
    public:
      InstanceView* create_instance_top_view(PhysicalManager *manager,
                                             AddressSpaceID source); 
      void notify_instance_deletion(PhysicalManager *deleted, 
                                    GenerationID old_gen);
      void convert_virtual_instance_top_views(
          const std::map<AddressSpaceID,RemoteTask*> &remote_instances);
      static void handle_create_top_view_request(Deserializer &derez, 
                            Runtime *runtime, AddressSpaceID source);
      static void handle_create_top_view_response(Deserializer &derez,
                                                   Runtime *runtime);
    protected:
      void pack_single_task(Serializer &rez, AddressSpaceID target);
      void unpack_single_task(Deserializer &derez, 
                              std::set<RtEvent> &ready_events);
      void pack_remote_context(Serializer &rez, AddressSpaceID target);
      virtual void unpack_remote_context(Deserializer &derez);
      void send_back_created_state(AddressSpaceID target, unsigned start,
                                   RegionTreeContext remote_outermost_context);
    public:
      const std::vector<PhysicalRegion>& begin_task(void);
      void end_task(const void *res, size_t res_size, bool owned);
      void post_end_task(const void *res, size_t res_size, bool owned);
      void unmap_all_mapped_regions(void);
    public:
      VariantImpl* select_inline_variant(TaskOp *child, 
                                         InlineTask *inline_task);
      void inline_child_task(TaskOp *child);
    public:
      void restart_task(void);
      const std::vector<PhysicalRegion>& get_physical_regions(void) const;
    public:
      void notify_profiling_results(Realm::ProfilingResponse &results);
      static void process_mapper_profiling(const void *args, size_t arglen);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
    public:
      virtual void resolve_false(void) = 0;
      virtual void launch_task(void);
      virtual bool early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(MustEpochOp *owner = NULL) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle) = 0;
      virtual bool can_early_complete(ApUserEvent &chain_event) = 0;
      virtual void return_virtual_instance(unsigned index, 
                                           InstanceSet &refs) = 0;
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
      virtual RemoteTask* find_outermost_context(void) = 0;
    public:
      virtual bool has_remote_state(void) const = 0;
      virtual void record_remote_state(void) = 0;
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst) = 0;
    public:
      RegionTreeContext find_enclosing_context(unsigned idx);
      // Override by RemoteTask
      virtual SingleTask* find_parent_context(void);
    public:
      // Override these methods from operation class
      virtual bool trigger_execution(void); 
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid) = 0;
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void find_enclosing_local_fields(
      LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos) = 0;
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant) = 0;
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned) = 0; 
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT) = 0;
    protected:
      // Boolean for each region saying if it is virtual mapped
      std::vector<bool> virtual_mapped;
      // Regions which are NO_ACCESS or have no privilege fields
      std::vector<bool> no_access_regions;
      Processor executing_processor;
    protected:
      std::vector<Processor> target_processors;
      // Hold the result of the mapping 
      LegionDeque<InstanceSet,TASK_INSTANCE_REGION_ALLOC>::tracked 
                                                             physical_instances;
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts
      LegionList<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::tracked
                                                   inline_regions; 
      // Context for this task
      RegionTreeContext context; 
      // Application tasks can manipulate these next two data
      // structures by creating regions and fields, make sure you are
      // holding the operation lock when you are accessing them
      std::deque<RegionRequirement>             created_requirements;
      std::vector<PhysicalRegion>               physical_regions;
    protected: // Instance top view data structures
      std::map<PhysicalManager*,InstanceView*>  instance_top_views;
      std::map<PhysicalManager*,RtUserEvent>    pending_top_views;
    protected: // Mapper choices 
      Mapper::ContextConfigOutput           context_configuration;
      VariantID                             selected_variant;
      TaskPriority                          task_priority;
      bool                                  perform_postmap;
      Mapper::TaskProfilingInfo             profiling_info;
    protected:
      // Events that must be triggered before we are done mapping
      std::set<RtEvent> map_applied_conditions;
    protected:
      // Track whether this task has finished executing
      unsigned outstanding_children_count;
      bool task_executed;
      LegionSet<Operation*,EXECUTING_CHILD_ALLOC>::tracked executing_children;
      LegionSet<Operation*,EXECUTED_CHILD_ALLOC>::tracked executed_children;
      LegionSet<Operation*,COMPLETE_CHILD_ALLOC>::tracked complete_children;
      // Traces for this task's execution
      LegionMap<TraceID,LegionTrace*,TASK_TRACES_ALLOC>::tracked traces;
      LegionTrace *current_trace;
      // Event for waiting when the number of mapping+executing
      // child operations has grown too large.
      bool valid_wait_event;
      RtUserEvent window_wait;
      std::deque<ApEvent> frame_events;
      RtEvent deferred_map;
      RtEvent deferred_complete;
      RtEvent pending_done;
      RtEvent last_registration;
      RtEvent dependence_precondition;
      RtEvent profiling_done; 
    protected:
      mutable bool leaf_cached, is_leaf_result;
      mutable bool inner_cached, is_inner_result;
      mutable bool has_virtual_instances_cached, has_virtual_instances_result;
    protected:
      // Number of sub-tasks ready to map
      unsigned outstanding_subtasks;
      // Number of mapped sub-tasks that are yet to run
      unsigned pending_subtasks;
      // Number of pending_frames
      unsigned pending_frames;
      // Event used to order operations to the runtime
      RtEvent context_order_event;
    protected:
      FenceOp *current_fence;
      GenerationID fence_gen;
#ifdef LEGION_SPY
      UniqueID current_fence_uid;
#endif
    protected:
      // Resources that can build up over a task's lifetime
      LegionDeque<Reservation,TASK_RESERVATION_ALLOC>::tracked context_locks;
      LegionDeque<ApBarrier,TASK_BARRIER_ALLOC>::tracked context_barriers;
      LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked local_fields;
    protected:
      // Some help for performing fast safe casts
      std::map<IndexSpace,Domain> safe_cast_domains;
    protected:
      // Information for tracking restrictions
      LegionMap<RegionTreeID,FieldMask>::aligned restricted_trees;
    };

    /**
     * \class MultiTask
     * This is the parent type for each of the multi-task
     * kinds of classes.
     */
    class MultiTask : public TaskOp {
    public:
      MultiTask(Runtime *rt);
      virtual ~MultiTask(void);
    protected:
      void activate_multi(void);
      void deactivate_multi(void);
    public:
      bool is_sliced(void) const;
      bool slice_index_space(void);
      bool trigger_slices(void);
      void clone_multi_from(MultiTask *task, const Domain &d, Processor p,
                            bool recurse, bool stealable);
      void assign_points(MultiTask *target, const Domain &d);
      void add_point(const DomainPoint &p, MinimalPoint *point);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
    public:
      virtual void trigger_dependence_analysis(void) = 0;
    public:
      virtual void resolve_false(void) = 0;
      virtual bool early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(MustEpochOp *owner = NULL) = 0;
      virtual void launch_task(void) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle) = 0;
      virtual bool map_and_launch(void) = 0;
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual void recapture_version_info(unsigned idx);
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual bool trigger_execution(void);
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant) = 0;
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator) = 0;
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner) = 0;
      virtual void register_must_epoch(void) = 0;
    public:
      virtual void add_created_region(LogicalRegion handle);
      virtual void add_created_field(FieldSpace handle, FieldID fid);
    public:
      void pack_multi_task(Serializer &rez, AddressSpaceID target);
      void unpack_multi_task(Deserializer &derez,
                             std::set<RtEvent> &ready_events);
    public:
      void initialize_reduction_state(void);
      void fold_reduction_future(const void *result, size_t result_size,
                                 bool owner, bool exclusive); 
    protected:
      std::list<SliceTask*> slices;
      std::vector<VersionInfo> version_infos;
      std::vector<RestrictInfo> restrict_infos;
      std::map<DomainPoint,MinimalPoint*> minimal_points;
      unsigned minimal_points_assigned;
      bool sliced;
    protected:
      ReductionOpID redop;
      const ReductionOp *reduction_op;
      // For handling reductions of types with serdez methods
      const SerdezRedopFns *serdez_redop_fns;
      size_t reduction_state_size;
      void *reduction_state; 
    };

    /**
     * \class IndividualTask
     * This class serves as the basis for all individual task
     * launch calls performed by the runtime.
     */
    class IndividualTask : public SingleTask {
    public:
      static const AllocationType alloc_type = INDIVIDUAL_TASK_ALLOC;
    public:
      IndividualTask(Runtime *rt);
      IndividualTask(const IndividualTask &rhs);
      virtual ~IndividualTask(void);
    public:
      IndividualTask& operator=(const IndividualTask &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      Future initialize_task(SingleTask *ctx,
                             const TaskLauncher &launcher, 
                             bool check_privileges,
                             bool track = true);
      Future initialize_task(SingleTask *ctx,
              Processor::TaskFuncID task_id,
              const std::vector<IndexSpaceRequirement> &indexes,
              const std::vector<RegionRequirement> &regions,
              const TaskArgument &arg,
              const Predicate &predicate,
              MapperID id, MappingTagID tag,
              bool check_privileges,
              bool track = true); 
      void initialize_paths(void);
      void set_top_level(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual void report_interfering_close_requirement(unsigned idx);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
    public:
      virtual void resolve_false(void);
      virtual bool early_map_task(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(MustEpochOp *owner = NULL);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual void return_virtual_instance(unsigned index, InstanceSet &refs);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
      virtual void recapture_version_info(unsigned idx);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
      virtual RemoteTask* find_outermost_context(void);
    public:
      virtual bool has_remote_state(void) const;
      virtual void record_remote_state(void);
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst);
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid);
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void find_enclosing_local_fields(
          LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos);
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant);
      virtual bool is_inline_task(void) const;
      virtual const std::vector<PhysicalRegion>& begin_inline_task(void);
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
    protected:
      void pack_remote_complete(Serializer &rez);
      void pack_remote_commit(Serializer &rez);
      void unpack_remote_mapped(Deserializer &derez);
      void unpack_remote_complete(Deserializer &derez);
      void unpack_remote_commit(Deserializer &derez);
    public:
      static void process_unpack_remote_mapped(Deserializer &derez);
      static void process_unpack_remote_complete(Deserializer &derez);
      static void process_unpack_remote_commit(Deserializer &derez);
    protected: 
      void *future_store;
      size_t future_size;
      Future result; 
      std::set<unsigned>          rerun_analysis_requirements; 
      std::set<Operation*>        child_operations;
      std::vector<RegionTreePath> privilege_paths;
      std::vector<VersionInfo>    version_infos;
      std::vector<RestrictInfo>   restrict_infos;
    protected:
      // Information for remotely executing task
      IndividualTask *orig_task; // Not a valid pointer when remote
      ApEvent remote_completion_event;
      UniqueID remote_unique_id;
      RegionTreeContext remote_outermost_context;
      UniqueID remote_owner_uid;
    protected:
      Future predicate_false_future;
      void *predicate_false_result;
      size_t predicate_false_size;
    protected:
      bool sent_remotely;
    protected:
      friend class Internal;
      // Special field for the top level task
      bool top_level_task;
      // Special field for inline tasks
      bool is_inline;
    protected:
      // For detecting when we have remote subtasks
      bool has_remote_subtasks;
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
    protected:
      std::map<PhysicalManager*,
        std::pair<unsigned/*ref count*/,bool/*created*/> > acquired_instances;
    };

    /**
     * \class PointTask
     * A point task is a single point of an index space task
     * launch.  It will primarily be managed by its enclosing
     * slice task owner.
     */
    class PointTask : public SingleTask {
    public:
      static const AllocationType alloc_type = POINT_TASK_ALLOC;
    public:
      PointTask(Runtime *rt);
      PointTask(const PointTask &rhs);
      virtual ~PointTask(void);
    public:
      PointTask& operator=(const PointTask &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
    public:
      virtual void resolve_false(void);
      virtual bool early_map_task(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(MustEpochOp *owner = NULL);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual void return_virtual_instance(unsigned index, InstanceSet &refs);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual void recapture_version_info(unsigned idx);
      virtual bool is_inline_task(void) const;
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
      virtual RemoteTask* find_outermost_context(void);
    public:
      virtual bool has_remote_state(void) const;
      virtual void record_remote_state(void);
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst);
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid);
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void find_enclosing_local_fields(
          LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos);
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      void initialize_point(SliceTask *owner, MinimalPoint *mp);
    protected:
      friend class SliceTask;
      SliceTask                   *slice_owner;
      ApUserEvent                 point_termination;
    protected:
      bool has_remote_subtasks;
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
    };

    /**
     * \class WrapperTask
     * A wrapper task is an abstract class that extends a
     * SingleTask and is used for maintaining the illusion
     * of a SingleTask without needing to go through the
     * whole operation pipeline.  There are two kinds of
     * wrapper tasks: RemoteTasks and InlineTasks both
     * of which provide a context in which to execute
     * child tasks either in a remote node, or in an 
     * inline context.
     */
    class WrapperTask : public SingleTask {
    public:
      WrapperTask(Runtime *rt);
      virtual ~WrapperTask(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
    public:
      virtual void trigger_dependence_analysis(void);
    public:
      virtual void resolve_false(void);
      virtual bool early_map_task(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(MustEpochOp *owner = NULL);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual void return_virtual_instance(unsigned index, InstanceSet &refs);
      virtual RemoteTask* find_outermost_context(void) = 0;
    public:
      virtual bool has_remote_state(void) const = 0;
      virtual void record_remote_state(void) = 0;
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst) = 0;
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid);
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void find_enclosing_local_fields(
      LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos) = 0;
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
    public:
      void activate_wrapper(void);
      void deactivate_wrapper(void);
    };

    /**
     * \class RemoteTask
     * A remote task doesn't actually represent an
     * executing task the way other single task does.
     * Instead it serves as a proxy for holding state
     * for a task that is executing on a remote node
     * but is launching child tasks that are being
     * sent to the current node.  It is reclaimed
     * once the original task finishes on the 
     * original node.
     */
    class RemoteTask : public WrapperTask {
    public:
      static const AllocationType alloc_type = REMOTE_TASK_ALLOC;
    public:
      RemoteTask(Runtime *rt);
      RemoteTask(const RemoteTask &rhs);
      virtual ~RemoteTask(void);
    public:
      RemoteTask& operator=(const RemoteTask &rhs);
    public:
      virtual int get_depth(void) const;
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      void initialize_remote(UniqueID context_uid, bool is_top_level);
    public:
      virtual RemoteTask* find_outermost_context(void);
      virtual UniqueID get_context_uid(void) const;
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
    public:
      virtual bool has_remote_state(void) const;
      virtual void record_remote_state(void);
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst);
      virtual SingleTask* find_parent_context(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void find_enclosing_local_fields(
          LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos);
      virtual void unpack_remote_context(Deserializer &derez);
    public:
      void add_top_region(LogicalRegion handle);
    public:
      void convert_virtual_instances(Deserializer &derez);
      static void handle_convert_virtual_instances(Deserializer &derez);
    protected:
      std::set<LogicalRegion> top_level_regions;
    protected:
      UniqueID remote_owner_uid;
      UniqueID parent_context_uid;
    protected:
      int depth;
      bool is_top_level_context;
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
      ApEvent remote_completion_event;
      std::vector<VersionInfo> version_infos;
    };

    /**
     * \class InlineTask
     * An inline task is a helper object that aides in
     * performing inline task operations
     */
    class InlineTask : public WrapperTask {
    public:
      static const AllocationType alloc_type = INLINE_TASK_ALLOC;
    public:
      InlineTask(Runtime *rt);
      InlineTask(const InlineTask &rhs);
      virtual ~InlineTask(void);
    public:
      InlineTask& operator=(const InlineTask &rhs);
    public:
      void initialize_inline_task(SingleTask *enclosing, TaskOp *clone);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const;
    public:
      virtual RemoteTask* find_outermost_context(void);
    public:
      virtual bool has_remote_state(void) const;
      virtual void record_remote_state(void);
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void find_enclosing_local_fields(
          LegionDeque<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::tracked &infos);
    public:
      virtual void register_new_child_operation(Operation *op);
      virtual void add_to_dependence_queue(Operation *op,
                                           bool has_lock);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      virtual void update_current_fence(FenceOp *op);
    protected:
      SingleTask *enclosing;
    };

    /**
     * \class IndexTask
     * An index task is used to represent an index space task
     * launch performed by the runtime.  It will only live
     * on the node on which it was created.  Eventually the
     * mapper will slice the index space, and the corresponding
     * slice tasks for the index space will be distributed around
     * the machine and eventually returned to this index space task.
     */
    class IndexTask : public MultiTask {
    public:
      static const AllocationType alloc_type = INDEX_TASK_ALLOC;
    public:
      IndexTask(Runtime *rt);
      IndexTask(const IndexTask &rhs);
      virtual ~IndexTask(void);
    public:
      IndexTask& operator=(const IndexTask &rhs);
    public:
      FutureMap initialize_task(SingleTask *ctx,
                                const IndexLauncher &launcher,
                                bool check_privileges,
                                bool track = true);
      Future initialize_task(SingleTask *ctx,
                             const IndexLauncher &launcher,
                             ReductionOpID redop,
                             bool check_privileges,
                             bool track = true);
      FutureMap initialize_task(SingleTask *ctx,
            Processor::TaskFuncID task_id,
            const Domain &launch_domain,
            const std::vector<IndexSpaceRequirement> &indexes,
            const std::vector<RegionRequirement> &regions,
            const TaskArgument &global_arg,
            const ArgumentMap &arg_map,
            const Predicate &predicate,
            bool must_parallelism,
            MapperID id, MappingTagID tag,
            bool check_privileges);
      Future initialize_task(SingleTask *ctx,
            Processor::TaskFuncID task_id,
            const Domain &launch_domain,
            const std::vector<IndexSpaceRequirement> &indexes,
            const std::vector<RegionRequirement> &regions,
            const TaskArgument &global_arg,
            const ArgumentMap &arg_map,
            ReductionOpID redop,
            const TaskArgument &init_value,
            const Predicate &predicate,
            bool must_parallelism,
            MapperID id, MappingTagID tag,
            bool check_privileges);
      void initialize_predicate(const Future &pred_future,
                                const TaskArgument &pred_arg);
      void initialize_paths(void);
      void annotate_early_mapped_regions(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual void report_interfering_close_requirement(unsigned idx);
      virtual FatTreePath* compute_fat_path(unsigned idx);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual void resolve_false(void);
      virtual bool early_map_task(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(MustEpochOp *owner = NULL);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool map_and_launch(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant);
      virtual bool is_inline_task(void) const;
      virtual const std::vector<PhysicalRegion>& begin_inline_task(void);
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator);
    public:
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner);
    public:
      virtual void register_must_epoch(void);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      void enumerate_points(void);
      void record_locally_mapped_slice(SliceTask *local_slice);
    public:
      void return_slice_mapped(unsigned points, long long denom,
                               RtEvent applied_condition);
      void return_slice_complete(unsigned points);
      void return_slice_commit(unsigned points);
    public:
      void unpack_slice_mapped(Deserializer &derez, AddressSpaceID source);
      void unpack_slice_complete(Deserializer &derez);
      void unpack_slice_commit(Deserializer &derez); 
    public:
      static void process_slice_mapped(Deserializer &derez, 
                                       AddressSpaceID source);
      static void process_slice_complete(Deserializer &derez);
      static void process_slice_commit(Deserializer &derez);
    protected:
      friend class SliceTask;
      ArgumentMap argument_map;
      FutureMap future_map;
      Future reduction_future;
      // The fraction used to keep track of what part of
      // the sliced index spaces we have seen
      Fraction<long long> slice_fraction;
      unsigned total_points;
      unsigned mapped_points;
      unsigned complete_points;
      unsigned committed_points;
      // Track whether or not we've received our commit command
      bool complete_received;
      bool commit_received;
    protected:
      Future predicate_false_future;
      void *predicate_false_result;
      size_t predicate_false_size;
    protected:
      std::set<unsigned>          rerun_analysis_requirements;
      std::vector<RegionTreePath> privilege_paths;
      std::deque<SliceTask*> locally_mapped_slices;
    protected:
      std::set<RtEvent> map_applied_conditions;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
    };

    /**
     * \class SliceTask
     * A slice task is a (possibly whole) fraction of an index
     * space task launch.  Once slice task object is made for
     * each slice created by the mapper when (possibly recursively)
     * slicing up the domain of the index space task launch.
     */
    class SliceTask : public MultiTask {
    public:
      static const AllocationType alloc_type = SLICE_TASK_ALLOC;
    public:
      SliceTask(Runtime *rt);
      SliceTask(const SliceTask &rhs);
      virtual ~SliceTask(void);
    public:
      SliceTask& operator=(const SliceTask &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
    public:
      virtual void resolve_false(void);
      virtual bool early_map_task(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(MustEpochOp *owner = NULL);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool map_and_launch(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(SingleTask *ctx, VariantImpl *variant);
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator);
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner);
    public:
      virtual void register_must_epoch(void);
      PointTask* clone_as_point_task(const DomainPoint &p,
                                     MinimalPoint *mp);
      void enumerate_points(void);
      void prewalk_slice(void);
      void apply_local_version_infos(std::set<RtEvent> &map_conditions);
      std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     get_acquired_instances_ref(void);
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
      public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      void return_privileges(PointTask *point);
      void return_virtual_instance(unsigned index, InstanceSet &refs,
                                   const RegionRequirement &req);
      void record_child_mapped(RtEvent child_complete);
      void record_child_complete(void);
      void record_child_committed(void);
    protected:
      void trigger_slice_mapped(void);
      void trigger_slice_complete(void);
      void trigger_slice_commit(void);
    protected:
      void pack_remote_mapped(Serializer &rez, RtEvent applied_condition);
      void pack_remote_complete(Serializer &rez); 
      void pack_remote_commit(Serializer &rez);
    public:
      static void handle_slice_return(Runtime *rt, Deserializer &derez);
    protected:
      friend class IndexTask;
      bool reclaim; // used for reclaiming intermediate slices
      std::deque<PointTask*> points;
    protected:
      unsigned mapping_index;
      unsigned num_unmapped_points;
      unsigned num_uncomplete_points;
      unsigned num_uncommitted_points;
    protected:
      // For knowing which fraction of the
      // domain we have (1/denominator)
      long long denominator;
      IndexTask *index_owner;
      ApEvent index_complete;
      UniqueID remote_unique_id;
      RegionTreeContext remote_outermost_context;
      bool locally_mapped;
      UniqueID remote_owner_uid;
    protected:
      // Temporary storage for future results
      std::map<DomainPoint,std::pair<void*,size_t> > temporary_futures;
      LegionDeque<InstanceRef>::aligned temporary_virtual_refs;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class DeferredSlicer
     * A class for helping with parallelizing the triggering
     * of slice tasks from within MultiTasks
     */
    class DeferredSlicer {
    public:
      struct DeferredSliceArgs {
      public:
        HLRTaskID hlr_id;
        DeferredSlicer *slicer;
        SliceTask *slice;
      };
    public:
      DeferredSlicer(MultiTask *owner);
      DeferredSlicer(const DeferredSlicer &rhs);
      ~DeferredSlicer(void);
    public:
      DeferredSlicer& operator=(const DeferredSlicer &rhs);
    public:
      bool trigger_slices(std::list<SliceTask*> &slices);
      void perform_slice(SliceTask *slice);
    public:
      static void handle_slice(const void *args);
    private:
      Reservation slice_lock;
      MultiTask *const owner;
      std::set<SliceTask*> failed_slices;
    };

    /**
     * \class MinimalPoint
     * A helper class for managing point specific data
     * until we are ready to expand to a full point task
     */
    class MinimalPoint {
    public:
      MinimalPoint(void);
      MinimalPoint(const MinimalPoint &rhs);
      ~MinimalPoint(void);
    public:
      MinimalPoint& operator=(const MinimalPoint &rhs);
    public:
      void add_projection_region(unsigned index, LogicalRegion handle);
      void add_argument(const TaskArgument &arg, bool own);
    public:
      void assign_argument(void *&local_arg, size_t &local_arglen);
      LogicalRegion find_logical_region(unsigned index);
    public:
      void pack(Serializer &rez);
      void unpack(Deserializer &derez);
    protected:
      std::map<unsigned,LogicalRegion> projections;
      void *arg;
      size_t arglen;
      bool own_arg;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_TASKS_H__
