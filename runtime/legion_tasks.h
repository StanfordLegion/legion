/* Copyright 2014 Stanford University
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
#include "legion_utilities.h"
#include "legion_allocation.h"

namespace LegionRuntime {
  namespace HighLevel {

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
      virtual MappableKind get_mappable_kind(void) const;
      virtual Task* as_mappable_task(void) const;
      virtual Copy* as_mappable_copy(void) const;
      virtual Inline* as_mappable_inline(void) const;
      virtual Acquire* as_mappable_acquire(void) const;
      virtual Release* as_mappable_release(void) const;
      virtual UniqueID get_unique_mappable_id(void) const;
    public:
      inline bool is_remote(void) const { return !runtime->is_local(orig_proc);}
      inline bool is_stolen(void) const { return (steal_count > 0); }
      inline bool is_locally_mapped(void) const { return map_locally; }
      inline bool is_premapped(void) const { return premapped; }
    protected:
      void activate_task(void);
      void deactivate_task(void);
    protected:
      void pack_base_task(Serializer &derez, AddressSpaceID target);
      void unpack_base_task(Deserializer &derez);
    public:
      void mark_stolen(Processor new_target);
      void initialize_base_task(SingleTask *ctx, bool track, 
                                const Predicate &p, 
                                Processor::TaskFuncID tid);
      void initialize_physical_contexts(void);
      void check_empty_field_requirements(void);
      size_t check_future_size(Future::Impl *impl);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void);
    public:
      virtual void trigger_dependence_analysis(void) = 0;
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void) = 0;
      virtual bool speculate(bool &value);
    public:
      virtual bool premap_task(void) = 0;
      virtual bool prepare_steal(void) = 0;
      virtual bool defer_mapping(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(bool mapper_invoked = false) = 0;
      virtual void launch_task(void) = 0;
      virtual bool is_stealable(void) const = 0;
    public:
      virtual Event get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      // Returns true if the task should be deactivated
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current) = 0;
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn) = 0;
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
      virtual void add_created_index(IndexSpace handle) = 0;
      virtual void add_created_region(LogicalRegion handle) = 0;
      virtual void add_created_field(FieldSpace handle, FieldID fid) = 0;
      virtual void remove_created_index(IndexSpace handle) = 0;
      virtual void remove_created_region(LogicalRegion handle) = 0;
      virtual void remove_created_field(FieldSpace handle, FieldID fid) = 0;
    public:
      void return_privilege_state(TaskOp *target);
      void pack_privilege_state(Serializer &rez, AddressSpaceID target);
      void unpack_privilege_state(Deserializer &derez);
      void perform_privilege_checks(void);
    public:
      InstanceRef find_premapped_region(unsigned idx);
      RegionTreeContext get_enclosing_physical_context(unsigned idx);
      void clone_task_op_from(TaskOp *rhs, Processor p, 
                              bool stealable, bool duplicate_args);
      void update_grants(const std::vector<Grant> &grants);
      void update_arrival_barriers(const std::vector<PhaseBarrier> &barriers);
      void compute_point_region_requirements(void);
      bool early_map_regions(void);
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
      std::map<unsigned/*idx*/,InstanceRef> early_mapped_regions;
    protected:
      std::deque<RegionTreeContext> enclosing_physical_contexts;
    protected:
      std::set<LogicalRegion>                   created_regions;
      std::set<std::pair<FieldSpace,FieldID> >  created_fields;
      std::set<FieldSpace>                      created_field_spaces;
      std::set<IndexSpace>                      created_index_spaces;
      std::set<LogicalRegion>                   deleted_regions;
      std::set<std::pair<FieldSpace,FieldID> >  deleted_fields;
      std::set<FieldSpace>                      deleted_field_spaces;
      std::set<IndexSpace>                      deleted_index_spaces;
    protected:
      bool complete_received;
      bool commit_received;
      bool children_complete;
      bool children_commit;
      bool children_complete_invoked;
      bool children_commit_invoked;
    protected:
      bool needs_state;
    protected:
      AllocManager *arg_manager;
    protected:
      UserEvent all_children_mapped;
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
            field_size(0), reclaim_event(Event::NO_EVENT) { }
        LocalFieldInfo(FieldSpace sp, FieldID f,
                       size_t size, Event reclaim)
          : handle(sp), fid(f), field_size(size),
            reclaim_event(reclaim) { }
      public:
        FieldSpace handle;
        FieldID    fid;
        size_t     field_size;
        Event      reclaim_event;
      };
      struct PostEndArgs {
      public:
        HLRTaskID hlr_id;
        SingleTask *proxy_this;
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
    public:
      void assign_context(RegionTreeContext ctx);
      RegionTreeContext release_context(void);
      virtual RegionTreeContext get_context(void) const;
      virtual ContextID get_context_id(void) const; 
    public:
      void destroy_user_lock(Reservation r);
      void destroy_user_barrier(Barrier b);
    public:
      PhysicalRegion get_physical_region(unsigned idx);
      void add_inline_task(InlineTask *inline_task);
    public:
      // The following set of operations correspond directly
      // to the complete_mapping, complete_operation, and
      // commit_operations performed by an operation.  Every
      // one of those calls invokes the corresponding one of
      // these calls to notify the parent context.
      virtual void register_child_operation(Operation *op);
      virtual void register_child_executed(Operation *op);
      virtual void register_child_complete(Operation *op);
      virtual void register_child_commit(Operation *op); 
      virtual void unregister_child_operation(Operation *op);
      virtual void register_fence_dependence(Operation *op);
    public:
      bool has_executing_operation(Operation *op);
      bool has_executed_operation(Operation *op);
      void print_children(void);
    public:
      virtual void update_current_fence(FenceOp *op);
    public:
      void begin_trace(TraceID tid);
      void end_trace(TraceID tid);
    public:
      void increment_outstanding(void);
      void decrement_outstanding(void);
      void increment_pending(void);
      void decrement_pending(void);
    public:
      void add_local_field(FieldSpace handle, FieldID fid, size_t size);
      void add_local_fields(FieldSpace handle, 
                            const std::vector<FieldID> &fields,
                            const std::vector<size_t> &fields_sizes);
      void allocate_local_field(const LocalFieldInfo &info);
    public:
      virtual void add_created_index(IndexSpace handle);
      virtual void add_created_region(LogicalRegion handle);
      virtual void add_created_field(FieldSpace handle, FieldID fid);
      virtual void remove_created_index(IndexSpace handle);
      virtual void remove_created_region(LogicalRegion handle);
      virtual void remove_created_field(FieldSpace handle, FieldID fid);
    public:
      void get_top_regions(std::vector<LogicalRegion> &top_regions);
      void analyze_destroy_index_space(IndexSpace handle, Operation *op);
      void analyze_destroy_index_partition(IndexPartition handle, 
                                           Operation *op);
      void analyze_destroy_field_space(FieldSpace handle, Operation *op);
      void analyze_destroy_fields(FieldSpace handle, Operation *op,
                                  const std::set<FieldID> &to_delete);
      void analyze_destroy_logical_region(LogicalRegion handle, Operation *op);
      void analyze_destroy_logical_partition(LogicalPartition handle,
                                             Operation *op);
    public:
      int has_conflicting_regions(MapOp *map, bool &parent_conflict,
                                  bool &inline_conflict);
      void find_conflicting_regions(TaskOp *task,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(CopyOp *copy,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(AcquireOp *acquire,
                                    std::vector<PhysicalRegion> &conflicting);
      void find_conflicting_regions(ReleaseOp *release,
                                    std::vector<PhysicalRegion> &conflicting);
      bool check_region_dependence(RegionTreeID tid, IndexSpace space,
                                  const RegionRequirement &our_req,
                                  const RegionUsage &our_usage,
                                  const RegionRequirement &req);
      void register_inline_mapped_region(PhysicalRegion &region);
      void unregister_inline_mapped_region(PhysicalRegion &region);
    public:
      bool is_region_mapped(unsigned idx);
      unsigned find_parent_region(unsigned idx, TaskOp *task);
      unsigned find_parent_index_region(unsigned idx, TaskOp *task);
      LegionErrorType check_privilege(const IndexSpaceRequirement &req) const;
      LegionErrorType check_privilege(const RegionRequirement &req, 
                                      FieldID &bad_field, 
                                      bool skip_privileges = false) const; 
      bool has_simultaneous_coherence(void);
      void check_simultaneous_restricted(
          RegionRequirement &child_requirement) const;
      void check_simultaneous_restricted(
          std::vector<RegionRequirement> &child_requirements) const;
      bool has_created_region(LogicalRegion handle) const;
      bool has_created_field(FieldSpace handle, FieldID fid) const;
    public:
      void check_index_subspace(IndexSpace handle, const char *caller);
      void check_index_subpartition(IndexPartition handle, const char *caller);
      void check_field_space(FieldSpace handle, const char *caller);
      void check_logical_subregion(LogicalRegion handle, const char *caller);
      void check_logical_subpartition(LogicalPartition handle,
                                      const char *caller);
    public:
      void unmap_all_regions(void);
    protected:
      bool map_all_regions(Processor target, Event user_event, 
                           bool mapper_invoked); 
      void initialize_region_tree_contexts(
          const std::vector<RegionRequirement> &clone_requirements,
          const std::vector<UserEvent> &unmap_events);
      void invalidate_region_tree_contexts(void);
    protected:
      void pack_single_task(Serializer &rez, AddressSpaceID target);
      void unpack_single_task(Deserializer &derez);
      void initialize_mapping_paths(void);
    public:
      void pack_parent_task(Serializer &rez);
    public:
      const std::vector<PhysicalRegion>& begin_task(void);
      void end_task(const void *res, size_t res_size, bool owned);
      void post_end_task(void);
      const std::vector<PhysicalRegion>& get_physical_regions(void) const;
      void inline_child_task(TaskOp *child);
      void restart_task(void);
    public:
      // These methods are VERY important.  They serialize premapping
      // operations to each individual field which considerably simplifies
      // the needed locking protocol in the premapping traversal.
      UserEvent begin_premapping(RegionTreeID tid, const FieldMask &mask);
      void end_premapping(RegionTreeID tid, UserEvent premap_event);
    public:
      PhysicalManager* get_instance(unsigned idx);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
    public:
      virtual void resolve_false(void) = 0;
      virtual void launch_task(void);
      virtual bool premap_task(void) = 0;
      virtual bool prepare_steal(void) = 0;
      virtual bool defer_mapping(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(bool mapper_invoked = false) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool can_early_complete(UserEvent &chain_event) = 0;
    public:
      virtual Event get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent) = 0;
      virtual RemoteTask* find_outermost_physical_context(void) = 0;
    public:
      // Override these methods from operation class
      virtual bool trigger_execution(void);
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current) = 0;
      virtual void find_enclosing_local_fields(
      LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos) = 0;
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn) = 0;
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned) = 0; 
    protected:
      std::vector<RegionTreePath> mapping_paths;
      // Boolean for each region saying if it is virtual mapped
      std::vector<bool> virtual_mapped;
      // Boolean for tracking if regions were mapped locally
      std::vector<bool> locally_mapped;
      // Boolean indicating if any regions have been deleted
      std::vector<bool> region_deleted; 
      // Boolean indicating if any index requirements have been deleted
      std::vector<bool> index_deleted;
      // Number of virtual mapped regions
      unsigned num_virtual_mappings;
      Processor executing_processor;
      // Hold the result of the mapping 
      LegionContainer<InstanceRef,TASK_INSTANCE_REGION_ALLOC>::deque 
                                                    physical_instances;
      // Hold the local instances mapped regions in our context
      // which we will need to close when the task completes
      LegionContainer<InstanceRef,TASK_LOCAL_REGION_ALLOC>::deque
                                                    local_instances;
      // Hold the physical regions for the task's execution
      std::vector<PhysicalRegion> physical_regions;
      // Keep track of inline mapping regions for this task
      // so we can see when there are conflicts
      LegionContainer<PhysicalRegion,TASK_INLINE_REGION_ALLOC>::list
                                                   inline_regions;
      // Context for this task
      RegionTreeContext context; 
    protected:
      // Track whether this task has finished executing
      LegionContainer<Operation*,EXECUTING_CHILD_ALLOC>::set executing_children;
      LegionContainer<Operation*,EXECUTED_CHILD_ALLOC>::set executed_children;
      LegionContainer<Operation*,COMPLETE_CHILD_ALLOC>::set complete_children;
      // Traces for this task's execution
      LegionKeyValue<TraceID,LegionTrace*,TASK_TRACES_ALLOC>::map traces;
      LegionTrace *current_trace;
      // Event for waiting when the number of mapping+executing
      // child operations has grown too large.
      bool valid_wait_event;
      UserEvent window_wait;
      Event deferred_map;
      Event deferred_complete;
    protected:
      // Number of sub-tasks ready to map
      unsigned outstanding_subtasks;
      // Number of mapped sub-tasks that are yet to run
      unsigned pending_subtasks;
    protected:
      FenceOp *current_fence;
      GenerationID fence_gen;
    protected:
      // For handling simultaneous coherence cases
      bool simultaneous_checked, has_simultaneous;
    protected:
      // Resources that can build up over a task's lifetime
      LegionContainer<Reservation,TASK_RESERVATION_ALLOC>::deque context_locks;
      LegionContainer<Barrier,TASK_BARRIER_ALLOC>::deque context_barriers;
      LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque
                                                                 local_fields;
      LegionContainer<InlineTask*,TASK_INLINE_ALLOC>::deque inline_tasks;
    protected:
      // Support for serializing premapping operations.  Note
      // we make it possible for operations accessing different
      // region trees or different fields to premap in parallel.
      std::map<RegionTreeID,std::map<Event,FieldMask> > premapping_events;
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
    protected:
      Event legion_spy_start;
    public:
      inline Event get_start_event(void) { return legion_spy_start; }
#endif
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
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
    public:
      virtual void trigger_dependence_analysis(void) = 0;
    public:
      virtual void resolve_false(void) = 0;
      virtual bool premap_task(void) = 0;
      virtual bool prepare_steal(void) = 0;
      virtual bool defer_mapping(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual bool perform_mapping(bool mapper_invoked = false) = 0;
      virtual void launch_task(void) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool map_and_launch(void) = 0;
    public:
      virtual Event get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual bool trigger_execution(void);
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current) = 0;
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn) = 0;
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator) = 0;
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner) = 0;
      virtual void register_must_epoch(void) = 0;
    public:
      virtual void add_created_index(IndexSpace handle);
      virtual void add_created_region(LogicalRegion handle);
      virtual void add_created_field(FieldSpace handle, FieldID fid);
      virtual void remove_created_index(IndexSpace handle);
      virtual void remove_created_region(LogicalRegion handle);
      virtual void remove_created_field(FieldSpace handle, FieldID fid);
    public:
      void pack_multi_task(Serializer &rez, bool pack_args, 
                           AddressSpaceID target);
      void unpack_multi_task(Deserializer &derez, bool unpack_args);
    public:
      void initialize_reduction_state(void);
      void fold_reduction_future(const void *result, size_t result_size,
                                 bool owner, bool exclusive);
    protected:
      bool sliced;
      Barrier must_barrier; // for must parallelism
      ArgumentMap argument_map;
      std::list<SliceTask*> slices;
    protected:
      ReductionOpID redop;
      const ReductionOp *reduction_op;
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
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void report_aliased_requirements(unsigned idx1, unsigned idx2);
    public:
      virtual void resolve_false(void);
      virtual bool premap_task(void);
      virtual bool prepare_steal(void);
      virtual bool defer_mapping(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(bool mapper_invoked = false);
      virtual bool is_stealable(void) const;
      virtual bool can_early_complete(UserEvent &chain_event);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent);
      virtual RemoteTask* find_outermost_physical_context(void);
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current);
      virtual void find_enclosing_local_fields(
          LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos);
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn);
    protected:
      void pack_remote_mapped(Serializer &rez);
      void pack_remote_complete(Serializer &rez);
      void pack_remote_commit(Serializer &rez);
      void unpack_remote_mapped(Deserializer &derez);
      void unpack_remote_complete(Deserializer &derez);
      void unpack_remote_commit(Deserializer &derez);
    public:
      void send_remote_state(AddressSpaceID target);
      static void handle_individual_request(Runtime *rt, Deserializer &derez,
                                            AddressSpaceID source);
      static void handle_individual_return(Runtime *rt, Deserializer &derez);
    public:
      static void process_unpack_remote_mapped(Deserializer &derez);
      static void process_unpack_remote_complete(Deserializer &derez);
      static void process_unpack_remote_commit(Deserializer &derez);
    protected: 
      void *future_store;
      size_t future_size;
      Future result; 
      std::set<Operation*> child_operations;
      std::vector<RegionTreePath> privilege_paths;
    protected:
      // Information for remotely executing task
      IndividualTask *orig_task; // Not a valid pointer when remote
      Event remote_completion_event;
      UniqueID remote_unique_id;
      RegionTreeContext remote_outermost_context;
      std::deque<RegionTreeContext> remote_contexts;
    protected:
      Future predicate_false_future;
      void *predicate_false_result;
      size_t predicate_false_size;
    protected:
      bool sent_remotely;
    protected:
      friend class Runtime;
      // Special field for the top level task
      bool top_level_task;
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
      virtual bool premap_task(void);
      virtual bool prepare_steal(void);
      virtual bool defer_mapping(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(bool mapper_invoked = false);
      virtual bool is_stealable(void) const;
      virtual bool can_early_complete(UserEvent &chain_event);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent);
      virtual RemoteTask* find_outermost_physical_context(void);
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current);
      virtual void find_enclosing_local_fields(
          LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos);
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
    public:
      void initialize_point(SliceTask *owner);
      void send_back_remote_state(AddressSpaceID target, unsigned index,
                                  RegionTreeContext remote_context,
                                  std::set<PhysicalManager*> &needed_managers);
      void send_back_created_state(AddressSpaceID target, unsigned start_idx, 
                                   RegionTreeContext remote_outermost,
                                   std::set<PhysicalManager*> &needed_managers);
    protected:
      friend class SliceTask;
      SliceTask *slice_owner;
      UserEvent point_termination;
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
      virtual bool premap_task(void);
      virtual bool prepare_steal(void);
      virtual bool defer_mapping(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(bool mapper_invoked = false);
      virtual bool is_stealable(void) const;
      virtual bool can_early_complete(UserEvent &chain_event);
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent) = 0;
      virtual RemoteTask* find_outermost_physical_context(void) = 0;
    public:
      virtual Event get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current);
      virtual void find_enclosing_local_fields(
      LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos) = 0;
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
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
      void initialize_remote(UniqueID uid);
      void unpack_parent_task(Deserializer &derez);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent);
      virtual RemoteTask* find_outermost_physical_context(void);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void find_enclosing_local_fields(
          LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos);
    public:
      void add_top_region(LogicalRegion handle);
    protected:
      std::set<LogicalRegion> top_level_regions;
#if defined(LEGION_LOGGING) || defined(LEGION_SPY)
    protected:
      Event remote_legion_spy_completion;
#endif
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
      virtual RegionTreeContext find_enclosing_physical_context(
                                                LogicalRegion parent);
      virtual RemoteTask* find_outermost_physical_context(void);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void find_enclosing_local_fields(
          LegionContainer<LocalFieldInfo,TASK_LOCAL_FIELD_ALLOC>::deque &infos);
    public:
      virtual void register_child_operation(Operation *op);
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
      virtual void report_aliased_requirements(unsigned idx1, unsigned idx2);
    public:
      virtual void resolve_false(void);
      virtual bool premap_task(void);
      virtual bool prepare_steal(void);
      virtual bool defer_mapping(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(bool mapper_invoked = false);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool map_and_launch(void);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current);
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn);
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator);
    public:
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner);
      virtual void register_must_epoch(void);
    public:
      void return_slice_mapped(unsigned points, long long denom);
      void return_slice_complete(unsigned points);
      void return_slice_commit(unsigned points);
    public:
      void unpack_slice_mapped(Deserializer &derez);
      void unpack_slice_complete(Deserializer &derez);
      void unpack_slice_commit(Deserializer &derez); 
    public:
      static void process_slice_mapped(Deserializer &derez);
      static void process_slice_complete(Deserializer &derez);
      static void process_slice_commit(Deserializer &derez);
    public:
      void send_remote_state(AddressSpaceID target, UniqueID uid);
      static void handle_slice_request(Runtime *rt, Deserializer &derez,
                                       AddressSpaceID source);
    protected:
      friend class SliceTask;
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
      std::vector<RegionTreePath> privilege_paths;
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
    public:
      virtual void resolve_false(void);
      virtual bool premap_task(void);
      virtual bool prepare_steal(void);
      virtual bool defer_mapping(void);
      virtual bool distribute_task(void);
      virtual bool perform_mapping(bool mapper_invoke = false);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool map_and_launch(void);
    public:
      virtual Event get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current);
      virtual void perform_inlining(SingleTask *ctx, InlineFnptr fn);
    public:
      virtual SliceTask* clone_as_slice_task(const Domain &d,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator);
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner);
      virtual void register_must_epoch(void);
      PointTask* clone_as_point_task(const DomainPoint &p);
      void enumerate_points(void);
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      void return_privileges(PointTask *point);
      void record_child_mapped(void);
      void record_child_complete(void);
      void record_child_committed(void);
    protected:
      void trigger_slice_mapped(void);
      void trigger_slice_complete(void);
      void trigger_slice_commit(void);
    protected:
      void pack_remote_mapped(Serializer &rez);
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
      Event index_complete;
      UniqueID remote_unique_id;
      std::deque<RegionTreeContext> remote_contexts;
      RegionTreeContext remote_outermost_context;
      bool locally_mapped;
    protected:
      // Temporary storage for future results
      std::map<DomainPoint,std::pair<void*,size_t>,
                DomainPoint::STLComparator> temporary_futures;
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

  }; // namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_TASKS_H__
