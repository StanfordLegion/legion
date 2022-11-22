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


#ifndef __LEGION_TASKS_H__
#define __LEGION_TASKS_H__

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/region_tree.h"
#include "legion/legion_mapping.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ExternalTask
     * An extention of the external-facing Task to help
     * with packing and unpacking them
     */
    class ExternalTask : public Task, public ExternalMappable {
    public:
      ExternalTask(void);
    public:
      void pack_external_task(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_task(Deserializer &derez, Runtime *runtime);
    public:
      static void pack_output_requirement(
          const OutputRequirement &req, Serializer &rez);
    public:
      static void unpack_output_requirement(
          OutputRequirement &req, Deserializer &derez);
    public:
      virtual void set_context_index(size_t index) = 0;
    protected:
      AllocManager *arg_manager;
    };

    /**
     * \class TaskRegions
     * This is a helper class for accessing the region requirements of a task
     */
    class TaskRequirements {
    public:
      TaskRequirements(Task &t) : task(t) { }
    public:
      inline size_t size(void) const 
        { return task.regions.size() + task.output_regions.size(); }
      inline bool is_output_created(unsigned idx) const
        { if (idx < task.regions.size()) return false;
          return (task.output_regions[idx-task.regions.size()].flags &
                        LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG); }
      inline RegionRequirement& operator[](unsigned idx)
        { return (idx < task.regions.size()) ? task.regions[idx] :
                        task.output_regions[idx - task.regions.size()]; }
      inline const RegionRequirement& operator[](unsigned idx) const
        { return (idx < task.regions.size()) ? task.regions[idx] : 
                        task.output_regions[idx - task.regions.size()]; }
    private:
      Task &task;
    };

    /**
     * \class TaskOp
     * This is the base task operation class for all
     * kinds of tasks in the system.
     */
    class TaskOp : public ExternalTask, public MemoizableOp<SpeculativeOp> {
    public:
      enum TaskKind {
        INDIVIDUAL_TASK_KIND,
        POINT_TASK_KIND,
        INDEX_TASK_KIND,
        SLICE_TASK_KIND,
        SHARD_TASK_KIND,
      };
    public:
      struct TriggerTaskArgs : public LgTaskArgs<TriggerTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_TASK_ID;
      public:
        TriggerTaskArgs(TaskOp *t)
          : LgTaskArgs<TriggerTaskArgs>(t->get_unique_op_id()), op(t) { }
      public:
        TaskOp *const op;
      };
      struct DeferMappingArgs : public LgTaskArgs<DeferMappingArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_MAPPING_TASK_ID;
      public:
        DeferMappingArgs(TaskOp *op, MustEpochOp *owner,
                         RtUserEvent done, unsigned cnt,
                         std::vector<unsigned> *performed,
                         std::vector<ApEvent> *eff)
          : LgTaskArgs<DeferMappingArgs>(op->get_unique_op_id()),
            proxy_this(op), must_op(owner), done_event(done),
            invocation_count(cnt), performed_regions(performed),
            effects(eff) { }
      public:
        TaskOp *const proxy_this;
        MustEpochOp *const must_op;
        const RtUserEvent done_event;
        const unsigned invocation_count;
        std::vector<unsigned> *const performed_regions;
        std::vector<ApEvent> *const effects;
      };
    public:
      TaskOp(Runtime *rt);
      virtual ~TaskOp(void);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual bool has_parent_task(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual const std::string& get_provenance_string(bool human = true) const;
      virtual const char* get_task_name(void) const;
      virtual bool is_reducing_future(void) const;
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void pack_profiling_requests(Serializer &rez,
                                           std::set<RtEvent> &applied) const;
    public:
      bool is_remote(void) const;
      inline bool is_stolen(void) const { return (steal_count > 0); }
      inline bool is_origin_mapped(void) const { return map_origin; }
      inline bool is_replicated(void) const { return replicate; }
      int get_depth(void) const;
    public:
      void set_current_proc(Processor current);
      inline void set_origin_mapped(bool origin) { map_origin = origin; }
      inline void set_replicated(bool repl) { replicate = repl; }
      inline void set_target_proc(Processor next) { target_proc = next; }
    protected:
      void activate_task(void);
      void deactivate_task(void);
    public:
      void set_must_epoch(MustEpochOp *epoch, unsigned index,
                          bool do_registration);
    public:
      void pack_base_task(Serializer &rez, AddressSpaceID target);
      void unpack_base_task(Deserializer &derez,
                            std::set<RtEvent> &ready_events);
      void pack_base_external_task(Serializer &rez, AddressSpaceID target);
      void unpack_base_external_task(Deserializer &derez);
    public:
      void mark_stolen(void);
      void initialize_base_task(InnerContext *ctx, bool track,
            const std::vector<StaticDependence> *dependences,
            const Predicate &p, Processor::TaskFuncID tid,
            Provenance *provenance);
      void check_empty_field_requirements(void);
    public:
      bool select_task_options(bool prioritize);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { exec_fence = false; return !regions.empty(); }
    public:
      virtual void trigger_dependence_analysis(void) = 0;
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched) = 0;
    public:
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void update_atomic_locks(const unsigned index,
                                       Reservation lock, bool exclusive);
      virtual unsigned find_parent_index(unsigned idx);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual RegionTreePath& get_privilege_path(unsigned idx);
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const;
      virtual std::map<PhysicalManager*,unsigned>*
                                            get_acquired_instances_ref(void);
    public:
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL) = 0;
      virtual void launch_task(bool inline_task = false) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool is_output_global(unsigned idx) const { return false; }
      virtual bool is_output_valid(unsigned idx) const { return false; }
    public:
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      // Returns true if the task should be deactivated
      virtual bool pack_task(Serializer &rez, AddressSpaceID target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions) = 0;
    public:
      void defer_distribute_task(RtEvent precondition);
      RtEvent defer_perform_mapping(RtEvent precondition, MustEpochOp *op,
                                    const DeferMappingArgs *args,
                                    unsigned invocation_count,
                                    std::vector<unsigned> *performed = NULL,
                                    std::vector<ApEvent> *effects = NULL);
      void defer_launch_task(RtEvent precondition);
      void enqueue_ready_task(bool use_target_processor,
                              RtEvent wait_on = RtEvent::NO_RT_EVENT);
    public:
      // Tell the parent context that this task is in a ready queue
      void activate_outstanding_task(void);
      void deactivate_outstanding_task(void);
    public:
      void perform_privilege_checks(void);
    public:
      void clone_task_op_from(TaskOp *rhs, Processor p,
                              bool stealable, bool duplicate_args);
      void update_grants(const std::vector<Grant> &grants);
      void update_arrival_barriers(const std::vector<PhaseBarrier> &barriers);
      void compute_point_region_requirements(void);
      void complete_point_projection(void);
      bool prepare_steal(void);
    public:
      void compute_parent_indexes(TaskContext *alt_context = NULL);
      void perform_intra_task_alias_analysis(bool is_tracing,
          LegionTrace *trace, std::vector<RegionTreePath> &privilege_paths);
    public:
      // From Memoizable
      virtual const RegionRequirement& get_requirement(unsigned idx) const
        { return logical_regions[idx]; }
    public: // helper for mapping, here because of inlining
      void validate_variant_selection(MapperManager *local_mapper,
                          VariantImpl *impl, Processor::Kind kind, 
                          const std::deque<InstanceSet> &physical_instances,
                          const char *call_name) const;
    public:
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
      TaskRequirements                          logical_regions;
      // A map of any locks that we need to take for this task
      std::map<Reservation,bool/*exclusive*/>   atomic_locks;
      // Set of acquired instances for this task
      std::map<PhysicalManager*,unsigned/*ref count*/> acquired_instances;
    protected:
      std::vector<unsigned>                     parent_req_indexes;
      // The version infos for this task
      LegionVector<VersionInfo>                 version_infos;
    protected:
      bool complete_received;
      bool commit_received;
    protected:
      bool options_selected;
      bool memoize_selected;
      bool map_origin;
      bool request_valid_instances;
      bool elide_future_return;
      bool replicate;
    protected:
      // For managing predication
      PredEvent true_guard;
      PredEvent false_guard;
    private:
      mutable bool is_local;
      mutable bool local_cached;
    protected:
      bool children_complete;
      bool children_commit;
    protected:
      MapperManager *mapper;
    public:
      // Index for this must epoch op
      unsigned must_epoch_index;
    public:
      // Static methods
      static void process_unpack_task(Runtime *rt, Deserializer &derez);
      static void process_remote_replay(Runtime *rt, Deserializer &derez);
    public:
      static void log_requirement(UniqueID uid, unsigned idx,
                                 const RegionRequirement &req);
    };

    /**
     * \class RemoteTaskOp
     * This is a remote copy of a TaskOp to be used
     * for mapper calls and other operations
     */
    class RemoteTaskOp : public ExternalTask, public RemoteOp {
    public:
      RemoteTaskOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteTaskOp(const RemoteTaskOp &rhs);
      virtual ~RemoteTaskOp(void);
    public:
      RemoteTaskOp& operator=(const RemoteTaskOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual int get_depth(void) const;
      virtual bool has_parent_task(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual const std::string& get_provenance_string(bool human = true) const;
      virtual const char* get_task_name(void) const;
      virtual Domain get_slice_domain(void) const;
      virtual ShardID get_shard_id(void) const;
      virtual size_t get_total_shards(void) const;
      virtual DomainPoint get_shard_point(void) const;
      virtual Domain get_shard_domain(void) const;
      virtual void set_context_index(size_t index);
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez);
    };

    /**
     * \class SingleTask
     * This is the parent type for each of the single class
     * kinds of classes.  It also serves as the type that
     * represents a context for each application level task.
     */
    class SingleTask : public TaskOp {
    public:
      struct MisspeculationTaskArgs :
        public LgTaskArgs<MisspeculationTaskArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MISSPECULATE_TASK_ID;
      public:
        MisspeculationTaskArgs(SingleTask *t)
          : LgTaskArgs<MisspeculationTaskArgs>(t->get_unique_op_id()),
            task(t) { }
      public:
        SingleTask *const task;
      };
      struct DeferTriggerTaskCompleteArgs :
        public LgTaskArgs<DeferTriggerTaskCompleteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_TRIGGER_TASK_COMPLETE_TASK_ID;
      public:
        DeferTriggerTaskCompleteArgs(SingleTask *t)
          : LgTaskArgs<DeferTriggerTaskCompleteArgs>(t->get_unique_op_id()),
            task(t) { }
      public:
        SingleTask *const task;
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
      // These two functions are only safe to call after
      // the task has had its variant selected
      bool is_leaf(void) const;
      bool is_inner(void) const;
      bool is_created_region(unsigned index) const;
      void update_no_access_regions(void);
      void clone_single_from(SingleTask *task);
    public:
      inline void clone_virtual_mapped(std::vector<bool> &target) const
        { target = virtual_mapped; }
      inline void clone_parent_req_indexes(std::vector<unsigned> &target) const
        { target = parent_req_indexes; }
      inline const std::deque<InstanceSet>&
        get_physical_instances(void) const { return physical_instances; }
      inline const std::vector<bool>& get_no_access_regions(void) const
        { return no_access_regions; }
      inline VariantID get_selected_variant(void) const
        { return selected_variant; }
      inline const std::set<RtEvent>& get_map_applied_conditions(void) const
        { return map_applied_conditions; }
      inline RtEvent get_profiling_reported(void) const
        { return profiling_reported; }
    public:
      RtEvent perform_versioning_analysis(const bool post_mapper);
      void initialize_map_task_input(Mapper::MapTaskInput &input,
                                     Mapper::MapTaskOutput &output,
                                     MustEpochOp *must_epoch_owner,
                                     std::vector<InstanceSet> &valid_instances);
      void finalize_map_task_output(Mapper::MapTaskInput &input,
                                    Mapper::MapTaskOutput &output,
                                    MustEpochOp *must_epoch_owner,
                                    std::vector<InstanceSet> &valid_instances);
    protected:
      void prepare_output_instance(unsigned index,
                                   InstanceSet &instance_set,
                                   const RegionRequirement &req,
                                   Memory target,
                                   const LayoutConstraintSet &constraints);
      void finalize_output_regions(void);
    public:
      void replay_map_task_output(void);
      virtual InnerContext* create_implicit_context(void);
      void set_shard_manager(ShardManager *manager);
    protected: // mapper helper call
      void validate_target_processors(const std::vector<Processor> &prcs) const;
    protected:
      void invoke_mapper(MustEpochOp *must_epoch_owner);
      void invoke_mapper_replicated(MustEpochOp *must_epoch_owner);
      RtEvent map_all_regions(MustEpochOp *must_epoch_owner,
                              const DeferMappingArgs *defer_args);
      void perform_post_mapping(const TraceInfo &trace_info);
      void check_future_return_bounds(FutureInstance *instance) const;
    protected:
      void pack_single_task(Serializer &rez, AddressSpaceID target);
      void unpack_single_task(Deserializer &derez,
                              std::set<RtEvent> &ready_events);
      void send_remote_context(AddressSpaceID target, RemoteTask *dst);
    public:
      virtual void pack_profiling_requests(Serializer &rez,
                                           std::set<RtEvent> &applied) const;
      virtual int add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &respone,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      void finalize_single_task_profiling(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual bool is_top_level_task(void) const { return false; }
#ifdef DEBUG_LEGION
      virtual bool is_implicit_top_level_task(void) const { return false; }
#endif
      virtual bool is_shard_task(void) const { return false; }
      virtual SingleTask* get_origin_task(void) const = 0;
    public:
      virtual void resolve_false(bool speculated, bool launched) = 0;
      virtual void launch_task(bool inline_task = false);
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL) = 0;
      virtual void handle_future_size(size_t return_type_size,
                                      bool has_return_type_size,
                                      std::set<RtEvent> &applied_events) = 0;
      virtual void trigger_replay(void);
      // For tasks that are sharded off by control replication
      virtual void shard_off(RtEvent mapped_precondition);
      virtual bool is_stealable(void) const = 0;
    public:
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      // Override these methods from operation class
      virtual void trigger_mapping(void);
    protected:
      friend class ShardManager;
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0; 
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions);
    public:
      virtual void handle_future(FutureInstance *instance,
                                 void *metadata, size_t metasize,
                                 FutureFunctor *functor,
                                 Processor future_proc,
                                 bool own_functor) = 0;
      virtual void handle_post_mapped(bool deferral,
                          RtEvent pre = RtEvent::NO_RT_EVENT) = 0;
      virtual void handle_misspeculation(void) = 0;
    public:
      // From Memoizable
      virtual ApEvent get_memo_completion(void) const;
      virtual void replay_mapping_output(void) { replay_map_task_output(); }
      virtual void set_effects_postcondition(ApEvent postcondition);
    public:
      void handle_remote_profiling_response(Deserializer &derez);
      static void process_remote_profiling_response(Deserializer &derez);
    public:
      void perform_concurrent_analysis(Processor target, RtEvent precondition);
      void trigger_children_complete(ApEvent all_children_complete);
    protected:
      virtual InnerContext* initialize_inner_execution_context(VariantImpl *v,
                                                            bool inline_task);
      virtual EquivalenceSet* create_initial_equivalence_set(unsigned idx);
    public:
      static void handle_deferred_task_complete(const void *args);
    protected:
      // Boolean for each region saying if it is virtual mapped
      std::vector<bool>                           virtual_mapped;
      // Regions which are NO_ACCESS or have no privilege fields
      std::vector<bool>                           no_access_regions; 
    protected:
      std::vector<Processor>                      target_processors;
      // Hold the result of the mapping 
      std::deque<InstanceSet>                     physical_instances;
      std::vector<std::vector<PhysicalManager*> > source_instances;
      std::vector<Memory>                         future_memories;
    protected: // Mapper choices 
      std::set<unsigned>                          untracked_valid_regions;
      VariantID                                   selected_variant;
      TaskPriority                                task_priority;
      bool                                        perform_postmap;
    protected:
      // origin-mapped cases need to know if they've been mapped or not yet
      bool                                  first_mapping;
      std::set<RtEvent>                     intra_space_mapping_dependences;
      // Events that must be triggered before we are done mapping
      std::set<RtEvent>                     map_applied_conditions;
      RtUserEvent                           deferred_complete_mapping;
      // The single task termination event encapsulates the exeuction of the
      // task being done and all child operations and their effects being done
      // It does NOT encapsulate the 'effects_complete' of this task
      // Only the actual operation completion event captures that
      ApUserEvent                           single_task_termination;
      // An event describing the fence event for concurrent execution
      ApEvent                               concurrent_fence_event;
      // Event recording when all "effects" are complete
      // The effects of the task include the following:
      // 1. the execution of the task
      // 2. the execution of all child ops of the task
      // 3. all copy-out operations of child ops
      // 4. all copy-out operations of the task itself
      // Note that this definition is recursive
      ApEvent                               task_effects_complete;
    protected:
      TaskContext*                          execution_context;
      RemoteTraceRecorder*                  remote_trace_recorder;
      // For replication of this task
      ShardManager*                         shard_manager;
    protected:
      std::map<AddressSpaceID,RemoteTask*>  remote_instances;
    protected:
      mutable bool leaf_cached, is_leaf_result;
      mutable bool inner_cached, is_inner_result;
    protected:
      // Profiling information
      struct SingleProfilingInfo : public Mapping::Mapper::TaskProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>      task_profiling_requests;
      std::vector<ProfilingMeasurementID>      copy_profiling_requests;
      std::vector<SingleProfilingInfo>                  profiling_info;
      RtUserEvent                                   profiling_reported;
      int                                           profiling_priority;
      int                                           copy_fill_priority;
      std::atomic<int>                  outstanding_profiling_requests;
      std::atomic<int>                  outstanding_profiling_reported;
#ifdef DEBUG_LEGION
    protected:
      // For checking that premapped instances didn't change during mapping
      std::map<unsigned/*index*/,
               std::vector<Mapping::PhysicalInstance> > premapped_instances;
#endif
    };

    /**
     * \class MultiTask
     * This is the parent type for each of the multi-task
     * kinds of classes.
     */
    class MultiTask : public TaskOp {
    public:
      class OutputOptions {
      public:
        OutputOptions(void) : store(0) { }
        OutputOptions(bool global, bool valid)
          : store((global ? 1 : 0) | (valid ? 2 : 0)) { } 
      public:
        inline bool global_indexing(void) const { return (store & 1); }
        inline bool valid_requirement(void) const { return (store & 2); }
      private:
        unsigned char store;
      };
      struct FutureHandles : public Collectable {
      public:
        std::map<DomainPoint,DistributedID> handles;
      };
    public:
      MultiTask(Runtime *rt);
      virtual ~MultiTask(void);
    protected:
      void activate_multi(void);
      void deactivate_multi(void);
    public:
      bool is_sliced(void) const;
      void slice_index_space(void);
      void trigger_slices(void);
      void clone_multi_from(MultiTask *task, IndexSpace is, Processor p,
                            bool recurse, bool stealable); 
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual bool is_reducing_future(void) const { return (redop > 0); }
      virtual Domain get_slice_domain(void) const;
      virtual ShardID get_shard_id(void) const { return 0; }
      virtual size_t get_total_shards(void) const { return 1; }
      virtual DomainPoint get_shard_point(void) const { return DomainPoint(0); }
      virtual Domain get_shard_domain(void) const 
        { return Domain(DomainPoint(0),DomainPoint(0)); }
    public:
      virtual void trigger_dependence_analysis(void) = 0;
    public:
      virtual void resolve_false(bool speculated, bool launched) = 0;
      virtual void premap_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL) = 0;
      virtual void launch_task(bool inline_task = false) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual void map_and_launch(void) = 0;
    public:
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual void trigger_mapping(void);
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions) = 0;
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
                      Processor p, bool recurse, bool stealable) = 0;
      virtual void reduce_future(const DomainPoint &point,
                                 FutureInstance *instance) = 0;
      virtual void register_must_epoch(void) = 0;
    public:
      // Methods for supporting intra-index-space mapping dependences
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point) = 0;
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped) = 0;
    public:
      // Support for concurrent execution of index tasks
      inline RtEvent get_concurrent_precondition(void) const
        { return concurrent_precondition; }
    public:
      void pack_multi_task(Serializer &rez, AddressSpaceID target);
      void unpack_multi_task(Deserializer &derez,
                             std::set<RtEvent> &ready_events);
    public:
      // Return true if it is safe to delete the future
      bool fold_reduction_future(FutureInstance *instance, 
          std::vector<ApEvent> &applied_effects, bool exclusive);
    protected:
      std::list<SliceTask*> slices;
      bool sliced;
    protected:
      IndexSpaceNode *launch_space; // global set of points
      IndexSpace internal_space; // local set of points
      FutureMap future_map;
      FutureHandles *future_handles;
      ReductionOpID redop;
      bool deterministic_redop;
      const ReductionOp *reduction_op;
      FutureMap point_arguments;
      std::vector<FutureMap> point_futures;
      std::vector<OutputOptions> output_region_options;
      // For handling reductions of types with serdez methods
      const SerdezRedopFns *serdez_redop_fns;
      FutureInstance *reduction_instance;
      ApEvent reduction_inst_precondition;
      std::vector<ApEvent> reduction_effects;
      // Only for handling serdez reductions
      void *serdez_redop_state;
      size_t serdez_redop_state_size;
      // Reduction metadata
      void *reduction_metadata;
      size_t reduction_metasize;
      // Temporary storage for future results
      std::map<DomainPoint,FutureInstance*> temporary_futures;
      // used for detecting cases where we've already mapped a mutli task
      // on the same node but moved it to a different processor
      bool first_mapping;
    protected:
      // Precondition for performing concurrent analyses across the points
      RtEvent concurrent_precondition;
      RtUserEvent concurrent_verified;
      std::map<DomainPoint,Processor> concurrent_processors;
    protected:
      bool children_complete_invoked;
      bool children_commit_invoked;
    protected:
      Future predicate_false_future;
      void *predicate_false_result;
      size_t predicate_false_size;
    protected:
      std::map<DomainPoint,RtEvent> intra_space_dependences;
    };

    /**
     * \class IndividualTask
     * This class serves as the basis for all individual task
     * launch calls performed by the runtime.
     */
    class IndividualTask : public SingleTask,
                           public LegionHeapify<IndividualTask> {
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
    protected:
      void activate_individual_task(void);
      void deactivate_individual_task(void);
      virtual SingleTask* get_origin_task(void) const { return orig_task; }
      virtual Domain get_slice_domain(void) const { return Domain::NO_DOMAIN; }
      virtual ShardID get_shard_id(void) const { return 0; }
      virtual size_t get_total_shards(void) const { return 1; }
      virtual DomainPoint get_shard_point(void) const { return DomainPoint(0); }
      virtual Domain get_shard_domain(void) const
        { return Domain(DomainPoint(0),DomainPoint(0)); }
    public:
      Future initialize_task(InnerContext *ctx,
                             const TaskLauncher &launcher,
                             Provenance *provenance,
                             bool track = true, bool top_level=false,
                             bool implicit_top_level = false,
                             std::vector<OutputRequirement> *outputs = NULL);
      void perform_base_dependence_analysis(void);
    protected:
      void create_output_regions(std::vector<OutputRequirement> &outputs);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2); 
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL);
      virtual void handle_future_size(size_t return_type_size,
                                      bool has_return_type_size,
                                      std::set<RtEvent> &applied_events);
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions);
      virtual bool is_stealable(void) const;
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual bool is_output_valid(unsigned idx) const;
    public:
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void handle_future(FutureInstance *instance,
                                 void *metadata, size_t metasize,
                                 FutureFunctor *functor,
                                 Processor future_proc,
                                 bool own_functor);
      virtual void handle_post_mapped(bool deferral, 
                          RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual void handle_misspeculation(void);
      virtual void prepare_map_must_epoch(void);
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual bool is_top_level_task(void) const { return top_level_task; }
#ifdef DEBUG_LEGION
      virtual bool is_implicit_top_level_task(void) const 
        { return implicit_top_level_task; }
#endif
    protected:
      void pack_remote_versions(Serializer &rez);
      void pack_remote_complete(Serializer &rez, RtEvent precondition);
      void pack_remote_commit(Serializer &rez);
      void unpack_remote_complete(Deserializer &derez);
      void unpack_remote_commit(Deserializer &derez);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
      virtual void complete_replay(ApEvent completion_event);
    public:
      static void process_unpack_remote_future_size(Deserializer &derez);
      static void process_unpack_remote_complete(Deserializer &derez);
      static void process_unpack_remote_commit(Deserializer &derez);
    protected: 
      Future result; 
      std::vector<RegionTreePath> privilege_paths;
    protected:
      std::vector<bool> valid_output_regions;
    protected:
      // Information for remotely executing task
      IndividualTask *orig_task; // Not a valid pointer when remote
      UniqueID remote_unique_id;
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
      bool implicit_top_level_task;
      // Whether we have to do intra-task alias analysis
      bool need_intra_task_alias_analysis;
    };

    /**
     * \class PointTask
     * A point task is a single point of an index space task
     * launch.  It will primarily be managed by its enclosing
     * slice task owner.
     */
    class PointTask : public SingleTask, public ProjectionPoint,
                      public LegionHeapify<PointTask> {
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
      virtual SingleTask* get_origin_task(void) const { return orig_task; }
      virtual Domain get_slice_domain(void) const 
        { return Domain(index_point,index_point); }
      virtual ShardID get_shard_id(void) const { return 0; }
      virtual size_t get_total_shards(void) const { return 1; }
      virtual DomainPoint get_shard_point(void) const { return DomainPoint(0); }
      virtual Domain get_shard_domain(void) const
        { return Domain(DomainPoint(0),DomainPoint(0)); }
      virtual bool is_reducing_future(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_replay(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL);
      virtual void handle_future_size(size_t return_type_size,
                                      bool has_return_type_size,
                                      std::set<RtEvent> &applied_events);
      virtual void shard_off(RtEvent mapped_precondition);
      virtual bool is_stealable(void) const;
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual bool is_output_global(unsigned idx) const; 
      virtual bool is_output_valid(unsigned idx) const;
    public:
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
    public:
      virtual void handle_future(FutureInstance *instance,
                                 void *metadata, size_t metasize,
                                 FutureFunctor *functor,
                                 Processor future_proc,
                                 bool own_functor);
      virtual void handle_post_mapped(bool deferral,
                          RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual void handle_misspeculation(void);
    public:
      // ProjectionPoint methods
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
      virtual void record_intra_space_dependences(unsigned index,
             const std::vector<DomainPoint> &dependences);
      virtual const Mappable* as_mappable(void) const { return this; }
    public:
      void initialize_point(SliceTask *owner, const DomainPoint &point,
                            const FutureMap &point_arguments, bool eager,
                            const std::vector<FutureMap> &point_futures);
    public:
      // From MemoizableOp
      virtual void complete_replay(ApEvent completion_event);
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const;
    public:
      // For collective instance creation
      virtual CollectiveManager* find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point);
      virtual bool finalize_collective_instance(MappingCallKind mapper_call,
                                                unsigned index, bool success);
      virtual void report_total_collective_instance_calls(MappingCallKind call,
                                                          unsigned total_calls);
    public: 
      bool has_remaining_inlining_dependences(
            std::map<PointTask*,unsigned> &remaining,
            std::map<RtEvent,std::vector<PointTask*> > &event_deps) const;
    protected:
      friend class SliceTask;
      PointTask                   *orig_task;
      SliceTask                   *slice_owner;
    protected:
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
    };

    /**
     * \class ShardTask
     * A shard task is copy of a single task that is used for
     * executing a single copy of a control replicated task.
     * It implements the functionality of a single task so that 
     * we can use it mostly transparently for the execution of 
     * a single shard.
     */
    class ShardTask : public SingleTask {
    public:
      ShardTask(Runtime *rt, ShardManager *manager, 
                ShardID shard_id, Processor target);
      ShardTask(const ShardTask &rhs);
      virtual ~ShardTask(void);
    public:
      ShardTask& operator=(const ShardTask &rhs);
    public:
      virtual void activate(void); 
      virtual void deactivate(void);
      virtual Domain get_slice_domain(void) const;
      virtual ShardID get_shard_id(void) const { return shard_id; }
      virtual size_t get_total_shards(void) const;
      virtual DomainPoint get_shard_point(void) const;
      virtual Domain get_shard_domain(void) const;
      virtual SingleTask* get_origin_task(void) const 
        { assert(false); return NULL; }
      virtual bool is_shard_task(void) const { return true; }
      virtual bool is_top_level_task(void) const; 
      // Set this to true so we always eagerly evaluate future functors
      // at the end of a task to get an actual future instance to pass back
      virtual bool is_reducing_future(void) const { return true; }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void resolve_false(bool speculated, bool launched);
      virtual bool distribute_task(void);
      virtual RtEvent perform_must_epoch_version_analysis(MustEpochOp *own);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL);
      virtual void handle_future_size(size_t return_type_size,
                                      bool has_return_type_size,
                                      std::set<RtEvent> &applied_events);
      virtual bool is_stealable(void) const;
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual std::map<PhysicalManager*,unsigned>*
                                       get_acquired_instances_ref(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      // Override these methods from operation class
      virtual void trigger_mapping(void); 
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid);
      virtual bool pack_task(Serializer &rez, AddressSpaceID target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events); 
      virtual void perform_inlining(VariantImpl *variant,
              const std::deque<InstanceSet> &parent_regions);
    public:
      virtual void handle_future(FutureInstance *instance,
                                 void *metadata, size_t metasize,
                                 FutureFunctor *functor,
                                 Processor future_proc,
                                 bool own_functor); 
      virtual void handle_post_mapped(bool deferral,
                          RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual void handle_misspeculation(void);
    protected:
      virtual InnerContext* initialize_inner_execution_context(VariantImpl *v,
                                                            bool inline_task);
      virtual EquivalenceSet* create_initial_equivalence_set(unsigned idx);
    public:
      virtual InnerContext* create_implicit_context(void);
    public:
      void launch_shard(void);
      void extract_event_preconditions(const std::deque<InstanceSet> &insts);
      void return_resources(ResourceTracker *target,
                            std::set<RtEvent> &preconditions);
      void report_leaks_and_duplicates(std::set<RtEvent> &preconditions);
      void handle_collective_message(Deserializer &derez);
      void handle_future_map_request(Deserializer &derez);
      void handle_disjoint_complete_request(Deserializer &derez);
      void handle_intra_space_dependence(Deserializer &derez);
      void handle_resource_update(Deserializer &derez,
                                  std::set<RtEvent> &applied);
      void handle_created_region_contexts(Deserializer &derez,
                                          std::set<RtEvent> &applied);
      void handle_trace_update(Deserializer &derez, AddressSpaceID source);
      ApBarrier handle_find_trace_shard_event(size_t temp_index, ApEvent event,
                                              ShardID remote_shard);
      ApBarrier handle_find_trace_shard_frontier(size_t temp_index, ApEvent event,
                                                 ShardID remote_shard);
      ReplicateContext* get_shard_execution_context(void) const;
    public:
      void initialize_implicit_task(InnerContext *context, TaskID tid,
                                    MapperID mid, Processor proxy);
      void complete_startup_initialization(void);
    public:
      const ShardID shard_id;
    protected:
      UniqueID remote_owner_uid;
      RtBarrier shard_barrier;
      bool all_shards_complete;
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
    class IndexTask : public CollectiveInstanceCreator<MultiTask>,
                      public LegionHeapify<IndexTask> {
    private:
      struct OutputRegionTagCreator {
      public:
        OutputRegionTagCreator(TypeTag *_type_tag, int _color_ndim)
          : type_tag(_type_tag), color_ndim(_color_ndim) { }
        template<typename DIM, typename COLOR_T>
        static inline void demux(OutputRegionTagCreator *creator)
        {
          switch (DIM::N + creator->color_ndim)
          {
#define DIMFUNC(DIM)                                             \
            case DIM:                                            \
              {                                                  \
                *creator->type_tag =                             \
                  NT_TemplateHelper::encode_tag<DIM, COLOR_T>(); \
                break;                                           \
              }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              assert(false);
          }
        }
      private:
        TypeTag *type_tag;
        int color_ndim;
      };
    public:
      static const AllocationType alloc_type = INDEX_TASK_ALLOC;
    public:
      IndexTask(Runtime *rt);
      IndexTask(const IndexTask &rhs);
      virtual ~IndexTask(void);
    public:
      IndexTask& operator=(const IndexTask &rhs);
    public:
      FutureMap initialize_task(InnerContext *ctx,
                                const IndexTaskLauncher &launcher,
                                IndexSpace launch_space,
                                Provenance *provenance,
                                bool track = true,
                                std::vector<OutputRequirement> *outputs = NULL);
      Future initialize_task(InnerContext *ctx,
                             const IndexTaskLauncher &launcher,
                             IndexSpace launch_space,
                             Provenance *provenance,
                             ReductionOpID redop,
                             bool deterministic,
                             bool track = true,
                             std::vector<OutputRequirement> *outputs = NULL);
      void initialize_predicate(const Future &pred_future,
                                const UntypedBuffer &pred_arg);
      void perform_base_dependence_analysis(void);
    protected:
      void create_output_regions(std::vector<OutputRequirement> &outputs,
                                 IndexSpace launch_space);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    protected:
      void activate_index_task(void);
      void deactivate_index_task(void);
    public:
      virtual void prepare_map_must_epoch(void);
    protected:
      typedef std::map<DomainPoint,DomainPoint> SizeMap;
      Domain compute_global_output_ranges(IndexSpaceNode *parent,
                                          IndexPartNode *part,
                                          const SizeMap& output_sizes,
                                          const SizeMap& local_sizes);
      void validate_output_sizes(unsigned index,
                                 const OutputRequirement& output_requirement,
                                 const SizeMap& output_sizes) const;
      virtual void finalize_output_regions(void);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual void trigger_ready(void);
      virtual void resolve_false(bool speculated, bool launched);
      virtual void premap_task(void);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL);
      virtual void launch_task(bool inline_task = false);
      virtual bool is_stealable(void) const;
      virtual void map_and_launch(void);
    public:
      virtual TaskKind get_task_kind(void) const;
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions);
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
                  Processor p, bool recurse, bool stealable);
    public:
      virtual void reduce_future(const DomainPoint &point, 
                                 FutureInstance *instance);
    public:
      virtual void pack_profiling_requests(Serializer &rez,
                                           std::set<RtEvent> &applied) const;
      virtual int add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &respone,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
    public:
      virtual void register_must_epoch(void);
    public:
      // Make this a virtual method so for control replication we can 
      // create a different type of future map for the task
      virtual FutureMapImpl* create_future_map(TaskContext *ctx,
                    IndexSpace launch_space, IndexSpace shard_space);
      // Also virtual for control replication override
      virtual void initialize_concurrent_analysis(bool replay);
      virtual RtEvent verify_concurrent_execution(const DomainPoint &point,
                                                  Processor target);
    public:
      // Methods for supporting intra-index-space mapping dependences
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped);
    public:
      void record_origin_mapped_slice(SliceTask *local_slice);
    protected:
      // Virtual so can be overridden by ReplIndexTask
      virtual void create_future_instances(std::vector<Memory> &target_mems);
      // Callback for control replication to perform reduction for sizes
      // and provide an event for when the result is ready
      virtual void finish_index_task_reduction(void);
      virtual RtEvent finish_index_task_complete(void);
    public:
      void return_slice_mapped(unsigned points, RtEvent applied_condition,
                               ApEvent slice_complete);
      void return_slice_complete(unsigned points, RtEvent applied_condition,
                             const std::map<unsigned,SizeMap> &output_sizes,
                             void *metadata = NULL, size_t metasize = 0);
      void return_slice_commit(unsigned points, RtEvent applied_condition);
    public:
      void unpack_slice_mapped(Deserializer &derez, AddressSpaceID source);
      void unpack_slice_complete(Deserializer &derez);
      void unpack_slice_commit(Deserializer &derez);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From CollectiveInstanceCreator
      virtual IndexSpaceNode *get_collective_space(void) const
        { return launch_space; }
    public:
      void enumerate_futures(const Domain &domain);
    public:
      static void process_slice_mapped(Deserializer &derez,
                                       AddressSpaceID source);
      static void process_slice_complete(Deserializer &derez);
      static void process_slice_commit(Deserializer &derez);
      static void process_slice_find_intra_dependence(Deserializer &derez);
      static void process_slice_record_intra_dependence(Deserializer &derez);
    protected:
      friend class SliceTask;
      Future reduction_future;
      unsigned total_points;
      unsigned mapped_points;
      unsigned complete_points;
      unsigned committed_points;
    protected:
      std::vector<RegionTreePath> privilege_paths;
      std::set<SliceTask*> origin_mapped_slices;
      std::vector<FutureInstance*> reduction_instances;
      std::vector<ApUserEvent> reduction_instances_ready;
      std::vector<Memory> serdez_redop_targets;
    protected:
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> complete_effects;
      std::set<RtEvent> complete_preconditions;
      std::set<RtEvent> commit_preconditions;
    protected:
      std::map<DomainPoint,RtUserEvent> pending_intra_space_dependences;
    protected:
      // Profiling information
      struct IndexProfilingInfo : public Mapping::Mapper::TaskProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>      task_profiling_requests;
      std::vector<ProfilingMeasurementID>      copy_profiling_requests;
      std::vector<IndexProfilingInfo>                   profiling_info;
      RtUserEvent                                   profiling_reported;
      int                                           profiling_priority;
      int                                           copy_fill_priority;
      std::atomic<int>                  outstanding_profiling_requests;
      std::atomic<int>                  outstanding_profiling_reported;
    protected:
      // Whether we have to do intra-task alias analysis
      bool need_intra_task_alias_analysis;
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned,unsigned> > interfering_requirements;
      std::map<DomainPoint,std::vector<LogicalRegion> > point_requirements;
#ifdef DEBUG_LEGION
    public:
      void check_point_requirements(
          const std::map<DomainPoint,std::vector<LogicalRegion> > &point_reqs);
#endif
    protected:
      // Sizes of subspaces for globally indexed output regions
      std::map<unsigned, SizeMap> all_output_sizes;
    };

    /**
     * \class SliceTask
     * A slice task is a (possibly whole) fraction of an index
     * space task launch.  Once slice task object is made for
     * each slice created by the mapper when (possibly recursively)
     * slicing up the domain of the index space task launch.
     */
    class SliceTask : public MultiTask, public ResourceTracker,
                      public LegionHeapify<SliceTask> {
    public:
      static const AllocationType alloc_type = SLICE_TASK_ALLOC;
    public:
      enum CollectiveInstMessage {
        SLICE_COLLECTIVE_FIND_OR_CREATE,
        SLICE_COLLECTIVE_FINALIZE,
        SLICE_COLLECTIVE_REPORT,
      };
    public:
      SliceTask(Runtime *rt);
      SliceTask(const SliceTask &rhs);
      virtual ~SliceTask(void);
    public:
      SliceTask& operator=(const SliceTask &rhs);
    public:
      inline UniqueID get_remote_owner_uid(void) const 
        { return remote_owner_uid; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_dependence_analysis(void);
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual void premap_task(void);
      virtual bool distribute_task(void);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL,
                                      const DeferMappingArgs *args = NULL);
      virtual void launch_task(bool inline_task = false);
      virtual bool is_stealable(void) const;
      virtual void map_and_launch(void);
      virtual bool is_output_global(unsigned idx) const;
      virtual bool is_output_valid(unsigned idx) const;
    public:
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual bool pack_task(Serializer &rez, AddressSpaceID target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(VariantImpl *variant,
                    const std::deque<InstanceSet> &parent_regions);
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
                  Processor p, bool recurse, bool stealable);
      virtual void reduce_future(const DomainPoint &point,
                                 FutureInstance *instance);
      void handle_future(const DomainPoint &point,
                         FutureInstance *instance, void *metadata, 
                         size_t metasize, FutureFunctor *functor,
                         Processor future_proc, bool own_functor); 
    public:
      virtual void register_must_epoch(void);
      PointTask* clone_as_point_task(const DomainPoint &point,
                                     bool inline_task);
      size_t enumerate_points(bool inline_task);
      FutureInstance* get_predicate_false_result(Processor point_proc,
                              const void *&metadata, size_t &metasize);
    public:
      void check_target_processors(void) const;
      void update_target_processor(void);
      void expand_replay_slices(std::list<SliceTask*> &slices);
      void find_commit_preconditions(std::set<RtEvent> &preconditions);
    protected:
      virtual void trigger_complete(void);
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      void return_privileges(TaskContext *point_context,
                             std::set<RtEvent> &preconditions);
      void record_point_mapped(RtEvent child_mapped, ApEvent child_complete,
          std::map<PhysicalManager*,unsigned> &child_acquired);
      void record_point_complete(RtEvent child_complete);
      void record_point_committed(RtEvent commit_precondition =
                                  RtEvent::NO_RT_EVENT);
    public:
      void handle_future_size(size_t future_size, const DomainPoint &p,
                              std::set<RtEvent> &applied_conditions);
      void record_output_sizes(const DomainPoint &point,
                               const std::vector<OutputRegion> &output_regions);
      RtEvent verify_concurrent_execution(const DomainPoint &point,
                                          Processor target);
    protected:
      void trigger_slice_mapped(void);
      void trigger_slice_complete(void);
      void trigger_slice_commit(void);
    protected:
      void pack_remote_mapped(Serializer &rez, RtEvent applied_condition,
                              ApEvent all_points_complete);
      void pack_remote_complete(Serializer &rez, RtEvent applied_condition);
      void pack_remote_commit(Serializer &rez, RtEvent applied_condition);
    public:
      static void handle_slice_return(Runtime *rt, Deserializer &derez);
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
      // From MemoizableOp
      virtual void trigger_replay(void);
      virtual void complete_replay(ApEvent instance_ready_event);
    public:
      // Methods for supporting intra-index-space mapping dependences
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 const DomainPoint &next,
                                                 RtEvent point_mapped);
    public:
      // For collective instance creation
      virtual CollectiveManager* find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point);
      virtual bool finalize_collective_instance(MappingCallKind mapper_call,
                                                unsigned index, bool success);
      virtual void report_total_collective_instance_calls(MappingCallKind call,
                                                          unsigned total_calls);
      static void handle_collective_instance_request(Deserializer &derez,
                                       AddressSpaceID source, Runtime *rutime);
      static void handle_collective_instance_response(Deserializer &derez,
                                                      Runtime *runtime);
      static void handle_verify_concurrent_execution(Deserializer &derez);
    protected:
      friend class IndexTask;
      friend class PointTask;
      friend class ReplMustEpochOp;
      std::vector<PointTask*> points;
    protected:
      unsigned num_unmapped_points;
      unsigned num_uncomplete_points;
      unsigned num_uncommitted_points;
    protected:
      IndexTask *index_owner;
      UniqueID remote_unique_id;
      bool origin_mapped;
      UniqueID remote_owner_uid;
      DomainPoint reduction_instance_point;
      // An event for tracking when origin-mapped slices on the owner
      // node have committed so we can trigger things appropriately
      RtUserEvent origin_mapped_complete;
    protected:
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> point_completions;
      std::set<RtEvent> complete_preconditions;
      std::set<RtEvent> commit_preconditions;
    protected:
      std::set<std::pair<DomainPoint,DomainPoint> > unique_intra_space_deps;
    protected:
      // Sizes of subspaces for globally indexed output regions
      std::map<unsigned,std::map<DomainPoint,DomainPoint> > all_output_sizes;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TASKS_H__
