/* Copyright 2018 Stanford University, NVIDIA Corporation
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
     * \class ResourceTracker 
     * A helper class for tracking which privileges an 
     * operation owns. This is inherited by multi-tasks
     * for aggregating the privilege results of their
     * children as well as task contexts for tracking
     * which privileges have been accrued or deleted
     * as part of the execution of the task.
     */
    class ResourceTracker {
    public:
      ResourceTracker(void);
      ResourceTracker(const ResourceTracker &rhs);
      virtual ~ResourceTracker(void);
    public:
      ResourceTracker& operator=(const ResourceTracker &rhs);
    public:
      virtual void register_region_creations(
                     const std::map<LogicalRegion,bool> &regions) = 0;
      virtual void register_region_deletions(
                          const std::set<LogicalRegion> &regions) = 0;
    public:
      virtual void register_field_creations(
            const std::map<std::pair<FieldSpace,FieldID>,bool> &fields) = 0;
      virtual void register_field_deletions(
                const std::set<std::pair<FieldSpace,FieldID> > &fields) = 0;
    public:
      virtual void register_field_space_creations(
                          const std::set<FieldSpace> &spaces) = 0;
      virtual void register_field_space_deletions(
                          const std::set<FieldSpace> &spaces) = 0;
    public:
      virtual void register_index_space_creations(
                          const std::set<IndexSpace> &spaces) = 0;
      virtual void register_index_space_deletions(
                          const std::set<IndexSpace> &spaces) = 0;
    public:
      virtual void register_index_partition_creations(
                          const std::set<IndexPartition> &parts) = 0;
      virtual void register_index_partition_deletions(
                          const std::set<IndexPartition> &parts) = 0;
    public:
      void return_privilege_state(ResourceTracker *target) const;
      void pack_privilege_state(Serializer &rez, 
                                AddressSpaceID target, bool returning) const;
      static void unpack_privilege_state(Deserializer &derez,
                                         ResourceTracker *target);
    protected:
      std::map<LogicalRegion,
               bool/*local*/>                   created_regions;
      std::map<std::pair<FieldSpace,FieldID>,
               bool/*local*/>                   created_fields;
      std::set<FieldSpace>                      created_field_spaces;
      std::set<IndexSpace>                      created_index_spaces;
      std::set<IndexPartition>                  created_index_partitions;
      std::set<LogicalRegion>                   deleted_regions;
      std::set<std::pair<FieldSpace,FieldID> >  deleted_fields;
      std::set<FieldSpace>                      deleted_field_spaces;
      std::set<IndexSpace>                      deleted_index_spaces;
      std::set<IndexPartition>                  deleted_index_partitions;
    };

    /**
     * \class ExternalTask
     * An extentions of the external Task with some
     * methods that help us internally for packing
     * and unpacking them.
     */
    class ExternalTask : public Task {
    public:
      ExternalTask(void);
    public:
      void pack_external_task(Serializer &rez, AddressSpaceID target);
      void unpack_external_task(Deserializer &derez, Runtime *runtime,
                                ReferenceMutator *mutator);
    public:
      virtual void set_context_index(unsigned index) = 0;
    protected:
      AllocManager *arg_manager;
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
      struct DeferDistributeArgs : public LgTaskArgs<DeferDistributeArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_DISTRIBUTE_TASK_ID;
      public:
        DeferDistributeArgs(TaskOp *op)
          : LgTaskArgs<DeferDistributeArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        TaskOp *const proxy_this;
      };
      struct DeferMappingArgs : public LgTaskArgs<DeferMappingArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_PERFORM_MAPPING_TASK_ID;
      public:
        DeferMappingArgs(TaskOp *op, MustEpochOp *owner)
          : LgTaskArgs<DeferMappingArgs>(op->get_unique_op_id()),
            proxy_this(op), must_op(owner) { }
      public:
        TaskOp *const proxy_this;
        MustEpochOp *const must_op;
      };
      struct DeferLaunchArgs : public LgTaskArgs<DeferLaunchArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_LAUNCH_TASK_ID;
      public:
        DeferLaunchArgs(TaskOp *op)
          : LgTaskArgs<DeferLaunchArgs>(op->get_unique_op_id()), 
            proxy_this(op) { }
      public:
        TaskOp *const proxy_this;
      };
      struct DeferredFutureSetArgs : public LgTaskArgs<DeferredFutureSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_FUTURE_SET_ID;
      public:
        DeferredFutureSetArgs(FutureImpl *tar, FutureImpl *res, TaskOp *t)
          : LgTaskArgs<DeferredFutureSetArgs>(t->get_unique_op_id()),
            target(tar), result(res), task_op(t) { }
      public:
        FutureImpl *const target;
        FutureImpl *const result;
        TaskOp *const task_op;
      };
      struct DeferredFutureMapSetArgs : 
        public LgTaskArgs<DeferredFutureMapSetArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_FUTURE_MAP_SET_ID;
      public:
        DeferredFutureMapSetArgs(FutureMapImpl *map, FutureImpl *res,
                                 Domain d, TaskOp *t)
          : LgTaskArgs<DeferredFutureMapSetArgs>(t->get_unique_op_id()),
            future_map(map), result(res), domain(d), task_op(t) { }
      public:
        FutureMapImpl *const future_map;
        FutureImpl *const result;
        const Domain domain;
        TaskOp *const task_op;
      };
      struct DeferredEnqueueArgs : public LgTaskArgs<DeferredEnqueueArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_ENQUEUE_TASK_ID;
      public:
        DeferredEnqueueArgs(ProcessorManager *man, TaskOp *t)
          : LgTaskArgs<DeferredEnqueueArgs>(t->get_unique_op_id()),
            manager(man), task(t) { }
      public:
        ProcessorManager *const manager;
        TaskOp *const task;
      };
    public:
      TaskOp(Runtime *rt);
      virtual ~TaskOp(void);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual void set_context_index(unsigned index);
      virtual int get_depth(void) const;
      virtual const char* get_task_name(void) const;
    public:
      bool is_remote(void) const;
      inline bool is_stolen(void) const { return (steal_count > 0); }
      inline bool is_origin_mapped(void) const { return map_origin; }
    public:
      void set_current_proc(Processor current);
      inline void set_origin_mapped(bool origin) { map_origin = origin; }
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
      void unpack_base_external_task(Deserializer &derez,
                                     ReferenceMutator *mutator); 
    public:
      void mark_stolen(void);
      void initialize_base_task(TaskContext *ctx, bool track, 
            const std::vector<StaticDependence> *dependences,
            const Predicate &p, Processor::TaskFuncID tid);
      void check_empty_field_requirements(void);
      size_t check_future_size(FutureImpl *impl);
    public:
      bool select_task_options(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void) = 0;
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched) = 0;
    public:
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      virtual ApEvent get_restrict_precondition(void) const;
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual unsigned find_parent_index(unsigned idx);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual RestrictInfo& get_restrict_info(unsigned idx);
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual const std::vector<RestrictInfo>* get_restrict_infos(void);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual void early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL) = 0;
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
      virtual void perform_inlining(void) = 0;
    public:
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
    public:
      RtEvent defer_distribute_task(RtEvent precondition);
      RtEvent defer_perform_mapping(RtEvent precondition, MustEpochOp *op);
      RtEvent defer_launch_task(RtEvent precondition);
    protected:
      void enqueue_ready_task(bool use_target_processor,
                              RtEvent wait_on = RtEvent::NO_RT_EVENT);
    protected:
      void pack_version_infos(Serializer &rez,
                              std::vector<VersionInfo> &infos,
                              const std::vector<bool> &full_version_info);
      void unpack_version_infos(Deserializer &derez,
                                std::vector<VersionInfo> &infos,
                                std::set<RtEvent> &ready_events);
    protected:
      void pack_restrict_infos(Serializer &rez, 
                               std::vector<RestrictInfo> &infos);
      void unpack_restrict_infos(Deserializer &derez,
                                 std::vector<RestrictInfo> &infos,
                                 std::set<RtEvent> &ready_events);
      void pack_projection_infos(Serializer &rez,
                                 std::vector<ProjectionInfo> &infos);
      void unpack_projection_infos(Deserializer &derez,
                                   std::vector<ProjectionInfo> &infos,
                                   IndexSpace launch_space);
    public:
      // Tell the parent context that this task is in a ready queue
      void activate_outstanding_task(void);
      void deactivate_outstanding_task(void); 
    public:
      void perform_privilege_checks(void);
    public:
      void find_early_mapped_region(unsigned idx, InstanceSet &ref);
      void clone_task_op_from(TaskOp *rhs, Processor p, 
                              bool stealable, bool duplicate_args);
      void update_grants(const std::vector<Grant> &grants);
      void update_arrival_barriers(const std::vector<PhaseBarrier> &barriers);
      void compute_point_region_requirements(void);
      void complete_point_projection(void);
      void early_map_regions(std::set<RtEvent> &applied_conditions,
                             const std::vector<unsigned> &must_premap);
      bool prepare_steal(void);
    public:
      void compute_parent_indexes(void);
      void perform_intra_task_alias_analysis(bool is_tracing,
          LegionTrace *trace, std::vector<RegionTreePath> &privilege_paths);
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
      // Early mapped regions
      std::map<unsigned/*idx*/,InstanceSet>     early_mapped_regions;
      // A map of any locks that we need to take for this task
      std::map<Reservation,bool/*exclusive*/>   atomic_locks;
    protected:
      std::vector<unsigned>                     parent_req_indexes; 
    protected:
      bool complete_received;
      bool commit_received;
    protected:
      bool options_selected;
      bool memoize_selected;
      bool map_origin;
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
      struct DeferredPostMappedArgs : 
        public LgTaskArgs<DeferredPostMappedArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_POST_MAPPED_ID;
      public:
        DeferredPostMappedArgs(SingleTask *t)
          : LgTaskArgs<DeferredPostMappedArgs>(t->get_unique_op_id()),
            task(t) { }
      public:
        SingleTask *const task;
      };
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
      bool has_virtual_instances(void) const;
      bool is_created_region(unsigned index) const;
      void update_no_access_regions(void);
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
    public:
      void initialize_map_task_input(Mapper::MapTaskInput &input,
                                     Mapper::MapTaskOutput &output,
                                     MustEpochOp *must_epoch_owner,
                                     std::vector<InstanceSet> &valid_instances);
      void finalize_map_task_output(Mapper::MapTaskInput &input,
                                    Mapper::MapTaskOutput &output,
                                    MustEpochOp *must_epoch_owner,
                                    std::vector<InstanceSet> &valid_instances); 
      void replay_map_task_output();
    protected: // mapper helper calls
      void validate_target_processors(const std::vector<Processor> &prcs) const;
      void validate_variant_selection(MapperManager *local_mapper,
                    VariantImpl *impl, const char *call_name) const;
    protected:
      void invoke_mapper(MustEpochOp *must_epoch_owner);
      void map_all_regions(ApEvent user_event,
                           MustEpochOp *must_epoch_owner = NULL); 
      void perform_post_mapping(void);
    protected:
      void pack_single_task(Serializer &rez, AddressSpaceID target);
      void unpack_single_task(Deserializer &derez, 
                              std::set<RtEvent> &ready_events);
    public:
      virtual void add_copy_profiling_request(
                                      Realm::ProfilingRequestSet &requests);
      virtual void handle_profiling_response(
                                const Realm::ProfilingResponse &respone);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual bool is_top_level_task(void) const { return false; }
    public:
      virtual void resolve_false(bool speculated, bool launched) = 0;
      virtual void launch_task(void);
      virtual void early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_must_epoch_version_analysis(MustEpochOp *own) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle) = 0;
      virtual bool can_early_complete(ApUserEvent &chain_event) = 0; 
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst) = 0;
    public:
      // Override these methods from operation class
      virtual void trigger_mapping(void); 
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid) = 0;
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0; 
      virtual void perform_inlining(void) = 0;
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned) = 0; 
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT) = 0;
      virtual void handle_misspeculation(void) = 0;
    public:
      // From Memoizable
      virtual void complete_replay(ApEvent completion_event);
    protected:
      // Boolean for each region saying if it is virtual mapped
      std::vector<bool> virtual_mapped;
      // Regions which are NO_ACCESS or have no privilege fields
      std::vector<bool> no_access_regions;
    protected:
      std::vector<Processor> target_processors;
      // Hold the result of the mapping 
      std::deque<InstanceSet> physical_instances;
    protected: // Mapper choices 
      VariantID                             selected_variant;
      TaskPriority                          task_priority;
      bool                                  perform_postmap;
    protected:
      // Events that must be triggered before we are done mapping
      std::set<RtEvent> map_applied_conditions;
    protected:
      TaskContext*                          execution_context;
    protected:
      mutable bool leaf_cached, is_leaf_result;
      mutable bool inner_cached, is_inner_result;
      mutable bool has_virtual_instances_cached, has_virtual_instances_result;
    protected:
      // Profiling information
      std::vector<ProfilingMeasurementID> task_profiling_requests;
      std::vector<ProfilingMeasurementID> copy_profiling_requests;
      int                                      profiling_priority;
      int                          outstanding_profiling_requests;
      RtUserEvent                              profiling_reported;
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
    public:
      virtual void trigger_dependence_analysis(void) = 0;
    public:
      virtual void resolve_false(bool speculated, bool launched) = 0;
      virtual void early_map_task(void) = 0;
      virtual bool distribute_task(void) = 0;
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL) = 0;
      virtual void launch_task(void) = 0;
      virtual bool is_stealable(void) const = 0;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle) = 0;
      virtual void map_and_launch(void) = 0;
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual RestrictInfo& get_restrict_info(unsigned idx);
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual const std::vector<RestrictInfo>* get_restrict_infos(void);
    public:
      virtual ApEvent get_task_completion(void) const = 0;
      virtual TaskKind get_task_kind(void) const = 0;
    public:
      virtual void trigger_mapping(void);
    protected:
      virtual void trigger_task_complete(void) = 0;
      virtual void trigger_task_commit(void) = 0;
    public:
      virtual bool pack_task(Serializer &rez, Processor target) = 0;
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events) = 0;
      virtual void perform_inlining(void) = 0;
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator) = 0;
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner) = 0;
      virtual void register_must_epoch(void) = 0;
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
      std::vector<ProjectionInfo> projection_infos;
      bool sliced;
    protected:
      IndexSpace launch_space; // global set of points
      IndexSpace internal_space; // local set of points
      ReductionOpID redop;
      const ReductionOp *reduction_op;
      FutureMap point_arguments;
      // For handling reductions of types with serdez methods
      const SerdezRedopFns *serdez_redop_fns;
      size_t reduction_state_size;
      void *reduction_state; 
    protected:
      bool children_complete_invoked;
      bool children_commit_invoked;
    protected:
      Future predicate_false_future;
      void *predicate_false_result;
      size_t predicate_false_size;
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
    public:
      Future initialize_task(TaskContext *ctx,
                             const TaskLauncher &launcher, 
                             bool check_privileges,
                             bool track = true);
      void set_top_level(void);
    public:
      RtEvent perform_versioning_analysis(void);
      virtual RtEvent perform_must_epoch_version_analysis(MustEpochOp *own);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
      virtual void record_restrict_postcondition(ApEvent postcondition);
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual void early_map_task(void);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual RestrictInfo& get_restrict_info(unsigned idx);
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual const std::vector<RestrictInfo>* get_restrict_infos(void);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual void send_remote_context(AddressSpaceID target, 
                                       RemoteTask *dst);
    public:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual void handle_misspeculation(void);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual void perform_physical_traversal(unsigned idx,
                                RegionTreeContext ctx, InstanceSet &valid);
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(void);
      virtual bool is_top_level_task(void) const { return top_level_task; }
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
    protected:
      void pack_remote_complete(Serializer &rez);
      void pack_remote_commit(Serializer &rez);
      void unpack_remote_mapped(Deserializer &derez);
      void unpack_remote_complete(Deserializer &derez);
      void unpack_remote_commit(Deserializer &derez);
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      static void process_unpack_remote_mapped(Deserializer &derez);
      static void process_unpack_remote_complete(Deserializer &derez);
      static void process_unpack_remote_commit(Deserializer &derez);
    protected: 
      void *future_store;
      size_t future_size;
      Future result; 
      std::set<Operation*>        child_operations;
      std::vector<RegionTreePath> privilege_paths;
      std::vector<VersionInfo>    version_infos;
      std::vector<RestrictInfo>   restrict_infos;
    protected:
      // Information for remotely executing task
      IndividualTask *orig_task; // Not a valid pointer when remote
      ApEvent remote_completion_event;
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
      // Whether we have to do intra-task alias analysis
      bool need_intra_task_alias_analysis;
    protected:
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
    protected:
      std::map<PhysicalManager*,
        std::pair<unsigned/*ref count*/,bool/*created*/> > acquired_instances;
      std::set<ApEvent> restrict_postconditions;
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
    public:
      void perform_versioning_analysis(std::set<RtEvent> &ready_events);
      virtual RtEvent perform_must_epoch_version_analysis(MustEpochOp *own);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual void early_map_task(void);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual bool can_early_complete(ApUserEvent &chain_event);
      virtual VersionInfo& get_version_info(unsigned idx);
      virtual RestrictInfo& get_restrict_info(unsigned idx);
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
      virtual const std::vector<VersionInfo>* get_version_infos(void);
      virtual const std::vector<RestrictInfo>* get_restrict_infos(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
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
      virtual void perform_inlining(void);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
      virtual void record_restrict_postcondition(ApEvent postcondition);
    public:
      virtual void handle_future(const void *res, 
                                 size_t res_size, bool owned);
      virtual void handle_post_mapped(RtEvent pre = RtEvent::NO_RT_EVENT);
      virtual void handle_misspeculation(void);
    public:
      // ProjectionPoint methods
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
    public:
      void initialize_point(SliceTask *owner, const DomainPoint &point,
                            const FutureMap &point_arguments);
      void send_back_created_state(AddressSpaceID target);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id() const;
    protected:
      friend class SliceTask;
      SliceTask                   *slice_owner;
      ApUserEvent                 point_termination;
      std::set<ApEvent>           restrict_postconditions;
    protected:
      std::map<AddressSpaceID,RemoteTask*> remote_instances;
    protected:
      std::vector<VersionInfo>    version_infos;
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
    class IndexTask : public MultiTask,
                      public LegionHeapify<IndexTask> {
    public:
      static const AllocationType alloc_type = INDEX_TASK_ALLOC;
    public:
      IndexTask(Runtime *rt);
      IndexTask(const IndexTask &rhs);
      virtual ~IndexTask(void);
    public:
      IndexTask& operator=(const IndexTask &rhs);
    public:
      FutureMap initialize_task(TaskContext *ctx,
                                const IndexTaskLauncher &launcher,
                                IndexSpace launch_space,
                                bool check_privileges,
                                bool track = true);
      Future initialize_task(TaskContext *ctx,
                             const IndexTaskLauncher &launcher,
                             IndexSpace launch_space,
                             ReductionOpID redop,
                             bool check_privileges,
                             bool track = true);
      void initialize_predicate(const Future &pred_future,
                                const TaskArgument &pred_arg);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual RegionTreePath& get_privilege_path(unsigned idx);
    public:
      virtual void resolve_false(bool speculated, bool launched);
      virtual void early_map_task(void);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual void map_and_launch(void);
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
      virtual void perform_inlining(void);
      virtual void end_inline_task(const void *result, 
                                   size_t result_size, bool owned);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
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
      void record_origin_mapped_slice(SliceTask *local_slice);
    public:
      void return_slice_mapped(unsigned points, long long denom,
                               RtEvent applied_condition, 
                               ApEvent restrict_postcondition);
      void return_slice_complete(unsigned points,
                                 ApEvent slice_postcondition);
      void return_slice_commit(unsigned points);
    public:
      void unpack_slice_mapped(Deserializer &derez, AddressSpaceID source);
      void unpack_slice_complete(Deserializer &derez);
      void unpack_slice_commit(Deserializer &derez); 
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      static void process_slice_mapped(Deserializer &derez, 
                                       AddressSpaceID source);
      static void process_slice_complete(Deserializer &derez);
      static void process_slice_commit(Deserializer &derez);
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
    protected:
      std::vector<RegionTreePath> privilege_paths;
      std::deque<SliceTask*> origin_mapped_slices;
    protected:
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> completion_preconditions;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
    protected:
      // Whether we have to do intra-task alias analysis
      bool need_intra_task_alias_analysis;
#ifdef DEBUG_LEGION
    protected:
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned,unsigned> > interfering_requirements;
    public:
      void check_point_requirements(
          const std::map<DomainPoint,std::vector<LogicalRegion> > &point_reqs);
#endif
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
      struct DeferMapAndLaunchArgs : public LgTaskArgs<DeferMapAndLaunchArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_MAP_AND_LAUNCH_TASK_ID;
      public:
        DeferMapAndLaunchArgs(SliceTask *t)
          : LgTaskArgs<DeferMapAndLaunchArgs>(t->get_unique_op_id()),
            proxy_this(t) { }
      public:
        SliceTask *proxy_this;
      };
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
      virtual void resolve_false(bool speculated, bool launched);
      virtual void early_map_task(void);
      virtual bool distribute_task(void);
      virtual RtEvent perform_mapping(MustEpochOp *owner = NULL);
      virtual void launch_task(void);
      virtual bool is_stealable(void) const;
      virtual bool has_restrictions(unsigned idx, LogicalRegion handle);
      virtual void map_and_launch(void);
    public:
      virtual ApEvent get_task_completion(void) const;
      virtual TaskKind get_task_kind(void) const;
    public:
      virtual bool pack_task(Serializer &rez, Processor target);
      virtual bool unpack_task(Deserializer &derez, Processor current,
                               std::set<RtEvent> &ready_events);
      virtual void perform_inlining(void);
    public:
      virtual SliceTask* clone_as_slice_task(IndexSpace is,
          Processor p, bool recurse, bool stealable,
          long long scale_denominator);
      virtual void handle_future(const DomainPoint &point, const void *result,
                                 size_t result_size, bool owner);
    public:
      virtual void register_must_epoch(void);
      PointTask* clone_as_point_task(const DomainPoint &point);
      void enumerate_points(void);
      const void* get_predicate_false_result(size_t &result_size);
    public:
      RtEvent perform_versioning_analysis(void);
      RtEvent perform_must_epoch_version_analysis(MustEpochOp *owner);
      std::map<PhysicalManager*,std::pair<unsigned,bool> >* 
                                     get_acquired_instances_ref(void);
      void check_target_processors(void) const;
      void update_target_processor(void);
    protected:
      virtual void trigger_task_complete(void);
      virtual void trigger_task_commit(void);
    public:
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      void return_privileges(TaskContext *point_context);
      void record_child_mapped(RtEvent child_complete, 
                               ApEvent restrict_postcondition);
      void record_child_complete(void);
      void record_child_committed(RtEvent commit_precondition = 
                                  RtEvent::NO_RT_EVENT);
    protected:
      void trigger_slice_mapped(void);
      void trigger_slice_complete(void);
      void trigger_slice_commit(void);
    protected:
      void pack_remote_mapped(Serializer &rez, RtEvent applied_condition);
      void pack_remote_complete(Serializer &rez, ApEvent slice_postcondition); 
      void pack_remote_commit(Serializer &rez);
    public:
      RtEvent defer_map_and_launch(RtEvent precondition);
    public:
      static void handle_slice_return(Runtime *rt, Deserializer &derez);
    public: // Privilege tracker methods
      virtual void register_region_creations(
                     const std::map<LogicalRegion,bool> &regions);
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
      // From MemoizableOp
      virtual void replay_analysis(void);
    protected:
      friend class IndexTask;
      friend class PointTask;
      std::vector<PointTask*> points;
    protected:
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
      bool origin_mapped;
      bool need_versioning_analysis;
      UniqueID remote_owner_uid;
    protected:
      // Temporary storage for future results
      std::map<DomainPoint,std::pair<void*,size_t> > temporary_futures;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> restrict_postconditions;
      std::set<RtEvent> commit_preconditions;
    };

  }; // namespace Internal 
}; // namespace Legion

#endif // __LEGION_TASKS_H__
