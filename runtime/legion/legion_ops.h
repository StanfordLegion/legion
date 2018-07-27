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


#ifndef __LEGION_OPERATIONS_H__
#define __LEGION_OPERATIONS_H__

#include "legion.h"
#include "legion/runtime.h"
#include "legion/region_tree.h"
#include "legion/legion_mapping.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/legion_analysis.h"
#include "legion/mapper_manager.h"

namespace Legion {
  namespace Internal {

    // Special typedef for predicates
    typedef PredicateImpl PredicateOp; 

    /**
     * \class Operation
     * The operation class serves as the root of the tree
     * of all operations that can be performed in a Legion
     * program.
     */
    class Operation : public ReferenceMutator, public ProfilingResponseHandler {
    public:
      enum OpKind {
        MAP_OP_KIND,
        COPY_OP_KIND,
        FENCE_OP_KIND,
        FRAME_OP_KIND,
        DELETION_OP_KIND,
        OPEN_OP_KIND,
        ADVANCE_OP_KIND,
        INTER_CLOSE_OP_KIND,
        READ_CLOSE_OP_KIND,
        POST_CLOSE_OP_KIND,
        VIRTUAL_CLOSE_OP_KIND,
        RETURN_CLOSE_OP_KIND,
        ACQUIRE_OP_KIND,
        RELEASE_OP_KIND,
        DYNAMIC_COLLECTIVE_OP_KIND,
        FUTURE_PRED_OP_KIND,
        NOT_PRED_OP_KIND,
        AND_PRED_OP_KIND,
        OR_PRED_OP_KIND,
        MUST_EPOCH_OP_KIND,
        PENDING_PARTITION_OP_KIND,
        DEPENDENT_PARTITION_OP_KIND,
        FILL_OP_KIND,
        ATTACH_OP_KIND,
        DETACH_OP_KIND,
        TIMING_OP_KIND,
        TRACE_CAPTURE_OP_KIND,
        TRACE_COMPLETE_OP_KIND,
        TRACE_REPLAY_OP_KIND,
        TRACE_BEGIN_OP_KIND,
        TRACE_SUMMARY_OP_KIND,
        TASK_OP_KIND,
        LAST_OP_KIND,
      };
      static const char *const op_names[LAST_OP_KIND];
#define OPERATION_NAMES {           \
        "Mapping",                  \
        "Copy",                     \
        "Fence",                    \
        "Frame",                    \
        "Deletion",                 \
        "Open",                     \
        "Advance",                  \
        "Inter Close",              \
        "Read Close",               \
        "Post Close",               \
        "Virtual Close",            \
        "Return Close",             \
        "Acquire",                  \
        "Release",                  \
        "Dynamic Collective",       \
        "Future Predicate",         \
        "Not Predicate",            \
        "And Predicate",            \
        "Or Predicate",             \
        "Must Epoch",               \
        "Pending Partition",        \
        "Dependent Partition",      \
        "Fill",                     \
        "Attach",                   \
        "Detach",                   \
        "Timing",                   \
        "Trace Capture",            \
        "Trace Complete",           \
        "Trace Replay",             \
        "Trace Begin",              \
        "Trace Summary",            \
        "Task",                     \
      }
    public:
      struct TriggerOpArgs : public LgTaskArgs<TriggerOpArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_OP_ID;
      public:
        TriggerOpArgs(Operation *o)
          : LgTaskArgs<TriggerOpArgs>(o->get_unique_op_id()), op(o) { }
      public:
        Operation *const op;
      };
      struct DeferredReadyArgs : public LgTaskArgs<DeferredReadyArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_READY_TRIGGER_ID;
      public:
        DeferredReadyArgs(Operation *op)
          : LgTaskArgs<DeferredReadyArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct DeferredEnqueueArgs : public LgTaskArgs<DeferredEnqueueArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_ENQUEUE_OP_ID;
      public:
        DeferredEnqueueArgs(Operation *op, LgPriority p)
          : LgTaskArgs<DeferredEnqueueArgs>(op->get_unique_op_id()),
            proxy_this(op), priority(p) { }
      public:
        Operation *const proxy_this;
        const LgPriority priority;
      };
      struct DeferredResolutionArgs :
        public LgTaskArgs<DeferredResolutionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_RESOLUTION_TRIGGER_ID;
      public:
        DeferredResolutionArgs(Operation *op)
          : LgTaskArgs<DeferredResolutionArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct DeferredExecuteArgs : public LgTaskArgs<DeferredExecuteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_EXECUTION_TRIGGER_ID;
      public:
        DeferredExecuteArgs(Operation *op)
          : LgTaskArgs<DeferredExecuteArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct DeferredExecArgs : public LgTaskArgs<DeferredExecArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_EXECUTE_ID;
      public:
        DeferredExecArgs(Operation *op)
          : LgTaskArgs<DeferredExecArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct TriggerCompleteArgs : public LgTaskArgs<TriggerCompleteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRIGGER_COMPLETE_ID;
      public:
        TriggerCompleteArgs(Operation *op)
          : LgTaskArgs<TriggerCompleteArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct DeferredCompleteArgs : public LgTaskArgs<DeferredCompleteArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMPLETE_ID;
      public:
        DeferredCompleteArgs(Operation *op)
          : LgTaskArgs<DeferredCompleteArgs>(op->get_unique_op_id()),
            proxy_this(op) { }
      public:
        Operation *const proxy_this;
      };
      struct DeferredCommitTriggerArgs : 
        public LgTaskArgs<DeferredCommitTriggerArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMMIT_TRIGGER_ID; 
      public:
        DeferredCommitTriggerArgs(Operation *op)
          : LgTaskArgs<DeferredCommitTriggerArgs>(op->get_unique_op_id()),
            proxy_this(op), gen(op->get_generation()) { }
      public:
        Operation *const proxy_this;
        const GenerationID gen;
      };
      struct DeferredCommitArgs : public LgTaskArgs<DeferredCommitArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COMMIT_ID;
      public:
        DeferredCommitArgs(Operation *op, bool d)
          : LgTaskArgs<DeferredCommitArgs>(op->get_unique_op_id()),
            proxy_this(op), deactivate(d) { }
      public:
        Operation *proxy_this;
        bool deactivate;
      };
    public:
      class MappingDependenceTracker {
      public:
        inline void add_mapping_dependence(RtEvent dependence)
          { mapping_dependences.insert(dependence); }
        inline void add_resolution_dependence(RtEvent dependence)
          { resolution_dependences.insert(dependence); }
        void issue_stage_triggers(Operation *op, Runtime *runtime, 
                                  MustEpochOp *must_epoch);
      private:
        std::set<RtEvent> mapping_dependences;
        std::set<RtEvent> resolution_dependences;
      };
      class CommitDependenceTracker {
      public:
        inline void add_commit_dependence(RtEvent dependence)
          { commit_dependences.insert(dependence); }
        bool issue_commit_trigger(Operation *op, Runtime *runtime);
      private:
        std::set<RtEvent> commit_dependences;
      };
    public:
      Operation(Runtime *rt);
      virtual ~Operation(void);
    public:
      static const char* get_string_rep(OpKind kind);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0; 
      virtual const char* get_logging_name(void) const = 0;
      virtual OpKind get_operation_kind(void) const = 0;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
      virtual Memoizable* get_memoizable(void) { return NULL; }
    protected:
      // Base call
      void activate_operation(void);
      void deactivate_operation(void);
    public:
      inline GenerationID get_generation(void) const { return gen; }
      inline RtEvent get_mapped_event(void) const { return mapped_event; }
      inline RtEvent get_resolved_event(void) const { return resolved_event; }
      inline ApEvent get_completion_event(void) const {return completion_event;}
      inline RtEvent get_commit_event(void) const { return commit_event; }
      inline ApEvent get_execution_fence_event(void) const 
        { return execution_fence_event; }
      inline bool has_execution_fence_event(void) const 
        { return execution_fence_event.exists(); }
      inline void set_execution_fence_event(ApEvent fence_event)
        { execution_fence_event = fence_event; }
      inline TaskContext* get_context(void) const { return parent_ctx; }
      inline UniqueID get_unique_op_id(void) const { return unique_op_id; } 
      virtual bool is_memoizing(void) const { return false; }
      inline bool is_tracing(void) const { return tracing; }
      inline bool is_tracking_parent(void) const { return track_parent; } 
      inline bool already_traced(void) const 
        { return ((trace != NULL) && !tracing); }
      inline LegionTrace* get_trace(void) const { return trace; }
      inline unsigned get_ctx_index(void) const { return context_index; }
    public:
      // Be careful using this call as it is only valid when the operation
      // actually has a parent task.  Right now the only place it is used
      // is in putting the operation in the right dependence queue which
      // we know happens on the home node and therefore the operations is
      // guaranteed to have a parent task.
      unsigned get_operation_depth(void) const; 
    public:
      void initialize_privilege_path(RegionTreePath &path,
                                     const RegionRequirement &req);
      void initialize_mapping_path(RegionTreePath &path,
                                   const RegionRequirement &req,
                                   LogicalRegion start_node);
      void initialize_mapping_path(RegionTreePath &path,
                                   const RegionRequirement &req,
                                   LogicalPartition start_node);
      void set_trace(LegionTrace *trace, bool is_tracing,
                     const std::vector<StaticDependence> *dependences);
      void set_trace_local_id(unsigned id);
      void set_must_epoch(MustEpochOp *epoch, bool do_registration);
    public:
      // Localize a region requirement to its parent context
      // This means that region == parent and the
      // coherence mode is exclusive
      static void localize_region_requirement(RegionRequirement &req);
      void release_acquired_instances(std::map<PhysicalManager*,
                        std::pair<unsigned,bool> > &acquired_instances);
    public:
      // Initialize this operation in a new parent context
      // along with the number of regions this task has
      void initialize_operation(TaskContext *ctx, bool track,
                                unsigned num_regions = 0,
          const std::vector<StaticDependence> *dependences = NULL);
    public:
      // Inherited from ReferenceMutator
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      RtEvent execute_prepipeline_stage(GenerationID gen,
                                        bool from_logical_analysis);
      // This is a virtual method because SpeculativeOp overrides
      // it to check for handling speculation before proceeding
      // with the analysis
      virtual void execute_dependence_analysis(void);
    public:
      // The following calls may be implemented
      // differently depending on the operation, but we
      // provide base versions of them so that operations
      // only have to overload the stages that they care
      // about modifying.
      // See if we have a preprocessing stage
      virtual bool has_prepipeline_stage(void) const;
      // The function call for made for all operations 
      // prior to entering the pipeline 
      virtual void trigger_prepipeline_stage(void);
      // The function to call for depence analysis
      virtual void trigger_dependence_analysis(void);
      // The function to call when the operation has all its
      // mapping depenedences satisfied
      // In general put this on the ready queue so the runtime
      // can invoke the trigger mapping call.
      virtual void trigger_ready(void);
      // The function to call for executing an operation
      // Note that this one is not invoked by the Operation class
      // but by the runtime, therefore any operations must be
      // placed on the ready queue in order for the runtime to
      // perform this mapping
      virtual void trigger_mapping(void);
      // The function to trigger once speculation is
      // ready to be resolved
      virtual void trigger_resolution(void);
      // The function to call once the operation is ready to complete
      virtual void trigger_complete(void);
      // The function to call when commit the operation is
      // ready to commit
      virtual void trigger_commit(void);
      // Helper function for deferring complete operations
      // (only used in a limited set of operations and not
      // part of the default pipeline)
      virtual void deferred_execute(void);
      // Helper function for deferring commit operations
      virtual void deferred_commit_trigger(GenerationID commit_gen);
      // A helper method for deciding what to do when we have
      // aliased region requirements for an operation
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      // A method for finding the parent index of a region
      // requirement for an operation which is necessary for
      // issuing close operation on behalf of the operation.
      virtual unsigned find_parent_index(unsigned idx);
      // Determine if this operation is an internal operation
      virtual bool is_internal_op(void) const { return false; }
      // Determine if this operation is a partition operation
      virtual bool is_partition_op(void) const { return false; }
      // Determine if this is a predicated operation
      virtual bool is_predicated_op(void) const { return false; }
    public: // virtual methods for mapping
      // Pick the sources for a copy operations
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      // Get a reference to our data structure for tracking acquired instances
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
      // Update the set of atomic locks for this operation
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      // Get the restrict precondition for this operation
      virtual ApEvent get_restrict_precondition(void) const;
      static ApEvent merge_restrict_preconditions(
          const std::vector<Grant> &grants,
          const std::vector<PhaseBarrier> &wait_barriers);
      // Record the restrict postcondition
      virtual void record_restrict_postcondition(ApEvent postcondition);
      virtual void add_copy_profiling_request(
                                        Realm::ProfilingRequestSet &reqeusts);
      // Report a profiling result for this operation
      virtual void handle_profiling_response(
                                  const Realm::ProfilingResponse &result);
    protected:
      void filter_copy_request_kinds(MapperManager *mapper,
          const std::set<ProfilingMeasurementID> &requests,
          std::vector<ProfilingMeasurementID> &results, bool warn_if_not_copy);
    public:
      // Help for creating temporary instances
      MaterializedView* create_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager* dst,
                              unsigned index, const FieldMask &needed_fields);
      void validate_temporary_instance(PhysicalManager *result,
                              std::set<PhysicalManager*> &previous_managers,
         const std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired,
                              const FieldMask &needed_fields, 
                              LogicalRegion needed_region,
                              MapperManager *mapper,
                              const char *mapper_call_name) const;
      void log_temporary_instance(PhysicalManager *result, unsigned index,
                                  const FieldMask &needed_fields) const;
    public:
      // The following are sets of calls that we can use to 
      // indicate mapping, execution, resolution, completion, and commit
      //
      // Add this to the list of ready operations
      void enqueue_ready_operation(RtEvent wait_on = RtEvent::NO_RT_EVENT,
                            LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY);
      // Indicate that we are done mapping this operation
      void complete_mapping(RtEvent wait_on = RtEvent::NO_RT_EVENT); 
      // Indicate when this operation has finished executing
      void complete_execution(RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate when we have resolved the speculation for
      // this operation
      void resolve_speculation(RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate that we are completing this operation
      // which will also verify any regions for our producers
      void complete_operation(RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate that we are committing this operation
      void commit_operation(bool do_deactivate,
                            RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate that this operation is hardened against failure
      void harden_operation(void);
      // Quash this task and do what is necessary to the
      // rest of the operations in the graph
      void quash_operation(GenerationID gen, bool restart);
    public:
      // For operations that wish to complete early they can do so
      // using this method which will allow them to immediately 
      // chain an event to directly trigger the completion event
      // Note that we don't support early completion if we're doing
      // inorder program execution
      inline bool request_early_complete(ApEvent chain_event) 
        {
          if (!runtime->program_order_execution)
          {
            need_completion_trigger = false;
            Runtime::trigger_event(completion_event, chain_event);
            return true;
          }
          else
            return false;
        }
      inline bool request_early_complete_no_trigger(ApUserEvent &to_trigger)
        {
          if (!runtime->program_order_execution)
          {
            need_completion_trigger = false;
            to_trigger = completion_event;
            return true;
          }
          else
            return false;
        }
      // For operations that need to trigger commit early,
      // then they should use this call to avoid races
      // which could result in trigger commit being
      // called twice.
      void request_early_commit(void);
    public:
      // Everything below here is implementation
      //
      // Call these two functions before and after
      // dependence analysis, they place a temporary
      // dependence on the operation so that it doesn't
      // prematurely trigger before the analysis is
      // complete.  The end call will trigger the
      // operation if it is complete.
      void begin_dependence_analysis(void);
      void end_dependence_analysis(void);
      // Operations for registering dependences and
      // then notifying them when being woken up
      // This call will attempt to register a dependence
      // from the operation on which it is called to the target
      // Return true if the operation has committed and can be 
      // pruned out of the list of mapping dependences.
      bool register_dependence(Operation *target, GenerationID target_gen);
      // This function call does everything that the previous one does, but
      // it also records information about the regions involved and how
      // whether or not they will be validated by the consuming operation.
      // Return true if the operation has committed and can be pruned
      // out of the list of dependences.
      bool register_region_dependence(unsigned idx, Operation *target,
                              GenerationID target_gen, unsigned target_idx,
                              DependenceType dtype, bool validates,
                              const FieldMask &dependent_mask);
      // This method is invoked by one of the two above to perform
      // the registration.  Returns true if we have not yet commited
      // and should therefore be notified once the dependent operation
      // has committed or verified its regions.
      bool perform_registration(GenerationID our_gen, 
                                Operation *op, GenerationID op_gen,
                                bool &registered_dependence,
                                MappingDependenceTracker *tracker,
                                RtEvent other_commit_event);
      // Check to see if the operation is still valid
      // for the given GenerationID.  This method is not precise
      // and may return false when the operation has committed.
      // However, the converse will never be occur.
      bool is_operation_committed(GenerationID gen);
      // Add and remove mapping references to tell an operation
      // how many places additional dependences can come from.
      // Once the mapping reference count goes to zero, no
      // additional dependences can be registered.
      bool add_mapping_reference(GenerationID gen);
      void remove_mapping_reference(GenerationID gen);
    public:
      // Some extra support for tracking dependences that we've 
      // registered as part of our logical traversal
      void record_logical_dependence(const LogicalUser &user);
      inline LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned&
          get_logical_records(void) { return logical_records; }
      inline LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned&
          get_logical_advances(void) { return logical_advances; }
      void clear_logical_records(void);
    public:
      // Notify when a region from a dependent task has 
      // been verified (flows up edges)
      void notify_regions_verified(const std::set<unsigned> &regions,
                                   GenerationID gen);
    public:
      // Help for finding the contexts for an operation
      InnerContext* find_logical_context(unsigned index);
      InnerContext* find_physical_context(unsigned index);
    public: // Support for mapping operations
      static void prepare_for_mapping(const InstanceRef &ref,
                                      MappingInstance &instance);
      static void prepare_for_mapping(const InstanceSet &valid,
                           std::vector<MappingInstance> &input_valid);
      static void prepare_for_mapping(const InstanceSet &valid,
                           const std::set<Memory> &filter_memories,
                           std::vector<MappingInstance> &input_valid);
      void compute_ranking(MapperManager            *mapper,
          const std::deque<MappingInstance>         &output,
          const InstanceSet                         &sources,
          std::vector<unsigned>                     &ranking) const;
    public:
      // Perform the versioning analysis for a projection requirement
      void perform_projection_version_analysis(const ProjectionInfo &proj_info,
                                      const RegionRequirement &owner_req,
                                      const RegionRequirement &local_req,
                                      const unsigned idx,
                                      const UniqueID logical_context_uid,
                                      VersionInfo &version_info,
                                      std::set<RtEvent> &ready_events);
#ifdef DEBUG_LEGION
    protected:
      virtual void dump_physical_state(RegionRequirement *req, unsigned idx,
                                       bool before = false,
                                       bool closing = false);
#endif
    public:
      Runtime *const runtime;
    protected:
      mutable LocalLock op_lock;
      GenerationID gen;
      UniqueID unique_op_id;
      // The issue index of this operation in the context
      unsigned context_index;
      // Operations on which this operation depends
      std::map<Operation*,GenerationID> incoming;
      // Operations which depend on this operation
      std::map<Operation*,GenerationID> outgoing;
      // Number of outstanding mapping references, once this goes to 
      // zero then the set of outgoing edges is fixed
      unsigned outstanding_mapping_references;
      // The set of unverified regions
      std::set<unsigned> unverified_regions;
      // For each of our regions, a map of operations to the regions
      // which we can verify for each operation
      std::map<Operation*,std::set<unsigned> > verify_regions;
      // Whether this operation has executed its prepipeline stage yet
      bool prepipelined;
#ifdef DEBUG_LEGION
      // Whether this operation has mapped, once it has mapped then
      // the set of incoming dependences is fixed
      bool mapped;
      // Whether this task has executed or not
      bool executed;
      // Whether speculation for this operation has been resolved
      bool resolved;
#endif
      // Whether this operation has completed, cannot commit until
      // both completed is set, and outstanding mapping references
      // has been gone to zero.
      bool completed;
      // Some operations commit out of order and if they do then
      // commited is set to prevent any additional dependences from
      // begin registered.
      bool committed;
      // Whether the physical instances for this region have been
      // hardened by copying them into reslient memories
      bool hardened;
      // Track whether trigger_commit has already been invoked
      bool trigger_commit_invoked;
      // Keep track of whether an eary commit was requested
      bool early_commit_request;
      // Indicate whether we are responsible for
      // triggering the completion event for this operation
      bool need_completion_trigger;
      // Are we tracking this operation in the parent's context
      bool track_parent;
      // The enclosing context for this operation
      TaskContext *parent_ctx;
      // The prepipeline event for this operation
      RtUserEvent prepipelined_event;
      // The mapped event for this operation
      RtUserEvent mapped_event;
      // The resolved event for this operation
      RtUserEvent resolved_event;
      // The completion event for this operation
      ApUserEvent completion_event;
      // The commit event for this operation
      RtUserEvent commit_event;
      // Previous execution fence if there was one
      ApEvent execution_fence_event;
      // The trace for this operation if any
      LegionTrace *trace;
      // Track whether we are tracing this operation
      bool tracing;
      // The id local to a trace
      unsigned trace_local_id;
      // Our must epoch if we have one
      MustEpochOp *must_epoch;
      // A set list or recorded dependences during logical traversal
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned logical_records;
      // A set of advance operations recorded during logical traversal
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned logical_advances;
      // Dependence trackers for detecting when it is safe to map and commit
      MappingDependenceTracker *mapping_tracker;
      CommitDependenceTracker  *commit_tracker;
    };

    /**
     * \class PredicateWaiter
     * An interface class for speculative operations
     * and compound predicates that allows them to
     * be notified when their constituent predicates
     * have been resolved.
     */
    class PredicateWaiter {
    public:
      virtual void notify_predicate_value(GenerationID gen, bool value) = 0;
    };

    /**
     * \class Predicate 
     * A predicate operation is an abstract class that
     * contains a method that allows other operations to
     * sample their values and see if they are resolved
     * or whether they are speculated values.
     */
    class PredicateImpl : public Operation {
    public:
      PredicateImpl(Runtime *rt);
    public:
      void activate_predicate(void);
      void deactivate_predicate(void);
    public:
      void add_predicate_reference(void);
      void remove_predicate_reference(void);
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    public:
      bool register_waiter(PredicateWaiter *waiter, 
                           GenerationID gen, bool &value);
      PredEvent get_true_guard(void);
      PredEvent get_false_guard(void);
      void get_predicate_guards(PredEvent &true_guard, PredEvent &false_guard);
      Future get_future_result(void);
    protected:
      void set_resolved_value(GenerationID pred_gen, bool value);
    protected:
      bool predicate_resolved;
      bool predicate_value;
      std::map<PredicateWaiter*,GenerationID> waiters;
    protected:
      RtUserEvent collect_predicate;
      unsigned predicate_references;
      PredEvent true_guard, false_guard;
    protected:
      Future result_future;
      bool can_result_future_complete;
    };

    /**
     * \class SpeculativeOp
     * A speculative operation is an abstract class
     * that serves as the basis for operation which
     * can be speculated on a predicate value.  They
     * will ask the predicate value for their value and
     * whether they have actually been resolved or not.
     * Based on that infomration the speculative operation
     * will decide how to manage the operation.
     */
    class SpeculativeOp : public Operation, PredicateWaiter {
    public:
      enum SpecState {
        PENDING_ANALYSIS_STATE,
        SPECULATE_TRUE_STATE,
        SPECULATE_FALSE_STATE,
        RESOLVE_TRUE_STATE,
        RESOLVE_FALSE_STATE,
      };
    public:
      SpeculativeOp(Runtime *rt);
    public:
      void activate_speculative(void);
      void deactivate_speculative(void);
    public:
      void initialize_speculation(TaskContext *ctx, bool track,unsigned regions,
          const std::vector<StaticDependence> *dependences, const Predicate &p);
      void register_predicate_dependence(void);
      virtual bool is_predicated_op(void) const;
      // Wait until the predicate is valid and then return
      // its value.  Give it the current processor in case it
      // needs to wait for the value
      bool get_predicate_value(Processor proc);
    public:
      // Override the execute dependence analysis call so 
      // we can decide whether to continue performing the 
      // dependence analysis here or not
      virtual void execute_dependence_analysis(void);
      virtual void trigger_resolution(void);
    public:
      // Call this method for inheriting classes 
      // to determine whether they should speculate 
      virtual bool query_speculate(bool &value, bool &mapping_only) = 0;
    public:
      // Every speculative operation will always get exactly one
      // call back to one of these methods after the predicate has
      // resolved. The 'speculated' parameter indicates whether the
      // operation was speculated by the mapper. The 'launch' parameter
      // indicates whether the operation has been issued into the 
      // pipeline for execution yet
      virtual void resolve_true(bool speculated, bool launched) = 0;
      virtual void resolve_false(bool speculated, bool launched) = 0;
    public:
      virtual void notify_predicate_value(GenerationID gen, bool value);
    protected:
      SpecState    speculation_state;
      PredicateOp *predicate;
      bool speculate_mapping_only;
      bool received_trigger_resolution;
    protected:
      RtUserEvent predicate_waiter; // used only when needed
    };

    /**
     * \class MemoizableOp
     * A memoizable operation is an abstract class
     * that serves as the basis for operation whose
     * physical analysis can be memoized.  Memoizable
     * operations go through an extra step in the mapper
     * to determine whether to memoize their physical analysis.
     */
    template<typename OP>
    class MemoizableOp : public OP, public Memoizable
    {
    public:
      enum MemoizableState {
        NO_MEMO,   // The operation is not subject to memoization
        MEMO_REQ,  // The mapper requested memoization on this operation
        RECORD,    // The runtime is recording analysis for this operation
        REPLAY,    // The runtime is replaying analysis for this opeartion
      };
    public:
      MemoizableOp(Runtime *rt);
      void initialize_memoizable(void);
      virtual Memoizable* get_memoizable(void) { return this; }
    protected:
      void pack_memoizable(Serializer &rez);
      void unpack_memoizable(Deserializer &derez);
    protected:
      void activate_memoizable(void);
    public:
      virtual void execute_dependence_analysis(void);
      virtual void replay_analysis(void) = 0;
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id() const;
      virtual ApEvent compute_sync_precondition(void) const
        { assert(false); return ApEvent::NO_AP_EVENT; }
      virtual void complete_replay(ApEvent complete_event)
        { assert(false); }
    protected:
      void invoke_memoize_operation(MapperID mapper_id);
      void set_memoize(bool memoize);
    public:
      virtual bool is_memoizing(void) const { return memo_state != NO_MEMO; }
      bool is_replaying(void) const { return memo_state == REPLAY; }
      bool is_recording(void) const { return memo_state == RECORD; }
    protected:
      // The physical trace for this operation if any
      PhysicalTemplate *tpl;
      // Track whether we are memoizing physical analysis for this operation
      MemoizableState memo_state;
      bool need_prepipeline_stage;
    };

    /**
     * \class MapOp
     * Mapping operations are used for computing inline mapping
     * operations.  Mapping operations will always update a
     * physical region once they have finished mapping.  They
     * then complete and commit immediately, possibly even
     * before the physical region is ready to be used.  This
     * also reflects that mapping operations cannot be rolled
     * back because once they have mapped, then information
     * has the ability to escape back to the application's
     * domain and can no longer be tracked by Legion.  Any
     * attempt to roll back an inline mapping operation
     * will result in the entire enclosing task context
     * being restarted.
     */
    class MapOp : public InlineMapping, public Operation,
                  public LegionHeapify<MapOp> {
    public:
      static const AllocationType alloc_type = MAP_OP_ALLOC;
    public:
      MapOp(Runtime *rt);
      MapOp(const MapOp &rhs);
      virtual ~MapOp(void);
    public:
      MapOp& operator=(const MapOp &rhs);
    public:
      PhysicalRegion initialize(TaskContext *ctx,
                                const InlineLauncher &launcher,
                                bool check_privileges);
      void initialize(TaskContext *ctx, const PhysicalRegion &region);
      inline const RegionRequirement& get_requirement(void) const
        { return requirement; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void deferred_execute(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual void record_restrict_postcondition(ApEvent postcondition);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(const InstanceSet &valid_instances,
                               InstanceSet &mapped_instances);
      virtual void add_copy_profiling_request(
                            Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                      const Realm::ProfilingResponse &response);
    protected:
      bool remap_region;
      ApUserEvent termination_event;
      PhysicalRegion region;
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::map<Reservation,bool> atomic_locks;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> mapped_preconditions;
    protected:
      MapperManager *mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class CopyOp
     * The copy operation provides a mechanism for applications
     * to directly copy data between pairs of fields possibly
     * from different region trees in an efficient way by
     * using the low-level runtime copy facilities. 
     */
    class CopyOp : public Copy, public MemoizableOp<SpeculativeOp>,
                   public LegionHeapify<CopyOp> {
    public:
      static const AllocationType alloc_type = COPY_OP_ALLOC;
    public:
      CopyOp(Runtime *rt);
      CopyOp(const CopyOp &rhs);
      virtual ~CopyOp(void);
    public:
      CopyOp& operator=(const CopyOp &rhs);
    public:
      void initialize(TaskContext *ctx,
                      const CopyLauncher &launcher,
                      bool check_privileges);
      void activate_copy(void);
      void deactivate_copy(void);
      void log_copy_requirements(void) const;
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual ApEvent get_restrict_precondition(void) const;
      virtual void record_restrict_postcondition(ApEvent postcondition);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
    protected:
      void check_copy_privilege(const RegionRequirement &req, unsigned idx,
                                bool permit_projection = false);
      void compute_parent_indexes(void);
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(void) const;
      virtual void complete_replay(ApEvent copy_complete_event);
    protected:
      template<bool IS_SRC>
      int perform_conversion(unsigned idx, const RegionRequirement &req,
                             std::vector<MappingInstance> &output,
                             InstanceSet &targets, bool is_reduce = false);
      inline void set_mapping_state(unsigned idx) 
        { current_index = idx; }
      virtual void add_copy_profiling_request(
                                      Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                                const Realm::ProfilingResponse &response);
    public:
      std::vector<RegionTreePath> src_privilege_paths;
      std::vector<RegionTreePath> dst_privilege_paths;
      std::vector<unsigned>       src_parent_indexes;
      std::vector<unsigned>       dst_parent_indexes;
      std::vector<VersionInfo>    src_versions;
      std::vector<VersionInfo>    dst_versions;
      std::vector<RestrictInfo>   src_restrict_infos;
      std::vector<RestrictInfo>   dst_restrict_infos;
    public: // These are only used for indirect copies
      std::vector<RegionTreePath> gather_privilege_paths;
      std::vector<RegionTreePath> scatter_privilege_paths;
      std::vector<unsigned>       gather_parent_indexes;
      std::vector<unsigned>       scatter_parent_indexes;
      std::vector<VersionInfo>    gather_versions;
      std::vector<VersionInfo>    scatter_versions;
      std::vector<RestrictInfo>   gather_restrict_infos;
      std::vector<RestrictInfo>   scatter_restrict_infos;
    protected: // for support with mapping
      MapperManager*              mapper;
      unsigned                    current_index;
    protected:
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::vector<std::map<Reservation,bool> > atomic_locks;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> restrict_postconditions;
    public:
      PredEvent                   predication_guard;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class IndexCopyOp
     * An index copy operation is the same as a copy operation
     * except it is an index space operation for performing
     * multiple copies with projection functions
     */
    class IndexCopyOp : public CopyOp {
    public:
      IndexCopyOp(Runtime *rt);
      IndexCopyOp(const IndexCopyOp &rhs);
      virtual ~IndexCopyOp(void);
    public:
      IndexCopyOp& operator=(const IndexCopyOp &rhs);
    public:
      void initialize(TaskContext *ctx,
                      const IndexCopyLauncher &launcher,
                      IndexSpace launch_space,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
    public:
      void handle_point_commit(RtEvent point_committed);
#ifdef DEBUG_LEGION
      void check_point_requirements(void);
#endif
    public:
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
    public:
      IndexSpace                    launch_space;
    public:
      std::vector<ProjectionInfo>   src_projection_infos;
      std::vector<ProjectionInfo>   dst_projection_infos;
      std::vector<ProjectionInfo>   gather_projection_infos;
      std::vector<ProjectionInfo>   scatter_projection_infos;
    protected:
      std::vector<PointCopyOp*>     points;
      unsigned                      points_committed;
      bool                          commit_request;
      std::set<RtEvent>             commit_preconditions;
#ifdef DEBUG_LEGION
    protected:
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned,unsigned> > interfering_requirements;
#endif
    };

    /**
     * \class PointCopyOp
     * A point copy operation is used for executing the
     * physical part of the analysis for an index copy
     * operation.
     */
    class PointCopyOp : public CopyOp, public ProjectionPoint {
    public:
      PointCopyOp(Runtime *rt);
      PointCopyOp(const PointCopyOp &rhs);
      virtual ~PointCopyOp(void);
    public:
      PointCopyOp& operator=(const PointCopyOp &rhs);
    public:
      void initialize(IndexCopyOp *owner, const DomainPoint &point);
#ifdef DEBUG_LEGION
      void check_domination(void) const;
#endif
      void launch(const std::set<RtEvent> &preconditions);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      // trigger_mapping same as base class
      virtual void trigger_commit(void);
    public:
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx,LogicalRegion result);
    public:
      virtual const ProjectionInfo* get_projection_info(unsigned idx);
    protected:
      IndexCopyOp*              owner;
    };

    /**
     * \class FenceOp
     * Fence operations give the application the ability to
     * enforce ordering guarantees between different tasks
     * in the same context which may become important when
     * certain updates to the region tree are desired to be
     * observed before a later operation either maps or 
     * runs.  To support these two kinds of guarantees, we
     * provide both mapping and executing fences.
     */
    class FenceOp : public Operation, public LegionHeapify<FenceOp> {
    public:
      enum FenceKind {
        MAPPING_FENCE,
        EXECUTION_FENCE,
        MIXED_FENCE,
      };
    public:
      static const AllocationType alloc_type = FENCE_OP_ALLOC;
    public:
      FenceOp(Runtime *rt);
      FenceOp(const FenceOp &rhs);
      virtual ~FenceOp(void);
    public:
      FenceOp& operator=(const FenceOp &rhs);
    public:
      void initialize(TaskContext *ctx, FenceKind kind);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      void perform_fence_analysis(bool update_fence = false);
      void update_current_fence(void);
    protected:
      FenceKind fence_kind;
      ApEvent execution_precondition;
    };

    /**
     * \class FrameOp
     * Frame operations provide a mechanism for grouping 
     * operations within the same context into frames. Frames
     * provide an application directed way of controlling the
     * number of outstanding operations in flight in a context
     * at any given time through the mapper interface.
     */
    class FrameOp : public FenceOp {
    public:
      static const AllocationType alloc_type = FRAME_OP_ALLOC;
    public:
      FrameOp(Runtime *rt);
      FrameOp(const FrameOp &rhs);
      virtual ~FrameOp(void);
    public:
      FrameOp& operator=(const FrameOp &rhs);
    public:
      void initialize(TaskContext *ctx);
      void set_previous(ApEvent previous);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_mapping(void);
      virtual void deferred_execute(void); 
    protected:
      ApEvent previous_completion;
    };

    /**
     * \class DeletionOp
     * In keeping with the deferred execution model, deletions
     * must be deferred until all other operations that were
     * issued earlier are done using the regions that are
     * going to be deleted.  Deletion operations defer deletions
     * until they are safe to be committed.
     */
    class DeletionOp : public Operation, public LegionHeapify<DeletionOp> {
    public:
      static const AllocationType alloc_type = DELETION_OP_ALLOC;
    public:
      enum DeletionKind {
        INDEX_SPACE_DELETION,
        INDEX_PARTITION_DELETION,
        FIELD_SPACE_DELETION,
        FIELD_DELETION,
        LOGICAL_REGION_DELETION,
        LOGICAL_PARTITION_DELETION,
      };
    public:
      DeletionOp(Runtime *rt);
      DeletionOp(const DeletionOp &rhs);
      virtual ~DeletionOp(void);
    public:
      DeletionOp& operator=(const DeletionOp &rhs);
    public:
      void initialize_index_space_deletion(TaskContext *ctx, IndexSpace handle);
      void initialize_index_part_deletion(TaskContext *ctx,
                                          IndexPartition handle);
      void initialize_field_space_deletion(TaskContext *ctx,
                                           FieldSpace handle);
      void initialize_field_deletion(TaskContext *ctx, FieldSpace handle,
                                      FieldID fid);
      void initialize_field_deletions(TaskContext *ctx, FieldSpace handle,
                                      const std::set<FieldID> &to_free);
      void initialize_logical_region_deletion(TaskContext *ctx, 
                                              LogicalRegion handle);
      void initialize_logical_partition_deletion(TaskContext *ctx, 
                                                 LogicalPartition handle);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void); 
      virtual void trigger_complete(void);
      virtual unsigned find_parent_index(unsigned idx);
    protected:
      DeletionKind kind;
      IndexSpace index_space;
      IndexPartition index_part;
      FieldSpace field_space;
      LogicalRegion logical_region;
      LogicalPartition logical_part;
      std::set<FieldID> free_fields;
      std::vector<unsigned> parent_req_indexes;
      ApEvent completion_precondition;
    }; 

    /**
     * \class InternalOp
     * The InternalOp class is an abstract intermediate class
     * for detecting when an operation is generated by the 
     * runtime and not one created by the runtime. This
     * distinction is primarily emplyed by the tracing 
     * infrastructure which can memoize analysis overheads
     * for application operations, but must still handle
     * internal operations correctly.
     */
    class InternalOp : public Operation {
    public:
      InternalOp(Runtime *rt);
      virtual ~InternalOp(void);
    public:
      void initialize_internal(Operation *creator, int creator_req_idx,
                               const LogicalTraceInfo &trace_info);
      void activate_internal(void);
      void deactivate_internal(void);
    public:
      virtual bool is_internal_op(void) const { return true; }
      virtual const FieldMask& get_internal_mask(void) const = 0;
    public:
      inline int get_internal_index(void) const { return creator_req_idx; }
      void record_trace_dependence(Operation *target, GenerationID target_gen,
                                   int target_idx, int source_idx, 
                                   DependenceType dtype,
                                   const FieldMask &dependent_mask);
      virtual unsigned find_parent_index(unsigned idx);
    protected:
      // These things are really only needed for tracing
      // Information about the operation that generated
      // this close operation so we don't register dependences on it
      Operation *create_op;
      GenerationID create_gen;
      // The source index of the region requirement from the original 
      // operation that generated this internal operation
      int creator_req_idx;
    };

    /**
     * \class OpenOp
     * Open operatoins are only visible internally inside
     * the runtime and are issued to open a region tree
     * down to the level immediately above the one being
     * accessed by a given operation. Open operations
     * record whether they are advancing the version
     * number information at a given level or not.
     */
    class OpenOp : public InternalOp, public LegionHeapify<OpenOp> {
    public:
      static const AllocationType alloc_type = OPEN_OP_ALLOC;
    public:
      OpenOp(Runtime *rt);
      OpenOp(const OpenOp &rhs);
      virtual ~OpenOp(void);
    public:
      OpenOp& operator=(const OpenOp &rhs);
    public:
      void initialize(const FieldMask &open_mask, RegionTreeNode *start, 
                      const RegionTreePath &path, 
                      const LogicalTraceInfo &trace_info,
                      Operation *creator, int req_idx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual const FieldMask& get_internal_mask(void) const;
    public:
      virtual void trigger_ready(void);
    protected:
      RegionTreeNode *start_node;
      RegionTreePath  open_path;
      FieldMask       open_mask;
    };

    /**
     * \class AdvanceOp
     * Advance operations are only visible internally inside
     * the runtime and are issued to advance version numbers
     * on intermediate nodes in the region tree when data
     * is being written to a subregion.
     */
    class AdvanceOp : public InternalOp, public LegionHeapify<AdvanceOp> {
    public:
      static const AllocationType alloc_type = ADVANCE_OP_ALLOC;
    public:
      AdvanceOp(Runtime *rt);
      AdvanceOp(const AdvanceOp &rhs);
      virtual ~AdvanceOp(void);
    public:
      AdvanceOp& operator=(const AdvanceOp &rhs);
    public:
      void initialize(RegionTreeNode *parent, const FieldMask &advance,
                      const LogicalTraceInfo &trace_info, Operation *creator, 
                      int req_idx, bool parent_is_upper_bound);
      void set_child_node(RegionTreeNode *child);
      void set_split_child_mask(const FieldMask &split_mask);
      void record_dirty_previous(unsigned depth, const FieldMask &dirty_mask);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual const FieldMask& get_internal_mask(void) const;
    public:
      virtual void trigger_ready(void);
    protected:
      RegionTreeNode *parent_node;
      RegionTreeNode *child_node; // inclusive
      FieldMask       advance_mask;
      FieldMask       split_child_mask; // only for partial reductions
      LegionMap<unsigned,FieldMask>::aligned dirty_previous;
      bool parent_is_upper_bound;
    };

    /**
     * \class CloseOp
     * Close operations are only visible internally inside
     * the runtime and are issued to help close up the 
     * physical region tree. There are two types of close
     * operations that both inherit from this class:
     * InterCloseOp and PostCloseOp.
     */
    class CloseOp : public Close, public InternalOp {
    public:
      static const AllocationType alloc_type = CLOSE_OP_ALLOC;
    public:
      CloseOp(Runtime *rt);
      CloseOp(const CloseOp &rhs);
      virtual ~CloseOp(void);
    public:
      CloseOp& operator=(const CloseOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
      virtual Mappable* get_mappable(void);
    public:
      void activate_close(void);
      void deactivate_close(void);
      // This is for post and virtual close ops
      void initialize_close(TaskContext *ctx,
                            const RegionRequirement &req, bool track);
      // These is for internal close ops
      void initialize_close(Operation *creator, unsigned idx,
                            unsigned parent_req_index,
                            const RegionRequirement &req,
                            const LogicalTraceInfo &trace_info);
      void perform_logging(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void) const = 0;
      virtual OpKind get_operation_kind(void) const = 0;
      virtual size_t get_region_count(void) const;
      virtual const FieldMask& get_internal_mask(void) const;
    public:
      virtual void trigger_commit(void);
    protected:
      RegionTreePath privilege_path;
      VersionInfo    version_info;
      RestrictInfo   restrict_info;
    };

    /**
     * \class InterCloseOp
     * Intermediate close operations are issued by the runtime
     * for closing up region trees as part of the normal execution
     * of an application.
     */
    class InterCloseOp : public CloseOp, public LegionHeapify<InterCloseOp> {
    public:
      struct DisjointCloseInfo {
      public:
        FieldMask close_mask;
        VersionInfo version_info;
        ClosedNode *close_node;
        std::set<RtEvent> ready_events;
      };
      struct DisjointCloseArgs : public LgTaskArgs<DisjointCloseArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DISJOINT_CLOSE_TASK_ID;
      public:
        DisjointCloseArgs(InterCloseOp *op, 
                          RegionTreeNode *child, InnerContext *ctx)
          : LgTaskArgs<DisjointCloseArgs>(op->get_unique_op_id()),
            proxy_this(op), child_node(child), context(ctx) { }
      public:
        InterCloseOp *const proxy_this;
        RegionTreeNode *const child_node;
        InnerContext *const context;
      };
    public:
      InterCloseOp(Runtime *runtime);
      InterCloseOp(const InterCloseOp &rhs);
      virtual ~InterCloseOp(void);
    public:
      InterCloseOp& operator=(const InterCloseOp &rhs);
    public:
      void initialize(TaskContext *ctx, const RegionRequirement &req,
                      ClosedNode *closed_tree, 
                      const LogicalTraceInfo &trace_info,
                      int close_idx, const VersionInfo &version_info,
                      const FieldMask &close_mask, Operation *create_op);
      ProjectionInfo& initialize_disjoint_close(const FieldMask &disjoint_mask,
                                                IndexSpace launch_space);
      DisjointCloseInfo* find_disjoint_close_child(unsigned index,
                                                   RegionTreeNode *child);
      void perform_disjoint_close(RegionTreeNode *child_to_close, 
                                  DisjointCloseInfo &close_info,
                                  InnerContext *context,
                                  std::set<RtEvent> &ready_events);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual const FieldMask& get_internal_mask(void) const;
    public:
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
    protected:
      void invoke_mapper(const InstanceSet &valid_instances);
      virtual void add_copy_profiling_request(
                                          Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                                    const Realm::ProfilingResponse &response);
    public:
      static void handle_disjoint_close(const void *args);
    protected:
      FieldMask close_mask;
      ClosedNode *closed_tree;
      InstanceSet chosen_instances;
    protected:
      // For disjoint partition closes with projections
      FieldMask disjoint_close_mask;
      ProjectionInfo projection_info;
      LegionMap<RegionTreeNode*,DisjointCloseInfo>::aligned children_to_close;
    protected:
      unsigned parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };
    
    /**
     * \class ReadCloseOp
     * Read close operations are close ops that act as 
     * place holders for closing up read-only partitions.
     * Closing a read-only partition doesn't actually involve
     * any work, but we do need something to ensure that all
     * the mapping dependences are satisfied for later operations
     * that traverse different subtrees. Read close operations
     * are summaries for all those dependences to reduce the
     * overhead of testing against everything in a subtree.
     */
    class ReadCloseOp : public CloseOp, public LegionHeapify<ReadCloseOp> {
    public:
      ReadCloseOp(Runtime *runtime);
      ReadCloseOp(const ReadCloseOp &rhs);
      virtual ~ReadCloseOp(void);
    public:
      ReadCloseOp& operator=(const ReadCloseOp &rhs);
    public:
      void initialize(TaskContext *ctx, const RegionRequirement &req,
                      const LogicalTraceInfo &trace_info, int close_idx,
                      const FieldMask &close_mask, Operation *create_op);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual const FieldMask& get_internal_mask(void) const;
    public:
      virtual unsigned find_parent_index(unsigned idx);
    protected:
      unsigned parent_req_index; 
    protected:
      FieldMask close_mask;
    };

    /**
     * \class PostCloseOp
     * Post close operations are issued by the runtime after a
     * task has finished executing and the region tree contexts
     * need to be closed up to the original physical instance
     * that was mapped by the parent task.
     */
    class PostCloseOp : public CloseOp {
    public:
      PostCloseOp(Runtime *runtime);
      PostCloseOp(const PostCloseOp &rhs);
      virtual ~PostCloseOp(void);
    public:
      PostCloseOp& operator=(const PostCloseOp &rhs);
    public:
      void initialize(TaskContext *ctx, unsigned index, 
                      const InstanceSet &target_instances); 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
    protected:
      virtual void add_copy_profiling_request(
                                          Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                                    const Realm::ProfilingResponse &response);
    protected:
      unsigned parent_idx;
      InstanceSet target_instances;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class VirtualCloseOp
     * Virtual close operations are issued by the runtime for
     * closing up virtual mappings to a composite instance
     * that can then be propagated back to the enclosing
     * parent task.
     */
    class VirtualCloseOp : public CloseOp {
    public:
      VirtualCloseOp(Runtime *runtime);
      VirtualCloseOp(const VirtualCloseOp &rhs);
      virtual ~VirtualCloseOp(void);
    public:
      VirtualCloseOp& operator=(const VirtualCloseOp &rhs);
    public:
      void initialize(TaskContext *ctx, unsigned index,
                      const RegionRequirement &req);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual unsigned find_parent_index(unsigned idx);
    protected:
      unsigned parent_idx;
    };

    /**
     * \class AcquireOp
     * Acquire operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class AcquireOp : public Acquire, public MemoizableOp<SpeculativeOp>,
                      public LegionHeapify<AcquireOp> {
    public:
      static const AllocationType alloc_type = ACQUIRE_OP_ALLOC;
    public:
      AcquireOp(Runtime *rt);
      AcquireOp(const AcquireOp &rhs);
      virtual ~AcquireOp(void);
    public:
      AcquireOp& operator=(const AcquireOp &rhs);
    public:
      void initialize(TaskContext *ctx, const AcquireLauncher &launcher,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const; 
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual ApEvent get_restrict_precondition(void) const;
    public: 
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(void) const;
      virtual void complete_replay(ApEvent acquire_complete_event);
    protected:
      void check_acquire_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      virtual void add_copy_profiling_request(
                                          Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                                    const Realm::ProfilingResponse &response);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      RestrictInfo      restrict_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class ReleaseOp
     * Release operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class ReleaseOp : public Release, public MemoizableOp<SpeculativeOp>,
                      public LegionHeapify<ReleaseOp> {
    public:
      static const AllocationType alloc_type = RELEASE_OP_ALLOC;
    public:
      ReleaseOp(Runtime *rt);
      ReleaseOp(const ReleaseOp &rhs);
      virtual ~ReleaseOp(void);
    public:
      ReleaseOp& operator=(const ReleaseOp &rhs);
    public:
      void initialize(TaskContext *ctx, const ReleaseLauncher &launcher,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual ApEvent get_restrict_precondition(void) const;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(void) const;
      virtual void complete_replay(ApEvent release_complete_event);
    protected:
      void check_release_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      virtual void add_copy_profiling_request(
                                          Realm::ProfilingRequestSet &reqeusts);
      virtual void handle_profiling_response(
                                    const Realm::ProfilingResponse &response);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      RestrictInfo      restrict_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class DynamicCollectiveOp
     * A class for getting values from a collective operation
     * and writing them into a future. This will also give
     * us the framework necessary to handle roll backs on 
     * collectives so we can memoize their results.
     */
    class DynamicCollectiveOp : public Mappable,
                                public MemoizableOp<Operation>,
                                public LegionHeapify<DynamicCollectiveOp> {
    public:
      static const AllocationType alloc_type = DYNAMIC_COLLECTIVE_OP_ALLOC;
    public:
      DynamicCollectiveOp(Runtime *rt);
      DynamicCollectiveOp(const DynamicCollectiveOp &rhs);
      virtual ~DynamicCollectiveOp(void);
    public:
      DynamicCollectiveOp& operator=(const DynamicCollectiveOp &rhs);
    public:
      Future initialize(TaskContext *ctx, const DynamicCollective &dc);
    public:
      // From Mappable
      virtual UniqueID get_unique_id(void) const { return unique_op_id; }
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
      virtual MappableType get_mappable_type(void) const
        { return DYNAMIC_COLLECTIVE_MAPPABLE; }
      virtual const Task* as_task(void) const { return NULL; }
      virtual const Copy* as_copy(void) const { return NULL; }
      virtual const InlineMapping* as_inline(void) const { return NULL; }
      virtual const Acquire* as_acquire(void) const { return NULL; }
      virtual const Release* as_release(void) const { return NULL; }
      virtual const Close* as_close(void) const { return NULL; }
      virtual const Fill* as_fill(void) const { return NULL; }
      virtual const Partition* as_partition(void) const { return NULL; }
      virtual const DynamicCollective* as_dynamic_collective(void) const
        { return &collective; }
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void deferred_execute(void);
      virtual void trigger_complete(void);
    protected:
      Future future;
      DynamicCollective collective;
    };

    /**
     * \class FuturePredOp
     * A class for making predicates out of futures.
     */
    class FuturePredOp : public PredicateOp, 
                         public LegionHeapify<FuturePredOp> {
    public:
      static const AllocationType alloc_type = FUTURE_PRED_OP_ALLOC;
    public:
      struct ResolveFuturePredArgs : public LgTaskArgs<ResolveFuturePredArgs> {
      public:
        static const LgTaskID TASK_ID = LG_RESOLVE_FUTURE_PRED_ID;
      public:
        ResolveFuturePredArgs(FuturePredOp *op)
          : LgTaskArgs<ResolveFuturePredArgs>(op->get_unique_op_id()),
            future_pred_op(op) { }
      public:
        FuturePredOp *const future_pred_op;
      };
    public:
      FuturePredOp(Runtime *rt);
      FuturePredOp(const FuturePredOp &rhs);
      virtual ~FuturePredOp(void);
    public:
      FuturePredOp& operator=(const FuturePredOp &rhs);
    public:
      void initialize(TaskContext *ctx, Future f);
      void resolve_future_predicate(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      const char* get_logging_name(void) const;
      OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
    protected:
      Future future;
    };

    /**
     * \class NotPredOp
     * A class for negating other predicates
     */
    class NotPredOp : public PredicateOp, PredicateWaiter,
                      public LegionHeapify<NotPredOp> {
    public:
      static const AllocationType alloc_type = NOT_PRED_OP_ALLOC;
    public:
      NotPredOp(Runtime *rt);
      NotPredOp(const NotPredOp &rhs);
      virtual ~NotPredOp(void);
    public:
      NotPredOp& operator=(const NotPredOp &rhs);
    public:
      void initialize(TaskContext *task, const Predicate &p);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void notify_predicate_value(GenerationID gen, bool value);
    protected:
      PredicateOp *pred_op;
    };

    /**
     * \class AndPredOp
     * A class for and-ing other predicates
     */
    class AndPredOp : public PredicateOp, PredicateWaiter,
                      public LegionHeapify<AndPredOp> {
    public:
      static const AllocationType alloc_type = AND_PRED_OP_ALLOC;
    public:
      AndPredOp(Runtime *rt);
      AndPredOp(const AndPredOp &rhs);
      virtual ~AndPredOp(void);
    public:
      AndPredOp& operator=(const AndPredOp &rhs);
    public:
      void initialize(TaskContext *task, 
                      const std::vector<Predicate> &predicates);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void notify_predicate_value(GenerationID pred_gen, bool value);
    protected:
      std::vector<PredicateOp*> previous;
      unsigned                  true_count;
      bool                      false_short;
    };

    /**
     * \class OrPredOp
     * A class for or-ing other predicates
     */
    class OrPredOp : public PredicateOp, PredicateWaiter,
                     public LegionHeapify<OrPredOp> {
    public:
      static const AllocationType alloc_type = OR_PRED_OP_ALLOC;
    public:
      OrPredOp(Runtime *rt);
      OrPredOp(const OrPredOp &rhs);
      virtual ~OrPredOp(void);
    public:
      OrPredOp& operator=(const OrPredOp &rhs);
    public:
      void initialize(TaskContext *task, 
                      const std::vector<Predicate> &predicates);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void notify_predicate_value(GenerationID pred_gen, bool value);
    protected:
      std::vector<PredicateOp*> previous;
      unsigned                  false_count;
      bool                      true_short;
    };

    /**
     * \class MustEpochOp
     * This operation is actually a meta-operation that
     * represents a collection of operations which all
     * must be guaranteed to be run in parallel.  It
     * mediates all the various stages of performing
     * these operations and ensures that they can all
     * be run in parallel or it reports an error.
     */
    class MustEpochOp : public Operation, public LegionHeapify<MustEpochOp> {
    public:
      static const AllocationType alloc_type = MUST_EPOCH_OP_ALLOC;
    public:
      struct DependenceRecord {
      public:
        inline void add_entry(unsigned op_idx, unsigned req_idx)
          { op_indexes.push_back(op_idx); req_indexes.push_back(req_idx); }
      public:
        std::vector<unsigned> op_indexes;
        std::vector<unsigned> req_indexes;
      };
    public:
      MustEpochOp(Runtime *rt);
      MustEpochOp(const MustEpochOp &rhs);
      virtual ~MustEpochOp(void);
    public:
      MustEpochOp& operator=(const MustEpochOp &rhs);
    public:
      FutureMap initialize(TaskContext *ctx,
                           const MustEpochLauncher &launcher,
                           bool check_privileges);
      void find_conflicted_regions(
          std::vector<PhysicalRegion> &unmapped); 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    public:
      void verify_dependence(Operation *source_op, GenerationID source_gen,
                             Operation *target_op, GenerationID target_gen);
      bool record_dependence(Operation *source_op, GenerationID source_gen,
                             Operation *target_op, GenerationID target_gen,
                             unsigned source_idx, unsigned target_idx,
                             DependenceType dtype);
      void must_epoch_map_task_callback(SingleTask *task, 
                                        Mapper::MapTaskInput &input,
                                        Mapper::MapTaskOutput &output);
      // Get a reference to our data structure for tracking acquired instances
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                                       get_acquired_instances_ref(void);
    public:
      void add_mapping_dependence(RtEvent precondition);
      void register_single_task(SingleTask *single, unsigned index);
      void register_slice_task(SliceTask *slice);
      void set_future(const DomainPoint &point, 
                      const void *result, size_t result_size, bool owned);
      void unpack_future(const DomainPoint &point, Deserializer &derez);
    public:
      // Methods for keeping track of when we can complete and commit
      void register_subop(Operation *op);
      void notify_subop_complete(Operation *op);
      void notify_subop_commit(Operation *op);
    public:
      RtUserEvent find_slice_versioning_event(UniqueID slice_id, bool &first);
    protected:
      int find_operation_index(Operation *op, GenerationID generation);
      TaskOp* find_task_by_index(int index);
    protected:
      std::vector<IndividualTask*>        indiv_tasks;
      std::vector<bool>                   indiv_triggered;
      std::vector<IndexTask*>             index_tasks;
      std::vector<bool>                   index_triggered;
    protected:
      // The component slices for distribution
      std::set<SliceTask*>         slice_tasks;
      // The actual base operations
      // Use a deque to keep everything in order
      std::deque<SingleTask*>      single_tasks;
    protected:
      Mapper::MapMustEpochInput    input;
      Mapper::MapMustEpochOutput   output;
      MapperID                     mapper_id;
      MappingTagID                 mapper_tag;
    protected:
      FutureMap result_map;
      unsigned remaining_subop_completes;
      unsigned remaining_subop_commits;
    protected:
      // Used to know if we successfully triggered everything
      // and therefore have all of the single tasks and a
      // valid set of constraints.
      bool triggering_complete;
      // Used for computing the constraints
      std::vector<std::set<SingleTask*> > task_sets;
      // Track the physical instances that we've acquired
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
    protected:
      std::map<std::pair<unsigned/*task index*/,unsigned/*req index*/>,
               unsigned/*dependence index*/> dependence_map;
      std::vector<DependenceRecord*> dependences;
      std::map<SingleTask*,unsigned/*single task index*/> single_task_map;
      std::vector<std::set<unsigned/*single task index*/> > mapping_dependences;
    protected:
      std::map<UniqueID,RtUserEvent> slice_version_events;
    };

    /**
     * \class MustEpochTriggerer
     * A helper class for parallelizing must epoch triggering
     */
    class MustEpochTriggerer {
    public:
      struct MustEpochIndivArgs : public LgTaskArgs<MustEpochIndivArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MUST_INDIV_ID;
      public:
        MustEpochIndivArgs(MustEpochTriggerer *trig, MustEpochOp *owner,
                           IndividualTask *t)
          : LgTaskArgs<MustEpochIndivArgs>(owner->get_unique_op_id()),
            triggerer(trig), task(t) { }
      public:
        MustEpochTriggerer *const triggerer;
        IndividualTask *const task;
      };
      struct MustEpochIndexArgs : public LgTaskArgs<MustEpochIndexArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MUST_INDEX_ID;
      public:
        MustEpochIndexArgs(MustEpochTriggerer *trig, MustEpochOp *owner,
                           IndexTask *t)
          : LgTaskArgs<MustEpochIndexArgs>(owner->get_unique_op_id()),
            triggerer(trig), task(t) { }
      public:
        MustEpochTriggerer *const triggerer;
        IndexTask *const task;
      };
    public:
      MustEpochTriggerer(MustEpochOp *owner);
      MustEpochTriggerer(const MustEpochTriggerer &rhs);
      ~MustEpochTriggerer(void);
    public:
      MustEpochTriggerer& operator=(const MustEpochTriggerer &rhs);
    public:
      void trigger_tasks(const std::vector<IndividualTask*> &indiv_tasks,
                         std::vector<bool> &indiv_triggered,
                         const std::vector<IndexTask*> &index_tasks,
                         std::vector<bool> &index_triggered);
      void trigger_individual(IndividualTask *task);
      void trigger_index(IndexTask *task);
    public:
      static void handle_individual(const void *args);
      static void handle_index(const void *args);
    private:
      const Processor current_proc;
      MustEpochOp *const owner;
      Reservation trigger_lock;
    };

    /**
     * \class MustEpochMapper
     * A helper class for parallelizing mapping for must epochs
     */
    class MustEpochMapper {
    public:
      struct MustEpochMapArgs : public LgTaskArgs<MustEpochMapArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MUST_MAP_ID;
      public:
        MustEpochMapArgs(MustEpochMapper *map, MustEpochOp *owner)
          : LgTaskArgs<MustEpochMapArgs>(owner->get_unique_op_id()),
            mapper(map) { }
      public:
        MustEpochMapper *const mapper;
        SingleTask *task;
      };
    public:
      MustEpochMapper(MustEpochOp *owner);
      MustEpochMapper(const MustEpochMapper &rhs);
      ~MustEpochMapper(void);
    public:
      MustEpochMapper& operator=(const MustEpochMapper &rhs);
    public:
      void map_tasks(const std::deque<SingleTask*> &single_tasks,
            const std::vector<std::set<unsigned> > &dependences);
      void map_task(SingleTask *task);
    public:
      static void handle_map_task(const void *args);
    private:
      MustEpochOp *const owner;
    };

    class MustEpochDistributor {
    public:
      struct MustEpochDistributorArgs : 
        public LgTaskArgs<MustEpochDistributorArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MUST_DIST_ID;
      public:
        MustEpochDistributorArgs(MustEpochOp *owner)
          : LgTaskArgs<MustEpochDistributorArgs>(owner->get_unique_op_id()) { }
      public:
        TaskOp *task;
      };
      struct MustEpochLauncherArgs : 
        public LgTaskArgs<MustEpochLauncherArgs> {
      public:
        static const LgTaskID TASK_ID = LG_MUST_LAUNCH_ID;
      public:
        MustEpochLauncherArgs(MustEpochOp *owner)
          : LgTaskArgs<MustEpochLauncherArgs>(owner->get_unique_op_id()) { }
      public:
        TaskOp *task;
      };
    public:
      MustEpochDistributor(MustEpochOp *owner);
      MustEpochDistributor(const MustEpochDistributor &rhs);
      ~MustEpochDistributor(void);
    public:
      MustEpochDistributor& operator=(const MustEpochDistributor &rhs);
    public:
      void distribute_tasks(Runtime *runtime,
                            const std::vector<IndividualTask*> &indiv_tasks,
                            const std::set<SliceTask*> &slice_tasks);
    public:
      static void handle_distribute_task(const void *args);
      static void handle_launch_task(const void *args);
    private:
      MustEpochOp *const owner;
    };

    /**
     * \class PendingPartitionOp
     * Pending partition operations are ones that must be deferred
     * in order to move the overhead of computing them off the 
     * application cores. In many cases deferring them is also
     * necessary to avoid possible application deadlock with
     * other pending partitions.
     */
    class PendingPartitionOp : public Operation,
                               public LegionHeapify<PendingPartitionOp> {
    public:
      static const AllocationType alloc_type = PENDING_PARTITION_OP_ALLOC;
    protected:
      enum PendingPartitionKind
      {
        EQUAL_PARTITION = 0,
        UNION_PARTITION,
        INTERSECTION_PARTITION,
        DIFFERENCE_PARTITION,
        RESTRICTED_PARTITION,
      };
      // Track pending partition operations as thunks
      class PendingPartitionThunk {
      public:
        virtual ~PendingPartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest) = 0;
        virtual void perform_logging(PendingPartitionOp* op) = 0;
      };
      class EqualPartitionThunk : public PendingPartitionThunk {
      public:
        EqualPartitionThunk(IndexPartition id, size_t g)
          : pid(id), granularity(g) { }
        virtual ~EqualPartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_equal_partition(op, pid, granularity); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        size_t granularity;
      };
      class UnionPartitionThunk : public PendingPartitionThunk {
      public:
        UnionPartitionThunk(IndexPartition id, 
                            IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2) { }
        virtual ~UnionPartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_union(op, pid, handle1, handle2); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class IntersectionPartitionThunk : public PendingPartitionThunk {
      public:
        IntersectionPartitionThunk(IndexPartition id, 
                            IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2) { }
        virtual ~IntersectionPartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_intersection(op, pid, handle1,
                                                          handle2); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class DifferencePartitionThunk : public PendingPartitionThunk {
      public:
        DifferencePartitionThunk(IndexPartition id, 
                            IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2) { }
        virtual ~DifferencePartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_difference(op, pid, handle1,
                                                        handle2); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class RestrictedPartitionThunk : public PendingPartitionThunk {
      public:
        RestrictedPartitionThunk(IndexPartition id, const void *tran,
                  size_t tran_size, const void *ext, size_t ext_size)
          : pid(id), transform(malloc(tran_size)), extent(malloc(ext_size))
        { memcpy(transform, tran, tran_size); memcpy(extent, ext, ext_size); }
        virtual ~RestrictedPartitionThunk(void)
          { free(transform); free(extent); }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_restriction(pid, 
                                              transform, extent); }
        virtual void perform_logging(PendingPartitionOp *op);
      protected:
        IndexPartition pid;
        void *const transform;
        void *const extent;
      };
      class CrossProductThunk : public PendingPartitionThunk {
      public:
        CrossProductThunk(IndexPartition b, IndexPartition s, LegionColor c)
          : base(b), source(s), part_color(c) { }
        virtual ~CrossProductThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_cross_product_partitions(op, base, source, 
                                                         part_color); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition base;
        IndexPartition source;
        LegionColor part_color;
      };
      class ComputePendingSpace : public PendingPartitionThunk {
      public:
        ComputePendingSpace(IndexSpace t, bool is,
                            const std::vector<IndexSpace> &h)
          : is_union(is), is_partition(false), target(t), handles(h) { }
        ComputePendingSpace(IndexSpace t, bool is, IndexPartition h)
          : is_union(is), is_partition(true), target(t), handle(h) { }
        virtual ~ComputePendingSpace(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { if (is_partition)
            return forest->compute_pending_space(op, target, handle, is_union);
          else
            return forest->compute_pending_space(op, target, 
                                                 handles, is_union); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        bool is_union, is_partition;
        IndexSpace target;
        IndexPartition handle;
        std::vector<IndexSpace> handles;
      };
      class ComputePendingDifference : public PendingPartitionThunk {
      public:
        ComputePendingDifference(IndexSpace t, IndexSpace i,
                                 const std::vector<IndexSpace> &h)
          : target(t), initial(i), handles(h) { }
        virtual ~ComputePendingDifference(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->compute_pending_space(op, target, initial, handles); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexSpace target, initial;
        std::vector<IndexSpace> handles;
      };
    public:
      PendingPartitionOp(Runtime *rt);
      PendingPartitionOp(const PendingPartitionOp &rhs);
      virtual ~PendingPartitionOp(void);
    public:
      PendingPartitionOp& operator=(const PendingPartitionOp &rhs);
    public:
      void initialize_equal_partition(TaskContext *ctx,
                                      IndexPartition pid, size_t granularity);
      void initialize_union_partition(TaskContext *ctx,
                                      IndexPartition pid, 
                                      IndexPartition handle1,
                                      IndexPartition handle2);
      void initialize_intersection_partition(TaskContext *ctx,
                                             IndexPartition pid, 
                                             IndexPartition handle1,
                                             IndexPartition handle2);
      void initialize_difference_partition(TaskContext *ctx,
                                           IndexPartition pid, 
                                           IndexPartition handle1,
                                           IndexPartition handle2);
      void initialize_restricted_partition(TaskContext *ctx,
                                           IndexPartition pid,
                                           const void *transform,
                                           size_t transform_size,
                                           const void *extent,
                                           size_t extent_size);
      void initialize_cross_product(TaskContext *ctx, IndexPartition base, 
                                    IndexPartition source, LegionColor color);
      void initialize_index_space_union(TaskContext *ctx, IndexSpace target, 
                                        const std::vector<IndexSpace> &handles);
      void initialize_index_space_union(TaskContext *ctx, IndexSpace target, 
                                        IndexPartition handle);
      void initialize_index_space_intersection(TaskContext *ctx, 
                                               IndexSpace target,
                                        const std::vector<IndexSpace> &handles);
      void initialize_index_space_intersection(TaskContext *ctx,
                                              IndexSpace target,
                                              IndexPartition handle);
      void initialize_index_space_difference(TaskContext *ctx, 
                                             IndexSpace target, 
                                             IndexSpace initial,
                                        const std::vector<IndexSpace> &handles);
      void perform_logging();
    public:
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual bool is_partition_op(void) const { return true; } 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    protected:
      PendingPartitionThunk *thunk;
    };

    /**
     * \class DependentPartitionOp
     * An operation for creating different kinds of partitions
     * which are dependent on mapping a region in order to compute
     * the resulting partition.
     */
    class DependentPartitionOp : public Partition, public Operation,
                                 public LegionHeapify<DependentPartitionOp> {
    public:
      static const AllocationType alloc_type = DEPENDENT_PARTITION_OP_ALLOC;
    protected:
      // Track dependent partition operations as thunks
      class DepPartThunk {
      public:
        virtual ~DepPartThunk(void) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances) = 0;
        virtual PartitionKind get_kind(void) const = 0;
        virtual IndexPartition get_partition(void) const = 0;
      };
      class ByFieldThunk : public DepPartThunk {
      public:
        ByFieldThunk(IndexPartition p)
          : pid(p) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_FIELD; }
        virtual IndexPartition get_partition(void) const { return pid; }
      protected:
        IndexPartition pid;
      };
      class ByImageThunk : public DepPartThunk {
      public:
        ByImageThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_IMAGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByImageRangeThunk : public DepPartThunk {
      public:
        ByImageRangeThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_IMAGE_RANGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByPreimageThunk : public DepPartThunk {
      public:
        ByPreimageThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_PREIMAGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class ByPreimageRangeThunk : public DepPartThunk {
      public:
        ByPreimageRangeThunk(IndexPartition p, IndexPartition proj)
          : pid(p), projection(proj) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_PREIMAGE_RANGE; }
        virtual IndexPartition get_partition(void) const { return pid; }
      protected:
        IndexPartition pid;
        IndexPartition projection;
      };
      class AssociationThunk : public DepPartThunk {
      public:
        AssociationThunk(IndexSpace d, IndexSpace r)
          : domain(d), range(r) { }
      public:
        virtual ApEvent perform(DependentPartitionOp *op,
            RegionTreeForest *forest, ApEvent instances_ready,
            const std::vector<FieldDataDescriptor> &instances);
        virtual PartitionKind get_kind(void) const { return BY_ASSOCIATION; }
        virtual IndexPartition get_partition(void) const
          { return IndexPartition::NO_PART; }
      protected:
        IndexSpace domain;
        IndexSpace range;
      };
    public:
      DependentPartitionOp(Runtime *rt);
      DependentPartitionOp(const DependentPartitionOp &rhs);
      virtual ~DependentPartitionOp(void);
    public:
      DependentPartitionOp& operator=(const DependentPartitionOp &rhs);
    public:
      void initialize_by_field(TaskContext *ctx, IndexPartition pid,
                               LogicalRegion handle, LogicalRegion parent,
                               FieldID fid, MapperID id, MappingTagID tag); 
      void initialize_by_image(TaskContext *ctx, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      void initialize_by_image_range(TaskContext *ctx, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      void initialize_by_preimage(TaskContext *ctx, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      void initialize_by_preimage_range(TaskContext *ctx, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag);
      void initialize_by_association(TaskContext *ctx, LogicalRegion domain,
                               LogicalRegion domain_parent, FieldID fid,
                               IndexSpace range, MapperID id, MappingTagID tag);
      void perform_logging(void) const;
      void log_requirement(void) const;
      const RegionRequirement& get_requirement(void) const;
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual ApEvent trigger_thunk(IndexSpace handle,
                                    const InstanceSet &mapped_instances);
      virtual unsigned find_parent_index(unsigned idx);
      virtual bool is_partition_op(void) const { return true; }
    public:
      virtual PartitionKind get_partition_kind(void) const;
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual void trigger_commit(void);
    public:
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual PhysicalManager* select_temporary_instance(PhysicalManager *dst,
                              unsigned index, const FieldMask &needed_fields);
      virtual void record_restrict_postcondition(ApEvent postcondition);
      virtual void add_copy_profiling_request(
                                        Realm::ProfilingRequestSet &reqeusts);
      // Report a profiling result for this operation
      virtual void handle_profiling_response(
                                  const Realm::ProfilingResponse &result);
    protected:
      void compute_parent_index(void);
      void select_partition_projection(void);
      void invoke_mapper(const InstanceSet &valid_instances,
                               InstanceSet &mapped_instances);
      void activate_dependent_op(void);
      void deactivate_dependent_op(void);
    public:
      void handle_point_commit(RtEvent point_committed);
    public:
      ProjectionInfo projection_info;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> restricted_postconditions;
      DepPartThunk *thunk;
    protected:
      MapperManager *mapper;
    protected:
      // For index versions of this operation
      IndexSpace                        launch_space;
      std::vector<FieldDataDescriptor>  instances;
      std::set<ApEvent>                 index_preconditions;
      std::vector<PointDepPartOp*>      points; 
      unsigned                          points_committed;
      bool                              commit_request;
      std::set<RtEvent>                 commit_preconditions;
#ifdef LEGION_SPY
      // Special helper event to make things look right for Legion Spy
      ApUserEvent                       intermediate_index_event;
#endif
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                     outstanding_profiling_requests;
      RtUserEvent                         profiling_reported;
    };

    /**
     * \class PointDepPartOp
     * This is a point class for mapping a particular 
     * subregion of a partition for a dependent partitioning
     * operation.
     */
    class PointDepPartOp : public DependentPartitionOp, public ProjectionPoint {
    public:
      PointDepPartOp(Runtime *rt);
      PointDepPartOp(const PointDepPartOp &rhs);
      virtual ~PointDepPartOp(void);
    public:
      PointDepPartOp& operator=(const PointDepPartOp &rhs);
    public:
      void initialize(DependentPartitionOp *owner, const DomainPoint &point);
      void launch(const std::set<RtEvent> &preconditions);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual ApEvent trigger_thunk(IndexSpace handle,
                                    const InstanceSet &mapped_instances);
      virtual void trigger_commit(void);
    public:
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
    public:
      DependentPartitionOp *owner;
    };

    /**
     * \class FillOp
     * Fill operations are used to initialize a field to a
     * specific value for a particular logical region.
     */
    class FillOp : public MemoizableOp<SpeculativeOp>, public Fill,
                   public LegionHeapify<FillOp> {
    public:
      static const AllocationType alloc_type = FILL_OP_ALLOC;
    public:
      FillOp(Runtime *rt);
      FillOp(const FillOp &rhs);
      virtual ~FillOp(void);
    public:
      FillOp& operator=(const FillOp &rhs);
    public:
      void initialize(TaskContext *ctx, const FillLauncher &launcher,
                      bool check_privileges);
      inline const RegionRequirement& get_requirement(void) const 
        { return requirement; }
      void activate_fill(void);
      void deactivate_fill(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual Mappable* get_mappable(void);
      virtual UniqueID get_unique_id(void) const;
      virtual unsigned get_context_index(void) const;
      virtual int get_depth(void) const;
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void deferred_execute(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_commit(void);
      virtual ApEvent get_restrict_precondition(void) const;
      virtual const ProjectionInfo* get_projection_info(void);
    public:
      void check_fill_privilege(void);
      void compute_parent_index(void);
      ApEvent compute_sync_precondition(void) const;
      void log_fill_requirement(void) const;
    public:
      // From MemoizableOp
      virtual void replay_analysis(void);
    public:
      RegionTreePath privilege_path;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      unsigned parent_req_index;
      void *value;
      size_t value_size;
      Future future;
      std::set<RtEvent> map_applied_conditions;
      PredEvent true_guard, false_guard;
    };
    
    /**
     * \class IndexFillOp
     * This is the same as a fill operation except for
     * applying a number of fill operations over an 
     * index space of points with projection functions.
     */
    class IndexFillOp : public FillOp {
    public:
      IndexFillOp(Runtime *rt);
      IndexFillOp(const IndexFillOp &rhs);
      virtual ~IndexFillOp(void);
    public:
      IndexFillOp& operator=(const IndexFillOp &rhs);
    public:
      void initialize(TaskContext *ctx,
                      const IndexFillLauncher &launcher,
                      IndexSpace launch_space,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_commit(void);
    public:
      void handle_point_commit(void);
#ifdef DEBUG_LEGION
      void check_point_requirements(void);
#endif
    public:
      virtual const ProjectionInfo* get_projection_info(void);
    public:
      ProjectionInfo                projection_info;
      IndexSpace                    launch_space;
    protected:
      std::vector<PointFillOp*>     points;
      unsigned                      points_committed;
      bool                          commit_request;
    };

    /**
     * \class PointFillOp
     * A point fill op is used for executing the
     * physical part of the analysis for an index
     * fill operation.
     */
    class PointFillOp : public FillOp, public ProjectionPoint {
    public:
      PointFillOp(Runtime *rt);
      PointFillOp(const PointFillOp &rhs);
      virtual ~PointFillOp(void);
    public:
      PointFillOp& operator=(const PointFillOp &rhs);
    public:
      void initialize(IndexFillOp *owner, const DomainPoint &point);
      void launch(const std::set<RtEvent> &preconditions);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      // trigger_mapping same as base class
      virtual void trigger_commit(void);
    public:
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx,LogicalRegion result);
    public:
      virtual const ProjectionInfo* get_projection_info(void);
    protected:
      IndexFillOp*              owner;
    };

    /**
     * \class AttachOp
     * Operation for attaching a file to a physical instance
     */
    class AttachOp : public Operation, public LegionHeapify<AttachOp> {
    public:
      static const AllocationType alloc_type = ATTACH_OP_ALLOC;
    public:
      AttachOp(Runtime *rt);
      AttachOp(const AttachOp &rhs);
      virtual ~AttachOp(void);
    public:
      AttachOp& operator=(const AttachOp &rhs);
    public:
      PhysicalRegion initialize(TaskContext *ctx,
                                const AttachLauncher &launcher,
                                bool check_privileges);
      inline const RegionRequirement& get_requirement(void) const 
        { return requirement; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_commit(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      PhysicalInstance create_instance(IndexSpaceNode *node,
                                       const std::vector<FieldID> &field_set,
                                       const std::vector<size_t> &field_sizes,
                                             LayoutConstraintSet &cons,
                                             ApEvent &ready_event);
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
    public:
      ExternalResource resource;
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      const char *file_name;
      std::map<FieldID,const char*> field_map;
      std::map<FieldID,void*> field_pointers_map;
      LegionFileMode file_mode;
      PhysicalRegion region;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
      InstanceManager *external_instance;
      LayoutConstraintSet layout_constraint_set;
    };

    /**
     * \class DetachOp
     * Operation for detaching a file from a physical instance
     */
    class DetachOp : public Operation, public LegionHeapify<DetachOp> {
    public:
      static const AllocationType alloc_type = DETACH_OP_ALLOC;
    public:
      DetachOp(Runtime *rt);
      DetachOp(const DetachOp &rhs);
      virtual ~DetachOp(void);
    public:
      DetachOp& operator=(const DetachOp &rhs);
    public:
      Future initialize_detach(TaskContext *ctx, PhysicalRegion region);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual bool has_prepipeline_stage(void) const { return true; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    protected:
      void compute_parent_index(void);
    public:
      PhysicalRegion region;
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      unsigned parent_req_index;
      Future result;
    };

    /**
     * \class TimingOp
     * Operation for performing timing measurements
     */
    class TimingOp : public Operation {
    public:
      TimingOp(Runtime *rt);
      TimingOp(const TimingOp &rhs);
      virtual ~TimingOp(void);
    public:
      TimingOp& operator=(const TimingOp &rhs);
    public:
      Future initialize(TaskContext *ctx, const TimingLauncher &launcher);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void deferred_execute(void);
      virtual void trigger_complete(void);
    protected:
      TimingMeasurement measurement;
      std::set<Future> preconditions;
      Future result;
    };

  }; //namespace Internal 
}; // namespace Legion 

#include "legion_ops.inl"

#endif // __LEGION_OPERATIONS_H__
