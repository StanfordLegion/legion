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


#ifndef __LEGION_OPERATIONS_H__
#define __LEGION_OPERATIONS_H__

#include "legion.h"
#include "legion/runtime.h"
#include "legion/region_tree.h"
#include "legion/legion_mapping.h"
#include "legion/legion_utilities.h"
#include "legion/legion_allocation.h"
#include "legion/legion_instances.h"
#include "legion/legion_analysis.h"
#include "legion/mapper_manager.h"

namespace Legion {
  namespace Internal {

    // Special typedef for predicates
    typedef PredicateImpl PredicateOp;  

    /**
     * \class Provenance
     */
    class Provenance : public Collectable {
    public:
      Provenance(const char *prov) : provenance(prov) { }
      Provenance(const void *buffer, size_t size)
        : provenance((const char*)buffer, size) { }
      Provenance(const Provenance &rhs) = delete;
      ~Provenance(void) { }
    public:
      Provenance& operator=(const Provenance &rhs) = delete;
    public:
      void serialize(Serializer &rez) const;
      static void serialize_null(Serializer &rez);
      static Provenance* deserialize(Deserializer &derez);
    public:
      const std::string provenance;
    };

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
        CREATION_OP_KIND,
        DELETION_OP_KIND,
        MERGE_CLOSE_OP_KIND,
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
        TUNABLE_OP_KIND,
        ALL_REDUCE_OP_KIND,
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
        "Creation",                 \
        "Deletion",                 \
        "Merge Close",              \
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
        "Tunable",                  \
        "All Reduce Op",            \
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
      struct DeferReleaseAcquiredArgs : 
        public LgTaskArgs<DeferReleaseAcquiredArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_RELEASE_ACQUIRED_TASK_ID;
      public:
        DeferReleaseAcquiredArgs(Operation *op, 
            std::vector<std::pair<PhysicalManager*,unsigned> > *insts)
          : LgTaskArgs<DeferReleaseAcquiredArgs>(op->get_unique_op_id()),
            instances(insts) { }
      public:
        std::vector<std::pair<PhysicalManager*,unsigned> > *const instances;
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
      struct OpProfilingResponse : public ProfilingResponseBase {
      public:
        OpProfilingResponse(ProfilingResponseHandler *h, 
                            unsigned s, unsigned d, bool f, bool t = false)
          : ProfilingResponseBase(h), src(s), dst(d), fill(f), task(t) { }
      public:
        unsigned src, dst;
        bool fill;
        bool task;
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
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { exec_fence = false; return true; }
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
      inline InnerContext* get_context(void) const { return parent_ctx; }
      inline UniqueID get_unique_op_id(void) const { return unique_op_id; } 
      virtual bool is_memoizing(void) const { return false; }
      inline bool is_tracing(void) const { return tracing; }
      inline bool is_tracking_parent(void) const { return track_parent; } 
      inline bool already_traced(void) const 
        { return ((trace != NULL) && !tracing); }
      inline LegionTrace* get_trace(void) const { return trace; }
      inline size_t get_ctx_index(void) const { return context_index; }
      inline Provenance* get_provenance(void) const 
        { return provenance; }
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
      void set_tracking_parent(size_t index);
      void set_trace(LegionTrace *trace,
                     const std::vector<StaticDependence> *dependences,
                     const LogicalTraceInfo *trace_info = NULL);
      void set_trace_local_id(unsigned id);
      void set_must_epoch(MustEpochOp *epoch, bool do_registration);
    public:
      // Localize a region requirement to its parent context
      // This means that region == parent and the
      // coherence mode is exclusive
      static void localize_region_requirement(RegionRequirement &req);
      // We want to release our valid references for mapping as soon as
      // possible after mapping is done so the garbage collector can do
      // deferred collection ASAP if it needs to. However, there is a catch:
      // instances which are empty have no GC references from the physical
      // analysis to protect them from collection. That's not a problem for
      // the GC, but it is for keeping their meta-data structures alive.
      // Our solution is just to keep the valid references on the emtpy
      // acquired instances until the very end of the operation as they
      // will not hurt anything.
      RtEvent release_nonempty_acquired_instances(RtEvent precondition,
          std::map<PhysicalManager*,unsigned> &acquired_insts);
      static void release_acquired_instances(
          std::map<PhysicalManager*,unsigned> &acquired_insts);
      static void handle_deferred_release(const void *args);
    public:
      // Initialize this operation in a new parent context
      // along with the number of regions this task has
      void initialize_operation(InnerContext *ctx, bool track,
                                unsigned num_regions = 0,
                                const char *prov = NULL,
          const std::vector<StaticDependence> *dependences = NULL);
      void initialize_operation(InnerContext *ctx, bool track,
                                unsigned num_regions,
                                Provenance *provenance,
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
      // Helper function for trigger execution 
      // (only used in a limited set of operations and not
      // part of the default pipeline)
      virtual void trigger_execution(void);
      // The function to trigger once speculation is
      // ready to be resolved
      virtual void trigger_resolution(void);
      // The function to call once the operation is ready to complete
      virtual void trigger_complete(void);
      // The function to call when commit the operation is
      // ready to commit
      virtual void trigger_commit(void);
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
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
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
      virtual void report_uninitialized_usage(const unsigned index,
                                              LogicalRegion handle,
                                              const RegionUsage usage,
                                              const char *field_string,
                                              RtUserEvent reported);
      // Get a reference to our data structure for tracking acquired instances
      virtual std::map<PhysicalManager*,unsigned>*
                                       get_acquired_instances_ref(void);
      // Update the set of atomic locks for this operation
      virtual void update_atomic_locks(const unsigned index, 
                                       Reservation lock, bool exclusive);
      // Get the restrict precondition for this operation
      static ApEvent merge_sync_preconditions(const TraceInfo &info,
                                const std::vector<Grant> &grants,
                                const std::vector<PhaseBarrier> &wait_barriers);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests, 
                               bool fill, unsigned count = 1);
      // Report a profiling result for this operation
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                        const Realm::ProfilingResponse &result,
                                        const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      // Compute the initial precondition for this operation
      virtual ApEvent compute_init_precondition(const TraceInfo &info);
      // Return the event to use for waiting for program order execution
      virtual ApEvent get_program_order_event(void) const 
        { return completion_event; }
    protected:
      void filter_copy_request_kinds(MapperManager *mapper,
          const std::set<ProfilingMeasurementID> &requests,
          std::vector<ProfilingMeasurementID> &results, bool warn_if_not_copy);
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
#ifdef DEBUG_LEGION
            assert(need_completion_trigger);
#endif
            need_completion_trigger = false;
            Runtime::trigger_event(NULL, completion_event, chain_event);
            return true;
          }
          else
            return false;
        }
      inline bool request_early_complete_no_trigger(ApUserEvent &to_trigger)
        {
          if (!runtime->program_order_execution)
          {
#ifdef DEBUG_LEGION
            assert(need_completion_trigger);
#endif
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
      // This function should only be called when we are tracing.
      // We record other possible interferences with prior operations
      // that are only not-interfering for privileges in case we make
      // an internal operation that we did not have before.
      void register_no_dependence(unsigned idx, Operation *target,
                              GenerationID target_gen, unsigned target_idx,
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
      inline LegionList<LogicalUser,LOGICAL_REC_ALLOC>&
          get_logical_records(void) { return logical_records; }
      void clear_logical_records(void);
    public:
      // Notify when a region from a dependent task has 
      // been verified (flows up edges)
      void notify_regions_verified(const std::set<unsigned> &regions,
                                   GenerationID gen);
    public:
      // Help for finding the contexts for an operation
      InnerContext* find_logical_context(unsigned index);
      InnerContext* find_physical_context(unsigned index,
                                          const RegionRequirement &req);
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
#ifdef DEBUG_LEGION
    protected:
      virtual void dump_physical_state(RegionRequirement *req, unsigned idx,
                                       bool before = false,
                                       bool closing = false);
#endif
    public:
      // Pack the needed parts of this operation for a remote operation
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      void pack_local_remote_operation(Serializer &rez) const;
    protected:
      static inline void add_launch_space_reference(IndexSpaceNode *node)
      {
        LocalReferenceMutator mutator;
        node->add_base_valid_ref(CONTEXT_REF, &mutator);
      }
      static inline bool remove_launch_space_reference(IndexSpaceNode *node)
      {
        if (node == NULL)
          return false;
        return node->remove_base_valid_ref(CONTEXT_REF);
      }
    public:
      Runtime *const runtime;
    protected:
      mutable LocalLock op_lock;
      GenerationID gen;
      UniqueID unique_op_id;
      // The issue index of this operation in the context
      size_t context_index;
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
      // Whether this operation is active or not
      bool activated;
      // Whether this operation has executed its prepipeline stage yet
      bool prepipelined;
      // Whether this operation has mapped, once it has mapped then
      // the set of incoming dependences is fixed
      bool mapped;
      // Whether this task has executed or not
      bool executed;
      // Whether speculation for this operation has been resolved
      bool resolved;
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
      InnerContext *parent_ctx;
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
      LegionList<LogicalUser,LOGICAL_REC_ALLOC> logical_records;
      // Dependence trackers for detecting when it is safe to map and commit
      // We allocate and free these every time to ensure that their memory
      // is always cleaned up after each operation
      MappingDependenceTracker *mapping_tracker;
      CommitDependenceTracker  *commit_tracker;
      // Provenance information for this operation
      Provenance *provenance;
    };

    /**
     * \class CollectiveInstanceCreator
     * This class provides a common base class for operations that need to 
     * provide support for the creation of collective instances
     */
    template<typename OP>
    class CollectiveInstanceCreator : public OP {
    public:
      CollectiveInstanceCreator(Runtime *rt);
      CollectiveInstanceCreator(const CollectiveInstanceCreator<OP> &rhs);
    public:
      virtual IndexSpaceNode* get_collective_space(void) const = 0;
    public:
      // For collective instances
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
    protected:
      typedef std::pair<MappingCallKind,unsigned> CollectiveKey;
      struct CollectiveInstance {
      public:
        CollectiveInstance(void) : manager(NULL), remaining(0), pending(0) { }
      public:
        CollectiveManager *manager;
        RtUserEvent ready_event;
        ApUserEvent instance_event;
        size_t remaining;
        size_t pending;
      };
      std::map<CollectiveKey,CollectiveInstance> collective_instances;
    };

    /**
     * \class ExternalMappable
     * This is class that provides some basic functionality for
     * packing and unpacking the data structures used by 
     * external facing operations
     */
    class ExternalMappable {
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      static void pack_mappable(const Mappable &mappable, Serializer &rez);
      static void pack_index_space_requirement(
          const IndexSpaceRequirement &req, Serializer &rez);
      static void pack_region_requirement(
          const RegionRequirement &req, Serializer &rez);
      static void pack_grant(
          const Grant &grant, Serializer &rez);
      static void pack_phase_barrier(
          const PhaseBarrier &barrier, Serializer &rez);
    public:
      static void unpack_mappable(Mappable &mappable, Deserializer &derez);
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
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { return false; }
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
      void initialize_speculation(InnerContext *ctx,bool track,unsigned regions,
          const std::vector<StaticDependence> *dependences, const Predicate &p,
          const char *provenance);
      void initialize_speculation(InnerContext *ctx,bool track,unsigned regions,
          const std::vector<StaticDependence> *dependences, const Predicate &p,
          Provenance *provenance);
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
     * \class Memoizable
     * An abstract class for retrieving trace local ids in physical tracing.
     */
    class Memoizable {
    public:
      virtual ~Memoizable(void) { }
      virtual bool is_memoizable_task(void) const = 0;
      virtual bool is_recording(void) const = 0;
      virtual bool is_memoizing(void) const = 0;
      virtual AddressSpaceID get_origin_space(void) const = 0;
      virtual PhysicalTemplate* get_template(void) const = 0;
      virtual ApEvent get_memo_completion(void) const = 0;
      virtual void replay_mapping_output(void) = 0;
      virtual Operation* get_operation(void) const = 0;
      virtual Operation::OpKind get_memoizable_kind(void) const = 0;
      // Return a trace local unique ID for this operation
      virtual TraceLocalID get_trace_local_id(void) const = 0;
      virtual ApEvent compute_sync_precondition(const TraceInfo *in) const = 0;
      virtual void set_effects_postcondition(ApEvent postcondition) = 0;
      virtual void complete_replay(ApEvent complete_event) = 0;
      virtual void find_equivalence_sets(Runtime *runtime, unsigned idx,
        const FieldMask &mask, FieldMaskSet<EquivalenceSet> &target) const = 0;
    protected:
      virtual const VersionInfo& get_version_info(unsigned idx) const = 0;
    public:
      virtual void pack_remote_memoizable(Serializer &rez, 
                                          AddressSpaceID target) const;
      virtual Memoizable* clone(Operation *op) { return this; }
    };

    class RemoteMemoizable : public Memoizable {
    public:
      RemoteMemoizable(Operation *op, Memoizable *original, 
                       AddressSpaceID origin, Operation::OpKind kind,
                       TraceLocalID tid, ApEvent completion_event,
                       bool is_memoizable_task, bool is_memoizing);
      virtual ~RemoteMemoizable(void);
    public:
      virtual bool is_memoizable_task(void) const;
      virtual bool is_recording(void) const;
      virtual bool is_memoizing(void) const;
      virtual AddressSpaceID get_origin_space(void) const;
      virtual PhysicalTemplate* get_template(void) const;
      virtual ApEvent get_memo_completion(void) const;
      virtual void replay_mapping_output(void);
      virtual Operation* get_operation(void) const;
      virtual Operation::OpKind get_memoizable_kind(void) const;
      // Return a trace local unique ID for this operation
      typedef std::pair<unsigned, DomainPoint> TraceLocalID;
      virtual TraceLocalID get_trace_local_id(void) const;
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const;
      virtual void set_effects_postcondition(ApEvent postcondition);
      virtual void complete_replay(ApEvent complete_event);
      virtual void find_equivalence_sets(Runtime *runtime, unsigned idx,
          const FieldMask &mask, FieldMaskSet<EquivalenceSet> &target) const;
    protected:
      virtual const VersionInfo& get_version_info(unsigned idx) const;
    public:
      virtual void pack_remote_memoizable(Serializer &rez, 
                                          AddressSpaceID target) const;
      virtual Memoizable* clone(Operation *op);
      static Memoizable* unpack_remote_memoizable(Deserializer &derez,
                                      Operation *op, Runtime *runtime);
      static void handle_eq_request(Deserializer &derez, Runtime *runtime,
                                    AddressSpaceID source);
      static void handle_eq_response(Deserializer &derez, Runtime *runtime);
    public:
      Operation *const op;
      Memoizable *const original; // not a valid pointer on remote nodes
      const AddressSpaceID origin;
      const Operation::OpKind kind;
      const TraceLocalID trace_local_id;
      const ApEvent completion_event;
      const bool is_mem_task;
      const bool is_memo;
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
    class MemoizableOp : public OP, public Memoizable {
    public:
      enum MemoizableState {
        NO_MEMO,   // The operation is not subject to memoization
        MEMO_REQ,  // The mapper requested memoization on this operation
        MEMO_RECORD,    // The runtime is recording analysis for this operation
        MEMO_REPLAY,    // The runtime is replaying analysis for this opeartion
      };
    public:
      MemoizableOp(Runtime *rt);
      void initialize_memoizable(void);
      virtual Operation* get_operation(void) const 
        { return const_cast<MemoizableOp<OP>*>(this); }
      virtual Memoizable* get_memoizable(void) { return this; }
    protected:
      void activate_memoizable(void);
    public:
      virtual void execute_dependence_analysis(void);
      virtual void trigger_replay(void) = 0;
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const;
      virtual PhysicalTemplate* get_template(void) const;
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const
        { assert(false); return ApEvent::NO_AP_EVENT; }
      virtual void set_effects_postcondition(ApEvent postcondition)
        { assert(false); }
      virtual void complete_replay(ApEvent complete_event)
        { assert(false); }
      virtual ApEvent get_memo_completion(void) const
        { return this->get_completion_event(); }
      virtual void replay_mapping_output(void) { /*do nothing*/ }
      virtual Operation::OpKind get_memoizable_kind(void) const
        { return this->get_operation_kind(); }
      virtual ApEvent compute_init_precondition(const TraceInfo &info);
      virtual void find_equivalence_sets(Runtime *runtime, unsigned idx, 
          const FieldMask &mask, FieldMaskSet<EquivalenceSet> &eqs) const;
    protected:
      void invoke_memoize_operation(MapperID mapper_id);
    public:
      virtual bool is_memoizing(void) const { return memo_state != NO_MEMO; }
      virtual bool is_recording(void) const { return memo_state == MEMO_RECORD;}
      inline bool is_replaying(void) const { return memo_state == MEMO_REPLAY; }
      virtual bool is_memoizable_task(void) const { return false; }
      virtual AddressSpaceID get_origin_space(void) const 
        { return this->runtime->address_space; }
      inline MemoizableState get_memoizable_state(void) const 
        { return memo_state; }
    protected:
      // The physical trace for this operation if any
      PhysicalTemplate *tpl;
      // Track whether we are memoizing physical analysis for this operation
      MemoizableState memo_state;
      bool need_prepipeline_stage;
    };

    /**
     * \class ExternalMapping
     * An extension of the external-facing InlineMapping to help 
     * with packing and unpacking them
     */
    class ExternalMapping : public InlineMapping, public ExternalMappable {
    public:
      ExternalMapping(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_mapping(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_mapping(Deserializer &derez, Runtime *runtime);
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
    class MapOp : public ExternalMapping, public Operation,
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
      PhysicalRegion initialize(InnerContext *ctx,
                                const InlineLauncher &launcher);
      void initialize(InnerContext *ctx, const PhysicalRegion &region,
                      const char *provenance);
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
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void update_atomic_locks(const unsigned index,
                                       Reservation lock, bool exclusive);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual ApEvent get_program_order_event(void) const;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
      bool invoke_mapper(InstanceSet &mapped_instances);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual DomainPoint get_shard_point(void) const;
    protected:
      bool remap_region;
      ApUserEvent ready_event;
      ApEvent termination_event;
      PhysicalRegion region;
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      VersionInfo version_info;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::map<Reservation,bool> atomic_locks;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      struct MapProfilingInfo : public Mapping::Mapper::InlineProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>           profiling_requests;
      std::vector<MapProfilingInfo>                     profiling_info;
      RtUserEvent                                   profiling_reported;
      int                                           profiling_priority;
      std::atomic<int>                  outstanding_profiling_requests;
      std::atomic<int>                  outstanding_profiling_reported;
    };

    /**
     * \class ExternalCopy
     * An extension of the external-facing Copy to help 
     * with packing and unpacking them
     */
    class ExternalCopy : public Copy, public ExternalMappable {
    public:
      ExternalCopy(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_copy(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_copy(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class CopyOp
     * The copy operation provides a mechanism for applications
     * to directly copy data between pairs of fields possibly
     * from different region trees in an efficient way by
     * using the low-level runtime copy facilities. 
     */
    class CopyOp : public ExternalCopy, public MemoizableOp<SpeculativeOp>,
                   public LegionHeapify<CopyOp> {
    public:
      static const AllocationType alloc_type = COPY_OP_ALLOC;
    public:
      enum ReqType {
        SRC_REQ = 0,
        DST_REQ = 1,
        GATHER_REQ = 2,
        SCATTER_REQ = 3,
      };
    public:
      struct DeferredCopyAcross : public LgTaskArgs<DeferredCopyAcross>,
                                  public PhysicalTraceInfo {
      public:
        static const LgTaskID TASK_ID = LG_DEFERRED_COPY_ACROSS_TASK_ID;
      public:
        DeferredCopyAcross(CopyOp *op, const PhysicalTraceInfo &info,
                           unsigned idx, ApEvent init, ApUserEvent local_pre,
                           ApUserEvent local_post, ApEvent collective_pre, 
                           ApEvent collective_post, PredEvent g, RtUserEvent a,
                           InstanceSet *src, InstanceSet *dst,
                           InstanceSet *gather, InstanceSet *scatter,
                           const bool preimages)
          : LgTaskArgs<DeferredCopyAcross>(op->get_unique_op_id()), 
            PhysicalTraceInfo(info), copy(op), index(idx),
            init_precondition(init), local_precondition(local_pre),
            local_postcondition(local_post), 
            collective_precondition(collective_pre), 
            collective_postcondition(collective_post), guard(g), applied(a),
            src_targets(src), dst_targets(dst), gather_targets(gather),
            scatter_targets(scatter), compute_preimages(preimages)
          // This is kind of scary, Realm is about to make a copy of this
          // without our knowledge, but we need to preserve the correctness
          // of reference counting on PhysicalTraceRecorders, so just add
          // an extra reference here that we will remove when we're handled.
          { if (rec != NULL) rec->add_recorder_reference(); }
      public:
        inline void remove_recorder_reference(void) const
          { if ((rec != NULL) && rec->remove_recorder_reference()) delete rec; }
      public:
        CopyOp *const copy;
        const unsigned index;
        const ApEvent init_precondition;
        const ApUserEvent local_precondition;
        const ApUserEvent local_postcondition;
        const ApEvent collective_precondition;
        const ApEvent collective_postcondition;
        const PredEvent guard;
        const RtUserEvent applied;
        InstanceSet *const src_targets;
        InstanceSet *const dst_targets;
        InstanceSet *const gather_targets;
        InstanceSet *const scatter_targets;
        const bool compute_preimages;
      };
    public:
      CopyOp(Runtime *rt);
      CopyOp(const CopyOp &rhs);
      virtual ~CopyOp(void);
    public:
      CopyOp& operator=(const CopyOp &rhs);
    public:
      void initialize(InnerContext *ctx,
                      const CopyLauncher &launcher);
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
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre, 
          const ApEvent local_post, ApEvent &collective_pre,
          ApEvent &collective_post, const TraceInfo &trace_info,
          const InstanceSet &instances, const RegionRequirement &req,
          const DomainPoint &key,
          std::vector<IndirectRecord> &records, const bool sources);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void update_atomic_locks(const unsigned index,
                                       Reservation lock, bool exclusive);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    protected:
      void check_copy_privileges(const bool permit_projection) const;
      void check_copy_privilege(const RegionRequirement &req, unsigned idx,
                                const bool permit_projection) const;
      void perform_type_checking(void) const;
      void compute_parent_indexes(void);
      void perform_copy_across(const unsigned index, 
                               const ApEvent init_precondition,
                               const ApUserEvent local_precondition,
                               const ApUserEvent local_postcondition,
                               const ApEvent collective_precondition,
                               const ApEvent collective_postcondition,
                               const PredEvent predication_guard,
                               const InstanceSet &src_targets,
                               const InstanceSet &dst_targets,
                               const InstanceSet *gather_targets,
                               const InstanceSet *scatter_targets,
                               const PhysicalTraceInfo &trace_info,
                               std::set<RtEvent> &applied_conditions,
                               const bool compute_preimages);
      void finalize_copy_profiling(void);
    public:
      static void handle_deferred_across(const void *args);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const;
      virtual void complete_replay(ApEvent copy_complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
    protected:
      template<ReqType REQ_TYPE>
      static const char* get_req_type_name(void);
      template<ReqType REQ_TYPE>
      int perform_conversion(unsigned idx, const RegionRequirement &req,
                             std::vector<MappingInstance> &output,
                             InstanceSet &targets, bool is_reduce = false);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    public:
      std::vector<RegionTreePath>           src_privilege_paths;
      std::vector<RegionTreePath>           dst_privilege_paths;
      std::vector<unsigned>                 src_parent_indexes;
      std::vector<unsigned>                 dst_parent_indexes;
      LegionVector<VersionInfo>             src_versions;
      LegionVector<VersionInfo>             dst_versions;
      std::vector<IndexSpaceExpression*>    copy_expressions;
    public: // These are only used for indirect copies
      std::vector<RegionTreePath>           gather_privilege_paths;
      std::vector<RegionTreePath>           scatter_privilege_paths;
      std::vector<unsigned>                 gather_parent_indexes;
      std::vector<unsigned>                 scatter_parent_indexes;
      std::vector<bool>                     gather_is_range;
      std::vector<bool>                     scatter_is_range;
      LegionVector<VersionInfo>             gather_versions;
      LegionVector<VersionInfo>             scatter_versions;
      std::vector<std::vector<IndirectRecord> > src_indirect_records;
      std::vector<std::vector<IndirectRecord> > dst_indirect_records;
    protected: // for support with mapping
      MapperManager*              mapper;
    protected:
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::vector<std::map<Reservation,bool> > atomic_locks;
      std::set<RtEvent> map_applied_conditions;
    public:
      PredEvent                   predication_guard;
    protected:
      struct CopyProfilingInfo : public Mapping::Mapper::CopyProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>         profiling_requests;
      std::vector<CopyProfilingInfo>                  profiling_info;
      RtUserEvent                                 profiling_reported;
      int                                         profiling_priority;
      std::atomic<int>                outstanding_profiling_requests;
      std::atomic<int>                outstanding_profiling_reported;
    public:
      bool                            possible_src_indirect_out_of_range;
      bool                            possible_dst_indirect_out_of_range;
      bool                            possible_dst_indirect_aliasing; 
    };

    /**
     * \class IndexCopyOp
     * An index copy operation is the same as a copy operation
     * except it is an index space operation for performing
     * multiple copies with projection functions
     */
    class IndexCopyOp : public CollectiveInstanceCreator<CopyOp> {
    public:
      IndexCopyOp(Runtime *rt);
      IndexCopyOp(const IndexCopyOp &rhs);
      virtual ~IndexCopyOp(void);
    public:
      IndexCopyOp& operator=(const IndexCopyOp &rhs);
    public:
      void initialize(InnerContext *ctx,
                      const IndexCopyLauncher &launcher,
                      IndexSpace launch_space);
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
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent &collective_pre,
          ApEvent &collective_post, const TraceInfo &trace_info,
          const InstanceSet &instances, const RegionRequirement &req,
          const DomainPoint &key,
          std::vector<IndirectRecord> &records, const bool sources); 
    public:
      virtual RtEvent find_intra_space_dependence(const DomainPoint &point);
      virtual void record_intra_space_dependence(const DomainPoint &point,
                                                 RtEvent point_mapped);
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From CollectiveInstanceCreator
      virtual IndexSpaceNode* get_collective_space(void) const 
        { return launch_space; }
    public:
      void enumerate_points(bool replaying);
      void handle_point_commit(RtEvent point_committed);
      void check_point_requirements(void);
    protected:
      void log_index_copy_requirements(void);
    public:
      IndexSpaceNode*                                    launch_space;
    protected:
      std::vector<PointCopyOp*>                          points;
      struct IndirectionExchange {
        std::set<ApEvent> local_preconditions;
        std::set<ApEvent> local_postconditions;
        std::vector<std::vector<IndirectRecord>*> src_records;
        std::vector<std::vector<IndirectRecord>*> dst_records;
        ApUserEvent collective_pre;
        ApUserEvent collective_post;
        RtUserEvent src_ready;
        RtUserEvent dst_ready;
      };
      std::vector<IndirectionExchange>                   collective_exchanges;
      unsigned                                           points_committed;
      bool                                       collective_src_indirect_points;
      bool                                       collective_dst_indirect_points;
      bool                                               commit_request;
      std::set<RtEvent>                                  commit_preconditions;
    protected:
      // For checking aliasing of points in debug mode only
      std::set<std::pair<unsigned,unsigned> > interfering_requirements; 
      std::map<DomainPoint,RtEvent> intra_space_dependences;
      std::map<DomainPoint,RtUserEvent> pending_intra_space_dependences;
    };

    /**
     * \class PointCopyOp
     * A point copy operation is used for executing the
     * physical part of the analysis for an index copy
     * operation.
     */
    class PointCopyOp : public CopyOp, public ProjectionPoint {
    public:
      friend class IndexCopyOp;
      PointCopyOp(Runtime *rt);
      PointCopyOp(const PointCopyOp &rhs);
      virtual ~PointCopyOp(void);
    public:
      PointCopyOp& operator=(const PointCopyOp &rhs);
    public:
      void initialize(IndexCopyOp *owner, const DomainPoint &point);
      void launch(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      // trigger_mapping same as base class
      virtual void trigger_commit(void);
      virtual RtEvent exchange_indirect_records(
          const unsigned index, const ApEvent local_pre,
          const ApEvent local_post, ApEvent &collective_pre,
          ApEvent &collective_post, const TraceInfo &trace_info,
          const InstanceSet &instances, const RegionRequirement &req,
          const DomainPoint &key,
          std::vector<IndirectRecord> &records, const bool sources);
    public:
      // For collective instances
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
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
      virtual void record_intra_space_dependences(unsigned idx,
                               const std::vector<DomainPoint> &region_deps);
      virtual const Mappable* as_mappable(void) const { return this; }
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const;
    protected:
      IndexCopyOp*                          owner;
      std::set<RtEvent>                     intra_space_mapping_dependences;
    };

    /**
     * \class FenceOp
     * Fence operations give the application the ability to
     * enforce ordering guarantees between different tasks
     * in the same context which may become important when
     * certain updates to the region tree are desired to be
     * observed before a later operation either maps or 
     * runs. All fences are mapping fences for correctness.
     * Fences all support the optional ability to be an 
     * execution fence.
     */
    class FenceOp : public MemoizableOp<Operation>, 
                    public LegionHeapify<FenceOp> {
    public:
      enum FenceKind {
        MAPPING_FENCE,
        EXECUTION_FENCE,
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
      Future initialize(InnerContext *ctx, FenceKind kind, bool need_future,
                        const char *provenance);
      Future initialize(InnerContext *ctx, FenceKind kind, bool need_future,
                        Provenance *provenance);
      inline void add_mapping_applied_condition(RtEvent precondition)
        { map_applied_conditions.insert(precondition); }
      inline void record_execution_precondition(ApEvent precondition)
        { execution_preconditions.insert(precondition); }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { exec_fence = (fence_kind == EXECUTION_FENCE); return exec_fence; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
#ifdef LEGION_SPY
      virtual void trigger_complete(void);
#endif
      virtual void trigger_replay(void);
      virtual void complete_replay(ApEvent complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
    protected:
      void activate_fence(void);
      void deactivate_fence(void);
      void perform_fence_analysis(bool update_fence = false);
      void update_current_fence(void);
    protected:
      FenceKind fence_kind;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> execution_preconditions;
      Future result;
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
      void initialize(InnerContext *ctx, const char *provenance);
      void set_previous(ApEvent previous);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
    protected:
      ApEvent previous_completion;
    };

    /**
     * \class CreationOp
     * A creation operation is used for deferring the creation of
     * an particular resource until some event has transpired such
     * as the resolution of a future.
     */
    class CreationOp : public Operation, public LegionHeapify<CreationOp> {
    public:
      static const AllocationType alloc_type = CREATION_OP_ALLOC;
    public:
      enum CreationKind {
        INDEX_SPACE_CREATION,
        FIELD_ALLOCATION,
        FUTURE_MAP_CREATION,
      };
    public:
      CreationOp(Runtime *rt);
      CreationOp(const CreationOp &rhs);
      virtual ~CreationOp(void);
    public:
      CreationOp& operator=(const CreationOp &rhs);
    public:
      void initialize_index_space(InnerContext *ctx, IndexSpaceNode *node,
                                  const Future &future, const char *provenance);
      void initialize_field(InnerContext *ctx, FieldSpaceNode *node,
                            FieldID fid, const Future &field_size,
                            const char *provenance);
      void initialize_fields(InnerContext *ctx, FieldSpaceNode *node,
                             const std::vector<FieldID> &fids,
                             const std::vector<Future> &field_sizes,
                             const char *provenance);
      void initialize_map(InnerContext *ctx, const char *provenance,
                          const std::map<DomainPoint,Future> &futures);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
    protected:
      CreationKind kind; 
      IndexSpaceNode *index_space_node;
      FieldSpaceNode *field_space_node;
      std::vector<Future> futures;
      std::vector<FieldID> fields;
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
      };
    public:
      DeletionOp(Runtime *rt);
      DeletionOp(const DeletionOp &rhs);
      virtual ~DeletionOp(void);
    public:
      DeletionOp& operator=(const DeletionOp &rhs);
    public:
      void set_deletion_preconditions(ApEvent precondition,
          const std::map<Operation*,GenerationID> &dependences);
    public:
      void initialize_index_space_deletion(InnerContext *ctx, IndexSpace handle,
                                   std::vector<IndexPartition> &sub_partitions,
                                   const bool unordered,Provenance *provenance);
      void initialize_index_part_deletion(InnerContext *ctx,IndexPartition part,
                                   std::vector<IndexPartition> &sub_partitions,
                                   const bool unordered,Provenance *provenance);
      void initialize_field_space_deletion(InnerContext *ctx,
                                           FieldSpace handle,
                                           const bool unordered,
                                           Provenance *provenance);
      void initialize_field_deletion(InnerContext *ctx, FieldSpace handle,
                                     FieldID fid, const bool unordered,
                                     FieldAllocatorImpl *allocator,
                                     Provenance *provenance);
      void initialize_field_deletions(InnerContext *ctx, FieldSpace handle,
                                      const std::set<FieldID> &to_free,
                                      const bool unordered,
                                      FieldAllocatorImpl *allocator,
                                      Provenance *provenance);
      void initialize_logical_region_deletion(InnerContext *ctx, 
                                              LogicalRegion handle,
                                              const bool unordered,
                                              Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void); 
      virtual void trigger_complete(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      DeletionKind kind;
      ApEvent execution_precondition;
      IndexSpace index_space;
      IndexPartition index_part;
      std::vector<IndexPartition> sub_partitions;
      FieldSpace field_space;
      FieldAllocatorImpl *allocator;
      LogicalRegion logical_region;
      std::set<FieldID> free_fields;
      std::vector<FieldID> local_fields;
      std::vector<FieldID> global_fields;
      std::vector<unsigned> local_field_indexes;
      std::vector<unsigned> parent_req_indexes;
      std::vector<unsigned> deletion_req_indexes;
      std::vector<bool> returnable_privileges;
      std::vector<RegionRequirement> deletion_requirements;
      LegionVector<VersionInfo> version_infos;
      std::set<RtEvent> map_applied_conditions;
      std::map<Operation*,GenerationID> dependences;
      bool has_preconditions;
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
     * \class ExternalClose
     * An extension of the external-facing Close to help 
     * with packing and unpacking them
     */
    class ExternalClose : public Close, public ExternalMappable {
    public:
      ExternalClose(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_close(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_close(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class CloseOp
     * Close operations are only visible internally inside
     * the runtime and are issued to help close up the 
     * physical region tree. There are two types of close
     * operations that both inherit from this class:
     * InterCloseOp and PostCloseOp.
     */
    class CloseOp : public ExternalClose, public InternalOp {
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
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual Mappable* get_mappable(void);
    public:
      void activate_close(void);
      void deactivate_close(void);
      // This is for post and virtual close ops
      void initialize_close(InnerContext *ctx,
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
    };

    /**
     * \class MergeCloseOp
     * merge close operations are issued by the runtime
     * for closing up region trees as part of the normal execution
     * of an application.
     */
    class MergeCloseOp : public CloseOp, public LegionHeapify<MergeCloseOp> {
    public:
      MergeCloseOp(Runtime *runtime);
      MergeCloseOp(const MergeCloseOp &rhs);
      virtual ~MergeCloseOp(void);
    public:
      MergeCloseOp& operator=(const MergeCloseOp &rhs);
    public:
      void initialize(InnerContext *ctx, const RegionRequirement &req,
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
#ifdef LEGION_SPY
      virtual void trigger_complete(void);
#endif
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
    class PostCloseOp : public CloseOp, public LegionHeapify<PostCloseOp> {
    public:
      PostCloseOp(Runtime *runtime);
      PostCloseOp(const PostCloseOp &rhs);
      virtual ~PostCloseOp(void);
    public:
      PostCloseOp& operator=(const PostCloseOp &rhs);
    public:
      void initialize(InnerContext *ctx, unsigned index, 
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
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    protected:
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      unsigned parent_idx;
      InstanceSet target_instances;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      struct CloseProfilingInfo : public Mapping::Mapper::CloseProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>          profiling_requests;
      std::vector<CloseProfilingInfo>                  profiling_info;
      RtUserEvent                                  profiling_reported;
      int                                          profiling_priority;
      std::atomic<int>                 outstanding_profiling_requests;
      std::atomic<int>                 outstanding_profiling_reported;
    };

    /**
     * \class VirtualCloseOp
     * Virtual close operations are issued by the runtime for
     * closing up virtual mappings to a composite instance
     * that can then be propagated back to the enclosing
     * parent task.
     */
    class VirtualCloseOp : public CloseOp, 
                           public LegionHeapify<VirtualCloseOp> {
    public:
      VirtualCloseOp(Runtime *runtime);
      VirtualCloseOp(const VirtualCloseOp &rhs);
      virtual ~VirtualCloseOp(void);
    public:
      VirtualCloseOp& operator=(const VirtualCloseOp &rhs);
    public:
      void initialize(InnerContext *ctx, unsigned index,
                      const RegionRequirement &req);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual unsigned find_parent_index(unsigned idx);
#ifdef LEGION_SPY
      virtual void trigger_complete(void);
#endif
    protected:
      std::set<RtEvent> map_applied_conditions;
      unsigned parent_idx;
    };

    /**
     * \class ExternalAcquire
     * An extension of the external-facing Acquire to help 
     * with packing and unpacking them
     */
    class ExternalAcquire : public Acquire, public ExternalMappable {
    public:
      ExternalAcquire(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_acquire(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_acquire(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class AcquireOp
     * Acquire operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class AcquireOp : public ExternalAcquire,public MemoizableOp<SpeculativeOp>,
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
      void initialize(InnerContext *ctx, const AcquireLauncher &launcher);
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
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public: 
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const;
      virtual void complete_replay(ApEvent acquire_complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
    protected:
      void check_acquire_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      void log_acquire_requirement(void);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      struct AcquireProfilingInfo : 
        public Mapping::Mapper::AcquireProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>            profiling_requests;
      std::vector<AcquireProfilingInfo>                  profiling_info;
      RtUserEvent                                    profiling_reported;
      int                                            profiling_priority;
      std::atomic<int>                   outstanding_profiling_requests;
      std::atomic<int>                   outstanding_profiling_reported;
    };

    /**
     * \class ExternalRelease
     * An extension of the external-facing Release to help 
     * with packing and unpacking them
     */
    class ExternalRelease: public Release, public ExternalMappable {
    public:
      ExternalRelease(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_release(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_release(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class ReleaseOp
     * Release operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class ReleaseOp : public ExternalRelease,public MemoizableOp<SpeculativeOp>,
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
      void initialize(InnerContext *ctx, const ReleaseLauncher &launcher);
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
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From Memoizable
      virtual ApEvent compute_sync_precondition(const TraceInfo *info) const;
      virtual void complete_replay(ApEvent release_complete_event);
      virtual const VersionInfo& get_version_info(unsigned idx) const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const;
    protected:
      void check_release_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      void log_release_requirement(void);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                      const Realm::ProfilingResponse &response,
                                      const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      struct ReleaseProfilingInfo : 
        public Mapping::Mapper::ReleaseProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>            profiling_requests;
      std::vector<ReleaseProfilingInfo>                  profiling_info;
      RtUserEvent                                    profiling_reported;
      int                                            profiling_priority;
      std::atomic<int>                   outstanding_profiling_requests;
      std::atomic<int>                   outstanding_profiling_reported;
    };

    /**
     * \class DynamicCollectiveOp
     * A class for getting values from a collective operation
     * and writing them into a future. This will also give
     * us the framework necessary to handle roll backs on 
     * collectives so we can memoize their results.
     */
    class DynamicCollectiveOp : public MemoizableOp<Operation>,
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
      Future initialize(InnerContext *ctx, const DynamicCollective &dc,
                        const char *provenance);
    public:
      virtual const VersionInfo& get_version_info(unsigned idx) const
        { assert(false); return *(new VersionInfo()); }
      virtual const RegionRequirement& get_requirement(unsigned idx) const
        { assert(false); return *(new RegionRequirement()); }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
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
      void initialize(InnerContext *ctx, Future f, const char *provenance);
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
      void initialize(InnerContext *task, const Predicate &p,
                      const char *provenance);
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
      void initialize(InnerContext *task, 
                      const std::vector<Predicate> &predicates,
                      const std::string &provenance);
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
      void initialize(InnerContext *task, 
                      const std::vector<Predicate> &predicates,
                      const std::string &provenance);
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
      inline FutureMap get_future_map(void) const { return result_map; }
    public:
      FutureMap initialize(InnerContext *ctx,
                           const MustEpochLauncher &launcher);
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
      virtual std::map<PhysicalManager*,unsigned>*
                                       get_acquired_instances_ref(void);
    public:
      void add_mapping_dependence(RtEvent precondition);
      void register_single_task(SingleTask *single, unsigned index);
      void register_slice_task(SliceTask *slice);
    public:
      // Methods for keeping track of when we can complete and commit
      void register_subop(Operation *op);
      void notify_subop_complete(Operation *op, RtEvent precondition);
      void notify_subop_commit(Operation *op, RtEvent precondition);
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
      std::map<PhysicalManager*,unsigned> acquired_instances;
    protected:
      std::map<std::pair<unsigned/*task index*/,unsigned/*req index*/>,
               unsigned/*dependence index*/> dependence_map;
      std::vector<DependenceRecord*> dependences;
      std::map<SingleTask*,unsigned/*single task index*/> single_task_map;
      std::vector<std::set<unsigned/*single task index*/> > mapping_dependences;
    protected:
      std::map<UniqueID,RtUserEvent> slice_version_events;
    protected:
      std::set<RtEvent> completion_preconditions, commit_preconditions;
      std::set<ApEvent> completion_effects;
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
        WEIGHT_PARTITION,
        UNION_PARTITION,
        INTERSECTION_PARTITION,
        INTERSECTION_WITH_REGION,
        DIFFERENCE_PARTITION,
        RESTRICTED_PARTITION,
        BY_DOMAIN_PARTITION,
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
      class WeightPartitionThunk : public PendingPartitionThunk {
      public:
        WeightPartitionThunk(IndexPartition id, const FutureMap &w, size_t g)
          : pid(id), weights(w), granularity(g) { }
        virtual ~WeightPartitionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_weights(op, pid, 
                                        weights, granularity); }
        virtual void perform_logging(PendingPartitionOp *op);
      protected:
        IndexPartition pid;
        FutureMap weights;
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
      class IntersectionWithRegionThunk: public PendingPartitionThunk {
      public:
        IntersectionWithRegionThunk(IndexPartition id, IndexPartition p, bool d)
          : pid(id), part(p), dominates(d) { }
        virtual ~IntersectionWithRegionThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_intersection(op, pid, 
                                                          part, dominates); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition part;
        const bool dominates;
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
      class FutureMapThunk : public PendingPartitionThunk {
      public:
        FutureMapThunk(IndexPartition id, const FutureMap &fm, bool inter)
          : pid(id), future_map(fm), perform_intersections(inter) { }
        virtual ~FutureMapThunk(void) { }
      public:
        virtual ApEvent perform(PendingPartitionOp *op,
                                RegionTreeForest *forest)
        { return forest->create_partition_by_domain(op, pid, future_map,
                                              perform_intersections); }
        virtual void perform_logging(PendingPartitionOp *op);
      protected:
        IndexPartition pid;
        FutureMap future_map;
        bool perform_intersections;
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
      void initialize_equal_partition(InnerContext *ctx, IndexPartition pid,
                                      size_t granularity, const char *prov);
      void initialize_weight_partition(InnerContext *ctx, IndexPartition pid,
                                const FutureMap &weights, size_t granularity,
                                const char *provenance);
      void initialize_union_partition(InnerContext *ctx,
                                      IndexPartition pid, 
                                      IndexPartition handle1,
                                      IndexPartition handle2,
                                      const char *provenance);
      void initialize_intersection_partition(InnerContext *ctx,
                                             IndexPartition pid, 
                                             IndexPartition handle1,
                                             IndexPartition handle2,
                                             const char *provenance);
      void initialize_intersection_partition(InnerContext *ctx,
                                             IndexPartition pid, 
                                             IndexPartition part,
                                             const bool dominates,
                                             const char *provenance);
      void initialize_difference_partition(InnerContext *ctx,
                                           IndexPartition pid, 
                                           IndexPartition handle1,
                                           IndexPartition handle2,
                                           const char *provenance);
      void initialize_restricted_partition(InnerContext *ctx,
                                           IndexPartition pid,
                                           const void *transform,
                                           size_t transform_size,
                                           const void *extent,
                                           size_t extent_size,
                                           const char *provenance);
      void initialize_by_domain(InnerContext *ctx, IndexPartition pid,
                                const FutureMap &future_map,
                                bool perform_intersections,
                                const char *provenance);
      void initialize_cross_product(InnerContext *ctx, IndexPartition base, 
                                    IndexPartition source, LegionColor color,
                                    const char *provenance);
      void initialize_index_space_union(InnerContext *ctx, IndexSpace target, 
                                        const std::vector<IndexSpace> &handles,
                                        const char *provenance);
      void initialize_index_space_union(InnerContext *ctx, IndexSpace target, 
                                        IndexPartition handle,
                                        const char *provenance);
      void initialize_index_space_intersection(InnerContext *ctx, 
                                               IndexSpace target,
                                        const std::vector<IndexSpace> &handles,
                                               const char *provenance);
      void initialize_index_space_intersection(InnerContext *ctx,
                                              IndexSpace target,
                                              IndexPartition handle,
                                              const char *provenance);
      void initialize_index_space_difference(InnerContext *ctx, 
                                             IndexSpace target, 
                                             IndexSpace initial,
                                        const std::vector<IndexSpace> &handles,
                                        const char *provenance);
      void perform_logging();
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_complete(void);
      virtual bool is_partition_op(void) const { return true; } 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    protected:
      PendingPartitionThunk *thunk;
      FutureMap future_map;
    };

    /**
     * \class ExternalPartition
     * An extension of the external-facing Partition to help 
     * with packing and unpacking them
     */
    class ExternalPartition: public Partition, public ExternalMappable {
    public:
      ExternalPartition(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_partition(Serializer &rez,AddressSpaceID target) const;
      void unpack_external_partition(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class DependentPartitionOp
     * An operation for creating different kinds of partitions
     * which are dependent on mapping a region in order to compute
     * the resulting partition.
     */
    class DependentPartitionOp : public ExternalPartition, 
                                 public CollectiveInstanceCreator<Operation>,
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
        virtual bool safe_projection(IndexPartition p) const { return false; }
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
        virtual bool safe_projection(IndexPartition p) const 
          { return (p == projection); }
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
        virtual bool safe_projection(IndexPartition p) const
          { return (p == projection); }
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
      void initialize_by_field(InnerContext *ctx, IndexPartition pid,
                               LogicalRegion handle, LogicalRegion parent,
                               FieldID fid, MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
      void initialize_by_image(InnerContext *ctx, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
      void initialize_by_image_range(InnerContext *ctx, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
      void initialize_by_preimage(InnerContext *ctx, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
      void initialize_by_preimage_range(InnerContext *ctx, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
      void initialize_by_association(InnerContext *ctx, LogicalRegion domain,
                               LogicalRegion domain_parent, FieldID fid,
                               IndexSpace range, MapperID id, MappingTagID tag,
                               const UntypedBuffer &marg,
                               const char *provenance);
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
                                    const InstanceSet &mapped_instances,
                                    const PhysicalTraceInfo &info,
                                    const DomainPoint &key);
      virtual unsigned find_parent_index(unsigned idx);
      virtual bool is_partition_op(void) const { return true; }
    public:
      virtual PartitionKind get_partition_kind(void) const;
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual size_t get_region_count(void) const;
      virtual void trigger_commit(void);
    public:
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,unsigned>*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      // Report a profiling result for this operation
      virtual void handle_profiling_response(const ProfilingResponseBase *base,
                                        const Realm::ProfilingResponse &result,
                                        const void *orig, size_t orig_length);
      virtual void handle_profiling_update(int count);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    public:
      // From CollectiveInstanceCreator
      virtual IndexSpaceNode* get_collective_space(void) const 
        { return launch_space; }
    public:
      // For collective instances
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
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
      void select_partition_projection(void);
      bool invoke_mapper(InstanceSet &mapped_instances);
      void activate_dependent_op(void);
      void deactivate_dependent_op(void);
      void finalize_partition_profiling(void);
    public:
      void handle_point_commit(RtEvent point_committed);
    public:
      VersionInfo version_info;
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      std::map<PhysicalManager*,unsigned> acquired_instances;
      std::set<RtEvent> map_applied_conditions;
      DepPartThunk *thunk;
    protected:
      MapperManager *mapper;
    protected:
      // For index versions of this operation
      IndexSpaceNode*                   launch_space;
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
      struct PartitionProfilingInfo :
        public Mapping::Mapper::PartitionProfilingInfo {
      public:
        void *buffer;
        size_t buffer_size;
      };
      std::vector<ProfilingMeasurementID>              profiling_requests;
      std::vector<PartitionProfilingInfo>                  profiling_info;
      RtUserEvent                                      profiling_reported;
      int                                              profiling_priority;
      std::atomic<int>                     outstanding_profiling_requests;
      std::atomic<int>                     outstanding_profiling_reported;
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
      void launch(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual ApEvent trigger_thunk(IndexSpace handle,
                                    const InstanceSet &mapped_instances,
                                    const PhysicalTraceInfo &trace_info,
                                    const DomainPoint &key);
      virtual void trigger_commit(void);
      virtual PartitionKind get_partition_kind(void) const;
    public:
      // For collective instances
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
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
      virtual void record_intra_space_dependences(unsigned idx,
                               const std::vector<DomainPoint> &region_deps);
      virtual const Mappable* as_mappable(void) const { return this; }
    public:
      DependentPartitionOp *owner;
    };

    /**
     * \class ExternalFill
     * An extension of the external-facing Fill to help 
     * with packing and unpacking them
     */
    class ExternalFill : public Fill, public ExternalMappable {
    public:
      ExternalFill(void);
    public:
      virtual void set_context_index(size_t index) = 0;
    public:
      void pack_external_fill(Serializer &rez, AddressSpaceID target) const;
      void unpack_external_fill(Deserializer &derez, Runtime *runtime);
    };

    /**
     * \class FillOp
     * Fill operations are used to initialize a field to a
     * specific value for a particular logical region.
     */
    class FillOp : public MemoizableOp<SpeculativeOp>, public ExternalFill,
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
      void initialize(InnerContext *ctx, const FillLauncher &launcher);
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
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual std::map<PhysicalManager*,unsigned>*
                                       get_acquired_instances_ref(void);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
    public:
      virtual bool has_prepipeline_stage(void) const
        { return need_prepipeline_stage; }
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
    public:
      virtual bool query_speculate(bool &value, bool &mapping_only);
      virtual void resolve_true(bool speculated, bool launched);
      virtual void resolve_false(bool speculated, bool launched);
    public:
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    public:
      void check_fill_privilege(void);
      void compute_parent_index(void);
      ApEvent compute_sync_precondition(const TraceInfo *info) const;
      void log_fill_requirement(void) const;
    public:
      // From Memoizable
      virtual const VersionInfo& get_version_info(unsigned idx) const
        { return version_info; }
      virtual const RegionRequirement& get_requirement(unsigned idx) const
        { return get_requirement(); }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    public:
      RegionTreePath privilege_path;
      VersionInfo version_info;
      unsigned parent_req_index;
      void *value;
      size_t value_size;
      Future future;
      FillView *fill_view;
      std::set<RtEvent> map_applied_conditions;
      PredEvent true_guard, false_guard;
    };
    
    /**
     * \class IndexFillOp
     * This is the same as a fill operation except for
     * applying a number of fill operations over an 
     * index space of points with projection functions.
     */
    class IndexFillOp : public CollectiveInstanceCreator<FillOp> {
    public:
      IndexFillOp(Runtime *rt);
      IndexFillOp(const IndexFillOp &rhs);
      virtual ~IndexFillOp(void);
    public:
      IndexFillOp& operator=(const IndexFillOp &rhs);
    public:
      void initialize(InnerContext *ctx,
                      const IndexFillLauncher &launcher,
                      IndexSpace launch_space);
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
      // From MemoizableOp
      virtual void trigger_replay(void);
    public:
      // From CollectiveInstanceCreator
      virtual IndexSpaceNode* get_collective_space(void) const 
        { return launch_space; }
    public:
      void enumerate_points(bool replaying);
      void handle_point_commit(void);
      void check_point_requirements(void);
    protected:
      void log_index_fill_requirement(void);
    public:
      IndexSpaceNode*               launch_space;
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
      void launch(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual void trigger_prepipeline_stage(void);
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      // trigger_mapping same as base class
      virtual void trigger_commit(void);
    public:
      // For collective instances
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
      // From ProjectionPoint
      virtual const DomainPoint& get_domain_point(void) const;
      virtual void set_projection_result(unsigned idx, LogicalRegion result);
      virtual void record_intra_space_dependences(unsigned idx,
                               const std::vector<DomainPoint> &region_deps);
      virtual const Mappable* as_mappable(void) const { return this; }
    public:
      // From Memoizable
      virtual TraceLocalID get_trace_local_id(void) const;
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
      PhysicalRegion initialize(InnerContext *ctx,
                                const AttachLauncher &launcher);
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
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual RtEvent check_for_coregions(void);
    public:
      LogicalRegion create_external_instance(void);
      PhysicalInstance create_instance(IndexSpaceNode *node,
                                       const std::vector<FieldID> &field_set,
                                       const std::vector<size_t> &field_sizes,
                                             LayoutConstraintSet &cons,
                                             ApEvent &ready_event,
                                             size_t &instance_footprint);
    protected:
      void activate_attach(void);
      void deactivate_attach(void);
      void check_privilege(void);
      void compute_parent_index(void);
      void log_requirement(void);
    public:
      ExternalResource resource;
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      const char *file_name;
      std::map<FieldID,const char*> field_map;
      std::map<FieldID,void*> field_pointers_map;
      LegionFileMode file_mode;
      PhysicalRegion region;
      unsigned parent_req_index;
      InstanceSet external_instances;
      ApUserEvent attached_event;
      std::set<RtEvent> map_applied_conditions;
      LayoutConstraintSet layout_constraint_set;
      size_t footprint;
      ApEvent termination_event;
      bool restricted;
      bool mapping;
    };

    /**
     * \class IndexAttachOp
     * This provides support for doing index space attach
     * operations where we are attaching external resources
     * to many subregions of a region tree with a single operation
     */
    class IndexAttachOp : public Operation,public LegionHeapify<IndexAttachOp> {
    public:
      static const AllocationType alloc_type = ATTACH_OP_ALLOC;
    public:
      IndexAttachOp(Runtime *rt);
      IndexAttachOp(const IndexAttachOp &rhs);
      virtual ~IndexAttachOp(void);
    public:
      IndexAttachOp& operator=(const IndexAttachOp &rhs);
    public:
      ExternalResources initialize(InnerContext *ctx,
                                   RegionTreeNode *upper_bound,
                                   IndexSpaceNode *launch_bounds,
                                   const IndexAttachLauncher &launcher,
                                   const std::vector<unsigned> &indexes);
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
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
    public:
      RtEvent find_coregions(PointAttachOp *point, LogicalRegion region,
          InstanceSet &instances, ApUserEvent &attached_event);
      void handle_point_commit(void);
    protected:
      void activate_index_attach(void);
      void deactivate_index_attach(void);
      void compute_parent_index(void);
      void check_privilege(void);
      void check_point_requirements(void);
      void log_requirement(void);
    protected:
      RegionRequirement                             requirement;
      ExternalResources                             resources;
      RegionTreePath                                privilege_path;
      IndexSpaceNode*                               launch_space;
      std::vector<PointAttachOp*>                   points;
      std::map<LogicalRegion,std::vector<PointAttachOp*> >  coregions;
      std::map<LogicalRegion,ApUserEvent>           coregions_attached;
      std::set<RtEvent>                             map_applied_conditions;
      unsigned                                      parent_req_index;
      unsigned                                      points_committed;
      bool                                          commit_request;
    };
    
    /**
     * \class PointAttachOp
     * An individual attach operation inside of an index attach operation
     */
    class PointAttachOp : public AttachOp {
    public:
      PointAttachOp(Runtime *rt);
      PointAttachOp(const PointAttachOp &rhs);
      virtual ~PointAttachOp(void);
    public:
      PointAttachOp& operator=(const PointAttachOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      PhysicalRegionImpl* initialize(IndexAttachOp *owner, InnerContext *ctx,
        const IndexAttachLauncher &launcher, const OrderingConstraint &ordering,
        const DomainPoint &point, unsigned index);
    public:
      // Overload to look for coregions between points
      virtual RtEvent check_for_coregions(void);
      virtual void trigger_ready(void);
      virtual void trigger_commit(void);
    protected:
      IndexAttachOp *owner;
      DomainPoint index_point;
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
      Future initialize_detach(InnerContext *ctx, PhysicalRegion region,
                               const bool flush, const bool unordered,
                               const char *provenance);
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
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      void activate_detach(void);
      void deactivate_detach(void);
      void compute_parent_index(void);
      void log_requirement(void);
    public:
      PhysicalRegion region;
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
      Future result;
      bool flush;
    };

    /**
     * \class IndexDetachOp
     * This is an index space detach operation for performing many detaches
     */
    class IndexDetachOp : public Operation,public LegionHeapify<IndexDetachOp> {
    public:
      static const AllocationType alloc_type = DETACH_OP_ALLOC;
    public:
      IndexDetachOp(Runtime *rt);
      IndexDetachOp(const IndexDetachOp &rhs);
      virtual ~IndexDetachOp(void);
    public:
      IndexDetachOp& operator=(const IndexDetachOp &rhs);
    public:
      Future initialize_detach(InnerContext *ctx, LogicalRegion parent,
                               RegionTreeNode *upper_bound,
                               IndexSpaceNode *launch_bounds,
                               ExternalResourcesImpl *external,
                               const std::vector<FieldID> &privilege_fields,
                               const std::vector<PhysicalRegion> &regions,
                               bool flush, bool unordered,
                               const char *provenance);
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
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
    public:
      void complete_detach(void);
      void handle_point_complete(void);
      void handle_point_commit(void);
    protected:
      void activate_index_detach(void);
      void deactivate_index_detach(void);
      void compute_parent_index(void);
      void log_requirement(void);
    protected:
      RegionRequirement                             requirement;
      ExternalResources                             resources;
      RegionTreePath                                privilege_path;
      IndexSpaceNode*                               launch_space;
      std::vector<PointDetachOp*>                   points;
      std::set<RtEvent>                             map_applied_conditions;
      Future                                        result;
      unsigned                                      parent_req_index;
      unsigned                                      points_completed;
      unsigned                                      points_committed;
      bool                                          complete_request;
      bool                                          commit_request;
    };

    /**
     * \class PointDetachOp
     * Indvidiual detach operations for an index space detach
     */
    class PointDetachOp : public DetachOp {
    public:
      PointDetachOp(Runtime *rt);
      PointDetachOp(const PointDetachOp &rhs);
      virtual ~PointDetachOp(void);
    public:
      PointDetachOp& operator=(const PointDetachOp &rhs);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
    public:
      void initialize_detach(IndexDetachOp *owner, InnerContext *ctx,
            const PhysicalRegion &region, const DomainPoint &point, bool flush);
    public:
      virtual void trigger_ready(void);
      virtual void trigger_complete(void);
      virtual void trigger_commit(void);
    protected:
      IndexDetachOp *owner;
      DomainPoint index_point;
    };

    /**
     * \class TimingOp
     * Operation for performing timing measurements
     */
    class TimingOp : public Operation, public LegionHeapify<TimingOp> {
    public:
      TimingOp(Runtime *rt);
      TimingOp(const TimingOp &rhs);
      virtual ~TimingOp(void);
    public:
      TimingOp& operator=(const TimingOp &rhs);
    public:
      Future initialize(InnerContext *ctx, const TimingLauncher &launcher);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { return false; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
    protected:
      TimingMeasurement measurement;
      std::set<Future> preconditions;
      Future result;
    };

    /**
     * \class TunableOp
     * Operation for performing tunable requests
     */
    class TunableOp : public Operation, public LegionHeapify<TunableOp> {
    public:
      TunableOp(Runtime *rt);
      TunableOp(const TunableOp &rhs);
      virtual ~TunableOp(void);
    public:
      TunableOp& operator=(const TunableOp &rhs);
    public:
      void activate_tunable(void);
      void deactivate_tunable(void);
      Future initialize(InnerContext *ctx, const TunableLauncher &launcher);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { return false; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
    protected:
      TunableID tunable_id;
      MapperID mapper_id;
      MappingTagID tag;
      void *arg;
      size_t argsize;
      size_t tunable_index;
      Future result;
      std::vector<Future> futures;
    };

    /**
     * \class AllReduceOp 
     * Operation for reducing future maps down to futures
     */
    class AllReduceOp : public Operation, public LegionHeapify<AllReduceOp> {
    public:
      AllReduceOp(Runtime *rt);
      AllReduceOp(const AllReduceOp &rhs);
      virtual ~AllReduceOp(void);
    public:
      AllReduceOp& operator=(const AllReduceOp &rhs);
    public:
      Future initialize(InnerContext *ctx, const FutureMap &future_map,
                        ReductionOpID redop, bool deterministic,
                        const char *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual bool invalidates_physical_trace_template(bool &exec_fence) const
        { return false; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_execution(void);
    protected:
      FutureMap future_map;
      const ReductionOp *redop; 
      Future result;
      bool deterministic;
    };

    /**
     * \class RemoteOp
     * This operation is a shim for operations on remote nodes
     * and is used by remote physical analysis traversals to handle
     * any requests they might have of the original operation.
     */
    class RemoteOp : public Operation {
    public:
      struct DeferRemoteOpDeletionArgs : 
        public LgTaskArgs<DeferRemoteOpDeletionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_REMOTE_OP_DELETION_TASK_ID;
      public:
        DeferRemoteOpDeletionArgs(Operation *o)
          : LgTaskArgs<DeferRemoteOpDeletionArgs>(o->get_unique_op_id()), 
            op(o) { }
      public:
        Operation *const op;
      };
    public:
      RemoteOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteOp(const RemoteOp &rhs);
      virtual ~RemoteOp(void);
    public:
      RemoteOp& operator=(const RemoteOp &rhs);
    public:
      virtual void unpack(Deserializer &derez,
                          ReferenceMutator &mutator) = 0;
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const = 0;
      virtual OpKind get_operation_kind(void) const = 0;
      virtual std::map<PhysicalManager*,unsigned>*
                                       get_acquired_instances_ref(void);
      // This should be the only mapper call that we need to handle
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking) = 0;
      virtual void add_copy_profiling_request(const PhysicalTraceInfo &info,
                               Realm::ProfilingRequestSet &requests,
                               bool fill, unsigned count = 1);
      virtual void report_uninitialized_usage(const unsigned index,
                                              LogicalRegion handle,
                                              const RegionUsage usage,
                                              const char *field_string,
                                              RtUserEvent reported);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const = 0;
    public:
      void defer_deletion(RtEvent precondition);
      void pack_remote_base(Serializer &rez) const;
      void unpack_remote_base(Deserializer &derez, Runtime *runtime,
                              std::set<RtEvent> &ready_events);
      void pack_profiling_requests(Serializer &rez, 
                                   std::set<RtEvent> &applied) const;
      void unpack_profiling_requests(Deserializer &derez);
      static void handle_deferred_deletion(const void *args);
      // Caller takes ownership of this object and must delete it when done
      static RemoteOp* unpack_remote_operation(Deserializer &derez,
                         Runtime *runtime, std::set<RtEvent> &ready_events);
      static void handle_report_uninitialized(Deserializer &derez);
      static void handle_report_profiling_count_update(Deserializer &derez);
    public:
      // This is a pointer to an operation on a remote node
      // it should never be dereferenced
      Operation *const remote_ptr;
      const AddressSpaceID source;
    protected:
      MapperManager *mapper;
    protected:
      std::vector<ProfilingMeasurementID> profiling_requests;
      int                                 profiling_priority;
      Processor                           profiling_target;
      RtUserEvent                         profiling_response;
      std::atomic<int>                    profiling_reports;
    };

    /**
     * \class RemoteMapOp
     * This is a remote copy of a MapOp to be used
     * for mapper calls and other operations
     */
    class RemoteMapOp : public ExternalMapping, public RemoteOp,
                        public LegionHeapify<RemoteMapOp> {
    public:
      RemoteMapOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteMapOp(const RemoteMapOp &rhs);
      virtual ~RemoteMapOp(void);
    public:
      RemoteMapOp& operator=(const RemoteMapOp &rhs); 
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking); 
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteCopyOp
     * This is a remote copy of a CopyOp to be used
     * for mapper calls and other operations
     */
    class RemoteCopyOp : public ExternalCopy, public RemoteOp,
                         public LegionHeapify<RemoteCopyOp> {
    public:
      RemoteCopyOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteCopyOp(const RemoteCopyOp &rhs);
      virtual ~RemoteCopyOp(void);
    public:
      RemoteCopyOp& operator=(const RemoteCopyOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteCloseOp
     * This is a remote copy of a CloseOp to be used
     * for mapper calls and other operations
     */
    class RemoteCloseOp : public ExternalClose, public RemoteOp,
                          public LegionHeapify<RemoteCloseOp> {
    public:
      RemoteCloseOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteCloseOp(const RemoteCloseOp &rhs);
      virtual ~RemoteCloseOp(void);
    public:
      RemoteCloseOp& operator=(const RemoteCloseOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteAcquireOp
     * This is a remote copy of a AcquireOp to be used
     * for mapper calls and other operations
     */
    class RemoteAcquireOp : public ExternalAcquire, public RemoteOp,
                            public LegionHeapify<RemoteAcquireOp> {
    public:
      RemoteAcquireOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteAcquireOp(const RemoteAcquireOp &rhs);
      virtual ~RemoteAcquireOp(void);
    public:
      RemoteAcquireOp& operator=(const RemoteAcquireOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteReleaseOp
     * This is a remote copy of a ReleaseOp to be used
     * for mapper calls and other operations
     */
    class RemoteReleaseOp : public ExternalRelease, public RemoteOp,
                            public LegionHeapify<RemoteReleaseOp> {
    public:
      RemoteReleaseOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteReleaseOp(const RemoteReleaseOp &rhs);
      virtual ~RemoteReleaseOp(void);
    public:
      RemoteReleaseOp& operator=(const RemoteReleaseOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteFillOp
     * This is a remote copy of a FillOp to be used
     * for mapper calls and other operations
     */
    class RemoteFillOp : public ExternalFill, public RemoteOp,
                         public LegionHeapify<RemoteFillOp> {
    public:
      RemoteFillOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteFillOp(const RemoteFillOp &rhs);
      virtual ~RemoteFillOp(void);
    public:
      RemoteFillOp& operator=(const RemoteFillOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemotePartitionOp
     * This is a remote copy of a DependentPartitionOp to be
     * used for mapper calls and other operations
     */
    class RemotePartitionOp : public ExternalPartition, public RemoteOp,
                              public LegionHeapify<RemotePartitionOp> {
    public:
      RemotePartitionOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemotePartitionOp(const RemotePartitionOp &rhs);
      virtual ~RemotePartitionOp(void);
    public:
      RemotePartitionOp& operator=(const RemotePartitionOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
      virtual const Task* get_parent_task(void) const;
      virtual PartitionKind get_partition_kind(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    protected:
      PartitionKind part_kind;
    };

    /**
     * \class RemoteAttachOp
     * This is a remote copy of a DetachOp to be used for 
     * mapper calls and other operations
     */
    class RemoteAttachOp : public RemoteOp,
                           public LegionHeapify<RemoteAttachOp> {
    public:
      RemoteAttachOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteAttachOp(const RemoteAttachOp &rhs);
      virtual ~RemoteAttachOp(void);
    public:
      RemoteAttachOp& operator=(const RemoteAttachOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteDetachOp
     * This is a remote copy of a DetachOp to be used for 
     * mapper calls and other operations
     */
    class RemoteDetachOp : public RemoteOp,
                           public LegionHeapify<RemoteDetachOp> {
    public:
      RemoteDetachOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteDetachOp(const RemoteDetachOp &rhs);
      virtual ~RemoteDetachOp(void);
    public:
      RemoteDetachOp& operator=(const RemoteDetachOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteDeletionOp
     * This is a remote copy of a DeletionOp to be used for 
     * mapper calls and other operations
     */
    class RemoteDeletionOp : public RemoteOp,
                             public LegionHeapify<RemoteDeletionOp> {
    public:
      RemoteDeletionOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteDeletionOp(const RemoteDeletionOp &rhs);
      virtual ~RemoteDeletionOp(void);
    public:
      RemoteDeletionOp& operator=(const RemoteDeletionOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteReplayOp
     * This is a remote copy of a trace replay op, it really doesn't
     * have to do very much at all other than implement the interface
     * for remote ops as it will only be used for checking equivalence
     * sets for valid physical template replay conditions
     */
    class RemoteReplayOp : public RemoteOp,
                           public LegionHeapify<RemoteReplayOp> {
    public:
      RemoteReplayOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteReplayOp(const RemoteReplayOp &rhs);
      virtual ~RemoteReplayOp(void);
    public:
      RemoteReplayOp& operator=(const RemoteReplayOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

    /**
     * \class RemoteSummaryOp
     * This is a remote copy of a trace summary op, it really doesn't
     * have to do very much at all other than implement the interface
     * for remote ops as it will only be used for updating state for
     * physical template replays
     */
    class RemoteSummaryOp : public RemoteOp,
                            public LegionHeapify<RemoteSummaryOp> {
    public:
      RemoteSummaryOp(Runtime *rt, Operation *ptr, AddressSpaceID src);
      RemoteSummaryOp(const RemoteSummaryOp &rhs);
      virtual ~RemoteSummaryOp(void);
    public:
      RemoteSummaryOp& operator=(const RemoteSummaryOp &rhs);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual size_t get_context_index(void) const;
      virtual void set_context_index(size_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void select_sources(const unsigned index,
                                  const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
      virtual void unpack(Deserializer &derez, ReferenceMutator &mutator);
    };

  }; //namespace Internal 
}; // namespace Legion 

#include "legion_ops.inl"

#endif // __LEGION_OPERATIONS_H__
