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


#ifndef __LEGION_OPERATIONS_H__
#define __LEGION_OPERATIONS_H__

#include "legion.h"
#include "region_tree.h"
#include "legion_mapping.h"
#include "legion_utilities.h"
#include "legion_allocation.h"
#include "legion_analysis.h"
#include "mapper_manager.h"

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
    class Operation : public ReferenceMutator {
    public:
      enum OpKind {
        MAP_OP_KIND,
        COPY_OP_KIND,
        FENCE_OP_KIND,
        FRAME_OP_KIND,
        DELETION_OP_KIND,
        INTER_CLOSE_OP_KIND,
        READ_CLOSE_OP_KIND,
        POST_CLOSE_OP_KIND,
        VIRTUAL_CLOSE_OP_KIND,
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
        "Inter Close",              \
        "Read Close",               \
        "Post Close",               \
        "Virtual Close",            \
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
        "Task",                     \
      }
    public:
      struct DeferredMappingArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
      struct DeferredResolutionArgs {
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
      struct DeferredExecuteArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
      struct TriggerCompleteArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
      struct DeferredCompleteArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
      struct DeferredCommitTriggerArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
        GenerationID gen;
      };
      struct DeferredCommitArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
        bool deactivate;
      };
      struct StateAnalysisArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_op;
        RtUserEvent ready_event;
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
      virtual const char* get_logging_name(void) = 0;
      virtual OpKind get_operation_kind(void) = 0;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
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
      inline SingleTask* get_parent(void) const { return parent_ctx; }
      inline UniqueID get_unique_op_id(void) const { return unique_op_id; } 
      inline bool is_tracing(void) const { return tracing; }
      inline bool is_tracking_parent(void) const { return track_parent; } 
      inline bool already_traced(void) const 
        { return ((trace != NULL) && !tracing); }
      inline LegionTrace* get_trace(void) const { return trace; }
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
      void set_trace(LegionTrace *trace, bool is_tracing);
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
      void initialize_operation(SingleTask *ctx, bool track,
                                unsigned num_regions = 0); 
    public:
      // Inherited from ReferenceMutator
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      // The following two calls may be implemented
      // differently depending on the operation, but we
      // provide base versions of them so that operations
      // only have to overload the stages that they care
      // about modifying.
      // The function to call for depence analysis
      virtual void trigger_dependence_analysis(void);
      // The function to call when the operation is ready to map 
      // In general put this on the ready queue so the runtime
      // can invoke the trigger mapping call.
      virtual void trigger_mapping(void);
      // For operations which are operating in a context which
      // has remote state, this call will be invoked by the 
      // runtime prior to calling trigger_execution to allow
      // the operation to specify an event precondition to wait
      // on for all remote state to arrive on the necessary node.
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      // The function to call for executing an operation
      // Note that this one is not invoked by the Operation class
      // but by the runtime, therefore any operations must be
      // placed on the ready queue in order for the runtime to
      // perform this mapping
      virtual bool trigger_execution(void);
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
      // A method for reporting other region requirements that 
      // interfere with a close operation generated by a 
      // different region requirement in the same op.
      virtual void report_interfering_close_requirement(unsigned idx);
      // A method for finding the parent index of a region
      // requirement for an operation which is necessary for
      // issuing close operation on behalf of the operation.
      virtual unsigned find_parent_index(unsigned idx);
      // A method for computing a fat region tree path for a
      // region requirement. This is only called for projection
      // region requirements so pretty much only index tasks.
      // The caller owns the resulting data structure.
      virtual FatTreePath* compute_fat_path(unsigned idx);
      // This is a special helper method for tracing which
      // needs to know explicitly about close operations
      virtual bool is_close_op(void) const { return false; }
      // Determine if this operation is a partition operation
      virtual bool is_partition_op(void) const { return false; }
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
    public:
      // The following are sets of calls that we can use to 
      // indicate mapping, execution, resolution, completion, and commit
      //
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
      void add_mapping_reference(GenerationID gen);
      void remove_mapping_reference(GenerationID gen);
      // Ask the operation to perform the state analysis
      RtEvent invoke_state_analysis(void);
    public:
      // Some extra support for tracking dependences that we've 
      // registered as part of our logical traversal
      void record_logical_dependence(const LogicalUser &user);
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned& 
                                    get_logical_records(void);
      void clear_logical_records(void);
    public:
      // Notify when a region from a dependent task has 
      // been verified (flows up edges)
      void notify_regions_verified(const std::set<unsigned> &regions,
                                   GenerationID gen);
    public: // Support for mapping operations
      static void prepare_for_mapping(const InstanceRef &ref,
                                      MappingInstance &instance);
      static void prepare_for_mapping(const InstanceSet &valid,
                           std::vector<MappingInstance> &input_valid);
      static void prepare_for_mapping(const InstanceSet &valid,
                           const std::set<Memory> &filter_memories,
                           std::vector<MappingInstance> &input_valid);
      static void compute_ranking(
          const std::deque<MappingInstance>         &output,
          const InstanceSet                         &sources,
          std::vector<unsigned>                     &ranking);
    public:
      Runtime *const runtime;
    protected:
      Reservation op_lock;
      GenerationID gen;
      UniqueID unique_op_id;
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
      SingleTask *parent_ctx;
      // The mapped event for this operation
      RtUserEvent mapped_event;
      // The resolved event for this operation
      RtUserEvent resolved_event;
      // The event for when any children this operation has are mapped
      //Event children_mapped;
      // The completion event for this operation
      ApUserEvent completion_event;
      // The commit event for this operation
      RtUserEvent commit_event;
      // The trace for this operation if any
      LegionTrace *trace;
      // Track whether we are tracing this operation
      bool tracing;
      // Our must epoch if we have one
      MustEpochOp *must_epoch;
      // A set list or recorded dependences during logical traversal
      LegionList<LogicalUser,LOGICAL_REC_ALLOC>::track_aligned logical_records;
      // A dependence tracker for this operation
      union {
        MappingDependenceTracker *mapping;
        CommitDependenceTracker  *commit;
      } dependence_tracker;
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
    public:
      bool register_waiter(PredicateWaiter *waiter, 
                           GenerationID gen, bool &value);
    protected:
      void set_resolved_value(GenerationID pred_gen, bool value);
    protected:
      bool predicate_resolved;
      bool predicate_value;
      std::map<PredicateWaiter*,GenerationID> waiters;
    protected:
      unsigned predicate_references;
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
        PENDING_MAP_STATE,
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
      void initialize_speculation(SingleTask *ctx, bool track, unsigned regions,
                                  const Predicate &p);
      void register_predicate_dependence(void);
      bool is_predicated(void) const;
      // Wait until the predicate is valid and then return
      // its value.  Give it the current processor in case it
      // needs to wait for the value
      bool get_predicate_value(Processor proc);
    public:
      // Override the mapping call so we can decide whether
      // to continue mapping this operation or not 
      // depending on the value of the predicate operation.
      virtual void trigger_mapping(void);
      virtual void trigger_resolution(void);
      virtual void deferred_execute(void);
    public:
      // Call this method for inheriting classes 
      // to indicate when they should map
      virtual bool speculate(bool &value) = 0;
      virtual void resolve_true(void) = 0;
      virtual void resolve_false(void) = 0;
    public:
      virtual void notify_predicate_value(GenerationID gen, bool value);
    protected:
      SpecState    speculation_state;
      PredicateOp *predicate;
      bool received_trigger_resolution;
    protected:
      RtUserEvent predicate_waiter; // used only when needed
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
    class MapOp : public InlineMapping, public Operation {
    public:
      static const AllocationType alloc_type = MAP_OP_ALLOC;
    public:
      MapOp(Runtime *rt);
      MapOp(const MapOp &rhs);
      virtual ~MapOp(void);
    public:
      MapOp& operator=(const MapOp &rhs);
    public:
      PhysicalRegion initialize(SingleTask *ctx,
                                const InlineLauncher &launcher,
                                bool check_privileges);
      PhysicalRegion initialize(SingleTask *ctx,
                                const RegionRequirement &req,
                                MapperID id, MappingTagID tag,
                                bool check_privileges);
      void initialize(SingleTask *ctx, const PhysicalRegion &region);
      inline const RegionRequirement& get_requirement(void) const
        { return requirement; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
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
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual int get_depth(void) const;
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(const InstanceSet &valid_instances,
                               InstanceSet &mapped_instances);
      void report_profiling_results(void);
    protected:
      bool remap_region;
      ApUserEvent termination_event;
      PhysicalRegion region;
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      VersionInfo version_info;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::map<Reservation,bool> atomic_locks;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      Mapper::InlineProfilingInfo profiling_results;
      RtUserEvent                 profiling_reported;
    };

    /**
     * \class CopyOp
     * The copy operation provides a mechanism for applications
     * to directly copy data between pairs of fields possibly
     * from different region trees in an efficient way by
     * using the low-level runtime copy facilities. 
     */
    class CopyOp : public Copy, public SpeculativeOp {
    public:
      static const AllocationType alloc_type = COPY_OP_ALLOC;
    public:
      CopyOp(Runtime *rt);
      CopyOp(const CopyOp &rhs);
      virtual ~CopyOp(void);
    public:
      CopyOp& operator=(const CopyOp &rhs);
    public:
      void initialize(SingleTask *ctx,
                      const CopyLauncher &launcher,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual void trigger_commit(void);
      virtual void report_interfering_requirements(unsigned idx1,unsigned idx2);
      virtual void report_interfering_close_requirement(unsigned idx);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void update_atomic_locks(Reservation lock, bool exclusive);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual int get_depth(void) const;
    protected:
      void check_copy_privilege(const RegionRequirement &req, 
                                unsigned idx, bool src);
      void compute_parent_indexes(void);
      template<bool IS_SRC>
      int perform_conversion(unsigned idx, const RegionRequirement &req,
                             std::vector<MappingInstance> &output,
                             InstanceSet &targets, bool is_reduce = false);
      inline void set_mapping_state(unsigned idx, bool is_src) 
        { current_index = idx; current_src = is_src; }
      void report_profiling_results(void);
    public:
      std::vector<RegionTreePath> src_privilege_paths;
      std::vector<RegionTreePath> dst_privilege_paths;
      std::vector<unsigned>       src_parent_indexes;
      std::vector<unsigned>       dst_parent_indexes;
      std::vector<VersionInfo>    src_versions;
      std::vector<VersionInfo>    dst_versions;
    protected: // for support with mapping
      MapperManager*              mapper;
      unsigned                    current_index;
      bool                        current_src;
    protected:
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::vector<std::map<Reservation,bool> > atomic_locks;
      std::set<RtEvent> map_applied_conditions;
    protected:
      Mapper::CopyProfilingInfo   profiling_results;
      RtUserEvent                 profiling_reported;
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
    class FenceOp : public Operation {
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
      void initialize(SingleTask *ctx, FenceKind kind);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void deferred_execute(void);
    protected:
      FenceKind fence_kind;
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
      void initialize(SingleTask *ctx);
      void set_previous(ApEvent previous);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual bool trigger_execution(void);
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
    class DeletionOp : public Operation {
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
      void initialize_index_space_deletion(SingleTask *ctx, IndexSpace handle);
      void initialize_index_part_deletion(SingleTask *ctx,
                                          IndexPartition handle);
      void initialize_field_space_deletion(SingleTask *ctx,
                                           FieldSpace handle);
      void initialize_field_deletion(SingleTask *ctx, FieldSpace handle,
                                      FieldID fid);
      void initialize_field_deletions(SingleTask *ctx, FieldSpace handle,
                                      const std::set<FieldID> &to_free);
      void initialize_logical_region_deletion(SingleTask *ctx, 
                                              LogicalRegion handle);
      void initialize_logical_partition_deletion(SingleTask *ctx, 
                                                 LogicalPartition handle);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
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
    }; 

    /**
     * \class CloseOp
     * Close operations are only visible internally inside
     * the runtime and are issued to help close up the 
     * physical region tree. There are two types of close
     * operations that both inherit from this class:
     * InterCloseOp and PostCloseOp.
     */
    class CloseOp : public Close, public Operation {
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
      virtual int get_depth(void) const;
    public:
      void activate_close(void);
      void deactivate_close(void);
      void initialize_close(SingleTask *ctx,
                            const RegionRequirement &req, bool track);
      void initialize_close(SingleTask *ctx, unsigned idx, bool track);
      void perform_logging(bool is_intermediate_close_op, bool read_only);
    public:
      // For recording trace dependences
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void) = 0;
      virtual OpKind get_operation_kind(void) = 0;
      virtual size_t get_region_count(void) const;
      virtual bool is_close_op(void) const { return true; }
    public:
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual void trigger_commit(void);
    protected:
      RegionTreePath privilege_path;
      VersionInfo    version_info;
      RestrictInfo   restrict_info;
    };

    /**
     * \class TraceCloseOp
     * This is a pure virtual class for handling the common
     * operations associated with tracing for close operations.
     */
    class TraceCloseOp : public CloseOp {
    public:
      TraceCloseOp(Runtime *runtime);
      virtual ~TraceCloseOp(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0;
      virtual const char* get_logging_name(void) = 0;
      virtual OpKind get_operation_kind(void) = 0;
    public:
      void initialize_trace_close_op(SingleTask *ctx, 
                                     const RegionRequirement &req,
                 const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                                     LegionTrace *trace, int close_idx,
                                     const FieldMask &close_mask,
                                     Operation *create_op);
      void activate_trace_close(void);
      void deactivate_trace_close(void);
    public:
      void record_trace_dependence(Operation *target, GenerationID target_gen,
                                   int target_idx, int source_idx, 
                                   DependenceType dtype,
                                   const FieldMask &dependent_mask);
      void add_next_child(const ColorPoint &next_child);
    public:
      inline const RegionRequirement& get_region_requirement(void) const
        { return requirement; }
      inline const LegionMap<ColorPoint,FieldMask>::aligned& 
        get_target_children(void) const { return target_children; }
      inline int get_close_index(void) const { return close_idx; }
    protected:
      // Points to close, and the fields to leave open
      LegionMap<ColorPoint,FieldMask/*leave open*/>::aligned target_children;
      std::set<ColorPoint> next_children;
    protected:
      // These things are really only needed for tracing
      // The source index from the original 
      // operation that generated this close operation
      int close_idx;
      // The field mask for the fields we are closing
      FieldMask close_mask;
      // Information about the operation that generated
      // this close operation so we don't register dependences on it
      Operation *create_op;
      GenerationID create_gen;
    };

    /**
     * \class InterCloseOp
     * Intermediate close operations are issued by the runtime
     * for closing up region trees as part of the normal execution
     * of an application.
     */
    class InterCloseOp : public TraceCloseOp {
    public:
      InterCloseOp(Runtime *runtime);
      InterCloseOp(const InterCloseOp &rhs);
      virtual ~InterCloseOp(void);
    public:
      InterCloseOp& operator=(const InterCloseOp &rhs);
    public:
      void initialize(SingleTask *ctx, const RegionRequirement &req,
                      const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                      LegionTrace *trace, int close_idx, 
                      const VersionInfo &close_info,
                      const VersionInfo &version_info,
                      const RestrictInfo &restrict_info,
                      const FieldMask &close_mask, Operation *create_op);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual bool trigger_execution(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    protected:
      int invoke_mapper(const InstanceSet &valid_instances,
                              InstanceSet &chosen_instances);
      void report_profiling_results(void);
    protected:
      unsigned parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      Mapper::CloseProfilingInfo  profiling_results;
      RtUserEvent                 profiling_reported;
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
    class ReadCloseOp : public TraceCloseOp {
    public:
      ReadCloseOp(Runtime *runtime);
      ReadCloseOp(const ReadCloseOp &rhs);
      virtual ~ReadCloseOp(void);
    public:
      ReadCloseOp& operator=(const ReadCloseOp &rhs);
    public:
      void initialize(SingleTask *ctx, const RegionRequirement &req,
                      const LegionMap<ColorPoint,FieldMask>::aligned &targets,
                      LegionTrace *trace, int close_idx,
                      const FieldMask &close_mask, Operation *create_op);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual unsigned find_parent_index(unsigned idx);
    protected:
      unsigned parent_req_index; 
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
      void initialize(SingleTask *ctx, unsigned index); 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    protected:
      void report_profiling_results(void);
    protected:
      unsigned parent_idx;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager *mapper;
    protected:
      Mapper::CloseProfilingInfo  profiling_results;
      RtUserEvent                 profiling_reported;
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
      void initialize(SingleTask *ctx, unsigned index);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void record_reference_mutation_effect(RtEvent event);
    protected:
      unsigned parent_idx;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class AcquireOp
     * Acquire operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class AcquireOp : public Acquire, public SpeculativeOp {
    public:
      static const AllocationType alloc_type = ACQUIRE_OP_ALLOC;
    public:
      AcquireOp(Runtime *rt);
      AcquireOp(const AcquireOp &rhs);
      virtual ~AcquireOp(void);
    public:
      AcquireOp& operator=(const AcquireOp &rhs);
    public:
      void initialize(SingleTask *ctx, const AcquireLauncher &launcher,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void); 
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public: 
      virtual UniqueID get_unique_id(void) const;
      virtual int get_depth(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    protected:
      void check_acquire_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      void report_profiling_results(void);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      Mapper::AcquireProfilingInfo  profiling_results;
      RtUserEvent                   profiling_reported;
    };

    /**
     * \class ReleaseOp
     * Release operations are used for performing
     * user-level software coherence when tasks own
     * regions with simultaneous coherence.
     */
    class ReleaseOp : public Release, public SpeculativeOp {
    public:
      static const AllocationType alloc_type = RELEASE_OP_ALLOC;
    public:
      ReleaseOp(Runtime *rt);
      ReleaseOp(const ReleaseOp &rhs);
      virtual ~ReleaseOp(void);
    public:
      ReleaseOp& operator=(const ReleaseOp &rhs);
    public:
      void initialize(SingleTask *ctx, const ReleaseLauncher &launcher,
                      bool check_privileges);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual void trigger_commit(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void select_sources(const InstanceRef &target,
                                  const InstanceSet &sources,
                                  std::vector<unsigned> &ranking);
      virtual std::map<PhysicalManager*,std::pair<unsigned,bool> >*
                   get_acquired_instances_ref(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual int get_depth(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    protected:
      void check_release_privilege(void);
      void compute_parent_index(void);
      void invoke_mapper(void);
      void report_profiling_results(void);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
      VersionInfo       version_info;
      unsigned          parent_req_index;
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
      std::set<RtEvent> map_applied_conditions;
    protected:
      MapperManager*    mapper;
    protected:
      Mapper::ReleaseProfilingInfo  profiling_results;
      RtUserEvent                   profiling_reported;
    };

    /**
     * \class DynamicCollectiveOp
     * A class for getting values from a collective operation
     * and writing them into a future. This will also give
     * us the framework necessary to handle roll backs on 
     * collectives so we can memoize their results.
     */
    class DynamicCollectiveOp : public Operation {
    public:
      static const AllocationType alloc_type = DYNAMIC_COLLECTIVE_OP_ALLOC;
    public:
      DynamicCollectiveOp(Runtime *rt);
      DynamicCollectiveOp(const DynamicCollectiveOp &rhs);
      virtual ~DynamicCollectiveOp(void);
    public:
      DynamicCollectiveOp& operator=(const DynamicCollectiveOp &rhs);
    public:
      Future initialize(SingleTask *ctx, const DynamicCollective &dc);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual bool trigger_execution(void);
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
    class FuturePredOp : public PredicateOp {
    public:
      static const AllocationType alloc_type = FUTURE_PRED_OP_ALLOC;
    public:
      struct ResolveFuturePredArgs {
        HLRTaskID hlr_id;
        FuturePredOp *future_pred_op;
      };
    public:
      FuturePredOp(Runtime *rt);
      FuturePredOp(const FuturePredOp &rhs);
      virtual ~FuturePredOp(void);
    public:
      FuturePredOp& operator=(const FuturePredOp &rhs);
    public:
      void initialize(SingleTask *ctx, Future f);
      void resolve_future_predicate(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      const char* get_logging_name(void);
      OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      Future future;
    };

    /**
     * \class NotPredOp
     * A class for negating other predicates
     */
    class NotPredOp : public PredicateOp, PredicateWaiter {
    public:
      static const AllocationType alloc_type = NOT_PRED_OP_ALLOC;
    public:
      NotPredOp(Runtime *rt);
      NotPredOp(const NotPredOp &rhs);
      virtual ~NotPredOp(void);
    public:
      NotPredOp& operator=(const NotPredOp &rhs);
    public:
      void initialize(SingleTask *task, const Predicate &p);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void notify_predicate_value(GenerationID gen, bool value);
    protected:
      PredicateOp *pred_op;
    };

    /**
     * \class AndPredOp
     * A class for and-ing other predicates
     */
    class AndPredOp : public PredicateOp, PredicateWaiter {
    public:
      static const AllocationType alloc_type = AND_PRED_OP_ALLOC;
    public:
      AndPredOp(Runtime *rt);
      AndPredOp(const AndPredOp &rhs);
      virtual ~AndPredOp(void);
    public:
      AndPredOp& operator=(const AndPredOp &rhs);
    public:
      void initialize(SingleTask *task, 
                      const Predicate &p1, const Predicate &p2);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void notify_predicate_value(GenerationID pred_gen, bool value);
    protected:
      PredicateOp *left;
      PredicateOp *right;
    protected:
      bool left_value;
      bool left_valid;
      bool right_value;
      bool right_valid;
    };

    /**
     * \class OrPredOp
     * A class for or-ing other predicates
     */
    class OrPredOp : public PredicateOp, PredicateWaiter {
    public:
      static const AllocationType alloc_type = OR_PRED_OP_ALLOC;
    public:
      OrPredOp(Runtime *rt);
      OrPredOp(const OrPredOp &rhs);
      virtual ~OrPredOp(void);
    public:
      OrPredOp& operator=(const OrPredOp &rhs);
    public:
      void initialize(SingleTask *task,
                      const Predicate &p1, const Predicate &p2);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void notify_predicate_value(GenerationID pred_gen, bool value);
    protected:
      PredicateOp *left;
      PredicateOp *right;
    protected:
      bool left_value;
      bool left_valid;
      bool right_value;
      bool right_valid;
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
    class MustEpochOp : public Operation {
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
      FutureMap initialize(SingleTask *ctx,
                                   const MustEpochLauncher &launcher,
                                   bool check_privileges);
      void find_conflicted_regions(
          std::vector<PhysicalRegion> &unmapped); 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
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
      std::vector<Mapper::MappingConstraint> constraints;
      // Used for computing the constraints
      std::vector<std::set<SingleTask*> > task_sets;
      // Track the physical instances that we've acquired
      std::map<PhysicalManager*,std::pair<unsigned,bool> > acquired_instances;
    protected:
      std::map<std::pair<unsigned/*task index*/,unsigned/*req index*/>,
               unsigned/*dependence index*/> dependence_map;
      std::vector<DependenceRecord*> dependences;
      std::map<SingleTask*,unsigned/*single task index*/> single_task_map;
    };

    /**
     * \class MustEpochTriggerer
     * A helper class for parallelizing must epoch triggering
     */
    class MustEpochTriggerer {
    public:
      struct MustEpochIndivArgs {
      public:
        HLRTaskID hlr_id;
        MustEpochTriggerer *triggerer;
        IndividualTask *task;
      };
      struct MustEpochIndexArgs {
        HLRTaskID hlr_id;
        MustEpochTriggerer *triggerer;
        IndexTask *task;
      };
    public:
      MustEpochTriggerer(MustEpochOp *owner);
      MustEpochTriggerer(const MustEpochTriggerer &rhs);
      ~MustEpochTriggerer(void);
    public:
      MustEpochTriggerer& operator=(const MustEpochTriggerer &rhs);
    public:
      bool trigger_tasks(const std::vector<IndividualTask*> &indiv_tasks,
                         std::vector<bool> &indiv_triggered,
                         const std::vector<IndexTask*> &index_tasks,
                         std::vector<bool> &index_triggered);
      void trigger_individual(IndividualTask *task);
      void trigger_index(IndexTask *task);
    public:
      static void handle_individual(const void *args);
      static void handle_index(const void *args);
    private:
      MustEpochOp *const owner;
      Reservation trigger_lock;
      std::set<IndividualTask*> failed_individual_tasks;
      std::set<IndexTask*> failed_index_tasks;
    };

    /**
     * \class MustEpochMapper
     * A helper class for parallelizing mapping for must epochs
     */
    class MustEpochMapper {
    public:
      struct MustEpochMapArgs {
      public:
        HLRTaskID hlr_id;
        MustEpochMapper *mapper;
        SingleTask *task;
      };
    public:
      MustEpochMapper(MustEpochOp *owner);
      MustEpochMapper(const MustEpochMapper &rhs);
      ~MustEpochMapper(void);
    public:
      MustEpochMapper& operator=(const MustEpochMapper &rhs);
    public:
      bool map_tasks(const std::deque<SingleTask*> &single_tasks);
      void map_task(SingleTask *task);
    public:
      static void handle_map_task(const void *args);
    private:
      MustEpochOp *const owner;
      bool success;
    };

    class MustEpochDistributor {
    public:
      struct MustEpochDistributorArgs {
      public:
        HLRTaskID hlr_id;
        TaskOp *task;
      };
      struct MustEpochLauncherArgs {
      public:
        HLRTaskID hlr_id;
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
    class PendingPartitionOp : public Operation {
    public:
      static const AllocationType alloc_type = PENDING_PARTITION_OP_ALLOC;
    protected:
      // Track pending partition operations as thunks
      class PendingPartitionThunk {
      public:
        virtual ~PendingPartitionThunk(void) { }
      public:
        virtual ApEvent perform(RegionTreeForest *forest) = 0;
        virtual void perform_logging(PendingPartitionOp* op) = 0;
      };
      class EqualPartitionThunk : public PendingPartitionThunk {
      public:
        EqualPartitionThunk(IndexPartition id, size_t g)
          : pid(id), granularity(g) { }
        virtual ~EqualPartitionThunk(void) { }
      public:
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_equal_partition(pid, granularity); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        size_t granularity;
      };
      class WeightedPartitionThunk : public PendingPartitionThunk {
      public:
        WeightedPartitionThunk(IndexPartition id, size_t g, 
                               const std::map<DomainPoint,int> &w)
          : pid(id), weights(w), granularity(g) { }
        virtual ~WeightedPartitionThunk(void) { }
      public:
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_weighted_partition(pid, granularity, weights); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        std::map<DomainPoint,int> weights;
        size_t granularity;
      };
      class UnionPartitionThunk : public PendingPartitionThunk {
      public:
        UnionPartitionThunk(IndexPartition id, 
                            IndexPartition h1, IndexPartition h2)
          : pid(id), handle1(h1), handle2(h2) { }
        virtual ~UnionPartitionThunk(void) { }
      public:
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_partition_by_union(pid, handle1, handle2); }
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
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_partition_by_intersection(pid, handle1, 
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
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_partition_by_difference(pid, handle1, 
                                                        handle2); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition pid;
        IndexPartition handle1;
        IndexPartition handle2;
      };
      class CrossProductThunk : public PendingPartitionThunk {
      public:
        CrossProductThunk(IndexPartition b, IndexPartition s,
                          std::map<DomainPoint,IndexPartition> &h)
          : base(b), source(s), handles(h) { }
        virtual ~CrossProductThunk(void) { }
      public:
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->create_cross_product_partitions(base, source, 
                                                         handles); }
        virtual void perform_logging(PendingPartitionOp* op);
      protected:
        IndexPartition base;
        IndexPartition source;
        std::map<DomainPoint,IndexPartition> handles;
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
        virtual ApEvent perform(RegionTreeForest *forest)
        { if (is_partition)
            return forest->compute_pending_space(target, handle, is_union);
          else
            return forest->compute_pending_space(target, handles, is_union); }
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
        virtual ApEvent perform(RegionTreeForest *forest)
        { return forest->compute_pending_space(target, initial, handles); }
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
      void initialize_equal_partition(SingleTask *ctx,
                                      IndexPartition pid, size_t granularity);
      void initialize_weighted_partition(SingleTask *ctx,
                                         IndexPartition pid, size_t granularity,
                                      const std::map<DomainPoint,int> &weights);
      void initialize_union_partition(SingleTask *ctx,
                                      IndexPartition pid, 
                                      IndexPartition handle1,
                                      IndexPartition handle2);
      void initialize_intersection_partition(SingleTask *ctx,
                                             IndexPartition pid, 
                                             IndexPartition handle1,
                                             IndexPartition handle2);
      void initialize_difference_partition(SingleTask *ctx,
                                           IndexPartition pid, 
                                           IndexPartition handle1,
                                           IndexPartition handle2);
      void initialize_cross_product(SingleTask *ctx,
                                    IndexPartition base, IndexPartition source,
                                std::map<DomainPoint,IndexPartition> &handles);
      void initialize_index_space_union(SingleTask *ctx, IndexSpace target, 
                                        const std::vector<IndexSpace> &handles);
      void initialize_index_space_union(SingleTask *ctx, IndexSpace target, 
                                        IndexPartition handle);
      void initialize_index_space_intersection(SingleTask *ctx, 
                                               IndexSpace target,
                                        const std::vector<IndexSpace> &handles);
      void initialize_index_space_intersection(SingleTask *ctx,
                                              IndexSpace target,
                                              IndexPartition handle);
      void initialize_index_space_difference(SingleTask *ctx, 
                                             IndexSpace target, 
                                             IndexSpace initial,
                                        const std::vector<IndexSpace> &handles);
      void perform_logging();
      inline ApEvent get_handle_ready(void) const { return handle_ready; }
    public:
      virtual bool trigger_execution(void);
      virtual bool is_partition_op(void) const { return true; } 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    protected:
      ApUserEvent handle_ready;
      PendingPartitionThunk *thunk;
    };

    /**
     * \class DependentPartitionOp
     * An operation for creating different kinds of partitions
     * which are dependent on mapping a region in order to compute
     * the resulting partition.
     */
    class DependentPartitionOp : public Operation {
    public:
      static const AllocationType alloc_type = DEPENDENT_PARTITION_OP_ALLOC;
    public:
      enum PartOpKind {
        BY_FIELD,
        BY_IMAGE,
        BY_PREIMAGE,
      };
    public:
      DependentPartitionOp(Runtime *rt);
      DependentPartitionOp(const DependentPartitionOp &rhs);
      virtual ~DependentPartitionOp(void);
    public:
      DependentPartitionOp& operator=(const DependentPartitionOp &rhs);
    public:
      void initialize_by_field(SingleTask *ctx, IndexPartition pid,
                               LogicalRegion handle, LogicalRegion parent,
                               const Domain &color_space, FieldID fid); 
      void initialize_by_image(SingleTask *ctx, IndexPartition pid,
                               LogicalPartition projection,
                               LogicalRegion parent, FieldID fid,
                               const Domain &color_space);
      void initialize_by_preimage(SingleTask *ctx, IndexPartition pid,
                               IndexPartition projection, LogicalRegion handle,
                               LogicalRegion parent, FieldID fid,
                               const Domain &color_space);
      void perform_logging();
      const RegionRequirement& get_requirement(void) const;
      inline ApEvent get_handle_ready(void) const { return handle_ready; }
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual bool is_partition_op(void) const { return true; }
      virtual FatTreePath* compute_fat_path(unsigned idx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
      virtual size_t get_region_count(void) const;
      virtual void trigger_commit(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    protected:
      void compute_parent_index(void);
    protected:
      ApUserEvent handle_ready;
      PartOpKind partition_kind;
      RegionRequirement requirement;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      IndexPartition partition_handle;
      Domain color_space;
      IndexPartition projection; /* for pre-image only*/
      RegionTreePath privilege_path;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class FillOp
     * Fill operations are used to initialize a field to a
     * specific value for a particular logical region.
     */
    class FillOp : public SpeculativeOp {
    public:
      static const AllocationType alloc_type = FILL_OP_ALLOC;
    public:
      FillOp(Runtime *rt);
      FillOp(const FillOp &rhs);
      virtual ~FillOp(void);
    public:
      FillOp& operator=(const FillOp &rhs);
    public:
      void initialize(SingleTask *ctx, LogicalRegion handle,
                      LogicalRegion parent, FieldID fid,
                      const void *ptr, size_t size,
                      const Predicate &pred, bool check_privileges);
      void initialize(SingleTask *ctx, LogicalRegion handle,
                      LogicalRegion parent, FieldID fid,const Future &f,
                      const Predicate &pred, bool check_privileges);
      void initialize(SingleTask *ctx, LogicalRegion handle,
                      LogicalRegion parent, 
                      const std::set<FieldID> &fields,
                      const void *ptr, size_t size,
                      const Predicate &pred, bool check_privileges);
      void initialize(SingleTask *ctx, LogicalRegion handle,
                      LogicalRegion parent, 
                      const std::set<FieldID> &fields, const Future &f,
                      const Predicate &pred, bool check_privileges);
      void initialize(SingleTask *ctx, const FillLauncher &launcher,
                      bool check_privileges);
      void perform_logging(void);
      inline const RegionRequirement& get_requirement(void) const 
        { return requirement; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual void deferred_execute(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_commit(void);
    public:
      void check_fill_privilege(void);
      void compute_parent_index(void);
      ApEvent compute_sync_precondition(void) const;
    protected:
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      unsigned parent_req_index;
      void *value;
      size_t value_size;
      Future future;
      std::set<RtEvent> map_applied_conditions;
    protected:
      std::vector<Grant>        grants;
      std::vector<PhaseBarrier> wait_barriers;
      std::vector<PhaseBarrier> arrive_barriers;
    };

    /**
     * \class AttachOp
     * Operation for attaching a file to a physical instance
     */
    class AttachOp : public Operation {
    public:
      static const AllocationType alloc_type = ATTACH_OP_ALLOC;
      enum ExternalType {
        HDF5_FILE,
        NORMAL_FILE,
        IN_MEMORY_DATA
      };
    public:
      AttachOp(Runtime *rt);
      AttachOp(const AttachOp &rhs);
      virtual ~AttachOp(void);
    public:
      AttachOp& operator=(const AttachOp &rhs);
    public:
      PhysicalRegion initialize_hdf5(
                                 SingleTask *ctx, const char *file_name,
                                 LogicalRegion handle, LogicalRegion parent,
                                 const std::map<FieldID,const char*> &field_map,
                                 LegionFileMode mode, bool check_privileges);
      PhysicalRegion initialize_file(
                                     SingleTask *ctx, const char *file_name,
                                     LogicalRegion handle, LogicalRegion parent,
                                     const std::vector<FieldID> &field_vec,
                                     LegionFileMode mode, bool check_privileges);
      inline const RegionRequirement& get_requirement(void) const 
        { return requirement; }
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual unsigned find_parent_index(unsigned idx);
      virtual void trigger_commit(void);
      virtual void record_reference_mutation_effect(RtEvent event);
    public:
      PhysicalInstance create_instance(const Domain &dom,
        const std::vector<size_t> &field_sizes, LayoutConstraintSet &cons);
    protected:
      void check_privilege(void);
      void compute_parent_index(void);
    public:
      RegionRequirement requirement;
      RegionTreePath privilege_path;
      VersionInfo version_info;
      RestrictInfo restrict_info;
      const char *file_name;
      std::map<FieldID,const char*> field_map;
      LegionFileMode file_mode;
      ExternalType file_type;
      PhysicalRegion region;
      unsigned parent_req_index;
      std::set<RtEvent> map_applied_conditions;
    };

    /**
     * \class Detach Op
     * Operation for detaching a file from a physical instance
     */
    class DetachOp : public Operation {
    public:
      static const AllocationType alloc_type = DETACH_OP_ALLOC;
    public:
      DetachOp(Runtime *rt);
      DetachOp(const DetachOp &rhs);
      virtual ~DetachOp(void);
    public:
      DetachOp& operator=(const DetachOp &rhs);
    public:
      void initialize_detach(SingleTask *ctx, PhysicalRegion region);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual size_t get_region_count(void) const;
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_remote_state_analysis(RtUserEvent ready_event);
      virtual bool trigger_execution(void);
      virtual unsigned find_parent_index(unsigned idx);
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
    };

    /**
     * \class TimingOp
     * Operation for performing timing measurements
     */
    class TimingOp : public Operation {
    public:
      enum MeasurementKind {
        ABSOLUTE_MEASUREMENT,
        MICROSECOND_MEASUREMENT,
        NANOSECOND_MEASUREMENT,
      };
    public:
      TimingOp(Runtime *rt);
      TimingOp(const TimingOp &rhs);
      virtual ~TimingOp(void);
    public:
      TimingOp& operator=(const TimingOp &rhs);
    public:
      Future initialize(SingleTask *ctx, const Future &pre);
      Future initialize_microseconds(SingleTask *ctx, const Future &pre);
      Future initialize_nanoseconds(SingleTask *ctx, const Future &pre);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
      virtual OpKind get_operation_kind(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void deferred_execute(void);
      virtual void trigger_complete(void);
    protected:
      MeasurementKind kind;
      Future precondition;
      Future result;
    };

  }; //namespace Internal 
}; // namespace Legion 

#endif // __LEGION_OPERATIONS_H__
