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


#ifndef __LEGION_OPERATIONS_H__
#define __LEGION_OPERATIONS_H__

#include "legion.h"
#include "region_tree.h"
#include "legion_allocation.h"

namespace LegionRuntime {
  namespace HighLevel {

    // Special typedef for predicates
    typedef Predicate::Impl PredicateOp;

    /**
     * \class Operation
     * The operation class serves as the root of the tree
     * of all operations that can be performed in a Legion
     * program.
     */
    class Operation {
    public:
      struct DeferredMappingArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
        MustEpochOp *must_epoch;
        GenerationID must_epoch_gen;
      };
      struct DeferredCompleteArgs {
      public:
        HLRTaskID hlr_id;
        Operation *proxy_this;
      };
    public:
      Operation(Runtime *rt);
      virtual ~Operation(void);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(void) = 0; 
      virtual const char* get_logging_name(void) = 0;
    protected:
      // Base call
      void activate_operation(void);
      void deactivate_operation(void);
    public:
      inline GenerationID get_generation(void) const { return gen; }
      inline Event get_children_mapped(void) const { return children_mapped; }
      inline Event get_completion_event(void) const { return completion_event; }
      inline SingleTask* get_parent(void) const { return parent_ctx; }
      inline UniqueID get_unique_op_id(void) const { return unique_op_id; } 
      inline bool is_tracing(void) const { return tracing; }
      inline bool already_traced(void) const 
        { return ((trace != NULL) && !tracing); }
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
      void set_trace(LegionTrace *trace);
      void set_must_epoch(MustEpochOp *epoch, unsigned index);
    public:
      // Localize a region requirement to its parent context
      // This means that region == parent and the
      // coherence mode is exclusive
      static void localize_region_requirement(RegionRequirement &req);
    public:
      // Initialize this operation in a new parent context
      // along with the number of regions this task has
      void initialize_operation(SingleTask *ctx, bool track,
                                Event children_mapped,
                                unsigned num_regions = 0); 
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
      // The function to call for executing an operation
      // Note that this one is not invoked by the Operation class
      // but by the runtime, therefore any operations must be
      // placed on the ready queue in order for the runtime to
      // perform this mapping
      virtual bool trigger_execution(void);
      // The function to trigger once speculation is
      // ready to be resolved
      virtual void trigger_resolution(void);
      // Helper function for deferring complete operations
      // (only used in a limited set of operations and not
      // part of the default pipeline)
      virtual void deferred_complete(void);
      // The function to call once the operation is ready to complete
      virtual void trigger_complete(void);
      // The function to call when commit the operation is
      // ready to commit
      virtual void trigger_commit(void);
      // A helper method for deciding what to do when we have
      // aliased region requirements for an operation
      virtual void report_aliased_requirements(unsigned idx1, unsigned idx2);
    public:
      // The following are sets of calls that we can use to 
      // indicate mapping, execution, resolution, completion, and commit
      //
      // Indicate that we are done mapping this operation
      void complete_mapping(void); 
      // Indicate when this operation has finished executing
      void complete_execution(void);
      // Indicate when we have resolved the speculation for
      // this operation
      void resolve_speculation(void);
      // Indicate that we are completing this operation
      // which will also verify any regions for our producers
      void complete_operation(void);
      // Indicate that we are committing this operation
      void commit_operation(void);
      // Indicate that this operation is hardened against failure
      void harden_operation(void);
      // Quash this task and do what is necessary to the
      // rest of the operations in the graph
      void quash_operation(GenerationID gen, bool restart);
    public:
      // For operations that need to trigger commit early,
      // then they should use this call to avoid races
      // which could result in trigger commit being
      // called twice.  It will return true if the
      // caller is allowed to call trigger commit.
      bool request_early_commit(void);
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
      // A more general case of the one above that gives information about
      // the two regions involved in the dependence and the dependence type.
      bool register_dependence(unsigned idx, Operation *target, 
                               GenerationID target_gen, unsigned target_idx,
                               DependenceType dtype);
      // This is a special case of register dependence that will
      // also mark that we can verify a region produced by an earlier
      // operation so that operation can commit earlier.
      // Return true if the operation has committed and can be pruned
      // out of the list of dependences.
      bool register_region_dependence(unsigned idx, Operation *target,
                              GenerationID target_gen, unsigned target_idx,
                              DependenceType dtype);
      // This method is invoked by one of the two above to perform
      // the registration.  Returns true if we have not yet commited
      // and should therefore be notified once the dependent operation
      // has committed or verified its regions.
      bool perform_registration(GenerationID our_gen, 
                                Operation *op, GenerationID op_gen,
                                bool &registered_dependence,
                                unsigned &op_mapping_deps,
                                unsigned &op_speculation_deps,
                                Event &children_mapped);
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
    public:
      // Notify when a mapping dependence is met (flows down edges)
      void notify_mapping_dependence(GenerationID gen);
      // Notify when a speculation dependence is met (flows down edges)
      void notify_speculation_dependence(GenerationID gen);
      // Notify when an operation has committed (flows up edges)
      void notify_commit_dependence(GenerationID gen);
      // Notify when a region from a dependent task has 
      // been verified (flows up edges)
      void notify_regions_verified(const std::set<unsigned> &regions,
                                   GenerationID gen);
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
      // Number of outstanding mapping dependences before triggering map
      unsigned outstanding_mapping_deps;
      // Number of outstanding speculation dependences 
      unsigned outstanding_speculation_deps;
      // Number of outstanding commit dependences before triggering commit
      unsigned outstanding_commit_deps;
      // Number of outstanding mapping references, once this goes to 
      // zero then the set of outgoing edges is fixed
      unsigned outstanding_mapping_references;
      // The set of unverified regions
      std::set<unsigned> unverified_regions;
      // For each of our regions, a map of operations to the regions
      // which we can verify for each operation
      std::map<Operation*,std::set<unsigned> > verify_regions;
      // Set of events from operations we depend that describe when
      // all of their children have mapped
      std::set<Event> dependent_children_mapped;
      // Whether this operation has mapped, once it has mapped then
      // the set of incoming dependences is fixed
      bool mapped;
      // Whether this task has executed or not
      bool executed;
      // Whether speculation for this operation has been resolved
      bool resolved;
      // Whether the physical instances for this region have been
      // hardened by copying them into reslient memories
      bool hardened;
      // Whether this operation has completed, cannot commit until
      // both completed is set, and outstanding mapping references
      // has been gone to zero.
      bool completed;
      // Some operations commit out of order and if they do then
      // commited is set to prevent any additional dependences from
      // begin registered.
      bool committed;
      // Track whether trigger mapped has been invoked
      bool trigger_mapping_invoked;
      // Track whether trigger resolution has been invoked
      bool trigger_resolution_invoked;
      // Track whether trigger complete has been invoked
      bool trigger_complete_invoked;
      // Track whether trigger_commit has already been invoked
      bool trigger_commit_invoked;
      // Indicate whether we are responsible for
      // triggering the completion event for this operation
      bool need_completion_trigger;
      // Are we tracking this operation in the parent's context
      bool track_parent;
      // The enclosing context for this operation
      SingleTask *parent_ctx;
      // The event for when any children this operation has are mapped
      Event children_mapped;
      // The completion event for this operation
      UserEvent completion_event;
      // The trace for this operation if any
      LegionTrace *trace;
      // Track whether we are tracing this operation
      bool tracing;
      // Our must epoch if we have one
      MustEpochOp *must_epoch;
      // Generation for out mapping epoch
      GenerationID must_epoch_gen;
      // The index in the must epoch
      unsigned must_epoch_index;
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
     * \class Predicate::Impl 
     * A predicate operation is an abstract class that
     * contains a method that allows other operations to
     * sample their values and see if they are resolved
     * or whether they are speculated values.
     */
    class Predicate::Impl : public Operation {
    public:
      Impl(Runtime *rt);
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
      void initialize_speculation(SingleTask *ctx, bool track, 
                                  Event child_event,
                                  unsigned regions, const Predicate &p);
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
      virtual void deferred_complete(void);
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
      UserEvent predicate_waiter; // used only when needed
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
    class MapOp : public Inline, public Operation {
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
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
    public:
      virtual MappableKind get_mappable_kind(void) const;
      virtual Task* as_mappable_task(void) const;
      virtual Copy* as_mappable_copy(void) const;
      virtual Inline* as_mappable_inline(void) const;
      virtual Acquire* as_mappable_acquire(void) const;
      virtual Release* as_mappable_release(void) const;
      virtual UniqueID get_unique_mappable_id(void) const;
    protected:
      void check_privilege(void);
    protected:
      bool remap_region;
      UserEvent termination_event;
      PhysicalRegion region;
      RegionTreePath privilege_path;
      RegionTreePath mapping_path;
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
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void deferred_complete(void);
      virtual void report_aliased_requirements(unsigned idx1, unsigned idx2);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
    public:
      virtual MappableKind get_mappable_kind(void) const;
      virtual Task* as_mappable_task(void) const;
      virtual Copy* as_mappable_copy(void) const;
      virtual Inline* as_mappable_inline(void) const;
      virtual Acquire* as_mappable_acquire(void) const;
      virtual Release* as_mappable_release(void) const;
      virtual UniqueID get_unique_mappable_id(void) const;
    protected:
      void check_copy_privilege(const RegionRequirement &req, 
                                unsigned idx, bool src);
    public:
      std::vector<RegionTreePath> src_privilege_paths;
      std::vector<RegionTreePath> dst_privilege_paths;
      std::vector<RegionTreePath> src_mapping_paths; 
      std::vector<RegionTreePath> dst_mapping_paths;
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
      static const AllocationType alloc_type = FENCE_OP_ALLOC;
    public:
      FenceOp(Runtime *rt);
      FenceOp(const FenceOp &rhs);
      virtual ~FenceOp(void);
    public:
      FenceOp& operator=(const FenceOp &rhs);
    public:
      void initialize(SingleTask *ctx, bool mapping);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void deferred_complete(void);
    protected:
      bool mapping_fence;
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
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_commit(void);
    protected:
      DeletionKind kind;
      IndexSpace index_space;
      IndexPartition index_part;
      FieldSpace field_space;
      LogicalRegion logical_region;
      LogicalPartition logical_part;
      std::set<FieldID> free_fields;
    }; 

    /**
     * \class CloseOp
     * Close operations are only visible internally inside
     * the runtime and are issued to help close up the 
     * physical region tree states to an existing physical
     * instance that a task context initially mapped.
     */
    class CloseOp : public Operation {
    public:
      static const AllocationType alloc_type = CLOSE_OP_ALLOC;
    public:
      CloseOp(Runtime *rt);
      CloseOp(const CloseOp &rhs);
      virtual ~CloseOp(void);
    public:
      CloseOp& operator=(const CloseOp &rhs);
    public:
      void initialize(SingleTask *ctx, unsigned index, 
                      const InstanceRef &reference);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void deferred_complete(void);
    protected:
      RegionRequirement requirement;
      InstanceRef reference;
      RegionTreePath privilege_path;
#ifdef DEBUG_HIGH_LEVEL
      unsigned parent_index;
#endif
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
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual void deferred_complete(void);
    public:
      virtual MappableKind get_mappable_kind(void) const;
      virtual Task* as_mappable_task(void) const;
      virtual Copy* as_mappable_copy(void) const;
      virtual Inline* as_mappable_inline(void) const;
      virtual Acquire* as_mappable_acquire(void) const;
      virtual Release* as_mappable_release(void) const;
      virtual UniqueID get_unique_mappable_id(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    protected:
      void check_acquire_privilege(void);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
#ifdef DEBUG_HIGH_LEVEL
      RegionTreePath    mapping_path;
#endif
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
    public:
      virtual void trigger_dependence_analysis(void);
      virtual bool trigger_execution(void);
      virtual void resolve_true(void);
      virtual void resolve_false(void);
      virtual bool speculate(bool &value);
      virtual void deferred_complete(void);
    public:
      virtual MappableKind get_mappable_kind(void) const;
      virtual Task* as_mappable_task(void) const;
      virtual Copy* as_mappable_copy(void) const;
      virtual Inline* as_mappable_inline(void) const;
      virtual Acquire* as_mappable_acquire(void) const;
      virtual Release* as_mappable_release(void) const;
      virtual UniqueID get_unique_mappable_id(void) const;
    public:
      const RegionRequirement& get_requirement(void) const;
    protected:
      void check_release_privilege(void);
    protected:
      RegionRequirement requirement;
      RegionTreePath    privilege_path;
#ifdef DEBUG_HIGH_LEVEL
      RegionTreePath    mapping_path;
#endif
    };

    /**
     * \class FuturePredOp
     * A class for making predicates out of futures.
     */
    class FuturePredOp : public Predicate::Impl {
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
    class NotPredOp : public Predicate::Impl, PredicateWaiter {
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
    class AndPredOp : public Predicate::Impl, PredicateWaiter {
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
    class OrPredOp : public Predicate::Impl, PredicateWaiter {
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
        DependenceRecord(unsigned op1, unsigned op2,
                         unsigned reg1, unsigned reg2,
                         DependenceType d)
          : op1_idx(op1), op2_idx(op2), 
            reg1_idx(reg1), reg2_idx(reg2),
            dtype(d) { }
      public:
        unsigned op1_idx;
        unsigned op2_idx;
        unsigned reg1_idx;
        unsigned reg2_idx;
        DependenceType dtype;
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
      void set_task_options(ProcessorManager *manager);
      void find_conflicted_regions(std::vector<PhysicalRegion> &unmapped); 
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void);
    public:
      virtual void trigger_dependence_analysis(void);
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
    public:
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
      // Needs to be a set to ensure deduplication
      std::set<SingleTask*>        single_tasks;
    protected:
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
    protected:
      std::vector<DependenceRecord> dependences;
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
      bool map_tasks(const std::set<SingleTask*> &single_tasks);
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

  }; //namespace HighLevel
}; // namespace LegionRuntime

#endif // __LEGION_OPERATIONS_H__
