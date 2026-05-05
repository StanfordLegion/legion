/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_OPERATION_H__
#define __LEGION_OPERATION_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/api/launchers.h"
#include "legion/tools/profiler.h"
#include "legion/utilities/fieldmask_map.h"
#include "legion/utilities/hasher.h"

namespace Legion {
  namespace Internal {

    // clang-format off
#define LEGION_OPERATION_KINDS(__op__)                                                  \
  __op__(MAP_OP_KIND, "Inline Mapping")                                                 \
  __op__(COPY_OP_KIND, "Copy")                                                          \
  __op__(FENCE_OP_KIND, "Fence")                                                        \
  __op__(FRAME_OP_KIND, "Frame")                                                        \
  __op__(CREATION_OP_KIND, "Creation")                                                  \
  __op__(DELETION_OP_KIND, "Deletion")                                                  \
  __op__(MERGE_CLOSE_OP_KIND, "Merge Close")                                            \
  __op__(POST_CLOSE_OP_KIND, "Post Close")                                              \
  __op__(REFINEMENT_OP_KIND, "Refinement")                                              \
  __op__(RESET_OP_KIND, "Reset")                                                        \
  __op__(ACQUIRE_OP_KIND, "Acquire")                                                    \
  __op__(RELEASE_OP_KIND, "Release")                                                    \
  __op__(DYNAMIC_COLLECTIVE_OP_KIND, "Dynamic Collective")                              \
  __op__(FUTURE_PRED_OP_KIND, "Future Predicate")                                       \
  __op__(NOT_PRED_OP_KIND, "Not Predicate")                                             \
  __op__(AND_PRED_OP_KIND, "And Predicate")                                             \
  __op__(OR_PRED_OP_KIND, "Or Predicate")                                               \
  __op__(MUST_EPOCH_OP_KIND, "Must Epoch")                                              \
  __op__(PENDING_PARTITION_OP_KIND, "Pending Partition")                                \
  __op__(DEPENDENT_PARTITION_OP_KIND, "Dependent Partition")                            \
  __op__(FILL_OP_KIND, "Fill")                                                          \
  __op__(DISCARD_OP_KIND, "Discard")                                                    \
  __op__(ATTACH_OP_KIND, "Attach")                                                      \
  __op__(DETACH_OP_KIND, "Detach")                                                      \
  __op__(TIMING_OP_KIND, "Timing")                                                      \
  __op__(TUNABLE_OP_KIND, "Tunable")                                                    \
  __op__(ALL_REDUCE_OP_KIND, "All-Reduce")                                              \
  __op__(TRACE_BEGIN_OP_KIND, "Trace Begin")                                            \
  __op__(TRACE_RECURRENT_OP_KIND, "Trace Recurrent")                                    \
  __op__(TRACE_COMPLETE_OP_KIND, "Trace Complete")                                      \
  __op__(TASK_OP_KIND, "Task")                                                          \
  __op__(LAST_OP_KIND, "Last")
    // clang-format on
    enum OpKind {
#define LEGION_OPERATION_ENUM(kind, name) kind,
      LEGION_OPERATION_KINDS(LEGION_OPERATION_ENUM)
#undef LEGION_OPERATION_ENUM
    };

    /**
     * \class Operation
     * The operation class serves as the root of the tree
     * of all operations that can be performed in a Legion
     * program.
     */
    class Operation : public ProfilingResponseHandler {
    public:
      static const char* const op_names[];
    public:
      struct TriggerOpArgs : public LgTaskArgs<TriggerOpArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_TRIGGER_OP_ID;
      public:
        TriggerOpArgs(void) = default;
        TriggerOpArgs(Operation* o)
          : LgTaskArgs<TriggerOpArgs>(false, false), op(o)
        { }
        inline void execute(void) const
        {
          implicit_operation = op;
          op->trigger_mapping();
        }
      public:
        Operation* op;
      };
      struct DeferReleaseAcquiredArgs
        : public LgTaskArgs<DeferReleaseAcquiredArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_RELEASE_ACQUIRED_TASK_ID;
      public:
        DeferReleaseAcquiredArgs(void) = default;
        DeferReleaseAcquiredArgs(
            Operation* op,
            std::vector<std::pair<PhysicalManager*, unsigned> >* insts)
          : LgTaskArgs<DeferReleaseAcquiredArgs>(false, false), instances(insts)
        { }
        void execute(void) const;
      public:
        std::vector<std::pair<PhysicalManager*, unsigned> >* instances;
      };
    public:
      struct OpProfilingResponse : public ProfilingResponseBase {
      public:
        template<typename OP>
        OpProfilingResponse(
            OP* op, unsigned s, unsigned d, bool f, bool t = false)
          : ProfilingResponseBase(op, op->get_unique_op_id()), src(s), dst(d),
            fill(f), task(t)
        { }
      public:
        unsigned src, dst;
        bool fill;
        bool task;
      };
    public:
      Operation(void);
      virtual ~Operation(void);
    public:
      static const char* get_string_rep(OpKind kind);
    public:
      virtual void activate(void) = 0;
      virtual void deactivate(bool free = true) = 0;
      virtual const char* get_logging_name(void) const = 0;
      virtual OpKind get_operation_kind(void) const = 0;
      virtual size_t get_region_count(void) const;
      virtual Mappable* get_mappable(void);
      virtual MemoizableOp* get_memoizable(void) { return nullptr; }
      virtual bool invalidates_physical_trace_template(bool& exec_fence) const
      {
        exec_fence = false;
        return true;
      }
      virtual Operation* get_origin_operation(void) { return this; }
      virtual unsigned get_output_offset() const;
      virtual const RegionRequirement& get_requirement(unsigned idx) const
      {
        std::abort();
      }
      virtual void analyze_region_requirements(
          IndexSpaceNode* launch_space = nullptr,
          ShardingFunction* func = nullptr,
          IndexSpace shard_space = IndexSpace::NO_SPACE);
    public:
      inline GenerationID get_generation(void) const { return gen; }
      RtEvent get_mapped_event(void);
      RtEvent get_commit_event(void);
      // Overload the above to check to see if we're still the right generation
      RtEvent get_commit_event(GenerationID gen);
      inline ApEvent get_execution_fence_event(void) const
      {
        return execution_fence_event;
      }
      inline bool has_execution_fence_event(void) const
      {
        return execution_fence_event.exists();
      }
      inline void set_execution_fence_event(ApEvent fence_event)
      {
        execution_fence_event = fence_event;
      }
      inline InnerContext* get_context(void) const { return parent_ctx; }
      inline UniqueID get_unique_op_id(void) const { return unique_op_id; }
      inline bool is_tracing(void) const { return tracing; }
      inline LogicalTrace* get_trace(void) const { return trace; }
      inline MustEpochOp* get_must_epoch_op(void) const { return must_epoch; }
      inline Provenance* get_provenance(void) const { return provenance; }
    public:
      uint64_t get_context_index(void) const;
      std::optional<uint64_t> get_context_index(GenerationID gen) const;
      void set_context_index(uint64_t index, ExceptionHandlerID handler);
      ExceptionHandlerID get_exception_handler(void);
    public:
      // Be careful using this call as it is only valid when the operation
      // actually has a parent task.  Right now the only place it is used
      // is in putting the operation in the right dependence queue which
      // we know happens on the home node and therefore the operations is
      // guaranteed to have a parent task.
      unsigned get_operation_depth(void) const;
    public:
      void set_trace(LogicalTrace* trace, bool recording, uint64_t opidx);
      void set_must_epoch(MustEpochOp* epoch, bool do_registration);
    public:
      // Localize a region requirement to its parent context
      // This means that region == parent and the
      // coherence mode is exclusive
      static void localize_region_requirement(RegionRequirement& req);
      // We want to release our valid references for mapping as soon as
      // possible after mapping is done so the garbage collector can do
      // deferred collection ASAP if it needs to. However, there is a catch:
      // instances which are empty have no GC references from the physical
      // analysis to protect them from collection. That's not a problem for
      // the GC, but it is for keeping their meta-data structures alive.
      // Our solution is just to keep the valid references on the emtpy
      // acquired instances until the very end of the operation as they
      // will not hurt anything.
      RtEvent release_nonempty_acquired_instances(
          RtEvent precondition,
          std::map<PhysicalManager*, unsigned>& acquired_insts);
      static void release_acquired_instances(
          std::map<PhysicalManager*, unsigned>& acquired_insts);
    public:
      // Initialize this operation in a new parent context
      // along with the number of regions this task has
      void initialize_operation(
          InnerContext* ctx, Provenance* provenance = nullptr);
      void set_provenance(Provenance* provenance, bool has_ref);
    public:
      RtEvent execute_prepipeline_stage(
          GenerationID gen, bool from_logical_analysis);
      void execute_dependence_analysis(void);
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
      // The function to call once the operation is ready to complete
      virtual void trigger_complete(ApEvent effects_done);
      // The function to call when commit the operation is
      // ready to commit
      virtual void trigger_commit(void);
      // A helper method for deciding what to do when we have
      // aliased region requirements for an operation
      virtual void report_interfering_requirements(
          unsigned idx1, unsigned idx2);
      // A method for finding the parent index of a region
      // requirement for an operation which is necessary for
      // issuing close operation on behalf of the operation.
      virtual unsigned find_parent_index(unsigned idx);
      // A sentinel value for the context to use when we're tracing so
      // we know we need to recompute the value of the parent index
      static constexpr unsigned TRACED_PARENT_INDEX =
          std::numeric_limits<unsigned>::max();
      // Determine if this operation is an internal operation
      virtual bool is_internal_op(void) const { return false; }
      // Determine if this operation is a partition operation
      virtual bool is_partition_op(void) const { return false; }
      // Determine if this is a predicated operation
      virtual bool is_predicated_op(void) const { return false; }
      // Determine if this operation is a tracing fence
      virtual bool is_tracing_fence(void) const { return false; }
      // Record the trace hash for this operation
      virtual bool record_trace_hash(TraceHashRecorder& recorder, uint64_t idx);
      static void hash_requirement(
          Murmur3Hasher& hasher, const RegionRequirement& req);
    public:  // virtual methods for mapping
      // Pick the sources for a copy operations
      virtual void select_sources(
          const unsigned index, PhysicalManager* target,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& points);
    public:
      // Methods for help in performing collective analysis/view creation
      virtual size_t get_collective_points(void) const;
      virtual bool perform_collective_analysis(
          CollectiveMapping*& mapping, bool& first_local);
      virtual bool find_shard_participants(std::vector<ShardID>& shards);
      virtual RtEvent convert_collective_views(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, const InstanceSet& targets,
          InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
          bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals);
      virtual RtEvent perform_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          const FieldMask& mask, unsigned parent_req_index);
    public:
      void verify_requirement(
          const RegionRequirement& req, unsigned index = 0,
          bool allow_projections = false) const;
      virtual void report_uninitialized_usage(
          const unsigned index, const char* field_string, RtUserEvent reported);
      // Get a reference to our data structure for tracking acquired instances
      virtual std::map<PhysicalManager*, unsigned>* get_acquired_instances_ref(
          void);
      // Update the set of atomic locks for this operation
      virtual void update_atomic_locks(
          const unsigned index, Reservation lock, bool exclusive);
      // Get the restrict precondition for this operation
      static ApEvent merge_sync_preconditions(
          const TraceInfo& info, const std::vector<Grant>& grants,
          const std::vector<PhaseBarrier>& wait_barriers);
      virtual int add_copy_profiling_request(
          const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
          bool fill, unsigned count = 1);
      // Report a profiling result for this operation
      virtual bool handle_profiling_response(
          const Realm::ProfilingResponse& response, const void* orig,
          size_t orig_length, LgEvent& fevent, bool& failed_alloc);
      virtual void handle_profiling_update(int count);
      // To notify
      ApEvent get_completion_event(void);
      // Record an application event that needs to trigger before this
      // operation can be considered completed
      virtual void record_completion_effect(ApEvent effect);
      virtual void record_completion_effect(
          ApEvent effect, std::set<RtEvent>& map_applied_events);
      virtual void record_completion_effects(const std::set<ApEvent>& effects);
      virtual void record_completion_effects(
          const std::vector<ApEvent>& effects);
      // Allow for forwarding completion effects in a very special case
      void forward_completion_effects(Operation* target);
    public:
      // Point-Wise analysis functions
      virtual bool is_pointwise_analyzable(void) const;
      virtual void register_pointwise_dependence(
          unsigned idx, const LogicalUser& previous);
      virtual void replay_pointwise_dependences(
          std::map<unsigned, std::vector<PointwiseDependence> >& dependences);
      virtual RtEvent find_pointwise_dependence(
          const DomainPoint& point, GenerationID gen, bool intra_space,
          RtUserEvent to_trigger = RtUserEvent::NO_RT_USER_EVENT,
          std::optional<unsigned> output_region = std::optional<unsigned>());
    protected:
      void filter_copy_request_kinds(
          MapperManager* mapper,
          const std::set<ProfilingMeasurementID>& requests,
          std::vector<ProfilingMeasurementID>& results, bool warn_if_not_copy);
    public:
      // The following are sets of calls that we can use to
      // indicate mapping, execution, resolution, completion, and commit
      //
      // Add this to the list of ready operations
      void enqueue_ready_operation(
          RtEvent wait_on = RtEvent::NO_RT_EVENT,
          LgPriority priority = LG_THROUGHPUT_WORK_PRIORITY);
      // Indicate that we are done mapping this operation
      void complete_mapping(RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate when this operation has finished executing
      void complete_execution(RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Indicate that we are completing this operation
      // which will also verify any regions for our producers
      // You should probably never set first_invocation yourself
      void complete_operation(
          ApEvent effects = ApEvent::NO_AP_EVENT, bool first_invocation = true);
      // Indicate that we are committing this operation
      void commit_operation(
          bool do_deactivate, RtEvent wait_on = RtEvent::NO_RT_EVENT);
      // Quash this task and do what is necessary to the
      // rest of the operations in the graph
      void quash_operation(GenerationID gen, bool restart);
      // Helper method for triggering execution
      ApEvent compute_effects(void);
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
      bool register_dependence(Operation* target, GenerationID target_gen);
      // This function call does everything that the previous one does, but
      // it also records information about the regions involved and how
      // whether or not they will be validated by the consuming operation.
      // Return true if the operation has committed and can be pruned
      // out of the list of dependences.
      bool register_region_dependence(
          unsigned idx, Operation* target, GenerationID target_gen,
          unsigned target_idx, DependenceType dtype,
          const FieldMask& dependent_mask);
      // This method is invoked by one of the two above to perform
      // the registration.  Returns true if we have not yet commited
      // and should therefore be notified once the dependent operation
      // has committed or verified its regions.
      bool perform_registration(
          GenerationID our_gen, Operation* op, GenerationID op_gen,
          bool& registered_dependence,
          std::atomic<unsigned>& mapping_dependences,
          std::atomic<unsigned>& commit_dependences,
          std::set<Operation*>& notifications);
      // Notify another operation that a mapping dependence is satisfied
      void satisfy_mapping_dependence(void);
      // Notify another operation that a commit dependence is satisfied
      void satisfy_commit_dependence(void);
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
      // Notify when a region from a dependent task has
      // been verified (flows up edges)
      void notify_hardened(void);
    public:
      // Help for finding the contexts for an operation
      InnerContext* find_physical_context(unsigned index);
    public:
      // Support for operations that compute futures
      void compute_task_tree_coordinates(TaskTreeCoordinates& coordinates);
      virtual ContextCoordinate get_task_tree_coordinate(void) const;
    public:  // Support for mapping operations
      static void prepare_for_mapping(
          PhysicalManager* manager, MappingInstance& instance);
      static void prepare_for_mapping(
          const std::vector<InstanceView*>& views,
          std::vector<MappingInstance>& input_valid,
          std::vector<MappingCollective>& collective_valid);
      static void prepare_for_mapping(
          const InstanceSet& valid,
          const local::FieldMaskMap<ReplicatedView>& collectives,
          std::vector<MappingInstance>& input_valid,
          std::vector<MappingCollective>& collective_valid);
      static void prepare_for_mapping(
          const InstanceSet& valid,
          const local::FieldMaskMap<ReplicatedView>& collectives,
          const std::set<Memory>& filter_memories,
          std::vector<MappingInstance>& input_valid,
          std::vector<MappingCollective>& collective_valid);
      void compute_ranking(
          MapperManager* mapper, const std::deque<MappingInstance>& output,
          const std::vector<InstanceView*>& sources,
          std::vector<unsigned>& ranking,
          std::map<unsigned, PhysicalManager*>& collective_insts) const;
      void log_mapping_decision(
          unsigned index, const RegionRequirement& req,
          const InstanceSet& targets, bool postmapping = false) const;
      void log_launch_space(IndexSpace handle) const;
      void log_virtual_mapping(
          unsigned index, const RegionRequirement& req) const;
    public:
      // Pack the needed parts of this operation for a remote operation
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const;
      void pack_local_remote_operation(Serializer& rez) const;
    protected:
      static void add_launch_space_reference(IndexSpaceNode* node);
      static bool remove_launch_space_reference(IndexSpaceNode* node);
    protected:
      void perform_dependence_analysis(
          unsigned idx, const RegionRequirement& req,
          const ProjectionInfo& projection_info,
          LogicalAnalysis& logical_analysis);
      void perform_versioning_analysis(
          unsigned idx, const RegionRequirement& req, VersionInfo& version_info,
          std::set<RtEvent>& ready_events,
          RtEvent* output_region_ready = nullptr,
          bool collective_rendezvous = false);
      void physical_premap_region(
          unsigned index, RegionRequirement& req,
          const VersionInfo& version_info, InstanceSet& valid_instances,
          local::FieldMaskMap<ReplicatedView>& collectives,
          std::set<RtEvent>& map_applied_events);
      void physical_convert_sources(
          const RegionRequirement& req,
          const std::vector<MappingInstance>& sources,
          std::vector<PhysicalManager*>& result,
          std::map<PhysicalManager*, unsigned>* acquired);
      int physical_convert_mapping(
          const RegionRequirement& req, std::vector<MappingInstance>& chosen,
          InstanceSet& result, RegionTreeID& bad_tree,
          std::vector<FieldID>& missing_fields,
          std::map<PhysicalManager*, unsigned>* acquired,
          std::vector<PhysicalManager*>& unacquired,
          const bool do_acquire_checks,
          const bool allow_partial_virtual = false);
      bool physical_convert_postmapping(
          const RegionRequirement& req, std::vector<MappingInstance>& chosen,
          InstanceSet& result, RegionTreeID& bad_tree,
          std::map<PhysicalManager*, unsigned>* acquired,
          std::vector<PhysicalManager*>& unacquired,
          const bool do_acquire_checks);
      void perform_missing_acquires(
          std::map<PhysicalManager*, unsigned>& acquired,
          const std::vector<PhysicalManager*>& unacquired);
      // Return a runtime event for when it's safe to perform
      // the registration for this equivalence set
      RtEvent physical_perform_updates(
          const RegionRequirement& req, const VersionInfo& version_info,
          unsigned index, ApEvent precondition, ApEvent term_event,
          const InstanceSet& targets,
          const std::vector<PhysicalManager*>& sources,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events, UpdateAnalysis*& analysis,
          const bool collective_rendezvous, const bool record_valid = true,
          const bool check_initialized = true, const bool defer_copies = true);
      // Return an event for when the copy-out effects of the
      // registration are done (e.g. for restricted coherence)
      ApEvent physical_perform_registration(
          RtEvent precondition, UpdateAnalysis* analysis,
          std::set<RtEvent>& map_applied_events, bool symbolic = false);
      // Same as the two above merged together
      ApEvent physical_perform_updates_and_registration(
          const RegionRequirement& req, const VersionInfo& version_info,
          unsigned index, ApEvent precondition, ApEvent term_event,
          const InstanceSet& targets,
          const std::vector<PhysicalManager*>& sources,
          const PhysicalTraceInfo& trace_info,
          std::set<RtEvent>& map_applied_events,
          const bool collective_rendezvous, const bool record_valid = true,
          const bool check_initialized = true);
    protected:
      mutable LocalLock op_lock;
      std::atomic<GenerationID> gen;
      UniqueID unique_op_id;
      // The issue index of this operation in the context
      uint64_t context_index;
      // The exception handler for this operation if it exists
      ExceptionHandlerID exception_handler;
    protected:
      // Operations on which this operation depends
      std::map<Operation*, GenerationID> incoming;
      // Operations which depend on this operation
      std::map<Operation*, GenerationID> outgoing;
      // Mapping dependences
      std::atomic<unsigned> remaining_mapping_dependences;
      // Commit dependences
      std::atomic<unsigned> remaining_commit_dependences;
      // Number of outstanding mapping references, once this goes to
      // zero then the set of outgoing edges is fixed
      unsigned outstanding_mapping_references;
      // This count is only used in resilience mode
      // If we're a hardened operation, this counts how many of our
      // outgoing operations are complete and therefore will not be
      // raising any kind of region exception on us
      // If we're not hardened then count how many hardened notifications
      // we've seen from our outgoing operations so we know when we'll
      // never be asked to rollback
      unsigned hardened_notifications;
      // This data structure is only used in resilience mode
      // If we have a dependence on a hardened operation then we record
      // all the hardened operations we verify so we can notify them
      // as soon as we are complete that we're not going to raise any
      // region exceptions on them
      std::set<Operation*> verification_notifications;
#ifdef LEGION_DEBUG
      // Whether this operation is active or not
      bool activated;
#endif
      // Whether this operation has executed its prepipeline stage yet
      uint8_t prepipelined;
      // Whether this operation has mapped, once it has mapped then
      // the set of incoming dependences is fixed
      bool mapped;
      // Whether this task has executed or not
      bool executed;
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
      // Whether we are tracking the parent context
      bool track_parent;
      // Track whether we are tracing this operation
      bool tracing;
      // The trace for this operation if any
      LogicalTrace* trace;
      // The id local to a trace
      size_t trace_local_id;
      // The enclosing context for this operation
      InnerContext* parent_ctx;
      // The prepipeline event for this operation
      RtUserEvent prepipelined_event;
      // The mapped event for this operation
      RtUserEvent mapped_event;
      // The commit event for this operation
      RtUserEvent commit_event;
      // Previous execution fence if there was one
      ApEvent execution_fence_event;
      // Our must epoch if we have one
      MustEpochOp* must_epoch;
    private:
      // Provenance information for this operation
      Provenance* provenance;
      // Track the completion events for this operation in case someone
      // decides that they are going to ask for it later
      std::set<ApEvent> completion_effects;
      // The completion event for this operation
      union CompletionEvent {
        CompletionEvent(void) : effects(ApEvent::NO_AP_EVENT) { }
        ApEvent effects;
        ApUserEvent pending;
      } completion_event;
      bool completion_set;
    };

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const Operation& op)
    //--------------------------------------------------------------------------
    {
      os << op.get_logging_name() << " (UID: " << op.get_unique_op_id() << ")";
      return os;
    }

    /**
     * \class ExternalMappable
     * This is class that provides some basic functionality for
     * packing and unpacking the data structures used by
     * external facing operations
     */
    class ExternalMappable {
    public:
      virtual void set_context_index(uint64_t index) = 0;
    public:
      static void pack_mappable(const Mappable& mappable, Serializer& rez);
      static void pack_index_space_requirement(
          const IndexSpaceRequirement& req, Serializer& rez);
      static void pack_region_requirement(
          const RegionRequirement& req, Serializer& rez);
      static void pack_grant(const Grant& grant, Serializer& rez);
      static void pack_phase_barrier(
          const PhaseBarrier& barrier, Serializer& rez);
    public:
      static void unpack_mappable(Mappable& mappable, Deserializer& derez);
      static void unpack_index_space_requirement(
          IndexSpaceRequirement& req, Deserializer& derez);
      static void unpack_region_requirement(
          RegionRequirement& req, Deserializer& derez);
      static void unpack_grant(Grant& grant, Deserializer& derez);
      static void unpack_phase_barrier(
          PhaseBarrier& barrier, Deserializer& derez);
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_OPERATION_H__
