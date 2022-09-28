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


#ifndef __LEGION_TRACE__
#define __LEGION_TRACE__

#include "legion.h"
#include "legion/legion_ops.h"
#include "legion/legion_analysis.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LegionTrace
     * This is the abstract base class for a trace object
     * and is used to support both static and dynamic traces
     */
    class LegionTrace : public Collectable {
    public:
      struct DependenceRecord {
      public:
        DependenceRecord(int idx)
          : operation_idx(idx), prev_idx(-1), next_idx(-1),
            validates(false), dtype(LEGION_TRUE_DEPENDENCE) { }
        DependenceRecord(int op_idx, int pidx, int nidx,
                         bool val, DependenceType d,
                         const FieldMask &m)
          : operation_idx(op_idx), prev_idx(pidx), 
            next_idx(nidx), validates(val), dtype(d),
            dependent_mask(m) { }
      public:
        inline bool merge(const DependenceRecord &record)
        {
          if ((operation_idx != record.operation_idx) ||
              (prev_idx != record.prev_idx) ||
              (next_idx != record.next_idx) ||
              (validates != record.validates) ||
              (dtype != record.dtype))
            return false;
          dependent_mask |= record.dependent_mask;
          return true;
        }
      public:
        int operation_idx;
        int prev_idx; // previous region requirement index
        int next_idx; // next region requirement index
        bool validates;
        DependenceType dtype;
        FieldMask dependent_mask;
      };
      struct AliasChildren {
      public:
        AliasChildren(unsigned req_idx, unsigned dep, const FieldMask &m)
          : req_index(req_idx), depth(dep), mask(m) { }
      public:
        unsigned req_index;
        unsigned depth;
        FieldMask mask;
      };
    public:
      enum TracingState {
        LOGICAL_ONLY,
        PHYSICAL_RECORD,
        PHYSICAL_REPLAY,
      };
    public:
      LegionTrace(InnerContext *ctx, TraceID tid, bool logical_only, 
                  Provenance *provenance);
      virtual ~LegionTrace(void);
    public:
      virtual bool is_static_trace(void) const = 0;
      inline TraceID get_trace_id(void) const { return tid; }
    public:
      virtual bool handles_region_tree(RegionTreeID tid) const = 0;
      virtual bool initialize_op_tracing(Operation *op,
                     const std::vector<StaticDependence> *dependences,
                     const LogicalTraceInfo *trace_info) = 0;
      virtual void register_operation(Operation *op, GenerationID gen) = 0; 
      virtual void record_dependence(Operation *target, GenerationID target_gen,
                                Operation *source, GenerationID source_gen) = 0;
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask) = 0;
      virtual void record_no_dependence(Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    const FieldMask &dependent_mask) = 0;
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask) = 0;
      virtual void end_trace_capture(void) = 0;
    public:
      // Called by task execution thread
      inline bool is_fixed(void) const { return fixed; }
      void fix_trace(Provenance *provenance);
    public:
      bool has_physical_trace(void) { return physical_trace != NULL; }
      PhysicalTrace* get_physical_trace(void) { return physical_trace; }
      void register_physical_only(Operation *op);
    public:
      void replay_aliased_children(std::vector<RegionTreePath> &paths) const;
      void end_trace_execution(FenceOp *fence_op);
    public:
      void initialize_tracing_state(void) { state = LOGICAL_ONLY; }
      void set_state_record(void) { state.store(PHYSICAL_RECORD); }
      void set_state_replay(void) { state.store(PHYSICAL_REPLAY); }
      bool is_recording(void) const { return state.load() == PHYSICAL_RECORD; }
      bool is_replaying(void) const { return state.load() == PHYSICAL_REPLAY; }
    public:
      inline void clear_blocking_call(void) { blocking_call_observed = false; }
      inline void record_blocking_call(void) { blocking_call_observed = true; }
      inline bool has_blocking_call(void) const {return blocking_call_observed;}
      inline bool has_intermediate_operations(void) const 
        { return has_intermediate_ops; }
      inline void reset_intermediate_operations(void)
        { has_intermediate_ops = false; }
      void invalidate_trace_cache(Operation *invalidator);
#ifdef LEGION_SPY
    public:
      virtual void perform_logging(
                          UniqueID prev_fence_uid, UniqueID curr_fence_uid) = 0;
    public:
      UniqueID get_current_uid_by_index(unsigned op_idx) const;
#endif
    public:
      InnerContext *const ctx;
      const TraceID tid;
      Provenance *const begin_provenance;
      // Set after end_trace is called
      Provenance *end_provenance;
    protected:
      std::vector<std::pair<Operation*,GenerationID> > operations; 
      // We also need a data structure to record when there are
      // aliased but non-interfering region requirements. This should
      // be pretty sparse so we'll make it a map
      std::map<unsigned,LegionVector<AliasChildren> > aliased_children;
      std::atomic<TracingState> state;
      // Pointer to a physical trace
      PhysicalTrace *physical_trace;
      unsigned last_memoized;
      unsigned physical_op_count;
      bool blocking_call_observed;
      bool has_intermediate_ops;
      bool fixed;
      std::set<std::pair<Operation*,GenerationID> > frontiers;
#ifdef LEGION_SPY
    protected:
      std::map<std::pair<Operation*,GenerationID>,UniqueID> current_uids;
      std::map<std::pair<Operation*,GenerationID>,unsigned> num_regions;
#endif
    };

    /**
     * \class StaticTrace
     * A static trace is a trace object that is used for 
     * handling cases where the application knows the dependneces
     * for a trace of operations
     */
    class StaticTrace : public LegionTrace,
                        public LegionHeapify<StaticTrace> {
    public:
      static const AllocationType alloc_type = STATIC_TRACE_ALLOC;
    public:
      StaticTrace(TraceID tid, InnerContext *ctx, bool logical_only,
                  Provenance *p, const std::set<RegionTreeID> *trees);
      StaticTrace(const StaticTrace &rhs);
      virtual ~StaticTrace(void);
    public:
      StaticTrace& operator=(const StaticTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return true; }
    public:
      virtual bool handles_region_tree(RegionTreeID tid) const;
      virtual bool initialize_op_tracing(Operation *op,
                              const std::vector<StaticDependence> *dependences,
                              const LogicalTraceInfo *trace_info);
      virtual void register_operation(Operation *op, GenerationID gen); 
      virtual void record_dependence(Operation *target,GenerationID target_gen,
                                     Operation *source,GenerationID source_gen);
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      virtual void record_no_dependence(Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    const FieldMask &dependent_mask);
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
      virtual void end_trace_capture(void);
#ifdef LEGION_SPY
    public:
      virtual void perform_logging(
                          UniqueID prev_fence_uid, UniqueID curr_fence_uid);
#endif
    protected:
      const LegionVector<DependenceRecord>&
                  translate_dependence_records(Operation *op, unsigned index);
    protected:
      std::deque<std::vector<StaticDependence> > static_dependences;
      std::deque<LegionVector<DependenceRecord> > translated_deps;
      std::set<RegionTreeID> application_trees;
    };

    /**
     * \class DynamicTrace
     * This class is used for memoizing the dynamic
     * dependence analysis for series of operations
     * in a given task's context.
     */
    class DynamicTrace : public LegionTrace,
                         public LegionHeapify<DynamicTrace> {
    public:
      static const AllocationType alloc_type = DYNAMIC_TRACE_ALLOC;
    public:
      struct OperationInfo {
      public:
        OperationInfo(Operation *op)
          : kind(op->get_operation_kind()), count(op->get_region_count()) { }
      public:
        Operation::OpKind kind;
        unsigned count;
      }; 
    public:
      DynamicTrace(TraceID tid, InnerContext *ctx, 
                   bool logical_only, Provenance *p);
      DynamicTrace(const DynamicTrace &rhs);
      virtual ~DynamicTrace(void);
    public:
      DynamicTrace& operator=(const DynamicTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return false; }
    public:
      virtual bool initialize_op_tracing(Operation *op,
                          const std::vector<StaticDependence> *dependences,
                          const LogicalTraceInfo *trace_info);
      virtual bool handles_region_tree(RegionTreeID tid) const;
      // Called by analysis thread
      virtual void register_operation(Operation *op, GenerationID gen);
      virtual void record_dependence(Operation *target,GenerationID target_gen,
                                     Operation *source,GenerationID source_gen);
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      virtual void record_no_dependence(Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    const FieldMask &dependent_mask);
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
      // Called by analysis thread
      virtual void end_trace_capture(void);
#ifdef LEGION_SPY
    public:
      virtual void perform_logging(
                          UniqueID prev_fence_uid, UniqueID curr_fence_uid);
#endif
    protected:
      // Insert a normal dependence for the current operation
      void insert_dependence(const DependenceRecord &record);
      // Insert an internal dependence for given key
      void insert_dependence(const std::pair<InternalOp*,GenerationID> &key,
                             const DependenceRecord &record);
    protected:
      // Only need this backwards lookup for recording dependences
      std::map<std::pair<Operation*,GenerationID>,unsigned> op_map;
      // Internal operations have a nasty interaction with traces because
      // we can generate different sets of internal operations each time we
      // run the trace depending on the state of the logical region tree.
      // Therefore, we keep track of internal ops done when capturing the trace
      // record transitive dependences on the other operations in the
      // trace that we would have interfered with had the internal operations
      // not been necessary.
      std::map<std::pair<InternalOp*,GenerationID>,
               LegionVector<DependenceRecord> > internal_dependences;
    protected: 
      // This is the generalized form of the dependences
      // For each operation, we remember a list of operations that
      // it dependens on and whether it is a validates the region
      std::deque<LegionVector<DependenceRecord> > dependences;
      // Metadata for checking the validity of a trace when it is replayed
      std::vector<OperationInfo> op_info;
    protected:
      bool tracing;
    };

    class TraceOp : public FenceOp {
    public:
      TraceOp(Runtime *rt);
      TraceOp(const TraceOp &rhs);
      virtual ~TraceOp(void);
    public:
      TraceOp& operator=(const TraceOp &rhs);
    public:
      virtual void execute_dependence_analysis(void);
    protected:
      LegionTrace *local_trace;
    };

    /**
     * \class TraceCaptureOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when a trace capture
     * is finished so the DynamicTrace object can compute the
     * dependences data structure.
     */
    class TraceCaptureOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_CAPTURE_OP_ALLOC;
    public:
      TraceCaptureOp(Runtime *rt);
      TraceCaptureOp(const TraceCaptureOp &rhs);
      virtual ~TraceCaptureOp(void);
    public:
      TraceCaptureOp& operator=(const TraceCaptureOp &rhs);
    public:
      void initialize_capture(InnerContext *ctx, bool has_blocking_call,
                    bool remove_trace_reference, const char *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      PhysicalTemplate *current_template;
      bool has_blocking_call;
      bool remove_trace_reference;
      bool is_recording;
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      TraceCompleteOp(Runtime *rt);
      TraceCompleteOp(const TraceCompleteOp &rhs);
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs);
    public:
      void initialize_complete(InnerContext *ctx, bool has_blocking_call,
                               const char *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    protected:
      PhysicalTemplate *current_template;
      bool replayed;
      bool has_blocking_call;
      bool is_recording;
    };

    /**
     * \class TraceReplayOp
     * This class represents trace operations which we inject
     * into the operation stream to replay a physical trace
     * if there is one that satisfies its preconditions.
     */
    class TraceReplayOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_REPLAY_OP_ALLOC;
    public:
      TraceReplayOp(Runtime *rt);
      TraceReplayOp(const TraceReplayOp &rhs);
      virtual ~TraceReplayOp(void);
    public:
      TraceReplayOp& operator=(const TraceReplayOp &rhs);
    public:
      void initialize_replay(InnerContext *ctx, LegionTrace *trace,
                             Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    };

    /**
     * \class TraceBeginOp
     * This class represents mapping fences which we inject
     * into the operation stream to begin a trace.  This fence
     * is by a TraceReplayOp if the trace allows physical tracing.
     */
    class TraceBeginOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_BEGIN_OP_ALLOC;
    public:
      TraceBeginOp(Runtime *rt);
      TraceBeginOp(const TraceBeginOp &rhs);
      virtual ~TraceBeginOp(void);
    public:
      TraceBeginOp& operator=(const TraceBeginOp &rhs);
    public:
      void initialize_begin(InnerContext *ctx, LegionTrace *trace,
                            Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    };

    class TraceSummaryOp : public TraceOp {
    public:
      static const AllocationType alloc_type = TRACE_SUMMARY_OP_ALLOC;
    public:
      TraceSummaryOp(Runtime *rt);
      TraceSummaryOp(const TraceSummaryOp &rhs);
      virtual ~TraceSummaryOp(void);
    public:
      TraceSummaryOp& operator=(const TraceSummaryOp &rhs);
    public:
      void initialize_summary(InnerContext *ctx,
                              PhysicalTemplate *tpl,
                              Operation *invalidator,
                              Provenance *provenance);
      void perform_logging(void);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                         std::set<RtEvent> &applied) const;
    protected:
      PhysicalTemplate *current_template;
    };

    /**
     * \class PhysicalTrace
     * This class is used for memoizing the dynamic physical dependence
     * analysis for series of operations in a given task's context.
     */
    class PhysicalTrace {
    public:
      PhysicalTrace(Runtime *runtime, LegionTrace *logical_trace);
      PhysicalTrace(const PhysicalTrace &rhs) = delete;
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace &rhs) = delete;
    public:
      void clear_cached_template(void) { current_template = NULL; }
      void check_template_preconditions(TraceReplayOp *op,
                                        std::set<RtEvent> &applied_events);
    public:
      PhysicalTemplate* get_current_template(void) { return current_template; }
      bool has_any_templates(void) const { return !templates.empty(); }
    public:
      void record_previous_template_completion(ApEvent template_completion)
        { previous_template_completion = template_completion; }
      ApEvent get_previous_template_completion(void) const
        { return previous_template_completion; }
    public:
      void set_current_execution_fence_event(ApEvent event)
        { execution_fence_event = event; }
      ApEvent get_current_execution_fence_event(void) const
        { return execution_fence_event; }
    public:
      PhysicalTemplate* start_new_template(void);
      ApEvent fix_trace(PhysicalTemplate *tpl, Operation *op,
                std::set<RtEvent> &applied_events, bool has_blocking_call);
      void record_intermediate_execution_fence(FenceOp *fence);
      void chain_replays(FenceOp *replay_op);
    public:
      const std::vector<Processor> &get_replay_targets(void)
        { return replay_targets; }
    public:
      void initialize_template(ApEvent fence_completion, bool recurrent);
    public:
      Runtime * const runtime;
      const LegionTrace *logical_trace;
      const bool perform_fence_elision;
    private:
      mutable LocalLock trace_lock;
      FenceOp *previous_replay;
      UniqueID previous_replay_gen;
      PhysicalTemplate* current_template;
      std::vector<PhysicalTemplate*> templates;
      unsigned nonreplayable_count;
      unsigned new_template_count;
    private:
      ApEvent previous_template_completion;
      ApEvent execution_fence_event;
      std::vector<Processor> replay_targets;
    private:
      bool intermediate_execution_fence;
    };

    /**
     * \class TraceViewSet
     */
    class TraceViewSet {
    public:
      struct FailedPrecondition {
        InstanceView *view;
        EquivalenceSet *eq;
        FieldMask mask;

        std::string to_string(TaskContext *ctx) const;
      };
    public:
      TraceViewSet(RegionTreeForest *forest);
      virtual ~TraceViewSet(void);
    public:
      void insert(InstanceView *view,
                  EquivalenceSet *eq,
                  const FieldMask &mask);
      void invalidate(InstanceView *view,
                      EquivalenceSet *eq,
                      const FieldMask &mask);
    public:
      bool dominates(InstanceView *view, EquivalenceSet *eq,
                     FieldMask &non_dominated) const;
      bool subsumed_by(const TraceViewSet &set,
                       FailedPrecondition *condition = NULL) const;
      bool has_refinements(void) const;
      bool empty(void) const;
    public:
      void dump(void) const;
    protected:
      typedef LegionMap<InstanceView*,
                        FieldMaskSet<EquivalenceSet>  > ViewSet;
    protected:
      RegionTreeForest * const forest;
    protected:
      ViewSet conditions;
      // Need to hold view references if we're going to be dumping
      // this trace even past the end of execution
      const bool view_references;
    };

    /**
     * \class TraceConditionSet
     */
    class TraceConditionSet : public TraceViewSet {
    public:
      TraceConditionSet(RegionTreeForest *forest);
      virtual ~TraceConditionSet(void);
    public:
      void make_ready(bool postcondition);
    public:
      bool require(Operation *op, std::set<RtEvent> &applied_events);
      void ensure(Operation *op, std::set<RtEvent> &applied_events);
      bool modifies_fields(const LegionMap<InstanceView*,
              FieldMaskSet<IndexSpaceExpression> > &instances) const;
    private:
      bool cached;
      // The following containers are populated only when the 'cached' is true.
      LegionVector<FieldMaskSet<InstanceView> >          views;
      LegionVector<VersionInfo>                          version_infos;
      // Managers that we keep valid
      std::vector<PhysicalManager*>                      valid_instances; 
    };

    /**
     * \class PhysicalTemplate
     * This class represents a recipe to reconstruct a physical task graph.
     * A template consists of a sequence of instructions, each of which is
     * interpreted by the template engine. The template also maintains
     * the interpreter state (operations and events). These are initialized
     * before the template gets executed.
     */
    class PhysicalTemplate : public PhysicalTraceRecorder,
                             public LegionHeapify<PhysicalTemplate> {
    public:
      struct ReplaySliceArgs : public LgTaskArgs<ReplaySliceArgs> {
      public:
        static const LgTaskID TASK_ID = LG_REPLAY_SLICE_TASK_ID;
      public:
        ReplaySliceArgs(PhysicalTemplate *t, unsigned si, bool recurrent)
          : LgTaskArgs<ReplaySliceArgs>(implicit_provenance),
            tpl(t), slice_index(si), recurrent_replay(recurrent) { }
      public:
        PhysicalTemplate *const tpl;
        const unsigned slice_index;
        const bool recurrent_replay;
      };
      struct TransitiveReductionArgs :
        public LgTaskArgs<TransitiveReductionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRANSITIVE_REDUCTION_TASK_ID;
      public:
        TransitiveReductionArgs(PhysicalTemplate *t)
          : LgTaskArgs<TransitiveReductionArgs>(implicit_provenance), tpl(t) { }
      public:
        PhysicalTemplate *const tpl;
      };
      struct DeleteTemplateArgs : public LgTaskArgs<DeleteTemplateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DELETE_TEMPLATE_TASK_ID;
      public:
        DeleteTemplateArgs(PhysicalTemplate *t)
          : LgTaskArgs<DeleteTemplateArgs>(implicit_provenance), tpl(t) { }
      public:
        PhysicalTemplate *const tpl;
      };
    private:
      struct ViewUser {
        ViewUser(const RegionUsage &r, unsigned u, IndexSpaceExpression *e)
          : usage(r), user(u), expr(e)
        {}
        const RegionUsage usage;
        const unsigned user;
        IndexSpaceExpression *const expr;
      };
    private:
      struct CachedMapping
      {
        VariantID               chosen_variant;
        TaskPriority            task_priority;
        bool                    postmap_task;
        std::vector<Processor>  target_procs;
        std::deque<InstanceSet> physical_instances;
      };
      typedef LegionMap<TraceLocalID,CachedMapping> CachedMappings;
    private:
      typedef LegionMap<InstanceView*,
                        FieldMaskSet<IndexSpaceExpression> >         ViewExprs;
      typedef LegionMap<InstanceView*,
                        FieldMaskSet<ViewUser> >                     ViewUsers;
      typedef std::map<RegionTreeID,std::set<InstanceView*> >        ViewGroups;
    public:
      PhysicalTemplate(PhysicalTrace *trace, ApEvent fence_event);
      PhysicalTemplate(const PhysicalTemplate &rhs);
    protected:
      virtual ~PhysicalTemplate(void);
    public:
      void initialize_replay(ApEvent fence_completion, bool recurrent);
      ApEvent get_completion_for_deletion(void) const;
      void find_execution_fence_preconditions(std::set<ApEvent> &preconditions);
    public:
      void finalize(Operation *op, bool has_blocking_call);
      void generate_conditions(void);
    public:
      struct Replayable {
        explicit Replayable(bool r)
          : replayable(r), message()
        {}
        Replayable(bool r, const char *m)
          : replayable(r), message(m)
        {}
        Replayable(bool r, const std::string &m)
          : replayable(r), message(m)
        {}
        Replayable(const Replayable &r)
          : replayable(r.replayable), message(r.message)
        {}
        operator bool(void) const { return replayable; }
        bool replayable;
        std::string message;
      };
    private:
      Replayable check_replayable(bool has_blocking_call) const;
    public:
      void optimize(bool do_transitive_reduction);
    private:
      void elide_fences(std::vector<unsigned> &gen);
      void propagate_merges(std::vector<unsigned> &gen);
      void transitive_reduction(bool deferred);
      void finalize_transitive_reduction(
          const std::vector<unsigned> &inv_topo_order,
          const std::vector<std::vector<unsigned> > &incoming_reduced);
      void propagate_copies(std::vector<unsigned> *gen);
      void eliminate_dead_code(std::vector<unsigned> &gen);
      void prepare_parallel_replay(const std::vector<unsigned> &gen);
      void push_complete_replays(void);
    public:
      bool check_preconditions(TraceReplayOp *op,
                               std::set<RtEvent> &applied_events);
      void apply_postcondition(TraceSummaryOp *op,
                               std::set<RtEvent> &applied_events);
    public:
      void register_operation(Operation *op);
      void execute_slice(unsigned slice_idx, bool recurrent_replay);
    public:
      void issue_summary_operations(InnerContext* context,
                                    Operation *invalidator,
                                    Provenance *provenance);
    public:
      void dump_template(void);
    private:
      void dump_instructions(const std::vector<Instruction*> &instructions);
#ifdef LEGION_SPY
    public:
      void set_fence_uid(UniqueID fence_uid) { prev_fence_uid = fence_uid; }
      UniqueID get_fence_uid(void) const { return prev_fence_uid; }
#endif
    public:
      inline bool is_replaying(void) const { return !recording.load(); }
      inline bool is_replayable(void) const { return replayable.replayable; }
      inline const std::string& get_replayable_message(void) const
        { return replayable.message; }
    public:
      virtual bool is_recording(void) const { return recording.load(); }
      virtual void add_recorder_reference(void) { /*do nothing*/ }
      virtual bool remove_recorder_reference(void) 
        { /*do nothing, never delete*/ return false; }
      virtual void pack_recorder(Serializer &rez, 
          std::set<RtEvent> &applied, const AddressSpaceID target);
    public:
      virtual void record_mapper_output(const TraceLocalID &tlid,
                             const Mapper::MapTaskOutput &output,
                             const std::deque<InstanceSet> &physical_instances,
                             std::set<RtEvent> &applied_events);
      void get_mapper_output(SingleTask *task,
                             VariantID &chosen_variant,
                             TaskPriority &task_priority,
                             bool &postmap_task,
                             std::vector<Processor> &target_proc,
                             std::deque<InstanceSet> &physical_instances) const;
      void get_task_reservations(SingleTask *task,
                             std::map<Reservation,bool> &reservations) const;
    public:
      virtual void record_get_term_event(Memoizable *memo);
      virtual void record_create_ap_user_event(ApUserEvent lhs, 
                                               Memoizable *memo);
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        Memoizable *memo);
    public:
      virtual void record_merge_events(ApEvent &lhs, 
                                       ApEvent rhs, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, 
                                       ApEvent e2, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, 
                            const std::set<ApEvent>& rhs, Memoizable *memo);
      virtual void record_merge_events(ApEvent &lhs, 
                            const std::vector<ApEvent>& rhs, Memoizable *memo);
    public:
      virtual void record_issue_copy(Memoizable *memo, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                             RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                             ApEvent precondition, PredEvent pred_guard);
      virtual void record_issue_across(Memoizable *memo, ApEvent &lhs, 
                             ApEvent collective_precondition,
                             ApEvent copy_precondition,
                             ApEvent src_indirect_precondition,
                             ApEvent dst_indirect_precondition,
                             CopyAcrossExecutor *executor);
      virtual void record_copy_views(ApEvent lhs, Memoizable *memo,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied);
      virtual void record_indirect_views(ApEvent indirect_done,ApEvent all_done,
                           Memoizable *memo, unsigned index, 
                           IndexSpaceExpression *expr,
                           const FieldMaskSet<InstanceView> &tracing_views,
                           std::set<RtEvent> &applied, PrivilegeMode priv);
      virtual void record_issue_fill(Memoizable *memo, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField> &fields,
                             const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                             FieldSpace handle, 
                             RegionTreeID tree_id,
#endif
                             ApEvent precondition, PredEvent pred_guard);
    public:
      virtual void record_op_view(Memoizable *memo,
                                  unsigned idx,
                                  InstanceView *view,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  bool update_validity);
      virtual void record_post_fill_view(FillView *view, const FieldMask &mask);
      virtual void record_fill_views(ApEvent lhs, Memoizable *memo,
                           unsigned idx, IndexSpaceExpression *expr, 
                           const FieldMaskSet<FillView> &tracing_srcs,
                           const FieldMaskSet<InstanceView> &tracing_dsts,
                           std::set<RtEvent> &applied_events,
                           bool reduction_initialization);
    private:
      void record_views(unsigned entry,
                        IndexSpaceExpression *expr,
                        const RegionUsage &usage,
                        const FieldMaskSet<InstanceView> &views,
                        const LegionList<FieldSet<EquivalenceSet*> > &eqs);
      void update_valid_views(InstanceView *view,
                              EquivalenceSet *eq,
                              const RegionUsage &usage,
                              const FieldMask &user_mask,
                              bool invalidates);
      void add_view_user(InstanceView *view,
                         const RegionUsage &usage,
                         unsigned user, IndexSpaceExpression *user_expr,
                         const FieldMask &user_mask);
      static void record_expression_views(ViewExprs &cviews,
                             IndexSpaceExpression *expr,
                             const FieldMaskSet<InstanceView> &views);
      void record_fill_views(const FieldMaskSet<FillView> &views);
    public:
      virtual void record_set_op_sync_event(ApEvent &lhs, Memoizable *memo);
      virtual void record_set_effects(Memoizable *memo, ApEvent &rhs);
      virtual void record_complete_replay(Memoizable *memo, ApEvent rhs);
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events); 
    public:
      void perform_replay(Runtime *rt, std::set<RtEvent> &replayed_events);
      void finish_replay(std::set<ApEvent> &postconditions);
      bool defer_template_deletion(ApEvent &pending_deletion,
                                   std::set<RtEvent> &applied_events);
    public:
      static void handle_replay_slice(const void *args);
      static void handle_transitive_reduction(const void *args, Runtime *rt);
      static void handle_delete_template(const void *args);
    public:
      RtEvent get_recording_done(void) const
        { return recording_done; }
      void trigger_recording_done(void);
      virtual RtEvent get_collect_event(void) const { return recording_done; }
    private:
      TraceLocalID find_trace_local_id(Memoizable *memo);
      unsigned find_memo_entry(Memoizable *memo);
      TraceLocalID record_memo_entry(Memoizable *memo, unsigned entry);
    private:
      unsigned convert_event(const ApEvent &event);
      unsigned find_event(const ApEvent &event) const;
      void insert_instruction(Instruction *inst);
    private:
      // Returns the set of last users for all <view,field mask,index expr>
      // tuples in the view_exprs
      void find_all_last_users(ViewExprs &view_exprs,
                               std::set<unsigned> &users);
      // Returns the set of last users for a given <view,field mask,index expr>
      // tuple
      void find_last_users(InstanceView *view,
                           IndexSpaceExpression *expr,
                           const FieldMask &mask,
                           std::set<unsigned> &users);
      void rewrite_preconditions(unsigned &precondition,
                           std::set<unsigned> &users,
                           const std::vector<Instruction*> &instructions,
                           std::vector<Instruction*> &new_instructions,
                           std::vector<unsigned> &gen,
                           unsigned &merge_starts);
      void parallelize_replay_event(unsigned &event_to_check,
                           unsigned slice_index,
                           const std::vector<unsigned> &gen,
                           const std::vector<unsigned> &slice_indices_by_inst,
                           std::map<unsigned,
                              std::pair<unsigned,unsigned> > &crossing_counts,
                           std::vector<Instruction*> &crossing_instructions);
    public:
      inline void update_last_fence(GetTermEvent *fence)
        { last_fence = fence; }
      inline ApEvent get_fence_completion(void) { return fence_completion; }
      void record_remote_memoizable(Memoizable *memo);
      void release_remote_memos(void);
    private:
      PhysicalTrace * const trace;
      std::atomic<bool> recording;
      Replayable replayable;
      mutable LocalLock template_lock;
      const unsigned fence_completion_id;
      const unsigned replay_parallelism;
    private:
      std::deque<std::map<TraceLocalID,Memoizable*> > operations;
      std::deque<std::pair<ApEvent,bool/*recurrent*/> > pending_replays;
      std::map<TraceLocalID,std::pair<unsigned,bool/*task*/> > memo_entries;
      // Remote memoizable objects that we have ownership for
      std::vector<Memoizable*> remote_memos;
    private:
      CachedMappings cached_mappings;
      std::map<TraceLocalID,std::map<Reservation,bool> > cached_reservations;
      bool has_virtual_mapping;
    private:
      GetTermEvent                    *last_fence;
    private:
      ApEvent                         fence_completion;
      std::vector<ApEvent>            events;
      std::map<unsigned,ApUserEvent>  user_events;
      std::map<ApEvent,unsigned>      event_map;
    private:
      std::vector<Instruction*>               instructions;
      std::vector<std::vector<Instruction*> > slices;
    private:
      std::map<unsigned/*event*/,unsigned/*consumers*/> crossing_events;
      // Frontiers of a template are a set of users whose events must
      // be carried over to the next replay for eliding the fence at the
      // beginning. For each user i in frontiers, frontiers[i] points to the
      // user i's event carried over from the previous replay. This data
      // structure is constructed by de-duplicating the last users of all
      // views used in the template, which are stored in view_users.
      // - frontiers[idx] == (event idx from the previous trace)
      // - after each replay, we do assignment 
      //    events[frontiers[idx]] = events[idx]
      std::map<unsigned,unsigned> frontiers;
    private:
      RtUserEvent recording_done;
      RtEvent transitive_reduction_done;
      std::atomic<std::vector<unsigned>*> pending_inv_topo_order;
      std::atomic<
        std::vector<std::vector<unsigned> >*> pending_transitive_reduction;
    private:
      std::map<TraceLocalID,ViewExprs> op_views;
      std::map<unsigned,ViewExprs>     copy_views;
      std::map<unsigned,ViewExprs>     src_indirect_views;
      std::map<unsigned,ViewExprs>     dst_indirect_views;
      std::vector<IssueAcross*>        across_copies;
    private:
      TraceConditionSet   pre, post;
      ViewGroups          view_groups;
      // This data structure holds a set of last users for each view.
      // Each user (which is an index in the event table) is associated with
      // a field mask, an index expression representing the working set within
      // the view, and privilege. For any given pair of view and index
      // expression, there can be either multiple readers/reducers or a single
      // writer.
      ViewUsers           view_users;
      std::set<ViewUser*> all_users;
    private:
      TraceViewSet pre_reductions;
      TraceViewSet post_reductions;
      TraceViewSet consumed_reductions;
    private:
      FieldMaskSet<FillView> pre_fill_views;
      FieldMaskSet<FillView> post_fill_views;
#ifdef LEGION_SPY
    private:
      UniqueID prev_fence_uid;
#endif
    private:
      friend class PhysicalTrace;
      friend class Instruction;
#ifdef DEBUG_LEGION
      friend class GetTermEvent;
      friend class CreateApUserEvent;
      friend class TriggerEvent;
      friend class MergeEvent;
      friend class AssignFenceCompletion;
      friend class IssueCopy;
      friend class IssueFill;
      friend class IssueAcross;
      friend class SetOpSyncEvent;
      friend class SetEffects;
      friend class CompleteReplay;
#endif
    };

    enum InstructionKind
    {
      GET_TERM_EVENT = 0,
      CREATE_AP_USER_EVENT,
      TRIGGER_EVENT,
      MERGE_EVENT,
      ISSUE_COPY,
      ISSUE_FILL,
      ISSUE_ACROSS,
      SET_OP_SYNC_EVENT,
      SET_EFFECTS,
      ASSIGN_FENCE_COMPLETION,
      COMPLETE_REPLAY,
    };

    /**
     * \class Instruction
     * This class is an abstract parent class for all template instructions.
     */
    class Instruction {
    public:
      Instruction(PhysicalTemplate& tpl, const TraceLocalID &owner);
      virtual ~Instruction(void) {};
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay) = 0;
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations) = 0;

      virtual InstructionKind get_kind(void) = 0;
      virtual GetTermEvent* as_get_term_event(void) { return NULL; }
      virtual CreateApUserEvent* as_create_ap_user_event(void) { return NULL; }
      virtual TriggerEvent* as_trigger_event(void) { return NULL; }
      virtual MergeEvent* as_merge_event(void) { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void) 
        { return NULL; }
      virtual IssueCopy* as_issue_copy(void) { return NULL; }
      virtual IssueFill* as_issue_fill(void) { return NULL; }
      virtual IssueAcross* as_issue_across(void) { return NULL; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void) { return NULL; }
      virtual SetEffects* as_set_effects(void) { return NULL; }
      virtual CompleteReplay* as_complete_replay(void) { return NULL; }
    public:
      const TraceLocalID owner;
    };

    /**
     * \class GetTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_memo_completion()
     */
    class GetTermEvent : public Instruction {
    public:
      GetTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                   const TraceLocalID& rhs, bool fence);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return GET_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class CreateApUserEvent
     * This instruction has the following semantics:
     *   events[lhs] = Runtime::create_ap_user_event()
     */
    class CreateApUserEvent : public Instruction {
    public:
      CreateApUserEvent(PhysicalTemplate& tpl, unsigned lhs,
                        const TraceLocalID &owner);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return CREATE_AP_USER_EVENT; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class TriggerEvent
     * This instruction has the following semantics:
     *   Runtime::trigger_event(events[lhs], events[rhs])
     */
    class TriggerEvent : public Instruction {
    public:
      TriggerEvent(PhysicalTemplate& tpl, unsigned lhs, unsigned rhs,
                   const TraceLocalID &owner);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return TRIGGER_EVENT; }
      virtual TriggerEvent* as_trigger_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      unsigned rhs;
    };

    /**
     * \class MergeEvent
     * This instruction has the following semantics:
     *   events[lhs] = Runtime::merge_events(events[rhs])
     */
    class MergeEvent : public Instruction {
    public:
      MergeEvent(PhysicalTemplate& tpl, unsigned lhs,
                 const std::set<unsigned>& rhs,
                 const TraceLocalID &owner);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return MERGE_EVENT; }
      virtual MergeEvent* as_merge_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      std::set<unsigned> rhs;
    };

    /**
     * \class AssignFenceCompletion
     * This instruction has the following semantics:
     *   events[lhs] = fence_completion
     */
    class AssignFenceCompletion : public Instruction {
      AssignFenceCompletion(PhysicalTemplate& tpl, unsigned lhs,
                            const TraceLocalID &owner);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return ASSIGN_FENCE_COMPLETION; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      PhysicalTemplate &tpl;
      unsigned lhs;
    };

    /**
     * \class IssueFill
     * This instruction has the following semantics:
     *
     *   events[lhs] = expr->fill(fields, fill_value, fill_size,
     *                            events[precondition_idx]);
     */
    class IssueFill : public Instruction {
    public:
      IssueFill(PhysicalTemplate& tpl,
                unsigned lhs, IndexSpaceExpression *expr,
                const TraceLocalID &op_key,
                const std::vector<CopySrcDstField> &fields,
                const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                FieldSpace handle, RegionTreeID tree_id,
#endif
                unsigned precondition_idx);
      virtual ~IssueFill(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return ISSUE_FILL; }
      virtual IssueFill* as_issue_fill(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      IndexSpaceExpression *expr;
      std::vector<CopySrcDstField> fields;
      void *fill_value;
      size_t fill_size;
#ifdef LEGION_SPY
      FieldSpace handle;
      RegionTreeID tree_id;
#endif
      unsigned precondition_idx;
    };

    /**
     * \class IssueCopy
     * This instruction has the following semantics:
     *   events[lhs] = expr->issue_copy(src_fields, dst_fields,
     *                                  events[precondition_idx],
     *                                  predicate_guard,
     *                                  redop, reduction_fold);
     */
    class IssueCopy : public Instruction {
    public:
      IssueCopy(PhysicalTemplate &tpl,
                unsigned lhs, IndexSpaceExpression *expr,
                const TraceLocalID &op_key,
                const std::vector<CopySrcDstField>& src_fields,
                const std::vector<CopySrcDstField>& dst_fields,
                const std::vector<Reservation>& reservations,
#ifdef LEGION_SPY
                RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                unsigned precondition_idx);
      virtual ~IssueCopy(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return ISSUE_COPY; }
      virtual IssueCopy* as_issue_copy(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      IndexSpaceExpression *expr;
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      std::vector<Reservation> reservations;
#ifdef LEGION_SPY
      RegionTreeID src_tree_id;
      RegionTreeID dst_tree_id;
#endif
      unsigned precondition_idx;
    };

    /**
     * \class IssueAcross
     * This instruction has the following semantics:
     *  events[lhs] = executor->execute(ops[key], predicate_guard,
     *                                  events[copy_precondition],
     *                                  events[src_indirect_precondition],
     *                                  events[dst_indirect_precondition])
     */
    class IssueAcross : public Instruction {
    public:
      IssueAcross(PhysicalTemplate &tpl, unsigned lhs,
                  unsigned copy_pre, unsigned collective_pre,
                  unsigned src_indirect_pre, unsigned dst_indirect_pre,
                  const TraceLocalID &op_key,
                  CopyAcrossExecutor *executor);
      virtual ~IssueAcross(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return ISSUE_ACROSS; }
      virtual IssueAcross* as_issue_across(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      unsigned copy_precondition;
      unsigned collective_precondition;
      unsigned src_indirect_precondition;
      unsigned dst_indirect_precondition;
      CopyAcrossExecutor *const executor;
    };

    /**
     * \class SetOpSyncEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].compute_sync_precondition()
     */
    class SetOpSyncEvent : public Instruction {
    public:
      SetOpSyncEvent(PhysicalTemplate& tpl, unsigned lhs,
                     const TraceLocalID& rhs);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return SET_OP_SYNC_EVENT; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class SetEffects
     * This instruction has the following semantics:
     *   operations[lhs].set_effects_postcondition(events[rhs])
     */
    class SetEffects : public Instruction {
    public:
      SetEffects(PhysicalTemplate& tpl, const TraceLocalID& lhs, unsigned rhs);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return SET_EFFECTS; }
      virtual SetEffects* as_set_effects(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned rhs;
    };

    /**
     * \class CompleteReplay
     * This instruction has the following semantics:
     *   operations[lhs]->complete_replay(events[rhs])
     */
    class CompleteReplay : public Instruction {
    public:
      CompleteReplay(PhysicalTemplate& tpl, const TraceLocalID& lhs,
                     unsigned rhs);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,Memoizable*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(
                           std::map<TraceLocalID,Memoizable*> &operations);

      virtual InstructionKind get_kind(void)
        { return COMPLETE_REPLAY; }
      virtual CompleteReplay* as_complete_replay(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned rhs;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TRACE__
