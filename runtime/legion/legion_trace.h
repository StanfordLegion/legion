/* Copyright 2023 Stanford University, NVIDIA Corporation
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
#include "legion/legion_allocation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LogicalTrace
     * The logical trace class captures the tracing information
     * for the logical dependence analysis so that it can be 
     * replayed without needing to perform the analysis again
     */
    class LogicalTrace : public Collectable {
    public:
      enum TracingState {
        LOGICAL_ONLY,
        PHYSICAL_RECORD,
        PHYSICAL_REPLAY,
      };
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
            next_idx(nidx), validates(val),
            dtype(d), dependent_mask(m) { }
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
      struct CloseInfo {
      public:
        CloseInfo(MergeCloseOp *op, unsigned idx,
#ifdef DEBUG_LEGION_COLLECTIVES
                  RegionTreeNode *n,
#endif
                  const RegionRequirement &r)
          : close_op(op), requirement(r), creator_idx(idx)
#ifdef DEBUG_LEGION_COLLECTIVES
            , node(n)
#endif
        { }
      public:
        MergeCloseOp *close_op; // only valid during capture
        RegionRequirement requirement;
        LegionVector<DependenceRecord> dependences;
        FieldMask close_mask;
        unsigned creator_idx;
#ifdef DEBUG_LEGION_COLLECTIVES
        RegionTreeNode *node;
#endif
      };
      struct OperationInfo {
      public:
        OperationInfo(Operation *op)
          : kind(op->get_operation_kind()),
            region_count(op->get_region_count()) { }
      public:
        LegionVector<DependenceRecord> dependences;
        LegionVector<CloseInfo> closes;
        Operation::OpKind kind;
        unsigned region_count;
      };
      class StaticTranslator {
      public:
        StaticTranslator(const std::set<RegionTreeID> *trs)
        { if (trs != NULL) trees.insert(trs->begin(), trs->end()); }
      public:
        inline bool skip_analysis(RegionTreeID tid) const
        { if (trees.empty()) return true; 
          else return (trees.find(tid) != trees.end()); }
        inline void push_dependences(const std::vector<StaticDependence> *deps)
        {
          AutoLock t_lock(translator_lock);
          if (deps != NULL)
            dependences.emplace_back(*deps);
          else
            dependences.resize(dependences.size() + 1);
        }
        inline void pop_dependences(std::vector<StaticDependence> &deps)
        {
          AutoLock t_lock(translator_lock);
#ifdef DEBUG_LEGION
          assert(!dependences.empty());
#endif
          deps.swap(dependences.front());
          dependences.pop_front();
        }
      public:
        LocalLock translator_lock;
        std::deque<std::vector<StaticDependence> > dependences;
        std::set<RegionTreeID> trees;
      };
    public:
      LogicalTrace(InnerContext *ctx, TraceID tid, bool logical_only,
                   bool static_trace, Provenance *provenance,
                   const std::set<RegionTreeID> *trees);
      ~LogicalTrace(void);
    public:
      inline TraceID get_trace_id(void) const { return tid; }
    public:
      bool initialize_op_tracing(Operation *op,
                     const std::vector<StaticDependence> *dependences = NULL);
      bool skip_analysis(RegionTreeID tid) const;
      size_t register_operation(Operation *op, GenerationID gen);
      void register_internal(InternalOp *op);
      void register_close(MergeCloseOp *op, unsigned creator_idx,
#ifdef DEBUG_LEGION_COLLECTIVES
                          RegionTreeNode *node,
#endif
                          const RegionRequirement &req);
      bool record_dependence(Operation *target, GenerationID target_gen,
                                Operation *source, GenerationID source_gen);
      bool record_region_dependence(Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
    public:
      // Called by task execution thread
      inline bool is_fixed(void) const { return fixed; }
      void fix_trace(Provenance *provenance);
    public:
      bool has_physical_trace(void) { return physical_trace != NULL; }
      PhysicalTrace* get_physical_trace(void) { return physical_trace; }
    public:
      void begin_trace_execution(FenceOp *fence_op);
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
    protected:
      void replay_operation_dependences(Operation *op,
          const LegionVector<DependenceRecord> &dependences);
      void translate_dependence_records(Operation *op, const unsigned index,
          const std::vector<StaticDependence> &dependences);
#ifdef LEGION_SPY
    public:
      UniqueID get_current_uid_by_index(unsigned op_idx) const;
#endif
    public:
      InnerContext *const context;
      const TraceID tid;
      Provenance *const begin_provenance;
      // Set after end_trace is called
      Provenance *end_provenance;
    protected:
      std::atomic<TracingState> state;
      // Pointer to a physical trace
      PhysicalTrace *physical_trace;
      bool blocking_call_observed;
      bool has_intermediate_ops;
      bool fixed;
      bool recording;
      size_t replay_index;
      std::deque<OperationInfo> replay_info;
      std::set<std::pair<Operation*,GenerationID> > frontiers;
      std::vector<std::pair<Operation*,GenerationID> > operations;
      // Only need this backwards lookup for trace capture
      std::map<std::pair<Operation*,GenerationID>,unsigned> op_map;
      FenceOp *trace_fence;
      GenerationID trace_fence_gen;
      StaticTranslator *static_translator;
#ifdef LEGION_SPY
    protected:
      std::map<std::pair<Operation*,GenerationID>,UniqueID> current_uids;
      std::map<std::pair<Operation*,GenerationID>,unsigned> num_regions;
#endif
    };

    class TraceOp : public FenceOp {
    public:
      TraceOp(Runtime *rt);
      TraceOp(const TraceOp &rhs);
      virtual ~TraceOp(void);
    public:
      TraceOp& operator=(const TraceOp &rhs);
    public:
      virtual bool is_tracing_fence(void) const override { return true; }
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
                    bool remove_trace_reference, Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
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
                               Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
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
      void initialize_replay(InnerContext *ctx, LogicalTrace *trace,
                             Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
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
      void initialize_begin(InnerContext *ctx, LogicalTrace *trace,
                            Provenance *provenance);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
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
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
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
      PhysicalTrace(Runtime *runtime, LogicalTrace *logical_trace);
      PhysicalTrace(const PhysicalTrace &rhs) = delete;
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace &rhs) = delete;
    public:
      bool check_memoize_consensus(size_t index);
      void reset_last_memoized(void);
      void clear_cached_template(void) { current_template = NULL; }
      void check_template_preconditions(TraceReplayOp *op,
                                        std::set<RtEvent> &applied_events);
      // Return true if we evaluated all the templates
      bool find_viable_templates(ReplTraceReplayOp *op, 
                                 std::set<RtEvent> &applied_events,
                                 unsigned templates_to_find,
                                 std::vector<int> &viable_templates);
      void select_template(unsigned template_index);
    public:
      inline PhysicalTemplate* get_current_template(void) 
        { return current_template; }
      inline bool has_any_templates(void) const { return !templates.empty(); }
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
      PhysicalTemplate* start_new_template(TaskTreeCoordinates &&cordinates);
      ApEvent record_replayable_capture(PhysicalTemplate *tpl,
                    std::set<RtEvent> &map_applied_conditions);
      void record_failed_capture(PhysicalTemplate *tpl);
      void record_intermediate_execution_fence(FenceOp *fence);
      void chain_replays(FenceOp *replay_op);
    public:
      const std::vector<Processor> &get_replay_targets(void)
        { return replay_targets; }
    public:
      void initialize_template(ApEvent fence_completion, bool recurrent);
    public:
      Runtime * const runtime;
      const LogicalTrace *logical_trace;
      const bool perform_fence_elision;
      ReplicateContext *const repl_ctx;
    private:
      mutable LocalLock trace_lock;
      FenceOp *previous_replay;
      UniqueID previous_replay_gen;
      PhysicalTemplate* current_template;
      std::vector<PhysicalTemplate*> templates;
      unsigned nonreplayable_count;
      unsigned new_template_count;
      size_t last_memoized;
    private:
      ApEvent previous_template_completion;
      ApEvent execution_fence_event;
      std::vector<Processor> replay_targets;
    private:
      bool intermediate_execution_fence;
    };

    /**
     * \class TraceViewSet
     * The trace view set stores a temporary collection of instance views
     * with valid expressions and fields for each instance. We maintain 
     * the important invariant here in this class that each physical
     * instance has at most one view representing it, which requires
     * anti-aliasing collective views.
     */
    class TraceViewSet {
    public:
      struct FailedPrecondition {
        LogicalView *view;
        IndexSpaceExpression *expr;
        FieldMask mask;

        std::string to_string(TaskContext *ctx) const;
      };
    public:
      TraceViewSet(InnerContext *context, DistributedID owner_did,
                   IndexSpaceExpression *expr, RegionTreeID tree_id);
      virtual ~TraceViewSet(void);
    public:
      void insert(LogicalView *view,
                  IndexSpaceExpression *expr,
                  const FieldMask &mask, bool antialiased = false);
      void invalidate(LogicalView *view,
                      IndexSpaceExpression *expr,
                      const FieldMask &mask,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL, 
                      bool antialiased = false);
      void invalidate_all_but(LogicalView *except,
                              IndexSpaceExpression *expr,
                              const FieldMask &mask,
           std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove = NULL,
           std::map<LogicalView*,unsigned> *view_refs_to_remove = NULL,
                              bool antialiased = false);
    public:
      bool dominates(LogicalView *view, IndexSpaceExpression *expr, 
                     FieldMask &non_dominated) const;
      void dominates(LogicalView *view, 
                     IndexSpaceExpression *expr, FieldMask mask,
                     FieldMaskSet<IndexSpaceExpression> &non_dominated,
                     FieldMaskSet<IndexSpaceExpression> *dominate = NULL) const;
      void filter_independent_fields(IndexSpaceExpression *expr,
                                     FieldMask &mask) const;
      bool subsumed_by(const TraceViewSet &set, bool allow_independent,
                       FailedPrecondition *condition = NULL) const;
      bool independent_of(const TraceViewSet &set,
                       FailedPrecondition *condition = NULL) const; 
      void record_first_failed(FailedPrecondition *condition = NULL) const;
      void transpose_uniquely(LegionMap<IndexSpaceExpression*,
                                        FieldMaskSet<LogicalView> > &target,
                          std::set<IndexSpaceExpression*> &unique_exprs) const;
      void find_overlaps(TraceViewSet &target, IndexSpaceExpression *expr,
                         const bool expr_covers, const FieldMask &mask) const;
      bool empty(void) const;
    public:
      void merge(TraceViewSet &target) const;
      void pack(Serializer &rez, AddressSpaceID target, 
                const bool pack_references) const;
      void unpack(Deserializer &derez, size_t num_views,
                  AddressSpaceID source, std::set<RtEvent> &ready_events);
      void unpack_references(void) const;
    public:
      void dump(void) const;
    public:
      InstanceView *find_instance_view(const std::vector<DistributedID> &dids);
    protected:
      bool has_overlapping_expressions(LogicalView *view,
                       const FieldMaskSet<IndexSpaceExpression> &left_exprs,
                       const FieldMaskSet<IndexSpaceExpression> &right_exprs,
                       FailedPrecondition *condition) const;
      void antialias_individual_view(IndividualView *view, FieldMask mask);
      void antialias_collective_view(CollectiveView *view, FieldMask mask,
                                     FieldMaskSet<InstanceView> &altviews);
    protected:
      typedef LegionMap<LogicalView*,
                        FieldMaskSet<IndexSpaceExpression> > ViewExprs;
    public:
      InnerContext *const context;
      IndexSpaceExpression *const expression;
      const RegionTreeID tree_id;
      const DistributedID owner_did;
    protected:
      // At most one expression per field
      ViewExprs conditions;
      bool has_collective_views;
    };

    /**
     * \class TraceConditionSet
     */
    class TraceConditionSet : public EqSetTracker, public Collectable,
                              public LegionHeapify<TraceConditionSet> {
    public:
      struct DeferTracePreconditionTestArgs :
        public LgTaskArgs<DeferTracePreconditionTestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_TRACE_PRECONDITION_TASK_ID;
      public:
        DeferTracePreconditionTestArgs(TraceConditionSet *s, Operation *o, 
                                       RtUserEvent d, RtUserEvent a)
          : LgTaskArgs<DeferTracePreconditionTestArgs>(o->get_unique_op_id()),
            set(s), op(o), done_event(d), applied_event(a) { }
      public:
        TraceConditionSet *const set;
        Operation *const op;
        const RtUserEvent done_event;
        const RtUserEvent applied_event;
      };
      struct DeferTracePostconditionTestArgs :
        public LgTaskArgs<DeferTracePostconditionTestArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_TRACE_POSTCONDITION_TASK_ID;
      public:
        DeferTracePostconditionTestArgs(TraceConditionSet *s, Operation *o, 
                                        RtUserEvent d)
          : LgTaskArgs<DeferTracePostconditionTestArgs>(o->get_unique_op_id()),
            set(s), op(o), done_event(d) { }
      public:
        TraceConditionSet *const set;
        Operation *const op;
        const RtUserEvent done_event;
      };
    public:
      TraceConditionSet(PhysicalTrace *trace, RegionTreeForest *forest, 
                        unsigned parent_req_index, IndexSpaceExpression *expr,
                        const FieldMask &mask, RegionTreeID tree_id);
      TraceConditionSet(const TraceConditionSet &rhs) = delete;
      virtual ~TraceConditionSet(void);
    public:
      TraceConditionSet& operator=(const TraceConditionSet &rhs) = delete;
    public:
      virtual void add_subscription_reference(unsigned count = 1)
        { add_reference(count); }
      virtual bool remove_subscription_reference(unsigned count = 1)
        { return remove_reference(count); }
      virtual RegionTreeID get_region_tree_id(void) const
        { return tree_id; }
      virtual IndexSpaceExpression* get_tracker_expression(void) const
        { return condition_expr; }
      virtual ReferenceSource get_reference_source_kind(void) const 
        { return TRACE_REF; }
    public:
      void invalidate_equivalence_sets(void);
      void capture(EquivalenceSet *set, const FieldMask &mask,
                   std::vector<RtEvent> &ready_events);
      void receive_capture(TraceViewSet *pre, TraceViewSet *anti,
                           TraceViewSet *post, std::set<RtEvent> &ready);
      bool is_empty(void) const;
      bool is_replayable(bool &not_subsumed, 
                         TraceViewSet::FailedPrecondition *failed);
      void dump_preconditions(void) const;
      void dump_anticonditions(void) const;
      void dump_postconditions(void) const;
    public:
      void test_require(Operation *op, std::set<RtEvent> &ready_events,
                        std::set<RtEvent> &applied_events);
      bool check_require(void);
      void ensure(Operation *op, std::set<RtEvent> &applied_events);
    public:
      static void handle_precondition_test(const void *args);
      static void handle_postcondition_test(const void *args);
    public:
      RtEvent recompute_equivalence_sets(UniqueID opid, 
                        const FieldMask &invalid_mask);
    public:
      InnerContext *const context;
      RegionTreeForest *const forest;
      IndexSpaceExpression *const condition_expr;
      const FieldMask condition_mask;
      const RegionTreeID tree_id;
      const unsigned parent_req_index;
    private:
      mutable LocalLock set_lock;
    private:
      TraceViewSet *precondition_views;
      TraceViewSet *anticondition_views;
      TraceViewSet *postcondition_views; 
      // Transpose of conditions for testing
      typedef LegionMap<IndexSpaceExpression*,
                        FieldMaskSet<LogicalView> > ExprViews;
      ExprViews preconditions;
      ExprViews anticonditions;
      ExprViews postconditions;
      // A unique set of index space expressions from the *_ views
      // This is needed because transpose_uniquely might not capture
      // all the needed expression references
      std::set<IndexSpaceExpression*> unique_view_expressions;
    private:
      std::vector<InvalidInstAnalysis*> precondition_analyses;
      std::vector<AntivalidInstAnalysis*> anticondition_analyses;
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
      struct CachedPremapping
      {
        std::vector<Memory>     future_locations;
      };
      typedef std::map<TraceLocalID,CachedPremapping> CachedPremappings;
      struct CachedMapping
      {
        VariantID               chosen_variant;
        TaskPriority            task_priority;
        bool                    postmap_task;
        std::vector<Processor>  target_procs;
        std::vector<Memory>     future_locations;
        std::deque<InstanceSet> physical_instances;
      };
      typedef LegionMap<TraceLocalID,CachedMapping> CachedMappings;
    private:
      struct CachedAllreduce {
        std::vector<Memory> target_memories;
        size_t future_size;
      };
    protected:
      struct InstanceUser {
      public:
        InstanceUser(void) : expr(NULL) { }
        InstanceUser(const UniqueInst &i, const RegionUsage &r,
                     IndexSpaceExpression *e, const FieldMask &m)
          : instance(i), usage(r), expr(e), mask(m) { }
      public:
        inline bool matches(const UniqueInst &inst, const RegionUsage &use,
                            IndexSpaceExpression *expression) const
        {
          if (inst != instance) return false;
          if (use != usage) return false;
          return (expr == expression);
        }
        inline bool matches(const InstanceUser &user) const
        {
          if (instance != user.instance) return false;
          if (usage != user.usage) return false;
          if (expr != user.expr) return false;
          return (mask == user.mask);
        }
      public:
        UniqueInst instance;
        RegionUsage usage;
        IndexSpaceExpression *expr;
        FieldMask mask;
      };
      typedef LegionVector<InstanceUser> InstUsers;
      struct LastUserResult {
      public:
        LastUserResult(const InstanceUser &u) : user(u) { }
      public:
        const InstanceUser &user;
        std::set<ApEvent> events;
        std::vector<unsigned> frontiers;
      };
    private:
      // State for deferring the transitive reduction into time-slices since
      // it is really expensive and we don't want it monopolizing a processor
      struct TransitiveReductionState {
      public:
        TransitiveReductionState(RtUserEvent d)
          : stage(0), iteration(0), num_chains(0), pos(-1), done(d) { }
      public:
        std::vector<unsigned> topo_order, inv_topo_order; 
        std::vector<unsigned> remaining_edges, chain_indices;
        std::vector<std::vector<unsigned> > incoming, outgoing;
        std::vector<std::vector<unsigned> > incoming_reduced;
        std::vector<std::vector<int> > all_chain_frontiers;
        std::map<TraceLocalID, GetTermEvent*> term_insts;
        std::map<TraceLocalID, ReplayMapping*> replay_insts;
        unsigned stage, iteration, num_chains;
        int pos;
        const RtUserEvent done;
      };
      struct TransitiveReductionArgs :
        public LgTaskArgs<TransitiveReductionArgs> {
      public:
        static const LgTaskID TASK_ID = LG_TRANSITIVE_REDUCTION_TASK_ID;
      public:
        TransitiveReductionArgs(PhysicalTemplate *t,TransitiveReductionState *s)
          : LgTaskArgs<TransitiveReductionArgs>(implicit_provenance),
            tpl(t), state(s) { }
      public:
        PhysicalTemplate *const tpl;
        TransitiveReductionState *const state;
      };
    public:
      PhysicalTemplate(PhysicalTrace *trace, ApEvent fence_event,
                       TaskTreeCoordinates &&cordinates);
      PhysicalTemplate(const PhysicalTemplate &rhs) = delete;
      virtual ~PhysicalTemplate(void);
    public:
      PhysicalTemplate& operator=(const PhysicalTemplate &rhs) = delete;
    public:
      virtual size_t get_sharded_template_index(void) const { return 0; }
      virtual void initialize_replay(ApEvent fence_completion, bool recurrent,
                                     bool need_lock = true);
      virtual void perform_replay(Runtime *rt, 
                                  std::set<RtEvent> &replayed_events);
      virtual RtEvent refresh_managed_barriers(void);
      virtual void finish_replay(std::set<ApEvent> &postconditions);
      virtual ApEvent get_completion_for_deletion(void) const;
    public:
      void find_execution_fence_preconditions(std::set<ApEvent> &preconditions);
      void finalize(InnerContext *context, Operation *op,
                    bool has_blocking_call);
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
    protected:
      virtual Replayable check_replayable(Operation *op, 
          InnerContext *context, bool has_blocking_call);
    public:
      void optimize(Operation *op, bool do_transitive_reduction);
    private:
      void find_all_last_instance_user_events(
                             std::vector<RtEvent> &frontier_events);
      void find_last_instance_events(const InstUsers &users,
                             std::vector<RtEvent> &frontier_events);
      void compute_frontiers(std::vector<RtEvent> &frontier_events);
      void elide_fences(std::vector<unsigned> &gen,
                        std::vector<RtEvent> &ready_events);
      void propagate_merges(std::vector<unsigned> &gen);
      void transitive_reduction(TransitiveReductionState *state, bool deferred);
      void finalize_transitive_reduction(
          const std::vector<unsigned> &inv_topo_order,
          const std::vector<std::vector<unsigned> > &incoming_reduced);
      void propagate_copies(std::vector<unsigned> *gen);
      void eliminate_dead_code(std::vector<unsigned> &gen);
      void prepare_parallel_replay(const std::vector<unsigned> &gen);
      void push_complete_replays(void);
    protected:
      virtual void sync_compute_frontiers(Operation *op,
                          const std::vector<RtEvent> &frontier_events);
      virtual void initialize_generators(std::vector<unsigned> &new_gen);
      virtual void initialize_eliminate_dead_code_frontiers(
                          const std::vector<unsigned> &gen,
                                std::vector<bool> &used);
      virtual void initialize_transitive_reduction_frontiers(
                          std::vector<unsigned> &topo_order,
                          std::vector<unsigned> &inv_topo_order);
      virtual void record_used_frontiers(std::vector<bool> &used,
                      const std::vector<unsigned> &gen) const;
      virtual void rewrite_frontiers(
                      std::map<unsigned,unsigned> &substitutions);
    public:
      // Variants for normal traces
      bool check_preconditions(TraceReplayOp *op,
                               std::set<RtEvent> &applied_events);
      void apply_postcondition(TraceSummaryOp *op,
                               std::set<RtEvent> &applied_events);
      // Variants for control replication traces 
      bool check_preconditions(ReplTraceReplayOp *op,
                               std::set<RtEvent> &applied_events);
      void apply_postcondition(ReplTraceSummaryOp *op,
                               std::set<RtEvent> &applied_events);
    public:
      void register_operation(MemoizableOp *op);
      void execute_slice(unsigned slice_idx, bool recurrent_replay);
    public:
      virtual void issue_summary_operations(InnerContext* context,
                                            Operation *invalidator,
                                            Provenance *provenance);
    public:
      void dump_template(void);
      virtual void dump_sharded_template(void) { }
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
      inline void record_no_consensus(void) { has_no_consensus = true; }
      inline bool get_no_consensus(void) { return has_no_consensus; }
    public:
      virtual bool is_recording(void) const { return recording.load(); }
      virtual void add_recorder_reference(void) { /*do nothing*/ }
      virtual bool remove_recorder_reference(void) 
        { /*do nothing, never delete*/ return false; }
      virtual void pack_recorder(Serializer &rez, 
                                 std::set<RtEvent> &applied);
    public:
      void record_premap_output(MemoizableOp *memo,
                                const Mapper::PremapTaskOutput &output,
                                std::set<RtEvent> &applied_events);
      void get_premap_output(IndexTask *task,
                             std::vector<Memory> &future_locations);     
      virtual void record_mapper_output(const TraceLocalID &tlid,
                             const Mapper::MapTaskOutput &output,
                             const std::deque<InstanceSet> &physical_instances,
                             std::set<RtEvent> &applied_events);
      void get_mapper_output(SingleTask *task,
                             VariantID &chosen_variant,
                             TaskPriority &task_priority,
                             bool &postmap_task,
                             std::vector<Processor> &target_proc,
                             std::vector<Memory> &future_locations,
                             std::deque<InstanceSet> &physical_instances) const;
      void get_task_reservations(SingleTask *task,
                             std::map<Reservation,bool> &reservations) const;
      void get_allreduce_mapping(AllReduceOp *op,
          std::vector<Memory> &target_memories, size_t &future_size);
    public:
      virtual void record_completion_event(ApEvent lhs, unsigned op_kind,
                                           const TraceLocalID &tlid);
      virtual void record_replay_mapping(ApEvent lhs, unsigned op_kind,
                          const TraceLocalID &tlid, bool register_memo);
      virtual void request_term_event(ApUserEvent &term_event);
      virtual void record_create_ap_user_event(ApUserEvent &lhs, 
                                               const TraceLocalID &tlid);
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        const TraceLocalID &tlid);
    public:
      virtual void record_merge_events(ApEvent &lhs, ApEvent rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2, 
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, ApEvent e1, ApEvent e2,
                                       ApEvent e3, const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::set<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::vector<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(PredEvent &lhs,
                                       PredEvent e1, PredEvent e2,
                                       const TraceLocalID &tlid);
      virtual void record_collective_barrier(ApBarrier bar, ApEvent pre,
                    const std::pair<size_t,size_t> &key, size_t arrival_count);
      virtual ShardID record_managed_barrier(ApBarrier bar,
                                             size_t total_arrivals);
      virtual void record_barrier_arrival(ApBarrier bar, ApEvent pre,
                    size_t arrival_count, std::set<RtEvent> &applied,
                    ShardID owner_shard);
    public:
      virtual void record_issue_copy(const TraceLocalID &tlid, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                             RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                             ApEvent precondition, PredEvent pred_guard,
                             LgEvent src_unique, LgEvent dst_unique,
                             int priority, CollectiveKind collective);
      virtual void record_issue_across(const TraceLocalID &tlid, ApEvent &lhs,
                             ApEvent collective_precondition,
                             ApEvent copy_precondition,
                             ApEvent src_indirect_precondition,
                             ApEvent dst_indirect_precondition,
                             CopyAcrossExecutor *executor);
      virtual void record_copy_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const UniqueInst &src_inst,
                           const UniqueInst &dst_inst,
                           const FieldMask &src_mask, const FieldMask &dst_mask,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           ReductionOpID redop, std::set<RtEvent> &applied);
      virtual void record_across_insts(ApEvent lhs, const TraceLocalID &tlid,
                           unsigned src_idx, unsigned dst_idx,
                           IndexSpaceExpression *expr,
                           const AcrossInsts &src_insts,
                           const AcrossInsts &dst_insts,
                           PrivilegeMode src_mode, PrivilegeMode dst_mode,
                           bool src_indirect, bool dst_indirect,
                           std::set<RtEvent> &applied);
      virtual void record_indirect_insts(ApEvent indirect_done,
                           ApEvent all_done, IndexSpaceExpression *expr,
                           const AcrossInsts &insts,
                           std::set<RtEvent> &applied, PrivilegeMode priv);
      virtual void record_issue_fill(const TraceLocalID &tlid, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField> &fields,
                             const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                             UniqueID fill_uid,
                             FieldSpace handle, 
                             RegionTreeID tree_id,
#endif
                             ApEvent precondition, PredEvent pred_guard,
                             LgEvent unique_event, int priority,
                             CollectiveKind collective);
    public:
      virtual void record_op_inst(const TraceLocalID &tlid,
                                  unsigned idx,
                                  const UniqueInst &inst,
                                  RegionNode *node,
                                  const RegionUsage &usage,
                                  const FieldMask &user_mask,
                                  bool update_validity,
                                  std::set<RtEvent> &applied);
      virtual void record_fill_inst(ApEvent lhs, IndexSpaceExpression *expr, 
                           const UniqueInst &inst,
                           const FieldMask &fill_mask,
                           std::set<RtEvent> &applied_events,
                           const bool reduction_initialization);
    protected:
      void record_instance_user(InstUsers &users,
                                const UniqueInst &instance,
                                const RegionUsage &usage,
                                IndexSpaceExpression *expr,
                                const FieldMask &mask,
                                std::set<RtEvent> &applied_events);
      virtual void record_mutated_instance(const UniqueInst &inst,
                                           IndexSpaceExpression *expr,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &applied_events);
    public:
      virtual void record_set_op_sync_event(ApEvent &lhs,
                                            const TraceLocalID &tlid);
      virtual void record_complete_replay(const TraceLocalID &tlid,
                                          ApEvent pre, ApEvent post,
                                          std::set<RtEvent> &applied_events);
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events); 
      virtual void record_future_allreduce(const TraceLocalID &tlid,
          const std::vector<Memory> &target_memories, size_t future_size);
    public:
      virtual void record_owner_shard(unsigned trace_local_id, ShardID owner);
      virtual void record_local_space(unsigned trace_local_id, IndexSpace sp);
      virtual void record_sharding_function(unsigned trace_local_id, 
                                            ShardingFunction *function);
    public:
      virtual ShardID find_owner_shard(unsigned trace_local_id);
      virtual IndexSpace find_local_space(unsigned trace_local_id);
      virtual ShardingFunction* find_sharding_function(unsigned trace_local_id);
    public: 
      bool defer_template_deletion(ApEvent &pending_deletion,
                                   std::set<RtEvent> &applied_events);
    public:
      static void handle_replay_slice(const void *args);
      static void handle_transitive_reduction(const void *args);
      static void handle_delete_template(const void *args);
    protected:
      void record_memo_entry(const TraceLocalID &tlid, unsigned entry,
                             unsigned op_kind);
    protected:
#ifdef DEBUG_LEGION
      // This is a virtual method in debug mode only since we have an
      // assertion that we want to check in the ShardedPhysicalTemplate
      virtual unsigned convert_event(const ApEvent &event, bool check = true);
#else
      unsigned convert_event(const ApEvent &event);
#endif
      virtual unsigned find_event(const ApEvent &event, AutoLock &tpl_lock);
      void insert_instruction(Instruction *inst);
    protected:
      // Returns the set of last users for all <view,field mask,index expr>
      // tuples in the inst_exprs, not that this is the 
      void find_all_last_users(const InstUsers &inst_users,
                               std::set<unsigned> &last_users) const;
      virtual unsigned find_frontier_event(ApEvent event, 
                               std::vector<RtEvent> &ready_events);
      // Check to see if any users are mutating these fields and expressions
      virtual bool are_read_only_users(InstUsers &inst_users);
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
    public:
      PhysicalTrace * const trace;
      const TaskTreeCoordinates coordinates;
    protected:
      std::atomic<bool> recording;
      // Count how many times we've been replayed so we know when we're going
      // to run out of phase barrier generations
      // Note we start this at 1 since some barriers are used as part of the
      // capture, while others are not used until the first replay, that throws
      // away one barrier generation on some barriers, but whatever
      size_t total_replays;
      Replayable replayable;
    protected:
      mutable LocalLock template_lock;
      const unsigned fence_completion_id;
    private:
      const unsigned replay_parallelism;
    protected:
      static constexpr unsigned NO_INDEX = UINT_MAX;
    protected:
      std::deque<std::map<TraceLocalID,MemoizableOp*> > operations;
      std::deque<std::pair<ApEvent,bool/*recurrent*/> > pending_replays;
      // Pair in memo_entries is <entry index, Operation::Kind>
      // This data structure is only used during template capture and
      // can be ignored after the template has been optimized
      std::map<TraceLocalID,std::pair<unsigned,unsigned> > memo_entries;
    private:
      CachedPremappings cached_premappings;
      CachedMappings cached_mappings;
      std::map<TraceLocalID,std::map<Reservation,bool> > cached_reservations;
      std::map<TraceLocalID,CachedAllreduce> cached_allreduces;
      bool has_virtual_mapping;
      bool has_no_consensus;
    protected:
      GetTermEvent                    *last_fence;
    protected:
      ApEvent                         fence_completion;
      std::vector<ApEvent>            events;
      std::map<unsigned,ApUserEvent>  user_events;
    protected:
      std::map<ApEvent,unsigned> event_map;
      std::map<ApEvent,BarrierAdvance*> managed_barriers;
      std::map<ApEvent,std::vector<BarrierArrival*> > managed_arrivals;
    protected:
      std::vector<Instruction*>               instructions;
      std::vector<std::vector<Instruction*> > slices;
      std::vector<std::vector<TraceLocalID> > slice_tasks;
    protected:
      std::map<unsigned/*event*/,unsigned/*consumers*/> crossing_events;
      // Frontiers of a template are a set of users whose events must
      // be carried over to the next replay for eliding the fence at the
      // beginning. We compute this data structure from the last users of
      // each physical instance named in the trace and then looking for
      // the locations of those events inside the trace.
      // After each replay, we do the assignment
      // events[frontiers[idx]] = events[idx]
      std::map<unsigned,unsigned> frontiers;
      // A cache of the specific last user results for individual instances
      std::map<UniqueInst,std::deque<LastUserResult> > instance_last_users;
    protected:
      RtEvent transitive_reduction_done;
      std::atomic<TransitiveReductionState*> finished_transitive_reduction;
    private:
      std::map<TraceLocalID,InstUsers> op_insts;
      std::map<unsigned,InstUsers>     copy_insts;
      std::map<unsigned,InstUsers>     src_indirect_insts;
      std::map<unsigned,InstUsers>     dst_indirect_insts;
      std::vector<IssueAcross*>        across_copies;
      std::map<DistributedID,IndividualView*> recorded_views;
      std::set<IndexSpaceExpression*>  recorded_expressions;
    protected:
      // Capture the names of all the instances that are mutated by this trace
      // and the index space expressions and fields that were mutated
      // THIS IS SHARDED FOR CONTROL REPLICATION!!!
      LegionMap<UniqueInst,FieldMaskSet<IndexSpaceExpression> > mutated_insts;
    protected:
      // Capture the set of regions that we saw operations for, we'll use this
      // at the end of the trace capture to compute the equivalence sets for
      // this trace and then extract the different condition sets for this trace
      // THESE ARE SHARDED FOR CONTROL REPLICATION!!!
      FieldMaskSet<RegionNode> trace_regions;
      // Parent context requirement indexes for each of the regions
      std::map<RegionNode*,unsigned> trace_region_parent_req_indexes;
      std::vector<TraceConditionSet*> conditions;
#ifdef LEGION_SPY
    private:
      UniqueID prev_fence_uid;
#endif
    private:
      friend class PhysicalTrace;
      friend class Instruction;
#ifdef DEBUG_LEGION
      friend class GetTermEvent;
      friend class ReplayMapping;
      friend class CreateApUserEvent;
      friend class TriggerEvent;
      friend class MergeEvent;
      friend class AssignFenceCompletion;
      friend class IssueCopy;
      friend class IssueFill;
      friend class IssueAcross;
      friend class SetOpSyncEvent;
      friend class CompleteReplay;
      friend class AcquireReplay;
      friend class ReleaseReplay;
      friend class BarrierArrival;
      friend class BarrierAdvance;
#endif
    };

    /**
     * \class ShardedPhysicalTemplate
     * This is an extension of the PhysicalTemplate class for handling
     * templates for control replicated contexts. It mostly behaves the
     * same as a normal PhysicalTemplate but has some additional 
     * extensions for handling the effects of control replication.
     */
    class ShardedPhysicalTemplate : public PhysicalTemplate {
    public:
      enum UpdateKind {
        UPDATE_MUTATED_INST,
        READ_ONLY_USERS_REQUEST,
        READ_ONLY_USERS_RESPONSE,
        TEMPLATE_BARRIER_REFRESH,
        FRONTIER_BARRIER_REFRESH,
        REMOTE_BARRIER_SUBSCRIBE,
      };
    public:
      struct DeferTraceUpdateArgs : public LgTaskArgs<DeferTraceUpdateArgs> {
      public:
        static const LgTaskID TASK_ID = LG_DEFER_TRACE_UPDATE_TASK_ID;
      public:
        DeferTraceUpdateArgs(ShardedPhysicalTemplate *target, 
                             UpdateKind kind, RtUserEvent done, 
                             Deserializer &derez, const UniqueInst &inst,
                             RtUserEvent deferral = 
                              RtUserEvent::NO_RT_USER_EVENT);
        DeferTraceUpdateArgs(ShardedPhysicalTemplate *target, 
                             UpdateKind kind, RtUserEvent done, 
                             const UniqueInst &inst, Deserializer &derez,
                             IndexSpaceExpression *expr,
                             RtUserEvent deferral = 
                              RtUserEvent::NO_RT_USER_EVENT);
        DeferTraceUpdateArgs(ShardedPhysicalTemplate *target, 
                             UpdateKind kind, RtUserEvent done, 
                             const UniqueInst &inst, Deserializer &derez,
                             const PendingRemoteExpression &pending);
        DeferTraceUpdateArgs(const DeferTraceUpdateArgs &args,
                             RtUserEvent deferral,IndexSpaceExpression *expr);
      public:
        ShardedPhysicalTemplate *const target;
        const UpdateKind kind;
        const RtUserEvent done;
        const UniqueInst inst;
        IndexSpaceExpression *const expr;
        const PendingRemoteExpression pending;
        const size_t buffer_size;
        void *const buffer;
        const RtUserEvent deferral_event;
      };
    public:
      ShardedPhysicalTemplate(PhysicalTrace *trace, ApEvent fence_event,
                              TaskTreeCoordinates &&coordinates,
                              ReplicateContext *repl_ctx);
      ShardedPhysicalTemplate(const ShardedPhysicalTemplate &rhs) = delete;
      virtual ~ShardedPhysicalTemplate(void);
    public:
      // Have to provide explicit overrides of operator new and 
      // delete here to make sure we get the right ones. C++ does
      // not let us have these in a sub-class or it doesn't know
      // which ones to pick from.
      static inline void* operator new(size_t count)
      { return legion_alloc_aligned<ShardedPhysicalTemplate,true>(count); }
      static inline void operator delete(void *ptr)
      { free(ptr); }
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        RtEvent continuation_pre;
        continuation_pre.id = 
          next_deferral_precondition.exchange(deferral_event.id);
        return continuation_pre;
      }
    public:
      virtual size_t get_sharded_template_index(void) const
        { return template_index; }
      virtual void initialize_replay(ApEvent fence_completion, bool recurrent,
                                     bool need_lock = true);
      virtual void perform_replay(Runtime *runtime, 
                                  std::set<RtEvent> &replayed_events);
      virtual RtEvent refresh_managed_barriers(void);
      virtual void finish_replay(std::set<ApEvent> &postconditions);
      virtual ApEvent get_completion_for_deletion(void) const;
      using PhysicalTemplate::record_merge_events;
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::set<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::vector<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_collective_barrier(ApBarrier bar, ApEvent pre,
                    const std::pair<size_t,size_t> &key, size_t arrival_count);
      virtual ShardID record_managed_barrier(ApBarrier bar,
                                             size_t total_arrivals);
      virtual void record_barrier_arrival(ApBarrier bar, ApEvent pre,
                    size_t arrival_count, std::set<RtEvent> &applied,
                    ShardID owner_shard);
      virtual void record_issue_copy(const TraceLocalID &tlid, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField>& src_fields,
                             const std::vector<CopySrcDstField>& dst_fields,
                             const std::vector<Reservation>& reservations,
#ifdef LEGION_SPY
                             RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
#endif
                             ApEvent precondition, PredEvent guard_event,
                             LgEvent src_unique, LgEvent dst_unique,
                             int priority, CollectiveKind collective);
      virtual void record_issue_fill(const TraceLocalID &tlid, ApEvent &lhs,
                             IndexSpaceExpression *expr,
                             const std::vector<CopySrcDstField> &fields,
                             const void *fill_value, size_t fill_size,
#ifdef LEGION_SPY
                             UniqueID fill_uid,
                             FieldSpace handle,
                             RegionTreeID tree_id,
#endif
                             ApEvent precondition, PredEvent guard_event,
                             LgEvent unique_event, int priority, 
                             CollectiveKind collective);
      virtual void record_issue_across(const TraceLocalID &tlid, ApEvent &lhs,
                             ApEvent collective_precondition,
                             ApEvent copy_precondition,
                             ApEvent src_indirect_precondition,
                             ApEvent dst_indirect_precondition,
                             CopyAcrossExecutor *executor);
    public:
      virtual void record_owner_shard(unsigned trace_local_id, ShardID owner);
      virtual void record_local_space(unsigned trace_local_id, IndexSpace sp);
      virtual void record_sharding_function(unsigned trace_local_id, 
                                            ShardingFunction *function);
      virtual void issue_summary_operations(InnerContext *context,
                                            Operation *invalidator,
                                            Provenance *provenance);
      virtual void dump_sharded_template(void);
    public:
      virtual ShardID find_owner_shard(unsigned trace_local_id);
      virtual IndexSpace find_local_space(unsigned trace_local_id);
      virtual ShardingFunction* find_sharding_function(unsigned trace_local_id);
    public:
      void prepare_collective_barrier_replay(
                            const std::pair<size_t,size_t> &key, ApBarrier bar);
    public:
      ApBarrier find_trace_shard_event(ApEvent event, ShardID remote_shard);
      void record_trace_shard_event(ApEvent event, ApBarrier result);
      ApBarrier find_trace_shard_frontier(ApEvent event, ShardID remote_shard);
      void record_trace_shard_frontier(unsigned frontier, ApBarrier result);
      void handle_trace_update(Deserializer &derez, AddressSpaceID source);
      static void handle_deferred_trace_update(const void *args, Runtime *rt);
    protected:
      bool handle_update_mutated_inst(const UniqueInst &inst, 
                            IndexSpaceExpression *ex, Deserializer &derez, 
                            std::set<RtEvent> &applied, RtUserEvent done, 
                            const DeferTraceUpdateArgs *dargs = NULL);
    protected:
#ifdef DEBUG_LEGION
      virtual unsigned convert_event(const ApEvent &event, bool check = true);
#endif
      virtual unsigned find_event(const ApEvent &event, AutoLock &tpl_lock);
      void request_remote_shard_event(ApEvent event, RtUserEvent done_event);
      static AddressSpaceID find_event_space(ApEvent event);
      virtual Replayable check_replayable(Operation *op,
          InnerContext *context, bool has_blocking_call);
    protected:
      ShardID find_inst_owner(const UniqueInst &inst);
      void find_owner_shards(AddressSpace owner, std::vector<ShardID> &shards);
    protected:
      virtual unsigned find_frontier_event(ApEvent event,
                        std::vector<RtEvent> &ready_events);
      virtual void record_mutated_instance(const UniqueInst &inst,
                                           IndexSpaceExpression *expr,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &applied_events);
      virtual bool are_read_only_users(InstUsers &inst_users);
      virtual void sync_compute_frontiers(Operation *op,
                          const std::vector<RtEvent> &frontier_events);
      virtual void initialize_generators(std::vector<unsigned> &new_gen);
      virtual void initialize_eliminate_dead_code_frontiers(
                          const std::vector<unsigned> &gen,
                                std::vector<bool> &used);
      virtual void initialize_transitive_reduction_frontiers(
                          std::vector<unsigned> &topo_order,
                          std::vector<unsigned> &inv_topo_order);
      virtual void record_used_frontiers(std::vector<bool> &used,
                      const std::vector<unsigned> &gen) const;
      virtual void rewrite_frontiers(
                      std::map<unsigned,unsigned> &substitutions);
    public:
      ReplicateContext *const repl_ctx;
      const ShardID local_shard;
      const size_t total_shards;
      // Make this last since it registers the template with the
      // context which can trigger calls into the template so 
      // everything must valid at this point
      const size_t template_index; 
    protected:
      std::map<ApEvent,RtEvent> pending_event_requests;
      // Barriers we don't managed and need to receive refreshes for
      std::map<ApEvent,BarrierAdvance*> local_advances;
      // Collective barriers from application operations
      // These will be updated by the application before each replay
      // Key is <trace local id, unique barrier name for this op>
      std::map<std::pair<size_t,size_t>,BarrierArrival*> collective_barriers;
      // Buffer up barrier updates as we're running ahead so that we can
      // apply them before we perform the trace replay
      std::deque<
            std::map<std::pair<size_t,size_t>,ApBarrier> > pending_collectives;
      std::map<AddressSpaceID,std::vector<ShardID> > did_shard_owners;
      std::map<unsigned/*Trace Local ID*/,ShardID> owner_shards;
      std::map<unsigned/*Trace Local ID*/,IndexSpace> local_spaces;
      std::map<unsigned/*Trace Local ID*/,ShardingFunction*> sharding_functions;
    protected:
      // Count how many refereshed barriers we've seen updated for when
      // we need to reset the phase barriers for a new round of generations
      size_t refreshed_barriers;
      // An event to signal when our advances are ready
      RtUserEvent update_advances_ready;
      // An event for chainging deferrals of update tasks
      std::atomic<Realm::Event::id_t> next_deferral_precondition;
    protected:
      // Count how many times we've done recurrent replay so we know when we're
      // going to run out of phase barrier generations
      size_t recurrent_replays;
      // Count how many frontiers ahave been updated so that we know when
      // they are done being updated
      size_t updated_frontiers;
      // An event to signal when our frontiers are ready
      RtUserEvent update_frontiers_ready;
    protected:
      // Data structures for fence elision
      // Local frontiers records barriers that should be arrived on 
      // based on events that we have here locally
      std::map<unsigned,ApBarrier> local_frontiers;
      // Remote shards that are subscribed to our local frontiers
      std::map<unsigned,std::set<ShardID> > local_subscriptions;
      // Remote frontiers records barriers that we should fill in as
      // events from remote shards
      std::vector<std::pair<ApBarrier,unsigned> > remote_frontiers;
      // Pending refreshes from remote nodes
      std::map<ApBarrier,ApBarrier> pending_refresh_frontiers;
      std::map<ApEvent,ApBarrier> pending_refresh_barriers;
    };

    enum InstructionKind
    {
      GET_TERM_EVENT = 0,
      REPLAY_MAPPING,
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
      BARRIER_ARRIVAL,
      BARRIER_ADVANCE,
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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay) = 0;
      typedef std::map<TraceLocalID,std::pair<unsigned,unsigned> > MemoEntries;
      virtual std::string to_string(const MemoEntries &memo_entires) = 0;

      virtual InstructionKind get_kind(void) = 0;
      virtual GetTermEvent* as_get_term_event(void) { return NULL; }
      virtual ReplayMapping* as_replay_mapping(void) { return NULL; }
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
      virtual BarrierArrival* as_barrier_arrival(void) { return NULL; }
      virtual BarrierAdvance* as_barrier_advance(void) { return NULL; }
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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return GET_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      friend class ShardedPhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class ReplayMapping
     * This instruction has the following semantics:
     *   events[lhs] = operations[owner].replay_mapping()
     */
    class ReplayMapping : public Instruction {
    public:
      ReplayMapping(PhysicalTemplate& tpl, unsigned lhs,
                    const TraceLocalID& rhs);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return REPLAY_MAPPING; }
      virtual ReplayMapping* as_replay_mapping(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      friend class ShardedPhysicalTemplate;
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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return MERGE_EVENT; }
      virtual MergeEvent* as_merge_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      friend class ShardedPhysicalTemplate;
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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
                UniqueID fill_uid, FieldSpace handle, RegionTreeID tree_id,
#endif
                unsigned precondition_idx, LgEvent unique_event,
                int priority, CollectiveKind collective);
      virtual ~IssueFill(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
      UniqueID fill_uid;
      FieldSpace handle;
      RegionTreeID tree_id;
#endif
      unsigned precondition_idx;
      LgEvent unique_event;
      int priority;
      CollectiveKind collective;
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
                unsigned precondition_idx,
                LgEvent src_unique, LgEvent dst_unique,
                int priority, CollectiveKind collective);
      virtual ~IssueCopy(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
      LgEvent src_unique, dst_unique;
      int priority;
      CollectiveKind collective;
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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

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
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return SET_OP_SYNC_EVENT; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
    };

    /**
     * \class CompleteReplay
     * This instruction has the following semantics:
     *   operations[lhs]->complete_replay(events[rhs])
     */
    class CompleteReplay : public Instruction {
    public:
      CompleteReplay(PhysicalTemplate& tpl, const TraceLocalID& lhs,
                     unsigned pre, unsigned post);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return COMPLETE_REPLAY; }
      virtual CompleteReplay* as_complete_replay(void)
        { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned pre, post;
    };

    /**
     * \class BarrierArrival
     * This instruction has the following semantics:
     * events[lhs] = barrier.arrive(events[rhs])
     */
    class BarrierArrival : public Instruction {
    public:
      BarrierArrival(PhysicalTemplate &tpl,
                     ApBarrier bar, unsigned lhs, unsigned rhs,
                     size_t arrival_count, bool managed);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return BARRIER_ARRIVAL; }
      virtual BarrierArrival* as_barrier_arrival(void)
        { return this; }
      void set_managed_barrier(ApBarrier newbar);
      void set_collective_barrier(ApBarrier newbar);
    private:
      friend class PhysicalTemplate;
      ApBarrier barrier;
      unsigned lhs, rhs;
      const size_t total_arrivals;
      const bool managed;
    };

    /**
     * \class BarrierAdvance
     * This instruction has the following semantics
     * events[lhs] = barrier
     * barrier.advance();
     */
    class BarrierAdvance : public Instruction {
    public:
      BarrierAdvance(PhysicalTemplate &tpl, ApBarrier bar,
                     unsigned lhs, size_t arrival_count, bool owner);
      virtual ~BarrierAdvance(void);
      virtual void execute(std::vector<ApEvent> &events,
                           std::map<unsigned,ApUserEvent> &user_events,
                           std::map<TraceLocalID,MemoizableOp*> &operations,
                           const bool recurrent_replay);
      virtual std::string to_string(const MemoEntries &memo_entires);

      virtual InstructionKind get_kind(void)
        { return BARRIER_ADVANCE; }
      virtual BarrierAdvance* as_barrier_advance(void)
        { return this; }
      inline ApBarrier get_current_barrier(void) const { return barrier; }
      ApBarrier record_subscribed_shard(ShardID remote_shard); 
      void refresh_barrier(ApEvent key,
          std::map<ShardID,std::map<ApEvent,ApBarrier> > &notifications);
      void remote_refresh_barrier(ApBarrier newbar);
    private:
      friend class PhysicalTemplate;
      ApBarrier barrier;
      std::vector<ShardID> subscribed_shards;
      unsigned lhs;
      const size_t total_arrivals;
      const bool owner;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TRACE__
