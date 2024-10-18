/* Copyright 2024 Stanford University, NVIDIA Corporation
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
      struct DependenceRecord {
      public:
        DependenceRecord(int idx)
          : operation_idx(idx), prev_idx(-1), next_idx(-1),
            dtype(LEGION_TRUE_DEPENDENCE) { }
        DependenceRecord(int op_idx, int pidx, int nidx,
                         DependenceType d, const FieldMask &m)
          : operation_idx(op_idx), prev_idx(pidx), 
            next_idx(nidx),
            dtype(d), dependent_mask(m) { }
      public:
        inline bool merge(const DependenceRecord &record)
        {
          if ((operation_idx != record.operation_idx) ||
              (prev_idx != record.prev_idx) ||
              (next_idx != record.next_idx) ||
              (dtype != record.dtype))
            return false;
          dependent_mask |= record.dependent_mask;
          return true;
        }
      public:
        int operation_idx;
        int prev_idx; // previous region requirement index
        int next_idx; // next region requirement index
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

#ifdef POINT_WISE_LOGICAL_ANALYSIS
      struct TracePointWisePreviousIndexTaskInfo : public LegionHeapify<TracePointWisePreviousIndexTaskInfo> {
      public:
        TracePointWisePreviousIndexTaskInfo(ProjectionSummary *shard_proj,
            Domain &index_domain,
            TraceLocalID prev_op_trace_idx, unsigned op_idx,
            GenerationID prev_op_gen, size_t ctx_index, unsigned dep_type,
            unsigned region_idx)
          : shard_proj(shard_proj),
            index_domain(index_domain),
          prev_op_trace_idx(prev_op_trace_idx), op_idx(op_idx),
          prev_op_gen(prev_op_gen),
          ctx_index(ctx_index), dep_type(dep_type), region_idx(region_idx)
      {}
      public:
        ProjectionSummary *shard_proj;
        Domain index_domain;
        TraceLocalID prev_op_trace_idx;
        unsigned op_idx;
        GenerationID prev_op_gen;
        size_t ctx_index;
        unsigned dep_type;
        unsigned region_idx;
      };
#endif

      struct OperationInfo {
      public:
        LegionVector<DependenceRecord> dependences;
        LegionVector<CloseInfo> closes;
#ifdef POINT_WISE_LOGICAL_ANALYSIS
        std::map<unsigned,bool> connect_to_next_points;
        std::map<unsigned,bool> connect_to_prev_points;
        std::map<unsigned,TracePointWisePreviousIndexTaskInfo> prev_ops;
#endif
      };
      struct VerificationInfo {
      public:
        VerificationInfo(Operation::OpKind k, TaskID tid,
            unsigned r, uint64_t h[2])
          : kind(k), task_id(tid), regions(r)
        { hash[0] = h[0]; hash[1] = h[1]; }
      public:
        Operation::OpKind kind;
        TaskID task_id;
        unsigned regions;
        uint64_t hash[2];
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
      inline size_t get_operation_count(void) const 
        { return replay_info.size(); }
    public:
      bool initialize_op_tracing(Operation *op,
                     const std::vector<StaticDependence> *dependences = NULL);
      void check_operation_count(void);
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
                                    DependenceType dtype,
                                    const FieldMask &dependent_mask);
#ifdef POINT_WISE_LOGICAL_ANALYSIS
      void set_next_point_wise_user(Operation *next_op,
          GenerationID next_gen, GenerationID source_gen,
          unsigned region_idx, Operation *source);
      void set_prev_point_wise_user(Operation *prev_op,
          GenerationID prev_gen, uint64_t prev_ctx_index,
          ProjectionSummary *shard_proj,
          unsigned region_idx, unsigned dep_type, unsigned prev_region_idx,
          Operation *source);
      void set_point_wise_dependences(size_t index, Operation *op);
#endif
    public:
      // Called by task execution thread
      inline bool is_fixed(void) const { return fixed; }
      void fix_trace(Provenance *provenance);
      inline void record_blocking_call(void) { blocking_call_observed = true; }
      inline bool get_and_clear_blocking_call(void)
        {
          const bool result = blocking_call_observed; 
          blocking_call_observed = false;
          return result;
        }
      inline void record_intermediate_fence(void)
        { intermediate_fence = true; }
      inline bool has_intermediate_fence(void) const
        { return intermediate_fence; }
      inline void reset_intermediate_fence(void)
        { intermediate_fence = false; }
    public:
      // Called during logical dependence analysis stage
      inline bool is_recording(void) const { return recording; }
      void begin_logical_trace(FenceOp *fence_op);
      void end_logical_trace(FenceOp *fence_op);
    public:
      bool has_physical_trace(void) const { return (physical_trace != NULL); }
      PhysicalTrace* get_physical_trace(void) { return physical_trace; }
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
      // Pointer to a physical trace
      PhysicalTrace *const physical_trace;
    protected:
      // Application stage of the pipeline
      std::vector<VerificationInfo> verification_infos;
      unsigned verification_index;
      bool blocking_call_observed;
      bool fixed;
      bool intermediate_fence;
    protected:
      struct OpInfo {
        Operation* op;
        GenerationID gen;
        uint64_t context_index;
      };
      // Logical dependence analysis stage of the pipeline
      bool recording;
      size_t replay_index;
      std::deque<OperationInfo> replay_info;
      std::set<std::pair<Operation*,GenerationID> > frontiers;
      //std::vector<std::pair<Operation*,GenerationID> > operations;
      std::vector<OpInfo > operations;
      // Only need this backwards lookup for trace capture
      //std::map<std::pair<Operation*,GenerationID>,unsigned> op_map;
      std::map<std::pair<Operation*,GenerationID>,
        std::pair<unsigned,unsigned>> op_map;
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
      virtual void pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                     std::set<RtEvent> &applied) const override;
    };

#if 0
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
#endif

    enum ReplayableStatus {
      REPLAYABLE,
      NOT_REPLAYABLE_BLOCKING,
      NOT_REPLAYABLE_CONSENSUS,
      NOT_REPLAYABLE_VIRTUAL,
      NOT_REPLAYABLE_REMOTE_SHARD,
    };

    enum IdempotencyStatus {
      IDEMPOTENT,
      NOT_IDEMPOTENT_SUBSUMPTION,
      NOT_IDEMPOTENT_ANTIDEPENDENT,
      NOT_IDEMPOTENT_REMOTE_SHARD,
    };

    /**
     * \class CompleteOp
     * A pure virtual interface for completion ops to implement
     * to help with completing captures and replays of templates
     */
    class CompleteOp {
    public:
      virtual FenceOp* get_complete_operation(void) = 0;
      virtual void begin_replayable_exchange(ReplayableStatus status) { }
      virtual void end_replayable_exchange(ReplayableStatus &status) { }
      virtual void begin_idempotent_exchange(IdempotencyStatus idempotent) { }
      virtual void end_idempotent_exchange(IdempotencyStatus &idempotent) { }
      virtual void sync_compute_frontiers(RtEvent event) { assert(false); }
      virtual void deduplicate_condition_sets(
          std::map<EquivalenceSet*,unsigned> &condition_sets) { }
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public TraceOp, public CompleteOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      TraceCompleteOp(Runtime *rt);
      TraceCompleteOp(const TraceCompleteOp &rhs);
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs);
    public:
      void initialize_complete(InnerContext *ctx, LogicalTrace *trace,
                               Provenance *provenance, bool remove_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void); 
    protected:
      virtual FenceOp* get_complete_operation(void) { return this; }
    protected:
      bool has_blocking_call;
      bool remove_trace_reference;
    };

#if 0
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
#endif

    /**
     * \class BeginOp
     * This is a pure virtual interface for operations that need to be 
     * performed across any kind of begin operation for tracing
     */
    class BeginOp {
    public:
      virtual bool allreduce_template_status(bool &valid, bool acquired)
      {
        if (acquired)
          return false;
        valid = false;
        return true;
      }
      virtual ApEvent get_begin_completion(void) = 0;
      virtual FenceOp* get_begin_operation(void) = 0;
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace *trace) = 0;
    };

    /**
     * \class TraceBeginOp
     * This class represents mapping fences which we inject
     * into the operation stream to begin a trace.  This fence
     * is by a TraceReplayOp if the trace allows physical tracing.
     */
    class TraceBeginOp : public TraceOp, public BeginOp {
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
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    public:
      virtual ApEvent get_begin_completion(void) 
        { return get_completion_event(); }
      virtual FenceOp* get_begin_operation(void) { return this; }
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace *trace);
    };

    /**
     * \class RecurrentOp
     * A recurrent op supports both the begin and complete interfaces
     */
    class RecurrentOp : public BeginOp, public CompleteOp { };

    /**
     * \class TraceRecurrentOp
     * This is a tracing operation that is inserted to invalidate an idempotent
     * trace replay once an invalidating operation is detected in the stream
     * of operations in the parent context. We make this a mapping fence so
     * we ensure that the resources from the template are freed up before
     * any other downstream operations attempt to map.
     */
    class TraceRecurrentOp : public TraceOp, public RecurrentOp {
    public:
      static const AllocationType alloc_type = TRACE_RECURRENT_OP_ALLOC;
    public:
      TraceRecurrentOp(Runtime *rt);
      TraceRecurrentOp(const TraceRecurrentOp &rhs) = delete;
      virtual ~TraceRecurrentOp(void);
    public:
      TraceRecurrentOp& operator=(const TraceRecurrentOp &rhs) = delete;
    public:
      void initialize_recurrent(InnerContext *ctx, LogicalTrace *trace,
         LogicalTrace *previous, Provenance *provenance, bool remove_reference);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_ready(void);
      virtual void trigger_mapping(void);
    public:
      virtual FenceOp* get_begin_operation(void) { return this; }
      virtual ApEvent get_begin_completion(void) 
        { return get_completion_event(); }
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace *trace);
    public:
      virtual FenceOp* get_complete_operation(void) { return this; }
    protected:
      LogicalTrace *previous;
      bool has_blocking_call;
      bool has_intermediate_fence;
      bool remove_trace_reference;
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
#if 0
    public:
      // Return true if we evaluated all the templates
      bool find_viable_templates(ReplTraceReplayOp *op, 
                                 std::set<RtEvent> &applied_events,
                                 unsigned templates_to_find,
                                 std::vector<int> &viable_templates);
      void select_template(unsigned template_index);
      ApEvent record_capture(PhysicalTemplate *tpl,
                    std::set<RtEvent> &map_applied_conditions);
      void record_failed_capture(PhysicalTemplate *tpl);
#endif
    public:
      inline bool has_current_template(void) const
        { return (current_template != NULL); }
      inline PhysicalTemplate* get_current_template(void) const
        { return current_template; }
      inline const std::vector<Processor> &get_replay_targets(void)
        { return replay_targets; }
      inline bool is_recording(void) const { return recording; }
      inline bool is_replaying(void) const { return !recording; }
      inline bool is_recurrent(void) const { return recurrent; }
      size_t get_expected_operation_count(void) const;
    public:
      void record_parent_req_fields(unsigned index, const FieldMask &mask);
      void find_condition_sets(std::map<EquivalenceSet*,unsigned> &sets) const;
      void refresh_condition_sets(FenceOp *op,
          std::set<RtEvent> &refresh_ready) const;
      bool begin_physical_trace(BeginOp *op,
          std::set<RtEvent> &map_applied_conditions,
          std::set<ApEvent> &execution_preconditions);
      void complete_physical_trace(CompleteOp *op,
          std::set<RtEvent> &map_applied_conditions,
          std::set<ApEvent> &execution_preconditions,
          bool has_blocking_call);
      bool replay_physical_trace(RecurrentOp *op,
          std::set<RtEvent> &map_applied_events,
          std::set<ApEvent> &execution_preconditions,
          bool has_blocking_call, bool has_intermediate_fence);
    protected:
      bool find_replay_template(BeginOp *op,
            std::set<RtEvent> &map_applied_conditions,
            std::set<ApEvent> &execution_preconditions);
      void begin_replay(BeginOp *op, bool recurrent,
                        bool has_intermediate_fence);
      bool complete_recording(CompleteOp *op,
          std::set<RtEvent> &map_applied_conditions,
          std::set<ApEvent> &execution_preconditions, bool has_blocking_call);
      void complete_replay(std::set<ApEvent> &completion_events);
    public:
      Runtime *const runtime;
      const LogicalTrace *const logical_trace;
      const bool perform_fence_elision;
    private:
      mutable LocalLock trace_lock;
      // This is a mapping from the parent region requirements
      // to the sets of fields referred to in the trace. We use
      // this to find the equivalence sets for a template
      LegionMap<unsigned,FieldMask> parent_req_fields;
      std::vector<PhysicalTemplate*> templates;
      PhysicalTemplate* current_template;
      unsigned nonreplayable_count;
      unsigned new_template_count; 
    private:
      std::vector<Processor> replay_targets;
      bool recording;
      bool recurrent;
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
      public:
        FailedPrecondition(void) : view(NULL), expr(NULL) { }
      public:
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
      void insert(LegionMap<LogicalView*,
                    FieldMaskSet<IndexSpaceExpression> > &views,
                    bool antialiased = false);
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
                     LegionMap<LogicalView*,
                     FieldMaskSet<IndexSpaceExpression> > &non_dominated) const;
      void filter_independent_fields(IndexSpaceExpression *expr,
                                     FieldMask &mask) const;
      bool subsumed_by(const TraceViewSet &set, bool allow_independent,
                       FailedPrecondition *condition = NULL) const;
      bool independent_of(const TraceViewSet &set,
                       FailedPrecondition *condition = NULL) const; 
      void record_first_failed(FailedPrecondition *condition = NULL) const;
      void transpose_uniquely(LegionMap<IndexSpaceExpression*,
                                  FieldMaskSet<LogicalView> > &target) const;
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
      TraceConditionSet(PhysicalTemplate *tpl, unsigned parent_req_index,
                        RegionTreeID tree_id, IndexSpaceExpression *expr,
                        FieldMaskSet<LogicalView> &&views);
      TraceConditionSet(const TraceConditionSet &rhs) = delete;
      virtual ~TraceConditionSet(void);
    public:
      TraceConditionSet& operator=(const TraceConditionSet &rhs) = delete;
    public:
      inline bool is_shared(void) const { return shared; }
      inline void mark_shared(void) { shared = true; }
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
      bool matches(IndexSpaceExpression *expr,
                   const FieldMaskSet<LogicalView> &views) const;
      void invalidate_equivalence_sets(void);
      void refresh_equivalence_sets(FenceOp *op,
          std::set<RtEvent> &ready_events);
      void dump_conditions(void) const;
    public:
      void test_preconditions(FenceOp *op, unsigned index,
                              std::vector<RtEvent> &ready_events,
                              std::set<RtEvent> &applied_events);
      bool check_preconditions(void);
      void test_anticonditions(FenceOp *op, unsigned index,
                               std::vector<RtEvent> &ready_events,
                               std::set<RtEvent> &applied_events);
      bool check_anticonditions(void);
      void apply_postconditions(FenceOp *op, unsigned index,
                                std::set<RtEvent> &applied_events);
    public:
      PhysicalTemplate *const owner;
      IndexSpaceExpression *const condition_expr;
      const FieldMaskSet<LogicalView> views;
      const RegionTreeID tree_id;
      const unsigned parent_req_index;
    private:
      mutable LocalLock set_lock;
    private:
      union {
        InvalidInstAnalysis *invalid;
        AntivalidInstAnalysis *antivalid;
      } analysis;
      bool shared;
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
      PhysicalTemplate(PhysicalTrace *trace, ApEvent fence_event);
      PhysicalTemplate(const PhysicalTemplate &rhs) = delete;
      virtual ~PhysicalTemplate(void);
    public:
      PhysicalTemplate& operator=(const PhysicalTemplate &rhs) = delete;
    public:
      virtual size_t get_sharded_template_index(void) const { return 0; }
      virtual void initialize_replay(ApEvent fence_completion, bool recurrent);
      virtual void start_replay(void);
      virtual RtEvent refresh_managed_barriers(void);
      virtual void finish_replay(std::set<ApEvent> &postconditions);
      virtual ApEvent get_completion_for_deletion(void) const;
    public:
      ReplayableStatus finalize(CompleteOp *op, bool has_blocking_call);
      IdempotencyStatus capture_conditions(CompleteOp *op);
      void receive_trace_conditions(TraceViewSet *preconditions,
          TraceViewSet *anticonditions, TraceViewSet *postconditions,
          unsigned parent_req_index, RegionTreeID tree_id,
          std::atomic<unsigned> *result);
      void refresh_condition_sets(FenceOp *op,
          std::set<RtEvent> &ready_events) const;
      bool acquire_instance_references(void) const;
      void release_instance_references(void) const;
    public:
      void optimize(CompleteOp *op, bool do_transitive_reduction);
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
      void check_finalize_transitive_reduction(void);
      void propagate_copies(std::vector<unsigned> *gen);
      void eliminate_dead_code(std::vector<unsigned> &gen);
      void prepare_parallel_replay(const std::vector<unsigned> &gen);
      void push_complete_replays(void);
    protected:
      virtual void sync_compute_frontiers(CompleteOp *op,
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
#if 0
      // Variant for normal traces
      bool check_preconditions(TraceReplayOp *op,
                               std::set<RtEvent> &applied_events);
      // Variant for control replication traces
      bool check_preconditions(ReplTraceReplayOp *op,
                               std::set<RtEvent> &applied_events);
#endif
      RtEvent test_preconditions(FenceOp *op,
                                 std::set<RtEvent> &applied_events);
      bool check_preconditions(void);
      void apply_postconditions(FenceOp *op,
                                std::set<RtEvent> &applied_events);
    public:
      bool can_start_replay(void);
      void register_operation(MemoizableOp *op);
      void execute_slice(unsigned slice_idx, bool recurrent_replay);
    public:
      void dump_template(void) const;
      virtual void dump_sharded_template(void) const { }
    private:
      void dump_instructions(
          const std::vector<Instruction*> &instructions) const;
#ifdef LEGION_SPY
    public:
      void set_fence_uid(UniqueID fence_uid) { prev_fence_uid = fence_uid; }
      UniqueID get_fence_uid(void) const { return prev_fence_uid; }
#endif
    public:
      inline bool is_replaying(void) const { return trace->is_replaying(); }
      inline bool is_replayable(void) const
        { return (replayable == REPLAYABLE); }
      inline bool is_idempotent(void) const 
        { return (idempotency == IDEMPOTENT); }
      inline void record_no_consensus(void) { has_no_consensus.store(true); }
    public:
      virtual bool is_recording(void) const { return trace->is_recording(); }
      virtual void add_recorder_reference(void) { /*do nothing*/ }
      virtual bool remove_recorder_reference(void) 
        { /*do nothing, never delete*/ return false; }
      virtual void pack_recorder(Serializer &rez); 
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
      RtBarrier get_concurrent_barrier(IndexTask *task);
      const std::vector<ShardID>& get_concurrent_shards(ReplIndexTask* task);
    public:
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
      virtual ShardID record_barrier_creation(ApBarrier &bar,
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
                             int priority, CollectiveKind collective,
                             bool record_effect);
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
                             CollectiveKind collective, bool record_effect);
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
                                          ApEvent pre,
                                          std::set<RtEvent> &applied_events);
      virtual void record_reservations(const TraceLocalID &tlid,
                                const std::map<Reservation,bool> &locks,
                                std::set<RtEvent> &applied_events); 
      virtual void record_future_allreduce(const TraceLocalID &tlid,
          const std::vector<Memory> &target_memories, size_t future_size);
      virtual void record_concurrent_barrier(IndexTask *task, RtBarrier bar,
          const std::vector<ShardID> &shards, size_t participants);
      void record_execution_fence(const TraceLocalID &tlid);
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
      PhysicalTrace * const trace;
    protected:
      // Count how many times we've been replayed so we know when we're going
      // to run out of phase barrier generations
      // Note we start this at 1 since some barriers are used as part of the
      // capture, while others are not used until the first replay, that throws
      // away one barrier generation on some barriers, but whatever
      size_t total_replays;
      ReplayableStatus replayable;
      IdempotencyStatus idempotency;
    protected:
      mutable LocalLock template_lock;
      const unsigned fence_completion_id;
    protected:
      static constexpr unsigned NO_INDEX = UINT_MAX;
    protected:
      std::map<TraceLocalID,MemoizableOp*> operations;
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
      std::atomic<bool> has_no_consensus;
      mutable TraceViewSet::FailedPrecondition failure;
    protected:
      CompleteReplay                  *last_fence;
    protected:
      RtEvent                         replay_precondition;
      RtUserEvent                     replay_postcondition;
      std::atomic<unsigned>           remaining_replays;
      std::atomic<unsigned>           total_logical;
      std::vector<ApEvent>            events;
      std::map<unsigned,ApUserEvent>  user_events;
    protected:
      std::map<ApEvent,unsigned> event_map;
      std::map<ApEvent,BarrierAdvance*> managed_barriers;
      std::map<ApEvent,std::vector<BarrierArrival*> > managed_arrivals;
      struct ConcurrentBarrier {
        std::vector<ShardID> shards;
        RtBarrier barrier;
        size_t participants;
      };
      std::map<TraceLocalID,ConcurrentBarrier> concurrent_barriers;
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
      std::vector<PhysicalManager*>    all_instances;
    protected:
      // Capture the names of all the instances that are mutated by this trace
      // and the index space expressions and fields that were mutated
      // THIS IS SHARDED FOR CONTROL REPLICATION!!!
      LegionMap<UniqueInst,FieldMaskSet<IndexSpaceExpression> > mutated_insts; 
    private:
      // THESE ARE SHARDED FOR CONTROL REPLICATION!!!
      // Each share has a disjoint set of trace conditions that they are
      // responsible for handling checking
      std::vector<TraceConditionSet*> preconditions; 
      std::vector<TraceConditionSet*> anticonditions;
      std::vector<TraceConditionSet*> postconditions;
#ifdef LEGION_SPY
    private:
      UniqueID prev_fence_uid;
#endif
    private:
      friend class PhysicalTrace;
      friend class Instruction;
#ifdef DEBUG_LEGION
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
      virtual void pack_recorder(Serializer &rez);
      virtual size_t get_sharded_template_index(void) const
        { return template_index; }
      virtual void initialize_replay(ApEvent fence_completion, bool recurrent);
      virtual void start_replay(void);
      virtual RtEvent refresh_managed_barriers(void);
      virtual void finish_replay(std::set<ApEvent> &postconditions);
      virtual ApEvent get_completion_for_deletion(void) const;
      virtual void record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                        const TraceLocalID &tlid);
      using PhysicalTemplate::record_merge_events;
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::set<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_merge_events(ApEvent &lhs, 
                                       const std::vector<ApEvent>& rhs,
                                       const TraceLocalID &tlid);
      virtual void record_collective_barrier(ApBarrier bar, ApEvent pre,
                    const std::pair<size_t,size_t> &key, size_t arrival_count);
      virtual ShardID record_barrier_creation(ApBarrier &bar,
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
                             int priority, CollectiveKind collective,
                             bool record_effect);
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
                             CollectiveKind collective, bool record_effect);
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
      virtual void dump_sharded_template(void) const;
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
      bool record_shard_event_trigger(ApUserEvent lhs, ApEvent rhs,
                                      const TraceLocalID &tlid);
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
      virtual void sync_compute_frontiers(CompleteOp *op,
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
      std::map<std::pair<size_t,size_t>,ApBarrier> pending_collectives;
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
      std::map<TraceLocalID,RtBarrier> pending_concurrent_barriers;
    };

    enum InstructionKind
    {
      REPLAY_MAPPING = 0,
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
                int priority, CollectiveKind collective, bool record_effect);
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
      bool record_effect;
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
                int priority, CollectiveKind collective, bool record_effect);
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
      bool record_effect;
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
     *   operations[lhs]->complete_replay(events[complete])
     */
    class CompleteReplay : public Instruction {
    public:
      CompleteReplay(PhysicalTemplate& tpl, const TraceLocalID& lhs,
                     unsigned complete);
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
      unsigned complete;
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
