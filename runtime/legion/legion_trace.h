/* Copyright 2017 Stanford University, NVIDIA Corporation
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
#include "runtime.h"
#include "legion_ops.h"

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
            validates(false), dtype(TRUE_DEPENDENCE) { }
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
      LegionTrace(TaskContext *ctx);
      virtual ~LegionTrace(void);
    public:
      virtual bool is_static_trace(void) const = 0;
      virtual bool is_dynamic_trace(void) const = 0;
      virtual StaticTrace* as_static_trace(void) = 0;
      virtual DynamicTrace* as_dynamic_trace(void) = 0;
    public:
      virtual bool is_fixed(void) const = 0;
      virtual bool handles_region_tree(RegionTreeID tid) const = 0;
      virtual void record_static_dependences(Operation *op,
                     const std::vector<StaticDependence> *dependences) = 0;
      virtual void register_operation(Operation *op, GenerationID gen) = 0; 
      virtual void record_dependence(Operation *target, GenerationID target_gen,
                                Operation *source, GenerationID source_gen) = 0;
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask) = 0;
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask) = 0;
    public:
      bool has_physical_trace() { return physical_trace != NULL; }
      PhysicalTrace* get_physical_trace() { return physical_trace; }
      void register_physical_only(Operation *op, GenerationID gen);
    public:
      void replay_aliased_children(std::vector<RegionTreePath> &paths) const;
      void end_trace_execution(FenceOp *fence_op);
    public:
      TaskContext *const ctx;
    protected:
      std::vector<std::pair<Operation*,GenerationID> > operations; 
      // We also need a data structure to record when there are
      // aliased but non-interfering region requirements. This should
      // be pretty sparse so we'll make it a map
      std::map<unsigned,LegionVector<AliasChildren>::aligned> aliased_children;
      // Pointer to a physical trace
      PhysicalTrace *physical_trace;
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
      StaticTrace(TaskContext *ctx, const std::set<RegionTreeID> *trees);
      StaticTrace(const StaticTrace &rhs);
      virtual ~StaticTrace(void);
    public:
      StaticTrace& operator=(const StaticTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return true; }
      virtual bool is_dynamic_trace(void) const { return false; }
      virtual StaticTrace* as_static_trace(void) { return this; }
      virtual DynamicTrace* as_dynamic_trace(void) { return NULL; }
    public:
      virtual bool is_fixed(void) const;
      virtual bool handles_region_tree(RegionTreeID tid) const;
      virtual void record_static_dependences(Operation *op,
                              const std::vector<StaticDependence> *dependences);
      virtual void register_operation(Operation *op, GenerationID gen); 
      virtual void record_dependence(Operation *target,GenerationID target_gen,
                                     Operation *source,GenerationID source_gen);
      virtual void record_region_dependence(
                                    Operation *target, GenerationID target_gen,
                                    Operation *source, GenerationID source_gen,
                                    unsigned target_idx, unsigned source_idx,
                                    DependenceType dtype, bool validates,
                                    const FieldMask &dependent_mask);
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
    protected:
      const LegionVector<DependenceRecord>::aligned&
                  translate_dependence_records(Operation *op, unsigned index);
    protected:
      std::deque<std::vector<StaticDependence> > static_dependences;
      std::deque<LegionVector<DependenceRecord>::aligned> translated_deps;
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
      DynamicTrace(TraceID tid, TaskContext *ctx, bool memoize);
      DynamicTrace(const DynamicTrace &rhs);
      virtual ~DynamicTrace(void);
    public:
      DynamicTrace& operator=(const DynamicTrace &rhs);
    public:
      virtual bool is_static_trace(void) const { return false; }
      virtual bool is_dynamic_trace(void) const { return true; }
      virtual StaticTrace* as_static_trace(void) { return NULL; }
      virtual DynamicTrace* as_dynamic_trace(void) { return this; }
    public:
      // Called by task execution thread
      virtual bool is_fixed(void) const { return fixed; }
      void fix_trace(void);
    public:
      // Called by analysis thread
      void end_trace_capture(void);
    public:
      virtual void record_static_dependences(Operation *op,
                          const std::vector<StaticDependence> *dependences);
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
      virtual void record_aliased_children(unsigned req_index, unsigned depth,
                                           const FieldMask &aliased_mask);
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
               LegionVector<DependenceRecord>::aligned> internal_dependences;
    protected: 
      // This is the generalized form of the dependences
      // For each operation, we remember a list of operations that
      // it dependens on and whether it is a validates the region
      std::deque<LegionVector<DependenceRecord>::aligned> dependences;
      // Metadata for checking the validity of a trace when it is replayed
      std::vector<OperationInfo> op_info;
    protected:
      const TraceID tid;
      bool fixed;
      bool tracing;
    };

    /**
     * \class TraceCaptureOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when a trace capture
     * is finished so the DynamicTrace object can compute the
     * dependences data structure.
     */
    class TraceCaptureOp : public FenceOp {
    public:
      static const AllocationType alloc_type = TRACE_CAPTURE_OP_ALLOC;
    public:
      TraceCaptureOp(Runtime *rt);
      TraceCaptureOp(const TraceCaptureOp &rhs);
      virtual ~TraceCaptureOp(void);
    public:
      TraceCaptureOp& operator=(const TraceCaptureOp &rhs);
    public:
      void initialize_capture(TaskContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      DynamicTrace *local_trace;
    };

    /**
     * \class TraceCompleteOp
     * This class represents trace operations which we inject
     * into the operation stream to mark when the execution
     * of a trace has been completed.  This fence operation
     * then registers dependences on all operations in the trace
     * and becomes the new current fence.
     */
    class TraceCompleteOp : public FenceOp {
    public:
      static const AllocationType alloc_type = TRACE_COMPLETE_OP_ALLOC;
    public:
      TraceCompleteOp(Runtime *rt);
      TraceCompleteOp(const TraceCompleteOp &rhs);
      virtual ~TraceCompleteOp(void);
    public:
      TraceCompleteOp& operator=(const TraceCompleteOp &rhs);
    public:
      void initialize_complete(TaskContext *ctx);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      LegionTrace *local_trace;
    };

    /**
     * \class TraceReplayOp
     * This class represents trace operations which we inject
     * into the operation stream to replay a physical trace
     * if there is one that satisfies its preconditions.
     */
    class TraceReplayOp : public FenceOp {
    public:
      static const AllocationType alloc_type = TRACE_REPLAY_OP_ALLOC;
    public:
      TraceReplayOp(Runtime *rt);
      TraceReplayOp(const TraceReplayOp &rhs);
      virtual ~TraceReplayOp(void);
    public:
      TraceReplayOp& operator=(const TraceReplayOp &rhs);
    public:
      void initialize_replay(TaskContext *ctx, LegionTrace *trace);
    public:
      virtual void activate(void);
      virtual void deactivate(void);
      virtual const char* get_logging_name(void) const;
      virtual OpKind get_operation_kind(void) const;
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
    protected:
      LegionTrace *local_trace;
    };

    struct CachedMapping
    {
      VariantID               chosen_variant;
      TaskPriority            task_priority;
      bool                    postmap_task;
      std::vector<Processor>  target_procs;
      std::deque<InstanceSet> physical_instances;
    };

    typedef
      LegionMap<std::pair<unsigned, DomainPoint>, CachedMapping>::aligned
      CachedMappings;

    /**
     * \class PhysicalTrace
     * This class is used for memoizing the dynamic physical dependence
     * analysis for series of operations in a given task's context.
     */
    class PhysicalTrace {
    public:
      PhysicalTrace(Runtime *runtime);
      PhysicalTrace(const PhysicalTrace &rhs);
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace &rhs);
    public:
      void check_template_preconditions();
      void get_current_template(PhysicalTraceInfo &trace_info,
                                bool allow_create = true);
      PhysicalTemplate* get_current_template() { return current_template; }
    private:
      void start_new_template();
    private:
      PhysicalTemplate* get_template(PhysicalTraceInfo &trace_info);
    public:
      void initialize_template(ApEvent fence_completion);
    public:
      void fix_trace(void);
      inline bool is_tracing(void) const { return tracing; }
      inline bool is_recurrent(void) const
        { return current_template != NULL &&
                 current_template == previous_template; }
      void finish_replay(void);
      void clear_cached_template(void) { previous_template = NULL; }
    public:
      Runtime *runtime;
    private:
      bool tracing;
      Reservation trace_lock;
      PhysicalTemplate* current_template;
      PhysicalTemplate* previous_template;
      std::vector<PhysicalTemplate*> templates;
    };

    typedef std::pair<unsigned, DomainPoint> TraceLocalId;

    /**
     * \class PhysicalTemplate
     * This class represents a recipe to reconstruct a physical task graph.
     * A template consists of a sequence of instructions, each of which is
     * interpreted by the template engine. The template also maintains
     * the interpreter state (operations and events). These are initialized
     * before the template gets executed.
     */
    struct PhysicalTemplate {
    public:
      PhysicalTemplate(PhysicalTrace *runtime);
      PhysicalTemplate(const PhysicalTemplate &rhs);
    private:
      friend class PhysicalTrace;
      ~PhysicalTemplate();
    public:
      void initialize(ApEvent fence_completion);
    private:
      static bool check_logical_open(RegionTreeNode *node, ContextID ctx,
                                     FieldMask fields);
      static bool check_logical_open(RegionTreeNode *node, ContextID ctx,
                                   LegionMap<Domain, FieldMask>::aligned projs);
    public:
      bool check_preconditions();
      void register_operation(Operation *op);
      void execute_all();
      void finalize();
      void optimize();
      void schedule();
      void reduce_fanout();
      void dump_template();
    public:
      inline bool is_tracing() const { return tracing; }
      inline bool is_replaying() const { return !tracing; }
      inline bool is_replayable() const { return replayable; }
    protected:
      static std::string view_to_string(const InstanceView *view);
      void sanity_check();
    public:
      void record_mapper_output(PhysicalTraceInfo &trace_info,
                                const Mapper::MapTaskOutput &output,
                             const std::deque<InstanceSet> &physical_instances);
      void get_mapper_output(PhysicalTraceInfo &trace_info,
                             VariantID &chosen_variant,
                             TaskPriority &task_priority,
                             bool &postmap_task,
                             std::vector<Processor> &target_proc,
                             std::deque<InstanceSet> &physical_instances) const;
    public:
      void record_get_term_event(PhysicalTraceInfo &trace_info,
                                 ApEvent lhs, SingleTask* task);
      void record_merge_events(PhysicalTraceInfo &trace_info, ApEvent &lhs,
                               ApEvent rhs);
      void record_merge_events(PhysicalTraceInfo &trace_info, ApEvent &lhs,
                               ApEvent e1, ApEvent e2);
      void record_merge_events(PhysicalTraceInfo &trace_info, ApEvent &lhs,
                               ApEvent e1, ApEvent e2, ApEvent e3);
      void record_merge_events(PhysicalTraceInfo &trace_info,
                               ApEvent &lhs,
                               const std::set<ApEvent>& rhs);
      void record_copy_views(PhysicalTraceInfo &trace_info,
                             InstanceView *src,
                             const FieldMask &src_mask,
                             ContextID src_logical_ctx,
                             ContextID src_physucal_ctx,
                             InstanceView *dst,
                             const FieldMask &dst_mask,
                             ContextID dst_logical_ctx,
                             ContextID dst_physical_ctx);
      void record_issue_copy(PhysicalTraceInfo &trace_info,
                             Operation* op, ApEvent &lhs,
                             const Domain &domain,
                         const std::vector<Domain::CopySrcDstField>& src_fields,
                         const std::vector<Domain::CopySrcDstField>& dst_fields,
                             ApEvent precondition,
#ifdef LEGION_SPY
                             LogicalRegion handle,
                             RegionTreeNode *intersect,
#endif
                             ReductionOpID redop = 0,
                             bool reduction_fold = true);
      void record_empty_copy(CompositeView *src,
                             const FieldMask &src_mask,
                             MaterializedView *dst,
                             const FieldMask &dst_mask,
                             ContextID logical_ctx);
      void record_set_ready_event(PhysicalTraceInfo &trace_info,
                                  Operation *op,
                                  unsigned region_idx,
                                  unsigned inst_idx,
                                  ApEvent ready_event,
                                  const RegionRequirement &req,
                                  InstanceView *view,
                                  const FieldMask &fields,
                                  ContextID logical_ctx,
                                  ContextID physical_ctx);
      void record_get_copy_term_event(PhysicalTraceInfo &trace_info,
                                      ApEvent lhs, CopyOp *copy);
      void record_set_copy_sync_event(PhysicalTraceInfo &trace_info,
                                      ApEvent &lhs, CopyOp *copy);
      void record_trigger_copy_completion(PhysicalTraceInfo &trace_info,
                                          CopyOp *copy, ApEvent rhs);
      void record_issue_fill(PhysicalTraceInfo &trace_info,
                             Operation *op, ApEvent &lhs,
                             const Domain &domain,
                             const std::vector<Domain::CopySrcDstField> &fields,
                             const void *fill_buffer, size_t fill_size,
                             ApEvent precondition
#ifdef LEGION_SPY
                             , LogicalRegion handle
#endif
                             );

    private:
      void record_ready_view(PhysicalTraceInfo &trace_info,
                             const RegionRequirement &req,
                             InstanceView *view,
                             const FieldMask &fields,
                             ContextID logical_ctx,
                             ContextID physical_ctx);
    private:
      PhysicalTrace *trace;
      bool tracing;
      bool replayable;
      Reservation template_lock;
      unsigned fence_completion_id;
      unsigned no_event_id;
    private:
      std::map<ApEvent, unsigned> event_map;
      std::vector<Instruction*> instructions;
      std::map<TraceLocalId, std::vector<Instruction*> > inst_map;
      std::map<TraceLocalId, unsigned> task_entries;
    public:
      ApEvent fence_completion;
      std::map<TraceLocalId, Operation*> operations;
      std::vector<ApEvent> events;
      std::vector<ApUserEvent> pending_events;
      std::vector<TraceLocalId> op_list;
      CachedMappings                                  cached_mappings;
      LegionMap<InstanceView*, FieldMask>::aligned    previous_valid_views;
      LegionMap<std::pair<RegionTreeNode*, ContextID>,
                FieldMask>::aligned                   previous_open_nodes;
      std::map<std::pair<RegionTreeNode*, ContextID>,
               LegionMap<Domain, FieldMask>::aligned> previous_projections;
      LegionMap<InstanceView*, FieldMask>::aligned    valid_views;
      LegionMap<InstanceView*, FieldMask>::aligned    reduction_views;
      LegionMap<FillView*,     FieldMask>::aligned    fill_views;
      LegionMap<InstanceView*, bool>::aligned         initialized;
      LegionMap<InstanceView*, ContextID>::aligned    logical_contexts;
      LegionMap<InstanceView*, ContextID>::aligned    physical_contexts;
    };

    enum InstructionKind
    {
      GET_TERM_EVENT = 0,
      GET_COPY_TERM_EVENT,
      MERGE_EVENT,
      ISSUE_COPY,
      ISSUE_FILL,
      SET_COPY_SYNC_EVENT,
      ASSIGN_FENCE_COMPLETION,
      SET_READY_EVENT,
      TRIGGER_COPY_COMPLETION,
      LAUNCH_TASK,
    };

    /**
     * \class Instruction
     * This class is an abstract parent class for all template instructions.
     */
    struct Instruction {
      Instruction(PhysicalTemplate& tpl);
      virtual ~Instruction() {};
      virtual void execute() = 0;
      virtual std::string to_string() = 0;

      virtual InstructionKind get_kind() = 0;
      virtual GetTermEvent* as_get_term_event() = 0;
      virtual MergeEvent* as_merge_event() = 0;
      virtual AssignFenceCompletion* as_assignment_fence_completion() = 0;
      virtual IssueCopy* as_issue_copy() = 0;
      virtual IssueFill* as_issue_fill() = 0;
      virtual SetReadyEvent* as_set_ready_event() = 0;
      virtual GetCopyTermEvent* as_get_copy_term_event() = 0;
      virtual SetCopySyncEvent* as_set_copy_sync_event() = 0;
      virtual TriggerCopyCompletion* as_triger_copy_completion() = 0;

      virtual Instruction* clone(PhysicalTemplate& tpl,
                               const std::map<unsigned, unsigned> &rewrite) = 0;

      virtual TraceLocalId get_owner(const TraceLocalId &key) = 0;

    protected:
      std::map<TraceLocalId, Operation*> &operations;
      std::vector<ApEvent> &events;
      std::vector<ApUserEvent> &pending_events;
    };

    /**
     * \class GetTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_task_completion()
     */
    struct GetTermEvent : public Instruction {
      GetTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                   const TraceLocalId& rhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return GET_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event()
        { return this; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalId rhs;
    };

    /**
     * \class MergeEvent
     * This instruction has the following semantics:
     *   events[lhs] = Runtime::merge_events(events[rhs])
     */
    struct MergeEvent : public Instruction {
      MergeEvent(PhysicalTemplate& tpl, unsigned lhs,
                 const std::set<unsigned>& rhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return MERGE_EVENT; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return this; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      std::set<unsigned> rhs;
    };

    /**
     * \class AssignFenceCompletion
     * This instruction has the following semantics:
     *   events[lhs] = fence_completion
     */
    struct AssignFenceCompletion : public Instruction {
      AssignFenceCompletion(PhysicalTemplate& tpl, unsigned lhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return ASSIGN_FENCE_COMPLETION; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return this; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return key; }

    private:
      friend struct PhysicalTemplate;
      ApEvent &fence_completion;
      unsigned lhs;
    };

    /**
     * \class IssueFill
     * This instruction has the following semantics:
     *
     *   events[lhs] = domain.fill(fields, requests,
     *                             fill_buffer, fill_size,
     *                             events[precondition_idx]);
     */
    struct IssueFill : public Instruction {
      IssueFill(PhysicalTemplate &tpl,
                unsigned lhs, const Domain &domain,
                const TraceLocalId &op_key,
                const std::vector<Domain::CopySrcDstField> &fields,
                const ReductionOp *reduction_op,
                unsigned precondition_idx
#ifdef LEGION_SPY
                , LogicalRegion handle
#endif
                );
      IssueFill(PhysicalTemplate& tpl,
                unsigned lhs, const Domain &domain,
                const TraceLocalId &op_key,
                const std::vector<Domain::CopySrcDstField> &fields,
                const void *fill_buffer, size_t fill_size,
                unsigned precondition_idx
#ifdef LEGION_SPY
                , LogicalRegion handle
#endif
                );
      virtual ~IssueFill();
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return ISSUE_FILL; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return this; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      Domain domain;
      TraceLocalId op_key;
      std::vector<Domain::CopySrcDstField> fields;
      void *fill_buffer;
      size_t fill_size;
      unsigned precondition_idx;
#ifdef LEGION_SPY
      LogicalRegion handle;
#endif
    };

    /**
     * \class IssueCopy
     * This instruction has the following semantics:
     *   events[lhs] = node->issue_copy(operations[op_key],
     *                                  src_fields, dst_fields,
     *                                  events[precondition_idx],
     *                                  predicate_guard, intersect,
     *                                  redop, reduction_fold);
     */
    struct IssueCopy : public Instruction {
      IssueCopy(PhysicalTemplate &tpl,
                unsigned lhs, const Domain &domain,
                const TraceLocalId &op_key,
                const std::vector<Domain::CopySrcDstField>& src_fields,
                const std::vector<Domain::CopySrcDstField>& dst_fields,
                unsigned precondition_idx,
#ifdef LEGION_SPY
                LogicalRegion handle,
                RegionTreeNode *intersect,
#endif
                ReductionOpID redop, bool reduction_fold);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return ISSUE_COPY; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return this; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      Domain domain;
      TraceLocalId op_key;
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      unsigned precondition_idx;
#ifdef LEGION_SPY
      LogicalRegion handle;
      RegionTreeNode *intersect;
#endif
      ReductionOpID redop;
      bool reduction_fold;
    };

    /**
     * \class SetReadyEvent
     * This instruction has the following semantics:
     *   operations[op_key]->get_physical_instances()[region_idx][inst_idx]
     *                      .set_ready_event(events[ready_event_idx])
     */
    struct SetReadyEvent : public Instruction {
      SetReadyEvent(PhysicalTemplate& tpl,
                    const TraceLocalId& op_key,
                    unsigned region_idx,
                    unsigned inst_idx,
                    unsigned ready_event_idx,
                    InstanceView *view);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return SET_READY_EVENT; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return this; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      TraceLocalId op_key;
      unsigned region_idx;
      unsigned inst_idx;
      unsigned ready_event_idx;
      InstanceView* view;
    };

    /**
     * \class GetCopyTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_completion_event()
     */
    struct GetCopyTermEvent : public Instruction {
      GetCopyTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                       const TraceLocalId& rhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return GET_COPY_TERM_EVENT; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return this; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalId rhs;
    };

    /**
     * \class SetCopySyncEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].compute_sync_precondition()
     */
    struct SetCopySyncEvent : public Instruction {
      SetCopySyncEvent(PhysicalTemplate& tpl, unsigned lhs,
                       const TraceLocalId& rhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return SET_COPY_SYNC_EVENT; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return this; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return rhs; }

    private:
      friend struct PhysicalTemplate;
      unsigned lhs;
      TraceLocalId rhs;
    };

    /**
     * \class TriggerCopyCompletion
     * This instruction has the following semantics:
     *   operations[lhs]->complete_copy_execution(events[rhs])
     */
    struct TriggerCopyCompletion : public Instruction {
      TriggerCopyCompletion(PhysicalTemplate& tpl, const TraceLocalId& lhs,
                            unsigned rhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return TRIGGER_COPY_COMPLETION; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return this; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return lhs; }

    private:
      friend struct PhysicalTemplate;
      TraceLocalId lhs;
      unsigned rhs;
    };

    struct LaunchTask : public Instruction {
      LaunchTask(PhysicalTemplate& tpl, const TraceLocalId& lhs);
      virtual void execute();
      virtual std::string to_string();

      virtual InstructionKind get_kind()
        { return LAUNCH_TASK; }
      virtual GetTermEvent* as_get_term_event()
        { return NULL; }
      virtual MergeEvent* as_merge_event()
        { return NULL; }
      virtual AssignFenceCompletion* as_assignment_fence_completion()
        { return NULL; }
      virtual IssueCopy* as_issue_copy()
        { return NULL; }
      virtual IssueFill* as_issue_fill()
        { return NULL; }
      virtual SetReadyEvent* as_set_ready_event()
        { return NULL; }
      virtual GetCopyTermEvent* as_get_copy_term_event()
        { return NULL; }
      virtual SetCopySyncEvent* as_set_copy_sync_event()
        { return NULL; }
      virtual TriggerCopyCompletion* as_triger_copy_completion()
        { return NULL; }

      virtual Instruction* clone(PhysicalTemplate& tpl,
                                 const std::map<unsigned, unsigned> &rewrite);

      virtual TraceLocalId get_owner(const TraceLocalId &key) { return op_key; }

    private:
      friend struct PhysicalTemplate;
      TraceLocalId op_key;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TRACE__
