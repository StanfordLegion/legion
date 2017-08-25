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
#ifdef LEGION_SPY
    protected:
      std::vector<UniqueID> current_uids;
      std::vector<unsigned> num_regions;
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
      DynamicTrace(TraceID tid, TaskContext *ctx);
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
     * \class PhysicalTrace
     * This class is used for memoizing the dynamic physical dependence
     * analysis for series of operations in a given task's context.
     */
    class PhysicalTrace {
    public:
      PhysicalTrace(void);
      PhysicalTrace(const PhysicalTrace &rhs);
      ~PhysicalTrace(void);
    public:
      PhysicalTrace& operator=(const PhysicalTrace &rhs);
    public:
      void record_target_views(PhysicalTraceInfo &trace_info,
                               unsigned idx,
                               const std::vector<InstanceView*> &target_views);
      void get_target_views(PhysicalTraceInfo &trace_info,
                            unsigned idx,
                            std::vector<InstanceView*> &target_views) const;
      void record_mapper_output(PhysicalTraceInfo &trace_info,
                                const Mapper::MapTaskOutput &output,
                             const std::deque<InstanceSet> &physical_instances);
      void get_mapper_output(PhysicalTraceInfo &trace_info,
                             VariantID &chosen_variant,
                             TaskPriority &task_priority,
                             bool &postmap_task,
                             std::vector<Processor> &target_proc,
                             std::deque<InstanceSet> &physical_instances) const;
    private:
      void set_current_template_id(PhysicalTraceInfo &trace_info);
    public:
      void find_or_create_template(PhysicalTraceInfo &trace_info);
    private:
      PhysicalTemplate* get_template(PhysicalTraceInfo &trace_info);
    public:
      void record_get_term_event(PhysicalTraceInfo &trace_info,
                                 ApEvent lhs,
                                 SingleTask* task);
      void record_merge_events(PhysicalTraceInfo &trace_info,
                               ApEvent &lhs,
                               const std::set<ApEvent>& rhs);
      void record_copy_views(PhysicalTraceInfo &trace_info,
                             InstanceView *src,
                             const FieldMask &src_mask,
                             ContextID src_ctx,
                             InstanceView *dst,
                             const FieldMask &dst_mask,
                             ContextID dst_ctx);
      void record_issue_copy(PhysicalTraceInfo &trace_info,
                             ApEvent lhs,
                             RegionTreeNode* node,
                             Operation* op,
                         const std::vector<Domain::CopySrcDstField>& src_fields,
                         const std::vector<Domain::CopySrcDstField>& dst_fields,
                             ApEvent precondition,
                             PredEvent predicate_guard,
                             RegionTreeNode *intersect = NULL,
                             ReductionOpID redop = 0,
                             bool reduction_fold = true);
      void record_set_ready_event(PhysicalTraceInfo &trace_info,
                                  Operation *op,
                                  unsigned region_idx,
                                  unsigned inst_idx,
                                  ApEvent ready_event,
                                  const RegionRequirement &req,
                                  InstanceView *view,
                                  const FieldMask &fields,
                                  ContextID ctx);
    private:
      void record_ready_view(PhysicalTraceInfo &trace_info,
                             const RegionRequirement &req,
                             InstanceView *view,
                             const FieldMask &fields,
                             ContextID ctx);
    public:
      void initialize_templates(ApEvent fence_completion);
      void execute_template(PhysicalTraceInfo &trace_info, SingleTask *task);
    public:
      void fix_trace(void);
      bool is_tracing(void) const { return tracing; }
    private:
      bool tracing;
      Reservation trace_lock;
      ApUserEvent check_complete_event;
      unsigned current_template_id;

      typedef LegionVector<LegionVector<InstanceView*>::aligned >::aligned
        CachedViews;

      struct CachedMapping
      {
        VariantID               chosen_variant;
        TaskPriority            task_priority;
        bool                    postmap_task;
        std::vector<Processor>  target_procs;
        std::deque<InstanceSet> physical_instances;
        CachedViews             target_views;
      };

      typedef
        LegionMap<std::pair<unsigned, DomainPoint>, CachedMapping>::aligned
        CachedMappings;

      CachedMappings cached_mappings;
      std::vector<PhysicalTemplate*> templates;
      LegionMap<InstanceView*, FieldMask>::aligned preconditions;
      LegionMap<InstanceView*, FieldMask>::aligned valid_views;
      LegionMap<InstanceView*, FieldMask>::aligned reduction_views;
      LegionMap<InstanceView*, bool>::aligned      initialized;
      LegionMap<InstanceView*, ContextID>::aligned context_ids;
    };

    /**
     * \class PhysicalTemplate
     * This class represents a recipe to reconstruct a physical task graph.
     * A template consists of a sequence of instructions, each of which is
     * interpreted by the template engine. The template also maintains
     * the interpreter state (operations and events). These are initialized
     * before the template gets executed.
     */
    struct PhysicalTemplate {
      PhysicalTemplate();
      ~PhysicalTemplate();
      PhysicalTemplate(const PhysicalTemplate &rhs);

      void initialize();
      void execute(PhysicalTraceInfo &trace_info, SingleTask *task);
      void finalize(
            const LegionMap<InstanceView*, FieldMask>::aligned &preconditions,
            const LegionMap<InstanceView*, FieldMask>::aligned &valid_views,
            const LegionMap<InstanceView*, FieldMask>::aligned &reduction_views,
            const LegionMap<InstanceView*, bool>::aligned      &initialized,
            const LegionMap<InstanceView*, ContextID>::aligned &context_ids);
      void dump_template();
      static std::string view_to_string(const InstanceView *view);
      void sanity_check();

      Reservation template_lock;

      ApEvent fence_completion;
      unsigned fence_completion_id;
      std::map<std::pair<unsigned, DomainPoint>, unsigned> task_entries;
      std::vector<std::vector<unsigned> > consumers;
      std::vector<unsigned> pending_producers;
      std::vector<unsigned> max_producers;
      std::map<std::pair<unsigned, DomainPoint>, SingleTask*> operations;
      std::vector<ApEvent> events;
      std::map<ApEvent, unsigned> event_map;
      std::vector<Instruction*> instructions;
      LegionMap<InstanceView*, FieldMask>::aligned preconditions;
      LegionMap<InstanceView*, FieldMask>::aligned valid_views;
      LegionMap<InstanceView*, FieldMask>::aligned reduction_views;
      LegionMap<InstanceView*, bool>::aligned      initialized;
      LegionMap<InstanceView*, ContextID>::aligned context_ids;
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

    protected:
      std::map<std::pair<unsigned, DomainPoint>, SingleTask*> &operations;
      std::vector<ApEvent>& events;
    };

    /**
     * \class GetTermEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].get_task_completion()
     */
    struct GetTermEvent : public Instruction {
      GetTermEvent(PhysicalTemplate& tpl, unsigned lhs,
                   const std::pair<unsigned, DomainPoint>& rhs);
      virtual void execute();
      virtual std::string to_string();

    private:
      unsigned lhs;
      std::pair<unsigned, DomainPoint> rhs;
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

    private:
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

    private:
      ApEvent &fence_completion;
      unsigned lhs;
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
                unsigned lhs, RegionTreeNode *node,
                const std::pair<unsigned, DomainPoint> &op_key,
                const std::vector<Domain::CopySrcDstField> &src_fields,
                const std::vector<Domain::CopySrcDstField> &dst_fields,
                unsigned precondition_idx, PredEvent predicate_guard,
                RegionTreeNode *intersect, ReductionOpID redop,
                bool reduction_fold);
      virtual void execute();
      virtual std::string to_string();

    private:
      unsigned lhs;
      RegionTreeNode* node;
      std::pair<unsigned, DomainPoint> op_key;
      std::vector<Domain::CopySrcDstField> src_fields;
      std::vector<Domain::CopySrcDstField> dst_fields;
      unsigned precondition_idx;
      PredEvent predicate_guard;
      RegionTreeNode *intersect;
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
                    const std::pair<unsigned, DomainPoint>& op_key,
                    unsigned region_idx,
                    unsigned inst_idx,
                    unsigned ready_event_idx);
      virtual void execute();
      virtual std::string to_string();

    private:
      std::pair<unsigned, DomainPoint> op_key;
      unsigned region_idx;
      unsigned inst_idx;
      unsigned ready_event_idx;
    };


  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_TRACE__
