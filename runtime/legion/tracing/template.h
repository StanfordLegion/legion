/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_TRACE_TEMPLATE_H__
#define __LEGION_TRACE_TEMPLATE_H__

#include "legion/analysis/equivalence_set.h"
#include "legion/tracing/physical.h"
#include "legion/tracing/recording.h"
#include "legion/tracing/viewset.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TraceConditionSet
     */
    class TraceConditionSet
      : public EqSetTracker,
        public Collectable,
        public Heapify<TraceConditionSet, CONTEXT_LIFETIME> {
    public:
      TraceConditionSet(
          PhysicalTemplate* tpl, unsigned parent_req_index,
          RegionTreeID tree_id, IndexSpaceExpression* expr,
          lng::FieldMaskMap<LogicalView>&& views);
      TraceConditionSet(const TraceConditionSet& rhs) = delete;
      virtual ~TraceConditionSet(void);
    public:
      TraceConditionSet& operator=(const TraceConditionSet& rhs) = delete;
    public:
      inline bool is_shared(void) const { return shared; }
      inline void mark_shared(void) { shared = true; }
    public:
      virtual void add_subscription_reference(unsigned count = 1) override
      {
        add_reference(count);
      }
      virtual bool remove_subscription_reference(unsigned count = 1) override
      {
        return remove_reference(count);
      }
      virtual RegionTreeID get_region_tree_id(void) const override
      {
        return tree_id;
      }
      virtual IndexSpaceExpression* get_tracker_expression(void) const override
      {
        return condition_expr;
      }
      virtual ReferenceSource get_reference_source_kind(void) const override
      {
        return TRACE_REF;
      }
    public:
      bool matches(
          IndexSpaceExpression* expr,
          const FieldMapView<LogicalView>& views) const;
      void invalidate_equivalence_sets(void);
      void refresh_equivalence_sets(
          FenceOp* op, std::set<RtEvent>& ready_events);
      void dump_conditions(void) const;
    public:
      void test_preconditions(
          FenceOp* op, unsigned index, std::vector<RtEvent>& ready_events,
          std::set<RtEvent>& applied_events);
      bool check_preconditions(void);
      void test_anticonditions(
          FenceOp* op, unsigned index, std::vector<RtEvent>& ready_events,
          std::set<RtEvent>& applied_events);
      bool check_anticonditions(void);
      void apply_postconditions(
          FenceOp* op, unsigned index, std::set<RtEvent>& applied_events);
    public:
      PhysicalTemplate* const owner;
      IndexSpaceExpression* const condition_expr;
      const lng::FieldMaskMap<LogicalView> views;
      const RegionTreeID tree_id;
      const unsigned parent_req_index;
    private:
      mutable LocalLock set_lock;
    private:
      union {
        InvalidInstAnalysis* invalid;
        AntivalidInstAnalysis* antivalid;
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
    class PhysicalTemplate
      : public PhysicalTraceRecorder,
        public Heapify<PhysicalTemplate, CONTEXT_LIFETIME> {
    public:
      struct ReplaySliceArgs : public LgTaskArgs<ReplaySliceArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_REPLAY_SLICE_TASK_ID;
        static constexpr bool IS_APPLICATION_TASK = true;
      public:
        ReplaySliceArgs(void) = default;
        ReplaySliceArgs(PhysicalTemplate* t, unsigned si, bool recurrent)
          : LgTaskArgs<ReplaySliceArgs>(false, true), tpl(t), slice_index(si),
            recurrent_replay(recurrent)
        { }
        void execute(void) const;
      public:
        PhysicalTemplate* tpl;
        unsigned slice_index;
        bool recurrent_replay;
      };
      struct DeleteTemplateArgs : public LgTaskArgs<DeleteTemplateArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DELETE_TEMPLATE_TASK_ID;
      public:
        DeleteTemplateArgs(void) = default;
        DeleteTemplateArgs(PhysicalTemplate* t)
          : LgTaskArgs<DeleteTemplateArgs>(true, true), tpl(t)
        { }
        void execute(void) const;
      public:
        PhysicalTemplate* tpl;
      };
    private:
      struct CachedPremapping {
        std::vector<Memory> future_locations;
      };
      typedef std::map<TraceLocalID, CachedPremapping> CachedPremappings;
      struct CachedMapping {
        VariantID chosen_variant;
        TaskPriority task_priority;
        bool postmap_task;
        std::vector<Processor> target_procs;
        std::vector<Memory> future_locations;
        std::map<std::pair<Memory, bool>, PoolBounds> pool_bounds;
        std::deque<InstanceSet> physical_instances;
      };
      typedef lng::map<TraceLocalID, CachedMapping> CachedMappings;
    private:
      struct CachedAllreduce {
        std::vector<Memory> target_memories;
        size_t future_size;
      };
    protected:
      struct InstanceUser {
      public:
        InstanceUser(void) : expr(nullptr) { }
        InstanceUser(
            const UniqueInst& i, const RegionUsage& r, IndexSpaceExpression* e,
            const FieldMask& m)
          : instance(i), usage(r), expr(e), mask(m)
        { }
      public:
        inline bool matches(
            const UniqueInst& inst, const RegionUsage& use,
            IndexSpaceExpression* expression) const
        {
          if (inst != instance)
            return false;
          if (use != usage)
            return false;
          return (expr == expression);
        }
        inline bool matches(const InstanceUser& user) const
        {
          if (instance != user.instance)
            return false;
          if (usage != user.usage)
            return false;
          if (expr != user.expr)
            return false;
          return (mask == user.mask);
        }
      public:
        UniqueInst instance;
        RegionUsage usage;
        IndexSpaceExpression* expr;
        FieldMask mask;
      };
      typedef ctx::vector<InstanceUser> InstUsers;
      struct LastUserResult {
      public:
        LastUserResult(const InstanceUser& u) : user(u) { }
      public:
        const InstanceUser& user;
        std::set<ApEvent> events;
        std::vector<unsigned> frontiers;
      };
    private:
      // State for deferring the transitive reduction into time-slices since
      // it is really expensive and we don't want it monopolizing a processor
      struct TransitiveReductionState {
      public:
        TransitiveReductionState(RtUserEvent d)
          : stage(0), iteration(0), num_chains(0), pos(-1), done(d)
        { }
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
    public:
      struct TransitiveReductionArgs
        : public LgTaskArgs<TransitiveReductionArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_TRANSITIVE_REDUCTION_TASK_ID;
      public:
        TransitiveReductionArgs(void) = default;
        TransitiveReductionArgs(
            PhysicalTemplate* t, TransitiveReductionState* s)
          : LgTaskArgs<TransitiveReductionArgs>(false, true), tpl(t), state(s)
        { }
        void execute(void) const;
      public:
        PhysicalTemplate* tpl;
        TransitiveReductionState* state;
      };
    public:
      PhysicalTemplate(PhysicalTrace* trace, ApEvent fence_event);
      PhysicalTemplate(const PhysicalTemplate& rhs) = delete;
      virtual ~PhysicalTemplate(void);
    public:
      PhysicalTemplate& operator=(const PhysicalTemplate& rhs) = delete;
    public:
      virtual size_t get_sharded_template_index(void) const { return 0; }
      virtual void initialize_replay(ApEvent fence_completion, bool recurrent);
      virtual void start_replay(void);
      virtual RtEvent refresh_managed_barriers(void);
      virtual void finish_replay(
          FenceOp* op, std::set<ApEvent>& postconditions);
      virtual ApEvent get_completion_for_deletion(void) const;
    public:
      ReplayableStatus finalize(CompleteOp* op, bool has_blocking_call);
      IdempotencyStatus capture_conditions(CompleteOp* op);
      void receive_trace_conditions(
          TraceViewSet* preconditions, TraceViewSet* anticonditions,
          TraceViewSet* postconditions,
          const FieldMapView<IndexSpaceExpression>& unique_dirty_exprs,
          unsigned parent_req_index, RegionTreeID tree_id,
          std::atomic<unsigned>* result);
      void refresh_condition_sets(
          FenceOp* op, std::set<RtEvent>& ready_events) const;
      bool acquire_instance_references(void) const;
      void release_instance_references(std::set<RtEvent>& applied) const;
    public:
      void optimize(CompleteOp* op, bool do_transitive_reduction);
    private:
      void find_all_last_instance_user_events(
          std::vector<RtEvent>& frontier_events);
      void find_last_instance_events(
          const InstUsers& users, std::vector<RtEvent>& frontier_events);
      void compute_frontiers(std::vector<RtEvent>& frontier_events);
      void elide_fences(
          std::vector<unsigned>& gen, std::vector<RtEvent>& ready_events);
      void propagate_merges(std::vector<unsigned>& gen);
      void transitive_reduction(TransitiveReductionState* state, bool deferred);
      void finalize_transitive_reduction(
          const std::vector<unsigned>& inv_topo_order,
          const std::vector<std::vector<unsigned> >& incoming_reduced);
      void check_finalize_transitive_reduction(void);
      void propagate_copies(std::vector<unsigned>* gen);
      void eliminate_dead_code(std::vector<unsigned>& gen);
      void prepare_parallel_replay(const std::vector<unsigned>& gen);
      void push_complete_replays(void);
    protected:
      virtual void sync_compute_frontiers(
          CompleteOp* op, const std::vector<RtEvent>& frontier_events);
      virtual void initialize_generators(std::vector<unsigned>& new_gen);
      virtual void initialize_eliminate_dead_code_frontiers(
          const std::vector<unsigned>& gen, std::vector<bool>& used);
      virtual void initialize_transitive_reduction_frontiers(
          std::vector<unsigned>& topo_order,
          std::vector<unsigned>& inv_topo_order);
      virtual void record_used_frontiers(
          std::vector<bool>& used, const std::vector<unsigned>& gen) const;
      virtual void rewrite_frontiers(
          std::map<unsigned, unsigned>& substitutions);
    public:
      RtEvent test_preconditions(
          FenceOp* op, std::set<RtEvent>& applied_events);
      bool check_preconditions(void);
      void apply_postconditions(FenceOp* op, std::set<RtEvent>& applied_events);
      void invalidate_equivalence_sets(void) const;
    public:
      bool can_start_replay(void);
      void register_operation(MemoizableOp* op);
      void execute_slice(unsigned slice_idx, bool recurrent_replay);
    public:
      void dump_template(void) const;
      virtual void dump_sharded_template(void) const { }
    private:
      void dump_instructions(
          const std::vector<Instruction*>& instructions) const;
    public:
      void set_fence_uid(UniqueID fence_uid) { prev_fence_uid = fence_uid; }
      UniqueID get_fence_uid(void) const { return prev_fence_uid; }
    public:
      inline bool is_replaying(void) const { return trace->is_replaying(); }
      inline bool is_replayable(void) const
      {
        return (replayable == REPLAYABLE);
      }
      inline bool is_idempotent(void) const
      {
        return (idempotency == IDEMPOTENT);
      }
      inline void record_no_consensus(void) { has_no_consensus.store(true); }
    public:
      virtual bool is_recording(void) const override
      {
        return trace->is_recording();
      }
      virtual void add_recorder_reference(void) override { /*do nothing*/ }
      virtual bool remove_recorder_reference(void) override
      { /*do nothing, never delete*/ return false; }
      virtual void pack_recorder(Serializer& rez) override;
    public:
      void record_premap_output(
          MemoizableOp* memo, const Mapper::PremapTaskOutput& output,
          std::set<RtEvent>& applied_events);
      void get_premap_output(
          IndexTask* task, std::vector<Memory>& future_locations);
      virtual void record_mapper_output(
          const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
          const std::deque<InstanceSet>& physical_instances, bool is_leaf,
          bool has_return_size, std::set<RtEvent>& applied_events) override;
      void get_mapper_output(
          SingleTask* task, VariantID& chosen_variant,
          TaskPriority& task_priority, bool& postmap_task,
          std::vector<Processor>& target_proc,
          std::vector<Memory>& future_locations,
          std::map<std::pair<Memory, bool>, PoolBounds>& pool_bounds,
          std::deque<InstanceSet>& physical_instances) const;
      void get_task_reservations(
          SingleTask* task, std::map<Reservation, bool>& reservations) const;
      void get_allreduce_mapping(
          AllReduceOp* op, std::vector<Memory>& target_memories,
          size_t& future_size);
      void initialize_concurrent_groups(IndexTask* task);
    public:
      virtual void record_replay_mapping(
          ApEvent lhs, unsigned op_kind, const TraceLocalID& tlid,
          std::set<RtEvent>& applied_events) override;
      virtual void request_term_event(ApUserEvent& term_event) override;
      virtual void record_create_ap_user_event(
          ApUserEvent& lhs, const TraceLocalID& tlid) override;
      virtual void record_trigger_event(
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
          std::set<RtEvent>& applied) override;
    public:
      virtual void record_merge_events(
          ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
          const TraceLocalID& tlid) override;
      virtual void record_merge_events(
          PredEvent& lhs, PredEvent e1, PredEvent e2,
          const TraceLocalID& tlid) override;
      virtual void record_collective_barrier(
          ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
          size_t arrival_count) override;
      virtual ShardID record_barrier_creation(
          ApBarrier& bar, size_t total_arrivals) override;
      virtual void record_barrier_arrival(
          ApBarrier bar, ApEvent pre, size_t arrival_count,
          std::set<RtEvent>& applied, ShardID owner_shard) override;
    public:
      virtual void record_issue_copy(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent pred_guard, LgEvent src_unique,
          LgEvent dst_unique, int priority, CollectiveKind collective,
          bool record_effect) override;
      virtual void record_issue_across(
          const TraceLocalID& tlid, ApEvent& lhs,
          ApEvent collective_precondition, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          CopyAcrossExecutor* executor) override;
      virtual void record_copy_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const UniqueInst& src_inst, const UniqueInst& dst_inst,
          const FieldMask& src_mask, const FieldMask& dst_mask,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, ReductionOpID redop,
          std::set<RtEvent>& applied) override;
      virtual void record_across_insts(
          ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
          unsigned dst_idx, IndexSpaceExpression* expr,
          const AcrossInsts& src_insts, const AcrossInsts& dst_insts,
          PrivilegeMode src_mode, PrivilegeMode dst_mode, bool src_indirect,
          bool dst_indirect, std::set<RtEvent>& applied) override;
      virtual void record_indirect_insts(
          ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
          const AcrossInsts& insts, std::set<RtEvent>& applied,
          PrivilegeMode priv) override;
      virtual void record_issue_fill(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
          LgEvent unique_event, int priority, CollectiveKind collective,
          bool record_effect) override;
    public:
      virtual void record_op_inst(
          const TraceLocalID& tlid, unsigned idx, const UniqueInst& inst,
          RegionNode* node, const RegionUsage& usage,
          const FieldMask& user_mask, bool update_validity,
          std::set<RtEvent>& applied) override;
      virtual void record_fill_inst(
          ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& inst,
          const FieldMask& fill_mask, std::set<RtEvent>& applied_events,
          const bool reduction_initialization) override;
    protected:
      void record_instance_user(
          InstUsers& users, const UniqueInst& instance,
          const RegionUsage& usage, IndexSpaceExpression* expr,
          const FieldMask& mask, std::set<RtEvent>& applied_events);
      virtual void record_mutated_instance(
          const UniqueInst& inst, IndexSpaceExpression* expr,
          const FieldMask& mask, std::set<RtEvent>& applied_events);
    public:
      virtual void record_set_op_sync_event(
          ApEvent& lhs, const TraceLocalID& tlid) override;
      virtual void record_complete_replay(
          const TraceLocalID& tlid, ApEvent pre,
          std::set<RtEvent>& applied_events) override;
      virtual void record_reservations(
          const TraceLocalID& tlid, const std::map<Reservation, bool>& locks,
          std::set<RtEvent>& applied_events) override;
      virtual void record_future_allreduce(
          const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
          size_t future_size) override;
      void record_concurrent_group(
          IndexTask* task, Color color, size_t local, size_t global,
          RtBarrier bar, const std::vector<ShardID>& shards);
      void record_execution_fence(const TraceLocalID& tlid);
    public:
      virtual void record_owner_shard(unsigned trace_local_id, ShardID owner);
      virtual void record_local_space(unsigned trace_local_id, IndexSpace sp);
      virtual void record_sharding_function(
          unsigned trace_local_id, ShardingFunction* function);
    public:
      virtual ShardID find_owner_shard(unsigned trace_local_id);
      virtual IndexSpace find_local_space(unsigned trace_local_id);
      virtual ShardingFunction* find_sharding_function(unsigned trace_local_id);
    public:
      bool defer_template_deletion(
          ApEvent& pending_deletion, std::set<RtEvent>& applied_events);
    protected:
      void record_memo_entry(
          const TraceLocalID& tlid, unsigned entry, unsigned op_kind);
    protected:
#ifdef LEGION_DEBUG
      // This is a virtual method in debug mode only since we have an
      // assertion that we want to check in the ShardedPhysicalTemplate
      virtual unsigned convert_event(const ApEvent& event, bool check = true);
#else
      unsigned convert_event(const ApEvent& event);
#endif
      virtual unsigned find_event(const ApEvent& event, AutoLock& tpl_lock);
      void insert_instruction(Instruction* inst);
    protected:
      // Returns the set of last users for all <view,field mask,index expr>
      // tuples in the inst_exprs, not that this is the
      void find_all_last_users(
          const InstUsers& inst_users, std::set<unsigned>& last_users) const;
      virtual unsigned find_frontier_event(
          ApEvent event, std::vector<RtEvent>& ready_events);
      // Check to see if any users are mutating these fields and expressions
      virtual bool are_read_only_users(InstUsers& inst_users);
      void rewrite_preconditions(
          unsigned& precondition, std::set<unsigned>& users,
          const std::vector<Instruction*>& instructions,
          std::vector<Instruction*>& new_instructions,
          std::vector<unsigned>& gen, unsigned& merge_starts);
      void parallelize_replay_event(
          unsigned& event_to_check, unsigned slice_index,
          const std::vector<unsigned>& gen,
          const std::vector<unsigned>& slice_indices_by_inst,
          std::map<unsigned, std::pair<unsigned, unsigned> >& crossing_counts,
          std::vector<Instruction*>& crossing_instructions);
    public:
      PhysicalTrace* const trace;
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
      static constexpr unsigned NO_INDEX = std::numeric_limits<unsigned>::max();
    protected:
      std::map<TraceLocalID, MemoizableOp*> operations;
      // Pair in memo_entries is <entry index, Operation::Kind>
      // This data structure is only used during template capture and
      // can be ignored after the template has been optimized
      std::map<TraceLocalID, std::pair<unsigned, unsigned> > memo_entries;
    private:
      CachedPremappings cached_premappings;
      CachedMappings cached_mappings;
      std::map<TraceLocalID, std::map<Reservation, bool> > cached_reservations;
      std::map<TraceLocalID, CachedAllreduce> cached_allreduces;
      bool has_virtual_mapping;
      bool has_non_leaf_task;
      bool has_variable_return_size;
      std::atomic<bool> has_no_consensus;
      mutable TraceViewSet::FailedPrecondition failure;
    protected:
      CompleteReplay* last_fence;
    protected:
      RtEvent replay_precondition;
      RtUserEvent replay_postcondition;
      ApEvent replay_complete;
      std::atomic<unsigned> remaining_replays;
      std::atomic<unsigned> total_logical;
      std::vector<ApEvent> events;
      std::map<unsigned, ApUserEvent> user_events;
    protected:
      std::map<ApEvent, unsigned> event_map;
      std::map<ApEvent, BarrierAdvance*> managed_barriers;
      std::map<ApEvent, std::vector<BarrierArrival*> > managed_arrivals;
      struct ConcurrentGroup {
        ConcurrentGroup(
            Color c, size_t l, size_t g, RtBarrier b,
            const std::vector<ShardID>& s)
          : shards(s), barrier(b), local(l), global(g), color(c)
        { }
        std::vector<ShardID> shards;
        RtBarrier barrier;
        size_t local;
        size_t global;
        Color color;
      };
      std::map<TraceLocalID, std::vector<ConcurrentGroup> > concurrent_groups;
    protected:
      std::vector<Instruction*> instructions;
      std::vector<std::vector<Instruction*> > slices;
      std::vector<std::vector<TraceLocalID> > slice_tasks;
    protected:
      std::map<unsigned /*event*/, unsigned /*consumers*/> crossing_events;
      // Frontiers of a template are a set of users whose events must
      // be carried over to the next replay for eliding the fence at the
      // beginning. We compute this data structure from the last users of
      // each physical instance named in the trace and then looking for
      // the locations of those events inside the trace.
      // After each replay, we do the assignment
      // events[frontiers[idx]] = events[idx]
      std::map<unsigned, unsigned> frontiers;
      // A cache of the specific last user results for individual instances
      std::map<UniqueInst, std::deque<LastUserResult> > instance_last_users;
    protected:
      RtEvent transitive_reduction_done;
      std::atomic<TransitiveReductionState*> finished_transitive_reduction;
    private:
      std::map<TraceLocalID, InstUsers> op_insts;
      std::map<unsigned, InstUsers> copy_insts;
      std::map<unsigned, InstUsers> src_indirect_insts;
      std::map<unsigned, InstUsers> dst_indirect_insts;
      std::vector<IssueAcross*> across_copies;
      std::map<DistributedID, IndividualView*> recorded_views;
      std::set<IndexSpaceExpression*> recorded_expressions;
      std::vector<PhysicalManager*> all_instances;
    protected:
      // Capture the names of all the instances that are mutated by this trace
      // and the index space expressions and fields that were mutated
      // THIS IS SHARDED FOR CONTROL REPLICATION!!!
      shrt::map<UniqueInst, shrt::FieldMaskMap<IndexSpaceExpression> >
          mutated_insts;
    private:
      // THESE ARE SHARDED FOR CONTROL REPLICATION!!!
      // Each share has a disjoint set of trace conditions that they are
      // responsible for handling checking
      std::vector<TraceConditionSet*> preconditions;
      std::vector<TraceConditionSet*> anticonditions;
      std::vector<TraceConditionSet*> postconditions;
    private:
      // For Legion Spy
      UniqueID prev_fence_uid;
    private:
      friend class PhysicalTrace;
      friend class Instruction;
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
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_TEMPLATE_H__
