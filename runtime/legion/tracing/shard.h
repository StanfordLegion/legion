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

#ifndef __LEGION_SHARD_TEMPLATE_H__
#define __LEGION_SHARD_TEMPLATE_H__

#include "legion/tracing/template.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ShardedPhysicalTemplate
     * This is an extension of the PhysicalTemplate class for handling
     * templates for control replicated contexts. It mostly behaves the
     * same as a normal PhysicalTemplate but has some additional
     * extensions for handling the effects of control replication.
     */
    class ShardedPhysicalTemplate
      : public HeapifyMixin<
            ShardedPhysicalTemplate, PhysicalTemplate, CONTEXT_LIFETIME> {
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
        static constexpr LgTaskID TASK_ID = LG_DEFER_TRACE_UPDATE_TASK_ID;
      public:
        DeferTraceUpdateArgs(void) = default;
        DeferTraceUpdateArgs(
            ShardedPhysicalTemplate* target, UpdateKind kind, RtUserEvent done,
            const UniqueInst& inst, Deserializer& derez,
            IndexSpaceExpression* expr,
            RtUserEvent deferral = RtUserEvent::NO_RT_USER_EVENT);
        DeferTraceUpdateArgs(
            const DeferTraceUpdateArgs& args, RtUserEvent deferral,
            IndexSpaceExpression* expr);
        void execute(void) const;
      public:
        ShardedPhysicalTemplate* target;
        UpdateKind kind;
        RtUserEvent done;
        UniqueInst inst;
        IndexSpaceExpression* expr;
        size_t buffer_size;
        void* buffer;
        RtUserEvent deferral_event;
      };
    public:
      ShardedPhysicalTemplate(
          PhysicalTrace* trace, ApEvent fence_event,
          ReplicateContext* repl_ctx);
      ShardedPhysicalTemplate(const ShardedPhysicalTemplate& rhs) = delete;
      virtual ~ShardedPhysicalTemplate(void);
    public:
      inline RtEvent chain_deferral_events(RtUserEvent deferral_event)
      {
        RtEvent continuation_pre;
        continuation_pre.id =
            next_deferral_precondition.exchange(deferral_event.id);
        return continuation_pre;
      }
    public:
      virtual void pack_recorder(Serializer& rez) override;
      virtual size_t get_sharded_template_index(void) const override
      {
        return template_index;
      }
      virtual void initialize_replay(
          ApEvent fence_completion, bool recurrent) override;
      virtual void start_replay(void) override;
      virtual RtEvent refresh_managed_barriers(void) override;
      virtual void finish_replay(
          FenceOp* op, std::set<ApEvent>& postconditions) override;
      virtual ApEvent get_completion_for_deletion(void) const override;
      virtual void record_trigger_event(
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
          std::set<RtEvent>& applied) override;
      using PhysicalTemplate::record_merge_events;
      virtual void record_merge_events(
          ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
          const TraceLocalID& tlid) override;
      virtual void record_collective_barrier(
          ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
          size_t arrival_count) override;
      virtual ShardID record_barrier_creation(
          ApBarrier& bar, size_t total_arrivals) override;
      virtual void record_barrier_arrival(
          ApBarrier bar, ApEvent pre, size_t arrival_count,
          std::set<RtEvent>& applied, ShardID owner_shard) override;
      virtual void record_issue_copy(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          ApEvent precondition, PredEvent guard_event, LgEvent src_unique,
          LgEvent dst_unique, int priority, CollectiveKind collective,
          bool record_effect) override;
      virtual void record_issue_fill(
          const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, ApEvent precondition, PredEvent guard_event,
          LgEvent unique_event, int priority, CollectiveKind collective,
          bool record_effect) override;
      virtual void record_issue_across(
          const TraceLocalID& tlid, ApEvent& lhs,
          ApEvent collective_precondition, ApEvent copy_precondition,
          ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
          CopyAcrossExecutor* executor) override;
    public:
      virtual void record_owner_shard(
          unsigned trace_local_id, ShardID owner) override;
      virtual void record_local_space(
          unsigned trace_local_id, IndexSpace sp) override;
      virtual void record_sharding_function(
          unsigned trace_local_id, ShardingFunction* function) override;
      virtual void dump_sharded_template(void) const override;
    public:
      virtual ShardID find_owner_shard(unsigned trace_local_id) override;
      virtual IndexSpace find_local_space(unsigned trace_local_id) override;
      virtual ShardingFunction* find_sharding_function(
          unsigned trace_local_id) override;
    public:
      void prepare_collective_barrier_replay(
          const std::pair<size_t, size_t>& key, ApBarrier bar);
    public:
      ApBarrier find_trace_shard_event(ApEvent event, ShardID remote_shard);
      void record_trace_shard_event(ApEvent event, ApBarrier result);
      ApBarrier find_trace_shard_frontier(ApEvent event, ShardID remote_shard);
      void record_trace_shard_frontier(unsigned frontier, ApBarrier result);
      void handle_trace_update(Deserializer& derez, AddressSpaceID source);
      bool record_shard_event_trigger(
          ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid);
    protected:
      bool handle_update_mutated_inst(
          const UniqueInst& inst, IndexSpaceExpression* ex, Deserializer& derez,
          std::set<RtEvent>& applied, RtUserEvent done,
          const DeferTraceUpdateArgs* dargs = nullptr);
    protected:
#ifdef LEGION_DEBUG
      virtual unsigned convert_event(
          const ApEvent& event, bool check = true) override;
#endif
      virtual unsigned find_event(
          const ApEvent& event, AutoLock& tpl_lock) override;
      void request_remote_shard_event(ApEvent event, RtUserEvent done_event);
      static AddressSpaceID find_event_space(ApEvent event);
    protected:
      ShardID find_inst_owner(const UniqueInst& inst);
      void find_owner_shards(AddressSpace owner, std::vector<ShardID>& shards);
    protected:
      virtual unsigned find_frontier_event(
          ApEvent event, std::vector<RtEvent>& ready_events) override;
      virtual void record_mutated_instance(
          const UniqueInst& inst, IndexSpaceExpression* expr,
          const FieldMask& mask, std::set<RtEvent>& applied_events) override;
      virtual bool are_read_only_users(InstUsers& inst_users) override;
      virtual void sync_compute_frontiers(
          CompleteOp* op, const std::vector<RtEvent>& frontier_events) override;
      virtual void initialize_generators(
          std::vector<unsigned>& new_gen) override;
      virtual void initialize_eliminate_dead_code_frontiers(
          const std::vector<unsigned>& gen, std::vector<bool>& used) override;
      virtual void initialize_transitive_reduction_frontiers(
          std::vector<unsigned>& topo_order,
          std::vector<unsigned>& inv_topo_order) override;
      virtual void record_used_frontiers(
          std::vector<bool>& used,
          const std::vector<unsigned>& gen) const override;
      virtual void rewrite_frontiers(
          std::map<unsigned, unsigned>& substitutions) override;
    public:
      ReplicateContext* const repl_ctx;
      const ShardID local_shard;
      const size_t total_shards;
      // Make this last since it registers the template with the
      // context which can trigger calls into the template so
      // everything must valid at this point
      const size_t template_index;
    protected:
      std::map<ApEvent, RtEvent> pending_event_requests;
      // Barriers we don't managed and need to receive refreshes for
      std::map<ApEvent, BarrierAdvance*> local_advances;
      // Collective barriers from application operations
      // These will be updated by the application before each replay
      // Key is <trace local id, unique barrier name for this op>
      std::map<std::pair<size_t, size_t>, BarrierArrival*> collective_barriers;
      // Buffer up barrier updates as we're running ahead so that we can
      // apply them before we perform the trace replay
      std::map<std::pair<size_t, size_t>, ApBarrier> pending_collectives;
      std::map<AddressSpaceID, std::vector<ShardID> > did_shard_owners;
      std::map<unsigned /*Trace Local ID*/, ShardID> owner_shards;
      std::map<unsigned /*Trace Local ID*/, IndexSpace> local_spaces;
      std::map<unsigned /*Trace Local ID*/, ShardingFunction*>
          sharding_functions;
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
      std::map<unsigned, ApBarrier> local_frontiers;
      // Remote shards that are subscribed to our local frontiers
      std::map<unsigned, std::set<ShardID> > local_subscriptions;
      // Remote frontiers records barriers that we should fill in as
      // events from remote shards
      std::vector<std::pair<ApBarrier, unsigned> > remote_frontiers;
      // Pending refreshes from remote nodes
      std::map<ApBarrier, ApBarrier> pending_refresh_frontiers;
      std::map<ApEvent, ApBarrier> pending_refresh_barriers;
      std::map<TraceLocalID, std::vector<std::pair<Color, RtBarrier> > >
          pending_concurrent_barriers;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_SHARD_TEMPLATE_H__
