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

#ifndef __LEGION_INSTRUCTIONS_H__
#define __LEGION_INSTRUCTIONS_H__

#include "legion/tracing/template.h"

namespace Legion {
  namespace Internal {

    enum InstructionKind {
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
      Instruction(PhysicalTemplate& tpl, const TraceLocalID& owner);
      virtual ~Instruction(void) { };
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) = 0;
      typedef std::map<TraceLocalID, std::pair<unsigned, unsigned> >
          MemoEntries;
      virtual std::string to_string(const MemoEntries& memo_entires) = 0;

      virtual InstructionKind get_kind(void) = 0;
      virtual ReplayMapping* as_replay_mapping(void) { return nullptr; }
      virtual CreateApUserEvent* as_create_ap_user_event(void)
      {
        return nullptr;
      }
      virtual TriggerEvent* as_trigger_event(void) { return nullptr; }
      virtual MergeEvent* as_merge_event(void) { return nullptr; }
      virtual AssignFenceCompletion* as_assignment_fence_completion(void)
      {
        return nullptr;
      }
      virtual IssueCopy* as_issue_copy(void) { return nullptr; }
      virtual IssueFill* as_issue_fill(void) { return nullptr; }
      virtual IssueAcross* as_issue_across(void) { return nullptr; }
      virtual SetOpSyncEvent* as_set_op_sync_event(void) { return nullptr; }
      virtual SetEffects* as_set_effects(void) { return nullptr; }
      virtual CompleteReplay* as_complete_replay(void) { return nullptr; }
      virtual BarrierArrival* as_barrier_arrival(void) { return nullptr; }
      virtual BarrierAdvance* as_barrier_advance(void) { return nullptr; }
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
      ReplayMapping(
          PhysicalTemplate& tpl, unsigned lhs, const TraceLocalID& rhs);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return REPLAY_MAPPING; }
      virtual ReplayMapping* as_replay_mapping(void) override { return this; }
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
      CreateApUserEvent(
          PhysicalTemplate& tpl, unsigned lhs, const TraceLocalID& owner);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return CREATE_AP_USER_EVENT;
      }
      virtual CreateApUserEvent* as_create_ap_user_event(void) override
      {
        return this;
      }
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
      TriggerEvent(
          PhysicalTemplate& tpl, unsigned lhs, unsigned rhs,
          const TraceLocalID& owner);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return TRIGGER_EVENT; }
      virtual TriggerEvent* as_trigger_event(void) override { return this; }
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
      MergeEvent(
          PhysicalTemplate& tpl, unsigned lhs, const std::set<unsigned>& rhs,
          const TraceLocalID& owner);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return MERGE_EVENT; }
      virtual MergeEvent* as_merge_event(void) override { return this; }
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
    public:
      AssignFenceCompletion(
          PhysicalTemplate& tpl, unsigned lhs, const TraceLocalID& owner);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return ASSIGN_FENCE_COMPLETION;
      }
      virtual AssignFenceCompletion* as_assignment_fence_completion(
          void) override
      {
        return this;
      }
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
      IssueFill(
          PhysicalTemplate& tpl, unsigned lhs, IndexSpaceExpression* expr,
          const TraceLocalID& op_key,
          const std::vector<CopySrcDstField>& fields, const void* fill_value,
          size_t fill_size, UniqueID fill_uid, FieldSpace handle,
          RegionTreeID tree_id, unsigned precondition_idx, LgEvent unique_event,
          int priority, CollectiveKind collective, bool record_effect);
      virtual ~IssueFill(void);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return ISSUE_FILL; }
      virtual IssueFill* as_issue_fill(void) override { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      IndexSpaceExpression* expr;
      std::vector<CopySrcDstField> fields;
      void* fill_value;
      size_t fill_size;
      UniqueID fill_uid;
      FieldSpace handle;
      RegionTreeID tree_id;
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
      IssueCopy(
          PhysicalTemplate& tpl, unsigned lhs, IndexSpaceExpression* expr,
          const TraceLocalID& op_key,
          const std::vector<CopySrcDstField>& src_fields,
          const std::vector<CopySrcDstField>& dst_fields,
          const std::vector<Reservation>& reservations,
          RegionTreeID src_tree_id, RegionTreeID dst_tree_id,
          unsigned precondition_idx, LgEvent src_unique, LgEvent dst_unique,
          int priority, CollectiveKind collective, bool record_effect);
      virtual ~IssueCopy(void);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return ISSUE_COPY; }
      virtual IssueCopy* as_issue_copy(void) override { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      IndexSpaceExpression* expr;
      std::vector<CopySrcDstField> src_fields;
      std::vector<CopySrcDstField> dst_fields;
      std::vector<Reservation> reservations;
      RegionTreeID src_tree_id;
      RegionTreeID dst_tree_id;
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
      IssueAcross(
          PhysicalTemplate& tpl, unsigned lhs, unsigned copy_pre,
          unsigned collective_pre, unsigned src_indirect_pre,
          unsigned dst_indirect_pre, const TraceLocalID& op_key,
          CopyAcrossExecutor* executor);
      virtual ~IssueAcross(void);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override { return ISSUE_ACROSS; }
      virtual IssueAcross* as_issue_across(void) override { return this; }
    private:
      friend class PhysicalTemplate;
      unsigned lhs;
      unsigned copy_precondition;
      unsigned collective_precondition;
      unsigned src_indirect_precondition;
      unsigned dst_indirect_precondition;
      CopyAcrossExecutor* const executor;
    };

    /**
     * \class SetOpSyncEvent
     * This instruction has the following semantics:
     *   events[lhs] = operations[rhs].compute_sync_precondition()
     */
    class SetOpSyncEvent : public Instruction {
    public:
      SetOpSyncEvent(
          PhysicalTemplate& tpl, unsigned lhs, const TraceLocalID& rhs);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return SET_OP_SYNC_EVENT;
      }
      virtual SetOpSyncEvent* as_set_op_sync_event(void) override
      {
        return this;
      }
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
      CompleteReplay(
          PhysicalTemplate& tpl, const TraceLocalID& lhs, unsigned complete);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return COMPLETE_REPLAY;
      }
      virtual CompleteReplay* as_complete_replay(void) override { return this; }
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
      BarrierArrival(
          PhysicalTemplate& tpl, ApBarrier bar, unsigned lhs, unsigned rhs,
          size_t arrival_count, bool managed);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return BARRIER_ARRIVAL;
      }
      virtual BarrierArrival* as_barrier_arrival(void) override { return this; }
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
      BarrierAdvance(
          PhysicalTemplate& tpl, ApBarrier bar, unsigned lhs,
          size_t arrival_count, bool owner);
      virtual ~BarrierAdvance(void);
      virtual void execute(
          std::vector<ApEvent>& events,
          std::map<unsigned, ApUserEvent>& user_events,
          std::map<TraceLocalID, MemoizableOp*>& operations,
          const bool recurrent_replay) override;
      virtual std::string to_string(const MemoEntries& memo_entires) override;

      virtual InstructionKind get_kind(void) override
      {
        return BARRIER_ADVANCE;
      }
      virtual BarrierAdvance* as_barrier_advance(void) override { return this; }
      inline ApBarrier get_current_barrier(void) const { return barrier; }
      ApBarrier record_subscribed_shard(ShardID remote_shard);
      void refresh_barrier(
          ApEvent key,
          std::map<ShardID, std::map<ApEvent, ApBarrier> >& notifications);
      void remote_refresh_barrier(ApBarrier newbar);
    private:
      friend class PhysicalTemplate;
      ApBarrier barrier;
      std::vector<ShardID> subscribed_shards;
      unsigned lhs;
      const size_t total_arrivals;
      const bool owner;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_INSTRUCTIONS_H__
