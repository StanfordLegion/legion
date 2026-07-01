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

#include "legion/tracing/instructions.h"
#include "legion/nodes/across.h"
#include "legion/nodes/expression.h"
#include "legion/operations/memoizable.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Instruction
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    Instruction::Instruction(PhysicalTemplate& tpl, const TraceLocalID& o)
      : owner(o)
    //--------------------------------------------------------------------------
    { }

    /////////////////////////////////////////////////////////////
    // ReplayMapping
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplayMapping::ReplayMapping(
        PhysicalTemplate& tpl, unsigned l, const TraceLocalID& r)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
    }

    //--------------------------------------------------------------------------
    void ReplayMapping::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(operations.find(owner) != operations.end());
      legion_assert(operations.find(owner)->second != nullptr);
      events[lhs] = operations[owner]->replay_mapping();
    }

    //--------------------------------------------------------------------------
    std::string ReplayMapping::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
      legion_assert(finder != memo_entries.end());
      ss << "events[" << lhs << "] = operations[" << owner
         << "].replay_mapping()    (op kind: "
         << Operation::op_names[finder->second.second] << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // CreateApUserEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CreateApUserEvent::CreateApUserEvent(
        PhysicalTemplate& tpl, unsigned l, const TraceLocalID& o)
      : Instruction(tpl, o), lhs(l)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      legion_assert(tpl.user_events.find(lhs) != tpl.user_events.end());
    }

    //--------------------------------------------------------------------------
    void CreateApUserEvent::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(user_events.find(lhs) != user_events.end());
      ApUserEvent ev = Runtime::create_ap_user_event(nullptr);
      events[lhs] = ev;
      user_events[lhs] = ev;
    }

    //--------------------------------------------------------------------------
    std::string CreateApUserEvent::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::create_ap_user_event()    "
         << "(owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // TriggerEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TriggerEvent::TriggerEvent(
        PhysicalTemplate& tpl, unsigned l, unsigned r, const TraceLocalID& o)
      : Instruction(tpl, o), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      legion_assert(rhs < tpl.events.size());
    }

    //--------------------------------------------------------------------------
    void TriggerEvent::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(events[lhs].exists());
      legion_assert(user_events[lhs].exists());
      legion_assert(events[lhs].id == user_events[lhs].id);
      Runtime::trigger_event_untraced(user_events[lhs], events[rhs]);
    }

    //--------------------------------------------------------------------------
    std::string TriggerEvent::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "Runtime::trigger_event(events[" << lhs << "], events[" << rhs
         << "])    (owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // MergeEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MergeEvent::MergeEvent(
        PhysicalTemplate& tpl, unsigned l, const std::set<unsigned>& r,
        const TraceLocalID& o)
      : Instruction(tpl, o), lhs(l), rhs(r)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      legion_assert(rhs.size() > 0);
#ifdef LEGION_DEBUG
      for (const unsigned& it : rhs)
      {
        legion_assert(it < tpl.events.size());
      }
#endif
    }

    //--------------------------------------------------------------------------
    void MergeEvent::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> to_merge;
      to_merge.reserve(rhs.size());
      for (const unsigned& it : rhs)
      {
        legion_assert(it < events.size());
        to_merge.emplace_back(events[it]);
      }
      ApEvent result = Runtime::merge_events(nullptr, to_merge);
      events[lhs] = result;
    }

    //--------------------------------------------------------------------------
    std::string MergeEvent::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::merge_events(";
      unsigned count = 0;
      for (const unsigned& it : rhs)
      {
        if (count++ != 0)
          ss << ",";
        ss << "events[" << it << "]";
      }
      ss << ")    (owner: " << owner << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // AssignFenceCompletion
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AssignFenceCompletion::AssignFenceCompletion(
        PhysicalTemplate& t, unsigned l, const TraceLocalID& o)
      : Instruction(t, o), lhs(l)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void AssignFenceCompletion::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      // This is a no-op since it gets assigned during initialize replay
    }

    //--------------------------------------------------------------------------
    std::string AssignFenceCompletion::to_string(
        const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fence_completion";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueCopy
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueCopy::IssueCopy(
        PhysicalTemplate& tpl, unsigned l, IndexSpaceExpression* e,
        const TraceLocalID& key, const std::vector<CopySrcDstField>& s,
        const std::vector<CopySrcDstField>& d,
        const std::vector<Reservation>& r, RegionTreeID src_tid,
        RegionTreeID dst_tid, unsigned pi, LgEvent src_uni, LgEvent dst_uni,
        int pr, CollectiveKind collect, bool effect)
      : Instruction(tpl, key), lhs(l), expr(e), src_fields(s), dst_fields(d),
        reservations(r), src_tree_id(src_tid), dst_tree_id(dst_tid),
        precondition_idx(pi), src_unique(src_uni), dst_unique(dst_uni),
        priority(pr), collective(collect), record_effect(effect)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      legion_assert(src_fields.size() > 0);
      legion_assert(dst_fields.size() > 0);
      legion_assert(precondition_idx < tpl.events.size());
      legion_assert(expr != nullptr);
      expr->add_base_expression_reference(TRACE_REF);
    }

    //--------------------------------------------------------------------------
    IssueCopy::~IssueCopy(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(TRACE_REF))
        delete expr;
    }

    //--------------------------------------------------------------------------
    void IssueCopy::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      std::map<TraceLocalID, MemoizableOp*>::const_iterator finder =
          operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner;
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
      legion_assert(finder != operations.end());
      legion_assert(finder->second != nullptr);
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = expr->issue_copy(
          finder->second, trace_info, dst_fields, src_fields, reservations,
          src_tree_id, dst_tree_id, precondition, PredEvent::NO_PRED_EVENT,
          src_unique, dst_unique, collective, record_effect, priority,
          true /*replay*/);
    }

    //--------------------------------------------------------------------------
    std::string IssueCopy::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = copy(operations[" << owner << "], "
         << "Index expr: " << expr->expr_id << ", {";
      for (unsigned idx = 0; idx < src_fields.size(); ++idx)
      {
        ss << "(" << std::hex << src_fields[idx].inst.id << "," << std::dec
           << src_fields[idx].subfield_offset << "," << src_fields[idx].size
           << "," << src_fields[idx].field_id << ","
           << src_fields[idx].serdez_id << ")";
        if (idx != src_fields.size() - 1)
          ss << ",";
      }
      ss << "}, {";
      for (unsigned idx = 0; idx < dst_fields.size(); ++idx)
      {
        ss << "(" << std::hex << dst_fields[idx].inst.id << "," << std::dec
           << dst_fields[idx].subfield_offset << "," << dst_fields[idx].size
           << "," << dst_fields[idx].field_id << ","
           << dst_fields[idx].serdez_id << ")";
        if (idx != dst_fields.size() - 1)
          ss << ",";
      }
      ss << "}, events[" << precondition_idx << "]";
      ss << ")";

      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueAcross
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueAcross::IssueAcross(
        PhysicalTemplate& tpl, unsigned l, unsigned copy, unsigned collective,
        unsigned src_indirect, unsigned dst_indirect, const TraceLocalID& key,
        CopyAcrossExecutor* exec)
      : Instruction(tpl, key), lhs(l), copy_precondition(copy),
        collective_precondition(collective),
        src_indirect_precondition(src_indirect),
        dst_indirect_precondition(dst_indirect), executor(exec)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      executor->add_reference();
    }

    //--------------------------------------------------------------------------
    IssueAcross::~IssueAcross(void)
    //--------------------------------------------------------------------------
    {
      if (executor->remove_reference())
        delete executor;
    }

    //--------------------------------------------------------------------------
    void IssueAcross::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      std::map<TraceLocalID, MemoizableOp*>::const_iterator finder =
          operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner;
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
      legion_assert(finder != operations.end());
      legion_assert(finder->second != nullptr);
      ApEvent copy_pre = events[copy_precondition];
      ApEvent src_indirect_pre = events[src_indirect_precondition];
      ApEvent dst_indirect_pre = events[dst_indirect_precondition];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = executor->execute(
          finder->second, PredEvent::NO_PRED_EVENT, copy_pre, src_indirect_pre,
          dst_indirect_pre, trace_info, true /*replay*/, recurrent_replay);
    }

    //--------------------------------------------------------------------------
    std::string IssueAcross::to_string(const MemoEntries& memo_entires)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = indirect(operations[" << owner << "], "
         << "Copy Across Executor: " << executor << ", {";
      ss << ", TODO: indirections";
      ss << "}, events[" << copy_precondition << "]";
      ss << ", events[" << collective_precondition << "]";
      ss << ", events[" << src_indirect_precondition << "]";
      ss << ", events[" << dst_indirect_precondition << "]";
      ss << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // IssueFill
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IssueFill::IssueFill(
        PhysicalTemplate& tpl, unsigned l, IndexSpaceExpression* e,
        const TraceLocalID& key, const std::vector<CopySrcDstField>& f,
        const void* value, size_t size, UniqueID uid, FieldSpace h,
        RegionTreeID tid, unsigned pi, LgEvent unique, int pr,
        CollectiveKind collect, bool effect)
      : Instruction(tpl, key), lhs(l), expr(e), fields(f), fill_size(size),
        fill_uid(uid), handle(h), tree_id(tid), precondition_idx(pi),
        unique_event(unique), priority(pr), collective(collect),
        record_effect(effect)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      legion_assert(fields.size() > 0);
      legion_assert(precondition_idx < tpl.events.size());
      expr->add_base_expression_reference(TRACE_REF);
      fill_value = malloc(fill_size);
      memcpy(fill_value, value, fill_size);
    }

    //--------------------------------------------------------------------------
    IssueFill::~IssueFill(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(TRACE_REF))
        delete expr;
      free(fill_value);
    }

    //--------------------------------------------------------------------------
    void IssueFill::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      std::map<TraceLocalID, MemoizableOp*>::const_iterator finder =
          operations.find(owner);
      if (finder == operations.end())
      {
        // Remote copy, should still be able to find the owner op here
        TraceLocalID local = owner;
        local.index_point = DomainPoint();
        finder = operations.find(local);
      }
      legion_assert(finder != operations.end());
      legion_assert(finder->second != nullptr);
      ApEvent precondition = events[precondition_idx];
      const PhysicalTraceInfo trace_info(finder->second, -1U);
      events[lhs] = expr->issue_fill(
          finder->second, trace_info, fields, fill_value, fill_size, fill_uid,
          handle, tree_id, precondition, PredEvent::NO_PRED_EVENT, unique_event,
          collective, record_effect, priority, true /*replay*/);
    }

    //--------------------------------------------------------------------------
    std::string IssueFill::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = fill(Index expr: " << expr->expr_id
         << ", {";
      for (unsigned idx = 0; idx < fields.size(); ++idx)
      {
        ss << "(" << std::hex << fields[idx].inst.id << "," << std::dec
           << fields[idx].subfield_offset << "," << fields[idx].size << ","
           << fields[idx].field_id << "," << fields[idx].serdez_id << ")";
        if (idx != fields.size() - 1)
          ss << ",";
      }
      ss << "}, events[" << precondition_idx << "])    (owner: " << owner
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // SetOpSyncEvent
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SetOpSyncEvent::SetOpSyncEvent(
        PhysicalTemplate& tpl, unsigned l, const TraceLocalID& r)
      : Instruction(tpl, r), lhs(l)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
    }

    //--------------------------------------------------------------------------
    void SetOpSyncEvent::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(operations.find(owner) != operations.end());
      legion_assert(operations.find(owner)->second != nullptr);
      MemoizableOp* memoizable = operations[owner];
      legion_assert(memoizable != nullptr);
      TraceInfo info(memoizable);
      ApEvent sync_condition = memoizable->compute_sync_precondition(info);
      events[lhs] = sync_condition;
    }

    //--------------------------------------------------------------------------
    std::string SetOpSyncEvent::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
      legion_assert(finder != memo_entries.end());
      ss << "events[" << lhs << "] = operations[" << owner
         << "].compute_sync_precondition()    (op kind: "
         << Operation::op_names[finder->second.second] << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // CompleteReplay
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CompleteReplay::CompleteReplay(
        PhysicalTemplate& tpl, const TraceLocalID& l, unsigned c)
      : Instruction(tpl, l), complete(c)
    //--------------------------------------------------------------------------
    {
      legion_assert(complete < tpl.events.size());
    }

    //--------------------------------------------------------------------------
    void CompleteReplay::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(operations.find(owner) != operations.end());
      legion_assert(operations.find(owner)->second != nullptr);
      MemoizableOp* memoizable = operations[owner];
      legion_assert(memoizable != nullptr);
      memoizable->complete_replay(events[complete]);
    }

    //--------------------------------------------------------------------------
    std::string CompleteReplay::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      MemoEntries::const_iterator finder = memo_entries.find(owner);
      legion_assert(finder != memo_entries.end());
      ss << "operations[" << owner << "].complete_replay(events[" << complete
         << "])    (op kind: " << Operation::op_names[finder->second.second]
         << ")";
      return ss.str();
    }

    /////////////////////////////////////////////////////////////
    // BarrierArrival
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BarrierArrival::BarrierArrival(
        PhysicalTemplate& tpl, ApBarrier bar, unsigned _lhs, unsigned _rhs,
        size_t arrivals, bool manage)
      : Instruction(tpl, TraceLocalID(0, DomainPoint())), barrier(bar),
        lhs(_lhs), rhs(_rhs), total_arrivals(arrivals), managed(manage)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      if (managed)
        Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    void BarrierArrival::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < events.size());
      runtime->phase_barrier_arrive(barrier, total_arrivals, events[rhs]);
      events[lhs] = barrier;
      if (managed)
        Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    std::string BarrierArrival::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::phase_barrier_arrive(" << std::hex
         << barrier.id << std::dec << ", events[";
      ss << rhs << "], managed : " << (managed ? "yes" : "no") << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    void BarrierArrival::set_collective_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
      legion_assert(!managed);
      barrier = newbar;
    }

    //--------------------------------------------------------------------------
    void BarrierArrival::set_managed_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
      legion_assert(managed);
      barrier = newbar;
    }

    /////////////////////////////////////////////////////////////
    // BarrierAdvance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BarrierAdvance::BarrierAdvance(
        PhysicalTemplate& tpl, ApBarrier bar, unsigned _lhs,
        size_t arrival_count, bool own)
      : Instruction(tpl, TraceLocalID(0, DomainPoint())), barrier(bar),
        lhs(_lhs), total_arrivals(arrival_count), owner(own)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < tpl.events.size());
      if (owner)
        Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    BarrierAdvance::~BarrierAdvance(void)
    //--------------------------------------------------------------------------
    {
      // Destroy our barrier if we're managing it
      if (owner)
        barrier.destroy_barrier();
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::execute(
        std::vector<ApEvent>& events,
        std::map<unsigned, ApUserEvent>& user_events,
        std::map<TraceLocalID, MemoizableOp*>& operations,
        const bool recurrent_replay)
    //--------------------------------------------------------------------------
    {
      legion_assert(lhs < events.size());
      events[lhs] = barrier;
      Runtime::advance_barrier(barrier);
    }

    //--------------------------------------------------------------------------
    std::string BarrierAdvance::to_string(const MemoEntries& memo_entries)
    //--------------------------------------------------------------------------
    {
      std::stringstream ss;
      ss << "events[" << lhs << "] = Runtime::barrier_advance(" << std::hex
         << barrier.id << std::dec << ")";
      return ss.str();
    }

    //--------------------------------------------------------------------------
    ApBarrier BarrierAdvance::record_subscribed_shard(ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner);
      subscribed_shards.emplace_back(remote_shard);
      return barrier;
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::refresh_barrier(
        ApEvent key,
        std::map<ShardID, std::map<ApEvent, ApBarrier> >& notifications)
    //--------------------------------------------------------------------------
    {
      legion_assert(owner);
      // Destroy the old barrier
      barrier.destroy_barrier();
      // Make the new barrier
      barrier = runtime->create_ap_barrier(total_arrivals);
      for (const ShardID& shard : subscribed_shards)
        notifications[shard][key] = barrier;
    }

    //--------------------------------------------------------------------------
    void BarrierAdvance::remote_refresh_barrier(ApBarrier newbar)
    //--------------------------------------------------------------------------
    {
      legion_assert(!owner);
      legion_assert(subscribed_shards.empty());
      barrier = newbar;
    }

  }  // namespace Internal
}  // namespace Legion
