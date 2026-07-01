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

#include "legion/tracing/recording.h"
#include "legion/contexts/replicate.h"
#include "legion/instances/physical.h"
#include "legion/managers/shard.h"
#include "legion/nodes/region.h"
#include "legion/operations/memoizable.h"
#include "legion/tracing/shard.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // LogicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTraceInfo::LogicalTraceInfo(
        Operation* op, unsigned idx, const RegionRequirement& r,
        const FieldMask& mask)
      : trace(op->get_trace()), req_idx(idx), req(r),
        skip_analysis(
            (trace != nullptr) && trace->skip_analysis(r.parent.get_tree_id()))
    //--------------------------------------------------------------------------
    {
      if (!skip_analysis && (trace != nullptr) && trace->has_physical_trace())
        trace->get_physical_trace()->record_parent_req_fields(
            op->find_parent_index(idx), mask);
    }

    /////////////////////////////////////////////////////////////
    // Unique Instance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UniqueInst::UniqueInst(void)
      : inst_did(0), view_did(0), analysis_space(0), tid(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueInst::UniqueInst(IndividualView* view)
      : inst_did(view->get_manager()->did), view_did(view->did),
        analysis_space(view->get_analysis_space(view->get_manager())),
        tid(view->get_manager()->tree_id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void UniqueInst::serialize(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      legion_assert(view_did != 0);
      legion_assert(inst_did != 0);
      rez.serialize(view_did);
      rez.serialize(inst_did);
      rez.serialize(analysis_space);
      rez.serialize(tid);
    }

    //--------------------------------------------------------------------------
    void UniqueInst::deserialize(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(view_did);
      derez.deserialize(inst_did);
      derez.deserialize(analysis_space);
      derez.deserialize(tid);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID UniqueInst::get_analysis_space(void) const
    //--------------------------------------------------------------------------
    {
      return analysis_space;
    }

    /////////////////////////////////////////////////////////////
    // Remote Trace Recorder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------

    RemoteTraceRecorder::RemoteTraceRecorder(
        AddressSpaceID origin, const TraceLocalID& tlid, PhysicalTemplate* tpl,
        DistributedID did, TraceID tid)
      : origin_space(origin), remote_tpl(tpl), repl_did(did), trace_id(tid)
    //--------------------------------------------------------------------------
    {
      legion_assert(remote_tpl != nullptr);
    }

    //--------------------------------------------------------------------------
    RemoteTraceRecorder::~RemoteTraceRecorder(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::add_recorder_reference(void)
    //--------------------------------------------------------------------------
    {
      add_reference();
    }

    //--------------------------------------------------------------------------
    bool RemoteTraceRecorder::remove_recorder_reference(void)
    //--------------------------------------------------------------------------
    {
      return remove_reference();
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::pack_recorder(Serializer& rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(origin_space);
      rez.serialize(remote_tpl);
      rez.serialize(repl_did);
      if (repl_did > 0)
        rez.serialize(trace_id);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_replay_mapping(
        ApEvent lhs, unsigned op_kind, const TraceLocalID& tlid,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_REPLAY_MAPPING);
          rez.serialize(applied);
          rez.serialize(lhs);
          rez.serialize(op_kind);
          tlid.serialize(rez);
        }
        rez.dispatch(origin_space);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_replay_mapping(lhs, op_kind, tlid, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::request_term_event(ApUserEvent& term_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          !term_event.exists() || term_event.has_triggered_faultignorant());
      if (runtime->address_space != origin_space)
      {
        RtUserEvent ready = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_REQUEST_TERM_EVENT);
          rez.serialize(&term_event);
          rez.serialize(ready);
        }
        rez.dispatch(origin_space);
        // Wait for the result to be set
        ready.wait();
      }
      else
        remote_tpl->request_term_event(term_event);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_create_ap_user_event(
        ApUserEvent& lhs, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_CREATE_USER_EVENT);
          rez.serialize(done);
          rez.serialize(&lhs);
          tlid.serialize(rez);
        }
        rez.dispatch(origin_space);
        // Need this to be done before returning because we need to ensure
        // that this event is recorded before anyone tries to trigger it
        done.wait();
      }
      else
        remote_tpl->record_create_ap_user_event(lhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_trigger_event(
        ApUserEvent lhs, ApEvent rhs, const TraceLocalID& tlid,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_TRIGGER_EVENT);
          rez.serialize(applied);
          rez.serialize(lhs);
          rez.serialize(rhs);
          tlid.serialize(rez);
        }
        rez.dispatch(origin_space);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_trigger_event(lhs, rhs, tlid, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(
        ApEvent& lhs, const ApEvent* rhs, size_t num_rhs,
        const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_MERGE_EVENTS);
          rez.serialize(done);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          tlid.serialize(rez);
          rez.serialize(num_rhs);
          rez.serialize(rhs, num_rhs * sizeof(ApEvent));
        }
        rez.dispatch(origin_space);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_merge_events(lhs, rhs, num_rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(
        PredEvent& lhs, PredEvent e1, PredEvent e2, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_MERGE_PRED_EVENTS);
          rez.serialize(done);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          rez.serialize(e1);
          rez.serialize(e2);
          tlid.serialize(rez);
        }
        rez.dispatch(origin_space);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_merge_events(lhs, e1, e2, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_collective_barrier(
        ApBarrier bar, ApEvent pre, const std::pair<size_t, size_t>& key,
        size_t arrivals)
    //--------------------------------------------------------------------------
    {
      // Should be no cases where this is called remotely
      std::abort();
    }

    //--------------------------------------------------------------------------
    ShardID RemoteTraceRecorder::record_barrier_creation(
        ApBarrier& bar, size_t total_arrivals)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        ShardID owner = 0;
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_BARRIER);
          rez.serialize(done);
          rez.serialize(&bar);
          rez.serialize(total_arrivals);
          rez.serialize(&owner);
        }
        rez.dispatch(origin_space);
        done.wait();
        return owner;
      }
      else
        return remote_tpl->record_barrier_creation(bar, total_arrivals);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_barrier_arrival(
        ApBarrier bar, ApEvent pre, size_t arrivals, std::set<RtEvent>& applied,
        ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
      legion_assert(bar.exists());
      if (runtime->address_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_BARRIER_ARRIVAL);
          rez.serialize(done);
          rez.serialize(bar);
          rez.serialize(pre);
          rez.serialize(arrivals);
          rez.serialize(owner_shard);
        }
        rez.dispatch(origin_space);
        applied.insert(done);
      }
      else
        remote_tpl->record_barrier_arrival(
            bar, pre, arrivals, applied, owner_shard);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_copy(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& src_fields,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, RegionTreeID src_tree_id,
        RegionTreeID dst_tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent src_unique, LgEvent dst_unique, int priority,
        CollectiveKind collective, bool copy_restricted)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ISSUE_COPY);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
          legion_assert(src_fields.size() == dst_fields.size());
          rez.serialize<size_t>(src_fields.size());
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
          {
            pack_src_dst_field(rez, src_fields[idx]);
            pack_src_dst_field(rez, dst_fields[idx]);
          }
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
          rez.serialize(src_tree_id);
          rez.serialize(dst_tree_id);
          rez.serialize(precondition);
          rez.serialize(pred_guard);
          rez.serialize(src_unique);
          rez.serialize(dst_unique);
          rez.serialize(priority);
          rez.serialize(collective);
          rez.serialize<bool>(copy_restricted);
        }
        rez.dispatch(origin_space);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_copy(
            tlid, lhs, expr, src_fields, dst_fields, reservations, src_tree_id,
            dst_tree_id, precondition, pred_guard, src_unique, dst_unique,
            priority, collective, copy_restricted);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_copy_insts(
        ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
        unsigned dst_idx, IndexSpaceExpression* expr,
        const UniqueInst& src_inst, const UniqueInst& dst_inst,
        const FieldMask& src_mask, const FieldMask& dst_mask,
        PrivilegeMode src_mode, PrivilegeMode dst_mode, ReductionOpID redop,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_COPY_INSTS);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(lhs);
          rez.serialize(src_idx);
          rez.serialize(dst_idx);
          rez.serialize(src_mode);
          rez.serialize(dst_mode);
          expr->pack_expression(rez, origin_space);
          src_inst.serialize(rez);
          dst_inst.serialize(rez);
          rez.serialize(src_mask);
          rez.serialize(dst_mask);
          rez.serialize(redop);
        }
        rez.dispatch(origin_space);
        applied.insert(done);
      }
      else
        remote_tpl->record_copy_insts(
            lhs, tlid, src_idx, dst_idx, expr, src_inst, dst_inst, src_mask,
            dst_mask, src_mode, dst_mode, redop, applied);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_across(
        const TraceLocalID& tlid, ApEvent& lhs, ApEvent collective_precondition,
        ApEvent copy_precondition, ApEvent src_indirect_precondition,
        ApEvent dst_indirect_precondition, CopyAcrossExecutor* executor)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote indirection
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_across_insts(
        ApEvent lhs, const TraceLocalID& tlid, unsigned src_idx,
        unsigned dst_idx, IndexSpaceExpression* expr,
        const AcrossInsts& src_insts, const AcrossInsts& dst_insts,
        PrivilegeMode src_mode, PrivilegeMode dst_mode, bool src_indirect,
        bool dst_indirect, std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote across
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_indirect_insts(
        ApEvent indirect_done, ApEvent all_done, IndexSpaceExpression* expr,
        const AcrossInsts& insts, std::set<RtEvent>& applied,
        PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote indirection
      std::abort();
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_fill(
        const TraceLocalID& tlid, ApEvent& lhs, IndexSpaceExpression* expr,
        const std::vector<CopySrcDstField>& fields, const void* fill_value,
        size_t fill_size, UniqueID fill_uid, FieldSpace handle,
        RegionTreeID tree_id, ApEvent precondition, PredEvent pred_guard,
        LgEvent unique_event, int priority, CollectiveKind collective,
        bool fill_restricted)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ISSUE_FILL);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
          rez.serialize<size_t>(fields.size());
          for (unsigned idx = 0; idx < fields.size(); idx++)
            pack_src_dst_field(rez, fields[idx]);
          rez.serialize(fill_size);
          rez.serialize(fill_value, fill_size);
          rez.serialize(fill_uid);
          rez.serialize(handle);
          rez.serialize(tree_id);
          rez.serialize(precondition);
          rez.serialize(pred_guard);
          rez.serialize(unique_event);
          rez.serialize(priority);
          rez.serialize(collective);
          rez.serialize<bool>(fill_restricted);
        }
        rez.dispatch(origin_space);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_fill(
            tlid, lhs, expr, fields, fill_value, fill_size, fill_uid, handle,
            tree_id, precondition, pred_guard, unique_event, priority,
            collective, fill_restricted);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_fill_inst(
        ApEvent lhs, IndexSpaceExpression* expr, const UniqueInst& inst,
        const FieldMask& inst_mask, std::set<RtEvent>& applied_events,
        const bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_FILL_INST);
          rez.serialize(done);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
          inst.serialize(rez);
          rez.serialize(inst_mask);
          rez.serialize<bool>(reduction_initialization);
        }
        rez.dispatch(origin_space);
        applied_events.insert(done);
      }
      else
        remote_tpl->record_fill_inst(
            lhs, expr, inst, inst_mask, applied_events,
            reduction_initialization);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_op_inst(
        const TraceLocalID& tlid, unsigned parent_req_index,
        const UniqueInst& inst, RegionNode* node, const RegionUsage& usage,
        const FieldMask& user_mask, bool update_validity,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_OP_INST);
          rez.serialize(applied);
          tlid.serialize(rez);
          rez.serialize(parent_req_index);
          inst.serialize(rez);
          rez.serialize(node->handle);
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize<bool>(update_validity);
        }
        rez.dispatch(origin_space);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_op_inst(
            tlid, parent_req_index, inst, node, usage, user_mask,
            update_validity, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_set_op_sync_event(
        ApEvent& lhs, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_SET_OP_SYNC);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(&lhs);
          rez.serialize(lhs);
        }
        rez.dispatch(origin_space);
        // wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_set_op_sync_event(lhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_mapper_output(
        const TraceLocalID& tlid, const Mapper::MapTaskOutput& output,
        const std::deque<InstanceSet>& physical_instances, bool is_leaf,
        bool has_return_size, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_MAPPER_OUTPUT);
          rez.serialize(applied);
          tlid.serialize(rez);
          // We actually only need a few things here
          rez.serialize<size_t>(output.target_procs.size());
          for (unsigned idx = 0; idx < output.target_procs.size(); idx++)
            rez.serialize(output.target_procs[idx]);
          rez.serialize<size_t>(output.future_locations.size());
          for (unsigned idx = 0; idx < output.future_locations.size(); idx++)
            rez.serialize(output.future_locations[idx]);
          rez.serialize<size_t>(output.leaf_pool_bounds.size());
          for (const std::pair<const Memory, PoolBounds>& it :
               output.leaf_pool_bounds)
          {
            rez.serialize(it.first);
            rez.serialize(it.second);
          }
          rez.serialize(output.chosen_variant);
          rez.serialize(output.task_priority);
          rez.serialize<bool>(output.postmap_task);
          rez.serialize<size_t>(physical_instances.size());
          for (const InstanceSet& it : physical_instances)
            it.pack_references(rez);
          rez.serialize<bool>(is_leaf);
          rez.serialize<bool>(has_return_size);
        }
        rez.dispatch(origin_space);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_mapper_output(
            tlid, output, physical_instances, is_leaf, has_return_size,
            applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_complete_replay(
        const TraceLocalID& tlid, ApEvent pre,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_COMPLETE_REPLAY);
          rez.serialize(applied);
          tlid.serialize(rez);
          rez.serialize(pre);
        }
        rez.dispatch(origin_space);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_complete_replay(tlid, pre, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_reservations(
        const TraceLocalID& tlid,
        const std::map<Reservation, bool>& reservations,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (runtime->address_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        RemoteTraceUpdate rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ACQUIRE_RELEASE);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize<size_t>(reservations.size());
          for (const std::pair<const Reservation, bool>& reservation :
               reservations)
          {
            rez.serialize(reservation.first);
            rez.serialize<bool>(reservation.second);
          }
        }
        rez.dispatch(origin_space);
        applied_events.insert(done);
      }
      else
        remote_tpl->record_reservations(tlid, reservations, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_future_allreduce(
        const TraceLocalID& tlid, const std::vector<Memory>& target_memories,
        size_t future_size)
    //--------------------------------------------------------------------------
    {
      // should never be called on a remote node
      std::abort();
    }

    //--------------------------------------------------------------------------
    /*static*/
    PhysicalTraceRecorder* RemoteTraceRecorder::unpack_remote_recorder(
        Deserializer& derez, const TraceLocalID& tlid)
    //--------------------------------------------------------------------------
    {
      AddressSpaceID origin_space;
      derez.deserialize(origin_space);
      PhysicalTemplate* remote_tpl;
      derez.deserialize(remote_tpl);
      DistributedID did;
      derez.deserialize(did);
      TraceID trace_id = 0;
      if (did > 0)
      {
        derez.deserialize(trace_id);
        ShardManager* manager =
            runtime->find_shard_manager(did, true /*can fail*/);
        if (manager != nullptr)
        {
          ReplicateContext* ctx = manager->find_local_context();
          return ctx->find_current_shard_template(trace_id);
        }
      }
      return new RemoteTraceRecorder(
          origin_space, tlid, remote_tpl, did, trace_id);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::handle_remote_update(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalTemplate* tpl;
      derez.deserialize(tpl);
      RemoteTraceKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case REMOTE_TRACE_RECORD_REPLAY_MAPPING:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApEvent lhs;
            derez.deserialize(lhs);
            unsigned op_kind;
            derez.deserialize(op_kind);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            std::set<RtEvent> applied_events;
            tpl->record_replay_mapping(lhs, op_kind, tlid, applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(
                  applied, Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_REQUEST_TERM_EVENT:
          {
            ApUserEvent* target;
            derez.deserialize(target);
            RtUserEvent ready;
            derez.deserialize(ready);
            ApUserEvent result;
            tpl->request_term_event(result);
            legion_assert(result.exists());
            RemoteTraceResponse rez;
            {
              RezCheck z2(rez);
              rez.serialize(REMOTE_TRACE_REQUEST_TERM_EVENT);
              rez.serialize(target);
              rez.serialize(result);
              rez.serialize(ready);
            }
            rez.dispatch(source);
            break;
          }
        case REMOTE_TRACE_CREATE_USER_EVENT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApUserEvent* target;
            derez.deserialize(target);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent result;
            tpl->record_create_ap_user_event(result, tlid);
            legion_assert(result.exists());
            RemoteTraceResponse rez;
            {
              RezCheck z2(rez);
              rez.serialize(REMOTE_TRACE_CREATE_USER_EVENT);
              rez.serialize(target);
              rez.serialize(result);
              rez.serialize(applied);
            }
            rez.dispatch(source);
            break;
          }
        case REMOTE_TRACE_TRIGGER_EVENT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            ApEvent rhs;
            derez.deserialize(rhs);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            std::set<RtEvent> applied_events;
            tpl->record_trigger_event(lhs, rhs, tlid, applied_events);
            Runtime::trigger_event(
                applied, Runtime::merge_events(applied_events));
            break;
          }
        case REMOTE_TRACE_MERGE_EVENTS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            ApEvent* event_ptr;
            derez.deserialize(event_ptr);
            ApEvent lhs;
            derez.deserialize(lhs);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            size_t num_rhs;
            derez.deserialize(num_rhs);
            const ApEvent lhs_copy = lhs;
            const ApEvent* rhs = (const ApEvent*)derez.get_current_pointer();
            tpl->record_merge_events(lhs, rhs, num_rhs, tlid);
            derez.advance_pointer(num_rhs * sizeof(ApEvent));
            if (lhs != lhs_copy)
            {
              RemoteTraceResponse rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_MERGE_EVENTS);
                rez.serialize(event_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              rez.dispatch(source);
            }
            else  // didn't change so just trigger
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_MERGE_PRED_EVENTS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            PredEvent* event_ptr;
            derez.deserialize(event_ptr);
            PredEvent lhs, e1, e2;
            derez.deserialize(lhs);
            derez.deserialize(e1);
            derez.deserialize(e2);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            PredEvent lhs_copy = lhs;
            tpl->record_merge_events(lhs_copy, e1, e2, tlid);
            if (lhs != lhs_copy)
            {
              RemoteTraceResponse rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_MERGE_PRED_EVENTS);
                rez.serialize(event_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              rez.dispatch(source);
            }
            else  // didn't change so just trigger
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_ISSUE_COPY:
          {
            RtUserEvent done;
            derez.deserialize(done);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent* lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            IndexSpaceExpression* expr =
                IndexSpaceExpression::unpack_expression(derez, source);
            size_t num_fields;
            derez.deserialize(num_fields);
            std::vector<CopySrcDstField> src_fields(num_fields);
            std::vector<CopySrcDstField> dst_fields(num_fields);
            for (unsigned idx = 0; idx < num_fields; idx++)
            {
              unpack_src_dst_field(derez, src_fields[idx]);
              unpack_src_dst_field(derez, dst_fields[idx]);
            }
            size_t num_reservations;
            derez.deserialize(num_reservations);
            std::vector<Reservation> reservations(num_reservations);
            for (unsigned idx = 0; idx < num_reservations; idx++)
              derez.deserialize(reservations[idx]);
            RegionTreeID src_tree_id, dst_tree_id;
            derez.deserialize(src_tree_id);
            derez.deserialize(dst_tree_id);
            ApEvent precondition;
            derez.deserialize(precondition);
            PredEvent pred_guard;
            derez.deserialize(pred_guard);
            LgEvent src_unique, dst_unique;
            derez.deserialize(src_unique);
            derez.deserialize(dst_unique);
            int priority;
            derez.deserialize(priority);
            CollectiveKind collective;
            derez.deserialize(collective);
            bool copy_restricted;
            derez.deserialize<bool>(copy_restricted);
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs;
            // Do the base call
            tpl->record_issue_copy(
                tlid, lhs, expr, src_fields, dst_fields, reservations,
                src_tree_id, dst_tree_id, precondition, pred_guard, src_unique,
                dst_unique, priority, collective, copy_restricted);
            if (lhs != lhs_copy)
            {
              RemoteTraceResponse rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_ISSUE_COPY);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              rez.dispatch(source);
            }
            else  // lhs was unchanged
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_COPY_INSTS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            unsigned src_idx, dst_idx;
            derez.deserialize(src_idx);
            derez.deserialize(dst_idx);
            PrivilegeMode src_mode, dst_mode;
            derez.deserialize(src_mode);
            derez.deserialize(dst_mode);
            IndexSpaceExpression* expr =
                IndexSpaceExpression::unpack_expression(derez, source);
            UniqueInst src_inst, dst_inst;
            src_inst.deserialize(derez);
            dst_inst.deserialize(derez);
            FieldMask src_mask, dst_mask;
            derez.deserialize(src_mask);
            derez.deserialize(dst_mask);
            ReductionOpID redop;
            derez.deserialize(redop);
            std::set<RtEvent> ready_events;
            tpl->record_copy_insts(
                lhs, tlid, src_idx, dst_idx, expr, src_inst, dst_inst, src_mask,
                dst_mask, src_mode, dst_mode, redop, ready_events);
            if (!ready_events.empty())
              Runtime::trigger_event(done, Runtime::merge_events(ready_events));
            else
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_ISSUE_FILL:
          {
            RtUserEvent done;
            derez.deserialize(done);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent* lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            IndexSpaceExpression* expr =
                IndexSpaceExpression::unpack_expression(derez, source);
            size_t num_fields;
            derez.deserialize(num_fields);
            std::vector<CopySrcDstField> fields(num_fields);
            for (unsigned idx = 0; idx < num_fields; idx++)
              unpack_src_dst_field(derez, fields[idx]);
            size_t fill_size;
            derez.deserialize(fill_size);
            const void* fill_value = derez.get_current_pointer();
            derez.advance_pointer(fill_size);
            UniqueID fill_uid;
            derez.deserialize(fill_uid);
            FieldSpace handle;
            derez.deserialize(handle);
            RegionTreeID tree_id;
            derez.deserialize(tree_id);
            ApEvent precondition;
            derez.deserialize(precondition);
            PredEvent pred_guard;
            derez.deserialize(pred_guard);
            LgEvent unique_event;
            derez.deserialize(unique_event);
            int priority;
            derez.deserialize(priority);
            CollectiveKind collective;
            derez.deserialize(collective);
            bool fill_restricted;
            derez.deserialize<bool>(fill_restricted);
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs;
            // Do the base call
            tpl->record_issue_fill(
                tlid, lhs, expr, fields, fill_value, fill_size, fill_uid,
                handle, tree_id, precondition, pred_guard, unique_event,
                priority, collective, fill_restricted);
            if (lhs != lhs_copy)
            {
              RemoteTraceResponse rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_ISSUE_FILL);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              rez.dispatch(source);
            }
            else  // lhs was unchanged
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_FILL_INST:
          {
            RtUserEvent done;
            derez.deserialize(done);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            IndexSpaceExpression* expr =
                IndexSpaceExpression::unpack_expression(derez, source);
            UniqueInst inst;
            inst.deserialize(derez);
            FieldMask inst_mask;
            derez.deserialize(inst_mask);
            bool reduction_initialization;
            derez.deserialize<bool>(reduction_initialization);
            std::set<RtEvent> ready_events;
            tpl->record_fill_inst(
                lhs, expr, inst, inst_mask, ready_events,
                reduction_initialization);
            if (!ready_events.empty())
              Runtime::trigger_event(done, Runtime::merge_events(ready_events));
            else
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_RECORD_OP_INST:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            unsigned index;
            derez.deserialize(index);
            UniqueInst inst;
            inst.deserialize(derez);
            LogicalRegion handle;
            derez.deserialize(handle);
            RegionUsage usage;
            derez.deserialize(usage);
            FieldMask user_mask;
            derez.deserialize(user_mask);
            bool update_validity;
            derez.deserialize<bool>(update_validity);
            RegionNode* node = runtime->get_node(handle);
            std::set<RtEvent> effects;
            tpl->record_op_inst(
                tlid, index, inst, node, usage, user_mask, update_validity,
                effects);
            if (!effects.empty())
              Runtime::trigger_event(applied, Runtime::merge_events(effects));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_SET_OP_SYNC:
          {
            RtUserEvent done;
            derez.deserialize(done);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent* lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            const ApUserEvent lhs_copy = lhs;
            tpl->record_set_op_sync_event(lhs, tlid);
            if (lhs != lhs_copy)
            {
              RemoteTraceResponse rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_SET_OP_SYNC);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              rez.dispatch(source);
            }
            else  // lhs didn't change
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_RECORD_MAPPER_OUTPUT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            size_t num_target_processors;
            derez.deserialize(num_target_processors);
            Mapper::MapTaskOutput output;
            output.target_procs.resize(num_target_processors);
            for (unsigned idx = 0; idx < num_target_processors; idx++)
              derez.deserialize(output.target_procs[idx]);
            size_t num_future_locations;
            derez.deserialize(num_future_locations);
            if (num_future_locations > 0)
            {
              output.future_locations.resize(num_future_locations);
              for (unsigned idx = 0; idx < num_future_locations; idx++)
                derez.deserialize(output.future_locations[idx]);
            }
            size_t num_pool_bounds;
            derez.deserialize(num_pool_bounds);
            for (unsigned idx = 0; idx < num_pool_bounds; idx++)
            {
              Memory memory;
              derez.deserialize(memory);
              derez.deserialize(output.leaf_pool_bounds[memory]);
            }
            derez.deserialize(output.chosen_variant);
            derez.deserialize(output.task_priority);
            derez.deserialize<bool>(output.postmap_task);
            size_t num_phy_instances;
            derez.deserialize(num_phy_instances);
            std::deque<InstanceSet> physical_instances(num_phy_instances);
            std::set<RtEvent> ready_events;
            for (unsigned idx = 0; idx < num_phy_instances; idx++)
              physical_instances[idx].unpack_references(derez, ready_events);
            bool is_leaf, has_return_size;
            derez.deserialize<bool>(is_leaf);
            derez.deserialize<bool>(has_return_size);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }

            std::set<RtEvent> applied_events;
            tpl->record_mapper_output(
                tlid, output, physical_instances, is_leaf, has_return_size,
                applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(
                  applied, Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_COMPLETE_REPLAY:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApEvent pre;
            derez.deserialize(pre);
            std::set<RtEvent> applied_events;
            tpl->record_complete_replay(tlid, pre, applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(
                  applied, Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_ACQUIRE_RELEASE:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            size_t num_reservations;
            derez.deserialize(num_reservations);
            std::map<Reservation, bool> reservations;
            for (unsigned idx = 0; idx < num_reservations; idx++)
            {
              Reservation reservation;
              derez.deserialize(reservation);
              derez.deserialize<bool>(reservations[reservation]);
            }
            std::set<RtEvent> applied_events;
            tpl->record_reservations(tlid, reservations, applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(
                  applied, Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_RECORD_BARRIER:
          {
            RtUserEvent done_event;
            derez.deserialize(done_event);
            ApBarrier* barrier;
            derez.deserialize(barrier);
            size_t arrivals;
            derez.deserialize(arrivals);
            ShardID* target;
            derez.deserialize(target);
            ApBarrier bar;
            ShardID owner = tpl->record_barrier_creation(bar, arrivals);
            RemoteTraceResponse rez;
            {
              RezCheck z2(rez);
              rez.serialize(REMOTE_TRACE_RECORD_BARRIER);
              rez.serialize(barrier);
              rez.serialize(bar);
              rez.serialize(target);
              rez.serialize(owner);
              rez.serialize(done_event);
            }
            rez.dispatch(source);
            break;
          }
        case REMOTE_TRACE_BARRIER_ARRIVAL:
          {
            RtUserEvent done_event;
            derez.deserialize(done_event);
            ApBarrier barrier;
            derez.deserialize(barrier);
            ApEvent pre;
            derez.deserialize(pre);
            size_t arrivals;
            derez.deserialize(arrivals);
            ShardID owner;
            derez.deserialize(owner);
            std::set<RtEvent> applied;
            tpl->record_barrier_arrival(barrier, pre, arrivals, applied, owner);
            if (!applied.empty())
              Runtime::trigger_event(
                  done_event, Runtime::merge_events(applied));
            else
              Runtime::trigger_event(done_event);
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceUpdate::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      RemoteTraceRecorder::handle_remote_update(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::handle_remote_response(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RemoteTraceKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case REMOTE_TRACE_REQUEST_TERM_EVENT:
        case REMOTE_TRACE_CREATE_USER_EVENT:
          {
            ApUserEvent* event_ptr;
            derez.deserialize(event_ptr);
            derez.deserialize(*event_ptr);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_MERGE_EVENTS:
        case REMOTE_TRACE_ISSUE_COPY:
        case REMOTE_TRACE_ISSUE_FILL:
        case REMOTE_TRACE_SET_OP_SYNC:
          {
            ApEvent* event_ptr;
            derez.deserialize(event_ptr);
            derez.deserialize(*event_ptr);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_MERGE_PRED_EVENTS:
          {
            PredEvent* event_ptr;
            derez.deserialize(event_ptr);
            derez.deserialize(*event_ptr);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_RECORD_BARRIER:
          {
            ApBarrier* barrier;
            derez.deserialize(barrier);
            derez.deserialize(*barrier);
            ShardID* target;
            derez.deserialize(target);
            derez.deserialize(*target);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        default:
          std::abort();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      RemoteTraceRecorder::handle_remote_response(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::pack_src_dst_field(
        Serializer& rez, const CopySrcDstField& field)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(field.inst);
      rez.serialize(field.field_id);
      rez.serialize(field.size);
      rez.serialize(field.redop_id);
      rez.serialize<bool>(field.red_fold);
      rez.serialize(field.serdez_id);
      rez.serialize(field.subfield_offset);
      rez.serialize(field.indirect_index);
      rez.serialize(field.fill_data.indirect);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::unpack_src_dst_field(
        Deserializer& derez, CopySrcDstField& field)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(field.inst);
      derez.deserialize(field.field_id);
      derez.deserialize(field.size);
      derez.deserialize(field.redop_id);
      derez.deserialize<bool>(field.red_fold);
      derez.deserialize(field.serdez_id);
      derez.deserialize(field.subfield_offset);
      derez.deserialize(field.indirect_index);
      derez.deserialize(field.fill_data.indirect);
    }

    /////////////////////////////////////////////////////////////
    // TraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(Operation* op)
      : rec(init_recorder(op)), tlid(init_tlid(op)),
        recording((rec == nullptr) ? false : rec->is_recording())
    //--------------------------------------------------------------------------
    {
      if (rec != nullptr)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceRecorder* TraceInfo::init_recorder(Operation* op)
    //--------------------------------------------------------------------------
    {
      if (op == nullptr)
        return nullptr;
      MemoizableOp* memo = op->get_memoizable();
      if (memo == nullptr)
        return nullptr;
      return memo->get_template();
    }

    //--------------------------------------------------------------------------
    /*static*/ TraceLocalID TraceInfo::init_tlid(Operation* op)
    //--------------------------------------------------------------------------
    {
      if (op == nullptr)
        return TraceLocalID();
      MemoizableOp* memo = op->get_memoizable();
      if (memo == nullptr)
        return TraceLocalID();
      return memo->get_trace_local_id();
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(SingleTask* task, RemoteTraceRecorder* r)
      : rec(r), tlid(task->get_trace_local_id()), recording(rec != nullptr)
    //--------------------------------------------------------------------------
    {
      if (recording)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(const TraceInfo& rhs)
      : rec(rhs.rec), tlid(rhs.tlid), recording(rhs.recording)
    //--------------------------------------------------------------------------
    {
      if (rec != nullptr)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(PhysicalTraceRecorder* r, const TraceLocalID& tld)
      : rec(r), tlid(tld), recording((r != nullptr) && r->is_recording())
    //--------------------------------------------------------------------------
    {
      if (rec != nullptr)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    TraceInfo::~TraceInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((rec != nullptr) && rec->remove_recorder_reference())
        delete rec;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(Operation* o, unsigned idx)
      : TraceInfo(o), index(idx), dst_index(idx), update_validity(true)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(
        const TraceInfo& info, unsigned idx, bool update /*=true*/)
      : TraceInfo(info), index(idx), dst_index(idx), update_validity(update)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(
        unsigned src_idx, const TraceInfo& info, unsigned dst_idx)
      : TraceInfo(info), index(src_idx), dst_index(dst_idx),
        update_validity(true)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(const PhysicalTraceInfo& rhs)
      : TraceInfo(rhs), index(rhs.index), dst_index(rhs.dst_index),
        update_validity(rhs.update_validity)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(
        const TraceLocalID& tlid, unsigned src_idx, unsigned dst_idx,
        bool update, PhysicalTraceRecorder* r)
      : TraceInfo(r, tlid), index(src_idx), dst_index(dst_idx),
        update_validity(update)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::record_op_inst(
        const RegionUsage& usage, const FieldMask& user_mask,
        const UniqueInst& inst, RegionNode* node, Operation* op,
        std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      sanity_check();
      rec->record_op_inst(
          tlid, op->find_parent_index(index), inst, node, usage, user_mask,
          update_validity, applied);
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::pack_trace_info(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(recording);
      if (recording)
      {
        legion_assert(rec != nullptr);
        tlid.serialize(rez);
        rez.serialize(index);
        rez.serialize(dst_index);
        rez.serialize<bool>(update_validity);
        rec->pack_recorder(rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceInfo PhysicalTraceInfo::unpack_trace_info(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      bool recording;
      derez.deserialize<bool>(recording);
      if (recording)
      {
        TraceLocalID tlid;
        tlid.deserialize(derez);
        unsigned index, dst_index;
        derez.deserialize(index);
        derez.deserialize(dst_index);
        bool update_validity;
        derez.deserialize(update_validity);
        PhysicalTraceRecorder* recorder =
            RemoteTraceRecorder::unpack_remote_recorder(derez, tlid);
        return PhysicalTraceInfo(
            tlid, index, dst_index, update_validity, recorder);
      }
      else
        return PhysicalTraceInfo(nullptr, -1U);
    }

  }  // namespace Internal
}  // namespace Legion
