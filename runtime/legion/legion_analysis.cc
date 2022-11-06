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

#include "legion.h"
#include "legion/runtime.h"
#include "legion/legion_ops.h"
#include "legion/legion_tasks.h"
#include "legion/region_tree.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_views.h"
#include "legion/legion_analysis.h"
#include "legion/legion_context.h"
#include "legion/legion_replication.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Users and Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(void)
      : GenericUser(), op(NULL), idx(0), gen(0), timeout(TIMEOUT)
#ifdef LEGION_SPY
        , uid(0)
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(Operation *o, unsigned id, const RegionUsage &u,
                             const FieldMask &m)
      : GenericUser(u, m), op(o), idx(id), 
        gen(o->get_generation()), timeout(TIMEOUT)
#ifdef LEGION_SPY
        , uid(o->get_unique_op_id())
#endif
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalUser::LogicalUser(Operation *o, GenerationID g, unsigned id, 
                             const RegionUsage &u, const FieldMask &m)
      : GenericUser(u, m), op(o), idx(id), gen(g), timeout(TIMEOUT)
#ifdef LEGION_SPY
        , uid(o->get_unique_op_id())
#endif
    //--------------------------------------------------------------------------
    {
    }

#ifdef ENABLE_VIEW_REPLICATION
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, IndexSpaceExpression *e,
                               UniqueID id, unsigned x, RtEvent collect,
                               bool cpy, bool cov)
      : usage(u), expr(e), op_id(id), index(x), collect_event(collect),
        copy_user(cpy), covers(cov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      expr->add_base_expression_reference(PHYSICAL_USER_REF);
    }
#else
    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const RegionUsage &u, IndexSpaceExpression *e,
                               UniqueID id, unsigned x, bool cpy, bool cov)
      : usage(u), expr(e), op_id(id), index(x), copy_user(cpy), covers(cov)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      expr->add_base_expression_reference(PHYSICAL_USER_REF);
    }
#endif

    //--------------------------------------------------------------------------
    PhysicalUser::PhysicalUser(const PhysicalUser &rhs) 
      : usage(rhs.usage), expr(rhs.expr), op_id(rhs.op_id), index(rhs.index),
#ifdef ENABLE_VIEW_REPLICATION
        collect_event(rhs.collect_event),
#endif
        copy_user(rhs.copy_user), covers(rhs.covers)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalUser::~PhysicalUser(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(expr != NULL);
#endif
      if (expr->remove_base_expression_reference(PHYSICAL_USER_REF))
        delete expr;
    }

    //--------------------------------------------------------------------------
    PhysicalUser& PhysicalUser::operator=(const PhysicalUser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PhysicalUser::pack_user(Serializer &rez, 
                                 const AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
#ifdef ENABLE_VIEW_REPLICATION
      rez.serialize(collect_event);
#endif
      rez.serialize(usage);
      expr->pack_expression(rez, target);
      rez.serialize(op_id);
      rez.serialize(index);
      rez.serialize<bool>(copy_user);
      rez.serialize<bool>(covers);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalUser* PhysicalUser::unpack_user(Deserializer &derez,
                          RegionTreeForest *forest, const AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
#ifdef ENABLE_VIEW_REPLICATION
      RtEvent collect_event;
      derez.deserialize(collect_event);
#endif
      RegionUsage usage;
      derez.deserialize(usage);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, forest, source);
      UniqueID op_id;
      derez.deserialize(op_id);
      unsigned index;
      derez.deserialize(index);
      bool copy_user, covers;
      derez.deserialize<bool>(copy_user);
      derez.deserialize<bool>(covers);
#ifdef ENABLE_VIEW_REPLICATION
      return new PhysicalUser(usage, expr, op_id, index, collect_event,
                              copy_user, covers);
#else
      return new PhysicalUser(usage, expr, op_id, index, copy_user, covers);
#endif
    }

    /////////////////////////////////////////////////////////////
    // VersionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(equivalence_sets.empty());
      assert(rhs.equivalence_sets.empty());
#endif
    }

    //--------------------------------------------------------------------------
    VersionInfo::~VersionInfo(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo& VersionInfo::operator=(const VersionInfo &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(equivalence_sets.empty());
      assert(rhs.equivalence_sets.empty());
#endif
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::pack_equivalence_sets(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(equivalence_sets.size());
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::unpack_equivalence_sets(Deserializer &derez, 
                              Runtime *runtime, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_sets;
      derez.deserialize(num_sets);
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask mask;
        derez.deserialize(mask);
        RtEvent ready_event;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready_event);
        equivalence_sets.insert(set, mask);
        if (ready_event.exists())
          ready_events.insert(ready_event);
      }
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_equivalence_set(EquivalenceSet *set,
                                             const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      equivalence_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      equivalence_sets.clear();
    }

    /////////////////////////////////////////////////////////////
    // LogicalTraceInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTraceInfo::LogicalTraceInfo(Operation *op, unsigned idx, 
                                       const RegionRequirement &r)
      : trace(((op->get_trace() != NULL) && 
                op->get_trace()->handles_region_tree(r.parent.get_tree_id())) ?
                op->get_trace() : NULL), req_idx(idx), req(r),
        already_traced((trace != NULL) && !op->is_tracing()),
        recording_trace((trace != NULL) && trace->is_recording()),
        replaying_trace((trace != NULL) && trace->is_replaying())
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Unique Instance
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UniqueInst::UniqueInst(InstanceView *view, DomainPoint point)
      : view_did(view->did), collective_point(point)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void UniqueInst::serialize(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(view_did != 0);
#endif
      rez.serialize(view_did);
      rez.serialize(collective_point);
    }

    //--------------------------------------------------------------------------
    void UniqueInst::deserialize(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(view_did);
      derez.deserialize(collective_point);
    }

    //--------------------------------------------------------------------------
    AddressSpaceID UniqueInst::get_analysis_space(Runtime *runtime) const
    //--------------------------------------------------------------------------
    {
      RtEvent ready;
      InstanceView *view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, ready));
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      return view->get_analysis_space(collective_point);
    }

    /////////////////////////////////////////////////////////////
    // Remote Trace Recorder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTraceRecorder::RemoteTraceRecorder(Runtime *rt, AddressSpaceID origin,
                                 AddressSpaceID local, const TraceLocalID &tlid,
                                 PhysicalTemplate *tpl, RtUserEvent applied,
                                 RtEvent collect)
      : runtime(rt), origin_space(origin), local_space(local),
        remote_tpl(tpl), applied_event(applied), collect_event(collect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_tpl != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    RemoteTraceRecorder::~RemoteTraceRecorder(void)
    //--------------------------------------------------------------------------
    {
      if (!applied_events.empty())
        Runtime::trigger_event(applied_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied_event);
    }

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
    void RemoteTraceRecorder::pack_recorder(Serializer &rez,
                                            std::set<RtEvent> &external_applied)
    //--------------------------------------------------------------------------
    {
      rez.serialize(origin_space);
      rez.serialize(remote_tpl);
      RtUserEvent remote_applied = Runtime::create_rt_user_event();
      rez.serialize(remote_applied);
      rez.serialize(collect_event);
      // Only need to store this one locally since we already hooked our whole 
      // chain of events into the operations applied set on the origin node
      // See PhysicalTemplate::pack_recorder
      AutoLock a_lock(applied_lock);
      applied_events.insert(remote_applied);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_get_term_event(ApEvent lhs,
                                     unsigned op_kind, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_GET_TERM);
          rez.serialize(applied);
          rez.serialize(lhs);
          rez.serialize(op_kind);
          tlid.serialize(rez);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_get_term_event(lhs, op_kind, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::request_term_event(ApUserEvent &term_event)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!term_event.exists() || term_event.has_triggered());
#endif
      if (local_space != origin_space)
      {
        RtUserEvent ready = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_REQUEST_TERM_EVENT);
          rez.serialize(&term_event);
          rez.serialize(ready);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait for the result to be set
        ready.wait();
      }
      else
        remote_tpl->request_term_event(term_event);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_create_ap_user_event(
                                     ApUserEvent &lhs, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_CREATE_USER_EVENT);
          rez.serialize(done);
          rez.serialize(&lhs);
          tlid.serialize(rez);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Need this to be done before returning because we need to ensure
        // that this event is recorded before anyone tries to trigger it
        done.wait();
      }
      else
        remote_tpl->record_create_ap_user_event(lhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                                   const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_TRIGGER_EVENT);
          rez.serialize(applied);
          rez.serialize(lhs);
          rez.serialize(rhs);
          tlid.serialize(rez);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_trigger_event(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent rhs,
                                                  const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(rhs);
        record_merge_events(lhs, rhs_events, tlid);
      }
      else
        remote_tpl->record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent e1,
                                           ApEvent e2, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(e1);
        rhs_events.insert(e2);
        record_merge_events(lhs, rhs_events, tlid);
      }
      else
        remote_tpl->record_merge_events(lhs, e1, e2, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent e1,
                                                  ApEvent e2, ApEvent e3,
                                                  const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(e1);
        rhs_events.insert(e2);
        rhs_events.insert(e3);
        record_merge_events(lhs, rhs_events, tlid);
      }
      else
        remote_tpl->record_merge_events(lhs, e1, e2, e3, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs,
                                                  const std::set<ApEvent>& rhs,
                                                  const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_MERGE_EVENTS);
          rez.serialize(done);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          tlid.serialize(rez);
          rez.serialize<size_t>(rhs.size());
          for (std::set<ApEvent>::const_iterator it = 
                rhs.begin(); it != rhs.end(); it++)
            rez.serialize(*it);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs,
                                                const std::vector<ApEvent>& rhs,
                                                const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_MERGE_EVENTS);
          rez.serialize(done);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          tlid.serialize(rez);
          rez.serialize<size_t>(rhs.size());
          for (std::vector<ApEvent>::const_iterator it = 
                rhs.begin(); it != rhs.end(); it++)
            rez.serialize(*it);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_merge_events(lhs, rhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_collective_barrier(ApBarrier bar, 
              ApEvent pre, const std::pair<size_t,size_t> &key, size_t arrivals)
    //--------------------------------------------------------------------------
    {
      // Should be no cases where this is called remotely
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_copy(const TraceLocalID &tlid,
                                 ApEvent &lhs, IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                             RegionTreeID src_tree_id,
                                             RegionTreeID dst_tree_id,
#endif
                                             ApEvent precondition, 
                                             PredEvent pred_guard,
                                             LgEvent src_unique,
                                             LgEvent dst_unique,
                                             int priority)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ISSUE_COPY);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
#ifdef DEBUG_LEGION
          assert(src_fields.size() == dst_fields.size());
#endif
          rez.serialize<size_t>(src_fields.size());
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
          {
            pack_src_dst_field(rez, src_fields[idx]);
            pack_src_dst_field(rez, dst_fields[idx]);
          }
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
#ifdef LEGION_SPY
          rez.serialize(src_tree_id);
          rez.serialize(dst_tree_id);
#endif
          rez.serialize(precondition);
          rez.serialize(pred_guard);
          rez.serialize(src_unique);
          rez.serialize(dst_unique);
          rez.serialize(priority);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_copy(tlid, lhs, expr, src_fields,
                              dst_fields, reservations,
#ifdef LEGION_SPY
                              src_tree_id, dst_tree_id,
#endif
                              precondition, pred_guard,
                              src_unique, dst_unique, priority);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_copy_insts(ApEvent lhs, 
                                              const TraceLocalID &tlid,
                                              unsigned src_idx,unsigned dst_idx,
                                              IndexSpaceExpression *expr,
                                              const UniqueInst &src_inst,
                                              const UniqueInst &dst_inst,
                                              const FieldMask &src_mask,
                                              const FieldMask &dst_mask,
                                              PrivilegeMode src_mode,
                                              PrivilegeMode dst_mode,
                                              ReductionOpID redop,
                                              std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
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
        runtime->send_remote_trace_update(origin_space, rez);
        applied.insert(done);
      }
      else
        remote_tpl->record_copy_insts(lhs, tlid, src_idx, dst_idx, expr,
                                 src_inst, dst_inst, src_mask, dst_mask,
                                 src_mode, dst_mode, redop, applied);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_across(const TraceLocalID &tlid,
                                              ApEvent &lhs,
                                              ApEvent collective_precondition,
                                              ApEvent copy_precondition,
                                              ApEvent src_indirect_precondition,
                                              ApEvent dst_indirect_precondition,
                                              CopyAcrossExecutor *executor)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote indirection
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_across_insts(ApEvent lhs, 
                                 const TraceLocalID &tlid,
                                 unsigned src_idx, unsigned dst_idx,
                                 IndexSpaceExpression *expr,
                                 const AcrossInsts &src_insts,
                                 const AcrossInsts &dst_insts,
                                 PrivilegeMode src_mode, PrivilegeMode dst_mode,
                                 bool src_indirect, bool dst_indirect,
                                 std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote across
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_indirect_insts(ApEvent indirect_done,
                                                    ApEvent all_done,
                                                    IndexSpaceExpression *expr,
                                                    const AcrossInsts &insts,
                                                    std::set<RtEvent> &applied,
                                                    PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
      // We should never get a call to record a remote indirection
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_fill(const TraceLocalID &tlid,
                                 ApEvent &lhs, IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                             const void *fill_value, 
                                             size_t fill_size,
#ifdef LEGION_SPY
                                             UniqueID fill_uid,
                                             FieldSpace handle,
                                             RegionTreeID tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard,
                                             LgEvent unique_event,
                                             int priority)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
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
#ifdef LEGION_SPY
          rez.serialize(fill_uid);
          rez.serialize(handle);
          rez.serialize(tree_id);
#endif
          rez.serialize(precondition);
          rez.serialize(pred_guard);  
          rez.serialize(unique_event);
          rez.serialize(priority);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_fill(tlid, lhs, expr, fields, 
                                      fill_value, fill_size, 
#ifdef LEGION_SPY
                                      fill_uid, handle, tree_id,
#endif
                                      precondition, pred_guard,
                                      unique_event, priority);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_fill_inst(ApEvent lhs,
                                 IndexSpaceExpression *expr, 
                                 const UniqueInst &inst,
                                 const FieldMask &inst_mask,
                                 std::set<RtEvent> &applied_events,
                                 const bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
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
        runtime->send_remote_trace_update(origin_space, rez);
        applied_events.insert(done);
      }
      else
        remote_tpl->record_fill_inst(lhs, expr, inst, inst_mask,
                                     applied_events, reduction_initialization);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_op_inst(const TraceLocalID &tlid,
                                             unsigned idx,
                                             const UniqueInst &inst,
                                             RegionNode *node,
                                             const RegionUsage &usage,
                                             const FieldMask &user_mask,
                                             bool update_validity,
                                             std::set<RtEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_OP_INST);
          rez.serialize(applied);
          tlid.serialize(rez);
          rez.serialize(idx);
          inst.serialize(rez);
          rez.serialize(node->handle);
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize<bool>(update_validity);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_op_inst(tlid, idx, inst, node, usage,
                                   user_mask, update_validity, effects);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_set_op_sync_event(ApEvent &lhs, 
                                                       const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_SET_OP_SYNC);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize(&lhs);
          rez.serialize(lhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_set_op_sync_event(lhs, tlid);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_mapper_output(const TraceLocalID &tlid,
                              const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances,
                              const std::vector<size_t> &future_size_bounds,
                              const std::vector<TaskTreeCoordinates> &coords,
                              std::set<RtEvent> &external_applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(output.future_locations.size() == future_size_bounds.size());
      assert(coords.size() == future_size_bounds.size());
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
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
          // Same size as the future locations
          for (unsigned idx = 0; idx < future_size_bounds.size(); idx++)
            rez.serialize(future_size_bounds[idx]);
          // Same size as the future locations
          for (unsigned idx = 0; idx < coords.size(); idx++)
          {
            const TaskTreeCoordinates &future_coordinates = coords[idx];
            rez.serialize<size_t>(future_coordinates.size());
            for (TaskTreeCoordinates::const_iterator it =
                  future_coordinates.begin(); it !=
                  future_coordinates.end(); it++)
              it->serialize(rez);
          }
          rez.serialize(output.chosen_variant);
          rez.serialize(output.task_priority);
          rez.serialize<bool>(output.postmap_task);
          rez.serialize<size_t>(physical_instances.size());
          for (std::deque<InstanceSet>::const_iterator it = 
               physical_instances.begin(); it != physical_instances.end(); it++)
            it->pack_references(rez);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_mapper_output(tlid, output, physical_instances,
                            future_size_bounds, coords, external_applied);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_set_effects(const TraceLocalID &tlid, 
                                                 ApEvent &rhs)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_SET_EFFECTS);
          rez.serialize(applied);
          tlid.serialize(rez);
          rez.serialize(rhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_set_effects(tlid, rhs);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_complete_replay(const TraceLocalID &tlid, 
                                                     ApEvent rhs)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_COMPLETE_REPLAY);
          rez.serialize(applied);
          tlid.serialize(rez);
          rez.serialize(rhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_complete_replay(tlid, rhs);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_reservations(const TraceLocalID &tlid,
                                 const std::map<Reservation,bool> &reservations,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ACQUIRE_RELEASE);
          rez.serialize(done);
          tlid.serialize(rez);
          rez.serialize<size_t>(reservations.size());
          for (std::map<Reservation,bool>::const_iterator it =
                reservations.begin(); it != reservations.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize<bool>(it->second);
          }
        }
        runtime->send_remote_trace_update(origin_space, rez);
        applied_events.insert(done);
      }
      else
        remote_tpl->record_reservations(tlid, reservations, applied_events); 
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteTraceRecorder* RemoteTraceRecorder::unpack_remote_recorder(
                Deserializer &derez, Runtime *runtime, const TraceLocalID &tlid)
    //--------------------------------------------------------------------------
    {
      AddressSpaceID origin_space;
      derez.deserialize(origin_space);
      PhysicalTemplate *remote_tpl;
      derez.deserialize(remote_tpl);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      RtEvent collect_event;
      derez.deserialize(collect_event);
      return new RemoteTraceRecorder(runtime, origin_space, 
                                     runtime->address_space, tlid,
                                     remote_tpl, applied_event, collect_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::handle_remote_update(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalTemplate *tpl;
      derez.deserialize(tpl);
      RemoteTraceKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case REMOTE_TRACE_RECORD_GET_TERM:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApEvent lhs;
            derez.deserialize(lhs);
            unsigned op_kind;
            derez.deserialize(op_kind);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            tpl->record_get_term_event(lhs, op_kind, tlid);
            Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_REQUEST_TERM_EVENT:
          {
            ApUserEvent *target;
            derez.deserialize(target);
            RtUserEvent ready;
            derez.deserialize(ready);
            ApUserEvent result;
            tpl->request_term_event(result);
#ifdef DEBUG_LEGION
            assert(result.exists());
#endif
            Serializer rez;
            {
              RezCheck z2(rez);
              rez.serialize(REMOTE_TRACE_REQUEST_TERM_EVENT);
              rez.serialize(target);
              rez.serialize(result);
              rez.serialize(ready);
            }
            runtime->send_remote_trace_response(source, rez);
            break;
          }
        case REMOTE_TRACE_CREATE_USER_EVENT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApUserEvent *target;
            derez.deserialize(target);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent result;
            tpl->record_create_ap_user_event(result, tlid);
#ifdef DEBUG_LEGION
            assert(result.exists());
#endif
            Serializer rez;
            {
              RezCheck z2(rez);
              rez.serialize(REMOTE_TRACE_CREATE_USER_EVENT);
              rez.serialize(target);
              rez.serialize(result);
              rez.serialize(applied);
            }
            runtime->send_remote_trace_response(source, rez);
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
            tpl->record_trigger_event(lhs, rhs, tlid);
            Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_MERGE_EVENTS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            ApEvent *event_ptr;
            derez.deserialize(event_ptr);
            ApEvent lhs;
            derez.deserialize(lhs);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            size_t num_rhs;
            derez.deserialize(num_rhs);
            const ApEvent lhs_copy = lhs;
            if (num_rhs == 2)
            {
              ApEvent e1, e2;
              derez.deserialize(e1);
              derez.deserialize(e2);
              tpl->record_merge_events(lhs, e1, e2, tlid);
            }
            else if (num_rhs == 3)
            {
              ApEvent e1, e2, e3;
              derez.deserialize(e1);
              derez.deserialize(e2);
              derez.deserialize(e3);
              tpl->record_merge_events(lhs, e1, e2, e3, tlid);
            }
            else
            {
              std::vector<ApEvent> rhs_events(num_rhs);
              for (unsigned idx = 0; idx < num_rhs; idx++)
              {
                ApEvent event;
                derez.deserialize(rhs_events[idx]);
              }
              tpl->record_merge_events(lhs, rhs_events, tlid);
            }
            if (lhs != lhs_copy)
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_MERGE_EVENTS);
                rez.serialize(event_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              runtime->send_remote_trace_response(source, rez);
            }
            else // didn't change so just trigger
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_ISSUE_COPY:
          {
            RtUserEvent done;
            derez.deserialize(done);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApUserEvent *lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr = 
              IndexSpaceExpression::unpack_expression(derez, forest, source);
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
#ifdef LEGION_SPY
            RegionTreeID src_tree_id, dst_tree_id;
            derez.deserialize(src_tree_id);
            derez.deserialize(dst_tree_id);
#endif
            ApEvent precondition;
            derez.deserialize(precondition);
            PredEvent pred_guard;
            derez.deserialize(pred_guard);
            LgEvent src_unique, dst_unique;
            derez.deserialize(src_unique);
            derez.deserialize(dst_unique);
            int priority;
            derez.deserialize(priority);
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs;
            // Do the base call
            tpl->record_issue_copy(tlid, lhs, expr,
                                   src_fields, dst_fields, reservations,
#ifdef LEGION_SPY
                                   src_tree_id, dst_tree_id,
#endif
                                   precondition, pred_guard,
                                   src_unique, dst_unique, priority);
            if (lhs != lhs_copy)
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_ISSUE_COPY);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              runtime->send_remote_trace_response(source, rez);
            }
            else // lhs was unchanged
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
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr =
              IndexSpaceExpression::unpack_expression(derez, forest, source);
            FieldMaskSet<InstanceView> tracing_srcs, tracing_dsts;
            UniqueInst src_inst, dst_inst;
            src_inst.deserialize(derez);
            dst_inst.deserialize(derez);
            FieldMask src_mask, dst_mask;
            derez.deserialize(src_mask);
            derez.deserialize(dst_mask);
            ReductionOpID redop;
            derez.deserialize(redop);
            std::set<RtEvent> ready_events;
            tpl->record_copy_insts(lhs, tlid, src_idx, dst_idx, expr,
                                   src_inst, dst_inst, src_mask, dst_mask,
                                   src_mode, dst_mode, redop, ready_events);
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
            ApUserEvent *lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr = 
              IndexSpaceExpression::unpack_expression(derez, forest, source);
            size_t num_fields;
            derez.deserialize(num_fields);
            std::vector<CopySrcDstField> fields(num_fields);
            for (unsigned idx = 0; idx < num_fields; idx++)
              unpack_src_dst_field(derez, fields[idx]);
            size_t fill_size;
            derez.deserialize(fill_size);
            const void *fill_value = derez.get_current_pointer();
            derez.advance_pointer(fill_size);
#ifdef LEGION_SPY
            UniqueID fill_uid;
            derez.deserialize(fill_uid);
            FieldSpace handle;
            derez.deserialize(handle);
            RegionTreeID tree_id;
            derez.deserialize(tree_id);
#endif
            ApEvent precondition;
            derez.deserialize(precondition);
            PredEvent pred_guard;
            derez.deserialize(pred_guard);
            LgEvent unique_event;
            derez.deserialize(unique_event);
            int priority;
            derez.deserialize(priority);
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs; 
            // Do the base call
            tpl->record_issue_fill(tlid, lhs, expr, fields,
                                   fill_value, fill_size,
#ifdef LEGION_SPY
                                   fill_uid, handle, tree_id,
#endif
                                   precondition, pred_guard,
                                   unique_event, priority);
            if (lhs != lhs_copy)
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_ISSUE_FILL);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              runtime->send_remote_trace_response(source, rez);
            }
            else // lhs was unchanged
              Runtime::trigger_event(done);
            break;
          }
        case REMOTE_TRACE_FILL_INST:
          {
            RtUserEvent done;
            derez.deserialize(done);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr = 
              IndexSpaceExpression::unpack_expression(derez, forest, source);
            UniqueInst inst;
            inst.deserialize(derez);
            FieldMask inst_mask;
            derez.deserialize(inst_mask);
            bool reduction_initialization;
            derez.deserialize<bool>(reduction_initialization);
            std::set<RtEvent> ready_events;
            tpl->record_fill_inst(lhs, expr, inst, inst_mask,
                                  ready_events, reduction_initialization);
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
            RegionNode *node = runtime->forest->get_node(handle);
            std::set<RtEvent> effects;
            tpl->record_op_inst(tlid, index, inst, node, usage,
                                user_mask, update_validity, effects);
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
            ApUserEvent *lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            const ApUserEvent lhs_copy = lhs;
            tpl->record_set_op_sync_event(lhs, tlid);
            if (lhs != lhs_copy)
            {
              Serializer rez;
              {
                RezCheck z2(rez);
                rez.serialize(REMOTE_TRACE_SET_OP_SYNC);
                rez.serialize(lhs_ptr);
                rez.serialize(lhs);
                rez.serialize(done);
              }
              runtime->send_remote_trace_response(source, rez);
            }
            else // lhs didn't change
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
            std::vector<size_t> future_size_bounds(num_future_locations);
            std::vector<TaskTreeCoordinates> coordinates(num_future_locations);
            if (num_future_locations > 0)
            {
              output.future_locations.resize(num_future_locations);
              for (unsigned idx = 0; idx < num_future_locations; idx++)
                derez.deserialize(output.future_locations[idx]);
              for (unsigned idx = 0; idx < num_future_locations; idx++)
                derez.deserialize(future_size_bounds[idx]);
              for (unsigned idx = 0; idx < num_future_locations; idx++)
              {
                TaskTreeCoordinates &coords = coordinates[idx];
                size_t num_coords;
                derez.deserialize(num_coords);
                coords.resize(num_coords);
                for (unsigned idx2 = 0; idx2 < num_coords; idx2++)
                  coords[idx2].deserialize(derez);
              }
            }
            derez.deserialize(output.chosen_variant);
            derez.deserialize(output.task_priority);
            derez.deserialize<bool>(output.postmap_task);
            size_t num_phy_instances;
            derez.deserialize(num_phy_instances);
            std::deque<InstanceSet> physical_instances(num_phy_instances);
            std::set<RtEvent> ready_events;
            for (unsigned idx = 0; idx < num_phy_instances; idx++)
              physical_instances[idx].unpack_references(runtime, derez,
                                                        ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            std::set<RtEvent> applied_events;
            tpl->record_mapper_output(tlid, output, physical_instances,
                      future_size_bounds, coordinates, applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(applied, 
                  Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_SET_EFFECTS:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApEvent postcondition;
            derez.deserialize(postcondition);
            tpl->record_set_effects(tlid, postcondition);
            Runtime::trigger_event(applied);
            break;
          }
        case REMOTE_TRACE_COMPLETE_REPLAY:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            tlid.deserialize(derez);
            ApEvent ready_event;
            derez.deserialize(ready_event);
            tpl->record_complete_replay(tlid, ready_event);
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
            std::map<Reservation,bool> reservations;
            for (unsigned idx = 0; idx < num_reservations; idx++)
            {
              Reservation reservation;
              derez.deserialize(reservation);
              derez.deserialize<bool>(reservations[reservation]);
            }
            std::set<RtEvent> applied_events;
            tpl->record_reservations(tlid, reservations, applied_events);
            if (!applied_events.empty())
              Runtime::trigger_event(applied, 
                  Runtime::merge_events(applied_events));
            else
              Runtime::trigger_event(applied);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::handle_remote_response(
                                                            Deserializer &derez)
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
            ApUserEvent *event_ptr;
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
            ApEvent *event_ptr;
            derez.deserialize(event_ptr);
            derez.deserialize(*event_ptr);
            RtUserEvent done;
            derez.deserialize(done);
            Runtime::trigger_event(done);
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTraceRecorder::pack_src_dst_field(
                                  Serializer &rez, const CopySrcDstField &field)
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
                                    Deserializer &derez, CopySrcDstField &field)
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
    TraceInfo::TraceInfo(Operation *op, bool init)
      : rec(init_recorder(op)), tlid(init_tlid(op)),
        recording((rec == NULL) ? false : rec->is_recording())
    //--------------------------------------------------------------------------
    {
      if (recording && init)
        record_get_term_event(op->get_memoizable());
      if (rec != NULL)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceRecorder* TraceInfo::init_recorder(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (op == NULL)
        return NULL;
      Memoizable *memo = op->get_memoizable();
      if (memo == NULL)
        return NULL;
      return memo->get_template();
    }

    //--------------------------------------------------------------------------
    /*static*/ TraceLocalID TraceInfo::init_tlid(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (op == NULL)
        return TraceLocalID();
      Memoizable *memo = op->get_memoizable();
      if (memo == NULL)
        return TraceLocalID();
      return memo->get_trace_local_id();
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(SingleTask *task, RemoteTraceRecorder *r, bool init)
      : rec(r), tlid(task->get_trace_local_id()), recording(rec != NULL)
    //--------------------------------------------------------------------------
    {
      if (recording)
      {
        rec->add_recorder_reference();
        if (init)
          record_get_term_event(task);
      }
    }

    //--------------------------------------------------------------------------
    void TraceInfo::record_get_term_event(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(recording);
      assert(memo != NULL);
#endif
      ApEvent completion = memo->get_memo_completion();
      rec->record_get_term_event(completion, memo->get_memoizable_kind(), tlid);
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(const TraceInfo &rhs)
      : rec(rhs.rec), tlid(rhs.tlid), recording(rhs.recording)
    //--------------------------------------------------------------------------
    {
      if (rec != NULL)
        rec->add_recorder_reference();
    }

   //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(PhysicalTraceRecorder *r, const TraceLocalID &tld)
      : rec(r), tlid(tld), recording((r != NULL) && r->is_recording())
    //--------------------------------------------------------------------------
    {
      if (rec != NULL)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    TraceInfo::~TraceInfo(void)
    //--------------------------------------------------------------------------
    {
      if ((rec != NULL) && rec->remove_recorder_reference())
        delete rec;
    }

    /////////////////////////////////////////////////////////////
    // PhysicalTraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(Operation *o, unsigned idx, bool init)
      : TraceInfo(o, init), index(idx), dst_index(idx), update_validity(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(const TraceInfo &info, 
                                         unsigned idx, bool update/*=true*/)
      : TraceInfo(info), index(idx), dst_index(idx), update_validity(update)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(unsigned src_idx, 
                                         const TraceInfo &info,unsigned dst_idx)
      : TraceInfo(info), index(src_idx), dst_index(dst_idx), 
        update_validity(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(const PhysicalTraceInfo &rhs)
      : TraceInfo(rhs), index(rhs.index), dst_index(rhs.dst_index),
        update_validity(rhs.update_validity)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalTraceInfo::PhysicalTraceInfo(const TraceLocalID &tlid,
                                         unsigned src_idx, unsigned dst_idx,
                                         bool update, PhysicalTraceRecorder *r)
      : TraceInfo(r, tlid), index(src_idx), dst_index(dst_idx),
        update_validity(update)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void PhysicalTraceInfo::pack_trace_info(Serializer &rez,
                                            std::set<RtEvent> &applied) const 
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(recording);
      if (recording)
      {
#ifdef DEBUG_LEGION
        assert(rec != NULL);
#endif
        tlid.serialize(rez);
        rez.serialize(index);
        rez.serialize(dst_index);
        rez.serialize<bool>(update_validity);
        rec->pack_recorder(rez, applied); 
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceInfo PhysicalTraceInfo::unpack_trace_info(
                                          Deserializer &derez, Runtime *runtime)
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
        RemoteTraceRecorder *recorder = 
          RemoteTraceRecorder::unpack_remote_recorder(derez, runtime, tlid);
        return PhysicalTraceInfo(tlid, index, dst_index,
                                 update_validity, recorder);
      }
      else
        return PhysicalTraceInfo(NULL, -1U, false);
    }

    /////////////////////////////////////////////////////////////
    // ProjectionInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionInfo::ProjectionInfo(Runtime *runtime, 
                                   const RegionRequirement &req, 
                                   IndexSpaceNode *launch_space, 
                                   ShardingFunction *f/*=NULL*/,
                                   IndexSpace shard_space/*=NO_SPACE*/)
      : projection((req.handle_type != LEGION_SINGULAR_PROJECTION) ? 
          runtime->find_projection_function(req.projection) : NULL),
        projection_type(req.handle_type), projection_space(
         (req.handle_type != LEGION_SINGULAR_PROJECTION) ? launch_space : NULL),
        sharding_function(f), sharding_space(shard_space.exists() ? 
            runtime->forest->get_node(shard_space) : 
              (f == NULL) ? NULL : projection_space)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool ProjectionInfo::is_complete_projection(RegionTreeNode *node,
                                                const LogicalUser &user) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_projecting());
#endif
      return projection->is_complete(node, user.op, user.idx, projection_space);
    }

    /////////////////////////////////////////////////////////////
    // PathTraverser 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(RegionTreePath &p)
      : path(p)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser::PathTraverser(const PathTraverser &rhs)
      : path(rhs.path)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PathTraverser::~PathTraverser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PathTraverser& PathTraverser::operator=(const PathTraverser &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool PathTraverser::traverse(RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      // Continue visiting nodes and then finding their children
      // until we have traversed the entire path.
      while (true)
      {
#ifdef DEBUG_LEGION
        assert(node != NULL);
#endif
        depth = node->get_depth();
        has_child = path.has_child(depth);
        if (has_child)
          next_child = path.get_child(depth);
        bool continue_traversal = node->visit_node(this);
        if (!continue_traversal)
          return false;
        if (!has_child)
          break;
        node = node->get_tree_child(next_child);
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInitializer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInitializer::CurrentInitializer(ContextID c)
      : ctx(c)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInitializer::CurrentInitializer(const CurrentInitializer &rhs)
      : ctx(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CurrentInitializer::~CurrentInitializer(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInitializer& CurrentInitializer::operator=(
                                                  const CurrentInitializer &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInitializer::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->initialize_current_state(ctx);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // CurrentInvalidator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(ContextID c, bool only)
      : ctx(c), users_only(only)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::CurrentInvalidator(const CurrentInvalidator &rhs)
      : ctx(0), users_only(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator::~CurrentInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CurrentInvalidator& CurrentInvalidator::operator=(
                                                  const CurrentInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx, users_only); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool CurrentInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_current_state(ctx, users_only);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // DeletionInvalidator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeletionInvalidator::DeletionInvalidator(ContextID c, const FieldMask &dm)
      : ctx(c), deletion_mask(dm)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator::DeletionInvalidator(const DeletionInvalidator &rhs)
      : ctx(0), deletion_mask(rhs.deletion_mask)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator::~DeletionInvalidator(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    DeletionInvalidator& DeletionInvalidator::operator=(
                                                 const DeletionInvalidator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask); 
      return true;
    }

    //--------------------------------------------------------------------------
    bool DeletionInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->invalidate_deleted_state(ctx, deletion_mask);
      return true;
    }

    /////////////////////////////////////////////////////////////
    // ProjectionTree
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionTree::ProjectionTree(IndexTreeNode *n, ShardID owner)
      : node(n), owner_shard(owner)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionTree::ProjectionTree(const ProjectionTree &rhs)
      : node(rhs.node), owner_shard(rhs.owner_shard)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ProjectionTree::~ProjectionTree(void)
    //--------------------------------------------------------------------------
    {
      for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it =
            children.begin(); it != children.end(); it++)
        delete it->second;
    }

    //--------------------------------------------------------------------------
    ProjectionTree& ProjectionTree::operator=(const ProjectionTree &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ProjectionTree::add_child(ProjectionTree *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(child != this);
#endif
      children[child->node] = child;
    }

    //--------------------------------------------------------------------------
    bool ProjectionTree::dominates(const ProjectionTree *other) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == other->node);
#endif
      // If we have no children we definitely dominate
      // Assuming of course that we are on the same shard
      if (children.empty())
      {
        if (other->children.empty())
          return (owner_shard == other->owner_shard);
        return other->all_same_shard(owner_shard);
      }
      // If we have children and the other one doesn't then we don't
      if (other->children.empty())
        return false;
      // Check to see if we have a child that dominates each of the
      // other trees children
      for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it =
            other->children.begin(); it != other->children.end(); it++)
      {
        std::map<IndexTreeNode*,ProjectionTree*>::const_iterator finder =
          children.find(it->first);
        if (finder == children.end())
          return false;
        if (!finder->second->dominates(it->second))
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionTree::disjoint(const ProjectionTree *other) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == other->node);
#endif
      if (children.empty() || other->children.empty())
        return false;
      if (!node->is_index_space_node() &&
          node->as_index_part_node()->is_disjoint())
      {
        // All children are disjoint, if there are any common ones
        // see if they are disjoint with respect to each other
        for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          std::map<IndexTreeNode*,ProjectionTree*>::const_iterator finder =
            other->children.find(it->first);
          if ((finder != other->children.end()) &&
              !it->second->disjoint(finder->second))
            return false;
        }
      }
      else
      {
        // Children are not disjoint, so any that don't match
        // cause the test to fail, otherwise we can still recurse
        if (node->is_index_space_node())
        {
          IndexSpaceNode *space = node->as_index_space_node();
          for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it1 =
                children.begin(); it1 != children.end(); it1++)
          {
            for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it2 =
                  other->children.begin(); it2 != other->children.end(); it2++)
            {
              if (it1->first == it2->first)
              {
                if (!it1->second->disjoint(it2->second))
                  return false;
              }
              else if (!space->are_disjoint(it1->first->color,
                                            it2->first->color))
                return false;
            }
          }

        }
        else
        {
          IndexPartNode *part = node->as_index_part_node();
          for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it1 =
                children.begin(); it1 != children.end(); it1++)
          {
            for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it2 =
                  other->children.begin(); it2 != other->children.end(); it2++)
            {
              if (it1->first == it2->first)
              {
                if (!it1->second->disjoint(it2->second))
                  return false;
              }
              else if (!part->are_disjoint(it1->first->color,it2->first->color))
                return false;
            }
          }
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionTree::all_same_shard(ShardID other_shard) const
    //--------------------------------------------------------------------------
    {
      if (children.empty())
        return (owner_shard == other_shard);
      for (std::map<IndexTreeNode*,ProjectionTree*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        if (!it->second->all_same_shard(other_shard))
          return false;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // LogicalState 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(RegionTreeNode *node, ContextID ctx)
      : owner(node)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalState::LogicalState(const LogicalState &rhs)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalState::~LogicalState(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalState& LogicalState::operator=(const LogicalState&rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LogicalState::check_init(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(field_states.empty());
      assert(curr_epoch_users.empty());
      assert(prev_epoch_users.empty());
      assert(!reduction_fields);
      assert(!disjoint_complete_tree);
      assert(disjoint_complete_accesses.empty());
      assert(disjoint_complete_child_counts.empty());
      assert(disjoint_complete_projections.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void LogicalState::clear_logical_users(void)
    //--------------------------------------------------------------------------
    {
      if (!curr_epoch_users.empty())
      {
        for (LegionList<LogicalUser,CURR_LOGICAL_ALLOC>::const_iterator it =
              curr_epoch_users.begin(); it != curr_epoch_users.end(); it++)
        {
          it->op->remove_mapping_reference(it->gen); 
        }
        curr_epoch_users.clear();
      }
      if (!prev_epoch_users.empty())
      {
        for (LegionList<LogicalUser,PREV_LOGICAL_ALLOC>::const_iterator it =
              prev_epoch_users.begin(); it != prev_epoch_users.end(); it++)
        {
          it->op->remove_mapping_reference(it->gen); 
        }
        prev_epoch_users.clear();
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::reset(void)
    //--------------------------------------------------------------------------
    {
      field_states.clear();
      clear_logical_users(); 
      reduction_fields.clear();
      outstanding_reductions.clear();
      disjoint_complete_tree.clear();
      if (!disjoint_complete_children.empty())
      {
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              disjoint_complete_children.begin(); it !=
              disjoint_complete_children.end(); it++)
          if (it->first->remove_base_valid_ref(DISJOINT_COMPLETE_REF))
            delete it->first;
        disjoint_complete_children.clear();
      }
      disjoint_complete_accesses.clear(); 
      disjoint_complete_child_counts.clear();
      if (!disjoint_complete_projections.empty())
      {
        for (FieldMaskSet<RefProjectionSummary>::const_iterator it =
              disjoint_complete_projections.begin(); it !=
              disjoint_complete_projections.end(); it++)
          if (it->first->remove_reference())
            delete it->first;
        disjoint_complete_projections.clear();
      }
    } 

    //--------------------------------------------------------------------------
    void LogicalState::clear_deleted_state(const FieldMask &deleted_mask)
    //--------------------------------------------------------------------------
    {
      for (LegionList<FieldState>::iterator it = field_states.begin();
            it != field_states.end(); /*nothing*/)
      {
        if (it->filter(deleted_mask))
          it = field_states.erase(it);
        else
          it++;
      }
      reduction_fields -= deleted_mask;
      if (!outstanding_reductions.empty())
      {
        std::vector<ReductionOpID> to_delete;
        for (LegionMap<ReductionOpID,FieldMask>::iterator it = 
              outstanding_reductions.begin(); it != 
              outstanding_reductions.end(); it++)
        {
          it->second -= deleted_mask;
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<ReductionOpID>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          outstanding_reductions.erase(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalState::merge(LogicalState &src, 
                             std::set<RegionTreeNode*> &to_traverse)
    //--------------------------------------------------------------------------
    {
      for (LegionList<FieldState>::iterator fit = 
            src.field_states.begin(); fit != src.field_states.end(); fit++)
      {
        for (FieldMaskSet<RegionTreeNode>::iterator it = 
              fit->open_children.begin(); it != fit->open_children.end(); it++)
          to_traverse.insert(it->first);
        // See if we can add it to any of the existing field states
        bool merged = false;
        for (LegionList<FieldState>::iterator it = 
              field_states.begin(); it != field_states.end(); it++)
        {
          if (!it->overlaps(*fit))
            continue;
          it->merge(*fit, owner);
          merged = true;
          break;
        }
        if (!merged)
          field_states.push_back(*fit);
      }
      src.field_states.clear();
      curr_epoch_users.splice(curr_epoch_users.end(), src.curr_epoch_users);
      prev_epoch_users.splice(prev_epoch_users.end(), src.prev_epoch_users);
      if (!!src.reduction_fields)
      {
        reduction_fields |= src.reduction_fields;
        src.reduction_fields.clear();
      }
      if (!src.outstanding_reductions.empty())
      {
        for (LegionMap<ReductionOpID,FieldMask>::const_iterator it =
              src.outstanding_reductions.begin(); it != 
              src.outstanding_reductions.end(); it++)
        {
          LegionMap<ReductionOpID,FieldMask>::iterator finder =
            outstanding_reductions.find(it->first);
          if (finder == outstanding_reductions.end())
            outstanding_reductions.insert(*it);
          else
            finder->second |= it->second;
        }
        src.outstanding_reductions.clear();
      }
      if (!!src.disjoint_complete_tree)
      {
        disjoint_complete_tree |= src.disjoint_complete_tree;
        src.disjoint_complete_tree.clear();
      }
      if (!src.disjoint_complete_children.empty())
      {
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              src.disjoint_complete_children.begin(); it !=
              src.disjoint_complete_children.end(); it++)
        {
          to_traverse.insert(it->first);
          // Remove duplicate references if they are already there
          // Otherwise the reference flows back with the node
          if (disjoint_complete_children.insert(it->first, it->second))
            it->first->remove_base_valid_ref(DISJOINT_COMPLETE_REF);
        }
        src.disjoint_complete_children.clear();
      }
      if (!src.disjoint_complete_accesses.empty())
      {
        for (FieldMaskSet<RegionTreeNode>::const_iterator it =
              src.disjoint_complete_accesses.begin(); it !=
              src.disjoint_complete_accesses.end(); it++)
          disjoint_complete_accesses.insert(it->first, it->second);
      }
      if (!src.disjoint_complete_child_counts.empty())
      {
        for (LegionMap<size_t,FieldMask>::const_iterator it =
              src.disjoint_complete_child_counts.begin(); it !=
              src.disjoint_complete_child_counts.end(); it++)
        {
          LegionMap<size_t,FieldMask>::iterator finder =
            disjoint_complete_child_counts.find(it->first);
          if (finder == disjoint_complete_child_counts.end())
            disjoint_complete_child_counts.insert(*it);
          else
            finder->second |= it->second;
        }
      }
      if (!src.disjoint_complete_projections.empty())
      {
        for (FieldMaskSet<RefProjectionSummary>::const_iterator it =
              src.disjoint_complete_projections.begin(); it !=
              src.disjoint_complete_projections.end(); it++)
          disjoint_complete_projections.insert(it->first, it->second);
        src.disjoint_complete_projections.clear();
      }
#ifdef DEBUG_LEGION
      src.check_init();
#endif
    }

    //--------------------------------------------------------------------------
    void LogicalState::swap(LogicalState &src,
                            std::set<RegionTreeNode*> &to_traverse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      check_init();
#endif
      field_states.swap(src.field_states);
      curr_epoch_users.swap(src.curr_epoch_users);
      prev_epoch_users.swap(src.prev_epoch_users);
      if (!!src.reduction_fields)
      {
        reduction_fields = src.reduction_fields;
        src.reduction_fields.clear();
      }
      outstanding_reductions.swap(src.outstanding_reductions);
      if (!!src.disjoint_complete_tree)
      {
        disjoint_complete_tree = src.disjoint_complete_tree;
        src.disjoint_complete_tree.clear();
      }
      disjoint_complete_accesses.swap(src.disjoint_complete_accesses);
      disjoint_complete_child_counts.swap(src.disjoint_complete_child_counts);
      disjoint_complete_children.swap(src.disjoint_complete_children);
      disjoint_complete_projections.swap(src.disjoint_complete_projections);
      for (LegionList<FieldState>::const_iterator fit = 
            field_states.begin(); fit != field_states.end(); fit++)
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              fit->open_children.begin(); it != fit->open_children.end(); it++)
          to_traverse.insert(it->first);
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            disjoint_complete_children.begin(); it != 
            disjoint_complete_children.end(); it++)
        to_traverse.insert(it->first);
#ifdef DEBUG_LEGION
      src.check_init();
#endif
    }

    /////////////////////////////////////////////////////////////
    // Projection Summary 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(void)
      : domain(NULL), projection(NULL), sharding(NULL), sharding_domain(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(IndexSpaceNode *is, 
                                         ProjectionFunction *p, 
                                         ShardingFunction *s,
                                         IndexSpaceNode *sd)
      : domain(is), projection(p), sharding(s), sharding_domain(sd)
    //--------------------------------------------------------------------------
    {
      if (domain != NULL)
        domain->add_base_valid_ref(FIELD_STATE_REF);
      if (sharding_domain != NULL)
        sharding_domain->add_base_valid_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(const ProjectionInfo &info)
      : domain(info.projection_space), projection(info.projection), 
        sharding(info.sharding_function), sharding_domain(info.sharding_space)
    //--------------------------------------------------------------------------
    {
      if (domain != NULL)
        domain->add_base_valid_ref(FIELD_STATE_REF);
      if (sharding_domain != NULL)
        sharding_domain->add_base_valid_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(ProjectionSummary &&rhs)
      : domain(rhs.domain), projection(rhs.projection),
        sharding(rhs.sharding), sharding_domain(rhs.sharding_domain)
    //--------------------------------------------------------------------------
    {
      rhs.domain = NULL;
      rhs.projection = NULL;
      rhs.sharding = NULL;
      rhs.sharding_domain = NULL;
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::ProjectionSummary(const ProjectionSummary &rhs)
      : domain(rhs.domain), projection(rhs.projection),
        sharding(rhs.sharding), sharding_domain(rhs.sharding_domain)
    //--------------------------------------------------------------------------
    {
      if (domain != NULL)
        domain->add_base_valid_ref(FIELD_STATE_REF);
      if (sharding_domain != NULL)
        sharding_domain->add_base_valid_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    ProjectionSummary::~ProjectionSummary(void)
    //--------------------------------------------------------------------------
    {
      if ((domain != NULL) && domain->remove_base_valid_ref(FIELD_STATE_REF))
        delete domain;
      if ((sharding_domain != NULL) && 
            sharding_domain->remove_base_valid_ref(FIELD_STATE_REF))
        delete sharding_domain;
    }

    //--------------------------------------------------------------------------
    ProjectionSummary& ProjectionSummary::operator=(const ProjectionSummary &ps)
    //--------------------------------------------------------------------------
    {
      if ((domain != NULL) && domain->remove_base_valid_ref(FIELD_STATE_REF))
        delete domain;
      if ((sharding_domain != NULL) && 
            sharding_domain->remove_base_valid_ref(FIELD_STATE_REF))
        delete sharding_domain;
      domain = ps.domain;
      projection = ps.projection;
      sharding = ps.sharding;
      sharding_domain = ps.sharding_domain;
      if (domain != NULL)
        domain->add_base_valid_ref(FIELD_STATE_REF);
      if (sharding_domain != NULL)
        sharding_domain->add_base_valid_ref(FIELD_STATE_REF);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::operator<(const ProjectionSummary &rhs) const
    //--------------------------------------------------------------------------
    {
      if (domain->handle < rhs.domain->handle)
        return true;
      else if (domain->handle > rhs.domain->handle)
        return false;
      else if (projection->projection_id < rhs.projection->projection_id)
        return true;
      else if (projection->projection_id > rhs.projection->projection_id)
        return false;
      else if ((sharding == NULL) && (rhs.sharding == NULL))
        return false;
      else if ((sharding == NULL) && (rhs.sharding != NULL))
        return true;
      else if (rhs.sharding == NULL)
        return false;
      else if (sharding->sharding_id < rhs.sharding->sharding_id)
        return true;
      else if (sharding->sharding_id > rhs.sharding->sharding_id)
        return false;
      else
        return sharding_domain->handle < rhs.sharding_domain->handle;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::operator==(const ProjectionSummary &rhs) const
    //--------------------------------------------------------------------------
    {
      if (domain->handle != rhs.domain->handle)
        return false;
      else if (projection->projection_id != rhs.projection->projection_id)
        return false;
      else if ((sharding == NULL) && (rhs.sharding == NULL))
        return true;
      else if ((sharding == NULL) && (rhs.sharding != NULL))
        return false;
      else if (rhs.sharding == NULL)
        return false;
      if (sharding->sharding_id != rhs.sharding->sharding_id)
        return false;
      if (sharding_domain->handle != rhs.sharding_domain->handle)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool ProjectionSummary::operator!=(const ProjectionSummary &rhs) const
    //--------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //--------------------------------------------------------------------------
    void ProjectionSummary::pack_summary(Serializer &rez) const 
    //--------------------------------------------------------------------------
    {
      rez.serialize(domain->handle);
      rez.serialize(projection->projection_id);
      domain->pack_valid_ref();
    }

    //--------------------------------------------------------------------------
    /*static*/ ProjectionSummary ProjectionSummary::unpack_summary(
                                 Deserializer &derez, RegionTreeForest *context)
    //--------------------------------------------------------------------------
    {
      ProjectionSummary result;
      IndexSpace handle;
      derez.deserialize(handle);
      result.domain = context->get_node(handle);
      result.domain->add_base_valid_ref(FIELD_STATE_REF);
      result.domain->unpack_valid_ref();
      ProjectionID pid;
      derez.deserialize(pid);
      result.projection = context->runtime->find_projection_function(pid);
      return result;
    }

    /////////////////////////////////////////////////////////////
    // RefProjectionSummary
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RefProjectionSummary::RefProjectionSummary(const ProjectionInfo &rhs)
      : ProjectionSummary(rhs), Collectable()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RefProjectionSummary::RefProjectionSummary(ProjectionSummary &&rhs)
      : ProjectionSummary(rhs), Collectable()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RefProjectionSummary::RefProjectionSummary(const RefProjectionSummary &rhs)
      : ProjectionSummary(rhs), Collectable()
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RefProjectionSummary::~RefProjectionSummary(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RefProjectionSummary& RefProjectionSummary::operator=(
                                                const RefProjectionSummary &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void RefProjectionSummary::project_refinement(RegionTreeNode *node,
                                        std::vector<RegionNode*> &regions) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding == NULL);
#endif
      projection->project_refinement(domain, node, regions);
    }

    //--------------------------------------------------------------------------
    void RefProjectionSummary::project_refinement(RegionTreeNode *node,
                      ShardID shard_id, std::vector<RegionNode*> &regions,
                      Provenance *provenance) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding != NULL);
#endif
      // Find the domain of points for this shard
      IndexSpace shard_handle = sharding->find_shard_space(shard_id, domain,
          (sharding_domain != NULL) ? sharding_domain->handle : domain->handle,
          provenance);
      if (shard_handle.exists())
      {
        IndexSpaceNode *shard_domain = node->context->get_node(shard_handle);
        projection->project_refinement(shard_domain, node, regions);
      }
    }

    /////////////////////////////////////////////////////////////
    // FieldState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, 
                           RegionTreeNode *child)
      : redop(0)
    //--------------------------------------------------------------------------
    {
      if (IS_READ_ONLY(user.usage))
        open_state = OPEN_READ_ONLY;
      else if (IS_WRITE(user.usage))
        open_state = OPEN_READ_WRITE;
      else if (IS_REDUCE(user.usage))
      {
        open_state = OPEN_SINGLE_REDUCE;
        redop = user.usage.redop;
      }
      if (open_children.insert(child, m))
        child->add_base_valid_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const RegionUsage &usage, const FieldMask &m,
                           ProjectionFunction *proj, IndexSpaceNode *proj_space,
                           ShardingFunction *fn, IndexSpaceNode *shard_space,
                           RegionTreeNode *node, bool dirty_reduction)
      : redop(0)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(proj != NULL);
#endif
      open_children.relax_valid_mask(m);
      if (IS_READ_ONLY(usage))
        open_state = OPEN_READ_ONLY_PROJ;
      else if (IS_REDUCE(usage))
      {
        if (dirty_reduction)
          open_state = OPEN_REDUCE_PROJ_DIRTY;
        else
          open_state = OPEN_REDUCE_PROJ;
        redop = usage.redop;
      }
      else
      {
        open_state = OPEN_READ_WRITE_PROJ;
        projections.insert(ProjectionSummary(proj_space, proj, fn,
                                             shard_space));
      }
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const FieldState &rhs)
      : open_state(rhs.open_state), redop(rhs.redop)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.open_children.empty());
      assert(rhs.projections.empty());
#endif
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(FieldState &&rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
      projections.swap(rhs.projections);
    }

    //--------------------------------------------------------------------------
    FieldState::~FieldState(void)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
        if (it->first->remove_base_valid_ref(FIELD_STATE_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    FieldState& FieldState::operator=(const FieldState &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(open_children.empty());
      assert(rhs.open_children.empty());
      assert(projections.empty());
      assert(rhs.projections.empty());
#endif
      open_state = rhs.open_state;
      redop = rhs.redop;
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldState& FieldState::operator=(FieldState &&rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
      projections.swap(rhs.projections);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (is_projection_state())
      {
        if (!rhs.is_projection_state())
          return false;
        // Both projection spaces, check to see if they have the same
        // set of projections
        if (!projections_match(rhs))
          return false;
        // if we make it past here they are all the same
      }
      else if (rhs.is_projection_state())
        return false;
      // Now check the privilege states
      if (redop == 0)
        return (open_state == rhs.open_state);
      else
      {
#ifdef DEBUG_LEGION
        assert((open_state == OPEN_SINGLE_REDUCE) ||
               (open_state == OPEN_MULTI_REDUCE) ||
               (open_state == OPEN_REDUCE_PROJ) ||
               (open_state == OPEN_REDUCE_PROJ_DIRTY));
        assert((rhs.open_state == OPEN_SINGLE_REDUCE) ||
               (rhs.open_state == OPEN_MULTI_REDUCE) ||
               (rhs.open_state == OPEN_REDUCE_PROJ) ||
               (rhs.open_state == OPEN_REDUCE_PROJ_DIRTY));
#endif
        // Only support merging reduction fields with exactly the
        // same mask which should be single fields for reductions
        return (valid_fields() == rhs.valid_fields());
      }
    }

    //--------------------------------------------------------------------------
    bool FieldState::projections_match(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (projections.size() != rhs.projections.size())
        return false;
      std::set<ProjectionSummary>::const_iterator it1 = projections.begin();
      std::set<ProjectionSummary>::const_iterator it2 = rhs.projections.begin();
      // zip the projections so we can compare them
      while (it1 != projections.end())
      {
        if ((*it1) != (*it2))
          return false;
        it1++; it2++;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void FieldState::merge(FieldState &rhs, RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      if (!rhs.open_children.empty())
      {
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              rhs.open_children.begin(); it != rhs.open_children.end(); it++)
          // Remove duplicate references if we already had it
          if (!open_children.insert(it->first, it->second))
            it->first->remove_base_valid_ref(FIELD_STATE_REF);
        rhs.open_children.clear();
      }
      else
        open_children.relax_valid_mask(rhs.open_children.get_valid_mask());
#ifdef DEBUG_LEGION
      assert(redop == rhs.redop);
      assert(projections_match(rhs));
#endif
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(!open_children.empty());
#endif
        // For the reductions, handle the case where we need to merge
        // reduction modes, if they are all disjoint, we don't need
        // to distinguish between single and multi reduce
        if (node->are_all_children_disjoint())
        {
          open_state = OPEN_READ_WRITE;
          redop = 0;
        }
        else
        {
          if (open_children.size() == 1)
            open_state = OPEN_SINGLE_REDUCE;
          else
            open_state = OPEN_MULTI_REDUCE;
        }
      }
      // no need to merge projections, we know they are the same
    }

    //--------------------------------------------------------------------------
    bool FieldState::filter(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (is_projection_state())
      {
#ifdef DEBUG_LEGION
        assert(open_children.empty());
#endif
        open_children.filter_valid_mask(mask);
        return !open_children.get_valid_mask();
      }
      else
      {
        std::vector<RegionTreeNode*> to_delete;
        for (FieldMaskSet<RegionTreeNode>::iterator it = 
              open_children.begin(); it != open_children.end(); it++)
        {
          it.filter(mask);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (to_delete.size() < open_children.size())
        {
          for (std::vector<RegionTreeNode*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            open_children.erase(*it);
            if ((*it)->remove_base_valid_ref(FIELD_STATE_REF))
              delete (*it);
          }
        }
        else
        {
          open_children.clear();
          for (std::vector<RegionTreeNode*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
            if ((*it)->remove_base_valid_ref(FIELD_STATE_REF))
              delete (*it);
        }
        open_children.tighten_valid_mask();
        return open_children.empty();
      }
    }

    //--------------------------------------------------------------------------
    void FieldState::add_child(RegionTreeNode *child, const FieldMask &mask) 
    //--------------------------------------------------------------------------
    {
      if (open_children.insert(child, mask))
        child->add_base_valid_ref(FIELD_STATE_REF);
    }

    //--------------------------------------------------------------------------
    void FieldState::remove_child(RegionTreeNode *child)
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<RegionTreeNode>::iterator finder = 
        open_children.find(child);
#ifdef DEBUG_LEGION
      assert(finder != open_children.end());
      assert(!finder->second);
#endif
      open_children.erase(finder);
      if (child->remove_base_valid_ref(FIELD_STATE_REF))
        delete child;
    }

    //--------------------------------------------------------------------------
    bool FieldState::can_elide_close_operation(Operation *op, unsigned index,
         const ProjectionInfo &info, RegionTreeNode *node, bool reduction) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // should be in a projection mode
      assert(is_projection_state());
      assert(!projections.empty());
#endif
      // This function is super important! It decides whether this new
      // projection info can be added to the current projection epoch, if
      // it can't then we will need a close operation to be inserted
      bool elide = true;
#ifndef LEGION_SPY
      // Technically we only need to do this analysis to insert close operations
      // if we're control replicated because close operations are the only way
      // to enforce dependences between points in different shards. Without
      // control replication, each index space operations guarantees that all
      // its points are mapped before it is mapped so we get an implicit fence
      // and there is no need for an explicit close operation. Legion Spy
      // though doesn't make this distinction though and will check for the
      // presence of a close operation no matter what.
      if (info.sharding_function != NULL)
#endif
      {
        // We have a sharding function so we are in a 
        // control replication context

        // See if all the index spaces dominate, and the projection
        // functions are all the same, and the sharding functions are
        // all the same, in which case we know this is totally data
        // parallel and each shard has the same subregion set
        bool check_expensive = true;
        for (std::set<ProjectionSummary>::const_iterator it = 
              projections.begin(); it != projections.end(); it++)
        {
          if (it->projection != info.projection)
          {
            elide = false;
            break;
          }
          if (it->sharding != info.sharding_function)
          {
            elide = false;
            // No need to check expensive here since we know
            // that we need the close operation no matter what
            check_expensive = false;
            break;
          }
          if ((it->domain != info.projection_space) && 
              !it->domain->dominates(info.projection_space))
          {
            elide = false;
            break;
          }
        }

        if (!elide && check_expensive)
        {
          // Next we're going to need to compute the actual interference
          // set so check to see if we've memoized the result or not
          if (!info.projection->find_elide_close_result(info, projections,
                                                        node, elide))
          {
            elide = expensive_elide_test(op, index, info, node, reduction);
            // Now memoize the results for later
            info.projection->record_elide_close_result(info, projections,
                                                       node, elide);
          }
        }
      }
      return elide;
    }

    //--------------------------------------------------------------------------
    void FieldState::record_projection_summary(const ProjectionInfo &info,
                                               RegionTreeNode *node)
    //--------------------------------------------------------------------------
    {
      projections.insert(ProjectionSummary(info));
    }

    //--------------------------------------------------------------------------
    bool FieldState::expensive_elide_test(Operation *op, unsigned index,
         const ProjectionInfo &info, RegionTreeNode *node, bool reduction) const
    //--------------------------------------------------------------------------
    {
      // We can't do this test if the projection function is not functional
      if (!info.projection->is_functional)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_SLOW_NON_FUNCTIONAL_PROJECTION,
            "We strongly encourage all projection functors to be functional, "
            "however, projection function %d is not and therefore an "
            "expensive analysis cannot be memoized. Please consider making "
            "it functional to avoid performance degredation.", 
            info.projection->projection_id)
        return false;
      }
      // Then check whether one of two conditions are true:
      // 1. We can build an injective mapping for each region in
      // the next projection where it is the same as or a subregion
      // of every region it interferes with
      // 2. If the new projection is not a reduction then if 
      // every access is independent of the prior writes then 
      // we know that we can safely add this to the epoch without
      // needing to do a close operation
      IndexTreeNode *root_source = node->get_row_source();
      ProjectionTree *prev = new ProjectionTree(root_source);
      // Construct the previous projection tree from all prior projections
      {
        std::map<IndexTreeNode*,ProjectionTree*> node_map;
        node_map[root_source] = prev;
        for (std::set<ProjectionSummary>::const_iterator it = 
              projections.begin(); it != projections.end(); it++)
        {
          if (!it->projection->is_functional)
          {
            REPORT_LEGION_WARNING(LEGION_WARNING_SLOW_NON_FUNCTIONAL_PROJECTION,
              "We strongly encourage all projection functors to be functional, "
              "however, projection function %d is not and therefore an "
              "expensive analysis cannot be memoized. Please consider making "
              "it functional to avoid performance degredation.",
              info.projection->projection_id)
            delete prev;
            return false;
          }
          it->projection->construct_projection_tree(op, index, node, it->domain,
                        it->sharding, it->sharding_domain, node_map);
        }
      }
      // Then construct the new projection tree
      ProjectionTree *next = 
        info.projection->construct_projection_tree(op, index, node, 
            info.projection_space, info.sharding_function, info.sharding_space);
      // First check to see if the previous dominates
      bool has_mapping = false;
      if (prev->dominates(next))
        has_mapping = true;
      bool all_disjoint = false;
      if (!has_mapping && !reduction) // no disjoint reductions
        all_disjoint = prev->disjoint(next);
      // Clean up our data structures
      delete prev;
      delete next;
#ifdef DEBUG_LEGION
      assert(!has_mapping || !all_disjoint); // can't both be true
#endif
      return (has_mapping || all_disjoint);
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask,
                                 RegionNode *node) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_WRITE:
          {
            logger->log("Field State: OPEN READ WRITE (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_SINGLE_REDUCE:
          {
            logger->log("Field State: OPEN SINGLE REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_MULTI_REDUCE:
          {
            logger->log("Field State: OPEN MULTI REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_READ_ONLY_PROJ:
          {
            logger->log("Field State: OPEN READ-ONLY PROJECTION %zd",
                        projections.size());
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %zd",
                        projections.size());
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %zd Mode %d",
                        projections.size(), redop);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        logger->log("Color %d   Mask %s", it->first->get_color(), mask_buffer);
        free(mask_buffer);
      }
      logger->up();
    }

    //--------------------------------------------------------------------------
    void FieldState::print_state(TreeStateLogger *logger,
                                 const FieldMask &capture_mask,
                                 PartitionNode *node) const
    //--------------------------------------------------------------------------
    {
      switch (open_state)
      {
        case NOT_OPEN:
          {
            logger->log("Field State: NOT OPEN (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_WRITE:
          {
            logger->log("Field State: OPEN READ WRITE (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_READ_ONLY:
          {
            logger->log("Field State: OPEN READ-ONLY (%ld)", 
                        open_children.size());
            break;
          }
        case OPEN_SINGLE_REDUCE:
          {
            logger->log("Field State: OPEN SINGLE REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_MULTI_REDUCE:
          {
            logger->log("Field State: OPEN MULTI REDUCE Mode %d (%ld)", 
                        redop, open_children.size());
            break;
          }
        case OPEN_READ_ONLY_PROJ:
          {
            logger->log("Field State: OPEN READ-ONLY PROJECTION %zd",
                        projections.size()); 
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %zd",
                        projections.size());
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %zd Mode %d",
                        projections.size());
            break;
          }
        case OPEN_REDUCE_PROJ_DIRTY:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION (Dirty) %zd "
                        "Mode %d", projections.size(), redop);
            break;
          }
        default:
          assert(false);
      }
      logger->down();
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            open_children.begin(); it != open_children.end(); it++)
      {
        IndexSpaceNode *color_space = node->row_source->color_space;
        DomainPoint color =
          color_space->delinearize_color_to_point(it->first->get_color());
        FieldMask overlap = it->second & capture_mask;
        if (!overlap)
          continue;
        char *mask_buffer = overlap.to_string();
        switch (color.get_dim())
        {
          case 1:
            {
              logger->log("Color %d   Mask %s", 
                          color[0], mask_buffer);
              break;
            }
#if LEGION_MAX_DIM >= 2
          case 2:
            {
              logger->log("Color (%d,%d)   Mask %s", 
                          color[0], color[1], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 3
          case 3:
            {
              logger->log("Color (%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 4
          case 4:
            {
              logger->log("Color (%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 5
          case 5:
            {
              logger->log("Color (%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2],
                          color[3], color[4], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 6
          case 6:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 7
          case 7:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d)   Mask %s", 
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 8
          case 8:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d,%d)   Mask %s",
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], color[7], mask_buffer);
              break;
            }
#endif
#if LEGION_MAX_DIM >= 9
          case 9:
            {
              logger->log("Color (%d,%d,%d,%d,%d,%d,%d,%d,%d)   Mask %s",
                          color[0], color[1], color[2], 
                          color[3], color[4], color[5], 
                          color[6], color[7], color[8], mask_buffer);
              break;
            }
#endif
          default:
            assert(false); // implemenent more dimensions
        }
        free(mask_buffer);
      }
      logger->up();
    }

    /////////////////////////////////////////////////////////////
    // Logical Closer 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalCloser::LogicalCloser(ContextID c, const LogicalUser &u, 
                                 RegionTreeNode *r, bool val)
      : ctx(c), user(u), root_node(r), validates(val),
        tracing(user.op->is_tracing()), close_op(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalCloser::~LogicalCloser(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_close_operation(const FieldMask &mask) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!mask);
#endif
      close_mask |= mask;
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::record_closed_user(const LogicalUser &user,
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      closed_users.push_back(user);
      LogicalUser &closed_user = closed_users.back();
      closed_user.field_mask = mask;
    }

#ifndef LEGION_SPY
    //--------------------------------------------------------------------------
    void LogicalCloser::pop_closed_user(void)
    //--------------------------------------------------------------------------
    {
      closed_users.pop_back();
    }
#endif

    //--------------------------------------------------------------------------
    void LogicalCloser::initialize_close_operations(LogicalState &state, 
                                             Operation *creator,
                                             const LogicalTraceInfo &trace_info,
                                             const bool check_for_refinements,
                                             const bool has_next_child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // These sets of fields better be disjoint
      assert(!!close_mask);
      assert(close_op == NULL);
#endif
      // Construct a reigon requirement for this operation
      // All privileges are based on the parent logical region
      RegionRequirement req;
      if (root_node->is_region())
        req = RegionRequirement(root_node->as_region_node()->handle,
            LEGION_READ_WRITE, LEGION_EXCLUSIVE, trace_info.req.parent);
      else
        req = RegionRequirement(root_node->as_partition_node()->handle, 0,
            LEGION_READ_WRITE, LEGION_EXCLUSIVE, trace_info.req.parent);
      InnerContext *context = creator->get_context();
#ifdef DEBUG_LEGION_COLLECTIVES
      close_op = context->get_merge_close_op(user, root_node);
#else
      close_op = context->get_merge_close_op();
#endif
      merge_close_gen = close_op->get_generation();
      req.privilege_fields.clear();
      root_node->column_source->get_field_set(close_mask,
                                             trace_info.req.privilege_fields,
                                             req.privilege_fields);
      close_op->initialize(context, req, trace_info, trace_info.req_idx, 
                           close_mask, creator);
      if (check_for_refinements && !!state.disjoint_complete_tree)
      {
        const FieldMask refinement_mask = 
          close_mask & state.disjoint_complete_tree; 
        if (!!refinement_mask)
        {
          // Record that this close op should make a new equivalence
          // set at this region and invalidate all the ones below
          const bool overwriting = HAS_WRITE_DISCARD(user.usage) &&
                    !has_next_child && !user.op->is_predicated_op() && 
                    !trace_info.recording_trace;
          close_op->record_refinements(refinement_mask, overwriting);
#ifdef DEBUG_LEGION
          assert(state.owner->is_region());
#endif
          // We're closing to a region, so invalidate all the children
          std::vector<RegionTreeNode*> to_delete;
          for (FieldMaskSet<RegionTreeNode>::iterator it = 
                state.disjoint_complete_children.begin(); it !=
                state.disjoint_complete_children.end(); it++)
          {
            const FieldMask overlap = refinement_mask & it->second;
            if (!overlap)
              continue;
            it->first->invalidate_disjoint_complete_tree(ctx, overlap, true);
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<RegionTreeNode*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              state.disjoint_complete_children.erase(*it);
              if ((*it)->remove_base_valid_ref(DISJOINT_COMPLETE_REF))
                delete (*it);
            }
            state.disjoint_complete_children.tighten_valid_mask();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::perform_dependence_analysis(const LogicalUser &current,
                                                    const FieldMask &open_below,
                             LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &cusers,
                             LegionList<LogicalUser,PREV_LOGICAL_ALLOC> &pusers)
    //--------------------------------------------------------------------------
    {
      // We also need to do dependence analysis against all the other operations
      // that this operation recorded dependences on above in the tree so we
      // don't run too early.
      LegionList<LogicalUser,LOGICAL_REC_ALLOC> &above_users = 
                                              current.op->get_logical_records();
      const LogicalUser merge_close_user(close_op, 0/*idx*/, RegionUsage(
            LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0/*redop*/), close_mask);
      register_dependences(close_op, merge_close_user, current, 
          open_below, closed_users, above_users, cusers, pusers);
      // Now we can remove our references on our local users
      for (LegionList<LogicalUser>::const_iterator it = 
            closed_users.begin(); it != closed_users.end(); it++)
      {
        it->op->remove_mapping_reference(it->gen);
      }
    }

    // If you are looking for LogicalCloser::register_dependences it can 
    // be found in region_tree.cc to make sure that templates are instantiated

    //--------------------------------------------------------------------------
    void LogicalCloser::update_state(LogicalState &state)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(state.owner == root_node);
#endif
      root_node->filter_prev_epoch_users(state, close_mask);
      root_node->filter_curr_epoch_users(state, close_mask, tracing);
      root_node->filter_disjoint_complete_accesses(state, close_mask); 
    }

    //--------------------------------------------------------------------------
    void LogicalCloser::register_close_operations(
                              LegionList<LogicalUser,CURR_LOGICAL_ALLOC> &users)
    //--------------------------------------------------------------------------
    {
      // No need to add mapping references, we did that in 
      // Note we also use the cached generation IDs since the close
      // operations have already been kicked off and might be done
      // LogicalCloser::register_dependences
      const LogicalUser close_user(close_op, merge_close_gen,0/*idx*/,
        RegionUsage(LEGION_READ_WRITE, LEGION_EXCLUSIVE,0/*redop*/),close_mask);
      users.push_back(close_user);
    }

    /////////////////////////////////////////////////////////////
    // KDNode
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RefinementTracker::RefinementTracker(Operation *o,std::set<RtEvent> &events)
      : op(o), context(op->get_context()), applied_events(events)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RefinementTracker::RefinementTracker(const RefinementTracker &rhs)
      : op(rhs.op), context(rhs.context), applied_events(rhs.applied_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RefinementTracker::~RefinementTracker(void)
    //--------------------------------------------------------------------------
    {
      if (!pending_refinements.empty())
      {
        const RegionTreeContext ctx = context->get_context();
        // Note that we walk these refinements in order for control replication
        for (LegionVector<PendingRefinement>::const_iterator it =
             pending_refinements.begin(); it != pending_refinements.end(); it++)
        {
          // Update the disjoint-complete tree for this refinement
          RegionNode *root = it->partition->parent;
          root->refine_disjoint_complete_tree(ctx.get_id(), it->partition, 
                  it->refinement_op, it->refinement_mask, applied_events);
#ifdef DEBUG_LEGION
          // Sanity check that we recorded refinements for
          // all the fields in the refinement mask
          it->refinement_op->verify_refinement_mask(it->refinement_mask);
#endif
          // Trigger this here so it happens in a determinsitic order
          // for control replication. It has to happen after we've got
          // the list of refinements to make from updating the 
          // disjoint-complete tree
          it->refinement_op->trigger_dependence_analysis();
          // Now we can finish the dependence analysis for this refinement
          // Grab these before we end the dependence analysis which will
          // dump the refinement op into the pipeline
          const GenerationID refinement_gen = 
            it->refinement_op->get_generation();
#ifdef LEGION_SPY
          const UniqueID refinement_uid = it->refinement_op->get_unique_op_id();
#endif
          it->refinement_op->end_dependence_analysis();
          // Record that our operation depends on this refinement
          op->register_region_dependence(it->index, it->refinement_op, 
              refinement_gen, 0/*index*/, LEGION_TRUE_DEPENDENCE,
              false/*validates*/, it->refinement_mask);
#ifdef LEGION_SPY
          LegionSpy::log_mapping_dependence(context->get_unique_id(),
              refinement_uid, 0/*index*/, op->get_unique_op_id(), it->index,
              LEGION_TRUE_DEPENDENCE);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    RefinementTracker& RefinementTracker::operator=(const RefinementTracker &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    RefinementOp* RefinementTracker::create_refinement(const LogicalUser &user,
                     PartitionNode *partition, const FieldMask &refinement_mask,
                     const LogicalTraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(op == user.op);
#endif
      RegionNode *root = partition->parent;
#ifdef DEBUG_LEGION_COLLECTIVES
      RefinementOp *refinement_op = context->get_refinement_op(user, root);
#else
      RefinementOp *refinement_op = context->get_refinement_op();
#endif
      refinement_op->initialize(op, user.idx, trace_info, root,refinement_mask);
      pending_refinements.emplace_back(
        PendingRefinement(refinement_op, partition, refinement_mask, user.idx));
      // Start the dependence analysis for this refinement now
      // We'll finish the dependence analysis in the destructor when we
      // know all the region requirements are traversed and we can safely
      // update the disjoint-complete tree
      refinement_op->begin_dependence_analysis();
      return refinement_op;
    }

    //--------------------------------------------------------------------------
    bool RefinementTracker::deduplicate(PartitionNode *child, FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      for (LegionVector<PendingRefinement>::iterator it =
            pending_refinements.begin(); it != pending_refinements.end(); it++)
      {
        if (it->partition->parent != child->parent)
          continue;
        const FieldMask overlap = it->refinement_mask & mask;
        if (!overlap)
          continue;
        if (it->partition != child)
        {
          if (overlap != it->refinement_mask)
          {
            // Filter off the fields we'll do later
            it->refinement_mask -= overlap;
            continue;
          }
          else
            // We can co-opt this refinement op to do this child instead
            it->partition = child;
        }
        mask -= overlap;
        if (!mask)
          return false;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Guard
    /////////////////////////////////////////////////////////////

#ifndef NON_AGGRESSIVE_AGGREGATORS
    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(RtUserEvent post, RtUserEvent applied)
      : guard_postcondition(post), effects_applied(applied),
        releasing_guards(false), read_only_guard(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(const CopyFillGuard &rhs)
      : guard_postcondition(rhs.guard_postcondition), 
        effects_applied(rhs.effects_applied)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }
#else
    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(RtUserEvent applied)
      : effects_applied(applied), releasing_guards(false),read_only_guard(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(const CopyFillGuard &rhs)
      : effects_applied(rhs.effects_applied)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }
#endif

    //--------------------------------------------------------------------------
    CopyFillGuard::~CopyFillGuard(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(releasing_guards); // should have done a release
      assert(guarded_sets.empty());
      assert(remote_release_events.empty());
#endif
    }

    //--------------------------------------------------------------------------
    CopyFillGuard& CopyFillGuard::operator=(const CopyFillGuard &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CopyFillGuard::pack_guard(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      AutoLock g_lock(guard_lock);
      // If we're already releasing a guard then there is no point in sending it 
      if (releasing_guards)
      {
        rez.serialize(RtUserEvent::NO_RT_USER_EVENT);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!guarded_sets.empty());
      assert(effects_applied.exists());
#endif
      // We only ever pack the effects applied event here because once a
      // guard is on a remote node then the guard postcondition is no longer
      // useful since all remote copy fill operations will need to key off
      // the effects applied event to be correct
      rez.serialize(effects_applied);
      rez.serialize<bool>(read_only_guard);
      // Make an event for recording when all the remote events are applied
      RtUserEvent remote_release = Runtime::create_rt_user_event();
      rez.serialize(remote_release);
      remote_release_events.push_back(remote_release);
    }

    //--------------------------------------------------------------------------
    /*static*/ CopyFillGuard* CopyFillGuard::unpack_guard(Deserializer &derez,
                                          Runtime *runtime, EquivalenceSet *set)
    //--------------------------------------------------------------------------
    {
      RtUserEvent effects_applied;
      derez.deserialize(effects_applied);
      if (!effects_applied.exists())
        return NULL;
#ifndef NON_AGGRESSIVE_AGGREGATORS
      // Note we use the effects applied event here twice because all
      // copy-fill aggregators on this node will need to wait for the
      // full effects to be applied of any guards on a remote node
      CopyFillGuard *result = 
        new CopyFillGuard(effects_applied, effects_applied);
#else
      CopyFillGuard *result = new CopyFillGuard(effects_applied);
#endif
      bool read_only_guard;
      derez.deserialize(read_only_guard);
#ifdef DEBUG_LEGION
      if (!result->record_guard_set(set, read_only_guard))
        assert(false);
#else
      result->record_guard_set(set, read_only_guard);
#endif
      RtUserEvent remote_release;
      derez.deserialize(remote_release);
      std::set<RtEvent> release_preconditions;
      result->release_guards(runtime, release_preconditions, true/*defer*/);
      if (!release_preconditions.empty())
        Runtime::trigger_event(remote_release,
            Runtime::merge_events(release_preconditions));
      else
        Runtime::trigger_event(remote_release);
      return result;
    }

    //--------------------------------------------------------------------------
    bool CopyFillGuard::record_guard_set(EquivalenceSet *set, bool read_only)
    //--------------------------------------------------------------------------
    {
      if (releasing_guards)
        return false;
      AutoLock g_lock(guard_lock);
      // Check again after getting the lock to avoid the race
      if (releasing_guards)
        return false;
#ifdef DEBUG_LEGION
      assert(guarded_sets.empty() || (read_only_guard == read_only));
#endif
      guarded_sets.insert(set);
      read_only_guard = read_only;
      return true;
    }

    //--------------------------------------------------------------------------
    bool CopyFillGuard::release_guards(Runtime *rt, std::set<RtEvent> &applied,
                                       bool force_deferral /*=false*/)
    //--------------------------------------------------------------------------
    {
      if (force_deferral || !effects_applied.has_triggered())
      {
        RtUserEvent released = Runtime::create_rt_user_event();
        // Meta-task will take responsibility for deletion
        CopyFillDeletion args(this, implicit_provenance, released);
        rt->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, effects_applied);
        applied.insert(released);
        return false;
      }
      else
        release_guarded_sets(applied);
      return true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillGuard::handle_deletion(const void *args)
    //--------------------------------------------------------------------------
    {
      const CopyFillDeletion *dargs = (const CopyFillDeletion*)args;
      std::set<RtEvent> released_preconditions;
      dargs->guard->release_guarded_sets(released_preconditions);
      if (!released_preconditions.empty())
        Runtime::trigger_event(dargs->released, 
            Runtime::merge_events(released_preconditions));
      else
        Runtime::trigger_event(dargs->released);
      delete dargs->guard;
    }

    //--------------------------------------------------------------------------
    void CopyFillGuard::release_guarded_sets(std::set<RtEvent> &released)
    //--------------------------------------------------------------------------
    {
      std::set<EquivalenceSet*> to_remove;
      {
        AutoLock g_lock(guard_lock);
#ifdef DEBUG_LEGION
        assert(!releasing_guards);
#endif
        releasing_guards = true;
        to_remove.swap(guarded_sets);
        if (!remote_release_events.empty())
        {
          released.insert(remote_release_events.begin(),
                          remote_release_events.end());
          remote_release_events.clear();
        }
      }
      if (!to_remove.empty())
      {
        if (read_only_guard)
        {
          for (std::set<EquivalenceSet*>::const_iterator it = 
                to_remove.begin(); it != to_remove.end(); it++)
            (*it)->remove_read_only_guard(this);
        }
        else
        {
          for (std::set<EquivalenceSet*>::const_iterator it = 
                to_remove.begin(); it != to_remove.end(); it++)
            (*it)->remove_reduction_fill_guard(this);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Aggregator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, Operation *o, 
                                           unsigned idx,CopyFillGuard *previous,
                                           bool t, PredEvent p)
#ifndef NON_AGGRESSIVE_AGGREGATORS
      : CopyFillGuard(Runtime::create_rt_user_event(), 
                      Runtime::create_rt_user_event()),
#else
      :  CopyFillGuard(Runtime::create_rt_user_event()), 
#endif
        forest(f), local_space(f->runtime->address_space), op(o), 
        src_index(idx), dst_index(idx),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_precondition((previous == NULL) ? RtEvent::NO_RT_EVENT :
                            RtEvent(previous->guard_postcondition)),
#else
        guard_precondition((previous == NULL) ? RtEvent::NO_RT_EVENT :
                            RtEvent(previous->effects_applied)),
#endif
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
      // Need to transitively chain effects across aggregators since they
      // each need to summarize all the ones that came before
      if (previous != NULL)
        effects.insert(previous->effects_applied);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, 
                                Operation *o, unsigned src_idx, unsigned dst_idx,
                                CopyFillGuard *previous, bool t, PredEvent p, 
                                RtEvent alternative_precondition)
#ifndef NON_AGGRESSIVE_AGGREGATORS
      : CopyFillGuard(Runtime::create_rt_user_event(), 
                      Runtime::create_rt_user_event()),
#else
      : CopyFillGuard(Runtime::create_rt_user_event()),
#endif
        forest(f), local_space(f->runtime->address_space), op(o), 
        src_index(src_idx), dst_index(dst_idx),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        guard_precondition((previous == NULL) ? alternative_precondition :
                            RtEvent(previous->guard_postcondition)),
#else
        guard_precondition((previous == NULL) ? alternative_precondition:
                            RtEvent(previous->effects_applied)),
#endif
        predicate_guard(p), track_events(t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((previous == NULL) || !alternative_precondition.exists());
#endif
      // Need to transitively chain effects across aggregators since they
      // each need to summarize all the ones that came before
      if (previous != NULL)
        effects.insert(previous->effects_applied);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(const CopyFillAggregator &rhs)
      : CopyFillGuard(rhs), 
        forest(rhs.forest), local_space(rhs.local_space), op(rhs.op),
        src_index(rhs.src_index), dst_index(rhs.dst_index), 
        guard_precondition(rhs.guard_precondition),
        predicate_guard(rhs.predicate_guard), track_events(rhs.track_events)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::~CopyFillAggregator(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
#ifndef NON_AGGRESSIVE_AGGREGATORS
      assert(guard_postcondition.has_triggered());
#endif
      assert(effects_applied.has_triggered());
#endif
      // Remove references from any views that we have
      for (std::set<LogicalView*>::const_iterator it = 
            all_views.begin(); it != all_views.end(); it++)
        if ((*it)->remove_base_valid_ref(AGGREGATOR_REF))
          delete (*it);
      all_views.clear();
      // Delete all our copy updates
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
            mit = sources.begin(); mit != sources.end(); mit++)
      {
        for (FieldMaskSet<Update>::const_iterator it = 
              mit->second.begin(); it != mit->second.end(); it++)
          delete it->first;
      }
      for (std::vector<LegionMap<InstanceView*,
                FieldMaskSet<Update> > >::const_iterator rit = 
            reductions.begin(); rit != reductions.end(); rit++)
      {
        for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
              mit = rit->begin(); mit != rit->end(); mit++)
        {
          for (FieldMaskSet<Update>::const_iterator it = 
                mit->second.begin(); it != mit->second.end(); it++)
            delete it->first;
        }
      }
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator& CopyFillAggregator::operator=(
                                                  const CopyFillAggregator &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::Update::Update(IndexSpaceExpression *exp,
                                const FieldMask &mask, CopyAcrossHelper *helper)
      : expr(exp), src_mask(mask), across_helper(helper)
    //--------------------------------------------------------------------------
    {
      expr->add_base_expression_reference(AGGREGATOR_REF);
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::Update::~Update(void)
    //--------------------------------------------------------------------------
    {
      if (expr->remove_base_expression_reference(AGGREGATOR_REF))
        delete expr;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::record_source_expressions(
                                            InstanceFieldExprs &src_exprs) const
    //--------------------------------------------------------------------------
    {
      FieldMaskSet<IndexSpaceExpression> &exprs = src_exprs[source];  
      FieldMaskSet<IndexSpaceExpression>::iterator finder = 
        exprs.find(expr);
      if (finder == exprs.end())
        exprs.insert(expr, src_mask);
      else
        finder.merge(src_mask);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::CopyUpdate::sort_updates(
                    std::map<InstanceView*, std::vector<CopyUpdate*> > &copies,
                    std::vector<FillUpdate*> &fills)
    //--------------------------------------------------------------------------
    {
      copies[source].push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::record_source_expressions(
                                            InstanceFieldExprs &src_exprs) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, we have no source expressions
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::FillUpdate::sort_updates(
                    std::map<InstanceView*, std::vector<CopyUpdate*> > &copies,
                    std::vector<FillUpdate*> &fills)
    //--------------------------------------------------------------------------
    {
      fills.push_back(this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_update(InstanceView *dst_view,
                                           LogicalView *src_view,
                                           const FieldMask &src_mask,
                                           IndexSpaceExpression *expr,
                                           EquivalenceSet *tracing_eq,
                                           ReductionOpID redop /*=0*/,
                                           CopyAcrossHelper *helper /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!expr->is_empty());
#endif
      if (src_view->is_deferred_view())
      {
#ifdef DEBUG_LEGION
        assert(redop == 0);
#endif
        DeferredView *def_view = src_view->as_deferred_view();
        if (def_view->is_fill_view())
          record_fill(dst_view, def_view->as_fill_view(), src_mask, expr,
                      tracing_eq, helper);
        else
          def_view->flatten(*this, dst_view, src_mask, expr, 
                            tracing_eq, helper);
      }
      else
      {
        InstanceView *inst_view = src_view->as_instance_view();
        update_fields |= src_mask;
        FieldMaskSet<Update> &updates = sources[dst_view];
        record_view(dst_view);
        record_view(inst_view);
        CopyUpdate *update = 
          new CopyUpdate(inst_view, src_mask, expr, redop, helper);
        if (helper == NULL)
          updates.insert(update, src_mask);
        else
          updates.insert(update, helper->convert_src_to_dst(src_mask));
        if (tracing_eq != NULL)
          update_tracing_valid_views(tracing_eq, inst_view, dst_view, 
                                     src_mask, expr, redop);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_updates(InstanceView *dst_view, 
                                    const FieldMaskSet<LogicalView> &src_views,
                                    const FieldMask &src_mask,
                                    IndexSpaceExpression *expr,
                                    EquivalenceSet *tracing_eq,
                                    ReductionOpID redop /*=0*/,
                                    CopyAcrossHelper *helper /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif
      update_fields |= src_mask;
      FieldMaskSet<Update> &updates = sources[dst_view];
      record_view(dst_view);
      if (src_views.size() == 1)
      {
        const LogicalView *view = src_views.begin()->first;
        const FieldMask record_mask = 
          src_views.get_valid_mask() & src_mask;
        if (!!record_mask)
        {
          if (view->is_instance_view())
          {
            InstanceView *inst = view->as_instance_view();
            record_view(inst);
            CopyUpdate *update = 
              new CopyUpdate(inst, record_mask, expr, redop, helper);
            if (helper == NULL)
              updates.insert(update, record_mask);
            else
              updates.insert(update, helper->convert_src_to_dst(record_mask));
            if (tracing_eq != NULL)
              update_tracing_valid_views(tracing_eq, inst, dst_view, 
                                         record_mask, expr, redop);
          }
          else
          {
            DeferredView *def = view->as_deferred_view();
            def->flatten(*this, dst_view, record_mask, expr, 
                         tracing_eq, helper);
          }
        }
      }
      else
      {
        // We have multiple views, so let's sort them
        LegionList<FieldSet<LogicalView*> > view_sets;
        src_views.compute_field_sets(src_mask, view_sets);
        for (LegionList<FieldSet<LogicalView*> >::const_iterator
              vit = view_sets.begin(); vit != view_sets.end(); vit++)
        {
          if (vit->elements.empty())
            continue;
          if (vit->elements.size() == 1)
          {
            // Easy case, just one view so do it  
            const LogicalView *view = *(vit->elements.begin());
            const FieldMask &record_mask = vit->set_mask;
            if (view->is_instance_view())
            {
              InstanceView *inst = view->as_instance_view();
              record_view(inst);
              CopyUpdate *update = 
                new CopyUpdate(inst, record_mask, expr, redop, helper);
              if (helper == NULL)
                updates.insert(update, record_mask);
              else
                updates.insert(update, helper->convert_src_to_dst(record_mask));
              if (tracing_eq != NULL)
                update_tracing_valid_views(tracing_eq, inst, dst_view, 
                                           record_mask, expr, redop);
            }
            else
            {
              DeferredView *def = view->as_deferred_view();
              def->flatten(*this, dst_view, record_mask, expr,
                           tracing_eq, helper);
            }
          }
          else
          {
            // Sort the views, prefer fills, then instances, then deferred
            FillView *fill = NULL;
            DeferredView *deferred = NULL;
            std::vector<InstanceView*> instances;
            for (std::set<LogicalView*>::const_iterator it = 
                  vit->elements.begin(); it != vit->elements.end(); it++)
            {
              if (!(*it)->is_instance_view())
              {
                DeferredView *def = (*it)->as_deferred_view();
                if (!def->is_fill_view())
                {
                  if (deferred == NULL)
                    deferred = def;
                }
                else
                {
                  fill = def->as_fill_view();
                  // Break out since we found what we're looking for
                  break;
                }
              }
              else
                instances.push_back((*it)->as_instance_view());
            }
            if (fill != NULL)
              record_fill(dst_view, fill, vit->set_mask, expr,
                          tracing_eq, helper);
            else if (!instances.empty())
            {
              if (instances.size() == 1)
              {
                // Easy, just one instance to use
                InstanceView *inst = instances.back();
                record_view(inst);
                CopyUpdate *update = 
                  new CopyUpdate(inst, vit->set_mask, expr, redop, helper);
                if (helper == NULL)
                  updates.insert(update, vit->set_mask);
                else
                  updates.insert(update, 
                      helper->convert_src_to_dst(vit->set_mask));
                if (tracing_eq != NULL)
                  update_tracing_valid_views(tracing_eq, inst, dst_view,
                                             vit->set_mask, expr, redop);
              }
              else
              {
                // Hard, multiple potential sources,
                // ask the mapper which one to use
                // First though check to see if we've already asked it
                bool found = false;
                std::map<InstanceView*,LegionVector<SourceQuery>>::
                  const_iterator finder = mapper_queries.find(dst_view);
                if (finder != mapper_queries.end())
                {
                  for (LegionVector<SourceQuery>::const_iterator qit = 
                        finder->second.begin(); qit != 
                        finder->second.end(); qit++)
                  {
                    if (qit->matches(vit->set_mask, instances))
                    {
                      found = true;
                      InstanceView *result = instances[qit->ranking.front()];
                      record_view(result);
                      CopyUpdate *update = new CopyUpdate(result,
                                    qit->query_mask, expr, redop, helper);
                      if (helper == NULL)
                        updates.insert(update, qit->query_mask);
                      else
                        updates.insert(update, 
                            helper->convert_src_to_dst(qit->query_mask));
                      if (tracing_eq != NULL)
                        update_tracing_valid_views(tracing_eq, result,
                                dst_view, qit->query_mask, expr, redop);
                      break;
                    }
                  }
                }
                if (!found)
                {
                  // If we didn't find the query result we need to do
                  // it for ourself, start by constructing the inputs
                  InstanceRef dst(dst_view->get_manager(),
                      helper == NULL ? vit->set_mask : 
                        helper->convert_src_to_dst(vit->set_mask));
                  InstanceSet sources(instances.size());
                  unsigned src_idx = 0;
                  for (std::vector<InstanceView*>::const_iterator it = 
                        instances.begin(); it != instances.end(); it++)
                    sources[src_idx++] = InstanceRef((*it)->get_manager(),
                                                     vit->set_mask);
                  std::vector<unsigned> ranking;
                  // Always use the source index for selecting sources
                  op->select_sources(src_index, dst, sources, ranking);
                  // Check to make sure that the ranking has sound output
                  unsigned count = 0;
                  std::vector<bool> unique_indexes(instances.size(), false);
                  for (std::vector<unsigned>::iterator it =
                        ranking.begin(); it != ranking.end(); /*nothing*/)
                  {
                    if (((*it) < unique_indexes.size()) && !unique_indexes[*it])
                    {
                      unique_indexes[*it] = true;
                      count++;
                      it++;
                    }
                    else // remove duplicates and out of bound entries
                      it = ranking.erase(it);
                  }
                  if (count < unique_indexes.size())
                  {
                    for (unsigned idx = 0; idx < unique_indexes.size(); idx++)
                      if (!unique_indexes[idx])
                        ranking.push_back(idx);
                  }
                  // We know that which ever one was chosen first is
                  // the one that satisfies all our fields since all
                  // these instances are valid for all fields
                  InstanceView *result = instances[ranking.front()];
                  // Record the update
                  record_view(result);
                  CopyUpdate *update = new CopyUpdate(result, vit->set_mask,
                                                      expr, redop, helper);
                  if (helper == NULL)
                    updates.insert(update, vit->set_mask);
                  else
                    updates.insert(update, 
                        helper->convert_src_to_dst(vit->set_mask));
                  if (tracing_eq != NULL)
                    update_tracing_valid_views(tracing_eq, result, dst_view,
                                               vit->set_mask, expr, redop);
                  // Save the result for the future
                  mapper_queries[dst_view].push_back(
                      SourceQuery(std::move(instances), 
                        std::move(ranking), vit->set_mask));
                }
              }
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(deferred != NULL);
#endif
              deferred->flatten(*this, dst_view, vit->set_mask, expr, 
                                tracing_eq, helper);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_partial_updates(InstanceView *dst_view, 
                 const LegionMap<LogicalView*,
                                 FieldMaskSet<IndexSpaceExpression> >&src_views,
                          const FieldMask &src_mask, IndexSpaceExpression *expr, 
                          EquivalenceSet *tracing_eq, ReductionOpID redop, 
                          CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!src_mask);
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif
      update_fields |= src_mask;
      record_view(dst_view);
      std::vector<InstanceView*> instances;
      std::vector<DeferredView*> deferred;
      FieldMaskSet<IndexSpaceExpression> remainders;
      remainders.insert(expr, src_mask);
      // Issue fills from immediately, otherwise record instances so that
      // we can ask the mapper what order it wants us to issue copies from
      for (LegionMap<LogicalView*,
            FieldMaskSet<IndexSpaceExpression> >::const_iterator vit =
            src_views.begin(); vit != src_views.end(); vit++)
      {
        FieldMask view_overlap = 
          vit->second.get_valid_mask() & remainders.get_valid_mask();;
        if (!view_overlap)
          continue;
        if (!vit->first->is_instance_view())
        {
          DeferredView *def = vit->first->as_deferred_view();
          if (def->is_fill_view())
          {
            // Skip any fills if we're doing a reduction, we only care
            // about valid instances here
            if (redop > 0)
              continue;
            FillView *fill = def->as_fill_view();
            // Join in the fields to see what overlaps
            LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
              FieldMask> fill_exprs;
            unique_join_on_field_mask_sets(remainders, vit->second, fill_exprs);
            bool need_tighten = false;
            for (LegionMap<std::pair<IndexSpaceExpression*,
                  IndexSpaceExpression*>,FieldMask>::const_iterator 
                  it = fill_exprs.begin(); it != fill_exprs.end(); it++)
            {
              IndexSpaceExpression *overlap =
               forest->intersect_index_spaces(it->first.first,it->first.second);
              const size_t overlap_size = overlap->get_volume();
              if (overlap_size == 0)
                continue;
              FieldMaskSet<IndexSpaceExpression>::iterator finder = 
                remainders.find(it->first.first);
#ifdef DEBUG_LEGION
              assert(finder != remainders.end());
#endif
              finder.filter(it->second);
              if (!finder->second)
                remainders.erase(finder);
              if (overlap_size < it->first.first->get_volume())
              {
                if (overlap_size == it->first.second->get_volume())
                  record_fill(dst_view, fill, it->second, 
                              it->first.second, tracing_eq);
                else
                  record_fill(dst_view, fill, it->second, overlap, tracing_eq);
                // Compute the difference
                IndexSpaceExpression *diff_expr = 
                  forest->subtract_index_spaces(it->first.first, overlap); 
                remainders.insert(diff_expr, it->second);
              }
              else // completely covers remainder expression
              {
                record_fill(dst_view, fill, it->second, 
                            it->first.first, tracing_eq);
                if (remainders.empty())
                  return;
                need_tighten = true;
              }
            }
            if (need_tighten)
              remainders.tighten_valid_mask();
          }
          else
            deferred.push_back(def);
        }
        else
          instances.push_back(vit->first->as_instance_view());
      }
      // If we get here, next try to sort the instances into whatever order
      // the mapper wants us to try to issue copies from them
      if (!instances.empty())
      {
        std::vector<unsigned> ranking;
        if (instances.size() > 1)
        {
          // Check to see if we can find it, if not then we'll need to
          // ask the mapper to compute it
          bool found = false;
          std::map<InstanceView*,LegionVector<SourceQuery>>::
            const_iterator finder = mapper_queries.find(dst_view);
          if (finder != mapper_queries.end())
          {
            for (LegionVector<SourceQuery>::const_iterator qit = 
                  finder->second.begin(); qit != 
                  finder->second.end(); qit++)
            {
              if (qit->matches(src_mask, instances))
              {
                found = true;
                ranking = qit->ranking;
                break;
              }
            }
          }
          if (!found)
          {
            InstanceRef dst(dst_view->get_manager(), 
                            remainders.get_valid_mask());
            InstanceSet sources(instances.size());
            unsigned src_idx = 0;
            for (std::vector<InstanceView*>::const_iterator it = 
                  instances.begin(); it != instances.end(); it++)
            {
              LegionMap<LogicalView*,FieldMaskSet<IndexSpaceExpression> >::
                const_iterator finder = src_views.find(*it);
#ifdef DEBUG_LEGION
              assert(finder != src_views.end());
#endif
              sources[src_idx++] = InstanceRef((*it)->get_manager(),
                 finder->second.get_valid_mask() & remainders.get_valid_mask());
            }
            // Always use the source index for selecting sources
            op->select_sources(src_index, dst, sources, ranking);
            // Check to make sure that the ranking has sound output
            unsigned count = 0;
            std::vector<bool> unique_indexes(instances.size(), false);
            for (std::vector<unsigned>::iterator it =
                  ranking.begin(); it != ranking.end(); /*nothing*/)
            {
              if (((*it) < unique_indexes.size()) && !unique_indexes[*it])
              {
                unique_indexes[*it] = true;
                count++;
                it++;
              }
              else // remove duplicates and out of bound entries
                it = ranking.erase(it);
            }
            if (count < unique_indexes.size())
            {
              for (unsigned idx = 0; idx < unique_indexes.size(); idx++)
                if (!unique_indexes[idx])
                  ranking.push_back(idx);
            }
          }
        }
        else
          ranking.push_back(0);
        for (unsigned idx = 0; idx < ranking.size(); idx++)
        {
          InstanceView *inst = instances[ranking[idx]];
          LegionMap<LogicalView*,FieldMaskSet<IndexSpaceExpression> >::
              const_iterator finder = src_views.find(inst);
#ifdef DEBUG_LEGION
          assert(finder != src_views.end());
#endif
          LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
            FieldMask> src_expressions;
          unique_join_on_field_mask_sets(remainders, finder->second,
                                         src_expressions);
          bool need_tighten = false;
          for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask>::const_iterator it = 
                src_expressions.begin(); it != src_expressions.end(); it++)
          {
            IndexSpaceExpression *overlap = 
              forest->intersect_index_spaces(it->first.first, it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              remainders.find(it->first.first);
#ifdef DEBUG_LEGION
            assert(finder != remainders.end());
#endif
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                record_update(dst_view, inst, it->second, it->first.second, 
                                tracing_eq, redop, across_helper);
              else
                record_update(dst_view, inst, it->second, overlap, 
                              tracing_eq, redop, across_helper);
              // Compute the difference
              IndexSpaceExpression *diff_expr = 
                forest->subtract_index_spaces(it->first.first, overlap);
              remainders.insert(diff_expr, it->second);
            }
            else // completely covers remainder expression
            {
              record_update(dst_view, inst, it->second, it->first.first, 
                            tracing_eq, redop, across_helper);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
      }
      if (!deferred.empty())
      {
#ifdef DEBUG_LEGION
        assert(redop == 0);
#endif
        for (unsigned idx = 0; idx < deferred.size(); idx++)
        {
          DeferredView *def = deferred[idx];
          LegionMap<LogicalView*,FieldMaskSet<IndexSpaceExpression> >::
              const_iterator finder = src_views.find(def);
#ifdef DEBUG_LEGION
          assert(finder != src_views.end());
#endif
          LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
            FieldMask> src_expressions;
          unique_join_on_field_mask_sets(remainders, finder->second,
                                         src_expressions);
          bool need_tighten = false;
          for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                FieldMask>::const_iterator it = 
                src_expressions.begin(); it != src_expressions.end(); it++)
          {
            IndexSpaceExpression *overlap = 
              forest->intersect_index_spaces(it->first.first, it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              remainders.find(it->first.first);
#ifdef DEBUG_LEGION
            assert(finder != remainders.end());
#endif
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                def->flatten(*this, dst_view, it->second, it->first.second,
                             tracing_eq, across_helper);
              else
                def->flatten(*this, dst_view, it->second, overlap,
                             tracing_eq, across_helper);
              // Compute the difference
              IndexSpaceExpression *diff_expr = 
                forest->subtract_index_spaces(it->first.first, overlap);
              remainders.insert(diff_expr, it->second);
            }
            else // completely covers remainder expression
            {
              def->flatten(*this, dst_view, it->second, it->first.first, 
                           tracing_eq, across_helper);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_fill(InstanceView *dst_view,
                                         FillView *src_view,
                                         const FieldMask &fill_mask,
                                         IndexSpaceExpression *expr,
                                         EquivalenceSet *tracing_eq,
                                         CopyAcrossHelper *helper /*=NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!fill_mask);
      assert(!expr->is_empty());
#endif
      update_fields |= fill_mask;
      record_view(src_view);
      record_view(dst_view);
      FillUpdate *update = new FillUpdate(src_view, fill_mask, expr, helper); 
      if (helper == NULL)
        sources[dst_view].insert(update, fill_mask);
      else
        sources[dst_view].insert(update, helper->convert_src_to_dst(fill_mask));
      if (tracing_eq != NULL)
      {
        if (dst_view->is_reduction_view())
          tracing_eq->update_tracing_anti_views(dst_view, expr, fill_mask);
        else
          update_tracing_valid_views(tracing_eq, src_view, dst_view,
                                     fill_mask, expr, 0/*redop*/);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_reductions(InstanceView *dst_view,
                                  const std::list<std::pair<ReductionView*,
                                            IndexSpaceExpression*> > &src_views,
                                  const unsigned src_fidx,
                                  const unsigned dst_fidx,
                                  EquivalenceSet *tracing_eq,
                                  CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!src_views.empty());
#endif 
      update_fields.set_bit(src_fidx);
      record_view(dst_view);
      const std::pair<InstanceView*,unsigned> dst_key(dst_view, dst_fidx);
      std::vector<ReductionOpID> &redop_epochs = reduction_epochs[dst_key];
      FieldMask src_mask, dst_mask;
      src_mask.set_bit(src_fidx);
      dst_mask.set_bit(dst_fidx);
      // Always start scanning from the first redop index
      unsigned redop_index = 0;
      for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
            const_iterator it = src_views.begin(); it != src_views.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(!it->second->is_empty());
#endif
        record_view(it->first);
        const ReductionOpID redop = it->first->get_redop();
        CopyUpdate *update =
          new CopyUpdate(it->first,src_mask,it->second,redop,across_helper);
        // Ignore shadows when tracing, we only care about the normal
        // preconditions and postconditions for the copies
        if (tracing_eq != NULL)
          update_tracing_valid_views(tracing_eq, it->first, dst_view, 
                                     src_mask, it->second, redop);
        // Scan along looking for a reduction op epoch that matches
        while ((redop_index < redop_epochs.size()) &&
                (redop_epochs[redop_index] != redop))
          redop_index++;
        if (redop_index == redop_epochs.size())
        {
#ifdef DEBUG_LEGION
          assert(redop_index <= reductions.size());
#endif
          // Start a new redop epoch if necessary
          redop_epochs.push_back(redop);
          if (reductions.size() == redop_index)
            resize_reductions(redop_index + 1);
        }
        reductions[redop_index][dst_view].insert(update, dst_mask);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::resize_reductions(size_t new_size)
    //--------------------------------------------------------------------------
    {
      std::vector<LegionMap<InstanceView*,FieldMaskSet<Update> > >
        new_reductions(new_size);
      for (unsigned idx = 0; idx < reductions.size(); idx++)
        new_reductions[idx].swap(reductions[idx]);
      reductions.swap(new_reductions);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_updates(const PhysicalTraceInfo &trace_info,
                      ApEvent precondition, const bool restricted_output,
                      const bool manage_dst_events,
                      std::map<InstanceView*,std::vector<ApEvent> > *dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!sources.empty() || !reductions.empty());
#endif
      if (guard_precondition.exists() && !guard_precondition.has_triggered())
      {
        CopyFillAggregation args(this, trace_info, precondition,
                                 manage_dst_events, restricted_output,
                                 op->get_unique_op_id(), dst_events);
        op->runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_DEFERRED_PRIORITY, guard_precondition);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!guard_precondition.exists() || 
              guard_precondition.has_triggered());
#endif
#ifndef NON_AGGRESSIVE_AGGREGATORS
      std::set<RtEvent> recorded_events;
#endif
      // Perform updates from any sources first
      if (!sources.empty())
        perform_updates(sources, trace_info, precondition, 
#ifdef NON_AGGRESSIVE_AGGREGATORS
            effects,
#else
            recorded_events,
#endif
            -1/*redop index*/, manage_dst_events, restricted_output,dst_events);
      // Then apply any reductions that we might have
      if (!reductions.empty())
      {
        // Skip any passes that we might have already done
        for (unsigned idx = 0; idx < reductions.size(); idx++)
          perform_updates(reductions[idx], trace_info, precondition,
#ifdef NON_AGGRESSIVE_AGGREGATORS
                          effects,
#else
                          recorded_events,
#endif
                          idx/*redop index*/, manage_dst_events,
                          restricted_output, dst_events);
      }
#ifndef NON_AGGRESSIVE_AGGREGATORS
      if (!recorded_events.empty())
        Runtime::trigger_event(guard_postcondition,
            Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(guard_postcondition);
      // Make sure the guard postcondition is chained on the deletion
      if (!effects.empty())
      {
        effects.insert(guard_postcondition);
        Runtime::trigger_event(effects_applied,
            Runtime::merge_events(effects));
      }
      else
        Runtime::trigger_event(effects_applied, guard_postcondition);
#else
      // We can also trigger our effects event once the effects are applied
      if (!effects.empty())
        Runtime::trigger_event(effects_applied,
            Runtime::merge_events(effects));
      else
        Runtime::trigger_event(effects_applied);
#endif
    } 

    //--------------------------------------------------------------------------
    ApEvent CopyFillAggregator::summarize(const PhysicalTraceInfo &info) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(track_events);
#endif
      if (!events.empty())
        return Runtime::merge_events(&info, events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_view(LogicalView *new_view)
    //--------------------------------------------------------------------------
    {
      std::pair<std::set<LogicalView*>::iterator,bool> result = 
        all_views.insert(new_view);
      if (result.second)
        new_view->add_base_valid_ref(AGGREGATOR_REF);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::update_tracing_valid_views(
                          EquivalenceSet *tracing_eq, LogicalView *src, 
                          LogicalView *dst, const FieldMask &mask, 
                          IndexSpaceExpression *expr, ReductionOpID redop) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tracing_eq != NULL);
#endif
      const RegionUsage read_usage(LEGION_READ_PRIV, LEGION_EXCLUSIVE, 0);
      tracing_eq->update_tracing_valid_views(src, expr, read_usage, mask, 
                                             false/*invalidates*/);
      // Only record the destination if this is not a copy across
      if (src_index == dst_index)
      {
        const RegionUsage write_usage((redop > 0) ? LEGION_REDUCE_PRIV : 
            LEGION_WRITE_PRIV, LEGION_EXCLUSIVE, redop);
        tracing_eq->update_tracing_valid_views(dst, expr, write_usage, mask, 
                                    false/*do not invalidate copies here*/);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::perform_updates(
         const LegionMap<InstanceView*,FieldMaskSet<Update> > &updates,
         const PhysicalTraceInfo &trace_info, const ApEvent precondition,
         std::set<RtEvent> &recorded_events, const int redop_index,
         const bool manage_dst_events, const bool restricted_output,
         std::map<InstanceView*,std::vector<ApEvent> > *dst_events)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> *target_events = NULL;
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
            uit = updates.begin(); uit != updates.end(); uit++)
      {
        ApEvent dst_precondition = precondition;
        // In the case where we're not managing destination events
        // then we need to incorporate any event postconditions from
        // previous passes as part of the preconditions for this pass
        if (!manage_dst_events)
        {
#ifdef DEBUG_LEGION
          assert(dst_events != NULL);
#endif
          // This only happens in the case of across copies
          std::map<InstanceView*,std::vector<ApEvent> >::iterator finder =
            dst_events->find(uit->first);
#ifdef DEBUG_LEGION
          assert(finder != dst_events->end());
#endif
          if (!finder->second.empty())
          {
            // Update our precondition to incude the copies from 
            // any previous passes that we performed
            finder->second.push_back(precondition);
            dst_precondition =
              Runtime::merge_events(&trace_info, finder->second);
            // Clear this for the next iteration
            // It's not obvious why this safe, but it is
            // We are guaranteed to issue at least one fill/copy that
            // will depend on this and therefore either test that it
            // has triggered or record itself back in the set of events
            // which gives us a transitive precondition
            finder->second.clear();
          }
          target_events = &finder->second;
        }
        // Group by fields first
        LegionList<FieldSet<Update*> > field_groups;
        uit->second.compute_field_sets(FieldMask(), field_groups);
        for (LegionList<FieldSet<Update*> >::const_iterator fit = 
              field_groups.begin(); fit != field_groups.end(); fit++)
        {
          const FieldMask &dst_mask = fit->set_mask;
          // Now that we have the src mask for these operations group 
          // them into fills and copies
          std::vector<FillUpdate*> fills;
          std::map<InstanceView* /*src*/,std::vector<CopyUpdate*> > copies;
          for (std::set<Update*>::const_iterator it = fit->elements.begin();
                it != fit->elements.end(); it++)
            (*it)->sort_updates(copies, fills);
          // Issue the copies and fills
          if (!fills.empty())
            issue_fills(uit->first, fills, recorded_events, dst_precondition,
                        dst_mask, trace_info, manage_dst_events,
                        restricted_output, target_events);
          if (!copies.empty())
            issue_copies(uit->first, copies, recorded_events, dst_precondition,
                         dst_mask, trace_info, manage_dst_events,
                         restricted_output, target_events);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_fills(InstanceView *target,
                                         const std::vector<FillUpdate*> &fills,
                                         std::set<RtEvent> &recorded_events,
                                         const ApEvent precondition, 
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool manage_dst_events,
                                         const bool restricted_output,
                                         std::vector<ApEvent> *dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fills.empty());
      assert(!!fill_mask); 
      // Should only have across helper on across copies
      assert((fills[0]->across_helper == NULL) || !manage_dst_events);
#endif
      PhysicalManager *manager = target->get_manager();
      if (fills.size() == 1)
      {
        FillUpdate *update = fills[0];
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        // Should cover all the fields
        if (fills[0]->across_helper != NULL)
        {
          const FieldMask src_mask =
            fills[0]->across_helper->convert_dst_to_src(fill_mask);
          assert(!(src_mask - update->src_mask));
        }
        else
        {
          assert(!(fill_mask - update->src_mask));
        }
#endif
#endif
        IndexSpaceExpression *fill_expr = update->expr;
        FillView *fill_view = update->source;
        const ApEvent result = manager->fill_from(fill_view, target, 
                                                  precondition, predicate_guard,
                                                  fill_expr, op, dst_index,
                                                  fill_mask, trace_info, 
                                                  recorded_events, effects,
                                                  fills[0]->across_helper,
                                                  manage_dst_events,
                                                  restricted_output);
        if (result.exists())
        {
          if (track_events)
            events.insert(result);
          if (dst_events != NULL)
            dst_events->push_back(result);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        FieldMask src_mask;
        if (fills[0]->across_helper != NULL)
          src_mask = fills[0]->across_helper->convert_dst_to_src(fill_mask);
        else
          src_mask = fill_mask;
        // These should all have had the same across helper
        for (unsigned idx = 1; idx < fills.size(); idx++)
          assert(fills[idx]->across_helper == fills[0]->across_helper);
#endif
#endif
        std::map<FillView*,std::set<IndexSpaceExpression*> > exprs;
        for (std::vector<FillUpdate*>::const_iterator it = 
              fills.begin(); it != fills.end(); it++)
        {
#ifdef DEBUG_LEGION
          // Should cover all the fields
          assert(!(src_mask - (*it)->src_mask));
          // Should also have the same across helper as the first one
          assert(fills[0]->across_helper == (*it)->across_helper);
#endif
          exprs[(*it)->source].insert((*it)->expr);
        }
        for (std::map<FillView*,std::set<IndexSpaceExpression*> >::
              const_iterator it = exprs.begin(); it != exprs.end(); it++)
        {
          IndexSpaceExpression *fill_expr = (it->second.size() == 1) ?
            *(it->second.begin()) : forest->union_index_spaces(it->second);
          // See if we have any work to do for tracing
          const ApEvent result = manager->fill_from(it->first, target,
                                                    precondition,
                                                    predicate_guard, fill_expr,
                                                    op, dst_index, 
                                                    fill_mask, trace_info,
                                                    recorded_events, effects, 
                                                    fills[0]->across_helper,
                                                    manage_dst_events,
                                                    restricted_output);
          if (result.exists())
          {
            if (track_events)
              events.insert(result);
            if (dst_events != NULL)
              dst_events->push_back(result);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_copies(InstanceView *target, 
               const std::map<InstanceView*,std::vector<CopyUpdate*> > &copies,
               std::set<RtEvent> &recorded_events,
               const ApEvent precondition, const FieldMask &copy_mask,
               const PhysicalTraceInfo &trace_info,
               const bool manage_dst_events, const bool restricted_output,
               std::vector<ApEvent> *dst_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!copies.empty());
      assert(!!copy_mask);
      assert((src_index == dst_index) || !manage_dst_events);
#endif
      PhysicalManager *target_manager = target->get_manager();
      for (std::map<InstanceView*,std::vector<CopyUpdate*> >::const_iterator
            cit = copies.begin(); cit != copies.end(); cit++)
      {
#ifdef DEBUG_LEGION
        assert(!cit->second.empty());
        // Should only have across helpers for across copies
        assert((cit->second[0]->across_helper == NULL) || !manage_dst_events);
#endif
        if (cit->second.size() == 1)
        {
          // Easy case of a single update copy
          CopyUpdate *update = cit->second[0];
#ifdef DEBUG_LEGION
#ifndef NDEBUG
          if (cit->second[0]->across_helper != NULL)
          {
            const FieldMask src_mask =
              cit->second[0]->across_helper->convert_dst_to_src(copy_mask);
            assert(!(src_mask - update->src_mask));
          }
          else
          {
            // Should cover all the fields
            assert(!(copy_mask - update->src_mask));
          }
#endif
#endif
          InstanceView *source = update->source;
          IndexSpaceExpression *copy_expr = update->expr;
          const ApEvent result = target_manager->copy_from(source, target,
                                    source->get_manager(), precondition,
                                    predicate_guard, update->redop, copy_expr,
                                    op, manage_dst_events ? dst_index
                                      : src_index, copy_mask, trace_info,
                                    recorded_events, effects,
                                    cit->second[0]->across_helper,
                                    manage_dst_events, restricted_output);
          if (result.exists())
          {
            if (track_events)
              events.insert(result);
            if (dst_events != NULL)
              dst_events->push_back(result);
          }
        }
        else
        {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
          FieldMask src_mask;
          if (cit->second[0]->across_helper != NULL)
            src_mask = 
              cit->second[0]->across_helper->convert_dst_to_src(copy_mask);
          else
            src_mask = copy_mask;
#endif
#endif
          // Have to group by source instances in order to merge together
          // different index space expressions for the same copy
          std::map<InstanceView*,std::set<IndexSpaceExpression*> > fused_exprs;
          const ReductionOpID redop = cit->second[0]->redop;
          for (std::vector<CopyUpdate*>::const_iterator it = 
                cit->second.begin(); it != cit->second.end(); it++)
          {
#ifdef DEBUG_LEGION
            // Should cover all the fields
            assert(!(src_mask - (*it)->src_mask));
            // Should have the same redop
            assert(redop == (*it)->redop);
            // Should also have the same across helper as the first one
            assert(cit->second[0]->across_helper == (*it)->across_helper);
#endif
            fused_exprs[(*it)->source].insert((*it)->expr);
          }
          for (std::map<InstanceView*,std::set<IndexSpaceExpression*> >::
               iterator it = fused_exprs.begin(); it != fused_exprs.end(); it++)
          {
            IndexSpaceExpression *copy_expr = (it->second.size() == 1) ?
                *(it->second.begin()) : forest->union_index_spaces(it->second);
            const ApEvent result = target_manager->copy_from(it->first, target,
                                    it->first->get_manager(), precondition,
                                    predicate_guard, redop, copy_expr, op,
                                    manage_dst_events ? dst_index : 
                                      src_index, copy_mask, trace_info,
                                    recorded_events, effects,
                                    cit->second[0]->across_helper,
                                    manage_dst_events, restricted_output);
            if (result.exists())
            {
              if (track_events)
                events.insert(result);
              if (dst_events != NULL)
                dst_events->push_back(result);
            }
          }
        }
      }
    } 

    //--------------------------------------------------------------------------
    /*static*/ void CopyFillAggregator::handle_aggregation(const void *args)
    //--------------------------------------------------------------------------
    {
      const CopyFillAggregation *cfargs = (const CopyFillAggregation*)args;
      cfargs->aggregator->issue_updates(*cfargs, cfargs->pre, 
          cfargs->restricted_output, cfargs->manage_dst_events,
          cfargs->dst_events);
      cfargs->remove_recorder_reference();
      if (cfargs->dst_events != NULL)
        delete cfargs->dst_events;
    } 

    /////////////////////////////////////////////////////////////
    // Physical Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(Runtime *rt, Operation *o, unsigned idx, 
                          IndexSpaceExpression *e, bool h, CollectiveMapping *m)
      : previous(rt->address_space), original_source(rt->address_space),
        runtime(rt), analysis_expr(e), collective_mapping(m), op(o), index(idx),
        owns_op(false), on_heap(h), recorded_instances(NULL), restricted(false), 
        parallel_traversals(false)
    //--------------------------------------------------------------------------
    {
      if (collective_mapping != NULL)
        collective_mapping->add_reference();
      analysis_expr->add_base_expression_reference(PHYSICAL_ANALYSIS_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(Runtime *rt, AddressSpaceID source, 
                               AddressSpaceID prev, Operation *o, unsigned idx,
                               IndexSpaceExpression *e, bool h, 
                               CollectiveMapping *mapping)
      : previous(prev), original_source(source), runtime(rt), analysis_expr(e),
        collective_mapping(mapping), op(o), index(idx), owns_op(true), 
        on_heap(h), recorded_instances(NULL), restricted(false), 
        parallel_traversals(false)
    //--------------------------------------------------------------------------
    {
      analysis_expr->add_base_expression_reference(PHYSICAL_ANALYSIS_REF);
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(const PhysicalAnalysis &rhs)
      : previous(0), original_source(0), runtime(NULL), analysis_expr(NULL),
        collective_mapping(NULL), op(NULL), index(0), owns_op(false), 
        on_heap(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::~PhysicalAnalysis(void)
    //--------------------------------------------------------------------------
    {
      if (analysis_expr->remove_base_expression_reference(
                                    PHYSICAL_ANALYSIS_REF))
        delete analysis_expr;
      if ((collective_mapping != NULL) && 
          collective_mapping->remove_reference())
        delete collective_mapping;
      if (recorded_instances != NULL)
        delete recorded_instances;
      if (owns_op && (op != NULL))
        delete op;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::traverse(EquivalenceSet *set,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    RtEvent precondition/*= NO_EVENT*/,
                                    const bool already_deferred /* = false*/)
    //--------------------------------------------------------------------------
    {
      if (!precondition.exists() || precondition.has_triggered())
      {
        if (set->set_expr == analysis_expr)
          perform_traversal(set, analysis_expr, true/*covers*/, mask, 
                            deferral_events, applied_events, already_deferred);
        else if (!set->set_expr->is_empty())
        {
          IndexSpaceExpression *expr = 
           runtime->forest->intersect_index_spaces(set->set_expr,analysis_expr);
          if (expr->is_empty())
            return;
          // Check to see this expression covers the equivalence set
          // If it does then we can use original set expression
          if (expr->get_volume() == set->set_expr->get_volume())
            perform_traversal(set, set->set_expr, true/*covers*/, mask, 
                              deferral_events, applied_events,already_deferred);
          else
            perform_traversal(set, expr, false/*covers*/, mask, deferral_events,
                              applied_events, already_deferred);
        }
        else
          perform_traversal(set, set->set_expr, true/*covers*/, mask,
                            deferral_events, applied_events, already_deferred);
      }
      else
        // This has to be the first time through and isn't really
        // a deferral of an the traversal since we haven't even
        // started the traversal yet
        defer_traversal(precondition, set, mask, deferral_events,
            applied_events, RtUserEvent::NO_RT_USER_EVENT, already_deferred);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::defer_traversal(RtEvent precondition,
                                           EquivalenceSet *set,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           RtUserEvent deferral_event,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Make sure that we record that this has parallel traversals
      const DeferPerformTraversalArgs args(this, set, mask, 
                                           deferral_event, already_deferred);
      runtime->issue_runtime_meta_task(args, 
          LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      deferral_events.insert(args.done_event);
      applied_events.insert(args.applied_event);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_remote(RtEvent precondition,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent PhysicalAnalysis::perform_updates(RtEvent precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent PhysicalAnalysis::perform_output(RtEvent precondition,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // only called by derived classes
      assert(false);
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::process_remote_instances(Deserializer &derez,
                                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_views;
      derez.deserialize(num_views);
      AutoLock a_lock(*this);
      if (recorded_instances == NULL)
        recorded_instances = new FieldMaskSet<LogicalView>();
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID view_did;
        derez.deserialize(view_did);
        RtEvent ready;  
        LogicalView *view = 
          runtime->find_or_request_logical_view(view_did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        recorded_instances->insert(view, mask);
      }
      bool remote_restrict;
      derez.deserialize(remote_restrict);
      if (remote_restrict)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::process_local_instances(
            const FieldMaskSet<LogicalView> &views, const bool local_restricted)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(*this);
      if (recorded_instances == NULL)
        recorded_instances = new FieldMaskSet<LogicalView>();
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            views.begin(); it != views.end(); it++)
        if (it->first->is_instance_view())
          recorded_instances->insert(it->first, it->second);
      if (local_restricted)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::filter_remote_expressions(
                                      FieldMaskSet<IndexSpaceExpression> &exprs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!remote_sets.empty());
#endif
      FieldMaskSet<IndexSpaceExpression> remote_exprs; 
      for (LegionMap<AddressSpaceID,FieldMaskSet<EquivalenceSet> >::
            const_iterator rit = remote_sets.begin(); 
            rit != remote_sets.end(); rit++)
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              rit->second.begin(); it != rit->second.end(); it++)
          remote_exprs.insert(it->first->set_expr, it->second);
      FieldMaskSet<IndexSpaceExpression> to_add;
      std::vector<IndexSpaceExpression*> to_remove;
      if (remote_exprs.size() > 1)
      {
        LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
        remote_exprs.compute_field_sets(FieldMask(), field_sets);
        for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
              fit = field_sets.begin(); fit != field_sets.end(); fit++)
        {
          IndexSpaceExpression *remote_expr = (fit->elements.size() == 1) ?
            *(fit->elements.begin()) : 
            runtime->forest->union_index_spaces(fit->elements);
          for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
                exprs.begin(); it != exprs.end(); it++)
          {
            const FieldMask overlap = it->second & fit->set_mask;
            if (!overlap)
              continue;
            IndexSpaceExpression *diff = 
              runtime->forest->subtract_index_spaces(it->first, remote_expr);
            if (!diff->is_empty())
              to_add.insert(diff, overlap);
            it.filter(overlap);
            if (!it->second)
              to_remove.push_back(it->first);
          }
        }
      }
      else
      {
        FieldMaskSet<IndexSpaceExpression>::const_iterator first = 
          remote_exprs.begin();
        
        for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
              exprs.begin(); it != exprs.end(); it++)
        {
          const FieldMask overlap = it->second & first->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *diff = 
            runtime->forest->subtract_index_spaces(it->first, first->first);
          if (!diff->is_empty())
            to_add.insert(diff, overlap);
          it.filter(overlap);
          if (!it->second)
            to_remove.push_back(it->first);
        }
      }
      if (!to_remove.empty())
      {
        for (std::vector<IndexSpaceExpression*>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          exprs.erase(*it);
      }
      if (!to_add.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              to_add.begin(); it != to_add.end(); it++)
          exprs.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    bool PhysicalAnalysis::report_instances(FieldMaskSet<LogicalView> &insts)
    //--------------------------------------------------------------------------
    {
      // No need for the lock since we shouldn't be mutating anything at 
      // this point anyway
      if (recorded_instances != NULL)
        recorded_instances->swap(insts);
      return restricted;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_remote(EquivalenceSet *set, 
                                         const FieldMask &mask,
                                         const AddressSpaceID owner)
    //--------------------------------------------------------------------------
    {
      if (parallel_traversals)
      {
        AutoLock a_lock(*this);
        remote_sets[owner].insert(set, mask);
      }
      else
        // No lock needed if we're the only one
        remote_sets[owner].insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_instance(LogicalView *view, 
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      if (recorded_instances == NULL)
        recorded_instances = new FieldMaskSet<LogicalView>();
      recorded_instances->insert(view, mask);
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_remote_instances(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      PhysicalAnalysis *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      std::set<RtEvent> ready_events;
      target->process_remote_instances(derez, ready_events);
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event); 
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformTraversalArgs::DeferPerformTraversalArgs(
        PhysicalAnalysis *ana, EquivalenceSet *s, const FieldMask &m, 
        RtUserEvent done, bool def)
      : LgTaskArgs<DeferPerformTraversalArgs>(ana->op->get_unique_op_id()),
        analysis(ana), set(s), mask(new FieldMask(m)), 
        applied_event(Runtime::create_rt_user_event()),
        done_event(done.exists() ? done : Runtime::create_rt_user_event()), 
        already_deferred(def)
    //--------------------------------------------------------------------------
    {
      analysis->record_parallel_traversals();
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/void PhysicalAnalysis::handle_deferred_traversal(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformTraversalArgs *dargs = 
        (const DeferPerformTraversalArgs*)args;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      std::set<RtEvent> deferral_events, applied_events;
      dargs->analysis->traverse(dargs->set, *(dargs->mask), 
          deferral_events, applied_events, RtEvent::NO_RT_EVENT, 
          dargs->already_deferred);
      if (!deferral_events.empty())
        Runtime::trigger_event(dargs->done_event,
            Runtime::merge_events(deferral_events));
      else
        Runtime::trigger_event(dargs->done_event);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->applied_event,
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->applied_event);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
      delete dargs->mask;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformRemoteArgs::DeferPerformRemoteArgs(
                                                          PhysicalAnalysis *ana)
      : LgTaskArgs<DeferPerformRemoteArgs>(ana->op->get_unique_op_id()), 
        analysis(ana), applied_event(Runtime::create_rt_user_event()),
        done_event(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_remote(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformRemoteArgs *dargs = (const DeferPerformRemoteArgs*)args;
      std::set<RtEvent> applied_events;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      const RtEvent done = dargs->analysis->perform_remote(RtEvent::NO_RT_EVENT,
                                      applied_events, true/*already deferred*/);
      Runtime::trigger_event(dargs->done_event, done);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->applied_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->applied_event);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformUpdateArgs::DeferPerformUpdateArgs(
                                                          PhysicalAnalysis *ana)
      : LgTaskArgs<DeferPerformUpdateArgs>(ana->op->get_unique_op_id()), 
        analysis(ana), applied_event(Runtime::create_rt_user_event()),
        done_event(Runtime::create_rt_user_event())
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_update(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformUpdateArgs *dargs = (const DeferPerformUpdateArgs*)args;
      std::set<RtEvent> applied_events;
      // Get this before doing anything
      const bool on_heap = dargs->analysis->on_heap;
      const RtEvent done =dargs->analysis->perform_updates(RtEvent::NO_RT_EVENT,
                                      applied_events, true/*already deferred*/); 
      Runtime::trigger_event(dargs->done_event, done);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->applied_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->applied_event);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::DeferPerformOutputArgs::DeferPerformOutputArgs(
                           PhysicalAnalysis *ana, const PhysicalTraceInfo &info)
      : LgTaskArgs<DeferPerformOutputArgs>(ana->op->get_unique_op_id()), 
        analysis(ana), trace_info(&info),
        applied_event(Runtime::create_rt_user_event()),
        effects_event(Runtime::create_ap_user_event(trace_info))
    //--------------------------------------------------------------------------
    {
      if (analysis->on_heap)
        analysis->add_reference();
    }

    //--------------------------------------------------------------------------
    /*static*/ void PhysicalAnalysis::handle_deferred_output(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPerformOutputArgs *dargs = (const DeferPerformOutputArgs*)args;
      std::set<RtEvent> applied_events;
      const bool on_heap = dargs->analysis->on_heap;
      const ApEvent effects = dargs->analysis->perform_output(
          RtEvent::NO_RT_EVENT, applied_events, true/*already deferred*/);
      // Get this before doing anything
      Runtime::trigger_event(dargs->trace_info, dargs->effects_event, effects);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->applied_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->applied_event);
      if (on_heap && dargs->analysis->remove_reference())
        delete dargs->analysis;
    }

    /////////////////////////////////////////////////////////////
    // Valid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(Runtime *rt, Operation *o,unsigned idx, 
                                  IndexSpaceExpression *expr, ReductionOpID red)
      : PhysicalAnalysis(rt, o, idx, expr, false/*on heap*/), redop(red), 
        target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(Runtime *rt, AddressSpaceID src, 
            AddressSpaceID prev, Operation *o, unsigned idx,
            IndexSpaceExpression *expr, ValidInstAnalysis *t, ReductionOpID red)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/), 
        redop(red), target_analysis(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(const ValidInstAnalysis &rhs)
      : PhysicalAnalysis(rhs), redop(0), target_analysis(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::~ValidInstAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ValidInstAnalysis& ValidInstAnalysis::operator=(const ValidInstAnalysis &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ValidInstAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_valid_instances(*this, expr, expr_covers, mask, deferral_events, 
                                applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent ValidInstAnalysis::perform_remote(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (std::map<AddressSpaceID,
                    FieldMaskSet<EquivalenceSet> >::const_iterator
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(redop);
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_request_instances(target, rez);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent ValidInstAnalysis::perform_updates(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (recorded_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  recorded_instances->begin(); it != 
                  recorded_instances->end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          return response_event;
        }
        else
          target_analysis->process_local_instances(*recorded_instances,
                                                   restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ValidInstAnalysis::handle_remote_request_instances(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;

      std::vector<EquivalenceSet*> eq_sets(num_eq_sets);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op =
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      ReductionOpID redop;
      derez.deserialize(redop);
      ValidInstAnalysis *target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      ValidInstAnalysis *analysis = new ValidInstAnalysis(runtime, 
          original_source, previous, op, index, expr, target, redop);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx],
            deferral_events, applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready = 
        analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Invalid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(Runtime *rt, Operation *o, 
                                  unsigned idx, IndexSpaceExpression *expr, 
                                  const FieldMaskSet<LogicalView> &valid_insts)
      : PhysicalAnalysis(rt, o, idx, expr, true/*on heap*/), 
        valid_instances(valid_insts), target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(Runtime *rt, AddressSpaceID src, 
                             AddressSpaceID prev, Operation *o, unsigned idx,
                             IndexSpaceExpression *expr, InvalidInstAnalysis *t,
                             const FieldMaskSet<LogicalView> &valid_insts)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/), 
        valid_instances(valid_insts), target_analysis(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(const InvalidInstAnalysis &rhs)
      : PhysicalAnalysis(rhs), valid_instances(rhs.valid_instances),
        target_analysis(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::~InvalidInstAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis& InvalidInstAnalysis::operator=(
                                                  const InvalidInstAnalysis &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void InvalidInstAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_invalid_instances(*this, expr, expr_covers,mask,deferral_events,
                                  applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent InvalidInstAnalysis::perform_remote(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(valid_instances.size());
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                valid_instances.begin(); it != valid_instances.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_request_invalid(target, rez);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent InvalidInstAnalysis::perform_updates(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (recorded_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  recorded_instances->begin(); it != 
                  recorded_instances->end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          return response_event;
        }
        else
          target_analysis->process_local_instances(*recorded_instances, 
                                                   restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void InvalidInstAnalysis::handle_remote_request_invalid(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMaskSet<LogicalView> valid_instances;
      size_t num_valid_instances;
      derez.deserialize<size_t>(num_valid_instances);
      for (unsigned idx = 0; idx < num_valid_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask view_mask;
        derez.deserialize(view_mask);
        valid_instances.insert(view, view_mask);
      }
      InvalidInstAnalysis *target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      InvalidInstAnalysis *analysis = new InvalidInstAnalysis(runtime, 
          original_source, previous, op, index, expr, target, valid_instances);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready = 
        analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Antivalid Inst Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::AntivalidInstAnalysis(Runtime *rt, Operation *o, 
                                  unsigned idx, IndexSpaceExpression *expr, 
                                  const FieldMaskSet<LogicalView> &anti_insts)
      : PhysicalAnalysis(rt, o, idx, expr, true/*on heap*/), 
        antivalid_instances(anti_insts), target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::AntivalidInstAnalysis(Runtime *rt,AddressSpaceID src, 
                           AddressSpaceID prev, Operation *o, unsigned idx,
                           IndexSpaceExpression *expr, AntivalidInstAnalysis *a,
                           const FieldMaskSet<LogicalView> &anti_insts)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/), 
        antivalid_instances(anti_insts), target_analysis(a)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::AntivalidInstAnalysis(
                                               const AntivalidInstAnalysis &rhs)
      : PhysicalAnalysis(rhs), antivalid_instances(rhs.antivalid_instances),
        target_analysis(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis::~AntivalidInstAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AntivalidInstAnalysis& AntivalidInstAnalysis::operator=(
                                                const AntivalidInstAnalysis &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void AntivalidInstAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_antivalid_instances(*this,expr,expr_covers,mask,deferral_events,
                                    applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent AntivalidInstAnalysis::perform_remote(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Easy out if we don't have remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> ready_events;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent ready = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(antivalid_instances.size());
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                antivalid_instances.begin(); it != 
                antivalid_instances.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_request_antivalid(target, rez);
        ready_events.insert(ready);
        applied_events.insert(applied);
      }
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    RtEvent AntivalidInstAnalysis::perform_updates(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (recorded_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  recorded_instances->begin(); it != 
                  recorded_instances->end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          return response_event;
        }
        else
          target_analysis->process_local_instances(*recorded_instances, 
                                                   restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void AntivalidInstAnalysis::handle_remote_request_antivalid(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMaskSet<LogicalView> antivalid_instances;
      size_t num_antivalid_instances;
      derez.deserialize<size_t>(num_antivalid_instances);
      for (unsigned idx = 0; idx < num_antivalid_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask view_mask;
        derez.deserialize(view_mask);
        antivalid_instances.insert(view, view_mask);
      }
      AntivalidInstAnalysis *target;
      derez.deserialize(target);
      RtUserEvent ready;
      derez.deserialize(ready);
      RtUserEvent applied;
      derez.deserialize(applied);

      AntivalidInstAnalysis *analysis = new AntivalidInstAnalysis(runtime, 
       original_source, previous, op, index, expr, target, antivalid_instances);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready = 
        analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(ready, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(ready);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Update Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(Runtime *rt, Operation *o, unsigned idx,
                     const RegionRequirement &req, RegionNode *rn, 
                     const InstanceSet &target_insts,
                     std::vector<InstanceView*> &target_vws,
                     std::vector<InstanceView*> &source_vws,
                     const PhysicalTraceInfo &t_info,
                     const ApEvent pre, const ApEvent term,
                     const bool check, const bool record, const bool skip)
      : PhysicalAnalysis(rt, o, idx, rn->row_source, true/*on heap*/), 
        usage(req), node(rn), target_instances(target_insts), 
        target_views(target_vws), source_views(source_vws), trace_info(t_info),
        precondition(pre), term_event(term),
        check_initialized(check && !IS_DISCARD(usage) && !IS_SIMULT(usage)), 
        record_valid(record), skip_output(skip), output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(Runtime *rt, AddressSpaceID src, 
                     AddressSpaceID prev, Operation *o, unsigned idx, 
                     const RegionUsage &use, RegionNode *rn, 
                     InstanceSet &target_insts,
                     std::vector<InstanceView*> &target_vws,
                     std::vector<InstanceView*> &source_vws,
                     const PhysicalTraceInfo &info,
                     const RtEvent user_reg, const ApEvent pre, 
                     const ApEvent term, const bool check, 
                     const bool record, const bool skip)
      : PhysicalAnalysis(rt, src, prev, o, idx, rn->row_source,true/*on heap*/),
        usage(use), node(rn), target_instances(target_insts), 
        target_views(target_vws), source_views(source_vws), trace_info(info), 
        precondition(pre), term_event(term),
        check_initialized(check), record_valid(record), skip_output(skip), 
        output_aggregator(NULL), remote_user_registered(user_reg)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(const UpdateAnalysis &rhs)
      : PhysicalAnalysis(rhs), usage(rhs.usage), node(rhs.node), 
        target_instances(rhs.target_instances), target_views(rhs.target_views),
        source_views(rhs.source_views), trace_info(rhs.trace_info), 
        precondition(rhs.precondition), term_event(rhs.term_event), 
        check_initialized(rhs.check_initialized),record_valid(rhs.record_valid),
        skip_output(rhs.skip_output)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis::~UpdateAnalysis(void)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis& UpdateAnalysis::operator=(const UpdateAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void UpdateAnalysis::record_uninitialized(const FieldMask &uninit,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (!uninitialized)
      {
#ifdef DEBUG_LEGION
        assert(!uninitialized_reported.exists());
#endif
        uninitialized_reported = Runtime::create_rt_user_event();
        applied_events.insert(uninitialized_reported);
      }
      uninitialized |= uninit;
    }

    //--------------------------------------------------------------------------
    void UpdateAnalysis::perform_traversal(EquivalenceSet *set,
                                           IndexSpaceExpression *expr,
                                           const bool expr_covers,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->update_set(*this, expr, expr_covers, mask, deferral_events, 
                      applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent UpdateAnalysis::perform_remote(RtEvent perform_precondition,
                                           std::set<RtEvent> &applied_events,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Easy out if we don't have any remote sets
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
#ifdef DEBUG_LEGION
      assert(!target_instances.empty());
      assert(target_instances.size() == target_views.size());
#endif
      if (!remote_user_registered.exists())
      {
#ifdef DEBUG_LEGION
        assert(original_source == runtime->address_space);
        assert(!user_registered.exists());
#endif
        user_registered = Runtime::create_rt_user_event(); 
        remote_user_registered = user_registered;
      }
      std::set<RtEvent> remote_events;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent updated = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        const ApUserEvent effects = Runtime::create_ap_user_event(&trace_info);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(node->handle);
          rez.serialize(usage);
          rez.serialize<size_t>(target_instances.size());
          for (unsigned idx = 0; idx < target_instances.size(); idx++)
          {
            const InstanceRef &ref = target_instances[idx];
            rez.serialize(ref.get_manager()->did);
            rez.serialize(target_views[idx]->did);
            rez.serialize(ref.get_valid_fields());
          }
          rez.serialize<size_t>(source_views.size());
          for (unsigned idx = 0; idx < source_views.size(); idx++)
            rez.serialize(source_views[idx]->did);
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(precondition);
          rez.serialize(term_event);
          rez.serialize(updated);
          rez.serialize(remote_user_registered);
          rez.serialize(applied);
          rez.serialize(effects);
          rez.serialize<bool>(check_initialized);
          rez.serialize<bool>(record_valid);
          rez.serialize<bool>(skip_output);
        }
        runtime->send_equivalence_set_remote_updates(target, rez);
        remote_events.insert(updated);
        applied_events.insert(applied);
        effects_events.insert(effects);
      }
      return Runtime::merge_events(remote_events);
    }

    //--------------------------------------------------------------------------
    RtEvent UpdateAnalysis::perform_updates(RtEvent perform_precondition,
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Report any uninitialized data now that we know the traversal is done
      if (!!uninitialized)
      {
#ifdef DEBUG_LEGION
        assert(check_initialized);
        assert(uninitialized_reported.exists());
#endif
        node->report_uninitialized_usage(op, index, usage, uninitialized,
                                         uninitialized_reported);
      }
      if (!input_aggregators.empty())
      {
#ifndef NON_AGGRESSIVE_AGGREGATORS
        const bool is_local = (original_source == runtime->address_space);
#endif
        for (std::map<RtEvent,CopyFillAggregator*>::const_iterator it = 
              input_aggregators.begin(); it != input_aggregators.end(); it++)
        {
          it->second->issue_updates(trace_info, precondition);
#ifdef NON_AGGRESSIVE_AGGREGATORS
          if (!it->second->effects_applied.has_triggered())
            guard_events.insert(it->second->effects_applied);
#else
          if (!it->second->effects_applied.has_triggered())
          {
            if (is_local)
            {
              if (!it->second->guard_postcondition.has_triggered())
                guard_events.insert(it->second->guard_postcondition);
              applied_events.insert(it->second->effects_applied);
            }
            else
              guard_events.insert(it->second->effects_applied);
          }
#endif
          if (it->second->release_guards(op->runtime, applied_events))
            delete it->second;
        }
      }
      if (!guard_events.empty())
        return Runtime::merge_events(guard_events);
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent UpdateAnalysis::perform_output(RtEvent perform_precondition,
                                           std::set<RtEvent> &applied_events,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformOutputArgs args(this, trace_info);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        // If we're skipping the output we still need to launch this 
        // meta-task to prevent the analysis from being deleted until
        // everything else is done, we just don't record any output
        if (!skip_output)
        {
          applied_events.insert(args.applied_event);
          return args.effects_event;
        }
        else
          return ApEvent::NO_AP_EVENT;
      }
      ApEvent result;
      if (output_aggregator != NULL)
      {
#ifdef DEBUG_LEGION
        assert(!skip_output);
#endif
        output_aggregator->issue_updates(trace_info, term_event,
                                         true/*restricted output*/);
        // We need to wait for the aggregator updates to be applied
        // here before we can summarize the output
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!output_aggregator->effects_applied.has_triggered())
          output_aggregator->effects_applied.wait();
#else
        if (!output_aggregator->guard_postcondition.has_triggered())
          output_aggregator->guard_postcondition.wait();
        if (!output_aggregator->effects_applied.has_triggered())
          applied_events.insert(output_aggregator->effects_applied);
#endif
        result = output_aggregator->summarize(trace_info);
        if (output_aggregator->release_guards(op->runtime, applied_events))
          delete output_aggregator;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void UpdateAnalysis::handle_remote_updates(Deserializer &derez, 
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      FieldMask user_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        user_mask |= eq_masks[idx];
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionUsage usage;
      derez.deserialize(usage);
      size_t num_targets;
      derez.deserialize(num_targets);
      InstanceSet targets(num_targets);
      std::vector<InstanceView*> target_views(num_targets, NULL);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        InstanceManager *manager = 
          runtime->find_or_request_instance_manager(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(did);
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        target_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        targets[idx] = InstanceRef(manager, valid_fields);
      }
      size_t num_sources;
      derez.deserialize(num_sources);
      std::vector<InstanceView*> source_views(num_sources);
      for (unsigned idx = 0; idx < num_sources; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        source_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      ApEvent precondition;
      derez.deserialize(precondition);
      ApEvent term_event;
      derez.deserialize(term_event);
      RtUserEvent updated;
      derez.deserialize(updated);
      RtEvent remote_user_registered;
      derez.deserialize(remote_user_registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent effects_done;
      derez.deserialize(effects_done);
      bool check_initialized;
      derez.deserialize(check_initialized);
      bool record_valid;
      derez.deserialize(record_valid);
      bool skip_output;
      derez.deserialize(skip_output);

      RegionNode *node = runtime->forest->get_node(handle);
      // This takes ownership of the remote operation
      UpdateAnalysis *analysis = new UpdateAnalysis(runtime, original_source,
          previous, op, index, usage, node, targets, target_views, source_views,
          trace_info, remote_user_registered, precondition, term_event, 
          check_initialized, record_valid, skip_output);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events; 
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      std::set<RtEvent> update_events;
      // If we have remote messages to send do that now
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          update_events.insert(remote_ready);
      }
      // Then perform the updates
      // Note that we need to capture all the effects of these updates
      // before we can consider them applied, so we can't use the 
      // applied_events data structure here
      const RtEvent updates_ready = 
        analysis->perform_updates(traversal_done, update_events);
      if (updates_ready.exists())
        update_events.insert(updates_ready);
      // We can trigger our updated event done when all the guards are done 
      if (!update_events.empty())
        Runtime::trigger_event(updated, Runtime::merge_events(update_events));
      else
        Runtime::trigger_event(updated);
      // If we have outputs we need for the user to be registered
      // before we can apply the output copies
      const ApEvent result = 
        analysis->perform_output(remote_user_registered, applied_events);
      if (effects_done.exists())
        Runtime::trigger_event(&trace_info, effects_done, result);
      // Do the rest of the triggers
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Acquire Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(Runtime *rt, Operation *o, 
                                     unsigned idx, IndexSpaceExpression *expr)
      : PhysicalAnalysis(rt, o, idx, expr, false/*on heap*/), 
        target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(Runtime *rt, AddressSpaceID src, 
                      AddressSpaceID prev, Operation *o, unsigned idx, 
                      IndexSpaceExpression *expr, AcquireAnalysis *t)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/), 
        target_analysis(t)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(const AcquireAnalysis &rhs)
      : PhysicalAnalysis(rhs), target_analysis(rhs.target_analysis)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    AcquireAnalysis::~AcquireAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AcquireAnalysis& AcquireAnalysis::operator=(const AcquireAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void AcquireAnalysis::perform_traversal(EquivalenceSet *set,
                                            IndexSpaceExpression *expr,
                                            const bool expr_covers,
                                            const FieldMask &mask,
                                            std::set<RtEvent> &deferral_events,
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->acquire_restrictions(*this, expr, expr_covers, mask, deferral_events,
                                applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent AcquireAnalysis::perform_remote(RtEvent perform_precondition, 
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      } 
      // Easy out if there is nothing to do
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> remote_events;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent returned = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(target_analysis);
        }
        runtime->send_equivalence_set_remote_acquires(target, rez);
        applied_events.insert(applied);
        remote_events.insert(returned);
      }
      return Runtime::merge_events(remote_events);
    }

    //--------------------------------------------------------------------------
    RtEvent AcquireAnalysis::perform_updates(RtEvent perform_precondition,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //-------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (recorded_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  recorded_instances->begin(); it != 
                  recorded_instances->end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          return response_event;
        }
        else
          target_analysis->process_local_instances(*recorded_instances, 
                                                   restricted);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void AcquireAnalysis::handle_remote_acquires(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr =
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent applied;
      derez.deserialize(applied);
      AcquireAnalysis *target;
      derez.deserialize(target);

      // This takes ownership of the operation
      AcquireAnalysis *analysis = new AcquireAnalysis(runtime, original_source,
                                            previous, op, index, expr, target);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Defer sending the updates until we're ready
      const RtEvent local_ready = 
        analysis->perform_updates(traversal_done, applied_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(returned, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(returned);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Release Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(Runtime *rt, Operation *o, unsigned idx, 
                                     ApEvent pre, IndexSpaceExpression *expr,
                                     const InstanceSet &target_insts,
                                     std::vector<InstanceView*> &target_vws,
                                     std::vector<InstanceView*> &source_vws, 
                                     const PhysicalTraceInfo &t_info)
      : PhysicalAnalysis(rt, o, idx, expr, false/*on heap*/), 
        precondition(pre), target_analysis(this),
        target_instances(target_insts), target_views(target_vws),
        source_views(source_vws), trace_info(t_info), release_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(Runtime *rt, AddressSpaceID src, 
          AddressSpaceID prev, Operation *o, unsigned idx, 
          IndexSpaceExpression *expr, ApEvent pre, ReleaseAnalysis *t, 
          InstanceSet &target_insts, std::vector<InstanceView*> &target_vws,
          std::vector<InstanceView*> &source_vws, const PhysicalTraceInfo &info)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/), 
        precondition(pre), target_analysis(t), target_instances(target_insts),
        target_views(target_vws), source_views(source_vws), trace_info(info),
        release_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(const ReleaseAnalysis &rhs)
      : PhysicalAnalysis(rhs), target_analysis(rhs.target_analysis), 
        target_instances(rhs.target_instances), target_views(rhs.target_views),
        source_views(rhs.source_views), trace_info(rhs.trace_info)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReleaseAnalysis::~ReleaseAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseAnalysis& ReleaseAnalysis::operator=(const ReleaseAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReleaseAnalysis::perform_traversal(EquivalenceSet *set,
                                            IndexSpaceExpression *expr,
                                            const bool expr_covers,
                                            const FieldMask &mask,
                                            std::set<RtEvent> &deferral_events,
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->release_restrictions(*this, expr, expr_covers, mask, deferral_events,
                                applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent ReleaseAnalysis::perform_remote(RtEvent perform_precondition, 
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Easy out if there is nothing to do
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      std::set<RtEvent> remote_events;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent returned = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(target_instances.size());
          for (unsigned idx = 0; idx < target_instances.size(); idx++)
          {
            const InstanceRef &ref = target_instances[idx];
            rez.serialize(ref.get_manager()->did);
            rez.serialize(target_views[idx]->did);
            rez.serialize(ref.get_valid_fields());
          }
          rez.serialize<size_t>(source_views.size());
          for (std::vector<InstanceView*>::const_iterator it =
                source_views.begin(); it != source_views.end(); it++)
            rez.serialize((*it)->did);
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(precondition);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(target_analysis);
          trace_info.pack_trace_info(rez, applied_events);
        }
        runtime->send_equivalence_set_remote_releases(target, rez);
        applied_events.insert(applied);
        remote_events.insert(returned);
      }
      return Runtime::merge_events(remote_events);
    }

    //--------------------------------------------------------------------------
    RtEvent ReleaseAnalysis::perform_updates(RtEvent perform_precondition, 
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Defer this if necessary
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // See if we have any instance names to send back
      if ((target_analysis != this) && (recorded_instances != NULL))
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(recorded_instances->size());
            for (FieldMaskSet<LogicalView>::const_iterator it = 
                  recorded_instances->begin(); it != 
                  recorded_instances->end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
            rez.serialize<bool>(restricted);
          }
          runtime->send_equivalence_set_remote_instances(original_source, rez);
          applied_events.insert(response_event);
        }
        else
          target_analysis->process_local_instances(*recorded_instances, 
                                                   restricted);
      }
      if (release_aggregator != NULL)
      {
        std::set<RtEvent> guard_events;
        release_aggregator->issue_updates(trace_info, precondition);
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (release_aggregator->effects_applied.has_triggered())
          guard_events.insert(release_aggregator->effects_applied);
#else
        if (release_aggregator->effects_applied.has_triggered())
        {
          if (original_source == runtime->address_space)
          {
            if (!release_aggregator->guard_postcondition.has_triggered())
              guard_events.insert(release_aggregator->guard_postcondition);
            applied_events.insert(release_aggregator->effects_applied);
          }
          else
            guard_events.insert(release_aggregator->effects_applied);
        }
#endif
        if (release_aggregator->release_guards(op->runtime, applied_events))
          delete release_aggregator;
        if (!guard_events.empty())
          return Runtime::merge_events(guard_events);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ReleaseAnalysis::handle_remote_releases(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      size_t num_targets;
      derez.deserialize(num_targets);
      InstanceSet target_instances(num_targets);
      std::vector<InstanceView*> target_views(num_targets, NULL);
      for (unsigned idx = 0; idx < num_targets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        InstanceManager *manager = 
          runtime->find_or_request_instance_manager(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(did);
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        target_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask valid_fields;
        derez.deserialize(valid_fields);
        target_instances[idx] = InstanceRef(manager, valid_fields);
      }
      size_t num_sources;
      derez.deserialize(num_sources);
      std::vector<InstanceView*> source_views(num_sources);
      for (unsigned idx = 0; idx < num_sources; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        source_views[idx] = static_cast<InstanceView*>(
            runtime->find_or_request_logical_view(did, ready));
        if (ready.exists())
          ready_events.insert(ready);
      }
      IndexSpaceExpression *expr =
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      ApEvent precondition;
      derez.deserialize(precondition);
      RtUserEvent returned;
      derez.deserialize(returned);
      RtUserEvent applied;
      derez.deserialize(applied);
      ReleaseAnalysis *target;
      derez.deserialize(target);
      const PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);

      ReleaseAnalysis *analysis = new ReleaseAnalysis(runtime, original_source,
          previous, op, index, expr, precondition, target, target_instances,
          target_views, source_views, trace_info);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      RtEvent ready_event;
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
      {
        const RtEvent remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
        if (remote_ready.exists())
          ready_events.insert(remote_ready);
      }
      // Note that we use the ready events here for applied so that
      // we can know that all our updates are done before we tell
      // the original source node that we've returned
      const RtEvent local_ready = 
        analysis->perform_updates(traversal_done, 
            (original_source == runtime->address_space) ?
              applied_events : ready_events);
      if (local_ready.exists())
        ready_events.insert(local_ready);
      if (!ready_events.empty())
        Runtime::trigger_event(returned, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(returned);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Copy Across Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::CopyAcrossAnalysis(Runtime *rt, Operation *o, 
        unsigned src_idx, unsigned dst_idx, const RegionRequirement &src_req,
        const RegionRequirement &dst_req, const InstanceSet &target_insts,
        const std::vector<InstanceView*> &target_vws, 
        const std::vector<InstanceView*> &source_vws,
        const ApEvent pre, const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned> &src_idxes,
        const std::vector<unsigned> &dst_idxes, 
        const PhysicalTraceInfo &t_info, const bool perf)
      : PhysicalAnalysis(rt, o, dst_idx, 
          rt->forest->get_node(dst_req.region)->row_source, true/*on heap*/), 
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)), 
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), src_usage(src_req), 
        dst_usage(dst_req), src_region(src_req.region), 
        dst_region(dst_req.region), target_instances(target_insts),
        target_views(target_vws), source_views(source_vws), precondition(pre),
        pred_guard(pred), redop(red), src_indexes(src_idxes), 
        dst_indexes(dst_idxes), across_helpers(perf ? 
              std::vector<CopyAcrossHelper*>() :
              create_across_helpers(src_mask, dst_mask, target_instances,
                                    src_indexes, dst_indexes)),
        trace_info(t_info), perfect(perf), across_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::CopyAcrossAnalysis(Runtime *rt, AddressSpaceID src, 
        AddressSpaceID prev, Operation *o, unsigned src_idx, unsigned dst_idx,
        const RegionUsage &src_use, const RegionUsage &dst_use, 
        const LogicalRegion src_reg, const LogicalRegion dst_reg, 
        const InstanceSet &target_insts,
        const std::vector<InstanceView*> &target_vws, 
        const std::vector<InstanceView*> &source_vws,
        const ApEvent pre, const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned> &src_idxes,
        const std::vector<unsigned> &dst_idxes, 
        const PhysicalTraceInfo &t_info, const bool perf)
      : PhysicalAnalysis(rt, src, prev, o, dst_idx,
          rt->forest->get_node(dst_reg)->row_source, true/*on heap*/),
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)), 
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), 
        src_usage(src_use), dst_usage(dst_use), src_region(src_reg), 
        dst_region(dst_reg), target_instances(target_insts), 
        target_views(target_vws), source_views(source_vws), precondition(pre),
        pred_guard(pred), redop(red), src_indexes(src_idxes), 
        dst_indexes(dst_idxes), across_helpers(perf ? 
              std::vector<CopyAcrossHelper*>() :
              create_across_helpers(src_mask, dst_mask, target_instances,
                                    src_indexes, dst_indexes)),
        trace_info(t_info), perfect(perf), across_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::CopyAcrossAnalysis(const CopyAcrossAnalysis &rhs)
      : PhysicalAnalysis(rhs), dst_mask(rhs.dst_mask), src_index(rhs.src_index),
        dst_index(rhs.dst_index), src_usage(rhs.src_usage), 
        dst_usage(rhs.dst_usage), src_region(rhs.src_region), 
        dst_region(rhs.dst_region), target_instances(rhs.target_instances),
        target_views(rhs.target_views), source_views(rhs.source_views),
        precondition(rhs.precondition), pred_guard(rhs.pred_guard), 
        redop(rhs.redop), src_indexes(rhs.src_indexes), 
        dst_indexes(rhs.dst_indexes), across_helpers(rhs.across_helpers), 
        trace_info(rhs.trace_info), perfect(rhs.perfect)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis::~CopyAcrossAnalysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!aggregator_guard.exists() || aggregator_guard.has_triggered());
#endif 
      for (std::vector<CopyAcrossHelper*>::const_iterator it = 
            across_helpers.begin(); it != across_helpers.end(); it++)
        delete (*it);
    }

    //--------------------------------------------------------------------------
    CopyAcrossAnalysis& CopyAcrossAnalysis::operator=(
                                                  const CopyAcrossAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CopyAcrossAnalysis::record_uninitialized(const FieldMask &uninit,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (!uninitialized)
      {
#ifdef DEBUG_LEGION
        assert(!uninitialized_reported.exists());
#endif
        uninitialized_reported = Runtime::create_rt_user_event();
        applied_events.insert(uninitialized_reported);
      }
      uninitialized |= uninit;
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator* CopyAcrossAnalysis::get_across_aggregator(void)
    //--------------------------------------------------------------------------
    {
      if (across_aggregator == NULL)
      {
#ifdef DEBUG_LEGION
        assert(!aggregator_guard.exists());
#endif
        aggregator_guard = Runtime::create_rt_user_event();
        across_aggregator = new CopyFillAggregator(runtime->forest, op, 
            src_index, dst_index, NULL/*no previous guard*/, true/*track*/,
            pred_guard, aggregator_guard);
      }
      return across_aggregator;
    }
    
    //--------------------------------------------------------------------------
    void CopyAcrossAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->issue_across_copies(*this, mask, expr, expr_covers, deferral_events,
                               applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent CopyAcrossAnalysis::perform_remote(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
#ifdef DEBUG_LEGION
      assert(target_instances.size() == target_views.size());
      assert(src_indexes.size() == dst_indexes.size());
#endif
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const ApUserEvent copy = Runtime::create_ap_user_event(&trace_info);
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(src_index);
          rez.serialize(dst_index);
          rez.serialize(src_usage);
          rez.serialize(dst_usage);
          rez.serialize<size_t>(target_instances.size());
          for (unsigned idx = 0; idx < target_instances.size(); idx++)
          {
            target_instances[idx].pack_reference(rez);
            rez.serialize(target_views[idx]->did); 
          }
          rez.serialize<size_t>(source_views.size());
          for (unsigned idx = 0; idx < source_views.size(); idx++)
            rez.serialize(source_views[idx]->did);
          rez.serialize(src_region);
          rez.serialize(dst_region);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(redop);
          rez.serialize<bool>(perfect);
          if (!perfect)
          {
            rez.serialize<size_t>(src_indexes.size());
            for (unsigned idx = 0; idx < src_indexes.size(); idx++)
            {
              rez.serialize(src_indexes[idx]);
              rez.serialize(dst_indexes[idx]);
            }
          }
          rez.serialize(applied);
          rez.serialize(copy);
          trace_info.pack_trace_info(rez, applied_events);
        }
        runtime->send_equivalence_set_remote_copies_across(target, rez);
        applied_events.insert(applied);
        copy_events.insert(copy);
      }
      // Filter all the remote expressions from the local ones here
      filter_remote_expressions(local_exprs);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent CopyAcrossAnalysis::perform_updates(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // Report any uninitialized data now that we know the traversal is done
      if (!!uninitialized)
      {
#ifdef DEBUG_LEGION
        assert(uninitialized_reported.exists());
#endif
        RegionNode *src_node = runtime->forest->get_node(src_region);
        src_node->report_uninitialized_usage(op, src_index, src_usage, 
                                uninitialized, uninitialized_reported);
      }
      if (across_aggregator != NULL)
      {
#ifdef DEBUG_LEGION
        assert(aggregator_guard.exists());
#endif
        // Trigger the guard event for the aggregator once all the 
        // actual guard events are done. Note that this is safe for
        // copy across aggregators because unlike other aggregators
        // they are moving data from one field to another so it is
        // safe to create entanglements between fields since they are
        // all going to be subsumed by the same completion event for
        // the copy-across operation anyway
        if (!guard_events.empty())
          Runtime::trigger_event(aggregator_guard, 
              Runtime::merge_events(guard_events));
        else
          Runtime::trigger_event(aggregator_guard);
        // Record the event field preconditions for each view
        std::map<InstanceView*,std::vector<ApEvent> > dst_events;
        for (unsigned idx = 0; idx < target_instances.size(); idx++)
        {
          const InstanceRef &ref = target_instances[idx];
          InstanceView *view = target_views[idx];
          // Always instantiate the entry in the map
          std::vector<ApEvent> &events = dst_events[view];
          const ApEvent event = ref.get_ready_event();
          if (!event.exists())
            continue;
          events.push_back(event);
        }
        // This is a copy-across aggregator so the destination events
        // are being handled by the copy operation that mapped the
        // target instance for us
        across_aggregator->issue_updates(trace_info, precondition,
            false/*restricted*/, false/*manage dst events*/, &dst_events);
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!across_aggregator->effects_applied.has_triggered())
          return across_aggregator->effects_applied;
#else
        if (!across_aggregator->effects_applied.has_triggered())
        {
          if (original_source == runtime->address_space)
          {
            applied_events.insert(across_aggregator->effects_applied);
            if (!across_aggregator->guard_postcondition.has_triggered())
              return across_aggregator->guard_postcondition;
          }
          else
            return across_aggregator->effects_applied;
        }
#endif
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent CopyAcrossAnalysis::perform_output(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformOutputArgs args(this, trace_info);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.effects_event;
      }
      if (across_aggregator != NULL)
      {
        const ApEvent result = across_aggregator->summarize(trace_info);
        if (result.exists())
          copy_events.insert(result);
        if (across_aggregator->release_guards(op->runtime, applied_events))
          delete across_aggregator;
      }
      if (!copy_events.empty())
        return Runtime::merge_events(&trace_info, copy_events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CopyAcrossAnalysis::handle_remote_copies_across(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      FieldMask src_mask;
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
        src_mask |= eq_masks[idx];
      }
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned src_index, dst_index;
      derez.deserialize(src_index);
      derez.deserialize(dst_index);
      RegionUsage src_usage, dst_usage;
      derez.deserialize(src_usage);
      derez.deserialize(dst_usage);
      size_t num_dsts;
      derez.deserialize(num_dsts);
      InstanceSet dst_instances(num_dsts);
      std::vector<InstanceView*> dst_views(num_dsts, NULL);
      for (unsigned idx = 0; idx < num_dsts; idx++)
      {
        RtEvent ready;
        dst_instances[idx].unpack_reference(runtime, derez, ready);
        if (ready.exists())
          ready_events.insert(ready);
        DistributedID did;
        derez.deserialize(did);
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        dst_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      size_t num_srcs;
      derez.deserialize(num_srcs);
      std::vector<InstanceView*> src_views(num_srcs, NULL);
      for (unsigned idx = 0; idx < num_srcs; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        src_views[idx] = static_cast<InstanceView*>(view);
        if (ready.exists())
          ready_events.insert(ready);
      }
      LogicalRegion src_handle, dst_handle;
      derez.deserialize(src_handle);
      derez.deserialize(dst_handle);
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      ReductionOpID redop;
      derez.deserialize(redop);
      bool perfect;
      derez.deserialize(perfect);
      std::vector<unsigned> src_indexes, dst_indexes;
      if (!perfect)
      {
        size_t num_indexes;
        derez.deserialize(num_indexes);
        src_indexes.resize(num_indexes);
        dst_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
        {
          derez.deserialize(src_indexes[idx]);
          derez.deserialize(dst_indexes[idx]);
        }
      }
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent copy;
      derez.deserialize(copy);
      const PhysicalTraceInfo trace_info =
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);

      std::vector<CopyAcrossHelper*> across_helpers;
      std::set<RtEvent> deferral_events, applied_events;
      RegionNode *dst_node = runtime->forest->get_node(dst_handle);
      IndexSpaceExpression *dst_expr = dst_node->get_index_space_expression();
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      // If we're not perfect we need to wait on the ready event here
      // because we need the dst_instances to be valid to construct
      // the copy-across helpers
      if (!perfect && ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      // This takes ownership of the op and the across helpers
      CopyAcrossAnalysis *analysis = new CopyAcrossAnalysis(runtime, 
          original_source, previous, op, src_index, dst_index,
          src_usage, dst_usage, src_handle, dst_handle, 
          dst_instances, dst_views, src_views, precondition, pred_guard, 
          redop, src_indexes, dst_indexes, trace_info, perfect);
      analysis->add_reference();
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events,
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      // Start with the source mask here in case we need to filter which
      // is all done on the source fields
      analysis->local_exprs.insert(dst_expr, src_mask);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = analysis->perform_remote(traversal_done, applied_events);
      RtEvent updates_ready;
      // Chain these so we get the local_exprs set correct
      if (remote_ready.exists() || analysis->has_across_updates())
        updates_ready = 
          analysis->perform_updates(remote_ready, applied_events); 
      const ApEvent result = 
        analysis->perform_output(updates_ready, applied_events);
      Runtime::trigger_event(&trace_info, copy, result);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      // Clean up our analysis
      if (analysis->remove_reference())
        delete analysis;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::vector<CopyAcrossHelper*> 
                          CopyAcrossAnalysis::create_across_helpers(
                                       const FieldMask &src_mask,
                                       const FieldMask &dst_mask,
                                       const InstanceSet &dst_instances,
                                       const std::vector<unsigned> &src_indexes,
                                       const std::vector<unsigned> &dst_indexes)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!dst_instances.empty());
#endif
      std::vector<CopyAcrossHelper*> result(dst_instances.size());
      for (unsigned idx = 0; idx < dst_instances.size(); idx++)
      {
        result[idx] = new CopyAcrossHelper(src_mask, src_indexes, dst_indexes);
        IndividualManager *manager = 
          dst_instances[idx].get_manager()->as_individual_manager();
        manager->initialize_across_helper(result[idx],
                              dst_mask, src_indexes, dst_indexes);
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Overwrite Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, Operation *o, 
                        unsigned idx, const RegionUsage &use,
                        IndexSpaceExpression *expr, LogicalView *view, 
                        const FieldMask &mask, const PhysicalTraceInfo &t_info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, o, idx, expr, true/*on heap*/), usage(use), 
        trace_info(t_info), precondition(pre), guard_event(guard), 
        pred_guard(pred), track_effects(track), add_restriction(restriction),
        output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
      if (view != NULL)
      {
        if (view->is_reduction_view())
          reduction_views.insert(view->as_reduction_view(), mask);
        else
          views.insert(view, mask);
      }
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, Operation *o, 
                        unsigned idx, const RegionUsage &use,
                        IndexSpaceExpression *e,
                        const FieldMaskSet<LogicalView> &vws,
                        const PhysicalTraceInfo &t_info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, o, idx, e, true/*on heap*/), usage(use), 
        trace_info(t_info), precondition(pre), guard_event(guard), 
        pred_guard(pred), track_effects(track), add_restriction(restriction), 
        output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            vws.begin(); it != vws.end(); it++)
      {
        if (it->first->is_reduction_view())
          reduction_views.insert(it->first->as_reduction_view(), it->second);
        else
          views.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, AddressSpaceID src, 
                        AddressSpaceID prev, Operation *o, unsigned idx, 
                        IndexSpaceExpression *expr, const RegionUsage &use,
                        FieldMaskSet<LogicalView> &vws,
                        FieldMaskSet<ReductionView> &reductions,
                        const PhysicalTraceInfo &info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/),
        usage(use), views(vws,true/*copy*/), 
        reduction_views(reductions,true/*copy*/), trace_info(info),
        precondition(pre), guard_event(guard), pred_guard(pred), 
        track_effects(track), add_restriction(restriction), 
        output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(const OverwriteAnalysis &rhs)
      : PhysicalAnalysis(rhs), usage(rhs.usage),
        trace_info(rhs.trace_info), precondition(rhs.precondition), 
        guard_event(rhs.guard_event), pred_guard(rhs.pred_guard), 
        track_effects(rhs.track_effects), add_restriction(rhs.add_restriction)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::~OverwriteAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis& OverwriteAnalysis::operator=(const OverwriteAnalysis &rs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void OverwriteAnalysis::perform_traversal(EquivalenceSet *set,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->overwrite_set(*this, expr, expr_covers, mask, deferral_events, 
                         applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::perform_remote(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // If there are no sets we're done
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpace target = rit->first;
        const RtUserEvent applied = Runtime::create_rt_user_event();
        const ApUserEvent effects = track_effects ? 
          Runtime::create_ap_user_event(&trace_info) : 
          ApUserEvent::NO_AP_USER_EVENT;
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(usage);
          rez.serialize<size_t>(views.size());
          if (!views.empty())
          {
            for (FieldMaskSet<LogicalView>::const_iterator it =
                  views.begin(); it != views.end(); it++)
            {
              rez.serialize(it->first->did);  
              rez.serialize(it->second);
            }
          }
          rez.serialize<size_t>(reduction_views.size());
          if (!reduction_views.empty())
          {
            for (FieldMaskSet<ReductionView>::const_iterator it =
                  reduction_views.begin(); it != reduction_views.end(); it++)
            {
              rez.serialize(it->first->did);  
              rez.serialize(it->second);
            }
          }
          trace_info.pack_trace_info(rez, applied_events);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(guard_event);
          rez.serialize<bool>(add_restriction);
          rez.serialize(applied);
          rez.serialize(effects);
        }
        runtime->send_equivalence_set_remote_overwrites(target, rez);
        applied_events.insert(applied);
        if (track_effects)
          effects_events.insert(effects);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::perform_updates(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformUpdateArgs args(this);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (output_aggregator != NULL)
      {
        output_aggregator->issue_updates(trace_info, precondition, 
                                         true/*restricted output*/);
        // Need to wait before we can get the summary
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!output_aggregator->effects_applied.has_triggered())
          return output_aggregator->effects_applied;
#else
        if (output_aggregator->effects_applied.has_triggered())
        {
          if (original_source == runtime->address_space)
          {
            applied_events.insert(output_aggregator->effects_applied);
            if (!output_aggregator->guard_postcondition.has_triggered())
              return output_aggregator->guard_postcondition;
          }
          else
            return output_aggregator->effects_applied;
        }
#endif
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent OverwriteAnalysis::perform_output(RtEvent perform_precondition,
                                              std::set<RtEvent> &applied_events,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformOutputArgs args(this, trace_info);
        runtime->issue_runtime_meta_task(args,
            LG_THROUGHPUT_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.effects_event;
      }
      if (output_aggregator != NULL)
      {
        const ApEvent result = output_aggregator->summarize(trace_info); 
        if (result.exists() && track_effects)
          effects_events.insert(result);
        if (output_aggregator->release_guards(op->runtime, applied_events))
          delete output_aggregator;
      }
      if (!effects_events.empty())
        return Runtime::merge_events(&trace_info, effects_events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void OverwriteAnalysis::handle_remote_overwrites(
                 Deserializer &derez, Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr =
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      RegionUsage usage;
      derez.deserialize(usage);
      size_t num_views;
      derez.deserialize(num_views);
      FieldMaskSet<LogicalView> views;
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        views.insert(view, mask);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      FieldMaskSet<ReductionView> reductions;
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        views.insert(static_cast<ReductionView*>(view), mask);
      }
      const PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime);
      PredEvent pred_guard;
      derez.deserialize(pred_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      RtEvent guard_event;
      derez.deserialize(guard_event);
      bool add_restriction;
      derez.deserialize(add_restriction);
      RtUserEvent applied;
      derez.deserialize(applied);
      ApUserEvent effects;
      derez.deserialize(effects);

      // This takes ownership of the operation
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime,
          original_source, previous, op, index, expr, usage, views, 
          reductions, trace_info, precondition, guard_event, pred_guard, 
          effects.exists(), add_restriction);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())
        remote_ready = 
          analysis->perform_remote(traversal_done, applied_events);
      RtEvent output_ready;
      if (traversal_done.exists() || analysis->has_output_updates())
        output_ready = 
          analysis->perform_updates(traversal_done, applied_events);
      const ApEvent result = analysis->perform_output(
         Runtime::merge_events(remote_ready, output_ready), applied_events);
      if (effects.exists())
        Runtime::trigger_event(NULL, effects, result);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Filter Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(Runtime *rt, Operation *o, unsigned idx,
                              IndexSpaceExpression *expr, InstanceView *view,
                              LogicalView *reg_view, const bool remove_restrict)
      : PhysicalAnalysis(rt, o, idx, expr, true/*on heap*/), inst_view(view), 
        registration_view(reg_view), remove_restriction(remove_restrict)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(Runtime *rt, AddressSpaceID src, 
                              AddressSpaceID prev, Operation *o, unsigned idx, 
                              IndexSpaceExpression *expr, InstanceView *view, 
                              LogicalView *reg_view, const bool remove_restrict)
      : PhysicalAnalysis(rt, src, prev, o, idx, expr, true/*on heap*/),
        inst_view(view), registration_view(reg_view), 
        remove_restriction(remove_restrict)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(const FilterAnalysis &rhs)
      : PhysicalAnalysis(rhs), inst_view(rhs.inst_view), 
        registration_view(rhs.registration_view), 
        remove_restriction(rhs.remove_restriction)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FilterAnalysis::~FilterAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FilterAnalysis& FilterAnalysis::operator=(const FilterAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FilterAnalysis::perform_traversal(EquivalenceSet *set,
                                           IndexSpaceExpression *expr,
                                           const bool expr_covers,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->filter_set(*this, expr, expr_covers, mask, deferral_events, 
                      applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent FilterAnalysis::perform_remote(RtEvent perform_precondition,
                                           std::set<RtEvent> &applied_events,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first;
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          if (inst_view != NULL)
            rez.serialize(inst_view->did);
          else
            rez.serialize<DistributedID>(0);
          if (registration_view != NULL)
            rez.serialize(registration_view->did);
          else
            rez.serialize<DistributedID>(0);
          rez.serialize(remove_restriction);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_filters(target, rez);
        applied_events.insert(applied);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void FilterAnalysis::handle_remote_filters(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr =
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      DistributedID view_did;
      derez.deserialize(view_did);
      InstanceView *inst_view = NULL;
      if (view_did != 0)
      {
        RtEvent view_ready;
        inst_view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
        if (view_ready.exists())
          ready_events.insert(view_ready);
      }
      derez.deserialize(view_did);
      LogicalView *registration_view = NULL;
      if (view_did != 0)
      {
        RtEvent view_ready;
        registration_view = 
          runtime->find_or_request_logical_view(view_did, view_ready);
        if (view_ready.exists())
          ready_events.insert(view_ready);
      }
      bool remove_restriction;
      derez.deserialize(remove_restriction);
      RtUserEvent applied;
      derez.deserialize(applied);

      // This takes ownership of the remote operation
      FilterAnalysis *analysis = new FilterAnalysis(runtime, original_source,
                                      previous, op, index, expr, inst_view, 
                                      registration_view, remove_restriction);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      RtEvent remote_ready;
      if (traversal_done.exists() || analysis->has_remote_sets())     
        analysis->perform_remote(traversal_done, applied_events);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Clone Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CloneAnalysis::CloneAnalysis(Runtime *rt, IndexSpaceExpression *expr,
               Operation *op, unsigned idx, FieldMaskSet<EquivalenceSet> &&srcs)
      : PhysicalAnalysis(rt, op, idx, expr, true/*on heap*/),
        sources(std::move(srcs))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloneAnalysis::CloneAnalysis(Runtime *rt, AddressSpaceID src,
                                 AddressSpaceID prev,IndexSpaceExpression *expr,
                                 Operation *op, unsigned idx,
                                 FieldMaskSet<EquivalenceSet> &&srcs)
      : PhysicalAnalysis(rt, src, prev, op, idx, expr, true/*on heap*/),
        sources(std::move(srcs))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloneAnalysis::CloneAnalysis(const CloneAnalysis &rhs)
      : PhysicalAnalysis(rhs), sources(rhs.sources)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CloneAnalysis::~CloneAnalysis(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloneAnalysis& CloneAnalysis::operator=(const CloneAnalysis &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CloneAnalysis::perform_traversal(EquivalenceSet *set,
                                          IndexSpaceExpression *expr,
                                          const bool expr_covers,
                                          const FieldMask &mask,
                                          std::set<RtEvent> &deferral_events,
                                          std::set<RtEvent> &applied_events,
                                          const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->clone_set(*this, expr, expr_covers, mask, deferral_events,
                     applied_events, already_deferred);
    }

    //--------------------------------------------------------------------------
    RtEvent CloneAnalysis::perform_remote(RtEvent perform_precondition,
                                          std::set<RtEvent> &applied_events,
                                          const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() && 
          !perform_precondition.has_triggered())
      {
        // Defer this until the precondition is met
        DeferPerformRemoteArgs args(this);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, perform_precondition);
        applied_events.insert(args.applied_event);
        return args.done_event;
      }
      // If there are no sets we're done
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      for (LegionMap<AddressSpaceID,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpace target = rit->first;
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(sources.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                sources.begin(); it != sources.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_remote_clones(target, rez);
        applied_events.insert(applied);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CloneAnalysis::handle_remote_clones(Deserializer &derez,
                                      Runtime *runtime, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, NULL);
      LegionVector<FieldMask> eq_masks(num_eq_sets);
      for (unsigned idx = 0; idx < num_eq_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        eq_sets[idx] = runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        derez.deserialize(eq_masks[idx]);
      }
      IndexSpaceExpression *expr =
        IndexSpaceExpression::unpack_expression(derez,runtime->forest,previous);
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      size_t num_sources;
      derez.deserialize(num_sources);
      FieldMaskSet<EquivalenceSet> sources;
      for (unsigned idx = 0; idx < num_sources; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready); 
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        sources.insert(set, mask);
      }
      RtUserEvent applied;
      derez.deserialize(applied);

      // This takes ownership of the operation
      CloneAnalysis *analysis = new CloneAnalysis(runtime, original_source,
                            previous, expr, op, index, std::move(sources));
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
        RtEvent::NO_RT_EVENT : Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, applied_events);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(Runtime *rt, DistributedID id,
                                   AddressSpaceID logical,
                                   RegionNode *node, bool reg_now,
                                   CollectiveMapping *mapping /*= NULL*/,
                                   const FieldMask *replicated /*= NULL*/)
      : DistributedCollectable(rt,
          LEGION_DISTRIBUTED_HELP_ENCODE(id, EQUIVALENCE_SET_DC),
          reg_now, mapping), region_node(node), 
        set_expr(node->row_source), tracing_preconditions(NULL),
        tracing_anticonditions(NULL), tracing_postconditions(NULL), 
        logical_owner_space(logical), migration_index(0), sample_count(0)
    //--------------------------------------------------------------------------
    {
      set_expr->add_nested_expression_reference(did);
      region_node->add_nested_resource_ref(did);
      next_deferral_precondition.store(0);
      if (replicated != NULL)
      {
#ifdef DEBUG_LEGION
        assert(mapping != NULL);
#endif
        if (replicated_states.insert(mapping, *replicated))
          mapping->add_reference();
      }
#ifdef LEGION_GC
      log_garbage.info("GC Equivalence Set %lld %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(const EquivalenceSet &rhs)
      : DistributedCollectable(rhs), region_node(NULL), set_expr(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::~EquivalenceSet(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_valid_instances.empty());
      assert(partial_valid_instances.empty());
      assert(!partial_valid_fields);
      assert(initialized_data.empty());
      assert(reduction_instances.empty());
      assert(!reduction_fields);
      assert(restricted_instances.empty());
      assert(!restricted_fields);
      assert(released_instances.empty());
      assert(tracing_preconditions == NULL);
      assert(tracing_anticonditions == NULL);
      assert(tracing_postconditions == NULL);
#endif
      if (set_expr->remove_nested_expression_reference(did))
        delete set_expr;
      if (region_node->remove_nested_resource_ref(did))
        delete region_node;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet& EquivalenceSet::operator=(const EquivalenceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_local(void)
    //--------------------------------------------------------------------------
    {
      if (!total_valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it =
              total_valid_instances.begin(); it != 
              total_valid_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
        total_valid_instances.clear();
      }
      if (!partial_valid_instances.empty())
      {
        for (ViewExprMaskSets::iterator pit = 
              partial_valid_instances.begin(); pit != 
              partial_valid_instances.end(); pit++)
        {
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
                pit->second.begin(); it != pit->second.end(); it++)
            if (it->first->remove_nested_expression_reference(did))
              delete it->first;
          if (pit->first->remove_nested_valid_ref(did))
            delete pit->first;
          pit->second.clear();
        }
        partial_valid_instances.clear();
        partial_valid_fields.clear();
      }
      if (!initialized_data.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              initialized_data.begin(); it != initialized_data.end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
        initialized_data.clear();
      }
      if (!reduction_instances.empty())
      {
        for (std::map<unsigned,std::list<std::pair<ReductionView*,
              IndexSpaceExpression*> > >::iterator it =
              reduction_instances.begin(); it != 
              reduction_instances.end(); it++)
        {
          while (!it->second.empty())
          {
            std::pair<ReductionView*,IndexSpaceExpression*> &back = 
              it->second.back();
            if (back.first->remove_nested_valid_ref(did))
              delete back.first;
            if (back.second->remove_nested_expression_reference(did))
              delete back.second;
            it->second.pop_back();
          }
        }
        reduction_instances.clear();
        reduction_fields.clear();
      }
      if (!restricted_instances.empty())
      {
        for (ExprViewMaskSets::iterator rit =
              restricted_instances.begin(); rit != 
              restricted_instances.end(); rit++)
        {
          for (FieldMaskSet<InstanceView>::const_iterator it =
                rit->second.begin(); it != rit->second.end(); it++)
            if (it->first->remove_nested_valid_ref(did))
              delete it->first;
          if (rit->first->remove_nested_expression_reference(did))
            delete rit->first;
          rit->second.clear();
        }
        restricted_instances.clear();
        restricted_fields.clear();
      }
      if (!released_instances.empty())
      {
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
              rit != released_instances.end(); rit++)
        {
          for (FieldMaskSet<InstanceView>::const_iterator it =
                rit->second.begin(); it != rit->second.end(); it++)
            if (it->first->remove_nested_valid_ref(did))
              delete it->first;
          if (rit->first->remove_nested_expression_reference(did))
            delete rit->first;
          rit->second.clear();
        }
        released_instances.clear();
      }
      if (tracing_preconditions != NULL)
      {
        delete tracing_preconditions;
        tracing_preconditions = NULL;
      }
      if (tracing_anticonditions != NULL)
      {
        delete tracing_anticonditions;
        tracing_anticonditions = NULL;
      }
      if (tracing_postconditions != NULL)
      {
        delete tracing_postconditions;
        tracing_postconditions = NULL;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_collective_references(unsigned local_valid)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(collective_mapping != NULL);
#endif
      if (is_owner())
      {
        if (local_valid > 0) 
          add_base_gc_ref(CONTEXT_REF, local_valid);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(local_valid > 0);
#endif
        add_base_gc_ref(CONTEXT_REF, local_valid);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_set(const RegionUsage &usage,
                                        const FieldMask &user_mask,
                                        const bool restricted,
                                        const InstanceSet &sources,
                                const std::vector<InstanceView*> &corresponding)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner() || 
              (!(user_mask - replicated_states.get_valid_mask())) ||
              region_node->row_source->is_empty());
      assert(sources.size() == corresponding.size());
#endif
      AutoLock eq(eq_lock);
      if (IS_REDUCE(usage))
      {
#ifdef DEBUG_LEGION
        // Reduction-only should always be restricted for now
        // Could change if we started issuing reduction close
        // operations at the end of a context
        assert(restricted);
#endif
        // Since these are restricted, we'll make these the actual
        // target logical instances and record them as restricted
        // instead of recording them as reduction instances
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask &view_mask = sources[idx].get_valid_fields();
          InstanceView *view = corresponding[idx];
          FieldMaskSet<LogicalView>::iterator finder = 
            total_valid_instances.find(view);
          if (finder == total_valid_instances.end())
          {
            total_valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did);
          }
          else
            finder.merge(view_mask);
          // Always restrict reduction-only users since we know the data
          // is going to need to be flushed anyway
          record_restriction(set_expr, true/*covers*/, view_mask, view);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < sources.size(); idx++)
        {
          const FieldMask &view_mask = sources[idx].get_valid_fields();
          InstanceView *view = corresponding[idx];
#ifdef DEBUG_LEGION
          assert(!view->is_reduction_view());
#endif
          FieldMaskSet<LogicalView>::iterator finder = 
            total_valid_instances.find(view);
          if (finder == total_valid_instances.end())
          {
            total_valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did);
          }
          else
            finder.merge(view_mask);
          // If this is restricted then record it
          if (restricted)
            record_restriction(set_expr, true/*covers*/, view_mask, view);
        }
      }
      // Record that this data is all valid
      update_initialized_data(set_expr, true/*covers*/, user_mask);
      // Update any restricted fields 
      if (restricted)
      {
#ifdef DEBUG_LEGION
        assert(!restricted_instances.empty());
#endif
        restricted_fields |= user_mask;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_valid_instances(ValidInstAnalysis &analysis,
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers, 
                                             const FieldMask &user_mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      // This is a read-only analysis so we don't need to invalidate
      // replicated state if we get a non-collective operation
      if (is_remote_analysis(analysis, user_mask, deferral_events,
            applied_events, false/*exclusive*/, true/*immutable*/))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(user_mask));
#endif
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      if (analysis.redop != 0)
      {
        // Iterate over all the fields
        int fidx = user_mask.find_first_set();
        while (fidx >= 0)
        {
          std::map<unsigned,std::list<
            std::pair<ReductionView*,IndexSpaceExpression*> > >::iterator
              current = reduction_instances.find(fidx);
          if (current != reduction_instances.end())
          {
            FieldMask local_mask;
            local_mask.set_bit(fidx);
            for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >
                  ::const_reverse_iterator it = current->second.rbegin(); it !=
                  current->second.rend(); it++)
            {
              PhysicalManager *manager = it->first->get_manager();
              if (manager->redop != analysis.redop)
                break;
              if (!expr_covers)
              {
                IndexSpaceExpression *overlap = 
                  runtime->forest->intersect_index_spaces(expr, it->second);
                if (overlap->is_empty())
                  continue;
              }
              analysis.record_instance(it->first, local_mask);
            }
          }
          fidx = user_mask.find_next_set(fidx+1);
        }
      }
      else
      {
        if (!(user_mask * total_valid_instances.get_valid_mask()))
        {
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                total_valid_instances.begin(); it != 
                total_valid_instances.end(); it++)
          {
            if (!it->first->is_instance_view())
              continue;
            const FieldMask overlap = it->second & user_mask;
            if (!overlap)
              continue;
            analysis.record_instance(it->first->as_instance_view(), overlap);
          }
        }
        if (!(user_mask * partial_valid_fields))
        {
          for (ViewExprMaskSets::const_iterator pit = 
                partial_valid_instances.begin(); pit !=
                partial_valid_instances.end(); pit++)
          {
            if (!pit->first->is_instance_view())
              continue;
            if (expr_covers)
            {
              const FieldMask overlap = 
                user_mask & pit->second.get_valid_mask();
              if (!!overlap)
                analysis.record_instance(pit->first->as_instance_view(), 
                                         overlap);
              continue;
            }
            else if (user_mask * pit->second.get_valid_mask())
              continue;
            FieldMask total_overlap;
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  pit->second.begin(); it != pit->second.end(); it++)
            {
              const FieldMask overlap = user_mask & it->second;
              if (!overlap)
                continue;
              IndexSpaceExpression *expr_overlap = 
                  runtime->forest->intersect_index_spaces(expr, it->first);   
              if (expr_overlap->is_empty())
                continue;
              total_overlap |= overlap;
            }
            if (!!total_overlap)
              analysis.record_instance(pit->first->as_instance_view(), 
                                       total_overlap);
          }
        }
      }
      if (!(user_mask * restricted_fields))
      {
        if (!expr_covers)
        {
          // Check for the set expr first which we know overlaps
          ExprViewMaskSets::const_iterator finder =
            restricted_instances.find(set_expr);
          if ((finder == restricted_instances.end()) || 
              (finder->second.get_valid_mask() * user_mask))
          {
            for (ExprViewMaskSets::const_iterator it =
                  restricted_instances.begin(); it != 
                  restricted_instances.end(); it++)
            {
              if (it == finder)
                continue;
              if (it->second.get_valid_mask() * user_mask)
                continue;
              IndexSpaceExpression *overlap = 
                runtime->forest->intersect_index_spaces(it->first, expr);
              if (overlap->is_empty())
                continue;
              analysis.record_restriction();
              break;
            }
          }
          else
            analysis.record_restriction();
        }
        else
          analysis.record_restriction();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_invalid_instances(InvalidInstAnalysis &analysis,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers, 
                                    const FieldMask &user_mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      // This is a read-only analysis so we don't need to invalidate
      // replicated state if we get a non-collective operation
      if (is_remote_analysis(analysis, user_mask, deferral_events,
            applied_events, false/*exclusive*/, true/*immutable*/))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(user_mask));
#endif
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            analysis.valid_instances.begin(); it !=
            analysis.valid_instances.end(); it++)
      {
        FieldMask invalid_mask = it->second & user_mask;
        if (!invalid_mask)
          continue;
        if (it->first->is_reduction_view())
        {
          // Handle reductions special
          ReductionView *reduction_view = it->first->as_reduction_view();
          if (!(invalid_mask - reduction_fields))
          {
            int fidx = invalid_mask.find_first_set();
            while (fidx >= 0)
            {
              std::map<unsigned,std::list<std::pair<ReductionView*,
                IndexSpaceExpression*> > >::const_iterator finder = 
                  reduction_instances.find(fidx);
              if (finder == reduction_instances.end())
                break;
              std::set<IndexSpaceExpression*> exprs;
              for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
                    const_reverse_iterator rit = finder->second.rbegin(); rit !=
                    finder->second.rend(); rit++)
              {
                // Can't go backwards through any reductions of a different type
                if (rit->first->get_redop() != reduction_view->get_redop())
                  break;
                if (rit->first != reduction_view)
                  continue;
                if ((rit->second == expr) || (rit->second == set_expr))
                {
                  // covers everything
                  invalid_mask.unset_bit(fidx);
                  exprs.clear();
                  break;
                }
                else
                  exprs.insert(rit->second);
              }
              if (!exprs.empty())
              {
                // See if they cover
                IndexSpaceExpression *union_expr = 
                  runtime->forest->union_index_spaces(exprs);
                IndexSpaceExpression *intersection =
                  runtime->forest->intersect_index_spaces(expr, union_expr);
                if (intersection->get_volume() == expr->get_volume())
                  invalid_mask.unset_bit(fidx);
                else // no point in checking the rest
                  break;
              }
              // No point in checking the rest if this wasn't covered
              else if (invalid_mask.is_set(fidx))
                break;
              fidx = invalid_mask.find_next_set(fidx+1);
            }
          }
          if (!!invalid_mask)
            analysis.record_instance(reduction_view, invalid_mask);
          continue;
        }
        // Check the covering instances first
        if (!total_valid_instances.empty())
        {
          FieldMaskSet<LogicalView>::const_iterator finder =
            total_valid_instances.find(it->first);
          if (finder != total_valid_instances.end())
          {
            invalid_mask -= finder->second;
            if (!invalid_mask)
              continue;
          }
        }
        if (!expr_covers && !!invalid_mask)
        {
          ViewExprMaskSets::const_iterator finder =
            partial_valid_instances.find(it->first);
          if ((finder != partial_valid_instances.end()) &&
              !(finder->second.get_valid_mask() * invalid_mask))
          {
            FieldMaskSet<IndexSpaceExpression>::const_iterator expr_finder =
              finder->second.find(expr);
            if (expr_finder != finder->second.end())
            {
              invalid_mask -= expr_finder->second;
              if (!invalid_mask)
                continue;
            }
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  finder->second.begin(); it != finder->second.end(); it++)
            {
              if (it->first == expr)
                continue;
              const FieldMask overlap = it->second & invalid_mask;
              if (!overlap)
                continue;
              IndexSpaceExpression *expr_overlap = 
                runtime->forest->intersect_index_spaces(expr, it->first);
              if (expr_overlap->get_volume() == expr->get_volume())
              {
                invalid_mask -= overlap;
                if (!invalid_mask)
                  break;
              }
            }
            if (!invalid_mask)
              continue;
          }
        }
        // If we get here it's because we're not valid for some expression
        // for these fields so record it
        analysis.record_instance(it->first, invalid_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_antivalid_instances(
                                            AntivalidInstAnalysis &analysis,
                                            IndexSpaceExpression *expr,
                                            const bool expr_covers, 
                                            const FieldMask &user_mask,
                                            std::set<RtEvent> &deferral_events,
                                            std::set<RtEvent> &applied_events,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      // This is a read-only analysis so we don't need to invalidate
      // replicated state if we get a non-collective operation
      if (is_remote_analysis(analysis, user_mask, deferral_events,
            applied_events, false/*exclusive*/, true/*immutable*/))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(user_mask));
#endif
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      for (FieldMaskSet<LogicalView>::const_iterator ait = 
            analysis.antivalid_instances.begin(); ait !=
            analysis.antivalid_instances.end(); ait++)
      {
        const FieldMask antivalid_mask = ait->second & user_mask;
        if (!antivalid_mask)
          continue;
        if (ait->first->is_reduction_view())
        {
          // Handle reductions special
          if (antivalid_mask * reduction_fields)
            continue;
          ReductionView *reduction_view = ait->first->as_reduction_view();
          int fidx = antivalid_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<unsigned,std::list<std::pair<ReductionView*,
              IndexSpaceExpression*> > >::const_iterator finder = 
                reduction_instances.find(fidx);
            if (finder != reduction_instances.end())
            {
              for (std::list<std::pair<ReductionView*,
                    IndexSpaceExpression*> >::const_iterator it = 
                    finder->second.begin(); it != finder->second.end(); it++)
              {
                if (it->first != reduction_view)
                  continue;
                FieldMask local_mask;
                local_mask.set_bit(fidx);
                if (!expr_covers)
                {
                  if ((it->second != set_expr) && (it->second != expr)) 
                  {
                    IndexSpaceExpression *intersection = 
                      runtime->forest->intersect_index_spaces(expr, it->second);
                    if (!intersection->is_empty())
                      analysis.record_instance(reduction_view, local_mask);
                  }
                  else
                    analysis.record_instance(reduction_view, local_mask);
                }
                else // they intersect so record it
                  analysis.record_instance(reduction_view, local_mask);
              }
            }
            fidx = antivalid_mask.find_next_set(fidx+1);
          }
        }
        else
        {
          // Check for it in the total valid instances first
          FieldMaskSet<LogicalView>::const_iterator total_finder = 
            total_valid_instances.find(ait->first);
          if (total_finder != total_valid_instances.end())
          {
            const FieldMask overlap = antivalid_mask & total_finder->second;
            if (!!overlap)
              analysis.record_instance(ait->first, overlap);
          }
          // Then check for it in the partial valid instances
          ViewExprMaskSets::const_iterator finder = 
            partial_valid_instances.find(ait->first);
          if (finder != partial_valid_instances.end())
          {
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
                  finder->second.begin(); it != finder->second.end(); it++)
            {
              const FieldMask overlap = it->second & antivalid_mask;
              if (!overlap)
                continue;
              if (!expr_covers)
              {
                if ((it->first != set_expr) && (it->first != expr))
                {
                  IndexSpaceExpression *intersection = 
                    runtime->forest->intersect_index_spaces(expr, it->first);
                  if (!intersection->is_empty())
                    analysis.record_instance(ait->first, overlap);
                }
                else
                  analysis.record_instance(ait->first, overlap);
              }
              else
                analysis.record_instance(ait->first, overlap);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set(UpdateAnalysis &analysis,
                                    IndexSpaceExpression *expr,
                                    const bool expr_covers,
                                    FieldMask user_mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred/*=false*/)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, user_mask, deferral_events,
            applied_events, expr_covers && IS_WRITE(analysis.usage)))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(user_mask));
#endif
      // Now that we're ready to perform the analysis 
      // we need to lock the analysis 
      AutoLock a_lock(analysis);
      // Check for any uninitialized data
      // Don't report uninitialized warnings for empty equivalence classes
      if (analysis.check_initialized && !set_expr->is_empty())
        check_for_uninitialized_data(analysis, expr, expr_covers, 
                                     user_mask, applied_events);
      if (analysis.output_aggregator != NULL)
        analysis.output_aggregator->clear_update_fields();
      if (IS_REDUCE(analysis.usage))
      {
        // Reduction-only
        // We only record reductions if the set expression is not empty
        // as we can't guarantee the reductions will ever be read for 
        // empty equivalence sets which can lead to leaked instances
        if (!expr->is_empty())
        {
          CopyFillAggregator *fill_aggregator = NULL;
          // See if we have an input aggregator that we can use now
          // for any fills that need to be done to initialize instances
          std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
            analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
          if (finder != analysis.input_aggregators.end())
            fill_aggregator = finder->second;
          FieldMask guard_fill_mask;
          // Record the reduction instances
          for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
          {
            ReductionView *red_view = 
              analysis.target_views[idx]->as_reduction_view();
            const ReductionOpID view_redop = red_view->get_redop(); 
#ifdef DEBUG_LEGION
            assert(view_redop == analysis.usage.redop);
#endif
            const FieldMask &update_fields = 
              analysis.target_instances[idx].get_valid_fields(); 
            reduction_fields |= update_fields;
            // Track the case where this reduction view is also
            // stored in the valid instances in which case we 
            // do not need to do any fills. This will only happen
            // if these fields are restricted because they are in
            // the total_valid_instances. 
            bool already_valid = false;
            if (!!restricted_fields && !(update_fields * restricted_fields) &&
                (total_valid_instances.find(red_view) != 
                 total_valid_instances.end()))
              already_valid = true;
#ifdef DEBUG_LEGION
            assert(partial_valid_instances.find(red_view) == 
                    partial_valid_instances.end());
#endif
            int fidx = update_fields.find_first_set();
            // Figure out which fields require a fill operation
            // in order initialize the reduction instances
            FieldMaskSet<IndexSpaceExpression> fill_exprs;
            while (fidx >= 0)
            {
              std::list<std::pair<ReductionView*,IndexSpaceExpression*> >
                &field_views = reduction_instances[fidx]; 
              // Scan through the reduction instances to see if we're
              // already in the list of valid reductions, if not then
              // we're going to need a fill to initialize the instance
              // Also check for the ABA problem on reduction instances
              // described in Legion issue #545 where we start out
              // with reductions of kind A, switch to reductions of
              // kind B, and then switch back to reductions of kind A
              // which will make it unsafe to re-use the instance
              bool found_covered = already_valid && 
                total_valid_instances[red_view].is_set(fidx);
              std::set<IndexSpaceExpression*> found_exprs;
              // We only need to do this check if it's not already-covered
              // In the case where we know that it is already covered
              // at this point it is restricted, so everything is being
              // flushed to it anyway
              if (!found_covered)
              {
                for (std::list<std::pair<ReductionView*,
                      IndexSpaceExpression*> >::iterator it =
                      field_views.begin(); it != field_views.end(); it++)
                {
                  if (it->first != red_view)
                  {
                    if (!found_covered && found_exprs.empty())
                      continue;
                    if (it->first->get_redop() == view_redop)
                      continue;
                    // Check for intersection
                    if (found_covered)
                    {
                      if (!expr_covers && (expr != it->second))
                      {
                        IndexSpaceExpression *overlap = 
                          runtime->forest->intersect_index_spaces(expr, 
                                                                  it->second);
                        if (overlap->is_empty())
                          continue;
                      }
                    }
                    else
                    {
                      // Check each of the individual expressions for overlap
                      bool all_disjoint = true;
                      for (std::set<IndexSpaceExpression*>::const_iterator fit =
                           found_exprs.begin(); fit != found_exprs.end(); fit++)
                      {
                        IndexSpaceExpression *overlap = 
                          runtime->forest->intersect_index_spaces(it->second,
                                                                  *fit);
                        if (overlap->is_empty())
                          continue;
                        all_disjoint = false;
                        break;
                      }
                      if (all_disjoint)
                        continue;
                    }
                    // If we make it here, report the ABA violation
                    REPORT_LEGION_FATAL(LEGION_FATAL_REDUCTION_ABA_PROBLEM,
                        "Unsafe re-use of reduction instance detected due "
                        "to alternating un-flushed reduction operations "
                        "%d and %d. Please report this use case to the "
                        "Legion developer's mailing list so that we can "
                        "help you address it.", view_redop,
                        it->first->get_redop())
                  }
                  else if (!found_covered)
                  {
                    if (!expr_covers)
                    {
                      if (expr != it->second)
                      {
                        IndexSpaceExpression *overlap = 
                          runtime->forest->intersect_index_spaces(expr,
                                                            it->second);
                        if (overlap->get_volume() < expr->get_volume())
                        {
                          found_exprs.insert(overlap);
                          // Promote this to be the union of the two
                          if (overlap->get_volume() < it->second->get_volume())
                          {
                            IndexSpaceExpression *union_expr =
                              runtime->forest->union_index_spaces(expr,
                                                            it->second);
                            union_expr->add_nested_expression_reference(did);
                            if (it->second->remove_nested_expression_reference(
                                                                          did))
                              delete it->second;
                            it->second = union_expr;
                          }
                          else
                          {
                            expr->add_nested_expression_reference(did);
                            if (it->second->remove_nested_expression_reference(
                                                                          did))
                              delete it->second;
                            it->second = expr;
                          }
                        }
                        else
                          found_covered = true;
                      }
                      else
                        found_covered = true;
                    }
                    else
                    {
                      if ((it->second != set_expr) &&
                          (it->second->get_volume() < set_expr->get_volume()))
                      {
                        found_exprs.insert(it->second);
                        // Promote this up to the full set expression
                        set_expr->add_nested_expression_reference(did);
                        // Since we're going to use the old expression, we need
                        // to keep it live until the end of the task
                        it->second->add_base_expression_reference(
                                                              LIVE_EXPR_REF);
                        ImplicitReferenceTracker::record_live_expression(
                                                                  it->second);
                        // Now we can remove the previous live reference
                        if (it->second->remove_nested_expression_reference(did))
                          delete it->second;
                        it->second = set_expr;
                      }
                      else
                        found_covered = true;
                    }
                  }
                }
                // See if there are any fill expressions that we need to do
                // These are also the expressions that we need to add to the
                // fields views set since they won't be described by prior
                // reductions already on the list
                if (!found_covered)
                {
                  FieldMask fill_mask;
                  fill_mask.set_bit(fidx);
                  if (!found_exprs.empty())
                  {
                    guard_fill_mask.set_bit(fidx);
                    // See if the union dominates the expression, if not
                    // put in the difference
                    IndexSpaceExpression *union_expr = 
                      runtime->forest->union_index_spaces(found_exprs);
                    if (union_expr->get_volume() < expr->get_volume())
                    {
                      IndexSpaceExpression *diff_expr =
                        runtime->forest->subtract_index_spaces(expr,union_expr);
                      fill_exprs.insert(diff_expr, fill_mask);
                      red_view->add_nested_valid_ref(did);
                      diff_expr->add_nested_expression_reference(did);
                      field_views.push_back(std::make_pair(red_view,diff_expr));
                    }
                  }
                  else
                  {
                    fill_exprs.insert(expr, fill_mask);
                    // No previous exprs, so record the full thing
                    red_view->add_nested_valid_ref(did);
                    expr->add_nested_expression_reference(did);
                    field_views.push_back(std::make_pair(red_view, expr));
                  }
                }
                else
                  guard_fill_mask.set_bit(fidx);
              }
              else
              {
                // This is already restricted, so just add it,
                // we'll be flushing it here shortly
                red_view->add_nested_valid_ref(did);
                expr->add_nested_expression_reference(did);
                field_views.push_back(std::make_pair(red_view, expr));
              }
#ifdef DEBUG_LEGION
              assert(!field_views.empty());
#endif
              fidx = update_fields.find_next_set(fidx+1);
            }
            if (!fill_exprs.empty())
            {
              if (fill_aggregator == NULL)
              {
                // Fill aggregators never need to wait for any other
                // aggregators since we know they won't depend on each other
                fill_aggregator = new CopyFillAggregator(runtime->forest,
                    analysis.op, analysis.index, analysis.index,
                    NULL/*no previous guard*/, false/*track events*/);
                analysis.input_aggregators[RtEvent::NO_RT_EVENT] = 
                  fill_aggregator;
              }
              // Record the fill operation on the aggregator
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    fill_exprs.begin(); it != fill_exprs.end(); it++)
                fill_aggregator->record_fill(red_view, red_view->fill_view,
                                  it->second, it->first, 
                                  analysis.trace_info.recording ? this : NULL);
              // Record this as a guard for later operations
              reduction_fill_guards.insert(fill_aggregator,
                                           fill_exprs.get_valid_mask());
#ifdef DEBUG_LEGION
              if (!fill_aggregator->record_guard_set(this, false/*read only*/))
                assert(false);
#else
              fill_aggregator->record_guard_set(this, false/*read only*/);
#endif
            }
          }
          // If we have any fills that were issued by a prior operation
          // that we need to reuse then check for them here. This is a
          // slight over-approximation for the mapping dependences because
          // we really only need to wait for fills to instances that we
          // care about it, but it should be minimal overhead and the
          // resulting event graph will still be precise
          if (!reduction_fill_guards.empty() && !!guard_fill_mask &&
              !(reduction_fill_guards.get_valid_mask() * guard_fill_mask))
          {
            for (FieldMaskSet<CopyFillGuard>::iterator it = 
                  reduction_fill_guards.begin(); it != 
                  reduction_fill_guards.end(); it++)
            {
              if (it->first == fill_aggregator)
                continue;
              const FieldMask guard_mask = guard_fill_mask & it->second;
              if (!guard_mask)
                continue;
              // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
              analysis.guard_events.insert(it->first->effects_applied);
#else
              if (analysis.original_source == local_space)
                analysis.guard_events.insert(it->first->guard_postcondition);
              else
                analysis.guard_events.insert(it->first->effects_applied);
#endif
            }
          }
          // Flush any reductions for restricted fields and expressions
          if (!restricted_instances.empty())
          {
            const FieldMask flush_mask = user_mask & restricted_fields;
            if (!!flush_mask)
            {
              // Find all the restrictions that we overlap with and
              // apply reductions to them to flush any data
              for (ExprViewMaskSets::const_iterator rit =
                    restricted_instances.begin(); rit != 
                    restricted_instances.end(); rit++)
              {
                const FieldMask overlap = 
                  flush_mask & rit->second.get_valid_mask(); 
                if (!overlap)
                  continue;
                if (expr_covers)
                {
                  // Expression covers the full restriction
                  if (rit->first->get_volume() == set_expr->get_volume())
                    apply_reductions(rit->second, set_expr, true/*covers*/, 
                        overlap,analysis.output_aggregator, NULL/*no guard*/,
                        analysis.op, analysis.index, true/*track events*/,
                        analysis.trace_info, NULL/*no applied expr tracking*/);
                  else
                    apply_reductions(rit->second, rit->first, false/*covers*/,
                        overlap,analysis.output_aggregator, NULL/*no guard*/,
                        analysis.op, analysis.index, true/*track events*/,
                        analysis.trace_info, NULL/*no applied expr tracking*/);
                }
                else if (rit->first == set_expr)
                {
                  // Restriction covers the full expression
                  apply_reductions(rit->second, expr, expr_covers, overlap,
                      analysis.output_aggregator, NULL/*no guard*/,
                      analysis.op, analysis.index, true/*track events*/,
                      analysis.trace_info, NULL/*no applied expr tracking*/);
                }
                else
                {
                  // Check for partial overlap of restrction
                  IndexSpaceExpression *expr_overlap =
                    runtime->forest->intersect_index_spaces(expr, rit->first);
                  if (expr_overlap->is_empty())
                    continue;
                  if (expr_overlap->get_volume() == expr->get_volume())
                    apply_reductions(rit->second, expr, expr_covers, overlap,
                        analysis.output_aggregator, NULL/*no guard*/,
                        analysis.op, analysis.index, true/*track events*/,
                        analysis.trace_info, NULL/*no applied expr tracking*/);
                  else
                    apply_reductions(rit->second, expr_overlap, false/*covers*/,
                        overlap,analysis.output_aggregator, NULL/*no guard*/,
                        analysis.op, analysis.index, true/*track events*/,
                        analysis.trace_info, NULL/*no applied expr tracking*/);
                }
              }
            }
          }
        }
      }
      else if (IS_WRITE(analysis.usage) && IS_DISCARD(analysis.usage))
      {
        // Write-only
        // Update the initialized data before messing with the user mask
        update_initialized_data(expr, expr_covers, user_mask);
        const FieldMask reduce_filter = reduction_fields & user_mask;
        if (!!reduce_filter)
          filter_reduction_instances(expr, expr_covers, reduce_filter);
        FieldMaskSet<InstanceView> new_instances;
        for (unsigned idx = 0; idx < analysis.target_instances.size(); idx++)
        {
          const FieldMask overlap = user_mask & 
            analysis.target_instances[idx].get_valid_fields(); 
          if (!overlap)
            continue;
          new_instances.insert(analysis.target_views[idx], overlap);
        }
        // Filter any normal instances that will be overwritten
        const FieldMask non_restricted = user_mask - restricted_fields; 
        if (!!non_restricted)
        {
          filter_valid_instances(expr, expr_covers, non_restricted);
          // Record any non-restricted instances
          record_instances(expr, expr_covers, non_restricted,
                           new_instances);
        }
        // Issue copy-out copies for any restricted fields
        if (!restricted_instances.empty())
        {
          const FieldMask restricted_mask = user_mask & restricted_fields;
          if (!!restricted_mask)
          {
            filter_unrestricted_instances(expr, expr_covers, 
                                          restricted_mask);
            // Record any of our instances that are unrestricted
            record_unrestricted_instances(expr, expr_covers, restricted_mask,
                                          new_instances);
            copy_out(expr, expr_covers, restricted_mask, new_instances,
                analysis.op, analysis.index, analysis.trace_info,
                analysis.output_aggregator);
          }
        }
      }
      else if (IS_READ_ONLY(analysis.usage) && !read_only_guards.empty() && 
                !(user_mask * read_only_guards.get_valid_mask()))
      {
        // If we're doing read-only mode, get the set of events that
        // we need to wait for before we can do our registration, this 
        // ensures that we serialize read-only operations correctly
        // In order to avoid deadlock we have to make different copy fill
        // aggregators for each of the different fields of prior updates
        FieldMaskSet<InstanceView> new_instances;
        for (unsigned idx = 0; idx < analysis.target_instances.size(); idx++)
        {
          const FieldMask overlap = user_mask & 
            analysis.target_instances[idx].get_valid_fields(); 
          if (!overlap)
            continue;
          new_instances.insert(analysis.target_views[idx], overlap);
        }
        FieldMaskSet<CopyFillAggregator> to_add;
        for (FieldMaskSet<CopyFillGuard>::iterator it = 
              read_only_guards.begin(); it != read_only_guards.end(); it++)
        {
          const FieldMask guard_mask = user_mask & it->second;
          if (!guard_mask)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          const RtEvent guard_event = it->first->effects_applied;
          analysis.guard_events.insert(guard_event);
#else
          const RtEvent guard_event = it->first->guard_postcondition;
          if (analysis.original_source == local_space)
            analysis.guard_events.insert(guard_event);
          else
            analysis.guard_events.insert(it->first->effects_applied);
#endif
          CopyFillAggregator *input_aggregator = NULL;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
            analysis.input_aggregators.find(guard_event);
          if (finder != analysis.input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != NULL)
              input_aggregator->clear_update_fields();
          } 
          // Use this to see if any new updates are recorded
          update_set_internal(input_aggregator, it->first, 
                              analysis.op, analysis.index,
                              analysis.usage, expr, expr_covers, 
                              guard_mask, new_instances, analysis.source_views,
                              analysis.trace_info, analysis.record_valid);
          // If we did any updates record ourselves as the new guard here
          if ((input_aggregator != NULL) && 
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            if (finder == analysis.input_aggregators.end())
              analysis.input_aggregators[guard_event] = input_aggregator;
            const FieldMask &update_mask = 
              input_aggregator->get_update_fields();
            // Record this as a guard for later operations
            to_add.insert(input_aggregator, update_mask);
#ifdef DEBUG_LEGION
            if (!input_aggregator->record_guard_set(this, true/*read only*/))
              assert(false);
#else
            input_aggregator->record_guard_set(this, true/*read only*/);
#endif
            // Remove the current guard since it doesn't matter anymore
            it.filter(update_mask);
          }
          user_mask -= guard_mask;
          if (!user_mask)
            break;
        }
        if (!to_add.empty())
        {
          for (FieldMaskSet<CopyFillAggregator>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            read_only_guards.insert(it->first, it->second);
        }
        // If we have unguarded fields we can easily do those
        if (!!user_mask)
        {
          CopyFillAggregator *input_aggregator = NULL;
          // See if we have an input aggregator that we can use now
          std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
            analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
          if (finder != analysis.input_aggregators.end())
          {
            input_aggregator = finder->second;
            if (input_aggregator != NULL)
              input_aggregator->clear_update_fields();
          }
          update_set_internal(input_aggregator, NULL/*no previous guard*/,
                              analysis.op, analysis.index,
                              analysis.usage, expr, expr_covers,
                              user_mask, new_instances, analysis.source_views,
                              analysis.trace_info, analysis.record_valid);
          // If we made the input aggregator then store it
          if ((input_aggregator != NULL) &&
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
            // Record this as a guard for later operations
            read_only_guards.insert(input_aggregator, 
                input_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
            if (!input_aggregator->record_guard_set(this, true/*read only*/))
              assert(false);
#else
            input_aggregator->record_guard_set(this, true/*read only*/);
#endif
          }
        }
      }
      else
      {
        // Read-write or read-only case
        // Read-only case if there are no guards
        CopyFillAggregator *input_aggregator = NULL;
        // See if we have an input aggregator that we can use now
        std::map<RtEvent,CopyFillAggregator*>::const_iterator finder = 
          analysis.input_aggregators.find(RtEvent::NO_RT_EVENT);
        if (finder != analysis.input_aggregators.end())
        {
          input_aggregator = finder->second;
          if (input_aggregator != NULL)
            input_aggregator->clear_update_fields();
        }
        FieldMaskSet<InstanceView> new_instances;
        for (unsigned idx = 0; idx < analysis.target_instances.size(); idx++)
        {
          const FieldMask overlap = user_mask & 
            analysis.target_instances[idx].get_valid_fields(); 
          if (!overlap)
            continue;
          new_instances.insert(analysis.target_views[idx], overlap);
        }
        update_set_internal(input_aggregator, NULL/*no previous guard*/,
                            analysis.op, analysis.index, analysis.usage,
                            expr, expr_covers, user_mask, new_instances,
                            analysis.source_views, analysis.trace_info,
                            analysis.record_valid);
        if (IS_WRITE(analysis.usage))
        {
          update_initialized_data(expr, expr_covers, user_mask);
          // Issue copy-out copies for any restricted fields if we wrote stuff
          if (!restricted_instances.empty())
          {
            const FieldMask restricted_mask = user_mask & restricted_fields;
            if (!!restricted_mask)
              copy_out(expr, expr_covers, restricted_mask, new_instances, 
                       analysis.op, analysis.index, analysis.trace_info,
                       analysis.output_aggregator);
          }
        }
        // If we made the input aggregator then store it
        if ((input_aggregator != NULL) &&
            ((finder == analysis.input_aggregators.end()) ||
             input_aggregator->has_update_fields()))
        {
          analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
          // Record this as a guard for later read-only operations
          if (IS_READ_ONLY(analysis.usage))
          {
            read_only_guards.insert(input_aggregator, 
                input_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
            if (!input_aggregator->record_guard_set(this, true/*read only*/))
              assert(false);
#else
            input_aggregator->record_guard_set(this, true/*read only*/);
#endif
          }
        }
      }
      // Update the post conditions for these views if we're recording 
      if (analysis.trace_info.recording)
      {
        if (tracing_postconditions == NULL)
          tracing_postconditions = 
            new TraceViewSet(runtime->forest, did, region_node);
        for (unsigned idx = 0; idx < analysis.target_instances.size(); idx++)
        {
          const FieldMask &mask = 
            analysis.target_instances[idx].get_valid_fields();
          update_tracing_valid_views(analysis.target_views[idx], expr,
              analysis.usage, mask, IS_WRITE(analysis.usage));
        }
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_migration(PhysicalAnalysis &analysis,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_EQUIVALENCE_SET_MIGRATION
      // Never migrate when we are replicated
      if (!replicated_states.empty())
        return;
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      const AddressSpaceID eq_source = analysis.original_source;
      // Record our user in the set of previous users
      bool found = false;
      std::vector<std::pair<AddressSpaceID,unsigned> > &current_samples = 
        user_samples[migration_index];
      for (std::vector<std::pair<AddressSpaceID,unsigned> >::iterator it =
            current_samples.begin(); it != current_samples.end(); it++)
      {
        if (it->first != eq_source)
          continue;
        found = true;
        it->second++;
        break;
      }
      if (!found)
        current_samples.push_back(
            std::pair<AddressSpaceID,unsigned>(eq_source,1));
      // Increase the sample count and if we haven't done enough
      // for a test then we can return and keep going
      if (++sample_count < SAMPLES_PER_MIGRATION_TEST)
      {
        // Check to see if the request bounced off a stale owner 
        // and we should send the update message
        if ((eq_source != analysis.previous) && (eq_source != local_space) &&
            (eq_source != logical_owner_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          runtime->send_equivalence_set_owner_update(eq_source, rez);
          applied_events.insert(notification_event);
        }
        return;
      }
      // Issue a warning and don't migrate if we hit this case
      if (current_samples.size() == SAMPLES_PER_MIGRATION_TEST)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_LARGE_EQUIVALENCE_SET_NODE_USAGE,
            "Internal runtime performance warning: equivalence set %llx of "
            "region (%d,%d,%d) has %zd different users which is the same as "
            "the sampling rate of %d. Region requirement %d of operation %s "
            "(UID %lld) triggered this warning. Please report this "
            "application use case to the Legion developers mailing list.",
            did, region_node->handle.get_index_space().get_id(),
            region_node->handle.get_field_space().get_id(),
            region_node->handle.get_tree_id(), current_samples.size(),
            SAMPLES_PER_MIGRATION_TEST, analysis.index,
            (analysis.op->get_operation_kind() == Operation::TASK_OP_KIND) ?
              static_cast<TaskOp*>(analysis.op)->get_task_name() :
            analysis.op->get_logging_name(), analysis.op->get_unique_op_id())
        // Reset the data structures for the next run
        current_samples.clear();
        sample_count = 0;
        return;
      }
      // Sort the current samples so that they are in order for
      // single epoch cases, for multi-epoch cases they will be
      // sorted by the summary computation below
      if ((MIGRATION_EPOCHS == 1) && (current_samples.size() > 1))
        std::sort(current_samples.begin(), current_samples.end());
      // Increment this for the next pass
      migration_index = (migration_index + 1) % MIGRATION_EPOCHS;
      std::vector<std::pair<AddressSpaceID,unsigned> > &next_samples = 
        user_samples[migration_index];
      if (MIGRATION_EPOCHS > 1)
      {
        // Compute the summary from all the epochs into the epoch
        // that we are about to clear
        std::map<AddressSpaceID,unsigned> summary(
            next_samples.begin(), next_samples.end());
        for (unsigned idx = 1; idx < MIGRATION_EPOCHS; idx++)
        {
          const std::vector<std::pair<AddressSpaceID,unsigned> > &other_samples
            = user_samples[(migration_index + idx) % MIGRATION_EPOCHS];
          for (std::vector<std::pair<AddressSpaceID,unsigned> >::const_iterator
                it = other_samples.begin(); it != other_samples.end(); it++)
          {
            std::map<AddressSpaceID,unsigned>::iterator finder = 
              summary.find(it->first);
            if (finder == summary.end())
              summary.insert(*it);
            else
              finder->second += it->second;
          }
        }
        next_samples.clear();
        next_samples.insert(next_samples.begin(),summary.begin(),summary.end());
      }
      AddressSpaceID new_logical_owner = logical_owner_space;
#ifdef DEBUG_LEGION
      assert(!next_samples.empty());
#endif
      if (next_samples.size() > 1)
      {
        int logical_owner_count = -1;
        // Figure out which node(s) has/have the most uses 
        // Make sure that the current owner node is sticky
        // if it is tied for the most uses
        unsigned max_count = next_samples[0].second; 
        AddressSpaceID max_user = next_samples[0].first;
        for (unsigned idx = 1; idx < next_samples.size(); idx++)
        {
          const AddressSpaceID user = next_samples[idx].first;
          const unsigned user_count = next_samples[idx].second;
          if (user == logical_owner_space)
            logical_owner_count = user_count;
          if (user_count < max_count)
            continue;
          // This is the part where we guarantee stickiness
          if ((user_count == max_count) && (user != logical_owner_space))
            continue;
          max_count = user_count;
          max_user = user;
        }
        if (logical_owner_count > 0)
        {
          if (logical_owner_space != max_user)
          {
            // If the logical owner is one of the current users then
            // we really better have a good reason to move this 
            // equivalence set to a new node. For now the difference 
            // between max_count and the current owner count has to
            // be greater than the number of nodes that we see participating
            // on this equivalence set. This heuristic should avoid 
            // the ping-pong case even when our sampling rate does not
            // naturally align with the number of nodes participating
            if ((max_count - unsigned(logical_owner_count)) >
                next_samples.size()) 
              new_logical_owner = max_user;
          }
        }
        else
          // If we didn't have the current logical owner then
          // just pick the maximum one
          new_logical_owner = max_user;
      }
      else
        // If all the requests came from the same node, send it there
        new_logical_owner = next_samples[0].first;
      // This always get reset here
      sample_count = 0;
      // Reset this for the next iteration
      next_samples.clear();
      // See if we are actually going to do the migration
      if (logical_owner_space == new_logical_owner)
      {
        // No need to do the migration in this case
        // Check to see if the request bounced off a stale owner 
        // and we should send the update message
        if ((eq_source != analysis.previous) && (eq_source != local_space) &&
            (eq_source != logical_owner_space))
        {
          RtUserEvent notification_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(logical_owner_space);
            rez.serialize(notification_event);
          }
          runtime->send_equivalence_set_owner_update(eq_source, rez);
          applied_events.insert(notification_event);
        }
        return;
      }
      // At this point we've decided to do the migration
      log_migration.info("Migrating Equivalence Set %llx from %d to %d",
          did, local_space, new_logical_owner);
      logical_owner_space = new_logical_owner;
      const FieldMask all_ones(LEGION_FIELD_MASK_FIELD_ALL_ONES);
      // Do the migration
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        pack_state(rez, logical_owner_space, set_expr, true/*covers*/,
                    all_ones, true/*pack guards*/);
      }
      runtime->send_equivalence_set_migration(logical_owner_space, rez);
      invalidate_state(set_expr, true/*covers*/, all_ones);
#endif // DISABLE_EQUIVALENCE_SET MIGRATION
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::defer_traversal(AutoTryLock &eq,
                                         PhysicalAnalysis &analysis,
                                         const FieldMask &mask,
                                         std::set<RtEvent> &deferral_events,
                                         std::set<RtEvent> &applied_events,
                                         const bool already_deferred)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!eq.has_lock());
#endif
      // See if we've already deferred this or not
      if (!already_deferred)
      {
        const RtUserEvent deferral_event = Runtime::create_rt_user_event();
        const RtEvent precondition = chain_deferral_events(deferral_event);
        analysis.defer_traversal(precondition, this, mask, deferral_events, 
                                 applied_events, deferral_event);
      }
      else
        analysis.defer_traversal(eq.try_next(), this, mask, deferral_events, 
                                 applied_events);
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::is_remote_analysis(PhysicalAnalysis &analysis,
                      const FieldMask &mask, std::set<RtEvent> &deferral_events,
                      std::set<RtEvent> &applied_events, const bool exclusive,
                      const bool immutable)
    //--------------------------------------------------------------------------
    {
      // Check to see if the analysis is replicated or not
      if (analysis.is_replicated())
      {
#ifdef DEBUG_LEGION
        assert(!immutable);
#endif
        CollectiveMapping *mapping = analysis.get_replicated_mapping();
        FieldMask replicated;
        std::set<RtEvent> exclusive_deferral_events;
        // Scan through each of our replicated states and see if they align
        if (!replicated_states.empty() && 
            !(replicated_states.get_valid_mask() * mask))
        {
          std::vector<CollectiveMapping*> to_delete;
          for (FieldMaskSet<CollectiveMapping>::iterator it =
                replicated_states.begin(); it != replicated_states.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            // Check to see if they match or not
            // If they match then we don't need to do anything
            // If they don't match then we need to deal with that
            if ((it->first != mapping) && ((*it->first) != (*mapping)))
            {
              // Mappings are not the same so see if we can make them the same
              // If we're exclusive we'll be able to modify them later
              if (!exclusive)
              {
                // Not exclusive so we can't change them
                // If we're the owner, we send out invalidations
                if (is_logical_owner())
                {
                  broadcast_replicated_state_updates(overlap, NULL, local_space,
                          applied_events, false/*need lock*/, false/*updates*/);
                  it.filter(overlap);
                  if (!it->second)
                    to_delete.push_back(it->first);
                }
                else
                  // record that we are being sent to the owner
                  analysis.record_remote(this, overlap, logical_owner_space);
              }
            }
            else
            {
              // These fields are replicated the right way so nothing to do
              replicated |= overlap;
              if (replicated == mask)
                break;
            }
          }
          for (std::vector<CollectiveMapping*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            replicated_states.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
          }
          if (!replicated_states.empty())
            replicated_states.tighten_valid_mask();
        }
        if (replicated != mask)
        {
          // If we have any unreplicated fields then check to see if
          // we can attempt to change the replicated mapping
          if (!exclusive)
          {
            // We can't change the mapping
            if (!is_logical_owner())
            {
              // Defer anything replicated
              // Send to the logical owner anything else
              if (!!replicated)
              {
                analysis.defer_traversal(RtEvent::NO_RT_EVENT, this,
                    replicated, deferral_events, applied_events);
                const FieldMask unreplicated = mask - replicated;
                analysis.record_remote(this, unreplicated, logical_owner_space);
              }
              else
                analysis.record_remote(this, mask, logical_owner_space); 
              return true;
            }
            // Otherwise we're on the logical owner so it doesn't
            // matter if we're replicated or not
          }
          else // we're exlcusive so we can change the mapping
          {
            const FieldMask unreplicated = mask - replicated;
            make_replicated_state(mapping, unreplicated, local_space,
                                  exclusive_deferral_events);
          }
        }
        // If we have any exclusive deferral events to defer the analysis
        // until we've been made replicated then do that now
        if (!exclusive_deferral_events.empty())
        {
          const RtEvent deferral_event = 
            Runtime::merge_events(exclusive_deferral_events);
          analysis.defer_traversal(deferral_event, this, mask,
                              deferral_events, applied_events);
          return true;
        }
        // We're all local at this point so continue the analysis
        return false;
      }
      else
      {
        // See if we are the logical owner or not
        if (is_logical_owner())
        {
          // See if we need to send any replicated invalidations
          if (!immutable && !replicated_states.empty() && 
              !(replicated_states.get_valid_mask() * mask))
          {
            // Send invalidations and record effects for when they are done
            std::vector<CollectiveMapping*> to_delete;
            for (FieldMaskSet<CollectiveMapping>::iterator it =
                 replicated_states.begin(); it != replicated_states.end(); it++)
            {
              const FieldMask overlap = it->second & mask;
              if (!overlap)
                continue;
              broadcast_replicated_state_updates(overlap, NULL, local_space,
                       applied_events, false/*need lock*/, false/*updates*/);
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            for (std::vector<CollectiveMapping*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              replicated_states.erase(*it);
              if ((*it)->remove_reference())
                delete (*it);
            }
            if (!replicated_states.empty())
              replicated_states.tighten_valid_mask();
          }
          // At this point everything is local and we're good to go
          return false;
        }
        else
        {
          if (immutable && !replicated_states.empty() &&
              !(replicated_states.get_valid_mask() * mask))
          {
            // We can do the traversal here for anything that's
            // replicated because we aren't mutating state
            FieldMask replicated;
            for (FieldMaskSet<CollectiveMapping>::const_iterator it =
                 replicated_states.begin(); it != replicated_states.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              // Can only traverse if it is local here
              if (it->first->contains(local_space))
                replicated |= overlap;
            }
            if (!!replicated)
            {
              if (replicated != mask)
              {
                // Defer all the replicated fields 
                analysis.defer_traversal(RtEvent::NO_RT_EVENT, this,
                    replicated, deferral_events, applied_events);
                // Send the unreplicated ones to the owner
                const FieldMask unreplicated = mask - replicated;
                analysis.record_remote(this, unreplicated, logical_owner_space);
                return true;
              }
              else // everything is local here, so we can traverse
                return false;
            }
            // Otherwise fall through and send it remote
          }
          // Not the logical owner, so just need to send it to the owner
          analysis.record_remote(this, mask, logical_owner_space);
          return true;
        }
      }
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::UpdateReplicatedFunctor::UpdateReplicatedFunctor(
                                  DistributedID id, const FieldMask &m,
                                  const CollectiveMapping *map,
                                  AddressSpaceID orig, AddressSpaceID skip,
                                  Runtime *rt, std::set<RtEvent> &ap)
      : did(id), mask(m), origin(orig), to_skip(skip), mapping(map), 
        runtime(rt), applied(ap)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::UpdateReplicatedFunctor::apply(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      if (target == to_skip)
        return;
      const RtUserEvent done_event = Runtime::create_rt_user_event();
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(mask);
        if (mapping != NULL)
          mapping->pack(rez);
        else
          rez.serialize<size_t>(0);
        rez.serialize(origin);
        rez.serialize(done_event);
      }
      runtime->send_equivalence_set_replication_update(target, rez);
      applied.insert(done_event);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::broadcast_replicated_state_updates(
                 const FieldMask &mask, CollectiveMapping *mapping,
                 const AddressSpaceID origin, std::set<RtEvent> &applied_events,
                 const bool need_lock, const bool perform_updates)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock eq(eq_lock);
        broadcast_replicated_state_updates(mask, mapping, origin, 
            applied_events, false/*need lock*/, perform_updates);
        return;
      }
      // If we're the owner, send messages to all the remote instances
      if (is_owner() && has_remote_instances())
      {
        UpdateReplicatedFunctor functor(did, mask, mapping, local_space, origin,
                                        runtime, applied_events);
        map_over_remote_instances(functor);
      }
      if (collective_mapping != NULL)
      {
        // Send it along to the other locations
        std::vector<AddressSpaceID> children;
        collective_mapping->get_children(origin, local_space, children);
        for (std::vector<AddressSpaceID>::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(mask);
            if (mapping != NULL)
              mapping->pack(rez);
            else
              rez.serialize<size_t>(0);
            rez.serialize(origin);
            rez.serialize(done_event);
          }
          runtime->send_equivalence_set_replication_update(*it, rez);
          applied_events.insert(done_event);
        }
      }
      else if (origin != owner_space)
      {
#ifdef DEBUG_LEGION
        assert(origin == local_space);
#endif
        // If the origin is not the owner space, send a message to it
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          if (mapping != NULL)
            mapping->pack(rez);
          else
            rez.serialize<size_t>(0);
          rez.serialize(origin);
          rez.serialize(done_event);
        }
        runtime->send_equivalence_set_replication_update(owner_space, rez);
        applied_events.insert(done_event);
      }
      // Once we get down here then we can just do our local updates 
      if (perform_updates)
        update_replicated_state(mapping, mask);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_replicated_state(CollectiveMapping *mapping,
                                                 const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Remove any conflicting mappings and record any local invalidations
      // we need to do here
      FieldMask invalidate_mask;
      if (!replicated_states.empty() && 
          !(mask * replicated_states.get_valid_mask()))
      {
        std::vector<CollectiveMapping*> to_delete;
        for (FieldMaskSet<CollectiveMapping>::iterator it =
              replicated_states.begin(); it != replicated_states.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          if (it->first->contains(local_space))
            invalidate_mask |= overlap;
          it.filter(overlap);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        for (std::vector<CollectiveMapping*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          replicated_states.erase(*it);
          if ((*it)->remove_reference())
            delete (*it);
        }
        if (!replicated_states.empty())
          replicated_states.tighten_valid_mask();
      }
      // If we're not included in the mapping then add it now, if we are 
      // included then we'll be added automatically by the requesters
      if (mapping != NULL)
      {
        if (mapping->contains(local_space))
        {
          // If the fields are still going to be valid in the new state
          // then we don't need to invalidate them
          if (!!invalidate_mask)
            invalidate_mask.clear();
        }
        else
        {
          if (replicated_states.insert(mapping, mask))
            mapping->add_reference();
        }
      }
      // If we don't have any local fields to invalidate then return
      if (!invalidate_mask)
        return;
      invalidate_state(set_expr, true/*covers*/, mask);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::PendingReplication::PendingReplication(CollectiveMapping *m,
                                                         unsigned notifications)
      : mapping(m), ready_event(Runtime::create_rt_user_event()),
        remaining_notifications(notifications)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping != NULL); 
#endif
      mapping->add_reference();
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::PendingReplication::~PendingReplication(void)
    //--------------------------------------------------------------------------
    {
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::make_replicated_state(CollectiveMapping *mapping,
                                    FieldMask mask, const AddressSpaceID source,
                                    std::set<RtEvent> &deferral_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping->contains(source));
#endif
      if (is_logical_owner())
      {
        // For the first notification, send any invalidations out to 
        // instances which are not contained in the new set
        std::vector<PendingReplication*> to_finalize;
        for (FieldMaskSet<PendingReplication>::iterator it =
              pending_states.begin(); it != pending_states.end(); it++)
        {
          if (mask * it->second)
            continue;
#ifdef DEBUG_LEGION
          // should overlap on all fields
          assert(!(it->second - mask));
          assert((*it->first->mapping) == *mapping);
#endif
          deferral_events.insert(it->first->ready_event);
          // Check to see if this is the last notification for this
          // pending replication state, if so we can finalize it
          if (--it->first->remaining_notifications == 0)
            to_finalize.push_back(it->first);
          mask -= it->second;
          if (!mask)
            break;
        }
        // Finalize any that are ready to be done
        if (!to_finalize.empty())
        {
          for (std::vector<PendingReplication*>::const_iterator it =
                to_finalize.begin(); it != to_finalize.end(); it++)
          {
            FieldMaskSet<PendingReplication>::iterator finder =
              pending_states.find(*it);
#ifdef DEBUG_LEGION
            assert(finder != pending_states.end());
#endif
            finalize_pending_replication(finder->first, finder->second,
                                         true/*first call*/);
            pending_states.erase(finder);
          }
          if (!pending_states.empty())
            pending_states.tighten_valid_mask();
        }
        // If we still have fields, then start a new pending request
        if (!!mask)
        {
          PendingReplication *pending = 
            new PendingReplication(mapping, mapping->size() - 1);
          // Send updates to all the nodes not in the mapping
          broadcast_replicated_state_updates(mask, mapping, local_space,
                                             pending->preconditions);
          pending_states.insert(pending, mask);
          deferral_events.insert(pending->ready_event);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(source == local_space);
#endif
        // First deduplicate requests
        for (FieldMaskSet<PendingReplication>::const_iterator it =
              pending_states.begin(); it != pending_states.end(); it++)
        {
          if (it->second * mask)
            continue;
#ifdef DEBUG_LEGION
          // should overlap on all fields
          assert(!(it->second - mask));
          assert((it->first->mapping == mapping) || 
                  ((*it->first->mapping) == *mapping));
#endif
          deferral_events.insert(it->first->ready_event);
          mask -= it->second;
          if (!mask)
            return;
        }
#ifdef DEBUG_LEGION
        assert(!!mask);
#endif
        // Then sending a notification to the logical owner with a
        // request for any valid fields that we need for this copy
        PendingReplication *pending = 
            new PendingReplication(mapping, 1/*notification from owner*/);
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          mapping->pack(rez);
          rez.serialize(pending);
          rez.serialize(local_space);
          rez.serialize(pending->ready_event);
        }
        runtime->send_equivalence_set_replication_request(logical_owner_space,
                                                          rez);
        deferral_events.insert(pending->ready_event);
        pending_states.insert(pending, mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_replication_request(const FieldMask &mask,
                         CollectiveMapping *mapping, PendingReplication *target,
                         const AddressSpaceID source, const RtEvent done_event)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're not the logical owner, keep forwarding this on until
      // we get to the logical owner
      if (!is_logical_owner())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(mask);
          mapping->pack(rez);
          rez.serialize(target);
          rez.serialize(source);
          rez.serialize(done_event);
        }
        runtime->send_equivalence_set_replication_request(logical_owner_space,
                                                          rez);
        return;
      }
      // Before we do anything else, go through and figure out what (if any)
      // update state we will need to pack to send to the new node
      // Do this before calling make_replicated_state which is when the
      // replicated_state data structure can change
      FieldMask update_mask = mask;
      if (!(mask * replicated_states.get_valid_mask()))
      {
        for (FieldMaskSet<CollectiveMapping>::const_iterator it =
              replicated_states.begin(); it != replicated_states.end(); it++)
        {
          if (update_mask * it->second)
            continue;
          if (!it->first->contains(source))
            continue;
          update_mask -= it->second;
          if (!update_mask)
            break;
        }
      }
      std::set<RtEvent> deferral_events;
      make_replicated_state(mapping, mask, source, deferral_events);
      RtEvent ready_event;
      if (!deferral_events.empty())
        ready_event = Runtime::merge_events(deferral_events);
      // Then send the response back to the source
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(mask);
        rez.serialize(target);
        rez.serialize(ready_event);
        rez.serialize(update_mask);
        if (!!update_mask)
        {
          rez.serialize(local_space);
          pack_state(rez, source, set_expr, true/*covers*/, mask, 
                      true/*pack guards*/); 
        }
      }
      runtime->send_equivalence_set_replication_response(source, rez);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_replication_response(
        PendingReplication *target, const FieldMask &mask, RtEvent precondition,
        const FieldMask &update_mask, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> unpacked_events;
      if (!!update_mask)
      {
        AddressSpaceID source;
        derez.deserialize(source);
        unpack_state_and_apply(derez, source, false/*forward*/,unpacked_events);
      }
      AutoLock eq(eq_lock);
      if (!unpacked_events.empty())
        target->preconditions.insert(unpacked_events.begin(), 
                                     unpacked_events.end());
      // Then finalize our replication
      if (precondition.exists())
        target->preconditions.insert(precondition);
      finalize_pending_replication(target, mask, true/*first*/);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DeferPendingReplicationArgs::DeferPendingReplicationArgs(
                   EquivalenceSet *s, PendingReplication *p, const FieldMask &m)
      : LgTaskArgs<DeferPendingReplicationArgs>(implicit_provenance),
        set(s), pending(p), mask(new FieldMask(m))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::finalize_pending_replication(
                             PendingReplication *pending, const FieldMask &mask,
                             const bool first, const bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock eq(eq_lock);
        finalize_pending_replication(pending, mask, first, false/*need lock*/);
        return;
      }
      if (first)
      {
        if (!pending->preconditions.empty())
          Runtime::trigger_event(pending->ready_event,
              Runtime::merge_events(pending->preconditions));
        else
          Runtime::trigger_event(pending->ready_event);
        if (!pending->ready_event.has_triggered())
        {
          // Need to defer adding this until it is ready
          DeferPendingReplicationArgs args(this, pending, mask);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, pending->ready_event);
          return;
        }
      }
#ifdef DEBUG
      assert(mask * replicated_states.get_valid_mask());
#endif
      if (replicated_states.insert(pending->mapping, mask))
        pending->mapping->add_reference();
      delete pending;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_pending_replication(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferPendingReplicationArgs *dargs = 
        (const DeferPendingReplicationArgs*)args;
      dargs->set->finalize_pending_replication(dargs->pending, *(dargs->mask),
          false/*first*/, true/*need lock*/);
      delete (dargs->mask);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::check_for_uninitialized_data(T &analysis, 
                      IndexSpaceExpression *expr, const bool expr_covers,
                      FieldMask uninit, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      // Do the easy check for the full cover which will be the common case
      FieldMaskSet<IndexSpaceExpression>::const_iterator finder =
        initialized_data.find(set_expr);
      if (finder != initialized_data.end())
      {
        uninit -= finder->second;
        if (!uninit)
          return;
      }
      if (!expr_covers)
      {
        finder = initialized_data.find(expr);
        if (finder != initialized_data.end())
        {
          uninit -= finder->second;
          if (!uninit)
            return;
        }
      }
      // All the rest of these are partial so only test them if 
      // expr_covers is false because we know they aren't covered otherwise
      if (!expr_covers)
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              initialized_data.begin(); it != initialized_data.end(); it++)
        {
          if (uninit * it->second)
            continue;
          // Don't actually need to subtract here since we don't care
          // about the difference size, just care about domination
          IndexSpaceExpression *overlap_expr = 
            runtime->forest->intersect_index_spaces(it->first, expr);
          if (overlap_expr->get_volume() != expr->get_volume())
            continue;
          uninit -= it->second;
          if (!uninit)
            return;
        }
      }
      // Record anything that we have left
      analysis.record_uninitialized(uninit, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_initialized_data(IndexSpaceExpression *expr,
                                                 const bool expr_covers,
                                                 const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      if (!expr_covers)
      {
        FieldMask subinit = user_mask;
        FieldMaskSet<IndexSpaceExpression>::iterator finder =
          initialized_data.find(set_expr);
        // Check to see if we've already initialized it for the full set_expr
        if (finder != initialized_data.end())
        {
          subinit -= finder->second;
          // Already initialized for full expression so we are done
          if (!subinit)
            return;
        }
        FieldMaskSet<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
              initialized_data.begin(); it != initialized_data.end(); it++)
        {
          if ((it->first == set_expr) || (it->first == expr))
            continue;
          const FieldMask overlap = subinit & it->second;
          if (!overlap)
            continue;
          // Compute the union expression
          IndexSpaceExpression *union_expr = 
            runtime->forest->union_index_spaces(it->first, expr);
          const size_t union_size = union_expr->get_volume();
#ifdef DEBUG_LEGION
          assert(union_size <= set_expr->get_volume());
#endif
          if (union_size == it->first->get_volume())
          {
            // Existing expression already covers expr
            subinit -= overlap;
            if (!subinit)
              break;
          }
          else if (union_size == set_expr->get_volume())
          {
            // Union is the same as the set expression
            if (finder != initialized_data.end())
              finder.merge(overlap);
            else
              to_add.insert(set_expr, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            subinit -= overlap;
            if (!subinit)
              break;
          }
          else if (union_size == expr->get_volume())
          {
            // New expression covers the old expression
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          else
          {
            // Union is bigger than both expression but not set_expr
            to_add.insert(union_expr, overlap);
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            subinit -= overlap;
            if (!subinit)
              break;
          }
        }
        // Add new ones
        if (!to_add.empty())
        {
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (initialized_data.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
        }
        // Delete after adding to keep expressions valid 
        if (!to_delete.empty())
        {
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (to_add.find(*it) != to_add.end())
              continue;
            initialized_data.erase(*it);
            if ((*it)->remove_nested_expression_reference(did))
              delete (*it);
          }
        }
        // Add the new expression if we still have fields to add
        if (!!subinit && initialized_data.insert(expr, subinit))
          expr->add_nested_expression_reference(did);
      }
      else
      {
        // Remove all other expressions with overlapping fields
        if (!(user_mask * initialized_data.get_valid_mask()))
        {
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
                initialized_data.begin(); it != initialized_data.end(); it++)
          {
            if (it->first == set_expr)
              continue;
            it.filter(user_mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              initialized_data.erase(*it);
              if ((*it)->remove_nested_expression_reference(did))
                delete (*it);
            }
          }
        }
        if (initialized_data.insert(set_expr, user_mask))
          set_expr->add_nested_expression_reference(did);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::record_instances(IndexSpaceExpression *expr,
                                          const bool expr_covers,
                                          const FieldMask &record_mask,
                                          const FieldMaskSet<T> &target_insts)
    //--------------------------------------------------------------------------
    {
      bool rebuild_partial = false;
      if (expr_covers)
      {
        if (!(target_insts.get_valid_mask() - record_mask))
        {
          for (typename FieldMaskSet<T>::const_iterator it =
                target_insts.begin(); it != target_insts.end(); it++)
          {
            if (total_valid_instances.insert(it->first, it->second))
              it->first->add_nested_valid_ref(did);
            // Check to see if there are any copies of this to filter
            // from the partially valid instances
            ViewExprMaskSets::iterator finder =
              partial_valid_instances.find(it->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * it->second))
            {
              rebuild_partial = true;
              if (!(finder->second.get_valid_mask() - it->second))
              {
                // Filter out the ones we now subsume
                std::vector<IndexSpaceExpression*> to_delete;    
                for (FieldMaskSet<IndexSpaceExpression>::iterator eit =
                     finder->second.begin(); eit != finder->second.end(); eit++)
                {
                  eit.filter(it->second);
                  if (!eit->second)
                    to_delete.push_back(eit->first);
                }
                for (std::vector<IndexSpaceExpression*>::const_iterator eit =
                      to_delete.begin(); eit != to_delete.end(); eit++)
                {
                  finder->second.erase(*eit);
                  if ((*eit)->remove_nested_expression_reference(did))
                    delete (*eit);
                }
                if (finder->second.empty())
                {
                  // Remove the reference, no need to check for deletion
                  // since we know we added the same reference above
                  finder->first->remove_nested_valid_ref(did);
                  partial_valid_instances.erase(finder);
                }
                else
                  finder->second.tighten_valid_mask();
              }
              else
              {
                // We're pruning everything so remove them all now
                for (FieldMaskSet<IndexSpaceExpression>::const_iterator eit =
                     finder->second.begin(); eit != finder->second.end(); eit++)
                  if (eit->first->remove_nested_expression_reference(did))
                    delete eit->first;
                // Remove the reference, no need to check for deletion
                // since we know we added the same reference above
                finder->first->remove_nested_valid_ref(did);
                partial_valid_instances.erase(finder);
              }
            }
          } 
        }
        else
        {
          for (typename FieldMaskSet<T>::const_iterator it =
                target_insts.begin(); it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & record_mask; 
            if (!valid_mask)
              continue;
            // Add it to the set
            if (total_valid_instances.insert(it->first, valid_mask))
              it->first->add_nested_valid_ref(did);
            // Check to see if there are any copies of this to filter
            // from the partially valid instances
            ViewExprMaskSets::iterator finder = 
              partial_valid_instances.find(it->first);
            if ((finder != partial_valid_instances.end()) &&
                !(finder->second.get_valid_mask() * valid_mask))
            {
              rebuild_partial = true;
              if (!(finder->second.get_valid_mask() - valid_mask))
              {
                // Filter out the ones we now subsume
                std::vector<IndexSpaceExpression*> to_delete;    
                for (FieldMaskSet<IndexSpaceExpression>::iterator eit =
                     finder->second.begin(); eit != finder->second.end(); eit++)
                {
                  eit.filter(valid_mask);
                  if (!eit->second)
                    to_delete.push_back(eit->first);
                }
                for (std::vector<IndexSpaceExpression*>::const_iterator eit =
                      to_delete.begin(); eit != to_delete.end(); eit++)
                {
                  finder->second.erase(*eit);
                  if ((*eit)->remove_nested_expression_reference(did))
                    delete (*eit);
                }
                if (finder->second.empty())
                {
                  // Remove the reference, no need to check for deletion
                  // since we know we added the same reference above
                  finder->first->remove_nested_valid_ref(did);
                  partial_valid_instances.erase(finder);
                }
                else
                  finder->second.tighten_valid_mask();
              }
              else
              {
                // We're pruning everything so remove them all now
                for (FieldMaskSet<IndexSpaceExpression>::const_iterator eit =
                     finder->second.begin(); eit != finder->second.end(); eit++)
                  if (eit->first->remove_nested_expression_reference(did))
                    delete eit->first;
                // Remove the reference, no need to check for deletion
                // since we know we added the same reference above
                finder->first->remove_nested_valid_ref(did);
                partial_valid_instances.erase(finder);
              }
            }
          }
        }
      }
      else
      {
        if (!(target_insts.get_valid_mask() - record_mask))
        {
          for (typename FieldMaskSet<T>::const_iterator it =
                target_insts.begin(); it != target_insts.end(); it++)
            if (record_partial_valid_instance(it->first, expr,
                                              it->second))
              rebuild_partial = true;
        }
        else
        {
          for (typename FieldMaskSet<T>::const_iterator it =
                target_insts.begin(); it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & record_mask; 
            if (!valid_mask)
              continue;
            if (record_partial_valid_instance(it->first, expr,
                                              valid_mask))
              rebuild_partial = true;
          }
        }
      }
      if (rebuild_partial)
      {
        partial_valid_fields.clear();
        for (ViewExprMaskSets::const_iterator it =
              partial_valid_instances.begin(); it !=
              partial_valid_instances.end(); it++)
          partial_valid_fields |= it->second.get_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::record_unrestricted_instances(
                                          IndexSpaceExpression *expr,
                                          const bool expr_covers,
                                          FieldMask record_mask,
                                          const FieldMaskSet<T> &target_insts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!(record_mask - restricted_fields));
#endif
      // Check to see if there are any restrictions which cover the whole
      // set and therefore we know that there are on partial coverings
      ExprViewMaskSets::const_iterator finder =
        restricted_instances.find(set_expr);
      if (finder != restricted_instances.end())
      {
        record_mask -= finder->second.get_valid_mask();
        if (!record_mask)
          return;
      }
      // The only fields left here are the partial restrictions
      FieldMaskSet<IndexSpaceExpression> restrictions;
      for (ExprViewMaskSets::const_iterator it = restricted_instances.begin();
            it != restricted_instances.end(); it++)
      {
        if (it == finder)
          continue;
        const FieldMask overlap = it->second.get_valid_mask() & record_mask;
        if (!overlap)
          continue;
        if (!expr_covers)
        {
          IndexSpaceExpression *overlap_expr =
            runtime->forest->intersect_index_spaces(expr, it->first);
          if (!overlap_expr->is_empty())
            restrictions.insert(overlap_expr, overlap);
        }
        else
          restrictions.insert(it->first, overlap);
      }
      // Sort these into grouped field sets so we can union them before
      // doing the subtraction to figure out what we can record
      LegionList<FieldSet<IndexSpaceExpression*> > restricted_sets;
      restrictions.compute_field_sets(record_mask, restricted_sets);
      bool need_partial_rebuild = false;
      for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
            rit = restricted_sets.begin(); rit != restricted_sets.end(); rit++)
      {
        IndexSpaceExpression *diff_expr = NULL;
        if (!rit->elements.empty())
        {
          IndexSpaceExpression *union_expr = 
            runtime->forest->union_index_spaces(rit->elements);
          diff_expr = runtime->forest->subtract_index_spaces(expr, union_expr);
        }
        else
          diff_expr = expr;
        if (!diff_expr->is_empty())
        {
          for (typename FieldMaskSet<T>::const_iterator it =
                target_insts.begin(); it != target_insts.end(); it++)
          {
            const FieldMask valid_mask = it->second & rit->set_mask; 
            if (!valid_mask)
              continue;
            if (record_partial_valid_instance(it->first, diff_expr, 
                                              valid_mask))
              need_partial_rebuild = true;
          }
        }
#ifdef DEBUG_LEGION
        record_mask -= rit->set_mask;
#endif
      }
#ifdef DEBUG_LEGION
      assert(!record_mask);
#endif
      if (need_partial_rebuild)
      {
        partial_valid_fields.clear();
        for (ViewExprMaskSets::const_iterator it = 
              partial_valid_instances.begin();  it !=
              partial_valid_instances.end(); it++)
          partial_valid_fields |= it->second.get_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::record_partial_valid_instance(LogicalView *target,
                              IndexSpaceExpression *expr, FieldMask valid_mask, 
                              bool check_total_valid)
    //--------------------------------------------------------------------------
    {
      bool need_rebuild = false;
      if (check_total_valid)
      {
        FieldMaskSet<LogicalView>::const_iterator finder = 
          total_valid_instances.find(target);
        if (finder != total_valid_instances.end())
        {
          valid_mask -= finder->second;
          if (!valid_mask)
            return need_rebuild;
        }
      }
      partial_valid_fields |= valid_mask;
      ViewExprMaskSets::iterator finder = partial_valid_instances.find(target);
      if (finder != partial_valid_instances.end())
      {
        // See if we have any overlapping field expressions to add this to 
        if (!(valid_mask * finder->second.get_valid_mask()))
        {
          std::vector<IndexSpaceExpression*> to_delete;
          FieldMaskSet<IndexSpaceExpression> to_add;
          bool need_tighten = false;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                finder->second.begin(); it != finder->second.end(); it++)
          {
            const FieldMask overlap = it->second & valid_mask;
            if (!overlap)
              continue;
            IndexSpaceExpression *union_expr = 
              runtime->forest->union_index_spaces(it->first, expr);
            const size_t union_size = union_expr->get_volume();
#ifdef DEBUG_LEGION
            assert(union_size <= set_expr->get_volume());
#endif
            if (union_size == set_expr->get_volume())
            {
              // Hurray, we now cover the full expr so we can get
              // promoted up to the total valid instances
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              if (total_valid_instances.insert(target, overlap))
                target->add_nested_valid_ref(did);
              need_tighten = true;
            }
            else if (union_size == expr->get_volume())
            {
              // We dominate the previous expression, so remove it
              // and put ourselves in
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              to_add.insert(expr, overlap);
            }
            else if (union_size > it->first->get_volume())
            {
              // Union dominates both so put it in instead
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              to_add.insert(union_expr, overlap);
            }
            // Else previous expr dominates so we can just leave it there
            valid_mask -= overlap;
            if (!valid_mask)
              break;
          }
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (to_add.find(*it) != to_add.end())
              continue;
            finder->second.erase(*it);
            if ((*it)->remove_nested_expression_reference(did))
              delete (*it);
          }
          if (!!valid_mask && finder->second.insert(expr, valid_mask))
            expr->add_nested_expression_reference(did);
          if (need_tighten)
          {
            if (finder->second.empty())
            {
              // Wow! everything got promoted up to total valid
              // instances, lucky us, remove the old partial stuff
              finder->first->remove_nested_valid_ref(did);
              partial_valid_instances.erase(finder);
            }
            else
              finder->second.tighten_valid_mask();
            if (!partial_valid_instances.empty())
              need_rebuild = true;
            else
              partial_valid_fields.clear();
          }
        }
        else if (finder->second.insert(expr, valid_mask))
          expr->add_nested_expression_reference(did);
      }
      else
      {
        partial_valid_instances[target].insert(expr, valid_mask);
        target->add_nested_valid_ref(did);
        expr->add_nested_expression_reference(did);
      }
      return need_rebuild;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_valid_instances(IndexSpaceExpression *expr,
                  const bool expr_covers, const FieldMask &filter_mask,
                  std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
                  std::map<LogicalView*,unsigned> *view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!filter_mask);
#endif
      if (expr_covers)
      {
        // If the expr covers we can just filter everything
        if (!(filter_mask * total_valid_instances.get_valid_mask()))
        {
          // Clear out the total valid instances first
          std::vector<LogicalView*> to_delete;
          for (FieldMaskSet<LogicalView>::iterator it = 
                total_valid_instances.begin(); it != 
                total_valid_instances.end(); it++)
          {
            const FieldMask overlap = it->second & filter_mask;
            if (!overlap)
              continue;
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<LogicalView*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              total_valid_instances.erase(*it);
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder = 
                  view_refs_to_remove->find(*it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[*it] = 1;
                else
                  finder->second += 1;
              }
              else if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            }
          }
        }
        if (!(filter_mask * partial_valid_fields))
        {
          // Then clear out the partial valid instances
          std::vector<LogicalView*> to_delete;
          for (ViewExprMaskSets::iterator pit =
                partial_valid_instances.begin(); pit != 
                partial_valid_instances.end(); pit++)
          {
            const FieldMask &summary_mask = pit->second.get_valid_mask();
            if (summary_mask * filter_mask)
              continue;
            else if (!(summary_mask - filter_mask))
            {
              // Invalidating all the expressions
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    pit->second.begin(); it != pit->second.end(); it++)
              {
                if (expr_refs_to_remove != NULL)
                {
                  std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                    expr_refs_to_remove->find(it->first);
                  if (finder == expr_refs_to_remove->end())
                    (*expr_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_expression_reference(did))
                  delete it->first;
              }
              to_delete.push_back(pit->first);
            }
            else
            {
              // Only invalidating some of the expressions
              std::vector<IndexSpaceExpression*> to_erase;
              for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                    pit->second.begin(); it != pit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              if (!to_erase.empty())
              {
                for (std::vector<IndexSpaceExpression*>::const_iterator it = 
                      to_erase.begin(); it != to_erase.end(); it++)
                {
                  pit->second.erase(*it);
                  if (expr_refs_to_remove != NULL)
                  {
                    std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                      expr_refs_to_remove->find(*it);
                    if (finder == expr_refs_to_remove->end())
                      (*expr_refs_to_remove)[*it] = 1;
                    else
                      finder->second += 1;
                  }
                  else if ((*it)->remove_nested_expression_reference(did))
                    delete (*it);
                }
              }
              pit->second.tighten_valid_mask();
            }
          }
          if (!to_delete.empty())
          {
            for (std::vector<LogicalView*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              partial_valid_instances.erase(*it);
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder = 
                  view_refs_to_remove->find(*it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[*it] = 1;
                else
                  finder->second += 1;
              }
              else if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            }
          }
          partial_valid_fields -= filter_mask;
        }
      }
      else
      {
        // If the expr does not cover then we have to do partial filtering
        // Filter any partial data first
        if (!(filter_mask * partial_valid_fields))
        {
          std::vector<LogicalView*> to_delete;
          FieldMask still_partial_valid;
          for (ViewExprMaskSets::iterator pit =
                partial_valid_instances.begin(); pit != 
                partial_valid_instances.end(); pit++)
          {
            FieldMask view_overlap = pit->second.get_valid_mask() & filter_mask;
            if (!view_overlap)
              continue;
            std::vector<IndexSpaceExpression*> to_erase;
            FieldMaskSet<IndexSpaceExpression> to_add;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                  pit->second.begin(); it != pit->second.end(); it++)
            {
              const FieldMask overlap = it->second & view_overlap;
              if (!overlap)
                continue;
              IndexSpaceExpression *diff = 
                runtime->forest->subtract_index_spaces(it->first, expr);
              if (diff->is_empty())
              {
                // filter expr covers, so remove it
                it.filter(overlap);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              else if (diff->get_volume() < it->first->get_volume())
              {
                // filter expr covers some, so these fields and make
                // the diff the new expression for these overlap fields
                it.filter(overlap);
                if (!it->second)
                  to_erase.push_back(it->first);
                to_add.insert(diff, overlap);
              }
              // else expr does not cover any so nothing to do here
              view_overlap -= overlap;
              if (!view_overlap)
                break;
            } 
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  to_add.begin(); it != to_add.end(); it++)
              if (pit->second.insert(it->first, it->second))
                it->first->add_nested_expression_reference(did);
            // Deletions after adding to make sure to keep referenes around
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  to_erase.begin(); it != to_erase.end(); it++)
            {
              // Don't delete if we just added it
              if (to_add.find(*it) != to_add.end())
                continue;
              FieldMaskSet<IndexSpaceExpression>::iterator finder =
                pit->second.find(*it);
#ifdef DEBUG_LEGION
              assert(finder != pit->second.end());
#endif
              if (!!finder->second)
                continue;
              pit->second.erase(finder);
              if (expr_refs_to_remove != NULL)
              {
                std::map<IndexSpaceExpression*,unsigned>::iterator finder2 =
                  expr_refs_to_remove->find(*it);
                if (finder2 == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[*it] = 1;
                else
                  finder2->second += 1;
              }
              else if ((*it)->remove_nested_expression_reference(did))
                delete (*it);
            }
            if (!pit->second.empty())
            {
              pit->second.tighten_valid_mask();
              still_partial_valid |= pit->second.get_valid_mask();
            }
            else
              to_delete.push_back(pit->first);
          }
          for (std::vector<LogicalView*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            partial_valid_instances.erase(*it);
            if (view_refs_to_remove != NULL)
            {
              std::map<LogicalView*,unsigned>::iterator finder = 
                view_refs_to_remove->find(*it);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
          }
          partial_valid_fields -= (filter_mask - still_partial_valid);
        }
        // Now we can filter the total valid instances back to the
        // partial valid instances
        if (!(filter_mask * total_valid_instances.get_valid_mask()))
        {
          std::vector<LogicalView*> to_delete;
          bool need_partial_rebuild = false;
          IndexSpaceExpression *diff_expr = NULL;
          for (FieldMaskSet<LogicalView>::iterator it =
                total_valid_instances.begin(); it != 
                total_valid_instances.end(); it++)
          {
            const FieldMask overlap = filter_mask & it->second;
            if (!overlap)
              continue;
            if (diff_expr == NULL)
            {
              diff_expr = runtime->forest->subtract_index_spaces(set_expr,expr);
#ifdef DEBUG_LEGION
              assert(!diff_expr->is_empty());
#endif
            }
            if (record_partial_valid_instance(it->first, diff_expr, 
                      overlap, false/*check total valid*/)) 
              need_partial_rebuild = true;
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          for (std::vector<LogicalView*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (view_refs_to_remove != NULL)
            {
              std::map<LogicalView*,unsigned>::iterator finder = 
                view_refs_to_remove->find(*it);
              if (finder == view_refs_to_remove->end())
                (*view_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
            total_valid_instances.erase(*it);
          }
          total_valid_instances.tighten_valid_mask();
          if (need_partial_rebuild)
          {
            partial_valid_fields.clear();
            for (ViewExprMaskSets::const_iterator it =
                  partial_valid_instances.begin(); it !=
                  partial_valid_instances.end(); it++)
              partial_valid_fields |= it->second.get_valid_mask();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_unrestricted_instances(
                             IndexSpaceExpression *expr, const bool expr_covers,
                             FieldMask filter_mask)
    //--------------------------------------------------------------------------
    {
      // Compute the expressions and field masks which are not restricted    
      // First remove any fields which are restricted for this set
      ExprViewMaskSets::const_iterator finder =
        restricted_instances.find(expr);
      if (finder != restricted_instances.end())
      {
        filter_mask -= finder->second.get_valid_mask();
        if (!filter_mask)
          return;
      }
      // Next see if there any full set restrictions which dominate our expr
      if (expr != set_expr)
      {
        finder = restricted_instances.find(set_expr);
        if (finder != restricted_instances.end())
        {
          filter_mask -= finder->second.get_valid_mask();
          if (!filter_mask)
            return;
        }
      }
      // If we're still here, then we now have to do the hard part of
      // computing the intefering expression sets
      FieldMaskSet<IndexSpaceExpression> restricted_sets;
      for (ExprViewMaskSets::const_iterator it = restricted_instances.begin();
            it != restricted_instances.end(); it++)
      {
        const FieldMask overlap = it->second.get_valid_mask() & filter_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression *expr_overlap = 
          runtime->forest->intersect_index_spaces(it->first, expr);
        if (expr_overlap->is_empty())
          continue;
        if (expr_overlap->get_volume() == expr->get_volume())
        {
          // If this expression dominates the expr we are done
          filter_mask -= overlap;
          if (!filter_mask)
            return;
        }
        restricted_sets.insert(expr_overlap, overlap);
      }
#ifdef DEBUG_LEGION
      assert(!!filter_mask);
#endif
      // compute the field sets and take the field differences
      LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
      restricted_sets.compute_field_sets(filter_mask, field_sets);
      for (LegionList<FieldSet<IndexSpaceExpression*> >::iterator it =
            field_sets.begin(); it != field_sets.end(); it++)
      {
        if (it->elements.empty())
        {
          filter_valid_instances(expr, expr_covers, it->set_mask);
          continue;
        }
        IndexSpaceExpression *union_expr =
          runtime->forest->union_index_spaces(it->elements);
        IndexSpaceExpression *diff_expr =
          runtime->forest->subtract_index_spaces(expr, union_expr);
        if (!diff_expr->is_empty())
          filter_valid_instances(diff_expr, false/*covers*/, it->set_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_reduction_instances(IndexSpaceExpression *expr,
                  const bool expr_covers, const FieldMask &filter_mask,
                  std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
                  std::map<LogicalView*,unsigned> *view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      int fidx = filter_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::list<
          std::pair<ReductionView*,IndexSpaceExpression*> > >::iterator
          finder = reduction_instances.find(fidx);
        if (finder != reduction_instances.end())
        {
          if (expr_covers)
          {
            for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
                  const_iterator it = finder->second.begin(); it !=
                  finder->second.end(); it++)
            {
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder = 
                  view_refs_to_remove->find(it->first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it->first] = 1;
                else
                  finder->second += 1;
              }
              else if (it->first->remove_nested_valid_ref(did))
                delete it->first;
              if (expr_refs_to_remove != NULL)
              {
                std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                  expr_refs_to_remove->find(it->second);
                if (finder == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[it->second] = 1;
                else
                  finder->second += 1;
              }
              else if (it->second->remove_nested_expression_reference(did))
                delete it->second;
            }
            reduction_instances.erase(finder);
            reduction_fields.unset_bit(fidx);
          }
          else
          {
            IndexSpaceExpression *full_diff = NULL;
            for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
                  iterator it = finder->second.begin(); it != 
                  finder->second.end(); /*nothing*/)
            {
              if (it->second == set_expr)
              {
                if (full_diff == NULL)
                {
                  full_diff = 
                    runtime->forest->subtract_index_spaces(set_expr, expr);
#ifdef DEBUG_LEGION
                  assert(!full_diff->is_empty());
#endif
                }
                full_diff->add_nested_expression_reference(did);
                if (expr_refs_to_remove != NULL)
                {
                  std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                    expr_refs_to_remove->find(it->second);
                  if (finder == expr_refs_to_remove->end())
                    (*expr_refs_to_remove)[it->second] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->second->remove_nested_expression_reference(did))
                  delete it->second;
                it->second = full_diff;
                it++;
              }
              else
              {
                IndexSpaceExpression *diff_expr = 
                  runtime->forest->subtract_index_spaces(it->second, expr);
                if (!diff_expr->is_empty())
                {
                  if (diff_expr->get_volume() < it->second->get_volume())
                  {
                    diff_expr->add_nested_expression_reference(did);
                    if (expr_refs_to_remove != NULL)
                    {
                      std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                        expr_refs_to_remove->find(it->second);
                      if (finder == expr_refs_to_remove->end())
                        (*expr_refs_to_remove)[it->second] = 1;
                      else
                        finder->second += 1;
                    }
                    else if (it->second->remove_nested_expression_reference(
                                                                        did))
                      delete it->second;
                    it->second = diff_expr;
                  }
                  // Otherwise, no overlap so we keep going
                  it++;
                }
                else
                {
                  if (view_refs_to_remove != NULL)
                  {
                    std::map<LogicalView*,unsigned>::iterator finder = 
                      view_refs_to_remove->find(it->first);
                    if (finder == view_refs_to_remove->end())
                      (*view_refs_to_remove)[it->first] = 1;
                    else
                      finder->second += 1;
                  }
                  else if (it->first->remove_nested_valid_ref(did))
                    delete it->first;
                  if (expr_refs_to_remove != NULL)
                  {
                    std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                      expr_refs_to_remove->find(it->second);
                    if (finder == expr_refs_to_remove->end())
                      (*expr_refs_to_remove)[it->second] = 1;
                    else
                      finder->second += 1;
                  }
                  else if (it->second->remove_nested_expression_reference(did))
                    delete it->second;
                  it = finder->second.erase(it);
                }
              }
            }
            if (finder->second.empty())
            {
              reduction_instances.erase(finder);
              reduction_fields.unset_bit(fidx);
            }
          }
        }
        fidx = filter_mask.find_next_set(fidx+1);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set_internal(
                                 CopyFillAggregator *&input_aggregator,
                                 CopyFillGuard *previous_guard,
                                 Operation *op, const unsigned index,
                                 const RegionUsage &usage,
                                 IndexSpaceExpression *expr, 
                                 const bool expr_covers,
                                 const FieldMask &user_mask,
                                 const FieldMaskSet<InstanceView> &target_insts,
                                 const std::vector<InstanceView*> &source_insts,
                                 const PhysicalTraceInfo &trace_info,
                                 const bool record_valid)
    //--------------------------------------------------------------------------
    {
      // Read-write or read-only
      // Issue fills and or copies to bring the target instances up to date
      make_instances_valid(input_aggregator, previous_guard, op, index, 
                           false/*track*/, expr, expr_covers, user_mask, 
                           target_insts, source_insts, trace_info);
      const bool is_write = IS_WRITE(usage);
      const FieldMask reduce_mask = reduction_fields & user_mask;
      const FieldMask restricted_mask = restricted_fields & user_mask;
      if (!!reduce_mask)
      {
        // Apply any reductions 
#ifdef DEBUG_LEGION
        assert(!target_insts.empty());
#endif
        FieldMaskSet<IndexSpaceExpression> applied_reductions;  
        apply_reductions(target_insts, expr, expr_covers, reduce_mask, 
                         input_aggregator, previous_guard, op, index, 
                         false/*track*/, trace_info,
                         is_write ? NULL : &applied_reductions);
        // If we're writing we're going to do an invalidation there anyway
        // so no need to bother with doing the invalidation based on the
        // reductions that have been applied
        if (!applied_reductions.empty())
        {
#ifdef DEBUG_LEGION
          assert(!is_write);
#endif
          // See if covered the full expressions for invalidation
          FieldMaskSet<IndexSpaceExpression>::iterator finder = 
            applied_reductions.find(expr);
          if (finder != applied_reductions.end())
          {
            if (!!restricted_mask)
            {
              const FieldMask overlap = finder->second & restricted_mask;
              if (!!overlap)
              {
                filter_unrestricted_instances(expr, expr_covers, overlap);
                finder.filter(overlap);
              }
            }
            if (!!finder->second)
              filter_valid_instances(expr, expr_covers, finder->second);
            // Remove the expression reference that flowed back
            if (finder->first->remove_nested_expression_reference(did))
              delete finder->first;
            applied_reductions.erase(finder);
          }
          if (!applied_reductions.empty())
          {
            // Handle the partial cases here
            LegionList<FieldSet<IndexSpaceExpression*> > reduced_sets;
            applied_reductions.compute_field_sets(FieldMask(), reduced_sets);
            for (LegionList<FieldSet<IndexSpaceExpression*> >::iterator
                  it = reduced_sets.begin(); it != reduced_sets.end(); it++)
            {
              IndexSpaceExpression *union_expr = 
                runtime->forest->union_index_spaces(it->elements);
              const size_t union_size = union_expr->get_volume();
              const size_t set_size = set_expr->get_volume();
#ifdef DEBUG_LEGION
              assert(union_size <= set_size);
#endif
              const bool union_covers = (union_size == set_size);
              if (!!restricted_mask)
              {
                const FieldMask overlap = it->set_mask & restricted_mask;
                if (!!overlap)
                {
                  filter_unrestricted_instances(union_expr, union_covers, 
                                                overlap);
                  it->set_mask -= overlap;
                }
              }
              if (!!it->set_mask)
                filter_valid_instances(union_expr, union_covers,
                                       it->set_mask);
            }
            // Remove expression references that flowed back
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  applied_reductions.begin(); it != 
                  applied_reductions.end(); it++)
              if (it->first->remove_nested_expression_reference(did))
                delete it->first;
          }
        }
      }
      if (is_write)
      {
        if (!!restricted_mask)
        {
          const FieldMask non_restricted_mask = user_mask - restricted_mask;
          if (!!non_restricted_mask)
            filter_valid_instances(expr, expr_covers, non_restricted_mask);
          filter_unrestricted_instances(expr, expr_covers, restricted_mask);
        }
        else
          filter_valid_instances(expr, expr_covers, user_mask);
      }
      // Finally record the valid instances that have been updated
      if (record_valid)
      {
        if (!!restricted_mask)
        {
          const FieldMask non_restricted = user_mask - restricted_mask;
          if (!!non_restricted)
            record_instances(expr, expr_covers, non_restricted,
                             target_insts);
          record_unrestricted_instances(expr, expr_covers, restricted_mask,
                             target_insts);
        }
        else
          record_instances(expr, expr_covers, user_mask, target_insts);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::make_instances_valid(CopyFillAggregator *&aggregator,
                                 CopyFillGuard *previous_guard,
                                 Operation *op, const unsigned index,
                                 const bool track_events,
                                 IndexSpaceExpression *expr,
                                 const bool expr_covers,
                                 const FieldMask &update_mask,
                                 const FieldMaskSet<InstanceView> &target_insts,
                                 const std::vector<InstanceView*> &source_insts,
                                 const PhysicalTraceInfo &trace_info,
                                 const bool skip_check,
                                 const int dst_index /*= -1*/,
                                 const ReductionOpID redop /*= 0*/,
                                 CopyAcrossHelper *across_helper/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty())
        return;
      FieldMaskSet<InstanceView> update_instances;
      for (FieldMaskSet<InstanceView>::const_iterator it =
            target_insts.begin(); it != target_insts.end(); it++)
      {
        FieldMask inst_mask = it->second & update_mask; 
        if (!inst_mask)
          continue;
        InstanceView *target = it->first;
        if (!skip_check)
        {
          // Check first to see if this is a total valid instance
          FieldMaskSet<LogicalView>::const_iterator finder = 
            total_valid_instances.find(target);
          if (finder != total_valid_instances.end())
          {
            inst_mask -= finder->second;
            if (!inst_mask)
              continue;
          }
          // Only check to see if it is a partial valid instance
          // if the expr does not cover
          if (!expr_covers)
          {
            const FieldMask partial_overlap = partial_valid_fields & inst_mask;
            if (!!partial_overlap)
            {
              ViewExprMaskSets::const_iterator partial_finder =
                partial_valid_instances.find(target);
              if (partial_finder != partial_valid_instances.end())
              {
                FieldMaskSet<IndexSpaceExpression>::const_iterator
                  expr_finder = partial_finder->second.find(expr);
                if (expr_finder != partial_finder->second.end())
                {
                  inst_mask -= expr_finder->second;
                  if (!inst_mask)
                    continue;
                }
              }
            }
          }
        }
#ifdef DEBUG_LEGION
        assert(inst_mask * update_instances.get_valid_mask());
#endif
        update_instances.insert(target, inst_mask);
      }
      if (update_instances.empty())
        return;
      for (FieldMaskSet<InstanceView>::iterator uit = 
            update_instances.begin(); uit != update_instances.end(); uit++)
      {
        // Check to see if we have to do the hairy path of having partially
        // valid subsets of the instance to worry about here
        if (!skip_check && !(uit->second * partial_valid_fields))
        {
          ViewExprMaskSets::const_iterator partial_finder =
            partial_valid_instances.find(uit->first);
          if (partial_finder != partial_valid_instances.end())
          {
            const FieldMask partial_valid = 
              uit->second & partial_finder->second.get_valid_mask();
            if (!!partial_valid)
            {
              FieldMaskSet<IndexSpaceExpression> partial_exprs;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    partial_finder->second.begin(); it !=
                    partial_finder->second.end(); it++)
              {
                const FieldMask overlap = it->second & partial_valid;
                if (!overlap)
                  continue;
#ifdef DEBUG_LEGION
                assert(overlap * partial_exprs.get_valid_mask());
#endif
                if (!expr_covers)
                {
                  IndexSpaceExpression *expr_overlap =
                    runtime->forest->intersect_index_spaces(expr, it->first);
                  const size_t expr_volume = expr_overlap->get_volume();
                  if (expr_volume == 0)
                    continue;
                  if (expr_volume == expr->get_volume())
                  {
                    // expression dominates us so we are valid
                    uit.filter(overlap);
                    if (!uit->second)
                      break;
                  }
                  else if (expr_volume == it->first->get_volume())
                    partial_exprs.insert(it->first, overlap);
                  else
                    partial_exprs.insert(expr_overlap, overlap);
                }
                else // expr covers so we know it all intersects
                  partial_exprs.insert(it->first, overlap);
              }
              if (!partial_exprs.empty())
              {
                for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                      partial_exprs.begin(); it != partial_exprs.end(); it++)
                {
                  IndexSpaceExpression *diff_expr = 
                    runtime->forest->subtract_index_spaces(expr, it->first);
                  if (!diff_expr->is_empty())
                    issue_update_copies_and_fills(uit->first, source_insts,
                       aggregator, previous_guard, op, index, track_events, 
                       diff_expr, false/*expr covers*/, it->second, trace_info,
                       dst_index, redop, across_helper); 
                }
                uit.filter(partial_exprs.get_valid_mask());
              }
            }
          }
        }
        // Whatever fields we have left here need updates for the whole expr
        if (!!uit->second)
          issue_update_copies_and_fills(uit->first, source_insts, aggregator, 
             previous_guard, op, index, track_events, expr, expr_covers,
             uit->second, trace_info, dst_index, redop, across_helper);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_update_copies_and_fills(InstanceView *target,
                                 const std::vector<InstanceView*> &source_views,
                                          CopyFillAggregator *&aggregator,
                                          CopyFillGuard *previous_guard,
                                          Operation *op, const unsigned index,
                                          const bool track_events,
                                          IndexSpaceExpression *expr,
                                          const bool expr_covers,
                                          FieldMask update_mask,
                                          const PhysicalTraceInfo &trace_info,
                                          const int dst_index,
                                          const ReductionOpID redop,
                                          CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
      // Before we do anything, if the user has provided an ordering of
      // source views, then go through and attempt to issue copies from
      // them before we do anything else
      if (!source_views.empty())
      {
        FieldMaskSet<IndexSpaceExpression> remainders;
        remainders.insert(expr, update_mask);
        for (std::vector<InstanceView*>::const_iterator src_it = 
              source_views.begin(); src_it != source_views.end(); src_it++)
        {
          // Check to see if it is in the list of total valid instances
          FieldMaskSet<LogicalView>::const_iterator total_finder = 
            total_valid_instances.find(*src_it);
          if ((total_finder != total_valid_instances.end()) &&
              !(remainders.get_valid_mask() * total_finder->second))
          {
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                  remainders.begin(); it != remainders.end(); it++)
            {
              const FieldMask overlap = it->second & total_finder->second;
              if (!overlap)
                continue;
              if (aggregator == NULL)
                aggregator = (dst_index >= 0) ?
                  new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                         previous_guard, track_events) :
                  new CopyFillAggregator(runtime->forest, op, index,
                                         previous_guard, track_events);
              aggregator->record_update(target, *src_it, overlap, it->first,
                  trace_info.recording ? this : NULL, redop);
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<IndexSpaceExpression*>::const_iterator it =
                    to_delete.begin(); it != to_delete.end(); it++)
                remainders.erase(*it);
              if (remainders.empty())
                return;
              remainders.tighten_valid_mask();
            }
          }
          // Next check to see if the instance has partial valid expressions
          ViewExprMaskSets::const_iterator partial_finder =
            partial_valid_instances.find(*src_it);
          if ((partial_finder == partial_valid_instances.end()) ||
              (partial_finder->second.get_valid_mask() * 
               remainders.get_valid_mask()))
            continue;
          // Compute the joins of the two field mask sets to get pairs of
          // index space expressions with the same fields
          LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,
                    FieldMask> join_expressions;
          unique_join_on_field_mask_sets(remainders, partial_finder->second, 
                                         join_expressions);
          bool need_tighten = false;
          for (LegionMap<std::pair<IndexSpaceExpression*,IndexSpaceExpression*>,          
                FieldMask>::const_iterator it = 
                join_expressions.begin(); it != join_expressions.end(); it++)
          {
            // Compute the intersection of the two index spaces 
            IndexSpaceExpression *overlap = 
              runtime->forest->intersect_index_spaces(it->first.first, 
                                                      it->first.second);
            const size_t overlap_size = overlap->get_volume();
            if (overlap_size == 0)
              continue;
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              remainders.find(it->first.first);
#ifdef DEBUG_LEGION
            assert(finder != remainders.end());
#endif
            finder.filter(it->second);
            if (!finder->second)
              remainders.erase(finder);
            if (aggregator == NULL)
              aggregator = (dst_index >= 0) ?
                new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                       previous_guard, track_events) :
                new CopyFillAggregator(runtime->forest, op, index,
                                       previous_guard, track_events);
            if (overlap_size < it->first.first->get_volume())
            {
              if (overlap_size == it->first.second->get_volume())
                aggregator->record_update(target, *src_it, it->second, 
                    it->first.second, trace_info.recording ? this : NULL,
                    redop);
              else
                aggregator->record_update(target, *src_it, it->second, overlap,
                    trace_info.recording ? this : NULL, redop);
              // Compute the difference to add to the remainders
              IndexSpaceExpression *diff = 
                runtime->forest->subtract_index_spaces(it->first.first,overlap);
              remainders.insert(diff, it->second);
            }
            else
            {
              // Covered the remainder expression
              aggregator->record_update(target, *src_it, it->second, 
                  it->first.first, trace_info.recording ? this : NULL, redop);
              if (remainders.empty())
                return;
              need_tighten = true;
            }
          }
          if (need_tighten)
            remainders.tighten_valid_mask();
        }
#ifdef DEBUG_LEGION
        assert(!remainders.empty());
#endif
        // It's too hard to track all the pairs of partial sets for
        // both the source and destination instances at the same
        // time, so we recurse on this method for any expressions that
        // are not the same as the original expression except this
        // time we will not have any sources to consider
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              remainders.begin(); it != remainders.end(); it++)
        {
          if (it->first != expr)
          {
            const std::vector<InstanceView*> empty_sources;
            issue_update_copies_and_fills(target, empty_sources, aggregator,
                            previous_guard, op, index, track_events, it->first,
                            false/*covers*/, it->second, trace_info, 
                            dst_index, redop, across_helper);
          }
          else // same expression so just keep the fields we need
            update_mask &= it->second;
        }
        // Fall through if we still have fields for this expression to handle
        if (!update_mask)
          return;
      }
      // We prefer bulk copies instead of lots of little copies, so do a quick
      // pass to see which fields we can find previous instances for that
      // completely cover our target without doing any intersection tests
      // If we find them then we'll just issue copies/fills from there, 
      // otherwise we'll build partial sets and do the expensive thing
      const FieldMask total_fields = 
        update_mask & total_valid_instances.get_valid_mask();
      if (!!total_fields)
      {
        if (aggregator == NULL)
          aggregator = (dst_index >= 0) ?
            new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                   previous_guard, track_events) :
            new CopyFillAggregator(runtime->forest, op, index,
                                   previous_guard, track_events);
        if (total_fields != total_valid_instances.get_valid_mask())
        {
          // Compute selected instances that are valid for us
          FieldMaskSet<LogicalView> total_instances;
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                total_valid_instances.begin(); it != 
                total_valid_instances.end(); it++)
          {
            const FieldMask overlap = it->second & update_mask;
            if (!overlap)
              continue;
            total_instances.insert(it->first, overlap);
          }
          aggregator->record_updates(target, total_instances, total_fields, 
              expr, trace_info.recording ? this : NULL, redop, across_helper);
        }
        else // Total valid instances covers everything!
          aggregator->record_updates(target, total_valid_instances,total_fields,
              expr, trace_info.recording ? this : NULL, redop, across_helper); 
        update_mask -= total_fields;
        if (!update_mask)
          return;
      }
      // Now look through the partial valid instances for both instances
      // that cover us as well as partially valid instances
      FieldMaskSet<LogicalView> cover_instances;
      LegionMap<LogicalView*,
        FieldMaskSet<IndexSpaceExpression> > partial_instances;
      for (ViewExprMaskSets::const_iterator pit =
            partial_valid_instances.begin(); pit != 
            partial_valid_instances.end(); pit++)
      {
        if (pit->second.get_valid_mask() * update_mask)
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              pit->second.begin(); it != pit->second.end(); it++)
        {
          FieldMask overlap = it->second & update_mask;
          if (!overlap)
            continue;
          if (!expr_covers)
          {
            if (it->first != expr)
            {
              IndexSpaceExpression *expr_overlap =
                runtime->forest->intersect_index_spaces(it->first, expr);
              const size_t overlap_volume = expr_overlap->get_volume();
              if (overlap_volume > 0)
              {
                if (overlap_volume < expr->get_volume())
                {
                  // partial overlap, only record this if we do not
                  // have any covering instances since we always prefer
                  // total coverings to partial ones
                  if (!cover_instances.empty())
                    overlap -= cover_instances.get_valid_mask();
                  if (!!overlap)
                  {
                    if (overlap_volume == it->first->get_volume())
                      partial_instances[pit->first].insert(it->first, overlap);
                    else
                      partial_instances[pit->first].insert(expr_overlap,
                                                           overlap);
                  }
                }
                else
                  cover_instances.insert(pit->first, overlap);
              }
            }
            else
              cover_instances.insert(pit->first, overlap);
          }
          else // expr covers so everything is partial
            partial_instances[pit->first].insert(it->first, overlap);
        }
      }
      if (!cover_instances.empty())
      {
        if (aggregator == NULL)
          aggregator = (dst_index >= 0) ?
            new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                   previous_guard, track_events) :
            new CopyFillAggregator(runtime->forest, op, index,
                                   previous_guard, track_events);
        aggregator->record_updates(target, cover_instances, 
            cover_instances.get_valid_mask(), expr, 
            trace_info.recording ? this : NULL, redop, across_helper);
        update_mask -= cover_instances.get_valid_mask();
        if (!update_mask)
          return;
      }
      // This is a horrible place to be, partial updates everywhere
      // so now we need to ask the mapper which order to do them in
      // Ask the copy fll aggregator to help us out with this since
      // its probably queried the mapper about this all before
      if (!partial_instances.empty())
      {
        if (aggregator == NULL)
          aggregator = (dst_index >= 0) ?
            new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                   previous_guard, track_events) :
            new CopyFillAggregator(runtime->forest, op, index,
                                   previous_guard, track_events);
        aggregator->record_partial_updates(target,partial_instances,update_mask,
                                       expr, trace_info.recording ? this : NULL,
                                       redop, across_helper);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_reductions(
                            const FieldMaskSet<InstanceView> &reduction_targets,
                            IndexSpaceExpression *expr, const bool expr_covers,
                            const FieldMask &reduction_mask,
                            CopyFillAggregator *&aggregator,
                            CopyFillGuard *previous_guard,
                            Operation *op, const unsigned index, 
                            const bool track_events,
                            const PhysicalTraceInfo &trace_info,
                            FieldMaskSet<IndexSpaceExpression> *applied_exprs,
                            CopyAcrossHelper *across_helper/*= NULL*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!reduction_targets.get_valid_mask());
      assert(!set_expr->is_empty());
#endif
      const bool track_exprs = (applied_exprs != NULL);
      for (FieldMaskSet<InstanceView>::const_iterator rit =
            reduction_targets.begin(); rit != reduction_targets.end(); rit++)
      {
        const FieldMask inst_mask = rit->second & reduction_mask;
        if (!inst_mask)
          continue;
        const bool target_is_reduction = rit->first->is_reduction_view();
        int fidx = inst_mask.find_first_set();
        while (fidx >= 0)
        {
          std::map<unsigned,std::list<
            std::pair<ReductionView*,IndexSpaceExpression*> > >::iterator 
              finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
          assert(finder != reduction_instances.end());
          assert(!finder->second.empty());
#endif 
          if (expr_covers)
          {
            // If the target is a reduction instance, check to see
            // that we at least have one reduction to apply
            if (target_is_reduction)
            {
              // Filter out all of our reductions
              for (std::list<std::pair<ReductionView*,
                             IndexSpaceExpression*> >::iterator it =
                    finder->second.begin(); it != 
                    finder->second.end(); /*nothing*/)
              {
                if (it->first == rit->first)
                {
                  if (it->first->remove_nested_valid_ref(did))
                    delete it->first;
                  if (it->second->remove_nested_expression_reference(did))
                    delete it->second;
                  it = finder->second.erase(it);
                }
                else
                  it++;
              }
              if (finder->second.empty())
              {
                // Quick out if there was nothing to apply
                reduction_instances.erase(finder);
                reduction_fields.unset_bit(fidx);
                fidx = inst_mask.find_next_set(fidx+1);
                continue;
              }
            }
            if (aggregator == NULL)
              aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                                previous_guard, track_events);
            aggregator->record_reductions(rit->first, finder->second, fidx,
              (across_helper == NULL) ? fidx : 
                across_helper->convert_src_to_dst(fidx),
              trace_info.recording ? this : NULL, across_helper);
            bool has_cover = false;
            for (std::list<std::pair<ReductionView*,
                  IndexSpaceExpression*> >::const_iterator it =
                  finder->second.begin(); it != finder->second.end(); it++)
            {
              if (it->second == set_expr)
                has_cover = true;
              if (it->first->remove_nested_valid_ref(did))
                delete it->first;
              // Only remove expression references here if we're not
              // tracking expressions
              if (!track_exprs &&
                  it->second->remove_nested_expression_reference(did))
                delete it->second;
            }
            if (track_exprs)
            {
              // See if we find ourselves, if not just record all of them
              FieldMask expr_mask;
              expr_mask.set_bit(fidx);
              if (!has_cover)
              {
                // Expression references flow back but remove duplicates
                for (std::list<std::pair<ReductionView*,
                      IndexSpaceExpression*> >::const_iterator it =
                      finder->second.begin(); it != finder->second.end(); it++)
                  if (!applied_exprs->insert(it->second, expr_mask) &&
                      it->second->remove_nested_expression_reference(did))
                    assert(false); // should never hit this
              }
              else
              {
                if (applied_exprs->insert(set_expr, expr_mask))
                  set_expr->add_nested_expression_reference(did);
                // Now we can remove the remaining expression references
                for (std::list<std::pair<ReductionView*,
                      IndexSpaceExpression*> >::const_iterator it =
                      finder->second.begin(); it != finder->second.end(); it++)
                  if (it->second->remove_nested_expression_reference(did))
                    delete it->second;
              }
            }
            // We applied all these reductions so we're done
            reduction_instances.erase(finder);
            reduction_fields.unset_bit(fidx);
          }
          else
          {
            bool has_cover = false;
            std::vector<
              std::pair<ReductionView*,IndexSpaceExpression*> > to_delete;
            std::list<
              std::pair<ReductionView*,IndexSpaceExpression*> > to_record;
            // expr does not cover so we need intersection tests
            for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
                  iterator it = finder->second.begin();
                  it != finder->second.end(); /*nothing*/)
            {
              if (target_is_reduction && (it->first == rit->first))
              {
                to_delete.push_back(*it);
                it = finder->second.erase(it);
              }
              else if (it->second == expr)
              {
                to_record.push_back(*it);
                to_delete.push_back(*it);
                if (track_exprs)
                  has_cover = true;
                it = finder->second.erase(it);
              }
              else if (it->second == set_expr)
              {
                to_record.push_back(std::make_pair(it->first, expr));
                if (track_exprs)
                  has_cover = true;
                IndexSpaceExpression *remainder = 
                  runtime->forest->subtract_index_spaces(set_expr, expr);
                remainder->add_nested_expression_reference(did);
                it->second = remainder;
                if (set_expr->remove_nested_expression_reference(did))
                  assert(false); // should never hit this
                it++;
              }
              else
              {
                IndexSpaceExpression *overlap = 
                  runtime->forest->intersect_index_spaces(expr, it->second);
                const size_t overlap_size = overlap->get_volume();
                if (overlap_size == 0)
                {
                  it++;
                  continue;
                }
                if (overlap_size == expr->get_volume())
                {
                  to_record.push_back(std::make_pair(it->first, expr));
                  if (track_exprs)
                    has_cover = true;
                }
                else
                  to_record.push_back(std::make_pair(it->first, overlap));
                if (overlap_size == it->second->get_volume())
                {
                  to_delete.push_back(*it);
                  it = finder->second.erase(it);
                }
                else
                {
                  IndexSpaceExpression *remainder = 
                    runtime->forest->subtract_index_spaces(it->second, expr);
                  remainder->add_nested_expression_reference(did);
                  if (it->second->remove_nested_expression_reference(did))
                    delete it->second;
                  it->second = remainder;
                  it++;
                }
              }
            }
            if (!to_record.empty())
            {
              if (aggregator == NULL)
                aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                                  previous_guard, track_events);
              aggregator->record_reductions(rit->first, to_record, fidx, 
                                      (across_helper == NULL) ? fidx : 
                                        across_helper->convert_src_to_dst(fidx),
                                      trace_info.recording ? this : NULL, 
                                      across_helper);
              if (track_exprs)
              {
                FieldMask expr_mask;
                expr_mask.set_bit(fidx);
                if (!has_cover)
                {
                  for (std::list<std::pair<ReductionView*,
                        IndexSpaceExpression*> >::const_iterator it = 
                        to_record.begin(); it != to_record.end(); it++)
                    if (applied_exprs->insert(it->second, expr_mask))
                      it->second->add_nested_expression_reference(did);
                }
                else if (applied_exprs->insert(expr, expr_mask))
                  expr->add_nested_expression_reference(did);
              }
            }
            if (!to_delete.empty())
            {
              for (std::vector<std::pair<ReductionView*,
                    IndexSpaceExpression*> >::const_iterator it =
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
                if (it->second->remove_nested_expression_reference(did))
                  delete it->second;
              }
            }
            if (finder->second.empty())
            {
              reduction_instances.erase(finder);
              reduction_fields.unset_bit(fidx);
            }
          }
          fidx = inst_mask.find_next_set(fidx+1);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void EquivalenceSet::copy_out(IndexSpaceExpression *expr, 
                                  const bool expr_covers,
                                  const FieldMask &restricted_mask, 
                                  const FieldMaskSet<T> &src_insts,
                                  Operation *op, const unsigned index,
                                  const PhysicalTraceInfo &trace_info,
                                  CopyFillAggregator *&aggregator)
    //--------------------------------------------------------------------------
    {
      if (expr->is_empty())
        return;
      // Iterate through the restrictions looking for overlaps
      for (ExprViewMaskSets::const_iterator rit = restricted_instances.begin();
            rit != restricted_instances.end(); rit++)
      {
        const FieldMask overlap = 
          rit->second.get_valid_mask() & restricted_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression *overlap_expr = NULL;
        if (expr_covers)
          overlap_expr = rit->first;
        else if (rit->first == set_expr)
          overlap_expr = expr;
        else
        {
          IndexSpaceExpression *over = 
            runtime->forest->intersect_index_spaces(rit->first, expr);
          if (over->is_empty())
            continue;
          const size_t over_size = over->get_volume();
          if (over_size == expr->get_volume())
            overlap_expr = expr;
          else if (over_size == rit->first->get_volume())
            overlap_expr = rit->first;
          else
            overlap_expr = over;
        }
        // Find the restricted destination instances for these fields
        LegionMap<std::pair<InstanceView*,T*>,FieldMask> restricted_copies;
        unique_join_on_field_mask_sets(rit->second,src_insts,restricted_copies);
        if (restricted_copies.empty())
          continue;
        for (typename LegionMap<std::pair<InstanceView*,T*>,FieldMask>::
              const_iterator it = restricted_copies.begin();
              it != restricted_copies.end(); it++)
        {
          if (it->first.first == it->first.second)
            continue;
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                  NULL/*no previous guard*/, true/*track*/);
          aggregator->record_update(it->first.first, it->first.second, overlap,
              overlap_expr, trace_info.recording ? this : NULL);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::acquire_restrictions(AcquireAnalysis &analysis, 
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers, 
                                             const FieldMask &acquire_mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, acquire_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, acquire_mask, deferral_events,
                             applied_events, expr_covers))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(acquire_mask));
#endif
      std::vector<IndexSpaceExpression*> to_delete;
      std::map<IndexSpaceExpression*,IndexSpaceExpression*> to_add;
      // Now we need to lock the analysis if we're going to do this traversal
      AutoLock a_lock(analysis);
      for (ExprViewMaskSets::iterator eit = restricted_instances.begin();
            eit != restricted_instances.end(); eit++)
      {
        FieldMask overlap = eit->second.get_valid_mask() & acquire_mask;
        if (!overlap)
          continue;
        IndexSpaceExpression *overlap_expr = NULL;
        bool done_early = false;
        if (!expr_covers && (eit->first != expr))
        {
          overlap_expr = 
            runtime->forest->intersect_index_spaces(eit->first, expr);
          const size_t overlap_size = overlap_expr->get_volume();
          if (overlap_size == 0)
            continue;
          if (overlap_size == eit->first->get_volume())
          {
            overlap_expr = eit->first;
            if (overlap_size == expr->get_volume())
              done_early = (overlap == acquire_mask);
          }
          else if (overlap_size == expr->get_volume())
            overlap_expr = expr;
        }
        else
        {
          overlap_expr = eit->first;
          if (eit->first == expr)
            done_early = (overlap == acquire_mask);
        }
        ExprViewMaskSets::iterator release_finder =
          released_instances.find(overlap_expr);
        if (release_finder == released_instances.end())
        {
          overlap_expr->add_nested_expression_reference(did);
          released_instances[overlap_expr];
          release_finder = released_instances.find(overlap_expr);
        }
        if (overlap_expr == eit->first)
        {
          // Total covering of expressions
          // so remove instances no longer restricted
          if (overlap == eit->second.get_valid_mask())
          {
            // All instances are going to be released
            if (!release_finder->second.empty())
            {
              // Insert and remove duplicate references
              for (FieldMaskSet<InstanceView>::const_iterator it =
                    eit->second.begin(); it != eit->second.end(); it++)
              {
                analysis.record_instance(it->first, it->second);
                if (!release_finder->second.insert(it->first, it->second) &&
                    it->first->remove_nested_valid_ref(did))
                  assert(false); // should never delete this
              }
              eit->second.clear();
            }
            else
            {
              for (FieldMaskSet<InstanceView>::const_iterator it =
                    eit->second.begin(); it != eit->second.end(); it++)
                analysis.record_instance(it->first, it->second);
              release_finder->second.swap(eit->second);
            }
            to_delete.push_back(eit->first);
          }
          else
          {
            // Filter instances whose fields overlap
            std::vector<InstanceView*> to_erase;
            for (FieldMaskSet<InstanceView>::iterator it = 
                  eit->second.begin(); it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              // Add it to the released instances
              if (release_finder->second.insert(it->first, inst_overlap))
                it->first->add_nested_valid_ref(did);
              // Remove it from here
              it.filter(inst_overlap);
              if (!it->second)
                to_erase.push_back(it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
            for (std::vector<InstanceView*>::const_iterator it =
                  to_erase.begin(); it != to_erase.end(); it++)
              if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            if (!eit->second.empty())
              eit->second.tighten_valid_mask();
            else
              to_delete.push_back(eit->first);
          }
        }
        else
        {
          // Only partial covering, so compute the difference
          // and record that we'll pull valid instances from here
          to_add[eit->first] = 
            runtime->forest->subtract_index_spaces(eit->first, expr);
          // The intersection gets merged back into relased sets
          for (FieldMaskSet<InstanceView>::const_iterator it =
                eit->second.begin(); it != eit->second.end(); it++)
          {
            const FieldMask inst_overlap = overlap & it->second;
            if (!inst_overlap)
              continue;
            analysis.record_instance(it->first, inst_overlap);
            if (release_finder->second.insert(it->first, it->second))
              it->first->add_nested_valid_ref(did);
            // Each field should only be represented by one instance
            overlap -= inst_overlap;
            if (!overlap)
              break;
          }
        }
        // If expressions matched and we handled all the fields then
        // we can be done since we know there are no other overlaps
        if (done_early)
          break;
      }
      for (std::map<IndexSpaceExpression*,IndexSpaceExpression*>::const_iterator
            eit = to_add.begin(); eit != to_add.end(); eit++)
      {
        if (restricted_instances.find(eit->second) ==restricted_instances.end())
          eit->second->add_nested_expression_reference(did);
        FieldMaskSet<InstanceView> &old_insts =restricted_instances[eit->first];
        FieldMaskSet<InstanceView> &new_insts=restricted_instances[eit->second];
        if (!new_insts.empty() || !!(old_insts.get_valid_mask() & acquire_mask))
        {
          std::vector<InstanceView*> to_erase;
          for (FieldMaskSet<InstanceView>::iterator it =
                old_insts.begin(); it != old_insts.end(); it++)
          {
            const FieldMask overlap = it->second & acquire_mask;
            if (!overlap)
              continue;
            if (new_insts.insert(it->first, overlap))
              it->first->add_nested_valid_ref(did);
            it.filter(overlap);
            if (!it->second)
              to_erase.push_back(it->first);
          }
          for (std::vector<InstanceView*>::const_iterator it =
                to_erase.begin(); it != to_erase.end(); it++)
          {
            old_insts.erase(*it);
            if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
          }
          if (old_insts.empty())
            to_delete.push_back(eit->first);
          else
            old_insts.tighten_valid_mask();
        }
        else
        {
          new_insts.swap(old_insts); 
          to_delete.push_back(eit->first);
        }
      }
      for (std::vector<IndexSpaceExpression*>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
      {
        restricted_instances.erase(*it);
        if ((*it)->remove_nested_expression_reference(did))
          delete (*it);
      }
      restricted_fields.clear();
      for (ExprViewMaskSets::const_iterator it = restricted_instances.begin();
            it != restricted_instances.end(); it++)
        restricted_fields |= it->second.get_valid_mask();
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::release_restrictions(ReleaseAnalysis &analysis, 
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers, 
                                             const FieldMask &release_mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, release_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, release_mask, deferral_events,
                             applied_events, expr_covers))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(release_mask));
#endif
      LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> > to_update;
      // We need to lock the analysis at this point
      AutoLock a_lock(analysis);
      // If the target views are empty then we are just restoring the
      // existing released instances, if we have target views then we
      // know what the restricted instaces are going to be but we still
      // need to filter out any previously released instances
      if (analysis.target_views.empty())
      {
        std::vector<IndexSpaceExpression*> to_delete;
        std::map<IndexSpaceExpression*,IndexSpaceExpression*> to_add;
        for (ExprViewMaskSets::iterator eit = released_instances.begin();
              eit != released_instances.end(); eit++)
        {
          FieldMask overlap = eit->second.get_valid_mask() & release_mask;
          if (!overlap)
            continue;
          IndexSpaceExpression *overlap_expr = NULL;
          if (!expr_covers && (eit->first != expr))
          {
            overlap_expr = 
              runtime->forest->intersect_index_spaces(eit->first, expr);
            const size_t overlap_size = overlap_expr->get_volume();
            if (overlap_size == 0)
              continue;
            if (overlap_size == eit->first->get_volume())
              overlap_expr = eit->first;
            else if (overlap_size == expr->get_volume())
              overlap_expr = expr;
          }
          else
            overlap_expr = eit->first;
          const bool overlap_covers = 
            (overlap_expr->get_volume() == set_expr->get_volume());
          if (overlap_expr == eit->first)
          {
            // Total covering of expressions
            // so move all instances back to being restricted
            std::vector<InstanceView*> to_erase;
            FieldMaskSet<InstanceView> &updates = to_update[eit->first];
            for (FieldMaskSet<InstanceView>::iterator it = 
                  eit->second.begin(); it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              updates.insert(it->first, inst_overlap);
              // Record this as a restricted instance
              record_restriction(overlap_expr, overlap_covers, inst_overlap,
                                 it->first);
              // Remove it from here
              it.filter(inst_overlap);
              if (!it->second)
                to_erase.push_back(it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
            for (std::vector<InstanceView*>::const_iterator it =
                  to_erase.begin(); it != to_erase.end(); it++)
            {
              eit->second.erase(*it);
              if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            }
            if (!eit->second.empty())
              eit->second.tighten_valid_mask();
            else
              to_delete.push_back(eit->first);
          }
          else
          {
            // Only partial covering, so compute the difference
            // and record that we'll pull valid instances from here
            to_add[eit->first] = 
              runtime->forest->subtract_index_spaces(eit->first, expr);
            FieldMaskSet<InstanceView> &updates = to_update[overlap_expr];
            // The intersection gets merged back into relased sets
            for (FieldMaskSet<InstanceView>::const_iterator it =
                  eit->second.begin(); it != eit->second.end(); it++)
            {
              const FieldMask inst_overlap = overlap & it->second;
              if (!inst_overlap)
                continue;
              analysis.record_instance(it->first, inst_overlap);
              updates.insert(it->first, inst_overlap);
              // Record this as a restricted instance
              record_restriction(overlap_expr, overlap_covers, inst_overlap,
                                 it->first);
              // Each field should only be represented by one instance
              overlap -= inst_overlap;
              if (!overlap)
                break;
            }
          }
        }
        // Record updates to the released sets
        for (std::map<IndexSpaceExpression*,IndexSpaceExpression*>::const_iterator
              eit = to_add.begin(); eit != to_add.end(); eit++)
        {
          if (released_instances.find(eit->first) == released_instances.end())
            eit->first->add_nested_expression_reference(did);
          FieldMaskSet<InstanceView> &new_insts = released_instances[eit->first];
          FieldMaskSet<InstanceView> &old_insts = released_instances[eit->second];
          if (!new_insts.empty() || !!(old_insts.get_valid_mask() & release_mask))
          {
            std::vector<InstanceView*> to_erase;
            for (FieldMaskSet<InstanceView>::iterator it =
                  old_insts.begin(); it != old_insts.end(); it++)
            {
              const FieldMask overlap = it->second & release_mask;
              if (!overlap)
                continue;
              if (new_insts.insert(it->first, overlap))
                it->first->add_nested_valid_ref(did);
              it.filter(overlap);
              if (!it->second)
                to_erase.push_back(it->first);
            }
            for (std::vector<InstanceView*>::const_iterator it =
                  to_erase.begin(); it != to_erase.end(); it++)
            {
              old_insts.erase(*it);
              if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            }
            if (old_insts.empty())
              to_delete.push_back(eit->first);
            else
              old_insts.tighten_valid_mask();
          }
          else
          {
            new_insts.swap(old_insts); 
            to_delete.push_back(eit->first);
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          released_instances.erase(*it);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
      }
      else
      {
        // If we're not restoring the released instance then we should
        // record the actual instances that we are making restricted
        // Make sure that we don't have any overlapping restrictions
        filter_restricted_instances(expr, expr_covers, release_mask);
        // Make sure that we remove any old released instances
        filter_released_instances(expr, expr_covers, release_mask);
        FieldMaskSet<InstanceView> &updates = to_update[expr];
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          InstanceView *view = analysis.target_views[idx];
          const FieldMask &mask = 
            analysis.target_instances[idx].get_valid_fields();
          updates.insert(view, mask);
          // Record this as a restricted instance
          record_restriction(expr, expr_covers, mask, view);
        }
      }
      // Now generate the copies for any updates to the restricted instances
      if (analysis.release_aggregator != NULL)
        analysis.release_aggregator->clear_update_fields();
      const RegionUsage release_usage(LEGION_READ_WRITE, LEGION_EXCLUSIVE, 0);
      for (LegionMap<IndexSpaceExpression*,
                     FieldMaskSet<InstanceView> >::const_iterator it =
            to_update.begin(); it != to_update.end(); it++)
        update_set_internal(analysis.release_aggregator, NULL/*no guard*/,
                            analysis.op, analysis.index, release_usage,
                            it->first, (it->first == set_expr), 
                            it->second.get_valid_mask(), it->second,
                            analysis.source_views, analysis.trace_info,
                            true/*record valid*/); 
      // Finally update the tracing postconditions now that we've recorded
      // any copies as part of the trace
      if (tracing_postconditions != NULL)
      {
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          InstanceView *restrict_view = analysis.target_views[idx];
          const FieldMask &restrict_mask =
            analysis.target_instances[idx].get_valid_fields();
          tracing_postconditions->invalidate_all_but(restrict_view, expr,
                                                     restrict_mask);
        }
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::record_restriction(IndexSpaceExpression *expr, 
                                            const bool expr_covers,
                                            const FieldMask &restrict_mask,
                                            InstanceView *restrict_view)
    //--------------------------------------------------------------------------
    {
      // This function only looks for merges of restrictions. It assumes
      // that restrictions are inherently non-overlapping and does not
      // attempt to enforce that there are no such overlaps

      // First see if we need to merge with any existing restrictions
      if (expr_covers)
      {
        // No need to check for merging, we should be independent
        ExprViewMaskSets::iterator restricted_finder =
          restricted_instances.find(set_expr);
        if (restricted_finder == restricted_instances.end())
        {
          set_expr->add_nested_expression_reference(did);
          restrict_view->add_nested_valid_ref(did);
          restricted_instances[set_expr].insert(restrict_view, restrict_mask);
        }
        else if (restricted_finder->second.insert(restrict_view, restrict_mask))
          restrict_view->add_nested_valid_ref(did);
      }
      else
      {
        // Check to see if we can union this expression with any others
        FieldMaskSet<IndexSpaceExpression> to_union;
        std::vector<IndexSpaceExpression*> to_delete;
        for (ExprViewMaskSets::iterator eit = restricted_instances.begin();
              eit != restricted_instances.end(); eit++)
        {
          FieldMaskSet<InstanceView>::iterator finder = 
            eit->second.find(restrict_view);
          if (finder == eit->second.end())
            continue;
          const FieldMask overlap = finder->second & restrict_mask;
          if (!overlap)
            continue;
          to_union.insert(eit->first, overlap);
          finder.filter(overlap);
          if (!finder->second)
          {
            if (finder->first->remove_nested_valid_ref(did))
              delete finder->first;
            eit->second.erase(finder);
            if (eit->second.empty())
              to_delete.push_back(eit->first);
          }
          else
            eit->second.tighten_valid_mask();
        }
        // Add in the new sets
        if (!to_union.empty())
        {
          LegionList<FieldSet<IndexSpaceExpression*> > expr_sets;
          to_union.compute_field_sets(FieldMask(), expr_sets);
          for (LegionList<FieldSet<IndexSpaceExpression*> >::iterator
                it = expr_sets.begin(); it != expr_sets.end(); it++)
          {
            it->elements.insert(expr);
            IndexSpaceExpression *union_expr = 
              runtime->forest->union_index_spaces(it->elements); 
            if (union_expr->get_volume() < set_expr->get_volume())
            {
              ExprViewMaskSets::iterator restricted_finder =
                restricted_instances.find(union_expr);
              if (restricted_finder == restricted_instances.end())
              {
                union_expr->add_nested_expression_reference(did);
                restrict_view->add_nested_valid_ref(did);
                restricted_instances[union_expr].insert(restrict_view, 
                                                      it->set_mask);
              }
              else if (restricted_finder->second.insert(restrict_view, 
                                                        it->set_mask))
                restrict_view->add_nested_valid_ref(did);
            }
            else
            {
              ExprViewMaskSets::iterator restricted_finder =
                restricted_instances.find(set_expr);
              if (restricted_finder == restricted_instances.end())
              {
                set_expr->add_nested_expression_reference(did);
                restrict_view->add_nested_valid_ref(did);
                restricted_instances[set_expr].insert(restrict_view, 
                                                      it->set_mask);
              }
              else if (restricted_finder->second.insert(restrict_view, 
                                                        it->set_mask))
                restrict_view->add_nested_valid_ref(did);
            }
          }
          const FieldMask remainder = restrict_mask - to_union.get_valid_mask();
          if (!!remainder)
          {
            ExprViewMaskSets::iterator restricted_finder =
              restricted_instances.find(expr);
            if (restricted_finder == restricted_instances.end())
            {
              expr->add_nested_expression_reference(did);
              restrict_view->add_nested_valid_ref(did);
              restricted_instances[expr].insert(restrict_view, remainder);
            }
            else if (restricted_finder->second.insert(restrict_view, remainder))
              restrict_view->add_nested_valid_ref(did);
          }
        }
        else
        {
          // Just record ourselves since there was nothing to merge
          ExprViewMaskSets::iterator restricted_finder =
            restricted_instances.find(expr);
          if (restricted_finder == restricted_instances.end())
          {
            expr->add_nested_expression_reference(did);
            restrict_view->add_nested_valid_ref(did);
            restricted_instances[expr].insert(restrict_view, restrict_mask);
          }
          else if (restricted_finder->second.insert(restrict_view, 
                                                    restrict_mask))
            restrict_view->add_nested_valid_ref(did);
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          restricted_instances.erase(*it);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
      }
#ifdef DEBUG_LEGION
      assert(!restricted_instances.empty());
#endif
      // Always update the restricted fields
      restricted_fields |= restrict_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_reductions(const unsigned fidx, 
           std::list<std::pair<ReductionView*,IndexSpaceExpression*> > &updates)
    //--------------------------------------------------------------------------
    {
      if (updates.empty())
        return;
      // Check for equivalence to the dst and then add our references
      const size_t volume = set_expr->get_volume();
      for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::iterator
            it = updates.begin(); it != updates.end(); it++)
      {
        it->first->add_nested_valid_ref(did);
        if (it->second->get_volume() == volume)
          it->second = set_expr;
        it->second->add_nested_expression_reference(did);
      }
      std::list<std::pair<ReductionView*,IndexSpaceExpression*> > &current =
        reduction_instances[fidx];
      current.splice(current.end(), updates);
      reduction_fields.set_bit(fidx);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_released(IndexSpaceExpression *expr, 
                    const bool expr_covers, FieldMaskSet<InstanceView> &updates)
    //--------------------------------------------------------------------------
    {
      if (expr->get_volume() == set_expr->get_volume())
        expr = set_expr;
      ExprViewMaskSets::iterator finder = released_instances.find(expr);
      if (finder != released_instances.end())
      {
        for (FieldMaskSet<InstanceView>::const_iterator it =
              updates.begin(); it != updates.end(); it++)
          if (finder->second.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did);
      }
      else
      {
        expr->add_nested_expression_reference(did);
        for (FieldMaskSet<InstanceView>::const_iterator it =
              updates.begin(); it != updates.end(); it++)
          it->first->add_nested_valid_ref(did);
        released_instances[expr].swap(updates);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_initialized_data(IndexSpaceExpression *expr, 
                  const bool expr_covers, const FieldMask &filter_mask, 
                  std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (initialized_data.empty() || 
          (filter_mask * initialized_data.get_valid_mask()))
        return;
      if (expr_covers)
      {
        if (!(initialized_data.get_valid_mask() - filter_mask))
        {
          // filter everything
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                initialized_data.begin(); it != initialized_data.end(); it++)
          {
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(it->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[it->first] = 1;
              else
                finder->second += 1;
            }
            else if (it->first->remove_nested_expression_reference(did))
              delete it->first;
          }
          initialized_data.clear();
        }
        else
        {
          // filter fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                initialized_data.begin(); it != initialized_data.end(); it++)
          {
            it.filter(filter_mask);
            if (!it->second)
              to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              initialized_data.erase(*it);
              if (expr_refs_to_remove != NULL)
              {
                std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                  expr_refs_to_remove->find(*it);
                if (finder == expr_refs_to_remove->end())
                  (*expr_refs_to_remove)[*it] = 1;
                else
                  finder->second += 1;
              }
              else if ((*it)->remove_nested_expression_reference(did))
                delete (*it);
            }
          }
          if (!initialized_data.empty())
            initialized_data.tighten_valid_mask();
        }
      }
      else
      {
        // Filter on fields first and then on expressions
        std::vector<IndexSpaceExpression*> to_delete;
        FieldMaskSet<IndexSpaceExpression> to_add;
        for (FieldMaskSet<IndexSpaceExpression>::iterator it =
              initialized_data.begin(); it != initialized_data.end(); it++)
        {
          const FieldMask overlap = filter_mask & it->second;
          if (!overlap)
            continue;
          if (it->first != set_expr)
          {
            IndexSpaceExpression *intersection =
              runtime->forest->intersect_index_spaces(it->first, expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            // We're removing this expression no matter what at this point
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            // See if there are any remaining points left
            if (volume < it->first->get_volume())
            {
              IndexSpaceExpression *diff = 
                runtime->forest->subtract_index_spaces(it->first, intersection);
              to_add.insert(diff, overlap);
            }
          }
          else // special case for when we know that the expr is the set expr
          {
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            to_add.insert(
              runtime->forest->subtract_index_spaces(it->first, expr), overlap);
          }
        }
        if (!to_add.empty())
        {
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (initialized_data.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
        }
        if (!to_delete.empty())
        {
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            if (to_add.find(*it) != to_add.end())
              continue;
            initialized_data.erase(*it);
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(*it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_expression_reference(did))
              delete (*it);
          }
        }
        if (!initialized_data.empty())
          initialized_data.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_restricted_instances(IndexSpaceExpression *expr,
                  const bool expr_covers, const FieldMask &filter_mask,
                  std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
                  std::map<LogicalView*,unsigned> *view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (restricted_instances.empty() || (filter_mask * restricted_fields))
        return;
      if (expr_covers)
      {
        if (!(restricted_fields - filter_mask))
        {
          // filter everything
          for (ExprViewMaskSets::const_iterator rit =
                restricted_instances.begin(); rit != 
                restricted_instances.end(); rit++)
          {
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(rit->first);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[rit->first] = 1;
              else
                finder->second += 1;
            }
            else if (rit->first->remove_nested_expression_reference(did))
              delete rit->first;
            for (FieldMaskSet<InstanceView>::const_iterator it =
                  rit->second.begin(); it != rit->second.end(); it++)
            {
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder = 
                  view_refs_to_remove->find(it->first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it->first] = 1;
                else
                  finder->second += 1;
              }
              else if (it->first->remove_nested_valid_ref(did))
                delete it->first;
            }
          }
          restricted_instances.clear();
          restricted_fields.clear();
        }
        else
        {
          // filter fields
          std::vector<IndexSpaceExpression*> to_delete;
          for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
                rit != restricted_instances.end(); rit++)
          {
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // delete all the views in this one
              for (FieldMaskSet<InstanceView>::const_iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.push_back(rit->first);
            }
            else
            {
              // filter views based on fields
              std::vector<InstanceView*> to_erase;
              for (FieldMaskSet<InstanceView>::iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              for (std::vector<InstanceView*>::const_iterator it =
                    to_erase.begin(); it != to_erase.end(); it++)
              {
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(*it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[*it] = 1;
                  else
                    finder->second += 1;
                }
                else if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
              }
              if (rit->second.empty())
                to_delete.push_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          for (std::vector<IndexSpaceExpression*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            restricted_instances.erase(*it);
            if (expr_refs_to_remove != NULL)
            {
              std::map<IndexSpaceExpression*,unsigned>::iterator finder =
                expr_refs_to_remove->find(*it);
              if (finder == expr_refs_to_remove->end())
                (*expr_refs_to_remove)[*it] = 1;
              else
                finder->second += 1;
            }
            else if ((*it)->remove_nested_expression_reference(did))
              delete (*it);
          }
          restricted_fields -= filter_mask;
        }
      }
      else
      {
        // Expression does not cover this equivalence set
        std::vector<IndexSpaceExpression*> to_delete;
        LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> > to_add;
        for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
              rit != restricted_instances.end(); rit++)
        {
          if (rit->second.get_valid_mask() * filter_mask)
            continue;
          IndexSpaceExpression *intersection = (rit->first == set_expr) ? expr :
            runtime->forest->intersect_index_spaces(rit->first, expr);
          const size_t volume = intersection->get_volume();
          if (volume == 0)
            continue;
          if (volume == rit->first->get_volume())
          {
            // Covers the whole expression
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // filter all of them
              for (FieldMaskSet<InstanceView>::const_iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.push_back(rit->first);
            }
            else
            {
              // fitler by fields
              std::vector<InstanceView*> to_erase;
              for (FieldMaskSet<InstanceView>::iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_erase.begin(); it != to_erase.end(); it++)
              {
                rit->second.erase(*it);
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(*it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[*it] = 1;
                  else
                    finder->second += 1;
                }
                else if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
              }
              if (rit->second.empty())
                to_delete.push_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          else
          {
            // Only covers part, so compute diff and put them in the add set
            IndexSpaceExpression *diff = 
              runtime->forest->subtract_index_spaces(rit->first, intersection);
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // All the views are flowing into to_add
              LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::iterator finder = 
                  to_add.find(diff);
              if (finder != to_add.end())
              {
                // Deduplicate references in to add
                for (FieldMaskSet<InstanceView>::const_iterator it =
                      rit->second.begin(); it != rit->second.end(); it++)
                  if (!finder->second.insert(it->first, it->second) &&
                      it->first->remove_nested_valid_ref(did))
                    assert(false); // should never hit this
              }
              else
                to_add[diff].swap(rit->second);
              to_delete.push_back(rit->first);
            }
            else
            {
              // Filter by fields
              FieldMaskSet<InstanceView> &add_set = to_add[diff];
              std::vector<InstanceView*> to_erase;
              for (FieldMaskSet<InstanceView>::iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                const FieldMask overlap = filter_mask & it->second;
                if (!overlap)
                  continue;
                if (add_set.insert(it->first, overlap))
                  it->first->add_nested_valid_ref(did);
                it.filter(overlap);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_erase.begin(); it != to_erase.end(); it++)
              {
                rit->second.erase(*it);
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(*it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[*it] = 1;
                  else
                    finder->second += 1;
                }
                else if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
              }
              if (rit->second.empty())
                to_delete.push_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
        }
        for (LegionMap<IndexSpaceExpression*,
              FieldMaskSet<InstanceView> >::iterator ait =
              to_add.begin(); ait != to_add.end(); ait++)
        {
          ExprViewMaskSets::iterator finder =
            restricted_instances.find(ait->first);
          if (finder != restricted_instances.end())
          {
            ait->first->add_nested_expression_reference(did);
            restricted_instances[ait->first].swap(ait->second);
          }
          else
          {
            for (FieldMaskSet<InstanceView>::const_iterator it =
                  ait->second.begin(); it != ait->second.end(); it++)
              // remove duplicate references
              if (!finder->second.insert(it->first, it->second) &&
                  it->first->remove_nested_valid_ref(did))
                assert(false); // should never hit this
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =  
              to_delete.begin(); it != to_delete.end(); it++)
        {
          if (to_add.find(*it) != to_add.end())
            continue;
          restricted_instances.erase(*it);
          if (expr_refs_to_remove != NULL)
          {
            std::map<IndexSpaceExpression*,unsigned>::iterator finder =
              expr_refs_to_remove->find(*it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[*it] = 1;
            else
              finder->second += 1;
          }
          else if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
        // Rebuild the restricted fields
        if (!restricted_instances.empty())
        {
          restricted_fields.clear();
          for (ExprViewMaskSets::const_iterator rit =
                restricted_instances.begin(); rit != 
                restricted_instances.end(); rit++)
            restricted_fields |= rit->second.get_valid_mask();
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_released_instances(IndexSpaceExpression *expr,
                  const bool expr_covers, const FieldMask &filter_mask,
                  std::map<IndexSpaceExpression*,unsigned> *expr_refs_to_remove,
                  std::map<LogicalView*,unsigned> *view_refs_to_remove)
    //--------------------------------------------------------------------------
    {
      if (released_instances.empty())
        return;
      if (expr_covers)
      {
        // filter fields
        std::vector<IndexSpaceExpression*> to_delete;
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
              rit != released_instances.end(); rit++)
        {
          if (!(rit->second.get_valid_mask() - filter_mask))
          {
            // delete all the views in this one
            for (FieldMaskSet<InstanceView>::const_iterator it =
                  rit->second.begin(); it != rit->second.end(); it++)
            {
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder =
                  view_refs_to_remove->find(it->first);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[it->first] = 1;
                else
                  finder->second += 1;
              }
              else if (it->first->remove_nested_valid_ref(did))
                delete it->first;
            }
            to_delete.push_back(rit->first);
          }
          else
          {
            // filter views based on fields
            std::vector<InstanceView*> to_erase;
            for (FieldMaskSet<InstanceView>::iterator it =
                  rit->second.begin(); it != rit->second.end(); it++)
            {
              it.filter(filter_mask);
              if (!it->second)
                to_erase.push_back(it->first);
            }
            for (std::vector<InstanceView*>::const_iterator it =
                  to_erase.begin(); it != to_erase.end(); it++)
            {
              if (view_refs_to_remove != NULL)
              {
                std::map<LogicalView*,unsigned>::iterator finder =
                  view_refs_to_remove->find(*it);
                if (finder == view_refs_to_remove->end())
                  (*view_refs_to_remove)[*it] = 1;
                else
                  finder->second += 1;
              }
              else if ((*it)->remove_nested_valid_ref(did))
                delete (*it);
            }
            if (rit->second.empty())
              to_delete.push_back(rit->first);
            else
              rit->second.tighten_valid_mask();
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =
              to_delete.begin(); it != to_delete.end(); it++)
        {
          released_instances.erase(*it);
          if (expr_refs_to_remove != NULL)
          {
            std::map<IndexSpaceExpression*,unsigned>::iterator finder =
              expr_refs_to_remove->find(*it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[*it] = 1;
            else
              finder->second += 1;
          }
          else if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
      }
      else
      {
        // Expression does not cover this equivalence set
        std::vector<IndexSpaceExpression*> to_delete;
        LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> > to_add;
        for (ExprViewMaskSets::iterator rit = released_instances.begin();
              rit != released_instances.end(); rit++)
        {
          if (rit->second.get_valid_mask() * filter_mask)
            continue;
          IndexSpaceExpression *intersection = (rit->first == set_expr) ? expr :
            runtime->forest->intersect_index_spaces(rit->first, expr);
          const size_t volume = intersection->get_volume();
          if (volume == 0)
            continue;
          if (volume == rit->first->get_volume())
          {
            // Covers the whole expression
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // filter all of them
              for (FieldMaskSet<InstanceView>::const_iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(it->first);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[it->first] = 1;
                  else
                    finder->second += 1;
                }
                else if (it->first->remove_nested_valid_ref(did))
                  delete it->first;
              }
              to_delete.push_back(rit->first);
            }
            else
            {
              // fitler by fields
              std::vector<InstanceView*> to_erase;
              for (FieldMaskSet<InstanceView>::iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                it.filter(filter_mask);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_erase.begin(); it != to_erase.end(); it++)
              {
                rit->second.erase(*it);
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(*it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[*it] = 1;
                  else
                    finder->second += 1;
                }
                else if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
              }
              if (rit->second.empty())
                to_delete.push_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
          else
          {
            // Only covers part, so compute diff and put them in the add set
            IndexSpaceExpression *diff = 
              runtime->forest->subtract_index_spaces(rit->first, intersection);
            if (!(rit->second.get_valid_mask() - filter_mask))
            {
              // All the views are flowing into to_add
              LegionMap<IndexSpaceExpression*,
                FieldMaskSet<InstanceView> >::iterator finder = 
                  to_add.find(diff);
              if (finder != to_add.end())
              {
                // Deduplicate references in to add
                for (FieldMaskSet<InstanceView>::const_iterator it =
                      rit->second.begin(); it != rit->second.end(); it++)
                  if (!finder->second.insert(it->first, it->second) &&
                      it->first->remove_nested_valid_ref(did))
                    assert(false); // should never hit this
              }
              else
                to_add[diff].swap(rit->second);
              to_delete.push_back(rit->first);
            }
            else
            {
              // Filter by fields
              FieldMaskSet<InstanceView> &add_set = to_add[diff];
              std::vector<InstanceView*> to_erase;
              for (FieldMaskSet<InstanceView>::iterator it =
                    rit->second.begin(); it != rit->second.end(); it++)
              {
                const FieldMask overlap = filter_mask & it->second;
                if (!overlap)
                  continue;
                if (add_set.insert(it->first, overlap))
                  it->first->add_nested_valid_ref(did);
                it.filter(overlap);
                if (!it->second)
                  to_erase.push_back(it->first);
              }
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_erase.begin(); it != to_erase.end(); it++)
              {
                rit->second.erase(*it);
                if (view_refs_to_remove != NULL)
                {
                  std::map<LogicalView*,unsigned>::iterator finder =
                    view_refs_to_remove->find(*it);
                  if (finder == view_refs_to_remove->end())
                    (*view_refs_to_remove)[*it] = 1;
                  else
                    finder->second += 1;
                }
                else if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
              }
              if (rit->second.empty())
                to_delete.push_back(rit->first);
              else
                rit->second.tighten_valid_mask();
            }
          }
        }
        for (LegionMap<IndexSpaceExpression*,
              FieldMaskSet<InstanceView> >::iterator ait =
              to_add.begin(); ait != to_add.end(); ait++)
        {
          ExprViewMaskSets::iterator finder =
            released_instances.find(ait->first);
          if (finder != released_instances.end())
          {
            ait->first->add_nested_expression_reference(did);
            released_instances[ait->first].swap(ait->second);
          }
          else
          {
            for (FieldMaskSet<InstanceView>::const_iterator it =
                  ait->second.begin(); it != ait->second.end(); it++)
              // remove duplicate references
              if (!finder->second.insert(it->first, it->second) &&
                  it->first->remove_nested_valid_ref(did))
                assert(false); // should never hit this
          }
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it =  
              to_delete.begin(); it != to_delete.end(); it++)
        {
          if (to_add.find(*it) != to_add.end())
            continue;
          released_instances.erase(*it);
          if (expr_refs_to_remove != NULL)
          {
            std::map<IndexSpaceExpression*,unsigned>::iterator finder =
              expr_refs_to_remove->find(*it);
            if (finder == expr_refs_to_remove->end())
              (*expr_refs_to_remove)[*it] = 1;
            else
              finder->second += 1;
          }
          else if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_across_copies(CopyAcrossAnalysis &analysis,
                                             const FieldMask &src_mask, 
                                             IndexSpaceExpression *expr,
                                             const bool expr_covers,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // While you might think this could a read-only lock since
      // we're just reading meta-data, that's not quite right because
      // we need exclusive access to data structures in check_for_migration
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, src_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, src_mask, deferral_events,
                             applied_events, false/*exclusive*/))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(src_mask));
#endif
      // We need to lock the analysis at this point
      AutoLock a_lock(analysis);
      check_for_uninitialized_data(analysis, expr, expr_covers, 
                                   src_mask, applied_events); 
      // TODO: Handle the case where we are predicated
      if (analysis.pred_guard.exists())
        assert(false);
      // See if there are any other predicate guard fields that we need
      // to have as preconditions before applying our owner updates
      if (!read_only_guards.empty() && 
          !(src_mask * read_only_guards.get_valid_mask()))
      {
        for (FieldMaskSet<CopyFillGuard>::iterator it = 
              read_only_guards.begin(); it != read_only_guards.end(); it++)
        {
          if (src_mask * it->second)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          analysis.guard_events.insert(it->first->effects_applied);
#else
          if (analysis.original_source == local_space)
            analysis.guard_events.insert(it->first->guard_postcondition);
          else
            analysis.guard_events.insert(it->first->effects_applied);
#endif
        }
      }
      // At this point we know we're going to need an aggregator since
      // this is an across copy and we have to be doing updates
      CopyFillAggregator *across_aggregator = analysis.get_across_aggregator();
      if (!analysis.perfect)
      {
        // The general case where fields don't align regardless of
        // whether we are doing a reduction across or not
#ifdef DEBUG_LEGION
        assert(!analysis.src_indexes.empty());
        assert(!analysis.dst_indexes.empty());
        assert(analysis.src_indexes.size() == analysis.dst_indexes.size());
        assert(analysis.across_helpers.size() == 
                analysis.target_instances.size());
#endif
        // First construct a map from dst indexes to src indexes 
        std::map<unsigned,unsigned> dst_to_src;
        for (unsigned idx = 0; idx < analysis.src_indexes.size(); idx++)
          dst_to_src[analysis.dst_indexes[idx]] = analysis.src_indexes[idx];
        // We want to group all the target views with their across helpers
        // so that we can issue them in bulk
        LegionMap<CopyAcrossHelper*,FieldMaskSet<InstanceView> > target_insts;
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          const FieldMask &dst_mask =
            analysis.target_instances[idx].get_valid_fields();
          // Compute a tmp mask based on the dst mask
          FieldMask source_mask;
          int fidx = dst_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<unsigned,unsigned>::const_iterator finder = 
              dst_to_src.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != dst_to_src.end());
#endif
            source_mask.set_bit(finder->second);
            fidx = dst_mask.find_next_set(fidx+1);
          }
          // This might not be the right equivalence set for all the
          // target instances, so filter down to the ones we apply to
          const FieldMask overlap = src_mask & source_mask;
          if (!overlap)
            continue;
          target_insts[analysis.across_helpers[idx]].insert(
              analysis.target_views[idx], overlap);
        }
#ifdef DEBUG_LEGION
        assert(!target_insts.empty());
#endif
        for (LegionMap<CopyAcrossHelper*,
              FieldMaskSet<InstanceView> >::const_iterator it = 
              target_insts.begin(); it != target_insts.end(); it++)
        {
          make_instances_valid(across_aggregator, NULL/*no guard*/,
              analysis.op, analysis.src_index, true/*track events*/, expr,
              expr_covers, it->second.get_valid_mask(), it->second, 
              analysis.source_views, analysis.trace_info,
              true/*skip check*/, analysis.dst_index, analysis.redop,it->first);
          // Only need to check for reductions if we're not reducing since
          // the runtime prevents reductions-across with different reduction ops
          if ((analysis.redop == 0) && !!reduction_fields)
          {
            const FieldMask reduction_mask = reduction_fields &
              it->second.get_valid_mask();
            if (!!reduction_mask)
              apply_reductions(it->second, expr, expr_covers, reduction_mask,
                  across_aggregator, NULL/*no guard*/, analysis.op,
                  analysis.index, true/*track events*/, analysis.trace_info, 
                  NULL/*no applied exprs*/, it->first);
          }
        }
      }
      else
      {
        // Fields align when doing this copy across so use the general path
        FieldMaskSet<InstanceView> target_instances;
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          // This might not be the right equivalence set for all the
          // target instances, so filter down to the ones we apply to
          const FieldMask mask = src_mask &
            analysis.target_instances[idx].get_valid_fields();
          if (!mask)
            continue;
          target_instances.insert(analysis.target_views[idx], mask);
        }
#ifdef DEBUG_LEGION
        assert(!target_instances.empty());
#endif
        make_instances_valid(across_aggregator, NULL/*no guard*/,
            analysis.op, analysis.src_index, true/*track events*/, expr, 
            expr_covers, src_mask, target_instances, analysis.source_views,
            analysis.trace_info, true/*skip check*/, 
            analysis.dst_index, analysis.redop);
        // Only need to check for reductions if we're not reducing since
        // the runtime prevents reductions-across with different reduction ops
        if ((analysis.redop == 0) && !!reduction_fields)
        {
          const FieldMask reduction_mask = src_mask & reduction_fields;
          if (!!reduction_mask)
            apply_reductions(target_instances, expr, expr_covers, 
                reduction_mask, across_aggregator, NULL/*no guard*/,
                analysis.op, analysis.index, true/*track events*/, 
                analysis.trace_info, NULL/*no need to track applied exprs*/);
        }
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::overwrite_set(OverwriteAnalysis &analysis, 
                                       IndexSpaceExpression *expr, 
                                       const bool expr_covers, 
                                       const FieldMask &overwrite_mask,
                                       std::set<RtEvent> &deferral_events,
                                       std::set<RtEvent> &applied_events,
                                       const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, overwrite_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, overwrite_mask, deferral_events,
                             applied_events, expr_covers))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(overwrite_mask));
#endif
      // Now that we're ready to perform the analysis 
      // we need to lock the analysis 
      AutoLock a_lock(analysis);
      if (analysis.output_aggregator != NULL)
        analysis.output_aggregator->clear_update_fields();
      // Two different cases here depending on whether we have a precidate 
      if (analysis.pred_guard.exists())
      {
#ifdef DEBUG_LEGION
        assert(!analysis.add_restriction); // shouldn't be doing this
#endif
        // We have a predicate so collapse everything to all the valid
        // instances and then do predicate fills to all those instances
        assert(false);
      }
      else
      {
        // In all cases we're going to remove any reductions we've overwriting
        const FieldMask reduce_filter = reduction_fields & overwrite_mask;
        if (!!reduce_filter)
          filter_reduction_instances(expr, expr_covers, reduce_filter);
        if (analysis.add_restriction || 
            !restricted_fields || (restricted_fields * overwrite_mask))
        {
          // Easy case, just filter everything and add the new view
          filter_valid_instances(expr, expr_covers, overwrite_mask);
          if (!analysis.views.empty())
            record_instances(expr, expr_covers, overwrite_mask, 
                             analysis.views);
        }
        else
        {
          // We overlap with some restricted fields so we can't filter
          // or update any restricted fields
          const FieldMask restricted_mask = overwrite_mask & restricted_fields;
          filter_unrestricted_instances(expr, expr_covers, 
                                        restricted_mask);
          const FieldMask non_restricted = overwrite_mask - restricted_mask;
          if (!!non_restricted)
            filter_valid_instances(expr, expr_covers, non_restricted);
          if (!analysis.views.empty())
          {
            record_unrestricted_instances(expr, expr_covers, restricted_mask,
                                          analysis.views);
            copy_out(expr, expr_covers, restricted_mask, analysis.views,
                     analysis.op, analysis.index, analysis.trace_info,
                     analysis.output_aggregator);
            if (!!non_restricted)
              record_instances(expr, expr_covers, non_restricted, 
                               analysis.views);
          }
        }
        if (!analysis.reduction_views.empty())
        {
          for (FieldMaskSet<ReductionView>::const_iterator it =
                analysis.reduction_views.begin(); it != 
                analysis.reduction_views.end(); it++)
          {
            int fidx = it->second.find_first_set();
            while (fidx >= 0)
            {
              reduction_instances[fidx].push_back(
                  std::make_pair(it->first, expr)); 
              it->first->add_nested_valid_ref(did);
              expr->add_nested_expression_reference(did);
              fidx = it->second.find_next_set(fidx+1);
            }
          }
          reduction_fields |= analysis.reduction_views.get_valid_mask();
        }
        if (analysis.add_restriction)
        {
#ifdef DEBUG_LEGION
          assert(analysis.views.size() == 1);
          FieldMaskSet<LogicalView>::const_iterator it = analysis.views.begin();
          LogicalView *log_view = it->first;
          assert(log_view->is_instance_view());
          assert(it->second == overwrite_mask);
#else
          LogicalView *log_view = analysis.views.begin()->first;
#endif
          InstanceView *inst_view = log_view->as_instance_view();
          record_restriction(expr,expr_covers,overwrite_mask,inst_view);
          if (tracing_postconditions != NULL)
            tracing_postconditions->invalidate_all_but(inst_view, expr,
                                                       overwrite_mask);
        }
      }
      // Record that there is initialized data for this equivalence set
      update_initialized_data(expr, expr_covers, overwrite_mask); 
      if (analysis.trace_info.recording)
      {
        if (tracing_postconditions == NULL)
          tracing_postconditions = 
            new TraceViewSet(runtime->forest, did, region_node);
        const RegionUsage usage(LEGION_WRITE_PRIV, LEGION_EXCLUSIVE, 0);
        for (FieldMaskSet<LogicalView>::const_iterator it =
              analysis.views.begin(); it != analysis.views.end(); it++)
          update_tracing_valid_views(it->first, expr, usage,
             it->second, true/*invalidates*/);
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_set(FilterAnalysis &analysis, 
                                    IndexSpaceExpression *expr, 
                                    const bool expr_covers, 
                                    const FieldMask &filter_mask, 
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, filter_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, filter_mask, deferral_events,
                             applied_events, expr_covers))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(filter_mask));
#endif
      // No need to lock the analysis here since we're not going to change it
      // Filter partial first since total could flow back here
      ViewExprMaskSets::iterator part_finder =
        partial_valid_instances.find(analysis.inst_view);
      if (part_finder != partial_valid_instances.end())
      {
        FieldMask part_overlap = 
          part_finder->second.get_valid_mask() & filter_mask;
        if (!!part_overlap)
        {
          FieldMaskSet<IndexSpaceExpression> to_add;
          std::vector<IndexSpaceExpression*> to_delete;
          for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
               part_finder->second.begin(); it != 
               part_finder->second.end(); it++)
          {
            const FieldMask overlap = it->second & part_overlap;
            if (!overlap)
              continue;
            if (!expr_covers && (it->first != expr))
            {
              IndexSpaceExpression *expr_overlap = 
                runtime->forest->intersect_index_spaces(it->first, expr);
              const size_t expr_size = expr_overlap->get_volume();
              if (expr_size == 0)
                continue;
              if (expr_size < it->first->get_volume())
              {
                IndexSpaceExpression *diff_expr = 
                  runtime->forest->subtract_index_spaces(it->first, 
                                                         expr_overlap);
#ifdef DEBUG_LEGION
                assert(!diff_expr->is_empty());
#endif
                to_add.insert(diff_expr, overlap);
              }
            }
            // cover at least some if it so this expression will be removed
            it.filter(overlap);
            if (!it->second)
              to_delete.push_back(it->first);
            // Field overlaps should only occur once
            part_overlap -= overlap;
            if (!part_overlap)
              break;
          }
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                to_add.begin(); it != to_add.end(); it++)
            if (part_finder->second.insert(it->first, it->second))
              it->first->add_nested_expression_reference(did);
          // Deletions after adds to keep references around
          if (!to_delete.empty())
          {
            for (std::vector<IndexSpaceExpression*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
            {
              FieldMaskSet<IndexSpaceExpression>::iterator finder =
                part_finder->second.find(*it);
#ifdef DEBUG_LEGION
              assert(finder != part_finder->second.end());
#endif
              if (!!finder->second)
                continue;
              part_finder->second.erase(*it);
              if ((*it)->remove_nested_expression_reference(did))
                delete (*it);
            }
          }
          if (part_finder->second.empty())
          {
            if (part_finder->first->remove_nested_valid_ref(did))
              delete part_finder->first;
            partial_valid_instances.erase(part_finder);
          }
          else
            part_finder->second.tighten_valid_mask();
          // Rebuild the partial valid fields
          partial_valid_fields.clear();
          if (!partial_valid_instances.empty())
          {
            for (ViewExprMaskSets::const_iterator it =
                  partial_valid_instances.begin(); it !=
                  partial_valid_instances.end(); it++)
              partial_valid_fields |= it->second.get_valid_mask();
          }
        }
      }
      FieldMaskSet<LogicalView>::iterator total_finder = 
        total_valid_instances.find(analysis.inst_view);
      if (total_finder != total_valid_instances.end())
      {
        const FieldMask total_overlap = total_finder->second & filter_mask;
        if (!!total_overlap)
        {
          if (!expr_covers)
          {
            // Compute the difference and store it in the partial valid fields    
            IndexSpaceExpression *diff_expr = 
              runtime->forest->subtract_index_spaces(set_expr, expr);
#ifdef DEBUG_LEGION
            assert(!diff_expr->is_empty());
#endif
            if (record_partial_valid_instance(analysis.inst_view, diff_expr,
                        total_overlap, false/*check total valid*/))
            {
              // Need to rebuild the partial valid fields
              partial_valid_fields.clear();
              for (ViewExprMaskSets::const_iterator it =
                    partial_valid_instances.begin(); it !=
                    partial_valid_instances.end(); it++)
                partial_valid_fields |= it->second.get_valid_mask();
            }
          }
          total_finder.filter(total_overlap);
          if (!total_finder->second)
          {
            if (total_finder->first->remove_nested_valid_ref(did))
              delete total_finder->first;
            total_valid_instances.erase(total_finder);
          }
          total_valid_instances.tighten_valid_mask();
        }
      }
      if (analysis.remove_restriction)
      {
#ifdef DEBUG_LEGION
        assert(analysis.inst_view != NULL);
#endif
        FieldMaskSet<IndexSpaceExpression> to_add;
        std::vector<IndexSpaceExpression*> to_delete;
        for (ExprViewMaskSets::iterator rit = restricted_instances.begin();
              rit != restricted_instances.end(); rit++)
        {
          FieldMaskSet<InstanceView>::iterator finder = 
            rit->second.find(analysis.inst_view);
          if (finder == rit->second.end())
            continue;
          const FieldMask overlap = finder->second & filter_mask;
          if (!overlap)
            continue;
          if (!expr_covers && (rit->first != expr))
          {
            IndexSpaceExpression *expr_overlap = 
              runtime->forest->intersect_index_spaces(rit->first, expr);
            const size_t overlap_size = expr_overlap->get_volume();
            if (overlap_size == 0)
              continue;
            if (overlap_size < rit->first->get_volume())
            {
              // Did not cover all of it so we have to compute the diff
              IndexSpaceExpression *diff_expr = 
                runtime->forest->subtract_index_spaces(rit->first, expr);
#ifdef DEBUG_LEGION
              assert(diff_expr != NULL);
#endif
              to_add.insert(diff_expr, overlap); 
            }
          }
          // If we get here, then we're definitely removing this 
          // restricted instances from this 
          finder.filter(overlap);
          if (!finder->second)
          {
            if (finder->first->remove_nested_valid_ref(did))
              delete finder->first;
            rit->second.erase(finder);
            if (rit->second.empty())
              to_delete.push_back(rit->first);
          }
        }
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              to_add.begin(); it != to_add.end(); it++)
        {
          ExprViewMaskSets::iterator finder =
            restricted_instances.find(it->first);
          if (finder == restricted_instances.end())
          {
            it->first->add_nested_expression_reference(did);
            analysis.inst_view->add_nested_valid_ref(did);
            restricted_instances[it->first].insert(analysis.inst_view, 
                                                   it->second);
          }
          else if (finder->second.insert(analysis.inst_view, it->second))
            analysis.inst_view->add_nested_valid_ref(did);
        }
        for (std::vector<IndexSpaceExpression*>::const_iterator it = 
              to_delete.begin(); it != to_delete.end(); it++)
        {
          ExprViewMaskSets::iterator finder =
            restricted_instances.find(*it);
#ifdef DEBUG_LEGION
          assert(finder != restricted_instances.end());
#endif
          // Check to see if it is still empty since we added things back
          if (!finder->second.empty())
            continue;
          restricted_instances.erase(finder);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
        // Rebuild the restricted fields
        restricted_fields.clear();
        for (ExprViewMaskSets::const_iterator it = restricted_instances.begin();
              it != restricted_instances.end(); it++)
          restricted_fields |= it->second.get_valid_mask();
        // If the data was restricted then we just removed the only
        // valid copy so we need to filter the initialized data
        filter_initialized_data(expr, expr_covers, filter_mask);
      }
      else
      {
        // Check to see if we still have initialized data for what we filtered
        if (!total_valid_instances.empty() || !partial_valid_instances.empty())
        {
          FieldMask to_check =
            filter_mask - total_valid_instances.get_valid_mask();
          if (!!to_check)
          {
            const FieldMask no_partial = to_check - partial_valid_fields;
            if (!!no_partial)
            {
              filter_initialized_data(expr, expr_covers, no_partial);
              to_check -= no_partial;
            }
            if (!!to_check)
            {
              FieldMaskSet<IndexSpaceExpression> to_filter;
              to_filter.insert(expr, to_check);
              for (ViewExprMaskSets::const_iterator pit =
                    partial_valid_instances.begin(); pit !=
                    partial_valid_instances.end(); pit++)
              {
                if (to_check * pit->second.get_valid_mask())
                  continue;
                LegionMap<std::pair<IndexSpaceExpression*,
                  IndexSpaceExpression*>,FieldMask> filter_sets;
                unique_join_on_field_mask_sets(to_filter, 
                                pit->second, filter_sets);
                for (LegionMap<std::pair<IndexSpaceExpression*,
                      IndexSpaceExpression*>,FieldMask>::const_iterator
                      it = filter_sets.begin(); it != filter_sets.end(); it++)
                {
                  IndexSpaceExpression *diff = 
                    runtime->forest->subtract_index_spaces(
                        it->first.first, it->first.second);
                  if (diff->get_volume() == it->first.first->get_volume())
                    continue;
                  FieldMaskSet<IndexSpaceExpression>::iterator finder =
                    to_filter.find(it->first.first);
#ifdef DEBUG_LEGION
                  assert(finder != to_filter.end());
#endif
                  finder.filter(it->second);
                  if (!finder->second)
                    to_filter.erase(finder);
                  if (!diff->is_empty())
                    to_filter.insert(diff, it->second);
                  else
                    to_check -= it->second;
                }
                if (to_filter.empty())
                  break;
              }
              if (!to_filter.empty())
              {
                for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                      to_filter.begin(); it != to_filter.end(); it++)
                {
                  const bool covers =
                    (it->first->get_volume() == set_expr->get_volume());
                  filter_initialized_data(it->first, covers, it->second);
                }
              }
            }
          }
        }
        else // everything empty so filter the whole set
          filter_initialized_data(set_expr,true/*covers*/, filter_mask);
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_set(CloneAnalysis &analysis, 
                                   IndexSpaceExpression *expr, 
                                   const bool expr_covers, 
                                   const FieldMask &clone_mask, 
                                   std::set<RtEvent> &deferral_events,
                                   std::set<RtEvent> &applied_events,
                                   const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, clone_mask, deferral_events,
                        applied_events, already_deferred);
        return;
      }
      if (is_remote_analysis(analysis, clone_mask, deferral_events,
                             applied_events, expr_covers))
        return;
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner() || has_replicated_fields(clone_mask));
#endif
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            analysis.sources.begin(); it != analysis.sources.end(); it++)
      {
        // Check that the fields overlap 
        const FieldMask overlap = clone_mask & it->second;
        if (!overlap)
          continue;
        // Check that the expressions overlap
        IndexSpaceExpression *overlap_expr = 
          runtime->forest->intersect_index_spaces(expr,
              it->first->region_node->row_source);
        if (overlap_expr->is_empty())
          continue;
        it->first->clone_to_local(this, overlap, applied_events,
            false/*invalidate overlap*/, true/*forward to owner*/);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_valid_views(LogicalView *view,
                                                    IndexSpaceExpression *expr,
                                                    const RegionUsage &usage,
                                                    const FieldMask &user_mask,
                                                    const bool invalidates)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we should be called from a copy
      // fill aggregator that is being built while already holding the lock
      if (HAS_READ(usage) && !IS_DISCARD(usage))
      {
        FieldMaskSet<IndexSpaceExpression> not_dominated;
        if (tracing_postconditions != NULL)
          tracing_postconditions->dominates(view,expr,user_mask,not_dominated);
        else
          not_dominated.insert(expr, user_mask);
        if (tracing_preconditions == NULL)
          tracing_preconditions = 
            new TraceViewSet(runtime->forest, did, region_node);
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              not_dominated.begin(); it != not_dominated.end(); it++)
          tracing_preconditions->insert(view, it->first, it->second);
        if (view->is_reduction_view())
        {
          // Invalidate this reduction view since we read it
          if (tracing_postconditions != NULL)
            tracing_postconditions->invalidate(view, expr, user_mask);
          return;
        }
        // Do not record read-only postconditions
        if (IS_READ_ONLY(usage))
          return;
      }
#ifdef DEBUG_LEGION
      assert(HAS_WRITE(usage));
#endif
      if (tracing_postconditions != NULL)
      {
        if (invalidates)
          tracing_postconditions->invalidate_all_but(view, expr, user_mask);
      }
      else
        tracing_postconditions = 
          new TraceViewSet(runtime->forest, did, region_node);
      tracing_postconditions->insert(view, expr, user_mask);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_tracing_anti_views(LogicalView *view,
                                                   IndexSpaceExpression *expr, 
                                                   const FieldMask &mask) 
    //--------------------------------------------------------------------------
    {
      if (tracing_anticonditions == NULL)
        tracing_anticonditions =
          new TraceViewSet(runtime->forest, did, region_node);
      tracing_anticonditions->insert(view, expr, mask);
    }

    //--------------------------------------------------------------------------
    RtEvent EquivalenceSet::capture_trace_conditions(TraceConditionSet *target,
                        AddressSpaceID target_space, IndexSpaceExpression *expr,
                        const FieldMask &mask, RtUserEvent ready_event)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);    
      // This always needs to be sent to the owner to handle the case where
      // we are figuring out which shard owns each precondition expression
      // We can only deduplicate if they go to the same place
      if (!is_logical_owner())
      {
        if (!ready_event.exists())
          ready_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(target);
          rez.serialize(target_space);
          expr->pack_expression(rez, logical_owner_space);
          rez.serialize(mask);
          rez.serialize(ready_event);
        }
        runtime->send_equivalence_set_capture_request(logical_owner_space, rez);
        return ready_event;
      }
      // If we get here then we are the ones to do the analysis
      TraceViewSet *previews = NULL;
      TraceViewSet *antiviews = NULL;
      TraceViewSet *postviews = NULL;
      // Compute the views to send back
      if (tracing_preconditions != NULL)
      {
        previews = 
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        tracing_preconditions->find_overlaps(*previews, expr, 
                                             (expr == set_expr), mask);
      }
      if (tracing_anticonditions != NULL)
      {
        antiviews =
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        tracing_anticonditions->find_overlaps(*antiviews, expr,
                                             (expr == set_expr), mask);
      }
      if (tracing_postconditions != NULL)
      {
        postviews =
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        tracing_postconditions->find_overlaps(*postviews, expr,
                                             (expr == set_expr), mask);
      }
      // Return the results
      RtEvent result = ready_event;
      if (target_space != local_space)
      {
#ifdef DEBUG_LEGION
        assert(ready_event.exists());
#endif
        // Send back the results to the target node
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(region_node->handle);
          if (previews != NULL)
            previews->pack(rez, target_space);
          else
            rez.serialize<size_t>(0);
          if (antiviews != NULL)
            antiviews->pack(rez, target_space);
          else
            rez.serialize<size_t>(0);
          if (postviews != NULL)
            postviews->pack(rez, target_space);
          else
            rez.serialize<size_t>(0);
          rez.serialize(ready_event);
        }
        runtime->send_equivalence_set_capture_response(target_space, rez);
        if (previews != NULL)
          delete previews;
        if (antiviews != NULL)
          delete antiviews;
        if (postviews != NULL)
          delete postviews;
      }
      else
      {
        std::set<RtEvent> ready_events;
        target->receive_capture(previews, antiviews, postviews, ready_events);
        if (!ready_events.empty())
        {
          if (ready_event.exists())
            Runtime::trigger_event(ready_event, 
                Runtime::merge_events(ready_events));
          else
            result = Runtime::merge_events(ready_events);
        }
        else if (ready_event.exists())
          Runtime::trigger_event(ready_event);
      }
      if (tracing_preconditions != NULL)
      {
        tracing_preconditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_preconditions->empty())
        {
          delete tracing_preconditions;
          tracing_preconditions = NULL;
        }
      }
      if (tracing_anticonditions != NULL)
      {
        tracing_anticonditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_anticonditions->empty())
        {
          delete tracing_anticonditions;
          tracing_anticonditions = NULL;
        }
      }
      if (tracing_postconditions != NULL)
      {
        tracing_postconditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_postconditions->empty())
        {
          delete tracing_postconditions;
          tracing_postconditions = NULL;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_read_only_guard(CopyFillGuard *guard)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're no longer the logical owner then it's because we were
      // migrated and there should be no guards so we're done
      if (read_only_guards.empty())
        return;
      // We could get here when we're not the logical owner if we've unpacked
      // ourselves but haven't become the owner yet, in which case we still
      // need to prune ourselves out of the list
      FieldMaskSet<CopyFillGuard>::iterator finder = 
        read_only_guards.find(guard);
      // It's also possible that the equivalence set is migrated away and
      // then migrated back before this guard is removed in which case we
      // won't find it in the update guards and can safely ignore it
      if (finder == read_only_guards.end())
        return;
      const bool should_tighten = !!finder->second;
      read_only_guards.erase(finder);
      if (should_tighten)
        read_only_guards.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_reduction_fill_guard(CopyFillGuard *guard)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're no longer the logical owner then it's because we were
      // migrated and there should be no guards so we're done
      if (reduction_fill_guards.empty())
        return;
      // We could get here when we're not the logical owner if we've unpacked
      // ourselves but haven't become the owner yet, in which case we still
      // need to prune ourselves out of the list
      FieldMaskSet<CopyFillGuard>::iterator finder = 
        reduction_fill_guards.find(guard);
      // It's also possible that the equivalence set is migrated away and
      // then migrated back before this guard is removed in which case we
      // won't find it in the update guards and can safely ignore it
      if (finder == reduction_fill_guards.end())
        return;
      const bool should_tighten = !!finder->second;
      reduction_fill_guards.erase(finder);
      if (should_tighten)
        reduction_fill_guards.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_replication_request(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready_event;
      EquivalenceSet *set = 
        runtime->find_or_request_equivalence_set(did, ready_event);
      FieldMask mask;
      derez.deserialize(mask);
      size_t total_spaces;
      derez.deserialize(total_spaces);
      CollectiveMapping *mapping = new CollectiveMapping(derez, total_spaces);
      mapping->add_reference();
      PendingReplication *target;
      derez.deserialize(target);
      AddressSpaceID source;
      derez.deserialize(source);
      RtEvent done_event;
      derez.deserialize(done_event);

      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      set->process_replication_request(mask, mapping, target,source,done_event);
      if (mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_replication_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready_event;
      EquivalenceSet *set = 
        runtime->find_or_request_equivalence_set(did, ready_event);
      FieldMask mask;
      derez.deserialize(mask);
      PendingReplication *target;
      derez.deserialize(target);
      RtEvent precondition;
      derez.deserialize(precondition);
      FieldMask update_mask;
      derez.deserialize(update_mask);

      if (precondition.exists())
        target->preconditions.insert(precondition);
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      set->process_replication_response(target, mask, precondition, 
                                        update_mask, derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_replication_update(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready_event;
      EquivalenceSet *set = 
        runtime->find_or_request_equivalence_set(did, ready_event);
      FieldMask mask;
      derez.deserialize(mask);
      size_t total_spaces;
      derez.deserialize(total_spaces);
      CollectiveMapping *mapping = NULL;
      if (total_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, total_spaces);
        mapping->add_reference();
      }
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      std::set<RtEvent> applied_events;
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      set->broadcast_replicated_state_updates(mask, mapping, origin,
          applied_events, true/*need lock*/);
      if (!applied_events.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
      if ((mapping != NULL) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::send_equivalence_set(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      // We should have had a request for this already
      assert(!has_remote_instance(target));
      assert((collective_mapping == NULL) || 
              !collective_mapping->contains(target));
#endif
      update_remote_instances(target);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(region_node->handle);
        // There be dragons here!
        // In the case where we first make a new equivalence set on a
        // remote node that is about to be the owner, we can't mark it
        // as the owner until it receives all an unpack_state or 
        // unpack_migration message which provides it valid meta-data
        // Therefore we'll tell it that we're the owner which will 
        // create a cycle in the forwarding graph. This won't matter for
        // unpack_migration as it's going to overwrite the data in the
        // equivalence set anyway, but for upack_state, we'll need to 
        // recognize when to break the cycle. Effectively whenever we
        // send an update to a remote node that we can tell has never
        // been the owner before (and therefore can't have migrated)
        // we know that we should just do the unpack there. This will
        // break the cycle and allow forward progress. Analysis messages
        // may go round and round a few times, but they have lower
        // priority and therefore shouldn't create a livelock.
        AutoLock eq(eq_lock,1,false/*exclusive*/);
        // is_ready tests whether this set expression has been set
        // it might not be in the case of an output region and we
        // don't want to block testing is_empty in that case if it 
        // hasn't been set
        if (set_expr->is_set() && !set_expr->is_empty())
        {
          if (target == logical_owner_space)
            rez.serialize(local_space);
          else
            rez.serialize(logical_owner_space);
        }
        else
          rez.serialize(logical_owner_space);
        // Also pack up any replicated states to send
        rez.serialize<size_t>(replicated_states.size());
        for (FieldMaskSet<CollectiveMapping>::const_iterator it =
              replicated_states.begin(); it != replicated_states.end(); it++)
        {
          it->first->pack(rez);
          rez.serialize(it->second);
        }
      }
      runtime->send_equivalence_set_response(target, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_equivalence_set_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->send_equivalence_set(source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_equivalence_set_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      LogicalRegion handle;
      derez.deserialize(handle);
      RegionNode *node = runtime->forest->get_node(handle);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);

      void *location;
      EquivalenceSet *set = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        set = new(location) EquivalenceSet(runtime, did, logical_owner,
                                           node, false/*register now*/);
      else
        set = new EquivalenceSet(runtime, did, logical_owner,
                                 node, false/*register now*/);
      set->unpack_replicated_states(derez);
      // Once construction is complete then we do the registration
      set->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_replicated_states(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_states;
      derez.deserialize(num_states);
      for (unsigned idx = 0; idx < num_states; idx++)
      {
        size_t total_spaces;
        derez.deserialize(total_spaces);
        CollectiveMapping *mapping = new CollectiveMapping(derez, total_spaces);
        FieldMask mask;
        derez.deserialize(mask);
        if (replicated_states.insert(mapping, mask))
          mapping->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_from(const AddressSpaceID target_space,
                                    EquivalenceSet *src, const FieldMask &mask,
                                    const bool forward_to_owner,
                                    std::set<RtEvent> &applied_events,
                                    const bool invalidate_overlap)
    //--------------------------------------------------------------------------
    {
      if (this == src)
      {
        // Empty equivalence sets can sometimes be asked to clone to themself
#ifdef DEBUG_LEGION
        assert(region_node->row_source->is_empty());
#endif
        return;
      }
      if (target_space != local_space)
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        src->clone_to_remote(did, target_space, region_node->row_source,
                 mask, done_event, invalidate_overlap, forward_to_owner);
        applied_events.insert(done_event);
      }
      else
      {
        AutoLock eq(eq_lock); 
        src->clone_to_local(this, mask, applied_events, 
                            invalidate_overlap, forward_to_owner);  
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_owner(const AddressSpaceID new_logical_owner)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
      // We should never be told that we're the new owner this way
      assert(new_logical_owner != local_space);
#endif
      // If we are the owner then we know this update is stale so ignore it
      if (!is_logical_owner())
        logical_owner_space = new_logical_owner;
    }

    //--------------------------------------------------------------------------
    RtEvent EquivalenceSet::make_owner(AddressSpaceID new_owner, RtEvent pre)
    //--------------------------------------------------------------------------
    {
      if (new_owner != local_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(pre);
          rez.serialize(done);
        }
        runtime->send_equivalence_set_make_owner(new_owner, rez);
        return done;
      }
      if (pre.exists() && !pre.has_triggered())
      {
        const DeferMakeOwnerArgs args(this);
        return runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, pre);
      }
      // If we make it here then we can finally mark ourselves the owner
      AutoLock eq(eq_lock);
      logical_owner_space = local_space;
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_make_owner(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMakeOwnerArgs *dargs = (const DeferMakeOwnerArgs*)args;
      dargs->set->make_owner(dargs->set->local_space, RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_make_owner(Deserializer &derez,
                                                      Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      RtEvent precondition;
      derez.deserialize(precondition);
      RtUserEvent done;
      derez.deserialize(done);

      if ((ready.exists() && !ready.has_triggered()) ||
          (precondition.exists() && !precondition.has_triggered()))
      {
        const DeferMakeOwnerArgs args(set);
        Runtime::trigger_event(done,
            runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
              Runtime::merge_events(ready, precondition)));
      }
      else
        Runtime::trigger_event(done, 
            set->make_owner(set->local_space, precondition));
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_owner_update(Deserializer &derez,
                                                        Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      AddressSpaceID new_owner;
      derez.deserialize(new_owner);
      RtUserEvent done;
      derez.deserialize(done);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->update_owner(new_owner);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_migration(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);

      std::set<RtEvent> ready_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state_and_apply(derez, source, false/*forward*/,ready_events);
      // Check to see if we're ready or we need to defer this
      if (!ready_events.empty())
        set->make_owner(runtime->address_space, 
            Runtime::merge_events(ready_events));
      else
        set->make_owner(runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_state(Serializer &rez,const AddressSpaceID target,
                             IndexSpaceExpression *expr, const bool expr_covers, 
                             const FieldMask &mask, const bool pack_guards)
    //--------------------------------------------------------------------------
    {
      LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > valid_updates;
      FieldMaskSet<IndexSpaceExpression> initialized_updates;
      std::map<unsigned,std::list<
        std::pair<ReductionView*,IndexSpaceExpression*> > > reduction_updates;
      LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> >
        restricted_updates, released_updates;
      FieldMaskSet<CopyFillGuard> read_only_guards, reduction_fill_guards;
      TraceViewSet *precondition_updates = NULL;
      TraceViewSet *anticondition_updates = NULL;
      TraceViewSet *postcondition_updates = NULL;
      find_overlap_updates(expr, expr_covers, mask, valid_updates,
                           initialized_updates, reduction_updates, 
                           restricted_updates, released_updates,
                           pack_guards ? &read_only_guards : NULL, 
                           pack_guards ? &reduction_fill_guards : NULL, 
                           precondition_updates, anticondition_updates, 
                           postcondition_updates);
      pack_updates(rez, target, valid_updates, initialized_updates,
           reduction_updates, restricted_updates, released_updates, 
           &read_only_guards, &reduction_fill_guards, precondition_updates, 
           anticondition_updates, postcondition_updates);
      if (precondition_updates != NULL)
        delete precondition_updates;
      if (anticondition_updates != NULL)
        delete anticondition_updates;
      if (postcondition_updates != NULL)
        delete postcondition_updates;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::pack_updates(Serializer &rez,
              const AddressSpaceID target,
              const LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<LogicalView> > &valid_updates,
              const FieldMaskSet<IndexSpaceExpression> &initialized_updates,
              const std::map<unsigned,std::list<std::pair<ReductionView*,
                  IndexSpaceExpression*> > > &reduction_updates,
              const LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<InstanceView> > &restricted_updates,
              const LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<InstanceView> > &released_updates,
              const FieldMaskSet<CopyFillGuard> *read_only_updates,
              const FieldMaskSet<CopyFillGuard> *reduction_fill_updates,
              const TraceViewSet *precondition_updates,
              const TraceViewSet *anticondition_updates,
              const TraceViewSet *postcondition_updates)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(valid_updates.size());
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<LogicalView> >::const_iterator vit =
            valid_updates.begin(); vit != valid_updates.end(); vit++)
      {
        vit->first->pack_expression(rez, target);
        rez.serialize<size_t>(vit->second.size());
        for (FieldMaskSet<LogicalView>::const_iterator it =
              vit->second.begin(); it != vit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(initialized_updates.size());
      for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
            initialized_updates.begin(); it != initialized_updates.end(); it++)
      {
        it->first->pack_expression(rez, target);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(reduction_updates.size());
      for (std::map<unsigned,std::list<std::pair<ReductionView*,
            IndexSpaceExpression*> > >::const_iterator rit =
            reduction_updates.begin(); rit != reduction_updates.end(); rit++)
      {
        rez.serialize(rit->first);
        rez.serialize<size_t>(rit->second.size());
        for (std::list<std::pair<ReductionView*,
              IndexSpaceExpression*> >::const_iterator it =
              rit->second.begin(); it != rit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          it->second->pack_expression(rez, target);
        }
      }
      rez.serialize<size_t>(restricted_updates.size());
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >::const_iterator rit =
            restricted_updates.begin(); rit != 
            restricted_updates.end(); rit++)
      {
        rit->first->pack_expression(rez, target);
        rez.serialize<size_t>(rit->second.size());
        for (FieldMaskSet<InstanceView>::const_iterator it =
              rit->second.begin(); it != rit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(released_updates.size());
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >::const_iterator rit =
            released_updates.begin(); rit != released_updates.end(); rit++)
      {
        rit->first->pack_expression(rez, target);
        rez.serialize<size_t>(rit->second.size());
        for (FieldMaskSet<InstanceView>::const_iterator it =
              rit->second.begin(); it != rit->second.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
      }
      if ((read_only_updates != NULL) && !read_only_updates->empty())
      {
        rez.serialize<size_t>(read_only_updates->size());
        for (FieldMaskSet<CopyFillGuard>::const_iterator it =
              read_only_updates->begin(); it != read_only_updates->end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
      if ((reduction_fill_updates != NULL) && !reduction_fill_updates->empty())
      {
        rez.serialize<size_t>(reduction_fill_updates->size());
        for (FieldMaskSet<CopyFillGuard>::const_iterator it =
              reduction_fill_updates->begin(); it != 
              reduction_fill_updates->end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
      if (precondition_updates != NULL)
        precondition_updates->pack(rez, target); 
      else
        rez.serialize<size_t>(0);
      if (anticondition_updates != NULL)
        anticondition_updates->pack(rez, target); 
      else
        rez.serialize<size_t>(0);
      if (postcondition_updates != NULL)
        postcondition_updates->pack(rez, target); 
      else
        rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_state_and_apply(Deserializer &derez,
                       const AddressSpaceID source, const bool forward_to_owner,
                       std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > valid_updates;
      FieldMaskSet<IndexSpaceExpression> initialized_updates;
      std::map<unsigned,std::list<
        std::pair<ReductionView*,IndexSpaceExpression*> > > reduction_updates;
      LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> >
        restricted_updates, released_updates;
      FieldMaskSet<CopyFillGuard> read_only_updates, reduction_fill_updates;
      std::set<RtEvent> ready_events;
      size_t num_valid;
      derez.deserialize(num_valid);
      for (unsigned idx1 = 0; idx1 < num_valid; idx1++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        size_t num_views;
        derez.deserialize(num_views);
        FieldMaskSet<LogicalView> &views = valid_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView *view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          views.insert(view, mask);
        }
      }
      size_t num_initialized;
      derez.deserialize(num_initialized);
      for (unsigned idx = 0; idx < num_initialized; idx++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        FieldMask mask;
        derez.deserialize(mask);
        initialized_updates.insert(expr, mask);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      for (unsigned idx1 = 0; idx1 < num_reductions; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        size_t num_views;
        derez.deserialize(num_views);
        std::list<std::pair<ReductionView*,IndexSpaceExpression*> > 
          &reductions = reduction_updates[fidx];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView *view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          IndexSpaceExpression *expr = 
            IndexSpaceExpression::unpack_expression(derez,
                                                    runtime->forest, source);
          reductions.push_back(std::pair<ReductionView*,IndexSpaceExpression*>(
                static_cast<ReductionView*>(view), expr));
        }
      }
      size_t num_restrictions;
      derez.deserialize(num_restrictions);
      for (unsigned idx1 = 0; idx1 < num_restrictions; idx1++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        size_t num_views;
        derez.deserialize(num_views);
        FieldMaskSet<InstanceView> &restrictions = restricted_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView *view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          restrictions.insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_releases;
      derez.deserialize(num_releases);
      for (unsigned idx1 = 0; idx1 < num_releases; idx1++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        size_t num_views;
        derez.deserialize(num_views);
        FieldMaskSet<InstanceView> &releases = released_updates[expr];
        for (unsigned idx2 = 0; idx2 < num_views; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          RtEvent ready;
          LogicalView *view = runtime->find_or_request_logical_view(did, ready);
          if (ready.exists() && !ready.has_triggered())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          releases.insert(static_cast<InstanceView*>(view), mask);
        }
      }
      size_t num_read_only_guards;
      derez.deserialize(num_read_only_guards);
      if (num_read_only_guards)
      {
        // Need to hold the lock here to prevent copy fill guard
        // deletions from removing this before we've registered it
        AutoLock eq(eq_lock);
        for (unsigned idx = 0; idx < num_read_only_guards; idx++)
        {
          CopyFillGuard *guard = 
            CopyFillGuard::unpack_guard(derez, runtime, this);
          FieldMask guard_mask;
          derez.deserialize(guard_mask);
          if (guard != NULL)
          {
            read_only_guards.insert(guard, guard_mask);
            read_only_updates.insert(guard, guard_mask);
          }
        }
      }
      size_t num_reduction_fill_guards;
      derez.deserialize(num_reduction_fill_guards);
      if (num_reduction_fill_guards)
      {
        // Need to hold the lock here to prevent copy fill guard
        // deletions from removing this before we've registered it
        AutoLock eq(eq_lock);
        for (unsigned idx = 0; idx < num_reduction_fill_guards; idx++)
        {
          CopyFillGuard *guard = 
            CopyFillGuard::unpack_guard(derez, runtime, this);
          FieldMask guard_mask;
          derez.deserialize(guard_mask);
          if (guard != NULL)
          {
            reduction_fill_guards.insert(guard, guard_mask);
            reduction_fill_updates.insert(guard, guard_mask);
          }
        }
      }
      size_t num_preconditions;
      derez.deserialize(num_preconditions);
      TraceViewSet *precondition_updates = NULL;
      if (num_preconditions > 0)
      {
        precondition_updates = 
          new TraceViewSet(runtime->forest, 0/*did*/, region_node); 
        precondition_updates->unpack(derez, num_preconditions, 
                                     source, ready_events);
      }
      size_t num_anticonditions;
      derez.deserialize(num_anticonditions);
      TraceViewSet *anticondition_updates = NULL;
      if (num_anticonditions > 0)
      {
        anticondition_updates = 
          new TraceViewSet(runtime->forest, 0/*did*/, region_node); 
        anticondition_updates->unpack(derez, num_anticonditions, 
                                     source, ready_events);
      }
      size_t num_postconditions;
      derez.deserialize(num_postconditions);
      TraceViewSet *postcondition_updates = NULL;
      if (num_postconditions > 0)
      {
        postcondition_updates = 
          new TraceViewSet(runtime->forest, 0/*did*/, region_node); 
        postcondition_updates->unpack(derez, num_postconditions, 
                                     source, ready_events);
      }
      if (!ready_events.empty())
      {
        const RtEvent ready_event = Runtime::merge_events(ready_events);
        if (ready_event.exists() && !ready_event.has_triggered())
        {
          // Defer this until it is ready to be performed
          const RtUserEvent applied_event = Runtime::create_rt_user_event();
          DeferApplyStateArgs args(this, applied_event, forward_to_owner,
              applied_events, valid_updates, initialized_updates, 
              reduction_updates, restricted_updates, released_updates,
              read_only_updates, reduction_fill_updates, precondition_updates,
              anticondition_updates, postcondition_updates);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, ready_event);
          return;
        }
      }
      // All the views are ready so we can add them now
      apply_state(valid_updates, initialized_updates, reduction_updates, 
                  restricted_updates, released_updates, precondition_updates,
                  anticondition_updates, postcondition_updates, 
                  &read_only_updates, &reduction_fill_updates,
                  applied_events, true/*need lock*/, forward_to_owner);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DeferApplyStateArgs::DeferApplyStateArgs(EquivalenceSet *s,
                                       RtUserEvent done, bool forward,
                                       std::set<RtEvent> &applied_events,
                                       ExprLogicalViews &valid,
                                       FieldMaskSet<IndexSpaceExpression> &init,
                                       ExprReductionViews &reductions,
                                       ExprInstanceViews &restricted,
                                       ExprInstanceViews &released,
                                       FieldMaskSet<CopyFillGuard> &read_only,
                                       FieldMaskSet<CopyFillGuard> &reduc_fill,
                                       TraceViewSet *preconditions,
                                       TraceViewSet *anticonditions,
                                       TraceViewSet *postconditions)
      : LgTaskArgs<DeferApplyStateArgs>(implicit_provenance), set(s),
        valid_updates(new LegionMap<IndexSpaceExpression*,
            FieldMaskSet<LogicalView> >()),
        initialized_updates(new FieldMaskSet<IndexSpaceExpression>()),
        reduction_updates(new std::map<unsigned,std::list<std::pair<
            ReductionView*,IndexSpaceExpression*> > >()),
        restricted_updates(new LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >()),
        released_updates(new LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >()), 
        read_only_updates(new FieldMaskSet<CopyFillGuard>()),
        reduction_fill_updates(new FieldMaskSet<CopyFillGuard>()),
        precondition_updates(preconditions),
        anticondition_updates(anticonditions),
        postcondition_updates(postconditions),
        done_event(done), forward_to_owner(forward)
    //--------------------------------------------------------------------------
    {
      for (ExprLogicalViews::const_iterator it =
            valid.begin(); it != valid.end(); it++)
        it->first->add_base_expression_reference(META_TASK_REF);
      valid_updates->swap(valid);
      for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
            init.begin(); it != init.end(); it++)
        it->first->add_base_expression_reference(META_TASK_REF);
      initialized_updates->swap(init);
      for (ExprReductionViews::const_iterator rit =
            reductions.begin(); rit != reductions.end(); rit++)
        for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
              const_iterator it = rit->second.begin();
              it != rit->second.end(); it++)
          it->second->add_base_expression_reference(META_TASK_REF);
      reduction_updates->swap(reductions);
      for (ExprInstanceViews::const_iterator it =
            restricted.begin(); it != restricted.end(); it++)
        it->first->add_base_expression_reference(META_TASK_REF);
      restricted_updates->swap(restricted);
      for (ExprInstanceViews::const_iterator it =
            released.begin(); it != released.end(); it++)
        it->first->add_base_expression_reference(META_TASK_REF);
      released_updates->swap(released);
      read_only_updates->swap(read_only);
      reduction_fill_updates->swap(reduc_fill);
      applied_events.insert(done_event);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::DeferApplyStateArgs::release_references(void) const
    //--------------------------------------------------------------------------
    {
      for (ExprLogicalViews::const_iterator it =
            valid_updates->begin(); it != valid_updates->end(); it++)
        if (it->first->remove_base_expression_reference(META_TASK_REF))
          delete it->first;
      for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
           initialized_updates->begin(); it != initialized_updates->end(); it++)
        if (it->first->remove_base_expression_reference(META_TASK_REF))
          delete it->first;
      for (ExprReductionViews::const_iterator rit =
            reduction_updates->begin(); rit != reduction_updates->end(); rit++)
        for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
              const_iterator it = rit->second.begin(); 
              it != rit->second.end(); it++)
          if (it->second->remove_base_expression_reference(META_TASK_REF))
            delete it->second;
      for (ExprInstanceViews::const_iterator it =
            restricted_updates->begin(); it != restricted_updates->end(); it++)
        if (it->first->remove_base_expression_reference(META_TASK_REF))
          delete it->first;
      for (ExprInstanceViews::const_iterator it =
            released_updates->begin(); it != released_updates->end(); it++)
        if (it->first->remove_base_expression_reference(META_TASK_REF))
          delete it->first;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_apply_state(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferApplyStateArgs *dargs = (const DeferApplyStateArgs*)args;
      std::set<RtEvent> applied_events;
      dargs->set->apply_state(*(dargs->valid_updates), 
          *(dargs->initialized_updates), *(dargs->reduction_updates),
          *(dargs->restricted_updates), *(dargs->released_updates),
          dargs->precondition_updates, dargs->anticondition_updates,
          dargs->postcondition_updates, dargs->read_only_updates,
          dargs->reduction_fill_updates, applied_events, 
          true/*needs lock*/, dargs->forward_to_owner);
      if (!applied_events.empty())
        Runtime::trigger_event(dargs->done_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(dargs->done_event);
      dargs->release_references();
      delete dargs->valid_updates;
      delete dargs->initialized_updates;
      delete dargs->reduction_updates;
      delete dargs->restricted_updates;
      delete dargs->released_updates;
      delete dargs->read_only_updates;
      delete dargs->reduction_fill_updates;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::invalidate_state(IndexSpaceExpression *expr,
                                  const bool expr_covers, const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      filter_valid_instances(expr, expr_covers, mask); 
      filter_reduction_instances(expr, expr_covers, mask); 
      filter_initialized_data(expr, expr_covers, mask);
      filter_restricted_instances(expr, expr_covers, mask);
      filter_released_instances(expr, expr_covers, mask);
      if (tracing_preconditions != NULL)
      {
        tracing_preconditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_preconditions->empty())
        {
          delete tracing_preconditions;
          tracing_preconditions = NULL;
        }
      }
      if (tracing_anticonditions != NULL)
      {
        tracing_anticonditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_anticonditions->empty())
        {
          delete tracing_anticonditions;
          tracing_anticonditions = NULL;
        }
      }
      if (tracing_postconditions != NULL)
      {
        tracing_postconditions->invalidate_all_but(NULL, expr, mask);
        if (tracing_postconditions->empty())
        {
          delete tracing_postconditions;
          tracing_postconditions = NULL;
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_to_local(EquivalenceSet *dst, FieldMask mask,
                     std::set<RtEvent> &applied_events, 
                     const bool invalidate_overlap, const bool forward_to_owner)
    //--------------------------------------------------------------------------
    {
      // Lock in exclusive mode if we're doing an invalidate
      AutoLock eq(eq_lock, invalidate_overlap ? 0 : 1, invalidate_overlap);
      if (!is_logical_owner())
      {
        FieldMask remote_mask = mask;
        if (!invalidate_overlap && !replicated_states.empty() &&
            !(mask * replicated_states.get_valid_mask()))
        {
          for (FieldMaskSet<CollectiveMapping>::const_iterator it =
                replicated_states.begin(); it != replicated_states.end(); it++)
          {
            if (remote_mask * it->second)
              continue;
            if (!it->first->contains(local_space))
              continue;
            remote_mask -= it->second;
            if (!remote_mask)
              break;
          }
        }
        if (!!remote_mask)
        {
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(dst->did);
            rez.serialize(local_space);
            rez.serialize(dst->region_node->row_source->handle);
            rez.serialize(remote_mask);
            rez.serialize(done_event);
            rez.serialize<bool>(invalidate_overlap);
            rez.serialize<bool>(forward_to_owner);
          }
          runtime->send_equivalence_set_clone_request(logical_owner_space, rez);
          applied_events.insert(done_event);
          mask -= remote_mask;
          if (!mask)
            return;
        }
      }
      // If we get here, we're performing the clone locally for these fields
      LegionMap<IndexSpaceExpression*,FieldMaskSet<LogicalView> > valid_updates;
      FieldMaskSet<IndexSpaceExpression> initialized_updates;
      std::map<unsigned,std::list<
        std::pair<ReductionView*,IndexSpaceExpression*> > > reduction_updates;
      LegionMap<IndexSpaceExpression*,FieldMaskSet<InstanceView> >
        restricted_updates, released_updates;
      TraceViewSet *precondition_updates = NULL;
      TraceViewSet *anticondition_updates = NULL;
      TraceViewSet *postcondition_updates = NULL;
      IndexSpaceExpression *overlap = NULL;
      if (!set_expr->is_empty())
      {
        overlap = runtime->forest->intersect_index_spaces(set_expr, 
                                      dst->region_node->row_source);
        const size_t overlap_volume = overlap->get_volume();
#ifdef DEBUG_LEGION
        assert(overlap_volume > 0);
#endif
        const bool overlap_covers = (overlap_volume == set_expr->get_volume());
        find_overlap_updates(overlap, overlap_covers, mask, valid_updates,
                             initialized_updates, reduction_updates, 
                             restricted_updates, released_updates,
                             NULL/*guards*/,NULL/*guards*/,
                             precondition_updates, anticondition_updates,
                             postcondition_updates);
      }
      else if (dst->set_expr->is_empty())
        find_overlap_updates(set_expr, true/*covers*/, mask, valid_updates,
                             initialized_updates, reduction_updates, 
                             restricted_updates, released_updates,
                             NULL/*guards*/,NULL/*guards*/,
                             precondition_updates, anticondition_updates,
                             postcondition_updates);
      // We hold the lock so calling back into the destination is safe
      dst->apply_state(valid_updates, initialized_updates, reduction_updates,
            restricted_updates, released_updates, precondition_updates,
            anticondition_updates, postcondition_updates, NULL/*guards*/,
            NULL/*guards*/, applied_events, false/*no lock*/, forward_to_owner);
      if (invalidate_overlap)
      {
        if (!set_expr->is_empty())
        {
          const bool overlap_covers = 
            (overlap->get_volume() == set_expr->get_volume());
          invalidate_state(overlap, overlap_covers, mask);
        }
        else
          invalidate_state(set_expr, true/*cover*/, mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::clone_to_remote(DistributedID target, 
                     AddressSpaceID target_space, IndexSpaceNode *target_node, 
                     const FieldMask &mask, RtUserEvent done_event, 
                     const bool invalidate_overlap, const bool forward_to_owner)
    //--------------------------------------------------------------------------
    {
      // Lock in exclusive mode if we're doing an invalidate
      AutoLock eq(eq_lock, invalidate_overlap ? 0 : 1, invalidate_overlap);
      if (!is_logical_owner())
      {
        bool forward = false;
        if (!invalidate_overlap)
        {
          // Check to see if we have replicated data for the all fields to use
          FieldMask remaining = mask;
          for (FieldMaskSet<CollectiveMapping>::const_iterator it =
                replicated_states.begin(); it != replicated_states.end(); it++)  
          {
            if (remaining * mask)
              continue;
            if (!it->first->contains(local_space))
              continue;
            remaining -= it->second;
            if (!remaining)
              break;
          }
          forward = !!remaining;
        }
        else // Always send any invalidations back to the owner
          forward = true;
        if (forward)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target);
            rez.serialize(target_space);
            rez.serialize(target_node->handle);
            rez.serialize(mask);
            rez.serialize(done_event);
            rez.serialize<bool>(invalidate_overlap);
            rez.serialize<bool>(forward_to_owner);
          }
          runtime->send_equivalence_set_clone_request(logical_owner_space, rez);
          return;
        }
      }
      IndexSpaceExpression *overlap = 
        runtime->forest->intersect_index_spaces(set_expr, target_node); 
#ifdef DEBUG_LEGION
      assert(!overlap->is_empty());
#endif
      const size_t overlap_volume = overlap->get_volume();
      const size_t set_volume = set_expr->get_volume();
      const bool overlap_covers = (overlap_volume == set_volume); 
      if (overlap_covers)
        overlap = set_expr;
      else if (overlap_volume == target_node->get_volume())
        overlap = target_node;
      // If we make it here, then we've got valid data for the all the fields
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(local_space);
        rez.serialize(done_event);
        rez.serialize<bool>(forward_to_owner);
        pack_state(rez, target_space, overlap, overlap_covers, mask, false);
      }
      runtime->send_equivalence_set_clone_response(target_space, rez);
      if (invalidate_overlap)
      {
        if (!set_expr->is_empty())
          invalidate_state(overlap, overlap_covers, mask);
        else
          invalidate_state(set_expr, true/*cover*/, mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_overlap_updates(
              IndexSpaceExpression *overlap_expr, const bool overlap_covers, 
              const FieldMask &mask, LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<LogicalView> > &valid_updates,
              FieldMaskSet<IndexSpaceExpression> &initialized_updates,
              std::map<unsigned,std::list<std::pair<ReductionView*,
                  IndexSpaceExpression*> > > &reduction_updates,
              LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<InstanceView> > &restricted_updates,
              LegionMap<IndexSpaceExpression*,
                  FieldMaskSet<InstanceView> > &released_updates,
              FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
              FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
              TraceViewSet *&precondition_updates,
              TraceViewSet *&anticondition_updates,
              TraceViewSet *&postcondition_updates) const
    //--------------------------------------------------------------------------
    {
      // Get updates from the total valid instances
      if (!total_valid_instances.empty() && 
          !(mask * total_valid_instances.get_valid_mask()))
      {
        if (!!(total_valid_instances.get_valid_mask() - mask))
        {
          // Need to filter on fields
          for (FieldMaskSet<LogicalView>::const_iterator it =
                total_valid_instances.begin(); it != 
                total_valid_instances.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            if (overlap_covers)
              valid_updates[set_expr].insert(it->first, overlap);
            else
              valid_updates[overlap_expr].insert(it->first, overlap);
          }
        }
        else
        {
          if (overlap_covers)
            valid_updates[set_expr] = total_valid_instances;
          else
            valid_updates[overlap_expr] = total_valid_instances;
        }
      }
      // Get updates from the partial valid instances
      if (!partial_valid_instances.empty() && !(mask * partial_valid_fields))
      {
        if (!!(partial_valid_fields - mask))
        {
          // Need to filter on fields
          for (ViewExprMaskSets::const_iterator pit =
                partial_valid_instances.begin(); pit !=
                partial_valid_instances.end(); pit++)
          {
            if (pit->second.get_valid_mask() * mask)
              continue;
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  pit->second.begin(); it != pit->second.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
              if (!overlap_covers)
              {
                // Check for expression overlap
                IndexSpaceExpression *intersection = 
                  runtime->forest->intersect_index_spaces(overlap_expr, 
                                                          it->first);
                const size_t volume = intersection->get_volume();
                if (volume == 0)
                  continue;
                if (volume == overlap_expr->get_volume())
                  valid_updates[overlap_expr].insert(pit->first, overlap);
                else if (volume == it->first->get_volume())
                  valid_updates[it->first].insert(pit->first, overlap);
                else
                  valid_updates[intersection].insert(pit->first, overlap);
              }
              else
                valid_updates[it->first].insert(pit->first, overlap);
            }
          }
        }
        else
        {
          // No filtering on fields, just check expressions if necessary
          for (ViewExprMaskSets::const_iterator pit =
                partial_valid_instances.begin(); pit !=
                partial_valid_instances.end(); pit++)
          {
            for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                  pit->second.begin(); it != pit->second.end(); it++)
            {
              if (!overlap_covers)
              {
                // Check for expression overlap
                IndexSpaceExpression *intersection = 
                  runtime->forest->intersect_index_spaces(overlap_expr, 
                                                          it->first);
                const size_t volume = intersection->get_volume();
                if (volume == 0)
                  continue;
                if (volume == overlap_expr->get_volume())
                  valid_updates[overlap_expr].insert(pit->first, it->second);
                else if (volume == it->first->get_volume())
                  valid_updates[it->first].insert(pit->first, it->second);
                else
                  valid_updates[intersection].insert(pit->first, it->second);
              }
              else
                valid_updates[it->first].insert(pit->first, it->second);
            }
          }
        }
      }
      // Get updates on the initialized data
      if (!initialized_data.empty() && 
          !(mask * initialized_data.get_valid_mask()))
      {
        if (!overlap_covers || !!(initialized_data.get_valid_mask() - mask))
        {
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                initialized_data.begin(); it != initialized_data.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            if (!overlap_covers)
            {
              IndexSpaceExpression *intersection = 
                runtime->forest->intersect_index_spaces(it->first,overlap_expr);
              const size_t volume = intersection->get_volume();
              if (volume == 0)
                continue;
              if (volume == overlap_expr->get_volume())
                initialized_updates.insert(overlap_expr, overlap);
              else if (volume == it->first->get_volume())
                initialized_updates.insert(it->first, overlap);
              else
                initialized_updates.insert(intersection, overlap);
            }
            else
              initialized_updates.insert(it->first, overlap);
          }
        }
        else
          initialized_updates = initialized_data;
      }
      // Get updates from the reductions
      if (!reduction_instances.empty() && !(mask * reduction_fields))
      {
        for (std::map<unsigned,std::list<std::pair<ReductionView*,
              IndexSpaceExpression*> > >::const_iterator rit =
              reduction_instances.begin(); rit != 
              reduction_instances.end(); rit++)
        {
          if (!mask.is_set(rit->first))
            continue;
          std::list<std::pair<ReductionView*,IndexSpaceExpression*> > &updates =
            reduction_updates[rit->first];
          updates = rit->second;
          if (!overlap_covers)
          {
            for (std::list<std::pair<ReductionView*,IndexSpaceExpression*> >::
                 iterator it = updates.begin(); it != updates.end();/*nothing*/)
            {
              if (it->second == set_expr)
              {
                it->second = overlap_expr;
                it++;
              }
              else
              {
                IndexSpaceExpression *intersection = 
                  runtime->forest->intersect_index_spaces(it->second, 
                                                          overlap_expr);
                const size_t volume = intersection->get_volume();
                if (volume > 0)
                {
                  if (volume == overlap_expr->get_volume())
                    it->second = overlap_expr;
                  else if (volume < it->second->get_volume())
                    it->second = intersection;
                  it++;
                }
                else
                  it = updates.erase(it);
              }
            }
          }
        }
      }
      // Get updates from the restricted instances
      if (!restricted_instances.empty() && !(mask * restricted_fields))
      {
        for (ExprViewMaskSets::const_iterator rit =
              restricted_instances.begin(); rit != 
              restricted_instances.end(); rit++)
        {
          if (mask * rit->second.get_valid_mask())
            continue;
          IndexSpaceExpression *restricted_overlap = rit->first;
          if (!overlap_covers)
          {
            IndexSpaceExpression *intersection = 
              runtime->forest->intersect_index_spaces(rit->first, overlap_expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == overlap_expr->get_volume())
              restricted_overlap = overlap_expr;
            else if (volume < rit->first->get_volume())
              restricted_overlap = intersection;
          }
          FieldMaskSet<InstanceView> &updates = 
            restricted_updates[restricted_overlap]; 
          for (FieldMaskSet<InstanceView>::const_iterator it =
                rit->second.begin(); it != rit->second.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            updates.insert(it->first, overlap);
          }
        }
      }
      // Get updates from the released instances
      if (!released_instances.empty())
      {
        for (ExprViewMaskSets::const_iterator rit = released_instances.begin();
              rit != released_instances.end(); rit++)
        {
          if (mask * rit->second.get_valid_mask())
            continue;
          IndexSpaceExpression *released_overlap = rit->first;
          if (!overlap_covers)
          {
            IndexSpaceExpression *intersection = 
              runtime->forest->intersect_index_spaces(rit->first, overlap_expr);
            const size_t volume = intersection->get_volume();
            if (volume == 0)
              continue;
            if (volume == overlap_expr->get_volume())
              released_overlap = overlap_expr;
            else if (volume < rit->first->get_volume())
              released_overlap = intersection;
          }
          FieldMaskSet<InstanceView> &updates = 
            released_updates[released_overlap]; 
          for (FieldMaskSet<InstanceView>::const_iterator it =
                rit->second.begin(); it != rit->second.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            updates.insert(it->first, overlap);
          }
        }
      }
      // There is something really scary here so be very careful
      // It might look like we have read-only guards even though
      // read_only_guard_updates is NULL. You might think that this
      // is very bad because we should be capturing those guards.
      // This should not be necessary though because the guards are
      // very conservative with these equivalence sets and they span
      // the whole equivalence set, even when the updates we care
      // about here might only be for a subset of the equivalence
      // set. Therefore it should be safe to ignore them in this case.
      if (!read_only_guards.empty() && 
          (read_only_guard_updates != NULL) &&
          !(mask * read_only_guards.get_valid_mask()))
      {
        for (FieldMaskSet<CopyFillGuard>::const_iterator it =
              read_only_guards.begin(); it != read_only_guards.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          read_only_guard_updates->insert(it->first, overlap);
        }
      }
      // See same "scary" comment above because it applies here too
      if (!reduction_fill_guards.empty() &&
          (reduction_fill_guard_updates != NULL) &&
          !(mask * reduction_fill_guards.get_valid_mask()))
      {
        for (FieldMaskSet<CopyFillGuard>::const_iterator it =
              reduction_fill_guards.begin(); it != 
              reduction_fill_guards.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          reduction_fill_guard_updates->insert(it->first, overlap);
        }
      }
      if (tracing_preconditions != NULL)
      {
        if (precondition_updates == NULL)
        {
          precondition_updates = 
            new TraceViewSet(runtime->forest, 0/*did*/, region_node);
          tracing_preconditions->find_overlaps(*precondition_updates,
                                 overlap_expr, overlap_covers, mask);
          if (precondition_updates->empty())
          {
            delete precondition_updates;
            precondition_updates = NULL;
          }
        }
        else
          tracing_preconditions->find_overlaps(*precondition_updates,
                                 overlap_expr, overlap_covers, mask);
      }
      if (tracing_anticonditions != NULL)
      {
        if (anticondition_updates == NULL)
        {
          anticondition_updates = 
            new TraceViewSet(runtime->forest, 0/*did*/, region_node);
          tracing_anticonditions->find_overlaps(*anticondition_updates,
                                  overlap_expr, overlap_covers, mask);
          if (anticondition_updates->empty())
          {
            delete anticondition_updates;
            anticondition_updates = NULL;
          }
        }
        else
          tracing_anticonditions->find_overlaps(*anticondition_updates,
                                  overlap_expr, overlap_covers, mask);
      }
      if (tracing_postconditions != NULL)
      {
        if (postcondition_updates == NULL)
        {
          postcondition_updates = 
            new TraceViewSet(runtime->forest, 0/*did*/, region_node);
          tracing_postconditions->find_overlaps(*postcondition_updates,
                                  overlap_expr, overlap_covers, mask);
          if (postcondition_updates->empty())
          {
            delete postcondition_updates;
            postcondition_updates = NULL;
          }
        }
        else
          tracing_postconditions->find_overlaps(*postcondition_updates,
                                  overlap_expr, overlap_covers, mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_state(LegionMap<IndexSpaceExpression*,
                      FieldMaskSet<LogicalView> > &valid_updates,
                  FieldMaskSet<IndexSpaceExpression> &initialized_updates,
                  std::map<unsigned,std::list<std::pair<ReductionView*,
                      IndexSpaceExpression*> > > &reduction_updates,
                  LegionMap<IndexSpaceExpression*,
                      FieldMaskSet<InstanceView> > &restricted_updates,
                  LegionMap<IndexSpaceExpression*,
                      FieldMaskSet<InstanceView> > &released_updates,
                  TraceViewSet *precondition_updates, 
                  TraceViewSet *anticondition_updates,
                  TraceViewSet *postcondition_updates,
                  FieldMaskSet<CopyFillGuard> *read_only_guard_updates,
                  FieldMaskSet<CopyFillGuard> *reduction_fill_guard_updates,
                  std::set<RtEvent> &applied_events, 
                  const bool needs_lock, const bool forward_to_owner)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        AutoLock eq(eq_lock);
        apply_state(valid_updates, initialized_updates, reduction_updates,
                    restricted_updates, released_updates, precondition_updates,
                    anticondition_updates, postcondition_updates, 
                    read_only_guard_updates, reduction_fill_guard_updates,
                    applied_events, false/*needs lock*/, forward_to_owner);
        return;
      }
      if (forward_to_owner && !is_logical_owner())
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        // Filter out any guard updates that have been pruned out
        // while we were not holding the lock. We know they've been
        // pruned because they will no longer be in the update_guards 
        if (read_only_guard_updates != NULL)
        {
          std::vector<CopyFillGuard*> to_delete;
          for (FieldMaskSet<CopyFillGuard>::const_iterator it =
                read_only_guard_updates->begin(); it != 
                read_only_guard_updates->end(); it++)
            if (read_only_guards.find(it->first) == read_only_guards.end())
              to_delete.push_back(it->first);
          if (!to_delete.empty())
          {
            for (std::vector<CopyFillGuard*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
              read_only_guard_updates->erase(*it);
          }
        }
        if (reduction_fill_guard_updates != NULL)
        {
          std::vector<CopyFillGuard*> to_delete;
          for (FieldMaskSet<CopyFillGuard>::const_iterator it =
                reduction_fill_guard_updates->begin(); it != 
                reduction_fill_guard_updates->end(); it++)
            if (reduction_fill_guards.find(it->first) == 
                reduction_fill_guards.end())
              to_delete.push_back(it->first);
          if (!to_delete.empty())
          {
            for (std::vector<CopyFillGuard*>::const_iterator it =
                  to_delete.begin(); it != to_delete.end(); it++)
              reduction_fill_guard_updates->erase(*it);
          }
        }
        // Forward this on to the logical owner
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(local_space);
          rez.serialize(done_event);
          rez.serialize<bool>(true); // forward to owner
          pack_updates(rez, logical_owner_space, valid_updates, 
                     initialized_updates, reduction_updates, restricted_updates,
                     released_updates, read_only_guard_updates, 
                     reduction_fill_guard_updates, precondition_updates,
                     anticondition_updates, postcondition_updates);
        }
        runtime->send_equivalence_set_clone_response(logical_owner_space, rez);
        applied_events.insert(done_event);
        return;
      }
      const size_t dst_volume = set_expr->get_volume();
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<LogicalView> >::const_iterator it = 
            valid_updates.begin(); it != valid_updates.end(); it++)
      {
        if (it->first->get_volume() == dst_volume)
          record_instances(set_expr, true/*covers*/,
              it->second.get_valid_mask(), it->second);
        else
          record_instances(it->first, false/*covers*/,
              it->second.get_valid_mask(), it->second);
      }
      for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
            initialized_updates.begin(); it != initialized_updates.end(); it++)
      {
        if (it->first->get_volume() == dst_volume)
          update_initialized_data(set_expr, true/*covers*/, it->second);
        else
          update_initialized_data(it->first,false/*covers*/,it->second);
      }
      for (std::map<unsigned,std::list<
            std::pair<ReductionView*,IndexSpaceExpression*> > >::iterator
            it = reduction_updates.begin(); it != reduction_updates.end(); it++)
        update_reductions(it->first, it->second);
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >::const_iterator rit =
            restricted_updates.begin(); rit != restricted_updates.end(); rit++)
      {
        const bool covers = (rit->first->get_volume() == dst_volume);
        for (FieldMaskSet<InstanceView>::const_iterator it =
              rit->second.begin(); it != rit->second.end(); it++)
          record_restriction(covers ? set_expr : rit->first, covers,
                             it->second, it->first);
      }
      for (LegionMap<IndexSpaceExpression*,
            FieldMaskSet<InstanceView> >::iterator it =
            released_updates.begin(); it != released_updates.end(); it++)
        update_released(it->first, (it->first->get_volume() == dst_volume),
                        it->second);
      if (precondition_updates != NULL)
      {
        if (tracing_preconditions == NULL)
          tracing_preconditions =
            new TraceViewSet(runtime->forest, *precondition_updates, did,
                             region_node);
        else
          precondition_updates->merge(*tracing_preconditions);
      }
      if (anticondition_updates != NULL)
      {
        if (tracing_anticonditions == NULL)
          tracing_anticonditions =
            new TraceViewSet(runtime->forest, *anticondition_updates, did,
                             region_node);
        else
          anticondition_updates->merge(*tracing_anticonditions);
      }
      if (postcondition_updates != NULL)
      {
        if (tracing_postconditions == NULL)
          tracing_postconditions =
            new TraceViewSet(runtime->forest, *postcondition_updates, did,
                             region_node);
        else
          postcondition_updates->merge(*tracing_postconditions);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_clone_request(Deserializer &derez,
                                                         Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      DistributedID target;
      derez.deserialize(target);
      AddressSpaceID target_space;
      derez.deserialize(target_space);
      IndexSpace handle;
      derez.deserialize(handle);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool invalidate_overlap, forward_to_owner;
      derez.deserialize<bool>(invalidate_overlap);
      derez.deserialize<bool>(forward_to_owner);
      
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      // Add a reference to make sure we don't race with sending the response
      set->add_base_resource_ref(RUNTIME_REF);
      if (target_space == runtime->address_space)
      {
        // We've been sent back to the owner node
        EquivalenceSet *dst = 
          runtime->find_or_request_equivalence_set(target, ready);
        std::set<RtEvent> applied_events;   
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
        dst->clone_from(target_space, set, mask, forward_to_owner,
                        applied_events, invalidate_overlap);
        if (!applied_events.empty())
          Runtime::trigger_event(done_event, 
              Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(done_event);
      }
      else
      {
        IndexSpaceNode *node = runtime->forest->get_node(handle);
        set->clone_to_remote(target, target_space, node, mask, done_event, 
                             invalidate_overlap, forward_to_owner);
      }
      if (set->remove_base_resource_ref(RUNTIME_REF))
        delete set;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_clone_response(Deserializer &derez,
                                                          Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      AddressSpaceID source;
      derez.deserialize(source);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool forward_to_owner;
      derez.deserialize(forward_to_owner);

      std::set<RtEvent> applied_events;
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state_and_apply(derez,source,forward_to_owner,applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_capture_request(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      TraceConditionSet *target;
      derez.deserialize(target);
      AddressSpaceID target_space;
      derez.deserialize(target_space);
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, source);
      FieldMask mask;
      derez.deserialize(mask);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->capture_trace_conditions(target,target_space,expr,mask,ready_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_capture_response(Deserializer &derez,
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TraceConditionSet *target;
      derez.deserialize(target);
      LogicalRegion handle;
      derez.deserialize(handle);
      TraceViewSet *previews = NULL;
      TraceViewSet *antiviews = NULL;
      TraceViewSet *postviews = NULL;
      RegionNode *region_node = NULL;
      size_t num_previews;
      derez.deserialize(num_previews);
      std::set<RtEvent> ready_events;
      if (num_previews > 0)
      {
        if (region_node == NULL)
          region_node = runtime->forest->get_node(handle);
        previews = 
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        previews->unpack(derez, num_previews, source, ready_events); 
      }
      size_t num_antiviews;
      derez.deserialize(num_antiviews);
      if (num_antiviews > 0)
      {
        if (region_node == NULL)
          region_node = runtime->forest->get_node(handle);
        antiviews =
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        antiviews->unpack(derez, num_antiviews, source, ready_events);
      }
      size_t num_postviews;
      derez.deserialize(num_postviews);
      if (num_postviews > 0)
      {
        if (region_node == NULL)
          region_node = runtime->forest->get_node(handle);
        postviews =
          new TraceViewSet(runtime->forest, 0/*no owner*/, region_node);
        postviews->unpack(derez, num_postviews, source, ready_events);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
#ifdef DEBUG_LEGION
      assert(done_event.exists());
#endif
      // Wait for the views to be ready before recording them
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        ready_events.clear();
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      target->receive_capture(previews, antiviews, postviews, ready_events); 
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    /////////////////////////////////////////////////////////////
    // Pending Equivalence Set 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PendingEquivalenceSet::PendingEquivalenceSet(RegionNode *node) 
      : region_node(node), new_set(NULL)
    //--------------------------------------------------------------------------
    {
      region_node->add_base_resource_ref(PENDING_REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    PendingEquivalenceSet::PendingEquivalenceSet(
                                               const PendingEquivalenceSet &rhs)
      : region_node(rhs.region_node)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PendingEquivalenceSet::~PendingEquivalenceSet(void)
    //--------------------------------------------------------------------------
    {
      if ((new_set != NULL) && 
          new_set->remove_base_gc_ref(PENDING_REFINEMENT_REF))
        delete new_set;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            previous_sets.begin(); it != previous_sets.end(); it++)
        if (it->first->remove_base_gc_ref(PENDING_REFINEMENT_REF))
          delete it->first;
      if (region_node->remove_base_resource_ref(PENDING_REFINEMENT_REF))
        delete region_node;
    }

    //--------------------------------------------------------------------------
    PendingEquivalenceSet& PendingEquivalenceSet::operator=(
                                               const PendingEquivalenceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PendingEquivalenceSet::record_previous(EquivalenceSet *set,
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            previous_sets.begin(); it != previous_sets.end(); it++)
        assert((set->region_node != it->first->region_node) ||
            (mask * it->second));
      if (previous_sets.insert(set, mask))
        set->add_base_gc_ref(PENDING_REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    void PendingEquivalenceSet::record_all(VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(previous_sets.empty());
#endif
      version_info.swap(previous_sets);
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            previous_sets.begin(); it != previous_sets.end(); it++)
        it->first->add_base_gc_ref(PENDING_REFINEMENT_REF);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* PendingEquivalenceSet::compute_refinement(
                              AddressSpaceID suggested_owner, Runtime *runtime,
                              std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      if (new_set == NULL)
      {
        new_set = new EquivalenceSet(runtime, 
            runtime->get_available_distributed_id(),
            suggested_owner, region_node, true/*register now*/);
        new_set->add_base_gc_ref(PENDING_REFINEMENT_REF);
        std::set<RtEvent> preconditions;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              previous_sets.begin(); it != previous_sets.end(); it++)
          new_set->clone_from(suggested_owner, it->first, it->second, 
                              false/*forward to owner*/, preconditions);
        if (!preconditions.empty())
          clone_event = new_set->make_owner(suggested_owner, 
              Runtime::merge_events(preconditions));
        else
          clone_event = new_set->make_owner(suggested_owner);
      }
      if (clone_event.exists())
      {
        if (!clone_event.has_triggered())
          ready_events.insert(clone_event);
        else
          clone_event = RtEvent::NO_RT_EVENT;
      }
      return new_set;
    }

    //--------------------------------------------------------------------------
    bool PendingEquivalenceSet::finalize(void)
    //--------------------------------------------------------------------------
    {
      if (!clone_event.exists() || clone_event.has_triggered())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it =
              previous_sets.begin(); it != previous_sets.end(); it++)
          if (it->first->remove_base_gc_ref(PENDING_REFINEMENT_REF))
            delete it->first;
        previous_sets.clear();
        // Indicate that we can delete this now
        return true;
      }
      else
      {
        // Launch a meta-task to remove these references and delete
        // the object once we know that the clone event has triggered
        DeferFinalizePendingSetArgs args(this);
        region_node->context->runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, clone_event);
        // Do no delete this now
        return false;
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void PendingEquivalenceSet::handle_defer_finalize(
                                                               const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferFinalizePendingSetArgs *dargs = 
        (const DeferFinalizePendingSetArgs*)args;
      delete dargs->pending;
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set Tracker
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void EqSetTracker::cancel_subscriptions(Runtime *runtime,
        const std::map<AddressSpaceID,std::vector<VersionManager*> > &to_cancel)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID local_space = runtime->address_space;
      for (std::map<AddressSpaceID,std::vector<VersionManager*> >::
            const_iterator ait = to_cancel.begin(); 
            ait != to_cancel.end(); ait++)
      {
        if (ait->first != local_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(this);
            rez.serialize<size_t>(ait->second.size());
            for (std::vector<VersionManager*>::const_iterator it =
                  ait->second.begin(); it != ait->second.end(); it++)
              rez.serialize(*it);
          }
          runtime->send_cancel_equivalence_sets_subscription(ait->first, rez);
        }
        else
        {
          for (std::vector<VersionManager*>::const_iterator it =
                ait->second.begin(); it != ait->second.end(); it++)
            if ((*it)->cancel_subscription(this, local_space) &&
                finish_subscription(*it, local_space))
              assert(false); // should never need to delete ourselves
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EqSetTracker::handle_cancel_subscription(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      EqSetTracker *subscriber;
      derez.deserialize(subscriber);
      size_t num_owners;
      derez.deserialize(num_owners);
      std::vector<VersionManager*> to_finish;
      for (unsigned idx = 0; idx < num_owners; idx++)
      {
        VersionManager *owner;
        derez.deserialize(owner);
        if (owner->cancel_subscription(subscriber, source))
          to_finish.push_back(owner);
      }
      if (!to_finish.empty())
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize<size_t>(0); // nothing to filter
          rez.serialize(subscriber);
          rez.serialize<size_t>(to_finish.size());
          for (std::vector<VersionManager*>::const_iterator it =
                to_finish.begin(); it != to_finish.end(); it++)
            rez.serialize(*it);
        }
        runtime->send_finish_equivalence_sets_subscription(source, rez);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EqSetTracker::finish_subscriptions(
        Runtime *runtime, VersionManager &manager,
        LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers,
        std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      const AddressSpaceID local_space = runtime->address_space;
      for (LegionMap<AddressSpaceID,SubscriberInvalidations>::const_iterator
            ait = subscribers.begin(); ait != subscribers.end(); ait++)
      {
        if (ait->first != local_space)
        {
          const RtUserEvent applied = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
#ifdef DEBUG_LEGION
            assert(!ait->second.subscribers.empty());
#endif
            rez.serialize<size_t>(ait->second.subscribers.size());
            rez.serialize<VersionManager*>(&manager);
            rez.serialize(applied);
            if (ait->second.delete_all)
              rez.serialize<size_t>(ait->second.subscribers.size());
            else
              rez.serialize<size_t>(ait->second.finished.size());
            for (FieldMaskSet<EqSetTracker>::const_iterator it =
                  ait->second.subscribers.begin(); it != 
                  ait->second.subscribers.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
            if (ait->second.finished.size() < ait->second.subscribers.size())
            {
              for (std::vector<EqSetTracker*>::const_iterator it =
                    ait->second.finished.begin(); it !=
                    ait->second.finished.end(); it++)
                rez.serialize(*it);
            }
          }
          runtime->send_finish_equivalence_sets_subscription(ait->first, rez);
          applied_events.insert(applied);
        }
        else
        {
          for (FieldMaskSet<EqSetTracker>::const_iterator it = 
                ait->second.subscribers.begin(); it != 
                ait->second.subscribers.end(); it++)
          {
            it->first->invalidate_equivalence_sets(it->second);
            if (ait->second.delete_all && 
                it->first->finish_subscription(&manager, local_space))
              delete it->first;
          }
          for (std::vector<EqSetTracker*>::const_iterator it =
                ait->second.finished.begin(); it != 
                ait->second.finished.end(); it++)
            if ((*it)->finish_subscription(&manager, local_space))
              delete *it;
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void EqSetTracker::handle_finish_subscription(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_subscribers;
      derez.deserialize(num_subscribers);
      if (num_subscribers > 0)
      {
        VersionManager *owner;
        derez.deserialize(owner);
        RtUserEvent done;
        derez.deserialize(done);
        size_t num_finished;
        derez.deserialize(num_finished);
        for (unsigned idx = 0; idx < num_subscribers; idx++)
        {
          EqSetTracker *subscriber;
          derez.deserialize(subscriber);
          FieldMask mask;
          derez.deserialize(mask);
          subscriber->invalidate_equivalence_sets(mask);
          if ((num_finished == num_subscribers) &&
              subscriber->finish_subscription(owner, source))
            delete subscriber;
        }
        if (num_finished < num_subscribers)
        {
          for (unsigned idx = 0; idx < num_finished; idx++)
          {
            EqSetTracker *to_finish;
            derez.deserialize(to_finish);
            if (to_finish->finish_subscription(owner, source))
              delete to_finish;
          }
        }
        Runtime::trigger_event(done);
      }
      else
      {
        EqSetTracker *subscriber;
        derez.deserialize(subscriber);
        size_t num_owners;
        derez.deserialize(num_owners);
        for (unsigned idx = 0; idx < num_owners; idx++)
        {
          VersionManager *owner;
          derez.deserialize(owner);
          if (subscriber->finish_subscription(owner, source))
            delete subscriber;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Version Manager 
    ///////////////////////////////////////////////////////////// 

    //--------------------------------------------------------------------------
    VersionManager::VersionManager(RegionTreeNode *n, ContextID c)
      : ctx(c), node(n), runtime(n->context->runtime)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionManager::~VersionManager(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pending_equivalence_sets.empty());
      assert(waiting_infos.empty());
      assert(equivalence_sets.empty());
      assert(equivalence_sets_ready.empty());
      assert(!disjoint_complete);
      assert(disjoint_complete_children.empty());
      assert(refinement_subscriptions.empty());
      assert(subscription_owners.empty());
#endif
    }

    //--------------------------------------------------------------------------
    void VersionManager::perform_versioning_analysis(InnerContext *context,
                   VersionInfo *version_info, RegionNode *region_node,
                   IndexSpaceExpression *expr, const bool expr_covers,
                   const FieldMask &version_mask, const UniqueID opid,
                   const AddressSpaceID source, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == region_node);
#endif
      // If we don't have equivalence classes for this region yet we 
      // either need to compute them or request them from the owner
      FieldMask remaining_mask(version_mask);
      bool has_waiter = false;
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        // Check to see if any computations of equivalence sets are in progress
        // If so we'll skip out early and go down the slow path which should
        // be a fairly rare thing to do
        if (!equivalence_sets_ready.empty())
        {
          for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
                equivalence_sets_ready.begin(); it != 
                equivalence_sets_ready.end(); it++)
          {
            if (remaining_mask * it->second)
              continue;
            // Skip out earlier if we have at least one thing to wait
            // for since we're going to have to go down the slow path
            has_waiter = true;
            break;
          }
        }
        // If we have a waiter, then don't bother doing this
        if (!has_waiter)
        {
          // Get any fields that are already ready
          if ((version_info != NULL) &&
              !(version_mask * equivalence_sets.get_valid_mask()))
            record_equivalence_sets(version_info, version_mask,
                                    expr, expr_covers, ready_events);
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields then we are done
          if (!remaining_mask)
            return;
        }
      }
      // Retake the lock in exclusive mode and make sure we don't lose the race
      RtUserEvent compute_event;
      {
        FieldMask waiting_mask;
        AutoLock m_lock(manager_lock);
        if (!equivalence_sets_ready.empty())
        {
          for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
                equivalence_sets_ready.begin(); it != 
                equivalence_sets_ready.end(); it++)
          {
            const FieldMask overlap = remaining_mask & it->second;
            if (!overlap)
              continue;
            ready_events.insert(it->first);
            waiting_mask |= overlap;
          }
          if (!!waiting_mask)
            remaining_mask -= waiting_mask;
        }
        // Get any fields that are already ready
        // Have to do this after looking for pending equivalence sets
        // to make sure we don't have pending outstanding requests
        if (!(remaining_mask * equivalence_sets.get_valid_mask()))
        {
          if (version_info != NULL)
            record_equivalence_sets(version_info, remaining_mask,
                                    expr, expr_covers, ready_events);
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields here and we're not waiting 
          // on any other computations then we're done
          if (!remaining_mask && !waiting_mask)
            return;
        }
        // If we still have remaining fields then we need to
        // do this computation ourselves
        if (!!remaining_mask)
        {
          compute_event = Runtime::create_rt_user_event();
          equivalence_sets_ready[compute_event] = remaining_mask; 
          ready_events.insert(compute_event);
          waiting_mask |= remaining_mask;
        }
#ifdef DEBUG_LEGION
        assert(!!waiting_mask);
#endif
        // Record that our version info is waiting for these fields
        if (version_info != NULL)
          waiting_infos.push_back(
             WaitingVersionInfo(version_info, waiting_mask, expr, expr_covers));
      }
      if (compute_event.exists())
      {
        // If we're an empty region then we can just get the names of any
        // empty equivalence sets directly from the region itself.
        // Otherwise, bounce this computation off the context so that we know
        // that we are on the right node to perform it
        const RtEvent ready = context->compute_equivalence_sets(this, 
            runtime->address_space, region_node, remaining_mask, opid, source);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch task to finalize the sets once they are ready
          LgFinalizeEqSetsArgs args(this, compute_event, opid); 
          runtime->issue_runtime_meta_task(args, 
                             LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(compute_event);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_equivalence_sets(VersionInfo *version_info,
                  const FieldMask &mask, IndexSpaceExpression *expr,
                  const bool expr_covers, std::set<RtEvent> &ready_events) const
    //--------------------------------------------------------------------------
    {
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
           equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        const FieldMask overlap = it->second & mask;
        if (!overlap)
          continue;
        if (!expr_covers)
        {
          IndexSpaceExpression *expr_overlap = 
            runtime->forest->intersect_index_spaces(expr, 
                                    it->first->set_expr);
          if (expr_overlap->is_empty())
            continue;
        }
        version_info->record_equivalence_set(it->first, overlap);
      }
      // If we have any disjoint complete events we need to record 
      // precondition events from them as well since they are not ready
      if (!disjoint_complete_ready.empty())
      {
        for (LegionMap<RtEvent,FieldMask>::const_iterator it =
              disjoint_complete_ready.begin(); it !=
              disjoint_complete_ready.end(); it++)
          if (!(mask * it->second))
            ready_events.insert(it->first);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_subscription(VersionManager *owner,
                                             AddressSpaceID space)
    //--------------------------------------------------------------------------
    {
      bool add_reference;
      {
        const std::pair<VersionManager*,AddressSpaceID> key(owner, space);
        AutoLock m_lock(manager_lock);
        add_reference = subscription_owners.empty();
        std::map<std::pair<VersionManager*,AddressSpaceID>,unsigned>::iterator
          finder = subscription_owners.find(key);
        if (finder == subscription_owners.end())
          subscription_owners[key] = 1;
        else
          finder->second++;
      }
      if (add_reference)
        node->add_base_resource_ref(VERSION_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    bool VersionManager::finish_subscription(VersionManager *owner,
                                             AddressSpaceID space)
    //--------------------------------------------------------------------------
    {
      bool remove_reference;
      {
        const std::pair<VersionManager*,AddressSpaceID> key(owner, space);
        AutoLock m_lock(manager_lock);
        std::map<std::pair<VersionManager*,AddressSpaceID>,unsigned>::iterator
          finder = subscription_owners.find(key);
#ifdef DEBUG_LEGION
        assert(finder != subscription_owners.end());
        assert(finder->second > 0);
#endif
        if (--finder->second == 0)
          subscription_owners.erase(finder);
        remove_reference = subscription_owners.empty();
      }
      // Do this last to avoid 
      if (remove_reference &&
          node->remove_base_resource_ref(VERSION_MANAGER_REF))
        delete node;
      // Never delete this directly
      return false;
    }

    //--------------------------------------------------------------------------
    bool VersionManager::cancel_subscription(EqSetTracker *subscriber,
                                             AddressSpaceID space)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::iterator
        refinement_finder = refinement_subscriptions.find(space);
      if (refinement_finder == refinement_subscriptions.end())
        return false;
      FieldMaskSet<EqSetTracker>::iterator finder =
        refinement_finder->second.find(subscriber);
      if (finder == refinement_finder->second.end())
        return false;
      refinement_finder->second.erase(finder);
      if (refinement_finder->second.empty())
        refinement_subscriptions.erase(refinement_finder);
      return true;
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_equivalence_set(EquivalenceSet *set,
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((node == set->region_node)
             || !node->as_region_node()->row_source->is_empty());
#endif
      AutoLock m_lock(manager_lock);
      if (equivalence_sets.insert(set, mask))
        set->add_base_resource_ref(VERSION_MANAGER_REF);
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_pending_equivalence_set(EquivalenceSet *set,
                                                        const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(mask * disjoint_complete);
#endif
      pending_equivalence_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void VersionManager::invalidate_equivalence_sets(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      // This is very subtle so pay attention!
      // If you invalidate any of the equivalence sets for a version manager
      // that is not part of the disjoint complete refinement then you have
      // to invalidate ALL the equivalence sets with the same fields since 
      // we use the presence of any equivalence set with a field to indicate
      // that this VersionManager has an up-to-date copy of the equivalence
      // sets corresponding to this logical region.
      if (mask * equivalence_sets.get_valid_mask())
        return;
      std::vector<EquivalenceSet*> to_delete;
      for (FieldMaskSet<EquivalenceSet>::iterator it =
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
        it.filter(mask);
        if (!it->second)
          to_delete.push_back(it->first);
      }
      for (std::vector<EquivalenceSet*>::const_iterator it =
            to_delete.begin(); it != to_delete.end(); it++)
      {
        equivalence_sets.erase(*it);
        if ((*it)->remove_base_resource_ref(VERSION_MANAGER_REF))
          assert(false); // should never end up deleting this here
      }
      equivalence_sets.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void VersionManager::finalize_equivalence_sets(RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> done_preconditions;
      {
        AutoLock m_lock(manager_lock);
        LegionMap<RtUserEvent,FieldMask>::iterator finder =
          equivalence_sets_ready.find(done_event);
#ifdef DEBUG_LEGION
        assert(finder != equivalence_sets_ready.end());
#endif
        // If there are any pending equivalence sets, move them into 
        // the actual equivalence sets
        if (!pending_equivalence_sets.empty() && 
            !(finder->second * pending_equivalence_sets.get_valid_mask()))
        {
          std::vector<EquivalenceSet*> to_delete;
          for (FieldMaskSet<EquivalenceSet>::iterator it = 
                pending_equivalence_sets.begin(); it !=
                pending_equivalence_sets.end(); it++)
          {
            // Once it's valid for any field then it's valid for all of them
            if (it->second * finder->second)
              continue;
#ifdef DEBUG_LEGION
            assert((node == it->first->region_node)
                   || !node->as_region_node()->row_source->is_empty());
#endif
            if (equivalence_sets.insert(it->first, it->second))
              it->first->add_base_resource_ref(VERSION_MANAGER_REF);
            to_delete.push_back(it->first);
          }
          if (!to_delete.empty())
          {
            if (to_delete.size() < pending_equivalence_sets.size())
            {
              for (std::vector<EquivalenceSet*>::const_iterator it =
                    to_delete.begin(); it != to_delete.end(); it++)
                pending_equivalence_sets.erase(*it);
              pending_equivalence_sets.tighten_valid_mask();
            }
            else
              pending_equivalence_sets.clear();
          }
        }
        if (!waiting_infos.empty())
        {
          for (LegionList<WaitingVersionInfo>::iterator wit = 
                waiting_infos.begin(); wit != waiting_infos.end(); /*nothing*/)
          {
            const FieldMask info_overlap = wit->waiting_mask & finder->second;
            if (!info_overlap)
            {
              wit++;
              continue;
            }
            record_equivalence_sets(wit->version_info, info_overlap,
                    wit->expr, wit->expr_covers, done_preconditions);
            wit->waiting_mask -= info_overlap;
            if (!wit->waiting_mask)
              wit = waiting_infos.erase(wit);
            else
              wit++;
          }
        }
        // We can relax the mask for the equivalence sets here so we don't
        // recompute in the case that we are empty
        equivalence_sets.relax_valid_mask(finder->second);
        equivalence_sets_ready.erase(finder);
      }
      if (!done_preconditions.empty())
        Runtime::trigger_event(done_event,
            Runtime::merge_events(done_preconditions));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void VersionManager::finalize_manager(void)
    //--------------------------------------------------------------------------
    {
      // We need to remove any tracked equivalence sets that we have
      FieldMaskSet<EquivalenceSet> to_remove;
      std::map<AddressSpaceID,std::vector<VersionManager*> > to_cancel;
      {
        AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
        // All these other resource should already be empty by the time
        // we are being finalized
        assert(pending_equivalence_sets.empty());
        assert(waiting_infos.empty());
        assert(equivalence_sets_ready.empty());
        assert(!disjoint_complete);
        assert(disjoint_complete_children.empty());
#endif
        if (!equivalence_sets.empty())
          to_remove.swap(equivalence_sets);
        else if (subscription_owners.empty())
          return;
        for (std::map<std::pair<VersionManager*,AddressSpaceID>,unsigned>::
              const_iterator it = subscription_owners.begin();
              it != subscription_owners.end(); it++)
          to_cancel[it->first.second].push_back(it->first.first);
      }
#ifdef DEBUG_LEGION
      assert(node->is_region());
#endif
      if (!to_cancel.empty())
        cancel_subscriptions(runtime, to_cancel);
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            to_remove.begin(); it != to_remove.end(); it++)
      {
#ifdef DEBUG_LEGION
        // This would be a valid assertion except for cases with control
        // replication where there is another node that owns the equivalence
        // set and we just happen to have a copy of it here
        //assert((it->first->region_node != node) ||
        //        it->first->region_node->row_source->is_empty());
#endif
        if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
          delete it->first;
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::initialize_versioning_analysis(EquivalenceSet *set,
                                                        const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we know this is initialization
#ifdef DEBUG_LEGION
      assert(node == set->region_node);
      assert(disjoint_complete * mask);
#endif
      disjoint_complete |= mask;
      if (equivalence_sets.insert(set, mask))
      {
        set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        set->add_base_resource_ref(VERSION_MANAGER_REF);
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::initialize_nonexclusive_virtual_analysis(
                                       const FieldMask &mask,
                                       const FieldMaskSet<EquivalenceSet> &sets)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we know this is initialization
#ifdef DEBUG_LEGION
      assert(disjoint_complete * sets.get_valid_mask());
#endif
      // We'll pretend like we're the root of the equivalence set tree
      // here even though we don't actually own these sets, we're just
      // marking it so that any analyses stop here. The logical analysis
      // will ensure that we are never refined
      disjoint_complete |= mask;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it =
            sets.begin(); it != sets.end(); it++)
        if (equivalence_sets.insert(it->first, it->second))
        {
          it->first->add_base_gc_ref(DISJOINT_COMPLETE_REF);
          it->first->add_base_resource_ref(VERSION_MANAGER_REF);
        }
    }

    //--------------------------------------------------------------------------
    void VersionManager::compute_equivalence_sets(const ContextID ctx,
                                          IndexSpaceExpression *expr,
                                          EqSetTracker *target,
                                          const AddressSpaceID target_space,
                                          FieldMask mask, InnerContext *context,
                                          const UniqueID opid,
                                          const AddressSpaceID original_source,
                                          std::set<RtEvent> &ready_events,
                                          FieldMaskSet<PartitionNode> &children,
                                          FieldMask &parent_traversal,
                                          std::set<RtEvent> &deferral_events,
                                          const bool downward_only)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node->is_region());
#endif
      FieldMask request_mask;
      RtUserEvent request_ready;
      RegionNode *region_node = node->as_region_node();
      if (downward_only)
      {
        AutoLock m_lock(manager_lock);
        // Check to see if we need to ask the context to finalize the
        // creation of any equivalence sets at this level
        // Check to see first if there are any pending computations
        // for disjoint complete equivalence sets
        if (!disjoint_complete_ready.empty())
        {
          std::vector<RtEvent> to_delete;
          for (LegionMap<RtEvent,FieldMask>::const_iterator it =
                disjoint_complete_ready.begin(); it != 
                disjoint_complete_ready.end(); it++)
          {
            if (it->first.has_triggered())
            {
              to_delete.push_back(it->first);
              continue;
            }
            if (it->second * mask)
              continue;
            deferral_events.insert(it->first);
            mask -= it->second;
            if (!mask)
              break;
          }
          if (!to_delete.empty())
          {
            for (std::vector<RtEvent>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              disjoint_complete_ready.erase(*it);
          }
        }
        // Now see if there are any fields left for which we still
        // need to perform any requests to finalize equivalence sets
        if (!!mask)
        {
          request_mask = mask - disjoint_complete;
          if (!!request_mask)
          {
            // We'll defer this computation until it is ready
            mask -= request_mask;
            request_ready = Runtime::create_rt_user_event();
            disjoint_complete_ready[request_ready] = request_mask;
            deferral_events.insert(request_ready);
          }
        }
      }
      // Release the lock before doing the call out to the context
      if (!!request_mask)
      {
#ifdef DEBUG_LEGION
        assert(request_ready.exists());
        assert(region_node->parent != NULL);
#endif
        // Record that this is now a disjoint complete child for invalidation
        // We propagate once when we make the pending refinement object and
        // again here because this might be a pending refinement from a remote
        // shard in a control replication context, so we have to do it again
        // in order to be sure that the parent knows about it
        region_node->parent->propagate_refinement(ctx,region_node,request_mask);
        // Ask the context to fill in the disjoint complete sets here
        if (context->finalize_disjoint_complete_sets(region_node, this,
                    request_mask, opid, original_source, request_ready) &&
            request_ready.has_triggered())
        {
          // Special case where the context did this right away
          // so we can restore the request fields and remove the
          // deferral event as we know that it no longer matters
          mask |= request_mask;
          deferral_events.erase(request_ready);
        }
        // If we don't have any local fields remaining then we are done
        if (!mask)
          return;
      }
      // If we have deferral events then save this traversal for another time
      if (!deferral_events.empty())
        return;
      bool new_subscriber = false;
      FieldMaskSet<EquivalenceSet> to_record;
      {
        // Do the local analysis on our owned equivalence sets
        AutoLock m_lock(manager_lock);
        if (!downward_only)
        {
          if (!disjoint_complete)
          {
            parent_traversal = mask;
            return;
          }
          parent_traversal = mask - disjoint_complete;
          if (!!parent_traversal)
          {
            mask -= parent_traversal;
            if (!mask)
              return;
          }
        }
        if (!disjoint_complete_children.empty())
        {
          const FieldMask children_overlap = mask & 
            disjoint_complete_children.get_valid_mask();
          if (!!children_overlap)
          {
            for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
                  disjoint_complete_children.begin(); it !=
                  disjoint_complete_children.end(); it++)
            {
              const FieldMask overlap = mask & it->second;
              if (!overlap)
                continue;
#ifdef DEBUG_LEGION
              assert(!it->first->is_region());
#endif
              children.insert(it->first->as_partition_node(), overlap);
              mask -= overlap;
              if (!mask)
                return;
            }
          }
        }
        // At this point we're done with the symbolic analysis so we
        // can actually test the expression for emptiness
        if ((expr != node->as_region_node()->row_source) && expr->is_empty())
          return;
        // If we make it here then we should have equivalence sets for
        // all these remaining fields
#ifdef DEBUG_LEGION
        assert(!equivalence_sets.empty() ||
                region_node->row_source->is_empty());
        assert(!(mask - equivalence_sets.get_valid_mask()) ||
                region_node->row_source->is_empty());
#endif
        // If we have any pending disjoint complete ready events then 
        // we need to record dependences on them as well
        if (!disjoint_complete_ready.empty())
        {
          for (LegionMap<RtEvent,FieldMask>::const_iterator it =
                disjoint_complete_ready.begin(); it !=
                disjoint_complete_ready.end(); it++)
            if (!(mask * it->second))
              ready_events.insert(it->first); 
        }
#ifdef DEBUG_LEGION
        FieldMask observed_sets;
#endif
        if (target_space != runtime->address_space)
        {
          FieldMaskSet<EquivalenceSet> to_send;
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            // We should only be sending equivalence sets that we
            // know we are the owners of
            // This used to be a valid assertion, but is no longer a valid
            // assertion if we have virtual-mapped parent region with
            // read-only privileges because then we have equivalence sets
            // which only overlap our region here
            //assert(it->first->region_node == node);
            observed_sets |= overlap;
#endif
            to_send.insert(it->first, overlap);
          }
          if (!to_send.empty())
          {
            // Record that we have a refinement tracker
            new_subscriber = refinement_subscriptions[target_space].insert(
                                          target, to_send.get_valid_mask());
            const RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(target);
              rez.serialize<bool>(new_subscriber);
              if (new_subscriber)
                rez.serialize(this);
              rez.serialize<size_t>(to_send.size());
              for (FieldMaskSet<EquivalenceSet>::const_iterator it =
                    to_send.begin(); it != to_send.end(); it++)
              {
                rez.serialize(it->first->did);
                rez.serialize(it->second);
              }
              rez.serialize(done);
            }
            runtime->send_compute_equivalence_sets_response(target_space, rez);
            ready_events.insert(done);
            
          }
        }
        else if (target != this)
        {
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
#ifdef DEBUG_LEGION
            // We should only be sending equivalence sets that we
            // know we are the owners of
            // This used to be a valid assertion, but is no longer a valid
            // assertion if we have virtual-mapped parent region with
            // read-only privileges because then we have equivalence sets
            // which only overlap our region here
            //assert(it->first->region_node == node);
            observed_sets |= overlap;
#endif
            to_record.insert(it->first, overlap);
          }
          if (!to_record.empty())
            // Record that we have a refinement tracker
            new_subscriber = refinement_subscriptions[target_space].insert(
                                        target, to_record.get_valid_mask());
        }
#ifdef DEBUG_LEGION
        else
          observed_sets = mask;
        assert(observed_sets == mask);
#endif
      }
      if (!to_record.empty())
      {
        if (new_subscriber)
          target->record_subscription(this, runtime->address_space);
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_record.begin(); it != to_record.end(); it++)
          target->record_equivalence_set(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_compute_equivalence_sets_response(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      EqSetTracker *target;
      derez.deserialize(target);
      bool new_subscriber;
      derez.deserialize<bool>(new_subscriber);
      if (new_subscriber)
      {
        VersionManager *owner;
        derez.deserialize(owner);
        target->record_subscription(owner, source);
      }
      size_t num_sets;
      derez.deserialize(num_sets);
      std::set<RtEvent> ready_events;
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask eq_mask;
        derez.deserialize(eq_mask);
        RtEvent ready;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready);
        if (ready.exists() && !ready.has_triggered())
        {
          target->record_pending_equivalence_set(set, eq_mask);
          ready_events.insert(ready);
        }
        else
          target->record_equivalence_set(set, eq_mask);
      }
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (!ready_events.empty())
        Runtime::trigger_event(done_event, Runtime::merge_events(ready_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_refinement(EquivalenceSet *set, 
                                           const FieldMask &mask, 
                                           FieldMask &parent_mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(set->region_node == node);
      // There should not be any other equivalence sets for these fields
      // This is a valid assertion in general, but not with control replication
      // where you can get two merge close ops updating subsets
      //assert(equivalence_sets.get_valid_mask() * mask);
#ifndef NDEBUG
      {
        FieldMaskSet<EquivalenceSet>::const_iterator finder =
          equivalence_sets.find(set);
        assert((finder == equivalence_sets.end()) ||
                (finder->second * mask));
      }
#endif
#endif
      if (equivalence_sets.insert(set, mask))
      {
        set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        set->add_base_resource_ref(VERSION_MANAGER_REF);
      }
      else
      {
        // Need to check whether we need to add the first valid reference
        // See if there were any previous fields that were valid
        FieldMaskSet<EquivalenceSet>::const_iterator finder =
          equivalence_sets.find(set);
#ifdef DEBUG_LEGION
        assert(finder != equivalence_sets.end());
#endif
        const FieldMask previous = finder->second - mask;
        if (previous * disjoint_complete)
          set->add_base_gc_ref(DISJOINT_COMPLETE_REF);
      }
      parent_mask = mask;
      if (!!disjoint_complete)
        parent_mask -= disjoint_complete;
      disjoint_complete |= mask;
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_empty_refinement(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(node->is_region());
      assert(node->as_region_node()->row_source->is_empty());
#endif
      disjoint_complete |= mask;
    }

    //--------------------------------------------------------------------------
    void VersionManager::compute_equivalence_sets(const FieldMask &mask, 
                                            FieldMask &parent_traversal, 
                                            FieldMask &children_traversal) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!node->is_region());
#endif
      AutoLock m_lock(manager_lock,1,false/*exclusive*/);
      if (!!disjoint_complete)
      {
        children_traversal = disjoint_complete & mask;
        if (!!children_traversal)
          parent_traversal = mask - children_traversal;
        else
          parent_traversal = mask;
      }
      else
        parent_traversal = mask;
    }

    //--------------------------------------------------------------------------
    void VersionManager::propagate_refinement(
                const std::vector<RegionNode*> &children, const FieldMask &mask,
                FieldMask &parent_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!node->is_region());
#endif
      AutoLock m_lock(manager_lock);
      for (std::vector<RegionNode*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert((disjoint_complete_children.find(*it) ==
                disjoint_complete_children.end()) ||
            (disjoint_complete_children[*it] * mask));
#endif
        if (disjoint_complete_children.insert(*it, mask))
          (*it)->add_base_valid_ref(VERSION_MANAGER_REF);
      }
      parent_mask = mask;
      if (!!disjoint_complete)
        parent_mask -= disjoint_complete;
      disjoint_complete |= mask;
    }

    //--------------------------------------------------------------------------
    void VersionManager::propagate_refinement(RegionTreeNode *child,
                                              const FieldMask &mask, 
                                              FieldMask &parent_mask) 
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      if (child != NULL)
      {
        // We can get duplicate propagations of refinements here because
        // of control replication. See VersionManager::compute_equivalence_sets 
        if (disjoint_complete_children.insert(child, mask))
          child->add_base_valid_ref(VERSION_MANAGER_REF);
      }
      parent_mask = mask;
      if (!!disjoint_complete)
        parent_mask -= disjoint_complete;
      disjoint_complete |= mask;
    }

    //--------------------------------------------------------------------------
    void VersionManager::invalidate_refinement(InnerContext &context,
                                      const FieldMask &mask, bool self,
                                      FieldMaskSet<RegionTreeNode> &to_traverse,
                                      LegionMap<AddressSpaceID,
                                        SubscriberInvalidations> &subscribers,
                                      std::vector<EquivalenceSet*> &to_release,
                                      bool nonexclusive_virtual_mapping_root)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);     
 #ifdef DEBUG_LEGION
      assert(to_traverse.empty());
#endif
      if (self)
      {
        if (node->is_region())
        {
          // Check to see if we have pending refinements we need
          // to tell the context that it can invalidate
          const FieldMask invalidate_mask = mask - disjoint_complete;
          if (!!invalidate_mask)
            context.invalidate_disjoint_complete_sets(node->as_region_node(), 
                                                      invalidate_mask);
        }
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        else
          assert(!(mask - disjoint_complete));
#endif
#endif
        disjoint_complete -= mask;
      }
      FieldMask children_overlap;
      if (!disjoint_complete_children.empty())
        children_overlap = mask & disjoint_complete_children.get_valid_mask();
      // Always invalidate any equivalence sets that we might have
      if (!equivalence_sets.empty() && 
          !(mask * equivalence_sets.get_valid_mask()))
      {
        std::vector<EquivalenceSet*> to_delete;
#ifdef DEBUG_LEGION
        assert(node->is_region());
#endif
        FieldMask untrack_mask;
        // Handle the nasty case where there is just one equivalence set
        // and the index space is empty so the summary valid mask is aliased
        if ((equivalence_sets.size() == 1) &&
            node->as_region_node()->row_source->is_empty())
        {
          FieldMaskSet<EquivalenceSet>::iterator finder =
            equivalence_sets.begin();
          const FieldMask overlap = finder->second & mask;
          if (!!overlap)
          {
            finder.filter(overlap);
            untrack_mask |= overlap;
            // Remove this if the only remaining fields are not refinements
            if (!finder->second || (finder->second * disjoint_complete))
            {
              // Remove this entirely from the set
              // The version manager resource reference flows back
              to_delete.push_back(finder->first);
              // Record this to be released once all the effects are applied
              to_release.push_back(finder->first);
            }
          }
        }
        else
        {
          // This is the common case
          for (FieldMaskSet<EquivalenceSet>::iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            // Skip any nodes that are not even part of a refinement 
            // Unless we are a non-exclusive virtual mapping root in
            // which case we do still want to invalidate these
            if ((it->first->region_node != node) && 
                !nonexclusive_virtual_mapping_root)
              continue;
            FieldMask overlap = it->second & mask;
            if (!overlap)
              continue;
            // If we have disjoint complete children then we do not actually
            // own these equivalence sets (we're just another observer) so
            // we can't actually remove them, just ignore them
            if (!!children_overlap)
            {
              overlap -= children_overlap;
              if (!overlap)
                continue;
            }
            untrack_mask |= overlap; 
            it.filter(overlap);
            if (!it->second)
            {
              to_delete.push_back(it->first);
              // Record this to be released once all the effects are applied
              to_release.push_back(it->first);
            }
          }
        }
        if (!!untrack_mask && !refinement_subscriptions.empty())
          filter_refinement_subscriptions(untrack_mask, subscribers);
        if (!to_delete.empty())
        {
          for (std::vector<EquivalenceSet*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
          {
            equivalence_sets.erase(*it);
            // Remove our version manager reference here, this shouldn't
            // end up deleting the equivalence set though since the 
            // to_release data structure still holds a disjoin-complete ref
            if ((*it)->remove_base_resource_ref(VERSION_MANAGER_REF))
              assert(false);
          }
        }
        equivalence_sets.tighten_valid_mask();
      }
      if (!!children_overlap)
      {
        std::vector<RegionTreeNode*> to_delete;
        for (FieldMaskSet<RegionTreeNode>::iterator it = 
              disjoint_complete_children.begin(); it != 
              disjoint_complete_children.end(); it++)
        {
          const FieldMask overlap = mask & it->second;
          if (!overlap)
            continue;
          it.filter(overlap);
          to_traverse.insert(it->first, overlap);
          // Add a reference for to_traverse
          it->first->add_base_valid_ref(VERSION_MANAGER_REF);
          if (!it->second)
            to_delete.push_back(it->first);
        }
        if (!to_delete.empty())
        {
          for (std::vector<RegionTreeNode*>::const_iterator it = 
                to_delete.begin(); it != to_delete.end(); it++)
          {
            disjoint_complete_children.erase(*it);
            if ((*it)->remove_base_valid_ref(VERSION_MANAGER_REF))
              delete (*it);
          }
        }
        disjoint_complete_children.tighten_valid_mask();
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::filter_refinement_subscriptions(const FieldMask &mask,
                 LegionMap<AddressSpaceID,SubscriberInvalidations> &subscribers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(subscribers.empty());
#endif
      for (LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::iterator 
            ait = refinement_subscriptions.begin(); 
            ait != refinement_subscriptions.end(); /*nothing*/)
      {
        const FieldMask space_overlap = ait->second.get_valid_mask() & mask;
        if (!space_overlap)
        {
          ait++;
          continue;
        }
        SubscriberInvalidations &to_untrack = subscribers[ait->first];
        to_untrack.delete_all = true;
        if (space_overlap != ait->second.get_valid_mask())
        {
          std::vector<EqSetTracker*> to_delete;
          for (FieldMaskSet<EqSetTracker>::iterator it =
                ait->second.begin(); it != ait->second.end(); it++)
          {
            const FieldMask overlap = it->second & space_overlap;
            if (!overlap)
              continue;
            to_untrack.subscribers.insert(it->first, overlap);
            it.filter(overlap);
            if (!it->second)
            {
              to_delete.push_back(it->first);
              if (!to_untrack.delete_all)
                to_untrack.finished.push_back(it->first);
            }
            else if (to_untrack.delete_all)
            {
              to_untrack.delete_all = false;
              if (to_untrack.subscribers.size() > 1)
              {
                to_untrack.finished.reserve(to_untrack.subscribers.size() - 1);
                for (FieldMaskSet<EqSetTracker>::const_iterator sit =
                      to_untrack.subscribers.begin(); sit !=
                      to_untrack.subscribers.end(); sit++)
                {
                  if (sit->first == it->first)
                    continue;
                  to_untrack.finished.push_back(sit->first);
                }
              }
            }
          }
          for (std::vector<EqSetTracker*>::const_iterator it =
                to_delete.begin(); it != to_delete.end(); it++)
            ait->second.erase(*it);
          if (ait->second.empty())
          {
            LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::iterator
              delete_it = ait++;
            refinement_subscriptions.erase(delete_it);
          }
          else
          {
            ait->second.tighten_valid_mask();
            ait++;
          }
        }
        else
        {
          to_untrack.subscribers.swap(ait->second);
          LegionMap<AddressSpaceID,FieldMaskSet<EqSetTracker> >::iterator
            delete_it = ait++;
          refinement_subscriptions.erase(delete_it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::merge(VersionManager &src, 
                               std::set<RegionTreeNode*> &to_traverse,
                               LegionMap<AddressSpaceID,
                                SubscriberInvalidations> &subscribers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(node == src.node);
      assert(!!src.disjoint_complete);
#endif
      if (!src.equivalence_sets.empty())
      {
        FieldMask untrack_mask;
        for (FieldMaskSet<EquivalenceSet>::iterator it = 
              src.equivalence_sets.begin(); it != 
              src.equivalence_sets.end(); it++)
        {
          if (it->first->region_node != node)
          {
            if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
              delete it->first;
            continue;
          }
#ifdef DEBUG_LEGION
          assert(!(it->second - src.disjoint_complete));
#endif
          untrack_mask |= it->second; 
          // Figure out whether we've already recorded this equivalence set
          FieldMaskSet<EquivalenceSet>::iterator finder = 
            equivalence_sets.find(it->first);
          if (finder != equivalence_sets.end())
          {
            // Check to see if we have recorded a valid reference yet
            // for this equivalence set
            if (!(finder->second * disjoint_complete))
            {
              // Remove the duplicate references
              if (it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF))
                assert(false); // should never end up deleting this
            }
            finder.merge(it->second);
            if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
              assert(false); // should never end up deleting this
          }
          else
            // Did not have this before so just insert it
            // References flow back
            equivalence_sets.insert(it->first, it->second);
        }
        if (!!untrack_mask && !refinement_subscriptions.empty())
            filter_refinement_subscriptions(untrack_mask, subscribers);
        src.equivalence_sets.clear();
      }
      disjoint_complete |= src.disjoint_complete;
      src.disjoint_complete.clear();
      if (!src.disjoint_complete_children.empty())
      {
        for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
              src.disjoint_complete_children.begin(); it !=
              src.disjoint_complete_children.end(); it++)
        {
          to_traverse.insert(it->first);
          // Remove duplicate references if it is already there
          // otherwise the references flow to the destination
          if (!disjoint_complete_children.insert(it->first, it->second))
            it->first->remove_base_valid_ref(VERSION_MANAGER_REF);
        }
        src.disjoint_complete_children.clear();
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::swap(VersionManager &src,
                              std::set<RegionTreeNode*> &to_traverse,
                              LegionMap<AddressSpaceID,
                                SubscriberInvalidations> &subscribers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(src.node == node);
      assert(!disjoint_complete);
      assert(!!src.disjoint_complete || (src.node->get_parent() == NULL));
      assert(equivalence_sets.empty());
      assert(disjoint_complete_children.empty());
#endif
      disjoint_complete = src.disjoint_complete;
      src.disjoint_complete.clear();
      if (!src.equivalence_sets.empty())
      {
        FieldMask untrack_mask;
        for (FieldMaskSet<EquivalenceSet>::iterator it = 
              src.equivalence_sets.begin(); it != 
              src.equivalence_sets.end(); it++)
        {
          if (it->first->region_node != node)
          {
            if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
              delete it->first;
            continue;
          }
#ifdef DEBUG_LEGION
          assert(!(it->second - disjoint_complete));
#endif
          // reference flows back
          if (!equivalence_sets.insert(it->first, it->second))
            assert(false); // should never already be there
          untrack_mask |= it->second;
        }
        if (!!untrack_mask && !refinement_subscriptions.empty())
          filter_refinement_subscriptions(untrack_mask, subscribers);
        src.equivalence_sets.clear();
      }
      disjoint_complete_children.swap(src.disjoint_complete_children);
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            disjoint_complete_children.begin(); it != 
            disjoint_complete_children.end(); it++)
        to_traverse.insert(it->first);
    }

    //--------------------------------------------------------------------------
    void VersionManager::pack_manager(Serializer &rez, const bool invalidate,
                          std::map<LegionColor,RegionTreeNode*> &to_traverse,
                          LegionMap<AddressSpaceID,
                            SubscriberInvalidations> &subscribers)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!disjoint_complete || 
          (node->is_region() && (node->as_region_node()->parent == NULL)));
#endif
      // No need for the lock here, should not be racing with anyone
      if (!equivalence_sets.empty())
      {
        const FieldMask eq_overlap = 
          equivalence_sets.get_valid_mask() & disjoint_complete;
        FieldMask untrack_mask;
        if (eq_overlap == disjoint_complete)
        {
          // We're sending all the equivalence sets
          rez.serialize<size_t>(equivalence_sets.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
            // Add a remote valid reference on these nodes to keep
            // them live until we can add on remotely.
            it->first->pack_global_ref();
            if (invalidate)
            {
              it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF);
              untrack_mask |= it->second; 
            }
          }
        }
        else if (!!eq_overlap)
        {
          // Count how many equivalence sets we need to send back
          std::vector<EquivalenceSet*> to_send;
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
            if (it->first->region_node != node)
            {
              if (invalidate && 
                  it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
                delete it->first;
            }
            else
            {
              to_send.push_back(it->first);
              // Add a remote valid reference on these nodes to keep
              // them live until we can add on remotely.
              it->first->pack_global_ref();
              if (invalidate)
              {
                it->first->remove_base_gc_ref(DISJOINT_COMPLETE_REF);
                untrack_mask |= it->second;
              }
            }
          }
          rez.serialize<size_t>(to_send.size());
          for (std::vector<EquivalenceSet*>::const_iterator it = 
                to_send.begin(); it != to_send.end(); it++)
          {
            rez.serialize((*it)->did);
            rez.serialize(equivalence_sets[*it]);
          }
        }
        else if (invalidate)
        {
          // Invalidate all the equivalence sets since none of them are going
          rez.serialize<size_t>(0);
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                equivalence_sets.begin(); it != equivalence_sets.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert(node != it->first->region_node);
#endif
            if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
              delete it->first;
          }
        }
        if (!!untrack_mask && !refinement_subscriptions.empty())
          filter_refinement_subscriptions(untrack_mask, subscribers);
      }
      else
        rez.serialize<size_t>(0);
      rez.serialize(disjoint_complete);
      rez.serialize<size_t>(disjoint_complete_children.size());
      for (FieldMaskSet<RegionTreeNode>::const_iterator it = 
            disjoint_complete_children.begin(); it !=
            disjoint_complete_children.end(); it++)
      {
        const LegionColor child_color = it->first->get_color();
        rez.serialize(child_color);
        rez.serialize(it->second);
        to_traverse[child_color] = it->first;
        // Return reference for to_traverse
        it->first->add_base_resource_ref(VERSION_MANAGER_REF);
        // Add a remote valid reference on these nodes to keep
        // them live until we can add on remotely.
        it->first->pack_valid_ref();
        if (invalidate && 
            it->first->remove_base_valid_ref(VERSION_MANAGER_REF))
          assert(false); // should never get here
      }
      if (invalidate)
      {
        disjoint_complete.clear();
        disjoint_complete_children.clear();
        equivalence_sets.clear();
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::unpack_manager(Deserializer &derez, 
      AddressSpaceID source, std::map<LegionColor,RegionTreeNode*> &to_traverse)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!disjoint_complete);
      assert(disjoint_complete_children.empty());
      assert(equivalence_sets.empty());
#endif
      // No need for the lock here, we should not be racing with anyone
      size_t num_equivalence_sets;
      derez.deserialize(num_equivalence_sets);
      std::set<RtEvent> ready_events;
      for (unsigned idx = 0; idx < num_equivalence_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        FieldMask eq_mask;
        derez.deserialize(eq_mask);
        RtEvent ready_event;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready_event);
        equivalence_sets.insert(set, eq_mask);
        if (ready_event.exists())
          ready_events.insert(ready_event);
      }
      derez.deserialize(disjoint_complete);
      size_t num_children;
      derez.deserialize(num_children);
      for (unsigned idx = 0; idx < num_children; idx++)
      {
        LegionColor child_color;
        derez.deserialize(child_color);
        FieldMask child_mask;
        derez.deserialize(child_mask);
        RegionTreeNode *child = node->get_tree_child(child_color);
        disjoint_complete_children.insert(child, child_mask);
        child->add_base_valid_ref(VERSION_MANAGER_REF);
        child->unpack_valid_ref();
        to_traverse[child_color] = child;
      }
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // Update the references on all our equivalence sets
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            equivalence_sets.begin(); it != equivalence_sets.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first->region_node == node);
#endif
        it->first->add_base_gc_ref(DISJOINT_COMPLETE_REF);
        it->first->add_base_resource_ref(VERSION_MANAGER_REF);
        it->first->unpack_global_ref();
      }
    }

    //--------------------------------------------------------------------------
    void VersionManager::print_physical_state(RegionTreeNode *node,
                                              const FieldMask &capture_mask,
                                              TreeStateLogger *logger)
    //--------------------------------------------------------------------------
    {
      logger->log("Equivalence Sets:");
      logger->down();
      // TODO: log equivalence sets
      assert(false);
      logger->up();
    } 

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_finalize_eq_sets(const void *args)
    //--------------------------------------------------------------------------
    {
      const LgFinalizeEqSetsArgs *fargs = (const LgFinalizeEqSetsArgs*)args;
      fargs->manager->finalize_equivalence_sets(fargs->compute);
    }

    /////////////////////////////////////////////////////////////
    // RegionTreePath 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegionTreePath::RegionTreePath(void) 
      : min_depth(0), max_depth(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::initialize(unsigned min, unsigned max)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min <= max);
#endif
      min_depth = min;
      max_depth = max;
      path.resize(max_depth+1, INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::register_child(unsigned depth, 
                                        const LegionColor color)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      path[depth] = color;
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::record_aliased_children(unsigned depth,
                                                 const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(min_depth <= depth);
      assert(depth <= max_depth);
#endif
      LegionMap<unsigned,FieldMask>::iterator finder = 
        interfering_children.find(depth);
      if (finder == interfering_children.end())
        interfering_children[depth] = mask;
      else
        finder->second |= mask;
    }

    //--------------------------------------------------------------------------
    void RegionTreePath::clear(void)
    //--------------------------------------------------------------------------
    {
      path.clear();
      min_depth = 0;
      max_depth = 0;
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    bool RegionTreePath::has_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      return (path[depth] != INVALID_COLOR);
    }

    //--------------------------------------------------------------------------
    LegionColor RegionTreePath::get_child(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      assert(min_depth <= depth);
      assert(depth <= max_depth);
      assert(has_child(depth));
      return path[depth];
    }
#endif

    //--------------------------------------------------------------------------
    const FieldMask* RegionTreePath::get_aliased_children(unsigned depth) const
    //--------------------------------------------------------------------------
    {
      if (interfering_children.empty())
        return NULL;
      LegionMap<unsigned,FieldMask>::const_iterator finder = 
        interfering_children.find(depth);
      if (finder == interfering_children.end())
        return NULL;
      return &(finder->second);
    }

    /////////////////////////////////////////////////////////////
    // InstanceRef 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(bool comp)
      : ready_event(ApEvent::NO_AP_EVENT), manager(NULL), local(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(const InstanceRef &rhs)
      : valid_fields(rhs.valid_fields), ready_event(rhs.ready_event),
        manager(rhs.manager), local(rhs.local)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::InstanceRef(InstanceManager *man, const FieldMask &m,ApEvent r)
      : valid_fields(m), ready_event(r), manager(man), local(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef::~InstanceRef(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceRef::operator=(const InstanceRef &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      local = rhs.local;
      manager = rhs.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator==(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      if (valid_fields != rhs.valid_fields)
        return false;
      if (ready_event != rhs.ready_event)
        return false;
      if (manager != rhs.manager)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::operator!=(const InstanceRef &rhs) const
    //--------------------------------------------------------------------------
    {
      return !(*this == rhs);
    }

    //--------------------------------------------------------------------------
    MappingInstance InstanceRef::get_mapping_instance(void) const
    //--------------------------------------------------------------------------
    {
      return MappingInstance(manager);
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_virtual_ref(void) const
    //--------------------------------------------------------------------------
    {
      if (manager == NULL)
        return true;
      return manager->is_virtual_manager(); 
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_resource_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      manager->add_base_resource_ref(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_resource_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      if (manager->remove_base_resource_ref(source))
        delete manager;
    }

    //--------------------------------------------------------------------------
    void InstanceRef::add_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      manager->as_physical_manager()->add_base_valid_ref(source);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      if (manager->as_physical_manager()->remove_base_valid_ref(source))
        delete manager;
    }

    //--------------------------------------------------------------------------
    Memory InstanceRef::get_memory(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      if (!manager->is_physical_manager())
        return Memory::NO_MEMORY;
      return manager->as_physical_manager()->get_memory();
    }

    //--------------------------------------------------------------------------
    PhysicalManager* InstanceRef::get_physical_manager(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
      assert(manager->is_physical_manager());
#endif
      return manager->as_physical_manager();
    }

    //--------------------------------------------------------------------------
    bool InstanceRef::is_field_set(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      unsigned index = manager->field_space_node->get_field_index(fid);
      return valid_fields.is_set(index);
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic> 
        InstanceRef::get_accessor(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      return manager->get_accessor();
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic>
        InstanceRef::get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      return manager->get_field_accessor(fid);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::pack_reference(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(valid_fields);
      rez.serialize(ready_event);
      if (manager != NULL)
        rez.serialize(manager->did);
      else
        rez.serialize<DistributedID>(0);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::unpack_reference(Runtime *runtime,
                                       Deserializer &derez, RtEvent &ready)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(valid_fields);
      derez.deserialize(ready_event);
      DistributedID did;
      derez.deserialize(did);
      if (did == 0)
        return;
      manager = runtime->find_or_request_instance_manager(did, ready);
      local = false;
    } 

    /////////////////////////////////////////////////////////////
    // InstanceSet 
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    InstanceSet::CollectableRef& InstanceSet::CollectableRef::operator=(
                                         const InstanceSet::CollectableRef &rhs)
    //--------------------------------------------------------------------------
    {
      valid_fields = rhs.valid_fields;
      ready_event = rhs.ready_event;
      local = rhs.local;
      manager = rhs.manager;
      return *this;
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(size_t init_size /*=0*/)
      : single((init_size <= 1)), shared(false)
    //--------------------------------------------------------------------------
    {
      if (init_size == 0)
        refs.single = NULL;
      else if (init_size == 1)
      {
        refs.single = new CollectableRef();
        refs.single->add_reference();
      }
      else
      {
        refs.multi = new InternalSet(init_size);
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::InstanceSet(const InstanceSet &rhs)
      : single(rhs.single)
    //--------------------------------------------------------------------------
    {
      // Mark that the other one is sharing too
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single == NULL)
        {
          shared = false;
          return;
        }
        shared = true;
        rhs.shared = true;
        refs.single->add_reference();
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet::~InstanceSet(void)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          delete (refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
    }

    //--------------------------------------------------------------------------
    InstanceSet& InstanceSet::operator=(const InstanceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // See if we need to delete our current one
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          delete (refs.single);
      }
      else
      {
        if (refs.multi->remove_reference())
          delete refs.multi;
      }
      // Now copy over the other one
      single = rhs.single; 
      if (single)
      {
        refs.single = rhs.refs.single;
        if (refs.single != NULL)
        {
          shared = true;
          rhs.shared = true;
          refs.single->add_reference();
        }
        else
          shared = false;
      }
      else
      {
        refs.multi = rhs.refs.multi;
        shared = true;
        rhs.shared = true;
        refs.multi->add_reference();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::make_copy(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shared);
#endif
      if (single)
      {
        if (refs.single != NULL)
        {
          CollectableRef *next = 
            new CollectableRef(*refs.single);
          next->add_reference();
          if (refs.single->remove_reference())
            delete (refs.single);
          refs.single = next;
        }
      }
      else
      {
        InternalSet *next = new InternalSet(*refs.multi);
        next->add_reference();
        if (refs.multi->remove_reference())
          delete refs.multi;
        refs.multi = next;
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator==(const InstanceSet &rhs) const
    //--------------------------------------------------------------------------
    {
      if (single != rhs.single)
        return false;
      if (single)
      {
        if (refs.single == rhs.refs.single)
          return true;
        if (((refs.single == NULL) && (rhs.refs.single != NULL)) ||
            ((refs.single != NULL) && (rhs.refs.single == NULL)))
          return false;
        return ((*refs.single) == (*rhs.refs.single));
      }
      else
      {
        if (refs.multi->vector.size() != rhs.refs.multi->vector.size())
          return false;
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          if (refs.multi->vector[idx] != rhs.refs.multi->vector[idx])
            return false;
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::operator!=(const InstanceSet &rhs) const
    //--------------------------------------------------------------------------
    {
      return !((*this) == rhs);
    }

    //--------------------------------------------------------------------------
    InstanceRef& InstanceSet::operator[](unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (shared)
        make_copy();
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(idx == 0);
        assert(refs.single != NULL);
#endif
        return *(refs.single);
      }
#ifdef DEBUG_LEGION
      assert(idx < refs.multi->vector.size());
#endif
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    const InstanceRef& InstanceSet::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // No need to make a copy if shared here since this is read-only
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(idx == 0);
        assert(refs.single != NULL);
#endif
        return *(refs.single);
      }
#ifdef DEBUG_LEGION
      assert(idx < refs.multi->vector.size());
#endif
      return refs.multi->vector[idx];
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::empty(void) const
    //--------------------------------------------------------------------------
    {
      if (single && (refs.single == NULL))
        return true;
      else if (!single && refs.multi->empty())
        return true;
      return false;
    }

    //--------------------------------------------------------------------------
    size_t InstanceSet::size(void) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == NULL)
          return 0;
        return 1;
      }
      if (refs.multi == NULL)
        return 0;
      return refs.multi->vector.size();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::resize(size_t new_size)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (new_size == 0)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            delete (refs.single);
          refs.single = NULL;
          shared = false;
        }
        else if (new_size > 1)
        {
          // Switch to multi
          InternalSet *next = new InternalSet(new_size);
          if (refs.single != NULL)
          {
            next->vector[0] = *(refs.single);
            if (refs.single->remove_reference())
              delete (refs.single);
          }
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else if (refs.single == NULL)
        {
          // New size is 1 but we were empty before
          CollectableRef *next = new CollectableRef();
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
      }
      else
      {
        if (new_size == 0)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.single = NULL;
          single = true;
          shared = false;
        }
        else if (new_size == 1)
        {
          CollectableRef *next = 
            new CollectableRef(refs.multi->vector[0]);
          if (refs.multi->remove_reference())
            delete (refs.multi);
          next->add_reference();
          refs.single = next;
          single = true;
          shared = false;
        }
        else
        {
          size_t current_size = refs.multi->vector.size();
          if (current_size != new_size)
          {
            if (shared)
            {
              // Make a copy
              InternalSet *next = new InternalSet(new_size);
              // Copy over the elements
              for (unsigned idx = 0; idx < 
                   ((current_size < new_size) ? current_size : new_size); idx++)
                next->vector[idx] = refs.multi->vector[idx];
              if (refs.multi->remove_reference())
                delete refs.multi;
              next->add_reference();
              refs.multi = next;
              shared = false;
            }
            else
            {
              // Resize our existing vector
              refs.multi->vector.resize(new_size);
            }
          }
          // Size is the same so there is no need to do anything
        }
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::clear(void)
    //--------------------------------------------------------------------------
    {
      // No need to copy since we are removing our references and not mutating
      if (single)
      {
        if ((refs.single != NULL) && refs.single->remove_reference())
          delete (refs.single);
        refs.single = NULL;
      }
      else
      {
        if (shared)
        {
          // Small optimization here, if we're told to delete it, we know
          // that means we were the last user so we can re-use it
          if (refs.multi->remove_reference())
          {
            // Put a reference back on it since we're reusing it
            refs.multi->add_reference();
            refs.multi->vector.clear();
          }
          else
          {
            // Go back to single
            refs.multi = NULL;
            single = true;
          }
        }
        else
          refs.multi->vector.clear();
      }
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::swap(InstanceSet &rhs)
    //--------------------------------------------------------------------------
    {
      // Swap references
      {
        InternalSet *other = rhs.refs.multi;
        rhs.refs.multi = refs.multi;
        refs.multi = other;
      }
      // Swap single
      {
        bool other = rhs.single;
        rhs.single = single;
        single = other;
      }
      // Swap shared
      {
        bool other = rhs.shared;
        rhs.shared = shared;
        shared = other;
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_instance(const InstanceRef &ref)
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        // No need to check for shared, we're going to make new things anyway
        if (refs.single != NULL)
        {
          // Make the new multi version
          InternalSet *next = new InternalSet(2);
          next->vector[0] = *(refs.single);
          next->vector[1] = ref;
          if (refs.single->remove_reference())
            delete (refs.single);
          next->add_reference();
          refs.multi = next;
          single = false;
          shared = false;
        }
        else
        {
          refs.single = new CollectableRef(ref);
          refs.single->add_reference();
        }
      }
      else
      {
        if (shared)
          make_copy();
        refs.multi->vector.push_back(ref);
      }
    }

    //--------------------------------------------------------------------------
    bool InstanceSet::is_virtual_mapping(void) const
    //--------------------------------------------------------------------------
    {
      if (empty())
        return true;
      if (size() > 1)
        return false;
      return refs.single->is_virtual_ref();
    }

    //--------------------------------------------------------------------------
    void InstanceSet::pack_references(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single == NULL)
        {
          rez.serialize<size_t>(0);
          return;
        }
        rez.serialize<size_t>(1);
        refs.single->pack_reference(rez);
      }
      else
      {
        rez.serialize<size_t>(refs.multi->vector.size());
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].pack_reference(rez);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::unpack_references(Runtime *runtime, Deserializer &derez, 
                                        std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      size_t num_refs;
      derez.deserialize(num_refs);
      if (num_refs == 0)
      {
        // No matter what, we can just clear out any references we have
        if (single)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            delete (refs.single);
          refs.single = NULL;
        }
        else
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          single = true;
        }
      }
      else if (num_refs == 1)
      {
        // If we're in multi, go back to single
        if (!single)
        {
          if (refs.multi->remove_reference())
            delete refs.multi;
          refs.multi = NULL;
          single = true;
        }
        // Now we can unpack our reference, see if we need to make one
        if (refs.single == NULL)
        {
          refs.single = new CollectableRef();
          refs.single->add_reference();
        }
        RtEvent ready;
        refs.single->unpack_reference(runtime, derez, ready);
        if (ready.exists())
          ready_events.insert(ready);
      }
      else
      {
        // If we're in single, go to multi
        // otherwise resize our multi for the appropriate number of references
        if (single)
        {
          if ((refs.single != NULL) && refs.single->remove_reference())
            delete (refs.single);
          refs.multi = new InternalSet(num_refs);
          refs.multi->add_reference();
          single = false;
        }
        else
          refs.multi->vector.resize(num_refs);
        // Now do the unpacking
        for (unsigned idx = 0; idx < num_refs; idx++)
        {
          RtEvent ready;
          refs.multi->vector[idx].unpack_reference(runtime, derez, ready);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      // We are always not shared when we are done
      shared = false;
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_resource_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->add_resource_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_resource_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_resource_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->remove_resource_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_resource_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::add_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->add_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_valid_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_valid_references(ReferenceSource source) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->remove_valid_reference(source);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_valid_reference(source);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::update_wait_on_events(std::set<ApEvent> &wait_on) const 
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
        {
          ApEvent ready = refs.single->get_ready_event();
          if (ready.exists())
            wait_on.insert(ready);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          ApEvent ready = refs.multi->vector[idx].get_ready_event();
          if (ready.exists())
            wait_on.insert(ready);
        }
      }
    }

    //--------------------------------------------------------------------------
    LegionRuntime::Accessor::RegionAccessor<
      LegionRuntime::Accessor::AccessorType::Generic> InstanceSet::
                                           get_field_accessor(FieldID fid) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
#ifdef DEBUG_LEGION
        assert(refs.single != NULL);
#endif
        return refs.single->get_field_accessor(fid);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
        {
          const InstanceRef &ref = refs.multi->vector[idx];
          if (ref.is_field_set(fid))
            return ref.get_field_accessor(fid);
        }
        assert(false);
        return refs.multi->vector[0].get_field_accessor(fid);
      }
    }

  }; // namespace Internal 
}; // namespace Legion

