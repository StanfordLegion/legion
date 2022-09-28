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
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionInfo::VersionInfo(const VersionInfo &rhs)
      : owner(NULL)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.owner == NULL);
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
      assert(rhs.owner == NULL);
      assert(equivalence_sets.empty());
      assert(rhs.equivalence_sets.empty());
#endif
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionInfo::record_equivalence_set(VersionManager *own,
                                             EquivalenceSet *set,
                                             const FieldMask &set_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((owner == NULL) || (owner == own));
#endif
      owner = own;
      equivalence_sets.insert(set, set_mask);
    }

    //--------------------------------------------------------------------------
    void VersionInfo::clear(void)
    //--------------------------------------------------------------------------
    {
      owner = NULL;
      equivalence_sets.clear();
    }

    /////////////////////////////////////////////////////////////
    // LogicalTraceInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalTraceInfo::LogicalTraceInfo(bool already_tr, LegionTrace *tr, 
                                       unsigned idx, const RegionRequirement &r)
      : already_traced(already_tr), trace(tr), req_idx(idx), req(r)
    //--------------------------------------------------------------------------
    {
      // If we have a trace but it doesn't handle the region tree then
      // we should mark that this is not part of a trace
      if ((trace != NULL) && 
          !trace->handles_region_tree(req.parent.get_tree_id()))
      {
        already_traced = false;
        trace = NULL;
      }
    }

    /////////////////////////////////////////////////////////////
    // Remote Trace Recorder
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTraceRecorder::RemoteTraceRecorder(Runtime *rt, AddressSpaceID origin,
                                    AddressSpaceID local, Memoizable *memo, 
                                    PhysicalTemplate *tpl, RtUserEvent applied,
                                    RtEvent collect)
      : runtime(rt), origin_space(origin), local_space(local), memoizable(memo),
        remote_tpl(tpl), applied_event(applied), collect_event(collect)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(remote_tpl != NULL);
#endif
    }

    //--------------------------------------------------------------------------
    RemoteTraceRecorder::RemoteTraceRecorder(const RemoteTraceRecorder &rhs)
      : runtime(rhs.runtime), origin_space(rhs.origin_space), 
        local_space(rhs.local_space), memoizable(rhs.memoizable), 
        remote_tpl(rhs.remote_tpl), applied_event(rhs.applied_event),
        collect_event(rhs.collect_event)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
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
      // Clean up our memoizable object if necessary
      if ((memoizable != NULL) && 
          (memoizable->get_origin_space() != local_space))
        delete memoizable;
    }

    //--------------------------------------------------------------------------
    RemoteTraceRecorder& RemoteTraceRecorder::operator=(
                                                 const RemoteTraceRecorder &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
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
               std::set<RtEvent> &external_applied, const AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      rez.serialize(origin_space);
      rez.serialize(target);
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
    void RemoteTraceRecorder::record_get_term_event(Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_GET_TERM);
          rez.serialize(applied);
          memo->pack_remote_memoizable(rez, origin_space);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_get_term_event(memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_create_ap_user_event(
                                              ApUserEvent lhs, Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_CREATE_USER_EVENT);
          rez.serialize(applied);
          rez.serialize(lhs);
          memo->pack_remote_memoizable(rez, origin_space);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Need this to be done before returning because we need to ensure
        // that this event is recorded before anyone tries to trigger it
        applied.wait();
      }
      else
        remote_tpl->record_create_ap_user_event(lhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_trigger_event(ApUserEvent lhs, ApEvent rhs,
                                                   Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
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
          memo->pack_remote_memoizable(rez, origin_space);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_trigger_event(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent rhs,
                                                  Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(rhs);
        record_merge_events(lhs, rhs_events, memo);
      }
      else
        remote_tpl->record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent e1,
                                                  ApEvent e2, Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(e1);
        rhs_events.insert(e2);
        record_merge_events(lhs, rhs_events, memo);
      }
      else
        remote_tpl->record_merge_events(lhs, e1, e2, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs, ApEvent e1,
                                                  ApEvent e2, ApEvent e3,
                                                  Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        std::set<ApEvent> rhs_events;
        rhs_events.insert(e1);
        rhs_events.insert(e2);
        rhs_events.insert(e3);
        record_merge_events(lhs, rhs_events, memo);
      }
      else
        remote_tpl->record_merge_events(lhs, e1, e2, e3, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs,
                                                  const std::set<ApEvent>& rhs,
                                                  Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
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
          memo->pack_remote_memoizable(rez, origin_space);
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
        remote_tpl->record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_merge_events(ApEvent &lhs,
                                                const std::vector<ApEvent>& rhs,
                                                Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
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
          memo->pack_remote_memoizable(rez, origin_space);
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
        remote_tpl->record_merge_events(lhs, rhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_copy(Memoizable *memo, ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField>& src_fields,
                                 const std::vector<CopySrcDstField>& dst_fields,
                                 const std::vector<Reservation> &reservations,
#ifdef LEGION_SPY
                                             RegionTreeID src_tree_id,
                                             RegionTreeID dst_tree_id,
#endif
                                             ApEvent precondition, 
                                             PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ISSUE_COPY);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
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
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_copy(memo, lhs, expr, src_fields, 
                              dst_fields, reservations,
#ifdef LEGION_SPY
                              src_tree_id, dst_tree_id,
#endif
                              precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_copy_views(ApEvent lhs, Memoizable *memo,
                                              unsigned src_idx,unsigned dst_idx,
                                              IndexSpaceExpression *expr,
                                 const FieldMaskSet<InstanceView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                              PrivilegeMode src_mode,
                                              PrivilegeMode dst_mode,
                                              bool src_indirect,
                                              bool dst_indirect,
                                              std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
      // Should never be recording indirect copies on remote nodes
      assert(!src_indirect);
      assert(!dst_indirect);
#endif
      if (local_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_COPY_VIEWS);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(lhs);
          rez.serialize(src_idx);
          rez.serialize(dst_idx);
          rez.serialize(src_mode);
          rez.serialize(dst_mode);
          expr->pack_expression(rez, origin_space);
          rez.serialize<size_t>(tracing_srcs.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                tracing_srcs.begin(); it != tracing_srcs.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(tracing_dsts.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                tracing_dsts.begin(); it != tracing_dsts.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
        }
        runtime->send_remote_trace_update(origin_space, rez);
        applied.insert(done);
      }
      else
        remote_tpl->record_copy_views(lhs, memo, src_idx, dst_idx, expr,
                tracing_srcs, tracing_dsts, src_mode, dst_mode,
                src_indirect, dst_indirect, applied);
    } 

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_across(Memoizable *memo,ApEvent &lhs,
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
    void RemoteTraceRecorder::record_indirect_views(ApEvent indirect_done,
                                                    ApEvent all_done,
                                                    Memoizable *memo,
                                                    unsigned indirect_index,
                                                    IndexSpaceExpression *expr,
                                        const FieldMaskSet<InstanceView> &views,
                                                    std::set<RtEvent> &applied,
                                                    PrivilegeMode privilege)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_INDIRECT_VIEWS);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(indirect_done);
          rez.serialize(all_done);
          rez.serialize(indirect_index);
          expr->pack_expression(rez, origin_space);
          rez.serialize<size_t>(views.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                views.begin(); it != views.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(privilege);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        applied.insert(done);
      }
      else
        remote_tpl->record_indirect_views(indirect_done, all_done, memo,
                        indirect_index, expr, views, applied, privilege);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_issue_fill(Memoizable *memo, ApEvent &lhs,
                                             IndexSpaceExpression *expr,
                                 const std::vector<CopySrcDstField> &fields,
                                             const void *fill_value, 
                                             size_t fill_size,
#ifdef LEGION_SPY
                                             FieldSpace handle,
                                             RegionTreeID tree_id,
#endif
                                             ApEvent precondition,
                                             PredEvent pred_guard)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ISSUE_FILL);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(&lhs);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
          rez.serialize<size_t>(fields.size());
          for (unsigned idx = 0; idx < fields.size(); idx++)
            pack_src_dst_field(rez, fields[idx]);
          rez.serialize(fill_size);
          rez.serialize(fill_value, fill_size);
#ifdef LEGION_SPY
          rez.serialize(handle);
          rez.serialize(tree_id);
#endif
          rez.serialize(precondition);
          rez.serialize(pred_guard);  
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // Wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_issue_fill(memo, lhs, expr, fields, 
                                      fill_value, fill_size, 
#ifdef LEGION_SPY
                                      handle, tree_id,
#endif
                                      precondition, pred_guard);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_fill_views(ApEvent lhs, Memoizable *memo,
                                 unsigned idx, IndexSpaceExpression *expr, 
                                 const FieldMaskSet<FillView> &tracing_srcs,
                                 const FieldMaskSet<InstanceView> &tracing_dsts,
                                 std::set<RtEvent> &applied_events,
                                 bool reduction_initialization)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        const RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_FILL_VIEWS);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(idx);
          rez.serialize(lhs);
          expr->pack_expression(rez, origin_space);
          rez.serialize<size_t>(tracing_srcs.size());
          for (FieldMaskSet<FillView>::const_iterator it = 
                tracing_srcs.begin(); it != tracing_srcs.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(tracing_dsts.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                tracing_dsts.begin(); it != tracing_dsts.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize<bool>(reduction_initialization);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        applied_events.insert(done);
      }
      else
        remote_tpl->record_fill_views(lhs, memo, idx, expr, tracing_srcs,
                  tracing_dsts, applied_events, reduction_initialization);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_post_fill_view(FillView *view,
                                                    const FieldMask &user_mask)
    //--------------------------------------------------------------------------
    {
      // this should never be called on remote nodes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_op_view(Memoizable *memo,
                                             unsigned idx,
                                             InstanceView *view,
                                             const RegionUsage &usage,
                                             const FieldMask &user_mask,
                                             bool update_validity)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_RECORD_OP_VIEW);
          rez.serialize(applied);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(idx);
          rez.serialize(view->did);
          rez.serialize(usage);
          rez.serialize(user_mask);
          rez.serialize<bool>(update_validity);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_op_view(memo, idx, view, usage, 
                                   user_mask, update_validity);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_set_op_sync_event(ApEvent &lhs, 
                                                       Memoizable *memo)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_SET_OP_SYNC);
          rez.serialize(done);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(&lhs);
          rez.serialize(lhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        // wait to see if lhs changes
        done.wait();
      }
      else
        remote_tpl->record_set_op_sync_event(lhs, memo);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_mapper_output(const TraceLocalID &tlid,
                              const Mapper::MapTaskOutput &output,
                              const std::deque<InstanceSet> &physical_instances,
                              std::set<RtEvent> &external_applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable->get_trace_local_id() == tlid);
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
          rez.serialize(tlid.first);
          rez.serialize(tlid.second);
          // We actually only need a few things here  
          rez.serialize<size_t>(output.target_procs.size());
          for (unsigned idx = 0; idx < output.target_procs.size(); idx++)
            rez.serialize(output.target_procs[idx]);
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
        remote_tpl->record_mapper_output(tlid, output, 
                physical_instances, external_applied);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_set_effects(Memoizable *memo, 
                                                 ApEvent &rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_SET_EFFECTS);
          rez.serialize(applied);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(rhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_set_effects(memo, rhs);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_complete_replay(Memoizable *memo, 
                                                     ApEvent rhs)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable == memo);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_COMPLETE_REPLAY);
          rez.serialize(applied);
          memo->pack_remote_memoizable(rez, origin_space);
          rez.serialize(rhs);
        }
        runtime->send_remote_trace_update(origin_space, rez);
        AutoLock a_lock(applied_lock);
        applied_events.insert(applied);
      }
      else
        remote_tpl->record_complete_replay(memo, rhs);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceRecorder::record_reservations(const TraceLocalID &tlid,
                                 const std::map<Reservation,bool> &reservations,
                                 std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memoizable->get_trace_local_id() == tlid);
#endif
      if (local_space != origin_space)
      {
        RtUserEvent done = Runtime::create_rt_user_event(); 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(remote_tpl);
          rez.serialize(REMOTE_TRACE_ACQUIRE_RELEASE);
          rez.serialize(done);
          rez.serialize(tlid.first);
          rez.serialize(tlid.second);
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
                        Deserializer &derez, Runtime *runtime, Memoizable *memo)
    //--------------------------------------------------------------------------
    {
      AddressSpaceID origin_space, local_space;
      derez.deserialize(origin_space);
      derez.deserialize(local_space);
      PhysicalTemplate *remote_tpl;
      derez.deserialize(remote_tpl);
      RtUserEvent applied_event;
      derez.deserialize(applied_event);
      RtEvent collect_event;
      derez.deserialize(collect_event);
      return new RemoteTraceRecorder(runtime, origin_space, local_space, memo,
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
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            tpl->record_get_term_event(memo);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              tpl->record_remote_memoizable(memo);
            break;
          }
        case REMOTE_TRACE_CREATE_USER_EVENT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            tpl->record_create_ap_user_event(lhs, memo);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
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
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            tpl->record_trigger_event(lhs, rhs, memo);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
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
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            size_t num_rhs;
            derez.deserialize(num_rhs);
            const ApEvent lhs_copy = lhs;
            if (num_rhs == 2)
            {
              ApEvent e1, e2;
              derez.deserialize(e1);
              derez.deserialize(e2);
              tpl->record_merge_events(lhs, e1, e2, memo);
            }
            else if (num_rhs == 3)
            {
              ApEvent e1, e2, e3;
              derez.deserialize(e1);
              derez.deserialize(e2);
              derez.deserialize(e3);
              tpl->record_merge_events(lhs, e1, e2, e3, memo);
            }
            else
            {
              std::vector<ApEvent> rhs_events(num_rhs);
              for (unsigned idx = 0; idx < num_rhs; idx++)
              {
                ApEvent event;
                derez.deserialize(rhs_events[idx]);
              }
              tpl->record_merge_events(lhs, rhs_events, memo);
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
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_ISSUE_COPY:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
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
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs;
            // Do the base call
            tpl->record_issue_copy(memo, lhs, expr, src_fields,
                                   dst_fields, reservations,
#ifdef LEGION_SPY
                                   src_tree_id, dst_tree_id,
#endif
                                   precondition, pred_guard);
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
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_COPY_VIEWS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
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
            std::set<RtEvent> ready_events;
            size_t num_srcs;
            derez.deserialize(num_srcs);
            for (unsigned idx = 0; idx < num_srcs; idx++)
            {
              DistributedID did;
              derez.deserialize(did);
              RtEvent ready;
              InstanceView *view = static_cast<InstanceView*>(
                  runtime->find_or_request_logical_view(did, ready));
              if (ready.exists() && !ready.has_triggered())
                ready_events.insert(ready);
              FieldMask mask;
              derez.deserialize(mask);
              tracing_srcs.insert(view, mask);
            }
            size_t num_dsts;
            derez.deserialize(num_dsts);
            for (unsigned idx = 0; idx < num_dsts; idx++)
            {
              DistributedID did;
              derez.deserialize(did);
              RtEvent ready;
              InstanceView *view = static_cast<InstanceView*>(
                  runtime->find_or_request_logical_view(did, ready));
              if (ready.exists() && !ready.has_triggered())
                ready_events.insert(ready);
              FieldMask mask;
              derez.deserialize(mask);
              tracing_dsts.insert(view, mask);
            } 
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              ready_events.clear();
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            tpl->record_copy_views(lhs, memo, src_idx, dst_idx, expr,
                  tracing_srcs, tracing_dsts, src_mode, dst_mode,
                  false/*indirect*/, false/*indirect*/, ready_events);
            if (!ready_events.empty())
              Runtime::trigger_event(done, Runtime::merge_events(ready_events));
            else
              Runtime::trigger_event(done);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_INDIRECT_VIEWS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            ApEvent indirect_done, all_done;
            derez.deserialize(indirect_done);
            derez.deserialize(all_done);
            unsigned indirect_index;
            derez.deserialize(indirect_index);
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr = 
              IndexSpaceExpression::unpack_expression(derez, forest, source);
            FieldMaskSet<InstanceView> tracing_views;
            std::set<RtEvent> ready_events;
            size_t num_views;
            derez.deserialize(num_views);
            for (unsigned idx = 0; idx < num_views; idx++)
            {
              DistributedID did;
              derez.deserialize(did);
              RtEvent ready;
              InstanceView *view = static_cast<InstanceView*>(
                  runtime->find_or_request_logical_view(did, ready));
              if (ready.exists() && !ready.has_triggered())
                ready_events.insert(ready);
              FieldMask mask;
              derez.deserialize(mask);
              tracing_views.insert(view, mask);
            }
            PrivilegeMode privilege;
            derez.deserialize(privilege);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              ready_events.clear();
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            tpl->record_indirect_views(indirect_done, all_done, memo,
                indirect_index, expr, tracing_views, ready_events, privilege);
            if (!ready_events.empty())
              Runtime::trigger_event(done, Runtime::merge_events(ready_events));
            else
              Runtime::trigger_event(done);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_ISSUE_FILL:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
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
            FieldSpace handle;
            derez.deserialize(handle);
            RegionTreeID tree_id;
            derez.deserialize(tree_id);
#endif
            ApEvent precondition;
            derez.deserialize(precondition);
            PredEvent pred_guard;
            derez.deserialize(pred_guard);
            // Use this to track if lhs changes
            const ApUserEvent lhs_copy = lhs; 
            // Do the base call
            tpl->record_issue_fill(memo, lhs, expr, fields,
                                   fill_value, fill_size,
#ifdef LEGION_SPY
                                   handle, tree_id,
#endif
                                   precondition, pred_guard);
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
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_FILL_VIEWS:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            unsigned index;
            derez.deserialize(index);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            RegionTreeForest *forest = runtime->forest;
            IndexSpaceExpression *expr = 
              IndexSpaceExpression::unpack_expression(derez, forest, source);
            FieldMaskSet<FillView> tracing_srcs;
            std::set<RtEvent> ready_events;
            size_t num_srcs;
            derez.deserialize(num_srcs);
            for (unsigned idx = 0; idx < num_srcs; idx++)
            {
              DistributedID did;
              derez.deserialize(did);
              RtEvent ready;
              FillView *view = static_cast<FillView*>(
                  runtime->find_or_request_logical_view(did, ready));
              if (ready.exists() && !ready.has_triggered())
                ready_events.insert(ready);
              FieldMask mask;
              derez.deserialize(mask);
              tracing_srcs.insert(view, mask);
            }
            FieldMaskSet<InstanceView> tracing_dsts;
            size_t num_dsts;
            derez.deserialize(num_dsts);
            for (unsigned idx = 0; idx < num_dsts; idx++)
            {
              DistributedID did;
              derez.deserialize(did);
              RtEvent ready;
              InstanceView *view = static_cast<InstanceView*>(
                  runtime->find_or_request_logical_view(did, ready));
              if (ready.exists() && !ready.has_triggered())
                ready_events.insert(ready);
              FieldMask mask;
              derez.deserialize(mask);
              tracing_dsts.insert(view, mask);
            }
            bool reduction_initialization;
            derez.deserialize<bool>(reduction_initialization);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              ready_events.clear();
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            tpl->record_fill_views(lhs, memo, index, expr, tracing_srcs,
                   tracing_dsts, ready_events, reduction_initialization);
            if (!ready_events.empty())
              Runtime::trigger_event(done, Runtime::merge_events(ready_events));
            else
              Runtime::trigger_event(done);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_RECORD_OP_VIEW:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            unsigned index;
            derez.deserialize(index);
            DistributedID did;
            derez.deserialize(did);
            RtEvent ready;
            InstanceView *view = static_cast<InstanceView*>(
                runtime->find_or_request_logical_view(did, ready));           
            RegionUsage usage;
            derez.deserialize(usage);
            FieldMask user_mask;
            derez.deserialize(user_mask);
            bool update_validity;
            derez.deserialize<bool>(update_validity);
            if (ready.exists() && !ready.has_triggered())
              ready.wait();
            tpl->record_op_view(memo, index, view, usage, 
                                user_mask, update_validity);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_SET_OP_SYNC:
          {
            RtUserEvent done;
            derez.deserialize(done);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            ApUserEvent *lhs_ptr;
            derez.deserialize(lhs_ptr);
            ApUserEvent lhs;
            derez.deserialize(lhs);
            const ApUserEvent lhs_copy = lhs;
            tpl->record_set_op_sync_event(lhs, memo);
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
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_RECORD_MAPPER_OUTPUT:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            derez.deserialize(tlid.first);
            derez.deserialize(tlid.second);
            size_t num_target_processors;
            derez.deserialize(num_target_processors);
            Mapper::MapTaskOutput output;
            output.target_procs.resize(num_target_processors);
            for (unsigned idx = 0; idx < num_target_processors; idx++)
              derez.deserialize(output.target_procs[idx]);
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
            tpl->record_mapper_output(tlid, output, 
                physical_instances, applied_events);
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
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            ApEvent postcondition;
            derez.deserialize(postcondition);
            tpl->record_set_effects(memo, postcondition);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_COMPLETE_REPLAY:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            Memoizable *memo = RemoteMemoizable::unpack_remote_memoizable(derez,
                                                           NULL/*op*/, runtime);
            ApEvent ready_event;
            derez.deserialize(ready_event);
            tpl->record_complete_replay(memo, ready_event);
            Runtime::trigger_event(applied);
            if (memo->get_origin_space() != runtime->address_space)
              delete memo;
            break;
          }
        case REMOTE_TRACE_ACQUIRE_RELEASE:
          {
            RtUserEvent applied;
            derez.deserialize(applied);
            TraceLocalID tlid;
            derez.deserialize(tlid.first);
            derez.deserialize(tlid.second);
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
#ifdef LEGION_SPY
      rez.serialize(field.inst_event);
#endif
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
#ifdef LEGION_SPY
      derez.deserialize(field.inst_event);
#endif
    }

    /////////////////////////////////////////////////////////////
    // TraceInfo
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(Operation *o, bool init)
      : op(o), memo((op == NULL) ? NULL : op->get_memoizable()), 
        rec((memo == NULL) ? NULL : memo->get_template()),
        recording((rec == NULL) ? false : rec->is_recording())
    //--------------------------------------------------------------------------
    {
      if (recording && init)
        record_get_term_event();
      if (rec != NULL)
        rec->add_recorder_reference();
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(SingleTask *task, RemoteTraceRecorder *r, bool init)
      : op(task), memo(task), rec(r), recording(rec != NULL)
    //--------------------------------------------------------------------------
    {
      if (recording)
      {
        rec->add_recorder_reference();
        if (init)
          record_get_term_event();
      }
    }

    //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(const TraceInfo &rhs)
      : op(rhs.op), memo(rhs.memo), rec(rhs.rec), recording(rhs.recording)
    //--------------------------------------------------------------------------
    {
      if (rec != NULL)
        rec->add_recorder_reference();
    }

   //--------------------------------------------------------------------------
    TraceInfo::TraceInfo(Operation *o, Memoizable *m, 
                         PhysicalTraceRecorder *r, const bool record)
      : op(o), memo(m), rec(r), recording(record)
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
    PhysicalTraceInfo::PhysicalTraceInfo(Operation *o, Memoizable *m, 
        unsigned src_idx,unsigned dst_idx,bool update,PhysicalTraceRecorder *r)
      : TraceInfo(o, m, r, (m != NULL)), index(src_idx), dst_index(dst_idx),
        update_validity(update)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<>
    void PhysicalTraceInfo::pack_trace_info<true>(Serializer &rez,
                                            std::set<RtEvent> &applied, 
                                            const AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(recording);
      if (recording)
      {
#ifdef DEBUG_LEGION
        assert(op != NULL);
        assert(memo != NULL);
        assert(rec != NULL);
#endif
        op->pack_remote_operation(rez, target, applied);
        memo->pack_remote_memoizable(rez, target);
        rez.serialize(index);
        rez.serialize(dst_index);
        rez.serialize<bool>(update_validity);
        rec->pack_recorder(rez, applied, target); 
      }
    }

    //--------------------------------------------------------------------------
    template<>
    void PhysicalTraceInfo::pack_trace_info<false>(Serializer &rez,
                                            std::set<RtEvent> &applied, 
                                            const AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<bool>(recording);
      if (recording)
      {
#ifdef DEBUG_LEGION
        assert(memo != NULL);
        assert(rec != NULL);
#endif
        memo->pack_remote_memoizable(rez, target);
        rez.serialize(index);
        rez.serialize(dst_index);
        rez.serialize<bool>(update_validity);
        rec->pack_recorder(rez, applied, target); 
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceInfo PhysicalTraceInfo::unpack_trace_info(
         Deserializer &derez, Runtime *runtime, std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      bool recording;
      derez.deserialize<bool>(recording);
      if (recording)
      {
        RemoteOp *op = 
          RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
        Memoizable *memo = 
          RemoteMemoizable::unpack_remote_memoizable(derez, op, runtime);
        unsigned index, dst_index;
        derez.deserialize(index);
        derez.deserialize(dst_index);
        bool update_validity;
        derez.deserialize(update_validity);
        // PhysicalTraceRecord takes possible ownership of memoizable
        PhysicalTraceRecorder *recorder = 
          RemoteTraceRecorder::unpack_remote_recorder(derez, runtime, memo);
        return PhysicalTraceInfo(op, memo, index, dst_index,
                                 update_validity, recorder);
      }
      else
        return PhysicalTraceInfo(NULL, -1U, false);
    }

    //--------------------------------------------------------------------------
    /*static*/ PhysicalTraceInfo PhysicalTraceInfo::unpack_trace_info(
                           Deserializer &derez, Runtime *runtime, Operation *op)
    //--------------------------------------------------------------------------
    {
      bool recording;
      derez.deserialize<bool>(recording);
      if (recording)
      {
#ifdef DEBUG_LEGION
        assert(op != NULL);
#endif
        Memoizable *memo = 
          RemoteMemoizable::unpack_remote_memoizable(derez, op, runtime);
        unsigned index, dst_index;
        derez.deserialize(index);
        derez.deserialize(dst_index);
        bool update_validity;
        derez.deserialize(update_validity);
        // PhysicalTraceRecord takes possible ownership of memoizable
        RemoteTraceRecorder *recorder = 
          RemoteTraceRecorder::unpack_remote_recorder(derez, runtime, memo);
        return PhysicalTraceInfo(op, memo, index, dst_index,
                                 update_validity, recorder);
      }
      else
        return PhysicalTraceInfo(op, -1U, false);
    }

    /////////////////////////////////////////////////////////////
    // ProjectionInfo 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionInfo::ProjectionInfo(Runtime *runtime, 
                     const RegionRequirement &req, IndexSpaceNode *launch_space)
      : projection((req.handle_type != LEGION_SINGULAR_PROJECTION) ? 
          runtime->find_projection_function(req.projection) : NULL),
        projection_type(req.handle_type), projection_space(launch_space)
    //--------------------------------------------------------------------------
    {
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
    // LogicalPathRegistrar
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(ContextID c, Operation *o,
                                       const FieldMask &m, RegionTreePath &p)
      : PathTraverser(p), ctx(c), field_mask(m), op(o)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::LogicalPathRegistrar(const LogicalPathRegistrar&rhs)
      : PathTraverser(rhs.path), ctx(0), field_mask(FieldMask()), op(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar::~LogicalPathRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalPathRegistrar& LogicalPathRegistrar::operator=(
                                                const LogicalPathRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask,false/*dominate*/);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar registrar(ctx, op, field_mask, false);
        return node->visit_node(&registrar);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalPathRegistrar::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask,false/*dominate*/);
      if (!has_child)
      {
        // If we're at the bottom, fan out and do all the children
        LogicalRegistrar registrar(ctx, op, field_mask, false);
        return node->visit_node(&registrar);
      }
      return true;
    }


    /////////////////////////////////////////////////////////////
    // LogicalRegistrar
    /////////////////////////////////////////////////////////////
    
    //--------------------------------------------------------------------------
    LogicalRegistrar::LogicalRegistrar(ContextID c, Operation *o,
                                       const FieldMask &m, bool dom)
      : ctx(c), field_mask(m), op(o), dominate(dom)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar::LogicalRegistrar(const LogicalRegistrar &rhs)
      : ctx(0), field_mask(FieldMask()), op(NULL), dominate(rhs.dominate)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar::~LogicalRegistrar(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LogicalRegistrar& LogicalRegistrar::operator=(const LogicalRegistrar &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_only_valid(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask, dominate);
      return true;
    }

    //--------------------------------------------------------------------------
    bool LogicalRegistrar::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      node->register_logical_dependences(ctx, op, field_mask, dominate);
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
    // Projection Epoch
    /////////////////////////////////////////////////////////////

    // C++ is really dumb
    const ProjectionEpochID ProjectionEpoch::first_epoch;

    //--------------------------------------------------------------------------
    ProjectionEpoch::ProjectionEpoch(ProjectionEpochID id, const FieldMask &m)
      : epoch_id(id), valid_fields(m)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch::ProjectionEpoch(const ProjectionEpoch &rhs)
      : epoch_id(rhs.epoch_id), valid_fields(rhs.valid_fields)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch::~ProjectionEpoch(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ProjectionEpoch& ProjectionEpoch::operator=(const ProjectionEpoch &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ProjectionEpoch::insert(ProjectionFunction *function, 
                                 IndexSpaceNode* node)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!valid_fields);
#endif
      write_projections[function].insert(node);
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
      assert(projection_epochs.empty());
      assert(!reduction_fields);
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
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
        delete *it;
      projection_epochs.clear();
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
    void LogicalState::advance_projection_epochs(const FieldMask &advance_mask)
    //--------------------------------------------------------------------------
    {
      // See if we can get some coalescing going on here
      std::map<ProjectionEpochID,ProjectionEpoch*> to_add; 
      for (std::list<ProjectionEpoch*>::iterator it = 
            projection_epochs.begin(); it != 
            projection_epochs.end(); /*nothing*/)
      {
        FieldMask overlap = (*it)->valid_fields & advance_mask;
        if (!overlap)
        {
          it++;
          continue;
        }
        const ProjectionEpochID next_epoch_id = (*it)->epoch_id + 1;
        std::map<ProjectionEpochID,ProjectionEpoch*>::iterator finder = 
          to_add.find(next_epoch_id);
        if (finder == to_add.end())
        {
          ProjectionEpoch *next_epoch = 
            new ProjectionEpoch((*it)->epoch_id+1, overlap);
          to_add[next_epoch_id] = next_epoch;
        }
        else
          finder->second->valid_fields |= overlap;
        // Filter the fields from our old one
        (*it)->valid_fields -= overlap;
        if (!((*it)->valid_fields))
        {
          delete (*it);
          it = projection_epochs.erase(it);
        }
        else
          it++;
      }
      if (!to_add.empty())
      {
        for (std::map<ProjectionEpochID,ProjectionEpoch*>::const_iterator it = 
              to_add.begin(); it != to_add.end(); it++)
          projection_epochs.push_back(it->second);
      }
    } 

    //--------------------------------------------------------------------------
    void LogicalState::update_projection_epochs(FieldMask capture_mask,
                                                const ProjectionInfo &info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!capture_mask);
#endif
      for (std::list<ProjectionEpoch*>::const_iterator it = 
            projection_epochs.begin(); it != projection_epochs.end(); it++)
      {
        FieldMask overlap = (*it)->valid_fields & capture_mask;
        if (!overlap)
          continue;
        capture_mask -= overlap;
        if (!capture_mask)
          return;
      }
      // If it didn't already exist, start a new projection epoch
      ProjectionEpoch *new_epoch = 
        new ProjectionEpoch(ProjectionEpoch::first_epoch, capture_mask);
      projection_epochs.push_back(new_epoch);
    }

    /////////////////////////////////////////////////////////////
    // FieldState 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldState::FieldState(void)
      : open_state(NOT_OPEN), redop(0), projection(NULL), projection_space(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const GenericUser &user, const FieldMask &m, 
                           RegionTreeNode *child, std::set<RtEvent> &applied)
      : redop(0), projection(NULL), projection_space(NULL)
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
      {
        WrapperReferenceMutator mutator(applied);
        child->add_base_valid_ref(FIELD_STATE_REF, &mutator);
      }
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const RegionUsage &usage, const FieldMask &m,
                           ProjectionFunction *proj, IndexSpaceNode *proj_space,
                           bool disjoint, bool dirty_reduction)
     : redop(0),projection(proj),projection_space(proj_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection != NULL);
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
      else if (disjoint && (projection->depth == 0))
        open_state = OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW;
      else
        open_state = OPEN_READ_WRITE_PROJ;
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(const FieldState &rhs)
      : open_state(rhs.open_state), redop(rhs.redop), 
        projection(rhs.projection), projection_space(rhs.projection_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(rhs.open_children.empty());
#endif
    }

    //--------------------------------------------------------------------------
    FieldState::FieldState(FieldState &&rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
      projection = rhs.projection;
      projection_space = rhs.projection_space;
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
#endif
      open_state = rhs.open_state;
      redop = rhs.redop;
      projection = rhs.projection;
      projection_space = rhs.projection_space;
      return *this;
    }

    //--------------------------------------------------------------------------
    FieldState& FieldState::operator=(FieldState &&rhs) noexcept
    //--------------------------------------------------------------------------
    {
      open_children.swap(rhs.open_children);
      open_state = rhs.open_state;
      redop = rhs.redop;
      projection = rhs.projection;
      projection_space = rhs.projection_space;
      return *this;
    }

    //--------------------------------------------------------------------------
    bool FieldState::overlaps(const FieldState &rhs) const
    //--------------------------------------------------------------------------
    {
      if (redop != rhs.redop)
        return false;
      if (projection != rhs.projection)
        return false;
      // Only do this test if they are both projections
      if ((projection != NULL) && (projection_space != rhs.projection_space))
        return false;
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
      assert(projection == rhs.projection);
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
    }

    //--------------------------------------------------------------------------
    bool FieldState::filter(const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (is_projection_state())
      {
#ifdef DEBUG_LEGION
        assert(projection != NULL);
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
    void FieldState::add_child(RegionTreeNode *child, const FieldMask &mask, 
                               std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (open_children.insert(child, mask))
      {
        WrapperReferenceMutator mutator(applied_events);
        child->add_base_valid_ref(FIELD_STATE_REF, &mutator);
      }
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
    bool FieldState::projection_domain_dominates(
                                               IndexSpaceNode *next_space) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(projection_space != NULL);
#endif
      if (projection_space == next_space)
        return true;
      // If the domains do not have the same type, the answer must be no
      if (projection_space->handle.get_type_tag() != 
          next_space->handle.get_type_tag())
        return false;
      return projection_space->dominates(next_space);
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
            logger->log("Field State: OPEN READ-ONLY PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION (Disjoint Shallow) %d",
                        projection->projection_id);
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        case OPEN_REDUCE_PROJ_DIRTY:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION (Dirty) %d Mode %d",
                        projection->projection_id, redop);
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
            logger->log("Field State: OPEN READ-ONLY PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION %d",
                        projection->projection_id);
            break;
          }
        case OPEN_READ_WRITE_PROJ_DISJOINT_SHALLOW:
          {
            logger->log("Field State: OPEN READ WRITE PROJECTION (Disjoint Shallow) %d",
                        projection->projection_id);
            break;
          }
        case OPEN_REDUCE_PROJ:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION %d Mode %d",
                        projection->projection_id, redop);
            break;
          }
        case OPEN_REDUCE_PROJ_DIRTY:
          {
            logger->log("Field State: OPEN REDUCE PROJECTION (Dirty) %d Mode %d",
                        projection->projection_id, redop);
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
                                             const LogicalTraceInfo &trace_info)
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
      close_op = creator->runtime->get_available_merge_close_op();
      merge_close_gen = close_op->get_generation();
      req.privilege_fields.clear();
      root_node->column_source->get_field_set(close_mask,
                                             trace_info.req.privilege_fields,
                                             req.privilege_fields);
      close_op->initialize(creator->get_context(), req, trace_info, 
                           trace_info.req_idx, close_mask, creator);
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
    template<int DIM>
    KDNode<DIM>::KDNode(IndexSpaceExpression *expr, Runtime *rt,
                        int ref_dim, int last)
      : runtime(rt), bounds(get_bounds(expr)), refinement_dim(ref_dim),
        last_changed_dim(last)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref_dim < DIM);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::KDNode(const Rect<DIM> &rect, Runtime *rt, 
                        int ref_dim, int last_dim)
      : runtime(rt), bounds(rect), refinement_dim(ref_dim), 
        last_changed_dim(last_dim)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(ref_dim < DIM);
#endif
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::KDNode(const KDNode<DIM> &rhs)
      : runtime(rhs.runtime), bounds(rhs.bounds), 
        refinement_dim(rhs.refinement_dim), 
        last_changed_dim(rhs.last_changed_dim)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>::~KDNode(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    KDNode<DIM>& KDNode<DIM>::operator=(const KDNode<DIM> &rhs)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Rect<DIM> KDNode<DIM>::get_bounds(IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      ApEvent wait_on;
      const Domain d = expr->get_domain(wait_on, true/*tight*/);
      if (wait_on.exists())
        wait_on.wait_faultignorant();
      return d.bounds<DIM,coord_t>();
    }

    //--------------------------------------------------------------------------
    template<int DIM>
    bool KDNode<DIM>::refine(std::vector<EquivalenceSet*> &subsets,
                           const FieldMask &refinement_mask, unsigned max_depth)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(subsets.size() > LEGION_MAX_BVH_FANOUT);
#endif
      std::vector<Rect<DIM> > subset_bounds(subsets.size());
      for (unsigned idx = 0; idx < subsets.size(); idx++)
        subset_bounds[idx] = get_bounds(subsets[idx]->set_expr);
      // Compute a splitting plane 
      coord_t split = 0;
      {
        // Sort the start and end of each equivalence set bounding rectangle
        // along the splitting dimension
        std::set<KDLine> lines;
        for (unsigned idx = 0; idx < subsets.size(); idx++)
        {
          lines.insert(KDLine(subset_bounds[idx].lo[refinement_dim],idx,true));
          lines.insert(KDLine(subset_bounds[idx].hi[refinement_dim],idx,false));
        }
        // Construct two lists by scanning from left-to-right and
        // from right-to-left of the number of rectangles that would
        // be inlcuded on the left or right side by each splitting plane
        std::map<coord_t,unsigned> left_inclusive, right_inclusive;
        unsigned count = 0;
        for (typename std::set<KDLine>::const_iterator it = lines.begin();
              it != lines.end(); it++)
        {
          // Only increment for new rectangles
          if (it->start)
            count++;
          // Always record the count for all splits
          left_inclusive[it->value] = count;
        }
        count = 0;
        for (typename std::set<KDLine>::const_reverse_iterator it = 
              lines.rbegin(); it != lines.rend(); it++)
        {
          // End of rectangles are the beginning in this direction
          if (!it->start)
            count++;
          // Always record the count for all splits
          right_inclusive[it->value] = count;
        }
#ifdef DEBUG_LEGION
        assert(left_inclusive.size() == right_inclusive.size());
#endif
        // We want to take the mini-max of the two numbers in order
        // to try to balance the splitting plane across the two sets
        unsigned split_max = subsets.size();
        for (std::map<coord_t,unsigned>::const_iterator it = 
              left_inclusive.begin(); it != left_inclusive.end(); it++)
        {
          const unsigned left = it->second;
          const unsigned right = right_inclusive[it->first];
          const unsigned max = (left > right) ? left : right;
          if (max < split_max)
          {
            split_max = max;
            split = it->first;
          }
        }
      }
      // Sort the subsets into left and right
      Rect<DIM> left_bounds, right_bounds;
      left_bounds = bounds;
      right_bounds = bounds;
      left_bounds.hi[refinement_dim] = split;
      right_bounds.lo[refinement_dim] = split+1;
      std::vector<EquivalenceSet*> left_set, right_set;
      for (unsigned idx = 0; idx < subsets.size(); idx++)
      {
        const Rect<DIM> &sub_bounds = subset_bounds[idx];
        if (left_bounds.overlaps(sub_bounds))
          left_set.push_back(subsets[idx]);
        if (right_bounds.overlaps(sub_bounds))
          right_set.push_back(subsets[idx]);
      }
      // Check for the non-convex case where we can't refine anymore
      if ((refinement_dim == last_changed_dim) && 
          ((left_set.size() == subsets.size()) ||
           (right_set.size() == subsets.size())))
        return false;
      // Recurse down the tree
      const int next_dim = (refinement_dim + 1) % DIM;
      bool left_changed = false;
      if ((left_set.size() > LEGION_MAX_BVH_FANOUT) && (max_depth > 0))
      {
        // If all the subsets span our splitting plane then we need
        // to either start tracking the last changed dimension or 
        // continue propagating the current one
        const int left_last_dim = (left_set.size() == subsets.size()) ? 
          ((last_changed_dim != -1) ? last_changed_dim : refinement_dim) : -1;
        KDNode<DIM> left(left_bounds, runtime, next_dim, left_last_dim);
        left_changed = left.refine(left_set, refinement_mask, max_depth - 1);
      }
      bool right_changed = false;
      if ((right_set.size() > LEGION_MAX_BVH_FANOUT) && (max_depth > 0))
      {
        // If all the subsets span our splitting plane then we need
        // to either start tracking the last changed dimension or 
        // continue propagating the current one
        const int right_last_dim = (right_set.size() == subsets.size()) ? 
          ((last_changed_dim != -1) ? last_changed_dim : refinement_dim) : -1;
        KDNode<DIM> right(right_bounds, runtime, next_dim, right_last_dim);
        right_changed = right.refine(right_set, refinement_mask, max_depth - 1);
      }
      // If the sum of the left and right equivalence sets 
      // are too big then build intermediate nodes for each one
      if (((left_set.size() + right_set.size()) > LEGION_MAX_BVH_FANOUT) &&
          (left_set.size() < subsets.size()) && !left_set.empty() &&
          (right_set.size() < subsets.size()) && !right_set.empty())
      {
        // Make a new equivalence class and record all the subsets
        const AddressSpaceID local_space = runtime->address_space;
        std::set<IndexSpaceExpression*> left_exprs, right_exprs;
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              left_set.begin(); it != left_set.end(); it++)
          left_exprs.insert((*it)->set_expr);
        IndexSpaceExpression *left_union_expr = 
          runtime->forest->union_index_spaces(left_exprs);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              right_set.begin(); it != right_set.end(); it++)
          right_exprs.insert((*it)->set_expr);
        IndexSpaceExpression *right_union_expr = 
          runtime->forest->union_index_spaces(right_exprs);
        EquivalenceSet *left_temp = new EquivalenceSet(runtime,
            runtime->get_available_distributed_id(), local_space,
            local_space, left_union_expr, NULL/*index space*/,
            true/*register now*/);
        EquivalenceSet *right_temp = new EquivalenceSet(runtime,
            runtime->get_available_distributed_id(), local_space,
            local_space, right_union_expr, NULL/*index space*/,
            true/*register now*/);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              left_set.begin(); it != left_set.end(); it++)
          left_temp->record_subset(*it, refinement_mask);
        for (std::vector<EquivalenceSet*>::const_iterator it = 
              right_set.begin(); it != right_set.end(); it++)
          right_temp->record_subset(*it, refinement_mask);
        subsets.clear();
        subsets.push_back(left_temp);
        subsets.push_back(right_temp);
        return true;
      }
      else if (left_changed || right_changed)
      {
        // If either right or left changed, then we need to recombine
        // and deduplicate the equivalence sets before we can return
        if (!left_set.empty() && !right_set.empty())
        {
          std::set<EquivalenceSet*> children;
          children.insert(left_set.begin(), left_set.end());
          children.insert(right_set.begin(), right_set.end());
          subsets.clear();
          subsets.insert(subsets.end(), children.begin(), children.end());
        }
        else if (!left_set.empty())
        {
          subsets.clear();
          subsets.insert(subsets.end(), left_set.begin(), left_set.end());
        }
        else
        {
          subsets.clear();
          subsets.insert(subsets.end(), right_set.begin(), right_set.end());
        }
        return true;
      }
      else // No changes were made
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Guard
    /////////////////////////////////////////////////////////////

#ifndef NON_AGGRESSIVE_AGGREGATORS
    //--------------------------------------------------------------------------
    CopyFillGuard::CopyFillGuard(RtUserEvent post, RtUserEvent applied)
      : guard_postcondition(post), effects_applied(applied),
        releasing_guards(false)
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
      : effects_applied(applied), releasing_guards(false)
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
      assert(effects_applied.exists());
#endif
      // We only ever pack the effects applied event here because once a
      // guard is on a remote node then the guard postcondition is no longer
      // useful since all remote copy fill operations will need to key off
      // the effects applied event to be correct
      rez.serialize(effects_applied);
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
#ifdef DEBUG_LEGION
      if (!result->record_guard_set(set))
        assert(false);
#else
      result->record_guard_set(set);
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
    bool CopyFillGuard::record_guard_set(EquivalenceSet *set)
    //--------------------------------------------------------------------------
    {
      if (releasing_guards)
        return false;
      AutoLock g_lock(guard_lock);
      // Check again after getting the lock to avoid the race
      if (releasing_guards)
        return false;
      guarded_sets.insert(set);
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
        for (std::set<EquivalenceSet*>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          (*it)->remove_update_guard(this);
      }
    }

    /////////////////////////////////////////////////////////////
    // Copy Fill Aggregator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, 
                                           Operation *o, unsigned idx, 
                                           RtEvent g, bool t, PredEvent p)
      : WrapperReferenceMutator(effects),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        CopyFillGuard(Runtime::create_rt_user_event(), 
                      Runtime::create_rt_user_event()),
#else
        CopyFillGuard(Runtime::create_rt_user_event()), 
#endif
        forest(f), local_space(f->runtime->address_space), op(o), 
        src_index(idx), dst_index(idx), guard_precondition(g), 
        predicate_guard(p), track_events(t), tracing_src_fills(NULL),
        tracing_srcs(NULL), tracing_dsts(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(RegionTreeForest *f, 
                                Operation *o, unsigned src_idx, unsigned dst_idx,
                                RtEvent g, bool t, PredEvent p)
      : WrapperReferenceMutator(effects),
#ifndef NON_AGGRESSIVE_AGGREGATORS
        CopyFillGuard(Runtime::create_rt_user_event(), 
                      Runtime::create_rt_user_event()),
#else
        CopyFillGuard(Runtime::create_rt_user_event()),
#endif
        forest(f), local_space(f->runtime->address_space), op(o), 
        src_index(src_idx), dst_index(dst_idx), guard_precondition(g),
        predicate_guard(p), track_events(t), tracing_src_fills(NULL),
        tracing_srcs(NULL), tracing_dsts(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CopyFillAggregator::CopyFillAggregator(const CopyFillAggregator &rhs)
      : WrapperReferenceMutator(effects), CopyFillGuard(rhs), 
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
      // Remove source precondition expression references
      for (std::map<InstanceView*,EventFieldExprs>::iterator vit =
            src_pre.begin(); vit != src_pre.end(); vit++)
      {
        for (EventFieldExprs::iterator eit =
              vit->second.begin(); eit != vit->second.end(); eit++)
        {
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                eit->second.begin(); it != eit->second.end(); it++)
            if (it->first->remove_base_expression_reference(AGGREGATOR_REF))
              delete it->first;
          eit->second.clear();
        }
        vit->second.clear();
      }
      src_pre.clear();
      // Remove destination precondition expression references
      for (std::map<InstanceView*,EventFieldExprs>::iterator vit =
            dst_pre.begin(); vit != dst_pre.end(); vit++)
      {
        for (EventFieldExprs::iterator eit =
              vit->second.begin(); eit != vit->second.end(); eit++)
        {
          for (FieldMaskSet<IndexSpaceExpression>::iterator it =
                eit->second.begin(); it != eit->second.end(); it++)
            if (it->first->remove_base_expression_reference(AGGREGATOR_REF))
              delete it->first;
          eit->second.clear();
        }
        vit->second.clear();
      }
      dst_pre.clear();
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
      // Clean up any data structures that we made for tracing
      if (tracing_src_fills != NULL)
        delete tracing_src_fills;
      if (tracing_srcs != NULL)
        delete tracing_srcs;
      if (tracing_dsts != NULL)
        delete tracing_dsts;
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
    void CopyFillAggregator::CopyUpdate::compute_source_preconditions(
                     RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                     const bool copy_across,
#endif
                     const std::map<InstanceView*,EventFieldExprs> &src_pre,
                     LegionMap<ApEvent,FieldMask> &preconditions) const
    //--------------------------------------------------------------------------
    {
      std::map<InstanceView*,EventFieldExprs>::const_iterator finder = 
        src_pre.find(source);
      if (finder == src_pre.end())
        return;
      for (EventFieldExprs::const_iterator eit = 
            finder->second.begin(); eit != finder->second.end(); eit++)
      {
        FieldMask set_overlap = src_mask & eit->second.get_valid_mask();
        if (!set_overlap)
          continue;
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              eit->second.begin(); it != eit->second.end(); it++)
        {
          const FieldMask overlap = set_overlap & it->second;
          if (!overlap)
            continue;
          IndexSpaceExpression *expr_overlap = 
            forest->intersect_index_spaces(expr, it->first);
          if (expr_overlap->is_empty())
            continue;
#ifdef DEBUG_LEGION
          // Since this is an equivalence set update there should be no users 
          // that are using just a part of it, should be all or nothing, with
          // the exception of copy across operations in which case it doesn't
          // matter because we don't need precise preconditions there
          if (copy_across)
            assert(expr_overlap->get_volume() == expr->get_volume());
#endif
          // Overlap in both so record it
          LegionMap<ApEvent,FieldMask>::iterator event_finder =
            preconditions.find(eit->first);
          if (event_finder == preconditions.end())
            preconditions[eit->first] = overlap;
          else
            event_finder->second |= overlap;
          set_overlap -= overlap;
          if (!set_overlap)
            break;
        }
      }
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
    void CopyFillAggregator::FillUpdate::compute_source_preconditions(
                     RegionTreeForest *forest,
#ifdef DEBUG_LEGION
                     const bool copy_across,
#endif
                     const std::map<InstanceView*,EventFieldExprs> &src_pre,
                     LegionMap<ApEvent,FieldMask> &preconditions) const
    //--------------------------------------------------------------------------
    {
      // Do nothing, we have no source preconditions to worry about
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
    void CopyFillAggregator::record_updates(InstanceView *dst_view, 
                                    const FieldMaskSet<LogicalView> &src_views,
                                    const FieldMask &src_mask,
                                    IndexSpaceExpression *expr,
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
          }
          else
          {
            DeferredView *def = view->as_deferred_view();
            def->flatten(*this, dst_view, record_mask, expr, helper);
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
            }
            else
            {
              DeferredView *def = view->as_deferred_view();
              def->flatten(*this, dst_view, record_mask, expr, helper);
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
              record_fill(dst_view, fill, vit->set_mask, expr, helper);
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
              }
              else
              {
                // Hard, multiple potential sources,
                // ask the mapper which one to use
                // First though check to see if we've already asked it
                bool found = false;
                const std::set<InstanceView*> instances_set(instances.begin(),
                                                            instances.end());
                std::map<InstanceView*,LegionVector<SourceQuery>>::
                  const_iterator finder = mapper_queries.find(dst_view);
                if (finder != mapper_queries.end())
                {
                  for (LegionVector<SourceQuery>::const_iterator qit = 
                        finder->second.begin(); qit != 
                        finder->second.end(); qit++)
                  {
                    if ((qit->query_mask == vit->set_mask) &&
                        (qit->sources == instances_set))
                    {
                      found = true;
                      record_view(qit->result);
                      CopyUpdate *update = new CopyUpdate(qit->result, 
                                    qit->query_mask, expr, redop, helper);
                      if (helper == NULL)
                        updates.insert(update, qit->query_mask);
                      else
                        updates.insert(update, 
                            helper->convert_src_to_dst(qit->query_mask));
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
                  // We know that which ever one was chosen first is
                  // the one that satisfies all our fields since all
                  // these instances are valid for all fields
                  InstanceView *result = ranking.empty() ? 
                    instances.front() : instances[ranking[0]];
                  // Record the update
                  record_view(result);
                  CopyUpdate *update = new CopyUpdate(result, vit->set_mask,
                                                      expr, redop, helper);
                  if (helper == NULL)
                    updates.insert(update, vit->set_mask);
                  else
                    updates.insert(update, 
                        helper->convert_src_to_dst(vit->set_mask));
                  // Save the result for the future
                  mapper_queries[dst_view].push_back(
                      SourceQuery(instances_set, vit->set_mask, result));
                }
              }
            }
            else
            {
#ifdef DEBUG_LEGION
              assert(deferred != NULL);
#endif
              deferred->flatten(*this, dst_view, vit->set_mask, expr, helper);
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_fill(InstanceView *dst_view,
                                         FillView *src_view,
                                         const FieldMask &fill_mask,
                                         IndexSpaceExpression *expr,
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
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_reductions(InstanceView *dst_view,
                                   const std::vector<ReductionView*> &src_views,
                                   const unsigned src_fidx,
                                   const unsigned dst_fidx,
                                   IndexSpaceExpression *expr,
                                   CopyAcrossHelper *across_helper)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!src_views.empty());
      assert(!expr->is_empty());
#endif 
      update_fields.set_bit(src_fidx);
      record_view(dst_view);
      for (std::vector<ReductionView*>::const_iterator it = 
            src_views.begin(); it != src_views.end(); it++)
        record_view(*it);
      const std::pair<InstanceView*,unsigned> dst_key(dst_view, dst_fidx);
      std::vector<ReductionOpID> &redop_epochs = reduction_epochs[dst_key];
      FieldMask src_mask, dst_mask;
      src_mask.set_bit(src_fidx);
      dst_mask.set_bit(dst_fidx);
      // Always start scanning from the first redop index
      unsigned redop_index = 0;
      for (std::vector<ReductionView*>::const_iterator it = 
            src_views.begin(); it != src_views.end(); it++)
      {
        const ReductionOpID redop = (*it)->get_redop();
        CopyUpdate *update =
          new CopyUpdate(*it, src_mask, expr, redop, across_helper);
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
            reductions.resize(redop_index + 1);
        }
        reductions[redop_index][dst_view].insert(update, dst_mask);
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_preconditions(InstanceView *view, 
                                   bool reading, EventFieldExprs &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!preconditions.empty());
#endif
      WrapperReferenceMutator mutator(effects);
      AutoLock p_lock(pre_lock);
      EventFieldExprs &pre = reading ? src_pre[view] : dst_pre[view]; 
      for (EventFieldExprs::iterator eit = preconditions.begin();
            eit != preconditions.end(); eit++)
      {
        EventFieldExprs::iterator event_finder = pre.find(eit->first);
        if (event_finder != pre.end())
        {
          // Need to do the merge manually 
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
                eit->second.begin(); it != eit->second.end(); it++)
          {
            FieldMaskSet<IndexSpaceExpression>::iterator finder = 
              event_finder->second.find(it->first);
            if (finder == event_finder->second.end())
            {
              // Keep a reference in case we are deferred
              it->first->add_base_expression_reference(AGGREGATOR_REF,&mutator);
              event_finder->second.insert(it->first, it->second);
            }
            else
              finder.merge(it->second);
          }
        }
        else // We can just swap this over
        {
          // Keep references in case we are deferred
          for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                eit->second.begin(); it != eit->second.end(); it++)
            it->first->add_base_expression_reference(AGGREGATOR_REF, &mutator);
          pre[eit->first].swap(eit->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::record_precondition(InstanceView *view,
                                                 bool reading, ApEvent event,
                                                 const FieldMask &mask,
                                                 IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
      AutoLock p_lock(pre_lock);
      FieldMaskSet<IndexSpaceExpression> &event_pre = 
        reading ? src_pre[view][event] : dst_pre[view][event];
      FieldMaskSet<IndexSpaceExpression>::iterator finder = 
        event_pre.find(expr);
      if (finder == event_pre.end())
      {
        event_pre.insert(expr, mask);
        // Keep a reference in case we are deferred
        WrapperReferenceMutator mutator(effects);
        expr->add_base_expression_reference(AGGREGATOR_REF, &mutator);
      }
      else
        finder.merge(mask);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_updates(const PhysicalTraceInfo &trace_info,
                                       ApEvent precondition,
                                       const bool has_src_preconditions,
                                       const bool has_dst_preconditions,
                                       const bool need_deferral, unsigned pass,
                                       bool need_pass_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!sources.empty() || !reductions.empty());
#endif
      if (need_deferral || 
          (guard_precondition.exists() && !guard_precondition.has_triggered()))
      {
        CopyFillAggregation args(this, trace_info, precondition, 
                    has_src_preconditions, has_dst_preconditions, 
                    op->get_unique_op_id(), pass, need_pass_preconditions);
        op->runtime->issue_runtime_meta_task(args, 
                           LG_THROUGHPUT_DEFERRED_PRIORITY, guard_precondition);
        return;
      }
#ifdef DEBUG_LEGION
      assert(!guard_precondition.exists() || 
              guard_precondition.has_triggered());
#endif
      if (pass == 0)
      {
        // Perform updates from any sources first
        if (!sources.empty())
        {
          const RtEvent deferral_event = 
            perform_updates(sources, trace_info, precondition, 
                -1/*redop index*/, has_src_preconditions, 
                has_dst_preconditions, need_pass_preconditions);
          if (deferral_event.exists())
          {
            CopyFillAggregation args(this, trace_info, precondition, 
                        has_src_preconditions, has_dst_preconditions,
                        op->get_unique_op_id(), pass, false/*need pre*/);
            op->runtime->issue_runtime_meta_task(args, 
                             LG_THROUGHPUT_DEFERRED_PRIORITY, deferral_event);
            return;
          }
        }
        // We made it through the first pass
        pass++;
        need_pass_preconditions = true;
      }
      // Then apply any reductions that we might have
      if (!reductions.empty())
      {
#ifdef DEBUG_LEGION
        assert(pass > 0);
#endif
        // Skip any passes that we might have already done
        for (unsigned idx = pass-1; idx < reductions.size(); idx++)
        {
          const RtEvent deferral_event = 
            perform_updates(reductions[idx], trace_info, precondition,
                            idx/*redop index*/, has_src_preconditions, 
                            has_dst_preconditions, need_pass_preconditions);
          if (deferral_event.exists())
          {
            CopyFillAggregation args(this, trace_info, precondition, 
                        has_src_preconditions, has_dst_preconditions,
                        op->get_unique_op_id(), pass, false/*need pre*/);
            op->runtime->issue_runtime_meta_task(args, 
                             LG_THROUGHPUT_DEFERRED_PRIORITY, deferral_event);
            return;
          }
          // Made it through this pass
          pass++;
          need_pass_preconditions = true;
        }
      }
#ifndef NON_AGGRESSIVE_AGGREGATORS
      Runtime::trigger_event(guard_postcondition);
#endif
      // We can also trigger our guard event once the effects are applied
      if (!effects.empty())
        Runtime::trigger_event(effects_applied,
            Runtime::merge_events(effects));
      else
        Runtime::trigger_event(effects_applied);
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
        new_view->add_base_valid_ref(AGGREGATOR_REF, this);
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::find_reduction_preconditions(InstanceView *view,
        const PhysicalTraceInfo &trace_info, IndexSpaceExpression *copy_expr,
        const FieldMask &copy_mask, UniqueID op_id, unsigned redop_index,
        std::set<RtEvent> &preconditions_ready)
    //--------------------------------------------------------------------------
    {
      // Break up the fields in the copy mask based on their
      // different reduction operators, we'll handle the special
      // case where they all have the same reduction ID since
      // it is going to be very common
      FieldMask first_mask;
      ReductionOpID first_redop = 0;
      LegionMap<ReductionOpID,FieldMask> *other_masks = NULL;
      int fidx = copy_mask.find_first_set();
      while (fidx >= 0)
      {
        const std::pair<InstanceView*,unsigned> key(view, fidx);
#ifdef DEBUG_LEGION
        assert(reduction_epochs.find(key) != reduction_epochs.end());
        assert(redop_index < reduction_epochs[key].size());
#endif
        const ReductionOpID op = reduction_epochs[key][redop_index];
        if (op != first_redop)
        {
          if (first_redop != 0)
          {
            if (other_masks == NULL)
              other_masks = new LegionMap<ReductionOpID,FieldMask>();
            (*other_masks)[op].set_bit(fidx);
          }
          else
          {
            first_redop = op;
            first_mask.set_bit(fidx);
          }
        }
        else
          first_mask.set_bit(fidx);
        fidx = copy_mask.find_next_set(fidx+1);
      }
      RtEvent first_ready = view->find_copy_preconditions(
          false/*reading*/, first_redop, first_mask, copy_expr, op_id,
          dst_index, *this, trace_info.recording, local_space);
      if (first_ready.exists())
        preconditions_ready.insert(first_ready);
      if (other_masks != NULL)
      {
        for (LegionMap<ReductionOpID,FieldMask>::const_iterator it =
              other_masks->begin(); it != other_masks->end(); it++)
        {
          RtEvent pre_ready = view->find_copy_preconditions(
              false/*reading*/, it->first, it->second, copy_expr, op_id, 
              dst_index, *this, trace_info.recording, local_space);
          if (pre_ready.exists())
            preconditions_ready.insert(pre_ready);
        }
        delete other_masks;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent CopyFillAggregator::perform_updates(
         const LegionMap<InstanceView*,FieldMaskSet<Update> > &updates,
         const PhysicalTraceInfo &trace_info, const ApEvent all_precondition,
         int redop_index, const bool has_src_preconditions, 
         const bool has_dst_preconditions, const bool needs_preconditions)
    //--------------------------------------------------------------------------
    {
      if (needs_preconditions && 
          (!has_src_preconditions || !has_dst_preconditions))
      {
        // First compute the access expressions for all the copies
        InstanceFieldExprs dst_exprs, src_exprs;
        for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
              uit = updates.begin(); uit != updates.end(); uit++)
        {
          FieldMaskSet<IndexSpaceExpression> &dst_expr = dst_exprs[uit->first];
          for (FieldMaskSet<Update>::const_iterator it = 
                uit->second.begin(); it != uit->second.end(); it++)
          {
            // Update the destinations first
            if (!has_dst_preconditions)
            {
#ifdef DEBUG_LEGION
              // We should not have an across helper in this case
              assert(it->first->across_helper == NULL);
#endif
              FieldMaskSet<IndexSpaceExpression>::iterator finder = 
                dst_expr.find(it->first->expr);
              if (finder == dst_expr.end())
                dst_expr.insert(it->first->expr, it->second);
              else
                finder.merge(it->second);
            }
            // Now record the source expressions
            if (!has_src_preconditions)
              it->first->record_source_expressions(src_exprs);
          }
        }
        // Next compute the event preconditions for these accesses
        std::set<RtEvent> preconditions_ready; 
        const UniqueID op_id = op->get_unique_op_id();
        if (!has_dst_preconditions)
        {
          dst_pre.clear();
          for (InstanceFieldExprs::const_iterator dit = 
                dst_exprs.begin(); dit != dst_exprs.end(); dit++)
          {
            if (dit->second.size() == 1)
            {
              // No need to do any kind of sorts here
              IndexSpaceExpression *copy_expr = dit->second.begin()->first;
              const FieldMask &copy_mask = dit->second.get_valid_mask();
              // See if we're doing reductions or not
              if (redop_index < 0)
              {
                // No reductions so do the normal precondition test
                RtEvent pre_ready = dit->first->find_copy_preconditions(
                    false/*reading*/, 0/*redop*/, copy_mask, copy_expr, op_id, 
                    dst_index, *this, trace_info.recording, local_space);
                if (pre_ready.exists())
                  preconditions_ready.insert(pre_ready);
              }
              else
                find_reduction_preconditions(dit->first, trace_info, copy_expr,
                    copy_mask, op_id, redop_index, preconditions_ready);
            }
            else
            {
              // Sort into field sets and merge expressions
              LegionList<FieldSet<IndexSpaceExpression*> > sorted_exprs;
              dit->second.compute_field_sets(FieldMask(), sorted_exprs);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
                    it = sorted_exprs.begin(); it != sorted_exprs.end(); it++)
              {
                const FieldMask &copy_mask = it->set_mask; 
                IndexSpaceExpression *copy_expr = (it->elements.size() == 1) ?
                  *(it->elements.begin()) : 
                  forest->union_index_spaces(it->elements);
                if (redop_index < 0)
                {
                  RtEvent pre_ready = dit->first->find_copy_preconditions(
                      false/*reading*/, 0/*redop*/, copy_mask, copy_expr, op_id,
                      dst_index, *this, trace_info.recording, local_space);
                  if (pre_ready.exists())
                    preconditions_ready.insert(pre_ready);
                }
                else
                  find_reduction_preconditions(dit->first, trace_info,copy_expr,
                      copy_mask, op_id, redop_index, preconditions_ready);
              }
            }
          }
        }
        if (!has_src_preconditions)
        {
          src_pre.clear();
          for (InstanceFieldExprs::const_iterator sit = 
                src_exprs.begin(); sit != src_exprs.end(); sit++)
          {
            if (sit->second.size() == 1)
            {
              // No need to do any kind of sorts here
              IndexSpaceExpression *copy_expr = sit->second.begin()->first;
              const FieldMask &copy_mask = sit->second.get_valid_mask();
              RtEvent pre_ready = sit->first->find_copy_preconditions(
                  true/*reading*/, 0/*redop*/, copy_mask, copy_expr, op_id, 
                  src_index, *this, trace_info.recording, local_space);
              if (pre_ready.exists())
                preconditions_ready.insert(pre_ready);
            }
            else
            {
              // Sort into field sets and merge expressions
              LegionList<FieldSet<IndexSpaceExpression*> > sorted_exprs;
              sit->second.compute_field_sets(FieldMask(), sorted_exprs);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
                    it = sorted_exprs.begin(); it != sorted_exprs.end(); it++)
              {
                const FieldMask &copy_mask = it->set_mask; 
                IndexSpaceExpression *copy_expr = (it->elements.size() == 1) ?
                  *(it->elements.begin()) : 
                  forest->union_index_spaces(it->elements);
                RtEvent pre_ready = sit->first->find_copy_preconditions(
                    true/*reading*/, 0/*redop*/, copy_mask, copy_expr, op_id,
                    src_index, *this, trace_info.recording, local_space);
                if (pre_ready.exists())
                  preconditions_ready.insert(pre_ready);
              }
            }
          }
        }
        // If necessary wait until all we have all the preconditions
        if (!preconditions_ready.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(preconditions_ready);
          if (wait_on.exists())
            return wait_on;
        }
      }
#ifndef UNSAFE_AGGREGATION
      // Iterate over the destinations and compute updates that have the
      // same preconditions on different fields
      std::map<std::set<ApEvent>,ApEvent> merge_cache;
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
            uit = updates.begin(); uit != updates.end(); uit++)
      {
        EventFieldUpdates update_groups;
        const EventFieldExprs &dst_preconditions = dst_pre[uit->first];
        for (FieldMaskSet<Update>::const_iterator it = 
              uit->second.begin(); it != uit->second.end(); it++)
        {
          // Compute the preconditions for this update
          // This is a little tricky for across copies because we need
          // to make sure that all the fields are in same field space
          // which will be the source field space, so we need to convert
          // some field masks back to that space if necessary
          LegionMap<ApEvent,FieldMask> preconditions;
          // Compute the destination preconditions first
          if (!dst_preconditions.empty())
          {
            for (EventFieldExprs::const_iterator pit = 
                  dst_preconditions.begin(); pit != 
                  dst_preconditions.end(); pit++)
            {
              FieldMask set_overlap = it->second & pit->second.get_valid_mask();
              if (!set_overlap)
                continue;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator eit =
                    pit->second.begin(); eit != pit->second.end(); eit++)
              {
                const FieldMask overlap = set_overlap & eit->second;
                if (!overlap)
                  continue;
                IndexSpaceExpression *expr_overlap = 
                  forest->intersect_index_spaces(eit->first, it->first->expr);
                if (expr_overlap->is_empty())
                  continue;
#ifdef DEBUG_LEGION
                // Since this is an equivalence set update there should 
                // be no users that are using just a part of it, should 
                // be all or nothing, unless this is a copy across in 
                // which case it doesn't matter
                if (src_index != dst_index)
                  assert(expr_overlap->get_volume() == 
                          it->first->expr->get_volume());
#endif
                // Overlap on both so add it to the set
                LegionMap<ApEvent,FieldMask>::iterator finder = 
                  preconditions.find(pit->first);
                // Make sure to convert back to the source field space
                // in the case of across copies if necessary
                if (finder == preconditions.end())
                {
                  if (it->first->across_helper == NULL)
                    preconditions[pit->first] = overlap;
                  else
                    preconditions[pit->first] = 
                      it->first->across_helper->convert_dst_to_src(overlap);
                }
                else
                {
                  if (it->first->across_helper == NULL)
                    finder->second |= overlap;
                  else
                    finder->second |= 
                      it->first->across_helper->convert_dst_to_src(overlap);
                }
                set_overlap -= overlap;
                // If we found preconditions on all our fields then we're done
                if (!set_overlap)
                  break;
              }
            }
          }
          // The compute the source preconditions for this update
          it->first->compute_source_preconditions(forest,
#ifdef DEBUG_LEGION
                                                  (src_index != dst_index),
#endif
                                                  src_pre, preconditions);
          if (preconditions.empty())
            // NO precondition so enter it with a no event
            update_groups[ApEvent::NO_AP_EVENT].insert(it->first, 
                                                       it->first->src_mask);
          else if (preconditions.size() == 1)
          {
            LegionMap<ApEvent,FieldMask>::const_iterator first =
              preconditions.begin();
            update_groups[first->first].insert(it->first, first->second);
            const FieldMask remainder = it->first->src_mask - first->second;
            if (!!remainder)
              update_groups[ApEvent::NO_AP_EVENT].insert(it->first, remainder);
          }
          else
          {
            // Group event preconditions by fields
            LegionList<FieldSet<ApEvent> > grouped_events;
            compute_field_sets<ApEvent>(it->first->src_mask,
                                        preconditions, grouped_events);
            for (LegionList<FieldSet<ApEvent> >::const_iterator ait =
                  grouped_events.begin(); ait != grouped_events.end(); ait++) 
            {
              ApEvent key;
              if (ait->elements.size() > 1)
              {
                // See if the set is in the cache or we need to compute it 
                std::map<std::set<ApEvent>,ApEvent>::const_iterator finder =
                  merge_cache.find(ait->elements);
                if (finder == merge_cache.end())
                {
                  key = Runtime::merge_events(&trace_info, ait->elements);
                  merge_cache[ait->elements] = key;
                }
                else
                  key = finder->second;
              }
              else if (ait->elements.size() == 1)
                key = *(ait->elements.begin());
              FieldMaskSet<Update> &group = update_groups[key]; 
              FieldMaskSet<Update>::iterator finder = group.find(it->first);
              if (finder != group.end())
                finder.merge(ait->set_mask);
              else
                group.insert(it->first, ait->set_mask);
            }
          }
        }
        // Now iterate over events and group by fields
        for (EventFieldUpdates::const_iterator eit = 
              update_groups.begin(); eit != update_groups.end(); eit++)
        {
          // Merge in the over-arching precondition if necessary
          const ApEvent group_precondition = all_precondition.exists() ? 
            Runtime::merge_events(&trace_info, all_precondition, eit->first) :
            eit->first;
          const FieldMaskSet<Update> &group = eit->second;
#ifdef DEBUG_LEGION
          assert(!group.empty());
#endif
          if (group.size() == 1)
          {
            // Only one update so no need to try to group or merge 
            std::vector<FillUpdate*> fills;
            std::map<InstanceView* /*src*/,std::vector<CopyUpdate*> > copies;
            Update *update = group.begin()->first;
            update->sort_updates(copies, fills);
            const FieldMask &update_mask = group.get_valid_mask();
            if (!fills.empty())
              issue_fills(uit->first, fills, group_precondition, 
                          update_mask, trace_info, has_dst_preconditions);
            if (!copies.empty())
              issue_copies(uit->first, copies, group_precondition, 
                           update_mask, trace_info, has_dst_preconditions);
          }
          else
          {
            // Group by fields
            LegionList<FieldSet<Update*> > field_groups;
            group.compute_field_sets(FieldMask(), field_groups);
            for (LegionList<FieldSet<Update*> >::const_iterator fit =
                  field_groups.begin(); fit != field_groups.end(); fit++)
            {
              std::vector<FillUpdate*> fills;
              std::map<InstanceView* /*src*/,
                       std::vector<CopyUpdate*> > copies;
              for (std::set<Update*>::const_iterator it = 
                    fit->elements.begin(); it != fit->elements.end(); it++)
                (*it)->sort_updates(copies, fills);
              if (!fills.empty())
                issue_fills(uit->first, fills, group_precondition,
                            fit->set_mask, trace_info, has_dst_preconditions);
              if (!copies.empty())
                issue_copies(uit->first, copies, group_precondition, 
                             fit->set_mask, trace_info, has_dst_preconditions);
            }
          }
        }
      } // iterate over dst instances
#else
      // This is the unsafe aggregation routine that just looks at fields
      // and expressions and doesn't consider event preconditions
      for (LegionMap<InstanceView*,FieldMaskSet<Update> >::const_iterator
            uit = updates.begin(); uit != updates.end(); uit++)
      {
        const EventFieldExprs &dst_preconditions = dst_pre[uit->first];
        // Group by fields first
        LegionList<FieldSet<Update*> > field_groups;
        uit->second.compute_field_sets(FieldMask(), field_groups);
        for (LegionList<FieldSet<Update*> >::const_iterator fit = 
              field_groups.begin(); fit != field_groups.end(); fit++)
        {
          const FieldMask &dst_mask = fit->set_mask;
          // Now that we have the src mask for these operations group 
          // them into fills and copies and then do their event analysis
          std::vector<FillUpdate*> fills;
          std::map<InstanceView* /*src*/,std::vector<CopyUpdate*> > copies;
          for (std::set<Update*>::const_iterator it = fit->elements.begin();
                it != fit->elements.end(); it++)
            (*it)->sort_updates(copies, fills);
          // Issue the copies and fills
          if (!fills.empty())
          {
            std::set<ApEvent> preconditions;
            if (all_precondition.exists())
              preconditions.insert(all_precondition);
            std::set<IndexSpaceExpression*> fill_exprs;
            for (std::vector<FillUpdate*>::const_iterator it = 
                  fills.begin(); it != fills.end(); it++)
              fill_exprs.insert((*it)->expr);
            IndexSpaceExpression *fill_expr = (fill_exprs.size() == 1) ?
              *(fill_exprs.begin()) : forest->union_index_spaces(fill_exprs);
            for (EventFieldExprs::const_iterator eit = 
                  dst_preconditions.begin(); eit != 
                  dst_preconditions.end(); eit++)
            {
              // If there are no overlapping fields we can skip it
              if (dst_mask * eit->second.get_valid_mask())
                continue;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    eit->second.begin(); it != eit->second.end(); it++)
              {
                if (it->second * dst_mask)
                  continue;
                IndexSpaceExpression *expr_overlap = 
                  forest->intersect_index_spaces(it->first, fill_expr);
                if (!expr_overlap->is_empty())
                {
                  preconditions.insert(eit->first);
                  break;
                }
              }
            }
            CopyAcrossHelper *across_helper = fills[0]->across_helper; 
            const FieldMask src_mask = (across_helper == NULL) ? dst_mask :
              across_helper->convert_dst_to_src(dst_mask);
            if (!preconditions.empty())
            {
              const ApEvent fill_precondition = 
                Runtime::merge_events(&trace_info, preconditions);
              issue_fills(uit->first, fills, fill_precondition, 
                          src_mask, trace_info, has_dst_preconditions);
            }
            else
              issue_fills(uit->first, fills, ApEvent::NO_AP_EVENT, 
                          src_mask, trace_info, has_dst_preconditions);
          }
          if (!copies.empty())
          {
            std::set<ApEvent> preconditions;
            if (all_precondition.exists())
              preconditions.insert(all_precondition);
            std::set<IndexSpaceExpression*> copy_exprs;
            for (std::map<InstanceView*,std::vector<CopyUpdate*> >::
                  const_iterator cit = copies.begin(); 
                  cit != copies.end(); cit++)
              for (std::vector<CopyUpdate*>::const_iterator it = 
                    cit->second.begin(); it != cit->second.end(); it++)
                copy_exprs.insert((*it)->expr);
            IndexSpaceExpression *copy_expr = (copy_exprs.size() == 1) ?
              *(copy_exprs.begin()) : forest->union_index_spaces(copy_exprs); 
            // Destination preconditions first
            for (EventFieldExprs::const_iterator eit = 
                  dst_preconditions.begin(); eit != 
                  dst_preconditions.end(); eit++)
            {
              // If there are no overlapping fields we can skip it
              if (dst_mask * eit->second.get_valid_mask())
                continue;
              for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
                    eit->second.begin(); it != eit->second.end(); it++)
              {
                if (it->second * dst_mask)
                  continue;
                IndexSpaceExpression *expr_overlap = 
                  forest->intersect_index_spaces(it->first, copy_expr);
                if (!expr_overlap->is_empty())
                {
                  preconditions.insert(eit->first);
                  break;
                }
              }
            }
            // Then do the source preconditions
            // Be careful that we get the destination fields right in the
            // case that this is an across copy
            CopyAcrossHelper *across_helper = 
              copies.begin()->second[0]->across_helper;
            const FieldMask src_mask = (across_helper == NULL) ? dst_mask :
              across_helper->convert_dst_to_src(dst_mask);
            LegionMap<ApEvent,FieldMask> src_preconds;
            for (std::map<InstanceView*,std::vector<CopyUpdate*> >::
                  const_iterator cit = copies.begin(); cit != 
                  copies.end(); cit++)
            {
              for (std::vector<CopyUpdate*>::const_iterator it =
                    cit->second.begin(); it != cit->second.end(); it++)
              {
                (*it)->compute_source_preconditions(forest,
#ifdef DEBUG_LEGION
                                                    (src_index != dst_index),
#endif
                                                    src_pre, src_preconds);
              }
            }
            for (LegionMap<ApEvent,FieldMask>::const_iterator it =
                  src_preconds.begin(); it != src_preconds.end(); it++)
            {
              if (it->second * dst_mask)
                continue;
              preconditions.insert(it->first);
            }
            if (!preconditions.empty())
            {
              const ApEvent copy_precondition = 
                Runtime::merge_events(&trace_info, preconditions);
              issue_copies(uit->first, copies, copy_precondition, 
                           src_mask, trace_info, has_dst_preconditions);
            }
            else
              issue_copies(uit->first, copies, ApEvent::NO_AP_EVENT, 
                           src_mask, trace_info, has_dst_preconditions);

          }
        }
      }
#endif
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_fills(InstanceView *target,
                                         const std::vector<FillUpdate*> &fills,
                                         ApEvent precondition, 
                                         const FieldMask &fill_mask,
                                         const PhysicalTraceInfo &trace_info,
                                         const bool has_dst_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!fills.empty());
      assert(!!fill_mask); 
#endif
      const UniqueID op_id = op->get_unique_op_id();
      PhysicalManager *manager = target->get_manager();
      if (fills.size() == 1)
      {
        FillUpdate *update = fills[0];
#ifdef DEBUG_LEGION
        // Should cover all the fields
        assert(!(fill_mask - update->src_mask));
#endif
        IndexSpaceExpression *fill_expr = update->expr;
        FillView *fill_view = update->source;
        // Check to see if we need to do any work for tracing
        if (trace_info.recording)
        {
          if (tracing_src_fills == NULL)
            tracing_src_fills = new FieldMaskSet<FillView>();
          else
            tracing_src_fills->clear();
          // Record the source view
          tracing_src_fills->insert(fill_view, fill_mask);
          if (tracing_dsts == NULL)
            tracing_dsts = new FieldMaskSet<InstanceView>();
          else
            tracing_dsts->clear();
          // Record the destination view, convert field mask if necessary 
          if (update->across_helper != NULL)
          {
            const FieldMask dst_mask = 
              update->across_helper->convert_src_to_dst(fill_mask);
            tracing_dsts->insert(target, dst_mask);
          }
          else
            tracing_dsts->insert(target, fill_mask);
        }
        const ApEvent result = manager->fill_from(fill_view, precondition,
                                                  predicate_guard, fill_expr,
                                                  fill_mask, trace_info, 
                                                  tracing_src_fills,
                                                  tracing_dsts, effects,
                                                  fills[0]->across_helper);
        // Record the fill result in the destination 
        if (result.exists())
        {
          const RtEvent collect_event = trace_info.get_collect_event();
          if (update->across_helper != NULL)
          {
            const FieldMask dst_mask = 
                update->across_helper->convert_src_to_dst(fill_mask);
            target->add_copy_user(false/*reading*/, 0, result, collect_event,
                                  dst_mask, fill_expr, op_id, dst_index,
                                  effects, trace_info.recording, local_space);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result, 
                                  dst_mask, fill_expr);
          }
          else
          {
            target->add_copy_user(false/*reading*/, 0, result, collect_event,
                                  fill_mask, fill_expr, op_id,dst_index,
                                  effects, trace_info.recording, local_space);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result,
                                  fill_mask, fill_expr);
          }
          if (track_events)
            events.insert(result);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
#ifndef NDEBUG
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
          assert(!(fill_mask - (*it)->src_mask));
          // Should also have the same across helper as the first one
          assert(fills[0]->across_helper == (*it)->across_helper);
#endif
          exprs[(*it)->source].insert((*it)->expr);
        }
        const FieldMask dst_mask = 
          (fills[0]->across_helper == NULL) ? fill_mask : 
           fills[0]->across_helper->convert_src_to_dst(fill_mask);
        // See if we have any work to do for tracing
        if (trace_info.recording)
        {
          // Destination is the same for all the fills
          if (tracing_dsts == NULL)
            tracing_dsts = new FieldMaskSet<InstanceView>();
          else
            tracing_dsts->clear();
          tracing_dsts->insert(target, dst_mask);
        }
        for (std::map<FillView*,std::set<IndexSpaceExpression*> >::
              const_iterator it = exprs.begin(); it != exprs.end(); it++)
        {
          IndexSpaceExpression *fill_expr = (it->second.size() == 1) ?
            *(it->second.begin()) : forest->union_index_spaces(it->second);
          if (trace_info.recording)
          {
            if (tracing_src_fills == NULL)
              tracing_src_fills = new FieldMaskSet<FillView>();
            else
              tracing_src_fills->clear();
            // Record the source view
            tracing_src_fills->insert(it->first, fill_mask);
          }
          // See if we have any work to do for tracing
          const ApEvent result = manager->fill_from(it->first, precondition,
                                                    predicate_guard, fill_expr,
                                                    fill_mask, trace_info,
                                                    tracing_src_fills,
                                                    tracing_dsts, effects,
                                                    fills[0]->across_helper);
          const RtEvent collect_event = trace_info.get_collect_event();
          if (result.exists())
          {
            target->add_copy_user(false/*reading*/, 0, result, collect_event,
                                  dst_mask, fill_expr, op_id, dst_index,
                                  effects, trace_info.recording, local_space);
            if (track_events)
              events.insert(result);
            // Record this for the next iteration if necessary
            if (has_dst_preconditions)
              record_precondition(target, false/*reading*/, result,
                                  dst_mask, fill_expr);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void CopyFillAggregator::issue_copies(InstanceView *target, 
                              const std::map<InstanceView*,
                                             std::vector<CopyUpdate*> > &copies,
                              ApEvent precondition, const FieldMask &copy_mask,
                              const PhysicalTraceInfo &trace_info,
                              const bool has_dst_preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!copies.empty());
      assert(!!copy_mask);
#endif
      const UniqueID op_id = op->get_unique_op_id();
      PhysicalManager *target_manager = target->get_manager();
      for (std::map<InstanceView*,std::vector<CopyUpdate*> >::const_iterator
            cit = copies.begin(); cit != copies.end(); cit++)
      {
#ifdef DEBUG_LEGION
        assert(!cit->second.empty());
#endif
        if (cit->second.size() == 1)
        {
          // Easy case of a single update copy
          CopyUpdate *update = cit->second[0];
#ifdef DEBUG_LEGION
          // Should cover all the fields
          assert(!(copy_mask - update->src_mask));
#endif
          InstanceView *source = update->source;
          IndexSpaceExpression *copy_expr = update->expr;
          // See if we have any work to do for tracing
          if (trace_info.recording)
          {
            if (tracing_srcs == NULL)
              tracing_srcs = new FieldMaskSet<InstanceView>();
            else
              tracing_srcs->clear();
            tracing_srcs->insert(source, copy_mask);
            if (tracing_dsts == NULL)
              tracing_dsts = new FieldMaskSet<InstanceView>();
            else
              tracing_dsts->clear();
            // Handle the across case properly here
            if (update->across_helper != NULL)
            {
              const FieldMask dst_mask = 
                update->across_helper->convert_src_to_dst(copy_mask);
              tracing_dsts->insert(target, dst_mask);
            }
            else
              tracing_dsts->insert(target, copy_mask);
          }
          const ApEvent result = target_manager->copy_from(
                                    source->get_manager(), precondition,
                                    predicate_guard, update->redop,
                                    copy_expr, copy_mask, trace_info,
                                    tracing_srcs, tracing_dsts, effects,
                                    cit->second[0]->across_helper);
          if (result.exists())
          {
            const RtEvent collect_event = trace_info.get_collect_event();
            source->add_copy_user(true/*reading*/, 0, result, collect_event,
                                  copy_mask, copy_expr, op_id,src_index,
                                  effects, trace_info.recording, local_space);
            if (update->across_helper != NULL)
            {
              const FieldMask dst_mask = 
                update->across_helper->convert_src_to_dst(copy_mask);
              target->add_copy_user(false/*reading*/, update->redop, result, 
                        collect_event, dst_mask, copy_expr, op_id, dst_index,
                        effects, trace_info.recording, local_space);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    dst_mask, copy_expr);
            }
            else
            {
              target->add_copy_user(false/*reading*/, update->redop, result, 
                  collect_event, copy_mask, copy_expr, op_id,dst_index,
                  effects, trace_info.recording, local_space);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    copy_mask, copy_expr);
            }
            if (track_events)
              events.insert(result);
          }
        }
        else
        {
          // Have to group by source instances in order to merge together
          // different index space expressions for the same copy
          std::map<InstanceView*,std::set<IndexSpaceExpression*> > src_exprs;
          const ReductionOpID redop = cit->second[0]->redop;
          for (std::vector<CopyUpdate*>::const_iterator it = 
                cit->second.begin(); it != cit->second.end(); it++)
          {
#ifdef DEBUG_LEGION
            // Should cover all the fields
            assert(!(copy_mask - (*it)->src_mask));
            // Should have the same redop
            assert(redop == (*it)->redop);
            // Should also have the same across helper as the first one
            assert(cit->second[0]->across_helper == (*it)->across_helper);
#endif
            src_exprs[(*it)->source].insert((*it)->expr);
          }
          const FieldMask dst_mask = 
            (cit->second[0]->across_helper == NULL) ? copy_mask : 
             cit->second[0]->across_helper->convert_src_to_dst(copy_mask);
          // If we're tracing we can get the destination now
          if (trace_info.recording)
          {
            if (tracing_dsts == NULL)
              tracing_dsts = new FieldMaskSet<InstanceView>();
            else
              tracing_dsts->clear();
            tracing_dsts->insert(target, dst_mask);
          }
          for (std::map<InstanceView*,std::set<IndexSpaceExpression*> >::
                const_iterator it = src_exprs.begin(); 
                it != src_exprs.end(); it++)
          {
            IndexSpaceExpression *copy_expr = (it->second.size() == 1) ? 
              *(it->second.begin()) : forest->union_index_spaces(it->second);
            // If we're tracing then get the source information
            if (trace_info.recording)
            {
              if (tracing_srcs == NULL)
                tracing_srcs = new FieldMaskSet<InstanceView>();
              else
                tracing_srcs->clear();
              tracing_srcs->insert(it->first, copy_mask);
            }
            const ApEvent result = target_manager->copy_from(
                                    it->first->get_manager(), precondition,
                                    predicate_guard, redop, copy_expr,
                                    copy_mask, trace_info, 
                                    tracing_srcs, tracing_dsts, effects,
                                    cit->second[0]->across_helper);
            const RtEvent collect_event = trace_info.get_collect_event();
            if (result.exists())
            {
              it->first->add_copy_user(true/*reading*/, 0, result,collect_event,
                                  copy_mask, copy_expr, op_id,src_index,
                                  effects, trace_info.recording, local_space);
              target->add_copy_user(false/*reading*/,redop,result,collect_event,
                                  dst_mask, copy_expr, op_id, dst_index,
                                  effects, trace_info.recording, local_space);
              if (track_events)
                events.insert(result);
              // Record this for the next iteration if necessary
              if (has_dst_preconditions)
                record_precondition(target, false/*reading*/, result,
                                    dst_mask, copy_expr);
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
          cfargs->has_src, cfargs->has_dst, false/*needs deferral*/, 
          cfargs->pass, cfargs->need_pass_preconditions);
      cfargs->remove_recorder_reference();
    } 

    /////////////////////////////////////////////////////////////
    // Physical Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(Runtime *rt, Operation *o, unsigned idx, 
                                       const VersionInfo &info, bool h)
      : previous(rt->address_space), original_source(rt->address_space),
        runtime(rt), op(o), index(idx), version_manager(info.get_manager()), 
        owns_op(false), on_heap(h), remote_instances(NULL), restricted(false), 
        parallel_traversals(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(Runtime *rt, AddressSpaceID source, 
                                AddressSpaceID prev, Operation *o, unsigned idx,
                                VersionManager *man, bool h)
      : previous(prev), original_source(source), runtime(rt), op(o), index(idx),
        version_manager(man), owns_op(true), on_heap(h), remote_instances(NULL), 
        restricted(false), parallel_traversals(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::PhysicalAnalysis(const PhysicalAnalysis &rhs)
      : previous(0), original_source(0), runtime(NULL), op(NULL), index(0),
        version_manager(NULL), owns_op(false), on_heap(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PhysicalAnalysis::~PhysicalAnalysis(void)
    //--------------------------------------------------------------------------
    {
      if (remote_instances != NULL)
        delete remote_instances;
      if (owns_op && (op != NULL))
        delete op;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::traverse(EquivalenceSet *set,
                                    const FieldMask &mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    const bool cached_set,
                                    RtEvent precondition/*= NO_EVENT*/,
                                    const bool already_deferred /* = false*/)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
      {
        // This has to be the first time through and isn't really
        // a deferral of an the traversal since we haven't even
        // started the traversal yet
        defer_traversal(precondition, set, mask, deferral_events,applied_events,
                   cached_set, RtUserEvent::NO_RT_USER_EVENT, already_deferred);
      }
      else
      {
        if (cached_set)
        {
          FieldMask stale_mask;
          perform_traversal(set, mask, deferral_events, applied_events,
                            &stale_mask, cached_set, already_deferred);
          if (!!stale_mask)
            stale_sets.insert(set, stale_mask);
        }
        else
          perform_traversal(set, mask, deferral_events, applied_events,
                            NULL/*remove*/, cached_set, already_deferred);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::defer_traversal(RtEvent precondition,
                                           EquivalenceSet *set,
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           const bool cached_set,
                                           RtUserEvent deferral_event,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      // Make sure that we record that this has parallel traversals
      const DeferPerformTraversalArgs args(this, set, mask, deferral_event, 
                                          cached_set, already_deferred);
      runtime->issue_runtime_meta_task(args, 
          LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      deferral_events.insert(args.done_event);
      applied_events.insert(args.applied_event);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::perform_traversal(EquivalenceSet *set,
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             FieldMask *stale_mask,
                                             const bool cached_set,
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
      if (remote_instances == NULL)
        remote_instances = new FieldMaskSet<InstanceView>();
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
        remote_instances->insert(static_cast<InstanceView*>(view), mask);
      }
      bool remote_restrict;
      derez.deserialize(remote_restrict);
      if (remote_restrict)
        restricted = true;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::process_local_instances(
           const FieldMaskSet<InstanceView> &views, const bool local_restricted)
    //--------------------------------------------------------------------------
    {
      AutoLock a_lock(*this);
      if (remote_instances == NULL)
        remote_instances = new FieldMaskSet<InstanceView>();
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            views.begin(); it != views.end(); it++)
        if (it->first->is_instance_view())
          remote_instances->insert(it->first, it->second);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
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
    bool PhysicalAnalysis::report_instances(FieldMaskSet<InstanceView> &insts)
    //--------------------------------------------------------------------------
    {
      // No need for the lock since we shouldn't be mutating anything at 
      // this point anyway
      if (remote_instances != NULL)
        remote_instances->swap(insts);
      return restricted;
    }

    //--------------------------------------------------------------------------
    bool PhysicalAnalysis::update_alt_sets(EquivalenceSet *set, FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (parallel_traversals)
      {
        // Need the lock in this case
        AutoLock a_lock(*this);
        FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(set);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!mask)
            return true; // early out
          finder.merge(mask);
        }
        else
          alt_sets.insert(set, mask);
      }
      else
      {
        // No parallel traversals means we're the only thread
        FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(set);
        // Remove any fields we already traversed
        if (finder != alt_sets.end())
        {
          mask -= finder->second;
          // If we already traversed it then we don't need to do it again 
          if (!mask)
            return true; // early out
          finder.merge(mask);
        }
        else
          alt_sets.insert(set, mask);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::filter_alt_sets(EquivalenceSet *set, 
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (parallel_traversals)
      {
        // Need the lock if there are parallel traversals
        AutoLock a_lock(*this);
        FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(set);
        if (finder != alt_sets.end())
        {
          finder.filter(mask);
          if (!finder->second)
            alt_sets.erase(finder);
        }
      }
      else
      {
        // No parallel traversals means no lock needed
        FieldMaskSet<EquivalenceSet>::iterator finder = alt_sets.find(set);
        if (finder != alt_sets.end())
        {
          finder.filter(mask);
          if (!finder->second)
            alt_sets.erase(finder);
        }
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_stale_set(EquivalenceSet *set,
                                            const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      if (parallel_traversals)
      {
        // Lock needed if we're doing parallel traversals
        AutoLock a_lock(*this);    
        stale_sets.insert(set, mask);
      }
      else
        // No lock needed if we're the only one
        stale_sets.insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_remote(EquivalenceSet *set, 
                                         const FieldMask &mask,
                                         const AddressSpaceID owner,
                                         const bool cached_set)
    //--------------------------------------------------------------------------
    {
      const std::pair<AddressSpaceID,bool> key(owner, cached_set);
      if (parallel_traversals)
      {
        AutoLock a_lock(*this);
        remote_sets[key].insert(set, mask);
      }
      else
        // No lock needed if we're the only one
        remote_sets[key].insert(set, mask);
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::update_stale_equivalence_sets(
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // No need for the lock here since we know there are no 
      // races because there are no more traversals being performed
#ifdef DEBUG_LEGION
      assert(!stale_sets.empty());
#endif
      // Check to see if we are on the local node for the version manager
      // or whether we need to send a message to record the stale sets
      if (original_source != runtime->address_space)
      {
        RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize<size_t>(stale_sets.size());
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                stale_sets.begin(); it != stale_sets.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(version_manager);
          rez.serialize(applied);
        }
        runtime->send_equivalence_set_stale_update(original_source, rez);
        applied_events.insert(applied);
      }
      else // the local node case
      {
        const RtEvent done = version_manager->record_stale_sets(stale_sets);
        if (done.exists())
          applied_events.insert(done);
      }
    }

    //--------------------------------------------------------------------------
    void PhysicalAnalysis::record_instance(InstanceView *view, 
                                           const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
      // Lock held from caller
      if (remote_instances == NULL)
        remote_instances = new FieldMaskSet<InstanceView>();
      remote_instances->insert(view, mask);
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
        RtUserEvent done, bool cached, bool def)
      : LgTaskArgs<DeferPerformTraversalArgs>(ana->op->get_unique_op_id()),
        analysis(ana), set(s), mask(new FieldMask(m)), 
        applied_event(Runtime::create_rt_user_event()),
        done_event(done.exists() ? done : Runtime::create_rt_user_event()), 
        cached_set(cached), already_deferred(def)
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
          deferral_events, applied_events, dargs->cached_set, 
          RtEvent::NO_RT_EVENT, dargs->already_deferred);
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
                                     const VersionInfo &info, ReductionOpID red)
      : PhysicalAnalysis(rt, o, idx, info, false/*on heap*/), 
        redop(red), target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ValidInstAnalysis::ValidInstAnalysis(Runtime *rt, AddressSpaceID src, 
                   AddressSpaceID prev, Operation *o, unsigned idx,
                   VersionManager *man, ValidInstAnalysis *t, ReductionOpID red)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
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
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             FieldMask *stale_mask,
                                             const bool cached_set,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_valid_instances(*this, mask, deferral_events, applied_events, 
                                cached_set, already_deferred);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(redop);
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (remote_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(remote_instances->size());
            for (FieldMaskSet<InstanceView>::const_iterator it = 
                 remote_instances->begin(); it != remote_instances->end(); it++)
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
          target_analysis->process_local_instances(*remote_instances,
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      ValidInstAnalysis *analysis = new ValidInstAnalysis(runtime, 
          original_source, previous, op, index, version_manager, target, redop);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
                                  unsigned idx, const VersionInfo &info, 
                                  const FieldMaskSet<InstanceView> &valid_insts)
      : PhysicalAnalysis(rt, o, idx, info, false/*on heap*/), 
        valid_instances(valid_insts), target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    InvalidInstAnalysis::InvalidInstAnalysis(Runtime *rt, AddressSpaceID src, 
        AddressSpaceID prev, Operation *o, unsigned idx, VersionManager *man,
        InvalidInstAnalysis *t, const FieldMaskSet<InstanceView> &valid_insts)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
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
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             FieldMask *stale_mask,
                                             const bool cached_set,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->find_invalid_instances(*this, mask, deferral_events, applied_events, 
                                  cached_set, already_deferred);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize<size_t>(valid_instances.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                valid_instances.begin(); it != valid_instances.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          rez.serialize(target_analysis);
          rez.serialize(ready);
          rez.serialize(applied);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (remote_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(remote_instances->size());
            for (FieldMaskSet<InstanceView>::const_iterator it = 
                 remote_instances->begin(); it != remote_instances->end(); it++)
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
          target_analysis->process_local_instances(*remote_instances, 
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
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      FieldMaskSet<InstanceView> valid_instances;
      size_t num_valid_instances;
      derez.deserialize<size_t>(num_valid_instances);
      for (unsigned idx = 0; idx < num_valid_instances; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        InstanceView *view = static_cast<InstanceView*>(
            runtime->find_or_request_logical_view(did, ready));
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      InvalidInstAnalysis *analysis = new InvalidInstAnalysis(runtime, 
          original_source, previous, op, index, version_manager, 
          target, valid_instances);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Wait for the equivalence sets to be ready if necessary
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
                     const VersionInfo &info, const RegionRequirement &req,
                     RegionNode *rn, const InstanceSet &target_insts,
                     std::vector<InstanceView*> &target_vws,
                     const PhysicalTraceInfo &t_info,
                     const ApEvent pre, const ApEvent term,
                     const bool check, const bool record)
      : PhysicalAnalysis(rt, o, idx, info, true/*on heap*/), usage(req), 
        node(rn), target_instances(target_insts), target_views(target_vws), 
        trace_info(t_info), precondition(pre), term_event(term), 
        check_initialized(check && !IS_DISCARD(usage) && !IS_SIMULT(usage)), 
        record_valid(record), output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(Runtime *rt, AddressSpaceID src, 
                     AddressSpaceID prev, Operation *o, unsigned idx, 
                     VersionManager *man, const RegionUsage &use, 
                     RegionNode *rn, InstanceSet &target_insts,
                     std::vector<InstanceView*> &target_vws,
                     const PhysicalTraceInfo &info,
                     const RtEvent user_reg, const ApEvent pre, 
                     const ApEvent term, const bool check, const bool record)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
        usage(use), node(rn), target_instances(target_insts), 
        target_views(target_vws), trace_info(info), precondition(pre), 
        term_event(term), check_initialized(check), record_valid(record), 
        output_aggregator(NULL), remote_user_registered(user_reg)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UpdateAnalysis::UpdateAnalysis(const UpdateAnalysis &rhs)
      : PhysicalAnalysis(rhs), usage(rhs.usage), node(rhs.node), 
        target_instances(rhs.target_instances), target_views(rhs.target_views),
        trace_info(rhs.trace_info), precondition(rhs.precondition), 
        term_event(rhs.term_event), check_initialized(rhs.check_initialized), 
        record_valid(rhs.record_valid)
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
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           FieldMask *stale_mask,
                                           const bool cached_set,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->update_set(*this, mask, deferral_events, applied_events, 
                      stale_mask, cached_set, already_deferred);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          trace_info.pack_trace_info<false>(rez, applied_events, target);
          rez.serialize(precondition);
          rez.serialize(term_event);
          rez.serialize(updated);
          rez.serialize(remote_user_registered);
          rez.serialize(applied);
          rez.serialize(effects);
          rez.serialize(version_manager);
          rez.serialize<bool>(check_initialized);
          rez.serialize<bool>(record_valid);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (!input_aggregators.empty())
      {
        const bool needs_deferral = !already_deferred || 
          (input_aggregators.size() > 1);
        for (std::map<RtEvent,CopyFillAggregator*>::const_iterator it = 
              input_aggregators.begin(); it != input_aggregators.end(); it++)
        {
          it->second->issue_updates(trace_info, precondition,
              false/*has src*/, false/*has dst*/, needs_deferral);
#ifdef NON_AGGRESSIVE_AGGREGATORS
          if (!it->second->effects_applied.has_triggered())
            guard_events.insert(it->second->effects_applied);
#else
          if (!it->second->guard_postcondition.has_triggered())
            guard_events.insert(it->second->guard_postcondition);
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
        applied_events.insert(args.applied_event);
        return args.effects_event;
      }
      ApEvent result;
      if (output_aggregator != NULL)
      {
        output_aggregator->issue_updates(trace_info, term_event);
        // We need to wait for the aggregator updates to be applied
        // here before we can summarize the output
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!output_aggregator->effects_applied.has_triggered())
          output_aggregator->effects_applied.wait();
#else
        if (!output_aggregator->guard_postcondition.has_triggered())
          output_aggregator->guard_postcondition.wait();
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
      PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime, op);
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool check_initialized;
      derez.deserialize(check_initialized);
      bool record_valid;
      derez.deserialize(record_valid);
      bool cached_sets;
      derez.deserialize(cached_sets);

      RegionNode *node = runtime->forest->get_node(handle);
      // This takes ownership of the remote operation
      UpdateAnalysis *analysis = new UpdateAnalysis(runtime, original_source,
          previous, op, index, version_manager, usage, node, targets, 
          target_views, trace_info, remote_user_registered, precondition, 
          term_event, check_initialized, record_valid);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events; 
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
                                     unsigned idx, const VersionInfo &info)
      : PhysicalAnalysis(rt, o, idx, info, false/*on heap*/), 
        target_analysis(this)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    AcquireAnalysis::AcquireAnalysis(Runtime *rt, AddressSpaceID src, 
                      AddressSpaceID prev, Operation *o, unsigned idx, 
                      VersionManager *man, AcquireAnalysis *t)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
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
                                            const FieldMask &mask,
                                            std::set<RtEvent> &deferral_events,
                                            std::set<RtEvent> &applied_events,
                                            FieldMask *stale_mask,
                                            const bool cached_set,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->acquire_restrictions(*this, mask, deferral_events, applied_events, 
                                stale_mask, cached_set, already_deferred);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(target_analysis);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (remote_instances != NULL)
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(remote_instances->size());
            for (FieldMaskSet<InstanceView>::const_iterator it = 
                 remote_instances->begin(); it != remote_instances->end(); it++)
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
          target_analysis->process_local_instances(*remote_instances, 
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      // This takes ownership of the operation
      AcquireAnalysis *analysis = new AcquireAnalysis(runtime, original_source,
                                  previous, op, index, version_manager, target);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
                                     ApEvent pre, const VersionInfo &info,
                                     const PhysicalTraceInfo &t_info)
      : PhysicalAnalysis(rt, o, idx, info, false/*on heap*/), 
        precondition(pre), target_analysis(this), trace_info(t_info),
        release_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(Runtime *rt, AddressSpaceID src, 
            AddressSpaceID prev, Operation *o, unsigned idx,VersionManager *man,
            ApEvent pre, ReleaseAnalysis *t, const PhysicalTraceInfo &info)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
        precondition(pre), target_analysis(t), trace_info(info), 
        release_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReleaseAnalysis::ReleaseAnalysis(const ReleaseAnalysis &rhs)
      : PhysicalAnalysis(rhs), target_analysis(rhs.target_analysis), 
        trace_info(rhs.trace_info)
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
                                            const FieldMask &mask,
                                            std::set<RtEvent> &deferral_events,
                                            std::set<RtEvent> &applied_events,
                                            FieldMask *stale_mask,
                                            const bool cached_set,
                                            const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->release_restrictions(*this, mask, deferral_events, applied_events, 
                                stale_mask, cached_set, already_deferred);
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(precondition);
          rez.serialize(returned);
          rez.serialize(applied);
          rez.serialize(target_analysis);
          trace_info.pack_trace_info<false>(rez, applied_events, target);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      // See if we have any instance names to send back
      if ((target_analysis != this) && (remote_instances != NULL))
      {
        if (original_source != runtime->address_space)
        {
          const RtUserEvent response_event = Runtime::create_rt_user_event();
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(target_analysis);
            rez.serialize(response_event);
            rez.serialize<size_t>(remote_instances->size());
            for (FieldMaskSet<InstanceView>::const_iterator it = 
                 remote_instances->begin(); it != remote_instances->end(); it++)
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
          target_analysis->process_local_instances(*remote_instances, 
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
        if (!release_aggregator->guard_postcondition.has_triggered())
          guard_events.insert(release_aggregator->guard_postcondition);
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
        PhysicalTraceInfo::unpack_trace_info(derez, runtime, op);
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      ReleaseAnalysis *analysis = new ReleaseAnalysis(runtime, original_source,
        previous, op, index, version_manager, precondition, target, trace_info);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      RtEvent ready_event;
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
        unsigned src_idx, unsigned dst_idx, const VersionInfo &info, 
        const RegionRequirement &src_req,
        const RegionRequirement &dst_req, const InstanceSet &target_insts,
        const std::vector<InstanceView*> &target_vws, const ApEvent pre,
        const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned> &src_idxes,
        const std::vector<unsigned> &dst_idxes, 
        const PhysicalTraceInfo &t_info, const bool perf)
      : PhysicalAnalysis(rt, o, dst_idx, info, true/*on heap*/), 
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)), 
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), src_usage(src_req), 
        dst_usage(dst_req), src_region(src_req.region), 
        dst_region(dst_req.region), target_instances(target_insts),
        target_views(target_vws), precondition(pre),pred_guard(pred),redop(red),
        src_indexes(src_idxes), dst_indexes(dst_idxes), 
        across_helpers(perf ? std::vector<CopyAcrossHelper*>() :
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
        const std::vector<InstanceView*> &target_vws, const ApEvent pre,
        const PredEvent pred, const ReductionOpID red,
        const std::vector<unsigned> &src_idxes,
        const std::vector<unsigned> &dst_idxes, 
        const PhysicalTraceInfo &t_info, const bool perf)
      : PhysicalAnalysis(rt, src, prev, o, dst_idx,NULL/*man*/,true/*on heap*/),
        src_mask(perf ? FieldMask() : initialize_mask(src_idxes)), 
        dst_mask(perf ? FieldMask() : initialize_mask(dst_idxes)),
        src_index(src_idx), dst_index(dst_idx), 
        src_usage(src_use), dst_usage(dst_use), src_region(src_reg), 
        dst_region(dst_reg), target_instances(target_insts), 
        target_views(target_vws), precondition(pre),pred_guard(pred),redop(red),
        src_indexes(src_idxes), dst_indexes(dst_idxes), 
        across_helpers(perf ? std::vector<CopyAcrossHelper*>() :
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
        target_views(rhs.target_views), precondition(rhs.precondition),
        pred_guard(rhs.pred_guard), redop(rhs.redop), 
        src_indexes(rhs.src_indexes), dst_indexes(rhs.dst_indexes),
        across_helpers(rhs.across_helpers), trace_info(rhs.trace_info),
        perfect(rhs.perfect)
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
            src_index, dst_index, aggregator_guard, true/*track*/, pred_guard);
      }
      return across_aggregator;
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
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->first.second); // should not be a cached set here
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          trace_info.pack_trace_info<false>(rez, applied_events, target);
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
#ifdef DEBUG_LEGION
      // CopyAcrossAnalysis should have no alt-set tracking because 
      // individual equivalence sets may need to be traversed multiple times
      assert(alt_sets.empty());
      assert(stale_sets.empty());
#endif
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
        // Use the destination expr since we know we we're only actually
        // issuing copies for that particular expression
        if (local_exprs.size() > 1)
        {
          LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
          local_exprs.compute_field_sets(FieldMask(), field_sets);
          for (LegionList<FieldSet<IndexSpaceExpression*> >::const_iterator
                it = field_sets.begin(); it != field_sets.end(); it++)
          {
            IndexSpaceExpression *expr = (it->elements.size() == 1) ? 
              *(it->elements.begin()) :
              runtime->forest->union_index_spaces(it->elements);
            if (expr->is_empty())
              continue;
            for (unsigned idx = 0; idx < target_instances.size(); idx++)
            {
              const InstanceRef &ref = target_instances[idx];
              const ApEvent event = ref.get_ready_event();
              if (!event.exists())
                continue;
              const FieldMask &mask = ref.get_valid_fields();
              // Convert these to destination fields if necessary
              const FieldMask overlap = mask & (perfect ? it->set_mask :
                  across_helpers[idx]->convert_src_to_dst(it->set_mask));
              if (!overlap)
                continue;
              InstanceView *view = target_views[idx];
              across_aggregator->record_precondition(view, false/*reading*/,
                                                     event, overlap, expr);
            }
          }
        }
        else
        {
          FieldMaskSet<IndexSpaceExpression>::const_iterator first = 
            local_exprs.begin();
          if (!first->first->is_empty())
          {
            for (unsigned idx = 0; idx < target_instances.size(); idx++)
            {
              const InstanceRef &ref = target_instances[idx];
              const ApEvent event = ref.get_ready_event();
              if (!event.exists())
                continue;
              const FieldMask &mask = ref.get_valid_fields();
              // Convert these to destination fields if necessary
              const FieldMask overlap = mask & (perfect ? first->second : 
                  across_helpers[idx]->convert_src_to_dst(first->second));
              if (!overlap)
                continue;
              InstanceView *view = target_views[idx];
              across_aggregator->record_precondition(view, false/*reading*/,
                                               event, overlap, first->first);
            }
          }
        }
        across_aggregator->issue_updates(trace_info, precondition,
            false/*has src preconditions*/, true/*has dst preconditions*/);
        // Need to wait before we can get the summary
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!across_aggregator->effects_applied.has_triggered())
          return across_aggregator->effects_applied;
#else
        if (!across_aggregator->guard_postcondition.has_triggered())
          return across_aggregator->guard_postcondition;
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
        PhysicalTraceInfo::unpack_trace_info(derez, runtime, op); 

      std::vector<CopyAcrossHelper*> across_helpers;
      std::set<RtEvent> deferral_events, applied_events;
      RegionNode *dst_node = runtime->forest->get_node(dst_handle);
      IndexSpaceExpression *dst_expr = dst_node->get_index_space_expression();
      // Make sure that all our pointers are ready
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      // This takes ownership of the op and the across helpers
      CopyAcrossAnalysis *analysis = new CopyAcrossAnalysis(runtime, 
          original_source, previous, op, src_index, dst_index,
          src_usage, dst_usage, src_handle, dst_handle, 
          dst_instances, dst_views, precondition, pred_guard, redop, 
          src_indexes, dst_indexes, trace_info, perfect);
      analysis->add_reference();
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
      {
        EquivalenceSet *set = eq_sets[idx];
        // Check that the index spaces intersect
        IndexSpaceExpression *overlap = 
          runtime->forest->intersect_index_spaces(set->set_expr, dst_expr);
        if (overlap->is_empty())
          continue;
        set->issue_across_copies(*analysis, eq_masks[idx], overlap,
                                 deferral_events, applied_events);
      }
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
    static inline std::set<LogicalView*> overwrite_insert_helper(LogicalView *v)
    //--------------------------------------------------------------------------
    {
      std::set<LogicalView*> result;
      if (v != NULL)
        result.insert(v);
      return result;
    }
    
    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, Operation *o, 
                        unsigned idx, const RegionUsage &use,
                        const VersionInfo &info, LogicalView *view, 
                        const PhysicalTraceInfo &t_info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, o, idx, info, true/*on heap*/), usage(use), 
        views(overwrite_insert_helper(view)), trace_info(t_info), 
        precondition(pre), guard_event(guard), pred_guard(pred), 
        track_effects(track), add_restriction(restriction),
        output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, Operation *o, 
                        unsigned idx, const RegionUsage &use,
                        const VersionInfo &info,const std::set<LogicalView*> &v,
                        const PhysicalTraceInfo &t_info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, o, idx, info, true/*on heap*/), usage(use), 
        views(v), trace_info(t_info), precondition(pre), guard_event(guard), 
        pred_guard(pred), track_effects(track), add_restriction(restriction), 
        output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(Runtime *rt, AddressSpaceID src, 
                        AddressSpaceID prev, Operation *o, unsigned idx, 
                        VersionManager *man, const RegionUsage &use, 
                        const std::set<LogicalView*> &v,
                        const PhysicalTraceInfo &info,
                        const ApEvent pre, const RtEvent guard, 
                        const PredEvent pred, const bool track, 
                        const bool restriction)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/), 
        usage(use), views(v), trace_info(info), precondition(pre), 
        guard_event(guard), pred_guard(pred), track_effects(track), 
        add_restriction(restriction), output_aggregator(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(const OverwriteAnalysis &rhs)
      : PhysicalAnalysis(rhs), usage(rhs.usage), views(rhs.views),
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
                                             const FieldMask &mask,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events,
                                             FieldMask *stale_mask,
                                             const bool cached_set,
                                             const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->overwrite_set(*this, mask, deferral_events, applied_events,
                         stale_mask, cached_set, already_deferred);
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
      WrapperReferenceMutator mutator(applied_events);
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpace target = rit->first.first;
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
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(usage);
          rez.serialize<size_t>(views.size());
          if (!views.empty())
          {
            for (std::set<LogicalView*>::const_iterator it = 
                  views.begin(); it != views.end(); it++)
            {
              (*it)->add_base_valid_ref(REMOTE_DID_REF, &mutator);
              rez.serialize((*it)->did);  
            }
          }
          trace_info.pack_trace_info<false>(rez, applied_events, target);
          rez.serialize(pred_guard);
          rez.serialize(precondition);
          rez.serialize(guard_event);
          rez.serialize<bool>(add_restriction);
          rez.serialize(applied);
          rez.serialize(effects);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (output_aggregator != NULL)
      {
        output_aggregator->issue_updates(trace_info, precondition);
        // Need to wait before we can get the summary
#ifdef NON_AGGRESSIVE_AGGREGATORS
        if (!output_aggregator->effects_applied.has_triggered())
          return output_aggregator->effects_applied;
#else
        if (!output_aggregator->guard_postcondition.has_triggered())
          return output_aggregator->guard_postcondition;
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
      RemoteOp *op = 
        RemoteOp::unpack_remote_operation(derez, runtime, ready_events);
      unsigned index;
      derez.deserialize(index);
      RegionUsage usage;
      derez.deserialize(usage);
      std::set<LogicalView*> views;
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView *view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        views.insert(view);
      }
      const PhysicalTraceInfo trace_info = 
        PhysicalTraceInfo::unpack_trace_info(derez, runtime, op);
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      // This takes ownership of the operation
      OverwriteAnalysis *analysis = new OverwriteAnalysis(runtime,
          original_source, previous, op, index, version_manager, usage, 
          views, trace_info, precondition, guard_event, pred_guard, 
          effects.exists(),  add_restriction);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
                              const VersionInfo &info, InstanceView *view,
                              LogicalView *reg_view, const bool remove_restrict)
      : PhysicalAnalysis(rt, o, idx, info, true/*on heap*/), inst_view(view), 
        registration_view(reg_view), remove_restriction(remove_restrict)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FilterAnalysis::FilterAnalysis(Runtime *rt, AddressSpaceID src, 
                              AddressSpaceID prev, Operation *o, unsigned idx, 
                              VersionManager *man, InstanceView *view, 
                              LogicalView *reg_view, const bool remove_restrict)
      : PhysicalAnalysis(rt, src, prev, o, idx, man, true/*on heap*/),
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
                                           const FieldMask &mask,
                                           std::set<RtEvent> &deferral_events,
                                           std::set<RtEvent> &applied_events,
                                           FieldMask *stale_mask,
                                           const bool cached_set,
                                           const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->filter_set(*this, mask, deferral_events, applied_events,
                      stale_mask, cached_set, already_deferred);
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
      // Filter has no perform_updates call so we apply this here
      if (!stale_sets.empty())
        update_stale_equivalence_sets(applied_events);
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      WrapperReferenceMutator mutator(applied_events);
      for (LegionMap<std::pair<AddressSpaceID,bool>,
                     FieldMaskSet<EquivalenceSet> >::const_iterator 
            rit = remote_sets.begin(); rit != remote_sets.end(); rit++)
      {
#ifdef DEBUG_LEGION
        assert(!rit->second.empty());
#endif
        const AddressSpaceID target = rit->first.first;
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
          rez.serialize(index);
          if (inst_view != NULL)
          {
            inst_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
            rez.serialize(inst_view->did);
          }
          else
            rez.serialize<DistributedID>(0);
          if (registration_view != NULL)
          {
            registration_view->add_base_valid_ref(REMOTE_DID_REF, &mutator);
            rez.serialize(registration_view->did);
          }
          else
            rez.serialize<DistributedID>(0);
          rez.serialize(remove_restriction);
          rez.serialize(applied);
          rez.serialize(version_manager);
          rez.serialize<bool>(rit->first.second);
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
      VersionManager *version_manager;
      derez.deserialize(version_manager);
      bool cached_sets;
      derez.deserialize(cached_sets);

      // This takes ownership of the remote operation
      FilterAnalysis *analysis = new FilterAnalysis(runtime, original_source,
         previous, op, index, version_manager, inst_view, registration_view, 
         remove_restriction);
      analysis->add_reference();
      std::set<RtEvent> deferral_events, applied_events;
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->traverse(eq_sets[idx], eq_masks[idx], deferral_events, 
                           applied_events, cached_sets, ready_event);
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
      // Nasty race here: make sure the ready has triggered before
      // removing the references here because the views are not valid
      if (ready_event.exists() && !ready_event.has_triggered())
        ready_event.wait();
      if (inst_view != NULL)
        inst_view->send_remote_valid_decrement(previous, NULL, applied);
      if (registration_view != NULL)
        registration_view->send_remote_valid_decrement(previous, NULL, applied);
    }

    /////////////////////////////////////////////////////////////
    // Equivalence Set
    /////////////////////////////////////////////////////////////

    // C++ is dumb
    const VersionID EquivalenceSet::init_version;

    //--------------------------------------------------------------------------
    EquivalenceSet::DisjointPartitionRefinement::DisjointPartitionRefinement(
     EquivalenceSet *owner, IndexPartNode *p, std::set<RtEvent> &applied_events)
      : owner_did(owner->did), partition(p), total_child_volume(0),
        partition_volume(partition->get_union_expression()->get_volume())
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(partition->is_disjoint());
#endif
      WrapperReferenceMutator mutator(applied_events);
      partition->add_nested_valid_ref(owner_did, &mutator);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DisjointPartitionRefinement::DisjointPartitionRefinement(
      const DisjointPartitionRefinement &rhs, std::set<RtEvent> &applied_events)
      : owner_did(rhs.owner_did), partition(rhs.partition), 
        children(rhs.get_children()), total_child_volume(children.size()), 
        partition_volume(rhs.get_volume())
    //--------------------------------------------------------------------------
    {
      WrapperReferenceMutator mutator(applied_events);
      partition->add_nested_valid_ref(owner_did, &mutator);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DisjointPartitionRefinement::~DisjointPartitionRefinement(
                                                                           void)
    //--------------------------------------------------------------------------
    {
      if (partition->remove_nested_valid_ref(owner_did))
        delete partition;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::DisjointPartitionRefinement::add_child(
                                    IndexSpaceNode *node, EquivalenceSet *child)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(children.find(node) == children.end());
#endif
      children[node] = child;
      total_child_volume += node->get_volume();
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EquivalenceSet::DisjointPartitionRefinement::find_child(
                                                     IndexSpaceNode *node) const
    //--------------------------------------------------------------------------
    {
      std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator finder = 
        children.find(node);
      if (finder == children.end())
        return NULL;
      return finder->second;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(Runtime *rt, DistributedID did,
                                   AddressSpaceID owner, AddressSpace logical,
                                   IndexSpaceExpression *expr,
                                   IndexSpaceNode *node,
                                   bool reg_now)
      : DistributedCollectable(rt, 
          LEGION_DISTRIBUTED_HELP_ENCODE(did, EQUIVALENCE_SET_DC), 
          owner, reg_now), set_expr(expr),
        index_space_node(node), logical_owner_space(logical),
        eq_state(is_logical_owner() ? MAPPING_STATE : INVALID_STATE), 
        next_deferral_precondition(0), subset_exprs(NULL), migration_index(0),
        sample_count(0), pending_analyses(0)
    //--------------------------------------------------------------------------
    {
      set_expr->add_nested_expression_reference(did);
      if (index_space_node != NULL)
      {
#ifdef DEBUG_LEGION
        // These two index space expressions should be equivalent
        // Although they don't have to be the same
        // These assertions are pretty expensive so we'll comment them
        // out for now, but put them back in if you think this invariant
        // is being violated
        //assert(runtime->forest->subtract_index_spaces(index_space_node,
        //                                              set_expr)->is_empty());
        //assert(runtime->forest->subtract_index_spaces(set_expr,
        //                                      index_space_node)->is_empty());
#endif
        index_space_node->add_nested_resource_ref(did);
      }
#ifdef LEGION_GC
      log_garbage.info("GC Equivalence Set %lld %d",
          LEGION_DISTRIBUTED_ID_FILTER(this->did), local_space);
#endif
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::EquivalenceSet(const EquivalenceSet &rhs)
      : DistributedCollectable(rhs), set_expr(NULL), index_space_node(NULL), 
        logical_owner_space(rhs.logical_owner_space)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::~EquivalenceSet(void)
    //--------------------------------------------------------------------------
    {
      if (set_expr->remove_nested_expression_reference(did))
        delete set_expr;
      if ((index_space_node != NULL) && 
          index_space_node->remove_nested_resource_ref(did))
        delete index_space_node;
      if (!subsets.empty())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
          if (it->first->remove_nested_resource_ref(did))
            delete it->first;
        subsets.clear();
      }
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
      }
      if (!reduction_instances.empty())
      {
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = reduction_instances.begin(); 
              rit != reduction_instances.end(); rit++)
        {
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
            if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
        }
      }
      if (!restricted_instances.empty())
      {
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              restricted_instances.begin(); it != 
              restricted_instances.end(); it++)
          if (it->first->remove_nested_valid_ref(did))
            delete it->first;
      }
      if (!disjoint_partition_refinements.empty())
      {
        for (FieldMaskSet<DisjointPartitionRefinement>::const_iterator it =
              disjoint_partition_refinements.begin(); it !=
              disjoint_partition_refinements.end(); it++)
          delete it->first;
      }
      if (!unrefined_remainders.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it =
              unrefined_remainders.begin(); it != 
              unrefined_remainders.end(); it++)
          if (it->first->remove_nested_expression_reference(did))
            delete it->first;
      }
      if (subset_exprs != NULL)
        delete subset_exprs;
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
    void EquivalenceSet::trigger_pending_analysis_event(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(waiting_event.exists());
#endif
      Runtime::trigger_event(waiting_event);
      waiting_event = RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_active(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_inactive(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_valid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::notify_invalid(ReferenceMutator *mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    } 

    //--------------------------------------------------------------------------
    AddressSpaceID EquivalenceSet::clone_from(const EquivalenceSet *parent,
                                              const FieldMask &clone_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Should be cloning from the parent on it's owner space
      assert(parent->logical_owner_space == this->local_space);
#endif
      // Take our lock in exclusive mode since we're going to be updating
      // our data structures
      AutoLock eq(eq_lock);
      // Check to see if we're the logical owner, if not then tell 
      // the refinement task where it should send the data
      if (!is_logical_owner())
        return logical_owner_space;
      // We are the logical owner so clone the meta data
      // No need for a mutator here since all the views already
      // have valid references being held by the parent equivalence set
      if (!parent->valid_instances.empty() && 
          !(clone_mask * parent->valid_instances.get_valid_mask()))
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              parent->valid_instances.begin(); it !=
              parent->valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          if (this->valid_instances.insert(it->first, overlap))
            it->first->add_nested_valid_ref(did);
        }
      }
      if (!!parent->reduction_fields)
      {
        const FieldMask reduc_overlap = parent->reduction_fields & clone_mask;
        if (!!reduc_overlap)
        {
          this->reduction_fields |= reduc_overlap;
          int fidx = reduc_overlap.find_first_set();
          while (fidx >= 0)
          {
            std::vector<ReductionView*> &reduc_insts = 
              this->reduction_instances[fidx];
#ifdef DEBUG_LEGION
            assert(reduc_insts.empty());
#endif
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = parent->reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != parent->reduction_instances.end());
#endif
            reduc_insts = finder->second;
            for (unsigned idx = 0; idx < reduc_insts.size(); idx++)
              reduc_insts[idx]->add_nested_valid_ref(did);
            fidx = reduc_overlap.find_next_set(fidx+1);
          }
        }
      }
      if (!parent->restricted_instances.empty() &&
          !(clone_mask * parent->restricted_instances.get_valid_mask()))
      {
        this->restricted_fields |= (clone_mask & parent->restricted_fields);
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              parent->restricted_instances.begin(); it !=
              parent->restricted_instances.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          if (this->restricted_instances.insert(it->first, overlap))
            it->first->add_nested_valid_ref(did);
        }
      }
      if (!parent->update_guards.empty() &&
          !(clone_mask * parent->update_guards.get_valid_mask()))
      {
        for (FieldMaskSet<CopyFillGuard>::const_iterator it = 
              parent->update_guards.begin(); it != 
              parent->update_guards.end(); it++)
        {
          const FieldMask overlap = it->second & clone_mask;
          if (!overlap)
            continue;
          // Only want to record this if it isn't in the process of
          // being pruned out. The record_guard_set method will check
          // for this return true if it is not being pruned out
          if (it->first->record_guard_set(this))
            update_guards.insert(it->first, overlap);
        }
      }
      // Return our space since we stored the data here
      return local_space;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::remove_update_guard(CopyFillGuard *guard)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      // If we're no longer the logical owner then it's because we were
      // migrated and there should be no guards so we're done
      if (!is_logical_owner() && update_guards.empty())
        return;
      // We could get here when we're not the logical owner if we've unpacked
      // ourselves but haven't become the owner yet, in which case we still
      // need to prune ourselves out of the list
      FieldMaskSet<CopyFillGuard>::iterator finder = update_guards.find(guard);
      // It's also possible that the equivalence set is migrated away and
      // then migrated back before this guard is removed in which case we
      // won't find it in the update guards and can safely ignore it
      if (finder == update_guards.end())
        return;
      const bool should_tighten = !!finder->second;
      update_guards.erase(finder);
      if (should_tighten)
        update_guards.tighten_valid_mask();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_unrefined_remainder(AutoLock &eq,
                                                       const FieldMask &mask,
                                                       AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      if (!is_logical_owner())
        return;
      bool first_pass = true;
      do 
      {
        // If this isn't the first pass then we need to wait
        if (!first_pass)
        {
#ifdef DEBUG_LEGION
          assert(waiting_event.exists());
#endif
          const RtEvent wait_on = waiting_event;
          eq.release();
          if (!wait_on.has_triggered())
            wait_on.wait();
          eq.reacquire();
          // When we wake up we have to do all the checks again
          // in case there were fields that weren't refined before
          // but are (partially or being) refined now
#ifdef DEBUG_LEGION
          // Should never be migrated while we are waiting here
          assert(is_logical_owner());
#endif
        }
        else
          first_pass = false;
        // Check for any disjoint pieces
        if (!disjoint_partition_refinements.empty())
        {
          FieldMask disjoint_overlap = 
            disjoint_partition_refinements.get_valid_mask() & mask;
          if (!!disjoint_overlap)
          {
            std::vector<DisjointPartitionRefinement*> to_delete;
            for (FieldMaskSet<DisjointPartitionRefinement>::iterator it = 
                  disjoint_partition_refinements.begin(); it !=
                  disjoint_partition_refinements.end(); it++)
            {
              const FieldMask overlap = it->second & disjoint_overlap;
              if (!overlap)
                continue;
              finalize_disjoint_refinement(it->first, overlap);
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              disjoint_overlap -= overlap;
              if (!disjoint_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<DisjointPartitionRefinement*>::const_iterator 
                    it = to_delete.begin(); it != to_delete.end(); it++)
              {
                disjoint_partition_refinements.erase(*it);
                delete (*it);
              }
              disjoint_partition_refinements.tighten_valid_mask();
            }
          }
        }
        // Check for unrefined remainder pieces too
        if (!unrefined_remainders.empty())
        {
          FieldMask unrefined_overlap = 
            unrefined_remainders.get_valid_mask() & mask;
          if (!!unrefined_overlap)
          {
            std::vector<IndexSpaceExpression*> to_delete;
            for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
                  unrefined_remainders.begin(); it != 
                  unrefined_remainders.end(); it++)
            {
              const FieldMask overlap = it->second & unrefined_overlap;
              if (!overlap)
                continue;
              add_pending_refinement(it->first, overlap, NULL/*node*/, source); 
              it.filter(overlap);
              if (!it->second)
                to_delete.push_back(it->first);
              unrefined_overlap -= overlap;
              if (!unrefined_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<IndexSpaceExpression*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                unrefined_remainders.erase(*it);
                if ((*it)->remove_nested_expression_reference(did))
                  delete (*it);
              }
              unrefined_remainders.tighten_valid_mask();
            }
          }
        }
      } 
      // See if we need to wait for any refinements to finish
      while (!(mask * pending_refinements.get_valid_mask()) ||
              !(mask * refining_fields));
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::refresh_refinement(RayTracer *target, 
                                const FieldMask &mask, RtUserEvent refresh_done)
    //--------------------------------------------------------------------------
    {
      ray_trace_equivalence_sets(target, set_expr, mask, 
          (index_space_node == NULL) ? IndexSpace::NO_SPACE : 
            index_space_node->handle, runtime->address_space, refresh_done);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::ray_trace_equivalence_sets(RayTracer *target,
                                                    IndexSpaceExpression *expr,
                                                    FieldMask ray_mask,
                                                    IndexSpace handle,
                                                    AddressSpaceID source,
                                                    RtUserEvent trace_done,
                                                    RtUserEvent deferral_event)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = runtime->forest;
#ifdef DEBUG_LEGION
      assert(expr != NULL);
      // An expensive sanity check if you want to turn it on
      //assert(forest->subtract_index_spaces(expr, set_expr)->is_empty());
#endif
      RtEvent refinement_done;
      std::set<RtEvent> done_events;
      FieldMaskSet<EquivalenceSet> to_traverse, pending_to_traverse;
      std::map<EquivalenceSet*,IndexSpaceExpression*> to_traverse_exprs;
      {
        // Try to get the lock, if we don't get it build a continuation 
        AutoTryLock eq(eq_lock);
        if (!eq.has_lock())
        {
          // We didn't get the lock so build a continuation
          // We need a name for our completion event that we can use for
          // the atomic compare and swap below
          if (!deferral_event.exists())
          {
            // If we haven't already been deferred then we need to 
            // add ourselves to the back of the list of deferrals
            deferral_event = Runtime::create_rt_user_event();
            const RtEvent continuation_pre = 
              chain_deferral_events(deferral_event);
            DeferRayTraceArgs args(this, target, expr, handle, source, 
                                   trace_done, deferral_event, ray_mask);
            runtime->issue_runtime_meta_task(args, 
                            LG_THROUGHPUT_DEFERRED_PRIORITY, continuation_pre);
          }
          else
          {
            // We've already been deferred and our precondition has already
            // triggered so just launch ourselves again whenever the lock
            // should be ready to try again
            DeferRayTraceArgs args(this, target, expr, handle, source, 
                                   trace_done, deferral_event, ray_mask);
            runtime->issue_runtime_meta_task(args,
                              LG_THROUGHPUT_DEFERRED_PRIORITY, eq.try_next());
          }
          return;
        }
        else if (!is_logical_owner())
        {
          // If we're not the owner node then send the request there
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize(target);
            expr->pack_expression(rez, logical_owner_space);
            rez.serialize(ray_mask);
            rez.serialize(handle);
            rez.serialize(source);
            rez.serialize(trace_done);
          }
          runtime->send_equivalence_set_ray_trace_request(logical_owner_space,
                                                          rez);
          // Trigger our deferral event if we had one
          if (deferral_event.exists())
            Runtime::trigger_event(deferral_event);
          return;
        }
        else if ((eq_state == REFINING_STATE) && 
                  !(ray_mask * refining_fields))
        {
          if (!transition_event.exists())
            transition_event = Runtime::create_rt_user_event();
          // If we're refining then we also need to defer this until 
          // the refinements that interfere with us are done
          DeferRayTraceArgs args(this, target, expr, handle, source, 
                                 trace_done, deferral_event, ray_mask);
          runtime->issue_runtime_meta_task(args,
                            LG_THROUGHPUT_DEFERRED_PRIORITY, transition_event);
          return;
        }
        // Handle the special case where we are exactly representing the
        // index space and we have not been refined yet. This a performance
        // optimization only and is not required for correctness
        if (handle.exists() && (index_space_node != NULL) &&
            (index_space_node->handle == handle) && 
            (subsets.empty() || (ray_mask * subsets.get_valid_mask())))
        {
          // Just record this as one of the results
          if (source != runtime->address_space)
          {
            // Not local so we need to send a message
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(ray_mask);
              rez.serialize(target);
              rez.serialize(trace_done);
            }
            runtime->send_equivalence_set_ray_trace_response(source, rez);
          }
          else // Local so we can update this directly
          {
            target->record_equivalence_set(this, ray_mask);
            Runtime::trigger_event(trace_done);
          } 
          // We're done with our traversal so trigger the deferral event
          if (deferral_event.exists())
            Runtime::trigger_event(deferral_event);
          return;
        }
        // First check to see which fields are in a disjoint refinement
        // and whether we can continue doing the disjoint refinement
        if (!disjoint_partition_refinements.empty())
        {
#ifdef DEBUG_LEGION
          assert(index_space_node != NULL);
#endif
          FieldMask disjoint_overlap = ray_mask & 
            disjoint_partition_refinements.get_valid_mask();
          if (!!disjoint_overlap)
          {
            FieldMaskSet<DisjointPartitionRefinement> to_add;
            std::vector<DisjointPartitionRefinement*> to_delete;
            // Iterate over the disjoint partition refinements and see 
            // which ones we overlap with
            for (FieldMaskSet<DisjointPartitionRefinement>::iterator it =
                  disjoint_partition_refinements.begin(); it !=
                  disjoint_partition_refinements.end(); it++)
            {
              FieldMask overlap = it->second & disjoint_overlap;
              if (!overlap)
                continue;
              // Remove this from the disjoint overlap now in case
              // we end up removing overlap fields later
              disjoint_overlap -= overlap;
              // This is the special case where we are refining 
              // a disjoint partition and all the refinements so far
              // have been specific instances of a subregion of the
              // disjoint partition, check to see if that is still true
              if (handle.exists())
              {
                IndexSpaceNode *node = runtime->forest->get_node(handle);
                if (node->parent == it->first->partition)
                {
                  // Record that we're handling all these ray fields
                  // before we go about filtering the fields out of overlap
                  ray_mask -= overlap;
                  // Another sub-region of the disjoint partition
                  // See if we already made the refinement or not
                  EquivalenceSet *child = it->first->find_child(node);
                  // If child is NULL then we haven't made it yet
                  if (child == NULL)
                  {
                    // We want to maintain the invariant that a disjoint
                    // partition refinement represents all the children
                    // being refined for all the same fields, so we need
                    // to split this disjoint partition refinement into
                    // two if there is a difference in fields
                    const FieldMask non_overlap = it->second - overlap;
                    if (!!non_overlap)
                    {
                      // Make a new disjoint partition refinement that is
                      // a copy of the old one up to this point
                      DisjointPartitionRefinement *copy_refinement = 
                        new DisjointPartitionRefinement(*(it->first),
                                                        done_events);
                      to_add.insert(copy_refinement, non_overlap);
                      // Filter the fields down to just the overlap
                      it.filter(non_overlap);
                    }
#ifdef DEBUG_LEGION
                    assert(it->second == overlap);
#endif
                    // Refine this for all the fields in the disjoint 
                    // partition refinement to maintain the invariant that
                    // all these chidren have been refined for all fields
                    child = add_pending_refinement(expr, overlap, node, source);
                    pending_to_traverse.insert(child, overlap);
                    to_traverse_exprs[child] = expr;
                    // If this is a pending refinement then we'll need to
                    // wait for it before traversing farther
                    if (!refinement_done.exists())
                    {
#ifdef DEBUG_LEGION
                      assert(waiting_event.exists());
#endif
                      refinement_done = waiting_event;
                    }
                    // Record this child for the future
                    it->first->add_child(node, child);
                    // Check to see if we've finished this disjoint partition
                    if (it->first->is_refined())
                    {
                      // If we're done with this disjoint pending partition
                      // then we can remove it from the set
                      to_delete.push_back(it->first); 
                      // If this wasn't a complete partition then we need to 
                      // add the difference into the remainder
                      if (!it->first->partition->is_complete())
                      {
                        IndexSpaceExpression *diff_expr = 
                          runtime->forest->subtract_index_spaces(set_expr, 
                              it->first->partition->get_union_expression());
#ifdef DEBUG_LEGION
                        assert((diff_expr != NULL) && !diff_expr->is_empty());
                        assert(unrefined_remainders.get_valid_mask() * overlap);
#endif
                        if (unrefined_remainders.insert(diff_expr, overlap))
                          diff_expr->add_nested_expression_reference(did);
                      }
                    }
                    // Remove these fields from the overlap indicating
                    // that we handled them
                    overlap.clear();
                  }
                  else
                  {
                    // Figure out which fields have already been refined
                    // and which ones are still pending, issue refinements
                    // for any fields that haven't been refined yet
                    FieldMaskSet<EquivalenceSet>::iterator finder = 
                      subsets.find(child);
                    if (finder != subsets.end())
                    {
                      const FieldMask eq_valid = overlap & finder->second;
                      if (!!eq_valid)
                      {
                        to_traverse.insert(child, eq_valid);
                        to_traverse_exprs[child] = expr;
                        overlap -= eq_valid;
                      }
                    }
                    // If we couldn't find it in the already valid set, check
                    // also in the pending refinements
                    if (!!overlap)
                    {
                      finder = pending_refinements.find(child);
                      if (finder != pending_refinements.end())
                      {
#ifdef DEBUG_LEGION
                        // All overlap fields should be dominated
                        assert(!(overlap - finder->second));
#endif
                        pending_to_traverse.insert(child, overlap);
                        to_traverse_exprs[child] = expr;
                        overlap.clear();
                        // If this is a pending refinement then we'll need to
                        // wait for it before traversing farther
                        if (!refinement_done.exists())
                        {
#ifdef DEBUG_LEGION
                          assert(waiting_event.exists());
#endif
                          refinement_done = waiting_event;
                        }
                      }
                    }
#ifdef DEBUG_LEGION
                    // Should have handled all the fields at this point
                    assert(!overlap);
#endif
                  }
                }
              }
              // If we get here and we still haven't done a disjoint
              // refinement then we can no longer allow it to continue
              if (!!overlap)
              {
                finalize_disjoint_refinement(it->first, overlap);
                it.filter(overlap);
                if (!it->second)
                  to_delete.push_back(it->first);
              }
              // If we handled our disjoint overlap fields then we're done
              if (!disjoint_overlap)
                break;
            }
            if (!to_delete.empty())
            {
              for (std::vector<DisjointPartitionRefinement*>::const_iterator 
                    it = to_delete.begin(); it != to_delete.end(); it++)
              {
                disjoint_partition_refinements.erase(*it);
                delete (*it);
              }
              disjoint_partition_refinements.tighten_valid_mask();
            }
            if (!to_add.empty())
            {
              if (!disjoint_partition_refinements.empty())
                for (FieldMaskSet<DisjointPartitionRefinement>::const_iterator
                      it = to_add.begin(); it != to_add.end(); it++)
                  disjoint_partition_refinements.insert(it->first, it->second);
              else
                disjoint_partition_refinements.swap(to_add);
            }
          }
        }
        // Next handle any fields which are refined or pending refined
        if (!!ray_mask)
        {
          FieldMaskSet<IndexSpaceExpression> intersections;
          if (!pending_refinements.empty() && 
              !(ray_mask * pending_refinements.get_valid_mask()))
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  pending_refinements.begin(); it != 
                  pending_refinements.end(); it++)
            {
              const FieldMask overlap = it->second & ray_mask;
              if (!overlap)
                continue;
              // Next check for expression overlap
              IndexSpaceExpression *expr_overlap = 
                forest->intersect_index_spaces(expr, it->first->set_expr);
              if (expr_overlap->is_empty())
                continue;
              pending_to_traverse.insert(it->first, overlap);
              to_traverse_exprs[it->first] = expr_overlap;
              intersections.insert(expr_overlap, overlap);
              // If this is a pending refinement then we'll need to
              // wait for it before traversing farther
              if (!refinement_done.exists())
              {
#ifdef DEBUG_LEGION
                assert(waiting_event.exists());
#endif
                refinement_done = waiting_event;
              }
            }
          }
          if (!subsets.empty() && 
              !(ray_mask * subsets.get_valid_mask()))
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  subsets.begin(); it != subsets.end(); it++)
            {
              const FieldMask overlap = it->second & ray_mask;
              if (!overlap)
                continue;
              // Next check for expression overlap
              IndexSpaceExpression *expr_overlap = 
                forest->intersect_index_spaces(expr, it->first->set_expr);
              if (expr_overlap->is_empty())
                continue;
              to_traverse.insert(it->first, overlap);
              to_traverse_exprs[it->first] = expr_overlap;
              intersections.insert(expr_overlap, overlap);
            }
          }
          // For all our intersections, compute the remainders after the
          // overlap and if they exist then perform refinements for them
          if (!intersections.empty())
          {
            if (intersections.size() > 1)
            {
              // Sort these into field mask sets
              LegionList<FieldSet<IndexSpaceExpression*> > field_sets;
              intersections.compute_field_sets(FieldMask(), field_sets);
              for (LegionList<FieldSet<IndexSpaceExpression*> >::iterator
                    it = field_sets.begin(); it != field_sets.end(); it++)
              {
                IndexSpaceExpression *diff = forest->subtract_index_spaces(expr,
                    forest->union_index_spaces(it->elements));
                if (!diff->is_empty())
                {
                  EquivalenceSet *child = 
                    add_pending_refinement(diff, it->set_mask, NULL, source);
                  pending_to_traverse.insert(child, it->set_mask);
                  to_traverse_exprs[child] = diff;
                  // If this is a pending refinement then we'll need to
                  // wait for it before traversing farther
                  if (!refinement_done.exists())
                  {
#ifdef DEBUG_LEGION
                    assert(waiting_event.exists());
#endif
                    refinement_done = waiting_event;
                  }
                  // We need to subtract this off any unrefined remainders
                  // or add the difference of it with the original set
                  // to the set of unrefined remainders
                  filter_unrefined_remainders(it->set_mask, diff);
                  if (!!it->set_mask)
                  {
                    IndexSpaceExpression *remainder = 
                      forest->subtract_index_spaces(set_expr, diff);
                    if (!remainder->is_empty())
                    {
#ifdef DEBUG_LEGION
                      assert(disjoint_partition_refinements.get_valid_mask() * 
                              it->set_mask);
                      assert(unrefined_remainders.get_valid_mask() * 
                              it->set_mask);
#endif
                      if (unrefined_remainders.insert(remainder, it->set_mask))
                        remainder->add_nested_expression_reference(did);
                    }
                  }
                }
              }
            }
            else
            {
              // Easy case with just one intersection
              FieldMaskSet<IndexSpaceExpression>::const_iterator
                first = intersections.begin();
              IndexSpaceExpression *diff = 
                forest->subtract_index_spaces(expr, first->first);
              if (!diff->is_empty())
              {
                EquivalenceSet *child = 
                  add_pending_refinement(diff, first->second, NULL, source);
                pending_to_traverse.insert(child, first->second);
                to_traverse_exprs[child] = diff;
                // If this is a pending refinement then we'll need to
                // wait for it before traversing farther
                if (!refinement_done.exists())
                {
#ifdef DEBUG_LEGION
                  assert(waiting_event.exists());
#endif
                  refinement_done = waiting_event;
                }
                // Subtract from any unrefined remainders
                FieldMask to_filter = first->second;
                filter_unrefined_remainders(to_filter, diff);
                if (!!to_filter)
                {
                  IndexSpaceExpression *remainder = 
                    forest->subtract_index_spaces(set_expr, diff);
                  if (!remainder->is_empty())
                  {
#ifdef DEBUG_LEGION
                    assert(disjoint_partition_refinements.get_valid_mask() * 
                            to_filter);
                    assert(unrefined_remainders.get_valid_mask() * 
                            to_filter);
#endif
                    if (unrefined_remainders.insert(remainder, to_filter))
                      remainder->add_nested_expression_reference(did);
                  }
                }
              }
            }
            // These fields are all remove from the ray mask
            // since they have now been handled
            ray_mask -= intersections.get_valid_mask();
          }
        }
        // If we still have fields left, see if we need a refinement
        if (!!ray_mask && (set_expr->expr_id != expr->expr_id) &&
            (expr->get_volume() < set_expr->get_volume()))
        {
#ifdef DEBUG_LEGION
          IndexSpaceExpression *diff =
            forest->subtract_index_spaces(set_expr, expr);
          assert(!diff->is_empty());
#endif
          // We're doing a refinement for the first time, see if 
          // we can make this a disjoint partition refeinement
          if ((index_space_node != NULL) && handle.exists())
          {
            FieldMask disjoint_mask = ray_mask;
            // We can't start a new disjoint mask for anything that
            // has already been partially refined
            if (!unrefined_remainders.empty())
              disjoint_mask -= unrefined_remainders.get_valid_mask();
            if (!!disjoint_mask)
            {
              IndexSpaceNode *node = runtime->forest->get_node(handle);
              // We can start a disjoint complete partition if there
              // is exactly one partition between the parent index
              // space for the equivalence class and the child index
              // space for the subset and the partition is disjoint
              if ((node->parent != NULL) && 
                  (node->parent->parent == index_space_node) &&
                  node->parent->is_disjoint())
              {
                DisjointPartitionRefinement *dis = 
                  new DisjointPartitionRefinement(this, 
                            node->parent, done_events);
                EquivalenceSet *child = 
                  add_pending_refinement(expr, disjoint_mask, node, source);
                pending_to_traverse.insert(child, disjoint_mask);
                to_traverse_exprs[child] = expr;
                // If this is a pending refinement then we'll need to
                // wait for it before traversing farther
                if (!refinement_done.exists())
                {
#ifdef DEBUG_LEGION
                  assert(waiting_event.exists());
#endif
                  refinement_done = waiting_event;
                }
                // Save this for the future
                dis->add_child(node, child);
#ifdef DEBUG_LEGION
                assert(disjoint_mask * unrefined_remainders.get_valid_mask());
#endif
                disjoint_partition_refinements.insert(dis, disjoint_mask);
                ray_mask -= disjoint_mask;
              }
            }
          }
          // If we didn't make a disjoint partition refeinement
          // then we need to do the normal kind of refinement
          if (!!ray_mask)
          {
            // Time to refine this since we only need a subset of it
            EquivalenceSet *child = 
              add_pending_refinement(expr, ray_mask, NULL, source); 
            pending_to_traverse.insert(child, ray_mask);
            to_traverse_exprs[child] = expr;
            // If this is a pending refinement then we'll need to
            // wait for it before traversing farther
            if (!refinement_done.exists())
            {
#ifdef DEBUG_LEGION
              assert(waiting_event.exists());
#endif
              refinement_done = waiting_event;
            }
            // Subtract from any unrefined remainders
            filter_unrefined_remainders(ray_mask, expr);
            if (!!ray_mask)
            {
#ifdef DEBUG_LEGION
              assert(disjoint_partition_refinements.get_valid_mask() * 
                      ray_mask);
              assert(unrefined_remainders.get_valid_mask() * 
                      ray_mask);
#else
              IndexSpaceExpression *diff =
                forest->subtract_index_spaces(set_expr, expr);
#endif
              if (unrefined_remainders.insert(diff, ray_mask))
                diff->add_nested_expression_reference(did);
              ray_mask.clear();
            }
          }
        }
        // Otherwise we can fall through because this means the
        // expressions are equivalent
      }
      // We've done our traversal, so if we had a deferral even we can 
      // trigger it now to signal to the next user that they can start
      if (deferral_event.exists())
        Runtime::trigger_event(deferral_event);
      // Any fields which are still valid should be recorded
      if (!!ray_mask)
      {
        // Not local so we need to send a message
        if (source != runtime->address_space)
        {
          // If there's nothing to do after this we can use
          // the trace_done event directly
          if (to_traverse.empty() && pending_to_traverse.empty())
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(ray_mask);
              rez.serialize(target);
              rez.serialize(trace_done);
            }
            runtime->send_equivalence_set_ray_trace_response(source, rez);
            return;
          }
          else
          {
            RtUserEvent done = Runtime::create_rt_user_event();
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(did);
              rez.serialize(ray_mask);
              rez.serialize(target);
              rez.serialize(done);
            }
            runtime->send_equivalence_set_ray_trace_response(source, rez);
            done_events.insert(done);
          }
        }
        else
          target->record_equivalence_set(this, ray_mask);
      }
      // Traverse anything we can now before we have to wait
      if (!to_traverse.empty())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++)
        {
          RtUserEvent done = Runtime::create_rt_user_event();
          std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
            finder = to_traverse_exprs.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != to_traverse_exprs.end());
#endif
          const IndexSpace subset_handle = 
            (handle.exists() && 
              (finder->second->get_volume() == expr->get_volume())) ? handle :
                IndexSpace::NO_SPACE;
          it->first->ray_trace_equivalence_sets(target, finder->second, 
              it->second, subset_handle, source, done);
          done_events.insert(done);
        }
        // Clear these since we are done doing them
        to_traverse.clear();
      }
      // Get the actual equivalence sets for any refinements we needed to
      // wait for because they weren't ready earlier
      if (!pending_to_traverse.empty())
      {
        // If we have a refinement to do then we need to wait for that
        // to be done before we continue our traversal
        if (refinement_done.exists() && !refinement_done.has_triggered())
        {
          // Defer this until the refinements are done
          FieldMaskSet<EquivalenceSet> *copy_traverse = 
            new FieldMaskSet<EquivalenceSet>();
          copy_traverse->swap(pending_to_traverse);
          std::map<EquivalenceSet*,IndexSpaceExpression*> *copy_exprs = 
            new std::map<EquivalenceSet*,IndexSpaceExpression*>();
          copy_exprs->swap(to_traverse_exprs);
          WrapperReferenceMutator mutator(done_events);
          for (std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
                it = copy_exprs->begin(); it != copy_exprs->end(); it++)
            it->second->add_base_expression_reference(META_TASK_REF, &mutator);
          const RtUserEvent done = Runtime::create_rt_user_event();
          DeferRayTraceFinishArgs args(target, source, copy_traverse,
              copy_exprs, expr->get_volume(), handle, done);
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, refinement_done);
          done_events.insert(done);
        }
        else
        {
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                pending_to_traverse.begin(); it != 
                pending_to_traverse.end(); it++)
          {
            RtUserEvent done = Runtime::create_rt_user_event();
            std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
              finder = to_traverse_exprs.find(it->first);
#ifdef DEBUG_LEGION
            assert(finder != to_traverse_exprs.end());
#endif
            const IndexSpace subset_handle = 
              (handle.exists() && 
                (finder->second->get_volume() == expr->get_volume())) ? handle :
                  IndexSpace::NO_SPACE;
            it->first->ray_trace_equivalence_sets(target, finder->second, 
                it->second, subset_handle, source, done);
            done_events.insert(done);
          }
        }
      }
      if (!done_events.empty())
        Runtime::trigger_event(trace_done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(trace_done);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_state(Serializer &rez,
                                    const FieldMask &pack_mask) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // Pack the valid instances
      rez.serialize<size_t>(valid_instances.size());
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & pack_mask;
          if (!!overlap)
          {
            rez.serialize(it->first->did);
            rez.serialize(overlap);
          }
          else
            rez.serialize<DistributedID>(0);
        }
      }
      // Pack the reduction instances
      if (!!reduction_fields)
      {
        const FieldMask reduc_mask = reduction_fields & pack_mask;
        if (!!reduc_mask)
        {
          rez.serialize<size_t>(reduc_mask.pop_count());
          int fidx = reduc_mask.find_first_set();
          while (fidx >= 0)
          {
            rez.serialize<unsigned>(fidx);
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
#endif
            rez.serialize<size_t>(finder->second.size());
            for (std::vector<ReductionView*>::const_iterator it = 
                  finder->second.begin(); it != finder->second.end(); it++)
              rez.serialize((*it)->did);
            fidx = reduc_mask.find_next_set(fidx+1);
          }
        }
        else
          rez.serialize<size_t>(0);
      }
      else
        rez.serialize<size_t>(0);
      // Pack the restricted instances
      if (!!restricted_fields)
      {
        const FieldMask restr_mask = restricted_fields & pack_mask;
        if (!!restr_mask)
        {
          rez.serialize<size_t>(restricted_instances.size());
          for (FieldMaskSet<InstanceView>::const_iterator it = 
                restricted_instances.begin(); it != 
                restricted_instances.end(); it++)
          {
            const FieldMask overlap = pack_mask & it->second;
            if (!!overlap)
            {
              rez.serialize(it->first->did);
              rez.serialize(overlap);
            }
            else
              rez.serialize<DistributedID>(0);
          }
        }
        else
          rez.serialize<size_t>(0);
      }
      else
        rez.serialize<size_t>(0);
      // Pack the version numbers
      rez.serialize<size_t>(version_numbers.size());
      if (!version_numbers.empty())
      {
        for (LegionMap<VersionID,FieldMask>::const_iterator it = 
              version_numbers.begin(); it != version_numbers.end(); it++)
        {
          const FieldMask overlap = pack_mask & it->second;
          if (!!overlap)
          {
            rez.serialize(it->first);
            rez.serialize(overlap);
          }
          else
            rez.serialize<VersionID>(0);
        }
      }
      // Pack the update guards
      if (!update_guards.empty() &&
          !(pack_mask * update_guards.get_valid_mask()))
      {
        FieldMaskSet<CopyFillGuard> remote_guards;
        for (FieldMaskSet<CopyFillGuard>::const_iterator it = 
              update_guards.begin(); it != update_guards.end(); it++)
        {
          const FieldMask overlap = pack_mask & it->second;
          if (!overlap)
            continue;
          remote_guards.insert(it->first, overlap);
        }
        rez.serialize<size_t>(remote_guards.size());
        for (FieldMaskSet<CopyFillGuard>::const_iterator it = 
              remote_guards.begin(); it != remote_guards.end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
      }
      else
        rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_state(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RtUserEvent done_event;
      derez.deserialize(done_event);
      bool initial_refinement;
      derez.deserialize(initial_refinement);
      // Do a quick test to see if we're still the owner, if not
      // then we can just forward this on immediately
      {
        AutoLock eq(eq_lock,1,false/*exlcusive*/);
        // Check to see if we're the initial refinement, if not
        // then we need to keep forwarding this on to wherever the
        // owner is, otherwise we can handle it now and make ourselves
        // the owner once we are ready
        if (!is_logical_owner() && !initial_refinement)
        {
          Serializer rez;
          // No RezCheck because of forwarding
          rez.serialize(did);
          rez.serialize(done_event);
          rez.serialize<bool>(false); // initial refinement
          // Just move the bytes over to the serializer and return
          const size_t bytes = derez.get_remaining_bytes();
          rez.serialize(derez.get_current_pointer(), bytes);
          runtime->send_equivalence_set_remote_refinement(
                                  logical_owner_space, rez);
          // Keep the deserializer happy
          derez.advance_pointer(bytes);
          return;
        }
      }
      // Keep track of ready events
      std::set<RtEvent> ready_events;
      // Unpack into local data structures which we'll update later
      FieldMaskSet<LogicalView> new_valid;
      size_t num_valid_insts;
      derez.deserialize(num_valid_insts);
      for (unsigned idx = 0; idx < num_valid_insts; idx++)
      {
        DistributedID valid_did;
        derez.deserialize(valid_did);
        if (valid_did == 0)
          continue;
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(valid_did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        new_valid.insert(view, mask);
      }
      size_t num_reduc_fields;
      derez.deserialize(num_reduc_fields);
      std::map<unsigned,std::vector<ReductionView*> > new_reductions;
      for (unsigned idx1 = 0; idx1 < num_reduc_fields; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        std::vector<ReductionView*> &new_views = new_reductions[fidx];
        size_t num_reduc_insts;
        derez.deserialize(num_reduc_insts);
        new_views.resize(num_reduc_insts);
        for (unsigned idx2 = 0; idx2 < num_reduc_insts; idx2++)
        {
          DistributedID reduc_did;
          derez.deserialize(reduc_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(reduc_did, ready);
          new_views[idx2] = static_cast<ReductionView*>(view);
          if (ready.exists())
            ready_events.insert(ready);
        }
      }
      size_t num_restrict_insts;
      derez.deserialize(num_restrict_insts);
      FieldMaskSet<InstanceView> new_restrictions;
      if (num_restrict_insts > 0)
      {
        for (unsigned idx = 0; idx < num_restrict_insts; idx++)
        {
          DistributedID valid_did;
          derez.deserialize(valid_did);
          if (valid_did == 0)
            continue;
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(valid_did, ready);
          if (ready.exists())
            ready_events.insert(ready);
          InstanceView *inst_view = static_cast<InstanceView*>(view);
          FieldMask mask;
          derez.deserialize(mask);
          new_restrictions.insert(inst_view, mask);
        }
      }
      size_t num_versions;
      derez.deserialize(num_versions);
      LegionMap<VersionID,FieldMask> new_versions;
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        if (vid == 0)
          continue;
        derez.deserialize(new_versions[vid]);
      }
      size_t num_guards;
      derez.deserialize(num_guards);
      if (num_guards > 0)
      {
        // Need to hold the lock here to prevent copy fill guard
        // deletions from removing this before we've registered it
        AutoLock eq(eq_lock);
        for (unsigned idx = 0; idx < num_guards; idx++)
        {
          CopyFillGuard *guard = 
            CopyFillGuard::unpack_guard(derez, runtime, this);
          FieldMask guard_mask;
          derez.deserialize(guard_mask);
          if (guard != NULL)
            update_guards.insert(guard, guard_mask);
        }
      }
      // If we have events to wait for then we need to defer this
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
        {
          // Defer the merge or forward until the views are ready
          FieldMaskSet<LogicalView> *view_copy = 
            new FieldMaskSet<LogicalView>();
          view_copy->swap(new_valid);
          std::map<unsigned,std::vector<ReductionView*> > *reduc_copy = 
            new std::map<unsigned,std::vector<ReductionView*> >();
          reduc_copy->swap(new_reductions);
          FieldMaskSet<InstanceView> *restrict_copy = 
            new FieldMaskSet<InstanceView>();
          restrict_copy->swap(new_restrictions);
          LegionMap<VersionID,FieldMask> *version_copy = 
            new LegionMap<VersionID,FieldMask>();
          version_copy->swap(new_versions);
          DeferMergeOrForwardArgs args(this, initial_refinement, view_copy, 
              reduc_copy, restrict_copy, version_copy, done_event);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, wait_on);
          return;
        }
        // Otherwise fall through to do the merge or forward now
      }
      // Either merge or forward the update
      merge_or_forward(done_event, initial_refinement, new_valid, 
                       new_reductions, new_restrictions, new_versions);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::merge_or_forward(const RtUserEvent done_event,
          bool initial_refinement, const FieldMaskSet<LogicalView> &new_views,
          const std::map<unsigned,std::vector<ReductionView*> > &new_reductions,
          const FieldMaskSet<InstanceView> &new_restrictions,
          const LegionMap<VersionID,FieldMask> &new_versions)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock);
      if (is_logical_owner() || initial_refinement)
      {
        // We're the owner so we can do the merge
        LocalReferenceMutator mutator;
        for (FieldMaskSet<LogicalView>::const_iterator it =
              new_views.begin(); it != new_views.end(); it++)
          if (valid_instances.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did, &mutator);
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = new_reductions.begin(); rit != new_reductions.end(); rit++)
        {
          reduction_fields.set_bit(rit->first);
          std::vector<ReductionView*> &reduc_insts = 
            reduction_instances[rit->first];
#ifdef DEBUG_LEGION
          assert(reduc_insts.empty());
#endif
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            reduc_insts.push_back(*it);
            (*it)->add_nested_valid_ref(did, &mutator);
          }
        }
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              new_restrictions.begin(); it != new_restrictions.end(); it++)
        {
          restricted_fields |= it->second;
          if (restricted_instances.insert(it->first, it->second))
            it->first->add_nested_valid_ref(did, &mutator);
        }
        for (LegionMap<VersionID,FieldMask>::const_iterator it =
              new_versions.begin(); it != new_versions.end(); it++)
        {
          LegionMap<VersionID,FieldMask>::iterator finder = 
            version_numbers.find(it->first);
          if (finder == version_numbers.end())
            version_numbers.insert(*it);
          else
            finder->second |= it->second;
        }
        Runtime::trigger_event(done_event, mutator.get_done_event());
        // See if we need to make this the owner now
        if (!is_logical_owner())
        {
          logical_owner_space = local_space;
          eq_state = MAPPING_STATE;
        }
      }
      else
      {
        // We're not the owner so we need to forward this on
        Serializer rez;
        // No RezCheck in case of forwarding
        rez.serialize(did);
        rez.serialize(done_event);
        rez.serialize<bool>(false); // initial refinement
        rez.serialize(new_views.size());
        for (FieldMaskSet<LogicalView>::const_iterator it =
              new_views.begin(); it != new_views.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        rez.serialize<size_t>(new_reductions.size());
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = new_reductions.begin(); rit != new_reductions.end(); rit++)
        {
          rez.serialize(rit->first);
          rez.serialize(rit->second.size());
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
            rez.serialize((*it)->did);
        }
        rez.serialize(new_restrictions.size());
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              new_restrictions.begin(); it != new_restrictions.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
        }
        rez.serialize(new_versions.size());
        for (LegionMap<VersionID,FieldMask>::const_iterator it =
              new_versions.begin(); it != new_versions.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        runtime->send_equivalence_set_remote_refinement(
                                logical_owner_space, rez);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::pack_migration(Serializer &rez, RtEvent done_migration)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pending_refinements.empty());
#endif
      std::map<LogicalView*,unsigned> *late_references = NULL;
      // Pack the valid instances
      rez.serialize<size_t>(valid_instances.size());
      if (!valid_instances.empty())
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (late_references == NULL)
            late_references = new std::map<LogicalView*,unsigned>();
          (*late_references)[it->first] = 1;
        }
        valid_instances.clear();
      }
      // Pack the reduction instances
      rez.serialize<size_t>(reduction_instances.size());
      if (!reduction_instances.empty())
      {
        for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              rit = reduction_instances.begin(); 
              rit != reduction_instances.end(); rit++)
        {
          rez.serialize(rit->first);
          rez.serialize<size_t>(rit->second.size());
          for (std::vector<ReductionView*>::const_iterator it = 
                rit->second.begin(); it != rit->second.end(); it++)
          {
            rez.serialize((*it)->did);
            if (late_references == NULL)
              late_references = new std::map<LogicalView*,unsigned>();
            (*late_references)[*it] = 1;
          }
        }
        reduction_instances.clear();
        reduction_fields.clear();
      }
      // Pack the restricted instances
      rez.serialize<size_t>(restricted_instances.size());  
      if (!restricted_instances.empty())
      {
        rez.serialize(restricted_fields);
        for (FieldMaskSet<InstanceView>::const_iterator it = 
              restricted_instances.begin(); it != 
              restricted_instances.end(); it++)
        {
          rez.serialize(it->first->did);
          rez.serialize(it->second);
          if (late_references == NULL)
            late_references = new std::map<LogicalView*,unsigned>();
          std::map<LogicalView*,unsigned>::iterator finder = 
            late_references->find(it->first);
          if (finder == late_references->end())
            (*late_references)[it->first] = 1;
          else
            finder->second += 1;
        }
        restricted_instances.clear();
        restricted_fields.clear();
      }
      // Pack the version numbers
      rez.serialize<size_t>(version_numbers.size());
      if (!version_numbers.empty())
      {
        for (LegionMap<VersionID,FieldMask>::const_iterator it = 
              version_numbers.begin(); it != version_numbers.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        version_numbers.clear();
      }
      // Pack the update guards
      rez.serialize<size_t>(update_guards.size());
      if (!update_guards.empty())
      {
        for (FieldMaskSet<CopyFillGuard>::const_iterator it = 
              update_guards.begin(); it != update_guards.end(); it++)
        {
          it->first->pack_guard(rez);
          rez.serialize(it->second);
        }
        update_guards.clear();
      }
      // Pack subsets
      // We're only allowed to keep the complete subsets on this node
      // so we need to filter anything that isn't fully refined
      // We still keep references to the equivalence sets though
      // so that they aren't deleted
      FieldMask incomplete_refinements;
      if (!unrefined_remainders.empty())
        incomplete_refinements = unrefined_remainders.get_valid_mask();
      if (!disjoint_partition_refinements.empty())
        incomplete_refinements |= 
          disjoint_partition_refinements.get_valid_mask();
      rez.serialize<size_t>(subsets.size());
      for (FieldMaskSet<EquivalenceSet>::iterator it = 
            subsets.begin(); it != subsets.end(); it++)
      {
        rez.serialize(it->first->did);
        rez.serialize(it->second);
        if (!!incomplete_refinements)
          it.filter(incomplete_refinements);
      }
      // Tighten the valid mask for future analyses
      if (!!incomplete_refinements)
        subsets.tighten_valid_mask();
      // No need to clear subsets since we can still maintain a copy of it
      // Pack remote subsets
      rez.serialize<size_t>(remote_subsets.size());
      if (!remote_subsets.empty())
      {
        for (std::set<AddressSpaceID>::const_iterator it = 
              remote_subsets.begin(); it != remote_subsets.end(); it++)
          rez.serialize(*it);
        remote_subsets.clear();
      }
      // Pack unrefined remainders
      rez.serialize<size_t>(unrefined_remainders.size());
      if (!unrefined_remainders.empty())
      {
        std::vector<IndexSpaceExpression*> *references = 
          new std::vector<IndexSpaceExpression*>();
        references->reserve(unrefined_remainders.size());
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator it = 
              unrefined_remainders.begin(); it != 
              unrefined_remainders.end(); it++)
        {
          it->first->pack_expression(rez, logical_owner_space);
          rez.serialize(it->second);
          references->push_back(it->first);
        }
        unrefined_remainders.clear();
        // Defer removing the references on these expressions until
        // the migration has been done
        DeferRemoveRefArgs args(references, did);
        runtime->issue_runtime_meta_task(args, 
            LG_THROUGHPUT_WORK_PRIORITY, done_migration);
      }
      // Pack disjoint partition refinements
      rez.serialize<size_t>(disjoint_partition_refinements.size());
      if (!disjoint_partition_refinements.empty())
      {
        for (FieldMaskSet<DisjointPartitionRefinement>::const_iterator it =
              disjoint_partition_refinements.begin(); it !=
              disjoint_partition_refinements.end(); it++)
        {
          rez.serialize(it->first->partition->handle);
          const std::map<IndexSpaceNode*,EquivalenceSet*> &children = 
            it->first->get_children();
          rez.serialize<size_t>(children.size());
          for (std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator 
                cit = children.begin(); cit != children.end(); cit++)
          {
            rez.serialize(cit->first->handle);
            rez.serialize(cit->second->did);
          }
          rez.serialize(it->second);
          delete it->first;
        }
        disjoint_partition_refinements.clear();
      }
      // Pack the user samples and counts
      rez.serialize(migration_index);
      for (unsigned idx = 0; idx < MIGRATION_EPOCHS; idx++)
      {
        std::vector<std::pair<AddressSpaceID,unsigned> > &samples = 
          user_samples[idx];
        rez.serialize<size_t>(samples.size());
        for (std::vector<std::pair<AddressSpaceID,unsigned> >::const_iterator
              it = samples.begin(); it != samples.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
        samples.clear();
      }
      if (late_references != NULL)
      {
        // Launch a task to remove the references once the migration is done
        RemoteRefTaskArgs args(this->did, RtUserEvent::NO_RT_USER_EVENT,
                               false/*add*/, late_references);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_WORK_PRIORITY, 
                                         done_migration); 
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::unpack_migration(Deserializer &derez, 
                                  AddressSpaceID source, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // All the preconditions before we can make this the owner
      std::set<RtEvent> owner_preconditions;
      AutoLock eq(eq_lock); 
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
      assert(valid_instances.empty());
      assert(reduction_instances.empty());
      assert(restricted_instances.empty());
      assert(version_numbers.empty());
      assert(update_guards.empty());
      assert(pending_refinements.empty());
      assert(remote_subsets.empty());
      assert(unrefined_remainders.empty());
      assert(disjoint_partition_refinements.empty());
#endif
      size_t num_valid_insts;
      derez.deserialize(num_valid_insts);
      for (unsigned idx = 0; idx < num_valid_insts; idx++)
      {
        DistributedID valid_did;
        derez.deserialize(valid_did);
        RtEvent ready;
        LogicalView *view = 
          runtime->find_or_request_logical_view(valid_did, ready);
        FieldMask mask;
        derez.deserialize(mask);
        valid_instances.insert(view, mask);
        if (ready.exists())
          owner_preconditions.insert(ready);
      }
      size_t num_reduc_fields;
      derez.deserialize(num_reduc_fields);
      for (unsigned idx1 = 0; idx1 < num_reduc_fields; idx1++)
      {
        unsigned fidx;
        derez.deserialize(fidx);
        reduction_fields.set_bit(fidx);
        size_t num_reduc_insts;
        derez.deserialize(num_reduc_insts);
        std::vector<ReductionView*> &reduc_views = reduction_instances[fidx];
        for (unsigned idx2 = 0; idx2 < num_reduc_insts; idx2++)
        {
          DistributedID reduc_did;
          derez.deserialize(reduc_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(reduc_did, ready);
          ReductionView *reduc_view = static_cast<ReductionView*>(view);
          reduc_views.push_back(reduc_view);
          if (ready.exists())
            owner_preconditions.insert(ready);
        }
      }
      size_t num_restrict_insts;
      derez.deserialize(num_restrict_insts);
      if (num_restrict_insts > 0)
      {
        derez.deserialize(restricted_fields);
        for (unsigned idx = 0; idx < num_restrict_insts; idx++)
        {
          DistributedID valid_did;
          derez.deserialize(valid_did);
          RtEvent ready;
          LogicalView *view = 
            runtime->find_or_request_logical_view(valid_did, ready);
          InstanceView *inst_view = static_cast<InstanceView*>(view);
          FieldMask mask;
          derez.deserialize(mask);
          restricted_instances.insert(inst_view, mask);
          if (ready.exists())
            owner_preconditions.insert(ready);
        }
      }
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx = 0; idx < num_versions; idx++)
      {
        VersionID vid;
        derez.deserialize(vid);
        derez.deserialize(version_numbers[vid]);
      }
      size_t num_guards;
      derez.deserialize(num_guards);
      for (unsigned idx = 0; idx < num_guards; idx++)
      {
        CopyFillGuard *guard = CopyFillGuard::unpack_guard(derez, runtime,this);
        FieldMask guard_mask;
        derez.deserialize(guard_mask);
        if (guard != NULL)
          update_guards.insert(guard, guard_mask);
      }
      FieldMaskSet<EquivalenceSet> new_subsets;
      size_t num_subsets;
      derez.deserialize(num_subsets);
      for (unsigned idx = 0; idx < num_subsets; idx++)
      {
        DistributedID subset_did;
        derez.deserialize(subset_did);
        RtEvent ready;
        EquivalenceSet *subset = 
          runtime->find_or_request_equivalence_set(subset_did, ready);
        if (ready.exists())
          owner_preconditions.insert(ready);
        FieldMask subset_mask;
        derez.deserialize(subset_mask);
        new_subsets.insert(subset, subset_mask);
      }
      size_t num_remote_subsets;
      derez.deserialize(num_remote_subsets);
      for (unsigned idx = 0; idx < num_remote_subsets; idx++)
      {
        AddressSpaceID remote;
        derez.deserialize(remote);
        remote_subsets.insert(remote);
      }
      size_t num_unrefined_remainders;
      derez.deserialize(num_unrefined_remainders);
      for (unsigned idx = 0; idx < num_unrefined_remainders; idx++)
      {
        IndexSpaceExpression *expr = 
          IndexSpaceExpression::unpack_expression(derez,runtime->forest,source);
        FieldMask mask;
        derez.deserialize(mask);
        if (unrefined_remainders.insert(expr, mask))
          expr->add_nested_expression_reference(did);
      }
      size_t num_disjoint_refinements;
      derez.deserialize(num_disjoint_refinements);
      for (unsigned idx1 = 0; idx1 < num_disjoint_refinements; idx1++)
      {
        IndexPartition handle;
        derez.deserialize(handle);
        IndexPartNode *part = runtime->forest->get_node(handle);
        DisjointPartitionRefinement *dis = 
          new DisjointPartitionRefinement(this, part, owner_preconditions);
        size_t num_children;
        derez.deserialize(num_children);
        for (unsigned idx2 = 0; idx2 < num_children; idx2++)
        {
          IndexSpace child;
          derez.deserialize(child);
          IndexSpaceNode *node = runtime->forest->get_node(child);
          DistributedID child_did;
          derez.deserialize(child_did);
          RtEvent ready;
          dis->add_child(node,
            runtime->find_or_request_equivalence_set(child_did, ready));
          if (ready.exists())
            owner_preconditions.insert(ready);
        }
        FieldMask mask;
        derez.deserialize(mask);
        disjoint_partition_refinements.insert(dis, mask);
      }
      derez.deserialize(migration_index);
      for (unsigned idx1 = 0; idx1 < MIGRATION_EPOCHS; idx1++)
      {
        size_t num_samples;
        derez.deserialize(num_samples);
        if (num_samples > 0)
        {
          std::vector<std::pair<AddressSpaceID,unsigned> > &samples =
            user_samples[idx1];
          samples.resize(num_samples);
          for (unsigned idx2 = 0; idx2 < num_samples; idx2++)
          {
            derez.deserialize(samples[idx2].first);
            derez.deserialize(samples[idx2].second);
          }
        }
      }
      // If there are any pending anayses we need to wait for them to finish
      if (pending_analyses > 0)
      {
#ifdef DEBUG_LEGION
        assert(!waiting_event.exists());
#endif
        waiting_event = Runtime::create_rt_user_event();
        owner_preconditions.insert(waiting_event);
      }
      if (!owner_preconditions.empty())
      {
        const RtEvent pre = Runtime::merge_events(owner_preconditions);
        if (pre.exists() && !pre.has_triggered())
        {
          // We need to defer this until later
          FieldMaskSet<EquivalenceSet> *owner_subsets = 
            new FieldMaskSet<EquivalenceSet>();
          owner_subsets->swap(new_subsets);
          // Defer the call to make this the owner until the event triggers
          DeferMakeOwnerArgs args(this, owner_subsets, done_event);
          runtime->issue_runtime_meta_task(args, 
              LG_LATENCY_DEFERRED_PRIORITY, pre);
          return;
        }
      }
      // If we fall through then we get to do the add now
      make_owner(&new_subsets, done_event, false/*need lock*/);
    }

    //--------------------------------------------------------------------------
    bool EquivalenceSet::make_owner(FieldMaskSet<EquivalenceSet> *new_subsets,
                                    RtUserEvent done_event, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock eq(eq_lock);
        return make_owner(new_subsets, done_event, false/*need lock*/);
      }
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
#endif
      // See if we need to defer this because there are outstanding analyses
      if (pending_analyses > 0)
      {
#ifdef DEBUG_LEGION
        assert(!waiting_event.exists());
#endif
        waiting_event = Runtime::create_rt_user_event();
        DeferMakeOwnerArgs args(this, new_subsets, done_event);
        runtime->issue_runtime_meta_task(args, 
            LG_LATENCY_DEFERRED_PRIORITY, waiting_event);
        return false;
      }
      // Now we can mark that we are the logical owner
      logical_owner_space = local_space;
      // If we were waiting for a valid copy of the subsets we now have it
      if (eq_state == PENDING_VALID_STATE)
      {
#ifdef DEBUG_LEGION
        assert(transition_event.exists()); 
#endif
        // We can trigger this transition event now that we have a valid
        // copy of the subsets (we are the logical owner)
        Runtime::trigger_event(transition_event);
        transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
      eq_state = MAPPING_STATE;
      LocalReferenceMutator mutator;
      // Add references to all the views that we've loaded
      for (FieldMaskSet<LogicalView>::const_iterator it =
            valid_instances.begin(); it != valid_instances.end(); it++)
        it->first->add_nested_valid_ref(did, &mutator);
      for (std::map<unsigned,std::vector<ReductionView*> >::const_iterator it1 =
           reduction_instances.begin(); it1 != reduction_instances.end(); it1++)
        for (std::vector<ReductionView*>::const_iterator it2 =
              it1->second.begin(); it2 != it1->second.end(); it2++)
          (*it2)->add_nested_valid_ref(did, &mutator);
      for (FieldMaskSet<InstanceView>::const_iterator it =
           restricted_instances.begin(); it != restricted_instances.end(); it++)
        it->first->add_nested_valid_ref(did, &mutator);
      // Update the subsets now that we are officially the owner
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            new_subsets->begin(); it != new_subsets->end(); it++)
        if (subsets.insert(it->first, it->second))
          it->first->add_nested_resource_ref(did);
      Runtime::trigger_event(done_event, mutator.get_done_event());
      return true;
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
    FieldMask EquivalenceSet::is_restricted(InstanceView *view)
    //--------------------------------------------------------------------------
    {
      AutoLock eq(eq_lock,1,false/*exclusive*/);
      FieldMask mask;

      FieldMaskSet<InstanceView>::const_iterator finder =
        restricted_instances.find(view);
      if (finder != restricted_instances.end())
        mask = finder->second;
      return mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::initialize_set(const RegionUsage &usage,
                                        const FieldMask &user_mask,
                                        const bool restricted,
                                        const InstanceSet &sources,
                                const std::vector<InstanceView*> &corresponding,
                                        std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(sources.size() == corresponding.size());
#endif
      WrapperReferenceMutator mutator(applied_events);
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
            valid_instances.find(view);
          if (finder == valid_instances.end())
          {
            valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            finder.merge(view_mask);
          // Always restrict reduction-only users since we know the data
          // is going to need to be flushed anyway
          FieldMaskSet<InstanceView>::iterator restricted_finder = 
            restricted_instances.find(view);
          if (restricted_finder == restricted_instances.end())
          {
            restricted_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            restricted_finder.merge(view_mask);
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
            valid_instances.find(view);
          if (finder == valid_instances.end())
          {
            valid_instances.insert(view, view_mask);
            view->add_nested_valid_ref(did, &mutator);
          }
          else
            finder.merge(view_mask);
          // If this is restricted then record it
          if (restricted)
          {
            FieldMaskSet<InstanceView>::iterator restricted_finder = 
              restricted_instances.find(view);
            if (restricted_finder == restricted_instances.end())
            {
              restricted_instances.insert(view, view_mask);
              view->add_nested_valid_ref(did, &mutator);
            }
            else
              restricted_finder.merge(view_mask);
          }
        }
      }
      // Update any restricted fields 
      if (restricted)
        restricted_fields |= user_mask;
      // Set the version numbers too
      version_numbers[init_version] |= user_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_valid_instances(ValidInstAnalysis &analysis,
                                              FieldMask user_mask,
                                             std::set<RtEvent> &deferral_events,
                                              std::set<RtEvent> &applied_events,
                                              const bool cached_set,
                                              const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events); 
        if (subsets.empty())
        {
          analysis.record_remote(this, user_mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = user_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            user_mask -= non_subset;
            if (!user_mask)
              return;
          }
        }
        // Otherwise we fall through and record our subsets
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(user_mask))
      {
        check_for_unrefined_remainder(eq, user_mask, 
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Update the user mask and the stale_mask if there is one
        user_mask -= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->find_valid_instances(analysis, it->second, deferral_events,
                                          applied_events, false/*original*/);
        eq.reacquire();
        // Return if our user mask is empty
        if (!user_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      if (analysis.redop != 0)
      {
        // Iterate over all the fields
        int fidx = user_mask.find_first_set();
        while (fidx >= 0)
        {
          std::map<unsigned,std::vector<ReductionView*> >::const_iterator
            current = reduction_instances.find(fidx);
          if (current != reduction_instances.end())
          {
            FieldMask local_mask;
            local_mask.set_bit(fidx);
            for (std::vector<ReductionView*>::const_reverse_iterator it = 
                  current->second.rbegin(); it != current->second.rend(); it++)
            {
              PhysicalManager *manager = (*it)->get_manager();
              if (manager->redop != analysis.redop)
                break;
              analysis.record_instance(*it, local_mask);
            }
          }
          fidx = user_mask.find_next_set(fidx+1);
        }
      }
      else
      {
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          if (!it->first->is_instance_view())
            continue;
          const FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          analysis.record_instance(it->first->as_instance_view(), overlap);
        }
      }
      if (has_restrictions(user_mask))
        analysis.record_restriction();
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::find_invalid_instances(InvalidInstAnalysis &analysis,
                                                FieldMask user_mask,
                                             std::set<RtEvent> &deferral_events,
                                              std::set<RtEvent> &applied_events,
                                                const bool cached_set,
                                                const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events); 
        if (subsets.empty())
        {
          analysis.record_remote(this, user_mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = user_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            user_mask -= non_subset;
            if (!user_mask)
              return;
          }
        }
        // Otherwise we fall through and record our subsets
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(user_mask))
      {
        check_for_unrefined_remainder(eq, user_mask, 
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Update the user mask and the stale_mask if there is one
        user_mask -= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->find_invalid_instances(analysis, it->second, 
              deferral_events, applied_events, false/*original*/);
        eq.reacquire();
        // Return if our user mask is empty
        if (!user_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      // Lock the analysis so we can perform updates here
      AutoLock a_lock(analysis);
      // See if our instances are valid for any fields we're traversing
      // and if not record them
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            analysis.valid_instances.begin(); it != 
            analysis.valid_instances.end(); it++)
      {
        FieldMask invalid_mask = it->second & user_mask;
        if (!invalid_mask)
          continue;
        FieldMaskSet<LogicalView>::const_iterator finder = 
          valid_instances.find(it->first);
        if (finder != valid_instances.end())
        {
          invalid_mask -= finder->second;
          if (!!invalid_mask)
            analysis.record_instance(it->first, invalid_mask);
        }
        else // Not valid for any of them so record it
          analysis.record_instance(it->first, invalid_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::defer_traversal(AutoTryLock &eq,
                                         PhysicalAnalysis &analysis,
                                         const FieldMask &mask,
                                         std::set<RtEvent> &deferral_events,
                                         std::set<RtEvent> &applied_events,
                                         const bool already_deferred,
                                         const bool cached_set)
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
                                 applied_events, cached_set, deferral_event);
      }
      else
        analysis.defer_traversal(eq.try_next(), this, mask, deferral_events, 
                                 applied_events, cached_set);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set(UpdateAnalysis &analysis,
                                    FieldMask user_mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    FieldMask *remove_mask, // can be NULL
                                    const bool cached_set/*=true*/,
                                    const bool already_deferred/*=false*/)
    //--------------------------------------------------------------------------
    {
      // Try to get the lock, if we don't defer the traversal
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, user_mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!cached_set && analysis.update_alt_sets(this, user_mask))
        return;
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events); 
        if (subsets.empty())
        {
          analysis.record_remote(this, user_mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = user_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            user_mask -= non_subset;
            if (!user_mask)
              return;
          }
        }
        // Otherwise we fall through and record our subsets
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(user_mask))
      {
        check_for_unrefined_remainder(eq, user_mask, 
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & user_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!cached_set)
          analysis.filter_alt_sets(this, to_traverse.get_valid_mask());
        // Update the user mask and the remove_mask if there is one
        user_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->update_set(analysis, it->second, deferral_events,
              applied_events, NULL/*remove mask*/, false/*original set*/);
        eq.reacquire();
        // Return if our user mask is empty
        if (!user_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      WrapperReferenceMutator mutator(applied_events);
      // Now that we're ready to perform the analysis 
      // we need to lock the analysis 
      AutoLock a_lock(analysis);
      // Check for any uninitialized data
      // Don't report uninitialized warnings for empty equivalence classes
      if (analysis.check_initialized && !set_expr->is_empty())
      {
        const FieldMask uninit = user_mask - valid_instances.get_valid_mask();
        if (!!uninit)
          analysis.record_uninitialized(uninit, applied_events);
      }
      if (analysis.output_aggregator != NULL)
        analysis.output_aggregator->clear_update_fields();
      if (IS_REDUCE(analysis.usage))
      {
        // Reduction-only
        // We only record reductions if the set expression is not empty
        // as we can't guarantee the reductions will ever be read for 
        // empty equivalence sets which can lead to leaked instances
        if (!set_expr->is_empty())
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
            // Track the case where this reduction view is also
            // stored in the valid instances in which case we 
            // do not need to do any fills. This will only happen
            // if these fields are restricted.
            bool already_valid = false;
            if (!!restricted_fields && !(update_fields * restricted_fields) &&
                (valid_instances.find(red_view) != valid_instances.end()))
              already_valid = true;
            int fidx = update_fields.find_first_set();
            // Figure out which fields require a fill operation
            // in order initialize the reduction instances
            FieldMask fill_mask;
            while (fidx >= 0)
            {
              std::vector<ReductionView*> &field_views = 
                reduction_instances[fidx];
              // Scan through the reduction instances to see if we're
              // already in the list of valid reductions, if not then
              // we're going to need a fill to initialize the instance
              // Also check for the ABA problem on reduction instances
              // described in Legion issue #545 where we start out
              // with reductions of kind A, switch to reductions of
              // kind B, and then switch back to reductions of kind A
              // which will make it unsafe to re-use the instance
              bool found = already_valid && restricted_fields.is_set(fidx);
              for (std::vector<ReductionView*>::const_iterator it =
                    field_views.begin(); it != field_views.end(); it++)
              {
                if ((*it) != red_view)
                {
                  if (found && ((*it)->get_redop() != view_redop))
                    REPORT_LEGION_FATAL(LEGION_FATAL_REDUCTION_ABA_PROBLEM,
                        "Unsafe re-use of reduction instance detected due "
                        "to alternating un-flushed reduction operations "
                        "%d and %d. Please report this use case to the "
                        "Legion developer's mailing list so that we can "
                        "help you address it.", view_redop, (*it)->get_redop())
                }
                else 
                  // we've seen it now so we don't need a fill
                  // keep going to check for ABA problem
                  found = true;
              }
              if (!found)
              {
                red_view->add_nested_valid_ref(did, &mutator); 
                field_views.push_back(red_view);
                // We need a fill for this field to initialize it
                fill_mask.set_bit(fidx); 
              }
              else
                guard_fill_mask.set_bit(fidx);
              fidx = update_fields.find_next_set(fidx+1);
            }
            if (!!fill_mask)
            {
              if (fill_aggregator == NULL)
              {
                fill_aggregator = new CopyFillAggregator(runtime->forest,
                    analysis.op, analysis.index, analysis.index,
                    RtEvent::NO_RT_EVENT, false/*track events*/);
                analysis.input_aggregators[RtEvent::NO_RT_EVENT] = 
                  fill_aggregator;
              }
              // Record the fill operation on the aggregator
              fill_aggregator->record_fill(red_view, red_view->fill_view,
                                           fill_mask, set_expr);
              // Record this as a guard for later operations
              update_guards.insert(fill_aggregator, fill_mask);
#ifdef DEBUG_LEGION
              if (!fill_aggregator->record_guard_set(this))
                assert(false);
#else
              fill_aggregator->record_guard_set(this);
#endif
            }
          }
          // If we have any fills that were issued by a prior operation
          // that we need to reuse then check for them here. This is a
          // slight over-approximation for the mapping dependences because
          // we really only need to wait for fills to instances that we
          // care about it, but it should be minimal overhead and the
          // resulting event graph will still be precise
          if (!update_guards.empty() && !!guard_fill_mask &&
              !(update_guards.get_valid_mask() * guard_fill_mask))
          {
            for (FieldMaskSet<CopyFillGuard>::iterator it = 
                  update_guards.begin(); it != update_guards.end(); it++)
            {
              if (it->first == fill_aggregator)
                continue;
              const FieldMask guard_mask = guard_fill_mask & it->second;
              if (!guard_mask)
                continue;
              // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
              const RtEvent guard_event = it->first->effects_applied;
#else
              const RtEvent guard_event = 
                (analysis.original_source == local_space) ?
                it->first->guard_postcondition :
                it->first->effects_applied;
#endif
              analysis.guard_events.insert(guard_event);
            }
          }
          // Flush any restricted fields
          if (!!restricted_fields)
          {
            const FieldMask reduce_mask = user_mask & restricted_fields;
            if (!!reduce_mask)
              apply_reductions(reduce_mask, analysis.output_aggregator,
                  RtEvent::NO_RT_EVENT, analysis.op, 
                  analysis.index, true/*track events*/); 
            // No need to record that we applied the reductions, we'll
            // discover that when we collapse the single/multi-reduce state
            reduction_fields |= (user_mask - restricted_fields);
          }
          else
            reduction_fields |= user_mask;
        }
      }
      else if (IS_WRITE(analysis.usage) && IS_DISCARD(analysis.usage))
      {
        // Write-only
        // Filter any reductions that we no longer need
        const FieldMask reduce_filter = reduction_fields & user_mask;
        if (!!reduce_filter)
          filter_reduction_instances(reduce_filter);
        // Filter any normal instances that will be overwritten
        const FieldMask non_restricted = user_mask - restricted_fields; 
        if (!!non_restricted)
        {
          filter_valid_instances(non_restricted);
          // Record any non-restricted instances
          record_instances(non_restricted, analysis.target_instances, 
                           analysis.target_views, mutator);
        }
        // Issue copy-out copies for any restricted fields
        if (!!restricted_fields)
        {
          const FieldMask restricted_mask = user_mask & restricted_fields;
          if (!!restricted_mask)
            copy_out(restricted_mask, analysis.target_instances,
                     analysis.target_views, analysis.op, 
                     analysis.index, analysis.output_aggregator);
        }
        // Advance our version numbers
        advance_version_numbers(user_mask);
      }
      else if (IS_READ_ONLY(analysis.usage) && !update_guards.empty() && 
                !(user_mask * update_guards.get_valid_mask()))
      {
        // If we're doing read-only mode, get the set of events that
        // we need to wait for before we can do our registration, this 
        // ensures that we serialize read-only operations correctly
        // In order to avoid deadlock we have to make different copy fill
        // aggregators for each of the different fields of prior updates
        FieldMask remainder_mask = user_mask;
        LegionVector<std::pair<CopyFillAggregator*,FieldMask> > to_add;
        for (FieldMaskSet<CopyFillGuard>::iterator it = 
              update_guards.begin(); it != update_guards.end(); it++)
        {
          const FieldMask guard_mask = remainder_mask & it->second;
          if (!guard_mask)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          const RtEvent guard_event = it->first->effects_applied;
#else
          const RtEvent guard_event = 
            (analysis.original_source == local_space) ?
            it->first->guard_postcondition :
            it->first->effects_applied;
#endif
          analysis.guard_events.insert(guard_event);
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
          update_set_internal(input_aggregator, guard_event, 
                              analysis.op, analysis.index,
                              analysis.usage, guard_mask, 
                              analysis.target_instances, 
                              analysis.target_views, applied_events,
                              analysis.record_valid);
          // If we did any updates record ourselves as the new guard here
          if ((input_aggregator != NULL) && 
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
#ifndef NON_AGGRESSIVE_AGGREGATORS
            // We also have to chain effects in this case 
            input_aggregator->record_reference_mutation_effect(
                                it->first->effects_applied);
#endif
            if (finder == analysis.input_aggregators.end())
              analysis.input_aggregators[guard_event] = input_aggregator;
            // Record this as a guard for later operations
            to_add.resize(to_add.size() + 1);
            std::pair<CopyFillAggregator*,FieldMask> &back = to_add.back();
            const FieldMask &update_mask = 
              input_aggregator->get_update_fields();
            back.first = input_aggregator;
            back.second = update_mask;
#ifdef DEBUG_LEGION
            if (!input_aggregator->record_guard_set(this))
              assert(false);
#else
            input_aggregator->record_guard_set(this);
#endif
            // Remove the current guard since it doesn't matter anymore
            it.filter(update_mask);
          }
          remainder_mask -= guard_mask;
          if (!remainder_mask)
            break;
        }
        if (!to_add.empty())
        {
          for (LegionVector<std::pair<CopyFillAggregator*,FieldMask> >::
                const_iterator it = to_add.begin(); it != to_add.end(); it++)
          {
#ifdef DEBUG_LEGION
            assert(it->second * refining_fields);
#endif
            update_guards.insert(it->first, it->second);
          }
        }
        // If we have unguarded fields we can easily do thos
        if (!!remainder_mask)
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
          update_set_internal(input_aggregator, RtEvent::NO_RT_EVENT, 
                              analysis.op, analysis.index,
                              analysis.usage, remainder_mask, 
                              analysis.target_instances, 
                              analysis.target_views, applied_events,
                              analysis.record_valid);
          // If we made the input aggregator then store it
          if ((input_aggregator != NULL) && 
              ((finder == analysis.input_aggregators.end()) ||
               input_aggregator->has_update_fields()))
          {
            analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
#ifdef DEBUG_LEGION
            assert(input_aggregator->get_update_fields() * refining_fields);
#endif
            // Record this as a guard for later operations
            update_guards.insert(input_aggregator, 
                input_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
            if (!input_aggregator->record_guard_set(this))
              assert(false);
#else
            input_aggregator->record_guard_set(this);
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
        update_set_internal(input_aggregator, RtEvent::NO_RT_EVENT, 
                            analysis.op, analysis.index, analysis.usage,
                            user_mask, analysis.target_instances, 
                            analysis.target_views, applied_events,
                            analysis.record_valid);
        if (IS_WRITE(analysis.usage))
        {
          advance_version_numbers(user_mask);
          // Issue copy-out copies for any restricted fields if we wrote stuff
          const FieldMask restricted_mask = restricted_fields & user_mask;
          if (!!restricted_mask)
            copy_out(restricted_mask, analysis.target_instances,
                     analysis.target_views, analysis.op, 
                     analysis.index, analysis.output_aggregator);
        }
        // If we made the input aggregator then store it
        if ((input_aggregator != NULL) && 
            ((finder == analysis.input_aggregators.end()) ||
             input_aggregator->has_update_fields()))
        {
          analysis.input_aggregators[RtEvent::NO_RT_EVENT] = input_aggregator;
#ifdef DEBUG_LEGION
          assert(input_aggregator->get_update_fields() * refining_fields);
#endif
          // Record this as a guard for later operations
          update_guards.insert(input_aggregator, 
              input_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
          if (!input_aggregator->record_guard_set(this))
            assert(false);
#else
          input_aggregator->record_guard_set(this);
#endif
        }
      }
      if ((analysis.output_aggregator != NULL) && 
           analysis.output_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(analysis.output_aggregator->get_update_fields() * 
                refining_fields);
#endif
        update_guards.insert(analysis.output_aggregator, 
            analysis.output_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
        if (!analysis.output_aggregator->record_guard_set(this))
          assert(false);
#else
        analysis.output_aggregator->record_guard_set(this);
#endif
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::update_set_internal(
                                 CopyFillAggregator *&input_aggregator,
                                 const RtEvent guard_event,
                                 Operation *op, const unsigned index,
                                 const RegionUsage &usage,
                                 const FieldMask &user_mask,
                                 const InstanceSet &target_instances,
                                 const std::vector<InstanceView*> &target_views,
                                 std::set<RtEvent> &applied_events,
                                 const bool record_valid)
    //--------------------------------------------------------------------------
    {
      // Read-write or read-only
      // Check for any copies from normal instances first
      issue_update_copies_and_fills(input_aggregator, guard_event, op, index,
          false/*track*/, user_mask, target_instances, target_views, set_expr);
      // Get the set of fields to filter, any for which we're about
      // to apply pending reductions or overwite, except those that
      // are restricted
      const FieldMask reduce_mask = reduction_fields & user_mask;
      const FieldMask restricted_mask = restricted_fields & user_mask;
      const bool is_write = IS_WRITE(usage);
      FieldMask filter_mask = is_write ? user_mask : reduce_mask;
      if (!!restricted_mask)
        filter_mask -= restricted_mask;
      if (!!filter_mask)
        filter_valid_instances(filter_mask);
      WrapperReferenceMutator mutator(applied_events);
      // Save the instances if they are not restricted
      // Otherwise if they are restricted then the restricted instances
      // are already listed as the valid views so there's nothing more
      // for us to have to do
      if (!!restricted_mask)
      {
        const FieldMask non_restricted = user_mask - restricted_fields;
        if (!!non_restricted)
          record_instances(non_restricted, target_instances, 
                           target_views, mutator); 
      }
      else if (record_valid)
        record_instances(user_mask, target_instances,
                         target_views, mutator);
      // Read-only instances that perform reductions still need to be
      // tracked for these fields because it is where the reductions 
      // are to be applied
      else if (!!reduce_mask)
        record_instances(reduce_mask, target_instances,
                         target_views, mutator);
      // Next check for any reductions that need to be applied
      if (!!reduce_mask)
        apply_reductions(reduce_mask, input_aggregator, guard_event,
                         op, index, false/*track events*/); 
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::check_for_migration(PhysicalAnalysis &analysis,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifndef DISABLE_EQUIVALENCE_SET_MIGRATION
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
            "Internal runtime performance warning: equivalence set %lld has "
            "%zd different users which is the same as the sampling rate of "
            "%d. Please report this application use case to the Legion "
            "developers mailing list.", did, current_samples.size(),
            SAMPLES_PER_MIGRATION_TEST)
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
      // Don't do any migrations if we have any pending refinements
      // or we have outstanding analyses that prevent it for now
      if (!pending_refinements.empty() || !!refining_fields || 
          (pending_analyses > 0))
      {
        // Reset the data structures for the next run
        sample_count = 0;
        user_samples[migration_index].clear();
        return;
      }
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
      // Add ourselves and remove the new owner from remote subsets
      remote_subsets.insert(local_space);
      remote_subsets.erase(logical_owner_space);
      // We can switch our eq_state to being remote valid
      eq_state = VALID_STATE;
      RtUserEvent done_migration = Runtime::create_rt_user_event();
      // Do the migration
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(done_migration);
        pack_migration(rez, done_migration);
      }
      runtime->send_equivalence_set_migration(logical_owner_space, rez);
      applied_events.insert(done_migration);
#endif // DISABLE_EQUIVALENCE_SET MIGRATION
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::acquire_restrictions(AcquireAnalysis &analysis,
                                          FieldMask acquire_mask,
                                          std::set<RtEvent> &deferral_events,
                                          std::set<RtEvent> &applied_events,
                                          FieldMask *remove_mask,
                                          const bool cached_set/*=true*/,
                                          const bool already_deferred/*=false*/)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, acquire_mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!cached_set && analysis.update_alt_sets(this, acquire_mask))
        return;
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          analysis.record_remote(this, acquire_mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = acquire_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            acquire_mask -= non_subset;
            if (!acquire_mask)
              return;
          }
        }
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(acquire_mask))
      {
        check_for_unrefined_remainder(eq, acquire_mask,
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & acquire_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!cached_set)
          analysis.filter_alt_sets(this, to_traverse.get_valid_mask());
        // Update the acquire mask and the remove_mask if there is one
        acquire_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->acquire_restrictions(analysis, it->second, deferral_events,
                    applied_events, NULL/*remove mask*/, false/*original set*/);
        eq.reacquire();
        // Return if our acquire user mask is empty
        if (!acquire_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      acquire_mask &= restricted_fields;
      if (!acquire_mask)
        return;
      // Now we need to lock the analysis if we're going to do this traversal
      AutoLock a_lock(analysis);
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            restricted_instances.begin(); it != restricted_instances.end();it++)
      {
        const FieldMask overlap = acquire_mask & it->second;
        if (!overlap)
          continue;
        InstanceView *view = it->first->as_instance_view();
        analysis.record_instance(view, overlap);
      }
      restricted_fields -= acquire_mask;
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::release_restrictions(ReleaseAnalysis &analysis,
                                          FieldMask release_mask,
                                          std::set<RtEvent> &deferral_events,
                                          std::set<RtEvent> &applied_events,
                                          FieldMask *remove_mask,
                                          const bool cached_set/*=true*/,
                                          const bool already_deferred/*=false*/)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, release_mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!cached_set && analysis.update_alt_sets(this, release_mask))
        return;
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          analysis.record_remote(this, release_mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = release_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            release_mask -= non_subset;
            if (!release_mask)
              return;
          }
        }
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(release_mask))
      {
        check_for_unrefined_remainder(eq, release_mask,
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & release_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!cached_set)
          analysis.filter_alt_sets(this, to_traverse.get_valid_mask());
        // Update the release mask and the remove_mask if there is one
        release_mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->release_restrictions(analysis, it->second, deferral_events,
                    applied_events, NULL/*remove mask*/, false/*original set*/);
        eq.reacquire();
        // Return if ourt release mask is empty
        if (!release_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      // At this point we need to lock the analysis
      AutoLock a_lock(analysis);
      // Find our local restricted instances and views and record them
      InstanceSet local_instances;
      std::vector<InstanceView*> local_views;
      for (FieldMaskSet<InstanceView>::const_iterator it = 
            restricted_instances.begin(); it != restricted_instances.end();it++)
      {
        const FieldMask overlap = it->second & release_mask;
        if (!overlap)
          continue;
        InstanceView *view = it->first->as_instance_view();
        local_instances.add_instance(InstanceRef(view->get_manager(), overlap));
        local_views.push_back(view);
        analysis.record_instance(view, overlap);
      }
      if (analysis.release_aggregator != NULL)
        analysis.release_aggregator->clear_update_fields();
      // Issue the updates
      issue_update_copies_and_fills(analysis.release_aggregator, 
                                    RtEvent::NO_RT_EVENT,
                                    analysis.op, analysis.index, 
                                    false/*track*/, release_mask,
                                    local_instances, local_views, set_expr);
      // Filter the valid views
      filter_valid_instances(release_mask);
      // Update with just the restricted instances
      WrapperReferenceMutator mutator(applied_events);
      record_instances(release_mask, local_instances, local_views, mutator);
      // See if we have any reductions to apply as well
      const FieldMask reduce_mask = release_mask & reduction_fields;
      if (!!reduce_mask)
        apply_reductions(reduce_mask, analysis.release_aggregator, 
                         RtEvent::NO_RT_EVENT, analysis.op, 
                         analysis.index, false/*track*/);
      // Add the fields back to the restricted ones
      restricted_fields |= release_mask;
      if ((analysis.release_aggregator != NULL) && 
           analysis.release_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(analysis.release_aggregator->get_update_fields() * 
                refining_fields);
#endif
        update_guards.insert(analysis.release_aggregator, 
            analysis.release_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
        if (!analysis.release_aggregator->record_guard_set(this))
          assert(false);
#else
        analysis.release_aggregator->record_guard_set(this);
#endif
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_across_copies(CopyAcrossAnalysis &analysis,
                                             FieldMask src_mask,
                                             IndexSpaceExpression *overlap,
                                             std::set<RtEvent> &deferral_events,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // No try lock since we can't defer this because of the overlap
      // Also, while you might think this could a read-only lock since
      // we're just reading meta-data, that's not quite right because
      // we need exclusive access to data structures in check_for_migration
      AutoLock eq(eq_lock);
      // No alt-set tracking here for copy across because we might
      // need to to traverse this multiple times with different expressions
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          analysis.record_remote(this, src_mask, 
                                 logical_owner_space, false/*cached set*/);
          return;
        }
        else
        {
          const FieldMask non_subset = src_mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, false/*cached set*/);
            src_mask -= non_subset;
            if (!src_mask)
              return;
          }
        }
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(src_mask))
      {
        check_for_unrefined_remainder(eq, src_mask,
                                      analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & src_mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // No alt-set tracking here, see comment above
        // Update the release mask and the remove_mask if there is one
        src_mask -= to_traverse.get_valid_mask();
        // No alt-set tracking here, see comment above
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
        {
          IndexSpaceExpression *subset_overlap = runtime->forest->
            intersect_index_spaces(it->first->set_expr, overlap);
          if (subset_overlap->is_empty())
            continue;
          it->first->issue_across_copies(analysis, it->second, subset_overlap,
                                         deferral_events, applied_events);
        }
        eq.reacquire();
        // Return if ourt source mask is empty
        if (!src_mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
      assert(IS_READ_ONLY(analysis.src_usage));
#endif
      // We need to lock the analysis at this point
      AutoLock a_lock(analysis);
      // Check for any uninitialized fields
      const FieldMask uninit = src_mask - valid_instances.get_valid_mask();
      if (!!uninit)
        analysis.record_uninitialized(uninit, applied_events);
      // TODO: Handle the case where we are predicated
      if (analysis.pred_guard.exists())
        assert(false);
      // See if there are any other predicate guard fields that we need
      // to have as preconditions before applying our owner updates
      if (!update_guards.empty() && 
          !(src_mask * update_guards.get_valid_mask()))
      {
        for (FieldMaskSet<CopyFillGuard>::iterator it = 
              update_guards.begin(); it != update_guards.end(); it++)
        {
          if (src_mask * it->second)
            continue;
          // No matter what record our dependences on the prior guards
#ifdef NON_AGGRESSIVE_AGGREGATORS
          const RtEvent guard_event = it->first->effects_applied;
#else
          const RtEvent guard_event = 
            (analysis.original_source == local_space) ?
            it->first->guard_postcondition :
            it->first->effects_applied;
#endif
          analysis.guard_events.insert(guard_event);
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
        // We need to figure out how to issue these copies ourself since
        // we need to map from one field to another
        // First construct a map from dst indexes to src indexes 
        std::map<unsigned,unsigned> dst_to_src;
        for (unsigned idx = 0; idx < analysis.src_indexes.size(); idx++)
          dst_to_src[analysis.dst_indexes[idx]] = analysis.src_indexes[idx];
        // Iterate over the target instances
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
          // Now find all the source instances for this destination
          FieldMaskSet<LogicalView> src_views;
          for (FieldMaskSet<LogicalView>::const_iterator it =
                valid_instances.begin(); it != valid_instances.end(); it++)
          {
            const FieldMask field_overlap = it->second & source_mask;
            if (!field_overlap)
              continue;
            src_views.insert(it->first, field_overlap);
          }
#ifdef DEBUG_LEGION
          if (src_views.empty()) // will only happen in error case
            continue;
#endif
          across_aggregator->record_updates(analysis.target_views[idx],
                            src_views, source_mask, overlap, analysis.redop, 
                            analysis.across_helpers[idx]);
        }
        // Now check for any reductions that need to be applied
        FieldMask reduce_mask = reduction_fields & src_mask;
        if (!!reduce_mask)
        {
#ifdef DEBUG_LEGION
          assert(analysis.redop == 0); // can't have reductions of reductions
#endif
          std::map<unsigned,unsigned> src_to_dst;
          for (unsigned idx = 0; idx < analysis.src_indexes.size(); idx++)
            src_to_dst[analysis.src_indexes[idx]] = analysis.dst_indexes[idx];
          int src_fidx = reduce_mask.find_first_set();
          while (src_fidx >= 0)
          {
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(src_fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
            assert(src_to_dst.find(src_fidx) != src_to_dst.end());
#endif
            const unsigned dst_fidx = src_to_dst[src_fidx];
            // Find the target targets and record them
            for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
            {
              const FieldMask target_mask = 
                analysis.target_instances[idx].get_valid_fields();
              if (!target_mask.is_set(dst_fidx))
                continue;
              across_aggregator->record_reductions(analysis.target_views[idx],
                                         finder->second, src_fidx, dst_fidx, 
                                         overlap, analysis.across_helpers[idx]);
            }
            src_fidx = reduce_mask.find_next_set(src_fidx+1);
          }
        }
      }
      else if (analysis.redop == 0)
      {
        // Fields align and we're not doing a reduction so we can just 
        // do a normal update copy analysis to figure out what to do
        issue_update_copies_and_fills(across_aggregator, RtEvent::NO_RT_EVENT,
                                      analysis.op, analysis.src_index, 
                                      true/*track effects*/, src_mask, 
                                      analysis.target_instances,
                                      analysis.target_views, overlap, 
                                      true/*skip check*/, analysis.dst_index);
        // We also need to check for any reductions that need to be applied
        const FieldMask reduce_mask = reduction_fields & src_mask;
        if (!!reduce_mask)
        {
          int fidx = reduce_mask.find_first_set();
          while (fidx >= 0)
          {
            std::map<unsigned,std::vector<ReductionView*> >::const_iterator
              finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
            assert(finder != reduction_instances.end());
#endif
            // Find the target targets and record them
            for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
            {
              const FieldMask target_mask = 
                analysis.target_instances[idx].get_valid_fields();
              if (!target_mask.is_set(fidx))
                continue;
              across_aggregator->record_reductions(analysis.target_views[idx],
                                           finder->second, fidx, fidx, overlap);
            }
            fidx = reduce_mask.find_next_set(fidx+1);
          }
        }
      }
      else
      {
        // Fields align but we're doing a reduction across
        // Find the valid views that we need for issuing the updates  
        FieldMaskSet<LogicalView> src_views;
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask overlap = it->second & src_mask;
          if (!overlap)
            continue;
          src_views.insert(it->first, overlap);
        }
        for (unsigned idx = 0; idx < analysis.target_views.size(); idx++)
        {
          const FieldMask &mask = 
            analysis.target_instances[idx].get_valid_fields(); 
#ifdef DEBUG_LEGION
          if (src_views.empty()) // will only happen in error case
            continue;
#endif
          across_aggregator->record_updates(analysis.target_views[idx], 
              src_views, mask, overlap, analysis.redop, NULL/*across*/);
        }
        // There shouldn't be any reduction instances to worry about here
#ifdef DEBUG_LEGION
        assert(reduction_fields * src_mask);
#endif
      } 
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::overwrite_set(OverwriteAnalysis &analysis,
                                       FieldMask mask,
                                       std::set<RtEvent> &deferral_events,
                                       std::set<RtEvent> &applied_events,
                                       FieldMask *remove_mask,
                                       const bool cached_set,
                                       const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!cached_set && analysis.update_alt_sets(this, mask))
        return;
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          analysis.record_remote(this, mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            mask -= non_subset;
            if (!mask)
              return;
          }
        }
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(mask))
      {
        check_for_unrefined_remainder(eq, mask, analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!cached_set)
          analysis.filter_alt_sets(this, to_traverse.get_valid_mask());
        // Update the mask and the remove_mask if there is one
        mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->overwrite_set(analysis, it->second, deferral_events,
              applied_events, NULL/*remove mask*/, false/*cachd set*/);
        eq.reacquire();
        // Return if ourt mask is empty
        if (!mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      // At this point we need to lock the analysis
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
        if (analysis.add_restriction || 
            !restricted_fields || (restricted_fields * mask))
        {
          // Easy case, just filter everything and add the new view
          const FieldMask reduce_filter = mask & reduction_fields;
          if (!!reduce_filter)
            filter_reduction_instances(reduce_filter);
          filter_valid_instances(mask);
          if (!analysis.views.empty())
          {
            for (std::set<LogicalView*>::const_iterator it = 
                  analysis.views.begin(); it != analysis.views.end(); it++)
            {
              FieldMaskSet<LogicalView>::iterator finder = 
                valid_instances.find(*it);
              if (finder == valid_instances.end())
              {
                WrapperReferenceMutator mutator(applied_events);
                (*it)->add_nested_valid_ref(did, &mutator);
                valid_instances.insert(*it, mask);
              }
              else
                finder.merge(mask);
            }
          }
        }
        else
        {
          // We overlap with some restricted fields so we can't filter
          // or update any restricted fields
          const FieldMask update_mask = mask - restricted_fields;
          if (!!update_mask)
          {
            const FieldMask reduce_filter = update_mask & reduction_fields;
            if (!!reduce_filter)
              filter_reduction_instances(reduce_filter);
            filter_valid_instances(update_mask);
            if (!analysis.views.empty())
            {
              for (std::set<LogicalView*>::const_iterator it = 
                    analysis.views.begin(); it != analysis.views.end(); it++)
              {
                FieldMaskSet<LogicalView>::iterator finder = 
                  valid_instances.find(*it);
                if (finder == valid_instances.end())
                {
                  WrapperReferenceMutator mutator(applied_events);
                  (*it)->add_nested_valid_ref(did, &mutator);
                  valid_instances.insert(*it, mask);
                }
                else
                  finder.merge(mask);
              }
            }
          }
        }
        // Advance the version numbers
        advance_version_numbers(mask);
        if (analysis.add_restriction)
        {
#ifdef DEBUG_LEGION
          assert(analysis.views.size() == 1);
          LogicalView *log_view = *(analysis.views.begin());
          assert(log_view->is_instance_view());
#else
          LogicalView *log_view = *(analysis.views.begin());
#endif
          InstanceView *inst_view = log_view->as_instance_view();
          FieldMaskSet<InstanceView>::iterator restricted_finder = 
            restricted_instances.find(inst_view);
          if (restricted_finder == restricted_instances.end())
          {
            WrapperReferenceMutator mutator(applied_events);
            inst_view->add_nested_valid_ref(did, &mutator);
            restricted_instances.insert(inst_view, mask);
          }
          else
            restricted_finder.merge(mask);
          restricted_fields |= mask; 
        }
        else if (!!restricted_fields && !analysis.views.empty())
        {
          // Check to see if we have any restricted outputs to write
          const FieldMask restricted_overlap = mask & restricted_fields;
          if (!!restricted_overlap)
          {
            // Pick a random view and copy from it
            LogicalView *log_view = *(analysis.views.begin());
            copy_out(restricted_overlap, log_view, analysis.op, 
                     analysis.index, analysis.output_aggregator);
          }
        }
      }
      if ((analysis.output_aggregator != NULL) &&
           analysis.output_aggregator->has_update_fields())
      {
#ifdef DEBUG_LEGION
        assert(analysis.output_aggregator->get_update_fields() * 
                refining_fields);
#endif
        update_guards.insert(analysis.output_aggregator, 
            analysis.output_aggregator->get_update_fields());
#ifdef DEBUG_LEGION
        if (!analysis.output_aggregator->record_guard_set(this))
          assert(false);
#else
        analysis.output_aggregator->record_guard_set(this);
#endif
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_set(FilterAnalysis &analysis, FieldMask mask,
                                    std::set<RtEvent> &deferral_events,
                                    std::set<RtEvent> &applied_events,
                                    FieldMask *remove_mask,
                                    const bool cached_set/*=true*/,
                                    const bool already_deferred/*=false*/)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        defer_traversal(eq, analysis, mask, deferral_events,
                        applied_events, already_deferred, cached_set);
        return;
      }
      if (!cached_set && analysis.update_alt_sets(this, mask))
        return;
      if (!is_logical_owner())
      {
        // First check to see if our subsets are up to date
        if (eq_state == INVALID_STATE)
          request_remote_subsets(applied_events);
        if (subsets.empty())
        {
          analysis.record_remote(this, mask, 
                                 logical_owner_space, cached_set);
          return;
        }
        else
        {
          const FieldMask non_subset = mask - subsets.get_valid_mask();
          if (!!non_subset)
          {
            analysis.record_remote(this, non_subset, 
                                   logical_owner_space, cached_set);
            mask -= non_subset;
            if (!mask)
              return;
          }
        }
      }
      // Guard to prevent migration while we may release the lock
      increment_pending_analyses();
      // If we've been refined, we need to get the names of 
      // the sub equivalence sets to try
      while (is_refined(mask))
      {
        check_for_unrefined_remainder(eq, mask, analysis.original_source);
        FieldMaskSet<EquivalenceSet> to_traverse;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          to_traverse.insert(it->first, overlap);
        }
        eq.release();
        // Remove ourselves if we recursed
        if (!cached_set)
          analysis.filter_alt_sets(this, to_traverse.get_valid_mask());
        // Update the mask and the remove_mask if there is one
        mask -= to_traverse.get_valid_mask();
        if (remove_mask != NULL)
          *remove_mask |= to_traverse.get_valid_mask();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_traverse.begin(); it != to_traverse.end(); it++) 
          it->first->filter_set(analysis, it->second, deferral_events,
              applied_events, NULL/*remove mask*/, false/*original set*/);
        eq.reacquire();
        // Return if ourt mask is empty
        if (!mask)
        {
          decrement_pending_analyses();
          return;
        }
      }
      decrement_pending_analyses();
#ifdef DEBUG_LEGION
      // Should only be here if we're the owner
      assert(is_logical_owner());
#endif
      // No need to lock the analysis here since we're not going to change it
      FieldMaskSet<LogicalView>::iterator finder = 
        valid_instances.find(analysis.inst_view);
      if (finder != valid_instances.end())
      {
        finder.filter(mask);
        if (!finder->second)
        {
          if (analysis.inst_view->remove_nested_valid_ref(did))
            delete analysis.inst_view;
          valid_instances.erase(finder);
        }
      }
      if ((analysis.registration_view != NULL) && 
          (analysis.registration_view != analysis.inst_view))
      {
        finder = valid_instances.find(analysis.registration_view);
        if (finder != valid_instances.end())
        {
          finder.filter(mask);
          if (!finder->second)
          {
            if (analysis.registration_view->remove_nested_valid_ref(did))
              delete analysis.registration_view;
            valid_instances.erase(finder);
          }
        }
      }
      if (analysis.remove_restriction)
      {
        restricted_fields -= mask;
#ifdef DEBUG_LEGION
        assert(analysis.inst_view != NULL);
#endif
        FieldMaskSet<InstanceView>::iterator restricted_finder = 
          restricted_instances.find(analysis.inst_view);
        if (restricted_finder != restricted_instances.end())
        {
          restricted_finder.filter(mask);
          if (!restricted_finder->second)
          {
            if (analysis.inst_view->remove_nested_valid_ref(did))
              delete analysis.inst_view;
            restricted_instances.erase(restricted_finder);
          }
        }
      }
      check_for_migration(analysis, applied_events);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::request_remote_subsets(std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_logical_owner());
      assert(eq_state == INVALID_STATE);
      assert(!transition_event.exists());
#endif
      // It's not actually ok to block here or we risk a hang so if we're
      // not already valid and haven't requested a valid copy yet then
      // go ahead and do that and record the event as an applied event 
      // to ensure we get the update for the next user
      transition_event = Runtime::create_rt_user_event();
      eq_state = PENDING_VALID_STATE;
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(local_space);
      }
      runtime->send_equivalence_set_subset_request(logical_owner_space, rez);
      applied.insert(transition_event);
    }
    
    //--------------------------------------------------------------------------
    void EquivalenceSet::record_instances(const FieldMask &record_mask,
                                 const InstanceSet &target_instances, 
                                 const std::vector<InstanceView*> &target_views,
                                          ReferenceMutator &mutator)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        const FieldMask valid_mask = 
          target_instances[idx].get_valid_fields() & record_mask;
        if (!valid_mask)
          continue;
        InstanceView *target = target_views[idx];
        // Add it to the set
        FieldMaskSet<LogicalView>::iterator finder = 
          valid_instances.find(target);
        if (finder == valid_instances.end())
        {
          target->add_nested_valid_ref(did, &mutator);
          valid_instances.insert(target, valid_mask);
        }
        else
          finder.merge(valid_mask);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::issue_update_copies_and_fills(
                                 CopyFillAggregator *&aggregator,
                                 const RtEvent guard_event,
                                 Operation *op, const unsigned index,
                                 const bool track_events,
                                 FieldMask update_mask,
                                 const InstanceSet &target_instances,
                                 const std::vector<InstanceView*> &target_views,
                                 IndexSpaceExpression *update_expr,
                                 const bool skip_check,
                                 const int dst_index /*= -1*/) const
    //--------------------------------------------------------------------------
    {
      if (update_expr->is_empty())
        return;
      if (!skip_check)
      {
        // Scan through and figure out which fields are already valid
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          FieldMaskSet<LogicalView>::const_iterator finder = 
            valid_instances.find(target_views[idx]);
          if (finder == valid_instances.end())
            continue;
          const FieldMask &needed_mask = 
            target_instances[idx].get_valid_fields();
          const FieldMask already_valid = needed_mask & finder->second;
          if (!already_valid)
            continue;
          update_mask -= already_valid;
          // If we're already valid for all the fields then we're done
          if (!update_mask)
            return;
        }
      }
#ifdef DEBUG_LEGION
      assert(!!update_mask);
#endif
      // Find the valid views that we need for issuing the updates  
      FieldMaskSet<LogicalView> valid_views;
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        const FieldMask overlap = it->second & update_mask;
        if (!overlap)
          continue;
        valid_views.insert(it->first, overlap);
      }
      // Can happen with uninitialized data, we handle this case
      // before calling this method
      if (valid_views.empty())
        return;
      if (target_instances.size() == 1)
      {
        if (aggregator == NULL)
          aggregator = (dst_index >= 0) ?
            new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                   guard_event, track_events) :
            new CopyFillAggregator(runtime->forest, op, index,
                                   guard_event, track_events); 
        aggregator->record_updates(target_views[0], valid_views, 
                                   update_mask, update_expr);
      }
      else if (valid_views.size() == 1)
      {
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask dst_mask = update_mask &
            target_instances[idx].get_valid_fields();
          if (!dst_mask)
            continue;
          if (aggregator == NULL)
            aggregator = (dst_index >= 0) ?
              new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                     guard_event, track_events) :
              new CopyFillAggregator(runtime->forest, op, index,
                                     guard_event, track_events);
          aggregator->record_updates(target_views[idx], valid_views,
                                     dst_mask, update_expr);
        }
      }
      else
      {
        for (unsigned idx = 0; idx < target_views.size(); idx++)
        {
          const FieldMask dst_mask = update_mask & 
            target_instances[idx].get_valid_fields();
          // Can happen in cases with uninitialized data
          if (!dst_mask)
            continue;
          FieldMaskSet<LogicalView> src_views;
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                valid_views.begin(); it != valid_views.end(); it++)
          {
            const FieldMask overlap = dst_mask & it->second;
            if (!overlap)
              continue;
            src_views.insert(it->first, overlap);
          }
          if (aggregator == NULL)
            aggregator = (dst_index >= 0) ?
              new CopyFillAggregator(runtime->forest, op, index, dst_index,
                                     guard_event, track_events) :
              new CopyFillAggregator(runtime->forest, op, index,
                                     guard_event, track_events);
          aggregator->record_updates(target_views[idx], src_views,
                                     dst_mask, update_expr);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_valid_instances(const FieldMask &filter_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!filter_mask);
#endif
      std::vector<LogicalView*> to_erase;
      for (FieldMaskSet<LogicalView>::iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        const FieldMask overlap = it->second & filter_mask;
        if (!overlap)
          continue;
        it.filter(overlap);
        if (!it->second)
          to_erase.push_back(it->first);
      }
      if (!to_erase.empty())
      {
        for (std::vector<LogicalView*>::const_iterator it = 
              to_erase.begin(); it != to_erase.end(); it++)
        {
          valid_instances.erase(*it);
          if ((*it)->remove_nested_valid_ref(did))
            delete (*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_reduction_instances(const FieldMask &to_filter)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!to_filter);
#endif
      int fidx = to_filter.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::vector<ReductionView*> >::iterator
          finder = reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
        assert(finder != reduction_instances.end());
#endif
        for (std::vector<ReductionView*>::const_iterator it = 
              finder->second.begin(); it != finder->second.end(); it++)
          if ((*it)->remove_nested_valid_ref(did))
            delete (*it);
        reduction_instances.erase(finder);
        fidx = to_filter.find_next_set(fidx+1);
      }
      reduction_fields -= to_filter;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::apply_reductions(const FieldMask &reduce_mask,
                                          CopyFillAggregator *&aggregator,
                                          const RtEvent guard_event,
                                          Operation *op, const unsigned index,
                                          const bool trace_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!reduce_mask);
      assert(!set_expr->is_empty());
#endif
      int fidx = reduce_mask.find_first_set();
      while (fidx >= 0)
      {
        std::map<unsigned,std::vector<ReductionView*> >::iterator finder = 
          reduction_instances.find(fidx);
#ifdef DEBUG_LEGION
        assert(finder != reduction_instances.end());
#endif
        if (!finder->second.empty())
        {
          // Find the target targets and record them
          for (FieldMaskSet<LogicalView>::const_iterator it = 
                valid_instances.begin(); it != valid_instances.end(); it++)
          {
            if (!it->second.is_set(fidx))
              continue;
            // Shouldn't have any deferred views here
            InstanceView *dst_view = it->first->as_instance_view();
            if (aggregator == NULL)
              aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                                  guard_event, trace_events);
            aggregator->record_reductions(dst_view, finder->second, fidx, 
                                          fidx, set_expr);
          }
          // Remove the reduction views from those available
          for (std::vector<ReductionView*>::const_iterator it = 
                finder->second.begin(); it != finder->second.end(); it++)
            if ((*it)->remove_nested_valid_ref(did))
              delete (*it);
        }
        reduction_instances.erase(finder);
        fidx = reduce_mask.find_next_set(fidx+1);
      }
      // Record that we advanced the version number in this case
      advance_version_numbers(reduce_mask);
      // These reductions have been applied so we are done
      reduction_fields -= reduce_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::copy_out(const FieldMask &restricted_mask,
                                  const InstanceSet &src_instances,
                                  const std::vector<InstanceView*> &src_views,
                                  Operation *op, const unsigned index,
                                  CopyFillAggregator *&aggregator) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!restricted_mask);
#endif
      if (set_expr->is_empty())
        return;
      if (valid_instances.size() == 1)
      {
        // Only 1 destination
        FieldMaskSet<LogicalView>::const_iterator first = 
          valid_instances.begin();
#ifdef DEBUG_LEGION
        assert(!(restricted_mask - first->second));
#endif
        InstanceView *dst_view = first->first->as_instance_view();
        FieldMaskSet<LogicalView> srcs;
        for (unsigned idx = 0; idx < src_views.size(); idx++)
        {
          if (first->first == src_views[idx])
            continue;
          const FieldMask overlap = 
            src_instances[idx].get_valid_fields() & restricted_mask;
          if (!overlap)
            continue;
          srcs.insert(src_views[idx], overlap);
        }
        if (!srcs.empty())
        {
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                      RtEvent::NO_RT_EVENT, true/*track*/);
          aggregator->record_updates(dst_view, srcs, restricted_mask, set_expr);
        }
      }
      else if (src_instances.size() == 1)
      {
        // Only 1 source
#ifdef DEBUG_LEGION
        assert(!(restricted_mask - src_instances[0].get_valid_fields()));
#endif
        FieldMaskSet<LogicalView> srcs;
        srcs.insert(src_views[0], restricted_mask);
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          if (it->first == src_views[0])
            continue;
          const FieldMask overlap = it->second & restricted_mask;
          if (!overlap)
            continue;
          InstanceView *dst_view = it->first->as_instance_view();
          if (aggregator == NULL)
            aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                      RtEvent::NO_RT_EVENT, true/*track*/);
          aggregator->record_updates(dst_view, srcs, overlap, set_expr);
        }
      }
      else
      {
        // General case for cross-products
        for (FieldMaskSet<LogicalView>::const_iterator it = 
              valid_instances.begin(); it != valid_instances.end(); it++)
        {
          const FieldMask dst_overlap = it->second & restricted_mask;
          if (!dst_overlap)
            continue;
          InstanceView *dst_view = it->first->as_instance_view();
          FieldMaskSet<LogicalView> srcs;
          for (unsigned idx = 0; idx < src_views.size(); idx++)
          {
            if (dst_view == src_views[idx])
              continue;
            const FieldMask src_overlap = 
              src_instances[idx].get_valid_fields() & dst_overlap;
            if (!src_overlap)
              continue;
            srcs.insert(src_views[idx], src_overlap);
          }
          if (!srcs.empty())
          {
            if (aggregator == NULL)
              aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                        RtEvent::NO_RT_EVENT, true/*track*/);
            aggregator->record_updates(dst_view, srcs, dst_overlap, set_expr);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::copy_out(const FieldMask &restricted_mask, 
                                   LogicalView *src_view, 
                                   Operation *op, const unsigned index,
                                   CopyFillAggregator *&aggregator) const
    //--------------------------------------------------------------------------
    {
      if (set_expr->is_empty())
        return;
      FieldMaskSet<LogicalView> srcs;
      srcs.insert(src_view, restricted_mask);
      for (FieldMaskSet<LogicalView>::const_iterator it = 
            valid_instances.begin(); it != valid_instances.end(); it++)
      {
        if (it->first == src_view)
          continue;
        const FieldMask overlap = it->second & restricted_mask;
        if (!overlap)
          continue;
        InstanceView *dst_view = it->first->as_instance_view();
        if (aggregator == NULL)
          aggregator = new CopyFillAggregator(runtime->forest, op, index,
                                    RtEvent::NO_RT_EVENT, true/*track*/);
        aggregator->record_updates(dst_view, srcs, overlap, set_expr);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::advance_version_numbers(FieldMask advance_mask)
    //--------------------------------------------------------------------------
    {
      std::vector<VersionID> to_remove; 
      for (LegionMap<VersionID,FieldMask>::iterator it = 
            version_numbers.begin(); it != version_numbers.end(); it++)
      {
        const FieldMask overlap = it->second & advance_mask;
        if (!overlap)
          continue;
        LegionMap<VersionID,FieldMask>::iterator finder = 
          version_numbers.find(it->first + 1);
        if (finder == version_numbers.end())
          version_numbers[it->first + 1] = overlap;
        else
          finder->second |= overlap;
        it->second -= overlap;
        if (!it->second)
          to_remove.push_back(it->first);
        advance_mask -= overlap;
        if (!advance_mask)
          break;
      }
      if (!to_remove.empty())
      {
        for (std::vector<VersionID>::const_iterator it = 
              to_remove.begin(); it != to_remove.end(); it++)
          version_numbers.erase(*it);
      }
      if (!!advance_mask)
        version_numbers[init_version] = advance_mask;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::perform_refinements(void)
    //--------------------------------------------------------------------------
    {
      RtUserEvent to_trigger;
      FieldMaskSet<EquivalenceSet> to_perform;
      std::set<EquivalenceSet*> remote_first_refs;
      std::set<RtEvent> remote_subsets_informed;
      do 
      {
        std::set<RtEvent> refinements_done;
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              to_perform.begin(); it != to_perform.end(); it++)
        {
          // Need the lock in read-only mode when doing the clone
          AutoLock eq(eq_lock,1,false/*exclusive*/);
          AddressSpaceID alt_space = it->first->clone_from(this, it->second);
          // If the user asked us to send it to a different node then do that
          if (alt_space != local_space)
          {
            RtUserEvent done_event = Runtime::create_rt_user_event();
            Serializer rez;
            // No RezCheck here because we might need to forward it
            rez.serialize(it->first->did);
            rez.serialize(done_event);
            // Determine whether this is the inital refinement or not
            // VERY IMPORTANT! You cannot use the subsets data structure
            // to test this for the case where we constructed a KD-tree
            // to build intermediate nodes into the refinement tree to
            // avoid exceeding our maximum fanout. Instead we use the 
            // remote_first_refs data structure to test this
            if (remote_first_refs.find(it->first) != remote_first_refs.end())
              rez.serialize<bool>(true); // initial refinement
            else
              rez.serialize<bool>(false); // not initial refinement
            pack_state(rez, it->second);
            runtime->send_equivalence_set_remote_refinement(
                it->first->logical_owner_space, rez);
            refinements_done.insert(done_event);
          }
        }
        if (!refinements_done.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(refinements_done);
          wait_on.wait();
        }
        AutoLock eq(eq_lock);
#ifdef DEBUG_LEGION
        assert(is_logical_owner());
        assert(waiting_event.exists());
        assert(eq_state == REFINING_STATE);
#endif
        // Add any new refinements to our set and record any
        // potentially complete fields
        FieldMask complete_mask;
        if (!to_perform.empty())
        {
#ifdef DEBUG_LEGION
          // These masks should be identical
          assert(refining_fields == to_perform.get_valid_mask());
#endif
          complete_mask = refining_fields;
          refining_fields.clear();
          // References were added to these sets when they were added
          // to the pending refinement queue, if they are already here
          // then we can remove the duplicate reference, no need to 
          // check for deletion since we know we hold another reference
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                to_perform.begin(); it != to_perform.end(); it++)
            if (!subsets.insert(it->first, it->second))
              it->first->remove_nested_resource_ref(did);
          to_perform.clear();
          remote_first_refs.clear();
          // See if there was anyone waiting for us to be done
          if (transition_event.exists())
          {
            Runtime::trigger_event(transition_event);
            transition_event = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
#ifdef DEBUG_LEGION
        assert(!refining_fields);
        assert(!transition_event.exists());
#endif
        // Fields which are still being refined are not complete
        while (!!complete_mask)
        {
          if (!pending_refinements.empty())
          {
            complete_mask -= pending_refinements.get_valid_mask();
            if (!complete_mask)
              break;
          }
          if (!unrefined_remainders.empty())
          {
            complete_mask -= unrefined_remainders.get_valid_mask();
            if (!complete_mask)
              break;
          }
          if (!disjoint_partition_refinements.empty())
            complete_mask -= disjoint_partition_refinements.get_valid_mask();
          // Only need one iteration of this loop
          break;
        }
        if (!!complete_mask)
        {
          FieldMaskSet<EquivalenceSet> complete_subsets;
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
          {
            const FieldMask overlap = complete_mask & it->second;
            if (!overlap)
              continue;
            complete_subsets.insert(it->first, overlap);
          }
          if (complete_subsets.size() > LEGION_MAX_BVH_FANOUT)
          {
            // Sort these info field mask sets
            LegionList<FieldSet<EquivalenceSet*> > field_sets;
            complete_subsets.compute_field_sets(complete_mask, field_sets);
            for (LegionList<FieldSet<EquivalenceSet*> >::const_iterator
                  fit = field_sets.begin(); fit != field_sets.end(); fit++)
            {
              if (fit->elements.size() <= LEGION_MAX_BVH_FANOUT)
                continue;
              KDTree *tree = NULL;
              switch (set_expr->get_num_dims())
              {
#define KDDIM(DIM) \
                case DIM: \
                  { \
                    tree = new KDNode<DIM>(set_expr, runtime, 0/*dim*/); \
                    break; \
                  }
                LEGION_FOREACH_N(KDDIM)
#undef KDDIM
                default:
                  assert(false);
              }
              // Refine the tree to make the new subsets
              std::vector<EquivalenceSet*> new_subsets(
                  fit->elements.begin(), fit->elements.end());
#ifdef LEGION_MAX_BVH_DEPTH
              unsigned max_depth = 0; 
              size_t bvh_ratio = new_subsets.size() / LEGION_MAX_BVH_FANOUT;
              while (bvh_ratio >>= 1)
                max_depth++;
#else
              unsigned max_depth = new_subsets.size();
#endif
              if (tree->refine(new_subsets, fit->set_mask, max_depth))
              {
                // Remove old references
                for (std::set<EquivalenceSet*>::const_iterator it = 
                      fit->elements.begin(); it != fit->elements.end(); it++)
                {
                  bool found = false;
                  for (std::vector<EquivalenceSet*>::const_iterator nit = 
                        new_subsets.begin(); nit != new_subsets.end(); nit++)
                  {
                    if ((*nit) != (*it))
                      continue;
                    found = true;
                    break;
                  }
                  // If the eq set is in the new set then there is nothing to do
                  if (found)
                    continue;
                  // If it's not in the new set, then we need to remove its
                  // fields from the existing subsets
                  FieldMaskSet<EquivalenceSet>::iterator finder = 
                    subsets.find(*it);
#ifdef DEBUG_LEGION
                  assert(finder != subsets.end());
#endif
                  finder.filter(fit->set_mask);
                  if (!finder->second)
                  {
                    if (finder->first->remove_nested_resource_ref(did))
                      delete finder->first;
                    subsets.erase(finder);
                  }
                  // Also remove it from the complete subsets
                  finder = complete_subsets.find(*it);
#ifdef DEBUG_LEGION
                  assert(finder != complete_subsets.end());
#endif
                  finder.filter(fit->set_mask);
                  if (!finder->second)
                    complete_subsets.erase(finder);
                }
                // Add new references
                for (std::vector<EquivalenceSet*>::const_iterator it =
                      new_subsets.begin(); it != new_subsets.end(); it++)
                {
                  if (subsets.insert(*it, fit->set_mask))
                    (*it)->add_nested_resource_ref(did);
                  // Also add it to the complete subsets
                  complete_subsets.insert(*it, fit->set_mask);
                }
              }
              // Clean up the tree
              delete tree;
            }
          }
          // If we're done refining then send updates to any
          // remote sets informing them of the complete set of subsets
          if (!remote_subsets.empty() && !complete_subsets.empty())
          {
            for (std::set<AddressSpaceID>::const_iterator rit = 
                  remote_subsets.begin(); rit != remote_subsets.end(); rit++)
            {
              const RtUserEvent informed = Runtime::create_rt_user_event();
              Serializer rez;
              {
                RezCheck z(rez);
                rez.serialize(did);
                rez.serialize(informed);
                rez.serialize<size_t>(complete_subsets.size());
                for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                      complete_subsets.begin(); it != 
                      complete_subsets.end(); it++)
                {
                  rez.serialize(it->first->did);
                  rez.serialize(it->second);
                }
              }
              runtime->send_equivalence_set_subset_update(*rit, rez);
              remote_subsets_informed.insert(informed);
            }
          }
          // Clean out these entries from our data structures
          if (!valid_instances.empty())
          {
            std::vector<LogicalView*> to_delete;
            for (FieldMaskSet<LogicalView>::iterator it = 
                  valid_instances.begin(); it != valid_instances.end(); it++)
            {
              it.filter(complete_mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<LogicalView*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                valid_instances.erase(*it);
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it); 
              }
              valid_instances.tighten_valid_mask();
            }
          }
          if (!reduction_instances.empty() && 
              !(reduction_fields * complete_mask))
          {
            for (std::map<unsigned,std::vector<ReductionView*> >::
                  iterator rit = reduction_instances.begin();
                  rit != reduction_instances.end(); /*nothing*/)
            {
              if (complete_mask.is_set(rit->first))
              {
                for (std::vector<ReductionView*>::const_iterator it = 
                    rit->second.begin(); it != rit->second.end(); it++)
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it);
                std::map<unsigned,std::vector<ReductionView*> >::iterator
                  to_delete = rit++;
                reduction_instances.erase(to_delete);
              }
              else
                rit++;
            }
            reduction_fields -= complete_mask;
          }
          if (!restricted_instances.empty() && 
              !(restricted_fields * complete_mask))
          {
            std::vector<InstanceView*> to_delete;
            for (FieldMaskSet<InstanceView>::iterator it = 
                  restricted_instances.begin(); it != 
                  restricted_instances.end(); it++)
            {
              it.filter(complete_mask);
              if (!it->second)
                to_delete.push_back(it->first);
            }
            if (!to_delete.empty())
            {
              for (std::vector<InstanceView*>::const_iterator it = 
                    to_delete.begin(); it != to_delete.end(); it++)
              {
                restricted_instances.erase(*it);
                if ((*it)->remove_nested_valid_ref(did))
                  delete (*it); 
              }
              restricted_instances.tighten_valid_mask();
            }
            restricted_fields -= complete_mask;
          }
          for (LegionMap<VersionID,FieldMask>::iterator it =
               version_numbers.begin(); it != version_numbers.end();/*nothing*/)
          {
            it->second -= complete_mask;
            if (!it->second)
            {
              LegionMap<VersionID,FieldMask>::iterator to_delete = it++;
              version_numbers.erase(to_delete);
            }
            else
              it++;
          }
        } 
        // See if we have more refinements to do
        if (pending_refinements.empty())
        {
          // Go back to the mapping state and trigger our done event
          eq_state = MAPPING_STATE;
          to_trigger = waiting_event;
          waiting_event = RtUserEvent::NO_RT_USER_EVENT;
        }
        else // there are more refinements to do so we go around again
        {
#ifdef DEBUG_LEGION
          assert(!refining_fields); // should be empty prior to this
#endif
          refining_fields = pending_refinements.get_valid_mask();
          to_perform.swap(pending_refinements);
          remote_first_refs.swap(remote_first_refinements);
        }
      } while (!to_perform.empty());
#ifdef DEBUG_LEGION
      assert(to_trigger.exists());
#endif
      // Make sure that everyone is informed before we return
      if (!remote_subsets_informed.empty())
        Runtime::trigger_event(to_trigger,
            Runtime::merge_events(remote_subsets_informed));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::record_subset(EquivalenceSet *set,
                                       const FieldMask &set_mask)
    //--------------------------------------------------------------------------
    {
      // This method is only called when adding extra levels to the 
      // equivalence set BVH data structure in order to reduce large
      // fanout. We don't need the lock and we shouldn't have any
      // remote copies of this equivalence set
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
      assert(!has_remote_instances());
#endif
      if (subsets.insert(set, set_mask))
        set->add_nested_resource_ref(did);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::finalize_disjoint_refinement(
               DisjointPartitionRefinement *dis, const FieldMask &finalize_mask)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(unrefined_remainders.get_valid_mask() * finalize_mask);
#endif
      // We're not going to be able to finish up this disjoint
      // partition refinement so restore this to the state
      // for normal traversal
      // Figure out if we finished refining or whether there
      // is still an unrefined remainder
      IndexPartNode *partition = dis->partition;
      if (!dis->is_refined())
      {
        std::set<LegionColor> current_colors;
        const std::map<IndexSpaceNode*,EquivalenceSet*> &children = 
          dis->get_children();
        for (std::map<IndexSpaceNode*,EquivalenceSet*>::const_iterator 
              it = children.begin(); it != children.end(); it++)
          current_colors.insert(it->first->color);
        // No matter what finish making all the children since making
        // disjoint partitions is a good thing
        if (partition->total_children == partition->max_linearized_color)
        {
          for (LegionColor color = 0; 
                color < partition->total_children; color++)
          {
            if (current_colors.find(color) != current_colors.end())
              continue;
            IndexSpaceNode *child = partition->get_child(color);
            if (child->is_empty())
              continue;
            add_pending_refinement(child, finalize_mask, 
                                   child, runtime->address_space);
            // Don't add this to the refinement as we might only be
            // finalizing for a subset of fields and the 
            // DisjointPartitionRefinement should only store entries
            // for children that have been refined for all fields
          }
        }
        else
        {
          ColorSpaceIterator *itr = 
            partition->color_space->create_color_space_iterator();
          while (itr->is_valid())
          {
            const LegionColor color = itr->yield_color();
            if (current_colors.find(color) != current_colors.end())
              continue;
            if (!partition->color_space->contains_color(color))
              continue;
            IndexSpaceNode *child = partition->get_child(color);
            if (child->is_empty())
              continue;
            add_pending_refinement(child, finalize_mask, 
                                   child, runtime->address_space);
            // Don't add this to the refinement as we might only be
            // finalizing for a subset of fields and the 
            // DisjointPartitionRefinement should only store entries
            // for children that have been refined for all fields
          }
          delete itr;
        }
      }
      if (!partition->is_complete())
      {
        // We had all the children, but the partition is not 
        // complete so we actually need to do the subtraction
        IndexSpaceExpression *diff_expr = 
          runtime->forest->subtract_index_spaces(set_expr, 
              partition->get_union_expression());
#ifdef DEBUG_LEGION
        assert((diff_expr != NULL) && !diff_expr->is_empty());
        assert(unrefined_remainders.get_valid_mask() * finalize_mask);
#endif
        if (unrefined_remainders.insert(diff_expr, finalize_mask))
          diff_expr->add_nested_expression_reference(did);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::filter_unrefined_remainders(FieldMask &to_filter,
                                                     IndexSpaceExpression *expr)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!!to_filter);
      assert(!expr->is_empty());
#endif
      if (unrefined_remainders.empty())
        return;
      if (to_filter * unrefined_remainders.get_valid_mask())
        return;
      FieldMaskSet<IndexSpaceExpression> to_add;
      std::vector<IndexSpaceExpression*> to_delete;
      for (FieldMaskSet<IndexSpaceExpression>::iterator it = 
            unrefined_remainders.begin(); it !=
            unrefined_remainders.end(); it++)
      {
        const FieldMask overlap = to_filter & it->second;
        if (!overlap)
          continue;
        IndexSpaceExpression *remainder = 
          runtime->forest->subtract_index_spaces(it->first, expr);
        if (!remainder->is_empty())
          to_add.insert(remainder, overlap);
        it.filter(overlap);
        if (!it->second)
          to_delete.push_back(it->first);
        to_filter -= overlap;
        if (!to_filter)
          break;
      }
      if (!to_delete.empty())
      {
        for (std::vector<IndexSpaceExpression*>::const_iterator 
              it = to_delete.begin(); it != to_delete.end(); it++)
        {
          unrefined_remainders.erase(*it);
          if ((*it)->remove_nested_expression_reference(did))
            delete (*it);
        }
        unrefined_remainders.tighten_valid_mask();
      }
      if (!to_add.empty())
      {
        for (FieldMaskSet<IndexSpaceExpression>::const_iterator
              it = to_add.begin(); it != to_add.end(); it++)
          if (unrefined_remainders.insert(it->first, it->second))
            it->first->add_nested_expression_reference(did);
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::send_equivalence_set(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_owner());
      // We should have had a request for this already
      assert(!has_remote_instance(target));
#endif
      update_remote_instances(target);
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
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
        // priority and therefore shouldn't create an livelock.
        {
          AutoLock eq(eq_lock,1,false/*exclusive*/);
          if (target == logical_owner_space)
            rez.serialize(local_space);
          else
            rez.serialize(logical_owner_space);
        }
        set_expr->pack_expression(rez, target);
        if (index_space_node != NULL)
          rez.serialize(index_space_node->handle);
        else
          rez.serialize(IndexSpace::NO_SPACE);
      }
      runtime->send_equivalence_set_response(target, rez);
    }

    //--------------------------------------------------------------------------
    EquivalenceSet* EquivalenceSet::add_pending_refinement(
                              IndexSpaceExpression *expr, const FieldMask &mask,
                              IndexSpaceNode *node, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_logical_owner());
#endif
      // See if we already have a subset with this expression
      EquivalenceSet *subset = NULL;
      if ((subset_exprs == NULL) && !subsets.empty())
      {
        // Fill in the data structure if it hasn't already been done
        // e.g. due to migration
        subset_exprs = new std::map<IndexSpaceExpression*,EquivalenceSet*>();
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              subsets.begin(); it != subsets.end(); it++)
          (*subset_exprs)[it->first->set_expr] = it->first;
      }
      if (subset_exprs != NULL)
      {
        std::map<IndexSpaceExpression*,EquivalenceSet*>::const_iterator
          finder = subset_exprs->find(expr);
        if (finder != subset_exprs->end())
          subset = finder->second;
      }
      if (subset == NULL)
      {
        // Make a new subset
        subset = new EquivalenceSet(runtime, 
            runtime->get_available_distributed_id(),
            local_space, source, expr, node, true/*register*/);
        if (subset_exprs == NULL)
          subset_exprs = new std::map<IndexSpaceExpression*,EquivalenceSet*>();
        // Save it in the set
        (*subset_exprs)[expr] = subset;
        if (pending_refinements.insert(subset, mask))
          subset->add_nested_resource_ref(did);
        // If this is going to be a remote first refinement record it
        if (source != local_space)
          remote_first_refinements.insert(subset);
      }
      else
      {
        // We should not have this subset already for these fields
#ifdef DEBUG_LEGION
        FieldMaskSet<EquivalenceSet>::const_iterator finder = 
          subsets.find(subset);
        assert((finder == subsets.end()) || (finder->second * mask));
        finder = pending_refinements.find(subset);
        assert((finder == pending_refinements.end()) || 
                (finder->second * mask));
#endif
        if (pending_refinements.insert(subset, mask))
          subset->add_nested_resource_ref(did);
      }
      // Launch the refinement task if there isn't one already running
      if (eq_state == MAPPING_STATE)
      {
#ifdef DEBUG_LEGION
        assert(!transition_event.exists());
        assert(!waiting_event.exists());
        assert(!refining_fields); // should be empty
#endif
        waiting_event = Runtime::create_rt_user_event();
        eq_state = REFINING_STATE;
        // Launch the refinement task to be performed
        RefinementTaskArgs args(this);
        runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY);
      }
      return subset;
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_request(AddressSpaceID source,
                                                RtUserEvent deferral_event)
    //--------------------------------------------------------------------------
    {
      AutoTryLock eq(eq_lock);
      if (!eq.has_lock())
      {
        // We didn't get the lock so build a continuation
        // We need a name for our completion event that we can use for
        // the atomic compare and swap below
        if (!deferral_event.exists())
        {
          // If we haven't already been deferred then we need to 
          // add ourselves to the back of the list of deferrals
          deferral_event = Runtime::create_rt_user_event();
          const RtEvent continuation_pre = 
            chain_deferral_events(deferral_event);
          DeferSubsetRequestArgs args(this, source, deferral_event);
          runtime->issue_runtime_meta_task(args, 
                          LG_LATENCY_DEFERRED_PRIORITY, continuation_pre);
        }
        else
        {
          // We've already been deferred and our precondition has already
          // triggered so just launch ourselves again whenever the lock
          // should be ready to try again
          DeferSubsetRequestArgs args(this, source, deferral_event);
          runtime->issue_runtime_meta_task(args,
                   LG_LATENCY_DEFERRED_PRIORITY, eq.try_next());
        }
        return;
      }
      if (!is_logical_owner())
      {
        // If we're not the owner anymore then forward on the request
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(source);
        }
        runtime->send_equivalence_set_subset_request(logical_owner_space, rez);
        if (deferral_event.exists())
          Runtime::trigger_event(deferral_event);
        return;
      }
      // If we arrived back at ourself after we were made the owner
      // then there is nothing for us to do, similarly if this is a 
      // duplicate then there is nothing for us to do
      if ((source == local_space) ||
          (remote_subsets.find(source) != remote_subsets.end()))
      {
        if (deferral_event.exists())
          Runtime::trigger_event(deferral_event);
        return;
      }
      // If we're in the process of doing a refinement, wait for
      // that to be done before we do anything else
      if (eq_state == REFINING_STATE)
      {
#ifdef DEBUG_LEGION
        assert(waiting_event.exists());
#endif
        DeferSubsetRequestArgs args(this, source, deferral_event);       
        runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, waiting_event);
        return;
      }
      // Record the remote subsets
      remote_subsets.insert(source);
      // Remote copies of the subsets either have to be empty or a 
      // full copy of the subsets with no partial refinements
      if (!subsets.empty())
      {
        FieldMask complete_mask = subsets.get_valid_mask();
        // Any fields for which we have partial refinements cannot be sent yet
        if (!pending_refinements.empty())
          complete_mask -= pending_refinements.get_valid_mask();
        if (!!refining_fields)
          complete_mask -= refining_fields;
        if (!unrefined_remainders.empty())
          complete_mask -= unrefined_remainders.get_valid_mask();
        if (!!disjoint_partition_refinements.empty())
          complete_mask -= disjoint_partition_refinements.get_valid_mask();
        if (!!complete_mask)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(did);
            rez.serialize<size_t>(subsets.size());
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  subsets.begin(); it != subsets.end(); it++)
            {
              const FieldMask overlap = it->second & complete_mask;
              if (!!overlap)
              {
                rez.serialize(it->first->did);
                rez.serialize(overlap);
              }
              else
                rez.serialize<DistributedID>(0);
            }
          }
          runtime->send_equivalence_set_subset_response(source, rez);
          if (deferral_event.exists())
            Runtime::trigger_event(deferral_event);
          return;
        }
      }
      // If we make it here then we just send a message with an 
      // empty set of subsets to allow forward progress to be made
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize<size_t>(0);
      }
      runtime->send_equivalence_set_subset_response(source, rez);
      if (deferral_event.exists())
        Runtime::trigger_event(deferral_event);
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_subsets;
      derez.deserialize(num_subsets);
      FieldMaskSet<EquivalenceSet> new_subsets;
      if (num_subsets > 0)
      {
        std::set<RtEvent> ready_events;
        for (unsigned idx = 0; idx < num_subsets; idx++)
        {
          DistributedID subdid;
          derez.deserialize(subdid);
          if (subdid == 0)
            continue;
          RtEvent ready;
          EquivalenceSet *set =
            runtime->find_or_request_equivalence_set(subdid, ready);
          if (ready.exists())
            ready_events.insert(ready);
          FieldMask mask;
          derez.deserialize(mask);
          new_subsets.insert(set, mask);
        }
        if (!ready_events.empty())
        {
          const RtEvent wait_on = Runtime::merge_events(ready_events);
          if (wait_on.exists())
            wait_on.wait();
        }
      }
      AutoLock eq(eq_lock);
      if (is_logical_owner())
      {
        // If we've since been made the logical owner then there
        // should be nothing else for us to do
#ifdef DEBUG_LEGION
        assert(new_subsets.empty());
#endif
        return;
      }
      else if (eq_state == PENDING_VALID_STATE)
      {
#ifdef DEBUG_LEGION
        assert(subsets.empty());
        assert(transition_event.exists());
        assert(!transition_event.has_triggered());
#endif
        if (!new_subsets.empty()) 
        {
          subsets.swap(new_subsets);
          // Add the references
          for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                subsets.begin(); it != subsets.end(); it++)
            it->first->add_nested_resource_ref(did);
        }
        // Update the state
        eq_state = VALID_STATE;
        // Trigger the transition state to wake up any waiters
        Runtime::trigger_event(transition_event);
        transition_event = RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    //--------------------------------------------------------------------------
    void EquivalenceSet::process_subset_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_subsets;
      derez.deserialize(num_subsets);
      if (num_subsets == 0)
        return;
      std::vector<EquivalenceSet*> new_subsets(num_subsets); 
      LegionVector<FieldMask> new_masks(num_subsets);
      std::set<RtEvent> wait_for;
      for (unsigned idx = 0; idx < num_subsets; idx++)
      {
        DistributedID subdid;
        derez.deserialize(subdid);
        RtEvent ready;
        new_subsets[idx] = 
          runtime->find_or_request_equivalence_set(subdid, ready);
        if (ready.exists())
          wait_for.insert(ready);
        derez.deserialize(new_masks[idx]);
      }
      if (!wait_for.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
      AutoLock eq(eq_lock);
      if (is_logical_owner())
        // If we've become the logical owner there is nothing to do
        return;
#ifdef DEBUG_LEGION
      assert(eq_state == VALID_STATE);
      assert(!transition_event.exists());
#endif
      for (unsigned idx = 0; idx < num_subsets; idx++)
        if (subsets.insert(new_subsets[idx], new_masks[idx]))
          new_subsets[idx]->add_nested_resource_ref(did);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_refinement(const void *args)
    //--------------------------------------------------------------------------
    {
      const RefinementTaskArgs *rargs = (const RefinementTaskArgs*)args;
      rargs->target->perform_refinements();
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_remote_references(const void *args)
    //--------------------------------------------------------------------------
    {
      const RemoteRefTaskArgs *rargs = (const RemoteRefTaskArgs*)args;
      if (rargs->done_event.exists())
      {
        LocalReferenceMutator mutator; 
        if (rargs->add_references)
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->add_nested_valid_ref(rargs->did, &mutator, it->second);
        }
        else
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->remove_nested_valid_ref(rargs->did, &mutator,it->second);
        }
        const RtEvent done_pre = mutator.get_done_event();
        Runtime::trigger_event(rargs->done_event, done_pre);
      }
      else
      {
        if (rargs->add_references)
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->add_nested_valid_ref(rargs->did, NULL, it->second);
        }
        else
        {
          for (std::map<LogicalView*,unsigned>::const_iterator it = 
                rargs->refs->begin(); it != rargs->refs->end(); it++)
            it->first->remove_nested_valid_ref(rargs->did, NULL, it->second);
        }
      }
      delete rargs->refs;
    }

    //--------------------------------------------------------------------------
    EquivalenceSet::DeferRayTraceArgs::DeferRayTraceArgs(EquivalenceSet *s, 
                          RayTracer *t, IndexSpaceExpression *e, 
                          IndexSpace h, AddressSpaceID o, RtUserEvent d,
                          RtUserEvent def, const FieldMask &m,
                          const PendingRemoteExpression *p)
      : LgTaskArgs<DeferRayTraceArgs>(implicit_provenance),
          set(s), target(t), expr(e), handle(h), origin(o), 
          done(d), deferral(def), ray_mask(new FieldMask(m)),
          pending((p == NULL) ? NULL : new PendingRemoteExpression(*p))
    //--------------------------------------------------------------------------
    {
      if (expr != NULL)
        expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace(const void *args,
                                                     Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferRayTraceArgs *dargs = (const DeferRayTraceArgs*)args;

      // See if we need to load the expression or not
      IndexSpaceExpression *expr = dargs->expr;
      if (expr == NULL)
        expr = runtime->forest->find_remote_expression(*(dargs->pending));
      dargs->set->ray_trace_equivalence_sets(dargs->target, expr,
                          *(dargs->ray_mask), dargs->handle, dargs->origin,
                          dargs->done, dargs->deferral);
      // Clean up our ray mask
      delete dargs->ray_mask;
      // Remove our expression reference too
      if ((dargs->expr != NULL) &&
          dargs->expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->expr;
      if (dargs->pending != NULL)
        delete dargs->pending;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_finish(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRayTraceFinishArgs *dargs = 
        (const DeferRayTraceFinishArgs*)args;
      std::set<RtEvent> done_events;
      for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
            dargs->to_traverse->begin(); it != dargs->to_traverse->end(); it++)
      {
        RtUserEvent done = Runtime::create_rt_user_event();
        std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator
          finder = dargs->exprs->find(it->first);
#ifdef DEBUG_LEGION
        assert(finder != dargs->exprs->end());
#endif
        const IndexSpace subset_handle = 
          (dargs->handle.exists() && 
            (finder->second->get_volume() == dargs->volume)) ? dargs->handle :
              IndexSpace::NO_SPACE;
        it->first->ray_trace_equivalence_sets(dargs->target, finder->second, 
            it->second, subset_handle, dargs->source, done);
        done_events.insert(done);
      }
      if (!done_events.empty())
        Runtime::trigger_event(dargs->done, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(dargs->done);
      for (std::map<EquivalenceSet*,IndexSpaceExpression*>::const_iterator it =
            dargs->exprs->begin(); it != dargs->exprs->end(); it++)
        if (it->second->remove_base_expression_reference(META_TASK_REF))
          delete it->second;
      delete dargs->to_traverse;
      delete dargs->exprs;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_request(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferSubsetRequestArgs *dargs = (const DeferSubsetRequestArgs*)args;
      dargs->set->process_subset_request(dargs->source, dargs->deferral);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_make_owner(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMakeOwnerArgs *dargs = (const DeferMakeOwnerArgs*)args;
      if (dargs->set->make_owner(dargs->new_subsets, dargs->done, 
                                 true/*need lock*/))
        delete dargs->new_subsets;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_merge_or_forward(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMergeOrForwardArgs *dargs = 
        (const DeferMergeOrForwardArgs*)args;
      dargs->set->merge_or_forward(dargs->done, dargs->initial, *(dargs->views),
          *(dargs->reductions), *(dargs->restricted), *(dargs->versions));
      delete dargs->views;
      delete dargs->reductions;
      delete dargs->restricted;
      delete dargs->versions;
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
    EquivalenceSet::DeferResponseArgs::DeferResponseArgs(DistributedID id, 
                          AddressSpaceID src, AddressSpaceID log, 
                          IndexSpaceExpression *ex,
                          const PendingRemoteExpression &p, IndexSpace h)
      : LgTaskArgs<DeferResponseArgs>(implicit_provenance),
        did(id), source(src), logical_owner(log), expr(ex),
        pending((expr != NULL) ? NULL : new PendingRemoteExpression(p)),
        handle(h)
    //--------------------------------------------------------------------------
    {
      if (expr != NULL)
        expr->add_base_expression_reference(META_TASK_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_equivalence_set_response(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      AddressSpaceID logical_owner;
      derez.deserialize(logical_owner);
      PendingRemoteExpression pending;
      RtEvent wait_for;
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest, 
                                                source, pending, wait_for);
      IndexSpace handle;
      derez.deserialize(handle);
      IndexSpaceNode *node = NULL; RtEvent wait_on;
      if (handle.exists())
        node = runtime->forest->get_node(handle, &wait_on);

      // Defer this if the index space expression isn't ready yet
      if (wait_for.exists() || wait_on.exists())
      {
        RtEvent precondition;
        if (wait_for.exists())
        {
          if (wait_on.exists())
            precondition = Runtime::merge_events(wait_for, wait_on);
          else
            precondition = wait_for;
        }
        else
          precondition = wait_on;
        if (precondition.exists() && !precondition.has_triggered())
        {
          DeferResponseArgs args(did, source, logical_owner, expr,
                                 pending, handle);
          runtime->issue_runtime_meta_task(args, LG_LATENCY_MESSAGE_PRIORITY,
                                           precondition);
          return;
        }
        // If we fall through we need to refetch things that we didn't get  
        if (expr == NULL)
          expr = runtime->forest->find_remote_expression(pending);
        if (handle.exists() && (node == NULL))
          node = runtime->forest->get_node(handle);
      }

      void *location;
      EquivalenceSet *set = NULL;
      if (runtime->find_pending_collectable_location(did, location))
        set = new(location) EquivalenceSet(runtime, did, source, logical_owner,
                                           expr, node, false/*register now*/);
      else
        set = new EquivalenceSet(runtime, did, source, logical_owner,
                                 expr, node, false/*register now*/);
      // Once construction is complete then we do the registration
      set->register_with_runtime(NULL/*no remote registration needed*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_deferred_response(const void *args,
                                                             Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      const DeferResponseArgs *dargs = (const DeferResponseArgs*)args;
      IndexSpaceExpression *expr = dargs->expr;
      if (expr == NULL)
        expr = runtime->forest->find_remote_expression(*(dargs->pending));
      IndexSpaceNode *node = NULL;
      if (dargs->handle.exists() && 
          (dargs->logical_owner == runtime->address_space))
        node = runtime->forest->get_node(dargs->handle);

      void *location;
      EquivalenceSet *set = NULL;
      if (runtime->find_pending_collectable_location(dargs->did, location))
        set = new(location) EquivalenceSet(runtime, dargs->did, dargs->source, 
            dargs->logical_owner, expr, node, false/*register now*/);
      else
        set = new EquivalenceSet(runtime, dargs->did, dargs->source, 
            dargs->logical_owner, expr, node, false/*register now*/);
      // Once construction is complete then we do the registration
      set->register_with_runtime(NULL/*no remote registration needed*/);
      // Remove our expression reference too
      if ((dargs->expr != NULL) &&
          dargs->expr->remove_base_expression_reference(META_TASK_REF))
        delete dargs->expr;
      if (dargs->pending != NULL)
        delete dargs->pending;
    }

    //--------------------------------------------------------------------------
    /*static*/void EquivalenceSet::handle_deferred_remove_refs(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferRemoveRefArgs *dargs = (const DeferRemoveRefArgs*)args;
      for (std::vector<IndexSpaceExpression*>::const_iterator it = 
            dargs->references->begin(); it != dargs->references->end(); it++)
        if ((*it)->remove_nested_expression_reference(dargs->source))
          delete (*it);
      delete dargs->references;
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_request(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      AddressSpaceID source;
      derez.deserialize(source);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->process_subset_request(source, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_response(
                                          Deserializer &derez, Runtime *runtime)
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
      set->process_subset_response(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_subset_update(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      DistributedCollectable *dc = runtime->find_distributed_collectable(did);
#ifdef DEBUG_LEGION
      EquivalenceSet *set = dynamic_cast<EquivalenceSet*>(dc);
      assert(set != NULL);
#else
      EquivalenceSet *set = static_cast<EquivalenceSet*>(dc);
#endif
      set->process_subset_update(derez);
      Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);

      RayTracer *target;
      derez.deserialize(target);
      PendingRemoteExpression pending;
      RtEvent expr_ready;
      IndexSpaceExpression *expr = 
        IndexSpaceExpression::unpack_expression(derez, runtime->forest,
                                                source, pending, expr_ready);
      FieldMask ray_mask;
      derez.deserialize(ray_mask);
      IndexSpace handle;
      derez.deserialize(handle);
      AddressSpaceID origin;
      derez.deserialize(origin);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (ready.exists() || expr_ready.exists())
      {
        const RtEvent defer = Runtime::merge_events(ready, expr_ready);
        if (defer.exists() && !defer.has_triggered())
        {
          // We need to defer this until things are ready
          DeferRayTraceArgs args(set, target, expr, 
                                 handle, origin, done_event,
                                 RtUserEvent::NO_RT_USER_EVENT,
                                 ray_mask, &pending); 
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY, defer); 
          return;
        }
        if (expr_ready.exists())
          expr = runtime->forest->find_remote_expression(pending);
        // Fall through and actually do the operation now
      }
      set->ray_trace_equivalence_sets(target, expr, ray_mask, handle, 
                                      origin, done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void EquivalenceSet::handle_ray_trace_response(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      FieldMask eq_mask;
      derez.deserialize(eq_mask);
      RayTracer *target;
      derez.deserialize(target);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      if (ready.exists() && !ready.has_triggered())
      {
        target->record_pending_equivalence_set(set, eq_mask);
        Runtime::trigger_event(done_event, ready);
      }
      else
      {
        target->record_equivalence_set(set, eq_mask);
        Runtime::trigger_event(done_event);
      }
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
      RtUserEvent done;
      derez.deserialize(done);

      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_migration(derez, source, done);
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
    /*static*/ void EquivalenceSet::handle_remote_refinement(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      EquivalenceSet *set = runtime->find_or_request_equivalence_set(did,ready);
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      set->unpack_state(derez);
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
    VersionManager::VersionManager(const VersionManager &rhs)
      : ctx(rhs.ctx), node(rhs.node), runtime(rhs.runtime)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersionManager::~VersionManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersionManager& VersionManager::operator=(const VersionManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersionManager::reset(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      if (!equivalence_sets.empty())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              equivalence_sets.begin(); it != equivalence_sets.end(); it++)
        {
          if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
            delete it->first;
        }
        equivalence_sets.clear();
      }
#ifdef DEBUG_LEGION
      assert(waiting_infos.empty());
      assert(equivalence_sets_ready.empty());
#endif
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::perform_versioning_analysis(InnerContext *context,
                             VersionInfo *version_info, RegionNode *region_node,
                             const FieldMask &version_mask, Operation *op)
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
          if (version_info != NULL)
          {
            if (!(version_mask * equivalence_sets.get_valid_mask()))
            {
              for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                   equivalence_sets.begin(); it != equivalence_sets.end(); it++)
              {
                const FieldMask overlap = it->second & version_mask;
                if (!overlap)
                  continue;
                version_info->record_equivalence_set(this, it->first, overlap);
              }
            }
          }
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields then we are done
          if (!remaining_mask)
            return RtEvent::NO_RT_EVENT;
        }
      }
      // Retake the lock in exclusive mode and make sure we don't lose the race
      RtUserEvent compute_event;
      std::set<RtEvent> wait_on;
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
            wait_on.insert(it->first);
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
          {
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  equivalence_sets.begin(); it != equivalence_sets.end(); it++)
            {
              const FieldMask overlap = it->second & remaining_mask;
              if (!overlap)
                continue;
              version_info->record_equivalence_set(this, it->first, overlap);
            }
          }
          remaining_mask -= equivalence_sets.get_valid_mask();
          // If we got all our fields here and we're not waiting 
          // on any other computations then we're done
          if (!remaining_mask && wait_on.empty())
            return RtEvent::NO_RT_EVENT;
        }
        // If we still have remaining fields then we need to
        // do this computation ourselves
        if (!!remaining_mask)
        {
          compute_event = Runtime::create_rt_user_event();
          equivalence_sets_ready[compute_event] = remaining_mask; 
          wait_on.insert(compute_event);
          waiting_mask |= remaining_mask;
        }
#ifdef DEBUG_LEGION
        assert(!!waiting_mask);
#endif
        // Record that our version info is waiting for these fields
        if (version_info != NULL)
          waiting_infos.insert(version_info, waiting_mask);
      }
      if (compute_event.exists())
      {
        IndexSpaceExpression *expr = region_node->row_source; 
        IndexSpace handle = region_node->row_source->handle;
        RtEvent ready = context->compute_equivalence_sets(this, 
                      region_node->get_tree_id(), handle, expr, 
                      remaining_mask, runtime->address_space);
        if (ready.exists() && !ready.has_triggered())
        {
          // Launch task to finalize the sets once they are ready
          LgFinalizeEqSetsArgs args(this, compute_event, 
                                    op->get_unique_op_id());
          runtime->issue_runtime_meta_task(args, 
                             LG_LATENCY_DEFERRED_PRIORITY, ready);
        }
        else
          finalize_equivalence_sets(compute_event);
      }
      return Runtime::merge_events(wait_on); 
    }

    //--------------------------------------------------------------------------
    void VersionManager::record_equivalence_set(EquivalenceSet *set,
                                                const FieldMask &mask)
    //--------------------------------------------------------------------------
    {
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
      pending_equivalence_sets.insert(set, mask);
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
        // See if there are any other events with overlapping fields,
        // if there are then we can't actually move over any pending
        // equivalence sets for those fields yet since we don't know
        // which are ours, just record dependences on those events
        if (equivalence_sets_ready.size() > 1)
        {
          FieldMask aliased;
          for (LegionMap<RtUserEvent,FieldMask>::const_iterator it =
                equivalence_sets_ready.begin(); it != 
                equivalence_sets_ready.end(); it++)
          {
            if (it->first == done_event)
              continue;
            const FieldMask overlap = it->second & finder->second;
            if (!overlap)
              continue;
            done_preconditions.insert(it->first);
            aliased |= overlap;
          }
          if (!!aliased)
            finder->second -= aliased;
        }
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
        if (!waiting_infos.empty() &&
            !(waiting_infos.get_valid_mask() * finder->second))
        {
          std::vector<VersionInfo*> to_delete;
          for (FieldMaskSet<VersionInfo>::iterator vit = 
                waiting_infos.begin(); vit != waiting_infos.end(); vit++)
          {
            const FieldMask info_overlap = vit->second & finder->second;
            if (!info_overlap)
              continue;
            for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
                  equivalence_sets.begin(); it != equivalence_sets.end(); it++)
            {
              const FieldMask overlap = info_overlap & it->second;
              if (!overlap)
                continue;
              vit->first->record_equivalence_set(this, it->first, overlap);
            }
            vit.filter(info_overlap);
            if (!vit->second)
              to_delete.push_back(vit->first);
          }
          if (!to_delete.empty())
          {
            for (std::vector<VersionInfo*>::const_iterator it = 
                  to_delete.begin(); it != to_delete.end(); it++)
              waiting_infos.erase(*it);
          }
        }
        equivalence_sets_ready.erase(finder);
      }
      if (!done_preconditions.empty())
        Runtime::trigger_event(done_event,
            Runtime::merge_events(done_preconditions));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent VersionManager::record_stale_sets(
                                       FieldMaskSet<EquivalenceSet> &stale_sets)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!stale_sets.empty());
#endif
      RtUserEvent compute_event;
      {
        FieldMask update_mask;
        AutoLock m_lock(manager_lock);
        // See which of our stale sets are still in the valid set, if they
        // are then remove their fields, otherwise record that we don't need
        // to update them
        for (FieldMaskSet<EquivalenceSet>::iterator it = 
              stale_sets.begin(); it != stale_sets.end(); it++)
        {
          FieldMaskSet<EquivalenceSet>::iterator finder = 
            equivalence_sets.find(it->first);
          if (finder != equivalence_sets.end())
          {
            const FieldMask need_refinement = it->second & finder->second;
            if (!!need_refinement)
            {
              update_mask |= need_refinement;
              finder.filter(need_refinement);
              if (!finder->second) // Reference flows back with stale sets
                equivalence_sets.erase(finder);
              else // Add a reference to keep the eq set live until we're done
                it->first->add_base_resource_ref(VERSION_MANAGER_REF);
              it.filter(~need_refinement);
#ifdef DEBUG_LEGION
              assert(!!it->second);
#endif
              continue;
            }
          }
          it.clear();
        }
        if (!update_mask)
          return RtEvent::NO_RT_EVENT;
        compute_event = Runtime::create_rt_user_event();
        equivalence_sets_ready[compute_event] = update_mask; 
      }
      // For these equivalence sets we need to perform additional ray 
      // traces to get the new refineemnts of the sets
      std::set<RtEvent> preconditions;
      for (FieldMaskSet<EquivalenceSet>::iterator it = 
            stale_sets.begin(); it != stale_sets.end(); it++)
      {
        // Skip any sets which didn't have valid fields
        if (!it->second)
          continue;
        RtUserEvent refresh_done = Runtime::create_rt_user_event();
        it->first->refresh_refinement(this, it->second, refresh_done);
        preconditions.insert(refresh_done);
        // Remove the reference that we are holding
        if (it->first->remove_base_resource_ref(VERSION_MANAGER_REF))
          delete it->first;
      }
      // Now launch a finalize task to complete the update 
      if (!preconditions.empty())
      {
        const RtEvent precondition = Runtime::merge_events(preconditions);
        if (precondition.exists() && !precondition.has_triggered())
        {
          LgFinalizeEqSetsArgs args(this, compute_event, implicit_provenance);
          runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY,
                                           precondition);
          return compute_event;
        }
      }
      // If we make it here we can do the finalize call now
      finalize_equivalence_sets(compute_event);
      return RtEvent::NO_RT_EVENT;
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

    //--------------------------------------------------------------------------
    /*static*/ void VersionManager::handle_stale_update(Deserializer &derez,
                                                        Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_sets;
      derez.deserialize(num_sets);
      FieldMaskSet<EquivalenceSet> stale_sets;
      std::set<RtEvent> ready_events;
      for (unsigned idx = 0; idx < num_sets; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        EquivalenceSet *set = 
          runtime->find_or_request_equivalence_set(did, ready);
        FieldMask mask;
        derez.deserialize(mask);
        stale_sets.insert(set, mask);
        if (ready.exists() && !ready.has_triggered())
          ready_events.insert(ready);
      }
      VersionManager *manager;
      derez.deserialize(manager);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }
      const RtEvent done = manager->record_stale_sets(stale_sets);
      Runtime::trigger_event(done_event, done);
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
    void InstanceRef::add_valid_reference(ReferenceSource source,
                                          ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      manager->add_base_valid_ref(source, mutator);
    }

    //--------------------------------------------------------------------------
    void InstanceRef::remove_valid_reference(ReferenceSource source,
                                             ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(manager != NULL);
#endif
      if (manager->remove_base_valid_ref(source, mutator))
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
    void InstanceSet::add_valid_references(ReferenceSource source,
                                           ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->add_valid_reference(source, mutator);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].add_valid_reference(source, mutator);
      }
    }

    //--------------------------------------------------------------------------
    void InstanceSet::remove_valid_references(ReferenceSource source,
                                              ReferenceMutator *mutator) const
    //--------------------------------------------------------------------------
    {
      if (single)
      {
        if (refs.single != NULL)
          refs.single->remove_valid_reference(source, mutator);
      }
      else
      {
        for (unsigned idx = 0; idx < refs.multi->vector.size(); idx++)
          refs.multi->vector[idx].remove_valid_reference(source, mutator);
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

    /////////////////////////////////////////////////////////////
    // VersioningInvalidator 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersioningInvalidator::VersioningInvalidator(void)
      : ctx(0), invalidate_all(true)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersioningInvalidator::VersioningInvalidator(RegionTreeContext c)
      : ctx(c.get_id()), invalidate_all(!c.exists())
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_region(RegionNode *node)
    //--------------------------------------------------------------------------
    {
      if (invalidate_all)
        node->invalidate_version_managers();
      else
        node->invalidate_version_state(ctx);
      return true;
    }

    //--------------------------------------------------------------------------
    bool VersioningInvalidator::visit_partition(PartitionNode *node)
    //--------------------------------------------------------------------------
    {
      // There is no version information on partitions
      return true;
    }

  }; // namespace Internal 
}; // namespace Legion

