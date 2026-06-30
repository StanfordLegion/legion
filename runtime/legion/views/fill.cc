/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/views/fill.h"
#include "legion/analysis/aggregator.h"
#include "legion/kernel/runtime.h"
#include "legion/instances/physical.h"
#include "legion/nodes/expression.h"
#include "legion/nodes/region.h"
#include "legion/utilities/collectives.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // FillView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FillView::FillView(
        DistributedID did, UniqueID op_uid, bool register_now,
        CollectiveMapping* map)
      : DeferredView(encode_fill_did(did), register_now, map),
        fill_op_uid(op_uid), value(nullptr), value_size(0),
        collective_first_active((map != nullptr) && map->contains(local_space))
    //--------------------------------------------------------------------------
    {
      // Add an extra reference here until we receive the value update
      add_base_resource_ref(PENDING_UNBOUND_REF);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Fill View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::FillView(
        DistributedID did, UniqueID op_uid, const void* val, size_t size,
        bool register_now, CollectiveMapping* map)
      : DeferredView(encode_fill_did(did), register_now, map),
        fill_op_uid(op_uid), value(malloc(size)), value_size(size),
        collective_first_active((map != nullptr) && map->contains(local_space))
    //--------------------------------------------------------------------------
    {
      legion_assert(value_size > 0);
      memcpy(value.load(), val, size);
#ifdef LEGION_GC
      log_garbage.info(
          "GC Fill View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    FillView::~FillView(void)
    //--------------------------------------------------------------------------
    {
      if (value.load() != nullptr)
        free(value.load());
    }

    //--------------------------------------------------------------------------
    void FillView::pack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      if (view_lamport_clocks.find(this) == view_lamport_clocks.end())
        pack_global_ref(view_lamport_clocks[this]);
    }

    //--------------------------------------------------------------------------
    void FillView::unpack_valid_ref(
        shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
        shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks)
    //--------------------------------------------------------------------------
    {
      shrt::map<LogicalView*, LamportClock>::iterator finder =
          view_lamport_clocks.find(this);
      if (finder == view_lamport_clocks.end())
        return;
      unpack_global_ref(finder->second);
      view_lamport_clocks.erase(finder);
    }

    //--------------------------------------------------------------------------
    void FillView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_owner());
      legion_assert(collective_mapping == nullptr);
      FillViewMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(fill_op_uid);
        AutoLock v_lock(view_lock);
        size_t size = value_size.load();
        rez.serialize<size_t>(size);
        if (size > 0)
          rez.serialize(value.load(), size);
        // Update the remote instances while holding the lock
        update_remote_instances(target);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void FillView::flatten(
        CopyFillAggregator& aggregator, InstanceView* dst_view,
        const FieldMask& src_mask, IndexSpaceExpression* expr,
        PredEvent pred_guard, const PhysicalTraceInfo& trace_info,
        EquivalenceSet* tracing_eq, CopyAcrossHelper* helper)
    //--------------------------------------------------------------------------
    {
      aggregator.record_fill(
          dst_view, this, src_mask, expr, pred_guard, tracing_eq, helper);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillViewMessage::handle(Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      UniqueID op_uid;
      derez.deserialize(op_uid);
      size_t value_size;
      derez.deserialize(value_size);

      void* location =
          runtime->find_or_create_pending_collectable_location<FillView>(did);
      FillView* view = nullptr;
      if (value_size > 0)
      {
        const void* value = derez.get_current_pointer();
        view = new (location)
            FillView(did, op_uid, value, value_size, false /*register now*/);
        derez.advance_pointer(value_size);
      }
      else
        view = new (location) FillView(did, op_uid, false /*register now*/);
      view->register_with_runtime();
    }

    //--------------------------------------------------------------------------
    bool FillView::matches(FillView* other)
    //--------------------------------------------------------------------------
    {
      if (value == nullptr)
      {
        RtEvent wait_on;
        {
          AutoLock v_lock(view_lock);
          if (value == nullptr)
          {
            value_ready = Runtime::create_rt_user_event();
            wait_on = value_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      legion_assert(value != nullptr);
      return other->matches(value, value_size);
    }

    //--------------------------------------------------------------------------
    bool FillView::matches(const void* other, size_t size)
    //--------------------------------------------------------------------------
    {
      if (value == nullptr)
      {
        RtEvent wait_on;
        {
          AutoLock v_lock(view_lock);
          if (value == nullptr)
          {
            value_ready = Runtime::create_rt_user_event();
            wait_on = value_ready;
          }
        }
        if (wait_on.exists())
          wait_on.wait();
      }
      legion_assert(value != nullptr);
      if (value_size != size)
        return false;
      return (memcmp(value, other, value_size) == 0);
    }

    //--------------------------------------------------------------------------
    bool FillView::set_value(const void* val, size_t size)
    //--------------------------------------------------------------------------
    {
      legion_assert(size > 0);
      legion_assert(val != nullptr);
      legion_assert(value.load() == nullptr);
      legion_assert(value_size.load() == 0);
      void* result = malloc(size);
      memcpy(result, val, size);
      // Take the lock and sent out any notifications
      AutoLock v_lock(view_lock);
      value_size.store(size);
      value.store(result);
      if (value_ready.exists())
        Runtime::trigger_event(value_ready);
      if (is_owner() && has_remote_instances())
      {
        FillViewValueMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(size);
          rez.serialize(val, size);
        }
        struct ValueFunctor {
          ValueFunctor(FillViewValueMessage& z) : rez(z) { }
          inline void apply(AddressSpaceID target)
          {
            if (target == runtime->address_space)
              return;
            rez.dispatch(target);
          }
          FillViewValueMessage& rez;
        };
        ValueFunctor functor(rez);
        map_over_remote_instances(functor);
      }
      return remove_base_resource_ref(PENDING_UNBOUND_REF);
    }

    //--------------------------------------------------------------------------
    /*static*/ void FillViewValueMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      size_t size;
      derez.deserialize(size);
      const void* value = derez.get_current_pointer();
      derez.advance_pointer(size);

      // This message can arrive out-of-order with respect to the creation
      // of the fill view on the remote node, so do the normal steps
      RtEvent ready;
      FillView* view = static_cast<FillView*>(
          runtime->find_or_request_logical_view(did, ready));
      if (ready.exists() && !ready.has_triggered())
        ready.wait();

      if (view->set_value(value, size))
        delete view;
    }

    //--------------------------------------------------------------------------
    ApEvent FillView::issue_fill(
        Operation* op, IndexSpaceExpression* fill_expr,
        IndividualView* dst_view, const FieldMask& fill_mask,
        const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& dst_fields,
        std::set<RtEvent>& applied_events, PhysicalManager* manager,
        ApEvent precondition, PredEvent pred_guard,
        CollectiveKind collective_kind, bool fill_restricted)
    //--------------------------------------------------------------------------
    {
      if (value_size.load() == 0)
      {
        // We don't know the value yet so we need to launch a task to
        // actually issue the fill once we know the value
        AutoLock v_lock(view_lock);
        if (value_size.load() == 0)
        {
          if (!value_ready.exists())
            value_ready = Runtime::create_rt_user_event();
          DeferIssueFill args(
              this, op, fill_expr, dst_view, fill_mask, trace_info, dst_fields,
              manager, precondition, pred_guard, collective_kind,
              fill_restricted, applied_events);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_DEFERRED_PRIORITY, value_ready);
          return args.done;
        }
      }
      // If we get here the that means we have a value and can issue
      // the fill from this fill view
      ApEvent result = fill_expr->issue_fill(
          op, trace_info, dst_fields, value.load(), value_size.load(),
          fill_op_uid, manager->field_space_node->handle, manager->tree_id,
          precondition, pred_guard, manager->get_unique_event(),
          collective_kind, fill_restricted);
      if (trace_info.recording)
      {
        const UniqueInst dst_inst(dst_view);
        trace_info.record_fill_inst(
            result, fill_expr, dst_inst, fill_mask, applied_events,
            (dst_view->get_redop() > 0));
      }
      return result;
    }

    //--------------------------------------------------------------------------
    FillView::DeferIssueFill::DeferIssueFill(
        FillView* v, Operation* o, IndexSpaceExpression* expr,
        IndividualView* dst_v, const FieldMask& mask,
        const PhysicalTraceInfo& info, const std::vector<CopySrcDstField>& dst,
        PhysicalManager* man, ApEvent pre, PredEvent guard,
        CollectiveKind collect, bool fill_restrict,
        std::set<RtEvent>& applied_events)
      : LgTaskArgs<DeferIssueFill>(false, false), view(v), op(o),
        fill_expr(expr), dst_view(dst_v),
        fill_mask(new HeapifyBox<FieldMask, OPERATION_LIFETIME>(mask)),
        trace_info(new PhysicalTraceInfo(info)),
        dst_fields(new std::vector<CopySrcDstField>(dst)), manager(man),
        precondition(pre), pred_guard(guard), collective(collect),
        applied(Runtime::create_rt_user_event()),
        done(Runtime::create_ap_user_event(&info)),
        fill_restricted(fill_restrict)
    //--------------------------------------------------------------------------
    {
      view->add_base_resource_ref(META_TASK_REF);
      dst_view->add_base_resource_ref(META_TASK_REF);
      fill_expr->add_base_expression_reference(META_TASK_REF);
      manager->add_base_resource_ref(META_TASK_REF);
      applied_events.insert(applied);
    }

    //--------------------------------------------------------------------------
    void FillView::DeferIssueFill::execute(void) const
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> applied_events;
      const ApEvent result = view->issue_fill(
          op, fill_expr, dst_view, *fill_mask, *trace_info, *dst_fields,
          applied_events, manager, precondition, pred_guard, collective,
          fill_restricted);
      Runtime::trigger_event(done, result, *trace_info, applied_events);
      Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      delete fill_mask;
      delete trace_info;
      delete dst_fields;
      if (view->remove_base_resource_ref(META_TASK_REF))
        delete view;
      if (dst_view->remove_base_resource_ref(META_TASK_REF))
        delete dst_view;
      if (fill_expr->remove_base_expression_reference(META_TASK_REF))
        delete fill_expr;
      if (manager->remove_base_resource_ref(META_TASK_REF))
        delete manager;
    }

  }  // namespace Internal
}  // namespace Legion
