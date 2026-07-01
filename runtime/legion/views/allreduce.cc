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

#include "legion/views/allreduce.h"
#include "legion/instances/physical.h"
#include "legion/nodes/field.h"
#include "legion/operations/remote.h"
#include "legion/views/fill.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // AllreduceView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AllreduceView::AllreduceView(
        DistributedID id, DistributedID ctx_did,
        const std::vector<IndividualView*>& views,
        const std::vector<DistributedID>& insts, bool register_now,
        CollectiveMapping* mapping, ReductionOpID redop_id)
      : CollectiveView(
            encode_allreduce_did(id), ctx_did, views, insts, register_now,
            mapping),
        redop(redop_id), reduction_op(runtime->get_reduction_op(redop)),
        fill_view(runtime->find_or_create_reduction_fill_view(redop)),
        unique_allreduce_tag(
            mapping->contains(local_space) ? mapping->find_index(local_space) :
                                             0),
        multi_instance(false), evaluated_multi_instance(false)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        legion_assert(local_views[idx]->get_redop() == redop);
      }
      fill_view->add_nested_resource_ref(did);
      // We reserve the 0 all-reduce tag to mean no-tag
      if (unique_allreduce_tag.load() == 0)
        unique_allreduce_tag.fetch_add(collective_mapping->size());
#ifdef LEGION_GC
      log_garbage.info(
          "GC Allreduce View %lld %d", LEGION_DISTRIBUTED_ID_FILTER(this->did),
          local_space);
#endif
    }

    //--------------------------------------------------------------------------
    AllreduceView::~AllreduceView(void)
    //--------------------------------------------------------------------------
    {
      if (fill_view->remove_nested_resource_ref(did))
        delete fill_view;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::send_view(AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      AllreduceViewMessage rez;
      {
        RezCheck z(rez);
        rez.serialize(did);
        rez.serialize(context_did);
        rez.serialize<size_t>(instances.size());
        rez.serialize(
            &instances.front(), instances.size() * sizeof(DistributedID));
        if (collective_mapping != nullptr)
          collective_mapping->pack(rez);
        else
          rez.serialize<size_t>(0);
        rez.serialize(redop);
      }
      rez.dispatch(target);
      update_remote_instances(target);
    }

    //--------------------------------------------------------------------------
    /*static*/ void AllreduceViewMessage::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did, ctx_did;
      derez.deserialize(did);
      derez.deserialize(ctx_did);
      size_t num_insts;
      derez.deserialize(num_insts);
      std::vector<DistributedID> instances(num_insts);
      derez.deserialize(&instances.front(), num_insts * sizeof(DistributedID));
      size_t num_spaces;
      derez.deserialize(num_spaces);
      CollectiveMapping* mapping = nullptr;
      if (num_spaces > 0)
      {
        mapping = new CollectiveMapping(derez, num_spaces);
        mapping->add_reference();
      }
      ReductionOpID redop;
      derez.deserialize(redop);
      void* location =
          runtime->find_or_create_pending_collectable_location<AllreduceView>(
              did);
      std::vector<IndividualView*> no_views;
      AllreduceView* view = new (location) AllreduceView(
          did, ctx_did, no_views, instances, false /*register now*/, mapping,
          redop);
      // Register only after construction
      view->register_with_runtime();
      if ((mapping != nullptr) && mapping->remove_reference())
        delete mapping;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::perform_collective_reduction(
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, ApEvent precondition,
        PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, Operation* op, const unsigned index,
        const FieldMask& copy_mask, const FieldMask& dst_mask,
        const DistributedID src_inst_did, const UniqueInst& dst_inst,
        const LgEvent dst_unique_event, const PhysicalTraceInfo& trace_info,
        const CollectiveKind collective_kind,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        ApUserEvent result, AddressSpaceID origin)
    //--------------------------------------------------------------------------
    {
      legion_assert(redop > 0);
      legion_assert(op != nullptr);
      legion_assert(result.exists());
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      unsigned target_index = 0;
      if (src_inst_did > 0)
      {
        target_index = std::numeric_limits<unsigned>::max();
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          if (local_views[idx]->get_manager()->did != src_inst_did)
            continue;
          target_index = idx;
          break;
        }
        legion_assert(target_index != std::numeric_limits<unsigned>::max());
      }
      IndividualView* local_view = local_views[target_index];
      PhysicalManager* local_manager = local_view->get_manager();
      // Get the dst_fields and reservations for performing the local reductions
      std::vector<CopySrcDstField> local_fields;
      local_manager->compute_copy_offsets(copy_mask, local_fields);

      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      // Get the precondition for performing reductions to one of our instances
      ApEvent reduce_pre;
      std::vector<Reservation> local_reservations;
      const UniqueID op_id = op->get_unique_op_id();
      if (!children.empty() || (instances.size() > 1))
      {
        // Compute the precondition for performing any reductions
        reduce_pre = local_view->find_copy_preconditions(
            false /*reading*/, redop, copy_mask, copy_expression, op_id, index,
            applied_events, trace_info);
        // If we're going to be doing reductions we need the reservations
        local_view->find_field_reservations(copy_mask, local_reservations);
        for (unsigned idx = 0; idx < local_fields.size(); idx++)
          local_fields[idx].set_redop(redop, true /*fold*/, true /*exclusive*/);
      }
      std::vector<ApEvent> reduce_events;
      // If we have any children, send them messages to reduce to our instance
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      const UniqueInst local_inst(local_view);
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveDistributeReduction rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          pack_fields(rez, local_fields, local_manager->get_unique_event());
          rez.serialize<size_t>(local_reservations.size());
          for (unsigned idx = 0; idx < local_reservations.size(); idx++)
            rez.serialize(local_reservations[idx]);
          rez.serialize(reduce_pre);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, child);
          rez.serialize(upper_bound);
          op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(dst_mask);
          rez.serialize<DistributedID>(0);  // no source point in this case
          local_inst.serialize(rez);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            if (!trace_barrier.exists())
            {
              trace_shard = trace_info.record_barrier_creation(
                  trace_barrier, children.size());
              reduce_events.emplace_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            rez.serialize(trace_shard);
          }
          else
          {
            const ApUserEvent reduced =
                Runtime::create_ap_user_event(&trace_info);
            rez.serialize(reduced);
            reduce_events.emplace_back(reduced);
          }
          rez.serialize(origin);
          rez.serialize(collective_kind);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      // Perform our local reductions
      if (local_views.size() > 1)
        reduce_local(
            local_manager, target_index, op, index, copy_expression,
            upper_bound, copy_mask, reduce_pre, predicate_guard, local_fields,
            local_reservations, local_inst, trace_info, collective_kind,
            reduce_events, applied_events, &recorded_events);
      if (!reduce_events.empty())
      {
        const ApEvent reduce_post =
            Runtime::merge_events(&trace_info, reduce_events);
        if (reduce_post.exists())
          local_view->add_copy_user(
              false /*reading*/, redop, reduce_post, copy_mask, copy_expression,
              upper_bound, op_id, index, recorded_events, trace_info.recording,
              runtime->address_space);
      }
      // Peform the reduction back to the destination
      const ApEvent read_pre = local_view->find_copy_preconditions(
          true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, trace_info);
      // Set the redops back to 0
      for (unsigned idx = 0; idx < local_fields.size(); idx++)
        local_fields[idx].set_redop(0, false /*fold*/);
      if (precondition.exists())
      {
        if (read_pre.exists())
          precondition =
              Runtime::merge_events(&trace_info, precondition, read_pre);
      }
      else
        precondition = read_pre;
      // Perform the reduction to the destination
      const ApEvent reduce_post = copy_expression->issue_copy(
          op, trace_info, dst_fields, local_fields, reservations,
          local_manager->tree_id, dst_inst.tid, precondition, predicate_guard,
          local_manager->get_unique_event(), dst_unique_event, collective_kind,
          false /*copy restricted*/);
      // Trigger the output
      Runtime::trigger_event(result, reduce_post, trace_info, applied_events);
      // Save the result, note that this reading of this final reduction
      // always dominates any incoming reductions so we don't need to
      // record them separately
      if (reduce_post.exists())
        local_view->add_copy_user(
            true /*reading*/, 0 /*redop*/, reduce_post, copy_mask,
            copy_expression, upper_bound, op_id, index, recorded_events,
            trace_info.recording, runtime->address_space);
      if (trace_info.recording)
        trace_info.record_copy_insts(
            reduce_post, copy_expression, local_inst, dst_inst, copy_mask,
            dst_mask, redop, applied_events);
    }

    //--------------------------------------------------------------------------
    void AllreduceView::reduce_local(
        const PhysicalManager* dst_manager, const unsigned dst_index,
        Operation* op, const unsigned index,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        const FieldMask& copy_mask, ApEvent precondition,
        PredEvent predicate_guard,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& dst_reservations,
        const UniqueInst& dst_inst, const PhysicalTraceInfo& trace_info,
        const CollectiveKind collective_kind,
        std::vector<ApEvent>& reduced_events, std::set<RtEvent>& applied_events,
        std::set<RtEvent>* recorded_events, const bool prepare_allreduce,
        std::vector<std::vector<CopySrcDstField> >* source_fields)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_views.empty());
      legion_assert(
          !prepare_allreduce || (reduced_events.size() == local_views.size()));
      legion_assert(prepare_allreduce == (source_fields != nullptr));
      legion_assert(
          !prepare_allreduce || (source_fields->size() == local_views.size()));
      legion_assert(prepare_allreduce == (recorded_events == nullptr));
      if (local_views.size() == 1)
        return;
      const UniqueID op_id = op->get_unique_op_id();
      if (multiple_local_memories)
      {
        // If there are multiple local instances on this node, then
        // we need to get a spanning tree to use for issuing the
        // broadcast copies across the local views, we use the
        // reversed order for performing a reduction tree
        const std::vector<std::pair<unsigned, unsigned> >& spanning_copies =
            find_spanning_broadcast_copies(dst_index);
        std::vector<bool> initialized(local_views.size(), false);
        unsigned reduced_events_offset = 0;
        if (!prepare_allreduce)
        {
          // Append onto the end of the reduced_events if there were already
          // events in the reduced_events vectors
          reduced_events_offset = reduced_events.size();
          reduced_events.resize(
              reduced_events_offset + local_views.size(), ApEvent::NO_AP_EVENT);
        }
        ApEvent* local_events = &reduced_events[reduced_events_offset];
        local_events[dst_index] = precondition;
        initialized[dst_index] = (source_fields != nullptr);
        std::vector<std::vector<CopySrcDstField> > fields(local_views.size());
        std::vector<std::vector<CopySrcDstField> >& local_fields =
            (source_fields != nullptr) ? *source_fields : fields;
        std::map<unsigned, std::vector<Reservation> > local_reservations;
        std::map<unsigned, std::vector<ApEvent> > reduction_preconditions;
        local_reservations[dst_index] = dst_reservations;
        // Note the reversed iterator <destination,source>
        for (std::vector<std::pair<unsigned, unsigned> >::const_reverse_iterator
                 it = spanning_copies.rbegin();
             it != spanning_copies.rend(); it++)
        {
          legion_assert(it->first != it->second);
          IndividualView* src_view = local_views[it->second];
          PhysicalManager* src_manager = src_view->get_manager();
          IndividualView* local_view = local_views[it->first];
          PhysicalManager* local_manager = local_view->get_manager();
          if (initialized[it->second])
          {
            // Save any reduction events into the view
            std::map<unsigned, std::vector<ApEvent> >::iterator finder =
                reduction_preconditions.find(it->second);
            legion_assert(it->second != dst_index);
            legion_assert(finder != reduction_preconditions.end());
            const ApEvent reduce_pre =
                Runtime::merge_events(&trace_info, finder->second);
            if (reduce_pre.exists())
              src_view->add_copy_user(
                  false /*reading*/, redop, reduce_pre, copy_mask,
                  copy_expression, upper_bound, op_id, index, *recorded_events,
                  trace_info.recording, runtime->address_space);
            reduction_preconditions.erase(finder);
          }
          else
            src_manager->compute_copy_offsets(
                copy_mask, local_fields[it->second]);
          ApEvent reduce_pre = src_view->find_copy_preconditions(
              !prepare_allreduce, 0 /*redop*/, copy_mask, copy_expression,
              op_id, index, applied_events, trace_info);
          if (!initialized[it->first])
          {
            // Initialize the destination
            local_events[it->first] = local_view->find_copy_preconditions(
                false /*reading*/, redop, copy_mask, copy_expression, op_id,
                index, applied_events, trace_info);
            local_manager->compute_copy_offsets(
                copy_mask, local_fields[it->first]);
            local_view->find_field_reservations(
                copy_mask, local_reservations[it->first]);
            initialized[it->first] = true;
          }
          if (local_events[it->first].exists())
          {
            if (reduce_pre.exists())
              reduce_pre = Runtime::merge_events(
                  &trace_info, reduce_pre, local_events[it->first]);
            else
              reduce_pre = local_events[it->first];
          }
          // Set the redop on dst fields
          set_redop(local_fields[it->first]);
          // Issue the copy
          local_events[it->second] = copy_expression->issue_copy(
              op, trace_info, local_fields[it->first], local_fields[it->second],
              local_reservations[it->first], local_manager->tree_id,
              src_manager->tree_id, reduce_pre, predicate_guard,
              src_manager->get_unique_event(),
              local_manager->get_unique_event(), collective_kind,
              false /*copy restricted*/);
          // Clear the redop in case we're reading them next
          clear_redop(local_fields[it->first]);
          // Save the state for later
          if (local_events[it->second].exists())
          {
            if (!prepare_allreduce)
              src_view->add_copy_user(
                  true /*reading*/, 0 /*redop*/, local_events[it->second],
                  copy_mask, copy_expression, upper_bound, op_id, index,
                  *recorded_events, trace_info.recording,
                  runtime->address_space);
            // Save it for a future reader
            reduction_preconditions[it->first].emplace_back(
                local_events[it->second]);
          }
          if (trace_info.recording)
          {
            const UniqueInst src_inst(src_view);
            const UniqueInst local_inst(local_view);
            trace_info.record_copy_insts(
                local_events[it->second], copy_expression, src_inst, local_inst,
                copy_mask, copy_mask, redop, applied_events);
          }
        }
        // Aggregate all the remaining reductions into the target
        legion_assert(reduction_preconditions.size() < 2);
        if (reduction_preconditions.empty())
        {
          // All the copies have already run so there are no
          // preconditions left for us here
          if (!prepare_allreduce)
            reduced_events.resize(reduced_events_offset);
        }
        else
        {
          std::map<unsigned, std::vector<ApEvent> >::iterator finder =
              reduction_preconditions.find(dst_index);
          legion_assert(finder != reduction_preconditions.end());
          local_events[dst_index] =
              Runtime::merge_events(&trace_info, finder->second);
          reduction_preconditions.erase(finder);
        }
      }
      else
      {
        // If all the local instances are in the same memory then we
        // might as well just issue copies from all the destinations to
        // the source the source since they'll all be fighting over the same
        // bandwidth anyway for the copies to be performed
        for (unsigned idx = 0; idx < local_views.size(); idx++)
        {
          if (idx == dst_index)
            continue;
          std::vector<CopySrcDstField> local_fields;
          std::vector<CopySrcDstField>& src_fields =
              prepare_allreduce ? (*source_fields)[idx] : local_fields;
          IndividualView* src_view = local_views[idx];
          PhysicalManager* src_manager = src_view->get_manager();
          src_manager->compute_copy_offsets(copy_mask, src_fields);
          // Technically we're reading here, but if we're going to be "writing"
          // the allreduce result then pretend like we're writing
          ApEvent reduce_pre = src_view->find_copy_preconditions(
              !prepare_allreduce /*reading*/, 0 /*redop*/, copy_mask,
              copy_expression, op_id, index, applied_events, trace_info);
          if (precondition.exists())
          {
            if (reduce_pre.exists())
              reduce_pre =
                  Runtime::merge_events(&trace_info, precondition, reduce_pre);
            else
              reduce_pre = precondition;
          }
          const ApEvent reduce_post = copy_expression->issue_copy(
              op, trace_info, dst_fields, src_fields, dst_reservations,
              dst_manager->tree_id, src_manager->tree_id, reduce_pre,
              predicate_guard, src_manager->get_unique_event(),
              dst_manager->get_unique_event(), collective_kind,
              false /*copy restricted*/);
          if (reduce_post.exists())
          {
            if (!prepare_allreduce)
            {
              reduced_events.emplace_back(reduce_post);
              src_view->add_copy_user(
                  true /*reading*/, 0 /*redop*/, reduce_post, copy_mask,
                  copy_expression, upper_bound, op_id, index, *recorded_events,
                  trace_info.recording, runtime->address_space);
            }
            else
              reduced_events[idx] = reduce_post;
          }
          if (trace_info.recording)
          {
            const UniqueInst src_inst(src_view);
            trace_info.record_copy_insts(
                reduce_post, copy_expression, src_inst, dst_inst, copy_mask,
                copy_mask, redop, applied_events);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    uint64_t AllreduceView::generate_unique_allreduce_tag(void)
    //--------------------------------------------------------------------------
    {
      // We should always be calling this one of the original collective
      // nodes for the allreduce view at the moment
      legion_assert(collective_mapping->contains(local_space));
      return unique_allreduce_tag.fetch_add(collective_mapping->size());
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeReduction::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent view_ready;
      AllreduceView* view = static_cast<AllreduceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      std::vector<CopySrcDstField> dst_fields;
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      LgEvent dst_unique_event = AllreduceView::unpack_fields(
          dst_fields, derez, ready_events, view, view_ready);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexSpace upper_bound;
      derez.deserialize(upper_bound);
      Operation* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      DistributedID src_inst_did;
      derez.deserialize(src_inst_did);
      UniqueInst dst_inst;
      dst_inst.deserialize(derez);
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        ShardID sid;
        derez.deserialize(sid);
        // Copy-elmination will take care of this for us
        // when the trace is optimized
        ready = Runtime::create_ap_user_event(&trace_info);
        runtime->phase_barrier_arrive(bar, 1 /*count*/, ready);
        trace_info.record_barrier_arrival(
            bar, ready, 1 /*count*/, applied_events, sid);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);
      CollectiveKind collective_kind;
      derez.deserialize(collective_kind);

      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      view->perform_collective_reduction(
          dst_fields, reservations, precondition, predicate_guard,
          copy_expression, upper_bound, op, index, copy_mask, dst_mask,
          src_inst_did, dst_inst, dst_unique_event, trace_info, collective_kind,
          recorded_events, applied_events, ready, origin);

      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    ApEvent AllreduceView::perform_hammer_reduction(
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, ApEvent precondition,
        PredEvent predicate_guard, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, Operation* op, const unsigned index,
        const FieldMask& copy_mask, const FieldMask& dst_mask,
        const UniqueInst& dst_inst, const LgEvent dst_unique_event,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, AddressSpaceID origin)
    //--------------------------------------------------------------------------
    {
      legion_assert(redop > 0);
      legion_assert(op != nullptr);
      legion_assert(!local_views.empty());
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      // Distribute out to the other nodes first
      std::vector<ApEvent> done_events;
      std::vector<AddressSpaceID> children;
      collective_mapping->get_children(origin, local_space, children);
      ApBarrier trace_barrier;
      ShardID trace_shard = 0;
      for (const AddressSpaceID& child : children)
      {
        const RtUserEvent recorded = Runtime::create_rt_user_event();
        const RtUserEvent applied = Runtime::create_rt_user_event();
        CollectiveHammerReduction rez;
        {
          RezCheck z(rez);
          rez.serialize(this->did);
          pack_fields(rez, dst_fields, dst_unique_event);
          rez.serialize<size_t>(reservations.size());
          for (unsigned idx = 0; idx < reservations.size(); idx++)
            rez.serialize(reservations[idx]);
          rez.serialize(precondition);
          rez.serialize(predicate_guard);
          copy_expression->pack_expression(rez, child);
          rez.serialize(upper_bound);
          op->pack_remote_operation(rez, child, applied_events);
          rez.serialize(index);
          rez.serialize(copy_mask);
          rez.serialize(dst_mask);
          dst_inst.serialize(rez);
          trace_info.pack_trace_info(rez);
          rez.serialize(recorded);
          rez.serialize(applied);
          if (trace_info.recording)
          {
            if (!trace_barrier.exists())
            {
              trace_shard = trace_info.record_barrier_creation(
                  trace_barrier, children.size());
              done_events.emplace_back(trace_barrier);
            }
            rez.serialize(trace_barrier);
            rez.serialize(trace_shard);
          }
          else
          {
            const ApUserEvent done = Runtime::create_ap_user_event(&trace_info);
            rez.serialize(done);
            done_events.emplace_back(done);
          }
          rez.serialize(origin);
        }
        rez.dispatch(child);
        recorded_events.insert(recorded);
        applied_events.insert(applied);
      }
      const UniqueID op_id = op->get_unique_op_id();
      // Issue the copies
      for (unsigned idx = 0; idx < local_views.size(); idx++)
      {
        IndividualView* local_view = local_views[idx];
        ApEvent src_pre = local_view->find_copy_preconditions(
            true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        if (src_pre.exists())
        {
          if (precondition.exists())
            src_pre = Runtime::merge_events(&trace_info, precondition, src_pre);
        }
        else
          src_pre = precondition;
        PhysicalManager* local_manager = local_view->get_manager();
        std::vector<CopySrcDstField> src_fields;
        local_manager->compute_copy_offsets(copy_mask, src_fields);
        const ApEvent copy_post = copy_expression->issue_copy(
            op, trace_info, dst_fields, src_fields, reservations,
            local_manager->tree_id, dst_inst.tid, src_pre, predicate_guard,
            local_manager->get_unique_event(), dst_unique_event,
            COLLECTIVE_HAMMER_REDUCTION, false /*copy restricted*/);
        if (copy_post.exists())
        {
          done_events.emplace_back(copy_post);
          local_view->add_copy_user(
              true /*reading*/, 0 /*redop*/, copy_post, copy_mask,
              copy_expression, upper_bound, op_id, index, recorded_events,
              trace_info.recording, runtime->address_space);
        }
        if (trace_info.recording)
        {
          const UniqueInst src_inst(local_view);
          trace_info.record_copy_insts(
              copy_post, copy_expression, src_inst, dst_inst, copy_mask,
              dst_mask, redop, applied_events);
        }
      }
      // Merge the done events together
      if (done_events.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(&trace_info, done_events);
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveHammerReduction::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID view_did;
      derez.deserialize(view_did);
      RtEvent view_ready;
      AllreduceView* view = static_cast<AllreduceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      std::vector<CopySrcDstField> dst_fields;
      std::set<RtEvent> recorded_events, ready_events, applied_events;
      LgEvent dst_unique_event = AllreduceView::unpack_fields(
          dst_fields, derez, ready_events, view, view_ready);
      size_t num_reservations;
      derez.deserialize(num_reservations);
      std::vector<Reservation> reservations(num_reservations);
      for (unsigned idx = 0; idx < num_reservations; idx++)
        derez.deserialize(reservations[idx]);
      ApEvent precondition;
      derez.deserialize(precondition);
      PredEvent predicate_guard;
      derez.deserialize(predicate_guard);
      IndexSpaceExpression* copy_expression =
          IndexSpaceExpression::unpack_expression(derez, source);
      IndexSpace upper_bound;
      derez.deserialize(upper_bound);
      Operation* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      FieldMask copy_mask, dst_mask;
      derez.deserialize(copy_mask);
      derez.deserialize(dst_mask);
      UniqueInst dst_inst;
      dst_inst.deserialize(derez);
      PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      RtUserEvent recorded, applied;
      derez.deserialize(recorded);
      derez.deserialize(applied);
      ApUserEvent ready;
      if (trace_info.recording)
      {
        ApBarrier bar;
        derez.deserialize(bar);
        ShardID sid;
        derez.deserialize(sid);
        ready = Runtime::create_ap_user_event(&trace_info);
        runtime->phase_barrier_arrive(bar, 1 /*count*/, ready);
        trace_info.record_barrier_arrival(
            bar, ready, 1 /*count*/, applied_events, sid);
      }
      else
        derez.deserialize(ready);
      AddressSpaceID origin;
      derez.deserialize(origin);

      if (view_ready.exists() && !view_ready.has_triggered())
        ready_events.insert(view_ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      const ApEvent result = view->perform_hammer_reduction(
          dst_fields, reservations, precondition, predicate_guard,
          copy_expression, upper_bound, op, index, copy_mask, dst_mask,
          dst_inst, dst_unique_event, trace_info, recorded_events,
          applied_events, origin);

      Runtime::trigger_event(ready, result, trace_info, applied_events);
      if (!recorded_events.empty())
        Runtime::trigger_event(
            recorded, Runtime::merge_events(recorded_events));
      else
        Runtime::trigger_event(recorded);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      delete op;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::perform_collective_allreduce(
        ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        Operation* op, const unsigned index, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events, const uint64_t allreduce_tag)
    //--------------------------------------------------------------------------
    {
      legion_assert(redop > 0);
      legion_assert(op != nullptr);
      legion_assert(collective_mapping != nullptr);
      legion_assert(collective_mapping->contains(local_space));
      // We're guaranteed to get one call to this function for each space
      // in the collective mapping from perform_collective_pointwise, so
      // we've already distributed control
      // Our job in this function is to build a butterfly all-reduce network
      // for exchanging the reduction data so each reduction instance in this
      // collective instance contains all the same data
      // There is a major complicating factor here because we can't do a
      // natural in-place all-reduce across our instances since the finish
      // event for Realm copies only says when the whole copy is done and not
      // when the copy has finished reading out from the source instance.
      // Furthermore, we can't control when the reductions into the destination
      // instances start happening as they precondition just governs the start
      // of the whole copy. Therefore, we need to fake an in-place all-reduce.
      // We fake things in one of two ways:
      // Case 1: If we know that each node has at least two instances, then
      //         we can use one instance as the source for outgoing reduction
      //         copies and the other as the destination for incoming
      //         reduction copies and ping pong between them.
      // Case 2: If we don't have at least two instances on each node then
      //         we will pair up nodes and have them do the same trick as in
      //         case 1 but using the two instances on adjacent nodes as the
      //         sources and destinations.
      // We handle unusual numbers of nodes that are not a power of the
      // collective radix in the normal way by picking a number of participants
      // that is the largest power of the radix still less than or equal to
      // the number of nodes and using an extra stage to fold-in the
      // non-participants values before doing the butterfly.

      // See if we've got to do the multi-node all-reduce
      if (collective_mapping->size() > 1)
      {
        if (is_multi_instance())
          // Case 1: each node has multiple instances
          perform_multi_allreduce(
              allreduce_tag, op, index, precondition, predicate_guard,
              copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, applied_events);
        else
          // Case 2: there are some nodes that only have one instance
          // Pair up nodes to have them cooperate to have two buffers
          // that we can ping-pong between to do the all-reduce "inplace"
          perform_single_allreduce(
              allreduce_tag, op, index, precondition, predicate_guard,
              copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, applied_events);
      }
      else
      {
        // Everything is local so this is easy
        std::vector<std::vector<CopySrcDstField> > local_fields(
            local_views.size());
        std::vector<std::vector<Reservation> > reservations(local_views.size());
        std::vector<ApEvent> instance_events(local_views.size());
        initialize_allreduce_with_reductions(
            precondition, predicate_guard, op, index, copy_expression,
            upper_bound, copy_mask, trace_info, applied_events, instance_events,
            local_fields, reservations);
        complete_initialize_allreduce_with_reductions(
            op, index, copy_expression, upper_bound, copy_mask, trace_info,
            recorded_events, applied_events, instance_events, local_fields);
        finalize_allreduce_with_broadcasts(
            predicate_guard, op, index, copy_expression, upper_bound, copy_mask,
            trace_info, recorded_events, applied_events, instance_events,
            local_fields);
        complete_finalize_allreduce_with_broadcasts(
            op, index, copy_expression, upper_bound, copy_mask, trace_info,
            recorded_events, instance_events);
      }
    }

    //--------------------------------------------------------------------------
    bool AllreduceView::is_multi_instance(void)
    //--------------------------------------------------------------------------
    {
      if (evaluated_multi_instance.load())
        return multi_instance.load();
      bool result = false;
      // Must have at least twice as many collective instances as nodes
      // in order for this to qualify as multi instance
      if (instances.size() >= (2 * collective_mapping->size()))
      {
        // Check that there is at least two instances on every node
        std::vector<unsigned> counts(collective_mapping->size(), 0);
        for (const DistributedID& it : instances)
        {
          const AddressSpaceID owner = runtime->determine_owner(it);
          legion_assert(collective_mapping->contains(owner));
          const unsigned index = collective_mapping->find_index(owner);
          counts[index]++;
        }
        result = true;
        for (unsigned idx = 0; idx < counts.size(); idx++)
        {
          if (counts[idx] > 1)
            continue;
          result = false;
          break;
        }
      }
      multi_instance.store(result);
      evaluated_multi_instance.store(true);
      return result;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::perform_single_allreduce(
        const uint64_t allreduce_tag, Operation* op, unsigned index,
        ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(!multi_instance);
      // Case 2: there are some nodes that only have one instance
      // Pair up nodes to have them cooperate to have two buffers
      // that we can ping-pong between to do the all-reduce "inplace"
      const int participants = collective_mapping->size() / 2;  // truncate
      const int local_index = collective_mapping->find_index(local_space);
      const int local_rank = local_index / 2;
      const int local_offset = local_index % 2;
      int collective_radix = runtime->legion_collective_radix;
      int collective_log_radix, collective_stages;
      int participating_ranks, collective_last_radix;
      const bool participating = configure_collective_settings(
          participants, local_rank, collective_radix, collective_log_radix,
          collective_stages, participating_ranks, collective_last_radix);
      std::vector<std::vector<CopySrcDstField> > local_fields(
          local_views.size());
      std::vector<std::vector<Reservation> > reservations(local_views.size());
      std::vector<ApEvent> instance_events(local_views.size());
      if (participating)
      {
        // Check to see if we need to handle stage -1 from non-participants
        // As well as from offset=1 down to offset=0
        if (local_offset == 0)
        {
          const ApEvent reduce_pre = initialize_allreduce_with_reductions(
              precondition, predicate_guard, op, index, copy_expression,
              upper_bound, copy_mask, trace_info, applied_events,
              instance_events, local_fields, reservations);
          // We definitely will be expecting our partner
          std::vector<int> expected_ranks(1, local_rank);
          // We could be expecting up to two non-participants
          // User their index instead of rank to avoid key collision
          const int nonpart_index = local_index + 2 * participating_ranks;
          for (int offset = 0; offset < 2; offset++)
          {
            const int rank = nonpart_index + offset;
            if (rank >= int(collective_mapping->size()))
              break;
            expected_ranks.emplace_back(rank);
          }
          std::vector<ApEvent> reduce_events;
          receive_allreduce_stage(
              0 /*index*/, allreduce_tag, -1 /*stage*/, op, reduce_pre,
              predicate_guard, copy_expression, copy_mask, trace_info,
              applied_events, local_fields[0], reservations[0],
              &expected_ranks.front(), expected_ranks.size(), reduce_events);
          complete_initialize_allreduce_with_reductions(
              op, index, copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, applied_events, instance_events, local_fields,
              &reduce_events);
        }
        else
        {
          // local_offset == 1
          initialize_allreduce_without_reductions(
              precondition, predicate_guard, op, index, copy_expression,
              upper_bound, copy_mask, trace_info, recorded_events,
              applied_events, instance_events, local_fields, reservations);
          // Just need to send the reduction down to our partner
          const AddressSpaceID target = (*collective_mapping)[local_index - 1];
          std::vector<ApEvent> read_events;
          send_allreduce_stage(
              allreduce_tag, -1 /*stage*/, local_rank, instance_events[0],
              predicate_guard, copy_expression, trace_info, local_fields[0],
              0 /*src index*/, &target, 1 /*target count*/, read_events);
          if (!read_events.empty())
          {
            legion_assert(read_events.size() == 1);
            instance_events[0] = read_events[0];
          }
        }
        // Do the stages
        for (int stage = 0; stage < collective_stages; stage++)
        {
          // Figure out the participating ranks
          std::vector<int> stage_ranks;
          if (stage < (collective_stages - 1))
          {
            // Normal radix
            stage_ranks.reserve(collective_radix);
            for (int r = 1; r < collective_radix; r++)
            {
              int target = local_rank ^ (r << (stage * collective_log_radix));
              stage_ranks.emplace_back(target);
            }
          }
          else
          {
            // Last stage so special radix
            stage_ranks.reserve(collective_last_radix);
            for (int r = 1; r < collective_last_radix; r++)
            {
              int target = local_rank ^ (r << (stage * collective_log_radix));
              stage_ranks.emplace_back(target);
            }
          }
          legion_assert(!stage_ranks.empty());
          // Always include ourselves in the ranks as well
          stage_ranks.emplace_back(local_rank);
          // Check to see if we're sending or receiving this stage
          if ((stage % 2) == local_offset)
          {
            // We're doing a sending stage
            std::vector<AddressSpaceID> targets(stage_ranks.size());
            for (unsigned idx = 0; idx < stage_ranks.size(); idx++)
            {
              // If we're even, send to the odd
              // If we're odd, send to the even
              const unsigned index =
                  2 * stage_ranks[idx] + ((local_offset == 0) ? 1 : 0);
              legion_assert(index < collective_mapping->size());
              targets[idx] = (*collective_mapping)[index];
            }
            std::vector<ApEvent> read_events;
            send_allreduce_stage(
                allreduce_tag, stage, local_rank, instance_events[0],
                predicate_guard, copy_expression, trace_info, local_fields[0],
                0 /*src index*/, &targets.front(), targets.size(), read_events);
            if (!read_events.empty())
              instance_events[0] =
                  Runtime::merge_events(&trace_info, read_events);
          }
          else
          {
            // We're doing a receiving stage
            // First issue a fill to initialize the instance
            // Realm should ignore the redop data on these fields
            instance_events[0] = copy_expression->issue_fill(
                op, trace_info, local_fields[0], reduction_op->identity,
                reduction_op->sizeof_rhs, fill_view->fill_op_uid,
                local_views[0]->manager->field_space_node->handle,
                local_views[0]->manager->tree_id, instance_events[0],
                predicate_guard, local_views[0]->manager->get_unique_event(),
                COLLECTIVE_BUTTERFLY_ALLREDUCE, false /*restricted*/);
            if (trace_info.recording)
            {
              const UniqueInst dst_inst(local_views[0]);
              trace_info.record_fill_inst(
                  instance_events[0], copy_expression, dst_inst, copy_mask,
                  applied_events, (redop > 0));
            }
            // Then check to see if we've received any reductions
            std::vector<ApEvent> reduce_events;
            set_redop(local_fields[0]);
            receive_allreduce_stage(
                0 /*index*/, allreduce_tag, stage, op, instance_events[0],
                predicate_guard, copy_expression, copy_mask, trace_info,
                applied_events, local_fields[0], reservations[0],
                &stage_ranks.front(), stage_ranks.size(), reduce_events);
            clear_redop(local_fields[0]);
            if (!reduce_events.empty())
              instance_events[0] =
                  Runtime::merge_events(&trace_info, reduce_events);
          }
        }
        // If we have to do stage -1 then we can do that now
        // Check to see if we have the valid data or not
        if ((collective_stages % 2) == local_offset)
        {
          const ApEvent broadcast_pre = finalize_allreduce_with_broadcasts(
              predicate_guard, op, index, copy_expression, upper_bound,
              copy_mask, trace_info, recorded_events, applied_events,
              instance_events, local_fields);
          // We have the valid data, send it to up to two
          // non-participants as well as our partner
          // If we're odd then make us even and vice-versa
          int partner_index = local_index + ((local_offset == 0) ? 1 : -1);
          const AddressSpaceID partner = (*collective_mapping)[partner_index];
          std::vector<AddressSpaceID> targets(1, partner);
          // Check for the two non-participants
          const unsigned offset = 2 * participating_ranks;
          const unsigned one = offset + local_index;
          if (one < collective_mapping->size())
            targets.emplace_back((*collective_mapping)[one]);
          const unsigned two = offset + partner_index;
          if (two < collective_mapping->size())
            targets.emplace_back((*collective_mapping)[two]);
          std::vector<ApEvent> read_events;
          send_allreduce_stage(
              allreduce_tag, -2 /*stage*/, local_rank, broadcast_pre,
              predicate_guard, copy_expression, trace_info, local_fields[0],
              0 /*src index*/, &targets.front(), targets.size(), read_events);
          complete_finalize_allreduce_with_broadcasts(
              op, index, copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, instance_events, &read_events);
        }
        else
        {
          // Not reducing here, just standard copy
          // See if we received the copy from our partner
          std::vector<ApEvent> reduce_events;
          // No reservations since this is a straight copy
          const std::vector<Reservation> no_reservations;
          receive_allreduce_stage(
              0 /*index*/, allreduce_tag, -2 /*stage*/, op, instance_events[0],
              predicate_guard, copy_expression, copy_mask, trace_info,
              applied_events, local_fields[0], no_reservations, &local_rank,
              1 /*total ranks*/, reduce_events);
          if (!reduce_events.empty())
          {
            legion_assert(reduce_events.size() == 1);
            instance_events[0] = reduce_events[0];
          }
          finalize_allreduce_without_broadcasts(
              predicate_guard, op, index, copy_expression, upper_bound,
              copy_mask, trace_info, recorded_events, applied_events,
              instance_events, local_fields);
        }
      }
      else
      {
        // Not a participant in the stages, so just need to do
        // the stage -1 send and receive
        initialize_allreduce_without_reductions(
            precondition, predicate_guard, op, index, copy_expression,
            upper_bound, copy_mask, trace_info, recorded_events, applied_events,
            instance_events, local_fields, reservations);
        // Truncate down
        const int target_rank = (local_index - 2 * participating_ranks) / 2;
        legion_assert(target_rank >= 0);
        // Then convert back to the appropriate index
        const int target_index = 2 * target_rank;
        legion_assert(target_index < int(collective_mapping->size()));
        const AddressSpaceID target = (*collective_mapping)[target_index];
        std::vector<ApEvent> read_events;
        // Intentionally use the local_index here to avoid key collisions
        send_allreduce_stage(
            allreduce_tag, -1 /*stage*/, local_index, instance_events[0],
            predicate_guard, copy_expression, trace_info, local_fields[0],
            0 /*src index*/, &target, 1 /*total targets*/, read_events);
        if (!read_events.empty())
        {
          legion_assert(read_events.size() == 1);
          instance_events[0] = read_events[0];
        }
        // Check to see if we received the copy back yet
        // Keep the redop data zeroed out since we're doing normal copies
        // No reservations since this is a straight copy
        const std::vector<Reservation> no_reservations;
        std::vector<ApEvent> reduce_events;
        receive_allreduce_stage(
            0 /*index*/, allreduce_tag, -2 /*stage*/, op, instance_events[0],
            predicate_guard, copy_expression, copy_mask, trace_info,
            applied_events, local_fields[0], no_reservations, &target_rank,
            1 /*total ranks*/, reduce_events);
        if (!reduce_events.empty())
        {
          legion_assert(reduce_events.size() == 1);
          instance_events[0] = reduce_events[0];
        }
        finalize_allreduce_without_broadcasts(
            predicate_guard, op, index, copy_expression, upper_bound, copy_mask,
            trace_info, recorded_events, applied_events, instance_events,
            local_fields);
      }
    }

    //--------------------------------------------------------------------------
    void AllreduceView::perform_multi_allreduce(
        const uint64_t allreduce_tag, Operation* op, unsigned index,
        ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      // Case 1: each node has multiple instances
      legion_assert(redop > 0);
      legion_assert(multi_instance);
      legion_assert(instances.size() > 1);
      const int participants = collective_mapping->size();
      const int local_rank = collective_mapping->find_index(local_space);
      int collective_radix = runtime->legion_collective_radix;
      int collective_log_radix, collective_stages;
      int participating_ranks, collective_last_radix;
      const bool participating = configure_collective_settings(
          participants, local_rank, collective_radix, collective_log_radix,
          collective_stages, participating_ranks, collective_last_radix);
      std::vector<std::vector<CopySrcDstField> > local_fields(
          local_views.size());
      std::vector<std::vector<Reservation> > reservations(local_views.size());
      std::vector<ApEvent> instance_events(local_views.size());
      if (participating)
      {
        // Check to see if we need to wait for a remainder copy
        // for any non-participating ranks
        int remainder_rank = local_rank + participating_ranks;
        if (collective_mapping->size() <= size_t(remainder_rank))
          remainder_rank = -1;
        if (remainder_rank >= 0)
        {
          const ApEvent reduce_pre = initialize_allreduce_with_reductions(
              precondition, predicate_guard, op, index, copy_expression,
              upper_bound, copy_mask, trace_info, applied_events,
              instance_events, local_fields, reservations);
          std::vector<ApEvent> reduce_events;
          receive_allreduce_stage(
              0 /*index*/, allreduce_tag, -1 /*stage*/, op, reduce_pre,
              predicate_guard, copy_expression, copy_mask, trace_info,
              applied_events, local_fields[0], reservations[0], &remainder_rank,
              1 /*total ranks*/, reduce_events);
          complete_initialize_allreduce_with_reductions(
              op, index, copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, applied_events, instance_events, local_fields,
              &reduce_events);
        }
        else
          initialize_allreduce_without_reductions(
              precondition, predicate_guard, op, index, copy_expression,
              upper_bound, copy_mask, trace_info, recorded_events,
              applied_events, instance_events, local_fields, reservations);
        unsigned src_inst_index = 0;
        unsigned dst_inst_index = 1;
        // Issue the stages
        for (int stage = 0; stage < collective_stages; stage++)
        {
          // Figure out where to send out messages first
          std::vector<int> stage_ranks;
          if (stage < (collective_stages - 1))
          {
            // Normal radix
            stage_ranks.reserve(collective_radix - 1);
            for (int r = 1; r < collective_radix; r++)
            {
              int target = local_rank ^ (r << (stage * collective_log_radix));
              stage_ranks.emplace_back(target);
            }
          }
          else
          {
            // Last stage so special radix
            stage_ranks.reserve(collective_last_radix - 1);
            for (int r = 1; r < collective_last_radix; r++)
            {
              int target = local_rank ^ (r << (stage * collective_log_radix));
              stage_ranks.emplace_back(target);
            }
          }
          legion_assert(!stage_ranks.empty());
          // Send out the messages to the dst ranks to perform copies
          std::vector<AddressSpaceID> targets(stage_ranks.size());
          for (unsigned idx = 0; idx < stage_ranks.size(); idx++)
            targets[idx] = (*collective_mapping)[stage_ranks[idx]];
          std::vector<ApEvent> src_events;
          send_allreduce_stage(
              allreduce_tag, stage, local_rank, instance_events[src_inst_index],
              predicate_guard, copy_expression, trace_info,
              local_fields[src_inst_index], src_inst_index, &targets.front(),
              targets.size(), src_events);
          // Issuse the fill for the destination instance
          // Realm should ignore the redop data on these fields
          instance_events[dst_inst_index] = copy_expression->issue_fill(
              op, trace_info, local_fields[dst_inst_index],
              reduction_op->identity, reduction_op->sizeof_rhs,
              fill_view->fill_op_uid,
              local_views[dst_inst_index]->manager->field_space_node->handle,
              local_views[dst_inst_index]->manager->tree_id,
              instance_events[dst_inst_index], predicate_guard,
              local_views[dst_inst_index]->manager->get_unique_event(),
              COLLECTIVE_BUTTERFLY_ALLREDUCE, false /*restricted*/);
          if (trace_info.recording)
          {
            const UniqueInst dst_inst(local_views[dst_inst_index]);
            trace_info.record_fill_inst(
                instance_events[dst_inst_index], copy_expression, dst_inst,
                copy_mask, applied_events, true /*reduction*/);
          }
          set_redop(local_fields[dst_inst_index]);
          // Issue the reduction from the source to the destination
          ApEvent local_precondition = Runtime::merge_events(
              &trace_info, instance_events[src_inst_index],
              instance_events[dst_inst_index]);
          const ApEvent local_post = copy_expression->issue_copy(
              op, trace_info, local_fields[dst_inst_index],
              local_fields[src_inst_index], reservations[dst_inst_index],
              local_views[src_inst_index]->manager->tree_id,
              local_views[dst_inst_index]->manager->tree_id, local_precondition,
              predicate_guard,
              local_views[src_inst_index]->manager->get_unique_event(),
              local_views[dst_inst_index]->manager->get_unique_event(),
              COLLECTIVE_BUTTERFLY_ALLREDUCE, false /*copy restricted*/);
          std::vector<ApEvent> dst_events;
          if (local_post.exists())
          {
            src_events.emplace_back(local_post);
            dst_events.emplace_back(local_post);
          }
          if (trace_info.recording)
          {
            const UniqueInst src_inst(local_views[src_inst_index]);
            const UniqueInst dst_inst(local_views[dst_inst_index]);
            trace_info.record_copy_insts(
                local_post, copy_expression, src_inst, dst_inst, copy_mask,
                copy_mask, redop, applied_events);
          }
          // Update the source instance precondition
          // to reflect all the reduction copies read from it
          if (!src_events.empty())
            instance_events[src_inst_index] =
                Runtime::merge_events(&trace_info, src_events);
          // Now check to see if we're received any messages
          // for this stage, and if not make place holders for them
          receive_allreduce_stage(
              dst_inst_index, allreduce_tag, stage, op,
              instance_events[dst_inst_index], predicate_guard, copy_expression,
              copy_mask, trace_info, applied_events,
              local_fields[dst_inst_index], reservations[dst_inst_index],
              &stage_ranks.front(), stage_ranks.size(), dst_events);
          clear_redop(local_fields[dst_inst_index]);
          if (!dst_events.empty())
            instance_events[dst_inst_index] =
                Runtime::merge_events(&trace_info, dst_events);
          // Update the src and dst instances for the next stage
          if (++src_inst_index == local_views.size())
            src_inst_index = 0;
          if (++dst_inst_index == local_views.size())
            dst_inst_index = 0;
        }
        // Send out the result to any non-participating ranks
        if (remainder_rank >= 0)
        {
          const ApEvent broadcast_pre = finalize_allreduce_with_broadcasts(
              predicate_guard, op, index, copy_expression, upper_bound,
              copy_mask, trace_info, recorded_events, applied_events,
              instance_events, local_fields, src_inst_index);
          std::vector<ApEvent> broadcast_events;
          const AddressSpaceID target = (*collective_mapping)[remainder_rank];
          send_allreduce_stage(
              allreduce_tag, -1 /*stage*/, local_rank, broadcast_pre,
              predicate_guard, copy_expression, trace_info,
              local_fields[src_inst_index], src_inst_index, &target,
              1 /*total targets*/, broadcast_events);
          complete_finalize_allreduce_with_broadcasts(
              op, index, copy_expression, upper_bound, copy_mask, trace_info,
              recorded_events, instance_events, &broadcast_events,
              src_inst_index);
        }
        else
          finalize_allreduce_without_broadcasts(
              predicate_guard, op, index, copy_expression, upper_bound,
              copy_mask, trace_info, recorded_events, applied_events,
              instance_events, local_fields, src_inst_index);
      }
      else
      {
        // Not a participant in the stages so just need to
        // do the stage -1 send and receive
        legion_assert(local_rank >= participating_ranks);
        initialize_allreduce_without_reductions(
            precondition, predicate_guard, op, index, copy_expression,
            upper_bound, copy_mask, trace_info, recorded_events, applied_events,
            instance_events, local_fields, reservations);
        const int mirror_rank = local_rank - participating_ranks;
        const AddressSpaceID target = (*collective_mapping)[mirror_rank];
        std::vector<ApEvent> read_events;
        send_allreduce_stage(
            allreduce_tag, -1 /*stage*/, local_rank, instance_events[0],
            predicate_guard, copy_expression, trace_info, local_fields[0],
            0 /*src index*/, &target, 1 /*total targets*/, read_events);
        if (!read_events.empty())
        {
          legion_assert(read_events.size() == 1);
          instance_events[0] = read_events[0];
        }
        // We can put this back in the first buffer without any
        // anti-dependences because we know the computation of the
        // result coming back had to already depend on the copy we
        // sent out to the target
        // Keep the local fields redop cleared since we're going to
        // doing direct copies here into these instance and not reductions
        std::vector<ApEvent> reduce_events;
        const std::vector<Reservation> no_reservations;
        receive_allreduce_stage(
            0 /*index*/, allreduce_tag, -1 /*stage*/, op, instance_events[0],
            predicate_guard, copy_expression, copy_mask, trace_info,
            applied_events, local_fields[0], no_reservations, &mirror_rank,
            1 /*total ranks*/, reduce_events);
        if (!reduce_events.empty())
        {
          legion_assert(reduce_events.size() == 1);
          instance_events[0] = reduce_events[0];
        }
        finalize_allreduce_without_broadcasts(
            predicate_guard, op, index, copy_expression, upper_bound, copy_mask,
            trace_info, recorded_events, applied_events, instance_events,
            local_fields);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent AllreduceView::initialize_allreduce_with_reductions(
        ApEvent precondition, PredEvent predicate_guard, Operation* op,
        unsigned index, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& applied_events,
        std::vector<ApEvent>& instance_events,
        std::vector<std::vector<CopySrcDstField> >& local_fields,
        std::vector<std::vector<Reservation> >& reservations)
    //--------------------------------------------------------------------------
    {
      const UniqueID op_id = op->get_unique_op_id();
      IndividualView* local_view = local_views.front();
      // Compute the reduction precondition for the first instance
      ApEvent reduce_pre = local_view->find_copy_preconditions(
          false /*reading*/, redop, copy_mask, copy_expression, op_id, index,
          applied_events, trace_info);
      if (precondition.exists())
      {
        if (reduce_pre.exists())
          reduce_pre =
              Runtime::merge_events(&trace_info, reduce_pre, precondition);
        else
          reduce_pre = precondition;
      }
      local_view->find_field_reservations(copy_mask, reservations.front());
      PhysicalManager* local_manager = local_view->get_manager();
      local_manager->compute_copy_offsets(copy_mask, local_fields.front());
      // Perform any local reductions and record their events
      set_redop(local_fields[0]);
      if (local_views.size() > 1)
      {
        const UniqueInst dst_inst(local_view);
        reduce_local(
            local_manager, 0 /*dst index*/, op, index, copy_expression,
            upper_bound, copy_mask, reduce_pre, predicate_guard,
            local_fields.front(), reservations.front(), dst_inst, trace_info,
            COLLECTIVE_BUTTERFLY_ALLREDUCE, instance_events, applied_events,
            nullptr /*recorded events*/, true /*allreduce*/, &local_fields);
        // We also need to populate the reservations here since the
        // reduce_local method will not do that for us
        for (unsigned idx = 1; idx < local_views.size(); idx++)
          local_views[idx]->find_field_reservations(
              copy_mask, reservations[idx]);
      }
      return reduce_pre;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::complete_initialize_allreduce_with_reductions(
        Operation* op, unsigned index, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events,
        std::vector<ApEvent>& instance_events,
        std::vector<std::vector<CopySrcDstField> >& local_fields,
        std::vector<ApEvent>* reduced)
    //--------------------------------------------------------------------------
    {
      ApEvent reduce_post;
      if (reduced != nullptr)
      {
        for (unsigned idx = 1; idx < instance_events.size(); idx++)
          if (instance_events[idx].exists())
            reduced->emplace_back(instance_events[idx]);
        reduce_post = Runtime::merge_events(&trace_info, *reduced);
      }
      else
        reduce_post = Runtime::merge_events(&trace_info, instance_events);
      const UniqueID op_id = op->get_unique_op_id();
      if (reduce_post.exists())
        local_views[0]->add_copy_user(
            false /*reading*/, redop, reduce_post, copy_mask, copy_expression,
            upper_bound, op_id, index, recorded_events, trace_info.recording,
            runtime->address_space);
      instance_events[0] = local_views[0]->find_copy_preconditions(
          false /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, trace_info);
      clear_redop(local_fields[0]);
    }

    //--------------------------------------------------------------------------
    void AllreduceView::initialize_allreduce_without_reductions(
        ApEvent precondition, PredEvent predicate_guard, Operation* op,
        unsigned index, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        std::set<RtEvent>& applied_events,
        std::vector<ApEvent>& instance_events,
        std::vector<std::vector<CopySrcDstField> >& local_fields,
        std::vector<std::vector<Reservation> >& reservations)
    //--------------------------------------------------------------------------
    {
      if (local_views.size() == 1)
      {
        const UniqueID op_id = op->get_unique_op_id();
        IndividualView* local_view = local_views.front();
        instance_events[0] = local_view->find_copy_preconditions(
            false /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
            index, applied_events, trace_info);
        local_view->find_field_reservations(copy_mask, reservations.front());
        PhysicalManager* local_manager = local_view->get_manager();
        local_manager->compute_copy_offsets(copy_mask, local_fields.front());
      }
      else
      {
        initialize_allreduce_with_reductions(
            precondition, predicate_guard, op, index, copy_expression,
            upper_bound, copy_mask, trace_info, applied_events, instance_events,
            local_fields, reservations);
        complete_initialize_allreduce_with_reductions(
            op, index, copy_expression, upper_bound, copy_mask, trace_info,
            recorded_events, applied_events, instance_events, local_fields);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent AllreduceView::finalize_allreduce_with_broadcasts(
        PredEvent predicate_guard, Operation* op, unsigned index,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        std::vector<ApEvent>& instance_events,
        const std::vector<std::vector<CopySrcDstField> >& local_fields,
        const unsigned final_index)
    //--------------------------------------------------------------------------
    {
      const UniqueID op_id = op->get_unique_op_id();
      IndividualView* local_view = local_views[final_index];
      if (instance_events[final_index].exists())
      {
        local_view->add_copy_user(
            false /*reading*/, 0 /*redop*/, instance_events[final_index],
            copy_mask, copy_expression, upper_bound, op_id, index,
            recorded_events, trace_info.recording, runtime->address_space);
        instance_events[final_index] = ApEvent::NO_AP_EVENT;
      }
      const ApEvent broadcast_pre = local_view->find_copy_preconditions(
          true /*reading*/, 0 /*redop*/, copy_mask, copy_expression, op_id,
          index, applied_events, trace_info);
      if (local_views.size() > 1)
      {
        const UniqueInst src_inst(local_view);
        broadcast_local(
            local_view->get_manager(), final_index, op, index, copy_expression,
            upper_bound, copy_mask, broadcast_pre, predicate_guard,
            local_fields[final_index], src_inst, trace_info,
            COLLECTIVE_BUTTERFLY_ALLREDUCE, instance_events, recorded_events,
            applied_events, true /*has instance events*/);
      }
      return broadcast_pre;
    }

    //--------------------------------------------------------------------------
    void AllreduceView::complete_finalize_allreduce_with_broadcasts(
        Operation* op, unsigned index, IndexSpaceExpression* copy_expression,
        IndexSpace upper_bound, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& recorded_events,
        const std::vector<ApEvent>& instance_events,
        std::vector<ApEvent>* broadcast, const unsigned final_index)
    //--------------------------------------------------------------------------
    {
      ApEvent broadcast_post;
      if (broadcast != nullptr)
      {
        for (unsigned idx = 0; idx < instance_events.size(); idx++)
          if ((idx != final_index) && instance_events[idx].exists())
            broadcast->emplace_back(instance_events[idx]);
        broadcast_post = Runtime::merge_events(&trace_info, *broadcast);
      }
      else
        broadcast_post = Runtime::merge_events(&trace_info, instance_events);
      const UniqueID op_id = op->get_unique_op_id();
      if (broadcast_post.exists())
        local_views[final_index]->add_copy_user(
            true /*reading*/, 0 /*redop*/, broadcast_post, copy_mask,
            copy_expression, upper_bound, op_id, index, recorded_events,
            trace_info.recording, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void AllreduceView::finalize_allreduce_without_broadcasts(
        PredEvent predicate_guard, Operation* op, unsigned index,
        IndexSpaceExpression* copy_expression, IndexSpace upper_bound,
        const FieldMask& copy_mask, const PhysicalTraceInfo& trace_info,
        std::set<RtEvent>& recorded_events, std::set<RtEvent>& applied_events,
        std::vector<ApEvent>& instance_events,
        const std::vector<std::vector<CopySrcDstField> >& local_fields,
        const unsigned final_index)
    //--------------------------------------------------------------------------
    {
      if (local_views.size() == 1)
      {
        if (instance_events[final_index].exists())
        {
          const UniqueID op_id = op->get_unique_op_id();
          IndividualView* local_view = local_views[final_index];
          local_view->add_copy_user(
              false /*reading*/, 0 /*redop*/, instance_events[final_index],
              copy_mask, copy_expression, upper_bound, op_id, index,
              recorded_events, trace_info.recording, runtime->address_space);
        }
      }
      else
      {
        finalize_allreduce_with_broadcasts(
            predicate_guard, op, index, copy_expression, upper_bound, copy_mask,
            trace_info, recorded_events, applied_events, instance_events,
            local_fields, final_index);
        complete_finalize_allreduce_with_broadcasts(
            op, index, copy_expression, upper_bound, copy_mask, trace_info,
            recorded_events, instance_events, nullptr /*broadcast events*/,
            final_index);
      }
    }

    //--------------------------------------------------------------------------
    void AllreduceView::send_allreduce_stage(
        const uint64_t allreduce_tag, const int stage, const int local_rank,
        ApEvent precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression,
        const PhysicalTraceInfo& trace_info,
        const std::vector<CopySrcDstField>& src_fields,
        const unsigned src_index, const AddressSpaceID* targets, size_t total,
        std::vector<ApEvent>& src_events)
    //--------------------------------------------------------------------------
    {
      ApBarrier src_bar;
      ShardID src_bar_shard = 0;
      const UniqueInst src_inst(local_views[src_index]);
      for (unsigned t = 0; t < total; t++)
      {
        CollectiveDistributeAllreduce rez;
        {
          RezCheck z(rez);
          rez.serialize(did);
          rez.serialize(allreduce_tag);
          rez.serialize(local_rank);
          rez.serialize(stage);
          pack_fields(
              rez, src_fields,
              local_views[src_index]->manager->get_unique_event());
          src_inst.serialize(rez);
          rez.serialize(precondition);
          rez.serialize<bool>(trace_info.recording);
          if (trace_info.recording)
          {
            if (!src_bar.exists())
            {
              src_bar_shard =
                  trace_info.record_barrier_creation(src_bar, total);
              src_events.emplace_back(src_bar);
            }
            rez.serialize(src_bar);
            rez.serialize(src_bar_shard);
          }
          else
          {
            const ApUserEvent src_done =
                Runtime::create_ap_user_event(&trace_info);
            rez.serialize(src_done);
            src_events.emplace_back(src_done);
          }
        }
        rez.dispatch(targets[t]);
      }
    }

    //--------------------------------------------------------------------------
    void AllreduceView::receive_allreduce_stage(
        const unsigned dst_index, const uint64_t allreduce_tag, const int stage,
        Operation* op, ApEvent dst_precondition, PredEvent predicate_guard,
        IndexSpaceExpression* copy_expression, const FieldMask& copy_mask,
        const PhysicalTraceInfo& trace_info, std::set<RtEvent>& applied_events,
        const std::vector<CopySrcDstField>& dst_fields,
        const std::vector<Reservation>& reservations, const int* expected_ranks,
        size_t total_ranks, std::vector<ApEvent>& dst_events)
    //--------------------------------------------------------------------------
    {
      legion_assert((stage != -2) || (total_ranks == 1));
      std::vector<AllReduceCopy> to_perform;
      {
        unsigned remaining = 0;
        AutoLock v_lock(view_lock);
        for (unsigned r = 0; r < total_ranks; r++)
        {
          const CopyKey key(allreduce_tag, expected_ranks[r], stage);
          std::map<CopyKey, AllReduceCopy>::iterator finder =
              all_reduce_copies.find(key);
          if (finder != all_reduce_copies.end())
          {
            to_perform.emplace_back(std::move(finder->second));
            all_reduce_copies.erase(finder);
          }
          else
            remaining++;
        }
        if (remaining > 0)
        {
          // If we still have outstanding copies, save a data
          // structure for them for when they arrive
          const std::pair<uint64_t, int> key(allreduce_tag, stage);
          legion_assert(remaining_stages.find(key) == remaining_stages.end());
          AllReduceStage& pending = remaining_stages[key];
          pending.dst_index = dst_index;
          pending.op = op;
          pending.copy_expression = copy_expression;
          copy_expression->add_nested_expression_reference(this->did);
          pending.copy_mask = copy_mask;
          pending.dst_fields = dst_fields;
          pending.reservations = reservations;
          pending.trace_info = new PhysicalTraceInfo(trace_info);
          pending.dst_precondition = dst_precondition;
          pending.predicate_guard = predicate_guard;
          pending.remaining_postconditions.reserve(remaining);
          for (unsigned idx = 0; idx < remaining; idx++)
          {
            const ApUserEvent post = Runtime::create_ap_user_event(&trace_info);
            pending.remaining_postconditions.emplace_back(post);
            dst_events.emplace_back(post);
          }
          if (trace_info.recording)
          {
            pending.applied_event = Runtime::create_rt_user_event();
            applied_events.insert(pending.applied_event);
          }
        }
      }
      if (!to_perform.empty())
      {
        const UniqueInst dst_inst(local_views[dst_index]);
        const LgEvent dst_unique_event =
            local_views[dst_index]->manager->get_unique_event();
        // Now we can perform any copies that we received
        for (const AllReduceCopy& it : to_perform)
        {
          const ApEvent pre = Runtime::merge_events(
              &trace_info, it.src_precondition, dst_precondition);
          const ApEvent post = copy_expression->issue_copy(
              op, trace_info, dst_fields, it.src_fields, reservations,
              it.src_inst.tid, dst_inst.tid, pre, predicate_guard,
              it.src_unique_event, dst_unique_event,
              COLLECTIVE_BUTTERFLY_ALLREDUCE, false /*copy restricted*/);
          if (trace_info.recording)
            trace_info.record_copy_insts(
                post, copy_expression, it.src_inst, dst_inst, copy_mask,
                copy_mask, redop, applied_events);
          if (it.barrier_postcondition.exists())
          {
            runtime->phase_barrier_arrive(
                it.barrier_postcondition, 1 /*count*/, post);
            if (trace_info.recording)
              trace_info.record_barrier_arrival(
                  it.barrier_postcondition, post, 1 /*count*/, applied_events,
                  it.barrier_shard);
          }
          else
          {
            legion_assert(it.src_postcondition.exists());
            Runtime::trigger_event(
                it.src_postcondition, post, trace_info, applied_events);
          }
          if (post.exists())
            dst_events.emplace_back(post);
        }
      }
    }

    //--------------------------------------------------------------------------
    void AllreduceView::process_distribute_allreduce(
        const uint64_t allreduce_tag, const int src_rank, const int stage,
        std::vector<CopySrcDstField>& src_fields,
        const ApEvent src_precondition, ApUserEvent src_postcondition,
        ApBarrier src_barrier, ShardID barrier_shard,
        const UniqueInst& src_inst, const LgEvent src_unique_event)
    //--------------------------------------------------------------------------
    {
      op::map<std::pair<uint64_t, int>, AllReduceStage>::iterator finder;
      {
        AutoLock v_lock(view_lock);
        const std::pair<uint64_t, int> stage_key(allreduce_tag, stage);
        finder = remaining_stages.find(stage_key);
        if (finder == remaining_stages.end())
        {
          const CopyKey key(allreduce_tag, src_rank, stage);
          legion_assert(all_reduce_copies.find(key) == all_reduce_copies.end());
          AllReduceCopy& copy = all_reduce_copies[key];
          copy.src_fields.swap(src_fields);
          copy.src_precondition = src_precondition;
          copy.src_postcondition = src_postcondition;
          copy.barrier_postcondition = src_barrier;
          copy.barrier_shard = barrier_shard;
          copy.src_inst = src_inst;
          copy.src_unique_event = src_unique_event;
          return;
        }
        legion_assert(!finder->second.remaining_postconditions.empty());
        // We can release the lock because we know map iterators are
        // not invalidated by insertion/deletion and any other copies
        // for this same stage are also just going to be reading except
        // for when we need to grab our event at the end to trigger
        // which we can re-take the lock to do
      }
      const UniqueInst dst_inst(local_views[finder->second.dst_index]);
      const ApEvent precondition = Runtime::merge_events(
          finder->second.trace_info, src_precondition,
          finder->second.dst_precondition);
      const ApEvent copy_post = finder->second.copy_expression->issue_copy(
          finder->second.op, *(finder->second.trace_info),
          finder->second.dst_fields, src_fields, finder->second.reservations,
          src_inst.tid, dst_inst.tid, precondition,
          finder->second.predicate_guard, src_unique_event,
          local_views[finder->second.dst_index]->manager->get_unique_event(),
          COLLECTIVE_BUTTERFLY_ALLREDUCE, false /*copy restricted*/);
      std::set<RtEvent> applied_events;
      if (finder->second.trace_info->recording)
        finder->second.trace_info->record_copy_insts(
            copy_post, finder->second.copy_expression, src_inst, dst_inst,
            finder->second.copy_mask, finder->second.copy_mask, redop,
            applied_events);
      if (src_barrier.exists())
      {
        runtime->phase_barrier_arrive(src_barrier, 1 /*count*/, copy_post);
        finder->second.trace_info->record_barrier_arrival(
            src_barrier, copy_post, 1 /*count*/, applied_events, barrier_shard);
      }
      else
      {
        legion_assert(src_postcondition.exists());
        Runtime::trigger_event(
            src_postcondition, copy_post, *finder->second.trace_info,
            finder->second.applied_events);
      }
      RtUserEvent applied;
      ApUserEvent to_trigger;
      PhysicalTraceInfo* trace_info = nullptr;
      IndexSpaceExpression* copy_expression = nullptr;
      {
        // Retake the lock and see if we're the last arrival
        AutoLock v_lock(view_lock);
        // Save any applied events that we have
        if (!applied_events.empty())
        {
          finder->second.applied_events.insert(
              applied_events.begin(), applied_events.end());
          applied_events.clear();
        }
        legion_assert(!finder->second.remaining_postconditions.empty());
        to_trigger = finder->second.remaining_postconditions.back();
        finder->second.remaining_postconditions.pop_back();
        if (finder->second.remaining_postconditions.empty())
        {
          // Last pass through, grab data and remove from the stages
          trace_info = finder->second.trace_info;
          copy_expression = finder->second.copy_expression;
          applied = finder->second.applied_event;
          applied_events.swap(finder->second.applied_events);
          remaining_stages.erase(finder);
        }
        else  // Need a copy of this
          trace_info = new PhysicalTraceInfo(*(finder->second.trace_info));
      }
      Runtime::trigger_event(
          to_trigger, copy_post, *trace_info, applied_events);
      if (applied.exists())
      {
        if (!applied_events.empty())
          Runtime::trigger_event(
              applied, Runtime::merge_events(applied_events));
        else
          Runtime::trigger_event(applied);
        applied_events.clear();
      }
      legion_assert(applied_events.empty());
      delete trace_info;
      if ((copy_expression != nullptr) &&
          copy_expression->remove_nested_expression_reference(this->did))
        delete copy_expression;
    }

    //--------------------------------------------------------------------------
    /*static*/ void CollectiveDistributeAllreduce::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      AllreduceView* view = static_cast<AllreduceView*>(
          runtime->find_or_request_logical_view(did, ready));
      uint64_t allreduce_tag;
      derez.deserialize(allreduce_tag);
      int src_rank;
      derez.deserialize(src_rank);
      int stage;
      derez.deserialize(stage);
      std::vector<CopySrcDstField> src_fields;
      std::set<RtEvent> ready_events;
      LgEvent src_unique_event = AllreduceView::unpack_fields(
          src_fields, derez, ready_events, view, ready);
      UniqueInst src_inst;
      src_inst.deserialize(derez);
      ApEvent src_precondition;
      derez.deserialize(src_precondition);
      bool recording;
      derez.deserialize<bool>(recording);
      ApBarrier src_barrier;
      ShardID barrier_shard = 0;
      ApUserEvent src_postcondition;
      if (recording)
      {
        derez.deserialize(src_barrier);
        derez.deserialize(barrier_shard);
      }
      else
        derez.deserialize(src_postcondition);

      if (ready.exists() && !ready.has_triggered())
        ready_events.insert(ready);
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        if (wait_on.exists() && !wait_on.has_triggered())
          wait_on.wait();
      }

      view->process_distribute_allreduce(
          allreduce_tag, src_rank, stage, src_fields, src_precondition,
          src_postcondition, src_barrier, barrier_shard, src_inst,
          src_unique_event);
    }

  }  // namespace Internal
}  // namespace Legion
