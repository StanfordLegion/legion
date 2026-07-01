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

// Included from across.h - do not include this directly

// Useful for IDEs
#include "legion/nodes/across.h"

namespace Legion {
  namespace Internal {

#ifdef DEFINE_NT_TEMPLATES

    // This is a bit out of place but needs to be here for instantiation to work

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructured*
        IndexSpaceOperationT<DIM, T>::create_across_unstructured(
            const std::map<Reservation, bool>& reservations,
            const bool compute_preimages, const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return new CopyAcrossUnstructuredT<DIM, T>(
          this, local_space, ApEvent::NO_AP_EVENT, reservations,
          compute_preimages, shadow_indirections);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructured* IndexSpaceNodeT<DIM, T>::create_across_unstructured(
        const std::map<Reservation, bool>& reservations,
        const bool compute_preimages, const bool shadow_indirections)
    //--------------------------------------------------------------------------
    {
      DomainT<DIM, T> local_space = get_tight_index_space();
      return new CopyAcrossUnstructuredT<DIM, T>(
          this, local_space, ApEvent::NO_AP_EVENT, reservations,
          compute_preimages, shadow_indirections);
    }

    /////////////////////////////////////////////////////////////
    // Templated Copy Across
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructuredT<DIM, T>::CopyAcrossUnstructuredT(
        IndexSpaceExpression* e, const DomainT<DIM, T>& domain, ApEvent ready,
        const std::map<Reservation, bool>& rsrvs, const bool preimages,
        const bool shadow)
      : CopyAcrossUnstructured(e, preimages, rsrvs), copy_domain(domain),
        copy_domain_ready(ready), shadow_indirections(shadow),
        shadow_layout(nullptr), need_src_indirect_precondition(true),
        need_dst_indirect_precondition(true),
        src_indirect_immutable_for_tracing(false),
        dst_indirect_immutable_for_tracing(false), has_empty_preimages(false)
    //--------------------------------------------------------------------------
    {
      expr->add_base_expression_reference(COPY_ACROSS_REF);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    CopyAcrossUnstructuredT<DIM, T>::~CopyAcrossUnstructuredT(void)
    //--------------------------------------------------------------------------
    {
      expr->record_index_space_user(last_copy);
      if (expr->remove_base_expression_reference(COPY_ACROSS_REF))
        delete expr;
      legion_assert(src_preimages.empty());
      legion_assert(dst_preimages.empty());
      // Clean up any preimages that we computed
      for (DomainT<DIM, T>& preimage : current_src_preimages)
        preimage.destroy(last_copy);
      for (DomainT<DIM, T>& preimage : current_dst_preimages)
        preimage.destroy(last_copy);
      // Destroy any shadow instances that we have
      for (std::pair<const Memory, ShadowInstance>& shadow_entry :
           shadow_instances)
        shadow_entry.second.instance.destroy(last_copy);
      // Lastly cleanup the indirections
      for (const CopyIndirection* indirection : indirections)
        delete indirection;
      if (shadow_layout != nullptr)
        delete shadow_layout;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent CopyAcrossUnstructuredT<DIM, T>::execute(
        Operation* op, PredEvent pred_guard, ApEvent copy_precondition,
        ApEvent src_indirect_precondition, ApEvent dst_indirect_precondition,
        const PhysicalTraceInfo& trace_info, const bool replay,
        const bool recurrent_replay, const unsigned stage)
    //--------------------------------------------------------------------------
    {
      if (stage == 0)
      {
        RtEvent src_preimages_ready, dst_preimages_ready;
        if (!src_indirections.empty() && compute_preimages &&
            (!src_indirect_immutable_for_tracing || !recurrent_replay))
        {
          // Compute new preimages and add the to the back of the queue
          ComputePreimagesHelper helper(
              this, op, src_indirect_precondition, true /*source*/);
          NT_TemplateHelper::demux<ComputePreimagesHelper>(
              src_indirect_type, &helper);
          if (helper.result.exists())
            src_preimages_ready = Runtime::protect_event(helper.result);
          AutoLock p_lock(preimage_lock);
          src_preimages.emplace_back(helper.new_preimages);
          if (spy_logging_level > LIGHT_SPY_LOGGING)
            src_preimage_preconditions.emplace_back(helper.result);
        }
        if (!dst_indirections.empty() && compute_preimages &&
            (!dst_indirect_immutable_for_tracing || !recurrent_replay))
        {
          // Compute new preimages and add them to the back of the queue
          ComputePreimagesHelper helper(
              this, op, dst_indirect_precondition, false /*source*/);
          NT_TemplateHelper::demux<ComputePreimagesHelper>(
              dst_indirect_type, &helper);
          if (helper.result.exists())
            dst_preimages_ready = Runtime::protect_event(helper.result);
          AutoLock p_lock(preimage_lock);
          dst_preimages.emplace_back(helper.new_preimages);
          if (spy_logging_level > LIGHT_SPY_LOGGING)
            dst_preimage_preconditions.emplace_back(helper.result);
        }
        // Make sure that all the stage 1's are ordered
        // by deferring execution if necessary
        if ((prev_done.exists() && !prev_done.has_triggered()) ||
            (src_preimages_ready.exists() &&
             !src_preimages_ready.has_triggered()) ||
            (dst_preimages_ready.exists() &&
             !dst_preimages_ready.has_triggered()))
        {
          const RtEvent defer = Runtime::merge_events(
              prev_done, src_preimages_ready, dst_preimages_ready);
          // Note that for tracing, we can't actually defer this in
          // the normal way because we need to actually get the real
          // finish event for the copy
          if (!trace_info.recording)
          {
            DeferCopyAcrossArgs args(
                this, op, pred_guard, copy_precondition,
                src_indirect_precondition, dst_indirect_precondition, replay,
                recurrent_replay, stage);
            prev_done = runtime->issue_runtime_meta_task(
                args, LG_LATENCY_DEFERRED_PRIORITY, defer);
            return args.done_event;
          }
          else
            defer.wait();
        }
      }
      // Need to rebuild indirections in the first time through or if we
      // are computing preimages and not doing a recurrent replay
      if (indirections.empty() || (!recurrent_replay && compute_preimages))
      {
        // Make a unique indirections identifier if necessary
        if (spy_logging_level > LIGHT_SPY_LOGGING)
          unique_indirections_identifier =
              runtime->get_unique_indirections_id();
        // No need for the lock here, we know we are ordered
        if (!indirections.empty())
        {
          for (typename std::vector<const CopyIndirection*>::const_iterator it =
                   indirections.begin();
               it != indirections.end(); it++)
            delete (*it);
          indirections.clear();
          individual_field_indexes.clear();
        }
        has_empty_preimages = false;
        // Prune preimages if necessary
        if (!src_indirections.empty())
        {
          if (!current_src_preimages.empty())
          {
            // Destroy any previous source preimage spaces
            for (DomainT<DIM, T>& preimage : current_src_preimages)
              preimage.destroy(last_copy);
          }
          if (compute_preimages)
          {
            // Get the next batch of src preimages to use
            AutoLock p_lock(preimage_lock);
            legion_assert(!src_preimages.empty());
            current_src_preimages.swap(src_preimages.front());
            src_preimages.pop_front();
            if (spy_logging_level > LIGHT_SPY_LOGGING)
            {
              legion_assert(!src_preimage_preconditions.empty());
              src_indirect_precondition = src_preimage_preconditions.front();
              src_preimage_preconditions.pop_front();
            }
          }
          RebuildIndirectionsHelper helper(
              this, op, src_indirect_precondition, true /*sources*/);
          NT_TemplateHelper::demux<RebuildIndirectionsHelper>(
              src_indirect_type, &helper);
          if (helper.empty)
            has_empty_preimages = true;
        }
        if (!dst_indirections.empty())
        {
          if (!current_dst_preimages.empty())
          {
            // Destroy any previous destination preimage spaces
            for (DomainT<DIM, T>& preimage : current_dst_preimages)
              preimage.destroy(last_copy);
          }
          if (compute_preimages)
          {
            // Get the next batch of dst preimages to use
            AutoLock p_lock(preimage_lock);
            legion_assert(!dst_preimages.empty());
            current_dst_preimages.swap(dst_preimages.front());
            dst_preimages.pop_front();
            if (spy_logging_level > LIGHT_SPY_LOGGING)
            {
              legion_assert(!dst_preimage_preconditions.empty());
              dst_indirect_precondition = dst_preimage_preconditions.front();
              dst_preimage_preconditions.pop_front();
            }
          }
          RebuildIndirectionsHelper helper(
              this, op, dst_indirect_precondition, false /*sources*/);
          NT_TemplateHelper::demux<RebuildIndirectionsHelper>(
              dst_indirect_type, &helper);
          if (helper.empty)
            has_empty_preimages = true;
        }
        if (spy_logging_level > LIGHT_SPY_LOGGING)
        {
          // This part isn't necessary for correctness but it helps Legion Spy
          // see the dependences between the preimages and copy operations
          if (src_indirect_precondition.exists() ||
              dst_indirect_precondition.exists())
            copy_precondition = Runtime::merge_events(
                nullptr, copy_precondition, src_indirect_precondition,
                dst_indirect_precondition);
        }
      }
      if (has_empty_preimages)
      {
        if (spy_logging_level > LIGHT_SPY_LOGGING)
        {
          ApUserEvent new_last_copy = Runtime::create_ap_user_event(nullptr);
          Runtime::trigger_event_untraced(new_last_copy);
          last_copy = new_last_copy;
          LegionSpy::log_indirect_events(
              op->get_unique_op_id(), expr->expr_id,
              unique_indirections_identifier, copy_precondition, last_copy);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            LegionSpy::log_indirect_field(
                last_copy, src_fields[idx].field_id,
                (idx < src_unique_events.size()) ? src_unique_events[idx] :
                                                   LgEvent::NO_LG_EVENT,
                src_fields[idx].indirect_index, dst_fields[idx].field_id,
                (idx < dst_unique_events.size()) ? dst_unique_events[idx] :
                                                   LgEvent::NO_LG_EVENT,
                dst_fields[idx].indirect_index, dst_fields[idx].redop_id);
          return last_copy;
        }
        else
          return ApEvent::NO_AP_EVENT;
      }
      legion_assert(src_fields.size() == dst_fields.size());
      // Now that we know we're going to do this copy add any profling requests
      Realm::ProfilingRequestSet requests;
      const unsigned total_copies = individual_field_indexes.empty() ?
                                        1 :
                                        individual_field_indexes.size();
      if (!replay)
        priority = op->add_copy_profiling_request(
            trace_info, requests, false /*fill*/, total_copies);
      ApEvent copy_pre;
      if (pred_guard.exists())
        copy_pre = Runtime::merge_events(
            nullptr, copy_precondition, ApEvent(pred_guard));
      else
        copy_pre = copy_precondition;
      if (!reservations.empty())
      {
        // TODO
        // Reservations are broken for indirection copies right now because we
        // need to exchange all the reservations collectively across the point
        // copy operations and then figure out which ones we need for each of
        // the indirection copies based on the instances used
        if (!indirections.empty())
          std::abort();
        // No need for tracing to know about the reservations
        for (const std::pair<const Reservation, bool>& it : reservations)
          copy_pre =
              Runtime::acquire_ap_reservation(it.first, it.second, copy_pre);
      }
      if (indirections.empty() || individual_field_indexes.empty())
      {
        if (!indirections.empty())
        {
          legion_assert(reservations.empty());
          // Merge in the indirection preconditions into the copy precondition
          // since that isn't included by default. Only need to do this for
          // non-pointwise
          // Note this code will need to change once we start handling
          // reservations correctly for indirection copies since we'll need
          // to merge these before acquiring the reservations
          copy_pre = Runtime::merge_events(
              nullptr, copy_pre, src_indirect_precondition,
              dst_indirect_precondition);
        }
        if (runtime->profiler != nullptr)
          runtime->profiler->add_copy_request(
              requests, this, op, copy_pre, total_copies);
        last_copy = ApEvent(copy_domain.copy(
            src_fields, dst_fields, indirections, requests, copy_pre,
            priority));
      }
      else
        last_copy = issue_individual_copies(op, copy_pre, requests);
      // Release any reservations
      for (const std::pair<const Reservation, bool>& it : reservations)
        Runtime::release_reservation(it.first, last_copy);
      if (pred_guard.exists())
      {
        // Protect against the poison from predication
        last_copy = Runtime::ignorefaults(last_copy);
        // Merge the preconditions into this result so they are still reflected
        // in the completion for this operation even if the operation ends up
        // being predicated out
        if (copy_precondition.exists())
        {
          if (last_copy.exists())
            last_copy =
                Runtime::merge_events(nullptr, last_copy, copy_precondition);
          else
            last_copy = copy_precondition;
        }
      }
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (!last_copy.exists())
          Runtime::rename_event(last_copy);
        legion_assert(op != nullptr);
        if (src_indirections.empty() && dst_indirections.empty())
        {
          LegionSpy::log_copy_events(
              op->get_unique_op_id(), expr->expr_id, src_tree_id, dst_tree_id,
              copy_precondition, last_copy, COLLECTIVE_NONE);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            LegionSpy::log_copy_field(
                last_copy, src_fields[idx].field_id, src_unique_events[idx],
                dst_fields[idx].field_id, dst_unique_events[idx],
                dst_fields[idx].redop_id);
        }
        else
        {
          LegionSpy::log_indirect_events(
              op->get_unique_op_id(), expr->expr_id,
              unique_indirections_identifier, copy_precondition, last_copy);
          for (unsigned idx = 0; idx < src_fields.size(); idx++)
            LegionSpy::log_indirect_field(
                last_copy, src_fields[idx].field_id,
                (idx < src_unique_events.size()) ? src_unique_events[idx] :
                                                   LgEvent::NO_LG_EVENT,
                src_fields[idx].indirect_index, dst_fields[idx].field_id,
                (idx < dst_unique_events.size()) ? dst_unique_events[idx] :
                                                   LgEvent::NO_LG_EVENT,
                dst_fields[idx].indirect_index, dst_fields[idx].redop_id);
        }
      }
      return last_copy;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void CopyAcrossUnstructuredT<DIM, T>::record_trace_immutable_indirection(
        bool source)
    //--------------------------------------------------------------------------
    {
      if (source)
        src_indirect_immutable_for_tracing = true;
      else
        dst_indirect_immutable_for_tracing = true;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    void CopyAcrossUnstructuredT<DIM, T>::release_shadow_instances(void)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Memory, ShadowInstance>& shadow_entry :
           shadow_instances)
        shadow_entry.second.instance.destroy(last_copy);
      shadow_instances.clear();
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent CopyAcrossUnstructuredT<DIM, T>::issue_individual_copies(
        Operation* op, const ApEvent precondition,
        const Realm::ProfilingRequestSet& requests)
    //--------------------------------------------------------------------------
    {
      legion_assert(compute_preimages);
      // TODO: fix reservation handling here
      legion_assert(reservations.empty());
      // This is the case of separate gather/scatter copies for each
      // of the individual preimages
      const bool gather = current_dst_preimages.empty();
      // Should be either a gather or a scatter, but not both
      legion_assert(current_src_preimages.empty() != gather);
      // Issue separate copies for each preimage
      std::vector<DomainT<DIM, T> >& preimages =
          gather ? current_src_preimages : current_dst_preimages;
      std::vector<CopySrcDstField>& fields = gather ? src_fields : dst_fields;
      legion_assert(preimages.size() == individual_field_indexes.size());
      legion_assert(preimages.size() == indirection_preconditions.size());
      std::vector<ApEvent> postconditions;
      for (unsigned idx = 0; idx < preimages.size(); idx++)
      {
        legion_assert(fields.size() == individual_field_indexes[idx].size());
        // Setup the indirect field indexes
        for (unsigned fidx = 0; fidx < fields.size(); fidx++)
          fields[fidx].indirect_index = individual_field_indexes[idx][fidx];
        ApEvent pre = precondition;
        // Check to see if we have an indirection precondition. In general
        // the precondition for the base indirection instance is already
        // encompased here because we had to read it to compute the preimages.
        // However, if we made shadow instances then we'll have a shadow
        // instance precondition for the shadow instances are ready.
        if (indirection_preconditions[idx].exists())
        {
          if (pre.exists())
            pre = Runtime::merge_events(
                nullptr, pre, indirection_preconditions[idx]);
          else
            pre = indirection_preconditions[idx];
        }
        Realm::ProfilingRequestSet preimage_requests;
        if (runtime->profiler != nullptr)
        {
          preimage_requests = requests;
          runtime->profiler->add_copy_request(preimage_requests, this, op, pre);
        }
        const ApEvent post(preimages[idx].copy(
            src_fields, dst_fields, indirections, preimage_requests, pre,
            priority));
        if (post.exists())
          postconditions.emplace_back(post);
      }
      if (postconditions.empty())
        return ApEvent::NO_AP_EVENT;
      return Runtime::merge_events(nullptr, postconditions);
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    Realm::InstanceLayoutGeneric*
        CopyAcrossUnstructuredT<DIM, T>::select_shadow_layout(bool source) const
    //--------------------------------------------------------------------------
    {
      const PhysicalInstance& indirect =
          source ? src_indirect_instance : dst_indirect_instance;
      const FieldID fid = source ? src_indirect_field : dst_indirect_field;
      // Get the layout for indirect instance
      const Realm::InstanceLayoutGeneric* layout = indirect.get_layout();
      // Get the size of the field that we need
      std::map<Realm::FieldID, Realm::InstanceLayoutGeneric::FieldLayout>::
          const_iterator finder = layout->fields.find(fid);
      legion_assert(finder != layout->fields.end());
      const size_t field_size = finder->second.size_in_bytes;
      // Don't use the base index space from the indirect instance because
      // it might be bigger than we need it to be. We could consider using
      // the preimage space, but we want this instance to be reusable even
      // if the preimage changes so we make it big enough to handle the
      // entire copy domain.
      std::vector<Rect<DIM, T> > covering;
      if (copy_domain.dense())
        covering.emplace_back(copy_domain.bounds);
      // See if we can compute a covering with up to 100% overhead
      // meaning this takes 2X space as all the points, if not we'll
      // just make pieces for all the rectangles
      else if (!copy_domain.compute_covering(
                   0 /*max rects*/, 100 /*allow up to 100% overhead*/,
                   covering))
      {
        // No covering just do all the rects individually
        for (Realm::IndexSpaceIterator itr(copy_domain); itr.valid; itr.step())
          covering.emplace_back(itr.rect);
      }
      // Figure out which order to iterate the dimensions based on the first
      // piece of the current indirection layout
      int dim_order[DIM];
      if (DIM > 1)
      {
        const Realm::InstanceLayout<DIM, T>* typed_layout =
            legion_safe_cast<const Realm::InstanceLayout<DIM, T>*>(layout);
        legion_assert(
            ((size_t)finder->second.list_idx) <
            typed_layout->piece_lists.size());
        legion_assert(
            !typed_layout->piece_lists[finder->second.list_idx].pieces.empty());
        const Realm::AffineLayoutPiece<DIM, T>* piece =
            legion_safe_cast<const Realm::AffineLayoutPiece<DIM, T>*>(
                typed_layout->piece_lists[finder->second.list_idx]
                    .pieces.front());
        // Sort dimensions based on the size of the strides
        std::map<size_t, int> strides;
        for (int d = 0; d < DIM; d++)
          strides.emplace(std::make_pair(piece->strides[d], d));
        for (int d = 0; d < DIM; d++)
        {
          std::map<size_t, int>::iterator next = strides.begin();
          dim_order[d] = next->second;
          strides.erase(next);
        }
      }
      else
        dim_order[0] = 0;
      const std::vector<Realm::FieldID> fields(1, fid);
      const std::vector<size_t> sizes(1, field_size);
      const Realm::InstanceLayoutConstraints constraints(fields, sizes, 0);
      Realm::InstanceLayoutGeneric* result =
          Realm::InstanceLayoutGeneric::choose_instance_layout<DIM, T>(
              copy_domain, covering, constraints, dim_order);
      result->alignment_reqd = layout->alignment_reqd;
      return result;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    PhysicalInstance
        CopyAcrossUnstructuredT<DIM, T>::allocate_shadow_indirection(
            Memory memory, UniqueID creator_uid, bool source,
            LgEvent& unique_event)
    //--------------------------------------------------------------------------
    {
      if (shadow_layout == nullptr)
        shadow_layout = select_shadow_layout(source);
      // See if we can allocate this in the target memory. If the instance
      // is not immediately available then we return a NO_INST since it is
      // not safe to do a deferred allocation here without risking deadlock.
      // Fortunately we can easily fall back to the previous indirection
      // instance if we have to so failing to allocate is not fatal.
      // Note that we're technically by-passing unbounded pool allocations
      // here which is not strictly safe, but since we're only going to hold
      // onto the memory if we allocation it immediately makes it safe.
      if (!unique_event.exists() && (runtime->profiler != nullptr))
        Runtime::rename_event(unique_event);
      MemoryManager::TaskLocalInstanceAllocator allocator(unique_event);
      ProfilingResponseBase base(&allocator, creator_uid, false);
      Realm::ProfilingRequestSet requests;
      Realm::ProfilingRequest& req = requests.add_request(
          runtime->find_local_group(), LG_LEGION_PROFILING_ID, &base,
          sizeof(base), LG_RESOURCE_PRIORITY);
      req.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
      if (runtime->profiler != nullptr)
        runtime->profiler->add_inst_request(
            requests, creator_uid, unique_event);
      PhysicalInstance result;
      const RtEvent ready(PhysicalInstance::create_instance(
          result, memory, *shadow_layout, requests));
      if (allocator.succeeded())
      {
        if (ready.exists())
        {
          ready.subscribe();
          if (!ready.has_triggered())
          {
            // Cannot permit a deferred allocation safely here since we're
            // not in the mapping stage of the pipeline
            result.destroy(ready);
            return PhysicalInstance::NO_INST;
          }
        }
        if (runtime->profiler != nullptr)
        {
          AutoLock p_lock(preimage_lock);
          profiling_shadow_instances[result] = unique_event;
        }
        return result;
      }
      else
        return PhysicalInstance::NO_INST;
    }

    //--------------------------------------------------------------------------
    template<int DIM, typename T>
    ApEvent CopyAcrossUnstructuredT<DIM, T>::update_shadow_indirection(
        PhysicalInstance shadow, LgEvent unique_event,
        ApEvent indirection_ready, const DomainT<DIM, T>& update_domain,
        Operation* op, size_t field_size, bool source) const
    //--------------------------------------------------------------------------
    {
      // No need for a trace info here since we should never be recording
      const PhysicalTraceInfo dummy_trace_info(nullptr, 0);
      const std::vector<Reservation> no_reservations;
      std::vector<CopySrcDstField> src_fields, dst_fields;
      CopySrcDstField& src_field = src_fields.emplace_back(CopySrcDstField());
      CopySrcDstField& dst_field = dst_fields.emplace_back(CopySrcDstField());
      if (source)
      {
        src_field.set_field(
            src_indirect_instance, src_indirect_field, field_size);
        dst_field.set_field(shadow, src_indirect_field, field_size);
        return expr->issue_copy_internal(
            op, update_domain, dummy_trace_info, dst_fields, src_fields,
            no_reservations, src_tree_id, src_tree_id, indirection_ready,
            PredEvent::NO_PRED_EVENT, src_indirect_instance_event, unique_event,
            COLLECTIVE_NONE, false /*record effect*/, priority,
            false /*replay*/);
      }
      else
      {
        src_field.set_field(
            dst_indirect_instance, dst_indirect_field, field_size);
        dst_field.set_field(shadow, dst_indirect_field, field_size);
        return expr->issue_copy_internal(
            op, update_domain, dummy_trace_info, dst_fields, src_fields,
            no_reservations, dst_tree_id, dst_tree_id, indirection_ready,
            PredEvent::NO_PRED_EVENT, dst_indirect_instance_event, unique_event,
            COLLECTIVE_NONE, false /*record effect*/, priority,
            false /*replay*/);
      }
    }
#endif  // defined(DEFINE_NT_TEMPLATES)

#ifdef DEFINE_NTNT_TEMPLATES
    //--------------------------------------------------------------------------
    template<int D1, typename T1>
    template<int D2, typename T2>
    ApEvent CopyAcrossUnstructuredT<D1, T1>::perform_compute_preimages(
        std::vector<DomainT<D1, T1> >& preimages, Operation* op,
        ApEvent precondition, const bool source)
    //--------------------------------------------------------------------------
    {
      const std::vector<IndirectRecord>& indirect_records =
          source ? src_indirections : dst_indirections;
      std::vector<Realm::IndexSpace<D2, T2> > targets(indirect_records.size());
      for (unsigned idx = 0; idx < indirect_records.size(); idx++)
        targets[idx] = indirect_records[idx].domain;
      ApEvent indirect_spaces_precondition;
      if (source ? need_src_indirect_precondition :
                   need_dst_indirect_precondition)
      {
        std::vector<ApEvent> preconditions;
        for (unsigned idx = 0; idx < indirect_records.size(); idx++)
        {
          const IndirectRecord& record = indirect_records[idx];
          ApEvent ready = record.domain_ready;
          if (ready.exists())
            preconditions.emplace_back(ready);
        }
        if (copy_domain_ready.exists())
          preconditions.emplace_back(copy_domain_ready);
        if (source)
        {
          // No need for tracing to know about this merge
          indirect_spaces_precondition =
              Runtime::merge_events(nullptr, preconditions);
          need_src_indirect_precondition = false;
        }
        else
        {
          indirect_spaces_precondition =
              Runtime::merge_events(nullptr, preconditions);
          need_dst_indirect_precondition = false;
        }
      }
      if (indirect_spaces_precondition.exists())
      {
        if (precondition.exists())
          precondition = Runtime::merge_events(
              nullptr, precondition, indirect_spaces_precondition);
        else
          precondition = indirect_spaces_precondition;
      }
      ApEvent result;
      if (is_range_indirection)
      {
        // Range preimage
        typedef Realm::FieldDataDescriptor<
            Realm::IndexSpace<D1, T1>, Realm::Rect<D2, T2> >
            RealmDescriptor;
        std::vector<RealmDescriptor> descriptors(1);
        RealmDescriptor& descriptor = descriptors.back();
        descriptor.inst =
            source ? src_indirect_instance : dst_indirect_instance;
        descriptor.field_offset =
            source ? src_indirect_field : dst_indirect_field;
        descriptor.index_space = copy_domain;
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_BY_PREIMAGE_RANGE, precondition);
        result = ApEvent(copy_domain.create_subspaces_by_preimage(
            descriptors, targets, preimages, requests, precondition));
      }
      else
      {
        // Point preimage
        typedef Realm::FieldDataDescriptor<
            Realm::IndexSpace<D1, T1>, Realm::Point<D2, T2> >
            RealmDescriptor;
        std::vector<RealmDescriptor> descriptors(1);
        RealmDescriptor& descriptor = descriptors.back();
        descriptor.inst =
            source ? src_indirect_instance : dst_indirect_instance;
        descriptor.field_offset =
            source ? src_indirect_field : dst_indirect_field;
        descriptor.index_space = copy_domain;
        Realm::ProfilingRequestSet requests;
        if (runtime->profiler != nullptr)
          runtime->profiler->add_partition_request(
              requests, op, DEP_PART_BY_PREIMAGE, precondition);
        result = ApEvent(copy_domain.create_subspaces_by_preimage(
            descriptors, targets, preimages, requests, precondition));
      }
      std::vector<ApEvent> valid_events;
      // We also need to make sure that all the sparsity maps are valid
      // on this node before we test them
      for (unsigned idx = 0; idx < preimages.size(); idx++)
      {
        const ApEvent valid(preimages[idx].make_valid());
        if (valid.exists())
        {
          if (implicit_profiler != nullptr)
            implicit_profiler->record_make_valid(valid);
          valid_events.emplace_back(valid);
        }
      }
      if (!valid_events.empty())
      {
        if (result.exists())
          valid_events.emplace_back(result);
        result = Runtime::merge_events(nullptr, valid_events);
      }
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == precondition)))
        Runtime::rename_event(result);
      LegionSpy::log_deppart_events(
          op->get_unique_op_id(), expr->expr_id, precondition, result,
          DEP_PART_BY_PREIMAGE);
      return result;
    }

    //--------------------------------------------------------------------------
    template<int D1, typename T1>
    template<int D2, typename T2>
    bool CopyAcrossUnstructuredT<D1, T1>::rebuild_indirections(
        Operation* op, ApEvent indirection_event, const bool source)
    //--------------------------------------------------------------------------
    {
      std::vector<CopySrcDstField>& fields = source ? src_fields : dst_fields;
      const std::vector<IndirectRecord>& indirect_records =
          source ? src_indirections : dst_indirections;
      std::map<Memory, ShadowInstance> old_shadows;
      old_shadows.swap(shadow_instances);
      nonempty_indexes.clear();
      if (compute_preimages)
      {
        std::vector<DomainT<D1, T1> >& preimages =
            source ? current_src_preimages : current_dst_preimages;
        for (unsigned idx = 0; idx < preimages.size(); idx++)
        {
          DomainT<D1, T1>& preimage = preimages[idx];
          DomainT<D1, T1> tightened = preimage.tighten();
          if (tightened.empty())
          {
            // Reclaim any sparsity maps eagerly
            preimage.destroy();
            preimage = DomainT<D1, T1>::make_empty();
          }
          else
          {
            preimage = tightened;
            nonempty_indexes.emplace_back(idx);
          }
        }
      }
      else
      {
        nonempty_indexes.resize(indirect_records.size());
        for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
          nonempty_indexes[idx] = idx;
      }
      typedef
          typename Realm::CopyIndirection<D1, T1>::template Unstructured<D2, T2>
              UnstructuredIndirection;
      // Legion Spy doesn't understand preimages, so go through and build
      // indirections for everything even if we are empty
      if (nonempty_indexes.empty() && (spy_logging_level <= LIGHT_SPY_LOGGING))
        return true;
      const unsigned base_offset = indirections.size();
      if (compute_preimages && (spy_logging_level <= LIGHT_SPY_LOGGING) &&
          (source ? dst_indirections.empty() : src_indirections.empty()))
      {
        // In the case that we've computed preimages, and we know we're just
        // doing a gather or a scatter (no full-indirections), then we
        // instead want to compute separate indirections for each
        // non-empty preimage because Realm's performance is better when
        // you have a single source or destination target for an indirection
        // Note we don't bother doing this with legion spy since it doesn't
        // know how to analyze these anyway
        individual_field_indexes.resize(nonempty_indexes.size());
        indirection_preconditions.resize(
            nonempty_indexes.size(), indirection_event);
        // If we're doing to shadow indirections, update the indirection
        // event to encompass the previous copy to handle WAR dependences
        // on the shadow instances
        if (shadow_indirections && last_copy.exists())
        {
          if (indirection_event.exists())
            indirection_event =
                Runtime::merge_events(nullptr, indirection_event, last_copy);
          else
            indirection_event = last_copy;
        }
        // If we're going to be doing shadow indirections then we want to
        // know how many different preimages need each shadow indirection
        // so we can determine whether to move just the preimage data or
        // all the indirection data to the shadow instance.
        std::map<Memory, unsigned /*non empty index*/> indirect_memories;
        if (shadow_indirections)
        {
          const Memory current = source ? src_indirect_instance.get_location() :
                                          dst_indirect_instance.get_location();
          for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
          {
            const unsigned nonempty_index = nonempty_indexes[idx];
            for (unsigned fidx = 0; fidx < fields.size(); fidx++)
            {
              const PhysicalInstance instance =
                  indirect_records[nonempty_index].instances[fidx];
              const Memory location = instance.get_location();
              if (location == current)
                continue;
              std::map<Memory, unsigned>::iterator finder =
                  indirect_memories.find(location);
              if (finder == indirect_memories.end())
                indirect_memories[location] = idx;
              else if (finder->second != idx)
                finder->second = nonempty_indexes.size();  // sentinel value
            }
          }
        }
        // We're also going to need to update preimages to match
        std::vector<DomainT<D1, T1> >& preimages =
            source ? current_src_preimages : current_dst_preimages;
        std::vector<DomainT<D1, T1> > new_preimages(nonempty_indexes.size());
        // Iterate over the non empty indexes and get instances for each field
        for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
        {
          const unsigned nonempty_index = nonempty_indexes[idx];
          std::vector<ApEvent> shadow_preconditions;
          // copy over the preimages to the set of dense non-empty preimages
          new_preimages[idx] = preimages[nonempty_index];
          std::vector<unsigned>& field_indexes = individual_field_indexes[idx];
          field_indexes.resize(fields.size());
          const unsigned offset = indirections.size();
          for (unsigned fidx = 0; fidx < fields.size(); fidx++)
          {
            const PhysicalInstance instance =
                indirect_records[nonempty_index].instances[fidx];
            // See if there is an unstructured index for this instance
            int indirect_index = -1;
            for (unsigned index = offset; index < indirections.size(); index++)
            {
              // It's safe to cast here because we know that the same types
              // made all these indirections as well
              const UnstructuredIndirection* unstructured =
                  static_cast<const UnstructuredIndirection*>(
                      indirections[index]);
              legion_assert(
                  shadow_indirections ||
                  (unstructured->inst ==
                   (source ? src_indirect_instance : dst_indirect_instance)));
              legion_assert(
                  unsigned(unstructured->field_id) ==
                  (source ? src_indirect_field : dst_indirect_field));
              legion_assert(unstructured->insts.size() == 1);
              if (unstructured->insts.back() != instance)
                continue;
              indirect_index = index;
              break;
            }
            if (indirect_index < 0)
            {
              // If we didn't make it then make it now
              UnstructuredIndirection* unstructured =
                  new UnstructuredIndirection();
              unstructured->field_id =
                  source ? src_indirect_field : dst_indirect_field;
              unstructured->inst =
                  source ? src_indirect_instance : dst_indirect_instance;
              const Memory memory = instance.get_location();
              // If we're doing shadow indirection instances check to see
              // if the indirection is in the same memory as the source
              // instance or not, if not make a shadow indirection
              if (shadow_indirections &&
                  (unstructured->inst.get_location() != memory))
              {
                // First check to see if we already have a new shadow
                // indirection instance to use
                std::map<Memory, ShadowInstance>::iterator finder =
                    shadow_instances.find(memory);
                if (finder == shadow_instances.end())
                {
                  // Didn't find it, see if we have an old shadow to reuse
                  finder = old_shadows.find(memory);
                  if (finder == old_shadows.end())
                  {
                    // Try to allocate a new shadow instance
                    LgEvent unique_event;
                    PhysicalInstance shadow = allocate_shadow_indirection(
                        memory, op->get_unique_op_id(), source, unique_event);
                    if (shadow.exists())
                    {
                      unstructured->inst = shadow;
                      // If we're successful issue a copy to it
                      // Check to see if we're issuing a copy to the whole
                      // domain or just the preimage subset of it
                      std::map<Memory, unsigned>::const_iterator memory_finder =
                          indirect_memories.find(memory);
                      legion_assert(memory_finder != indirect_memories.end());
                      ApEvent ready = update_shadow_indirection(
                          shadow, unique_event, indirection_event,
                          (memory_finder->second == nonempty_index) ?
                              new_preimages[idx] :
                              copy_domain,
                          op,
                          is_range_indirection ? sizeof(Rect<D2, T2>) :
                                                 sizeof(Point<D2, T2>),
                          source);
                      shadow_instances.emplace(std::make_pair(
                          memory, ShadowInstance{shadow, ready, unique_event}));
                      shadow_preconditions.emplace_back(ready);
                    }
                    // else default to using the original indirection
                  }
                  else
                  {
                    unstructured->inst = finder->second.instance;
                    // Issue a copy to bring the old shadow instance up
                    // to date with the new indirection field
                    // Check to see if we're issuing a copy to the whole
                    // domain or just the preimage subset of it
                    std::map<Memory, unsigned>::const_iterator memory_finder =
                        indirect_memories.find(memory);
                    legion_assert(memory_finder != indirect_memories.end());
                    ApEvent ready = update_shadow_indirection(
                        finder->second.instance, finder->second.unique_event,
                        indirection_event,
                        (memory_finder->second == nonempty_index) ?
                            new_preimages[idx] :
                            copy_domain,
                        op,
                        is_range_indirection ? sizeof(Rect<D2, T2>) :
                                               sizeof(Point<D2, T2>),
                        source);
                    // Update the new shadows instances
                    shadow_instances.emplace(std::make_pair(
                        memory, ShadowInstance{
                                    finder->second.instance, ready,
                                    finder->second.unique_event}));
                    // Remove this instance from the old shadow instances
                    old_shadows.erase(finder);
                    shadow_preconditions.emplace_back(ready);
                  }
                }
                else
                {
                  unstructured->inst = finder->second.instance;
                  shadow_preconditions.emplace_back(finder->second.ready);
                }
              }
              unstructured->is_ranges = is_range_indirection;
              unstructured->oor_possible = false;
              unstructured->aliasing_possible =
                  source ? false /*no aliasing*/ : possible_dst_aliasing;
              unstructured->subfield_offset = 0;
              unstructured->insts.emplace_back(instance);
              unstructured->spaces.resize(1);
              unstructured->spaces.back() =
                  indirect_records[nonempty_index].domain;
              // No next indirections yet...
              unstructured->next_indirection = nullptr;
              indirect_index = indirections.size();
              indirections.emplace_back(unstructured);
            }
            field_indexes[fidx] = indirect_index;
          }
          if (!shadow_preconditions.empty())
            indirection_preconditions[idx] =
                Runtime::merge_events(nullptr, shadow_preconditions);
        }
        // Now we can swap in the new preimages
        preimages.swap(new_preimages);
      }
      else
      {
        // Now that we have the non-empty indexes we can go through and make
        // the indirections for each of the fields. We'll try to share
        // indirections as much as possible wherever we can
        for (unsigned fidx = 0; fidx < fields.size(); fidx++)
        {
          // Compute our physical instances for this field
          std::vector<PhysicalInstance> instances;
          if (spy_logging_level > LIGHT_SPY_LOGGING)
          {
            instances.resize(indirect_records.size());
            for (unsigned idx = 0; idx < indirect_records.size(); idx++)
              instances[idx] = indirect_records[idx].instances[fidx];
          }
          else
          {
            instances.resize(nonempty_indexes.size());
            for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
              instances[idx] =
                  indirect_records[nonempty_indexes[idx]].instances[fidx];
          }
          // See if there is an unstructured index which already is what we want
          int indirect_index = -1;
          // Search through all the existing copy indirections starting from
          // the offset and check to see if we can reuse them
          for (unsigned index = base_offset; index < indirections.size();
               index++)
          {
            // It's safe to cast here because we know that the same types
            // made all these indirections as well
            const UnstructuredIndirection* unstructured =
                static_cast<const UnstructuredIndirection*>(
                    indirections[index]);
            legion_assert(
                unstructured->inst ==
                (source ? src_indirect_instance : dst_indirect_instance));
            legion_assert(
                unsigned(unstructured->field_id) ==
                (source ? src_indirect_field : dst_indirect_field));
            legion_assert(unstructured->insts.size() == instances.size());
            bool instances_match = true;
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              if (unstructured->insts[idx] == instances[idx])
                continue;
              instances_match = false;
              break;
            }
            if (!instances_match)
              continue;
            // If we made it here we can reuse this indirection
            indirect_index = index;
            break;
          }
          if (indirect_index < 0)
          {
            // If we didn't make it then make it now
            UnstructuredIndirection* unstructured =
                new UnstructuredIndirection();
            unstructured->field_id =
                source ? src_indirect_field : dst_indirect_field;
            unstructured->inst =
                source ? src_indirect_instance : dst_indirect_instance;
            unstructured->is_ranges = is_range_indirection;
            unstructured->oor_possible = compute_preimages ? false :
                                         source ? possible_src_out_of_range :
                                                  possible_dst_out_of_range;
            unstructured->aliasing_possible =
                source ? false /*no aliasing*/ : possible_dst_aliasing;
            unstructured->subfield_offset = 0;
            unstructured->insts.swap(instances);
            unstructured->spaces.resize(nonempty_indexes.size());
            for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
              unstructured->spaces[idx] =
                  indirect_records[nonempty_indexes[idx]].domain;
            // No next indirections yet...
            unstructured->next_indirection = nullptr;
            indirect_index = indirections.size();
            indirections.emplace_back(unstructured);
            if (spy_logging_level > LIGHT_SPY_LOGGING)
            {
              // If we made a new indirection then log it with Legion Spy
              LegionSpy::log_indirect_instance(
                  unique_indirections_identifier, indirect_index,
                  source ? src_indirect_instance_event :
                           dst_indirect_instance_event,
                  unstructured->field_id);
              for (const IndirectRecord& it : indirect_records)
                LegionSpy::log_indirect_group(
                    unique_indirections_identifier, indirect_index,
                    it.instance_events[fidx], it.index_space.get_id());
            }
          }
          fields[fidx].indirect_index = indirect_index;
        }
      }
      // If we have any residual old shadow instances that are no longer
      // needed then we can delete them
      for (const std::pair<const Memory, ShadowInstance>& shadow_entry :
           old_shadows)
        shadow_entry.second.instance.destroy(last_copy);
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (compute_preimages)
        {
          const size_t nonempty_size = nonempty_indexes.size();
          // Go through and fix-up all the indirections for execution
          for (typename std::vector<const CopyIndirection*>::const_iterator it =
                   indirections.begin() + base_offset;
               it != indirections.end(); it++)
          {
            UnstructuredIndirection* unstructured =
                const_cast<UnstructuredIndirection*>(
                    static_cast<const UnstructuredIndirection*>(*it));
            std::vector<PhysicalInstance> instances(nonempty_size);
            for (unsigned idx = 0; idx < nonempty_indexes.size(); idx++)
              instances[idx] = unstructured->insts[nonempty_indexes[idx]];
            unstructured->insts.swap(instances);
          }
        }
        return nonempty_indexes.empty();
      }
      else  // Not empty
        return false;
    }
#endif  // defined(DEFINE_NTNT_TEMPLATES)

  }  // namespace Internal
}  // namespace Legion
