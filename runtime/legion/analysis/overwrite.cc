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

#include "legion/analysis/overwrite.h"
#include "legion/analysis/aggregator.h"
#include "legion/analysis/equivalence_set.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/index.h"
#include "legion/operations/remote.h"
#include "legion/views/reduction.h"
#include "legion/utilities/instance_set.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Overwrite Analysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(
        Operation* o, unsigned idx, const RegionUsage& use,
        IndexSpaceNode* expr, LogicalView* view, const FieldMask& mask,
        const PhysicalTraceInfo& t_info, CollectiveMapping* mapping,
        const ApEvent pre, const PredEvent true_g, const PredEvent false_g,
        const bool restriction, const bool first_local)
      : PhysicalAnalysis(
            o, idx, expr, true /*on heap*/, false /*immutable*/,
            true /*exclusive*/, mapping, first_local),
        usage(use), upper_bound(expr->handle), trace_info(t_info),
        precondition(pre), true_guard(true_g), false_guard(false_g),
        add_restriction(restriction), output_aggregator(nullptr)
    //--------------------------------------------------------------------------
    {
      if (view != nullptr)
      {
        if (view->is_reduction_kind())
          reduction_views.insert(view->as_instance_view(), mask);
        else
          views.insert(view, mask);
      }
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(
        Operation* o, unsigned idx, const RegionUsage& use,
        IndexSpaceNode* expr, const PhysicalTraceInfo& t_info,
        const ApEvent pre, const bool restriction)
      : PhysicalAnalysis(
            o, idx, expr, true /*on heap*/, false /*immutable*/,
            true /*exclusive*/),
        usage(use), upper_bound(expr->handle), trace_info(t_info),
        precondition(pre), add_restriction(restriction),
        output_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(
        Operation* o, unsigned idx, const RegionUsage& use,
        IndexSpaceExpression* expr,
        const op::FieldMaskMap<LogicalView>& overwrite_views,
        const PhysicalTraceInfo& t_info, const ApEvent pre,
        const bool restriction)
      : PhysicalAnalysis(
            o, idx, expr, true /*on heap*/, false /*immutable*/,
            true /*exclusive*/),
        usage(use), upper_bound(IndexSpace::NO_SPACE), trace_info(t_info),
        precondition(pre), add_restriction(restriction),
        output_aggregator(nullptr)
    //--------------------------------------------------------------------------
    {
      for (op::FieldMaskMap<LogicalView>::const_iterator it =
               overwrite_views.begin();
           it != overwrite_views.end(); it++)
      {
        if (it->first->is_reduction_kind())
          reduction_views.insert(it->first->as_instance_view(), it->second);
        else
          views.insert(it->first, it->second);
      }
    }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::OverwriteAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* o, unsigned idx,
        IndexSpaceExpression* expr, const RegionUsage& use, IndexSpace bound,
        op::FieldMaskMap<LogicalView>& vws,
        op::FieldMaskMap<InstanceView>& reductions,
        const PhysicalTraceInfo& t_info, const ApEvent pre,
        const PredEvent true_g, const PredEvent false_g,
        CollectiveMapping* mapping, const bool first_local,
        const bool restriction)
      : PhysicalAnalysis(
            src, prev, o, idx, expr, true /*on heap*/, false /*immutable*/,
            mapping, true /*exclusive*/, first_local),
        usage(use), upper_bound(bound), trace_info(t_info),
        views(vws, true /*copy*/), reduction_views(reductions, true /*copy*/),
        precondition(pre), true_guard(true_g), false_guard(false_g),
        add_restriction(restriction), output_aggregator(nullptr)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OverwriteAnalysis::~OverwriteAnalysis(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::convert_views(
        LogicalRegion region, const InstanceSet& targets,
        unsigned analysis_index)
    //--------------------------------------------------------------------------
    {
      legion_assert(targets.size() == 1);
      target_instances.resize(targets.size());
      for (unsigned idx = 0; idx < targets.size(); idx++)
        target_instances[idx] = targets[idx].get_physical_manager();
      InnerContext* context = op->find_physical_context(index);
      if (op->perform_collective_analysis(
              collective_mapping, collective_first_local))
      {
        if (collective_mapping != nullptr)
        {
          std::vector<IndividualView*> indiv(targets.size());
          context->convert_individual_views(targets, indiv, collective_mapping);
          target_views.resize(indiv.size());
          for (unsigned idx = 0; idx < indiv.size(); idx++)
            target_views[idx].insert(
                indiv[idx], targets[idx].get_valid_fields());
        }
        else
          return op->convert_collective_views(
              index, analysis_index, region, targets, context,
              collective_mapping, collective_first_local, target_views,
              collective_arrivals);
      }
      else
        context->convert_analysis_views(targets, target_views);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::perform_traversal(
        RtEvent precondition, const VersionInfo& version_info,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
        return defer_traversal(precondition, version_info, applied_events);
      if (!target_views.empty())
      {
        for (op::FieldMaskMap<InstanceView>::const_iterator it =
                 target_views[0].begin();
             it != target_views[0].end(); it++)
        {
          if (it->first->is_reduction_view())
            reduction_views.insert(it->first->as_reduction_view(), it->second);
          else
            views.insert(it->first, it->second);
        }
      }
      return PhysicalAnalysis::perform_traversal(
          precondition, version_info, applied_events);
    }

    //--------------------------------------------------------------------------
    bool OverwriteAnalysis::perform_analysis(
        EquivalenceSet* set, IndexSpaceExpression* expr, const bool expr_covers,
        const FieldMask& mask, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      set->overwrite_set(
          *this, expr, expr_covers, mask, applied_events, already_deferred);
      // Perform a check for migration after this
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::perform_remote(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_remote(perform_precondition, applied_events);
      // If there are no sets we're done
      if (remote_sets.empty())
        return RtEvent::NO_RT_EVENT;
      for (op::map<AddressSpaceID, op::FieldMaskMap<EquivalenceSet> >::
               const_iterator rit = remote_sets.begin();
           rit != remote_sets.end(); rit++)
      {
        legion_assert(!rit->second.empty());
        const AddressSpace target = rit->first;
        const RtUserEvent applied = Runtime::create_rt_user_event();
        RemoteOverwriteAnalysis rez;
        {
          RezCheck z(rez);
          rez.serialize(original_source);
          rez.serialize<size_t>(rit->second.size());
          for (op::FieldMaskMap<EquivalenceSet>::const_iterator it =
                   rit->second.begin();
               it != rit->second.end(); it++)
          {
            rez.serialize(it->first->did);
            rez.serialize(it->second);
          }
          analysis_expr->pack_expression(rez, target);
          op->pack_remote_operation(rez, target, applied_events);
          rez.serialize(index);
          rez.serialize(usage);
          rez.serialize(upper_bound);
          rez.serialize<size_t>(views.size());
          if (!views.empty())
          {
            for (op::FieldMaskMap<LogicalView>::const_iterator it =
                     views.begin();
                 it != views.end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
          }
          rez.serialize<size_t>(reduction_views.size());
          if (!reduction_views.empty())
          {
            for (op::FieldMaskMap<InstanceView>::const_iterator it =
                     reduction_views.begin();
                 it != reduction_views.end(); it++)
            {
              rez.serialize(it->first->did);
              rez.serialize(it->second);
            }
          }
          trace_info.pack_trace_info(rez);
          rez.serialize(true_guard);
          rez.serialize(false_guard);
          rez.serialize(precondition);
          rez.serialize<bool>(add_restriction);
          // We only need to pack the collective mapping once when going
          // from the origin space to the next space
          CollectiveMapping* mapping = get_replicated_mapping();
          if ((mapping != nullptr) &&
              (original_source == runtime->address_space))
          {
            mapping->pack(rez);
            rez.serialize<bool>(is_collective_first_local());
          }
          else
            rez.serialize<size_t>(0);
          rez.serialize(applied);
        }
        rez.dispatch(target);
        applied_events.insert(applied);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent OverwriteAnalysis::perform_registration(
        RtEvent precondition, const RegionUsage& usage,
        std::set<RtEvent>& applied_events, ApEvent init_precondition,
        ApEvent termination, ApEvent& instances_ready, bool symbolic)
    //--------------------------------------------------------------------------
    {
      legion_assert(termination.exists());
      legion_assert(!trace_info.recording);
      if (precondition.exists() && !precondition.has_triggered())
        return defer_registration(
            precondition, usage, applied_events, trace_info, init_precondition,
            termination, instances_ready, symbolic);
      const UniqueID op_id = op->get_unique_op_id();
      const size_t op_ctx_index = op->get_context_index();
      const AddressSpaceID local_space = runtime->address_space;
      // In this case we know the expression should be a region
      IndexSpaceNode* expr_node =
          legion_safe_cast<IndexSpaceNode*>(analysis_expr);
      std::vector<RtEvent> registered_events;
      std::vector<ApEvent> inst_ready_events;
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        for (op::FieldMaskMap<InstanceView>::const_iterator it =
                 target_views[idx].begin();
             it != target_views[idx].end(); it++)
        {
          size_t view_collective_arrivals = 0;
          if (!collective_arrivals.empty())
          {
            std::map<InstanceView*, size_t>::const_iterator finder =
                collective_arrivals.find(it->first);
            legion_assert(finder != collective_arrivals.end());
            view_collective_arrivals = finder->second;
          }
          const ApEvent ready = it->first->register_user(
              usage, it->second, expr_node, op_id, op_ctx_index, index,
              termination, target_instances[idx], collective_mapping,
              view_collective_arrivals, registered_events, applied_events,
              trace_info, local_space, symbolic);
          if (ready.exists())
            inst_ready_events.emplace_back(ready);
        }
      }
      if (!inst_ready_events.empty())
      {
        if ((inst_ready_events.size() > 1) || init_precondition.exists())
        {
          if (init_precondition.exists())
            inst_ready_events.emplace_back(init_precondition);
          instances_ready =
              Runtime::merge_events(&trace_info, inst_ready_events);
        }
        else
          instances_ready = inst_ready_events.back();
      }
      else
        instances_ready = init_precondition;
      if (!registered_events.empty())
        return Runtime::merge_events(registered_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    ApEvent OverwriteAnalysis::perform_output(
        RtEvent perform_precondition, std::set<RtEvent>& applied_events,
        const bool already_deferred)
    //--------------------------------------------------------------------------
    {
      if (perform_precondition.exists() &&
          !perform_precondition.has_triggered())
        return defer_output(
            perform_precondition, trace_info, false /*track*/, applied_events);
      if (output_aggregator != nullptr)
      {
        output_aggregator->issue_updates(
            trace_info, precondition, true /*restricted output*/);
        if (!output_aggregator->effects_applied.has_triggered())
          applied_events.insert(output_aggregator->effects_applied);
        if (output_aggregator->release_guards(applied_events))
          delete output_aggregator;
      }
      return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    IndexSpace OverwriteAnalysis::get_collective_match_space(void) const
    //--------------------------------------------------------------------------
    {
      // The only reason the upper bound index space should not exists should
      // be for tracing cases. We only invoke this method when performing
      // copies/fills for overwrites with restricted cases which shouldn't
      // be happening for tracing so we should always have such a bound.
      // If you hit the assert you've got a trace condition output being
      // applied ot an equivalence set with restrictions which is forcing
      // some copies to be performed which would be really strange.
      if (!upper_bound.exists())
        std::abort();
      return upper_bound;
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteOverwriteAnalysis::handle(
        Deserializer& derez, AddressSpaceID previous)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      AddressSpaceID original_source;
      derez.deserialize(original_source);
      size_t num_eq_sets;
      derez.deserialize(num_eq_sets);
      std::set<RtEvent> ready_events;
      std::vector<EquivalenceSet*> eq_sets(num_eq_sets, nullptr);
      local::vector<FieldMask> eq_masks(num_eq_sets);
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
      IndexSpaceExpression* expr =
          IndexSpaceExpression::unpack_expression(derez, previous);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      unsigned index;
      derez.deserialize(index);
      RegionUsage usage;
      derez.deserialize(usage);
      IndexSpace upper_bound;
      derez.deserialize(upper_bound);
      size_t num_views;
      derez.deserialize(num_views);
      op::FieldMaskMap<LogicalView> views;
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        views.insert(view, mask);
      }
      size_t num_reductions;
      derez.deserialize(num_reductions);
      op::FieldMaskMap<InstanceView> reductions;
      for (unsigned idx = 0; idx < num_reductions; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        RtEvent ready;
        LogicalView* view = runtime->find_or_request_logical_view(did, ready);
        if (ready.exists())
          ready_events.insert(ready);
        FieldMask mask;
        derez.deserialize(mask);
        reductions.insert(static_cast<InstanceView*>(view), mask);
      }
      std::set<RtEvent> deferral_events, applied_events;
      const PhysicalTraceInfo trace_info =
          PhysicalTraceInfo::unpack_trace_info(derez);
      PredEvent true_guard, false_guard;
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      ApEvent precondition;
      derez.deserialize(precondition);
      bool add_restriction;
      derez.deserialize(add_restriction);
      bool first_local = true;
      size_t collective_mapping_size;
      derez.deserialize(collective_mapping_size);
      CollectiveMapping* collective_mapping =
          ((collective_mapping_size) > 0) ?
              new CollectiveMapping(derez, collective_mapping_size) :
              nullptr;
      if (collective_mapping != nullptr)
        derez.deserialize<bool>(first_local);
      RtUserEvent applied;
      derez.deserialize(applied);

      // This takes ownership of the operation
      OverwriteAnalysis* analysis = new OverwriteAnalysis(
          original_source, previous, op, index, expr, usage, upper_bound, views,
          reductions, trace_info, precondition, true_guard, false_guard,
          collective_mapping, first_local, add_restriction);
      analysis->add_reference();
      // Make sure that all our pointers are ready
      RtEvent ready_event;
      if (!ready_events.empty())
        ready_event = Runtime::merge_events(ready_events);
      for (unsigned idx = 0; idx < eq_sets.size(); idx++)
        analysis->analyze(
            eq_sets[idx], eq_masks[idx], deferral_events, applied_events,
            ready_event);
      const RtEvent traversal_done = deferral_events.empty() ?
                                         RtEvent::NO_RT_EVENT :
                                         Runtime::merge_events(deferral_events);
      if (traversal_done.exists() || analysis->has_remote_sets())
        analysis->perform_remote(traversal_done, applied_events);
      if (traversal_done.exists() || analysis->has_output_updates())
        analysis->perform_output(traversal_done, applied_events);
      // Now we can trigger our applied event
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
      if (analysis->remove_reference())
        delete analysis;
    }

  }  // namespace Internal
}  // namespace Legion
