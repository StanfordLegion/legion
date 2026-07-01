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

#include "legion/analysis/registration.h"
#include "legion/contexts/inner.h"
#include "legion/nodes/region.h"
#include "legion/operations/operation.h"
#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // RegistrationAnalysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RegistrationAnalysis::RegistrationAnalysis(
        Operation* op, unsigned index, RegionNode* node, bool on_heap,
        const PhysicalTraceInfo& t_info, bool exclusive)
      : PhysicalAnalysis(
            op, index, node->row_source, on_heap, false /*immutable*/,
            exclusive),
        region(node), context_index(op->get_context_index()), trace_info(t_info)
    //--------------------------------------------------------------------------
    {
      region->add_base_resource_ref(PHYSICAL_ANALYSIS_REF);
    }

    //--------------------------------------------------------------------------
    RegistrationAnalysis::RegistrationAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* op, unsigned index,
        RegionNode* node, bool on_heap,
        std::vector<PhysicalManager*>&& target_insts,
        op::vector<op::FieldMaskMap<InstanceView> >&& target_vws,
        std::vector<IndividualView*>&& source_vws,
        const PhysicalTraceInfo& t_info, CollectiveMapping* mapping,
        bool first_local, bool exclusive)
      : PhysicalAnalysis(
            src, prev, op, index, node->row_source, on_heap,
            false /*immutable*/, mapping, exclusive, first_local),
        region(node), context_index(op->get_context_index()),
        trace_info(t_info), target_instances(target_insts),
        target_views(target_vws), source_views(source_vws)
    //--------------------------------------------------------------------------
    {
      legion_assert(on_heap);
      legion_assert(target_instances.size() == target_views.size());
      region->add_base_resource_ref(PHYSICAL_ANALYSIS_REF);
    }

    //--------------------------------------------------------------------------
    RegistrationAnalysis::RegistrationAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* op, unsigned index,
        RegionNode* node, bool on_heap, const PhysicalTraceInfo& t_info,
        CollectiveMapping* mapping, bool first_local, bool exclusive)
      : PhysicalAnalysis(
            src, prev, op, index, node->row_source, on_heap,
            false /*immutable*/, mapping, exclusive, first_local),
        region(node), context_index(op->get_context_index()), trace_info(t_info)
    //--------------------------------------------------------------------------
    {
      legion_assert(on_heap);
      region->add_base_resource_ref(PHYSICAL_ANALYSIS_REF);
    }

    //--------------------------------------------------------------------------
    RegistrationAnalysis::~RegistrationAnalysis(void)
    //--------------------------------------------------------------------------
    {
      if (region->remove_base_resource_ref(PHYSICAL_ANALYSIS_REF))
        delete region;
    }

    //--------------------------------------------------------------------------
    RtEvent RegistrationAnalysis::convert_views(
        LogicalRegion region, const InstanceSet& targets,
        const std::vector<PhysicalManager*>* sources, const RegionUsage* usage,
        bool collective_rendezvous, unsigned analysis_index)
    //--------------------------------------------------------------------------
    {
      InnerContext* context = op->find_physical_context(index);
      if ((sources != nullptr) && !sources->empty())
        context->convert_individual_views(*sources, source_views);
      target_instances.resize(targets.size());
      for (unsigned idx = 0; idx < targets.size(); idx++)
        target_instances[idx] = targets[idx].get_physical_manager();
      // Find any atomic locks we need to take for these instances
      // Note that for now we also treat exclusive-reductions as
      // needing to be atomic since we don't have a semantics for
      // what exclusive reductions mean today
      // Note this needs to be done eagerly and cannot be deferred!
      // All reductions need to get an atomic lock since they can race
      // with copy reductions to the same instance as the task
      if ((usage != nullptr) && (IS_ATOMIC(*usage) || IS_REDUCE(*usage)))
      {
        std::vector<IndividualView*> individual_views;
        context->convert_individual_views(target_instances, individual_views);
        // If we're doing a reduction, we need exclusive coherence for any
        // exclusive or atomic coherence, otherwise non-exclusive is fine
        // since that will still prevent races with reduction copies
        const bool exclusive = IS_REDUCE(*usage) ?
                                   (IS_EXCLUSIVE(*usage) || IS_ATOMIC(*usage)) :
                                   HAS_WRITE(*usage);
        for (unsigned idx = 0; idx < individual_views.size(); idx++)
          individual_views[idx]->find_atomic_reservations(
              targets[idx].get_valid_fields(), op, index, exclusive);
      }
      if (collective_rendezvous)
        return op->convert_collective_views(
            index, analysis_index, region, targets, context, collective_mapping,
            collective_first_local, target_views, collective_arrivals);
      else if (op->perform_collective_analysis(
                   collective_mapping, collective_first_local))
      {
        if (collective_mapping != nullptr)
        {
          std::vector<IndividualView*> views(targets.size());
          context->convert_individual_views(targets, views, collective_mapping);
          target_views.resize(views.size());
          for (unsigned idx = 0; idx < views.size(); idx++)
            target_views[idx].insert(
                views[idx], targets[idx].get_valid_fields());
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
    RtEvent RegistrationAnalysis::perform_registration(
        RtEvent precondition, const RegionUsage& usage,
        std::set<RtEvent>& applied_events, ApEvent init_precondition,
        ApEvent termination, ApEvent& instances_ready, bool symbolic)
    //--------------------------------------------------------------------------
    {
      legion_assert(region != nullptr);
      legion_assert(termination.exists());
      if (precondition.exists() && !precondition.has_triggered())
        return defer_registration(
            precondition, usage, applied_events, trace_info, init_precondition,
            termination, instances_ready, symbolic);
      // Perform the registration
      const UniqueID op_id = op->get_unique_op_id();
      const size_t op_ctx_index = op->get_context_index();
      const AddressSpaceID local_space = runtime->address_space;
      IndexSpaceNode* expr_node = region->row_source;
      legion_assert(expr_node == analysis_expr);
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
        if (init_precondition.exists())
          inst_ready_events.emplace_back(init_precondition);
        instances_ready = Runtime::merge_events(&trace_info, inst_ready_events);
      }
      else
        instances_ready = init_precondition;
      if (trace_info.recording)
      {
        InnerContext* context = op->find_physical_context(index);
        std::vector<IndividualView*> individual_views;
        context->convert_individual_views(target_instances, individual_views);
        for (unsigned idx = 0; idx < individual_views.size(); idx++)
        {
          const UniqueInst unique_inst(individual_views[idx]);
          trace_info.record_op_inst(
              usage, target_views[idx].get_valid_mask(), unique_inst, region,
              op, applied_events);
        }
      }
      if (!registered_events.empty())
        return Runtime::merge_events(registered_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    IndexSpace RegistrationAnalysis::get_collective_match_space(void) const
    //--------------------------------------------------------------------------
    {
      return region->row_source->handle;
    }

  }  // namespace Internal
}  // namespace Legion
