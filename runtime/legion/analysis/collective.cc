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

#include "legion/analysis/collective.h"
#include "legion/operations/remote.h"
#include "legion/views/collective.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // RemoteCollectiveAnalysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void CollectiveAnalysis::pack_collective_analysis(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(get_context_index());
      rez.serialize(get_requirement_index());
      rez.serialize(get_match_space());
      Operation* op = get_operation();
      op->pack_remote_operation(rez, target, applied);
      const PhysicalTraceInfo& trace_info = get_trace_info();
      trace_info.pack_trace_info(rez);
    }

    //--------------------------------------------------------------------------
    RemoteCollectiveAnalysis::RemoteCollectiveAnalysis(
        size_t ctx_index, unsigned req_index, IndexSpace match, RemoteOp* op,
        Deserializer& derez)
      : context_index(ctx_index), requirement_index(req_index),
        match_space(match), operation(op),
        trace_info(PhysicalTraceInfo::unpack_trace_info(derez))
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteCollectiveAnalysis::~RemoteCollectiveAnalysis(void)
    //--------------------------------------------------------------------------
    {
      delete operation;
    }

    //--------------------------------------------------------------------------
    Operation* RemoteCollectiveAnalysis::get_operation(void) const
    //--------------------------------------------------------------------------
    {
      return operation;
    }

    //--------------------------------------------------------------------------
    /*static*/ RemoteCollectiveAnalysis* RemoteCollectiveAnalysis::unpack(
        Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t context_index;
      derez.deserialize(context_index);
      unsigned requirement_index;
      derez.deserialize(requirement_index);
      IndexSpace match_space;
      derez.deserialize(match_space);
      RemoteOp* op = RemoteOp::unpack_remote_operation(derez);
      return new RemoteCollectiveAnalysis(
          context_index, requirement_index, match_space, op, derez);
    }

    /////////////////////////////////////////////////////////////
    // CollectiveCopyFillAnalysis
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CollectiveCopyFillAnalysis::CollectiveCopyFillAnalysis(
        Operation* op, unsigned index, RegionNode* node, bool on_heap,
        const PhysicalTraceInfo& t_info, bool exclusive)
      : RegistrationAnalysis(op, index, node, on_heap, t_info, exclusive)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CollectiveCopyFillAnalysis::CollectiveCopyFillAnalysis(
        AddressSpaceID src, AddressSpaceID prev, Operation* op, unsigned index,
        RegionNode* node, bool on_heap,
        std::vector<PhysicalManager*>&& target_insts,
        op::vector<op::FieldMaskMap<InstanceView> >&& target_vws,
        std::vector<IndividualView*>&& source_vws,
        const PhysicalTraceInfo& t_info, CollectiveMapping* mapping,
        bool first_local, bool exclusive)
      : RegistrationAnalysis(
            src, prev, op, index, node, on_heap, std::move(target_insts),
            std::move(target_vws), std::move(source_vws), t_info, mapping,
            first_local, exclusive)
    //--------------------------------------------------------------------------
    {
      legion_assert(on_heap);
      legion_assert(target_instances.size() == target_views.size());
      // Remote case so no registration to perform
    }

    //--------------------------------------------------------------------------
    RtEvent CollectiveCopyFillAnalysis::perform_traversal(
        RtEvent precondition, const VersionInfo& version_info,
        std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
        return defer_traversal(precondition, version_info, applied_events);
      // Record ourselves with any collective target views before doing
      // the traversal so that we will be available for collective copies
      for (unsigned idx = 0; idx < target_views.size(); idx++)
      {
        PhysicalManager* manager = target_instances[idx];
        for (const std::pair<InstanceView* const, FieldMask>& it :
             target_views[idx])
        {
          if (!it.first->is_collective_view())
            continue;
          CollectiveView* collective = it.first->as_collective_view();
          collective->register_collective_analysis(
              manager, this, applied_events);
        }
      }
      return RegistrationAnalysis::perform_traversal(
          precondition, version_info, applied_events);
    }

  }  // namespace Internal
}  // namespace Legion
