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

#ifndef __LEGION_COLLECTIVE_ANALYSIS_H__
#define __LEGION_COLLECTIVE_ANALYSIS_H__

#include "legion/analysis/registration.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CollectiveAnalysis
     * This base class provides a virtual interface for collective
     * analyses. Collective analyses are registered with the collective
     * views to help with performing collective copy operations since
     * control for collective copy operations originates from just a
     * single physical analysis. We don't want to attribute all those
     * copies and trace recordings to a single analysis, so instead
     * we register CollectiveAnalysis objects with collective views
     * so that they can be used to help issue the copies.
     */
    class CollectiveAnalysis {
    public:
      virtual ~CollectiveAnalysis(void) { }
      virtual size_t get_context_index(void) const = 0;
      virtual unsigned get_requirement_index(void) const = 0;
      virtual IndexSpace get_match_space(void) const = 0;
      virtual Operation* get_operation(void) const = 0;
      virtual const PhysicalTraceInfo& get_trace_info(void) const = 0;
      void pack_collective_analysis(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied_events) const;
      virtual void add_analysis_reference(void) = 0;
      virtual bool remove_analysis_reference(void) = 0;
    };

    /**
     * \class RemoteCollectiveAnalysis
     * This class contains the needed data structures for representing
     * a physical analysis when registered on a remote collective view.
     */
    class RemoteCollectiveAnalysis : public CollectiveAnalysis,
                                     public Collectable {
    public:
      RemoteCollectiveAnalysis(
          size_t ctx_index, unsigned req_index, IndexSpace match_space,
          RemoteOp* op, Deserializer& derez);
      virtual ~RemoteCollectiveAnalysis(void);
      virtual size_t get_context_index(void) const override
      {
        return context_index;
      }
      virtual unsigned get_requirement_index(void) const override
      {
        return requirement_index;
      }
      virtual IndexSpace get_match_space(void) const override
      {
        return match_space;
      }
      virtual Operation* get_operation(void) const override;
      virtual const PhysicalTraceInfo& get_trace_info(void) const override
      {
        return trace_info;
      }
      virtual void add_analysis_reference(void) override { add_reference(); }
      virtual bool remove_analysis_reference(void) override
      {
        return remove_reference();
      }
      static RemoteCollectiveAnalysis* unpack(Deserializer& derez);
    public:
      const size_t context_index;
      const unsigned requirement_index;
      const IndexSpace match_space;
      RemoteOp* const operation;
      const PhysicalTraceInfo trace_info;
    };

    /**
     * \class CollectiveCopyFillAnalysis
     * This is an intermediate base class for analyses that helps support
     * performing collective copies and fills on a destination collective
     * instance. It works be registering itself with the local collective
     * instance for its point so any collective copies and fills can be
     * attributed to the correct operation. After the analysis is done
     * then the analysis will unregister itself with the collecitve instance.
     */
    class CollectiveCopyFillAnalysis : public RegistrationAnalysis,
                                       public CollectiveAnalysis {
    public:
      CollectiveCopyFillAnalysis(
          Operation* op, unsigned index, RegionNode* node, bool on_heap,
          const PhysicalTraceInfo& trace_info, bool exclusive);
      CollectiveCopyFillAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, bool on_heap,
          std::vector<PhysicalManager*>&& target_insts,
          op::vector<op::FieldMaskMap<InstanceView> >&& target_views,
          std::vector<IndividualView*>&& source_views,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, bool first_local,
          bool exclusive);
      virtual ~CollectiveCopyFillAnalysis(void) { }
    public:
      virtual size_t get_context_index(void) const override
      {
        return context_index;
      }
      virtual unsigned get_requirement_index(void) const override
      {
        return index;
      }
      virtual IndexSpace get_match_space(void) const override
      {
        return get_collective_match_space();
      }
      virtual Operation* get_operation(void) const override { return op; }
      virtual const PhysicalTraceInfo& get_trace_info(void) const override
      {
        return trace_info;
      }
      virtual void add_analysis_reference(void) override { add_reference(); }
      virtual bool remove_analysis_reference(void) override
      {
        return remove_reference();
      }
      virtual RtEvent perform_traversal(
          RtEvent precondition, const VersionInfo& version_info,
          std::set<RtEvent>& applied_events) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COLLECTIVE_ANALYSIS_H__
