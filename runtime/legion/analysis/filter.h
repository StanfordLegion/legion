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

#ifndef __LEGION_FILTER_ANALYSIS_H__
#define __LEGION_FILTER_ANALYSIS_H__

#include "legion/analysis/registration.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FilterAnalysis
     * For performing filter traversals on equivalence set trees
     */
    class FilterAnalysis : public RegistrationAnalysis,
                           public Heapify<FilterAnalysis, OPERATION_LIFETIME> {
    public:
      FilterAnalysis(
          Operation* op, unsigned index, RegionNode* node,
          const PhysicalTraceInfo& trace_info,
          const bool remove_restriction = false);
      FilterAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, const PhysicalTraceInfo& trace_info,
          const op::FieldMaskMap<InstanceView>& filter_views,
          CollectiveMapping* mapping, const bool first_local,
          const bool remove_restriction,
          std::map<InstanceView*, LamportClock>&& clocks);
      FilterAnalysis(const FilterAnalysis& rhs) = delete;
      virtual ~FilterAnalysis(void);
    public:
      FilterAnalysis& operator=(const FilterAnalysis& rhs) = delete;
    public:
      virtual RtEvent perform_traversal(
          RtEvent precondition, const VersionInfo& version_info,
          std::set<RtEvent>& applied_events) override;
      virtual bool perform_analysis(
          EquivalenceSet* set, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
      virtual RtEvent perform_remote(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
    public:
      op::FieldMaskMap<InstanceView> filter_views;
      const std::map<InstanceView*, LamportClock> remote_lamport_clocks;
      const bool remove_restriction;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FILTER_ANALYSIS_H__
