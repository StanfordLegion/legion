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

#ifndef __LEGION_RELEASE_ANALYSIS_H__
#define __LEGION_RELEASE_ANALYSIS_H__

#include "legion/analysis/collective.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ReleaseAnalysis
     * For performing releases on equivalence set trees
     */
    class ReleaseAnalysis
      : public CollectiveCopyFillAnalysis,
        public Heapify<ReleaseAnalysis, OPERATION_LIFETIME> {
    public:
      ReleaseAnalysis(
          Operation* op, unsigned index, ApEvent precondition, RegionNode* node,
          const PhysicalTraceInfo& trace_info);
      ReleaseAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, ApEvent precondition,
          ReleaseAnalysis* target,
          std::vector<PhysicalManager*>&& target_instances,
          op::vector<op::FieldMaskMap<InstanceView> >&& target_views,
          std::vector<IndividualView*>&& source_views,
          const PhysicalTraceInfo& info, CollectiveMapping* mapping,
          const bool first_local);
      ReleaseAnalysis(const ReleaseAnalysis& rhs) = delete;
      virtual ~ReleaseAnalysis(void);
    public:
      ReleaseAnalysis& operator=(const ReleaseAnalysis& rhs) = delete;
    public:
      virtual bool perform_analysis(
          EquivalenceSet* set, IndexSpaceExpression* expr,
          const bool expr_covers, const FieldMask& mask,
          std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
      virtual RtEvent perform_remote(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
      virtual RtEvent perform_updates(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
    public:
      const ApEvent precondition;
      ReleaseAnalysis* const target_analysis;
    public:
      // Can only safely be accessed when analysis is locked
      CopyFillAggregator* release_aggregator;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_RELEASE_ANALYSIS_H__
