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

#ifndef __LEGION_UPDATE_ANALYSIS_H__
#define __LEGION_UPDATE_ANALYSIS_H__

#include "legion/analysis/collective.h"

namespace Legion {
  namespace Internal {

    /**
     * \class UpdateAnalysis
     * For performing updates on equivalence set trees
     */
    class UpdateAnalysis : public CollectiveCopyFillAnalysis,
                           public Heapify<UpdateAnalysis, OPERATION_LIFETIME> {
    public:
      UpdateAnalysis(
          Operation* op, unsigned index, const RegionRequirement& req,
          RegionNode* node, const PhysicalTraceInfo& trace_info,
          const ApEvent precondition, const ApEvent term_event,
          const bool check_initialized, const bool record_valid);
      UpdateAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, const RegionUsage& usage, RegionNode* node,
          std::vector<PhysicalManager*>&& target_instances,
          op::vector<op::FieldMaskMap<InstanceView> >&& target_views,
          std::vector<IndividualView*>&& source_views,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, const RtEvent user_registered,
          const ApEvent precondition, const ApEvent term_event,
          const bool check_initialized, const bool record_valid,
          const bool first_local);
      UpdateAnalysis(const UpdateAnalysis& rhs) = delete;
      virtual ~UpdateAnalysis(void);
    public:
      UpdateAnalysis& operator=(const UpdateAnalysis& rhs) = delete;
    public:
      bool has_output_updates(void) const
      {
        return (output_aggregator != nullptr);
      }
    public:
      void record_uninitialized(
          const FieldMask& uninit, std::set<RtEvent>& applied_events);
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
      virtual RtEvent perform_registration(
          RtEvent precondition, const RegionUsage& usage,
          std::set<RtEvent>& registered_events, ApEvent init_precondition,
          ApEvent termination_event, ApEvent& instances_ready,
          bool symbolic = false) override;
      virtual ApEvent perform_output(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
    public:
      const RegionUsage usage;
      const ApEvent precondition;
      const ApEvent term_event;
      const bool check_initialized;
      const bool record_valid;
    public:
      // Have to lock the analysis to access these safely
      std::map<RtEvent, CopyFillAggregator*> input_aggregators;
      CopyFillAggregator* output_aggregator;
      std::set<RtEvent> guard_events;
      // For tracking uninitialized data
      FieldMask uninitialized;
      RtUserEvent uninitialized_reported;
      // For remote tracking
      RtEvent remote_user_registered;
      RtUserEvent user_registered;
    public:
      // Event for when target instances are ready
      ApEvent instances_ready;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_UPDATE_ANALYSIS_H__
