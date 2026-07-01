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

#ifndef __LEGION_REGISTRATION_ANALYSIS_H__
#define __LEGION_REGISTRATION_ANALYSIS_H__

#include "legion/analysis/physical.h"
#include "legion/tracing/recording.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RegistrationAnalysis
     * A registration analysis is a kind of physical analysis that
     * also supports performing registration on the views of the
     */
    class RegistrationAnalysis : public PhysicalAnalysis {
    public:
      RegistrationAnalysis(
          Operation* op, unsigned index, RegionNode* node, bool on_heap,
          const PhysicalTraceInfo& trace_info, bool exclusive);
      RegistrationAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, bool on_heap,
          std::vector<PhysicalManager*>&& target_insts,
          op::vector<op::FieldMaskMap<InstanceView> >&& target_views,
          std::vector<IndividualView*>&& source_views,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, bool first_local,
          bool exclusive);
      // Remote registration analysis with no views
      RegistrationAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, bool on_heap,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, bool first_local,
          bool exclusive);
    public:
      virtual ~RegistrationAnalysis(void);
    public:
      RtEvent convert_views(
          LogicalRegion region, const InstanceSet& targets,
          const std::vector<PhysicalManager*>* sources = nullptr,
          const RegionUsage* usage = nullptr,
          bool collective_rendezvous = false, unsigned analysis_index = 0);
    public:
      virtual RtEvent perform_registration(
          RtEvent precondition, const RegionUsage& usage,
          std::set<RtEvent>& applied_events, ApEvent init_precondition,
          ApEvent termination_event, ApEvent& instances_ready,
          bool symbolic = false) override;
      virtual IndexSpace get_collective_match_space(void) const override;
    public:
      RegionNode* const region;
      const size_t context_index;
      const PhysicalTraceInfo trace_info;
    public:
      // Be careful to only access these after they are ready
      std::vector<PhysicalManager*> target_instances;
      op::vector<op::FieldMaskMap<InstanceView> > target_views;
      std::map<InstanceView*, size_t> collective_arrivals;
      std::vector<IndividualView*> source_views;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REGISTRATION_ANALYSIS_H__
