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

#ifndef __LEGION_OVERWRITE_ANALYSIS_H__
#define __LEGION_OVERWRITE_ANALYSIS_H__

#include "legion/analysis/physical.h"
#include "legion/tracing/recording.h"

namespace Legion {
  namespace Internal {

    /**
     * \class OverwriteAnalysis
     * For performing overwrite traversals on equivalence set trees
     */
    class OverwriteAnalysis
      : public PhysicalAnalysis,
        public Heapify<OverwriteAnalysis, OPERATION_LIFETIME> {
    public:
      OverwriteAnalysis(
          Operation* op, unsigned index, const RegionUsage& usage,
          IndexSpaceNode* node, LogicalView* view, const FieldMask& mask,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, const ApEvent precondition,
          const PredEvent true_guard = PredEvent::NO_PRED_EVENT,
          const PredEvent false_guard = PredEvent::NO_PRED_EVENT,
          const bool add_restriction = false, const bool first_local = true);
      // Also local but with a full set of instances
      OverwriteAnalysis(
          Operation* op, unsigned index, const RegionUsage& usage,
          IndexSpaceNode* node, const PhysicalTraceInfo& trace_info,
          const ApEvent precondition, const bool add_restriction = false);
      // Also local but with a full set of views
      OverwriteAnalysis(
          Operation* op, unsigned index, const RegionUsage& usage,
          IndexSpaceExpression* expr,
          const op::FieldMaskMap<LogicalView>& overwrite_views,
          const PhysicalTraceInfo& trace_info, const ApEvent precondition,
          const bool add_restriction = false);
      OverwriteAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, IndexSpaceExpression* expr, const RegionUsage& usage,
          IndexSpace upper_bound, op::FieldMaskMap<LogicalView>& views,
          op::FieldMaskMap<InstanceView>& reduction_views,
          const PhysicalTraceInfo& trace_info, const ApEvent precondition,
          const PredEvent true_guard, const PredEvent false_guard,
          CollectiveMapping* mapping, const bool first_local,
          const bool add_restriction);
      OverwriteAnalysis(const OverwriteAnalysis& rhs) = delete;
      virtual ~OverwriteAnalysis(void);
    public:
      OverwriteAnalysis& operator=(const OverwriteAnalysis& rhs) = delete;
    public:
      bool has_output_updates(void) const
      {
        return (output_aggregator != nullptr);
      }
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
      virtual RtEvent perform_registration(
          RtEvent precondition, const RegionUsage& usage,
          std::set<RtEvent>& applied_events, ApEvent init_precondition,
          ApEvent termination_event, ApEvent& instances_ready,
          bool symbolic = false) override;
      virtual ApEvent perform_output(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
      virtual IndexSpace get_collective_match_space(void) const override;
    public:
      RtEvent convert_views(
          LogicalRegion region, const InstanceSet& targets,
          unsigned analysis_index = 0);
    public:
      const RegionUsage usage;
      const IndexSpace upper_bound;
      const PhysicalTraceInfo trace_info;
      op::FieldMaskMap<LogicalView> views;
      op::FieldMaskMap<InstanceView> reduction_views;
      const ApEvent precondition;
      const PredEvent true_guard;
      const PredEvent false_guard;
      const bool add_restriction;
    public:
      // Can only safely be accessed when analysis is locked
      CopyFillAggregator* output_aggregator;
    protected:
      std::vector<PhysicalManager*> target_instances;
      op::vector<op::FieldMaskMap<InstanceView> > target_views;
      std::map<InstanceView*, size_t> collective_arrivals;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_OVERWRITE_ANALYSIS_H__
