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

#ifndef __LEGION_COPY_ACROSS_H__
#define __LEGION_COPY_ACROSS_H__

#include "legion/analysis/physical.h"
#include "legion/managers/message.h"
#include "legion/tracing/recording.h"

namespace Legion {
  namespace Internal {

    /**
     * \class CopyAcrossAnalysis
     * For performing copy across traversals on equivalence set trees
     */
    class CopyAcrossAnalysis
      : public PhysicalAnalysis,
        public Heapify<CopyAcrossAnalysis, OPERATION_LIFETIME> {
    public:
      CopyAcrossAnalysis(
          Operation* op, unsigned src_index, unsigned dst_index,
          const RegionRequirement& src_req, const RegionRequirement& dst_req,
          const InstanceSet& target_instances,
          const op::vector<op::FieldMaskMap<InstanceView>>& target_views,
          const std::vector<IndividualView*>& source_views,
          const ApEvent precondition, const ApEvent dst_ready,
          const PredEvent pred_guard, const ReductionOpID redop,
          const std::vector<unsigned>& src_indexes,
          const std::vector<unsigned>& dst_indexes,
          const PhysicalTraceInfo& trace_info, const bool perfect);
      CopyAcrossAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned src_index, unsigned dst_index, const RegionUsage& src_usage,
          const RegionUsage& dst_usage, const LogicalRegion src_region,
          const LogicalRegion dst_region, const ApEvent dst_ready,
          std::vector<PhysicalManager*>&& target_instances,
          op::vector<op::FieldMaskMap<InstanceView>>&& target_views,
          std::vector<IndividualView*>&& source_views,
          const ApEvent precondition, const PredEvent pred_guard,
          const ReductionOpID redop, const std::vector<unsigned>& src_indexes,
          const std::vector<unsigned>& dst_indexes,
          const PhysicalTraceInfo& trace_info, const bool perfect);
      CopyAcrossAnalysis(const CopyAcrossAnalysis& rhs) = delete;
      virtual ~CopyAcrossAnalysis(void);
    public:
      CopyAcrossAnalysis& operator=(const CopyAcrossAnalysis& rhs) = delete;
    public:
      bool has_across_updates(void) const
      {
        return (across_aggregator != nullptr);
      }
      void record_uninitialized(
          const FieldMask& uninit, std::set<RtEvent>& applied_events);
      CopyFillAggregator* get_across_aggregator(void);
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
      virtual ApEvent perform_output(
          RtEvent precondition, std::set<RtEvent>& applied_events,
          const bool already_deferred = false) override;
      virtual IndexSpace get_collective_match_space(void) const override
      {
        return dst_region.get_index_space();
      }
    public:
      static inline FieldMask initialize_mask(const std::vector<unsigned>& idxs)
      {
        FieldMask result;
        for (unsigned idx = 0; idx < idxs.size(); idx++)
          result.set_bit(idxs[idx]);
        return result;
      }
      static void handle_remote_copies_across(
          Deserializer& derez, AddressSpaceID previous);
      static CopyAcrossHelper* create_across_helper(
          const FieldMask& src_mask, const FieldMask& dst_mask,
          const std::vector<PhysicalManager*>& dst_instances,
          const op::vector<op::FieldMaskMap<InstanceView>>& dst_views,
          const std::vector<unsigned>& src_indexes,
          const std::vector<unsigned>& dst_indexes);
      static std::vector<PhysicalManager*> convert_instances(
          const InstanceSet& instances);
    public:
      const FieldMask src_mask;
      const FieldMask dst_mask;
      const unsigned src_index;
      const unsigned dst_index;
      const RegionUsage src_usage;
      const RegionUsage dst_usage;
      const LogicalRegion src_region;
      const LogicalRegion dst_region;
      const ApEvent targets_ready;
      const std::vector<PhysicalManager*> target_instances;
      const op::vector<op::FieldMaskMap<InstanceView>> target_views;
      const std::vector<IndividualView*> source_views;
      const ApEvent precondition;
      const PredEvent pred_guard;
      const ReductionOpID redop;
      const std::vector<unsigned> src_indexes;
      const std::vector<unsigned> dst_indexes;
      CopyAcrossHelper* const across_helper;
      const PhysicalTraceInfo trace_info;
      const bool perfect;
    public:
      // Can only safely be accessed when analysis is locked
      FieldMask uninitialized;
      RtUserEvent uninitialized_reported;
      op::FieldMaskMap<IndexSpaceExpression> local_exprs;
      std::vector<ApEvent> copy_events;
      std::set<RtEvent> guard_events;
    protected:
      CopyFillAggregator* across_aggregator;
      RtUserEvent aggregator_guard;  // Guard event for the aggregator
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COPY_ACROSS_H__
