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

#ifndef __LEGION_VALID_ANALYSIS_H__
#define __LEGION_VALID_ANALYSIS_H__

#include "legion/analysis/physical.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ValidInstAnalysis
     * For finding valid instances in equivalence set trees
     */
    class ValidInstAnalysis
      : public PhysicalAnalysis,
        public Heapify<ValidInstAnalysis, OPERATION_LIFETIME> {
    public:
      ValidInstAnalysis(
          Operation* op, unsigned index, IndexSpaceExpression* expr,
          ReductionOpID redop = 0);
      ValidInstAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, IndexSpaceExpression* expr, ValidInstAnalysis* target,
          ReductionOpID redop);
      ValidInstAnalysis(const ValidInstAnalysis& rhs) = delete;
      virtual ~ValidInstAnalysis(void);
    public:
      ValidInstAnalysis& operator=(const ValidInstAnalysis& rhs) = delete;
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
      const ReductionOpID redop;
      ValidInstAnalysis* const target_analysis;
    };

    /**
     * \class InvalidInstAnalysis
     * For finding which of a set of instances are not valid across
     * a set of equivalence sets
     */
    class InvalidInstAnalysis
      : public PhysicalAnalysis,
        public Heapify<InvalidInstAnalysis, OPERATION_LIFETIME> {
    public:
      InvalidInstAnalysis(
          Operation* op, unsigned index, IndexSpaceExpression* expr,
          const lng::FieldMaskMap<LogicalView>& valid_instances);
      InvalidInstAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, IndexSpaceExpression* expr,
          InvalidInstAnalysis* target,
          const op::FieldMaskMap<LogicalView>& valid_instances);
      InvalidInstAnalysis(const InvalidInstAnalysis& rhs) = delete;
      virtual ~InvalidInstAnalysis(void);
    public:
      InvalidInstAnalysis& operator=(const InvalidInstAnalysis& rhs) = delete;
    public:
      inline bool has_invalid(void) const
      {
        return (
            (recorded_instances != nullptr) && !recorded_instances->empty());
      }
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
      const op::FieldMaskMap<LogicalView> valid_instances;
      InvalidInstAnalysis* const target_analysis;
    };

    /**
     * \class AntivalidInstAnalysis
     * For checking that some views are not in the set of valid instances
     */
    class AntivalidInstAnalysis
      : public PhysicalAnalysis,
        public Heapify<AntivalidInstAnalysis, OPERATION_LIFETIME> {
    public:
      AntivalidInstAnalysis(
          Operation* op, unsigned index, IndexSpaceExpression* expr,
          const op::FieldMaskMap<LogicalView>& anti_instances);
      AntivalidInstAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, IndexSpaceExpression* expr,
          AntivalidInstAnalysis* target,
          const op::FieldMaskMap<LogicalView>& anti_instances);
      AntivalidInstAnalysis(const AntivalidInstAnalysis& rhs) = delete;
      virtual ~AntivalidInstAnalysis(void);
    public:
      AntivalidInstAnalysis& operator=(const AntivalidInstAnalysis& r) = delete;
    public:
      inline bool has_antivalid(void) const
      {
        return (
            (recorded_instances != nullptr) && !recorded_instances->empty());
      }
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
      const op::FieldMaskMap<LogicalView> antivalid_instances;
      AntivalidInstAnalysis* const target_analysis;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_VALID_ANALYSIS_H__
