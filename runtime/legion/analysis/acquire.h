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

#ifndef __LEGION_ACQUIRE_ANALYSIS_H__
#define __LEGION_ACQUIRE_ANALYSIS_H__

#include "legion/analysis/registration.h"
#include "legion/managers/message.h"

namespace Legion {
  namespace Internal {

    /**
     * \class AcquireAnalysis
     * For performing acquires on equivalence set trees
     */
    class AcquireAnalysis
      : public RegistrationAnalysis,
        public Heapify<AcquireAnalysis, OPERATION_LIFETIME> {
    public:
      AcquireAnalysis(
          Operation* op, unsigned index, RegionNode* node,
          const PhysicalTraceInfo& t_info);
      AcquireAnalysis(
          AddressSpaceID src, AddressSpaceID prev, Operation* op,
          unsigned index, RegionNode* node, AcquireAnalysis* target,
          const PhysicalTraceInfo& trace_info,
          CollectiveMapping* collective_mapping, bool first_local);
      AcquireAnalysis(const AcquireAnalysis& rhs) = delete;
      virtual ~AcquireAnalysis(void);
    public:
      AcquireAnalysis& operator=(const AcquireAnalysis& rhs) = delete;
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
      static void handle_remote_acquires(
          Deserializer& derez, AddressSpaceID previous);
    public:
      AcquireAnalysis* const target_analysis;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_ACQUIRE_ANALYSIS_H__
