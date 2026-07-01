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

#ifndef __LEGION_RESET_OPERATION_H__
#define __LEGION_RESET_OPERATION_H__

#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ResetOp
     * A reset operation is an operation that goes through
     * the execution pipeline for the sole purpose of reseting
     * the equivalence sets of particular region in the region tree
     * so that later operations can select new equivalence sets.
     */
    class ResetOp : public Operation {
    public:
      ResetOp(void);
      ResetOp(const ResetOp& rhs) = delete;
      virtual ~ResetOp(void);
    public:
      ResetOp& operator=(const ResetOp& rhs) = delete;
    public:
      void initialize(
          InnerContext* ctx, LogicalRegion parent, LogicalRegion region,
          const std::set<FieldID>& fields);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual size_t get_region_count(void) const override;
      virtual const RegionRequirement& get_requirement(
          unsigned idx) const override
      {
        return requirement;
      }
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual unsigned find_parent_index(unsigned idx) override;
    protected:
      RegionRequirement requirement;
      unsigned parent_req_index;
    };

    /**
     * \class ReplResetOp
     * A reset operation that is aware it is being executed
     * in a control replicated context
     */
    class ReplResetOp : public ResetOp {
    public:
      ReplResetOp(void);
      ReplResetOp(const ReplResetOp& rhs) = delete;
      virtual ~ReplResetOp(void);
    public:
      ReplResetOp& operator=(const ReplResetOp& rhs) = delete;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
    protected:
      RtBarrier reset_barrier;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_RESET_OPERATION_H__
