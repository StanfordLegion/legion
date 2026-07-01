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

#ifndef __LEGION_DYNAMIC_COLLECTIVE_H__
#define __LEGION_DYNAMIC_COLLECTIVE_H__

#include "legion/operations/memoizable.h"

namespace Legion {
  namespace Internal {

    /**
     * \class DynamicCollectiveOp
     * A class for getting values from a collective operation
     * and writing them into a future. This will also give
     * us the framework necessary to handle roll backs on
     * collectives so we can memoize their results.
     */
    class DynamicCollectiveOp : public MemoizableOp {
    public:
      DynamicCollectiveOp(void);
      DynamicCollectiveOp(const DynamicCollectiveOp& rhs) = delete;
      virtual ~DynamicCollectiveOp(void);
    public:
      DynamicCollectiveOp& operator=(const DynamicCollectiveOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, const DynamicCollective& dc,
          Provenance* provenance);
    public:
      virtual const VersionInfo& get_version_info(unsigned idx) const
      {
        std::abort();
      }
      virtual const RegionRequirement& get_requirement(
          unsigned idx) const override
      {
        std::abort();
      }
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    public:
      // From MemoizableOp
      virtual void trigger_replay(void) override;
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_execution(void) override;
    protected:
      Future future;
      DynamicCollective collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_DYNAMIC_COLLECTIVE_H__
