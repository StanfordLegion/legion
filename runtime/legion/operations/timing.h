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

#ifndef __LEGION_TIMING_OPERATION_H__
#define __LEGION_TIMING_OPERATION_H__

#include "legion/operations/fence.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TimingOp
     * Operation for performing timing measurements
     */
    class TimingOp : public FenceOp {
    public:
      TimingOp(void);
      TimingOp(const TimingOp& rhs) = delete;
      virtual ~TimingOp(void);
    public:
      TimingOp& operator=(const TimingOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, const TimingLauncher& launcher,
          Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    public:
      virtual void trigger_complete(ApEvent complete) override;
      virtual void trigger_commit(void) override;
      virtual void perform_measurement(void) override;
    protected:
      TimingMeasurement measurement;
      RtEvent measured;
    };

    /**
     * \class ReplTimingOp
     * A timing operation that is aware that it is
     * being executed in a control replication context
     */
    class ReplTimingOp : public ReplFenceOp {
    public:
      ReplTimingOp(void);
      ReplTimingOp(const ReplTimingOp& rhs) = delete;
      virtual ~ReplTimingOp(void);
    public:
      ReplTimingOp& operator=(const ReplTimingOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, const TimingLauncher& launcher,
          Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        return false;
      }
    public:
      virtual void trigger_complete(ApEvent complete) override;
      virtual void trigger_commit(void) override;
      virtual void perform_measurement(void) override;
    public:
      inline void set_timing_collective(ValueBroadcast<long long>* collective)
      {
        timing_collective = collective;
      }
    protected:
      TimingMeasurement measurement;
      RtEvent measured;
      ValueBroadcast<long long>* timing_collective;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TIMING_OPERATION_H__
