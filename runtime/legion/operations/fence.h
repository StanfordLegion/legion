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

#ifndef __LEGION_FENCE_H__
#define __LEGION_FENCE_H__

#include "legion/operations/memoizable.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FenceOp
     * Fence operations give the application the ability to
     * enforce ordering guarantees between different tasks
     * in the same context which may become important when
     * certain updates to the region tree are desired to be
     * observed before a later operation either maps or
     * runs. All fences are mapping fences for correctness.
     * Fences all support the optional ability to be an
     * execution fence.
     */
    class FenceOp : public MemoizableOp {
    public:
      enum FenceKind {
        MAPPING_FENCE,
        EXECUTION_FENCE,
      };
    public:
      struct DeferTimingMeasurementArgs
        : public LgTaskArgs<DeferTimingMeasurementArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_TIMING_MEASUREMENT_TASK_ID;
      public:
        DeferTimingMeasurementArgs(void) = default;
        DeferTimingMeasurementArgs(FenceOp* o)
          : LgTaskArgs<DeferTimingMeasurementArgs>(false, false), op(o)
        { }
        void execute(void) const;
      public:
        FenceOp* op;
      };
    public:
      FenceOp(void);
      FenceOp(const FenceOp& rhs) = delete;
      virtual ~FenceOp(void);
    public:
      FenceOp& operator=(const FenceOp& rhs) = delete;
    public:
      Future initialize(
          InnerContext* ctx, FenceKind kind, bool need_future,
          Provenance* provenance);
      inline void add_mapping_applied_condition(RtEvent precondition)
      {
        map_applied_conditions.insert(precondition);
      }
      inline void record_execution_precondition(ApEvent precondition)
      {
        execution_preconditions.insert(precondition);
      }
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual bool invalidates_physical_trace_template(
          bool& exec_fence) const override
      {
        exec_fence = (fence_kind == EXECUTION_FENCE);
        return exec_fence;
      }
      FenceKind get_fence_kind(void) { return fence_kind; }
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_mapping(void) override;
      virtual void trigger_replay(void) override;
      virtual void complete_replay(ApEvent complete_event) override;
      virtual void trigger_complete(ApEvent complete) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
      virtual const VersionInfo& get_version_info(unsigned idx) const;
    public:
      virtual void perform_measurement(void);
    protected:
      FenceKind fence_kind;
      std::set<RtEvent> map_applied_conditions;
      std::set<ApEvent> execution_preconditions;
      Future result;
    };

    /**
     * \class ReplFenceOp
     * A fence operation that is aware that it is being
     * executed in a control replicated context. Currently
     * this only applies to mixed and execution fences.
     */
    class ReplFenceOp : public FenceOp {
    public:
      ReplFenceOp(void);
      ReplFenceOp(const ReplFenceOp& rhs) = delete;
      virtual ~ReplFenceOp(void);
    public:
      ReplFenceOp& operator=(const ReplFenceOp& rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual void trigger_dependence_analysis(void);
      virtual void trigger_mapping(void);
      virtual void trigger_replay(void);
      virtual void trigger_complete(ApEvent complete);
    protected:
      void initialize_fence_barriers(ReplicateContext* repl_ctx = nullptr);
    protected:
      RtBarrier mapping_fence_barrier;
      ApBarrier execution_fence_barrier;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FENCE_H__
