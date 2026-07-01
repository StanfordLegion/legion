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

#ifndef __LEGION_BEGIN_OPERATION_H__
#define __LEGION_BEGIN_OPERATION_H__

#include "legion/operations/trace.h"

namespace Legion {
  namespace Internal {

    /**
     * \class BeginOp
     * This is a pure virtual interface for operations that need to be
     * performed across any kind of begin operation for tracing
     */
    class BeginOp {
    public:
      virtual bool allreduce_template_status(bool& valid, bool acquired)
      {
        if (acquired)
          return false;
        valid = false;
        return true;
      }
      virtual ApEvent get_begin_completion(void) = 0;
      virtual FenceOp* get_begin_operation(void) = 0;
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace* trace) = 0;
    };

    /**
     * \class TraceBeginOp
     * This class represents mapping fences which we inject
     * into the operation stream to begin a trace.  This fence
     * is by a TraceReplayOp if the trace allows physical tracing.
     */
    class TraceBeginOp : public TraceOp,
                         public BeginOp {
    public:
      TraceBeginOp(void);
      TraceBeginOp(const TraceBeginOp& rhs) = delete;
      virtual ~TraceBeginOp(void);
    public:
      TraceBeginOp& operator=(const TraceBeginOp& rhs) = delete;
    public:
      void initialize_begin(
          InnerContext* ctx, LogicalTrace* trace, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    public:
      virtual ApEvent get_begin_completion(void) override
      {
        return get_completion_event();
      }
      virtual FenceOp* get_begin_operation(void) override { return this; }
      virtual PhysicalTemplate* create_fresh_template(
          PhysicalTrace* trace) override;
    };

    class ReplBeginOp : public ReplTraceOp,
                        public BeginOp {
    public:
      ReplBeginOp(void) : ReplTraceOp() { }
      virtual ~ReplBeginOp(void) { }
    };

    // Mixin class for adding support for replicated complete interfaces
    template<typename OP>
    class ReplTraceBegin : public OP {
    public:
      ReplTraceBegin(void);
      virtual ~ReplTraceBegin(void) { }
    protected:
      void initialize_begin(ReplicateContext* ctx, LogicalTrace* trace);
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual ApEvent get_begin_completion(void)
      {
        return this->get_completion_event();
      }
      virtual FenceOp* get_begin_operation(void) { return this; }
      virtual PhysicalTemplate* create_fresh_template(PhysicalTrace* trace);
      virtual bool allreduce_template_status(bool& valid, bool acquired);
    protected:
      void perform_template_creation_barrier(void);
    private:
      // If we do back-to-back executions of different traces
      // then we fuse the invalidation of the previous trace into the
      // begin operation of the next trace
      std::vector<CollectiveID> status_collective_ids;
      SlowBarrier* slow_barrier;
      CollectiveID slow_barrier_id;
    };

    /**
     * \class ReplTraceBeginOp
     * Control replicated version of trace begin op
     */
    class ReplTraceBeginOp : public ReplTraceBegin<ReplBeginOp> {
    public:
      ReplTraceBeginOp(void);
      ReplTraceBeginOp(const ReplTraceBeginOp& rhs) = delete;
      virtual ~ReplTraceBeginOp(void);
    public:
      ReplTraceBeginOp& operator=(const ReplTraceBeginOp& rhs) = delete;
    public:
      void initialize_begin(
          ReplicateContext* ctx, LogicalTrace* trace, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_BEGIN_OPERATION_H__
