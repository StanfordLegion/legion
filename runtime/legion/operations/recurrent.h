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

#ifndef __LEGION_RECURRENT_OPERATION_H__
#define __LEGION_RECURRENT_OPERATION_H__

#include "legion/operations/begin.h"
#include "legion/operations/complete.h"

namespace Legion {
  namespace Internal {

    /**
     * \class RecurrentOp
     * A recurrent op supports both the begin and complete interfaces
     */
    class RecurrentOp : public BeginOp,
                        public CompleteOp { };

    /**
     * \class TraceRecurrentOp
     * This is a tracing operation that is inserted to invalidate an idempotent
     * trace replay once an invalidating operation is detected in the stream
     * of operations in the parent context. We make this a mapping fence so
     * we ensure that the resources from the template are freed up before
     * any other downstream operations attempt to map.
     */
    class TraceRecurrentOp : public TraceOp,
                             public RecurrentOp {
    public:
      TraceRecurrentOp(void);
      TraceRecurrentOp(const TraceRecurrentOp& rhs) = delete;
      virtual ~TraceRecurrentOp(void);
    public:
      TraceRecurrentOp& operator=(const TraceRecurrentOp& rhs) = delete;
    public:
      void initialize_recurrent(
          InnerContext* ctx, LogicalTrace* trace, LogicalTrace* previous,
          Provenance* provenance, bool remove_reference);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    public:
      virtual FenceOp* get_begin_operation(void) override { return this; }
      virtual ApEvent get_begin_completion(void) override
      {
        return get_completion_event();
      }
      virtual PhysicalTemplate* create_fresh_template(
          PhysicalTrace* trace) override;
    public:
      virtual FenceOp* get_complete_operation(void) override { return this; }
    protected:
      LogicalTrace* previous;
      bool has_blocking_call;
      bool has_intermediate_fence;
      bool remove_trace_reference;
    };

    class ReplRecurrentOp : public ReplTraceOp,
                            public RecurrentOp {
    public:
      ReplRecurrentOp(void) : ReplTraceOp() { }
      virtual ~ReplRecurrentOp(void) { }
    };

    /**
     * \class ReplTraceRecurrentOp
     * Control replicated version of TraceRecurrentOp
     */
    class ReplTraceRecurrentOp
      : public ReplTraceBegin<ReplTraceComplete<ReplRecurrentOp> > {
    public:
      ReplTraceRecurrentOp(void);
      ReplTraceRecurrentOp(const ReplTraceRecurrentOp& rhs) = delete;
      virtual ~ReplTraceRecurrentOp(void);
    public:
      ReplTraceRecurrentOp& operator=(const ReplTraceRecurrentOp& rhs) = delete;
    public:
      void initialize_recurrent(
          ReplicateContext* ctx, LogicalTrace* trace, LogicalTrace* previous,
          Provenance* provenance, bool remove_reference);
      void perform_logging(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual void trigger_dependence_analysis(void) override;
      virtual void trigger_ready(void) override;
      virtual void trigger_mapping(void) override;
      virtual bool record_trace_hash(
          TraceHashRecorder& recorder, uint64_t idx) override;
    protected:
      LogicalTrace* previous;
      bool has_blocking_call;
      bool has_intermediate_fence;
      bool remove_trace_reference;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_RECURRENT_OPERATION_H__
