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

#ifndef __LEGION_MEMOIZABLE_H__
#define __LEGION_MEMOIZABLE_H__

#include "legion/api/sync_impl.h"
#include "legion/kernel/runtime.h"
#include "legion/operations/operation.h"
#include "legion/tools/spy.h"
#include "legion/tracing/template.h"
#include "legion/utilities/coordinates.h"

namespace Legion {
  namespace Internal {

    /**
     * \class MemoizableOp
     * A memoizable operation is an abstract class
     * that serves as the basis for operation whose
     * physical analysis can be memoized.  Memoizable
     * operations go through an extra step in the mapper
     * to determine whether to memoize their physical analysis.
     */
    class MemoizableOp : public Operation {
    public:
      enum MemoizableState {
        NO_MEMO,      // The operation is not subject to memoization
        MEMO_RECORD,  // The runtime is recording analysis for this operation
        MEMO_REPLAY,  // The runtime is replaying analysis for this opeartion
      };
    public:
      MemoizableOp(void);
      virtual ~MemoizableOp(void);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
    public:
      inline PhysicalTemplate* get_template(void) const { return tpl; }
      inline bool is_recording(void) const { return memo_state == MEMO_RECORD; }
      inline bool is_replaying(void) const { return memo_state == MEMO_REPLAY; }
      inline MemoizableState get_memoizable_state(void) const
      {
        return memo_state;
      }
    public:
      virtual void trigger_replay(void) = 0;
      virtual TraceLocalID get_trace_local_id(void) const
      {
        return TraceLocalID(trace_local_id, DomainPoint());
      }
      virtual ApEvent compute_sync_precondition(const TraceInfo& info) const
      {
        std::abort();
      }
      virtual void complete_replay(ApEvent complete) { std::abort(); }
      virtual ApEvent replay_mapping(void) { std::abort(); }
      virtual MemoizableOp* get_memoizable(void) override { return this; }
    protected:
      void set_memoizable_state(void);
      bool can_memoize_operation(void);
      template<typename OP, bool HAS_SYNCS>
      static ApEvent compute_sync_precondition_with_syncs(
          OP* op, const TraceInfo& trace_info);
    protected:
      // The physical trace for this operation if any
      PhysicalTemplate* tpl;
      // Track whether we are memoizing physical analysis for this operation
      MemoizableState memo_state;
    };

    /**
     * \class Memoizable
     * The memoizable class overrides certain pipeline stages to help
     * with making decisions about what to memoize
     */
    template<typename OP>
    class Memoizable : public OP {
    public:
      template<typename... Args>
      Memoizable(Args&&... args) : OP(std::forward<Args>(args)...)
      { }
      virtual ~Memoizable(void) { }
    public:
      virtual void trigger_ready(void) override;
      virtual ApEvent compute_sync_precondition(
          const TraceInfo& info) const override;
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/operations/memoizable.inl"

#endif  // __LEGION_MEMOIZABLE_H__
