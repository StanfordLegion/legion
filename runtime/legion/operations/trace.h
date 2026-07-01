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

#ifndef __LEGION_TRACE_OPERATION_H__
#define __LEGION_TRACE_OPERATION_H__

#include "legion/operations/fence.h"
#include "legion/operations/remote.h"

namespace Legion {
  namespace Internal {

    /**
     * \class TraceOp
     * This class serves as the base class for all tracing related operations
     */
    class TraceOp : public FenceOp {
    public:
      TraceOp(void);
      TraceOp(const TraceOp& rhs) = delete;
      virtual ~TraceOp(void);
    public:
      TraceOp& operator=(const TraceOp& rhs) = delete;
    public:
      virtual bool is_tracing_fence(void) const override { return true; }
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    };

    /**
     * \class ReplTraceOp
     * Base class for all replicated trace operations
     */
    class ReplTraceOp : public ReplFenceOp {
    public:
      ReplTraceOp(void);
      ReplTraceOp(const ReplTraceOp& rhs) = delete;
      virtual ~ReplTraceOp(void);
    public:
      ReplTraceOp& operator=(const ReplTraceOp& rhs) = delete;
    public:
      virtual bool is_tracing_fence(void) const override { return true; }
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
    protected:
      struct TemplateStatus {
        bool all_valid;
        bool any_not_acquired;
      };
      struct StatusReduction {
        typedef TemplateStatus RHS;
        template<bool EXCLUSIVE>
        static inline void fold(RHS& rhs1, RHS rhs2)
        {
          if (!rhs2.all_valid)
            rhs1.all_valid = false;
          if (rhs2.any_not_acquired)
            rhs1.any_not_acquired = true;
        }
      };
    };

    /**
     * \class RemoteTraceOp
     * This is a remote copy of a trace op, it really doesn't
     * have to do very much at all other than implement the interface
     * for remote ops as it will only be used for updating state for
     * physical template replays
     */
    class RemoteTraceOp : public RemoteOp,
                          public Heapify<RemoteTraceOp, OPERATION_LIFETIME> {
    public:
      RemoteTraceOp(Operation* ptr, AddressSpaceID src, OpKind k);
      RemoteTraceOp(const RemoteTraceOp& rhs) = delete;
      virtual ~RemoteTraceOp(void);
    public:
      RemoteTraceOp& operator=(const RemoteTraceOp& rhs) = delete;
    public:
      virtual UniqueID get_unique_id(void) const;
      virtual uint64_t get_context_index(void) const;
      virtual void set_context_index(uint64_t index);
      virtual int get_depth(void) const;
    public:
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
      virtual void pack_remote_operation(
          Serializer& rez, AddressSpaceID target,
          std::set<RtEvent>& applied) const override;
      virtual void unpack(Deserializer& derez) override;
    public:
      const OpKind kind;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_TRACE_OPERATION_H__
