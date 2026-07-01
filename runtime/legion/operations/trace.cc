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

#include "legion/operations/trace.h"
#include "legion/contexts/inner.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // TraceOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TraceOp::TraceOp(void) : FenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TraceOp::~TraceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void TraceOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceOp::ReplTraceOp(void) : ReplFenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTraceOp::~ReplTraceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplTraceOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // Remote Trace Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTraceOp::RemoteTraceOp(Operation* ptr, AddressSpaceID src, OpKind k)
      : RemoteOp(ptr, src), kind(k)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    RemoteTraceOp::~RemoteTraceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID RemoteTraceOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteTraceOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTraceOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteTraceOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const char* RemoteTraceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[kind];
    }

    //--------------------------------------------------------------------------
    OpKind RemoteTraceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return kind;
    }

    //--------------------------------------------------------------------------
    void RemoteTraceOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
    }

    //--------------------------------------------------------------------------
    void RemoteTraceOp::unpack(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      // Nothing for the moment
    }

  }  // namespace Internal
}  // namespace Legion
