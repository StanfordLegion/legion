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

#include "legion/operations/frame.h"
#include "legion/contexts/inner.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Frame Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FrameOp::FrameOp(void) : FenceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    FrameOp::~FrameOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void FrameOp::initialize(InnerContext* ctx, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      FenceOp::initialize(
          ctx, EXECUTION_FENCE, false /*need future*/, provenance);
    }

    //--------------------------------------------------------------------------
    void FrameOp::activate(void)
    //--------------------------------------------------------------------------
    {
      FenceOp::activate();
    }

    //--------------------------------------------------------------------------
    void FrameOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      FenceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* FrameOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[FRAME_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind FrameOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return FRAME_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void FrameOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Increment the number of mapped frames
      parent_ctx->increment_frame();
      FenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void FrameOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // This frame has finished executing so it is no longer mapped
      parent_ctx->decrement_frame();
      // This frame is also finished so we can tell the context
      parent_ctx->finish_frame(this);
      FenceOp::trigger_commit();
    }

  }  // namespace Internal
}  // namespace Legion
