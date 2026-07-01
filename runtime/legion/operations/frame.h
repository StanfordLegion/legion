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

#ifndef __LEGION_FRAME_H__
#define __LEGION_FRAME_H__

#include "legion/operations/fence.h"

namespace Legion {
  namespace Internal {

    /**
     * \class FrameOp
     * Frame operations provide a mechanism for grouping
     * operations within the same context into frames. Frames
     * provide an application directed way of controlling the
     * number of outstanding operations in flight in a context
     * at any given time through the mapper interface.
     */
    class FrameOp : public FenceOp {
    public:
      FrameOp(void);
      FrameOp(const FrameOp& rhs) = delete;
      virtual ~FrameOp(void);
    public:
      FrameOp& operator=(const FrameOp& rhs) = delete;
    public:
      void initialize(InnerContext* ctx, Provenance* provenance);
    public:
      virtual void activate(void) override;
      virtual void deactivate(bool free = true) override;
      virtual const char* get_logging_name(void) const override;
      virtual OpKind get_operation_kind(void) const override;
    public:
      virtual void trigger_mapping(void) override;
      virtual void trigger_commit(void) override;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FRAME_H__
