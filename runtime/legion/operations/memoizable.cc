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

#include "legion/operations/memoizable.h"
#include "legion/contexts/inner.h"
#include "legion/managers/mapper.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Memoizable Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MemoizableOp::MemoizableOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MemoizableOp::~MemoizableOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MemoizableOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      tpl = nullptr;
      memo_state = NO_MEMO;
    }

    //--------------------------------------------------------------------------
    void MemoizableOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      Operation::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    void MemoizableOp::set_memoizable_state(void)
    //--------------------------------------------------------------------------
    {
      // Can be called multiple times so handle that case
      if (memo_state != NO_MEMO)
        return;
      if ((trace != nullptr) && trace->has_physical_trace() &&
          !is_tracing_fence())
      {
        PhysicalTrace* physical = trace->get_physical_trace();
        tpl = physical->get_current_template();
        legion_assert(tpl != nullptr);
        if (physical->is_recording())
        {
          memo_state = MEMO_RECORD;
          // Check to see if the mapper is going to allow us to memoize
          // the result of this or not, if not inform the trace that
          // this recording needs to be invalidated
          if (!can_memoize_operation())
            tpl->record_no_consensus();
        }
        else
        {
          memo_state = MEMO_REPLAY;
          tpl->register_operation(this);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool MemoizableOp::can_memoize_operation(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(trace != nullptr);
      legion_assert(!runtime->no_tracing);
      legion_assert(!runtime->no_physical_tracing);
      Mappable* mappable = get_mappable();
      if (mappable != nullptr)
      {
        Mapper::MemoizeInput input;
        Mapper::MemoizeOutput output;
        input.trace_id = trace->get_trace_id();
        Processor mapper_proc = parent_ctx->get_executing_processor();
        MapperManager* mapper =
            runtime->find_mapper(mapper_proc, mappable->map_id);
        legion_assert(mappable != nullptr);
        mapper->invoke_memoize_operation(mappable, input, output);
        return output.memoize;
      }
      else
        return true;
    }

  }  // namespace Internal
}  // namespace Legion
