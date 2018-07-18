/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// Included from legion_ops.h - do not include this directly

#include "legion/legion_trace.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Memoizable Operation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    MemoizableOp<OP>::MemoizableOp(Runtime *rt)
      : OP(rt), Memoizable()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::initialize_memoizable(void)
    //--------------------------------------------------------------------------
    {
      tpl = NULL;
      memo_state = NO_MEMO;
      need_prepipeline_stage = false;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::pack_memoizable(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(memo_state);
      rez.serialize(need_prepipeline_stage);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::unpack_memoizable(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(memo_state);
      derez.deserialize(need_prepipeline_stage);
      // TODO: remote mapping is not yet supported in dynamic tracing
      tpl = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::activate_memoizable(void)
    //--------------------------------------------------------------------------
    {
      tpl = NULL;
      memo_state = NO_MEMO;
      need_prepipeline_stage = false;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      invoke_memoize_operation(OP::get_mappable()->map_id);
#ifdef DEBUG_LEGION
      assert(memo_state == NO_MEMO || memo_state == MEMO_REQ);
#endif
      if (memo_state == MEMO_REQ)
      {
#ifdef DEBUG_LEGION
        assert(OP::trace != NULL);
#endif
        PhysicalTrace *physical_trace = OP::trace->get_physical_trace();
        if (physical_trace == NULL)
        {
          REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
              "Invalid memoization request. An operation cannot be memoized "
              "when it is in a logical-only trace. Please change the mapper "
              "not to request memoization or allow memoization for the trace.");
        }
        tpl = physical_trace->get_current_template();
        if (tpl == NULL)
        {
          OP::trace->set_state_record();
          tpl = physical_trace->start_new_template(
              OP::parent_ctx->get_current_execution_fence_event());
          assert(tpl != NULL);
        }

        if (tpl->is_replaying())
        {
#ifdef DEBUG_LEGION
          assert(OP::trace->is_replaying());
          assert(tpl->is_replaying());
#endif
          memo_state = REPLAY;
          OP::trace->register_physical_only(this, OP::gen);
          OP::resolve_speculation();
          replay_analysis();
          return;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(OP::trace->is_recording());
          assert(tpl->is_recording());
#endif
          memo_state = RECORD;
        }
      }
      need_prepipeline_stage = true;
      OP::execute_dependence_analysis();
    };

    //--------------------------------------------------------------------------
    template<typename OP>
    TraceLocalID MemoizableOp<OP>::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(OP::trace_local_id, DomainPoint());
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::invoke_memoize_operation(MapperID mapper_id)
    //--------------------------------------------------------------------------
    {
      // If we're not in a trace or tracing isn't enabled then don't do anything
      if ((OP::trace != NULL) && !this->runtime->no_tracing &&
          !this->runtime->no_physical_tracing)
      {
        Mapper::MemoizeInput  input;
        Mapper::MemoizeOutput output;
        input.trace_id = OP::trace->get_trace_id();
        output.memoize = false;
        Processor mapper_proc = OP::parent_ctx->get_executing_processor();
        MapperManager *mapper = OP::runtime->find_mapper(mapper_proc,mapper_id);
        Mappable *mappable = OP::get_mappable();
#ifdef DEBUG_LEGION
        assert(mappable != NULL);
#endif
        mapper->invoke_memoize_operation(mappable, &input, &output);
        if (OP::trace == NULL && output.memoize)
          REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
              "Invalid mapper output from 'memoize_operation'. Mapper requested"
              " memoization of an operation that is not being traced.");
        set_memoize(output.memoize);
      }
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::set_memoize(bool memoize)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo_state == NO_MEMO);
#endif
      if (memoize && !this->runtime->no_tracing && 
          !this->runtime->no_physical_tracing)
        memo_state = MEMO_REQ;
    }

  }; // namespace Internal
}; // namespace Legion 
