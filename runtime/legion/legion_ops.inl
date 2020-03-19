/* Copyright 2020 Stanford University, NVIDIA Corporation
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
#ifdef DEBUG_LEGION
      assert(memo_state == NO_MEMO); // should always be no memo here
#endif
      PhysicalTrace *const physical_trace = (OP::trace == NULL) ? NULL :
          OP::trace->get_physical_trace();
      // Only invoke memoization if we are doing physical tracing
      if (physical_trace != NULL)
        invoke_memoize_operation(this->get_mappable()->map_id);
#ifdef DEBUG_LEGION
      assert(memo_state == NO_MEMO || memo_state == MEMO_REQ);
#endif
      if (memo_state == MEMO_REQ)
      {
        tpl = physical_trace->get_current_template();
        if (tpl == NULL)
        {
          OP::trace->set_state_record();
          tpl = physical_trace->start_new_template();
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
          this->resolve_speculation();
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
    PhysicalTemplate* MemoizableOp<OP>::get_template(void) const
    //--------------------------------------------------------------------------
    {
      return tpl;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    ApEvent MemoizableOp<OP>::compute_init_precondition(
                                                    const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      ApEvent sync_precondition = compute_sync_precondition(&trace_info);
      if (sync_precondition.exists())
      {
        if (this->execution_fence_event.exists())
          return Runtime::merge_events(&trace_info, sync_precondition,
                                       this->execution_fence_event);
        else
          return sync_precondition;
      }
      else
        return this->execution_fence_event;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    RtEvent MemoizableOp<OP>::complete_memoizable(RtEvent complete_event)
    //--------------------------------------------------------------------------
    {
      if (tpl != NULL)
        complete_event =
          Runtime::merge_events(complete_event, tpl->get_recording_done());
      return complete_event;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::find_equivalence_sets(Runtime *runtime, unsigned idx, 
                 const FieldMask &mask, FieldMaskSet<EquivalenceSet> &eqs) const
    //--------------------------------------------------------------------------
    {
      const VersionInfo &info = get_version_info(idx);
      const FieldMaskSet<EquivalenceSet> &sets = info.get_equivalence_sets();
      if (mask != sets.get_valid_mask())
      {
        for (FieldMaskSet<EquivalenceSet>::const_iterator it = 
              sets.begin(); it != sets.end(); it++)
        {
          const FieldMask overlap = it->second & mask;
          if (!overlap)
            continue;
          eqs.insert(it->first, overlap);
        }
      }
      else
        eqs = sets;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void MemoizableOp<OP>::invoke_memoize_operation(MapperID mapper_id)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(OP::trace != NULL);
      assert(!this->runtime->no_tracing);
      assert(!this->runtime->no_physical_tracing);
#endif
      Mapper::MemoizeInput  input;
      Mapper::MemoizeOutput output;
      input.trace_id = OP::trace->get_trace_id();
      output.memoize = false;
      Processor mapper_proc = OP::parent_ctx->get_executing_processor();
      MapperManager *mapper = OP::runtime->find_mapper(mapper_proc,mapper_id);
      Mappable *mappable = this->get_mappable();
#ifdef DEBUG_LEGION
      assert(mappable != NULL);
#endif
      mapper->invoke_memoize_operation(mappable, &input, &output);
      if (output.memoize)
        memo_state = MEMO_REQ;
    }

  }; // namespace Internal
}; // namespace Legion 
