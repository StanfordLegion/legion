/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// Useful for IDEs
#include "legion/legion_trace.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Collective Instance Creator
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename OP>
    CollectiveInstanceCreator<OP>::CollectiveInstanceCreator(Runtime *rt)
      : OP(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    CollectiveInstanceCreator<OP>::CollectiveInstanceCreator(
                                      const CollectiveInstanceCreator<OP> &rhs)
      : OP(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    CollectiveManager* 
        CollectiveInstanceCreator<OP>::find_or_create_collective_instance(
                                  MappingCallKind mapper_call, unsigned index,
                                  const LayoutConstraintSet &constraints,
                                  const std::vector<LogicalRegion> &regions,
                                  Memory::Kind kind, size_t *footprint,
                                  LayoutConstraintKind *unsat_kind,
                                  unsigned *unsat_index,
                                  DomainPoint &collective_point)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      bool found = false;
      CollectiveManager *manager = NULL;
      const CollectiveKey key(mapper_call, index);
      {
        AutoLock o_lock(this->op_lock);
        typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
          collective_instances.find(key);
        if (finder != collective_instances.end())
        {
          if (finder->second.remaining > 0)
          {
            found = true;
            manager = finder->second.manager;
            if ((manager == NULL) && (--finder->second.remaining == 0) &&
                (finder->second.pending == 0))
              collective_instances.erase(finder);
          }
          else
            wait_on = finder->second.ready_event;
        }
        else
        {
          CollectiveInstance &inst = collective_instances[key];
          inst.ready_event = Runtime::create_rt_user_event();
        }
      }
      if (!found)
      {
        if (!wait_on.exists())
        {
          IndexSpaceNode *collective_space = get_collective_space();
          const size_t total_points = collective_space->get_volume();
#ifdef DEBUG_LEGION
          assert(total_points > 0);
#endif
          // We need to make the collective instance and record it
          InstanceBuilder builder(regions, constraints, this->runtime);
          ApUserEvent instance_ready = Runtime::create_ap_user_event(NULL);
          manager = builder.create_collective_instance(this->runtime->forest,
              kind, collective_space, unsat_kind, unsat_index, instance_ready, 
              footprint);
          if (manager == NULL)
          {
            Runtime::trigger_event(NULL, instance_ready);
            instance_ready = ApUserEvent::NO_AP_USER_EVENT;
          }
          AutoLock o_lock(this->op_lock);
          typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
            collective_instances.find(key);
#ifdef DEBUG_LEGION
          assert(finder != collective_instances.end());
          assert(finder->second.manager == NULL);
          assert(finder->second.remaining == 0);
          assert(finder->second.ready_event.exists());
          assert(!finder->second.ready_event.has_triggered());
#endif
          finder->second.manager = manager;
          Runtime::trigger_event(finder->second.ready_event);
          finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
          finder->second.instance_event = instance_ready;
          finder->second.remaining = total_points;
          if ((manager == NULL) && (--finder->second.remaining == 0) &&
              (finder->second.pending == 0))
            collective_instances.erase(finder);
          return manager;
        }
        else
        {
          wait_on.wait();
          AutoLock o_lock(this->op_lock);
          typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
            collective_instances.find(key);
#ifdef DEBUG_LEGION
          assert(finder != collective_instances.end());
          assert(finder->second.remaining > 0);
#endif
          manager = finder->second.manager;
          if ((manager == NULL) && (--finder->second.remaining > 0) &&
              (finder->second.pending == 0))
            collective_instances.erase(finder);
        }
      }
      if (manager != NULL)
      {
        // Check that the regions and shapes match
        // Check that the constraints match
        if (!manager->meets_regions(regions, true/*tight*/) ||
            !manager->layout->match_layout(constraints, 
                                  manager->layout->total_dims))
        {
          // If we failed to match then set the manager back to NULL
          // to signal that layout constraints didn't match
          AutoLock o_lock(this->op_lock);
          typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
            collective_instances.find(key);
#ifdef DEBUG_LEGION
          assert(finder != collective_instances.end());
          assert(finder->second.remaining > 0);
#endif
          finder->second.manager = NULL; 
          // If we have other creators waiting to check to see if everyone
          // succeeded then we can wake them up now since we failed
          if (finder->second.ready_event.exists())
          {
            Runtime::trigger_event(finder->second.ready_event);
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
          }
          if (finder->second.instance_event.exists())
          {
            Runtime::trigger_event(NULL, finder->second.instance_event);
            finder->second.instance_event = ApUserEvent::NO_AP_USER_EVENT;
          }
          if ((--finder->second.remaining == 0) && 
              (finder->second.pending == 0))
            collective_instances.erase(finder);
          return NULL;
        }
      }
      return manager;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    bool CollectiveInstanceCreator<OP>::finalize_collective_instance(
                        MappingCallKind call_kind, unsigned index, bool success)
    //--------------------------------------------------------------------------
    {
      RtUserEvent wait_on;
      ApUserEvent instance_event;
      CollectiveManager *manager = NULL;
      const CollectiveKey key(call_kind, index);
      {
        AutoLock o_lock(this->op_lock);
        typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
          collective_instances.find(key);
#ifdef DEBUG_LEGION
        assert(finder != collective_instances.end());
        assert(finder->second.remaining > 0);
#endif
        if (finder->second.manager == NULL)
        {
          if ((--finder->second.remaining == 0) && 
              (finder->second.pending == 0))
            collective_instances.erase(finder);
          return false;
        }
        if (!success)
        {
          finder->second.manager = NULL;
          if (finder->second.ready_event.exists())
          {
            Runtime::trigger_event(finder->second.ready_event);
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
          }
#ifdef DEBUG_LEGION
          assert(finder->second.instance_event.exists());
#endif
          Runtime::trigger_event(NULL, finder->second.instance_event);
          finder->second.instance_event = ApUserEvent::NO_AP_USER_EVENT;
          if ((--finder->second.remaining == 0) && 
              (finder->second.pending == 0))
            collective_instances.erase(finder);
          return false;
        }
        if (finder->second.pending++ == 0)
          finder->second.ready_event = Runtime::create_rt_user_event();
        if (--finder->second.remaining == 0)
        {
          manager = finder->second.manager;
          instance_event = finder->second.instance_event;
        }
        else
          wait_on = finder->second.ready_event;
      }
      if (manager != NULL)
      {
        manager->finalize_collective_instance(instance_event);
        Runtime::trigger_event(wait_on);
      }
      else
        wait_on.wait();
      AutoLock o_lock(this->op_lock);
      typename std::map<CollectiveKey,CollectiveInstance>::iterator finder =
        collective_instances.find(key);
#ifdef DEBUG_LEGION
      assert(finder != collective_instances.end());
      assert(finder->second.pending > 0);
      assert(finder->second.remaining == 0);
#endif
      const bool result = (finder->second.manager != NULL);
      if (--finder->second.pending == 0)
        collective_instances.erase(finder);
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename OP>
    void CollectiveInstanceCreator<OP>::report_total_collective_instance_calls(
                                MappingCallKind call_kind, unsigned total_calls)
    //--------------------------------------------------------------------------
    {
      // This goes through and looks for unaligned collectives that haven't
      // been handled and need to fail
      AutoLock o_lock(this->op_lock);
      for (typename std::map<CollectiveKey,CollectiveInstance>::iterator it = 
           collective_instances.begin(); it != collective_instances.end(); it++)
      {
        if (it->first.first < call_kind)
          continue;
        else if (it->first.first > call_kind)
          break;
        if (it->first.second < total_calls)
          continue;
        it->second.manager = NULL;
        if (it->second.ready_event.exists())
        {
          Runtime::trigger_event(it->second.ready_event);
          it->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
        }
        if (it->second.instance_event.exists())
        {
          Runtime::trigger_event(NULL, it->second.instance_event);
          it->second.instance_event = ApUserEvent::NO_AP_USER_EVENT;
        }
      }
    }

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
          memo_state = MEMO_REPLAY;
          OP::trace->register_physical_only(this);
          trigger_replay();
          return;
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(OP::trace->is_recording());
          assert(tpl->is_recording());
#endif
          memo_state = MEMO_RECORD;
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
